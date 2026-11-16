.. _site-configuration:

Site Configuration
==================

FrequenSolve runs jobs through a configured execution :term:`site`. A site can
target FrequenSol Cloud, a local solver installation, or an :term:`HPC`
cluster. Most user scripts should call ``fs.Site()`` and let the
:term:`site configuration file` choose the active backend.

Persistent SLURM sessions
-------------------------

For small iterative workloads, reserve one batch allocation and submit all
operations to its resident scheduler. Entering the context waits for the
controller to become ready. It remains active while the queue is empty::

   from frequensolve.orchestrator.sites import Site, AdaptiveWorkers

   site = Site(profile="stampede3")
   with site.session(
       nodes=1,
       ranks_per_node=8,
       duration="02:00:00",
       workers=AdaptiveWorkers(min_ranks=1, max_ranks_per_task=2),
   ) as session:
       runs = [session.submit(job) for job in jobs]
       results = session.wait_all(runs, check=True)

A session is itself a normal Slurm batch job, with a FrequenSolve controller
as its resident process. The controller starts solver steps inside the actual
allocation environment, accepts further runs and keeps their status on the
shared filesystem. ``attach_session`` reconnects a Python client to that
controller; it does not SSH into a compute node or reconstruct Slurm variables.
An arbitrary batch or interactive allocation has no such controller and cannot
be borrowed through this API. For a single independent simulation, the usual
``site.run(job)`` batch workflow remains available.

The resource values above are illustrative. Allocation resources are fixed;
``AdaptiveWorkers`` bounds the ranks of individual frequency tasks. Tasks use
memory estimates when needed and retain MPI rank counts required by saved
partitioned artifacts. Initialization, smoothing, packing and curvature also
reserve resources through the same scheduler. Stampede3 uses ``ibrun`` offsets
and ``task_affinity``; generic ``srun`` sites use exclusive job steps. Generic
``mpirun``/``mpiexec`` launchers execute one process group at a time. On a GPU
partition the allocation exports the partition's ``environment``, sizes
workers by device memory, and each ``ibrun`` step's ranks select the device
under their slot, as in batch sweeps.

``site.provision(nodes, tasks, duration)`` and
``site.attach_allocation(job_id)`` are deprecated. ``provision`` now starts
a persistent session with ``tasks`` ranks spread evenly over ``nodes``;
``site.run``/``site.submit`` in ``"auto"`` or ``"attached"`` mode and attached
curvature use it until ``site.deprovision()`` or ``site.close()``.
``attach_allocation`` reconnects only to an allocation running a FrequenSolve
session. Earlier versions launched attached work over SSH on the allocation's
batch host, where the shell lacks the job's Slurm and module environment, so
launchers such as TACC's ``ibrun`` could not start there. Prefer
``site.session(...)`` directly.

``session.submit`` returns the usual ``RunHandle``. Its ``id`` identifies this
run; ``handle.backend["allocation_id"]`` identifies the enclosing Slurm job.
``handle.cancel()`` cancels the run without releasing the allocation. Explicit
sessions fail if the controller dies or a request cannot fit; they never
silently submit a replacement batch job. Solver processes start afresh for
individual operations; the session does not retain live solver factorizations.

The controller accepts atomically published requests on the site's shared work
filesystem through the existing SSH/file-transfer connection. ``request_id``
may be supplied to ``session.submit`` as 32 lowercase hexadecimal characters;
republication with the same id and payload does not execute the request twice.
An id cannot be reused for different work. Durable session records remain under
``<work_dir>/.fs_sessions/<session_id>`` after the allocation ends.

The SLURM ``work_dir`` is a persistent project directory. Before uploading inputs,
FrequenSolve records its originating local project directory and rejects uploads
from a different project. Repeated runs and updated inputs from the same project
continue to use the existing layout; no new project directories are created.
Use a separate ``work_dir`` for each local project. Older remote directories are
adopted only when their saved project metadata identifies the same origin;
ambiguous ownership or a moved local project produces an error before upload.

Creating the session directory and publishing each lease renewal, request and
command are login-node filesystem commands (``mkdir``, ``mv`` and ``ln``).
Each may take up to ``session_command_timeout`` seconds (default 120), set in
the profile's ``run_config`` or per session with
``site.session(session_command_timeout=...)`` or
``PersistentAllocation(session_command_timeout=...)``. A timeout or failure
names the command and remote path. The 15-second bound on solver identity
queries does not apply to these commands.

To reconnect to a running session, use its ``session.session_id``::

   with site.attach_session(session_id) as borrowed:
       run = borrowed.handle(job, job_id=run_id)
       result = run.wait(check=True)

A borrowed context only detaches on exit. The context that created the
allocation owns its lifetime. On normal exit it drains work: runs already in
flight finish, including their packing and output writes, and the controller
then exits, ending the batch job; ``scancel`` is used only if the job is still
listed ten seconds after the controller reported that it closed.
``cleanup_timeout`` caps the drain; the default ``None`` waits as long as the
controller heartbeat keeps advancing (the walltime still bounds it). If the
cap is reached, or the heartbeat stops advancing during the drain, exit emits
a ``RuntimeWarning`` naming the session and Slurm allocation and leaves the
allocation running rather than killing work mid-write: the controller cancels
unfinished work once the lease lapses, and Slurm ends the job at walltime.
Reattach with ``site.attach_session(session_id)`` or release it with
``scancel``. On an exception or interruption the context cancels outstanding
work, waits up to ``cleanup_timeout`` (120 seconds when unset) for the
controller, and cancels the allocation; cleanup errors there are logged and
never replace the original exception. Cleanup errors after the body has
completed (for example a lost SSH connection) are reported as a
``RuntimeWarning`` with the session and allocation ids instead of being
raised, so results computed in the context are not lost. Cleanup does not
close caller-owned site connections or delete solver artifacts.

If the client cannot establish who else is attached, cleanup leaves the
allocation running and warns instead of assuming it is unused. A failed drain
also preserves the remaining lease grace period for reconnection. After a
transport failure, ``session.release()`` can be retried; this explicitly ends
the allocation, including other clients' work. Ordinary owner cleanup checks
for other clients again on the controller before ending a shared allocation.
If that response is lost, automatic cleanup preserves the lease grace period;
it does not escalate to ``scancel`` and risk interrupting a new borrower.

The controller writes a heartbeat with an increasing ``sequence`` from a
dedicated thread, so per-run status writes on a slow work filesystem cannot
delay it. The client declares the controller lost only when two successful
reads, a poll interval apart, show a heartbeat unchanged for longer than
``heartbeat_timeout`` on the client's monotonic clock while Slurm still
reports the allocation running; the runs then fail. Timestamps written on the
compute node are never compared with the client's clock, so clock skew does
not matter, and failed status reads are not counted as evidence of loss. The
default ``heartbeat_timeout="auto"`` uses the site's
``run_config.scheduler_heartbeat_timeout`` (default 300 seconds) as configured;
pass seconds to choose a window for one session or ``None`` to disable the
check. Shared scratch filesystems can stall metadata operations for minutes
during large writes, so a session whose window is below 120 seconds logs a
warning when it starts. ``PersistentAllocation`` accepts the same ``heartbeat_timeout`` and
``cleanup_timeout`` options. The controller writes its output, the Python
traceback of an unhandled error, and, on a fatal signal such as a
segmentation fault, every thread's traceback to
``<work_dir>/.fs_sessions/<session_id>/controller.log``. A lost heartbeat, a
controller that reported ``failed``, or a job Slurm ended names that file
(``session.controller_log``) in the reported status. Processed requests and commands move to
``processed/`` subdirectories of the session directory, so the controller
lists only new work.

Clients renew leases from a background thread while connected, including
during long uploads and project synchronization. The controller times each
renewal with its own monotonic clock, so client clock skew does not expire a
lease. If every client disappears, the controller cancels unfinished work and
exits after ``lease_timeout`` (default 300 seconds) without a renewal, unless
the session lingers (see below); the hard Slurm walltime still bounds the
allocation. A warning signal 120 seconds before walltime stops admission and
marks unfinished runs timed out. Resume an inversion from its last accepted
checkpoint in a new session. Automatic allocation rollover is not supported.

Reusing a session across scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To let a later script reuse an allocation instead of waiting in the queue
again, start sessions with a reuse scope and, optionally, a linger period::

   with site.session(
       nodes=2,
       ranks_per_node=48,
       duration="04:00:00",
       reuse=project,  # or a string naming the scope
       linger="30m",  # stay up this long once nothing is attached or running
   ) as session:
       session.submit(job)

With ``reuse``, entering the context first looks for a session that this user
started with the same scope, and adopts it when all of these hold:

- its Slurm job is this user's, named ``FS_session``, queued or running, and
  recorded by a session directory under the site's work directory, so no
  other job ever qualifies;
- it has the same nodes, ranks per node, queue, account, solver executable,
  launcher, worker limits, GPU layout and module and environment setup;
- at least ``min_time_left`` of walltime remains (default 15 minutes);
- no other client is attached; ``share=True`` also joins a session that
  another client is using.

Otherwise the context queues a new allocation as before and records the
scope; each skipped session is reported with the reason. ``session.reused``
tells which happened. A project scope combines the project's name with its
resolved path, so unrelated projects that share a name never share
allocations; a string scope can be shared deliberately across projects or
machines. ``site.attach_session(reuse=project)`` adopts a matching session but
raises instead of queueing one. A client that exited on the same host is
recognized at once; otherwise its lease must lapse (``lease_timeout``) before
another client may adopt the session.

``linger`` (seconds, or a duration such as ``"30m"``, ``"2h"`` or
``"01:30:00"``; default 0) keeps the allocation once no client is attached and
nothing runs. A context with ``linger`` therefore only detaches on exit,
after waiting for its own runs (normal exit) or cancelling them (exception),
and reports how long the allocation stays up. Interrupting a lingering
session while it waits in the queue leaves it queued, so the next script
keeps its place. If a client disappears without detaching, its runs finish;
resubmitting such a job after adopting the session waits on the existing run
instead of starting it again. The longest ``linger`` among the clients
attached last applies. Without ``linger`` a session ends with its owner, as
before, unless other clients are still attached.

End a lingering allocation with ``session.release()`` (also after the context
has exited), ``site.release_sessions(project)`` (the scope's idle sessions;
``include_attached=True`` also ends ones in use and cancels their work) or
``scancel``. ``site.sessions(reuse=...)`` lists queued and running sessions
with their state, attached clients, idle time, linger and remaining walltime.
``PersistentAllocation`` accepts the same ``reuse``, ``linger``, ``share`` and
``min_time_left`` options.

Quick Start
-----------

The default :term:`site configuration file` is ``~/.frequensolve/site.toml``. On first
use, if that file does not exist, ``fs.Site()`` creates a starter file and
raises an exception asking you to review it. Edit the file, then rerun the same
script or notebook. The starter keeps FrequenSol Cloud active by default and
includes editable local, generic SLURM, and Stampede3 profiles with placeholder
values. These profiles are ordinary TOML tables, but remain inactive until
selected by ``default`` or ``fs.Site(profile=...)``.

Cloud is the quickest path for most new users because the solver installation
is managed remotely:

.. code-block:: toml

   default = "cloud"

   [sites.cloud]
   type = "aws"
   domain = "app.frequensol.com"
   interactive = true
   verbose = true

Use the active site in Python with:

.. code-block:: python

   import frequensolve as fs

   site = fs.Site()
   print(type(site).__name__)

Once a site is configured, the same object submits saved or newly authored
``TimeDomainJob`` and ``FrequencyDomainJob`` objects.

For local or :term:`HPC` execution, install the relevant
:term:`optional extras <optional extra>` and configure a matching profile. See
:doc:`frequensolve_directory` for where the file is stored and how to move
user-local FrequenSolve storage.

Choosing a Site Type
--------------------

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - ``type`` value
     - Install extra
     - Requirement
   * - ``aws``
     - ``frequensolve[cloud]``
     - FrequenSol Cloud account, license, and network access.
   * - ``local``
     - ``frequensolve[parallel]``
     - Local :term:`fast solver` installation configured in the profile.
   * - ``slurm``
     - ``frequensolve[hpc]``
     - Generic :term:`SSH`/:term:`SLURM` access, allocation details, and a
       solver installation on the cluster.

``stampede3`` and ``tacc`` remain accepted as compatibility aliases for a
generic ``slurm`` site using the built-in ``stampede3`` preset.

Selecting Profiles
------------------

Most users should keep multiple named profiles:

.. code-block:: toml

   default = "cloud"

   [sites.cloud]
   type = "aws"
   domain = "app.frequensol.com"
   interactive = true
   verbose = true

   [sites.local]
   type = "local"
   solver = "/opt/frequensol/bin/fs3d_s"
   shutdown_on_completion = true

   [sites.hpc]
   type = "slurm"
   preset = "stampede3"
   work_dir = "/shared/username/frequensolve"
   default_partition = "skx-dev"
   nodes = 1
   duration = "00:30:00"

The profile name and backend type are separate. In the example above,
``hpc`` is the profile selected with ``fs.Site(profile="hpc")``, while the
``stampede3`` preset supplies the standard host, launcher, partition limits,
and node shapes for the generic SLURM backend.

Select a profile in Python with:

.. code-block:: python

   site = fs.Site()                 # uses default
   local = fs.Site(profile="local")
   cluster = fs.Site(profile="hpc")

Profile values can be overridden at construction time without restating the
rest of the profile. For example, this keeps the local profile's configured
``solver`` path while using one Dask worker for a memory-heavy run:

.. code-block:: python

   local = fs.Site(profile="local", n_workers=1, threads_per_worker=16)

Cloud resource shapes are accepted only from a named ``site.toml`` profile.
They cannot be overridden on ``fs.Site(...)`` or ``submit(...)``. A Cloud
``cluster_id`` can also be selected for an individual submission as below.

Set ``FREQUENSOLVE_SITE_CONFIG`` or pass ``fs.Site(config_path=...)`` when a
test, notebook, or shared workstation should use a different config file:

.. code-block:: python

   shared = fs.Site(config_path="/path/to/site.toml", profile="hpc")

Direct constructors such as ``fs.LocalSite(...)`` and ``fs.AWSSite(...)``
remain available when code intentionally targets one backend.
``fs.Stampede3Site(...)`` remains as a compatibility adapter for existing code
and persisted run records.

Cloud cluster selection
-----------------------

Choose a default cluster on the Cloud application's **Compute** page, or copy
its SDK configuration to select it explicitly. ``fs.Site(profile="cloud")``
selects your saved connection settings; ``cluster_id`` selects the cluster.

.. code-block:: toml

   [sites.cloud]
   type = "aws"
   domain = "app.frequensol.com"
   cluster_id = "research-cluster"

Override that selection for one run:

.. code-block:: python

   site = fs.Site(profile="cloud")
   run = site.submit(job, cluster_id="frequensol-shared-us-east-1")

Selection uses the submission override, then the configured ``cluster_id``,
then your personal Cloud default, then FrequenSol shared compute when no default
is saved. Missing, unavailable or unauthorized clusters fail without falling back.

Cluster resolution happens before local output reuse. The SDK checks the
previous run's registered cluster identity, and Cloud revalidates that identity
before submission. If it changed during preparation, retry after reviewing the
cluster in Compute. Changing your default does not redirect existing runs;
logs, results, and cancellation continue to use their simulation identifier.

Partition selection and compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both ``partition`` and ``allow_unverified_compute`` can be configured in the
site file or overridden in ``submit``. Each submit argument overrides its
configured value independently, including an explicit ``False``.

.. code-block:: toml

   default = "research"

   [sites.research]
   type = "aws"
   domain = "app.frequensol.com"
   cluster_id = "research-cluster"
   partition = "lab-mpi"
   allow_unverified_compute = true

.. code-block:: python

   site = fs.Site(profile="research")
   run = site.submit(job, partition="verified-cpu", allow_unverified_compute=False)

Without an explicit partition, Cloud selects a compatible registered partition
by priority, then identifier. The Compute page exposes execution modes, node
hardware, MPI/network information, CPU-sharing support, and resource ceilings.
``cluster_id`` is a stable cluster identifier, separate from its display name
and your ``fs.Site`` connection name. The shared cluster uses
``frequensol-shared-us-east-1``; other clusters have their own identifiers.

Compatibility checks are best effort. ``allow_unverified_compute=True`` requires
an explicit partition and can override missing metadata or incorrect findings.
Any user with submission access to a customer-owned cluster may use it without
additional approval. Platform-owned shared clusters permit this override only
when their catalog policy explicitly allows it. Authorization, billing,
enforced resource limits, and executable protocol requirements still apply.
Usage is billed at standard rates even if compatibility issues cause failure.

Each accepted run freezes its selected partition, catalog revision, override
choice, and compatibility findings. These are available through
``run.execution.partition``, ``run.execution.partition_catalog_revision``,
``run.execution.allow_unverified_compute``, and
``run.execution.compatibility_findings``. Later catalog changes do not redirect
accepted runs or make a previous placement eligible for local output reuse.

.. _site-config-spec:

Site Configuration File Spec
----------------------------

``fs.Site()`` reads the active :term:`site configuration file` and returns the
configured execution backend. The default path is ``~/.frequensolve/site.toml``
unless :term:`FREQUENSOLVE_HOME`, :term:`FREQUENSOLVE_SITE_CONFIG`, or
``fs.Site(config_path=...)`` changes it.

Supported Shape
~~~~~~~~~~~~~~~

Every site configuration file must define a default profile and one or more
named profiles:

.. code-block:: toml

   default = "local"

   [sites.local]
   type = "local"
   solver = "/opt/frequensol/bin/fs3d_s"
   shutdown_on_completion = true

Top-Level Keys
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Meaning
   * - ``default``
     - Name of the profile under ``[sites.<name>]`` used by ``fs.Site()``.
   * - ``[sites.<name>]``
     - Named profile table. Each profile must set ``type``.

Site Table Keys
~~~~~~~~~~~~~~~

Every ``[sites.<name>]`` table supports the following structural keys:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Meaning
   * - ``type``
     - Backend selector. Supported values include ``aws``, ``local``,
       ``slurm``, ``stampede3``, and ``tacc``.
   * - ``kwargs``
     - Optional nested table merged into the selected profile. Prefer flat keys
       unless you need to pass a backend-specific argument without mixing it
       with config fields.
   * - ``preset``
     - Optional built-in site preset overlaid by the profile. Currently
       ``stampede3`` is provided for generic ``slurm`` sites.

For named profiles, the profile identifier comes from the table header, such as
``[sites.cloud]``.

Cloud Profiles
~~~~~~~~~~~~~~

Cloud profiles create ``AWSSite`` instances and require the ``cloud`` extra.

.. code-block:: toml

   [sites.cloud]
   type = "aws"
   domain = "app.frequensol.com"
   interactive = true
   verbose = true

Cloud execution uses the registered shared cluster. Select the resources
in the profile, then submit jobs without resource overrides:

.. code-block:: toml

   [sites.cloud-slurm]
   type = "aws"
   domain = "app.frequensol.com"
   cluster_id = "frequensol-shared-us-east-1"

   [sites.cloud-slurm.execution_resources]
   nodes = 1
   mpi_ranks = 1
   wall_time_seconds = 1800

.. code-block:: python

   site = fs.Site(profile="cloud-slurm")
   run = site.submit(job)

Omitting the execution settings selects your default Cloud cluster with one node, one MPI
rank, and a one-hour wall time. The environment must have an available registered
site before submission. The SDK never creates a cluster.

``type = "aws"`` uses FrequenSol Cloud authentication, storage, and billing.
``type = "slurm"`` uses direct SSH access to a customer-managed cluster.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Meaning
   * - ``domain``
     - App domain used to fetch public Cloud configuration. Defaults to
       ``FREQUENSOL_DOMAIN`` when omitted.
   * - ``interactive``
     - Prompt for missing login credentials. Defaults to false.
   * - ``verbose``
     - Print user-facing status messages in addition to logging.
   * - ``email`` / ``password``
     - Accepted for non-interactive login. Prefer cached login state or a secrets
       manager instead of storing credentials in ``site.toml``.
   * - ``cluster_id``
     - An authorized Cloud cluster identifier. Omit to follow your personal
       default cluster, or FrequenSol shared compute when none is saved.
   * - ``execution_resources``
     - ``nodes``, ``mpi_ranks``, and ``wall_time_seconds`` are required together.
       The managed site supports one node with one MPI rank, or two nodes with
       1-4 ranks per node, and a wall time of 60-7200 seconds. Optional ``cpu``
       and ``memory_mib`` constrain the allocation.

Batch profiles and the old ``execution_backend``, ``compute_mode``, and
``slurm_*`` Cloud settings are rejected. The service owns partition and cluster
configuration.

Local Profiles
~~~~~~~~~~~~~~

Local profiles create ``LocalSite`` instances and require the ``parallel``
extra.

.. code-block:: toml

   [sites.local]
   type = "local"
   solver = "/opt/frequensol/bin/fs3d_s"
   shutdown_on_completion = true
   n_workers = 2
   dashboard_port = 8787

   [sites.local.environment]

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Meaning
   * - ``shutdown_on_completion``
     - Close local :term:`Dask` resources after a run completes.
   * - ``solver``
     - Path to the installed local solver executable.
   * - ``n_workers``
     - Number of local Dask workers. If omitted, FrequenSolve chooses from
       system resources.
   * - ``threads_per_worker``
     - Threads per local Dask worker.
   * - ``memory_per_worker``
     - Memory per local worker in megabytes.
   * - ``environment``
     - Non-secret environment variables added to worker and solver processes.
       Known credential variables are removed from inherited subprocess
       environments and rejected here.
   * - ``dashboard_host``
     - Hostname for the Dask dashboard. Defaults to ``localhost``.
   * - ``dashboard_port``
     - Dashboard port. ``0`` lets Dask choose an available port.
   * - ``curvature_ranks``
     - MPI ranks for row-distributed Sauce curvature operations; they share
       the worker thread budget. Defaults to ``1``. Mesh transfer and sampling
       always run on one rank.
   * - ``background_layout``
     - Files of imaging background checkpoints: ``rank`` (default) or
       ``shared`` (see `Partitioned Solver Inputs`_).
   * - ``verbose``
     - Print user-facing status messages in addition to logging.

With the ``parallel`` extra installed, default local worker sizing uses
Dask's available CPU and memory limits, including process affinity and supported
container limits. Physical host cores remain an upper bound. Explicit worker,
thread and memory settings are retained, and configurations exceeding the
available budget are rejected before workers start. Container detection follows
the installed Dask/distributed version; unusual scheduler or cgroup layouts may
still require explicit worker settings.

A task submitted with ``procs_per_job`` greater than one runs under
``mpirun`` with ``-nthreads`` set to its share of the worker thread budget.
Open MPI would bind each rank, and all of its threads, to a single core, and
concurrent tasks to the same cores, so local multi-rank launches set
``PRTE_MCA_hwloc_default_binding_policy`` and
``OMPI_MCA_hwloc_base_binding_policy`` to ``none`` unless the environment
already sets them. Tasks that read a linearization's saved inputs run on the
rank count that wrote them (see `Partitioned Solver Inputs`_).

Local Host Settings
~~~~~~~~~~~~~~~~~~~

The optional top-level ``[host]`` table describes the machine running the
FrequenSolve Python process. It is not part of any remote execution profile.

.. code-block:: toml

   [host]
   tmp_dir = "/tmp/frequensolve"

``tmp_dir`` controls local disposable staging for project transfer bundles and
SFTP tarballs. When omitted, FrequenSolve uses Python's platform temp directory
(``tempfile.gettempdir()``), typically ``/tmp`` on Linux, a per-user
``/var/folders/...`` path on macOS, and ``%TEMP%`` or ``%TMP%`` on Windows.

HPC Profiles
~~~~~~~~~~~~

Generic :term:`SLURM` profiles create ``SlurmSite`` instances and require the ``hpc``
extra.

.. code-block:: toml

   [sites.cluster]
   type = "slurm"
   hostname = "login.example.edu"
   username = "username"
   credential = "example-primary"
   ssh_key = "~/.ssh/id_ed25519"
   solver = "/work/shared/frequensol/FS_seismic"
   work_dir = "/shared/username/frequensolve"
   scratch_dir = "/scratch/username/frequensolve"
   tmp_dir = "/scratch/username/frequensolve/tmp"
   default_partition = "debug"
   account = "allocation"
   transfer_method = "rsync"
   modules = []
   launcher_args = []

   # Add one table per partition using limits and node resources from your
   # cluster documentation or administrator. Memory values are in MiB.
   [sites.cluster.partitions.debug]
   max_duration = "02:00:00"
   min_nodes = 1
   max_nodes = 4
   cores_per_node = 64
   sockets_per_node = 2
   memory_per_node = 262144
   gpus_per_node = 0
   gpu_memory_per_gpu = 0

   [sites.cluster.environment]
   # LD_LIBRARY_PATH = "${PARALLEL_HDF5_LIB}:${LD_LIBRARY_PATH}"

   [sites.cluster.run_config]
   nodes = 2
   duration = "00:30:00"
   ranks_per_node = 4
   ranks_per_task = 1
   scheduler_heartbeat_timeout = 300
   # Optional controller-enforced deadline for a full-rank MPI check before sizing.
   # mpi_health_check_timeout = "00:02:00"

An Enterprise HPC installation adds a closed, generated ``enterprise_hpc``
table to the existing Slurm site configuration. It does not add constructor
options for individual Enterprise HPC settings. Do not hand-author it: the
Deployment installer records only a profile identifier, the installed bundle
root, the bundle manifest digest, and its schema identifier. The digest-anchored
bundle manifest binds the compatibility row, schemas, solver, and Python artifact. Host trust,
paths, modules, partitions, launcher, and resource defaults remain ordinary
Slurm profile settings. FrequenSolve verifies the combined profile before
transfer or scheduler spend. When ``enterprise_hpc`` is absent, generic Slurm
configuration and behavior are unchanged.

Stampede3 profiles use the built-in preset and create generic ``SlurmSite``
instances. The preferred setup command prompts for the username and writes the
profile:

.. code-block:: bash

   frequensolve site configure stampede3 \
     --account your-tacc-project-id \
     --solver /work/shared/frequensol/FS_seismic

The generated profile is equivalent to this minimal configuration:

.. code-block:: toml

   [sites.stampede3]
   type = "slurm"
   preset = "stampede3"
   username = "username"
   solver = "/work/shared/frequensol/FS_seismic"
   # Omit work_dir to use $WORK/frequensolve. Move it to $SCRATCH/frequensolve
   # when the work filesystem is slow (purged; fetch results to keep):
   # storage = "scratch"
   # Or choose another writable base; a leading $WORK or $SCRATCH is resolved
   # on the login node:
   # work_dir = "$SCRATCH/frequensolve-runs"
   # Optional future location for models and other high-I/O data:
   # scratch_dir = "$SCRATCH/frequensolve"
   # tmp_dir = "$SCRATCH/frequensolve/tmp"

   [sites.stampede3.run_config]
   account = "your-tacc-project-id"
   nodes = 1
   duration = "00:30:00"
   ranks_per_node = 2
   ranks_per_task = 1

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Meaning
   * - ``transfer_method``
     - ``rsync`` or ``sftp``. Defaults to ``rsync``.
   * - ``default_partition``
     - SLURM partition used by default. A top-level ``queue`` and the direct
       constructor's ``default_queue`` remain compatibility aliases.
   * - ``hostname``
     - Login host for generic ``slurm`` profiles.
   * - ``username``
     - SSH login name. This is not a secret and belongs in the profile.
   * - ``credential``
     - Stable lookup name that separates this profile's secrets in the OS
       keyring.
   * - ``ssh_key``
     - Optional local private-key path. SSH agent keys are attempted first.
   * - ``solver``
     - Absolute path to the remote ``FS_seismic`` router executable.
       FrequenSolve launches this configured executable for initialization,
       frequency tasks, imaging postprocessing, packing, and curvature
       postprocessing.
   * - ``work_dir``
     - Remote base directory for relative project, simulation, and job
       paths. It may be on any writable remote filesystem. Absolute paths in
       those definitions are not rebased. Persistent-session directories,
       scheduler heartbeat and status files, and curvature mirrors without
       ``scratch_dir`` live below it. When omitted, FrequenSolve uses
       ``$WORK/frequensolve``, or ``$SCRATCH/frequensolve`` with
       ``storage = "scratch"``. ``work_dir``, ``scratch_dir`` and ``tmp_dir``
       are absolute paths or start with one remote variable, such as
       ``"$SCRATCH/frequensolve"`` or ``"${WORK}/runs"``, which FrequenSolve
       resolves once in the login shell and logs.
   * - ``storage``
     - ``"work"`` (default) or ``"scratch"``: the remote filesystem variable,
       ``$WORK`` or ``$SCRATCH``, below which ``work_dir`` defaults to
       ``frequensolve``. Use ``"scratch"`` when the work filesystem is slow;
       scratch filesystems are usually purged and not backed up. It cannot be
       combined with ``work_dir``.
   * - ``scratch_dir``
     - Optional remote scratch directory reserved for future model and
       high-I/O storage. Sauce curvature postprocessing stages its HDF5 inputs
       and results below ``curvature/`` here, or below ``work_dir`` when unset,
       until ``site.remove_curvature_files()`` deletes them.
   * - ``transfer_stall_timeout``
     - Seconds an upload or download may make no progress before it is
       stopped: rsync's ``--timeout``, the SFTP size watchdog for OpenSSH
       control-socket transfers, or the Paramiko channel timeout. Defaults to
       120. Slow transfers that keep moving are never cut off.
   * - ``transfer_attempts``
     - Attempts per upload or download after a stall or SSH connection
       failure (default 3; ``1`` disables retries), with a 2, 4, ... second
       backoff. Before retrying, FrequenSolve checks that the login connection
       still completes a round trip; if a shared OpenSSH control connection no
       longer answers, it switches to another live one under
       ``~/.ssh/control`` (for example one reopened with ``frequensolve site
       connect``) and stops retrying when none is found. It never closes your
       control connection, which would need MFA to reopen. rsync retries
       resume from the partially transferred file (``--partial``); SFTP
       retries restart the file. Missing files, permission errors and failed
       size verification are not retried.
   * - ``tmp_dir``
     - Optional remote directory for transient transfer tarballs on the
       login node. Defaults to ``/tmp``, which ``storage`` does not change.
   * - ``modules``
     - Environment modules loaded from left to right before the solver starts.
       A preset may supply a runtime stack; an explicit profile array replaces
       it when a different solver build needs other modules.
   * - ``environment``
     - Non-secret environment variables exported for solver runs. Credential
       variables are rejected and must use the credential mechanisms below.
       Simple ``${NAME}`` references expand after every configured module has
       loaded; other shell expressions remain escaped.
   * - ``mpi_wrapper``
     - MPI launcher such as ``srun`` or ``ibrun``. Batch jobs give every rank
       its own ``cores_per_node // ranks_per_node`` cores for its threads:
       ``srun --cpus-per-task``, TACC ``ibrun ... task_affinity``, or, for Open
       MPI ``mpirun``/``mpiexec`` on Linux, ``--map-by slot:PE=<threads>
       --bind-to core`` with ``OMP_PLACES=cores`` and ``OMP_PROC_BIND=close``
       unless the environment sets them. Other ``mpirun``/``mpiexec`` ranks
       stay unbound. Binding or mapping options in ``launcher_args`` or Open
       MPI binding variables in the environment take precedence. Batch scripts
       request ``--cpus-per-task`` except with ``ibrun``.
   * - ``launcher_args``
     - Arguments passed to every MPI launcher invocation, such as SLURM's
       ``--kill-on-bad-exit=1`` and ``--wait=30``.
   * - ``poll_interval``
     - Seconds between scheduler status polls.
   * - ``scheduler_heartbeat_timeout``
     - Maximum seconds without a new adaptive-scheduler or persistent-session
       controller heartbeat before a running SLURM job is reported failed.
       Defaults to 300; shorter windows misfire when the shared work
       filesystem stalls. Set it to ``None`` through ``SlurmRunConfig`` to
       disable the check.
   * - ``session_command_timeout``
     - Seconds one persistent-session filesystem command on the login node
       (session directory creation and lease, request and command publication)
       may take before it fails with the command and path named. Defaults to
       120. Raise it when the work filesystem is slow.
   * - ``mpi_health_check_timeout``
     - Optional SLURM step time limit for a full-rank solver MPI health check
       before sizing. The solver must support ``--mpi-health-check``. Disabled
       by default for compatibility with older solver installations. Requires
       batch mode with ``srun``; persistent sessions and other MPI launchers
       reject this option before input upload or launch.
   * - ``account``
     - HPC allocation/account name.
   * - ``max_duration``
     - Maximum allowed run duration for generic SLURM config validation.
   * - ``min_nodes`` / ``max_nodes``
     - Allowed node-count bounds for generic SLURM config validation.
   * - ``cores_per_node`` / ``memory_per_node``
     - Generic SLURM node shape metadata.
   * - ``partitions``
     - Optional tables keyed by partition name. Each table can define
       ``max_duration``, node bounds, cores, sockets, memory, and GPUs per node.
       When a run selects another known partition, FrequenSolve resolves its
       limits and node shape before validating and sizing the run. On a
       partition with ``gpus_per_node`` (GPU devices, or stacks, per node) runs
       default to one MPI rank per device unless the run sets
       ``ranks_per_node``, ranks of tasks that share a node select different
       devices, and ``gpu_memory_per_gpu`` (MiB per device) also bounds each
       rank's share of a task's memory estimate. A partition's
       ``environment`` table is exported by jobs on that partition; the site
       ``environment`` overrides its values. The Stampede3 ``pvc`` preset uses
       it to enable GPU-aware MPI.
   * - ``nodes``
     - Requested nodes for submitted jobs.
   * - ``duration``
     - Requested wall time, such as ``"00:30:00"``.
   * - ``ranks_per_node`` / ``ranks_per_task``
     - MPI rank layout for solver runs. ``procs_per_node`` and
       ``procs_per_task`` remain compatibility aliases. Batch jobs and
       persistent sessions size each task's ranks from its memory estimate;
       ``ranks_per_task`` passed to ``session.submit`` sets each task's
       minimum. Tasks that read a linearization's saved inputs
       run on the rank count that wrote them; a job whose tasks all do so
       skips sizing (see `Partitioned Solver Inputs`_).
   * - ``notify_on`` / ``notify_email``
     - Scheduler notification settings when supported.
   * - ``run_path``
     - Optional remote run directory override.
   * - ``slurm_args``
     - Extra scheduler arguments as a TOML array of strings.
   * - ``curvature_run_config``
     - Optional table of run fields (``nodes``, ``ranks_per_node``,
       ``duration``, ``queue``, ...) for Sauce curvature batch jobs. The run
       fields above apply when it is omitted. Without ``ranks_per_node``,
       curvature uses one rank per socket.
   * - ``background_layout``
     - Files of imaging background checkpoints: ``rank`` (default) or
       ``shared`` (see `Partitioned Solver Inputs`_).
   * - ``verbose``
     - Print user-facing status messages in addition to logging.

For generic ``slurm`` profiles, config fields and run fields may be written
flat as shown above, or grouped under nested ``config`` and ``run_config``
tables. Preset values are loaded first, nested profile tables are merged over
them, and explicit construction-time overrides are applied last.

Adaptive Slurm Task Placement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adaptive ``srun`` sweeps map reserved rank slots onto the allocation's ordered
hosts and per-node task counts, including unequal capacities. Each step uses
an explicit repeated-host file with ``--distribution=arbitrary --exclusive
--exact``. Tasks can share a node on disjoint CPUs; they do not use
``--overlap``. Allocation-wide node and task placement variables are removed
from the step environment. Thread counts, CPU binding and memory settings are
preserved. Conflicting explicit placement options in ``launcher_args`` are
rejected. Multi-component heterogeneous Slurm jobs are not supported.

This applies to both observation and imaging tasks. A launched ``srun`` is
reported as ``launching`` until rank zero starts on a worker and records its
Slurm step ID. Startup waits longer than 60 seconds appear in the status and
scheduler log; reserved slots remain held until the launcher exits. This
signal confirms step startup, not completion of solver MPI initialization.

Developers can run a bounded, solver-free placement check inside an explicitly
approved, otherwise idle allocation with at least three rank slots. It starts
two concurrent steps using one and two ranks, checks actual hosts against their
host files, and terminates pending launchers on timeout. The working directory
must be shared by all allocated nodes; it does not submit a batch job:

.. code-block:: bash

   FS_SLURM_PLACEMENT_PROBE=1 FS_SLURM_PROBE_DIR=/approved/shared/scratch \
     python -m pytest -m hpc tests/test_adaptive_scheduler_slurm_live.py

Partitioned Solver Inputs
~~~~~~~~~~~~~~~~~~~~~~~~~

Sauce partitions version-3 objective states and vectors, receiver states and
background checkpoints by MPI rank, records the rank count of the task that
wrote them, and rejects a later JVP, VJP, normal or receiver action that reads
them on another count (a background is recomputed instead). Version-5
objective states and vectors reload on any rank count and impose none. Local
and SLURM sites therefore run each task reading a partitioned input on
the rank count recorded in its producer's manifest, read from the fetched
copy or, when absent, from the remote project in one login command; all other
tasks keep their adaptive sizing. A batch job whose tasks all have recorded
counts skips the sizing pass. A required count larger than the job's ranks per
task fails before submission. Persistent sessions pin each task the same way.

``site.supports_background_reuse`` is ``True`` for local and SLURM sites,
which keep result directories in place for later actions. FrequenSol Cloud
does not keep results for later workers, so it reports ``False`` and
workflows recompute the background there. Source batching follows the job's
acquisition and is unchanged by the site. A background checkpoint is a task
manifest (``fs-background-state-2``) and the HDF5 files it names below
``<manifest stem>/generations/<run>/``; the profile key ``background_layout``
chooses them for the site's file system: ``rank`` (default) writes one file
per MPI rank without collective I/O, ``shared`` one file per task through
parallel HDF5, which keeps the file count independent of the rank count.
Reuse needs the producer's rank count in both layouts, and Sauce's
``FS_BACKGROUND_LAYOUT`` environment variable (for example in the profile's
``environment``) overrides the setting.
``site.remove_result_files(job, ["background_1.json", ...])`` deletes files by
path relative to a job's result directory once they are no longer needed,
together with the directories that this leaves empty.

Module And Library Setup
~~~~~~~~~~~~~~~~~~~~~~~~

Keep site-specific runtime dependencies in the profile. Modules load in the
listed order, and environment exports run afterward. For example, a build that
needs a parallel-HDF5 module to take precedence over another dependency can be
configured without changing Python code:

.. code-block:: toml

   [sites.cluster]
   modules = ["dependency/1", "parallel-hdf5/2"]

   [sites.cluster.environment]
   LD_LIBRARY_PATH = "${PARALLEL_HDF5_LIB}:${LD_LIBRARY_PATH}"

The generated setup is equivalent to:

.. code-block:: bash

   module load dependency/1
   module load parallel-hdf5/2
   module list
   export LD_LIBRARY_PATH="${PARALLEL_HDF5_LIB}:${LD_LIBRARY_PATH}"

Only simple braced references such as ``${NAME}`` are expanded. This supports
composition with variables created by module files while preventing profile
values from becoming arbitrary shell commands.

The common resource settings can be passed directly when constructing a
profile-based HPC site, without creating a ``SlurmRunConfig``:

.. code-block:: python

   site = fs.Site(
       profile="stampede3",
       queue="spr",
       nodes=8,
       ranks_per_node=8,
       duration="00-00:30:00",
   )

The same keywords can override those defaults for one submission:

.. code-block:: python

   run = site.submit(
       job,
       queue="spr",
       nodes=8,
       ranks_per_node=8,
       duration="00-00:30:00",
   )

The packaged ``site_presets.toml`` catalog currently defines Stampede3's
``spr``, ``icx``, ``skx``, and ``skx-dev`` CPU partitions, its ``pvc`` GPU
partition (see :doc:`stampede3_getting_started`), plus its standard
Intel MPI, PETSc, and parallel-HDF5 runtime modules. These values are defaults:
local profile values can override them, and TACC's ``qlimits`` output remains
authoritative because queue policy can change without notice. See the
`Stampede3 user guide <https://docs.tacc.utexas.edu/hpc/stampede3/>`_ for the
current system and queue details.

.. warning::

   Intel MPI asynchronous progress is not yet fully supported. It can sometimes
   race during MPI initialization, so FrequenSolve currently rejects
   ``mpi_async_progress = true``. Leave the option unset or set it to ``false``.

The intended resource layout reserves one core per MPI rank for progress. For
example, an ``spr`` node has 112 cores; with eight ranks per node, asynchronous
progress would reserve the last core of each 14-core rank block and leave 13
solver threads per rank.

FrequenSolve sets ``MKL_NUM_THREADS=1`` and ``MKL_DYNAMIC=FALSE`` for local and
SLURM solver processes so MKL does not add nested threading underneath the
solver's own thread controls. HPC launch scripts also default to
``OMP_WAIT_POLICY=PASSIVE`` and ``KMP_STACKSIZE=20M``.
Stampede3 PVC batch and persistent-session launches set ``KMP_BLOCKTIME=0``
so idle OpenMP workers sleep immediately, leaving CPU time for GPU submission.
Other HPC partitions retain the ``20ms`` block-time default.
Explicit profile ``environment`` values override these defaults. Local users
may set ``OMP_NUM_THREADS`` under the selected profile's ``environment`` table.
HPC run scripts export ``OMP_NUM_THREADS=$n_threads`` alongside the existing
``-nthreads $n_threads`` solver argument.
``KMP_STACKSIZE`` is likewise profile-controlled because its appropriate value
depends on the solver launch and HPC runtime.

Credentials, Compatibility, And Precedence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passwords and SSH-key passphrases must not be written to ``site.toml``.
FrequenSolve first tries an existing SSH control socket, the SSH agent, and the
configured private key. When a password or key passphrase must be entered, the
prompt is hidden. After authentication succeeds, the value is saved through
``keyring`` to macOS Keychain, Windows Credential Manager, or a supported Linux
Secret Service. Two-factor codes are prompted for every login and are never
saved.

On a headless system without a usable keyring, authentication continues for the
current session and prompts again next time. SSH agents are preferred for these
systems. Process environment variables such as ``HPC_USERNAME``,
``HPC_PASSWORD``, and ``SSH_PASSPHRASE`` remain available for headless
credential automation. Non-secret site settings, including usernames,
allocation accounts, solver paths, and work directories, belong in
``site.toml``. FrequenSolve no longer loads a project ``.env`` file
automatically.

FrequenSolve loads its scheduler templates and adaptive runner from installed
package resources. It does not read ``PYTHONPATH`` or require a source-checkout
path for site setup. Supported solver builds own their compiled-in data
resources.

Site settings are selected in this order: an explicit constructor argument,
the selected ``site.toml`` profile, then the backend default. Unknown SSH host
keys are rejected; connect once with the system ``ssh`` client to verify and
save a site's key before first use.

Debugging SSH And SLURM
~~~~~~~~~~~~~~~~~~~~~~~

Enable package debug logging after constructing or loading the project and
before creating the site. The console shows the active stage while the file
keeps the complete trace:

.. code-block:: python

   import frequensolve as fs

   fs.configure_logging(
       level="DEBUG",
       console=True,
       log_file="./frequensolve-debug.log",
   )
   site = fs.Site(profile="stampede3")

The HPC backend also writes ``/tmp/log/frequensolve/hpc.log``. Debug messages
identify each remote command and transfer backend without logging passwords,
private keys, key passphrases, or two-factor codes. OpenSSH commands that reuse
a verified control socket are non-interactive and bounded by connection and
command timeouts, so an expired socket produces an exception instead of an
invisible credential prompt. SFTP file transfers over the socket have no total
time limit, so multi-gigabyte files can finish, but fail once the transferred
size stops changing for the command timeout (120 seconds).

The site CLI uses a bounded remote probe rather than relying only on the local
OpenSSH master process when deciding that a shared connection is alive. Inspect
managed connections with ``frequensolve site connections``. Close the default
profile with ``frequensolve site disconnect``, select another one with
``--profile``, or close every configured SSH connection with
``frequensolve site disconnect --all``. A timed-out probe causes
``frequensolve site connect`` to close and remove the stale socket before it
authenticates again.

At DEBUG logging level, ``rsync`` streams file names and ``-P`` transfer
progress to the console. At INFO and higher levels, FrequenSolve runs rsync
quietly with partial-transfer preservation and includes captured stderr only
when the transfer fails. The site's ``verbose`` setting continues to control
other interactive status messages but does not enable rsync output.

Before uploading a project, FrequenSolve inspects HDF5 files in its disposable
transfer staging directory. Set top-level ``[host].tmp_dir`` to choose where
this local staging happens; otherwise Python's platform temp directory is used.
Files dominated by space from deleted or replaced datasets are repacked there,
so only live HDF5 objects cross the network. The project's source HDF5 files
are not modified.

Simulation saves also treat the newly serialized JSON as authoritative for the
simulation input store: datasets no longer referenced by that JSON are removed,
and a store with substantial deleted-object space is atomically repacked. This
prevents repeated authoring saves or a change from an in-memory property to a
remote file reference from leaving obsolete arrays in the simulation HDF5 file.


Named Cloud execution sites
---------------------------

A named profile can select the installed managed execution site without naming
partitions, instance types, or cluster provisioning settings::

    [sites.hosted]
    type = "aws"
    domain = "example.frequensol.com"
    cluster_id = "frequensol-shared-us-east-1"

    [sites.hosted.execution_resources]
    nodes = 1
    mpi_ranks = 1
    wall_time_seconds = 600
    cpu = 2
    memory_mib = 3891

Select it with ``fs.Site(profile="hosted")``. ``cpu`` and ``memory_mib`` are
optional allocation ceilings; the site planner rejects a frequency plan that
exceeds them. Omit ``execution_resources`` to use one node, one MPI rank, and a
one-hour wall time, allowing the existing site planner to choose CPU and memory.
Batch and legacy managed-Slurm profiles are no longer supported. Use the named
site and its resource table; remove old backend and partition settings.

``run.execution`` and ``result.execution`` expose ``cluster_id``,
``logical_attempt_id``, ``provider_job_id``, ``state``, and ``failure_reason``.
Provider identity comes from the current Cloud record. Status requests require
the current GraphQL schema and report incompatible deployments directly.

Cloud results belong to a submission. Keep the returned run handle when running
the same job again::

    first = site.submit(job)
    first_result = first.wait()
    second = site.submit(job, rerun=True)
    first_result.traces()
    files = first_result.output_files(kind="vtu", existing=True)

The first result continues to fetch the first submission. Cloud downloads check
that submission's authenticated result location and save beneath
``jobs/<simulation>/<job>/results/runs/<simulation-id>/``. Each run handle owns a
copy of the job's result layout and nested simulation metadata, including receiver
groups and components. Later edits to the authored simulation do not reinterpret
earlier results. Downloads do not replace the authored job's results. Calling a fetch method directly with the authored job selects its latest
submission. Results require the matching Slurm Cloud and runtime deployment;
unscoped historical result directories are not used as a fallback.

For Cloud imaging runs, ``save_path`` must be inside the job's result directory.
Invalid imaging paths and snapshot preparation failures are rejected before the
submission API is called, so they cannot leave a running job without a handle.
Local and SSH imaging paths are unaffected.

Only the registered FrequenSol-managed sites are supported here. This does not
discover, provision, or connect arbitrary external clusters. Site operations and
advanced provider settings belong to the service deployment, outside job APIs.


Dispatcher installation selection
---------------------------------

When ``solver`` selects ``FS_seismic`` by a directory-qualified path, local and
SLURM sites set ``FS_SOLVER_PATH`` to that directory for execution. An inherited
shell setting or a loaded module therefore cannot redirect the dispatcher to an
older installation. An explicit ``environment.FS_SOLVER_PATH`` still overrides
this default for installations that intentionally separate the dispatcher and
backend executables. A bare ``FS_seismic`` command retains normal site lookup.

HPC launch scripts are stored under each remote job's ``logs/batch`` directory
with a unique name for every submission. Concurrent submissions no longer share
an overwriteable ``sweep.slurm`` or ``sweep.sh`` file. This isolates launch
scripts; concurrent runs writing the same job's simulation or result files still
require separate job/output directories.

Strict Cloud retries
~~~~~~~~~~~~~~~~~~~~

On a managed Cloud site with ``strict-partial-retry.v1`` installed, retry a
terminal run after its Credit settlement completes:

.. code-block:: python

   retry = site.submit(job, retry=True)

Pass the authored ``job``, not a prior run's immutable result snapshot. The server
selects the latest compatible, terminal, Credit-settled run by submission date
(not by how many frequencies it can reuse), then verifies its completed frequencies and archived
artifacts, then creates a separate run for failed, missing or invalid work.
Unchanged scientific inputs, output requests, frequency ordering, solver/runtime
version, execution mode and MPI topology are required. Resource budgets such as
memory and timeout may change through the named execution profile. Both
independent-frequency and adaptive execution support this strict subset.

Reused frequencies have no new compute usage. New processing uses the ordinary
rates and run minimum, and the original charge remains unchanged. Older runs
without verified retry receipts require a full rerun. To explicitly recompute
all frequencies, use ``site.submit(job, force=True)``.
Changed scientific inputs or solver versions also require a full rerun.
Local and Slurm sites also accept ``retry=True``; they already reuse completed
task outputs by default, so the flag makes that intent explicit there.

Saved frequency plans (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Saved-plan APIs require a compatible Cloud service and qualified solver/hardware
installation. They are currently capability gated and are not enabled by adding
a site configuration option. Legacy explicit ``nodes`` and ``mpi_ranks`` retain
their existing meanings; automatic selection uses separate policy ceilings.

``site.plan(job)`` returns a ``PlanHandle``. Preparing private inputs is free;
paid sizing starts only after accepting the actual quote within an explicit
decimal-string Credit maximum. The handle exposes ``id``, ``revision``,
``status``, ``rows``, ``refresh()``, ``wait()``, ``edit()``, ``validate()``,
``cancel()`` and ``export(path)``. Validation may return an additional paid
sizing quote; it does not silently authorize it. A reviewed execution uses
``site.execute_plan(plan, maximum_credits=execution_budget)`` and a fresh quote.

Use ``plan.quote_execution()`` to inspect the current execution Credit maximum
without reserving Credits or compute. Execute accepts an explicit Credit bound;
it never prompts repeatedly for work already authorized. For an intentional new
run of a completed plan, supply a deliberately new ``idempotency_key`` to
``plan.execute(...)`` or ``site.execute_plan(...)``. This requests a fresh quote,
authorization and run identity after the server verifies prior settlement.
Without a new key, execute recovers the existing accepted run. An accepted failed
solver job is never retried automatically, and a second execution cannot start
while the same plan is already running.


The experimental automatic-submission path returns a ``RunHandle`` even when
source preparation or planning has not produced a Simulation yet. Its ``id``
remains ``None`` until the server publishes the actual execution identity;
``run.plan`` retains the saved plan. Polling does not issue a new execution or
expand authorization. A pause raises ``PlanExecutionPaused`` with its ``plan``
attached, so validation or pricing changes remain reviewable rather than becoming
a fabricated failed simulation. Execution logs and solver outputs are unavailable
before a real execution exists. Combined automatic authorization and its Cloud
orchestration must be installed before enabling this workflow.

After ``plan.refresh()``, ``plan.progress`` provides the last controller status,
completed/failed/canceled evaluation or frequency counts, and an observation
time. It returns ``None`` when an older server has no compatible status snapshot.
Reading this property does not poll or launch work. Submitted allocations are
reported as waiting for compute or results; a scheduler job identity alone does
not establish that a worker has started. Progress is diagnostic and cannot
replace validated sizing evidence or settled Credit receipts.

Planning captures a private copy of staged scientific interpretation. Later
authoring or staging edits cannot change an accepted plan's result reader. Saved
plans retain their original native input fingerprints while final storage is
scoped to the accepted run. Cancellation is pinned to that run and revision,
including after the same user's plan handle is refreshed or reused.

Settled execution outputs
~~~~~~~~~~~~~~~~~~~~~~~~~

``plan.results()`` inspects the latest run retained on the handle. Pass
``execution_id=...`` to inspect an earlier settled run without changing the
current plan. Reads and downloads do not authorize compute or retry solver work.

.. code-block:: python

   results = plan.results(execution_id=run.id)
   local_run = results.download("./downloaded-results")
   artifacts = results.artifact_catalog(local_run, frequency_index=0)
   traces = results.trace_dataset(local_run, frequency_index=0)

Downloads verify exact provider versions and file digests before publishing a
new local run directory. Existing directories are never overwritten. Each
frequency retains its own native files and layout-specific geometry; failed or
canceled siblings remain in the result index. ``artifact_catalog`` reads the
producer's logical artifact records. ``trace_dataset`` opens one converged
receiver frequency with its matching survey metadata, preserving the original
task index and complex frequency. It rejects changed local data, incompatible
trace families, missing metadata and nonconverged results. It does not manufacture
a combined packed sweep or missing multi-node memory measurements. Inspect
``results.data`` for requested versus actual allocations and scientific status.
This native-archive workflow also requires compatible, qualified Cloud/Runtime
readers and remains gated until installation and live acceptance are complete.


Saved allocation revisions
~~~~~~~~~~~~~~~~~~~~~~~~~~

``site.list_plans(project_key=project_key)`` returns one page of owned saved-plan
and pending-preparation summaries, an optional opaque ``cursor``, and reader
availability. Pass that cursor unchanged on the next call to list more. Listing
does not authorize compute or require a new quote; use an item's ``plan_id`` with
``site.open_plan(...)`` to inspect the complete revision.

``site.open_plan(plan_id, project_key=project_key, revision=revision)`` opens an
exact immutable allocation revision for inspection and export. The historical
handle is read-only; refreshing it keeps that revision selected. Open the plan
without ``revision`` to return to its current editable state. Historical reads
require current project membership but do not reserve compute, authorize a
charge or require a new cluster grant.
