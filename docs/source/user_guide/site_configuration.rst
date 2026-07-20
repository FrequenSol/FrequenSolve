.. _site-configuration:

Site Configuration
==================

FrequenSolve runs jobs through a configured execution :term:`site`. A site can
target FrequenSol Cloud, a local solver installation, or an :term:`HPC`
cluster. Most user scripts should call ``fs.Site()`` and let the
:term:`site configuration file` choose the active backend.

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

Set ``FREQUENSOLVE_SITE_CONFIG`` or pass ``fs.Site(config_path=...)`` when a
test, notebook, or shared workstation should use a different config file:

.. code-block:: python

   shared = fs.Site(config_path="/path/to/site.toml", profile="hpc")

Direct constructors such as ``fs.LocalSite(...)`` and ``fs.AWSSite(...)``
remain available when code intentionally targets one backend.
``fs.Stampede3Site(...)`` remains as a compatibility adapter for existing code
and persisted run records.

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

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Meaning
   * - ``domain``
     - FrequenSol app domain used to fetch public cloud configuration. If
       omitted, ``AWSSite`` tries ``FREQUENSOL_DOMAIN``.
   * - ``interactive``
     - When ``true``, prompt for missing login credentials. Defaults to
       non-interactive behavior.
   * - ``verbose``
     - Print user-facing status messages in addition to logging.
   * - ``email`` / ``password``
     - Accepted by ``AWSSite`` for non-interactive login, but should not be
       stored in ``site.toml``. Prefer cached login state or a secrets manager.

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
   * - ``verbose``
     - Print user-facing status messages in addition to logging.

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

   [sites.cluster.environment]
   # LD_LIBRARY_PATH = "${PARALLEL_HDF5_LIB}:${LD_LIBRARY_PATH}"

   [sites.cluster.run_config]
   nodes = 2
   duration = "00:30:00"
   ranks_per_node = 4
   ranks_per_task = 1
   scheduler_heartbeat_timeout = 60

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
   # Omit work_dir to use $WORK/frequensolve, or choose another writable base:
   # work_dir = "/shared/username/frequensolve"
   # Optional future location for models and other high-I/O data:
   # scratch_dir = "/scratch/username/frequensolve"
   # tmp_dir = "/scratch/username/frequensolve/tmp"

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
       frequency tasks, imaging postprocessing, and packing.
   * - ``work_dir``
     - Absolute remote base directory for relative project, simulation, and job
       paths. It may be on any writable remote filesystem. Absolute paths in
       those definitions are not rebased. When omitted, FrequenSolve uses
       ``$WORK/frequensolve``.
   * - ``scratch_dir``
     - Optional absolute remote scratch directory reserved for future model and
       high-I/O storage. FrequenSolve records this setting but does not use it
       for job data yet.
   * - ``tmp_dir``
     - Optional remote directory for transient transfer tarballs and
       provisioning scripts. Use a concrete absolute path on the login
       filesystem. Defaults to ``/tmp``.
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
     - MPI launcher such as ``srun`` or ``ibrun``.
   * - ``poll_interval``
     - Seconds between scheduler status polls.
   * - ``scheduler_heartbeat_timeout``
     - Maximum seconds without a new adaptive-scheduler heartbeat before a
       running SLURM job is reported failed. Defaults to 60. Set it to
       ``None`` through ``SlurmRunConfig`` to disable the check.
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
       limits and node shape before validating and sizing the run.
   * - ``nodes``
     - Requested nodes for submitted jobs.
   * - ``duration``
     - Requested wall time, such as ``"00:30:00"``.
   * - ``ranks_per_node`` / ``ranks_per_task``
     - MPI rank layout for solver runs. ``procs_per_node`` and
       ``procs_per_task`` remain compatibility aliases.
   * - ``notify_on`` / ``notify_email``
     - Scheduler notification settings when supported.
   * - ``run_path``
     - Optional remote run directory override.
   * - ``slurm_args``
     - Extra scheduler arguments as a TOML array of strings.
   * - ``verbose``
     - Print user-facing status messages in addition to logging.

For generic ``slurm`` profiles, config fields and run fields may be written
flat as shown above, or grouped under nested ``config`` and ``run_config``
tables. Preset values are loaded first, nested profile tables are merged over
them, and explicit construction-time overrides are applied last.

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
``spr``, ``icx``, ``skx``, and ``skx-dev`` CPU partitions plus its standard
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
invisible credential prompt.

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
