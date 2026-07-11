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
script or notebook.

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
   * - ``stampede3`` or ``tacc``
     - ``frequensolve[hpc]``
     - Stampede3 access, allocation details, and a solver installation on the
       cluster.

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
   type = "stampede3"
   rel_path = "scratch/frequensolve_tutorials"
   queue = "skx-dev"
   nodes = 1
   duration = "00:30:00"

The profile name and backend type are separate. In the example above,
``hpc`` is the profile selected with ``fs.Site(profile="hpc")``, while
``type = "stampede3"`` selects the Stampede3 backend.

Select a profile in Python with:

.. code-block:: python

   site = fs.Site()                 # uses default
   local = fs.Site(profile="local")
   cluster = fs.Site(profile="hpc")

Set ``FREQUENSOLVE_SITE_CONFIG`` or pass ``fs.Site(config_path=...)`` when a
test, notebook, or shared workstation should use a different config file:

.. code-block:: python

   shared = fs.Site(config_path="/path/to/site.toml", profile="hpc")

Direct constructors such as ``fs.LocalSite(...)``, ``fs.AWSSite(...)``, and
``fs.Stampede3Site(...)`` remain available when code intentionally targets one
backend.

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
   shutdown_on_completion = true
   n_workers = 2
   dashboard_port = 8787

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
   * - ``dashboard_host``
     - Hostname for the Dask dashboard. Defaults to ``localhost``.
   * - ``dashboard_port``
     - Dashboard port. ``0`` lets Dask choose an available port.
   * - ``verbose``
     - Print user-facing status messages in addition to logging.

HPC Profiles
~~~~~~~~~~~~

Generic :term:`SLURM` profiles create ``SlurmSite`` instances and require the ``hpc``
extra.

.. code-block:: toml

   [sites.cluster]
   type = "slurm"
   hostname = "login.example.edu"
   username = "jsmith"
   credential = "example-primary"
   ssh_key = "~/.ssh/id_ed25519"
   solver = "/work/shared/frequensol/fs3d_s"
   work_dir = "/scratch/jsmith"
   python_path = "/home/jsmith/FrequenSolve"
   rel_path = "frequensolve/tutorials"
   queue = "debug"
   account = "allocation"
   transfer_method = "rsync"

   [sites.cluster.run_config]
   nodes = 2
   duration = "00:30:00"
   ranks_per_node = 4
   ranks_per_task = 1

Stampede3 profiles create ``Stampede3Site`` instances:

.. code-block:: toml

   [sites.stampede3]
   type = "stampede3"
   username = "jsmith"
   credential = "tacc-primary"
   ssh_key = "~/.ssh/id_ed25519"
   solver = "/work/shared/frequensol/fs3d_s"
   work_dir = "/scratch/jsmith"
   rel_path = "scratch/frequensolve_tutorials"
   queue = "skx-dev"

   [sites.stampede3.run_config]
   nodes = 1
   duration = "00:30:00"

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Meaning
   * - ``rel_path``
     - Required remote project/work directory relative to the site work root.
   * - ``transfer_method``
     - ``rsync`` or ``sftp``. Defaults to ``rsync``.
   * - ``queue`` / ``default_queue``
     - Queue used for the site default and for run submissions unless
       overridden.
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
     - Solver executable path on the remote system.
   * - ``work_dir``
     - Remote work-directory root. ``rel_path`` is appended to this value.
   * - ``python_path``
     - Optional local FrequenSolve source path used for scheduler templates.
   * - ``mpi_wrapper``
     - MPI launcher such as ``srun`` or ``ibrun``.
   * - ``poll_interval``
     - Seconds between scheduler status polls.
   * - ``account``
     - HPC allocation/account name.
   * - ``max_duration``
     - Maximum allowed run duration for generic SLURM config validation.
   * - ``min_nodes`` / ``max_nodes``
     - Allowed node-count bounds for generic SLURM config validation.
   * - ``cores_per_node`` / ``memory_per_node``
     - Generic SLURM node shape metadata.
   * - ``nodes``
     - Requested nodes for submitted jobs.
   * - ``duration``
     - Requested wall time, such as ``"00:30:00"``.
   * - ``procs_per_node`` / ``procs_per_task``
     - Process layout for solver runs.
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
tables. ``stampede3`` profiles use Stampede3's built-in machine config and
therefore do not accept a nested ``config`` table.

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
``HPC_PASSWORD``, ``TACC_USERNAME``, ``TACC_PASSWORD``, ``SSH_PASSPHRASE``,
``LOCAL_SOLVER_EXECUTABLE``, ``FS_SOLVER_EXECUTABLE``,
``STAMPEDE3_SOLVER_EXECUTABLE``, ``FS_HPC_WORK_DIR``,
``STAMPEDE3_WORK_DIR``, and ``FS_PYTHON_PATH`` remain compatibility fallbacks
for direct constructors and automation. FrequenSolve no longer loads a project
``.env`` file automatically.

Values are selected in this order: an explicit constructor argument, the
selected ``site.toml`` profile, a process environment fallback, then the
backend default. Unknown SSH host keys are rejected; connect once with the
system ``ssh`` client to verify and save a site's key before first use.
