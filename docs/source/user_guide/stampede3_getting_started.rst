.. _stampede3-getting-started:

Getting Started On Stampede3
============================

FrequenSolve uses Stampede3 as a remote execution site. The normal setup is:

- Run Python, notebooks, and ``frequensolve`` on your laptop or workstation.
- Let ``fs.Site()`` connect to Stampede3, stage the saved project, submit a
  SLURM job, poll it, and fetch its results.
- Keep the separately licensed fast solver executables on Stampede3.

Installing the Python package does not install or license the solver. Obtain
the Stampede3 solver installation path and any required runtime modules from the FrequenSol
administrator before trying to submit a job.

Prerequisites
-------------

You need:

- A TACC account with an active Stampede3 allocation and MFA configured.
- Your TACC username and, if your login belongs to more than one project, the
  allocation/project id to charge.
- A licensed fast solver installation on Stampede3 whose required concrete
  executables are executable by your account.
- Python 3.10 through 3.14, ``ssh``, and preferably ``rsync`` on the computer
  where you will run FrequenSolve.

TACC's `Stampede3 user guide
<https://docs.tacc.utexas.edu/hpc/stampede3/>`_ is authoritative for current
login, filesystem, and queue policy. Queue limits can change; run ``qlimits``
on Stampede3 before increasing a resource request.

Install A Pinned Release
------------------------

Create an isolated environment on the laptop or workstation that will launch
your Python code. Replace ``0.3.0`` with the exact version supplied by your
FrequenSol administrator:

.. code-block:: bash

   python3 -m venv ~/.venvs/frequensolve
   . ~/.venvs/frequensolve/bin/activate
   python -m pip install --upgrade pip
   python -m pip install "frequensolve[hpc]==0.3.0"
   python -c 'import frequensolve as fs; print(fs.__version__)'

Use an exact version so every user runs the same API and solver contract. Add
the ``visual`` extra if this environment will also plot with matplotlib or
PyVista:

.. code-block:: bash

   python -m pip install "frequensolve[hpc,visual]==0.3.0"

PyPI installs the library, but not the repository's example notebooks. To use
the examples for the same release, clone its matching tag:

.. code-block:: bash

   git clone --depth 1 --branch v0.3.0 \
     https://github.com/FrequenSol/FrequenSolve.git
   python -m pip install jupyterlab
   python -m jupyter lab \
     FrequenSolve/examples/tutorials/02_sites/02_hpc_sites.ipynb

Verify TACC Access And Remote Values
------------------------------------

Connect once with the system SSH client before using FrequenSolve. This both
tests TACC/MFA access and lets you verify Stampede3's host key:

.. code-block:: bash

   ssh your-tacc-username@stampede3.tacc.utexas.edu

From the Stampede3 shell, record the paths and policies that apply to your
account:

.. code-block:: bash

   echo "$WORK"
   echo "$SCRATCH"
   qlimits
   test -x /absolute/path/to/solver-installation/FS_seismic \
       && echo "FS_seismic is executable"

When basing a configured directory on ``$WORK`` or ``$SCRATCH``, use the actual
path printed by ``echo``. Other absolute writable remote paths are equally
valid. TOML does not expand remote shell variables, so do not write
``work_dir = "$WORK/frequensolve"`` or
``scratch_dir = "$SCRATCH/frequensolve"``.

Stampede3's ``$SCRATCH`` is appropriate for active, high-I/O job data but is
not backed up and files are subject to purge. ``$WORK`` is persistent across
TACC systems but has a quota and is not intended for high-intensity parallel
I/O. If ``work_dir`` is omitted, FrequenSolve asks the remote login shell for
``$WORK`` and uses ``$WORK/frequensolve``. The optional ``scratch_dir`` can name
a complete absolute directory below ``$SCRATCH`` for future model and high-I/O
storage, but FrequenSolve does not place job data there yet. An explicit
``work_dir`` may point to any other writable Stampede3 filesystem; the default
does not require configured values to live below ``$WORK``.

Reuse An Authenticated SSH Session
----------------------------------

TACC requires MFA. The most reliable setup is to open an OpenSSH control socket
once, complete MFA there, and let FrequenSolve reuse that authenticated
connection:

.. code-block:: bash

   mkdir -p ~/.ssh/control
   ssh -M -S ~/.ssh/control/stampede3 \
     -o ControlPersist=8h \
     your-tacc-username@stampede3.tacc.utexas.edu

After login succeeds, you can exit the interactive shell; ``ControlPersist``
keeps the socket available for its configured lifetime. Re-run the command when
it expires.

FrequenSolve can also try an SSH agent, a configured private key, or
keyboard-interactive password/MFA authentication. Never put a password, private
key contents, key passphrase, or MFA code in ``site.toml``. TACC specifically
warns users not to run ``ssh-keygen`` on Stampede3; manage client authentication
from your local computer and the TACC account portal.

Create ``site.toml``
--------------------

The default configuration path is ``~/.frequensolve/site.toml`` on the computer
running Python. The first call to ``fs.Site()`` creates a starter file if one is
missing and asks you to review it:

.. code-block:: bash

   python -c 'import frequensolve as fs; fs.Site()'

Replace the starter's Stampede3 placeholders with a minimal profile like this:

.. code-block:: toml

   default = "stampede3"

   [sites.stampede3]
   type = "slurm"
   preset = "stampede3"
   username = "your-tacc-username"
   credential = "tacc-stampede3"
   solver = "/absolute/path/to/solver-installation/FS_seismic"
   # work_dir = "/another/writable/remote/path/frequensolve"
   # Reserved for future model and high-I/O storage; not used for jobs yet:
   # scratch_dir = "/absolute/path/printed/by/echo-SCRATCH/frequensolve"
   default_partition = "skx-dev"
   transfer_method = "rsync"
   modules = []
   verbose = true

   # Set this only when TACC requires an explicit project/allocation.
   # account = "your-tacc-project-id"

   # Add this only when you actually use this local private key.
   # ssh_key = "~/.ssh/id_ed25519"

   [sites.stampede3.run_config]
   nodes = 1
   duration = "00:30:00"
   ranks_per_node = 4
   ranks_per_task = 1
   poll_interval = 10

The fields have distinct purposes:

``preset``
   Supplies the Stampede3 login host, ``ibrun`` launcher, known CPU partitions,
   and node shapes packaged with FrequenSolve.

``credential``
   A non-secret label used to separate this profile's entries in the local OS
   keyring. It is not a TACC password or token.

``solver``
   Absolute remote path to the licensed ``FS_seismic`` router. FrequenSolve
   launches this executable for every solver phase, and it chooses the concrete
   implementation from the saved simulation.

``work_dir``
   Absolute remote base directory used when project, simulation, or job paths
   are relative. It can be on any writable remote filesystem. Absolute paths
   are used as written instead. Omit it to use ``$WORK/frequensolve``.

``scratch_dir``
   Optional complete absolute remote scratch directory reserved for future
   model and high-I/O storage. It is accepted and exposed by the site today,
   but job storage does not use it yet.

``account``
   TACC project/allocation charged by SLURM. It is normally necessary only when
   your login has multiple projects or TACC asks you to select one explicitly.

``modules``
   Module names loaded in the generated SLURM script before the solver starts.
   Leave the list empty unless the supplied solver build requires a specific
   runtime stack. Use exact module names supplied with that build. They load in
   list order. Values under ``[sites.stampede3.environment]`` are exported
   afterward and may compose module-defined variables with ``${NAME}`` syntax.

``run_config``
   Default SLURM request. ``skx-dev`` is suitable for a short first check, but
   TACC's live ``qlimits`` output is authoritative. The built-in preset also
   knows the ``skx``, ``icx``, and ``spr`` CPU partitions.

Test The Site Configuration
---------------------------

With the Python environment active and the SSH control socket open, construct
the site without submitting work:

.. code-block:: bash

   python - <<'PY'
   import frequensolve as fs

   site = fs.Site()
   print("backend:", type(site).__name__)
   print("login host:", site.login_host)
   print("remote work directory:", site.work_dir)
   print("solver executable:", site.executable)
   site.close()
   PY

Expected results include a ``SlurmSite`` backend, the Stampede3 login host, and
the configured work base (``$WORK/frequensolve`` when it is omitted from the
profile). Site construction authenticates and checks the remote work base; use
the earlier remote ``test -x`` command to verify the ``FS_seismic`` router.

Solver Router And Precision
---------------------------

FrequenSolve launches ``FS_seismic`` for every Stampede3 solver phase. The
router reads the saved simulation's top-level ``dimension`` and
``Solver.precision``. Single precision is the default; request double precision
before saving or submitting the job:

.. code-block:: python

   sim = project.new_simulation(
       name="model_3d",
       physics="acoustic",
       dimension=3,
   )
   sim.solver.precision = "double"

The Python API writes both values into the simulation JSON for the router.

Submit A First Job
------------------

Author the project and job locally, then submit the job through the configured
site. The HPC tutorial builds a complete small acoustic example. Its submission
pattern is:

.. code-block:: python

   import frequensolve as fs

   # Build or load a saved FrequenSolve job first.
   # job = fs.SimulationJob.load("/path/to/job.json")

   site = fs.Site(
       queue="skx-dev",
       nodes=1,
       ranks_per_node=4,
       duration="00:30:00",
   )
   run = site.submit(job, fetch=True)
   result = run.wait()

   print(result.status)
   print("successful:", result.successful)
   print(result.logs())

The resource arguments override the profile for this Python process. The same
arguments can be passed to ``site.submit(...)`` to override a single job. Start
with a small development-queue request, inspect the generated scheduler and
solver logs, and only then scale the mesh, frequencies, nodes, or duration.

FrequenSolve submits computation to SLURM compute nodes; do not run solver work
on a Stampede3 login node.

Troubleshooting
---------------

Unknown or changed SSH host key
   Connect with the system ``ssh`` command, verify the fingerprint through TACC,
   and let OpenSSH update ``~/.ssh/known_hosts``. FrequenSolve intentionally
   rejects unknown or mismatched keys.

Repeated password or MFA prompts
   Reopen the control socket under ``~/.ssh/control``. Passwords may be stored in
   a supported local OS keyring after successful authentication, but MFA codes
   are never stored.

``Invalid account`` or allocation failure
   Confirm the account in the TACC portal. Remove ``account`` when your login has
   only one default project, or set it to the exact project id when needed.

Unknown partition or rejected time/node request
   Run ``qlimits`` on Stampede3 and update the profile or per-run override. TACC
   policy can change after the packaged preset was released.

Solver not found or shared-library error
   Confirm the absolute ``solver`` path from a compute node and add only the
   modules required by that solver build. The PyPI package does not contain the
   solver.

Transfer failure
   Confirm that local ``rsync`` is installed and that ``work_dir`` is writable.
   Set ``transfer_method = "sftp"`` as a slower fallback when rsync is
   unavailable.

More diagnostics
   Enable ``fs.configure_logging(level="DEBUG", console=True,
   log_file="./frequensolve-debug.log")`` before creating the site. The HPC
   backend also writes ``/tmp/log/frequensolve/hpc.log`` on the local computer.
   See :doc:`site_configuration` for the full field reference.

Installing Python On Stampede3 Itself
-------------------------------------

Remote authoring is possible for advanced users, but it is not required for
normal Stampede3 execution. TACC provides Python through a module:

.. code-block:: bash

   module load python
   python -m venv "$WORK/venvs/frequensolve"
   . "$WORK/venvs/frequensolve/bin/activate"
   python -m pip install --upgrade pip
   python -m pip install "frequensolve==0.3.0"

This is useful for authoring or inspecting saved contracts on TACC. The
``SlurmSite`` workflow is designed for a client machine connecting to Stampede3;
do not make self-SSH from a login node the default user setup. Run notebooks on
a workstation unless you intentionally configure a TACC-supported interactive
session, and submit all compute-heavy work through SLURM.
