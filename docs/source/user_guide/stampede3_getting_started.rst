.. _stampede3-getting-started:

Getting Started On Stampede3
============================

FrequenSolve uses Stampede3 as a remote execution site. The normal setup is:

- Run Python, notebooks, and ``frequensolve`` on your laptop or workstation.
- Let ``fs.Site()`` connect to Stampede3, stage the saved project, submit a
  SLURM job, poll it, and fetch its results.
- Keep the separately licensed fast solver executables on Stampede3.

Installing the Python package does not install or license the solver. Obtain
the Stampede3 solver installation path from the FrequenSol administrator
before trying to submit a job. The built-in site preset supplies the standard
runtime modules; a different solver build may require an explicit override.

Prerequisites
-------------

You need:

- A TACC account with an active Stampede3 allocation and MFA configured.
- Your TACC username and, if your login belongs to more than one project, the
  allocation/project id to charge.
- A licensed fast solver installation on Stampede3 whose required concrete
  executables are executable by your account.
- Python 3.10 through 3.14, ``ssh``, and ``rsync`` on the computer
  where you will run FrequenSolve.

TACC's `Stampede3 user guide
<https://docs.tacc.utexas.edu/hpc/stampede3/>`_ is authoritative for current
login, filesystem, and queue policy. Queue limits can change; run ``qlimits``
on Stampede3 before increasing a resource request.

Install A Pinned Release
------------------------

Create an isolated environment on the laptop or workstation that will launch
your Python code. Replace ``0.3.0`` with a later exact version when one is
supplied by your FrequenSol administrator:

.. code-block:: bash

   python3 -m venv ~/.venvs/frequensolve
   . ~/.venvs/frequensolve/bin/activate
   python -m pip install --upgrade pip
   python -m pip install "frequensolve[hpc]==0.3.0"
   frequensolve --version

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

Configure Stampede3
-------------------

Generate a focused Stampede3 profile instead of hand-editing the multi-site
starter file:

.. code-block:: bash

   frequensolve site configure stampede3 \
     --account your-tacc-project-id \
     --solver /absolute/path/to/solver-installation/FS_seismic

Enter your TACC username when prompted. The command writes
``~/.frequensolve/site.toml`` with conservative ``skx-dev`` defaults. Shared
machine settings come from the packaged ``stampede3`` preset, including the
login host, ``ibrun``, partition shapes, and the modules ``intel/25.1``,
``impi/21.15``, ``petsc/3.23``, and ``phdf5``.

The profile does not need explicit ``credential``, ``ssh_key``, ``hostname``,
``modules``, or partition-shape entries. It contains no password, private key,
passphrase, or MFA token. If a custom site config already exists, the command
refuses to overwrite it unless ``--force`` is passed; back up that file before
choosing to replace it.

Authenticate Once Per Work Session
----------------------------------

TACC requires MFA. FrequenSolve can create and manage the reusable OpenSSH
control connection for you:

.. code-block:: bash

   frequensolve site connect

Enter your TACC password and MFA code when requested. Existing OpenSSH agent
and key settings are honored automatically, but a separately configured SSH
key is not required. The shared connection remains in the background for up to
eight hours, so no interactive SSH terminal needs to remain open. Running the
command again safely reuses an active connection. Later scripts, notebooks,
``fs.Site()`` instances, and ``rsync`` transfers discover the same socket even
when they run in different processes, so they do not request separate MFA
tokens.

FrequenSolve does not modify ``~/.ssh/config``. Users who prefer standard
OpenSSH configuration can follow the :download:`optional connection-sharing
guide <../../guides/ssh-connection-sharing.md>`.

Test The Site Configuration
---------------------------

Verify access, ``$WORK`` resolution, the solver path, and preset modules without
submitting a Slurm job:

.. code-block:: bash

   frequensolve site check

The default remote work directory is ``$WORK/frequensolve``. An advanced
profile may instead set a concrete absolute ``work_dir``. TOML does not expand
remote shell variables, so do not write ``work_dir = "$WORK/frequensolve"``.
Stampede3 ``$SCRATCH`` is not backed up and is subject to purge; consult TACC's
current storage policy before configuring alternate job storage.

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
   # job = fs.BaseJob.load("/path/to/job.json")

   site = fs.Site(
       queue="skx-dev",
       nodes=1,
       ranks_per_node=2,
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
   Run ``frequensolve site connect`` again. Passwords may be stored in a
   supported local OS keyring after successful authentication, but MFA codes
   are never stored.

``Invalid account`` or allocation failure
   Confirm the account in the TACC portal. Remove ``account`` when your login has
   only one default project, or set it to the exact project id when needed.

Unknown partition or rejected time/node request
   Run ``qlimits`` on Stampede3 and update the profile or per-run override. TACC
   policy can change after the packaged preset was released.

Solver not found or shared-library error
   Run ``frequensolve site check`` and confirm the absolute ``solver`` path. The
   PyPI package does not contain the solver.

Transfer failure
   Confirm that local ``rsync`` is installed and that ``work_dir`` is writable.

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
