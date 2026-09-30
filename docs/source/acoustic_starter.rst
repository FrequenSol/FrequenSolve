Your first Cloud simulation
===========================

Follow one small acoustic model from preparation to a result you can inspect.
This is tutorial ``acoustic-starter-v1``, the same fixed 10 Hz scenario prepared
by the :doc:`simulation assistant <user_guide/simulation_assistant_mcp>`.
You can prepare and validate it without a Cloud account or a solver. Submitting
is a separate, explicit action and consumes Credits for work performed.

1. Install and download
-----------------------

Use Python 3.10–3.14 in a virtual environment on Linux or macOS. On Windows,
run these commands and your notebook kernel inside WSL2; native Windows is
unsupported. Keep the workspace inside the Linux filesystem.

.. code-block:: console

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install "frequensolve[cloud,visual]"
   python -c 'import frequensolve as fs; print(fs.__version__)'

Download :download:`acoustic_starter.py
<../../tutorials/00_getting_started/acoustic_starter.py>` and, if you prefer
cells, :download:`acoustic_starter.ipynb
<../../tutorials/00_getting_started/acoustic_starter.ipynb>` into the same folder.
Select this virtual environment as your notebook kernel. The script's
``--help`` lists the available actions. These instructions belong to the
package version shown at the top of this documentation; check that your
installed version contains ``frequensolve site check`` before continuing.

2. Prepare and inspect locally
------------------------------

.. code-block:: console

   python acoustic_starter.py

This saves a project, simulation and job under ``acoustic-starter-v1/`` and
validates their authoring settings. It does not log in, upload or run a solver.
An existing nonempty directory is preserved: choose another with
``--directory ./my-starter`` when preparing a separate example.

The model is 1 km wide and 0.5 km deep, with an interface at 0.25 km. The upper
and lower P-wave velocities are 2.0 and 2.8 km/s; densities are 2.2 and
2.4 g/cm³. A scalar source sits at (0.5, 0.025) km. There are 101 pressure
receivers at 0.05 km depth, a pressure-free top, and absorbing sides/bottom.
The fixed mesh is evaluated for this starter; changing its frequency is a new
modeling decision. One 10 Hz frequency creates one solver task and requests
receiver responses plus a domain pressure VTK output.

3. Check access and configure Cloud
----------------------------------

Sign in to Cloud, open **Compute**, and copy the Python configuration for your
ready shared profile into ``~/.frequensolve/site.toml``. Use the host and profile
name shown in your app; do not copy a host from someone else's example. The
local profile name (``cloud`` below) and Cloud compute-profile name are separate.
Keep passwords and tokens out of TOML and notebooks.

.. code-block:: console

   frequensolve site check --local --profile cloud

This validates local TOML and selection settings without creating a config,
authenticating or contacting Cloud. Success does not prove your subscription,
seat, balance, storage or compute is ready. Check those in Cloud first. If your
package does not recognize the command or ``compute_profile``, install the
compatible version linked by your Cloud guide; do not silently remove the
selector and run elsewhere.

4. Submit deliberately
----------------------

.. code-block:: console

   python acoustic_starter.py --submit --profile cloud

This loads the prepared inputs, prompts for normal Cloud sign-in if needed,
and submits the job. It prints a run ID before waiting. Keep that ID with the
inputs; open **Projects** in Cloud to follow preparation, queueing, solving and
results. Queueing is not a failure. Do not repeat submission because a page is
slow to refresh. The browser shows the status of accepted work; closing a tab
does not cancel it. Use the run's explicit cancel control when needed.

No fixed runtime or Credit price is promised: queue time and performed work
vary. Use the run's usage record after completion for the actual charge.

5. Read and interpret the result
-------------------------------

A successful run prints receiver metadata and the domain VTK filenames. In
Cloud open the run's **Results** and inspect the pressure field, receiver
positions and layer interface. In the notebook, plot the magnitude of the
frequency-domain pressure along receiver position; phase gives additional
information about the response. These are complex amplitudes at 10 Hz, not a
time-domain seismogram. A successful/converged solve does not establish that a
model represents your experiment.

Retain the project, simulation, job, run ID and installed package version.
Use :doc:`user_guide/site_configuration` for attaching to a specific existing
run and retrieving its results; fetching an authored job's latest run is not
an instruction to submit again. Download logs with results when investigating
a failure. Preserve complete outputs before attempting a partial retry.

If something blocks you
-----------------------

- **Package or optional module missing:** activate the same virtual environment
  used by the notebook and install ``cloud,visual``; MCP needs ``mcp`` separately.
- **No profile or invalid TOML:** copy configuration from **Compute**, then run
  ``site check`` again. Never paste secrets into an AI conversation.
- **Sign-in or access denied:** use Cloud password recovery and check your seat
  and subscription. If blocked, open **Support**; running a job is not required.
- **Compute unavailable or insufficient Credits:** follow the action shown by
  Cloud; a local config check cannot change availability or billing.
- **Failed run or incomplete results:** open that run's diagnostics and retain
  its ID. Read :doc:`user_guide/site_configuration` before retrying. Each new run
  can perform additional billable work.

Continue with :doc:`quickstart` for time-domain and richer output examples or
:doc:`tutorials/index` for the deeper modeling learning path.
