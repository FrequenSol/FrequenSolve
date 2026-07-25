Simulation assistant MCP
========================

The private-beta simulation assistant lets an MCP-capable agent use the
knowledge and validation rules shipped with the installed FrequenSolve
package. It is designed to help a user start a small supported simulation
without guessing package APIs or solver fields.

The local server can:

- find a vetted package example;
- create the known-small 2D acoustic draft;
- validate a draft or supported saved JSON artifact;
- render deterministic starter Python without executing it;
- inspect a supported artifact;
- preview frequencies, task count, assumptions, and expected outputs; and
- explain stable validation codes in plain language.

The first-beta generated starter is fixed at the catalog's evaluated 10 Hz
setup. The agent may change safe names and receiver count, but it must not
claim that another frequency is supported by this fixed starter mesh.

It cannot submit, run, upload, change, or delete a simulation. It does not
provide raw GraphQL access.

Install and check
-----------------

Install FrequenSolve with its optional MCP support in a long-lived virtual
environment. Replace the example absolute path once, then use that same path
when configuring Codex:

.. code-block:: console

   python -m venv /absolute/path/to/frequensolve-mcp-venv
   /absolute/path/to/frequensolve-mcp-venv/bin/python -m pip install "frequensolve[mcp]"
   /absolute/path/to/frequensolve-mcp-venv/bin/frequensolve-mcp doctor

The doctor uses an in-memory connection. It does not contact the network,
submit a simulation, or require credentials.

To include the optional read-only Cloud tools, install both extras:

.. code-block:: console

   /absolute/path/to/frequensolve-mcp-venv/bin/python -m pip install "frequensolve[mcp,cloud]"

To use the generated starter with a configured SLURM/HPC site, install the MCP
and HPC extras:

.. code-block:: console

   /absolute/path/to/frequensolve-mcp-venv/bin/python -m pip install "frequensolve[mcp,hpc]"

Cloud reads reuse an existing ``aws`` or ``cloud`` profile from
``~/.frequensolve/site.toml`` and that profile's cached Cognito login. The MCP
never accepts a password, token, account ID, or user ID.

Before starting the MCP, create or refresh the cached login once in a normal
interactive terminal. The examples below use the supplied ``cloud`` profile.
If your private-beta invitation names a different profile, replace ``cloud``
with that exact name in both the login and Codex commands:

.. code-block:: console

   /absolute/path/to/frequensolve-mcp-venv/bin/python -c 'import frequensolve as fs; fs.Site(profile="cloud", interactive=True, force_login=True)'

Enter credentials only at those interactive prompts. If this is the first
FrequenSolve site command on the machine, it creates
``~/.frequensolve/site.toml`` and asks you to review it. Keep the supplied
``cloud`` profile or add the profile from your private-beta invitation, then
run the matching command again. Restart the MCP after any later re-login so it
reads the refreshed cache. ``force_login=True`` leaves the current cached login
untouched unless the new authentication succeeds.

Add it to Codex
---------------

Choose an existing directory that may contain simulation JSON files. Register
that directory with a short safe name:

.. code-block:: console

   codex mcp add frequensolve -- \
     /absolute/path/to/frequensolve-mcp-venv/bin/frequensolve-mcp serve \
     --allow-root project=/absolute/path/to/project-root \
     --cloud-profile cloud
   codex mcp list

Restart Codex after adding the server. In the Codex desktop app, the same
server can be added under **Settings > MCP servers > Add server** as a
``STDIO`` server. Use
``/absolute/path/to/frequensolve-mcp-venv/bin/frequensolve-mcp`` as the command
and enter ``serve --allow-root project=/absolute/path/to/project-root
--cloud-profile cloud`` as its arguments. The absolute executable path keeps
the server working when Codex does not inherit the shell's ``PATH``.

The allowed root is optional. Without one, draft, validation, rendering,
preview, knowledge, and explanation tools still work; saved-artifact tools
have no files they are permitted to read.

The Cloud profile is also optional. When omitted, the default profile in
``site.toml`` is selected. If Cloud dependencies, configuration, or a cached
login are unavailable, the local setup tools continue to work and Cloud tools
return a short safe readiness error.

Read-only Cloud tools
---------------------

Four Cloud tools help an agent monitor the signed-in user's work:

- check seat, subscription, credit, storage, and compute readiness;
- list a bounded page of the user's simulations;
- read one simulation summary or its bounded stored diagnostics; and
- list relative result-artifact metadata without downloading file contents.

These tools use five fixed versioned queries owned by the Cloud service. They
do not accept raw GraphQL, tenant selectors, credentials, bucket names, or
object keys. They cannot submit, cancel, upload, download, change, or delete
anything. Missing and unauthorized simulation IDs receive the same safe
response. The ``cloud_*`` monitoring tools are Cloud-only; they do not monitor
jobs submitted to SLURM/HPC sites.

Safe file access
----------------

Saved-artifact tools accept only an allowed-root name plus a relative JSON
path. Absolute paths, parent traversal, symlinks, oversized JSON, unsupported
artifact types, and references to external data files are rejected. Configure
the narrowest directory the agent needs. Do not expose a home directory,
credential directory, or filesystem root.

The first beta accepts inline ``SeismicSimulation``, ``FrequencyDomainJob``,
and ``TimeDomainJob`` JSON. Compact grid-backed coordinates, gridded
properties, and wavefield artifacts are rejected because loading them can
expand a small JSON description into large in-memory arrays.

First conversation
------------------

Ask the agent:

.. code-block:: text

   Help me create and validate the known-small 2D acoustic FrequenSolve
   simulation. Show me the preview and starter Python, but do not run it.

The agent should use the server's version-matched resources, create a bounded
draft, validate it with the installed package, and show the deterministic
preview and starter code. Review the result before using it in a real project.

Prepare, submit, and monitor
----------------------------

The private-beta handoff has a deliberate write boundary:

1. Let the MCP prepare, validate, preview, and render the starter.
2. Review the generated Python and submit it separately outside the MCP with
   ``fs.Site(profile='cloud')`` or ``fs.Site(profile='hpc')``. The generated
   draft and rendered script are execution-site-neutral; the MCP cannot perform
   this write.
3. After the simulation exists, ask the agent to check readiness, list your
   simulations, read the owned simulation, and explain its bounded diagnostics
   or result metadata when it was submitted to Cloud.

This keeps setup assistance and read-only monitoring available to the agent
without giving the MCP submission, cancellation, upload, or delete authority.
