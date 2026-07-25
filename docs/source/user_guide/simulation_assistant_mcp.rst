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

Install FrequenSolve with its optional MCP support:

.. code-block:: console

   python -m pip install "frequensolve[mcp]"
   frequensolve-mcp doctor

The doctor uses an in-memory connection. It does not contact the network,
submit a simulation, or require credentials.

Add it to Codex
---------------

Choose an existing directory that may contain simulation JSON files. Register
that directory with a short safe name:

.. code-block:: console

   codex mcp add frequensolve -- frequensolve-mcp serve \
     --allow-root project=/absolute/path/to/project-root
   codex mcp list

Restart Codex after adding the server. In the Codex desktop app, the same
server can be added under **Settings > MCP servers > Add server** as a
``STDIO`` server. Use ``frequensolve-mcp`` as the command and enter
``serve --allow-root project=/absolute/path/to/project-root`` as its arguments.

The allowed root is optional. Without one, draft, validation, rendering,
preview, knowledge, and explanation tools still work; saved-artifact tools
have no files they are permitted to read.

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
