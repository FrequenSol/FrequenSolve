Simulation assistant MCP
========================

The local simulation assistant lets any compliant MCP client use the knowledge
and validation rules shipped with the installed FrequenSolve package. It is
designed to help a user start a small supported simulation without guessing
package APIs or solver fields.

The server implements the stable ``2026-07-28`` Model Context Protocol with the
official MCP Python SDK 2.x. The local server uses standard ``stdio``. Reusable
hosted profiles use standard stateless Streamable HTTP with JSON responses by
default. It has no dependency on Codex, Claude, Cursor, or another AI vendor.
The official SDK also negotiates the earlier handshake-era protocol with older
compliant clients.

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

Capability profiles
-------------------

The package defines three explicit capability manifests. A manifest fixes the
exact tools, resources, prompts, schema identities, size and concurrency
limits, package version, and source revision for that profile. Adding a tool to
the package does not add it to any profile until its reviewed manifest is also
changed.

``local``
  Preserves the released local surface: seven setup/file tools, four optional
  self-scoped Cloud reads, eleven resources, and five prompts over ``stdio``.
  This is the default used by ``frequensolve-mcp serve`` and ``doctor``.

``public-onboarding``
  Provides six bounded setup tools, nine package-knowledge resources, and four
  prompts. Its input schemas accept only the fixed vetted scenario and
  in-memory drafts. It has no allowed-root, filesystem, Cloud, credential,
  account selector, external-reference, or network capability.

``authenticated-cloud``
  Provides only four customer Cloud tools, the identity and Cloud-read-contract
  resources, and the monitoring prompt. It requires an executor supplied by an
  authenticated host. It cannot reuse a local ``site.toml`` profile or cached
  Cognito credentials, and the package revalidates every host result against
  the packaged fixed read contract.

Read ``frequensolve://simulation-assistant/identity`` to inspect the active
machine-readable manifest. Hosted service authentication, authorization,
deployment, rate limits, audit logging, and environment promotion are owned by
the hosting application, not this package.

Install and check
-----------------

Install FrequenSolve with its optional MCP support in a long-lived virtual
environment. Replace the example absolute path once, then use that same path
when configuring your MCP client:

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

Before starting the MCP, establish a persistent cached login once in a normal
interactive terminal. The example below uses the ``cloud`` profile. If your
Cloud setup names a different profile, replace ``cloud`` with that exact name
in both the login and server arguments:

.. code-block:: console

   /absolute/path/to/frequensolve-mcp-venv/bin/python -c 'import frequensolve as fs; fs.Site(profile="cloud", interactive=True, force_login=True)'

Enter credentials only at those interactive prompts. If this is the first
FrequenSolve site command on the machine, it creates
``~/.frequensolve/site.toml`` and asks you to review it. Keep or add the
intended Cloud profile, then run the matching command again. The successful
site login stores refreshable Cognito state in the private FrequenSolve cache.
Before a later MCP session, the same command can refresh that state. Restart
the MCP after a refresh so it reads the current cache. ``force_login=True``
leaves the existing cached login untouched unless the new authentication
succeeds.

Configure any stdio MCP client
------------------------------

Choose an existing directory that may contain simulation JSON files and give
it a short safe root name. Configure a local ``stdio`` server with this command
and argument array:

.. code-block:: json

   {
     "mcpServers": {
       "frequensolve": {
         "command": "/absolute/path/to/frequensolve-mcp-venv/bin/frequensolve-mcp",
         "args": [
           "serve",
           "--allow-root",
           "project=/absolute/path/to/project-root",
           "--cloud-profile",
           "cloud"
         ]
       }
     }
   }

``mcpServers`` is a common client configuration shape, not a required MCP wire
field. If a client uses a settings form or different configuration keys, map
the same command and arguments to its local ``stdio`` server fields. Use the
absolute executable path because graphical clients may not inherit the shell's
``PATH``. No URL, port, API key, AI-vendor setting, or proprietary extension is
required.

The allowed root is optional. Without one, draft, validation, rendering,
preview, knowledge, and explanation tools still work; saved-artifact tools
have no files they are permitted to read.

The Cloud profile is also optional. When omitted, the default profile in
``site.toml`` is selected. If Cloud dependencies, configuration, or a cached
login are unavailable, the local setup tools continue to work and Cloud tools
return a short safe readiness error.

Protocol compatibility
----------------------

The server advertises tools, resources, and prompts with standard MCP
primitives only. A current client negotiates protocol ``2026-07-28`` through
``server/discover``. A handshake-era client negotiates the newest earlier
version supported by that client and the official SDK. The server does not use
sampling, roots, vendor extensions, or a host-specific transport.

Hosted transport embedding
--------------------------

Hosting code can build the public profile and obtain a self-contained ASGI
application for one exact MCP endpoint:

.. code-block:: python

   from frequensolve.mcp_server.server import build_server

   server = build_server(capability_profile="public-onboarding")
   application = server.create_streamable_http_app(
       path="/mcp",
       allowed_hosts=("mcp.sandbox.example",),
       allowed_origins=("https://approved-client.example",),
   )

The application runs the official SDK session manager in stateless mode,
rejects requests outside the exact path, enforces the package request-size
limit before parsing, and enables host/origin protection. Standard ASGI
lifespan is included. The default emits one JSON response per bounded request;
set ``json_response=False`` only when the host needs the official SDK's
request-scoped SSE behavior.

The authenticated profile must be given an async host executor:

.. code-block:: python

   async def execute_customer_read(operation, arguments):
       # The host derives identity from its verified request context and invokes
       # only the matching fixed, self-scoped read operation.
       return await customer_read_service.execute(operation, arguments)

   server = build_server(
       capability_profile="authenticated-cloud",
       hosted_cloud_executor=execute_customer_read,
   )

The executor receives only a reviewed operation name and bounded arguments. It
must derive the user and tenant from verified host authentication rather than
accepting identity selectors. Do not pass a bearer token, password, raw query,
AWS client, or environment endpoint through an MCP tool argument. The actual
host should wrap the ASGI application with its authentication, rate-limit,
audit, and observability middleware before deployment.

The official MCP Inspector can launch the same executable and arguments over
``stdio`` when troubleshooting client setup. The Inspector is a diagnostic
client; it is not required at runtime.

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

The Cloud and HPC handoff has a deliberate write boundary:

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
Configure Cloud, local, and generic SLURM profiles through
:ref:`site-configuration`. For Stampede3-specific account, allocation, MFA,
and solver setup, follow :doc:`stampede3_getting_started`.
