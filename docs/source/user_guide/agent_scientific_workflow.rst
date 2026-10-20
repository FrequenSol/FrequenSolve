Agent scientific workflow (Sandbox pilot)
=========================================

This separate local MCP profile prepares and analyzes the curated 3D,
double-precision acoustic example at 2 and 20 Hz. It serves Codex and Claude
Code over stdio. The existing simulation-assistant and read-only Cloud profiles
keep their current grants and behavior. Pilot availability requires a separately
deployed Sandbox gateway and an eligible participant; installing this package
does not establish live acceptance.

Install the candidate with the ``mcp`` and ``notebook`` optional extras in an
explicit environment. Use the issuer and dedicated workflow client ID supplied
by the Sandbox operator::

    frequensolve-workflow login --profile sandbox --issuer https://YOUR-ISSUER --client-id YOUR-WORKFLOW-CLIENT
    frequensolve-workflow status --profile sandbox

Login opens a browser for signed-in consent. This adapter has a private token
cache under ``~/.frequensolve/workflow`` and never reads ordinary SDK Cognito or
AWS credentials. Concurrent clients share refresh safely. An uncertain refresh
requires login again; it never falls back to another credential source.
``frequensolve-workflow logout --profile sandbox`` revokes the adapter tokens.
Cloud Settings can disconnect grants. New transfer tickets stop after
revocation; previously issued tickets expire after 60 seconds.

Configure both clients with the same absolute executable, explicit profile,
allowed root and Sandbox product origin. A generic stdio configuration is::

    {
      "mcpServers": {
        "frequensolve-workflow": {
          "command": "/absolute/environment/bin/frequensolve-workflow",
          "args": ["serve", "--profile", "sandbox",
                   "--allow-root", "pilot=/absolute/existing/pilot-directory",
                   "--cloud-url", "http://127.0.0.1:5173"]
        }
      }
    }

Apply this entry to the client's supported MCP configuration format. Use one
explicit root per working area; symlinks, traversal and replacement of existing
files are rejected. Selected local scientific data can be returned to the model
provider chosen in that client. The adapter's restricted scope does not constrain
an agent separately allowed to read broad credentials or control a signed-in
browser. Acceptance must exclude those alternative authorities.

Workflow and recovery
---------------------

Read ``frequensolve://workflow/capabilities`` for the versioned manifest. The four
tools are ``workflow_discover``, ``workflow_create_project``, ``workflow_plan``
and ``workflow_results``. Tool arguments use a ``request`` object. Replies have
``ok`` and a JSON ``payload`` string.

Discover the eligible cluster, then create a fresh local project. Names are
metadata; each project receives a new immutable identity. Preparation serializes
only the fixed scientific template and its dependencies. Planning creation needs
the discovered ``cluster_id``, local project and a durable ``intent`` name.
Creation obtains a quote and never authorizes paid work. Follow its Cloud review
link, where the human approves paid planning. Read the saved plan, revise exact
resources with a new intent, validate, and obtain a separate approval if paid
validation is required. Quote and approve the exact execution revision separately.
The backend decides whether existing qualification evidence suffices.

Every mutation saves the exact request, expected revision and idempotency key
before sending. On timeout or restart, reuse the same project and intent. A new
intent expresses new work. Browser approvals retain their own identity through
reloads, including retries after a completed execution. Cancellation remains in
Cloud; it is not immediate termination or a refund.

Observation never resumes remembered authorization. Interpret scientific
outcome, output publication and financial settlement independently. Discovery
reports scheduler/configuration observations separately from capacity; free/busy
capacity is unavailable. Credit history scans bounded pages, merges immutable
operation IDs, and withholds its total until the scan is complete. Quotes,
authorized maxima, rated usage and captured Credits are distinct. Exposure is
explicitly unavailable in this historical projection, never a Chargebee hold.
Chargebee remains authoritative. Refresh after settlement to obtain current
complete history; balance differences cannot establish workflow cost.

Analysis and notebooks
----------------------

Select exact execution and frequency identities. Bounded downloads verify
checksums and retain provenance without exposing transfer URLs. Analysis uses
native coordinate pressure in Pa and receiver/source coordinates in metres;
producer metadata is retained. It does not apply the solver-to-coordinate
conversion a second time. Compatible references must match source and receiver
identities/coordinates, units, frequency/Laplace coordinates, scaling and encoding.
Missing/nonfinite values and coverage remain explicit. Zero-magnitude phase is
undefined. Comparisons use complex differences and RMS, with no interpolation,
rescaling, phase unwrapping or time-domain reconstruction.

The generated directory contains editable ``analysis.ipynb``, verified input
snapshots, ``executed.ipynb``, ``replay.json`` and static JSON/HTML reports. Run All
contains analysis only and needs no live Cloud credential. The official replay
uses a fresh kernel, cleared outputs, an explicit interpreter and finite timeout,
and records incomplete evidence on failure. Edited notebooks remain ordinary
Jupyter documents; generate a fresh official bundle for new checked evidence.
Use ``frequensolve.workflow.notebooks.verify_replay(path)`` to check whether
retained evidence still applies after code, input, output or environment changes.
It preserves the historical receipt and reports current mismatches.
Local replay, verified input identity, service origin, operation version,
convergence/qualification and agent interpretation are separate assurances.
Notebook trust does not establish scientific correctness. See NBClient's
execution model at https://nbclient.readthedocs.io/en/latest/client.html and
notebook signatures at https://nbformat.readthedocs.io/en/latest/api.html#notebook-signatures.
