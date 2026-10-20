Agent scientific workflow (Sandbox pilot)
=========================================

This separate local MCP profile prepares and analyzes the curated 3D,
double-precision acoustic example at 2 and 20 Hz. The current pilot uses OpenAI
Codex over stdio. The existing simulation-assistant and read-only Cloud profiles
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

Configure Codex with an absolute executable, explicit profile, allowed root
and Sandbox product origin. Its MCP configuration uses TOML::

    [mcp_servers.frequensolve-workflow]
    command = "/absolute/environment/bin/frequensolve-workflow"
    args = ["serve", "--profile", "sandbox",
            "--allow-root", "pilot=/absolute/existing/pilot-directory",
            "--cloud-url", "http://127.0.0.1:5173"]
    required = true
    enabled_tools = ["workflow_discover", "workflow_create_project",
                     "workflow_plan", "workflow_results"]

The combined plan and result tools include preparation and local file writes,
so Codex can request tool approval even for a read action. For unattended use of
this restricted profile, explicitly approve only these four tools in the client
configuration; leave other servers and tools excluded. For example::

    [mcp_servers.frequensolve-workflow.tools.workflow_discover]
    approval_mode = "approve"
    [mcp_servers.frequensolve-workflow.tools.workflow_create_project]
    approval_mode = "approve"
    [mcp_servers.frequensolve-workflow.tools.workflow_plan]
    approval_mode = "approve"
    [mcp_servers.frequensolve-workflow.tools.workflow_results]
    approval_mode = "approve"

These client settings allow the adapter's bounded operations. They do not grant
paid-stage authority: planning, paid validation and execution still require the
human's separate Cloud review. See the official Codex MCP configuration at
https://learn.chatgpt.com/docs/extend/mcp?surface=cli.

Use one explicit root per working area; symlinks, traversal and replacement of
existing files are rejected. Selected local scientific data can be returned to the model
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
intent expresses new work. To resume paused input preparation, repeat ``create``
with the original intent; the adapter reconciles the exact saved request with
Cloud. Read calls remain observation only. Browser approvals retain their own identity through
reloads, including retries after a completed execution. Cancellation remains in
Cloud; it is not immediate termination or a refund.

Pilot acceptance includes reopening the same saved workflow in a fresh Codex
process and preserving its operation identities. The adapter remains a standard
stdio MCP server; other clients require separate acceptance before joining the
pilot.

Native review rehearsal
-----------------------

An optional fifth tool, ``workflow_review``, can display an authoritative
proposal and request structured input directly through an MCP client's form.
Start the server with ``--interactive-review`` and add ``workflow_review`` to
the client's enabled tools. Use an interactive client with MCP elicitation
enabled, a human approval reviewer and a tool timeout of at least 240 seconds.
Keep other servers, shell, browser and app authorities excluded during acceptance.
The default server still exposes only the original four tools.

This tool is a **no-spend rehearsal**. Its response never authorizes planning,
validation or execution. Ordinary workflow grants continue to reject paid
actions. A general tool-permission setting does not substitute for stage approval.

Call ``workflow_review`` with ``action="preview"``, the allowed ``root_id``,
``local_project`` and existing ``plan_id``. Present the returned scientific
assumptions, input identity, target, resource bounds/revision, quote and charging
implications in the conversation. Then call ``action="request_input"`` with
the returned ``expected_fingerprint`` and a durable local ``intent``. The tool
itself sends the complete proposal through the MCP form; tool arguments cannot
contain an approval decision. The form allows cancellation, a resource-change
request, or confirmation of this rehearsal. Requested changes are returned as
input and are not automatically applied.

The tool reads the authoritative proposal before and after input. Material
changes invalidate the response. Timeout, lost connection, malformed input or
missing form support leave incomplete evidence and a Cloud review link. Reuse
the same intent after uncertainty, or call ``action="recover"`` to read its
original local evidence. A repeated intent never silently opens another prompt.
An interrupted pending prompt is not acceptance; a deliberate new rehearsal
requires a new intent. Local evidence is bound to the explicit connection,
principal, project and exact proposal and contains no transfer credentials.

Receipts explicitly report ``paid_authority=false``,
``backend_authorized=false`` and ``human_approval_verified=false``. Even a
successful client confirmation has origin ``client_reported_unattested``.
MCP clients can answer forms programmatically. Transport tests or an accepting
test harness therefore do not prove that a human saw or approved the request.
Qualify the exact client, display and policy before enabling native paid
authority; retain Cloud review when the client cannot be trusted.
The existing paid receipts, idempotency and dispatch machinery remain authoritative.

Codex's documented elicitation request is described at
https://learn.chatgpt.com/docs/app-server#mcp-server-elicitation-requests.

Native paid-stage approval
--------------------------

Native paid approval is a separate opt-in capability. It trusts the local MCP
client and adapter to report the user's form response. FrequenSol cannot
independently attest that a human answered: anyone possessing the new connection's
credentials can exercise its paid permission. The one-time connection consent
discloses this change of authority. Existing read, preparation and result grants
are never upgraded. Keep those profiles when this trust boundary is unsuitable.

Use the dedicated Sandbox native-approval client registration and a **new**
profile name. The normal preparation client cannot request the additional scope::

    frequensolve-workflow login --profile sandbox-native \
      --issuer https://your-sandbox-gateway \
      --client-id YOUR_NATIVE_APPROVAL_CLIENT_ID --native-approval

This login requests ``workflow-approve`` in addition to the existing workflow
scopes. Review and explicitly allow client paid approvals on the signed-in
connection page. No stage is authorized by connecting. To expose
``workflow_approve``, run ``serve --profile sandbox-native --native-approval``
with the usual allowed root and Cloud origin, and add only that tool to the
client's enabled tool list. Keep MCP elicitation enabled with a human reviewer;
exclude shell, browser and other authorities during acceptance. Ordinary
``serve`` still exposes the unchanged four-tool profile.

Call ``workflow_approve`` with ``action="preview"``, ``root_id``,
``local_project``, ``plan_id`` and one ``stage``: ``planning``, ``validation`` or
``execution``. The service returns the complete authoritative proposal and its
opaque ``proposal_fingerprint``. Call ``action="request_input"`` with that
fingerprint as ``expected_fingerprint`` and a deliberate stable ``intent``.
The client form contains the full proposal. To approve, the user chooses
``approve_paid_stage`` and enters its exact ``maximum_credits``. These decisions
cannot be supplied as tool arguments. Cancellation, incomplete responses and
change requests send no paid mutation. Changes are returned for review, never
automatically applied. A rehearsal confirmation cannot be converted into paid
approval.

Each stage needs its own proposal and form; ``run_after_planning`` remains false.
Backend-generated fingerprints bind scientific inputs, revision, target,
resources, quote and Credit maximum. Before a new paid request, the service
checks the current proposal and appends immutable client attribution linked to
the existing domain receipt. That record is neither an approval token nor a
dispatch state machine. Existing services still enforce ownership, admission,
current pricing/context, reservations, idempotency and settlement.

The adapter saves evidence before displaying the form and the exact paid request
and idempotency key before transport. ``action="recover"`` reuses an already
confirmed request through restart or an uncertain response, including after
settlement. It does not open another form or allocate a new execution identity.
Only an explicit new run uses a new intent and receives a fresh form. Interrupted
or canceled prompts cannot be promoted into confirmation. Local evidence stays
bound to the explicit connection, user, project and proposal.

An accepted server response records ``backend_authorized=true`` separately from
``human_approval_verified=false`` and ``evidence_origin=native_client_reported``.
Authorization does not establish solver success, downloadable results or settled
charges. Revoking the connection prevents new requests; Cloud review and
cancellation remain available independently.

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
