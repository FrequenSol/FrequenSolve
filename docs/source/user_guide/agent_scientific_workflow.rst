Agent scientific workflow (Sandbox pilot)
=========================================

This separate local MCP profile prepares and analyzes the curated 3D,
double-precision acoustic example at 2 and 20 Hz. The current pilot uses OpenAI
Codex over stdio. The existing simulation-assistant and read-only Cloud profiles
keep their current grants and behavior. Pilot availability requires a separately
deployed Sandbox gateway and an eligible participant; installing this package
does not establish live acceptance.

The study unfolds in your MCP client
------------------------------------

Once connected, start with a scientific question, for example:

    Show me the acoustic example, explain what we will learn at 2 and 20 Hz,
    and walk me through the model, cost and results here.

The assistant should show the specified source/receiver geometry, explain the
assumptions and wavelength estimates, then present the resource proposal. You
can ask questions or request a resource change in ordinary language. The
assistant manages saved identities and recovery; you do not need to copy plan
IDs, fingerprints or commands between sessions.

For a connection with native paid approval enabled, each paid stage uses the
client's approval form and a separate exact Credit-limit confirmation. With an
ordinary connection, only paid approval requires the signed-in Cloud review.
Follow progress, captured Credits, receiver plots and reference comparisons in
the same conversation. Finish by asking for an editable analysis notebook.
Cloud remains an optional place to inspect the saved study.

Returning to a study
~~~~~~~~~~~~~~~~~~~~

Ask, for example, "Reopen my acoustic study and show only the 20 Hz results."
The assistant uses ``workflow_workspace`` to find named projects in your allowed
workspace, then lists their owned saved plans. If names are ambiguous, it asks
which one you mean. It does not create another project or run to recover old
work. Local metadata is a navigation hint; Cloud still checks access on reads.
Folder browsing is explicit, does not follow symlinks and is limited to 1,000
entries per folder, with 25 folders/projects per page. Contents can change
between pages, so clients should deduplicate by identity and local path.

After generating a notebook, ask "Open my notebook." The ``reproduce`` chapter
reopens its clickable paths when its saved files and exact study, project, plan
revision, run, input/result fingerprints and selected frequencies match. A fresh
adapter process can recover the same bookmark without regenerating the notebook
or running a kernel. The chapter reads current owned result metadata; it does
not request transfer tickets or download contents.

A notebook for another run, question or frequency selection is identified as
different. Older bookmarks without an exact binding remain recoverable through
the workspace tool, but are not attached to the current study as verified
notebooks. An incomplete replay offers the editable notebook for inspection,
without presenting its outputs as successful evidence. Changed or missing files
withhold the links. If current metadata cannot be confirmed, retry the same read;
uncertainty does not call for regeneration or replacement solver work.

The workspace tool can also recover the project's latest notebook independently
of a selected study. That view checks saved bytes and reports its recorded
context, without claiming it matches the current study. Unchanged bytes do not
establish that the current environment matches, and a local replay is not service
attestation. Earlier, unbookmarked notebooks remain on disk. Editable cells
inside the client require separate host support; an ``ipynb`` link alone does not
establish that capability.

``workflow_present`` supplies one focused chapter at a time: ``overview``,
``model`` (including estimates), ``plan``, ``progress``, ``results`` or
``reproduce``. It returns ordinary MCP text and PNG image blocks, plus structured
provenance. Coordinate/numeric tables provide a text alternative. The client
controls the final rendering: these are not embedded interactive widgets, and
an image-capable desktop client and a text terminal need not look identical.
Custom MCP Apps components require separate client support and acceptance.

The presentation reads existing owned state and cannot authorize compute.
Saved analysis must match the current input, plan revision and execution.
Pressure magnitude and phase are recalculated from its recorded complex values;
the view labels the local origin and does not claim that input/reference files
were reverified or that a notebook has replayed successfully. Request verified
result retrieval and notebook replay for that evidence. Interpretation remains
separate from measured results and scientific qualification.

Operator setup for the Sandbox pilot
------------------------------------

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

Generate the exact configuration for your installed environment::

    frequensolve-workflow codex-config --profile sandbox \
      --allow-root studies=/absolute/existing/studies \
      --cloud-url https://YOUR-SANDBOX-CLOUD-ORIGIN \
      --output /absolute/new/frequensolve-codex.toml

This writes a new file and preserves all existing client settings. Add its MCP
server section through your client's supported configuration workflow. Keep the
configuration private: it names your local working area but contains no tokens.
The generator excludes native paid approval and rehearsal tools unless explicitly
requested. Only the three observation tools are pre-approved at the client tool
layer. This setting grants no paid authority.

The generated configuration uses an absolute Python executable and module
entrypoint. An equivalent configuration with the installed console script is::

    [mcp_servers.frequensolve-workflow]
    command = "/absolute/environment/bin/frequensolve-workflow"
    args = ["serve", "--profile", "sandbox",
            "--allow-root", "pilot=/absolute/existing/pilot-directory",
            "--cloud-url", "http://127.0.0.1:5173"]
    required = true
    enabled_tools = ["workflow_discover", "workflow_create_project",
                     "workflow_plan", "workflow_results", "workflow_study",
                     "workflow_present", "workflow_workspace"]

The combined plan and result tools include preparation and local file writes,
so Codex can request tool approval even for a read action. For unattended use of
this restricted profile, explicitly approve only the intended tools in the client
configuration; leave other servers and tools excluded. For example::

    [mcp_servers.frequensolve-workflow.tools.workflow_discover]
    approval_mode = "approve"
    [mcp_servers.frequensolve-workflow.tools.workflow_create_project]
    approval_mode = "approve"
    [mcp_servers.frequensolve-workflow.tools.workflow_plan]
    approval_mode = "approve"
    [mcp_servers.frequensolve-workflow.tools.workflow_results]
    approval_mode = "approve"
    [mcp_servers.frequensolve-workflow.tools.workflow_study]
    approval_mode = "approve"
    [mcp_servers.frequensolve-workflow.tools.workflow_present]
    approval_mode = "approve"
    [mcp_servers.frequensolve-workflow.tools.workflow_workspace]
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

Read ``frequensolve://workflow/capabilities`` for the versioned manifest. The core
tools are ``workflow_discover``, ``workflow_create_project``, ``workflow_plan``
and ``workflow_results``; ``workflow_study`` persists the scientific narrative
and generates its notebook, and ``workflow_present`` presents it in the client.
Tool arguments use a ``request`` object. Replies have ``ok`` and a JSON
``payload`` string. Presentation replies also contain readable narrative and
ordinary MCP image blocks; they do not duplicate image bytes in structured JSON.

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

An optional tool, ``workflow_review``, can display an authoritative
proposal and request structured input directly through an MCP client's form.
Start the server with ``--interactive-review`` and add ``workflow_review`` to
the client's enabled tools. Use an interactive client with MCP elicitation
enabled, a human approval reviewer and a tool timeout of at least 240 seconds.
Keep other servers, shell, browser and app authorities excluded during acceptance.
The default server excludes both approval tools.

This tool is a **no-spend rehearsal**. Its response never authorizes planning,
validation or execution. Ordinary workflow grants continue to reject paid
actions. A general tool-permission setting does not substitute for stage approval.

Call ``workflow_review`` with ``action="preview"``, the allowed ``root_id``,
``local_project`` and existing ``plan_id``. Present the returned scientific
assumptions, input identity, target, resource bounds/revision, quote and charging
implications in the conversation. Then call ``action="request_input"`` with
the returned ``expected_fingerprint`` and a durable local ``intent``. The tool
itself sends a readable scientific/resource/Credit summary followed by every
exact proposal setting through the MCP form; tool arguments cannot
contain an approval decision. The form offers ``Cancel``, ``Request changes`` and ``Confirm rehearsal``.
Only a change request opens a note prompt; cancellation and confirmation do not
ask for an optional note. Existing protocol labels remain accepted. Requested
changes are returned as input and are not automatically applied.

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
``serve`` excludes paid approval.

Call ``workflow_approve`` with ``action="preview"``, ``root_id``,
``local_project``, ``plan_id`` and one ``stage``: ``planning``, ``validation`` or
``execution``. The service returns the complete authoritative proposal and its
opaque ``proposal_fingerprint``. Call ``action="request_input"`` with that
fingerprint as ``expected_fingerprint`` and a deliberate stable ``intent``.
The client form leads with the exact stage and Credit maximum, scientific
assumptions, resource rows and changes from automatic sizing. Charging and
cancellation terms follow, including the quoted minimum and failure policy.
Every authoritative setting, including unknown fields, is then shown in exact
path/value tables. Missing values remain explicit. Oversized or unsupported
reviews fail safely rather than silently truncating settings. The underlying
proposal and its fingerprints are unchanged.

To approve, the user chooses
``Approve this stage``, then enters its exact Credit maximum in a separate
required prompt. Approval does not ask for an optional note. Choosing
``Request changes`` instead opens a required change-description prompt. These decisions
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

A failed discovery, saved-plan/result listing or Credit-history read reports
``failure_code="read_unconfirmed"`` and an unknown observation outcome. Check the
explicit connection, saved project identity or result availability, then retry
that same read with the same saved IDs. The read cannot authorize or resume paid
work, and failure does not establish that an operation or result is absent.
Preparation, quotation, transfer-ticket and mutation requests remain outside
this observation retry path; preserve their existing request/intent identities
when their outcomes are uncertain.

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
and records incomplete evidence on failure. The MCP operation reserves the new
destination and retains its notebook, inputs and initial ``attempt.json`` before
starting a kernel. That initial record remains incomplete; a complete
``replay.json`` and current identity verification establish local replay success.
Failed replay returns an explicit incomplete status and evidence path, preserves
available execution evidence, and publishes no successful report. Preserve the
failed attempt before choosing a new destination; repeating its path cannot
overwrite evidence or start another kernel. Edited notebooks remain ordinary
Jupyter documents; generate a fresh official bundle for new checked evidence.
Use ``frequensolve.workflow.notebooks.verify_replay(path)`` to check whether
retained evidence still applies after code, input, output or environment changes.
It preserves the historical receipt and reports current mismatches.
Local replay, verified input identity, service origin, operation version,
convergence/qualification and agent interpretation are separate assurances.
Notebook trust does not establish scientific correctness. See NBClient's
execution model at https://nbclient.readthedocs.io/en/latest/client.html and
notebook signatures at https://nbformat.readthedocs.io/en/latest/api.html#notebook-signatures.
