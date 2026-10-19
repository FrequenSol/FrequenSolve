# Project identity foundation (opt-in, not Cloud-active)

This module is the first bounded slice of Cloud #2273 and SDK #303. It is not
a released change to Cloud project identity. Existing `Project`, upload,
planning, result and Trash behavior remains unchanged. Do not send the new ID
as an existing `project_key` or use it to construct S3 paths.

The local manifest contains only a version and a lowercase UUIDv4. It is stored
at `.frequensolve/project-identity.json` inside the project folder. Display names
and folder locations can change independently; neither is used to generate ID.

```python
from frequensolve.project_identity import ProjectIdentity

# Explicit opt-in only; the project folder must already exist.
identity = ProjectIdentity.new()
identity.save_new("./my-project")

# Moving the complete project folder preserves the manifest and ID.
same_identity = ProjectIdentity.load("./renamed-project")

# For a genuinely new study, create a new identity in an existing empty root.
fork = same_identity.fork_to("./new-study")

# Explicitly copying an existing identity means continuation, not a new project.
# This is local document import and does not establish Cloud ownership/access.
candidate = ProjectIdentity.from_dict(same_identity.to_dict())
candidate.save_new("./existing-project-copy")
```

`save_new` is create-only and never overwrites an existing manifest. Concurrent
creates have one winner. A conflict requires an explicit load or a new project;
it is not silently rebound to whichever identity was written first. Reads are
bounded to 4 KiB, reject duplicate/unknown fields and unsupported versions, and
do not regenerate missing/corrupt state. A crash can leave an invalid document;
fail closed and recover deliberately rather than assigning a replacement UUID.

The first file-operation pilot requires POSIX directory-descriptor/no-follow
support. It rejects symlinked final roots, metadata directories and manifest
files, and rejects nonregular manifests without waiting for FIFO data. Other
platforms fail with a clear unsupported-operation error. The pure manifest and
reference parsers require only the Python standard library on supported Python
versions. No site config or credentials are written.

The Cloud registration core must verify the authenticated account and user,
conditionally claim the ID and require expected revisions/idempotency receipts.
Knowing, editing or copying a UUID never grants access. The coordinated gate
remains off until uploads, saved plans, project summaries and deletion fences
all use the same authoritative binding. No historical storage or record mapping
is performed by these helpers.

`legacy_reference` and `reference_token` provide the shared tagged contract.
They do not normalize old keys or infer immutable identity from UUID-shaped
text. Exact structured contract vectors live under
`tests/contracts/project-identity-v1.json` and match Cloud's fixture byte-for-byte.
Normal `Project.copy` continues copying its existing authoring inputs and does
not copy/activate this manifest. Use the explicit identity methods only when
participating in the coordinated future pilot.


## Coordinated Cloud binding candidate

With the coordinated Cloud capability enabled for the authenticated participant,
`project.enable_identity()` creates and saves the manifest before the first
upload. `Project.load` retains it; a missing or conflicting expected manifest
fails closed. Moving the whole folder preserves the ID. `Project.copy` forks an
opt-in project's ID and rejects an existing identity at the destination before
copying files. Legacy projects keep their previous paths and behavior.

The AWS adapter binds the UUID before transferring any inputs, verifies the
server's exact storage prefix, and stages new objects under `pi_<uuid>`. Folder
names never change that binding. Unknown/disabled backends and foreign or
conflicting bindings fail rather than falling back to folder identity. Direct
submission and plan creation send the explicit ID. Saved-plan edits and reads
retain the ID and exact revision; mismatched response owners are rejected.

This candidate depends on the unmerged planning pilot and Cloud #2298. It is
not stable publication or general activation. It does not migrate old projects,
move retained objects, change billing or launch tests automatically.

Copying claims the destination before writing inputs and reserves a fresh UUID
before any immutable inputs are copied. A concurrent copy is rejected. A stopped
copy leaves its claim for explicit recovery; uploads refuse an unfinished copy.
After moving a project, use `Project.load(new_path).load_job(...)` so saved jobs
receive the relocated project context.
