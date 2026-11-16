# Independent frequency execution plans

`frequency-execution-plan.v1` is the shared SDK, Cloud and Runtime contract.
All allocations use safe integer CPUs, MiB, and seconds. Frequencies are decimal
strings so cross-language content hashes do not depend on floating-point JSON
formatting. Fingerprints use sorted JSON keys and UTF-8 without ASCII escaping.

A revision binds the input snapshot, generated artifacts, native solver and
runtime identities, cluster capability fingerprint and sizing policy. Each row
contains automatic and requested allocations separately. MPI ranks are uniform
per node; total CPUs and total memory are derived rather than editable fields.
Layout sizing evidence is supplied by authenticated compute operations. Clients
may edit requests but cannot supply authoritative evidence or validation status.

The memory-fit-first policy selects qualified, evaluated layouts, preferring
fewer feasible nodes then fewer ranks. Qualified profiles carry CPU-thread
recommendations; an observation for a different thread/rank layout cannot be
reused. Rank-local resident estimates are preserved. A conservative node bound
uses the largest `ranks_per_node` estimates, never aggregate memory divided by
rank count. Runtime overhead is additional; native safety is already applied.

Edits create immutable revisions. Layout edits require matching sizing evidence
or a quoted compute-validation operation. Reservation increases can reuse sizing
evidence; reductions below its requirement fail validation. No frequency jobs
may launch until every row of the exact revision is validated. A submitted
revision never changes. Historical explicit topology profiles retain their
meaning; automatic capacity ceilings are distinct from exact-layout overrides.

The example files are synthetic conformance vectors. Their copies in the Cloud
and Runtime test fixtures must remain byte-identical when the contract changes.

The context freezes `memory_policy`: runtime calibration numerator/denominator,
worker overhead in MiB and the minimum memory reservation. Native safety is
already applied before these separately recorded allowances. Calibration cannot
reduce native estimates. Match the installed policy, including its minimum;
never silently substitute new defaults when replaying a revision.

Automatic selection uses the hardware catalog's `recommended_layouts` subset of
`qualified_layouts`. Other qualified thread layouts remain available for explicit
validated overrides. The initial Slurm integration validates whole-minute time
limits so a scheduler rounding step cannot exceed an authorization unnoticed.

Input digests may bind a service-owned manifest of immutable provider object
versions; this must not be advertised as a measured SHA-256 of each source file.
Large manifests and native evidence stay in tenant-scoped service storage.


### Controller progress diagnostics

`frequency-plan-progress.v1` (`progress.json`) is a separate bounded diagnostic
notification for one authenticated planning operation or exact execution. It
contains the accepted submission digest, monotonically increasing durable
sequence, UTC observation time, stage, total and complete counts for `ready`,
`intent`, `uncertain`, `submitted`, `completed`, `failed`, `canceled` and
`not_started`. Totals are 1–8 planning evaluations or 1–96 frequency jobs.
The `digest` is SHA-256 over canonical JSON with `digest` omitted.
Only the installed Root producer may publish it through the registered result
queue. It is not sizing evidence, actual allocation telemetry or billing proof.
A submitted job is waiting for compute or results, not necessarily running.

Receivers bind the accepted operation, tenant, plan, registration, physical
controller and AWS sender before storage. Duplicate or older sequence numbers
do not replace newer progress; the same sequence with different content is an
integrity error. A later revision/run or settled operation ignores old progress.
This record never mutates a plan revision, starts/retries work, captures/releases
Credits or acknowledges financial completion. The public projection exposes
counts, stage, observation time and a server-generated message, with no private
provider or credential identities. Polling is explicit and optional.
