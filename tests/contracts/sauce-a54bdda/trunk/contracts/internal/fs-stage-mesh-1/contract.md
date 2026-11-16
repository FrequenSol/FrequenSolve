# Frozen frequency mesh companion

Status: internal; capture/replay is experimental. Controlled patch PML remains
gated pending derivative acceptance.

One companion owns `initial.h5` (full hp checkpoint before solver setup),
`refinements.h5` (the solver h/p decision journal), and `final.h5` (the full hp
checkpoint after setup). The `fs-stage-mesh-1` JSON manifest contains `context`,
`execution_identity`, and a `files` object mapping these three fixed names to
`sha256` and `bytes`. File identities cover exact bytes. The manifest identity is
the SHA-256 of its exact bytes, prefixed with `sha256:` and retained independently.

The execution identity is the native canonical JSON SHA-256 of `context`.
Context includes the immutable input-stage identity, dimension, order encoding,
solver precision, physical and adaptation frequencies, PML sizing frequency,
geometry parameter values, source positions and simulation execution settings.
Pinned model definitions are represented by the input-stage identity and are
validated separately before candidate application. Simulation names, project
paths, output requests and serialization metadata do not affect this identity.
Referenced execution files contribute content identities and dataset locators,
so moving an unchanged bundle does not change the execution identity.

Experimental `fwi_operator/controls/stage_mesh` has either `mode: capture` or
`mode: replay` with `manifest` and `identity`. Both require `pml_stage` and `state`.
Capture uses the `linearize` action and its objective configuration, requires the
exact canonical stage state file, writes working artifacts
under the task run-contract directory's `stage_mesh/`, and stops after hierarchy
construction without wave solves. The SDK publishes the complete verified
directory atomically beside the input stage and refuses replacement. Capture does
not modify the input stage bundle.

Replay verifies the manifest, all files and execution context before mesh mutation.
It restores the initial hp mesh after frequency-specific GMP/PML stretching,
replays the solver decisions while assembling the configured hierarchy with the
candidate physical materials, then compares final topology and every node order
without changing the realized solver mesh. Reads recheck committed bytes afterward.
Mismatch is fatal; no fallback to candidate-dependent adaptation is permitted.

The low-level SDK interface uses one frequency per job. A stage band owns a
separate companion per patch/frequency, with normal frequency-dependent PML
sizing. No common stage-band sizing frequency is imposed. Different MPI ownership is permitted when
the complete replicated topology and canonical context agree. Reduced distributed
mesh views remain unsupported by the hp restart primitive.
