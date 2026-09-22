# Joint control directions and covectors

The `direction` input for JVP/normal and `covector` output for
linearize/VJP/normal use the same HDF5 vector contract. Required string datasets:
`/schema = fs-control-vector-1`, `/packing = real_interleaved`,
`/state_fingerprint`, `/control_registry_fingerprint`.

Each active block has `/controls/<qualified block ID>` in its declared real
optimizer coordinates. Distributed material blocks preserve global control IDs
and basis identity. Complex blocks interleave real/imaginary parts; the real
Euclidean dot product equals the real-Hermitian complex pairing, with no factor
of two. Source pullbacks reduce once at their provider; material pullbacks use
native control ownership. Do not allreduce the concatenated joint vector again.

The ordered `controls/active` list is mandatory and authoritative. Empty spaces
are valid for linearize/VJP; JVP and normal require a nonempty space. The saved
baseline and registry identities must match exactly. Output filenames receive
the normal frequency-task suffix. In a job with more than one frequency task the
`direction` input first tries the task-suffixed sibling `<stem>_<task><ext>`,
then the exact path.

Covectors written by `linearize`, `vjp` and `normal` additionally carry one
packed support bitmask per active block, `/support/<qualified block ID>` (uint8,
8 DOFs per byte, LSB first, vector order), the float64 scalar
`/support_min_support`, and with `controls/support_measure: true` the uint8
`/support_measure/<qualified block ID>` quantized measure
`round(255 s_i / max s_i)`. The measure is `s_i = sum_q w_q |dm/dc_i|` at the
frozen baseline; DOFs below `controls/min_support` times the block's median
nonzero measure are unsupported. Direction inputs need none of these datasets.
The same groups appear in `fs-control-state-1` exports, where inactive blocks
report full support. See
[control support](../../../docs/imaging/controls.md#control-support). State
exports additionally record `/scaling/<mechanism block>` for cross-task replay;
vectors carry no scaling and stay in the executing task's coordinates.

`normal` returns the fused joint Gauss-Newton action `J* J direction`, including
cross terms between active blocks. It is equivalent to JVP followed by VJP under
the saved derivative policy. It excludes residual second derivatives,
regularization and nuisance-variable Schur elimination.

Provider-specific source/model artifacts and the old `joint_direction`,
`joint_covector`, and `model_normal` spellings are rejected by these operations.
The separate WRI workflow retains its explicit material-only fields.
