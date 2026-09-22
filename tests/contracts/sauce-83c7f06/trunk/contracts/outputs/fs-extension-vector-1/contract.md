# Auxiliary model-extension vectors

HDF5 tangent/covector files contain scalar string datasets:

- `/schema`: `fs-extension-vector-1`.
- `/fingerprint`: exact basis/property/axis descriptor fingerprint.
- `/baseline`: frozen objective-state fingerprint.
- `/role`: `tangent` for input directions and solved taps, `covector` for dual outputs.

For one-based field index `p` and axis index `k`, `/fields/p/lag/k` (time lag)
or `/fields/p/offset/k` (spatial half-offset) is a one-dimensional real64 dataset in global spatial control-ID order. Its length
is `spatial_count` in the companion `fs-model-extension-1` JSON descriptor.
Fields are ordered as in the request. The descriptor has a zero-based JSON
`fields` array, with each entry recording `control`,
`basis`, `property`, `spatial_count`, and either physical lag `seconds` or
`half_offsets_meters` (one spatial vector per offset). Root `fingerprint` and `baseline` provide the required bindings. A mismatched basis,
extension axis, baseline, column extent, role or nonfinite input is rejected.

Tap amplitudes are signed perturbations in the physical property's catalog
units, with lag quadrature weights absorbed. They do not change physical
material bounds or nonlinear control coordinates. The pairing is the ordinary
real Euclidean sum over all fields, global control IDs and axis entries, with no
extra factor of two or lag-spacing weight.

Distributed inputs read only locally required spatial runs. Forward operations
exchange two contracted real/imaginary columns per lag field. Spatial offsets
retain one real amplitude per offset and use bounded native donor exchanges.
Reverse operations reduce spatial ghost contributions before lag expansion, and outputs write
only owned global runs. Replicated maps reduce their dual once across ranks.
Empty ranks participate in all collectives; vectors are never globally gathered.

## Shared-band integration boundary

This version binds one frozen baseline. Equal spatial counts do not make files
interchangeable across frequency baselines. A shared-band driver needs a common
coordinate identity plus explicit per-frequency bindings; do not bypass existing
fingerprint checks. The planned dual-driver and band-state semantics are in the
[iteration/composition design](../../internal/fs-extension-iteration-1/contract.md).
This note does not change the current HDF5 layout or permit a band manifest in
place of `/baseline`.
