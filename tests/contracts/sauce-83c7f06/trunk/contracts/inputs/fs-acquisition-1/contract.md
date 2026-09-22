# FS Acquisition Contract v1

Status: initial
Visibility: public
Schema id: `fs-acquisition-1`

## Summary

Acquisition configuration defines source groups, receiver groups, batching, and
receiver output channels.

## Required Behavior

- `source_groups` must contain at least one source group.
- Current source group types include `PointSource`, `CompoundSource`,
  `Microseismic`, and `GaussianBeam`.
- `receiver_groups` defines zero or more receiver groups.
- Receiver `post_process` defaults to false and requests reconstruction only
  when every active physics supports it. Acoustic and elastic DPG formulations
  are currently supported. If any active physics is unsupported, initialization
  emits one warning and disables post-processing for the entire problem.
- `max_batch` limits active source groups per batch.
- Dense source encodings should use `HDF5Dense` in saved simulations. The
  dataset stores complex coefficients as `(field, source, complex=2)` in h5py
  order; `JsonDense` remains a compact compatibility form for small examples.
- Receiver group `device/components` must contain at least one component for
  receivers that write traces.
- A receiver component may define one complex scalar `weight`. Pointwise
  weights belong to `EncodedReceiver`. Its bulk HDF5 table is stored as
  `(encoding * component, receiver, complex=2)` in h5py order, with
  encoding-major/component-minor rows. The receiver dimension covers the
  complete fixed receiver-group geometry.
- `EncodedReceiver.encoding_count` defines the number of encoded responses.
  Base component metadata occurs once in `device/components`; the runtime
  expands it to one output channel per encoding/component pair. Explicit
  `encoding_names` may be inline or stored in the weight file through
  `weights/names_dataset`.
- `EncodedReceiver` may set `reduction` to `sum` or `mean` to combine its fixed
  receiver geometry, or `none` to retain pointwise traces. Its weights always
  define the forward operator; adjoint modeling applies the Hermitian transpose.
- `ReceiverArray` defines physical receiver-node `offsets` around every
  receiver-group coordinate. Scalar or per-coordinate `offset_units` are
  accepted. `mean` (the default) or `sum` combines only the nodes belonging to
  each coordinate; `none` exposes all expanded nodes.
- Receiver weight values remain in their referenced bulk dataset and are not
  copied into receiver metadata or preprocessing provenance.
- Receiver coordinate blocks may come from files, inline arrays, grids, or DAS
  fiber descriptions.
- `CoordsFromFile.file` may be a plain path or an HDF5 object locator such as
  `simulation.h5:inputs/acquisition/receivers/surface/coordinates`. The `file`
  locator is authoritative; optional hash metadata is for external tooling.
- Receiver coordinate blocks, including DAS fiber core coordinate files, accept
  `system` plus scalar or per-component `units`. Values are converted into solver
  physical coordinates before sampling.
- `ReceiverFiber` DAS devices prefer `gauge_length`, `channel_spacing`, and
  `sample_spacing`. `channel_spacing` defaults to `gauge_length`.
  `sample_spacing` defines the reusable sample grid; when omitted, the runtime
  uses `gauge_length / points_per_gauge`.
- A helical `ReceiverFiber` requires `radius` and exactly one of `angle` or
  `pitch`. `angle` is the actual winding angle between the fiber tangent and
  the cable/core axis, not its complement. Numeric angles are degrees by
  default; `{ "value": ..., "units": "deg" }` and
  `{ "value": ..., "units": "rad" }` are also accepted. Angles must be
  strictly between 0 and 90 degrees. `pitch` remains supported for backward
  compatibility, with `pitch = 2*pi*radius/tan(angle)`.
- In two-dimensional simulations, a helical fiber direction is the in-plane
  projection of the normalized three-dimensional tangent. The projected
  direction is not renormalized after its out-of-plane component is removed.
- Receiver `sampling/_type` defaults to `DenseAllToAll`. `DenseAllToAll` and
  `Dense` do not use sparse trace identity metadata.
- Sparse sampling types are `Sparse`, `SurveyPairs`, `OffsetDomain`,
  `HDF5TraceStore`, and `SPSFiles`. Sparse layout preparation requires source
  coordinates so receiver points can be positioned and source-receiver geometry
  can be computed internally.
- Inline `Sparse` sampling accepts external trace identity only: `trace_id`,
  `source_id`, `receiver_id`, `component`, `weight`, and optional source,
  receiver, and component names. Internal point ranges, sample maps,
  receiver-position ids, channel numbers, component ids, field records, offsets,
  and azimuths are not public sparse input fields.
- Sparse trace `weight` is a real32-compatible scalar. A value of `0.0`
  disables the trace for sparse evaluation.
- `SurveyPairs` and `SPSFiles` may read `source_file`, `receiver_file`, and
  `relation_file`; they may also read an existing sparse HDF5 layout from
  `layout_file`.
- SPS import reads revision 1.0 or 2.1 fixed columns. `sps_revision` (`auto`,
  `1.0`, `2.1`) selects the layout; `auto` uses the `H00` header, then the point
  record layout, and one revision applies to all three files. Line names are
  text keys (numeric names compare by value). The vertical coordinate is surface
  elevation less point depth; 2D builds use easting and that vertical. Relation
  spreads are uniform in point number between `from` and `to`, so the point step
  need not be one. Unreadable fields, duplicate points and relations naming
  absent points are errors that identify the file and record. Other header
  records (CRS, units), statics and datum are not applied.
- `HDF5TraceStore` reads sparse trace layouts from `layout_file`.
- `OffsetDomain` builds a sparse layout from receiver coordinates, source
  coordinates, and the `offset_domain` bounds. Offsets and azimuths are computed
  by the runtime and are not persisted as input trace metadata.


## Compatibility

This schema focuses on shared source/receiver shape, public sparse sampling
metadata, and the DAS spacing controls used by current `ReceiverFiber` readers.
Seismic mechanism internals use the shared seismic-mechanism schema;
device-specific DAS geometry details remain permissive.

Interim weighted `ReceiverArray` payloads remain accepted and retain their
historical all-geometry reduction and component names. Older `ReceiverArray`
payloads with neither offsets nor weights are treated as point receivers. New
writers use `EncodedReceiver` for weights and `ReceiverArray` for physical
offset arrays.

### Thermal source strengths

With thermal physics, scalar/monopole heaters default to 1 W in 3D and 1 W/m
in Cartesian 2D (power per out-of-plane thickness). Unit-bearing amplitudes and
directions use the same power dimension, including prefixed units such as mW.
A bare numeric amplitude multiplies that default strength.


## Seismic fault mechanisms

A tensor source may specify a `mechanism` instead of raw `direction` components.
Use `double_couple` with `strike`, `dip`, and `rake` for a shear event. Numeric
angles default to degrees; `angle_units = "rad"` and individual `{value, units}`
angles are supported. Strike is clockwise from north, dip is between 0 and 90
degrees below horizontal, and rake is between -180 and 180 degrees. Positive
rake is up-dip: +90 is reverse faulting and -90 is normal faulting. Alternatively,
specify perpendicular, nonzero `fault_normal` and `slip_vector` directions.
Angles and vectors are mutually exclusive.

Use a source-level `amplitude = {"value": ..., "units": "N*m"}` for scalar moment,
or `moment_magnitude` for Mw. They are mutually exclusive. Mw follows the GCMT
conversion `M0 = 10^(1.5 Mw + 9.1)` N m. A bare numeric amplitude retains the
shared source convention: it multiplies the default 1e9 N m strength. Omitted
strength also defaults to 1e9 N m. Mechanism-local `amplitude` remains accepted
as a physical moment, with `units` defaulting to N*m; it cannot be combined with
source-level strength. Scalar moments must be finite and positive.

`clvd` specifies its symmetry `axis` as a vector or `{azimuth, plunge}` in degrees
(default) or explicit angular units. `microseismic` combines optional `dc`,
`clvd`, and `iso` objects with signed `weight` values. An omitted component is
absent; a present component defaults to weight 1. Zero-weight components do not
require an orientation. The legacy DC `axis = {strike, dip, rake, units}` and
CLVD fault-normal `axis = {strike, dip, units}` spellings remain accepted.

The default `normalization = "scalar_moment"` normalizes the combined tensor so
`sqrt(sum(M_ij**2)/2) = M0`. Signed mixture weights determine shape, not separate
component moments. An all-zero or cancelling mixture cannot be assigned a scalar
moment. `normalization = "legacy"` is available only for `microseismic`: it divides
component weights by their absolute sum and preserves the historical DC, CLVD,
and identity-tensor amplitudes without normalizing the final scalar moment.
This mode preserves mixture scaling, not the previous incorrect angle conversion.
Moment magnitude requires `scalar_moment` normalization.

`moment_tensor` accepts a symmetric real 3x3 `tensor` with `units` defaulting to
N*m. Without a separate strength, entries are physical moment components. With
an amplitude or magnitude, the tensor supplies the shape and is normalized to
that scalar moment. An all-zero raw tensor is permitted only without a separate
strength. Packing into the solver's Voigt order occurs after rotation; authored
shear components are physical tensor entries, not Mandel-scaled entries.

### Mechanism coordinates

Position `coordinates.system` and orientation `mechanism.system` are independent.
The orientation system defaults to `global`, regardless of the position system.
A named orientation uses its local orthonormal basis evaluated at the event's
Cartesian position. Translation does not act on the tensor. Geographic bases
vary with longitude/latitude; explicitly selected surface frames use the local
surface tangent/normal basis and require resolved geometry context.

For fault-plane angles, the reference frame's first, second, and third basis
vectors define local east, north, and down, respectively. Thus strike is relative
to that frame's north; a rotated survey frame defines survey north, not true
geographic north. Surface-relative angles are measured relative to the declared
surface frame and its positive normal. Cartesian, geographic, and surface frames
support angle descriptions. Cylindrical and spherical frames require explicit
vectors or a tensor because their axes do not define a fault strike reference.

Vector and raw tensor components default to `convention = "local"`. `NED`, `END`,
and `ENU` explicitly reorder or reverse the reference-frame axes. Legacy
`system = "NED"`, `"END"`, or `"ENU"` selects that component convention in the
global reference frame. Fault-plane angles already define their convention and
must omit the separate `convention` field. The runtime rotates the complete
3x3 tensor into Cartesian components before packing or reducing it.

Ordinary planar 2D sources reject nonzero omitted-axis moment components rather
than silently discarding them. Compatible in-plane tensors retain the shared
unit-thickness source convention. 2.5D retains all six tensor components. These
rules also apply to the legacy MicroSeismic_t initializer.

Use `source_signature` for event spectra and timing, with stable source ids and
the existing per-Hz derivatives. Mechanism strength is frequency-independent;
its tensor is multiplied by the supplied signature without implicit conversion
between moment and moment-rate spectra. A moment-rate spectrum must therefore
be converted to the solver's required source spectrum by the authoring layer,
including the intended treatment of zero frequency.

### Compatibility and validation

Earlier microseismic readers ignored degree units, reversed the horizontal and
vertical dip limits, and treated NED entries as solver Cartesian entries. Those
behaviors are corrected; previously authored microseismic results may change.
Validate migrated cases against their intended physical mechanism, not those
incorrect results. The old commented coordinate-module mechanism readers are
superseded by the shared source-mechanism reader.

## Integrated receivers

`ReceiverIntegral` expands `device.geometry` (Polyline or TriangleSurface) into
weighted samples and one output receiver. Orientation supplies tangent/normal
directions. It supports dense Cartesian 3D sampling, and excludes `n_avg`,
explicit directions, offsets and device weights. Triangle indices are one-based.
`voltage`, `magnetic_flux` and `coil_voltage` require line, surface and surface
geometry respectively; their output dimensions are V, Wb and V. `B` and
`B_x/y/z/all` represent constitutive magnetic induction; existing `magnetic`
aliases retain their μ0 H definition. `magnetic_intensity` and its `_all/x/y/z`
variants provide H in A/m. Integer component `time_derivative` in
[-2,2] composes powers of iω and changes the output dimension accordingly.
Receiver quadrature is fixed at initialization; check convergence across cells.
See [native instruments](../../../docs/imaging/native-instruments.md).
