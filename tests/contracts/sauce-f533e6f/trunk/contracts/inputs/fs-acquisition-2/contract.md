# FS Acquisition Contract v2

Status: initial
Visibility: public
Schema id: `fs-acquisition-2`

## Summary

Acquisition configuration defines an ordered source catalog from point-source
geometry, distributed boundary loadings, or both; optional point-source
encoding into RHS fields; receiver groups; batching; and receiver output
channels.

## Required Behavior

- At least one of `source_geometry`, `boundary_loadings`, or
  `impedance_sources` is required.
- When present, `source_geometry` must define at least one physical source (a point or retained line).
- `source_geometry._type` is one of `Inline`, `HDF5`, `SPSFiles`, or `Quadrature`.
- Inline `source_geometry` permits each SourcePoint to override the geometry's
  default physical `kind` and basis. This allows scalar, vector, tensor,
  monopole, gradient, and dipole point mechanisms in one RHS catalog.
- HDF5 and SPS source geometries remain homogeneous: all SourcePoints share the
  geometry-level `kind` and basis.
- SPS source geometry accepts `sps_revision` (`auto`, `1.0`, `2.1`) and places each
  source at surface elevation less point depth; see the
  [SPS import rules](../fs-acquisition-1/contract.md).
- In acoustic and coupled physics, explicit `volume_injection` sources are
  volume injection rates: m^3/s in every dimension. Planar 2D distributes Q over
  a fixed physical 1 m out-of-plane thickness, giving internal line rate Q / 1 m
  independently of coordinate units and reference scales. Source metadata retains Q in m^3/s.
  First-order pressure/velocity equations load Q; frequency-domain pressure
  Galerkin loads i*omega*Q, with exp(+i*omega*t). There is no source-density
  divisor. Their ideal reciprocal receiver measures pressure. This convention
  must be selected by source kind and applies to `physics: acoustic` and to the
  acoustic (fluid) region of `coupled`, `coupled_axisym`, `coupled2`, and `coupled_aep`
  models (including `acoustic_axisym`); each point must lie in an acoustic element, and a point inside an
  elastic or poroelastic region fails mapping. Other physics reject the kind.
  Acoustic `isotropic_moment` and legacy `monopole` sources are isotropic moments `A` (N m), see
  [point-source loading](#point-source-loading-of-forces-and-moments), in transient
  solves too. For transient volume injection, smooth time signatures are differentiated
  analytically. Step volume-rate signatures are
  rejected because their derivative is an impulse. Fresh volume-injection runs require
  initially quiet Q and dQ/dt (relative tolerance 1e-8); increase the pulse delay
  or start earlier to match the solver's zero initial state. `gradient` and `dipole` are
  gradient-of-delta force sources, so their physical source strengths use
  moment units (`N*m`); `dipole` is directed while `gradient` is isotropic.
- Without a physical strength, a source uses its kind's default strength:
  1e6 N for `vector` (force) sources; explicit `volume_injection` sources use
  1 m^3/s. Acoustic/coupled `scalar` and `pressure_monopole` use 1 m^3/s^2.
  Elastic scalar, `isotropic_moment`/`monopole`, and `gradient`,
  `dipole`, and `tensor` moment sources use 1e9 N m. The default scales the
  authored direction or mechanism, so a force direction of length 2 gives 2e6 N.
  A bare numeric `amplitude` multiplies the default; an amplitude with units, a
  `moment_magnitude`, or unit-bearing direction components replace it. Current
  sources default to 1 in their units and thermal sources to 1 W (1 W/m in 2D).
  Traces are responses to this strength; `/survey/source_geometry` records it.
- Cartesian Maxwell point sources accept `electric_current` and
  `magnetic_current`, with a three-component solver-frame direction. Their
  physical moment vector is amplitude times direction; direction is not
  automatically normalized. Use a unit direction to specify moment magnitude
  through amplitude alone. Their amplitudes use `A*m` and `V*m` in 3D, or
  moments per unit invariant length in `A` and `V` in 2D. Impressed magnetic
  current is a mathematical Maxwell RHS, not a loop magnetic dipole specified
  in `A*m^2`. These sources support ordinary forward and native model/source
  imaging actions; modal and spectral-derivative current loads are not enabled.
- All SourcePoints in one geometry share one explicit or automatically resolved
  domain.
- `GaussianBeam` is not a `source_geometry` type.
- `impedance_sources` defines analytic nonhomogeneous incident fields
  applied through impedance boundary operators. Accepted types are
  `GaussianBeam`, `PointSource`, `MaxwellPlaneWave`, and `PoroelasticPlaneWave`. Maxwell plane waves can
  supply the logical RHS catalog when no physical point-source catalog exists.
- Gaussian impedance-beam `rhs` values are one-based indices in the active RHS
  batch. All referenced fields must fit in a single acquisition batch, and
  multiple beam entries assigned to the same RHS are summed.
- Gaussian impedance and equivalent beams use `source/theta` and `source/phi`
  as global simulation-coordinate propagation angles. Their direction is
  independent of the orientation or normal of any boundary carrying the load;
  in two dimensions, `theta = 0` propagates along the positive first axis.
- Point impedance sources evaluate the same outgoing homogeneous acoustic Green
  field as point equivalent sources. Each entry selects a one-based `rhs` and
  defines `source/origin`, positive reference slowness `source/n0`, optional
  `source/amplitude`, and optional phase in degrees `source/phase`. They are
  intended for the boundary of an excised source region.
- Maxwell plane-wave impedance sources solve the two transverse eigenmodes of
  the local complex permittivity and permeability tensors. `direction` is a
  three-component global propagation direction; `electric` and optional
  `electric_imag` select and phase-align the nearest eigenpolarization. Each
  wave may provide a stable `source` name and positive `rhs_normalization`; the
  runtime otherwise generates the source name from its one-based `rhs`. The
  resulting electric and magnetic Cauchy fields therefore remain consistent
  with vacuum, conductive-geological, and cold-plasma material tensors.
  `amplitude` (default 1 V/m) is the physical incident electric-field
  amplitude: a bare number is read in V/m, and a `{value, units}` object may
  use any electric-field unit (`V/m`, `mV/m`, ...). The runtime converts it to
  solver units, so the incident field, and the forced solution, do not depend
  on `Mesh/adapt/f_low` or the EM scaling mode. Isotropic 2-D and 3-D plane waves
  use the unit transverse projection of `electric`; anisotropic waves use the
  selected unit-norm eigenpolarization; Cartesian 2.5D waves project the
  authored `electric` vector without renormalization (see Compatibility).
- `equivalent_sources` is experimental. When configured, the runtime emits one
  initialization warning. It optionally injects incident Cauchy data through an
  additive DPG source sheet while leaving the interface trial and trace
  operators unchanged. A `pml_interface` surface selects the physical side of
  the named original boundary, not the generated PML outer boundary. An
  `internal_plane` surface selects root-element sides coincident with its
  `point` and `normal`; `emission_side` chooses the retained incident-field
  half-space and defaults to `positive`.
- `equivalent_sources/incident_field/kind` accepts `gaussian_beam`,
  `point_source`, and `scholte_wave`. Gaussian beams use the same RHS, source,
  and global-direction conventions as `impedance_sources`.
- A `scholte_wave` equivalent source requires an `internal_plane` on the
  fluid-solid interface. Either `emission_side` is accepted and both impose
  the same sheet: the fluid-side trace equals the solid-side trace plus the
  incident Cauchy data (pressure and normal velocity jump weights +1/+1, the
  `ScholteSheetOptions_t` default). `negative` (against the fluid-to-solid
  normal) loads the incident pressure and normal velocity on the acoustic
  sides; `positive` loads the incident velocity and traction on the elastic
  sides. The `scholte-wave` study uses `negative`, whose data are exactly the
  two jumps of the analytic sheet response. The incident field defines
  orthogonal `tangent` and `normal` directions, an interface `origin`,
  isotropic fluid speed/density, and isotropic solid P-wave speed, S-wave
  speed, and density. The normal points from fluid into solid. The runtime
  solves the exact isotropic Scholte secular equation and applies
  pressure/normal velocity on the acoustic portion and velocity/traction on the
  elastic portion of the sheet.
- The optional Scholte `surface/tangential_window` restricts the exact Cauchy
  data to an interval along the propagation tangent, measured from the
  incident-field origin. A C2 quintic ramp is applied over `taper_width` at
  both interval endpoints and the source is zero outside the interval. This
  supports a smoothly windowed source on a fluid-solid interface as well as a
  finite aperture on other internal source planes.
- A point equivalent source supplies an outgoing homogeneous acoustic Green
  field. Its `source/origin`, positive reference slowness `source/n0`, and
  optional `source/amplitude`, and optional phase in degrees `source/phase`
  define that field; positive phase multiplies it by `exp(+i phase)`, and `rhs`
  selects its one-based active RHS.
- Point fields support two transparent RHS-only constructions. An
  `adaptive_box` is a closed Huygens sheet selected from exterior adaptive leaf
  faces and therefore must coincide with leaf-element sides. Its `half_width`
  may be one scalar or one positive length per active axis. Optional
  `enrichment_order` sets the polynomial order of every leaf element in or
  touching the closed box; conformity closure may promote adjacent elements.
  Its optional `loading` is `huygens` by default and applies both analytic
  Cauchy data so that the incident field cancels inside. `monopole` retains
  only the normal-velocity sheet and `dipole` retains only the pressure sheet.
  A `volume_shell`
  applies the strong acoustic operator to a quintic radial cutoff: the field is
  zero at and inside `inner_radius`, equals the Green field at and outside
  `outer_radius`, and transitions smoothly between them. The Green singularity
  is never evaluated by either construction.
- Equivalent-source surface and volume support is cached after each mesh-state
  change and respects distributed leaf ownership, so every selected load is
  assembled once under MPI.
- Before production promotion, incident-field evaluation must be fully
  separated from mesh-support selection, mesh-owned support introduced in place
  of geometric leaf scans, and broader adaptive/distributed validation
  completed.
- Source names in one `source_geometry` must be unique.
- Large homogeneous source-point catalogs should use HDF5 coordinates. Optional
  custom names belong in `names_dataset` in the same file; generated
  `source_000001` names require no stored name table.
- HDF5 source coordinates take their units from the geometry's `units` (scalar),
  then from a string `units` attribute on the coordinate dataset (one entry for
  all axes, or one per axis), then km; top-level `Units/defaults` does not
  apply. When both `units` and the attribute are present they must name the
  same unit (identical text, or the same dimension and scale); otherwise the
  geometry is rejected. Angular axes of `system` follow the coordinate-system
  rules. Receiver and DAS coordinate files follow the same rule
  ([fs-acquisition-1](../fs-acquisition-1/contract.md)).
- `source_encoding` is optional. When omitted, the runtime uses identity
  encoding: one RHS field per SourcePoint, with field names copied from source
  names.
- `source_signature` is the production physical-source spectrum. Its complex
  values `q` are indexed by physical frequency, optional Laplace damping, and
  stable physical-source identifiers. The runtime requires an exact task
  coordinate and never extrapolates. When omitted, `q = 1` and `q_f = 0`.
- HDF5 values use h5py order `(frequency, source, complex=2)`, optionally
  `(laplace, frequency, source, complex=2)`. Complex parts are real then imaginary.
  Laplace coordinates are nonpositive imaginary frequencies in Hz, so the
  transform kernel is `exp(-2*pi*i*(f+i*laplace)*t)`. A rank-three undamped
  table is invalid at nonzero Laplace frequency. Axes, values and derivatives
  must be finite, frequency coordinates strictly increasing, and task matching
  unambiguous within relative tolerance `1e-6` (absolute floor `1e-14` Hz).
- A zero signature at a spectral notch is valid. A fully silenced field retains
  the original encoding's finite normalization and reference position; a nonzero
  signature derivative remains active.
- Signatures are applied before encoding. The effective RHS map and its
  physical-Hz derivative are `C = E diag(q)` and
  `C_f = E_f diag(q) + E diag(q_f)`.
- `source_signature/spectral_derivative = total` requires
  `frequency_derivative_dataset`; `frozen` sets `q_f = 0`. The derivative and
  its cotangent use physical-Hz units. Stable source identifiers must match the
  source-geometry order exactly.
- A total signature may also list `higher_frequency_derivative_datasets`, the
  raw derivatives of orders two and above with the dataset's shape. Spectral
  planes above first order with a total source (a windowed phase comparison)
  require them through the hierarchy order, an encoding with `E_f = 0`, and
  point-force sources that share one spectrum per RHS.
- `source_encoding._type` is one of `Named`, `JsonDense`, or `HDF5Dense`.
- `Named` encodings define sparse terms by source name.
- `JsonDense` encodings define field-major coefficient arrays and are intended
  only for small examples and compatibility.
- Saved dense encodings should use `HDF5Dense`. The bulk dataset has h5py shape
  `(field, source, complex=2)` and is read as a complex coefficient matrix with
  conceptual shape `[n_source, n_rhs]`. Small field labels may be stored in the
  dataset's `field` attribute; source-axis values are not copied into metadata.
- Frequency-dependent `HDF5Dense` encodings use h5py shape
  `(frequency, field, source, complex=2)` and name a one-dimensional physical
  frequency axis with `frequencies_dataset`. Each task reads only its matching
  coefficient slice; a missing frequency is an input error.
- Authored encoding coefficients are frozen by default, so `E_f = 0`.
  `spectral_derivative = total` requires `frequency_derivative_dataset`, whose
  values are `dE/df` in inverse physical Hz and whose storage shape matches the
  coefficient dataset.
- Acquisition HDF5 references use separate `file` and `dataset` fields. The
  runtime continues to accept legacy `file:dataset` receiver-coordinate
  locators, but new writers do not duplicate the dataset path in `file`.
- JSON source-encoding coefficients are either real numbers or `[real, imag]`
  arrays. Object-form complex numbers are intentionally not part of this
  contract.
- `conjugate_coefficients` lazily conjugates all encoding coefficients in the
  runtime. This is the frequency-domain time-reversal path for both inline and
  HDF5 encodings and does not copy or rewrite coefficient arrays.
- Dense encodings must have a source dimension equal to the number of source
  SourcePoints.
- Explicit encodings must define at least one RHS field, and every RHS field
  must have at least one nonzero effective coefficient.
- The runtime uses its tuned `max_batch = 64` default. Input generators must
  omit this property; it is reserved for explicit advanced-user tuning in
  exceptional memory or hardware configurations. It limits active RHS fields
  per batch, not physical SourcePoints.
- `write_vtk` controls task-1 acquisition schematic output. When omitted, the
  runtime writes source/receiver VTK schematics only when the schematic would
  contain fewer than 1,000,000 source and receiver sample points.
- Sparse receiver `source_id` values refer to RHS field ids.
- Identity RHS reference coordinates are source coordinates.
- Encoded RHS reference coordinates use the coefficient-magnitude centroid.
  Explicit encoding reference coordinates are deprecated; physical source
  geometry and the simulation coordinate system are authoritative.
- Physical source strength belongs in `source_geometry`; `source_encoding`
  coefficients are dimensionless complex multipliers.
- `boundary_loadings` accepts deterministic `SurfacePressure` entries and
  low-rank `SurfacePressureCovarianceFactor` entries. Each entry targets the
  stable name of a boundary-condition assignment containing `gravity_surface`.
- Boundary fields are independent sampled-field stores layered above the
  existing spatial xarray reader. Different fields may use different spatial
  grids. The runtime selects the active source and frequency before spatial
  interpolation, so frequency and time are not generalized as xarray spatial
  axes.
- A frequency-domain pressure field supplies a strictly increasing physical-Hz
  list and stores `[real, imag]` components for each frequency. The active solve
  frequency must match one stored frequency. Complex values are positive-frequency
  phasors for the solver's `exp(+i omega t)` convention.
- Frequency-domain fields may also supply `derivative_df`, `derivative_ds`,
  `derivative_d2f`, and `derivative_d2s` xarray datasets. They use the same
  spatial layout and `[real, imag]` frequency packing as `data`; first derivatives
  are pressure per physical Hz and second derivatives are pressure per physical
  Hz squared. A `forward_df` or `forward_ds` workflow requires the selected
  derivatives through its requested `derivative_order` for every frequency-domain
  surface-pressure field. Ordinary forward workflows do not require them.
- A time-domain pressure field supplies uniformly sampled real pressure,
  physical `time_origin` and `time_step` in seconds, and optional `hann` window
  and mean detrending. FFTW builds (`build.fftw`) transform aligned bins with
  FFTW; every other frequency, and every bin in FFTW-free builds, uses a direct
  DFT, with amplitude normalization. It
  obtains first physical-frequency derivatives from the first time moment and
  second derivatives from the second time moment of those samples, so time-domain
  fields need no derivative datasets.
- Each field carries an authoring-generated positive `rhs_normalization` in its
  declared pressure units. Sauce divides the load by the corresponding
  nondimensional source scale during assembly and restores that scale in output
  metadata, matching the existing point-source normalization path and keeping
  small physical pressure units numerically active.
- Boundary source labels share the acquisition RHS catalog with encoded point
  fields. Repeated labels contribute additively to the same RHS, while distinct
  labels define distinct sources and batches.
- A covariance-factor loading represents `Q_p = L L^H` using the explicit
  convention `E[s s^H]`. Its packed HDF5 component axis is frequency-major and
  stores `[factor_1.real, factor_1.imag, ..., factor_rank.real,
  factor_rank.imag]` for each frequency. `rank` is fixed across the frequency
  axis; `rank_per_frequency` records the numerical rank before fixed-axis
  padding, and inactive trailing columns are stored as zeros. This metadata lets
  downstream products avoid padded columns while preserving a stable source
  axis across frequency shards.
- Every covariance column becomes a distinct RHS named
  `source_prefix_<one-based-factor-index>`. These names and `factor_set` values
  must be unique: covariance RHSs are never deduplicated or additively merged.
  `rhs_normalizations` provides one positive pressure-coordinate scale per
  column. Sauce applies no sidedness conversion or implicit factor of two.
- The propagated receiver samples for covariance RHSs are the response factor
  `Z = H L`. The existing trace payload retains those complex values in physical
  receiver coordinates; `/survey/source_statistics` identifies their factor
  set, column, rank, spectral convention, and factorization provenance.
- Surface-pressure loading is currently supported by the coupled acoustic DPG
  discretization. It enters the nonhomogeneous gravity-surface flux relation;
  the homogeneous gravity-surface operator is unchanged.

## Metadata

- `/survey/sources` describes RHS fields and includes `source_id`,
  logical `source_name`, `field_record`, `coordinates`, and `source_scale`.
  Generated identity names may be represented by name-encoding metadata instead
  of a materialized string dataset in trace-store outputs.
- `/survey/source_geometry` describes the physical SourcePoints. Per source,
  `source_kind_name` is the input kind, `strength` the physical strength of the
  load behind the traces, in `strength_units` (the kind's default units, such as
  `N`, `N*m`, or acoustic volume-rate units). Vector kinds report the Euclidean norm, tensors the scalar
  moment `sqrt(sum(M_ij**2)/2)` and scalar kinds the magnitude; it reflects the
  basis at write time, including updated source controls. `strength_origin` is
  `specified` when the input fixed the strength physically, `default` when the
  kind's default strength was assumed, `scaled_default` when a bare numeric
  amplitude multiplied that default, and `none` for strength-free ray origins
  (whose `strength` is NaN). Trace values are responses to this strength, not
  per unit source; encoding weights in `/survey/source_encoding` are
  dimensionless multipliers on top of it.
- `/survey/source_encoding` describes source-encoding provenance.
- `/survey/source_statistics` is present when covariance-factor RHSs exist. It
  joins to the source axis by `source_id` and marks those traces as
  `receiver_response_factor` with receiver convention `E[d d^H]`.

## Compatibility

This is a breaking replacement for `fs-acquisition-1` source groups. Public
input should use `Acquisition/source_geometry` and optional
`Acquisition/source_encoding`; `source_groups` is not part of this contract.

For Cartesian 2.5D Maxwell, `MaxwellPlaneWave` uses the active transverse
wavenumber to construct a consistent isotropic plane mode. The first two
`direction` components specify its in-plane direction, and `electric` uses
solver components `(x,z,-y)`. Anisotropic modal plane-wave sources are rejected.

Each wave accepts `profile`, default `plane`. The optional `cylindrical_bessel`
profile requires cylindrical Maxwell geometry and vacuum. It prescribes the
exact modal field `E_Z=J_|n|(omega R/c)` and matching H using the simulation's
`toroidal_mode`. `amplitude`, `phase`, and `rhs` apply; `direction`, `electric`,
and `origin` do not alter this fixed verification profile.


### Cached receiver maps

Point-to-element root maps are cached beside the simulation mesh and reused only
when every receiver's final position (after files, grids, array offsets,
coordinate systems, datums and units) and the initial mesh match the cached ones
within 1e-9 of the interior domain diagonal (at least 1e-6 m). Tasks that reuse
trace metadata from their own init skip reading receiver coordinates; that
shortcut trusts the job and simulation JSON, not the contents of referenced
coordinate, coordinate-system, surface, datum or elevation files. Re-run init
after editing such a file in place. See the runtime guide's shared simulation
caches for details.

## MT layered-earth excitation

`impedance_sources` also accepts `_type: "MTLayered"`, `background`, and `waves`.
The background contains `surface_m` (default zero) and a top-down `layers` list.
Each layer requires exactly one of `conductivity_s_m` and `resistivity_ohm_m`.
Finite layers require positive `thickness_m`; the final half-space must omit it.
Relative permittivity/permeability default to one. Optional Cole–Cole
resistivity parameters are `chargeability` in [0,1), positive `time_constant_s`,
and `exponent` in (0,1], defaulting to 0, 1, and 1. Background values are SI.

Waves contain one-based `rhs`, horizontal solver-basis `electric`, and optional
`source`, `amplitude` (surface E in V/m, default 1), and `phase` (degrees).
Polarizations are normalized. The runtime manages RHS scaling and rejects
`rhs_normalization`. Use two independent polarizations to extract MT responses.
The provider supplies total layered background fields on impedance boundaries,
with vacuum continuation above the surface. Depth is axis 2 in 2D or axis 3 in
3D; cylindrical geometry and nonzero strike wavenumber are unsupported.
The actual FEM model is specified independently. See `examples/mt-layered.json`
and the Maxwell `MT.md` guide for station conventions and output extraction.


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
shared source convention: it multiplies the default 1e9 N m moment. Omitted
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

### Point-source loading of forces and moments

Elastic and poroelastic point sources load the momentum equation in every
formulation. A `vector` source is a force `F` (N): the momentum test receives
`F . w(x_s)`. Moment-type sources are equivalent body forces
`f = -div(M delta(x - x_s))`: the momentum test receives `M : grad(w)(x_s)`,
each symmetric entry `M_ij` entering once per `(i, j)` pair, with no shear
doubling. The load is therefore independent of material properties, of the
frequency, and of the runtime reference scales; DPG velocity-stress and
displacement Galerkin solves of the same source agree (DPG velocity is
`i*omega` times Galerkin displacement under `exp(+i*omega*t)`). No source loads
the constitutive (stress) test.

- `tensor`/`moment_tensor`: `M` is the authored physical moment tensor (N m).
  `M_ij > 0` on the diagonal is an outward force pair along axis `i`.
- `scalar`/`monopole` in elastic physics: the isotropic moment `M = +A I`, with
  `A` the authored moment (N m). A positive amplitude is an explosion:
  compressional, outward first motion, the same polarity as a positive acoustic
  `scalar`/`volume_injection` source (positive volume injection, positive
  pressure). Galerkin and DPG use this sign.
- `scalar`/`monopole` in poroelastic physics: the same isotropic moment
  `M = +A I` (N m) acting on the total (bulk) stress, i.e. the body force
  `-A grad(delta)` on the total-momentum equation; it is not a pore-fluid
  injection. Poroelastic moment sources (`scalar`, `monopole`, `tensor`) are
  rejected in the axisymmetric layout.
- `isotropic_moment`/`monopole` in frequency-domain acoustic physics: the same isotropic
  moment `M = +A I`. In a fluid it is the stress glut `-A delta I`, equivalent to
  the volume injection `Q = i*omega*A/kappa(x_s)` with the physical, attenuated
  bulk modulus at the source point (`kappa = rho c^2`; the 2.5D wavenumber term
  of the effective compressibility is excluded). First-order (DPG) solves load
  `i*omega*beta*A` on the pressure equation and pressure Galerkin loads
  `-omega^2*beta*A` (`beta = 1/kappa`). This load is material-dependent: its
  material-control derivative (`i*omega*A*dbeta/dm` for the parameter-space
  property, density, and Qp), frequency derivative, and source-location partial
  (`load * dlog(beta)/dx`) are differentiated analytically. An isotropic moment of
  amplitude `A` therefore radiates the same P wave whether it sits in a fluid or
  in a solid of the same bulk modulus in the limit `mu -> 0`.
- `isotropic_moment`/`monopole` in transient acoustic physics: the time-domain counterpart
  (`i*omega <-> d/dt` under `exp(+i*omega*t)`). With `A(t)` the authored moment
  times the `Time/source` signature, the source is the volume injection
  `Q(t) = beta(x_s)*dA/dt` (`beta = 1/(rho c^2)`, real: transient acoustic models
  are non-attenuating), and the pressure Galerkin stepper loads
  `dQ/dt = beta*d2A/dt2`, using the signature's analytic second derivative.
  Positive `A` is compression, as in the frequency domain, and it is the same
  physical source as the transient elastic `scalar` (`M = +A I` scaled by the
  signature value), so a coupled transient model is consistent across the
  fluid-solid interface. Step signatures are rejected in models with acoustic
  physics (their second derivative is impulsive); a constant signature radiates
  nothing. Unlike `volume_injection`, an initially quiet signature is not
  enforced: a signature that is not quiet at the start time is truncated there.
- `gradient`: `M = -A I` (the force `A grad(delta)`); `dipole`: `M = -A d d^T`.
- Equivalent-source (Cauchy sheet) and point loads share one RHS normalization:
  equivalent-source loads are divided by the active RHS's point-source
  `rhs_normalization` before the solve, and outputs multiply it back, so an
  equivalent-source response depends only on its incident-field amplitude
  (in solver units, which the default scaling keeps independent of the
  adaptation frequency `Mesh/adapt/f_adapt`) and never on a placeholder point
  source. `Mesh/adapt/f_low` is only the lower edge of the frequency band.
- 2.5D: the out-of-plane derivative of the test is the formulation's algebraic
  factor `i*k_y`, so `M_iy` loads the momentum value and `M_ix`, `M_iz` load the
  in-plane gradient. All six components are used.
- Ordinary planar 2D keeps the shared omitted-thickness convention: force and
  moment kinds are distributed over 1 km (a 250 kN force is a 250 N/m line force);
  acoustic `volume_injection` uses 1 m. 2.5D, cylindrical, and 3D sources are not
  rescaled.
- Unsupported layouts fail instead of producing a wrong amplitude: elastic
  `scalar`, `monopole`, and `tensor` sources in the axisymmetric and torsional DPG
  layouts, poroelastic `scalar`, `monopole`, and `tensor` sources in the axisymmetric layout, and
  `scalar`/`monopole`/`tensor` sources in Willis cells. Axisymmetric Galerkin
  supports them: isotropic kinds include the hoop moment `M_theta_theta = M_rr`,
  while authored tensors have `M_theta_theta = M_yy = 0` because planar mechanisms
  reject omitted-axis components. Exact source-location
  derivatives of gradient loads (elastic moment kinds) are unsupported and fail
  explicitly.

A point on an element boundary divides its load among the incident elements whose
physics accepts the source kind: acoustic accepts `scalar`, `monopole`,
`volume_injection`, `vector`, and `dipole`; elastic accepts `scalar`, `monopole`,
`vector`, `tensor`, `gradient`, and `dipole`; poroelastic accepts `scalar`,
`monopole`, `vector`, and `tensor`. A `tensor` source on a conforming
fluid-solid interface therefore acts on the solid in full, while a `vector` or
`scalar` source there is shared between both sides. Mapping fails when no
element containing a point accepts its kind; a `volume_injection` point outside
every acoustic element reports that it must lie in an acoustic region.
Receivers apply the same physics-compatible incidence rule.

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

Inline event strength overrides inherit the default source shape. A source-local
mechanism or explicit kind/direction replaces the default shape as a unit; it
is not recursively merged with a different mechanism. HDF5 and SPS may share a
local mechanism, but its Cartesian tensor is evaluated separately at every
source location. See [the microseismic example](examples/microseismic.json).

### Explicit finite-current reference example

The [finite-wire example](examples/finite-wire.json) represents Gaussian line
quadrature using existing `electric_current` point loads and a `Named` encoding
with one transmitter RHS. Point moments retain signed current times oriented
length in A*m; coordinates retain their authored SI units. Closed-loop compilers
use electric-current segments. They must not substitute an impressed magnetic
current without its physical conversion. Quadrature refinement and field accuracy
remain the caller's responsibility; this does not add a native `wire` or `loop`
source kind. See [instrument composition](../../../docs/imaging/em-operators.md)
for independent reference calculations. Production wires and loops should use
the retained line-current source form below.

## Compact Gaussian source supports

An Inline source may set `support.type="gaussian"`, with positive physical
`sigma: {value, units}`. This preserves one physical source identity and its
signature/encoding. The existing amplitude remains the integrated strength.
The radial Gaussian has a C2 taper to zero at `sigma`; samples outside one sigma
are excluded. `cutoff` is fixed at 1. By default, positive quadrature samples
regularize the source: reference boxes subdivide to at most half `sigma`, with
at least three points per axis and higher order where the source basis needs it.
Supplying `tolerance` selects adaptive integration instead; `boundary="reject"`
also uses adaptive integration, with a default tolerance of `1e-4`.
Only samples in the selected domain contribute to the global normalization, so
an interface source restricted to one side retains its full integrated force.

The backend integrates on owned physical elements, subdivides reference cells
for narrow kernels, and normalizes collectively across MPI ranks. PML cells,
nonaccepting physics, and cells outside `source_geometry.domain` are excluded.
`boundary="truncate_renormalize"` preserves integrated strength after clipping;
`boundary="reject"` requires the integrated volume support to fit within
quadrature tolerance. Empty/negligible overlap fails. Logs report retained mass
and centroid displacement; clipping can shift the centroid.

Transient elastic and acoustic loads retain their unweighted spatial covectors
for a fixed assembly and encoding. Resident history scatter retains elastic and
acoustic source components on the device and applies the signature and its time
derivatives there. Evolving attenuating acoustic moments are staged each step.
`FS_SOURCE_CACHE=0` disables spatial covector reuse on host and device for
quadrature-reference checks. The source benchmark in
`test/benchmarks/benchmark_gaussian_sources.py` compares cached and reference
fields on identical meshes, including domain-restricted interface sources.
Profiling separates `source/quadrature` and `source/build`; `source/device` and
`source/host` count resident source routes, while recovery counters expose CPU fallback.

Optional `support.surface` selects a named exterior elastic free/Neumann surface
for a Gaussian traction. Surface integration uses physical surface measure and
requires `truncate_renormalize`; it is distinct from a volume force clipped at
that surface. `support` and the older uniform `footprint` are mutually exclusive.

Supported workflows are Cartesian 2D/3D acoustic and elastic forward simulations,
including frequency-domain and transient Galerkin loads. Source-control and
inversion workflows and cylindrical/2.5D supports are rejected explicitly.
GSLS acoustic isotropic moments use each accepted sample’s material response,
including supports crossing attenuation interfaces. Shared causal pole histories
preserve this response across checkpoint/restart. Force, pressure-forcing, and
volume-injection supports retain their existing temporal conventions. In planar
2D the existing invariant-thickness convention remains unchanged. Omitting support preserves
Dirac loading. See `test/e2e/gaussian_source_case.py` for clipping and MPI checks.

## Mesh-fitted vector-force footprints

An individual Inline vector source may set `footprint.surface` to an existing
named mesh boundary surface. `footprint.quadrature_order` defaults to 4 and is
an integer from 1 through 8. Declare footprints on individual sources, not on
`source_geometry` or its defaults. HDF5 and SPS coordinate catalogs do not define
footprints. Sources without footprints retain their existing point behavior.

The surface defines the complete support; source coordinates remain a reference
location in the mesh. With a unit `direction`, physical `amplitude` is total force
in N, distributed uniformly over the selected surface and normalized by its
quadrature measure. Direction retains its Cartesian vector meaning; it is not
inferred from the surface normal. The existing 2D invariant-thickness convention
still applies. A surface label does not assign a boundary condition.

Supported runtime combinations are Cartesian 2D/3D elastic and elastic2 DPG
forward frequency solves and initialization/sizing. All selected faces must be
exterior, non-PML, and free/Neumann in all traction components. Unknown/empty
sets, internal faces, constrained/impedance faces, Galerkin, other physics,
transient solves, and inverse/control workflows are rejected. Geometry and MPI
partition changes rebuild owned-face quadrature with collective normalization.

Each footprint remains one physical source for signatures and encoding:
`C = E diag(q)`. Signature IDs refer to physical sources, never quadrature points.
This supports prescribed plate force spectra; it does not predict contact force
from a pilot, plate mechanics, lift-off, or coupled harmonics. See
[the example](examples/elastic-force-footprints.json) and
[the source guide](../../../docs/imaging/sources.md#prescribed-plate-force-footprints).

## Retained line-current sources

`source_geometry._type: Quadrature` accepts named Polyline wires or closed loops.
Each item has `geometry`, `amplitude: {value, units}` and an optional current-kind
override. Electric current uses A; impressed magnetic current uses V. One catalog
entry and encoded source signature represents each complete instrument.
Segments are integrated cell by cell against Maxwell test functions, with the
authored quadrature order as a minimum. Cartesian 3D affine cells are supported;
curved cells and positive-length coincidence with a cell face are rejected.
An open wire has endpoint current divergence; a closed loop has none. This is
an impressed filament, not a finite-radius conductor or contact-impedance model.
See [native instruments](../../../docs/imaging/native-instruments.md) and
[example](examples/native-instruments.json).

### Poroelastic plane-wave verification source

`PoroelasticPlaneWave` requires `frequency_hz`, an isotropic `medium` in SI units,
and `waves`. Each wave selects `fast_p` or `slow_p`, a Cartesian direction, RHS,
optional origin, pressure amplitude in Pa, and phase in degrees. Origins retain
standard acquisition length-unit semantics. Boundary-only acquisition derives
logical sources from contiguous RHS indices; the default source names are
`poroelastic_plane_wave_<rhs>`. All fields must fit in one acquisition batch.
The declared medium must match the homogeneous model. Drained bulk modulus must
be smaller than grain bulk modulus. The runtime requires a matching positive
real frequency and ordinary Cartesian poroelastic propagation; it rejects
axisymmetric and 2.5D modes. Frequency/material derivatives of this incident
source are not supplied. See [validation scope](../../../docs/imaging/poroelastic-validation.md)
and the [example](examples/poroelastic-plane-wave.json).

The [calibrated point survey example](examples/calibrated-point-survey.json)
shows simultaneous source signatures and receiver component responses; supply
the referenced HDF5 tables at the actual task coordinates.

Experimental patch execution may supply `active_sources`, a nonempty unique list
of physical source IDs. The full physical catalog and canonical source controls
remain unchanged; only selected source batches execute. Source encoding and
boundary loadings are unsupported with this selection. Receiver surveys must
restrict observation rows consistently; source selection alone does not alter
a receiver layout or objective normalization. Validation happens before any
solve: a sparse survey row with nonzero weight whose source is not selected is
rejected when the layout is prepared, and an objective bound to a dense receiver
group is rejected when the objective is configured.

## Acoustic pressure-wavelet monopoles

Use `pressure_monopole` for a wavelet applied directly to the pressure equation:
`beta*p_tt - div(rho^-1 grad(p)) = S(t)*delta`. Amplitude `S` defaults to
1 m^3/s^2; planar 2D spreads it over a fixed 1 m invariant thickness. In a
homogeneous 3D medium (including an on-axis axisymmetric source), the direct
pressure pulse preserves the signature up to delay and geometrical spreading.
Planar 2D propagation and heterogeneous media can change the received shape.

Transient Galerkin loads the signature value. Frequency pressure Galerkin loads
`S`; first-order formulations load `S/(i*omega)` and require nonzero frequency.
The source is fluid-only, including coupled models. Explicit `volume_injection`
continues to describe a volume rate (m^3/s), and explicit `isotropic_moment`/legacy `monopole`
continue to describe an isotropic moment (N*m); those signatures produce first
and second time derivatives in the transient pressure equation respectively.

In acoustic and coupled simulations, `scalar` is an alias for `pressure_monopole`
in both transient and frequency-domain workflows. Its amplitude is now m^3/s^2
(default 1). Migrate earlier acoustic `scalar` sources with N*m amplitudes to
`isotropic_moment` to retain their original physical behavior. Elastic-only
`scalar` and explicit isotropic mechanisms remain moment sources.

### Native transient convolution arrays

Static source encodings may include `convolution`, using the shared
`convolutionFilters` definition. Source shape is `(encoding, source, tap)`;
`EncodedReceiver.device.convolution` uses `(encoding, component, receiver, tap)`.
Taps are finite real dimensionless FIR coefficients. Tap `k` multiplies the
input at `t - k*dt`; no `dt` factor is applied. Existing real encoding weights
multiply taps. Inline `taps` are flattened in C order. HDF5 stores
`(channel, element, tap)`, flattening encoding/component into channel.

These operators require a native transient workflow and a single source batch.
Source delays are evaluated at the method's physical forcing times, including
derivative channels. Receiver `dt` must equal the integration timestep; dense
`sum`/`mean` receivers advance on every accepted step independently of output
cadence, starting with zero prehistory. Simulation duration controls tail capture.
Vector/tensor component outputs share the component's FIR. Receiver checkpoints
retain pending outputs and require the same receiver definition and partition when restored.
Filtered acoustic isotropic moments with GSLS are rejected. Spectral source
signatures, frequency-dependent source weights, non-real encoding/mechanism
weights, and spectral receiver transfer functions are invalid for transients.
