# FS Material Model Contract v1

Status: initial
Visibility: public
Schema id: `fs-material-model-1`

## Summary

Material model configuration assigns physical properties to mesh block
subdomains. Each subdomain binds one `mesh_block_id` to a set of scalar,
file-backed, or expression-derived properties and an optional material physics
kind.

The material model is part of the Core API because physics setup, element
assembly, imaging property requests, and support diagnostics all rely on the
same property catalog and domain-to-material mapping.

## Versioning

- A material block with `"schema": "fs-material-model-1"` is interpreted by this
  contract.
- Current Fortran readers consume `subdomains`. The schema keeps `layers` only as
  a deprecated alias for staged tooling; callers that hand JSON to current solver
  readers must normalize `layers` to `subdomains`.
- Additional fields are allowed so applications can stage problem-specific
  material metadata without changing this core shape.

## Subdomain Contract

- `subdomains` is an array of material subdomain objects.
- `mesh_block_id` is required and maps the material subdomain to a mesh region.
- `name` is optional diagnostic text.
- `physics` may name a registered physics family or variant such as `acoustic`,
  `elastic:iso`, `elastic:viscous_fluid`, `elastic:vti`, `elastic:tti`, `em:plasma`,
  `em:geophysics`, or `em:vacuum`.
- If `physics` is omitted, the active property set is passed to the physics
  catalog for inference. Ambiguous or invalid property sets are validation errors.
- `properties` is required and maps catalog property names to property values.
  The only non-property key accepted inside this object is `grid`, which supplies
  shared metadata for file-backed binary properties.
- New material examples should use the canonical lowercase seismic names `vp`,
  `vs`, `rho`, `qp`, and `qs`. Current legacy inputs may still use the mixed-case
  aliases `Vp`, `Vs`, `Rho`, `Qp`, and `Qs`.

## Layered Surface Geometry Contract

`Model/surfaces` defines the ordered graph surfaces used by
`LayeredMeshGenerator`. Surface entries are listed in formation order, normally
from top to bottom. Each regular surface has a `depth` length property. The
runtime accepts scalar lengths, `{ "value": ..., "units": ... }` objects,
file-backed xarray properties, and inline xarray-style properties with `dims`,
`coords`, and `data`.

A fracture surface is a compact way to describe a thin layer whose aperture may
open and close independently of the surrounding model surfaces. A surface entry
is treated as a fracture when `_type` is `Fracture`.
Fractures require both:

- `depth`: the center surface.
- `gap`: the geometric aperture, using the same length-property forms as
  `depth`.

The layered mesh expands a fracture into two generated graph surfaces,
`depth - 0.5 * max(gap, 0)` and `depth + 0.5 * max(gap, 0)`. Zero or negative
gap values collapse the fracture back to the center surface. If `name` is
provided, generated surfaces are named `<name>_top` and `<name>_bottom`; callers
may override these names with `top_name` and `bottom_name`. Generated `surface_N`
aliases count the expanded surface list, so one fracture consumes two aliases.

`gap` is geometry only. It specifies the width/aperture of the opened fracture;
it does not contain fluid or material properties. To assign the material inside
the opened fracture explicitly, set `mesh_block_id` on the fracture entry. That
id must match `subdomains[*].mesh_block_id`; the referenced subdomain is
reserved for the fracture interval and is not consumed by the ordinary formation
layer sequence. If `mesh_block_id` is omitted, the opened fracture interval is
assigned by ordinary layered formation order for compatibility.

A `LayeredModel` surfaces list may also carry implicit-only entries from
`fs-implicit-geometry-1` (`rbf`, `rbf_level_set`, `plane`, `sphere`, Boolean
expressions, ...). An entry without `depth` whose `_type` is not `elevation` or
`fracture` is not a horizon: formation order, `surface_N` aliases, `interface`,
layer assignment, PML handling, and guided-adaptivity surface ids count graph
surfaces only. The implicit registry still numbers every entry by list position,
so blends, `regions`, and coordinate systems resolve the entry by `name` or
signed id, and an `rbf_level_set` `control` publishes `model.<id>` inside the
layered model. See [layered rbf control](examples/layered-rbf-control.json).
The 3D layered generator requires implicit-only entries to follow every graph
surface.

## Multi-region Geometry Contract

`Model/regions` is the opt-in material-volume arrangement used by geometry
queries and material classification. Each entry has a stable `name`, a positive
`mesh_block_id`, an integer `priority` (default `0`), and a recursive
`geometry` expression. An implicit leaf selects a named or signed numeric
surface with `phi < 0` inside. Provider/BREP leaves name a registered provider
and closed volume or shell. Hard `union`, `intersection`, and `difference`
nodes recursively combine leaves.

At a point contained by multiple raw regions, the highest integer priority
owns the material. Equal-priority interior overlap between different regions
is invalid, while coincident shared boundaries are valid material interfaces.
The physical domain is the union of effective regions; points outside every
region are exterior. Region expressions used for bounded-domain workflows must
be bounded.

For this opt-in path, `Model/surfaces` may contain named primitives and
Boolean expressions from `fs-implicit-geometry-1`. The graph-surface form
remains the `LayeredMeshGenerator` contract; a layered model may mix both kinds
in one list as described above.

Fracture surfaces may override global Krauklis controls with
`adapt/krauklis`. The compact normal fields are `normal/h0`,
`normal/thickness`, and `normal/grading`; `h0` is the wall-adjacent target
element size divided by the selected guided-mode decay length. Tangential
fields are `tangential/factor`, `tangential/thickness`, and
`tangential/grading`. They control the wall EPW multiplier, the number of decay
lengths over which wall-parallel sizing is propagated, and the power-law return
to ordinary sizing.

Fluid-solid interface surfaces may similarly override global Scholte controls
with `adapt/scholte`. The compact fields have the same meanings as for
Krauklis. `normal/factor` limits the finest wall-normal size relative to
Scholte-wave EPW sizing and defaults to `4`; the normal profile still coarsens
to the local body-wave EPW size on both sides of the interface. A normal `h0`
of `0` disables the normal profile, while a tangential `thickness` of `0` keeps
only wall-local tangential sizing.
`enabled` disables both Scholte profiles when false. `lower_bound` floors the
Scholte velocity used for sizing and accepts either solver units or a
`{ "value": ..., "units": ... }` velocity object.

The legacy flat fields `normal_h_over_delta`, `normal_thickness`,
`normal_grading`, `tangential_thickness`, `tangential_grading`, and
`tangential_epw_mult` remain accepted for compatibility. On fracture surfaces
they control Krauklis sizing. Legacy Scholte objects at `Model/scholte` and
`Model/surfaces[*]/scholte` also remain accepted. Compact controls take
precedence over legacy controls at the same scope, and surface-local controls
take precedence over global controls.

`Model/gravity` sets gravitational acceleration for the acoustic UW-DPG
gravity free-surface operator and its mesh-sizing model. It defaults to standard
gravity, `9.80665 m/s^2`. A numeric value is already in solver acceleration
units; a `{ "value": ..., "units": ... }` object is nondimensionalized by the
active unit runtime.

Finite-depth gravity-wave sizing is opt-in under
`Mesh/adapt/gravity_surface` or
`Model/surfaces[0]/adapt/gravity_surface`. The top surface bounds the acoustic
water formation and the next regular interface supplies local water depth.
Tangential wavelength follows `omega^2 = g k tanh(k h)`; no separate mesh or
resolution representation is introduced.

## Borehole Geometry Contract

Boreholes keep geometry separate from material definition. Borehole layers
reference existing `subdomains[*].mesh_block_id` values; they do not define
properties inline. During layered mesh generation, cells whose centroid falls
inside a borehole layer use that layer's referenced mesh block id instead of the
formation layer id.

Current solver support covers 2D `LayeredMeshGenerator` meshes and the initial
3D layered path for vertical boreholes. A borehole has a vertical axis, an
`extent` bounded by `Model/surfaces`, concentric material `layers`, ordered
radial mesh `surfaces`, and optional local axial `plugs`. The 3D path requires
`axis/y` and does not yet support plugs:

- `name` identifies the borehole and is used by mesh-spacing controls.
- `angular_split` optionally requests forced angular h-refinement passes on
  this borehole's hexahedral material-layer and annular-padding cells after 3D
  layered mesh generation and before wavefield adaptivity. The root sketch
  remains four angular sectors; `angular_split: 1` yields eight hexahedral
  angular sectors, `2` yields sixteen, and so on. Prism core splitting is not
  seeded by this control.
- `axis/x` is a length quantity. In 3D, `axis/y` is also required. The current
  reader also accepts `x` and `y` as compatibility shorthands.
- `extent/top` and `extent/bottom` reference model surfaces by name, generated
  fracture boundary name, one-based expanded-surface index, or aliases `top`,
  `bottom`, and `surface_N`.
- `layers[*].mesh_block_id` must match a material subdomain in `subdomains`.
- `layers[*].inner_surface` and `layers[*].outer_surface` may name the radial
  walls bounding the layer. If omitted, the first layer starts at the axis, each
  later layer starts at the previous layer's outer surface, and each layer uses
  the surface at the same array index as its outer wall.
- `surfaces[*].name` identifies a radial wall within the borehole. Runtime
  surface lookups use `<borehole>_<surface>` unless the surface name is already
  globally qualified.
- `surfaces` may include additional ordered radial mesh walls inside a material
  layer. When a layer's `outer_surface` skips over intermediate surfaces, those
  intervals are generated with the same `mesh_block_id`; this is analogous to an
  `interface=false` surface in a layered formation.
- `surfaces[*].r` is the cumulative radius for that wall. The first layer starts
  at `r=0`; each following layer starts at the previous wall radius.
- `surfaces[*].r` may be a scalar length quantity, a file-backed xarray
  property, or an inline profile with `value`, `dims`, and `coords`. In 2D,
  radius profiles are evaluated at cell-centroid depth. In 3D, radial sketch
  loops sample the radius as a function of polar angle and depth.
- In 3D, `surfaces[*].type` may be `cylinder` or `simple`. Scalar-radius walls
  default to implicit cylinders; non-scalar radial profiles default to
  `SimpleSurface_t` walls in borehole-local cylindrical coordinates.
- 3D `type: simple` walls may also provide a Simple-surface `function` with
  `axis`, `coord_system`, `periodic`, and `period` fields instead of `r`.
- `plugs[*].mesh_block_id` must match a material subdomain in `subdomains`.
- `plugs[*].top` and `plugs[*].bottom` are length quantities in model depth
  coordinates. Plug intervals in the same borehole must not overlap.
- `plugs[*].r` is the cumulative radial extent of the obstruction from the
  borehole axis. It may use the same scalar, file-backed, or inline profile
  forms as a borehole wall radius.
- `nearfield/h0` and `nearfield/growth` optionally define a Stoneley
  wall-normal nearfield adaptivity envelope from the borehole wall signed
  distance. `h0` is a wall-adjacent target normal mesh size; numeric values use
  the material model length scale unless units are supplied. `h0` defaults to
  `1 m`, and `h0: 0` disables the nearfield envelope. `growth` is a continuous
  geometric growth factor and defaults to `2.0`.
- `adapt/stoneley` may be supplied on a borehole or an individual borehole
  layer. It uses `normal/{h0,thickness,grading}` and
  `tangential/{factor,thickness,grading}` like the other guided-wave families.
  Layer values override borehole values, and both override global
  `Mesh/adapt/stoneley` values.
- The legacy flat normal and tangential fields remain accepted on boreholes and
  borehole layers. Compact controls take precedence at the same scope; layer
  controls take precedence over borehole and global controls.

Formation layers are still assigned from the non-borehole material subdomains in
order. Material subdomains referenced by borehole layers are reserved for those
layers and do not consume formation layer slots. Material subdomains referenced
by borehole plugs are reserved the same way. The 2D layered generator uses a
vertical x-grid, so variable-radius layers are applied as centroid material
overrides; `horizontal_spacing/include_borehole_edges` adds scalar/profile/xarray
radius breakpoints to the x-grid rather than exact curved borehole boundaries.
The 3D layered generator inserts one square embedding box per borehole, builds a
radial sketch inside that square, then fills the surrounding formation with
explicit plan-view quads and direct connector quads between neighboring
boreholes rather than a tensor-product x-y grid. Those plan cells are lofted
through the model surface stack. Within 2D plug depth intervals, generated
borehole cells from the axis to `plugs[*].r` use the plug `mesh_block_id`; any
remaining annulus keeps the underlying borehole layer domain.

## Property Contract

Physical material properties have default input units. An explicit property-level
`units` string must have a compatible physical dimension; incompatible units
are rejected for active properties, including file-backed and expression
properties. Unused properties belonging only to inactive physics remain accepted
without loading or validating their values. Dimensionless quantities accept
unlabelled values or `"units": "1"`.

Elastic and poroelastic orientation angles `phi` and `theta` default to radians.
For example, `"phi": {"value": 30, "units": "deg"}` evaluates to pi/6 radians.
This convention also applies when solver scaling is disabled. Unlabelled angles
retain their existing radian interpretation.

Each property value may be one of these forms:

- Inline scalar or vector quantity, such as `"vp": 1.5`.
- Explicit scalar property object, such as `"rho": { "value": 2.2,
  "units": "g/cc" }`.
- File-backed property object with `file` and optional `format`, `absolute`,
  `scale`, and `grid`.
- Expression property object with `expr`, optional `depends_on`, optional
  `units`, and optional final `scale`.
- Parameterized property object with one stable block `id`, an ordinary
  `reference` property, an `identity`, `log`, `inverse`, or `logit` transform, and a sparse
  coordinate or mesh `control` map. Coefficients are ordered within the block;
  mesh controls additionally have stable integer IDs within their property space.

The transforms keep the controlled material property independent of its
physics parameter space. If `r(x)` is the reference and `c(x)` is the control
map, `identity` evaluates `r+c`, `log` evaluates `r exp(c)`, and `inverse`
evaluates `1/(1/r+c)`. The inverse form therefore controls the reciprocal of the
named physical property without introducing separate catalog properties or
combined physics parameter spaces. In particular, `transform: "inverse"` on
`qp` or `qs` provides additive inverse-Q controls with physical-property chain
rule `dQ/d(1/Q) = -Q^2`. Inverse transforms require a strictly positive
reference and updates that keep `1/r+c` positive.

`logit` evaluates `sigmoid(log(r/(1-r))+c)` for a dimensionless reference
strictly inside `(0,1)`. Its control derivative is `p(1-p) dc`; spatial gradients
also differentiate the reference. This supplies bounded native chargeability
and IP exponent controls; use `log` for the positive time constant. Zero control
reproduces the reference exactly, and no units rescaling other than one is
accepted. Updates that round to an endpoint in floating-point arithmetic are
rejected, not clipped. Fixed endpoints such as zero chargeability may still be
ordinary properties. Scalar, Cartesian batch, query, bounds and control
JVP/VJP paths use the same transform. See
[bounded IP controls](examples/bounded-ip-controls.json).

- Blend property object with one named implicit `surface`, a positive `width`,
  and ordinary `inside` and `outside` endpoint providers. Negative surface phi
  selects `inside`; positive phi selects `outside`. A cubic C1 smoothstep spans
  the centered interval `[-width/2, width/2]`. A blend decorates one property in
  one material subdomain and does not create or classify material layers.

For a blend weight `w(phi)`, the exact control derivative is
`dm = (1-w) dm_outside + w dm_inside + (m_inside-m_outside) w'(phi) dphi`.
This includes both endpoint-property controls and active implicit-surface
controls in native Born and RTM sensitivity paths. PML material blending is a
separate compatibility mechanism and is not applied to blend endpoints.

Parameterized controls may use an arbitrary-degree `bspline` map with an
explicit knot vector. The compact `hat` map represents a uniform nodal grid:

```json
{
  "kind": "hat",
  "coordinate_system": "global",
  "axis": "z",
  "origin": 0.0,
  "spacing": 0.25,
  "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0]
}
```

Coefficient `i` is the model update at `origin + (i-1) * spacing`; adjacent
values are joined by piecewise-linear hat functions. The update and its
sensitivity are zero outside the first and last control locations. `origin`
defaults to zero. Knot, origin, and spacing values use model coordinate units by
default; an optional `units` string may declare another compatible length unit.
They are converted to the active solver coordinate scale exactly like Xarray
axes. The enclosing parameterized-property `id` addresses the whole
coefficient vector in control HDF5 files and remains necessary when a model has
multiple controlled properties or layers.

A mesh control refers to a named definition in `Model/property_spaces`:

```json
{
  "property_spaces": {
    "materials": {"frequency": 10.0, "epw": 2.0, "artifact": "material-controls.h5"}
  }
}
```

Use `"control": {"kind": "mesh", "space": "materials"}` inside a parameterized
property. The first use generates a continuous first-order nodal basis from the
initial geometry and material reference state. Material interfaces have separate
controls. Tensor-product cells use multilinear hats. Frequency is in Hz; EPW may
be scalar or have one entry per spatial dimension. Solution frequency, acquisition
grading, and solver refinement settings do not size this hierarchy.

The artifact freezes topology and coefficient IDs. Later frequencies and material
linearizations reuse it. An existing artifact must match the initial topology,
material groups, and control sizing. Select a new artifact explicitly to start a
new space; transferring optimization vectors between spaces is not automatic.
Controls initially vanish, reproducing the ordinary reference property. Named
coefficient checkpoints carry the frozen basis identity and are read by local
coefficient slices. Inline global coefficient arrays are not accepted for mesh
controls. The usual identity, log, inverse, and logit transforms apply unchanged.

Only roots intersected by a subdomain, including physical roots needed for PML
extension, are retained locally. PML cells introduce no independent controls.
Mesh-specific smoothing is unavailable; requests for a nontrivial mesh-control
regularizer are rejected. Coordinate-control regularizers remain available.

For scalar and file-backed objects, `scale` multiplies the catalog scale for that
property. For expression objects, `scale` is a final multiplier on the root
expression result. Public material JSON should prefer explicit units or
catalog-scale values over hidden unit assumptions.

Properties recognized by an inactive physics family are accepted for domain
matching and scoring, but their values are not evaluated.
For example, an acoustic simulation may use a seismic model that also supplies
`vs`, `qs`, or elastic anisotropy properties. All supplied symbols participate
in material matching, scoring, and unsupported-property diagnostics, including
symbols belonging to inactive families. An explicit simulation physics still
permits a supported subset of the model properties; unused file-backed data is
not loaded. Definitions from
active physics families govern shared property names; incompatible definitions
from simultaneously active families are errors.

Unknown keys under `properties` are input errors. Current readers do not treat
descriptive poroelastic names as aliases: use `k_dry`, `mu_dry`, `k_solid`,
`k_fluid`, `rho_solid`, `rho_fluid`, `kappa`, and `viscosity`, not
`drained_bulk_modulus`, `shear_modulus`, `solid_bulk_modulus`,
`fluid_bulk_modulus`, `solid_density`, `fluid_density`, `permeability`, or
`fluid_viscosity`. The catalog name `permeability` is the EM magnetic
permeability property; poroelastic hydraulic permeability is `kappa`.

Acoustic and elastic properties may instead select a parameter space
with the case-insensitive short names `Sp`, `Ss`, `K`, `Ks`, `Kp`, `lambda`,
and `mu`. Accepted primary tuples are `(Sp,rho)`, `(K,rho)`, `(Sp,Ss,rho)`,
`(K,Ks,rho)`, `(Kp,Ks,rho)`, and `(lambda,mu,rho)`. The existing velocity
tuples `(vp,rho)` and `(vp,vs,rho)` remain supported. TI/VTI and TTI accept
all five elastic tuples with Thomsen `epsilon`, `gamma`, and `delta`; TTI also
accepts orientation angles. The reference moduli mean `Kp = rho*vp²`,
`Ks = mu = rho*vs²`, `K = Kp - 4*Ks/3`, and `lambda = Kp - 2*mu` in the
material symmetry frame, before attenuation. For anisotropic media, `K` is a
reference-coordinate modulus rather than an isotropic bulk response of the
full tensor. Density derivatives hold the selected tuple's other coordinates
fixed. Do not mix primary tuples in one material layer.

## Attenuation Configuration

Seismic Q attenuation is configured once for the model rather than per layer:

```json
"attenuation": {
  "model": "kjartansson",
  "reference_frequency": 10.0
}
```

`model` is matched case-insensitively and currently accepts `kjartansson` or
`none`. `reference_frequency`, `f0`, and `f_ref` are mutually exclusive aliases.
Each accepts either a positive bare scalar in Hz or a unit-bearing scalar such
as `{ "value": 0.01, "units": "kHz" }`. The reference defaults to 10 Hz and
identifies the frequency at which the real-valued material parameters are
defined. Layer properties continue to provide `qp`, `qs`, `qk`, and `qmu`.
Selecting `none` ignores those Q values but does not disable poroelastic JKD
hydraulic dispersion.

Subdomains may also define a sibling `fields` object for auxiliary
non-canonical data sources. `fields` keys are arbitrary user field names except
for the metadata key `grid`, and must not collide with catalog property names.
Auxiliary fields may be scalar values, parameterized providers, or file-backed xarray fields with the same
file, grid, interpolation, `fill_invalid`, `units`, `scale`, and path behavior as
file-backed properties. Auxiliary fields do not participate in physics
inference, are not assembled directly, and are evaluated only when referenced by
expression properties. Current runtime readers do not support expression-backed
auxiliary fields.

The seismic adaptivity reader also recognizes optional control properties:
`vadapt` replaces the wavespeed used for element-per-wavelength sizing,
`epw_mult` multiplies the requested EPW target locally, and `hmin` provides a
local minimum element size. `hmax` provides a local maximum element size that
can force refinement independent of frequency. `epw_mult` is clamped to at
least `1.0`, so it only requests equal or finer EPW sizing. `hmin` and `hmax`
are absolute length controls and are therefore not frequency-dependent.
Without `vadapt`, seismic EPW sizing uses `vp` for acoustic materials, `vs` for
elastic and elastic-frame poroelastic materials, and the minimum direct-Biot
body-wave speed for direct poroelastic materials.

File-backed properties use the xarray material backend. Relative paths resolve
from `FS_env%ProjectPath` unless `absolute` is true. New HDF5 references use a
plain `file` path and a separate authoritative `dataset` path. The runtime
continues to accept a legacy `file:dataset` locator when reading existing
inputs; `hash` is optional tooling metadata. For each name in an HDF5
property's `dims` attribute,
the same-named coordinate attribute may contain either the coordinate values
directly or one string naming a rank-one real dataset in the same HDF5 file.
A rooted dataset name such as `/coordinates/x` is resolved from the file root;
an unrooted name such as `coordinates/x` is resolved relative to the property
dataset's parent group. Referenced coordinate datasets may contain 32- or
64-bit reals and must have the corresponding property-axis extent. The
property dataset's `axis_units` attribute continues to supply coordinate units
for either representation. RSF properties use the `.rsf` header for axis
counts, origin, spacing, labels, units, and the `in=` payload path. DDS
properties use the `.dds` header for the same xarray metadata, accepting common
spellings such as `n`, `spacing`, `origin`, `axis1`, and `data_file`. The
runtime supports native float payloads with `esize=4`. Binary `.bin` properties
require a `grid` either on the property object or as `properties/grid` on the
enclosing subdomain properties object. Binary auxiliary fields use the same
rule with their own field object or `fields/grid`.

Xarray-backed material, surface, and borehole-radius properties may specify
`interpolation` as `linear` or `bezier`; omitted values default to `linear`.
File-backed xarrays also accept `fill_invalid`; the default `nearest` repairs
NaN/Inf samples by propagating the nearest valid grid value, and for `vs`-style
shear velocity properties also repairs non-positive samples. Use
`fill_invalid: "none"` to disable this repair and preserve strict validation.
They may also specify `valid_range` with optional `lower`/`upper` bounds
(`min`/`max` aliases are accepted); samples outside that range are treated as
invalid by the same repair and validation path. Bounds may be bare numbers or
scalar `{value, units}` objects when the repair threshold should carry explicit
unit metadata.
Stored HDF5 real datasets may be single or double precision. Runtime xarray
coordinate handling, interpolation, and evaluation are performed in double
precision.

Grid `dims` is an array of axis-name strings so non-Cartesian systems can name
axes such as `lon`, `depth`, `theta`, or `radius`.

Expression properties are lazy derived values. The runtime initializes scalar and
file-backed properties first, then parses `expr` trees, resolves `ref` nodes
against the same subdomain property catalog, rejects inactive refs,
self-references, and cycles, and activates expressions in dependency order before
physics inference. Expression refs return internal solver values and do not
materialize sampled fields. Batched property requests share a per-call layer
cache so canonical dependencies are evaluated once per point and expression
properties can depend on other expression properties.

### Differentiable material expressions

Native Maxwell DPG control actions can differentiate primary expressions through
parameterized primary properties and auxiliary `fields`. The control-derivative
subset is `value`, `ref`, `field`, `neg`, `add`, `sub`, `mul`, `div`, `exp`, `log`,
`convert` and `magnitude`, including recursively referenced expressions. Division
requires a nonzero denominator; logarithms require positive values. Other nodes
remain available for forward material evaluation but are rejected when active
controls require an unsupported derivative. Active auxiliary controls must reach
at least one supported Maxwell physical property; unused active fields fail
validation. This does not enable auxiliary controls for other PDE formulations.

The [Cholesky conductivity example](examples/cholesky-conductivity-controls.json)
uses three positive log-controlled diagonal fields and three signed identity
fields. Expressions form `sigma = reference L L.T` at material samples, producing
all six conductivity entries. Every latent block is registered once. Point/batch
values, spatial gradients and bounds use the existing expression evaluator;
Born/RTM actions use exact batched field-chain JVP/VJP. Unit conversions happen
at the ordinary expression/property boundary. Interpolating the Cholesky fields
before constructing the tensor preserves positive definiteness in exact
arithmetic; the native constitutive law still checks numerical admissibility.
The legacy single-block stencil accessor cannot represent this coupled map and
rejects active expressions; use the batched derivative interface.

Supported expression nodes are:

- `{"value": 0.25, "units": "km/s"}`: numeric literal, nondimensionalized at
  parse time when units are present.
- `{"ref": "vp"}`: same-subdomain reference to an active scalar, file-backed, or
  expression property.
- `{"field": "vp_orig"}`: same-subdomain reference to an auxiliary field from
  the layer `fields` object. Field refs are evaluated at the same query point as
  canonical property refs and do not affect physics inference.
- `{"var": "z"}`: reference to a top-level `symbols` binding. Current runtime
  bindings support coordinate variables such as
  `"symbols": {"z": {"kind": "coordinate", "system": "global", "axis": "z", "units": "m"}}`.
  If `axis` is omitted in a symbol binding, the symbol name is used. Coordinate
  variables evaluate at the same material query point as property refs.
  Surface-relative systems are supported after geometry coordinate bindings are
  resolved. Tangential axes are affine; the normal axis follows the bound
  surface and its gradient.
- `{"coord": {"system": "global", "axis": "z", "units": "m"}}` or
  `{"op": "coord", "system": "global", "axis": "z", "units": "m"}`: inline
  coordinate value. `coordinate` is accepted as an alias for `coord`.
- `{"op": "neg", "arg": node}`.
- Unary functions
  `{"op": "exp"|"log"|"log10"|"sqrt"|"abs"|"sin"|"cos"|"tan"|"tanh", "arg": node}`.
  `log` is the natural logarithm. `log`, `log10`, and `sqrt` use their standard
  positive-domain math behavior at point evaluation; interval bounds become
  conservative when an input range crosses outside the domain.
- Binary
  `{"op": "add"|"sub"|"mul"|"div"|"pow"|"min"|"max", "args": [left, right]}`.
- Predicate and logical operators
  `{"op": "<"|"<="|">"|">="|"=="|"!="|"and"|"or", "args": [left, right]}`
  return `1.0` for true and `0.0` for false. `and`, `or`, and `not` treat
  nonzero values as true.
- `{"op": "not", "arg": node}` returns `1.0` when the argument is zero and
  `0.0` otherwise.
- `{"op": "case", "branches": [{"if": condition, "then": value}], "else": fallback}`
  selects the first branch whose `if` expression is nonzero, otherwise evaluates
  `else`. Branches are evaluated lazily for point values and gradients; interval
  bounds use the conservative union of branch ranges.
- `{"op": "clamp", "args": [value, min, max]}` evaluates as
  `min(max(value, min), max)`.
- `{"op": "atan2", "args": [y, x]}` uses the standard two-argument arctangent
  convention and returns radians.
- `{"op": "convert", "arg": node, "units": "m/s"}` and
  `{"op": "magnitude", "arg": node, "units": "m/s"}`: fixed unit conversions at
  parse time. The top-level property `units` field nondimensionalizes the root
  expression result.

Example:

```json
{
  "vp": 2.0,
  "vs": {
    "expr": { "op": "mul", "args": [ { "value": 0.5 }, { "ref": "vp" } ] },
    "depends_on": [ "vp" ]
  },
  "rho": {
    "expr": {
      "op": "mul",
      "args": [
        { "value": 0.31 },
        {
          "op": "pow",
          "args": [
            { "op": "magnitude", "arg": { "ref": "vp" }, "units": "m/s" },
            { "value": 0.25 }
          ]
        }
      ]
    },
    "units": "g/cc",
    "depends_on": [ "vp" ]
  },
  "vadapt": {
    "expr": { "op": "add", "args": [ { "ref": "vs" }, { "value": 0.25, "units": "km/s" } ] },
    "depends_on": [ "vs" ]
  },
  "epw_mult": 1.5,
  "hmin": {
    "value": 0.025,
    "units": "km"
  },
  "hmax": {
    "value": 0.1,
    "units": "km"
  }
}
```

The xarray backend uses this coordinate contract internally:

- `x(D)` is the solver's internal coordinate storage used at query and mesh edges.
- `xc(3)` is an expanded Cartesian working coordinate.
- `q(3)` is the coordinate tuple in the property's selected coordinate system.

Readers and evaluators must explicitly expand/project between `x(D)` and
`xc(3)`; coordinate-system transforms operate only on `q(3) <-> xc(3)` and do
not infer behavior from array rank. The selected coordinate system may still be
3D when the solver is running in 2D; only `x(D)` is tied to solver dimension.

## Catalog Contract

Property names are meaningful only after the application registers and seals a
property catalog. The seismic application currently registers `vp`, `vs`, `rho`,
`qp`, `qs`, `Sp`, `Ss`, `K`, `Ks`, `Kp`, `lambda`, `mu`, `vadapt`,
`epw_mult`, `hmin`, `hmax`, `epsilon`, `gamma`, `delta`,
`phi`, `theta`, `electron_density`, `mag_x`, `mag_y`, `mag_z`, `conductivity`,
`resistivity`, `permittivity`, `permeability`, `k_dry`, `mu_dry`, `k_solid`,
`k_fluid`, `rho_solid`, `rho_fluid`, `porosity`, `tortuosity`, `kappa`,
`viscosity`, `bulk_viscosity`, `specific_storage`, `qk`, `qmu`, `viscous_length`, and
`biot_frequency`.

The alternate acoustic and isotropic-elastic parameters use these catalog unit
scales:

- `Sp` and `Ss`: `s/km`
- `K`, `Ks`, `Kp`, `lambda`, and `mu`: `GPa`

The EM material properties use these catalog unit scales:

- `electron_density`: `1/m^3`
- `mag_x`, `mag_y`, and `mag_z`: `T`
- `conductivity`: `S/m`
- `resistivity`: `Ohm*m`
- `permittivity`: `F/m`
- `permeability`: `H/m`

The poroelastic material properties use these catalog unit scales:

- `k_dry`, `mu_dry`, `k_solid`, and `k_fluid`: `GPa`
- `rho_solid` and `rho_fluid`: `g/cc`
- `kappa`: `D`
- `viscosity`: `cP`
- `viscous_length`: `m`
- `biot_frequency`: `Hz`
- `porosity` is the dimensionless pore-volume fraction, bounded from zero to one.
- `tortuosity` is a dimensionless inertial ratio with a minimum of one.
- Quality factors `qp`, `qs`, `qk`, `qmu` and Thomsen ratios `epsilon`, `gamma`,
  `delta` are dimensionless in every physics that accepts them.
- Orientation angles `phi` and `theta`: `rad` (explicit `deg` is also accepted).

Maxwell `chargeability` and `ip_exponent`, and the mesh control `epw_mult`, are
also dimensionless. Plasma `ion_mass` and `ion_charge` are dimensionless ratios
to the proton mass and elementary charge, respectively; they are not inputs in
kilograms or coulombs.

For poroelasticity, `kappa` is hydraulic permeability and `viscosity` is
pore-fluid dynamic viscosity. The legacy/descriptive names
`drained_bulk_modulus`, `shear_modulus`, `solid_bulk_modulus`,
`fluid_bulk_modulus`, `solid_density`, `fluid_density`, `permeability`, and
`fluid_viscosity` are not part of this contract.

Pressure-form Darcy layers use `physics: "darcy"` and require `kappa`,
`viscosity`, and `specific_storage`. `kappa/viscosity` forms the heterogeneous
diffusion coefficient and `specific_storage` multiplies the pressure time
derivative. Their catalog scales are `D`, `cP`, and `1/GPa`, respectively.
Mixed pressure/flux Darcy variables are not part of v1.

The physics catalog defines which active property sets are valid for each physics
kind. The current seismic material families include:

- `acoustic`: accepts `(vp,rho)`, `(Sp,rho)`, or `(K,rho)` and optionally `qp`.
- `elastic:iso`: accepts `(vp,vs,rho)`, `(Sp,Ss,rho)`, `(K,Ks,rho)`,
  `(Kp,Ks,rho)`, or `(lambda,mu,rho)` and optionally `qp` and `qs`.
- `elastic:vti`: accepts any of the five `elastic:iso` tuples plus required
  `epsilon`; optionally accepts `qp`, `qs`, `gamma`, and `delta`.
- `elastic:tti`: accepts the same tuples plus required `epsilon`; optionally
  accepts `qp`, `qs`, `gamma`, `delta`, `phi`, and `theta`.
- `poroelastic:direct`: requires direct drained-frame Biot inputs `k_dry`,
  `mu_dry`, `k_solid`, `k_fluid`, `rho_solid`, `rho_fluid`, `porosity`,
  `tortuosity`, `kappa`, and `viscosity`; optionally accepts `qk`, `qmu`, and
  `vadapt`.
- `poroelastic:direct_jkd`: uses the same direct Biot inputs, additionally
  requires `viscous_length`, and optionally accepts `biot_frequency`.
- `poroelastic:iso`, `poroelastic:vti`, and `poroelastic:tti`: retain the
  `(vp,vs,rho)` frame parameterization, with the corresponding anisotropy
  properties, plus `k_solid`,
  `k_fluid`, `rho_solid`, `rho_fluid`, `porosity`, `tortuosity`, `kappa`, and
  `viscosity`.
- `poroelastic:iso_jkd`, `poroelastic:vti_jkd`, and `poroelastic:tti_jkd`:
  add the JKD-style dynamic hydraulic closure and therefore require
  `viscous_length`, with optional `biot_frequency`.
- `darcy`: requires `kappa`, `viscosity`, and `specific_storage`.
- `em:plasma`: requires `electron_density`; optionally accepts `mag_x`,
  `mag_y`, `mag_z`, `ion_mass`, `ion_charge`, `electron_collision`, and
  `ion_collision`; forbids geophysical EM properties.
- `em:geophysics`: accepts `conductivity` or `resistivity`, and optionally
  accepts `permittivity` and `permeability`; forbids plasma EM properties.
- `em:vacuum`: accepts no material properties and uses runtime vacuum defaults.

## Diagnostics

Validation and support tooling should surface these failures as structured
diagnostics:

- missing `subdomains` for a runnable material model
- duplicate or unmapped mesh block ids
- borehole layer `mesh_block_id` not present in `subdomains`
- malformed borehole extent or unknown extent surface
- malformed borehole layer/surface ordering, including unconsumed radial mesh
  surfaces
- missing, negative, or non-increasing borehole surface `r`
- borehole plug `mesh_block_id` not present in `subdomains`
- malformed or overlapping borehole plug intervals
- missing, negative, or out-of-borehole borehole plug `r`
- fracture surface missing `depth` or `gap`
- malformed fracture `depth` or `gap` property
- fracture `mesh_block_id` without a corresponding `subdomains` entry
- unsupported borehole layer fields from older drafts, including `role`,
  `cells`, `r`, `radius`, `inner_radius`, and `outer_radius`
- deprecated borehole `parts`
- unknown property name
- expression reference to a missing, inactive, or self property
- expression dependency cycle
- malformed expression `arg`/`args` or unsupported expression `op`
- property specified both inline and as an object in an incompatible way
- missing file or missing grid metadata for file-backed binary properties
- unsupported file format
- unknown or ambiguous physics kind
- required property missing for the selected physics kind
- forbidden property present for the selected physics kind
- hard-range violation, NaN, or Inf in evaluated property data


## Compatibility

This schema documents the shared material-model shape, not every
problem-specific physics completeness rule. The catalog and physics registry are
the source of truth for property names, required property sets, defaults, scales,
and range checks.

### Collisional cold plasma

`em:plasma` additionally accepts `ion_mass` (mass in proton masses, default 1),
`ion_charge` (positive charge in elementary charges, default 1), and
`electron_collision` / `ion_collision` (nonnegative collision rates in `1/s`,
default 0). The mass and charge must be positive. Ion density is
`electron_density/ion_charge`; `ion_mass: 2` selects deuterium. The dielectric
uses SI species constants with the solver's `exp(+i omega t)` convention.
Exact collisionless resonances are errors; nonzero collision rates supply
physical regularization. Magnetic components use the Maxwell solver frame:
`(R,Z,phi)` in cylindrical mode and `(x,z,-y)` in Cartesian 2.5D mode.

## Thermal materials

Layers with `physics: "thermal"` require positive finite `thermal_conductivity`
(default unit `W/m/K`) and volumetric `heat_capacity = rho*c_p` (default unit
`J/m^3/K`). Optional finite `heat_source` is signed volumetric heat generation
(default unit `W/m^3`, default value zero). Coefficients may vary spatially but
are fixed during a transient run. See `examples/thermal-materials.json`.

## Anisotropic conductive earth and induced polarization

`em:geophysics` accepts the symmetric DC tensor properties `conductivity_xx`,
`conductivity_yy`, `conductivity_zz`, `conductivity_xy`, `conductivity_xz`, and
`conductivity_yz` in S/m, in the solver field-component frame. A nonzero tensor
must be positive definite and cannot be combined with nonzero scalar
conductivity/resistivity. All-zero conductivity is insulating. Signed off-diagonal
components are allowed; symmetry is implicit.

`chargeability` m, `ip_time_constant` tau (seconds), and `ip_exponent` c default
to 0, 1 second, and 1. The runtime requires 0 <= m < 1, tau > 0, and 0 < c <= 1.
For s=i omega, the entire DC tensor is multiplied by
`(1+(s*tau)^c)/(1+(1-m)*(s*tau)^c)`. This is the Cole–Cole resistivity convention,
with sigma(infinity)=sigma_DC/(1-m). Scalar and tensor materials use the same
law and support complex frequency and frequency derivatives through order 4.
The background excitation has its own independently specified layer properties.


### Stationary viscous fluid coupled to elasticity

`elastic:viscous_fluid` models frequency-domain linearized Navier–Stokes about a
stationary, isentropic fluid. Use simulation `physics: "elastic"` and Cartesian
2D or 3D velocity–stress DPG with `form_execution: "compiled"`. Fluid and
isotropic solid subdomains share full velocity and traction traces, enforcing
no slip and traction equilibrium without an acoustic interface penalty.

Required material properties are positive `K` (isentropic bulk modulus, GPa),
`rho` (g/cc), `viscosity` (dynamic viscosity, cP), and `vadapt` (km/s).
`bulk_viscosity` is optional, nonnegative, in cP, and defaults to zero.
Unit-bearing property values may specify SI units such as `Pa`, `kg/m^3`,
and `Pa*s`. Do not supply elastic wave speeds, Q factors, or anisotropy on
fluid layers. Isotropic elastic layers retain their existing parameterizations.

The time convention is exp(i omega t). With eta the dynamic viscosity and zeta
the bulk viscosity, the material supplies `mu = i*omega*eta` and
`lambda = K + i*omega*(zeta - 2*eta/3)`. The existing Forms stress–velocity
system therefore solves momentum and continuity with the pressure eliminated.
In 2D this is plane strain with the three-dimensional Newtonian deviatoric
coefficient 2/3. `pressure` recovers thermodynamic fluid pressure from total
stress; in solids it retains the mean compressive stress convention.
`stress` is total Cauchy stress and `velocity` is fluid particle velocity.
A `fixed` exterior boundary is a stationary no-slip wall; `free` prescribes
zero total traction.

The mesh must resolve `delta = sqrt(2*eta/(rho*omega))` in SI units.
Choose `vadapt` conservatively for the requested frequency range; the shear
phase speed is `sqrt(2*eta*omega/rho)`. A fixed speed at the lowest frequency
is conservative for higher frequencies under wavelength-based adaptation.
Use local `hmax` and sufficient polynomial order near walls and check mesh
convergence. The explicit `vadapt` requirement prevents automatic use of the
much longer compressional wavelength for a boundary-layer problem.

This first version excludes zero frequency, transient and static workflows,
mean flow, temperature and thermal boundary layers, anisotropic solid mixtures,
and cylindrical/2.5D formulations. Elastic material-parameter inversion
sensitivities are not defined for this fluid parameter space. The constitutive
law provides analytic frequency derivatives; existing Forms machinery handles
frequency-dependent compliance and assembly.

Forward frequency/Laplace derivative workflows are rejected for viscous-fluid
models in this first version: thermodynamic pressure recovery has its own
frequency-dependent receiver map. Constitutive frequency tangents are available,
but complete receiver-map product rules remain to be added before those workflows
can report correct pressure derivatives.

### Frequency-domain Willis elasticity

An elastic subdomain may supply `willis` alongside `properties`. Its optional
`density`, `stress_velocity`, and `momentum_strain` tensors implement
`stress = C strain + S velocity` and `momentum = R strain + rho velocity`,
using `exp(+i omega t)`. Each tensor has a row-major `real` matrix, optional
matching `imag` matrix, and optional `units`. Density defaults to `kg/m^3`;
both coupling tensors default to `Pa*s/m`. Tensor density replaces scalar
inertia; scalar `rho` remains the reference used by stiffness parameterizations.
Absent couplings are zero, and no symmetry or reciprocity relation is inferred.

Shapes are D×D, DV×D, and D×DV respectively. Engineering strain order is
`xx, zz, xz` in 2D and `xx, yy, zz, yz, xz, xy` in 3D; shear entries are twice
tensor shear strain. Tensors are constant per subdomain, in global Cartesian
axes, and do not inherit stiffness orientation angles. The initial support is
compiled Cartesian frequency-domain DPG with fixed/traction boundaries and
body-force sources. PML on Willis material, impedance boundaries on Willis
material, transient and reduced-dimensional formulations are unsupported.
Ordinary subdomains keep the scalar-density elastic form and storage.

### Localized tensor hat controls

`parameterized.control.kind: tensor_hat` composes two or three uniform nodal
axes in one named coordinate system. Supply equally sized `axes`, `shape`,
`origin`, and positive `spacing` arrays. Axis counts are at least two; the
coefficient count is `product(shape)`. The first listed axis varies fastest.
The runtime rejects duplicate resolved axes, inactive axes, nonfinite geometry,
invalid counts and integer overflow before creating the lattice.

The map preserves existing identity/log/inverse/logit property transforms and sparse
JVP/VJP behavior. At most four or eight coefficients contribute at a point.
Outside the closed lattice support the update is zero; use zero boundary
coefficients for a continuous compact target. A nonzero boundary coefficient
creates a jump to the reference property outside the box. Grid geometry and
coordinate transforms must stay fixed during material differentiation.

Control identities include axis order, knot geometry and resolved coordinate
frames. Tensor hat maps support identity regularization and native variational
Tikhonov/TV/TGV smoothing with `derivative_order: 1` through
`control_sensitivities.Smoothing`, using the same lambda/alpha, epsilon,
iterations and `input_role` semantics as 1D hat blocks; `derivative_order: 2`
is rejected because the multilinear basis has no second derivative.
