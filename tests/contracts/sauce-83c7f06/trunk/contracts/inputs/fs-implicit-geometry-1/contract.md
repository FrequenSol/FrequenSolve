# FS Implicit Geometry Contract v1

Status: initial
Visibility: public
Schema id: `fs-implicit-geometry-1`

## Summary

Implicit geometry defines signed-distance-like surfaces used by the GMP
geometry kernel and mesh adaptation tools. JSON surface lists and declared GMP
surface records are compiled into `surface_registry`; GMP implicit entities
then bind signed surface ids into pure evaluators for curves and rectangles.

## Surface Registry

- `surfaces` is an ordered list. Runtime ids are 1-based list indices. A
  nonempty `name` gives a stable reference that is preferred in new input.
- Each surface object must have `_type`.
- When a GMP mesh is supplied, its declared `Plane` and `Cylinder` surface
  records are appended after the JSON `surfaces` list in active surface order.
- Signed numeric references use the integer id sign: positive means the stored
  surface orientation, negative reverses phi and gradient. Binary Boolean
  operands and n-ary child `id` values may instead use a surface name; their
  separate scale field controls orientation and scaling.
- References may also carry `scale`; this multiplies the signed reference.
- Surface registry objects are allocated once during initialization and then
  accessed through pointers. Referenced surfaces must outlive all bound
  `SurfaceRef_t` values.

## Supported Surface Types

- `simple`: coordinate graph surface `q_axis = function(q_other...)`.
  Fields are `axis`, optional `coord_system` (`global` by default),
  `function`, optional `units`, `periodic`, and `period`. `function` may be a
  scalar quantity, a file-backed xarray property, or an inline xarray object
  with `data`, `dims`, and `coords`.
- `elevation`: wraps the regular elevation-surface implementation.
- `plane`: `p0`, `n`.
- `sphere`: `c`, `r`.
- `ellipsoid`: `center`, positive axis-aligned `radii`. Its field is scaled to
  length units and has the exact ellipsoid zero set; it is not claimed to be
  the exact Euclidean distance away from the surface.
- `cone`: `apex`, nonzero `axis`, and `half_angle` in radians in `(0, pi/2)`.
  The surface is the infinite forward circular cone without a cap.
- `cylinder`: `center`, `axis`, `radius`, optional `capped` and `half_length`.
- `box`: `center`, `half_size`, optional `oriented`.
- `rbf` and `rbf_level_set`: compact Wendland-C2 expansion with
  `support_radius`, one coordinate row per `centers` entry, a matching
  `coefficients` vector, and optional `bias`. The field is
  `bias + sum_i coefficients[i] psi(||x-centers[i]||/support_radius)`.
  Compact support makes evaluation sparse after initialization builds a uniform
  bin index. The initialized surface and index are replicated on each rank and
  rejected if their persistent footprint reaches 1 GiB.
- `capsule`: `a`, `b`, `r`.
- `transform`: `id`, optional `scale`, `t`, uniform `s`, and optional matrix
  `R`.
- `offset`: `id`, optional `scale`, `delta`; `r` is accepted as a compatibility
  synonym when `delta` is not supplied.
- `abs`: `id`, optional `scale`, optional `eps`.
- `repeat`: `id`, optional `scale`, `period`, `shift`.
- `angular_repeat`: `id`, optional `scale`, `center`, `axis`, `ntheta` or
  `dtheta`, `theta0`, `pz`, `z0`, `theta_stagger`, and `stagger_half`.
- Binary booleans: `union`, `intersection`, `difference`, `smooth_union`,
  `smooth_intersection`, `smooth_difference` with `a`, `b`, optional `sa`,
  `sb`, and `k`. `a` and `b` accept either signed numeric ids or names.
- N-ary booleans: `nunion`, `nintersection`, `ndifference`,
  `nsmooth_union`, `nsmooth_intersection`, `nsmooth_difference` with
  `children`.

## Evaluation Semantics

- `phi(x) < 0` denotes the inside/selected side for primitive SDFs unless the
  reference sign reverses it.
- `simple` evaluates `phi = q_axis(x) - function(x)`. Angle-like axes such as
  cylindrical `theta` wrap residuals by default; `periodic` and `period` may
  override that behavior.
- `grad(x)` returns the corresponding signed gradient.
- `phi_grad` is the preferred primitive operation because it keeps value and
  gradient orientation consistent.
- Batched evaluators may override the default point loop for performance, but
  must return the same values as scalar `phi_grad`.
- Boolean smoothing parameter `k <= 0` means hard min/max behavior. Positive
  `k` activates the smooth variant for smooth boolean operators.
- An RBF `control` publishes the whole authored-order coefficient vector under
  one stable `id`; coefficients remain positional and do not receive individual
  ids. `maximum_displacement` and `feasibility_band` configure the
  representation-owned step cap used to limit sampled normal zero-set motion.
  Their defaults are `0.25 * support_radius` and `support_radius`.

## GMP Binding Semantics

- A GMP implicit curve uses `D + 1` signed surface ids: constraint surfaces
  followed by endpoint surfaces.
- A GMP implicit rectangle uses five signed surface ids: one base surface and
  four boundary surfaces. In GMP v2 quad records these ids are stored in the
  standard integer payload after the four vertex ids.
- GMP implicit ids are local to the GMP mesh surface table. Runtime binding
  remaps them after appending GMP surfaces to any JSON-defined model surfaces.
- `bind_implicit(gmp)` replaces each implicit entity's `Idata(1)` with an
  internal bound-reference handle. After binding, evaluators read through the
  handle rather than interpreting raw ids again.
- GMP v2 text persistence for declared surfaces and `Implicit` curve records is
  owned by `fs-gmp-mesh-2`. This contract covers runtime surface semantics and
  binding/evaluation behavior.

## Surface Intersections

- The runtime can compute 2D intersections between registered graph surfaces
  through `surface_intersections_2d`.
- For 1D xarray-backed `simple` and `elevation` surfaces, intersections are
  found from the linear-interpolation polyline knots, not from a fixed sampling
  count.
- Scalar `simple` surfaces are represented by the corresponding straight graph
  segment over the requested bounds.


## Compatibility

The historical comment examples sometimes show `type`; the current registry
reader requires `_type`. New JSON should use `_type`.
