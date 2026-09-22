# FS Simulation Contract v1

Status: initial
Visibility: public
Schema id: `fs-simulation-1`

## Summary

A simulation document defines the physical model, coordinate systems, mesh,
solver options, acquisition, boundary conditions, and outputs for a FrequenSolve
run. It may live in a separate file or be embedded inline inside an `fs-job-1`
document.

## Versioning

- A simulation with `"schema": "fs-simulation-1"` is interpreted by this
  contract.
- A simulation without `schema` is treated as legacy v0 by current readers.

## Required Behavior

- `project_path` defines the project root used for model, mesh, and output path
  resolution when the simulation is run directly. For job-driven runs, a
  job-level `project_path` takes precedence when present.
- `physics` selects the physics family/formulation. Current public values
  include `coupled`, `coupled2`, `coupled_static`, `coupled_axisym`,
  `coupled_axisym_torsion`, `coupled_aep`, `coupled_aep_axisym`, `acoustic`,
  `acoustic_axisym`, `elastic`, `elastic2`, `elastic_static`,
  `elastic_axisym`, `elastic_axisym_torsion`, `poroelastic`,
  `poroelastic_axisym`, `darcy`, `thermal`, `em`, and `smooth`.
  `thermal` selects the scalar H1 heat equation with backward Euler and requires
  Galerkin discretization and the transient workflow in Cartesian 2D or 3D.
  See [thermal formulation](../../../problems/Physics/Thermal/README.md). `darcy` selects the
  pressure-form H1 Galerkin diffusion/storage formulation and therefore
  requires `Discretization/method: "Galerkin"` and `workflow: "transient"`. The
  `acoustic`, `elastic`, and `coupled` values support the Cartesian,
  full-dimensional Galerkin transient workflow; `coupled` advances acoustic
  pressure and elastic displacement through the physical interface form. The
  `coupled_aep` value selects a mixed acoustic-elastic-poroelastic UW-DPG
  problem, and `coupled_aep_axisym` selects its non-torsional 2D cylindrical
  axisymmetric counterpart. The `poroelastic_axisym` value selects the
  standalone 2D cylindrical axisymmetric Biot poroelastic formulation. The
  `elastic2` and `coupled2` values select the weak-symmetry elastic
  UW-DPG formulation, while `elastic_static` and `coupled_static` select the
  displacement-form static elastic UW-DPG formulation. The `acoustic_axisym`
  value selects ordinary acoustic material physics on the 2D cylindrical
  axisymmetric metric. The
  `elastic_axisym`/`coupled_axisym` values select the
  non-torsional 2D cylindrical axisymmetric elastic formulation, and
  `elastic_axisym_torsion`/`coupled_axisym_torsion` select the torsional
  formulation. These formulation keys retain ordinary elastic material physics.
- `axisymmetric`, when true, selects the corresponding non-torsional 2D
  cylindrical axisymmetric variant for base seismic physics values `acoustic`,
  `elastic`, `coupled`, `coupled_aep`, and `poroelastic`. Existing `_axisym`
  physics aliases remain accepted for compatibility.
- `dimension` is normally `2` or `3`; current half-dimension workflows may use
  `2.5`. Axisymmetric formulation keys are valid only for 2D
  simulations. If `global_coordinate_system` is omitted, these formulations
  assume an r-z cylindrical global system with fixed theta; existing 2D `x`/`z`
  mesh and material-grid inputs remain accepted with `x` interpreted as radial
  `r`.
- `Discretization`, `Mesh`, `BCs`, `Acquisition`, and `Outputs` are consumed by
  solver setup in the Seismic workflows.
- `Discretization/DPG_enrich` defaults to `1`; it and `DPG_alpha` and
  `DPG_penalty` configure problem-owned DPG enrichment and Gram weights.
  `DPG_trial_to_test` selects
  `store` (the default) to retain trial-to-test actions or `compute` to rebuild
  them element by element when follow-on loads need them. Static-condensation
  load factors remain stored; the recomputed action uses the base-frequency
  Gram matrix without its frequency derivative. `DPG_gram_omega` selects `real`
  (the default, which discards the imaginary part) or `complex` (which retains
  the full complex angular frequency) for DPG Gram assembly.
- `Solver/schur_precision` controls both storage and application of
  static-condensation factors and accepts `fp32` (the compatibility default),
  `fp64`, or the active solver's `native` precision. `Solver/schur_storage`
  accepts `auto`, `memory`, or `disk`; `auto` follows the existing `use_disk`
  and memory-limit policy. `Solver/schur_batching` accepts `grouped` (the
  compatibility default) or `single`, which keeps one process-local factor bank
  spanning all local assembly groups. The `memory` plus `single` combination is
  the resident, no-temporary-I/O path; transient solver setup forces both
  settings for repeated time steps. Setting
  `Solver/sparse_rhs` to `false` additionally preallocates one contiguous dense
  RHS bank for the Schur records, eliminating per-record RHS allocation and
  sparse-column packing on repeated dense loads. When disk storage is required,
  `Solver/schur_temp_backend` accepts the legacy `hdf5` container or `flat`,
  which performs concurrent positional I/O over the contiguous factor bank and
  is intended for high-throughput local scratch storage.
- `Solver/block_schur/enabled` opts a transient coupled acoustic-elastic
  Galerkin simulation into a partitioned interface solve. The acoustic and
  elastic diagonal blocks use the symmetric FS_MG path, while the nonsymmetric
  interface action is iterated with the configured `maximum_iterations`,
  `relative_tolerance`, `absolute_tolerance`, and `relaxation`. Linear interface
  matrices are retained after operator assembly and reused for each outer action
  when `store_interface_operators` is true; when false, each action is
  reassembled.
  `inner_tolerance` caps the deliberately inexact FS_MG block solves,
  while `inner_forcing` tightens them with the measured coupled residual. Outer
  convergence is checked from the full coupled residual, using a fixed norm
  reference for the time step, rather than the iterate update. The current outer
  policy is relaxed fixed point; disabling or omitting this object keeps the
  general MUMPS path.
- `helmholtz_projection/enabled` requests an L2 projection of the compressional
  part of physical velocity after each solve, with either a complementary or
  independently projected shear part. Naming `p_velocity` or `s_velocity` in a
  receiver or ParaView field request enables the projection automatically.
  For coupled acoustic-elastic problems,
  the default projection domain is the complete mesh and its source is the
  piecewise field formed from acoustic velocity in fluid cells and elastic
  velocity in solid cells. DPG uses the native D-component L2 fields. Galerkin
  derives acoustic velocity as `-grad(p)/(i*omega*rho)` and elastic velocity as
  `i*omega*displacement`. The auxiliary scalar potential satisfies
  `(grad(phi), grad(w)) = (velocity, grad(w))` with natural outer-boundary
  conditions and a continuous potential across material interfaces.
  Legacy `part: "p"` routes `grad(phi)` through ordinary acoustic, elastic, or
  unqualified velocity receiver and ParaView channels; `part: "s"` routes
  the selected S projection. `parts: ["p", "s"]` retains ordinary velocity
  channels and appends separately named `p_velocity` and `s_velocity` products
  from the same physical and auxiliary solves. Receiver components retain their
  authored names, followed by `p_` and `s_` copies; ParaView retains its ordinary
  velocity item and appends the named parts. The names can also be requested
  directly, including Cartesian forms such as `p_velocity_x` and `s_velocity_z`.
  With explicit named fields and no legacy `part`, ordinary `velocity` remains
  undecomposed; this also applies to `Outputs/wavefields`, whose fields are
  requested explicitly rather than expanded. The undecomposed field uses the
  auxiliary element-local L2 velocity copy (exact for native L2 velocity up to
  numerical accuracy; an L2 approximation for derived Galerkin velocity).
  Receiver groups and ParaView definitions that already name P/S fields retain
  their authored layout even when `parts` is supplied; no automatic copies are
  added to those definitions.
  By default, the two projected parts therefore
  reconstruct the piecewise physical velocity. `part` and `parts` are mutually exclusive.
  Non-velocity receiver and ParaView attributes remain on their native plans
  while this output projection is active.
- `helmholtz_projection/s_projection` defaults to `complement`. Setting it to
  `solenoidal` projects S independently: in 2D it solves a scalar stream-potential
  Poisson problem and emits its rotated gradient; in 3D it solves a gauge-fixed
  H1 vector-potential Poisson problem and emits its curl. This produces a smooth,
  divergence-free S field, but boundary and harmonic components mean P+S is not
  required to reconstruct the original velocity exactly. Both potentials remain
  blocks of the same auxiliary job and share one solver invocation.
- `helmholtz_projection/gauge_penalty` defaults to `1e-10`. It is a relative,
  rank-one element gauge used to make constant potential nullspaces nonsingular;
  it perturbs projected derivatives only at the configured relative scale.
  `sparse_rhs` and `sparse_rhs_tol` configure the auxiliary
  solve. `helmholtz_projection` and `l2_projection` are mutually exclusive.
- `BCs` is an array of boundary-condition entries. Each entry provides
  `conditions` and `boundaries`. When multiple conditions are listed, the
  reader merges their per-variable flags:
  zero means "no condition", identical nonzero flags are accepted, and
  conflicting nonzero flags on the same variable are rejected.
- Boundary names are interpreted as labeled GMP boundary sets when every
  configured boundary can be resolved that way; otherwise the reader falls back
  to geometric side names such as `x_min` and `z_max`.
- PML outer boundaries created during extrusion use an impedance termination by
  default. Use the condition `pml:impedance` to request this explicitly or
  `pml:fixed` to use a fixed termination. They may also be targeted explicitly as
  `<boundary>__pml_outer`, for example `x_min__pml_outer` or
  `z_max__pml_outer`, when an override is needed.
- PML entries may define `domain_extension` to control only the generated PML
  volume-element material domains on that BC. `domain_extension/default` assigns
  the fallback PML domain, `preserve` lists source domains that keep their
  adjacent interior material, and `remap` maps specific source domains to
  replacement domains. Domain references may be `mesh_block_id` integers,
  material subdomain names, or GMP region labels. Boundary labels and BC
  assignment remain unchanged. `domain_extension/blend` optionally blends
  material properties from the PML cell domain toward a target domain over
  normalized PML depth, using `linear`, `smoothstep`, or `smootherstep`
  profiles. Blend source domains are kept from falling through to `default` so
  mesh-domain visualization and quadrature material evaluation use the same
  base domain. Blend keys may name either the post-remap PML domain or an
  original source domain that is first passed through explicit remap entries;
  endpoints must resolve to compatible material physics after remapping.
- `pml_reflection` is the preferred damping control. When both
  `pml_constant` and `pml_reflection` are provided, `pml_reflection` takes
  precedence and derives the stretch constant from the reference wave speed.
  `pml_constant` remains available for compatibility when no reflection target
  is provided.
- `BCs[*]/name` is optional and is only a diagnostic label; condition behavior
  comes from `conditions`.
- For acoustic UW-DPG runs, `gravity_surface` imposes the homogeneous linear
  free-surface relation `v_n = i omega p/(rho g)` using the local acoustic
  density and `Model/gravity`. The condition is not registered for the
  pressure-only Galerkin formulation. Nonhomogeneous free-surface data are not
  part of this contract version.
- Poroelastic free surfaces usually need a composite mechanical/fluid condition.
  Use `conditions: ["free", "drained"]` for a traction-free drained boundary
  (`t_hat = 0`, `p_hat = 0`) or `conditions: ["free", "sealed"]` for a
  traction-free sealed boundary (`t_hat = 0`, `q_hat = 0`).
- For non-torsional axisymmetric elastic runs, use `axis` or `symmetric_r` on
  the `r = 0` boundary. This imposes only the regularity traces
  `u_hat_r = 0` and `t_hat_z = 0`; `u_hat_z` and radial traction remain free.
  Ordinary `free`, `impedance`, or pressure/normal-traction conditions are not
  valid axis conditions.
- For axisymmetric poroelastic runs, use `axis` or `symmetric_r` on the
  `r = 0` boundary. The axis condition constrains all poroelastic trace
  unknowns (`p_hat`, `u_hat`, `q_hat`, and `t_hat`) for regularity.
- `Mesh/adapt/f_low` optionally sets a lower bound on the physical frequency
  used for mesh adaptivity. It is required and must be positive for transient
  wave simulations; `f0` controls nondimensionalization and is not a substitute.
- `Mesh/adapt/elems_per_wave` sets the target adapted elements per wavelength
  before solver-grid refinements are applied. It may be scalar or an object
  keyed by active global coordinate-axis names.
- `Mesh/adapt/order` sets the initial polynomial order assigned to the root
  mesh before adaptivity and solver refinement. It defaults to `1` and may be
  scalar or keyed by active global coordinate-axis names.
- `Mesh/adapt/f_high` optionally sets an upper bound on the physical frequency
  used for mesh adaptivity.
- `Mesh/adapt/f_adapt`, when present, selects an explicit physical frequency
  used for mesh adaptivity.
- `Mesh/adapt/coarsen_bounds` selects conservative coarsened material bounds
  during mesh adaptivity and defaults to `true` for every workflow. Set it to
  `false` explicitly to request strict segment bounds.
- Material-jump adaptation is configured under `Mesh/adapt/jump`. Presence of
  that object enables the policy unless `enabled` is `false`.
  `contrast_threshold` is the reference contrast used to normalize jump
  strength rather than a hard activation cutoff. Positive sub-threshold scores
  receive a fractional refinement multiplier; a composite score of at least
  `1` receives the full `factor`. The interpolation is performed in refinement
  levels, so the multiplier is
  `factor**(min(score, 1)**activation_decay)`. `activation_decay` defaults to
  `1`; values above `1` suppress weak jumps more strongly, while values below
  `1` retain more of their strength. A candidate must still have an effective
  width no greater than `width_fraction` of the element-axis length.
  `dead_fraction` excludes transitions near either element interface and
  defaults to `0.01`; `decay_power` controls the symmetric score decay from the
  midpoint toward those dead bands and defaults to `0.75`. Within the accepted
  width, sharpness linearly increases the score from `1x` at `width_fraction`
  to `2x` for a zero-width limit. Each `a` step in `Solver/refinements` divides
  the effective jump factor by two, down to a minimum of `1`, so adaptive-hp
  refinement does not duplicate the jump-driven EPW increase.
- `Solver/refinements` optionally provides one refinement program per
  solver-grid transition. Each character is one sequential refinement step;
  supported tokens are `h`, `p`, and `a`, and repeated tokens such as `pp` or
  `hh` apply multiple steps before the next solver grid is assembled.
- `Solver/grids` defaults to three when `refinements` is omitted, or to one
  more than the number of supplied refinement programs. An explicit grid count
  takes precedence; transitions beyond the supplied programs repeat the last
  program. Frequency-domain Galerkin setup can reduce this to one grid.
- `Solver/mode` accepts `fast`, `robust`, or `memory`. These are construction
  presets, not convergence guarantees. Omission retains the ordinary defaults.
  `fast` selects BF16 patches when native BF16 is enabled and FP16 otherwise.
  In 3D it also defaults coarse macros to the same half format, selects PCG
  coarse solves, and uses log-growth cycle visits. In 2D with MUMPS it retains
  direct coarse solves and V cycles; without MUMPS it uses PCG and log-growth
  visits. `preg` and `preg_c` remain `0.01`. `robust` defaults patches to FP32;
  `memory` defaults them to FP16 and applies its construction tolerances.
  Explicit values override preset defaults. An explicit `solve_precision`,
  `operator_precision`, or `patch_precision` suppresses preset patch precision.
  Presets leave communication precision native unless explicitly changed.
- `fast` enables relaxed assembly unless `Solver/relaxed_assembly`
  explicitly supplies a value. Assembly defaults to CPU;
  `Solver/forms_backend` explicitly selects another assembly backend.
- `Solver/cycle` may be a positive scalar recursive visit count or an explicit
  object. The object's default `anchor` is `fine`: `visits: [1, 3]` uses one
  visit at the active fine grid and three at the next grid. `anchor: "coarse"`
  indexes upward from the coarse grid; that grid uses its coarse solve, so
  the first recursive level uses the second entry. Missing entries use one visit. The only supported `kind` is `symmetric`.
- `Solver/hp/order` defines the preferred p-order policy for adaptive hp token
  `a`. It may be a constant integer or a first-match branch expression with
  `op: "case"`, branch objects containing `if` and integer `then` p orders,
  and a required integer `else` p order. Conditions may compare
  `{ "var": "epw" }` with numeric values using symbolic operators such as `>`,
  `>=`, `<`, and `<=`; boolean composition may use `and`/`&` and `or`/`|`.
- `Solver/hp/overrides` optionally lists classifier-specific adaptive hp order
  policies. Each entry provides `classifier`, `value`, and `order`; currently
  `classifier: "physics"` is supported, and `value` is the physics name to
  match. A matching entry replaces `Solver/hp/order`; unmatched elements use
  `Solver/hp/order`.
- `Solver/hp/order_x`, `order_y`, and `order_z` optionally override or clamp
  the p-order policy for a single element axis. A direct integer or case
  expression replaces `Solver/hp/order` for that axis. An object with `min`
  and/or `max` clamps the global order. An object with `policy` uses that
  axis-specific policy and then applies any `min`/`max` clamp. `order_z` is
  ignored in two-dimensional runs.
- `Solver/patch_storage` accepts only `blocky` or `compact`. `blocky` is the
  default for BF16 patches and stores every upper-triangle tile, including
  diagonal tiles, as a full provider-ready BF16 panel. `compact` retains the
  compact upper-triangle representation. `blocky` storage requires
  resolved BF16 patch precision. Other precisions default to `compact`.
- `Solver/patch_precision_filter` enables adaptive per-patch low-precision/FP32
  storage when `Solver/patch_precision` is `fp16` or `bf16`. The default `none`
  preserves uniform low-precision storage. `row` uses a rigorous packed-matrix
  row-sum energy-error bound; `collatz` applies
  `patch_precision_collatz_steps` positive-vector refinements to
  tighten that bound; and `spectral` reuses the patch Cholesky factor to
  compute the exact FP32-to-lowp energy-error radius without another
  factorization. `patch_precision_tol` defaults to `0.2`; patches above the
  selected metric threshold remain in FP32. `print_patch_precision_filter` reports one
  global rank-0 summary per grid with low-precision and filtered-to-FP32 counts,
  plus global statistics for every evaluated tier metric. Adaptive patch precision currently
  supports CPU execution in the mixed-precision build with the direct
  operator-bank backend (`SINGLE_MG=0`).
- `Solver/coarse_precision` accepts `native` (the default), `fp16`, or `bf16`.
  The half formats store macro operators below the finest active grid in
  16-bit blocks and accumulate their products in FP32. Residual and correction
  vectors remain in the active solve precision. In 3D, `Solver/mode: "fast"`
  defaults to `bf16` when native BF16 is enabled and `fp16` otherwise; an
  explicit `coarse_precision` takes precedence.
- `Solver/comm_precision` controls the FS_MG wire payload. The default `native`
  path communicates in the active solve precision. `fp16` and `bf16`
  communicate real and imaginary lanes in the selected half format and
  reconstruct active-precision values before accumulation; this conversion is
  lossy and can change Krylov convergence. Half precision is used for
  restriction/extension and for macro and
  smoother exchanges below the active fine grid. CUDA finest-grid exchanges
  follow `fine_grid_comm_precision`; fresh CUDA residuals and direct
  coarse-solver exchanges always use native precision.
  Only `native`, `fp16`, and `bf16` are accepted. Presets leave the native
  communication default unchanged.
- `Solver/fine_grid_comm_precision` selects CUDA finest-grid wire precision:
  `auto` (default) follows `comm_precision` in TF32 mode and keeps native
  communication in strict FP32 mode; `configured` follows `comm_precision`
  for ordinary macro and smoother exchanges; `native-residual` keeps every
  macro/operator product and residual evaluation at native FP32 communication
  while smoothing follows `comm_precision`; `native` keeps both at native
  FP32, restoring the previous finest-grid policy. Fresh FP32/FP64 residual
  recomputations always communicate in native FP32 under every policy.
  This setting changes communication only. Coarse grids, transfers, CPU and
  Metal retain their existing policies. The residual policy includes `A*p`
  products and the macro residual evaluations within smoothing cycles; the
  patch/smoother exchanges remain independently configured.
- `Solver/comm_mode` defaults to `overlap`. The selected schedule remains fixed
  for the run. `blocking-p2p` and `persist-p2p` remain available as explicit
  selections.
- `Solver/comm_progress_interval_ms` sets the target cadence for thread-zero
  progress polling inside dynamically scheduled patch and macro-operator work.
  It defaults to 2 ms; zero disables timed intra-group polling without removing
  mandatory communication-stage progress calls.
- `Solver/comm_progress_budget_us` bounds an early cooperative MPI progress
  slice after compute group 2. It defaults to 50 microseconds and returns as
  soon as any receive completes; zero replaces the slice with one ordinary
  nonblocking progress poll. The slice polls only active assembly receives and
  never waits indefinitely for a particular rank.
- `Solver/patch_ownership_slack` allows patch ownership to trade a
  bounded amount of compute balance for lower predicted off-rank traffic. It
  defaults to 0.05 and is clamped to the inclusive range 0 through 0.25.
- `Solver/scale_macro` enables symmetric diagonal scaling of the macro system
  during Krylov solves and defaults to true. When macro scaling is enabled, MUMPS diagonal
  scaling is disabled for the coarse/direct solver.
- `Solver/mitigate_blowup` enables the CPU PCG blow-up monitor when
  `blowup_window` is positive. It defaults to false. `save_best` defaults to
  this flag; explicitly setting `save_best: false` keeps the monitor while
  avoiding the extra solution-sized best-iterate buffer.
- `Solver/blowup_window` controls the PCG blow-up monitor window. Positive
  values permit soft true-residual PCG restarts, when mitigation is enabled and the recent residual trend
  grows sharply; zero disables this monitor. The default is `10`.
- `Solver/blowup_grace` controls how many soft blow-up restarts are allowed
  before PCG treats residual blow-up as a hard failure and enters recovery or
  Krylov fallback. The default is `1`.
- `Solver/blowup_hard_factor` controls the residual growth factor relative to
  the best PCG residual that immediately escalates blow-up detection to hard
  recovery or fallback. The default is `100`; zero disables this hard-factor
  cap.
- `Solver/replace_residual` controls how often PCG recomputes the
  true residual and restarts its search direction from the current solution.
  Positive values enable replacement; zero disables it. The default is `1000`.
  The native-real path recomputes without a direction restart.
- `Solver/recompute_residual` independently refreshes the PCG residual
  `b-Ax` every N iterations without resetting search-direction history. The
  default is `0` for the active Metal solver and `20` (every twenty
  iterations) for CPU and CUDA; explicit values override these defaults.
  CPU fallback retains the CPU default. `0` disables this policy, including
  candidate/final verification; stopping then uses the recursive residual.
  Positive values also verify candidate convergence and the final residual,
  including after Krylov fallback. CPU FP32 solves compute fresh element
  products, assembly, MPI exchange and subtraction in FP64, then store the
  residual in FP32. Operator storage and ordinary iterations stay FP32. CPU
  FP64 solves retain native arithmetic. Finite-precision solution storage can
  still limit the attainable tolerance.
  Set `replace_residual: 0` to disable scheduled direction restarts; existing
  instability recovery remains active. Metal retains native residual arithmetic.
  CUDA refreshes widen native dense products and segmented sums to FP64; stored
  vectors and communication remain FP32. CUDA widening currently requires
  `op_max_blocks: 1`. This evaluates the resident FP32 operator, not a separately
  assembled FP64 operator. Widening scratch is reused and capped at 1 GiB per
  CUDA backend; large operators and batches are tiled to respect that bound.
- `Solver/iterative_refinement` (default `false`) verifies and repairs each CPU
  FP32 iterative solve in FP64 against the ordinary convergence reference.
  The ordinary solve runs first, stopping at the larger of `tolerance` and
  `refinement_inner_tolerance`. Its residual `b-Ax` is recomputed with FP64
  products against the stored FP32 operator and normalized by the same per-RHS
  reference and active lanes as the ordinary solve (the smoothed initial
  residual for complex solves, `||b||` for native-real solves). While that
  misses `tolerance`, FP32 correction solves accumulate into an FP64 solution,
  each asking only for the reduction still needed, measured against its own
  load and floored at `refinement_inner_tolerance` (default `1e-4`);
  `refinement_max_steps` (default `10`) bounds them. The tolerance and the
  reported residual apply to the FP64 iterate. The returned solution is its FP32
  rounding, whose own residual can remain at the FP32 floor: refinement improves
  solution accuracy, not the stored solution's residual. Refinement stops early
  on an unstable or non-finite solve or a correction that fails to reduce the
  residual, and reports that status. It keeps one FP32 copy of the load and one
  FP64 solution per right-hand side, 24 bytes per complex unknown and
  right-hand side, plus the FP64 fresh-residual workspace. FP64 builds ignore the
  option; GPU solves and reduced-precision solve grids reject it.
- `Solver/recompute_residual_fp32` defaults to `5`. In CUDA TF32 mode it
  inserts fresh FP32 residual evaluations with native FP32 communication between
  the FP64 refreshes. Set it to `10` for ten-iteration spacing, or `0` to disable
  only this intermediate tier. FP64 refreshes take precedence on coincident
  iterations and candidate/final verification. `recompute_residual: 0` disables
  both tiers. Other backends and `cuda:fp32` ignore this intermediate interval.
  Only the finest-grid outer residual changes policy; coarse-grid work keeps
  its configured arithmetic and communication precision.
- `Mesh/adapt/hmin` and `Mesh/adapt/hmax` optionally set global minimum and
  maximum adapted element sizes in the simulation's default length units.
- `Mesh/adapt/aspect_limit` sets the baseline maximum element-size ratio
  between coordinate axes during directional refinement and defaults to `3`.
  The effective limit is raised when needed to accommodate anisotropic
  elements-per-wave targets.
- `Mesh/adapt/scholte`, `Mesh/adapt/krauklis`, and `Mesh/adapt/stoneley`
  provide the same compact guided-wave controls. `normal/h0` is the
  wall-adjacent element size divided by the active evanescent decay length,
  `normal/thickness` is the number of decay lengths in the normal profile, and
  `normal/grading` is its coarsening exponent. For Scholte waves,
  `normal/factor` limits the finest wall-normal size relative to Scholte-wave
  EPW sizing and defaults to `4`; the normal profile still coarsens to the
  local body-wave EPW size on both sides of the interface. Tangential controls
  use `tangential/factor`, `tangential/thickness`, and `tangential/grading`.
  Surface-, borehole-, and borehole-layer-local `adapt/<wave>` values override
  these global values field by field.
  Scholte controls additionally accept `enabled` (default `true`) and
  `lower_bound`, an optional floor on the velocity used for mesh sizing.
  `lower_bound` may be a solver-unit scalar or a unit-bearing object such as
  `{ "value": 600, "units": "m/s" }`.
- `Mesh/adapt/gravity_surface` opts the top acoustic formation into
  finite-depth gravity-wave sizing. The local phase speed solves
  `omega^2 = g k tanh(k h)`, where `h` is the separation between the top model
  surface and the next regular interface. Sizing is applied only in directions
  tangent to the free surface. `tangential/thickness` is a multiple of local
  water depth and defaults to `1`; `tangential/factor` is an additional EPW
  multiplier and defaults to `1`. The same block under
  `Model/surfaces[0]/adapt/gravity_surface` overrides global controls.
- `Mesh/adapt/normal_growth` optionally limits the side-normal size ratio
  between neighboring elements during adaptivity.
- `Mesh/adapt/geometry_error`, when present and enabled, adds h-adaptation
  based on the coordinate error between projected isoparametric geometry DOFs
  and exact GMP geometry. `tolerance` or `relative_tolerance` is normalized by
  each element's bounding-box diagonal, `absolute_tolerance` is interpreted in
  the simulation's default length units, and the effective threshold is the
  maximum of the relative and absolute tolerances.
- Geometry-error adaptation runs after material, source, and receiver
  adaptation for the initial mesh. `max_refinements` limits the total
  geometry-error passes, and `reserve_refinements` leaves the final passes for
  the last h-like solver hierarchy refinements.
- `Mesh/adapt/src_grading`, `Mesh/adapt/rcv_grading`, and each
  `Mesh/adapt/surface_gradings` entry may set `factor > 1` for stronger local
  refinement and `power > 0` to curve the normalized transition between `d0`
  and `d1`; `power: 1` is linear and `power: 2` is quadratic with the same
  endpoint factors. When `factor` is used, it is the factor at `d0` and the
  grading transitions to `1` at `d1`. Gradings may use `factor_max` and
  `factor_min` to set the two endpoint factors explicitly. `factor`, `factor_max`,
  `factor_min`, and `power` may be scalar or objects keyed by active global
  coordinate-axis names, such as `{ "r": 2, "z": 1 }` for the default
  axisymmetric cylindrical system. Surface names may target model surfaces,
  generated fracture boundary names such as `<fracture>_top` and
  `<fracture>_bottom`, registered borehole wall surfaces such as
  `<borehole>_<surface>`, or generated `surface_N` aliases.
- `Mesh/generator/l_bound` and `Mesh/generator/u_bound` accept either numeric
  vectors or `{ "value": [...], "units": "..." }` objects. `Mesh/generator/units`
  supplies default length units for both bounds; per-bound units override it.
  When no units are supplied, values use the simulation length scale.
- `Mesh/generator` is optional when `Mesh/file` is not set. In that case the
  solver creates a default `LayeredMeshGenerator`, derives bounds from
  `Model/x_limits` and `Model/y_limits` for 3D. Vertical extents are supplied
  by the layered model surfaces; `Model/z_limits` is not required or used by
  this default generator path. The generated mesh is written as `mesh.gmp` in
  the simulation directory, or in the project directory for embedded simulation
  documents.
- `Mesh/generator/element_size` is an optional ceiling on root-element spacing
  for `LayeredMeshGenerator`. It accepts a positive scalar, a vector with one
  value per active global axis, or an object keyed by active global axis names
  such as `{ "x": 100, "y": 75, "z": 50 }`. Counts are rounded up so the
  actual spacing is uniform on each regular horizontal axis and never exceeds
  its ceiling. It may also use `{ "value": ..., "units": ..., "scale": ... }`;
  otherwise `Mesh/generator/units` supplies its length units.
- `Mesh/generator/n` supplies explicit root counts for `LayeredMeshGenerator`
  and is mutually exclusive with `element_size`. Regular horizontal-axis
  entries are used exactly. The final layered-axis entry is accepted as part of
  the dimension-sized vector but replaced by the expanded material layer-stack
  count. The resulting root grid must contain at least one cell per MPI rank.
  When both `n` and `element_size` are omitted, the solver derives the size from
  the minimum model adaptivity wavespeed, requested EPW, and maximum adapt
  frequency. Frequency precedence is positive
  `Mesh/adapt/f_adapt`, positive `Mesh/adapt/f_high`, then the largest absolute
  real frequency in the job `f_list` clamped by `f_low`. The root size is one
  dyadic level coarser than the 110%-of-target spacing, so its first required
  refinement lands early in the `(EPW, 2*EPW)` interval.
- Automatically sized layered generator paths provide at least as many initial
  cells as MPI ranks. When necessary, they double one regular horizontal-axis
  count at a time, selecting the currently least-resolved axis, until the rank
  floor is met. Explicit `n` is never changed; an undersized explicit grid is a
  configuration error.
- `Mesh/generator/horizontal_spacing` may request borehole-aligned horizontal
  coordinates and local x refinement for `LayeredMeshGenerator`. Controls refer
  to named `Model/boreholes` entries with `around_borehole`; `max_size` is the
  local maximum horizontal cell size and `padding` extends the controlled
  interval beyond the radius envelope computed from radius breakpoints.
- `Mesh/generator/_type: "SweptBoreholeMeshGenerator"` builds a near-wellbore
  3D mesh by sweeping a borehole cross-section along a piecewise-linear
  `centerline`. It does not embed the borehole in a larger formation model.
- `Mesh/generator/centerline/points` is an ordered list of 3D points. Each point
  may be a raw numeric coordinate vector or an object with `value`, optional
  `units`, and optional `scale`. `centerline/n` sets the total number of sweep
  intervals distributed by segment length; `centerline/segment_counts` may
  provide per-segment counts instead.
- `Mesh/generator/centerline/reference_normal` or
  `Mesh/generator/centerline/frame/reference_normal` orients the first swept
  cross-section. Later frames are transported with minimal twist.
- `Mesh/generator/borehole` defines the swept cross-section sketch. The solver
  also accepts `Mesh/generator/sketch` as an alias. The sketch uses ordered
  `surfaces[*].r` cumulative radial walls and `layers[*].mesh_block_id`,
  `inner_surface`, and `outer_surface` with the same outward ordering rules as
  layered boreholes. Swept plugs are not supported yet.
- `Mesh/generator/core_type` selects the center topology for 3D layered
  boreholes and swept borehole sketches. The default `prism` mode fills the
  first radial interval with a triangular fan extruded to prisms; `hex` keeps
  the older central quad loop and hex core.
- `Mesh/generator/core_ratio` sets the central O-grid loop radius as a fraction
  of the first wall radius only when `core_type` is `hex`, and
  `Mesh/generator/start_angle` sets the first of the four sampled polar sketch
  angles in radians.
- `Mesh/generator/implicit_walls: true` on `SweptBoreholeMeshGenerator`
  registers swept radial wall implicit surfaces and tags radial wall quads as
  `GMP_Implicit`. This is distinct from the parametric `GMP_Sweep` geometry
  path; the saved GMP still stores local clipping planes, while the wall SDFs
  are reconstructed from the generator definition before implicit binding.
- `Mesh/generator/clip_to_envelope` clips layered surface frontiers against the
  first and last model surfaces before cell emission. This is useful for dipping
  faults or fracture bands that should terminate at the free surface or model
  bottom instead of protruding past the physical envelope.
- `Mesh/generator/triangulate_strips` makes the layered strip mesher prefer
  triangles over x-aligned quads. When enabled, same-source collinear boundary
  points are collapsed before triangulation so straight dipping interfaces can
  be meshed with large triangles instead of warped quadrilateral strips.
- `Mesh/generator/share_fracture_neighbor_vertices` propagates x breakpoints
  across frontiers adjacent to generated fracture surfaces. This is opt-in
  because those extra shared cuts can force long transition triangles in
  dipping-fracture termination regions.
- `Discretization/geometry` selects element geometry evaluation: `iso`
  uses projected isoparametric geometry DOFs, while `exact` evaluates curved GMP
  geometry directly during element integration.
- `Discretization/geom_tol` sets the geometry coordinate tolerance used by GMP
  and reference-coordinate checks. It defaults to `1.0e-8`; this option is
  grouped with `Discretization/geometry` for now and may move under `Mesh` in a
  future contract.
- `Discretization/order` is rejected. Initial mesh order now lives at
  `Mesh/adapt/order`.
- `Discretization/form_execution` accepts `compiled` (the default) and the
  compatibility spelling `legacy`. Built-in acoustic, elastic, Maxwell and
  poroelastic DPG operators and their interface couplings use canonical Forms.
  Cartesian frequency-domain acoustic/elastic Galerkin and supported transient
  Galerkin families also use Forms under either setting. `legacy` does not
  reactivate retired handwritten kernels; unavailable Forms are reported explicitly.
- `Discretization/normal_backend`, `Solver/cholesky_backend`, and
  `Solver/herk_backend` are deprecated compatibility options for the retired
  scalar normal-system implementation. Their `cpu`/`gpu` values remain
  validated but have no effect on Forms. Use `Solver/gpu_backend` to select
  the complete device pipeline, or `none` for CPU execution.
- `Solver/relaxed_assembly` enables relaxed normal-system assembly. It defaults
  to `false` unless `Solver/mode` is `fast`, which defaults an unset value to
  `true`. The removed `Discretization/relaxed_assembly` location is rejected;
  use the Solver field.
- `Discretization/gram_reg` sets the finite, nonnegative relative Gram diagonal
  inflation in relaxed assembly (default `1e-5`). The FP32 Cholesky attempt uses
  `G(i,i) * (1 + gram_reg)`. Set `0` to disable inflation.
  Unrelaxed assembly and the existing FP64 fallback use
  the original Gram matrix. The setting applies to Forms normal assembly.
- `Solver/macro_precision` accepts `native` or `fp32`. In a double solver,
  `fp32` stores retained macro operators in single precision and performs macro
  matrix-vector products in FP32 while keeping outer Krylov vectors, reductions,
  residual updates, transfers, and other solver arithmetic in double precision.
- `Solver/galerkin_multigrid` opts Galerkin discretizations into the configured
  `Solver/grids` hierarchy for frequency-domain jobs. It defaults to `false`,
  preserving their one-grid direct MUMPS path. Transient Galerkin jobs retain
  the configured hierarchy independently of this flag. All Galerkin jobs
  require a MUMPS-enabled build and select MUMPS for coarse solves.
- Unit scaling is controlled by `disable_scaling`, `scaling`, `f0`,
  `omega_nd_center`, `modulus_nd_center`, `length_scale`, `time_scale`,
  `mass_scale`, and `Units/defaults`.
  `scaling: "robust"` enables a two-pass material-probing path that chooses
  final runtime scales from robust material centers before assembly setup.
  In robust mode, `omega_nd_center` defaults to `1`, and
  `modulus_nd_center` defaults to the same value so first-order elastic
  `omega*rho` and `omega*S` terms are balanced around order one.
- EM simulations use a time reference `f0/f_scale` and length reference
  `c0*f0/f_scale` by default, with `c0` expressed in km/s. These values are
  installed in the standard unit basis, giving unit normalized vacuum light
  speed and placing the reference angular frequency near `2*pi*f0`
  (`2*pi*10` by default). Material coefficients, coordinates, and frequency
  are converted through the units system rather than compatibility aliases.
  Explicit `time_scale` and `length_scale` values override these defaults;
  `disable_scaling` means that inputs already use a consistent normalized
  Maxwell basis.
- Without an explicit `vadapt`, Maxwell mesh adaptation derives local resolution
  from permittivity, permeability and conductivity (or reciprocal resistivity).
  `Mesh/adapt/elems_per_wave` applies to the shorter of the phase wavelength
  `2*pi/abs(Re(k))` and the one-e-fold decay distance `1/abs(Im(k))`, evaluated
  at the adaptation frequency. Thus conductive earth is sized by skin depth,
  while a lossless dielectric is sized by wavelength. Material segment bounds
  provide conservative sizing for heterogeneous media; tensor conductivity and
  Cole–Cole response use conservative bounds. Explicit `vadapt` retains its
  existing override semantics. Cold-plasma media require that override; it must
  account for both phase and attenuation scales, not just phase velocity.
- `Solver` remains permissive while FS_MG options are still read directly by
  solver code.


## Solver configuration boundaries

`Solver/precision` selects `single` (default) or `double` in the dispatcher;
calling a backend executable directly fixes that choice. The separate
`solve_precision` and `operator_precision` options seed retained workspace and
operator settings, then explicit `macro_precision` and `patch_precision` take
precedence. These options do not change the executable's scalar kind.

The CPU iteration budget `max_iter` defaults to 300. `tolerance` is the iterative
convergence target; `acceptance_residual` is a separate final residual ceiling,
defaulting to `1e-3`, in CPU and resident GPU bookkeeping. FGMRES fallback defaults
to enabled with restart length 5 and a separate budget of 50; a zero
`fgmres_max_iter` uses the remaining `max_iter` budget.

`gpu_backend` accepts `none`, `metal`, `cuda` (TF32 enabled), or `cuda:fp32`.
`forms_backend` selects all supported assembly stages together: `cpu` (default),
`auto`, `metal`, or `cuda`. Omission selects CPU assembly independently of the
solve backend. Explicit `auto` follows `gpu_backend`, including its compiled
default. Select `metal` or `cuda` explicitly to enable GPU assembly. An explicit selection changes assembly only; the resident
solve backend remains `gpu_backend`. Metal automatically uses verified FP32
fine macro construction, patch accumulation/inversion, parity checking and
compatible device-to-solver publication. Unsupported stages, FP64 requirements
and numerical recovery retain CPU fallbacks. There are no per-stage activation
flags. `FS_PATCH_PROFILE` and other timing diagnostics do not change routing.
`op_max_blocks`, `op_min_block`, and `op_target_block` select upper-triangular
block storage independently for each CUDA/Metal resident symmetric operator
execution group. Defaults are `1`, `1`, and `0`, preserving full dense storage.
`op_max_blocks` and `op_min_block` must be positive integers;
`op_target_block` must be a nonnegative integer.

The final rounded execution-group order `N` determines blocking for every
operator in that group. The maximum permitted count per axis is
`max(1, min(op_max_blocks, floor(N/op_min_block)))`. This enforces
`op_min_block` on the resulting balanced block sizes, which differ by at most
one. A group smaller than `op_min_block` remains whole. With
`op_min_block=256`, rounded groups below 512 stay 1×1, order 512 can use 2×2,
and order 1024 can use 4×4. An operator of order 511 grouped at 512 therefore
can use 2×2.

When `op_target_block=0`, use the permitted count above. Otherwise also cap it
by the nearest integer to `N/op_target_block`, at least one, with exact
half-integer ties selecting fewer blocks. The selected count is stored with
each group; changing RHS count does not reinterpret its operator storage.

For example, `op_max_blocks=4`, `op_min_block=128`, `op_target_block=512`
selects 2×2 for order 1000 and 4×4 for order 2000. The target is soft; a
minimum block size or maximum count can result in larger blocks. Diagonal tiles
remain full, upper off-diagonal tiles are stored once, and application reuses
them by transpose/conjugate transpose through vendor GEMMs. Alignment padding
can reduce the ideal memory saving. More blocks can increase application time
and low-precision accumulation error. These settings apply to the resident GPU
dense execution representation, independently of `patch_storage`; CPU solver
storage is unaffected.

`gpu_operator_batch_mib` is a positive integer (default `4096`) controlling the
estimated matrix and row-metadata budget per resident dense operator group on
both CUDA and Metal. It also applies when the compiled backend is selected
implicitly. It is not a total GPU memory limit or a preallocation: shape buckets,
communication scheduling, and indexing limits can produce smaller groups. One
supported operator is indivisible even if it exceeds the requested budget.

Omission selects the compiled backend; availability and precision constraints
are checked at runtime. `gpu_transfer` and `metal_source` are read only with an
explicit backend selection. Resident GPU execution uses PCG for the coarse solve.

Redundant aliases are rejected: use `refinements`, `mitigate_blowup`,
`mumps_precision`, and `scale_macro` instead of `refinement_flags`,
`mitigate_blowups`, `mumps_single`, and `diagonal_scaling`. `hp_switch` is removed.
Use `fast` instead of `fast-3`; canonical precision values are `native`, `fp16`,
`bf16`, `fp32`, and `fp64` where the field permits them. `mumps_precision` uses
`single` or `double`. Storage and filter spellings are the schema enums;
`comm_mode` uses `persist-p2p`, and `comm_precision` has no `solve`/`fp32`/`fp64`
compatibility spellings. Assembly backend gates accept `cpu` or `gpu`;
`metal` is reserved for the resident `gpu_backend` selector. Geometry uses
`exact` or `iso`. Invalid backend names are errors.

`mumps_precision` controls MUMPS storage and arithmetic independently of the
executable precision. A single executable honors `double` while retaining its
single MG operators and vectors. This improves accumulation/factorization/solve
accuracy but cannot recover assembly precision. Omission preserves the single
default in single builds. Double builds retain their forced-double policy for
general and complex-symmetric systems. Double MUMPS factors consume more memory;
only the selected precision instance allocates its backing arrays.

The Solver schema remains intentionally partial for subsystem-specific expert
fields. Passing schema validation does not establish availability or numerical
support for a precision/backend combination. See the
[solver API guide](../../../src/Solvers/FS_MG/README.md) for lifecycle and
configuration ownership.

## Compatibility

This schema intentionally allows additional properties. Subsystem contracts can
tighten individual blocks as Fortran readers are converted to typed config
classes.

### Maxwell modes on 2D meshes

For `physics: em`, `dimension: 2.5` enables the Cartesian transverse Fourier
workflow using job `k_list` and optional `k_weights`. The internal orthonormal
Maxwell components are `(x,z,-y)`; the Fourier factor is `exp(i k_y y)`.

An integer `toroidal_mode` instead selects a single cylindrical harmonic
`exp(i n phi)` on the R-Z cross section, with `phi=-theta`. It accepts dimension
2 or 2.5 on a 2D build and does not execute Cartesian k quadrature.
`axisymmetric: true` selects this geometry with n=0 by default. Field and
magnetic-background components are `(R,Z,phi)`. The radius must be positive.
These modal Maxwell workflows require compiled DPG Forms and support primal
impedance excitation. Axis constraints, cylindrical PML, point-current sources,
and modal adjoint/incremental solves are not supported.

## Conductive EM reference scaling

Set root `em_reference_conductivity` to a positive finite representative conductivity
in S/m (for example 0.01 for 100 Ohm*m earth). The EM initializer then uses
`epsilon_ref = sigma_ref/(2*pi*f_ref)`, `mu_ref = mu0`, and `E_ref = 1 V/m`.
Here `f_ref` is the positive physical frequency selected for runtime scaling,
including `Mesh/adapt/f_low` or `f_adapt`. All spectral evaluations within this
basis hold the reference fixed. At `f_ref`, a matching earth has solver
effective permittivity approximately -i and permeability 1. Air retains its
physical displacement-current contrast.

`f0` (default 10; 1 is useful for MT) or `time_scale` sets the time reference.
Length follows from `1/sqrt(mu_ref*epsilon_ref)`, and the mass/current references
preserve both constitutive equations and SI field conversion. Magnetic receiver
output remains mu0 H in T. This option cannot be combined with `length_scale`,
`mass_scale`, or `disable_scaling: true`. Omission preserves vacuum EM scaling.
These controls belong at the simulation root, not under `Units`.

### Initialization sizing frequencies

`Solver/sizing_frequency_ratio` defaults to `1.5` and must be finite and greater
than one. Smaller values request denser geometric sampling. Both ordinary and
adaptive-hp sizing select at least four distinct effective task frequencies when
available, including the minimum and maximum. Fewer distinct frequencies are
all evaluated. Task frequencies follow the mesh adaptivity `f_low`, `f_high`, and
`f_adapt` policy before selection; tasks outside the active range reuse the
corresponding endpoint estimates. No sampling or extrapolation occurs outside
that range.

Memory, disk, and DOF estimates use log-log interpolation between samples, with
linear interpolation across zero-valued endpoints. These are resource estimates,
not guaranteed upper bounds for discontinuous mesh/order changes. Adaptive-hp
samples retain independent mesh construction.

For example, set `"sizing_frequency_ratio": 1.2` inside `Solver` for denser sizing.
FrequenSolve accepts the same option as `SolverConfig(sizing_frequency_ratio=1.2)`.
