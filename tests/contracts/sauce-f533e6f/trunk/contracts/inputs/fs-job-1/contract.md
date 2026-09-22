# FS Job Contract v1

Status: initial
Visibility: public
Schema id: `fs-job-1`

## Summary

See the maintained [imaging guide](../../../docs/imaging/README.md) for imaging
workflow selection, configuration, qualification and known gaps.

A job file defines the workflow, task controls, result path, and simulation
source for a FrequenSolve run. Frequency-domain workflows use `f_list`; the
frequency-independent `raytrace` and `eikonal` workflows use versioned
`RayTracing` and `Eikonal` blocks.

## Versioning

- A job with `"schema": "fs-job-1"` is interpreted by this contract.
- A job without `schema` is treated as legacy v0 by current Fortran readers.
- v1 preserves legacy string simulation paths while adding inline simulation
  objects and file references.

## Required Behavior

- `name` names the task family and contributes to temporary task names.
- `project_path`, when present, defines the project root for this job and takes
  precedence over `simulation/project_path`; explicit command-line and
  environment project-path overrides remain higher precedence.
- `workflow` selects the run mode. Current public values are `forward`,
  `forward_df`, `forward_ds`, `adjoint`, `smooth`, `rtm`, `focus`, `born`,
  `lsrtm_gradient`, `fwi_operator`, `modal`, `size`, `raytrace`, `eikonal`, and
  `transient`.
- `simulation` may be a legacy path string, an `fs-file-ref-1` object, or an
  inline `fs-simulation-1` object. Legacy relative path strings may be resolved
  under job-level `project_path` when that field is present.
- `result_path` is honored as absolute when it starts with `/` (or a Windows
  drive prefix), even if the directory does not exist yet. Non-absolute paths
  keep the legacy behavior: an existing local path is used as provided,
  otherwise the path is resolved relative to the project path.
- The only command-line result-directory override is `--result-dir`; the job
  JSON field remains `result_path`.
- `f_list` is required by frequency workflows and prohibited for `raytrace`,
  `eikonal`, and `transient`. Current frequency readers
  accept complex values; real numeric values are the common forward path.
- `workflow: "forward_df"` computes the forward field and its analytic
  derivative with respect to real physical frequency, writing receiver datasets
  with the ordinary name and the `_df` suffix. `workflow: "forward_ds"` does
  the same for imaginary physical frequency and the `_ds` suffix. Each workflow
  computes only its selected derivative direction. `derivative_order` defaults
  to one and accepts orders through four. The solver writes every order through
  the requested maximum using `_df`, `_d2f`, `_d3f`, and `_d4f` (or the
  corresponding `s` suffixes). Higher derivatives use a raw-derivative
  recurrence with the frozen DPG optimal-test map and retain only one extra
  condensed right-hand side while the full DOF buffer is reused in place.
  Orders above one currently require DPG. Their scope is standard acoustic,
  elastic, poroelastic, or compiled Maxwell physics, plus coupled acoustic-elastic and
  acoustic-elastic-poroelastic DPG, with native receiver variables. The DPG
  optimal-test map is frozen. For acoustic, elastic, and poroelastic physics,
  artificial impedance and point-source spectra are frozen. Maxwell differentiates
  its impedance admittance, conductive and cold-plasma material dispersion, and
  Cartesian PML tensors. Maxwell supports 2D, 3D, and 2.5D at fixed transverse
  wavenumber or toroidal mode. Its incident-field derivative supports isotropic
  `MaxwellPlaneWave` and vacuum `cylindrical_bessel` impedance sources; Gaussian
  beams and anisotropic incident eigenmodes are rejected. Maxwell phase-objective
  adjoint gradients remain unsupported. Maxwell recurrence loads use per-source
  normalization estimated from the previous solution magnitude and frequency;
  receiver value frames restore the raw physical-frequency derivatives.
  Acoustic and elastic PML stretching is differentiated; poroelastic PML
  transforms remain frozen. Kjartansson constant-Q material dispersion is differentiated for
  acoustic, ISO/TI/TTI elastic, and direct or ISO/VTI/TTI-frame poroelastic
  materials; poroelastic Darcy/JKD frequency dependence is also differentiated. DPG gravity surfaces are
  differentiated, including time-domain external pressure and frequency-domain
  pressure with the selected explicit derivative datasets. A frequency-domain
  field missing a derivative required by the requested order is rejected. Attenuative tangents require nonzero
  frequency. `Discretization/DPG_trial_to_test` in the selected simulation
  controls whether the base-frequency DPG trial-to-test action is stored or
  recomputed element by element for the tangent load. The former
  `frequency_tangents` job object is not accepted.
- `workflow: "raytrace"` requires a job-level `RayTracing` object satisfying
  `fs-ray-tracing-1` and prohibits `f_list`. Ray tracing is not dispatched as a
  synthetic zero-frequency task.
- `workflow: "eikonal"` requires a job-level `Eikonal` object satisfying
  `fs-eikonal-1`, prohibits `f_list`, and runs the real64 native-mesh acoustic
  first-arrival solver without constructing FEM degrees of freedom.
- `workflow: "transient"` requires a job-level `Time` object satisfying
  `fs-time-domain-1` and runs as one task without `f_list`. The first public
  slice supports full-dimensional Cartesian Galerkin acoustic, elastic, and
  coupled acoustic-elastic waves with Newmark average acceleration, plus
  pressure-form Galerkin Darcy flow and temperature-form thermal diffusion with backward Euler. Receiver and
  configured wavefield outputs are written as deterministic per-step HDF5
  shards, and optional rank-local checkpoints can resume the same fixed
  interval.
- Job files must not define `f_adapt`. Mesh adaptivity frequency controls now
  live in `simulation/Mesh/adapt` as `f_low`, `f_high`, and `f_adapt`.
- `k_list` is required for half-dimension workflows. `k_weights` may provide
  matching quadrature weights; otherwise current readers compute trapezoid
  weights from sorted signed `k_list`. `k_units` defaults to `1/m`.
- `Imaging` is required by imaging workflows and smooth runs. Forward plus imaging
  evaluation cases should satisfy `inputs/fs-imaging-1`; smooth-only runs may use
  a narrower subset while the shared reader remains permissive.
- `Modal` is required by the `modal` workflow. It defines an elliptic contour,
  a range of physical source-encoding fields used as probes, a configurable
  simultaneous-RHS limit, initial quadrature order, adaptive-refinement and
  residual tolerances, numerical-rank tolerance, and an optional retained-mode cap.
  Its `center` defaults to and, when supplied, must
  match the active `f_list` entry so initialization and every contour solve use
  one fixed reference discretization. A zero `probe_count` uses up to 32 available
  source probes independently of `rhs_batch_size`. Beyn reduction uses recovered L2 field coefficients;
  trace unknowns are not included. Verified extraction doubles `quadrature_order`
  until successive pole sets stabilize or `max_refinements` is exhausted.
  `require_operator_residual` cannot pass with the current field-only seismic
  adapter because a true DPG residual also needs trace unknowns. Saturated probe rank and unverified pole sets
  are reported explicitly; the DPG response family remains experimental and is
  not certified analytic in complex frequency.
- `lsrtm_gradient` is the Cartesian linearized-residual workflow. For every
  frequency task it solves the background forward problem, solves the Born
  forward problem for the supplied Cartesian image direction, forms
  `F0 + Jm - d` and its L2 objective inside the executable, and solves one
  incremental adjoint to write the batch gradient. It requires observed data
  for every configured misfit receiver group and does not use native
  `control_sensitivities`.
- `fwi_operator` exposes an immutable, task-local objective linearization.
  `linearize` writes version-3 distributed objective state and an optional
  gradient. `jvp`, `vjp`, and `normal` use one ordered `controls.active`
  subspace and canonical `direction`/`covector` files. Joint `normal` computes
  `J*J`, including cross-block terms, with saved weights, phase floors and masks.
  Every action validates the frozen state, control registry and MPI partition.
  Complex controls use real-interleaved coordinates and the real-Hermitian pairing.
  The `wri` action instead solves one reduced wavefield-reconstruction
  subproblem per source-encoding RHS,
  `min_u 0.5 ||B(m)u-q||^2_G^-1 + 0.5 lambda ||Pu-d||^2_s^-2`.
  The positive `wri/penalty` is `lambda`; `wri/data_scale` resolves to a scale in each
  selected receiver component's coordinate units (unit-aware `auto` by default). The observation term is
  inserted into the uncondensed DPG normal system before bubble condensation,
  so the reconstructed field uses the same coupling and static-condensation
  path as an ordinary solve. When `control_sensitivities` is present, Sauce
  uses the envelope theorem to write the reduced model covector from local
  PDE-residual contractions; no wavefield adjoint solve is required.
  At a receiver point incident on multiple broken DPG elements, Sauce applies
  the penalty to every incidence with weights summing to one. This keeps the
  assembled operator and reported objective identical while also discouraging
  disagreement between reconstructed traces at the observation boundary.
  This implementation supports full-dimensional acoustic, classic elastic and
  Maxwell DPG, multiple waveform least-squares receiver groups, dense or sparse point
  receivers, identity or acoustic/elastic `up_down` projections, and point-local
  preprocessing. Select groups with `receiver_groups`, select one with
  `receiver_group`, or omit both to use all imaging groups. The selectors are
  mutually exclusive; selected names must be unique and resolve to imaging
  misfits. Each group uses its own observed data, projection, and pipeline.
  Observed-stage hooks operate on fixed data. Simulated stages must be fixed
  linear local operations; residual stages supply finite nonnegative objective
  weights. Projection precedes preprocessing in receiver coordinate units, then
  WRI component scales are applied. Spatial trace-pair filters, postprocessed
  fields, and sample averaging remain unsupported. FrequenSolve remains responsible for selecting and updating
  the penalty, alternating or variable-projection model/source updates, and
  continuation across tasks.

  Source-independent dense WRI observations share the reconstruction matrix
  and factorization across source batches, including a shorter final batch.
  Sparse layouts and potentially source-dependent preprocessing error before solver
  setup if any configured batch has multiple RHSs. Batch sizes are never reduced
  automatically. Explicit single-RHS batches remain supported and rebuild per source.
  Receiver/component-only masks and trace weights preserve dense factor reuse.
  Observed group/component `trace_normalize` requires one complete source batch;
  multiple batches are rejected to avoid batch-dependent normalization.
  Missing sparse rows impose no observation penalty. Sparse sampling weights
  are retained. The objective artifact's `/data_scales` concatenates component
  scales and labels each with its `receiver_groups` attribute. PDE and data objective terms and model
  covectors undo numerical RHS normalization, so numerical source scaling does
  not change the relative penalty. This corrects the earlier single-RHS path,
  whose data/PDE balance depended on that normalization. Existing WRI penalty
  choices may therefore require retuning. The new default `data_scale: auto`
  (including omission) uses one solver unit per component, expressed in that
  component's coordinate units. This is independent of sources and observations.
  A positive number supplies an explicit coordinate-unit scale; extreme
  data/PDE weight ratios can still be ill-conditioned. Existing jobs that omitted
  the field change behavior; explicit `data_scale: 1` restores that scale choice.
  The objective artifact always writes `/data_scales`, with `components`,
  `coordinate_units`, and `policy` attributes (`auto_solver_units` or `explicit`).
  The legacy scalar `/data_scale` is written only for explicit numeric input.

  Optional `wri/curvature: fixed_wavefield | joint_schur` changes
  `model_covector` to a model-normal action on `model_direction`. The former
  holds the reconstructed field fixed; the latter eliminates its increment
  from the joint Gauss–Newton system. Both retain cross-parameter entries and
  require `Solver/relaxed_assembly=false`, frozen Gram weights, and unwindowed material controls in uncoupled
  acoustic, classic elastic or Maxwell DPG. They are positive-semidefinite approximations,
  not exact reduced Hessians. The objective output still describes the base
  reconstruction; its `value` dataset records `curvature` and `gram_derivative`
  attributes. WRI reconstructs the base state within each invocation; these
  actions do not consume a waveform/phase saved-linearization artifact.
  See [WRI curvature and costs](../../../docs/imaging/wri.md).
- `control_sensitivities` selects native material-control sensitivities instead
  of a Cartesian image for a `born`, `rtm`, or `focus` workflow. `born` requires a
  `direction` HDF5 file for its JVP; `rtm` requires a `gradient` HDF5 output path
  for its VJP. An optional `objective` HDF5 path makes the same RTM invocation
  write the robust scalar data objective evaluated before its retained-state
  adjoint solve. Each RTM frequency task writes `gradient_<task>.h5` and, when
  requested, `objective_<task>.h5`; the standard
  `--smooth` postprocess applies optional nonnegative `weights`, sums those
  coefficient covectors and scalar objectives, writes `raw_gradient` (or
  `gradient_raw.h5`), and writes the final `gradient`. Optional `Smoothing` applies a
  representation-owned Tikhonov, TV, or second-order TGV variational Riesz map
  after aggregation. The parts may be native `/controls/<block>` files or
  `fwi_operator` covectors (`fs-control-vector-1`, qualified
  `/controls/model.<block>` datasets); the reader accepts both layouts, ignores
  blocks outside `active`, and rejects an unsupported `/schema` or `/packing`.
  Joint sources produce joint outputs: model blocks are written as
  `/controls/model.<block>`, the remaining joint blocks (source, reflectivity)
  are summed with the same weights, the `/support` masks are copied from the
  first source, and `/schema`, `/packing`, `/state_fingerprint` and
  `/control_registry_fingerprint` are carried through (every part must share
  both fingerprints), so the smoothed covector can be used directly as
  `fwi_operator.direction`. Native sources produce native outputs.
  Optional `input` names one such vector explicitly: the postprocess then reads
  that file instead of the `_<task>` parts, copies it to `raw_gradient`, and
  writes the smoothed `gradient`; `weights` and `objective`/`focus` aggregation
  are ignored. `f_list` remains required because wavelength-relative smoothing
  scales use its largest frequency. Relative `gradient`, `raw_gradient` and
  `objective` paths (and the focus `objective`) resolve under the job's result
  directory, like the `fwi_operator` outputs, so the same relative string names
  both the `fwi_operator.covector` parts and the smoothing input; a relative
  `input` is looked up there first and otherwise like `fwi_operator.direction`.
  `input_role: dual` is the default because a native VJP is already a weak
  coefficient-space load; `primal` mass-weights supplied nodal values before
  solving. `lambda` multiplies the conservative P inverse wavenumber
  `wavelength/(2*pi)` of the control's owning material layer at the largest
  frequency in `f_list`;
  `alpha` may override the resulting Tikhonov/TV variational coefficient
  directly, and `reference_wavelength` may override only the model-derived
  wavelength. For TGV, the wavelength-scaled defaults are
  `alpha1 = lambda*wavelength/(2*pi)` and
  `alpha2 = tgv_ratio*alpha1^2`; explicit `alpha1` and `alpha2` may replace both
  defaults together. Native control smoothing currently requires a
  MUMPS-enabled build. The optional
  `current` HDF5 file replaces the material model's
  configured control coefficients before the baseline operator is assembled.
  Optional `active` lists the model-parameter blocks included in the packed
  direction and gradient vectors; omitted `active` selects all registered
  blocks. Parameters outside that subspace remain fixed at their current values.
  Optional `spatial_window` restricts native JVP and VJP sensitivity
  contractions to one physical `x`, `y`, or `z` interval while leaving the
  baseline forward fields and receiver residuals unchanged. `minimum` and
  `maximum` define the closed interval; `taper` defaults to zero and otherwise
  gives the raised-cosine transition distance at both edges, so it cannot
  exceed half the interval width. `units` defaults to metres. The same real
  weight is applied to volume, model-dependent source, and impedance terms in
  the Born and transpose paths.

  Native Galerkin actions now include uncoupled acoustic primary-property and
  elastic material controls, including model-dependent Robin/impedance terms;
  axisymmetric acoustic Galerkin is supported. TTI `theta`/`phi` controls use
  analytic angular derivatives. Phase actions include mixed material/frequency
  terms for acoustic and elastic Galerkin volumes and classic/weak-symmetry
  coupled DPG. Remaining formulation and boundary restrictions are listed in
  [the derivative capability guide](../../../docs/imaging/derivatives.md).
  [Multiparameter curvature diagnostics](../../../docs/imaging/curvature.md)
  operate on the saved `normal` action without choosing an optimizer.

  Cartesian Maxwell DPG supports ordinary material Born/JVP, RTM/VJP, and
  shared `fwi_operator` `linearize`, `jvp`, `vjp`, and `normal` actions.
  Geophysical properties, localized controls and complex physical-current
  signatures use the shared registry and versioned control/objective artifacts;
  see [EM workflows](../../../docs/imaging/em-geophysics.md). `normal` is a
  Gauss–Newton/positive-IRLS action, not a full Newton or source-eliminated
  Hessian. Saved operators reject nonlinear `source_scalar_fit` preprocessing;
  the legacy RTM calibration path remains separate.

  Maxwell total derivatives include fixed-stretch PML constitutive pullbacks
  wherever authored controls have support. Impedance admittance and supported
  material-dependent isotropic incident fields are differentiated. Ordinary
  current-source receiver derivatives are qualified for volume E/H channels.
  WRI has a separate receiver and curvature policy. Modal, cylindrical,
  geometry, phase-objective and source-taper extensions are not enabled here.

  `gram_derivative` defaults to `frozen`, preserving the existing approximate
  DPG derivative and runtime cost. Opt in with `total` for ordinary acoustic
  DPG Born/RTM or `fwi_operator` actions `linearize`, `jvp`, `vjp`, `normal`, and `wri`. The WRI
  derivative includes `-0.5 Re(v^H dG v)`, where `v=G^-1(l-Bu)`. Ordinary
  RTM/FWI additionally differentiates the trial-to-test map, including both
  `dB^H v` and the bilinear Gram correction. These are analytic contractions
  of graph-norm coefficients, not finite-difference element kernels or
  explicitly formed inverse derivatives. WRI reuses its existing optimal
  residual; RTM/FWI caches an additional element-local optimal residual and
  currently factors/solves its Gram matrix once per source batch. The frozen
  path performs none of this additional work. Total currently requires
  `Solver/relaxed_assembly=false`: the approximate fast-mode assembly can be
  inconsistent with the separately evaluated residual objective. Use fp64 Schur
  storage and a tight solve tolerance for verification. Both RTM and WRI keep
  frozen Gram as their default; total is an opt-in verification mode.
  Total requires full-dimensional Cartesian acoustic, classic elastic or
  Maxwell DPG material controls; Galerkin, weak-symmetry elasticity, geometry,
  focus, phase derivatives, source tapers and spatial windows are rejected.
  WRI normals still require frozen Gram. Keep the mesh, polynomial orders,
  quadrature, PML stretch and material-extension geometry fixed during checks.
  Seismic and WRI model gradients exclude PML elements; ordinary Maxwell
  includes its fixed-stretch constitutive pullback. Model-dependent refinement
  is not differentiated. Smoothing remains a postprocessing step, not part of
  the raw objective covector. Qualification remains combination-specific; see
  [capabilities](../../../docs/imaging/capabilities.md).

- `focus` backpropagates the observed traces and minimizes unnormalized negative
  wavefield energy around each encoded source center. `kind` selects:
  - `trfwi` (default): acoustic `-1/2 integral w |p|^2 dx`, where `w` is a
    Gaussian approximation to point evaluation. `softening` is its standard
    deviation in km.
  - `weft`: acoustic volumetric strain `theta = -p/K`, or elastic strain
    `epsilon`, with objective `-1/2 integral w |theta|^2 dx` or
    `-1/2 integral w epsilon:conjg(epsilon) dx`. `w` is the product of cosine
    tapers, zero outside a source-centered box whose half-width is `softening`
    km. Elastic DPG recovers strain from stress and constitutive compliance;
    Galerkin uses the symmetric displacement gradient. The compliance's
    material derivative is included in the gradient.
  These are frequency-domain adaptations inspired by WEFT and TRFWI. WEFT's
  acoustic volumetric measure and elastic rotation-invariant strain norm replace
  the paper's componentwise moment-tensor imaging functional. Frequencies are
  independent: there is no onset-time window, frequency coupling, or energy
  normalization. For complex frequencies the prescribed damping is retained.
  Scaling observations by `a` scales the objective and gradient by `|a|^2`;
  their global phase has no effect. Zero observations produce zero outputs.
  The aperture must be resolved by the mesh and quadrature. PML elements are
  excluded from evaluation. Supports Cartesian acoustic and classic elastic
  DPG/Galerkin with material controls, subject to their existing sensitivity
  restrictions. DPG observations must use native fields (`post_process: false`);
  material-dependent receiver operators are not supported. Willis coupling,
  geometry controls, 2.5D, and sensitivity tapers/windows are excluded.
  `distance_power` belongs to the removed signed-pressure objective and is
  rejected. Each frequency writes `/value` in its objective HDF5 shard and a
  control-gradient shard; existing frequency `weights` aggregate both products.
  See [wavefield focusing](../../../docs/imaging/focusing.md) for equations,
  examples, and the source papers.

- Cartesian image `Smoothing` accepts `tikhonov`/`l2`, `tv`, and `tgv`/`tgv2`.
  The TGV2 image smoother uses the mixed first-order form
  `alpha1*|grad(m)-w| + alpha2*|sym(grad(w))|` with scalar `m` and an auxiliary
  H1 vector `w`. It therefore requires only first derivatives of ordinary H1
  finite-element fields; it does not require a Hessian, an H2 space, or C1
  elements. `derivative_order` is consequently fixed at one for the image
  path. With wavelength-relative scaling,
  `alpha1=lambda*wavelength/(2*pi)` and
  `alpha2=tgv_ratio*alpha1^2`; an explicit `alpha1`/`alpha2` pair overrides
  those defaults. `epsilon` controls the fixed split-Bregman penalties, and
  `iterations` controls the nonlinear shrink/update count.


## Compatibility

This schema is intentionally permissive through `additionalProperties` because
job-level imaging and smoothing blocks are still read by subsystem-specific
Fortran code. The versioned `RayTracing` subtree is strict even though the job
container remains permissive.

A `forward` job may supply `control_sensitivities/current` to load a named
coefficient checkpoint before solution adaptation. This updates the nonlinear
material state without enabling Born or adjoint sensitivity assembly. Mesh
coefficient datasets must match the frozen property-space identity; they are
read as locally required slices.

The Cartesian 2.5D Maxwell workflow uses the existing signed `k_list` quadrature.
A simulation selecting cylindrical EM with `toroidal_mode` instead solves one
discrete toroidal harmonic and does not consume `k_list` or `k_weights`.


## Total spectral kernel derivatives

An `rtm` or `fwi_operator` job may add `"kernel_derivative": {"order": 4, "axis": "fourier"}`.
`order` defaults to one and accepts zero through four; `axis` defaults to
`fourier` (real physical Hz), or may be `laplace` (imaginary physical Hz).
The job requires `Image` with acoustic `fwi:acoustic`, property `vp`, an empty
image preprocessing pipeline, and field retention `none` or `all`.

Sauce solves both field hierarchies d0 through dn, differentiates the acoustic
material prefactor, and writes the complete product-rule derivative of the
physical Vp sensitivity kernel. All orders are raw derivatives, without
factorial normalization. One base operator is assembled per frequency and
reused for `2*(order+1)` solves per source batch. The ordinary forward traces
and misfit are evaluated once per batch.

The source signatures and source/receiver encoding weights are frozen at the
base frequency, as is the receiver residual driving the adjoint hierarchy.
Thus this is a spectral kernel derivative with a fixed data dual. It is not
by itself the gradient of a derivative-data objective or the spectral
derivative of a full FWI gradient with a frequency-dependent residual.
The DPG optimal-test map and artificial impedance also remain frozen, as in
`forward_df`. This option is separate from `control_sensitivities/gram_derivative`.

`residual` selects the data dual (default `"base"`, as above):

- `"derivative"` compares the raw order-n simulated traces with order-n observed
  traces: a `t^n` weighting of the damped traces.
- `"window"` with `"window": [a0, ..., an]` compares `sum_k a_k t^k` weighted
  traces (t in seconds) with one common residual; the polynomial degree sets
  the order.
- `"jet"` sums one l2 misfit per order, weighted by `"order_weights"`
  (default ones, highest positive). It applies the triangular transpose of the
  derivative recurrence by adding each order's scaled receiver adjoint to the
  matching plane of the upward adjoint recurrence.

In every mode the forward hierarchy samples the needed dn fields into
`<group>_dnf` or `<group>_dns` datasets (`_df`/`_ds` at first order), the
product-rule accumulator is unchanged, and the image is the exact gradient of
the selected objective up to the frozen quantities above. Every mode costs
`2*(order+1)` solves: a jet folds each order's weighted receiver covector into
the single adjoint solve of its plane. A time-domain
SeismicStore supplies observed `t^k` moments directly; an HDF5 trace file or
packed trace root must provide each needed derivative group.
With `control_sensitivities` and a `derivative`, `window` or `jet` residual the
job writes the native control gradient and objective of that time-weighted
misfit instead of an image, using the exact discrete transpose of the
recurrence and analytic mixed frequency/material partials through order four
(acoustic DPG, frozen Gram, unstretched cells). `workflow: fwi_operator`
accepts the same key with a `derivative` or `window` residual and a waveform
comparison for `linearize`, `jvp`, `vjp` and `normal`; `jet` is rejected there.
`"source_derivative": "total"` includes the point-source frequency derivative
in the forward hierarchy, matching `forward_df` and recorded time moments.

The initial scope is Cartesian 2D/3D acoustic DPG, nonzero complex frequency,
point sources, native receivers, and lossless or Kjartansson material response.
PML propagation is differentiated. Projection, WRI, control sensitivities with
a `base` residual, phase-derivative comparisons, incident/equivalent sources, and gravity-surface
forcing are rejected. Builds must support parallel HDF5, including serial runs.
See the [native workflow guide](../../../docs/imaging/spectral-kernels.md)
and [job example](examples/total-kernel-derivative.json).

### Shared inversion state and source controls

The required `fwi_operator/controls/active` list selects ordered blocks named
`model.<control ID>` and `source.<physical ID>.<position|mechanism|signature|signature_df>`.
Unknown, repeated and unsupported blocks fail. `controls/state` optionally
applies a complete canonical baseline before assembly; `controls/state_output`
exports that baseline. `controls/manifest` exports the resolved registry and all
MPI ownership descriptors. Both output paths are exact in a single-task job and
receive the `_<task>` suffix before the extension when `f_list` has more than one
entry. A `state_output` records each mechanism block's physical scaling, so a
baseline written by one frequency task replays in another
([fs-control-state-1](../fs-control-state-1/contract.md)).

Operator inputs `state` (jvp, vjp, normal), `direction`, `objective_vector`
(vjp), `extension/direction` and `model_direction` are resolved per task when
`f_list` has more than one entry: the task-suffixed sibling `<stem>_<task><ext>`
is probed first, as an absolute path or as a regular file under the project path
(and, for inputs that are searched, as given). A missing sibling never fails the
job: the exact path is then resolved as in a single-task job. Single-task jobs
use the exact path. `controls/state` is always exact. A saved linearization
records its `task`; another task index rejects it.

`state_output` and `covector` files also carry one packed support bitmask per
block, `/support/<qualified block>`, from the frozen-baseline measure
`s_i = sum_q w_q |dm/dc_i(x_q)|` over the volume quadrature Sauce visits during
material interpolation. `controls/min_support` (default `0.01`) marks a DOF
unsupported when `s_i` is below that fraction of the block's median nonzero
measure; `controls/support_measure: true` adds `/support_measure/<block>`.
Both keys apply to `fwi_operator` only; the native `control_sensitivities`
workflow does not export support. See
[Shared inversion controls](../../../docs/imaging/controls.md#control-support).

Use `direction` for JVP/normal and `covector` for gradient/VJP/normal. The shared
HDF5 coordinate and identity contract is documented in
[Shared inversion controls](../../../docs/imaging/controls.md).
Objective state/vector version 3 supports MPI reload on the same realized mesh
partition. Legacy provider/joint vector fields and objective v1/v2 are removed
from the FWI interface. WRI retains its separate material contract above.

`source_controls` defines only the location derivative policy: `analytic` (default).
The deprecated `local_fd4` spelling is accepted as an alias for `analytic`; it no
longer enables finite differences. `reference_step` remains accepted with its
legacy default 1e-4 and range [1e-6,1e-2], but is unused. Unsupported analytic
basis or coefficient derivatives fail explicitly. Source positions remain fixed
inside an action. Candidate coordinates and mechanisms require a new baseline.
See [source controls](../../../docs/imaging/sources.md) for coordinates,
discretization capabilities and geometric restrictions. FrequenSolve orchestrates
state updates and optimization between calls.

### Auxiliary model extension

`fwi_operator/extension` selects `linearize`, `jvp`, `vjp`, `normal`, or `solve`
in an auxiliary physical-property space. Omit `control_sensitivities`. Each field
names a direct parameterized material property by its unqualified `control` ID
and defines exactly one axis:

- `lags`: uniform `count`, `origin`, `spacing`, and explicit time `units`.
- `offsets`: `half_offsets` as an array of spatial coordinate vectors and explicit
  length `units`. Each vector is half the separation between incident and test
  samples. `packet_mb` bounds native donor exchange. The midpoint halo artifact
  is inferred from the named mesh property space; `artifact` can override it.

Fields borrow spatial control maps, including meshed properties. Tap values use
the property catalog units without nonlinear model transforms or bounds. The lag
sum absorbs quadrature weights. Auxiliary inputs and outputs use
`extension/direction` and `extension/covector` with
[fs-extension-vector-1](../../outputs/fs-extension-vector-1/contract.md).
`extension/manifest` exports global spatial counts, physical axis coordinates and
basis/baseline identities. Linearize may return the residual extension covector.
Output filenames receive the usual frequency-task suffix; relative paths use
ResultPath. `extension/direction` follows the per-task input resolution of the
shared control inputs above. The ordinary `state` and `objective_vector` formats
remain unchanged.

`solve` requires `extension/solver` with positive `damping`, `solution`, and
`report` paths. It solves one regularized quadratic shared across source batches
of this frequency task. `field_scales` supplies one physical amplitude per field:
taps equal scale times the Krylov coordinate. The normal receives damping squared
plus the applicable squared axis penalty: `lag_penalty*tau/lag_scale` or
`offset_penalty*|half_offset|/offset_scale`. A nonzero axis penalty requires its
positive scale value and matching time/length units. An optional auxiliary
`direction` supplies a warm start. Without `objective_vector`, the target is the
negative baseline residual; otherwise that vector is the target in the frozen
objective coordinates. `normal` itself remains unregularized.

CG verifies its final true residual and reports iterations, normal actions,
convergence, residual norms and quadratic change. `cache_mb` bounds resident
incident checkpoints per rank; excess batches use temporary storage. A zero
budget forces spilling. `workspace_mb` separately bounds retained reduced-gradient
wavefields. `require_convergence` rejects an unconverged solve after emitting its
diagnostic artifacts.

For a reduced background gradient, select material blocks in `controls/active`
when creating the state, then use `solve`, `model_gradient=true`, and a top-level
`covector` output. This path requires a converged inner solve, waveform L2, Huber, or Student-t
objectives and the observed-data target (no `objective_vector`). Robust observed-data
solves update the true residual and IRLS metric, use backtracking, and check the
regularized objective gradient before publishing a reduced gradient. The
`max_outer_iterations`, `max_line_search`, `gradient_relative_tolerance`, and
`gradient_absolute_tolerance` solver settings control this iteration. The physical
covector includes both propagation dependencies, mixed scattering/material
partials, and supported source/receiver material derivatives. Acoustic and
isotropic, TI, and TTI elastic mixed volume coefficients are implemented. DPG keeps its test
metric fixed. Source/geometry controls and unsupported constitutive mixed
partials are rejected. For auxiliary-only actions, physical block selection does
not perturb the fixed background; `controls/active=[]` is sufficient.

For a reduced physical-model Gauss–Newton action, use `solve` with
`reduced_normal: {}`, an ordinary control-vector `direction`, and a top-level
`covector`. Select the same active material blocks when linearizing and applying
the action. This first solves for the optimal extension, then eliminates its
linearized response using the regularized extension normal. Time lags and spatial
offsets use the same operation. `reduced_normal` optionally overrides
`relative_tolerance`, `absolute_tolerance`, and `max_iterations` for the response
solve; omitted values inherit the extension solver settings.

With physical prediction derivative `G`, scaled extension derivative `B`, frozen
positive receiver curvature `W`, and extension-coordinate regularizer `D`, the
returned action is `G*WG - G*WB (B*WB + D)^(-1) B*WG`. For robust losses, `W` is the
positive IRLS approximation at the solved prediction. This is a Gauss–Newton
Schur approximation, not the exact reduced Hessian. It omits residual-weighted
second derivatives. Both the optimized extension and response solve must converge
before a physical covector is published, irrespective of `require_convergence`.
The report includes a `reduced_normal` object with method `gauss_newton_schur`,
response iterations, true residual, and convergence status. This option is
mutually exclusive with `model_gradient=true` and requires the observed-data
target. The same material-only and fixed DPG-metric restrictions apply. See the
[example](examples/fwi-operator-reduced-normal.json).

Real Fourier frequencies, full-dimensional waveform comparisons and
`Solver/relaxed_assembly=false` are required. Volume scattering excludes PML and
boundary coefficients. Factors and incident states are reused inside a request.
Each frequency task currently has its own inner solve; a common extension across
a band requires composition of frequency normals and right-hand sides before
one shared solve. See the [solver guide](../../../src/Core/Manage/Simulation/extension.md).

The [iteration/composition design](../../internal/fs-extension-iteration-1/contract.md)
requires both Sauce-owned and FrequenSolve Python-owned iteration over the same
operator boundary, with frequency scheduling independent of driver choice.
Native callbacks exist; a retained Python session/parallel-band interface is
planned. This outline introduces no accepted job fields. File-based external
quadratic iteration can use the current actions, but does not retain factors
across executable invocations or supply robust candidate reweighting.


## Joint background and coordinate reflectivity

`fwi_operator/reflectivity` selects a total-field, zero-lag Born prediction
`u + P S(m,r)u`. It supports the standard `linearize`, `jvp`, `vjp`, and `normal`
actions, with the same saved objective-state and joint control-vector contracts.
It is mutually exclusive with `extension`, whose auxiliary vectors and inner
solve have different semantics. The optimizer and frequency scheduling remain
FrequenSolve responsibilities.

Select `parameterization: vp_ip` for acoustic `(Vp,Ip)`, `vp_vs_ip` for elastic
`(Vp,Vs,Ip)`, or `ip_is_rho` for elastic `(Ip,Is,rho)`. Here `Ip=rho*Vp` and
`Is=rho*Vs` use real reference-axis velocities; anisotropy, orientation and
attenuation remain separate background quantities. Reflectivity values are
signed absolute coordinate perturbations in native solver units. They do not
inherit material transforms or bounds, and are not dimensionless interface
reflection coefficients.

Each field specifies `name`, one-based material `layer`, and one-based chart
`axis`. It registers `reflectivity.<name>` in the shared control registry before
active selection. Choose either an independent `control` object (the existing
hat or B-spline map with initial coefficients) or `basis`, the identifier of a
direct material control map. A borrowed basis starts at zero reflectivity;
`controls/state` may replace it. Borrowing the spatial map does not tie the
reflectivity values or coordinate meaning to that material property or its
active selection. A complete `controls/state_output` includes both background
and reflectivity. The registry fingerprint includes the coordinate map, layer,
spatial basis, baseline values and active selection.

The acquisition adapter samples total physical fields, includes supported
material receiver derivatives in both directions, and uses the existing waveform
objective linearization. Consequently `normal` is the data/objective GN/IRLS
normal, including background–reflectivity cross blocks. It is not the unweighted
physical-field normal of the lower-level native session. Robust objective
weights are frozen at the saved baseline, just as in ordinary FWI actions.

This workflow currently requires full-dimensional first-order acoustic or classic
elastic DPG, compiled Forms, unrelaxed assembly, native receiver channels, and the
frozen trial-to-test policy. Coupled physics, Galerkin, 2.5D, axisymmetry, phase
objectives, explicit Dirichlet data, and sensitivity tapers are
rejected. Volume reflectivity excludes PML cells. `workspace_mb` bounds the
retained joint session; solver and acquisition buffers have their own owners.
One source batch is active at a time, and all propagations reuse the background
factors.

See [the example](examples/fwi-operator-reflectivity.json) and the
[native joint API](../../../docs/imaging/extended/joint_born.md).

## Retaining successful tasks for convergence retries

`preserve_task_outputs` is a boolean, default `false`. When true, packing retains
raw task files and a successful frequency task may be reused after changing
numerical solver settings. Failed tasks are still retried. Full original job and
simulation hashes remain in each producer result; the separate compatibility
fingerprint only controls this explicit reuse policy.

Changes to the physical simulation, acquisition, output request, task frequencies,
mesh controls, or structural solver fields (`grids`, `refinements`,
`refinement_flags`, `hp`, `galerkin_multigrid`, `relaxed_assembly`, `mode`)
invalidate reuse. The CLI PML frequency must also match. `--fresh` forces a new
solve under either policy. FrequenSolve refreshes external-input hashes when
saving/staging; save again after changing referenced input files.

The default packs completed shards, commits the packed dataset references to
their task results, then retires the duplicate shards and shared trace metadata.
Previously committed packed products survive unsuccessful replacement attempts
in both modes. An immutable segment still referenced by a current task or pack
is retained. This policy does not delete user-selected FWI checkpoint stems.
