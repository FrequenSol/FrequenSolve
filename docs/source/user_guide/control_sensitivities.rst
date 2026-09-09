Parameterized Properties and Control Sensitivities
==================================================

Parameterized properties describe a low-dimensional model directly in material
coordinates. They avoid first constructing a Cartesian image and then sampling
that image during finite-element assembly. Sauce evaluates the control basis at
the same quadrature points as the physical operator.

Optional total DPG gradients
----------------------------

``RTMControlSensitivityJob(..., gram_derivative="total")`` opts into analytic
Gram-matrix and trial-to-test derivatives in compatible Sauce builds. The
default, ``"frozen"``, preserves the existing approximate gradient and does not
allocate the additional forward-residual cache or perform its Gram solve.
For a native RWI job, set
``control_sensitivities: {"gram_derivative": "total", ...}`` in its job contract.

Writing :math:`v=G^{-1}(l-Bu)`, the reduced RWI objective contributes
:math:`-\tfrac12\operatorname{Re}(v^H\,\delta G\,v)` in addition to the
operator/source contraction. This uses the existing optimal residual; it does
not construct or invert a derivative matrix. Ordinary RTM/FWI also needs the
derivative of the trial-to-test map; the optional path includes that term and
caches a separately Gram-solved forward residual per element and source batch.
The graph-norm material contraction is analytic and shared with the acoustic
forms coefficient module, not a new legacy element kernel.

Frozen Gram remains the default for both RTM and WRI. Use ``total`` primarily
for objective-gradient verification; its practical effect depends on the case.
Total mode requires ``Solver/relaxed_assembly=false``. Fast mode otherwise
enables relaxed assembly, which can make the solved normal system inconsistent
with the separately evaluated residual objective. For finite-difference checks,
also use ``Solver/schur_precision="fp64"`` and a tight solver tolerance.

The initial total-gradient implementation supports full-dimensional Cartesian
acoustic material controls for RTM and native FWI ``linearize``, ``vjp``, and
``wri`` actions. Unsupported elastic, Galerkin, 2.5-D, cylindrical, geometry,
phase-derivative, focus, source-taper, spatial-window, and Born/JVP/model-normal
combinations are rejected. Do not pair a total VJP with the frozen Born JVP
when testing an adjoint identity. Keep the mesh, element orders, and quadrature
fixed for objective finite differences. All PML elements remain excluded from
sensitivities in both modes. The PML uses the one-dimensional outward extension
of the boundary model, not independent inversion controls; hold that extension
fixed during derivative checks. Thus ``total`` refers to the physical-domain
material derivative including its Gram dependence. Smoothing/preconditioning changes the
raw covector and should be disabled for these checks.

Control blocks and coefficients
-------------------------------

An ``id`` names one complete controlled property, not an individual control
point. Coefficients inside that block are positional and always retain their
declared order. The block name is the only identifier needed to match a model
property with ``/controls/<id>`` datasets in direction, current-model, and
gradient HDF5 files.

For a single controlled property, construct the vector layout from the property
itself instead of repeating the block metadata:

.. code-block:: python

   import frequensolve as fs

   sediment_sp = fs.ParameterizedProperty(
       {"file": "starting_model.h5", "dataset": "/Sp"},
       id="sediment_sp",
       units="s/km",
       transform="log",
       control=fs.HatControl(
           coordinate_system="top_relative",
           axis="below",
           origin=0.0,
           spacing=50.0,
           units="m",
           coefficients=[0.0] * 7,
       ),
   )

   controls = fs.ControlSpace.from_property(sediment_sp)
   direction = controls.write_hdf5("direction.h5", [0, 0, 1, 0, 0, 0, 0])

For several properties, use ``ControlSpace.from_properties(...)``. Blocks are
flattened in lexical block-ID order, while coefficients remain in their native
order within each block. This ordering is independent of material allocation,
MPI rank count, and thread count.

Material controls are real-valued. ``ControlSpace`` writes float64 coefficient,
direction, and gradient datasets and rejects complex inputs rather than
discarding an imaginary component. Sauce wavefields and frequency-domain
receiver data remain complex; the VJP convention is the transpose for real
model parameters,

.. math::

   \operatorname{Re}\langle J\,\delta\theta, y\rangle
   = \delta\theta^\mathsf{T} J^\mathsf{T}y.

Uniform hat controls
--------------------

``HatControl`` is the simple gridded 1-D case. Coefficient ``i`` is located at

.. math::

   x_i = x_0 + i\,\Delta x,

and neighboring values are joined by piecewise-linear hat functions. At any
coordinate, at most two coefficients contribute. Evaluation, a
:term:`JVP`, and a :term:`VJP` scatter therefore take constant work per
quadrature point and do not form a dense interpolation matrix. The update and
its sensitivity are zero outside the first and last nodes.

Use ``BSplineControl`` when nonuniform knots or higher-order continuity are
useful. Cubic is the default degree; any valid degree and knot vector may be
supplied. Both maps support global and registered coordinate systems, including
the surface-relative ``below`` coordinate used for a 1-D model beneath fixed
bathymetry.

Property transforms
-------------------

For ordered coefficients :math:`\theta_i` and basis functions :math:`B_i(x)`,
the control field is

.. math::

   s(x) = \sum_i B_i(x)\theta_i.

The ``identity`` transform adds this field to the reference property. The
``log`` transform multiplies the reference by :math:`\exp(s)` and is useful for
positive properties such as velocity, slowness, density, and moduli. Zero
coefficients reproduce the reference model exactly. Identity coefficients use
the property's units; log coefficients are dimensionless.

Born JVP and RTM VJP jobs
-------------------------

``BornControlSensitivityJob`` applies the native physical-operator
linearization to one control direction and writes incremental receiver data:

.. code-block:: python

   born = fs.BornControlSensitivityJob(
       "sediment_sp_jvp",
       simulation,
       [4.0 - 1.5j, 6.0 - 1.0j, 8.0 - 0.5j],
       direction=direction,
       current="current_controls.h5",
   )

``RTMControlSensitivityJob`` applies the exact real-model transpose of that
linearization and writes the coefficient gradient:

.. code-block:: python

   rtm = fs.RTMControlSensitivityJob(
       "sediment_sp_vjp",
       simulation,
       [4.0 - 1.5j, 6.0 - 1.0j, 8.0 - 0.5j],
       observed={"seabed": "observed_traces.h5"},
       gradient="control_gradient.h5",
       current="current_controls.h5",
   )

The current vector is optional when the coefficients embedded in the material
model are already current. After a nonlinear update, the forward operator and
its factors must be reassembled. With the default ``gram_derivative="frozen"``,
the native JVP/VJP does not differentiate the Gram matrix or optimal-test map.

This is the physical, optimize-then-discretize DPG gradient used by the native
Born/adjoint kernels. ``gram_derivative="total"`` instead differentiates the
fixed-discretization DPG normal equations for the supported VJP workflows,
including Gram and trial-to-test terms. Both modes reassemble the forward
operator at every nonlinear model; ``frozen`` describes the derivative
approximation, not reuse of a stale forward operator.

Time-reversal focusing
----------------------

``TimeReversalFocusJob`` implements a data-domain focusing objective that does
not require a modeled forward wavefield. For each frequency, Sauce
backpropagates the observed receiver data and evaluates

.. math::

   J_f(m) = -\operatorname{Re}\int_\Omega
   \frac{q_f(x;m)}{\left(\lVert x-x_s\rVert^2+\epsilon^2\right)^{p/2}}\,dx,

where :math:`q_f` is the observed-data adjoint field, :math:`x_s` is the
encoded source-field center, :math:`\epsilon` is ``softening``, and :math:`p`
is ``distance_power``. Minimizing the negative sign favors a large, correctly
phased focus near the target. The standard postprocessing stage sums the
per-frequency objective and gradient shards with the supplied ``weights``.

.. code-block:: python

   focus = fs.TimeReversalFocusJob(
       "focus",
       simulation,
       frequencies,
       observed={"seabed": "observed_traces.h5"},
       gradient="focus_gradient.h5",
       objective_file="focus_objective.h5",
       softening=12.5,
       distance_power=1.0,
       weights=frequency_weights,
       preprocess=[fs.PreprocessHook.trace_weight(receiver_weights)],
   )

The objective itself uses one observed-data adjoint solve per frequency. Its
model gradient uses a second solve with the inverse-distance pressure
functional, then contracts that focus-dual field with the saved observed-data
field through the native control VJP. Both solves reuse the same factorization. The
focus objective therefore avoids forward modeling while retaining an exact
two-state gradient rather than differentiating a heuristic image after the
fact. The scalar objective is stored in the HDF5 ``/value`` dataset. Keeping
loss values in a dataset leaves room for future per-source
or source-subset objectives used by inexpensive line-search evaluations.

Receiver and residual preprocess hooks are applied before backpropagation, so
``trace_weight`` can select or weight traces without materializing a second
receiver geometry. The current implementation supports full-dimensional
acoustic DPG problems. The linear objective is polarity sensitive; observed
data and receiver encodings must use a consistent phase convention.

Time reversal conjugates the observed receiver coefficients, not the complex
frequency. Sauce retains the original attenuating frequency in both solves and
throughout the PML, so the absorbing layer continues to dissipate energy.
Conjugating the frequency would turn physical attenuation into gain and is not
part of this workflow.

Receiver-wavenumber filtering
-----------------------------

Frequency-local spatial filters can be attached directly to a misfit receiver
group. They operate after receiver-device evaluation, including DAS gauge
integration, and use the authored receiver order as the along-cable FFT axis.

.. code-block:: python

   from frequensolve.units import ureg as u

   das = job.misfit.receiver_groups[0]
   das.add_scholte_notch(
       850 * u.m / u.s,
       relative_half_width=0.03,
       relative_taper_width=0.02,
   )
   das.add_slow_velocity_mute(
       1200 * u.m / u.s,
       1800 * u.m / u.s,
       mode="reject_slow",
   )

``scholte_notch`` smoothly rejects both signed wavenumber branches around
:math:`|k|=2\pi f/v`; a measured angular ``wavenumber`` may be supplied instead
of phase velocity. ``slow_velocity_mute`` rejects apparent velocities below its
stop/pass transition. ``mode="keep_slow"`` applies the complementary fan for
paired P/S experiments.

Both filters require a complete, regularly sampled dense cable. Curved cables
are supported when consecutive along-cable spacings are uniform. The same
linear operator is applied to observed, simulated, and Born traces, and its
exact conjugate transpose is used by the VJP, giving
:math:`J^H P^H P r` for an L2 objective.

Derivative validation
---------------------

Material controls are real even though frequency-domain data and receiver
encodings are complex. ``real_adjoint_test`` checks the corresponding native
JVP/VJP identity,

.. math::

   \operatorname{Re}\langle Jp,y\rangle = p^\mathsf{T}J^\mathsf{T}y,

without incorrectly generating complex model perturbations. The convenience
``dot_test`` method on ``ControlLeastSquaresProblem`` uses the same pairing.

``gradient_taylor_test`` validates any callback-based scalar objective and
gradient. It reports the unsubtracted first-order remainder, the
gradient-subtracted second-order remainder, and optional centered directional
derivative errors. For example:

.. code-block:: python

   report = fs.gradient_taylor_test(
       evaluate_objective,
       evaluate_gradient,
       model,
       direction,
       steps=[0.08, 0.04, 0.02, 0.01],
       symmetric=True,
   )

For the native physical DPG gradient, a fully reassembled Taylor curve reaches
a quadratic regime before flattening at the frozen test-map discretization
error. Refining the DPG mesh should lower that floor. A fixed-test-map check of
the JVP/VJP pair remains an exact algebraic transpose test.

Inverse-Hessian illumination preconditioning
---------------------------------------------

``estimate_gauss_newton_diagonal`` estimates the reduced source--receiver
illumination

.. math::

   D \simeq \operatorname{diag}(J^T W^2 J + R^T R)

without forming a Jacobian. Each Rademacher probe is authored in the isometric
real data layout and requires one VJP, but no Born solve or normal-Hessian
product. The estimate is independent of the data residual; the quadratic
regularization contribution is added exactly.

``DiagonalInverseHessian`` applies blockwise relative damping and dynamic-range
clipping before exposing a callable inverse action:

.. code-block:: python

   estimate = fs.estimate_gauss_newton_diagonal(
       problem,
       model,
       probe_count=4,
       seed=20260830,
   )
   inverse_hessian = fs.DiagonalInverseHessian(
       estimate.total,
       block_sizes=[block.size for block in controls.blocks],
       relative_damping=1.0e-2,
       maximum_inverse_ratio=1.0e3,
   )
   result = fs.minimize_lbfgs(
       problem.objective,
       problem.gradient,
       model,
       preconditioner=inverse_hessian,
   )

For L-BFGS this callable supplies only the initial inverse-Hessian action in
the two-loop recursion. Objective gradients and accepted secant differences
remain the exact, unnormalized VJP gradients, and the Armijo line search uses
their true directional derivative. This differs from nonlinear RTM image
normalization, which must not replace a control gradient or linear adjoint.

Frequency aggregation and variational smoothing
------------------------------------------------

An RTM task writes one coefficient-space covector per frequency. The standard
FrequenSolve postprocessing step then forms the weighted sum and applies the
control map's native finite-element Riesz map. This is the same ``--smooth``
stage used by Cartesian images, so local and submitted jobs schedule it after
all frequency tasks have completed:

.. code-block:: python

   rtm = fs.RTMControlSensitivityJob(
       "sediment_sp_vjp",
       simulation,
       frequencies,
       observed={"seabed": "observed_traces.h5"},
       gradient="control_gradient.h5",
       raw_gradient="control_gradient_raw.h5",
       weights=frequency_weights,
       smoothing=fs.VariationalSmoothing(
           kind="tikhonov",       # or "tv" / "tgv"
           wavelength_fraction=0.5,
           derivative_order=1,
           input_role="dual",
       ),
   )

For a requested ``control_gradient.h5``, task ``i`` writes
``control_gradient_i.h5``. Postprocessing preserves the weighted aggregate as
``control_gradient_raw.h5`` and writes the variationally smoothed result to the
requested final path. With no smoothing configuration, both files are still
written and contain the same raw aggregate. Omitting ``raw_gradient`` selects
the default ``<gradient_stem>_raw.h5`` path; it does not disable that output.

The native Tikhonov system is

.. math::

   (M + \alpha K_p)\,g = b,

where :math:`M` is the control-basis mass matrix and :math:`K_p` is its
derivative Gram matrix. By default, Sauce derives
:math:`\alpha=(\lambda v_{p,\min}/(2\pi f_{\mathrm{ref}}))^{2p}` independently for
each owning material layer. Here :math:`\lambda` is the authored smoothing
multiplier of the local inverse wavenumber, :math:`v_{p,\min}` is that layer's conservative P-wave speed, and
:math:`f_{\mathrm{ref}}` is the largest physical frequency in the aggregated
job. ``alpha`` or ``reference_wavelength`` may be supplied explicitly when a
model-derived wavelength is inappropriate. Native VJPs are dual coefficient loads, so their
right-hand side is :math:`b` directly. Set ``input_role="primal"`` only for
nodal coefficient values; those are first mapped with :math:`M`. TV uses the
same basis and quadrature with a lagged-diffusivity iteration.

Second-order TGV introduces an auxiliary spline field :math:`w` and minimizes

.. math::

   \frac12\lVert g\rVert_M^2-b(g)
   + \alpha_1\lVert g'-w\rVert_1
   + \alpha_2\lVert w'\rVert_1.

The coupled finite-element system is solved by lagged diffusivity with MUMPS.
This permits affine or smoothly varying intervals while retaining sharp
interfaces, avoiding much of TV's staircase bias. Wavelength scaling uses
:math:`\alpha_1=\lambda v_{p,\min}/(2\pi f_{\mathrm{ref}})` and
:math:`\alpha_2=r\alpha_1^2`, where ``tgv_ratio`` is the dimensionless balance
:math:`r` (one by default). ``alpha1`` and ``alpha2`` can instead be supplied
together as low-level overrides.

Native control smoothing currently requires a MUMPS-enabled Sauce build. The
assembled mass and regularization matrices remain sparse in the control basis;
MUMPS performs the control-system analysis, factorization, and solve. A build
without MUMPS rejects this postprocessing mode explicitly.

Cartesian image smoothing also accepts an explicit illumination policy through
``VariationalSmoothing(illumination_normalization=...)``. ``"none"`` is the
FrequenSolve default and preserves the raw linear adjoint before its FEM Riesz
map. ``"source"`` divides by a fixed forward/source-illumination diagonal, so
the map remains linear in the residual or Born data. ``"cross"`` additionally
uses adjoint illumination; it is nonlinear in the data and is intended only for
display or heuristic RTM conditioning, never as the adjoint in a matrix-free
``J.T @ J`` operator. The descriptive aliases ``"linear"`` and
``"nonlinear"`` canonicalize to ``"source"`` and ``"cross"`` respectively.

These policies are evaluated by Sauce while it aggregates native image shards.
FrequenSolve only validates and serializes the configuration; it does not
perform image cross-correlation or smoothing in Python.

Field representations and transfers
-----------------------------------

``FieldRepresentation`` is the common finite-dimensional seam for controls and
images. ``ControlRepresentation`` implements the one-dimensional hat/B-spline
basis, while ``CartesianGridRepresentation`` wraps the existing multilinear
image grid. An ``EvaluationContext`` names the coordinate system, semantic
axes, sample shape, and coordinates rather than passing grid-specific argument
lists through every operation.

All representations expose coefficient-to-sample evaluation and its exact
transpose, ``pullback``. Primal fields can be transferred by evaluating on a
chosen context and least-squares projecting into the target representation:

.. code-block:: python

   samples = source.evaluate(source_coefficients, display_context)
   image_coefficients = source.transfer_to(
       image_representation, source_coefficients, display_context
   )

This distinction matters: an image desired for visualization is a primal
evaluation/projection, while an imaging sensitivity is a dual and must use the
transpose pullback. A future FEM-mesh material backend can implement the same
representation contract without changing optimizers or visualization code.

Least-squares optimization
--------------------------

``ComplexDataRealifier`` exposes complex receiver samples to a real optimizer
by interleaving their real and imaginary parts. This operation preserves the
complex Euclidean norm and the real-model transpose convention; it does not
make the material controls complex. ``ControlLeastSquaresProblem`` combines
that mapping with nonlinear forward and native JVP callbacks, optional data
weights, and optional ``QuadraticRegularization``:

.. code-block:: python

   history = fs.OptimizationHistory(
       "loss_history.json", metadata={"run": "single_source_1d"}
   )
   problem = fs.ControlLeastSquaresProblem(
       controls,
       observed,
       forward=forward,
       jacobian=jvp_columns,
       weights=trace_weights,
       regularization=fs.QuadraticRegularization(curvature, weight=alpha),
       history=history,
   )

   options = fs.InexactNewtonOptions(
       max_iterations=20,
       max_cg_iterations=controls.size,
       gradient_tolerance=1e-5,
   )
   result = fs.minimize_inexact_newton(
       problem.objective,
       problem.gradient,
       problem.gauss_newton_product,
       controls.zeros(),
       options=options,
   )

The history distinguishes actual objective evaluations from optimizer
iterations and atomically replaces its JSON file after every record. Use
``OptimizationCheckpoint`` for the current real control vector and counters,
and ``OptimizationResult`` for the terminal status, loss components, and scalar
quality metrics. These formats are optimizer-independent; SciPy least-squares
drivers remain supported through ``problem.residual`` and ``problem.jacobian``.

``minimize_inexact_newton`` implements matrix-free Newton-CG with an
Eisenstat--Walker inner stopping criterion, non-positive-curvature truncation,
box constraints, and an Armijo backtracking line search. The callback supplied
as ``hessian_product`` determines the method: ``problem.gauss_newton_product``
applies :math:`J^T W^2 J + R^T R`, while a complete reduced-Hessian callback
selects full inexact Newton without changing the optimizer. A native JVP/VJP
pair can be supplied instead of an explicit Jacobian, so the normal matrix is
never constructed for large control spaces.

Frequency and Laplace continuation
----------------------------------

``ContinuationSchedule`` keeps each fixed objective explicit. A typical FWI
schedule starts with fewer real frequencies and stronger Laplace damping, then
adds frequencies while reducing the magnitude of the imaginary shift:

.. code-block:: python

   schedule = fs.ContinuationSchedule.joint_frequency_laplace(
       frequency_bands_hz=[
           [4.0, 6.0],
           [4.0, 6.0, 8.0, 10.0],
           [4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
       ],
       laplace_damping_hz=[1.5, 0.5, 0.05],
       max_iterations=[3, 4, 5],
   )
   continuation = fs.run_continuation(schedule, initial, solve_stage)

With Sauce's current transform convention, the helper emits negative imaginary
frequencies. ``solve_stage(stage, model)`` owns data selection and optimization
for one fixed stage and returns an object exposing ``model`` or ``x``.
``run_continuation`` warm-starts the next stage from that terminal vector.
Explicit ``ContinuationStage`` objects can represent nonuniform damping or
arbitrary frequency groups when a Cartesian schedule is inappropriate.

Changing model resolution between stages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``transition(previous_stage, next_stage, accepted_model)`` to
``run_continuation`` to change parameterization between solves. The hook is
called only between stages, receives a copy of the preceding terminal vector,
and returns the next stage's initial vector. The returned size may differ;
the optimizer result must still match its own stage's initial size. Each
``solve_stage`` invocation should create a fresh optimizer and rebuild the
control space, model/backend, bounds, regularization and preconditioner. Do not
reuse L-BFGS pairs or Newton state across a change of basis.

For unchanged property references, transforms and coordinate frames, transfer
the latent field through its representation, for example:

.. code-block:: python

   def transition(previous, following, accepted):
       source = representations[previous.name]
       target = representations[following.name]
       return source.transfer_to(
           target, accepted, common_sample_context,
           weights=quadrature_weights, tolerance=1.e-12,
       )

   result = fs.run_continuation(
       schedule, coarse_initial, solve_stage, transition=transition,
   )

``FieldRepresentation.project`` and ``transfer_to`` expose the LSQR stopping
``tolerance`` (default ``1.e-6``); it is not a physical transfer-error bound.
Use adequate sampling/quadrature for both bases, evaluate the transferred
physical field, and reject unacceptable projection or bounds-clipping error
before launching the next solve. With different references or transforms,
evaluate the physical material first and fit it in the destination
parameterization; raw coefficient interpolation is not a valid general
handoff. Future mesh representations can use the same hook with their own
prolongation or projection operation.

``result.model`` belongs to the final space only. ``stage_results`` and
``stage_initial_models`` retain the vectors in each stage's own space, so they
need not form a rectangular array. Checkpoints must identify the stage and
the complete parameterization, including basis/mesh, coordinates, transform
and reference; matching vector length or block names alone is insufficient.
The runner does not infer transfer semantics or validate backend identities.
Without ``transition``, the existing fixed-space behavior is unchanged.

Pass an ``iteration_callback`` to ``ControlLeastSquaresProblem`` when a driver
should replace its checkpoint after every new iterate rather than only at
normal termination.

Current solver support
----------------------

Native control sensitivities are supported for standard full-dimensional
acoustic and primary isotropic elastic DPG formulations. Supported acoustic
controls include compatible ``Vp``, ``Sp``, ``K``, ``beta``, and ``rho``
parameter spaces. Elastic controls use the existing isotropic
velocity/slowness, bulk/shear, p-modulus/shear, lambda/shear, and density
derivative paths. Unsupported formulations and model-dependent terms are
rejected explicitly instead of silently producing incomplete gradients.
