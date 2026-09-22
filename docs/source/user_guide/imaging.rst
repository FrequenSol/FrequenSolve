Imaging and Inversion
=====================

``frequensolve.imaging`` declares an inverse problem once and then works with
short scalar calls and SciPy-style linear operators. One
:class:`~frequensolve.imaging.ImagingProblem` binds a simulation, a control
space, observed data, a misfit, the frequencies and a site; gradients,
Jacobians, normal operators and the :term:`FWI`, :term:`LSRTM`, :term:`RTM`
and focusing workflows all derive from it.

.. code-block:: python

   import frequensolve as fs
   from frequensolve import imaging as im

   u = fs.ureg

The same names are also available at the package root (``fs.ImagingProblem``,
``fs.DepthProfile``, ``fs.Misfit`` ...); the examples below use the ``im``
alias so that imaging objects stand out. The tutorial notebook
:download:`03_imaging.ipynb <../../../tutorials/06_outputs/03_imaging.ipynb>`
runs the complete workflow on a small model.

Overview
--------

The layers, lowest first:

- **Control spaces** (:class:`~frequensolve.imaging.ControlSpace`) turn material
  profiles, lattices, interfaces, source parameters and reflectivity fields
  into one real vector with transforms, bounds, coordinates and a support
  mask.
- **Data** (:class:`~frequensolve.imaging.ObservedData`,
  :class:`~frequensolve.imaging.DataVector`) and the **misfit**
  (:class:`~frequensolve.imaging.Misfit`) define the objective Sauce evaluates.
- The **problem** (:class:`~frequensolve.imaging.ImagingProblem`) and its
  **linearizations** expose ``value``, ``gradient``, ``jacobian`` and
  ``normal`` at any point.
- **Penalties, preconditioners and smoothing** live in FrequenSolve
  (:class:`~frequensolve.imaging.Tikhonov`, :class:`~frequensolve.imaging.TV`,
  :class:`~frequensolve.imaging.Diagonal`, :class:`~frequensolve.imaging.Smoothing`).
- **Workflows** (:class:`~frequensolve.imaging.FWI`, :class:`~frequensolve.imaging.LSRTM`,
  :func:`~frequensolve.imaging.rtm`, :func:`~frequensolve.imaging.sensitivity_kernel`,
  :class:`~frequensolve.imaging.TimeReversalFocus`) run the outer loops.
- **Low-level jobs** (:class:`~frequensolve.imaging.FWIOperatorJob`,
  :class:`~frequensolve.imaging.ControlGradientJob`,
  :class:`~frequensolve.imaging.ImageKernelJob`,
  :class:`~frequensolve.imaging.SmoothJob`) are what the layers above submit.

Sauce owns the physics and gradient smoothing; FrequenSolve owns bounds,
stages, continuation, penalties, optimizers, checkpoints and history. The
generic optimizer toolkit (:mod:`frequensolve.inversion`: L-BFGS, Newton-CG,
continuation schedules, history, derivative checks) is reused underneath and
remains available on its own.

Control spaces
--------------

Blocks
~~~~~~

A :term:`control block` names one Sauce control registry block (or one family
of blocks). It is defined by where it lives (a subdomain or surface), its
basis (exactly one of ``spacing``, ``count``, ``nodes``) and, optionally, how
its values are constrained (``transform`` and ``limits``). Extent is never
authored; it is the subdomain's span measured from the profile's datum. Lengths accept
plain numbers in the model's units or Pint quantities.

.. list-table::
   :header-rows: 1
   :widths: 34 26 40

   * - Block
     - Sauce block(s)
     - Notes
   * - :class:`~frequensolve.imaging.DepthProfile` ``(prop, subdomain, datum="top", spacing | count | nodes, transform, limits)``
       and ``DepthProfile.bspline(..., degree=3)``
     - ``model.<id>`` (hat or B-spline map)
     - Vertical 1-D profile; ``datum`` says where depth is measured from
       (below). Hat nodes end at the subdomain boundaries, B-spline knots
       are open-uniform over the same span.
   * - :class:`~frequensolve.imaging.GridParameters` ``(props, subdomain=None, spacing | shape | grid, transform, limits)``
     - one ``model.<id>`` per property (tensor-hat lattice)
     - Lattice over the subdomain's bounding box, or the whole model; nodes
       without support are frozen (see below).
   * - :class:`~frequensolve.imaging.MeshParameters` ``(prop, subdomain, frequency, epw)``
     - ``model.<id>`` (mesh-native nodal controls)
     - Sauce sizes and prunes the property space; the coefficient count is
       known after the first linearization.
   * - :class:`~frequensolve.imaging.InterfaceParameters` ``(surface, maximum_displacement, feasibility_band)``
     - ``model.<id>`` (implicit-surface control)
     - Coefficients of a radial-basis surface; Sauce defaults the limiter and
       band to 0.25 and 1.0 times the support radius.
   * - :class:`~frequensolve.imaging.SourceParameters` ``(sources="all", position, mechanism, signature, signature_df)``
     - ``source.<i>.{position, mechanism, signature, signature_df}``
     - One block per source and quantity; complex blocks are packed
       real-interleaved.
   * - :class:`~frequensolve.imaging.ReflectivityParameters` ``(parameterization, fields=[ReflectivityField(...)])``
     - ``reflectivity.<name>``
     - Joint reflectivity fields (see :ref:`imaging-extension`); no transform
       or limits.

.. code-block:: python

   space = im.ControlSpace(
       vp=im.DepthProfile("vp", "sediment", spacing=25 * u.m, transform="log", limits=(1450, 3500)),
       rho=im.DepthProfile.bspline("rho", "sediment", count=20, transform="log"),
       salt=im.InterfaceParameters("salt_top", maximum_displacement=150 * u.m),
       src=im.SourceParameters(position=True, signature=True),
   )
   lattice = im.GridParameters(["vp", "rho"], "salt_body", spacing=[100 * u.m, 50 * u.m])
   whole_model = im.GridParameters("vp", spacing=[100 * u.m, 50 * u.m])

A single block can be passed directly as ``controls=``. ``limits=(lo, hi)``
are value limits in physical units that the optimizer enforces as bound
constraints; they are optional and independent of extent.

Blocks and coefficients
~~~~~~~~~~~~~~~~~~~~~~~

The space fixes the vector: blocks are contiguous in the order they were
declared, coefficients keep their native order inside each block, and this
ordering is what Sauce receives as ``controls.active``. It does not depend on
material allocation, MPI rank count or thread count.

.. code-block:: python

   space.keys                   # ('vp', 'rho', 'salt', 'src')
   space = problem.space        # the same space, resolved against the simulation
   space.blocks                 # ('model.vp', 'model.rho', 'model.salt_top', 'source.1.position', ...)
   space.size, space.sizes      # optimizer vector length and per-block sizes
   space.restrict(["vp", "src.signature"])      # ordered subspace by key or address
   space.zeros(), space.random(seed=1)
   blocks = space.unpack(space.zeros())        # {'model.vp': array, 'model.rho': array, ...}
   space.pack(blocks)                          # back to one ControlVector
   space.bounds                 # optimizer-coordinate bounds derived from limits

Everything that depends on the layout (``blocks``, ``size``, ``slices``,
``zeros`` ...) needs the blocks resolved against a simulation: the number of
profile nodes follows from the subdomain's span, and source blocks expand to
one block per source. ``controls.bind(simulation)`` does that on its own
(and exposes the ``controls`` payload Sauce receives); the problem does it
when it is declared, so ``problem.space`` is the resolved space.

Control coefficients are real. For ordered coefficients :math:`\theta_i` and
basis functions :math:`B_i(x)` the control field is
:math:`s(x) = \sum_i B_i(x)\,\theta_i`. The ``identity`` transform adds this
field to the reference property; the ``log`` transform multiplies the
reference by :math:`\exp(s)` and is the natural choice for positive
properties such as velocity, slowness, density and moduli. Zero coefficients
reproduce the reference model exactly. Identity coefficients use the
property's units; log coefficients are dimensionless.

A :class:`~frequensolve.imaging.DepthProfile` without ``degree`` is a hat
profile: coefficient ``i`` sits at node :math:`x_i` and neighbouring values
are joined by piecewise-linear hat functions, so at most two coefficients
contribute at any point and evaluation, :term:`JVP` and :term:`VJP` scatter
cost constant work per quadrature point. ``DepthProfile.bspline`` uses a
B-spline basis (cubic by default) when higher-order continuity is wanted.

A depth profile is always vertical. Its ``datum`` says where depth is
measured from; ``spacing``, ``count``, ``nodes`` and ``limits`` are measured
in that frame:

.. code-block:: python

   im.DepthProfile("vp", "sediment", count=40)                       # datum="top": depth below the layer's upper surface
   im.DepthProfile("vp", "sediment", datum="global", count=40)       # the model's global vertical coordinate z
   im.DepthProfile("vp", "sediment", datum="below_seabed", count=40) # a coordinate system registered on the simulation
   im.DepthProfile("vp", "sediment", datum="bottom", count=40)       # depth below a named model surface

- ``"top"`` (default) follows the subdomain's upper surface, which is the
  natural choice for a 1-D column beneath bathymetry; the surface-relative
  system Sauce evaluates it in is created internally.
- ``"global"`` uses global ``z``; the extent is the subdomain's
  ``[min(upper surface), max(lower surface)]``.
- A coordinate-system name uses that system's single vertical axis
  (direction ``z``), oriented by the axis' ``positive``; a user-authored
  seabed-relative system is the typical case.
- A model-surface name measures depth below that surface; the extent is the
  subdomain's span measured from it and need not start at zero.

The keywords win over names; a name that is both a coordinate system and a
surface is rejected as ambiguous. Rendered profiles use the ``depth``
dimension (``z`` for ``"global"``) and plot vertically, depth increasing
downwards.

.. _imaging-support:

Support masks and conditioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A coefficient whose basis support does not intersect its subdomain has zero
sensitivity; one with a sliver of support has nearly zero. With level-set
surfaces, background nodes inside the body and interface nodes far from the
zero level set are in the same situation. Left in the optimizer vector, such
coefficients wreck the conditioning.

Sauce measures the support of every coefficient as its derivative measure at
the baseline, :math:`s_i = \sum_q w_q\,|\partial m/\partial c_i(x_q)|` over the
quadrature samples visited during material interpolation, applies the
relative threshold ``min_support`` (default :math:`10^{-2}` times the block's
median nonzero :math:`s_i`) and writes one packed bitmask per block into the
control state. The rule covers hat, B-spline and tensor-hat nodes outside a
subdomain, background blocks masked by a level-set blend, interface nodes
whose basis never meets the feasibility band, and reflectivity fields; mesh
controls are pruned natively and report full support.

FrequenSolve applies the mask: frozen coefficients are excluded from the
optimizer vector, written as zeros in Sauce vectors and states, and dropped
from read-back gradients.

.. code-block:: python

   problem.space.support["vp"]          # boolean mask, False = frozen
   problem.space.frozen_indices
   v.to_xarray("vp", frozen=float("nan"))   # render frozen nodes as gaps
   v.plot("vp")                          # frozen nodes appear as gaps

Because level-set support depends on the state, the mask is taken from the
first linearization of each stage and held fixed for that stage (the vector
dimension cannot change inside a stage solve), then refreshed at the stage
transition; nodes that become supported as an interface moves are picked up
by the next stage. ``ImagingProblem(min_support=...)`` and
``Stage(min_support=...)`` change the threshold. Before the first
linearization FrequenSolve uses a geometric fallback for layered models
(exact for 1-D profiles in any datum, bounding box for lattices), and
``space.with_support({...})`` installs an explicit mask.

State versus vector
~~~~~~~~~~~~~~~~~~~

Sauce distinguishes the complete baseline (every block) from directions and
covectors (active blocks only). Two vector types mirror that:

- :class:`~frequensolve.imaging.ControlState` is the complete baseline over
  every block. ``problem.state`` is the current one;
  ``state.with_update(vector)`` returns a new state, ``state["vp"]``,
  ``state.to_xarray()``, ``state.plot()`` and ``state.save``/``load`` are
  available.
- :class:`~frequensolve.imaging.ControlVector` lives on an active subspace
  (``space.restrict(...)``): gradients, directions and updates. It supports
  ndarray arithmetic, ``v["vp"]``, ``v.to_xarray()``, ``v.to_mesh()``,
  ``v.per_source()`` for source blocks, ``v.plot()``, ``v.norm()``,
  ``v.dot(w)``, ``v.clip()`` to the bounds, and ``save``/``load``.

Plain ndarrays are accepted everywhere a typed vector is.

Observed data and misfit
------------------------

Observed data comes from a finished forward job, a trace store, a
:class:`~frequensolve.seismic.traces.TraceDataset`, or a mapping of receiver
groups to files; frequencies are inferred when the source declares them.
Receiver-group names must match the simulation's acquisition, because the
misfit pairs observed and simulated groups by name.

.. code-block:: python

   observed = im.ObservedData(observed_job)                       # frequencies from the job
   observed = im.ObservedData({"seabed": "obs.h5", "das": "das.h5"}, frequencies=[3.0, 5.0])
   observed = im.ObservedData(observed_job, derivatives={"df": observed_df_job},
                              source_basis="source_geometry", missing="zero")

The misfit maps one to one onto Sauce's objective terms: a loss (``l2``,
``huber``, ``student_t``), a comparison (``waveform``, ``phase_derivative``),
a normalization (``observed_rms`` by default, ``explicit``, or
``Normalization.balance_artifact(file)`` written by a ``calibrate`` job),
per-group weights, preprocessing hooks and a receiver projection.

.. code-block:: python

   misfit = im.Misfit.huber(
       delta=1.345,
       preprocess=[im.Preprocess.offset_taper(d0=100 * u.m, d1=250 * u.m)],
   )
   misfit = im.Misfit(loss="l2", comparison="phase_derivative", normalization="observed_rms")
   misfit = im.Misfit.terms(
       im.ObjectiveTerm("hydrophone", loss="huber", weight=1.0),
       im.ObjectiveTerm("das", loss="l2", comparison="phase_derivative", weight=0.3),
   )
   misfit = im.Misfit.l2(projection=im.ReceiverProjection.up_down(impedance=1.5e6))

:class:`~frequensolve.imaging.Preprocess` has one constructor per Sauce hook
kind. Objective weights (``offset_power``, ``offset_taper``,
``component_scale``, ``frequency_weight``, ``trace_mask``, ``trace_weight``)
act on the residual; data transforms (``trace_normalize``,
``amplitude_clip``) act on the observed data; ``source_scalar_fit`` and
``source_spectrum_correction`` calibrate the source; ``receiver_ar1_whiten``,
``scholte_notch`` and ``slow_velocity_mute`` are fixed linear receiver
operators applied to observed, simulated and Born traces alike, with their
exact conjugate transpose used by the VJP.

.. code-block:: python

   das_hooks = [
       im.Preprocess.scholte_notch(850 * u.m / u.s, relative_half_width=0.03, relative_taper_width=0.02),
       im.Preprocess.slow_velocity_mute(1200 * u.m / u.s, 1800 * u.m / u.s, mode="reject_slow"),
   ]
   misfit = im.Misfit.l2(group_preprocess={"das": das_hooks})

``scholte_notch`` smoothly rejects both signed wavenumber branches around
:math:`|k| = 2\pi f / v` (a measured angular ``wavenumber`` may replace the
phase velocity); ``slow_velocity_mute`` rejects apparent velocities below its
stop/pass transition, and ``mode="keep_slow"`` applies the complementary fan.
Both filters need a complete, regularly sampled dense cable and use the
authored receiver order as the along-cable FFT axis.

A :class:`~frequensolve.imaging.DataVector` is a complex vector over the
:class:`~frequensolve.imaging.DataSpace` of the acquisition, ordered by
receiver group and then ``(frequency, source, component, receiver)`` with the
receiver varying fastest. ``problem.forward(v)``, ``problem.residual(v)``,
``J @ dv`` and ``problem.observed_vector()`` return one;
``vector.to_dataset()`` renders it as an xarray dataset by receiver group.

The problem, linearizations and operators
-----------------------------------------

.. code-block:: python

   problem = im.ImagingProblem(
       simulation,
       controls=space,
       observed=observed,
       misfit=misfit,
       frequencies=[2.0, 3.0, 5.0, 8.0],
       site=site,
       smoothing=None,
       workdir="fwi",
       name="fwi",
   )
   problem.space, problem.data_space, problem.state, problem.observed_data
   problem.capabilities()        # static validation of blocks, misfit and physics
   problem.dry_run()             # the linearize payload without submitting it

   v = problem.vector()          # active slice of the current state
   problem.value(v)              # scalar misfit at v (a ControlVector or array)
   problem.gradient(v)           # ControlVector
   lin = problem.linearize(v)    # one saved Sauce state per frequency
   lin.value, lin.report, lin.gradient
   J = lin.jacobian              # J @ dv -> DataVector, J.H @ r -> ControlVector
   H = lin.normal                # frozen Gauss-Newton Re(J^H W J), self-adjoint

The simulation is deep-copied when the problem is declared; the caller's
object is never mutated. Every linearization maps to one saved Sauce
``fwi_operator`` state and is cached by a fingerprint of the problem and the
full control state, so ``value``, ``gradient``, ``jacobian`` and ``normal``
at the same point cost one job family. Derivative actions (``jvp``, ``vjp``,
``normal``) reuse the saved state and are memoized per input vector.

Operators are :class:`scipy.sparse.linalg.LinearOperator` subclasses that
know their control and data spaces, accept typed vectors or plain arrays, and
compose with ``@``, ``+`` and scalars, so ``H + alpha * R.T @ R`` drops into
:func:`scipy.sparse.linalg.cg` or, with the ``inversion`` extra, into PyLops
through ``operator.to_pylops()``.

.. code-block:: python

   from scipy.sparse.linalg import cg

   R = im.Tikhonov(alpha=1e-2).bind(problem.space).operator()
   step, info = cg(H + R.T @ R, -lin.gradient, maxiter=20)   # one Gauss-Newton step

``problem.restrict(frequencies=..., active=...)`` returns a stage view that
shares the state, cache and site; a view may also override the misfit (or
just its ``loss``), the ``smoothing``, the support threshold and per-frequency
``weights``.

Sign and adjoint conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Material controls are real while frequency-domain data are complex. The
covector convention is the real part of the Hermitian pairing,

.. math::

   \operatorname{Re}\langle J\,\delta\theta, y\rangle
   = \delta\theta^\mathsf{T}\,\bigl(J^{\mathsf H} y\bigr)_{\mathrm{Re}},

with no factor of two: ``J.H @ r`` returns exactly that real covector.
Sauce's covector is the gradient of the misfit, that is the adjoint applied
to ``simulated - observed``; the classic RTM image in the
``observed - simulated`` convention is its negative. Frequency weights scale
the objective side (value, gradient, normal operator) while ``J`` stays
unweighted. Smoothing, when attached to the problem, is applied to
``lin.gradient`` only; ``J`` and ``H`` stay exact so that adjoint identities
hold.

Derivative validation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   report = problem.check(v, steps=(0.1, 0.03, 0.01, 0.003))
   report["adjoint"], report["normal"], report["taylor"], report["passed"]
   lin.jacobian.dot_test(seed=0)

``check`` runs the adjoint test :math:`\operatorname{Re}\langle J\,dv, r\rangle
= \langle dv, J^{\mathsf H} r\rangle` on random vectors, the normal consistency
test :math:`H\,dv = J^{\mathsf H} W J\,dv`, and a Taylor test of
``value``/``gradient`` (:func:`~frequensolve.inversion.validation.gradient_taylor_test`)
that reports the first- and second-order remainders. A fully reassembled
Taylor curve reaches a quadratic regime before flattening at the frozen
test-map discretization error; refining the mesh lowers that floor. For
finite-difference checks keep the mesh, element orders and quadrature fixed,
use a tight solver tolerance, and disable smoothing and preconditioning, which
change the raw covector. PML elements are excluded from sensitivities; the PML
uses the outward extension of the boundary model, which is held fixed.

Regularization, smoothing and preconditioning
---------------------------------------------

Three families with different roles:

- **Penalties** are part of the objective. They are evaluated in Python on
  the block coordinates (after the transform, frozen nodes excluded) and
  expose ``value``, ``gradient``, a Hessian operator and, for quadratic
  penalties, ``operator()`` returning ``R`` with ``hessian == R.T @ R``.
- **Smoothing** is Sauce's native gradient smoothing: a step transform on
  covectors that is not part of the objective.
- **Preconditioners** change the linear solve only.

.. code-block:: python

   penalty = im.Tikhonov(alpha=1e-2, order=1)                       # first-derivative seminorm per block
   penalty = im.TV(alpha=1e-3) + 0.5 * im.Tikhonov(alpha=1e-2, order=2)
   penalty = im.Tikhonov(alpha=1e-2, weights={"vp": 1.0, "salt": 0.1}, reference=problem.vector())
   penalty = im.Tikhonov(alpha=1e-2, length=100 * u.m)               # derivatives per 100 m instead

Penalties are scale free. Each lattice block is measured on its
nondimensional coordinate :math:`\xi = (x - x_0)/L`, with :math:`L` the
block's span per axis, and the penalty is a quadrature-weighted
discretization of a continuous seminorm over the unit interval (square,
cube):

.. math::

   \mathrm{Tikhonov}_k(c) = \tfrac12\,\alpha \sum_{\text{axes}}
   \int_{[0,1]^d} \Bigl|\frac{\partial^k (c - c_{\mathrm{ref}})}{\partial \xi^k}\Bigr|^2 d\xi,
   \qquad
   \mathrm{TV}(c) = \alpha \int_{[0,1]^d}
   \Bigl(\sqrt{|\nabla_\xi (c - c_{\mathrm{ref}})|^2 + \epsilon^2} - \epsilon\Bigr) d\xi .

First differences are weighted by their edge length (exact for the
piecewise-linear hat profile), second differences by their dual-cell length,
other lattice axes by trapezoid weights; TV evaluates the gradient per
lattice cell. The value of a fixed smooth field therefore converges as the
profile is refined instead of growing with the node count, and because the
default misfit normalization (``observed_rms``) makes the data term order
one, ``alpha`` between :math:`10^{-3}` and :math:`10^{-1}` is a meaningful
range. ``length=`` (a length, a per-axis sequence or a ``block -> length``
mapping) measures derivatives per physical length instead of per block span.
B-spline profiles difference their coefficients at the Greville abscissae.
Material blocks have unit weight; source, interface and reflectivity blocks
are unpenalized unless ``weights`` names them, in which case they receive a
ridge toward the reference. :class:`~frequensolve.imaging.TV` has a
lagged-diffusivity Hessian; ``TV(order=2)`` is a second-order TV. Full TGV is
a Sauce-side smoothing (below), not a Python penalty.
:class:`~frequensolve.imaging.Quadratic` wraps an arbitrary matrix.

Smoothing
~~~~~~~~~

.. code-block:: python

   smoothing = im.Smoothing(kind="tikhonov", wavelength_fraction=0.5, derivative_order=1)
   smoothing = im.Smoothing(kind="tgv", wavelength_fraction=0.3, tgv_ratio=1.0)

   smoothed_problem = im.ImagingProblem(simulation, controls=space, observed=observed,
                                        frequencies=[3.0, 5.0], site=site, smoothing=smoothing)
   smoothed = im.smooth(problem.gradient(v), smoothing, problem)   # one explicit vector

Attached to the problem (or overridden per stage), every gradient and
covector is smoothed by Sauce's ``smooth`` postprocess; :func:`~frequensolve.imaging.smooth`
runs the same Riesz map on one explicit vector through a
:class:`~frequensolve.imaging.SmoothJob`. History records it as a transform.

The native Tikhonov system is :math:`(M + \alpha K_p)\,g = b`, where
:math:`M` is the control-basis mass matrix and :math:`K_p` its derivative
Gram matrix. By default Sauce derives
:math:`\alpha = (\lambda\, v_{p,\min} / (2\pi f_{\mathrm{ref}}))^{2p}`
independently for each owning material layer, with :math:`\lambda` the
authored ``wavelength_fraction``, :math:`v_{p,\min}` that layer's conservative
P-wave speed and :math:`f_{\mathrm{ref}}` the largest physical frequency of
the job. ``alpha`` or ``reference_wavelength`` may be supplied explicitly.
TV uses the same basis and quadrature with a lagged-diffusivity iteration.
Second-order TGV introduces an auxiliary spline field :math:`w` and minimizes

.. math::

   \tfrac12\lVert g\rVert_M^2 - b(g)
   + \alpha_1\lVert g' - w\rVert_1
   + \alpha_2\lVert w'\rVert_1,

with :math:`\alpha_1 = \lambda\, v_{p,\min} / (2\pi f_{\mathrm{ref}})` and
:math:`\alpha_2 = r\,\alpha_1^2`, where ``tgv_ratio`` is :math:`r`; this keeps
sharp interfaces while allowing affine trends and avoids much of TV's
staircase bias. Native control smoothing requires a MUMPS-enabled Sauce build
and is implemented for hat, B-spline and mesh blocks; lattices are not yet
smoothed Sauce-side.

For Cartesian image kernels the same object emits the image-stacking
smoothing. Its ``illumination_normalization`` policy is ``"none"`` by default
(the raw linear adjoint before its Riesz map); ``"source"`` divides by a
fixed source-illumination diagonal and stays linear in the data;
``"cross"`` also uses adjoint illumination, is nonlinear, and is meant for
display only, never as the adjoint inside a ``J.T @ J`` operator.

Preconditioners
~~~~~~~~~~~~~~~

.. code-block:: python

   preconditioner = im.Diagonal(probe_count=4, relative_damping=1e-2, maximum_inverse_ratio=1e3)
   preconditioner = im.FromOperator(my_inverse_hessian_action)

:class:`~frequensolve.imaging.Diagonal` estimates
:math:`\operatorname{diag}(\operatorname{Re} J^{\mathsf H} W J + R^{\mathsf T} R)`
with a few Rademacher probes of the normal operator (one normal action per
probe, no Jacobian is formed), adds the penalty's exact curvature diagonal,
and applies block-wise relative damping and dynamic-range clipping before
exposing the inverse action. For L-BFGS this supplies only the initial
inverse-Hessian action of the two-loop recursion; gradients and secant pairs
remain the exact, unnormalized covectors, and the line search uses their true
directional derivative. This differs from nonlinear RTM image normalization,
which must never replace a control gradient or a linear adjoint. Bound
preconditioners are refreshed at every stage start and every
``preconditioner_refresh`` accepted iterations.

FWI: stages, continuation and checkpoints
-----------------------------------------

A :class:`~frequensolve.imaging.Stage` names the frequencies, iteration
budget and active blocks of one continuation stage, with optional loss,
misfit, penalty, smoothing, optimizer and frequency-weight overrides.

.. code-block:: python

   stages = [
       im.Stage([2, 3], iterations=8, active=["src.signature"]),      # source calibration first
       im.Stage([2, 3], iterations=10, active=["salt"]),              # interface only
       im.Stage([3, 5], iterations=15, active=["vp", "salt"]),        # joint
       im.Stage([5, 8], iterations=15, active=["vp", "rho"], loss="l2"),
   ]
   stages = im.Stage.bands([[3], [3, 5], [5, 8]], iterations=[10, 10, 15], active=["vp"])
   stages = im.Stage.alternate([source_stage, model_stage], rounds=3)   # variable projection

Frequency and Laplace continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A typical schedule starts with few real frequencies and strong Laplace
damping, then adds frequencies while reducing the imaginary shift.
``Stage.frequency_laplace_bands`` wraps the generic
:class:`~frequensolve.inversion.continuation.ContinuationSchedule`: each
named band lists its frequencies and a nonincreasing sequence of damping
values ending at zero, and one stage is emitted per damping value (an
optional ``frequency_counts`` sequence selects growing frequency prefixes
instead of the full band). With Sauce's transform convention the damping
enters as a negative imaginary part.

.. code-block:: python

   stages = im.Stage.frequency_laplace_bands(
       [
           {"name": "low", "frequencies_hz": [4.0, 6.0], "laplace_damping_hz": [1.5, 0.5, 0.0]},
           {"name": "mid", "frequencies_hz": [4.0, 6.0, 8.0, 10.0], "laplace_damping_hz": [0.5, 0.0]},
       ],
       iterations=[3, 3, 4, 4, 5],
       active=["vp"],
   )
   [stage.frequencies for stage in stages]     # ((4-1.5j, 6-1.5j), (4-0.5j, 6-0.5j), (4, 6), ...)

The problem must be declared with every complex frequency the stages use.

Running and resuming
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fwi = im.FWI(
       problem,
       stages=stages,
       optimizer=im.LBFGS(memory=10, step_limit=0.015),     # or im.NewtonCG(max_cg_iterations=20)
       penalty=im.Tikhonov(alpha=1e-2),
       preconditioner=im.Diagonal(probe_count=4),
       checkpoint="checkpoint.h5",
       history="history.json",
   )
   result = fwi.run(resume=True)
   result.state                 # complete ControlState
   result.history               # every evaluation and iteration
   result.stages                # one StageResult per stage
   result.simulation            # the simulation at the recovered state
   result.vector().to_xarray("vp").plot()
   result.save("fwi_result")    # state.h5, history.json, result.json
   result = im.FWIResult.load("fwi_result", problem)

``step_limit`` caps the RMS update of every block per iteration in optimizer
coordinates; ``scaling="curvature"`` applies a per-block diagonal change of
variables estimated with one JVP per block at every stage start. A checkpoint
(``fs-optimization-checkpoint-1`` plus a ``<stem>.state.h5`` control state) is
written after every accepted iteration; ``run(resume=True)`` skips completed
stages and continues an interrupted one with its remaining budget, and
rejects a checkpoint of a different problem, block layout, stage active set or
frequencies. The history distinguishes objective evaluations from optimizer
iterations and atomically replaces its JSON file after every record.
``fwi.solve_stage(stage, state)`` runs one stage for custom loops, and
``callback`` receives an :class:`~frequensolve.imaging.FWIIteration` after
every accepted iteration.

The repository benchmark ``benchmarks/imaging/fwi_1d_profile.py`` is a
complete example: a layered sediment column with a low-velocity notch, a
smooth start, Huber misfit with an offset taper, Tikhonov penalty, L-BFGS
with a step cap and a diagonal preconditioner over two frequency bands.

Activating blocks by hand
~~~~~~~~~~~~~~~~~~~~~~~~~

A stage is a restricted problem whose vectors live in the restricted space
while the state stays whole and is threaded through:

.. code-block:: python

   stage = problem.restrict(frequencies=[2, 3], active=["src.signature"])
   v0 = stage.vector(problem.state)                 # active slice of the state
   g = stage.gradient(v0)                           # ControlVector on the restricted space
   problem.state = problem.state.with_update(stage.space, v0 - 0.1 * g)

Every linearization sends ``controls.active = stage.space.blocks`` and the
complete ``controls.state``, so any stage may activate any subset.

Changing resolution between stages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A stage may change the control layout for itself and every later stage with
``Stage(controls=...)``: a complete :class:`~frequensolve.imaging.ControlSpace`
or a mapping ``{block key: new block spec}`` replacing those blocks.

.. code-block:: python

   stages = [
       im.Stage([3, 5], iterations=15, active=["vp"]),                  # coarse profile
       im.Stage([5, 8], iterations=15, active=["vp"],
                controls={"vp": im.DepthProfile("vp", "sediment", spacing=25 * u.m)}),
   ]
   result = im.FWI(problem, stages=stages, checkpoint="fwi.h5").run()
   result.problem               # the problem of the last stage (fine layout)
   result.state                 # complete state on that layout

At the transition :class:`~frequensolve.imaging.FWI` builds the new problem
with :meth:`ImagingProblem.with_controls <frequensolve.imaging.ImagingProblem.with_controls>`,
which shares the simulation, site, backend, observed data, misfit,
frequencies, workdir and linearization cache (keyed by the new layout) and
transfers the accepted state block by block: unchanged blocks are copied,
profile and lattice blocks are evaluated on their basis and least-squares
projected into the new basis (exact when the new basis represents the old
field, e.g. a hat profile refined by bisection), and interface, source and
reflectivity blocks must keep their layout. The next stage starts with fresh
optimizer state. Checkpoints record each stage's control layout, so
``run(resume=True)`` rebuilds the right space and rejects a mismatched
refinement. The same transfer is available by hand:

.. code-block:: python

   fine_problem = problem.with_controls({"vp": im.DepthProfile("vp", "sediment", count=81)})
   v_fine = problem.space.transfer_to(fine_problem.space, v_coarse)   # vectors only

The transfer is a least-squares projection with an LSQR tolerance, not a
physical error bound: evaluate the transferred field and reject unacceptable
projection or clipping error before relying on it. With different references
or transforms, fit the physical material in the destination parameterization
instead of interpolating raw coefficients.

LSRTM, RTM, kernels and focusing
--------------------------------

.. code-block:: python

   image = im.rtm(problem)                                      # gradient at the current state
   dm = im.LSRTM(problem, iterations=15, penalty=None).run()   # LSQR on lin.jacobian
   dm = im.LSRTM(problem, iterations=15, method="cg", damping=1e-3).run()
   kernels = im.sensitivity_kernel(problem, grid, properties=["vp"], condition="fwi")
   kernels.raw["vp"].plot.imshow(x="x", y="z", yincrease=False)
   focus = im.TimeReversalFocus(problem.restrict(active=["vp"]), softening=12.5 * u.km)
   focus.value(), focus.gradient()           # at the current state; material blocks only

:func:`~frequensolve.imaging.rtm` returns Sauce's covector, the gradient of the
misfit with respect to the active blocks (negate it for the classic
``observed - simulated`` image). :class:`~frequensolve.imaging.LSRTM` linearizes
once and solves :math:`\min_{dm}\ \tfrac12\lVert J\,dm + r\rVert_W^2 +
\tfrac12\,\mathrm{damping}\,\lVert dm\rVert^2 + P(dm)` with everything frozen
at the linearization point, either by LSQR on the real-stacked Jacobian
(``method="lsqr"``, needs the data residual and therefore one forward
solve) or by conjugate gradients on the Gauss-Newton normal equations
(``method="cg"``, uses ``lin.normal`` and ``lin.gradient`` only). It is
typically run over :class:`~frequensolve.imaging.GridParameters` or
reflectivity blocks.

:func:`~frequensolve.imaging.sensitivity_kernel` images on a Cartesian grid
(``Imaging.grid``) and returns an :class:`~frequensolve.imaging.ImageSet` with
xarray ``raw``, ``smoothed`` and ``incremental`` datasets on ``(z, x)``. With
``observed=None`` (the default) Sauce uses zero data and the images are the
pure model sensitivity kernels; ``observed=True`` images the misfit residual
instead. ``condition="fwi"`` resolves to the property-gradient condition of
the physics; other condition names are passed verbatim. Kernels run on a
simulation copy with the current state installed
(``problem.simulation_at(v)``, also :attr:`FWIResult.simulation
<frequensolve.imaging.FWIResult.simulation>`). It installs material and
interface coefficients and every changed source block it can represent:

- ``source.<i>.position`` moves the inline source point;
- ``source.<i>.signature`` :math:`q` multiplies source :math:`i`'s column of
  the source encoding (:math:`C = E\,\operatorname{diag}(q)`; an identity
  encoding is written out explicitly). :math:`q` is frequency independent, so
  a complex :math:`q` applies a frequency-independent gain :math:`|q|` and
  phase :math:`\arg q`;
- ``source.<i>.mechanism`` needs Sauce's ``/scaling/<block>`` (the physical
  strength of one stored coordinate, in ``/scaling_units/<block>``), which a
  state export from a current Sauce records; the physical components become
  the inline source's ``amplitude`` (scalar kinds), unit ``direction`` and
  ``amplitude`` (vector and dipole kinds) or a ``moment_tensor`` mechanism
  (``xx, zz, xz`` in 2D, ``xx, yy, zz, yz, xz, xy`` in 3D). Complex
  components must share one phase, which is applied like a signature.

A mechanism without scaling, ``signature_df`` (an additive per-Hz term),
reflectivity and mesh blocks raise :class:`NotImplementedError`. Focusing runs
on the authored simulation and accepts changed material blocks only.

:class:`~frequensolve.imaging.TimeReversalFocus` is a data-domain focusing
objective that needs no modeled forward wavefield. For each frequency Sauce
back-propagates the observed data and evaluates

.. math::

   J_f(m) = -\operatorname{Re}\int_\Omega
   \frac{q_f(x;m)}{\left(\lVert x - x_s\rVert^2 + \epsilon^2\right)^{p/2}}\,dx,

where :math:`q_f` is the observed-data adjoint field, :math:`x_s` the encoded
source-field center and :math:`\epsilon` the ``softening`` length; minimizing
favors a large, correctly phased focus near the target. The gradient uses a
second solve with the inverse-distance functional and contracts the focus
dual field with the saved observed-data field through the native control
VJP; both solves reuse one factorization. Time reversal conjugates the
observed coefficients, not the complex frequency, so the PML keeps
dissipating energy. The objective is polarity sensitive: observed data and
receiver encodings must share a phase convention.

.. _imaging-extension:

Extension and reflectivity
--------------------------

Model extension (FWIME) attaches an auxiliary tap space to material blocks:
one time-lag axis (``Lags``) or one spatial half-offset axis
(``HalfOffsets``) per field, with its own inner solve. The extended problem
satisfies the same protocol as the problem, so ``im.FWI`` accepts it
unchanged.

.. code-block:: python

   from frequensolve.imaging.extension import Extension, Lags

   ext = Extension(
       fields=[Lags("vp", count=5, origin=-20 * u.ms, spacing=10 * u.ms)],
       damping=0.1, lag_penalty=1.0, lag_scale=20 * u.ms,
       tolerance=1e-6, max_iterations=100, require_convergence=True,
   )
   xp = problem.restrict(active=["vp"]).extend(ext)   # material blocks only; shares state, site and cache
   xp.capabilities()                         # real frequencies, waveform terms, no source blocks
   v = xp.vector()                           # active slice of the current state
   xp.value(v), xp.gradient(v)               # reduced objective and background gradient
   xp.normal(v)                              # reduced Gauss-Newton Schur operator
   B = xp.linearize(v).jacobian              # tap-space Jacobian; B.H @ r -> ExtensionVector
   solutions = xp.solve_all(v)               # one (ExtensionVector, ExtensionSolveReport) per frequency
   taps, report = xp.restrict(frequencies=[3.0]).solve(v)   # a single-frequency inner solve
   taps.to_xarray()["vp"].sel(lag=0).plot()
   result = im.FWI(xp, stages=stages).run()  # FWIME

Each frequency solves its own inner problem (``solve`` therefore needs a
single-frequency view; ``solve_all`` returns every task's solution); reduced
covectors are summed with the stage's frequency weights like ordinary
gradients.

:class:`~frequensolve.imaging.ReflectivityParameters` is an ordinary block.
When the space contains one, the problem emits the joint
background-plus-reflectivity operator, and the ``reflectivity.<name>``
blocks take part in ``restrict``, gradients, the normal operator (with
background-reflectivity cross terms) and stages like any other block. A
:class:`~frequensolve.imaging.ReflectivityField` borrows a material map with
``basis="vp"`` or brings its own with ``control=im.DepthProfile(...)``.
Extension and reflectivity are mutually exclusive in one problem and the
combination is rejected at construction.

.. code-block:: python

   space = im.ControlSpace(
       vp=im.DepthProfile("vp", "sediment", spacing=25 * u.m, transform="log"),
       refl=im.ReflectivityParameters("vp_ip", fields=[im.ReflectivityField("ip", layer=2, axis=2, basis="vp")]),
   )
   stages = [
       im.Stage([3, 5], iterations=10, active=["vp"]),
       im.Stage([5, 8], iterations=15, active=["vp", "refl"]),
   ]

Low-level jobs
--------------

Four job classes in :mod:`frequensolve.imaging.jobs` cover every Sauce
imaging workflow. The layers above submit them; use them directly when a
workflow needs an action the problem does not expose.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Job
     - Sauce workflow
     - Use
   * - :class:`~frequensolve.imaging.FWIOperatorJob`
     - ``fwi_operator`` (``calibrate``, ``linearize``, ``jvp``, ``vjp``,
       ``normal``, ``wri``, ``solve``)
     - One action over the shared control registry, including extension,
       reflectivity and WRI.
   * - :class:`~frequensolve.imaging.ControlGradientJob`
     - ``rtm`` / ``born`` / ``focus`` with ``control_sensitivities``
     - Native control VJP, Born JVP or focusing objective; ``gram_derivative="total"``
       opts into the total DPG gradient (Gram and trial-to-test terms) for
       verification.
   * - :class:`~frequensolve.imaging.ImageKernelJob`
     - ``rtm`` / ``born`` / ``lsrtm_gradient`` with ``Imaging.grid``
     - Cartesian image kernels; ``load_images()`` returns an ``ImageSet``.
   * - :class:`~frequensolve.imaging.SmoothJob`
     - ``smooth`` postprocess
     - Smooth an existing job's per-task parts or one explicit vector.

Input files resolve relative to the project root and outputs relative to the
job result directory; task-suffixed outputs follow Sauce's
``<stem>_<task><ext>`` rule. Site artifact catalogs classify imaging outputs
by role (``image``, ``gradient``, ``objective``, ``state``,
``objective_vector``, ``extension``); the readers in
:mod:`frequensolve.imaging` (``ControlVectorFile``, ``ControlStateFile``,
``ObjectiveReport``, ``ImageSet`` ...) open them.

Sauce contract mapping
----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Sauce keys
   * - ``ControlSpace`` blocks
     - material ``parameterized`` blocks, geometry ``control``,
       ``fwi_operator.reflectivity``; ``controls.active`` is the restricted
       block order
   * - ``problem.state``
     - ``controls.state`` (``fs-control-state-1``), ``controls.state_output``,
       ``controls.manifest``
   * - ``linearize`` / ``gradient``
     - ``action=linearize``, ``covector``, ``model_gradient``
   * - ``J @ dv``, ``J.H @ r``, ``H @ dv``
     - ``jvp`` / ``vjp`` / ``normal`` with ``direction``, ``objective_vector``,
       ``covector``
   * - ``FWIOperatorJob(action="calibrate")``
     - ``action=calibrate``, ``balance`` (``normalization.scale.kind=balance_artifact``)
   * - ``FWIOperatorJob(action="wri")``
     - ``action=wri``, ``wri.{penalty, data_scale, receiver_groups, curvature}``,
       ``model_direction`` / ``model_covector``
   * - ``Extension``
     - ``fwi_operator.extension.{fields, solver}``, ``extension.direction`` /
       ``covector`` / ``manifest``, ``action=solve``, ``model_gradient``,
       ``reduced_normal``
   * - ``Misfit``
     - ``Imaging.misfit.objective_terms[]``
   * - ``Smoothing``
     - ``control_sensitivities.Smoothing`` plus the ``smooth`` postprocess
       (``Imaging.Smoothing`` for image kernels)
   * - ``sensitivity_kernel``
     - ``Imaging.grid``, ``Imaging.images``
   * - bounds, stages, penalties, optimizers, checkpoints
     - FrequenSolve only

Solver support
--------------

Native control sensitivities are supported for standard full-dimensional
acoustic and primary isotropic elastic DPG formulations. Supported acoustic
controls include compatible ``Vp``, ``Sp``, ``K``, ``beta`` and ``rho``
parameter spaces; elastic controls use the isotropic velocity/slowness,
bulk/shear, p-modulus/shear, lambda/shear and density derivative paths.
Unsupported formulations and model-dependent terms are rejected explicitly
(``problem.capabilities()`` reports them up front) instead of silently
producing incomplete gradients. Model extension requires real frequencies,
waveform comparisons and no source or geometry blocks.
