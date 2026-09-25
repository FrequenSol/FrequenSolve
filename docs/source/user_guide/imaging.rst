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
- **Regularization, preconditioners and smoothing** are configured in FrequenSolve
  (:class:`~frequensolve.imaging.Tikhonov`, :class:`~frequensolve.imaging.TV`,
  :class:`~frequensolve.imaging.Diagonal`, :class:`~frequensolve.imaging.Smoothing`).
- **Workflows** (:class:`~frequensolve.imaging.FWI`, :class:`~frequensolve.imaging.LSRTM`,
  :func:`~frequensolve.imaging.rtm`, :func:`~frequensolve.imaging.sensitivity_kernel`,
  :class:`~frequensolve.imaging.TimeReversalFocus`) run the outer loops.
- **Low-level jobs** (:class:`~frequensolve.imaging.FWIOperatorJob`,
  :class:`~frequensolve.imaging.ControlGradientJob`,
  :class:`~frequensolve.imaging.ImageKernelJob`,
  :class:`~frequensolve.imaging.SmoothJob`) are what the layers above submit.

Sauce owns the physics, native regularization energies and proximal solves;
FrequenSolve assembles the objective and owns bounds, stages, continuation,
optimizers, checkpoints and history. The
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

Every frequency task of a multi-frequency linearize measures support on its
own mesh, but the optimizer vector is shared by all tasks, so a coefficient is
supported only if every task supports it (logical AND of the per-task masks).
A coefficient that one task cannot resolve would otherwise be moved by the
other tasks' gradients alone, and the frozen set would depend on which
frequency happens to be task 1.

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

Material and interface blocks start at the coefficients FrequenSolve authors
into the simulation. Every other block (source positions, mechanisms and
signatures, reflectivity fields, mesh blocks) starts at a baseline only Sauce
knows, exported by ``controls.state_output`` together with the mechanism
``/scaling/<block>`` and ``/scaling_units/<block>``. The first linearize at the
authored point requests that export (registry discovery) and adopts it:
``problem.state`` then carries Sauce's baseline for those blocks and
FrequenSolve's values for material and interface blocks. On a space with such
blocks, asking for ``problem.state``, ``problem.vector()`` or
``problem.state_from(...)`` before any linearize runs that value-only
discovery linearize first, so an optimizer never starts (or steps) from a
placeholder mechanism. Material/interface-only spaces are known locally and
submit nothing; ``problem.dry_run()`` never submits.

.. _imaging-source-coordinates:

Source control coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~

``source.<i>.position`` is in metres, as Sauce exports it, whatever units the
simulation authors its source points in: FrequenSolve converts to and from
the points' coordinate units (a point's own units, else the simulation's
default length units, else Sauce's default ``km``) when it builds placeholder
baselines and when :meth:`~frequensolve.imaging.ImagingProblem.simulation_at`
moves a source. Signatures are dimensionless multipliers.

``source.<i>.mechanism`` needs more care. Sauce stores mechanisms in the
nondimensional units of the task that writes them, and with robust runtime
scaling those units depend on the task frequency: one physical source has
different coordinates in the 5 Hz and 6 Hz tasks of one job. State exports
record ``/scaling/<block>``, the physical strength of one stored coordinate
(:math:`s_t` in task :math:`t`), but direction and covector vectors stay in
the executing task's coordinates. FrequenSolve therefore fixes one reference
scale :math:`s_\mathrm{ref}` per mechanism block and uses
:math:`y = \text{physical} / s_\mathrm{ref}` as the optimizer coordinate:

- :math:`s_\mathrm{ref}` is task 1's ``/scaling`` of the first discovered
  registry baseline (``problem.mechanism_scaling``). It is fixed for the
  problem's lifetime: :meth:`~frequensolve.imaging.ImagingProblem.restrict`
  views and stages with other frequencies,
  :meth:`~frequensolve.imaging.ImagingProblem.with_controls` and checkpoint
  resume keep it (checkpoints record it, and a resume under another reference
  is rejected).
- Every linearize of a space with active mechanism blocks requests each
  task's ``state_output`` and reads :math:`s_t` from it.
- A direction enters task :math:`t` as :math:`y\,s_\mathrm{ref}/s_t`; task
  covectors are converted back and summed,
  :math:`g = \sum_t g_t\,s_\mathrm{ref}/s_t`, for the gradient, ``J.H`` and
  the Gauss-Newton normal
  :math:`\sum_t (s_\mathrm{ref}/s_t)^2 J_t^{\mathsf T} W J_t`.
- States carry ``/scaling`` = :math:`s_\mathrm{ref}`, so Sauce rescales staged
  ``controls.state`` values into each task's units. A state written with
  another ``/scaling`` is converted on assignment; a mechanism block without
  one is taken as reference coordinates.
- ``simulation_at`` installs the physical source :math:`y\,s_\mathrm{ref}`.

Without the conversion the per-task pieces stay mutually adjoint (dot tests
pass) while the gradient of a multi-frequency objective is wrong by the
factors :math:`s_\mathrm{ref}/s_t`; ``problem.check()`` includes a Taylor test
that detects it.

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
and ``problem.observed_vector()`` return one;
``vector.to_dataset()`` renders it as an xarray dataset by receiver group.
Operator vectors (``J @ dv`` and VJP inputs) instead use ``lin.data_space``,
whose segments are objective term IDs and whose coordinates come from the
saved state. Dense terms preserve source/component/receiver axes; sparse or
projected terms expose an objective-row axis. Multiple terms can share one
receiver group. ``lin.objective_residual()`` returns the frozen residual in
these same weighted comparison coordinates.

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
at the same point cost one job. Each action is one job carrying every
frequency of the view (one Sauce task per frequency, distributed by the site
like any multi-frequency job). Derivative actions (``jvp``, ``vjp``,
``normal``) reuse the saved state and are memoized per input vector; their
per-task inputs are written as ``<stem>_<task><ext>`` beside the stem the job
names, which Sauce resolves in each task.

Operators are :class:`scipy.sparse.linalg.LinearOperator` subclasses that
know their control and data spaces, accept typed vectors or plain arrays, and
compose with ``@``, ``+`` and scalars, so ``H + alpha * R.T @ R`` drops into
:func:`scipy.sparse.linalg.cg` or, with the ``inversion`` extra, into PyLops
through ``operator.to_pylops()``.

.. code-block:: python

   import numpy as np
   from scipy.sparse.linalg import cg

   # An explicit custom quadratic on optimizer coordinates.
   R = im.Quadratic(np.eye(lin.space.size), weight=1e-2).bind(lin.space).operator()
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
use a tight solver tolerance and the raw ``gradient``. Configured smoothing
and optimizer preconditioning do not alter this derivative. PML elements are excluded from sensitivities; the PML
uses the outward extension of the boundary model, which is held fixed.

Regularization, smoothing and preconditioning
---------------------------------------------

FrequenSolve minimizes the data objective plus the model regularization value.
Sauce owns the native control discretization, including meshed controls, B-spline
profiles and tensor hat lattices. Use ``regularization=`` on ``FWI``, ``Stage`` or
``LSRTM``:

.. code-block:: python

   regularization = im.NativeRegularization(
       im.Smoothing(kind="tv", wavelength_fraction=0.3),
       iterations=1000,
   )
   result = im.FWI(problem, stages, regularization=regularization).run()

``im.Tikhonov(alpha=...)``, ``im.TV(alpha=...)`` and ``im.TGV(alpha1=..., alpha2=...)``
are also dispatched to Sauce by these workflows. A problem's ``smoothing=``
configuration supplies the native regularizer when no explicit workflow or stage
regularization is given. An explicit regularization overrides that inherited
configuration, avoiding duplicate terms.

TV and TGV use split-Bregman shrinkage. ``epsilon`` controls the splitting
parameter; it does not round off the absolute-value norm. The native energies are

.. math::

   R_{\mathrm{Tik}}(m) = \tfrac12\alpha\int |D^p m|^2\,dx,\qquad
   R_{\mathrm{TV}}(m) = \sqrt{\alpha}\int |D^p m|\,dx,

   R_{\mathrm{TGV}}(m) = \min_w\int\alpha_1|\nabla m-w|
       +\alpha_2|\operatorname{sym}\nabla w|\,dx.

Spatial axes use km (angular axes use radians). First derivatives are supported
by all native bases; second derivatives require a B-spline basis of degree at
least two. TGV uses first derivatives. Each included material block contributes
its native integral. Non-material controls can receive separate custom terms,
such as ``Quadratic``.

Sauce evaluates the energy at every line-search trial and solves the constrained
proximal update in the optimizer's metric. Native regularization selects
proximal-gradient backtracking instead of a Newton-CG or L-BFGS step; history
records the effective optimizer. Newton/L-BFGS preconditioners do not apply to
this path; use coordinate ``scaling`` to set its diagonal metric. Bounds and
frozen coefficients enter the
proximal problem itself. Regularization uses the full model, including fixed
values; an optional ``NativeRegularization(reference=full_state)`` regularizes
the difference from a complete reference state. Native nonconvergence fails the
callback rather than reporting a valid value or accepting an unfinished update.

Wavelength-derived weights and amplitude scales are frozen at the beginning of
each stage and saved in checkpoints. With amplitude normalization scale ``a``,
the energy is ``a**2 * R((m-reference)/a)``. Coordinate scaling uses the matching
proximal metric. Native weights differ from the former Python unit-domain
coefficient regularizers; their ``alpha`` values are not interchangeable.
``Tikhonov``, ``TV`` and ``TGV`` are configuration objects for Sauce; they have
no Python value, gradient, Hessian or finite-difference implementation.
Pass them to FWI/LSRTM. Their former ``length`` and per-block ``weights`` options
are removed. Use ``NativeRegularization`` for wavelength-based configuration,
reference states and callback tolerances, and ``Quadratic`` for an explicit
custom matrix on optimizer coordinates.

``Quadratic`` and custom smooth regularizers retain the value, gradient and
``hessian_operator`` protocol. Smooth terms can be added to one native model
regularizer. The underlying ``ImagingProblem.value``, ``gradient`` and ``normal``
remain data-objective operations. ``gradient`` is its actual derivative; there
is no separate ``smoothed_gradient`` API. The explicit ``im.smooth(vector,
configuration, problem)`` operation remains available for vector processing.

LSRTM uses the same composite solver for native regularization of its image
(update), with zero fixed image coefficients and a frozen data Jacobian. Its
ordinary CG/LSQR paths remain available without native regularization or with
quadratic custom terms. TV is no longer approximated by a single Hessian frozen
at zero.

Extension solves already include their auxiliary tap regularization in the
reduced objective and Schur normal. Outer model regularization is added once;
``loss.data`` contains the reduced objective including the inner term, and
``loss.regularization`` records the background model term.

Preconditioners
~~~~~~~~~~~~~~~

.. code-block:: python

   preconditioner = im.Diagonal(probe_count=4, relative_damping=1e-2, maximum_inverse_ratio=1e3)
   preconditioner = im.FromOperator(my_inverse_curvature_action)

:class:`~frequensolve.imaging.Diagonal` estimates
:math:`\operatorname{diag}(\operatorname{Re} J^{\mathsf H} W J + R^{\mathsf T} R)`
with a few Rademacher probes of the normal operator (one normal action per
probe, no Jacobian is formed), adds the regularization curvature diagonal,
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
misfit, regularization, smoothing, optimizer and frequency-weight overrides.

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
       regularization=im.Tikhonov(alpha=1e-2),
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
smooth start, Huber misfit with an offset taper, Tikhonov regularization, L-BFGS
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

Equal coefficient counts do not imply equal bases: changing hats to B-splines
still projects the state. Vector arithmetic and state updates require matching
bases, coordinates and transforms. ``transfer_to`` transfers coefficient
fields; it is not a conversion of raw gradient covectors. If a field transfer
is ``c_new = T @ c_old``, derivatives pull back with ``T.T``.

Mixed-parameterization plotting example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The runnable :download:`mixed_parameterizations_2d.py
<../../../examples/mixed_parameterizations_2d.py>` defines initial and truth
2D acoustic models with a hat velocity profile, B-spline density profile,
tensor-hat velocity lattice, and a controlled RBF salt boundary. The salt-host
velocity is a ``BlendProperty(surface, width=..., inside=..., outside=...)``;
negative level-set values select the inside provider. A numeric width is in
model length units; a Pint length carries explicit units. Branch properties
must use the same value units.

.. code-block:: bash

   python examples/mixed_parameterizations_2d.py \
       --solver /path/to/fs2d_s --output /path/to/new-output-directory

This requires the visual dependencies and a local Sauce imaging build. It
runs forward models and joint adjoint jobs, plots physical material from
Sauce's volume VTK output and coefficients/covectors through ``state.plot``
and ``gradient.plot``, checks each block's adjoint pairing, and verifies
profile and tensor-grid refinement. Results include figures, HDF5 states/gradients and
``checks.json``. It uses no synthetic gradient substitute.

Physical material plots and coefficient plots have different meanings:
B-spline coefficients are not point samples, and material transforms and
references are applied only in the physical model. For blended material,
use solver property output; Python ``sample_uniform`` does not evaluate the
implicit geometry. The example uses volume output because some Sauce builds
do not implement material properties in the rectangular-grid VTK writer.

To view a depth profile or tensor-hat lattice as an evaluated 2D field, supply
a Cartesian display grid. This evaluates the actual basis, resolves depth
relative to layer surfaces, and masks other subdomains:

.. code-block:: python

   grid = fs.CartesianGrid(n=[181, 129], x0=[0, 0], x1=[1.2, 0.85],
                           dims=["x", "z"], units="km")
   sampled = problem.state.to_grid(grid, "shallow_vp")  # xarray.DataArray
   problem.state.plot("shallow_vp", grid=grid)
   gradient.plot("shallow_vp", grid=grid)

These fields precede the material reference and transform. Expanding a raw
gradient covector in the control basis is a visualization only, not a
physical gradient density or a gradient transfer to grid parameters. The
example labels that distinction and includes a single tensor-hat basis image.

Adapted property meshes
~~~~~~~~~~~~~~~~~~~~~~~

``PropertyMesh`` reads the actual leaf topology and hanging-node constraint
matrix from a Sauce property-space artifact. It evaluates display vertices
as ``geometry.basis @ coefficients``; the number of vertices need not equal
the number of independent controls. It never infers connectivity from point
locations or overlays control-node markers.

.. code-block:: python

   geometry = im.PropertyMesh.read("velocity.h5", material=1)
   controls = problem.state.to_mesh(geometry, "vp", units="km")
   controls.save("controls.vtu")
   problem.state.plot("vp", mesh=geometry, units="km")
   problem.gradient().plot("vp", mesh=geometry, units="km")

``material`` is the one-based material group in the artifact. Registered
basis identities are checked to reject a different mesh with the same
coefficient count. As above, plotting a covector in the primal basis is a
coefficient display, not an L2 gradient density.

Axis-aligned 2D mesh plots sample the original cell shape functions at pixel
centres before applying the colormap. This preserves bilinear quad fields
across adapted cells, avoiding the diagonal artifacts caused by rendering
two linear triangles per quad. ``resolution=600`` controls the number of
pixels along the longer axis; it does not alter the control mesh. Signed
fields use a symmetric default color range so zero is the neutral color.

To export physical properties on their own mesh, add this request to the
job's outputs (``"vp"`` identifies the named property space):

.. code-block:: python

   output = fs.VtkOutput.property_mesh(
       "vp", subdomain="rock", properties=["vp", "rho"]
   )
   state_file = problem.save_state("current-state.h5")

The complete state file includes inactive source baselines and mesh basis
identities needed by a standalone job's ``control_state``. The native VTU
writer evaluates the current physical material, including its reference,
transform and blends. ``VtkOutput.grid`` can instead sample physical
properties on a Cartesian grid with current Sauce builds.

The runnable :download:`meshed_controls_2d.py
<../../../examples/meshed_controls_2d.py>` computes a real adjoint gradient
with ``gram_derivative="total"`` (including the DPG test-map dependence),
compares Python with native property output and grid sampling, and plots the
independently adapted solution and property meshes.

These paths require a Sauce build with property-mesh output and optional
``visualization_kind``/``visualization_points_m`` artifact datasets. Older
artifacts must be regenerated for Python geometry reading. The artifact
contains geometry frozen at creation; native VTU uses the live geometry.
Both display linear cells through leaf vertices, so curved geometry and
nonlinear interior property variation are approximated between vertices.
Use solver grid sampling when physical interior values are required.

LSRTM, RTM, kernels and focusing
--------------------------------

.. code-block:: python

   image = im.rtm(problem)                                      # gradient at the current state
   dm = im.LSRTM(problem, iterations=15, regularization=None).run()   # LSQR on lin.jacobian
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
(``method="lsqr"``, uses ``lin.objective_residual()`` from the saved Sauce
state) or by conjugate gradients on the Gauss-Newton normal equations
(``method="cg"``, uses ``lin.normal`` and ``lin.gradient`` only). It is
typically run over :class:`~frequensolve.imaging.GridParameters` or
reflectivity blocks. Older saved states without an objective residual must
be regenerated before LSQR; CG can still use them. Quadratic regularization terms keep
their reference model in both solvers.

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

- ``source.<i>.position`` (metres) moves the inline source point, converted to
  the point's authored coordinate units;
- ``source.<i>.signature`` :math:`q` multiplies source :math:`i`'s column of
  the source encoding (:math:`C = E\,\operatorname{diag}(q)`; an identity
  encoding is written out explicitly). :math:`q` is frequency independent, so
  a complex :math:`q` applies a frequency-independent gain :math:`|q|` and
  phase :math:`\arg q`;
- ``source.<i>.mechanism`` needs a coordinate scale (the state's
  ``/scaling/<block>``, normally the reference :math:`s_\mathrm{ref}` of
  :ref:`imaging-source-coordinates`, in ``/scaling_units/<block>``), which a
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

One ``solve`` job carries every frequency, and each frequency task solves
its own inner problem (``solve`` therefore needs a single-frequency view;
``solve_all`` returns every task's solution); reduced covectors are summed
with the stage's frequency weights like ordinary gradients.

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
   * - bounds, stages, regularization, optimizers, checkpoints
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
