Imaging and Inversion
=====================

``frequensolve.imaging`` declares an inverse problem once and then works with
short scalar calls and SciPy-style linear operators. One
:class:`~frequensolve.imaging.ImagingProblem` binds a simulation, a control
space, observed data, a misfit, the frequencies and a site; gradients,
Jacobians, normal operators, the :term:`FWI`, :term:`LSRTM` and :term:`RTM`
workflows and the coherent focusing objectives all derive from it.

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
  :func:`~frequensolve.imaging.rtm`, :func:`~frequensolve.imaging.sensitivity_kernel`)
  run the outer loops.
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

Root patch FWI
--------------

``ImagingProblem(patches=...)`` evaluates assigned physical shots and per-shot
receiver apertures through a composite linearization:

.. code-block:: python

   patches = fs.PatchSet.around_sources(
       shots_per_patch=4, max_offset=4 * u.km, padding=2 * u.km,
   )
   problem = im.ImagingProblem(
       sim, controls=controls, observed=observed,
       frequencies=[3, 7.5], patches=patches, site=site,
   )
   prepared = problem.prepare_patches()
   prepared.plot()
   lin = problem.linearize()
   children = lin.jobs
   result = im.FWI(problem, [im.Stage([3, 7.5], 10)], optimizer=im.LBFGS()).run()

Preparation inspects geometry and acquisition without wave solves. Named meshed
material artifacts must already exist when explicitly preparing; ordinary
linearization first discovers the full-parent registry and material basis.
The first stage evaluation also resolves objective scales and reduction mass
on the full parent. Children inherit those values. Original observation keys
determine canonical data rows, and shared coefficients receive contributions
from every applicable patch. Global regularization is evaluated once.

Each stage pins its complete baseline materials for PML and captures a separate
realized wave mesh for every patch/frequency pair. Candidates retain those
meshes and PML values; a stage transition refreshes them. ``Stage(patches=...)``
overrides a problem policy. ``problem.restrict(patches=None)`` selects full-domain
execution. ``lin.jobs`` exposes every child; ``lin.job`` requires a single child.

To optimize only each patch's requested material core, pass:

.. code-block:: python

   updates = im.PatchUpdates(mode="local_serial", local_steps=1, check_every=1)
   result = im.FWI(problem, stages, optimizer=im.LBFGS(), patch_updates=updates).run()

Serial mode visits patches in prepared order and publishes accepted core changes
before the next patch. ``mode="local_parallel"`` starts all proposals from one
sweep baseline and averages overlap increments in native control coordinates,
including log controls. Coefficients outside participating cores stay fixed;
restricted regularization retains their connections to core coefficients.
Bounds and step limits also apply to the combined parallel result.

One local stage iteration is one sweep. Combined objective checks run after each
sweep by default; increases are recorded and updates retained. ``check_every=None``
disables periodic checks while retaining endpoint evaluations. History labels
local patch iterations separately from combined objective evaluations.

Patch checkpoints preserve immutable stages, frequency meshes, model epochs,
global L-BFGS history and partial local visits/proposals. Resume verifies those
artifacts and reopens the interrupted stage. Preserve the checkpoint's referenced
state files and the stage bundle when moving or archiving a run.

The current path accepts global Cartesian physical point shots, dense parent
point-receiver groups and independent waveform terms. Geometry/source-position
controls, globally encoded sources, preprocessing, receiver transforms and
receiver-diagonal probes are rejected. Local updates require material controls
and L-BFGS. PML certification and automatic construction retries remain deferred.

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
``huber``, ``student_t``), a comparison (``waveform``, ``phase_derivative``,
``spectral_derivative``),
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

For joint waveform and full-complex frequency-derivative FWI, use independent
terms. Each term's default normalization uses its own observed RMS; the
derivative is with respect to Hz at fixed Laplace damping, not a phase quotient.
Supply matching ``df`` observations and the same source-derivative policy.

.. code-block:: python

   misfit = im.Misfit.terms(
       im.ObjectiveTerm("seabed", id="u", weight=0.5),
       im.ObjectiveTerm("seabed", id="uf", weight=0.5,
                        comparison=im.Comparison.spectral_derivative()),
   )

The material ``linearize`` receiver-probe diagonal supports both terms under
the same DPG/frozen-Gram, shared dense receiver, no-preprocessing restrictions.
It includes the derivative term's triangular adjoint and mixed material/frequency
contribution; it is not the waveform-only diagonal.

Higher-order derivatives and smooth time windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For orders up to four, select the existing spectral-data workflow on the
problem. It applies the same derivative or polynomial window to every waveform
term and carries that selection through ``linearize``, ``J``, ``J.H`` and
``normal``. This is a derivative-data objective, not a derivative of an ordinary
FWI gradient.

.. code-block:: python

   problem = im.ImagingProblem(
       simulation, controls=controls, observed=im.ObservedData(observed_job),
       frequencies=[1.0-0.2j, 2.0-0.2j, 3.0-0.2j],
       misfit=im.Misfit.l2(), site=site,
       kernel_derivative={"axis": "fourier", "residual": "derivative", "order": 4},
   )
   linearization = problem.linearize(gradient=True)

   # Polynomial coefficients in physical seconds: p(t) = (t / 3 s)^4.
   windowed = problem.restrict(kernel_derivative={
       "axis": "fourier", "residual": "window",
       "window": [0, 0, 0, 0, 1 / 3**4],
   })

To export separate control gradients for orders 0–4 in one job, use a
``ControlGradientJob`` on a simulation with bound controls:

.. code-block:: python

   job = im.ControlGradientJob(
       "spectral_orders", simulation, frequencies, kind="rtm",
       observed={"seabed": observed_path}, gradient="gradient.h5",
       objective_file="objective.h5",
       kernel_derivative={"axis": "fourier", "residual": "derivative",
                          "order": 4, "source_derivative": "total",
                          "export_lower_orders": True},
   )
   site.run(job, check=True)
   gradients = [job.gradient_file(derivative_order=k) for k in range(5)]

Each order has its own objective and gradient, aggregated across frequency tasks.
The forward hierarchy and factorization are shared; independent adjoints still
require 15 solves, for 20 total instead of 30 across separate jobs. This export
counts solves per frequency and RHS batch; it does not create multiple saved
linearizations or matching Hessian diagonals.

Observations must contain the matching orders at the same complex frequencies
and source-derivative policy. Generate them with a ``forward_df`` job and
``derivative_order=4``; use its complete trace store as ``observed``. The groups
are named ``<group>_df``, ``<group>_d2f``, through ``<group>_d4f``. This route
does not use the first-order ``ObservedData(derivatives={"df": ...})`` binding.
Canonical time-domain trace stores may instead supply traces from which the
reader computes the corresponding damped time moments. Physical-shot field
data need not be pre-encoded:

.. code-block:: python

   observed = im.ObservedData(
       im.TraceStoreRef("field_traces.h5"), source_basis="source_geometry",
   )
   problem = im.ImagingProblem(
       simulation, controls=controls, observed=observed,
       frequencies=[1.0-0.2j, 2.0-0.2j, 3.0-0.2j],
       kernel_derivative={"axis": "fourier", "residual": "derivative", "order": 4},
   )

The reader applies the simulation's source encoding to each observed order
using physical source IDs, for both dense and sparse receivers. Encoding
weights are held fixed during this combination; derivatives of frequency-varying
encoding weights are not added. Trace geometry, components, units and frequency
conventions must still match the simulation.

At ``f - i*sigma``, the effective amplitude window is
``p(t) exp(-2*pi*sigma*t)``. For ``p(t)=t**n``, its peak is at
``n/(2*pi*sigma)`` seconds. Least squares squares this amplitude weighting;
a finite sampled frequency band is not an exact full-band time-domain norm.
Window coefficients are converted to complex Fourier-derivative weights
automatically. ``axis="laplace"`` instead differentiates the imaginary
frequency coordinate; it does not enable damping by itself.

For a different polynomial on each receiver/source-field pair, supply an
explicit degree and HDF5 coefficient tables instead of a common list::

   kernel_derivative={
       "axis": "fourier", "residual": "window", "order": 4,
       "window": {"seabed": "windows.h5:/coefficients"},
   }

Each real table has shape ``(order+1, global_receiver, source_field)`` in
Python/HDF5 ordering, with ascending powers of seconds and shared coefficients
across components. Paths resolve from the project. Every objective receiver group
requires a table. Use unencoded shots for physical offset-dependent windows;
an encoded shot has no single physical source offset. The solver caches only
the active source batch and local receivers, and uses the same coefficients
for observations, predictions and adjoints without extra propagation stages.
This option requires dense point receivers without averaging, global trace
preprocessing or frequency/material dependence. It does not support WRI or
probe diagonals. Keep coefficient files fixed with the saved objective state.
A shifted polynomial has an earlier lobe: a direct-arrival zero is not a causal
mute and does not necessarily suppress earlier refracted arrivals.

To specify explicit receiver/shot delays in seconds, prepare the frozen table
with the SDK helper. Delays can be picked first arrivals, geometric direct
arrivals, or eikonal receiver times (transpose the source-major eikonal array)::

   table = im.write_receiver_window("windows.h5", delay_seconds)
   kernel_derivative = {
       "axis": "fourier", "residual": "window", "order": 4,
       "window": {"seabed": table},
   }

``delay_seconds`` has shape ``(global_receiver, source_field)``. An HDF5 dataset
can be passed directly; preparation streams bounded receiver/source blocks.
The default polynomial is ``(t-delay)**4``; ``coefficients=[...]`` shifts any
common polynomial of degree at most four. The helper creates a new file and
never overwrites existing objective data. This adds no propagation stages.

Instantaneous traveltime inferred from ``u_f/u`` is not automatically a first
arrival: interfering events and weak amplitudes can make it unreliable. If it
is used to estimate delays, screen the estimates first and freeze them before
linearization. The solver does not differentiate receiver-window delays, and
no automatic instantaneous-traveltime picking is performed by this helper.

This path supports native material controls in acoustic, elastic and coupled
DPG with frozen Gram derivatives. Sources are fixed; projection, FWIME,
Galerkin and the shared receiver-probe diagonal are not supported here.
Separate independently weighted order terms still use separate problem views;
the existing joint ``u``/``u_f`` comparison remains available as above.

Centered WRI also accepts this selection through ``FWIOperatorJob``:

.. code-block:: python

   job = im.FWIOperatorJob(
       "windowed_wri", simulation, [1.0-0.2j, 2.0-0.2j, 3.0-0.2j],
       action="wri", observed=observed, covector="gradient.h5",
       objective="objective.h5", wri={"penalty": 10.0},
       kernel_derivative={"residual": "window", "window": [0, 0, 0, 0, 1 / 3**4]},
   )

This reconstructs the windowed reference prediction against equally windowed
observations while keeping the base-frequency PDE-energy metric. An identity
window gives ordinary centered WRI. Degree ``n`` costs ``n+2`` solves for the
objective/reconstruction, or ``2*(n+1)`` including its gradient, per source batch.
It requires fixed frequency/material-independent receiver rows. Material gradients
accept ``gram_derivative="total"``; the default ``"frozen"`` omits material Gram
terms. Neither option differentiates the Gram matrix in frequency. The original
formulation and spectral WRI normals/diagonals are not supported.

The default source policy is ``frozen``; supported total source spectra can be
included for gradients, but Born/GN actions currently require frozen sources.
The time-window interpretation applies to the complete signal only when the
source spectrum is constant or its derivatives are included; frozen-source
derivatives otherwise window the medium response at fixed source amplitude.
One factorization is reused: an order-``n`` objective/gradient or JVP/VJP takes
``2*(n+1)`` solves per RHS batch; a GN action takes ``3*(n+1)``.

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
these same weighted comparison coordinates: Sauce's observed-minus-simulated
comparison error, of which ``lin.jacobian`` is the derivative.
``lin.simulated()`` and ``lin.observed()`` return the modeled and observed
values of the same rows (exact for unit-weight waveform terms), and
``lin.modeled_vjp(g)`` pulls a dual ``g`` on the modeled data back to the
controls (``per_task=True`` keeps the frequency tasks apart), which is how
objectives composed in Python, such as :ref:`imaging-focusing`, differentiate.

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

For meshed materials, low-level ``FWIOperatorJob`` and ``ControlGradientJob``
default to ``sensitivity_quadrature="auto"``, which retains wavefield quadrature
for meshed materials and consistent discrete Jacobian actions. Tensor volume
pullbacks instead default to nodal sensitivities (see below). Explicit
``"material_intersections"`` evaluates volume pullbacks on wavefield/material-cell
intersections. This experimental option leaves forward
assembly, JVPs, face quadrature and wavefield DOFs unchanged, so its covector is
an approximate continuous sensitivity and need not match a finite-difference
check of the discrete objective. Compare both policies using the same material
artifact and model; check objective reduction as well as the gradient image.
Waveform and spectral receiver-probe diagonal contractions use the same selected
policy as the gradient, with wavefield quadrature as the default. Probe fields
share a bounded memory cache and spill to run-temporary files as needed;
contractions tile source columns without changing forward-solve batching. They assemble all
mixed contributions into the constrained control basis and MPI-reduce before
squaring each source response separately. With intersections, these are positive
preconditioner approximations, not exact discrete GN diagonals.
Born/JVP, normal and WRI curvature/diagonal jobs retain ``"wavefield"`` by default
and reject an explicit request for intersections. Intersection
geometry is cached with bounded per-worker storage; uncut elements keep the
ordinary rule. Coarse material meshes generally incur only the tree lookup,
while cut-element work grows with the number of intersected leaves.

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
proximal update in the optimizer's metric. In FWI, Tikhonov uses native exact
coefficient gradient and Hessian callbacks and preserves the requested L-BFGS
or Newton-CG optimizer. This requires a solver supporting the native
``gradient`` regularization operation. TV/TGV (and native LSRTM terms) select
proximal-gradient backtracking; history
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

For shared dense waveform receivers, request a receiver-encoded diagonal during
the gradient run (requires the matching updated Sauce backend)::

   lin = problem.linearize(receiver_diagonal={})
   diagonal = lin.receiver_diagonal
   damping = 0.01 * diagonal.values.max()
   update = -lin.gradient / (diagonal + damping)

The empty mapping uses ``min(16, encoded source RHS count)`` probes; pass
``{"probes": 32, "seed": 7}`` to override. Source batching is unchanged.
The backend contracts complex sensitivities into the actual material basis and
MPI-reduces before squaring. This initial path requires frozen-Gram DPG,
quadratic waveform loss, source-independent weights, material-independent
receiver channels, and no preprocessing. The accessor sums frequency-weighted
diagonals. Damping above is an explicit example, not an amplitude calibration.
For additive Vp controls in km/s, the update is in km/s; the gradient is not.

For separate spectral-derivative gradients, request matching diagonals in the
same job::

   job = im.ControlGradientJob(
       "orders", simulation, frequencies, kind="rtm", observed=observed,
       gradient="gradient.h5",
       kernel_derivative={"axis": "fourier", "residual": "derivative",
                          "order": 4, "export_lower_orders": True},
       receiver_diagonal={"probes": 16, "seed": 7},
   )
   fourth_order_diagonal = job.diagonal_file(derivative_order=4)

One shared receiver-adjoint hierarchy supplies all five diagonals, including
each order's observed-data normalization. Mixed spectral/material terms are
summed in constrained controls and MPI-reduced before squaring each source's
response separately. Five stages of 16 probe RHS are reused across all source
batches; contractions and retained test fields add cost and memory. This requires
common receiver weights up to an order scalar and frequency/material-independent
channels. Spectral windows are not supported by this shared-diagonal path.
Postprocessing aggregates each diagonal with the gradient's frequency weights,
without applying gradient smoothing or an amplitude calibration.

WRI jobs may request ``wri={"penalty": 10, "diagonal": "diagonal.h5"}``.
Use ``wri={"penalty": 10, "formulation": "original"}`` for the original
broken-test residual objective; ``"centered"`` is the default. Centered WRI
subtracts the current model's minimum PDE energy and gradient using an ordinary
forward reference solve. Both variants reconstruct the same wavefield. The
fixed-wavefield diagonal remains an uncentered preconditioner, not the Hessian
of the centered objective.

For a normal action, supply ``direction="direction.h5"``. Centered WRI defaults
to ``wri={"penalty": ..., "curvature": "metric_frozen"}``, requiring two
additional solves per source RHS. This positive approximation includes the
reference-wavefield response and agrees with centered GN at exact self-data.
Choose ``"exact_gn"`` for the full centered residual normal (four additional
solves). Both freeze Gram and receiver weights, but differentiate the reference
and reconstructed normal equations. Original WRI defaults to ``"joint_schur"``;
explicit ``"fixed_wavefield"`` and ``"joint_schur"`` remain uncentered surrogates
when used with the centered objective. The local diagonal described below is
not a diagonal of either new centered action.

``job.diagonal_file(task)`` contains the fixed-wavefield GN diagonal, with the
same normalization as its gradient. Aggregate both using
``job.wri_reduction_weights()`` before division. This local approximation
omits wavefield relaxation; it overestimates the original formulation's reduced
GN curvature, not necessarily the centered formulation's curvature.

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

For a supplied forward-energy density, :class:`~frequensolve.imaging.SourceEnergy`
integrates the control basis against the energy and squared material-transform derivative:

.. code-block:: python

   metric = im.SourceEnergy(
       energy={"vp": forward_energy},
       grid=image_grid,
       relative_damping=1e-2,
   ).bind(linearization.space)
   direction = -metric.apply(linearization.gradient)

This supports material hat/B-spline profiles and tensor-hat grids, with the
bound model's coordinate and subdomain maps. It assembles the lumped diagonal
``B.T @ (quadrature * energy * transform_derivative**2)`` without dense
matrices or extra wave solves; for partition-of-unity bases this is the row sum
of ``B.T @ diag(...) @ B``, so smooth kernels map to physical-unit updates. Non-identity controls require the
physical ``transform_derivative`` samples (for log controls, the current
physical property). Unilluminated coefficients receive zero updates; inspect
``metric.raw_diagonal`` for coverage. Energy is fixed at the supplied baseline;
rebuild the metric when that baseline changes.

Sum energy over frequencies/RHS with weights matching the covector before
binding. Default quadrature uses the image grid's numeric length units;
``quadrature_weights`` can supply a different measure. This is a source-energy
pseudo-Hessian, not an automatically calibrated objective Hessian: bare
pressure-squared does not establish velocity-update units, and differently
normalized objectives need their corresponding metric scaling. Do not
normalize the energy by its RMS when physical scaling matters.

Spatial control smoothing is separate and does not automatically apply this
metric. Passing an illumination mode to control smoothing is rejected rather
than silently ignored. Cartesian image smoothing retains its explicit
``illumination_normalization="source"`` option.
If smoothing the resulting coefficient direction, use ``input_role="primal"``;
the default dual-input Riesz map would apply an additional mass inverse.

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

Spectral selection per stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Stage(kernel_derivative=...)`` selects derivative-only data or a polynomial
window for that stage. The selection contributes to the linearization identity
and does not modify the parent problem or neighboring stages. For a waveform
problem, this schedule returns to ordinary data after a derivative stage:

.. code-block:: python

   stages = [
       im.Stage([2, 3], iterations=8),
       im.Stage([2, 3], iterations=8, kernel_derivative={
           "axis": "fourier", "residual": "derivative", "order": 1,
           "source_derivative": "total",
       }),
       im.Stage([2, 3], iterations=8),
   ]

An omitted stage selection inherits the problem's selection. If the parent is
already spectral, first use ``problem.restrict(kernel_derivative=None)`` to
obtain a waveform parent; ``Stage(kernel_derivative=None)`` means inherit.
Observations must contain the matching derivative orders. First-order total-source
spectral Born and normal actions require the supported fixed-source acoustic DPG
configuration and frozen Gram derivatives; higher-order Born actions require
``source_derivative="frozen"``. Changing this selection changes the objective,
so each stage starts fresh optimizer history.

Optimizer stopping tolerances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``LBFGS`` and ``NewtonCG`` accept separate absolute and relative tolerances
for the projected gradient norm and objective decrease:

.. code-block:: python

   optimizer = im.LBFGS(
       grad_abs_tol=0.0,
       grad_rel_tole=1e-6,
       obj_abs_tol=0.0,
       obj_rel_tol=1e-9,
   )

Each threshold is ``absolute + relative * abs(F_initial)``. ``F_initial``
is the total initial objective (including regularization) for that stage,
not the previous iteration's objective. It is retained in checkpoints on
restart. The example shows the defaults. A zero initial objective contributes
zero to the relative term; use an absolute tolerance if needed in that case.
There is no unit-dependent floor of one. Standalone minimizers accept
``initial_objective`` in their options to retain the reference across restarts.

The objective test detects small decreases, not a small objective value;
``objective_target`` remains the separate absolute target-value test.
``objective_tolerance_momentum`` optionally averages decreases, and
``objective_minimum_iterations`` delays that stopping test. Zero objective
tolerances still detect an exactly zero decrease. Gradient tests use optimizer
coordinates (the proximal-gradient mapping when using a nonsmooth regularizer).
``step_tolerance`` remains a model-space step test, not an objective-relative
test. These settings do not rescale the objective or search directions.

Existing explicit ``gradient_tolerance`` means an absolute gradient tolerance
and suppresses the default relative gradient term. Existing
``objective_tolerance`` now means decrease relative to the fixed initial
objective. Prefer the four explicit names above.

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

For meshed controls, explicitly list the mesh blocks to adapt at a stage boundary:

.. code-block:: python

   im.Stage([5, 8], iterations=15,
            controls={"vp": im.MeshParameters("vp", "sediment", frequency=8,
                                              epw=1, transform="log")},
            mesh_averaging_wavelengths=0.5)

Sauce sizes those blocks from the latest accepted physical material, averaging
**slowness** within each material. The window half-width is the requested fraction
of that material's volume-weighted harmonic-mean wavelength, estimated on the old
basis, at the largest requested control sizing frequency. A positive five-point
Gauss rule per axis samples a truncated Gaussian window. The window does not
shrink with candidate cells. EPW applies to this sizing field; it is a resolution
heuristic, not a forward-solver accuracy guarantee. Averaging suppresses sharp
local refinement but can spread moderate refinement into neighboring regions; it
does not guarantee fewer total controls. The forward reference and
regularization prior are unchanged.

The new immutable artifacts and coefficients are frozen throughout the stage,
including line searches. FrequenSolve places stage artifacts under the problem's
work directory using a hash of the accepted state and sizing policy. Naming a mesh
block again explicitly requests adaptation
even if its frequency and EPW are unchanged. Omitted blocks retain their bases.
Checkpoints retain each stage's accepted entry state to reconstruct the same
sizing input on restart; retain those accompanying ``*.stage_*.h5`` files and the
property artifacts.

Mesh transfer defaults to nodal interpolation, available explicitly as
``transfer=im.Transfer.nodal()``. To project onto **either a finer or a coarser
mesh**, use ``transfer=im.Transfer.l2()``. Add ``smooth`` to smooth the transferred
field, with a length quantity or a bare number in meters:

.. code-block:: python

   from frequensolve.units import ureg as u

   projected = problem.with_controls(
       {"vp": im.MeshParameters("vp", "sediment", frequency=3,
                                epw=2, transform="log")},
       transfer=im.Transfer.l2(smooth=100 * u.m),
   )

The same policy works in continuation stages:

.. code-block:: python

   stage = im.Stage(
       [3, 4], iterations=15, controls={"vp": target_mesh},
       transfer=im.Transfer.l2(smooth=100 * u.m),
   )

``im.Transfer.l2()`` omits smoothing; ``im.Transfer.nodal()`` selects
interpolation. The legacy ``mesh_transfer`` and ``mesh_smoothing_length``
keywords remain supported, but cannot be combined with ``transfer``.

For wavespeed-dependent smoothing, use a fraction of the **local accepted
wavelength** instead of a fixed length:

.. code-block:: python

   transfer = im.Transfer.l2(smooth_wavelengths=0.1, frequency=3 * u.Hz)

The physical smoothing length is ``0.1 * v(x) / f``: higher-velocity regions
receive more smoothing, lower-velocity regions less. ``v(x)`` is sampled from
the accepted source model and frozen for the entire transfer, before any
coarsening. For acoustics this is Vp; elastic models use the native minimum
propagating wavespeed. Omit ``frequency`` to use the largest target material-mesh
sizing frequency. ``smooth`` and ``smooth_wavelengths`` are mutually exclusive.
The variable coefficient stays inside the stiffness integral, preserving symmetry
and positive definiteness; this is not pointwise rescaling after smoothing.

Projection solves
``(M_target + length**2 K_target) c_target = B_target,source c_source``;
zero length gives ordinary L2 projection. Native quadrature integrates both
constrained bases on source/target cell intersections using the physical geometry.
The matrix-free conjugate-gradient solve checks its true residual. Affine-cell
polynomial integrands are integrated to quadrature accuracy; curved geometry
retains numerical quadrature error. There is no global dense mass matrix.
Smoothing and projection occur only when the layout is explicitly replaced.

This transfers model coefficients, **not gradient covectors**. For log controls
it smooths the log update while retaining the original reference model. Constants
are preserved; unsmoothed projection also preserves any field representable in
both spaces. General coarsening loses unresolved detail, and smoothing deliberately
changes nonconstant fields. Neither operation enforces pointwise bounds; physical
limits are applied afterward and may change conservation. The property, material,
transform and block identity must remain unchanged. Setup runs on one MPI rank;
subsequent stage solves may use MPI. New basis identities invalidate cached
operators, and each optimization stage starts with fresh optimizer state.
The initial problem's authored mesh declaration keeps its existing sizing
behavior; use ``Stage(controls=...)`` at stage zero to request averaged sizing there.

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

LSRTM, RTM and kernels
----------------------

.. code-block:: python

   image = im.rtm(problem)                                      # gradient at the current state
   dm = im.LSRTM(problem, iterations=15, regularization=None).run()   # LSQR on lin.jacobian
   dm = im.LSRTM(problem, iterations=15, method="cg", damping=1e-3).run()
   kernels = im.sensitivity_kernel(problem, grid, properties=["vp"], condition="fwi")
   kernels.raw["vp"].plot.imshow(x="x", y="z", yincrease=False)

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
reflectivity and mesh blocks raise :class:`NotImplementedError`.

.. _imaging-focusing:

Coherent focusing
-----------------

Time-reversal focusing measures how well the back-propagated observed data
refocus at each source. By reciprocity the back-propagated field sampled at
source :math:`s` is :math:`C_{f,s} = \sum_r \overline{d_{f,s}(r)}\,o_{f,s}(r)`;
summing frequencies coherently in a Gaussian lag window of width
:math:`\sigma` gives the energy
:math:`E_s = \operatorname{Re}(C_s^{\mathsf H} K C_s)` with
:math:`K_{ff'} = e^{-2\pi^2\sigma^2 (f-f')^2}`. The focusing ratio
:math:`\rho_s = E_s / (\lVert d_s\rVert^2 \lVert o_s\rVert^2)` lies in
:math:`[0, 1]` and the objective is the mean defocus :math:`J = 1 - \bar\rho`:

.. code-block:: python

   focus = problem.focus(im.Focusing(window=0.1 * u.s))
   focus.value(), focus.gradient()            # all active controls
   im.FWI(focus, stages, optimizer=im.LBFGS()).run()

``focus`` is a problem view (``restrict``, ``linearize``, ``value``,
``gradient``, ``state``, ``space``) whose misfit is a unit-weight waveform L2,
so Sauce's saved rows are the raw data; inherited ``kernel_derivative``
selections are cleared. It has no normal operator. Its result cache follows
the underlying problem's ``cache_capacity``. Point focusing reuses an
equivalent cached raw-pressure L2 linearization and then requests one ``vjp``.
Without a matching cache entry it first requests a value-only ``linearize``;
the baseline L2 gradient is not needed. Receiver-state caching is separate
from PDE field retention: the native VJP may replay the forward solve.
The focus is invariant to a common complex amplitude factor per source;
relative receiver amplitudes and phases still affect it.

:class:`~frequensolve.imaging.SourceAperture` softens the focus spatially with
a cosine-tapered node grid around every source (one physical source per
right-hand side with a real encoding weight):

.. code-block:: python

   aperture = im.SourceAperture(0.3 * u.km, 0.05 * u.km,
                                bounds=[(None, None), (0.001, None)])  # below z = 0
   linear = problem.focus(im.Focusing(0.1, aperture=aperture, strategy="linear"))
   weft = problem.focus(im.Focusing(0.1, aperture=aperture, strategy="pointwise"))

``strategy="linear"`` averages, then squares: every source becomes one
extended source :math:`D_s = \sum_k w_k d_{s,k}` in one encoded right-hand
side and the ratio above is evaluated on :math:`D` (the cost of the point
focus). ``strategy="pointwise"`` squares, then integrates, as in WEFT:
:math:`J = 1 - \operatorname{mean}_s \sum_k \bar w_k E_{s,k} /
(\lVert d_{s,k}\rVert^2 \lVert o_s\rVert^2)`. Correlations and energies
are evaluated together on ``coarse`` nodes per axis. Their normalized ratios
are interpolated to the tapered aperture grid, and the gradient applies the
transpose of that interpolation. Increasing ``coarse`` converges to the full
pointwise quadrature without mixing exact correlations with approximate norms.
Auxiliary sources use sparse named encodings and zero observed data; observations
are mapped from the original acquisition's encoded rows, including when the
input observations use the physical-shot basis.
Source coordinates retain their units and coordinate system, and physical-source
signature spectra are repeated and reordered with the aperture nodes.
Pointwise aperture focusing currently requires dense receiver sampling; sparse
surveys are rejected before submission. Linear aperture focusing retains the
original sparse receiver layout.

Aperture views require an active material-only space (use
``problem.restrict(active=["vp", "rho"])`` as appropriate). Inactive source
and other non-material controls must retain their authored values; changing
them requires a new simulation. Point focus differentiates every active control.

Both softenings bias the kinematics: an off-center node source is rewarded
for mimicking the real source, so the point focus is the unbiased objective
and the aperture is best kept small compared with the wavelength.

.. _imaging-extension:

Extension and reflectivity
--------------------------

Model extension (FWIME) attaches an auxiliary tap space to material blocks:
one time-lag axis (``Lags``) or one spatial half-offset axis
(``HalfOffsets``) per field, with its own inner solve. The extended problem
satisfies the same protocol as the problem, so ``im.FWI`` accepts it
unchanged. Real and complex physical frequencies are supported by the updated
Sauce backend. For ``f = f_real + 1j*f_imag`` and lag ``tau`` seconds, each tap
is multiplied by ``exp(2*pi*f_imag*tau) * exp(-1j*2*pi*f_real*tau)``.
Adjoints retain the conjugate of the full factor; tap coordinates remain real.
For example, declare ``frequencies=[3.0-2.0j]`` on the underlying problem;
FrequenSolve exports ``f_list: [[3.0, -2.0]]``. Large imaginary-frequency/lag
products can impair conditioning; Sauce rejects factors that are not
representable in its working precision (single precision in production builds).

.. code-block:: python

   from frequensolve.imaging.extension import Extension, Lags

   ext = Extension(
       fields=[Lags("vp", count=5, origin=-20 * u.ms, spacing=10 * u.ms)],
       damping=0.1, lag_penalty=1.0, lag_scale=20 * u.ms,
       tolerance=1e-6, max_iterations=100, require_convergence=True,
   )
   xp = problem.restrict(active=["vp"]).extend(ext)   # material blocks only; shares state, site and cache
   xp.capabilities()                         # real or complex frequencies; waveform terms, no source blocks
   v = xp.vector()                           # active slice of the current state
   xp.value(v), xp.gradient(v)               # reduced objective and background gradient
   # xp.normal(v) is currently available only for independent/single-frequency fits.
   B = xp.linearize(v).jacobian              # tap-space Jacobian; B.H @ r -> ExtensionVector
   solutions = xp.solve_all(v)               # one (ExtensionVector, ExtensionSolveReport) per frequency
   taps, report = xp.solve(v)                # one shared fit over the whole band
   taps.to_xarray()["vp"].sel(lag=0).plot()
   result = im.FWI(xp, stages=stages).run()  # FWIME

``Extension`` defaults to ``frequency_coupling="shared"``. One tap vector is
fit to all frequencies: weighted data normals and right-hand sides are summed
before the common solve, with the regularizer added once. This is not an average
of separately fitted taps. The reduced background gradient uses those same
shared taps and frequency weights. ``solve_all`` returns per-task copies with
their individual saved-state identities; ``solve`` returns the common taps and
aggregate report.

For progress images during an L2 fit, set
``gradient_checkpoints="iterations/gradient"`` on ``Extension``. Set
``gradient_checkpoint_interval=10`` to compute and export only every tenth
accepted CG iterate (default 1); skipped iterations incur no checkpoint solves.
Selected iterations export ``gradient_cg_<iteration>_<task>.h5`` and a matching JSON
completion record. These are fixed-tap background covectors, **not** stationary
reduced gradients until the inner fit converges. They add three propagation
solves per source batch and frequency per checkpoint. They reuse the existing
factors and do not restart CG. Aggregate task covectors with the same frequency
weights as the fit; the JSON data objectives are unweighted per-frequency values.

The first shared implementation supports waveform L2 values and gradients on
``LocalSite``. It launches one persistent MPI group per frequency, keeping factors
and incident checkpoints alive during the inner iteration. ``procs_per_job`` is
the spatial rank count **per frequency**; the site's total thread budget must be
at least the frequency count times that rank count. Memory is the sum of the
simultaneously resident frequency workers (including their checkpoint budgets).
All frequencies must use the same physical control basis, axis and field scales;
freeze meshed property artifacts and, for profile controls, reference units.
The solver checks these identities and maps differently partitioned coordinates
by global control ID. A failed worker fails the band; partial frequency retry or
reuse is not valid for the shared iteration.

Shared robust-loss fits, shared reduced-Schur actions and automatic remote-site
launching are not yet supported. Set ``frequency_coupling="independent"`` only
when separate per-frequency extensions are intentionally wanted; that mode
retains the existing robust and reduced-Schur workflows.

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
     - ``rtm`` / ``born`` with ``control_sensitivities``
     - Native control VJP or Born JVP; ``gram_derivative="total"``
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
producing incomplete gradients. Model extension supports real or complex
frequencies and requires waveform comparisons and no active source or geometry
blocks; relaxed assembly is accepted as an approximation.

WRI observed-data calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WRI defaults to ``objective_normalization="observed_energy"``: a common
observed-energy divisor scales the objective, gradient and curvature without
changing the reconstructed wavefield or PDE/data penalty balance. This is not
illumination compensation. ``job.objective_value()`` and the backend's covector
reduction use a ratio of weighted sums over frequencies, not a sum of ratios.

For a fixed pre-inversion calibration, run a ``FWIOperatorJob(action="wri",
wri={"penalty": penalty, "normalization_only": True}, objective="scale.h5")``
with the same observations, preprocessing and frequency weights as inversion.
It performs no wavefield assembly or solves and needs no covector output.
Read ``divisor = calibration_job.wri_normalization_divisor()`` and set
``wri={"penalty": penalty, "objective_normalization": divisor}`` on subsequent
inversion jobs. Retain this divisor throughout model updates and line searches.
``"none"`` selects the unnormalized objective; a zero observed-energy automatic
calibration is rejected. Recalibrate when the observation selection, weights,
penalty or data scales change, not when the model changes.

``FWIOperatorJob(action="wri")`` supports full-dimensional coupled acoustic–elastic
DPG with classic elasticity. All WRI curvature modes
require frozen Gram weights and unwindowed material controls; relaxed assembly
is accepted as an approximation. The PDE objective includes the interface continuity penalties.
``gram_derivative="total"`` applies to objective gradients in both the fluid and
solid domains; it does not enable total-Gram curvature. WRI still excludes 2.5-D.

These capabilities require a Sauce build containing the coupled-WRI and
complex-frequency extension updates. The unchanged ``fs-job-1`` schema version
alone does not establish backend support; older executables may reject these
jobs. The FrequenSolve contract fixture records the targeted compatibility
updates separately from its historical Sauce pin.

Spectral field storage
~~~~~~~~~~~~~~~~~~~~~~

Spectral FWI defaults to ``"field_storage": "auto"`` in ``kernel_derivative``.
After building and solving with the solver, Sauce estimates the entire retained
field hierarchy from the DOF layout and RHS batch size. It switches to disk if
that payload exceeds 80% of currently available memory, conservatively shared
among MPI ranks on the host. Set ``field_memory_fraction`` in [0,1] to adjust
headroom, or use explicit ``"memory"`` / ``"disk"`` overrides. Unknown memory
availability selects disk. This is a budget check, not a memory reservation.
Disk mode keeps
recurrence snapshots and forward trial fields in read-only, demand-paged local
scratch files without changing precision or adding PDE solves. Use local SSD
scratch with enough free capacity. The active solve and factors remain in memory,
and operating-system paging does not impose a strict RSS limit. Scratch fields
are automatically discarded and are separate from persistent checkpoints.

Tensor-point sensitivities
~~~~~~~~~~~~~~~~~~~~~~~~~~

The default ``sensitivity_quadrature="auto"`` samples tensor-hat volume
sensitivities at control nodes without cell-volume weights for RTM and FWI
pullbacks. The gridded-image mapper supports curved elements and averages shared
samples. These are approximate nodal sensitivities, not integrated coefficient
covectors, expressed per physical km^D rather than solver-coordinate volume.
Tensor controls must be full-dimensional Cartesian grids sharing one
layout per material layer. Depth-only layers keep native quadrature; mixed
tensor/depth controls within a layer are rejected pending strip integration.

Use explicit ``"wavefield"`` for discrete coefficient derivatives and transpose
tests. Auto keeps Born/JVP, normal and WRI curvature actions discrete. Forward
assembly and face terms are unchanged. Smoothing generated tensor sensitivities
uses nodal input; explicit input vectors retain their configured input role.

Material-aware volume assembly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For meshed-material seismic models, opt in with
``fs.Discretization(material_quadrature="material_intersections")`` and
``Solver.forms_backend="cpu"`` (the default). This uses the common material/wave
cell rule for volume assembly and coefficient derivatives, including curved
geometry. Standard assembly remains the default. Faces retain their native rule;
tensor/depth partitions are not implemented. Leave sensitivity quadrature at
``auto`` or ``wavefield`` to inherit the assembly rule. Total DPG Gram/test-map
derivatives still require the separately supported ``gram_derivative="total"``.


Native material mass metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``im.NativeMass()`` supplies ``gamma*M^{-1}`` as the inverse metric for L-BFGS
or Newton-CG, using the consistent mass matrix on the native material geometry
and constrained basis. The objective still returns raw coefficient derivatives;
line searches and secant products retain their coefficient-space pairings.
By default, ``gamma`` is calibrated once per stage from a directional
Gauss-Newton-plus-regularization curvature. ``curvature_scale=False`` uses one.
With ``approximation="diagonal"``, assemble ``D=diag(M)`` over native material
elements, including constrained-basis contributions. Cache this positive array
by native basis identity and coefficient layout across frequency stages; apply
``gamma*g/D`` without native calls or iterative solves. A changed basis/geometry
identity triggers reassembly. This Jacobi approximation is not row-sum lumping.
The bound object's ``riesz`` method returns the unscaled L2 gradient in consistent
mode and its diagonal approximation otherwise; ``mass`` always applies ``M``. Completely active material blocks are
required; partial support masks and non-material controls are rejected.
A solver supporting native ``mass`` and ``mass_inverse`` callbacks is required
for consistent mode; diagonal mode requires ``mass_diagonal``.
