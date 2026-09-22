Imaging and Inversion
=====================

``frequensolve.imaging`` declares an inverse problem once and then works with
short scalar calls and SciPy-style linear operators. One
:class:`~frequensolve.imaging.ImagingProblem` binds a simulation, a control
space, observed data, a misfit, the frequencies and a site; gradients,
Jacobians, normal operators and the FWI, LSRTM, RTM and focusing workflows
all derive from it.

.. code-block:: python

   from frequensolve import imaging as im

Overview
--------

The layers, lowest first:

- **Control spaces** (:class:`~frequensolve.imaging.ControlSpace`) turn material
  profiles, lattices, interfaces and source parameters into one real vector
  with transforms, bounds and coordinates.
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

Sauce owns the physics and gradient smoothing; FrequenSolve owns bounds,
stages, continuation, penalties, optimizers, checkpoints and history.

Control spaces
--------------

A block names one Sauce control registry block: where it lives (a subdomain
or surface), its basis (exactly one of ``spacing``, ``count``, ``nodes``) and,
optionally, a transform and physical value limits. Extent is never authored;
it is the subdomain's span.

.. code-block:: python

   space = im.ControlSpace(
       vp=im.DepthProfile("vp", "sediment", spacing=25.0, transform="log", limits=(1450, 3500)),
       rho=im.DepthProfile.bspline("rho", "sediment", count=20, transform="log"),
       salt=im.InterfaceParameters("salt_top", maximum_displacement=150.0),
       src=im.SourceParameters(position=True, signature=True),
   )
   space.blocks                 # ('model.vp', 'model.rho', 'model.salt_top', 'source.1.position', ...)
   space.restrict(["vp", "src.signature"])
   space.zeros(), space.random(seed=1), space.pack({"vp": ..., "rho": ...})
   space.bounds                 # optimizer-coordinate bounds derived from limits

:class:`~frequensolve.imaging.GridParameters` places a tensor-hat lattice over a
subdomain (or the whole model), :class:`~frequensolve.imaging.MeshParameters`
uses Sauce's mesh-native controls, and a single block can be passed directly as
``controls=``. Lengths accept plain numbers in the model's units or Pint
quantities.

Two vector types travel through the API. A
:class:`~frequensolve.imaging.ControlState` is the complete baseline over every
block (``problem.state``); a :class:`~frequensolve.imaging.ControlVector` lives
on the active subspace of a stage and supports arithmetic, ``v["vp"]``,
``v.to_xarray()``, ``v.plot()`` and ``save``/``load``. Coefficients without
basis support inside their subdomain are frozen by Sauce's support masks and
drop out of the optimizer vector automatically.

Observed data and misfit
------------------------

Observed data comes from a finished forward job, a trace store, or a mapping
of receiver groups to files; frequencies are inferred when the source declares
them.

.. code-block:: python

   observed = im.ObservedData(observed_job)
   observed = im.ObservedData({"seabed": "obs.h5", "das": "das.h5"}, frequencies=[3.0, 5.0])

The misfit maps one to one onto Sauce's objective terms: a loss (``l2``,
``huber``, ``student_t``), a comparison (``waveform``, ``phase_derivative``),
a normalization (``observed_rms`` by default) and preprocessing hooks.

.. code-block:: python

   misfit = im.Misfit.huber(
       delta=1.345,
       preprocess=[im.Preprocess.offset_taper(d0=100.0, d1=250.0)],
   )
   misfit = im.Misfit(loss="l2", comparison="phase_derivative", normalization="observed_rms")
   misfit = im.Misfit.terms(
       im.ObjectiveTerm("hydrophone", loss="huber", weight=1.0),
       im.ObjectiveTerm("das", loss="l2", comparison="phase_derivative", weight=0.3),
   )

Linearize and operators
-----------------------

.. code-block:: python

   problem = im.ImagingProblem(
       simulation,
       controls=space,
       observed=observed,
       misfit=misfit,
       frequencies=[3.0, 5.0, 8.0],
       site=site,
       workdir="fwi",
       name="fwi",
   )

   problem.value(v)             # scalar misfit at v (a ControlVector or array)
   problem.gradient(v)          # ControlVector
   lin = problem.linearize(v)   # one saved Sauce state per frequency
   lin.value, lin.report, lin.gradient
   J = lin.jacobian             # J @ dv -> DataVector, J.H @ r -> ControlVector
   H = lin.normal               # frozen Gauss-Newton J^H W J, self-adjoint

Every linearization maps to a saved Sauce state and is cached by fingerprint,
so ``problem.value(v)``, ``problem.gradient(v)`` and ``problem.jacobian(v)``
at the same point cost one job family. Operators are
:class:`scipy.sparse.linalg.LinearOperator` subclasses and compose with ``@``,
``+`` and scalars, so ``H + alpha * R.T @ R`` drops into ``scipy.sparse.linalg.cg``
or PyLops. ``problem.check(v)`` runs the adjoint, normal-consistency and
Taylor tests; ``problem.restrict(frequencies=..., active=...)`` returns a stage
view that shares the state.

Regularization and preconditioning
----------------------------------

Penalties are evaluated in Python on the block coordinates and add to the
objective; they expose ``value``, ``gradient`` and a Hessian operator for
Newton-CG.

.. code-block:: python

   penalty = im.Tikhonov(alpha=1e-2, order=1)          # finite differences per block
   penalty = im.TV(alpha=1e-3) + 0.5 * im.Tikhonov(alpha=1e-2, order=2)

Smoothing is Sauce's native gradient smoothing: attach it to the problem
(``smoothing=im.Smoothing(kind="tgv", wavelength_fraction=0.3)``) so every
gradient is smoothed, or call ``im.smooth(vector, smoothing, problem)``.

Preconditioners change the linear solve only.
:class:`~frequensolve.imaging.Diagonal` estimates the Gauss-Newton diagonal
with a few randomized normal actions and applies a damped inverse per block;
:class:`~frequensolve.imaging.FromOperator` wraps any operator.

.. code-block:: python

   preconditioner = im.Diagonal(probe_count=4, relative_damping=1e-2, maximum_inverse_ratio=1e3)

FWI stages and checkpoints
--------------------------

A :class:`~frequensolve.imaging.Stage` names the frequencies, iteration
budget and active blocks of one continuation stage, with optional loss,
penalty, smoothing and optimizer overrides. ``Stage.bands`` builds one stage
per frequency band.

.. code-block:: python

   stages = [
       im.Stage([2, 3], iterations=8, active=["src.signature"]),   # source calibration
       im.Stage([2, 3], iterations=10, active=["salt"]),           # interface only
       im.Stage([3, 5], iterations=15, active=["vp", "salt"]),     # joint
   ]
   stages = im.Stage.bands([[3], [3, 5], [5, 8]], iterations=[10, 10, 15], active=["vp"])

   fwi = im.FWI(
       problem,
       stages=stages,
       optimizer=im.LBFGS(memory=10, step_limit=0.015),      # or im.NewtonCG(max_cg_iterations=20)
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

``step_limit`` caps the RMS update of every block per iteration in optimizer
coordinates. A checkpoint is written after every accepted iteration;
``run(resume=True)`` skips completed stages and continues an interrupted one
with its remaining budget. ``fwi.solve_stage(stage, state)`` runs one stage for
custom loops. The repository benchmark ``benchmarks/imaging/fwi_1d_profile.py``
is a complete example: a layered sediment column with a low-velocity notch,
a smooth start, Huber misfit with offset taper, Tikhonov penalty, L-BFGS with
a step cap and diagonal preconditioner over two frequency bands.

LSRTM, RTM, kernels and focusing
--------------------------------

.. code-block:: python

   image = im.rtm(problem)                                # gradient at the current state
   dm = im.LSRTM(problem, iterations=15, penalty=None).run()   # LSQR on lin.jacobian
   kernels = im.sensitivity_kernel(problem, grid, properties=["vp"], condition="fwi")
   kernels.raw["vp"].plot()
   focus = im.TimeReversalFocus(problem, softening=12.5)
   focus.value(v), focus.gradient(v)

:func:`~frequensolve.imaging.rtm` returns Sauce's covector, the gradient of the
misfit with respect to the active blocks; the classic RTM image in the
``observed - simulated`` convention is its negative. LSRTM linearizes once and
solves the least-squares problem on the Jacobian, typically over
:class:`~frequensolve.imaging.GridParameters`. Sensitivity kernels image on a
Cartesian grid and return an :class:`~frequensolve.imaging.ImageSet`.

Extension and reflectivity (planned)
------------------------------------

Model extension (FWIME lags or half-offsets with a reduced inner solve) and
reflectivity blocks share the same problem protocol: ``problem.extend(ext)``
returns an extended problem with ``value``, ``gradient`` and a reduced normal
operator, and ``im.FWI`` accepts it unchanged;
:class:`~frequensolve.imaging.ReflectivityParameters` is an ordinary block that
joins stages like any other. Both are scheduled for the next phase of the
imaging API.
