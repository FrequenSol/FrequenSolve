Direct time-domain simulation
=============================

``TransientJob`` advances acoustic pressure, elastic displacement, or coupled
acoustic–elastic waves in physical time. ``TimeDomainJob`` retains its harmonic
sweep and reconstruction API. A transient job defaults to one solver task containing all encoded sources.
For acoustic angular sweeps, ``n_modes`` creates one task per mode. Both use
native time samples with no frequency packing step.

For an existing Cartesian 2D/3D or axisymmetric 2D simulation ``sim``,
``TransientJob`` fills in unset Galerkin and lossless attenuation settings on the
supplied simulation. Explicit settings are preserved and incompatible choices
are rejected. Set a physical bandwidth using ``time.f_max`` or an existing
``mesh.adapt.f_adapt``:

.. code-block:: python

   import frequensolve as fs

   job = fs.TransientJob(
       "pulse",
       sim,
       time=fs.TimeIntegration(start=0, end=1, dt=fs.Q_(1, "ms"), f_max=30),
       source=fs.TimeFunction.ricker(frequency=10, delay=0.3),
       output_schedule=fs.TimeOutput(times=[0.3, 0.5, 0.75]),
       checkpoint=fs.CheckpointSchedule(stride=100),
   )
   job += fs.WavefieldOutput(
       name="pressure", field="pressure",
       grid=fs.CartesianGrid(x0=[0, 0], x1=[1, 0.5], n=[81, 41],
                             dims=["x", "z"], units="km"),
   )
   job.save()
   result = site.run(job, check=True)  # a configured LocalSite, SlurmSite or AWSSite

If mesh adaptivity is unset, ``time.f_max`` creates it with
``elems_per_wave=2`` and ``order=2``. Existing mesh settings are retained; only
an unset ``f_adapt`` is filled from ``f_max``. A source peak frequency is never
used to guess bandwidth. Because setup updates ``sim``, use a separate simulation
if harmonic and transient jobs need different discretization or attenuation.
For attenuation, explicitly configure ``model.attenuation_model="gsls"`` and its
frequency band; this SDK supports GSLS for Cartesian transients, while
axisymmetric transients require lossless materials.

Waiting for a transient job on a local or SLURM site displays completed/total
time steps and an estimated finish time once a measured step rate is available.
Sauce publishes a small ``progress.json`` at roughly 5% increments, independent
of field/trace output cadence; intermediate steps perform no progress-file I/O.
Restart estimates use only advances performed after loading the checkpoint and
exclude setup time. Estimates assume similar cost per step and do not include
final output/teardown work. Angular-mode jobs sum step counts across tasks and
show an estimate once every unfinished task has a measured rate. Older solver
builds show "waiting for solver progress" until completion.

Use ``field="displacement"`` for elastic wavefields. A scalar receiver
component measuring displacement also needs a direction, for example
``ReceiverComponent(name="ux", field="displacement", direction=[1, 0])``.
Receiver geometry and source
strengths stay in ``sim.acquisition``; its spectral source signatures are not
used by this workflow. All encoded sources must fit in one acquisition batch.
``f_adapt`` is the highest frequency the mesh should resolve and serializes as
``Mesh/adapt/f_adapt``. Choose it for the source bandwidth; ``f0`` remains the
nondimensionalization reference. The timestep must divide the interval exactly;
stability of implicit integration does not guarantee temporal accuracy.
Average-acceleration Newmark has no explicit element-crossing CFL bound for
the linear wave system. Choose the timestep by halving it and comparing
arrival times, phase and amplitudes over the full interval of interest.
The source bandwidth, dispersive guided waves and accumulated phase error
can require substantially more samples per cycle than a stability argument
would suggest. Output cadence is independent of the integration timestep.

Time inputs accept plain seconds or Pint quantities, including ``start``,
``end``, ``dt``, source ``delay``/``width``, and output ``times``/``interval``.
Frequency inputs accept plain hertz or quantities such as ``fs.Q_(0.03, "kHz")``.
Stored configurations use seconds and hertz. To refine a frozen configuration, use
``refined = job.time.with_dt(job.time.dt / 2)``. More general edits use
``job.time.replace(end=2)`` or ``job.time.replace(window_error=0.01)``. These helpers
reconcile resolved selectors: an explicit ``dt`` clears error targets, and an
error target recomputes ``dt``. For phase-fitted Gauss, changing ``dt`` preserves
the physical fitting frequency. Use these helpers instead of ``dataclasses.replace``.

Integration scheme and temporal accuracy
----------------------------------------

Select an order and optionally a scheme. Currently ``order=2`` defaults to
``scheme="newmark"`` (average acceleration), and ``order=4`` defaults to
``scheme="gauss"`` (two-stage Gauss collocation). A scheme without an order
selects its supported order. Unsupported combinations are rejected.
Existing ``TimeIntegration(0, 1, 0.001)`` calls retain their Newmark behavior;
explicit ``method=fs.Newmark(...)`` and ``method=fs.Gauss()`` are also supported.
Choose either ``method`` or ``scheme``/``order``.

.. code-block:: python

   job = fs.TransientJob(
       "marine-gauss4", sim,
       time=fs.TimeIntegration(
           start=0, end=5, scheme="gauss", order=4, dt=0.025,
       ),
       source=fs.TimeFunction.ricker(1.5, delay=1.0),
       output_schedule=fs.TimeOutput(stride=10),
   )

Gauss uses one complex stage solve per timestep and retains a real accepted
state. Sauce selects FGMRES for this scheme; do not explicitly request PCG.
PML and GSLS can be used within their existing physics restrictions. Reduced
fractures still require average-acceleration Newmark.

To fit the temporal phase at one frequency, set ``f_0`` in Hz. Zero selects
ordinary Gauss-4. For example, to fit a 40 Hz component with a 4 ms step:

.. code-block:: python

   time = fs.TimeIntegration(start=0, end=2, order=4, dt=0.004, f_0=40)
   print(time.b)
   # Equivalent: method=fs.Gauss(f_0=fs.Q_(40, "Hz"))
   refined = time.with_dt(0.002)  # Retains the 40 Hz fitting frequency.

The internal angle is ``theta_0=2*pi*f_0*dt`` (dimensionless radians per step).
The coefficient is ``b=(1-theta_0/(2*tan(theta_0/2)))/theta_0**2``, evaluated
with its stable small-angle limit ``b=1/12``. Require ``f_0*dt < 0.5``.
Sauce applies the corresponding shift, source samples and acceptance weights
through the same single-complex-solve path, including PML and attenuation
memories. State storage and kernel launch counts do not change.

Nonzero tuning requires explicit ``dt``; the automatic error selectors below
apply to standard Gauss-4. Retaining a fixed ``f_0`` when refining dt preserves
fourth-order convergence. Fitting one frequency removes its undamped temporal
phase error, but does not bound broadband wavefield error; other frequencies
can have phase lead. The reported error estimates are signed.

Instead of ``dt``, specify one of two harmonic phase-error targets:

.. code-block:: python

   # At most 0.1% phase lag per period at the highest resolved frequency.
   time = fs.TimeIntegration(
       start=0, end=5, order=4, f_max=4, period_error=0.001,
   )

   # At most three degrees of accumulated phase lag over the whole window.
   time = fs.TimeIntegration(
       start=0, end=5, scheme="gauss", f_max=4,
       window_error=fs.Q_(3, "degree"),
   )
   # Equivalent angle units are accepted; a plain number means radians.
   time = fs.TimeIntegration(
       start=0, end=5, order=4, f_max=4, window_error=0.05,
   )
   print(time.scheme, time.order, time.dt, time.steps)
   print(time.estimated_period_error, time.estimated_window_error)

``period_error`` is dimensionless: the numerical phase lag divided by the
physical phase advance over one period, not a percentage value. ``window_error``
is an accumulated phase lag in radians, or a scalar Pint quantity in radians
or degrees (including quantities from another Pint registry). Internally it
is normalized to radians. For a duration ``T`` and frequency ``f_max``, the
window estimate equals the period estimate multiplied by ``2*pi*f_max*T``.

Specify exactly one of ``dt``, ``period_error`` or ``window_error``. The error
selectors require an explicit positive ``f_max`` in Hz; no frequency is inferred
from the source peak or from ``f0``. It can match the mesh's ``f_adapt`` bandwidth.
Selection uses the exact harmonic amplification phase for the chosen scheme,
rounds the step count upward, and sets ``dt=(end-start)/steps``. It also enforces
at least two samples per highest-frequency period. This is fixed-step selection
performed once before submission, not runtime adaptivity. Custom Newmark
coefficients require explicit ``dt``.

For a five-second window, ``f_max=4`` and ``window_error=0.05`` give 171 Gauss
steps (about 29.24 ms) or 1,818 Newmark steps (about 2.75 ms). These are undamped
harmonic temporal-dispersion estimates, not bounds on total wavefield error:
spatial discretization, iterative solves, source sampling, attenuation and PML
introduce other errors. ``estimated_period_error`` and ``estimated_window_error``
are available when ``f_max`` is supplied; they return ``None`` for custom Newmark
coefficients or timesteps above the two-samples-per-period limit.

Saving a job retains the selection target in frontend ``time_selection``
metadata; Sauce receives only the resolved method and timestep in ``Time``.
Output times must still align with the chosen grid; use a step-based ``stride``
when allowing automatic timestep selection. Restart uses the same resolved grid
and integration scheme as the original run.

Solver tolerance
----------------

Set the linear-solve tolerance on ``sim.solver.tolerance``. Leaving it unset
uses the backend default: ``1e-3`` for both Newmark and Gauss transient solves,
and ``1e-4`` for frequency-domain solves. Explicit values are preserved.

Snapshot schedules
------------------

Choose one of three schedules, in physical seconds or completed steps:

.. code-block:: python

   fs.TimeOutput(times=[0.3, 0.5, 0.75])  # explicit physical times
   fs.TimeOutput(interval=0.025)         # every 25 ms from start
   fs.TimeOutput(stride=25)              # every 25 integration steps

``output_schedule`` selects wavefield snapshots and ParaView output. Receiver
samples inherit it unless ``trace_schedule`` supplies a separate ``TimeOutput``.
For example, use ``output_schedule=fs.TimeOutput(interval=0.002)`` and
``trace_schedule=fs.TimeOutput(interval=0.0002)`` for 2 ms snapshots and 0.2 ms
gathers. Both schedules use the same integration grid. Requests
must align with the integration grid; off-grid times are rejected. The default
is every step. The final sample is always included. ``write_initial=False``
omits step zero unless explicitly requested in ``times``. Cadences are mutually
exclusive. The constructor also accepts ``output_stride=25`` and
``write_initial=False`` as conveniences when no ``output_schedule`` is supplied.

ParaView time series
--------------------

Add visualization fields with the regular output API:

.. code-block:: python

   job += fs.VtkOutput(name="pressure", fields=["pressure"], format="vtu")
   # To store mesh coordinates and connectivity only once:
   job += fs.VtkOutput(name="compact", fields=["pressure"], format="xdmf")

Open ``ParaView/pressure.pvd`` or ``ParaView/compact.xmf`` after the run.
VTU snapshots embed standard ``TimeValue`` metadata in physical seconds; the
collection preserves irregular times and negative starts. Appended binary
(default) and ASCII VTU are supported. Restart collections contain only samples
emitted by the resumed run.

Transient visualization writes explicitly requested fields, registered material
properties, and mesh information, with real parts by default. Material properties
and mesh diagnostics such as ``Order`` and ``Domain`` must be requested explicitly.
Imaginary parts are rejected; ``parts=["abs"]`` requests magnitude explicitly. Encoded sources
produce separate field arrays, optionally filtered with ``sources=[...]``.
The spatial ``output_schedule`` controls visualization output.

Each independent VTU snapshot contains its own mesh. XDMF stores coordinates
and connectivity once in ``compact.h5`` and references that geometry from each
time sample; subsequent HDF5 groups contain fields only. XML snapshots retain
small time and dataset references. Volume and surface targets are supported,
subject to existing XDMF restrictions (single cell type for quadratic elements,
no 2D edge surfaces). Transient VTR and post-processing are unsupported.

Velocity and Helmholtz decomposition
------------------------------------

Request ``fields=["velocity"]`` for native particle velocity without a
projection. Only the decomposed ``p_velocity`` and ``s_velocity`` fields
automatically enable Helmholtz projection.

For acoustic, elastic, or coupled transients, including axisymmetry, request
native, P, and S velocity fields explicitly:

.. code-block:: python

   job += fs.VtkOutput(
       name="elastic_velocity", fields=["velocity", "p_velocity", "s_velocity"],
   )

The projection runs at spatial output times and at trace times that need projected
fields. Ordinary pressure, displacement and strain traces do not trigger extra
projection solves.
In solids, ``velocity`` is the accepted Newmark rate. In fluids it is
``-grad(integral(p dt))/rho``, starting from zero fluid velocity. The pressure
integral advances at every accepted step and is checkpointed, regardless of output
cadence. Older checkpoints without that integral cannot recover fluid velocity.
The default S component is the complement ``velocity - p_velocity``. Use
``field="displacement"`` in standalone elastic jobs with ``displacement``, ``p_displacement`` and
``s_displacement`` to decompose displacement instead. Decomposed outputs must use the selected field family. Ordinary pressure,
displacement, strain, and material-property outputs remain available alongside
the projection and sample the physical solution. To include native velocity,
select velocity decomposition. Scalar vector receiver components need
a measurement direction. Explicit advanced configurations use ``parts=["p", "s"]``
to retain the native field; the legacy single ``part`` replacement mode is
unsupported by ``TransientJob``. For example, displacement decomposition uses
``sim.extra["helmholtz_projection"] = {"field": "displacement", "parts": ["p", "s"]}``.
The automatic velocity default sets only the field family; it does not enable
projection until decomposed outputs are requested.

A complete runnable example is provided in
``tutorials/01_modeling_basics/elastic_transient_helmholtz.py``.
The fluid-filled, steel-cased borehole example
``tutorials/01_modeling_basics/axisymmetric_transient_borehole.py`` uses
``model.add_borehole(layers=[...])`` with acoustic fluid, elastic casing, and
homogeneous formation. Axisymmetric fields use physical radial and axial
components; ``_r`` and ``_x`` select the radial component. The cylindrical
projection uses the volume weight ``2*pi*r``. S is the complement of P; direct
solenoidal and torsional projections are unsupported. Axisymmetric acoustic and isotropic elastic transients support axis-aligned PML
with impedance termination. The formation-fracture comparison is in
``tutorials/01_modeling_basics/axisymmetric_borehole_fractures.py``.

Reading native samples
----------------------

After a local run, or fetching remote output through the site's normal fetch
methods:

.. code-block:: python

   fields = job.wavefields.open()
   print(fields.groups, fields.times)
   pressure = fields.read("pressure", steps=[300, 500])
   traces = job.traces.open()  # when receiver groups are configured
   recorded = traces.time_domain("surface")

``TransientDataset`` returns xarray arrays with physical ``time`` and integer
``step`` coordinates. Dense outputs retain source, component, and receiver axes;
sparse outputs retain trace IDs. Component units and available receiver
coordinates come from the committed metadata. Reading loads the selected group
and steps; use ``steps`` to bound memory for large wavefields. There is no
frequency synthesis or temporal upscaling. The reader requires the successful
operation manifest and every requested shard, checks its input fingerprints,
and ignores stray files from older runs.

Sources, integration and restart
--------------------------------

``TimeFunction`` supports Ricker, Gaussian, sine, constant and delayed-step
functions, with analytic first and second derivatives available through
``evaluate(time, derivative=...)``. Its amplitude multiplies the physical
source strengths. Forces and elastic moments load its value; acoustic volume
injection loads its first derivative; acoustic scalar/monopole moments load its
second derivative times compressibility. Impulsive step derivatives are
unsupported. The SDK requires initially quiet wave loads to be compatible with
the solver's zero initial state; delay a pulse or start the interval earlier.

Recorded signatures reuse the existing ``SampledWavelet`` through
``TimeFunction.sampled``; synthetic ``KlauderWavelet`` definitions can also be
sampled explicitly with ``wavelet.sample(times)``. For a measured wavelet:

.. code-block:: python

   recording = fs.SampledWavelet(samples, dt=0.001, t0=-2.0)
   source = fs.TimeFunction.sampled(recording, delay=0.008)
   # Include the negative-time side and a quiet initial state.
   time = fs.TimeIntegration(start=-2.002, end=6.0, dt=0.002)
   job = fs.TransientJob("recorded", simulation, time=time, source=source)

The recorded values and clock are preserved; Fourier ``normalization`` does
not rescale physical-time samples. No filtering, taper, causal shift, or
resampling is implicit. Samples must be dimensionless multipliers of the
acquisition strengths; use ``recording.relative_to(strength, units=...)``
explicitly for physical recordings. The job embeds the samples, so packaging
and fingerprints include them without a separate file.

Between samples the backend and ``source.evaluate`` use cubic Hermite
interpolation with centered interior slopes and zero endpoint slopes. Values
outside the recorded interval are zero. Derivative loads require zero endpoint
samples: explicitly pad or taper a recording before using acoustic volume
injection or isotropic moments. The first derivative is continuous for zero
endpoints; the second is piecewise linear (right interval at interior knots,
left interval at the last sample). Source sampling and integration steps are
independent; choose the integration step and mesh adaptation frequency for the
recorded bandwidth. Sampled sources do not infer a PML frequency floor.
This addition requires a backend supporting ``Time.source.kind="sampled"``;
older builds reject that kind rather than substituting another wavelet.

``TimeIntegration`` defaults to ``Newmark(beta=0.25, gamma=0.5)``. Other Newmark
coefficients must satisfy the backend's unconditional-stability constraint.
Acoustic and elastic PML require the default coefficients and axis-aligned layers. ``pml_cfs_frequency`` specifies a
nonnegative complex-frequency shift in Hz; zero retains the unshifted form.
Use the same shift on overlapping PML directions. Nonzero shifts use CPU
memory updates. They can improve late-time stability in strongly layered models,
but check reflection error and decay for the chosen shift. Reduced borehole PML
currently requires zero shift. Axisymmetric PML runs on
the CPU. Coupled interfaces inside PML require a monolithic solve and at most
one nonzero tangential stretch (including fluid–casing interfaces in axial PML).
Interfaces at corners with two tangential stretches are unsupported. Cartesian 2.5D,
frequency-domain attenuation laws, spectral signatures, boundary loadings and harmonic velocity or
acceleration recovery are outside this SDK API. Request primary or supported
constitutive fields. Native transient imaging workflows are not exposed here.

Checkpoints contain rank-local state and any PML sidecars. Copy the entire
checkpoint directory before resuming. Keep the original interval, method,
mesh, materials, acquisition and rank count; the backend validates integrator
compatibility but cannot establish physical equivalence of changed inputs.

.. code-block:: python

   resumed = job.with_restart("/path/to/checkpoint-copy", name="resumed")
   resumed.save()
   site.run(resumed, check=True)

Relative restart paths resolve below the original job's result directory.
Restart inputs are fingerprinted and must stay immutable; select a separate
checkpoint output directory when writing new checkpoints. Remote staging copies
local restart inputs with the job. Checkpoints are not ordinary durable output
artifacts: preserve them on the execution site or explicitly copy the complete
directory before using a local restart. A resumed dataset contains only
scheduled samples at or after the checkpoint. Restart does not reconstruct
missing earlier samples. Reuse is strict for the complete trajectory;
``force=True`` requests a new run. Transient jobs require
``tolerate_failures=0``; an incomplete trajectory is a failed run.

Off-axis acoustic point sources
-------------------------------

An axisymmetric acoustic model can represent an off-axis scalar point source by
running real angular modes. This requires a rotationally invariant geometry,
materials and boundary conditions. Keep ``dimension=2`` and select
``physics="acoustic_axisym"`` (or ``physics="acoustic", axisymmetric=True``).
Set ``n_modes=N`` on the job to solve orders 0 through N-1. This requires a
Sauce build with transient mode-task support.

For an existing acoustic axisymmetric simulation and configured site:

.. code-block:: python

   job = fs.TransientJob(
       "point_source", sim, n_modes=9,  # orders 0, ..., 8
       time=fs.TimeIntegration(start=0, end=1, dt=0.001),
       source=fs.TimeFunction.gaussian(width=0.03, delay=0.3),
   )
   job += fs.WavefieldOutput(name="pressure", field="pressure", grid=grid)
   site.run(job, check=True)

   fields = job.wavefields.open()
   pressure = fields.read("pressure", delta_theta=0.7)
   around_borehole = fields.read("pressure", delta_theta=[0, 1.57, 3.14])
   coefficients = fields.read_modes("pressure")  # leading mode dimension
   mode_two = job.wavefields.open(mode=2).read("pressure")

The backend maps task 1 to order 0, task 2 to order 1, and so on. Existing local,
Slurm and AWS task scheduling handles the sweep without copying simulations or
writing a run loop. Each task publishes its own result manifest. Samples and
ParaView collections live below ``results/modes/mode_000000/``, etc.; checkpoint
and restart roots likewise contain ``modes/mode_000000/`` subdirectories. Custom
sample and visualization output paths must be relative to the result root.

Reconstruction requires every requested mode, matching coordinates and time
samples, and successful results for the current inputs. Missing or failed tasks
can be rerun independently: a normal rerun retains successful modes. ``force=True``
reruns all modes. Copy the entire checkpoint root to restart all modes using
``with_restart``. The normal saved-job loader restores ``n_modes`` automatically;
the same modal access is available through ``job.traces.open()``.

Omitting ``n_modes`` preserves the original single-task layout. The low-level
``simulation.azimuthal_mode`` selector remains available for individual-mode
experiments, but cannot be combined with job-owned ``n_modes``.

``delta_theta`` is receiver azimuth minus source azimuth, in radians. The reader
computes ``P0 + 2*sum(Pm*cos(m*delta_theta))`` using signed real pressure samples.
The standalone ``fs.reconstruct_azimuthal_pressure`` helper also accepts NumPy
arrays or aligned xarray DataArrays; an angle vector adds a
leading ``azimuth`` dimension. It rejects missing modes, complex fields and
mismatched coordinates. Keep the same source strengths and samples in every
mode; do not divide strengths by the number of modes or sum magnitudes. Sources
at different azimuths must be reconstructed separately before their pressures
are summed.

Use scalar/monopole moment or volume-injection sources. Order zero retains the
ring-source solution. Use ``axis`` or ``symmetric_r`` at the axis; positive
orders enforce zero pressure there. Increase both the number of modes and mesh
resolution until receiver traces converge, especially near the source. The
existing bandwidth adaptation does not estimate an angular truncation order.
Cylindrical PML and checkpoint/restart are supported; preserve the complete
checkpoint directory and the same angular order.

Positive orders currently support pressure output, not vector velocity or
Helmholtz projection. Elastic casing, coupled fluid-solid angular modes and
frequency-domain angular modes require further implementation.
