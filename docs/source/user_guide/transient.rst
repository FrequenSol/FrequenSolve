Direct time-domain simulation
=============================

``TransientJob`` advances acoustic pressure, elastic displacement, or coupled
acoustic–elastic waves in physical time. ``TimeDomainJob`` retains its harmonic
sweep and reconstruction API. A transient job is one solver task containing all
encoded sources, with native time samples and no frequency packing step.

Configure a Cartesian 2D/3D or axisymmetric 2D simulation with lossless materials, Galerkin
formulation, and an explicit mesh bandwidth. For an existing simulation ``sim``:

.. code-block:: python

   import frequensolve as fs

   sim.discretization = fs.Discretization(method="Galerkin")
   sim.model.attenuation_model = "none"
   sim.mesh.set_adapt(elems_per_wave=2, order=2, f_adapt=30)

   job = fs.TransientJob(
       "pulse",
       sim,
       time=fs.TimeIntegration(start=0, end=1, dt=0.001),
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

Snapshot schedules
------------------

Choose one of three schedules, in physical seconds or completed steps:

.. code-block:: python

   fs.TimeOutput(times=[0.3, 0.5, 0.75])  # explicit physical times
   fs.TimeOutput(interval=0.025)         # every 25 ms from start
   fs.TimeOutput(stride=25)              # every 25 integration steps

The shared schedule selects receiver samples, wavefield snapshots, and ParaView output. Requests
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

Transient visualization writes only explicitly requested fields, with real
parts by default. Material properties and mesh diagnostics such as ``Order``
and ``Domain`` are not added. Property/info items and imaginary parts are
rejected; ``parts=["abs"]`` requests magnitude explicitly. Encoded sources
produce separate field arrays, optionally filtered with ``sources=[...]``.
The shared ``TimeOutput`` schedule controls visualization output.

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
require Helmholtz projection.

For acoustic, elastic, or coupled transients, including axisymmetry, request
native, P, and S velocity fields explicitly:

.. code-block:: python

   sim.extra["helmholtz_projection"] = {
       "enabled": True, "field": "velocity", "parts": ["p", "s"],
       "s_projection": "complement",
   }
   job += fs.VtkOutput(
       name="elastic_velocity", fields=["velocity", "p_velocity", "s_velocity"],
   )

The projection runs once per scheduled output time before the fields are written.
In solids, ``velocity`` is the accepted Newmark rate. In fluids it is
``-grad(integral(p dt))/rho``, starting from zero fluid velocity. The pressure
integral advances at every accepted step and is checkpointed, regardless of output
cadence. Older checkpoints without that integral cannot recover fluid velocity.
The default S component is the complement ``velocity - p_velocity``. Use
``field="displacement"`` in standalone elastic jobs with ``displacement``, ``p_displacement`` and
``s_displacement`` to decompose displacement instead. All outputs in a projected
job must use the selected field family. Scalar vector receiver components need
a measurement direction. ``parts`` must be explicit to retain the native field;
the legacy single ``part`` replacement mode is unsupported by ``TransientJob``.

A complete runnable example is provided in
``tutorials/01_modeling_basics/elastic_transient_helmholtz.py``.
The fluid-filled, steel-cased borehole example
``tutorials/01_modeling_basics/axisymmetric_transient_borehole.py`` uses
``model.add_borehole(layers=[...])`` with acoustic fluid, elastic casing, and
homogeneous formation. Axisymmetric fields use physical radial and axial
components; ``_r`` and ``_x`` select the radial component. The cylindrical
projection uses the volume weight ``2*pi*r``. S is the complement of P; direct
solenoidal and torsional projections are unsupported. Axisymmetric acoustic and isotropic elastic transients support unshifted PML
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

``TimeIntegration`` defaults to ``Newmark(beta=0.25, gamma=0.5)``. Other Newmark
coefficients must satisfy the backend's unconditional-stability constraint.
Acoustic and elastic PML require the default coefficients, unshifted
axis-aligned layers, and zero ``pml_cfs_frequency``. Axisymmetric PML runs on
the CPU. Coupled interfaces inside PML require a monolithic solve and at most
one nonzero tangential stretch (including fluid–casing interfaces in axial PML).
Interfaces at corners with two tangential stretches are unsupported. 2.5D,
attenuation, spectral signatures, boundary loadings and harmonic velocity or
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
