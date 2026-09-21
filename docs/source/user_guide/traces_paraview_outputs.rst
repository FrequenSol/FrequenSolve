Traces, ParaView, and Imaging Outputs
=====================================

Outputs are job-owned. :term:`Trace` output is enabled by default; :term:`ParaView`
and wavefield outputs are requested on jobs when needed.

Related tutorials:

- :download:`Traces <../../../tutorials/06_outputs/01_traces.ipynb>`
  for :term:`TraceDataset <trace dataset>` reads, :term:`HDF5`-backed data, and
  :term:`SEG-Y` export.
- :download:`ParaView and VTK <../../../tutorials/06_outputs/02_paraview_vtk.ipynb>`
  for :term:`VTK`/:term:`VTU` field, surface, plane, and :term:`PML` output controls.
- :download:`Imaging <../../../tutorials/06_outputs/03_imaging.ipynb>`
  for :term:`RTM` and :term:`FWI`-gradient image requests.
- :download:`Acoustic modeling output workflow <../../../tutorials/01_modeling_basics/01_acoustic.ipynb>`
  for the first end-to-end trace and ParaView output workflow.

TraceDataset
------------

``RunResult.traces()`` returns ``TraceDataset``:

.. code-block:: python

   traces = result.traces(upscale=4)
   group = traces.groups[0]
   component = traces.components(group)[0]
   source = traces.sources(group)[0]

   fd = traces.fd(group, component, source)
   td = traces.td(
       group,
       component,
       source,
       fs.RickerWavelet(f=15.0),
       upscale=4,
   )
   ld = traces.ld(
       group,
       component,
       source,
       fs.RickerWavelet(f=15.0),
       upscale=4,
   )

``RunResult.traces()`` and ``RunResult.wavefields()`` require successful output
files. For failed, cancelled, or timed-out runs, inspect ``result.status`` and
``result.logs()`` before attempting to read traces or wavefields.

:term:`Trace` reads return :term:`xarray` ``DataArray`` objects. Trace files
are :term:`HDF5`-backed and may be consolidated into a local cache with
``traces.consolidate()``. :term:`SEG-Y`
export is available through the trace-record helpers when the ``seismic-io``
extra is installed.

Wavelets are applied when :term:`time-domain` traces are reconstructed. For
``RickerWavelet``, the peak is placed at physical time zero while the generated
signal still includes pre-zero-time samples. If ``center`` is not supplied,
``RickerWavelet(f=...)`` uses one period of center padding,
``1 / f``.

Derivative-assisted time reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TimeDomainJob`` accepts the same ``phase_derivatives`` maximum as
``FrequencyDomainJob``. A positive value selects ``forward_df`` while retaining
the job's uniform frequency sweep, so receiver and gridded-wavefield results
include ``_df`` through the requested order. This can be combined with
``laplace`` or ``damping_factor``; the derivatives are taken with respect to
real physical frequency along the fixed Laplace-damped contour.

.. code-block:: python

   damped = fs.TimeDomainJob(
       name="damped_derivatives",
       simulation=sim,
       f_min=0.0,
       f_max=2.5 * f0,
       T_max=1.0,
       damping_factor=10.0,
       phase_derivatives=3,
   )

An experimental Hermite path can reduce the number of solved frequencies while
retaining the target time window:

.. code-block:: python

   job = fs.TimeDomainJob(
       name="time_hermite",
       simulation=sim,
       f_min=1.0,
       f_max=2.5 * f0,
       T_max=1.0,
       reconstruction="hermite",
       sample_every=4,
       high_frequency_taper=4.0,  # transition width in hertz
       interpolation_time_shift=0.35,
   )

   result = site.run(job)
   traces = result.traces()
   gather = traces.td(group, component, source, fs.RickerWavelet(f=f0))

This selects the solver's ``forward_df`` workflow and normally keeps every
fourth frequency. The trace reader reconstructs the omitted spectrum with a
cubic Hermite interpolant. It first interpolates the unweighted solver response
from ``R`` and ``R_df``, then evaluates and applies the wavelet on the dense
frequency grid. Hermite reconstruction automatically requests at least first
order; a larger explicit ``phase_derivatives`` value retains higher-order
datasets for other analysis.

``interpolation_time_shift`` removes a reference linear phase before the
Hermite interpolation and restores it on the dense grid. A positive value
``tau`` demodulates a delayed response proportional to
``exp(-2j * pi * f * tau)``. The transformed Hermite slope includes the phase
factor's derivative as well as the solver response derivative. Choose the
shift near the arrival time whose phase rotation should be flattened; a single
shift cannot simultaneously flatten separated arrivals, and receiver-dependent
travel times may require reconstructing receivers with different shifts.

With ``high_frequency_taper=True``, reconstruction adds a virtual endpoint one
solved-frequency interval above the highest frequency. Both its value and slope
are zero, producing a continuously differentiable rolloff instead of abruptly
padding the last nonzero sample with zeros. A positive number may be supplied
instead to set the continuation width in hertz; the numeric width is preserved
in saved jobs and trace manifests. This option also works with
``reconstruction="standard"``. It automatically requests the first phase
derivative and uses the simulated slope at the highest frequency for the
continuation; all original frequency samples remain unchanged.

Frequency derivatives double the information available at each frequency but
do not remove the frequency-domain sampling limit imposed by late arrivals.
Check the reconstructed traces against a standard sweep when increasing
``sample_every``; sampling every fourth target frequency is deliberately an
aggressive experimental setting.

:term:`Laplace-domain` time sweeps can be requested with
``TimeDomainJob(..., damping_factor=...)`` or the lower-level
``TimeDomainJob(..., laplace=...)`` offset. ``traces.ld(...)`` reconstructs the
damped Laplace-domain time series directly. ``traces.td(...)`` applies the
matching amplitude compensation automatically when Laplace metadata are present;
pass ``laplace_compensation="off"`` to inspect the uncompensated result.

Phase-derivative and Helmholtz wavefields
-----------------------------------------

``FrequencyDomainJob`` and ``TimeDomainJob`` can request physical-frequency
derivatives through fourth order. The value is the highest order to compute;
Sauce also writes every lower order, including the base field at order zero:

.. code-block:: python

   job = fs.FrequencyDomainJob(
       name="elastic_derivatives",
       simulation=sim,
       f_list=[20.0],
       phase_derivatives=4,
       outputs=[
           fs.WavefieldOutput(
               name="p_velocity",
               field="p_velocity",
               grid=grid,
           ),
           fs.WavefieldOutput(
               name="s_velocity",
               field="s_velocity",
               grid=grid,
           ),
       ],
   )

   result = site.run(job)
   wavefields = result.wavefields()
   s_velocity_d3f = wavefields.fd(
       "s_velocity_d3f",
       "s_velocity",
       source=1,
   )

A positive ``phase_derivatives`` value selects Sauce's ``forward_df`` workflow
and serializes its maximum as ``derivative_order``. Use an integer from 0
through 4; zero is the ordinary forward solve unless derivative-assisted time
reconstruction or tapering requires first order. Naming a wavefield output
``s_velocity`` therefore makes its third-order result available as the
``s_velocity_d3f`` group. The first-order suffix is ``_df``.

When a physical-frequency derivative trace group is reconstructed directly,
``traces.td(...)`` and ``traces.ld(...)`` apply ``i**order`` before the real
inverse transform. Raw derivatives alternate between anti-Hermitian and
Hermitian symmetry; this phase factor restores Hermitian parity and yields the
real time-moment representation. Frequency-domain reads remain raw and
unchanged.

The ``p_velocity`` and ``s_velocity`` fields request the compressional and
shear Helmholtz projections. Sauce enables the decomposition automatically
when either field is present, so no separate simulation option is required for
the default projection. Component forms such as ``p_velocity_x`` and
``s_velocity_z`` are also accepted. Advanced projection settings can still be
passed as the simulation's ``helmholtz_projection`` option.

VTK / ParaView Output
---------------------

:term:`ParaView output` is usually requested from a single-frequency job:

``VtkOutput`` is the public authoring name because VTU/VTR is the normal output
path. The same request can select XDMF when needed, and it continues to serialize
under the solver's ``Outputs.ParaView`` contract.

.. code-block:: python

   job = fs.FrequencyDomainJob(
       name="freq_20hz",
       simulation=sim,
       f_list=[20.0],
       outputs=[
           fs.VtkOutput.domain(
               name="pv",
               fields=["pressure"],
               properties=["vp", "rho", "Subdomain"],
               show_pml=True,
               upscale=1,
               order=2,
           )
       ],
   )

The public API exposes domain, surface, and grid targets. ``domain`` means the
entire computational domain in both 2D and 3D; it maps to the solver's
dimension-independent ``"volume"`` target internally:

.. code-block:: python

   fs.VtkOutput.domain(fields=["pressure"])
   fs.VtkOutput.surface(surfaces="top", fields=["pressure"])
   fs.VtkOutput.surface(
       plane={"axis": "z", "value": 0.25, "units": "km"},
       parts=["real", "imag", "abs"],
   )

Configured requests are available as ``job.outputs.vtk`` and their resolved
local paths as ``job.vtk_outputs``. Sites use the same vocabulary when results
need to be transferred explicitly:

.. code-block:: python

   site.fetch_vtk(job)
   job.vtk_outputs

``order`` controls the :term:`polynomial order` used when exporting fields for
visualization. The domain and surface helpers accept ``upscale`` values from 0
through 2 under their target mesh options; omission means 0. Use ``upscale=0``
for a native, low-cost mesh :term:`QC` view, and increase it when smoother field
images are more important than smaller files. Grid targets do not accept
``upscale``.

VTK/PyVista Helpers
-------------------

``read_vtu`` and ``plot_vtu`` provide quick Python inspection of solver :term:`VTU`
files:

.. code-block:: python

   files = result.output_files(suffix=".vtu", existing=True)
   mesh = fs.read_vtu(files[0])
   fs.vtu_fields(mesh)
   fs.plot_vtu(mesh, field="pressure", part="real", scalar_bar=True)

:term:`ParaView` remains the recommended application for large meshes, multiple
datasets, and interactive analysis. :term:`PyVista` is best for lightweight notebook
figures and saved screenshots.

Imaging Output
--------------

Imaging jobs use the :term:`RTM` workflow and are usually created with
``simulation.imaging(...)``:

.. code-block:: python

   image_grid = fs.CartesianGrid(
       n=[161, 81],
       x0=[0.0, 0.0],
       x1=[1.2, 0.5],
   )

   job = sim.imaging(
       name="rtm",
       observed=observed_job_or_trace_path,
       grid=image_grid,
       parameters=["vp", "vs", "rho"],
       fields=["velocity"],
       condition="up_down",
       weights=[1.0, 0.8, 0.45],
       misfit_norm="L2",
   )

``parameters`` request :term:`FWI`-gradient image conditions. The public names
``"vp"``, ``"vs"``, and ``"rho"`` serialize to solver properties ``Vp``,
``Vs``, and ``Rho``. ``fields`` and ``condition`` request diagnostic image
conditions. For exact solver-condition names, pass ``images={...}``; values of
the form ``"FWI:Vp"`` request property-gradient images, while other strings are
passed as image-condition names.

Observed data may be supplied as a trace-producing job, a :term:`TraceDataset <trace dataset>`, or a
filesystem path. Receiver-group names in the observed data must match the
:term:`simulation` acquisition, because the imaging misfit pairs observed and simulated
receiver groups by name.

Arbitrary nonnegative residual weights can be attached to a misfit receiver
group. The array shape determines whether weights vary by receiver,
component/receiver, or source/component/receiver; sparse trace-catalog weights
use an explicit ``layout="sparse_trace"``. Saved jobs put production arrays in
a job-owned HDF5 file and retain only its reference in job JSON and provenance:

.. code-block:: python

   group = job.misfit.receiver_groups[0]
   group.add_trace_weights(
       quality_weights,  # shape: (source, component, receiver)
       name="data_quality",
   )

Successful imaging runs can be opened directly from local files with
``job.load_images()``. This does not contact an execution site:

.. code-block:: python

   job = fs.load(job_file)
   image_db = job.load_images()
   raw = image_db.raw_images
   smoothed = image_db.smoothed_images

For a remote run, call ``site.fetch_image(job)`` once to copy the image files
locally. Subsequent sessions can reload the job and use ``job.load_images()``
without reconnecting to the site.

Both properties return :term:`xarray` ``Dataset`` objects on the requested image grid.
The raw dataset reads the solver ``image/raw`` group; the smoothed dataset reads
``image/smoothed`` when the solver writes that group.
