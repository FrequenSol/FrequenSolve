Traces, ParaView, and Imaging Outputs
=====================================

Outputs are job-owned. :term:`Trace` output is enabled by default; :term:`ParaView`
and wavefield outputs are requested on jobs when needed.

Related tutorials:

- :download:`Traces <../../../examples/tutorials/06_outputs/01_traces.ipynb>`
  for :term:`TraceDataset <trace dataset>` reads, :term:`HDF5`-backed data, and
  :term:`SEG-Y` export.
- :download:`ParaView and VTK <../../../examples/tutorials/06_outputs/02_paraview_vtk.ipynb>`
  for :term:`VTK`/:term:`VTU` field, surface, plane, and :term:`PML` output controls.
- :download:`Imaging <../../../examples/tutorials/06_outputs/03_imaging.ipynb>`
  for :term:`RTM` and :term:`FWI`-gradient image requests.
- :download:`Acoustic modeling output workflow <../../../examples/tutorials/01_modeling_basics/01_acoustic.ipynb>`
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

:term:`Laplace-domain` time sweeps can be requested with
``TimeDomainJob(..., damping_factor=...)`` or the lower-level
``TimeDomainJob(..., laplace=...)`` offset. ``traces.ld(...)`` reconstructs the
damped Laplace-domain time series directly. ``traces.td(...)`` applies the
matching amplitude compensation automatically when Laplace metadata are present;
pass ``laplace_compensation="off"`` to inspect the uncompensated result.

ParaView Output
---------------

:term:`ParaView output` is usually requested from a single-frequency job:

.. code-block:: python

   job = fs.FrequencyDomainJob(
       name="freq_20hz",
       simulation=sim,
       f_list=[20.0],
       outputs=[
           fs.ParaviewOutput(
               name="pv",
               fields=["pressure"],
               properties=["vp", "rho", "Subdomain"],
               show_pml=True,
               upscale=1,
               order=2,
           )
       ],
   )

The public API exposes volume, surface, and grid targets:

.. code-block:: python

   fs.ParaviewOutput.volume(fields=["pressure"])
   fs.ParaviewOutput.surface(surfaces="top", fields=["pressure"])
   fs.ParaviewOutput.surface(
       plane={"axis": "z", "value": 0.25, "units": "km"},
       parts=["real", "imag", "abs"],
   )

``order`` controls the :term:`polynomial order` used when exporting fields for
visualization. ``upscale`` controls extra sampling inside elements. Use
``upscale=0`` for a native, low-cost mesh :term:`QC` view, and increase it when smoother
field images are more important than smaller files.

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

Successful imaging runs can be opened with ``site.fetch_image(job)`` or
``ImageDatabase``:

.. code-block:: python

   image_db = site.fetch_image(job)
   raw = image_db.raw_images
   smoothed = image_db.smoothed_images

Both properties return :term:`xarray` ``Dataset`` objects on the requested image grid.
The raw dataset reads the solver ``image/raw`` group; the smoothed dataset reads
``image/phi`` when the solver writes that group.
