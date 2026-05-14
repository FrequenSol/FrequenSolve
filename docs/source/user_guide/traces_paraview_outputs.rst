Traces, ParaView, And Imaging Outputs
=====================================

Outputs are job-owned. Trace output is enabled by default; ParaView and
wavefield outputs are requested on jobs when needed.

Primary tutorials:

- :download:`Traces <../../../examples/tutorials/06_outputs/01_traces.ipynb>`
- :download:`ParaView and VTK <../../../examples/tutorials/06_outputs/02_paraview_vtk.ipynb>`
- :download:`Imaging <../../../examples/tutorials/06_outputs/03_imaging.ipynb>`
- :download:`Acoustic modeling output workflow <../../../examples/tutorials/01_modeling_basics/01_acoustic.ipynb>`

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

Trace reads return ``xarray.DataArray`` objects. Trace files are HDF5-backed and
may be consolidated into a local cache with ``traces.consolidate()``. SEGY
export is available through the trace-record helpers when the ``seismic-io``
extra is installed.

Wavelets are applied when time-domain traces are reconstructed. For
``RickerWavelet``, the peak is placed at physical time zero while the generated
signal still includes pre-zero-time samples. The clearer keyword for that
padding is ``pre_time``; the older ``center`` keyword is retained as an alias.
If neither is supplied, ``RickerWavelet(f=...)`` uses one period of pre-time,
``1 / f``.

ParaView Output
---------------

ParaView output is usually requested from a single-frequency job:

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

``order`` controls the polynomial order used when exporting fields for
visualization. ``upscale`` controls extra sampling inside elements. Use
``upscale=0`` for a native, low-cost mesh QC view, and increase it when smoother
field images are more important than smaller files.

VTK/PyVista Helpers
-------------------

``read_vtu`` and ``plot_vtu`` provide quick Python inspection of solver VTU
files:

.. code-block:: python

   files = result.output_files(suffix=".vtu", existing=True)
   mesh = fs.read_vtu(files[0])
   fs.vtu_fields(mesh)
   fs.plot_vtu(mesh, field="pressure", part="real", scalar_bar=True)

ParaView remains the recommended application for large meshes, multiple
datasets, and interactive analysis. PyVista is best for lightweight notebook
figures and saved screenshots.

Imaging Output
--------------

Imaging jobs use the ``RTM`` workflow and are usually created with
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

``parameters`` request FWI-gradient image conditions. The public names
``"vp"``, ``"vs"``, and ``"rho"`` serialize to solver properties ``Vp``,
``Vs``, and ``Rho``. ``fields`` and ``condition`` request diagnostic image
conditions. For exact solver-condition names, pass ``images={...}``; values of
the form ``"FWI:Vp"`` request property-gradient images, while other strings are
passed as image-condition names.

Observed data may be supplied as a trace-producing job, a ``TraceDataset``, or a
filesystem path. Receiver-group names in the observed data must match the
simulation acquisition, because the imaging misfit pairs observed and simulated
receiver groups by name.

Successful imaging runs can be opened with ``site.fetch_image(job)`` or
``ImageDatabase``:

.. code-block:: python

   image_db = site.fetch_image(job)
   raw = image_db.raw_images
   smoothed = image_db.smoothed_images

Both properties return ``xarray.Dataset`` objects on the requested image grid.
The raw dataset reads the solver ``image/raw`` group; the smoothed dataset reads
``image/phi`` when the solver writes that group.
