Tutorials
=========

The tutorial notebooks live under ``examples/tutorials``. They are linked from
the documentation site as notebook files rather than rendered or executed during
the Sphinx build. This keeps documentation builds independent of local solver,
cloud, and HPC availability.

How To Use These Tutorials
--------------------------

The notebooks are ordered as a learning path. Start with acoustic modeling even
if your target problem is elastic, poroelastic, or coupled: the first tutorial
introduces the project layout, layered model vocabulary, generated meshes,
acquisition objects, jobs, traces, logs, and ParaView output. Later notebooks
reuse that same shape and add one idea at a time.

Each tutorial should leave the reader with three concrete artifacts:

1. A small project directory whose generated inputs can be inspected.
2. One strict solver run cell that either succeeds or leaves logs/results in
   place for debugging.
3. A plotted or listed result that shows what the API option changed.

The companion user-guide pages define the catalog-style tables and conceptual
reference material; the notebooks show the same concepts in complete runnable
workflows.

Learning Path
-------------

Read the sections in order when learning the package for the first time:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Stage
     - What you learn
     - Why it comes here
   * - Modeling basics
     - Projects, simulations, material properties, mesh sizing, acquisition,
       strict jobs, traces, and ParaView QC.
     - This is the vocabulary every later tutorial reuses.
   * - Sites
     - Local, cloud, and HPC execution plus saved project/job loading.
     - Once a job is well-authored, execution location and later reuse should
       be site/file choices.
   * - Velocity models
     - Units, coordinate systems, topography, and layered geometry.
     - Larger models need explicit coordinates and inspectable property data.
   * - Meshing
     - Generated meshes, adaptivity fields, and gradings.
     - Mesh controls connect model geometry, sources, receivers, and cost.
   * - Surveys
     - Receiver devices, DAS, source mechanisms, batching, and sparse layouts.
     - Production data volume is controlled by acquisition design.
   * - Outputs
     - Trace stores, ``xarray`` reads, SEG-Y export, ParaView/VTK products, and imaging outputs.
     - Results need to be reusable after the solver run completes.
   * - Performance
     - Frequency-domain QC, time-domain sweep timing, source batching, receiver sampling, and imaging cost.
     - Production runs need measurable diagnostics before they get expensive.

Modeling Basics
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - Notebook
     - Focus
     - File
   * - Acoustic
     - Project layout, layered models, mesh order, EPW, traces, and ParaView.
     - :download:`01_acoustic.ipynb <../../../examples/tutorials/01_modeling_basics/01_acoustic.ipynb>`
   * - Elastic
     - Elastic layers, receivers, attenuation through ``Qp``/``Qs``, and VTK QC.
     - :download:`02_elastic.ipynb <../../../examples/tutorials/01_modeling_basics/02_elastic.ipynb>`
   * - Poroelastic
     - Elastic-frame properties plus pore-fluid properties, traces, and VTK QC.
     - :download:`03_poroelastic.ipynb <../../../examples/tutorials/01_modeling_basics/03_poroelastic.ipynb>`
   * - Coupled
     - Mixed material domains with domain-specific receivers and VTK QC.
     - :download:`04_coupled.ipynb <../../../examples/tutorials/01_modeling_basics/04_coupled.ipynb>`
   * - 2.5D, 3D, Axisymmetric
     - Acoustic model dimensionality and cylindrical axisymmetric setup.
     - :download:`05_acoustic_25d_3d_axisymmetric.ipynb <../../../examples/tutorials/01_modeling_basics/05_acoustic_25d_3d_axisymmetric.ipynb>`
   * - Laplace And Time-Domain
     - Laplace-domain damping, compensated time-domain reads, and wrap-around control.
     - :download:`06_laplace_time_domain.ipynb <../../../examples/tutorials/01_modeling_basics/06_laplace_time_domain.ipynb>`

Site Tutorials
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - Notebook
     - Focus
     - File
   * - AWS Site
     - Cloud authentication, separate trace/QC jobs, storage, and result fetching.
     - :download:`01_aws_site.ipynb <../../../examples/tutorials/02_sites/01_aws_site.ipynb>`
   * - HPC Sites
     - SLURM run configuration, remote paths, trace jobs, and QC output jobs.
     - :download:`02_hpc_sites.ipynb <../../../examples/tutorials/02_sites/02_hpc_sites.ipynb>`
   * - Local Site
     - Local Dask-backed trace/QC execution with an installed solver.
     - :download:`03_local_site.ipynb <../../../examples/tutorials/02_sites/03_local_site.ipynb>`
   * - Save And Load Projects
     - Persisting and reopening projects, simulations, and job JSON files.
     - :download:`04_save_load_projects_jobs.ipynb <../../../examples/tutorials/02_sites/04_save_load_projects_jobs.ipynb>`

Velocity Model Building
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - Notebook
     - Focus
     - File
   * - Variable Properties And Units
     - ``xarray`` material properties, Pint quantities, and unit metadata.
     - :download:`01_variable_properties_units.ipynb <../../../examples/tutorials/03_velocity_model_building/01_variable_properties_units.ipynb>`
   * - Coordinate Systems
     - Topography, surface-relative properties, and ``surface.below()`` points.
     - :download:`02_coordinate_systems.ipynb <../../../examples/tutorials/03_velocity_model_building/02_coordinate_systems.ipynb>`
   * - Layered Models
     - Non-interface surfaces, borehole subdomains, and uniform sampling.
     - :download:`03_layered_models.ipynb <../../../examples/tutorials/03_velocity_model_building/03_layered_models.ipynb>`

Meshing
-------

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - Notebook
     - Focus
     - File
   * - Meshes Versus Generators
     - Generated meshes, supplied meshes, GMP, and PyVista screenshots.
     - :download:`01_mesh_vs_generators.ipynb <../../../examples/tutorials/04_meshing/01_mesh_vs_generators.ipynb>`
   * - Adaptivity Fields
     - ``vadapt``, ``epw_mult``, ``hmin``, and ``hmax``.
     - :download:`02_adaptivity_fields.ipynb <../../../examples/tutorials/04_meshing/02_adaptivity_fields.ipynb>`
   * - Gradings
     - Source, receiver, and model-surface grading controls.
     - :download:`03_gradings.ipynb <../../../examples/tutorials/04_meshing/03_gradings.ipynb>`

Survey Tutorials
----------------

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - Notebook
     - Focus
     - File
   * - Receivers
     - Multi-component devices and dense receiver groups.
     - :download:`01_receivers.ipynb <../../../examples/tutorials/05_surveys/01_receivers.ipynb>`
   * - DAS
     - Straight and helical DAS plus pointwise strain.
     - :download:`02_das.ipynb <../../../examples/tutorials/05_surveys/02_das.ipynb>`
   * - Sources
     - Scalar point sources, compound sources, and source batching.
     - :download:`03_sources.ipynb <../../../examples/tutorials/05_surveys/03_sources.ipynb>`
   * - Sparse Surveys
     - Offset-domain and explicit sparse source-receiver layouts.
     - :download:`04_sparse_surveys.ipynb <../../../examples/tutorials/05_surveys/04_sparse_surveys.ipynb>`

Output Tutorials
----------------

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - Notebook
     - Focus
     - File
   * - Traces
     - ``TraceDataset``, HDF5 traces, xarray reads, and SEGY export.
     - :download:`01_traces.ipynb <../../../examples/tutorials/06_outputs/01_traces.ipynb>`
   * - ParaView And VTK
     - Volume, surface, plane, field, property, source, and PML output controls.
     - :download:`02_paraview_vtk.ipynb <../../../examples/tutorials/06_outputs/02_paraview_vtk.ipynb>`
   * - Imaging
     - RTM imaging jobs, FWI-gradient image requests, image grids, and ``ImageDatabase`` reads.
     - :download:`03_imaging.ipynb <../../../examples/tutorials/06_outputs/03_imaging.ipynb>`

Performance Tutorials
---------------------

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - Notebook
     - Focus
     - File
   * - Performance
     - Frequency-domain QC, time-domain phase timings, source batching, receiver sampling cost, and imaging assembly reuse.
     - :download:`01_performance.ipynb <../../../examples/tutorials/07_performance/01_performance.ipynb>`

Related User Guide Pages
------------------------

- :doc:`../user_guide/physics_materials_boundaries`
- :doc:`../user_guide/api_to_contracts`
- :doc:`../user_guide/projects_jobs_sites`
- :doc:`../user_guide/velocity_models_coordinates`
- :doc:`../user_guide/mesh_generation_adaptivity`
- :doc:`../user_guide/surveys_sources_receivers`
- :doc:`../user_guide/traces_paraview_outputs`
