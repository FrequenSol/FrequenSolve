Tutorials
=========

The tutorial notebooks live under `examples/tutorials
<https://github.com/FrequenSol/FrequenSolve/tree/v2/examples/tutorials>`__
in the GitHub repository. They are linked from the documentation site as
notebook files rather than rendered or executed during the Sphinx build. This
keeps documentation builds independent of local solver, cloud, and :term:`HPC`
availability.

Running a Notebook
------------------

1. Install the :term:`Python API` and any extras listed in the notebook table below.
2. Download the notebook from this page or the `FrequenSolve repository
   <https://github.com/FrequenSol/FrequenSolve/tree/v2/examples/tutorials>`__.
3. Open it in JupyterLab, Jupyter Notebook, VS Code, or another notebook
   environment.
4. Configure a :term:`site` before running solver cells. Cloud tutorials need a
   FrequenSol Cloud account and license; local tutorials need a local solver
   installation; HPC tutorials need :term:`SSH`/:term:`SLURM` credentials and a
   solver installation on the cluster.
5. Run cells in order. If a solver cell fails, inspect the generated
   :term:`project` directory, :term:`job` logs, and :term:`run result`
   before rerunning.

The Requirements column below lists prerequisites in addition to the base
``frequensolve`` install. Terms such as :term:`EPW`, :term:`PML`, :term:`DAS`,
:term:`RTM`, and :term:`FWI` are defined in :doc:`../glossary`.

How to Use These Tutorials
--------------------------

The notebooks are ordered as a learning path. Start with acoustic modeling even
if your target problem is elastic, :term:`poroelastic`, or coupled: the first
tutorial introduces the project layout, layered model vocabulary, generated
meshes, acquisition objects, jobs, traces, logs, and :term:`ParaView output`.
Later notebooks reuse that same shape and add one idea at a time.

Each tutorial should leave the reader with three concrete artifacts:

1. A small project directory whose generated inputs can be inspected.
2. One :term:`strict job` cell that either succeeds or leaves logs/results in
   place for debugging.
3. A plotted or listed result that shows what the selected option changed.

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
       :term:`strict jobs <strict job>`, traces, and :term:`ParaView` QC.
     - This is the vocabulary every later tutorial reuses.
   * - Sites
     - Local, cloud, and :term:`HPC` execution plus saved project/job loading.
     - Once a job is well-authored, execution location and later reuse should
       be site/file choices.
   * - Velocity models
     - Units, coordinate systems, topography, and layered geometry.
     - Larger models need explicit coordinates and inspectable property data.
   * - Meshing
     - Generated meshes, adaptivity fields, and gradings.
     - Mesh controls connect model geometry, sources, receivers, and cost.
   * - Surveys
     - :term:`Receiver devices <receiver device>`, :term:`DAS`, source
       mechanisms, batching, and sparse layouts.
     - Production data volume is controlled by acquisition design.
   * - Outputs
     - Trace stores, :term:`xarray` reads, :term:`SEG-Y` export,
       :term:`ParaView`/:term:`VTK` products, and imaging outputs.
     - Results need to be reusable after the solver run completes.
   * - Performance
     - :term:`Frequency-domain` QC, :term:`time-domain` sweep timing,
       :term:`source batching`, receiver sampling, and imaging cost.
     - Production runs need measurable diagnostics before they get expensive.

Modeling Basics
---------------

.. list-table::
   :header-rows: 1
   :widths: 24 40 20 16

   * - Notebook
     - Focus
     - Requirements
     - File
   * - Acoustic
     - Project layout, layered models, mesh order, :term:`EPW`, traces, and
       :term:`ParaView`.
     - ``visual`` plus a configured site for solver cells.
     - :download:`01_acoustic.ipynb <../../../examples/tutorials/01_modeling_basics/01_acoustic.ipynb>`
   * - Elastic
     - Elastic layers, receivers, :term:`attenuation` through ``Qp``/``Qs``,
       and :term:`VTK` QC.
     - ``visual`` plus a configured site.
     - :download:`02_elastic.ipynb <../../../examples/tutorials/01_modeling_basics/02_elastic.ipynb>`
   * - Poroelastic
     - Elastic-frame properties plus pore-fluid properties, traces, and
       :term:`VTK` QC.
     - ``visual`` plus a configured site.
     - :download:`03_poroelastic.ipynb <../../../examples/tutorials/01_modeling_basics/03_poroelastic.ipynb>`
   * - Coupled
     - Mixed material domains with domain-specific receivers and VTK QC.
     - ``visual`` plus a configured site.
     - :download:`04_coupled.ipynb <../../../examples/tutorials/01_modeling_basics/04_coupled.ipynb>`
   * - 2.5D, 3D, Axisymmetric
     - Acoustic model dimensionality and cylindrical axisymmetric setup.
     - ``visual`` plus a configured site.
     - :download:`05_acoustic_25d_3d_axisymmetric.ipynb <../../../examples/tutorials/01_modeling_basics/05_acoustic_25d_3d_axisymmetric.ipynb>`
   * - Laplace and Time-Domain
     - :term:`Laplace-domain` damping, compensated :term:`time-domain` reads,
       and wrap-around control.
     - ``visual`` plus a configured site.
     - :download:`06_laplace_time_domain.ipynb <../../../examples/tutorials/01_modeling_basics/06_laplace_time_domain.ipynb>`

Site Tutorials
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 40 20 16

   * - Notebook
     - Focus
     - Requirements
     - File
   * - AWS Site
     - Cloud authentication, separate trace/QC jobs, storage, and result fetching.
     - ``cloud``, account, license, and network access.
     - :download:`01_aws_site.ipynb <../../../examples/tutorials/02_sites/01_aws_site.ipynb>`
   * - HPC Sites
     - :term:`SLURM` run configuration, remote paths, trace jobs, and QC output
       jobs.
     - ``hpc``, :term:`SSH`/:term:`SLURM` credentials, allocation, and cluster
       solver.
     - :download:`02_hpc_sites.ipynb <../../../examples/tutorials/02_sites/02_hpc_sites.ipynb>`
   * - Local Site
     - Local :term:`Dask`-backed trace/QC execution with an installed solver.
     - ``parallel``, local solver, and :term:`FS_SOLVER_PATH`.
     - :download:`03_local_site.ipynb <../../../examples/tutorials/02_sites/03_local_site.ipynb>`
   * - Save and Load Projects
     - Persisting and reopening projects, simulations, and job JSON files.
     - Configured site only for rerun cells.
     - :download:`04_save_load_projects_jobs.ipynb <../../../examples/tutorials/02_sites/04_save_load_projects_jobs.ipynb>`

Velocity Model Building
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 40 20 16

   * - Notebook
     - Focus
     - Requirements
     - File
   * - Variable Properties and Units
     - :term:`xarray` material properties, :term:`Pint` quantities, and unit
       metadata.
     - ``visual`` plus a configured site.
     - :download:`01_variable_properties_units.ipynb <../../../examples/tutorials/03_velocity_model_building/01_variable_properties_units.ipynb>`
   * - Coordinate Systems
     - Topography, surface-relative properties, and ``surface.below()`` points.
     - ``visual`` plus a configured site.
     - :download:`02_coordinate_systems.ipynb <../../../examples/tutorials/03_velocity_model_building/02_coordinate_systems.ipynb>`
   * - Layered Models
     - Non-interface surfaces, :term:`borehole` subdomains, and uniform
       sampling.
     - ``visual`` plus a configured site.
     - :download:`03_layered_models.ipynb <../../../examples/tutorials/03_velocity_model_building/03_layered_models.ipynb>`

Meshing
-------

.. list-table::
   :header-rows: 1
   :widths: 24 40 20 16

   * - Notebook
     - Focus
     - Requirements
     - File
   * - Meshes versus Generators
     - Generated meshes, supplied meshes, :term:`GMP`, and :term:`PyVista`
       screenshots.
     - ``visual`` plus a configured site for solver cells.
     - :download:`01_mesh_vs_generators.ipynb <../../../examples/tutorials/04_meshing/01_mesh_vs_generators.ipynb>`
   * - Adaptivity Fields
     - ``vadapt``, ``epw_mult``, ``hmin``, and ``hmax``.
     - ``visual`` plus a configured site.
     - :download:`02_adaptivity_fields.ipynb <../../../examples/tutorials/04_meshing/02_adaptivity_fields.ipynb>`
   * - Gradings
     - :term:`Source <source grading>`, :term:`receiver <receiver grading>`,
       and model-:term:`surface grading` controls.
     - ``visual`` plus a configured site.
     - :download:`03_gradings.ipynb <../../../examples/tutorials/04_meshing/03_gradings.ipynb>`

Survey Tutorials
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 40 20 16

   * - Notebook
     - Focus
     - Requirements
     - File
   * - Receivers
     - Multi-component devices and :term:`dense survey` receiver groups.
     - ``visual`` plus a configured site.
     - :download:`01_receivers.ipynb <../../../examples/tutorials/05_surveys/01_receivers.ipynb>`
   * - DAS
     - Straight and helical DAS plus pointwise strain.
     - ``visual`` plus a configured site.
     - :download:`02_das.ipynb <../../../examples/tutorials/05_surveys/02_das.ipynb>`
   * - Sources
     - Scalar point sources, compound sources, and :term:`source batching`.
     - ``visual`` plus a configured site.
     - :download:`03_sources.ipynb <../../../examples/tutorials/05_surveys/03_sources.ipynb>`
   * - Sparse Surveys
     - Offset-domain and explicit :term:`sparse survey` source-receiver
       layouts.
     - ``visual`` plus a configured site.
     - :download:`04_sparse_surveys.ipynb <../../../examples/tutorials/05_surveys/04_sparse_surveys.ipynb>`

Output Tutorials
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 40 20 16

   * - Notebook
     - Focus
     - Requirements
     - File
   * - Traces
     - :term:`TraceDataset <trace dataset>`, :term:`HDF5` traces,
       :term:`xarray` reads, and :term:`SEG-Y` export.
     - ``seismic-io`` for :term:`SEG-Y` plus a configured site.
     - :download:`01_traces.ipynb <../../../examples/tutorials/06_outputs/01_traces.ipynb>`
   * - ParaView and VTK
     - Volume, surface, plane, field, property, source, and :term:`PML` output
       controls.
     - ``visual`` plus a configured site.
     - :download:`02_paraview_vtk.ipynb <../../../examples/tutorials/06_outputs/02_paraview_vtk.ipynb>`
   * - Imaging
     - :term:`RTM` imaging jobs, :term:`FWI`-gradient image requests, image
       grids, and ``ImageDatabase`` reads.
     - Configured site; cloud or HPC recommended for larger runs.
     - :download:`03_imaging.ipynb <../../../examples/tutorials/06_outputs/03_imaging.ipynb>`

Performance Tutorials
---------------------

.. list-table::
   :header-rows: 1
   :widths: 24 40 20 16

   * - Notebook
     - Focus
     - Requirements
     - File
   * - Performance
     - :term:`Frequency-domain` QC, :term:`time-domain` phase timings,
       :term:`source batching`, receiver sampling cost, and imaging assembly
       reuse.
     - Configured site; cloud or HPC recommended.
     - :download:`01_performance.ipynb <../../../examples/tutorials/07_performance/01_performance.ipynb>`

Related User Guide Pages
------------------------

- :doc:`../user_guide/physics_materials_boundaries`
- :doc:`../user_guide/api_to_contracts`
- :doc:`../user_guide/projects_jobs_sites`
- :doc:`../user_guide/site_configuration`
- :doc:`../user_guide/velocity_models_coordinates`
- :doc:`../user_guide/mesh_generation_adaptivity`
- :doc:`../user_guide/surveys_sources_receivers`
- :doc:`../user_guide/traces_paraview_outputs`
- :doc:`../glossary`
