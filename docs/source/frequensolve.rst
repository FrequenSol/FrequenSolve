FrequenSolve
============

This generated reference is organized by Python module. If you are looking for
a class by task, start here:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Task
     - Reference area
   * - Create or load projects
     - :doc:`frequensolve.project`
   * - Inspect version-matched simulation knowledge
     - :doc:`frequensolve.knowledge`
   * - Build simulations and request outputs
     - :doc:`frequensolve.simulation`
   * - Define materials, layers, surfaces, and boreholes
     - :doc:`frequensolve.model`
   * - Configure meshes and boundary conditions
     - :doc:`frequensolve.mesh`
   * - Define sources, receivers, surveys, traces, and wavelets
     - :doc:`frequensolve.seismic`
   * - Create jobs and configure local, cloud, or HPC execution
     - :doc:`frequensolve.orchestrator`
   * - Plot traces, layered models, and VTK/VTU files
     - :doc:`frequensolve.plotting`
   * - Work with coordinate frames
     - :doc:`frequensolve.geometry`
   * - Work with units and exported array stores
     - :doc:`frequensolve.util`
   * - Validate simulations and jobs before export
     - :doc:`frequensolve.validation`
   * - Declare an imaging problem: control spaces, observed data, misfit
     - :doc:`frequensolve.imaging` (``ControlSpace``, ``DepthProfile``,
       ``GridParameters``, ``ObservedData``, ``Misfit``, ``ImagingProblem``)
   * - Run FWI, LSRTM, RTM, sensitivity kernels, or focusing
     - :doc:`frequensolve.imaging` (``FWI``, ``Stage``, ``LSRTM``, ``rtm``,
       ``sensitivity_kernel``, ``TimeReversalFocus``)
   * - Work with Jacobian and normal operators, penalties, smoothing
     - :doc:`frequensolve.imaging` (``Jacobian``, ``Normal``, ``Tikhonov``,
       ``TV``, ``Smoothing``, ``Diagonal``)
   * - Model extension (FWIME) and reflectivity
     - :doc:`frequensolve.imaging` (``Extension``, ``Lags``, ``HalfOffsets``,
       ``ReflectivityParameters``)
   * - Author low-level Sauce imaging jobs and read their artifacts
     - :doc:`frequensolve.imaging` (``FWIOperatorJob``, ``ControlGradientJob``,
       ``ImageKernelJob``, ``SmoothJob``, ``ImageSet``)
   * - Generic optimizers, continuation, history, derivative checks
     - :doc:`frequensolve.inversion`
   * - Parameterized properties and implicit surfaces
     - :doc:`frequensolve.model` (``model.parameterization``,
       ``model.implicit_geometry``)

The conceptual user guide explains how these pieces fit together. Use this
reference when you need constructor arguments, method names, or class members.

.. toctree::
   :maxdepth: 4

   frequensolve.geometry
   frequensolve.imaging
   frequensolve.inversion
   frequensolve.knowledge
   frequensolve.mesh
   frequensolve.model
   frequensolve.orchestrator
   frequensolve.plotting
   frequensolve.project
   frequensolve.seismic
   frequensolve.simulation
   frequensolve.util
   frequensolve.validation
