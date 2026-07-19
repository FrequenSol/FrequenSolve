Glossary
========

This glossary defines terms that appear throughout the FrequenSolve Python API documentation,
tutorials, user guide, and solver-facing reference pages.

Common concept terms are lower case. Acronyms, environment variables, product
names, file formats, protocols, package names, and proper names keep their
standard capitalization.

.. glossary::
   :sorted:

   ASDF
      Adaptable Seismic Data Format, an HDF5-based format used by parts of the
      seismic Python ecosystem for waveform and metadata exchange.

   attenuation
      Loss of wave energy during propagation. FrequenSolve material definitions
      can include attenuation through quality factors and related parameters.

   anisotropy
      Direction-dependent material behavior. The docs describe isotropic,
      vertical transverse isotropic, and tilted transverse isotropic material
      options.

   AWS
      Amazon Web Services. FrequenSolve cloud workflows use AWS-backed services
      for authentication, storage, and job execution.

   Biot
      Poroelastic theory used to model coupled solid-frame and pore-fluid wave
      propagation.

   borehole
      A well or subsurface path used to place sources, receivers, or geometry
      relative to a modeled volume.

   channel spacing
      Distance between adjacent receiver channels along a distributed acoustic
      sensing fiber.

   cloud site
      A configured site that submits jobs to FrequenSol Cloud instead of to a
      local solver executable or an HPC cluster.

   Cognito
      AWS identity service used by FrequenSolve cloud authentication flows.

   component
      A measured or simulated direction or quantity in a trace output, such as
      pressure or a velocity component.

   Dask
      Python parallel computing library used by local parallel execution paths.

   DAS
      Distributed acoustic sensing. In FrequenSolve, DAS receivers model
      fiber-style strain measurements.

   dense survey
      Acquisition layout that includes every source-receiver-component trace
      pair implied by the selected sources, receivers, and components.

   EPW
      Elements per wavelength. This mesh adaptivity target controls the number
      of elements used across the shortest wavelength.

   fast solver
      The separately licensed numerical solver executable that runs
      FrequenSolve jobs.

   finalizer
      Solver-side step that completes and indexes trace outputs so the FrequenSolve Python API can
      read the finalized trace files.

   FREQUENSOLVE_HOME
      Environment variable that changes the FrequenSolve user directory used for
      package-managed local state.

   FREQUENSOLVE_SITE_CONFIG
      Environment variable that points directly to the site configuration file
      the FrequenSolve Python API should read.

   frequency-domain
      Solver workflow for frequency-domain simulations, where responses are
      computed at selected frequencies.

   FrequenSolve user directory
      Local directory used by the FrequenSolve Python API for package-managed configuration and
      cached state. By default this is ``~/.frequensolve``.

   fracture
      A discontinuity or surface feature that can be represented in model or
      borehole geometry.

   FS_SOLVER_PATH
      Environment variable used by local execution paths to locate the fast
      solver executable.

   FWI
      Full waveform inversion workflow. FWI-gradient image requests produce
      property-gradient imaging outputs.

   gauge length
      Length interval over which a DAS channel measures strain.

   GMP
      FrequenSolve's generated mesh package format for externally supplied
      meshes.

   HDF5
      Hierarchical Data Format 5, used by FrequenSolve for array-heavy solver
      inputs and outputs.

   HPC
      High-performance computing. In these docs, HPC usually means a remote
      cluster accessed through SSH and scheduled with SLURM.

   HPC site
      A configured site that submits jobs to a remote HPC cluster.

   JKD
      Johnson-Koplik-Dashen dynamic permeability model for poroelastic
      simulations.

   job
      A runnable solver request created from a simulation, selected numerics,
      and requested outputs.

   JSON
      Text data format used by FrequenSolve for solver contracts, manifests,
      and saved project metadata.

   Laplace-domain
      Frequency-domain formulation that evaluates responses at complex-valued
      frequencies.

   local site
      A configured site that runs the solver on the same machine as the Python
      process.

   manifest
      Metadata file that describes generated solver inputs, run artifacts, or
      output files.

   mesh adaptivity
      Mesh generation behavior that refines or coarsens elements based on
      wavelength, geometry, source, receiver, or surface criteria.

   mesh block ID
      Integer identifier assigned to a mesh block or subdomain so material and
      boundary settings can target the correct region.

   optional extra
      Named optional dependency group installed with syntax such as
      ``frequensolve[visual]``.

   output request
      Job configuration entry that asks the solver to write a specific output,
      such as traces, ParaView files, or imaging products.

   packed trace file
      Trace output file that stores multiple trace records together for
      efficient transfer and reading.

   ParaView
      Interactive visualization application for inspecting large solver output
      files such as VTK/VTU meshes and fields.

   ParaView output
      Solver output intended for inspection in ParaView or compatible VTK
      readers.

   Pint
      `Python units library <https://pint.readthedocs.io/>`__ used by the FrequenSolve Python API
      to represent values with physical units.

   PML
      Perfectly matched layer. This absorbing boundary treatment truncates the
      computational domain while reducing artificial reflections.

   polynomial order
      Degree of the basis functions used by an element in the numerical
      discretization.

   poroelastic
      Material behavior that models coupled waves in a porous solid frame and
      pore fluid.

   project
      Saved workspace containing simulations, jobs, and generated solver inputs
      or outputs.

   P-wave
      Compressional body wave.

   PyPI
      Python Package Index, the default package registry used for installing
      released FrequenSolve packages.

   PyVista
      Python visualization library used for lightweight notebook figures and
      screenshots of VTK/VTU data.

   Python API
      The public Python classes, functions, and methods exposed by the FrequenSolve Python API.

   QC
      Quality control. In these docs, QC usually refers to quick checks that
      verify model setup, mesh quality, or output sanity.

   quality factor
      Dimensionless measure of attenuation. Higher values indicate lower
      damping.

   receiver device
      Physical or modeled receiver object that can produce one or more measured
      components.

   receiver grading
      Mesh refinement behavior near receivers.

   receiver group
      Named collection of receivers used to organize an acquisition layout.

   rerun fingerprint
      Identifier derived from run inputs and settings that helps decide whether
      an existing run artifact can be reused.

   Ricker wavelet
      Common zero-phase source wavelet used in seismic examples and tutorials.

   RTM
      Reverse time migration imaging workflow.

   run manifest
      Manifest written for a solver run. It records the run directory,
      generated inputs, requested outputs, and related metadata.

   run result
      Object returned after a submitted job completes or is inspected. In the
      FrequenSolve Python API this is represented by ``RunResult``.

   S3
      Amazon Simple Storage Service, used by cloud workflows for object storage.

   sample spacing
      Time or distance interval between adjacent samples in a signal, trace, or
      receiver layout.

   SEG-Y
      Standard seismic trace exchange format.

   shard
      One part of a larger output dataset split across multiple files or
      storage objects.

   simulation
      Model definition that combines geometry, materials, sources, receivers,
      boundaries, and numerics before it is turned into a job.

   site
      A configured execution backend for running jobs. Sites can target
      FrequenSol Cloud, a local solver installation, or an HPC cluster.

   site configuration file
      User-editable TOML file that defines named execution sites and the default
      site used by ``fs.Site()``.

   SLURM
      Cluster workload manager used by supported HPC execution sites.

   solver contract
      The JSON and HDF5 files exported by the FrequenSolve Python API and consumed by launchers and
      the fast solver.

   source batching
      Splitting sources into batches so large surveys can be run or stored in
      manageable pieces.

   source grading
      Mesh refinement behavior near sources.

   source geometry
      Physical source-point catalog exported in ``fs-acquisition-2``. Geometry
      is distinct from the right-hand-side fields derived from those points.

   source encoding
      Optional mapping from physical source points to solver source fields.
      Named encodings are sparse; JSON and HDF5 dense encodings store matrices.

   source field
      One addressable solver right-hand side. Without explicit source encoding,
      each physical source point is one identity field.

   source group
      Deprecated pre-``fs-acquisition-2`` logical-source representation. Use
      source geometry and source encoding for current exports.

   sparse survey
      Acquisition layout that selects a subset of source-receiver-component
      trace pairs instead of the full dense Cartesian product.

   SSH
      Secure Shell protocol used to connect to remote HPC systems.

   subdomain
      Named or identified region of a model or mesh that can receive different
      material, boundary, or output settings.

   surface grading
      Mesh refinement behavior near topography, interfaces, or other modeled
      surfaces.

   S-wave
      Shear body wave.

   Thomsen parameters
      Parameterization commonly used for weak elastic anisotropy.

   time-domain
      Solver workflow for time-domain simulations, where wavefields are evolved
      through time.

   TOML
      Configuration file format used for the FrequenSolve site configuration
      file.

   trace
      Time or frequency series recorded for a source, receiver, and component.

   trace group
      Logical grouping of trace outputs, often used to organize selected
      source-receiver-component combinations.

   trace dataset
      FrequenSolve Python API reader for HDF5-backed solver trace outputs. In the Python API this
      is represented by ``TraceDataset``. Trace reads return
      ``xarray.DataArray`` objects.

   TTI
      Tilted transverse isotropy, an anisotropic material model whose symmetry
      axis is tilted relative to the coordinate axes.

   upscaling
      Mapping fine-scale model properties onto a coarser solver grid or mesh.

   VTI
      Vertical transverse isotropy, an anisotropic material model with a
      vertical symmetry axis.

   VTK
      Visualization Toolkit file family used by ParaView and PyVista for mesh
      and field inspection.

   VTR
      VTK rectilinear-grid file format.

   VTU
      VTK unstructured-grid file format.

   xarray
      Python library for labeled arrays. FrequenSolve trace reads return
      ``xarray.DataArray`` objects.

   XDMF
      XML metadata format commonly paired with HDF5 arrays for visualization
      and data exchange.
