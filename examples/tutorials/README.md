# FrequenSolve Tutorials

These notebooks are the release tutorial set for the FrequenSolve Python API.
They are meant to be run from this repository, linked from the Sphinx
documentation, and read directly by users who want complete examples.

The notebooks use a consistent public API style:

- `import frequensolve as fs`
- project-owned simulations from `project.new_simulation(...)`
- layered models with `properties={...}`
- explicit `TimeDomainJob` and `FrequencyDomainJob` objects
- job-owned outputs, including ParaView/VTK requests
- trace reads through `TraceDataset`

Each notebook should read as a short modeling lesson:

1. State the physical or workflow goal in plain language.
2. Define the relevant vocabulary before using it in code.
3. Build a small model that can be visually inspected.
4. Run strict jobs without hiding solver failures.
5. Plot or list results and explain what the reader should look for.
6. Connect the Python object back to the exported solver contract when that
   helps users debug or scale the workflow.

The intended user path is incremental. Section 1 teaches the core model/job/site
workflow. Later sections reuse the same pattern while adding one new idea at a
time: remote execution, coordinate systems, mesh controls, survey layouts, and
output inspection. Avoid introducing a new API shape when an earlier tutorial
pattern will do.

Read the collection as one story:

1. Modeling basics introduce the project/simulation/job/result loop.
2. Site tutorials move the same jobs between local, cloud, and HPC execution.
3. Velocity-model tutorials make units, coordinates, and geometry explicit.
4. Meshing tutorials show how geometry and acquisition guide adaptive meshes.
5. Survey tutorials teach what the solver records and how traces are selected.
6. Output tutorials make trace, ParaView, and imaging artifacts reusable after a run.
7. Performance tutorials teach frequency-domain QC, source batching, timing
   diagnostics, and scaling habits for production runs.

Every notebook should answer the same reader questions: What am I modeling?
Which object owns this decision? What file or result proves it worked? What
should I inspect before scaling this pattern to a larger job?

Solver execution cells are intentionally strict. If the local, cloud, or HPC
solver fails, the notebook should fail at the run cell so the job logs and
result directory remain visible. Tutorial code should fix Python/API errors,
not hide solver-side failures behind broad exception handlers.

## Layout

| Folder | Focus |
| --- | --- |
| `01_modeling_basics` | Acoustic, elastic, poroelastic, coupled, dimensionality, Laplace/time-domain, and axisymmetric borehole basics. |
| `02_sites` | AWS, HPC, local execution, and saved project/job loading. |
| `03_velocity_model_building` | Units, coordinate systems, surface-relative properties, and layered model tools. |
| `04_meshing` | Mesh generators, supplied meshes, adaptivity fields, and gradings. |
| `05_surveys` | Receiver devices, DAS, source mechanisms, batching, and sparse surveys. |
| `06_outputs` | Trace HDF5 access, `xarray` reads, ParaView/VTK output, imaging output, and PyVista screenshots. |
| `07_performance` | Frequency-domain QC, time-domain timing diagnostics, source batching, receiver sampling, and imaging cost. |

## Running

Most notebooks write scratch projects under `./scratch/tutorials/...` relative
to the notebook working directory. Local examples require a working fast solver
installation. Site examples require the corresponding AWS, HPC, or local site
configuration.

The documentation build links these notebooks as downloadable files; Sphinx does
not execute or render notebook cells.
