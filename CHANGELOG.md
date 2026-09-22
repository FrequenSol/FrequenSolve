# Changelog

Notable changes to FrequenSolve should be documented here when a release is
prepared.

## Unreleased

- Removed the old imaging/FWI job layer in favour of `frequensolve.imaging`,
  which is now re-exported from the root namespace (`fs.ImagingProblem`,
  `fs.Misfit`, `fs.FWI`, ...). Deleted without replacement aliases:
  `frequensolve.simulation.jobs.imaging` (`ImagingJob`, `LSRTMGradientJob`,
  `LSRTMNormalJob` -> `ImageKernelJob`; `Misfit`, `MisfitComparison`,
  `MisfitGroup`, `PreprocessHook` -> `imaging.Misfit`/`imaging.Preprocess`;
  `HDF5TraceStore`, `ObservedTraceDerivatives` -> `imaging.TraceStoreRef`;
  `ImageDatabase` -> `imaging.ImageSet`), `frequensolve.simulation.jobs.fwi`
  (`FWIProblem`, `ModelSpace`, `DataSpace`, `FrequenSolveJacobian`,
  `ImageSpec`, `build_imaging_job` -> `imaging.ImagingProblem`,
  `imaging.ControlSpace`, `imaging.DataSpace`, `imaging.Jacobian`,
  `imaging.ImageSpec`), `frequensolve.simulation.jobs.control_sensitivity`
  (`ControlBlock`, `ControlSpace` -> `imaging.ControlSpace`;
  `RTMControlSensitivityJob`, `BornControlSensitivityJob`,
  `TimeReversalFocusJob` -> `imaging.ControlGradientJob`),
  `SeismicSimulation.fwi/imaging_job/imaging`, and
  `frequensolve.model.representation.VariationalSmoothing`
  (-> `imaging.SmoothingConfig`). Execution sites now drive image and
  gradient products through the artifact catalog roles (`image`, `gradient`,
  `objective`, `state`, `objective_vector`, `extension`) and the job
  postprocess protocol: `site.fetch_image` accepts any job with
  `load_images()`, `fetch_outputs` fetches every postprocess role, and
  `site.submit` honours a job's `postprocess_only` class attribute (set on
  `imaging.SmoothJob`) so smooth jobs run only the `--smooth` step. The
  SLURM sweep templates take `postprocess_job` instead of `imaging_job`.
  `frequensolve.inversion` least-squares adapters accept any control space
  with `size` and `pack` (`ControlSpaceLike`).

- Current development targets the `v2`/`v2_sam` line.
- Expanded the public authoring API with symbolic property expressions,
  coordinate-aware remapping, attenuation configuration, layered-model and
  borehole helpers, and VTK output builders.
- Added simulation studies that materialize Cartesian products or explicit
  cases from named parameter choices with configurable simulation name
  templates.
- Added generic `frequensolve.load(...)` dispatch and strengthened project,
  simulation, job, trace-store, relocation, and HDF5 lifecycle handling.
- Refined acquisition, imaging, trace, and output contracts to match the
  current Sauce solver schemas, including encoded sources and nested trace
  shards.
- Added data-driven local and Slurm execution profiles, secure reusable HPC
  credentials and transports, resumable task planning, and adaptive scheduling.
- Added release-evidence-backed preferred Solver metadata plus
  warn/strict/off identity checks for local and HPC execution sites.
- Renamed `frequensolve.simulation.numerics_manager` to
  `frequensolve.simulation.solver`; direct imports from the old module path must
  be updated.
- Added the `hpc` optional dependency group; `parallel` remains an equivalent
  compatibility alias.
- Public Python docs are published through the `FrequenSol/cloud-amplify` docs
  application instead of the removed `docs/host` Terraform stack.
- CI verifies Python 3.10 through 3.14.
- Versioned the customer Cloud readiness contract to `2.0.0`; readiness now
  reports authoritative Solver Compute Unit availability and usage instead of
  the retired legacy compute-credit balance.

## 0.0.1

- Initial tagged FrequenSolve Python API baseline.
