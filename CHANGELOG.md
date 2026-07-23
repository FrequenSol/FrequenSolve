# Changelog

Notable changes to FrequenSolve should be documented here when a release is
prepared.

## Unreleased

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
- Added release-evidence-backed preferred FrequenSolver metadata plus
  warn/strict/off identity checks for local and HPC execution sites.
- Renamed `frequensolve.simulation.numerics_manager` to
  `frequensolve.simulation.solver`; direct imports from the old module path must
  be updated.
- Added the `hpc` optional dependency group; `parallel` remains an equivalent
  compatibility alias.
- Public Python docs are published through the `FrequenSol/cloud-amplify` docs
  application instead of the removed `docs/host` Terraform stack.
- CI verifies Python 3.10 through 3.14.

## 0.0.1

- Initial tagged FrequenSolve Python API baseline.
