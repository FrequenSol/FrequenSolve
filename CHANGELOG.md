# Changelog

Notable changes to FrequenSolve should be documented here when a release is
prepared.

## Unreleased

- `NativeCurvature` deletes each operation's staged `input.h5` once its
  result is verified, unless the result has an HDF5 external link into it
  (warm-start factors link their unchanged `/modes`); a failed operation
  keeps its input. The optimizers pass the SDK's own objective,
  preconditioner and proximal adapters read-only views of their vectors and
  adopt the arrays they return (about 6 fewer model-sized allocations per
  L-BFGS iteration, 17 per Newton-CG iteration with 5 CG steps); user
  callables still receive private copies and their results are copied.
  With `scaling`, the physical-coordinate iteration records passed to
  callbacks rescale each vector on first access. Native proximal
  regularization rewrites a staged metric or bound file only when its content
  changes.
- `SeismicSimulation.save()` (called by every `job.save()`) skips rewriting
  the simulation JSON when the serialized content equals what the object last
  wrote and the file is unchanged since; inline `DepthProfile`/`GridParameters`
  coefficients made every job pay an O(controls) re-save. Remote staging
  copies one mapped simulation JSON per content and remote project
  (preserving its mtime, so rsync skips it) and scans input references
  without the coefficients. Job, task and compatibility fingerprints hash a
  saved JSON document once per file state.
- Added persistent SLURM allocation sessions with an allocation-resident
  scheduler, adaptive frequency workers, per-run cancellation and reconnectable
  run handles. Initialization, smoothing, packing and curvature share the same
  resource pool. `FWI.run(execution=PersistentAllocation(...))` owns a session
  through the complete inversion and releases it on exit; an open session may
  also be borrowed. Controller liveness is judged by an advancing heartbeat
  sequence on the client's monotonic clock (`heartbeat_timeout`, default the
  site's `scheduler_heartbeat_timeout`), never by comparing hosts' clocks;
  leases renew during uploads and tolerate clock skew. Owned sessions drain
  without `scancel` while the controller is alive (`cleanup_timeout=None` by
  default), and cleanup failures after successful work warn instead of
  discarding results. `FWI.run` saves the final result before releasing its
  allocation.
- `scheduler_heartbeat_timeout` now defaults to 300 seconds (was 60) for
  adaptive-scheduler batch runs and persistent sessions, and the starter site
  configuration writes 300: shared scratch filesystems can stall heartbeat
  writes for minutes. Explicitly configured values are honoured; a session
  whose window is below 120 seconds logs a warning when it starts.
- Persistent-session filesystem commands on the login node (session directory
  creation and the `mv`/`ln` that publish lease renewals, requests and
  commands) are bounded by the new `run_config.session_command_timeout`
  (default 120 s; also `site.session(session_command_timeout=...)` and
  `PersistentAllocation(session_command_timeout=...)`) instead of the 15 s
  solver identity-query bound, which failed sessions on slow work
  filesystems. A timeout or failure names the command and remote path instead
  of reporting "Enterprise HPC preflight could not verify operate allocation
  session".
- SLURM uploads and downloads retry stalls and SSH connection failures:
  `transfer_attempts` (site profile, default 3) attempts with a short backoff,
  after checking that the login connection still answers and switching to
  another live `~/.ssh/control` socket if it does not (the user's control
  master is never closed). `transfer_stall_timeout` (default 120 s) sets how
  long a transfer may make no progress; rsync now also passes it as
  `--timeout`, and rsync retries resume partial files. OpenSSH SFTP
  connection failures raise `ConnectionError` and control-socket status 255
  raises `SSHTransportError` (both subclasses of the previous `OSError` and
  `RuntimeError`).

- Optimizer iteration records (`InexactNewtonIteration`) share the accepted
  model, gradients and steps as read-only views instead of copying five
  vectors per iteration; callbacks must copy a vector before modifying it.
  `NativeCurvature` deletes a staged BFGS history once its operation finishes
  (`NativeCurvature.retain(history)` keeps it for several operations). Native
  regularization caches key the optimizer vector by a parallel block digest
  without forming the full model. `CurvatureTransfer.refresh` and
  `NativeCurvature.refresh_curvature` accept Sauce's `symmetry_tolerance`;
  refreshed factors must report `hessian_asymmetry`, which the transfer
  provenance records.
- FWI checkpoints (`fs-imaging-fwi-checkpoint-2`) keep L-BFGS restart arrays
  as float64 HDF5 files in a bounded `<stem>.restart` directory instead of
  JSON metadata, writing each secant pair once; `fs-imaging-fwi-checkpoint-1`
  checkpoints are rejected. `minimize_lbfgs` reports `LBFGSRestart` states
  (`fs-lbfgs-restart-2`) that reference the optimizer's arrays; dict restart
  states are no longer accepted. A stage interrupted during its end-of-stage
  curvature/uncertainty factorization is finished on resume, and each stage
  factorizes at most once. Fingerprints identify arrays with at least 4096
  elements by their bytes, so large-array fingerprints change once.
- `LSRTM` runs native `Tikhonov` with CG for every `method` (no proximal
  jobs, no gradient job at the zero image, no final residual product). TV/TGV
  proximal iterations start from `1/lambda_max` (two power iterations) and
  stop relative to the initial proximal-gradient mapping, so they no longer
  depend on data or image units. CG is an in-house safeguarded PCG: a non-SPD
  preconditioner raises, nonpositive curvature stops with `status == -1`, and
  convergence on the last iteration counts. `info["jobs"]` and
  `info["background"]` report job counts and checkpoint reuse/misses.
  Background reuse also requires `site.supports_background_reuse`; the run
  deletes the checkpoint it created unless `keep_background=True`, evicted
  linearizations delete theirs, and `LinearizationCache.background_budget`
  bounds retained checkpoint bytes. `minimize_proximal_gradient` gains
  `initial_step`, `relative_tolerance` and `curvature_steps`. FWI requests
  receiver probes for `Diagonal(probes="receiver")`; patch FWI rejects them.
- Deferred the experimental HV and AWI objectives; removed their Python
  configurations and `ImagingProblem` adapters.
- Receiver linearization, JVP and VJP cover multiple groups sharing the same
  PDE solves; group adjoint loads are combined before solving.

- Added frequency-coherent time-reversal focusing objectives:
  `imaging.Focusing` (Gaussian lag window), `ImagingProblem.focus(...)` /
  `imaging.FocusingProblem` (a problem view `FWI` accepts, objective
  `1 - mean focusing ratio`) and `imaging.SourceAperture` for spatial softening
  with `strategy="linear"` (extended source) or `"pointwise"` (WEFT-style node
  energies). `Linearization` gains `simulated()`, `observed()`,
  `residual_sign`, `modeled_vjp(g, per_task=...)` and `vjp_tasks(r)`.
- Removed `TimeReversalFocus` and `ControlGradientJob(kind="focus")`: Sauce no
  longer provides the per-frequency `focus` workflow, whose energies were not
  focusing objectives. Use `imaging.Focusing`.

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
