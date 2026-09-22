# Pinned Sauce contracts (imaging Phase 0)

Copied verbatim from `FrequenSol/Sauce@5e076241f6d52ed0c36b45911ff9bc8b12f1400a` (`imaging-api-phase0`, 2026-09-21): `trunk/contracts/fragments`, inputs fs-acquisition-1 fs-acquisition-2 fs-coordinate-system-1 fs-eikonal-1 fs-imaging-1 fs-implicit-geometry-1 fs-job-1 fs-material-model-1 fs-output-config-1 fs-simulation-1 fs-units-1 and the imaging output contracts. Refresh only when FrequenSolve intentionally adopts newer Sauce contracts; keep the SHA explicit.

Changes adopted relative to the previous `sauce-83c7f06` pin:

- fs-job-1: `control_sensitivities.input` (explicit vector for the `smooth` postprocess) and `fwi_operator.controls.min_support` / `support_measure`.
- fs-control-vector-1: `/support/<block>`, `/support_measure/<block>` and `/support_min_support` datasets on covector files.
- fs-material-model-1: LayeredModel `surfaces` may mix implicit-only entries with graph horizons (`examples/layered-rbf-control.json`); `tensor_hat` smoothing wording.

Refreshed 2026-09-21 to `5e076241` (adds the implicit-geometry length-scaling fix: all implicit surface lengths are in model units and nondimensionalized with the model; `blend` `width` reads a length).
