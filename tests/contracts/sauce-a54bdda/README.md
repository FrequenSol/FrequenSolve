# Pinned Sauce input contracts

These JSON Schema files are copied verbatim from
`FrequenSol/Sauce@a54bdda81c98780fb4b805b92cf6df6c6e8bd29a` (`origin/main` on
2026-07-18). They provide offline consumer-contract fixtures for FrequenSolve
simulation and acquisition-v2 tests.

Source paths:

- `trunk/contracts/inputs/fs-simulation-1/schema.json`
- `trunk/contracts/inputs/fs-material-model-1/schema.json`
- `trunk/contracts/inputs/fs-output-config-1/schema.json`
- `trunk/contracts/inputs/fs-units-1/schema.json`
- `trunk/contracts/inputs/fs-acquisition-2/schema.json`
- `trunk/contracts/inputs/fs-coordinate-system-1/schema.json`
- `trunk/contracts/inputs/fs-acquisition-1/schema.json`
- `trunk/contracts/fragments/fs-common-defs.schema.json`

Refresh these fixtures only when FrequenSolve intentionally adopts newer Sauce
contracts. Keep the commit SHA and copied paths explicit so test results are
traceable to the accepted consumer schemas.
