# Pinned Sauce input contracts

These JSON Schema files were initially copied from
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

The material property fixture additionally adopts the `parameterized` property
alternative and its parameterized/hat/B-spline/mesh control definitions from
`FrequenSol/Sauce@e1fd719cb58747022c9a14412adc592c3c13e060`, at the same
`trunk/contracts/inputs/fs-material-model-1/schema.json` path. This focused
extension verifies the public `ParameterizedProperty` serializer without
adopting unrelated newer material representations. The producer reader is
`trunk/src/Model/Fields/parameterized.sm.f90`.
