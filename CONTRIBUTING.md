# Contributing

FrequenSolve is the Python API for authoring and running FrequenSol
simulations, with optional local, cloud, HPC, visual, and docs dependencies.
Keep changes small, test the relevant marked lanes explicitly, and keep public
examples aligned with the current project-owned simulation API.

## Development Setup

FrequenSolve supports Python 3.10 through 3.14. For local development, create a
virtual environment from the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,parallel,cloud,visual]"
```

Use the `docs` extra when you only need documentation dependencies:

```bash
python -m pip install -e ".[docs]"
```

## Code Style

- Use type hints for public function arguments and return values.
- Document public classes and functions with Google-style docstrings.
- Run formatting and lint hooks before opening a pull request:

```bash
pre-commit run --all-files
```

The first strict typing ratchet covers the public optional-dependency loading
boundary:

```bash
make typecheck
```

Keep that configured file clean, and expand `[tool.mypy].files` one coherent
module boundary at a time. Do not weaken the existing checks to admit a new
module; fix or narrowly document its baseline before adding it.

## Testing

Run the deterministic non-integration lane used by CI:

```bash
make test
```

This enforces the current coverage ratchet: at least 64.5% combined coverage,
69.0% line coverage, and 51.8% branch coverage (the measured branch baseline is
51.876%, conventionally shown as 52%).

The default `pytest` configuration also excludes tests that require solvers,
external services, credentials, schedulers, manual input, or visual baselines:

```bash
python -m pytest
```

Tests must demonstrate observable behavior or validate a structured contract.
Do not add pytest checks that only read tracked files and search for expected
strings. Workflow syntax and expressions are validated by the `actionlint`
pre-commit hook; documentation is validated by the Sphinx build.

Core serialization invariants use Hypothesis to exercise round trips across a
wide input range. Prefer focused properties such as lossless structured
round-tripping over examples that merely repeat implementation details.

Select marked lanes explicitly when you have the required environment:

```bash
python -m pytest -m integration
python -m pytest -m cloud
python -m pytest -m hpc
python -m pytest -m visual
```

Regenerate visual baselines only when the rendered output is intentionally
changed:

```bash
make generate_reference_images
```

With the `parallel`, `visual`, and `seismic-io` extras installed, run the
PR-safe optional lane that CI uses:

```bash
make test-optional-extras
```

## Pull Request Process

1. Create a focused branch for the change.
2. Add or update tests for behavior changes.
3. Update docs and examples when public API or workflow guidance changes.
4. Run `make test` and any affected marked lanes locally.
5. Open a pull request. The CI matrix in
   `.github/workflows/cicd-workflow.yml` runs for `v2` pull requests and
   pushes. Protect `v2` with the stable `Required CI` job; for maintenance
   branches, run the same local checks before handoff or ask a maintainer to
   retarget or manually dispatch CI.

Release and deployment workflows are handled by GitHub Actions. Do not add PyPI
tokens, cloud credentials, or solver licenses to the repository. See
[RELEASING.md](RELEASING.md) for maintainer release steps.

## Documentation

Build the documentation locally:

```bash
cd docs
make html
```

Published Python docs are staged by the `FrequenSol/cloud-amplify`
`docs-site-app` through its manual `Publish Python Docs` workflow. The old
`docs/host` Terraform project has been removed from this repository.
