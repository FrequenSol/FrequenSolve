# Releasing

FrequenSolve releases use the standard PyPA build tools and trusted
publishing through GitHub Actions.

## Prerequisites

- A version tag for the release.
- The `Release` workflow in `.github/workflows/release.yml` for release
  candidates, TestPyPI publishes, and published GitHub releases.
- The `Publish PyPI` workflow in `.github/workflows/publish-pypi.yml` for the
  PyPI-only tag-ref publish path.
- A PyPI project owner must configure trusted publishing for the package
  indexes and workflows that will be used:

| Index | Repository | Workflow | Environment |
| --- | --- | --- | --- |
| TestPyPI | `FrequenSol/FrequenSolve` | `release.yml` | `testpypi` |
| PyPI | `FrequenSol/FrequenSolve` | `release.yml` | `pypi` |
| PyPI | `FrequenSol/FrequenSolve` | `publish-pypi.yml` | `pypi` |

Do not add PyPI passwords or API tokens to the repository.

## Local Checks

Run packaging checks before publishing:

```bash
python -m build
python -m twine check dist/*
```

For a full release candidate, also run the contributor verification lane that is
relevant to the changed surface. See [CONTRIBUTING.md](CONTRIBUTING.md) for test
commands.

## Publishing Paths

Use the `Release` workflow for the normal maintainer flow:

- `workflow_dispatch` with `repository=testpypi` publishes a tagged build to
  TestPyPI.
- Publishing a GitHub Release publishes the tagged build to PyPI.
- `workflow_dispatch` with `repository=pypi` is available for a manual PyPI
  publish from the selected tag when a maintainer intentionally chooses that
  path.

The `Publish PyPI` workflow is a narrower PyPI-only path. It publishes with
PyPI trusted publishing on release publication or manual dispatch, but it
rejects non-tag refs. Create and select the intended version tag before using
either PyPI publishing path.
