# Releasing

FrequenSolve releases use the standard PyPA build tools and the repository's
`Publish PyPI` GitHub Actions workflow.

## Prerequisites

- A version tag for the release.
- The `Publish PyPI` workflow in `.github/workflows/publish-pypi.yml`.
- A PyPI project owner must configure trusted publishing for:
  - repository: `FrequenSol/FrequenSolve`
  - workflow: `publish-pypi.yml`
  - environment: `pypi`

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

## Publishing

The `Publish PyPI` workflow publishes with PyPI trusted publishing on release
publication or manual dispatch. The workflow only publishes from tag refs,
including manual dispatches, so create and select the intended release tag
before publishing.
