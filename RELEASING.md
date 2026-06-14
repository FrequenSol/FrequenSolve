# Releasing

FrequenSolve releases use the standard PyPA build tools and trusted
publishing through GitHub Actions.

## Prerequisites

- A `v`-prefixed PEP 440 version tag for the release. Release candidate tags
  look like `v0.2.0rc1`; final release tags look like `v0.2.0`.
- FrequenSolve intentionally does not use a checked-in `VERSION` file.
  Versioneer reads the package version from the exact Git tag being built, with
  the leading `v` removed by `tag_prefix = "v"` in `pyproject.toml`.
- The `Create Release Candidate` workflow for creating the next release
  candidate tag and GitHub prerelease.
- The `Create Release` workflow for promoting a release candidate tag to the
  final release tag.
- The `Publish Package` workflow in `.github/workflows/release.yml` for
  building distributions and publishing to the selected package index.
- A PyPI project owner must configure trusted publishing for the package
  indexes and workflows that will be used:

| Index | Repository | Workflow | Environment |
| --- | --- | --- | --- |
| TestPyPI | `FrequenSol/FrequenSolve` | `release.yml` | `testpypi` |
| PyPI | `FrequenSol/FrequenSolve` | `release.yml` | `pypi` |

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

Use the release workflows for the normal maintainer flow:

- Run `Create Release Candidate` with a final base version such as `0.2.0`.
  The workflow creates the next tag in that release line, such as `v0.2.0rc1`,
  and publishes a GitHub prerelease. The prerelease event triggers
  `Publish Package`, which builds the package, attaches the distributions to
  the GitHub Release, and publishes `0.2.0rc1` to TestPyPI.
- Run `Create Release` with an approved release candidate tag such as
  `v0.2.0rc1`. The workflow creates `v0.2.0` on the same commit and publishes a
  final GitHub Release. The release event triggers `Publish Package`, which
  rebuilds from the final tag, attaches the distributions to the GitHub Release,
  and publishes `0.2.0` to PyPI.

`Publish Package` also supports manual `workflow_dispatch` from a selected tag
for maintainers who need to retry publishing intentionally. The workflow must be
run with `--ref` set to the release tag, not a branch, so Versioneer can resolve
the clean package version from that tag. Choose `repository=testpypi` for
release candidates and `repository=pypi` for final releases.

Retry a release-candidate publish:

```bash
gh workflow run release.yml \
  --repo FrequenSol/FrequenSolve \
  --ref v0.2.0rc1 \
  -f repository=testpypi
```

Retry a final release publish:

```bash
gh workflow run release.yml \
  --repo FrequenSol/FrequenSolve \
  --ref v0.2.0 \
  -f repository=pypi
```

`Publish Package` runs `scripts/validate_release_version.py` before publishing.
The validator requires a tag ref named `v<Versioneer version>` and rejects
dirty, untagged, branch-derived, local-version, or non-PEP-440 builds such as
`0.0.1+278.gccbbd6f` or `0.2.0-rc.1`.

## Installing A TestPyPI Release Candidate

Release candidates published to TestPyPI are not visible to normal PyPI
installs. To install one, explicitly select TestPyPI for the FrequenSolve
package and use PyPI as the secondary index for dependencies:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  frequensolve==0.2.0rc1
```

You can also put the same index configuration in a temporary requirements file:

```text
--index-url https://test.pypi.org/simple/
--extra-index-url https://pypi.org/simple/

frequensolve==0.2.0rc1
```

Then install it with:

```bash
python -m pip install -r requirements-testpypi.txt
```

Do not add TestPyPI as a normal global pip fallback. `--extra-index-url` mixes
candidate packages from multiple indexes, and pip chooses the best matching
version rather than treating the extra index as a strict fallback. Keep TestPyPI
usage explicit and temporary for release validation.
