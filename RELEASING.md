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

The release-candidate workflow also requires access to the organization-owned
FrequenSolveDockerImage reusable workflow and the FrequenSolver Builder GitHub
App secret. Its workflow reference is pinned to a reviewed DockerImage commit;
do not replace that pin with a mutable branch or tag.

## Local Checks

Run packaging checks before publishing:

```bash
python -m build
python -m twine check dist/*
```

For a full release candidate, the exact source commit must already have a
successful `Required CI` job. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
local equivalents.

## Publishing Paths

Use the release workflows for the normal maintainer flow:

- Run `Create Release Candidate` with a final base version such as `0.2.0`.
  The workflow first resolves the selected ref to an immutable SHA, verifies a
  successful exact-SHA `Required CI` run, and calls the pinned
  FrequenSolveDockerImage workflow with that SHA. Only after the solver-backed
  workflow returns the expected passing marker, commit, and artifact does it
  create the next tag in that release line, such as `v0.2.0rc1`, and publish a
  GitHub prerelease. The prerelease includes `release-evidence.json` plus the
  checksum-bound `frequensolve-test-evidence.tar.gz` archive containing the
  heavy-run JSON, JUnit, branch coverage, and visual comparison output. Its
  event triggers
  `Publish Package`, which builds the package, attaches the distributions to
  the GitHub Release, and publishes `0.2.0rc1` to TestPyPI.
- Run `Create Release` with an approved release candidate tag such as
  `v0.2.0rc1`. The workflow revalidates the attached exact-SHA CI and solver
  evidence, creates `v0.2.0` on the same commit, and carries both evidence
  assets into the final GitHub Release. The release event triggers `Publish Package`,
  which validates the evidence again, rebuilds from the final tag, attaches the
  distributions to the GitHub Release, and publishes `0.2.0` to PyPI.

`Publish Package` also supports manual `workflow_dispatch` for maintainers who
need to retry publishing intentionally. Set the required `release_tag` input to
the immutable release tag; the workflow checks out that exact tag and reads its
matching release evidence before building. Keep `--ref` on the same tag so the
reviewed workflow definition and the package source are aligned. Choose
`repository=testpypi` for release candidates and `repository=pypi` for final
releases.

Retry a release-candidate publish:

```bash
gh workflow run release.yml \
  --repo FrequenSol/FrequenSolve \
  --ref v0.2.0rc1 \
  -f release_tag=v0.2.0rc1 \
  -f repository=testpypi
```

Retry a final release publish:

```bash
gh workflow run release.yml \
  --repo FrequenSol/FrequenSolve \
  --ref v0.2.0 \
  -f release_tag=v0.2.0 \
  -f repository=pypi
```

`Publish Package` requires both matching evidence assets, verifies the archive
checksum and machine-readable heavy results, and reruns the exact-SHA CI
validation before it builds or publishes. It also runs
`scripts/validate_release_version.py`. The version validator requires a tag ref
named `v<Versioneer version>` and rejects
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
