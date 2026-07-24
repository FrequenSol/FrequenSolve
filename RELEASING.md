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
- An immutable, published final FrequenSolver GitHub release (`vX.Y.Z`) chosen
  for the package release. Drafts, prereleases, branches, and mutable refs are
  not accepted as the preferred runtime.
- A PyPI project owner must configure trusted publishing for the package
  indexes and workflows that will be used:

| Index | Repository | Workflow | Environment |
| --- | --- | --- | --- |
| TestPyPI | `FrequenSol/FrequenSolve` | `release.yml` | `testpypi` |
| PyPI | `FrequenSol/FrequenSolve` | `release.yml` | `pypi` |

Do not add PyPI passwords or API tokens to the repository.

The default `standard` validation profile does not dispatch a Docker build.
The optional `solver-backed` profile also requires the FrequenSolver Builder
GitHub App secret so it can dispatch the private organization-owned
FrequenSolveDockerImage workflow. FrequenSolve is public, so GitHub does not
permit a direct reusable-workflow call into that private repository. The
dispatcher requires DockerImage `main` to resolve to the reviewed commit before
it starts, passes exact dependency commits with a high-entropy request ID, and
rejects duplicate or returned runs whose identity differs from that pin.
Long-running builds are polled through bounded windows with fresh read-only App
tokens; the dispatch token has write access only to DockerImage.

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

- Run `Create Release Candidate` with a final base version such as `0.2.0` and
  the final `frequensolver_release` tag selected for that package line. Choose
  one explicit validation profile:

  - `standard` (the default) requires successful exact-tree `Required CI` for
    the immutable source commit and records the immutable preferred
    FrequenSolver release and commit. It does not dispatch Docker or claim that
    the solver pairing was exercised. Its GitHub prerelease contains exactly
    one release evidence asset, `release-evidence.json`, and no heavy archive.
  - `solver-backed` requires the same exact-tree CI and immutable
    FrequenSolver identity, then API-dispatches the pinned private
    FrequenSolveDockerImage workflow with the exact FrequenSolve, Sauce, and
    FS_MUMPS commits. The returned evidence must bind the request, workflow,
    commits, no-push mode, and passing heavy-test contract. Its GitHub
    prerelease contains `release-evidence.json` plus one checksum-bound
    `frequensolve-test-evidence.tar.gz` archive.

  After the selected profile passes, the workflow creates the next tag in the
  release line, such as `v0.2.0rc1`, and publishes a GitHub prerelease. That
  event triggers `Publish Package`, which builds the package, attaches the
  distributions, and publishes `0.2.0rc1` to TestPyPI.
- Run `Create Release` with an approved release candidate tag such as
  `v0.2.0rc1`. The workflow derives the validation profile only from the sealed
  evidence, revalidates its exact-SHA CI and profile-specific asset set, creates
  `v0.2.0` on the same commit, and carries that asset set unchanged into the
  final GitHub Release. The release event triggers `Publish Package`, which
  validates the evidence again, rebuilds from the final tag, attaches the
  distributions to the GitHub Release, and publishes `0.2.0` to PyPI.

Release creation treats the profile-specific assets as one sealed set. A retry
may replace the complete set while the GitHub Release is still a draft. A
published release is reused only when its assets exactly match the newly sealed
set. Standard releases require exactly one `release-evidence.json` and zero
heavy archives; solver-backed releases require exactly one of each. A stale,
extra, or incomplete published set fails before release notes or assets are
changed. Legacy v2 evidence remains readable as solver-backed evidence.

`Publish Package` also supports manual `workflow_dispatch` for maintainers who
need to retry publishing intentionally. Set the required `release_tag` input to
the immutable release tag; the workflow checks out that exact tag and reads its
matching release evidence before building. Keep `--ref` on the same tag so the
reviewed workflow definition and the package source are aligned. Choose
the publication target by publishing the matching GitHub Release: prereleases
route to TestPyPI and final releases route to PyPI. The retry workflow derives
that target from the release metadata; it has no separate repository input.

Retry a release-candidate publish:

```bash
gh workflow run release.yml \
  --repo FrequenSol/FrequenSolve \
  --ref v0.2.0rc1 \
  -f release_tag=v0.2.0rc1
```

Retry a final release publish:

```bash
gh workflow run release.yml \
  --repo FrequenSol/FrequenSolve \
  --ref v0.2.0 \
  -f release_tag=v0.2.0
```

`Publish Package` requires the exact profile-specific evidence asset set and
reruns exact-SHA CI validation before it builds or publishes. For
solver-backed evidence it also verifies the archive checksum and
machine-readable heavy results. It materializes
`frequensolver_compatibility.json` from that sealed evidence in a git-free
staging tree, so the wheel and sdist carry the preferred immutable
FrequenSolver release, commit, validation profile, and evidence URL without
making Versioneer report a dirty version. It also runs
`scripts/validate_release_version.py`. The version validator requires a tag ref
named `v<Versioneer version>` and rejects
dirty, untagged, branch-derived, local-version, or non-PEP-440 builds such as
`0.0.1+278.gccbbd6f` or `0.2.0-rc.1`.

## Runtime FrequenSolver Compatibility

`LocalSite` and `SlurmSite` query the configured executable directly with
`--identity-json` once before the first submission. `frequensolve site check`
performs the same check for an SSH-backed profile. No query or warning occurs
when the Python package is imported, and individual solver tasks do not repeat
the check. Remote identity probes time out after 15 seconds so a legacy solver
cannot leave site validation or submission blocked indefinitely.

The `frequensolver_policy` site setting supports:

- `warn` (default): continue but warn when the identity is unavailable or the
  release/commit differs from the preferred pair, or when a standard release
  did not run solver-backed validation.
- `strict`: refuse submission unless solver-backed evidence confirms the exact
  release and commit. An identity match from a standard release remains
  `untested` and is rejected.
- `off`: skip the identity query explicitly.

The environment variable `FREQUENSOLVE_FREQUENSOLVER_POLICY` supplies the
policy when a site does not set one. Different versions are described as
untested rather than necessarily incompatible.

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
