# FrequenSolve Tests

This directory contains the test suite for the FrequenSolve Python project. The
default lane is deterministic and avoids solver binaries, cloud services,
schedulers, manual input, and visual baselines. Tests that need those resources
use explicit pytest markers.

1. **Unit Tests**: These tests can be run independently without requiring the
   full solver code. The `unit` marker is an affirmative classification used by
   contract-tool tests; unmarked deterministic tests remain part of the same
   default lane.

2. **Opt-in Marked Tests**: These tests require extra resources and are marked
   with `integration`, `cloud`, `hpc`, `interactive`, or `visual`. They test
   solver execution, external services, scheduler access, manual workflows, or
   image baselines.

Test files can contain both unit tests and opt-in marked tests.

## Quality Bar

Tests must exercise behavior or a real contract boundary. Good tests call the
public API, execute a parser or validator, round-trip an artifact, or run an
integration boundary with controlled dependencies. A pytest test that only
reads a tracked workflow, documentation, source, or style file and asserts that
literal strings are present does not meet this bar. Use the tool that owns the
format instead: Sphinx for documentation, `actionlint` for GitHub Actions,
pre-commit for repository policy, and a schema validator for structured
contracts.

Property-based tests are appropriate for stable invariants such as structured
solver-payload round trips. Keep strategies bounded and deterministic under the
normal Hypothesis profile so the PR-safe lane remains fast.

Marker names are strict. An unknown marker fails collection so a misspelled
`integration`, `cloud`, `hpc`, `interactive`, or `visual` marker cannot put a
resource-dependent test into the default lane.

Markers describe execution approval, not optional-package ownership. In
particular, `cloud` means a test contacts a real cloud boundary and
`interactive` means it needs input or an authenticated session. Credential-free
AWS adapter behavior is selected by the `cloud` package contract without either
approval marker. The `hpc` approval marker is reserved for live scheduler
checks. The thin real-cluster canary tracked by issue #83 is invoked manually
with `python -m frequensolve.orchestrator.sites.hpc.live_canary`; it is never
scheduled and is not part of required PR CI. It exercises public submission,
observation, fetch/load, and cancellation using the Deployment-generated
Enterprise profile. Deployment owns policy, cleanup, retention, performance,
provenance aggregation, and certification evidence.

## Writing Tests

### Test Framework

We use pytest as our testing framework. Some key features we use include:

- **Fixtures**: We make extensive use of pytest fixtures to create reusable test components. Fixtures can be used to set up test data, create test environments, or provide common test utilities.

- **Markers**: We use pytest markers to categorize tests. The default
  `python -m pytest` lane excludes `integration`, `cloud`, `hpc`,
  `interactive`, and `visual` tests.

### Writing Image Comparison Tests

We use pytest-mpl for comparing matplotlib-generated images. Here's a guide for writing image comparison tests:

1. **Test Structure**:
   - Use the `@pytest.mark.mpl_image_compare` decorator
   - Your test must return a single matplotlib figure that has not been closed
   - Example:
     ```python
     @pytest.mark.mpl_image_compare
     def test_my_plot():
         fig, ax = plt.subplots()
         plot_something(ax=ax)  # Note the ax parameter
         return fig
     ```

2. **Plotting Functions**:
   - Write plotting functions to accept an optional `ax` parameter
   - This allows tests to pass in an axis for plotting, making it easier to test
   - Example:
     ```python
     def plot_something(ax=None):
         if ax is None:
             fig, ax = plt.subplots()
         # ... plotting code ...
         return ax.figure if ax is not None else fig
     ```

3. **Reference Images**:
   - New image comparison tests will be skipped until reference images are generated
   - Reference images are stored in `tests/reference_images/`
   - Generate reference images using:
     - `make generate_reference_images` in this repo for non-integration,
       non-cloud, non-HPC visual tests
     - For solver-backed images, use the `frequensolve-test-evidence` artifact
       from an exact, pinned FrequenSolveDockerImage run. Inspect the generated
       scientific plot and record the run URL and FrequenSolve SHA in the PR
       that promotes it to `tests/reference_images/`.

4. **Test Output**:
   - Image comparison tests create output in `tests/output/`
   - Each test gets its own folder with comparison results
   - A `fig_comparison.html` report is generated
   - Open the HTML report in a browser to:
     - View generated images
     - Compare with reference images
     - See detailed differences

## Running Tests

### Using Makefile Targets

The project includes several Makefile targets for running tests:

- `make test`: Runs deterministic non-integration tests with line and branch
  coverage reporting
  - Generates XML and JSON coverage reports
  - Enforces the 64.5% combined, 69.0% line, and 51.8% branch ratchets
  - Skips integration, cloud, HPC, interactive, and visual tests

- `make validate-optional-extra-contracts`: Validates that
  `tests/optional-extra-contracts.json` covers every advertised runtime extra,
  preserves aliases, selects behavior tests, and matches the strict marker
  policy. Run it in the full test environment with `dev`, `parallel`, and
  `cloud` installed; required CI owns this validation.

- `make test-optional-extra-contract EXTRA=visual`: Runs one manifest-owned
  behavior and package-coverage contract in an environment where that extra is
  installed.

- `make test-optional-extras`: Convenience target for the local visual and
  seismic-IO contracts. Required CI derives all base and runtime-extra jobs from
  the manifest and installs each built distribution independently.

- `make generate_reference_images`: Generates reference images for matplotlib tests
  - Creates baseline images in `tests/reference_images/`
  - Skips integration, cloud, HPC, and interactive tests

## Continuous Integration

The project uses GitHub Actions for continuous integration. The
`.github/workflows/cicd-workflow.yml` workflow runs for `v2` pull requests and
pushes, and can also be started manually. It includes:

1. **Test Job**:
   - Runs on Python 3.10 through 3.14
   - Installs the package with pip using `.[dev,parallel,cloud]`
   - Runs pre-commit hooks once on Python 3.10, including `actionlint` against
     the GitHub Actions workflows
   - Executes the deterministic non-integration test lane, with coverage on
     Python 3.10
   - Preserves the Python 3.10 coverage report for a small downstream Codecov
     upload job; only that job receives OIDC permission, and upload failures
     fail the aggregate gate

2. **Optional Package Contract Matrix**:
   - Derives one Python 3.12 job per base/runtime-extra contract from
     `tests/optional-extra-contracts.json`
   - Resolves every direct requirement at its declared lower bound, installs
     the built wheel or source distribution in a clean runner, and executes an
     owning behavior path
   - Uploads per-contract coverage JSON and enforces reviewed package-level
     floors without contacting a solver, cloud service, or scheduler

3. **Native Platform Behavior**:
   - Pull requests that change package/runtime behavior install the exact built
     wheel on native `macos-15` and run the full deterministic suite against the
     installed package with the normal line, branch, and combined ratchets
   - The public-preview native `ubuntu-24.04-arm` contract is manual-only while
     runner reliability is measured; it installs only the base wheel and runs a
     representative path/process, serialization, units, geometry, validation,
     and result-loading suite
   - Both contracts fail on an unexpected OS/architecture or checkout-source
     coverage and retain exact commit/artifact, pip resolver, Python/platform,
     NumPy/BLAS, FFT, JUnit, and coverage evidence for 14 days

4. **Integration Test Job**:
   - Triggers the `FrequenSol/FrequenSolveDockerImage` CI workflow with the
     current FrequenSolve branch
   - Waits for that downstream workflow to finish
   - Downloads the downstream test artifacts into `tests/output/`
   - Requires the GitHub App secrets configured for the repository workflow

5. **Documentation Job**:
   - Builds the project documentation
   - Uploads documentation as an artifact

6. **Build And Package Smoke Jobs**:
   - Builds the Python package
   - Checks package metadata with `twine`
   - Uploads the `dist/` artifact
   - Installs and imports both the wheel and sdist on Ubuntu and macOS

7. **Required CI Job**:
   - Aggregates every PR-safe lane under the stable `Required CI` name used by
     branch rules and release evidence

## SDK Performance Evidence

`scripts/run_sdk_performance.py` measures Python-side SDK behavior only. Its
small/large scenarios cover acquisition serialization, job planning and
packing, packed trace access, validation, and result metadata loading. It does
not start the native solver, MPI, a cloud service, or a scheduler.

Pull requests run every scenario once as a correctness smoke check. Wall-time
thresholds are intentionally not enforced on that shared runner. The scheduled
and manual `SDK Performance Evidence` workflow uses `ubuntu-24.04`, Python
3.12, bounded warm-ups/samples, and a 15-minute job timeout. Every run retains
raw samples, variance statistics, peak Python heap, the exact repository
commit, dependency versions, CPU/architecture, and the complete runner image
identity.

To establish or replace `tests/performance/sdk-baseline.json`:

1. Merge harness changes before collecting baseline evidence so the repository
   identity is clean and exact.
2. Run the manual workflow at least twice with the default two warm-ups and
   seven samples. Confirm all ten scenarios ran, inspect raw values and
   coefficients of variation, and reject evidence collected after runner or
   dependency drift.
3. Set reviewed median wall-time and peak-Python-heap ceilings for every
   scenario. Keep the recorded comparison runner identity (all `runner` fields
   except the ephemeral `runnerName`) and exact third-party `dependencies`
   mapping. The measured FrequenSolve build remains bound by `repository.commit`
   instead of forcing a rebaseline after every dynamic package-version change.
4. Submit the baseline as a separate review. Scheduled/manual runs then pass it
   through `--baseline` and fail on missing scenarios, empty measurements,
   runner/dependency drift, or an exceeded ceiling.

Rebaseline only after explaining the regression or intentional improvement in
the reviewing pull request. Retain the old and new workflow artifact links,
exact commits, raw samples, variance, dependency delta, and runner delta so a
higher ceiling cannot be introduced silently.

## Important Notes

- While you can run tests directly in this repository for quick checks, it's
  recommended to run the full solver-backed suite through the
  FrequenSolveDockerImage workflow before releasing.
- Local `python -m pytest` and `make test` intentionally skip solver, cloud,
  HPC, interactive, and visual lanes unless you select those markers
  explicitly.
- Solver-backed integration tests are crucial for ensuring the solver works
  correctly in a production environment. The manual release-candidate workflow
  requires that exact-SHA evidence before it creates a tag.
