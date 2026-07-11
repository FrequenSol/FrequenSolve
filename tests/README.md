# FrequenSolve Tests

This directory contains the test suite for the FrequenSolve Python project. The
default lane is deterministic and avoids solver binaries, cloud services,
schedulers, manual input, and visual baselines. Tests that need those resources
use explicit pytest markers.

1. **Unit Tests**: These tests can be run independently without requiring the full solver code. They focus on testing individual components and functions in isolation.

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

Marker names are strict. An unknown marker fails collection so a misspelled
`integration`, `cloud`, `hpc`, `interactive`, or `visual` marker cannot put a
resource-dependent test into the default lane.

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
     - Recommended: Run the corresponding workflow in FrequenSolveDockerImage
       to generate solver-backed reference images

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

- `make test`: Runs deterministic non-integration tests with coverage reporting and matplotlib baseline testing
  - Generates XML coverage reports
  - Compares matplotlib figures against baseline images
  - Generates an HTML summary of matplotlib comparisons
  - Skips integration, cloud, HPC, interactive, and visual tests

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
   - Executes the deterministic non-integration test lane with `make test`
   - Uploads coverage reports to Codecov

2. **Integration Test Job**:
   - Triggers the `FrequenSol/FrequenSolveDockerImage` CI workflow with the
     current FrequenSolve branch
   - Waits for that downstream workflow to finish
   - Downloads the downstream test artifacts into `tests/output/`
   - Requires the GitHub App secrets configured for the repository workflow

3. **Documentation Job**:
   - Builds the project documentation
   - Uploads documentation as an artifact

4. **Build Job**:
   - Builds the Python package
   - Checks package metadata with `twine`
   - Uploads the `dist/` artifact

## Important Notes

- While you can run tests directly in this repository for quick checks, it's
  recommended to run the full solver-backed suite through the
  FrequenSolveDockerImage workflow before releasing.
- Local `python -m pytest` and `make test` intentionally skip solver, cloud,
  HPC, interactive, and visual lanes unless you select those markers
  explicitly.
- Solver-backed integration tests are crucial for ensuring the solver works
  correctly in a production environment and should be run before any release.
