# FrequenSolve Tests

This directory contains the test suite for the FrequenSolve Python project. The tests are organized into two categories:

1. **Unit Tests**: These tests can be run independently without requiring the full solver code. They focus on testing individual components and functions in isolation.

2. **Integration Tests**: These tests require the full solver code to be present and are marked with the `integration` pytest marker. They test the interaction between different components and the solver's functionality as a whole.

Test files can contain both unit tests and integration tests.

## Writing Tests

### Test Framework

We use pytest as our testing framework. Some key features we use include:

- **Fixtures**: We make extensive use of pytest fixtures to create reusable test components. Fixtures can be used to set up test data, create test environments, or provide common test utilities.

- **Markers**: We use pytest markers to categorize tests. The `integration` marker is used to identify integration tests.

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
     - `make generate_reference_images` in this repo (non-integration tests only)
     - Recommended: Run same command in FrequenSolveDockerImage to generate all reference images (including for integration tests)

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

- `make test`: Runs all non-integration tests with coverage reporting and matplotlib baseline testing
  - Generates XML coverage reports
  - Compares matplotlib figures against baseline images
  - Generates an HTML summary of matplotlib comparisons
  - Skips all integration tests

- `make generate_reference_images`: Generates reference images for matplotlib tests
  - Creates baseline images in `tests/reference_images/`
  - Skips all integration tests (doesn't generate images for integration tests either)

### Using Cursor for Test Analysis

When working with tests in Cursor, you can leverage AI assistance to analyze test failures and prioritize fixes:

1. **Running Tests in Docker**:
   - Run the full test suite in FrequenSolveDockerImage
   - Copy the terminal output into Cursor chat
   - Ask Cursor to analyze the failures

2. **Failure Analysis**:
   - Ask Cursor to rate each failure by probability of being caused by source code issues
   - Example prompt: "Rate each of these failures by probability that the failure is caused by an underlying source code issue. List in descending order so I can address issues that are most urgent."
   - Cursor will help prioritize which failures to investigate first

3. **Using Analysis Results**:
   - Focus on high-probability source code issues first
   - Use Cursor to help investigate specific failures
   - Get suggestions for potential fixes
   - Verify fixes by running tests again

## Continuous Integration

The project uses GitHub Actions for continuous integration. The workflow (`cicd-workflow.yml`) includes:

1. **Test Job**:
   - Runs on Python 3.10
   - Installs dependencies using Poetry
   - Runs pre-commit hooks
   - Executes all **non-integration** tests
   - Uploads coverage reports to Codecov (future capability)

2. **Documentation Job**:
   - Builds the project documentation
   - Uploads documentation as an artifact

3. **Deploy Documentation Job**:
   - Deploys documentation to AWS (only on main branch pushes)

4. **Build Job**:
   - Builds the Python package (only on main branch pushes)

## Important Notes

- While you can run tests directly in this repository for quick checks, it's recommended to run the full test suite (including integration tests) from the FrequenSolveDockerImage repository before releasing.
- The CI pipeline only runs non-integration tests since integration tests are run in the FrequenSolveDockerImage repo.
- Integration tests are crucial for ensuring the solver works correctly in a production environment and should be run before any release.
- Use Cursor's AI capabilities to help analyze and debug test failures, especially for complex integration tests.
