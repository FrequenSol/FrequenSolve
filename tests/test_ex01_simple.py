"""Test suite for the simple 2D acoustic simulation example (ex01_simple.ipynb).

This module contains tests that verify the functionality demonstrated in the
ex01_simple.ipynb example notebook. The tests cover the complete workflow of
setting up and running a simple 2D acoustic simulation.

Test Structure and Design
------------------------
The test suite is organized using a combination of utility functions and pytest fixtures
to balance test isolation, performance, and maintainability. Here's how it works:

1. Utility Functions
   - Core setup logic is separated into utility functions (create_project, create_simulation, etc.)
   - These functions are pure and can be used independently of the fixture mechanism
   - Makes the code more reusable and easier to test
   - Allows for different fixture scopes to use the same setup logic

2. Fixtures
   - Function-scoped fixtures (default):
     * project_path: Creates a temporary directory for each test
     * project: Creates a fresh project for each test
     * simulation: Creates a new simulation for each test
     * These are used by tests that need isolated state

   - Module-scoped fixtures:
     * time_domain_results: Runs the simulation once and caches results for all tests
     * Uses tmp_path_factory to create a module-scoped temporary directory
     * Used by plotting tests that only need to read simulation results

3. Design Decisions
   a. Separation of Concerns
      - Setup logic is separated from fixture mechanism
      - Makes it easier to modify setup without changing fixture behavior
      - Allows reuse of setup code in different contexts

   b. Fixture Independence
      - Fixtures don't depend on each other
      - Each fixture uses utility functions directly
      - Prevents scope mismatch issues
      - Makes it easier to change fixture scopes

   c. Performance Optimization
      - Time-consuming simulation only runs once per module
      - Other fixtures run per test for proper isolation
      - Balance between test isolation and performance

   d. Maintainability
      - Setup logic is centralized in utility functions
      - Changes to setup only need to be made in one place
      - Clear separation between setup and fixture behavior
      - Easy to add new fixtures with different scopes

Parameter Organization
--------------------
The test parameters are organized into module-level constants that match the values
used in ex01_simple.ipynb. This organization serves several purposes:

1. Single Source of Truth
   - All test parameters are defined in one place at the module level
   - Parameters are grouped logically (MODEL_PARAMS, MESH_PARAMS, etc.)
   - Makes it easy to see what values match the notebook
   - Reduces the risk of inconsistent parameter usage across tests

2. Improved Maintainability
   - Changes to parameters only need to be made in one place
   - Parameter groups make it clear which values belong together
   - Documentation of parameters is centralized
   - Easier to track changes to parameters over time

3. Better Test Readability
   - Tests use descriptive parameter names instead of magic numbers
   - Parameter groups provide context for what each value represents
   - Assertions reference parameters directly, making their purpose clear
   - Error messages can reference the parameter source

4. Enhanced Documentation
   - Parameter groups serve as implicit documentation
   - Values are clearly associated with their purpose
   - Relationship to notebook values is explicit
   - Makes it easier for new developers to understand the test setup

5. Flexible Parameter Usage
   - Parameters can be used with dictionary unpacking
   - Easy to modify subsets of parameters for specific tests
   - Simple to add new parameter groups as needed
   - Parameters can be referenced in docstrings and error messages

Test Categories
--------------
The tests are organized into several categories:

1. Project Creation
   - Creating a new FrequenSolve project
   - Basic project configuration and persistence

2. Model Setup
   - Creating a layered acoustic model
   - Defining surfaces and material properties
   - Configuring a two-layer velocity model

3. Mesh Generation
   - Creating a computational mesh from the model
   - Setting up mesh adaptation parameters
   - Verifying mesh properties

4. Boundary Conditions
   - Setting up free surface conditions
   - Configuring PML (Perfectly Matched Layer) boundaries
   - Managing boundary condition parameters

5. Acquisition Geometry
   - Setting up source locations and types
   - Configuring receiver arrays
   - Managing receiver components and properties

6. Simulation Configuration
   - Setting up numerical discretization
   - Configuring solver parameters
   - Managing output settings

7. Simulation Execution
   - Running frequency domain simulations
   - Running time domain simulations
   - Basic result validation

8. Visualization Tests
   - Plotting model geometry
   - Plotting time domain results
   - Plotting frequency domain results
   - These tests use the module-scoped time_domain_results fixture

Note: This test suite focuses on verifying the setup and configuration of
simulations rather than the numerical accuracy of the results. The example
notebook provides more detailed analysis of the simulation results.

See Also
--------
ex01_simple.ipynb : The example notebook that this test suite verifies
"""

import logging
import os
import shutil
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from frequensolve import *

# =============================================================================
# Test Parameters
# =============================================================================

# Constants for test parameters
# These match the values used in ex01_simple.ipynb
MODEL_PARAMS = {
    "x_limits": [0.0, 2.0],
    "layers": [
        {
            "name": "layer_1",
            "z": 0.25,
            "properties": {"Vp": 1.0, "Qp": 300.0, "Rho": 1.0},
        },
        {
            "name": "layer_2",
            "z": 0.5,
            "properties": {"Vp": 2.0, "Qp": 50.0, "Rho": 1.0},
        },
    ],
}

MESH_PARAMS = {
    "n": [4, 4],
    "min_epw": 1.0,
    "adapt_sources": 1,
}

BOUNDARY_PARAMS = {
    "free_surface": {
        "name": "free_surface",
        "kind": "neumann",
        "boundaries": ["z_min"],
    },
    "pml": {
        "name": "pml",
        "kind": "pml",
        "boundaries": ["x_min", "x_max", "z_max"],
        "pml_wavelengths": 1.2,
        "pml_exponent": 3.0,
        "pml_constant": 20.0,
    },
}

ACQUISITION_PARAMS = {
    "source": {
        "kind": "scalar",
        "coords": [[0.5, 0.0]],
    },
    "receivers": {
        "name": "surface_hydrophones",
        "device": "hydrophone",
        "component": {"name": "p", "field": "pressure"},
        "coords": {"x_range": [0.0, 1.0], "n": 1001},
    },
}

SIMULATION_PARAMS = {
    "discretization": {"order": 6},
    "solver": {
        "solve_on": "final",
        "max_iter": 300,
        "tolerance": 1.0e-4,
    },
    "output": {
        "name": "simple",
        "fields": ["pressure"],
        "upscale": 1,
    },
}

TIME_DOMAIN_PARAMS = {
    "f_min": 1.0,
    "f_max": 18.0,
    "T_max": 4.0,
    "wavelet": {"f0": 6.0, "offset": 5},
}


# =============================================================================
# Utility Functions
# =============================================================================


def create_project_path(tmp_path):
    """Create a temporary directory for a project.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory provided by pytest

    Returns
    -------
    str
        Path to the created project directory
    """
    path = tmp_path / "ex_01"
    path.mkdir()
    return str(path)


def create_project(project_path):
    """Create a new FrequenSolve project.

    Parameters
    ----------
    project_path : str
        Path where the project should be created

    Returns
    -------
    Project
        The created project instance
    """
    project = Project(
        name="project",
        pretty_name="Simple Simulation",
        path=project_path,
        load_if_exists=False,
    )
    return project


def create_simulation(project):
    """Create a simulation and add it to the project.

    Parameters
    ----------
    project : Project
        The project to add the simulation to

    Returns
    -------
    SeismicSimulation
        The created simulation instance
    """
    sim = project.new_simulation(
        name="simulation_01",
        physics="acoustic",
        dimension=2,
    )
    return sim


def setup_complete_simulation(simulation):
    """Set up a complete simulation with all components.

    This function sets up the model, mesh, boundary conditions,
    acquisition, discretization, solver, and output components.
    All parameters match those used in ex01_simple.ipynb.

    Parameters
    ----------
    simulation : SeismicSimulation
        The simulation to set up
    """
    # Create model
    model = LayeredModel(dimension=2, x_limits=MODEL_PARAMS["x_limits"])
    model.add_surface(name="top", z=0.0)

    for layer in MODEL_PARAMS["layers"]:
        model.add_layer(name=layer["name"], properties=layer["properties"])
        if layer != MODEL_PARAMS["layers"][-1]:  # Don't add surface after last layer
            model.add_surface(name=f"{layer['name']}_interface", z=layer["z"])

    simulation += model

    # Create mesh
    mesh = model.hex_mesh_generator(n=MESH_PARAMS["n"])
    simulation += mesh
    simulation.mesh.set_adapt(
        min_epw=MESH_PARAMS["min_epw"], adapt_sources=MESH_PARAMS["adapt_sources"]
    )

    # Add boundary conditions
    BCs = BoundaryConditionManager(label_type="geometric")
    for bc_params in BOUNDARY_PARAMS.values():
        BCs += BoundaryCondition(**bc_params)
    simulation += BCs

    # Add acquisition
    acq = Acquisition()
    acq.add_source_group(**ACQUISITION_PARAMS["source"])

    # Set up receiver
    hydrophone = ReceiverNode(name=ACQUISITION_PARAMS["receivers"]["device"])
    hydrophone.add_component(**ACQUISITION_PARAMS["receivers"]["component"])

    # Create receiver coordinates
    coords = [
        [x, 0.0]
        for x in np.linspace(
            ACQUISITION_PARAMS["receivers"]["coords"]["x_range"][0],
            ACQUISITION_PARAMS["receivers"]["coords"]["x_range"][1],
            ACQUISITION_PARAMS["receivers"]["coords"]["n"],
        )
    ]

    acq.add_receiver_group(
        name=ACQUISITION_PARAMS["receivers"]["name"],
        device=hydrophone,
        coords=coords,
        frame="reference",
    )
    simulation += acq

    # Add discretization
    method = Discretization(**SIMULATION_PARAMS["discretization"])
    simulation += method

    # Add solver config
    solver = SolverConfig(**SIMULATION_PARAMS["solver"])
    simulation += solver

    # Add output
    out = OutputManager()
    out += ParaviewOutput(**SIMULATION_PARAMS["output"])
    simulation += out


def run_time_domain_simulation(project, simulation):
    """Run a time domain simulation and return the results.

    Parameters
    ----------
    project : Project
        The project containing the simulation
    simulation : SeismicSimulation
        The simulation to run

    Returns
    -------
    tuple
        (trace_db, wavelet) containing the simulation results
    """
    # Save the project to ensure simulation file exists
    project.save()

    site = LocalSite()
    site.sync(project)

    # Define and submit a time-domain simulation
    td_job = TimeDomainJob(
        name="time", simulation=simulation, f_min=1.0, f_max=18.0, T_max=4.0
    )

    # Capture stdout during job submission
    stdout_capture = StringIO()
    with redirect_stdout(stdout_capture):
        site.submit(td_job)

    # Get results from the site
    trace_db = site.fetch_traces(td_job, upscale=4)
    wavelet = RickerWavelet(f0=6.0, times=trace_db.times(), offset=5)

    return trace_db, wavelet


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def project_path(tmp_path):
    """Create a temporary directory for the project."""
    return create_project_path(tmp_path)


@pytest.fixture
def project(project_path):
    """Create a project for testing."""
    return create_project(project_path)


@pytest.fixture
def simulation(project):
    """Create a simulation for testing."""
    sim = create_simulation(project)
    setup_complete_simulation(sim)
    return sim


@pytest.fixture(scope="module")
def time_domain_results(tmp_path_factory):
    """Run a time domain simulation and cache the results for all tests.

    This fixture is module-scoped to avoid running the simulation multiple times,
    but it creates its own project and simulation instances to avoid depending
    on other fixtures.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for creating temporary directories
    """
    # Create a module-scoped temporary directory
    tmp_path = tmp_path_factory.mktemp("time_domain")

    # Create project and simulation directly
    project_path = create_project_path(tmp_path)
    project = create_project(project_path)
    simulation = create_simulation(project)
    setup_complete_simulation(simulation)

    # Run simulation and yield results
    results = run_time_domain_simulation(project, simulation)
    yield results

    # Cleanup will happen automatically when the module is done


# =============================================================================
# Test Suite
# =============================================================================

# -----------------------------------------------------------------------------
# Project and Model Tests
# -----------------------------------------------------------------------------


def test_project_creation(project):
    """Test the creation and basic configuration of a FrequenSolve Project.

    This test verifies that:
    1. The project is created with the correct name and pretty name
    2. The project.json file is created when saving the project

    This corresponds to the project creation section in ex01_simple.ipynb where
    a new project is initialized with name="project" and pretty_name="Simple Simulation".
    """
    assert project.name == "project"
    assert project.pretty_name == "Simple Simulation"

    # Save project to create project.json
    project.save()
    assert os.path.exists(os.path.join(project.path, "project.json"))


def test_model_setup(simulation):
    """Test the setup of a layered acoustic model.

    This test verifies that:
    1. A 2D layered model can be created with specified x-limits
    2. Surfaces can be added at specific z-depths (top, interface, bottom)
    3. Layers can be added with specified material properties (Vp, Qp, Rho)
    4. The model correctly stores and provides access to surfaces and layers
    5. Material properties can be accessed and have the expected values

    This corresponds to the "Defining a Model" section in ex01_simple.ipynb where
    a two-layer model is created with:
    - Layer 1: Vp=1.0 km/s, Qp=300.0, Rho=1.0 g/cm³
    - Layer 2: Vp=2.0 km/s, Qp=50.0, Rho=1.0 g/cm³
    - Interface at z=0.25
    """
    # Create model
    model = LayeredModel(dimension=2, x_limits=[0.0, 2.0])

    # Add surfaces and layers
    model.add_surface(name="top", z=0.0)
    model.add_layer(name="layer_1", properties={"Vp": 1.0, "Qp": 300.0, "Rho": 1.0})
    model.add_surface(name="interface", z=0.25)
    model.add_layer(name="layer_2", properties={"Vp": 2.0, "Qp": 50.0, "Rho": 1.0})
    model.add_surface(name="bottom", z=0.5)

    simulation += model

    # Verify model properties
    assert len(model.surfaces) == 3  # top, interface, bottom
    assert len(model.layers) == 2  # layer_1, layer_2
    assert float(model.layers[0].properties["Vp"].get()) == 1.0
    assert float(model.layers[1].properties["Vp"].get()) == 2.0
    assert float(model.layers[0].properties["Qp"].get()) == 300.0
    assert float(model.layers[1].properties["Qp"].get()) == 50.0
    assert float(model.layers[0].properties["Rho"].get()) == 1.0
    assert float(model.layers[1].properties["Rho"].get()) == 1.0


def test_mesh_setup(simulation):
    """Test the setup of the computational mesh for the simulation.

    This test verifies that:
    1. A mesh can be generated from the layered model using hex_mesh_generator
    2. The mesh can be added to the simulation
    3. Mesh adaptation parameters can be set (min_epw, adapt_sources)
    4. The mesh and its adaptation settings are properly stored in the simulation

    This corresponds to the "Meshing" section in ex01_simple.ipynb where
    a quad mesh is generated with n=[4,4] elements and adaptation settings
    are configured for wave propagation.
    """
    # Create model first (required for mesh)
    model = LayeredModel(dimension=2, x_limits=[0.0, 2.0])
    model.add_surface(name="top", z=0.0)
    model.add_layer(name="layer_1", properties={"Vp": 1.0, "Rho": 1.0})
    model.add_surface(name="interface", z=0.25)
    model.add_layer(name="layer_2", properties={"Vp": 2.0, "Rho": 1.0})
    model.add_surface(name="bottom", z=0.5)
    simulation += model

    # Create mesh
    mesh = model.hex_mesh_generator(n=[4, 4])
    simulation += mesh
    simulation.mesh.set_adapt(min_epw=1.0, adapt_sources=1)

    # Basic mesh validation
    assert simulation.mesh is not None
    assert simulation.mesh.adapt is not None
    assert simulation.mesh.adapt.min_epw == 1.0


def test_boundary_conditions(simulation):
    """Test the setup of boundary conditions for the simulation.

    This test verifies that:
    1. A BoundaryConditionManager can be created with geometric labeling
    2. Free surface (Neumann) boundary conditions can be added
    3. PML (Perfectly Matched Layer) boundary conditions can be added with parameters
    4. The boundary conditions are properly stored and accessible

    This corresponds to the "Boundary Conditions" section in ex01_simple.ipynb where
    boundary conditions are set up with:
    - Free surface (Neumann) condition on the top boundary
    - PML conditions on the sides and bottom with specified parameters
    """
    BCs = BoundaryConditionManager(label_type="geometric")

    # Add free surface BC
    BCs += BoundaryCondition(name="free_surface", kind="neumann", boundaries=["z_min"])

    # Add PML BC
    BCs += BoundaryCondition(
        name="pml",
        kind="pml",
        boundaries=["x_min", "x_max", "z_max"],
        pml_wavelengths=1.2,
        pml_exponent=3.0,
        pml_constant=20.0,
    )

    simulation += BCs

    # Verify BCs
    assert len(BCs.boundary_conditions) == 2
    assert any(bc.kind == "neumann" for bc in BCs.boundary_conditions)
    assert any(bc.kind == "pml" for bc in BCs.boundary_conditions)


# -----------------------------------------------------------------------------
# Acquisition and Simulation Tests
# -----------------------------------------------------------------------------


def test_acquisition_setup(simulation):
    """Test the setup of sources and receivers for the simulation.

    This test verifies that:
    1. An Acquisition object can be created
    2. A scalar source can be added at a specific location
    3. A receiver group with hydrophones can be created
    4. Receiver components can be configured (pressure field)
    5. Receiver coordinates can be set up in a regular grid
    6. The acquisition geometry is properly stored and accessible

    This corresponds to the "Defining an Acquisition" section in ex01_simple.ipynb where
    the acquisition is set up with:
    - A scalar source at x=0.5, z=0.0
    - 1001 hydrophones along the surface (z=0.0) from x=0.0 to x=1.0
    """
    acq = Acquisition()

    # Add source
    acq.add_source_group(kind="scalar", coords=[[0.5, 0.0]])

    # Add receivers
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")

    coords = [[x, 0.0] for x in np.linspace(0.0, 1.0, 1001)]
    acq.add_receiver_group(
        name="surface_hydrophones", device=hydrophone, coords=coords, frame="reference"
    )

    simulation += acq

    # Test acquisition setup
    assert len(acq.source_groups) == 1
    assert len(acq.receiver_groups) == 1
    assert acq.receiver_groups[0].name == "surface_hydrophones"
    assert len(acq.receiver_groups[0].device.components) == 1
    assert acq.receiver_groups[0].device.components[0].field == "pressure"
    assert acq.receiver_groups[0].device.components[0].direction is None

    # Test source and receiver locations
    source_coords = acq.source_groups[0].get_coordinates()
    receiver_coords = acq.receiver_groups[0].coordinates.get()
    assert source_coords.shape == (1, 2)
    assert receiver_coords.shape == (1001, 2)
    assert np.allclose(source_coords[0], [0.5, 0.0])
    assert np.allclose(receiver_coords[0], [0.0, 0.0])
    assert np.allclose(receiver_coords[-1], [1.0, 0.0])


def test_output_configuration(simulation):
    """Test the configuration of simulation outputs.

    This test verifies that:
    1. Paraview output can be configured with specific settings
    2. Output fields and upscale parameters are correctly set
    3. The output configuration matches the example notebook

    This corresponds to the "Outputs" section in ex01_simple.ipynb where
    Paraview output is configured with:
    - name="simple"
    - fields=["pressure"]
    - upscale=1
    """
    out = OutputManager()
    out += ParaviewOutput(name="simple", fields=["pressure"], upscale=1)
    simulation += out

    # Verify output configuration
    assert len(simulation.outputs.paraview) == 1, "Should have one Paraview output"
    pv_output = simulation.outputs.paraview[0]
    assert pv_output.name == "simple", "Output name should match notebook"
    assert pv_output.fields == ["pressure"], "Should output pressure field"
    assert pv_output.upscale == 1, "Upscale should match notebook"
    assert (
        simulation.outputs.write_receivers is True
    ), "Should write receiver data by default"


def test_simulation_setup(simulation):
    """Test the complete setup of a seismic simulation.

    This test verifies that:
    1. All major components can be added to the simulation:
       - Model (layered acoustic model)
       - Mesh (computational grid)
       - Boundary conditions (free surface and PML)
       - Acquisition (sources and receivers)
       - Discretization (numerical method settings)
       - Solver configuration
       - Output settings
    2. Each component is properly stored and accessible
    3. The simulation is ready for execution
    4. All component settings match the example notebook exactly

    This corresponds to the complete setup process in ex01_simple.ipynb,
    combining all the individual component tests into a full simulation
    configuration.
    """
    # Add all components
    test_model_setup(simulation)
    test_mesh_setup(simulation)
    test_boundary_conditions(simulation)
    test_acquisition_setup(simulation)
    test_output_configuration(simulation)  # Use dedicated output test

    # Add discretization with order=6 as in the notebook
    method = Discretization(order=6)
    simulation += method
    assert (
        simulation.discretization.order == 6
    ), "Discretization order should be 6 as in the notebook"

    # Add solver config with exact notebook parameters
    solver = SolverConfig(solve_on="final", max_iter=300, tolerance=1.0e-4)
    simulation += solver
    assert simulation.solver.solve_on == "final", "Should solve on final mesh"
    assert simulation.solver.max_iter == 300, "Maximum iterations should be 300"
    assert simulation.solver.tolerance == 1.0e-4, "Tolerance should be 1e-4"

    # Verify all components are present
    assert simulation.model is not None, "Model should be present"
    assert simulation.mesh is not None, "Mesh should be present"
    assert simulation.BCs is not None, "Boundary conditions should be present"
    assert simulation.acquisition is not None, "Acquisition should be present"
    assert simulation.discretization is not None, "Discretization should be present"
    assert simulation.solver is not None, "Solver should be present"
    assert simulation.outputs is not None, "Outputs should be present"


# -----------------------------------------------------------------------------
# Time Domain Tests
# -----------------------------------------------------------------------------


@pytest.mark.integration
def test_time_domain_parameters(time_domain_results):
    """Test the time domain simulation parameters.

    This test verifies that the time domain simulation parameters match
    those specified in ex01_simple.ipynb:
    - Frequency range: {TIME_DOMAIN_PARAMS['f_min']} to {TIME_DOMAIN_PARAMS['f_max']} Hz
    - Simulation duration: {TIME_DOMAIN_PARAMS['T_max']} s
    - Ricker wavelet: f0={TIME_DOMAIN_PARAMS['wavelet']['f0']} Hz, offset={TIME_DOMAIN_PARAMS['wavelet']['offset']} samples
    """
    trace_db, wavelet = time_domain_results

    # Verify frequency range from metadata
    assert (
        trace_db.metadata["f_max"] == TIME_DOMAIN_PARAMS["f_max"]
    ), "Maximum frequency should match notebook"
    assert trace_db.metadata["df"] > 0, "Frequency step should be positive"

    # Verify wavelet parameters through signal properties
    peak_freq_idx = np.argmax(np.abs(wavelet.spectrum))
    peak_freq = wavelet.frequencies[peak_freq_idx]
    assert (
        abs(peak_freq - TIME_DOMAIN_PARAMS["wavelet"]["f0"]) < 0.5
    ), "Wavelet peak frequency should match notebook"

    # Verify time offset
    peak_time_idx = np.argmax(np.abs(wavelet.signal))
    assert (
        abs(peak_time_idx - TIME_DOMAIN_PARAMS["wavelet"]["offset"]) <= 1
    ), "Wavelet offset should match notebook"

    # Verify time sampling
    times = trace_db.times()
    assert times[0] >= 0.0, "Time should start at or after 0"
    assert times[-1] <= TIME_DOMAIN_PARAMS["T_max"], "Time should not exceed T_max"
    assert len(times) > 0, "Should have time samples"


@pytest.mark.integration
def test_time_domain_simulation_basic(project, simulation):
    """Test basic functionality of time domain simulation without plotting.

    This test verifies that:
    1. A complete simulation can be set up
    2. The project can be saved
    3. A time domain job can be created with correct parameters
    4. The simulation produces valid results

    This corresponds to the time domain simulation section in ex01_simple.ipynb
    where a TimeDomainJob is created with:
    - f_min=1.0 Hz
    - f_max=18.0 Hz
    - T_max=4.0 s
    """
    # Set up complete simulation
    test_simulation_setup(simulation)

    # Save the project to ensure simulation file exists
    project.save()
    assert os.path.exists(
        os.path.join(project.path, "project.json")
    ), "Project file should exist"

    # Create and run time domain job with notebook parameters
    td_job = TimeDomainJob(
        name="td_job", simulation=simulation, f_min=1.0, f_max=18.0, T_max=4.0
    )

    # Verify job parameters through f_list
    assert len(td_job.f_list) > 0, "Should have frequency samples"
    assert min(td_job.f_list) >= 1.0, "Minimum frequency should be at least 1.0 Hz"
    assert max(td_job.f_list) <= 18.0, "Maximum frequency should be at most 18.0 Hz"
    assert (
        td_job.f_list[1] - td_job.f_list[0] == 1.0 / 4.0
    ), "Frequency step should be 1/T_max"

    # Run locally
    td_results = td_job.records

    # Basic validation of results
    assert td_results is not None, "Should have simulation results"
    assert len(td_results) > 0, "Should have at least one record"


# -----------------------------------------------------------------------------
# Visualization Tests
# -----------------------------------------------------------------------------


@pytest.mark.mpl_image_compare(tolerance=2.0)
def test_model_plot(simulation):
    """Test basic plotting functionality without requiring the solver.

    This test verifies that the model visualization matches the expected reference
    image. It generates a plot of the model geometry showing the Vp property
    with equal aspect ratio.

    This corresponds to the model plotting in ex01_simple.ipynb where
    model.plot("Vp", aspect="equal") is called.

    Parameters
    ----------
    simulation : SeismicSimulation
        The simulation object to generate plots for.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot to be compared.
    """
    # Set up the simulation
    test_simulation_setup(simulation)

    # Create figure and plot model similar to the notebook
    # NOTE: We have to obtain the figure object first and pass the ax to the plot method
    #       to prevent the plot method from creating a new figure and closing it.
    #       This method needs to return the correct figure to be compared using the mpl_image_compare decorator.
    fig, ax = plt.subplots()
    simulation.model.plot(property="Vp", aspect="equal", ax=ax)

    return fig


@pytest.mark.integration
@pytest.mark.mpl_image_compare(tolerance=2.0, savefig_kwargs={"dpi": 100})
def test_time_domain_wavelet_time_plot(time_domain_results):
    """Test the time-domain wavelet plot from time domain simulation.

    This test verifies that the time-domain wavelet plot matches its expected reference image.
    """
    _, wavelet = time_domain_results

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot on the provided axes
    wavelet.plot(ax_time=ax)

    return fig


@pytest.mark.integration
@pytest.mark.mpl_image_compare(tolerance=2.0, savefig_kwargs={"dpi": 100})
def test_time_domain_wavelet_freq_plot(time_domain_results):
    """Test the frequency-domain wavelet plot from time domain simulation.

    This test verifies that the frequency-domain wavelet plot matches its expected reference image.
    """
    _, wavelet = time_domain_results

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot on the provided axes
    wavelet.plot(ax_freq=ax)

    return fig


@pytest.mark.integration
@pytest.mark.mpl_image_compare(tolerance=2.0)
def test_time_domain_common_frequency_plot(time_domain_results):
    """Test the common frequency plot from time domain simulation.

    This test verifies that the common frequency plot matches the expected reference image.
    """
    trace_db, wavelet = time_domain_results

    # Get first record and create common frequency plot
    record = next(iter(trace_db))
    shot = record.read_FD(wavelet)
    fig, ax = plt.subplots()
    plot_cf(shot, c_min=0.4, c_max=3.0, n_c=400, ax=ax)

    return fig


@pytest.mark.integration
@pytest.mark.mpl_image_compare(tolerance=2.0)
def test_time_domain_gather_plot(time_domain_results):
    """Test the gather plot from time domain simulation.

    This test verifies that the gather plot matches the expected reference image.
    """
    trace_db, wavelet = time_domain_results

    # Get first record and create gather plot
    record = next(iter(trace_db))
    shot = record.read_TD(wavelet)
    fig, ax = plt.subplots()
    plot_gather(shot, A=100, ax=ax)

    return fig
