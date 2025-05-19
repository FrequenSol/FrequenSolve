"""Test suite for the simple 2D acoustic simulation example (ex01_simple.ipynb).

This module contains tests that verify the functionality demonstrated in the
ex01_simple.ipynb example notebook. The tests cover the complete workflow of
setting up and running a simple 2D acoustic simulation, including:

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

The tests are designed to verify both the individual components and their
integration into a complete simulation. Some tests are marked as integration
tests (using @pytest.mark.integration) as they require running the actual
simulation solver (most likely in the docker container).

Note: This test suite focuses on verifying the setup and configuration of
simulations rather than the numerical accuracy of the results. The example
notebook provides more detailed analysis of the simulation results.

See Also
--------
ex01_simple.ipynb : The example notebook that this test suite verifies
"""

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from frequensolve import *


@pytest.fixture
def project_path(tmp_path):
    """Create a temporary directory for the project."""
    path = tmp_path / "ex_01"
    path.mkdir()
    return str(path)


@pytest.fixture
def project(project_path):
    """Create a project for testing."""
    project = Project(
        name="project",
        pretty_name="Simple Simulation",
        path=project_path,
        load_if_exists=False,
    )
    return project


@pytest.fixture
def simulation(project):
    """Create a simulation for testing."""
    sim = SeismicSimulation(
        name="simulation_01",
        physics="acoustic",
        dimension=2,
    )
    project += sim
    return sim


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
    3. Layers can be added with specified material properties (Vp, Rho)
    4. The model correctly stores and provides access to surfaces and layers
    5. Material properties can be accessed and have the expected values

    This corresponds to the "Defining a Model" section in ex01_simple.ipynb where
    a two-layer model is created with:
    - Layer 1: Vp=1.0 km/s, Rho=1.0 g/cm³
    - Layer 2: Vp=2.0 km/s, Rho=1.0 g/cm³
    - Interface at z=0.25
    """
    # Create model
    model = LayeredModel(dimension=2, x_limits=[0.0, 2.0])

    # Add surfaces and layers
    model.add_surface(name="top", z=0.0)
    model.add_layer(name="layer_1", properties={"Vp": 1.0, "Rho": 1.0})
    model.add_surface(name="interface", z=0.25)
    model.add_layer(name="layer_2", properties={"Vp": 2.0, "Rho": 1.0})
    model.add_surface(name="bottom", z=0.5)

    simulation += model

    # Verify model properties
    assert len(model.surfaces) == 3  # top, interface, bottom
    assert len(model.layers) == 2  # layer_1, layer_2
    assert float(model.layers[0].properties["Vp"].get()) == 1.0
    assert float(model.layers[1].properties["Vp"].get()) == 2.0


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

    This corresponds to the complete setup process in ex01_simple.ipynb,
    combining all the individual component tests into a full simulation
    configuration.
    """
    # Add all components
    test_model_setup(simulation)
    test_mesh_setup(simulation)
    test_boundary_conditions(simulation)
    test_acquisition_setup(simulation)

    # Add discretization
    method = Discretization(order=4)
    simulation += method

    # Add solver config
    solver = SolverConfig(solve_on="final", max_iter=300, tolerance=1.0e-4)
    simulation += solver

    # Add output
    out = OutputManager()
    out += ParaviewOutput(name="simple", fields=["pressure"], upscale=2)
    simulation += out

    # Verify simulation components
    assert simulation.model is not None
    assert simulation.mesh is not None
    assert simulation.BCs is not None
    assert simulation.acquisition is not None
    assert simulation.discretization is not None
    assert simulation.solver is not None
    assert simulation.outputs is not None


def test_project_save_load(project, simulation):
    """Test saving and loading a project with a complete simulation setup.

    This test verifies that:
    1. A project with a complete simulation can be saved to disk
    2. The saved project can be loaded back
    3. All simulation components are preserved through the save/load cycle
    4. The loaded project maintains all properties and relationships

    This corresponds to the project saving and loading section in ex01_simple.ipynb
    where the project is saved and then loaded back from the project.json file.
    """
    # Set up complete simulation
    test_simulation_setup(simulation)

    # Save project and store path
    project.save()
    project_path = os.path.join(project.path, "project.json")
    original_path = project.path  # Store the path before deleting project
    assert os.path.exists(project_path), "Project file was not created"

    # Delete project and simulation objects
    del simulation
    del project

    # Load project back
    loaded_project = Project.load(project_path)

    # Verify project properties
    assert loaded_project.name == "project"
    assert loaded_project.pretty_name == "Simple Simulation"
    assert loaded_project.path == original_path  # Compare with stored path

    # Get the simulation from loaded project
    loaded_sim = loaded_project.simulations["simulation_01"]

    # Verify simulation properties
    assert loaded_sim.name == "simulation_01"
    assert loaded_sim.physics == "acoustic"
    assert loaded_sim.dimension == 2

    # Verify all components are present and properly loaded
    assert loaded_sim.model is not None
    assert loaded_sim.mesh is not None
    assert loaded_sim.BCs is not None
    assert loaded_sim.acquisition is not None
    assert loaded_sim.discretization is not None
    assert loaded_sim.solver is not None
    assert loaded_sim.outputs is not None

    # Verify model properties
    assert len(loaded_sim.model.surfaces) == 3
    assert len(loaded_sim.model.layers) == 2
    assert float(loaded_sim.model.layers[0].properties["Vp"].get()) == 1.0
    assert float(loaded_sim.model.layers[1].properties["Vp"].get()) == 2.0

    # Verify acquisition setup
    assert len(loaded_sim.acquisition.source_groups) == 1
    assert len(loaded_sim.acquisition.receiver_groups) == 1
    assert loaded_sim.acquisition.receiver_groups[0].name == "surface_hydrophones"

    # Verify solver configuration
    assert loaded_sim.solver.solve_on == "final"
    assert loaded_sim.solver.max_iter == 300
    assert loaded_sim.solver.tolerance == 1.0e-4

    # Verify output configuration
    assert loaded_sim.outputs.write_receivers is True
    assert len(loaded_sim.outputs.paraview) == 1
    assert loaded_sim.outputs.paraview[0].name == "simple"
    assert "pressure" in loaded_sim.outputs.paraview[0].fields
    assert len(loaded_sim.outputs.wavefields) == 0  # No wavefield outputs in this test


@pytest.mark.mpl_image_compare(tolerance=2.0)
def test_plot_basic(simulation):
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
@pytest.mark.mpl_image_compare(tolerance=2.0)
def test_plot_verification_integration(simulation):
    """Test complete visualization including mesh generation (requires solver).

    This test verifies that the complete visualization outputs match the expected reference
    images. It generates plots for:
    1. Model geometry (layers and surfaces)
    2. Mesh with boundary conditions (requires solver)
    3. Acquisition geometry (sources and receivers)
    4. Frequency domain results (if available)

    The test uses pytest-mpl to compare the generated plots against reference
    images stored in tests/reference_images/. The tolerance parameter allows for
    small differences in rendering between systems.

    Note: This is marked as an integration test as it requires the solver to
    generate the mesh for proper visualization.

    Parameters
    ----------
    simulation : SeismicSimulation
        The simulation object to generate plots for.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plots to be compared.
    """
    # Set up the simulation
    test_simulation_setup(simulation)

    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)

    # Plot 1: Model geometry
    ax1 = fig.add_subplot(gs[0, 0])
    simulation.model.plot(property="Vp", ax=ax1)
    ax1.set_title("Model Geometry")

    # Plot 2: Mesh with boundary conditions
    ax2 = fig.add_subplot(gs[0, 1])
    simulation.mesh.mesh.plot(
        ax=ax2
    )  # Now we can plot the mesh since this is an integration test
    simulation.BCs.plot(ax=ax2)
    ax2.set_title("Mesh and Boundary Conditions")

    # Plot 3: Acquisition geometry
    ax3 = fig.add_subplot(gs[1, 0])
    simulation.acquisition.plot(ax=ax3)
    ax3.set_title("Acquisition Geometry")

    # Plot 4: Frequency domain results (if available)
    ax4 = fig.add_subplot(gs[1, 1])
    if hasattr(simulation, "results") and simulation.results is not None:
        simulation.results.plot(ax=ax4)
        ax4.set_title("Frequency Domain Results")
    else:
        ax4.text(
            0.5,
            0.5,
            "No results available",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_title("Frequency Domain Results (Not Available)")

    plt.tight_layout()
    return fig


@pytest.mark.integration
def test_frequency_domain_simulation(project, simulation):
    """Test running a frequency domain simulation.

    This test verifies that:
    1. A complete simulation can be set up
    2. The project can be saved
    3. A frequency domain job can be created and run
    4. The simulation produces results

    This corresponds to the frequency domain simulation section in ex01_simple.ipynb
    where a single frequency (10 Hz) is simulated.

    Note: This is marked as an integration test as it requires running the actual
    simulation solver.
    """
    # Set up complete simulation
    test_simulation_setup(simulation)

    # Save project
    project.save()

    # Create and run frequency domain job
    fd_job = FrequencyDomainJob(name="fd_job", simulation=simulation, f_list=[10.0])

    # Run locally
    fd_results = fd_job.records

    # Basic validation of results
    assert fd_results is not None


@pytest.mark.integration
def test_time_domain_simulation(project, simulation):
    """Test running a time domain simulation.

    This test verifies that:
    1. A complete simulation can be set up
    2. The project can be saved
    3. A time domain job can be created and run
    4. The simulation produces results

    This corresponds to the time domain simulation section in ex01_simple.ipynb
    where multiple frequencies (1-18 Hz) are simulated and combined to create
    a time domain response.

    Note: This is marked as an integration test as it requires running the actual
    simulation solver and may take significant time to complete.
    """
    # Set up complete simulation
    test_simulation_setup(simulation)

    # Save project
    project.save()

    # Create and run time domain job
    td_job = TimeDomainJob(
        name="td_job", simulation=simulation, f_min=1.0, f_max=18.0, T_max=2.0
    )

    # Run locally
    td_results = td_job.records

    # Basic validation of results
    assert td_results is not None
