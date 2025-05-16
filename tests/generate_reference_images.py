"""Script to generate reference images for plot verification tests.

This script creates a set of reference images that will be used by the plot
verification tests. It should be run once to create the baseline images, and
then the images should be committed to the repository.

Usage:
    python generate_reference_images.py
"""

import os
from pathlib import Path

from test_ex01_simple import test_plot_verification

from frequensolve.project import Project
from frequensolve.simulation.simulation import SeismicSimulation


def main():
    """Generate reference images for plot verification tests."""
    # Create reference images directory if it doesn't exist
    ref_dir = Path(__file__).parent / "reference_images"
    ref_dir.mkdir(exist_ok=True)

    # Create a temporary project and simulation
    project = Project(
        name="temp_project",
        pretty_name="Temporary Project for Reference Images",
        path="scratch/reference_images",
        load_if_exists=False,
    )
    simulation = SeismicSimulation(name="temp_sim", physics="acoustic", dimension=2)
    project += simulation

    # Generate the plots and save them
    print("Generating reference images...")
    fig = test_plot_verification(simulation)
    fig.savefig(ref_dir / "plot_verification.png", dpi=100)
    print(f"Reference images saved to {ref_dir}")

    # Clean up
    plt.close(fig)
    del simulation
    del project


if __name__ == "__main__":
    main()
