"""Verify adapted property-mesh plotting against native Sauce property output.

Run with --solver /path/to/fs2d_s --output /tmp/new-directory. Requires a
solver built with property_mesh output and leaf visualization artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import xarray as xr
from matplotlib.collections import PolyCollection
from scipy.spatial import ConvexHull, cKDTree

import frequensolve as fs
from frequensolve import imaging as im
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.plotting.vtu import _rasterize_planar_field


def make_simulation(project, name):
    sim = project.new_simulation(
        name=name,
        physics="acoustic",
        dimension=2,
        units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
    )
    x = np.linspace(0, 1.2, 49)
    z = np.linspace(0, 0.8, 9)
    vp = xr.DataArray(
        np.broadcast_to(1.2 + 3 * x, (len(z), len(x))).copy(),
        dims=["z", "x"],
        coords={"x": x, "z": z},
    )
    model = fs.LayeredModel(name="adapted", dimension=2, x_limits=[0, 1.2])
    model.add_surface(0.0, name="top")
    model.add_layer(name="rock", properties={"vp": fs.Property(vp), "rho": 2.0})
    model.add_surface(0.8, name="bottom")
    sim += model
    sim += model.hex_mesh_generator([2, 2])
    sim.mesh.set_adapt(
        elems_per_wave=1.0, order=3, f_low=4.0, f_high=4.0, adapt_order=False
    )
    sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
    sim += fs.BoundaryCondition(
        conditions=["pml"],
        boundaries=["x_min", "x_max", "z_max"],
        pml_wavelengths=0.6,
        pml_constant=20.0,
    )
    acq = fs.Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.3, 0.05]])
    rec = fs.ReceiverNode(name="hydrophone")
    rec.add_component(name="p", field="pressure")
    acq.add_receiver_group(
        name="surface",
        device=rec,
        coords=[[x, 0.04] for x in np.linspace(0.05, 1.15, 17)],
    )
    sim += acq
    sim += fs.Discretization()
    sim += fs.SolverConfig(solve_on="final", max_iter=500, tolerance=1e-4)
    return sim


def plot_mesh(ax, mesh, field, title, *, signed=False):
    """Evaluate the original cells at pixels; never replace quads by triangles."""
    raster = _rasterize_planar_field(mesh, field)
    # Solver VTU embeds 2D in x-y; PropertyMesh uses physical x-z.
    depth_axis = 2 if np.ptp(mesh.points[:, 2]) > 0 else 1
    values = raster.cell_data[field].reshape(
        raster.dimensions[depth_axis] - 1, raster.dimensions[0] - 1
    )
    limits = {}
    if signed:
        limit = np.nanmax(np.abs(values))
        limits = dict(vmin=-limit, vmax=limit)
    artist = ax.imshow(
        values,
        extent=(0, 1.2, 0.8, 0),
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r" if signed else "viridis",
        **limits,
    )
    ax.set(
        title=title, xlabel="x [km]", ylabel="depth [km]", ylim=(0.8, 0), xlim=(0, 1.2)
    )
    ax.figure.colorbar(artist, ax=ax)


def run(solver, output):
    output.mkdir(parents=True, exist_ok=False)
    project = fs.Project(
        name="mesh_plot", path=output / "project", load_if_exists=False
    )
    sim = make_simulation(project, "initial")
    controls = im.ControlSpace(
        vp=im.MeshParameters(
            "vp", "rock", frequency=12.0, epw=2.0, artifact=str(output / "velocity.h5")
        )
    )
    with LocalSite(
        solver=solver,
        n_workers=1,
        threads_per_worker=4,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        observed = fs.FrequencyDomainJob(
            "observed", make_simulation(project, "truth"), [4.0]
        )
        assert site.run(observed, check=True).successful
        problem = im.ImagingProblem(
            sim,
            controls=controls,
            observed=im.ObservedData(observed),
            site=site,
            min_support=0,
            name="mesh_plot",
        )
        state = problem.state  # discovers the independent coefficient count
        artifacts = list(output.glob("velocity*.h5"))
        assert len(artifacts) == 1, artifacts
        geometry = im.PropertyMesh.read(artifacts[0], material=1)
        hanging = np.diff(geometry.basis.indptr) > 1
        assert hanging.any(), "This proof must contain constrained hanging vertices"
        counts = np.diff(geometry.basis.indptr)
        master = np.flatnonzero(counts == 1)
        locations = np.zeros((geometry.basis.shape[1], 3))
        locations[geometry.basis.indices[geometry.basis.indptr[master]]] = (
            geometry.points[master] / 1000
        )
        update = 0.12 * np.sin(4 * locations[:, 0]) * np.cos(3 * locations[:, 2])
        problem.state = state.with_update(problem.full_space.pack({"vp": update}))
        mesh = problem.state.to_mesh(geometry, "vp", units="km")
        mesh.save(output / "control_mesh.vtu")
        plotter = pv.Plotter(off_screen=True, window_size=(1000, 600))
        problem.state.plot("vp", mesh=geometry, units="km", plotter=plotter, show=False)
        plotter.screenshot(str(output / "mesh_plotter.png"))
        plotter.close()
        property_output = fs.VtkOutput.property_mesh(
            "vp",
            subdomain="rock",
            name="property",
            path="property",
            sources=[1],
            items=[fs.output_property("vp", output_name="vp", units="km/s")],
        )
        solution_output = fs.VtkOutput.volume(
            name="solution",
            path="solution",
            sources=[1],
            show_pml=False,
            items=[fs.output_property("vp", output_name="vp", units="km/s")],
        )
        grid_output = fs.VtkOutput.grid(
            fs.CartesianGrid(
                n=[45, 31],
                x0=[0.01, 0.01],
                x1=[1.19, 0.79],
                dims=["x", "z"],
                units="km",
            ),
            name="grid",
            path="grid",
            sources=[1],
            items=[fs.output_property("vp", output_name="vp", units="km/s")],
        )
        state_file = problem.save_state(output / "state.h5")
        job = im.FWIOperatorJob(
            "export",
            problem.simulation,
            [4.0],
            action="linearize",
            active=["model.vp"],
            state="linearization.json",
            control_state=state_file,
            covector="gradient.h5",
            control_active=["vp"],
            gram_derivative="total",
            misfit=problem.misfit.to_fs(problem.observed_groups),
            outputs=[
                property_output,
                solution_output,
                grid_output,
                fs.OutputUnits(geometry="km"),
            ],
        )
        assert site.run(job, check=True).successful
        # Include the DPG test-map/Gram derivative on this deliberately coarse
        # solution mesh; its frozen approximation differs appreciably here.
        gradient = im.ControlVector.from_file(job.covector_file(1), problem.space)
        assert np.isfinite(gradient.norm()) and gradient.norm() > 0
        gradient_mesh = gradient.to_mesh(geometry, "vp", units="km")
        gradient_mesh.save(output / "gradient_mesh.vtu")
        plotter = pv.Plotter(off_screen=True, window_size=(1000, 600))
        gradient.plot("vp", mesh=geometry, units="km", plotter=plotter, show=False)
        plotter.screenshot(str(output / "gradient_plotter.png"))
        plotter.close()
        property_file = next(job.vtk_outputs["property"].rglob("*.vtu"))
        physical = pv.read(property_file)
        solution = pv.read(next(job.vtk_outputs["solution"].rglob("*.vtu")))
        sampled = pv.read(next(job.vtk_outputs["grid"].rglob("*.vtr")))
        assert physical.n_cells == mesh.n_cells
        assert solution.n_cells != physical.n_cells
        assert np.isfinite(sampled["vp"]).all()
        depth_axis = 2 if np.ptp(physical.points[:, 2]) > 0 else 1
        # Both outputs use the same leaf vertices. Match coordinates explicitly;
        # VTK cell probing can miss boundary vertices after Float32 rounding.
        distance, index = cKDTree(mesh.points[:, [0, 2]]).query(
            physical.points[:, [0, depth_axis]]
        )
        assert np.max(distance) < 2e-7
        expected = 1.2 + 3 * physical.points[:, 0] + mesh["vp"][index]
        error = float(np.max(np.abs(physical["vp"] - expected)))
        assert error < 2e-5, error
        grid_depth_axis = 2 if np.ptp(sampled.points[:, 2]) else 1
        points = np.column_stack(
            (
                sampled.points[:, 0],
                np.zeros(sampled.n_points),
                sampled.points[:, grid_depth_axis],
            )
        )
        sampled_controls = pv.PolyData(points).sample(mesh)
        assert sampled_controls["vtkValidPointMask"].all()
        grid_expected = 1.2 + 3 * points[:, 0] + sampled_controls["vp"]
        grid_error = float(np.max(np.abs(sampled["vp"] - grid_expected)))
        assert grid_error < 2e-5, grid_error
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        plot_mesh(
            axes[0],
            physical,
            "vp",
            f"Physical vp on property mesh: {physical.n_cells} cells",
        )
        plot_mesh(axes[1], mesh, "vp", "Python: constrained control field", signed=True)
        plot_mesh(
            axes[2],
            gradient_mesh,
            "vp",
            "Total derivative: coefficient-gradient display",
            signed=True,
        )
        fig.savefig(output / "property_mesh.png", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, data, label in zip(
            axes, (solution, physical), ("Solution mesh", "Property mesh")
        ):
            depth = 2 if np.ptp(data.points[:, 2]) else 1
            # Solution cells can include high-order edge/interior nodes. The
            # straight-sided demo's boundary is their convex hull, not the
            # node storage order used by VTK's higher-order cell types.
            polygons = []
            for i in range(data.n_cells):
                points = data.get_cell(i).points[:, [0, depth]]
                polygons.append(points[ConvexHull(points).vertices])
            ax.add_collection(
                PolyCollection(
                    polygons, facecolor="#edf4fc", edgecolor="#34506b", linewidth=0.6
                )
            )
            ax.set(
                xlim=(0, 1.2),
                ylim=(0.8, 0),
                xlabel="x [km]",
                ylabel="depth [km]",
                title=f"{label}: {data.n_cells} leaf cells",
            )
        fig.savefig(output / "mesh_topology.png", dpi=150)
        plt.close(fig)
        checks = {
            "property_cells": physical.n_cells,
            "solution_cells": solution.n_cells,
            "independent_coefficients": geometry.basis.shape[1],
            "display_vertices": mesh.n_points,
            "hanging_vertex_occurrences": int(hanging.sum()),
            "native_property_max_error_km_s": error,
            "grid_property_max_error_km_s": grid_error,
            "gradient_norm": gradient.norm(),
            "gram_derivative": "total",
            "property_vtu": str(property_file),
        }
        (output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.solver.resolve(), args.output.resolve())
