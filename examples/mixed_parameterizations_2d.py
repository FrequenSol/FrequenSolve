"""Run and plot a real 2D Sauce problem with four control families.

Run from a FrequenSolve checkout with visual dependencies and a local imaging
solver, e.g.:

    python examples/mixed_parameterizations_2d.py --solver /path/to/fs2d_s \
        --output .codex/mixed-parameterizations

The fresh output directory contains initial/truth projects, solver-produced
physical material grids, exact adjoint covectors, figures and checks.json.
No synthetic gradients are used. Physical material is evaluated by Sauce on
its output mesh and interpolated to a display grid by PyVista. DepthProfile
is the public API for 1D depth controls. The salt host uses
B-spline density controls and an RBF-blended velocity; the layer above uses hat
velocity controls, and the layer below uses a tensor-hat velocity lattice.
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

import frequensolve as fs
from frequensolve import imaging as im
from frequensolve.geometry.grids import CartesianGrid
from frequensolve.model import BlendProperty
from frequensolve.orchestrator.sites.local import LocalSite

FREQUENCIES = [4.0, 6.0]
KEYS = ("shallow_vp", "host_rho", "deep_vp", "salt")


def controls():
    """Use different parameterizations on distinct physical properties."""
    return im.ControlSpace(
        shallow_vp=im.DepthProfile("vp", "shallow", datum="global", count=5),
        host_rho=im.DepthProfile.bspline("rho", "salt_host", datum="global", count=6),
        deep_vp=im.GridParameters("vp", "deep", shape=[9, 4]),
        salt=im.InterfaceParameters("salt_boundary", maximum_displacement=0.04),
    )


def simulation(project, name):
    sim = project.new_simulation(
        name=name,
        physics="acoustic",
        dimension=2,
        units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
    )
    model = fs.LayeredModel(name="mixed", dimension=2, x_limits=[0, 1.2])
    model.add_surface(0.0, name="top")
    model.add_layer(name="shallow", properties={"vp": 1.8, "rho": 1.8})
    model.add_surface(0.20, name="host_top")
    model.add_layer(
        name="salt_host",
        properties={
            "vp": BlendProperty("salt_boundary", width=0.025, inside=4.2, outside=2.5),
            "rho": 2.15,
        },
    )
    model.add_surface(0.60, name="host_bottom")
    model.add_layer(name="deep", properties={"vp": 3.0, "rho": 2.4})
    model.add_surface(0.85, name="bottom")
    model += fs.RBFSurface(
        name="salt_boundary",
        support_radius=0.38,
        bias=0.045,
        centers=[[0.46, 0.39], [0.64, 0.42], [0.78, 0.36]],
        coefficients=[-0.18, -0.20, -0.17],
    )
    sim += model
    sim += model.hex_mesh_generator([12, 9])
    sim.mesh.set_adapt(
        elems_per_wave=2.0,
        order=4,
        f_low=min(FREQUENCIES),
        f_high=max(FREQUENCIES),
        adapt_order=False,
    )
    sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
    sim += fs.BoundaryCondition(
        conditions=["pml"],
        boundaries=["x_min", "x_max", "z_max"],
        pml_wavelengths=1.2,
        pml_exponent=3.0,
        pml_constant=20.0,
    )
    acquisition = fs.Acquisition()
    acquisition.add_sources(kind="scalar", coords=[[0.25, 0.04], [0.95, 0.04]])
    receiver = fs.ReceiverNode(name="hydrophone")
    receiver.add_component(name="p", field="pressure")
    acquisition.add_receiver_group(
        name="surface",
        device=receiver,
        coords=[[x, 0.035] for x in np.linspace(0.08, 1.12, 27)],
    )
    sim += acquisition
    sim += fs.Discretization()
    # The single-precision multigrid build reaches a residual floor near 1e-5.
    sim += fs.SolverConfig(solve_on="final", max_iter=500, tolerance=1e-4)
    return sim


def coefficients(space, truth=False):
    """Authored model coefficients, separate from derivatives computed later."""
    z = space.block("shallow_vp").control.coordinates
    rho_z = space.block("host_rho").control.coordinates
    nodes = space.block("deep_vp").control.coordinates
    strength = 1.0 if truth else 0.55
    return space.pack(
        {
            "shallow_vp": strength * (0.05 + 0.20 * z / 0.2),
            "host_rho": strength * (0.08 * np.sin(np.pi * (rho_z - 0.2) / 0.4)),
            "deep_vp": strength
            * (
                0.25
                * np.sin(np.pi * nodes[:, 0] / 1.2)
                * np.sin(np.pi * (nodes[:, 1] - 0.6) / 0.25)
            ),
            "salt": np.array([-0.18, -0.20, -0.17])
            + (np.array([-0.025, 0.015, -0.020]) if truth else 0),
        }
    )


def install_state(bound, vector):
    """Install authored values using public model/control objects."""
    sim = bound.simulation
    for block in bound.resolved_blocks:
        if block.kind == "interface":
            surface = sim.model.implicit_surfaces[0]
            surface.coefficients = vector[block.name].copy()
        else:
            layer = next(s for s in sim.model.subdomains if s.name == block.subdomain)
            layer.properties[block.prop] = layer.properties[
                block.prop
            ].with_coefficients(vector[block.name])
    sim.save()
    return sim


def material_output():
    return fs.VtkOutput.volume(
        name="material",
        path="material",
        sources=[1],
        upscale=2,
        show_pml=False,
        items=[
            fs.output_property("vp", output_name="vp", units="km/s"),
            fs.output_property("rho", output_name="rho", units="g/cm^3"),
        ],
    )


def read_material(job):
    files = sorted(job.vtk_outputs["material"].rglob("*.vtu"))
    if not files:
        raise RuntimeError(f"No solver material grid below {job.vtk_outputs}")
    mesh = pv.read(files[0])
    if "vp" not in mesh.point_data:
        mesh = mesh.cell_data_to_point_data()
    # In 2D, Sauce embeds physical (x,z) in VTK's x-z plane.
    points = mesh.points
    vertical = 2 if np.ptp(points[:, 2]) else 1
    x, z = np.linspace(0.001, 1.199, 181), np.linspace(0.001, 0.849, 129)
    xx, zz = np.meshgrid(x, z)
    coords = np.zeros((xx.size, 3))
    coords[:, 0], coords[:, vertical] = xx.ravel(), zz.ravel()
    sampled = pv.PolyData(coords).sample(mesh)
    if not np.all(sampled["vtkValidPointMask"]):
        raise AssertionError("Solver output mesh does not cover the display grid")
    result = {}
    for name in ("vp", "rho"):
        array = np.asarray(sampled.point_data[name]).reshape(z.size, x.size)
        if not np.all(np.isfinite(array)):
            raise AssertionError(f"Non-finite solver material {name}")
        result[name] = array
    return x, z, result


def save_figure(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_models(initial, truth, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row, (name, unit) in enumerate((("vp", "km/s"), ("rho", "g/cm³"))):
        lo = min(initial[2][name].min(), truth[2][name].min())
        hi = max(initial[2][name].max(), truth[2][name].max())
        for col, (data, title) in enumerate(((initial, "Initial"), (truth, "Truth"))):
            artist = axes[row, col].pcolormesh(
                data[0],
                data[1],
                data[2][name],
                shading="auto",
                cmap="viridis",
                vmin=lo,
                vmax=hi,
            )
            axes[row, col].set_title(f"{title} {name} [{unit}]")
            fig.colorbar(artist, ax=axes[row, col], shrink=0.8)
        diff = truth[2][name] - initial[2][name]
        bound = max(float(np.max(np.abs(diff))), 1e-12)
        artist = axes[row, 2].pcolormesh(
            initial[0],
            initial[1],
            diff,
            shading="auto",
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
        )
        axes[row, 2].set_title(f"Truth − initial {name} [{unit}]")
        fig.colorbar(artist, ax=axes[row, 2], shrink=0.8)
    for ax in axes.flat:
        ax.set(xlabel="x [km]", ylabel="depth [km]", ylim=(0.85, 0))
        ax.axhline(0.2, color="white", lw=0.6, alpha=0.7)
        ax.axhline(0.6, color="white", lw=0.6, alpha=0.7)
    fig.suptitle(
        "Physical material evaluated by Sauce • hats / B-spline / grid / RBF salt",
        fontsize=14,
    )
    save_figure(fig, output / "models.png")


def plot_controls(state, gradient, output):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for col, key in enumerate(KEYS):
        state.plot(key, ax=axes[0, col])
        gradient.plot(
            key, ax=axes[1, col], **({"cmap": "RdBu_r"} if key == "deep_vp" else {})
        )
        axes[0, col].set_title(f"State coefficients: {key}")
        axes[1, col].set_title(f"Adjoint covector: {key}")
    fig.suptitle(
        "Native control plots • gradient = dΦ/d(coefficient), no smoothing", fontsize=14
    )
    save_figure(fig, output / "controls_and_gradients.png")


def display_grid(depth_limits=(0, 0.85)):
    return CartesianGrid(
        n=[181, 129],
        x0=[0, depth_limits[0]],
        x1=[1.2, depth_limits[1]],
        dims=["x", "z"],
        units="km",
    )


def plot_control_fields(state, gradient, output):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    titles = ("Hat depth profile", "B-spline depth profile", "Tensor-hat grid")
    limits = ((0, 0.2), (0.2, 0.6), (0.6, 0.85))
    for col, (key, title, depth) in enumerate(zip(KEYS[:3], titles, limits)):
        grid = display_grid(depth)
        for row, vector in enumerate((state, gradient)):
            field = vector.to_grid(grid, key)
            bound = max(float(np.nanmax(np.abs(field))), 1e-12)
            vector.plot(
                key,
                grid=grid,
                ax=axes[row, col],
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
                cbar_kwargs={
                    "label": (
                        (
                            "rho update [g/cm³]"
                            if key == "host_rho"
                            else "vp update [km/s]"
                        )
                        if row == 0
                        else "Basis expansion of dΦ/dc"
                    )
                },
            )
            axes[row, col].set(
                title=f"{title}: {'control field' if row == 0 else 'gradient display'}",
                xlabel="x [km]",
                ylabel="depth [km]",
                xlim=(0, 1.2),
                ylim=depth[::-1],
            )
    fig.suptitle(
        "Controls evaluated over each layer\n"
        "Gradient row expands coefficient derivatives in the same basis; it is not a physical gradient density",
        fontsize=12,
    )
    save_figure(fig, output / "control_fields_2d.png")

    space = state.space.without_support().restrict(["deep_vp"])
    one = space.zeros()
    block = space.block("deep_vp")
    # Sauce packs x fastest. Choose one interior node.
    one.values[len(block.coords["x"]) + len(block.coords["x"]) // 2] = 1
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    one.plot(
        "deep_vp",
        grid=display_grid(limits[2]),
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=1,
        cbar_kwargs={"label": "Basis function value"},
    )
    ax.set(
        title="One tensor-hat coefficient = 1; all others = 0\n"
        "Linear in x × linear in depth gives a 2D tent",
        xlabel="x [km]",
        ylabel="depth [km]",
        xlim=(0, 1.2),
        ylim=(0.85, 0.6),
    )
    save_figure(fig, output / "one_tensor_hat.png")


def check_refinement(problem, output):
    fine = problem.with_controls(
        {
            "shallow_vp": im.DepthProfile("vp", "shallow", datum="global", count=9),
            "deep_vp": im.GridParameters("vp", "deep", shape=[17, 7]),
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
    errors = {}
    for row, (key, limits) in enumerate(
        (("shallow_vp", (0, 0.2)), ("deep_vp", (0.6, 0.85)))
    ):
        grid = display_grid(limits)
        coarse = problem.state.to_grid(grid, key)
        refined = fine.state.to_grid(grid, key)
        diff = refined - coarse
        error = float(np.nanmax(np.abs(diff.values)))
        errors[key] = error
        assert error < 1e-8, (key, error)
        bound = max(float(np.nanmax(np.abs(coarse))), 1e-12)
        for col, (field, label) in enumerate(
            ((coarse, "Original"), (refined, "Refined"), (diff, "Difference"))
        ):
            field.plot.pcolormesh(
                ax=axes[row, col],
                x="x",
                y="z",
                yincrease=False,
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
                cbar_kwargs={"label": "vp update [km/s]"},
            )
            axes[row, col].set(
                title=f"{key}: {label}", xlim=(0, 1.2), ylim=limits[::-1]
            )
        axes[row, 2].set_title(f"Maximum change = {error:.2e} km/s")
    fig.suptitle("Refinement preserves the evaluated 2D fields (shared color scales)")
    save_figure(fig, output / "refinement.png")
    fine.state.save(output / "refined_state.h5")
    return {
        "profile_max_error": errors["shallow_vp"],
        "lattice_max_error": errors["deep_vp"],
        "old_size": problem.full_space.full_size,
        "new_size": fine.full_space.full_size,
    }


def run(solver, output):
    output.mkdir(parents=True, exist_ok=False)
    project = fs.Project(
        name="mixed_controls", path=output / "project", load_if_exists=False
    )
    bound = controls().bind(simulation(project, "truth"))
    truth = install_state(bound, coefficients(bound, truth=True))
    initial = simulation(project, "initial")
    with LocalSite(
        solver=solver,
        n_workers=1,
        threads_per_worker=4,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        observed = fs.FrequencyDomainJob("observed", truth, FREQUENCIES)
        print("Running true model / observed traces", flush=True)
        assert site.run(observed, check=True).successful
        truth_output = fs.FrequencyDomainJob(
            "truth_model",
            truth,
            FREQUENCIES[:1],
            outputs=[material_output(), fs.OutputUnits(geometry="km")],
        )
        assert site.run(truth_output, check=True).successful
        problem = im.ImagingProblem(
            initial,
            controls=controls(),
            observed=im.ObservedData(observed),
            site=site,
            name="mixed",
            min_support=0,
        )
        problem.state = problem.state.with_update(coefficients(problem.full_space))
        current = problem.simulation_at()
        forward = fs.FrequencyDomainJob(
            "initial_model",
            current,
            FREQUENCIES[:1],
            outputs=[material_output(), fs.OutputUnits(geometry="km")],
        )
        print("Running initial model / physical material output", flush=True)
        assert site.run(forward, check=True).successful
        plot_models(read_material(forward), read_material(truth_output), output)
        print("Computing joint adjoint gradients", flush=True)
        lin = problem.linearize()
        gradient = lin.gradient
        assert gradient is not None and np.all(np.isfinite(gradient.values))
        norms = {key: float(np.linalg.norm(gradient[key])) for key in KEYS}
        assert all(value > 0 for value in norms.values()), norms
        plot_controls(problem.state, gradient, output)
        plot_control_fields(problem.state, gradient, output)
        problem.state.save(output / "state.h5")
        gradient.save(
            output / "gradient.h5",
            state_fingerprint=lin.state_fingerprint,
            registry_fingerprint=lin.registry_fingerprint,
        )
        print("Checking Jacobian adjoint identity and refinement", flush=True)
        dot = lin.jacobian.dot_test(seed=7, tolerance=5e-3)
        assert dot["passed"], dot
        residual = lin.objective_residual()
        block_checks = {}
        for key in KEYS:
            # Isolate each block so the large shallow gradient cannot hide a
            # broken grid, spline or interface adjoint in the combined test.
            direction = lin.space.zeros()
            sl = lin.space.slices[lin.space.block(key).name]
            direction.values[sl] = gradient.values[sl]
            direction = direction / direction.norm()
            lhs = lin.jvp(direction).dot(residual)
            rhs = gradient.dot(direction)
            error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
            block_checks[key] = {"relative_error": error, "passed": error < 5e-3}
            assert block_checks[key]["passed"], (key, lhs, rhs, error)
        refinement = check_refinement(problem, output)
        checks = {
            "gradient_provenance": "Sauce FWI adjoint, no synthetic data-space surrogate",
            "solver": str(solver),
            "frequencies_hz": FREQUENCIES,
            "objective": float(lin.value),
            "block_gradient_norms": norms,
            "adjoint_test": dot,
            "block_adjoint_tests": block_checks,
            "refinement": refinement,
            "figures": [
                "models.png",
                "controls_and_gradients.png",
                "control_fields_2d.png",
                "one_tensor_hat.png",
                "refinement.png",
            ],
        }
        (output / "checks.json").write_text(
            json.dumps(checks, indent=2, default=str) + "\n"
        )
        print(json.dumps(checks, indent=2, default=str), flush=True)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh output directory"
    )
    args = parser.parse_args()
    run(args.solver.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
