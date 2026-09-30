# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Native root extraction, PML and all-root forward parity through LocalSite."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from frequensolve.mesh import BoundaryCondition
from frequensolve.mesh._root_patch import RootPatchDescriptor
from frequensolve.mesh.patches import PatchSet
from frequensolve.model.layered import LayeredModel
from frequensolve.model.parameterization import (
    MeshControl,
    MeshPropertySpace,
    ParameterizedProperty,
)
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project import Project
from frequensolve.seismic import Acquisition, ReceiverNode
from frequensolve.simulation import Discretization, FrequencyDomainJob, SolverConfig
from frequensolve.simulation.artifact_contract import (
    TaskResult,
    load_operation_result,
    task_result_path,
)
from frequensolve.simulation.jobs._patches import PatchPreparationJob
from frequensolve.units import ureg
from tests.test_imaging_jobs import CONTRACT_ROOT, VALIDATOR

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1200)]


def _solver(dimension):
    value = os.environ.get(f"FS_PATCH_SOLVER_{dimension}D")
    if not value or not Path(value).is_file():
        pytest.skip(f"Set FS_PATCH_SOLVER_{dimension}D to a local solver executable")
    return Path(value)


def _parent(
    tmp_path, dimension, physics, *, curved=False, root_columns=4, mesh_space=None
):
    project = Project(name="patches", path=tmp_path / "project", load_if_exists=False)
    simulation = project.new_simulation(
        name="parent", dimension=dimension, physics=physics
    )
    model = LayeredModel(
        dimension=dimension,
        x_limits=[0, 1],
        **({"y_limits": [0, 0.4]} if dimension == 3 else {}),
    )
    offset = 0
    if curved:
        x = np.linspace(0, 1, 33)
        values = 0.02 * np.cos(2 * np.pi * x)
        coordinates = {"x": x}
        if dimension == 3:
            y = np.linspace(0, 0.4, 17)
            values = values[:, None] + 0.01 * np.cos(2 * np.pi * y[None, :] / 0.4)
            coordinates["y"] = y
        offset = xr.DataArray(values, dims=list(coordinates), coords=coordinates)

    def vp(value, layer):
        # Optionally put every layer's Vp on one named Sauce mesh property space.
        if mesh_space is None:
            return value
        return ParameterizedProperty(
            value, id=f"vp_{layer}", control=MeshControl(mesh_space)
        )

    model.add_surface(name="top", depth=offset)
    model.add_layer(
        name="fluid", physics="acoustic", properties={"Vp": vp(1.5, "fluid"), "Rho": 1}
    )
    model.add_surface(name="interface", depth=offset + 0.25)
    properties = {"Vp": vp(2.5, "lower"), "Rho": 2}
    if physics == "coupled":
        properties["Vs"] = 1.2
    model.add_layer(
        name="lower",
        physics="elastic" if physics == "coupled" else "acoustic",
        properties=properties,
    )
    model.add_surface(name="bottom", depth=offset + 0.6)
    simulation += model
    if root_columns is None:
        # No generator: Sauce sizes the mesh from f_list and keys its artifacts.
        simulation.mesh.set_adapt(elems_per_wave=0.5, order=2, f_low=3)
    else:
        simulation += model.hex_mesh_generator(
            n=[root_columns, 2] if dimension == 2 else [root_columns, 2, 2]
        )
        simulation.mesh.set_adapt(elems_per_wave=0.5, order=2, f_low=3, f_high=3)
    simulation += BoundaryCondition(conditions=["free"], boundaries=["z_min"])
    boundaries = ["x_min", "x_max", "z_max"] + (
        ["y_min", "y_max"] if dimension == 3 else []
    )
    simulation += BoundaryCondition(
        conditions=["pml"],
        boundaries=boundaries,
        pml_wavelengths=1,
        pml_reflection=1e-4,
    )
    acquisition = Acquisition()
    acquisition.add_sources(
        kind="scalar", coords=[[0.2, 0.08]] if dimension == 2 else [[0.2, 0.2, 0.08]]
    )
    device = ReceiverNode(name="hydrophone")
    device.add_component(name="p", field="pressure")
    coords = [[x, 0.05] if dimension == 2 else [x, 0.2, 0.05] for x in (0.1, 0.2, 0.3)]
    acquisition.add_receiver_group(name="surface", device=device, coords=coords)
    simulation += acquisition
    simulation += Discretization()
    simulation += SolverConfig(
        precision="single", grids=1, mumps_precision="single", tolerance=1e-4
    )
    simulation.save()
    return simulation


def _child(parent, preparation, index, name):
    report = preparation.geometry_report
    child = parent.copy(name)
    child.mesh.mesh = None
    child.mesh.file = str(preparation._result_path / report["parent_file"])
    child.mesh.format = "gmp"
    child.mesh.root_patch = RootPatchDescriptor.from_fs(
        report["patches"][index]["descriptor"]
    )
    if report["patches"][index]["cut_boundary"]:
        child += BoundaryCondition(
            conditions=["pml"],
            boundaries=["patch_cut"],
            pml_wavelengths=1,
            pml_reflection=1e-4,
        )
    return child


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("physics", ["acoustic", "coupled"])
def test_root_patch_preparation_and_forward_parity(tmp_path, dimension, physics):
    executable = _solver(dimension)
    parent = _parent(tmp_path, dimension, physics)
    with LocalSite(
        solver=executable,
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        lower = [0] * dimension
        upper = [1000, 600] if dimension == 2 else [1000, 400, 600]
        patch_upper = list(upper)
        patch_upper[0] = 300
        preparation = PatchPreparationJob(
            "prepare",
            parent,
            [3, 4],
            {
                "units": "m",
                "patches": [
                    {"name": "all", "lower": lower, "upper": upper, "padding": 0},
                    {
                        "name": "left",
                        "lower": lower,
                        "upper": patch_upper,
                        "padding": 0,
                    },
                ],
            },
        )
        assert site.run(preparation, check=True).successful
        assert preparation.run_state()["task_summary"]["failed"] == 0
        report = preparation.geometry_report
        schema = json.loads(
            (CONTRACT_ROOT / "outputs/fs-patch-geometry-1/schema.json").read_text()
        )
        VALIDATOR.evolve(schema=schema).validate(report)
        full, cut = report["patches"]
        assert full["descriptor"]["roots"] == list(range(1, report["root_count"] + 1))
        assert not full["cut_boundary"]
        assert 0 < cut["root_fraction"] < 1
        assert cut["cut_boundary"] > 0
        assert preparation.plan_tasks()["pending_indices"] == []
        assert preparation.plan_tasks(force=True)["pending_indices"] == [0]
        distributed = PatchPreparationJob(
            "prepare_mpi", parent, [3, 4], preparation.request
        )
        assert site.run(distributed, check=True, procs_per_job=2).successful
        mpi_report = distributed.geometry_report
        assert mpi_report["parent_fingerprint"] == report["parent_fingerprint"]
        assert mpi_report["roots"] == report["roots"]
        assert mpi_report["patches"] == report["patches"]
        results = []
        for name, simulation in (
            ("parent", parent),
            ("all", _child(parent, preparation, 0, "all")),
            ("left", _child(parent, preparation, 1, "left")),
        ):
            job = FrequencyDomainJob("forward", simulation, [3])
            assert site.run(job, check=True).successful
            values = job.traces.open().fd("surface", "p").values
            assert np.all(np.isfinite(values))
            assert np.linalg.norm(values) > 0
            results.append(values)
            if name == "left":
                mpi_job = FrequencyDomainJob("forward_mpi", simulation, [3])
                ranks = 4 if dimension == 3 and physics == "coupled" else 2
                assert site.run(mpi_job, check=True, procs_per_job=ranks).successful
                mpi_values = mpi_job.traces.open().fd("surface", "p").values
                mpi_error = np.linalg.norm(mpi_values - values) / np.linalg.norm(values)
                assert mpi_error < 2e-4
        error = np.linalg.norm(results[1] - results[0]) / np.linalg.norm(results[0])
        assert error < 1e-4
        assert set(report["roots"]) == {
            "cell",
            "domain",
            "lower",
            "upper",
            "sampled_lower",
            "sampled_upper",
        }
        assert np.shape(report["roots"]["lower"]) == (dimension, report["root_count"])
        prepared = PatchSet.around_sources(
            shots_per_patch=1, max_offset=120 * ureg.m, padding=0 * ureg.m
        ).prepare(parent, [3, 4], site=site, edge_samples=True)
        assert prepared.acquisition[0]["receivers"]["surface"] == {1: (1, 2, 3)}
        assert prepared.geometry["parent_fingerprint"] == report["parent_fingerprint"]
        VALIDATOR.evolve(schema=schema).validate(prepared.geometry)
        roots = prepared.geometry["roots"]
        assert roots["edge_count"] == [4 if dimension == 2 else 12] * len(roots["cell"])
        assert np.shape(roots["edge_points"]) == (
            dimension,
            9 * sum(roots["edge_count"]),
        )
        assert all(
            item["preview_float64_payload_bytes"] > 0
            for item in prepared.storage_estimates["patches"]
        )
        assert (
            prepared.geometry["patches"][0]["descriptor"]["roots"]
            == cut["descriptor"]["roots"]
        )
        assert len(prepared.jobs) == 2
        with pytest.raises(ValueError, match="multiple jobs"):
            _ = prepared.job


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("physics", ["acoustic", "coupled"])
def test_preparation_closes_material_supports_before_buffering(
    tmp_path, dimension, physics
):
    import h5py

    parent = _parent(tmp_path, dimension, physics, mesh_space="vp")
    artifact = Path(parent.project_path) / "parent-support.h5"
    parent.model.property_spaces["vp"] = MeshPropertySpace(
        artifact=artifact.name if dimension == 2 else str(artifact),
        frequency=3,
        epw=1,
    )
    parent.save()
    with LocalSite(
        solver=_solver(dimension),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        inventory = PatchPreparationJob("inventory", parent, [3], {"units": "m"})
        assert site.run(inventory, check=True).successful
        report = inventory.geometry_report
        assert not artifact.exists()  # Inventory must not generate a material basis.
        assert not (Path(parent._file).parent / "root_point_mapping.h5").exists()
        # The solver's own first use of the mesh space builds the parent basis.
        basis = FrequencyDomainJob("basis", parent, [3])
        assert site.run(basis, check=True).successful
        assert artifact.exists()
        # A sized mesh uses the declared (literal) path, and the run records it.
        (record,) = TaskResult.read(
            task_result_path(basis._result_path, 1), result_path=basis._result_path
        ).material_artifacts
        assert record.resolve(parent.project_path) == artifact.resolve()
        assert (record.space, record.mesh_keyed, record.mesh_key) == ("vp", False, None)
        with h5py.File(artifact, "r") as h5:
            space = h5["property_space"]
            identity = space["identity"][()].decode().strip()
            directory = space["root_directory"][()]
            root_offset = space["root_control_offset"][()]
            root_controls = space["root_control_ids"][()]
            control_offset = space["control_root_offset"][()]
            control_roots = space["control_roots"][()]
        # Rows are (canonical root, GMP cell, material-layer slot, kind); the slot
        # is domain_to_layer(domain), the position of the GMP domain's subdomain.
        subdomains = json.loads(Path(parent._file).read_text())["Model"]["subdomains"]
        layer = {item["mesh_block_id"]: i + 1 for i, item in enumerate(subdomains)}
        slot_of_cell = {int(row[1]): slot for slot, row in enumerate(directory)}
        physical_slots = [slot_of_cell[cell] for cell in report["roots"]["cell"]]
        assert [directory[slot, 2] for slot in physical_slots] == [
            layer[domain] for domain in report["roots"]["domain"]
        ]
        canonical = [int(directory[slot, 0]) for slot in physical_slots]
        physical = {root: i + 1 for i, root in enumerate(canonical)}
        slot_of_root = {int(row[0]): slot for slot, row in enumerate(directory)}

        # 0-based CSR tables in directory-slot order; control IDs are 1-based rows.
        def controls_of(roots):
            return sorted(
                {
                    int(control)
                    for root in roots
                    for control in root_controls[
                        root_offset[slot_of_root[root]] : root_offset[
                            slot_of_root[root] + 1
                        ]
                    ]
                }
            )

        def roots_of(controls):
            return sorted(
                {
                    int(root)
                    for control in controls
                    for root in control_roots[
                        control_offset[control - 1] : control_offset[control]
                    ]
                }
            )

        core = 1
        core_controls = controls_of([canonical[core - 1]])
        closure = roots_of(core_controls)
        support_ids = sorted(physical[root] for root in closure)
        # The fixture must distinguish one closure step from recursive growth.
        assert core in support_ids and len(support_ids) > 1
        assert len(roots_of(controls_of(closure))) > len(closure)
        artifact_bytes = artifact.read_bytes()
        request = {
            "units": "m",
            "patches": [
                {"name": "core", "roots": [core], "padding": 0},
                {"name": "buffered", "roots": [core], "padding": 1},
            ],
        }
        preparation = PatchPreparationJob("supports", parent, [3], request)
        assert site.run(preparation, check=True).successful
        output = preparation.geometry_report
        assert artifact.read_bytes() == artifact_bytes
        schema = json.loads(
            (CONTRACT_ROOT / "outputs/fs-patch-geometry-1/schema.json").read_text()
        )
        VALIDATOR.evolve(schema=schema).validate(output)
        selected, buffered = output["patches"]
        assert selected["core_roots"] == [core]
        # Support growth closes the core coefficients' supports once; it must not
        # activate the other coefficients of the added support roots.
        assert selected["support_roots"] == selected["buffer_roots"] == support_ids
        coverage = selected["material_coverage"][0]
        assert coverage["basis_identity"] == identity
        assert coverage["control_ids"] == core_controls
        assert dict(zip(coverage["physical_roots"], coverage["material_roots"])) == {
            physical[root]: root for root in closure
        }
        assert buffered["material_coverage"] == selected["material_coverage"]
        assert set(support_ids) <= set(buffered["buffer_roots"])
        assert len(buffered["buffer_roots"]) > len(support_ids)
        distributed = PatchPreparationJob("supports_mpi", parent, [3], request)
        assert site.run(distributed, check=True, procs_per_job=2).successful
        assert distributed.geometry_report["patches"] == output["patches"]
        assert preparation.results_exist()
        with h5py.File(artifact, "r+") as h5:
            h5["property_space/identity"][()] = np.bytes_("changed-basis")
        assert not preparation.results_exist()


@pytest.mark.parametrize("dimension", [2, 3])
def test_preparation_reads_mesh_keyed_material_artifact(tmp_path, dimension):
    import h5py

    parent = _parent(
        tmp_path, dimension, "acoustic", root_columns=None, mesh_space="vp"
    )
    literal = Path(parent.project_path) / "parent-support.h5"
    parent.model.property_spaces["vp"] = MeshPropertySpace(
        artifact=literal.name, frequency=3, epw=1
    )
    parent.save()
    with LocalSite(
        solver=_solver(dimension),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        # Two bands key two generated meshes, so two artifacts coexist.
        records = {}
        for name, band in (("basis", [3]), ("basis_high", [3, 4])):
            basis = FrequencyDomainJob(name, parent, band)
            assert site.run(basis, check=True).successful
            task = TaskResult.read(
                task_result_path(basis._result_path, 1),
                result_path=basis._result_path,
            )
            (records[max(band)],) = task.material_artifacts
        assert not literal.exists()
        written = sorted(literal.parent.glob("parent-support.*.h5"))
        resolved = {
            band: record.resolve(parent.project_path)
            for band, record in records.items()
        }
        assert written == sorted(resolved.values())
        assert records[3].mesh_key != records[4].mesh_key
        for band, record in records.items():
            assert record.space == "vp" and record.mesh_keyed
            assert record.sizing_frequency == band
            assert resolved[band] == record.path.resolve()
            assert resolved[band].name == f"parent-support.{record.mesh_key}.h5"
            assert record.bytes == resolved[band].stat().st_size
            with h5py.File(resolved[band], "r") as h5:
                identity = h5["property_space/identity"][()].decode().strip()
            assert record.basis_identity == identity
        request = {
            "units": "m",
            "patches": [{"name": "core", "roots": [1], "padding": 0}],
        }
        for name, band in (("keyed_high", [3, 4]), ("keyed", [3])):
            preparation = PatchPreparationJob(name, parent, band, request)
            record = records[max(band)]
            assert preparation._input_fingerprint_payload()["vp"] == {
                **preparation._path_content_fingerprint(resolved[max(band)]),
                "basis_identity": record.basis_identity,
            }
            assert site.run(preparation, check=True).successful
            # geometry_report also verifies patch_prepare read the selected file.
            report = preparation.geometry_report
            coverage = report["patches"][0]["material_coverage"]
            assert coverage[0]["basis_identity"] == record.basis_identity
            operation = load_operation_result(preparation._result_path, "patch_prepare")
            (used,) = operation.material_artifacts
            assert used.resolve(parent.project_path) == resolved[max(band)]
            assert (used.mesh_key, used.built) == (record.mesh_key, False)
        assert preparation.results_exist()
        with h5py.File(resolved[3], "r+") as h5:
            h5["property_space/identity"][()] = np.bytes_("changed-basis")
        assert not preparation.results_exist()


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("physics", ["acoustic", "coupled"])
def test_curved_patch_checks_actual_point_containment(tmp_path, dimension, physics):
    from copy import deepcopy

    from frequensolve.mesh.patches import PatchSet
    from frequensolve.units import ureg

    parent = _parent(tmp_path, dimension, physics, curved=True, root_columns=8)
    with LocalSite(
        solver=_solver(dimension),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        inventory = PatchPreparationJob("inventory", parent, [3], {"units": "m"})
        assert site.run(inventory, check=True).successful
        report = inventory.geometry_report
        inside = [200, 80] if dimension == 2 else [200, 200, 80]
        boundary = [0, 20] if dimension == 2 else [0, 200, 10]
        exterior = [10, 18] if dimension == 2 else [10, 200, 8]
        # This point is outside the curved free surface but inside a root's AABB.
        sampled_lower = np.array(report["roots"]["sampled_lower"]).T
        sampled_upper = np.array(report["roots"]["sampled_upper"]).T
        assert np.any(
            np.all(sampled_lower <= exterior, axis=1)
            & np.all(sampled_upper >= exterior, axis=1)
        )
        assert "edge_points" not in report["roots"]
        request = {
            "units": "m",
            "edge_samples": True,
            "patches": [
                {
                    "name": "all",
                    "roots": list(range(1, report["root_count"] + 1)),
                    "padding": 0,
                    "points": [
                        {"kind": "source", "id": 7, "coordinates": inside},
                        {
                            "kind": "receiver",
                            "id": 9,
                            "group": "surface",
                            "coordinates": boundary,
                        },
                    ],
                }
            ],
        }
        valid = PatchPreparationJob("contained", parent, [3], request)
        assert site.run(valid, check=True).successful
        VALIDATOR.evolve(
            schema=json.loads(
                (CONTRACT_ROOT / "outputs/fs-patch-geometry-1/schema.json").read_text()
            )
        ).validate(valid.geometry_report)
        edge_count = valid.geometry_report["roots"]["edge_count"]
        edges = np.asarray(valid.geometry_report["roots"]["edge_points"]).reshape(
            dimension, sum(edge_count), 9
        )
        # Curved roots: interior samples leave the chord between edge ends.
        assert (
            np.max(np.abs(edges[:, :, 4] - 0.5 * (edges[:, :, 0] + edges[:, :, 8]))) > 1
        )
        patch = valid.geometry_report["patches"][0]
        assert patch["acquisition_checked"]
        assert len(patch["point_roots"]) == 2
        assert set(patch["point_roots"]) <= set(patch["descriptor"]["roots"])
        mpi = PatchPreparationJob("contained_mpi", parent, [3], request)
        assert site.run(mpi, check=True, procs_per_job=2).successful
        assert mpi.geometry_report["patches"] == valid.geometry_report["patches"]
        prepared = PatchSet.around_sources(
            shots_per_patch=1, max_offset=120 * ureg.m, padding=0 * ureg.m
        ).prepare(parent, [3, 7.5], site=site)
        assert prepared.geometry["patches"][0]["acquisition_checked"]
        assert len(prepared.geometry["patches"][0]["point_roots"]) == 4
        for kind, coordinates in (("source", exterior), ("receiver", exterior)):
            bad_request = deepcopy(request)
            index = 0 if kind == "source" else 1
            bad_request["patches"][0]["points"][index]["coordinates"] = coordinates
            bad = PatchPreparationJob(f"outside_{kind}", parent, [3], bad_request)
            site.run(bad, check=False)
            errors = list(Path(bad._result_path).glob("_fs_run/**/error.json"))
            assert any(f"Cannot locate {kind}" in path.read_text() for path in errors)
            assert not bad.results_exist()
        omitted = deepcopy(request)
        omitted["patches"][0]["roots"] = [report["root_count"]]
        bad = PatchPreparationJob("omitted_source_root", parent, [3], omitted)
        site.run(bad, check=False)
        errors = list(Path(bad._result_path).glob("_fs_run/**/error.json"))
        assert any("Cannot locate source 7" in path.read_text() for path in errors)
        assert not bad.results_exist()


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("physics", ["acoustic", "coupled"])
@pytest.mark.parametrize("cut", [False, True], ids=["all_roots", "cut"])
def test_patch_children_preserve_global_acquisition_ids(
    tmp_path, dimension, physics, cut
):
    from frequensolve.mesh.patches import Patch
    from frequensolve.seismic.sources import SourceGeometry

    parent = _parent(tmp_path, dimension, physics, root_columns=8)
    parent.acquisition.source_geometry = SourceGeometry.points(
        kind="scalar",
        coords=[
            [x, 0.08] if dimension == 2 else [x, 0.2, 0.08] for x in (0.2, 0.8, 0.25)
        ],
    )
    from frequensolve.seismic.receivers import CoordsArray

    parent.acquisition.receiver_groups[0].coordinates = CoordsArray(
        coordinates=np.array(
            [
                [x, 0.05] if dimension == 2 else [x, 0.2, 0.05]
                for x in (0.1, 0.2, 0.3, 0.7, 0.8, 0.9)
            ]
        )
    )
    if cut:
        parent.acquisition.add_receiver_group(
            name="excluded",
            device=parent.acquisition.receiver_groups[0].device,
            coords=[[0.5, 0.05]] if dimension == 2 else [[0.5, 0.2, 0.05]],
        )
    with LocalSite(
        solver=_solver(dimension),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        inventory = PatchPreparationJob("catalog", parent, [3], {"units": "m"})
        assert site.run(inventory, check=True).successful
        roots = tuple(range(1, inventory.geometry_report["root_count"] + 1))
        # Axis-major bounds: index [0] is the x extent of every root.
        x_lower = inventory.geometry_report["roots"]["sampled_lower"][0]
        x_upper = inventory.geometry_report["roots"]["sampled_upper"][0]
        west = (
            tuple(root for root, x in zip(roots, x_upper) if x <= 501) if cut else roots
        )
        east = (
            tuple(root for root, x in zip(roots, x_lower) if x >= 499) if cut else roots
        )
        patches = PatchSet(
            [
                Patch(name="west", roots=west, sources=(1, 3)),
                Patch(name="east", roots=east, sources=(2,)),
            ],
            max_offset=np.full(dimension, 120) * ureg.m,
            padding=0 * ureg.m,
        )
        reference = FrequencyDomainJob("full", parent, [3])
        if not cut:
            assert site.run(reference, check=True).successful
        composite = FrequencyDomainJob("selected", parent, [3], patches=patches)
        result = site.run(composite, check=True, procs_per_job=2)
        assert result.successful
        prepared = result.prepared
        if cut:
            assert all(p["cut_boundary"] for p in prepared.geometry["patches"])
            assert all(p["root_fraction"] < 1 for p in prepared.geometry["patches"])
        children = tuple(job.simulation for job in result.jobs)
        assert len(children) == 2
        for job, child, selection in zip(result.jobs, children, prepared.acquisition):
            assert [group.name for group in child.acquisition.receiver_groups] == [
                "surface"
            ]
            assert child.acquisition.known_source_field_count() == 3
            assert child.acquisition.extra["active_sources"] == list(
                selection["sources"]
            )
            traces = child.acquisition.surveys[0].traces
            assert {(t.source_id, t.receiver_id) for t in traces} == {
                (s, r)
                for s, receivers in selection["receivers"]["surface"].items()
                for r in receivers
            }
            for source in selection["sources"]:
                actual = job.traces.open().fd("surface", "p", source=source)
                receivers = selection["receivers"]["surface"][source]
                assert len(receivers) > 0
                assert actual.values.size == len(receivers)
                assert np.all(np.isfinite(actual.values))
                assert np.linalg.norm(actual.values) > 0
                if cut:
                    continue
                expected = reference.traces.open().fd("surface", "p", source=source)
                np.testing.assert_allclose(
                    actual.values.reshape(-1),
                    expected.values.reshape(-1)[np.array(receivers) - 1],
                    rtol=3e-4,
                    atol=1e-10,
                )
        records = [
            Path(job._result_path) / "_fs_run/tasks/task_000001/result.json"
            for job in result.jobs
        ] + [
            Path(job._result_path) / "_fs_run/operations/patch_prepare/result.json"
            for job in prepared.jobs
        ]
        times = [path.stat().st_mtime_ns for path in records]
        restored = FrequencyDomainJob.load(composite.job_file)
        repeated = site.run(restored, check=True, procs_per_job=2)
        assert repeated.successful
        assert repeated.prepared.geometry == prepared.geometry
        assert [path.stat().st_mtime_ns for path in records] == times
        if not cut and physics == "acoustic" and dimension == 2:
            # Changes must still invalidate committed inputs. A wider aperture
            # changes child rows; a model change also invalidates preparation.
            restored.patches.max_offset = tuple(150 for _ in range(dimension))
            changed_policy = site.run(restored, check=True, procs_per_job=2)
            assert changed_policy.successful
            assert changed_policy.prepared.acquisition != prepared.acquisition
            assert [path.stat().st_mtime_ns for path in records[:2]] != times[:2]
            inventory_record = records[-2]
            inventory_time = inventory_record.stat().st_mtime_ns
            restored.simulation.model.layers["lower"].properties["Vp"] = 2.6
            changed_model = site.run(restored, check=True, procs_per_job=2)
            assert changed_model.successful
            assert inventory_record.stat().st_mtime_ns != inventory_time
