# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

import frequensolve as fs
from frequensolve.mesh.patches import _bisect_sources
from frequensolve.seismic import Acquisition, ReceiverNode


def _simulation(tmp_path):
    simulation = fs.SeismicSimulation(
        name="parent", physics="acoustic", dimension=2, project_path=tmp_path
    )
    acquisition = Acquisition()
    acquisition.add_sources(
        kind="scalar", coords=[[0, 0.1], [1, 0.1], [1, 0.1], [3, 0.1]]
    )
    device = ReceiverNode(name="hydrophones")
    device.add_component(name="p", field="pressure")
    acquisition.add_receiver_group(
        name="near", device=device, coords=[[0.1, 0.1], [0.9, 0.1], [3.1, 0.1]]
    )
    acquisition.add_receiver_group(name="far", device=device, coords=[[10, 0.1]])
    simulation += acquisition
    return simulation


def _inventory():
    # Axis-major inventory arrays: one list per dimension with a value per root.
    return {
        "root_count": 4,
        "roots": {
            "cell": [10 * (i + 1) for i in range(4)],
            "domain": [1] * 4,
            "sampled_lower": [[i * 1000 for i in range(4)], [0] * 4],
            "sampled_upper": [[(i + 1) * 1000 for i in range(4)], [2000] * 4],
        },
    }


def test_deterministic_bisection_bounds_group_size_and_breaks_ties():
    points = np.array([[0, 0], [1, 0], [1, 0], [3, 0]])
    assert _bisect_sources(points, 2) == ((1, 2), (3, 4))
    for count in range(1, 40):
        points = np.zeros((count, 3))
        groups = _bisect_sources(points, 3)
        assert max(map(len, groups)) <= 3
        assert [source for group in groups for source in group] == list(
            range(1, count + 1)
        )


def test_aperture_is_per_shot_and_preserves_group_identity(tmp_path):
    simulation = _simulation(tmp_path)
    patches = fs.PatchSet.around_sources(
        shots_per_patch=2, max_offset=200 * fs.ureg.m, padding=0.3 * fs.ureg.km
    )
    request, selection = patches._requests(simulation, _inventory())
    assert selection[0]["sources"] == (1, 2)
    assert selection[0]["receivers"]["near"] == {1: (1,), 2: (2,)}
    assert selection[0]["receivers"]["far"] == {1: (), 2: ()}
    assert selection[1]["receivers"]["near"] == {3: (2,), 4: (3,)}
    assert selection[0]["retained_pairs"] == 2
    assert selection[0]["excluded_pairs"] == 6
    assert request["patches"][0]["padding"] == 300
    assert request["patches"][0]["lower"] == [0, 0]
    assert request["patches"][0]["upper"] == [1000, 2000]
    assert request["patches"][0]["points"] == [
        {"kind": "source", "id": 1, "coordinates": [0, 100]},
        {"kind": "source", "id": 2, "coordinates": [1000, 100]},
        {"kind": "receiver", "group": "near", "id": 1, "coordinates": [100, 100]},
        {"kind": "receiver", "group": "near", "id": 2, "coordinates": [900, 100]},
    ]
    assert patches.pml.pml_wavelengths == 1
    assert patches.pml.pml_reflectivity == 1e-4


def test_box_aperture_selects_by_each_direction(tmp_path):
    from frequensolve.seismic.receivers import CoordsArray

    simulation = _simulation(tmp_path)
    simulation.acquisition.receiver_groups[0].coordinates = CoordsArray(
        coordinates=np.array([[0.19, 0.29], [0.19, 0.31]])
    )
    patches = fs.PatchSet.around_sources(
        shots_per_patch=4,
        max_offset=np.array([200, 200]) * fs.ureg.m,
        padding=0 * fs.ureg.m,
    )
    _, selection = patches._requests(simulation, _inventory())
    assert selection[0]["receivers"]["near"][1] == (1,)
    patches.max_offset = (200, 200, 200)
    with pytest.raises(ValueError, match="simulation dimension"):
        patches._requests(simulation, _inventory())


@pytest.mark.parametrize("explicit", [False, True])
def test_patch_policy_job_roundtrip_and_fingerprint(tmp_path, explicit):
    from frequensolve.orchestrator.sites.base import BaseSite
    from tests.test_imaging_jobs import _assert_valid

    options = dict(
        max_offset=np.array([200, 50]) * fs.ureg.m,
        padding=300 * fs.ureg.m,
        depth=np.array([0, 2000]) * fs.ureg.m,
    )
    patches = (
        fs.PatchSet(
            [fs.Patch(name="all", roots=(1, 2), sources=(1, 2, 3, 4))], **options
        )
        if explicit
        else fs.PatchSet.around_sources(shots_per_patch=2, **options)
    )
    simulation = _simulation(tmp_path)
    job = fs.FrequencyDomainJob("patched", simulation, [3, 7.5], patches=patches)
    saved = job.save()
    restored = fs.FrequencyDomainJob.load(saved)
    assert restored.patches.to_dict() == patches.to_dict()
    payload = _assert_valid(json.loads(saved.read_text()))
    original = job._fingerprint_job_payload(payload, include_frequencies=True)
    restored.patches.padding += 1
    assert original != restored._fingerprint_job_payload(
        restored.to_fs(), include_frequencies=True
    )
    # Direct submission must never run the uncut parent, even with validation off.
    with pytest.raises(ValueError, match="site.run"):
        BaseSite().prepare_job(job, validate=False)


def test_depth_is_explicit_and_cannot_discard_selected_acquisition(tmp_path):
    simulation = _simulation(tmp_path)
    patches = fs.PatchSet.around_sources(
        shots_per_patch=2,
        max_offset=200 * fs.ureg.m,
        padding=0 * fs.ureg.m,
        depth=np.array([0, 1]) * fs.ureg.km,
    )
    request, _ = patches._requests(simulation, _inventory())
    assert request["patches"][0]["upper"][-1] == 1000
    patches.depth = (500, 1000)
    with pytest.raises(ValueError, match="depth excludes"):
        patches._requests(simulation, _inventory())


def test_explicit_patches_require_complete_disjoint_shots_and_real_roots(tmp_path):
    simulation = _simulation(tmp_path)
    patch = fs.Patch(name="west", roots=[2, 1], sources=[2, 1])
    assert patch.roots == (1, 2)
    with pytest.raises(FrozenInstanceError):
        patch.sources = (1,)
    other = fs.Patch(name="east", roots=[3, 4], sources=[3, 4])
    patches = fs.PatchSet(
        [patch, other], max_offset=200 * fs.ureg.m, padding=1 * fs.ureg.m
    )
    request, selection = patches._requests(simulation, _inventory())
    assert request["patches"][0]["roots"] == [1, 2]
    assert selection[1]["sources"] == (3, 4)
    missing = fs.PatchSet([patch], max_offset=200 * fs.ureg.m, padding=1 * fs.ureg.m)
    with pytest.raises(ValueError, match="exactly once"):
        missing._requests(simulation, _inventory())
    invalid = fs.PatchSet(
        [fs.Patch(name="bad", roots=[5], sources=[1, 2, 3, 4])],
        max_offset=200 * fs.ureg.m,
        padding=0 * fs.ureg.m,
    )
    with pytest.raises(ValueError, match="root absent"):
        invalid._requests(simulation, _inventory())


@pytest.mark.parametrize(
    "updates",
    [
        {"max_offset": 1},
        {"padding": 0},
        {"depth": [0, 1]},
        {"max_offset": 1 * fs.ureg.s},
        {"padding": -1 * fs.ureg.m},
        {"padding": np.nan * fs.ureg.m},
        {"shots_per_patch": True},
    ],
)
def test_patch_policy_rejects_ambiguous_units_and_invalid_values(updates):
    kwargs = dict(shots_per_patch=2, max_offset=1 * fs.ureg.km, padding=100 * fs.ureg.m)
    kwargs.update(updates)
    with pytest.raises((ValueError, TypeError)):
        fs.PatchSet.around_sources(**kwargs)


def test_unsupported_acquisition_is_rejected_before_scheduling(tmp_path):
    simulation = _simulation(tmp_path)
    simulation.acquisition.source_encoding = object()
    patches = fs.PatchSet.around_sources(
        shots_per_patch=2, max_offset=1 * fs.ureg.km, padding=0 * fs.ureg.m
    )
    with pytest.raises(NotImplementedError, match="physical point shots"):
        patches.prepare(simulation, [3], site=None)


@pytest.mark.parametrize("proof", [None, [], [5, 5, 5, 5], [1, 1, 1, 1]])
def test_preparation_requires_complete_native_containment(tmp_path, monkeypatch, proof):
    from frequensolve.simulation.jobs._patches import PatchPreparationJob

    def report(job):
        geometry = _inventory()
        geometry["patches"] = []
        for patch in job.request.get("patches", []):
            entry = {"name": patch["name"], "descriptor": {"roots": [1, 2, 3, 4]}}
            if proof is not None:
                entry.update(acquisition_checked=True, point_roots=proof)
            geometry["patches"].append(entry)
        return geometry

    monkeypatch.setattr(PatchPreparationJob, "geometry_report", property(report))
    patches = fs.PatchSet.around_sources(
        shots_per_patch=2, max_offset=200 * fs.ureg.m, padding=0 * fs.ureg.m
    )
    site = SimpleNamespace(run=lambda *args, **kwargs: None)
    if proof == [1, 1, 1, 1]:
        result = patches.prepare(_simulation(tmp_path), [3], site=site)
        assert len(result.geometry["patches"]) == 2
    else:
        with pytest.raises(ValueError, match="lacks native acquisition containment"):
            patches.prepare(_simulation(tmp_path), [3], site=site)


def test_prepared_preview_requires_requested_edge_samples():
    geometry = {
        "dimension": 2,
        **_inventory(),
        "patches": [{"name": "one", "descriptor": {"roots": [1, 2]}}],
    }
    prepared = fs.PreparedPatchSet(
        geometry, [], (), fs.BoundaryCondition(conditions=["pml"])
    )
    estimate = prepared.storage_estimates["patches"][0]
    assert estimate["root_id_payload_bytes"] == 16
    assert estimate["preview_float64_payload_bytes"] is None
    with pytest.raises(ValueError, match="edge_samples=True"):
        prepared.plot()
    geometry["roots"].update(edge_count=[4] * 4, edge_points=[[0.0] * 9 * 15] * 2)
    with pytest.raises(ValueError, match="edge_points"):
        fs.PreparedPatchSet(
            geometry, [], (), fs.BoundaryCondition(conditions=["pml"])
        ).plot()


@pytest.mark.visual
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("support_added", [False, True])
def test_prepared_plot_preserves_native_curved_samples_and_acquisition(
    dimension, support_added
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.linspace(0, 1, 9)
    points = np.array([t, t * (1 - t)] if dimension == 2 else [t, 0.5 * t, t * (1 - t)])
    geometry = {
        "dimension": dimension,
        "root_count": 1,
        "roots": {
            "cell": [10],
            "domain": [1],
            "sampled_lower": [[0]] * dimension,
            "sampled_upper": [[1]] * dimension,
            "edge_count": [1],
            "edge_points": points.tolist(),
        },
        "patches": [
            {
                "name": "curved",
                "descriptor": {"roots": [1]},
                "core_roots": [1],
                "added_roots": [],
            }
        ],
    }
    if support_added:
        # A second root's edge samples pack after the first root's columns.
        roots = geometry["roots"]
        geometry["root_count"] = 2
        roots["cell"].append(20)
        roots["domain"].append(1)
        roots["sampled_lower"] = [[0, 0.1]] * dimension
        roots["sampled_upper"] = [[1, 1.1]] * dimension
        roots["edge_count"].append(1)
        roots["edge_points"] = np.concatenate([points, points + 0.1], axis=1).tolist()
        geometry["patches"][0]["descriptor"]["roots"] = [1, 2]
        geometry["patches"][0]["support_roots"] = [1, 2]
    acquisition = [
        {
            "name": "curved",
            "source_coordinates": [[0.2] * dimension],
            "receiver_coordinates": {"p": {1: [0.4] * dimension}},
        }
    ]
    prepared = fs.PreparedPatchSet(
        geometry, acquisition, (), fs.BoundaryCondition(conditions=["pml"])
    )
    geometry["roots"]["edge_points"][0][0] = 1000
    assert prepared.geometry["roots"]["edge_points"][0][0] == 0
    assert prepared.storage_estimates["patches"][0][
        "preview_float64_payload_bytes"
    ] == 8 * dimension * 9 * (1 + support_added)
    ax = prepared.plot()
    try:
        ax.figure.canvas.draw()
        assert len(ax.collections) == 5 + support_added
        if support_added:
            addition = ax.collections[3]
            samples = (
                addition.get_segments()[0]
                if dimension == 2
                else addition._segments3d[0]
            )
            np.testing.assert_allclose(samples, (points + 0.1).T)
        curve = ax.collections[2]
        samples = curve.get_segments()[0] if dimension == 2 else curve._segments3d[0]
        np.testing.assert_allclose(samples, points.T)
    finally:
        plt.close(ax.figure)
