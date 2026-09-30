"""Prior semantics and saved statistical bases across mesh refinement."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging._native_regularization import bind_workflow_regularization
from frequensolve.imaging.curvature import CurvatureResult, NativeCurvature
from frequensolve.imaging.statistics import mesh_descriptor
from frequensolve.model.parameterization import MeshControl, MeshPropertySpace
from tests.statistical_mesh_fixture import write_mesh
from tests.test_imaging_statistics import StatisticalSite, oracle_runner
from tests.test_imaging_workflows import _problem

pytestmark = pytest.mark.unit


def test_frozen_prior_energy_uses_stage_values_and_composes(tmp_path):
    problem = _problem(tmp_path, StatisticalSite())
    problem.linearize()
    reference = problem.state
    masks = {
        name: np.ones(problem.space.block(name).size, dtype=bool)
        for name in problem.space.blocks
    }
    masks["model.vp"][[0, 3]] = False
    active = problem.space.with_support(masks)
    baseline = im.ControlState(reference.space, reference.values + np.arange(1, 9))
    stage = SimpleNamespace(state=baseline)
    prior = im.GaussianPrior(reference=reference, std={"vp": 2, "rho": 3})
    bound = prior.bind(active, problem=problem, linearization=stage)
    point = baseline.vector(active)
    scale = np.r_[np.full(5, 2), np.full(3, 3)]
    expected = 0.5 * np.sum(((baseline.values - reference.values) / scale) ** 2)
    assert bound.value(point) == pytest.approx(expected)
    np.testing.assert_allclose(
        bound.gradient(point).values,
        (baseline.values - reference.values)[active.active_indices]
        / scale[active.active_indices] ** 2,
    )
    # Frozen energy is independent of the active vector, and changes the
    # objective fingerprint when the fixed state changes.
    assert bound.value(im.ControlVector(bound.reference, active)) == pytest.approx(
        (1**2 + 4**2) / 8
    )
    changed = baseline.values.copy()
    changed[0] += 1
    rebound = prior.bind(
        active,
        problem=problem,
        linearization=SimpleNamespace(state=im.ControlState(reference.space, changed)),
    )
    assert rebound.identity != bound.identity
    composed, native = bind_workflow_regularization(
        2 * prior + prior, active, problem, stage
    )
    assert native is None
    assert composed.value(point) == pytest.approx(3 * expected)
    np.testing.assert_allclose(
        composed.gradient(point).values, 3 * bound.gradient(point).values
    )
    np.testing.assert_allclose(
        composed.curvature_diagonal(point), 3 * bound.curvature_diagonal(point)
    )


def _mesh_space(base, descriptor):
    """Install the resolved layout which Sauce normally supplies via its registry."""
    old = base.restrict("vp")
    block = old.block("vp")
    material = [sub.name for sub in old.simulation.model.subdomains].index(
        block.subdomain
    ) + 1
    identity = dict(
        schema="sauce-parameterized-property-identity-1",
        control=f"{descriptor['identity']}/material/{material}",
        transform="identity",
    )
    basis = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    block = replace(
        block,
        kind="mesh",
        size=descriptor["size"],
        dims=(),
        coords=None,
        baseline=np.zeros(descriptor["size"]),
        control=MeshControl("material"),
        basis_identity=basis,
    )
    space = old._clone([block], {})
    space._property_spaces = {
        "material": MeshPropertySpace(descriptor["path"], frequency=8, epw=2)
    }
    return space


def _mesh_runner(request_path):
    request = json.loads(request_path.read_text())
    if request["method"] != "mesh_prior":
        return oracle_runner(request_path)
    with h5py.File(request["input"]) as h5:
        mean, std = h5["reference"][()], h5["prior_std"][()]
        metadata = json.loads(h5["metadata"][()])
    with h5py.File(request["target_mesh"]) as h5:
        size = int(h5["property_space/header"][1])
    weights = (
        np.full(4, 0.25)
        if size == 4
        else np.array([0.125, 0.25, 0.25, 0.125, 0.125, 0.125])
    )
    # The fixture's refined points are midpoint lifts of the original basis.
    interpolation = (
        np.eye(4)
        if size == 4
        else np.array(
            [
                [1, 0, 0, 0],
                [0.5, 0.5, 0, 0],
                [0, 0, 0.5, 0.5],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ]
        )
    )
    assert len(mean) == 4  # Every stage lifts the original declaration directly.
    metadata["schema"] = "fs-curvature-output-1"
    with h5py.File(request["output"], "w") as h5:
        h5["metadata"] = np.bytes_(json.dumps(metadata))
        h5["reference"] = interpolation @ mean
        h5["prior_std"] = (interpolation @ std) / np.sqrt(weights)
        h5["measure_weights"] = weights


def test_volume_prior_lifts_original_fields_and_keeps_constant_energy(tmp_path):
    coarse, _ = write_mesh(tmp_path / "coarse.h5")
    refined, _ = write_mesh(tmp_path / "refined.h5", refined=True)
    problem = _problem(tmp_path, StatisticalSite())
    problem.linearize()
    old, new = _mesh_space(problem.space, coarse), _mesh_space(problem.space, refined)
    native = NativeCurvature(workdir=tmp_path / "native", runner=_mesh_runner)
    context = SimpleNamespace(
        backend=SimpleNamespace(curvature=lambda: native), state=None
    )
    reference = im.ControlState(old, [2, 5, 4, 1])
    prior = im.GaussianPrior(reference, std=2)
    original = prior.bind(old, problem=context)
    lifted = prior.bind(new, problem=context)
    np.testing.assert_allclose(lifted.reference, [2, 3.5, 2.5, 1, 5, 4])
    for bound in (original, lifted):
        assert bound.value(bound.reference + 2) == pytest.approx(0.5)
    coefficient_prior = im.GaussianPrior(reference, std=2, mesh_measure="coefficients")
    assert coefficient_prior.bind(old, problem=context).value(
        original.reference + 2
    ) == pytest.approx(2)
    assert coefficient_prior.bind(new, problem=context).value(
        lifted.reference + 2
    ) == pytest.approx(3)


def test_saved_uncertainty_restores_original_mesh_against_refined_problem(tmp_path):
    coarse, _ = write_mesh(tmp_path / "coarse.h5")
    refined, _ = write_mesh(tmp_path / "refined.h5", refined=True)
    problem = _problem(tmp_path, StatisticalSite())
    problem.linearize()
    old, new = _mesh_space(problem.space, coarse), _mesh_space(problem.space, refined)
    native = NativeCurvature(workdir=tmp_path / "native", runner=oracle_runner)
    path = tmp_path / "factors.h5"
    metadata = dict(
        schema="fs-curvature-output-1",
        state="old-stage",
        coordinates="old-basis",
        rank=0,
    )
    with h5py.File(path, "w") as h5:
        h5["metadata"] = np.bytes_(json.dumps(metadata))
        h5["base_inverse_diagonal"] = np.arange(1, 5)
        h5["prior_std"] = np.ones(4)
        h5["variance"] = np.arange(1, 5)
        h5["standard_deviation"] = np.sqrt(np.arange(1, 5))
    result = im.UncertaintyResult(
        CurvatureResult(path, metadata),
        im.ControlVector([2, 5, 4, 1], old),
        native=native,
        units={"model.vp": "m/s"},
    )
    saved = result.save(tmp_path / "saved")
    restored = im.UncertaintyResult.load(saved, new, native=native)
    assert restored.space.size == 4
    assert new.size == 6
    np.testing.assert_allclose(restored.std("vp"), result.std("vp"))
    np.testing.assert_allclose(
        (restored.covariance @ restored.space.ones()).values, [1, 2, 3, 4]
    )
    # Both the result descriptor and its reconstructed control space point to
    # the copied old mesh, so later native operations cannot use the final one.
    descriptor = mesh_descriptor(restored.space, restored.space.block("vp"))
    assert descriptor["identity"] == coarse["identity"]
    assert descriptor["path"] == str(saved / "model.vp.mesh.h5")
    with h5py.File(saved / "model.vp.mesh.h5", "a") as h5:
        del h5["property_space/identity"]
        h5["property_space/identity"] = np.bytes_("changed")
    with pytest.raises(ValueError, match="mesh identity changed"):
        im.UncertaintyResult.load(saved, new, native=native)
