"""Curvature transfer policy, immutable seeds and verified native dispatch."""

import json

import h5py
import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging._curvature_transfer import _CurvatureSeed, _history_inverse
from frequensolve.imaging.curvature import CurvatureResult
from tests.statistical_mesh_fixture import write_mesh
from tests.test_imaging_curvature import (
    checkpoint,
    solver_output,
    spec_digests,
    write_result,
)

pytestmark = pytest.mark.unit

FACTOR_NAMES = ("base_inverse_diagonal", "prior_std", "modes", "eigenvalues")


def _runner(calls, *, damping=1, omit_seed_rank=False, asymmetry=0.0):
    def run(path):
        request = json.loads(path.read_text())
        arrays, metadata = solver_output(request)
        method = request["method"]
        base = arrays.get("base_inverse_diagonal")
        if base is not None:
            metadata["controls"] = len(base)
        if method.startswith("bfgs"):
            if "seed_modes" in arrays and not omit_seed_rank:
                metadata["seed_rank"] = len(arrays["seed_modes"])
        if method == "curvature_basis":
            metadata["basis_tolerance"] = request["basis_tolerance"]
            metadata["rank"] = min(
                request.get("rank", len(arrays["directions"])),
                len(arrays["directions"]),
            )
        if method == "curvature_warm_start":
            metadata["transfer_damping"] = damping
        if method == "curvature_refresh" and asymmetry is not None:
            metadata["symmetry_tolerance"] = request.get("symmetry_tolerance", 1e-2)
            metadata["hessian_asymmetry"] = asymmetry
        calls.append((request, arrays, metadata))
        outputs = {}
        if method in {"curvature_refresh", "curvature_warm_start", "bfgs_rsvd"}:
            modes = arrays.get(
                "modes", arrays.get("seed_modes", np.empty((0, len(base))))
            )
            eig = arrays.get("eigenvalues", arrays.get("seed_eigenvalues", np.empty(0)))
            metadata["rank"] = len(eig)
            outputs = dict(
                base_inverse_diagonal=base,
                prior_std=arrays.get("prior_std", np.ones(len(base))),
                modes=modes,
                eigenvalues=eig,
                variance=base,
                standard_deviation=np.sqrt(base),
            )
        elif method == "curvature_basis":
            outputs = dict(directions=arrays["directions"][: metadata["rank"]])
        elif method == "mesh_directions":
            outputs = dict(directions=arrays["directions"])
        write_result(request, metadata, outputs)

    return run


def test_public_transfer_policy_is_validated_and_immutable():
    assert im.CurvatureTransfer.reset().method == "reset"
    assert im.CurvatureTransfer.warm_start(rank=7).rank == 7
    assert (
        im.CurvatureTransfer.refresh(rank=3, basis_tolerance=1e-3).basis_tolerance
        == 1e-3
    )
    for kwargs in (
        {"method": "copy"},
        {"rank": 0},
        {"rank": 2.5},
        {"rank": True},
        {"basis_tolerance": 0},
        {"basis_tolerance": 1e-10},
        {"basis_tolerance": 1},
        {"basis_tolerance": np.nan},
        {"method": "refresh", "symmetry_tolerance": 0},
        {"method": "refresh", "symmetry_tolerance": 1e-10},
        {"method": "refresh", "symmetry_tolerance": 1},
        {"method": "refresh", "symmetry_tolerance": np.nan},
        {"method": "warm_start", "symmetry_tolerance": 1e-3},
    ):
        with pytest.raises(ValueError):
            im.CurvatureTransfer(**kwargs)
    with pytest.raises(AttributeError):
        im.CurvatureTransfer.refresh().rank = 3
    assert im.CurvatureTransfer.refresh().symmetry_tolerance is None
    policy = im.CurvatureTransfer.refresh(symmetry_tolerance=np.float64(5e-3))
    assert (
        type(policy.symmetry_tolerance) is float and policy.symmetry_tolerance == 5e-3
    )


def test_refresh_requests_and_checks_the_hessian_symmetry_tolerance(tmp_path):
    calls = []
    arrays = ([2, 3], [[1, 0], [0, 1]], [[4, 0], [0, 5]])
    identities = dict(state="stage-2", coordinates="fine")
    native = im.NativeCurvature(
        workdir=tmp_path / "a", runner=_runner(calls, asymmetry=3e-3)
    )
    # Without a tolerance the request leaves Sauce's default, which it reports.
    default = native.refresh_curvature(*arrays, **identities)
    assert "symmetry_tolerance" not in calls[-1][0]
    assert default.metadata["symmetry_tolerance"] == 1e-2
    assert default.metadata["hessian_asymmetry"] == 3e-3
    explicit = native.refresh_curvature(*arrays, symmetry_tolerance=5e-3, **identities)
    assert calls[-1][0]["symmetry_tolerance"] == 5e-3
    assert explicit.metadata["symmetry_tolerance"] == 5e-3
    configured = im.NativeCurvature(
        workdir=tmp_path / "b",
        runner=_runner(calls, asymmetry=3e-3),
        symmetry_tolerance=4e-3,
    )
    configured.refresh_curvature(*arrays, **identities)
    assert calls[-1][0]["symmetry_tolerance"] == 4e-3
    for value in (0, 1e-10, 1, np.nan):
        with pytest.raises(ValueError, match="symmetry_tolerance"):
            native.refresh_curvature(*arrays, symmetry_tolerance=value, **identities)
    count = len(calls)
    with pytest.raises(ValueError, match="symmetry_tolerance"):
        im.NativeCurvature(workdir=tmp_path, runner=native.runner, symmetry_tolerance=1)
    assert len(calls) == count
    for runner, message in (
        (_runner([], asymmetry=None), "lacks symmetry_tolerance"),
        (_runner([], asymmetry=2e-2), "invalid hessian_asymmetry"),
    ):
        with pytest.raises(ValueError, match=message):
            im.NativeCurvature(workdir=tmp_path / "c", runner=runner).refresh_curvature(
                *arrays, **identities
            )


def test_seed_history_roundtrip_preserves_metric_provenance_and_digest(tmp_path):
    modes = np.array([[1.0, 2.0], [0.5, -1.0]])
    history = im.BFGSHistory(
        [2, 3],
        state="new-objective",
        coordinates="new-basis",
        seed_modes=modes,
        seed_eigenvalues=[0.5, -0.1],
        provenance={"method": "refresh", "previous_stage": 1},
    )
    history(checkpoint([dict(step=[1, 0], difference=[2, 0])], 1))
    modes[:] = 0
    restored = im.BFGSHistory.load(history.save(tmp_path / "history.h5"))
    assert restored.seed_rank == 2
    assert restored.digest == history.digest
    assert restored.provenance == history.provenance
    np.testing.assert_equal(restored.seed_modes, [[1, 2], [0.5, -1]])
    assert not restored.seed_modes.flags.writeable
    provenance = restored.provenance
    provenance["method"] = "warm_start"
    assert restored.provenance["method"] == "refresh"
    other = im.BFGSHistory(
        [2, 3],
        state="new-objective",
        coordinates="new-basis",
        seed_modes=restored.seed_modes,
        seed_eigenvalues=[0.5, -0.1],
        provenance={"method": "warm_start"},
    )
    assert other.digest != history.digest


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed_modes": [[1, 2]]},
        {"seed_eigenvalues": [1]},
        {"seed_modes": [[1, 2, 3]], "seed_eigenvalues": [1]},
        {"seed_modes": [[1, 2]], "seed_eigenvalues": [1, 2]},
        {"seed_modes": [[1j, 2]], "seed_eigenvalues": [1]},
        {"seed_modes": [[1, np.nan]], "seed_eigenvalues": [1]},
        {"provenance": {"bad": float("nan")}},
    ],
)
def test_seed_history_rejects_invalid_coordinates(kwargs):
    with pytest.raises(ValueError):
        im.BFGSHistory([1, 1], state="s", coordinates="c", **kwargs)


def test_seed_is_staged_once_and_default_rank_is_left_to_generalized_native_method(
    tmp_path,
):
    calls = []
    native = im.NativeCurvature(workdir=tmp_path, runner=_runner(calls))
    history = im.BFGSHistory(
        [2, 3],
        state="s",
        coordinates="c",
        seed_modes=[[1, 0]],
        seed_eigenvalues=[0.5],
        provenance={"method": "refresh"},
    )
    with native.retain(history):
        native.bfgs_uncertainty(history)
        native.inverse_action(history, [1, 2])
        assert len(list((tmp_path / "histories").glob("*.h5"))) == 1
    assert not list((tmp_path / "histories").glob("*.h5"))
    assert "rank" not in calls[0][0]
    for request, arrays, metadata in calls:
        assert request["seed_rank"] == metadata["seed_rank"] == 1
        np.testing.assert_equal(arrays["seed_modes"], [[1, 0]])
        with h5py.File(request["input"]) as h5:
            assert isinstance(h5.get("seed_modes", getlink=True), h5py.ExternalLink)
    missing = im.NativeCurvature(
        workdir=tmp_path / "old-solver", runner=_runner([], omit_seed_rank=True)
    )
    with pytest.raises(ValueError, match="lacks seed_rank"):
        missing.bfgs_uncertainty(history)
    empty = im.BFGSHistory([1], state="s", coordinates="c")
    native.bfgs_uncertainty(empty)
    assert "seed_rank" not in calls[-1][0]
    assert "seed_modes" not in calls[-1][1]


def test_native_transfer_methods_preserve_batches_identities_and_factor_digests(
    tmp_path,
):
    calls = []
    native = im.NativeCurvature(workdir=tmp_path / "requests", runner=_runner(calls))
    base, directions = [2, 3], [[1, 0], [0, 1]]
    basis = native.transfer_basis(
        base, directions, state="stage-2", coordinates="fine", rank=1
    )
    np.testing.assert_equal(basis.read("directions"), [[1, 0]])
    refreshed = native.refresh_curvature(
        base, directions, [[4, 0], [0, 5]], state="stage-2", coordinates="fine"
    )
    warm = native.warm_start_curvature(
        base, [[1, 0]], [0.5], state="stage-2", coordinates="fine"
    )
    for factors in (refreshed, warm):
        assert factors.verify() == factors.output_digests
        np.testing.assert_equal(factors.read("prior_std"), [1, 1])
    applied = native.covariance(warm, vectors=[1, 2])
    read = applied.metadata["factors_digests"]
    assert read == {key: warm.output_digests[key] for key in read}
    assert set(read) == {f"/{name}" for name in FACTOR_NAMES}
    assert warm.metadata["transfer_damping"] == 1
    source, _ = write_mesh(tmp_path / "old.h5")
    target, _ = write_mesh(tmp_path / "new.h5", refined=True)
    native.mesh_directions(source, target, [[1, 2, 3, 4]])
    request, arrays, _ = calls[-1]
    assert request["source_identity"] == request["state"] == source["identity"]
    assert request["target_identity"] == request["coordinates"] == target["identity"]
    assert request["dimension"] == 2  # FS_seismic routes mesh requests by it
    np.testing.assert_equal(arrays["directions"], [[1, 2, 3, 4]])
    np.testing.assert_equal(calls[1][1]["images"], [[4, 0], [0, 5]])


def test_transfer_validation_precedes_native_run(tmp_path):
    calls = []
    native = im.NativeCurvature(workdir=tmp_path, runner=_runner(calls))
    identities = dict(state="s", coordinates="c")
    with pytest.raises(ValueError):
        native.transfer_basis([-1], [[1]], **identities)
    with pytest.raises(ValueError):
        native.transfer_basis([1], [[1, 2]], **identities)
    with pytest.raises(ValueError):
        native.transfer_basis([1], [[1]], basis_tolerance=0, **identities)
    with pytest.raises(ValueError):
        native.transfer_basis([1], [[1]], basis_tolerance=1e-10, **identities)
    with pytest.raises(ValueError):
        native.refresh_curvature([1], [[1]], [[1, 2]], **identities)
    with pytest.raises(ValueError):
        native.warm_start_curvature([1], [[1]], [1, 2], **identities)
    assert not calls
    bad = im.NativeCurvature(workdir=tmp_path / "bad", runner=_runner([], damping=-1))
    with pytest.raises(ValueError, match="transfer_damping"):
        bad.warm_start_curvature([1], [[1]], [1], **identities)


def test_loaded_seed_unwhitens_factors_and_applies_without_io(tmp_path, monkeypatch):
    path = tmp_path / "seed.h5"
    stored = dict(
        base_inverse_diagonal=[2.0, 3.0],
        prior_std=[2, 0.5],
        modes=[[1.0, 2.0]],
        eigenvalues=[0.5],
    )
    metadata = dict(
        rank=1, state="old", coordinates="coarse", output_digests=spec_digests(stored)
    )
    with h5py.File(path, "w") as h5:
        for name, value in stored.items():
            h5[name] = value
    factors = CurvatureResult(path, metadata)
    seed = _CurvatureSeed(factors, provenance={"method": "warm_start"})
    np.testing.assert_equal(seed.physical_base, [8, 0.75])
    np.testing.assert_equal(seed.modes, [[2, 1]])
    dense = np.diag([8, 0.75]) + 0.5 * np.outer([2, 1], [2, 1])

    def fail(*args, **kwargs):
        raise AssertionError("A seed action must not read the factor file")

    monkeypatch.setattr(CurvatureResult, "read", fail)
    monkeypatch.setattr(CurvatureResult, "verify", fail)
    np.testing.assert_allclose(seed.apply([1, 2]), dense @ [1, 2])
    np.testing.assert_allclose(seed.apply(np.eye(2)), dense)
    assert seed.provenance["factors_identity"] == seed.digest == factors.identity
    with pytest.raises(ValueError):
        seed.apply([1, 2, 3])


def test_history_applies_initial_seed_in_physical_coordinates_without_execution():
    history = im.BFGSHistory(
        [2, 3],
        state="stage",
        coordinates="whitened",
        seed_modes=[[1, 2]],
        seed_eigenvalues=[0.5],
        provenance={"method": "refresh"},
    )
    # The checkpoint's secants do not modify the restored initial metric:
    # L-BFGS replays those updates separately from its frozen preconditioner.
    history(checkpoint([dict(step=[1, 0], difference=[4, 0])], 1))
    scale = np.array([2, 0.5])
    apply = _history_inverse(history, scale)
    # S (B + V diag(e) V^T) S with physical diagonal [8, 0.75] and mode [2, 1].
    dense = np.diag([8, 0.75]) + 0.5 * np.outer([2, 1], [2, 1])
    np.testing.assert_allclose(apply(None, np.array([1.0, 2.0])), [12, 3.5])
    np.testing.assert_allclose(apply(None, np.eye(2)), dense)
    baseline = _history_inverse(
        im.BFGSHistory([2, 3], state="s", coordinates="c"), scale
    )
    np.testing.assert_equal(baseline(None, np.array([1.0, 2.0])), [8, 1.5])
    with pytest.raises(ValueError):
        _history_inverse(history, np.array([1.0]))
    with pytest.raises(ValueError):
        _history_inverse(history, np.array([1.0, 0.0]))


def test_seed_factor_pin_rejects_an_altered_artifact(tmp_path):
    calls = []
    native = im.NativeCurvature(workdir=tmp_path, runner=_runner(calls))
    factors = native.warm_start_curvature(
        [1, 2], [[1, 0]], [0.5], state="s", coordinates="c"
    )
    with h5py.File(factors.path, "a") as h5:
        h5["base_inverse_diagonal"][0] = 99
    with pytest.raises(ValueError, match="factors changed"):
        _CurvatureSeed(factors)
