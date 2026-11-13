"""Native point-confidence snapshots retain a fixed objective across trials."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from frequensolve.imaging._reliability import freeze, point_weights
from frequensolve.imaging.misfit import Comparison, Loss, Misfit


def test_local_confidence_is_gain_invariant_and_targets_notches():
    from frequensolve.imaging.receiver_pool import ReceiverPool

    x = np.arange(101)[:, None]
    pool = ReceiverPool.from_coordinates(x, 6, taper="gaussian", sigma=2)
    amplitude = np.exp(-0.1 * x[:, 0])
    amplitude[50] = 0
    keys = np.column_stack((np.ones(101, int), np.arange(1, 102), np.ones(101, int)))
    weights = []
    for gain in (1e-20, 1, 1e20):
        values = np.column_stack((gain * amplitude, np.zeros(101)))
        weights.append(
            point_weights(keys, values, [[1]], pool=pool, relative_floor=0.1)
        )
    np.testing.assert_allclose(weights[0], weights[1], rtol=1e-14)
    np.testing.assert_allclose(weights[2], weights[1], rtol=1e-14)
    assert weights[1][0, 0, 50] == 0
    assert np.count_nonzero(weights[1] < 0.9) == 1


def test_weights_use_coordinate_keys_and_observed_floor():
    keys = [[2, 2, 1], [1, 1, 1], [2, 1, 1], [1, 2, 1]]
    values = [[3, 4], [0, 0], [1, 0], [2, 0]]
    weight = point_weights(keys, values, [[2, 1]])
    np.testing.assert_allclose(weight[:, 0], [[0, 0.5], [0.5, 25 / 26]])


def test_array_confidence_pools_energy_before_ratio():
    from frequensolve.imaging.receiver_pool import ReceiverPool

    pool = ReceiverPool.from_coordinates(np.arange(3)[:, None], 3, taper="uniform")
    keys = [[1, 3, 1], [1, 1, 1], [1, 2, 1]]
    values = np.array([[0, 0], [3, 0], [0, 0]])
    for gain in (1e-20, 1, 1e20):
        weight = point_weights(keys, gain * values, [[gain]], attribute_pool=pool)
        np.testing.assert_allclose(weight, [[[0.75, 0.75, 0.75]]])
    local = point_weights(
        keys, values, [[1]], attribute_pool=pool, pool=pool, relative_floor=0.1
    )
    np.testing.assert_allclose(local, 1 / 1.01)
    with pytest.raises(ValueError, match="complete receiver layout"):
        point_weights(keys[:1], values[:1], [[1]], attribute_pool=pool)


@pytest.mark.parametrize(
    "keys,values,floor",
    [
        ([[1, 1, 1], [1, 1, 1]], [[1, 0], [1, 0]], [[1]]),
        ([[0, 1, 1]], [[1, 0]], [[1]]),
        ([[1, 1, 1]], [[float("nan"), 0]], [[1]]),
        ([[1, 1, 1]], [[1, 0]], [[-1]]),
    ],
)
def test_invalid_snapshot_is_rejected(keys, values, floor):
    with pytest.raises(ValueError):
        point_weights(keys, values, floor)


class Problem:
    def __init__(self, root, misfit=None):
        self.workdir = root
        self.misfit = misfit or Misfit(
            loss=Loss.huber(1),
            comparison=Comparison.phase_derivative(
                reliability="predicted", confidence_floor=0.05
            ),
            normalization=1,
        )
        self.observed_groups = [SimpleNamespace(name="p")]
        self.patches = self.kernel_derivative = None

    def with_misfit(self, misfit):
        return Problem(self.workdir, misfit)

    def linearize(self, v=None, gradient=False):
        model = 1 if v is None else v
        frequencies = [2 - 0.5j, 2 - 0.15j]
        for task in [1, 2]:
            cache = self.workdir / f"state_{task}_cache.h5"
            with h5py.File(cache, "w") as h:
                h["terms/0/coordinate_keys"] = [[1, 1, 1], [1, 2, 1]]
                h["terms/0/simulated"] = [[model * task, 0], [0, 0]]
                h["terms/0/phase_floor"] = [[1]]
            (self.workdir / f"state_{task}.json").write_text(
                json.dumps(
                    {
                        "terms": [
                            {
                                "id": "p",
                                "cache": {
                                    "file": cache.name,
                                    "group": "/terms/0",
                                },
                            }
                        ],
                    }
                )
            )
        return SimpleNamespace(
            frequencies=frequencies,
            point=SimpleNamespace(values=np.array([model])),
            job=SimpleNamespace(
                job_file=self.workdir / "job.json",
                state_file=lambda task: self.workdir / f"state_{task}.json",
            ),
        )


def test_snapshot_persists_across_trials_and_refresh_changes_identity(tmp_path):
    problem = Problem(tmp_path)
    frozen = freeze(problem, 1)
    term = frozen.misfit.objective_terms(["p"])[0]
    assert term.loss.kind == "huber"
    assert term.comparison.reliability is None
    assert term.comparison.confidence_floor == 0.05
    assert problem.misfit.comparison.reliability == "predicted"
    rows = term.preprocess[-1].to_fs()["params"]["frequency_weights"]
    assert [(r["frequency_hz"], r["laplace_hz"]) for r in rows] == [
        (2, -0.5),
        (2, -0.15),
    ]
    # Candidate evaluations overwrite their own receiver caches, never snapshots.
    frozen.linearize(10)
    with h5py.File(frozen.reliability_snapshot) as h:
        np.testing.assert_allclose(h["terms/0/frequency_0"][:], [[[0.5, 0]]])
        np.testing.assert_allclose(h["terms/0/frequency_1"][:], [[[0.8, 0]]])
    refreshed = freeze(problem, 2)
    assert refreshed.reliability_snapshot != frozen.reliability_snapshot
    assert freeze(frozen, 3) is frozen
    assert freeze(problem, 1).reliability_snapshot == frozen.reliability_snapshot


def test_array_snapshot_preserves_native_comparison_and_freezes_trials(
    tmp_path, monkeypatch
):
    import frequensolve as fs
    from frequensolve import imaging as im
    from frequensolve.imaging import _synthetic_comparison

    array = im.SyntheticArray.uniform(radius=fs.Q_(2, "m"))
    misfit = Misfit(
        comparison=Comparison.phase_derivative(
            array=array, reliability="predicted", relative_amplitude_floor=0.1
        ),
        normalization=1,
    )
    monkeypatch.setattr(
        _synthetic_comparison,
        "receiver_coordinates",
        lambda problem: {"p": fs.Q_([[0], [1]], "m")},
    )
    problem = Problem(tmp_path, misfit)
    frozen = freeze(problem, 1)
    term = frozen.misfit.objective_terms(["p"])[0]
    assert term.comparison.array == array
    assert term.comparison.reliability is None
    frozen.linearize(10)
    with h5py.File(frozen.reliability_snapshot) as h:
        np.testing.assert_allclose(h["terms/0/frequency_0"][:], [[[1 / 3, 1 / 3]]])
        assert json.loads(h.attrs["attribute_arrays"])["0"] == array.to_dict()
    assert freeze(frozen, 3) is frozen
    assert freeze(problem, 2).reliability_snapshot != frozen.reliability_snapshot
