"""FWI restart state: exact resume, bounded storage, finalize-only resume."""

import json

import h5py
import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging import _restart_store
from frequensolve.imaging._backend import _jsonable, fingerprint
from frequensolve.imaging._restart_store import RestartStore
from frequensolve.imaging.workflows import _state_epoch
from frequensolve.inversion import OptimizationCheckpoint
from frequensolve.inversion.optimization import LBFGSRestart
from tests.test_imaging_curvature_continuation import TransferSite
from tests.test_imaging_workflows import TIGHT, _problem

pytestmark = pytest.mark.unit


class Killed(Exception):
    """A walltime kill inside the solver service."""


class KillingSite(TransferSite):
    """Kill the ``kill``-th ``bfgs_rsvd`` request (1-based)."""

    def __init__(self, *args, kill=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kill = kill
        self.factorizations = 0

    def run_curvature(self, path):
        if json.loads(path.read_text())["method"] == "bfgs_rsvd":
            self.factorizations += 1
            if self.factorizations == self.kill:
                raise Killed(f"killed during factorization {self.kill}")
        return super().run_curvature(path)


def _fwi(path, site, *, stages=2, policy=None, uncertainty=None, **options):
    frequencies = (4, 5, 6)[:stages]
    problem = _problem(
        path, site, misfit=im.Misfit.l2(noise_std=1), frequencies=frequencies
    )
    problem.linearize()
    prior = im.GaussianPrior(problem.state, std={"vp": 2, "rho": 3})
    options.setdefault("optimizer", im.LBFGS(**TIGHT))
    iterations = options.pop("iterations", 3)
    return im.FWI(
        problem,
        [im.Stage([f], iterations) for f in frequencies],
        regularization=prior,
        curvature=policy,
        uncertainty=uncertainty,
        **options,
    )


def _iterates(history):
    """Last accepted model digest per (stage, stage iteration)."""
    return {
        (r.metrics["stage_index"], r.metrics["stage_iteration"]): r.model_digest
        for r in history.iterations
    }


@pytest.mark.parametrize(
    "policy, uncertainty, scaling",
    [
        (im.CurvatureTransfer.refresh(rank=3), None, None),
        (im.CurvatureTransfer.refresh(rank=3), None, "curvature"),
        (im.CurvatureTransfer.warm_start(rank=2), None, None),
        (None, im.BFGSUncertainty(stages="all"), None),
    ],
)
def test_resumed_iterates_are_bitwise_identical(tmp_path, policy, uncertainty, scaling):
    expected = _fwi(
        tmp_path / "expected",
        TransferSite(),
        policy=policy,
        uncertainty=uncertainty,
        scaling=scaling,
        history=tmp_path / "expected" / "history.json",
    ).run()

    def stop(event):
        if event.stage_index == 1 and event.stage_iteration == 2:
            raise RuntimeError("interrupted")

    options = dict(
        policy=policy,
        uncertainty=uncertainty,
        scaling=scaling,
        checkpoint=tmp_path / "resumed" / "ckpt" / "ckpt.h5",
        history=tmp_path / "resumed" / "history.json",
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        _fwi(tmp_path / "resumed", TransferSite(), callback=stop, **options).run()
    saved = OptimizationCheckpoint.load(options["checkpoint"])
    assert saved.metadata["stage_index"] == 1
    assert saved.metadata["stage_iteration"] == 2
    record = json.loads(saved.metadata["restart"])
    assert record["optimizer"]["pairs"] and "optimizer_restart" not in saved.metadata
    resumed = _fwi(tmp_path / "resumed", TransferSite(), **options).run()
    assert resumed.stages[1].resumed and not resumed.stages[1].skipped
    np.testing.assert_array_equal(resumed.state.values, expected.state.values)
    assert _iterates(resumed.history) == _iterates(expected.history)
    if uncertainty is not None:
        identity = np.eye(resumed.uncertainty.space.size)
        np.testing.assert_array_equal(
            resumed.uncertainty.covariance.matmat(identity),
            expected.uncertainty.covariance.matmat(identity),
        )


@pytest.mark.parametrize("uncertainty", [None, im.BFGSUncertainty(stages="all")])
def test_checkpoint_storage_is_bounded_and_each_pair_is_written_once(
    tmp_path, monkeypatch, uncertainty
):
    written = []
    put = _restart_store.StageFiles.put

    def counted(self, name, arrays, attrs=None):
        if name not in self._known:
            written.append((self.name, name))
        return put(self, name, arrays, attrs)

    monkeypatch.setattr(_restart_store.StageFiles, "put", counted)
    memory, iterations = 2, 7
    checkpoint = tmp_path / "ckpt" / "ckpt.h5"
    snapshots = []

    def inspect(event):
        # Only the checkpoint, its state and the restart store exist; never
        # per-iteration archives.
        names = {p.name for p in checkpoint.parent.iterdir()}
        assert names == {"ckpt.h5", "ckpt.state.h5", "ckpt.restart"}
        stage_dirs = list((checkpoint.parent / "ckpt.restart").iterdir())
        assert len(stage_dirs) == 1
        snapshots.append(sorted(p.name for p in stage_dirs[0].iterdir()))

    workflow = _fwi(
        tmp_path,
        TransferSite(),
        policy=im.CurvatureTransfer.refresh(rank=2),
        uncertainty=uncertainty,
        optimizer=im.LBFGS(memory=memory, **TIGHT),
        iterations=iterations,
        checkpoint=checkpoint,
        callback=inspect,
    )
    result = workflow.run()
    assert len(snapshots) > 2 * memory
    for names in snapshots:
        pairs = [n for n in names if n.startswith("pair_")]
        models = [n for n in names if n.startswith("model_")]
        # Limited memory caps transfer runs; full-history UQ keeps the stage.
        assert len(pairs) <= (iterations if uncertainty else memory)
        assert len(models) <= 1 and set(names) == {"stage.h5", *pairs, *models}
    largest = max(len(names) for names in snapshots)
    if uncertainty is None:
        # The cap is reached: stage.h5 plus exactly ``memory`` live pairs.
        assert largest == memory + 1
    else:
        assert largest <= iterations + 2
    # Every pair file was written exactly once over the whole run.
    assert len(written) == len(set(written))
    assert sum(name == "stage.h5" for _, name in written) == 2
    # The completed checkpoint references no restart arrays.
    assert not RestartStore(checkpoint).files()
    assert not list(checkpoint.parent.glob("ckpt.curvature_*"))
    assert not list(checkpoint.parent.glob("ckpt.bfgs_*"))
    assert all(stage.success for stage in result.stages)


def test_restart_store_commit_removes_superseded_and_leftover_files(tmp_path):
    store = RestartStore(tmp_path / "run.h5")
    old = store.new_stage(0)
    old.put("stage.h5", dict(scaling=np.ones(3)))
    files = store.new_stage(1)
    files.put("pair_1.h5", dict(step=np.ones(3), difference=np.ones(3)))
    files.put("pair_2.h5", dict(step=np.ones(3), difference=np.ones(3)))
    (files.directory / ".pair_3.h5.deadbeef.tmp.h5").write_bytes(b"partial")
    record = dict(
        schema=_restart_store.RESTART_SCHEMA, directory=files.name, files=["pair_2.h5"]
    )
    store.commit(record)
    assert {p.name for p in store.files()} == {"pair_2.h5"}
    reopened = RestartStore(tmp_path / "run.h5").open_stage(record)
    arrays, _ = reopened.get("pair_2.h5")
    assert not arrays["step"].flags.writeable
    store.commit(None)
    assert not store.files()
    with pytest.raises(ValueError, match="invalid directory"):
        store.open_stage(dict(record, directory="../elsewhere"))
    # Before its first accepted pair an unscaled optimizer needs no file: the
    # checkpoint model is its iterate.
    model = np.arange(3.0)
    state = LBFGSRestart(
        model=model,
        steps=(),
        gradient_differences=(),
        pair_ids=(),
        history_size=4,
        accepted_iterations=0,
        initial_objective=2.0,
    )
    empty = store.new_stage(2)
    saved = empty.save_state(state, model=False)
    assert not empty.directory.exists() and saved["model"] is None
    record = dict(record, directory=empty.name, files=[])
    restored = store.open_stage(record).load_state(saved, model)
    np.testing.assert_array_equal(restored.model, model)
    assert restored.pair_count == 0 and not restored.model.flags.writeable


def test_transfer_kill_during_factorization_finalizes_before_the_next_stage(tmp_path):
    policy = im.CurvatureTransfer.warm_start(rank=3)
    expected = _fwi(
        tmp_path / "expected", TransferSite(), stages=3, policy=policy
    ).run()
    site = KillingSite(kill=2)  # stage 1's end-of-stage factorization
    options = dict(stages=3, policy=policy, checkpoint=tmp_path / "ckpt.h5")
    with pytest.raises(Killed):
        _fwi(tmp_path / "run", site, **options).run()
    saved = OptimizationCheckpoint.load(options["checkpoint"])
    assert saved.metadata["stage_index"] == 1
    assert saved.metadata["stage_finished"] and not saved.metadata["stage_completed"]
    resumed_site = KillingSite()
    workflow = _fwi(tmp_path / "run", resumed_site, **options)
    linearized = []
    solve_stage = workflow._solve_stage
    workflow._solve_stage = lambda index, *a, **k: (
        linearized.append(index) or solve_stage(index, *a, **k)
    )
    result = workflow.run()
    assert linearized == [2]  # stage 1 finalizes without optimizing again
    assert result.stages[1].metrics["finalized_after_resume"]
    assert result.stages[2].metrics["curvature_transfer"]["source_stage"] == 1
    np.testing.assert_array_equal(result.state.values, expected.state.values)
    assert resumed_site.factorizations == 1  # stage 2 is final: nothing else


def test_uncertainty_kill_during_final_factorization_publishes_the_posterior(tmp_path):
    uncertainty = im.BFGSUncertainty()
    expected = _fwi(
        tmp_path / "expected", TransferSite(), uncertainty=uncertainty
    ).run()
    options = dict(uncertainty=uncertainty, checkpoint=tmp_path / "ckpt.h5")
    with pytest.raises(Killed):
        _fwi(tmp_path / "run", KillingSite(kill=1), **options).run()
    result = _fwi(tmp_path / "run", KillingSite(), **options).run()
    assert [s.skipped for s in result.stages] == [True, False]
    assert result.stages[-1].uncertainty is not None
    identity = np.eye(result.uncertainty.space.size)
    np.testing.assert_array_equal(
        result.uncertainty.covariance.matmat(identity),
        expected.uncertainty.covariance.matmat(identity),
    )
    again = _fwi(tmp_path / "run", KillingSite(), **options).run()
    assert all(s.skipped for s in again.stages)
    assert again.stages[-1].uncertainty is not None


def test_early_stop_then_kill_keeps_the_optimizer_status_without_reoptimizing(
    tmp_path,
):
    options = dict(
        uncertainty=im.BFGSUncertainty(),
        checkpoint=tmp_path / "ckpt.h5",
        optimizer=im.LBFGS(gradient_tolerance=1e30),
        iterations=5,
    )
    with pytest.raises(Killed):
        _fwi(tmp_path, KillingSite(kill=1), **options).run()
    saved = OptimizationCheckpoint.load(options["checkpoint"])
    assert saved.metadata["stage_iteration"] < 5 and saved.metadata["stage_finished"]
    site = KillingSite()
    workflow = _fwi(tmp_path, site, **options)
    submitted = len(site.submissions)
    result = workflow.run()
    final = result.stages[-1]
    assert final.metrics["finalized_after_resume"] and final.iterations == 0
    assert final.message == saved.metadata["stage_message"]
    assert final.status == saved.metadata["stage_status"] == 1
    assert final.uncertainty is not None
    assert len(site.submissions) == submitted  # no objective evaluation


@pytest.mark.parametrize(
    "uncertainty, policies, expected",
    [
        # Transfer only: the final stage seeds nothing.
        (None, ["refresh", "refresh"], 1),
        # A reset successor needs no factors either.
        (None, ["refresh", "reset"], 0),
        # Uncertainty on every stage shares its factorization with transfer.
        (im.BFGSUncertainty(stages="all"), ["refresh", "refresh"], 2),
        (im.BFGSUncertainty(stages="all", rank=1), ["refresh", "refresh"], 2),
    ],
)
def test_each_stage_factorizes_at_most_once_and_only_when_used(
    tmp_path, uncertainty, policies, expected
):
    site = TransferSite()
    workflow = _fwi(
        tmp_path,
        site,
        policy=im.CurvatureTransfer.refresh(rank=3),
        uncertainty=uncertainty,
    )
    for i, method in enumerate(policies):
        workflow.stages[i] = im.Stage(
            workflow.stages[i].frequencies,
            3,
            curvature=(
                im.CurvatureTransfer.reset()
                if method == "reset"
                else im.CurvatureTransfer.refresh(rank=3)
            ),
        )
    result = workflow.run()
    requests = [
        request
        for request, _ in site.__dict__.get("transfer_calls", [])
        if request["method"] == "bfgs_rsvd"
    ]
    assert len(requests) == expected
    if uncertainty is not None:
        # One factorization serves the posterior and the next stage's seed.
        assert requests[0].get("rank") == (None if uncertainty.rank is None else 3)
        assert result.stages[1].metrics["curvature_transfer"]["source_stage"] == 0
        assert all(stage.uncertainty is not None for stage in result.stages)


def test_workflow_smoothing_override_survives_resume(tmp_path):
    def build(site, callback=None):
        problem = _problem(
            tmp_path,
            site,
            misfit=im.Misfit.l2(noise_std=1),
            frequencies=(4, 6),
            smoothing={"kind": "tikhonov", "alpha": 0.1},
        )
        problem.linearize()
        return im.FWI(
            problem,
            [im.Stage([4], 3), im.Stage([6], 3)],
            optimizer=im.LBFGS(**TIGHT),
            smoothing=False,
            scaling="curvature",
            curvature=im.CurvatureTransfer.refresh(rank=2),
            checkpoint=tmp_path / "ckpt.h5",
            callback=callback,
        )

    def stop(event):
        if event.stage_index == 1 and event.stage_iteration == 1:
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        build(TransferSite(), stop).run()
    result = build(TransferSite()).run()
    assert result.stages[1].resumed and not result.stages[1].skipped


def test_legacy_json_checkpoint_is_rejected_with_a_clear_message(tmp_path):
    path = tmp_path / "ckpt.h5"
    _fwi(tmp_path, TransferSite(), checkpoint=path, stages=1).run()
    saved = OptimizationCheckpoint.load(path)
    legacy = dict(saved.metadata, schema="fs-imaging-fwi-checkpoint-1")
    OptimizationCheckpoint(
        saved.model, saved.iteration, saved.evaluations, saved.loss, legacy
    ).save(path)
    with pytest.raises(ValueError, match="kept optimizer restart state as JSON"):
        _fwi(tmp_path, TransferSite(), checkpoint=path, stages=1).run()


def test_restart_files_hold_float64_datasets_not_json(tmp_path):
    checkpoint = tmp_path / "ckpt" / "ckpt.h5"

    def stop(event):
        if event.stage_iteration == 2:
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        _fwi(
            tmp_path,
            TransferSite(),
            policy=im.CurvatureTransfer.refresh(rank=2),
            scaling="curvature",
            checkpoint=checkpoint,
            callback=stop,
        ).run()
    saved = OptimizationCheckpoint.load(checkpoint)
    record = json.loads(saved.metadata["restart"])
    assert len(saved.metadata["restart"]) < 1000
    directory = checkpoint.parent / "ckpt.restart" / record["directory"]
    size = saved.model.size
    with h5py.File(directory / f"pair_{record['optimizer']['pairs'][-1]}.h5") as h5:
        assert h5["step"].dtype == np.float64 and h5["step"].shape == (size,)
    with h5py.File(directory / record["optimizer"]["model"]) as h5:
        assert h5["model"].shape == (size,)
    with h5py.File(directory / "stage.h5") as h5:
        assert {"scaling", "base_inverse_diagonal"} <= set(h5)


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------


def test_small_array_fingerprints_keep_their_json_identity():
    values = np.linspace(0.0, 1.0, 50)
    mask = np.arange(50) % 3 == 0
    assert fingerprint(x=values) == fingerprint(x=values.tolist())
    # Boolean masks enter as 0/1 on both paths, like the former call sites.
    assert fingerprint(m=mask) == fingerprint(m=mask.astype(int).tolist())


def test_large_array_fingerprints_hash_canonical_bytes():
    rng = np.random.default_rng(3)
    values = rng.normal(size=5000)
    identity = _jsonable(values)
    assert set(identity) == {"fs_array", "kind", "shape"}
    assert identity["kind"] == "float" and identity["shape"] == [5000]
    # Canonical float64: a float32 field matches its exact widening.
    narrow = values.astype(np.float32)
    assert fingerprint(x=narrow) == fingerprint(x=narrow.astype(np.float64))
    changed = values.copy()
    changed[1234] = np.nextafter(changed[1234], np.inf)
    assert fingerprint(x=changed) != fingerprint(x=values)
    assert fingerprint(x=values.reshape(50, 100)) != fingerprint(x=values)
    mask = values > 0
    assert _jsonable(mask)["kind"] == "bool"
    assert fingerprint(m=mask) != fingerprint(m=~mask)
    assert _jsonable(values + 1j * values)["kind"] == "complex"
    assert _jsonable(np.arange(5000, dtype=np.int32)) == _jsonable(np.arange(5000))
    # Object and other arrays keep the JSON path.
    assert isinstance(_jsonable(np.array(["a"] * 5000, dtype=object)), list)


def test_state_epoch_matches_the_file_representation_identity(tmp_path):
    problem = _problem(tmp_path, TransferSite())
    problem.linearize()
    state = problem.state
    saved = state.to_file()
    legacy = fingerprint(
        blocks=saved.blocks,
        control_spaces=saved.control_spaces,
        scaling=saved.scaling,
        scaling_units=saved.scaling_units,
    ).removeprefix("sha256:")
    assert _state_epoch(state) == legacy


def test_custom_penalty_declaration_is_bound_once_per_stage(tmp_path):
    binds = []

    class CountedPenalty(im.Regularization):
        def bind(self, space):
            binds.append(space.size)
            bound = im.Quadratic(np.eye(space.size), weight=0.1).bind(space)
            bound.identity = "counted-penalty"
            return bound

    iterations = 5
    workflow = _fwi(
        tmp_path,
        TransferSite(),
        policy=im.CurvatureTransfer.refresh(rank=2),
        iterations=iterations,
        checkpoint=tmp_path / "ckpt.h5",
    )
    workflow.regularization = CountedPenalty()
    result = workflow.run()
    checkpoints = sum(1 + stage.iterations for stage in result.stages)
    assert checkpoints > 2 * len(result.stages)
    # One bind for the objective and one for the checkpoint declaration.
    assert len(binds) == 2 * len(result.stages)


def test_history_adopts_handed_over_seed_blocks_and_copies_writable_ones(tmp_path):
    from frequensolve.imaging.curvature import BFGSHistory

    block = np.arange(1.0, 7.0).reshape(2, 3)
    block.flags.writeable = False
    eigenvalues = np.array([0.5, 0.25])
    history = BFGSHistory(
        [1.0, 2.0, 3.0],
        state="s",
        coordinates="c",
        seed_modes=block,
        seed_eigenvalues=eigenvalues,
    )
    # A read-only block is handed over: no controls-by-rank copy.
    assert np.shares_memory(history.seed_modes, block)
    writable = np.arange(1.0, 7.0).reshape(2, 3)
    other = BFGSHistory(
        [1.0, 2.0, 3.0],
        state="s",
        coordinates="c",
        seed_modes=writable,
        seed_eigenvalues=eigenvalues,
    )
    writable[:] = 0
    np.testing.assert_array_equal(other.seed_modes, block)
    loaded = BFGSHistory.load(history.save(tmp_path / "history.h5"))
    np.testing.assert_array_equal(loaded.seed_modes, block)
    assert not loaded.seed_modes.flags.writeable
    bad = np.array([[1.0, np.nan, 2.0]])
    bad.flags.writeable = False
    with pytest.raises(ValueError, match="finite"):
        BFGSHistory(
            [1.0, 2.0, 3.0],
            state="s",
            coordinates="c",
            seed_modes=bad,
            seed_eigenvalues=[1.0],
        )


def test_factor_verification_streams_leaf_tiles(tmp_path):
    import tracemalloc

    from frequensolve.imaging._block_digest import block_digest
    from frequensolve.imaging.curvature import CurvatureResult

    modes = np.random.default_rng(5).normal(size=(2, 2_000_000))
    path = tmp_path / "factors.h5"
    with h5py.File(path, "w") as h5:
        h5["modes"] = modes
    result = CurvatureResult(path, dict(output_digests={"/modes": block_digest(modes)}))
    size = modes.nbytes
    del modes
    tracemalloc.start()
    assert result.verify() == result.output_digests
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # A few row tiles are resident, never the whole controls-by-rank block.
    assert peak < size / 2
    with h5py.File(path, "a") as h5:
        h5["modes"][1, 1_999_999] += 1.0
    with pytest.raises(ValueError, match="changed after they were recorded"):
        result.verify()
