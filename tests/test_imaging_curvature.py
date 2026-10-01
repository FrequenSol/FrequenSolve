# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""History integrity and native staging, without duplicating backend algebra."""

import hashlib
import json
import os
import struct
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from frequensolve.imaging import BFGSHistory, NativeCurvature
from frequensolve.imaging._block_digest import block_digest, stacked_block_digest
from frequensolve.imaging.curvature import CurvatureResult
from frequensolve.inversion.optimization import (
    LBFGSOptions,
    LBFGSRestart,
    minimize_lbfgs,
)

pytestmark = pytest.mark.unit

# Element type Sauce reads each dataset into; every other dataset is float64.
SAUCE_TYPES = dict(
    offsets="<i8",
    indices="<i4",
    source_roots="<i4",
    target_roots="<i4",
    grid_shape="<i4",
    smoothing_radii="<i4",
)
FACTOR_METHODS = {"bfgs_rsvd", "curvature_refresh", "curvature_warm_start"}


def spec_digest(values, dtype="<f8"):
    """fs-block-sha256-1, written independently of the SDK: 65536-row leaves over every column."""
    array = np.ascontiguousarray(values, dtype=dtype)
    rows = array.shape[-1] if array.ndim else 1
    table = array.reshape(int(np.prod(array.shape[:-1])), rows)
    leaves = b"".join(
        hashlib.sha256(table[:, start : start + 65536].tobytes()).digest()
        for start in range(0, rows, 65536)
    )
    header = struct.pack(
        f"<18s3s{array.ndim + 2}Q",
        b"fs-block-sha256-1",
        np.dtype(dtype).str[1:].encode(),
        array.ndim,
        *array.shape,
        65536,
    )
    return "fs-block-sha256-1:" + hashlib.sha256(header + leaves).hexdigest()


def spec_digests(arrays):
    return {
        f"/{name}": spec_digest(value, SAUCE_TYPES.get(name, "<f8"))
        for name, value in arrays.items()
    }


def write_result(request, metadata, outputs):
    """Publish a fake Sauce result; reusable factors record their dataset digests."""
    if request["method"] in FACTOR_METHODS:
        metadata["output_digests"] = spec_digests(outputs)
    with h5py.File(request["output"], "w") as h5:
        h5["metadata"] = np.bytes_(json.dumps(metadata))
        for name, value in outputs.items():
            h5[name] = value


def solver_output(request):
    """Read a request like Sauce and return its arrays and reported metadata.

    Counts and digests describe what was actually read (history arrays are
    followed through their external links), not values echoed from the request.
    """
    with h5py.File(request["input"], "r") as h5:
        arrays = {key: h5[key][()] for key in h5 if key != "metadata"}
        metadata = json.loads(h5["metadata"][()])
    output = dict(
        metadata,
        schema="fs-curvature-output-1",
        dtype="float64",
        input_digests=spec_digests(arrays),
    )
    method = request["method"]
    if method.startswith("bfgs"):
        output.update(
            controls=arrays["base_inverse_diagonal"].size,
            history_pairs=len(arrays["steps"]),
        )
    elif method.startswith("covariance"):
        with h5py.File(request["factors"], "r") as h5:
            output["controls"] = h5["prior_std"].size
            output["factors_digests"] = spec_digests(
                {
                    name: h5[name][()]
                    for name in (
                        "base_inverse_diagonal",
                        "prior_std",
                        "modes",
                        "eigenvalues",
                    )
                    if name in h5
                }
            )
    elif method == "gaussian_prior":
        output["controls"] = arrays["point"].size
    elif method == "rickett":
        output.update(controls=arrays["reference"].size, padding=request["padding"])
    elif method.startswith("mesh"):
        with h5py.File(request["target_mesh"], "r") as h5:
            output["controls"] = int(h5["property_space/header"][1])
    return arrays, output


def checkpoint(pairs, accepted):
    """An L-BFGS iteration event retaining ``pairs`` after ``accepted`` iterations."""
    steps = tuple(np.asarray(p["step"], dtype=float) for p in pairs)
    differences = tuple(np.asarray(p["difference"], dtype=float) for p in pairs)
    return SimpleNamespace(
        optimizer_state=LBFGSRestart(
            model=np.zeros(steps[0].size if steps else 1),
            steps=steps,
            gradient_differences=differences,
            pair_ids=tuple(range(accepted - len(steps) + 1, accepted + 1)),
            history_size=max(len(steps), 1),
            accepted_iterations=accepted,
            initial_objective=1.0,
        )
    )


def test_history_real_optimizer_and_round_trip(tmp_path):
    history = BFGSHistory([0.2, 0.1], state="quadratic-final", coordinates="linear")
    diagonal = np.array([3.0, 7.0])
    minimize_lbfgs(
        lambda x: 0.5 * np.dot(x * diagonal, x),
        lambda x: diagonal * x,
        [1.0, 2.0],
        preconditioner=lambda x, g: history.base_inverse_diagonal * g,
        options=LBFGSOptions(
            history_size=20, max_iterations=10, gradient_tolerance=1e-6
        ),
        callback=history,
    )
    assert len(history.steps) > 0
    restored = BFGSHistory.load(history.save(tmp_path / "history.h5"))
    np.testing.assert_array_equal(restored.steps, history.steps)
    np.testing.assert_array_equal(
        restored.gradient_differences, history.gradient_differences
    )
    assert restored.coordinates == history.coordinates


@pytest.mark.parametrize("pairs", [[], [dict(step=[0.0, 1.0], difference=[0.0, 2.0])]])
def test_history_rejects_reset_or_truncation(pairs):
    history = BFGSHistory([1.0, 1.0], state="s", coordinates="c")
    history(checkpoint([dict(step=[1.0, 0.0], difference=[2.0, 0.0])], 1))
    with pytest.raises(ValueError, match="truncated or reset"):
        history(checkpoint(pairs, 2))


def test_history_stage_and_optimizer_guards():
    history = BFGSHistory([1.0], state="s", coordinates="c")
    history(SimpleNamespace(stage_index=0, diagnostics=checkpoint([], 0)))
    with pytest.raises(ValueError, match="separate history per stage"):
        history(SimpleNamespace(stage_index=1, diagnostics=checkpoint([], 0)))
    with pytest.raises(ValueError, match="L-BFGS"):
        BFGSHistory([1.0], state="s", coordinates="c")(
            SimpleNamespace(optimizer_state=None)
        )


def test_stage_identity_survives_history_round_trip(tmp_path):
    history = BFGSHistory([1.0], state="s", coordinates="c")
    history(SimpleNamespace(stage_index=2, diagnostics=checkpoint([], 0)))
    restored = BFGSHistory.load(history.save(tmp_path / "history.h5"))
    with pytest.raises(ValueError, match="separate history per stage"):
        restored(SimpleNamespace(stage_index=3, diagnostics=checkpoint([], 0)))


@pytest.mark.parametrize(
    "options", [{"rank": 1.5}, {"oversampling": True}, {"seed": 2**32}]
)
def test_integer_options_do_not_silently_change_request(tmp_path, options):
    calls = []
    backend = NativeCurvature(workdir=tmp_path, runner=fake_runner(calls))
    with pytest.raises(ValueError):
        backend.bfgs_uncertainty(
            BFGSHistory([1.0], state="s", coordinates="c"), **options
        )
    assert not calls


def fake_runner(calls, mismatch=None, missing=()):
    def run(path):
        request = json.loads(path.read_text())
        arrays, output = solver_output(request)
        outputs = dict(marker=[17.0])
        if request["method"] == "bfgs_rsvd":
            output["rank"] = 0
            base = arrays["base_inverse_diagonal"]
            outputs.update(
                base_inverse_diagonal=base,
                prior_std=arrays["prior_std"],
                variance=base * arrays["prior_std"] ** 2,
                standard_deviation=np.sqrt(base) * arrays["prior_std"],
            )
            output["output_digests"] = spec_digests(outputs)
        calls.append((request, arrays, dict(output)))
        output.update(mismatch or {})
        for key in missing:
            del output[key]
        with h5py.File(request["output"], "w") as h5:
            h5["metadata"] = np.bytes_(json.dumps(output))
            for name, value in outputs.items():
                h5[name] = value

    return run


def test_bfgs_staging_preserves_backend_inputs(tmp_path):
    calls = []
    backend = NativeCurvature(workdir=tmp_path, runner=fake_runner(calls))
    history = BFGSHistory([0.2, 0.5], state="s", coordinates="c")
    history(checkpoint([dict(step=[1.0, 0.0], difference=[3.0, 0.0])], 1))
    result = backend.bfgs_uncertainty(history, prior_std=0.1, rank=2, seed=7)
    request, arrays, _ = calls[0]
    assert request["method"] == "bfgs_rsvd"
    assert request["rank"] == 2
    np.testing.assert_array_equal(arrays["steps"], history.steps)
    np.testing.assert_array_equal(arrays["prior_std"], [0.1, 0.1])
    assert result.path.name == "result.h5"
    np.testing.assert_array_equal(result.read("marker"), [17.0])
    backend.inverse_action(history, [1.0, 2.0])
    np.testing.assert_array_equal(calls[1][1]["vectors"], [[1.0, 2.0]])
    assert calls[0][0]["output"] != calls[1][0]["output"]


def test_rickett_normal_scheduling_and_depth_axis_packing(tmp_path):
    calls, directions = [], []

    def normal(direction):
        directions.append(direction.copy())
        return direction * 3

    backend = NativeCurvature(workdir=tmp_path, runner=fake_runner(calls))
    reference = np.arange(24.0).reshape(4, 2, 3)
    backend.rickett(
        reference,
        reference,
        normal=normal,
        state="s",
        coordinates="c",
        depth_axis=0,
        smoothing_radii=(2, 1, 0),
        damping=0.2,
    )
    np.testing.assert_array_equal(directions[0], reference.ravel())
    arrays = calls[0][1]
    np.testing.assert_array_equal(
        arrays["reference"], np.moveaxis(reference, 0, -1).ravel()
    )
    np.testing.assert_array_equal(arrays["normal_reference"], 3 * arrays["reference"])
    np.testing.assert_array_equal(arrays["grid_shape"], [4, 3, 2])
    np.testing.assert_array_equal(arrays["smoothing_radii"], [2, 0, 1])
    with pytest.raises(ValueError, match="exactly one"):
        backend.rickett(reference, reference, state="s", coordinates="c")


def test_mismatched_result_is_not_published(tmp_path):
    backend = NativeCurvature(
        workdir=tmp_path, runner=fake_runner([], dict(state="old"))
    )
    history = BFGSHistory([1.0], state="s", coordinates="c")
    with pytest.raises(ValueError, match="mismatched state"):
        backend.bfgs_uncertainty(history)
    assert not list(tmp_path.rglob("result.h5"))


@pytest.mark.parametrize(
    ("mismatch", "missing", "message"),
    [
        (
            dict(input_digests={"/vectors": "fs-block-sha256-1:" + "0" * 64}),
            (),
            "mismatched input_digests",
        ),
        (dict(controls=3), (), "mismatched controls"),
        (dict(history_pairs=0), (), "mismatched history_pairs"),
        (dict(dtype="float32"), (), "mismatched dtype"),
        (None, ("input_digests",), "lacks input_digests"),
        (None, ("history_pairs",), "lacks history_pairs"),
    ],
)
def test_solver_must_report_what_it_read(tmp_path, mismatch, missing, message):
    backend = NativeCurvature(
        workdir=tmp_path, runner=fake_runner([], mismatch, missing)
    )
    history = BFGSHistory([1.0, 2.0], state="s", coordinates="c")
    history(checkpoint([dict(step=[1.0, 0.0], difference=[3.0, 0.0])], 1))
    with pytest.raises(ValueError, match=message):
        backend.inverse_action(history, [1.0, 2.0])
    assert not list(tmp_path.rglob("result.h5"))


def test_history_is_staged_once_per_digest_and_linked(tmp_path):
    calls, links = [], []

    def runner(path):
        # Verified inputs are deleted: record the link while Sauce reads it.
        with h5py.File(json.loads(path.read_text())["input"], "r") as h5:
            links.append(h5.get("steps", getlink=True))
        fake_runner(calls)(path)

    backend = NativeCurvature(workdir=tmp_path, runner=runner)
    history = BFGSHistory([0.2, 0.5], state="s", coordinates="c")
    history(checkpoint([dict(step=[1.0, 0.0], difference=[3.0, 0.0])], 1))
    with pytest.raises(ValueError, match="read-only"):
        history.steps[0, 0] = 2.0
    staged = backend.workdir / "histories" / f"{history.digest}.h5"
    with backend.retain(history):
        backend.inverse_action(history, [1.0, 2.0])
        written = staged.stat().st_mtime_ns
        backend.inverse_action(history, [[3.0, 4.0], [5.0, 6.0]])
        backend.bfgs_uncertainty(history)
        assert list(staged.parent.iterdir()) == [staged]
        assert staged.stat().st_mtime_ns == written
    # The block's end deletes the staged history; the results never link it,
    # nor their inputs, which are deleted once each result is verified.
    assert not list(staged.parent.iterdir())
    assert not list(tmp_path.rglob("input.h5"))
    assert len(list(tmp_path.rglob("result.h5"))) == len(calls) == 3
    for (request, arrays, _), link in zip(calls, links):
        assert isinstance(link, h5py.ExternalLink)
        # Relative links keep the request and histories relocatable together.
        assert link.filename == f"../histories/{staged.name}"
        np.testing.assert_array_equal(arrays["steps"], history.steps)
        np.testing.assert_array_equal(
            arrays["gradient_differences"], history.gradient_differences
        )
    # A repeated checkpoint keeps the digest; a new pair stages a new file.
    digest = history.digest
    history(checkpoint([dict(step=[1.0, 0.0], difference=[3.0, 0.0])], 1))
    assert history.digest == digest
    history(
        checkpoint(
            [
                dict(step=[1.0, 0.0], difference=[3.0, 0.0]),
                dict(step=[0.0, 1.0], difference=[0.0, 2.0]),
            ],
            2,
        )
    )
    assert history.digest != digest
    backend.inverse_action(history, [1.0, 2.0])
    assert calls[-1][2]["history_pairs"] == 2
    assert links[-1].filename == f"../histories/{history.digest}.h5" != staged.name
    # Outside a retain block every operation deletes the history it staged.
    assert not list(staged.parent.iterdir())
    restored = BFGSHistory.load(history.save(tmp_path / "archive.h5"))
    assert restored.digest == history.digest


def test_factor_digests_are_recorded_and_checked_without_rehashing(tmp_path):
    calls = []
    backend = NativeCurvature(workdir=tmp_path, runner=fake_runner(calls))
    history = BFGSHistory([0.2, 0.5], state="s", coordinates="c")
    factors = backend.bfgs_uncertainty(history, prior_std=[1.0, 2.0], rank=0)
    with h5py.File(factors.path, "r") as h5:
        written = {name: h5[name][()] for name in h5 if name != "metadata"}
    assert factors.output_digests == spec_digests(written)
    assert factors.verify() == factors.output_digests
    backend.covariance(factors, vectors=[1.0, 2.0])
    request, _, reported = calls[-1]
    assert request["factors"] == str(factors.path)
    assert set(reported["factors_digests"]) == {"/base_inverse_diagonal", "/prior_std"}
    lying = NativeCurvature(
        workdir=tmp_path / "lying",
        runner=fake_runner(
            [], dict(factors_digests={"/prior_std": "fs-block-sha256-1:" + "0" * 64})
        ),
    )
    with pytest.raises(ValueError, match="mismatched factors_digests"):
        lying.covariance(factors, vectors=[1.0, 2.0])
    # A factor file changed after production is caught by what Sauce reports
    # reading, without the SDK rehashing the file first.
    with h5py.File(factors.path, "a") as h5:
        h5["prior_std"][0] = 3.0
    with pytest.raises(ValueError, match="mismatched factors_digests"):
        backend.covariance(factors, vectors=[1.0, 2.0])
    assert len(list(tmp_path.rglob("result.h5"))) == 2
    with pytest.raises(ValueError, match="changed after they were recorded"):
        factors.verify()
    unrecorded = CurvatureResult(
        factors.path, dict(factors.metadata, output_digests={})
    )
    count = len(calls)
    with pytest.raises(ValueError, match="no output_digests"):
        backend.covariance(unrecorded, vectors=[1.0, 2.0])
    assert len(calls) == count


def test_input_digests_must_name_exactly_the_staged_datasets(tmp_path):
    history = BFGSHistory([1.0, 2.0], state="s", coordinates="c")
    history(checkpoint([dict(step=[1.0, 0.0], difference=[3.0, 0.0])], 1))

    def edited(change):
        def run(path):
            request = json.loads(path.read_text())
            arrays, output = solver_output(request)
            change(output["input_digests"])
            with h5py.File(request["output"], "w") as h5:
                h5["metadata"] = np.bytes_(json.dumps(output))

        return run

    for change in (
        lambda digests: digests.pop("/vectors"),
        lambda digests: digests.update(extra=digests["/vectors"]),
    ):
        backend = NativeCurvature(workdir=tmp_path, runner=edited(change))
        with pytest.raises(ValueError, match="mismatched input_digests"):
            backend.inverse_action(history, [1.0, 2.0])
    assert not list(tmp_path.rglob("result.h5"))


def test_history_digests_are_computed_once_per_content(tmp_path, monkeypatch):
    import frequensolve.imaging.curvature as curvature

    hashed = []

    def counted(values, dtype=np.float64):
        hashed.append(np.shape(values))
        return block_digest(values, dtype)

    def counted_stack(vectors, length, dtype=np.float64):
        # Secant datasets are hashed from the shared pair vectors, unstacked.
        hashed.append((len(vectors), length))
        return stacked_block_digest(vectors, length, dtype)

    monkeypatch.setattr(curvature, "block_digest", counted)
    monkeypatch.setattr(curvature, "stacked_block_digest", counted_stack)
    history = BFGSHistory(
        [0.2, 0.5],
        state="s",
        coordinates="c",
        seed_modes=[[1.0, 0.0]],
        seed_eigenvalues=[0.5],
    )
    history(checkpoint([dict(step=[1.0, 0.0], difference=[3.0, 0.0])], 1))
    backend = NativeCurvature(
        workdir=tmp_path, runner=fake_runner([], dict(seed_rank=1))
    )
    for _ in range(3):
        backend.inverse_action(history, [1.0, 2.0])
    # Five history datasets once, plus the staged vectors of each request.
    assert sorted(hashed[:5]) == [(1,), (1, 2), (1, 2), (1, 2), (2,)]
    assert hashed[5:] == [(1, 2)] * 3
    assert history.dataset_digests == spec_digests(
        dict(
            base_inverse_diagonal=history.base_inverse_diagonal,
            steps=history.steps,
            gradient_differences=history.gradient_differences,
            seed_modes=history.seed_modes,
            seed_eigenvalues=history.seed_eigenvalues,
        )
    )
    del hashed[:]
    history(
        checkpoint(
            [
                dict(step=[1.0, 0.0], difference=[3.0, 0.0]),
                dict(step=[0.0, 1.0], difference=[0.0, 2.0]),
            ],
            2,
        )
    )
    history.dataset_digests
    # Only the grown secant arrays are rehashed.
    assert hashed == [(2, 2), (2, 2)]


def test_tampered_linked_history_is_rejected(tmp_path):
    calls = []
    backend = NativeCurvature(workdir=tmp_path, runner=fake_runner(calls))
    history = BFGSHistory([0.2, 0.5], state="s", coordinates="c")
    history(checkpoint([dict(step=[1.0, 0.0], difference=[3.0, 0.0])], 1))
    staged = backend.workdir / "histories" / f"{history.digest}.h5"
    with backend.retain(history):
        backend.inverse_action(history, [1.0, 2.0])
        with h5py.File(staged, "a") as h5:
            h5["steps"][0, 1] = 1e-3
        with pytest.raises(ValueError, match="mismatched input_digests"):
            backend.inverse_action(history, [1.0, 2.0])
    assert len(list(tmp_path.rglob("result.h5"))) == 1
    # The tampered file is gone, so the next operation restages it intact.
    assert not staged.exists()
    backend.inverse_action(history, [1.0, 2.0])
    assert len(list(tmp_path.rglob("result.h5"))) == 2


@pytest.mark.parametrize("failure", ["runner", "verification"])
def test_failed_operation_deletes_its_staged_history(tmp_path, failure):
    def runner(path):
        if failure == "runner":
            raise RuntimeError("solver crashed")
        fake_runner([], mismatch=dict(history_pairs=7))(path)

    backend = NativeCurvature(workdir=tmp_path, runner=runner)
    history = BFGSHistory([0.2, 0.5], state="s", coordinates="c")
    history(checkpoint([dict(step=[1.0, 0.0], difference=[3.0, 0.0])], 1))
    with pytest.raises((RuntimeError, ValueError)):
        backend.bfgs_uncertainty(history)
    assert not list((tmp_path / "histories").iterdir())


def test_operations_in_flight_share_one_staged_history(tmp_path):
    calls, listings = [], []
    history = BFGSHistory([0.2, 0.5], state="s", coordinates="c")
    history(checkpoint([dict(step=[1.0, 0.0], difference=[3.0, 0.0])], 1))
    staged = tmp_path / "histories" / f"{history.digest}.h5"

    def runner(path):
        if not calls:
            # A second operation on the same history while this one runs, from
            # another instance (Backend.curvature() returns a new one per call).
            identity = staged.stat().st_ino, staged.stat().st_mtime_ns
            NativeCurvature(workdir=tmp_path, runner=fake_runner(calls)).inverse_action(
                history, [1.0, 2.0]
            )
            listings.append(list(staged.parent.iterdir()))
            assert (staged.stat().st_ino, staged.stat().st_mtime_ns) == identity
        fake_runner(calls)(path)

    NativeCurvature(workdir=tmp_path, runner=runner).bfgs_uncertainty(history)
    assert [request["method"] for request, _, _ in calls] == [
        "bfgs_action",
        "bfgs_rsvd",
    ]
    # The nested operation left the file to the one still running.
    assert listings == [[staged]]
    assert not staged.exists()


def test_rickett_sends_explicit_reflected_padding_and_damping(tmp_path):
    calls = []
    backend = NativeCurvature(workdir=tmp_path, runner=fake_runner(calls))
    reference = np.arange(1.0, 13.0).reshape(3, 4)

    def request(**options):
        backend.rickett(
            reference,
            reference,
            normal_reference=reference,
            state="s",
            coordinates="c",
            depth_axis=1,
            **options,
        )
        sent = calls[-1][0]
        return sent["padding"], sent.get("damping"), sent.get("relative_damping")

    # Exactly one damping key is sent; Sauce rejects requests carrying both.
    assert request() == (4, None, 0.01)
    assert request(damping=0.2) == (4, 0.2, None)
    assert request(relative_damping=0.0, padding=0) == (0, None, 0.0)
    assert request(damping=0.0, padding=2) == (2, 0.0, None)
    count = len(calls)
    with pytest.raises(ValueError, match="mutually exclusive"):
        request(damping=0.0, relative_damping=0.1)
    with pytest.raises(ValueError, match="relative_damping"):
        request(relative_damping=-1.0)
    assert len(calls) == count
    lying = NativeCurvature(workdir=tmp_path, runner=fake_runner([], dict(padding=0)))
    with pytest.raises(ValueError, match="mismatched padding"):
        lying.rickett(
            reference, reference, normal_reference=reference, state="s", coordinates="c"
        )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("FS_CURVATURE_SOLVER"), reason="Requires a Sauce curvature build"
)
def test_real_optimizer_history_through_native_backend(tmp_path):
    history = BFGSHistory([1.0], state="s", coordinates="prior-whitened")
    minimize_lbfgs(
        lambda x: 1.5 * np.dot(x, x),
        lambda x: 3 * x,
        [1.0],
        preconditioner=lambda x, g: history.base_inverse_diagonal * g,
        options=LBFGSOptions(history_size=10, max_iterations=5),
        callback=history,
    )
    backend = NativeCurvature(os.environ["FS_CURVATURE_SOLVER"], workdir=tmp_path)
    result = backend.bfgs_uncertainty(history, prior_std=0.1)
    np.testing.assert_allclose(result.read("variance"), [0.01 / 3], rtol=1e-12)
    np.testing.assert_allclose(
        backend.inverse_action(history, [1.0]).read("actions"), [[1 / 3]], rtol=1e-12
    )
    reference = np.cos(2 * np.pi * np.arange(16) / 16)
    # An undamped periodic envelope of a whole-period cosine is exactly one.
    result = backend.rickett(
        reference,
        4 * reference,
        normal=lambda x: 4 * x,
        state="s",
        coordinates="regular-grid",
        damping=0.0,
        padding=0,
    )
    np.testing.assert_allclose(
        result.read("normalized"), reference, rtol=5e-6, atol=1e-7
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("FS_CURVATURE_SOLVER"), reason="Requires a Sauce curvature build"
)
def test_native_digests_reuse_factors_and_reject_a_tampered_linked_history(tmp_path):
    rng = np.random.default_rng(3)
    size = 65536 + 100  # Leaves straddle the end of the first 65536-row block.
    history = BFGSHistory(rng.uniform(0.5, 2.0, size), state="s", coordinates="c")
    steps = rng.normal(size=(2, size))
    history(
        checkpoint(
            [dict(step=s.tolist(), difference=(2 * s).tolist()) for s in steps], 2
        )
    )
    backend = NativeCurvature(os.environ["FS_CURVATURE_SOLVER"], workdir=tmp_path)
    staged = tmp_path / "histories" / f"{history.digest}.h5"
    with backend.retain(history):
        factors = backend.bfgs_uncertainty(history, prior_std=0.5)
        assert factors.verify() == factors.output_digests
        applied = backend.covariance(factors, vectors=rng.normal(size=(2, size)))
        assert set(applied.metadata["factors_digests"]) == {
            "/base_inverse_diagonal",
            "/prior_std",
            "/modes",
            "/eigenvalues",
        }
        with h5py.File(staged, "a") as h5:
            h5["steps"][1, size - 1] *= 1.5
        with pytest.raises(ValueError, match="mismatched input_digests"):
            backend.inverse_action(history, rng.normal(size=size))
    # Factors never link the staged history, which is gone after the block.
    assert not staged.exists() and factors.verify() == factors.output_digests
