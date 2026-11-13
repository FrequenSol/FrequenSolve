"""Version-5 objective states: dataset digests, canonical JSON and the state reader.

Sauce identifies every dataset of an ``fs-objective-linearization-5`` cache and
``fs-objective-vector-5`` file by the ``fs-block-sha256-1`` digest of the
dataset flattened to one axis, and fingerprints manifests by the SHA-256 of
``nlohmann::json::dump()`` text. These tests pin both definitions against the
contract text and Sauce's published example, then exercise the SDK reader.
"""

import hashlib
import json
import os
import struct
from pathlib import Path

import h5py
import numpy as np
import pytest
from jsonschema import Draft202012Validator

from frequensolve.imaging._block_digest import block_digest, flat_block_digest
from frequensolve.imaging._canonical_json import nlohmann_double
from frequensolve.imaging._objective import ObjectiveState
from frequensolve.imaging.data import canonical_json_sha256
from tests.imaging_fakes import FakeImagingSite, reseal_state, seal_state_manifest
from tests.test_imaging_problem import _problem

CONTRACTS = (
    Path(__file__).parent
    / "contracts"
    / "sauce-f533e6f"
    / "trunk"
    / "contracts"
    / "outputs"
)
STATE_CONTRACT = CONTRACTS / "fs-objective-linearization-5"


def spec_flat_digest(values, dtype):
    """``fs-block-sha256-1`` of a dataset flattened to one axis, from the contract text."""
    flat = np.ascontiguousarray(values, dtype=np.dtype(dtype).newbyteorder("<"))
    flat = flat.reshape(-1)
    leaves = b"".join(
        hashlib.sha256(flat[start : start + 65536].tobytes()).digest()
        for start in range(0, flat.size, 65536)
    )
    header = (
        b"fs-block-sha256-1\0"
        + np.dtype(dtype).str[1:].encode()
        + b"\0"
        + struct.pack("<3Q", 1, flat.size, 65536)
    )
    return "fs-block-sha256-1:" + hashlib.sha256(header + leaves).hexdigest()


# ---------------------------------------------------------------------------
# dataset digests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "values",
    [
        np.arange(120_000, dtype=np.int32).reshape(40_000, 3),  # two leaves
        np.linspace(-1, 1, 80_002, dtype=np.float32).reshape(40_001, 2),
        np.array([7], dtype=np.int64),
        np.array(3, dtype=np.int32),  # a scalar is one element
        np.zeros((0, 2), dtype=np.float32),
        np.random.default_rng(1).standard_normal((5, 4)),
    ],
    ids=["i4-keys", "f4-values", "i8", "scalar", "empty", "f8-table"],
)
def test_flat_digests_follow_the_contract_definition(values):
    assert flat_block_digest(values) == spec_flat_digest(values, values.dtype)
    # The digest identifies the stored type, whatever its byte order.
    swapped = values.astype(values.dtype.newbyteorder(">"))
    assert flat_block_digest(swapped) == flat_block_digest(values)


@pytest.mark.unit
def test_flattening_differs_from_row_axis_digests_and_supports_f4():
    keys = np.arange(30, dtype=np.int32).reshape(10, 3)
    # Without flattening, rows are the last (component) axis.
    assert flat_block_digest(keys) != block_digest(keys, np.int32)
    assert flat_block_digest(keys) == block_digest(keys.reshape(-1), np.int32)
    values = np.ones(6, dtype=np.float32)
    assert block_digest(values, np.float32) == spec_flat_digest(values, "<f4")
    assert flat_block_digest(values) != flat_block_digest(values.astype(np.float64))
    with pytest.raises(ValueError, match="Unsupported"):
        flat_block_digest(np.ones(2, dtype=np.complex64))


@pytest.mark.unit
def test_sauce_example_digests_and_fingerprints_reproduce():
    """Datasets of Sauce's published example state hash to its recorded digests."""
    manifest = json.loads((STATE_CONTRACT / "examples" / "canonical.json").read_text())
    term = manifest["terms"][0]
    digests = term["cache"]["digests"]
    n_rhs, rows = term["runtime"]["n_rhs"], term["runtime"]["n_global_rows"]
    # Dense row r = rhs + n_rhs*(receiver - 1) for one component.
    keys = [[(r - 1) % n_rhs + 1, (r - 1) // n_rhs + 1, 1] for r in range(1, rows + 1)]
    datasets = {
        "metadata": np.array([5, 0, n_rhs, 1, 1, 0, 0], dtype=np.int32),
        "n_global_rows": np.array([rows], dtype=np.int64),
        "metric_weight": np.ones(1),
        "active_mask": np.ones((1, n_rhs), dtype=np.int32),
        "rhs_required": np.ones(n_rhs, dtype=np.int32),
        "coordinate_keys": np.array(keys, dtype=np.int32),
        "static_weight": np.ones(rows, dtype=np.float32),
        "robust_weight": np.ones(rows, dtype=np.float32),
    }
    for name, values in datasets.items():
        assert flat_block_digest(values) == digests[name], name
    assert canonical_json_sha256(digests) == term["runtime"]["cache_fingerprint"]
    basis = {
        k: v
        for k, v in manifest.items()
        if k not in {"manifest_fingerprint", "state_fingerprint"}
    }
    assert canonical_json_sha256(basis) == manifest["manifest_fingerprint"]
    vector = json.loads(
        (
            CONTRACTS / "fs-objective-vector-5" / "examples" / "canonical.json"
        ).read_text()
    )
    assert vector["state_fingerprint"] == manifest["state_fingerprint"]
    assert (
        vector["terms"][0]["digests"]["coordinate_keys"] == digests["coordinate_keys"]
    )
    vector_basis = {k: v for k, v in vector.items() if k != "manifest_fingerprint"}
    assert canonical_json_sha256(vector_basis) == vector["manifest_fingerprint"]


# ---------------------------------------------------------------------------
# canonical JSON
# ---------------------------------------------------------------------------

# Doubles as nlohmann/json 3.11.3 (Sauce's include/json_reader.hpp) dumps them;
# Python's repr gives other text for the last six.
NLOHMANN_DOUBLES = [
    (0.0, "0.0"),
    (-0.0, "-0.0"),
    (1.0, "1.0"),
    (4.0, "4.0"),
    (0.1, "0.1"),
    (1e-05, "1e-05"),
    (0.0001, "0.0001"),
    (0.0001234, "0.0001234"),
    (1e16, "1e+16"),
    (1e21, "1e+21"),
    (5e-324, "5e-324"),
    (1.7976931348623157e308, "1.7976931348623157e+308"),
    (0.30000000000000004, "0.30000000000000004"),
    (1e15, "1e+15"),
    (1.5e15, "1.5e+15"),
    (2905011221249542.5, "2.9050112212495425e+15"),
    (-6016.951217939863, "-6016.9512179398635"),
    (-276804372109801.38, "-276804372109801.37"),
    (7.036870839547745e177, "7.0368708395477454e+177"),
]


@pytest.mark.unit
def test_canonical_json_formats_doubles_like_sauce():
    for value, text in NLOHMANN_DOUBLES:
        assert nlohmann_double(value) == text, value
    assert sum(json.dumps(value) != text for value, text in NLOHMANN_DOUBLES) == 6
    assert nlohmann_double(float("nan")) == "null"
    payload = {"b": [value for value, _ in NLOHMANN_DOUBLES], "a": "é", "c": [1, True]}
    text = (
        '{"a":"é","b":['
        + ",".join(text for _, text in NLOHMANN_DOUBLES)
        + '],"c":[1,true]}'
    )
    expected = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    assert canonical_json_sha256(payload) == expected
    # Payloads without floats keep the plain sorted compact form.
    plain = {"z": [1, 2], "a": {"k": "v"}}
    compact = json.dumps(plain, sort_keys=True, separators=(",", ":"))
    assert canonical_json_sha256(plain) == (
        "sha256:" + hashlib.sha256(compact.encode()).hexdigest()
    )


# ---------------------------------------------------------------------------
# ObjectiveState
# ---------------------------------------------------------------------------


def _linearized(tmp_path, **fake_options):
    fake = FakeImagingSite(seed=7, **fake_options)
    _, problem = _problem(tmp_path, fake)
    lin = problem.linearize()
    return fake, lin


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f8", "f4"])
def test_objective_state_v5_reads_canonical_rows(tmp_path, dtype):
    fake, lin = _linearized(tmp_path, n_ranks=3, state_dtype=dtype)
    surrogate = fake.linearizations[lin.state_fingerprint]
    schema = json.loads((STATE_CONTRACT / "schema.json").read_text())
    for task, frequency in enumerate(lin.frequencies, start=1):
        path = lin.job.state_file(task)
        manifest = json.loads(path.read_text())
        # The fake keeps its surrogate under "fake"; the rest is the contract.
        Draft202012Validator(schema).validate(
            {k: v for k, v in manifest.items() if k != "fake"}
        )
        assert not list(path.parent.glob(f"{path.stem}*rank*"))
        state = ObjectiveState(path)
        # Written on three ranks, the state records no partition to pin.
        assert (state.schema, state.vector_schema, state.n_ranks) == (
            "fs-objective-linearization-5",
            "fs-objective-vector-5",
            None,
        )
        assert state.value_dtype == np.dtype(dtype)
        # Duals are unpartitioned and use the state's precision.
        assert state.vector_options() == {
            "schema": "fs-objective-vector-5",
            "dtype": np.dtype(dtype),
        }
        simulated = surrogate.J @ surrogate.m
        tolerance = 1e-6 if dtype is np.float32 else 0.0
        for layout in surrogate.space.term_layouts(frequency=frequency):
            order = np.argsort(layout.row_ids)  # canonical rows by row id
            term = state.terms[layout.id]
            np.testing.assert_array_equal(term["keys"], layout.coordinate_keys[order])
            np.testing.assert_allclose(
                term["simulated"],
                simulated[layout.indices[order]],
                rtol=tolerance,
                atol=tolerance,
            )
            np.testing.assert_allclose(
                term["residual"],
                (simulated - surrogate.d)[layout.indices[order]],
                rtol=tolerance,
                atol=tolerance,
            )
            assert term["convention"] == "simulated_minus_observed"
        assert state.term_configs[layout.id][0]["reduction"] == "sum"


@pytest.mark.unit
def test_objective_state_v5_reports_a_partition_earlier_builds_recorded(tmp_path):
    _, lin = _linearized(tmp_path)
    path = lin.job.state_file(1)
    manifest = json.loads(path.read_text())
    manifest["partition"] = {"n_ranks": 4, "compatibility": "same_mesh_partition"}
    path.write_text(json.dumps(seal_state_manifest(manifest)))
    state = ObjectiveState(path)
    assert state.n_ranks == 4
    # Canonical duals stay unpartitioned whatever rank count the state records.
    assert "n_ranks" not in state.vector_options()


@pytest.mark.unit
def test_single_precision_states_take_single_precision_duals(tmp_path):
    """Operator duals of an f4 state are unpartitioned f4 version-5 files."""
    fake, lin = _linearized(tmp_path, n_ranks=2, state_dtype=np.float32)
    surrogate = fake.linearizations[lin.state_fingerprint]
    with lin.problem.backend.recording() as jobs:
        r = lin.data_space.random(seed=4)
        covector = lin.vjp(r)
    (vjp,) = [job for job in jobs if job.action == "vjp"]
    for task in range(1, len(lin.frequencies) + 1):
        manifest_path = vjp.task_input(vjp.objective_vector, task)
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema"] == "fs-objective-vector-5"
        assert "partition" not in manifest
        with h5py.File(manifest_path.parent / manifest["file"], "r") as h5:
            assert h5["terms/0/values"].dtype == np.float32
    # The dual is rounded to single precision on its way to the solver.
    rounded = r.values.astype(np.complex64).astype(np.complex128)
    expected = np.real(surrogate.J.conj().T @ rounded)
    np.testing.assert_allclose(
        covector.values, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max()
    )


@pytest.mark.unit
def test_objective_state_v5_rejects_corrupt_or_inconsistent_states(tmp_path):
    _, lin = _linearized(tmp_path)
    path = lin.job.state_file(1)
    cache = path.with_name(f"{path.stem}_cache.h5")
    ObjectiveState(path)

    with h5py.File(cache, "r+") as h5:
        h5["terms/0/simulated"][0, 0] += 1
    with pytest.raises(ValueError, match="digest mismatch for 'simulated'"):
        ObjectiveState(path)
    reseal_state(path)
    ObjectiveState(path)

    manifest = json.loads(path.read_text())
    manifest["terms"][0]["weight"] = 2.0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest fingerprint mismatch"):
        ObjectiveState(path)

    manifest["terms"][0]["weight"] = 1.0
    digests = manifest["terms"][0]["cache"]["digests"]
    digests["simulated"] = "fs-block-sha256-1:" + "0" * 64
    path.write_text(json.dumps(seal_state_manifest(manifest)))
    with pytest.raises(ValueError, match="differ from the cache fingerprint"):
        ObjectiveState(path)

    reseal_state(path)
    manifest = json.loads(path.read_text())
    term = manifest["terms"][0]
    del term["cache"]["digests"]["coordinate_keys"]
    term["runtime"]["cache_fingerprint"] = canonical_json_sha256(
        term["cache"]["digests"]
    )
    path.write_text(json.dumps(seal_state_manifest(manifest)))
    with pytest.raises(ValueError, match="no digest of 'coordinate_keys'"):
        ObjectiveState(path)

    manifest = reseal_state(path)
    del manifest["manifest_fingerprint"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no manifest fingerprint"):
        ObjectiveState(path)


# ---------------------------------------------------------------------------
# Sauce-produced states and vectors (needs a version-5 build)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(1800)
def test_sauce_version5_digests_verify_in_the_sdk(tmp_path):
    """Every dataset Sauce writes hashes, in the SDK, to the digest it recorded."""
    from frequensolve.imaging import DepthProfile, ImagingProblem, ObservedData
    from frequensolve.orchestrator.sites.local import LocalSite
    from frequensolve.project import Project
    from frequensolve.simulation import FrequencyDomainJob
    from tests.test_imaging_integration import (
        START_VP,
        TRUTH_VP,
        _executable,
        _simulation,
    )

    site = LocalSite(solver=_executable(), n_workers=1)
    project = Project(name="v5", path=tmp_path / "project", load_if_exists=False)
    truth = _simulation(project, "truth", TRUTH_VP)
    observed = FrequencyDomainJob("observed", truth, [4.0, 6.0])
    assert site.run(observed, check=True).successful
    problem = ImagingProblem(
        _simulation(project, "initial", START_VP),
        controls=DepthProfile("vp", "layer_2", count=3),
        observed=ObservedData(observed),
        site=site,
        name="v5",
    )
    lin = problem.linearize()
    path = lin.job.state_file(1)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "fs-objective-linearization-5":
        pytest.skip(f"{os.environ.get('FS_SAUCE_EXECUTABLE')} writes another state")
    Draft202012Validator(
        json.loads((STATE_CONTRACT / "schema.json").read_text())
    ).validate(manifest)
    assert not list(lin.job._result_path.glob("*rank_*"))
    tags = set()
    for term in manifest["terms"]:
        with h5py.File(path.parent / term["cache"]["file"], "r") as h5:
            group = h5[term["cache"]["group"]]
            names = {name for name in group if isinstance(group[name], h5py.Dataset)}
            assert names == set(term["cache"]["digests"])
            for name in names:
                data = group[name][()]
                tags.add(np.dtype(data.dtype).str[1:])
                assert flat_block_digest(data) == term["cache"]["digests"][name], name
    assert {"i4", "i8", "f4", "f8"} <= tags
    state = ObjectiveState(path)
    assert state.value_dtype == np.float32
    with problem.backend.recording() as jobs:
        data = lin.jvp(lin.space.ones() * 1e-3)
        covector = lin.vjp(data)
    (jvp,) = [job for job in jobs if job.action == "jvp"]
    (vjp,) = [job for job in jobs if job.action == "vjp"]
    for task in range(1, len(lin.frequencies) + 1):
        for job in (jvp, vjp):
            manifest_path = job.task_input(job.objective_vector, task)
            vector = json.loads(manifest_path.read_text())
            assert vector["schema"] == "fs-objective-vector-5"
            assert "partition" not in vector
            with h5py.File(manifest_path.parent / vector["file"], "r") as h5:
                for index, term in enumerate(vector["terms"]):
                    for name, digest in term["digests"].items():
                        stored = h5[f"terms/{index}/{name}"][()]
                        assert flat_block_digest(stored) == digest
                        if name == "values":
                            assert stored.dtype == np.float32
    assert np.isfinite(covector.values).all() and covector.norm() > 0
