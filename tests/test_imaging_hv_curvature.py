# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Reduced mixed HV curvature independent of the wave solver."""

import json

import h5py
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from frequensolve.imaging._hv_curvature import HVReceiverCurvature, _tangent_values
from frequensolve.imaging._receiver import ReceiverCollection
from tests.test_imaging_receiver import _state

pytestmark = pytest.mark.unit


def test_hv_saddle_elimination_matches_dense_schur_and_is_positive():
    rng = np.random.default_rng(42)
    basis = rng.normal(size=(4, 4))
    primal = basis.T @ basis + np.eye(4)
    coupling = rng.normal(size=(4, 1))
    matrix = np.block([[primal, coupling], [coupling.T, -np.ones((1, 1))]])
    trace = np.zeros((5, 3))
    trace[:2] = rng.normal(size=(2, 3))
    free = np.array([2, 3, 4])
    metric = HVReceiverCurvature(
        csr_matrix(matrix), csr_matrix(trace), free, scale=3, weight=0.4
    )
    expected = (
        trace.T
        @ (
            matrix
            - matrix[:, free]
            @ np.linalg.solve(matrix[np.ix_(free, free)], matrix[free])
        )
        @ trace
        * 0.4
        / 9
    )
    actual = np.column_stack([metric.apply(row) for row in np.eye(3)])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(actual, actual.T, rtol=1e-5, atol=1e-7)
    assert np.linalg.eigvalsh(actual).min() >= -1e-7
    assert metric.factor.L.dtype == np.float32


def test_hv_curvature_rejects_trace_on_free_dofs_and_asymmetry():
    matrix = csr_matrix(np.eye(3))
    with pytest.raises(ValueError, match="boundary"):
        HVReceiverCurvature(matrix, matrix, [1])
    with pytest.raises(ValueError, match="symmetric"):
        HVReceiverCurvature(csr_matrix([[1, 1], [0, 1]]), csr_matrix([[1], [0]]), [1])


def test_hv_curvature_handles_zero_direction_without_nonfinite_refinement():
    metric = HVReceiverCurvature(
        csr_matrix([[2, -1], [-1, 2]]), csr_matrix([[1], [0]]), [1]
    )
    np.testing.assert_array_equal(metric.apply([0]), [0])
    np.testing.assert_allclose(metric.apply([2]), [3])


def test_hv_curvature_reads_native_string_padding_and_scaling(tmp_path):
    path = tmp_path / "metric.h5"
    with h5py.File(path, "w") as handle:
        handle["schema"] = np.bytes_("fs-hv-receiver-gn-1      ")
        handle["size"] = [2]
        handle["source_id"] = [3]
        handle["receiver_id"] = [42]
        handle["amplitude_scale"] = [2.0]
        handle["term_weight"] = [0.5]
        handle["predicted"] = [1.0]
        handle["observed"] = [0.5]
        handle["free_dofs"] = [1]
        for name, matrix in (
            ("matrix", csr_matrix([[2, -1], [-1, 2]])),
            ("trace", csr_matrix([[1], [0]])),
        ):
            sparse = matrix.tocoo()
            handle[f"{name}/row"] = sparse.row
            handle[f"{name}/column"] = sparse.col
            handle[f"{name}/value"] = sparse.data.astype(np.float32)
    metric = HVReceiverCurvature.read(path)
    assert metric.source_id == 3
    np.testing.assert_array_equal(metric.receiver_ids, [42])
    np.testing.assert_allclose(metric.apply([2]), [3 * 0.5 / 4])


def test_hv_tangent_reader_checks_role_and_exact_state_identity(tmp_path):
    collection = ReceiverCollection.read(_state(tmp_path, "receiver"))
    values = np.array([1j, 2, 3 + 4j])
    path = collection.write_dual(
        tmp_path / "vector.json", {"line": (values, values * 0)}
    )
    payload = json.loads(path.read_text())
    with pytest.raises(ValueError, match="receiver state"):
        _tangent_values(path, collection)
    payload["role"] = "tangent"
    path.write_text(json.dumps(payload))
    np.testing.assert_array_equal(_tangent_values(path, collection)["line"], values)
    payload["source_state_sha256"] = "another state"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="receiver state"):
        _tangent_values(path, collection)
