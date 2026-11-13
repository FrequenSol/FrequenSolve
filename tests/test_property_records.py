# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

import h5py
import numpy as np
import pytest

from frequensolve.imaging._property_records import (
    PackedPropertyReader,
    decode_record,
    encode_record,
)


@pytest.mark.parametrize("count", [0, 1, 3, 4])
def test_record_codec_preserves_bits(count):
    ints = np.array([-(2**31), 2**31 - 1, -1, 0], dtype="<i4")[:count]
    ids = np.array([-(2**63), 2**63 - 1, 2**32 + 1], dtype="<i8")
    bits = np.array(
        [0, -(2**63), 0x7FF0000000000000, 0x7FF8000000000055, 1, 0x0010000000000000],
        dtype="<i8",
    )
    encoded = encode_record(ints, ids, bits.view("<f8"))
    actual = decode_record(encoded)
    np.testing.assert_array_equal(actual[0], ints)
    np.testing.assert_array_equal(actual[1], ids)
    np.testing.assert_array_equal(actual[2].view("<i8"), bits)
    assert all(np.shares_memory(value, encoded) for value in actual if value.size)
    np.testing.assert_array_equal(encode_record(*actual), encoded)


@pytest.mark.parametrize(
    "mutation", ["truncated", "negative", "overflow", "version", "padding", "extra"]
)
def test_record_codec_rejects_malformed(mutation):
    words = encode_record([-1], [2**32], [1.0])
    if mutation == "truncated":
        words = words[:4]
    elif mutation == "negative":
        words[2] = -1
    elif mutation == "overflow":
        words[2] = 2**63 - 1
    elif mutation == "version":
        words[1] = 9
    elif mutation == "padding":
        words[5] = -1
    else:
        words = np.r_[words, 0]
    with pytest.raises(ValueError):
        decode_record(words)


def _packed_fixture(path):
    # Two roots share control 2, whose inverse incidence lives only at root 1.
    records = []
    for root in (1, 2):
        header = [1, 5, root, 5, 1, 0, 0, 0, 0, 8, 5, 4, 2, 3, 3 if root == 1 else 1]
        support = [1, 1, 2] if root == 1 else [2]
        offsets = [0, 1, 3] if root == 1 else [0, 0, 1]
        owners = [1, 1] if root == 1 else [1, 2]
        ints = header + [1, 2, 3, 4, 5] + [1, 2, 1, 2] + owners + offsets + support
        records.append(encode_record(ints, [root, root + 1], [0.0] * 8 + [1.0] * 4))
    with h5py.File(path, "w") as h5:
        g = h5.create_group("property_space")
        g["root_directory"] = np.array([[1, 15, 1, 5], [2, 25, 1, 5]], dtype="i4")
        g["root_geometry"] = np.zeros((2, 4, 2))
        g["root_linear_geometry"] = [1, 1]
        g["payload_offsets"] = [0, len(records[0]), sum(map(len, records))]
        g["payload"] = np.concatenate(records)


def test_solver_reader_needs_no_visualization_and_selects_records(tmp_path):
    path = tmp_path / "solver.h5"
    _packed_fixture(path)
    with h5py.File(path) as h5:
        reader = PackedPropertyReader(h5["property_space"], 2)
        roots = list(reader.roots([1, 1], reference=True))
        assert [root["root_id"] for root in roots] == [2]
        assert roots[0]["vertices"].shape == (1, 4, 2)
        assert "visualization_kind" not in roots[0]
        with pytest.raises(ValueError, match="visualization"):
            list(reader.roots(visualization=True))
        assert list(reader.roots([])) == []
        with pytest.raises(ValueError, match="outside"):
            list(reader.roots([2]))


def test_reader_rejects_payload_extent_mismatch(tmp_path):
    path = tmp_path / "solver.h5"
    _packed_fixture(path)
    with h5py.File(path, "a") as h5:
        h5["property_space/payload_offsets"][-1] += 1
        with pytest.raises(ValueError, match="offsets"):
            PackedPropertyReader(h5["property_space"], 2)


def test_public_golden_record():
    import json
    from pathlib import Path

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/property-record-v1.json").read_text()
    )
    bits = np.array([int(value, 16) for value in fixture["float_bits"]], dtype="<u8")
    words = encode_record(fixture["integers"], fixture["ids"], bits.view("<f8"))
    np.testing.assert_array_equal(words, fixture["words"])
    actual = decode_record(words)
    np.testing.assert_array_equal(actual[0], fixture["integers"])
    np.testing.assert_array_equal(actual[1], fixture["ids"])
    np.testing.assert_array_equal(actual[2].view("<u8"), bits)
