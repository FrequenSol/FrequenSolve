# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Internal lossless property codec and bounded, selected-root artifact reader."""

from __future__ import annotations

from typing import Iterator

import h5py
import numpy as np

RECORD_MAGIC = 1179865682
BATCH_WORDS = 1024 * 1024


def encode_record(integers, ids, reals) -> np.ndarray:
    """Encode typed sections without numeric conversion of floating-point values."""
    integers = np.asarray(integers, dtype="<i4").reshape(-1)
    ids = np.asarray(ids, dtype="<i8").reshape(-1)
    reals = np.asarray(reals, dtype="<f8").reshape(-1)
    return np.concatenate(
        (
            np.array(
                [RECORD_MAGIC, 1, len(integers), len(ids), len(reals)], dtype="<i8"
            ),
            np.pad(integers, (0, len(integers) % 2)).view("<i8"),
            ids,
            reals.view("<i8"),
        )
    )


def decode_record(words) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return NumPy views after checking the version, lengths, and odd padding."""
    words = np.asarray(words)
    if words.ndim != 1 or words.dtype.kind != "i" or words.dtype.itemsize != 8:
        raise ValueError("Invalid packed property record dtype")
    words = np.asarray(words, dtype="<i8")
    if len(words) < 5:
        raise ValueError("Truncated packed property record")
    if tuple(words[:2]) != (RECORD_MAGIC, 1):
        raise ValueError(
            "Unsupported packed property record codec; regenerate artifact"
        )
    ni, ng, nr = map(int, words[2:5])
    niw = (ni + 1) // 2
    if min(ni, ng, nr) < 0 or 5 + niw + ng + nr != len(words):
        raise ValueError("Invalid packed property record lengths")
    integers = words[5 : 5 + niw].view("<i4")
    if ni % 2 and integers[-1] != 0:
        raise ValueError("Invalid packed property integer padding")
    return (
        integers[:ni],
        words[5 + niw : 5 + niw + ng],
        words[5 + niw + ng :].view("<f8"),
    )


class PackedPropertyReader:
    """Validate the common directory; solver iteration never reads visualization."""

    def __init__(self, group: h5py.Group, dimension: int):
        self.group, self.dimension = group, dimension
        self.bounds = np.asarray(group["payload_offsets"])
        self.directory = np.asarray(group["root_directory"])
        payload = group["payload"]
        if (
            dimension not in (2, 3)
            or self.directory.ndim != 2
            or self.directory.shape[1] != 4
            or not len(self.directory)
            or self.directory.dtype.kind != "i"
            or self.directory.dtype.itemsize != 4
            or len(np.unique(self.directory[:, 1])) != len(self.directory)
            or np.any(self.directory <= 0)
            or np.any(np.diff(self.directory[:, 0].astype(np.int64)) <= 0)
        ):
            raise ValueError("Invalid packed property directory")
        if (
            self.bounds.dtype.kind != "i"
            or self.bounds.dtype.itemsize != 8
            or self.bounds.shape != (len(self.directory) + 1,)
            or self.bounds[0] != 0
            or np.any(self.bounds[1:] <= self.bounds[:-1])
            or payload.ndim != 1
            or self.bounds[-1] != len(payload)
            or payload.dtype.kind != "i"
            or payload.dtype.itemsize != 8
            or payload.chunks is not None
            or payload.compression is not None
        ):
            raise ValueError("Invalid packed property offsets or payload")
        if (
            group["root_geometry"].shape
            != (len(self.directory), 2**dimension, dimension)
            or group["root_geometry"].dtype.kind != "f"
            or group["root_geometry"].dtype.itemsize != 8
            or group["root_linear_geometry"].shape != (len(self.directory),)
        ):
            raise ValueError("Invalid packed property root geometry")

    def roots(
        self, slots=None, *, reference=False, visualization=False
    ) -> Iterator[dict]:
        """Read unique selected slots in contiguous batches, retaining array views."""
        if slots is None:
            slots = np.arange(len(self.directory))
        slots = np.asarray(slots, dtype=np.int64).reshape(-1)
        if np.any(slots < 0) or np.any(slots >= len(self.directory)):
            raise ValueError("Property root slot is outside the directory")
        slots = np.unique(slots)
        visual = None
        if visualization:
            names = (
                "visualization_offsets",
                "visualization_points_m",
                "visualization_kind",
            )
            if any(name not in self.group for name in names):
                raise ValueError("This artifact has no leaf visualization geometry")
            visual = np.asarray(self.group[names[0]])
            if (
                visual.shape != self.bounds.shape
                or visual[0] != 0
                or np.any(visual[1:] < visual[:-1])
                or visual[-1] != len(self.group[names[1]])
                or visual[-1] != len(self.group[names[2]])
            ):
                raise ValueError("Invalid packed property visualization offsets")
        position = 0
        while position < len(slots):
            end = position + 1
            first = int(slots[position])
            while end < len(slots):
                if (
                    slots[end] != slots[end - 1] + 1
                    or self.bounds[slots[end] + 1] - self.bounds[first] > BATCH_WORDS
                ):
                    break
                end += 1
            start_word = int(self.bounds[first])
            words = self.group["payload"][
                start_word : int(self.bounds[slots[end - 1] + 1])
            ]
            for slot in slots[position:end]:
                lo, hi = map(int, self.bounds[slot : slot + 2])
                integers, ids, reals = decode_record(
                    words[lo - start_word : hi - start_word]
                )
                if len(integers) < 15:
                    raise ValueError("Truncated packed property root")
                nt, nv, raw, no, nw, ng, ns, incidence = map(int, integers[7:15])
                if (
                    min(nt, nv, raw, no, nw, ng, ns, incidence) < 0
                    or no < 1
                    or len(integers) != 15 + nt + nv + no + nw + ng + ns + incidence
                    or len(ids) != ng
                    or len(reals) != raw + nw
                    or ns != ng + 1
                    or (nv and raw)
                ):
                    raise ValueError("Invalid packed property root lengths")
                k = 15 + nt + nv
                offsets = integers[k : k + no]
                index = integers[k + no : k + no + nw]
                k += no + nw
                owners = integers[k : k + ng]
                support_offset = integers[k + ng : k + ng + ns]
                support_roots = integers[k + ng + ns :]
                if (
                    offsets[0] != 1
                    or offsets[-1] != nw + 1
                    or np.any(offsets[1:] < offsets[:-1])
                    or np.any(index < 1)
                    or np.any(index > ng)
                    or np.any(ids <= 0)
                    or np.any(ids[1:] <= ids[:-1])
                    or np.any(owners <= 0)
                    or support_offset[0] != 0
                    or support_offset[-1] != incidence
                    or np.any(support_offset[1:] < support_offset[:-1])
                    or np.any(support_roots <= 0)
                ):
                    raise ValueError("Invalid packed property constraints or support")
                meta = integers[:4]
                info = self.directory[slot]
                if (
                    meta[0] != info[2]
                    or meta[3] != info[3]
                    or 10 * int(meta[2]) + int(meta[1]) != info[1]
                ):
                    raise ValueError(
                        "Packed property root disagrees with initial directory"
                    )
                result = dict(
                    meta=meta,
                    offset=offsets,
                    index=index,
                    weight=reals[raw:],
                    global_ids=ids,
                    owners=owners,
                    support_offset=support_offset,
                    support_roots=support_roots,
                    root_id=int(info[0]),
                )
                if reference:
                    result["physical_vertices"] = self.group["root_geometry"][slot]
                    if nv:
                        result["reference_vertex_words"] = integers[
                            15 + nt : 15 + nt + nv
                        ]
                    else:
                        if not raw or raw % (self.dimension * 2**self.dimension):
                            raise ValueError("Invalid floating property vertex shape")
                        result["vertices"] = reals[:raw].reshape(
                            -1, 2**self.dimension, self.dimension
                        )
                if visual is not None:
                    v0, v1 = map(int, visual[slot : slot + 2])
                    result["visualization_points_m"] = self.group[
                        "visualization_points_m"
                    ][v0:v1]
                    result["visualization_kind"] = self.group["visualization_kind"][
                        v0:v1
                    ]
                yield result
            position = end
