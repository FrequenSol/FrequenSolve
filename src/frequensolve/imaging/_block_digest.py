# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""``fs-block-sha256-1`` dataset identities shared with Sauce curvature operations.

A dataset's rows are its last axis and its columns the product of the leading
axes. Leaf ``b`` hashes rows ``[b*B, (b+1)*B)`` of every column in turn
(``B = 65536``); the dataset digest hashes a header holding the element type
and shape, followed by the leaf digests. Sauce computes the same value from the
arrays it reads or writes, in parallel over leaves and ranks, so neither side
rereads a file to identify its contents. Leaves are hashed here on a shared
thread pool; ``hashlib`` releases the GIL for each column segment.
"""

from __future__ import annotations

import hashlib
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Optional, Sequence

import numpy as np

SCHEME = "fs-block-sha256-1"
BLOCK_ROWS = 65536
# ``u1`` only identifies SDK-side byte views (boolean support masks in
# fingerprints); Sauce stages and reports f8/i8/i4 datasets.
_TAGS = {
    np.dtype(np.float64): "f8",
    np.dtype(np.int64): "i8",
    np.dtype(np.int32): "i4",
    np.dtype(np.uint8): "u1",
}
# Smaller arrays are hashed on the calling thread.
_PARALLEL_BYTES = 1 << 22
_POOL: Optional[ThreadPoolExecutor] = None
_POOL_LOCK = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(
                max_workers=min(32, os.cpu_count() or 1),
                thread_name_prefix="fs-block-digest",
            )
        return _POOL


def block_digest(values: Any, dtype: Any = np.float64) -> str:
    """Return the digest of ``values`` held as ``dtype``, Sauce's element type for the dataset."""
    element = np.dtype(dtype)
    if element not in _TAGS:
        raise ValueError(f"Unsupported block digest element type {element}")
    array = np.ascontiguousarray(values, dtype=element.newbyteorder("<"))
    shape = array.shape
    rows = shape[-1] if shape else 1
    columns = int(np.prod(shape[:-1], dtype=np.int64)) if len(shape) > 1 else 1
    table = array.reshape(columns, rows)

    def leaf(index: int) -> bytes:
        digest = hashlib.sha256()
        start, stop = index * BLOCK_ROWS, min((index + 1) * BLOCK_ROWS, rows)
        if columns and start == 0 and stop == rows:
            # A leaf spanning every row is the whole array in memory order.
            digest.update(memoryview(table).cast("B"))
        else:
            for segment in table[:, start:stop]:
                digest.update(segment)
        return digest.digest()

    count = -(-rows // BLOCK_ROWS)
    if count > 1 and array.nbytes >= _PARALLEL_BYTES:
        leaves = b"".join(_pool().map(leaf, range(count)))
    else:
        leaves = b"".join(map(leaf, range(count)))
    header = (
        SCHEME.encode()
        + b"\0"
        + _TAGS[element].encode()
        + b"\0"
        + struct.pack(f"<{len(shape) + 2}Q", len(shape), *shape, BLOCK_ROWS)
    )
    return f"{SCHEME}:{hashlib.sha256(header + leaves).hexdigest()}"


def stacked_block_digest(
    vectors: Sequence[Any], length: int, dtype: Any = np.float64
) -> str:
    """Return ``block_digest(np.stack(vectors))`` without forming the stack.

    ``vectors`` are the columns of a ``(len(vectors), length)`` dataset, e.g.
    separately held BFGS secant pairs; each is hashed in place, leaf by leaf.
    """
    element = np.dtype(dtype)
    if element not in _TAGS:
        raise ValueError(f"Unsupported block digest element type {element}")
    columns = [
        np.ascontiguousarray(vector, dtype=element.newbyteorder("<")).reshape(-1)
        for vector in vectors
    ]
    rows = int(length)
    if any(column.size != rows for column in columns):
        raise ValueError("Stacked block digest vectors must share their length")
    shape = (len(columns), rows)

    def leaf(index: int) -> bytes:
        digest = hashlib.sha256()
        start, stop = index * BLOCK_ROWS, min((index + 1) * BLOCK_ROWS, rows)
        for column in columns:
            digest.update(column[start:stop].data)
        return digest.digest()

    count = -(-rows // BLOCK_ROWS)
    if count > 1 and len(columns) * rows * element.itemsize >= _PARALLEL_BYTES:
        leaves = b"".join(_pool().map(leaf, range(count)))
    else:
        leaves = b"".join(map(leaf, range(count)))
    header = (
        SCHEME.encode()
        + b"\0"
        + _TAGS[element].encode()
        + b"\0"
        + struct.pack(f"<{len(shape) + 2}Q", len(shape), *shape, BLOCK_ROWS)
    )
    return f"{SCHEME}:{hashlib.sha256(header + leaves).hexdigest()}"


def dataset_block_digest(dataset: Any, dtype: Any = np.float64) -> str:
    """Return ``block_digest(dataset[()], dtype)`` reading one leaf tile at a time.

    ``dataset`` is an HDF5 dataset (anything with ``shape`` and slicing). Each
    leaf is one hyperslab ``[..., b*B:(b+1)*B]``, so verifying a stored
    controls-by-rank block holds a few row tiles, never the whole dataset.
    Tiles are read in order on the calling thread and hashed on the shared
    pool, with a bounded number in flight.
    """
    element = np.dtype(dtype)
    if element not in _TAGS:
        raise ValueError(f"Unsupported block digest element type {element}")
    scalar = not dataset.shape
    # ``block_digest`` sees a scalar as one element (``ascontiguousarray``).
    shape = (1,) if scalar else tuple(int(size) for size in dataset.shape)
    rows = shape[-1]
    columns = int(np.prod(shape[:-1], dtype=np.int64)) if len(shape) > 1 else 1
    order = element.newbyteorder("<")

    def tile(index: int) -> np.ndarray:
        start, stop = index * BLOCK_ROWS, min((index + 1) * BLOCK_ROWS, rows)
        values = dataset[()] if scalar else dataset[..., start:stop]
        return np.ascontiguousarray(values, dtype=order).reshape(columns, stop - start)

    def leaf(table: np.ndarray) -> bytes:
        digest = hashlib.sha256()
        for segment in table:
            digest.update(segment.data)
        return digest.digest()

    count = -(-rows // BLOCK_ROWS)
    inflight = max(2, min(8, os.cpu_count() or 1))
    pending: list = []
    leaves = []
    for index in range(count):
        if count == 1:
            leaves.append(leaf(tile(index)))
            break
        pending.append(_pool().submit(leaf, tile(index)))
        if len(pending) >= inflight:
            leaves.append(pending.pop(0).result())
    leaves.extend(future.result() for future in pending)
    header = (
        SCHEME.encode()
        + b"\0"
        + _TAGS[element].encode()
        + b"\0"
        + struct.pack(f"<{len(shape) + 2}Q", len(shape), *shape, BLOCK_ROWS)
    )
    return f"{SCHEME}:{hashlib.sha256(header + b''.join(leaves)).hexdigest()}"


def block_digests(
    arrays: Mapping[str, Any], dtypes: Optional[Mapping[str, Any]] = None
) -> dict[str, str]:
    """Map ``{name: array}`` to ``{"/name": digest}``; ``dtypes`` overrides float64."""
    dtypes = dtypes or {}
    return {
        f"/{name}": block_digest(value, dtypes.get(name, np.float64))
        for name, value in arrays.items()
    }
