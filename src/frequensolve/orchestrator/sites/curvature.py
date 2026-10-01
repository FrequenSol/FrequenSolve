"""Launch planning shared by sites that run Sauce ``--curvature`` requests.

An ``fs-curvature-request-1`` file names one method, an HDF5 input whose
datasets may be relative external links, the output path and, for some
methods, covariance factors and property-space meshes. Sauce partitions
row-distributed methods across MPI ranks; mesh transfer and sampling run on
one rank. Sites launch the configured solver executable as is; the
``FS_seismic`` dispatcher routes the request to its backend.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union

__all__ = [
    "FILE_KEYS",
    "OPEN_MPI_UNBOUND",
    "SINGLE_RANK_METHODS",
    "dispatcher_hint",
    "external_link_targets",
    "positive_count",
    "read_request",
    "single_rank",
]

SINGLE_RANK_METHODS = frozenset({"mesh_prior", "mesh_sample", "mesh_directions"})
"""Methods Sauce rejects on more than one MPI rank."""

FILE_KEYS = ("input", "output", "factors", "source_mesh", "target_mesh")
"""Request keys that hold file paths."""

OPEN_MPI_UNBOUND = {
    "PRTE_MCA_hwloc_default_binding_policy": "none",
    "OMPI_MCA_hwloc_base_binding_policy": "none",
}
"""Disable Open MPI's default one-core binding (5.x and 4.x names).

Other MPI implementations ignore these variables.
"""


def read_request(path: Union[str, Path]) -> dict[str, Any]:
    """Load a curvature request as a JSON object."""

    request = json.loads(Path(path).read_text())
    if not isinstance(request, dict):
        raise ValueError(f"Curvature request {path} must be a JSON object")
    return request


def single_rank(request: Mapping[str, Any]) -> bool:
    """Return whether Sauce requires one MPI rank for this request."""

    return request.get("method") in SINGLE_RANK_METHODS


def positive_count(value: Any, name: str) -> int:
    """Validate a rank or thread count without truncating floats or bools."""

    try:
        count = operator.index(value)
    except TypeError:
        count = 0
    if isinstance(value, bool) or count < 1:
        raise ValueError(f"{name} must be a positive integer")
    return count


def dispatcher_hint(executable: Any, log: str) -> Optional[str]:
    """Explain a failure of an ``FS_seismic`` that cannot route curvature."""

    if Path(str(executable)).name == "FS_seismic" and "Job file not provided" in log:
        return "this FS_seismic predates --curvature routing; update the solver"
    return None


def external_link_targets(path: Union[str, Path]) -> list[str]:
    """Return the distinct file names of HDF5 external links in ``path``.

    Names are returned as stored, without opening their targets; HDF5
    resolves relative names against the linking file's directory.
    """

    import h5py

    names: dict[str, None] = {}

    def collect(_name: str, link: Any) -> None:
        if isinstance(link, h5py.ExternalLink):
            names.setdefault(str(link.filename))

    with h5py.File(path, "r") as h5:
        h5.visititems_links(collect)
    return list(names)
