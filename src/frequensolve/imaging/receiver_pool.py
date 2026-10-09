# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Opt-in sparse sharing of sampled receiver values between local arrays."""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.spatial import cKDTree


class ReceiverPool:
    """Fixed nonnegative sample-to-array map with unit row sums.

    Each column identifies an already evaluated receiver sample; a sample may
    contribute to several output rows. ``centers`` identifies one input sample
    per output, for shot/component grouping. Square maps default to one output
    per input. No PDE sampling or receiver-device behavior is changed.
    """

    def __init__(self, weights, *, centers=None, sample_keys=None):
        raw = weights.data if issparse(weights) else weights
        if np.iscomplexobj(raw):
            raise ValueError("pool weights must be real")
        matrix = csr_matrix(weights, dtype=float, copy=True)
        if (
            not all(matrix.shape)
            or not np.isfinite(matrix.data).all()
            or np.any(matrix.data < 0)
        ):
            raise ValueError("pool weights must be finite, nonnegative and nonempty")
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        mass = np.asarray(matrix.sum(axis=1)).ravel()
        if not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError("every pool output needs positive sample weight")
        matrix.data /= np.repeat(mass, np.diff(matrix.indptr))
        if centers is None:
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("rectangular pools require center sample indices")
            centers = np.arange(matrix.shape[0])
        centers = np.asarray(centers)
        if (
            centers.shape != (matrix.shape[0],)
            or centers.dtype.kind not in "iu"
            or np.any(centers < 0)
            or np.any(centers >= matrix.shape[1])
        ):
            raise ValueError("pool centers must index one input sample per output")
        self._centers = np.array(centers, dtype=np.intp, copy=True)
        self._centers.flags.writeable = False
        self._matrix, self._transpose = matrix, matrix.T.tocsr()
        for m in (self._matrix, self._transpose):
            for a in (m.data, m.indices, m.indptr):
                a.flags.writeable = False
        self.sample_keys = (
            None if sample_keys is None else tuple(tuple(k) for k in sample_keys)
        )
        if self.sample_keys is not None and len(self.sample_keys) != self.sample_count:
            raise ValueError("pool sample keys must identify every input row")

    @property
    def sample_count(self):
        return self._matrix.shape[1]

    @property
    def output_count(self):
        return self._matrix.shape[0]

    @property
    def nnz(self):
        """Number of sample-to-output contributions, not a dense matrix size."""
        return self._matrix.nnz

    @property
    def centers(self):
        return self._centers.copy()

    def validate_groups(self, groups):
        """Reject windows crossing a term, shot, component or caller partition."""
        groups = np.asarray(groups)
        if groups.shape != (self.sample_count,):
            raise ValueError("pool groups must identify every input sample")
        output_groups = np.repeat(groups[self._centers], np.diff(self._matrix.indptr))
        if np.any(groups[self._matrix.indices] != output_groups):
            raise ValueError(
                "receiver pools must not mix groups (terms, shots or components)"
            )

    def apply(self, values):
        """Accumulate shared samples into output arrays."""
        values = np.asarray(values)
        if values.ndim not in (1, 2) or values.shape[0] != self.sample_count:
            raise ValueError("pool input shape differs from sample count")
        return self._matrix @ values

    def scatter(self, values):
        """Sum all overlapping array loads back into their original samples."""
        values = np.asarray(values)
        if values.ndim not in (1, 2) or values.shape[0] != self.output_count:
            raise ValueError("pool dual shape differs from output count")
        return self._transpose @ values

    @classmethod
    def from_coordinates(
        cls,
        coordinates,
        radius,
        *,
        groups=None,
        taper="uniform",
        sigma=None,
        sample_weights=None,
        sample_keys=None,
    ):
        """Build physical-radius windows once, independently within each group.

        Coordinates are ``(sample, dimension)`` or 1D distance along a cable.
        Supply both coordinates and radius as length quantities, or both as
        numeric values in the same units. No ordering or regular spacing is
        assumed. ``sample_weights`` can supply nonnegative quadrature weights
        for irregularly sampled arrays. Output centers coincide with samples.
        """
        quantity = hasattr(coordinates, "to") and hasattr(coordinates, "magnitude")
        if quantity != (hasattr(radius, "to") and hasattr(radius, "magnitude")):
            raise ValueError(
                "coordinates and radius must both carry length units or both be numeric"
            )
        if sigma is not None and quantity != hasattr(sigma, "to"):
            raise ValueError("sigma and coordinates must have matching length units")
        if quantity:
            if sigma is not None:
                sigma = sigma.to("m").magnitude
            coordinates, radius = (
                coordinates.to("m").magnitude,
                radius.to("m").magnitude,
            )
        if np.iscomplexobj(coordinates) or np.iscomplexobj(radius):
            raise ValueError("pool coordinates and radius must be real")
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim == 1:
            coordinates = coordinates[:, None]
        if (
            coordinates.ndim != 2
            or not all(coordinates.shape)
            or not np.isfinite(coordinates).all()
        ):
            raise ValueError("pool coordinates must be a finite nonempty matrix")
        if np.ndim(radius) != 0 or not np.isfinite(radius) or radius <= 0:
            raise ValueError("pool radius must be positive and finite")
        if taper not in {"uniform", "hann", "gaussian"}:
            raise ValueError("pool taper must be uniform, hann or gaussian")
        if taper == "gaussian":
            if (
                sigma is None
                or np.iscomplexobj(sigma)
                or np.ndim(sigma) != 0
                or not np.isfinite(sigma)
                or sigma <= 0
            ):
                raise ValueError("Gaussian pool sigma must be positive and finite")
        elif sigma is not None:
            raise ValueError("sigma is only supported by Gaussian pools")
        n = len(coordinates)
        groups = np.zeros(n, int) if groups is None else np.asarray(groups)
        if groups.shape != (n,) or (
            groups.dtype.kind in "fc" and not np.isfinite(groups).all()
        ):
            raise ValueError("pool groups must identify every input sample")
        weights = np.ones(n) if sample_weights is None else np.asarray(sample_weights)
        if (
            np.iscomplexobj(weights)
            or weights.shape != (n,)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
        ):
            raise ValueError("sample weights must be finite and nonnegative")
        rows, columns, values = [], [], []
        # Cache geometry shared by shots/components; sparse membership is built
        # once and then reused in every objective/JVP/VJP evaluation.
        geometries = {}
        order = np.argsort(groups, kind="stable")
        splits = np.flatnonzero(groups[order][1:] != groups[order][:-1]) + 1
        for indices in np.split(order, splits):
            points = coordinates[indices]
            signature = points.shape, points.tobytes()
            if signature not in geometries:
                neighbors = cKDTree(points).query_ball_point(points, float(radius))
                r = np.repeat(np.arange(len(points)), [len(a) for a in neighbors])
                c = np.concatenate(neighbors).astype(np.intp)
                a = np.ones(len(c))
                if taper == "hann":
                    distance = np.linalg.norm(points[r] - points[c], axis=1) / radius
                    a = 0.5 * (1 + np.cos(np.pi * np.minimum(distance, 1)))
                elif taper == "gaussian":
                    distance = np.linalg.norm(points[r] - points[c], axis=1)
                    a = np.exp(-0.5 * (distance / sigma) ** 2)
                geometries[signature] = r, c, a
            r, c, a = geometries[signature]
            rows.append(indices[r])
            columns.append(indices[c])
            values.append(a * weights[indices[c]])
        matrix = csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(n, n),
        )
        return cls(matrix, sample_keys=sample_keys)
