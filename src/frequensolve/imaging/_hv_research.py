# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Experimental shared-deformation Q1 HV action for receiver attributes.

Independent of the native higher-order HV implementation. Intended for research
comparisons with explicit mesh convergence checks, not native HV curvature.
All sparse factorizations and solves use FP32 with residual refinement.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.sparse.linalg import splu


@dataclass
class HVResult:
    value: float
    dual: np.ndarray
    iterations: int
    stationarity: float
    velocity: np.ndarray


class HVResearchSolver:
    """Q1 path and mixed velocity/curvature action, shared by all channels."""

    def __init__(self, intervals=128, path_intervals=4):
        if intervals < 2 or path_intervals < 2:
            raise ValueError("HV research meshes require at least two intervals")
        self.nx, self.nt = intervals + 1, path_intervals + 1
        self.size = self.nx * self.nt
        i, j = np.meshgrid(np.arange(intervals), np.arange(path_intervals))
        a = (j * self.nx + i).ravel()
        self.nodes = np.column_stack((a, a + 1, a + self.nx + 1, a + self.nx))
        q, w = leggauss(3)
        q, w = (q + 1) / 2, w / 2
        r, s = np.meshgrid(q, q)
        r, s = r.ravel(), s.ravel()
        self.shape = np.array([(1 - r) * (1 - s), r * (1 - s), r * s, (1 - r) * s]).T
        self.dx = intervals * np.array([-(1 - s), 1 - s, s, -s]).T
        self.dt = path_intervals * np.array([-(1 - r), -r, r, 1 - r]).T
        self.weight = np.outer(w, w).ravel() / intervals / path_intervals
        self.rows = np.repeat(self.nodes, 4, axis=1).ravel()
        self.columns = np.tile(self.nodes, (1, 4)).ravel()
        self.mass = self.assemble(self.shape.T @ (self.weight[:, None] * self.shape))
        self.stiff = self.assemble(self.dx.T @ (self.weight[:, None] * self.dx))
        self.free_path = np.arange(self.nx, self.size - self.nx)
        self.free_velocity = (
            np.arange(self.size).reshape(self.nt, self.nx)[:, 1:-1].ravel()
        )
        self.free_mixed = np.r_[self.free_velocity, self.free_velocity + self.size]

    def assemble(self, local):
        local = np.broadcast_to(local, (len(self.nodes), 4, 4))
        return sparse.coo_matrix(
            (local.ravel(), (self.rows, self.columns)), shape=(self.size, self.size)
        ).tocsr()

    def path_matrix(self, v):
        velocity = np.einsum("qa,ca->cq", self.shape, v[self.nodes])
        transport = self.dt[None] + velocity[:, :, None] * self.dx[None]
        return self.assemble(
            np.einsum("cqa,q,cqb->cab", transport, self.weight, transport)
        )

    def velocity_system(self, f, kappa, lam, epsilon):
        fx = np.einsum("qa,cak->cqk", self.dx, f[self.nodes])
        ft = np.einsum("qa,cak->cqk", self.dt, f[self.nodes])
        coeff = kappa + np.sum(fx**2, axis=2)
        local = np.einsum("qa,cq,q,qb->cab", self.shape, coeff, self.weight, self.shape)
        load = np.einsum(
            "qa,cq,q->ca", self.shape, -np.sum(fx * ft, axis=2), self.weight
        )
        rhs = np.bincount(self.nodes.ravel(), weights=load.ravel(), minlength=self.size)
        a = self.assemble(local) + lam * self.stiff
        cross = np.sqrt(epsilon) * self.stiff
        return sparse.bmat([[a, cross], [cross, -self.mass]], format="csr"), np.r_[
            rhs, np.zeros(self.size)
        ]

    @staticmethod
    def solve_free(matrix, rhs, free):
        result = np.zeros_like(rhs, dtype=float)
        if not len(free):
            return result
        a = matrix[free][:, free].tocsc()
        b = rhs[free]
        if not np.any(b):
            return result
        # Symmetric equilibration is important for the mixed indefinite block.
        diagonal = np.sqrt(
            np.maximum(np.asarray(abs(a).max(axis=1).toarray()).ravel(), 1e-30)
        )
        scale = 1 / diagonal
        a_scaled = (sparse.diags(scale) @ a @ sparse.diags(scale)).tocsc()
        factor = splu(a_scaled.astype(np.float32))
        scale_rhs = scale if b.ndim == 1 else scale[:, None]
        x = factor.solve((b * scale_rhs).astype(np.float32)).astype(float) * scale_rhs
        for _ in range(8):
            residual = b - a @ x
            if np.linalg.norm(residual) <= 1e-4 * np.linalg.norm(b):
                break
            x += factor.solve((residual * scale_rhs).astype(np.float32)) * scale_rhs
        else:
            raise RuntimeError("HV auxiliary solve did not reach 1e-4 residual")
        result[free] = x
        return result

    def solve(
        self,
        predicted,
        observed,
        *,
        kappa=0.05,
        lam=0.01,
        epsilon=0.001,
        max_iterations=300,
        tolerance=1e-4,
    ):
        p, d = np.asarray(predicted, float), np.asarray(observed, float)
        if p.ndim == 1:
            p, d = p[:, None], d[:, None]
        if p.shape != d.shape or p.shape[0] != self.nx or not np.isfinite([p, d]).all():
            raise ValueError("HV requires finite matching endpoint features")
        if np.array_equal(p, d):
            return HVResult(0.0, np.zeros_like(p), 0, 0.0, np.zeros((self.nt, self.nx)))
        t = np.linspace(0, 1, self.nt)[:, None, None]
        f = ((1 - t) * p[None] + t * d[None]).reshape(self.size, -1)
        v, q = np.zeros(self.size), np.zeros(self.size)
        previous = np.inf
        for iteration in range(max_iterations + 1):
            path = self.path_matrix(v)
            f += self.solve_free(path, -path @ f, self.free_path)
            block, rhs = self.velocity_system(f, kappa, lam, epsilon)
            reaction = path @ f
            energy = 0.5 * (
                np.sum(f * reaction)
                + kappa * (v @ self.mass @ v)
                + lam * (v @ self.stiff @ v)
                + q @ self.mass @ q
            )
            residual = max(
                np.linalg.norm(reaction[self.free_path]),
                np.linalg.norm((block @ np.r_[v, q] - rhs)[self.free_mixed]),
            )
            residual /= max(1.0, np.linalg.norm(f), np.linalg.norm(rhs))
            if energy > previous + 1e-5 * max(1.0, abs(previous)):
                raise RuntimeError("HV alternating action increased")
            if (
                np.isfinite(previous)
                and abs(energy - previous) <= tolerance * max(1e-12, abs(previous))
                and residual <= tolerance
            ):
                return HVResult(
                    float(energy),
                    reaction[: self.nx],
                    iteration,
                    residual,
                    v.reshape(self.nt, self.nx),
                )
            previous = energy
            mixed = self.solve_free(block, rhs, self.free_mixed)
            v, q = mixed[: self.size], mixed[self.size :]
        raise RuntimeError(
            f"HV did not converge after {max_iterations} iterations: residual {residual:g}"
        )


def endpoint_projection(count, intervals):
    """L2-project every receiver sample to Q1 endpoints, with exact transpose.

    Integrate products of piecewise linear sample and endpoint functions on
    the union of both grids. This uses all input samples, including narrow peaks.
    """
    sample = np.linspace(0, 1, count)
    nodes = np.linspace(0, 1, intervals + 1)
    cuts = np.unique(np.r_[sample, nodes])
    q, w = leggauss(2)
    mid, half = (cuts[1:] + cuts[:-1]) / 2, np.diff(cuts) / 2
    x = (mid[:, None] + half[:, None] * q).ravel()
    weight = (half[:, None] * w).ravel()

    def basis(grid):
        left = np.clip(np.searchsorted(grid, x) - 1, 0, len(grid) - 2)
        fraction = (x - grid[left]) / (grid[left + 1] - grid[left])
        return sparse.coo_matrix(
            (
                np.column_stack((1 - fraction, fraction)).ravel(),
                (
                    np.repeat(np.arange(len(x)), 2),
                    np.column_stack((left, left + 1)).ravel(),
                ),
            ),
            shape=(len(x), len(grid)),
        ).tocsr()

    a, b = basis(nodes), basis(sample)
    mass = (a.T @ sparse.diags(weight) @ a).tocsc()
    cross = (a.T @ sparse.diags(weight) @ b).tocsc()
    factor = splu(mass.astype(np.float32))
    # Small endpoint matrix, retained once and shared by shots and objectives.
    projection = factor.solve(cross.toarray().astype(np.float32)).astype(float)
    if np.linalg.norm(mass @ projection - cross.toarray()) > 1e-4 * np.linalg.norm(
        cross.data
    ):
        raise RuntimeError("HV endpoint projection failed its residual check")
    return projection
