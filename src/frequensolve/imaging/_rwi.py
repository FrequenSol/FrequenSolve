# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Shared reflectivity elimination over an already frequency-reduced normal."""

from __future__ import annotations

import numpy as np

from frequensolve.inversion._krylov import _preconditioned_cg, _real_vector


class _ReflectivitySystem:
    """Slice one pinned joint normal; add the shared coefficient penalty once."""

    def __init__(self, normal, size, outer, inner, *, damping, solver):
        self.normal = normal
        self.size = size
        self.outer = np.asarray(outer, dtype=int)
        self.inner = np.asarray(inner, dtype=int)
        indices = np.concatenate([self.outer, self.inner])
        if (
            indices.ndim != 1
            or np.any(indices < 0)
            or np.any(indices >= size)
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError(
                "RWI block indices must be disjoint and in the joint space"
            )
        if not np.isfinite(damping) or damping <= 0:
            raise ValueError("RWI damping must be finite and positive")
        self.damping = damping
        self.solver = dict(solver)
        self.actions = 0

    def product(self, indices, values):
        """Embed a block direction and return all blocks of the shared product."""
        direction = np.zeros(self.size)
        direction[indices] = _real_vector(values, size=len(indices), name="direction")
        result = _real_vector(
            self.normal(direction), size=self.size, name="joint normal"
        )
        if not np.all(np.isfinite(result)):
            raise ValueError("RWI joint normal must be finite")
        self.actions += 1
        return result

    def inner_normal(self, values):
        """Apply the summed reflectivity normal and one penalty action."""
        return self.product(self.inner, values)[self.inner] + self.damping * values

    def solve(self, rhs, *, reference_norm=None):
        """Require the true shared response residual to meet the requested tolerance."""
        initial = self.actions
        threshold = self.solver["absolute_tolerance"]
        if reference_norm is not None:
            threshold = max(
                threshold, self.solver["relative_tolerance"] * reference_norm
            )
        solution, report = _preconditioned_cg(
            self.inner_normal,
            rhs,
            iterations=self.solver["max_iterations"],
            tolerance=(
                self.solver["relative_tolerance"] if reference_norm is None else 0.0
            ),
            absolute_tolerance=threshold,
            verify_residual=True,
        )
        report["normal_actions"] = self.actions - initial
        report["reference_norm"] = (
            report["rhs_norm"] if reference_norm is None else reference_norm
        )
        report["tolerance"] = (
            max(threshold, self.solver["relative_tolerance"] * report["rhs_norm"])
            if reference_norm is None
            else threshold
        )
        if not report["converged"]:
            raise RuntimeError(
                "RWI reflectivity response did not converge: "
                f"{report['message']} (residual {report['residual_norm']:.6g})"
            )
        return solution, report

    def fit_step(self, gradient, reflectivity):
        """Return the affine inner update from an arbitrary warm start."""
        gradient = _real_vector(
            gradient, size=len(self.inner), name="reflectivity gradient"
        )
        reflectivity = _real_vector(
            reflectivity, size=len(self.inner), name="reflectivity"
        )
        initial = self.actions
        residual = gradient + self.damping * reflectivity
        # For affine F(r)=B r-d, B* d = (B*B + damping I) r - grad Phi(r).
        # This reference is independent of the warm start; a tiny correction
        # must not force arbitrarily small absolute PDE/receiver errors.
        reference = -residual
        if np.any(reflectivity):
            reference = self.inner_normal(reflectivity) - residual
        solution, report = self.solve(
            -residual, reference_norm=float(np.linalg.norm(reference))
        )
        report["normal_actions"] = self.actions - initial
        return solution, report

    def reduced_normal(self, direction):
        """Apply A - C.T (B + damping I)^-1 C using the summed joint blocks."""
        initial = self.actions
        direct = self.product(self.outer, direction)
        response, report = self.solve(direct[self.inner])
        correction = self.product(self.inner, response)
        report["normal_actions"] = self.actions - initial
        return direct[self.outer] - correction[self.outer], report
