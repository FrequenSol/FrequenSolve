# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Shared safeguarded Krylov implementation for imaging coordinators."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def _real_vector(value: Any, *, size: Optional[int] = None, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if size is not None and array.size != size:
        raise ValueError(f"{name} has size {array.size}; expected {size}")
    return array


def _preconditioned_cg(
    matvec: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    *,
    iterations: int,
    tolerance: float,
    precondition: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    callback: Optional[Callable[[np.ndarray], None]] = None,
    absolute_tolerance: float = 0.0,
    verify_residual: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve ``A x = b`` from ``x = 0`` by safeguarded preconditioned CG.

    By default each iteration applies ``A`` once and nothing else does: convergence is
    tested on the recurrence residual after every update (a solve converging
    on its last allowed iteration succeeds), and the final residual is
    reported from the recurrence rather than recomputed. ``r^T M r <= 0``
    raises because the preconditioner is not SPD. ``p^T A p <= 0`` (an
    indefinite operator, or an FP32 normal that is only approximately
    symmetric) stops with status ``-1`` and returns the previous iterate.
    Status is ``0`` when converged and ``iterations`` at the limit.
    ``verify_residual`` checks apparent convergence with a fresh action. If
    recurrence drift hides a larger residual, CG restarts from that residual
    within the remaining iteration budget. Acceptance always requires a true
    residual below max(absolute_tolerance, tolerance * ||b||).
    """

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("CG tolerance must be finite and nonnegative")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("CG absolute tolerance must be finite and nonnegative")
    b = _real_vector(rhs, name="right-hand side")
    x = np.zeros_like(b)
    b_norm = float(np.linalg.norm(b))
    info: Dict[str, Any] = {
        "iterations": 0,
        "status": 0,
        "converged": True,
        "residual_norm": b_norm,
        "rhs_norm": b_norm,
        "message": "zero right-hand side",
        "relative_residual_history": [0.0 if b_norm == 0 else 1.0],
    }
    threshold = max(absolute_tolerance, tolerance * b_norm)
    if b_norm == 0.0 or (math.isfinite(b_norm) and b_norm <= threshold):
        info["message"] = "initial residual tolerance reached"
        if verify_residual:
            info["true_residual_norm"] = b_norm
        return x, info
    if not math.isfinite(b_norm):
        raise ValueError("CG right-hand side must be finite")

    def preconditioned(r: np.ndarray) -> Tuple[np.ndarray, float]:
        z = (
            r
            if precondition is None
            else _real_vector(precondition(r), size=r.size, name="preconditioner")
        )
        rho = float(r @ z)
        if not math.isfinite(rho) or rho <= 0.0:
            raise ValueError(
                f"CG preconditioner is not SPD: r^T M r = {rho:.6g} for a "
                "nonzero residual"
            )
        return z, rho

    r = b.copy()
    z, rho = preconditioned(r)
    p = np.array(z, copy=True)
    residual_norm = b_norm
    status, message, done = iterations, "iteration limit reached", 0
    checked_norm = None
    replacements = 0

    def true_residual():
        residual = b - _real_vector(matvec(x), size=b.size, name="operator product")
        norm = float(np.linalg.norm(residual))
        if not math.isfinite(norm):
            raise ValueError("CG true residual is not finite")
        return residual, norm

    for k in range(1, iterations + 1):
        q = _real_vector(matvec(p), size=b.size, name="operator product")
        curvature = float(p @ q)
        if not math.isfinite(curvature):
            raise ValueError("CG operator product is not finite")
        if curvature <= 0.0:
            status = -1
            message = (
                f"nonpositive curvature p^T A p = {curvature:.6g} at iteration {k}; "
                "the operator is indefinite or inaccurate along this direction"
            )
            break
        alpha = rho / curvature
        x += alpha * p
        checked_norm = None
        r -= alpha * q
        del q
        done = k
        residual_norm = float(np.linalg.norm(r))
        info["relative_residual_history"].append(residual_norm / b_norm)
        if callback is not None:
            callback(x)
        if residual_norm <= threshold:
            if verify_residual:
                r, checked_norm = true_residual()
                residual_norm = checked_norm
                info["relative_residual_history"][-1] = residual_norm / b_norm
            if residual_norm <= threshold:
                status, message = 0, "relative residual tolerance reached"
                break
            if k < iterations:
                # A fresh residual is no longer conjugate to the old search
                # direction. Restart rather than continuing the stale recurrence.
                z, rho = preconditioned(r)
                p = np.array(z, copy=True)
                replacements += 1
                continue
        if k == iterations:
            break
        z, rho_next = preconditioned(r)
        p *= rho_next / rho
        p += z
        rho = rho_next
    if verify_residual:
        if checked_norm is None:
            _, checked_norm = true_residual()
        residual_norm = checked_norm
        if status >= 0:
            status = 0 if residual_norm <= threshold else max(iterations, 1)
            message = (
                "true residual tolerance reached"
                if status == 0
                else "true residual exceeds tolerance"
            )
        info["true_residual_norm"] = residual_norm
        info["residual_replacements"] = replacements
    info.update(
        iterations=done,
        status=status,
        converged=status == 0,
        residual_norm=residual_norm,
        message=message,
    )
    return x, info
