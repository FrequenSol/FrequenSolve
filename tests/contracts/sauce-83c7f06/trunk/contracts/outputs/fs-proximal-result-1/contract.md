# FS Proximal Result Contract v1

Status: initial
Visibility: public
Contract id: `fs-proximal-result-1`

This artifact records a distributed TV or second-order TGV proximal solve in a
scalar or positive grid-diagonal metric, including the solution reference,
regularizer value, iteration count, primal and dual residuals and tolerances,
and convergence status. Nonconvergence fails by default; `return_best` permits
an artifact with `converged = false`.
