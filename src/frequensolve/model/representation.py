"""Discrete field representations and variational transfer operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import lsqr

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.model.parameterization import BSplineControl, HatControl
from frequensolve.util.mixins import ExportContext

__all__ = [
    "CartesianGridRepresentation",
    "ControlRepresentation",
    "EvaluationContext",
    "FieldRepresentation",
    "VariationalSmoothing",
]


class EvaluationContext:
    """Coordinates at which a field representation is evaluated.

    Coordinate arrays are flattened in C order. Plain axis names belong to the
    selected ``coordinate_system``; ``(system, axis)`` keys may be used when a
    context carries more than one coordinate frame.
    """

    def __init__(
        self,
        coordinates: Mapping[Any, Any],
        *,
        coordinate_system: str = "global",
        shape: Optional[Sequence[int]] = None,
        dims: Optional[Sequence[str]] = None,
    ):
        system = str(coordinate_system).strip()
        if not system:
            raise ValueError("evaluation context requires a coordinate system")
        normalized: dict[tuple[str, str], np.ndarray] = {}
        size: Optional[int] = None
        for key, values in coordinates.items():
            if isinstance(key, tuple):
                if len(key) != 2:
                    raise ValueError("coordinate keys must be axes or (system, axis)")
                key_system, axis = (str(part).strip() for part in key)
            else:
                key_system, axis = system, str(key).strip()
            if not key_system or not axis:
                raise ValueError("coordinate system and axis names cannot be empty")
            array = np.asarray(values, dtype=np.float64)
            if array.size == 0 or not np.all(np.isfinite(array)):
                raise ValueError("evaluation coordinates must be nonempty and finite")
            array = np.ravel(array, order="C")
            if size is None:
                size = int(array.size)
            elif array.size != size:
                raise ValueError(
                    "all evaluation-coordinate arrays must have equal size"
                )
            normalized[(key_system, axis)] = np.array(array, copy=True)
        if size is None:
            raise ValueError("evaluation context requires at least one coordinate axis")
        resolved_shape = (size,) if shape is None else tuple(int(n) for n in shape)
        if any(n < 1 for n in resolved_shape) or int(np.prod(resolved_shape)) != size:
            raise ValueError("evaluation-context shape does not match coordinate size")
        resolved_dims = tuple(str(dim) for dim in (dims or ()))
        if resolved_dims and len(resolved_dims) != len(resolved_shape):
            raise ValueError("evaluation-context dims do not match its shape")
        self._coordinates = normalized
        self.coordinate_system = system
        self.shape = resolved_shape
        self.dims = resolved_dims
        self.size = size

    @classmethod
    def from_grid(cls, grid: CartesianGrid) -> "EvaluationContext":
        """Return the grid-node context in the grid's xarray storage order."""

        array_dims = tuple(grid.dims[::-1])
        axes = {
            dim: np.linspace(grid.x0[i], grid.x1[i], grid.n[i])
            for i, dim in enumerate(grid.dims)
        }
        mesh = np.meshgrid(*(axes[dim] for dim in array_dims), indexing="ij")
        return cls(
            {dim: values for dim, values in zip(array_dims, mesh)},
            coordinate_system=grid.system or "global",
            shape=grid.shape,
            dims=array_dims,
        )

    def coordinate(self, axis: str, coordinate_system: str = "global") -> np.ndarray:
        """Return one flattened coordinate axis without copying it."""

        key = (str(coordinate_system).strip(), str(axis).strip())
        try:
            return self._coordinates[key]
        except KeyError as error:
            raise KeyError(
                f"evaluation context has no {key[0]!r} coordinate axis {key[1]!r}"
            ) from error


@dataclass(frozen=True)
class VariationalSmoothing:
    """Representation-independent variational smoothing configuration."""

    kind: str = "tikhonov"
    wavelength_fraction: float = 1.0
    alpha: Optional[float] = None
    alpha1: Optional[float] = None
    alpha2: Optional[float] = None
    tgv_ratio: float = 1.0
    reference_wavelength: Optional[float] = None
    derivative_order: int = 1
    epsilon: float = 1.0e-3
    iterations: int = 5
    input_role: str = "dual"
    normalize_amplitude: Optional[bool] = None
    illumination_normalization: str = "none"

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind == "l2":
            kind = "tikhonov"
        if kind not in {"none", "tikhonov", "tv", "tgv"}:
            raise ValueError(f"unsupported smoothing kind {self.kind!r}")
        wavelength_fraction = float(self.wavelength_fraction)
        alpha = None if self.alpha is None else float(self.alpha)
        alpha1 = None if self.alpha1 is None else float(self.alpha1)
        alpha2 = None if self.alpha2 is None else float(self.alpha2)
        tgv_ratio = float(self.tgv_ratio)
        reference_wavelength = (
            None
            if self.reference_wavelength is None
            else float(self.reference_wavelength)
        )
        epsilon = float(self.epsilon)
        derivative_order = int(self.derivative_order)
        iterations = int(self.iterations)
        input_role = str(self.input_role).strip().lower()
        normalize_amplitude = self.normalize_amplitude
        illumination_normalization = (
            str(self.illumination_normalization).strip().lower()
        )
        if not np.isfinite(wavelength_fraction) or wavelength_fraction < 0.0:
            raise ValueError(
                "smoothing wavelength fraction must be finite and nonnegative"
            )
        if alpha is not None and (not np.isfinite(alpha) or alpha < 0.0):
            raise ValueError("smoothing alpha must be finite and nonnegative")
        for name, value in (("alpha1", alpha1), ("alpha2", alpha2)):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"smoothing {name} must be finite and nonnegative")
        if (alpha1 is None) != (alpha2 is None):
            raise ValueError("TGV alpha1 and alpha2 must be supplied together")
        if alpha1 is not None and (alpha1 == 0.0) != (alpha2 == 0.0):
            raise ValueError("TGV alpha1 and alpha2 must both be positive or both zero")
        if not np.isfinite(tgv_ratio) or tgv_ratio <= 0.0:
            raise ValueError("TGV ratio must be finite and positive")
        if kind == "tgv" and alpha is not None:
            raise ValueError("TGV smoothing uses alpha1 and alpha2 rather than alpha")
        if kind != "tgv" and alpha1 is not None:
            raise ValueError("alpha1 and alpha2 are only valid for TGV smoothing")
        if reference_wavelength is not None and (
            not np.isfinite(reference_wavelength) or reference_wavelength <= 0.0
        ):
            raise ValueError(
                "smoothing reference wavelength must be finite and positive"
            )
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("smoothing epsilon must be finite and positive")
        if derivative_order not in {1, 2}:
            raise ValueError("smoothing derivative order must be one or two")
        if iterations < 1:
            raise ValueError("smoothing iterations must be positive")
        if input_role not in {"dual", "primal"}:
            raise ValueError("smoothing input role must be 'dual' or 'primal'")
        if normalize_amplitude is not None and not isinstance(
            normalize_amplitude, (bool, np.bool_)
        ):
            raise ValueError("smoothing amplitude normalization must be boolean")
        illumination_normalization = {
            "linear": "source",
            "nonlinear": "cross",
        }.get(illumination_normalization, illumination_normalization)
        if illumination_normalization not in {"none", "source", "cross"}:
            raise ValueError(
                "illumination normalization must be 'none', 'source' (linear), "
                "or 'cross' (nonlinear)"
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "wavelength_fraction", wavelength_fraction)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "alpha1", alpha1)
        object.__setattr__(self, "alpha2", alpha2)
        object.__setattr__(self, "tgv_ratio", tgv_ratio)
        object.__setattr__(self, "reference_wavelength", reference_wavelength)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "derivative_order", derivative_order)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "input_role", input_role)
        object.__setattr__(
            self,
            "normalize_amplitude",
            None if normalize_amplitude is None else bool(normalize_amplitude),
        )
        object.__setattr__(
            self, "illumination_normalization", illumination_normalization
        )

    def amplitude_normalization_enabled(self) -> bool:
        """Return whether nonlinear smoothing uses a dimensionless input block."""

        if self.normalize_amplitude is not None:
            return self.normalize_amplitude
        if self.kind == "tgv":
            return self.alpha1 is None
        return self.alpha is None

    def resolved_alpha(self) -> float:
        """Return the direct variational coefficient for local application."""

        if self.alpha is not None:
            return self.alpha
        if self.reference_wavelength is None:
            raise ValueError(
                "local control smoothing requires alpha or reference_wavelength; "
                "Sauce --smooth can derive wavelength from the material model"
            )
        length = self.wavelength_fraction * self.reference_wavelength / (2.0 * np.pi)
        return length ** (2 * self.derivative_order)

    def resolved_tgv_weights(self) -> tuple[float, float]:
        """Return the first- and second-order TGV weights."""

        if self.alpha1 is not None and self.alpha2 is not None:
            return self.alpha1, self.alpha2
        if self.reference_wavelength is None:
            raise ValueError(
                "local TGV smoothing requires alpha1/alpha2 or "
                "reference_wavelength; Sauce --smooth can derive wavelength "
                "from the material model"
            )
        length = self.wavelength_fraction * self.reference_wavelength / (2.0 * np.pi)
        return length, self.tgv_ratio * length * length

    def to_fs(self, *, include_illumination: bool = True) -> dict[str, Any]:
        """Serialize the common smoothing contract consumed by Sauce."""

        payload = {
            "type": self.kind,
            "lambda": self.wavelength_fraction,
            "derivative_order": self.derivative_order,
            "epsilon": self.epsilon,
            "iterations": self.iterations,
            "input_role": self.input_role,
        }
        if self.alpha is not None:
            payload["alpha"] = self.alpha
        if self.alpha1 is not None and self.alpha2 is not None:
            payload["alpha1"] = self.alpha1
            payload["alpha2"] = self.alpha2
        if self.kind == "tgv" and self.tgv_ratio != 1.0:
            payload["tgv_ratio"] = self.tgv_ratio
        if self.reference_wavelength is not None:
            payload["reference_wavelength"] = self.reference_wavelength
        if self.normalize_amplitude is not None:
            payload["normalize_amplitude"] = self.normalize_amplitude
        if include_illumination:
            payload["illumination_normalization"] = self.illumination_normalization
        return payload

    @classmethod
    def from_value(
        cls, value: Optional["VariationalSmoothing | Mapping[str, Any]"]
    ) -> Optional["VariationalSmoothing"]:
        """Normalize an optional authored smoothing value."""

        if value is None or isinstance(value, VariationalSmoothing):
            return value
        payload = dict(value)
        if "type" in payload and "kind" not in payload:
            payload["kind"] = payload.pop("type")
        if "lambda" in payload and "wavelength_fraction" not in payload:
            payload["wavelength_fraction"] = payload.pop("lambda")
        if "strength" in payload:
            payload.setdefault("alpha", payload.pop("strength"))
        if (
            "normalize_illumination" in payload
            and "illumination_normalization" not in payload
        ):
            payload["illumination_normalization"] = (
                "cross" if payload.pop("normalize_illumination") else "none"
            )
        payload.pop("normalize_coordinate", None)
        return cls(**payload)


class FieldRepresentation(ABC):
    """A finite-dimensional field with explicit evaluation and pullback maps."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of representation coefficients."""

    @abstractmethod
    def sampling_operator(
        self, context: EvaluationContext, *, derivative_order: int = 0
    ) -> csr_matrix:
        """Return the sparse coefficient-to-sample operator."""

    def evaluate(
        self,
        coefficients: Sequence[float],
        context: EvaluationContext,
        *,
        derivative_order: int = 0,
        reshape: bool = True,
    ) -> np.ndarray:
        """Evaluate coefficients at a context through the sparse local map."""

        vector = self._vector(coefficients, "field coefficients")
        values = np.asarray(
            self.sampling_operator(context, derivative_order=derivative_order) @ vector
        )
        return values.reshape(context.shape) if reshape else values

    def pullback(
        self,
        samples: Any,
        context: EvaluationContext,
        *,
        derivative_order: int = 0,
    ) -> np.ndarray:
        """Apply the exact transpose sampling map to sample-space duals."""

        values = np.asarray(samples, dtype=np.float64).reshape(-1)
        if values.size != context.size or not np.all(np.isfinite(values)):
            raise ValueError("sample duals must be finite and match the context")
        operator = self.sampling_operator(context, derivative_order=derivative_order)
        return np.asarray(operator.T @ values).reshape(-1)

    def project(
        self,
        samples: Any,
        context: EvaluationContext,
        *,
        weights: Optional[Any] = None,
        damping: float = 0.0,
        tolerance: float = 1.0e-6,
    ) -> np.ndarray:
        """Least-squares project point samples into this representation."""

        values = np.asarray(samples, dtype=np.float64).reshape(-1)
        if values.size != context.size or not np.all(np.isfinite(values)):
            raise ValueError("projection samples must be finite and match the context")
        operator = self.sampling_operator(context)
        if weights is not None:
            scale = np.asarray(weights, dtype=np.float64).reshape(-1)
            if (
                scale.size != context.size
                or np.any(scale < 0.0)
                or not np.all(np.isfinite(scale))
            ):
                raise ValueError("projection weights must be finite and nonnegative")
            scale = np.sqrt(scale)
            operator = diags(scale) @ operator
            values = scale * values
        damping = float(damping)
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("projection damping must be finite and nonnegative")
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("projection tolerance must be finite and positive")
        return lsqr(
            operator, values, damp=np.sqrt(damping), atol=tolerance, btol=tolerance
        )[0]

    def transfer_to(
        self,
        target: "FieldRepresentation",
        coefficients: Sequence[float],
        context: EvaluationContext,
        *,
        weights: Optional[Any] = None,
        damping: float = 0.0,
        tolerance: float = 1.0e-6,
    ) -> np.ndarray:
        """Evaluate here and project into another representation."""

        samples = self.evaluate(coefficients, context, reshape=False)
        return target.project(
            samples, context, weights=weights, damping=damping, tolerance=tolerance
        )

    def _vector(self, values: Sequence[float], name: str) -> np.ndarray:
        vector = np.asarray(values)
        if np.iscomplexobj(vector):
            raise ValueError(f"{name} must be real-valued")
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vector.size != self.size or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain {self.size} finite values")
        return vector


class ControlRepresentation(FieldRepresentation):
    """One-dimensional hat/B-spline control evaluation and transfer map."""

    def __init__(self, control: HatControl | BSplineControl):
        if not isinstance(control, (HatControl, BSplineControl)):
            raise TypeError("control representation requires a hat or B-spline control")
        self.control = control
        if isinstance(control, HatControl):
            coordinates = control.coordinates
            self.degree = 1
            self.knots = np.concatenate(
                ([coordinates[0]], coordinates, [coordinates[-1]])
            )
        else:
            self.degree = control.degree
            self.knots = np.array(control.knots, copy=True)
        self.axis = control.axis
        self.coordinate_system = control.coordinate_system

    @property
    def size(self) -> int:
        """Return the number of spline coefficients."""

        return self.control.size

    @property
    def span(self) -> float:
        """Return the active coordinate-interval length."""

        return float(self.knots[self.size] - self.knots[self.degree])

    def sampling_operator(
        self, context: EvaluationContext, *, derivative_order: int = 0
    ) -> csr_matrix:
        """Build the sparse local B-spline evaluation operator."""

        derivative_order = int(derivative_order)
        if derivative_order < 0 or derivative_order > 2:
            raise ValueError("B-spline derivative order must be zero, one, or two")
        if derivative_order > self.degree:
            raise ValueError(
                f"degree-{self.degree} controls do not have classical "
                f"derivative order {derivative_order}"
            )
        coordinates = context.coordinate(self.axis, self.coordinate_system)
        lower = self.knots[self.degree]
        upper = self.knots[self.size]
        active = (coordinates >= lower) & (coordinates <= upper)
        if not np.any(active):
            return csr_matrix((context.size, self.size), dtype=np.float64)
        rows = np.flatnonzero(active)
        points = coordinates[active]
        if derivative_order == 0:
            local = BSpline.design_matrix(
                points, self.knots, self.degree, extrapolate=False
            ).tocsr()
        else:
            coefficients = np.eye(self.size, dtype=np.float64)
            values = BSpline(self.knots, coefficients, self.degree, extrapolate=False)(
                points, nu=derivative_order
            )
            local = csr_matrix(np.nan_to_num(values, copy=False))
        coo = local.tocoo()
        return csr_matrix(
            (coo.data, (rows[coo.row], coo.col)),
            shape=(context.size, self.size),
        )


class CartesianGridRepresentation(FieldRepresentation):
    """Multilinear finite-dimensional representation on a Cartesian grid."""

    def __init__(self, grid: CartesianGrid):
        if not isinstance(grid, CartesianGrid):
            raise TypeError("Cartesian grid representation requires a CartesianGrid")
        self.grid = grid

    @property
    def size(self) -> int:
        """Return the total number of Cartesian nodal coefficients."""

        return int(np.prod(self.grid.n))

    @property
    def coefficient_shape(self) -> tuple[int, ...]:
        """Return the xarray-compatible coefficient shape."""

        return self.grid.shape

    @property
    def node_context(self) -> EvaluationContext:
        """Return the evaluation context at every Cartesian node."""

        return EvaluationContext.from_grid(self.grid)

    def sampling_operator(
        self, context: EvaluationContext, *, derivative_order: int = 0
    ) -> csr_matrix:
        """Build the multilinear grid sampling operator and its transpose pair."""

        if derivative_order != 0:
            raise ValueError(
                "Cartesian multilinear derivative sampling is not implemented"
            )
        array_dims = tuple(self.grid.dims[::-1])
        axes = {
            dim: np.linspace(self.grid.x0[i], self.grid.x1[i], self.grid.n[i])
            for i, dim in enumerate(self.grid.dims)
        }
        coordinates = [
            context.coordinate(dim, self.grid.system or "global") for dim in array_dims
        ]
        lower: list[np.ndarray] = []
        fraction: list[np.ndarray] = []
        active = np.ones(context.size, dtype=bool)
        for dim, values in zip(array_dims, coordinates):
            axis = axes[dim]
            active &= (values >= axis[0]) & (values <= axis[-1])
            if axis.size == 1:
                lower.append(np.zeros(context.size, dtype=int))
                fraction.append(np.zeros(context.size, dtype=np.float64))
                active &= np.isclose(values, axis[0])
                continue
            spacing = axis[1] - axis[0]
            position = (values - axis[0]) / spacing
            index = np.floor(position).astype(int)
            index = np.clip(index, 0, axis.size - 2)
            lower.append(index)
            fraction.append(np.clip(position - index, 0.0, 1.0))

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        data: list[np.ndarray] = []
        active_rows = np.flatnonzero(active)
        shape = self.grid.shape
        choices = [range(1) if n == 1 else range(2) for n in shape]
        for corner in product(*choices):
            indices = []
            weight = np.ones(active_rows.size, dtype=np.float64)
            for axis_index, side in enumerate(corner):
                indices.append(lower[axis_index][active_rows] + side)
                if shape[axis_index] > 1:
                    value = fraction[axis_index][active_rows]
                    weight *= value if side else 1.0 - value
            rows.append(active_rows)
            cols.append(np.ravel_multi_index(tuple(indices), shape, order="C"))
            data.append(weight)
        if not rows:
            return csr_matrix((context.size, self.size), dtype=np.float64)
        return csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(context.size, self.size),
        )

    def to_fs(self, ctx: Optional[ExportContext] = None) -> dict[str, Any]:
        """Serialize the existing Sauce Cartesian-grid representation."""

        return self.grid.to_fs(ctx)
