"""Typed physical quantities for receiver expressions and material weighting.

Physics factories expose symbolic fields and material properties. Values resolve
against the simulation at sampling time; selecting a dependency never writes it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping

from frequensolve.units import is_quantity, unit_expression, ureg
from frequensolve.util.physics import canonical_dimension, canonical_physics

__all__ = [
    "Field",
    "MaterialProperty",
    "ReceiverExpression",
    "PhysicsNamespace",
    "acoustic",
    "elastic",
    "poroelastic",
    "electromagnetic",
]


def _unit(value: Any) -> str:
    text = unit_expression(ureg.Unit(value))
    return "1" if text in {"", "dimensionless"} else text


def _axes(dimension: Any) -> tuple[str, ...]:
    return ("x", "z") if dimension == 2 else ("x", "y", "z")


def _labels(rank: int, dimension: Any) -> tuple[str, ...]:
    axes = _axes(dimension)
    if rank == 0:
        return ("",)
    if rank == 1:
        return axes
    if rank == 2:
        return (
            ("xx", "zz", "xz")
            if dimension == 2
            else ("xx", "yy", "zz", "yz", "xz", "xy")
        )
    raise ValueError(
        "Full fourth-order tensors are written through materials diagnostics"
    )


@dataclass(frozen=True)
class ReceiverExpression:
    """Immutable shaped expression, linear in wavefields, with physical units.

    ``*`` scales a field by a scalar material expression; ``@`` applies a
    constitutive tensor. ``.x`` and ``.xx`` select physical tensor components.
    Frequency-independent material expressions support arithmetic and powers.
    """

    kind: str
    units: str
    rank: int = 0
    physics: str | None = None
    dimension: int | float | None = None
    name: str = ""
    args: tuple[ReceiverExpression, ...] = ()
    value: float | None = None
    selection: tuple[str, ...] = ()
    wavefield: bool = False

    @property
    def shape(self) -> tuple[int | None, ...]:
        """Physical tensor shape; unresolved dimensions remain ``None``."""
        width = None if self.dimension is None else len(_axes(self.dimension))
        if self.physics == "em" and self.rank == 1:
            width = 3
        return (width,) * self.rank

    def to(self, units: Any) -> ReceiverExpression:
        """Select output units without changing the physical expression."""
        units = _unit(units)
        if ureg.Unit(units).dimensionality != ureg.Unit(self.units).dimensionality:
            raise ValueError("Output units are incompatible with the expression")
        # Return the base expression type; namespace leaves have custom constructors.
        return ReceiverExpression(**{**self.__dict__, "units": units})

    def _context(
        self, other: ReceiverExpression
    ) -> tuple[str | None, int | float | None]:
        if self.physics and other.physics and self.physics != other.physics:
            raise ValueError(
                "Expressions from different physics require separate receiver outputs"
            )
        if self.dimension and other.dimension and self.dimension != other.dimension:
            raise ValueError("Expression dimensions do not match")
        return self.physics or other.physics, self.dimension or other.dimension

    def _binary(self, other: Any, op: str) -> ReceiverExpression:
        other = _expression(other)
        physics, dimension = self._context(other)
        wavefield = self.wavefield or other.wavefield
        if op in {"add", "sub"}:
            if self.rank != other.rank:
                raise ValueError("Addition requires matching tensor shapes")
            if (
                ureg.Unit(self.units).dimensionality
                != ureg.Unit(other.units).dimensionality
            ):
                raise ValueError("Addition requires compatible physical units")
            if self.wavefield != other.wavefield:
                raise ValueError(
                    "Adding material offsets to wavefields is not supported"
                )
            rank, units = self.rank, self.units
        elif op in {"mul", "div"}:
            if op == "div" and (other.rank or other.wavefield):
                raise ValueError("Division requires a scalar material expression")
            if self.rank and other.rank:
                raise ValueError("Use @ for constitutive tensor contraction")
            if self.wavefield and other.wavefield:
                raise ValueError(
                    "Receiver expressions must remain linear in wavefields"
                )
            if self.rank == 4 or other.rank == 4:
                raise ValueError(
                    "Apply constitutive tensors with @ before scalar scaling"
                )
            rank = max(self.rank, other.rank)
            units = _unit(
                ureg.Unit(self.units) * ureg.Unit(other.units)
                if op == "mul"
                else ureg.Unit(self.units) / ureg.Unit(other.units)
            )
        elif op == "matmul":
            if (
                self.rank != 4
                or self.wavefield
                or other.rank != 2
                or not other.wavefield
            ):
                raise ValueError(
                    "@ requires a material constitutive tensor and a symmetric field tensor"
                )
            rank, units = 2, _unit(ureg.Unit(self.units) * ureg.Unit(other.units))
        else:
            raise ValueError(op)
        return ReceiverExpression(
            op, units, rank, physics, dimension, args=(self, other), wavefield=wavefield
        )

    def __add__(self, other: Any) -> ReceiverExpression:
        return self._binary(other, "add")

    def __radd__(self, other: Any) -> ReceiverExpression:
        return _expression(other)._binary(self, "add")

    def __sub__(self, other: Any) -> ReceiverExpression:
        return self._binary(other, "sub")

    def __rsub__(self, other: Any) -> ReceiverExpression:
        return _expression(other)._binary(self, "sub")

    def __mul__(self, other: Any) -> ReceiverExpression:
        return self._binary(other, "mul")

    def __rmul__(self, other: Any) -> ReceiverExpression:
        return _expression(other)._binary(self, "mul")

    def __truediv__(self, other: Any) -> ReceiverExpression:
        return self._binary(other, "div")

    def __rtruediv__(self, other: Any) -> ReceiverExpression:
        return _expression(other)._binary(self, "div")

    def __matmul__(self, other: Any) -> ReceiverExpression:
        return self._binary(other, "matmul")

    def __neg__(self) -> ReceiverExpression:
        return self * -1

    def __pow__(self, power: Any) -> ReceiverExpression:
        if self.wavefield or self.rank:
            raise ValueError("Powers require scalar material expressions")
        exponent = _expression(power)
        if exponent.kind != "value" or exponent.units != "1":
            raise ValueError("Powers require a dimensionless literal exponent")
        return ReceiverExpression(
            "pow",
            _unit(ureg.Unit(self.units) ** exponent.value),
            physics=self.physics,
            dimension=self.dimension,
            args=(self, exponent),
        )

    def __getitem__(self, component: Any) -> ReceiverExpression:
        if self.rank == 0:
            raise ValueError("A scalar expression has no tensor components")
        if isinstance(component, int):
            if self.rank != 1 or self.dimension is None:
                raise ValueError(
                    "Integer vector selection requires an explicit dimension"
                )
            axes = _axes(3 if self.physics == "em" else self.dimension)
            if not 0 <= component < len(axes):
                raise IndexError(component)
            component = axes[component]
        selection = tuple(component) if isinstance(component, str) else tuple(component)
        if len(selection) != self.rank or any(
            axis not in {"x", "y", "z"} for axis in selection
        ):
            raise ValueError("Select one axis per physical tensor index")
        if (
            self.dimension is not None
            and any(axis not in _axes(self.dimension) for axis in selection)
            and self.physics != "em"
        ):
            raise ValueError("Component is unavailable in this dimension")
        if self.rank == 4:
            raise ValueError(
                "Fourth-order component selection is not yet supported; use compliance @ stress or materials diagnostics"
            )
        return ReceiverExpression(
            "select",
            self.units,
            physics=self.physics,
            dimension=self.dimension,
            args=(self,),
            selection=selection,
            wavefield=self.wavefield,
        )

    def __getattr__(self, name: str) -> ReceiverExpression:
        if name and len(name) == self.rank and set(name) <= {"x", "y", "z"}:
            return self[name]
        raise AttributeError(name)

    def coefficient(self, *, basis: str = "mandel") -> dict[str, Any]:
        """Serialize a material coefficient for receivers or frozen objective weighting."""
        if self.wavefield:
            raise ValueError("Objective coefficients must depend only on materials")
        if self.rank == 4:
            return {"tensor": self.name, "basis": basis}
        if self.rank:
            raise ValueError(
                "Material coefficients must be scalar or constitutive tensors"
            )
        return {"expr": self._material_node()}

    def _material_node(self) -> dict[str, Any]:
        if self.kind == "material":
            return {"ref": self.name}
        if self.kind == "value":
            return {
                "value": self.value,
                **({"units": self.units} if self.units != "1" else {}),
            }
        if self.kind in {"add", "sub", "mul", "div", "pow"}:
            return {
                "op": self.kind,
                "args": [arg._material_node() for arg in self.args],
            }
        raise ValueError("Unsupported material expression")

    def _native(self, dimension: Any) -> dict[str, Any]:
        if not self.wavefield:
            raise ValueError(
                "Material-only outputs must be requested with materials= diagnostics"
            )
        if self.kind == "field":
            prefix = {"em": "maxwell", "poroelastic": "poroelastic"}.get(
                self.physics, self.physics
            )
            name = (
                "solid_velocity"
                if self.physics == "poroelastic" and self.name == "velocity"
                else self.name
            )
            # The existing induction plan is registered under its short name.
            qualifier = (
                ""
                if self.physics == "em" and self.name == "magnetic_induction"
                else f"{prefix}:"
            )
            name = qualifier + name + ("_all" if self.rank else "")
            return {
                "kind": "field",
                "name": name,
                "packing": self.name if self.rank == 2 else "scalar",
            }
        if self.kind == "select":
            parent = self.args[0]
            labels = _labels(parent.rank, 3 if self.physics == "em" else dimension)
            label = "".join(self.selection)
            if label not in labels:
                label = label[::-1]
            if label not in labels:
                raise ValueError(
                    f"Component {''.join(self.selection)!r} is unavailable in dimension {dimension}"
                )
            factor = (
                1 / math.sqrt(2) if parent.rank == 2 and label[0] != label[1] else 1
            )
            return {
                "kind": "select",
                "child": parent._native(dimension),
                "index": labels.index(label) + 1,
                "factor": factor,
            }
        left, right = self.args
        if self.kind in {"add", "sub"}:
            return {
                "kind": self.kind,
                "args": [left._native(dimension), right._native(dimension)],
            }
        if self.kind == "matmul":
            coefficient, child = left, right
        elif self.kind == "mul":
            coefficient, child = (right, left) if left.wavefield else (left, right)
        elif self.kind == "div":
            coefficient, child = 1 / right, left
        else:
            raise ValueError("Unsupported receiver expression")
        return {
            "kind": "scale",
            "coefficient": coefficient.coefficient(),
            "child": child._native(dimension),
        }

    def receiver_components(
        self, name: str, dimension: Any, physics: str | None = None
    ) -> list[Any]:
        """Expand explicitly selected outputs into labelled physical scalar channels."""
        from frequensolve.seismic.receivers import ReceiverComponent

        if not self.wavefield:
            raise ValueError(
                "Material-only outputs must be requested with materials= diagnostics"
            )

        if self.dimension is not None and self.dimension != dimension:
            raise ValueError("Expression dimension does not match the simulation")
        if physics is not None:
            actual = canonical_physics(physics)
            if (
                actual not in {self.physics, "coupled", "coupled_aep", None}
                and self.physics is not None
            ):
                raise ValueError("Expression physics does not match the simulation")
        dimension = canonical_dimension(dimension)
        labels = _labels(self.rank, 3 if self.physics == "em" else dimension)
        result = []
        for label in labels:
            expression = self[label] if label else self
            result.append(
                ReceiverComponent(
                    name=label or name,
                    field=f"expression:{name}",
                    units=self.units,
                    expression=expression._native(dimension),
                )
            )
        return result


class Field(ReceiverExpression):
    """Symbolic field reference, with units and physical tensor rank."""

    def __init__(
        self,
        name: str,
        *,
        units: Any,
        rank: int = 0,
        physics: str,
        dimension: Any = None,
    ):
        super().__init__(
            "field", _unit(units), rank, physics, dimension, name, wavefield=True
        )


class MaterialProperty(ReceiverExpression):
    """Symbolic material property, sampled from the selected simulation layer."""

    def __init__(
        self,
        name: str,
        *,
        units: Any,
        rank: int = 0,
        physics: str,
        dimension: Any = None,
    ):
        super().__init__("material", _unit(units), rank, physics, dimension, name)


def _expression(value: Any) -> ReceiverExpression:
    if isinstance(value, ReceiverExpression):
        return value
    units = "1"
    if is_quantity(value):
        units, value = _unit(value.units), value.magnitude
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise TypeError(
            "Expression constants must be finite real scalars or scalar quantities"
        )
    return ReceiverExpression("value", units, value=float(value))


class _Quantities:
    def __init__(self, values: Mapping[str, ReceiverExpression]):
        self._values = MappingProxyType(dict(values))

    def __getattr__(self, name: str) -> ReceiverExpression:
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(
                f"Unknown quantity {name!r}; available: {', '.join(self._values)}"
            ) from None

    def __dir__(self) -> list[str]:
        return sorted(self._values)


@dataclass(frozen=True)
class PhysicsNamespace:
    """Discoverable symbolic quantities for one solver physics family."""

    name: str
    fields: _Quantities
    materials: _Quantities


_SEISMIC = {
    "rho": "kg/m^3",
    "vp": "m/s",
    "qp": "1",
    "Sp": "s/m",
    "K": "Pa",
    "beta": "1/Pa",
}
_ELASTIC = {
    **_SEISMIC,
    "vs": "m/s",
    "qs": "1",
    "Ss": "s/m",
    "Ks": "Pa",
    "Kp": "Pa",
    "lambda": "Pa",
    "mu": "Pa",
    "epsilon": "1",
    "gamma": "1",
    "delta": "1",
    "phi": "rad",
    "theta": "rad",
    "viscosity": "Pa*s",
    "bulk_viscosity": "Pa*s",
}
_PORO = {
    "rho": "kg/m^3",
    "vp": "m/s",
    "vs": "m/s",
    "qp": "1",
    "qs": "1",
    "k_dry": "Pa",
    "mu_dry": "Pa",
    "k_solid": "Pa",
    "k_fluid": "Pa",
    "rho_solid": "kg/m^3",
    "rho_fluid": "kg/m^3",
    "porosity": "1",
    "tortuosity": "1",
    "qk": "1",
    "qmu": "1",
    "kappa": "m^2",
    "viscosity": "Pa*s",
    "viscous_length": "m",
    "biot_frequency": "Hz",
    "epsilon": "1",
    "gamma": "1",
    "delta": "1",
    "phi": "rad",
    "theta": "rad",
}
_EM = {
    "conductivity": "S/m",
    "resistivity": "ohm*m",
    "permittivity": "F/m",
    "permeability": "H/m",
    "electron_density": "1/m^3",
    "ion_mass": "1",
    "ion_charge": "1",
    "electron_collision": "1/s",
    "ion_collision": "1/s",
    "chargeability": "1",
    "ip_exponent": "1",
    "ip_time_constant": "s",
    **{f"mag_{axis}": "T" for axis in "xyz"},
    **{
        f"conductivity_{label}": "S/m" for label in ("xx", "yy", "zz", "xy", "xz", "yz")
    },
}


def _namespace(
    name: str,
    field_specs: Mapping[str, tuple[str, int]],
    material_specs: Mapping[str, str],
    dimension: Any,
) -> PhysicsNamespace:
    if dimension is not None:
        dimension = canonical_dimension(dimension)
    fields = {
        key: Field(key, units=units, rank=rank, physics=name, dimension=dimension)
        for key, (units, rank) in field_specs.items()
    }
    materials = {
        key: MaterialProperty(key, units=units, physics=name, dimension=dimension)
        for key, units in material_specs.items()
    }
    if "rho" in materials:
        materials["density"] = materials["rho"]
    if name == "elastic":
        for tensor, units in (("compliance", "1/Pa"), ("stiffness", "Pa")):
            materials[tensor] = MaterialProperty(
                tensor, units=units, rank=4, physics=name, dimension=dimension
            )
    return PhysicsNamespace(name, _Quantities(fields), _Quantities(materials))


def acoustic(*, dimension: Any = None) -> PhysicsNamespace:
    """Acoustic pressure, velocity, and scalar material properties."""
    return _namespace(
        "acoustic",
        {
            "pressure": ("Pa", 0),
            "velocity": ("m/s", 1),
        },
        _SEISMIC,
        dimension,
    )


def elastic(*, dimension: Any = None) -> PhysicsNamespace:
    """Elastic fields and Mandel compliance/stiffness tensors."""
    return _namespace(
        "elastic",
        {
            "velocity": ("m/s", 1),
            "stress": ("Pa", 2),
            "strain": ("1", 2),
            "pressure": ("Pa", 0),
            "displacement": ("m", 1),
        },
        _ELASTIC,
        dimension,
    )


def poroelastic(*, dimension: Any = None) -> PhysicsNamespace:
    """Poroelastic solid velocity, fluid flux, stress, pressure, and properties."""
    return _namespace(
        "poroelastic",
        {
            "velocity": ("m/s", 1),
            "fluid_flux": ("m/s", 1),
            "stress": ("Pa", 2),
            "pressure": ("Pa", 0),
        },
        _PORO,
        dimension,
    )


def electromagnetic(*, dimension: Any = None) -> PhysicsNamespace:
    """Maxwell fields and geophysical/plasma scalar material properties."""
    return _namespace(
        "em",
        {"electric": ("V/m", 1), "magnetic": ("T", 1), "magnetic_induction": ("T", 1)},
        _EM,
        dimension,
    )
