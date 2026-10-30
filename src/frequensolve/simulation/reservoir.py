"""Compositional reservoir models for Sauce's native HDG executor.

``ReservoirModel.to_fs()`` emits an ``fs-reservoir-model-1`` document. Quantities
accept a number in the contract's default unit (SI, with times in days and rates
in m^2/day per unit thickness), a Pint quantity, or a ``{"value", "units"}``
mapping. Their form is preserved on export except component critical constants,
which the contract requires as SI scalars. Geology is the ``kappa`` property of
a single ``reservoir:compositional`` ``fs-material-model-1`` subdomain and may be
a scalar, file-backed, or expression property.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Integral, Real
from typing import Any, Mapping, Optional, Sequence

from frequensolve.units import _scalar_in_units, is_quantity, value_and_units_to_fs

__all__ = [
    "ReservoirComponent",
    "ReservoirWater",
    "ReservoirFluid",
    "ReservoirRock",
    "CoreyStoneII",
    "ReservoirDomain",
    "ReservoirWell",
    "InjectionStream",
    "Alternation",
    "SchedulePhase",
    "InitialState",
    "ReservoirModel",
]

SCHEMA = "fs-reservoir-model-1"
_NUMERICS = {
    "pressure_scale",
    "amount_scale",
    "time_stepping",
    "adaptation",
    "output",
    "stabilization",
}


def _magnitude(value: Any, name: str) -> float:
    """Return the numeric magnitude of a number, Pint quantity, or value/units mapping."""
    if is_quantity(value):
        value = value.magnitude
    elif isinstance(value, Mapping):
        value = value.get("value")
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real number or quantity")
    return float(value)


def _positive(value: Any, name: str) -> Any:
    if _magnitude(value, name) <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _fractions(values: Sequence[float], count: int, name: str) -> tuple:
    values = tuple(float(_magnitude(v, name)) for v in values)
    if len(values) != count:
        raise ValueError(f"{name} needs one mole fraction per component ({count})")
    if any(v < 0 or v > 1 for v in values) or abs(sum(values) - 1) > 1e-10:
        raise ValueError(f"{name} must lie in [0, 1] and sum to one")
    return values


def _fields(data: Mapping[str, Any], allowed: set, name: str) -> dict:
    if not isinstance(data, Mapping):
        raise TypeError(f"{name} must be a mapping")
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
    return dict(data)


def _emit(value: Any) -> Any:
    return value_and_units_to_fs(value)


def _optional(payload: dict, **values: Any) -> dict:
    payload.update(
        {key: _emit(value) for key, value in values.items() if value is not None}
    )
    return payload


@dataclass(frozen=True)
class ReservoirComponent:
    """Peng-Robinson hydrocarbon component in SI units (K, Pa, kg/mol, m^3/mol)."""

    name: str
    critical_temperature: float
    critical_pressure: float
    acentric_factor: float
    molecular_weight: float
    critical_volume: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.isidentifier()
            or self.name != self.name.lower()
        ):
            raise ValueError("Component name must be a lowercase identifier")
        for key, unit in {
            "critical_temperature": "K",
            "critical_pressure": "Pa",
            "molecular_weight": "kg/mol",
            "critical_volume": "m^3/mol",
        }.items():
            value = _scalar_in_units(getattr(self, key), unit, key)
            if value <= 0:
                raise ValueError(f"Component {key} must be positive")
            object.__setattr__(self, key, value)
        object.__setattr__(
            self,
            "acentric_factor",
            _scalar_in_units(self.acentric_factor, "dimensionless", "acentric_factor"),
        )

    def to_fs(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> ReservoirComponent:
        return cls(**_fields(data, {f.name for f in fields(cls)}, "component"))


@dataclass(frozen=True)
class ReservoirWater:
    """Slightly compressible water with constant viscosity (defaults Pa, kg/m^3, 1/Pa, Pa*s)."""

    reference_pressure: Any
    reference_density: Any
    compressibility: Any
    viscosity: Any

    def __post_init__(self) -> None:
        _positive(self.reference_density, "reference_density")
        _positive(self.viscosity, "viscosity")
        if _magnitude(self.compressibility, "compressibility") < 0:
            raise ValueError("Water compressibility must be nonnegative")

    def to_fs(self) -> dict:
        return {
            "model": "slightly_compressible",
            **{f.name: _emit(getattr(self, f.name)) for f in fields(self)},
        }

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> ReservoirWater:
        data = _fields(data, {"model", *(f.name for f in fields(cls))}, "water")
        if data.pop("model", "slightly_compressible") != "slightly_compressible":
            raise ValueError("Water model must be slightly_compressible")
        return cls(**data)


@dataclass(frozen=True)
class ReservoirFluid:
    """Hydrocarbon components, Peng-Robinson flash settings, and the water phase."""

    components: tuple
    water: ReservoirWater
    binary_interaction: Optional[tuple] = None
    flash_tolerance: Optional[float] = None
    flash_maximum_iterations: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        names = [c.name for c in self.components]
        if not 2 <= len(names) <= 11 or len(set(names)) != len(names):
            raise ValueError("Reservoir fluid needs 2 to 11 uniquely named components")
        if self.binary_interaction is not None:
            kij = tuple(tuple(float(v) for v in row) for row in self.binary_interaction)
            n = len(names)
            if len(kij) != n or any(len(row) != n for row in kij):
                raise ValueError(
                    "binary_interaction must be square in the component count"
                )
            if any(kij[i][j] != kij[j][i] for i in range(n) for j in range(n)) or any(
                kij[i][i] for i in range(n)
            ):
                raise ValueError(
                    "binary_interaction must be symmetric with a zero diagonal"
                )
            object.__setattr__(self, "binary_interaction", kij)
        if self.flash_maximum_iterations is not None and (
            isinstance(self.flash_maximum_iterations, bool)
            or not isinstance(self.flash_maximum_iterations, Integral)
            or self.flash_maximum_iterations < 1
        ):
            raise ValueError("flash_maximum_iterations must be a positive integer")

    def to_fs(self) -> dict:
        payload: dict = {"components": [c.to_fs() for c in self.components]}
        if self.binary_interaction is not None:
            payload["binary_interaction"] = [
                list(row) for row in self.binary_interaction
            ]
        payload["equation_of_state"] = _optional(
            {"model": "peng_robinson"},
            flash_tolerance=self.flash_tolerance,
            flash_maximum_iterations=self.flash_maximum_iterations,
        )
        payload["viscosity"] = {"model": "lohrenz_bray_clark"}
        payload["water"] = self.water.to_fs()
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> ReservoirFluid:
        data = _fields(
            data,
            {
                "components",
                "binary_interaction",
                "equation_of_state",
                "viscosity",
                "water",
            },
            "fluid",
        )
        eos = _fields(
            data.get("equation_of_state", {}),
            {"model", "flash_tolerance", "flash_maximum_iterations"},
            "equation_of_state",
        )
        if eos.get("model", "peng_robinson") != "peng_robinson":
            raise ValueError("Only the peng_robinson equation of state is supported")
        if (
            _fields(data.get("viscosity", {}), {"model"}, "viscosity").get(
                "model", "lohrenz_bray_clark"
            )
            != "lohrenz_bray_clark"
        ):
            raise ValueError("Only lohrenz_bray_clark viscosity is supported")
        return cls(
            components=tuple(ReservoirComponent.from_fs(c) for c in data["components"]),
            water=ReservoirWater.from_fs(data["water"]),
            binary_interaction=data.get("binary_interaction"),
            flash_tolerance=eos.get("flash_tolerance"),
            flash_maximum_iterations=eos.get("flash_maximum_iterations"),
        )


@dataclass(frozen=True)
class ReservoirRock:
    """Compressible rock: porosity = reference_porosity exp(compressibility (p - reference_pressure))."""

    reference_pressure: Any
    reference_porosity: float
    compressibility: Any

    def __post_init__(self) -> None:
        if not 0 < _magnitude(self.reference_porosity, "reference_porosity") < 1:
            raise ValueError("reference_porosity must lie in (0, 1)")

    def to_fs(self) -> dict:
        return {
            "model": "compressible",
            **{f.name: _emit(getattr(self, f.name)) for f in fields(self)},
        }

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> ReservoirRock:
        data = _fields(data, {"model", *(f.name for f in fields(cls))}, "rock")
        if data.pop("model", "compressible") != "compressible":
            raise ValueError("Rock model must be compressible")
        return cls(**data)


@dataclass(frozen=True)
class CoreyStoneII:
    """Corey water/gas and Stone-II oil relative permeability; omitted endpoints use the contract defaults."""

    connate_water: float
    residual_oil_water: float
    residual_oil_gas: float
    critical_gas: float
    water_endpoint: Optional[float] = None
    gas_endpoint: Optional[float] = None
    oil_water_endpoint: Optional[float] = None
    oil_gas_endpoint: Optional[float] = None
    water_exponent: Optional[float] = None
    gas_exponent: Optional[float] = None
    oil_water_exponent: Optional[float] = None
    oil_gas_exponent: Optional[float] = None

    def __post_init__(self) -> None:
        swc, sorw, sorg, sgc = (
            _magnitude(getattr(self, k), k)
            for k in (
                "connate_water",
                "residual_oil_water",
                "residual_oil_gas",
                "critical_gas",
            )
        )
        if min(swc, sorw, sorg, sgc) < 0 or swc + sorw >= 1 or swc + sgc + sorg >= 1:
            raise ValueError(
                "Relative-permeability endpoints leave no mobile oil range"
            )

    def to_fs(self) -> dict:
        return _optional(
            {"model": "corey_stone2"},
            **{f.name: getattr(self, f.name) for f in fields(self)},
        )

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> CoreyStoneII:
        data = _fields(
            data, {"model", *(f.name for f in fields(cls))}, "relative_permeability"
        )
        if data.pop("model", "corey_stone2") != "corey_stone2":
            raise ValueError("Relative permeability model must be corey_stone2")
        return cls(**data)


@dataclass(frozen=True)
class ReservoirDomain:
    """Rectangle [0, length] x [0, height] (z upward) on nx-by-nz native roots."""

    length: Any
    height: Any
    root_elements: tuple
    degree: Optional[int] = None

    def __post_init__(self) -> None:
        _positive(self.length, "length")
        _positive(self.height, "height")
        roots = tuple(self.root_elements)
        if len(roots) != 2 or any(
            isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in roots
        ):
            raise ValueError("root_elements must be two positive integers")
        object.__setattr__(self, "root_elements", roots)
        if self.degree is not None and self.degree not in range(1, 7):
            raise ValueError("degree must be in 1..6")

    def to_fs(self) -> dict:
        payload = {
            "length": _emit(self.length),
            "height": _emit(self.height),
            "root_elements": list(self.root_elements),
        }
        return _optional(payload, degree=self.degree)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> ReservoirDomain:
        return cls(**_fields(data, {f.name for f in fields(cls)}, "domain"))


@dataclass(frozen=True)
class ReservoirWell:
    """Rate well with a Gaussian completion about ``location`` ([x, z], default m)."""

    name: str
    type: str
    location: Any
    completion_radius: Any
    refinement_radius: Any = None
    completion_kind: Optional[str] = None

    def __post_init__(self) -> None:
        if self.type not in ("injector", "producer"):
            raise ValueError("Well type must be injector or producer")
        if self.completion_kind not in (None, "gaussian"):
            raise ValueError("Only gaussian completions are supported")

    def to_fs(self) -> dict:
        completion = _optional(
            {}, kind=self.completion_kind, radius=self.completion_radius
        )
        payload = {
            "name": self.name,
            "type": self.type,
            "location": _emit(self.location),
            "completion": completion,
        }
        return _optional(payload, refinement_radius=self.refinement_radius)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> ReservoirWell:
        data = _fields(
            data,
            {"name", "type", "location", "completion", "refinement_radius"},
            "well",
        )
        completion = _fields(data.pop("completion"), {"kind", "radius"}, "completion")
        return cls(
            completion_radius=completion["radius"],
            completion_kind=completion.get("kind"),
            **data,
        )


@dataclass(frozen=True)
class InjectionStream:
    """Injected water or hydrocarbon stream; hydrocarbon streams carry one mole fraction per component."""

    name: str
    phase: str
    composition: Optional[tuple] = None
    pressure: Any = None

    def __post_init__(self) -> None:
        if self.phase not in ("water", "hydrocarbon"):
            raise ValueError("Stream phase must be water or hydrocarbon")
        if (self.phase == "hydrocarbon") != (self.composition is not None):
            raise ValueError(
                "Hydrocarbon streams, and only they, require a composition"
            )
        if self.composition is not None:
            object.__setattr__(self, "composition", tuple(self.composition))

    def to_fs(self) -> dict:
        payload = {"name": self.name, "phase": self.phase}
        if self.pressure is not None:
            payload["pressure"] = _emit(self.pressure)
        if self.composition is not None:
            payload["composition"] = list(self.composition)
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> InjectionStream:
        return cls(**_fields(data, {f.name for f in fields(cls)}, "injection stream"))


@dataclass(frozen=True)
class Alternation:
    """Alternate ``stream`` slugs of ``period`` days, starting ``start`` days into the phase."""

    stream: str
    start: Any
    period: Any

    def __post_init__(self) -> None:
        if _magnitude(self.start, "start") < 0:
            raise ValueError("Alternation start must be nonnegative")
        _positive(self.period, "period")

    def to_fs(self) -> dict:
        return {
            "stream": self.stream,
            "start": _emit(self.start),
            "period": _emit(self.period),
        }

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> Alternation:
        return cls(**_fields(data, {"stream", "start", "period"}, "alternation"))


@dataclass(frozen=True)
class SchedulePhase:
    """Consecutive control phase: voidage-replacement ``rate`` (m^2/day) for ``duration`` days."""

    duration: Any
    rate: Any
    stream: Optional[str] = None
    alternation: Optional[Alternation] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if _scalar_in_units(self.duration, "day", "duration") <= 0:
            raise ValueError("duration must be positive")
        rate = _magnitude(self.rate, "rate")
        if rate < 0:
            raise ValueError("Phase rate must be nonnegative")
        if rate > 0 and self.stream is None:
            raise ValueError("An injecting phase requires a stream")

    def to_fs(self) -> dict:
        payload = {"name": self.name} if self.name is not None else {}
        payload.update(duration=_emit(self.duration), rate=_emit(self.rate))
        if self.stream is not None:
            payload["stream"] = self.stream
        if self.alternation is not None:
            payload["alternation"] = self.alternation.to_fs()
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> SchedulePhase:
        data = _fields(data, {f.name for f in fields(cls)}, "schedule phase")
        if "alternation" in data:
            data["alternation"] = Alternation.from_fs(data["alternation"])
        return cls(**data)


@dataclass(frozen=True)
class InitialState:
    """Linear hydrostatic pressure about a datum with uniform saturation and composition."""

    datum_elevation: Any
    datum_pressure: Any
    hydrostatic_density: Any
    water_saturation: float
    composition: tuple

    def __post_init__(self) -> None:
        _positive(self.datum_pressure, "datum_pressure")
        if not 0 < _magnitude(self.water_saturation, "water_saturation") < 1:
            raise ValueError("Initial water saturation must lie in (0, 1)")
        object.__setattr__(self, "composition", tuple(self.composition))

    def to_fs(self) -> dict:
        payload = {f.name: _emit(getattr(self, f.name)) for f in fields(self)}
        payload["composition"] = list(self.composition)
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> InitialState:
        return cls(**_fields(data, {f.name for f in fields(cls)}, "initial state"))


@dataclass(frozen=True)
class ReservoirModel:
    """Complete ``fs-reservoir-model-1`` case for Sauce's 2D compositional HDG executor."""

    temperature: Any
    fluid: ReservoirFluid
    rock: ReservoirRock
    relative_permeability: CoreyStoneII
    permeability: Any
    domain: ReservoirDomain
    wells: tuple
    injection_streams: tuple
    schedule: tuple
    initial_state: InitialState
    gravity: Any = None
    numerics: Optional[Mapping[str, Any]] = None
    name: Optional[str] = None
    description: Optional[str] = None
    geology_name: Optional[str] = None

    def __post_init__(self) -> None:
        for key in ("wells", "injection_streams", "schedule"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        _positive(self.temperature, "temperature")
        nc = len(self.fluid.components)
        _fractions(self.initial_state.composition, nc, "initial_state.composition")
        if any(v <= 0 for v in self.initial_state.composition):
            raise ValueError("Initial composition must be strictly positive")
        names = [s.name for s in self.injection_streams]
        if not names or len(set(names)) != len(names):
            raise ValueError("Injection streams need unique names")
        for stream in self.injection_streams:
            if stream.composition is not None:
                _fractions(
                    stream.composition, nc, f"stream {stream.name!r} composition"
                )
        for phase in self.schedule:
            used = [phase.stream] + (
                [phase.alternation.stream] if phase.alternation else []
            )
            unknown = {s for s in used if s is not None} - set(names)
            if unknown:
                raise ValueError(f"Unknown injection streams: {sorted(unknown)}")
        wells = [w.name for w in self.wells]
        if len(set(wells)) != len(wells) or {w.type for w in self.wells} != {
            "injector",
            "producer",
        }:
            raise ValueError(
                "Wells need unique names and at least one injector and one producer"
            )
        if not self.schedule:
            raise ValueError("Reservoir schedule requires a phase")
        if self.numerics is not None:
            numerics = _fields(self.numerics, _NUMERICS, "numerics")
            scale = numerics.get("amount_scale")
            if scale is not None and len(scale) != nc + 1:
                raise ValueError(
                    "numerics.amount_scale needs a water entry plus one entry per component"
                )

    @property
    def end_time(self) -> float:
        """Scheduled run end in days, converting every phase's time units."""
        return sum(
            _scalar_in_units(p.duration, "day", "duration") for p in self.schedule
        )

    def to_fs(self) -> dict:
        payload: dict = {"schema": SCHEMA}
        _optional(payload, name=self.name, description=self.description)
        payload["temperature"] = _emit(self.temperature)
        if self.gravity is not None:
            payload["gravity"] = _emit(self.gravity)
        subdomain = {"name": self.geology_name} if self.geology_name is not None else {}
        subdomain.update(
            mesh_block_id=1,
            physics="reservoir:compositional",
            properties={"kappa": _emit(self.permeability)},
        )
        payload.update(
            fluid=self.fluid.to_fs(),
            rock=self.rock.to_fs(),
            relative_permeability=self.relative_permeability.to_fs(),
            Model={"schema": "fs-material-model-1", "subdomains": [subdomain]},
            domain=self.domain.to_fs(),
            wells=[w.to_fs() for w in self.wells],
            injection_streams=[s.to_fs() for s in self.injection_streams],
            schedule={"phases": [p.to_fs() for p in self.schedule]},
            initial_state=self.initial_state.to_fs(),
        )
        if self.numerics is not None:
            payload["numerics"] = dict(self.numerics)
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> ReservoirModel:
        data = _fields(
            data,
            {
                "schema",
                "name",
                "description",
                "temperature",
                "gravity",
                "fluid",
                "rock",
                "relative_permeability",
                "Model",
                "domain",
                "wells",
                "injection_streams",
                "schedule",
                "initial_state",
                "numerics",
            },
            "reservoir model",
        )
        if data.get("schema") != SCHEMA:
            raise ValueError(f"Expected schema {SCHEMA!r}")
        model = _fields(data["Model"], {"schema", "subdomains"}, "Model")
        if (
            model.get("schema", "fs-material-model-1") != "fs-material-model-1"
            or len(model["subdomains"]) != 1
        ):
            raise ValueError(
                "Model must be an fs-material-model-1 document with one subdomain"
            )
        layer = _fields(
            model["subdomains"][0],
            {"name", "mesh_block_id", "physics", "properties"},
            "subdomain",
        )
        if layer.get("physics") != "reservoir:compositional" or set(
            layer["properties"]
        ) != {"kappa"}:
            raise ValueError(
                "The subdomain must be reservoir:compositional with only a kappa property"
            )
        schedule = _fields(data["schedule"], {"phases"}, "schedule")
        return cls(
            temperature=data["temperature"],
            gravity=data.get("gravity"),
            name=data.get("name"),
            description=data.get("description"),
            fluid=ReservoirFluid.from_fs(data["fluid"]),
            rock=ReservoirRock.from_fs(data["rock"]),
            relative_permeability=CoreyStoneII.from_fs(data["relative_permeability"]),
            permeability=layer["properties"]["kappa"],
            geology_name=layer.get("name"),
            domain=ReservoirDomain.from_fs(data["domain"]),
            wells=tuple(ReservoirWell.from_fs(w) for w in data["wells"]),
            injection_streams=tuple(
                InjectionStream.from_fs(s) for s in data["injection_streams"]
            ),
            schedule=tuple(SchedulePhase.from_fs(p) for p in schedule["phases"]),
            initial_state=InitialState.from_fs(data["initial_state"]),
            numerics=data.get("numerics"),
        )
