"""Python structures defining boundary conditions"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from frequensolve.util.mixins import ExtraFieldsMixin, merge_extra
from frequensolve.util.named_list import NamedList

__all__ = ["BoundaryCondition", "BoundaryConditions"]

ConditionInput = Union[str, Sequence[str]]
BoundaryLabel = Union[str, int]

_DEFAULT_PML_REFLECTION = 1e-3
_REMOVED_BOUNDARY_CONDITION_FIELDS = {"kind"}


def _reject_removed_fields(payload: Mapping[str, Any], owner: str) -> None:
    removed = sorted(_REMOVED_BOUNDARY_CONDITION_FIELDS.intersection(payload))
    if removed:
        names = ", ".join(removed)
        raise TypeError(f"{owner} no longer accepts removed field(s): {names}")


def _resolve_pml_reflection(
    pml_reflection: Optional[float],
    pml_reflectivity: Optional[float],
) -> Optional[float]:
    if pml_reflection is not None and pml_reflectivity is not None:
        raise ValueError("Specify only one of pml_reflection or pml_reflectivity")
    return pml_reflection if pml_reflection is not None else pml_reflectivity


def _normalize_condition(condition: Any) -> str:
    value = str(condition).strip().lower()
    if not value:
        raise ValueError("Boundary condition names cannot be empty")
    return value


def _normalize_conditions(value: ConditionInput) -> List[str]:
    if isinstance(value, str):
        return [_normalize_condition(value)]
    return [_normalize_condition(condition) for condition in value]


def _normalize_boundaries(
    value: Union[BoundaryLabel, Sequence[BoundaryLabel]],
) -> List[BoundaryLabel]:
    if isinstance(value, (str, int)):
        return [value]
    return list(value)


@dataclass(init=False)
class BoundaryCondition(ExtraFieldsMixin):
    """Defines one or more boundary conditions applied to a boundary set.

    Attributes:
       name (str): BC name
       conditions (List[str]): BC conditions applied to this boundary set
       boundaries (List[str]): List of boundaries where BC should be applied
       pml_wavelengths (float): PML width in wavelengths
       pml_exponent (float): PML complex stretching exponent
       pml_constant (float): PML complex stretching constant

    :warning:
       - When using PML, it is recommended to verify that the PML width and constant are
         sufficient to avoid reflections.
    """

    name: Optional[str] = None
    boundaries: List[BoundaryLabel] = field(default_factory=list)
    conditions: Optional[ConditionInput] = None

    pml_wavelengths: float = 2.0
    pml_exponent: float = 3.0
    pml_reflection: Optional[float] = None
    pml_constant: Optional[float] = 20.0
    stretch_limit: float = 0.25
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: Optional[str] = None,
        boundaries: Optional[Union[BoundaryLabel, Sequence[BoundaryLabel]]] = None,
        conditions: ConditionInput = None,
        pml_wavelengths: float = 2.0,
        pml_exponent: float = 3.0,
        pml_reflection: Optional[float] = None,
        pml_reflectivity: Optional[float] = None,
        pml_constant: Optional[float] = 20.0,
        stretch_limit: float = 0.25,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        _reject_removed_fields(kwargs, "BoundaryCondition")
        extra_payload = copy.deepcopy(dict(extra or {}))
        _reject_removed_fields(extra_payload, "BoundaryCondition")
        if "pml_reflectivity" in extra_payload:
            pml_reflectivity = _resolve_pml_reflection(
                pml_reflectivity,
                extra_payload.pop("pml_reflectivity"),
            )
        self.name = name
        self.boundaries = [] if boundaries is None else boundaries
        self.conditions = conditions
        self.pml_wavelengths = pml_wavelengths
        self.pml_exponent = pml_exponent
        self.pml_reflection = _resolve_pml_reflection(pml_reflection, pml_reflectivity)
        self.pml_constant = pml_constant
        self.stretch_limit = stretch_limit
        self._init_extra(extra_payload, **kwargs)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.conditions is None:
            raise ValueError("BoundaryCondition requires `conditions`")
        if self.pml_reflection is None:
            self.pml_reflection = _DEFAULT_PML_REFLECTION

        self.conditions = _normalize_conditions(self.conditions)
        self.boundaries = _normalize_boundaries(self.boundaries)

    @classmethod
    def from_fs(cls, data: Dict) -> "BoundaryCondition":
        data = copy.deepcopy(data)
        data.pop("_type", None)
        _reject_removed_fields(data, "BoundaryCondition")
        return cls(
            name=data.pop("name", None),
            boundaries=data.pop("boundaries"),
            conditions=data.pop("conditions"),
            pml_wavelengths=data.pop("pml_wavelengths", 2.0),
            pml_exponent=data.pop("pml_exponent", 3.0),
            pml_constant=data.pop("pml_constant", 20.0),
            pml_reflection=data.pop("pml_reflection", None),
            pml_reflectivity=data.pop("pml_reflectivity", None),
            stretch_limit=data.pop("stretch_limit", 0.25),
            extra=data,
        )

    def has_condition(self, condition: str) -> bool:
        return _normalize_condition(condition) in self.conditions

    def to_fs(self, ctx=None) -> Dict:
        bc_dict = {
            "conditions": list(self.conditions),
            "boundaries": self.boundaries,
            **(
                {
                    "pml_wavelengths": self.pml_wavelengths,
                    "pml_exponent": self.pml_exponent,
                    **(
                        {"pml_constant": self.pml_constant}
                        if self.pml_constant is not None
                        else {}
                    ),
                    "pml_reflection": self.pml_reflection,
                    "stretch_limit": self.stretch_limit,
                }
                if self.has_condition("pml")
                else {}
            ),
        }
        if self.name:
            bc_dict["name"] = self.name
        return merge_extra(bc_dict, self.extra, "BoundaryCondition")


class BoundaryConditions(NamedList):
    """Named list of boundary conditions attached to a simulation."""

    def __init__(
        self,
        conditions: Optional[
            Iterable[Union[BoundaryCondition, Mapping[str, Any]]]
        ] = None,
    ) -> None:
        super().__init__()
        self._boundaries: set[BoundaryLabel] = set()
        for condition in conditions or []:
            self.append(condition)

    @staticmethod
    def _coerce_bc(
        bc: Union[BoundaryCondition, Mapping[str, Any]],
    ) -> BoundaryCondition:
        if isinstance(bc, Mapping):
            bc = BoundaryCondition.from_fs(bc)
        if not isinstance(bc, BoundaryCondition):
            raise TypeError(f"Expected BoundaryCondition, got {type(bc).__name__}")
        return bc

    def _rebuild_boundaries(self) -> None:
        self._boundaries = {boundary for bc in self for boundary in bc.boundaries}

    def __bool__(self) -> bool:
        return len(self) > 0

    def append(self, bc: Union[BoundaryCondition, Mapping[str, Any]]) -> None:
        bc = self._coerce_bc(bc)
        self._boundaries.update(bc.boundaries)
        super().append(bc)

    def __setitem__(
        self,
        key: Union[str, int],
        value: Union[BoundaryCondition, Mapping[str, Any]],
    ) -> None:
        super().__setitem__(key, self._coerce_bc(value))
        self._rebuild_boundaries()

    def __iadd__(
        self, bc: Union[BoundaryCondition, Mapping[str, Any]]
    ) -> "BoundaryConditions":
        """Append a boundary condition and return the collection."""
        self.append(bc)
        return self

    def verify(self, mesh: Any) -> None:
        """Verify that labeled mesh boundaries have boundary conditions."""
        boundaries = getattr(mesh, "boundaries", None)
        if boundaries is None:
            return
        for boundary in boundaries:
            if boundary not in self._boundaries:
                raise ValueError(f"Boundary {boundary} is not assigned a BC")

    @classmethod
    def from_fs(cls, data: List[Dict]) -> "BoundaryConditions":
        data = copy.deepcopy(data)
        if not isinstance(data, list):
            raise TypeError("BCs must be a list of boundary condition payloads")
        return cls(BoundaryCondition.from_fs(bc_data) for bc_data in data)

    def to_fs(self, ctx=None) -> List[Dict]:
        return [bc.to_fs(ctx) for bc in self]
