"""Python structures defining boundary conditions"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

__all__ = ["BoundaryCondition", "BoundaryConditions", "BoundaryConditionManager"]

ConditionInput = Union[str, Sequence[str]]
BoundaryLabel = Union[str, int]

_CONDITION_ALIASES = {
    "neumann": "free",
}
_UNSET = object()
_DEFAULT_PML_REFLECTION = 1e-3


def _normalize_conditions(value: ConditionInput, *, field_name: str) -> List[str]:
    if isinstance(value, str):
        raw_conditions = [value]
    else:
        try:
            raw_conditions = list(value)
        except TypeError as exc:
            raise TypeError(
                f"{field_name} must be a string or sequence of strings"
            ) from exc

    conditions: List[str] = []
    for item in raw_conditions:
        condition = str(item).strip().lower()
        if not condition:
            raise ValueError(f"{field_name} cannot contain empty condition names")
        condition = _CONDITION_ALIASES.get(condition, condition)
        if condition not in conditions:
            conditions.append(condition)

    if not conditions:
        raise ValueError(f"{field_name} must contain at least one condition")
    return conditions


def _resolve_pml_reflection(
    pml_reflection: Any = _UNSET,
    pml_reflectivity: Any = _UNSET,
) -> float:
    if pml_reflection is not _UNSET and pml_reflectivity is not _UNSET:
        raise ValueError("Specify only one of pml_reflection or pml_reflectivity")
    if pml_reflectivity is not _UNSET:
        return pml_reflectivity
    if pml_reflection is not _UNSET:
        return pml_reflection
    return _DEFAULT_PML_REFLECTION


@dataclass
class BoundaryCondition:
    """Defines one or more boundary conditions applied to a boundary set.

    Attributes:
       name (str): BC name
       conditions (List[str]): BC conditions applied to this boundary set
       kind (str): Compatibility alias for a single condition
       boundaries (List[str]): List of boundaries where BC should be applied
       pml_wavelengths (float): PML width in wavelengths
       pml_exponent (float): PML complex stretching exponent
       pml_constant (float): PML complex stretching constant

    :warning:
       - When using PML, it is recommended to verify that the PML width and constant are
         sufficient to avoid reflections.
    """

    name: Optional[str] = None
    kind: Optional[ConditionInput] = None
    boundaries: List[BoundaryLabel] = field(default_factory=list)
    conditions: Optional[ConditionInput] = None

    pml_wavelengths: float = 2.0
    pml_exponent: float = 3.0
    pml_constant: float = 20.0
    pml_reflection: float = 1e-3
    stretch_limit: float = 0.25

    def __init__(
        self,
        name: Optional[str] = None,
        kind: Optional[ConditionInput] = None,
        boundaries: Optional[Sequence[BoundaryLabel]] = None,
        conditions: Optional[ConditionInput] = None,
        pml_wavelengths: float = 2.0,
        pml_exponent: float = 3.0,
        pml_constant: float = 20.0,
        pml_reflection: Any = _UNSET,
        stretch_limit: float = 0.25,
        **kwargs,
    ) -> None:
        pml_reflectivity = kwargs.pop("pml_reflectivity", _UNSET)
        pml_reflection = _resolve_pml_reflection(
            pml_reflection,
            pml_reflectivity,
        )
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected BoundaryCondition argument(s): {names}")

        self.name = name
        self.kind = kind
        self.boundaries = list(boundaries or [])
        self.conditions = conditions
        self.pml_wavelengths = pml_wavelengths
        self.pml_exponent = pml_exponent
        self.pml_constant = pml_constant
        self.pml_reflection = pml_reflection
        self.stretch_limit = stretch_limit
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.conditions is None and self.kind is None:
            raise ValueError("BoundaryCondition requires `conditions` or `kind`")

        if self.conditions is None:
            conditions = _normalize_conditions(self.kind, field_name="kind")  # type: ignore[arg-type]
        else:
            conditions = _normalize_conditions(self.conditions, field_name="conditions")
            if self.kind is not None:
                kind_conditions = _normalize_conditions(self.kind, field_name="kind")
                if kind_conditions != conditions:
                    raise ValueError(
                        "`kind` and `conditions` describe different boundary "
                        "conditions; use only `conditions` for multi-condition BCs"
                    )

        self.conditions = conditions
        self.kind = conditions[0]
        self.boundaries = list(self.boundaries)

    @classmethod
    def from_fs(cls, data: Dict) -> "BoundaryCondition":
        if "conditions" in data:
            condition_kwargs = {"conditions": data["conditions"]}
        else:
            condition_kwargs = {"kind": data["kind"]}
        pml_reflection = _resolve_pml_reflection(
            data.get("pml_reflection", _UNSET),
            data.get("pml_reflectivity", _UNSET),
        )
        return cls(
            name=data.get("name"),
            boundaries=data["boundaries"],
            pml_wavelengths=data.get("pml_wavelengths", 2.0),
            pml_exponent=data.get("pml_exponent", 3.0),
            pml_constant=data.get("pml_constant", 20.0),
            pml_reflection=pml_reflection,
            stretch_limit=data.get("stretch_limit", 1.0),
            **condition_kwargs,
        )

    def has_condition(self, condition: str) -> bool:
        return condition.strip().lower() in self.conditions

    def to_fs(self, ctx=None) -> Dict:
        bc_dict = {
            "conditions": list(self.conditions),
            "boundaries": self.boundaries,
            **(
                {
                    "pml_wavelengths": self.pml_wavelengths,
                    "pml_exponent": self.pml_exponent,
                    "pml_constant": self.pml_constant,
                    "pml_reflection": self.pml_reflection,
                    "stretch_limit": self.stretch_limit,
                }
                if self.has_condition("pml")
                else {}
            ),
        }
        if self.name:
            bc_dict["name"] = self.name
        return bc_dict


@dataclass
class BoundaryConditions:
    """Collection of boundary conditions attached to a simulation.

    Attributes:
       boundary_conditions (List[BoundaryCondition]): List of boundary conditions
    """

    boundary_conditions: List[BoundaryCondition] = field(default_factory=list)
    label_type: Optional[Literal["geometric", "labeled"]] = None
    _boundaries: set[BoundaryLabel] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._boundaries.update(
            boundary for bc in self.boundary_conditions for boundary in bc.boundaries
        )

    def __bool__(self) -> bool:
        return bool(self.boundary_conditions)

    def append(self, bc: BoundaryCondition) -> None:
        self.add_BC(bc)

    def add_BC(self, bc: BoundaryCondition) -> None:
        """Adds a boundary condition to the collection.

        Args:
           bc (BoundaryCondition): The boundary condition to add

        Notes:
           - A BC can be applied to multiple boundaries. A boundary can now carry
             multiple conditions by using ``conditions=[...]`` on the BC.
        """
        self._boundaries.update(bc.boundaries)
        self.boundary_conditions.append(bc)

    def __iadd__(self, bc: BoundaryCondition) -> "BoundaryConditions":
        """Overrides += operator to invoke add_BC"""
        self.add_BC(bc)
        return self

    def verify(self, mesh: Any) -> None:
        """Verifies that the boundary conditions are valid, and that each boundary has a BC assigned to it"""
        if self.label_type == "geometric":
            if mesh.dimension == 2:
                labels = ["x_min", "x_max", "z_min", "z_max"]
            else:
                labels = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
            for boundary in labels:
                if boundary not in self._boundaries:
                    raise ValueError(f"Boundary {boundary} is not assigned a BC")
        else:
            if mesh.boundaries is None:
                raise ValueError(
                    "Mesh boundaries are not labeled, and label_type is 'labeled'"
                )

            # TODO: check that all boundary elements are assigned a boundary label

            for boundary in mesh.boundaries:
                if boundary not in self._boundaries:
                    raise ValueError(f"Boundary {boundary} is not assigned a BC")

    @classmethod
    def from_fs(cls, data: Dict | List[Dict]) -> "BoundaryConditions":
        if isinstance(data, list):
            return cls(
                boundary_conditions=[
                    BoundaryCondition.from_fs(bc_data) for bc_data in data
                ]
            )
        return cls(
            label_type=data.get("label_type"),
            boundary_conditions=[
                BoundaryCondition.from_fs(bc_data)
                for bc_data in data.get("boundary_conditions", [])
            ],
        )

    def to_fs(self, ctx=None) -> Dict:
        return {
            "boundary_conditions": [bc.to_fs(ctx) for bc in self.boundary_conditions],
        }


class BoundaryConditionManager(BoundaryConditions):
    """Deprecated compatibility alias for BoundaryConditions.

    New code should add BoundaryCondition objects directly to a simulation.
    """

    def __init__(
        self,
        label_type: Optional[Literal["geometric", "labeled"]] = "geometric",
        boundary_conditions: Optional[List[BoundaryCondition]] = None,
        _boundaries: Optional[set[BoundaryLabel]] = None,
    ) -> None:
        warnings.warn(
            "BoundaryConditionManager is deprecated; add BoundaryCondition objects "
            "directly to a simulation or use BoundaryConditions.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            boundary_conditions=list(boundary_conditions or []),
            label_type=label_type,
        )
        if _boundaries is not None:
            self._boundaries.update(_boundaries)
