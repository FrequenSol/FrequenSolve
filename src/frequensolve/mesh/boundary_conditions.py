"""Boundary-condition models for FrequenSolve mesh boundaries."""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from frequensolve.util.mixins import ExtraFieldsMixin, merge_extra
from frequensolve.util.named_list import NamedList

__all__ = ["BoundaryCondition", "BoundaryConditions"]

ConditionInput = Union[str, Sequence[str]]
BoundaryLabel = Union[str, int]

_DEFAULT_PML_REFLECTIVITY = 1e-2
_REMOVED_BOUNDARY_CONDITION_FIELDS = {"kind"}


def _reject_removed_fields(payload: Mapping[str, Any], owner: str) -> None:
    removed = sorted(_REMOVED_BOUNDARY_CONDITION_FIELDS.intersection(payload))
    if removed:
        names = ", ".join(removed)
        raise TypeError(f"{owner} no longer accepts removed field(s): {names}")


def _resolve_pml_reflectivity(
    pml_reflectivity: Optional[float],
    pml_reflection: Optional[float],
) -> Optional[float]:
    if pml_reflectivity is not None and pml_reflection is not None:
        raise ValueError("Specify only one of pml_reflectivity or pml_reflection")
    return pml_reflectivity if pml_reflectivity is not None else pml_reflection


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
    """One named boundary-condition assignment.

    A boundary condition can apply one or more condition names, such as
    ``"pml"`` or solver-specific boundary operators, to one or more mesh
    boundary labels. PML-related parameters are serialized only when ``"pml"``
    is among the conditions.

    Args:
        name: Optional human-readable assignment name.
        boundaries: Boundary label or labels where the condition applies.
        conditions: Condition name or names to apply.
        pml_wavelengths: PML width measured in local wavelengths.
        pml_exponent: PML complex-stretching exponent.
        pml_reflectivity: Target PML reflectivity coefficient.
        pml_constant: Optional PML stretching constant.
        extra: Additional solver-facing fields.
        **kwargs: Additional solver-facing fields.

    Raises:
        ValueError: If required conditions are missing or conflicting PML
            reflection arguments are supplied.
        TypeError: If removed solver fields are provided.
    """

    name: Optional[str] = None
    boundaries: List[BoundaryLabel] = field(default_factory=list)
    conditions: Optional[ConditionInput] = None

    pml_wavelengths: float = 0.5
    pml_exponent: float = 3.0
    pml_reflectivity: Optional[float] = None
    pml_constant: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: Optional[str] = None,
        boundaries: Optional[Union[BoundaryLabel, Sequence[BoundaryLabel]]] = None,
        conditions: ConditionInput = None,
        pml_wavelengths: float = 0.5,
        pml_exponent: float = 3.0,
        pml_reflectivity: Optional[float] = None,
        pml_constant: Optional[float] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        pml_reflection = kwargs.pop("pml_reflection", None)
        _reject_removed_fields(kwargs, "BoundaryCondition")
        extra_payload = copy.deepcopy(dict(extra or {}))
        _reject_removed_fields(extra_payload, "BoundaryCondition")
        self.name = name
        self.boundaries = [] if boundaries is None else boundaries
        self.conditions = conditions
        self.pml_wavelengths = 0.5 if pml_wavelengths is None else pml_wavelengths
        self.pml_exponent = 3.0 if pml_exponent is None else pml_exponent
        self.pml_reflectivity = _resolve_pml_reflectivity(
            pml_reflectivity,
            pml_reflection,
        )
        self.pml_constant = pml_constant
        self._init_extra(extra_payload, **kwargs)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.conditions is None:
            raise ValueError("BoundaryCondition requires `conditions`")
        if self.pml_reflectivity is None:
            self.pml_reflectivity = _DEFAULT_PML_REFLECTIVITY

        self.conditions = _normalize_conditions(self.conditions)
        self.boundaries = _normalize_boundaries(self.boundaries)

    @classmethod
    def from_fs(cls, data: Dict) -> "BoundaryCondition":
        """Deserialize a boundary-condition assignment.

        Args:
            data: Serialized boundary-condition mapping.

        Returns:
            ``BoundaryCondition`` instance.
        """

        data = copy.deepcopy(data)
        data.pop("_type", None)
        _reject_removed_fields(data, "BoundaryCondition")
        pml_reflection = data.pop("pml_reflection", None)
        pml_reflectivity = data.pop("pml_reflectivity", None)
        kwargs: Dict[str, Any] = {
            "name": data.pop("name", None),
            "boundaries": data.pop("boundaries"),
            "conditions": data.pop("conditions"),
        }
        if "pml_wavelengths" in data:
            kwargs["pml_wavelengths"] = data.pop("pml_wavelengths")
        if "pml_exponent" in data:
            kwargs["pml_exponent"] = data.pop("pml_exponent")
        if "pml_constant" in data:
            kwargs["pml_constant"] = data.pop("pml_constant")
        if pml_reflection is not None or pml_reflectivity is not None:
            kwargs["pml_reflectivity"] = _resolve_pml_reflectivity(
                pml_reflectivity,
                pml_reflection,
            )
        kwargs["extra"] = data
        return cls(**kwargs)

    def has_condition(self, condition: str) -> bool:
        """Return whether this assignment includes a condition.

        Args:
            condition: Condition name to check. The name is normalized using the
                same rules as constructor input.

        Returns:
            ``True`` when the normalized condition is present.
        """

        return _normalize_condition(condition) in self.conditions

    def to_fs(self, ctx=None) -> Dict:
        """Serialize the assignment for solver input.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible boundary-condition payload.
        """

        bc_dict = {
            "conditions": list(self.conditions),
            "boundaries": self.boundaries,
            **(
                {
                    "pml_wavelengths": self.pml_wavelengths,
                    "pml_exponent": self.pml_exponent,
                    "pml_reflectivity": self.pml_reflectivity,
                    **(
                        {"pml_constant": self.pml_constant}
                        if self.pml_constant is not None
                        else {}
                    ),
                }
                if self.has_condition("pml")
                else {}
            ),
        }
        if self.name:
            bc_dict["name"] = self.name
        return merge_extra(bc_dict, self.extra, "BoundaryCondition")


class BoundaryConditions(NamedList):
    """Named collection of boundary conditions attached to a simulation.

    Args:
        conditions: Initial boundary conditions or serialized condition
            mappings.
    """

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
        """Append a boundary condition and update covered boundary labels.

        Args:
            bc: Boundary condition instance or serialized condition mapping.

        Raises:
            TypeError: If ``bc`` cannot be converted to ``BoundaryCondition``.
        """

        bc = self._coerce_bc(bc)
        self._boundaries.update(bc.boundaries)
        super().append(bc)

    def __setitem__(
        self,
        key: Union[str, int],
        value: Union[BoundaryCondition, Mapping[str, Any]],
    ) -> None:
        """Replace a boundary condition by name or index.

        Args:
            key: Boundary-condition name or integer index.
            value: Boundary condition instance or serialized mapping.
        """

        super().__setitem__(key, self._coerce_bc(value))
        self._rebuild_boundaries()

    def __iadd__(
        self, bc: Union[BoundaryCondition, Mapping[str, Any]]
    ) -> "BoundaryConditions":
        """Append a boundary condition and return the collection."""

        self.append(bc)
        return self

    def verify(self, mesh: Any) -> None:
        """Verify that all labeled mesh boundaries have assignments.

        Args:
            mesh: Mesh-like object with a ``boundaries`` attribute.

        Raises:
            ValueError: If a mesh boundary label is not covered by any stored
                boundary condition.
        """

        boundaries = getattr(mesh, "boundaries", None)
        if boundaries is None:
            return
        for boundary in boundaries:
            if boundary not in self._boundaries:
                raise ValueError(f"Boundary {boundary} is not assigned a BC")

    @classmethod
    def from_fs(cls, data: List[Dict]) -> "BoundaryConditions":
        """Deserialize a boundary-condition collection.

        Args:
            data: List of serialized boundary-condition mappings.

        Returns:
            ``BoundaryConditions`` instance.

        Raises:
            TypeError: If ``data`` is not a list.
        """

        data = copy.deepcopy(data)
        if not isinstance(data, list):
            raise TypeError("BCs must be a list of boundary condition payloads")
        return cls(BoundaryCondition.from_fs(bc_data) for bc_data in data)

    def to_fs(self, ctx=None) -> List[Dict]:
        """Serialize all boundary conditions for solver input.

        Args:
            ctx: Optional export context forwarded to each condition.

        Returns:
            List of JSON-compatible boundary-condition payloads.
        """

        return [bc.to_fs(ctx) for bc in self]
