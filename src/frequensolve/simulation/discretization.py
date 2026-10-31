"""Finite-element discretization settings for simulations."""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from frequensolve.util.mixins import ExportContext, ExtraFieldsMixin, merge_extra

__all__ = ["Discretization"]


@dataclass
class Discretization(ExtraFieldsMixin):
    """Finite-element discretization settings for a simulation.

    Args:
        **kwargs: Optional solver-facing discretization settings. The default
            discretization requires no explicit fields.

    ``Discretization(method="Galerkin", quadrature="ainsworth")`` selects
    Ainsworth-Wajid volume quadrature on quadrilaterals and hexahedra in
    supporting backends. Other element shapes and boundary integrals retain
    their existing rules. The default is ``quadrature="gauss"``.

    Raises:
        ValueError: If legacy ``order`` is supplied here instead of through mesh
            adaptivity.
    """

    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Create discretization settings.

        Solver order is now configured through mesh adaptivity, so legacy
        ``order`` values are rejected with an actionable error.
        """

        if "order" in kwargs:
            raise ValueError(
                "'order' has moved from Discretization to mesh adaptivity; "
                "use mesh.set_adapt(..., order=...) instead."
            )
        self._init_extra(None, **kwargs)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "Discretization":
        """Deserialize discretization settings from solver JSON."""

        payload = copy.deepcopy(dict(data))
        return cls(**payload)

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        """Serialize discretization settings for solver input."""

        return merge_extra({}, self.extra, "Discretization")
