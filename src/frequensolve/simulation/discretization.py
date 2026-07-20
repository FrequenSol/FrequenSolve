"""Finite-element discretization settings for simulations."""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict

from frequensolve.util.mixins import ExtraFieldsMixin, merge_extra

__all__ = ["Discretization"]


@dataclass
class Discretization(ExtraFieldsMixin):
    """Finite-element discretization settings for a simulation.

    Args:
        **kwargs: Optional solver-facing discretization settings. The default
            discretization requires no explicit fields.

    Raises:
        ValueError: If legacy ``order`` is supplied here instead of through mesh
            adaptivity.
    """

    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        **kwargs,
    ):
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
    def from_fs(cls, data: Dict[str, Any]) -> "Discretization":
        """Deserialize discretization settings from solver JSON."""

        data = copy.deepcopy(data)
        return cls(**data)

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        """Serialize discretization settings for solver input."""

        return merge_extra({}, self.extra, "Discretization")
