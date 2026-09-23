"""Material properties blended across a named implicit surface."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from frequensolve.model.property import Property
from frequensolve.units import is_quantity, quantity_to_fs, unit_expression, ureg
from frequensolve.util.mixins import merge_extra

__all__ = ["BlendProperty"]


class BlendProperty(Property):
    """Two property providers separated by a Sauce implicit surface.

    Negative level-set values select ``inside``. ``width`` is the positive
    transition length, in model units for a number or explicit Pint units.
    Both providers use the wrapper's property units. Geometry evaluation and
    its control derivatives are performed by Sauce; request VTK property
    output to plot the evaluated material.
    """

    def __init__(
        self,
        surface: str,
        *,
        width: Any,
        inside: Any,
        outside: Any,
        units: Any = None,
        **extra: Any,
    ):
        self.surface = str(surface).strip()
        if not self.surface:
            raise ValueError("blend requires a non-empty surface name")
        if is_quantity(width):
            width.to("m")
            width = quantity_to_fs(width)
        if isinstance(width, Mapping):
            width = dict(width)
            if set(width) != {"value", "units"}:
                raise ValueError("blend width requires value and units")
            (float(width["value"]) * ureg(width["units"])).to("m")
            number = float(width["value"])
        else:
            number = float(width)
            width = number
        if not np.isfinite(number) or number <= 0:
            raise ValueError("blend width must be finite and positive")
        self.width = width
        self.inside = Property.from_value(inside)
        self.outside = Property.from_value(outside)
        resolved_units = units or self.inside.units or self.outside.units
        resolved_units = (
            None if resolved_units is None else unit_expression(resolved_units)
        )
        for provider in (self.inside, self.outside):
            if isinstance(provider, BlendProperty) or provider.expression is not None:
                raise ValueError(
                    "blend branches require ordinary or parameterized properties"
                )
            if provider.units is not None and provider.units != resolved_units:
                raise ValueError("blend branches must use the same property units")
        super().__init__(0.0, units=resolved_units, **extra)

    @property
    def is_constant(self) -> bool:
        return False

    def get(self, grid=None):
        raise ValueError(
            "BlendProperty needs Sauce geometry evaluation; use VTK property output"
        )

    def to_fs(
        self, ctx=None, file=None, dataset=None, preserve_inline_coordinates=False
    ):
        providers = {}
        for name in ("inside", "outside"):
            # Each branch needs its own output location for gridded data.
            branch_file = None
            if file is not None:
                from pathlib import Path

                path = Path(file)
                branch_file = path.with_name(f"{path.stem}_{name}{path.suffix}")
            payload = getattr(self, name).to_fs(
                ctx=ctx,
                file=branch_file,
                dataset=f"{dataset}/{name}" if dataset else None,
                preserve_inline_coordinates=preserve_inline_coordinates,
            )
            payload.pop("units", None)
            providers[name] = payload
        payload = {
            "blend": {
                "surface": self.surface,
                "width": copy.deepcopy(self.width),
                **providers,
            }
        }
        if self.units is not None:
            payload["units"] = self.units
        return merge_extra(payload, self.extra, "BlendProperty")

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]):
        payload = dict(data)
        config = dict(payload.pop("blend"))
        return cls(**config, **payload)
