"""Model and material-property APIs."""

from frequensolve.model.dispersion import (
    DispersionRelation,
    DispersionScaling,
    PowerLawDispersion,
    TablulatedDispersion,
)
from frequensolve.model.model import ModelBase, ModelSubdomain
from frequensolve.model.property import (
    Property,
    PropertyExpression,
    PropertyMap,
    canonical_property_name,
    prop,
)

__all__ = [
    "DispersionRelation",
    "DispersionScaling",
    "ModelBase",
    "ModelSubdomain",
    "PowerLawDispersion",
    "Property",
    "PropertyExpression",
    "PropertyMap",
    "TablulatedDispersion",
    "canonical_property_name",
    "prop",
]
