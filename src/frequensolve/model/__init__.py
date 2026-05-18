"""Model and material-property APIs."""

from frequensolve._exports import unique_exports
from frequensolve.model.dispersion import *  # noqa: F403
from frequensolve.model.dispersion import __all__ as _dispersion_all
from frequensolve.model.layered import *  # noqa: F403
from frequensolve.model.layered import __all__ as _layered_all
from frequensolve.model.model import *  # noqa: F403
from frequensolve.model.model import __all__ as _model_all
from frequensolve.model.property import *  # noqa: F403
from frequensolve.model.property import __all__ as _property_all

__all__ = unique_exports(
    _dispersion_all,
    _model_all,
    _layered_all,
    _property_all,
)
