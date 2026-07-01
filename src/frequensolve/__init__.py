"""Public package entrypoint for the FrequenSolve Python SDK.

The root namespace re-exports the stable authoring API from the main
subpackages so users can write ``import frequensolve as fs``. Optional execution
backends are guarded in ``frequensolve.orchestrator.sites``: their names are
always importable, but missing extras fail with an install hint when used.
"""

from frequensolve._exports import unique_exports
from frequensolve._loading import *  # noqa: F403
from frequensolve._loading import __all__ as _loading_all
from frequensolve._version import get_versions
from frequensolve.geometry import *  # noqa: F403
from frequensolve.geometry import __all__ as _geometry_all
from frequensolve.mesh import *  # noqa: F403
from frequensolve.mesh import __all__ as _mesh_all
from frequensolve.model import *  # noqa: F403
from frequensolve.model import __all__ as _model_all
from frequensolve.orchestrator import *  # noqa: F403
from frequensolve.orchestrator import __all__ as _orchestrator_all
from frequensolve.plotting import *  # noqa: F403
from frequensolve.plotting import __all__ as _plotting_all
from frequensolve.project import *  # noqa: F403
from frequensolve.project import __all__ as _project_all
from frequensolve.seismic import *  # noqa: F403
from frequensolve.seismic import __all__ as _seismic_all
from frequensolve.simulation import *  # noqa: F403
from frequensolve.simulation import __all__ as _simulation_all
from frequensolve.units import *  # noqa: F403
from frequensolve.units import __all__ as _units_all
from frequensolve.util import *  # noqa: F403
from frequensolve.util import __all__ as _util_all
from frequensolve.validation import *  # noqa: F403
from frequensolve.validation import __all__ as _validation_all

_colormap_all = ["get_colormap", "RdYlBu", "RdYlBu_r", "BuGrOr", "BuGrOr_r"]

__version__ = get_versions()["version"]

__all__ = [
    "__version__",
    *unique_exports(
        _project_all,
        _units_all,
        _geometry_all,
        _model_all,
        _mesh_all,
        _seismic_all,
        _plotting_all,
        _simulation_all,
        _orchestrator_all,
        _util_all,
        _validation_all,
        _loading_all,
        _colormap_all,
    ),
]


def __getattr__(name):
    if name in _colormap_all:
        from frequensolve.util import colormaps

        return getattr(colormaps, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *__all__})
