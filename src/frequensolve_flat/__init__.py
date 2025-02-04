from frequensolve.geometry import *
from frequensolve.mesh import *
from frequensolve.model import *
from frequensolve.orchestrator import *
from frequensolve.project import *
from frequensolve.seismic import *
from frequensolve.simulation import *
from frequensolve.util import *

__all__ = []
from frequensolve.geometry import __all__ as geometry_all
from frequensolve.mesh import __all__ as mesh_all
from frequensolve.model import __all__ as model_all
from frequensolve.orchestrator import __all__ as orchestrator_all
from frequensolve.project import __all__ as project_all
from frequensolve.seismic import __all__ as seismic_all
from frequensolve.simulation import __all__ as simulation_all
from frequensolve.util import __all__ as util_all

__all__ += project_all
__all__ += simulation_all
__all__ += util_all
__all__ += mesh_all
__all__ += model_all
__all__ += seismic_all
__all__ += orchestrator_all
__all__ += geometry_all

# Version info
from frequensolve._version import get_versions

__version__ = get_versions()["version"]
