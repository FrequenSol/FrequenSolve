# First, import version info
from frequensolve._version import get_versions

__version__ = get_versions()["version"]

# Load environment
from dotenv import load_dotenv

from frequensolve.geometry import *
from frequensolve.mesh import *

# Then import in dependency order
from frequensolve.model import *
from frequensolve.orchestrator import *
from frequensolve.project import *
from frequensolve.seismic import *
from frequensolve.simulation import *

# Core imports that don't depend on others
from frequensolve.util import *

load_dotenv()
