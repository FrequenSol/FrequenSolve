# First, import version info
from frequensolve._version import get_versions

__version__ = get_versions()["version"]

import matplotlib.font_manager
import matplotlib.pyplot as plt

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


def _configure_matplotlib_fonts():
    """Configure default matplotlib fonts based on what's available in the system."""
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]

    # Try to find a suitable sans-serif font
    if "Helvetica" in available_fonts:
        plt.rcParams["font.family"] = "Helvetica"
    if "DejaVu Sans" in available_fonts:
        plt.rcParams["font.family"] = "DejaVu Sans"
    elif "Arial" in available_fonts:
        plt.rcParams["font.family"] = "Arial"
    else:
        plt.rcParams["font.family"] = "sans-serif"


# Configure fonts when the library is imported
_configure_matplotlib_fonts()

load_dotenv()
