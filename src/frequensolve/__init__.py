#import sys
#sys.path.append("/Users/jacobbadger/.pyenv/versions/3.9.13/lib/python3.9/site-packages")

from . import util       # noqa
from . import seismic    # noqa
from . import mesh       # noqa
from . import model      # noqa
from . import simulation # noqa
from . import project    # noqa
from . import executor   # noqa
from . import geometry   # noqa

__all__ = ["seismic", "util", "mesh", "model", "simulation", "project", "executor", "geometry"]
__all__ += project.__all__
print(project.__all__)
__all__ += simulation.__all__
__all__ += util.__all__
__all__ += mesh.__all__
__all__ += model.__all__
__all__ += seismic.__all__
__all__ += executor.__all__
__all__ += geometry.__all__

from . import _version
__version__ = _version.get_versions()['version']
