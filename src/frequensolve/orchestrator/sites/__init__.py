from .base import *  # noqa
from .local import *  # noqa
from .stampede3 import *  # noqa

try:
    from .aws import *  # noqa
    from .hpc import *  # noqa
except ModuleNotFoundError as e:
    print(
        "The 'parallel' dependencies are required to use remote sites,"
        " install them with 'poetry install --with parallel'"
    )
except Exception as e:
    print(e)
