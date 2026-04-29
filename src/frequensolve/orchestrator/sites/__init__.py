from .base import *  # noqa

try:
    from .local import *  # noqa
except ModuleNotFoundError as e:
    print(
        "The 'dask' dependencies are required to use LocalSite,"
        " install them with 'poetry install --with dask'"
    )
except Exception as e:
    print(e)

try:
    from .aws import *  # noqa
    from .hpc import *  # noqa
    from .stampede3 import *  # noqa
except ModuleNotFoundError as e:
    print(
        "The 'parallel' dependencies are required to use remote sites,"
        " install them with 'poetry install --with parallel'"
    )
except Exception as e:
    print(e)
