from .base_site import *  # noqa
from .local import *  # noqa

try:
    from .aws import *  # noqa
    from .hpc import *  # noqa
except Exception as e:
    print(f"Error importing HPC site: {e}")
    print(
        "The 'parrallel' dependencies are required to use remote sites,"
        " install them with 'poetry install --with parallel'"
    )
