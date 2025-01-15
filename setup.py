from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "frequensolve.mesh._mesh",
        ["src/frequensolve/mesh/mesh_bindings.cpp"],
    ),
]

setup(
    # ... other args ...
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
) 