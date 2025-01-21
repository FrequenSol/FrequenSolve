from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import versioneer

ext_modules = [
    Pybind11Extension(
        "frequensolve.mesh._mesh",
        ["src/frequensolve/mesh/mesh_bindings.cpp"],
    ),
]

setup(
    version=versioneer.get_version(),
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": build_ext,
        **versioneer.get_cmdclass(),
    },
)