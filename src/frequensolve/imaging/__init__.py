"""Imaging and inversion authoring API built on the Sauce imaging contracts.

Layers, lowest first: the contract layer (:mod:`frequensolve.imaging.data`,
:mod:`frequensolve.imaging.misfit`, :mod:`frequensolve.imaging._artifacts`,
:mod:`frequensolve.imaging.jobs`), control spaces
(:mod:`frequensolve.imaging.controls`), the problem and its linearizations
(:mod:`frequensolve.imaging.problem`) with Jacobian and normal operators
(:mod:`frequensolve.imaging.operators`), regularization and preconditioners
(:mod:`frequensolve.imaging.regularization`), and the workflows
(:mod:`frequensolve.imaging.workflows`, :mod:`frequensolve.imaging.results`),
and the model extension (:mod:`frequensolve.imaging.extension`).
Import it as ``from frequensolve import imaging as im``.
"""

from frequensolve._exports import unique_exports
from frequensolve.imaging._artifacts import *  # noqa: F403
from frequensolve.imaging._artifacts import __all__ as _artifacts_all
from frequensolve.imaging._native_regularization import (
    NativeRegularization as NativeRegularization,
)
from frequensolve.imaging.controls import *  # noqa: F403
from frequensolve.imaging.controls import __all__ as _controls_all
from frequensolve.imaging.data import *  # noqa: F403
from frequensolve.imaging.data import __all__ as _data_all
from frequensolve.imaging.extension import *  # noqa: F403
from frequensolve.imaging.extension import __all__ as _extension_all
from frequensolve.imaging.jobs import *  # noqa: F403
from frequensolve.imaging.jobs import __all__ as _jobs_all
from frequensolve.imaging.misfit import *  # noqa: F403
from frequensolve.imaging.misfit import __all__ as _misfit_all
from frequensolve.imaging.operators import *  # noqa: F403
from frequensolve.imaging.operators import __all__ as _operators_all
from frequensolve.imaging.problem import *  # noqa: F403
from frequensolve.imaging.problem import __all__ as _problem_all
from frequensolve.imaging.property_mesh import PropertyMesh as PropertyMesh
from frequensolve.imaging.regularization import *  # noqa: F403
from frequensolve.imaging.regularization import __all__ as _regularization_all
from frequensolve.imaging.results import *  # noqa: F403
from frequensolve.imaging.results import __all__ as _results_all
from frequensolve.imaging.workflows import *  # noqa: F403
from frequensolve.imaging.workflows import __all__ as _workflows_all

__all__ = unique_exports(
    ["PropertyMesh", "NativeRegularization"],
    _controls_all,
    _data_all,
    _misfit_all,
    _artifacts_all,
    _jobs_all,
    _problem_all,
    _operators_all,
    _regularization_all,
    _workflows_all,
    _results_all,
    _extension_all,
)
