"""Imaging and inversion authoring API built on the Sauce imaging contracts.

Phase 0 exposes the contract layer and Phase 1 the control spaces
(:mod:`frequensolve.imaging.controls`): observed data and misfit terms
(:mod:`frequensolve.imaging.data`, :mod:`frequensolve.imaging.misfit`), the
solver artifact readers and writers (:mod:`frequensolve.imaging._artifacts`),
and the four low-level jobs (:mod:`frequensolve.imaging.jobs`). Import it as
``from frequensolve import imaging as im``.
"""

from frequensolve._exports import unique_exports
from frequensolve.imaging._artifacts import *  # noqa: F403
from frequensolve.imaging._artifacts import __all__ as _artifacts_all
from frequensolve.imaging.controls import *  # noqa: F403
from frequensolve.imaging.controls import __all__ as _controls_all
from frequensolve.imaging.data import *  # noqa: F403
from frequensolve.imaging.data import __all__ as _data_all
from frequensolve.imaging.jobs import *  # noqa: F403
from frequensolve.imaging.jobs import __all__ as _jobs_all
from frequensolve.imaging.misfit import *  # noqa: F403
from frequensolve.imaging.misfit import __all__ as _misfit_all

__all__ = unique_exports(
    _controls_all, _data_all, _misfit_all, _artifacts_all, _jobs_all
)
