import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class _Space:
    size = 2


class _Problem:
    data_space = _Space()
    model_space = _Space()

    @staticmethod
    def apply_jacobian(vector):
        return np.asarray(vector) * 2

    @staticmethod
    def apply_adjoint(vector):
        return np.asarray(vector) * 3


def test_pylops_extra_drives_forward_and_adjoint_operator_behavior():
    pylops = pytest.importorskip("pylops")
    from frequensolve.simulation.jobs.fwi import FrequenSolveJacobian

    operator = FrequenSolveJacobian(_Problem())

    assert isinstance(operator, pylops.LinearOperator)
    np.testing.assert_allclose(operator @ np.array([1.0, 2.0]), [2.0, 4.0])
    np.testing.assert_allclose(operator.H @ np.array([1.0, 2.0]), [3.0, 6.0])


def test_inversion_module_remains_importable_without_pylops():
    code = """
import builtins
import importlib

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'pylops' or name.startswith('pylops.'):
        raise ImportError('pylops deliberately unavailable')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
module = importlib.import_module('frequensolve.simulation.jobs.fwi')
if module._LinearOperator.__module__.split('.')[0] != 'scipy':
    raise SystemExit('missing-extra fallback did not use scipy')
"""

    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
