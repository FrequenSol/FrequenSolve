import pytest

import frequensolve as fs

pytestmark = pytest.mark.unit


def test_base_install_executes_public_authoring_and_units_behavior():
    assert fs.__version__
    assert fs.Project.__name__ == "Project"
    assert str(1 * fs.ureg.meter) == "1 meter"
