from types import SimpleNamespace

import pytest

from frequensolve.util import system_info

pytestmark = pytest.mark.unit


def test_cpu_info_tolerates_platform_without_cpu_frequency(monkeypatch):
    fake_psutil = SimpleNamespace(
        cpu_count=lambda *, logical: 8 if logical else 4,
        virtual_memory=lambda: SimpleNamespace(total=16 * 1024**3),
    )
    monkeypatch.setattr(system_info, "psutil", fake_psutil)

    info = system_info.SystemInfo().get_cpu_info()

    assert info["physical_cores"] == 4
    assert info["logical_cores"] == 8
    assert info["cpu_freq"] == {}
    assert info["memory"] == 16 * 1024
