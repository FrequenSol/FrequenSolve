"""Local defaults use available execution budgets, not unrestricted host totals."""

import builtins
from types import SimpleNamespace

import pytest

from frequensolve.orchestrator.sites.local import config


@pytest.fixture
def resources(monkeypatch):
    host = {"physical_cores": 14, "logical_cores": 28, "memory": 8192.0}
    limits = {"cores": 2, "bytes": 4096 * 1024**2}
    monkeypatch.setattr(config.SystemInfo, "get_cpu_info", lambda self: host)
    monkeypatch.setattr(
        config, "_parallel_resource_limits", lambda: (limits["cores"], limits["bytes"])
    )
    return host, limits


def test_local_defaults_respect_container_limits(resources):
    actual = config.LocalSiteConfig()
    assert actual.cores == 2
    assert actual.memory == 4096.0
    assert actual.mpi_wrapper == "mpirun"


def test_unrestricted_parallel_limits_do_not_increase_physical_host_budget(resources):
    _, limits = resources
    limits.update(cores=28, bytes=16384 * 1024**2)
    actual = config.LocalSiteConfig()
    assert actual.cores == 14
    assert actual.memory == 8192.0


@pytest.mark.parametrize("physical", [None, 0, -1])
def test_unavailable_physical_count_falls_back_to_logical_then_applies_limit(
    resources, physical
):
    host, limits = resources
    host["physical_cores"] = physical
    limits["cores"] = 20
    assert config.LocalSiteConfig().cores == 20


def test_missing_cpu_metadata_uses_os_fallback(resources, monkeypatch):
    host, limits = resources
    host.update(physical_cores=None, logical_cores=None)
    limits.update(cores=None, bytes=None)
    monkeypatch.setattr(config.os, "cpu_count", lambda: 3)
    assert config.LocalSiteConfig().cores == 3
    monkeypatch.setattr(config.os, "cpu_count", lambda: None)
    assert config.LocalSiteConfig().cores == 1


def test_missing_host_memory_uses_available_backend_budget(resources):
    host, _ = resources
    host["memory"] = None
    assert config.LocalSiteConfig().memory == 4096.0


def test_no_parallel_extra_preserves_host_defaults(resources):
    _, limits = resources
    limits.update(cores=None, bytes=None)
    actual = config.LocalSiteConfig()
    assert actual.cores == 14
    assert actual.memory == 8192.0


def test_byte_to_mib_conversion_preserves_fractional_budget(resources):
    _, limits = resources
    limits["bytes"] = 1536 * 1024
    assert config.LocalSiteConfig().memory == 1.5


def test_optional_resource_detection_does_not_require_parallel_imports(monkeypatch):
    original_import = builtins.__import__

    def without_parallel(name, *args, **kwargs):
        if name in {"dask.system", "distributed.system"}:
            raise ImportError("parallel extra absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_parallel)
    assert config._parallel_resource_limits() == (None, None)


def test_detection_uses_current_backend_helpers(monkeypatch):
    original_import = builtins.__import__

    def backend_resources(name, *args, **kwargs):
        if name == "dask.system":
            return SimpleNamespace(cpu_count=lambda: 3)
        if name == "distributed.system":
            return SimpleNamespace(memory_limit=lambda: 123456789)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", backend_resources)
    assert config._parallel_resource_limits() == (3, 123456789)


@pytest.mark.parametrize("explicit", [False, True])
def test_worker_settings_consume_available_budget_without_changing_explicit_values(
    resources, monkeypatch, explicit
):
    from frequensolve.orchestrator.sites.local import LocalSite

    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/fake/test-solver")
    kwargs = (
        dict(n_workers=1, threads_per_worker=1, memory_per_worker=512)
        if explicit
        else {}
    )
    with LocalSite(**kwargs) as site:
        workers, threads, memory = site._cluster_settings()
        assert (workers, threads, memory) == ((1, 1, 512) if explicit else (2, 1, 1843))
        assert workers * threads <= site.config.cores
        assert workers * memory <= site.config.memory


@pytest.mark.parametrize("overcommit", ["cpu", "memory"])
def test_explicit_overcommit_is_rejected_before_starting_workers(
    resources, monkeypatch, overcommit
):
    from frequensolve.orchestrator.sites.local import LocalSite

    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/fake/test-solver")
    kwargs = (
        dict(n_workers=3, threads_per_worker=1)
        if overcommit == "cpu"
        else dict(n_workers=1, threads_per_worker=1, memory_per_worker=8192)
    )
    with LocalSite(**kwargs) as site:
        with pytest.raises(ValueError, match="exceed available"):
            site._initialize_dask()
        assert site._dask_client is None
        assert site._dask_cluster is None
