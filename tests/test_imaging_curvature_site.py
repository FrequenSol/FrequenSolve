"""Local curvature operations honor the configured solver and worker budget."""

import subprocess

import pytest

from frequensolve.orchestrator.sites.local.site import LocalSite


@pytest.mark.parametrize(
    ("configured_threads", "active_threads", "expected_threads"),
    [(None, None, 1), (4, None, 4), (4, 8, 8)],
)
def test_curvature_uses_site_solver_threads_and_environment(
    tmp_path, monkeypatch, configured_threads, active_threads, expected_threads
):
    solver = tmp_path / "configured-solver"
    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: str(solver))
    site = LocalSite(
        threads_per_worker=configured_threads,
        environment={"FS_CURVATURE_TEST": "configured"},
    )
    site._active_threads_per_worker = active_threads
    original_environment = site.env.copy()
    request = tmp_path / "request.json"
    request.write_text("{}")
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        kwargs["stdout"].write("curvature completed\n")

    monkeypatch.setattr(subprocess, "run", run)
    site.run_curvature(request)

    args, options = calls.pop()
    assert args == [
        str(solver),
        "-nthreads",
        str(expected_threads),
        "--curvature",
        str(request),
    ]
    assert options["cwd"] == request.parent
    assert options["env"]["FS_CURVATURE_TEST"] == "configured"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        assert options["env"][name] == str(expected_threads)
    assert site.env == original_environment
    assert options["check"] is True
    assert options["stderr"] is subprocess.STDOUT
    assert (tmp_path / "solver.log").read_text() == "curvature completed\n"


@pytest.mark.parametrize("argument", ["environment", "env"])
def test_curvature_preserves_explicit_thread_limits(tmp_path, monkeypatch, argument):
    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/solver")
    limits = {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "3",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "2",
    }
    site = LocalSite(threads_per_worker=8, **{argument: limits})
    request = tmp_path / "request.json"
    request.write_text("{}")
    environments = []
    monkeypatch.setattr(
        subprocess, "run", lambda args, **kwargs: environments.append(kwargs["env"])
    )

    site.run_curvature(request)

    for name, value in limits.items():
        assert environments[0][name] == value


def test_curvature_solver_failure_propagates_with_log(tmp_path, monkeypatch):
    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/solver")
    site = LocalSite()
    request = tmp_path / "request.json"
    request.write_text("{}")

    def fail(args, **kwargs):
        kwargs["stdout"].write("invalid covariance\n")
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        site.run_curvature(request)
    assert (tmp_path / "solver.log").read_text() == "invalid covariance\n"
