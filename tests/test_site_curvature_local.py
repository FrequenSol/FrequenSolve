"""Local Sauce curvature launches: ranks, thread budgets and rank binding."""

import functools
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from frequensolve.imaging import BFGSHistory, NativeCurvature
from frequensolve.imaging._backend import Backend
from frequensolve.orchestrator.sites.base import BaseSite
from frequensolve.orchestrator.sites.local.site import LocalSite
from tests.statistical_mesh_fixture import write_mesh
from tests.test_imaging_curvature import checkpoint


def _site(monkeypatch, solver, **options):
    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: str(solver))
    return LocalSite(**options)


def _request(tmp_path, method, **extra):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(dict(method=method, **extra)))
    return path


def _capture(monkeypatch):
    calls = []
    monkeypatch.setattr(
        subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs))
    )
    return calls


@pytest.mark.unit
@pytest.mark.parametrize(
    "method", ["bfgs_rsvd", "bfgs_action", "covariance_project", "rickett"]
)
def test_distributed_methods_split_the_thread_budget_across_ranks(
    tmp_path, monkeypatch, method
):
    site = _site(monkeypatch, "/opt/sauce/fs2d_s", threads_per_worker=8)
    site.curvature_ranks = 2
    request = _request(tmp_path, method)
    calls = _capture(monkeypatch)

    site.run_curvature(request)

    args, options = calls.pop()
    assert args == [
        "mpirun",
        "-np",
        "2",
        "/opt/sauce/fs2d_s",
        "-nthreads",
        "4",
        "--curvature",
        str(request),
    ]
    assert options["env"]["OMP_NUM_THREADS"] == "4"
    # Open MPI would otherwise bind each rank, and all its threads, to one core.
    assert options["env"]["PRTE_MCA_hwloc_default_binding_policy"] == "none"
    assert options["env"]["OMPI_MCA_hwloc_base_binding_policy"] == "none"
    assert options["cwd"] == tmp_path


@pytest.mark.unit
@pytest.mark.parametrize("method", ["mesh_prior", "mesh_sample", "mesh_directions"])
def test_single_rank_methods_keep_the_whole_budget_on_one_rank(
    tmp_path, monkeypatch, method
):
    site = _site(monkeypatch, "/opt/sauce/fs2d_s", threads_per_worker=10)
    request = _request(tmp_path, method)
    calls = _capture(monkeypatch)

    site.run_curvature(request, ranks=4)
    site.run_curvature(request, ranks=3, threads_per_rank=2)

    assert [args for args, _ in calls] == [
        ["/opt/sauce/fs2d_s", "-nthreads", count, "--curvature", str(request)]
        for count in ("10", "6")
    ]
    assert [kwargs["env"]["OMP_NUM_THREADS"] for _, kwargs in calls] == ["10", "6"]
    for _, kwargs in calls:
        for name in ("PRTE_MCA_hwloc_default_binding_policy",):
            assert kwargs["env"].get(name) == site.env.get(name)


@pytest.mark.unit
def test_explicit_threads_per_rank_and_validation(tmp_path, monkeypatch):
    site = _site(monkeypatch, "/opt/sauce/fs2d_s")
    request = _request(tmp_path, "gaussian_prior")
    calls = _capture(monkeypatch)

    site.run_curvature(request, ranks=3, threads_per_rank=2)

    assert calls[0][0][:6] == [
        "mpirun",
        "-np",
        "3",
        "/opt/sauce/fs2d_s",
        "-nthreads",
        "2",
    ]
    for options in ({"ranks": 0}, {"ranks": 1.5}, {"threads_per_rank": True}):
        with pytest.raises(ValueError, match="positive integer"):
            site.run_curvature(request, **options)
    with pytest.raises(ValueError, match="curvature_ranks"):
        _site(monkeypatch, "/opt/sauce/fs2d_s", curvature_ranks=0)


@pytest.mark.unit
def test_explicit_open_mpi_binding_policy_is_kept(tmp_path, monkeypatch):
    site = _site(
        monkeypatch,
        "/opt/sauce/fs2d_s",
        environment={"PRTE_MCA_hwloc_default_binding_policy": "package"},
    )
    calls = _capture(monkeypatch)

    site.run_curvature(_request(tmp_path, "bfgs_rsvd"), ranks=2)

    environment = calls[0][1]["env"]
    assert environment["PRTE_MCA_hwloc_default_binding_policy"] == "package"
    assert environment["OMPI_MCA_hwloc_base_binding_policy"] == "none"


@pytest.mark.unit
def test_dispatcher_runs_as_configured_and_old_routing_gets_a_hint(
    tmp_path, monkeypatch
):
    dispatcher = tmp_path / "install" / "FS_seismic"
    site = _site(monkeypatch, dispatcher)
    request = _request(tmp_path, "mesh_sample", dimension=2)
    calls = []

    def fail(args, **kwargs):
        calls.append(args)
        kwargs["stdout"].write("Job file not provided.\n")
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError, match="predates --curvature"):
        site.run_curvature(request)
    assert calls[0][0] == str(dispatcher)

    backend = _site(monkeypatch, "/opt/sauce/fs2d_s")
    with pytest.raises(subprocess.CalledProcessError) as error:
        backend.run_curvature(request)
    assert "predates" not in str(error.value)


@pytest.mark.unit
def test_backend_binds_site_options_and_rejects_unknown_ones(tmp_path, monkeypatch):
    site = _site(monkeypatch, "/opt/sauce/fs2d_s", threads_per_worker=8)
    backend = Backend(site, tmp_path)

    default = backend.curvature()
    assert default.runner == site.run_curvature
    configured = backend.curvature(ranks=4)
    assert isinstance(configured.runner, functools.partial)
    assert configured.runner.keywords == {"ranks": 4}
    with pytest.raises(TypeError, match="Unsupported curvature options"):
        backend.curvature(nodes=2)

    class Unsupported(BaseSite):
        pass

    with pytest.raises(NotImplementedError, match="Unsupported does not run"):
        Backend(Unsupported(), tmp_path).curvature()
    with pytest.raises(NotImplementedError, match="Unsupported does not run"):
        Unsupported().run_curvature(tmp_path / "request.json")


def _history(rng, size, pairs):
    factor = rng.normal(size=(size, size)) / np.sqrt(size)
    hessian = factor @ factor.T + np.diag(rng.uniform(0.5, 2.0, size))
    history = BFGSHistory(rng.uniform(0.5, 2.0, size), state="s", coordinates="c")
    steps = rng.normal(size=(pairs, size))
    history(
        checkpoint(
            [dict(step=s, difference=hessian @ s) for s in steps],
            pairs,
        )
    )
    return history


@pytest.mark.integration
def test_native_operations_agree_across_rank_counts(tmp_path, monkeypatch):
    """Row-distributed Sauce results match one rank; mesh methods stay on one."""

    solver = os.environ.get("FS_CURVATURE_SOLVER")
    if not solver:
        pytest.skip("Set FS_CURVATURE_SOLVER to a Sauce solver with --curvature")
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun is required for multi-rank curvature")
    # An inherited FS_SOLVER_PATH would redirect FS_seismic to another install.
    monkeypatch.delenv("FS_SOLVER_PATH", raising=False)
    site = LocalSite(solver=solver, threads_per_worker=4)
    rng = np.random.default_rng(20261001)
    size, pairs = 1009, 7
    history = _history(rng, size, pairs)
    prior = rng.uniform(0.5, 1.5, size)
    vectors = rng.normal(size=(3, size))
    grid = rng.normal(size=(17, 23))
    normal = grid * rng.uniform(0.5, 2.0, size=grid.shape)
    mesh, _ = write_mesh(tmp_path / "mesh.h5")

    results = {}
    for ranks in (1, 2):
        native = NativeCurvature(
            workdir=tmp_path / f"ranks-{ranks}",
            runner=functools.partial(site.run_curvature, ranks=ranks),
        )
        factors = native.bfgs_uncertainty(history, prior_std=prior)
        results[ranks] = dict(
            factors=factors,
            action=native.inverse_action(history, vectors),
            covariance=native.covariance(factors, vectors=vectors),
            rickett=native.rickett(
                grid,
                grid,
                normal_reference=normal,
                state="s",
                coordinates="c",
                smoothing_radii=[2, 3],
            ),
            mesh=native.mesh_prior(mesh, mesh, np.ones(4), np.full(4, 0.5)),
        )
        for name in ("factors", "action", "covariance", "rickett"):
            assert results[ranks][name].metadata["ranks"] == ranks
            assert results[ranks][name].metadata["threads"] == 4 // ranks
        # Mesh transfer is forced onto one rank with the whole thread budget.
        assert results[ranks]["mesh"].metadata["ranks"] == 1
        assert results[ranks]["mesh"].metadata["threads"] == 4
    one, two = results[1], results[2]
    for name, dataset in (
        ("factors", "eigenvalues"),
        ("factors", "variance"),
        ("factors", "standard_deviation"),
        ("action", "actions"),
        ("covariance", "actions"),
        ("rickett", "weights"),
        ("rickett", "normalized"),
        ("mesh", "prior_std"),
        ("mesh", "measure_weights"),
    ):
        np.testing.assert_allclose(
            two[name].read(dataset), one[name].read(dataset), rtol=1e-10, atol=1e-13
        )
