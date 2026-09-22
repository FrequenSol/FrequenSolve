"""Phase 4 removal of the old imaging/FWI job layer (imaging_api.md section 9).

The old ``simulation.jobs.{imaging,fwi,control_sensitivity}`` modules are gone,
``frequensolve.imaging`` is promoted into the flat root namespace, and the
execution sites drive imaging jobs through the generic postprocess protocol and
the artifact catalog roles instead of ``ImagingJob`` special cases.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import frequensolve as fs
import frequensolve.imaging as im
from frequensolve.inversion import ControlLeastSquaresProblem, ControlObjectiveProblem
from frequensolve.orchestrator.sites.base import POSTPROCESS_ARTIFACT_ROLES
from frequensolve.orchestrator.sites.local.site import LocalSite
from frequensolve.simulation.jobs import BaseJob
from frequensolve.simulation.simulation import SeismicSimulation
from tests.test_slurm_site_refactor import (
    DummyJob,
    DummySlurmSite,
    DummySSHClientClass,
    hpc,
)

pytestmark = pytest.mark.unit

_DELETED_MODULES = (
    "frequensolve.simulation.jobs.imaging",
    "frequensolve.simulation.jobs.fwi",
    "frequensolve.simulation.jobs.control_sensitivity",
)
_DELETED_NAMES = (
    "ImagingJob",
    "LSRTMGradientJob",
    "LSRTMNormalJob",
    "HDF5TraceStore",
    "MisfitComparison",
    "MisfitGroup",
    "ObservedTraceDerivatives",
    "PreprocessHook",
    "ImageDatabase",
    "FWIProblem",
    "ModelSpace",
    "FrequenSolveJacobian",
    "ControlBlock",
    "BornControlSensitivityJob",
    "RTMControlSensitivityJob",
    "TimeReversalFocusJob",
    "VariationalSmoothing",
)


# ---------------------------------------------------------------------------
# Root namespace
# ---------------------------------------------------------------------------


def test_root_namespace_exports_the_imaging_api():
    assert fs.Misfit is im.Misfit
    assert "imaging" in fs._PUBLIC_MODULE_ORDER
    assert "imaging" in fs._PUBLIC_EXPORT_ORDER
    for name in (
        "ImagingProblem",
        "DepthProfile",
        "Misfit",
        "FWI",
        "ControlSpace",
        "DataSpace",
        "Preprocess",
        "ImageKernelJob",
        "SmoothJob",
        "ControlGradientJob",
        "FWIOperatorJob",
    ):
        assert name in im.__all__, name
        assert name in fs.__all__, name
        assert getattr(fs, name) is getattr(im, name), name


def test_imaging_exports_do_not_collide_with_other_public_modules():
    imaging = set(im.__all__)
    for module_name in fs._PUBLIC_MODULE_ORDER:
        if module_name == "imaging":
            continue
        module = importlib.import_module(f"frequensolve.{module_name}")
        assert not imaging & set(module.__all__), module_name


def test_old_job_layer_is_gone():
    for module_name in _DELETED_MODULES:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)
    jobs = importlib.import_module("frequensolve.simulation.jobs")
    representation = importlib.import_module("frequensolve.model.representation")
    for name in _DELETED_NAMES:
        assert name not in jobs.__all__, name
        assert name not in representation.__all__, name
        assert name not in fs.__all__, name
        with pytest.raises(AttributeError):
            getattr(fs, name)
    for method in ("fwi", "imaging_job", "imaging"):
        assert not hasattr(SeismicSimulation, method), method


def test_job_deserialization_still_resolves_imaging_jobs(tmp_path):
    simulation = SeismicSimulation(
        name="controlled", physics="acoustic", dimension=2, project_path=tmp_path
    )
    simulation.save()
    job = im.ControlGradientJob(
        "born", simulation, [2.0], kind="born", direction=tmp_path / "direction.h5"
    )
    (tmp_path / "direction.h5").write_bytes(b"direction")

    restored = BaseJob.load(job.save())

    assert isinstance(restored, im.ControlGradientJob)
    assert restored.to_fs() == job.to_fs()
    with pytest.raises(ValueError, match="Unknown job class"):
        BaseJob.from_fs({"_type": "ImagingJob", "workflow": "imaging"})


# ---------------------------------------------------------------------------
# postprocess_only
# ---------------------------------------------------------------------------


def test_postprocess_only_defaults_to_false_and_smooth_jobs_opt_in():
    assert BaseJob.postprocess_only is False
    assert im.SmoothJob.postprocess_only is True
    assert im.ImageKernelJob.postprocess_only is False
    assert im.ControlGradientJob.postprocess_only is False


class _PostprocessOnlyJob:
    """Job-shaped stand-in that declares itself postprocess-only."""

    name = "smooth"
    n_tasks = 2
    postprocess_only = True
    trace_manifest = None
    run_metadata = None
    _job_id = None

    def __init__(self, tmp_path):
        self.states = []
        self._file = tmp_path / "job.json"
        self._file.write_text("{}")
        self._stdout_path = tmp_path / "logs"

    def is_run_current(self):
        return False

    def requires_postprocess(self):
        return True

    def postprocess_part_outputs_exist(self):
        return True

    def postprocess_output_exists(self):
        return False

    def needs_postprocess(self):
        return True

    def write_run_state(self, status="completed", **extra):
        self.states.append((status, extra))

    def _remote_path(self, work_dir):
        return Path(work_dir) / "jobs" / self.name


def test_local_submit_honors_the_job_postprocess_only_attribute(monkeypatch, tmp_path):
    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/bin/echo")
    site = LocalSite()
    monkeypatch.setattr(site, "check_solver_compatibility", lambda **_: None)
    monkeypatch.setattr(site, "prepare_job", lambda *_, **__: None)
    monkeypatch.setattr(
        site,
        "_submit_local_tasks",
        lambda *args, **kwargs: pytest.fail("frequency tasks were planned"),
    )

    run = site.submit(_PostprocessOnlyJob(tmp_path), shutdown_on_completion=False)

    assert run.backend["smooth_only"] is True
    assert run.backend["futures"] == []
    assert run.backend["task_plan"]["pending_indices"] == []


def test_slurm_submit_honors_the_job_postprocess_only_attribute(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: True))
    site = DummySlurmSite("project/run")
    monkeypatch.setattr(site, "prepare_job", lambda *_, **__: None)
    monkeypatch.setattr(site, "_reattach_inflight_run", lambda *_, **__: None)
    monkeypatch.setattr(site, "is_run_current", lambda job: False)
    monkeypatch.setattr(site, "_remote_postprocess_needed", lambda job: False)
    monkeypatch.setattr(
        site,
        "_submit_attached",
        lambda *args, **kwargs: pytest.fail("attached sweep was submitted"),
    )
    captured = {}

    def submit_batch(job, config, **kwargs):
        captured.update(kwargs)
        return "42"

    monkeypatch.setattr(site, "_submit_slurm_batch", submit_batch)

    run = site.submit(_PostprocessOnlyJob(tmp_path))

    assert run.id == "42"
    assert captured["smooth_only"] is True
    with pytest.raises(ValueError, match="batch"):
        site.submit(_PostprocessOnlyJob(tmp_path), mode="attached")
    with pytest.raises(ValueError, match="postprocess_only"):
        site.submit(DummyJob(), postprocess_only=True)


def test_slurm_batch_scripts_take_the_postprocess_job_flag(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")

    script = site._sweep_SLURM_script(
        n_tasks=2,
        n_nodes=1,
        stdout="/scratch/user/jobs/rtm/logs",
        duration="00-00:10:00",
        postprocess_job=True,
    )
    plain = site._sweep_SLURM_script(
        n_tasks=2,
        n_nodes=1,
        stdout="/scratch/user/jobs/rtm/logs",
        duration="00-00:10:00",
    )

    assert "--smooth" in script
    assert "--smooth" not in plain


def test_aws_submit_rejects_postprocess_only_submissions(tmp_path):
    aws = pytest.importorskip("frequensolve.orchestrator.sites.aws.aws")
    site = object.__new__(aws.AWSSite)

    with pytest.raises(NotImplementedError, match="postprocess-only"):
        site.submit(_PostprocessOnlyJob(tmp_path))
    with pytest.raises(NotImplementedError, match="postprocess-only"):
        site.submit(SimpleNamespace(name="forward"), postprocess_only=True)


# ---------------------------------------------------------------------------
# Image and postprocess fetching collapse onto catalog roles
# ---------------------------------------------------------------------------


def test_local_fetch_image_opens_image_sets_by_job(monkeypatch):
    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/bin/echo")
    site = LocalSite()
    first = SimpleNamespace(name="a", load_images=lambda: "images-a")
    second = SimpleNamespace(name="b", load_images=lambda: "images-b")

    assert site.fetch_image(first) == "images-a"
    assert site.fetch_image([first, second]) == {"a": "images-a", "b": "images-b"}
    with pytest.raises(TypeError, match="ImageKernelJob"):
        site.fetch_image([first, SimpleNamespace(name="forward")])


def test_slurm_fetch_outputs_fetches_every_postprocess_role(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")
    calls = []

    def fetch_artifacts(job, **kwargs):
        calls.append(kwargs)
        return (
            [tmp_path / "results" / "image.h5"]
            if not kwargs["include_defaults"]
            else []
        )

    monkeypatch.setattr(site, "fetch_artifacts", fetch_artifacts)
    job = SimpleNamespace(
        name="rtm",
        _local_path=tmp_path,
        requires_postprocess=lambda: True,
        load_images=lambda: "images",
    )

    site.fetch_outputs(job)

    assert calls[0] == {"requests": (), "include_defaults": True}
    assert calls[1]["operations"] == ("smooth",)
    assert calls[1]["include_defaults"] is False
    assert tuple(request.role for request in calls[1]["requests"]) == (
        POSTPROCESS_ARTIFACT_ROLES
    )
    assert {
        "image",
        "gradient",
        "objective",
        "state",
        "objective_vector",
        "extension",
    } <= set(POSTPROCESS_ARTIFACT_ROLES)


# ---------------------------------------------------------------------------
# Inversion toolkit no longer depends on the deleted ControlSpace
# ---------------------------------------------------------------------------


class _Space:
    def __init__(self, size):
        self.size = size

    def pack(self, values):
        return np.concatenate(
            [np.asarray(v, dtype=float).ravel() for v in values.values()]
        )


def test_least_squares_adapters_accept_any_control_space_like_object():
    matrix = np.array([[1.0 + 1.0j, 0.5], [0.25j, 2.0]])
    problem = ControlLeastSquaresProblem(
        _Space(2),
        matrix @ np.array([0.1, 0.2]),
        forward=lambda model: matrix @ model,
        jacobian=lambda model: matrix,
    )

    np.testing.assert_allclose(
        problem.residual({"a": [0.1], "b": [0.2]}), 0.0, atol=1e-12
    )
    with pytest.raises(ValueError, match="real-valued"):
        problem.residual(np.array([0.1 + 0.0j, 0.2]))
    with pytest.raises(ValueError, match="expected \\(2,\\)"):
        problem.residual(np.array([0.1, 0.2, 0.3]))
    with pytest.raises(TypeError, match="control space"):
        ControlObjectiveProblem(object(), value_gradient=lambda model: (0.0, model))
