import json
from pathlib import Path

import numpy as np
import pytest

from frequensolve.imaging._artifacts import ControlStateFile, ControlVectorFile
from frequensolve.imaging._backend import (
    Backend,
    LinearizationCache,
    LinearizationEntry,
    content_fingerprint,
    fingerprint,
    frequency_weights,
    read_manifest,
    read_report,
    read_smoothed_covector,
    read_state_output,
    read_task_objective_vectors,
    reduce_covectors,
    total_value,
    write_task_objective_vectors,
)
from frequensolve.imaging.data import DataSpace, DataVector
from frequensolve.imaging.jobs import FWIOperatorJob
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.orchestrator.sites.base import RunFailedError
from frequensolve.project.project import Project
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import ReceiverComponent, ReceiverNode
from tests.imaging_fakes import FakeImagingSite

pytestmark = pytest.mark.unit

ACTIVE = ["model.vp", "model.rho"]
SIZES = {"model.vp": 5, "model.rho": 3}
FREQUENCIES = [4.0, 6.0]


def _simulation(tmp_path):
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="fake", physics="elastic", dimension=2)
    sim.model.x_limits = [0.0, 1.0]
    sim.model.z_limits = [0.0, 1.0]
    sim.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0.0, 0.0], u_bound=[1.0, 1.0], n=[1, 1])
    )
    acq = Acquisition()
    acq.add_sources(
        kind="vector", coords=np.array([[0.4, 0.1], [0.6, 0.1]]), direction=[0.0, 1.0]
    )
    device = ReceiverNode(
        name="geophone", components=[ReceiverComponent(name="vz", field="velocity")]
    )
    acq.add_receiver_group(
        name="surface",
        device=device,
        coords=np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
    )
    sim.acquisition = acq
    sim.save()
    return sim


@pytest.fixture
def setup(tmp_path):
    sim = _simulation(tmp_path)
    fake = FakeImagingSite(SIZES, seed=7)
    backend = Backend(fake, sim.project_path, submit_options={"n_ranks": 2})
    return sim, fake, backend


def _baseline_state(backend, seed=1):
    rng = np.random.default_rng(seed)
    state = ControlStateFile(
        {name: rng.standard_normal(size) for name, size in SIZES.items()}
    )
    path = backend.staging_dir("baseline") / "state.h5"
    state.write(path)
    return state, path


def _linearize(backend, sim, *, control_state=None, **kwargs):
    job = FWIOperatorJob(
        backend.job_name("linearize"),
        sim,
        FREQUENCIES,
        action="linearize",
        active=ACTIVE,
        state="state.json",
        covector="gradient.h5",
        objective="report.json",
        control_state=control_state,
        state_output="baseline.h5",
        manifest="controls.json",
        **kwargs,
    )
    backend.run(job)
    return job


def _direction(backend, lin, values, name):
    vector = ControlVectorFile.from_packed(
        values,
        lin.sizes,
        state_fingerprint=lin.state_fingerprint,
        control_registry_fingerprint=lin.control_registry_fingerprint,
    )
    return vector.write(backend.staging_dir("directions") / f"{name}.h5")


def _only(fake):
    assert len(fake.linearizations) == 1
    return next(iter(fake.linearizations.values()))


# ---------------------------------------------------------------------------
# linearize: covectors, reports, baseline, manifest
# ---------------------------------------------------------------------------


def test_linearize_reduces_task_covectors_reports_state_and_manifest(setup):
    sim, fake, backend = setup
    state, state_path = _baseline_state(backend)
    job = _linearize(backend, sim, control_state=state_path)
    lin = _only(fake)

    m = np.concatenate([state[name] for name in ACTIVE])
    np.testing.assert_allclose(lin.m, m)
    space = DataSpace.from_simulation(sim, frequencies=FREQUENCIES)
    assert lin.J.shape == (space.size, sum(SIZES.values()))

    parts = [ControlVectorFile.read(job.covector_file(t)) for t in (1, 2)]
    assert [set(p.names) for p in parts] == [set(ACTIVE)] * 2  # HDF5 order
    reduced = reduce_covectors(job)
    np.testing.assert_allclose(
        reduced.pack(ACTIVE), parts[0].pack(ACTIVE) + parts[1].pack(ACTIVE)
    )
    np.testing.assert_allclose(reduced.pack(ACTIVE), lin.gradient, rtol=1e-12)
    assert reduced.state_fingerprint == lin.state_fingerprint
    assert reduced.control_registry_fingerprint == lin.control_registry_fingerprint
    assert reduced.support == {}  # all-ones masks are dropped
    assert reduced.support_mask("model.vp").all()

    weighted = reduce_covectors(job, weights=[2.0, 0.5])
    np.testing.assert_allclose(
        weighted.pack(ACTIVE), 2.0 * parts[0].pack(ACTIVE) + 0.5 * parts[1].pack(ACTIVE)
    )
    with pytest.raises(ValueError, match="frequency weights"):
        frequency_weights(job, [1.0])

    reports = read_report(job)
    assert len(reports) == 2
    residual = lin.residual
    for task, report in enumerate(reports, start=1):
        rows = lin.rows(FREQUENCIES[task - 1])
        expected = 0.5 * float(np.vdot(residual[rows], residual[rows]).real)
        assert report.total == pytest.approx(expected)
        assert [term.id for term in report.terms] == ["surface"]
        assert report.state_fingerprint == lin.state_fingerprint
    assert total_value(reports) == pytest.approx(
        0.5 * float(np.vdot(residual, residual).real)
    )
    assert total_value(reports, [1.0, 0.0]) == pytest.approx(reports[0].total)

    output = read_state_output(job)
    assert set(output.names) == set(state.names)
    np.testing.assert_allclose(output["model.vp"], state["model.vp"])
    assert output.support_mask("model.rho").all()

    manifest = read_manifest(job)
    assert manifest.fingerprint == lin.control_registry_fingerprint
    assert manifest.active_names == tuple(ACTIVE)
    assert manifest.active_layout() == {
        "model.vp": slice(0, 5),
        "model.rho": slice(5, 8),
    }
    assert manifest.state_size == sum(SIZES.values())


def test_linearize_without_control_state_uses_zero_baseline(setup):
    sim, fake, backend = setup
    job = _linearize(backend, sim)
    lin = _only(fake)
    np.testing.assert_array_equal(lin.m, np.zeros(8))
    reduced = reduce_covectors(job)
    np.testing.assert_allclose(reduced.pack(ACTIVE), -np.real(lin.J.conj().T @ lin.d))
    # the surrogate is stable across fake instances and frequency subsets
    J, d, _ = FakeImagingSite(SIZES, seed=7).surrogate(sim, ACTIVE, SIZES, FREQUENCIES)
    np.testing.assert_array_equal(J, lin.J)
    np.testing.assert_array_equal(d, lin.d)
    J6, _, space6 = fake.surrogate(sim, ACTIVE, SIZES, [6.0])
    np.testing.assert_array_equal(J6, lin.J[lin.rows(6.0)])
    assert space6.size == J6.shape[0]
    forward = fake.forward(sim, reduced, FREQUENCIES, active=ACTIVE)
    np.testing.assert_allclose(forward.values, lin.J @ reduced.pack(ACTIVE))
    np.testing.assert_array_equal(fake.observed(sim, ACTIVE, FREQUENCIES).values, lin.d)


# ---------------------------------------------------------------------------
# jvp / vjp / normal through the file layer
# ---------------------------------------------------------------------------


def test_objective_vectors_round_trip_per_task(setup):
    sim, fake, backend = setup
    space = DataSpace.from_simulation(sim, frequencies=FREQUENCIES)
    r = space.random(seed=3)
    job = FWIOperatorJob(
        "roundtrip",
        sim,
        FREQUENCIES,
        action="jvp",
        active=ACTIVE,
        state="state.json",
        direction="unused.h5",
        objective_vector="dual.json",
    )
    state_fp = "sha256:" + "a" * 64
    paths = write_task_objective_vectors(job, r, space, state_fp)
    assert [p.name for p in paths] == ["dual_1.json", "dual_2.json"]
    assert all(p.with_suffix(".h5").is_file() for p in paths)
    back = read_task_objective_vectors(job, space, state_fingerprint=state_fp)
    np.testing.assert_allclose(back.values, r.values)
    for task in (1, 2):
        part = DataVector.read_objective_vector(
            paths[task - 1], space, frequency=FREQUENCIES[task - 1]
        )
        rows = np.concatenate(
            [l.indices for l in space.term_layouts(frequency=FREQUENCIES[task - 1])]
        )
        np.testing.assert_allclose(part.values[rows], r.values[rows])
        other = np.setdiff1d(np.arange(space.size), rows)
        np.testing.assert_array_equal(part.values[other], 0.0)
    with pytest.raises(ValueError, match="another baseline"):
        read_task_objective_vectors(job, space, state_fingerprint="sha256:" + "b" * 64)


def test_jvp_vjp_normal_are_adjoint_consistent_through_files(setup):
    sim, fake, backend = setup
    _, state_path = _baseline_state(backend)
    linearize = _linearize(backend, sim, control_state=state_path)
    lin = _only(fake)
    space = lin.space
    rng = np.random.default_rng(11)
    dv = rng.standard_normal(lin.J.shape[1])
    direction = _direction(backend, lin, dv, "dv")

    jvp = FWIOperatorJob(
        backend.job_name("jvp"),
        sim,
        FREQUENCIES,
        action="jvp",
        active=ACTIVE,
        state=linearize.state_file(),
        direction=direction,
        objective_vector="jvp.json",
    )
    backend.run(jvp)
    j_dv = read_task_objective_vectors(
        jvp, space, state_fingerprint=lin.state_fingerprint
    )
    np.testing.assert_allclose(j_dv.values, lin.J @ dv, rtol=1e-12)

    r = space.random(seed=5)
    vjp = FWIOperatorJob(
        backend.job_name("vjp"),
        sim,
        FREQUENCIES,
        action="vjp",
        active=ACTIVE,
        state=linearize.state_file(),
        objective_vector=backend.staging_dir("duals") / "r.json",
        covector="vjp.h5",
    )
    write_task_objective_vectors(vjp, r, space, lin.state_fingerprint)
    backend.run(vjp)
    jh_r = reduce_covectors(vjp)
    np.testing.assert_allclose(
        jh_r.pack(ACTIVE), np.real(lin.J.conj().T @ r.values), rtol=1e-12
    )
    assert jh_r.state_fingerprint == lin.state_fingerprint
    # <J dv, r>_Re == <dv, J^H r>
    assert j_dv.dot(r) == pytest.approx(float(dv @ jh_r.pack(ACTIVE)), rel=1e-10)

    normal = FWIOperatorJob(
        backend.job_name("normal"),
        sim,
        FREQUENCIES,
        action="normal",
        active=ACTIVE,
        state=linearize.state_file(),
        direction=direction,
        covector="normal.h5",
    )
    backend.run(normal)
    h_dv = reduce_covectors(normal).pack(ACTIVE)
    np.testing.assert_allclose(h_dv, np.real(lin.J.conj().T @ (lin.J @ dv)), rtol=1e-12)

    # normal == vjp(jvp(dv)) through the files
    vjp2 = FWIOperatorJob(
        backend.job_name("vjp"),
        sim,
        FREQUENCIES,
        action="vjp",
        active=ACTIVE,
        state=linearize.state_file(),
        objective_vector=jvp.objective_vector_file(),
        covector="vjp2.h5",
    )
    backend.run(vjp2)
    np.testing.assert_allclose(reduce_covectors(vjp2).pack(ACTIVE), h_dv, rtol=1e-12)

    # every action of the linearization was submitted with the pinned profile
    assert [s["action"] for s in fake.submissions] == [
        "linearize",
        "jvp",
        "vjp",
        "normal",
        "vjp",
    ]
    assert all(s["options"] == {"n_ranks": 2, "fetch": True} for s in fake.submissions)


def test_direction_bound_to_another_state_fails_the_run(setup):
    sim, fake, backend = setup
    linearize = _linearize(backend, sim)
    lin = _only(fake)
    stale = ControlVectorFile.from_packed(
        np.ones(8),
        lin.sizes,
        state_fingerprint="sha256:" + "c" * 64,
        control_registry_fingerprint=lin.control_registry_fingerprint,
    ).write(backend.staging_dir("directions") / "stale.h5")
    jvp = FWIOperatorJob(
        backend.job_name("jvp"),
        sim,
        FREQUENCIES,
        action="jvp",
        active=ACTIVE,
        state=linearize.state_file(),
        direction=stale,
        objective_vector="jvp.json",
    )
    with pytest.raises(RunFailedError, match="another objective state"):
        backend.run(jvp)
    result = backend.run(jvp, check=False)
    assert not result.successful
    assert result.status.state == "failed"


# ---------------------------------------------------------------------------
# smoothing postprocess
# ---------------------------------------------------------------------------


def test_smoothing_postprocess_writes_weighted_aggregate_and_raw(setup):
    sim, fake, backend = setup
    job = _linearize(
        backend, sim, smoothing={"type": "tv", "lambda": 0.2}, weights=[2.0, 0.5]
    )
    assert job.requires_postprocess()
    parts = [ControlVectorFile.read(job.covector_file(t)).pack(ACTIVE) for t in (1, 2)]
    expected = 2.0 * parts[0] + 0.5 * parts[1]
    smoothed = read_smoothed_covector(job)
    np.testing.assert_allclose(smoothed.pack(ACTIVE), expected)
    np.testing.assert_allclose(
        read_smoothed_covector(job, raw=True).pack(ACTIVE), expected
    )
    np.testing.assert_allclose(reduce_covectors(job).pack(ACTIVE), expected)
    assert not job.needs_postprocess()

    job.covector_file().unlink()
    assert job.needs_postprocess()
    result = backend.run(job, postprocess_only=True)
    assert result.successful
    np.testing.assert_allclose(read_smoothed_covector(job).pack(ACTIVE), expected)
    assert fake.submissions[-1]["options"] == {"n_ranks": 2, "fetch": True}

    plain = _linearize(backend, sim)
    with pytest.raises(ValueError, match="no smoothing postprocess"):
        read_smoothed_covector(plain)


# ---------------------------------------------------------------------------
# backend: naming, families, dry run
# ---------------------------------------------------------------------------


def test_backend_names_jobs_runs_families_and_describes_dry_runs(setup):
    sim, fake, backend = setup
    assert backend.job_name() == "imaging_0001"
    assert backend.job_name("jvp") == "imaging_jvp_0002"
    assert backend.owns(backend.staging_dir("x") / "y.h5")
    assert not backend.owns(sim.project_path.parent)
    with pytest.raises(ValueError, match="run\\(\\) owns it"):
        Backend(fake, sim.project_path, submit_options={"check": True})

    jobs = [
        FWIOperatorJob(
            backend.job_name("linearize"),
            sim,
            [frequency],
            action="linearize",
            active=ACTIVE,
            state="state.json",
            covector="gradient.h5",
        )
        for frequency in FREQUENCIES
    ]
    results = backend.run_many(jobs)
    assert [r.job.name for r in results] == [job.name for job in jobs]
    assert all(r.successful for r in results)
    assert backend.run_many([]) == []
    assert len(fake.linearizations) == 1  # same baseline, one state fingerprint

    plan = backend.dry_run(jobs[0])
    assert plan["action"] == "linearize"
    assert plan["n_tasks"] == 1
    assert plan["submit_options"] == {"n_ranks": 2}
    assert plan["outputs"]["state"] == [str(jobs[0].state_file(1))]
    assert plan["outputs"]["covector"] == [str(jobs[0].covector_file(1))]
    assert plan["job"]["fwi_operator"]["action"] == "linearize"
    json.dumps(plan)  # JSON compatible

    smoothed = FWIOperatorJob(
        "smoothed",
        sim,
        FREQUENCIES,
        action="linearize",
        active=ACTIVE,
        state="state.json",
        covector="gradient.h5",
        smoothing={"type": "tv"},
    )
    plan = backend.dry_run(smoothed)
    assert plan["requires_postprocess"]
    assert plan["outputs"]["smoothed_covector"] == str(smoothed.covector_file())
    assert plan["outputs"]["raw_covector"].endswith("gradient_raw.h5")
    assert fake.submissions[-1]["job"] == jobs[-1].name  # dry runs never submit


# ---------------------------------------------------------------------------
# cache and fingerprints
# ---------------------------------------------------------------------------


def test_linearization_cache_evicts_lru_entries_inside_workdir(tmp_path):
    sim = _simulation(tmp_path)
    workdir = sim.project_path
    cache = LinearizationCache(workdir, capacity=2)
    entries = []
    for index in range(3):
        job = FWIOperatorJob(
            f"lin{index}",
            sim,
            [4.0],
            action="linearize",
            active=ACTIVE,
            state="state.json",
            covector="gradient.h5",
        )
        directory = job._result_path
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "state_1.json").write_text("{}")
        entries.append(LinearizationEntry(f"fp{index}", job))
    assert entries[0].directory == entries[0].job._result_path
    assert entries[0].state == entries[0].job.state_file()
    assert entries[0].covector == entries[0].job.covector_file()
    assert entries[0].report is None

    assert cache.put(entries[0]) == []
    assert cache.put(entries[1]) == []
    assert cache.get("fp0") is entries[0]  # refresh: fp1 becomes least recent
    evicted = cache.put(entries[2])
    assert [e.fingerprint for e in evicted] == ["fp1"]
    assert cache.keys() == ["fp0", "fp2"]
    assert "fp1" not in cache and len(cache) == 2
    assert not entries[1].directory.exists()
    assert entries[0].directory.exists() and entries[2].directory.exists()
    assert cache.get("missing") is None

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "state_1.json").write_text("{}")
    foreign = LinearizationEntry("foreign", entries[0].job, directory=outside)
    cache.put(foreign)
    assert cache.keys() == ["fp2", "foreign"]
    assert not entries[0].directory.exists()
    cache.evict("foreign")
    assert outside.exists()  # never deletes outside the workdir
    assert cache.clear()[0] is entries[2]
    assert len(cache) == 0 and not entries[2].directory.exists()
    assert cache.evicted == ["fp1", "fp0", "foreign", "fp2"]
    with pytest.raises(ValueError):
        LinearizationCache(workdir, capacity=0)


def test_linearization_entry_from_job_reads_fingerprints(setup):
    sim, fake, backend = setup
    job = _linearize(backend, sim)
    lin = _only(fake)
    entry = LinearizationEntry.from_job("key", job, point="m0")
    assert entry.state_fingerprint == lin.state_fingerprint
    assert entry.control_registry_fingerprint == lin.control_registry_fingerprint
    assert entry.manifest == job.manifest_file()
    assert entry.state_output == job.state_output_file()
    assert entry.extra == {"point": "m0"}
    with pytest.raises(ValueError, match="linearize job"):
        LinearizationEntry.from_job(
            "key",
            FWIOperatorJob(
                "n",
                sim,
                [4.0],
                action="normal",
                active=ACTIVE,
                state="s.json",
                direction="d.h5",
                covector="c.h5",
            ),
        )


def test_fingerprint_is_stable_and_detects_changes(tmp_path):
    parts = dict(
        problem="fwi",
        state="sha256:" + "0" * 64,
        active=("model.vp", "model.rho"),
        frequencies=np.array([4.0, 6.0]),
        misfit={"objective": {"kind": "l2"}},
        observed={"surface": {"kind": "file", "sha256": "sha256:x"}},
    )
    reference = fingerprint(**parts)
    assert reference.startswith("sha256:") and len(reference) == 71
    assert fingerprint(**parts) == reference
    assert fingerprint(**{**parts, "frequencies": [4.0, 6.0]}) == reference
    assert fingerprint(**{**parts, "frequencies": [4.0, 6.5]}) != reference
    assert fingerprint(**{**parts, "active": ("model.rho", "model.vp")}) != reference
    assert (
        fingerprint(**{**parts, "misfit": {"objective": {"kind": "huber"}}})
        != reference
    )
    assert fingerprint(a=1 + 2j) != fingerprint(a=1 - 2j)
    with pytest.raises(ValueError):
        fingerprint()

    path = tmp_path / "observed.h5"
    path.write_bytes(b"one")
    first = content_fingerprint(path)
    assert first["kind"] == "file"
    path.write_bytes(b"two")
    assert content_fingerprint(path) != first
    assert fingerprint(observed=first) != fingerprint(
        observed=content_fingerprint(path)
    )


def test_backend_fetch_defaults_and_explicit_override(setup):
    sim, fake, backend = setup
    job = _linearize(backend, sim)
    assert fake.submissions[-1]["options"]["fetch"] is True
    remote_only = Backend(fake, sim.project_path, submit_options={"fetch": False})
    remote_only.submit(job)
    assert fake.submissions[-1]["options"]["fetch"] is False


class _RemoteRecordingSite(FakeImagingSite):
    """Fake remote site recording uploads under a remote project root."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.work_dir = Path("/remote/project")
        self.uploads = []

    def put(self, local_path, remote_path):
        self.uploads.append((Path(local_path), Path(remote_path)))


def test_remote_sites_receive_client_written_operator_inputs(tmp_path):
    sim = _simulation(tmp_path)
    # Imaging problems run derived simulations that know only their project path.
    sim._project = None
    site = _RemoteRecordingSite(SIZES, seed=7)
    backend = Backend(site, sim.project_path)
    ops = backend.staging_dir("ops", "0001")
    for task in (1, 2):
        (ops / f"direction_{task}.h5").write_bytes(b"d")
    produced = Path(sim.project_path) / "jobs" / sim.name / "linearize" / "results"
    produced.mkdir(parents=True)
    (produced / "baseline.h5").write_bytes(b"b")
    job = FWIOperatorJob(
        backend.job_name("jvp"),
        sim,
        [4.0, 6.0],
        action="jvp",
        active=ACTIVE,
        state="state.json",
        direction=ops / "direction.h5",
        control_state=produced / "baseline.h5",
        objective_vector="jvp.json",
    )

    backend._stage_remote_inputs(job)

    root = Path(sim.project_path).resolve()
    # Per-task inputs travel to the same relative location; solver results stay remote.
    assert site.uploads == [
        (
            ops / f"direction_{task}.h5",
            Path("/remote/project")
            / (ops / f"direction_{task}.h5").resolve().relative_to(root),
        )
        for task in (1, 2)
    ]
    # Sites without a remote work directory read inputs in place.
    site.uploads.clear()
    site.work_dir = None
    backend._stage_remote_inputs(job)
    assert site.uploads == []


def test_remote_staging_uploads_objective_vector_payloads(tmp_path):
    sim = _simulation(tmp_path)
    site = _RemoteRecordingSite(SIZES, seed=7)
    backend = Backend(site, sim.project_path)
    ops = backend.staging_dir("ops", "0002")
    for task in (1, 2):
        (ops / f"dual_{task}.h5").write_bytes(b"rows")
        (ops / f"dual_{task}.json").write_text(json.dumps({"file": f"dual_{task}.h5"}))
    job = FWIOperatorJob(
        backend.job_name("vjp"),
        sim,
        [4.0, 6.0],
        action="vjp",
        active=ACTIVE,
        state="state.json",
        covector="vjp.h5",
        objective_vector=ops / "dual.json",
    )

    backend._stage_remote_inputs(job)

    assert sorted(local.name for local, _ in site.uploads) == [
        "dual_1.h5",
        "dual_1.json",
        "dual_2.h5",
        "dual_2.json",
    ]
    assert {remote.parent for _, remote in site.uploads} == {
        Path("/remote/project")
        / ops.resolve().relative_to(Path(sim.project_path).resolve())
    }
