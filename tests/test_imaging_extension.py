"""Fake-site tests of :mod:`frequensolve.imaging.extension` (FWIME layer)."""

from __future__ import annotations

import numpy as np
import pytest

from frequensolve.imaging import (
    ControlSpace,
    ControlVector,
    DataVector,
    DepthProfile,
    ImagingProblem,
    Misfit,
    SourceParameters,
)
from frequensolve.imaging._artifacts import ExtensionSolveReport, ExtensionVectorFile
from frequensolve.imaging.extension import (
    ExtendedProblem,
    Extension,
    ExtensionLinearization,
    ExtensionSpace,
    ExtensionVector,
    HalfOffsets,
    Lags,
)
from frequensolve.imaging.jobs import FWIOperatorJob
from frequensolve.imaging.operators import (
    ExtensionJacobian,
    ExtensionNormal,
    ModelOperator,
    ReducedNormal,
)
from frequensolve.imaging.workflows import FWI, NewtonCG, Stage
from frequensolve.inversion.validation import gradient_taylor_test
from frequensolve.simulation import SolverConfig
from frequensolve.units import ureg as u
from tests.imaging_fakes import FakeImagingSite, layered_simulation
from tests.test_imaging_jobs import _assert_valid

pytestmark = pytest.mark.unit

FREQUENCIES = [4.0, 6.0]


def _space():
    return ControlSpace(
        vp=DepthProfile("vp", "sediment", count=5),
        rho=DepthProfile("rho", "sediment", count=3),
    )


def _extension(**kwargs):
    options = dict(damping=0.3, lag_penalty=1.0, lag_scale=20 * u.ms, tolerance=1e-8)
    options.update(kwargs)
    return Extension(
        [Lags("vp", count=3, origin=-10 * u.ms, spacing=10 * u.ms)], **options
    )


def _simulation(tmp_path, *, relaxed=False):
    sim = layered_simulation(tmp_path / "project")
    sim += SolverConfig(relaxed_assembly=relaxed)
    sim.save()
    return sim


def _problem(tmp_path, fake, **kwargs):
    sim = _simulation(tmp_path)
    options = dict(
        controls=_space(),
        observed={"surface": tmp_path / "observed.h5"},
        frequencies=FREQUENCIES,
        site=fake,
        name="fwime",
    )
    options.update(kwargs)
    return sim, ImagingProblem(sim, **options)


@pytest.fixture
def fake():
    return FakeImagingSite(seed=7)


@pytest.fixture
def setup(tmp_path, fake):
    sim, problem = _problem(tmp_path, fake)
    xp = problem.extend(_extension())
    return sim, problem, xp


def _surrogates(fake, lin):
    """Return ``(FakeLinearization, FakeExtension)`` behind ``lin``."""

    base = fake.linearizations[lin.state_fingerprint]
    sizes = lin.problem.full_space.sizes
    return base, fake.extension_surrogate(base, lin.extension_space.fields_fs(), sizes)


def _dense_reduced_normal(fake, lin):
    base, ext = _surrogates(fake, lin)
    solver = lin.extension.solver_fs()
    total = np.zeros((base.J.shape[1], base.J.shape[1]))
    for weight, frequency in zip(lin.frequency_weights, lin.frequencies):
        rows = base.rows(frequency)
        total += weight * ext.reduced_normal_matrix(rows, base.J[rows], solver)
    return total


# ---------------------------------------------------------------------------
# fields and extension
# ---------------------------------------------------------------------------


def test_lags_and_half_offsets_normalize_units():
    lags = Lags("vp", count=3, origin=-10 * u.ms, spacing=0.01 * u.s)
    assert (lags.count, lags.origin, lags.spacing, lags.units) == (3, -10.0, 10.0, "ms")
    np.testing.assert_allclose(lags.coordinates(), [-0.01, 0.0, 0.01])
    assert lags.to_fs("acoustic_vp") == {
        "control": "acoustic_vp",
        "lags": {"count": 3, "origin": -10.0, "spacing": 10.0, "units": "ms"},
    }
    offsets = HalfOffsets("vp", [[-25.0, 0.0], [0.0, 0.0], [25.0, 0.0]], packet_mb=8)
    assert offsets.n_axis == 3 and offsets.units == "m"
    np.testing.assert_allclose(offsets.coordinates()[:, 0], [-25.0, 0.0, 25.0])
    assert offsets.to_fs()["offsets"]["packet_mb"] == 8.0
    km = HalfOffsets("vp", np.array([[0.1, 0.0]]) * u.km)
    np.testing.assert_allclose(km.coordinates(), [[100.0, 0.0]])
    assert km.units == "m"

    with pytest.raises(ValueError, match="count"):
        Lags("vp", count=0, origin=0.0, spacing=1.0)
    with pytest.raises(ValueError, match="spacing"):
        Lags("vp", count=2, origin=0.0, spacing=0.0)
    with pytest.raises(ValueError, match="quantity"):
        Lags("vp", count=2, origin=1 * u.m, spacing=1.0)
    with pytest.raises(ValueError, match="half_offsets"):
        HalfOffsets("vp", [[1.0, 2.0, 3.0, 4.0]])
    with pytest.raises(ValueError, match="packet_mb"):
        HalfOffsets("vp", [[1.0, 0.0]], packet_mb=0.0)


def test_extension_validates_and_serializes():
    ext = _extension(
        field_scales=[1500.0], cache_mb=0, reduced_normal={"max_iterations": 10}
    )
    payload = ext.to_fs({"vp": "acoustic_vp"})
    assert payload["fields"][0]["control"] == "acoustic_vp"
    assert payload["solver"] == {
        "damping": 0.3,
        "relative_tolerance": 1e-8,
        "absolute_tolerance": 0.0,
        "max_iterations": 100,
        "require_convergence": True,
        "lag_penalty": 1.0,
        "lag_scale": {"value": 20.0, "units": "ms"},
        "field_scales": [1500.0],
        "cache_mb": 0.0,
    }
    assert payload["reduced_normal"] == {"max_iterations": 10}
    assert ext.lag_scale_seconds() == pytest.approx(0.02)
    assert ext.controls == ("vp",)

    with pytest.raises(ValueError, match="damping"):
        _extension(damping=0.0)
    with pytest.raises(ValueError, match="lag_penalty > 0 requires lag_scale"):
        Extension(
            [Lags("vp", count=2, origin=0.0, spacing=1.0)], damping=0.1, lag_penalty=1.0
        )
    with pytest.raises(ValueError, match="offset_penalty > 0"):
        Extension([HalfOffsets("vp", [[1.0, 0.0]])], damping=0.1, offset_penalty=1.0)
    with pytest.raises(ValueError, match="field_scales"):
        _extension(field_scales=[1.0, 2.0])
    with pytest.raises(ValueError, match="duplicate"):
        Extension(
            [
                Lags("vp", count=2, origin=0.0, spacing=1.0),
                HalfOffsets("vp", [[1.0, 0.0]]),
            ],
            damping=0.1,
        )
    with pytest.raises(ValueError, match="reduced_normal"):
        _extension(reduced_normal={"bogus": 1})
    with pytest.raises(ValueError, match="at least one field"):
        Extension([], damping=0.1)
    with pytest.raises(ValueError, match="convertible"):
        Extension(
            [Lags("vp", count=2, origin=0.0, spacing=1.0)],
            damping=0.1,
            lag_penalty=1.0,
            lag_scale=(1.0, "m"),
        )


# ---------------------------------------------------------------------------
# tap space and vectors
# ---------------------------------------------------------------------------


def test_extension_space_layout_and_vector_round_trip(setup, tmp_path):
    sim, problem, xp = setup
    space = xp.extension_space
    assert isinstance(space, ExtensionSpace)
    assert space.controls == ("vp",)
    assert space.control_ids == {"vp": "vp"}
    assert space.spatial_counts == {"vp": 5}
    assert space.size == 15 and space.shapes() == {"vp": (5, 3)}
    assert space.slices == {"vp": slice(0, 15)}
    np.testing.assert_allclose(space.axis_coordinates("vp"), [-0.01, 0.0, 0.01])
    assert space.fields_fs()[0]["control"] == "vp"

    taps = space.pack({"vp": np.arange(15.0).reshape(5, 3, order="F")})
    np.testing.assert_array_equal(taps.values, np.arange(15.0))
    np.testing.assert_array_equal(taps["vp"], np.arange(15.0).reshape(5, 3, order="F"))
    assert taps.fields()["vp"].shape == (5, 3)

    other = space.random(3)
    assert isinstance(taps + other, ExtensionVector)
    np.testing.assert_allclose(
        (2.0 * taps - other).values, 2.0 * taps.values - other.values
    )
    assert taps.dot(other) == pytest.approx(float(taps.values @ other.values))
    assert taps.norm() == pytest.approx(np.linalg.norm(taps.values))
    assert (-taps).values[1] == -1.0 and taps == taps.copy()

    file = taps.to_file(
        fingerprint="sha256:ext", baseline="sha256:state", role="tangent"
    )
    assert isinstance(file, ExtensionVectorFile) and file.size == 15
    assert file.fields[0].axis == "lag" and file.fields[0].control == "vp"
    path = file.write(tmp_path / "taps.h5")
    back = ExtensionVector.from_file(path, space)
    assert back == taps
    np.testing.assert_array_equal(ExtensionVectorFile.read(path).pack(), taps.values)

    dataset = taps.to_xarray()
    assert dataset["vp"].dims == ("depth", "lag") or dataset["vp"].dims[-1] == "lag"
    np.testing.assert_allclose(dataset["vp"].coords["lag"].values, [-0.01, 0.0, 0.01])
    np.testing.assert_array_equal(dataset["vp"].values, taps["vp"])

    with pytest.raises(ValueError, match="entries"):
        ExtensionVector(np.zeros(4), space)
    with pytest.raises(ValueError, match="real"):
        ExtensionVector(np.zeros(15, dtype=complex), space)
    wrong = ExtensionSpace(
        problem.full_space, Lags("vp", count=2, origin=0.0, spacing=1.0)
    )
    with pytest.raises(ValueError, match="shape"):
        ExtensionVector.from_file(path, wrong)
    with pytest.raises(ValueError, match="material"):
        ExtensionSpace(
            ControlSpace(
                vp=DepthProfile("vp", "sediment", count=2), src=SourceParameters()
            ).bind(sim),
            Lags("src", count=2, origin=0.0, spacing=1.0),
        )
    with pytest.raises(ValueError, match="not a block"):
        ExtensionSpace(problem.full_space, Lags("qp", count=2, origin=0.0, spacing=1.0))


# ---------------------------------------------------------------------------
# construction and capabilities
# ---------------------------------------------------------------------------


def test_extend_shares_state_and_rejects_unsupported_setups(tmp_path, fake, setup):
    sim, problem, xp = setup
    assert isinstance(xp, ExtendedProblem)
    assert xp.problem is problem and xp.state is problem.state
    assert xp.space.blocks == ("model.vp", "model.rho")
    assert xp.frequencies == FREQUENCIES and xp.smoothing is None
    assert xp.identity()["extension"]["fields"][0]["control"] == "vp"
    report = xp.capabilities()
    assert report["ok"] and report["taps"] == 15 and report["losses"] == ["l2"]

    view = xp.restrict(frequencies=[4.0], active=["vp"])
    assert isinstance(view, ExtendedProblem)
    assert view.frequencies == [4.0] and view.space.blocks == ("model.vp",)
    assert view.extension is xp.extension and view.state is problem.state

    # source block active
    with pytest.raises(ValueError, match="material control blocks"):
        problem.restrict(active=["vp"])  # sanity: base restrict is fine
        ImagingProblem(
            sim,
            controls=ControlSpace(
                vp=DepthProfile("vp", "sediment", count=2), src=SourceParameters()
            ),
            observed={"surface": "observed.h5"},
            frequencies=FREQUENCIES,
            site=fake,
            name="src",
        ).extend(_extension())
    # phase comparison
    with pytest.raises(ValueError, match="waveform"):
        problem.restrict(misfit=Misfit(comparison="phase_derivative")).extend(
            _extension()
        )
    # unrelaxed assembly is required
    relaxed = ImagingProblem(
        _simulation(tmp_path / "relaxed", relaxed=True),
        controls=_space(),
        observed={"surface": "observed.h5"},
        frequencies=FREQUENCIES,
        site=fake,
        name="relaxed",
    )
    with pytest.raises(ValueError, match="relaxed_assembly"):
        relaxed.extend(_extension())
    # Laplace frequencies
    with pytest.raises(ValueError, match="real frequencies"):
        ImagingProblem(
            sim,
            controls=_space(),
            observed={"surface": "observed.h5"},
            frequencies=[4.0 - 0.5j],
            site=fake,
            name="laplace",
        ).extend(_extension())
    # field must borrow a block of the space
    with pytest.raises(ValueError, match="not a block"):
        problem.extend(
            Extension([Lags("qp", count=2, origin=0.0, spacing=1.0)], damping=0.1)
        )
    with pytest.raises(TypeError):
        problem.extend({"fields": []})


def test_reflectivity_blocks_are_rejected(tmp_path, fake):
    from frequensolve.imaging import ReflectivityField, ReflectivityParameters

    sim = _simulation(tmp_path)
    problem = ImagingProblem(
        sim,
        controls=ControlSpace(
            vp=DepthProfile("vp", "sediment", count=3),
            refl=ReflectivityParameters(
                parameterization="vp_ip",
                fields=[ReflectivityField("ip", layer=2, axis=2, basis="vp")],
            ),
        ),
        observed={"surface": "observed.h5"},
        frequencies=FREQUENCIES,
        site=fake,
        name="refl",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        problem.extend(_extension())
    with pytest.raises(ValueError, match="mutually exclusive"):
        problem.restrict(active=["vp"]).extend(_extension())


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


def _extension_jobs(fake):
    return [
        job
        for job in fake.jobs
        if isinstance(job, FWIOperatorJob) and job.extension is not None
    ]


def test_extension_jobs_validate_against_the_pinned_schema(setup, fake):
    sim, problem, xp = setup
    view = xp.restrict(frequencies=[4.0])
    lin = view.linearize()
    taps, report = view.solve()
    view.gradient()
    dv = view.space.random(1)
    view.normal() @ dv
    t = view.extension_space.random(2)
    lin.jacobian @ t
    lin.jacobian.H @ lin.data_space.random(3)
    lin.tap_normal @ t

    jobs = _extension_jobs(fake)
    kinds = {
        (job.action, job.model_gradient, job.reduced_normal is not None) for job in jobs
    }
    assert kinds == {
        ("linearize", False, False),
        ("solve", False, False),
        ("solve", True, False),
        ("solve", False, True),
        ("jvp", False, False),
        ("vjp", False, False),
        ("normal", False, False),
    }
    for job in jobs:
        payload = _assert_valid(job.to_fs())
        op = payload["fwi_operator"]
        assert op["controls"]["active"] == ["model.vp", "model.rho"]
        assert op["extension"]["fields"] == [
            {
                "control": "vp",
                "lags": {"count": 3, "origin": -10.0, "spacing": 10.0, "units": "ms"},
            }
        ]
        assert "control_sensitivities" not in payload
        if job.action == "solve":
            solver = op["extension"]["solver"]
            assert solver["damping"] == 0.3 and solver["lag_scale"] == {
                "value": 20.0,
                "units": "ms",
            }
            assert solver["solution"].endswith("taps.h5")
            assert solver["report"].endswith("inner_solve.json")
            if job.reduced_normal is not None:
                assert "direction" in op and "covector" in op
                assert op["reduced_normal"] == {}
            elif job.model_gradient:
                assert op["model_gradient"] is True and "covector" in op
            else:
                assert "covector" not in op
        elif job.action == "linearize":
            assert op["extension"]["manifest"].endswith("extension.json")
            assert op["extension"]["covector"].endswith("residual.h5")
            assert "covector" not in op
        elif job.action == "jvp":
            assert "direction" in op["extension"] and "direction" not in op
        elif job.action == "vjp":
            assert "covector" in op["extension"] and "covector" not in op
        else:
            assert {"direction", "covector"} <= set(op["extension"])
    # one single-frequency job per action on the saved task
    assert all(job.n_tasks == 1 for job in jobs)
    assert all(job.f_list == [4.0] for job in jobs)


def test_dry_run_describes_the_extension_linearize(setup):
    sim, problem, xp = setup
    payload = xp.dry_run()
    assert payload["action"] == "linearize" and payload["n_tasks"] == 2
    assert payload["taps"] == 15
    assert payload["extension"]["fields"][0]["control"] == "vp"
    assert payload["job"]["fwi_operator"]["extension"]["manifest"].endswith(
        "extension.json"
    )
    assert payload["registry_discovery"] is True


# ---------------------------------------------------------------------------
# linearization, solve, value, gradient
# ---------------------------------------------------------------------------


def test_linearize_caches_and_carries_the_residual_covector(setup, fake):
    sim, problem, xp = setup
    lin = xp.linearize()
    assert isinstance(lin, ExtensionLinearization)
    assert lin.frequencies == FREQUENCIES and lin.space.blocks == (
        "model.vp",
        "model.rho",
    )
    assert lin.extension_space is xp.extension_space
    assert len(lin.manifests) == 2 and len(lin.extension_fingerprints) == 2
    assert lin.registry_fingerprint == problem._shared.manifest.fingerprint
    base, ext = _surrogates(fake, lin)
    assert lin.extension_fingerprints[0] == (ext.fingerprint, base.state_fingerprint)
    np.testing.assert_allclose(
        lin.covector.values, np.real(ext.B.conj().T @ base.residual)
    )
    assert lin.baseline_value == pytest.approx(
        0.5 * np.vdot(base.residual, base.residual).real
    )
    # cached: no new jobs for the same point
    count = len(fake.jobs)
    assert xp.linearize() is lin and xp.linearize(problem.vector()) is lin
    assert len(fake.jobs) == count
    # a moved point is a new linearization
    moved = xp.linearize(problem.vector() + 0.1)
    assert moved is not lin and moved.fingerprint != lin.fingerprint


def test_solve_matches_the_closed_form_taps(setup, fake):
    sim, problem, xp = setup
    with pytest.raises(ValueError, match="single-frequency"):
        xp.solve()
    solutions = xp.solve_all()
    assert len(solutions) == 2
    lin = xp.linearize()
    base, ext = _surrogates(fake, lin)
    solver = xp.extension.solver_fs()
    for (taps, report), frequency in zip(solutions, FREQUENCIES):
        rows = base.rows(frequency)
        _z, expected = ext.solve(rows, base.residual[rows], solver)
        assert isinstance(taps, ExtensionVector) and isinstance(
            report, ExtensionSolveReport
        )
        np.testing.assert_allclose(taps.values, expected, rtol=1e-10, atol=1e-12)
        assert report.converged and report.lag_scale_seconds == pytest.approx(0.02)
        assert report.reduced_objective is not None
    view = xp.restrict(frequencies=[6.0])
    taps, report = view.solve()
    np.testing.assert_allclose(taps.values, solutions[1][0].values)
    assert view.value() == pytest.approx(report.reduced_objective)
    assert xp.value() == pytest.approx(sum(r.reduced_objective for _, r in solutions))
    assert xp.value() < lin.baseline_value


def test_unconverged_inner_solve_is_reported(tmp_path, fake):
    sim, problem = _problem(tmp_path, fake)
    xp = problem.extend(_extension(max_iterations=0))
    with pytest.raises(RuntimeError, match="did not converge"):
        xp.value()
    lenient = problem.extend(_extension(max_iterations=0, require_convergence=False))
    assert np.isfinite(lenient.value())
    assert not lenient.linearize().converged


def test_value_and_gradient_pass_the_taylor_test(setup, fake):
    sim, problem, xp = setup
    x0 = problem.vector().values + 0.05
    direction = xp.space.random(5).values

    def value(m):
        return xp.value(m)

    def gradient(m):
        g = xp.gradient(m)
        assert isinstance(g, ControlVector) and g.space.equivalent(xp.space)
        return g.values

    lin = xp.linearize(x0)
    base, ext = _surrogates(fake, lin)
    solver = xp.extension.solver_fs()
    expected = np.zeros(base.J.shape[1])
    for frequency in FREQUENCIES:
        rows = base.rows(frequency)
        _z, taps = ext.solve(rows, base.residual[rows], solver)
        expected += np.real(
            base.J[rows].conj().T @ (base.residual[rows] + ext.B[rows] @ taps)
        )
    np.testing.assert_allclose(gradient(x0), expected, rtol=1e-9, atol=1e-12)
    # the reduced gradient came from the solve + model_gradient family (no extra solve)
    before = len(fake.jobs)
    xp.value(x0)
    assert len(fake.jobs) == before

    report = gradient_taylor_test(
        value, gradient, x0, direction, steps=(1e-1, 3e-2, 1e-2, 3e-3)
    )
    assert report["passed"], report


def test_frequency_weights_scale_value_gradient_and_normal(setup):
    sim, problem, xp = setup
    weighted = xp.restrict(weights=[2.0, 0.5])
    lin = xp.linearize()
    wlin = weighted.linearize()
    parts = [r.reduced_objective for r in lin.solve_reports]
    assert weighted.value() == pytest.approx(2.0 * parts[0] + 0.5 * parts[1])
    single_a = xp.restrict(frequencies=[4.0]).gradient().values
    single_b = xp.restrict(frequencies=[6.0]).gradient().values
    np.testing.assert_allclose(
        weighted.gradient().values, 2.0 * single_a + 0.5 * single_b
    )
    dv = xp.space.random(9)
    ha = xp.restrict(frequencies=[4.0]).normal() @ dv
    hb = xp.restrict(frequencies=[6.0]).normal() @ dv
    np.testing.assert_allclose(
        (wlin.normal @ dv).values, 2.0 * ha.values + 0.5 * hb.values, rtol=1e-10
    )


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------


def test_extension_jacobian_is_typed_and_passes_the_adjoint_test(setup, fake):
    sim, problem, xp = setup
    lin = xp.linearize()
    B = lin.jacobian
    assert isinstance(B, ExtensionJacobian) and isinstance(B, ModelOperator)
    assert B.shape == (lin.data_space.size, 15) and B.dtype == np.complex128
    assert B.domain is lin.extension_space and B.range is lin.data_space
    assert "ExtensionSpace[15]" in repr(B)
    base, ext = _surrogates(fake, lin)
    t = lin.extension_space.random(1)
    b_t = B @ t
    assert isinstance(b_t, DataVector)
    np.testing.assert_allclose(b_t.values, ext.B @ t.values, rtol=1e-12)
    r = lin.data_space.random(2)
    bh_r = B.H @ r
    assert isinstance(bh_r, ExtensionVector) and bh_r.space is lin.extension_space
    np.testing.assert_allclose(
        bh_r.values, np.real(ext.B.conj().T @ r.values), rtol=1e-12
    )
    report = B.dot_test(seed=4, tolerance=1e-10)
    assert report["passed"], report
    with pytest.raises(ValueError, match="different extension space"):
        B @ ExtensionSpace(
            problem.full_space, Lags("vp", count=3, origin=0.0, spacing=2.0)
        ).random(1)

    N = lin.tap_normal
    assert isinstance(N, ExtensionNormal) and N.H is N and N.shape == (15, 15)
    n_t = N @ t
    np.testing.assert_allclose(
        n_t.values, np.real(ext.B.conj().T @ (ext.B @ t.values)), rtol=1e-12
    )
    np.testing.assert_allclose(n_t.values, (B.H @ (B @ t)).values, rtol=1e-12)
    # memoized per input
    count = len(fake.jobs)
    B @ t
    N @ t
    assert len(fake.jobs) == count


def test_reduced_normal_is_self_adjoint_and_equals_the_schur_complement(setup, fake):
    sim, problem, xp = setup
    H = xp.normal()
    assert isinstance(H, ReducedNormal) and H.H is H and H.T is H
    assert H.shape == (8, 8) and H.domain is H.range
    assert H.linearization is xp.linearize() and H.point == xp.linearize().point
    dense = _dense_reduced_normal(fake, xp.linearize())
    a = xp.space.random(1)
    b = xp.space.random(2)
    ha = H @ a
    assert isinstance(ha, ControlVector)
    np.testing.assert_allclose(ha.values, dense @ a.values, rtol=1e-9, atol=1e-12)
    hb = H @ b
    assert np.dot(ha.values, b.values) == pytest.approx(np.dot(a.values, hb.values))
    count = len(fake.jobs)
    H @ a
    assert len(fake.jobs) == count
    # the reduced normal is the GN normal minus a positive semidefinite part
    base, ext = _surrogates(fake, xp.linearize())
    gn = np.real(base.J.conj().T @ base.J)
    assert np.all(np.linalg.eigvalsh(gn - dense) >= -1e-9)
    reports = [
        ExtensionSolveReport.load(job.extension_report_file(1))
        for job in _extension_jobs(fake)
        if job.reduced_normal is not None
    ]
    assert reports and all(
        r.reduced_normal.method == "gauss_newton_schur" for r in reports
    )
    assert ReducedNormal(xp, xp.vector()).linearization is xp.linearize()


def test_check_runs_every_identity(setup):
    sim, problem, xp = setup
    report = xp.restrict(frequencies=[4.0]).check(seed=3, tolerance=1e-8)
    assert report["adjoint"]["passed"] and report["normal"]["passed"]
    assert report["reduced_normal"]["passed"] and report["taylor"]["passed"]
    assert report["passed"]


# ---------------------------------------------------------------------------
# FWI on the extended problem
# ---------------------------------------------------------------------------


def test_fwi_decreases_the_reduced_objective(setup, fake):
    sim, problem, xp = setup
    problem.state = problem.state_from(np.linspace(-0.5, 0.5, 8))
    initial = xp.value()
    fwi = FWI(
        xp,
        stages=[Stage(FREQUENCIES, iterations=4)],
        optimizer=NewtonCG(max_cg_iterations=8),
    )
    result = fwi.run(resume=False)
    assert result.stages[0].success
    assert result.stages[0].final_loss.total < initial
    assert result.stages[0].final_loss.total == pytest.approx(
        xp.value(result.state.vector(xp.space))
    )
    assert problem.state is result.state
    assert any(job.reduced_normal is not None for job in _extension_jobs(fake))
    assert any(job.model_gradient for job in _extension_jobs(fake))
