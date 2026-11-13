from types import SimpleNamespace

import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging.focusing import (
    _interpolation,
    _source_rows,
    coherent_focus,
    lag_kernel,
    unit_misfit,
)
from tests.imaging_fakes import FakeImagingSite
from tests.test_imaging_problem import _mechanism_problem, _problem, _surrogate

pytestmark = pytest.mark.unit


def _data(rng, n_f=3, n_s=4, n_r=5):
    rows = []
    for t in range(n_f):
        index = t * n_s * n_r + np.arange(n_s * n_r)
        rows.append((index, np.repeat(np.arange(n_s), n_r)))
    size = n_f * n_s * n_r
    o = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    d = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    return d, o, rows


def test_lag_kernel_is_a_unit_diagonal_gaussian_of_frequency_differences():
    frequencies = [1.0, 1.5 - 0.2j, 3.0]
    np.testing.assert_array_equal(lag_kernel(frequencies, 0.0), np.ones((3, 3)))
    kernel = lag_kernel(frequencies, 0.1)
    np.testing.assert_allclose(np.diag(kernel), 1.0)
    np.testing.assert_allclose(kernel, kernel.T)
    assert kernel[0, 2] == pytest.approx(np.exp(-2 * np.pi**2 * 0.01 * 4.0))


def test_coherent_focus_is_perfect_for_proportional_data():
    rng = np.random.default_rng(0)
    _, o, rows = _data(rng)
    J, G, ratio = coherent_focus((0.3 - 2.0j) * o, o, rows, lag_kernel([1, 2, 3], 0.0))
    assert J == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(ratio, 1.0)
    np.testing.assert_allclose(G, 0.0, atol=1e-12)


def test_coherent_focus_ratio_is_bounded_and_its_dual_matches_finite_differences():
    rng = np.random.default_rng(1)
    d, o, rows = _data(rng)
    kernel = lag_kernel([1.0, 1.5, 2.5], 0.1)
    J, G, ratio = coherent_focus(d, o, rows, kernel)
    assert np.all((ratio >= 0) & (ratio <= 1)) and 0 <= J <= 1
    dd = rng.standard_normal(d.shape) + 1j * rng.standard_normal(d.shape)
    h = 1e-6
    fd = (
        coherent_focus(d + h * dd, o, rows, kernel)[0]
        - coherent_focus(d - h * dd, o, rows, kernel)[0]
    ) / (2 * h)
    assert np.real(np.vdot(G, dd)) == pytest.approx(fd, rel=1e-7)


def test_linearization_exposes_modeled_and_observed_rows(tmp_path):
    fake = FakeImagingSite(seed=7)
    _, problem = _problem(tmp_path, fake)
    lin = problem.linearize(gradient=True)
    surrogate = _surrogate(fake, lin)
    np.testing.assert_array_equal(lin.simulated().values, surrogate.J @ surrogate.m)
    np.testing.assert_array_equal(lin.observed().values, surrogate.d)
    rng = np.random.default_rng(2)
    g = rng.standard_normal(lin.data_space.size) + 1j * rng.standard_normal(
        lin.data_space.size
    )
    np.testing.assert_allclose(
        lin.modeled_vjp(g).values, np.real(surrogate.J.conj().T @ g), atol=1e-12
    )
    parts = lin.modeled_vjp(g, per_task=True)
    assert len(parts) == len(lin.frequencies)
    np.testing.assert_allclose(sum(p.values for p in parts), lin.modeled_vjp(g).values)


def _moved(problem, rng):
    return problem.vector().values + rng.standard_normal(problem.vector().values.size)


def test_point_focus_gradient_matches_finite_differences(tmp_path):
    _, problem = _problem(tmp_path, FakeImagingSite(seed=7))
    focus = problem.focus(im.Focusing(window=0.05))
    rng = np.random.default_rng(3)
    x, dx = _moved(focus, rng), rng.standard_normal(focus.vector().values.size)
    lin = focus.linearize(x)
    assert 0 <= lin.value <= 1 and lin.ratio.shape == (2,)
    h = 1e-6
    fd = (focus.value(x + h * dx) - focus.value(x - h * dx)) / (2 * h)
    assert lin.gradient.values @ dx == pytest.approx(fd, rel=1e-6)
    with pytest.raises(NotImplementedError):
        lin.normal


@pytest.mark.parametrize("baseline_gradient", [False, True])
@pytest.mark.parametrize("window", [0.0, 0.1])
def test_point_focus_reuses_equivalent_l2_baseline(tmp_path, baseline_gradient, window):
    fake = FakeImagingSite(seed=7)
    _, problem = _problem(tmp_path, fake, misfit=unit_misfit(["surface"]))
    problem.linearize(gradient=False)  # Discover the fake's authored registry.
    fake.jobs.clear()
    point = np.linspace(0.1, 0.8, problem.space.size)
    baseline = problem.linearize(point, gradient=baseline_gradient)
    focus = problem.focus(im.Focusing(window=window))
    result = focus.linearize(point)
    assert [job.action for job in fake.jobs] == ["linearize", "vjp"]
    assert focus.problem.linearize(point, gradient=False) is baseline
    surrogate = _surrogate(fake, baseline)
    from frequensolve.imaging.focusing import _rows

    expected, dual, _ = coherent_focus(
        baseline.simulated().values,
        baseline.observed().values,
        _rows(baseline),
        lag_kernel(baseline.frequencies, window),
    )
    assert result.value == pytest.approx(expected)
    np.testing.assert_allclose(
        result.gradient.values, np.real(surrogate.J.conj().T @ dual), atol=1e-12
    )
    assert focus.linearize(point) is result
    assert len(fake.jobs) == 2


@pytest.mark.parametrize("existing_baseline", [False, True])
def test_point_focus_requests_only_receiver_baseline(tmp_path, existing_baseline):
    fake = FakeImagingSite(seed=7)
    _, problem = _problem(
        tmp_path,
        fake,
        misfit=im.Misfit.l2(normalization=im.Normalization.explicit(2.0)),
    )
    problem.linearize(gradient=False)
    fake.jobs.clear()
    point = np.linspace(0.1, 0.8, problem.space.size)
    if existing_baseline:
        problem.linearize(point)
    focus = problem.focus(im.Focusing(window=0.0))
    result = focus.linearize(point)
    # A differently normalized L2 state cannot supply raw-pressure rows.
    jobs = fake.jobs[int(existing_baseline) :]
    assert [job.action for job in jobs] == ["linearize", "vjp"]
    assert jobs[0].covector is None
    assert result.gradient is not None


def test_focusing_problem_views_keep_the_objective_and_fwi_reduces_it(tmp_path):
    _, problem = _problem(tmp_path, FakeImagingSite(seed=7))
    focus = problem.focus(im.Focusing(window=0.05))
    stage = focus.restrict(active=["vp"])
    assert isinstance(stage, im.FocusingProblem) and stage.space.blocks == ("model.vp",)
    assert stage.smoothing is None
    with pytest.raises(ValueError, match="cannot be replaced"):
        focus.restrict(misfit=im.Misfit.l2())
    rng = np.random.default_rng(4)
    problem.state = problem.state_from(_moved(problem, rng))
    start = problem.vector().values.copy()
    im.FWI(focus, im.Stage(frequencies=[4.0, 6.0], iterations=3)).run()
    assert focus.value(problem.vector()) < focus.value(start)


def test_focusing_control_transfer_preserves_objective(tmp_path):
    _, problem = _problem(tmp_path, FakeImagingSite(seed=7))
    focus = problem.focus(im.Focusing(window=0.05))
    focus.state = focus.state_from(_moved(focus, np.random.default_rng(40)))
    refined = focus.with_controls({"vp": im.DepthProfile("vp", "sediment", count=6)})
    assert isinstance(refined, im.FocusingProblem)
    assert refined.focusing is focus.focusing
    assert refined._aux is not focus._aux
    assert refined.full_space.size == focus.full_space.size + 1
    assert isinstance(refined.linearize(), im.FocusingLinearization)


@pytest.mark.parametrize("strategy", ["linear", "pointwise"])
def test_aperture_only_solves_the_focusing_adjoint(tmp_path, strategy):
    fake = FakeImagingSite(seed=7)
    _, problem = _problem(tmp_path, fake)
    focus = problem.focus(
        im.Focusing(aperture=im.SourceAperture(0.3, 0.1), strategy=strategy)
    )
    result = focus.linearize(_moved(focus, np.random.default_rng(41)))
    assert result.gradient is not None
    assert all(job.covector is None for job in fake.jobs if job.action == "linearize")
    assert sum(job.action == "vjp" for job in fake.jobs) == 1


def test_source_aperture_nodes_bounds_and_coarse_interpolation():
    aperture = im.SourceAperture(0.3, 0.1, bounds=[(None, None), (0.05, None)])
    offsets, weights = aperture.offsets(2, "km")
    assert offsets.shape == (25, 2) and weights.max() == pytest.approx(1.0)
    centers = np.array([[1.0, 0.1], [2.0, 0.1]])
    kept, w, coarse = aperture.nodes(centers, "km")
    assert np.all(centers[:, 1:2] + kept[:, 1] >= 0.05) and len(kept) == 15
    assert len(coarse) == 9 and np.all(w > 0)
    a = _interpolation(kept, coarse)
    field = lambda p: 2.0 + 3.0 * p[:, 0] - 1.5 * p[:, 1] + 0.5 * p[:, 0] * p[:, 1]
    np.testing.assert_allclose(a @ field(coarse), field(kept), atol=1e-12)
    with pytest.raises(ValueError, match="no nodes"):
        im.SourceAperture(0.3, 0.1, bounds=[(None, None), (1.0, None)]).nodes(
            centers, "km"
        )


def _encoded(weights):
    encoding = SimpleNamespace(
        frequencies=None, weights=np.asarray(weights), encoding_type="JsonDense"
    )
    return SimpleNamespace(acquisition=SimpleNamespace(source_encoding=encoding))


def test_spatial_focusing_needs_one_real_weighted_source_per_rhs():
    index, weight = _source_rows(_encoded([[0, 2.0, 0], [1.0, 0, 0]]), 3)
    np.testing.assert_array_equal(index, [1, 0])
    np.testing.assert_array_equal(weight, [2.0, 1.0])
    with pytest.raises(NotImplementedError, match="one physical source"):
        _source_rows(_encoded([[1.0, 1.0, 0]]), 3)
    with pytest.raises(NotImplementedError, match="real"):
        _source_rows(_encoded([[1j, 0, 0]]), 3)
    index, weight = _source_rows(
        SimpleNamespace(acquisition=SimpleNamespace(source_encoding=None)), 2
    )
    np.testing.assert_array_equal(index, [0, 1])


@pytest.mark.parametrize("mechanisms", [False, True])
def test_task_vjp_reads_each_covector_once(tmp_path, monkeypatch, mechanisms):
    from frequensolve.imaging._artifacts import ControlVectorFile

    _, problem = (
        _mechanism_problem(tmp_path)
        if mechanisms
        else _problem(tmp_path, FakeImagingSite(seed=7))
    )
    lin = problem.linearize()
    dual = np.ones(lin.data_space.size, complex)
    reads = []
    original = ControlVectorFile.read

    def read(path, **kwargs):
        reads.append(str(path))
        return original(path, **kwargs)

    monkeypatch.setattr(ControlVectorFile, "read", read)
    parts = lin.modeled_vjp(dual, per_task=True)
    assert len(reads) == len(lin.frequencies)
    assert len(set(reads)) == len(reads)
    np.testing.assert_allclose(
        sum(p.values for p in parts), lin.modeled_vjp(dual).values
    )


def _source_problem(tmp_path):
    return _problem(
        tmp_path,
        FakeImagingSite(seed=7),
        controls=im.ControlSpace(
            vp=im.DepthProfile("vp", "sediment", count=5),
            sources=im.SourceParameters(position=True, signature=False),
        ),
    )[1]


def test_point_focus_differentiates_active_source_positions(tmp_path):
    problem = _source_problem(tmp_path)
    focus = problem.focus(im.Focusing(window=0.05))
    rng = np.random.default_rng(17)
    x = _moved(focus, rng)
    direction = np.zeros_like(x)
    for block in focus.space.resolved_blocks:
        if block.kind not in {"profile", "grid", "mesh"}:
            direction[focus.space.slices[block.name]] = rng.normal(size=block.size)
    lin = focus.linearize(x)
    h = 1e-6
    fd = (focus.value(x + h * direction) - focus.value(x - h * direction)) / (2 * h)
    assert abs(fd) > 1e-5
    assert lin.gradient.values @ direction == pytest.approx(fd, rel=1e-6, abs=1e-9)


def test_aperture_rejects_active_or_changed_nonmaterial_controls(tmp_path):
    problem = _source_problem(tmp_path)
    specification = im.Focusing(aperture=im.SourceAperture(0.3, 0.1))
    with pytest.raises(ValueError, match="active material-only"):
        problem.focus(specification)
    focus = problem.restrict(active="vp").focus(specification)
    state = problem.state
    blocks = state.blocks()
    blocks["source.1.position"] = blocks["source.1.position"] + 0.1
    moved = im.ControlState.from_blocks(problem.full_space, blocks)
    with pytest.raises(ValueError, match="authored value"):
        focus.value(moved)
    with pytest.raises(ValueError, match="authored value"):
        focus.state = moved


@pytest.mark.parametrize("strategy", ["linear", "pointwise"])
def test_aperture_gradient_and_frequency_stages(tmp_path, strategy):
    _, problem = _problem(tmp_path, FakeImagingSite(seed=7))
    focus = problem.focus(
        im.Focusing(0.05, aperture=im.SourceAperture(0.3, 0.1), strategy=strategy)
    )
    rng = np.random.default_rng(25)
    x, dx = _moved(focus, rng), rng.normal(size=focus.space.size)
    values = []
    for frequencies in ([4.0], [6.0], [4.0, 6.0]):
        view = focus.restrict(frequencies=frequencies)
        lin = view.linearize(x)
        assert 0 <= lin.value <= 1
        h = 1e-6
        fd = (view.value(x + h * dx) - view.value(x - h * dx)) / (2 * h)
        assert lin.gradient.values @ dx == pytest.approx(fd, rel=1e-6, abs=1e-9)
        aux = (
            view._coarse(lin.state)
            if strategy == "pointwise"
            else view._extended(lin.state)
        )
        assert aux.frequencies == frequencies
        values.append(lin.value)
    assert values[0] != values[1]


@pytest.mark.parametrize("strategy", ["linear", "pointwise"])
def test_aperture_keeps_observations_in_original_encoded_basis(tmp_path, strategy):
    observed = im.ObservedData(
        im.TraceStoreRef(tmp_path / "shots.h5"), source_basis="source_geometry"
    )
    _, problem = _problem(tmp_path, FakeImagingSite(seed=7), observed=observed)
    focus = problem.focus(
        im.Focusing(0.05, aperture=im.SourceAperture(0.3, 0.1), strategy=strategy)
    )
    state = problem.state_from(np.ones(problem.space.size))
    lin = focus._coarse(state) if strategy == "pointwise" else focus._extended(state)
    repeats = len(focus._geometry()["coarse"]) if strategy == "pointwise" else 1
    remapped = focus._observations(lin, repeats)
    base = focus.problem.linearize(gradient=False)
    # Auxiliary operators must not read the old physical-shot file using node ids.
    assert all(group.observed is None for group in lin.problem.observed_groups)
    assert base.problem.observed_groups[0].observed.source_basis == "source_geometry"
    for frequency in lin.frequencies:
        original = base.data_space.term_layouts(frequency=frequency)[0]
        layout = lin.data_space.term_layouts(frequency=frequency)[0]
        values = dict(
            zip(
                map(tuple, original.coordinate_keys),
                base.observed().values[original.indices],
            )
        )
        for index, (source, receiver, component) in zip(
            layout.indices, layout.coordinate_keys
        ):
            assert (
                remapped[index]
                == values[((source - 1) // repeats + 1, receiver, component)]
            )


def test_pointwise_interpolates_matching_normalized_ratios_without_cancellation():
    # Opposing node correlations used to make the global calibration 0/0.
    coarse_d = np.array([1.0, -1.0], complex)
    observed = np.ones(2, complex)
    interpolation = _interpolation(
        np.arange(5.0)[:, None], np.array([0.0, 4.0])[:, None]
    )
    taper = np.array([0.25, 0.75, 1.0, 0.75, 0.25])
    weights = taper / taper.sum() @ interpolation
    value, gradient, ratios = coherent_focus(
        coarse_d, observed, [(np.arange(2), np.arange(2))], np.ones((1, 1)), weights
    )
    assert value == 0.0
    np.testing.assert_allclose(gradient, 0.0)
    np.testing.assert_array_equal(interpolation @ ratios, np.ones(5))


@pytest.mark.parametrize("method,nodes", [("_coarse", 3), ("_extended", 5)])
def test_aperture_encodings_store_only_nonzero_entries(
    tmp_path, monkeypatch, method, nodes
):
    _, problem = _problem(tmp_path, FakeImagingSite())
    focus = problem.focus(im.Focusing(aperture=im.SourceAperture(0.3, 0.1)))
    sources = 1000
    geometry = dict(
        points=np.zeros((sources, 1)),
        source=np.arange(sources),
        weight=np.ones(sources),
        coarse=np.arange(3.0)[:, None],
        offsets=np.arange(5.0)[:, None],
        w=np.ones(5),
    )
    monkeypatch.setattr(focus, "_geometry", lambda: geometry)
    captured = []

    def auxiliary(tag, points, encoding):
        captured.append(encoding)
        return SimpleNamespace(linearize=lambda *args, **kwargs: None)

    monkeypatch.setattr(focus, "_aux_problem", auxiliary)
    monkeypatch.setattr(focus, "_aux_state", lambda *args: None)
    getattr(focus, method)(None)
    encoding = captured[0]
    assert encoding.weights is None and encoding.encoding_type == "Named"
    assert sum(len(field.terms) for field in encoding.fields) == sources * nodes


@pytest.mark.parametrize("default_units,expected", [({}, "km"), ({"length": "m"}, "m")])
def test_aperture_preserves_implicit_source_length_units(
    tmp_path, default_units, expected
):
    from frequensolve.imaging.focusing import _source_points
    from frequensolve.seismic.sources import SourceGeometry

    simulation, _ = _problem(tmp_path, FakeImagingSite())
    simulation.units.defaults = default_units
    simulation.acquisition.set_sources(
        SourceGeometry.points(kind="scalar", coords=[[0.5, 0.08]])
    )
    points, units, _, _ = _source_points(simulation)
    assert units == expected
    np.testing.assert_array_equal(points, [[0.5, 0.08]])


def test_aperture_normalizes_inline_coordinate_metadata(tmp_path):
    from frequensolve import CoordinateValue
    from frequensolve.imaging.focusing import _source_points
    from frequensolve.seismic.sources import PointSource, SourceGeometry
    from frequensolve.units import ureg

    simulation, _ = _problem(tmp_path, FakeImagingSite())
    geometry = SourceGeometry(
        kind="scalar",
        sources=[
            PointSource("a", np.array([1000.0, 10.0]) * ureg.m),
            PointSource("b", CoordinateValue([2.0, 0.01], units="km")),
            PointSource("c", [3.0, 0.01]),  # implicit simulation units: km
        ],
    )
    simulation.acquisition.set_sources(geometry)
    # Include the serialization round trip performed by ImagingProblem.
    points, units, system, _ = _source_points(simulation.copy("coordinate_copy"))
    assert units == "m" and system is None
    np.testing.assert_allclose(points, [[1000, 10], [2000, 10], [3000, 10]])
    for source in geometry.sources:
        source.coordinates = CoordinateValue([1.0, 0.01], units="km", system="local")
    assert _source_points(simulation)[2] == "local"
    geometry.sources[-1].coordinates.system = "global"
    with pytest.raises(ValueError, match="one coordinate system"):
        _source_points(simulation)


def test_aperture_instances_do_not_overwrite_each_others_jobs(tmp_path):
    _, problem = _problem(tmp_path, FakeImagingSite(seed=7))
    problem.state = problem.state_from(np.arange(problem.space.size) + 1.0)
    first, second = [
        problem.focus(
            im.Focusing(
                0.05,
                aperture=im.SourceAperture(0.3, 0.1, coarse=count),
                strategy="pointwise",
            )
        )
        for count in (2, 3)
    ]
    first.value()
    first_job = first._coarse(first.state).job
    saved = first_job.state_file(1).read_bytes()
    second.value()
    second_job = second._coarse(second.state).job
    assert first_job.state_file(1) != second_job.state_file(1)
    assert first_job.state_file(1).read_bytes() == saved
    x = first.vector().values
    direction = np.arange(x.size) + 0.5
    h = 1e-6
    derivative = first.gradient().values @ direction
    finite_difference = (
        first.value(x + h * direction) - first.value(x - h * direction)
    ) / (2 * h)
    assert derivative == pytest.approx(finite_difference, rel=1e-6, abs=1e-9)
    assert first.restrict()._stage_key("coarse") == first._stage_key("coarse")


@pytest.mark.parametrize("laplace", [[0.0], [0.0, -0.2]])
@pytest.mark.parametrize("policy", ["total", "frozen"])
def test_node_signatures_repeat_reorder_and_preserve_spectral_axes(
    tmp_path, laplace, policy
):
    import h5py

    from frequensolve.imaging.focusing import _node_signature
    from frequensolve.seismic import GainDelay, SourceSignature
    from frequensolve.util.mixins import ExportContext

    signals = {1: GainDelay(gain=0.5, delay=0.01), 2: GainDelay(gain=2.0, delay=0.03)}
    signature = SourceSignature(
        signals, frequencies=[4, 6], laplace=laplace, spectral_derivative=policy
    )
    reference = signature.to_fs(ExportContext(tmp_path), source_count=2)
    # Repeated/reversed IDs cross the bounded-copy block boundary.
    indices = np.tile([1, 1, 0], 100)
    mapped = _node_signature(reference, tmp_path, indices, tmp_path / "nodes")
    assert mapped["hash"] != reference["hash"]
    with (
        h5py.File(tmp_path / reference["file"]) as original,
        h5py.File(mapped["file"]) as nodes,
    ):
        np.testing.assert_array_equal(nodes["source_ids"], np.arange(1, 301))
        np.testing.assert_array_equal(nodes["frequency"], original["frequency"])
        if len(laplace) > 1:
            np.testing.assert_array_equal(nodes["laplace"], original["laplace"])
        for name in ["q", "q_f"] if policy == "total" else ["q"]:
            np.testing.assert_array_equal(
                nodes[name], original[name][:][..., indices, :]
            )


@pytest.mark.parametrize("strategy", ["linear", "pointwise"])
def test_auxiliary_signatures_follow_selected_physical_sources(tmp_path, strategy):
    import h5py

    from frequensolve.imaging.focusing import _source_points
    from frequensolve.seismic import GainDelay, SourceSignature

    simulation, problem = _problem(tmp_path, FakeImagingSite())
    simulation.acquisition.source_signature = SourceSignature(
        {1: GainDelay(delay=0.01), 2: GainDelay(delay=0.03)}, frequencies=[4, 6]
    )
    simulation.acquisition.encode_sources([[0.0, 1.0], [1.0, 0.0]])
    problem = im.ImagingProblem(
        simulation,
        controls=im.ControlSpace(**problem.full_space.specs),
        observed={"surface": tmp_path / "observed.h5"},
        frequencies=[4, 6],
        site=FakeImagingSite(),
    )
    focus = problem.focus(
        im.Focusing(aperture=im.SourceAperture(0.3, 0.1), strategy=strategy)
    )
    g = focus._geometry()
    offsets = g["coarse"] if strategy == "pointwise" else g["offsets"]
    points = (g["points"][g["source"]][:, None, :] + offsets[None]).reshape(-1, 2)
    aux = focus._aux_problem(
        "coarse" if strategy == "pointwise" else "extended", points, None
    )
    signature = aux.simulation.acquisition.source_signature
    path = aux.simulation.project_path / signature["file"]
    with h5py.File(path) as table:
        values = table["q"][:]
        actual = values[..., 0] + 1j * values[..., 1]
        expected = np.exp(
            -2j
            * np.pi
            * np.array([4, 6])[:, None]
            * np.repeat([0.03, 0.01], len(offsets))[None, :]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        assert table["source_ids"].size == len(_source_points(aux.simulation)[0])


@pytest.mark.parametrize(
    "selection",
    [
        {"axis": "fourier", "residual": "derivative", "order": 1},
        {"axis": "fourier", "residual": "window", "window": [0, 1]},
    ],
)
def test_focusing_clears_spectral_selection_on_all_views(tmp_path, selection):
    _, problem = _problem(tmp_path, FakeImagingSite(), kernel_derivative=selection)
    focus = problem.focus(im.Focusing(aperture=im.SourceAperture(0.3, 0.1)))
    assert focus.problem.kernel_derivative is None
    assert focus.restrict(kernel_derivative=selection).problem.kernel_derivative is None
    assert problem.kernel_derivative == selection
    g = focus._geometry()
    aux = focus._aux_problem("extended", g["points"], None)
    for view in (focus.problem, aux):
        job = view._linearize_job(view.space, None, gradient=False)
        assert job.kernel_derivative is None


def test_pointwise_sparse_survey_rejected_before_submission(tmp_path):
    from frequensolve.seismic.sparse_survey import (
        ReceiverSampling,
        SparseSurvey,
        SparseTrace,
    )

    fake = FakeImagingSite()
    _, problem = _problem(tmp_path, fake)
    acquisition = problem.simulation.acquisition
    acquisition.add_survey(
        SparseSurvey(name="shots", traces=[SparseTrace(source=1, receiver=1, point=1)])
    )
    acquisition.receiver_groups[0].sampling = ReceiverSampling.sparse("shots")
    submissions = len(fake.submissions)
    with pytest.raises(NotImplementedError, match="requires dense receiver sampling"):
        problem.focus(
            im.Focusing(aperture=im.SourceAperture(0.3, 0.1), strategy="pointwise")
        )
    assert len(fake.submissions) == submissions
    focus = problem.focus(
        im.Focusing(aperture=im.SourceAperture(0.3, 0.1), strategy="linear")
    )
    assert focus.problem.simulation.acquisition.receiver_groups[0].survey == "shots"


def test_focusing_cache_is_bounded_and_recomputes_evicted_results(tmp_path):
    _, problem = _problem(tmp_path, FakeImagingSite(), cache_capacity=2)
    focus = problem.focus(im.Focusing())
    x = np.arange(problem.space.size) + 1.0
    first = focus.linearize(x)
    assert focus.linearize(x, gradient=False) is first
    for step in range(1, 5):
        focus.value(x + step)
        assert len(focus._cache) <= 2
    assert all(result is not first for result in focus._cache.values())
    recomputed = focus.linearize(x)
    assert recomputed is not first
    assert recomputed.value == pytest.approx(first.value)
    np.testing.assert_allclose(recomputed.gradient.values, first.gradient.values)
