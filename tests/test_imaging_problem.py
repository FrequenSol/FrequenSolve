import json

import numpy as np
import pytest

from frequensolve.imaging import (
    ControlSpace,
    ControlState,
    ControlVector,
    DataVector,
    DepthProfile,
    ImagingProblem,
    Jacobian,
    Linearization,
    Misfit,
    Normal,
    ObservedData,
    SourceParameters,
)
from frequensolve.imaging._artifacts import ControlStateFile, ControlVectorFile
from frequensolve.model.parameterization import ParameterizedProperty
from tests.imaging_fakes import FakeImagingSite, layered_simulation

pytestmark = pytest.mark.unit

FREQUENCIES = [4.0, 6.0]
ACTIVE = ["model.vp", "model.rho"]


def _space():
    return ControlSpace(
        vp=DepthProfile("vp", "sediment", count=5),
        rho=DepthProfile("rho", "sediment", count=3),
    )


def _problem(tmp_path, fake, **kwargs):
    sim = layered_simulation(tmp_path / "project")
    options = dict(
        controls=_space(),
        observed={"surface": tmp_path / "observed.h5"},
        frequencies=FREQUENCIES,
        site=fake,
        name="fwi",
    )
    options.update(kwargs)
    return sim, ImagingProblem(sim, **options)


@pytest.fixture
def fake():
    return FakeImagingSite(seed=7)


@pytest.fixture
def setup(tmp_path, fake):
    sim, problem = _problem(tmp_path, fake)
    return sim, problem


def _surrogate(fake, lin):
    return fake.linearizations[lin.state_fingerprint]


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_construction_binds_a_renamed_copy_and_leaves_the_simulation_untouched(
    tmp_path, fake
):
    sim = layered_simulation(tmp_path / "project")
    authored = sim.project_path / "simulations" / "shelf" / "shelf.json"
    before = authored.read_bytes()

    problem = ImagingProblem(
        sim,
        controls=_space(),
        observed={"surface": "observed.h5"},
        frequencies=FREQUENCIES,
        site=fake,
        name="fwi",
    )
    problem.linearize()

    assert authored.read_bytes() == before
    sediment = next(s for s in sim.model.subdomains if s.name == "sediment")
    assert not isinstance(sediment.properties["vp"], ParameterizedProperty)
    bound = next(s for s in problem.simulation.model.subdomains if s.name == "sediment")
    assert isinstance(bound.properties["vp"], ParameterizedProperty)
    assert problem.simulation.name == "shelf__fwi"
    assert problem.space.blocks == tuple(ACTIVE)
    assert problem.full_space.blocks == tuple(ACTIVE)
    assert problem.frequencies == FREQUENCIES
    assert problem.data_space.size == 2 * 2 * 1 * 3
    assert problem.workdir == sim.project_path / "imaging" / "fwi"
    assert problem.site is fake
    assert problem.min_support is None and problem.smoothing is None
    assert [group.name for group in problem.observed_groups] == ["surface"]
    assert problem.observed_groups[0].observed == sim.project_path / "observed.h5"
    assert isinstance(problem.observed_data, ObservedData)
    assert "fwi" in repr(problem)


def test_construction_accepts_observed_data_and_infers_frequencies(tmp_path, fake):
    sim = layered_simulation(tmp_path / "project")
    observed = ObservedData({"surface": "observed.h5"}, frequencies=[3.0, 5.0])

    problem = ImagingProblem(
        sim, controls=_space(), observed=observed, site=fake, workdir=tmp_path / "w"
    )

    assert problem.observed_data is observed
    assert problem.frequencies == [3.0, 5.0]
    assert problem.workdir == tmp_path / "w"
    assert isinstance(problem.misfit, Misfit)
    with pytest.raises(ValueError, match="frequencies are required"):
        ImagingProblem(
            sim, controls=_space(), observed={"surface": "observed.h5"}, site=fake
        )
    with pytest.raises(TypeError, match="Misfit"):
        ImagingProblem(
            sim,
            controls=_space(),
            observed=observed,
            site=fake,
            misfit={"objective": {"kind": "l2"}},
        )
    with pytest.raises(ValueError, match="min_support"):
        ImagingProblem(
            sim, controls=_space(), observed=observed, site=fake, min_support=-1.0
        )


def test_capabilities_reject_source_controls_with_phase_comparisons(tmp_path, fake):
    sim = layered_simulation(tmp_path / "project")
    controls = ControlSpace(
        vp=DepthProfile("vp", "sediment", count=4), src=SourceParameters()
    )
    problem = ImagingProblem(
        sim,
        controls=controls,
        observed={"surface": "observed.h5"},
        frequencies=FREQUENCIES,
        site=fake,
    )
    report = problem.capabilities()
    assert report["ok"] and report["kinds"] == ["profile", "source"]
    assert report["comparisons"] == ["waveform"]

    with pytest.raises(ValueError, match="waveform comparisons"):
        ImagingProblem(
            sim,
            controls=controls,
            observed={"surface": "observed.h5"},
            frequencies=FREQUENCIES,
            site=fake,
            misfit=Misfit(comparison="phase_derivative"),
        )
    with pytest.raises(ValueError, match="model"):
        ImagingProblem(
            sim,
            controls=SourceParameters(),
            observed={"surface": "observed.h5"},
            frequencies=FREQUENCIES,
            site=fake,
            smoothing={"type": "tv"},
        )


# ---------------------------------------------------------------------------
# state and vectors
# ---------------------------------------------------------------------------


def test_state_vector_and_state_from_round_trip(setup):
    _sim, problem = setup
    state = problem.state
    assert isinstance(state, ControlState)
    assert state.space.blocks == tuple(ACTIVE)
    np.testing.assert_array_equal(state.values, np.zeros(8))

    v = problem.vector()
    assert isinstance(v, ControlVector) and v.size == 8
    update = problem.state_from(np.arange(8.0))
    np.testing.assert_array_equal(update.values, np.arange(8.0))
    assert problem.state is state  # state_from does not mutate

    problem.state = update
    np.testing.assert_array_equal(problem.vector().values, np.arange(8.0))
    with pytest.raises(TypeError):
        problem.state = np.arange(8.0)
    with pytest.raises(ValueError, match="control blocks"):
        problem.state = ControlState(problem.space.restrict("vp"), np.zeros(5))
    with pytest.raises(ValueError):
        problem.vector(ControlState(problem.space.restrict("vp"), np.zeros(5)))


# ---------------------------------------------------------------------------
# linearize
# ---------------------------------------------------------------------------


def test_linearize_matches_the_surrogate_value_gradient_and_report(setup, fake):
    _sim, problem = setup
    problem.state = problem.state_from(np.linspace(-1.0, 1.0, 8))

    lin = problem.linearize()

    assert isinstance(lin, Linearization)
    surrogate = _surrogate(fake, lin)
    np.testing.assert_allclose(surrogate.m, np.linspace(-1.0, 1.0, 8))
    residual = surrogate.residual
    assert lin.value == pytest.approx(0.5 * float(np.vdot(residual, residual).real))
    assert lin.gradient is not None
    np.testing.assert_allclose(lin.gradient.values, surrogate.gradient, rtol=1e-12)
    assert lin.gradient.space.equivalent(lin.space)
    assert lin.report == {"surface": pytest.approx(lin.value)}
    assert len(lin.reports) == 2 and lin.frequencies == FREQUENCIES
    assert lin.point == problem.vector()
    assert lin.state is problem.state
    assert lin.registry_fingerprint == surrogate.control_registry_fingerprint
    assert lin.fingerprint.startswith("sha256:")
    assert lin.support.all() and lin.support_masks.keys() == set(ACTIVE)
    assert lin.job.action == "linearize" and lin.job.active == ACTIVE
    # a material-only space authors the point in the working simulation
    # instead of staging a ``controls.state`` file
    assert lin.job.control_state is None
    installed = {
        prop.id: np.asarray(prop.control.coefficients)
        for subdomain in problem.simulation.model.subdomains
        for prop in subdomain.properties.values()
        if isinstance(prop, ParameterizedProperty)
    }
    assert {name: values.size for name, values in installed.items()} == {
        "vp": 5,
        "rho": 3,
    }
    np.testing.assert_allclose(
        np.concatenate([installed["vp"], installed["rho"]]),
        np.linspace(-1.0, 1.0, 8),
    )
    assert lin.job.misfit.to_fs()["objective_terms"][0]["receiver_group"] == "surface"
    assert lin.entry.state_fingerprint == lin.state_fingerprint
    assert isinstance(lin.jacobian, Jacobian) and lin.jacobian is lin.jacobian
    assert isinstance(lin.normal, Normal) and lin.normal is lin.normal
    assert "gradient=True" in repr(lin)

    assert problem.value() == pytest.approx(lin.value)
    assert problem.gradient() is lin.gradient
    assert problem.jacobian() is lin.jacobian
    assert problem.normal() is lin.normal
    # the moved state needed one single-frequency registry-discovery linearize
    # at the authored point first; the authored point over every frequency is
    # a new (value-only, state-less) job
    assert [s["action"] for s in fake.submissions] == ["linearize", "linearize"]
    discovery = problem.linearize(problem.full_space.zeros(), gradient=False)
    assert discovery.job.control_state is None and len(fake.submissions) == 3
    assert discovery.frequencies == FREQUENCIES


def test_linearize_caches_by_fingerprint_and_misses_after_state_updates(setup, fake):
    _sim, problem = setup

    first = problem.linearize()
    assert problem.linearize() is first
    assert problem.linearize(problem.vector()) is first
    assert problem.linearize(np.zeros(8)) is first
    assert problem.value() == first.value
    # two frequencies: one single-frequency registry discovery, then the job
    assert len(fake.submissions) == 2
    assert len(problem.cache.keys()) == 2 and problem.cache.keys()[-1] == (
        first.fingerprint
    )

    second = problem.linearize(problem.vector() + 1.0)
    assert second is not first and len(fake.submissions) == 3
    assert problem.linearize() is first  # the state itself did not move

    problem.state = problem.state.with_update(problem.vector() + 1.0)
    assert problem.linearize() is second  # same point: same fingerprint
    assert len(fake.submissions) == 3

    third = problem.linearize(problem.vector() + 1.0)
    assert len(fake.submissions) == 4
    assert problem.cache.keys() == [second.fingerprint, third.fingerprint]
    assert not first.entry.directory.exists()  # evicted staging directory
    assert third.entry.directory.exists()

    problem.clear_cache()
    assert problem.cache.keys() == []
    assert not third.entry.directory.exists()
    assert problem.linearize() is not second
    assert len(fake.submissions) == 5


def test_value_only_linearizations_are_upgraded_when_a_gradient_is_needed(setup, fake):
    _sim, problem = setup

    value_only = problem.linearize(gradient=False)
    assert value_only.gradient is None
    assert value_only.job.covector is None
    assert (
        value_only.state_fingerprint == _surrogate(fake, value_only).state_fingerprint
    )
    assert problem.value() == value_only.value and len(fake.submissions) == 2

    with_gradient = problem.linearize()
    assert with_gradient is not value_only and with_gradient.gradient is not None
    assert len(fake.submissions) == 3
    assert problem.linearize(gradient=False) is with_gradient
    assert problem.cache.keys()[-1] == with_gradient.fingerprint
    assert len(problem.cache.keys()) == 2  # plus the registry discovery


def test_restricted_views_share_state_and_submit_only_their_tasks(setup, fake):
    _sim, problem = setup
    problem.state = problem.state_from(np.linspace(0.5, 2.0, 8))
    full = problem.linearize()

    stage = problem.restrict(active=["vp"], frequencies=[6.0])
    assert stage.space.blocks == ("model.vp",)
    assert stage.frequencies == [6.0]
    assert stage.state is problem.state
    assert stage.vector().size == 5
    np.testing.assert_array_equal(stage.vector().values, np.linspace(0.5, 2.0, 8)[:5])
    assert stage.data_space.size == problem.data_space.size // 2

    lin = stage.linearize()
    assert lin.job.n_tasks == 1 and lin.job.active == ["model.vp"]
    assert len(lin.reports) == 1
    surrogate = _surrogate(fake, lin)
    rows = surrogate.rows(6.0)
    residual = surrogate.J[rows] @ surrogate.m - surrogate.d[rows]
    assert lin.value == pytest.approx(0.5 * float(np.vdot(residual, residual).real))
    assert lin.gradient.size == 5
    np.testing.assert_allclose(
        lin.gradient.values, np.real(surrogate.J[rows].conj().T @ residual)
    )
    assert stage.gradient().size == 5

    # the stage writes back into the shared state
    problem.state = problem.state.with_update(stage.vector() - stage.gradient())
    np.testing.assert_array_equal(
        problem.vector().values[5:], np.linspace(0.5, 2.0, 8)[5:]
    )
    assert problem.linearize() is not full
    assert stage.linearize() is not lin

    with pytest.raises(ValueError, match="not one of"):
        problem.restrict(frequencies=[5.0])
    with pytest.raises(ValueError, match="twice"):
        problem.restrict(frequencies=[4.0, 4.0])
    with pytest.raises(KeyError):
        problem.restrict(active=["qp"])
    assert problem.restrict().space.blocks == tuple(ACTIVE)


def test_smoothing_reads_the_smoothed_aggregate_covector(tmp_path, fake):
    _sim, problem = _problem(tmp_path, fake, smoothing={"type": "tv", "lambda": 0.3})

    assert problem.smoothing is not None and problem.smoothing.kind == "tv"
    lin = problem.linearize()

    assert lin.job.requires_postprocess()
    assert lin.job.control_active == ["vp", "rho"]  # unqualified model blocks
    assert lin.job.smoothing.kind == "tv"
    assert (
        lin.job.covector_file().is_file() and lin.job.covector_file(raw=True).is_file()
    )
    smoothed = ControlVectorFile.read(lin.job.covector_file()).pack(ACTIVE)
    np.testing.assert_allclose(lin.gradient.values, smoothed)
    np.testing.assert_allclose(lin.gradient.values, _surrogate(fake, lin).gradient)
    plan = problem.dry_run()
    assert plan["requires_postprocess"]
    assert plan["job"]["control_sensitivities"]["Smoothing"]["type"] == "tv"

    value_only = problem.restrict(frequencies=[4.0]).linearize(gradient=False)
    assert not value_only.job.requires_postprocess()


def test_frozen_dofs_are_dropped_from_vectors_and_expanded_to_sauce_layout(
    tmp_path,
):
    fake = FakeImagingSite(seed=3, support_masks={"model.vp": [1, 0, 1, 1, 0]})
    _sim, problem = _problem(tmp_path, fake, min_support=0.05)
    problem.state = problem.state_from(np.arange(1.0, 9.0))
    assert problem.space.size == 8  # no mask before the first linearize

    lin = problem.linearize()

    assert lin.job.min_support == 0.05
    assert lin.job.to_fs()["fwi_operator"]["controls"]["min_support"] == 0.05
    assert lin.job.state_output is None  # exports belong to the discovery job
    assert lin.space.size == 6 and lin.space.full_size == 8
    assert lin.space.min_support == 0.05
    np.testing.assert_array_equal(lin.support["vp"], [1, 0, 1, 1, 0])
    assert lin.support.frozen_count == 2
    assert problem.space.size == 6  # the view adopted the mask
    assert problem.vector().size == 6
    np.testing.assert_array_equal(problem.vector().values, [1, 3, 4, 6, 7, 8])
    surrogate = fake.linearizations[lin.state_fingerprint]
    np.testing.assert_allclose(
        lin.gradient.values, surrogate.gradient[[0, 2, 3, 5, 6, 7]]
    )

    # expansion on the way in, drop on the way out
    dv = lin.space.random(1)
    expanded = lin.space.to_sauce_vector(dv)
    np.testing.assert_array_equal(expanded[[1, 4]], 0.0)
    J = lin.jacobian
    np.testing.assert_allclose((J @ dv).values, surrogate.J @ expanded, rtol=1e-12)
    direction = ControlVectorFile.read(
        lin.entry.directory / "ops" / "0001" / "direction_1.h5"
    )
    np.testing.assert_array_equal(direction.pack(ACTIVE), expanded)
    np.testing.assert_array_equal(direction.support_mask("model.vp"), [1, 0, 1, 1, 0])
    r = lin.data_space.random(2)
    jh_r = J.H @ r
    assert jh_r.size == 6
    np.testing.assert_allclose(
        jh_r.values, np.real(surrogate.J.conj().T @ r.values)[[0, 2, 3, 5, 6, 7]]
    )

    # frozen values survive updates; restricted views inherit the mask
    problem.state = problem.state.with_update(problem.vector() * 0.0)
    np.testing.assert_array_equal(problem.state.values, [0, 2, 0, 0, 5, 0, 0, 0])
    assert problem.restrict(active=["vp"]).space.size == 3
    assert problem.restrict(active=["rho"]).space.size == 3
    assert problem.check(taylor=False)["passed"]


def test_geometric_support_is_the_fallback_when_sauce_supplies_no_masks(
    tmp_path, fake, monkeypatch
):
    sim = layered_simulation(tmp_path / "project")
    # A profile authored on z in [0, 400] over the 200..1500 m sediment layer:
    # the node at z = 0 has no support inside the layer.
    controls = DepthProfile("vp", "sediment", datum="global", nodes=[0.0, 200.0, 400.0])
    problem = ImagingProblem(
        sim,
        controls=controls,
        observed={"surface": "observed.h5"},
        frequencies=[4.0],
        site=fake,
    )
    monkeypatch.setattr(ImagingProblem, "_read_masks", lambda self, job, space: None)

    lin = problem.linearize()

    np.testing.assert_array_equal(lin.support["vp"], [False, True, True])
    assert lin.space.size == 2 and lin.gradient.size == 2


# ---------------------------------------------------------------------------
# forward, residual, simulation_at
# ---------------------------------------------------------------------------


def test_forward_residual_and_observed_use_the_site_hooks(setup, fake):
    _sim, problem = setup
    problem.state = problem.state_from(np.linspace(-2.0, 2.0, 8))
    lin = problem.linearize()
    surrogate = _surrogate(fake, lin)

    forward = problem.forward()
    observed = problem.observed_vector()
    residual = problem.residual()
    assert isinstance(forward, DataVector) and forward.space == problem.data_space
    np.testing.assert_allclose(forward.values, surrogate.J @ surrogate.m)
    np.testing.assert_allclose(observed.values, surrogate.d)
    np.testing.assert_allclose(residual.values, surrogate.residual)
    assert residual.dot(residual) == pytest.approx(2.0 * lin.value)
    assert len(fake.submissions) == 2  # discovery + linearize; hooks never submit

    stage = problem.restrict(frequencies=[4.0])
    rows = surrogate.rows(4.0)
    np.testing.assert_allclose(stage.residual().values, surrogate.residual[rows])


def test_simulation_at_installs_material_coefficients(setup):
    sim, problem = setup
    values = np.arange(1.0, 9.0)

    installed = problem.simulation_at(values)

    sediment = next(s for s in installed.model.subdomains if s.name == "sediment")
    np.testing.assert_array_equal(
        sediment.properties["vp"].control.coefficients, values[:5]
    )
    np.testing.assert_array_equal(
        sediment.properties["rho"].control.coefficients, values[5:]
    )
    bound = next(s for s in problem.simulation.model.subdomains if s.name == "sediment")
    np.testing.assert_array_equal(bound.properties["vp"].control.coefficients, 0.0)
    assert installed.name == problem.simulation.name
    assert not isinstance(
        next(s for s in sim.model.subdomains if s.name == "sediment").properties["vp"],
        ParameterizedProperty,
    )


def _source_problem(tmp_path, fake, **source):
    sim = layered_simulation(tmp_path / "sources")
    controls = ControlSpace(
        vp=DepthProfile("vp", "sediment", count=4),
        src=SourceParameters(**source),
    )
    problem = ImagingProblem(
        sim,
        controls=controls,
        observed={"surface": tmp_path / "observed.h5"},
        frequencies=FREQUENCIES,
        site=fake,
        name="src",
    )
    return sim, problem


def _source_coordinates(simulation):
    return simulation.acquisition.source_point_coords()


def test_simulation_at_skips_unchanged_source_blocks(tmp_path, fake):
    sim, problem = _source_problem(tmp_path, fake, signature=True)
    assert problem.space.blocks == (
        "model.vp",
        "source.1.signature",
        "source.2.signature",
    )

    authored = problem.simulation_at()
    sediment = next(s for s in authored.model.subdomains if s.name == "sediment")
    np.testing.assert_array_equal(sediment.properties["vp"].control.coefficients, 0.0)

    # a material update installs vp and leaves the (authored) signatures alone
    update = problem.vector().values.copy()
    update[:4] = [0.1, 0.2, 0.3, 0.4]
    moved = problem.simulation_at(update)
    sediment = next(s for s in moved.model.subdomains if s.name == "sediment")
    np.testing.assert_allclose(
        sediment.properties["vp"].control.coefficients, [0.1, 0.2, 0.3, 0.4]
    )
    np.testing.assert_array_equal(
        _source_coordinates(moved), _source_coordinates(problem.simulation)
    )
    # a restricted material view installs the full state the same way
    view = problem.restrict(active=["vp"])
    installed = view.simulation_at(np.full(4, 0.5))
    sediment = next(s for s in installed.model.subdomains if s.name == "sediment")
    np.testing.assert_allclose(sediment.properties["vp"].control.coefficients, 0.5)


def test_simulation_at_installs_a_changed_source_signature(tmp_path, fake):
    _sim, problem = _source_problem(tmp_path, fake, signature=True)
    assert problem.simulation.acquisition.source_encoding is None  # identity
    q = 0.5 + 0.25j
    state = ControlState.from_blocks(
        problem.full_space,
        {
            "model.vp": np.zeros(4),
            "source.1.signature": np.array([1.0 + 0.0j]),
            "source.2.signature": np.array([q]),
        },
    )

    installed = problem.simulation_at(state)

    # C = E diag(q): the identity encoding is made explicit (one field per
    # source, named after it) and source 2's column carries q
    encoding = installed.acquisition.source_encoding
    assert encoding.encoding_type == "JsonDense"
    assert encoding.field_names() == installed.acquisition.source_point_names()
    np.testing.assert_allclose(encoding.weights, np.diag([1.0, q]))
    payload = installed.acquisition.to_fs()["source_encoding"]
    assert payload["_type"] == "JsonDense"
    assert problem.simulation.acquisition.source_encoding is None  # untouched

    # an authored dense (or conjugated) encoding is scaled column-wise
    authored = np.array([[1.0, 2.0j], [0.5, -1.0]])
    problem.simulation.acquisition.encode_sources(authored)
    scaled = problem.simulation_at(state).acquisition.source_encoding
    np.testing.assert_allclose(scaled.weights, authored * np.array([1.0, q]))
    problem.simulation.acquisition.encode_sources(authored, conjugate=True)
    scaled = problem.simulation_at(state).acquisition.source_encoding
    np.testing.assert_allclose(
        np.conj(scaled.weights), np.conj(authored) * np.array([1.0, q])
    )
    problem.simulation.acquisition.source_encoding = None


def test_simulation_at_scales_named_encoding_terms(tmp_path, fake):
    from frequensolve.seismic.sources import SourceEncoding

    _sim, problem = _source_problem(tmp_path, fake, signature=True)
    names = problem.simulation.acquisition.source_point_names()
    problem.simulation.acquisition.source_encoding = SourceEncoding.named(
        {"shot": {names[0]: 1.0, names[1]: 2.0}, "other": {names[0]: 1.0}}
    )
    state = problem.state_from(np.concatenate([np.zeros(4), [1.0, 0.0, 0.0, -1.0]]))

    installed = problem.simulation_at(state)

    fields = {f.name: f.terms for f in installed.acquisition.source_encoding.fields}
    assert complex(fields["shot"][names[1]]) == pytest.approx(-2.0j)
    assert complex(fields["shot"][names[0]]) == pytest.approx(1.0)
    assert fields["other"] == {names[0]: 1.0}


def _mechanism_state(problem, tmp_path, mechanism, *, scaling=None, units=None):
    """Write a synthetic Sauce state export (with /scaling) and read it back."""

    blocks = {
        "model.vp": np.zeros(4),
        "source.1.mechanism": np.array([1.0 + 0.0j]),
        "source.2.mechanism": np.asarray(mechanism, dtype=complex),
    }
    file = ControlStateFile(
        blocks,
        scaling=scaling or {},
        scaling_units=units or {},
    )
    path = file.write(tmp_path / "state_output.h5")
    return ControlState.load(path, problem.full_space.without_support())


def test_simulation_at_installs_a_scaled_mechanism(tmp_path, fake):
    _sim, problem = _source_problem(tmp_path, fake, mechanism=True, signature=False)
    scaling = {"source.1.mechanism": 1.0e9, "source.2.mechanism": 2.5e8}
    units = {"source.1.mechanism": "N*m", "source.2.mechanism": "N*m"}
    state = _mechanism_state(problem, tmp_path, [4.0], scaling=scaling, units=units)
    assert state.scaling == scaling and state.scaling_units == units
    # scaling travels with updates and file round trips
    moved = state.with_update(problem.space, state.vector(problem.space).values)
    assert moved.scaling == scaling
    again = ControlState.load(
        state.save(tmp_path / "again.h5"), problem.full_space.without_support()
    )
    assert again.scaling == scaling and again.scaling_units == units

    installed = problem.simulation_at(state)

    points = installed.acquisition.source_geometry.sources
    # physical scalar strength = coordinate * scaling in the scaling units
    assert points[0].amplitude == {"value": 1.0e9, "units": "N*m"}
    assert points[1].amplitude == {"value": 1.0e9, "units": "N*m"}
    assert installed.acquisition.source_encoding is None  # real: no phase
    exported = installed.acquisition.to_fs()["source_geometry"]["sources"][1]
    assert exported["amplitude"] == {"value": 1.0e9, "units": "N*m"}

    # a complex coordinate installs its magnitude and applies the phase
    # through the encoding like a signature
    phased = _mechanism_state(problem, tmp_path, [2.0j], scaling=scaling, units=units)
    installed = problem.simulation_at(phased)
    assert installed.acquisition.source_geometry.sources[1].amplitude["value"] == (
        pytest.approx(5.0e8)
    )
    np.testing.assert_allclose(
        installed.acquisition.source_encoding.weights, np.diag([1.0, 1.0j])
    )


@pytest.mark.parametrize(
    "kind, dimension, components, expected",
    [
        ("vector", 2, [3.0, 4.0], {"direction": [0.6, 0.8], "amplitude": 10.0}),
        ("dipole", 3, [0.0, 0.0, -2.0], {"direction": [0, 0, -1.0], "amplitude": 4.0}),
        (
            "tensor",
            2,
            [1.0, -1.0, 0.5],
            {"tensor": [[2.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -2.0]]},
        ),
        (
            "tensor",
            3,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            {"tensor": [[2.0, 12.0, 10.0], [12.0, 4.0, 8.0], [10.0, 8.0, 6.0]]},
        ),
    ],
)
def test_mechanism_components_map_to_the_source_basis(
    kind, dimension, components, expected
):
    from types import SimpleNamespace

    from frequensolve.imaging.controls import ResolvedBlock
    from frequensolve.imaging.problem import _install_source_mechanism
    from frequensolve.seismic.acquisition import Acquisition
    from frequensolve.seismic.sources import SourceGeometry

    coords = [[0.0] * dimension, [1.0] * dimension]
    geometry = SourceGeometry.points(kind=kind, coords=coords)
    geometry.sources[1].amplitude = 7.0  # replaced (tensor: dropped)
    simulation = SimpleNamespace(
        dimension=dimension, acquisition=Acquisition(source_geometry=geometry)
    )
    values = np.zeros(2 * len(components))
    values[0::2] = components
    block = ResolvedBlock(
        name="source.2.mechanism",
        key="src",
        address="src.mechanism",
        size=values.size,
        complex=True,
        kind="source",
        source_id=2,
        quantity="mechanism",
    )

    phase = _install_source_mechanism(simulation, block, values, 2.0, None)

    point = geometry.sources[1]
    assert phase == pytest.approx(1.0)
    if "tensor" in expected:
        assert point.mechanism == {
            "type": "moment_tensor",
            "tensor": expected["tensor"],
            "units": "N*m",
        }
        assert point.amplitude is None
    else:
        np.testing.assert_allclose(point.direction, expected["direction"])
        units = "N" if kind == "vector" else "N*m"
        assert point.amplitude == {"value": expected["amplitude"], "units": units}
    assert geometry.sources[0].amplitude is None  # other sources untouched
    # mixed phases have no real representation
    mixed = values.copy()
    mixed[1] = 1.0
    with pytest.raises(NotImplementedError, match="different phases"):
        _install_source_mechanism(simulation, block, mixed, 2.0, None)


def test_simulation_at_rejects_a_mechanism_without_scaling(tmp_path, fake):
    _sim, problem = _source_problem(tmp_path, fake, mechanism=True, signature=False)
    state = _mechanism_state(problem, tmp_path, [4.0])
    assert state.scaling == {}
    with pytest.raises(NotImplementedError, match=r"'source\.1\.mechanism'.*scaling"):
        problem.simulation_at(state)
    # Sauce's registry baseline supplies the scaling when the state has none
    problem._shared.baseline = ControlStateFile(
        state.blocks(),
        scaling={"source.1.mechanism": 3.0, "source.2.mechanism": 3.0},
        scaling_units={"source.1.mechanism": "N*m", "source.2.mechanism": "N*m"},
    )
    installed = problem.simulation_at(state)
    assert installed.acquisition.source_geometry.sources[1].amplitude == {
        "value": 12.0,
        "units": "N*m",
    }


def test_staged_states_carry_the_mechanism_scaling(tmp_path):
    fake = FakeImagingSite({"source.1.mechanism": 2, "source.2.mechanism": 2}, seed=7)
    _sim, problem = _source_problem(tmp_path, fake, mechanism=True, signature=False)
    problem.linearize(gradient=False)  # learns the registry baseline
    shared = problem._shared
    names = [n for n in shared.baseline.names if n.endswith(".mechanism")]
    shared.baseline = ControlStateFile(
        shared.baseline.blocks,
        scaling={name: 2.0 for name in names},
        scaling_units={name: "N*m" for name in names},
    )
    state = problem.state_from(problem.vector().values + 0.1)
    _stage, path = problem._stage_state("sha256:" + "ab" * 32, state)
    staged = ControlStateFile.read(path)
    # the state carries no scaling of its own: its mechanism blocks are read
    # in task coordinates, so no baseline scaling is attached to them
    assert staged.scaling == {}
    scaled = ControlState(
        state.space,
        state.values,
        scaling={name: 5.0 for name in names},
        scaling_units={name: "N" for name in names},
    )
    _stage, path = problem._stage_state("sha256:" + "cd" * 32, scaled)
    staged = ControlStateFile.read(path)
    assert staged.scaling == {name: 5.0 for name in names}
    assert staged.scaling_units == {name: "N" for name in names}


def test_simulation_at_rejects_a_changed_signature_derivative(tmp_path, fake):
    _sim, problem = _source_problem(tmp_path, fake, signature=False, signature_df=True)
    values = problem.vector().values.copy()
    values[-1] = 0.3
    with pytest.raises(NotImplementedError, match=r"'source\.2\.signature_df'"):
        problem.simulation_at(values)


def test_simulation_at_moves_changed_source_positions(tmp_path, fake):
    _sim, problem = _source_problem(tmp_path, fake, position=True, signature=False)
    authored = _source_coordinates(problem.simulation)
    np.testing.assert_array_equal(authored, [[1000.0, 10.0], [2000.0, 10.0]])
    state = ControlState.from_blocks(
        problem.full_space,
        {
            "model.vp": np.full(4, 0.2),
            "source.1.position": authored[0],
            "source.2.position": np.array([2100.0, 15.0]),
        },
    )

    moved = problem.simulation_at(state)

    np.testing.assert_array_equal(
        _source_coordinates(moved), [[1000.0, 10.0], [2100.0, 15.0]]
    )
    np.testing.assert_array_equal(_source_coordinates(problem.simulation), authored)
    sediment = next(s for s in moved.model.subdomains if s.name == "sediment")
    np.testing.assert_allclose(sediment.properties["vp"].control.coefficients, 0.2)


# ---------------------------------------------------------------------------
# dry run and checks
# ---------------------------------------------------------------------------


def test_dry_run_describes_the_linearize_job_without_submitting(setup, fake):
    _sim, problem = setup

    plan = problem.dry_run()

    assert fake.submissions == []
    assert plan["action"] == "linearize" and plan["active"] == ACTIVE
    assert plan["n_tasks"] == 2 and plan["frequencies"] == FREQUENCIES
    assert plan["job"]["fwi_operator"]["action"] == "linearize"
    assert plan["job"]["fwi_operator"]["controls"]["active"] == ACTIVE
    assert plan["job"]["Imaging"]["misfit"]["objective_terms"][0]["objective"] == {
        "kind": "l2"
    }
    assert plan["outputs"]["covector"][0].endswith("gradient_1.h5")
    json.dumps(plan)
    # two frequencies: the registry is discovered by a single-frequency job
    assert plan["registry_discovery"]
    assert "state" not in plan["job"]["fwi_operator"]["controls"]
    moved = problem.dry_run(np.ones(8))
    assert moved["registry_discovery"] and fake.submissions == []
    assert plan["fingerprint"] == problem.linearize().fingerprint
    assert plan["fingerprint"] != moved["fingerprint"]
    moved = problem.dry_run(np.ones(8))
    assert not moved["registry_discovery"]
    assert not problem.dry_run()["registry_discovery"]
    # material-only spaces author the point inline: no ``controls.state``
    assert "state" not in moved["job"]["fwi_operator"]["controls"]


def test_check_reports_passing_adjoint_normal_and_taylor_tests(setup):
    _sim, problem = setup
    problem.state = problem.state_from(np.linspace(0.1, 0.8, 8))

    report = problem.check(seed=5, steps=(1e-1, 5e-2, 2.5e-2))

    assert report["passed"]
    assert report["adjoint"]["passed"] and report["adjoint"]["relative_error"] < 1e-10
    assert report["normal"]["passed"] and report["normal"]["relative_error"] < 1e-10
    assert report["taylor"]["passed"]
    assert report["taylor"]["first_order_rates"][-1] == pytest.approx(2.0, abs=1e-6)
    assert report["fingerprint"] == problem.linearize().fingerprint
    assert problem.check(taylor=False).keys() == {
        "fingerprint",
        "adjoint",
        "normal",
        "passed",
    }
