"""Joint background + coordinate reflectivity on the solver-free fake site.

Covers the ``fwi_operator.reflectivity`` payload path of
:class:`frequensolve.imaging.ReflectivityParameters` (borrowed bases and own
maps), the Sauce capability rules enforced by
:meth:`ImagingProblem.capabilities`, and the joint gradient / Jacobian /
normal / FWI behaviour of a ``model.vp`` + ``reflectivity.ip`` space.
"""

import json

import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging._artifacts import ControlStateFile
from frequensolve.imaging.workflows import FWI, LBFGS, Stage
from frequensolve.simulation import Discretization, SolverConfig
from tests.imaging_fakes import FakeImagingSite, layered_simulation
from tests.test_imaging_jobs import JOB_EXAMPLES, _assert_valid, _shape

pytestmark = pytest.mark.unit

FREQUENCIES = [4.0, 6.0]
VP_COUNT = 5
TIGHT = dict(gradient_tolerance=1e-10, objective_tolerance=0.0, step_tolerance=0.0)


def _reflectivity(**kwargs):
    field = dict(layer=2, axis=2, basis="vp")
    field.update(kwargs)
    return im.ReflectivityParameters(
        "vp_ip", fields=[im.ReflectivityField("ip", **field)]
    )


def _space(**extra):
    return im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", count=VP_COUNT),
        refl=_reflectivity(),
        **extra,
    )


def _example():
    return json.loads((JOB_EXAMPLES / "fwi-operator-reflectivity.json").read_text())


@pytest.fixture
def fake():
    return FakeImagingSite(block_sizes={"reflectivity.ip": VP_COUNT}, seed=3)


def _problem(tmp_path, fake, *, controls=None, subdir="project", **kwargs):
    sim = layered_simulation(tmp_path / subdir)
    options = dict(
        controls=_space() if controls is None else controls,
        observed={"surface": tmp_path / "observed.h5"},
        frequencies=FREQUENCIES,
        site=fake,
        name="fwi",
    )
    options.update(kwargs)
    return sim, im.ImagingProblem(sim, **options)


@pytest.fixture
def setup(tmp_path, fake):
    return _problem(tmp_path, fake)


# ---------------------------------------------------------------------------
# payload path
# ---------------------------------------------------------------------------


def test_bound_space_emits_the_pinned_reflectivity_payload(tmp_path):
    sim = layered_simulation(tmp_path / "project")

    bound = _space().bind(sim)

    expected = _example()["fwi_operator"]["reflectivity"]
    expected["fields"][0]["basis"] = "vp"  # the example borrows ``acoustic_vp``
    assert bound.blocks == ("model.vp", "reflectivity.ip")
    assert bound.sizes == {"model.vp": VP_COUNT, "reflectivity.ip": VP_COUNT}
    assert bound.reflectivity_payload() == expected
    assert bound.controls_payload() == {"active": ["model.vp", "reflectivity.ip"]}
    assert bound.block("refl").kind == "reflectivity"
    assert bound.block("refl.ip").name == "reflectivity.ip"
    assert bound.restrict("vp").reflectivity_payload() is None
    assert bound.restrict("refl").reflectivity_payload() == expected
    np.testing.assert_array_equal(im.ControlState.from_simulation(bound)["refl"], 0.0)
    # the borrowed basis renders on the material block's coordinates
    vp, ip = bound.block("vp"), bound.block("refl")
    assert ip.dims == vp.dims == ("below",)
    np.testing.assert_array_equal(ip.coords["below"], vp.coords["below"])
    assert ip.coordinate_system == vp.coordinate_system == "seabed_below"
    assert ip.transform == "identity" and ip.lower == -np.inf and ip.upper == np.inf


def test_linearize_job_carries_reflectivity_and_validates_against_the_schema(
    setup, fake
):
    _sim, problem = setup

    lin = problem.linearize()

    payload = _assert_valid(lin.job.to_fs())
    op = payload["fwi_operator"]
    example = _example()["fwi_operator"]
    example["reflectivity"]["fields"][0]["basis"] = "vp"
    assert op["controls"]["active"] == ["model.vp", "reflectivity.ip"]
    assert _shape(op["reflectivity"]) == example["reflectivity"]
    assert op["action"] == "linearize" and "extension" not in op
    # every submitted fwi_operator job (discovery, linearize) carried it
    assert len(fake.jobs) == 2
    for job in fake.jobs:
        assert job.to_fs()["fwi_operator"]["reflectivity"] == example["reflectivity"]
        assert job.active == ["model.vp", "reflectivity.ip"]


def test_basis_resolves_user_keys_independently_of_authoring_order(tmp_path):
    sim = layered_simulation(tmp_path / "project")
    space = im.ControlSpace(
        refl=_reflectivity(basis="velocity"),
        velocity=im.DepthProfile("vp", "sediment", count=4, id="sed_vp"),
        lattice=im.GridParameters(["vp", "rho"], "water", spacing=[1000.0, 100.0]),
        rho_refl=im.ReflectivityParameters(
            "vp_ip",
            fields=[
                im.ReflectivityField("rho_w", layer=1, axis=1, basis="lattice.rho")
            ],
        ),
    )

    bound = space.bind(sim)

    assert bound.blocks == (
        "reflectivity.ip",
        "model.sed_vp",
        "model.lattice_vp",
        "model.lattice_rho",
        "reflectivity.rho_w",
    )
    assert bound.sizes["reflectivity.ip"] == 4
    assert bound.sizes["reflectivity.rho_w"] == bound.sizes["model.lattice_rho"]
    fields = bound.reflectivity_payload()["fields"]
    assert [f["basis"] for f in fields] == ["sed_vp", "lattice_rho"]
    assert bound.block("rho_refl").dims == bound.block("lattice.rho").dims
    with pytest.raises(KeyError, match="reflectivity basis 'rho' names no"):
        im.ControlSpace(
            vp=im.DepthProfile("vp", "sediment", count=4),
            refl=_reflectivity(basis="rho"),
        ).bind(sim)
    with pytest.raises(KeyError, match="'lattice'"):
        # a two-property lattice key is ambiguous; address the property
        im.ControlSpace(
            lattice=im.GridParameters(["vp", "rho"], "water", spacing=[1000.0, 100.0]),
            refl=_reflectivity(basis="lattice", layer=1),
        ).bind(sim)


def test_basis_accepts_a_material_control_already_in_the_simulation(tmp_path):
    sim = layered_simulation(tmp_path / "project")
    primed = im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", count=3, id="acoustic_vp")
    )
    primed_sim = primed.bind(sim).simulation

    bound = im.ControlSpace(refl=_reflectivity(basis="acoustic_vp")).bind(primed_sim)

    assert bound.blocks == ("reflectivity.ip",)
    assert bound.sizes == {"reflectivity.ip": 3}
    assert bound.reflectivity_payload() == _example()["fwi_operator"]["reflectivity"]


def test_own_map_depth_profile_is_authored_like_a_material_profile(tmp_path):
    sim = layered_simulation(tmp_path / "project")
    space = im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", count=VP_COUNT),
        refl=im.ReflectivityParameters(
            "vp_ip",
            fields=[
                im.ReflectivityField(
                    "ip",
                    layer=2,
                    axis=2,
                    control=im.DepthProfile("ip", "sediment", count=8),
                ),
                im.ReflectivityField(
                    "vp",
                    layer=2,
                    axis=1,
                    control=im.DepthProfile.bspline(
                        "vp", "sediment", count=6, degree=2
                    ),
                ),
            ],
            workspace_mb=256,
        ),
    )
    assert not space.resolved  # the own maps need the layer extent

    bound = space.bind(sim)

    assert bound.blocks == ("model.vp", "reflectivity.ip", "reflectivity.vp")
    assert bound.sizes == {
        "model.vp": VP_COUNT,
        "reflectivity.ip": 8,
        "reflectivity.vp": 6,
    }
    payload = bound.reflectivity_payload()
    assert payload["workspace_mb"] == 256.0
    hat, bspline = [f["control"] for f in payload["fields"]]
    assert hat["kind"] == "hat" and hat["coordinate_system"] == "seabed_below"
    assert hat["axis"] == "below" and hat["origin"] == 0.0
    assert hat["coefficients"] == [0.0] * 8
    assert np.isclose(hat["spacing"] * 7, 1300.0)  # sediment: 200 .. 1500 m
    assert bspline["kind"] == "bspline" and bspline["degree"] == 2
    assert len(bspline["coefficients"]) == 6
    # the own map is not installed as a material property
    sediment = bound.simulation.model.layers["sediment"]
    assert sediment.properties["vp"].id == "vp"
    assert not hasattr(sediment.properties["rho"], "control")
    ip = bound.block("refl.ip")
    assert ip.dims == ("below",) and ip.coords["below"].size == 8
    assert ip.subdomain == "sediment"
    np.testing.assert_array_equal(
        im.ControlState.from_simulation(bound)["refl"]["reflectivity.ip"], 0.0
    )

    job = im.FWIOperatorJob(
        "own-map",
        bound.simulation,
        [3.0],
        action="linearize",
        active=list(bound.blocks),
        state="state.json",
        covector="gradient.h5",
        reflectivity=payload,
    )
    fields = _assert_valid(job.to_fs())["fwi_operator"]["reflectivity"]["fields"]
    assert [f["name"] for f in fields] == ["ip", "vp"]


def test_own_map_rejects_transforms_limits_and_layer_mismatches(tmp_path):
    sim = layered_simulation(tmp_path / "project")
    with pytest.raises(ValueError, match="no transform"):
        im.ReflectivityField(
            "ip",
            2,
            2,
            control=im.DepthProfile("ip", "sediment", count=4, transform="log"),
        )
    with pytest.raises(ValueError, match="no limits"):
        im.ReflectivityField(
            "ip",
            2,
            2,
            control=im.DepthProfile("ip", "sediment", count=4, limits=(0, 1)),
        )
    with pytest.raises(TypeError, match="hat or B-spline control or a DepthProfile"):
        im.ReflectivityField(
            "ip", 2, 2, control=im.GridParameters("vp", spacing=[1.0, 1.0])
        )
    mismatched = im.ControlSpace(
        refl=im.ReflectivityParameters(
            "vp_ip",
            fields=[
                im.ReflectivityField(
                    "ip",
                    layer=1,
                    axis=2,
                    control=im.DepthProfile("ip", "sediment", count=4),
                )
            ],
        )
    )
    with pytest.raises(
        ValueError, match="layer 1 but its DepthProfile lives in 'sediment'"
    ):
        mismatched.bind(sim)


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


def _rejected(tmp_path, fake, match, *, controls=None, prepare=None, **kwargs):
    sim = layered_simulation(tmp_path / "project", name=f"case_{len(fake.jobs)}")
    if prepare is not None:
        prepare(sim)
        sim.save()
    with pytest.raises(ValueError, match=match):
        im.ImagingProblem(
            sim,
            controls=_space() if controls is None else controls,
            observed={"surface": tmp_path / "observed.h5"},
            frequencies=FREQUENCIES,
            site=fake,
            name="fwi",
            **kwargs,
        )


def test_capabilities_reject_galerkin_relaxed_assembly_and_fast_mode(tmp_path, fake):
    def galerkin(sim):
        sim.discretization = Discretization(method="Galerkin")

    def relaxed(sim):
        sim.solver = SolverConfig(relaxed_assembly=True)

    def fast(sim):
        sim.solver = SolverConfig(mode="fast")

    _rejected(
        tmp_path,
        fake,
        r"Discretization\(method='DPG'\) \(got 'Galerkin'\)",
        prepare=galerkin,
    )
    _rejected(tmp_path, fake, "unrelaxed assembly", prepare=relaxed)
    _rejected(tmp_path, fake, "unrelaxed assembly", prepare=fast)

    def explicit(sim):
        sim.discretization = Discretization(method="DPG", form_execution="compiled")
        sim.solver = SolverConfig(mode="fast", relaxed_assembly=False)

    sim = layered_simulation(tmp_path / "ok")
    explicit(sim)
    sim.save()
    problem = im.ImagingProblem(
        sim,
        controls=_space(),
        observed={"surface": tmp_path / "observed.h5"},
        frequencies=FREQUENCIES,
        site=fake,
        name="fwi",
    )
    assert problem.capabilities()["ok"]


def test_capabilities_reject_unsupported_physics_and_geometry(tmp_path, fake):
    def coupled(sim):
        sim.physics = "coupled"

    def axisymmetric(sim):
        sim._axisymmetric = True

    def two_and_a_half(sim):
        sim.dimension = 2.5

    _rejected(
        tmp_path,
        fake,
        "acoustic or elastic physics \\(simulation uses 'coupled'\\)",
        prepare=coupled,
    )
    _rejected(tmp_path, fake, "axisymmetric", prepare=axisymmetric)
    _rejected(
        tmp_path, fake, r"2D or 3D simulation \(dimension 2.5\)", prepare=two_and_a_half
    )


def test_capabilities_reject_phase_objectives_interfaces_signature_df_and_extension(
    tmp_path, fake
):
    _rejected(
        tmp_path,
        fake,
        r"reflectivity requires waveform comparisons \(misfit uses \['phase_derivative'\]\)",
        misfit=im.Misfit(comparison="phase_derivative"),
    )
    _rejected(
        tmp_path,
        fake,
        "rejects interface \\(geometry\\) controls: model.salt_top",
        controls=_space(salt=im.InterfaceParameters("salt_top")),
    )
    _rejected(
        tmp_path,
        fake,
        "rejects signature_df source blocks: source.1.signature_df",
        controls=_space(src=im.SourceParameters(signature=False, signature_df=True)),
    )
    # ordinary source blocks join the joint registry (Sauce e2e
    # ``joint-reflectivity-sources-acoustic``)
    _sim, problem = _problem(
        tmp_path,
        fake,
        subdir="sources",
        controls=_space(src=im.SourceParameters(signature=True, position=True)),
    )
    report = problem.capabilities()
    assert report["ok"] and "reflectivity" in report["kinds"]
    # an extended problem passes its extension; Sauce rejects the pair
    report = problem.capabilities(extension=object())
    assert report["errors"] == [
        "reflectivity and extension are mutually exclusive in one problem"
    ]
    assert problem.restrict(active=["vp"]).capabilities(extension=object())["ok"]


# ---------------------------------------------------------------------------
# joint linearization
# ---------------------------------------------------------------------------


def test_linearize_returns_both_blocks_and_restrictions_slice_them(setup, fake):
    _sim, problem = setup

    lin = problem.linearize()

    assert lin.space.blocks == ("model.vp", "reflectivity.ip")
    assert lin.gradient.size == 2 * VP_COUNT
    assert np.all(np.isfinite(lin.gradient.values))
    assert lin.gradient["refl"].shape == (VP_COUNT,)
    assert lin.gradient["vp"].shape == (VP_COUNT,)
    surrogate = fake.linearizations[lin.state_fingerprint]
    np.testing.assert_allclose(lin.gradient.values, surrogate.gradient)
    assert "reflectivity.ip" in lin.manifest.names
    baseline = ControlStateFile.read(fake.jobs[0].state_output_file())
    np.testing.assert_array_equal(baseline["reflectivity.ip"], 0.0)

    only = problem.restrict(active=["refl"])
    assert only.space.blocks == ("reflectivity.ip",)
    assert only.vector().size == VP_COUNT
    only_lin = only.linearize()
    grad = only.gradient()
    assert grad.size == VP_COUNT and np.all(np.isfinite(grad.values))
    # the fake seeds one surrogate per active subspace: compare with its own
    np.testing.assert_allclose(
        grad.values, fake.linearizations[only_lin.state_fingerprint].gradient
    )
    background = problem.restrict(active=["vp"])
    assert background.space.blocks == ("model.vp",)
    assert background.vector().size == VP_COUNT
    assert "refl" not in background.space
    assert background.linearize().job.reflectivity is None
    assert only.linearize().job.reflectivity is not None


def test_joint_jacobian_is_adjoint_and_normal_is_symmetric(setup):
    _sim, problem = setup
    lin = problem.linearize()
    J, N = lin.jacobian, lin.normal

    dv = lin.space.random(1)
    r = lin.data_space.random(2)
    jh_r = J.H @ r
    assert isinstance(jh_r, im.ControlVector) and jh_r.size == 2 * VP_COUNT
    assert np.isclose((J @ dv).dot(r), dv.dot(jh_r), rtol=1e-10)
    report = J.dot_test(seed=3, tolerance=1e-10)
    assert report["passed"], report

    a, b = lin.space.random(4), lin.space.random(5)
    assert np.isclose((N @ a).dot(b), (N @ b).dot(a), rtol=1e-10)
    # the cross blocks are populated: a pure reflectivity direction moves vp
    pure = lin.space.zeros()
    pure = lin.space.pack(
        {"model.vp": np.zeros(VP_COUNT), "reflectivity.ip": np.ones(VP_COUNT)}
    )
    assert np.linalg.norm((N @ pure)["vp"]) > 0.0


def test_two_stage_fwi_activates_reflectivity_in_the_second_stage(setup, fake):
    _sim, problem = setup
    stages = [
        Stage([4.0], 4, active=["vp"], name="background"),
        Stage([4.0, 6.0], 30, active=["vp", "refl"], name="joint"),
    ]

    result = FWI(problem, stages, optimizer=LBFGS(**TIGHT)).run()

    assert [s.name for s in result.stages] == ["background", "joint"]
    assert result.stages[0].active == ("model.vp",)
    assert result.stages[1].active == ("model.vp", "reflectivity.ip")
    assert result.stages[0].space.size == VP_COUNT
    assert result.stages[1].space.size == 2 * VP_COUNT
    for stage in result.stages:
        assert stage.final_loss.total < stage.initial_loss.total
    assert np.linalg.norm(result.state["refl"]) > 0.0
    np.testing.assert_allclose(problem.vector().values, result.state.values)
    assert result.stages[0].vector.size == VP_COUNT


def test_state_updates_round_trip_through_a_control_state_file(setup, fake, tmp_path):
    _sim, problem = setup
    problem.linearize()  # learns the registry baseline
    moved = problem.state.with_update(
        problem.vector() + np.linspace(1.0, 2.0, 2 * VP_COUNT)
    )

    path = moved.save(tmp_path / "state.h5")
    reloaded = im.ControlState.load(path, problem.full_space)

    file = ControlStateFile.read(path)
    assert set(file.names) == {"model.vp", "reflectivity.ip"}
    np.testing.assert_allclose(file["reflectivity.ip"], moved["refl"])
    np.testing.assert_allclose(reloaded.values, moved.values)
    np.testing.assert_allclose(
        reloaded["refl"], np.linspace(1.0, 2.0, 2 * VP_COUNT)[VP_COUNT:]
    )

    # a linearization away from the authored point stages the joint state
    problem.state = moved
    lin = problem.linearize()
    staged = ControlStateFile.read(lin.job.control_state)
    np.testing.assert_allclose(staged["reflectivity.ip"], moved["refl"])
    np.testing.assert_allclose(staged["model.vp"], moved["vp"])
    surrogate = fake.linearizations[lin.state_fingerprint]
    np.testing.assert_allclose(surrogate.m, moved.values)


def test_reflectivity_vectors_render_on_the_borrowed_basis_coordinates(setup):
    _sim, problem = setup
    lin = problem.linearize()

    array = lin.gradient.to_xarray("refl")
    dataset = lin.gradient.to_xarray()

    vp = lin.space.block("vp")
    assert array.dims == ("below",)
    np.testing.assert_array_equal(array["below"].values, vp.coords["below"])
    np.testing.assert_allclose(array.values, lin.gradient["refl"])
    assert array.attrs["block"] == "reflectivity.ip"
    assert array.attrs["coordinate_system"] == "seabed_below"
    assert array.attrs["transform"] == "identity"
    assert set(dataset.data_vars) == {"vp", "refl.ip"}  # block addresses
    np.testing.assert_array_equal(
        dataset["refl.ip"]["below"].values, vp.coords["below"]
    )


def test_sauce_reflectivity_support_masks_are_not_adopted(tmp_path):
    # Sauce 5e07624 writes an all-zero reflectivity support bitmask alongside
    # a nonzero covector; the problem must keep every reflectivity DOF.
    fake = FakeImagingSite(
        block_sizes={"reflectivity.ip": VP_COUNT},
        seed=3,
        support_masks={
            "model.vp": [1, 0, 1, 1, 1],
            "reflectivity.ip": [0] * VP_COUNT,
        },
    )
    _sim, problem = _problem(tmp_path, fake, min_support=0.01)

    lin = problem.linearize()

    np.testing.assert_array_equal(lin.support["vp"], [1, 0, 1, 1, 1])
    assert lin.support["refl"].all()
    assert lin.space.size == 4 + VP_COUNT
    assert lin.gradient["refl"].shape == (VP_COUNT,)
    surrogate = fake.linearizations[lin.state_fingerprint]
    np.testing.assert_allclose(lin.gradient["refl"], surrogate.gradient[VP_COUNT:])
    assert np.linalg.norm(lin.gradient["refl"]) > 0.0


def test_capabilities_warn_about_surface_coordinate_reflectivity_maps(tmp_path, fake):
    _sim, below = _problem(tmp_path, fake)  # ``below`` profile: seabed_below
    _sim, global_z = _problem(
        tmp_path,
        fake,
        subdir="global",
        controls=im.ControlSpace(
            vp=im.DepthProfile("vp", "sediment", axis="z", count=VP_COUNT),
            refl=_reflectivity(),
        ),
    )

    warnings = below.capabilities()["warnings"]
    assert len(warnings) == 1 and warnings[0].endswith(": reflectivity.ip")
    assert "Surface-coordinate control map has no evaluation context" in warnings[0]
    assert below.capabilities()["ok"]
    assert global_z.space.block("refl").coordinate_system == "global"
    assert global_z.capabilities()["warnings"] == []
