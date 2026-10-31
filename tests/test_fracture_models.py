"""Fracture representation export and compatibility boundaries."""

import pytest
import xarray as xr

from frequensolve.model.layered import LayeredModel
from frequensolve.model.layered.surfaces import Fracture

REDUCED = [
    "linear_slip",
    "lower_dimensional_fluid",
    "hydraulic",
    "lower_dimensional_viscous_fluid",
]


@pytest.mark.parametrize("modes", [0, 65, 1.5, True, float("nan"), float("inf")])
def test_viscous_fracture_rejects_invalid_aperture_modes(modes):
    with pytest.raises(ValueError, match="aperture_modes"):
        Fracture(
            "f",
            0.5,
            {"value": 0.01},
            fracture_model="lower_dimensional_viscous_fluid",
            constitutive={"aperture_modes": modes},
        )


def test_viscous_fracture_constitutive_roundtrip():
    params = dict(
        fluid_density=1000,
        fluid_bulk_modulus=2.25e9,
        dynamic_viscosity=0.001,
        aperture_modes=12,
    )
    fracture = Fracture(
        "f",
        0.5,
        {"value": 0.01},
        fracture_model="lower_dimensional_viscous_fluid",
        constitutive=params,
    )
    payload = fracture.to_fs()
    assert payload["constitutive"] == params
    assert isinstance(payload["constitutive"]["aperture_modes"], int)
    assert Fracture.from_fs(payload).to_fs() == payload


@pytest.mark.parametrize("termination", ["pml", "pml:impedance", "pml:fixed"])
def test_pml_termination_preserves_profile(termination):
    from frequensolve.mesh.boundary_conditions import BoundaryCondition

    bc = BoundaryCondition(
        boundaries=["x_max"],
        conditions=[termination],
        pml_wavelengths=3,
        pml_reflectivity=1e-4,
        pml_exponent=4,
    )
    payload = bc.to_fs()
    assert payload["pml_wavelengths"] == 3
    assert payload["pml_reflectivity"] == 1e-4
    assert payload["pml_exponent"] == 4
    assert BoundaryCondition.from_fs(payload).to_fs() == payload


def make_model():
    model = LayeredModel(dimension=2, x_limits=[0, 1])
    model.add_surface(name="top", depth=0)
    model.add_layer(name="rock", mesh_block_id=1, properties={"Vp": 2})
    return model


@pytest.mark.parametrize("kind", REDUCED)
def test_reduced_model_roundtrip_has_no_volume_material(kind):
    model = make_model()
    gap = xr.DataArray([-0.01, 0.02, -0.01], dims=["x"], coords={"x": [0, 0.5, 1]})
    fracture = model.add_fracture("f", depth=0.5, gap=gap, fracture_model=kind)
    model.add_surface(name="bottom", depth=1)
    assert fracture.is_reduced
    payload = model.to_fs()
    assert len(payload["subdomains"]) == 1
    surface = payload["surfaces"][1]
    assert surface["fracture_model"] == kind
    assert "mesh_block_id" not in surface
    assert surface["gap"]["data"] == [-0.01, 0.02, -0.01]
    assert LayeredModel.from_fs(payload).to_fs() == payload


@pytest.mark.parametrize("kind", REDUCED)
@pytest.mark.parametrize(
    "extra",
    [
        {"mesh_block_id": 2},
        {"properties": {"Vp": 1.5}},
        {"physics": "acoustic"},
        {"p_enrich": 1},
    ],
)
def test_reduced_rejects_volume_configuration(kind, extra):
    with pytest.raises(ValueError, match="Reduced fractures"):
        Fracture("f", 0.5, {"value": 0.01}, fracture_model=kind, **extra)


def test_resolved_fluid_filled_material_roundtrip():
    model = make_model()
    fracture = model.add_fracture(
        "f",
        depth=0.5,
        gap={"value": 0.01},
        fracture_model="fluid_filled",
        properties={"Vp": 1.5, "Rho": 1},
    )
    model.add_surface(name="bottom", depth=1)
    assert not fracture.is_reduced
    payload = model.to_fs()
    assert payload["subdomains"][1]["physics"] == "acoustic"
    assert (
        payload["surfaces"][1]["mesh_block_id"]
        == payload["subdomains"][1]["mesh_block_id"]
    )
    assert LayeredModel.from_fs(payload).to_fs() == payload


def test_model_selector_validation_and_legacy_default():
    assert "fracture_model" not in Fracture("f", 0.5, {"value": 0.1}).to_fs()
    with pytest.raises(ValueError, match="Unsupported fracture_model"):
        Fracture("f", 0.5, {"value": 0.1}, fracture_model="typo")
    with pytest.raises(ValueError, match="require acoustic"):
        Fracture("f", 0.5, {"value": 0.1}, fracture_model="fluid_filled")
    with pytest.raises(ValueError, match="require acoustic physics"):
        Fracture(
            "f",
            0.5,
            {"value": 0.1},
            fracture_model="fluid_filled",
            mesh_block_id=2,
            physics="elastic",
        )


@pytest.mark.parametrize("kind", REDUCED)
def test_transient_job_rejects_reduced_fracture_without_elasticity(tmp_path, kind):
    import frequensolve as fs

    project = fs.Project(name="p", path=tmp_path / "p")
    sim = project.new_simulation(name="wave", physics="acoustic", dimension=2)
    sim.discretization = fs.Discretization(method="Galerkin")
    model = make_model()
    model.add_fracture("f", depth=0.5, gap={"value": 0.01}, fracture_model=kind)
    model.add_surface(name="bottom", depth=1)
    model.attenuation_model = "none"
    sim.model = model
    with pytest.raises(
        ValueError,
        match="Cartesian 2D/3D or axisymmetric 2D elastic or coupled physics",
    ):
        fs.TransientJob(
            "pulse",
            sim,
            time=fs.TimeIntegration(0, 1, 0.1),
            source=fs.TimeFunction.ricker(10, delay=0.3),
        )


@pytest.mark.parametrize("kind", REDUCED)
def test_reduced_geometry_authoring_allowed_but_frequency_solve_rejected(
    tmp_path, kind
):
    import frequensolve as fs

    project = fs.Project(name="p", path=tmp_path / "p")
    sim = project.new_simulation(name="wave", physics="acoustic", dimension=2)
    model = make_model()
    model.add_fracture("f", depth=0.5, gap={"value": 0.01}, fracture_model=kind)
    model.add_surface(name="bottom", depth=1)
    sim.model = model
    assert not any(
        issue.code.startswith("model.fracture") for issue in sim.validate().errors
    )

    job = fs.FrequencyDomainJob("frequency", sim, f_list=[10])
    issues = [
        issue
        for issue in job.validate().errors
        if issue.code.startswith("model.fracture")
    ]
    assert [issue.code for issue in issues] == ["model.fracture.solve.unsupported"]
    assert issues[0].path == "model.surfaces[1].fracture_model"


def test_reduced_geometry_dimension_preflight(tmp_path):
    import frequensolve as fs

    project = fs.Project(name="p", path=tmp_path / "p")
    sim = project.new_simulation(name="wave", physics="acoustic", dimension=3)
    model = LayeredModel(dimension=3, x_limits=[0, 1], y_limits=[0, 1])
    model.add_surface(name="top", depth=0)
    model.add_layer(name="rock", mesh_block_id=1, properties={"Vp": 2})
    model.add_fracture(
        "f", depth=0.5, gap={"value": 0.01}, fracture_model="linear_slip"
    )
    model.add_surface(name="bottom", depth=1)
    sim.model = model
    assert "model.fracture.dimension.unsupported" not in {
        issue.code for issue in sim.validate().errors
    }


@pytest.mark.parametrize("kind", REDUCED)
@pytest.mark.parametrize(
    "physics, dimension",
    [
        ("elastic", 2),
        ("elastic_axisym", 2),
        ("coupled_axisym", 2),
        ("elastic", 3),
        ("coupled", 3),
    ],
)
@pytest.mark.parametrize("count", [1, 3])
def test_native_transient_fracture_authoring(tmp_path, kind, physics, dimension, count):
    import frequensolve as fs

    project = fs.Project(name="p", path=tmp_path / "p")
    sim = project.new_simulation(name="wave", physics=physics, dimension=dimension)
    sim.discretization = fs.Discretization(method="Galerkin")
    model = make_model()
    if dimension == 3:
        model = LayeredModel(dimension=3, x_limits=[0, 1], y_limits=[0, 1])
        model.add_surface(name="top", depth=0)
        model.add_layer(name="rock", properties={"Vp": 2.5, "Vs": 1.2, "Rho": 2})
    constitutive = {"fluid_density": 1000, "fluid_bulk_modulus": 2.25e9, "leakoff": 0}
    for index in range(count):
        model.add_fracture(
            f"f{index}",
            depth=(index + 1) / (count + 1),
            gap={"value": 1e-5},
            fracture_model=kind,
            constitutive=constitutive,
        )
    model.add_surface(name="bottom", depth=1)
    model.attenuation_model = "none"
    sim.model = model
    sim.mesh.set_adapt(elems_per_wave=2, f_adapt=250)
    job = fs.TransientJob(
        "pulse",
        sim,
        time=fs.TimeIntegration(0, 1, 0.00025),
        source=fs.TimeFunction.ricker(100, delay=0.015),
    )
    job.validate_outputs()
    payload = model.to_fs()
    assert payload["surfaces"][1]["constitutive"] == constitutive
    assert LayeredModel.from_fs(payload).to_fs() == payload


@pytest.mark.parametrize(
    "constitutive",
    [
        {"fluid_density": 0},
        {"fluid_bulk_modulus": -1},
        {"dynamic_viscosity": float("nan")},
        {"leakoff": -1},
        {"normal_stiffness": -1},
        {"tangential_stiffness": True},
        {"typo": 1},
    ],
)
def test_invalid_native_fracture_constitutive(constitutive):
    with pytest.raises(ValueError):
        Fracture(
            "f",
            0.5,
            {"value": 1e-5},
            fracture_model="hydraulic",
            constitutive=constitutive,
        )
