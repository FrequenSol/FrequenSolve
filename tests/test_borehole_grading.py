"""Behavioral coverage for physical borehole grading and axial EPW controls."""

import json
import warnings

import pytest
from pint.errors import DimensionalityError

import frequensolve as fs

u = fs.ureg


def model_with_well(**kwargs):
    model = fs.LayeredModel(dimension=2, x_limits=fs.Q_([0, 50], "m"))
    model.add_surface(name="top", depth=0 * u.m)
    model.add_layer(
        name="formation", physics="elastic", properties={"Vp": 2600, "Vs": 1400}
    )
    model.add_surface(name="bottom", depth=200 * u.m)
    well = model.add_borehole(
        name="well",
        x=0 * u.m,
        surfaces=[fs.BoreholeSurface("wall", 0.0762 * u.m)],
        layers=[
            fs.BoreholeLayer(
                "fluid",
                outer_surface="wall",
                physics="acoustic",
                properties={"Vp": 1500},
            )
        ],
        **kwargs,
    )
    return model, well


def test_typed_grading_pint_save_and_roundtrip(tmp_path):
    grading = fs.BoreholeGrading(wall_size=10 * u.cm, growth=1.5, axial_factor=1.5)
    assert fs.BoreholeGrading.from_fs(grading.to_fs()).to_fs() == grading.to_fs()
    with warnings.catch_warnings(record=True) as caught:
        model, well = model_with_well(grading=grading)
    assert not caught
    project = fs.Project(name="test", path=tmp_path)
    sim = project.new_simulation("hole", physics="coupled_axisym", dimension=2)
    sim += model
    payload = json.loads(sim.save().read_text())["Model"]
    borehole = payload["boreholes"][0]
    assert borehole["nearfield"] == {"h0": {"value": 10, "units": "cm"}, "growth": 1.5}
    assert borehole["adapt"]["stoneley"]["tangential"]["factor"] == 1.5
    assert (
        fs.LayeredModel.from_fs(model.to_fs()).to_fs()["boreholes"]
        == model.to_fs()["boreholes"]
    )


def test_legacy_physical_keywords_are_recognized():
    with warnings.catch_warnings(record=True) as caught:
        model, _ = model_with_well(near_h0=0.1 * u.m, near_growth=1.5)
    assert not caught
    assert model.to_fs()["boreholes"][0]["nearfield"] == {
        "h0": {"value": 0.1, "units": "m"},
        "growth": 1.5,
    }


def test_set_grading_and_layer_override():
    model, well = model_with_well()
    assert well.set_grading(wall_size=0.1 * u.m, growth=1.5, axial_factor=2) is well
    assert well.to_fs()["adapt"]["stoneley"]["tangential"]["factor"] == 2
    layer = fs.BoreholeLayer(
        "fluid", grading=fs.BoreholeGrading(wall_size=0.02 * u.m, axial_factor=1.25)
    )
    assert layer.extra["nearfield"]["h0"] == {"value": 0.02, "units": "m"}
    assert layer.extra["adapt"]["stoneley"]["tangential"]["factor"] == 1.25


@pytest.mark.parametrize(
    "settings",
    [
        {"wall_size": -0.1 * u.m},
        {"wall_size": float("nan")},
        {"growth": 1},
        {"growth": float("inf")},
        {"axial_factor": 0.5},
        {"axial_factor": float("nan")},
    ],
)
def test_invalid_grading_is_rejected(settings):
    with pytest.raises(ValueError):
        fs.BoreholeGrading(**settings)


def test_wall_size_requires_a_length():
    with pytest.raises(DimensionalityError):
        fs.BoreholeGrading(wall_size=0.1 * u.s)


def test_conflicting_controls_are_rejected():
    with pytest.raises(ValueError, match="legacy"):
        model_with_well(grading=fs.BoreholeGrading(), near_h0=0.1 * u.m)
    _, well = model_with_well(
        grading=fs.BoreholeGrading(axial_factor=1.5),
        adapt={"stoneley": {"tangential": {"factor": 2}}},
    )
    with pytest.raises(ValueError, match="Conflicting"):
        well.to_fs()


def test_set_grading_replaces_imported_axial_factor():
    model, _ = model_with_well(grading=fs.BoreholeGrading(axial_factor=1.5))
    loaded = fs.LayeredModel.from_fs(model.to_fs())
    loaded.boreholes["well"].set_grading(wall_size=0.1 * u.m, axial_factor=2)
    assert (
        loaded.to_fs()["boreholes"][0]["adapt"]["stoneley"]["tangential"]["factor"] == 2
    )


def test_native_quantity_requires_length_units():
    with pytest.raises(DimensionalityError):
        fs.BoreholeGrading(wall_size={"value": 0.1, "units": "s"})


@pytest.mark.parametrize("guided", [True, False])
def test_guided_aspect_controls_save_and_roundtrip(tmp_path, guided):
    project = fs.Project(name="aspect", path=tmp_path)
    sim = project.new_simulation("hole", physics="coupled_axisym", dimension=2)
    sim.mesh.set_adapt(
        order=4,
        elems_per_wave=1.5,
        f_adapt=200,
        aspect_limit=4,
        guided_anisotropy=guided,
    )
    payload = json.loads(sim.save().read_text())["Mesh"]["adapt"]
    assert payload["aspect_limit"] == 4
    assert payload["guided_anisotropy"] is guided
    restored = fs.MeshAdaptor.from_fs(payload)
    assert restored.aspect_limit == 4
    assert restored.guided_anisotropy is guided
    assert restored.to_fs() == payload


@pytest.mark.parametrize("limit", [0.5, float("nan"), float("inf"), True])
def test_invalid_aspect_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="aspect_limit"):
        fs.MeshAdaptor(elems_per_wave=1.5, aspect_limit=limit)


def test_guided_anisotropy_requires_boolean():
    with pytest.raises(TypeError, match="guided_anisotropy"):
        fs.MeshAdaptor(elems_per_wave=1.5, guided_anisotropy="false")
