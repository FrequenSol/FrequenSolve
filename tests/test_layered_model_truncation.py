import pytest
import xarray as xr

from frequensolve.model.layered import LayeredModel
from frequensolve.units import ureg as u


def _three_layer_model(*, ordering="top_down"):
    model = LayeredModel(
        dimension=2,
        x_limits=[0.0, 1.0] * u.km,
        ordering=ordering,
    )
    if ordering == "top_down":
        specifications = [
            ("top", 0.0, None),
            ("shallow", 1.0, "upper"),
            ("deep", 2.0, "middle"),
            ("bottom", 3.0, "lower"),
        ]
    else:
        specifications = [
            ("bottom", 3.0, None),
            ("deep", 2.0, "lower"),
            ("shallow", 1.0, "middle"),
            ("top", 0.0, "upper"),
        ]

    first_name, first_depth, _ = specifications[0]
    model.add_surface(name=first_name, depth=first_depth * u.km)
    for surface_name, surface_depth, layer_name in specifications[1:]:
        model.add_layer(
            name=layer_name,
            mesh_block_id={"upper": 1, "middle": 2, "lower": 3}[layer_name],
            properties={"Vp": (1.5 + surface_depth) * u.km / u.s},
        )
        model.add_surface(name=surface_name, depth=surface_depth * u.km)
    return model


def test_truncate_replaces_lower_boundary_and_discards_deeper_geometry():
    model = _three_layer_model()

    truncated = model.truncate(
        name="section_bottom",
        depth=1.5 * u.km,
    )
    cutting_surface = truncated.surfaces[-1]

    assert truncated is not model
    assert cutting_surface.cutting is True
    assert truncated.surface_names == ["top", "shallow", "section_bottom"]
    assert truncated.layer_names == ["upper", "middle"]
    assert [subdomain.mesh_block_id for subdomain in truncated.subdomains] == [1, 2]
    assert truncated.layers[-1].lower is cutting_surface
    assert truncated.z_limits_in("m") == pytest.approx((0.0, 1500.0))
    assert model.surface_names == ["top", "shallow", "deep", "bottom"]
    assert model.layer_names == ["upper", "middle", "lower"]
    assert [subdomain.mesh_block_id for subdomain in model.subdomains] == [1, 2, 3]

    payload = truncated.to_fs()
    assert payload["surfaces"][-1] == {
        "name": "section_bottom",
        "interface": True,
        "cutting": True,
        "depth": {"value": 1.5, "units": "km"},
    }
    assert LayeredModel.from_fs(payload).to_fs() == payload


def test_truncate_keeps_a_surface_that_crosses_the_cut():
    x = xr.DataArray([0.0, 1.0], dims=["x"], attrs={"units": "km"})
    crossing = xr.DataArray(
        [1.0, 2.0],
        dims=["x"],
        coords={"x": x},
        attrs={"units": "km"},
    )
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0] * u.km)
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(name="upper", mesh_block_id=1, properties={"Vp": 2.0})
    model.add_surface(name="crossing", depth=crossing)
    model.add_layer(name="lower", mesh_block_id=2, properties={"Vp": 3.0})
    model.add_surface(name="bottom", depth=3.0 * u.km)

    truncated = model.truncate(depth=1.5 * u.km)

    assert truncated.surface_names == ["top", "crossing", "cutting_surface"]
    assert truncated.layer_names == ["upper", "lower"]
    assert truncated.surfaces["crossing"].interface is True
    assert model.surface_names == ["top", "crossing", "bottom"]


def test_truncate_demotes_a_crossing_old_boundary_to_marker_surface():
    bottom = xr.DataArray(
        [1.0, 2.0],
        dims=["x"],
        coords={"x": [0.0, 1.0]},
    )
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer", mesh_block_id=1, properties={"Vp": 2.0})
    model.add_surface(name="bottom", depth=bottom)

    truncated = model.truncate(depth=1.5)

    assert truncated.surface_names == ["top", "bottom", "cutting_surface"]
    assert truncated.surfaces["bottom"].interface is False
    assert truncated.layers[0].lower is truncated.surfaces["cutting_surface"]
    assert truncated.to_fs()["surfaces"][1]["interface"] is False
    assert model.surface_names == ["top", "bottom"]
    assert model.layers[0].lower is model.surfaces["bottom"]


def test_truncate_preserves_bottom_up_material_order():
    model = _three_layer_model(ordering="bottom_up")

    truncated = model.truncate(name="section_bottom", depth=1.5 * u.km)

    assert truncated.surface_names == ["section_bottom", "shallow", "top"]
    assert truncated.layer_names == ["middle", "upper"]
    assert truncated.layers[0].lower is truncated.surfaces["section_bottom"]
    assert truncated.layers[0].upper is truncated.surfaces["shallow"]
    assert truncated.layers[1].lower is truncated.surfaces["shallow"]
    assert truncated.layers[1].upper is truncated.surfaces["top"]
    assert model.surface_names == ["bottom", "deep", "shallow", "top"]

    payload = truncated.to_fs()
    assert payload["surfaces"][0]["cutting"] is True
    assert LayeredModel.from_fs(payload).to_fs() == payload


@pytest.mark.parametrize(
    ("ordering", "expected_surfaces"),
    [
        ("top_down", ["top", "section_bottom"]),
        ("bottom_up", ["section_bottom", "top"]),
    ],
)
def test_truncate_replaces_a_coincident_internal_surface(ordering, expected_surfaces):
    model = _three_layer_model(ordering=ordering)

    truncated = model.truncate(name="section_bottom", depth=1.0 * u.km)

    assert truncated.surface_names == expected_surfaces
    assert truncated.layer_names == ["upper"]
    assert truncated.surfaces["section_bottom"].cutting is True
    assert truncated.layers[0].lower is truncated.surfaces["section_bottom"]
    assert model.surface_names == (
        ["top", "shallow", "deep", "bottom"]
        if ordering == "top_down"
        else ["bottom", "deep", "shallow", "top"]
    )


@pytest.mark.parametrize(
    ("ordering", "expected_surfaces"),
    [
        ("top_down", ["top", "shallow", "deep", "section_bottom"]),
        ("bottom_up", ["section_bottom", "deep", "shallow", "top"]),
    ],
)
def test_truncate_replaces_a_coincident_lower_boundary(ordering, expected_surfaces):
    model = _three_layer_model(ordering=ordering)

    truncated = model.truncate(name="section_bottom", depth=3.0 * u.km)

    assert truncated.surface_names == expected_surfaces
    assert truncated.layer_names == (
        ["upper", "middle", "lower"]
        if ordering == "top_down"
        else ["lower", "middle", "upper"]
    )
    assert truncated.surfaces["section_bottom"].cutting is True
    boundary_layer = (
        truncated.layers[-1] if ordering == "top_down" else truncated.layers[0]
    )
    assert boundary_layer.lower is truncated.surfaces["section_bottom"]
    assert all(surface.name != "bottom" for surface in truncated.surfaces)


@pytest.mark.parametrize("ordering", ["top_down", "bottom_up"])
def test_truncate_rejects_a_cut_at_the_top_boundary(ordering):
    model = _three_layer_model(ordering=ordering)

    with pytest.raises(ValueError, match="removes the entire model"):
        model.truncate(depth=0.0 * u.km)


def test_truncate_removes_discarded_fracture_material():
    gap = xr.DataArray(
        [0.0, 0.01, 0.0],
        dims=["x"],
        coords={"x": [0.0, 0.5, 1.0]},
    )
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="upper", mesh_block_id=1, properties={"Vp": 2.0})
    model.add_surface(name="interface", depth=1.0)
    model.add_layer(name="middle", mesh_block_id=2, properties={"Vp": 2.5})
    model.add_fracture(
        name="deep_fracture",
        depth=2.0,
        gap=gap,
        mesh_block_id=20,
        properties={"Vp": 1.5},
    )
    model.add_layer(name="lower", mesh_block_id=3, properties={"Vp": 3.0})
    model.add_surface(name="bottom", depth=3.0)

    truncated = model.truncate(depth=1.5)

    assert truncated.surface_names == ["top", "interface", "cutting_surface"]
    assert truncated.layer_names == ["upper", "middle"]
    assert [subdomain.mesh_block_id for subdomain in truncated.subdomains] == [1, 2]
    assert model.surface_names == ["top", "interface", "deep_fracture", "bottom"]
    assert [subdomain.mesh_block_id for subdomain in model.subdomains] == [1, 2, 20, 3]


def test_truncate_rejects_extension_and_uninspectable_surface_depths():
    model = _three_layer_model()
    original_surface_names = model.surface_names

    with pytest.raises(ValueError, match="cannot extend"):
        model.truncate(depth=4.0 * u.km)
    assert model.surface_names == original_surface_names

    model.surfaces["deep"].depth.darr = None
    with pytest.raises(ValueError, match="not materialized"):
        model.truncate(depth=1.5 * u.km)


def test_add_surface_can_label_cutting_role_without_truncating():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer", properties={"Vp": 1.0})
    model.add_surface(name="bottom", depth=1.0, cutting=True)

    payload = model.to_fs()

    assert "cutting" not in payload["surfaces"][0]
    assert payload["surfaces"][1]["cutting"] is True
    assert LayeredModel.from_fs(payload).surfaces[-1].cutting is True
