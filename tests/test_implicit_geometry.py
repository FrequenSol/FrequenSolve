import copy
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource

from frequensolve.model import LayeredModel, ModelBase
from frequensolve.model.implicit_geometry import (
    IMPLICIT_SURFACE_TYPES,
    ImplicitSurface,
    ImplicitSurfaceControl,
    RBFSurface,
    implicit_surface_from_fs,
    is_implicit_surface_payload,
    split_surface_payloads,
)

CONTRACT_ROOT = (
    Path(__file__).parent / "contracts" / "sauce-83c7f06" / "trunk" / "contracts"
)
IMPLICIT_SCHEMA = CONTRACT_ROOT / "inputs" / "fs-implicit-geometry-1" / "schema.json"
MATERIAL_SCHEMA = CONTRACT_ROOT / "inputs" / "fs-material-model-1" / "schema.json"


def _registry() -> Registry:
    registry = Registry()
    for schema_file in list(CONTRACT_ROOT.rglob("schema.json")) + list(
        CONTRACT_ROOT.glob("fragments/*.json")
    ):
        contents = json.loads(schema_file.read_text())
        if "$id" not in contents:
            continue  # a few pinned output schemas are unaddressable fragments
        registry = registry.with_resource(
            contents["$id"], Resource.from_contents(contents)
        )
    return registry


def _validator(schema_path: Path, pointer: str = "") -> Draft202012Validator:
    schema_id = json.loads(schema_path.read_text())["$id"]
    return Draft202012Validator({"$ref": schema_id + pointer}, registry=_registry())


def _surface_validator() -> Draft202012Validator:
    return _validator(IMPLICIT_SCHEMA, "#/$defs/surface")


def _salt_surface(**overrides) -> RBFSurface:
    kwargs = dict(
        name="salt_boundary",
        support_radius=1500.0,
        centers=[[-2400.0, 1800.0], [0.0, 2700.0], [2400.0, 1750.0]],
        coefficients=[-900.0, -1400.0, -850.0],
        bias=700.0,
    )
    kwargs.update(overrides)
    return RBFSurface(**kwargs)


def test_rbf_surface_without_control_matches_implicit_geometry_contract():
    surface = _salt_surface()

    payload = surface.to_fs()
    assert payload == {
        "_type": "rbf_level_set",
        "name": "salt_boundary",
        "kernel": "wendland_c2",
        "support_radius": 1500.0,
        "bias": 700.0,
        "centers": [[-2400.0, 1800.0], [0.0, 2700.0], [2400.0, 1750.0]],
        "coefficients": [-900.0, -1400.0, -850.0],
    }
    _surface_validator().validate(payload)
    _validator(IMPLICIT_SCHEMA).validate(
        {"schema": "fs-implicit-geometry-1", "surfaces": [payload]}
    )
    assert surface.size == 3
    assert surface.ndim == 2
    np.testing.assert_array_equal(surface.coordinates, surface.centers)
    assert surface.control is None
    restored = implicit_surface_from_fs(payload)
    assert isinstance(restored, RBFSurface)
    assert restored.to_fs() == payload


def test_rbf_surface_with_control_matches_implicit_geometry_contract():
    surface = _salt_surface(
        control=ImplicitSurfaceControl(
            "salt_boundary_rbf", maximum_displacement=150.0, feasibility_band=1200.0
        ),
        level_set=False,
    )

    payload = surface.to_fs()
    assert payload["_type"] == "rbf"
    assert payload["control"] == {
        "id": "salt_boundary_rbf",
        "maximum_displacement": 150.0,
        "feasibility_band": 1200.0,
    }
    _surface_validator().validate(payload)

    restored = RBFSurface.from_fs(payload)
    assert restored.level_set is False
    assert restored.control == surface.control
    assert restored.to_fs() == payload

    # Controls may also be authored as plain mappings with Sauce defaults.
    minimal = _salt_surface(control={"id": "salt"})
    assert minimal.to_fs()["control"] == {"id": "salt"}
    _surface_validator().validate(minimal.to_fs())
    with pytest.raises(ValidationError):
        _surface_validator().validate(
            {**minimal.to_fs(), "control": {"id": "salt", "unknown": 1.0}}
        )


def test_rbf_surface_with_coefficients_keeps_geometry_and_control():
    surface = _salt_surface(control={"id": "salt", "maximum_displacement": 10.0})

    updated = surface.with_coefficients([1.0, 2.0, 3.0])

    np.testing.assert_array_equal(updated.coefficients, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(surface.coefficients, [-900.0, -1400.0, -850.0])
    assert updated.control == surface.control
    assert updated.control is not surface.control
    assert updated.to_fs()["coefficients"] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"name": " "}, "non-empty name"),
        ({"support_radius": 0.0}, "support_radius must be positive"),
        ({"centers": [1.0, 2.0]}, r"\(n, 2\) or \(n, 3\)"),
        ({"centers": [[0.0, 0.0, 0.0, 0.0]]}, r"\(n, 2\) or \(n, 3\)"),
        ({"coefficients": [1.0]}, "one value per center"),
        ({"coefficients": [1.0, np.inf, 0.0]}, "finite"),
        ({"coefficients": [1j, 0.0, 0.0]}, "real-valued"),
        ({"bias": np.nan}, "bias must be finite"),
        ({"control": {"id": "a/b"}}, "HDF5-safe"),
        ({"control": {"id": "a", "maximum_displacement": -1.0}}, "must be positive"),
        ({"control": {"id": "a", "feasibility_band": 0.0}}, "must be positive"),
    ],
)
def test_rbf_surface_rejects_invalid_authoring(overrides, message):
    with pytest.raises(ValueError, match=message):
        _salt_surface(**overrides)


def test_rbf_surface_rejects_foreign_kernels_and_types():
    payload = _salt_surface().to_fs()
    with pytest.raises(ValueError, match="kernel"):
        RBFSurface.from_fs({**payload, "kernel": "gaussian"})
    with pytest.raises(ValueError, match="expected an rbf surface"):
        RBFSurface.from_fs({**payload, "_type": "sphere"})
    with pytest.raises(TypeError, match="ImplicitSurfaceControl"):
        _salt_surface(control=object())


def test_generic_implicit_surfaces_round_trip_and_validate():
    sphere = ImplicitSurface("ball", "sphere", c=[0.0, 0.0, 0.0], r=2.0)
    cut = ImplicitSurface(
        "cut", "difference", fields={"a": "ball", "b": "salt_boundary", "k": 0.5}
    )

    for surface in (sphere, cut):
        payload = surface.to_fs()
        _surface_validator().validate(payload)
        restored = implicit_surface_from_fs(payload)
        assert isinstance(restored, ImplicitSurface)
        assert restored.type == surface.type
        assert restored.fields == surface.fields
        assert restored.to_fs() == payload
    assert sphere.to_fs() == {
        "_type": "sphere",
        "name": "ball",
        "c": [0.0, 0.0, 0.0],
        "r": 2.0,
    }
    with pytest.raises(ValueError, match="unsupported implicit surface type"):
        ImplicitSurface("bad", "torus")
    with pytest.raises(ValueError, match="requires _type"):
        ImplicitSurface.from_fs({"name": "no_type"})
    assert IMPLICIT_SURFACE_TYPES == set(
        json.loads(IMPLICIT_SCHEMA.read_text())["$defs"]["surface"]["properties"][
            "_type"
        ]["enum"]
    )


def test_surface_payload_classification_keeps_graph_surfaces():
    graph = {"name": "top", "interface": True, "depth": {"value": 0.0}}
    fracture = {"_type": "Fracture", "name": "f", "depth": 1.0, "gap": 0.1}
    elevation_graph = {"_type": "elevation", "name": "e", "depth": 2.0}
    implicit = {"_type": "sphere", "name": "s", "c": [0, 0], "r": 1}

    assert not is_implicit_surface_payload(graph)
    assert not is_implicit_surface_payload(fracture)
    assert not is_implicit_surface_payload(elevation_graph)
    assert is_implicit_surface_payload(implicit)
    assert is_implicit_surface_payload({"type": "RBF", "name": "legacy_key"})
    assert split_surface_payloads([graph, implicit, fracture]) == (
        [implicit],
        [graph, fracture],
    )
    assert split_surface_payloads(None) == ([], [])


def _layered_model_with_salt() -> LayeredModel:
    model = LayeredModel(name="salt_blend", dimension=2, x_limits=[-5000.0, 5000.0])
    model.add_surface(0.0, name="top")
    model.add_layer(
        name="earth",
        physics="acoustic",
        properties={"vp": 2500.0, "rho": 2200.0},
    )
    model.add_surface(5000.0, name="bottom")
    model += _salt_surface(control={"id": "salt_boundary_rbf"})
    return model


def test_layered_model_exports_implicit_surfaces_after_graph_surfaces():
    model = _layered_model_with_salt()
    model.add_implicit_surface(ImplicitSurface("ball", "sphere", c=[0.0, 0.0], r=1.0))

    payload = model.to_fs()
    names = [surface["name"] for surface in payload["surfaces"]]
    assert names == ["top", "bottom", "salt_boundary", "ball"]
    assert payload["surfaces"][2]["_type"] == "rbf_level_set"
    assert payload["surfaces"][2]["control"] == {"id": "salt_boundary_rbf"}
    assert model.surface_names == ["top", "bottom"]
    assert model.z_limits == (0.0, 5000.0)
    assert [layer.name for layer in model.layers] == ["earth"]
    _validator(MATERIAL_SCHEMA).validate({"schema": "fs-material-model-1", **payload})
    with pytest.raises(ValidationError):
        _validator(MATERIAL_SCHEMA).validate(
            {
                "schema": "fs-material-model-1",
                **payload,
                "surfaces": payload["surfaces"][:2]
                + [{**payload["surfaces"][2], "_type": "spline"}],
            }
        )


def test_layered_model_round_trips_implicit_surfaces_and_layers():
    model = _layered_model_with_salt()
    payload = model.to_fs()

    restored = LayeredModel.from_fs(copy.deepcopy(payload))

    assert restored.surface_names == ["top", "bottom"]
    assert [layer.name for layer in restored.layers] == ["earth"]
    assert restored.layers["earth"].upper.name == "top"
    assert restored.layers["earth"].lower.name == "bottom"
    assert [surface.name for surface in restored.implicit_surfaces] == ["salt_boundary"]
    salt = restored.implicit_surfaces["salt_boundary"]
    assert isinstance(salt, RBFSurface)
    assert salt.control == ImplicitSurfaceControl("salt_boundary_rbf")
    assert restored.to_fs() == payload
    assert "surfaces" not in restored.extra

    # Implicit surfaces listed before graph surfaces are still separated.
    reordered = copy.deepcopy(payload)
    reordered["surfaces"] = reordered["surfaces"][2:] + reordered["surfaces"][:2]
    assert LayeredModel.from_fs(reordered).to_fs() == payload


def test_layered_model_rejects_duplicate_surface_names_across_kinds():
    model = _layered_model_with_salt()

    with pytest.raises(ValueError, match="already used"):
        model += _salt_surface()
    with pytest.raises(ValueError, match="already used"):
        model += ImplicitSurface("top", "sphere", c=[0.0, 0.0], r=1.0)
    with pytest.raises(TypeError, match="implicit surface"):
        model.add_implicit_surface(object())


def test_base_model_round_trips_implicit_surfaces_and_pass_through_surfaces():
    model = ModelBase(
        name="regions",
        dimension=3,
        implicit_surfaces=[
            {
                "_type": "box",
                "name": "outer",
                "center": [0, 0, 0],
                "half_size": [10, 10, 10],
            },
            ImplicitSurface(
                "bore", "cylinder", center=[0, 0, 0], axis=[0, 0, 1], radius=1
            ),
            ImplicitSurface("rock_volume", "difference", a="outer", b="bore"),
        ],
    )

    payload = model.to_fs()
    assert [surface["name"] for surface in payload["surfaces"]] == [
        "outer",
        "bore",
        "rock_volume",
    ]
    _validator(MATERIAL_SCHEMA).validate({"schema": "fs-material-model-1", **payload})
    restored = ModelBase.from_fs(payload)
    assert [surface.type for surface in restored.implicit_surfaces] == [
        "box",
        "cylinder",
        "difference",
    ]
    assert restored.to_fs() == payload

    # Unknown graph surfaces on a base model stay pass-through, ahead of
    # the implicit surfaces, without colliding with the typed key.
    graph = {"name": "top", "depth": {"value": 0.0}}
    mixed = ModelBase.from_fs({**payload, "surfaces": [graph] + payload["surfaces"]})
    assert mixed.extra["surfaces"] == [graph]
    assert mixed.to_fs()["surfaces"] == [graph] + payload["surfaces"]
