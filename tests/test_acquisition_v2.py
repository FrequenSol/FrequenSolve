import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from frequensolve import (
    Acquisition,
    CoordinateValue,
    Direction,
    DistributedSource,
    PointSource,
    SourceEncoding,
    SourceGeometry,
    SourceGroup,
)
from frequensolve.units import ureg

CONTRACT_ROOT = (
    Path(__file__).parent / "contracts" / "sauce-a54bdda" / "trunk" / "contracts"
)
ACQUISITION_SCHEMA = CONTRACT_ROOT / "inputs" / "fs-acquisition-2" / "schema.json"


def _sauce_acquisition_validator() -> Draft202012Validator:
    registry = Registry()
    for schema_file in CONTRACT_ROOT.rglob("*.json"):
        contents = json.loads(schema_file.read_text())
        resource = Resource.from_contents(contents)
        registry = registry.with_resource(contents["$id"], resource)
    schema = json.loads(ACQUISITION_SCHEMA.read_text())
    return Draft202012Validator(schema, registry=registry)


def _five_point_four_field_acquisition() -> Acquisition:
    geometry = SourceGeometry.points(
        kind="scalar",
        coords=[
            [0.25, 0.05],
            [0.50, 0.05],
            [0.75, 0.05],
            [0.45, 0.08],
            [0.55, 0.08],
        ],
        names=["shot_left", "shot_center", "shot_right", "pair_pos", "pair_neg"],
    )
    encoding = SourceEncoding.named(
        [
            DistributedSource.named("shot_left", {"shot_left": 1.0}),
            DistributedSource.named("shot_center", {"shot_center": 1.0}),
            DistributedSource.named("shot_right", {"shot_right": 1.0}),
            DistributedSource.named("difference", {"pair_pos": 1.0, "pair_neg": -1.0}),
        ]
    )
    return Acquisition(source_geometry=geometry, source_encoding=encoding)


def test_inline_identity_geometry_matches_pinned_sauce_schema_and_roundtrips():
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar",
            coords=[[0.25, 0.05], [0.75, 0.05]],
            names=["left", "right"],
        )
    )

    payload = acquisition.to_fs()

    _sauce_acquisition_validator().validate(payload)
    assert payload["schema"] == "fs-acquisition-2"
    assert "source_groups" not in payload
    assert "source_encoding" not in payload
    assert acquisition.source_point_count() == 2
    assert acquisition.source_field_count() == 2
    assert Acquisition.from_fs(payload).to_fs() == payload


def test_source_amplitudes_export_dimensionless_and_physical_units():
    scalar = Acquisition()
    scalar.add_sources(
        kind="scalar",
        coords=[[0.25, 0.05]],
        amplitude=2.5,
    )

    assert scalar.to_fs()["source_geometry"]["defaults"]["amplitude"] == 2.5

    vector = Acquisition()
    vector.add_sources(
        kind="vector",
        coords=[[0.5, 0.05]],
        direction=[0.0, 1.0],
        amplitude=20.0 * ureg.kN,
    )

    payload = vector.to_fs()
    assert payload["source_geometry"]["defaults"]["amplitude"] == {
        "value": 20.0,
        "units": "kN",
    }
    assert Acquisition.from_fs(payload).to_fs() == payload


def test_file_source_defaults_serialize_unit_bearing_amplitudes():
    geometry = SourceGeometry.hdf5(
        "sources.h5",
        dataset="source_points",
        kind="scalar",
        defaults={
            "mechanism": "isotropic",
            "amplitude": 1.0e6 * ureg.N * ureg.m,
        },
    )

    payload = geometry.to_fs()

    assert payload["defaults"] == {
        "mechanism": {"type": "isotropic"},
        "amplitude": {"value": 1.0e6, "units": "m*N"},
    }


@pytest.mark.parametrize("geometry_type", ["points", "inline", "hdf5", "sps"])
def test_source_direction_defaults_match_pinned_schema_and_roundtrip(geometry_type):
    direction = Direction.vector(np.asarray([0.0, 1.0]), units="N")
    defaults = {"direction": direction}

    if geometry_type == "points":
        geometry = SourceGeometry.points(
            kind="vector",
            coords=[[0.5, 0.05]],
            defaults=defaults,
        )
    elif geometry_type == "inline":
        geometry = SourceGeometry.inline(
            kind="vector",
            sources=[PointSource(name="shot", coordinates=[0.5, 0.05])],
            defaults=defaults,
        )
    elif geometry_type == "hdf5":
        geometry = SourceGeometry.hdf5(
            "sources.h5",
            dataset="/sources",
            kind="vector",
            defaults=defaults,
        )
    else:
        geometry = SourceGeometry.sps(
            "sources.sps",
            kind="vector",
            defaults=defaults,
        )

    payload = Acquisition(source_geometry=geometry).to_fs()

    assert payload["source_geometry"]["defaults"]["direction"] == {
        "value": [0.0, 1.0],
        "units": "N",
    }
    json.dumps(payload)
    _sauce_acquisition_validator().validate(payload)
    assert Acquisition.from_fs(payload).to_fs() == payload


def test_source_axis_direction_default_matches_pinned_schema():
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="vector",
            coords=[[0.5, 0.05]],
            defaults={"direction": Direction.axis_direction("z")},
        )
    )

    payload = acquisition.to_fs()

    assert payload["source_geometry"]["defaults"]["direction"] == {"direction": "z"}
    _sauce_acquisition_validator().validate(payload)


def test_point_source_axis_direction_roundtrips_from_pinned_schema():
    payload = {
        "schema": "fs-acquisition-2",
        "source_geometry": {
            "_type": "Inline",
            "kind": "vector",
            "sources": [
                {
                    "name": "shot",
                    "coordinates": [0.5, 0.05],
                    "direction": {"direction": "z"},
                }
            ],
        },
        "receiver_groups": [],
    }

    _sauce_acquisition_validator().validate(payload)
    loaded = Acquisition.from_fs(payload)

    assert loaded.to_fs() == payload


@pytest.mark.parametrize(
    "direction, message",
    [
        (
            Direction.vector([0.0, 1.0], system="survey"),
            "cannot include a coordinate system",
        ),
        (
            Direction.basis(["x", "z"]),
            "not supported by fs-acquisition-2",
        ),
    ],
)
def test_unsupported_source_direction_metadata_is_rejected(direction, message):
    with pytest.raises(ValueError, match=message):
        geometry = SourceGeometry.points(
            kind="vector",
            coords=[[0.5, 0.05]],
            defaults={"direction": direction},
        )
        Acquisition(source_geometry=geometry).to_fs()


def test_five_points_and_four_named_fields_match_pinned_sauce_schema():
    acquisition = _five_point_four_field_acquisition()

    payload = acquisition.to_fs()

    _sauce_acquisition_validator().validate(payload)
    assert len(payload["source_geometry"]["sources"]) == 5
    assert len(payload["source_encoding"]["fields"]) == 4
    assert payload["source_encoding"]["fields"][-1]["terms"] == [
        {"source": "pair_pos", "coefficient": 1.0},
        {"source": "pair_neg", "coefficient": -1.0},
    ]
    assert acquisition.source_point_count() == 5
    assert acquisition.source_field_count() == 4
    assert Acquisition.from_fs(payload).to_fs() == payload


def test_json_dense_encoding_roundtrips_against_pinned_sauce_schema():
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar",
            coords=[[0.0, 0.0], [1.0, 0.0]],
            names=["left", "right"],
        ),
        source_encoding=SourceEncoding.dense(
            [[1.0, 1.0], [0.0, -1.0]],
            names=["left_only", "difference"],
        ),
    )

    payload = acquisition.to_fs()

    _sauce_acquisition_validator().validate(payload)
    assert payload["source_encoding"]["_type"] == "JsonDense"
    assert Acquisition.from_fs(payload).to_fs() == payload


def test_encoded_reference_coordinates_preserve_explicit_and_implicit_metadata():
    geometry = SourceGeometry.points(
        kind="scalar",
        coords=[[0.0, 0.05], [1.0, 0.05]],
        names=["left", "right"],
        units="km",
        system="survey",
    )
    explicit = Acquisition(
        source_geometry=geometry,
        source_encoding=SourceEncoding.dense(
            [[1.0], [0.0]],
            names=["explicit"],
            reference_coordinates=CoordinateValue(
                [750.0, 40.0],
                units="m",
                system="survey",
            ),
        ),
    )
    implicit = Acquisition(
        source_geometry=geometry,
        source_encoding=SourceEncoding.named(
            [DistributedSource.named("midpoint", {"left": 1.0, "right": -1.0})]
        ),
    )

    explicit_reference = explicit.source_coords(1, preserve_metadata=True)
    implicit_reference = implicit.source_coords(1, preserve_metadata=True)

    assert isinstance(explicit_reference, CoordinateValue)
    assert explicit_reference.value == [750.0, 40.0]
    assert explicit_reference.units == "m"
    assert explicit_reference.system == "survey"
    assert isinstance(implicit_reference, CoordinateValue)
    assert implicit_reference.value == [0.5, 0.05]
    assert implicit_reference.units == "km"
    assert implicit_reference.system == "survey"
    assert np.allclose(implicit.source_coords(1), [0.5, 0.05])


def test_implicit_encoded_reference_normalizes_compatible_source_units():
    geometry = SourceGeometry.inline(
        kind="scalar",
        sources=[
            PointSource(
                name="left",
                coordinates=CoordinateValue(
                    [0.0, 0.05],
                    units="km",
                    system="survey",
                ),
            ),
            PointSource(
                name="right",
                coordinates=CoordinateValue(
                    [1000.0, 50.0],
                    units="m",
                    system="survey",
                ),
            ),
        ],
    )
    acquisition = Acquisition(
        source_geometry=geometry,
        source_encoding=SourceEncoding.named(
            [DistributedSource.named("midpoint", {"left": 1.0, "right": 1.0})]
        ),
    )

    reference = acquisition.source_coords(1, preserve_metadata=True)

    assert isinstance(reference, CoordinateValue)
    assert reference.units == "km"
    assert reference.system == "survey"
    assert np.allclose(reference.value, [0.5, 0.05])
    assert np.allclose(acquisition.source_coords(1), [0.5, 0.05])


def test_implicit_encoded_reference_rejects_inconsistent_source_systems():
    geometry = SourceGeometry.inline(
        kind="scalar",
        sources=[
            PointSource(
                name="left",
                coordinates=CoordinateValue([0.0, 0.05], units="km"),
            ),
            PointSource(
                name="right",
                coordinates=CoordinateValue(
                    [1.0, 0.05],
                    units="km",
                    system="survey",
                ),
            ),
        ],
    )
    acquisition = Acquisition(
        source_geometry=geometry,
        source_encoding=SourceEncoding.named(
            [DistributedSource.named("midpoint", {"left": 1.0, "right": 1.0})]
        ),
    )

    with pytest.raises(ValueError, match="one coordinate system"):
        acquisition.source_coords(1, preserve_metadata=True)


def test_hdf5_geometry_and_encoding_match_pinned_sauce_schema_and_roundtrip():
    geometry = SourceGeometry.hdf5(
        "inputs/sources.h5",
        dataset="/sources",
        kind="scalar",
        name="catalog",
        domain=2,
        system="model",
        units="m",
    )
    encoding = SourceEncoding.hdf5(
        "inputs/encoding.h5",
        dataset="/coefficients",
        name="encoded_fields",
        field_names_dataset="/field_names",
        reference_coordinates_dataset="/reference_coordinates",
    )
    acquisition = Acquisition(
        source_geometry=geometry,
        source_encoding=encoding,
    )

    payload = acquisition.to_fs()

    _sauce_acquisition_validator().validate(payload)
    assert payload["source_geometry"] == {
        "_type": "HDF5",
        "name": "catalog",
        "domain": 2,
        "kind": "scalar",
        "file": "inputs/sources.h5",
        "dataset": "/sources",
        "system": "model",
        "units": "m",
    }
    assert payload["source_encoding"] == {
        "_type": "HDF5Dense",
        "name": "encoded_fields",
        "file": "inputs/encoding.h5",
        "dataset": "/coefficients",
        "field_names_dataset": "/field_names",
        "reference_coordinates_dataset": "/reference_coordinates",
    }
    assert acquisition.known_source_point_count() is None
    assert acquisition.known_source_field_count() is None
    assert encoding.field_names() == []
    with pytest.raises(ValueError, match="coordinates are external"):
        encoding.reference_coordinates(geometry)
    assert Acquisition.from_fs(payload).to_fs() == payload


@pytest.mark.parametrize(
    "geometry",
    [
        SourceGeometry.hdf5(
            "inputs/sources.h5",
            dataset="/sources",
            kind="scalar",
            count=3,
        ),
        SourceGeometry.sps(
            "inputs/sources.sps",
            kind="scalar",
            count=3,
        ),
    ],
    ids=["hdf5", "sps"],
)
def test_external_source_geometry_count_roundtrips(geometry):
    acquisition = Acquisition(source_geometry=geometry)

    payload = acquisition.to_fs()
    loaded = Acquisition.from_fs(payload)

    assert payload["source_geometry"]["count"] == 3
    assert loaded.known_source_point_count() == 3
    assert loaded.known_source_field_count() == 3
    assert loaded.to_fs() == payload


def test_external_source_encoding_count_roundtrips():
    acquisition = Acquisition(
        source_geometry=SourceGeometry.hdf5(
            "inputs/sources.h5",
            dataset="/sources",
            kind="scalar",
            count=4,
        ),
        source_encoding=SourceEncoding.hdf5(
            "inputs/encoding.h5",
            dataset="/coefficients",
            count=2,
        ),
    )

    payload = acquisition.to_fs()
    loaded = Acquisition.from_fs(payload)

    assert payload["source_geometry"]["count"] == 4
    assert payload["source_encoding"]["count"] == 2
    assert loaded.known_source_point_count() == 4
    assert loaded.known_source_field_count() == 2
    assert loaded.to_fs() == payload


def test_per_point_directions_serialize_on_source_atoms():
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="vector",
            coords=[[0.0, 0.0], [1.0, 0.0]],
            direction=[[1.0, 0.0], [0.0, 1.0]],
        )
    )

    payload = acquisition.to_fs()

    assert "defaults" not in payload["source_geometry"]
    assert [
        source["direction"] for source in payload["source_geometry"]["sources"]
    ] == [[1.0, 0.0], [0.0, 1.0]]
    _sauce_acquisition_validator().validate(payload)


def test_legacy_source_groups_are_migrated_but_never_reexported():
    acquisition = Acquisition.from_fs(
        {
            "source_groups": [
                {
                    "source": {
                        "_type": "PointSource",
                        "name": "shot",
                        "kind": "scalar",
                        "coordinates": [0.5, 0.05],
                    }
                }
            ],
            "receiver_groups": [],
        }
    )

    payload = acquisition.to_fs()

    assert "source_groups" not in payload
    assert payload["source_geometry"]["sources"][0]["name"] == "shot"
    with pytest.warns(DeprecationWarning, match="source_groups"):
        assert acquisition.source_groups[0].source.name == "shot"


def test_tagged_v1_source_groups_are_migrated_to_v2():
    acquisition = Acquisition.from_fs(
        {
            "schema": "fs-acquisition-1",
            "source_groups": [
                {
                    "source": {
                        "_type": "PointSource",
                        "name": "shot",
                        "kind": "scalar",
                        "coordinates": [0.5, 0.05],
                    }
                }
            ],
            "receiver_groups": [],
        }
    )

    payload = acquisition.to_fs()

    _sauce_acquisition_validator().validate(payload)
    assert payload["schema"] == "fs-acquisition-2"
    assert "source_groups" not in payload


def test_legacy_point_source_positional_kind_and_domain_are_preserved():
    default_named = PointSource("scalar", [0.25, 0.05])
    source = PointSource("scalar", [0.5, 0.05], name="shot", domain=7)

    assert default_named.name == "point"
    assert default_named.kind == "scalar"
    assert source.kind == "scalar"
    assert source.name == "shot"
    assert source.coordinates == [0.5, 0.05]
    assert source.domain == 7
    assert source.to_fs()["domain"] == 7

    acquisition = Acquisition(source_groups=[SourceGroup(source=source)])
    payload = acquisition.to_fs()

    assert payload["source_geometry"]["kind"] == "scalar"
    assert payload["source_geometry"]["domain"] == 7
    assert "domain" not in payload["source_geometry"]["sources"][0]
    _sauce_acquisition_validator().validate(payload)
    with pytest.warns(DeprecationWarning, match="source_groups"):
        assert acquisition.source_groups[0].source.domain == 7


def test_source_groups_compatibility_view_rejects_mutation_without_state_loss():
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar", coords=[[0.5, 0.05]], names=["shot"]
        )
    )
    before = acquisition.to_fs()
    replacement = [
        SourceGroup(source=PointSource("scalar", [0.75, 0.05], name="other"))
    ]

    with pytest.warns(DeprecationWarning, match="source_groups"):
        groups = acquisition.source_groups
    with pytest.raises(TypeError, match="read-only compatibility view"):
        groups.append(replacement[0])
    with pytest.warns(DeprecationWarning, match="source_groups"):
        with pytest.raises(TypeError, match="read-only compatibility view"):
            acquisition.source_groups = replacement

    assert acquisition.to_fs() == before


def test_inline_point_kind_must_match_geometry_kind():
    geometry = SourceGeometry.inline(
        kind="scalar",
        sources=[PointSource(name="bad", coordinates=[0.5, 0.05], kind="vector")],
    )

    with pytest.raises(ValueError, match="kind must match"):
        Acquisition(source_geometry=geometry).to_fs()

    normalized_point = PointSource(
        name="normalized", coordinates=[0.5, 0.05], kind="scalar"
    )
    normalized_point.kind = " Scalar "
    normalized = Acquisition(
        source_geometry=SourceGeometry.inline(kind="scalar", sources=[normalized_point])
    ).to_fs()
    assert normalized["source_geometry"]["sources"][0]["kind"] == "scalar"
    _sauce_acquisition_validator().validate(normalized)


def test_legacy_compound_without_direction_omits_empty_direction():
    acquisition = Acquisition.from_fs(
        {
            "schema": "fs-acquisition-1",
            "source_groups": [
                {
                    "source": {
                        "_type": "CompoundSource",
                        "name": "difference",
                        "kind": "scalar",
                        "coordinates": [[0.45, 0.08], [0.55, 0.08]],
                    }
                }
            ],
            "receiver_groups": [],
        }
    )

    payload = acquisition.to_fs()

    assert all(
        "direction" not in point for point in payload["source_geometry"]["sources"]
    )
    _sauce_acquisition_validator().validate(payload)


def test_unnamed_source_fallback_matches_sauce_and_detects_collisions():
    unnamed = SourceGeometry.inline(
        kind="scalar", sources=[PointSource(coordinates=[0.5, 0.05])]
    )
    assert unnamed.point_names() == ["source_000001"]
    _sauce_acquisition_validator().validate(
        Acquisition(source_geometry=unnamed).to_fs()
    )

    collision = SourceGeometry.inline(
        kind="scalar",
        sources=[
            PointSource(coordinates=[0.25, 0.05]),
            PointSource(name="source_000001", coordinates=[0.75, 0.05]),
        ],
    )
    with pytest.raises(ValueError, match="names must be unique"):
        Acquisition(source_geometry=collision).to_fs()


def test_named_encoding_requires_explicit_inline_source_names():
    geometry = SourceGeometry.inline(
        kind="scalar", sources=[PointSource(coordinates=[0.5, 0.05])]
    )
    encoding = SourceEncoding.named(
        [DistributedSource.named("field", {"source_000001": 1.0})]
    )

    with pytest.raises(ValueError, match="requires explicit names"):
        Acquisition(source_geometry=geometry, source_encoding=encoding).to_fs()


def test_acquisition_extra_rejects_legacy_source_groups():
    geometry = SourceGeometry.points(
        kind="scalar", coords=[[0.5, 0.05]], names=["shot"]
    )
    with pytest.raises(ValueError, match="extra cannot contain legacy source_groups"):
        Acquisition(source_geometry=geometry, extra={"source_groups": []})

    acquisition = Acquisition(source_geometry=geometry)
    acquisition.extra["source_groups"] = []
    with pytest.raises(ValueError, match="extra cannot contain legacy source_groups"):
        acquisition.to_fs()


def test_export_rejects_missing_geometry_and_inconsistent_encoding():
    with pytest.raises(ValueError, match="requires source_geometry"):
        Acquisition().to_fs()

    geometry = SourceGeometry.points(
        kind="scalar",
        coords=[[0.0, 0.0]],
        names=["known"],
    )
    unknown_encoding = SourceEncoding.named(
        [DistributedSource.named("bad", {"unknown": 1.0})]
    )
    with pytest.raises(ValueError, match="unknown sources"):
        Acquisition(
            source_geometry=geometry,
            source_encoding=unknown_encoding,
        ).to_fs()

    dense_encoding = SourceEncoding.dense([[1.0]], names=["field"])
    dense_encoding.fields[0].coefficients.append(0.0)
    with pytest.raises(ValueError, match="coefficient count"):
        Acquisition(
            source_geometry=geometry,
            source_encoding=dense_encoding,
        ).to_fs()


def test_compound_source_adapter_emits_named_encoding():
    acquisition = Acquisition()
    with pytest.warns(DeprecationWarning, match="add_compound_source"):
        acquisition.add_compound_source(
            kind="scalar",
            coords=[[0.45, 0.08], [0.55, 0.08]],
            weights=[1.0, -1.0],
        )

    payload = acquisition.to_fs()

    assert len(payload["source_geometry"]["sources"]) == 2
    assert payload["source_encoding"]["fields"][0]["terms"] == [
        {"source": "source_0_point_001", "coefficient": 1.0},
        {"source": "source_0_point_002", "coefficient": -1.0},
    ]
    _sauce_acquisition_validator().validate(payload)


def test_deprecated_helpers_preserve_zero_based_logical_source_names():
    acquisition = Acquisition()
    with pytest.warns(DeprecationWarning):
        acquisition.add_source_group(
            kind="scalar",
            coords=[[0.25, 0.05], [0.50, 0.05], [0.75, 0.05]],
        )
    with pytest.warns(DeprecationWarning):
        acquisition.add_compound_source(
            kind="scalar",
            coords=[[0.45, 0.08], [0.55, 0.08]],
            weights=[1.0, -1.0],
        )

    assert acquisition.source_field_names() == [
        "source_0",
        "source_1",
        "source_2",
        "source_3",
    ]
    with pytest.warns(DeprecationWarning, match="source_groups"):
        assert [group.source.name for group in acquisition.source_groups] == [
            "source_0",
            "source_1",
            "source_2",
            "source_3",
        ]
