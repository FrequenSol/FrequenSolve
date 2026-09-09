import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from frequensolve import (
    Acquisition,
    CoordinateValue,
    Direction,
    DistributedSource,
    EncodedReceiver,
    EncodedSource,
    PointSource,
    ReceiverArray,
    ReceiverComponent,
    SourceEncoding,
    SourceGeometry,
    SourceGroup,
)
from frequensolve.seismic.receivers import ReceiverDevice, ReceiverNode
from frequensolve.seismic.sparse_survey import (
    EvalSample,
    SparseSurvey,
    SparseTraceTable,
)
from frequensolve.units import ureg
from frequensolve.util.mixins import ExportContext
from frequensolve.util.store import SimulationStore

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
            EncodedSource.named("shot_left", {"shot_left": 1.0}),
            EncodedSource.named("shot_center", {"shot_center": 1.0}),
            EncodedSource.named("shot_right", {"shot_right": 1.0}),
            EncodedSource.named("difference", {"pair_pos": 1.0, "pair_neg": -1.0}),
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
            [[1.0, 0.0], [1.0, -1.0]],
            names=["left_only", "difference"],
        ),
    )

    payload = acquisition.to_fs()

    _sauce_acquisition_validator().validate(payload)
    assert payload["source_encoding"]["_type"] == "JsonDense"
    assert Acquisition.from_fs(payload).to_fs() == payload


@pytest.mark.parametrize(
    "weights",
    [
        [[1.0 + 2.0j, -3.0j], [0.0, 0.5]],
        [[1.0 + 2.0j, -3.0j, 4.0], [0.0, 0.5, -2.0 + 1.0j]],
    ],
)
def test_dense_encoding_materializes_in_hdf5_without_source_axis_metadata(
    tmp_path, weights
):
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar",
            coords=[[float(index), 0.0] for index in range(len(weights[0]))],
        ),
        source_encoding=SourceEncoding.dense(
            weights,
            names=["focus", "reference"],
        ),
    )
    # Sauce consumes one source-coefficient vector per encoded field in JSON,
    # and the same field-major values as (field, source, complex) in HDF5.
    inline = acquisition.to_fs()
    _sauce_acquisition_validator().validate(inline)
    for field, expected in zip(inline["source_encoding"]["fields"], weights):
        actual = [
            complex(*value) if isinstance(value, list) else complex(value)
            for value in field["coefficients"]
        ]
        np.testing.assert_allclose(actual, expected)
    store = SimulationStore(tmp_path / "simulation.h5", project_path=tmp_path)
    payload = acquisition.to_fs(ExportContext(tmp_path, store=store))

    encoding = payload["source_encoding"]
    assert encoding == {
        "_type": "HDF5Dense",
        "file": "simulation.h5",
        "dataset": "inputs/acquisition/source_encoding/coefficients",
        "field_names_dataset": "inputs/acquisition/source_encoding/field_names",
        "hash": encoding["hash"],
    }
    with h5py.File(tmp_path / "simulation.h5", "r") as h5:
        stored = h5[encoding["dataset"]]
        assert stored.shape == (2, len(weights[0]), 2)
        assert "field" not in stored.attrs
        assert "source" not in stored.attrs
        assert list(h5[encoding["field_names_dataset"]].asstr()[:]) == [
            "focus",
            "reference",
        ]
        np.testing.assert_allclose(
            stored[..., 0] + 1j * stored[..., 1],
            weights,
        )


def test_frequency_dense_encoding_materializes_one_hdf5_tensor(tmp_path):
    weights = np.zeros((3, 2, 4), dtype=np.complex64)
    weights[:, 0, :] = 1.0
    weights[:, 1, :] = np.asarray(
        [
            [1.0, -1.0, 1.0, -1.0],
            [1.0j, -1.0j, 1.0j, -1.0j],
            [-1.0, -1.0, 1.0, 1.0],
        ]
    )
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar",
            coords=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        )
    )
    encoding = acquisition.encode_sources(
        weights,
        frequencies=[2.0, 3.0, 4.0],
        names=["sum", "changing_code"],
    )
    store = SimulationStore(tmp_path / "simulation.h5", project_path=tmp_path)
    payload = acquisition.to_fs(ExportContext(tmp_path, store=store))

    source_encoding = payload["source_encoding"]
    assert encoding.weights is weights
    assert source_encoding["_type"] == "HDF5Dense"
    assert source_encoding["frequencies_dataset"].endswith("/frequencies")
    _sauce_acquisition_validator().validate(payload)
    with h5py.File(tmp_path / "simulation.h5", "r") as h5:
        stored = h5[source_encoding["dataset"]]
        frequencies = h5[source_encoding["frequencies_dataset"]]
        assert stored.shape == (3, 2, 4, 2)
        assert frequencies[...].tolist() == [2.0, 3.0, 4.0]
        assert "frequency" not in stored.attrs
        np.testing.assert_allclose(
            stored[..., 0] + 1.0j * stored[..., 1],
            weights,
        )

    with pytest.raises(ValueError, match="requires a simulation/project store"):
        encoding.to_fs()
    assert store.prune_unreferenced(payload) == []
    _sauce_acquisition_validator().validate(payload)


def test_large_source_catalog_materializes_coordinates_and_names_in_hdf5(tmp_path):
    count = 1_000
    coordinates = np.column_stack((np.linspace(0.0, 1.0, count), np.zeros(count)))
    names = [f"shot-{index}" for index in range(count)]
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar",
            coords=coordinates,
            names=names,
            units="m",
            system="global",
        )
    )
    store = SimulationStore(tmp_path / "simulation.h5", project_path=tmp_path)

    payload = acquisition.to_fs(ExportContext(tmp_path, store=store))
    geometry = payload["source_geometry"]

    assert geometry["_type"] == "HDF5"
    assert geometry["file"] == "simulation.h5"
    assert "sources" not in geometry
    assert geometry["dataset"] == "inputs/acquisition/source_geometry/coordinates"
    assert geometry["names_dataset"] == "inputs/acquisition/source_geometry/names"
    with h5py.File(tmp_path / "simulation.h5", "r") as h5:
        np.testing.assert_allclose(h5[geometry["dataset"]][:], coordinates)
        assert list(h5[geometry["names_dataset"]].asstr()[:]) == names
    assert store.prune_unreferenced(payload) == []
    _sauce_acquisition_validator().validate(payload)


def test_bulk_source_catalog_keeps_default_names_implicit(tmp_path):
    count = 1_000
    coordinates = np.column_stack((np.arange(count), np.zeros(count)))
    geometry = SourceGeometry.points(kind="scalar", coords=coordinates, units="m")
    store = SimulationStore(tmp_path / "simulation.h5", project_path=tmp_path)

    payload = geometry.to_fs(ExportContext(tmp_path, store=store))

    assert geometry.is_bulk
    assert geometry.point_count == count
    assert payload["_type"] == "HDF5"
    assert "names_dataset" not in payload
    with h5py.File(tmp_path / "simulation.h5", "r") as h5:
        np.testing.assert_allclose(h5[payload["dataset"]][:], coordinates)


def test_large_sparse_survey_materializes_without_inline_trace_metadata(tmp_path):
    survey = SparseSurvey.from_product(
        "production survey",
        sources=range(1, 6),
        receivers=range(1, 101),
    )
    ctx = ExportContext(tmp_path, Path("simulations/model"))

    payload = survey.to_fs(ctx)

    assert payload["name"] == "production survey"
    assert payload["_type"] == "HDF5TraceStore"
    assert payload["layout_file"].startswith(
        "simulations/model/surveys/production_survey-"
    )
    assert payload["layout_file"].endswith(".h5")
    assert "traces" not in payload
    with h5py.File(tmp_path / payload["layout_file"], "r") as h5:
        assert h5["survey/traces/trace_id"].shape == (500,)
        assert "source_name" not in h5["survey/traces"]
        assert "receiver_name" not in h5["survey/traces"]
        assert "component_name" not in h5["survey/traces"]
        assert "offset" not in h5["survey/traces"]
        assert "azimuth" not in h5["survey/traces"]


def test_sparse_trace_table_vectorizes_named_component_products(tmp_path):
    table = SparseTraceTable.from_product(
        sources=range(1, 3),
        receivers=[10, 20],
        components=["p", "vx"],
        receiver_points={10: 1, 20: 2},
    )
    survey = SparseSurvey.from_table("columnar", table)

    columns = table.columns({"p": 1, "vx": 2})

    assert survey.trace_count == 8
    assert columns["source_id"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert columns["receiver_id"].tolist() == [10, 10, 20, 20] * 2
    assert columns["receiver_position_id"].tolist() == [1, 1, 2, 2] * 2
    assert columns["component"].tolist() == [1, 2] * 4
    assert columns["component_name"].tolist() == ["p", "vx"] * 4

    numeric = SparseTraceTable.from_product(sources=[1], receivers=[1, 2])
    assert numeric.columns()["component"].dtype == np.int64

    path = survey.write_hdf5(
        tmp_path / "named-components.h5", component_map={"p": 1, "vx": 2}
    )
    with h5py.File(path, "r") as h5:
        assert "component_name" not in h5["survey/traces"]
        assert h5["survey/components/component_name"].asstr()[:].tolist() == [
            "p",
            "vx",
        ]


def test_complex_source_responses_are_conveniently_time_reversed():
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar",
            coords=[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            names=["left", "center", "right"],
        )
    )

    weights = np.asarray(
        [[1.0 + 2.0j, -3.0j, 0.5 - 0.25j]],
        dtype=np.complex64,
    )
    encoding = acquisition.encode_sources(
        weights,
        names=["focus"],
        conjugate=True,
    )
    payload = acquisition.to_fs()

    assert encoding.weights is weights
    assert payload["source_encoding"]["conjugate_coefficients"] is True
    assert payload["source_encoding"]["fields"][0]["coefficients"] == [
        [1.0, 2.0],
        [0.0, -3.0],
        [0.5, -0.25],
    ]
    assert encoding.time_reversed().time_reversed().to_fs() == encoding.to_fs()
    _sauce_acquisition_validator().validate(payload)
    assert Acquisition.from_fs(payload).to_fs() == payload


def test_dense_source_encoding_keeps_one_shared_production_matrix():
    weights = np.ones((4, 250_000), dtype=np.complex64)

    encoding = SourceEncoding.dense(weights)
    reversed_encoding = encoding.time_reversed()

    assert encoding.weights is weights
    assert reversed_encoding.weights is weights
    assert reversed_encoding.conjugate_coefficients is True
    assert encoding.conjugate_coefficients is False
    for index, field in enumerate(encoding.fields):
        assert np.shares_memory(field.coefficients, weights[index, :])


def test_encoded_source_uses_encoding_major_weights_and_keeps_legacy_aliases():
    weights = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]],
        dtype=np.complex64,
    )
    encoding = SourceEncoding.dense(weights, names=["left", "difference"])

    assert encoding.weights is weights
    np.testing.assert_allclose(encoding.fields[0].coefficients, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(encoding.fields[1].coefficients, [0.0, 1.0, -1.0])

    legacy_coefficients = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.complex64,
    )
    with pytest.warns(DeprecationWarning, match="coefficients"):
        legacy_encoding = SourceEncoding(
            encoding_type="JsonDense",
            fields=[EncodedSource(name="left"), EncodedSource(name="difference")],
            coefficients=legacy_coefficients,
        )
    np.testing.assert_allclose(legacy_encoding.weights, weights)
    with pytest.warns(DeprecationWarning, match="coefficients"):
        legacy_factory_encoding = SourceEncoding.dense(
            coefficients=legacy_coefficients,
            names=["left", "difference"],
        )
    np.testing.assert_allclose(legacy_factory_encoding.weights, weights)
    with pytest.warns(DeprecationWarning, match="coefficients"):
        legacy_frequency_encoding = SourceEncoding.frequency_dense(
            coefficients=legacy_coefficients[np.newaxis, :, :],
            frequencies=[2.0],
            names=["left", "difference"],
        )
    np.testing.assert_allclose(
        legacy_frequency_encoding.weights,
        weights[np.newaxis, :, :],
    )

    acquisition = Acquisition()
    acquisition.add_sources(
        kind="scalar",
        coords=[[0.0, 0.0], [1.0, 0.0]],
        names=["left", "right"],
    )
    encoded = acquisition.add_encoded_source("sum", {"left": 1.0, "right": 1.0})
    assert isinstance(encoded, EncodedSource)
    assert acquisition.source_field(1) is encoded

    legacy_acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar",
            coords=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        )
    )
    with pytest.warns(DeprecationWarning, match="coefficients"):
        legacy_acquisition_encoding = legacy_acquisition.encode_sources(
            coefficients=legacy_coefficients
        )
    np.testing.assert_allclose(legacy_acquisition_encoding.weights, weights)

    with pytest.warns(DeprecationWarning, match="DistributedSource"):
        legacy_field = DistributedSource.named("left", {"left": 1.0})
    assert isinstance(legacy_field, EncodedSource)
    with pytest.warns(DeprecationWarning, match="add_distributed_source"):
        legacy_added = acquisition.add_distributed_source("right", {"right": 1.0})
    assert isinstance(legacy_added, EncodedSource)


def test_acquisition_accessors_reject_zero_and_list_device_fields():
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(kind="scalar", coords=[[0.0, 0.0]])
    )
    device = ReceiverNode()
    device.add_component("pressure", "pressure")
    acquisition.add_receiver_group("surface", device, [[0.0, 0.0]])

    assert acquisition.list_fields() == ["surface:pressure"]
    assert acquisition.source_field(1) is acquisition.source_geometry.sources[0]
    with pytest.raises(IndexError, match="Source index 0"):
        acquisition.source_field(0)
    with pytest.raises(IndexError, match="Source index 0"):
        acquisition.source(0)
    with pytest.raises(IndexError, match="Source index 0"):
        acquisition.source_coords(0)


def test_duplicate_receiver_names_are_rejected_without_mutation():
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(kind="scalar", coords=[[0.0, 0.0]])
    )
    for _ in range(2):
        device = ReceiverNode()
        device.add_component("pressure", "pressure")
        acquisition.add_receiver_group("surface", device, [[0.0, 0.0]])

    with pytest.raises(ValueError, match="Receiver group names must be unique"):
        acquisition.to_fs()
    assert [group.name for group in acquisition.receiver_groups] == [
        "surface",
        "surface",
    ]


def test_encoded_receivers_share_geometry_and_support_complex_time_reversal(tmp_path):
    device = EncodedReceiver(
        name="focused_arrays",
        components=[ReceiverComponent(name="pressure", field="pressure")],
    )
    device.add_encoding(
        name="target_a",
        weights=[1.0 + 2.0j, -0.5j, 0.25],
        conjugate=True,
    )
    device.add_encoding(
        name="target_b",
        weights=[-1.0j, 0.5 + 0.25j, 2.0],
        conjugate=True,
    )
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar",
            coords=[[0.5, 0.5]],
            names=["probe"],
        )
    )
    acquisition.add_receiver_group(
        name="surface",
        device=device,
        coords=[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
    )
    assert acquisition.receiver_groups[0].output_size == 1

    store = SimulationStore(tmp_path / "simulation.h5", project_path=tmp_path)
    ctx = ExportContext(tmp_path, store=store)
    payload = acquisition.to_fs(ctx)
    receiver = payload["receiver_groups"][0]

    assert receiver["device"]["_type"] == "EncodedReceiver"
    assert receiver["device"]["encoding_count"] == 2
    assert receiver["device"]["encoding_names"] == ["target_a", "target_b"]
    assert receiver["device"]["reduction"] == "sum"
    assert len(receiver["device"]["components"]) == 1
    assert all(
        "weights" not in component for component in receiver["device"]["components"]
    )
    assert receiver["device"]["weights"] == {
        "_type": "HDF5Dense",
        "file": "simulation.h5",
        "dataset": "inputs/acquisition/receivers/surface/weights",
        "format": "HDF5",
        "hash": receiver["device"]["weights"]["hash"],
    }
    with h5py.File(tmp_path / "simulation.h5", "r") as h5:
        stored = h5["inputs/acquisition/receivers/surface/weights"][:]
        assert stored.shape == (2, 3, 2)
        np.testing.assert_allclose(
            stored[..., 0] + 1j * stored[..., 1],
            np.asarray(
                [
                    [1.0 - 2.0j, 0.5j, 0.25],
                    [1.0j, 0.5 - 0.25j, 2.0],
                ]
            ),
        )
        assert (
            "receiver" not in h5["inputs/acquisition/receivers/surface/weights"].attrs
        )
    _sauce_acquisition_validator().validate(payload)
    assert Acquisition.from_fs(payload).to_fs() == payload


def test_encoded_receiver_bulk_tensor_is_lazy_shared_and_multicomponent(tmp_path):
    weights = np.asarray(
        [
            [
                [1.0 + 2.0j, -0.5j, 0.25],
                [-1.0j, 0.5 + 0.25j, 2.0],
            ],
            [
                [0.5 - 1.0j, 0.75j, -0.25],
                [2.0j, -0.5 + 0.5j, 1.0],
            ],
        ],
        dtype=np.complex64,
    )
    device = EncodedReceiver(
        components=[
            ReceiverComponent(name="vx", field="velocity", direction=[1.0, 0.0]),
            ReceiverComponent(name="vz", field="velocity", direction=[0.0, 1.0]),
        ],
        encoding_names=["target_a", "target_b"],
        weights=weights,
    )
    assert device.weights is weights
    assert [component.name for component in device.output_components()] == [
        "target_a:vx",
        "target_a:vz",
        "target_b:vx",
        "target_b:vz",
    ]

    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar",
            coords=[[0.5, 0.5]],
            names=["probe"],
        )
    )
    acquisition.add_receiver_group(
        name="surface",
        device=device,
        coords=[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
    )
    store = SimulationStore(tmp_path / "simulation.h5", project_path=tmp_path)
    payload = acquisition.to_fs(ExportContext(tmp_path, store=store))
    serialized = payload["receiver_groups"][0]["device"]

    with h5py.File(tmp_path / "simulation.h5", "r") as h5:
        stored = h5[serialized["weights"]["dataset"]][:]
        assert stored.shape == (4, 3, 2)
        np.testing.assert_allclose(
            stored[..., 0] + 1j * stored[..., 1],
            weights.reshape(4, 3),
        )
    _sauce_acquisition_validator().validate(payload)
    assert Acquisition.from_fs(payload).to_fs() == payload


def test_encoded_receiver_rejects_weights_that_do_not_match_shared_geometry():
    device = EncodedReceiver(
        components=[ReceiverComponent(name="pressure", field="pressure")]
    )
    device.add_encoding(name="focus", weights=[1.0, 2.0])
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar",
            coords=[[0.5, 0.5]],
        )
    )
    acquisition.add_receiver_group(
        name="surface",
        device=device,
        coords=[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
    )

    with pytest.raises(ValueError, match="have 2 receivers.*has 3 points"):
        acquisition.to_fs()


def test_receiver_array_is_a_physical_offset_array_around_each_coordinate():
    device = ReceiverArray(
        components=[ReceiverComponent(name="pressure", field="pressure")],
        offsets=[[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]],
        offset_units="m",
    )
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(kind="scalar", coords=[[0.5, 0.5]])
    )
    acquisition.add_receiver_group(
        name="surface",
        device=device,
        coords=[[0.0, 0.0], [1.0, 0.0]],
    )
    assert acquisition.receiver_groups[0].output_size == 2
    device.reduction = "none"
    assert acquisition.receiver_groups[0].output_size == 6
    device.reduction = "mean"

    payload = acquisition.to_fs()
    serialized = payload["receiver_groups"][0]["device"]

    assert serialized == {
        "_type": "ReceiverArray",
        "components": [{"name": "pressure", "field": "pressure"}],
        "offsets": [[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]],
        "offset_units": "m",
        "reduction": "mean",
    }
    _sauce_acquisition_validator().validate(payload)
    assert Acquisition.from_fs(payload).to_fs() == payload


def test_interim_weighted_receiver_array_payload_migrates_to_encoded_receiver():
    legacy = {
        "_type": "ReceiverArray",
        "components": [
            {"name": "focus_a", "field": "pressure"},
            {"name": "focus_b", "field": "pressure"},
        ],
        "reduction": "sum",
        "weights": {
            "_type": "HDF5Dense",
            "file": "weights.h5",
            "dataset": "/weights",
            "format": "HDF5",
        },
    }

    with pytest.warns(DeprecationWarning, match="use EncodedReceiver"):
        device = ReceiverDevice.from_fs(legacy)

    assert isinstance(device, EncodedReceiver)
    assert [component.name for component in device.output_components()] == [
        "focus_a",
        "focus_b",
    ]
    assert device.to_fs()["_type"] == "EncodedReceiver"


def test_legacy_receiver_array_without_offsets_loads_as_receiver_node():
    legacy = {
        "_type": "ReceiverArray",
        "components": [{"name": "pressure", "field": "pressure"}],
    }

    with pytest.warns(DeprecationWarning, match="loading.*as ReceiverNode"):
        device = ReceiverDevice.from_fs(legacy)

    assert isinstance(device, ReceiverNode)
    assert device.to_fs()["_type"] == "ReceiverNode"


def test_large_encoded_receiver_names_are_materialized_in_hdf5(tmp_path):
    names = [f"target_{index:03d}" for index in range(65)]
    weights = np.ones((65, 1, 3), dtype=np.complex64)
    device = EncodedReceiver(
        components=[ReceiverComponent(name="pressure", field="pressure")],
        encoding_names=names,
        weights=weights,
    )
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(kind="scalar", coords=[[0.5, 0.5]])
    )
    acquisition.add_receiver_group(
        name="surface",
        device=device,
        coords=[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
    )
    store = SimulationStore(tmp_path / "simulation.h5", project_path=tmp_path)

    payload = acquisition.to_fs(ExportContext(tmp_path, store=store))
    serialized = payload["receiver_groups"][0]["device"]

    assert "encoding_names" not in serialized
    assert serialized["weights"]["names_dataset"].endswith("/encoding_names")
    with h5py.File(tmp_path / "simulation.h5", "r") as h5:
        stored_names = h5[serialized["weights"]["names_dataset"]][:].astype(str)
        assert stored_names.tolist() == names
    _sauce_acquisition_validator().validate(payload)


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
            [[1.0, 0.0]],
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
            [EncodedSource.named("midpoint", {"left": 1.0, "right": -1.0})]
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


@pytest.mark.parametrize("encoding_type", ["Named", "JsonDense"])
def test_implicit_encoded_reference_normalizes_compatible_source_units(encoding_type):
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
    encoding = (
        SourceEncoding.named(
            [EncodedSource.named("midpoint", {"left": 1.0, "right": 1.0})]
        )
        if encoding_type == "Named"
        else SourceEncoding.dense([[1.0, 1.0]], names=["midpoint"])
    )
    acquisition = Acquisition(
        source_geometry=geometry,
        source_encoding=encoding,
    )

    reference = acquisition.source_coords(1, preserve_metadata=True)

    assert isinstance(reference, CoordinateValue)
    assert reference.units == "km"
    assert reference.system == "survey"
    assert np.allclose(reference.value, [0.5, 0.05])
    assert np.allclose(acquisition.source_coords(1), [0.5, 0.05])


@pytest.mark.parametrize("encoding_type", ["Named", "JsonDense"])
def test_implicit_encoded_reference_ignores_inactive_source_metadata(encoding_type):
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
            PointSource(
                name="inactive",
                coordinates=CoordinateValue(
                    [1.0, 1.0],
                    units="s",
                    system="unrelated",
                ),
            ),
        ],
    )
    encoding = (
        SourceEncoding.named(
            [
                EncodedSource.named(
                    "midpoint",
                    {"left": 1.0, "right": 1.0, "inactive": 0.0},
                )
            ]
        )
        if encoding_type == "Named"
        else SourceEncoding.dense([[1.0, 1.0, 0.0]], names=["midpoint"])
    )
    acquisition = Acquisition(
        source_geometry=geometry,
        source_encoding=encoding,
    )

    reference = acquisition.source_coords(1, preserve_metadata=True)

    assert isinstance(reference, CoordinateValue)
    assert reference.units == "km"
    assert reference.system == "survey"
    assert np.allclose(reference.value, [0.5, 0.05])


@pytest.mark.parametrize("encoding_type", ["Named", "JsonDense"])
@pytest.mark.parametrize(
    "right_coordinates, message",
    [
        (
            CoordinateValue(
                [1.0, 0.05],
                units="km",
                system="survey",
            ),
            "one coordinate system",
        ),
        (
            CoordinateValue([1.0, 0.05], units="s"),
            "compatible coordinate units",
        ),
    ],
    ids=["coordinate-system", "coordinate-units"],
)
def test_implicit_encoded_reference_rejects_incompatible_active_source_metadata(
    encoding_type,
    right_coordinates,
    message,
):
    geometry = SourceGeometry.inline(
        kind="scalar",
        sources=[
            PointSource(
                name="left",
                coordinates=CoordinateValue([0.0, 0.05], units="km"),
            ),
            PointSource(
                name="right",
                coordinates=right_coordinates,
            ),
        ],
    )
    encoding = (
        SourceEncoding.named(
            [EncodedSource.named("midpoint", {"left": 1.0, "right": 1.0})]
        )
        if encoding_type == "Named"
        else SourceEncoding.dense([[1.0, 1.0]], names=["midpoint"])
    )
    acquisition = Acquisition(
        source_geometry=geometry,
        source_encoding=encoding,
    )

    with pytest.raises(ValueError, match=message):
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
    with pytest.warns(DeprecationWarning, match="deprecated and ignored"):
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


def test_external_source_encoding_count_roundtrips(tmp_path):
    encoding_file = tmp_path / "encoding.h5"
    with h5py.File(encoding_file, "w") as h5:
        h5.create_dataset("coefficients", data=np.ones((2, 4, 2)))
    acquisition = Acquisition(
        source_geometry=SourceGeometry.hdf5(
            "inputs/sources.h5",
            dataset="/sources",
            kind="scalar",
            count=4,
        ),
        source_encoding=SourceEncoding.hdf5(
            encoding_file,
            dataset="/coefficients",
            count=2,
        ),
    )

    payload = acquisition.to_fs()
    loaded = Acquisition.from_fs(payload)

    assert payload["source_geometry"]["count"] == 4
    assert "count" not in payload["source_encoding"]
    _sauce_acquisition_validator().validate(payload)
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
        [EncodedSource.named("field", {"source_000001": 1.0})]
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
        [EncodedSource.named("bad", {"unknown": 1.0})]
    )
    with pytest.raises(ValueError, match="unknown sources"):
        Acquisition(
            source_geometry=geometry,
            source_encoding=unknown_encoding,
        ).to_fs()

    dense_encoding = SourceEncoding.dense([[1.0]], names=["field"])
    dense_encoding.fields[0].coefficients = np.append(
        dense_encoding.fields[0].coefficients,
        0.0,
    )
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


@pytest.mark.parametrize("encoding_kind", ["identity", "dense", "frequency"])
def test_materialized_source_counts_survive_round_trip(tmp_path, encoding_kind):
    count = 201
    geometry = SourceGeometry.points(kind="scalar", coords=np.zeros((count, 2)))
    encoding = None
    if encoding_kind == "dense":
        encoding = SourceEncoding.dense(
            np.ones((3, count), dtype=complex), names=["focus", "reference", "other"]
        )
    elif encoding_kind == "frequency":
        encoding = SourceEncoding.frequency_dense(
            weights=np.ones((2, 3, count), dtype=complex),
            frequencies=[1.0, 2.0],
            names=["focus", "reference", "other"],
        )
    acquisition = Acquisition(source_geometry=geometry, source_encoding=encoding)
    store = SimulationStore(tmp_path / "simulation.h5", project_path=tmp_path)
    payload = acquisition.to_fs(ExportContext(tmp_path, store=store))
    if encoding is not None:
        payload["source_encoding"]["file"] = str(tmp_path / "simulation.h5")
    restored = Acquisition.from_fs(payload)
    assert restored.known_source_point_count() == count
    fields = count if encoding_kind == "identity" else 3
    assert restored.known_source_field_count() == fields
    assert restored.list_sources() == list(range(1, fields + 1))
    if encoding is not None:
        assert restored.source_field_names() == ["focus", "reference", "other"]
    _sauce_acquisition_validator().validate(payload)


def test_external_receiver_names_survive_simulation_load(tmp_path):
    from frequensolve.simulation.jobs.fwi import DataSpace
    from frequensolve.simulation.simulation import SeismicSimulation

    names = [f"channel_{index:03d}" for index in range(65)]
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(kind="scalar", coords=[[0.0, 0.0]]),
        source_encoding=SourceEncoding.dense(
            np.ones((2, 1), dtype=complex), names=["focus", "reference"]
        ),
    )
    acquisition.add_receiver_group(
        "surface",
        EncodedReceiver(
            components=[ReceiverComponent(name="p", field="pressure")],
            encoding_names=names,
            weights=np.ones((65, 1, 3), dtype=complex),
        ),
        coords=[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
    )
    simulation = SeismicSimulation(
        name="named", physics="acoustic", dimension=2, project_path=tmp_path
    )
    simulation.acquisition = acquisition
    restored = SeismicSimulation.load(simulation.save())
    assert restored.acquisition.list_fields() == [f"surface:{name}" for name in names]
    assert restored.acquisition.list_sources() == [1, 2]
    assert restored.acquisition.source_field_names() == ["focus", "reference"]
    assert DataSpace.from_simulation(restored, [1.0]).size == 130


def test_encoded_receiver_time_reversal_shares_and_conjugates_weights(tmp_path):
    weights = np.asarray([[1.0 + 2.0j, 3.0 - 4.0j]])
    device = EncodedReceiver(
        components=[ReceiverComponent(name="p", field="pressure")], weights=weights
    )
    reversed_device = device.time_reversed()
    block, conjugate = next(reversed_device._authored_weight_blocks())
    assert np.shares_memory(block, device.weights)
    assert conjugate
    assert not next(reversed_device.time_reversed()._authored_weight_blocks())[1]
    store = SimulationStore(tmp_path / "simulation.h5", project_path=tmp_path)
    payload = reversed_device.to_fs(
        ExportContext(tmp_path, store=store), group_name="surface", point_count=2
    )
    with h5py.File(store.path, "r") as h5:
        split = h5[payload["weights"]["dataset"]][:]
    np.testing.assert_allclose(split[..., 0] + 1j * split[..., 1], weights.conj())
    np.testing.assert_allclose(device.weights[0, 0], weights[0])


def test_single_receiver_split_weights_use_explicit_component_axis():
    split = np.asarray([[[[1.0, 2.0]]], [[[3.0, 4.0]]]])
    components = [ReceiverComponent(name="p", field="pressure")]
    device = EncodedReceiver(components=components, weights=split)
    device.validate_size(1)
    np.testing.assert_allclose(device.weights[:, 0, 0], [1 + 2j, 3 + 4j])
    canonical = EncodedReceiver(components=components, weights=split[:, 0])
    canonical.validate_size(2)
    np.testing.assert_allclose(canonical.weights[:, 0], [[1, 2], [3, 4]])


@pytest.mark.parametrize("geometry", [{"x": [1.0, 2.0]}, {"direction": [0.0, 1.0]}])
def test_large_survey_keeps_authored_eval_geometry_inline(tmp_path, geometry):
    survey = SparseSurvey.from_product("geometry", sources=[1], receivers=range(1, 202))
    survey.add_eval_sample(EvalSample(sample_id=1, point_id=1, **geometry))
    payload = survey.to_fs(ExportContext(tmp_path))
    assert payload["_type"] != "HDF5TraceStore"
    restored = SparseSurvey.from_fs(payload)
    for key, value in geometry.items():
        assert getattr(restored.eval_samples[0], key) == value
