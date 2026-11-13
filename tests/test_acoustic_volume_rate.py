"""Physical acoustic source authoring and its reciprocal pressure measurement."""

import numpy as np
import pytest

from frequensolve.seismic import Acquisition, PointSource, SourceGeometry
from frequensolve.units import ureg as u


@pytest.mark.parametrize("kind", ["volume_injection"])
def test_volume_rate_and_reciprocal_receiver(kind):
    source = PointSource(coords=[0, 0], kind=kind, amplitude=2 * u.m**3 / u.s)
    payload = source.to_fs()
    assert payload["amplitude"]["value"] == 2
    assert u.Unit(payload["amplitude"]["units"]) == u.m**3 / u.s
    receiver = source.reciprocal_receiver(physics="acoustic")
    assert receiver.to_fs() == {"name": "p", "field": "pressure", "units": "Pa"}
    assert receiver.response is None
    assert receiver.transfer is None
    assert receiver.weight is None


def test_reciprocal_catalog_does_not_materialize_bulk_points():
    geometry = SourceGeometry.points(kind="volume_injection", coords=np.zeros((100, 2)))
    before = geometry._storage
    receiver = geometry.reciprocal_receiver(
        physics="acoustic", name="hydrophone", units="kPa"
    )
    assert geometry._storage is before
    assert receiver.name == "hydrophone"
    assert receiver.units == "kPa"


@pytest.mark.parametrize(
    "physics,kind,units",
    [
        ("elastic", "volume_injection", "Pa"),
        ("acoustic", "vector", "Pa"),
        ("acoustic", "volume_injection", "m/s"),
        ("acoustic", "scalar", "Pa"),
        ("acoustic", "monopole", "Pa"),
    ],
)
def test_unsupported_pairings_are_explicit(physics, kind, units):
    source = PointSource(coords=[0, 0], kind=kind)
    with pytest.raises(ValueError):
        source.reciprocal_receiver(physics=physics, units=units)


def test_catalog_kind_overrides_are_checked():
    geometry = SourceGeometry(
        kind="volume_injection", sources=[PointSource(coords=[0, 0], kind="vector")]
    )
    with pytest.raises(ValueError, match="volume_injection"):
        geometry.reciprocal_receiver(physics="acoustic")


@pytest.mark.parametrize(
    "defaults", [{"kind": "scalar"}, {"mechanism": {"type": "isotropic"}}]
)
def test_reciprocal_receiver_rejects_legacy_geometry_defaults(defaults):
    geometry = SourceGeometry.points(
        kind="volume_injection", coords=[[0, 0]], defaults=defaults
    )
    with pytest.raises(ValueError):
        geometry.reciprocal_receiver(physics="acoustic")


def test_reciprocal_receiver_rejects_conflicting_point_mechanism():
    source = PointSource(
        coords=[0, 0], kind="volume_injection", mechanism={"type": "isotropic"}
    )
    with pytest.raises(ValueError):
        source.reciprocal_receiver(physics="acoustic")


def test_explicit_mixed_catalog_keeps_each_source_definition():
    acquisition = Acquisition(
        source_geometry=SourceGeometry(
            kind="scalar",
            sources=[
                PointSource(coords=[0, 0], kind="scalar", amplitude=2 * u.N * u.m),
                PointSource(
                    coords=[0, 0], kind="volume_injection", amplitude=3 * u.m**3 / u.s
                ),
            ],
        )
    )
    points = acquisition.to_fs()["source_geometry"]["sources"]
    assert [point["kind"] for point in points] == ["scalar", "volume_injection"]
    assert u.Unit(points[0]["amplitude"]["units"]) == u.N * u.m
    assert u.Unit(points[1]["amplitude"]["units"]) == u.m**3 / u.s
