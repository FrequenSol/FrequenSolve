"""Native DAS placement preserves explicit nonuniform gauge centers."""

import pytest

import frequensolve as fs


def test_center_mode_roundtrip_and_units():
    fiber = fs.ReceiverFiber(
        gauge_length=23.93 * fs.ureg.m,
        points_per_gauge=129,
        coordinate_mode="gauge_centers",
        components=[fs.ReceiverComponent(name="strain", field="elastic:strain")],
    )
    payload = fiber.to_fs()
    assert payload["coordinate_mode"] == "gauge_centers"
    assert payload["points_per_gauge"] == 129
    assert "channel_spacing" not in payload
    assert fs.ReceiverFiber.from_fs(payload).to_fs() == payload


def test_path_default_unchanged():
    payload = fs.ReceiverFiber(gauge_length=10, points_per_gauge=129).to_fs()
    assert "coordinate_mode" not in payload
    assert payload["channel_spacing"] == 10
    assert fs.ReceiverFiber.from_fs(payload).to_fs() == payload


@pytest.mark.parametrize(
    "extra", [dict(channel_spacing=12), dict(radius=1), dict(pitch=1), dict(angle=30)]
)
def test_centers_reject_conflicting_geometry(extra):
    with pytest.raises(ValueError):
        fs.ReceiverFiber(gauge_length=10, coordinate_mode="gauge_centers", **extra)


def test_invalid_mode():
    with pytest.raises(ValueError, match="coordinate_mode"):
        fs.ReceiverFiber(gauge_length=10, coordinate_mode="unknown")
