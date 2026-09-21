import pytest
from hypothesis import given
from hypothesis import strategies as st

from frequensolve.geometry.frame import Axis, Direction

pytestmark = [pytest.mark.unit, pytest.mark.property_contract]

SAFE_TEXT = st.text(
    alphabet=st.characters(
        categories=("Ll", "Lu", "Nd"),
        include_characters="_-",
    ),
    min_size=1,
    max_size=24,
)
FINITE_FLOATS = st.floats(
    min_value=-1.0e9,
    max_value=1.0e9,
    allow_nan=False,
    allow_infinity=False,
)


@given(
    name=SAFE_TEXT,
    direction=st.sampled_from(("x", "y", "z")),
    positive=st.one_of(st.none(), st.sampled_from(("up", "down"))),
    origin=st.one_of(st.none(), FINITE_FLOATS),
    extension=SAFE_TEXT,
)
def test_axis_solver_payload_roundtrip_preserves_supported_and_extension_fields(
    name,
    direction,
    positive,
    origin,
    extension,
):
    axis = Axis(
        name=name,
        direction=direction,
        positive=positive,
        origin=origin,
        extra={"solver_extension": extension},
    )

    payload = axis.to_fs()

    assert Axis.from_fs(payload).to_fs() == payload


@given(
    values=st.lists(FINITE_FLOATS, min_size=1, max_size=4),
    system=st.one_of(st.none(), SAFE_TEXT),
    extension=SAFE_TEXT,
)
def test_vector_direction_solver_payload_roundtrip_preserves_components(
    values,
    system,
    extension,
):
    direction = Direction.vector(values, system=system)
    direction.extra["solver_extension"] = extension

    payload = direction.to_fs()

    assert Direction.from_fs(payload).to_fs() == payload
