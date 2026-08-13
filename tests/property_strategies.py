"""Bounded Hypothesis strategies for Python SDK and serialization contracts.

Native solver-parser fuzzing belongs to FrequenSol/Sauce#53.  This module stays
inside FrequenSolve's credential-free Python model, validation, serialization,
and filesystem-safety boundary.
"""

import string

from hypothesis import strategies as st

SAFE_NAMES = st.text(
    alphabet=string.ascii_letters + string.digits + "_-",
    min_size=1,
    max_size=24,
)

FINITE_FLOATS = st.floats(
    min_value=-1.0e6,
    max_value=1.0e6,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)

POSITIVE_FLOATS = st.floats(
    min_value=1.0e-3,
    max_value=1.0e6,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@st.composite
def _rectangular_coordinate_rows(draw):
    dimensions = draw(st.integers(min_value=1, max_value=3))
    return draw(
        st.lists(
            st.lists(
                FINITE_FLOATS,
                min_size=dimensions,
                max_size=dimensions,
            ),
            min_size=1,
            max_size=5,
        )
    )


COORDINATE_ROWS = _rectangular_coordinate_rows()
NONRECTANGULAR_COORDINATE_ROWS = st.tuples(
    st.lists(FINITE_FLOATS, min_size=1, max_size=2),
    st.lists(FINITE_FLOATS, min_size=3, max_size=3),
).map(list)

UNIT_EXPRESSIONS = st.sampled_from(("m", "km", "s", "ms", "Hz", "kg/m^3", "m/s", "Pa"))
INVALID_UNIT_EXPRESSIONS = st.sampled_from(
    ("not_a_frequensolve_unit", "unknown_solver_length_unit")
)

VALID_DIMENSIONS = st.sampled_from((2, 2.5, 3, "2D", "2.5D", "3D"))
INVALID_DIMENSIONS = st.sampled_from((None, 0, 1, 4, -2, 2.25, "", "1D", "4D", "two"))

ACQUISITION_CASES = st.fixed_dictionaries(
    {
        "rows": COORDINATE_ROWS,
        "units": st.sampled_from(("m", "km")),
        "system": SAFE_NAMES,
    }
)

OUTPUT_SELECTIONS = st.fixed_dictionaries(
    {
        "path": st.lists(SAFE_NAMES, min_size=1, max_size=4).map(
            lambda parts: "/".join(parts)
        ),
        "component": st.sampled_from(("pressure", "velocity_x", "velocity_z")),
    }
)

SIMULATION_CASES = st.fixed_dictionaries(
    {
        "name": SAFE_NAMES,
        "dimension": VALID_DIMENSIONS,
        "physics": st.just("acoustic"),
    }
)

JOB_CASES = st.fixed_dictionaries(
    {
        "name": SAFE_NAMES,
        "frequencies": st.lists(
            POSITIVE_FLOATS,
            min_size=1,
            max_size=5,
            unique=True,
        ),
        "outputs": OUTPUT_SELECTIONS,
    }
)

SAFE_RELATIVE_PATHS = st.lists(SAFE_NAMES, min_size=1, max_size=4).map(
    lambda parts: "/".join(parts)
)

UNSAFE_PATHS = st.sampled_from(
    (
        "../escape.json",
        "nested/../../escape.json",
        "/absolute/result.json",
        "bad\\windows.json",
        "bad\nname.json",
        "bad\x00name.json",
    )
)

UNSAFE_RELATIVE_PATHS = st.sampled_from(
    (
        "../escape.json",
        "nested/../../escape.json",
        "bad\\windows.json",
        "bad\nname.json",
        "bad\x00name.json",
    )
)
