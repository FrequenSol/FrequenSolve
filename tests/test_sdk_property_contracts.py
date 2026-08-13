import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st
from pint.errors import UndefinedUnitError

from frequensolve.geometry.frame import CoordinateValue
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.model.property import _ensure_minimum_coordinates
from frequensolve.project.project import Project
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.sources import SourceEncoding, SourceGeometry
from frequensolve.simulation.jobs import FrequencyDomainJob, JobLayout
from frequensolve.simulation.outputs import JobOutputs, TraceOutput
from frequensolve.units import Q_, UnitConfig, quantity_to_fs
from tests.property_strategies import (
    ACQUISITION_CASES,
    FINITE_FLOATS,
    INVALID_DIMENSIONS,
    INVALID_UNIT_EXPRESSIONS,
    JOB_CASES,
    NONRECTANGULAR_COORDINATE_ROWS,
    OUTPUT_SELECTIONS,
    POSITIVE_FLOATS,
    SAFE_NAMES,
    SAFE_RELATIVE_PATHS,
    SIMULATION_CASES,
    UNIT_EXPRESSIONS,
    UNSAFE_PATHS,
    UNSAFE_RELATIVE_PATHS,
)

pytestmark = [pytest.mark.unit, pytest.mark.property_contract]


@given(
    values=st.lists(FINITE_FLOATS, min_size=1, max_size=6),
    units=UNIT_EXPRESSIONS,
    system=SAFE_NAMES,
    extension=SAFE_NAMES,
)
def test_coordinate_value_roundtrip_preserves_units_system_and_extensions(
    values,
    units,
    system,
    extension,
):
    coordinate = CoordinateValue(
        values,
        units=units,
        system=system,
        extra={"solver_extension": extension},
    )

    payload = coordinate.to_fs()

    assert CoordinateValue.from_fs(payload).to_fs() == payload


@given(
    case=ACQUISITION_CASES,
)
def test_inline_acquisition_roundtrip_preserves_coordinate_shape_and_metadata(
    case,
):
    rows = case["rows"]
    names = [f"source_{index}" for index in range(len(rows))]
    geometry = SourceGeometry.points(
        kind="scalar",
        coords=rows,
        names=names,
        units=case["units"],
        system=case["system"],
    )
    acquisition = Acquisition(source_geometry=geometry)

    payload = acquisition.to_fs()
    roundtrip = Acquisition.from_fs(payload)

    assert roundtrip.to_fs() == payload
    assert roundtrip.source_geometry.coordinates().shape == (
        len(rows),
        len(rows[0]),
    )


@given(
    case=ACQUISITION_CASES,
    coefficients=st.lists(
        POSITIVE_FLOATS,
        min_size=1,
        max_size=5,
    ),
)
def test_dense_source_encoding_roundtrip_preserves_coefficients(case, coefficients):
    rows = case["rows"]
    source_count = len(rows)
    field_count = len(coefficients)
    names = [f"source_{index}" for index in range(source_count)]
    matrix = np.asarray(
        [coefficients for _ in range(source_count)],
        dtype=float,
    )
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar",
            coords=rows,
            names=names,
        ),
        source_encoding=SourceEncoding.dense(
            matrix,
            names=[f"field_{index}" for index in range(field_count)],
        ),
    )

    payload = acquisition.to_fs()

    assert Acquisition.from_fs(payload).to_fs() == payload


@given(
    reference=POSITIVE_FLOATS,
    default_units=UNIT_EXPRESSIONS,
    scale_units=UNIT_EXPRESSIONS,
    extension=SAFE_NAMES,
)
def test_unit_configuration_roundtrip_preserves_defaults_scales_and_extensions(
    reference,
    default_units,
    scale_units,
    extension,
):
    config = UnitConfig(
        f0=reference,
        defaults={"field": default_units},
        scales={"solver_scale": scale_units},
        units_extra={"solver_extension": extension},
    )

    payload = config.to_fs()

    assert UnitConfig.from_fs(payload).to_fs() == payload


@given(
    magnitude=FINITE_FLOATS,
    conversion=st.sampled_from((("m", "km"), ("km", "m"), ("s", "ms"), ("ms", "s"))),
)
def test_quantity_conversion_roundtrip_preserves_magnitude(magnitude, conversion):
    source_units, target_units = conversion
    original = Q_(magnitude, source_units)

    roundtrip = original.to(target_units).to(source_units)
    payload = quantity_to_fs(original)

    assert roundtrip.magnitude == pytest.approx(magnitude)
    assert payload["units"] == source_units


@given(units=INVALID_UNIT_EXPRESSIONS)
def test_invalid_unit_expressions_fail_with_unit_context(units):
    with pytest.raises(UndefinedUnitError) as exc_info:
        Q_(1.0, units)

    assert units in str(exc_info.value)


@given(rows=NONRECTANGULAR_COORDINATE_ROWS)
def test_nonrectangular_source_coordinates_fail_with_shape_context(rows):
    with pytest.raises(ValueError, match="coordinates.*shape"):
        SourceGeometry.points(kind="scalar", coords=rows)


def test_ragged_source_coordinate_regression_reports_stable_shape_error():
    with pytest.raises(
        ValueError,
        match=r"^source coordinates must have shape \(n, dim\)$",
    ):
        SourceGeometry.points(kind="scalar", coords=[[0.0], [0.0, 1.0]])


@given(case=SIMULATION_CASES)
def test_simulation_schema_and_dimension_normalization_are_stable(case):
    with tempfile.TemporaryDirectory() as temporary:
        project = Project(name="project", path=Path(temporary) / case["name"])
        simulation = project.new_simulation(**case)

        payload = simulation.to_fs()

        assert payload["schema"] == "fs-simulation-1"
        assert payload["dimension"] in (2, 2.5, 3)


@given(dimension=INVALID_DIMENSIONS)
def test_invalid_simulation_dimensions_fail_with_field_context(dimension):
    with tempfile.TemporaryDirectory() as temporary:
        project = Project(name="project", path=Path(temporary))

        with pytest.raises(ValueError, match="dimension must be"):
            project.new_simulation(
                name="simulation",
                physics="acoustic",
                dimension=dimension,
            )


@st.composite
def _coordinate_arrays(draw):
    dimensions = draw(st.integers(min_value=1, max_value=3))
    sizes = draw(
        st.lists(
            st.integers(min_value=1, max_value=3),
            min_size=dimensions,
            max_size=dimensions,
        )
    )
    names = ("x", "y", "z")[:dimensions]
    shape = tuple(sizes)
    values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    coordinates = {
        name: np.arange(size, dtype=float) for name, size in zip(names, sizes)
    }
    return xr.DataArray(values, dims=names, coords=coordinates)


@given(data=_coordinate_arrays())
def test_minimum_coordinate_expansion_preserves_values_and_axis_order(data):
    expanded = _ensure_minimum_coordinates(data)

    assert expanded.dims == data.dims
    assert all(expanded.sizes[dimension] >= 2 for dimension in data.dims)
    original_selection = {
        dimension: expanded.coords[dimension].values[: data.sizes[dimension]]
        for dimension in data.dims
    }
    xr.testing.assert_equal(expanded.sel(original_selection), data)


@given(selection=OUTPUT_SELECTIONS)
def test_trace_output_safe_relative_path_roundtrips(selection):
    payload = JobOutputs(
        traces=TraceOutput(
            path=selection["path"],
            components=[selection["component"]],
        )
    ).to_fs()

    assert JobOutputs.from_fs(payload).to_fs() == payload


@given(path=UNSAFE_PATHS)
def test_trace_output_rejects_paths_outside_result_directory(path):
    with pytest.raises(ValueError, match="safe relative path"):
        TraceOutput(path=path)


@given(path=SAFE_RELATIVE_PATHS)
def test_job_layout_preserves_safe_project_relative_paths(path):
    project_path = Path("/bounded/project")
    simulation_path = f"{path}.json"
    layout = JobLayout.from_payload(
        {
            "project_path": str(project_path),
            "simulation": simulation_path,
            "result_path": f"results/{path}",
            "name": "job",
        }
    )

    assert layout.simulation_file == project_path / simulation_path
    assert layout.result_dir == project_path / "results" / path


@given(path=UNSAFE_RELATIVE_PATHS)
def test_job_layout_rejects_unsafe_relative_paths(path):
    with pytest.raises(ValueError, match="unsafe|escape"):
        JobLayout.from_payload(
            {
                "project_path": "/bounded/project",
                "simulation": path,
                "result_path": "results/job",
                "name": "job",
            }
        )


@given(case=JOB_CASES)
def test_saved_frequency_job_roundtrip_preserves_public_contract(
    case,
):
    with tempfile.TemporaryDirectory() as temporary:
        project = Project(name="project", path=Path(temporary))
        simulation = project.new_simulation(
            name="simulation",
            physics="acoustic",
            dimension=2,
        )
        simulation.mesh = MeshManager(
            HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1])
        )
        simulation.save()
        job = FrequencyDomainJob(
            name=case["name"],
            simulation=simulation,
            f_list=case["frequencies"],
            outputs=TraceOutput(
                path=case["outputs"]["path"],
                components=[case["outputs"]["component"]],
            ),
        )

        loaded = FrequencyDomainJob.load(job.save())

        assert loaded.name == case["name"]
        assert loaded.simulation.name == simulation.name
        assert loaded.to_fs()["schema"] == "fs-job-1"
        assert loaded.to_fs(project_relative=True) == job.to_fs(project_relative=True)


def test_job_frequency_payload_regression_is_plain_json_data():
    with tempfile.TemporaryDirectory() as temporary:
        project = Project(name="project", path=Path(temporary))
        simulation = project.new_simulation(
            name="simulation",
            physics="acoustic",
            dimension=2,
        )
        simulation.save()
        job = FrequencyDomainJob(
            name="frequency-list",
            simulation=simulation,
            f_list=[1.0, 2.0],
        )

        assert job.to_fs()["f_list"] == [1.0, 2.0]


@given(
    case=ACQUISITION_CASES,
    extra_names=st.integers(min_value=1, max_value=3),
)
def test_source_geometry_name_count_failure_is_deterministic(case, extra_names):
    rows = case["rows"]
    names = [f"source_{index}" for index in range(len(rows) + extra_names)]

    with pytest.raises(ValueError, match="names must have exactly .* entries"):
        SourceGeometry.points(kind="scalar", coords=rows, names=names)
