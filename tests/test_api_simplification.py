import copy
import hashlib
import inspect
import json
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
import sympy as sp
import xarray as xr

from frequensolve.geometry.frame import (
    Axis,
    CoordinateSystem,
    CoordinateValue,
    Direction,
    SurfaceCoordinateSystem,
)
from frequensolve.mesh.boundary_conditions import (
    BoundaryCondition,
    BoundaryConditions,
)
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import (
    DistanceGrading,
    MeshAdaptor,
    MeshManager,
    SurfaceGrading,
)
from frequensolve.model.layered import LayeredModel
from frequensolve.model.model import ModelBase, ModelSubdomain
from frequensolve.model.property import Property, coord, prop, ref, remap
from frequensolve.project import Project
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import (
    CoordsArray,
    CoordsFromFile,
    ReceiverComponent,
    ReceiverFiber,
    ReceiverNode,
)
from frequensolve.seismic.sparse_survey import SparseSurvey
from frequensolve.seismic.trace_store import TraceStore
from frequensolve.seismic.traces import TraceDataset
from frequensolve.seismic.wavelet import RickerWavelet
from frequensolve.simulation.jobs import FrequencyDomainJob
from frequensolve.simulation.jobs.artifacts import OutputArtifact, TraceManifest
from frequensolve.simulation.outputs import (
    AxisAlignedPlane,
    JobOutputs,
    OutputUnits,
    ParaViewOutput,
    ParaviewOutput,
    TraceOutput,
    WavefieldOutput,
    field,
    info,
    output_property,
    outputs,
    paraview,
    wavefield,
)
from frequensolve.simulation.physics import (
    CoupledAEPComponents,
    ElasticComponents,
    EMComponents,
    PoroelasticComponents,
    components_for_physics,
)
from frequensolve.simulation.simulation import SeismicSimulation
from frequensolve.units import Q_, UnitConfig
from frequensolve.units import ureg as u


def test_property_map_is_editable_and_canonicalizes_names():
    subdomain = ModelSubdomain(
        mesh_block_id=1,
        properties={"Vp": 1.5 * u.km / u.s, "Rho": 2.2 * u.g / u.cm**3},
    )

    subdomain.properties["Qp"] = 300
    subdomain.properties["VP"] = 2.0 * u.km / u.s
    del subdomain.properties["qp"]

    payload = subdomain.to_fs()

    assert set(payload["properties"]) == {"vp", "rho"}
    assert payload["properties"]["vp"] == {"value": 2.0, "units": "km/s"}
    assert payload["properties"]["rho"] == {"value": 2.2, "units": "g/cm^3"}


def test_property_does_not_mutate_input_dataarray_when_scaled():
    data = xr.DataArray(np.array([1.0, 2.0]), dims=["x"], coords={"x": [0.0, 1.0]})

    prop = Property(data, scale=2.0)

    assert data.values.tolist() == [1.0, 2.0]
    assert prop.get().values.tolist() == [2.0, 4.0]


def test_property_map_copies_property_instances_on_assignment():
    source = Property(1.0)
    props = ModelSubdomain(mesh_block_id=1, properties={"vp": source}).properties

    source += 1.0

    assert props["vp"].get() == 1.0
    assert source.get() == 2.0


def test_structured_remote_property_ref_does_not_touch_filesystem():
    prop = Property.file(
        "/server/only/vp.bin",
        scale=0.001,
        units="m/s",
        absolute=True,
    )

    payload = prop.to_fs()

    assert payload == {
        "file": "/server/only/vp.bin",
        "absolute": True,
        "scale": 0.001,
        "units": "m/s",
    }


def test_property_file_valid_range_serializes_lower_bound():
    prop = Property.file(
        "/server/only/vp.rsf",
        absolute=True,
        valid_min=1500.0 * u.m / u.s,
    )

    payload = prop.to_fs()

    assert payload == {
        "file": "/server/only/vp.rsf",
        "format": "rsf",
        "absolute": True,
        "valid_range": {"lower": {"value": 1500.0, "units": "m/s"}},
    }


def test_property_valid_range_preserves_bound_units():
    prop = Property.file(
        "/server/only/vp.rsf",
        units="km/s",
        absolute=True,
        valid_range={
            "min": 1500.0 * u.m / u.s,
            "max": {"value": 4500.0, "units": "m/s"},
        },
        fill_invalid="none",
    )

    payload = prop.to_fs()

    assert payload["units"] == "km/s"
    assert payload["fill_invalid"] == "none"
    assert payload["valid_range"] == {
        "lower": {"value": 1500.0, "units": "m/s"},
        "upper": {"value": 4500.0, "units": "m/s"},
    }


def test_property_valid_range_roundtrips_from_serialized_payload():
    prop = Property.from_value(
        {
            "file": "/server/only/vp.rsf",
            "absolute": True,
            "valid_range": {"min": 1500.0, "max": 4500.0},
            "fill_invalid": "nearest",
        }
    )

    payload = prop.to_fs()

    assert payload["valid_range"] == {"lower": 1500.0, "upper": 4500.0}
    assert payload["fill_invalid"] == "nearest"

    with pytest.raises(ValueError, match="fill_invalid"):
        Property.file("/server/only/vp.rsf", absolute=True, fill_invalid="linear")


def test_property_file_remote_true_serializes_solver_visible_ref():
    prop = Property.file(
        "/server/only/vp.bin",
        scale=0.001,
        units="m/s",
        remote=True,
    )

    payload = prop.to_fs()

    assert payload == {
        "file": "/server/only/vp.bin",
        "absolute": True,
        "scale": 0.001,
        "units": "m/s",
    }


def test_structured_remote_property_payload_uses_remote_flag():
    prop = Property.from_value(
        {
            "file": "/server/only/vp.bin",
            "remote": True,
            "scale": 0.001,
            "units": "m/s",
        }
    )

    payload = prop.to_fs()

    assert payload == {
        "file": "/server/only/vp.bin",
        "absolute": True,
        "scale": 0.001,
        "units": "m/s",
    }


def test_legacy_string_property_loads_as_structured_ref():
    subdomain = ModelSubdomain(
        mesh_block_id=1,
        properties={"Vp": "remote:/server/vp.bin|0.001|xz"},
    )

    payload = subdomain.to_fs()

    assert payload["properties"]["vp"]["file"] == "/server/vp.bin"
    assert payload["properties"]["vp"]["absolute"] is True
    assert payload["properties"]["vp"]["scale"] == 0.001


def test_remote_hdf5_property_locator_infers_format():
    payload = Property.file("remote:/server/model/vp.h5:vp", units="m/s").to_fs()

    assert payload == {
        "file": "/server/model/vp.h5:vp",
        "format": "hdf5",
        "absolute": True,
        "units": "m/s",
    }


def test_remote_rsf_property_infers_format():
    payload = Property.file("remote:/server/model/vp.rsf", units="m/s").to_fs()

    assert payload == {
        "file": "/server/model/vp.rsf",
        "format": "rsf",
        "absolute": True,
        "units": "m/s",
    }


def test_rsf_property_reader_uses_header_grid_and_sidecar(tmp_path):
    header = tmp_path / "vp.rsf"
    sidecar = tmp_path / "vp.rsf@"
    values = np.arange(24, dtype=np.float32)
    values.tofile(sidecar)
    header.write_text(
        "\n".join(
            [
                "n1=2",
                "n2=3",
                "n3=4",
                "d1=0.5",
                "d2=0.25",
                "d3=0.75",
                "o1=0.0",
                "o2=1.0",
                "o3=2.0",
                'label1="X"',
                'label2="Y"',
                'label3="Depth"',
                'unit1="km"',
                'unit2="km"',
                'unit3="km"',
                'label="Vp"',
                'unit="m/s"',
                'data_format="native_float"',
                "esize=4",
                'in="./vp.rsf@"',
            ]
        )
    )

    data = Property.read(header)
    prop = Property(header)

    assert data.dims == ("x", "y", "z")
    assert data.shape == (2, 3, 4)
    assert data.name == "Vp"
    assert data.attrs["units"] == "m/s"
    assert data.coords["z"].attrs["label"] == "Depth"
    assert data.coords["x"].attrs["units"] == "km"
    np.testing.assert_allclose(data.coords["x"], [0.0, 0.5])
    np.testing.assert_allclose(data.coords["y"], [1.0, 1.25, 1.5])
    np.testing.assert_allclose(data.coords["z"], [2.0, 2.75, 3.5, 4.25])
    np.testing.assert_array_equal(data.values.reshape(-1, order="F"), values)
    assert prop.units == "m/s"


def test_derived_property_expressions_export_to_solver_ast():
    subdomain = ModelSubdomain(
        mesh_block_id=1,
        properties={
            "Vp": Property.file("/server/model/vp.h5:vp", units="m/s", absolute=True),
            "Vs": 0.5 * prop("Vp"),
            "Rho": Property.expr(
                0.31 * prop("vp").magnitude("m/s") ** 0.25,
                units="g/cm^3",
            ),
            "mu": prop("rho") * prop("vs") ** 2,
        },
    )

    payload = subdomain.to_fs()
    props = payload["properties"]

    assert props["vp"] == {
        "file": "/server/model/vp.h5:vp",
        "format": "hdf5",
        "absolute": True,
        "units": "m/s",
    }
    assert props["vs"] == {
        "expr": {
            "op": "mul",
            "args": [{"value": 0.5}, {"ref": "vp"}],
        },
        "depends_on": ["vp"],
    }
    assert props["rho"] == {
        "expr": {
            "op": "mul",
            "args": [
                {"value": 0.31},
                {
                    "op": "pow",
                    "args": [
                        {
                            "op": "magnitude",
                            "arg": {"ref": "vp"},
                            "units": "m/s",
                        },
                        {"value": 0.25},
                    ],
                },
            ],
        },
        "depends_on": ["vp"],
        "units": "g/cm^3",
    }
    assert props["mu"]["depends_on"] == ["rho", "vs"]


def test_branch_property_expression_exports_case_ast():
    vp = sp.Symbol("Vp")
    qp = sp.Piecewise(
        (140.0, vp > 4.5),
        (90.0, (vp >= 3.5) & (vp < 4.5)),
        (60.0, True),
        evaluate=False,
    )
    subdomain = ModelSubdomain(
        mesh_block_id=1,
        properties={
            "Vp": 4.0,
            "Qp": qp,
        },
    )

    payload = subdomain.to_fs()

    assert payload["properties"]["qp"] == {
        "expr": {
            "op": "case",
            "branches": [
                {
                    "if": {
                        "op": ">",
                        "args": [{"ref": "vp"}, {"value": 4.5}],
                    },
                    "then": {"value": 140.0},
                },
                {
                    "if": {
                        "op": "and",
                        "args": [
                            {
                                "op": ">=",
                                "args": [{"ref": "vp"}, {"value": 3.5}],
                            },
                            {
                                "op": "<",
                                "args": [{"ref": "vp"}, {"value": 4.5}],
                            },
                        ],
                    },
                    "then": {"value": 90.0},
                },
            ],
            "else": {"value": 60.0},
        },
        "depends_on": ["vp"],
    }


def test_branch_expression_rejects_python_boolean_context():
    with pytest.raises(TypeError, match="Expression conditions"):
        bool(ref("Vp") > 4.5)


def test_remap_expression_macro_lowers_to_existing_ops():
    output = Property.expr(
        remap(
            prop("Vp"),
            from_range=(800, 1000),
            to_range=(600, 1000),
            units="m/s",
            clamp=True,
        )
    )

    payload = output.to_fs()

    assert payload["expr"] == {
        "op": "clamp",
        "args": [
            {
                "op": "add",
                "args": [
                    {"value": 600, "units": "m/s"},
                    {
                        "op": "mul",
                        "args": [
                            {
                                "op": "sub",
                                "args": [
                                    {"ref": "vp"},
                                    {"value": 800, "units": "m/s"},
                                ],
                            },
                            {"value": 2.0},
                        ],
                    },
                ],
            },
            {"value": 600, "units": "m/s"},
            {"value": 1000, "units": "m/s"},
        ],
    }
    assert payload["depends_on"] == ["vp"]


def test_remap_expression_macro_accepts_composed_expression():
    base_vs = 0.5 * prop("Vp")
    output = Property.expr(
        remap(
            base_vs,
            from_range=(800, 1000),
            to_range=(600, 1000),
            units="m/s",
            clamp=True,
        )
    )

    payload = output.to_fs()

    affine = payload["expr"]["args"][0]
    shifted = affine["args"][1]["args"][0]
    assert payload["expr"]["op"] == "clamp"
    assert shifted == {
        "op": "sub",
        "args": [
            {
                "op": "mul",
                "args": [{"value": 0.5}, {"ref": "vp"}],
            },
            {"value": 800, "units": "m/s"},
        ],
    }
    assert payload["depends_on"] == ["vp"]


def test_remap_expression_macro_can_preserve_values_outside_source_range():
    base_vs = 0.5 * prop("Vp")
    output = Property.expr(
        remap(
            base_vs,
            from_range=(800, 1000),
            to_range=(600, 1000),
            units="m/s",
            outside="preserve",
        )
    )

    payload = output.to_fs()
    base_node = {
        "op": "mul",
        "args": [{"value": 0.5}, {"ref": "vp"}],
    }

    assert payload["expr"] == {
        "op": "case",
        "branches": [
            {
                "if": {
                    "op": "and",
                    "args": [
                        {
                            "op": ">=",
                            "args": [
                                base_node,
                                {"value": 800, "units": "m/s"},
                            ],
                        },
                        {
                            "op": "<=",
                            "args": [
                                base_node,
                                {"value": 1000, "units": "m/s"},
                            ],
                        },
                    ],
                },
                "then": {
                    "op": "add",
                    "args": [
                        {"value": 600, "units": "m/s"},
                        {
                            "op": "mul",
                            "args": [
                                {
                                    "op": "sub",
                                    "args": [
                                        base_node,
                                        {"value": 800, "units": "m/s"},
                                    ],
                                },
                                {"value": 2.0},
                            ],
                        },
                    ],
                },
            }
        ],
        "else": base_node,
    }
    assert payload["depends_on"] == ["vp"]


def test_remap_expression_macro_converts_quantity_ranges():
    expr = remap(
        prop("Vp"),
        from_range=(0.8 * u.km / u.s, 1.0 * u.km / u.s),
        to_range=(600.0 * u.m / u.s, 1.0 * u.km / u.s),
        units="m/s",
        clamp=False,
    )

    payload = expr.to_fs()

    assert payload["op"] == "add"
    assert payload["args"][0] == {"value": 600.0, "units": "m/s"}
    assert payload["args"][1]["args"][0]["args"][1] == {
        "value": 800.0,
        "units": "m/s",
    }
    assert payload["args"][1]["args"][1] == {"value": 2.0}

    with pytest.raises(ValueError, match="distinct"):
        remap(prop("Vp"), from_range=(800, 800), to_range=(600, 1000), units="m/s")
    with pytest.raises(ValueError, match="clamp or outside"):
        remap(
            prop("Vp"),
            from_range=(800, 1000),
            to_range=(600, 1000),
            units="m/s",
            clamp=True,
            outside="preserve",
        )


def test_expression_symbols_bind_variables_to_coordinate_axes():
    z = sp.Symbol("z")
    qp = Property.expr(
        sp.Piecewise(
            (60.0, z < 50.0),
            (80.0 + 0.4 * z, z < 200.0),
            (160.0, True),
            evaluate=False,
        ),
        symbols={"z": coord("interface_depth", "z", units="m")},
    )

    payload = qp.to_fs()

    assert payload == {
        "expr": {
            "op": "case",
            "branches": [
                {
                    "if": {"op": "<", "args": [{"var": "z"}, {"value": 50.0}]},
                    "then": {"value": 60.0},
                },
                {
                    "if": {"op": "<", "args": [{"var": "z"}, {"value": 200.0}]},
                    "then": {
                        "op": "add",
                        "args": [
                            {"value": 80.0},
                            {
                                "op": "mul",
                                "args": [{"value": 0.4}, {"var": "z"}],
                            },
                        ],
                    },
                },
            ],
            "else": {"value": 160.0},
        },
        "symbols": {
            "z": {
                "kind": "coordinate",
                "system": "interface_depth",
                "axis": "z",
                "units": "m",
            }
        },
    }


def test_sympy_piecewise_requires_true_fallback():
    z = sp.Symbol("z")

    with pytest.raises(ValueError, match="True fallback"):
        Property.expr(
            sp.Piecewise((60.0, z < 50.0), evaluate=False),
            symbols={"z": coord("interface_depth", "z")},
        )


def test_general_sympy_property_expression_exports_ast():
    vp, vs = sp.symbols("Vp Vs")

    payload = Property.expr(sp.Abs(vp - vs) + sp.exp(vs / vp)).to_fs()

    assert payload == {
        "expr": {
            "op": "add",
            "args": [
                {
                    "op": "abs",
                    "arg": {
                        "op": "add",
                        "args": [
                            {"ref": "vp"},
                            {
                                "op": "mul",
                                "args": [{"value": -1}, {"ref": "vs"}],
                            },
                        ],
                    },
                },
                {
                    "op": "exp",
                    "arg": {
                        "op": "mul",
                        "args": [
                            {"ref": "vs"},
                            {
                                "op": "pow",
                                "args": [{"ref": "vp"}, {"value": -1}],
                            },
                        ],
                    },
                },
            ],
        },
        "depends_on": ["vp", "vs"],
    }


def test_sympy_functions_constants_and_min_max_export_to_ast():
    vp, vs = sp.symbols("Vp Vs")

    trig = Property.expr(sp.sin(vp) + sp.cos(vs)).to_fs()["expr"]
    scaled = Property.expr(sp.pi * vp).to_fs()["expr"]
    bounded = Property.expr(sp.Min(vp, vs, 5000)).to_fs()["expr"]

    assert trig == {
        "op": "add",
        "args": [
            {"op": "cos", "arg": {"ref": "vs"}},
            {"op": "sin", "arg": {"ref": "vp"}},
        ],
    }
    assert scaled == {
        "op": "mul",
        "args": [{"value": pytest.approx(3.141592653589793)}, {"ref": "vp"}],
    }
    assert bounded == {
        "op": "min",
        "args": [{"value": 5000}, {"ref": "vp"}, {"ref": "vs"}],
    }


def test_unsupported_sympy_functions_raise_clear_error():
    vp = sp.Symbol("Vp")

    with pytest.raises(ValueError, match="Unsupported SymPy function"):
        Property.expr(sp.gamma(vp))


def test_expression_symbol_bindings_roundtrip_and_accept_coordinate_system_alias():
    data = {
        "expr": {
            "op": "<",
            "args": [{"var": "z"}, {"value": 10.0}],
        },
        "symbols": {
            "z": {
                "kind": "coordinate",
                "coordinate_system": "interface_depth",
                "axis": "z",
                "units": "meter",
            }
        },
    }

    payload = Property.from_value(data).to_fs()

    assert payload == {
        "expr": {
            "op": "<",
            "args": [{"var": "z"}, {"value": 10.0}],
        },
        "symbols": {
            "z": {
                "kind": "coordinate",
                "system": "interface_depth",
                "axis": "z",
                "units": "meter",
            }
        },
    }


def test_derived_property_expressions_roundtrip_without_mutating_input():
    data = {
        "mesh_block_id": 1,
        "properties": {
            "Vp": {"value": 1500.0, "units": "m/s"},
            "Vs": {
                "expr": {
                    "op": "mul",
                    "args": [{"value": 0.5}, {"ref": "Vp"}],
                },
                "depends_on": ["Vp"],
            },
        },
    }
    original = copy.deepcopy(data)

    subdomain = ModelSubdomain.from_fs(data)
    payload = subdomain.to_fs()

    assert data == original
    assert payload["properties"]["vs"] == {
        "expr": {
            "op": "mul",
            "args": [{"value": 0.5}, {"ref": "vp"}],
        },
        "depends_on": ["vp"],
    }


def test_coordinate_aware_values_export_to_contract_shape():
    point = CoordinateValue([1.0, 0.25], units="km", system="survey")
    direction = Direction.axis_direction("z", system="survey")

    assert point.to_fs() == {
        "value": [1.0, 0.25],
        "units": "km",
        "system": "survey",
    }
    assert direction.to_fs() == {
        "type": "coordinate_axis",
        "system": "survey",
        "axis": "z",
    }


def test_surface_coordinate_system_helpers_export_acquisition_metadata():
    surface = CoordinateSystem.surface("free_surface", surface="top", normal="up")

    assert surface.to_fs() == {
        "_type": "SurfaceCoordinateSystem",
        "name": "free_surface",
        "surface": "top",
        "axes": [{"name": "z", "direction": "z", "positive": "up"}],
        "inherit_axes": True,
    }
    roundtrip = CoordinateSystem.from_fs(surface.to_fs())
    assert isinstance(roundtrip, SurfaceCoordinateSystem)
    assert roundtrip.to_fs() == surface.to_fs()
    with pytest.raises(ValueError, match="SimpleSurfaceCoordinateSystem"):
        CoordinateSystem.from_fs(
            {
                "_type": "SimpleSurfaceCoordinateSystem",
                "name": "legacy",
                "surface": "top",
            }
        )
    assert (
        CoordinateSystem.surface(
            "offset_surface",
            surface="top",
            normal="up",
            offset=0.1,
        ).to_fs()["fixed_axis"]
        == "z"
    )
    assert surface.with_offset("offset_surface", 0.1).to_fs()["fixed_axis"] == "z"
    assert surface.on([0.5, 1.5], units="km").to_fs() == {
        "value": [[0.5, 0.0], [1.5, 0.0]],
        "units": "km",
        "system": "free_surface",
    }
    assert surface.points(np.array([0.5, 1.5]) * u.km).to_fs() == {
        "value": [[0.5, 0.0], [1.5, 0.0]],
        "units": "km",
        "system": "free_surface",
    }
    assert surface.points(np.array([0.5]) * u.km, offset=-25.0 * u.m).to_fs() == {
        "value": [[0.5, -0.025]],
        "units": "km",
        "system": "free_surface",
    }
    assert surface.above(np.array([0.5]) * u.km, 25.0 * u.m).to_fs() == {
        "value": [[0.5, 0.025]],
        "units": "km",
        "system": "free_surface",
    }
    assert surface.below(np.array([0.5]) * u.km, 25.0 * u.m).to_fs() == {
        "value": [[0.5, -0.025]],
        "units": "km",
        "system": "free_surface",
    }
    assert surface.below(np.array([[0.5, 0.025]]) * u.km).to_fs() == {
        "value": [[0.5, -0.025]],
        "units": "km",
        "system": "free_surface",
    }
    depth_surface = CoordinateSystem.surface(
        "depth_surface", surface="top", normal="down"
    )
    assert depth_surface.above(np.array([0.5]) * u.km, 25.0 * u.m).to_fs() == {
        "value": [[0.5, -0.025]],
        "units": "km",
        "system": "depth_surface",
    }
    assert depth_surface.below(np.array([0.5]) * u.km, 25.0 * u.m).to_fs() == {
        "value": [[0.5, 0.025]],
        "units": "km",
        "system": "depth_surface",
    }
    assert surface.on([[0.0, 1.0], [2.0, 3.0]], offset=[0.0, 10.0]).to_fs() == {
        "value": [[0.0, 1.0, 0.0], [2.0, 3.0, 10.0]],
        "system": "free_surface",
    }

    acq = Acquisition()
    acq.add_sources(
        kind="scalar",
        coords=surface.points(np.array([0.5]) * u.km, offset=-25.0 * u.m),
    )
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq.add_receiver_group(
        name="surface",
        device=hydrophone,
        coords=surface.on([0.0, 1.0], units="km"),
    )

    payload = acq.to_fs()
    assert payload["source_geometry"]["sources"][0]["coordinates"] == {
        "value": [0.5, -0.025],
        "units": "km",
        "system": "free_surface",
    }
    assert payload["receiver_groups"][0]["coordinates"] == {
        "_type": "CoordsArray",
        "value": [[0.0, 0.0], [1.0, 0.0]],
        "units": "km",
        "system": "free_surface",
    }


def test_surface_coordinate_system_points_grid_builds_surface_carpet():
    surface = CoordinateSystem.surface("free_surface", surface="top", normal="up")

    assert surface.points_grid(x=[0.0, 0.25, 1.0], units="km").to_fs() == {
        "value": [[0.0, 0.0], [0.25, 0.0], [1.0, 0.0]],
        "units": "km",
        "system": "free_surface",
    }
    assert surface.points_grid(
        x=[0.0, 0.25, 1.0],
        y=[2.0, 3.0],
        units="km",
        below=25.0 * u.m,
    ).to_fs() == {
        "value": [
            [0.0, 2.0, -0.025],
            [0.25, 2.0, -0.025],
            [1.0, 2.0, -0.025],
            [0.0, 3.0, -0.025],
            [0.25, 3.0, -0.025],
            [1.0, 3.0, -0.025],
        ],
        "units": "km",
        "system": "free_surface",
    }

    with pytest.raises(ValueError, match="Specify only one of above or below"):
        surface.points_grid(x=[0.0, 1.0], above=1.0, below=1.0)


def test_simulation_registers_surface_coordinate_system(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))

    system = sim.add_surface_coordinate_system("free_surface", "top", normal="up")
    payload = sim.to_fs()

    assert system.name == "free_surface"
    assert payload["coordinate_systems"] == [
        {
            "_type": "SurfaceCoordinateSystem",
            "name": "free_surface",
            "surface": "top",
            "axes": [{"name": "z", "direction": "z", "positive": "up"}],
            "inherit_axes": True,
        }
    ]


def test_simulation_accepts_named_surface_coordinate_systems(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))

    system = SurfaceCoordinateSystem(
        name="interface_relative",
        surface="interface",
        axes=[
            Axis("offset", direction="x", origin=5.0 * u.km),
            Axis("depth", direction="z", positive="down"),
        ],
    )
    sim += system

    assert sim.coordinate_systems["interface_relative"] is system
    assert sim.coordinate_system["interface_relative"] is system
    assert sim.to_fs()["coordinate_systems"] == [
        {
            "_type": "SurfaceCoordinateSystem",
            "name": "interface_relative",
            "surface": "interface",
            "axes": [
                {
                    "name": "offset",
                    "direction": "x",
                    "origin": {"value": 5.0, "units": "km"},
                },
                {"name": "depth", "direction": "z", "positive": "down"},
            ],
            "inherit_axes": True,
        }
    ]


def test_simulation_model_surface_helper_uses_surface_offsets_in_points(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))

    top = sim.model_surface("top")
    sources = top.below(np.array([0.5]) * u.km, 25.0 * u.m)
    receivers = top.points(np.array([0.0, 1.0]) * u.km)

    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=sources)
    acq.add_receiver_group(name="surface", device=hydrophone, coords=receivers)
    sim.acquisition = acq

    payload = sim.to_fs()

    assert payload["coordinate_systems"] == [
        {
            "_type": "SurfaceCoordinateSystem",
            "name": "top",
            "surface": "top",
            "axes": [{"name": "z", "direction": "z", "positive": "up"}],
            "inherit_axes": True,
        },
    ]
    assert payload["Acquisition"]["source_geometry"]["sources"][0]["coordinates"] == {
        "value": [0.5, -0.025],
        "units": "km",
        "system": "top",
    }
    assert payload["Acquisition"]["receiver_groups"][0]["coordinates"] == {
        "_type": "CoordsArray",
        "value": [[0.0, 0.0], [1.0, 0.0]],
        "units": "km",
        "system": "top",
    }


def test_acquisition_carpet_helpers_place_dim_minus_one_coords_on_surface(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=3,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0, 0, 0], u_bound=[1, 1, 1], n=[1, 1, 1])
    )

    top = sim.model_surface("top")
    assert top.on([[0.5, 0.25]], units="km").to_fs() == {
        "value": [[0.5, 0.25, 0.0]],
        "units": "km",
        "system": "top",
    }
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq = Acquisition()
    receiver_group = acq.add_receiver_carpet(
        name="surface",
        device=hydrophone,
        surface=top,
        x=[0.0, 0.25, 1.0],
        y=[0.0, 0.5],
        units="km",
    )
    shots = acq.add_sources(
        kind="scalar",
        coords=top.points_grid(x=[0.0, 0.25, 1.0], y=[0.0, 0.5], units="km"),
    )
    sim.acquisition = acq

    payload = sim.to_fs()

    assert receiver_group.name == "surface"
    assert len(shots) == 6
    assert payload["Acquisition"]["receiver_groups"][0]["coordinates"] == {
        "_type": "CoordsArray",
        "value": [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.25, 0.5, 0.0],
            [1.0, 0.5, 0.0],
        ],
        "units": "km",
        "system": "top",
    }
    assert payload["Acquisition"]["source_geometry"]["sources"][0]["coordinates"] == {
        "value": [0.0, 0.0, 0.0],
        "units": "km",
        "system": "top",
    }
    assert payload["Acquisition"]["source_geometry"]["sources"][-1]["coordinates"] == {
        "value": [1.0, 0.5, 0.0],
        "units": "km",
        "system": "top",
    }


def test_solver_frame_key_is_not_exported_and_legacy_input_is_ignored():
    subdomain = ModelSubdomain.from_fs(
        {"mesh_block_id": 1, "frame": "reference", "properties": {"vp": 1500.0}}
    )
    assert "frame" not in subdomain.to_fs()

    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq.add_receiver_group(name="surface", device=hydrophone, coords=[[0.0, 0.0]])

    payload = acq.to_fs()
    assert "frame" not in payload["source_geometry"]["sources"][0]
    assert "frame" not in payload["receiver_groups"][0]

    legacy = Acquisition.from_fs(
        {
            "source_geometry": {
                "_type": "Inline",
                "kind": "scalar",
                "sources": [
                    {
                        "name": "shot",
                        "frame": "reference",
                        "coordinates": [0.5, 0.0],
                    }
                ],
            },
            "receiver_groups": [
                {
                    "name": "surface",
                    "device": hydrophone.to_fs(),
                    "frame": "reference",
                    "coordinates": {"_type": "CoordsArray", "coords": [[0.0, 0.0]]},
                }
            ],
        }
    )
    legacy_payload = legacy.to_fs()
    assert "frame" not in legacy_payload["source_geometry"]["sources"][0]
    assert "frame" not in legacy_payload["receiver_groups"][0]

    with pytest.raises(TypeError, match="frame"):
        acq.add_receiver_group(
            name="bad", device=hydrophone, coords=[[0.0, 0.0]], frame="reference"
        )


def test_acquisition_accepts_quantity_source_and_receiver_coordinates():
    acq = Acquisition()
    geophone = ReceiverNode(name="geophone")
    geophone.add_component(name="v_z", field="velocity", direction=[0.0, 1.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        acq.add_sources(
            kind="vector",
            coords=Q_([[0.5, 0.05]], "km"),
            direction=[0.0, 1.0],
        )
        receiver_coords = [Q_([x, 0.0], "km") for x in np.linspace(0.1, 0.9, 41)]
        acq.add_receiver_group(
            name="surface",
            device=geophone,
            coords=receiver_coords,
        )

    assert not [
        warning
        for warning in caught
        if warning.category.__name__ == "UnitStrippedWarning"
    ]

    payload = acq.to_fs()
    assert payload["source_geometry"]["sources"][0]["coordinates"] == {
        "value": [0.5, 0.05],
        "units": "km",
    }
    receiver_payload = payload["receiver_groups"][0]["coordinates"]
    assert receiver_payload["_type"] == "CoordsArray"
    assert receiver_payload["units"] == "km"
    assert receiver_payload["value"][0] == [0.1, 0.0]
    assert receiver_payload["value"][-1] == [0.9, 0.0]


def test_source_and_receiver_coordinate_arrays_are_float64():
    acq = Acquisition()
    geophone = ReceiverNode(name="geophone")
    geophone.add_component(name="v_z", field="velocity", direction=[0.0, 1.0])

    acq.add_sources(
        kind="scalar",
        coords=np.asarray([[0.125, 0.0], [0.875, 0.0]], dtype=np.float32),
        names=["left", "right"],
    )
    acq.add_distributed_source("compound", {"left": 1.0, "right": 1.0})
    acq.add_receiver_group(
        name="surface",
        device=geophone,
        coords=np.asarray([[0.1, 0.0], [0.9, 0.0]], dtype=np.float32),
    )

    assert acq.source_point_coords().dtype == np.dtype("float64")
    assert acq.receiver_groups[0].coordinates.coordinates.dtype == np.dtype("float64")


def test_unit_payload_mappings_roundtrip_without_method_values():
    receiver = ReceiverFiber.from_fs(
        {
            "_type": "ReceiverFiber",
            "components": [{"name": "eps", "field": "strain"}],
            "gauge_length": {"value": np.array(10.0), "units": "m"},
            "channel_spacing": {"value": [5.0, 10.0], "units": "m"},
        }
    )

    payload = receiver.to_fs()

    assert payload["gauge_length"] == {"value": 10.0, "units": "m"}
    assert payload["channel_spacing"] == {"value": [5.0, 10.0], "units": "m"}
    json.dumps(payload)


def test_acquisition_accepts_array_quantity_receiver_coordinates():
    acq = Acquisition()
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")

    receiver_coords = Q_([[x, 50.0] for x in np.linspace(0.0, 1000.0, 3)], "m")
    acq.add_receiver_group(name="surface", device=hydrophone, coords=receiver_coords)

    payload = acq.to_fs()
    receiver_payload = payload["receiver_groups"][0]["coordinates"]
    assert receiver_payload == {
        "_type": "CoordsArray",
        "value": [[0.0, 50.0], [500.0, 50.0], [1000.0, 50.0]],
        "units": "m",
    }


def test_ricker_wavelet_defaults_center_to_one_period():
    wavelet = RickerWavelet(f=10.0)

    assert wavelet.center == pytest.approx(0.1)

    times = np.linspace(0.0, 0.5, 501)
    signal = wavelet.evaluate(times)
    peak_time = times[int(np.argmax(np.abs(signal)))] - wavelet.center

    assert peak_time == pytest.approx(0.0, abs=times[1] - times[0])


def test_ricker_wavelet_accepts_center():
    wavelet = RickerWavelet(f=20.0, center=0.2)

    assert wavelet.center == pytest.approx(0.2)


def test_ricker_wavelet_blackman_window_matches_signal_sampling():
    wavelet = RickerWavelet(f=80.0, window=("blackman", 0.2))
    times = np.linspace(0.0, 0.4, 1601)
    signal = wavelet.evaluate(times)

    assert signal.shape == times.shape
    assert len(wavelet.spectrum) == len(wavelet.frequencies)
    assert np.isfinite(wavelet.spectrum).all()

    peak = int(np.argmax(np.abs(signal)))
    assert times[peak] - wavelet.center == pytest.approx(0.0, abs=times[1] - times[0])


def test_ricker_wavelet_recenter_refreshes_signal_and_spectrum():
    wavelet = RickerWavelet(f=10.0, center=0.0)
    times = np.linspace(0.0, 1.0, 1001)
    signal_zero = wavelet.evaluate(times).copy()
    spectrum_zero = wavelet.spectrum.copy()

    with pytest.raises(AttributeError):
        wavelet.center = 0.1
    with pytest.raises(AttributeError):
        wavelet.causal = True
    with pytest.raises(AttributeError):
        wavelet.scale = 2.0
    signal_centered = wavelet.recenter(0.1)

    assert signal_centered is wavelet.signal
    assert wavelet.center == pytest.approx(0.1)
    assert wavelet.causal is False
    assert wavelet.scale == pytest.approx(1.0)
    assert int(np.argmax(np.abs(wavelet.signal))) == 100
    assert not np.array_equal(wavelet.signal, signal_zero)
    assert not np.array_equal(wavelet.spectrum, spectrum_zero)


def test_ricker_wavelet_zero_center_remains_bandlimited():
    f0 = 40.0
    center = 0.0
    times = np.linspace(0.0, 1.0, 1001)
    wavelet = RickerWavelet(f=f0, center=center)
    signal = wavelet.evaluate(times)

    assert int(np.argmax(np.abs(signal))) == 0

    spectrum = np.abs(wavelet.spectrum)
    high_frequency_level = np.mean(spectrum[-20:]) / np.max(spectrum)
    assert high_frequency_level < 1.0e-6


def test_receiver_fiber_exports_preferred_das_spacing_contract():
    das = ReceiverFiber(
        name="das",
        gauge_length=0.01,
        channel_spacing=0.0125,
        sample_spacing=0.002,
    )
    das.add_component(name="eps_tt", field="strain", direction=[1.0, 0.0])

    payload = das.to_fs()

    assert payload["_type"] == "ReceiverFiber"
    assert payload["gauge_length"] == 0.01
    assert payload["channel_spacing"] == 0.0125
    assert payload["sample_spacing"] == 0.002
    assert payload["components"][0]["field"] == "strain"
    assert "points_per_gauge" not in payload


def test_receiver_fiber_exports_points_per_gauge_when_sample_spacing_is_omitted():
    das = ReceiverFiber(gauge_length=0.01, points_per_gauge=5)

    assert das.gauge_length == 0.01
    assert das.channel_spacing == 0.01
    assert das.points_per_gauge == 5
    assert das.to_fs()["gauge_length"] == 0.01
    assert das.to_fs()["channel_spacing"] == 0.01
    assert das.to_fs()["points_per_gauge"] == 5

    loaded = ReceiverFiber.from_fs(
        {
            "name": "loaded_das",
            "components": [{"name": "eps_tt", "field": "strain"}],
            "gauge_length": 0.02,
            "channel_spacing": 0.03,
            "sample_spacing": 0.004,
            "points_per_gauge": 7,
        }
    )

    assert loaded.gauge_length == 0.02
    assert loaded.channel_spacing == 0.03
    assert loaded.sample_spacing == 0.004
    assert loaded.points_per_gauge == 7


def test_receiver_fiber_exports_quantity_lengths_with_units():
    das = ReceiverFiber(
        gauge_length=10 * u.m,
        sample_spacing=2 * u.m,
        radius=0.5 * u.m,
        pitch=25 * u.m,
    )

    payload = das.to_fs()

    assert payload["gauge_length"] == {"value": 10, "units": "m"}
    assert payload["channel_spacing"] == {"value": 10, "units": "m"}
    assert payload["sample_spacing"] == {"value": 2, "units": "m"}
    assert payload["radius"] == {"value": 0.5, "units": "m"}
    assert payload["pitch"] == {"value": 25, "units": "m"}

    explicit = ReceiverFiber(
        gauge_length=0.01 * u.km,
        channel_spacing=12.5 * u.m,
        sample_spacing=2 * u.m,
    ).to_fs()

    assert explicit["gauge_length"] == {"value": 0.01, "units": "km"}
    assert explicit["channel_spacing"] == {"value": 12.5, "units": "m"}


def test_receiver_device_name_is_optional_and_omitted_when_absent():
    node = ReceiverNode()
    node.add_component(name="p", field="pressure")

    node_payload = node.to_fs()

    assert node.name is None
    assert node_payload["_type"] == "ReceiverNode"
    assert "name" not in node_payload
    assert node_payload["components"][0]["field"] == "pressure"

    das = ReceiverFiber(gauge_length=0.01, sample_spacing=0.002)
    das.add_component(name="eps_tt", field="strain", direction=[1.0, 0.0])

    das_payload = das.to_fs()

    assert das.name is None
    assert das_payload["_type"] == "ReceiverFiber"
    assert "name" not in das_payload
    assert das_payload["gauge_length"] == 0.01
    assert das_payload["channel_spacing"] == 0.01

    loaded = ReceiverNode.from_fs({"components": [{"name": "p", "field": "pressure"}]})

    assert loaded.name is None
    assert loaded.components[0].field == "pressure"


def test_named_source_encoding_replaces_compound_source_weights():
    acq = Acquisition()
    acq.add_sources(
        kind="vector",
        coords=[[0.45, 0.1], [0.55, 0.1]],
        names=["left", "right"],
        direction=[0.0, 1.0],
    )
    acq.add_distributed_source("dipole_like", {"left": 1.0, "right": -1.0})

    payload = acq.to_fs()

    assert payload["source_geometry"]["defaults"]["direction"] == [0.0, 1.0]
    assert payload["source_encoding"]["fields"][0]["terms"] == [
        {"source": "left", "coefficient": 1.0},
        {"source": "right", "coefficient": -1.0},
    ]


def test_wavefield_output_uses_grid_contract():
    output = WavefieldOutput(
        name="movie",
        field="pressure",
        dims=("z", "r"),
        coords={
            "z": [0.0, 0.1, 0.25, 0.5],
            "r": [0.0, 0.2, 0.5],
        },
        units="km",
    )

    payload = output.to_fs()
    assert output.name == "movie"
    assert output.fields == ["pressure"]
    assert output.grid["dims"] == ["z", "r"]
    assert payload["_type"] == "WavefieldOutput"
    assert payload["field"] == "pressure"
    assert "fields" not in payload
    assert payload["grid"]["_type"] == "XArrayGrid"
    assert payload["grid"]["dims"] == ["z", "r"]
    assert payload["grid"]["coords"]["z"]["data"] == [0.0, 0.1, 0.25, 0.5]
    assert payload["grid"]["coords"]["r"]["data"] == [0.0, 0.2, 0.5]
    assert payload["grid"]["units"] == "km"


def test_wavefield_output_accepts_grid_object_and_sources():
    grid = xr.DataArray(
        np.empty((3, 2)),
        dims=("z", "r"),
        coords={"z": [0.0, 0.25, 1.0], "r": [0.0, 0.4]},
    )
    grid.coords["z"].attrs["units"] = "m"
    grid.coords["r"].attrs["units"] = "m"

    output = WavefieldOutput(
        "elastic_movie",
        grid=grid,
        fields=["velocity", "stress"],
        sources=[2],
    )
    payload = output.to_fs()

    assert output.grid["dims"] == ["z", "r"]
    assert output.fields == ["velocity", "stress"]
    assert payload["sources"] == [2]
    assert payload["fields"] == ["velocity", "stress"]
    assert payload["grid"]["_type"] == "XArrayGrid"
    assert payload["grid"]["coords"]["z"]["data"] == [0.0, 0.25, 1.0]
    assert payload["grid"]["coords"]["r"]["data"] == [0.0, 0.4]
    assert payload["grid"]["coords"]["z"]["units"] == "m"


def test_wavefield_output_accepts_receiver_device_components():
    grid = xr.DataArray(
        np.empty((2, 2)),
        dims=("z", "r"),
        coords={"z": [0.0, 1.0], "r": [0.0, 1.0]},
    )
    device = ReceiverNode(name="elastic_wavefield")
    device.add_component(name="vz", field="velocity", direction=[0.0, 1.0])
    device.add_component(name="szz", field="stress")

    output = WavefieldOutput("elastic_movie", grid=grid, device=device, sources=[1])
    payload = output.to_fs()
    loaded = WavefieldOutput.from_fs(payload)

    assert output.fields == ["velocity", "stress"]
    assert output.component_names == ["vz", "szz"]
    assert "fields" not in payload
    assert payload["device"]["_type"] == "ReceiverNode"
    assert payload["device"]["components"][0]["name"] == "vz"
    assert payload["device"]["components"][0]["field"] == "velocity"
    assert loaded.fields == ["velocity", "stress"]
    assert loaded.component_names == ["vz", "szz"]
    with pytest.raises(ValueError, match="either device or field/fields"):
        WavefieldOutput("bad", grid=grid, device=device, field="pressure")


def test_elastic_velocity_is_canonical_and_displacement_is_removed():
    assert ElasticComponents.primary == ["velocity", "stress"]
    assert ElasticComponents.check_components(["velocity", "stress"]) == [
        "velocity",
        "stress",
    ]
    with pytest.raises(ValueError, match="displacement"):
        ElasticComponents.check_components(["displacement", "stress"])

    component = ReceiverComponent(name="vz", field="velocity")
    assert component.field == "velocity"
    assert component.to_fs()["field"] == "velocity"

    paraview = ParaviewOutput(fields=["velocity", "stress"])
    assert paraview.fields == ["velocity", "stress"]
    assert paraview.to_fs()["fields"] == [
        "velocity",
        "stress",
    ]
    wavefield_grid = xr.DataArray(
        np.empty((2, 2)),
        dims=("z", "r"),
        coords={"z": [0.0, 1.0], "r": [0.0, 1.0]},
    )
    assert WavefieldOutput(fields=["velocity"], grid=wavefield_grid).fields == [
        "velocity"
    ]
    grid_payload = WavefieldOutput(fields=["velocity"], grid=wavefield_grid).to_fs()[
        "grid"
    ]
    assert WavefieldOutput.from_fs(
        {"fields": ["velocity"], "grid": grid_payload}
    ).fields == ["velocity"]


def test_new_physics_component_sets_are_available():
    assert PoroelasticComponents.check_components(
        ["velocity", "fluid_velocity", "pressure"]
    ) == ["velocity", "fluid_flux", "pressure"]
    assert CoupledAEPComponents.check_components(
        ["pressure", "velocity", "fluid_velocity", "stress"]
    ) == ["pressure", "velocity", "fluid_flux", "stress"]
    assert EMComponents.check_components(["electric", "magnetic"]) == [
        "electric",
        "magnetic",
    ]


def test_simulation_accepts_new_physics_dimension_and_axisymmetry(tmp_path):
    sim = SeismicSimulation(
        name="elastic_axisym",
        physics="elastic",
        dimension="2D",
        axisymmetric=True,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))

    payload = sim.to_fs()

    assert sim.physics == "elastic_axisym"
    assert sim.dimension == 2
    assert sim.model.dimension == 2
    assert payload["physics"] == "elastic_axisym"
    assert payload["dimension"] == 2
    assert payload["axisymmetric"] is True
    assert payload["Model"]["dimension"] == 2


def test_acoustic_axisymmetric_physics_normalizes_to_solver_key(tmp_path):
    sim = SeismicSimulation(
        name="acoustic_axisym",
        physics="acoustic",
        dimension=2,
        axisymmetric=True,
        project_path=tmp_path,
    )

    assert sim.physics == "acoustic_axisym"
    assert sim.axisymmetric is True


def test_axisymmetric_assignment_updates_solver_physics(tmp_path):
    sim = SeismicSimulation(
        name="acoustic",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )

    assert "axisymmetric" not in sim.__dict__
    assert sim.physics == "acoustic"
    assert sim.axisymmetric is False

    sim.axisymmetric = True
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))

    assert sim.physics == "acoustic_axisym"
    assert sim.axisymmetric is True
    assert sim.to_fs()["physics"] == "acoustic_axisym"
    assert sim.to_fs()["axisymmetric"] is True

    sim.axisymmetric = False

    assert sim.physics == "acoustic"
    assert sim.axisymmetric is False
    assert sim.to_fs()["physics"] == "acoustic"
    assert "axisymmetric" not in sim.to_fs()


def test_explicit_axisymmetric_physics_sets_axisymmetric_flag(tmp_path):
    sim = SeismicSimulation(
        name="acoustic_axisym",
        physics="acoustic_axisym",
        dimension=2,
        project_path=tmp_path,
    )

    assert sim.physics == "acoustic_axisym"
    assert sim.axisymmetric is True
    assert components_for_physics("acoustic_axisym") is not None


def test_private_path_api_warns_as_deprecated(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )

    with pytest.warns(DeprecationWarning, match="_set_path"):
        sim._set_path(tmp_path, Path("simulations"))
    with pytest.warns(DeprecationWarning, match="_path"):
        assert sim._path == tmp_path / "simulations" / "simple"


def test_project_new_simulation_accepts_coupled_aep_physics(tmp_path):
    project = Project(name="project", path=tmp_path)
    sim = project.new_simulation(
        name="coupled_aep",
        physics="coupled-aep",
        dimension=2,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))

    assert sim.physics == "coupled_aep"
    assert sim.to_fs()["physics"] == "coupled_aep"
    assert components_for_physics("coupled_aep") is CoupledAEPComponents


def test_axisymmetric_rejects_unsupported_physics(tmp_path):
    with pytest.raises(ValueError, match="acoustic, elastic, and coupled"):
        SeismicSimulation(
            name="bad",
            physics="poro",
            dimension=2,
            axisymmetric=True,
            project_path=tmp_path,
        )


def test_project_new_simulation_preserves_typed_and_extra_options(tmp_path):
    project = Project(name="project", path=tmp_path)
    sim = project.new_simulation(
        name="em_model",
        physics="electromagnetic",
        dimension="3D",
        l2_projection={"enabled": True},
    )

    assert sim.physics == "EM"
    assert sim.dimension == 3
    assert sim.extra["l2_projection"] == {"enabled": True}


def test_project_new_simulation_accepts_default_units(tmp_path):
    project = Project(name="project", path=tmp_path)
    sim = project.new_simulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        units={
            "length": u.km,
            "velocity": "km/s",
            "density": u.g / u.cm**3,
        },
    )

    assert sim.units.to_fs()["Units"]["defaults"] == {
        "length": "km",
        "velocity": "km/s",
        "density": "g/cm^3",
    }

    sim_with_explicit_name = project.new_simulation(
        name="simple_default_units",
        physics="acoustic",
        dimension=2,
        default_units={"length": "m"},
    )
    assert sim_with_explicit_name.units.to_fs()["Units"]["defaults"] == {"length": "m"}


def test_axisymmetric_rejects_3d_domain(tmp_path):
    with pytest.raises(ValueError, match="dimension=2"):
        SeismicSimulation(
            name="bad",
            physics="elastic",
            dimension=3,
            axisymmetric=True,
            project_path=tmp_path,
        )


def test_axisymmetric_rejects_25d_domain(tmp_path):
    with pytest.raises(ValueError, match="dimension=2"):
        SeismicSimulation(
            name="bad",
            physics="elastic",
            dimension=2.5,
            axisymmetric=True,
            project_path=tmp_path,
        )


def test_layered_model_treats_25d_as_2d_model():
    model = LayeredModel(dimension="2.5D", x_limits=[0.0, 1.0])

    assert model.dimension == 2


def test_layered_model_infers_non_interface_surface_from_ordering():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer", properties={"vp": 1500.0, "rho": 1000.0})
    model.add_surface(name="marker", depth=0.2)
    model.add_surface(name="bottom", depth=0.5)

    payload = model.to_fs()

    assert payload["surfaces"][1]["name"] == "marker"
    assert payload["surfaces"][1]["interface"] is False
    assert payload["surfaces"][2]["interface"] is True


def test_output_paths_must_be_relative_to_result_directory():
    with pytest.raises(ValueError, match="relative"):
        TraceOutput(path="/tmp/traces")
    with pytest.raises(ValueError, match="relative"):
        ParaviewOutput(path="/tmp/paraview")
    with pytest.raises(ValueError, match="relative"):
        WavefieldOutput(path="/tmp/wavefields")


def test_trace_dataset_long_domain_methods_match_short_aliases():
    class DummyStore:
        def read_FD(self, *args, **kwargs):
            return ("fd", args, kwargs)

        def read_TD(self, *args, **kwargs):
            return ("td", args, kwargs)

        def read_LD(self, *args, **kwargs):
            return ("ld", args, kwargs)

    traces = TraceDataset.__new__(TraceDataset)
    traces._store = DummyStore()
    wavelet = object()

    assert traces.frequency_domain("surface", "p", source=7) == traces.fd(
        "surface",
        "p",
        source=7,
    )
    assert traces.time_domain("surface", "p", 7, wavelet) == traces.td(
        "surface",
        "p",
        7,
        wavelet,
    )
    assert traces.laplace_domain("surface", "p", 7, wavelet) == traces.ld(
        "surface",
        "p",
        7,
        wavelet,
    )


def test_job_outputs_adds_outputs_and_always_exports_traces():
    outputs = JobOutputs()

    outputs += ParaviewOutput(name="pv", fields=["pressure"])
    payload = outputs.to_fs()

    assert payload["traces"]["path"] == "traces"
    assert payload["ParaView"][0]["name"] == "pv"


def test_job_outputs_make_named_outputs_unique():
    grid = xr.DataArray(
        np.zeros((2, 2)),
        dims=("z", "r"),
        coords={"z": [0.0, 1.0], "r": [0.0, 1.0]},
    )
    config = JobOutputs()

    config += [
        ParaviewOutput(name="snapshot", fields=["pressure"]),
        ParaviewOutput(name="snapshot", fields=["velocity"]),
        WavefieldOutput(name="snapshot", field="pressure", grid=grid),
        WavefieldOutput(name="snapshot", field="velocity", grid=grid),
    ]
    first_payload = config.to_fs()
    second_payload = config.to_fs()

    assert [out["name"] for out in first_payload["ParaView"]] == [
        "snapshot",
        "snapshot_1",
    ]
    assert [out["name"] for out in first_payload["wavefields"]] == [
        "snapshot_2",
        "snapshot_3",
    ]
    assert second_payload == first_payload


def test_output_config_helper_exports_units_paraview_and_wavefields():
    grid = xr.DataArray(
        np.zeros((2, 2)),
        dims=("z", "r"),
        coords={"z": [0.0, 1.0], "r": [0.0, 2.0]},
        attrs={"units": "m"},
    )

    config = outputs(
        units=OutputUnits(
            geometry="ft",
            pressure="psi",
            fields={"acoustic:pressure": "psi"},
            properties={"Vp": "ft/s"},
        ),
        traces="trace_products",
        paraview=paraview.surface(
            "quicklook",
            shell=True,
            items=[
                field("pressure", output_name="p", units="psi", parts="magnitude"),
                output_property("Vp", units="ft/s"),
                info("Domain"),
            ],
        ),
        wavefields=wavefield("pressure", grid=grid),
    )

    payload = config.to_fs()

    assert payload["Units"] == {
        "geometry": "ft",
        "dimensions": {"pressure": "psi"},
        "fields": {"acoustic:pressure": "psi"},
        "properties": {"Vp": "ft/s"},
    }
    assert payload["traces"]["path"] == "trace_products"
    assert payload["ParaView"][0]["name"] == "quicklook"
    assert payload["ParaView"][0]["target"]["selection"] == [{"kind": "shell"}]
    assert payload["ParaView"][0]["items"] == [
        {
            "kind": "field",
            "field": "pressure",
            "name": "p",
            "units": "psi",
            "parts": ["abs"],
        },
        {"kind": "property", "property": "Vp", "units": "ft/s"},
        {"kind": "info", "info": "Domain"},
    ]
    assert payload["wavefields"][0]["name"] == "pressure_wavefield"
    assert payload["wavefields"][0]["field"] == "pressure"
    assert "fields" not in payload["wavefields"][0]


def test_paraview_factory_supports_alias_and_structured_items():
    output = paraview.volume(
        "volume",
        items=[
            paraview.field("velocity", basis=["x", "z"], units="m/s"),
            paraview.prop("Rho", units="g/cc"),
        ],
        format="xmf",
    )

    payload = output.to_fs()

    assert isinstance(output, ParaViewOutput)
    assert payload["writer"] == {"format": "xdmf", "encoding": "hdf5"}
    assert payload["target"] == {"kind": "volume"}
    assert payload["items"][0]["basis"] == {
        "type": "coordinate_basis",
        "system": "global",
        "components": ["x", "z"],
    }
    assert payload["items"][1] == {
        "kind": "property",
        "property": "Rho",
        "units": "g/cc",
    }


def test_outputs_units_accepts_shorthand_mapping():
    payload = outputs(
        units={
            "geometry": "ft",
            "pressure": "psi",
            "properties": {"Vp": "ft/s"},
        },
    ).to_fs()

    assert payload["Units"] == {
        "geometry": "ft",
        "dimensions": {"pressure": "psi"},
        "properties": {"Vp": "ft/s"},
    }


def test_job_add_output_exports_outputs_from_job_json(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job += [ParaviewOutput(name="pv", fields=["pressure"])]
    payload = job.to_fs()

    assert "Outputs" not in sim.to_fs()
    assert payload["Outputs"]["traces"]["path"] == "traces"
    assert payload["Outputs"]["ParaView"][0]["name"] == "pv"


def test_job_output_convenience_methods_export_contract(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.output_units(geometry="ft", pressure="psi")
    job.traces(path="trace_products")
    job.paraview("quicklook", fields="pressure")

    payload = job.to_fs()["Outputs"]

    assert payload["Units"] == {
        "geometry": "ft",
        "dimensions": {"pressure": "psi"},
    }
    assert payload["traces"]["path"] == "trace_products"
    assert payload["ParaView"][0]["name"] == "quicklook"
    assert payload["ParaView"][0]["fields"] == ["pressure"]


def test_paraview_output_requires_single_frequency_job(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    job = FrequencyDomainJob(
        name="freq",
        simulation=sim,
        f_list=[10.0, 20.0],
        outputs=ParaviewOutput(name="pv", fields=["pressure"]),
    )

    with pytest.raises(ValueError, match="single-frequency"):
        job.to_fs()


def test_paraview_surface_output_exports_solver_contract():
    output = ParaviewOutput.surface(
        name="free_surface",
        surfaces="top",
        fields=["pressure"],
        properties=["Vp"],
        format="xdmf",
        execute_on="final",
        coordinates="global",
        order=3,
        upscale=2,
        show_pml=False,
    )

    payload = output.to_fs()

    assert payload["execute_on"] == "final"
    assert payload["writer"] == {"format": "xdmf", "encoding": "hdf5"}
    assert "source" not in payload
    assert payload["coordinates"] == {"system": "global"}
    assert payload["target"] == {
        "kind": "surface",
        "mesh": {"order": 3, "upscale": 2, "show_pml": False},
        "selection": [{"kind": "model_surface", "name": "top"}],
    }
    assert payload["fields"] == ["pressure"]
    assert payload["properties"] == ["Vp"]


def test_paraview_volume_constructor_exports_solver_contract():
    output = ParaviewOutput.volume(
        name="volume",
        fields=["pressure"],
        parts="real",
        upscale=1,
    )

    payload = output.to_fs()

    assert payload["target"] == {"kind": "volume"}
    assert payload["items"] == [
        {"kind": "field", "field": "pressure", "parts": ["real"]},
    ]
    assert "fields" not in payload
    assert "properties" not in payload


def test_paraview_defaults_to_vtu_appended_binary():
    output = ParaviewOutput(name="pv", fields=["pressure"])
    payload = output.to_fs()

    assert output.format == "vtu"
    assert payload["writer"] == {"format": "vtu", "encoding": "appended"}
    assert "target" not in payload
    assert "source" not in payload
    assert "sources" not in payload

    with pytest.raises(ValueError, match="format"):
        ParaviewOutput(name="pv", fields=["pressure"], format="vtk")


def test_paraview_sources_all_omits_solver_sources_key():
    default_payload = ParaviewOutput(name="pv", fields=["pressure"]).to_fs()
    all_payload = ParaviewOutput(
        name="pv",
        fields=["pressure"],
        sources="all",
    ).to_fs()
    scalar_payload = ParaviewOutput(
        name="pv",
        fields=["pressure"],
        sources=2,
    ).to_fs()
    array_payload = ParaviewOutput(
        name="pv",
        fields=["pressure"],
        sources=np.arange(1, 4),
    ).to_fs()

    assert "sources" not in default_payload
    assert "sources" not in all_payload
    assert scalar_payload["sources"] == [2]
    assert array_payload["sources"] == [1, 2, 3]

    with pytest.raises(ValueError, match="sources"):
        ParaviewOutput(name="pv", fields=["pressure"], sources="first")


def test_paraview_output_omits_fields_when_not_requested():
    payload = ParaviewOutput(name="pv", properties=["vp"]).to_fs()

    assert "fields" not in payload
    assert payload["properties"] == ["vp"]


def test_paraview_parts_do_not_default_missing_fields_to_all():
    payload = ParaviewOutput(name="pv", properties=["vp"], parts="real").to_fs()

    assert "fields" not in payload
    assert "properties" not in payload
    assert payload["items"] == [{"kind": "property", "property": "vp"}]


def test_paraview_parts_are_compact_and_validated():
    output = ParaviewOutput(
        name="real_pressure",
        fields=["pressure"],
        properties=["vp"],
        parts="real",
    )

    payload = output.to_fs()

    assert "fields" not in payload
    assert "properties" not in payload
    assert payload["items"] == [
        {"kind": "field", "field": "pressure", "parts": ["real"]},
        {"kind": "property", "property": "vp"},
    ]

    assert ParaviewOutput(fields=["pressure"], parts="magnitude").parts == ["abs"]

    with pytest.raises(ValueError, match="parts"):
        ParaviewOutput(fields=["pressure"], parts="phase")


def test_axis_aligned_plane_can_be_mixed_with_surface_names():
    output = ParaviewOutput.surface(
        name="surface_cuts",
        surfaces=[
            "sea_surface",
            "sea_floor",
            "bottom",
            AxisAlignedPlane(x=0.5 * u.km),
            AxisAlignedPlane(
                "offset",
                2.0,
                units="km",
                tolerance=10.0 * u.m,
            ),
        ],
        fields=["pressure"],
        upscale=0,
        order=2,
        show_pml=False,
    )

    payload = output.to_fs()

    assert payload["target"]["selection"] == [
        {"kind": "model_surface", "name": "sea_surface"},
        {"kind": "model_surface", "name": "sea_floor"},
        {"kind": "model_surface", "name": "bottom"},
        {
            "kind": "plane",
            "system": "global",
            "axis": "x",
            "value": {"value": 0.5, "units": "km"},
        },
        {
            "kind": "plane",
            "system": "global",
            "axis": "offset",
            "value": {"value": 2.0, "units": "km"},
            "tolerance": {"value": 10.0, "units": "m"},
        },
    ]
    assert payload["target"]["mesh"] == {
        "order": 2,
        "upscale": 0,
        "show_pml": False,
    }

    with pytest.raises(ValueError, match="exactly one axis"):
        AxisAlignedPlane(x=0.5 * u.km, z=1.0 * u.km)


def test_paraview_grid_and_plane_selection_serialize_with_units():
    grid = {
        "system": "global",
        "axes": [
            {"name": "x", "value": [0.0, 1.0], "units": "km"},
            {"name": "z", "value": [0.0, 0.5], "units": "km"},
        ],
    }
    output = ParaviewOutput.surface(
        name="depth_slice",
        plane={"axis": "z", "value": 0.5 * u.km, "tolerance": 10.0 * u.m},
        fields=["pressure"],
    )

    payload = output.to_fs()

    assert "source" not in payload
    assert payload["writer"] == {"format": "vtu", "encoding": "appended"}
    assert payload["target"]["selection"] == [
        {
            "kind": "plane",
            "system": "global",
            "axis": "z",
            "value": {"value": 0.5, "units": "km"},
            "tolerance": {"value": 10.0, "units": "m"},
        }
    ]

    grid_output = ParaviewOutput.grid(
        grid,
        fields=["pressure"],
    )

    grid_payload = grid_output.to_fs()
    assert grid_payload["target"] == {"kind": "grid", "grid": grid}
    assert "source" not in grid_payload
    assert grid_payload["writer"] == {"format": "vtu", "encoding": "appended"}


def test_paraview_from_fs_preserves_new_blocks_and_extra():
    data = {
        "_type": "ParaviewOutput",
        "name": "surface",
        "path": "pv",
        "target": {
            "kind": "surface",
            "selection": [{"kind": "boundary", "labels": ["free"]}],
        },
        "writer": {"format": "xdmf", "distribution": "root"},
        "source": {"kind": "internal_future_source"},
        "items": [{"field": "pressure"}],
        "advanced_solver_flag": True,
    }

    output = ParaviewOutput.from_fs(data)
    data["target"]["kind"] = "volume"
    payload = output.to_fs()

    assert payload["target"]["kind"] == "surface"
    assert payload["target"]["selection"] == [{"kind": "boundary", "labels": ["free"]}]
    assert payload["writer"] == {
        "format": "xdmf",
        "encoding": "hdf5",
        "distribution": "root",
    }
    assert payload["source"] == {"kind": "internal_future_source"}
    assert payload["items"] == [{"kind": "field", "field": "pressure"}]
    assert "fields" not in payload
    assert payload["advanced_solver_flag"] is True


def test_extra_fields_are_preserved_but_cannot_collide():
    subdomain = ModelSubdomain(
        mesh_block_id=1,
        properties={"vp": 1500},
        extra={"solver_material_tag": "sediment"},
    )
    assert subdomain.to_fs()["solver_material_tag"] == "sediment"

    subdomain.extra["properties"] = {}
    with pytest.raises(ValueError, match="collide"):
        subdomain.to_fs()


def test_unit_config_round_trips_defaults_scales_and_extra():
    config = UnitConfig.from_fs(
        {
            "f0": 8.0,
            "Units": {
                "defaults": {"velocity": "km/s"},
                "scales": {"f0": "Hz"},
                "solver_unit_mode": "native",
            },
            "disable_scaling": False,
        }
    )

    assert config.to_fs() == {
        "disable_scaling": False,
        "f0": 8.0,
        "Units": {
            "solver_unit_mode": "native",
            "scales": {"f0": "Hz"},
            "defaults": {"velocity": "km/s"},
        },
    }


def test_simulation_export_includes_units_coordinates_and_extra_without_mutating_load_input(
    tmp_path,
):
    data = {
        "_type": "SeismicSimulation",
        "name": "simple",
        "project_path": str(tmp_path),
        "physics": "acoustic",
        "dimension": 2,
        "f0": 8.0,
        "Units": {"defaults": {"velocity": "km/s"}},
        "global_coordinate_system": {"type": "cartesian", "name": "global"},
        "coordinate_systems": [{"type": "cylindrical", "name": "well"}],
        "Model": {
            "_type": "ModelBase",
            "name": "model",
            "dimension": 2,
            "subdomains": [
                {
                    "mesh_block_id": 1,
                    "properties": {
                        "Vp": {"value": 1.5, "units": "km/s"},
                        "Rho": {"value": 2.2, "units": "g/cm^3"},
                    },
                }
            ],
        },
        "Mesh": {
            "generator": {
                "_type": "HexMeshGenerator",
                "l_bound": [0, 0],
                "u_bound": [1, 1],
                "n": [1, 1],
                "units": "km",
            },
            "adapt": {"elems_per_wave": 2.0},
        },
        "advanced_solver_flag": True,
    }
    original = copy.deepcopy(data)

    sim = SeismicSimulation.from_fs(data)
    payload = sim.to_fs()

    assert data == original
    assert payload["schema"] == "fs-simulation-1"
    assert payload["Units"]["defaults"]["velocity"] == "km/s"
    assert payload["global_coordinate_system"] == {
        "type": "cartesian",
        "name": "global",
    }
    assert payload["coordinate_systems"] == [{"type": "cylindrical", "name": "well"}]
    assert payload["advanced_solver_flag"] is True
    assert payload["Model"]["subdomains"][0]["properties"]["vp"] == {
        "value": 1.5,
        "units": "km/s",
    }


def test_manual_simulation_export_with_new_api(tmp_path):
    model = ModelBase(name="model", dimension=2)
    model += ModelSubdomain(
        mesh_block_id=1,
        properties={"Vp": 1.5 * u.km / u.s, "Rho": 2.2 * u.g / u.cm**3},
    )

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.model = model
    sim.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1], units="km")
    )
    sim.global_coordinate_system = CoordinateSystem.cartesian(name="global")

    payload = sim.to_fs()

    assert payload["schema"] == "fs-simulation-1"
    assert payload["Mesh"]["generator"]["units"] == "km"
    assert payload["Model"]["subdomains"][0]["properties"]["rho"]["units"] == "g/cm^3"


def test_mesh_surface_gradings_export_to_fast_solver_contract():
    mesh = MeshManager(
        HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1], units="km")
    )
    mesh.set_adapt(
        elems_per_wave={"x": 2.0, "z": 3.0},
        f_low=10.0,
        f_high=30.0,
        surface_gradings=[
            SurfaceGrading(
                surface="top",
                d0=0.0,
                d1=0.05,
                factor=2.5,
                power=2.0,
            ),
            {
                "surface": "interface",
                "mode": "inside",
                "d0": 0.02,
                "d1": 0.1,
                "factor_max": 4.0,
                "factor_min": 1.25,
                "power": 0.5,
                "phi_scale": -1.0,
            },
        ],
    )

    payload = mesh.to_fs()

    assert payload["adapt"]["f_low"] == 10.0
    assert payload["adapt"]["f_high"] == 30.0
    assert "f_adapt" not in payload["adapt"]
    assert payload["adapt"]["surface_gradings"] == [
        {
            "surface": "top",
            "mode": "abs_band",
            "d0": 0.0,
            "d1": 0.05,
            "factor": 2.5,
            "power": 2.0,
        },
        {
            "surface": "interface",
            "mode": "inside",
            "d0": 0.02,
            "d1": 0.1,
            "factor_max": 4.0,
            "factor_min": 1.25,
            "power": 0.5,
            "phi_scale": -1.0,
        },
    ]


def test_mesh_source_receiver_gradings_export_to_fast_solver_contract():
    mesh = MeshManager(
        HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1], units="km")
    )
    mesh.set_adapt(
        epw=2.0,
        source_grading=DistanceGrading(
            d0=0.01,
            d1=0.08,
            factor=4.0,
            power=2.0,
        ),
        receiver_grading={"d0": 0.02, "d1": 0.12, "factor": 3.0, "power": 0.5},
    )

    payload = mesh.to_fs()

    assert payload["adapt"]["src_grading"] == {
        "d0": 0.01,
        "d1": 0.08,
        "factor": 4.0,
        "power": 2.0,
    }
    assert payload["adapt"]["rcv_grading"] == {
        "d0": 0.02,
        "d1": 0.12,
        "factor": 3.0,
        "power": 0.5,
    }


def test_mesh_gradings_accept_pint_distance_units():
    mesh = MeshManager(
        HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1], units="km")
    )
    mesh.set_adapt(
        epw=2.0,
        source_grading=DistanceGrading(
            d0=10.0 * u.m,
            d1=80.0 * u.m,
            factor=4.0,
        ),
        receiver_grading={
            "d0": 0.01 * u.km,
            "d1": 0.06 * u.km,
            "factor": 2.0,
        },
        surface_gradings=[
            SurfaceGrading(
                surface="interface",
                d0=5.0 * u.m,
                d1=25.0 * u.m,
                factor=2.0,
            )
        ],
    )

    payload = mesh.to_fs()["adapt"]

    assert payload["src_grading"] == {
        "d0": {"value": 10.0, "units": "m"},
        "d1": {"value": 80.0, "units": "m"},
        "factor": 4.0,
    }
    assert payload["rcv_grading"] == {
        "d0": {"value": 0.01, "units": "km"},
        "d1": {"value": 0.06, "units": "km"},
        "factor": 2.0,
    }
    assert payload["surface_gradings"][0]["d0"] == {"value": 5.0, "units": "m"}
    assert payload["surface_gradings"][0]["d1"] == {"value": 25.0, "units": "m"}


def test_mesh_gradings_default_to_linear_power_and_require_positive_power():
    assert DistanceGrading(d0=0.0, d1=1.0, factor=2.0).power == 1.0
    assert SurfaceGrading(surface="interface", d0=0.0, d1=1.0, factor=2.0).power == 1.0
    assert "power" not in DistanceGrading(d0=0.0, d1=1.0, factor=2.0).to_fs()
    assert (
        "power"
        not in SurfaceGrading(surface="interface", d0=0.0, d1=1.0, factor=2.0).to_fs()
    )

    with pytest.raises(ValueError, match="power"):
        DistanceGrading(d0=0.0, d1=1.0, factor=2.0, power=0.0)
    with pytest.raises(ValueError, match="power"):
        SurfaceGrading(surface="interface", d0=0.0, d1=1.0, factor=2.0, power=-1.0)
    with pytest.raises(ValueError, match="power"):
        DistanceGrading(
            d0=0.0,
            d1=1.0,
            factor={"offset": 2.0, "depth": 1.5},
            power={"offset": 1.0, "depth": 0.0},
        )


def test_mesh_gradings_accept_axis_mapped_factor_and_power():
    mesh = MeshManager()
    mesh.set_adapt(
        elems_per_wave={"offset": 2.0, "depth": 3.0},
        order={"offset": 4, "depth": 3},
        source_grading=DistanceGrading(
            d0=0.01,
            d1=0.08,
            factor={"offset": 4.0, "depth": 2.0},
            power={"offset": 2.0, "depth": 1.5},
        ),
        receiver_grading={
            "d0": 0.02,
            "d1": 0.12,
            "factor": {"offset": 3.0, "depth": 1.5},
            "power": {"offset": 1.25, "depth": 2.0},
        },
        surface_gradings=[
            {
                "surface": "interface",
                "d1": 0.05,
                "factor_max": {"offset": 4.0, "depth": 2.0},
                "factor_min": {"offset": 1.0, "depth": 1.0},
                "power": {"offset": 2.0, "depth": 1.2},
            }
        ],
    )
    mesh.set_source_grading(
        d0=0.0,
        d1=25.0,
        factor={"offset": 2.0, "depth": 1.5},
        power={"offset": 1.0, "depth": 2.0},
    )
    mesh.add_surface_grading(
        "free_surface",
        d1=10.0,
        factor={"offset": 2.0, "depth": 1.25},
        power={"offset": 1.5, "depth": 2.0},
    )

    payload = mesh.adapt.to_fs()

    assert payload["elems_per_wave"] == {"offset": 2.0, "depth": 3.0}
    assert payload["order"] == {"offset": 4, "depth": 3}
    assert payload["src_grading"] == {
        "d0": 0.0,
        "d1": 25.0,
        "factor": {"offset": 2.0, "depth": 1.5},
        "power": {"offset": 1.0, "depth": 2.0},
    }
    assert payload["rcv_grading"] == {
        "d0": 0.02,
        "d1": 0.12,
        "factor": {"offset": 3.0, "depth": 1.5},
        "power": {"offset": 1.25, "depth": 2.0},
    }
    assert payload["surface_gradings"][0] == {
        "surface": "interface",
        "mode": "abs_band",
        "d0": 0.0,
        "d1": 0.05,
        "factor_max": {"offset": 4.0, "depth": 2.0},
        "factor_min": {"offset": 1.0, "depth": 1.0},
        "power": {"offset": 2.0, "depth": 1.2},
    }
    assert payload["surface_gradings"][1] == {
        "surface": "free_surface",
        "mode": "abs_band",
        "d0": 0.0,
        "d1": 10.0,
        "factor": {"offset": 2.0, "depth": 1.25},
        "power": {"offset": 1.5, "depth": 2.0},
    }


def test_mesh_source_receiver_gradings_are_editable():
    mesh = MeshManager()
    mesh.set_adapt(elems_per_wave=2.0)
    mesh.set_source_grading(d0=0.0, d1=25.0, factor=2.0)
    mesh.set_receiver_grading(d0=5.0, d1=40.0, factor=3.0)
    mesh.adapt.source_grading.factor = 2.5
    mesh.adapt.receiver_grading.d1 = 45.0

    payload = mesh.adapt.to_fs()

    assert payload["src_grading"]["factor"] == 2.5
    assert payload["rcv_grading"]["d1"] == 45.0


def test_mesh_adapt_uses_elems_per_wave_and_accepts_epw_alias():
    mesh = MeshManager()
    mesh.set_adapt(epw=3.0)

    assert mesh.adapt.elems_per_wave == 3.0
    assert mesh.adapt.epw == 3.0
    assert mesh.adapt.min_epw == 3.0
    assert mesh.adapt.to_fs() == {"elems_per_wave": 3.0, "order": 3}
    assert MeshAdaptor.from_fs({"epw": 4.0}).to_fs() == {
        "elems_per_wave": 4.0,
        "order": 3,
    }
    assert MeshAdaptor(elems_per_wave=2.0, f_adapt=5.0, f_high=30.0).to_fs() == {
        "elems_per_wave": 2.0,
        "order": 3,
        "f_low": 5.0,
        "f_high": 30.0,
    }
    assert MeshAdaptor.from_fs(
        {"elems_per_wave": 2.0, "f_adapt": 6.0, "f_high": 20.0}
    ).to_fs() == {
        "elems_per_wave": 2.0,
        "order": 3,
        "f_low": 6.0,
        "f_high": 20.0,
    }

    with pytest.raises(ValueError, match="Specify only one"):
        mesh.set_adapt(elems_per_wave=2.0, epw=3.0)
    with pytest.raises(ValueError, match="Specify only one"):
        mesh.set_adapt(elems_per_wave=2.0, f_low=5.0, f_adapt=6.0)


def test_mesh_adapt_order_accepts_branch_expression():
    epw = sp.Symbol("epw")
    p_order = sp.Piecewise(
        (2, epw > 4.0),
        (3, epw > 3.0),
        (4, True),
        evaluate=False,
    )
    mesh = MeshManager()
    mesh.set_adapt(epw=2.0, order=p_order, hp={"order": p_order})

    payload = mesh.adapt.to_fs()
    expected = {
        "op": "case",
        "branches": [
            {
                "if": {"op": ">", "args": [{"var": "epw"}, {"value": 4.0}]},
                "then": {"value": 2},
            },
            {
                "if": {"op": ">", "args": [{"var": "epw"}, {"value": 3.0}]},
                "then": {"value": 3},
            },
        ],
        "else": {"value": 4},
    }
    hp_expected = {
        "op": "case",
        "branches": [
            {
                "if": {"op": ">", "args": [{"var": "epw"}, {"value": 4.0}]},
                "then": 2,
            },
            {
                "if": {"op": ">", "args": [{"var": "epw"}, {"value": 3.0}]},
                "then": 3,
            },
        ],
        "else": 4,
    }

    assert payload["order"] == expected
    assert payload["hp"]["order"] == hp_expected


def test_mesh_hp_adaptivity_serializes_order_ranges_and_axis_policies():
    epw = sp.Symbol("epw")
    global_order = sp.Piecewise(
        (2, epw > 4.0),
        (3, epw > 3.0),
        (4, True),
        evaluate=False,
    )
    z_order = sp.Piecewise(
        (5, epw > 5.0),
        (4, True),
        evaluate=False,
    )
    mesh = MeshManager()
    mesh.set_adapt(
        epw=2.0,
        hp={
            "order": global_order,
            "order_x": {"min": 2, "max": 4},
            "order_y": {"min": 2, "max": 3},
            "order_z": {"policy": z_order, "min": 3, "max": 5},
        },
    )

    payload = mesh.adapt.to_fs()["hp"]

    assert payload == {
        "order": {
            "op": "case",
            "branches": [
                {
                    "if": {"op": ">", "args": [{"var": "epw"}, {"value": 4.0}]},
                    "then": 2,
                },
                {
                    "if": {"op": ">", "args": [{"var": "epw"}, {"value": 3.0}]},
                    "then": 3,
                },
            ],
            "else": 4,
        },
        "order_x": {"min": 2, "max": 4},
        "order_y": {"min": 2, "max": 3},
        "order_z": {
            "policy": {
                "op": "case",
                "branches": [
                    {
                        "if": {"op": ">", "args": [{"var": "epw"}, {"value": 5.0}]},
                        "then": 5,
                    },
                ],
                "else": 4,
            },
            "min": 3,
            "max": 5,
        },
    }


def test_mesh_hp_adaptivity_serializes_physics_order_overrides():
    epw = sp.Symbol("epw")
    default_order = sp.Piecewise(
        (2, epw > 4.0),
        (4, True),
        evaluate=False,
    )
    acoustic_order = sp.Piecewise(
        (3, epw > 2.5),
        (5, True),
        evaluate=False,
    )
    mesh = MeshManager()
    mesh.set_adapt(
        epw=2.0,
        hp={
            "order": default_order,
            "overrides": [
                {
                    "classifier": "physics",
                    "value": "acoustic",
                    "order": acoustic_order,
                },
            ],
            "order_x": {"policy": 6, "min": 2, "max": 6},
        },
    )

    payload = mesh.adapt.to_fs()["hp"]

    assert payload == {
        "order": {
            "op": "case",
            "branches": [
                {
                    "if": {"op": ">", "args": [{"var": "epw"}, {"value": 4.0}]},
                    "then": 2,
                },
            ],
            "else": 4,
        },
        "overrides": [
            {
                "classifier": "physics",
                "value": "acoustic",
                "order": {
                    "op": "case",
                    "branches": [
                        {
                            "if": {
                                "op": ">",
                                "args": [{"var": "epw"}, {"value": 2.5}],
                            },
                            "then": 3,
                        },
                    ],
                    "else": 5,
                },
            },
        ],
        "order_x": {"policy": 6, "min": 2, "max": 6},
    }


def test_mesh_hp_adaptivity_rejects_unsupported_override_classifier():
    mesh = MeshManager()
    mesh.set_adapt(
        epw=2.0,
        hp={
            "order": 4,
            "overrides": [
                {
                    "classifier": "domain",
                    "value": "acoustic",
                    "order": 3,
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="classifier 'physics'"):
        mesh.adapt.to_fs()


def test_mesh_hp_p_order_alias_serializes_as_order_and_rejects_conflict():
    epw = sp.Symbol("epw")
    p_order = sp.Piecewise((2, epw > 4.0), (3, True), evaluate=False)

    mesh = MeshManager()
    mesh.set_adapt(epw=2.0, hp={"p_order": p_order})

    with pytest.warns(DeprecationWarning, match="hp.p_order"):
        payload = mesh.adapt.to_fs()

    assert "p_order" not in payload["hp"]
    assert payload["hp"]["order"] == {
        "op": "case",
        "branches": [
            {
                "if": {"op": ">", "args": [{"var": "epw"}, {"value": 4.0}]},
                "then": 2,
            },
        ],
        "else": 3,
    }

    mesh = MeshManager()
    mesh.set_adapt(epw=2.0, hp={"order": 3, "p_order": 4})

    with pytest.raises(ValueError, match="hp.order or hp.p_order"):
        mesh.adapt.to_fs()


def test_mesh_surface_gradings_accept_mapping_and_are_editable():
    mesh = MeshManager()
    mesh.set_adapt(
        elems_per_wave=2.0,
        surface_gradings={
            "fault": {"d1": 50.0, "factor": 3.0, "mode": "band"},
        },
    )
    mesh.add_surface_grading("free_surface", d1=10.0, factor=2.0)
    mesh.adapt.surface_gradings[0].factor = 4.0

    payload = mesh.adapt.to_fs()

    assert payload["surface_gradings"][0]["surface"] == "fault"
    assert payload["surface_gradings"][0]["factor"] == 4.0
    assert payload["surface_gradings"][1]["surface"] == "free_surface"


def test_mesh_surface_gradings_roundtrip_and_preserve_extra():
    data = {
        "adapt": {
            "min_epw": 2.0,
            "surface_gradings": [
                {
                    "surface": "interface",
                    "d1": 0.15,
                    "factor": 2.0,
                    "custom_solver_flag": True,
                }
            ],
            "src_grading": {"d0": 0.01, "d1": 0.08, "factor": 4.0},
            "rcv_grading": {"d0": 0.02, "d1": 0.12, "factor": 3.0},
            "adapt_sources": 1,
            "f_high": 25.0,
        },
        "generator": {
            "_type": "HexMeshGenerator",
            "l_bound": [0, 0],
            "u_bound": [1, 1],
            "n": [1, 1],
        },
    }
    original = copy.deepcopy(data)

    manager = MeshManager.from_fs(data)
    payload = manager.to_fs()

    assert data == original
    assert payload["adapt"]["elems_per_wave"] == 2.0
    assert payload["adapt"]["f_high"] == 25.0
    assert "min_epw" not in payload["adapt"]
    assert payload["adapt"]["adapt_sources"] == 1
    assert payload["adapt"]["surface_gradings"][0]["custom_solver_flag"] is True
    assert payload["adapt"]["src_grading"] == {"d0": 0.01, "d1": 0.08, "factor": 4.0}
    assert payload["adapt"]["rcv_grading"] == {"d0": 0.02, "d1": 0.12, "factor": 3.0}


def test_mesh_manager_allows_solver_default_mesh_without_generator_or_file():
    manager = MeshManager()
    manager.set_adapt(epw=3.0, order=4)

    assert manager.to_fs() == {
        "adapt": {
            "elems_per_wave": 3.0,
            "order": 4,
        }
    }


def test_mesh_manager_rejects_incomplete_file_mesh_configuration():
    with pytest.raises(ValueError, match="file.*format"):
        MeshManager(file="mesh.gmsh").to_fs()

    with pytest.raises(ValueError, match="file.*format"):
        MeshManager(format="Gmsh").to_fs()


def test_array_properties_materialize_to_simulation_hdf5_with_hash(tmp_path):
    grid = xr.DataArray(
        np.arange(6, dtype=np.float32).reshape(3, 2),
        dims=("x", "z"),
        coords={"x": [0.0, 0.5, 1.0], "z": [0.0, 1.0]},
        attrs={"units": "km/s"},
    )
    model = ModelBase(name="model", dimension=2)
    model += ModelSubdomain(mesh_block_id=1, properties={"vp": grid})

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.model = model
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))

    payload = sim.to_fs()
    prop = payload["Model"]["subdomains"][0]["properties"]["vp"]

    assert prop["format"] == "hdf5"
    assert (
        prop["file"]
        == "simulations/simple/simple.h5:inputs/model/subdomains/1/properties/vp"
    )
    assert prop["dataset"] == "inputs/model/subdomains/1/properties/vp"
    assert prop["hash"].startswith("blake3:")

    with h5py.File(tmp_path / "simulations/simple/simple.h5", "r") as h5:
        dset = h5[prop["dataset"]]
        assert dset.attrs["fs_hash"] == prop["hash"]
        assert list(dset.attrs["dims"]) == ["x", "z"]
        assert list(dset.attrs["x"]) == [0.0, 0.5, 1.0]

    with h5py.File(tmp_path / "simulations/simple/simple.h5", "a") as h5:
        h5[prop["dataset"]].attrs["sentinel"] = "kept"

    payload_again = sim.to_fs()
    assert (
        payload_again["Model"]["subdomains"][0]["properties"]["vp"]["hash"]
        == prop["hash"]
    )
    with h5py.File(tmp_path / "simulations/simple/simple.h5", "r") as h5:
        assert h5[prop["dataset"]].attrs["sentinel"] == "kept"


def test_large_receiver_coordinates_materialize_to_simulation_hdf5(tmp_path):
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq.add_receiver_group(
        name="surface",
        device=hydrophone,
        coords=[[x, 0.0] for x in np.linspace(0.0, 1.0, 201)],
    )

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.acquisition = acq
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))

    payload = sim.to_fs()
    coords = payload["Acquisition"]["receiver_groups"][0]["coordinates"]

    assert (
        coords["file"]
        == "simulations/simple/simple.h5:inputs/acquisition/receivers/surface/coordinates"
    )
    assert coords["hash"].startswith("blake3:")
    with h5py.File(tmp_path / "simulations/simple/simple.h5", "r") as h5:
        assert "inputs/acquisition/receivers/surface/coordinates" in h5
        dset = h5["inputs/acquisition/receivers/surface/coordinates"]
        assert dset.dtype == np.dtype("float64")

    acq.receiver_groups[0].coordinates = CoordsArray(
        coordinates=np.array([[x, 0.1] for x in np.linspace(0.0, 1.0, 201)])
    )
    updated = sim.to_fs()
    updated_coords = updated["Acquisition"]["receiver_groups"][0]["coordinates"]

    assert updated_coords["file"] == coords["file"]
    assert updated_coords["hash"].startswith("blake3:")
    assert updated_coords["hash"] != coords["hash"]
    assert isinstance(acq.receiver_groups[0].coordinates, CoordsArray)


def test_large_xarray_receiver_coordinates_preserve_units_when_materialized(tmp_path):
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=CoordinateValue([0.5, 0.0], units="m"))
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    receiver_coords = xr.DataArray(
        np.array([[0.1, z] for z in np.linspace(0.0, 200.0, 201)]),
        dims=("receiver", "coordinate"),
        coords={"coordinate": ["x", "z"]},
        attrs={"units": "m", "system": "global"},
    )
    acq.add_receiver_group(
        name="surface",
        device=hydrophone,
        coords=receiver_coords,
    )

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
        units=UnitConfig(defaults={"length": "km"}),
    )
    sim.acquisition = acq
    sim.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0, 0], u_bound=[0.04, 0.2], units="km", n=[1, 1])
    )

    payload = sim.to_fs()
    coords = payload["Acquisition"]["receiver_groups"][0]["coordinates"]

    assert coords["_type"] == "CoordsFromFile"
    assert coords["units"] == "m"
    assert coords["system"] == "global"
    with h5py.File(tmp_path / "simulations/simple/simple.h5", "r") as h5:
        dset = h5["inputs/acquisition/receivers/surface/coordinates"]
        assert dset.attrs["units"] == "m"
        assert dset.attrs["system"] == "global"


def test_large_receiver_coordinates_use_default_length_units_when_materialized(
    tmp_path,
):
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq.add_receiver_group(
        name="surface",
        device=hydrophone,
        coords=np.array([[0.1, z] for z in np.linspace(0.0, 0.2, 201)]),
    )

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
        units=UnitConfig(defaults={"length": "km"}),
    )
    sim.acquisition = acq
    sim.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0, 0], u_bound=[0.04, 0.2], units="km", n=[1, 1])
    )

    payload = sim.to_fs()
    coords = payload["Acquisition"]["receiver_groups"][0]["coordinates"]

    assert coords["_type"] == "CoordsFromFile"
    assert coords["units"] == "km"
    with h5py.File(tmp_path / "simulations/simple/simple.h5", "r") as h5:
        dset = h5["inputs/acquisition/receivers/surface/coordinates"]
        assert dset.attrs["units"] == "km"


def test_large_receiver_coordinates_inline_without_simulation_path():
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq.add_receiver_group(
        name="surface",
        device=hydrophone,
        coords=[[x, 0.0] for x in np.linspace(0.0, 1.0, 201)],
    )

    coords = acq.to_fs()["receiver_groups"][0]["coordinates"]

    assert coords["_type"] == "CoordsArray"
    assert len(coords["coords"]) == 201
    assert "file" not in coords


def test_receiver_coordinate_file_exports_project_relative_locator(tmp_path):
    old_file = tmp_path.parent / "old_project" / "simulations" / "simple" / "simple.h5"
    current_file = tmp_path / "simulations" / "simple" / "simple.h5"
    current_file.parent.mkdir(parents=True)
    current_file.touch()

    coords = CoordsFromFile(
        file=old_file,
        format="HDF5",
        dset="inputs/acquisition/receivers/surface/coordinates",
    )
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq.add_receiver_group(name="surface", device=hydrophone, coords=coords)

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.acquisition = acq
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))

    sim_file = sim.save()
    payload = json.loads(sim_file.read_text())

    assert (
        payload["Acquisition"]["receiver_groups"][0]["coordinates"]["file"]
        == "simulations/simple/simple.h5:inputs/acquisition/receivers/surface/coordinates"
    )


def test_loaded_receiver_coordinate_file_resolves_project_relative_get(tmp_path):
    coord_file = tmp_path / "simulations" / "simple" / "simple.h5"
    coord_file.parent.mkdir(parents=True)
    values = np.array([[0.25, 0.0], [0.75, 0.0]])
    with h5py.File(coord_file, "w") as h5:
        h5.create_dataset("coords", data=values)

    coords = CoordsFromFile(
        file="simulations/simple/simple.h5",
        format="HDF5",
        dset="coords",
    )
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq.add_receiver_group(name="surface", device=hydrophone, coords=coords)

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.acquisition = acq
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))

    loaded = SeismicSimulation.load(sim.save())
    loaded_coords = loaded.acquisition.receiver_groups[0].coordinates

    np.testing.assert_allclose(loaded_coords.get(), values)


def test_loaded_receiver_coordinate_file_uses_json_location_not_cwd(
    tmp_path, monkeypatch
):
    source_project = tmp_path / "source_project"
    copied_project = tmp_path / "copied_project"
    unrelated_dir = tmp_path / "unrelated"
    unrelated_dir.mkdir()

    coord_file = source_project / "inputs" / "receiver_coords.h5"
    coord_file.parent.mkdir(parents=True)
    source_values = np.array([[0.10, 0.0], [0.90, 0.0]])
    copied_values = np.array([[0.25, 0.0], [0.75, 0.0]])
    with h5py.File(coord_file, "w") as h5:
        h5.create_dataset("coords", data=source_values)

    coords = CoordsFromFile(file=coord_file, format="HDF5", dset="coords")
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq.add_receiver_group(name="surface", device=hydrophone, coords=coords)

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=source_project,
    )
    sim.acquisition = acq
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    shutil.copytree(source_project, copied_project)
    with h5py.File(copied_project / "inputs" / "receiver_coords.h5", "w") as h5:
        h5.create_dataset("coords", data=copied_values)

    monkeypatch.chdir(unrelated_dir)

    loaded = SeismicSimulation.load(
        copied_project / "simulations" / "simple" / "simple.json"
    )
    loaded_coords = loaded.acquisition.receiver_groups[0].coordinates

    assert loaded.project_path == copied_project.resolve()
    assert loaded_coords.file == copied_project / "inputs" / "receiver_coords.h5"
    np.testing.assert_allclose(loaded_coords.get(), copied_values)


def test_receiver_coordinate_file_rejects_remote_reference():
    with pytest.raises(ValueError, match="does not support remote coordinate files"):
        CoordsFromFile(
            file="remote:/server/receiver_coords.h5",
            format="HDF5",
            dset="coords",
        )

    with pytest.raises(ValueError, match="does not support remote coordinate files"):
        CoordsFromFile.from_fs(
            {
                "_type": "CoordsFromFile",
                "file": "s3://bucket/receiver_coords.h5:coords",
                "format": "HDF5",
            }
        )


def test_receiver_coordinate_file_hash_changes_with_hdf5_contents(tmp_path):
    coord_file = tmp_path / "receiver_coords.h5"
    values = np.array([[x, 0.0] for x in np.linspace(0.0, 1.0, 12)])
    with h5py.File(coord_file, "w") as h5:
        h5.create_dataset("coords", data=values)

    coords = CoordsFromFile(file=coord_file, format="HDF5", dset="coords")
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq.add_receiver_group(name="surface", device=hydrophone, coords=coords)

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.acquisition = acq
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))

    payload = sim.to_fs()
    first_hash = payload["Acquisition"]["receiver_groups"][0]["coordinates"]["hash"]

    with h5py.File(coord_file, "a") as h5:
        h5["coords"][:, 1] = 0.1

    updated = sim.to_fs()
    second_hash = updated["Acquisition"]["receiver_groups"][0]["coordinates"]["hash"]

    assert first_hash.startswith("blake3:")
    assert second_hash.startswith("blake3:")
    assert second_hash != first_hash


def test_remote_input_files_include_simulation_hdf5_store(tmp_path):
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq.add_receiver_group(
        name="surface",
        device=hydrophone,
        coords=np.array([[0.1, z] for z in np.linspace(0.0, 0.2, 201)]),
    )

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
        units=UnitConfig(defaults={"length": "km"}),
    )
    sim.acquisition = acq
    sim.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0, 0], u_bound=[0.04, 0.2], units="km", n=[1, 1])
    )
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()

    files = job.remote_input_files(Path("/remote/project"))

    assert (
        tmp_path / "simulations/simple/simple.h5",
        Path("/remote/project/simulations/simple/simple.h5"),
    ) in files


def test_remote_input_files_skip_remote_property_refs(tmp_path):
    model = ModelBase(name="model", dimension=2)
    model += ModelSubdomain(
        mesh_block_id=1,
        properties={"vp": Property.file("/server/only/vp.bin", remote=True)},
    )
    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.model = model
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()

    files = job.remote_input_files(Path("/remote/project"))

    assert files == []


def test_remote_input_files_include_rsf_sidecar(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    header = model_dir / "vp.rsf"
    sidecar = model_dir / "vp.rsf@"
    np.ones(4, dtype=np.float32).tofile(sidecar)
    header.write_text(
        "\n".join(
            [
                "n1=2",
                "n2=2",
                "d1=0.1",
                "d2=0.1",
                "o1=0.0",
                "o2=0.0",
                'data_format="native_float"',
                "esize=4",
                'in="./vp.rsf@"',
            ]
        )
    )

    model = ModelBase(name="model", dimension=2)
    model += ModelSubdomain(
        mesh_block_id=1,
        properties={"vp": Property.file(header, units="m/s")},
    )
    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.model = model
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()

    files = job.remote_input_files(Path("/remote/project"))

    assert (header, Path("/remote/project/models/vp.rsf")) in files
    assert (sidecar, Path("/remote/project/models/vp.rsf@")) in files


def test_trace_output_exports_only_traces_key(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    payload = job.to_fs()

    assert payload["Outputs"]["traces"]["path"] == "traces"
    assert "receivers" not in payload["Outputs"]
    assert "Outputs" not in sim.to_fs()


def test_job_trace_files_use_new_names(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0])
    job.save()

    assert isinstance(job.trace_manifest, TraceManifest)
    assert [str(file) for file in job.trace_manifest.files] == [
        str(tmp_path / "jobs/simple/freq/results/traces/traces_1.h5"),
        str(tmp_path / "jobs/simple/freq/results/traces/traces_2.h5"),
    ]
    assert job.traces.manifest == job.trace_manifest


def test_job_wavefields_use_output_requests_not_trace_receiver_groups(tmp_path):
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq.add_receiver_group(
        name="surface",
        device=hydrophone,
        coords=[[0.0, 0.0], [1.0, 0.0]],
    )

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.acquisition = acq
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0])
    wavefield_grid = xr.DataArray(
        np.empty((2, 2)),
        dims=("z", "r"),
        coords={"z": [0.0, 1.0], "r": [0.0, 1.0]},
    )
    job += WavefieldOutput(
        name="pressure_wavefield",
        field="pressure",
        grid=wavefield_grid,
    )
    job.save()

    assert job.trace_outputs.groups == ["surface"]
    assert job.trace_manifest.groups == ["surface"]
    assert job.traces.manifest.groups == ["surface"]
    assert job.wavefield_trace_outputs.groups == ["pressure_wavefield"]
    assert job.wavefield_manifest.groups == ["pressure_wavefield"]
    assert job.wavefields.manifest.groups == ["pressure_wavefield"]
    assert job.wavefield_manifest.output_path == (
        tmp_path / "jobs/simple/freq/results/wavefields"
    )
    wavefield_payload = job.wavefield_outputs["pressure_wavefield"]["grid"]
    assert wavefield_payload["_type"] == "XArrayGrid"
    assert wavefield_payload["dims"] == ["z", "r"]
    assert wavefield_payload["coords"]["z"]["data"] == [0.0, 1.0]
    assert wavefield_payload["coords"]["r"]["data"] == [0.0, 1.0]
    assert job.wavefield_outputs["pressure_wavefield"]["fields"] == ["pressure"]
    assert job.wavefield_outputs["pressure_wavefield"]["component_names"] == [
        "pressure"
    ]
    assert job.wavefield_outputs["pressure_wavefield"]["components"] == [
        "pressure_wavefield:pressure"
    ]
    payload = job.to_fs()
    assert payload["Outputs"]["wavefields"][0]["name"] == "pressure_wavefield"
    assert "pressure_wavefield" not in {
        group["name"] for group in sim.to_fs()["Acquisition"]["receiver_groups"]
    }


def test_job_wavefield_artifacts_do_not_override_duplicate_output_names(tmp_path):
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.acquisition = acq
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    grid = xr.DataArray(
        np.empty((2, 2)),
        dims=("z", "r"),
        coords={"z": [0.0, 1.0], "r": [0.0, 1.0]},
    )
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    job += [
        WavefieldOutput(name="snapshot", field="pressure", grid=grid),
        WavefieldOutput(name="snapshot", field="pressure", grid=grid),
    ]

    spec = job.wavefield_trace_outputs

    assert spec.groups == ["snapshot", "snapshot_1"]
    assert set(spec.wavefields) == {"snapshot", "snapshot_1"}
    assert spec.wavefields["snapshot_1"]["components"] == ["snapshot_1:pressure"]


def test_job_wavefields_open_with_unsaved_simulation_file(tmp_path):
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.acquisition = acq
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))

    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job += WavefieldOutput(
        name="pressure_wavefield",
        field="pressure",
        dims=("z", "r"),
        coords={"z": [0.0, 1.0], "r": [0.0, 1.0, 2.0]},
    )

    output_path = job.wavefield_trace_outputs.path
    output_path.mkdir(parents=True)
    values = np.arange(6, dtype=np.float32)
    with h5py.File(output_path / "traces_1.h5", "w") as h5:
        h5.create_dataset("frequency", data=10.0)
        dset = h5.create_dataset(
            "pressure_wavefield",
            data=np.stack([values, np.zeros_like(values)], axis=-1).reshape(6, 1, 1, 2),
        )
        dset.attrs["dims"] = ["receiver", "component", "shot"]
        dset.attrs["component"] = ["pressure"]
        dset.attrs["shot"] = [1]

    wavefields = job.wavefields.open()
    fd = wavefields.fd("pressure_wavefield", "pressure", source=1)

    assert wavefields.manifest.simulation == (
        tmp_path / "simulations" / "simple" / "simple.json"
    )
    np.testing.assert_allclose(fd.values[0], values)


def test_job_wavefield_outputs_export_device_component_metadata(tmp_path):
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])
    sim = SeismicSimulation(
        name="simple",
        physics="elastic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.acquisition = acq
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    device = ReceiverNode(name="elastic_device")
    device.add_component(name="vz", field="velocity", direction=[0.0, 1.0])
    device.add_component(name="szz", field="stress")
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    job += WavefieldOutput(
        name="elastic_wavefield",
        device=device,
        dims=("z", "r"),
        coords={"z": [0.0, 1.0], "r": [0.0, 1.0]},
    )

    metadata = job.wavefield_outputs["elastic_wavefield"]
    manifest_metadata = job.wavefield_trace_outputs.wavefields["elastic_wavefield"]

    assert metadata["fields"] == ["velocity", "stress"]
    assert metadata["component_names"] == ["vz", "szz"]
    assert metadata["components"] == ["elastic_wavefield:vz", "elastic_wavefield:szz"]
    assert metadata["component_specs"][0]["direction"] == [0.0, 1.0]
    assert metadata["device"]["components"][1]["name"] == "szz"
    assert manifest_metadata["component_names"] == ["vz", "szz"]


def test_trace_dataset_resolves_legacy_receiver_files(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    job.save()
    legacy = tmp_path / "jobs/simple/freq/results/traces/receivers_1.h5"
    legacy.parent.mkdir(parents=True)
    legacy.touch()

    traces = TraceDataset.from_job(job)

    assert traces.files == [str(legacy)]
    assert traces.paths == [legacy]


def test_trace_dataset_rejects_empty_manifest(tmp_path):
    manifest = TraceManifest(
        files=[],
        frequencies={},
        groups=[],
        simulation=tmp_path / "sim.json",
        result_path=tmp_path,
        output_path=tmp_path,
        project_path=tmp_path,
    )

    with pytest.raises(ValueError, match="at least one trace file"):
        TraceDataset.from_manifest(manifest)


def test_trace_dataset_from_manifest_preserves_output_artifacts(tmp_path):
    trace_file = tmp_path / "results" / "traces" / "traces_1.h5"
    trace_file.parent.mkdir(parents=True)
    trace_file.touch()
    artifact = OutputArtifact(
        path=tmp_path / "results" / "wavefields" / "pressure_1.vtu",
        kind="paraview",
    )
    manifest = TraceManifest(
        files=[trace_file],
        frequencies={1: 10.0},
        groups=["surface"],
        simulation=tmp_path / "simulation.json",
        result_path=tmp_path / "results",
        output_path=tmp_path / "results" / "traces",
        project_path=tmp_path,
        artifacts=[artifact],
    )

    traces = TraceDataset.from_manifest(manifest)

    assert traces.manifest.artifacts == [artifact]


def test_wavefield_dataset_missing_artifacts_fails_without_shard_warnings(tmp_path):
    output_path = tmp_path / "results" / "wavefields"
    manifest = TraceManifest(
        files=[output_path / "traces_1.h5", output_path / "traces_2.h5"],
        frequencies={1: 10.0, 2: 20.0},
        groups=["pressure_wavefield"],
        simulation=tmp_path / "simulation.json",
        result_path=tmp_path / "results",
        output_path=output_path,
        project_path=tmp_path,
        wavefields={
            "pressure_wavefield": {
                "grid": {
                    "_type": "XArrayGrid",
                    "dims": ["z", "r"],
                    "coords": {
                        "z": {"data": [0.0, 1.0]},
                        "r": {"data": [0.0, 1.0]},
                    },
                },
                "fields": ["pressure"],
            }
        },
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(FileNotFoundError, match="No wavefield trace files"):
            TraceDataset.from_manifest(manifest)

    assert not any(
        "Trace file is missing" in str(warning.message) for warning in caught
    )


def test_trace_manifest_accepts_solver_packed_trace_product(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    packed = trace_dir / "traces.h5"
    packed.touch()
    (trace_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-trace-manifest-1",
                "packed": {
                    "format": "hdf5",
                    "schema": "fs-traces-packed-1",
                    "relative_path": "traces/traces.h5",
                },
            }
        )
    )
    manifest = TraceManifest(
        files=[trace_dir / "traces_1.h5", trace_dir / "traces_2.h5"],
        frequencies={1: 10.0, 2: 20.0},
        groups=["surface"],
        simulation=tmp_path / "simulation.json",
        result_path=tmp_path / "results",
        output_path=trace_dir,
        project_path=tmp_path,
    )

    assert manifest.packed_file == packed
    assert manifest.existing_files == [packed]
    assert manifest.complete


def test_trace_dataset_matches_packed_frequency_values_when_task_ids_shift(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    packed = trace_dir / "traces.h5"
    packed.touch()
    (trace_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-trace-manifest-1",
                "packed": {
                    "format": "hdf5",
                    "schema": "fs-traces-packed-1",
                    "relative_path": "traces/traces.h5",
                },
                "frequencies": [
                    {"task_id": 10, "frequency": 10.0, "status": "packed"},
                    {"task_id": 20, "frequency": 20.0, "status": "packed"},
                    {"task_id": 30, "frequency": 30.0, "status": "packed"},
                ],
            }
        )
    )
    manifest = TraceManifest(
        files=[trace_dir / "traces_1.h5", trace_dir / "traces_2.h5"],
        frequencies={1: 10.0, 2: 20.0},
        groups=["surface"],
        simulation=tmp_path / "simulation.json",
        result_path=tmp_path / "results",
        output_path=trace_dir,
        project_path=tmp_path,
    )

    assert manifest.missing_packed_frequencies == {}
    assert manifest.complete

    traces = TraceDataset.from_manifest(manifest)

    assert traces.manifest.files == [packed]
    assert traces.manifest.frequencies == {1: 10.0, 2: 20.0}


def test_trace_dataset_uses_matching_shard_when_packed_manifest_is_stale(tmp_path):
    trace_dir = tmp_path / "results" / "wavefields"
    shard_dir = trace_dir / "shards"
    shard_dir.mkdir(parents=True)
    packed = trace_dir / "traces.h5"
    packed.touch()
    shard = shard_dir / "f_50.00000_hz.h5"
    with h5py.File(shard, "w") as h5:
        h5.create_dataset("frequency", data=50.0)
        h5.create_dataset("laplace", data=-0.5)
        dset = h5.create_dataset("wavefields_f", data=np.zeros((1, 1, 1, 2)))
        dset.attrs["dims"] = ["receiver", "component", "source"]
    (trace_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-trace-manifest-1",
                "packed": {
                    "format": "hdf5",
                    "schema": "fs-traces-packed-1",
                    "relative_path": "wavefields/traces.h5",
                },
                "frequencies": [
                    {"task_id": 1, "frequency": 100.0, "status": "packed"},
                ],
            }
        )
    )
    manifest = TraceManifest(
        files=[trace_dir / "traces_1.h5"],
        frequencies={1: 50.0},
        groups=["wavefields_f"],
        simulation=tmp_path / "simulation.json",
        result_path=tmp_path / "results",
        output_path=trace_dir,
        project_path=tmp_path,
        laplace={1: -0.5},
        wavefields={"wavefields_f": {"fields": ["pressure"]}},
    )

    with pytest.warns(RuntimeWarning, match="missing 1 of 1 expected frequencies"):
        traces = TraceDataset.from_manifest(manifest)

    assert traces.manifest.files == [shard]
    assert traces.manifest.frequencies == {1: 50.0}
    assert traces.manifest.laplace == {1: -0.5}


def test_trace_dataset_uses_matching_shard_without_packed_manifest(tmp_path):
    trace_dir = tmp_path / "results" / "wavefields"
    shard_dir = trace_dir / "shards"
    shard_dir.mkdir(parents=True)
    shard = shard_dir / "f_50.00000_hz.h5"
    with h5py.File(shard, "w") as h5:
        h5.create_dataset("frequency", data=50.0)
        h5.create_dataset("laplace", data=-0.5)
        dset = h5.create_dataset("wavefields_f", data=np.zeros((1, 1, 1, 2)))
        dset.attrs["dims"] = ["receiver", "component", "source"]
    manifest = TraceManifest(
        files=[trace_dir / "traces_1.h5"],
        frequencies={1: 50.0},
        groups=["wavefields_f"],
        simulation=tmp_path / "simulation.json",
        result_path=tmp_path / "results",
        output_path=trace_dir,
        project_path=tmp_path,
        laplace={1: -0.5},
        wavefields={"wavefields_f": {"fields": ["pressure"]}},
    )

    traces = TraceDataset.from_manifest(manifest)

    assert traces.manifest.files == [shard]
    assert traces.manifest.frequencies == {1: 50.0}
    assert traces.manifest.laplace == {1: -0.5}


def _write_frequency_trace_shard(path, group, *, frequency, value=1.0):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        h5.create_dataset("frequency", data=frequency)
        h5.create_dataset("laplace", data=-0.5)
        dset = h5.create_dataset(
            group,
            data=np.full((1, 1, 1, 2), value, dtype=np.float32),
        )
        dset.attrs["dims"] = ["receiver", "component", "shot"]
        trace_group = h5.require_group(f"/survey/receiver_groups/{group}/traces")
        trace_group.create_dataset("receiver_id", data=np.array([101], dtype=np.int32))
        trace_group.create_dataset("source_id", data=np.array([7], dtype=np.int32))
        trace_group.create_dataset(
            "component_name",
            data=np.array(["p"], dtype=string_dtype),
        )


def test_trace_dataset_uses_nested_frequency_named_shards_when_files_missing(tmp_path):
    result_path = tmp_path / "results"
    shard_dir = result_path / "traces" / "shards"
    shard_dir.mkdir(parents=True)
    shard_1 = shard_dir / "trace_frequency_10.00000_hz.h5"
    shard_2 = shard_dir / "trace_frequency_20.00000_hz.h5"
    _write_frequency_trace_shard(shard_1, "surface", frequency=10.0, value=1.0)
    _write_frequency_trace_shard(shard_2, "surface", frequency=20.0, value=2.0)
    manifest = TraceManifest(
        files=[result_path / "traces_1.h5", result_path / "traces_2.h5"],
        frequencies={1: 10.0, 2: 20.0},
        groups=["surface"],
        simulation=tmp_path / "simulation.json",
        result_path=result_path,
        output_path=result_path,
        project_path=tmp_path,
    )

    traces = TraceDataset.from_manifest(manifest)

    assert traces.manifest.files == [shard_1, shard_2]
    assert traces.groups == ["surface"]
    assert traces.frequencies("surface").tolist() == [10.0, 20.0]


def test_trace_store_builds_vds_from_shards_when_all_trace_files_missing(tmp_path):
    result_path = tmp_path / "results"
    shard_dir = result_path / "traces" / "shards"
    shard_dir.mkdir(parents=True)
    _write_frequency_trace_shard(
        shard_dir / "trace_frequency_10.00000_hz.h5",
        "surface",
        frequency=10.0,
        value=1.0,
    )
    _write_frequency_trace_shard(
        shard_dir / "trace_frequency_20.00000_hz.h5",
        "surface",
        frequency=20.0,
        value=2.0,
    )
    store = TraceStore(
        metadata={
            "groups": ["surface"],
            "output_path": result_path,
            "result_path": result_path,
            "f_map": {1: 10.0, 2: 20.0},
        },
        files=[result_path / "traces_1.h5", result_path / "traces_2.h5"],
    )

    with pytest.warns(RuntimeWarning, match="creating a VDS from 2 matching"):
        assert store.groups == ["surface"]

    assert Path(store._consolidated) == result_path / "traces_vds.h5"
    assert store.frequencies("surface").tolist() == [10.0, 20.0]


def _write_indexed_packed_trace_product(
    path, group, *, frequencies, values, laplace=None
):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    numbers = np.arange(1, len(frequencies) + 1, dtype=np.int32)
    laplace_values = (
        np.zeros(len(frequencies), dtype=float)
        if laplace is None
        else np.asarray(laplace, dtype=float)
    )
    with h5py.File(path, "w") as h5:
        h5.create_dataset("frequency", data=np.asarray(frequencies, dtype=float))
        h5.create_dataset("laplace", data=laplace_values)
        h5.create_dataset("task_id", data=numbers)
        h5.create_dataset(
            "survey/packed_layout_kind",
            data=np.array(["indexed_frequency_trace_v1"], dtype=string_dtype),
        )
        catalog = h5.require_group("survey/receiver_groups/_catalog")
        catalog.create_dataset("group_name", data=np.array([group], dtype=string_dtype))
        catalog.create_dataset(
            "dataset_path", data=np.array([f"/{group}"], dtype=string_dtype)
        )
        catalog.create_dataset(
            "layout_kind", data=np.array(["dense_trace_v1"], dtype=string_dtype)
        )
        trace_group = h5.require_group(f"survey/receiver_groups/{group}/traces")
        trace_group.create_dataset("receiver_id", data=np.array([101], dtype=np.int32))
        trace_group.create_dataset("source_id", data=np.array([7], dtype=np.int32))
        trace_group.create_dataset(
            "component_name", data=np.array(["p"], dtype=string_dtype)
        )
        h5.require_group("trace_index/datasets")
        h5.create_dataset(
            "trace_index/schema_version",
            data=np.array(["fs-trace-index-1"], dtype=string_dtype),
        )
        h5.create_dataset(
            "trace_index/layout_kind",
            data=np.array(["indexed_frequency_trace_v1"], dtype=string_dtype),
        )
        h5.create_dataset(
            "trace_index/data_root", data=np.array(["/trace_data"], dtype=string_dtype)
        )
        h5.create_dataset("trace_index/dataset_number", data=numbers)
        h5.create_dataset(
            "trace_index/frequency", data=np.asarray(frequencies, dtype=float)
        )
        h5.create_dataset("trace_index/laplace", data=laplace_values)
        h5.create_dataset("trace_index/task_id", data=numbers)
        h5.create_dataset(
            "trace_index/shard_file",
            data=np.array(
                [f"wavefields/{group}/f_{freq:.5f}_hz.h5" for freq in frequencies],
                dtype=string_dtype,
            ),
        )
        h5.create_dataset("trace_index/datasets/dataset_number", data=numbers)
        h5.create_dataset(
            "trace_index/datasets/source_path",
            data=np.array([f"/{group}"] * len(frequencies), dtype=string_dtype),
        )
        h5.create_dataset(
            "trace_index/datasets/packed_path",
            data=np.array(
                [f"/trace_data/{group}/{number:06d}" for number in numbers],
                dtype=string_dtype,
            ),
        )
        for number, value in zip(numbers, values):
            dset = h5.create_dataset(
                f"trace_data/{group}/{number:06d}",
                data=np.array([[[[value, 0.0]]]], dtype=np.float32),
            )
            dset.attrs["dims"] = ["receiver", "component", "shot"]
            dset.attrs["layout_kind"] = ["dense_trace_v1"]
            dset.attrs["receiver"] = np.array([101], dtype=np.int32)
            dset.attrs["component"] = np.array(["p"], dtype=string_dtype)
            dset.attrs["shot"] = np.array([7], dtype=np.int32)


def test_trace_dataset_filters_indexed_packed_rows_by_frequency_and_laplace(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    packed = trace_dir / "traces.h5"
    _write_indexed_packed_trace_product(
        packed,
        "surface",
        frequencies=[7.999999999999999, 8.0],
        laplace=[-0.1, -0.2],
        values=[1.0, 2.0],
    )
    manifest = TraceManifest(
        files=[packed],
        frequencies={1: 8.0},
        laplace={1: -0.2},
        groups=["surface"],
        simulation=tmp_path / "simulation.json",
        result_path=tmp_path / "results",
        output_path=trace_dir,
        project_path=tmp_path,
    )

    traces = TraceDataset.from_manifest(manifest)
    fd = traces.fd("surface", "p", source=7)

    assert traces.frequencies("surface").tolist() == [8.0]
    assert fd.coords["frequency"].values.tolist() == [8.0]
    assert fd.coords["laplace"].values.tolist() == pytest.approx([-0.2])
    assert fd.values[:, 0].real.tolist() == [2.0]


def test_trace_dataset_keeps_indexed_packed_rows_without_manifest_laplace(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    packed = trace_dir / "traces.h5"
    _write_indexed_packed_trace_product(
        packed,
        "surface",
        frequencies=[8.0],
        laplace=[-0.2],
        values=[2.0],
    )
    manifest = TraceManifest(
        files=[packed],
        frequencies={1: 8.0},
        groups=["surface"],
        simulation=tmp_path / "simulation.json",
        result_path=tmp_path / "results",
        output_path=trace_dir,
        project_path=tmp_path,
    )

    traces = TraceDataset.from_manifest(manifest)
    fd = traces.fd("surface", "p", source=7)

    assert fd.coords["laplace"].values.tolist() == pytest.approx([-0.2])
    assert fd.values[:, 0].real.tolist() == [2.0]


def test_trace_dataset_uses_named_wavefield_packed_products(tmp_path):
    result_path = tmp_path / "results"
    trace_dir = result_path / "wavefields"
    trace_dir.mkdir(parents=True)
    frequencies = [10.0, 20.0]
    full = trace_dir / "full.h5"
    fracture = trace_dir / "fracture.h5"

    _write_indexed_packed_trace_product(
        full,
        "full",
        frequencies=frequencies,
        values=[1.0, 2.0],
    )
    _write_indexed_packed_trace_product(
        fracture,
        "fracture",
        frequencies=frequencies,
        values=[3.0, 4.0],
    )
    for name in ("full", "fracture"):
        output_manifest = trace_dir / name / "manifest.json"
        output_manifest.parent.mkdir()
        output_manifest.write_text(
            json.dumps(
                {
                    "schema": "fs-trace-manifest-1",
                    "packed": {
                        "format": "hdf5",
                        "schema": "fs-traces-packed-1",
                        "layout": "indexed_frequency_trace_v1",
                        "relative_path": f"wavefields/{name}.h5",
                    },
                    "frequencies": [
                        {
                            "task_id": index,
                            "frequency": frequency,
                            "status": "packed",
                            "relative_path": (
                                f"wavefields/{name}/f_{frequency:.5f}_hz.h5"
                            ),
                        }
                        for index, frequency in enumerate(frequencies, start=1)
                    ],
                }
            )
        )

    manifest = TraceManifest(
        files=[trace_dir / "traces_1.h5", trace_dir / "traces_2.h5"],
        frequencies={1: 10.0, 2: 20.0},
        groups=["full", "fracture"],
        simulation=tmp_path / "simulation.json",
        result_path=result_path,
        output_path=trace_dir,
        project_path=tmp_path,
        wavefields={
            "full": {"fields": ["pressure"]},
            "fracture": {"fields": ["pressure"]},
        },
    )

    traces = TraceDataset.from_manifest(manifest)

    assert manifest.packed_files == [full, fracture]
    assert manifest.complete is True
    assert traces.manifest.files == [full, fracture]
    assert traces.groups == ["full", "fracture"]
    assert traces.frequencies("full").tolist() == [10.0, 20.0]
    assert traces.frequencies("fracture").tolist() == [10.0, 20.0]
    assert traces.components("full").tolist() == ["p"]

    full_fd = traces.fd("full", "p", source=7)
    fracture_fd = traces.fd("fracture", "p", source=7)

    assert full_fd.values[:, 0].real.tolist() == [1.0, 2.0]
    assert fracture_fd.values[:, 0].real.tolist() == [3.0, 4.0]


def test_trace_dataset_reports_packed_product_with_no_requested_frequencies(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    packed = trace_dir / "traces.h5"
    packed.touch()
    (trace_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-trace-manifest-1",
                "packed": {
                    "format": "hdf5",
                    "schema": "fs-traces-packed-1",
                    "relative_path": "traces/traces.h5",
                },
                "frequencies": [
                    {"task_id": 1, "frequency": 1.0, "status": "packed"},
                    {"task_id": 2, "frequency": 2.0, "status": "packed"},
                ],
            }
        )
    )
    manifest = TraceManifest(
        files=[trace_dir / "traces_1.h5", trace_dir / "traces_2.h5"],
        frequencies={1: 10.0, 2: 20.0},
        groups=["surface"],
        simulation=tmp_path / "simulation.json",
        result_path=tmp_path / "results",
        output_path=trace_dir,
        project_path=tmp_path,
    )

    with pytest.warns(RuntimeWarning, match="missing 2 of 2 expected frequencies"):
        with pytest.raises(ValueError, match="contains no frequencies requested"):
            TraceDataset.from_manifest(manifest)


def test_job_run_fingerprint_requires_matching_outputs_and_changes_with_simulation(
    tmp_path,
):
    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    job.save()
    assert not job.is_run_current()

    trace_file = tmp_path / "jobs/simple/freq/results/traces/traces_1.h5"
    trace_file.parent.mkdir(parents=True)
    trace_file.touch()
    job.write_run_state(status="completed")
    assert job.is_run_current()

    job += WavefieldOutput(
        name="pressure",
        fields=["pressure"],
        dims=("z", "r"),
        coords={"z": [0.0, 1.0], "r": [0.0, 1.0]},
    )
    assert not job.is_run_current()
    job.save()
    job.write_run_state(status="completed")
    assert job.is_run_current()

    payload = json.loads(sim._file.read_text())
    payload["Solver"]["max_iter"] = 123
    sim._file.write_text(json.dumps(payload))

    assert not job.is_run_current()


def test_job_run_current_accepts_fast_solver_run_manifest_hashes(tmp_path):
    def sha256(path):
        return f"sha256:{hashlib.sha256(Path(path).read_bytes()).hexdigest()}"

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    job.save()

    trace_file = tmp_path / "jobs/simple/freq/results/traces/traces_1.h5"
    trace_file.parent.mkdir(parents=True)
    trace_file.touch()

    run_dir = job._result_path / "_fs_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-run-manifest-1",
                "exit_status": "success",
                "job_file_sha256": sha256(job._file),
                "simulation_file_sha256": sha256(sim._file),
            }
        )
    )

    assert job.is_run_current()

    payload = json.loads(sim._file.read_text())
    payload["Solver"]["max_iter"] = 123
    sim._file.write_text(json.dumps(payload))

    assert not job.is_run_current()


def test_job_run_current_accepts_current_fast_solver_run_manifest_schema(tmp_path):
    def sha256(path):
        return f"sha256:{hashlib.sha256(Path(path).read_bytes()).hexdigest()}"

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    job.save()
    trace_file = tmp_path / "jobs/simple/freq/results/traces/traces_1.h5"
    trace_file.parent.mkdir(parents=True)
    trace_file.touch()

    run_dir = job._result_path / "_fs_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-run-manifest-1",
                "exit_status": {"code": 0, "status": "success"},
                "inputs": {
                    "job_file": {"hash": sha256(job._file), "path": str(job._file)},
                    "simulation_file": {
                        "hash": sha256(sim._file),
                        "path": str(sim._file),
                    },
                },
            }
        )
    )

    assert job.is_run_current()


def test_job_run_current_ignores_rank_and_thread_counts(tmp_path):
    def sha256(path):
        return f"sha256:{hashlib.sha256(Path(path).read_bytes()).hexdigest()}"

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    job.save()
    trace_file = tmp_path / "jobs/simple/freq/results/traces/traces_1.h5"
    trace_file.parent.mkdir(parents=True)
    trace_file.touch()

    run_dir = job._result_path / "_fs_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-run-manifest-1",
                "exit_status": {"code": 0, "status": "success"},
                "execution": {
                    "mpi": {"ranks": 128},
                    "openmp": {"threads": 3},
                },
                "inputs": {
                    "job_file": {"hash": sha256(job._file), "path": str(job._file)},
                    "simulation_file": {
                        "hash": sha256(sim._file),
                        "path": str(sim._file),
                    },
                },
            }
        )
    )

    assert job.is_run_current()


def test_job_run_current_reacts_to_receiver_coordinate_updates(tmp_path):
    def sha256(path):
        return f"sha256:{hashlib.sha256(Path(path).read_bytes()).hexdigest()}"

    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq.add_receiver_group(
        name="surface",
        device=hydrophone,
        coords=[[x, 0.0] for x in np.linspace(0.0, 1.0, 12)],
    )

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.acquisition = acq
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    job.save()

    trace_file = tmp_path / "jobs/simple/freq/results/traces/traces_1.h5"
    trace_file.parent.mkdir(parents=True)
    trace_file.touch()

    run_dir = job._result_path / "_fs_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-run-manifest-1",
                "exit_status": "success",
                "job_file_sha256": sha256(job._file),
                "simulation_file_sha256": sha256(sim._file),
            }
        )
    )
    assert job.is_run_current()

    acq.receiver_groups[0].coordinates = CoordsArray(
        coordinates=np.array([[x, 0.1] for x in np.linspace(0.0, 1.0, 12)])
    )
    sim.save()

    assert not job.is_run_current()


def test_job_run_current_accepts_solver_packed_trace_product(tmp_path):
    def sha256(path):
        return f"sha256:{hashlib.sha256(Path(path).read_bytes()).hexdigest()}"

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0])
    job.save()

    trace_dir = tmp_path / "jobs/simple/freq/results/traces"
    trace_dir.mkdir(parents=True)
    (trace_dir / "traces.h5").touch()
    (trace_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-trace-manifest-1",
                "packed": {
                    "format": "hdf5",
                    "schema": "fs-traces-packed-1",
                    "relative_path": "traces/traces.h5",
                },
            }
        )
    )
    run_dir = job._result_path / "_fs_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-run-manifest-1",
                "exit_status": "success",
                "job_file_sha256": sha256(job._file),
                "simulation_file_sha256": sha256(sim._file),
            }
        )
    )

    assert job.trace_outputs_exist()
    assert job.is_run_current()


def test_trace_dataset_combines_jobs_by_frequency_and_deduplicates_overlap(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.save()

    low = FrequencyDomainJob(name="low", simulation=sim, f_list=[10.0, 20.0])
    high = FrequencyDomainJob(name="high", simulation=sim, f_list=[20.0, 30.0])
    low.save()
    high.save()
    for job in [low, high]:
        for file in job.trace_manifest.files:
            path = Path(file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    with pytest.warns(RuntimeWarning, match="Duplicate trace frequencies"):
        traces = TraceDataset.from_jobs([low, high])

    assert traces.metadata["f_map"] == {1: 10.0, 2: 20.0, 3: 30.0}
    assert traces.files == [
        str(tmp_path / "jobs/simple/low/results/traces/traces_1.h5"),
        str(tmp_path / "jobs/simple/low/results/traces/traces_2.h5"),
        str(tmp_path / "jobs/simple/high/results/traces/traces_2.h5"),
    ]


def test_trace_store_uses_dataset_backed_survey_trace_tables(tmp_path):
    trace_files = []
    string_dtype = h5py.string_dtype(encoding="utf-8")
    for idx, freq in enumerate([10.0, 20.0], start=1):
        file = tmp_path / f"traces_{idx}.h5"
        trace_files.append(str(file))
        with h5py.File(file, "w") as h5:
            h5.create_dataset("frequency", data=freq)
            dset = h5.create_dataset(
                "surface",
                data=np.zeros((2, 2, 1, 2), dtype=np.float32),
            )
            dset.attrs["dims"] = ["receiver", "component", "shot"]
            trace_group = h5.require_group("/survey/receiver_groups/surface/traces")
            trace_group.create_dataset(
                "receiver_id",
                data=np.array([101, 102, 101, 102], dtype=np.int32),
            )
            trace_group.create_dataset(
                "source_id",
                data=np.array([7, 7, 7, 7], dtype=np.int32),
            )
            trace_group.create_dataset(
                "component_name",
                data=np.array(["p", "vx", "p", "vx"], dtype=string_dtype),
            )
            h5.require_group("/survey/receivers").create_dataset(
                "coordinates",
                data=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
            )

    traces = TraceDataset.from_manifest(
        TraceManifest(
            files=[Path(file) for file in trace_files],
            frequencies={1: 10.0, 2: 20.0},
            groups=["surface"],
            simulation=tmp_path / "simulation.json",
            result_path=tmp_path / "results",
            output_path=tmp_path / "results" / "traces",
            project_path=tmp_path,
        )
    )
    traces.consolidate()

    assert traces.groups == ["surface"]
    assert traces.receivers("surface").tolist() == [101, 102]
    assert traces.sources("surface").tolist() == [7]
    assert traces.components("surface").tolist() == ["p", "vx"]
    assert traces.frequencies("surface").tolist() == [10.0, 20.0]
    assert traces.survey_tables()["receivers"]["coordinates"] == [
        [0.0, 0.0],
        [1.0, 0.0],
    ]
    fd = traces.fd("surface", "p", source=7)
    assert fd.dims == ("frequency", "receiver")
    assert fd.coords["frequency"].values.tolist() == [10.0, 20.0]
    assert fd.coords["receiver"].values.tolist() == [101, 102]
    assert type(fd).__name__ == "DataArray"
    assert (tmp_path / "results" / "_fs_run" / "cache" / "traces_vds.h5").exists()


def test_trace_summary_is_not_ansi_escaped_in_notebook_repr(tmp_path, capsys):
    trace_files = []
    for idx, freq in enumerate([0.5, 1.0], start=1):
        file = tmp_path / f"traces_{idx}.h5"
        trace_files.append(file)
        with h5py.File(file, "w") as h5:
            h5.create_dataset("frequency", data=freq)
            dset = h5.create_dataset(
                "surface",
                data=np.zeros((2, 2, 1, 2), dtype=np.float32),
            )
            dset.attrs["dims"] = ["receiver", "component", "shot"]
            dset.attrs["receiver"] = np.array([1, 2], dtype=np.int32)
            dset.attrs["shot"] = np.array([1], dtype=np.int32)
            dset.attrs["component"] = np.array(["p", "v_z"], dtype="S")

    traces = TraceDataset.from_manifest(
        TraceManifest(
            files=trace_files,
            frequencies={1: 0.5, 2: 1.0},
            groups=["surface"],
            simulation=tmp_path / "simulation.json",
            result_path=tmp_path / "results",
            output_path=tmp_path / "results" / "traces",
            project_path=tmp_path,
        )
    )

    summary = traces.summary

    assert isinstance(summary, str)
    assert "\x1b" not in summary
    assert str(summary).startswith("surface\n  Receivers")
    assert repr(summary).startswith("surface\n")
    assert not repr(summary).startswith(("'", '"'))
    assert "\x1b[38;5;248mReceivers\x1b[0m" in traces.format_summary(colorize=True)

    returned = traces.print_summary()
    captured = capsys.readouterr()

    assert returned == summary
    assert captured.out == str(summary)
    assert captured.err == ""


def test_trace_store_uses_first_task_metadata_for_compact_later_tasks(tmp_path):
    trace_files = []
    string_dtype = h5py.string_dtype(encoding="utf-8")
    for idx, freq in enumerate([10.0, 20.0, 30.0], start=1):
        file = tmp_path / f"traces_{idx}.h5"
        trace_files.append(file)
        with h5py.File(file, "w") as h5:
            h5.create_dataset("frequency", data=freq)
            dset = h5.create_dataset(
                "surface",
                data=np.full((2, 2, 1, 2), idx, dtype=np.float32),
            )

            if idx == 1:
                dset.attrs["dims"] = ["receiver", "component", "shot"]
                trace_group = h5.require_group("/survey/receiver_groups/surface/traces")
                trace_group.create_dataset(
                    "receiver_id",
                    data=np.array([101, 102, 101, 102], dtype=np.int32),
                )
                trace_group.create_dataset(
                    "source_id",
                    data=np.array([7, 7, 7, 7], dtype=np.int32),
                )
                trace_group.create_dataset(
                    "component_name",
                    data=np.array(["p", "vx", "p", "vx"], dtype=string_dtype),
                )
                h5.require_group("/survey/receivers").create_dataset(
                    "coordinates",
                    data=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
                )

    traces = TraceDataset.from_manifest(
        TraceManifest(
            files=trace_files,
            frequencies={1: 10.0, 2: 20.0, 3: 30.0},
            groups=["surface"],
            simulation=tmp_path / "simulation.json",
            result_path=tmp_path / "results",
            output_path=tmp_path / "results" / "traces",
            project_path=tmp_path,
        )
    )
    traces.consolidate()

    assert traces.groups == ["surface"]
    assert traces.receivers("surface").tolist() == [101, 102]
    assert traces.sources("surface").tolist() == [7]
    assert traces.components("surface").tolist() == ["p", "vx"]
    assert traces.survey_tables()["receivers"]["coordinates"] == [
        [0.0, 0.0],
        [1.0, 0.0],
    ]

    fd = traces.fd("surface", "p", source=7)
    assert fd.dims == ("frequency", "receiver")
    assert fd.coords["frequency"].values.tolist() == [10.0, 20.0, 30.0]
    assert fd.coords["receiver"].values.tolist() == [101, 102]
    assert fd.values.real.tolist() == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

    with h5py.File(trace_files[1], "r") as h5:
        assert "survey" not in h5
        assert "dims" not in h5["surface"].attrs


def test_trace_store_uses_shared_trace_metadata_for_payload_shards(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    shard_dir = trace_dir / "shards"
    shard_dir.mkdir(parents=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    metadata = trace_dir / "trace_metadata.h5"
    with h5py.File(metadata, "w") as h5:
        h5.create_dataset(
            "metadata/schema_version",
            data=np.bytes_("fs_trace_metadata_v1            "),
        )
        h5.create_dataset(
            "survey/schema_version",
            data=np.array(
                [np.bytes_("fs_seismic_trace_store_v1        ")],
            ),
        )
        h5.create_dataset(
            "survey/sources/source_id", data=np.array([7], dtype=np.int32)
        )
        receivers = h5.require_group("survey/receiver_groups/surface/receivers")
        receivers.create_dataset(
            "receiver_id", data=np.array([101, 102], dtype=np.int32)
        )
        receivers.create_dataset(
            "coordinates",
            data=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        )
        components = h5.require_group("survey/receiver_groups/surface/components")
        components.create_dataset(
            "component_name",
            data=np.array(["p", "vx"], dtype=string_dtype),
        )
        catalog = h5.require_group("survey/receiver_groups/_catalog")
        catalog.create_dataset(
            "group_name",
            data=np.array(["surface"], dtype=string_dtype),
        )
        catalog.create_dataset(
            "dataset_path",
            data=np.array(["/surface"], dtype=string_dtype),
        )
        template = h5.create_dataset(
            "surface",
            data=np.zeros((1, 2, 2, 2), dtype=np.float32),
        )
        template.attrs["dims"] = ["receiver", "component", "shot"]
        template.attrs["layout_kind"] = ["dense_trace_v1"]
        template.attrs["units"] = ["m/s"]

    shards = []
    for task_id, frequency in enumerate([10.0, 20.0], start=1):
        path = shard_dir / f"f_{frequency:.5f}_hz.h5"
        shards.append(path)
        data = np.zeros((1, 2, 2, 2), dtype=np.float32)
        data[0, 0, :, 0] = float(task_id)
        data[0, 1, :, 0] = float(task_id + 10)
        with h5py.File(path, "w") as h5:
            h5.create_dataset("frequency", data=frequency)
            h5.create_dataset("laplace", data=0.0)
            h5.create_dataset("task_id", data=task_id)
            h5.create_dataset(
                "trace_metadata_file",
                data=np.array(
                    [np.bytes_("traces/trace_metadata.h5     ")],
                ),
            )
            h5.create_dataset("surface", data=data)

    traces = TraceDataset.from_manifest(
        TraceManifest(
            files=[trace_dir / "traces_1.h5", trace_dir / "traces_2.h5"],
            frequencies={1: 10.0, 2: 20.0},
            groups=["surface"],
            simulation=tmp_path / "simulation.json",
            result_path=tmp_path / "results",
            output_path=trace_dir,
            project_path=tmp_path,
        )
    )
    traces.consolidate()

    assert traces.manifest.files == shards
    assert traces.groups == ["surface"]
    assert traces.sources("surface").tolist() == [7]
    assert traces.receivers("surface").tolist() == [101, 102]
    assert traces.components("surface").tolist() == ["p", "vx"]
    assert traces.frequencies("surface").tolist() == [10.0, 20.0]

    fd = traces.fd("surface", "p", source=7)
    assert fd.dims == ("frequency", "receiver")
    assert fd.coords["receiver"].values.tolist() == [101, 102]
    assert fd.values.real.tolist() == [[1.0, 1.0], [2.0, 2.0]]

    with h5py.File(shards[0], "r") as h5:
        assert "survey" not in h5
        assert "dims" not in h5["surface"].attrs

    with h5py.File(traces._store._consolidated, "r") as h5:
        assert "survey" in h5
        assert h5["surface"].attrs["units"].tolist() == ["m/s"]

    store = TraceStore(
        metadata={
            "groups": ["surface"],
            "output_path": trace_dir,
            "result_path": tmp_path / "results",
            "f_map": {1: 10.0, 2: 20.0},
        },
        files=[trace_dir / "traces_1.h5", trace_dir / "traces_2.h5"],
    )
    with pytest.warns(RuntimeWarning, match="creating a VDS from 2 matching"):
        assert store.groups == ["surface"]


def test_trace_store_uses_solver_packed_trace_file_without_vds_warnings(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    trace_files = [trace_dir / f"traces_{idx}.h5" for idx in range(1, 4)]
    packed = trace_dir / "traces.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    (trace_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-trace-manifest-1",
                "packed": {
                    "format": "hdf5",
                    "schema": "fs-traces-packed-1",
                    "relative_path": "traces/traces.h5",
                },
                "frequencies": [
                    {"task_id": 1, "frequency": 10.0, "status": "packed"},
                    {"task_id": 2, "frequency": 20.0, "status": "packed"},
                    {"task_id": 3, "frequency": 30.0, "status": "packed"},
                    {"task_id": 4, "frequency": 40.0, "status": "packed"},
                ],
            }
        )
    )
    with h5py.File(packed, "w") as h5:
        h5.create_dataset("frequency", data=np.array([10.0, 20.0, 30.0, 40.0]))
        h5.create_dataset("laplace", data=np.zeros(4))
        h5.create_dataset("task_id", data=np.array([1, 2, 3, 4], dtype=np.int32))
        h5.create_dataset(
            "survey/packed_layout_kind",
            data=np.array(["packed_frequency_trace_v1"], dtype=string_dtype),
        )
        dset = h5.create_dataset(
            "surface",
            data=np.array(
                [
                    [[[[1.0, 0.0], [2.0, 0.0]]]],
                    [[[[3.0, 0.0], [4.0, 0.0]]]],
                    [[[[5.0, 0.0], [6.0, 0.0]]]],
                    [[[[7.0, 0.0], [8.0, 0.0]]]],
                ],
                dtype=np.float32,
            ),
        )
        dset.attrs["dims"] = ["receiver", "component", "shot", "frequency"]
        dset.attrs["layout_kind"] = ["dense_trace_v1"]
        dset.attrs["receiver"] = np.array([101, 102], dtype=np.int32)
        dset.attrs["component"] = np.array(["p"], dtype=string_dtype)
        dset.attrs["shot"] = np.array([7], dtype=np.int32)

    traces = TraceDataset.from_manifest(
        TraceManifest(
            files=trace_files,
            frequencies={1: 10.0, 2: 20.0, 3: 30.0},
            groups=["surface"],
            simulation=tmp_path / "simulation.json",
            result_path=tmp_path / "results",
            output_path=trace_dir,
            project_path=tmp_path,
        )
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        consolidated = traces.consolidate()

    assert consolidated == packed
    assert not any(
        "Trace file is missing" in str(warning.message) for warning in caught
    )
    assert not (tmp_path / "results" / "_fs_run" / "cache" / "traces_vds.h5").exists()
    assert traces.groups == ["surface"]
    assert traces.manifest.frequencies == {1: 10.0, 2: 20.0, 3: 30.0}
    assert traces.frequencies("surface").tolist() == [10.0, 20.0, 30.0]
    assert traces.laplace("surface").tolist() == [0.0, 0.0, 0.0]
    assert traces.receivers("surface").tolist() == [101, 102]
    assert traces.sources("surface").tolist() == [7]
    assert traces.components("surface").tolist() == ["p"]

    fd = traces.fd("surface", "p", source=7)
    assert fd.dims == ("frequency", "receiver")
    assert fd.coords["frequency"].values.tolist() == [10.0, 20.0, 30.0]
    assert fd.coords["receiver"].values.tolist() == [101, 102]
    assert fd.values.real.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_trace_dataset_td_compensates_laplace_domain_amplitudes(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    packed = trace_dir / "traces.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    period = 1.0
    laplace = -np.log(10.0) / (2.0 * np.pi * period)

    with h5py.File(packed, "w") as h5:
        h5.create_dataset("frequency", data=np.array([1.0, 2.0]))
        h5.create_dataset("laplace", data=np.array([laplace, laplace]))
        h5.create_dataset(
            "survey/packed_layout_kind",
            data=np.array(["packed_frequency_trace_v1"], dtype=string_dtype),
        )
        dset = h5.create_dataset(
            "surface",
            data=np.array(
                [
                    [[[[1.0, 0.0]]]],
                    [[[[0.25, 0.0]]]],
                ],
                dtype=np.float32,
            ),
        )
        dset.attrs["dims"] = ["receiver", "component", "shot", "frequency"]
        dset.attrs["layout_kind"] = ["dense_trace_v1"]
        dset.attrs["receiver"] = np.array([101], dtype=np.int32)
        dset.attrs["component"] = np.array(["p"], dtype=string_dtype)
        dset.attrs["shot"] = np.array([7], dtype=np.int32)

    traces = TraceDataset.open(packed)
    wavelet = RickerWavelet(f=0.5, center=0.0)
    laplace_time = traces.ld("surface", "p", source=7, wavelet=wavelet)
    compensated = traces.td(
        "surface",
        "p",
        source=7,
        wavelet=RickerWavelet(f=0.5, center=0.0),
    )

    raw_time = np.linspace(0.0, period, laplace_time.sizes["time"] + 1)[:-1]
    gain = np.exp(-2.0 * np.pi * laplace * raw_time)

    np.testing.assert_allclose(
        compensated.values[:, 0],
        laplace_time.values[:, 0] * gain,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert traces.laplace("surface").tolist() == pytest.approx([laplace, laplace])
    assert laplace_time.attrs["domain"] == "laplace_time"
    assert laplace_time.attrs["laplace_compensated"] is False
    assert compensated.attrs["domain"] == "time"
    assert compensated.attrs["laplace_compensated"] is True
    assert compensated.attrs["damping_factor"] == pytest.approx(10.0)


def test_trace_dataset_td_ignores_zero_mixed_laplace_frequency(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    packed = trace_dir / "traces.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    frequencies = np.array([1.0, 2.0, 3.0])
    laplace = np.array([-0.25, -0.25, 0.0])
    data = np.zeros((frequencies.size, 1, 1, 1, 2), dtype=np.float32)
    data[0, 0, 0, 0, 0] = 1.0
    data[1, 0, 0, 0, 0] = 0.5

    with h5py.File(packed, "w") as h5:
        h5.create_dataset("frequency", data=frequencies)
        h5.create_dataset("laplace", data=laplace)
        h5.create_dataset(
            "survey/packed_layout_kind",
            data=np.array(["packed_frequency_trace_v1"], dtype=string_dtype),
        )
        dset = h5.create_dataset("surface", data=data)
        dset.attrs["dims"] = ["receiver", "component", "shot", "frequency"]
        dset.attrs["layout_kind"] = ["dense_trace_v1"]
        dset.attrs["receiver"] = np.array([101], dtype=np.int32)
        dset.attrs["component"] = np.array(["p"], dtype=string_dtype)
        dset.attrs["shot"] = np.array([7], dtype=np.int32)

    traces = TraceDataset.open(packed)
    td = traces.td(
        "surface",
        "p",
        source=7,
        wavelet=RickerWavelet(f=0.5, center=0.0),
    )

    assert td.attrs["laplace"] == pytest.approx(-0.25)
    assert td.attrs["laplace_compensated"] is True


def test_trace_dataset_td_rejects_nonzero_mixed_laplace_frequency(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    packed = trace_dir / "traces.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    frequencies = np.array([1.0, 2.0, 3.0])
    data = np.zeros((frequencies.size, 1, 1, 1, 2), dtype=np.float32)
    data[..., 0] = 1.0

    with h5py.File(packed, "w") as h5:
        h5.create_dataset("frequency", data=frequencies)
        h5.create_dataset("laplace", data=np.array([-0.25, -0.25, 0.0]))
        h5.create_dataset(
            "survey/packed_layout_kind",
            data=np.array(["packed_frequency_trace_v1"], dtype=string_dtype),
        )
        dset = h5.create_dataset("surface", data=data)
        dset.attrs["dims"] = ["receiver", "component", "shot", "frequency"]
        dset.attrs["layout_kind"] = ["dense_trace_v1"]
        dset.attrs["receiver"] = np.array([101], dtype=np.int32)
        dset.attrs["component"] = np.array(["p"], dtype=string_dtype)
        dset.attrs["shot"] = np.array([7], dtype=np.int32)

    traces = TraceDataset.open(packed)
    with pytest.raises(ValueError, match="uniform Laplace offset"):
        traces.td(
            "surface",
            "p",
            source=7,
            wavelet=RickerWavelet(f=0.5, center=0.0),
        )


def test_trace_dataset_fd_uses_ordinary_wavelet_spectrum_for_laplace_data(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    packed = trace_dir / "traces.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    frequencies = np.arange(0.0, 202.0, 2.0)
    laplace = np.full(frequencies.size, -0.25)
    data = np.zeros((frequencies.size, 1, 1, 1, 2), dtype=np.float32)
    data[..., 0] = 1.0

    with h5py.File(packed, "w") as h5:
        h5.create_dataset("frequency", data=frequencies)
        h5.create_dataset("laplace", data=laplace)
        h5.create_dataset(
            "survey/packed_layout_kind",
            data=np.array(["packed_frequency_trace_v1"], dtype=string_dtype),
        )
        dset = h5.create_dataset("surface", data=data)
        dset.attrs["dims"] = ["receiver", "component", "shot", "frequency"]
        dset.attrs["layout_kind"] = ["dense_trace_v1"]
        dset.attrs["receiver"] = np.array([101], dtype=np.int32)
        dset.attrs["component"] = np.array(["p"], dtype=string_dtype)
        dset.attrs["shot"] = np.array([7], dtype=np.int32)

    traces = TraceDataset.open(packed)
    wavelet = RickerWavelet(f=60.0, center=0.0)
    fd = traces.fd("surface", "p", source=7, wavelet=wavelet)

    base_wavelet = RickerWavelet(f=60.0, center=0.0)
    base_wavelet.times = traces.times(upscale=1)
    expected = xr.DataArray(
        base_wavelet.spectrum,
        dims=["frequency"],
        coords={"frequency": base_wavelet.frequencies},
    ).interp(frequency=frequencies, kwargs={"fill_value": 0})

    np.testing.assert_allclose(
        fd.values[:, 0],
        expected.values,
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_trace_dataset_td_spectrum_is_wavelet_limited_before_interpolation(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    packed = trace_dir / "traces.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    frequencies = np.arange(0.0, 126.0, 1.0)
    data = np.zeros((frequencies.size, 1, 1, 1, 2), dtype=np.float32)
    data[..., 0] = 1.0

    with h5py.File(packed, "w") as h5:
        h5.create_dataset("frequency", data=frequencies)
        h5.create_dataset(
            "survey/packed_layout_kind",
            data=np.array(["packed_frequency_trace_v1"], dtype=string_dtype),
        )
        dset = h5.create_dataset("surface", data=data)
        dset.attrs["dims"] = ["receiver", "component", "shot", "frequency"]
        dset.attrs["layout_kind"] = ["dense_trace_v1"]
        dset.attrs["receiver"] = np.array([101], dtype=np.int32)
        dset.attrs["component"] = np.array(["p"], dtype=string_dtype)
        dset.attrs["shot"] = np.array([7], dtype=np.int32)

    traces = TraceDataset.open(packed)
    wavelet = RickerWavelet(f=10.0, center=0.0373)
    td = traces.td("surface", "p", source=7, wavelet=wavelet, upscale=4)

    base_wavelet = RickerWavelet(f=10.0, center=0.0373)
    base_wavelet.times = traces.times(upscale=1)
    coarse_spectrum = xr.DataArray(
        base_wavelet.spectrum,
        dims=["frequency"],
        coords={"frequency": base_wavelet.frequencies},
    )
    expected_spectrum = coarse_spectrum.interp(
        frequency=wavelet.frequencies,
        kwargs={"fill_value": 0},
    ).values
    expected = np.fft.irfft(expected_spectrum)
    np.testing.assert_allclose(
        td.values[:, 0],
        expected,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    actual_spectrum = np.fft.rfft(td.values[:, 0])
    np.testing.assert_allclose(
        actual_spectrum,
        expected_spectrum,
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_trace_dataset_td_center_zero_does_not_inject_broadband_stripes(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    packed = trace_dir / "traces.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    frequencies = np.arange(0.0, 501.0, 1.0)
    data = np.zeros((frequencies.size, 1, 1, 1, 2), dtype=np.float32)
    data[..., 0] = 1.0

    with h5py.File(packed, "w") as h5:
        h5.create_dataset("frequency", data=frequencies)
        h5.create_dataset(
            "survey/packed_layout_kind",
            data=np.array(["packed_frequency_trace_v1"], dtype=string_dtype),
        )
        dset = h5.create_dataset("surface", data=data)
        dset.attrs["dims"] = ["receiver", "component", "shot", "frequency"]
        dset.attrs["layout_kind"] = ["dense_trace_v1"]
        dset.attrs["receiver"] = np.array([101], dtype=np.int32)
        dset.attrs["component"] = np.array(["p"], dtype=string_dtype)
        dset.attrs["shot"] = np.array([7], dtype=np.int32)

    traces = TraceDataset.open(packed)
    td = traces.td(
        "surface",
        "p",
        source=7,
        wavelet=RickerWavelet(f=40.0, center=0.0),
    )
    spectrum = np.abs(np.fft.rfft(td.values[:, 0]))
    high_frequency_level = np.mean(spectrum[-20:]) / np.max(spectrum)

    assert high_frequency_level < 1.0e-2


def test_trace_dataset_td_applies_wavelet_before_interpolating_oscillatory_response(
    tmp_path,
):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    packed = trace_dir / "traces.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    frequencies = np.arange(0.0, 127.0, 1.0)
    delay = 0.173
    response = np.exp(-2.0j * np.pi * frequencies * delay)
    data = np.zeros((frequencies.size, 1, 1, 1, 2), dtype=np.float32)
    data[:, 0, 0, 0, 0] = response.real
    data[:, 0, 0, 0, 1] = response.imag

    with h5py.File(packed, "w") as h5:
        h5.create_dataset("frequency", data=frequencies)
        h5.create_dataset(
            "survey/packed_layout_kind",
            data=np.array(["packed_frequency_trace_v1"], dtype=string_dtype),
        )
        dset = h5.create_dataset("surface", data=data)
        dset.attrs["dims"] = ["receiver", "component", "shot", "frequency"]
        dset.attrs["layout_kind"] = ["dense_trace_v1"]
        dset.attrs["receiver"] = np.array([101], dtype=np.int32)
        dset.attrs["component"] = np.array(["p"], dtype=string_dtype)
        dset.attrs["shot"] = np.array([7], dtype=np.int32)

    traces = TraceDataset.open(packed)
    wavelet = RickerWavelet(f=40.0, center=0.0)
    td = traces.td("surface", "p", source=7, wavelet=wavelet, upscale=4)

    base_wavelet = RickerWavelet(f=40.0, center=0.0)
    base_wavelet.times = traces.times(upscale=1)
    coarse_response = xr.DataArray(
        response * base_wavelet.spectrum,
        dims=["frequency"],
        coords={"frequency": frequencies},
    )
    expected_spectrum = coarse_response.interp(
        frequency=wavelet.frequencies,
        kwargs={"fill_value": 0},
    ).values

    actual_spectrum = np.fft.rfft(td.values[:, 0])
    np.testing.assert_allclose(
        actual_spectrum,
        expected_spectrum,
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_trace_store_reads_indexed_solver_packed_trace_file(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    trace_files = [trace_dir / f"traces_{idx}.h5" for idx in range(1, 4)]
    packed = trace_dir / "traces.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(packed, "w") as h5:
        h5.create_dataset("frequency", data=np.array([20.0, 10.0, 30.0, 40.0]))
        h5.create_dataset("laplace", data=np.zeros(4))
        h5.create_dataset("task_id", data=np.array([2, 1, 3, 4], dtype=np.int32))
        h5.create_dataset(
            "survey/packed_layout_kind",
            data=np.array(["indexed_frequency_trace_v1"], dtype=string_dtype),
        )
        catalog = h5.require_group("survey/receiver_groups/_catalog")
        catalog.create_dataset(
            "group_name", data=np.array(["surface"], dtype=string_dtype)
        )
        catalog.create_dataset(
            "dataset_path", data=np.array(["/surface"], dtype=string_dtype)
        )
        catalog.create_dataset(
            "layout_kind", data=np.array(["dense_trace_v1"], dtype=string_dtype)
        )
        trace_group = h5.require_group("survey/receiver_groups/surface/traces")
        trace_group.create_dataset(
            "receiver_id", data=np.array([101, 102], dtype=np.int32)
        )
        trace_group.create_dataset("source_id", data=np.array([7, 7], dtype=np.int32))
        trace_group.create_dataset(
            "component_name", data=np.array(["p", "p"], dtype=string_dtype)
        )
        h5.require_group("trace_index/datasets")
        h5.create_dataset(
            "trace_index/schema_version",
            data=np.array(["fs-trace-index-1"], dtype=string_dtype),
        )
        h5.create_dataset(
            "trace_index/layout_kind",
            data=np.array(["indexed_frequency_trace_v1"], dtype=string_dtype),
        )
        h5.create_dataset(
            "trace_index/data_root", data=np.array(["/trace_data"], dtype=string_dtype)
        )
        h5.create_dataset(
            "trace_index/dataset_number", data=np.array([1, 2, 3, 4], dtype=np.int32)
        )
        h5.create_dataset(
            "trace_index/frequency", data=np.array([20.0, 10.0, 30.0, 40.0])
        )
        h5.create_dataset("trace_index/laplace", data=np.zeros(4, dtype=np.float64))
        h5.create_dataset(
            "trace_index/task_id", data=np.array([2, 1, 3, 4], dtype=np.int32)
        )
        h5.create_dataset(
            "trace_index/shard_file",
            data=np.array(
                [
                    "traces/shards/traces_2.h5",
                    "traces/shards/traces_1.h5",
                    "traces/shards/traces_3.h5",
                    "traces/shards/traces_4.h5",
                ],
                dtype=string_dtype,
            ),
        )
        h5.create_dataset(
            "trace_index/datasets/dataset_number",
            data=np.array([1, 2, 3, 4], dtype=np.int32),
        )
        h5.create_dataset(
            "trace_index/datasets/source_path",
            data=np.array(
                ["/surface", "/surface", "/surface", "/surface"],
                dtype=string_dtype,
            ),
        )
        h5.create_dataset(
            "trace_index/datasets/packed_path",
            data=np.array(
                [
                    "/trace_data/surface/000001",
                    "/trace_data/surface/000002",
                    "/trace_data/surface/000003",
                    "/trace_data/surface/000004",
                ],
                dtype=string_dtype,
            ),
        )
        for number, values in {
            "000001": [3.0, 4.0],
            "000002": [1.0, 2.0],
            "000003": [5.0, 6.0],
            "000004": [7.0, 8.0],
        }.items():
            dset = h5.create_dataset(
                f"trace_data/surface/{number}",
                data=np.array(
                    [[[[values[0], 0.0], [values[1], 0.0]]]], dtype=np.float32
                ),
            )
            dset.attrs["dims"] = ["receiver", "component", "shot"]
            dset.attrs["layout_kind"] = ["dense_trace_v1"]
            dset.attrs["receiver"] = np.array([101, 102], dtype=np.int32)
            dset.attrs["component"] = np.array(["p"], dtype=string_dtype)
            dset.attrs["shot"] = np.array([7], dtype=np.int32)

    traces = TraceDataset.from_manifest(
        TraceManifest(
            files=trace_files,
            frequencies={1: 10.0, 2: 20.0, 3: 30.0},
            groups=["surface"],
            simulation=tmp_path / "simulation.json",
            result_path=tmp_path / "results",
            output_path=trace_dir,
            project_path=tmp_path,
        )
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        consolidated = traces.consolidate()

    assert consolidated == packed
    assert not any(
        "Trace file is missing" in str(warning.message) for warning in caught
    )
    assert traces.groups == ["surface"]
    assert traces.frequencies("surface").tolist() == [10.0, 20.0, 30.0]
    assert traces.receivers("surface").tolist() == [101, 102]
    assert traces.sources("surface").tolist() == [7]
    assert traces.components("surface").tolist() == ["p"]

    fd = traces.fd("surface", "p", source=7)
    assert fd.dims == ("frequency", "receiver")
    assert fd.coords["frequency"].values.tolist() == [10.0, 20.0, 30.0]
    assert fd.coords["receiver"].values.tolist() == [101, 102]
    assert fd.values.real.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_trace_store_detects_sibling_packed_trace_file_before_warning(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    shard_dir = trace_dir / "shards"
    trace_files = [shard_dir / f"traces_{idx}.h5" for idx in range(1, 4)]
    packed = trace_dir / "traces.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(packed, "w") as h5:
        h5.create_dataset("frequency", data=np.array([10.0, 20.0, 30.0]))
        dset = h5.create_dataset(
            "surface",
            data=np.array(
                [
                    [[[[1.0, 0.0], [2.0, 0.0]]]],
                    [[[[3.0, 0.0], [4.0, 0.0]]]],
                    [[[[5.0, 0.0], [6.0, 0.0]]]],
                ],
                dtype=np.float32,
            ),
        )
        dset.attrs["dims"] = ["receiver", "component", "shot", "frequency"]
        dset.attrs["layout_kind"] = ["dense_trace_v1"]
        dset.attrs["receiver"] = np.array([101, 102], dtype=np.int32)
        dset.attrs["component"] = np.array(["p"], dtype=string_dtype)
        dset.attrs["shot"] = np.array([7], dtype=np.int32)

    traces = TraceDataset.from_manifest(
        TraceManifest(
            files=trace_files,
            frequencies={1: 10.0, 2: 20.0, 3: 30.0},
            groups=["surface"],
            simulation=tmp_path / "simulation.json",
            result_path=tmp_path / "results",
            output_path=trace_dir,
            project_path=tmp_path,
        )
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        consolidated = traces.consolidate()

    assert consolidated == packed
    assert not any(
        "Trace file is missing" in str(warning.message) for warning in caught
    )
    assert traces.frequencies("surface").tolist() == [10.0, 20.0, 30.0]


def test_trace_store_omits_missing_later_trace_files_and_refreshes(tmp_path):
    trace_files = []
    string_dtype = h5py.string_dtype(encoding="utf-8")
    for idx, freq in enumerate([10.0, 20.0, 30.0], start=1):
        file = tmp_path / f"traces_{idx}.h5"
        trace_files.append(file)
        if idx == 2:
            continue
        with h5py.File(file, "w") as h5:
            h5.create_dataset("frequency", data=freq)
            dset = h5.create_dataset(
                "surface",
                data=np.full((1, 1, 1, 2), idx, dtype=np.float32),
            )
            if idx == 1:
                dset.attrs["dims"] = ["receiver", "component", "shot"]
                trace_group = h5.require_group("/survey/receiver_groups/surface/traces")
                trace_group.create_dataset(
                    "receiver_id",
                    data=np.array([101], dtype=np.int32),
                )
                trace_group.create_dataset(
                    "source_id",
                    data=np.array([7], dtype=np.int32),
                )
                trace_group.create_dataset(
                    "component_name",
                    data=np.array(["p"], dtype=string_dtype),
                )

    traces = TraceDataset.from_manifest(
        TraceManifest(
            files=trace_files,
            frequencies={1: 10.0, 2: 20.0, 3: 30.0},
            groups=["surface"],
            simulation=tmp_path / "simulation.json",
            result_path=tmp_path / "results",
            output_path=tmp_path / "results" / "traces",
            project_path=tmp_path,
        )
    )

    with pytest.warns(RuntimeWarning, match="Trace file is missing"):
        traces.consolidate()
    assert traces.frequencies("surface").tolist() == [10.0, 30.0]

    with h5py.File(trace_files[1], "w") as h5:
        h5.create_dataset("frequency", data=20.0)
        h5.create_dataset(
            "surface",
            data=np.full((1, 1, 1, 2), 2, dtype=np.float32),
        )

    traces.consolidate()
    assert traces.frequencies("surface").tolist() == [10.0, 20.0, 30.0]


def test_trace_store_can_consolidate_when_first_trace_file_is_missing(tmp_path):
    trace_files = [tmp_path / f"traces_{idx}.h5" for idx in range(1, 4)]
    string_dtype = h5py.string_dtype(encoding="utf-8")
    for idx, file in enumerate(trace_files[1:], start=2):
        with h5py.File(file, "w") as h5:
            h5.create_dataset("frequency", data=10.0 * idx)
            dset = h5.create_dataset(
                "surface",
                data=np.full((1, 1, 1, 2), idx, dtype=np.float32),
            )
            if idx == 2:
                dset.attrs["dims"] = ["receiver", "component", "shot"]
                trace_group = h5.require_group("/survey/receiver_groups/surface/traces")
                trace_group.create_dataset(
                    "receiver_id",
                    data=np.array([101], dtype=np.int32),
                )
                trace_group.create_dataset(
                    "source_id",
                    data=np.array([7], dtype=np.int32),
                )
                trace_group.create_dataset(
                    "component_name",
                    data=np.array(["p"], dtype=string_dtype),
                )

    traces = TraceDataset.from_manifest(
        TraceManifest(
            files=trace_files,
            frequencies={1: 10.0, 2: 20.0, 3: 30.0},
            groups=["surface"],
            simulation=tmp_path / "simulation.json",
            result_path=tmp_path / "results",
            output_path=tmp_path / "results" / "traces",
            project_path=tmp_path,
        )
    )

    with pytest.warns(RuntimeWarning, match="Trace file is missing"):
        traces.consolidate()

    assert traces.frequencies("surface").tolist() == [20.0, 30.0]
    assert traces.receivers("surface").tolist() == [101]


def test_trace_store_normalizes_dense_trace_axis_order(tmp_path):
    trace_files = []
    string_dtype = h5py.string_dtype(encoding="utf-8")
    for idx, freq in enumerate([10.0, 20.0], start=1):
        file = tmp_path / f"traces_{idx}.h5"
        trace_files.append(str(file))
        with h5py.File(file, "w") as h5:
            h5.create_dataset("frequency", data=freq)
            dset = h5.create_dataset(
                "surface",
                data=np.zeros((1, 1, 3, 2), dtype=np.float32),
            )
            dset.attrs["dims"] = ["receiver", "component", "shot"]
            dset.attrs["layout_kind"] = ["dense_trace_v1"]
            trace_group = h5.require_group("/survey/receiver_groups/surface/traces")
            trace_group.create_dataset(
                "receiver_id",
                data=np.array([101, 102, 103], dtype=np.int32),
            )
            trace_group.create_dataset(
                "source_id",
                data=np.array([7, 7, 7], dtype=np.int32),
            )
            trace_group.create_dataset(
                "component_name",
                data=np.array(["p", "p", "p"], dtype=string_dtype),
            )

    traces = TraceDataset.from_manifest(
        TraceManifest(
            files=[Path(file) for file in trace_files],
            frequencies={1: 10.0, 2: 20.0},
            groups=["surface"],
            simulation=tmp_path / "simulation.json",
            result_path=tmp_path / "results",
            output_path=tmp_path / "results" / "traces",
            project_path=tmp_path,
        )
    )

    fd = traces.fd("surface", "p", source=7)

    assert fd.dims == ("frequency", "receiver")
    assert fd.shape == (2, 3)
    assert fd.coords["receiver"].values.tolist() == [101, 102, 103]


def test_sparse_survey_authoring_exports_fast_solver_contract():
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")

    survey = SparseSurvey("marine")
    survey.add_trace(
        source=1,
        receiver=101,
        point=1,
        component="p",
        receiver_name="R101",
        source_name="S1",
    )
    survey.add_trace(source=1, receiver=203, point=2, component="p")
    acq.add_sparse_receiver_group(
        "streamer",
        hydrophone,
        coords=[[0.0, 0.0], [1.0, 0.0]],
        survey=survey,
    )

    payload = acq.to_fs()
    group = payload["receiver_groups"][0]
    sparse = payload["surveys"][0]

    assert group["sampling"] == {"_type": "Sparse", "survey": "marine"}
    assert sparse["_type"] == "Sparse"
    assert sparse["traces"][0]["trace_id"] == 1
    assert sparse["traces"][0]["source_id"] == 1
    assert sparse["traces"][0]["receiver_id"] == 101
    assert sparse["traces"][0]["receiver_position_id"] == 101
    assert sparse["traces"][0]["point_first"] == 1
    assert sparse["traces"][0]["point_last"] == 1
    assert sparse["traces"][0]["component"] == 1
    assert sparse["traces"][0]["component_name"] == "p"
    assert sparse["traces"][1]["receiver_id"] == 203
    assert sparse["traces"][1]["point_first"] == 2


def test_sparse_hdf5_survey_reference_does_not_touch_server_file():
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.5, 0.0]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    survey = SparseSurvey.file("marine", "/server/surveys/marine_layout.h5")

    acq.add_sparse_receiver_group(
        "streamer",
        hydrophone,
        coords=[[0.0, 0.0]],
        survey=survey,
    )

    payload = acq.to_fs()

    assert payload["receiver_groups"][0]["sampling"] == {
        "_type": "HDF5TraceStore",
        "survey": "marine",
    }
    assert payload["surveys"][0] == {
        "name": "marine",
        "_type": "HDF5TraceStore",
        "layout_file": "/server/surveys/marine_layout.h5",
    }


def test_sparse_survey_writes_fast_solver_hdf5_trace_store(tmp_path):
    survey = SparseSurvey("marine")
    survey.add_trace(source=1, receiver=1, component=1, point=1, component_name="p")
    survey.add_eval_sample(sample_id=1, point=1, receiver_position=1)
    survey.add_trace_sample(trace=1, sample=1, component=1, weight=0.5)

    file = survey.write_hdf5(tmp_path / "marine_layout.h5")

    with h5py.File(file, "r") as h5:
        assert h5["/survey/traces/trace_id"][:].tolist() == [1]
        assert h5["/survey/traces/source_id"][:].tolist() == [1]
        assert h5["/survey/traces/receiver_id"][:].tolist() == [1]
        assert h5["/survey/traces/receiver_position_id"][:].tolist() == [1]
        assert h5["/survey/traces/component_id"][:].tolist() == [1]
        assert h5["/survey/eval_samples/sample_id"][:].tolist() == [1]
        assert h5["/survey/eval_samples/receiver_position_id"][:].tolist() == [1]
        assert h5["/survey/trace_samples/weight"][:].tolist() == [0.5]


def test_sparse_survey_roundtrip_does_not_mutate_input():
    data = {
        "name": "marine",
        "_type": "Sparse",
        "traces": [
            {
                "trace_id": 1,
                "source_id": 1,
                "receiver_id": 10,
                "receiver_position_id": 10,
                "component": 1,
                "point_first": 1,
            }
        ],
        "eval_samples": [{"sample_id": 1, "point_id": 1, "recveiver_position_id": 10}],
        "advanced_layout_flag": True,
    }
    original = copy.deepcopy(data)

    survey = SparseSurvey.from_fs(data)
    payload = survey.to_fs()

    assert data == original
    assert payload["advanced_layout_flag"] is True
    assert payload["eval_samples"][0]["receiver_position_id"] == 10
    assert payload["eval_samples"][0]["recveiver_position_id"] == 10


def test_boundary_condition_serializes_multiple_conditions_without_name():
    bc = BoundaryCondition(
        conditions=["free", "sealed"],
        boundaries=["z_min"],
    )

    payload = bc.to_fs()

    assert bc.conditions == ["free", "sealed"]
    assert payload == {
        "conditions": ["free", "sealed"],
        "boundaries": ["z_min"],
    }


def test_boundary_condition_from_fs_uses_current_shape_and_preserves_extra():
    bc = BoundaryCondition.from_fs(
        {
            "name": "top",
            "conditions": ["free", "sealed"],
            "boundaries": ["z_min"],
            "solver_bc_flag": True,
        }
    )

    assert bc.conditions == ["free", "sealed"]
    assert bc.to_fs()["solver_bc_flag"] is True


def test_boundary_condition_rejects_kind_alias():
    with pytest.raises(TypeError, match="kind"):
        BoundaryCondition(kind="free", boundaries=["z_min"])
    with pytest.raises(TypeError, match="kind"):
        BoundaryCondition.from_fs({"kind": "free", "boundaries": ["z_min"]})


def test_boundary_condition_serializes_pml_reflectivity():
    bc = BoundaryCondition(
        conditions=["pml"],
        boundaries=["x_min"],
        pml_reflectivity=1e-2,
    )
    from_fs = BoundaryCondition.from_fs(
        {
            "conditions": ["pml"],
            "boundaries": ["x_min"],
            "pml_reflectivity": 1e-3,
        }
    )

    assert bc.pml_reflectivity == 1e-2
    assert bc.to_fs()["pml_reflectivity"] == 1e-2
    assert "pml_reflection" not in bc.to_fs()
    assert from_fs.to_fs()["pml_reflectivity"] == 1e-3
    with pytest.raises(ValueError, match="pml_reflectivity or pml_reflection"):
        BoundaryCondition(
            conditions=["pml"],
            boundaries=["x_min"],
            pml_reflection=1e-3,
            pml_reflectivity=1e-2,
        )


def test_boundary_condition_accepts_legacy_pml_reflection_quietly():
    signature = inspect.signature(BoundaryCondition)
    bc = BoundaryCondition(
        conditions=["pml"],
        boundaries=["x_min"],
        pml_reflection=2.5e-4,
    )
    from_fs = BoundaryCondition.from_fs(
        {
            "conditions": ["pml"],
            "boundaries": ["x_min"],
            "pml_reflection": 3.5e-4,
        }
    )

    assert "pml_reflection" not in signature.parameters
    assert bc.pml_reflectivity == 2.5e-4
    assert bc.to_fs()["pml_reflectivity"] == 2.5e-4
    assert "pml_reflection" not in bc.to_fs()
    assert from_fs.to_fs()["pml_reflectivity"] == 3.5e-4
    assert "pml_reflection" not in from_fs.to_fs()


def test_boundary_conditions_collection_allows_shared_boundary_conditions():
    conditions = BoundaryConditions()
    conditions += BoundaryCondition(
        name="free_surface",
        conditions=["free"],
        boundaries=["z_min"],
    )
    conditions += BoundaryCondition(
        name="sealed_surface",
        conditions=["sealed"],
        boundaries=["z_min"],
    )

    assert len(conditions) == 2
    assert conditions["free_surface"].conditions == ["free"]
    assert [bc.boundaries for bc in conditions] == [
        ["z_min"],
        ["z_min"],
    ]
    assert conditions._boundaries == {"z_min"}


def test_boundary_conditions_from_fs_uses_flat_named_list():
    conditions = BoundaryConditions.from_fs(
        [
            {
                "name": "top",
                "conditions": ["free"],
                "boundaries": [101],
            }
        ]
    )

    payload = conditions.to_fs()

    assert conditions["top"].boundaries == [101]
    assert payload == [
        {"conditions": ["free"], "boundaries": [101], "name": "top"},
    ]
    with pytest.raises(TypeError, match="BCs must be a list"):
        BoundaryConditions.from_fs({"boundary_conditions": []})


def test_simulation_accepts_boundary_conditions_directly(tmp_path):
    sim = SeismicSimulation(
        name="sim",
        physics="poroelastic",
        dimension=2,
        project_path=tmp_path,
        mesh=MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1])),
        BCs=[
            {
                "name": "a_bc",
                "conditions": ["free", "sealed"],
                "boundaries": ["z_min"],
            }
        ],
    )

    sim += BoundaryCondition(name="pml_xmin", conditions=["pml"], boundaries=["x_min"])

    payload = sim.to_fs()

    assert sim.BCs["a_bc"].conditions == ["free", "sealed"]
    assert "__dict__" not in SeismicSimulation.__dict__
    assert "__dict__" not in Acquisition.__dict__
    assert payload["BCs"] == [
        {
            "conditions": ["free", "sealed"],
            "boundaries": ["z_min"],
            "name": "a_bc",
        },
        {
            "conditions": ["pml"],
            "boundaries": ["x_min"],
            "pml_wavelengths": 0.5,
            "pml_exponent": 3.0,
            "pml_reflectivity": 0.01,
            "name": "pml_xmin",
        },
    ]
