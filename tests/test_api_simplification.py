import copy
import hashlib
import json
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

from frequensolve.geometry.frame import CoordinateSystem, CoordinateValue, Direction
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import DistanceGrading, MeshManager, SurfaceGrading
from frequensolve.model.model import ModelBase, ModelSubdomain
from frequensolve.model.property import Property, prop
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import (
    CoordsFromFile,
    ReceiverComponent,
    ReceiverNode,
)
from frequensolve.seismic.sparse_survey import SparseSurvey
from frequensolve.seismic.traces import TraceDataset
from frequensolve.simulation.artifacts import OutputArtifact, TraceManifest
from frequensolve.simulation.jobs import FrequencyDomainJob
from frequensolve.simulation.outputs import (
    JobOutputs,
    ParaviewOutput,
    TraceOutput,
    WavefieldOutput,
)
from frequensolve.simulation.physics import ElasticComponents
from frequensolve.simulation.simulation import SeismicSimulation
from frequensolve.units import UnitConfig
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


def test_legacy_string_property_loads_as_structured_ref():
    subdomain = ModelSubdomain(
        mesh_block_id=1,
        properties={"Vp": "remote:/server/vp.bin|0.001|xz"},
    )

    payload = subdomain.to_fs()

    assert payload["properties"]["vp"]["file"] == "/server/vp.bin"
    assert payload["properties"]["vp"]["absolute"] is True
    assert payload["properties"]["vp"]["scale"] == 0.001


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
        "type": "surface",
        "name": "free_surface",
        "surface": "top",
        "normal": "up",
    }
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
    acq.add_source_group(
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
    assert payload["source_groups"][0]["source"]["coordinates"] == {
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
            "type": "surface",
            "name": "free_surface",
            "surface": "top",
            "normal": "up",
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
    acq.add_source_group(kind="scalar", coords=sources)
    acq.add_receiver_group(name="surface", device=hydrophone, coords=receivers)
    sim.acquisition = acq

    payload = sim.to_fs()

    assert payload["coordinate_systems"] == [
        {"type": "surface", "name": "top", "surface": "top", "normal": "up"},
    ]
    assert payload["Acquisition"]["source_groups"][0]["source"]["coordinates"] == {
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


def test_solver_frame_key_is_not_exported_and_legacy_input_is_ignored():
    subdomain = ModelSubdomain.from_fs(
        {"mesh_block_id": 1, "frame": "reference", "properties": {"vp": 1500.0}}
    )
    assert "frame" not in subdomain.to_fs()

    acq = Acquisition()
    acq.add_source_group(kind="scalar", coords=[[0.5, 0.0]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq.add_receiver_group(name="surface", device=hydrophone, coords=[[0.0, 0.0]])

    payload = acq.to_fs()
    assert "frame" not in payload["source_groups"][0]["source"]
    assert "frame" not in payload["receiver_groups"][0]

    legacy = Acquisition.from_fs(
        {
            "source_groups": [
                {
                    "source": {
                        "_type": "PointSource",
                        "name": "shot",
                        "kind": "scalar",
                        "frame": "reference",
                        "coordinates": [0.5, 0.0],
                    }
                }
            ],
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
    assert "frame" not in legacy_payload["source_groups"][0]["source"]
    assert "frame" not in legacy_payload["receiver_groups"][0]

    with pytest.raises(TypeError, match="frame"):
        acq.add_receiver_group(
            name="bad", device=hydrophone, coords=[[0.0, 0.0]], frame="reference"
        )


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
    assert WavefieldOutput(fields=["velocity"]).fields == ["velocity"]
    assert WavefieldOutput.from_fs({"fields": ["velocity"]}).fields == ["velocity"]


def test_output_paths_must_be_relative_to_result_directory():
    with pytest.raises(ValueError, match="relative"):
        TraceOutput(path="/tmp/traces")
    with pytest.raises(ValueError, match="relative"):
        ParaviewOutput(path="/tmp/paraview")
    with pytest.raises(ValueError, match="relative"):
        WavefieldOutput(path="/tmp/wavefields")


def test_job_outputs_adds_outputs_and_always_exports_traces():
    outputs = JobOutputs()

    outputs += ParaviewOutput(name="pv", fields=["pressure"])
    payload = outputs.to_fs()

    assert payload["traces"]["path"] == "traces"
    assert payload["ParaView"][0]["name"] == "pv"


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


def test_paraview_defaults_to_vtu_appended_binary():
    output = ParaviewOutput(name="pv", fields=["pressure"])
    payload = output.to_fs()

    assert output.format == "vtu"
    assert payload["writer"] == {"format": "vtu", "encoding": "appended"}
    assert "target" not in payload
    assert "source" not in payload

    with pytest.raises(ValueError, match="format"):
        ParaviewOutput(name="pv", fields=["pressure"], format="vtk")


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

    with pytest.raises(ValueError, match="parts"):
        ParaviewOutput(fields=["pressure"], parts="magnitude")


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
    assert payload["items"] == [{"field": "pressure"}]
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
            "adapt": {"min_epw": 2.0},
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


def test_mesh_surface_gradings_export_to_sauce_contract():
    mesh = MeshManager(
        HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1], units="km")
    )
    mesh.set_adapt(
        min_epw={"x": 2.0, "z": 3.0},
        f_adapt=10.0,
        surface_gradings=[
            SurfaceGrading(surface="top", d0=0.0, d1=0.05, mult=2.5),
            {
                "surface": "interface",
                "mode": "inside",
                "d0": 0.02,
                "d1": 0.1,
                "mult_max": 4.0,
                "mult_min": 1.25,
                "phi_scale": -1.0,
            },
        ],
    )

    payload = mesh.to_fs()

    assert payload["adapt"]["surface_gradings"] == [
        {
            "surface": "top",
            "mode": "abs_band",
            "d0": 0.0,
            "d1": 0.05,
            "mult": 2.5,
        },
        {
            "surface": "interface",
            "mode": "inside",
            "d0": 0.02,
            "d1": 0.1,
            "mult_max": 4.0,
            "mult_min": 1.25,
            "phi_scale": -1.0,
        },
    ]


def test_mesh_source_receiver_gradings_export_to_sauce_contract():
    mesh = MeshManager(
        HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1], units="km")
    )
    mesh.set_adapt(
        min_epw=2.0,
        source_grading=DistanceGrading(d0=0.01, d1=0.08, mult=4.0),
        receiver_grading={"d0": 0.02, "d1": 0.12, "mult_max": 3.0},
    )

    payload = mesh.to_fs()

    assert payload["adapt"]["src_grading"] == {
        "d0": 0.01,
        "d1": 0.08,
        "mult": 4.0,
    }
    assert payload["adapt"]["rcv_grading"] == {
        "d0": 0.02,
        "d1": 0.12,
        "mult": 3.0,
    }


def test_mesh_source_receiver_gradings_are_editable():
    mesh = MeshManager()
    mesh.set_adapt(min_epw=2.0)
    mesh.set_source_grading(d0=0.0, d1=25.0, mult=2.0)
    mesh.set_receiver_grading(d0=5.0, d1=40.0, mult=3.0)
    mesh.adapt.source_grading.mult = 2.5
    mesh.adapt.receiver_grading.d1 = 45.0

    payload = mesh.adapt.to_fs()

    assert payload["src_grading"]["mult"] == 2.5
    assert payload["rcv_grading"]["d1"] == 45.0


def test_mesh_surface_gradings_accept_mapping_and_are_editable():
    mesh = MeshManager()
    mesh.set_adapt(
        min_epw=2.0,
        surface_gradings={
            "fault": {"d1": 50.0, "mult": 3.0, "mode": "band"},
        },
    )
    mesh.add_surface_grading("free_surface", d1=10.0, mult=2.0)
    mesh.adapt.surface_gradings[0].mult = 4.0

    payload = mesh.adapt.to_fs()

    assert payload["surface_gradings"][0]["surface"] == "fault"
    assert payload["surface_gradings"][0]["mult"] == 4.0
    assert payload["surface_gradings"][1]["surface"] == "free_surface"


def test_mesh_surface_gradings_roundtrip_and_preserve_extra():
    data = {
        "adapt": {
            "min_epw": 2.0,
            "surface_gradings": [
                {
                    "surface": "interface",
                    "d1": 0.15,
                    "mult": 2.0,
                    "custom_solver_flag": True,
                }
            ],
            "src_grading": {"d0": 0.01, "d1": 0.08, "mult": 4.0},
            "rcv_grading": {"d0": 0.02, "d1": 0.12, "mult": 3.0},
            "adapt_sources": 1,
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
    assert payload["adapt"]["adapt_sources"] == 1
    assert payload["adapt"]["surface_gradings"][0]["custom_solver_flag"] is True
    assert payload["adapt"]["src_grading"] == {"d0": 0.01, "d1": 0.08, "mult": 4.0}
    assert payload["adapt"]["rcv_grading"] == {"d0": 0.02, "d1": 0.12, "mult": 3.0}


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
    acq.add_source_group(kind="scalar", coords=[[0.5, 0.0]])
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

    payload = sim.to_fs()
    coords = payload["Acquisition"]["receiver_groups"][0]["coordinates"]

    assert (
        coords["file"]
        == "simulations/simple/simple.h5:inputs/acquisition/receivers/surface/coordinates"
    )
    assert coords["hash"].startswith("blake3:")
    with h5py.File(tmp_path / "simulations/simple/simple.h5", "r") as h5:
        assert "inputs/acquisition/receivers/surface/coordinates" in h5


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
    acq.add_source_group(kind="scalar", coords=[[0.5, 0.0]])
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

    job += WavefieldOutput(name="pressure", fields=["pressure"])
    assert not job.is_run_current()
    job.save()
    job.write_run_state(status="completed")
    assert job.is_run_current()

    payload = json.loads(sim._file.read_text())
    payload["Solver"]["max_iter"] = 123
    sim._file.write_text(json.dumps(payload))

    assert not job.is_run_current()


def test_job_run_current_accepts_sauce_run_manifest_hashes(tmp_path):
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


def test_job_run_current_accepts_current_sauce_run_manifest_schema(tmp_path):
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
                ],
            }
        )
    )
    with h5py.File(packed, "w") as h5:
        h5.create_dataset("frequency", data=np.array([10.0, 20.0, 30.0]))
        h5.create_dataset("laplace", data=np.zeros(3))
        h5.create_dataset("task_id", data=np.array([1, 2, 3], dtype=np.int32))
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
    assert traces.frequencies("surface").tolist() == [10.0, 20.0, 30.0]
    assert traces.receivers("surface").tolist() == [101, 102]
    assert traces.sources("surface").tolist() == [7]
    assert traces.components("surface").tolist() == ["p"]

    fd = traces.fd("surface", "p", source=7)
    assert fd.dims == ("frequency", "receiver")
    assert fd.coords["frequency"].values.tolist() == [10.0, 20.0, 30.0]
    assert fd.coords["receiver"].values.tolist() == [101, 102]
    assert fd.values.real.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


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


def test_sparse_survey_authoring_exports_sauce_contract():
    acq = Acquisition()
    acq.add_source_group(kind="scalar", coords=[[0.5, 0.0]])
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
    acq.add_source_group(kind="scalar", coords=[[0.5, 0.0]])
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


def test_sparse_survey_writes_sauce_hdf5_trace_store(tmp_path):
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
