import copy
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from frequensolve.imaging import controls as im
from frequensolve.imaging._artifacts import (
    ControlRegistryManifest,
    ControlStateFile,
    ControlVectorFile,
)
from frequensolve.model import LayeredModel
from frequensolve.model.implicit_geometry import RBFSurface
from frequensolve.model.parameterization import (
    BSplineControl,
    HatControl,
    MeshControl,
    ParameterizedProperty,
    TensorHatControl,
)
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.sources import SourceGeometry
from frequensolve.simulation.simulation import SeismicSimulation
from frequensolve.units import ureg as u

CONTRACT_ROOT = (
    Path(__file__).parent / "contracts" / "sauce-f533e6f" / "trunk" / "contracts"
)
MATERIAL_SCHEMA = CONTRACT_ROOT / "inputs" / "fs-material-model-1" / "schema.json"


def _registry() -> Registry:
    registry = Registry()
    for schema_file in list(CONTRACT_ROOT.rglob("schema.json")) + list(
        CONTRACT_ROOT.glob("fragments/*.json")
    ):
        contents = json.loads(schema_file.read_text())
        if "$id" not in contents:
            continue
        registry = registry.with_resource(
            contents["$id"], Resource.from_contents(contents)
        )
    return registry


def _material_validator() -> Draft202012Validator:
    schema_id = json.loads(MATERIAL_SCHEMA.read_text())["$id"]
    return Draft202012Validator({"$ref": schema_id}, registry=_registry())


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

WATER = (0.0, 200.0)
SEDIMENT = (200.0, 1500.0)
X_LIMITS = (0.0, 4000.0)


def _layered_simulation(tmp_path, *, sources=2, kind="scalar") -> SeismicSimulation:
    model = LayeredModel(name="shelf", dimension=2, x_limits=list(X_LIMITS))
    model.add_surface(WATER[0], name="top")
    model.add_layer(
        name="water", physics="acoustic", properties={"vp": 1500.0, "rho": 1000.0}
    )
    model.add_surface(WATER[1], name="seabed")
    model.add_layer(
        name="sediment",
        physics="acoustic",
        properties={"vp": 1900.0, "rho": 2000.0, "Sp": 0.5},
    )
    model.add_surface(SEDIMENT[1], name="bottom")
    model += RBFSurface(
        name="salt_top",
        support_radius=800.0,
        centers=[[1000.0, 900.0], [2000.0, 900.0], [3000.0, 900.0]],
        coefficients=[-100.0, -200.0, -100.0],
        bias=50.0,
    )
    coords = [[1000.0 * (i + 1), 10.0] for i in range(sources)]
    return SeismicSimulation(
        name="shelf",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
        model=model,
        acquisition=Acquisition(
            source_geometry=SourceGeometry.points(kind=kind, coords=coords)
        ),
    )


@pytest.fixture
def simulation(tmp_path):
    return _layered_simulation(tmp_path)


def _full_space() -> im.ControlSpace:
    return im.ControlSpace(
        vp=im.DepthProfile(
            "vp", "sediment", spacing=100.0, transform="log", limits=(1450, 3500)
        ),
        rho=im.DepthProfile.bspline("rho", "sediment", count=6, transform="log"),
        grid=im.GridParameters(["vp", "rho"], "water", spacing=[1000.0, 100.0]),
        salt=im.InterfaceParameters("salt_top", maximum_displacement=150.0),
        src=im.SourceParameters(position=True, signature=True),
        refl=im.ReflectivityParameters(
            "vp_ip", fields=[im.ReflectivityField("ip", layer=2, axis=2, basis="vp")]
        ),
    )


# ---------------------------------------------------------------------------
# block validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: im.DepthProfile("vp", "sediment"), "exactly one of"),
        (lambda: im.DepthProfile("vp", "sediment", spacing=10, count=3), "exactly one"),
        (lambda: im.DepthProfile("vp", "sediment", count=1), "at least 2"),
        (lambda: im.DepthProfile("vp", "sediment", spacing=-1.0), "positive"),
        (
            lambda: im.DepthProfile("vp", "sediment", nodes=[0.0, 1.0, 0.5]),
            "increasing",
        ),
        (
            lambda: im.DepthProfile("vp", "sediment", count=3, transform="exp"),
            "transform",
        ),
        (
            lambda: im.DepthProfile("vp", "sediment", count=3, limits=(2, 1)),
            "lower < upper",
        ),
        (lambda: im.DepthProfile("vp", "sediment", count=3, limits=(1,)), "pair"),
        (lambda: im.DepthProfile("vp", "sediment", count=3, id="a/b"), "HDF5-safe"),
        (lambda: im.DepthProfile("vp", "sediment", count=3, degree=0), "degree"),
        (
            lambda: im.GridParameters("vp", shape=[4, 4], spacing=[1.0, 1.0]),
            "exactly one",
        ),
        (lambda: im.GridParameters("vp", shape=[1, 4]), ">= 2"),
        (lambda: im.GridParameters(["vp", "vp"], shape=[4, 4]), "distinct"),
        (
            lambda: im.MeshParameters("vp", "sediment", frequency=0.0, epw=2.0),
            "positive",
        ),
        (
            lambda: im.MeshParameters("vp", "sediment", 5.0, 2.0, artifact="x.json"),
            ".h5",
        ),
        (lambda: im.InterfaceParameters("salt", maximum_displacement=0.0), "positive"),
        (lambda: im.SourceParameters(signature=False), "at least one"),
        (lambda: im.SourceParameters(sources=[1, 1]), "distinct"),
        (lambda: im.SourceParameters(sources="some"), "'all'"),
        (lambda: im.ReflectivityField("1ip", 1, 1, basis="vp"), "A-Za-z"),
        (lambda: im.ReflectivityField("ip", 0, 1, basis="vp"), "one-based"),
        (lambda: im.ReflectivityField("ip", 1, 1), "exactly one of basis"),
        (
            lambda: im.ReflectivityParameters(
                "vp_ip", fields=[im.ReflectivityField("ip", 1, 3, basis="vp")]
            ),
            "axis must be <= 2",
        ),
        (lambda: im.ReflectivityParameters("vp_rho", fields=[]), "parameterization"),
    ],
)
def test_blocks_reject_invalid_authoring(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_blocks_are_frozen_and_normalize_inputs():
    block = im.DepthProfile("vp", "sediment", nodes=[0, 10, 20], transform="LOG")
    assert block.nodes == (0.0, 10.0, 20.0)
    assert block.transform == "log"
    with pytest.raises(AttributeError):
        block.prop = "rho"  # type: ignore[misc]
    assert im.DepthProfile.bspline("vp", "sediment", count=5).degree == 3
    assert im.SourceParameters(sources=(2, 1)).quantities == ("signature",)
    assert im.GridParameters("vp", shape=[4, 4]).props == ("vp",)


def test_control_space_rejects_duplicates_and_non_blocks():
    with pytest.raises(TypeError, match="not a control block"):
        im.ControlSpace(vp=object())
    with pytest.raises(ValueError, match="at least one block"):
        im.ControlSpace()
    block = im.DepthProfile("vp", "sediment", count=4)
    with pytest.raises(ValueError, match="duplicate control key"):
        im.ControlSpace(block, vp=block)


def test_unbound_space_reports_unresolved_layout():
    space = im.ControlSpace(vp=im.DepthProfile("vp", "sediment", spacing=10.0))
    assert not space.resolved
    assert space.keys == ("vp",)
    with pytest.raises(im.UnresolvedControlError, match="bind"):
        space.size
    # Explicit global nodes need no simulation.
    resolved = im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", datum="global", nodes=[0.0, 1.0, 2.0])
    )
    assert resolved.resolved
    assert resolved.blocks == ("model.vp",)
    assert resolved.size == 3


# ---------------------------------------------------------------------------
# extents from the layered model
# ---------------------------------------------------------------------------


def test_profile_extent_is_the_layer_span_below_its_top_surface(simulation):
    bound = im.ControlSpace(vp=im.DepthProfile("vp", "sediment", spacing=100.0)).bind(
        simulation
    )
    control = bound.block("vp").control
    assert isinstance(control, HatControl)
    thickness = SEDIMENT[1] - SEDIMENT[0]
    assert control.axis == "depth"
    assert control.coordinate_system == "seabed_depth"
    np.testing.assert_allclose(control.coordinates[0], 0.0)
    np.testing.assert_allclose(control.coordinates[-1], thickness)
    assert control.size == 14
    system = next(
        s for s in bound.simulation.coordinate_systems if s.name == "seabed_depth"
    )
    assert system.surface_ref == "seabed"
    assert [axis.name for axis in system.axes] == ["depth"]
    assert simulation.coordinate_systems == []


def test_profile_spacing_is_a_maximum_that_ends_on_the_boundaries(simulation):
    bound = im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", datum="global", spacing=400.0)
    ).bind(simulation)
    control = bound.block("vp").control
    np.testing.assert_allclose(control.origin, SEDIMENT[0])
    assert control.spacing <= 400.0
    np.testing.assert_allclose(control.coordinates[-1], SEDIMENT[1])
    assert control.coordinate_system == "global"


def test_profile_count_and_nodes_and_pint_spacing(simulation):
    counted = im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", datum="global", count=5)
    ).bind(simulation)
    np.testing.assert_allclose(
        counted.block("vp").control.coordinates, np.linspace(*SEDIMENT, 5)
    )
    explicit = im.ControlSpace(
        vp=im.DepthProfile(
            "vp", "sediment", datum="global", nodes=[100.0, 600.0, 1100.0]
        )
    ).bind(simulation)
    assert explicit.block("vp").control.origin == 100.0
    assert explicit.block("vp").control.spacing == 500.0
    with pytest.raises(ValueError, match="uniformly spaced"):
        im.ControlSpace(
            vp=im.DepthProfile("vp", "sediment", datum="global", nodes=[0.0, 1.0, 3.0])
        ).bind(simulation)
    # Pint spacing converts to the model's length unit (Sauce default km) and
    # records the unit on the control.
    quantity = im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", datum="global", spacing=0.5 * u.km)
    ).bind(simulation)
    control = quantity.block("vp").control
    assert control.units == "km"
    assert control.spacing <= 0.5


def test_bspline_profile_uses_open_uniform_knots(simulation):
    bound = im.ControlSpace(
        rho=im.DepthProfile.bspline(
            "rho", "sediment", count=6, degree=3, datum="global"
        )
    ).bind(simulation)
    control = bound.block("rho").control
    assert isinstance(control, BSplineControl)
    assert control.degree == 3
    assert control.size == 6
    knots = np.asarray(control.knots)
    np.testing.assert_allclose(knots[:4], SEDIMENT[0])
    np.testing.assert_allclose(knots[-4:], SEDIMENT[1])
    np.testing.assert_allclose(knots[3:-3], np.linspace(*SEDIMENT, 4))
    spaced = im.ControlSpace(
        rho=im.DepthProfile.bspline("rho", "sediment", spacing=650.0, degree=2)
    ).bind(simulation)
    assert spaced.block("rho").control.size == 2 + 2
    with pytest.raises(ValueError, match="degree \\+ 1"):
        im.ControlSpace(
            rho=im.DepthProfile.bspline("rho", "sediment", count=3, degree=3)
        ).bind(simulation)


def test_explicit_surface_coordinate_system_extent(simulation):
    from frequensolve.geometry.frame import Axis, SurfaceCoordinateSystem

    simulation.coordinate_systems.append(
        SurfaceCoordinateSystem(
            "above_bottom",
            "bottom",
            axes=[Axis("height", direction="z", positive="up")],
            normal="up",
        )
    )
    bound = im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", datum="above_bottom", count=3)
    ).bind(simulation)
    block = bound.block("vp")
    control = block.control
    # the system's vertical axis, oriented by its ``positive``
    assert (control.axis, control.coordinate_system) == ("height", "above_bottom")
    np.testing.assert_allclose(control.coordinates, [0.0, 650.0, 1300.0])
    assert block.dims == ("depth",) and not block.downward


def test_depth_profile_datums_resolve_their_frame_and_extent(simulation):
    from frequensolve.geometry.frame import Axis, SurfaceCoordinateSystem

    thickness = SEDIMENT[1] - SEDIMENT[0]
    # a user-authored seabed-relative system, used by name
    simulation.coordinate_systems.append(
        SurfaceCoordinateSystem(
            "below_seabed",
            "seabed",
            axes=[Axis("below", direction="z", positive="down")],
            normal="down",
        )
    )
    bound = im.ControlSpace(
        top=im.DepthProfile("vp", "sediment", count=3),
        glob=im.DepthProfile("rho", "sediment", datum="global", count=3),
        user=im.DepthProfile("vp", "water", datum="below_seabed", count=3),
        surf=im.DepthProfile("rho", "water", datum="bottom", count=3),
    ).bind(simulation)

    top, glob = bound.block("top"), bound.block("glob")
    user, surf = bound.block("user"), bound.block("surf")
    # "top": depth below the subdomain's upper surface (internal system)
    assert top.control.axis == "depth"
    assert top.control.coordinate_system == "seabed_depth"
    assert top.dims == ("depth",) and top.axis_label == "depth below seabed"
    np.testing.assert_allclose(top.control.coordinates, [0.0, 650.0, thickness])
    # "global": the model's vertical coordinate; the sediment top is not z=0
    assert (glob.control.axis, glob.control.coordinate_system) == ("z", "global")
    assert glob.dims == ("z",) and glob.axis_label == "z"
    np.testing.assert_allclose(glob.control.coordinates, [200.0, 850.0, 1500.0])
    # a coordinate system: its vertical axis; the water layer lies above it
    assert (user.control.axis, user.control.coordinate_system) == (
        "below",
        "below_seabed",
    )
    np.testing.assert_allclose(user.control.coordinates, [-200.0, -100.0, 0.0])
    assert user.dims == ("depth",) and user.axis_label == "depth below seabed"
    # a surface name: depth below that surface (need not start at 0)
    assert surf.control.axis == "depth"
    assert surf.control.coordinate_system == "bottom_depth"
    np.testing.assert_allclose(surf.control.coordinates, [-1500.0, -1400.0, -1300.0])
    names = [s.name for s in bound.simulation.coordinate_systems]
    assert names.count("seabed_depth") == 1 and "bottom_depth" in names
    # nodes are measured in the datum frame
    nodes = im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", datum="bottom", nodes=[-1300.0, 0.0])
    ).bind(simulation)
    assert nodes.block("vp").control.coordinate_system == "bottom_depth"
    np.testing.assert_allclose(nodes.block("vp").control.coordinates, [-1300.0, 0.0])
    assert simulation.coordinate_systems[-1].name == "below_seabed"  # untouched


def test_depth_profile_datum_errors(simulation):
    from frequensolve.geometry.frame import Axis, CoordinateSystem

    simulation.coordinate_systems.append(
        CoordinateSystem(
            name="bottom",
            axes=[Axis("depth", direction="z")],
            inherit_axes=False,
        )
    )
    with pytest.raises(
        ValueError, match="both a coordinate system and a model surface"
    ):
        im.ControlSpace(
            vp=im.DepthProfile("vp", "sediment", datum="bottom", count=3)
        ).bind(simulation)
    with pytest.raises(
        ValueError, match="unknown DepthProfile datum 'nowhere'.*seabed"
    ):
        im.ControlSpace(
            vp=im.DepthProfile("vp", "sediment", datum="nowhere", count=3)
        ).bind(simulation)
    simulation.coordinate_systems.append(
        CoordinateSystem(
            name="flat",
            axes=[Axis("x", direction="x")],
            inherit_axes=False,
        )
    )
    with pytest.raises(ValueError, match="0 vertical axes"):
        im.ControlSpace(
            vp=im.DepthProfile("vp", "sediment", datum="flat", count=3)
        ).bind(simulation)
    simulation.coordinate_systems.append(
        CoordinateSystem(
            name="twice",
            axes=[Axis("a", direction="z"), Axis("b", direction="z")],
        )
    )
    with pytest.raises(ValueError, match="2 vertical axes"):
        im.ControlSpace(
            vp=im.DepthProfile("vp", "sediment", datum="twice", count=3)
        ).bind(simulation)
    for bad in ("", "  ", None, 3):
        with pytest.raises(ValueError, match="datum must be a non-empty string"):
            im.DepthProfile("vp", "sediment", datum=bad, count=3)
    with pytest.raises(TypeError):
        im.DepthProfile("vp", "sediment", axis="z", count=3)
    with pytest.raises(TypeError):
        im.DepthProfile("vp", "sediment", coordinate_system="global", count=3)
    # surface datums need a simulation to anchor them; global nodes do not
    with pytest.raises(im.UnresolvedControlError, match="datum 'top'"):
        im.DepthProfile("vp", "sediment", nodes=[0.0, 1.0]).build_control(None)
    assert not im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", nodes=[0.0, 1.0])
    ).resolved


def test_bind_reports_unknown_subdomains_and_properties(simulation):
    with pytest.raises(KeyError, match="no layer 'basement'"):
        im.ControlSpace(vp=im.DepthProfile("vp", "basement", count=3)).bind(simulation)
    with pytest.raises(KeyError, match="no property 'vs'"):
        im.ControlSpace(vs=im.DepthProfile("vs", "sediment", count=3)).bind(simulation)
    with pytest.raises(KeyError, match="no implicit surface"):
        im.ControlSpace(s=im.InterfaceParameters("nothing")).bind(simulation)


def test_grid_covers_the_subdomain_bounding_box(simulation):
    bound = im.ControlSpace(
        grid=im.GridParameters(["vp", "rho"], "water", spacing=[1000.0, 100.0])
    ).bind(simulation)
    assert bound.blocks == ("model.grid_vp", "model.grid_rho")
    control = bound.block("grid.vp").control
    assert isinstance(control, TensorHatControl)
    assert tuple(control.axes) == ("x", "z")
    assert control.shape == (5, 3)
    np.testing.assert_allclose(control.origin, [X_LIMITS[0], WATER[0]])
    np.testing.assert_allclose(control.axis_coordinates[0], np.linspace(*X_LIMITS, 5))
    np.testing.assert_allclose(control.axis_coordinates[1], np.linspace(*WATER, 3))
    whole = im.ControlSpace(
        vp=im.GridParameters("vp", shape=[3, 4], subdomain="sediment")
    ).bind(simulation)
    control = whole.block("vp").control
    assert control.shape == (3, 4)
    np.testing.assert_allclose(control.axis_coordinates[1], np.linspace(*SEDIMENT, 4))
    with pytest.raises(ValueError, match="exactly one subdomain"):
        im.ControlSpace(vp=im.GridParameters("vp", shape=[3, 4])).bind(simulation)


# ---------------------------------------------------------------------------
# ordering, packing, bounds
# ---------------------------------------------------------------------------


def test_qualified_ordering_follows_authoring_and_restrict(simulation):
    bound = _full_space().bind(simulation)
    assert bound.blocks == (
        "model.vp",
        "model.rho",
        "model.grid_vp",
        "model.grid_rho",
        "model.salt_top",
        "source.1.position",
        "source.2.position",
        "source.1.signature",
        "source.2.signature",
        "reflectivity.ip",
    )
    assert bound.qualified_names == bound.blocks
    assert bound.controls_payload() == {"active": list(bound.blocks)}
    stage = bound.restrict(["src.signature", "salt", "model.vp", "refl.ip"])
    assert stage.blocks == (
        "source.1.signature",
        "source.2.signature",
        "model.salt_top",
        "model.vp",
        "reflectivity.ip",
    )
    assert stage.keys == ("vp", "salt", "src", "refl")
    assert stage.restrict("src.2.signature").blocks == ("source.2.signature",)
    assert stage.restrict("source.1.signature").blocks == ("source.1.signature",)
    assert "grid" not in stage
    assert "vp" in stage
    with pytest.raises(KeyError):
        stage.restrict(["grid"])
    with pytest.raises(ValueError, match="selected twice"):
        bound.restrict(["vp", "model.vp"])
    single = im.ControlSpace(im.DepthProfile("vp", "sediment", count=3)).bind(
        simulation
    )
    assert single.keys == ("vp",)
    assert single.blocks == ("model.vp",)


def test_pack_and_unpack_round_trip_with_complex_source_blocks(simulation):
    bound = _full_space().bind(simulation).restrict(["src", "vp"])
    assert bound.block("source.1.signature").complex
    assert bound.sizes["source.1.signature"] == 2
    assert bound.sizes["source.1.position"] == 2
    values = {
        "src.position": {
            "source.1.position": [1000.0, 10.0],
            "source.2.position": [2000.0, 12.0],
        },
        "src.signature": {
            "source.1.signature": [1.0 + 2.0j],
            "source.2.signature": [3.0 - 1.0j],
        },
        "vp": np.arange(bound.sizes["model.vp"], dtype=float),
    }
    vector = bound.pack(values)
    assert isinstance(vector, im.ControlVector)
    assert vector.size == bound.size
    np.testing.assert_array_equal(vector.values[:8], [1000, 10, 2000, 12, 1, 2, 3, -1])
    unpacked = bound.unpack(vector)
    np.testing.assert_array_equal(unpacked["source.1.signature"], [1.0 + 2.0j])
    np.testing.assert_array_equal(unpacked["source.2.position"], [2000.0, 12.0])
    np.testing.assert_array_equal(vector["vp"], values["vp"])
    np.testing.assert_array_equal(vector["source.2.signature"], [3.0 - 1.0j])
    table = vector.per_source()
    assert set(table) == {1, 2}
    np.testing.assert_array_equal(table[2]["signature"], [3.0 - 1.0j])
    np.testing.assert_array_equal(table[1]["position"], [1000.0, 10.0])
    assert bound.pack(bound.unpack(vector)) == vector
    with pytest.raises(ValueError, match="is real"):
        bound.pack({**bound.unpack(vector), "source.1.position": [1j, 0.0]})
    with pytest.raises(ValueError, match="missing block"):
        bound.pack({"vp": values["vp"]})
    with pytest.raises(ValueError, match="expects 1 complex"):
        bound.pack({**bound.unpack(vector), "source.1.signature": [1j, 2j]})


def test_control_vector_arithmetic_and_space_checks(simulation):
    bound = _full_space().bind(simulation)
    a = bound.ones()
    b = bound.random(seed=1)
    assert bound.random(seed=1) == b
    assert isinstance(a + b, im.ControlVector)
    np.testing.assert_allclose((2.0 * a - b / 2.0).values, 2.0 - b.values / 2.0)
    np.testing.assert_allclose((a + np.asarray(b)).values, 1.0 + b.values)
    np.testing.assert_allclose((-a).values, -1.0)
    assert a.dot(a) == pytest.approx(bound.size)
    assert a.norm() == pytest.approx(np.sqrt(bound.size))
    assert np.asarray(a).shape == (bound.size,)
    assert len(a) == bound.size
    other = bound.restrict(["vp"]).zeros()
    with pytest.raises(ValueError, match="different spaces"):
        a + other
    with pytest.raises(ValueError, match="shape"):
        a + np.zeros(3)
    with pytest.raises(ValueError, match="entries"):
        im.ControlVector(np.zeros(3), bound)
    clipped = (10.0 * a).clip()
    lower, upper = bound.bounds
    assert np.all(clipped.values <= upper)
    np.testing.assert_allclose(clipped["vp"], np.log(3500 / 1900))
    assert a.copy() == a
    assert a.copy() is not a


@pytest.mark.parametrize(
    "transform, limits, expected",
    [
        ("identity", (1500.0, 2500.0), (1500.0 - 1900.0, 2500.0 - 1900.0)),
        ("log", (1450.0, 3500.0), (np.log(1450 / 1900), np.log(3500 / 1900))),
        ("inverse", (1450.0, 3500.0), (1 / 3500 - 1 / 1900, 1 / 1450 - 1 / 1900)),
        ("identity", (None, 2500.0 * u.m / u.s), (-np.inf, 600.0)),
    ],
)
def test_bounds_are_expressed_in_optimizer_coordinates(
    simulation, transform, limits, expected
):
    simulation.model.layers["sediment"].set_property("vp", 1900.0 * u.m / u.s)
    bound = im.ControlSpace(
        vp=im.DepthProfile(
            "vp", "sediment", count=4, transform=transform, limits=limits
        ),
        rho=im.DepthProfile("rho", "sediment", count=3),
    ).bind(simulation)
    lower, upper = bound.bounds
    np.testing.assert_allclose(lower[:4], expected[0])
    np.testing.assert_allclose(upper[:4], expected[1])
    assert np.all(np.isneginf(lower[4:])) and np.all(np.isposinf(upper[4:]))


def test_logit_bounds_use_a_dimensionless_reference(simulation):
    bound = im.ControlSpace(
        sp=im.DepthProfile(
            "Sp", "sediment", count=3, transform="logit", limits=(0.25, 0.75)
        )
    ).bind(simulation)
    lower, upper = bound.bounds
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    np.testing.assert_allclose(lower, logit(0.25) - logit(0.5))
    np.testing.assert_allclose(upper, logit(0.75) - logit(0.5))
    with pytest.raises(ValueError, match="positive"):
        im.ControlSpace(
            vp=im.DepthProfile(
                "vp", "sediment", count=3, transform="log", limits=(-1.0, 10.0)
            )
        ).bind(simulation)


def test_pint_limits_convert_to_the_property_units(simulation):
    simulation.model.layers["sediment"].set_property("vp", 1.9 * u.km / u.s)
    bound = im.ControlSpace(
        vp=im.DepthProfile(
            "vp", "sediment", count=2, limits=(1500 * u.m / u.s, 2500 * u.m / u.s)
        )
    ).bind(simulation)
    lower, upper = bound.bounds
    np.testing.assert_allclose(lower, 1.5 - 1.9)
    np.testing.assert_allclose(upper, 2.5 - 1.9)


def _depth_reference(simulation, prop, values_at_depth):
    """Author ``prop`` of the sediment as a vertical profile in global depth."""

    z = np.linspace(SEDIMENT[0], SEDIMENT[1], 27)
    simulation.model.layers["sediment"].set_property(
        prop, xr.DataArray(values_at_depth(z), dims=["z"], coords={"z": z})
    )


def _vp_at(z):
    return 1800.0 + 0.5 * (np.asarray(z) - SEDIMENT[0])  # 1800 .. 2450 m/s


def _logit(p):
    return np.log(p / (1.0 - p))


@pytest.mark.parametrize(
    "prop, transform, limits, expected",
    [
        ("vp", "identity", (1700.0, 3000.0), lambda r: (1700.0 - r, 3000.0 - r)),
        (
            "vp",
            "log",
            (1700.0, 3000.0),
            lambda r: (np.log(1700.0 / r), np.log(3000.0 / r)),
        ),
        (
            "vp",
            "inverse",
            (1700.0, 3000.0),
            lambda r: (1.0 / 3000.0 - 1.0 / r, 1.0 / 1700.0 - 1.0 / r),
        ),
        (
            "Sp",
            "logit",
            (0.2, 0.8),
            lambda r: (_logit(0.2) - _logit(r), _logit(0.8) - _logit(r)),
        ),
        (
            "vp",
            "identity",
            (None, 3000.0),
            lambda r: (np.full_like(r, -np.inf), 3000.0 - r),
        ),
    ],
)
def test_varying_reference_gives_per_node_bounds(
    simulation, prop, transform, limits, expected
):
    if prop == "vp":
        reference = _vp_at
    else:

        def reference(z):
            return 0.3 + 0.4 * (np.asarray(z) - SEDIMENT[0]) / 1300.0

    _depth_reference(simulation, prop, reference)
    bound = im.ControlSpace(
        p=im.DepthProfile(prop, "sediment", count=5, transform=transform, limits=limits)
    ).bind(simulation)

    block = bound.block("p")
    below = np.linspace(0.0, 1300.0, 5)
    np.testing.assert_allclose(block.coords["depth"], below)
    # hat nodes sit on the reference samples: z = seabed + below
    r = reference(SEDIMENT[0] + below)
    lower, upper = expected(r)
    assert block.lower.shape == block.upper.shape == (5,)
    np.testing.assert_allclose(block.lower, lower)
    np.testing.assert_allclose(block.upper, upper)
    np.testing.assert_allclose(bound.bounds[0], lower)
    np.testing.assert_allclose(bound.bounds[1], upper)


def test_per_node_bounds_are_respected_by_clip_and_support_masks(simulation):
    _depth_reference(simulation, "vp", _vp_at)
    bound = im.ControlSpace(
        vp=im.DepthProfile(
            "vp", "sediment", count=5, transform="log", limits=(1700.0, 3000.0)
        ),
        rho=im.DepthProfile("rho", "sediment", count=3),
    ).bind(simulation)
    r = _vp_at(SEDIMENT[0] + np.linspace(0.0, 1300.0, 5))
    lower, upper = np.log(1700.0 / r), np.log(3000.0 / r)

    high = im.ControlVector(np.full(bound.size, 10.0), bound).clip()
    low = im.ControlVector(np.full(bound.size, -10.0), bound).clip()
    np.testing.assert_allclose(high["vp"], upper)
    np.testing.assert_allclose(low["vp"], lower)
    np.testing.assert_allclose(high["rho"], 10.0)  # unbounded block
    # clipped values reproduce the physical limits at every node
    np.testing.assert_allclose(r * np.exp(high["vp"]), 3000.0)
    np.testing.assert_allclose(r * np.exp(low["vp"]), 1700.0)

    mask = np.array([True, False, True, True, False])
    frozen = bound.with_support({"vp": mask})
    f_lower, f_upper = frozen.bounds
    np.testing.assert_allclose(f_lower[:3], lower[mask])
    np.testing.assert_allclose(f_upper[:3], upper[mask])
    clipped = im.ControlVector(np.full(frozen.size, 10.0), frozen).clip()
    np.testing.assert_allclose(clipped.values[:3], upper[mask])


def test_bspline_bounds_are_enforced_at_the_greville_abscissae(simulation):
    _depth_reference(simulation, "vp", _vp_at)
    bound = im.ControlSpace(
        vp=im.DepthProfile.bspline(
            "vp", "sediment", datum="global", count=6, limits=(1700.0, 3000.0)
        )
    ).bind(simulation)

    block = bound.block("vp")
    knots = np.asarray(block.control.knots)
    greville = np.array([knots[i + 1 : i + 4].mean() for i in range(6)])
    np.testing.assert_allclose(block.coords["z"], greville)
    assert greville[0] == SEDIMENT[0] and greville[-1] == SEDIMENT[1]
    np.testing.assert_allclose(block.lower, 1700.0 - _vp_at(greville))
    np.testing.assert_allclose(block.upper, 3000.0 - _vp_at(greville))


def test_lattice_bounds_follow_the_reference_per_lattice_node(simulation):
    x = np.linspace(*X_LIMITS, 5)
    z = np.linspace(*SEDIMENT, 6)
    field = 1800.0 + 0.1 * x[:, None] + 0.5 * (z[None, :] - SEDIMENT[0])
    simulation.model.layers["sediment"].set_property(
        "vp", xr.DataArray(field, dims=["x", "z"], coords={"x": x, "z": z})
    )
    bound = im.ControlSpace(
        vp=im.GridParameters("vp", "sediment", shape=[3, 4], limits=(1700.0, 3500.0))
    ).bind(simulation)

    block = bound.block("vp")
    nodes = block.control.coordinates  # (size, 2), first axis fastest
    r = 1800.0 + 0.1 * nodes[:, 0] + 0.5 * (nodes[:, 1] - SEDIMENT[0])
    np.testing.assert_allclose(block.lower, 1700.0 - r)
    np.testing.assert_allclose(block.upper, 3500.0 - r)


def test_laterally_varying_reference_bounds_hold_along_the_whole_iso_line(
    simulation,
):
    x = np.linspace(*X_LIMITS, 9)
    z = np.linspace(*SEDIMENT, 3)
    field = np.repeat((1800.0 + 0.1 * x)[:, None], z.size, axis=1)  # 1800 .. 2200
    simulation.model.layers["sediment"].set_property(
        "vp", xr.DataArray(field, dims=["x", "z"], coords={"x": x, "z": z})
    )
    bound = im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", count=4, limits=(1700.0, 3000.0))
    ).bind(simulation)
    block = bound.block("vp")
    np.testing.assert_allclose(block.lower, 1700.0 - 1800.0)
    np.testing.assert_allclose(block.upper, 3000.0 - 2200.0)

    with pytest.raises(ValueError, match="infeasible"):
        im.ControlSpace(
            vp=im.DepthProfile("vp", "sediment", count=4, limits=(1900.0, 2000.0))
        ).bind(simulation)


def test_saved_file_backed_reference_is_read_from_the_project(simulation):
    _depth_reference(simulation, "vp", _vp_at)
    simulation.save()
    working = simulation.copy("shelf_copy")
    sediment = next(s for s in working.model.subdomains if s.name == "sediment")
    assert sediment.properties["vp"].data is None  # a project file reference

    bound = im.ControlSpace(
        vp=im.DepthProfile(
            "vp", "sediment", count=5, transform="log", limits=(1700.0, 3000.0)
        )
    ).bind(working)

    r = _vp_at(SEDIMENT[0] + np.linspace(0.0, 1300.0, 5))
    np.testing.assert_allclose(bound.block("vp").lower, np.log(1700.0 / r))
    np.testing.assert_allclose(bound.block("vp").upper, np.log(3000.0 / r))


def test_limits_on_an_unreadable_reference_name_the_block(simulation, tmp_path):
    from frequensolve.model.property import Property

    simulation.model.layers["sediment"].set_property(
        "vp", Property(str(tmp_path / "missing.rsf"), read=False)
    )
    with pytest.raises(ValueError, match=r"'model\.vp'.*cannot be evaluated"):
        im.ControlSpace(
            vp=im.DepthProfile("vp", "sediment", count=3, limits=(1500.0, 3000.0))
        ).bind(simulation)
    # without limits the file reference binds as before
    bound = im.ControlSpace(vp=im.DepthProfile("vp", "sediment", count=3)).bind(
        simulation
    )
    assert np.isneginf(bound.block("vp").lower)


# ---------------------------------------------------------------------------
# support masks
# ---------------------------------------------------------------------------


def test_support_masks_freeze_dofs_and_expand_sauce_vectors(simulation):
    bound = _full_space().bind(simulation).restrict(["vp", "src.signature"])
    assert bound.support.all()
    assert bound.size == bound.full_size
    mask = np.ones(bound.sizes["model.vp"], dtype=bool)
    mask[[0, 1, -1]] = False
    frozen = bound.with_support({"vp": mask}, min_support=0.05)
    assert frozen.min_support == 0.05
    assert frozen.support.frozen_count == 3
    assert frozen.size == bound.size - 3
    assert frozen.full_size == bound.full_size
    np.testing.assert_array_equal(frozen.support["vp"], mask)
    assert frozen.slices["model.vp"] == slice(0, mask.sum())
    assert frozen.full_slices["model.vp"] == slice(0, mask.size)
    assert frozen.controls_payload()["min_support"] == 0.05

    vector = frozen.ones()
    full = frozen.to_sauce_vector(vector)
    assert full.size == frozen.full_size
    np.testing.assert_array_equal(full[:2], 0.0)
    assert full[mask.size - 1] == 0.0
    assert np.count_nonzero(full) == frozen.size
    back = frozen.from_sauce_vector(full)
    assert back == vector
    # Frozen DOFs render as NaN and pack drops them.
    rendered = vector.to_xarray("vp")
    assert np.isnan(rendered.values[[0, 1, -1]]).all()
    assert np.nanmin(rendered.values) == 1.0
    rendered = vector.to_xarray("vp", frozen=0.0)
    assert rendered.values[0] == 0.0
    assert not frozen.equivalent(bound)
    assert frozen.without_support().equivalent(bound)

    with pytest.raises(ValueError, match="needs"):
        bound.with_support({"vp": mask[:-1]})
    with pytest.raises(ValueError, match="whole coefficients"):
        bound.with_support({"source.1.signature": [True, False]})
    with pytest.raises(ValueError, match="entries"):
        frozen.from_sauce_vector(np.zeros(frozen.size))


def test_support_masks_can_come_from_sauce_state_files(simulation):
    bound = im.ControlSpace(vp=im.DepthProfile("vp", "sediment", count=5)).bind(
        simulation
    )
    mask = np.array([False, True, True, True, False])
    state_file = ControlStateFile(
        {"model.vp": np.arange(5.0), "model.other": [1.0]},
        support={"model.vp": mask},
        support_min_support=0.01,
    )
    frozen = bound.with_support(state_file)
    np.testing.assert_array_equal(frozen.support["vp"], mask)
    assert frozen.size == 3


def test_geometric_support_freezes_nodes_outside_the_layer(simulation):
    bound = im.ControlSpace(
        vp=im.DepthProfile(
            "vp", "sediment", datum="global", nodes=np.arange(0.0, 2001.0, 250.0)
        ),
        rho=im.DepthProfile.bspline(
            "rho",
            "sediment",
            datum="global",
            nodes=[0.0, 500.0, 1000.0, 1500.0, 2000.0],
        ),
        inside=im.DepthProfile("vp", "water", count=3),
    ).bind(simulation)
    masks = bound.geometric_support()
    # Hat nodes at z >= 1750 have no support inside [200, 1500].
    np.testing.assert_array_equal(
        masks["model.vp"], [True, True, True, True, True, True, True, False, False]
    )
    # The last cubic B-spline coefficient is supported on (1500, 2000] only.
    np.testing.assert_array_equal(
        masks["model.rho"], [True, True, True, True, True, True, False]
    )
    assert "model.inside" not in masks
    frozen = bound.with_geometric_support()
    assert frozen.size == bound.size - 3
    assert isinstance(frozen, im.BoundControlSpace)
    below = im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", nodes=np.arange(0.0, 2001.0, 500.0))
    ).bind(simulation)
    np.testing.assert_array_equal(
        below.geometric_support()["model.vp"], [True, True, True, True, False]
    )


# ---------------------------------------------------------------------------
# binding installs controls into a copy
# ---------------------------------------------------------------------------


def test_bind_installs_controls_into_a_copy_that_validates(simulation):
    original = copy.deepcopy(simulation.model.to_fs())
    bound = _full_space().bind(simulation)
    assert simulation.model.to_fs() == original
    assert bound.simulation is not simulation
    assert bound.simulation.model is not simulation.model

    sediment = bound.simulation.model.layers["sediment"]
    vp = sediment.properties["vp"]
    assert isinstance(vp, ParameterizedProperty)
    assert vp.id == "vp"
    assert vp.transform == "log"
    assert vp.reference.to_fs() == {"value": 1900.0}
    np.testing.assert_array_equal(vp.coefficients, 0.0)
    assert isinstance(sediment.properties["rho"], ParameterizedProperty)
    water = bound.simulation.model.layers["water"]
    assert water.properties["vp"].id == "grid_vp"
    assert water.properties["rho"].id == "grid_rho"
    assert isinstance(water.properties["vp"].control, TensorHatControl)
    salt = bound.simulation.model.implicit_surfaces["salt_top"]
    assert salt.control.id == "salt_top"
    assert salt.control.maximum_displacement == 150.0
    assert simulation.model.implicit_surfaces["salt_top"].control is None

    payload = {"schema": "fs-material-model-1", **bound.simulation.model.to_fs()}
    _material_validator().validate(payload)
    hat = payload["subdomains"][1]["properties"]["vp"]["parameterized"]["control"]
    assert hat["kind"] == "hat"
    assert hat["coordinate_system"] == "seabed_depth"
    assert hat["axis"] == "depth"
    systems = [s.to_fs() for s in bound.simulation.coordinate_systems]
    assert systems[0]["_type"] == "SurfaceCoordinateSystem"
    assert systems[0]["surface"] == "seabed"

    assert bound.reflectivity_payload() == {
        "parameterization": "vp_ip",
        "fields": [{"name": "ip", "layer": 2, "axis": 2, "basis": "vp"}],
    }
    assert bound.source_controls_payload() == {"location_method": "analytic"}
    assert bound.restrict(["vp"]).reflectivity_payload() is None
    assert bound.restrict(["vp"]).source_controls_payload() is None
    assert bound.mesh_property_spaces == {}


def test_bind_rejects_conflicting_block_ids(simulation):
    with pytest.raises(ValueError, match="already used"):
        im.ControlSpace(
            a=im.DepthProfile("vp", "sediment", count=3, id="shared"),
            b=im.DepthProfile("rho", "sediment", count=3, id="shared"),
        ).bind(simulation)
    with pytest.raises(ValueError, match="already parameterized"):
        im.ControlSpace(
            a=im.DepthProfile("vp", "sediment", count=3),
            b=im.DepthProfile("vp", "sediment", count=4, id="again"),
        ).bind(simulation)


def test_rebinding_an_already_parameterized_property_replaces_its_control(simulation):
    first = im.ControlSpace(vp=im.DepthProfile("vp", "sediment", count=3)).bind(
        simulation
    )
    second = im.ControlSpace(vp=im.DepthProfile("vp", "sediment", count=5)).bind(
        first.simulation
    )
    vp = second.simulation.model.layers["sediment"].properties["vp"]
    assert vp.control.size == 5
    assert vp.reference.to_fs() == {"value": 1900.0}


def test_mesh_parameters_register_property_spaces_and_wait_for_the_manifest(
    simulation,
):
    space = im.ControlSpace(
        mesh=im.MeshParameters("vp", "sediment", frequency=8.0 * u.Hz, epw=2.0),
        rho=im.DepthProfile("rho", "sediment", count=3),
    )
    bound = space.bind(simulation)
    assert bound.blocks == ("model.mesh", "model.rho")
    assert not bound.resolved
    with pytest.raises(im.UnresolvedControlError, match="manifest"):
        bound.size
    spaces = bound.simulation.model.property_spaces
    assert set(spaces) == {"mesh"}
    assert spaces["mesh"].to_fs() == {
        "artifact": "mesh.h5",
        "frequency": 8.0,
        "epw": 2.0,
    }
    assert isinstance(
        bound.simulation.model.layers["sediment"].properties["vp"].control, MeshControl
    )
    assert "property_spaces" not in simulation.model.to_fs()
    payload = {"schema": "fs-material-model-1", **bound.simulation.model.to_fs()}
    _material_validator().validate(payload)

    manifest = ControlRegistryManifest.from_dict(
        {
            "schema": "fs-control-registry-1",
            "fingerprint": "sha256:" + "0" * 64,
            "blocks": [
                _registry_block(1, "model.mesh", 7),
                _registry_block(2, "model.rho", 3),
            ],
            "active_blocks": [1, 2],
            "active_offsets": [1, 8],
        }
    )
    sized = bound.with_manifest(manifest)
    assert sized.resolved
    assert sized.size == 10
    assert isinstance(sized, im.BoundControlSpace)
    vector = sized.ones()
    with pytest.raises(NotImplementedError, match="to_mesh"):
        vector.to_xarray("mesh")
    with pytest.raises(ValueError, match="MeshManager configuration"):
        vector.to_mesh()
    wrong = ControlRegistryManifest.from_dict(
        {
            "schema": "fs-control-registry-1",
            "fingerprint": "sha256:" + "0" * 64,
            "blocks": [
                _registry_block(1, "model.mesh", 7),
                _registry_block(2, "model.rho", 4),
            ],
        }
    )
    with pytest.raises(ValueError, match="3 DOFs locally"):
        bound.with_manifest(wrong)


def _registry_block(block_id, name, size, *, offset=1, components=2, complex_=False):
    return {
        "id": block_id,
        "name": name,
        "binding": [1, 1, block_id],
        "layout": [offset, size, components, 2 if complex_ else 1],
        "units": "",
        "actions": 3,
        "transform": 0,
        "scaling": [0.0, 1.0, -1.7976931348623157e308, 1.7976931348623157e308],
        "basis_identity": "",
        "distributed": False,
    }


def test_space_from_manifest_uses_registry_layout_and_values():
    manifest = ControlRegistryManifest.from_dict(
        {
            "schema": "fs-control-registry-1",
            "fingerprint": "sha256:" + "1" * 64,
            "blocks": [
                {
                    **_registry_block(1, "source.1.position", 2),
                    "components": ["x", "z"],
                },
                _registry_block(2, "model.vp", 3, offset=3),
                _registry_block(
                    3, "source.1.signature", 2, offset=6, components=1, complex_=True
                ),
            ],
            "active_blocks": [2],
            "active_offsets": [0, 1, 0],
            "coordinates": [513.7, 34.1, 0.0, 0.0, 0.0, 1.0, 0.0],
            "values": [513.7, 34.1, 0.0, 0.0, 0.0, 1.0, 0.0],
        }
    )
    space = im.ControlSpace.from_manifest(manifest)
    assert space.blocks == ("source.1.position", "model.vp", "source.1.signature")
    assert space.size == 7
    assert space.block("source.1.signature").complex
    assert space.block("source.1.position").components == ("x", "z")
    state = im.ControlState.from_manifest(manifest)
    np.testing.assert_array_equal(state["source.1.position"], [513.7, 34.1])
    np.testing.assert_array_equal(state["source.1.signature"], [1.0 + 0.0j])
    active = space.restrict(list(manifest.active_names))
    assert active.blocks == ("model.vp",)
    assert state.vector(active).size == 3


# ---------------------------------------------------------------------------
# state and vector files
# ---------------------------------------------------------------------------


def test_control_state_from_simulation_carries_authored_baselines(simulation):
    bound = _full_space().bind(simulation)
    state = im.ControlState.from_simulation(bound)
    assert state.size == bound.full_size
    np.testing.assert_array_equal(state["vp"], 0.0)
    np.testing.assert_array_equal(state["salt"], [-100.0, -200.0, -100.0])
    # position placeholders are in Sauce's frame (metres); the fixture's
    # source points declare no units, i.e. Sauce's default km
    positions = state["src.position"]
    np.testing.assert_array_equal(positions["source.1.position"], [1.0e6, 1.0e4])
    np.testing.assert_array_equal(positions["source.2.position"], [2.0e6, 1.0e4])
    np.testing.assert_array_equal(state["source.1.signature"], [1.0 + 0.0j])
    np.testing.assert_array_equal(state["refl.ip"], 0.0)
    with pytest.raises(ValueError, match="finite"):
        im.ControlState(bound, np.full(bound.full_size, np.nan))


def test_control_state_round_trips_through_state_files(simulation, tmp_path):
    bound = _full_space().bind(simulation)
    mask = np.ones(bound.sizes["model.vp"], dtype=bool)
    mask[:2] = False
    frozen = bound.with_support({"vp": mask}, min_support=0.02)
    state = im.ControlState.from_blocks(
        frozen,
        {
            name: (
                np.arange(1, size // 2 + 1) * (1.0 + 1.0j)
                if frozen.block(name).complex
                else np.arange(size, dtype=float)
            )
            for name, size in frozen.sizes.items()
        },
    )
    artifact = state.to_file()
    assert isinstance(artifact, ControlStateFile)
    assert artifact.names == frozen.blocks
    np.testing.assert_array_equal(artifact.support["model.vp"], mask)
    assert artifact.support_min_support == 0.02
    np.testing.assert_array_equal(artifact["source.2.signature"], [1.0, 1.0])

    path = state.save(tmp_path / "state.h5")
    loaded = im.ControlState.load(path, bound)
    assert loaded == state
    # The file's masks are adopted when the target space freezes nothing.
    np.testing.assert_array_equal(loaded.space.support["vp"], mask)
    assert loaded.space.min_support == 0.02
    assert ControlStateFile.read(path).blocks.keys() == set(frozen.blocks)

    stage = frozen.restrict(["src.signature", "vp"])
    vector = state.vector(stage)
    assert vector.size == stage.size
    np.testing.assert_array_equal(vector["source.1.signature"], [1.0 + 1.0j])
    updated = state.with_update(vector + 10.0)
    np.testing.assert_array_equal(updated["vp"][:2], state["vp"][:2])  # frozen kept
    np.testing.assert_array_equal(updated["vp"][2:], state["vp"][2:] + 10.0)
    np.testing.assert_array_equal(updated["salt"], state["salt"])
    np.testing.assert_array_equal(updated["source.1.signature"], [11.0 + 11.0j])
    two_arg = state.with_update(stage, np.asarray(vector) + 10.0)
    assert two_arg == updated
    with pytest.raises(ValueError, match="no block"):
        state.vector(
            im.ControlSpace(vs=im.DepthProfile("rho", "water", count=3)).bind(
                simulation
            )
        )


def test_control_vector_round_trips_through_vector_files(simulation, tmp_path):
    bound = _full_space().bind(simulation)
    stage = bound.restrict(["src.signature", "vp"])
    mask = np.ones(stage.sizes["model.vp"], dtype=bool)
    mask[-1] = False
    stage = stage.with_support({"vp": mask})
    vector = stage.random(seed=7)
    artifact = vector.to_file(
        state_fingerprint="sha256:state", registry_fingerprint="sha256:registry"
    )
    assert isinstance(artifact, ControlVectorFile)
    assert artifact.names == stage.blocks
    assert artifact["model.vp"][-1] == 0.0
    np.testing.assert_array_equal(artifact.support["model.vp"], mask)
    with pytest.raises(ValueError, match="require"):
        vector.save(tmp_path / "unbound.h5")
    path = vector.save(
        tmp_path / "vector.h5",
        state_fingerprint="sha256:state",
        registry_fingerprint="sha256:registry",
    )
    read = ControlVectorFile.read(path)
    assert read.state_fingerprint == "sha256:state"
    loaded = im.ControlVector.load(path, stage)
    assert loaded == vector
    # Reading onto the unfrozen restriction adopts the file's support masks.
    adopted = im.ControlVector.from_file(read, bound.restrict(["src.signature", "vp"]))
    np.testing.assert_array_equal(adopted.space.support["vp"], mask)
    assert adopted == vector
    # The native control-sensitivity layout has no spelling for source blocks.
    with pytest.raises(ValueError, match="unqualified"):
        vector.to_file(native=True)
    native = bound.restrict(["vp", "rho"]).zeros().to_file(native=True)
    assert native.names == ("vp", "rho")
    with pytest.raises(ValueError, match="no block"):
        im.ControlVector.from_file(read, bound.restrict(["rho"]))


# ---------------------------------------------------------------------------
# rendering and transfer
# ---------------------------------------------------------------------------


def test_to_xarray_exposes_block_coordinates(simulation):
    bound = _full_space().bind(simulation)
    vector = bound.random(seed=3)
    profile = vector.to_xarray("vp")
    assert isinstance(profile, xr.DataArray)
    assert profile.dims == ("depth",)
    np.testing.assert_allclose(profile.coords["depth"], np.arange(14) * 100.0)
    np.testing.assert_array_equal(profile.values, vector["vp"])
    assert profile.attrs["coordinate_system"] == "seabed_depth"
    assert profile.attrs["transform"] == "log"

    spline = vector.to_xarray("rho")
    control = bound.block("rho").control
    np.testing.assert_allclose(spline.coords["depth"], control.coordinates)

    lattice = vector.to_xarray("grid.vp")
    assert lattice.dims == ("x", "z")
    assert lattice.shape == (5, 3)
    np.testing.assert_allclose(lattice.coords["x"], np.linspace(*X_LIMITS, 5))
    np.testing.assert_allclose(lattice.coords["z"], np.linspace(*WATER, 3))
    control = bound.block("grid.vp").control
    np.testing.assert_array_equal(
        lattice.values, vector["grid.vp"].reshape(control.shape, order="F")
    )
    np.testing.assert_array_equal(
        lattice.values, control.with_coefficients(vector["grid.vp"]).grid
    )

    salt = vector.to_xarray("salt")
    assert salt.dims == ("center",)
    np.testing.assert_allclose(salt.coords["x"], [1000.0, 2000.0, 3000.0])

    dataset = vector.to_xarray()
    assert set(dataset.data_vars) == {
        "vp",
        "rho",
        "grid.vp",
        "grid.rho",
        "salt",
        "refl.ip",
    }
    with pytest.raises(ValueError, match="per_source"):
        vector.to_xarray("src.1.signature")
    state = im.ControlState.from_simulation(bound)
    assert set(state.to_xarray().data_vars) == set(dataset.data_vars)


def test_transfer_to_between_profile_resolutions_preserves_affine_fields(simulation):
    fine = im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", datum="global", count=27),
        rho=im.DepthProfile.bspline("rho", "sediment", datum="global", count=8),
    ).bind(simulation)
    coarse = im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", datum="global", count=14),
        rho=im.DepthProfile.bspline("rho", "sediment", datum="global", count=5),
    ).bind(simulation)
    z_fine = fine.block("vp").coords["z"]
    rho_fine = fine.block("rho").coords["z"]  # Greville abscissae
    vector = fine.pack({"vp": 2.0 + 0.001 * z_fine, "rho": 1.0 - 0.0005 * rho_fine})
    transferred = fine.transfer_to(coarse, vector)
    assert transferred.space is coarse
    np.testing.assert_allclose(
        transferred["vp"], 2.0 + 0.001 * coarse.block("vp").coords["z"], atol=1e-8
    )
    np.testing.assert_allclose(
        transferred["rho"], 1.0 - 0.0005 * coarse.block("rho").coords["z"], atol=1e-8
    )
    # Back-transfer reproduces the fine coefficients for affine fields too.
    back = coarse.transfer_to(fine, transferred)
    np.testing.assert_allclose(back.values, vector.values, atol=1e-8)


def test_transfer_to_handles_lattices_and_copies_identical_blocks(simulation):
    coarse = im.ControlSpace(
        g=im.GridParameters("vp", "water", shape=[5, 3]),
        s=im.InterfaceParameters("salt_top"),
    ).bind(simulation)
    fine = im.ControlSpace(
        g=im.GridParameters("vp", "water", shape=[9, 5]),
        s=im.InterfaceParameters("salt_top"),
    ).bind(simulation)
    nodes = coarse.block("g").control.coordinates
    vector = coarse.pack(
        {"g": 1.0 + 0.001 * nodes[:, 0] + 0.01 * nodes[:, 1], "s": [1.0, 2.0, 3.0]}
    )
    transferred = coarse.transfer_to(fine, vector)
    target = fine.block("g").control.coordinates
    np.testing.assert_allclose(
        transferred["g"], 1.0 + 0.001 * target[:, 0] + 0.01 * target[:, 1], atol=1e-8
    )
    np.testing.assert_array_equal(transferred["s"], [1.0, 2.0, 3.0])
    other = im.ControlSpace(
        g=im.DepthProfile("vp", "water", datum="global", count=3)
    ).bind(simulation)
    with pytest.raises(ValueError, match="cannot transfer"):
        coarse.transfer_to(other, vector)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


def test_source_parameters_resolve_ids_and_component_counts(tmp_path):
    simulation = _layered_simulation(tmp_path, sources=3, kind="vector")
    bound = im.ControlSpace(
        src=im.SourceParameters(
            sources=[3, 1], position=True, mechanism=True, signature_df=True
        )
    ).bind(simulation)
    assert bound.blocks == (
        "source.3.position",
        "source.1.position",
        "source.3.mechanism",
        "source.1.mechanism",
        "source.3.signature",
        "source.1.signature",
        "source.3.signature_df",
        "source.1.signature_df",
    )
    assert bound.sizes["source.3.position"] == 2
    assert bound.sizes["source.3.mechanism"] == 4  # two complex components
    assert bound.sizes["source.1.signature_df"] == 2
    assert bound.block("source.3.mechanism").components == ("c1", "c2")
    table = bound.ones().per_source()
    assert set(table) == {1, 3}
    assert set(table[3]) == {"position", "mechanism", "signature", "signature_df"}
    with pytest.raises(ValueError, match="no source id"):
        im.ControlSpace(src=im.SourceParameters(sources=[4])).bind(simulation)
    with pytest.raises(ValueError, match="no source blocks"):
        im.ControlSpace(vp=im.DepthProfile("vp", "water", count=2)).bind(
            simulation
        ).zeros().per_source()


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


@pytest.fixture
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    yield pyplot
    pyplot.close("all")


def test_plot_renders_profile_lattice_interface_and_source_blocks(simulation, plt):
    bound = _full_space().bind(simulation)
    mask = np.ones(bound.sizes["model.vp"], dtype=bool)
    mask[3] = False
    bound = bound.with_support({"vp": mask})
    vector = bound.random(3)

    ax = vector.plot("vp")
    assert isinstance(ax, plt.Axes)
    (line,) = ax.lines
    assert ax.yaxis_inverted()  # "depth" runs downwards
    ydata = np.asarray(line.get_ydata(), dtype=float)
    xdata = np.asarray(line.get_xdata(), dtype=float)
    np.testing.assert_allclose(ydata, bound.block("vp").control.coordinates)
    assert np.isnan(xdata[3]) and np.count_nonzero(np.isnan(xdata)) == 1
    np.testing.assert_allclose(np.delete(xdata, 3), vector["vp"][mask])
    assert ax.get_ylabel().startswith("depth")

    ax = vector.plot("grid.vp", cmap="viridis")
    (mesh,) = ax.collections
    assert mesh.get_array().size == bound.block("grid.vp").size
    assert mesh.get_cmap().name == "viridis"
    assert ax.yaxis_inverted() and len(ax.figure.axes) == 2  # colorbar attached
    limit = float(np.max(np.abs(vector["grid.vp"])))
    assert mesh.get_clim() == pytest.approx((-limit, limit))

    ax = vector.plot("salt", color="k")
    (line,) = ax.lines
    np.testing.assert_allclose(line.get_ydata(), vector["salt"])
    np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2])

    ax = vector.plot("src.signature")
    assert len(ax.patches) == 4  # two sources x (Re, Im)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2"]
    heights = [p.get_height() for p in ax.patches]
    table = vector.per_source()
    np.testing.assert_allclose(
        heights,
        [
            table[1]["signature"][0].real,
            table[2]["signature"][0].real,
            table[1]["signature"][0].imag,
            table[2]["signature"][0].imag,
        ],
    )
    ax = vector.plot("src.position")
    assert len(ax.patches) == 4 and ax.get_legend() is not None

    ax = vector.plot("refl.ip")
    assert len(ax.lines) == 1

    axes = vector.plot()
    assert isinstance(axes, list) and len(axes) == 8  # blocks + 2 source groups
    figure, own = plt.subplots()
    assert vector.plot("rho", ax=own) is own
    with pytest.raises(ValueError, match="ax takes one"):
        vector.plot("src", ax=own)
    with pytest.raises(KeyError):
        vector.plot("nope")


def test_plot_slices_three_dimensional_lattices(plt):
    from frequensolve.geometry.grids import CartesianGrid

    grid = CartesianGrid(n=[3, 2, 4], x0=[0.0, 0.0, 0.0], x1=[1.0, 1.0, 2.0])
    space = im.ControlSpace(g=im.GridParameters("vp", grid=grid))
    vector = space.random(1)
    ax = vector.plot("g")
    (mesh,) = ax.collections
    assert mesh.get_array().size == 3 * 4  # middle y plane, (x, z)
    assert ax.yaxis_inverted()
    lattice = vector["g"].reshape((3, 2, 4), order="F")
    np.testing.assert_allclose(mesh.get_array().reshape(4, 3), lattice[:, 1, :].T)
    ax = vector.plot("g", slice={"x": 0})
    (mesh,) = ax.collections
    assert mesh.get_array().size == 2 * 4
    assert ax.get_xlabel() == "y"
    with pytest.raises(ValueError, match="lattice axes"):
        vector.plot("g", slice={"t": 0})
    with pytest.raises(ValueError, match="3-D lattices"):
        im.ControlSpace(
            g=im.GridParameters(
                "vp", grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0])
            )
        ).zeros().plot("g", slice={"x": 0})


def test_control_state_plot_draws_the_baseline(simulation, plt):
    bound = _full_space().bind(simulation)
    state = im.ControlState.from_simulation(bound)
    ax = state.plot("salt")
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [-100.0, -200.0, -100.0])


def _mesh_space(simulation):
    bound = im.ControlSpace(
        mesh=im.MeshParameters("vp", "sediment", frequency=8.0 * u.Hz, epw=2.0),
        rho=im.DepthProfile("rho", "sediment", count=3),
    ).bind(simulation)
    manifest = ControlRegistryManifest.from_dict(
        {
            "schema": "fs-control-registry-1",
            "fingerprint": "sha256:" + "0" * 64,
            "blocks": [
                _registry_block(1, "model.mesh", 7),
                _registry_block(2, "model.rho", 3, offset=8),
            ],
            "active_blocks": [1, 2],
            "active_offsets": [1, 8],
        }
    )
    mask = np.ones(7, dtype=bool)
    mask[5] = False
    return bound.with_manifest(manifest).with_support({"mesh": mask})


def test_to_mesh_attaches_point_arrays_or_explains_the_missing_geometry(simulation):
    pv = pytest.importorskip("pyvista")
    space = _mesh_space(simulation)
    vector = space.random(2)
    points = pv.PolyData(np.random.default_rng(0).random((7, 3)))

    grid = vector.to_mesh(points)
    assert grid is not points and grid.n_points == 7
    values = np.asarray(grid.point_data["vp"])
    assert np.isnan(values[5])
    np.testing.assert_allclose(np.delete(values, 5), vector["mesh"][[0, 1, 2, 3, 4, 6]])
    assert "rho" not in grid.point_data
    with pytest.raises(ValueError, match="MeshManager configuration"):
        vector.to_mesh()
    with pytest.raises(ValueError, match="7 coefficients but the mesh has 4"):
        vector.to_mesh(pv.PolyData(np.zeros((4, 3))))
    with pytest.raises(ValueError, match="does not address a mesh block"):
        vector.to_mesh(points, key="rho")
    with pytest.raises(ValueError, match="no mesh blocks"):
        _full_space().bind(simulation).zeros().to_mesh(points)
    with pytest.raises(NotImplementedError, match="to_mesh"):
        vector.to_xarray("mesh")
    with pytest.raises(ValueError, match="one at a time"):
        vector.plot("mesh", ax=object())


def test_plot_of_mesh_blocks_uses_a_pyvista_plotter(simulation):
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    vector = _mesh_space(simulation).random(2)
    points = pv.PolyData(np.random.default_rng(0).random((7, 3)))
    plotter = vector.plot("mesh", mesh=points, show=False, notebook=False)
    try:
        assert isinstance(plotter, pv.Plotter)
        assert len(plotter.renderer.actors) >= 1
    finally:
        plotter.close()


def test_equivalent_rejects_different_bases_and_transforms(simulation):
    hat = im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", datum="global", count=5)
    ).bind(simulation)
    for spec in (
        im.DepthProfile.bspline("vp", "sediment", datum="global", count=5),
        im.DepthProfile("vp", "sediment", datum="top", count=5),
        im.DepthProfile("vp", "sediment", datum="global", count=5, transform="log"),
    ):
        other = im.ControlSpace(vp=spec).bind(simulation)
        assert not hat.equivalent(other)
        with pytest.raises(ValueError):
            _ = hat.ones() + other.ones()
        with pytest.raises(ValueError):
            hat.to_sauce_vector(other.ones())
        state = im.ControlState(hat, np.ones(hat.full_size))
        with pytest.raises(ValueError, match="differs in basis"):
            state.vector(other)
        with pytest.raises(ValueError, match="differs in basis"):
            state.with_update(other.ones())
    # Coefficients and bounds can change without changing the basis.
    altered = copy.deepcopy(hat.simulation)
    for layer in altered.model.subdomains:
        prop = layer.properties.get("vp")
        if isinstance(prop, ParameterizedProperty):
            layer.properties["vp"] = prop.with_coefficients(np.arange(5.0))
    bounded = im.ControlSpace(
        vp=im.DepthProfile(
            "vp", "sediment", datum="global", count=5, limits=(1500, 4000)
        )
    ).bind(altered)
    assert hat.equivalent(bounded)


def test_transfer_matches_each_physical_source_before_family_address(simulation):
    source = im.ControlSpace(
        src=im.SourceParameters(sources=[1, 2], position=True, signature=True)
    ).bind(simulation)
    vector = source.from_sauce_vector(np.arange(source.size, dtype=float) + 1)
    np.testing.assert_array_equal(
        source.transfer_to(source, vector).values, vector.values
    )
    reordered = im.ControlSpace(
        src=im.SourceParameters(sources=[2, 1], position=True, signature=True)
    ).bind(simulation)
    moved = source.transfer_to(reordered, vector)
    for name in source.blocks:
        np.testing.assert_array_equal(moved[name], vector[name])
    only_second = reordered.restrict(["source.2.signature"])
    np.testing.assert_array_equal(
        source.transfer_to(only_second, vector)["source.2.signature"],
        vector["source.2.signature"],
    )
    with pytest.raises(ValueError, match="no block matching"):
        only_second.transfer_to(source, only_second.ones())


@pytest.mark.parametrize("spline", [False, True])
def test_profile_to_grid_evaluates_basis_and_masks_other_layers(simulation, spline):
    from frequensolve.geometry.grids import CartesianGrid
    from frequensolve.model.representation import (
        ControlRepresentation,
        EvaluationContext,
    )

    spec = im.DepthProfile.bspline if spline else im.DepthProfile
    space = im.ControlSpace(vp=spec("vp", "sediment", count=5)).bind(simulation)
    vector = space.pack({"vp": [0.0, 0.3, -0.2, 0.4, 0.1]})
    grid = CartesianGrid(n=[7, 21], x0=[0, 0], x1=[4000, 1500], dims=["x", "z"])
    image = vector.to_grid(grid, "vp")
    assert image.dims == ("z", "x")
    assert np.isnan(image.sel(z=0)).all()
    depths = image.z.values - 200
    expected = ControlRepresentation(space.block("vp").control).evaluate(
        vector["vp"],
        EvaluationContext({"depth": depths}, coordinate_system="seabed_depth"),
    )
    valid = depths >= 0
    np.testing.assert_allclose(
        image.values[valid], np.repeat(expected[valid, None], 7, axis=1)
    )
    state = im.ControlState(space, vector.values)
    np.testing.assert_allclose(state.to_grid(grid, "vp"), image, equal_nan=True)


def test_tensor_to_grid_interpolates_between_control_nodes(simulation, plt):
    from frequensolve.geometry.grids import CartesianGrid

    space = im.ControlSpace(g=im.GridParameters("vp", "water", shape=[3, 3])).bind(
        simulation
    )
    # Only the centre coefficient is nonzero: a separable 2D tent.
    coefficients = np.zeros(9)
    coefficients[4] = 1
    vector = space.pack({"g": coefficients})
    grid = CartesianGrid(n=[5, 5], x0=[0, 0], x1=[4000, 200], dims=["x", "z"])
    sampled = vector.to_grid(grid, "g")
    tent = np.array([0, 0.5, 1, 0.5, 0])
    np.testing.assert_allclose(sampled.values, np.outer(tent, tent))
    ax = vector.plot("g", grid=grid)
    assert ax.yaxis_inverted()
    assert len(ax.collections) == 1
    frozen = space.with_support(
        {"g": [True, True, True, True, False, True, True, True, True]}
    )
    masked = frozen.ones().to_grid(grid, "g")
    assert np.isnan(masked.values[2, 2]) and np.isfinite(masked.values[0, 0])


def test_profile_to_grid_follows_sloping_surface_and_converts_units(simulation):
    from frequensolve.geometry.grids import CartesianGrid
    from frequensolve.model.property import Property
    from frequensolve.units import UnitConfig

    simulation.units = UnitConfig(defaults={"length": "m"})
    simulation.model.surfaces["seabed"].depth = Property(
        xr.DataArray([100.0, 300.0], dims=["x"], coords={"x": [0.0, 4000.0]})
    )
    space = im.ControlSpace(vp=im.DepthProfile("vp", "sediment", count=5)).bind(
        simulation
    )
    # The control field equals distance below the sloping seabed.
    vector = space.pack({"vp": space.block("vp").control.coordinates})
    grid = CartesianGrid(n=[5, 4], x0=[0, 0], x1=[4, 1.5], dims=["x", "z"], units="km")
    sampled = vector.to_grid(grid, "vp")
    expected = sampled.z.values[:, None] * 1000 - np.linspace(100, 300, 5)
    expected[expected < 0] = np.nan
    np.testing.assert_allclose(sampled.values, expected, equal_nan=True)
    assert sampled.x.attrs["units"] == "km"
