import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")

from frequensolve.model.layered import LayeredModel
from frequensolve.units import ureg as u


def test_layered_model_exports_units_and_coordinate_systems():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(depth=0.0 * u.km, name="top", system="top_surface")
    model.add_layer(
        name="layer",
        properties={
            "Vp": 1.5 * u.km / u.s,
            "Rho": {
                "value": 1000.0,
                "units": "kg/m^3",
                "system": "density_grid",
            },
        },
        system="surface_depth",
    )
    model.add_surface(
        depth={"value": 0.5, "units": "km"},
        name="bottom",
        coordinate_system="depth_surface",
    )

    payload = model.to_fs()

    assert "z_ref" not in payload["surfaces"][0]
    assert "z_phys" not in payload["surfaces"][0]
    assert payload["surfaces"][0]["depth"] == {
        "value": 0.0,
        "units": "km",
        "system": "top_surface",
    }
    assert "z_ref" not in payload["surfaces"][1]
    assert "z_phys" not in payload["surfaces"][1]
    assert payload["surfaces"][1]["depth"] == {
        "value": 0.5,
        "units": "km",
        "system": "depth_surface",
    }
    assert payload["subdomains"][0]["properties"]["vp"] == {
        "value": 1.5,
        "units": "km/s",
        "system": "surface_depth",
    }
    assert payload["subdomains"][0]["properties"]["rho"] == {
        "value": 1000.0,
        "units": "kg/m^3",
        "system": "density_grid",
    }


def test_layered_model_samples_physical_grid_without_reference_remap():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    interface = xr.DataArray(
        [0.25, 0.75],
        dims=["x"],
        coords={"x": [0.0, 1.0]},
    )

    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="upper", properties={"Vp": 1.0})
    model.add_surface(name="interface", depth=interface)
    model.add_layer(name="lower", properties={"Vp": 2.0})
    model.add_surface(name="bottom", depth=1.0)

    sampled = model.sample_uniform([3, 5])

    np.testing.assert_allclose(sampled.coords["x"].values, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        sampled.coords["z"].values,
        [0.0, 0.25, 0.5, 0.75, 1.0],
    )
    np.testing.assert_allclose(
        sampled["vp"].values,
        [
            [1.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 2.0, 2.0],
        ],
    )


def test_layered_model_sampling_converts_mixed_property_units():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="upper", properties={"Vp": 1.5 * u.km / u.s})
    model.add_surface(name="interface", depth=0.5)
    model.add_layer(name="lower", properties={"Vp": 2000.0 * u.m / u.s})
    model.add_surface(name="bottom", depth=1.0)

    sampled = model.sample_uniform([2, 3])
    depths, log = model.get_1D_log("Vp", x=0.5, dz=0.5, units="m/s")

    assert sampled["vp"].attrs["units"] == "km/s"
    np.testing.assert_allclose(
        sampled["vp"].values,
        [[1.5, 2.0, 2.0], [1.5, 2.0, 2.0]],
    )
    assert model.extreme_values("Vp") == (1.5, 2.0)
    assert model.extreme_values("Vp", units="m/s") == (1500.0, 2000.0)
    np.testing.assert_allclose(depths, [0.0, 0.5])
    np.testing.assert_allclose(log, [1500.0, 2000.0])


def test_layered_model_plot_uses_property_units():
    import matplotlib.pyplot as plt

    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer", properties={"Vp": 1500.0 * u.m / u.s})
    model.add_surface(name="bottom", depth=1.0)

    fig, ax = plt.subplots()
    image = model.plot(
        "Vp",
        resolution=[2, 2],
        ax=ax,
        surfaces=False,
    )

    assert image.colorbar.ax.get_ylabel() == "Vp [m/s]"
    plt.close(fig)


def test_layered_model_export_does_not_mutate_bottom_surface_interface():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer", properties={"Vp": 1.0})
    model.add_surface(name="bottom", depth=1.0)
    model.surfaces[-1].interface = False

    payload = model.to_fs()

    assert payload["surfaces"][-1]["interface"] is True
    assert model.surfaces[-1].interface is False


def test_layered_model_update_from_dataset_does_not_mutate_input_dataset():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer", properties={"Vp": 1.0})
    model.add_surface(name="bottom", depth=1.0)
    dataset = xr.Dataset(
        {
            "vp": xr.DataArray(
                [[np.nan, 2.0], [3.0, 4.0]],
                dims=["x", "z"],
                coords={"x": [0.0, 1.0], "z": [0.0, 1.0]},
            )
        }
    )

    model.update_from_dataset(dataset)

    assert np.isnan(dataset["vp"].values[0, 0])
    assert not np.isnan(model.layers[0].properties["vp"].get().values).any()


def test_layered_model_validates_sampling_and_completion():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])

    with pytest.raises(ValueError, match="at least two surfaces"):
        model.to_fs()

    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer", properties={"Vp": 1.0})
    model.add_surface(name="bottom", depth=1.0)

    with pytest.raises(ValueError, match="sample counts"):
        model.sample_uniform([3, 3, 3])
    with pytest.raises(ValueError, match=">= 2"):
        model.sample_uniform([1, 3])


def test_layered_model_rejects_deprecated_frame_kwargs():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])

    with pytest.raises(TypeError, match="frame"):
        model.add_surface(name="top", depth=0.0, frame="reference")

    model.add_surface(name="top", depth=0.0)
    with pytest.raises(TypeError, match="frame"):
        model.add_layer(name="layer", properties={"Vp": 1.0}, frame="reference")

    with pytest.raises(ValueError, match="system"):
        model.add_layer(
            name="layer",
            properties={"Vp": 1.0},
            system="surface_depth",
            coordinate_system="other",
        )
