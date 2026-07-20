import matplotlib

matplotlib.use("Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from frequensolve.plotting.analysis import compute_nrms, compute_timelag
from frequensolve.plotting.animate import animate_gather, animate_wavefield
from frequensolve.plotting.layered import _plot_acquisition
from frequensolve.plotting.traces import (
    diff_gathers,
    plot_cf,
    plot_gather,
    plot_timelag,
    plot_wavefield,
    plot_xf,
)
from frequensolve.plotting.vtu import (
    _scalar_bar_args,
    plot_vtu,
    plot_vtu_slice,
    read_vtu,
    vtu_fields,
)
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import ReceiverNode
from frequensolve.seismic.sources import SourceGeometry
from frequensolve.seismic.traces import TraceDataset
from frequensolve.simulation.jobs.artifacts import TraceManifest


def _time_trace(n_time=64, n_receiver=8):
    time = np.linspace(0.0, 0.63, n_time)
    receiver = np.linspace(0.0, 0.7, n_receiver)
    data = np.zeros((n_time, n_receiver), dtype=float)
    for i in range(n_receiver):
        center = 12 + i
        data[:, i] = np.exp(-0.5 * ((np.arange(n_time) - center) / 3.0) ** 2)
    trace = xr.DataArray(
        data,
        dims=("time", "receiver"),
        coords={"time": time, "receiver": receiver},
    )
    trace.coords["time"].attrs["units"] = "s"
    trace.coords["receiver"].attrs["units"] = "km"
    return trace


def _frequency_trace(n_frequency=24, n_receiver=8):
    frequency = np.linspace(0.0, 30.0, n_frequency)
    receiver = np.linspace(0.0, 0.7, n_receiver)
    phase = np.exp(1j * 2.0 * np.pi * frequency[:, None] * receiver[None, :] / 2.0)
    return xr.DataArray(
        phase,
        dims=("frequency", "receiver"),
        coords={"frequency": frequency, "receiver": receiver},
    )


@pytest.mark.parametrize("geometry_type", ["HDF5", "SPSFiles"])
def test_plot_acquisition_skips_external_source_catalogs(geometry_type):
    if geometry_type == "HDF5":
        geometry = SourceGeometry.hdf5("sources.h5", dataset="/sources", kind="scalar")
    else:
        geometry = SourceGeometry.sps("sources.sps", kind="scalar")
    acquisition = Acquisition(source_geometry=geometry)
    acquisition.add_receiver_group(
        name="line",
        device=ReceiverNode(name="hydrophone"),
        coords=[[0.0, 0.0], [1.0, 0.0]],
    )

    fig, ax = plt.subplots()
    try:
        _plot_acquisition(None, acquisition, ax)

        assert [collection.get_label() for collection in ax.collections] == [
            "Receivers (line)"
        ]
    finally:
        plt.close(fig)


def test_plot_gather_returns_figure_and_axis():
    fig, ax = plot_gather(_time_trace(), A=1.0)

    assert fig is ax.figure
    assert len(ax.images) == 1


def test_diff_gathers_returns_three_axes():
    trace = _time_trace()
    fig, axes = diff_gathers(trace, 0.8 * trace)

    assert len(axes) == 3
    assert all(axis.figure is fig for axis in axes)


def test_frequency_plots_accept_dataarrays():
    trace = _frequency_trace()

    fig_xf, ax_xf = plot_xf(trace)
    fig_cf, ax_cf = plot_cf(trace, n_c=16)

    assert fig_xf is ax_xf.figure
    assert fig_cf is ax_cf.figure
    assert len(ax_cf.images) == 1


def test_gather_plots_accept_common_image_options():
    trace = _frequency_trace()

    for plotter, kwargs in ((plot_xf, {}), (plot_cf, {"n_c": 16})):
        fig, ax = plotter(
            trace,
            title="Common title",
            grid=True,
            colorbar=True,
            aspect="auto",
            interpolation="nearest",
            **kwargs,
        )

        assert fig is ax.figure
        assert ax.get_title() == "Common title"
        assert any(line.get_visible() for line in ax.get_xgridlines())
        assert len(fig.axes) == 2

    time_trace = _time_trace()
    fig, axes = diff_gathers(
        time_trace,
        0.8 * time_trace,
        title="Difference title",
        grid=True,
        colorbar=True,
        interpolation="nearest",
    )

    assert fig._suptitle.get_text() == "Difference title"
    assert all(any(line.get_visible() for line in ax.get_xgridlines()) for ax in axes)
    assert len(fig.axes) == 6


def test_plot_gather_points_frequency_domain_traces_to_right_api():
    trace = _frequency_trace()

    with pytest.raises(ValueError, match=r"traces\.td.*plot_xf"):
        plot_gather(trace, T_max=0.3)


def test_timelag_and_nrms_helpers_return_per_receiver_arrays():
    trace = _time_trace()
    shifted = trace.shift(time=1, fill_value=0.0)

    lag = compute_timelag(trace, shifted, window_length=0.12)
    nrms = compute_nrms(trace, trace, shifted, window_length=0.12)
    fig, ax = plot_timelag(trace, shifted, window_length=0.12)

    assert lag.dims == ("receiver",)
    assert nrms.dims == ("receiver",)
    assert lag.shape == (trace.sizes["receiver"],)
    assert nrms.shape == (trace.sizes["receiver"],)
    assert lag.attrs["units"] == "ms"
    assert nrms.attrs["units"] == "%"
    assert fig is ax.figure


def test_animate_gather_returns_animation():
    trace = _time_trace(n_time=4, n_receiver=4)

    animation = animate_gather(trace, grid_shape=(2, 2))

    assert animation is not None
    text = animation._fig.axes[0].texts[0]
    assert text.get_text() == "t = 0 s"
    assert text.get_position() == pytest.approx((0.02, 0.02))
    assert text.get_ha() == "left"
    assert text.get_va() == "bottom"
    animation._func(1)
    assert text.get_text() == "t = 0.21 s"


def test_animate_gather_preserves_manual_grid_shape_orientation():
    trace = _time_trace(n_time=2, n_receiver=6)

    animation = animate_gather(trace, grid_shape=(2, 3))
    image = animation._fig.axes[0].images[0]

    assert image.get_array().shape == (3, 2)


def _wavefield_trace_attrs(tmp_path):
    return {
        "wavefield_output": "wavefield",
        "wavefield_grid": {
            "_type": "XArrayGrid",
            "dims": ["z", "r"],
            "coords": {
                "z": {"data": [0.0, 1.0], "units": "km"},
                "r": {"data": [0.0, 1.0, 2.0], "units": "km"},
            },
            "units": "km",
        },
        "project_path": str(tmp_path),
    }


def _fixed_axis_wavefield_trace_attrs(tmp_path):
    return {
        "wavefield_output": "wavefield",
        "wavefield_grid": {
            "_type": "XArrayGrid",
            "dims": ["x", "y", "z"],
            "coords": {
                "x": {"data": [0.0], "units": "m"},
                "y": {"data": [0.0, 1.0], "units": "m"},
                "z": {"data": [0.0, 1.0, 2.0], "units": "m"},
            },
            "units": "m",
        },
        "project_path": str(tmp_path),
    }


def test_animate_wavefield_infers_grid_receiver_shape(tmp_path):
    attrs = _wavefield_trace_attrs(tmp_path)

    trace = xr.DataArray(
        np.arange(12, dtype=float).reshape(2, 6),
        dims=("time", "receiver"),
        coords={"time": [0.0, 1.0], "receiver": np.arange(1, 7)},
        attrs=attrs,
    )

    animation = animate_wavefield(trace)
    image = animation._fig.axes[0].images[0]

    assert image.get_array().shape == (2, 3)
    assert list(image.get_extent()) == [0.0, 2.0, 1.0, 0.0]


def test_animate_wavefield_accepts_singleton_3d_grid_dimension(tmp_path):
    attrs = _fixed_axis_wavefield_trace_attrs(tmp_path)
    trace = xr.DataArray(
        np.arange(12, dtype=float).reshape(2, 6),
        dims=("time", "receiver"),
        coords={"time": [0.0, 1.0], "receiver": np.arange(1, 7)},
        attrs=attrs,
    )

    animation = animate_wavefield(trace)
    image = animation._fig.axes[0].images[0]

    assert image.get_array().shape == (2, 3)
    assert list(image.get_extent()) == [0.0, 2.0, 1.0, 0.0]


def test_plot_wavefield_uses_single_frequency_grid_metadata(tmp_path):
    attrs = _wavefield_trace_attrs(tmp_path)
    data = np.array(
        [np.arange(6, dtype=float) + 1j * np.arange(10, 16, dtype=float)],
        dtype=np.complex128,
    )
    trace = xr.DataArray(
        data,
        dims=("frequency", "receiver"),
        coords={"frequency": [10.0], "receiver": np.arange(1, 7)},
        attrs=attrs,
    )

    expected_by_mode = {
        "real": data[0].real.reshape(2, 3),
        "imag": data[0].imag.reshape(2, 3),
        "abs": np.abs(data[0]).reshape(2, 3),
    }
    for mode, expected in expected_by_mode.items():
        fig, ax = plot_wavefield(trace, mode=mode)
        image = ax.images[0]

        assert fig is ax.figure
        assert image.get_array().shape == (2, 3)
        np.testing.assert_allclose(np.asarray(image.get_array()), expected)
        assert list(image.get_extent()) == [0.0, 2.0, 1.0, 0.0]
        expected_vmin = 0 if mode == "abs" else -image.get_clim()[1]
        assert image.get_clim()[0] == expected_vmin


def test_plot_wavefield_accepts_singleton_3d_grid_dimension(tmp_path):
    attrs = _fixed_axis_wavefield_trace_attrs(tmp_path)
    data = np.array([np.arange(6, dtype=float) + 1j], dtype=np.complex128)
    trace = xr.DataArray(
        data,
        dims=("frequency", "receiver"),
        coords={"frequency": [10.0], "receiver": np.arange(1, 7)},
        attrs=attrs,
    )

    fig, ax = plot_wavefield(trace, mode="real")

    assert fig is ax.figure
    np.testing.assert_allclose(
        np.asarray(ax.images[0].get_array()), data[0].real.reshape(2, 3)
    )
    assert ax.get_xlabel() == "Z [m]"
    assert ax.get_ylabel() == "Y [m]"


def test_plot_wavefield_supports_nonuniform_grid_metadata(tmp_path):
    attrs = {
        "wavefield_output": "wavefield",
        "wavefield_grid": {
            "_type": "XArrayGrid",
            "dims": ["z", "r"],
            "coords": {
                "z": {"data": [0.0, 0.25, 1.0]},
                "r": {"data": [0.0, 2.0]},
            },
        },
        "project_path": str(tmp_path),
    }
    trace = xr.DataArray(
        np.arange(6, dtype=float) + 1j,
        dims=("receiver",),
        coords={"frequency": 10.0, "receiver": np.arange(1, 7)},
        attrs=attrs,
    )

    fig, ax = plot_wavefield(trace, mode="real")

    assert fig is ax.figure
    assert len(ax.images) == 0
    assert len(ax.collections) == 1


def test_plot_wavefield_accepts_scalar_frequency_coordinate(tmp_path):
    attrs = _wavefield_trace_attrs(tmp_path)
    data = np.arange(6, dtype=float) + 1j
    trace = xr.DataArray(
        data,
        dims=("receiver",),
        coords={"frequency": 10.0, "receiver": np.arange(1, 7)},
        attrs=attrs,
    )

    fig, ax = plot_wavefield(trace, mode="real")

    assert fig is ax.figure
    assert ax.images[0].get_array().shape == (2, 3)


def test_plot_wavefield_reads_grid_from_wavefield_manifest(tmp_path):
    grid = {
        "_type": "XArrayGrid",
        "dims": ["z", "r"],
        "coords": {
            "z": {"data": [0.0, 1.0]},
            "r": {"data": [0.0, 1.0, 2.0]},
        },
    }
    output_path = tmp_path / "results" / "wavefields"
    output_path.mkdir(parents=True)
    trace_file = output_path / "traces_1.h5"
    values = np.arange(6, dtype=np.float32)

    with h5py.File(trace_file, "w") as h5:
        h5.create_dataset("frequency", data=10.0)
        dset = h5.create_dataset(
            "pressure_wavefield",
            data=np.stack([values, np.zeros_like(values)], axis=-1).reshape(6, 1, 1, 2),
        )
        dset.attrs["dims"] = ["receiver", "component", "shot"]
        dset.attrs["component"] = ["pressure"]
        dset.attrs["shot"] = [1]

    wavefields = TraceDataset.from_manifest(
        TraceManifest(
            files=[trace_file],
            frequencies={1: 10.0},
            groups=["pressure_wavefield"],
            simulation=tmp_path / "simulation.json",
            result_path=tmp_path / "results",
            output_path=output_path,
            project_path=tmp_path,
            wavefields={
                "pressure_wavefield": {
                    "grid": grid,
                    "fields": ["pressure"],
                    "sources": ["1"],
                }
            },
        )
    )

    trace = wavefields.fd("pressure_wavefield", "pressure", source=1)
    fig, ax = plot_wavefield(trace, mode="real")

    assert fig is ax.figure
    np.testing.assert_allclose(
        np.asarray(ax.images[0].get_array()), values.reshape(2, 3)
    )


def test_plot_wavefield_rejects_frequency_sweeps(tmp_path):
    attrs = _wavefield_trace_attrs(tmp_path)
    trace = xr.DataArray(
        np.zeros((2, 6), dtype=np.complex128),
        dims=("frequency", "receiver"),
        coords={"frequency": [5.0, 10.0], "receiver": np.arange(1, 7)},
        attrs=attrs,
    )

    with pytest.raises(ValueError, match="exactly one frequency"):
        plot_wavefield(trace)


def test_animate_wavefield_supports_single_frequency_harmonic(tmp_path):
    attrs = _wavefield_trace_attrs(tmp_path)
    trace = xr.DataArray(
        1j * np.ones((1, 6), dtype=np.complex128),
        dims=("frequency", "receiver"),
        coords={"frequency": [5.0], "receiver": np.arange(1, 7)},
        attrs=attrs,
    )

    animation = animate_wavefield(trace, frames=4)
    image = animation._fig.axes[0].images[0]
    text = animation._fig.axes[0].texts[0]

    assert text.get_text() == "t = 0 s"
    np.testing.assert_allclose(np.asarray(image.get_array()), np.zeros((2, 3)))
    animation._func(1)
    np.testing.assert_allclose(np.asarray(image.get_array()), -np.ones((2, 3)))
    assert text.get_text() == "t = 0.05 s"


def test_wavefield_plots_accept_dask_backed_amplitude_limit(tmp_path):
    pytest.importorskip("dask.array")
    attrs = _wavefield_trace_attrs(tmp_path)
    values = np.arange(6, dtype=float) + 1j * np.arange(1, 7, dtype=float)
    trace = xr.DataArray(
        values[None, :],
        dims=("frequency", "receiver"),
        coords={"frequency": [5.0], "receiver": np.arange(1, 7)},
        attrs=attrs,
    ).chunk({"frequency": 1, "receiver": 6})
    A = np.std(np.abs(trace))
    expected_A = float(np.std(np.abs(values)))

    fig, ax = plot_wavefield(trace, A=A, mode="real")
    animation = animate_wavefield(trace, A=A, frames=2)

    assert fig is ax.figure
    np.testing.assert_allclose(ax.images[0].get_clim(), (-expected_A, expected_A))
    np.testing.assert_allclose(
        animation._fig.axes[0].images[0].get_clim(),
        (-expected_A, expected_A),
    )


def test_animate_wavefield_rejects_frequency_sweeps(tmp_path):
    attrs = _wavefield_trace_attrs(tmp_path)
    trace = xr.DataArray(
        np.zeros((2, 6), dtype=np.complex128),
        dims=("frequency", "receiver"),
        coords={"frequency": [5.0, 10.0], "receiver": np.arange(1, 7)},
        attrs=attrs,
    )

    with pytest.raises(ValueError, match="exactly one frequency"):
        animate_wavefield(trace)


def _write_sample_vtu(tmp_path):
    pv = pytest.importorskip("pyvista")
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = np.array([4, 0, 1, 2, 3])
    cell_types = np.array([pv.CellType.TETRA], dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    mesh.point_data["pressure_real"] = np.array([0.0, 1.0, 0.5, -0.25])
    mesh.point_data["pressure_imag"] = np.array([0.0, 0.25, 0.5, 1.0])
    mesh.point_data["pressure_1_re"] = np.array([0.0, 1.0, 0.5, -0.25])
    mesh.point_data["pressure_1_im"] = np.array([0.0, 0.25, 0.5, 1.0])
    mesh.point_data["velocity"] = np.column_stack(
        [
            np.linspace(0.0, 1.0, 4),
            np.linspace(1.0, 2.0, 4),
            np.linspace(2.0, 3.0, 4),
        ]
    )
    mesh.point_data["strain"] = np.column_stack(
        [
            np.linspace(0.0, 1.0, 4),
            np.linspace(1.0, 2.0, 4),
            np.linspace(2.0, 3.0, 4),
            np.linspace(3.0, 4.0, 4),
        ]
    )
    mesh.point_data["stress_re"] = np.column_stack(
        [
            np.linspace(0.0, 1.0, 4),
            np.linspace(1.0, 2.0, 4),
            np.linspace(2.0, 3.0, 4),
        ]
    )
    mesh.point_data["stress_im"] = 2.0 * mesh.point_data["stress_re"]
    path = tmp_path / "fields.vtu"
    mesh.save(path, binary=False)
    text = path.read_text()
    metadata = {
        "pressure_real": ('fs_display_name="pressure (Real)"', 'fs_units="Pa"'),
        "pressure_imag": ('fs_display_name="pressure (Imaginary)"', 'fs_units="Pa"'),
        "pressure_1_re": ('FS_DISPLAY_NAME="pressure (Real)"', 'FS_UNITS="Pa"'),
        "pressure_1_im": ('fs_display_name="pressure (Imaginary)"', 'fs_units="Pa"'),
        "velocity": ('fs_display_name="velocity"', 'fs_units="m/s"'),
        "strain": (
            'fs_display_name="strain"',
            'fs_units="1"',
            'fs_components="rr,zz,rz,tt"',
        ),
        "stress_re": (
            'fs_display_name="stress (Real)"',
            'fs_units="Pa"',
            'fs_components="xx,zz,xz"',
        ),
        "stress_im": (
            'fs_display_name="stress (Imaginary)"',
            'fs_units="Pa"',
            'fs_components="xx,zz,xz"',
        ),
    }
    for name, attrs in metadata.items():
        text = text.replace(
            f'Name="{name}"',
            f'Name="{name}" {" ".join(attrs)}',
        )
    path.write_text(text)
    return path


def _write_source_indexed_pressure_vtu(tmp_path):
    pv = pytest.importorskip("pyvista")
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = np.array([4, 0, 1, 2, 3])
    cell_types = np.array([pv.CellType.TETRA], dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    mesh.point_data["pressure_1_re"] = np.array([0.0, 1.0, 0.5, -0.25])
    mesh.point_data["pressure_1_im"] = np.array([0.0, 0.25, 0.5, 1.0])
    mesh.point_data["pressure_1_abs"] = np.hypot(
        mesh.point_data["pressure_1_re"],
        mesh.point_data["pressure_1_im"],
    )
    path = tmp_path / "source_indexed_pressure.vtu"
    mesh.save(path, binary=False)
    return path


def _write_multi_source_strain_vtu(tmp_path):
    pv = pytest.importorskip("pyvista")
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = np.array([4, 0, 1, 2, 3])
    cell_types = np.array([pv.CellType.TETRA], dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    mesh.point_data["strain_1_im"] = np.column_stack(
        [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([10.0, 20.0, 30.0, 40.0]),
            np.array([100.0, 200.0, 300.0, 400.0]),
        ]
    )
    mesh.point_data["strain_2_im"] = 2.0 * mesh.point_data["strain_1_im"]
    path = tmp_path / "multi_source_strain.vtu"
    mesh.save(path, binary=False)
    return path


def test_vtu_helpers_read_fields_and_plot(tmp_path):
    path = _write_sample_vtu(tmp_path)

    mesh = read_vtu(path)
    fields = vtu_fields(mesh)
    plotter, plotted_mesh = plot_vtu(
        path,
        "pressure",
        part="abs",
        show=False,
        return_mesh=True,
    )

    assert "pressure_real" in fields
    assert "pressure_imag" in fields
    assert "Pressure (Magnitude) [Pa]" in plotted_mesh.point_data
    assert plotter is not None
    plotter.close()


def test_vtu_base_field_alias_accepts_unique_source_indexed_pressure(tmp_path):
    path = _write_source_indexed_pressure_vtu(tmp_path)

    plotter, plotted_mesh = plot_vtu(
        path,
        "pressure",
        part="re",
        show=False,
        return_mesh=True,
    )

    assert "Pressure (Real)" in plotted_mesh.point_data
    np.testing.assert_allclose(
        plotted_mesh.point_data["Pressure (Real)"],
        plotted_mesh.point_data["pressure_1_re"],
    )
    plotter.close()


def test_vtu_plot_accepts_source_number_for_source_indexed_fields(tmp_path):
    path = _write_multi_source_strain_vtu(tmp_path)

    plotter, plotted_mesh = plot_vtu(
        path,
        "strain",
        part="im",
        component="zz",
        source=2,
        show=False,
        return_mesh=True,
    )

    assert "Strain ZZ (Imaginary)" in plotted_mesh.point_data
    np.testing.assert_allclose(
        plotted_mesh.point_data["Strain ZZ (Imaginary)"],
        plotted_mesh.point_data["strain_2_im"][:, 1],
    )
    plotter.close()


def test_vtu_plot_requires_source_for_ambiguous_source_indexed_fields(tmp_path):
    path = _write_multi_source_strain_vtu(tmp_path)

    with pytest.raises(KeyError, match="ambiguous"):
        plot_vtu(
            path,
            "strain",
            part="im",
            component="zz",
            show=False,
        )


def test_vtu_abs_accepts_solver_re_im_names(tmp_path):
    path = _write_sample_vtu(tmp_path)

    plotter, plotted_mesh = plot_vtu(
        path,
        "pressure_1",
        part="abs",
        show=False,
        return_mesh=True,
    )

    assert "Pressure (Magnitude) [Pa]" in plotted_mesh.point_data
    plotter.close()


def test_vtu_part_aliases_accept_solver_re_im_names(tmp_path):
    path = _write_sample_vtu(tmp_path)

    real_plotter, real_mesh = plot_vtu(
        path,
        "pressure_1",
        part="re",
        show=False,
        return_mesh=True,
    )
    imag_plotter, imag_mesh = plot_vtu(
        path,
        "pressure_1",
        part="im",
        vmin=-1.0,
        vmax=1.0,
        show=False,
        return_mesh=True,
    )

    assert "Pressure (Real) [Pa]" in real_mesh.point_data
    assert "Pressure (Imaginary) [Pa]" in imag_mesh.point_data
    np.testing.assert_allclose(
        real_mesh.point_data["Pressure (Real) [Pa]"],
        real_mesh.point_data["pressure_1_re"],
    )
    np.testing.assert_allclose(
        imag_mesh.point_data["Pressure (Imaginary) [Pa]"],
        imag_mesh.point_data["pressure_1_im"],
    )
    real_plotter.close()
    imag_plotter.close()


def test_vtu_default_view_orients_x_right_and_depth_down(tmp_path):
    pv = pytest.importorskip("pyvista")
    path = _write_sample_vtu(tmp_path)

    plotter, _ = plot_vtu(
        path,
        "pressure_1",
        part="re",
        show=False,
        return_mesh=True,
    )

    np.testing.assert_allclose(plotter.camera.GetViewUp(), (0.0, -1.0, 0.0))
    np.testing.assert_allclose(
        plotter.camera.GetDirectionOfProjection(),
        (0.0, 0.0, 1.0),
    )
    coordinate = pv._vtk.vtkCoordinate()
    coordinate.SetCoordinateSystemToWorld()
    coordinate.SetValue(0.0, 0.0, 0.0)
    origin = coordinate.GetComputedDisplayValue(plotter.renderer)
    coordinate.SetValue(1.0, 0.0, 0.0)
    positive_x = coordinate.GetComputedDisplayValue(plotter.renderer)
    coordinate.SetValue(0.0, 1.0, 0.0)
    positive_depth = coordinate.GetComputedDisplayValue(plotter.renderer)
    assert positive_x[0] > origin[0]
    assert positive_depth[1] < origin[1]
    plotter.close()


def test_vtu_plot_accepts_camera_zoom(tmp_path):
    path = _write_sample_vtu(tmp_path)

    plotter, _ = plot_vtu(
        path,
        "pressure_1",
        part="re",
        zoom=1.25,
        show=False,
        return_mesh=True,
    )

    assert plotter is not None
    plotter.close()


def test_vtu_plot_rejects_non_positive_camera_zoom(tmp_path):
    path = _write_sample_vtu(tmp_path)

    with pytest.raises(ValueError, match="zoom must be positive"):
        plot_vtu(path, "pressure_1", zoom=0.0, show=False)


def test_vtu_default_scalar_bar_is_horizontal_centered_and_wide():
    show, args = _scalar_bar_args(True, None, title="Pressure")

    assert show is True
    assert args["title"] == "Pressure"
    assert args["vertical"] is False
    assert args["position_x"] == pytest.approx(0.22)
    assert args["position_y"] == pytest.approx(0.06)
    assert args["width"] == pytest.approx(0.56)
    assert args["height"] == pytest.approx(0.08)


def test_vtu_scalar_bar_options_override_defaults():
    show, args = _scalar_bar_args(
        {"vertical": True, "width": 0.1},
        {"title": "Custom"},
        title="Pressure",
    )

    assert show is True
    assert args["title"] == "Custom"
    assert args["vertical"] is True
    assert args["width"] == pytest.approx(0.1)
    assert args["position_x"] == pytest.approx(0.22)


def test_vtu_part_aliases_accept_explicit_solver_array_names(tmp_path):
    path = _write_sample_vtu(tmp_path)

    plotter, plotted_mesh = plot_vtu(
        path,
        "pressure_1_im",
        part="imag",
        show=False,
        return_mesh=True,
    )

    assert "Pressure (Imaginary) [Pa]" in plotted_mesh.point_data
    np.testing.assert_allclose(
        plotted_mesh.point_data["Pressure (Imaginary) [Pa]"],
        plotted_mesh.point_data["pressure_1_im"],
    )
    plotter.close()


def test_vtu_slice_supports_vector_components(tmp_path):
    path = _write_sample_vtu(tmp_path)

    plotter, plotted_mesh = plot_vtu_slice(
        path,
        "velocity",
        component="z",
        show=False,
        return_mesh=True,
    )

    assert "Velocity Z [m/s]" in plotted_mesh.point_data
    assert plotter is not None
    plotter.close()


def test_vtu_plot_supports_metadata_tensor_components(tmp_path):
    path = _write_sample_vtu(tmp_path)

    plotter, plotted_mesh = plot_vtu(
        path,
        "strain",
        component="rz",
        show=False,
        return_mesh=True,
    )

    assert "Strain RZ [1]" in plotted_mesh.point_data
    np.testing.assert_allclose(
        plotted_mesh.point_data["Strain RZ [1]"],
        plotted_mesh.point_data["strain"][:, 2],
    )
    plotter.close()


def test_vtu_abs_supports_components_from_paired_real_imag_fields(tmp_path):
    path = _write_sample_vtu(tmp_path)

    plotter, plotted_mesh = plot_vtu(
        path,
        "stress",
        part="abs",
        component="xz",
        show=False,
        return_mesh=True,
    )

    assert "Stress XZ (Magnitude) [Pa]" in plotted_mesh.point_data
    np.testing.assert_allclose(
        plotted_mesh.point_data["Stress XZ (Magnitude) [Pa]"],
        np.hypot(
            plotted_mesh.point_data["stress_re"][:, 2],
            plotted_mesh.point_data["stress_im"][:, 2],
        ),
    )
    plotter.close()


def test_vtu_plot_supports_solver_tensor_component_fallbacks(tmp_path):
    path = _write_sample_vtu(tmp_path)

    plotter, plotted_mesh = plot_vtu(
        path,
        "velocity",
        component="z",
        show=False,
        return_mesh=True,
    )

    assert "Velocity Z [m/s]" in plotted_mesh.point_data
    np.testing.assert_allclose(
        plotted_mesh.point_data["Velocity Z [m/s]"],
        plotted_mesh.point_data["velocity"][:, 2],
    )
    plotter.close()
