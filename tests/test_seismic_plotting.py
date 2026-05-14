import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr

from frequensolve.plotting.analysis import compute_nrms, compute_timelag
from frequensolve.plotting.animate import animate_gather
from frequensolve.plotting.traces import (
    diff_gathers,
    plot_cf,
    plot_gather,
    plot_timelag,
    plot_xf,
)
from frequensolve.plotting.vtu import plot_vtu, plot_vtu_slice, read_vtu, vtu_fields


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
    path = tmp_path / "fields.vtu"
    mesh.save(path, binary=False)
    text = path.read_text()
    metadata = {
        "pressure_real": ('fs_display_name="pressure (Real)"', 'fs_units="Pa"'),
        "pressure_imag": ('fs_display_name="pressure (Imaginary)"', 'fs_units="Pa"'),
        "pressure_1_re": ('FS_DISPLAY_NAME="pressure (Real)"', 'FS_UNITS="Pa"'),
        "pressure_1_im": ('fs_display_name="pressure (Imaginary)"', 'fs_units="Pa"'),
        "velocity": ('fs_display_name="velocity"', 'fs_units="m/s"'),
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


def test_vtu_default_view_orients_depth_down(tmp_path):
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
        (0.0, 0.0, -1.0),
    )
    plotter.close()


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
