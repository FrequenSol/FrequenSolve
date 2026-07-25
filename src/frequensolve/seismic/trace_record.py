"""Trace array helpers.

Trace reads now return plain :class:`xarray.DataArray` objects.  This module is
kept for SEGY export helpers and a light compatibility type alias.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from xarray import DataArray, register_dataarray_accessor

from frequensolve._optional import optional_dependency_error
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import ReceiverGroup, coordinate_array_metadata
from frequensolve.seismic.sources import SourceGroup
from frequensolve.units import unit_expression

TraceRecord = DataArray

__all__ = [
    "array_to_segy",
]


def _required_trace_attribute(trace: DataArray, name: str) -> Any:
    try:
        return trace.attrs[name]
    except KeyError:
        raise ValueError(
            "Trace is missing required FrequenSolve metadata attribute "
            f"{name!r}; reconstructing acquisition geometry for SEG-Y export "
            "requires the original trace metadata. Xarray arithmetic can drop "
            "DataArray attributes. Save `trace_attrs = dict(trace.attrs)` before "
            "the operation and restore them with "
            "`result.attrs.update(trace_attrs)` before calling "
            "`result.fs.to_segy(...)`."
        ) from None


def _source_group(trace: DataArray) -> SourceGroup:
    with open(_required_trace_attribute(trace, "simulation"), "r") as f:
        sim = json.load(f)
    source_index = int(trace.attrs.get("source_id", trace.attrs.get("source_group", 1)))
    acquisition = Acquisition.from_fs(sim["Acquisition"])
    return acquisition.source(source_index)


def _source_coordinates(
    trace: DataArray,
    *,
    preserve_metadata: bool = False,
) -> Any:
    with open(_required_trace_attribute(trace, "simulation"), "r") as f:
        sim = json.load(f)
    acquisition = Acquisition.from_fs(sim["Acquisition"])
    source_index = int(trace.attrs.get("source_id", trace.attrs.get("source_group", 1)))
    coordinates = acquisition.source_coords(
        source_index,
        preserve_metadata=preserve_metadata,
    )
    if preserve_metadata:
        return coordinates
    return np.asarray(coordinates, dtype=float)


def _receiver_group(trace: DataArray) -> ReceiverGroup:
    with open(_required_trace_attribute(trace, "simulation"), "r") as f:
        sim = json.load(f)

    group_name = _required_trace_attribute(trace, "receiver_group")
    for receiver_group in sim["Acquisition"]["receiver_groups"]:
        if receiver_group["name"] == group_name:
            break
    else:
        raise ValueError(f"Receiver group {group_name} not found in simulation.")

    coordinates = receiver_group.get("coordinates")
    if isinstance(coordinates, dict) and coordinates.get("_type") == "CoordsFromFile":
        file = Path(coordinates["file"])
        if not file.is_absolute():
            receiver_group["coordinates"]["file"] = str(
                Path(_required_trace_attribute(trace, "project_path")) / file
            )
    return ReceiverGroup.from_fs(receiver_group)


def _segy_output_unit(units_out: str, ureg) -> tuple[int, Any]:
    units_out = units_out.lower()
    if units_out not in {"m", "ft"}:
        raise ValueError("units_out must be 'm' or 'ft'")
    if units_out == "m":
        return 1, ureg.meter
    return 2, ureg.foot


def _coordinate_scale(units: Any, default_units: str, output_unit: Any, ureg) -> float:
    input_units = unit_expression(units) if units is not None else default_units
    return float(ureg(input_units).to(output_unit).magnitude)


def _coordinates_in_output_units(
    values: Any,
    *,
    units: Any = None,
    default_units: str,
    output_unit: Any,
    ureg,
) -> np.ndarray:
    scale = _coordinate_scale(units, default_units, output_unit, ureg)
    return np.asarray(values, dtype=float) * scale


def _source_coordinates_in_output_units(
    coordinates: Any,
    *,
    default_units: str,
    output_unit: Any,
    ureg,
) -> np.ndarray:
    values, units, _ = coordinate_array_metadata(coordinates)
    values = _coordinates_in_output_units(
        values,
        units=units,
        default_units=default_units,
        output_unit=output_unit,
        ureg=ureg,
    )
    if values.ndim == 0:
        raise ValueError("source coordinates must contain at least one coordinate")
    if values.ndim == 1:
        return values
    if values.ndim == 2 and len(values) == 1:
        return values[0]
    raise ValueError("source reference coordinates must be a single vector")


@register_dataarray_accessor("fs")
class TraceAccessor:
    """FrequenSolve helpers for trace ``DataArray`` objects.

    Args:
        trace: Trace data array being accessed through ``trace.fs``.
    """

    def __init__(self, trace: DataArray):
        self._trace = trace

    @property
    def source_group(self) -> SourceGroup:
        """Return source-group metadata reconstructed from trace attributes."""

        return _source_group(self._trace)

    @property
    def source_coordinates(self) -> np.ndarray:
        """Return source-field reference coordinates for this trace."""

        return _source_coordinates(self._trace)

    @property
    def receiver_group(self) -> ReceiverGroup:
        """Return receiver-group metadata reconstructed from trace attributes."""

        return _receiver_group(self._trace)

    def to_segy(
        self,
        file: str | Path,
        units_in: str = "km",
        units_out: str = "m",
        **kwargs,
    ) -> Path:
        """Write a time-domain trace gather to a SEG-Y file.

        Args:
            file: Destination SEG-Y file path.
            units_in: Assumed input coordinate units when trace metadata does
                not provide units.
            units_out: Output coordinate units, either ``"m"`` or ``"ft"``.
            **kwargs: Optional export controls, including ``T_max`` and
                ``preview``.

        Returns:
            Path to the written SEG-Y file.
        """

        import datetime

        import pint

        try:
            import segyio
        except ModuleNotFoundError as exc:
            raise optional_dependency_error(
                "SEG-Y trace export",
                extra="seismic-io",
                dependencies=("segyio",),
                error=exc,
            ) from exc

        trace = self._trace
        coordinates = {"length": 1, "arcseconds": 2, "degrees": 3, "DMS": 4}

        t_max = kwargs.get("T_max", None)
        preview = kwargs.get("preview", False)
        td = trace.sel(time=slice(None, t_max)) if t_max is not None else trace
        n_samples = td.shape[0]

        receiver_group = self.receiver_group
        source_coordinates = _source_coordinates(trace, preserve_metadata=True)

        ureg = pint.UnitRegistry()
        coordinate_units, output_unit = _segy_output_unit(units_out, ureg)
        receiver_coords = _coordinates_in_output_units(
            receiver_group.coordinates.get(),
            units=getattr(receiver_group.coordinates, "units", None),
            default_units=units_in,
            output_unit=output_unit,
            ureg=ureg,
        )
        source_coords = _source_coordinates_in_output_units(
            source_coordinates,
            default_units=units_in,
            output_unit=output_unit,
            ureg=ureg,
        )

        n_traces = receiver_group.size
        t0 = td.coords["time"].values[0]
        dt = td.coords["time"].values[1] - td.coords["time"].values[0]
        sample_interval = int(dt * 1e6)
        time_samples = td.coords["time"].values

        now = datetime.datetime.now()
        source_x = int(source_coords[0])
        source_elev = -int(source_coords[-1])

        spec = segyio.spec()
        spec.format = 5
        spec.samples = time_samples * 1e3
        spec.sample_rate = sample_interval
        spec.tracecount = n_traces

        file = Path(file)
        with segyio.create(str(file), spec) as f:
            f.bin[segyio.BinField.MeasurementSystem] = coordinate_units
            f.bin[segyio.BinField.Interval] = sample_interval

            for itrace in range(n_traces):
                recv_x = int(receiver_coords[itrace, 0])
                recv_elev = -int(receiver_coords[itrace, -1])
                f.header[itrace].update(
                    {
                        segyio.TraceField.TRACE_SEQUENCE_LINE: itrace + 1,
                        segyio.TraceField.TRACE_SEQUENCE_FILE: itrace + 1,
                        segyio.TraceField.FieldRecord: 1,
                        segyio.TraceField.TraceNumber: itrace + 1,
                        segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: sample_interval,
                        segyio.TraceField.CoordinateUnits: coordinates["length"],
                        segyio.TraceField.DelayRecordingTime: int(t0 * 1000),
                        segyio.TraceField.SourceX: source_x,
                        segyio.TraceField.SourceY: 0,
                        segyio.TraceField.SourceDepth: -source_elev,
                        segyio.TraceField.GroupX: recv_x,
                        segyio.TraceField.GroupY: 0,
                        segyio.TraceField.ReceiverGroupElevation: -recv_elev,
                        segyio.TraceField.YearDataRecorded: now.year,
                        segyio.TraceField.DayOfYear: now.timetuple().tm_yday,
                        segyio.TraceField.HourOfDay: now.hour,
                        segyio.TraceField.MinuteOfHour: now.minute,
                        segyio.TraceField.SecondOfMinute: now.second,
                    }
                )
                f.trace[itrace] = (
                    td.sel(receiver=(itrace + 1)).data.copy().astype(np.float32)
                )

        if preview:
            with segyio.open(str(file), mode="r", strict=False) as sgy:
                print(
                    {
                        "file": file,
                        "size_kb": Path(file).stat().st_size / 1024,
                        "tracecount": sgy.tracecount,
                        "samples": sgy.samples,
                        "sample_interval_us": sgy.bin[segyio.BinField.Interval],
                    }
                )
        return file


def array_to_segy(
    samples: np.ndarray,
    data: np.ndarray,
    fname: str,
    units_in: str = "km",
    units_out: str = "m",
    **kwargs: Any,
) -> None:
    """Write a single time-domain array to a minimal SEG-Y file.

    Args:
        samples: Time sample values in seconds.
        data: Trace values with shape ``(n_samples, n_traces)`` or compatible
            one-dimensional input.
        fname: Destination SEG-Y file path.
        units_in: Assumed input coordinate units for metadata.
        units_out: Output coordinate units, either ``"m"`` or ``"ft"``.
        **kwargs: Optional SEG-Y header and preview controls.
    """

    import datetime

    import pint

    try:
        import segyio
    except ModuleNotFoundError as exc:
        raise optional_dependency_error(
            "SEG-Y trace export",
            extra="seismic-io",
            dependencies=("segyio",),
            error=exc,
        ) from exc

    ureg = pint.UnitRegistry()
    if units_out not in {"m", "ft"}:
        raise ValueError("units_out must be 'm' or 'ft'")
    iunit = ureg(units_in)

    if units_out.lower() == "m":
        coordinate_units = 1
    else:
        coordinate_units = 2
    iunit.to(ureg.meter if coordinate_units == 1 else ureg.foot)

    n_samples = samples.shape[0]
    sample_interval = int((samples[1] - samples[0]) * 1e3)
    now = datetime.datetime.now()

    spec = segyio.spec()
    spec.sorting = 1
    spec.format = 5
    spec.samples = samples
    spec.ilines = [0]
    spec.xlines = [0]

    with segyio.create(fname, spec) as f:
        f.bin[segyio.BinField.Interval] = sample_interval
        f.bin[segyio.BinField.Samples] = n_samples
        f.bin[segyio.BinField.MeasurementSystem] = coordinate_units
        f.header[0].update(
            {
                segyio.TraceField.TRACE_SEQUENCE_LINE: 1,
                segyio.TraceField.TRACE_SEQUENCE_FILE: 1,
                segyio.TraceField.FieldRecord: 0,
                segyio.TraceField.TraceNumber: 0,
                segyio.TraceField.CoordinateUnits: 1,
                segyio.TraceField.SourceX: 0,
                segyio.TraceField.SourceY: 0,
                segyio.TraceField.SourceDepth: 0,
                segyio.TraceField.GroupX: 0,
                segyio.TraceField.GroupY: 0,
                segyio.TraceField.ReceiverGroupElevation: 0,
                segyio.TraceField.YearDataRecorded: now.year,
                segyio.TraceField.DayOfYear: now.timetuple().tm_yday,
                segyio.TraceField.HourOfDay: now.hour,
                segyio.TraceField.MinuteOfHour: now.minute,
                segyio.TraceField.SecondOfMinute: now.second,
            }
        )
        f.trace[0] = np.single(data.copy())
