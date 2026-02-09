import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
from xarray import DataArray

from frequensolve.seismic.receivers import ReceiverFiber, ReceiverGroup
from frequensolve.seismic.sources import SourceGroup
from frequensolve.seismic.wavelet import Wavelet
from frequensolve.simulation.sampling import Sampling, UniformSweepSampling

try:
    import pyfftw

    pyfftw.interfaces.cache.enable()
    fft = pyfftw.interfaces.numpy_fft
    pyfftw.config.NUM_THREADS = 4
except:
    warnings.warn("pyfftw not found, using numpy for FFT (slow)")
    import numpy.fft as fft

__all__ = [
    "ShotRecord",
    "array_to_segy",
]


class ShotRecord(DataArray):
    """Container for storing a shot record, including source, receiver info, and field data.

    A ShotRecord may represent frequency-domain (FD) or time-domain (TD) data, along with the
    sampling info. It is aware of the source, receiver configuration, and raw data arrays.

    Attributes:
       type (str): "FD" or "TD" indicating frequency- or time-domain data.
       number (int): Shot number (e.g., source index).
       sampling (Sampling): A Sampling object describing frequency/time ranges.
       source (Source): The associated Source object for this shot.
       receiver_group (ReceiverGroup): The associated ReceiverGroup object.
       field (str): Field name (e.g. "pressure", "displacement").
       data (np.ndarray): The raw data array, shape depends on FD or TD usage.
    """

    __slots__ = ()

    @property
    def _source_group(self) -> SourceGroup:
        with open(self.attrs["simulation"], "r") as f:
            sim = json.load(f)
        shot = self.attrs["source_group"]
        sgroup = sim["Acquisition"]["source_groups"][shot - 1]
        return SourceGroup.from_dict(sgroup)

    @property
    def _receiver_group(self) -> ReceiverGroup:
        with open(self.attrs["simulation"], "r") as f:
            sim = json.load(f)

        # TODO: this is a hack since receivers read from project path
        cwd = os.getcwd()
        os.chdir(self.attrs["project_path"])
        group = self.attrs["receiver_group"]
        for rgroup in sim["Acquisition"]["receiver_groups"]:
            if rgroup["name"] == group:
                break
        else:
            raise ValueError(f"Receiver group {group} not found in simulation.")
        rgrp = ReceiverGroup.from_dict(rgroup)
        os.chdir(cwd)
        return rgrp

    def to_segy(self, file: str, units_in: str = "km", units_out: str = "m", **kwargs):
        """
        Writes DataArray as SEGY file

        Args:
            file: Output filename
            units_in: Input coordinate units (default: "km")
            units_out: Output coordinate units (default: "m")
            **kwargs: Additional keyword arguments
        """
        import datetime

        import pint
        import segyio

        coordinates = {"length": 1, "arcseconds": 2, "degrees": 3, "DMS": 4}

        # TODO: check these and make conversion
        src_code = {
            "u_z": 1,
            "u_y": 2,
            "u_x": 3,
            "impulsive_z": 11,
            "impulsive_y": 12,
            "impulsive_x": 13,
        }

        recv_code = {
            "generic": 1,
            "signal": 9,
            "pressure": 11,
            "u_z": 12,
            "u_y": 13,
            "u_x": 14,
        }

        T_max = kwargs.get("T_max", None)
        preview = kwargs.get("preview", False)
        if T_max is not None:
            td = self.sel(time=slice(None, T_max))
        else:
            td = self
        n_samples = td.shape[0]

        rgroup = self._receiver_group
        source = self._source_group.source

        # Initialize the pint unit registry
        ureg = pint.UnitRegistry()
        assert units_out == "m" or units_out == "ft"
        iunit = ureg(units_in)

        # specify coordinates
        if units_out.lower() == "m":
            coordinate_units = 1
            ounit = ureg.meter
        elif units_out.lower() == "ft":
            coordinate_units = 2
            ounit = ureg.foot
        else:
            raise ValueError("units_out must be 'meters' or 'feet'")
        scale = iunit.to(ounit).magnitude

        # Get sampling parameters
        n_traces = rgroup.size
        t0 = td.coords["time"].values[0] / 1000
        dt = td.coords["time"].values[1] - td.coords["time"].values[0]  # Seconds
        sample_interval = int(dt * 1e3)  # Microseconds
        time_samples = td.coords["time"].values

        now = datetime.datetime.now()
        year = now.year
        day = now.timetuple().tm_yday
        hour = now.hour
        minute = now.minute
        second = now.second

        source_x = int((source.coordinates[0] * scale))
        source_elev = -int((source.coordinates[-1] * scale))

        # SEGY file settings
        spec = segyio.spec()
        spec.format = 5
        spec.samples = time_samples  # Time samples (milliseconds)
        spec.sample_rate = sample_interval
        spec.tracecount = n_traces

        # Create a SEGY file and write data
        with segyio.create(file, spec) as f:
            f.bin[segyio.BinField.MeasurementSystem] = (
                coordinate_units  # 1 for meters, 2 for feet
            )
            f.bin[segyio.BinField.Interval] = int(dt * 1e3)

            for itrace in range(n_traces):
                recv_x = int((rgroup.coordinates[itrace, 0] * scale))
                recv_elev = -int((rgroup.coordinates[itrace, -1] * scale))

                f.header[itrace].update(
                    {
                        # Trace number
                        segyio.TraceField.TRACE_SEQUENCE_LINE: itrace + 1,
                        segyio.TraceField.TRACE_SEQUENCE_FILE: itrace + 1,
                        segyio.TraceField.FieldRecord: 1,
                        segyio.TraceField.TraceNumber: itrace + 1,
                        segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: sample_interval,
                        segyio.TraceField.CoordinateUnits: coordinates["length"],
                        segyio.TraceField.DelayRecordingTime: int(t0 * 1000),
                        # Source
                        segyio.TraceField.SourceX: source_x,
                        segyio.TraceField.SourceY: 0,
                        segyio.TraceField.SourceDepth: -source_elev,
                        # Receiver location
                        segyio.TraceField.GroupX: recv_x,
                        segyio.TraceField.GroupY: 0,
                        segyio.TraceField.ReceiverGroupElevation: -recv_elev,
                        # Set time
                        segyio.TraceField.YearDataRecorded: year,
                        segyio.TraceField.DayOfYear: day,
                        segyio.TraceField.HourOfDay: hour,
                        segyio.TraceField.MinuteOfHour: minute,
                        segyio.TraceField.SecondOfMinute: second,
                    }
                )
                f.trace[itrace] = (
                    td.sel(receiver=(itrace + 1)).data.copy().astype(np.float32)
                )

        if preview:
            with segyio.open(file, mode="r", strict=False) as sgy:
                print(f"\nSEGY File Contents:")
                print(f"File size: {os.path.getsize(file) / 1024:.1f} KB")
                print(f"Number of traces: {sgy.tracecount}")
                print(f"Samples per trace: {sgy.samples}")
                print(f"Sample interval: {sgy.bin[segyio.BinField.Interval]} μs")
                print(
                    f"Measurement system: {'meters' if sgy.bin[segyio.BinField.MeasurementSystem] == 1 else 'feet'}"
                )
                print(f"\nFirst trace:")
                itrace = 3
                print(
                    f"Source coordinates (x,y,z): "
                    f"({sgy.header[itrace][segyio.TraceField.SourceX]},"
                    f" {sgy.header[itrace][segyio.TraceField.SourceY]},"
                    f" {sgy.header[itrace][segyio.TraceField.SourceDepth]})"
                )
                print(
                    f"Receiver coordinates (x,y,z): "
                    f"({sgy.header[itrace][segyio.TraceField.GroupX]},"
                    f" {sgy.header[itrace][segyio.TraceField.GroupY]},"
                    f" {sgy.header[itrace][segyio.TraceField.ReceiverGroupElevation]})"
                )
                print(
                    f"Data min/max: {sgy.trace[0].min():.2e} / {sgy.trace[0].max():.2e}"
                )


# --------------------------------------------
# Output to SEGY
# --------------------------------------------
def array_to_segy(
    samples: np.ndarray,
    data: np.ndarray,
    fname: str,
    units_in: str = "km",
    units_out: str = "m",
    **kwargs,
):
    """
    Writes time-domain field as segy file

    Parameters
    ----------
    key : {group.name}:{field.name}
    shot: Source number
    """
    import datetime

    import pint
    import segyio

    ureg = pint.UnitRegistry()
    assert units_out == "m" or units_out == "ft"
    iunit = ureg(units_in)

    if units_out.lower() == "m":
        coordinate_units = 1
        ounit = ureg.meter
    elif units_out.lower() == "ft":
        coordinate_units = 2
        ounit = ureg.foot
    else:
        raise ValueError("units_out must be 'meters' or 'feet'")
    scale = iunit.to(ounit).magnitude

    n_samples = samples.shape[0]
    n_traces = 1
    sample_interval = int((samples[1] - samples[0]) * 1e3)

    now = datetime.datetime.now()

    ilines = [0]
    xlines = [0]

    # SEGY file settings
    spec = segyio.spec()
    spec.sorting = 1
    spec.format = 5
    spec.samples = samples
    spec.ilines = ilines  # Inline indices
    spec.xlines = xlines  # Crossline indices

    with segyio.create(fname, spec) as f:
        f.bin[segyio.BinField.Interval] = sample_interval
        f.bin[segyio.BinField.Samples] = n_samples
        f.bin[segyio.BinField.MeasurementSystem] = (
            coordinate_units  # 1 for meters, 2 for feet
        )

        itrace = -1
        for i_recv in ilines:
            for x_recv in xlines:
                itrace += 1
                f.header[i_recv].update(
                    {
                        # Trace number
                        segyio.TraceField.TRACE_SEQUENCE_LINE: 1,
                        segyio.TraceField.TRACE_SEQUENCE_FILE: 1,
                        segyio.TraceField.FieldRecord: itrace,
                        segyio.TraceField.TraceNumber: itrace,
                        segyio.TraceField.CoordinateUnits: 1,
                        # Source
                        segyio.TraceField.SourceX: 0,
                        segyio.TraceField.SourceY: 0,
                        segyio.TraceField.SourceDepth: 0,
                        # Receiver location
                        segyio.TraceField.GroupX: 0,
                        segyio.TraceField.GroupY: 0,
                        segyio.TraceField.ReceiverGroupElevation: 0,
                        # Set time
                        segyio.TraceField.YearDataRecorded: now.year,
                        segyio.TraceField.DayOfYear: now.timetuple().tm_yday,
                        segyio.TraceField.HourOfDay: now.hour,
                        segyio.TraceField.MinuteOfHour: now.minute,
                        segyio.TraceField.SecondOfMinute: now.second,
                    }
                )
                f.trace[itrace] = np.single(data.copy())
