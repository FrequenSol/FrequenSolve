import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np

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
    "read_shot_TD",
    "read_shot_FD",
    "Record",
    "read_frequency",
    "array_to_segy",
]


@dataclass
class ShotRecord:
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

    type: str
    number: int
    sampling: Sampling
    source_group: SourceGroup
    receiver_group: ReceiverGroup
    field: str
    data: np.ndarray

    def write_segy(
        self, fname: str, units_in: str = "km", units_out: str = "m", **kwargs
    ):
        """
        Writes time-domain field as SEGY file

        Args:
            fname: Output filename
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

        Tf = kwargs.get("Tf", None)
        preview = kwargs.get("preview", False)
        nTf, Tf = self.sampling.cutoff(Tf)

        rgroup = self.receiver_group
        source = self.source_group.source

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
        n_samples = nTf
        n_traces = rgroup.size
        dt = self.sampling.dT  # Time step in seconds
        sample_interval = int(dt * 1e6)  # Convert to microseconds

        # Create time samples array (milliseconds)
        time_samples = np.arange(n_samples) * dt * 1000

        now = datetime.datetime.now()
        year = now.year
        day = now.timetuple().tm_yday
        hour = now.hour
        minute = now.minute
        second = now.second

        recv_x = int((rgroup.coordinates[itrace, 0] * scale))
        recv_elev = -int((rgroup.coordinates[itrace, -1] * scale))

        source_x = int((source.coordinates[0] * scale))
        source_elev = -int((source.coordinates[-1] * scale))

        # SEGY file settings
        spec = segyio.spec()
        spec.format = 5
        spec.samples = time_samples  # Time samples (milliseconds)

        # Create a SEGY file and write data
        with segyio.create(fname, spec) as f:
            f.bin[segyio.BinField.MeasurementSystem] = (
                coordinate_units  # 1 for meters, 2 for feet
            )

            for itrace in range(n_traces):
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
                    self.data[:n_samples, itrace].copy().astype(np.float32)
                )

        if preview:
            with segyio.open(fname, mode="r", strict=False) as sgy:
                print(f"\nSEGY File Contents:")
                print(f"File size: {os.path.getsize(fname) / 1024:.1f} KB")
                print(f"Number of traces: {sgy.tracecount}")
                print(f"Samples per trace: {sgy.samples}")
                print(f"Sample interval: {sgy.bin[segyio.BinField.Interval]} μs")
                print(
                    f"Measurement system: {'meters' if sgy.bin[segyio.BinField.MeasurementSystem] == 1 else 'feet'}"
                )
                print(f"\nFirst trace:")
                print(
                    f"Source coordinates (x,y,z): "
                    f"({sgy.header[0][segyio.TraceField.SourceX]},"
                    f" {sgy.header[0][segyio.TraceField.SourceY]},"
                    f" {sgy.header[0][segyio.TraceField.SourceDepth]})"
                )
                print(
                    f"Receiver coordinates (x,y,z): "
                    f"({sgy.header[0][segyio.TraceField.GroupX]},"
                    f" {sgy.header[0][segyio.TraceField.GroupY]},"
                    f" {sgy.header[0][segyio.TraceField.ReceiverGroupElevation]})"
                )
                print(
                    f"Data min/max: {sgy.trace[0].min():.2e} / {sgy.trace[0].max():.2e}"
                )

    def write_segy_TGS(
        self, fname: str, units_in: str = "km", units_out: str = "m", **kwargs
    ):
        """Write a time-domain shot to a SEGY file.

        This version uses TGS's SEGY library that may be convenient for the cloud
        but is lacking documentation and is still under development.

        Args:
           fname (str): Output SEGY file name.
           units_in (str): Units of the input coordinates (defaults to "km").
           units_out (str): Units for the output coordinates (defaults to "m"). Must be 'm' or 'ft'.
           kwargs (dict): Additional options, such as 'Tf' for cutoff time.

        Raises:
           AssertionError: If shot type is not "TD".
           ValueError: If units_out is not "m" or "ft".
        """
        import datetime
        from pathlib import Path

        import pint
        from segy import SegyFile
        from segy.factory import SegyFactory
        from segy.standards import get_segy_standard

        # Ensure correct shot type
        assert self.type == "TD", "SEGY output is only valid for time-domain (TD) data."

        # Unit conversion checks
        ureg = pint.UnitRegistry()
        if units_out.lower() not in ["m", "ft"]:
            raise ValueError("units_out must be 'm' or 'ft' (meters or feet).")
        iunit = ureg(units_in)
        ounit = ureg.meter if units_out.lower() == "m" else ureg.foot
        scale = iunit.to(ounit).magnitude

        # Basic geometry and dimension info
        group = self.receiver_group
        source = self.source
        dim = len(source.coordinates)

        # Optional cutoff time
        Tf = kwargs.get("Tf", None)
        nTf, Tf = self.sampling.cutoff(Tf)  # number of time samples after cutoff

        n_traces = group.size
        n_samples = nTf
        interval = int(self.sampling.dT * 1e6)  # sample interval in microseconds
        trace_datetime = datetime.datetime.now()

        print(interval, n_samples, n_traces)

        # Build SEG-Y config
        config = {
            "spec": get_segy_standard(1.0),
            "samples_per_trace": n_samples,
            "sample_interval": interval,
        }

        factory = SegyFactory(**config)
        txt = factory.create_textual_header()
        bin_ = factory.create_binary_header()

        # # Update binary header schema
        # factory.binary_header_schema.trace_sorting_code = 5  # Common source point
        # factory.binary_header_schema.measurement_system = 1 if units_out.lower() == "m" else 2  # 1 = meters, 2 = feet

        # Recreate binary header with updated schema
        bin_ = factory.create_binary_header()

        headers = factory.create_trace_header_template(size=n_traces)
        samples = factory.create_trace_sample_template(size=n_traces)

        # Populate headers and data
        for itr in range(n_traces):
            headers[itr]["trace_seq_num_reel"] = itr + 1
            headers[itr]["inline"] = itr + 1
            headers[itr]["crossline"] = 1

            # Source position
            headers[itr]["source_coord_x"] = int(source.coordinates[0] * scale)
            if dim == 2:
                headers[itr]["source_coord_y"] = 0
            else:
                headers[itr]["source_coord_y"] = int(source.coordinates[1] * scale)
            headers[itr]["source_surface_elevation"] = int(
                -source.coordinates[-1] * scale
            )

            # Receiver position
            headers[itr]["group_coord_x"] = int(group.coordinates[itr, 0] * scale)
            if dim == 2:
                headers[itr]["group_coord_y"] = 0
            else:
                headers[itr]["group_coord_y"] = int(group.coordinates[itr, 1] * scale)
            headers[itr]["receiver_group_elevation"] = int(
                -group.coordinates[itr, -1] * scale
            )

            # Trace data (slice out the first n_samples from each trace)
            samples[itr] = self.data[:n_samples, itr].copy()

        traces = factory.create_traces(samples=samples, headers=headers)

        # Write the file
        with Path(fname).open(mode="wb") as f:
            f.write(txt)
            f.write(bin_)
            f.write(traces)

        # sgy = SegyFile(fname)
        # sgy.binary_header.to_dataframe()

        # # print(f"file size: {sgy.file_size / 1024**3:0.2f} GiB")
        # # print(f"num traces: {sgy.num_traces:,}")
        # # print(f"sample rate: {sgy.sample_interval}")
        # # print(f"num samples: {sgy.samples_per_trace}")
        # # print(f"sample labels: {sgy.sample_labels // 1000}")


@dataclass
class Record:
    file: str
    project_path: Path  # TODO: this is a hack, fix it.
    simulation: str
    df: float
    f_max: float
    f_map: Dict[str, float]
    _upscale: int

    def __init__(self, record: str, meta: Dict[str, Any], upscale: int = 1):
        self.file = record
        self.project_path = meta["project"]
        self.simulation = Path(meta["simulation"]).with_suffix(".json")
        self.df = meta["df"]
        self.f_max = meta["f_max"]
        self.f_map = meta["f_map"]
        self.upscale = upscale

    @property
    def is_consolidated(self) -> bool:
        """Check if records have been consolidated into single h5 files."""
        fbase = "_".join(self.file.split(":")[0].split("_")[:-1])
        consolidated_file = f"{fbase}_consolidated.h5"
        if not os.path.exists(consolidated_file):
            return False
        return True

    @property
    def group(self) -> str:
        fbase, _ = self.file.split(":")
        fbase = "_".join(fbase.split("_")[:-1])
        return Path(fbase).name

    @property
    def field(self) -> str:
        _, comp = self.file.split(":")
        return "_".join(comp.split("_")[:-1])

    @property
    def source(self) -> int:
        _, comp = self.file.split(":")
        return int(comp.split("_")[-1])

    @property
    def file_base(self) -> str:
        fbase, _ = self.file.split(":")
        return "_".join(fbase.split("_")[:-1])

    @property
    def upscale(self) -> int:
        return self._upscale

    @upscale.setter
    def upscale(self, upscale: int) -> None:
        self._upscale = upscale

    @property
    def source_group(self) -> SourceGroup:
        with open(self.simulation, "r") as f:
            sim = json.load(f)
        group = self.source
        sgroup = sim["Acquisition"]["source_groups"][group - 1]
        return SourceGroup.from_dict(sgroup)

    @property
    def receiver_group(self) -> ReceiverGroup:
        with open(self.simulation, "r") as f:
            sim = json.load(f)
        group = self.group
        for rgroup in sim["Acquisition"]["receiver_groups"]:
            if rgroup["name"] == group:
                break
        else:
            raise ValueError(f"Receiver group {group} not found in simulation.")
        return ReceiverGroup.from_dict(rgroup)

    def read_TD(self, wavelet: Wavelet) -> ShotRecord:
        sampling = self.sampling(self.upscale)
        wavelet.times = sampling.T_list
        return read_shot_TD(self, wavelet, self.upscale)

    def read_FD(self, wavelet: Wavelet) -> ShotRecord:
        sampling = self.sampling(1)
        wavelet.times = sampling.T_list
        return read_shot_FD(self, wavelet)

    def sampling(self, upscale: Optional[int] = None, t_shift: float = 0.0) -> Sampling:
        upscale = self.upscale if upscale is None else upscale
        return UniformSweepSampling(
            f_min=0.0, f_max=self.f_max, df=self.df, upscale=upscale, t_shift=t_shift
        )

    def times(self, upscale: Optional[int] = None) -> np.ndarray:
        return self.sampling(upscale).T_list

    def read_consolidated(self) -> np.ndarray:
        """Read data from a consolidated HDF5 file.

        Args:
            group (str): Name of the receiver group
            field (str): Field component to read (e.g. 'u_x', 'u_z')
            source (int): Source number

        Returns:
            np.ndarray: Complex array containing the frequency domain data
                       with shape (n_frequencies, n_receivers)

        Raises:
            FileNotFoundError: If consolidated file does not exist
            KeyError: If requested data not found in file
        """
        fbase = self.file_base
        field = self.field
        src = self.source

        consolidated_file = f"{fbase}_consolidated.h5"
        if not os.path.exists(consolidated_file):
            raise FileNotFoundError(f"Consolidated file {consolidated_file} not found")

        with h5py.File(consolidated_file, "r") as f:
            # List all available fields in the file
            available_fields = list(f.keys())
            if field not in available_fields:
                raise KeyError(
                    f"Field {field} not found in {consolidated_file}. Available fields: {available_fields}"
                )

            source_group = f[field][f"source_{src}"]
            real_data = source_group["real"][:]
            imag_data = source_group["imag"][:]
            return real_data.astype(np.complex64) + 1j * imag_data.astype(np.complex64)


def read_shot_TD(
    record: Record, wavelet: Wavelet, upscale: Optional[int] = None
) -> ShotRecord:
    """Read a time-domain shot record.

    Args:
        record (Record): Record to read.
        wavelet (Wavelet): Wavelet to use for the shot record.
        upscale (int): Upscale factor for the shot record.

    Returns:
        ShotRecord: The time-domain shot record.
    """
    field = record.field
    isrc = record.source

    # TODO: This is a hack, fix it.
    cwd = os.getcwd()
    os.chdir(record.project_path)

    recv_group = record.receiver_group
    src_group = record.source_group

    upscale = record.upscale if upscale is None else upscale
    sampling = record.sampling(upscale, t_shift=wavelet.center)
    try:
        fd_record = read_shot_FD(record, wavelet)
    except Exception as e:
        raise ValueError(f"Failed reading record: {e}")

    # TODO: end hack
    os.chdir(cwd)

    nf = sampling.nfreq
    nF = sampling.nFreq

    # If upscaled, create a bigger array for inverse transform
    if nF > nf:
        FD = np.zeros((nF, recv_group.size), dtype=np.csingle)
        FD[:nf, :] = fd_record.data[:nf, :]
        td = fft.irfft(FD, axis=0)
    else:
        td = fft.irfft(fd_record.data, axis=0)

    return ShotRecord(
        type="TD",
        number=isrc,
        sampling=sampling,
        source_group=src_group,
        receiver_group=recv_group,
        field=field,
        data=td,
    )


def read_shot_FD(record: Record, wavelet: Wavelet) -> ShotRecord:
    """Read a frequency-domain shot record.

    Args:
        record (Record): Record to read.
        wavelet (Wavelet): Wavelet to use for the shot record.
        sampling (Sampling): Sampling to use for the shot record.

    Returns:
        ShotRecord: The frequency-domain shot record.
    """

    if record.is_consolidated:
        return read_shot_FD_consolidated(record, wavelet)

    # TODO: This is a hack, fix it.
    cwd = os.getcwd()
    os.chdir(record.project_path)

    fbase = record.file_base
    field = record.field
    isrc = record.source
    recv_group = record.receiver_group
    src_group = record.source_group
    nrecv = recv_group.size
    sampling = record.sampling(upscale=1, t_shift=wavelet.center)
    nf = sampling.nfreq

    spectrum = wavelet.spectrum
    fmap = record.f_map

    u = np.zeros((nf, nrecv), dtype=np.csingle)
    for i, freq in record.f_map.items():
        file = f"{fbase}_{i}.h5"
        ifreq = round(freq / sampling.df)
        omega = np.csingle(2 * np.pi * freq)

        if not os.path.exists(file):
            continue
            # raise FileNotFoundError(f"File {file} does not exist.")
        with h5py.File(file, "r") as f:
            # Read real and imaginary parts in one operation
            im_data = f[f"{field}_{isrc}_im"][()]
            re_data = f[f"{field}_{isrc}_re"][()]
            u[ifreq, :] = re_data + np.csingle(1j) * im_data

            # Check for invalid values
            if np.any(~np.isfinite(u[ifreq, :])) or np.any(np.abs(u[ifreq, :]) > 1e8):
                u[ifreq, :] = 0
                print(f"Invalid values for frequency {freq} Hz")
                continue

            scale = spectrum[ifreq]
            if isinstance(recv_group.device, ReceiverFiber):
                scale *= np.csingle(1j * omega)
            u[ifreq, :] *= scale
    os.chdir(cwd)

    return ShotRecord(
        type="FD",
        number=isrc,
        sampling=sampling,
        source_group=src_group,
        receiver_group=recv_group,
        field=field,
        data=u,
    )


def read_shot_FD_consolidated(record: Record, wavelet: Wavelet) -> ShotRecord:
    """Read a frequency-domain shot record.

    Args:
        record (Record): Record to read.
        wavelet (Wavelet): Wavelet to use for the shot record.
        sampling (Sampling): Sampling to use for the shot record.

    Returns:
        ShotRecord: The frequency-domain shot record.
    """
    # TODO: This is a hack, fix it.
    cwd = os.getcwd()
    os.chdir(record.project_path)

    fbase = record.file_base
    field = record.field
    isrc = record.source
    recv_group = record.receiver_group
    src_group = record.source_group
    sampling = record.sampling(upscale=1)
    nrecv = recv_group.size
    nf = sampling.nfreq

    u = np.zeros((nf, nrecv), dtype=np.csingle)
    data = record.read_consolidated()

    fmap = record.f_map
    spectrum = wavelet.spectrum

    for i, freq in record.f_map.items():
        ifreq = round(freq / sampling.df)
        omega = np.csingle(2 * np.pi * freq)

        scale = spectrum[ifreq]
        if isinstance(recv_group.device, ReceiverFiber):
            scale *= np.csingle(1j * omega)
        u[ifreq, :] = scale * data[i - 1, :]

    return ShotRecord(
        type="FD",
        number=isrc,
        sampling=sampling,
        source_group=src_group,
        receiver_group=recv_group,
        field=field,
        data=u,
    )


def read_frequency(record: Record, ifreq: int) -> np.ndarray:
    """Read a frequency-domain shot record.

    Args:
        record (Record): Record to read.
        ifreq (int): Frequency to read.

    Returns:
        ShotRecord: The frequency-domain shot record.
    """

    fbase = record.file_base
    field = record.field
    isrc = record.source

    if record.is_consolidated:
        u = record.read_consolidated()[ifreq - 1, :]
    else:
        file = f"{fbase}_{ifreq}.h5"

        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} does not exist.")

        with h5py.File(file, "r") as f:
            u = np.csingle(1j) * f[f"{field}_{isrc}_im"][()]
            u += f[f"{field}_{isrc}_re"][()]
    return u


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
