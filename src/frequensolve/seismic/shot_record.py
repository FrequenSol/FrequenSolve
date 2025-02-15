import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np

from frequensolve.seismic.receivers import ReceiverFiber, ReceiverGroup
from frequensolve.seismic.sources import Source
from frequensolve.seismic.wavelet import Wavelet
from frequensolve.simulation.sampling import Sampling, UniformSweepSampling

__all__ = ["ShotRecord", "read_shot_TD", "read_shot_FD", "Record"]


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
    source: Source
    receiver_group: ReceiverGroup
    field: str
    data: np.ndarray

    def write_segy(
        self, fname: str, units_in: str = "km", units_out: str = "m", **kwargs
    ):
        """Write a time-domain shot to a SEGY file.

        Uses SEGY to create a valid SEG-Y file with
        correct geometry headers. This method is only valid if the shot type is "TD".

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
        dim = len(source.coord)

        # Optional cutoff time
        Tf = kwargs.get("Tf", None)
        nTf, Tf = self.sampling.cutoff(Tf)  # number of time samples after cutoff

        n_traces = group.size
        n_samples = nTf
        interval = int(self.sampling.dT * 1e6)  # sample interval in microseconds
        trace_datetime = datetime.datetime.now()

        # Build SEG-Y config
        config = {
            "spec": get_segy_standard(1.0),
            "samples_per_trace": n_samples,
            "sample_interval": interval,
            "trace_sorting_code": 5,  # Common source point
            "measurement_system_code": "meters" if units_out.lower() == "m" else "feet",
        }

        factory = SegyFactory(**config)
        txt = factory.create_textual_header()
        bin_ = factory.create_binary_header()
        headers = factory.create_trace_header_template(size=n_traces)
        samples = factory.create_trace_sample_template(size=n_traces)

        # Populate headers and data
        for itr in range(n_traces):
            headers[itr]["trace_seq_num_reel"] = itr + 1
            headers[itr]["inline"] = itr + 1
            headers[itr]["crossline"] = 1

            # Source position
            headers[itr]["source_coord_x"] = int(source.coord[0] * scale)
            if dim == 2:
                headers[itr]["source_coord_y"] = 0
            else:
                headers[itr]["source_coord_y"] = int(source.coord[1] * scale)
            headers[itr]["source_surface_elevation"] = int(-source.coord[-1] * scale)

            # Receiver position
            headers[itr]["group_coord_x"] = int(group.coord[itr, 0] * scale)
            if dim == 2:
                headers[itr]["group_coord_y"] = 0
            else:
                headers[itr]["group_coord_y"] = int(group.coord[itr, 1] * scale)
            headers[itr]["receiver_group_elevation"] = int(
                -group.coord[itr, -1] * scale
            )

            # Trace data (slice out the first n_samples from each trace)
            samples[itr] = self.data[:n_samples, itr].copy()

        traces = factory.create_traces(samples=samples, headers=headers)

        # Write the file
        with Path(fname).open(mode="wb") as f:
            f.write(txt)
            f.write(bin_)
            f.write(traces)


@dataclass
class Record:
    file: str
    project_path: Path  # TODO: this is a hack, fix it.
    simulation: str
    df: float
    f_max: float
    f_map: Dict[str, float]

    def sampling(self, upscale: int = 1) -> Sampling:
        return UniformSweepSampling(
            f_min=0.0, f_max=self.f_max, df=self.df, upscale=upscale
        )

    def times(self, upscale: int = 1) -> np.ndarray:
        return self.sampling(upscale).t_list

    def __init__(self, record: str, meta: Dict[str, Any]):
        self.file = record
        self.project_path = meta["project"]
        self.simulation = Path(meta["simulation"]).with_suffix(".json")
        self.df = meta["df"]
        self.f_max = meta["f_max"]
        self.f_map = meta["f_map"]

    def read_TD(self, wavelet: Wavelet, upscale: int = 1) -> ShotRecord:
        return read_shot_TD(self, wavelet, upscale)

    def read_FD(self, wavelet: Wavelet) -> ShotRecord:
        return read_shot_FD(self, wavelet)


# TODO: Upscale needs to be consistent in different places, make it so it's only specified once.
def read_shot_TD(record: Record, wavelet: Wavelet, upscale: int = 1) -> ShotRecord:
    """Read a time-domain shot record.

    Args:
        record (Record): Record to read.
        wavelet (Wavelet): Wavelet to use for the shot record.
        upscale (int): Upscale factor for the shot record.

    Returns:
        ShotRecord: The time-domain shot record.
    """
    with open(record.simulation, "r") as f:
        sim = json.load(f)

    fbase, comp = record.file.split(":")
    group_name = "_".join(Path(fbase).name.split("_")[:-1])
    field = "_".join(comp.split("_")[:-1])
    isrc = int(comp.split("_")[-1])

    for rgroup in sim["Acquisition"]["receiver_groups"]:
        if rgroup["name"] == group_name:
            break
    else:
        raise ValueError(f"Receiver group {group_name} not found in simulation.")

    source = sim["Acquisition"]["source_group"]["sources"][isrc - 1]

    # TODO: This is a hack, fix it.
    cwd = os.getcwd()
    os.chdir(record.project_path)

    group = ReceiverGroup.from_dict(rgroup)
    source = Source.from_dict(source)

    # Get sampling from metadata
    sampling = record.sampling(upscale)

    # Try to read frequency domain data first
    try:
        fd_record = read_shot_FD(record, wavelet)
    except Exception as e:
        raise ValueError(f"failed reading record: {e}")

    # TODO: This is a hack.
    os.chdir(cwd)

    # Convert to time domain using FFT
    try:
        import pyfftw.interfaces.numpy_fft as fft
    except:
        warnings.warn("pyfftw not found, using numpy for FFT (slow)")
        import numpy.fft as fft

    nf = sampling.nfreq
    nF = sampling.nFreq

    # If upscaled, create a bigger array for inverse transform
    if nF > nf:
        FD = np.zeros((nF, group.size), dtype=np.csingle)
        FD[:nf, :] = fd_record.data[:nf, :]
        td = fft.irfft(FD, axis=0)
    else:
        td = fft.irfft(fd_record.data, axis=0)

    return ShotRecord(
        type="TD",
        number=isrc,
        sampling=sampling,
        source=source,
        receiver_group=group,
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
    # TODO: This is a hack, fix it.
    cwd = os.getcwd()
    os.chdir(record.project_path)

    with open(record.simulation, "r") as f:
        sim = json.load(f)

    fbase, comp = record.file.split(":")
    fbase = "_".join(fbase.split("_")[:-1])
    group_name = Path(fbase).name
    field = "_".join(comp.split("_")[:-1])
    isrc = int(comp.split("_")[-1])

    for rgroup in sim["Acquisition"]["receiver_groups"]:
        if rgroup["name"] == group_name:
            break
    else:
        raise ValueError(f"Receiver group {group_name} not found in simulation.")
    source = sim["Acquisition"]["source_group"]["sources"][isrc - 1]

    group = ReceiverGroup.from_dict(rgroup)
    source = Source.from_dict(source)

    sampling = record.sampling(upscale=1)

    nrecv = group.size
    nf = sampling.nfreq

    # Initialize complex data array
    u = np.zeros((nf, nrecv), dtype=np.csingle)

    fmap = record.f_map

    for i, freq in record.f_map.items():
        file = f"{fbase}_{i}.h5"
        ifreq = round(freq / sampling.df)

        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} does not exist.")

        with h5py.File(file, "r") as f:
            u[ifreq, :] += np.csingle(1j) * f[f"{field}_{isrc}_im"][()]
            u[ifreq, :] += f[f"{field}_{isrc}_re"][()]

            u[ifreq, :] *= wavelet.spectrum[ifreq]

            # For fiber-type receivers, multiply by iω for strain *rate*
            if isinstance(group.device, ReceiverFiber):
                i_omega = np.csingle(1j * 2 * np.pi * freq)
                u[ifreq, :] *= i_omega

    # TODO: This is a hack.
    os.chdir(cwd)

    return ShotRecord(
        type="FD",
        number=isrc,
        sampling=sampling,
        source=source,
        receiver_group=group,
        field=field,
        data=u,
    )
