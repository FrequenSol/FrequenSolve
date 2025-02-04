import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from ..simulation.config import *  # noqa
from ..simulation.sampling import *  # noqa
from .wavelet import *  # noqa

__all__ = ["Signal", "AnalyticalSignal", "SignalFromFile"]


# ----------------------------------------------------------------------
# Signal
# ----------------------------------------------------------------------
@dataclass(kw_only=True)
class Signal:
    """Base class for seismic signals.

    Attributes:
       kind (Literal["from_file","analytical"]): The type of signals.
       samples_out (np.ndarray): The output time samples.
    """

    domain_out: Literal["time", "frequency"]
    samples_out: np.ndarray
    phase: Optional[Literal["causal", "zero"]] = None

    def get(self, i: int):
        raise NotImplementedError("This class must be overwritten by subclasses.")

    def __dict__(self):
        raise NotImplementedError("This class must be overwritten by subclasses.")

    @classmethod
    def from_dict(cls, sim: SimulationConfig, data: Dict):
        raise NotImplementedError("This class must be overwritten by subclasses.")


@dataclass(kw_only=True)
class AnalyticalSignal(Signal):
    """Wrapper object for getting analytical wavelet at each source

    Attributes:
       f_pts (List[float]): The frequencies for the wavelet.
       offset (int): The time offset for the wavelet.
       sigma (Optional[float]): The width of the wavelet taper.
    """

    kind: Literal["Ricker", "Klauder", "Ormsby"]
    type: Literal["analytical"] = "analytical"
    f_pts: List[float] = field(default_factory=list)
    offset: int = 0
    sigma: Optional[float] = None

    @classmethod
    def from_dict(cls, sim: SimulationConfig, data: Dict) -> "AnalyticalSignal":
        kind = data["kind"]
        f_pts = data["f_pts"]

        domain = sim.tf_domain
        if domain == "time":
            samples = sim.samples.t_list
        else:
            samples = sim.samples.frequencies

        assert kind in ["Ricker", "Ormsby", "Klauder"]
        if f_pts is None:
            raise ValueError(
                "Generated wavelets require specified frequencies:\n"
                "  Ricker:  f=[f_central]\n"
                "  Klauder: f=[f1, f2]\n"
                "  Ormsby:  f=[f1, f2, f3, f4]"
            )

        return cls(
            domain_out=domain,
            samples_out=samples,
            kind=kind,
            f_pts=f_pts,
            sigma=data["sigma"] if "sigma" in data else None,
            offset=data["offset"] if "offset" in data else 0,
            phase=data.get("phase"),
        )

    def __dict__(self):
        return {
            "type": self.type,
            "kind": self.kind,
            "f_pts": self.f_pts,
            **({"sigma": self.sigma} if self.sigma is not None else {}),
            **({"offset": self.offset} if self.offset > 0 else {}),
            **({"phase": self.phase} if self.phase is not None else {}),
        }

    def get(self, i: int):
        """Generate signal wavelet for given source at specified samples."""
        return Wavelet.generate(
            kind=self.kind,
            f_pts=self.f_pts,
            times=self.samples_out,
            offset=self.offset,
            sigma=self.sigma,
            phase=self.phase,
        )


@dataclass(kw_only=True)
class SignalFromFile(Signal):
    """Wrapper object for getting signal for sources (and adjoint sources)

    Attributes:
       file_format (str): The format of the file.
       file (str): The path to the file.
       interval (Optional[float]): The interval between samples.
       samples (Optional[np.ndarray]): The samples to use for the signal.
    """

    type: Literal["from_file"] = "from_file"
    file_format: Literal["HDF5", "SEGY"]
    file: Union[str, Path]
    interval: Optional[float] = None
    domain_in: Literal["time", "frequency"]
    samples_in: Optional[np.ndarray] = None
    id_format: Tuple[str, str] = ("", "")

    def __dict__(self) -> Dict:
        return {
            "file": str(self.file),
            "file_format": self.file_format,
            "domain": self.domain_out,
            **(
                {"samples": self.samples_out.tolist()}
                if self.samples_out is not None
                else {}
            ),
            **({"interval": self.interval} if self.interval is not None else {}),
            **({"phase": self.phase} if self.phase else {}),
        }

    def __post_init__(self):
        match = re.search("(\{)\w([:\w]*\})", self.file)
        self.id_format = (match[0], match[1] + match[2])

        try:
            id = self.id_format[1].format(64)
        except BaseException as e:
            raise ValueError(
                "Invalid format specified in signal file. "
                "Format may be blank, but if specified (for padding, etc.) "
                "it must be a valid python integer format.\n\n"
                f"Format '{self.id_format[1]}' extracted from '{self.file}' raised error: {e}"
            )

        if self.samples_in is None:
            if self.interval is not None:
                n = len(self.signal)
                T = (n - 1) * self.interval
                self.samples_in = np.linspace(0, T, n)
            elif self.file_format == "SEGY":
                import segyio

                with segyio.open(self.file, ignore_geometry=True) as f:
                    self.samples_in = f.samples
            else:
                raise ValueError(
                    "Input sampling could not be determined from file and"
                    "an explicit 'interval' was not specified."
                )

    def get(self, i: int):
        """Read Signal from file and evaluate at specified samples."""
        id = self.id_format[1].format(i)
        fname = self.file.replace(self.id_format[0], id)
        return self._get_wavelet(fname, self.samples_out)

    def _read(self, fname: Union[str, Path]) -> np.ndarray:
        """Read signal data from file."""
        if self.file_format == "HDF5":
            return self._read_hdf5(fname)
        elif self.file_format == "SEGY":
            return self._read_segy(fname)
        else:
            raise ValueError(
                f"Unknown format: '{self.file_format}'. "
                "Supported formats: HDF5, SEGY."
            )

    def _read_hdf5(self, fname: Union[str, Path]) -> np.ndarray:
        """Read data from an HDF5 file/dataset."""
        import h5py

        if ":" not in fname:
            raise ValueError(
                "HDF5 file reader expects 'file.h5:dataset_name'. "
                f"Received: '{fname}'."
            )

        h5file, dset = fname.split(":", 1)
        with h5py.File(h5file, "r") as f:
            if dset not in f:
                raise ValueError(f"Dataset '{dset}' not found in HDF5 file '{h5file}'.")
            return f[dset][()]

    def _read_segy(self, fname: Union[str, Path]) -> np.ndarray:
        """Read data from a SEG-Y file."""
        import segyio

        trace = int(fname.split(":")[1])
        with segyio.open(fname, ignore_geometry=True) as f:
            return f.trace[trace]

    def _get_wavelet(
        self, file: Union[str, Path], interp: str = "cubic", **kwargs
    ) -> "Wavelet":
        """Read a signal from file and interpolate onto a time grid.

        Args:
           file (str): Path to file (with optional dataset name for HDF5).
           times (np.ndarray): Target time array.
           interp (str): Interpolation method ('linear' or 'cubic').

        Returns:
           A Wavelet object.
        """
        if not file:
            raise ValueError("No file path provided to SignalFromFile.get_wavelet.")
        signal = self._read(file, **kwargs)
        return self._interp_signal(signal, interp)

    def _interp_signal(self, signal, interp: str = "cubic") -> "Wavelet":
        """Interpolate signal onto a new time grid.

        Args:
           interp (str): Interpolation method ('linear' or 'cubic').

        Returns:
           A Wavelet object with the interpolated signal.
        """

        if self.samples is None:
            raise ValueError(
                "No sampling specified. signal should be "
                "initialized with either 'samples' or 'interval'."
            )

        sig_interp = self._interpolate(self.samples_out, self.samples, signal, interp)
        return Wavelet(self.samples_out, sig_interp)

    @staticmethod
    def _interpolate(
        x_new: np.ndarray, x: np.ndarray, y: np.ndarray, kind: str
    ) -> np.ndarray:
        """Interpolate y(x) onto x_new using the specified method.

        Args:
           x_new (np.ndarray): New sample poitns.
           x (np.ndarray): Original sample points.
           y (np.ndarray): Original values.
           kind (str): Interpolation method ('linear' or 'cubic').

        Returns:
           np.ndarray: Values at new sample positions
        """
        if kind == "linear":
            return np.interp(x_new, x, y, left=0.0, right=0.0)
        elif kind == "cubic":
            from scipy.interpolate import CubicSpline

            spl = CubicSpline(x, y)
            return spl(x_new)
        else:
            raise ValueError(
                f"Unsupported interpolation kind: '{kind}'. " "Use 'linear' or 'cubic'."
            )
