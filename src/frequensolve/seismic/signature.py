import numpy as np
import re

from dataclasses  import dataclass, field
from typing       import Optional, List, Literal, Tuple

from ..util.input_parser import *  # noqa
from .sampling    import *  # noqa
from .wavelet     import *  # noqa

__all__ = ['Signature', 'GeneratedSignature', 'SignatureFromFile']

# ----------------------------------------------------------------------
# Signature
# ----------------------------------------------------------------------
@dataclass
class Signature:
   """Base class for seismic signatures.

   Attributes:
      kind (Literal["from_file","Ricker","Klauder","Ormsby"]): The type of signature.
      samples_out (np.ndarray): The output time samples.
   """
   kind:          Literal["from_file","Ricker","Klauder","Ormsby"]
   samples_out:   np.ndarray
   
   def get(self, i: int):
      raise NotImplementedError("This class must be overwritten by subclasses.")
      
   def __str__(self):
      raise NotImplementedError("This class must be overwritten by subclasses.")


@dataclass
class GeneratedSignature(Signature):
   """Wrapper object for getting wavelet at each source

   Attributes:
      f_pts (List[float]): The frequencies for the wavelet.
      offset (int): The time offset for the wavelet.
      sigma (Optional[float]): The width of the wavelet taper.
   """
   f_pts:   List[float] = field(default_factory=list)
   offset:  int = 0
   sigma:   Optional[float] = None
   
   @classmethod
   def from_block(cls, input: "InputParser", block: "InputBlock") -> "Wavelet":
      """Create a GeneratedSignature from an input block.

      Args:
         input (InputParser): The input parser.
         block (InputBlock): The input block.

      Returns:
         GeneratedSignature: The generated signature.
      """

      # Get samples from frequency sweep
      f_min, f_max, df = input.sweep_params
      samples = Sampling(f_min, f_max, df)
      times   = samples.times

      kind        = block.args.get("kind")
      f_pts       = str_to_array(block.args.get("f"))
      taper_sigma = float(block.args.get("taper_width"))
      offset      = int(block.args.get("offset", 0))

      assert kind in ["Ricker", "Ormsby", "Klauder"]
      if f_pts is None:
         raise ValueError(
            "Generated wavelets require specified frequencies:\n"
            "  Ricker:  f=[f_central]\n"
            "  Klauder: f=[f1, f2]\n"
            "  Ormsby:  f=[f1, f2, f3, f4]"
         )
         
      return cls(
         samples_out = times,
         kind        = kind,
         f_pts       = f_pts,
         sigma       = taper_sigma,
         offset      = offset
      )
   
   def get(self, i: int):
      """Generate wavelet for given source at specified samples.

      Args:
         i (int): The source number.

      Returns:
         A new Wavelet instance.
      """
      return Wavelet.generate(
         kind    = self.kind,
         f_pts   = self.f_pts,
         times   = self.samples_out,
         offset  = self.offset,
         sigma   = self.sigma
      )
            
   def __str__(self):
      f = " ".join(map(str, self.f_pts))
      out = (
         "   [Signature]\n"
        f"      kind   = {self.kind}\n",
        f"      f      = {f}\n"
        f"      offset = {self.offset}\n"
      )
      if self.sigma:
         out += f"      taper_sigma = {self.sigma}\n"
      out += "   []\n"
      return out
            
            
@dataclass
class SignatureFromFile(Signature):
   """Wrapper object for getting wavelet for sources (and adjoint sources)

   Attributes:
      file_format (str): The format of the file.
      file (str): The path to the file.
      interval (Optional[float]): The interval between samples.
      samples (Optional[np.ndarray]): The samples to use for the wavelet.
   """
   file_format:   str
   file:          str
   interval:      Optional[float]      = None
   # TODO: allow specifying file or numpy array for samples (or specifying trace number in SEGY file)
   samples:       Optional[np.ndarray] = None
   
   # Derived in __post_init__
   id_format:     Tuple[str,str]         = ("","")
   
   
   @classmethod
   def from_block(cls, input: "InputParser", block: "InputBlock") -> "Wavelet":
      """Create a SignatureFromFile from an input block.

      Args:
         input (InputParser): The input parser.
         block (InputBlock): The input block.

      Returns:
         SignatureFromFile: The signature from file.
      """
      # Get samples from frequency sweep
      f_min, f_max, df = input.sweep_params
      samples = Sampling(f_min, f_max, df)
      times   = samples.times
      
      kind     = block.args.get("kind")
      interval = block.args.get("interval")
      format   = block.args.get("file_format")
      file     = block.args.get("file")

      assert kind == "from_file"
      if not file:
         raise ValueError("Wavelet block with kind='file' must specify 'file' path.")
      if not format:
         raise ValueError("Wavelet block with kind='file' must specify 'format'.")
         
      sig = cls(samples_out = times,
                kind        = kind,
                interval    = interval,
                file_format = format,
                file        = file)
      sig.__post_init__()
      return sig
   
   def __post_init__(self):
      match = re.search("(\{)\w([:\w]*\})", self.file)
      self.id_format = (match[0], match[1] + match[2])
      
      try:
         id = self.id_format[1].format(64)
      except BaseException as e:
         raise ValueError( "Invalid format specified in signature file. "
                           "Format may be blank, but if specified (for padding, etc.) "
                           "it must be a valid python integer format.\n\n"
                          f"Format '{self.id_format[1]}' extracted from '{self.file}' raised error: {e}")
                          
      if self.samples is None:
         if self.interval is not None:
            n = len(self.signal)
            T = (n - 1) * self.interval
            self.samples = np.linspace(0, T, n)
         elif self.file_format == "SEGY":
            # TODO: read_segy_samples(self.file)
            pass
      
   
   def get(self, i: int):
      """Read wavelet from file and evaluate at specified samples.

      Args:
         i (int): The source number.

      Returns:
         A new Wavelet instance.
      """
      
      id = self.id_format[1].format(i)
      fname = self.file.replace(self.id_format[0],id)
      return self.get_wavelet(fname, self.samples_out)


   def read(self, fname, **kwargs) -> np.ndarray:
      """Read signal data from file.

      Args:
         fname (str): Path to file (may include dataset name for HDF5).
         kwargs (dict): Additional arguments (e.g. 'trace' for SEG-Y).

      Returns:
         np.ndarray: The signal data.
      """
      if self.format == "HDF5":
         return self.read_hdf5(fname)
      elif self.format == "SEGY":
         trace = kwargs.get("trace", 0)
         return self.read_segy(fname, trace=trace)
      elif self.format == "CSV":
         return self.read_csv(fname,**kwargs)
      elif self.format == "binary":
         return self.read_binary(fname,**kwargs)
      else:
         raise ValueError(f"Unknown format: '{self.format}'. "
                          "Supported formats: HDF5, SEGY, CSV, binary.")


   def read_hdf5(self, fname: str) -> np.ndarray:
      """Read data from an HDF5 file/dataset.

      Args:
         fname (str): Path like 'file.h5:dataset_name'.

      Returns:
         np.ndarray: The signal data.
      """
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


   def read_segy(self, fname: str, trace: int = 0) -> np.ndarray:
      """Read data from a SEG-Y file.

      Args:
         file (str): Path to SEG-Y file.
         trace (int): Trace index to read (if multiple traces).
      """
      # For now, raise an error if this is unimplemented.
      raise NotImplementedError("SEGY reading is not yet implemented.")


   def read_csv(self, fname: str, sep: str = ",") -> np.ndarray:
      """Read data from a CSV-like text file.

      Args:
         file (str): Path to file.
         sep (str): Delimiter for CSV data.
      """
      try:
         return np.fromfile(file=fname, dtype=float, sep=sep)
      except OSError as e:
         raise ValueError(f"Failed to read CSV file '{self.file}': {e}")


   def read_binary(self,
                   fname:  str,
                   dtype:  str = "float",
                   count:  int = -1,
                   offset: int = 0,
                   sep:    str = "") -> np.ndarray:
      """Read data from a binary file.

      Args:
         file (str): Path to binary file.
         dtype (str): Data type (e.g. 'float32', 'int32').
         count (int): Number of items to read (-1 means read all).
         offset (int): Byte offset in the file.
         sep (str): Delimiter (if empty, means pure binary).
      """
      try:
         return np.fromfile(
            file   = fname,
            dtype  = np.dtype(dtype),
            count  = count,
            sep    = sep,
            offset = offset
         )
      except OSError as e:
         raise ValueError(f"Failed to read binary file '{self.file}': {e}")


   def interp_signal(self, signal, interp: str = "cubic") -> "Wavelet":
      """Interpolate signal onto a new time grid.

      Args:
         interp (str): Interpolation method ('linear' or 'cubic').

      Returns:
         A Wavelet object with the interpolated signal.
      """

      if self.samples is None:
         raise ValueError("No sampling specified. Signature should be "
                          "initialized with either 'samples' or 'interval'.")

      sig_interp = self.interpolate(self.samples_out, self.samples, signal, interp)
      return Wavelet(self.samples_out, sig_interp)


   def get_wavelet(self,
                   file:   str,
                   interp: str = "cubic",
                   **kwargs) -> "Wavelet":
      """Read a signal from file and interpolate onto a time grid.

      Args:
         file (str): Path to file (with optional dataset name for HDF5).
         times (np.ndarray): Target time array.
         interp (str): Interpolation method ('linear' or 'cubic').

      Returns:
         A Wavelet object.
      """
      if not file:
         raise ValueError("No file path provided to SignatureReader.get_wavelet.")
      signal = self.read(file, **kwargs)
      return self.interp_signal(signal, interp)


   @staticmethod
   def interpolate(x_new:  np.ndarray,
                   x:      np.ndarray,
                   y:      np.ndarray,
                   kind:   str) -> np.ndarray:
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
         raise ValueError(f"Unsupported interpolation kind: '{kind}'. "
                          "Use 'linear' or 'cubic'.")

   def __str__(self):
      f = " ".join(map(str, self.f_pts))
      out = (
         "   [Signature]\n"
        f"      kind        = from_file\n",
        f"      file_format = {self.file_format}\n"
        f"      file        = {self.file}\n"
      )
      if self.interval:
         out += f"      interval = {self.interval}\n"
      out += "   []\n"
      return out
