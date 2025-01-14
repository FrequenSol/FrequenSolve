import numpy as np

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
   kind:          Literal["from_file","Ricker","Klauder","Ormsby"]
   samples_out:   np.ndarray
   
   def get(self, i: int):
      raise NotImplementedError("This class must be overwritten by subclasses.")
      
   def __str__(self):
      raise NotImplementedError("This class must be overwritten by subclasses.")


@dataclass
class GeneratedSignature(Signature):
   """
   @class   GeneratedSignature
   @brief   Wrapper object for getting wavelet at each source
   """
   f_pts:   List[float] = field(default_factory=list)
   offset:  int = 0
   sigma:   Optional[float] = None
   
   @classmethod
   def from_block(cls, input: "InputParser", block: "InputBlock") -> "Wavelet":

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
      """
      @brief   Generate wavelet for given source at specified samples
      @param   isrc     Source number
      @return  A new Wavelet instance.
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
   """
   @class   SignatureFromFile
   @brief   Wrapper object for getting wavelet for sources (and adjoint sources)
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
      if not fmt:
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
      except Error as e:
         raise ValueError( "Invalid format specified in signature file. "
                           "Format may be blank, but if specified (for padding, etc.) "
                           "it must be a valid python integer format.\n\n"
                          f"Format '{self.id_format[1]}' extracted from '{file}' raised error: {e}")
                          
      if self.samples is None:
         if self.interval is not None:
            n = len(signal)
            T = (n - 1) * self.interval
            self.samples = np.linspace(0, T, n)
         elif self.file_format == "SEGY":
            # TODO: read_segy_samples(self.file)
            pass
      
   
   def get(self, i: int):
      """
      @brief   Read wavelet from file and evaluate at specified samples
      @param   isrc     Source number
      @return  A new Wavelet instance.
      """
      
      id = self.id_format[1].format(i)
      fname = self.file.replace(self.id_format[0],id)
      return self.get_wavelet(fname, self.samples_out)


   def read(self, fname, **kwargs) -> np.ndarray:
      """
      @brief   Read signal data from file
      @param   fname    Path to file (may include dataset name for HDF5).
      @param   kwargs   Additional arguments (e.g. 'trace' for SEG-Y).
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
      """
      @brief Read data from an HDF5 file/dataset.
      @param fname  Path like 'file.h5:dataset_name'
      """
      import h5py

      if ":" not in key:
         raise ValueError(
            "HDF5 file reader expects 'file.h5:dataset_name'. "
            f"Received: '{key}'."
         )

      h5file, dset = fname.split(":", 1)
      with h5py.File(h5file, "r") as f:
         if dset not in f:
            raise ValueError(f"Dataset '{dset}' not found in HDF5 file '{h5file}'.")
         return f[dset][()]


   def read_segy(self, fname: str, trace: int = 0) -> np.ndarray:
      """
      @brief Read data from a SEG-Y file.
      @param file  Path to SEG-Y file.
      @param trace Trace index to read (if multiple traces).
      """
      # For now, raise an error if this is unimplemented.
      raise NotImplementedError("SEGY reading is not yet implemented.")


   def read_csv(self, fname: str, sep: str = ",") -> np.ndarray:
      """
      @brief Read data from a CSV-like text file.
      @param file Path to file.
      @param sep  Delimiter for CSV data.
      """
      try:
         return np.fromfile(file=fname, dtype=float, sep=sep)
      except OSError as e:
         raise ValueError(f"Failed to read CSV file '{file}': {e}")


   def read_binary(self,
                   fname:   str,
                   dtype:  str = "float",
                   count:  int = -1,
                   offset: int = 0,
                   sep:    str = "") -> np.ndarray:
      """
      @brief Read data from a binary file.
      @param file   Path to binary file.
      @param dtype  Data type (e.g. 'float32', 'int32').
      @param count  Number of items to read (-1 means read all).
      @param offset Byte offset in the file.
      @param sep    Delimiter (if empty, means pure binary).
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
         raise ValueError(f"Failed to read binary file '{file}': {e}")


   def interp_signal(self, signal, interp: str = "cubic") -> "Wavelet":
      """
      @brief Interpolate signal onto a new time grid.
      @param interp  Interpolation method ('linear' or 'cubic').
      @return A Wavelet object with the interpolated signal.
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
      """
      @brief Read a signal from file and interpolate onto a time grid.
      @param file    Path to file (with optional dataset name for HDF5).
      @param times   Target time array.
      @param interp  'linear' or 'cubic'.
      @return        A Wavelet object.
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
      """
      @brief Interpolate y(x) onto x_new using the specified method.
      @param x_new   New sample poitns.
      @param x       Original sample points.
      @param y       Original values.
      @param kind    'linear' or 'cubic'.
      @return        Values at new sample positions
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
