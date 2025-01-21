"""Python structures defining seismic acquisition geometry and sampling"""

import os
import numpy as np
import warnings

from dataclasses import dataclass, field
from typing import List, Optional, Dict

from ..util.input_parser   import *  # noqa
from .sources              import *  # noqa
from .receivers            import *  # noqa
from ..simulation.sampling import *  # noqa
from ..simulation.config   import *  # noqa
from .shot                 import *  # noqa
from .signals              import *  # noqa

__all__ = ['Acquisition']


@dataclass
class Acquisition:
   """Defines a seismic acquisition setup, including sources, receivers, and sampling.

   This class reads the input file to retrieve blocks describing sources, receivers, and 
   wavelet signatures. It then aggregates them into a single cohesive acquisition definition.

   Attributes:
      samples (Sampling): A Sampling object describing frequency/time sampling.
      source_group (SourceGroup): A group of source objects describing all shot points.
      receiver_groups (List[ReceiverGroup]): A list of ReceiverGroup objects (stations, geophones, or fibers).
   """
   
   samples:          Sampling
   source_group:     SourceGroup
   receiver_groups:  List[ReceiverGroup] = field(default_factory=list)


   def to_dict(self) -> Dict:
      """Converts the Acquisition to a dictionary representation.
      
      Returns:
         Dict: Dictionary containing the Acquisition configuration.
      """
      return {
         "source_group": self.source_group.to_dict(),
         "receiver_groups": [group.to_dict() for group in self.receiver_groups],
      }
   

   @classmethod
   def from_dict(cls, sim: SimulationConfig,dict: Dict) -> 'Acquisition':
      """Creates an Acquisition instance from a dictionary representation.
      
      Args:
         dict (Dict): Dictionary containing the Acquisition configuration.
         
      Returns:
         Acquisition: A new Acquisition instance.
      """
      return cls(
         source_group    = SourceGroup.from_dict(dict["source_group"]),
         receiver_groups = [ReceiverGroup.from_dict(group) for group in dict["receiver_groups"]],
         samples         = sim.sampling
      )
      
   def __str__(self) -> str:
      out =  str(self.source_group)
      out += "[Receiver]\n"
      out += str(self.receiver_group)
      out += "[]\n\n"
      return out


   @classmethod
   def from_file(cls, input_file, **kwargs):
      """Build an Acquisition from a given input file.

      Args:
         input_file (str): Path to the input file or an existing InputParser.
         kwargs (dict): Additional arguments, e.g., 'upscale' for Sampling.

      Returns:
         Acquisition:      A group of sources and corresponding receivers.
      """
      
      input = InputParser.read(input_file)
      f_min, f_max, df = input.sweep_params
      
      dim = input.get_block("Problem").args.get("dimension")
      
      # Set up sampling with optional upscale
      samples = UniformSweepSampling(f_min, f_max, df, upscale = kwargs.get("upscale"))
         
      # Construct source group
      src_block = input.get_block("Source")
      sources   = SourceGroup.from_block(input, src_block)
      
      # Construct receiver groups
      recv_blocks = input.get_block("Receiver").sub_blocks
      receivers   = [ReceiverGroup.from_block(input, block) for block in recv_blocks]

      return cls(
         samples         = samples,
         source_group    = sources,
         receiver_groups = receivers
      )
      

   def add_source_group(self,
                        kind:      str,
                        coords:    np.ndarray,
                        direction: np.ndarray,
                        frame:     str = "phyiscal"):
      """Add a group of recievers with common kind, frame, and direction.
      
      Args:
         kind (str): Kind of the receiver group (e.g., "station", "geophone", "fiber").
         coords (np.ndarray): Coordinates of the receiver group.
         direction (np.ndarray): Direction of the receiver group.
         frame (str): Frame of the receiver group (e.g., "physical", "global").
      """
                        
      for row in coords:
         isrc = len(self.source_group.sources)
         self.source_group.sources.append(
            Source(
               kind        = kind,
               frame       = frame,
               coordinates = row,
               direction   = direction,
               name        = f"source_{isrc}"
            )
         )
         

   def add_reciever_group(self,
                          name:        str,
                          device:      ReceiverDevice,
                          coordinates: np.ndarray,
                          frame:       str = "phyiscal"):
      """Add a group of recievers with common kind, frame, and direction.

      Args:
         name (str):                Name of the receiver group.
         device (ReceiverDevice):   Device defining receiver type and components.
         coordinates (np.ndarray):  Coordinates of the receiver group.
         frame (str): Frame of the receiver group (e.g., "physical", "global").
      """
                        
      self.reciever_groups.append(
         ReceiverGroup(
            name        = name,
            device      = device,
            frame       = frame,
            coordinates = coordinates
         )
      )


   def list_fields(self, recv_name: str = "") -> List[str]:
      """List available fields for a specified receiver group or for all groups.

      If a receiver group name is provided, only that group is searched. Otherwise,
      all receiver groups are included.

      Args:
         recv_name (str): Name of the receiver group (optional).

      Returns:
         List[str]: A list of strings representing the form "groupName:fieldName".
      """
      field_list = []
      
      if recv_name:
         # List fields for one group
         group = self.receiver_group(recv_name)
         for field in group.components:
            file = f"{group.name}:{field.name}"
            field_list.append(file)
      else:
         # List fields for all groups
         for group in self.receiver_groups:
            for field in group.components:
               file = f"{group.name}:{field.name}"
               field_list.append(file)
      return field_list
      
      
   def list_sources(self) -> List[int]:
      """Get a list of source numbers.

      Returns:
         List[int]: A list of integers [1..N] where N is the number of sources.
      """
      return list(range(1, len(self.source_group.sources) + 1))
           
           
   def receiver_group(self, name: str) -> Optional[ReceiverGroup]:
      """Retrieve a named receiver group by its block name.

      Args:
         name (str): The receiver group name.

      Returns:
         ReceiverGroup: The matching ReceiverGroup, or None if not found.
      """
      for group in self.receiver_groups:
         if group.name == name:
            return group
      return None
      
      
   def source(self, isrc: int) -> Source:
      """Retrieve a source by index.

      Args:
         isrc (int): Source number (1-based).

      Returns:
         Source: The Source object at that index.

      Raises:
         IndexError: If isrc is out of range.
      """
      try:
         return self.source_group.sources[isrc-1]
      except IndexError:
         raise IndexError(f"Source index {isrc} is out of range.")


   def read_shot_FD(self, key: str, isrc: int) -> Shot:
      """Read frequency-domain shot data, then apply the wavelet signature.

      Args:
         key (str): A string like "groupName:fieldName".
         isrc (int): The source number (1-based).

      Returns:
         Shot: A Shot object containing FD data.
      """
      
      try:
         import h5py
      except:
         print("h5py not found, skipping frequency-domain data")
         return None
      
      group_name, field = key.split(":")
      group = self.receiver_group(group_name)
      nrecv = group.size

      if isinstance(self.samples, UniformSweepSampling):
         of = self.samples.ofreq
         nf = self.samples.nfreq
         f_max = self.samples.f_max

         wavelet  = self.source_group.signal(isrc)
         spectrum = wavelet.spectrum
      else:
         of = 0
         spectrum = np.ones([self.samples.nfreq])
      
      u = np.zeros((nf, nrecv), dtype=np.csingle)
      
      # Loop over frequencies and load data
      for ifreq, freq in enumerate(self.samples.freqs):
         file = os.path.join(group.directory, f"{group_name}_{ifreq}.h5")
         i_omega = np.csingle(1j * 2 * np.pi * freq)
         
         if ifreq >= of and not os.path.exists(file):
            warnings.warn(f"File {file} does not exist.", UserWarning)
         else:
            with h5py.File(file, "r") as f:
               # Real + imaginary parts
               u[ifreq, :] += np.csingle(1j) * f[f"{field}_{isrc}_im"][()]
               u[ifreq, :] +=              f[f"{field}_{isrc}_re"][()]
               
               # Apply wavelet
               u[ifreq, :] *= spectrum[ifreq]
               
               # For fiber-type receivers, multiply by iω for strain *rate*
               if group.kind == 'fiber':
                  u[ifreq, :] *= i_omega
               
               f.close()
         
      return Shot(type           = "FD",
                  number         = isrc,
                  samples        = self.samples,
                  source         = self.source(isrc),
                  receiver_group = group,
                  field          = field,
                  data           = u)


   def read_shot_TD(self, key: str, isrc: int) -> Shot:
      """Read time-domain shot data by first reconstructing from the frequency-domain.

      Args:
         key (str): A string like "groupName:fieldName".
         isrc (int): The source number (1-based).

      Returns:
         Shot: A Shot object containing time-domain data.
      """
      
      if not isinstance(self.samples,UniformSweepSampling):
         raise ValueError("Time-domain data is only supported for uniform sweep sampling.")
      
      try:
         import pyfftw.interfaces.numpy_fft as fft
      except:
         print('pyfftw not found, using numpy for FFT (slow)')
         import numpy.fft as fft
      
      group_name, field = key.split(":")
      group = self.receiver_group(group_name)
      nrecv = group.size
   
      nf = self.samples.nfreq
      nF = self.samples.nFreq
      
      fd = self.read_shot_FD(key, isrc)
      
      # If upscaled, create a bigger array for inverse transform
      if nF > nf:
         FD = np.zeros((nF, nrecv), dtype=np.csingle)
         FD[:nf, :] = fd.data[:nf, :]
         del fd
         td = fft.irfft(FD, axis=0)
         del FD
      else:
         td = fft.irfft(fd.data, axis=0)
         del fd
         
      return Shot(type           = "TD",
                  number         = isrc,
                  samples        = self.samples,
                  source         = self.source(isrc),
                  receiver_group = group,
                  field          = field,
                  data           = td)
