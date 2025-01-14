import os, re
import numpy as np

from dataclasses import dataclass, field
from typing import List, Optional

from ..util.input_parser import *  # noqa
from .sources            import *  # noqa
from .receivers          import *  # noqa
from .sampling           import *  # noqa


__all__ = ['Acquisition', 'Shot']

@dataclass
class Shot:
   """
   @class Shot
   @brief Container for storing a shot record, including source, receiver info, and field data.
   @details A Shot may represent frequency-domain (FD) or time-domain (TD) data, along with the
            sampling info. It is aware of the source, receiver configuration, and raw data arrays.

   @param type            "FD" or "TD" indicating frequency- or time-domain data.
   @param number          Shot number (e.g., source index).
   @param samples         A Sampling object describing frequency/time ranges.
   @param source          The associated Source object for this shot.
   @param receiver_group  The associated ReceiverGroup object.
   @param field           Field name (e.g. "pressure", "displacement").
   @param data            The raw data array, shape depends on FD or TD usage.
   """

   type:           str
   number:         int
   samples:        Sampling
   source:         SourceGroup
   receiver_group: ReceiverGroup
   field:          str
   data:           np.ndarray

   def write_segy(self,
                  fname:     str,
                  units_in:  str = "km",
                  units_out: str = "m",
                  **kwargs):
      """
      @brief Write a time-domain shot to a SEGY file.
      @details Uses segyio or a similar library to create a valid SEG-Y file with
               correct geometry headers. This method is only valid if the shot type is "TD".

      @param fname      Output SEGY file name.
      @param units_in   Units of the input coordinates (defaults to "km").
      @param units_out  Units for the output coordinates (defaults to "m"). Must be 'm' or 'ft'.
      @param kwargs     Additional options, such as 'Tf' for cutoff time.
      @throws AssertionError if shot type is not "TD".
      @throws ValueError     if units_out is not "m" or "ft".
      """
      import datetime
      import pint
      from pathlib import Path
      from segy.factory   import SegyFactory
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
      group  = self.receiver_group
      source = self.source
      dim    = len(source.coord)
      
      # Optional cutoff time
      Tf = kwargs.get("Tf", None)
      nTf, Tf = self.samples.cutoff(Tf)  # number of time samples after cutoff
      
      n_traces       = group.size
      n_samples      = nTf
      interval       = int(self.samples.dT * 1e6)  # sample interval in microseconds
      trace_datetime = datetime.datetime.now()
      
      # Build SEG-Y config
      config = {
         "spec":                    get_segy_standard(1.0),
         "samples_per_trace":       n_samples,
         "sample_interval":         interval,
         "trace_sorting_code":      5,  # Common source point
         "measurement_system_code": "meters" if units_out.lower() == "m" else "feet"
      }
      
      factory = SegyFactory(**config)
      txt     = factory.create_textual_header()
      bin_    = factory.create_binary_header()
      headers = factory.create_trace_header_template(size=n_traces)
      samples = factory.create_trace_sample_template(size=n_traces)
                  
      # Populate headers and data
      for itr in range(n_traces):
         headers[itr]["trace_seq_num_reel"] = itr + 1
         headers[itr]["inline"]    = itr + 1
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
         headers[itr]["receiver_group_elevation"] = int(-group.coord[itr, -1] * scale)
         
         # Trace data (slice out the first n_samples from each trace)
         samples[itr] = self.data[:n_samples, itr].copy()
         
      traces = factory.create_traces(samples=samples, headers=headers)
      
      # Write the file
      with Path(fname).open(mode="wb") as f:
         f.write(txt)
         f.write(bin_)
         f.write(traces)


@dataclass
class Acquisition:
   """
   @class Acquisition
   @brief Defines a seismic acquisition setup, including sources, receivers, and sampling.
   @details This class reads the input file to retrieve blocks describing sources,
            receivers, and wavelet signatures. It then aggregates them into a single
            cohesive acquisition definition.

   @param samples          A Sampling object describing frequency/time sampling.
   @param receiver_groups  A list of ReceiverGroup objects (stations, geophones, or fibers).
   @param source_group     A group of source objects describing all shot points.
   """
   
   samples:          Sampling
   source_group:     SourceGroup
   receiver_groups:  List[ReceiverGroup] = field(default_factory=list)


   @classmethod
   def from_file(cls, input_file, **kwargs):
      """
      @brief Class method to build an Acquisition from a given input file.
      @details This method:
               1) Reads the input file (with InputParser).
               2) Extracts frequency sampling parameters (f_min, f_max, df).
               4) Builds sources and receiver groups
      @param input_file A path to the input file or an existing InputParser.
      @param kwargs     Additional arguments, e.g., 'upscale' for Sampling.
      @return An initialized Acquisition object.
      """
      
      input = InputParser.read(input_file)
      f_min, f_max, df = input.sweep_params
      
      dim = input.get_block("Problem").args.get("dimension")
      
      # Set up sampling with optional upscale
      samples = Sampling(f_min, f_max, df, upscale = kwargs.get("upscale"))
         
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
      """
      @brief Add a group of recievers with common kind, frame, and direction.
      """
                        
      isrc = len(self.source_group.sources)
      for row in coords:
         self.source_group.sources.append(
            Source(
               kind      = kind,
               frame     = frame,
               coords    = row,
               direction = direction,
               name = f"source_{isrc}"
            )
         )
         
   def add_reciever_group(self,
                          name:       str,
                          kind:       str,
                          coords:     np.ndarray,
                          components: List[ReceiverComponent],
                          frame:      str = "phyiscal"):
      """
      @brief Add a group of recievers with common kind, frame, and direction.
      """
                        
      isrc = len(self.source_group.sources)
      for row in coords:
         self.reciever_groups.append(
            ReceiverGroup(
               name       = name,
               kind       = kind,
               frame      = frame,
               coords     = coords,
               components = components
            )
         )
      
   def __str__(self) -> str:
      out =  str(self.source_group)
      out += "[Receiver]\n"
      out += str(self.receiver_group)
      out += "[]\n\n"
      return out


   def list_fields(self, recv_name: str = "") -> List[str]:
      """
      @brief List available fields for a specified receiver group or for all groups.
      @details If a receiver group name is provided, only that group is searched. Otherwise,
               all receiver groups are included.
      @param  recv_name  Name of the receiver group (optional).
      @return A list of strings representing the form "groupName:fieldName".
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
      """
      @brief Get a list of source numbers.
      @return A list of integers [1..N] where N is the number of sources.
      """
      return list(range(1, len(self.source_group.sources) + 1))
           
           
   def receiver_group(self, name: str) -> Optional[ReceiverGroup]:
      """
      @brief Retrieve a named receiver group by its block name.
      @param name  The receiver group name.
      @return The matching ReceiverGroup, or None if not found.
      """
      for group in self.receiver_groups:
         if group.name == name:
            return group
      return None
      
      
   def source(self, isrc: int) -> Source:
      """
      @brief Retrieve a source by index.
      @param isrc Source number (1-based).
      @return The Source object at that index.
      @throws IndexError if isrc is out of range.
      """
      return self.source_group.sources[isrc-1]


   def read_shot_FD(self, key: str, isrc: int) -> Shot:
      """
      @brief Read frequency-domain shot data, then apply the wavelet signature.
      @details
        1) Splits `key` into group name and field name (format: "group:field").
        2) Reads the .h5 files for each frequency, applying wavelet spectrum.
        3) Returns a Shot object with 'type' = "FD" and the complex data array.

      @param key   A string like "groupName:fieldName".
      @param isrc  The source number (1-based).
      @return A Shot object containing FD data.
      """
      
      import h5py
      
      of = self.samples.ofreq
      nf = self.samples.nfreq
      f_max = self.samples.f_max
      
      group_name, field = key.split(":")
      group = self.receiver_group(group_name)
      nrecv = group.size
      
      # Retrieve wavelet
      wavelet  = self.source_group.signature(isrc)
      spectrum = wavelet.spectrum
      
      u = np.zeros((nf, nrecv), dtype=np.csingle)
      
      # Loop over frequencies and load data
      for ifreq in range(of, nf):
         file = os.path.join(group.directory, f"{group_name}_{ifreq}.h5")
         i_omega = np.csingle(1j * 2 * np.pi * ifreq / nf * f_max)
         
         if not os.path.exists(file):
            print(f"Warning: {file} does not exist.")
         else:
            with h5py.File(file, "r") as f:
               # Real + imaginary parts
               u[ifreq, :] += np.csingle(1j) * f[f"{field}_{isrc}_im"][()]
               u[ifreq, :] +=              f[f"{field}_{isrc}_re"][()]
               
               # Apply wavelet
               u[ifreq, :] *= spectrum[ifreq]
               
               # For fiber-type receivers, multiply by iω for strain rate
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
      """
      @brief Read time-domain shot data by first reconstructing from the frequency-domain.
      @details
        1) Calls read_shot_FD to load the frequency-domain shot.
        2) Performs an inverse FFT (irfft).
        3) Returns a Shot object with 'type' = "TD" and the real data array.

      @param key   A string like "groupName:fieldName".
      @param isrc  The source number (1-based).
      @return A Shot object containing time-domain data.
      """
      
      try:
         import pyfftw.interfaces.numpy_fft as fft
      except:
         print('pyfftw not found, using numpy for FFT (slow)')
         import numpy.fft as fft
      
      
      group_name, field = key.split(":")
      group = self.receiver_group(group_name)
      nrecv = group.size
   
      nf = self.samples.nfreq
      nF = self.samples.nFreq  # upscaled frequency domain size (if any)
      
      fd = self.read_shot_FD(key, isrc)
      
      # If upscaled, create a bigger array for inverse transform
      if nF > nf:
         FD = np.zeros((nF, nrecv), dtype=np.csingle)
         FD[:nf, :] = fd.data[:nf, :]
         del fd
         td = fft.irfft(FD, axis=0)  # inverse FFT
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
