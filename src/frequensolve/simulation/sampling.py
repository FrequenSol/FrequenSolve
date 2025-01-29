import os
import numpy as np

from dataclasses import dataclass
from typing import Optional, List

__all__ = ['Sampling', 'DiscreteSampling', 'UniformSweepSampling']


@dataclass
class Sampling:
   """Base class for sampling parameters."""
   pass


@dataclass
class DiscreteSampling(Sampling):
   """Sampling parameters for a seismic study at discrete frequencies.

   Attributes:
      freq (List[float]): The frequencies.
   """
   f_list: List[float]

   @property
   def nfreq(self):
      return len(self.freqs)
   
   def __dict__(self) -> dict:
      return {
         "f_list": self.f_list,
      }


@dataclass
class UniformSweepSampling(Sampling):
   """Sampling parameters for a seismic study at uniform frequency steps.

   Attributes:
      f_min (float): Minimum frequency (Hz).
      f_max (float): Maximum frequency (Hz).
      df (float):    Frequency spacing (Hz).
      upscale (int): Integer multiple for upscaling the time-sampling rate.
   """
   f_min: float
   f_max: float
   df:    float
   upscale: int = 1
      
   @property
   def T(self):
      return 1/self.df
      
   @property
   def ofreq(self):
      return int(self.f_min/self.df)
      
   @property
   def nfreq(self):
      return int(self.f_max/self.df) + 1
      
   @property
   def ntime(self):
      return int(2*(self.nfreq - 1))
      
   @property
   def t_list(self):
      return np.linspace(0,self.T,self.ntime+1)
      
   @property
   def f_list(self):
      return np.linspace(0,self.f_max,self.nfreq)
      
   @property
   def dt(self):
      return self.T / self.ntime
         
   # Upscaled
   @property
   def nFreq(self):
      return self.upscale * self.nfreq
   
   @property
   def nTime(self):
      return int(2*(self.nFreq - 1))
      
   @property
   def T_list(self):
      return np.linspace(0,self.T,self.nTime+1)
      
   @property
   def dT(self):
     return self.T / self.nTime
     

   def cutoff(self, Tf: Optional[float] = None):
      """Cutoff the time-domain sampling to a specified maximum time.

      Args:
         Tf (float, optional): Maximum time (s). Defaults to None.

      Returns:
         tuple: The number of time samples and the maximum time.
      """
      if Tf:
         Tl  = self.T_list
         nTf = np.searchsorted(Tl, Tf, side='left')
         nTf = np.minimum(nTf,self.nTime)
         return nTf, Tl[nTf]
      else:
         return self.nTime, self.T
      

   def __dict__(self) -> dict:
      return {
         "f_min": self.f_min,
         "f_max": self.f_max,
         "df": self.df,
         "upscale": self.upscale,
      }
   

   @classmethod
   def from_dict(cls, data: dict) -> 'Sampling':
      return cls(f_min = data["f_min"], 
                 f_max = data["f_max"], 
                 df    = data["df"],
                 upscale = data.get("upscale",1))

