import os
import numpy as np

from dataclasses import dataclass
from typing import Optional

__all__ = ['Sampling']

@dataclass
class Sampling:

   """
   @class   Sampling
   @brief   Computes time- and frequency-domain sampling parameters from
            frequency sweep (f_min, f_max, df)
   @details If upscale > 1, Sampling.Times gives upscaled times. Upscaling simply
            zero-pads frequencies, increasing the time-domain sampling rate
   @param   f_min   Minimum frequency (Hz)
   @param   f_min   Maximum frequency (Hz)
   @param   df      Frequency spacing (Hz)
   @param   upscale Increases time-sampling by integer multiple
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
   def times(self):
      return np.linspace(0,self.T,self.ntime+1)
      
   @property
   def freq(self):
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
   def Times(self):
      return np.linspace(0,self.T,self.nTime+1)
      
   @property
   def dT(self):
     return self.T / self.nTime
     
   def cutoff(self, Tf: Optional[float] = None):
      if Tf:
         Tl  = self.Times
         nTf = np.searchsorted(Tl, Tf, side='left')
         nTf = np.minimum(nTf,self.nTime)
         return nTf, Tl[nTf]
      else:
         return self.nTime, self.T

   def __str__(self) -> str:
      out =  "[Study]\n"
      out += "   [ParameterSweep]\n"
      out += "      freq = {" + f"{self.f_min}:{self.f_max}:{self.df}" + "}"
      out += "   []\n"
      out += "[]\n\n"
