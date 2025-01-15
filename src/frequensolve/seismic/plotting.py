import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .sources     import *    # noqa
from .receivers   import *    # noqa
from .acquisition import Shot # noqa

__all__ = [
   'plot_gather','animate_gather','plot_gather_diff','plot_timelag',
   'plot_xf','plot_cf'
]

try:
   plt.rcParams["font.family"]     = "sans-serif"
   plt.rcParams["font.sans-serif"] = ["Helvetica"]
except:
   pass


#--------------------------------------------
# Time-domain Plot Functions
#--------------------------------------------
def plot_gather(shot: Shot, **kwargs):
   """Plot a 2D shot gather for a time-domain shot.
   
   Renders a simple 2D image of amplitude over (receiver X-position) vs. time.
   Useful for visualizing wavefield arrivals at multiple receivers. The plot shows
   amplitude variations using a specified colormap, with time increasing downward
   and receiver position along the horizontal axis.
   
   Args:
      shot (Shot): A Shot object of type='TD' containing time-domain data.
   
   Keyword Args:
      A (float):       Amplitude scaling for display (default 1).
      units (str):     Length units for labeling X axis (default "km").
      cmap (str):      Matplotlib colormap name (default "grey").
      figsize (tuple): Figure size (width, height) (default (8,8)).
      fontsize (int):  Font size for labels/ticks (default 14).
      Tf (float):      Cutoff time in seconds. If None, uses full time range.
      save (str):      If provided, saves the figure to this path.
   
   Raises:
      AssertionError: If shot is not a time-domain shot (type != "TD").
   """
   assert shot.type == "TD"
   
   A        = kwargs.get("A", 1)
   units    = kwargs.get("units", "km")
   cmap     = kwargs.get("cmap", "grey")
   figsize  = kwargs.get("figsize", (8,8))
   fontsize = kwargs.get("fontsize", 14)
   
   plt.rcParams.update({'font.size': fontsize})
   
   Tf = kwargs.get("Tf", None)
   nTf, Tf = shot.samples.cutoff(Tf)
   
   source = shot.source
   group  = shot.receiver_group
   
   x0 = np.min(group.coords[:,0])
   x1 = np.max(group.coords[:,0])
   xlabel = f"X ({units})"
   if x0 == x1:
      x0 = np.min(group.coords[:,1])
      x1 = np.max(group.coords[:,1])
      xlabel = f"Depth ({units})"
   
   fig = plt.figure(1, figsize=figsize)
   plt.clf()
   
   plt.xlabel(xlabel)
   plt.ylabel("Time (s)")
   
   # Plot gather as an image
   plt.imshow(
      shot.data[:nTf,:],
      origin='upper',
      cmap=cmap,
      extent=[x0, x1, Tf, 0],
      vmin=-A,
      vmax= A,
      aspect='auto'
   )
   plt.tight_layout()
   
   if "save" in kwargs:
      file = kwargs["save"]
      plt.savefig(file, bbox_inches='tight')
      plt.close()
   else:
      plt.show()


def animate_gather(shot: Shot, **kwargs):
   """Animate a shot gather on a 2D grid (time-domain).
    
   This function assumes the receiver group data can be reshaped into a 2D grid
   (e.g. for snapshot-like visualization in time).

   Args:
      shot (Shot): A Shot object of type='TD'.

   Keyword Args:
      A (float): Amplitude scaling for display (default 1).
      cmap (str): Matplotlib colormap (default "RdGy").
      interval (int): Animation interval in milliseconds (default 50).
      units (str): Length units for labeling X, Z axes (default "km").
      figsize (tuple): Figure size (width, height).
      fontsize (int): Font size for labels/ticks.
      Tf (float): Cutoff time in seconds.
      save (str): If provided, saves an .mp4 to this path (without extension).

   Raises:
      ValueError: If the associated receiver group is not grid-based.
   """
   assert shot.type == "TD"

   def plot_frame(u, i, fig, frames):
      frames.append([
         plt.imshow(
            u[i,:,:].transpose(),
            origin='upper',
            cmap=cmap,
            vmin=-A,
            vmax= A,
            extent=[x0[0], x1[0], x1[1], x0[1]]
         )
      ])

   Tf = kwargs.get("Tf", None)
   nTf, Tf = shot.samples.cutoff(Tf)
   
   A        = kwargs.get("A", 1)
   units    = kwargs.get("units", "km")
   cmap     = kwargs.get("cmap", "RdGy")
   interval = kwargs.get("interval", 50)
   figsize  = kwargs.get("figsize", (8,8))
   fontsize = kwargs.get("fontsize", 10)
   
   plt.rcParams.update({'font.size': fontsize})
   
   group = shot.receiver_group

   try:
      x0 = group.grid.x0
      x1 = group.grid.x1
      n  = group.grid.n
      shot.data = shot.data[:nTf,:].reshape((nTf, n[0], n[1]))
   except OSError as e:
      raise ValueError("animate_gather currently only works with 'grid' receivers") from e

   frames = []
   fig = plt.figure(1, figsize=figsize)
   
   plt.xlabel(f"X ({units})")
   plt.ylabel(f"Z ({units})")
   
   ax = plt.gca()
   ax.set_axis_off()
   
   for i in range(nTf):
      plot_frame(shot.data, i, fig, frames)
      
   ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True)
   if "save" in kwargs:
      file = kwargs["save"]
      if file.endswith(".mp4"):
         file = file.replace(".mp4", "")
      ani.save(f"{file}.mp4")
   else:
      plt.show()
      
   del(frames, fig)


def plot_gather_diff(shot1: Shot, shot2: Shot, **kwargs):
   """Plot the difference between two time-domain shot gathers side-by-side.
   
   Shows the "baseline", "perturbed", and "difference" in a 3-panel figure.

   Args:
      shot1 (Shot): First TD Shot (baseline).
      shot2 (Shot): Second TD Shot (perturbed).

   Keyword Args:
      A (float): Amplitude scaling for display.
      units (str): Length units for labeling X axis (default "km").
      cmap (str): Matplotlib colormap (default "grey").
      figsize (tuple): Figure size (width, height).
      fontsize (int): Font size for labels/ticks.
      Tf (float): Cutoff time in seconds.
      save (str): If provided, saves figure to this path.

   Raises:
      AssertionError: If shots are not time-domain or time lengths differ.
   """
   assert shot1.type == "TD" and shot2.type == "TD"
   assert shot1.nTime == shot2.nTime and shot1.T == shot2.T
   
   A        = kwargs.get("A", 1)
   units    = kwargs.get("units","km")
   cmap     = kwargs.get("cmap","grey")
   figsize  = kwargs.get("figsize",(15,4))
   fontsize = kwargs.get("fontsize",12)
   
   plt.rcParams.update({'font.size': fontsize})
   
   Tf = kwargs.get("Tf", None)
   nTf, Tf = shot1.samples.cutoff(Tf)
   
   source = shot1.source
   group  = shot1.receiver_group
   
   x0 = np.min(group.coords[:,0])
   x1 = np.max(group.coords[:,0])
   xlabel = f"X ({units})"
   if x0 == x1:
      x0 = np.min(group.coords[:,1])
      x1 = np.max(group.coords[:,1])
      xlabel = f"Depth ({units})"
   
   fig = plt.figure(1, figsize=figsize)
   plt.clf()
   
   gs = fig.add_gridspec(1,34)
   ax1 = fig.add_subplot(gs[0,0:10])
   ax2 = fig.add_subplot(gs[0,12:22])
   ax3 = fig.add_subplot(gs[0,24:34])
   
   ax1.set_xlabel(xlabel)
   ax2.set_xlabel(xlabel)
   ax3.set_xlabel(xlabel)
   ax1.set_ylabel('Time (s)')
   ax1.set_title('Baseline')
   ax2.set_title('Perturbed')
   ax3.set_title('Difference')
   
   ax1.imshow(
      shot1.data[:nTf,:],
      origin='upper',
      cmap=cmap,
      extent=[x0, x1, Tf, 0],
      vmin=-A,
      vmax= A,
      aspect='auto'
   )
   ax2.imshow(
      shot2.data[:nTf,:],
      origin='upper',
      cmap=cmap,
      extent=[x0, x1, Tf, 0],
      vmin=-A,
      vmax= A,
      aspect='auto'
   )
   ax3.imshow(
      shot1.data[:nTf,:] - shot2.data[:nTf,:],
      origin='upper',
      cmap=cmap,
      extent=[x0, x1, Tf, 0],
      vmin=-A,
      vmax= A,
      aspect='auto'
   )
   
   if "save" in kwargs:
      file = kwargs["save"]
      plt.savefig(file, bbox_inches='tight')
      plt.close()
   else:
      plt.show()
   del(fig)


def window_first_arrival(signal, signal2, sampling_rate,
                     threshold_ratio, window_length,
                     window_type='tukey', alpha=0.5):
   """Window and extract first arrivals from a pair of seismic signals.
   
   Uses the Hilbert transform to identify first arrivals and applies a smooth window
   around them. This is useful for isolating primary arrivals before cross-correlation
   or other analysis.
   
   Args:
      signal (np.ndarray): Primary time series to analyze for first arrival.
      signal2 (np.ndarray): Secondary time series to window (same length as signal).
      sampling_rate (float): Sampling rate in samples per second.
      threshold_ratio (float): Fraction of peak envelope amplitude for first-arrival detection.
      window_length (float): Total length of the time window in seconds.
      window_type (str, optional): Type of window to apply ('tukey' or 'gaussian'). 
         Defaults to 'tukey'.
      alpha (float, optional): Shape parameter for Tukey window. Defaults to 0.5.
         Only used if window_type='tukey'.
   
   Returns:
      tuple[np.ndarray, np.ndarray]: A tuple containing:
         - smoothed1: The windowed primary signal
         - smoothed2: The windowed secondary signal
         
   Raises:
      ValueError: If window_type is not 'tukey' or 'gaussian'.
      ValueError: If signals have different lengths.
   """
   from scipy.signal import hilbert, tukey
   
   # Step 1: Envelope
   analytic_signal = hilbert(signal)
   envelope = np.abs(analytic_signal)
   
   # Step 2: Find first arrival index
   threshold = threshold_ratio * np.max(envelope)
   ifirst    = np.where(envelope > threshold)[0][0]
   
   # Step 3: Window length in samples
   hwl = int(window_length * sampling_rate / 2)
   i1 = max(ifirst - hwl, 0)
   i2 = min(ifirst + hwl, len(signal))
   
   # Step 4: Create window
   window = np.zeros(len(signal))
   if window_type == 'tukey':
      window_section = tukey(i2 - i1, alpha=alpha)
   else:
      # Gaussian
      x = np.linspace(-1, 1, i2 - i1)
      sigma = 0.5
      window_section = np.exp(-0.5 * (x / sigma)**2)
   
   window[i1:i2] = window_section
   
   # Step 5: Apply window
   smoothed1 = signal  * window
   smoothed2 = signal2 * window

   return smoothed1, smoothed2


def plot_timelag(shot1: Shot, shot2: Shot, **kwargs):
   """Plot time lag analysis between two time-domain shots.
   
   Performs cross-correlation between corresponding traces in two shots to estimate
   time shifts. Useful for analyzing velocity perturbations or timing differences
   between baseline and monitor surveys.
   
   Args:
      shot1 (Shot): First TD Shot (baseline).
      shot2 (Shot): Second TD Shot (monitor/perturbed).

   Keyword Args:
      A (float): Amplitude scaling for display.
      units (str): Length units for labeling X axis (default "km").
      cmap (str): Matplotlib colormap (default "grey").
      figsize (tuple): Figure size (width, height).
      fontsize (int): Font size for labels/ticks.
      Tf (float): Cutoff time in seconds.
      save (str): If provided, saves figure to this path.
      max_lag (float): Maximum time lag to consider in seconds.
   
   Raises:
      AssertionError: If shots are not time-domain or have different geometries.
      ValueError: If shots have incompatible sampling.
   """
   assert shot1.type == "TD" and shot2.type == "TD"
   assert shot1.nTime == shot2.nTime and shot1.T == shot2.T

   A        = kwargs.get("A", 1)
   units    = kwargs.get("units","km")
   cmap     = kwargs.get("cmap","grey")
   figsize  = kwargs.get("figsize",(8,8))
   fontsize = kwargs.get("fontsize",14)
   
   plt.rcParams.update({'font.size': fontsize})
   
   Tf = kwargs.get("Tf", None)
   nTf, Tf = shot1.samples.cutoff(Tf)
   
   source = shot1.source
   group  = shot1.receiver_group
   
   x0 = np.min(group.coords[:,0])
   x1 = np.max(group.coords[:,0])
   xlabel = f"X ({units})"
   if x0 == x1:
      x0 = np.min(group.coords[:,1])
      x1 = np.max(group.coords[:,1])
      xlabel = f"Depth ({units})"
   
   nT     = shot1.samples.nTime
   n_recv = group.size
   rate   = (nT - 1) / shot1.samples.T  # samples/second

   # Compute lag time per receiver
   lag = np.zeros((n_recv), dtype=np.single)
   for i in range(n_recv):
      tr1, tr2 = window_first_arrival(
         shot1.data[:,i],
         shot2.data[:,i],
         rate, 0.2, 0.01
      )
      cor = np.argmax(np.correlate(tr1, tr2, 'full'))
      cor -= (nT - 1)
      lag[i] = -(cor / rate * 1000)  # ms
   
   x = np.linspace(x0, x1, n_recv)
   
   # Plot
   fig = plt.figure(1, figsize=figsize)
   plt.clf()
   
   plt.plot(x, lag)
   ax = plt.gca()
   ax.set_xlim([x0, x1])
   ax.set_ylim([0, 1.0])  # may adjust for range of lag
   plt.xlabel(xlabel)
   plt.ylabel('Δt (ms)')
   
   if "save" in kwargs:
      file = kwargs["save"]
      plt.savefig(file, bbox_inches='tight')
      plt.close()
   else:
      plt.show()
   del(fig)


#--------------------------------------------
# Frequency-domain Plot Functions
#--------------------------------------------
def plot_xf(shot: Shot, **kwargs):
   """Plot a frequency-domain shot gather.
   
   Creates a 2D plot showing amplitude vs frequency and receiver position.
   Useful for analyzing frequency content at different receiver locations.
   
   Args:
      shot (Shot): A Shot object of type='FD' containing frequency-domain data.

   Keyword Args:
      A (float): Amplitude scaling for display (default 1).
      units (str): Length units for labeling X axis (default "km").
      cmap (str): Matplotlib colormap name (default "grey").
      figsize (tuple): Figure size (width, height) (default (8,8)).
      fontsize (int): Font size for labels/ticks (default 14).
      save (str): If provided, saves the figure to this path.
   
   Raises:
      AssertionError: If shot is not a frequency-domain shot (type != "FD").
   """
   assert shot.type == "FD"

   A        = kwargs.get("A", 1)
   units    = kwargs.get("units","km")
   cmap     = kwargs.get("cmap","grey")
   figsize  = kwargs.get("figsize",(8,8))
   fontsize = kwargs.get("fontsize",14)
   
   plt.rcParams.update({'font.size': fontsize})
   
   f_min = shot.samples.f_min
   f_max = shot.samples.f_max
   nf    = shot.samples.nfreq

   source = shot.source
   group  = shot.receiver_group

   x0 = np.min(group.coords[:,0])
   x1 = np.max(group.coords[:,0])
   xlabel = f"X ({units})"
   if x0 == x1:
      x0 = np.min(group.coords[:,1])
      x1 = np.max(group.coords[:,1])
      xlabel = f"Depth ({units})"

   # Plot
   plt.ylabel("f (Hz)")
   plt.imshow(
      shot.data[:,:].real,
      origin='lower',
      cmap=cmap,
      extent=[x0, x1, 0, f_max],
      aspect='auto'
   )
   
   if "save" in kwargs:
      file = kwargs["save"]
      plt.savefig(file, bbox_inches='tight')
   else:
      plt.show()


def plot_cf(shot: Shot, **kwargs):
   """Create a CF (phase velocity vs. frequency) plot using a smooth-windowed Radon transform.
   
   For a frequency-domain shot, estimates wave speed distribution by testing different 
   velocities (c) for a given frequency (f). Uses a windowed Radon transform approach
   to identify dominant wave speeds at each frequency.
   
   Args:
      shot (Shot): A frequency-domain Shot object.

   Keyword Args:
      A (float): Amplitude scaling for color max (default 1).
      units (str): Label for distance ("km" or "m").
      cmap (str): Matplotlib colormap (default "RdYlBu_r").
      figsize (tuple): Figure size (width, height) (default (8,8)).
      fontsize (int): Font size for labels/ticks (default 14).
      symm (bool): If True, uses symmetrical approach for half of the array.
      c_min (float): Minimum wave speed for transform (default 0.5).
      c_max (float): Maximum wave speed for transform (default 6.0).
      n_c (int): Number of wave speed samples (default 500).
      save (str): File path to save figure.
   
   Raises:
      AssertionError: If shot is not a frequency-domain shot (type != "FD").
      ValueError: If receiver geometry is incompatible with transform.
   """
   assert shot.type == "FD"
   from scipy.ndimage import gaussian_filter1d
   
   A        = kwargs.get("A", 1)
   units    = kwargs.get("units", "km")
   cmap     = kwargs.get("cmap", "RdYlBu_r")
   figsize  = kwargs.get("figsize", (8,8))
   fontsize = kwargs.get("fontsize", 14)
   
   symm  = kwargs.get("symm", False)
   c_min = kwargs.get("c_min", 0.5)
   c_max = kwargs.get("c_max", 6.0)
   n_c   = kwargs.get("n_c", 500)
   
   source = shot.source
   group  = shot.receiver_group
   
   x0 = np.min(group.coords[:,0])
   x1 = np.max(group.coords[:,0])
   
   # If symmetrical, consider half the domain
   if symm:
      n1 = group.size // 2
      xl = group.coords[:n1:-1,0] - source.coords[0]
      shot.data = shot.data[:,:n1:-1]
   else:
      n1 = group.size
      xl = group.coords[:,0] - source.coords[0]

   f_min = shot.samples.f_min
   f_max = shot.samples.f_max
   nf    = shot.samples.nfreq

   fl = np.linspace(0, f_max, nf)
   cl = np.linspace(c_min, c_max, n_c)
   cf = np.zeros((nf, n_c), dtype=np.single)
   
   # Define window function for spatial damping
   w = np.zeros((n1), dtype=np.single)
   i1 = n1 // 16
   i2 = n1 - i1
   w[i1:i2] = 1
   w = gaussian_filter1d(w, n1 // 32)

   # Evaluate radon-like transform
   for ifreq, f in enumerate(fl):
      for (ic, c) in enumerate(cl):
         v = np.exp(1j * f * 2*np.pi * xl / c) * w
         cf[ifreq, ic] = np.abs(np.dot(shot.data[ifreq,:], v))
         
   plt.xlabel("f (Hz)")
   plt.ylabel(f"c ({units}/s)")
   plt.imshow(
      cf[:,:].transpose(),
      origin='lower',
      cmap=cmap,
      vmin=0,
      vmax=A*np.max(cf),
      extent=[0, f_max, c_min, c_max],
      aspect='auto'
   )
   
   if "save" in kwargs:
      file = kwargs["save"]
      plt.savefig(file, bbox_inches='tight')
   else:
      plt.show()
