"""
Builds a survey database for source and receiver coordinates. Implemented with
xarray + Dask for scalability.

The database can be used as follows for sorting and slicing (note, the survey
object doesn't load traces quite yet; it's designed to index traces in the future)

# Example 1: Slice receivers from ID=5 to ID=15, all shots, times up to 1 second
subset = ds.sel(
    receiver = slice(5, 15),
    time     = slice(0, 1.0)
)

# Example 2: Filter by offset < 100
offset_subset = ds.where(ds.offset < 100, drop=True)

# Example 3: Filter by shot with x > 1500
shot_subset = ds.where(ds.source_x > 1500, drop=True)
"""

import xarray as xr
import numpy as np
import bitarray as bits

from pathlib         import Path
from dataclasses     import dataclass
from typing          import Union, Optional, List, Dict
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

__all__ = ['SeismicSurvey']

@dataclass
class SeismicSurvey:
   def __init__(self,
                dim:       int, 
                src:       Optional[Union[str, Path, np.ndarray, xr.DataArray]] = None, 
                recv:      Optional[Union[str, Path, np.ndarray, xr.DataArray]] = None, 
                src_recv:  Optional[Union[str, Path, np.ndarray, xr.DataArray]] = None, 
                format:    Optional[str] = None, **kwargs):
      """
      Initialize the survey object by reading source and receiver coordinates from files or arrays
      and building an xarray "database" for accessing and sorting coordinates.

      The source and receiver coordinates can be provided separately (for source-independent 
      receiver coordinates) or together (for source-dependent receiver coordinates).

      Args:
         dim:       int, dimension of the survey
         src:       array or path giving source coordinates
         recv:      array or path giving receiver coordinates
         src_recv:  array or path giving source-receiver coordinates
         format:    if the preceding are paths, this specifies the file format
         **kwargs:  additional arguments
      """
      chunk = kwargs.get("chunk", 100)
      self.dim   = dim
      self.chunk = chunk

      # Puts input into xarray with expected shape and chunking
      def clean_input(arg):
         if isinstance(arg, str) or isinstance(arg, Path):
            out = self.read_source_coords(arg, format, **kwargs)
         elif isinstance(arg, np.ndarray):
            if arg.ndim == 3:
               if dim == 2:
                  out = xr.DataArray(arg, 
                                    dims   = ["iarg","dir"],
                                    coords = {"iarg": np.arange(arg.shape[0]),
                                              "dir" : ["x","z"]},
                                    chunk  = {"iarg": chunk**2})
               elif dim == 3:
                  out = xr.DataArray(arg, 
                                    dims   = ["iarg","dir"],
                                    coords = {"iarg": np.arange(arg.shape[0]),
                                              "dir" : ["x","y","z"]},
                                    chunk  = {"iarg": chunk**2})
            else:
               if dim == 2:
                  out = xr.DataArray(arg, 
                                    dims   = ["iarg","type","dir"],
                                    coords = {"type": ["src","recv"],
                                              "dir" : ["x","z"]},
                                    chunk  = {"iarg": chunk**2})
               elif dim == 3:
                  out = xr.DataArray(arg, 
                                    dims=["iarg","type","dir"],
                                    coords = {"type": ["src","recv"],
                                              "dir" : ["x","y","z"]},
                                    chunk  = {"iarg": chunk**2})
         elif isinstance(arg, xr.DataArray):
            out = arg
         else:
            raise ValueError("'src' must be a string, Path, numpy array, or xarray DataArray")
         return out

      # Build survey "database" for accessing and sorting coordinates (reciever indp) 
      if src_recv is None:
         if src is None or recv is None:
            raise ValueError("either a combined 'src_recv' or separate 'src' and 'recv' file or data must be provided")
         # Receiver independent of source
         src_coords = clean_input(src)
         recv_coords = clean_input(recv)

         # Define chunked dataset with source and receiver coordinates
         ds = xr.Dataset(
                  coords={
                     "source":   np.arange(np.shape(src_coords)[0]),
                     "receiver": np.arange(np.shape(recv_coords)[0]),
                     **{"source_"   + k: (("source",),   src_coords.dir[k] ) 
                                                for k in src_coords.dir.keys()},
                     **{"receiver_" + k: (("receiver",), recv_coords.dir[k]) 
                                                for k in recv_coords.dir.keys()},
                  }
               )
         
         # Add midpoint and offset coordinates with chunking
         ds = ds.assign_coords({
                  "midpoint_x": (("source", "receiver"),
                                 xr.DataArray(
                                    0.5 * (ds.source_x + ds.receiver_x),
                                    chunks={'source': chunk, 'receiver': chunk}
                                 )),
                  **({"midpoint_y": (("source", "receiver"),
                                 xr.DataArray(
                                    0.5 * (ds.receiver_y + ds.receiver_y),
                                    chunks={'source': chunk, 'receiver': chunk}
                                 ))}
                     if dim == 3 else {}),
               })
         ds = ds.assign_coords({
                  "offset_x": (("source", "receiver"),
                              xr.DataArray(
                                 - (ds.source_x - ds.receiver_x),
                                 chunks={'source': chunk, 'receiver': chunk}
                              )),
                  **({"offset_y": (("source", "receiver"),
                              xr.DataArray(
                                 - (ds.source_y - ds.receiver_y),
                                 chunks={'source': chunk, 'receiver': chunk}
                              ))}
                     if dim == 3 else {}),
               })
         # Write database and unique receiver list and indices to Zarr
         xr.to_zarr(ds.astype(np.float32), "surveyDB.zarr")
         urecv = [ds["receiver_" + k].values.astype(np.float32) for k in ds.receiver_.keys()]
         xr.Dataset({ "urecv": (("receiver",), urecv.astype(np.float32))
                  }).to_zarr("surveyDB_urecv.zarr")
         xr.Dataset({ "src_to_urecv": (("receiver"), np.arange(len(urecv),dtype=np.uint32))
                  }).to_zarr("surveyDB_src_to_urecv.zarr")
         
         # Delete old variables
         del ds, urecv, src_coords, recv_coords
         
         # Read database in chunked format
         self.spatialDB = xr.open_zarr("surveyDB.zarr", chunks={"source": 32})
         self.n_receiver            = self.spatialDB.receiver.size
         self.unique_receivers      = None
         self.source_to_receivers   = None
         self.recievers_independent = True

      elif src is None and recv is None:
         # Receiver depends on source
         src_recv_coords = clean_input(src_recv)

         src  = src_recv_coords.sel(type="src")
         recv = src_recv_coords.sel(type="recv")
         sources, source_indices = np.unique(
            src.data(),
            axis=0,
            return_inverse=True
         ).compute()

         # Get unique reciever list and indices
         urecv, iurecv = np.unique(
            recv.data(),
            axis=0,
            return_inverse=True
         ).compute()

         receivers = {}
         src_to_urecv = {}
         for idx in range(len(sources)):
            matching_coords = np.where(source_indices == idx)[0]
            receivers[idx] = recv.isel(iarg=matching_coords)
            src_to_urecv[idx] = iurecv[matching_coords]

         # Dataset mapping sources to unique receiver indices
         max_receivers = max(r.sizes["iarg"] for r in receivers.values())

         src_to_urecv = xr.DataArray(src_to_urecv, dims=["source", "receiver"]
                                    ).fillna(np.nan).pad(irecv=(0, max_receivers))
         
         xr.Dataset({ "urecv": (("receiver",), urecv.astype(np.float32))
                     }).to_zarr("surveyDB_urecv.zarr")
         xr.Dataset({ "src_to_urecv": (("source", "receiver"), iurecv.astype(np.uint32))
                     }).to_zarr("surveyDB_src_to_urecv.zarr")
         del urecv, iurecv, src_recv_coords, src_to_urecv

         recv_full = xr.DataArray(receivers, dims=["source", "receiver", "dir"]
                        ).fillna(np.nan).pad(irecv=(0, max_receivers))

         # This stores unique receiver indices for each source
         ds = xr.Dataset(
            coords={
               "source": np.arange(len(sources)),
               "receiver": np.arange(max_receivers),
               **{"source_"   + k: (("source",),   sources.dir[k] ) for k in sources.dir.keys()},
               **{"receiver_" + k: (("receiver",), recv_full.isel(dir=k)) for k in recv_full.dir.keys()}
            }
         )

         # Add midpoint coordinates with chunking
         ds = ds.assign_coords({
                  "midpoint_x": (("source", "receiver"),
                                 xr.DataArray(
                                    0.5 * (ds.source_x + 
                                          receivers.isel(receiver=ds.urecv_ind).x),
                                    chunks={'source': chunk, 'receiver': chunk}
                                 )),
                  **({"midpoint_y": (("source", "receiver"),
                                 xr.DataArray(
                                    0.5 * (ds.source_y +
                                          receivers.isel(receiver=ds.urecv_ind).y),
                                    chunks={'source': chunk, 'receiver': chunk}
                                 ))}
                     if dim == 3 else {}),
               })
         
         # Add offset coordinates with chunking
         ds = ds.assign_coords({
                  "offset_x": (("source", "receiver"),
                              xr.DataArray(
                                 receivers.isel(receiver=ds.urecv_ind).x - ds.source_x,
                                 chunks={'source': chunk, 'receiver': chunk}
                              )),
                  **({"offset_y": (("source", "receiver"),
                              xr.DataArray(
                                 receivers.isel(receiver=ds.urecv_ind).y - ds.source_y,
                                 chunks={'source': chunk, 'receiver': chunk}
                              ))}
                     if dim == 3 else {}),
               })

         # Dump database to file  
         xr.to_zarr(ds.astype(np.float32), "surveyDB.zarr")
         del ds, recv_full
            
         # Read database in chunked format
         self.spatialDB           = xr.open_zarr("surveyDB.zarr", chunks={"source": 32})
         self.unique_receivers    = xr.open_zarr("surveyDB_urecv.zarr", 
                                                 chunks={"receiver": chunk**2})
         self.source_to_receivers = xr.open_zarr("surveyDB_src_to_urecv.zarr", 
                                                 chunks={"source": chunk, "receiver": chunk})
         self.recievers_independent = False
         self.n_receiver = len(self.unique_receivers)
      else:
         raise ValueError("either a combined 'src_recv' file or separate 'src' and 'recv' files must be provided")

      # TODO: add units and convert if needed
      self.spatialDB.attrs = {"units": "", "description": "Spatial database for sources and receivers"}
      self.n_source = self.spatialDB.source.size


   def save(self, path: Union[str, Path], cluster: Optional[int] = None):
      """Save the survey object to file"""
      db = self._get_clusterDB(cluster) if cluster else self.spatialDB
      xr.to_zarr(db, path)   


   def _detect_format(self, src_file: Union[str, Path]):
      suffix2format = {"segy": "SEGY",
                       "csv": "CSV",
                       "txt": "plaintext", 
                       "h5": "HDF5"}
      format = suffix2format[src_file.split(".")[-1]]
      return format


   def read_source_coords(self, src_file: Union[str, Path], format: Optional[str] = None, **kwargs):
      """
      Read source coordinates from a file.
      """
      if format is None:
         format = self._detect_format(src_file)

      if format == "SEGY":
         return self._read_segy(src_file, **kwargs)
      elif format == "CSV":
         return self._read_csv(src_file, **kwargs)
      elif format == "plaintext":
         return self._read_plaintext(src_file, **kwargs)
      elif format == "HDF5":
         return self._read_hdf5(src_file, **kwargs)
      else:
         raise ValueError(f"Unsupported format: {format}")


   def read_receiver_coords(self, recv_file: Union[str, Path], format: Optional[str] = None, **kwargs):
      """
      Read receiver coordinates from a file.
      """
      if format is None:
         format = self._detect_format(recv_file)

      if format == "SEGY":
         return self._read_segy(recv_file, **kwargs)
      elif format == "CSV":
         return self._read_csv(recv_file, **kwargs)
      elif format == "plaintext":
         return self._read_plaintext(recv_file, **kwargs)
      elif format == "HDF5":
         return self._read_hdf5(recv_file, **kwargs)
      else:
         raise ValueError(f"Unsupported format: {format}")


   def read_source_receiver_coords(self, src_recv_file: Union[str, Path], format: Optional[str] = None, **kwargs):
      """
      Read source and receiver coordinates from a file.
      """
      if format is None:
         format = self._detect_format(src_recv_file)

      if format == "SEGY":
         return self._read_segy(src_recv_file, **kwargs)
      elif format == "CSV":
         return self._read_csv(src_recv_file, **kwargs)
      elif format == "plaintext":
         return self._read_plaintext(src_recv_file, **kwargs)
      elif format == "HDF5":
         return self._read_hdf5(src_recv_file, **kwargs)
      else:
         raise ValueError(f"Unsupported format: {format}")


# SPS files
# P3  files
   def _read_segy(self, src_recv_file: Union[str, Path], **kwargs):
      """
      Read source and receiver coordinates from a SEGY file.
      """
      raise NotImplementedError
   

   def _read_csv(self, src_recv_file: Union[str, Path], **kwargs):
      """
      Read source and receiver coordinates from a CSV file.
      """
      raise NotImplementedError
   

   def _read_plaintext(self, src_recv_file: Union[str, Path], **kwargs):
      """
      Read source and receiver coordinates from a plaintext file.
      """
      raise NotImplementedError
   

   def _read_hdf5(self, src_recv_file: Union[str, Path], **kwargs):
      """
      Read source and receiver coordinates from an HDF5 file.
      """
      raise NotImplementedError
   

   def get_source_batches(self, group_size: int = 64, **kwargs):
      """Partition sources into batches (batches are simulated simulataneously)
      
      Batchin ~32-128 sources improves vectorization and cache pressure, improving efficiency
      (often by a factor of 8-16x)
      """
      spread_bins = kwargs.get("fbins", 1)
      cluster_size = spread_bins * group_size
      if self.cluster_size < self.n_source:
         self._cluster_sources(cluster_size)
      if spread_bins > 1:
         self._spread_sources(group_size)


   def _plot_source_batches(self, **kwargs):
      """Plot source batches"""
      import matplotlib.pyplot as plt

      rcv_opt = {"color": "blue", "marker": "x", "linestyle": "", "label": "Receiver", "markersize": 18}
      src_opt = {"color": "red",  "marker": "o", "linestyle": "", "label": "Source", "markersize": 18}

      for ic, cluster in enumerate(self.spatialDB.cluster.values):
         recv_ind = self._get_common_receivers(ic)
         if self.dim == 2:
            recv_coords = np.column_stack([self.spatialDB.receiver_x.values, np.zeros_like(self.spatialDB.receiver_x.values)])
            src_coords  = np.column_stack([self.spatialDB.source_x.values,  np.zeros_like(self.spatialDB.source_x.values)])
         elif self.dim == 3:
            recv_coords = np.column_stack([self.spatialDB.receiver_x.values, self.spatialDB.receiver_y.values])
            src_coords  = np.column_stack([self.spatialDB.source_x.values,   self.spatialDB.source_y.values])

         plt.scatter(recv_coords[:,0], recv_coords[:,1], **rcv_opt)
         plt.scatter(src_coords[:,0],  src_coords[:,1],  **src_opt)
         plt.title(f"Cluster {ic}")
         plt.xlabel("x")
         plt.ylabel("y")
         plt.legend()
         plt.show()


   def clip_window(self, window: List[List[float]], cluster: Optional[int] = None):
      """
      Clip the survey to a rectangular window.

      Args:
         window:  bounds (dim = self.dim - 1)for the clip window
         cluster: Optional cluster index to clip only a specific source group
         
      Returns:
         clipped xarray.Dataset containing only sources/receivers within the window.
      """
      # Get the spatial database (subset of the full dataset if a cluster is provided)
      db = self._get_clusterDB(cluster) if cluster else self.spatialDB

      if isinstance(window, (np.ndarray, list)) and hasattr(window[0], "__len__"):
         window = np.array(window).flatten()

      if self.dim == 2:
         if len(window) != 2:
            raise ValueError("Window must be a list of [[xmin, xmax]] or flattened [xmin, xmax].")
         xmin, xmax = window
         if isinstance(db, xr.Dataset) or isinstance(db, xr.DataArray):
            clipped = db.sel(x=slice(xmin, xmax))
         else:
            raise TypeError("The input spatialDB must be an xarray Dataset or DataArray.")
      elif self.dim == 3:
         if len(window) != 4:
            raise ValueError("Window must be a list of [[xmin, xmax], [ymin, ymax]] or flattened [xmin, xmax, ymin, ymax].")
         xmin, xmax, ymin, ymax = window
         if isinstance(db, xr.Dataset) or isinstance(db, xr.DataArray):
            clipped = db.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))
         else:
            raise TypeError("The input spatialDB must be an xarray Dataset or DataArray.")
      return clipped


   def get_offsets(self, cluster: Optional[int] = None) -> xr.Dataset:
      """Get source-receiver offsets (useful for weighting source response)"""
      db = self._get_clusterDB(cluster) if cluster else self.spatialDB
      return np.abs(db) if self.dim==2 else xr.apply_ufunc(
                                                      np.hypot,
                                                      db.offset_x, db.offset_y,
                                                      dask="parallelized"
                                                   )


   def set_reciever_weights(self, w: Union[xr.DataArray, np.ndarray], cluster: Optional[int] = None, **kwargs):
      """Set receiver weights"""
      chunk = kwargs.get("chunk", 100)

      db = self._get_clusterDB(cluster) if cluster else self.spatialDB
      db = db.assign_coords(("source", "receiver"),
                              xr.DataArray(
                                 {
                                    "w": w,
                                    "mask": self._mask_missing_recievers(cluster)
                                 },
                                 chunks={'source': chunk, 'receiver': chunk}
                              ))
      return db


   def _cluster_sources(self, cluster_size: int = 64):
      """Cluster sources that are spatially close to each other.
      
      Batching sources requires taking the union of their patches. Putting the sources close together 
      can thus reduce the required patch size (or when using a DD-approach for inversion it could also
      be beneficial to have the sources close together).
      """
      n_clusters = self.n_source // cluster_size
      if self.dim == 2:
         points = self.spatialDB.source_x.values
      elif self.dim == 3:
         points = np.column_stack([self.spatialDB.source_x.values, self.spatialDB.source_y.values])

      kmeans = KMeans(n_clusters=n_clusters, random_state=42)
      cluster_labels = kmeans.fit_predict(points)
      cluster_labels = self._balance_group_sizes(points, cluster_labels, n_clusters)

      self.spatialDB = self.spatialDB.assign_coords(cluster=("source", cluster_labels))
      return cluster_labels
   

   def _spread_sources(self, n_clusters: int):
      """Spread sources over n_clusters. 

      This is intended to be preceeded by _cluster_sources with a larger cluster size than desired.
      For example, you could cluster sources into groups of ~256, then spread them into groups of 64.
      
      The idea here is that in FD FWI, computing the gradient with a handful of frequencies is known
      to improve convergence. Because we have a fast iterative solver, the cost of perturbing frequency 
      between source groups is relatively mild. So we spread the sources over n_clusters and then simulate
      each group with a different frequency.
      """
      if self.dim == 2:
         points = self.spatialDB.source_x.values
      elif self.dim == 3:
         points = np.column_stack([self.spatialDB.source_x.values, self.spatialDB.source_y.values])

      distance_matrix = squareform(pdist(points))
      distance_matrix = np.exp(-distance_matrix)

      kmeans = KMeans(n_clusters=n_clusters)
      groups = kmeans.fit_predict(distance_matrix)

      return self._balance_group_sizes(points, groups, n_clusters)


   def _balance_group_sizes(self, points: np.ndarray, cluster_labels: np.ndarray, n_clusters: int):
      """Balance group sizes by stealing from the rich and giving to the poor.
      
      This is a simple algorithm, it doesn't take into account the distance between sources.
      """
      sizes = []
      for i in range(n_clusters):
         cluster_size = np.sum(cluster_labels == i)
         sizes.append(cluster_size)

      while np.max(sizes) - np.min(sizes) > 1:
         i = np.argmin(sizes)
         j = np.argmax(sizes)

         isrc = np.argwhere(cluster_labels==j)[0]
         cluster_labels[isrc] = i
         sizes[i] += 1
         sizes[j] -= 1

         for i in range(n_clusters):
            if sizes[i] > np.min(sizes):
               sizes[i] -= 1
               sizes[i+1] += 1

      return cluster_labels


   def _get_unique_receivers(self, cluster: Optional[int] = None):
      """Get list of unique receivers in spatialDB"""
      if self.recievers_independent:
         return np.arange(self.n_receiver,dtype=np.uint32)
      else:
         if cluster is None:
            db = self.spatialDB
            recv_indices = np.unique(self.source_to_receivers.values).compute()
         else:
            db = self._get_clusterDB(cluster)
            recv_indices = np.unique(self.source_to_receivers.isel(
                                                         source=db.source.values
                                                      ).values).compute()
         return recv_indices
      

   def _get_clusterDB(self, cluster: int):
      """Get the xarray "database" for a given cluster"""
      return self.spatialDB.isel(source=self.spatialDB.cluster == cluster)
   

   def _mask_missing_recievers(self, cluster: Optional[int] = None):
      """When reciever locations change with source, we take the union of all 
      reciever locations and then for each reciever, mask out the sources that 
      don't have a reciever at that location.
      """
      if self.recievers_independent:
         return np.ones(self.n_receiver, dtype=bool)
      else:
         if cluster is None:
            db = self.spatialDB
         else:
            db = self._get_clusterDB(cluster)

         recv_list = self._get_unique_receivers(cluster)
         mask = [bits.bitarray(db.source.size) for _ in range(len(recv_list))]

         src_indices = db.source.values
         recv_indices = self.source_to_receivers.isel(
            source=src_indices
         ).values
         
         # Set bits to 1 for present receivers
         for isrc in src_indices:
            for ircv in recv_indices:
               if not np.isnan(ircv):
                  mask[int(ircv)][int(isrc)] = 1
               
         return mask
