import json, toml, yaml

from pathlib      import Path
from dataclasses  import dataclass, field
from typing       import Optional, Union, Dict, List, Literal, Tuple

from ..orchestrator.jobs.base_job import BaseJob

from .config                     import *  # noqa
from .sampling                   import *  # noqa
from .output_manager             import *  # noqa
from .numerics_manager           import *  # noqa
from ..model.model               import *  # noqa
from ..mesh.mesh                 import *  # noqa
from ..mesh.mesh_generators      import *  # noqa
from ..mesh.mesh_manager         import *  # noqa
from ..mesh.boundary_conditions  import *  # noqa
from ..seismic.acquisition       import *  # noqa
from ..util.memoization          import *  # noqa
from ..util.class_registry       import *  # noqa

__all__ = ['CustomJSONEncoder','BaseSimulation', 'SeismicSimulation', 'CustomTOMLEncoder']

class CustomJSONEncoder(json.JSONEncoder):
   """Custom JSON encoder for Simulation objects."""
   def default(self, obj):
      import numpy as np
      if isinstance(obj, (np.integer, np.floating, np.bool_)):
         return obj.item()
      if isinstance(obj, np.ndarray):
         return obj.tolist()
      if isinstance(obj, Path):
         return str(obj)
      if hasattr(obj, '__dict__'):
         return obj.__dict__()
      return super().default(obj)


def custom_json_decoder(obj):
    if '_type' in obj:
        class_name = obj['_type']
        if class_name in class_registry:
            model_class = class_registry[class_name]
            return model_class.from_dict(obj)
    return obj


class CustomTOMLEncoder(toml.TomlEncoder):
   """Custom TOML encoder for Simulation objects."""
   def __init__(self):
      super().__init__()
      
   def dump_value(self, obj):
      import xarray as xr
      import numpy as np
      
      if isinstance(obj, (np.integer, np.floating, np.bool_)):
         return obj.item()
      if isinstance(obj, np.ndarray):
         return obj.tolist()
      if isinstance(obj, xr.DataArray):
         return obj.values.tolist()
      if isinstance(obj, Path):
         return str(obj)
      if hasattr(obj, 'tolist'):
         return obj.tolist()
      try:
         return str(obj)
      except:
         print(f"Cannot encode object of type {type(obj)}")
         return None


@register_class
class BaseSimulation(SimulationConfig):
   """Container for simulation configuration.
   
   Attributes:
      model (ModelBase):               Model configuration.
      mesh (MeshManager):              Mesh configuration.
      BCs (BoundaryConditionManager):  Boundary condition configuration.
      numerics (NumericsManager):      Numerics configuration.
      output (OutputManager):          Output configuration.
   """
   model:          Optional[ModelBase]      = None
   mesh:           Optional[MeshManager]    = None
   BCs:            BoundaryConditionManager = field(default_factory=BoundaryConditionManager)
   solver:         SolverConfig             = field(default_factory=SolverConfig)
   discretization: Discretization           = field(default_factory=Discretization)  
   outputs:        OutputManager            = field(default_factory=OutputManager) 

   @classmethod
   def from_dict(cls, data: Dict) -> 'BaseSimulation':
      class_name = data["_type"]
      if class_name in class_registry:
         simulation_class = class_registry[class_name]
         return simulation_class.from_dict(data)
      else:
         raise ValueError(f"Unknown simulation class: {class_name}")
      
   def __post_init__(self):
      from ..util.ansi_colors import ANSIColorCodes as c
      if isinstance(self.mesh, Mesh) or \
         isinstance(self.mesh, BaseMeshGenerator):
         print(f"{c.note}Note: Simulation was initialized with a Mesh or MeshGenerator; specifying the mesh via a\n"
                "the 'MeshManager' class is recommended as it specifies mesh parallelism and adaptivity parameters.\n"
                "We've specified a default MeshManager for you but for fine grained control you'll want to specify\n"
               f"your own.{c.none}")
         self.mesh = MeshManager(self.mesh)

   def __dict__(self) -> Dict:
      from ..util.ansi_colors import ANSIColorCodes as c
      if isinstance(self.mesh, Mesh) or \
         isinstance(self.mesh, BaseMeshGenerator):
         print(f"{c.note}Note: Simulation was initialized with a Mesh or MeshGenerator; specifying the mesh via a\n"
                "the 'MeshManager' class is recommended as it specifies mesh parallelism and adaptivity parameters.\n"
                "We've specified a default MeshManager for you but for fine grained control you'll want to specify\n"
               f"your own.{c.none}")  
         self.mesh = MeshManager(self.mesh)
      return {
         "_type": self.__class__.__name__,
         **super().__dict__(),
         **({"Model":          self.model.__dict__()}          if self.model else {}),
         **({"Mesh":           self.mesh.__dict__()}           if self.mesh else {}),
         **({"BCs":            self.BCs.__dict__()}            if self.BCs else {}),
         **({"Solver":         self.solver.__dict__()}         if self.solver else {}),
         **({"Discretization": self.discretization.__dict__()} if self.discretization else {}),
         **({"Outputs":        self.outputs.__dict__()}        if self.outputs else {}),
      }


@register_class
@dataclass(kw_only=True)
class SeismicSimulation(BaseSimulation):
   """Container for seismic simulation configuration.
   
   Attributes:
      name (str):                      Name of the simulation.
      model (ModelBase):               Model configuration.
      mesh (MeshManager):              Mesh configuration.
      BCs (BoundaryConditionManager):  Boundary condition configuration.
      solver (SolverConfig):           Solver configuration.
      discretization (Discretization): Discretization configuration.
      output (OutputManager):          Output configuration.   
      acquisition (Acquisition):       Acquisition configuration.
   """
   name:           str
   physics:        Literal["acoustic", "elastic", "plasma"]
   dimension:      Literal[2, 3]
   model:          Optional[ModelBase]      = None
   mesh:           Optional[MeshManager]    = None
   BCs:            BoundaryConditionManager = field(default_factory=BoundaryConditionManager)
   solver:         SolverConfig             = field(default_factory=SolverConfig)
   discretization: Discretization           = field(default_factory=Discretization)  
   outputs:        OutputManager            = field(default_factory=OutputManager) 
   acquisition:    Optional[Acquisition]    = None


   @classmethod
   def from_dict(cls, data: Dict) -> "SeismicSimulation":
      name      = data["name"]
      physics   = data["physics"]
      dimension = data["dimension"]

      sim = cls(name=name, physics=physics, dimension=dimension)

      if "Model" in data:
         sim.model = ModelBase.from_dict(data["Model"])
      if "Mesh" in data:
         sim.mesh = MeshManager.from_dict(data["Mesh"])
      if "BCs" in data:
         sim.BCs = BoundaryConditionManager.from_dict(data["BCs"])
      if "Solver" in data:
         sim.solver = SolverConfig.from_dict(data["Solver"])
      if "Discretization" in data:
         sim.discretization = Discretization.from_dict(data["Discretization"])
      if "Outputs" in data:
         sim.outputs = OutputManager.from_dict(data["Outputs"])
      if "Acquisition" in data:
         sim.acquisition = Acquisition.from_dict(data["Acquisition"])

      return sim 

   @classmethod
   def load(cls, path: Union[str, Path], **kwargs) -> 'SeismicSimulation':
      """Load seismic simulation from JSON file."""

      with open(path, "r") as f:
         data = json.load(f)
         return cls.from_dict(data)

   def __dict__(self) -> Dict:
      dict = super().__dict__()
      dict.update({
         "_type": self.__class__.__name__,
         **({"Acquisition":    self.acquisition.__dict__()}    if self.acquisition else {}),
      })
      return dict

   def as_json(self, **kwargs) -> str:
      indent = kwargs.get("indent", 3)
      return json.dumps(self.__dict__(), cls=CustomJSONEncoder, indent=indent, **kwargs)

   def as_toml(self, **kwargs) -> str:
      indent = kwargs.get("indent", 3)
      return toml.dumps(self.__dict__(), encoder=CustomTOMLEncoder(), indent=indent, **kwargs)
   
   def as_yaml(self, **kwargs) -> str:

      def numpy_representer(dumper, data):
         """Convert numpy values to native Python types."""
         return dumper.represent_float(float(data))

      indent = kwargs.get("indent", 3)
      try:
         import numpy.typing as npt
         yaml.add_representer(npt.Float64, numpy_representer)
         yaml.add_representer(npt.Float32, numpy_representer)
         yaml.add_representer(npt.Int64, lambda dumper, data: dumper.represent_int(int(data)))
         yaml.add_representer(npt.Int32, lambda dumper, data: dumper.represent_int(int(data)))
         
         return yaml.dump(
            self.__dict__(), 
            indent=indent,
            default_flow_style=False,
            sort_keys=False,
            **kwargs
         )
      except Exception as e:
         print(f"Failed to convert to YAML: {e}")
         return self.__repr__()

   def save(self, path: Union[str, Path], **kwargs) -> str:
      """Save seismic simulation to JSON file."""
      file = (Path(path) / f"{self.name}.json").resolve()

      if not file.parent.exists():
         file.parent.mkdir(parents=True, exist_ok=True)

      indent = kwargs.get("indent", 3)
      with open(file, "w") as f:
         json.dump(self.__dict__(), f, cls=CustomJSONEncoder, indent=indent, **kwargs)
      return file.relative_to(self._proj_path)
      
   def check(self) -> bool:
      """Check the simulation is defined correctly."""
      return True
   
   @quantize
   @memoized_func
   def _estimate_memory(self,f) -> int:
      """Estimate memory required for the simulation."""
      pass

   def _set_path(self, proj_path: Path, rel_path: Path):
      self._proj_path = proj_path
      self._rel_path = rel_path/self.name
      if self.acquisition:
         self.acquisition._set_path(self._proj_path, self._rel_path)
      if self.model:
         self.model._set_path(self._proj_path, self._rel_path)
      if self.mesh:
         self.mesh._set_path(self._proj_path, self._rel_path)

   @property
   def _path(self) -> Path:
      return self._proj_path/self._rel_path


# TODO: loop over source groups
# TODO: also generate jobs for particular source groups and frequencies
   # def _generate_jobs(self, 
   #                    f_list: List[float]) -> List[BaseJob]:
   #    jobs = []
   #    for f in f_list:
   #       mem = self._estimate_memory(f)
   #       jobs.append(BaseJob(self, f, required_memory = mem))
   #    return jobs
      
   # def get_material_mesh(self, f_max: float) -> MaterialMesh:
   #    """Get the mesh the material model is defined on given the maximum frequency."""
   #    pass







  # TODO: add checkpointing for FWI so that even when job is killed we can restart from where we left off

   # def load_shot_FD(self, key: str, isrc: int) -> Shot:
   #    """Read frequency-domain shot data, then apply the wavelet signature.

   #    Args:
   #       key (str): A string like "groupName:fieldName".
   #       isrc (int): The source number (1-based).

   #    Returns:
   #       Shot: A Shot object containing FD data.
   #    """
      
   #    try:
   #       import h5py
   #    except:
   #       print("h5py not found, skipping frequency-domain data")
   #       return None
      
   #    group_name, field = key.split(":")
   #    group = self.receiver_group(group_name)
   #    nrecv = group.size

   #    if isinstance(self.sampling, UniformSweepSampling):
   #       of = self.sampling.ofreq
   #       nf = self.sampling.nfreq
   #       f_max = self.sampling.f_max

   #       wavelet  = self.source_group.signal(isrc)
   #       spectrum = wavelet.spectrum
   #    else:
   #       of = 0
   #       spectrum = np.ones([self.sampling.nfreq])
      
   #    u = np.zeros((nf, nrecv), dtype=np.csingle)
      
   #    # Loop over frequencies and load data
   #    for ifreq, freq in enumerate(self.sampling.freqs):
   #       file = os.path.join(group.directory, f"{group_name}_{ifreq}.h5")
   #       i_omega = np.csingle(1j * 2 * np.pi * freq)
         
   #       if ifreq >= of and not os.path.exists(file):
   #          warnings.warn(f"File {file} does not exist.", UserWarning)
   #       else:
   #          with h5py.File(file, "r") as f:
   #             # Real + imaginary parts
   #             u[ifreq, :] += np.csingle(1j) * f[f"{field}_{isrc}_im"][()]
   #             u[ifreq, :] +=              f[f"{field}_{isrc}_re"][()]
               
   #             # Apply wavelet
   #             u[ifreq, :] *= spectrum[ifreq]
               
   #             # For fiber-type receivers, multiply by iω for strain *rate*
   #             if group.kind == 'fiber':
   #                u[ifreq, :] *= i_omega
               
   #             f.close()
         
   #    return ShotRecord(type           = "FD",
   #                      number         = isrc,
   #                      sampling       = self.sampling,
   #                      source         = self.source(isrc),
   #                      receiver_group = group,
   #                      field          = field,
   #                      data           = u)


   # def read_shot_TD(self, key: str, isrc: int) -> Shot:
   #    """Read time-domain shot data by first reconstructing from the frequency-domain.

   #    Args:
   #       key (str): A string like "groupName:fieldName".
   #       isrc (int): The source number (1-based).

   #    Returns:
   #       Shot: A Shot object containing time-domain data.
   #    """
      
   #    if not isinstance(self.sampling,UniformSweepSampling):
   #       raise ValueError("Time-domain data is only supported for uniform sweep sampling.")
      
   #    try:
   #       import pyfftw.interfaces.numpy_fft as fft
   #    except:
   #       print('pyfftw not found, using numpy for FFT (slow)')
   #       import numpy.fft as fft
      
   #    group_name, field = key.split(":")
   #    group = self.receiver_group(group_name)
   #    nrecv = group.size
   
   #    nf = self.sampling.nfreq
   #    nF = self.sampling.nFreq
      
   #    fd = self.read_shot_FD(key, isrc)
      
   #    # If upscaled, create a bigger array for inverse transform
   #    if nF > nf:
   #       FD = np.zeros((nF, nrecv), dtype=np.csingle)
   #       FD[:nf, :] = fd.data[:nf, :]
   #       del fd
   #       td = fft.irfft(FD, axis=0)
   #       del FD
   #    else:
   #       td = fft.irfft(fd.data, axis=0)
   #       del fd
         
   #    return Shot(type           = "TD",
   #                number         = isrc,
   #                sampling       = self.sampling,
   #                source         = self.source(isrc),
   #                receiver_group = group,
   #                field          = field,
   #                data           = td)
   




   # def signal(self, irecv: int) -> Wavelet:
   #    """Retrieves the signal for a specific receiver.
      
   #    Used in adjoint calculations where receivers act as sources.
      
   #    Args:
   #       irecv (int): 1-based index of the receiver.
         
   #    Returns:
   #       Wavelet: The signal associated with the specified receiver.
   #    """
   #    return self.signals.get(irecv)
   
   
   # def signature(self, isrc: int):
   #    if self.signals:
   #       return self.signals.get(isrc)
   #    else:
   #       return None


