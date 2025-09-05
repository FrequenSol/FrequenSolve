import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.seismic.wavelet import Wavelet
from frequensolve.simulation.jobs import SimulationJob
from frequensolve.simulation.sampling import UniformSweepSampling
from frequensolve.simulation.simulation import SeismicSimulation
from frequensolve.util.class_registry import class_registry, register_class

__all__ = [
    "RTMImagingJob",
    "Misfit",
    "MisfitGroup",
    "RawImage",
]


@dataclass(kw_only=True)
class RawImage:
    """Raw image."""

    path: Path
    parts: int
    shape: Tuple[int, ...]

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Image path {self.path} does not exist")

    @property
    def f_list(self):
        import h5py

        f_list = np.zeros(self.parts)
        for i in range(self.parts):
            file = self.image_file(i + 1)
            with h5py.File(file, "r") as f:
                f_list[i] = f["frequency"][()]
        return f_list

    def image_file(self, part: int):
        return self.path / f"image_{part}.h5"

    def image(self, part: int):
        import h5py

        file = self.image_file(part + 1)
        with h5py.File(file, "r") as f:
            im = np.reshape(f["image"][:], self.shape)
            return im

    def stack(
        self, omega_exp: int = 0, wavelet: Optional[Wavelet] = None
    ) -> np.ndarray:
        import h5py

        n_freq = self.parts
        f_list = self.f_list

        weights = np.ones(self.parts, dtype=np.float64)
        if n_freq > 1 and wavelet is not None:
            f_max = f_list.max()
            f_list = np.sort(f_list)
            df = np.diff(f_list).min()

            sampling = UniformSweepSampling(
                f_min=0.0,
                f_max=f_max,
                df=df,
            )
            wavelet.times = sampling.t_list

            frequencies = wavelet.frequencies
            spectrum = abs(wavelet.spectrum)
            for i, f in enumerate(self.f_list):
                idx = abs(frequencies - f).argmin()
                weights[i] = abs(spectrum[idx])

        w_norm = np.sqrt(np.sum(np.abs(weights) ** 2))
        weights /= w_norm

        img = np.zeros(self.shape)
        for i in range(n_freq):
            file = self.image_file(i + 1)
            f0 = f_list.max()
            omega = 2.0 * np.pi * f_list[i]
            w = omega ** (omega_exp - 2) * weights[i] ** 2
            with h5py.File(file, "r") as f:
                im = np.reshape(f["image"][:], self.shape)
                img += im * w
        return img


@dataclass(kw_only=True)
class MisfitGroup:
    """Base class for misfit groups."""

    name: str = ""
    observed: Union[str, Path] = ""
    simulated: Union[str, Path] = ""

    def __post_init__(self):
        self.observed = Path(self.observed) / f"receivers"
        self.simulated = Path(self.simulated) / f"receivers"

    def __dict__(self) -> Dict:
        return {
            "name": self.name,
            "observed": self.observed,
            "simulated": self.simulated,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MisfitGroup":
        return cls(
            name=data["name"],
            observed=data["observed"],
            simulated=data["simulated"],
        )


@dataclass(kw_only=True)
class Misfit:
    """Base class for misfit functions."""

    type: Literal["L2"] = "L2"
    receiver_groups: List[MisfitGroup] = field(default_factory=list)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def __dict__(self) -> Dict:
        return {
            "type": self.type,
            "receiver_groups": [group.__dict__() for group in self.receiver_groups],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Misfit":
        return cls(
            type=data["type"],
            receiver_groups=[
                MisfitGroup.from_dict(group) for group in data["receiver_groups"]
            ],
        )


@register_class
@dataclass(kw_only=True)
class RTMImagingJob(SimulationJob):
    """Defines RTM image."""

    misfit: Misfit = field(default_factory=Misfit)
    data_path: Union[str, Path]
    save_path: Union[str, Path]
    grid: CartesianGrid = field(default_factory=CartesianGrid)
    keep_forward: bool = False
    keep_adjoint: bool = False
    imaging_condition: Literal["energy", "up_down"] = "energy"
    weights: List[float] = field(default_factory=list)
    reassemble_adjoint: bool = False

    def __init__(
        self,
        name: str,
        simulation: SeismicSimulation,
        data_path: Union[str, Path],
        f_list: List[float],
        resolution: List[int],
        imaging_condition: Literal["energy", "up_down"] = "energy",
        weights: Optional[List[float]] = None,
        wavelet: Optional[Wavelet] = None,
        misfit_type: Literal["L2"] = "L2",
        keep_forward: bool = False,
        keep_adjoint: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        reassemble_adjoint: bool = False,
        overwrite: bool = True,
        max_versions: int = 5,
        **kwargs,
    ) -> None:
        simulation.save()
        super().__init__(
            name=name,
            simulation=simulation,
            f_list=f_list,
            workflow="RTM",
            overwrite=overwrite,
            max_versions=max_versions,
        )

        # This is a simple way to ensure that recevier paths are correct
        sim_file = simulation._file
        with open(sim_file, "r") as f:
            sim_data = json.load(f)
        f_sim = self._result_path / sim_data["Outputs"]["receivers"]["path"]

        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {self.data_path} does not exist")

        if save_path is not None:
            self.save_path = Path(save_path).resolve()
        else:
            self.save_path = self._result_path / "imaging"
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)

        if wavelet is not None:
            self.wavelet = wavelet
            frequencies = self.wavelet.frequencies
            spectrum = abs(self.wavelet.spectrum)
            self.weights = []
            for f in self.f_list:
                idx = abs(frequencies - f).argmin()
                self.weights.append(spectrum[idx])
        else:
            assert weights is not None, "Either wavelet or weights must be provided"
            self.weights = weights

        self.kwargs = kwargs
        self.imaging_condition = imaging_condition
        self.keep_forward = keep_forward
        self.keep_adjoint = keep_adjoint

        self.misfit = Misfit(type=misfit_type)
        for receiver_group in simulation.acquisition.receiver_groups:
            self.misfit.receiver_groups.append(
                MisfitGroup(
                    name=receiver_group.name,
                    observed=Path(data_path),
                    simulated=f_sim,
                )
            )

        # TODO: this will only work for a layered model right now
        if simulation.model.dimension == 2:
            x_limits = simulation.model.x_limits
            z_limits = simulation.model.z_limits
            x0 = [x_limits[0], z_limits[0]]
            x1 = [x_limits[1], z_limits[1]]
            assert (
                len(resolution) == 2
            ), "For 2D models, resolution must be a length 2 array"
        elif simulation.model.dimension == 3:
            x_limits = simulation.model.x_limits
            y_limits = simulation.model.y_limits
            z_limits = simulation.model.z_limits
            x0 = [x_limits[0], y_limits[0], z_limits[0]]
            x1 = [x_limits[1], y_limits[1], z_limits[1]]
            assert (
                len(resolution) == 3
            ), "For 3D models, resolution must be a length 3 array"
        else:
            raise ValueError(f"Unknown model dimension: {simulation.model.dimension}")

        self.grid = CartesianGrid(x0=x0, x1=x1, n=resolution)
        self.reassemble_adjoint = reassemble_adjoint

    def _remote_image_path(self, work_dir: Union[Path, str]):
        """Get local path but with version number."""
        rel_path = self.save_path.relative_to(self.project_path)
        remote = Path(work_dir) / rel_path
        return remote

    @property
    def _local_image_path(self):
        return self.save_path

    def __dict__(self) -> Dict:
        imaging = {
            "data_path": self.data_path,
            "save_path": self.save_path,
            "misfit": self.misfit.__dict__(),
            "grid": self.grid.__dict__(),
            "keep_forward": self.keep_forward,
            "keep_adjoint": self.keep_adjoint,
            "imaging_condition": self.imaging_condition,
            "weights": self.weights,
            "reassemble_adjoint": self.reassemble_adjoint,
            **self.kwargs,
        }
        return {
            **super().__dict__(),
            "Imaging": imaging,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RTMImagingJob":
        image_data = data.pop("Image")
        grid = image_data.pop("grid", None)
        resolution = grid.pop("n", None)
        job = cls(
            name=data.pop("name", None),
            simulation=SeismicSimulation.load(data.pop("simulation")),
            f_list=data.pop("f_list", None),
            data_path=image_data.pop("data_path", None),
            resolution=resolution,
            imaging_condition=image_data.pop("imaging_condition", None),
            keep_forward=image_data.pop("keep_forward", None),
            keep_adjoint=image_data.pop("keep_adjoint", None),
            save_path=image_data.pop("save_path", None),
            weights=image_data.pop("weights", None),
            reassemble_adjoint=image_data.pop("reassemble_adjoint", None),
            **image_data,
        )
        job.misfit = Misfit.from_dict(image_data["misfit"])
        return job


# @register_class
# @dataclass(kw_only=True)
# class ImagingJob(ABC, SimulationJob):
#     """Base class for seismic imaging jobs."""

#     misfit: Misfit = field(default_factory=Misfit)
#     data_path: Union[str, Path] = None
#     save_path: Union[str, Path] = None

#     @classmethod
#     def from_dict(cls, data: Dict) -> "ImagingJob":
#         class_name = data.pop("_type")
#         if class_name in class_registry:
#             image_class = class_registry[class_name]
#             return image_class.from_dict(data)
#         else:
#             raise ValueError(f"Unknown image class: {class_name}")

#     @abstractmethod
#     def __dict__(self) -> Dict:
#         pass
