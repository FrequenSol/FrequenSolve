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
    "ImagingJob",
    "Misfit",
    "MisfitGroup",
    "ImageDatabase",
    "extract_frequencies_for_job",
]


@dataclass(kw_only=True)
class ImageDatabase:
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

    def image_file(self, part: Optional[int] = None):
        if part is None:
            return self.path / f"image.h5"
        else:
            return self.path / f"image_{part}.h5"

    @property
    def raw_images(self):
        return self.read_images("raw")

    @property
    def smoothed_images(self):
        return self.read_images("phi")

    def read_images(self, group):
        import h5py
        import xarray as xr

        images = xr.Dataset()
        file = self.image_file()
        with h5py.File(file, "r") as f:
            h5group = f["image"][group]
            properties = [p.decode("utf-8") for p in h5group["properties"]]
            for prop in properties:
                attrs = h5group[prop].attrs
                x0 = attrs.get("x0", None)[::-1]
                x1 = attrs.get("x1", None)[::-1]
                n = attrs.get("n_grid", None)[::-1]
                dims = attrs.get("dims", None)[::-1]
                coords = {}
                for i, dim in enumerate(dims):
                    coords[dim] = np.linspace(x0[i], x1[i], n[i])
                im = xr.DataArray(
                    data=h5group[prop][:].reshape(n),
                    dims=dims,
                    coords=coords,
                )
                images[prop] = im
        return images


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

    norm: Literal["L2"] = "L2"
    receiver_groups: List[MisfitGroup] = field(default_factory=list)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def __dict__(self) -> Dict:
        return {
            "norm": self.norm,
            "receiver_groups": [group.__dict__() for group in self.receiver_groups],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Misfit":
        return cls(
            norm=data["norm"],
            receiver_groups=[
                MisfitGroup.from_dict(group) for group in data["receiver_groups"]
            ],
        )


@register_class
@dataclass(kw_only=True)
class ImagingJob(SimulationJob):
    """Defines imaging job."""

    misfit: Misfit = field(default_factory=Misfit)
    data_path: Union[str, Path]
    save_path: Union[str, Path]
    grid: CartesianGrid = field(default_factory=CartesianGrid)
    keep_forward: bool = False
    keep_adjoint: bool = False
    keep_unstacked: bool = False
    images: dict = field(default_factory=dict)
    weights: List[float] = field(default_factory=list)
    reassemble_adjoint: bool = False

    def __init__(
        self,
        name: str,
        simulation: SeismicSimulation,
        data_path: Union[str, Path],
        f_list: List[float],
        resolution: List[int],
        images: dict = field(default_factory=dict),
        weights: Optional[List[float]] = None,
        wavelet: Optional[Wavelet] = None,
        misfit_norm: Literal["L2"] = "L2",
        keep_forward: bool = False,
        keep_adjoint: bool = False,
        keep_unstacked: bool = False,
        regularization: Optional[dict] = None,
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
            self.weights = None

        if regularization is not None:
            self.regularization = regularization
        else:
            self.regularization = {
                "type": "TV",
                "lambda": 1.0,
                "epsilon": 1.0,
                "iterations": 5,
            }

        self.kwargs = kwargs
        self.images = images
        self.keep_forward = keep_forward
        self.keep_adjoint = keep_adjoint
        self.keep_unstacked = keep_unstacked
        self.regularization = regularization
        self.misfit = Misfit(norm=misfit_norm)
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
        images = []
        for key, value in self.images.items():
            tmp = value.split(":")
            if tmp[0] == "FWI":
                cond = tmp[0]
                prop = tmp[1]
            else:
                cond = value
                prop = None
            images.append(
                {
                    "name": key,
                    "IC": cond,
                    **({"property": prop} if prop is not None else {}),
                }
            )

        imaging = {
            "data_path": self.data_path,
            "save_path": self.save_path,
            "misfit": self.misfit.__dict__(),
            "grid": self.grid.__dict__(),
            "keep_forward": self.keep_forward,
            "keep_adjoint": self.keep_adjoint,
            "keep_unstacked": self.keep_unstacked,
            "images": images,
            "weights": self.weights,
            "reassemble_adjoint": self.reassemble_adjoint,
            "Smoothing": self.regularization,
            **self.kwargs,
        }
        return {
            **super().__dict__(),
            "Imaging": imaging,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ImagingJob":
        image_data = data.pop("Image")
        grid = image_data.pop("grid", None)
        resolution = grid.pop("n", None)
        images = image_data.pop("images", None)
        images = {image["name"]: image["IC"] for image in images}
        job = cls(
            name=data.pop("name", None),
            simulation=SeismicSimulation.load(data.pop("simulation")),
            f_list=data.pop("f_list", None),
            data_path=image_data.pop("data_path", None),
            resolution=resolution,
            images=images,
            keep_forward=image_data.pop("keep_forward", None),
            keep_adjoint=image_data.pop("keep_adjoint", None),
            keep_unstacked=image_data.pop("keep_unstacked", None),
            save_path=image_data.pop("save_path", None),
            regularization=image_data.pop("Smoothing", None),
            weights=image_data.pop("weights", None),
            reassemble_adjoint=image_data.pop("reassemble_adjoint", None),
            **image_data,
        )
        job.misfit = Misfit.from_dict(image_data["misfit"])
        return job


def extract_frequencies_for_job(job: ImagingJob, td):
    import warnings

    import h5py
    import xarray as xr

    try:
        import pyfftw

        pyfftw.interfaces.cache.enable()
        fft = pyfftw.interfaces.numpy_fft
        pyfftw.config.NUM_THREADS = 4
    except:
        warnings.warn("pyfftw not found, using numpy for FFT (slow)")
        import numpy.fft as fft

    # Make frequency domain data array
    shape = list(td.shape)
    fd_dims = ["time" if d == "frequency" else d for d in td.dims]
    fd_coords = {d: td.coords[d] for d in fd_dims if d != "time"}
    fd_coords["frequency"] = job.f_list
    fd_coords["complex"] = ["real", "imag"]
    fd_coords["component"] = "v_z"
    for i, d in enumerate(td.dims):
        if d == "time":
            fd_dims[i] = "frequency"
            shape[i] = len(job.f_list)
    if "component" not in td.dims:
        shape.append(1)
        fd_dims.append("component")
    fd_dims.append("complex")
    fd_shape = [*shape, 2]
    fd = xr.DataArray(
        np.zeros(fd_shape),
        dims=fd_dims,
        coords=fd_coords,
    )
    fd = fd.transpose("frequency", "source", "component", "receiver", "complex")

    # Compute frequency domain data
    taxis = td.coords["time"]
    dt = taxis[1] - taxis[0]
    freqs = job.f_list
    for i, f in enumerate(freqs):
        exp = xr.DataArray(
            np.exp(2 * np.pi * 1j * f * td.coords["time"]),
            dims=["time"],
            coords={"time": td.coords["time"]},
        )
        tmp = (td * exp).sum(dim="time") * dt
        ctmp = np.empty([*tmp.shape, 2], dtype=np.float32)
        ctmp[:, :, 0] = tmp.real
        ctmp[:, :, 1] = tmp.imag
        fd.data[i] = ctmp.reshape(fd.data[i].shape)

    # Assumes only one receiver group
    group = job.simulation.acquisition.receiver_groups[0]

    # Write hdf5 files
    for i, f in enumerate(job.f_list):
        file = job.data_path / f"receivers_{i + 1}.h5"
        with h5py.File(file, "w") as f:
            f[group.name] = fd.data[i].astype(np.float32)
