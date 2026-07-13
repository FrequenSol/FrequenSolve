"""Imaging-job definitions and readers for solver imaging products."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.seismic.wavelet import Wavelet
from frequensolve.simulation.jobs.base import BaseJob
from frequensolve.simulation.simulation import SeismicSimulation
from frequensolve.util.class_registry import register_class

__all__ = [
    "ImagingJob",
    "Misfit",
    "MisfitGroup",
    "ImageDatabase",
    "extract_frequencies_for_job",
]


@dataclass(kw_only=True)
class ImageDatabase:
    """Reader for solver imaging output files.

    Args:
        path: Directory containing aggregate and per-frequency image HDF5
            files.
        parts: Number of per-frequency image parts.
        shape: Expected image-grid shape.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """

    path: Path
    parts: int
    shape: Tuple[int, ...]

    def __post_init__(self):
        """Normalize and validate the image output directory."""

        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Image path {self.path} does not exist")

    @property
    def f_list(self):
        """Return frequency values recorded in per-part image files.

        Returns:
            NumPy array with one frequency per image part.
        """

        import h5py

        f_list = np.zeros(self.parts)
        for i in range(self.parts):
            file = self.image_file(i + 1)
            with h5py.File(file, "r") as f:
                f_list[i] = f["frequency"][()]
        return f_list

    def image_file(self, part: Optional[int] = None):
        """Return the aggregate or per-frequency image HDF5 file path.

        Args:
            part: Optional one-based image part number. When omitted, the
                aggregate ``image.h5`` path is returned.

        Returns:
            Path to the requested image file.
        """

        if part is None:
            return self.path / "image.h5"
        else:
            return self.path / f"image_{part}.h5"

    def require_aggregate(self) -> Path:
        """Return the aggregate image file or raise a solver-specific error.

        Returns:
            Path to ``image.h5``.

        Raises:
            FileNotFoundError: If the aggregate file written by the solver
                smoothing/stacking workflow is missing.
        """

        file = self.image_file()
        if file.exists():
            return file
        part_files = [self.image_file(i + 1) for i in range(self.parts)]
        existing_parts = [path.name for path in part_files if path.exists()]
        detail = (
            f" Found per-frequency image shard(s): {', '.join(existing_parts)}."
            if existing_parts
            else " No per-frequency image shards were found either."
        )
        raise FileNotFoundError(
            f"Aggregate image file {file} is missing. Sauce writes image.h5 "
            "during the imaging --smooth postprocess after image_N.h5 shards "
            f"are produced.{detail}"
        )

    @property
    def raw_images(self):
        """Return images from the solver ``image/raw`` group.

        Returns:
            Dataset containing raw image volumes.
        """

        return self.read_images("raw")

    @property
    def smoothed_images(self):
        """Return images from the solver ``image/smoothed`` group.

        Returns:
            Dataset containing smoothed image volumes.
        """

        return self.read_images("smoothed")

    @staticmethod
    def _decode_label(value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.bytes_):
            return value.decode("utf-8")
        return str(value)

    @classmethod
    def _decode_attr(cls, value):
        if value is None:
            return None
        if isinstance(value, (bytes, np.bytes_)):
            return cls._decode_label(value)
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return cls._decode_attr(value.item())
            decoded = [cls._decode_attr(item) for item in value.tolist()]
            return decoded[0] if len(decoded) == 1 else decoded
        if isinstance(value, (list, tuple)):
            decoded = [cls._decode_attr(item) for item in value]
            return decoded[0] if len(decoded) == 1 else decoded
        if isinstance(value, np.generic):
            return value.item()
        return value

    @classmethod
    def _axis_units(cls, attrs, dims):
        units = cls._decode_attr(attrs.get("axis_units"))
        if units is None:
            return [None] * len(dims)
        if isinstance(units, str):
            return [units] * len(dims)
        return list(units)[::-1]

    def read_images(self, group):
        """Read one solver image group into an ``xarray.Dataset``.

        Args:
            group: HDF5 group under ``image`` to read, such as ``"raw"`` or
                ``"smoothed"``.

        Returns:
            Dataset containing one data variable per imaged property.
        """

        import h5py
        import xarray as xr

        images = xr.Dataset()
        file = self.require_aggregate()
        with h5py.File(file, "r") as f:
            h5group = f["image"][group]
            properties = [self._decode_label(p) for p in h5group["properties"]]
            for prop in properties:
                h5data = h5group[prop]
                attrs = h5data.attrs
                x0 = np.asarray(attrs["x0"])[::-1]
                x1 = np.asarray(attrs["x1"])[::-1]
                n = np.asarray(attrs["n_grid"], dtype=int)[::-1]
                dims = [self._decode_label(dim) for dim in attrs["dims"][::-1]]
                axis_units = self._axis_units(attrs, dims)
                coords = {}
                for i, dim in enumerate(dims):
                    coords[dim] = np.linspace(x0[i], x1[i], n[i])
                im = xr.DataArray(
                    data=h5data[:].reshape(n),
                    dims=dims,
                    coords=coords,
                )
                for dim, units in zip(dims, axis_units):
                    if units:
                        im.coords[dim].attrs["units"] = units
                units = self._decode_attr(attrs.get("units"))
                if units:
                    im.attrs["units"] = units
                for attr_name in ("coordinate_system", "value_scale", "value_storage"):
                    value = self._decode_attr(attrs.get(attr_name))
                    if value is not None:
                        im.attrs[attr_name] = value
                images[prop] = im
        return images


@dataclass(kw_only=True)
class MisfitGroup:
    """Observed and simulated trace paths for one receiver group misfit.

    Args:
        name: Receiver group name.
        observed: Trace-store root for observed receiver data, or
            ``None`` to request solver-side zero data.
        simulated: Trace-store root for simulated receiver data.
    """

    name: str = ""
    observed: Optional[Union[str, Path]] = None
    simulated: Union[str, Path] = ""

    def __post_init__(self):
        self.observed = None if self.observed is None else Path(self.observed)
        self.simulated = Path(self.simulated)

    def to_fs(self, ctx=None, *, project_relative: bool = False) -> Dict:
        """Serialize the receiver-group misfit path mapping.

        Args:
            ctx: Optional export context accepted for API consistency.
            project_relative: Accepted for API consistency; path conversion is
                handled by ``ImagingJob``.

        Returns:
            JSON-compatible receiver-group misfit payload.
        """

        return {
            "name": self.name,
            "observed": self.observed,
            "simulated": self.simulated,
        }

    @classmethod
    def from_fs(cls, data: Dict) -> "MisfitGroup":
        """Deserialize a receiver-group misfit path mapping.

        Args:
            data: Serialized misfit-group payload.

        Returns:
            ``MisfitGroup`` with trace-store roots restored.
        """

        return cls(
            name=data["name"],
            observed=data["observed"],
            simulated=data["simulated"],
        )


@dataclass(kw_only=True)
class Misfit:
    """Misfit norm and receiver groups used by an imaging job.

    Args:
        norm: Misfit norm name. Currently the solver-facing contract supports
            ``"L2"``.
        receiver_groups: Receiver-group misfit path mappings.
    """

    norm: Literal["L2"] = "L2"
    receiver_groups: List[MisfitGroup] = field(default_factory=list)

    def to_fs(self, ctx=None, *, project_relative: bool = False) -> Dict:
        """Serialize the imaging misfit configuration.

        Args:
            ctx: Optional export context forwarded to receiver groups.
            project_relative: Accepted for API consistency; path conversion is
                handled by ``ImagingJob``.

        Returns:
            JSON-compatible misfit payload.
        """

        return {
            "norm": self.norm,
            "receiver_groups": [group.to_fs(ctx) for group in self.receiver_groups],
        }

    @classmethod
    def from_fs(cls, data: Dict) -> "Misfit":
        """Deserialize an imaging misfit configuration.

        Args:
            data: Serialized misfit payload.

        Returns:
            ``Misfit`` with receiver groups restored.
        """

        return cls(
            norm=data["norm"],
            receiver_groups=[
                MisfitGroup.from_fs(group) for group in data["receiver_groups"]
            ],
        )


@register_class
@dataclass(kw_only=True)
class ImagingJob(BaseJob):
    """Reverse-time migration or FWI-imaging job configuration.

    Args:
        name: Job name used in project paths and serialized payloads.
        simulation: Seismic simulation used to compute simulated data.
        data_path: Optional directory containing observed trace data. When
            ``None``, the solver receives ``null`` and uses zero data to compute
            sensitivity kernels.
        f_list: Frequencies to image.
        resolution: Optional image-grid resolution used when ``grid`` is not
            provided.
        grid: Optional explicit Cartesian image grid.
        images: Mapping from image name to solver imaging condition.
        weights: Optional per-frequency weights.
        wavelet: Optional wavelet whose spectrum is sampled for weights.
        misfit_norm: Misfit norm name.
        keep_forward: Keep forward wavefields after imaging.
        keep_adjoint: Keep adjoint wavefields after imaging.
        keep_unstacked: Keep per-source or per-frequency image contributions.
        regularization: Optional smoothing/regularization payload.
        save_path: Optional image-output directory.
        reassemble_adjoint: Request adjoint reassembly from stored pieces.
        **kwargs: Extra imaging payload fields preserved on export.

    Raises:
        FileNotFoundError: If a non-null ``data_path`` does not exist.
        ValueError: If weights do not match frequencies or a grid cannot be
            inferred from the simulation model.
    """

    misfit: Misfit = field(default_factory=Misfit)
    data_path: Optional[Union[str, Path]]
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
        data_path: Optional[Union[str, Path]] = None,
        f_list: Optional[List[float]] = None,
        resolution: Optional[List[int]] = None,
        grid: Optional[CartesianGrid] = None,
        images: Optional[dict] = None,
        weights: Optional[List[float]] = None,
        wavelet: Optional[Wavelet] = None,
        misfit_norm: Literal["L2"] = "L2",
        keep_forward: bool = False,
        keep_adjoint: bool = False,
        keep_unstacked: bool = False,
        regularization: Optional[dict] = None,
        save_path: Optional[Union[str, Path]] = None,
        reassemble_adjoint: bool = False,
        **kwargs,
    ) -> None:
        """Create an imaging job.

        The job compares simulated traces against observed data at
        ``data_path`` and writes image products to ``save_path`` or the job
        result directory.
        """

        if "misfit_type" in kwargs:
            misfit_norm = kwargs.pop("misfit_type")
        if f_list is None:
            raise ValueError("f_list is required for an imaging job")
        simulation.save()
        super().__init__(
            name=name,
            simulation=simulation,
            f_list=f_list,
            workflow="RTM",
        )

        f_sim = self.trace_outputs.path

        self.data_path = None if data_path is None else Path(data_path)
        if self.data_path is not None and not self.data_path.exists():
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
                self.weights.append(float(spectrum[idx]))
        elif weights is not None:
            self.weights = np.asarray(weights, dtype=float).reshape(-1).tolist()
            if len(self.weights) != len(self.f_list):
                raise ValueError(
                    "Imaging weights must contain one value per frequency; "
                    f"got {len(self.weights)} weights for {len(self.f_list)} frequencies"
                )
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
        self.images = dict(images or {})
        self.keep_forward = keep_forward
        self.keep_adjoint = keep_adjoint
        self.keep_unstacked = keep_unstacked
        self.misfit = Misfit(norm=misfit_norm)
        for receiver_group in simulation.acquisition.receiver_groups:
            self.misfit.receiver_groups.append(
                MisfitGroup(
                    name=receiver_group.name,
                    observed=self.data_path,
                    simulated=f_sim,
                )
            )

        if grid is not None:
            self.grid = grid
        # TODO: this will only work for a layered model right now
        elif simulation.model.dimension == 2:
            if resolution is None:
                raise ValueError("resolution is required when grid is not provided")
            x_limits = simulation.model.x_limits
            z_limits = simulation.model.z_limits
            x0 = [x_limits[0], z_limits[0]]
            x1 = [x_limits[1], z_limits[1]]
            assert (
                len(resolution) == 2
            ), "For 2D models, resolution must be a length 2 array"
            self.grid = CartesianGrid(x0=x0, x1=x1, n=resolution)
        elif simulation.model.dimension == 3:
            if resolution is None:
                raise ValueError("resolution is required when grid is not provided")
            x_limits = simulation.model.x_limits
            y_limits = simulation.model.y_limits
            z_limits = simulation.model.z_limits
            x0 = [x_limits[0], y_limits[0], z_limits[0]]
            x1 = [x_limits[1], y_limits[1], z_limits[1]]
            assert (
                len(resolution) == 3
            ), "For 3D models, resolution must be a length 3 array"
            self.grid = CartesianGrid(x0=x0, x1=x1, n=resolution)
        else:
            raise ValueError(f"Unknown model dimension: {simulation.model.dimension}")
        self.reassemble_adjoint = reassemble_adjoint

    def _remote_image_path(self, work_dir: Union[Path, str]):
        """Get local path but with version number."""
        rel_path = self.save_path.relative_to(self.project_path)
        remote = Path(work_dir) / rel_path
        return remote

    @property
    def _local_image_path(self):
        return self.save_path

    def image_file(self, part: Optional[int] = None) -> Path:
        """Return the local aggregate or per-frequency image file path."""

        if part is None:
            return self.save_path / "image.h5"
        return self.save_path / f"image_{part}.h5"

    def image_output_exists(self) -> bool:
        """Return whether the aggregate image product exists locally."""

        return self.image_file().is_file()

    def image_part_outputs_exist(self) -> bool:
        """Return whether every per-frequency image shard exists locally."""

        return all(
            self.image_file(part).is_file() for part in range(1, self.n_tasks + 1)
        )

    def needs_image_smoothing(self) -> bool:
        """Return whether local shards are present but ``image.h5`` is missing."""

        return self.image_part_outputs_exist() and not self.image_output_exists()

    def is_run_current(self) -> bool:
        """Return whether the imaging run and aggregate image are current."""

        return super().is_run_current() and self.image_output_exists()

    def _export_path(
        self,
        path: Optional[Union[str, Path]],
        *,
        project_relative: bool = False,
    ):
        if path is None:
            return None
        path = Path(path)
        if not project_relative:
            return path
        try:
            return path.resolve().relative_to(self._project_path())
        except ValueError:
            return path

    def _misfit_to_fs(self, ctx=None, *, project_relative: bool = False) -> Dict:
        payload = self.misfit.to_fs(ctx)
        if not project_relative:
            return payload
        for group in payload["receiver_groups"]:
            group["observed"] = self._export_path(
                group["observed"], project_relative=True
            )
            group["simulated"] = self._export_path(
                group["simulated"], project_relative=True
            )
        return payload

    @staticmethod
    def _resolve_saved_path(
        path: Union[str, Path, None],
        *,
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        if path is None:
            return None
        path = Path(path)
        if path.is_absolute():
            return path
        if project_path is not None:
            return Path(project_path) / path
        if base_path is not None:
            project_root = BaseJob._project_root_from_job_path(Path(base_path))
            if project_root is not None:
                return project_root / path
            return Path(base_path).resolve() / path
        return path

    def to_fs(self, ctx=None, *, project_relative: bool = False) -> Dict:
        """Serialize this imaging job to the solver job contract.

        Args:
            ctx: Optional export context accepted for API consistency.
            project_relative: When true, emit project-relative paths where
                possible.

        Returns:
            JSON-compatible imaging job payload.
        """

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
            "data_path": self._export_path(
                self.data_path, project_relative=project_relative
            ),
            "save_path": self._export_path(
                self.save_path, project_relative=project_relative
            ),
            "misfit": self._misfit_to_fs(ctx, project_relative=project_relative),
            "grid": self.grid.to_fs(ctx),
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
            **super().to_fs(project_relative=project_relative),
            "Image": imaging,
        }

    @classmethod
    def from_fs(
        cls,
        data: Dict,
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> "ImagingJob":
        """Deserialize an imaging job from a saved job payload.

        Args:
            data: Serialized imaging job payload.
            base_path: Optional directory used to resolve relative paths.
            project_path: Optional project root used to resolve
                project-relative paths.

        Returns:
            Reconstructed ``ImagingJob``.

        Raises:
            KeyError: If the payload does not include an imaging section.
        """

        data = dict(data)
        stored_project_path = data.get("project_path")
        resolved_project_path = project_path or stored_project_path
        image_data = data.pop("Image", data.pop("Imaging", None))
        if image_data is None:
            raise KeyError("ImagingJob data must include an 'Image' section")
        grid = image_data.pop("grid", None)
        grid_obj = CartesianGrid.from_fs(grid) if grid is not None else None
        resolution = list(grid_obj.n) if grid_obj is not None else None
        misfit = Misfit.from_fs(image_data.pop("misfit"))
        images = image_data.pop("images", None) or []
        images = {
            image["name"]: (
                f"FWI:{image['property']}"
                if image.get("IC") == "FWI" and image.get("property") is not None
                else image["IC"]
            )
            for image in images
        }
        simulation_ref = data.pop("simulation")
        simulation_path = Path(simulation_ref)
        if not simulation_path.is_absolute():
            project_path = project_path or data.get("project_path")
            if project_path is None and base_path is not None:
                project_path = BaseJob._project_root_from_job_path(Path(base_path))
            if project_path is not None:
                simulation_path = Path(project_path) / simulation_path

        project_path = project_path or data.get("project_path")
        data_path = cls._resolve_saved_path(
            image_data.pop("data_path", None),
            base_path=base_path,
            project_path=resolved_project_path,
        )
        save_path = cls._resolve_saved_path(
            image_data.pop("save_path", None),
            base_path=base_path,
            project_path=resolved_project_path,
        )
        for group in misfit.receiver_groups:
            group.observed = cls._resolve_saved_path(
                group.observed,
                base_path=base_path,
                project_path=resolved_project_path,
            )
            group.simulated = cls._resolve_saved_path(
                group.simulated,
                base_path=base_path,
                project_path=resolved_project_path,
            )

        job = cls(
            name=data.pop("name", None),
            simulation=SeismicSimulation.load(simulation_path),
            f_list=data.pop("f_list", None),
            data_path=data_path,
            resolution=resolution,
            grid=grid_obj,
            images=images,
            keep_forward=image_data.pop("keep_forward", None),
            keep_adjoint=image_data.pop("keep_adjoint", None),
            keep_unstacked=image_data.pop("keep_unstacked", None),
            save_path=save_path,
            regularization=image_data.pop("Smoothing", None),
            weights=image_data.pop("weights", None),
            reassemble_adjoint=image_data.pop("reassemble_adjoint", None),
            **image_data,
        )
        job.misfit = misfit
        return job


def extract_frequencies_for_job(job: ImagingJob, td):
    """Extract job frequencies from time traces into solver-ready HDF5 files.

    Args:
        job: Imaging job whose frequency list and data path define the output.
        td: Time-domain trace data array.

    Returns:
        ``None``. One HDF5 trace file is written for each job frequency.
    """

    import h5py
    import xarray as xr

    if job.data_path is None:
        raise ValueError(
            "Cannot extract observed frequencies for an imaging job without data_path"
        )

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
        file = job.data_path / f"traces_{i + 1}.h5"
        with h5py.File(file, "w") as f:
            f[group.name] = fd.data[i].astype(np.float32)
