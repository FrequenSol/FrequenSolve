import json
import os
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from frequensolve.simulation.simulation import BaseSimulation, CustomJSONEncoder
from frequensolve.util.class_registry import class_registry, register_class

__all__ = ["SimulationJob", "FrequencyDomainJob", "TimeDomainJob"]

# TODO: Give job a status field
# TODO: Enable max versions (overwrite after a certain number of versions)


@register_class
@dataclass
class SimulationJob(ABC):
    name: str
    simulation: BaseSimulation
    workflow: str
    f_list: List[Union[float, complex]]
    overwrite: bool = True
    max_versions: int = 5
    _file: Optional[Path] = None
    _job_id: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationJob":
        class_name = d["_type"]
        if class_name in class_registry:
            job_class = class_registry[class_name]
            return job_class.from_dict(d)
        else:
            raise ValueError(f"Unknown job class: {class_name}")

    @classmethod
    def load(cls, path: Union[Path, str]):
        path = Path(path).resolve()
        with open(path, "r") as f:
            data = json.load(f)
            job = cls.from_dict(data)
        job._file = path
        return job

    def __dict__(self):
        if self.simulation._file is None:
            raise ValueError("Simulation has not been saved.")
        if isinstance(self.f_list[0], complex):
            f_list = np.array([[f.real, -abs(f.imag)] for f in self.f_list])
        else:
            f_list = np.array(self.f_list)
        return {
            "_type": self.__class__.__name__,
            "name": self.name,
            "simulation": str(self.simulation._file),
            "workflow": self.workflow,
            "f_list": f_list,
        }

    @property
    def n_tasks(self):
        return len(self.f_list)

    @property
    def records(self):
        """Lists records that should be produced by a job.

        Returns:
            dict: Dictionary containing:
                - datasets: Dictionary of datasets
                - frequencies: List of frequencies
                - simulation: Path to simulation file
        """

        output = self.trace_outputs

        # For now we have to get entire files (with all sources, etc.)
        path = output["path"]
        records = {
            "groups": output["groups"],
            "frequencies": {},
            "simulation": self.simulation._file,
        }
        records["groups"] = output["groups"]
        records["files"] = []
        for i, freq in enumerate(output["frequencies"]):
            ifreq = i + 1
            file = os.path.join(path, f"receivers_{ifreq}.h5")
            records["files"].append(file)
            records["frequencies"][ifreq] = freq

        return records

    @property
    def paraview_outputs(self) -> dict:
        """Lists ParaView outputs.

        Returns:
            dict: Dictionary containing:
                - ParaView: ParaView outputs
        """
        sim = self.simulation

        sim_file = sim._file
        with open(sim_file, "r") as f:
            sim_data = json.load(f)

        pv_out = {}
        for out in sim_data["Outputs"]["ParaView"]:
            pv_out[out["name"]] = out["path"]

        return pv_out

    @property
    def trace_path(self) -> dict:
        """Lists receiver trace groups.

        Returns:
            - traces: Receiver traces
        """
        return self.trace_outputs["path"]

    @property
    def trace_outputs(self) -> dict:
        """Lists receiver trace groups.

        Returns:
            - traces: Receiver traces
        """
        sim = self.simulation
        receivers = sim.acquisition.receiver_groups

        sim_file = sim._file
        with open(sim_file, "r") as f:
            sim_data = json.load(f)

        out = sim_data["Outputs"]["receivers"]

        recv_out = {}
        recv_out["path"] = self._result_path / out["path"]
        recv_out["frequencies"] = self.f_list
        recv_out["groups"] = []
        recv_out["components"] = []
        recv_out["sources"] = []

        for group in receivers:
            recv_out["groups"].append(group.name)
            for component in group.device.components:
                recv_out["components"].append(f"{group.name}:{component.name}")

        for isrc, sgroup in enumerate(sim.acquisition.source_groups):
            recv_out["sources"].append(f"{isrc+1}")

        return recv_out

    @property
    def wavefield_outputs(self) -> dict:
        """Lists wavefield outputs.

        Returns:
            - wavefields: Wavefield outputs
        """
        sim = self.simulation
        receivers = sim.acquisition.receiver_groups

        sim_file = sim._file
        with open(sim_file, "r") as f:
            sim_data = json.load(f)

        outputs = sim_data["Outputs"]["wavefields"]

        wave_out = {}
        for out in outputs:
            wave_out["domain"] = (self.__class__.__name__,)
            wave_out["path"] = out["path"]
            wave_out["frequencies"] = self.f_list
            wave_out["grid"] = out["grid"]
            wave_out["components"] = []
            wave_out["sources"] = []

            for group in receivers:
                for component in group.device.components:
                    wave_out["components"].append(f"{group.name}:{component.name}")

            for isrc, source in enumerate(sim.acquisition.source_group.sources):
                wave_out["sources"].append(f"{isrc+1}")

        return wave_out

    def _new_version(
        self, site: Optional[str] = None, remote_path: Optional[Union[Path, str]] = None
    ):
        manifest = self._manifest
        version = manifest["current_version"] + 1

        # TODO: loop through directory and check whether it actually exists
        manifest["current_version"] = version

        local_path = self._local_path / f"v{version}"
        if remote_path is not None:
            remote_path = Path(remote_path) / f"v{version}"
        else:
            remote_path = None

        if site is None:
            site = "LocalSite"
            path = local_path
        else:
            path = remote_path

        manifest["versions"][version] = {
            "created": datetime.now().isoformat(),
            "site": site,
            "path": f"{path}",
        }
        self._manifest_file.write_text(
            json.dumps(manifest, cls=CustomJSONEncoder, indent=3)
        )
        return local_path, remote_path

    def save(self):
        project_path = self.project_path
        if not self.overwrite:
            job_path, _ = self._new_version()
        else:
            job_path = self._local_path

        file = job_path / f"{self.name}.json"
        parent = file.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        self._file = file
        data = self.__dict__()
        data["result_path"] = str(self._result_path.relative_to(self.project_path))
        file.write_text(json.dumps(data, cls=CustomJSONEncoder, indent=3))
        return file

    def save_for_remote(self, site: str, remote_project: Union[Path, str]):
        """Save the job for remote simulation.

        Args:
            site (str): The site to save the job for.
            remote_project (Union[Path, str]): The remote project to save the job to.
        """
        remote_project = Path(remote_project)
        remote_path = remote_project / "jobs" / self.simulation.name / self.name

        if not self.overwrite:
            local_path, remote_path = self._new_version(site, remote_path)
        else:
            local_path = self._local_path
        local_path.mkdir(parents=True, exist_ok=True)
        data = self.__dict__()

        local_project = f"{self.simulation._proj_path}"

        # Recursively process the data dictionary
        def replace_path(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    replace_path(value)
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            replace_path(item)
                if isinstance(value, Path):
                    if local_project in str(value):
                        d[key] = str(value).replace(local_project, str(remote_project))
                if isinstance(value, str):
                    if local_project in value:
                        d[key] = value.replace(local_project, str(remote_project))
            return d

        local_file = local_path / f"{self.name}.json"
        remote_file = remote_path / f"{self.name}.json"

        data = replace_path(data)
        self._file = local_file

        data["result_path"] = str(self._result_path.relative_to(self.project_path))
        local_file.write_text(json.dumps(data, cls=CustomJSONEncoder, indent=3))

        return local_file, remote_file

    def _remote_path(self, work_dir: Union[Path, str]):
        """Get remote job path."""
        work_dir = Path(work_dir)
        base = work_dir / "jobs" / self.simulation.name / self.name
        if not self.overwrite:
            version = self._version
            path = base / f"v{version}"
        else:
            path = base
        return path

    @property
    def _local_path(self):
        """Get local job path."""
        project_path = self.project_path
        return project_path / "jobs" / self.simulation.name / self.name

    @property
    def _save_path(self):
        """Get local path but with version number."""
        if not self.overwrite:
            version = self._version + 1
            path = self._local_path / f"v{version}"
        else:
            path = self._local_path
        return path

    @property
    def _manifest_file(self):
        return self._local_path / "manifest.json"

    @property
    def _manifest(self):
        if not self._manifest_file.exists():
            self._create_manifest()
        return json.loads(self._manifest_file.read_text())

    def _create_manifest(self):
        manifest_file = self._manifest_file
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_data = {
            "current_version": 0,
            "versions": {},
        }
        manifest_file.write_text(
            json.dumps(manifest_data, cls=CustomJSONEncoder, indent=3)
        )

    @property
    def _version(self):
        return self._manifest["current_version"]

    @property
    def _stdout_path(self):
        """Path where solver stdout is stored."""
        return self._file.parent / "logs"

    # TODO: note that for now these will always be local; will be troublesome for remote simulation
    @property
    def _result_path(self):
        """Path where solver results will be stored."""
        if self._file is None:
            return self._save_path / "results"
        else:
            return self._file.parent / "results"

    @property
    def project_path(self):
        return self.simulation._proj_path


@register_class
class FrequencyDomainJob(SimulationJob):
    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_list: List[Union[float, complex]],
        overwrite: bool = True,
        max_versions: int = 5,
    ):
        workflow = "forward"

        # Ensure that imaginary component (laplace) is negative
        for f in f_list:
            if isinstance(f, complex):
                f = f.real - 1j * abs(f.imag)
        super().__init__(name, simulation, workflow, f_list, overwrite, max_versions)

    @classmethod
    def from_dict(cls, d: dict):
        sim = BaseSimulation.load(d["simulation"])
        shape = d["f_list"].shape
        if len(shape) == 1:
            f_list = d["f_list"].tolist()
        else:
            f_list = [f[0] - 1j * abs(f[1]) for f in d["f_list"]]
        f_list = np.array(f_list)
        return cls(
            name=d["name"],
            simulation=sim,
            f_list=f_list,
        )


@register_class
class TimeDomainJob(SimulationJob):
    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_max: float,
        f_min: float = 0.0,
        s_laplace: float = 0.0,
        df: Optional[float] = None,
        T_max: Optional[float] = None,
        overwrite: bool = True,
        max_versions: int = 5,
    ):
        if T_max is not None:
            if df is not None:
                raise ValueError("Either df or T must be provided (not both)")
            df = 1.0 / T_max

        if f_min == 0.0:
            f_min = f_min + df
        f_list = np.arange(f_min, f_max + df / 2, df)

        s_laplace = -abs(s_laplace)
        f_list = f_list + 1j * s_laplace

        workflow = "forward"
        super().__init__(name, simulation, workflow, f_list, overwrite, max_versions)

    @classmethod
    def from_dict(cls, d: dict):
        if isinstance(d["f_list"][0], float):
            f_min = d["f_list"][0]
            f_max = d["f_list"][-1]
            df = d["f_list"][1] - d["f_list"][0]
            s_laplace = 0.0
        else:
            f_min = d["f_list"][0][0]
            f_max = d["f_list"][-1][0]
            df = d["f_list"][1][0] - d["f_list"][0][0]
            s_laplace = d["f_list"][0][1]
        sim = BaseSimulation.load(d["simulation"])
        fl = np.arange(f_min, f_max + df / 2, df)
        fl += 1j * s_laplace
        assert np.allclose(d["f_list"], fl), "Frequency does not appear to be uniform"
        return cls(
            name=d["name"],
            simulation=sim,
            f_min=f_min,
            f_max=f_max,
            df=df,
        )
