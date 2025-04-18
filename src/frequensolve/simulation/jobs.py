import json
import os
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from numpy import arange

from frequensolve.simulation.simulation import BaseSimulation, CustomJSONEncoder
from frequensolve.util.class_registry import class_registry, register_class

__all__ = ["SimulationJob", "FrequencyDomainJob", "TimeDomainJob"]


@register_class
@dataclass
class SimulationJob(ABC):
    name: str
    simulation: BaseSimulation
    f_list: List[float]
    _file: Optional[Path] = None
    _job_id: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict, project_dir: Optional[Path] = None) -> "SimulationJob":
        class_name = d["_type"]
        if class_name in class_registry:
            job_class = class_registry[class_name]
            return job_class.from_dict(d, project_dir=project_dir)
        else:
            raise ValueError(f"Unknown job class: {class_name}")

    @classmethod
    def load(cls, path: Path):
        project_dir = Path(path).parent.parent
        job = cls.from_dict(json.loads(path.read_text()), project_dir=project_dir)
        job._file = path
        return job

    def __dict__(self):
        if self.simulation._file is None:
            raise ValueError("Simulation has not been saved.")

        return {
            "_type": self.__class__.__name__,
            "name": self.name,
            "simulation": str(self.simulation._file),
            "f_list": self.f_list,
        }

    def save(self, path: Optional[Path] = None):
        if path is None:
            sim_file = str(self.simulation._file)
            parent = Path(sim_file.replace("simulations/", "jobs/")).parent
        else:
            raise ValueError(
                "Directory hierarchy is currently somewhat rigid;"
                "for now specifying save path not supported."
            )
        parent.mkdir(parents=True, exist_ok=True)

        file = (parent / self.name).with_suffix(".json")
        file.write_text(json.dumps(self.__dict__(), cls=CustomJSONEncoder, indent=3))

        # Save the file name
        self._file = file
        return file

    def save_for_remote(self, remote_proj: Union[Path, str]):
        sim_file = str(self.simulation._file)
        parent = Path(sim_file.replace("simulations/", "jobs/")).parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        data = self.__dict__()
        proj_dir = str(self.simulation._file.parent.parent)
        data["simulation"] = data["simulation"].replace(proj_dir, str(remote_proj))

        file = (parent / self.name).with_suffix(".json")
        file.write_text(json.dumps(data, cls=CustomJSONEncoder, indent=3))

        # Save the file name
        self._file = file
        return file

    @property
    def project_path(self):
        return Path(self.simulation._file).parent.parent

    @property
    def n_tasks(self):
        return len(self.f_list)

    @property
    def records(self):
        """Get records from Frontera.

        Args:
            outputs: A dictionary of outputs to get.
        """

        output = self.trace_outputs

        # For now we have to get entire files (with all sources, etc.)
        path = output["path"]
        records = {
            "datasets": {},
            "frequencies": {},
            "simulation": self.simulation._file,
        }
        for components in output["components"]:
            group, comp = components.split(":")
            for i, freq in enumerate(output["frequencies"]):
                ifreq = i + 1
                for src in output["sources"]:
                    record = group + "_" + str(ifreq) + ".h5"
                    dset = comp + "_" + str(src)
                    file = os.path.join(path, record)
                    if file not in records["datasets"]:
                        records["datasets"][file] = []
                    records["datasets"][file].append(dset)
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
        recv_out["domain"] = (self.__class__.__name__,)
        recv_out["path"] = out["path"]
        recv_out["frequencies"] = self.f_list
        recv_out["components"] = []
        recv_out["sources"] = []

        for group in receivers:
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


@register_class
class FrequencyDomainJob(SimulationJob):
    def __init__(self, name: str, simulation: BaseSimulation, f_list: List[float]):
        self.name = name
        self.simulation = simulation
        self.f_list = f_list

    @classmethod
    def from_dict(cls, d: dict, project_dir: Optional[Path] = None):
        return cls(
            name=d["name"],
            simulation=BaseSimulation.load(d["simulation"], project_dir=project_dir),
            f_list=d["f_list"],
        )


@register_class
class TimeDomainJob(SimulationJob):
    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_max: float,
        f_min: float = 0.0,
        df: Optional[float] = None,
        T_max: Optional[float] = None,
    ):
        if T_max is not None:
            if df is not None:
                raise ValueError("Either df or T must be provided (not both)")
            df = 1.0 / T_max

        if f_min == 0.0:
            f_min = f_min + df
        f_list = arange(f_min, f_max + df / 2, df)

        self.name = name
        self.simulation = simulation
        self.f_list = f_list

    @classmethod
    def from_dict(cls, d: dict, project_dir: Optional[Path] = None):
        return cls(
            name=d["name"],
            simulation=BaseSimulation.load(d["simulation"], project_dir=project_dir),
            f_list=d["f_list"],
        )
