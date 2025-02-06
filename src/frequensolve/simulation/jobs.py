import json
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
            parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)

        file = (parent / self.name).with_suffix(".json")
        file.write_text(json.dumps(self.__dict__(), cls=CustomJSONEncoder, indent=3))

        # Save the file name
        self._file = file
        return file

    @property
    def n_tasks(self):
        return len(self.f_list)


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
        T: Optional[float] = None,
    ):
        if T is not None:
            if df is not None:
                raise ValueError("Either df or T must be provided (not both)")
            df = 1.0 / T

        if f_min == 0.0:
            f_min = f_min + df
        f_list = arange(f_min, f_max, df)

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
