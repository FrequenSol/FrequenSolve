"""Forward simulation job classes.

Frequency-domain jobs run an explicit list of frequency or Laplace samples.
Time-domain jobs derive a uniform frequency sweep from ``f_min``/``f_max`` and
either ``df`` or ``T_max`` so the solver can reconstruct time traces.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np

from frequensolve.simulation.jobs.base import BaseJob
from frequensolve.simulation.outputs import JobOutputs, Output
from frequensolve.simulation.simulation import BaseSimulation
from frequensolve.util.class_registry import register_class

__all__ = [
    "FrequencyDomainJob",
    "TimeDomainJob",
]


@register_class
class FrequencyDomainJob(BaseJob):
    """Forward job that solves explicitly requested frequency samples.

    Args:
        name: Job name used in project paths and serialized payloads.
        simulation: Simulation object to run.
        f_list: Frequencies to solve. Complex values encode Laplace damping in
            their imaginary component; damping is normalized to a negative
            imaginary value for the solver.
        outputs: Optional output request or output collection.
    """

    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_list: List[Union[float, complex]],
        outputs: Optional[Union[Output, Iterable[Output], JobOutputs]] = None,
    ):
        workflow = "forward"
        frequencies = np.asarray(f_list)
        if np.iscomplexobj(frequencies):
            frequencies = np.asarray([f.real - 1j * abs(f.imag) for f in frequencies])
        super().__init__(
            name,
            simulation,
            workflow,
            frequencies.tolist(),
            JobOutputs(outputs),
        )

    @classmethod
    def from_fs(
        cls,
        d: dict,
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ):
        """Deserialize a saved frequency-domain forward job.

        Args:
            d: Serialized job payload.
            base_path: Optional directory used to resolve relative simulation
                paths.
            project_path: Optional project root used to remap project-relative
                paths.

        Returns:
            Reconstructed ``FrequencyDomainJob``.
        """

        sim = BaseJob._load_simulation_for_job(
            d["simulation"],
            base_path=base_path,
            project_path=project_path or d.get("project_path"),
            source_project=d.get("project_path"),
        )
        f_list = cls._decode_frequencies(d["f_list"])
        job = cls(
            name=d["name"],
            simulation=sim,
            f_list=f_list,
            outputs=JobOutputs.from_fs(d.get("Outputs")),
        )
        job._job_id = d.get("job_id")
        return job


@register_class
class TimeDomainJob(BaseJob):
    """Forward job defined by a uniform frequency sweep for time traces.

    Args:
        name: Job name used in project paths and serialized payloads.
        simulation: Simulation object to run.
        f_max: Maximum frequency in the sweep.
        f_min: Minimum frequency in the sweep. A zero minimum is advanced to
            the first positive frequency increment.
        damping_factor: Optional time-domain damping factor converted to a
            Laplace value.
        laplace: Optional explicit Laplace damping value. Mutually exclusive
            with ``damping_factor``.
        df: Frequency spacing. Mutually exclusive with ``T_max``.
        T_max: Time-domain period used to derive ``df`` as ``1 / T_max``.
        outputs: Optional output request or output collection.

    Raises:
        ValueError: If damping options conflict, neither ``df`` nor ``T_max``
            is supplied, spacing is non-positive, or ``f_max`` is not greater
            than ``f_min``.
    """

    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_max: float,
        f_min: float = 0.0,
        damping_factor: Optional[float] = None,
        laplace: Optional[float] = None,
        df: Optional[float] = None,
        T_max: Optional[float] = None,
        outputs: Optional[Union[Output, Iterable[Output], JobOutputs]] = None,
    ):
        if damping_factor is not None and laplace is not None:
            raise ValueError("Specify only one of damping_factor or laplace")
        if df is None and T_max is None:
            raise ValueError("TimeDomainJob requires either df or T_max")
        if T_max is not None:
            if df is not None:
                raise ValueError("Specify only one of df or T_max")
            if T_max <= 0:
                raise ValueError("T_max must be positive")
            df = 1.0 / T_max
        if df <= 0:
            raise ValueError("df must be positive")
        if f_max <= f_min:
            raise ValueError("f_max must be greater than f_min")

        period = 1.0 / df
        if damping_factor is not None:
            if damping_factor < 1.0:
                raise ValueError("damping_factor must be greater than or equal to 1")
            laplace = -np.log(float(damping_factor)) / (2.0 * np.pi * period)

        if f_min == 0.0:
            f_min = f_min + df
        f_list = np.arange(f_min, f_max + df / 2, df)

        laplace = -abs(float(laplace or 0.0))
        if laplace != 0.0:
            f_list = f_list + 1j * laplace

        workflow = "forward"
        super().__init__(name, simulation, workflow, f_list, JobOutputs(outputs))

    @classmethod
    def from_fs(
        cls,
        d: dict,
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ):
        """Deserialize a saved time-domain job from its frequency list.

        Args:
            d: Serialized job payload.
            base_path: Optional directory used to resolve relative simulation
                paths.
            project_path: Optional project root used to remap project-relative
                paths.

        Returns:
            Reconstructed ``TimeDomainJob``.

        Raises:
            ValueError: If the saved frequency list is too short or is not a
                uniform sweep.
        """

        f_list = cls._decode_frequencies(d["f_list"])
        if f_list.size < 2:
            raise ValueError("TimeDomainJob requires at least two frequencies")

        f_min = float(np.real(f_list[0]))
        f_max = float(np.real(f_list[-1]))
        df = float(np.real(f_list[1] - f_list[0]))
        laplace = float(np.imag(f_list[0]))
        expected = np.arange(f_min, f_max + df / 2, df)
        if laplace != 0.0:
            expected = expected + 1j * laplace
        if not np.allclose(f_list, expected):
            raise ValueError("Frequency list does not appear to be uniform")

        sim = BaseJob._load_simulation_for_job(
            d["simulation"],
            base_path=base_path,
            project_path=project_path or d.get("project_path"),
            source_project=d.get("project_path"),
        )
        job = cls(
            name=d["name"],
            simulation=sim,
            f_min=f_min,
            f_max=f_max,
            df=df,
            laplace=laplace,
            outputs=JobOutputs.from_fs(d.get("Outputs")),
        )
        job._job_id = d.get("job_id")
        return job
