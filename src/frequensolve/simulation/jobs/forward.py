"""Forward simulation job classes.

Frequency-domain jobs run an explicit list of frequency or Laplace samples.
Time-domain jobs derive a uniform frequency sweep from ``f_min``/``f_max`` and
either ``df`` or ``T_max`` so the solver can reconstruct time traces.
"""

from numbers import Integral
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Union

import numpy as np

from frequensolve.simulation.jobs.base import BaseJob
from frequensolve.simulation.outputs import JobOutputs, Output
from frequensolve.simulation.simulation import BaseSimulation
from frequensolve.util.class_registry import register_class
from frequensolve.util.mixins import ExportContext

__all__ = [
    "FrequencyDomainJob",
    "TimeDomainJob",
]


def _phase_derivative_order(value: int) -> int:
    """Normalize one public phase-derivative order."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("phase_derivatives must be an integer from 0 to 4")
    order = int(value)
    if not 0 <= order <= 4:
        raise ValueError("phase_derivatives must be an integer from 0 to 4")
    return order


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
        k_list: Optional signed physical Fourier wavenumbers for 2.5D jobs.
        k_weights: Optional quadrature weights paired with ``k_list``.
        k_units: Optional units for ``k_list`` and ``k_weights``.

        phase_derivatives: Highest physical-frequency derivative to compute,
            from zero (the default) through four. A positive value selects the
            solver's ``forward_df`` workflow and includes every derivative from
            first order through the requested order.

    Raises:
        ValueError: If ``phase_derivatives`` is not an integer from zero
            through four.
    """

    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_list: List[Union[float, complex]],
        outputs: Optional[Union[Output, Iterable[Output], JobOutputs]] = None,
        k_list: Optional[Iterable[float]] = None,
        k_weights: Optional[Iterable[float]] = None,
        k_units: Optional[str] = None,
        phase_derivatives: int = 0,
    ):
        phase_derivatives = _phase_derivative_order(phase_derivatives)

        workflow = "forward_df" if phase_derivatives else "forward"
        frequencies = np.asarray(f_list)
        if np.iscomplexobj(frequencies):
            frequencies = np.asarray([f.real - 1j * abs(f.imag) for f in frequencies])
        super().__init__(
            name=name,
            simulation=simulation,
            workflow=workflow,
            f_list=frequencies.tolist(),
            outputs=JobOutputs(outputs),
            k_list=k_list,
            k_weights=k_weights,
            k_units=k_units,
        )
        self.phase_derivatives: int = phase_derivatives

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, project_relative: bool = False
    ) -> dict:
        """Serialize the requested derivative order using Sauce's job contract."""

        payload = super().to_fs(ctx, project_relative=project_relative)
        if self.phase_derivatives:
            payload["derivative_order"] = self.phase_derivatives
        return payload

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
        workflow = str(d.get("workflow", "forward"))
        phase_derivatives = (
            int(d.get("derivative_order", 1)) if workflow == "forward_df" else 0
        )
        job = cls(
            name=d["name"],
            simulation=sim,
            f_list=f_list,
            outputs=JobOutputs.from_fs(d.get("Outputs")),
            k_list=d.get("k_list"),
            k_weights=d.get("k_weights"),
            k_units=d.get("k_units"),
            phase_derivatives=phase_derivatives,
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
        k_list: Optional signed physical Fourier wavenumbers for 2.5D jobs.
        k_weights: Optional quadrature weights paired with ``k_list``.
        k_units: Optional units for ``k_list`` and ``k_weights``.

        reconstruction: Frequency-to-time reconstruction method. ``"standard"``
            preserves the uniform forward sweep. ``"hermite"`` requests
            ``forward_df`` and keeps every ``sample_every``-th frequency.
        sample_every: Positive integer selecting every Nth point on the target
            frequency grid for a Hermite solve. Defaults to four.
        high_frequency_taper: High-frequency continuation width in hertz.
            ``True`` uses one solved-frequency interval and a positive number
            specifies the width explicitly. The continuation uses the simulated
            first derivative at the highest frequency and is available for both
            standard and Hermite reconstruction.
        interpolation_time_shift: Time shift in seconds used to remove a linear
            phase trend before Hermite interpolation and restore it afterward.
        phase_derivatives: Highest physical-frequency derivative to compute,
            from zero through four. Hermite reconstruction and a nonzero
            ``high_frequency_taper`` require at least first order and enable it
            automatically.

    Raises:
        ValueError: If damping options conflict, neither ``df`` nor ``T_max``
            is supplied, spacing is non-positive, or ``f_max`` is not greater
            than ``f_min``, or ``phase_derivatives`` is outside zero through
            four.
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
        k_list: Optional[Iterable[float]] = None,
        k_weights: Optional[Iterable[float]] = None,
        k_units: Optional[str] = None,
        reconstruction: Literal["standard", "hermite"] = "standard",
        sample_every: Optional[int] = None,
        high_frequency_taper: Union[bool, float] = False,
        interpolation_time_shift: float = 0.0,
        phase_derivatives: int = 0,
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
        phase_derivatives = _phase_derivative_order(phase_derivatives)
        reconstruction = str(reconstruction).lower()
        if reconstruction not in {"standard", "hermite"}:
            raise ValueError("reconstruction must be 'standard' or 'hermite'")
        if sample_every is None:
            sample_every = 4 if reconstruction == "hermite" else 1
        if isinstance(sample_every, bool) or not isinstance(sample_every, Integral):
            raise ValueError("sample_every must be an integer of at least one")
        sample_every = int(sample_every)
        if sample_every < 1:
            raise ValueError("sample_every must be an integer of at least one")
        if reconstruction == "standard" and sample_every != 1:
            raise ValueError("sample_every is only used by Hermite reconstruction")
        interpolation_time_shift = float(interpolation_time_shift)
        if not np.isfinite(interpolation_time_shift):
            raise ValueError("interpolation_time_shift must be finite")
        if reconstruction == "standard" and interpolation_time_shift != 0.0:
            raise ValueError(
                "interpolation_time_shift is only used by Hermite reconstruction"
            )
        if isinstance(high_frequency_taper, (bool, np.bool_)):
            high_frequency_taper = bool(high_frequency_taper)
        else:
            high_frequency_taper = float(high_frequency_taper)
            if not np.isfinite(high_frequency_taper) or high_frequency_taper < 0.0:
                raise ValueError(
                    "high_frequency_taper width must be finite and non-negative"
                )
        if reconstruction == "hermite" or high_frequency_taper:
            phase_derivatives = max(1, phase_derivatives)

        period = 1.0 / df
        if damping_factor is not None:
            if damping_factor < 1.0:
                raise ValueError("damping_factor must be greater than or equal to 1")
            laplace = -np.log(float(damping_factor)) / (2.0 * np.pi * period)

        target_df = float(df)
        if f_min == 0.0:
            f_min = f_min + target_df
        dense_f_list = np.arange(f_min, f_max + target_df / 2, target_df)
        f_list = dense_f_list
        if reconstruction == "hermite":
            f_list = dense_f_list[::sample_every]
            dense_endpoint = dense_f_list[-1]
            if not np.isclose(
                f_list[-1], dense_endpoint, rtol=0.0, atol=target_df * 1.0e-8
            ):
                # Keep every Hermite solve frequency bit-for-bit on the fine
                # target grid. Appending the user-supplied f_max can differ by
                # a few ulps from np.arange's endpoint and breaks exact subset
                # checks even though the physical frequencies are equivalent.
                f_list = np.append(f_list, dense_endpoint)

        laplace = -abs(float(laplace or 0.0))
        if laplace != 0.0:
            f_list = f_list + 1j * laplace

        workflow = "forward_df" if phase_derivatives else "forward"
        super().__init__(
            name=name,
            simulation=simulation,
            workflow=workflow,
            f_list=f_list,
            outputs=JobOutputs(outputs),
            k_list=k_list,
            k_weights=k_weights,
            k_units=k_units,
        )
        self.phase_derivatives: int = phase_derivatives
        self.reconstruction: str = reconstruction
        self.sample_every: int = sample_every
        self.target_df: float = target_df
        self.high_frequency_taper: bool = high_frequency_taper
        self.interpolation_time_shift: Optional[float] = interpolation_time_shift

    @property
    def time_reconstruction(self) -> dict:
        """Return the persisted frequency-to-time reconstruction settings."""

        return {
            "method": self.reconstruction,
            "target_df": self.target_df,
            "sample_every": self.sample_every,
            "high_frequency_taper": self.high_frequency_taper,
            "interpolation_time_shift": self.interpolation_time_shift,
        }

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, project_relative: bool = False
    ) -> dict:
        """Serialize the time reconstruction settings with the solver job."""

        payload = super().to_fs(ctx, project_relative=project_relative)
        if self.phase_derivatives:
            payload["derivative_order"] = self.phase_derivatives
        if self.reconstruction == "hermite" or self.high_frequency_taper:
            payload["time_reconstruction"] = self.time_reconstruction
        return payload

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
        reconstruction = d.get("time_reconstruction") or {}
        method = str(reconstruction.get("method", "standard"))
        workflow = str(d.get("workflow", "forward"))
        phase_derivatives = (
            int(d.get("derivative_order", 1)) if workflow == "forward_df" else 0
        )
        sample_every = int(
            reconstruction.get(
                "sample_every",
                reconstruction.get(
                    "frequency_reduction", 4 if method == "hermite" else 1
                ),
            )
        )
        df = float(reconstruction.get("target_df", np.real(f_list[1] - f_list[0])))
        laplace = float(np.imag(f_list[0]))
        if method == "standard":
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
            k_list=d.get("k_list"),
            k_weights=d.get("k_weights"),
            k_units=d.get("k_units"),
            phase_derivatives=phase_derivatives,
            reconstruction=method,
            sample_every=sample_every,
            high_frequency_taper=reconstruction.get("high_frequency_taper", False),
            interpolation_time_shift=float(
                reconstruction.get("interpolation_time_shift", 0.0)
            ),
        )
        job.f_list = f_list.tolist()
        job._job_id = d.get("job_id")
        return job
