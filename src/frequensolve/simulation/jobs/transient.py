"""Direct physical-time wave propagation through Sauce's transient workflow."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from frequensolve.simulation.artifact_contract import (
    TaskResult,
    load_operation_result,
    operation_result_path,
    task_result_path,
)
from frequensolve.simulation.jobs.base import BaseJob
from frequensolve.simulation.jobs.run_state import SkipPolicy, TaskRunPlan
from frequensolve.simulation.outputs import JobOutputs
from frequensolve.simulation.time import (
    CheckpointSchedule,
    Gauss,
    Newmark,
    TimeFunction,
    TimeIntegration,
    TimeOutput,
    _fields,
    _path,
)
from frequensolve.util.class_registry import register_class

__all__ = ["TransientJob"]


@register_class
class TransientJob(BaseJob):
    """Fixed-step acoustic, elastic or coupled wave simulation.

    ``n_modes=N`` creates N independent axisymmetric acoustic tasks, for angular
    orders 0 through N-1. Omit it for the legacy single-trajectory workflow.

    Unset simulation settings are filled with Galerkin and lossless attenuation.
    Cartesian 2D/3D also supports explicitly configured GSLS attenuation.
    Supply a positive mesh ``f_adapt`` or ``time.f_max``. Source geometry and strengths remain
    in its acquisition; ``source`` supplies their shared physical-time function.
    Cartesian point loads use automatic Gaussian support; ``point_source="dirac"``
    opts out. Explicit source supports take precedence. The width uses local
    material speed and ``time.f_max`` or the inferred wavelet/mesh bandwidth.
    ``output_schedule`` selects spatial snapshots and, by default, receiver samples.
    ``trace_schedule`` optionally overrides receiver sampling. The existing
    ``TimeDomainJob`` continues to provide harmonic reconstruction.

    Restart paths identify complete rank-local checkpoint directories. Keep the
    original interval and physical inputs. Copy a checkpoint before restarting
    if the new run will write to that same directory.
    """

    frequency_independent = True
    supports_trace_packing = False
    config_key = "Time"

    def __init__(
        self,
        name,
        simulation,
        *,
        time: TimeIntegration,
        source: TimeFunction,
        point_source: str = "auto",
        outputs=None,
        n_modes: int | None = None,
        output_schedule: TimeOutput | None = None,
        trace_schedule: TimeOutput | None = None,
        output_stride: int | None = None,
        write_initial: bool | None = None,
        checkpoint: CheckpointSchedule | None = None,
        restart: str | Path | None = None,
    ):
        if not isinstance(time, TimeIntegration):
            raise TypeError("time must be TimeIntegration")
        if not isinstance(source, TimeFunction):
            raise TypeError("source must be TimeFunction")
        if output_schedule is not None and (
            output_stride is not None or write_initial is not None
        ):
            raise ValueError(
                "Pass output_schedule or output_stride/write_initial, not both"
            )
        self.n_modes = n_modes
        self.time = time
        self.source = source
        if point_source not in {"auto", "dirac"}:
            raise ValueError("point_source must be auto or dirac")
        self.point_source = point_source
        self.output_schedule = output_schedule or TimeOutput(
            stride=output_stride,
            write_initial=True if write_initial is None else write_initial,
        )
        self.trace_schedule = trace_schedule
        if trace_schedule is not None and not isinstance(trace_schedule, TimeOutput):
            raise TypeError("Expected TimeOutput for trace_schedule")
        self.checkpoint = checkpoint or CheckpointSchedule()
        if not isinstance(self.output_schedule, TimeOutput) or not isinstance(
            self.checkpoint, CheckpointSchedule
        ):
            raise TypeError("Expected TimeOutput and CheckpointSchedule")
        self.restart = None if restart is None else _path(restart, "restart")
        self._configure_simulation(simulation)
        super().__init__(name, simulation, "transient", [], JobOutputs(outputs))
        self.validate_outputs()

    def _configure_simulation(self, simulation) -> None:
        """Fill unset transient defaults on the supplied simulation, preserving overrides."""
        from frequensolve.units import _scalar_in_units

        simulation.discretization.extra.setdefault("method", "Galerkin")
        # Sauce infers the required P/S parts from output and receiver fields.
        # Its legacy time-domain default is displacement; select velocity here
        # without enabling projection for jobs that do not request it.
        projection = simulation.extra.setdefault("helmholtz_projection", {})
        if isinstance(projection, dict) and not (
            projection.get("enabled")
            or projection.get("parts")
            or projection.get("part")
        ):
            projection.setdefault("field", "velocity")
        model = simulation.model
        if model is not None and model.attenuation_model is None:
            # An explicit attenuation reference/configuration is not an unset law.
            if model.reference_frequency is None and not model._attenuation_extra:
                model.attenuation_model = "none"
        adapt = simulation.mesh.adapt
        if adapt is None and self.time.f_max is not None:
            simulation.mesh.set_adapt(
                elems_per_wave=2, order=2, f_adapt=self.time.f_max
            )
        elif adapt is not None:
            if adapt.f_adapt is None and self.time.f_max is not None:
                adapt.f_adapt = self.time.f_max
            elif adapt.f_adapt is not None:
                adapt.f_adapt = _scalar_in_units(adapt.f_adapt, "Hz", "f_adapt")

    @property
    def n_tasks(self):
        return 1 if self.n_modes is None else self.n_modes

    def mode_for_task(self, task):
        if (
            isinstance(task, bool)
            or not isinstance(task, Integral)
            or not 1 <= task <= self.n_tasks
        ):
            raise IndexError(f"Task {task} is outside 1..{self.n_tasks}")
        return (
            task - 1
            if self.n_modes is not None
            else (self.simulation.azimuthal_mode or 0)
        )

    def task_manifest_file(self, task=1, *, result_path=None):
        self.mode_for_task(task)
        root = self._result_path if result_path is None else result_path
        return (
            task_result_path(root, task)
            if self.n_modes is not None
            else operation_result_path(root, "transient")
        )

    def task_fingerprint_payload(self, task):
        payload = super().task_fingerprint_payload(task)
        if self.n_modes is not None:
            payload["azimuthal_mode"] = self.mode_for_task(task)
        return payload

    def _time_payload(self) -> dict:
        payload = {
            **self.time.to_fs(),
            "source": self.source.to_fs(),
            "output": self.output_schedule.to_fs(),
            "checkpoint": self.checkpoint.to_fs(),
        }
        if self.trace_schedule is not None:
            payload["trace_output"] = self.trace_schedule.to_fs()
        if self.n_modes is not None:
            payload["n_modes"] = self.n_modes
        if self.restart is not None:
            payload["restart"] = {"path": self.restart}
        payload["point_source"] = self._point_source_payload()
        return payload

    def _point_source_payload(self) -> dict:
        """Resolve bandwidth here; Sauce samples physical speed on the mapped source."""
        sim = self.simulation
        if (
            self.point_source == "dirac"
            or sim.axisymmetric
            or sim.physics.endswith("_axisym")
        ):
            return {"type": "dirac"}
        if hasattr(self, "_imported_point_source"):
            return copy.deepcopy(self._imported_point_source)
        if self.time.f_max is not None:
            frequency = self.time.f_max
        else:
            adapt = sim.mesh.adapt
            frequency = float(adapt.f_adapt or 0) if adapt is not None else 0.0
            source = self.source
            if source.kind == "ricker":
                frequency = max(frequency, 3 * source.frequency)
            elif source.kind == "sine":
                frequency = max(frequency, source.frequency)
            elif source.kind == "gaussian":
                frequency = max(
                    frequency, np.sqrt(2 * np.log(1000)) / (2 * np.pi * source.width)
                )
            elif source.kind == "sampled":
                samples = np.asarray(source.wavelet.samples)
                spectrum = np.abs(np.fft.rfft(samples, n=max(256, 4 * len(samples))))
                frequencies = np.fft.rfftfreq(
                    2 * (len(spectrum) - 1), source.wavelet.dt
                )
                active = np.flatnonzero(spectrum > 1e-3 * spectrum.max())
                if active.size:
                    frequency = max(frequency, float(frequencies[active[-1]]))
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError(
                "Automatic point sources require time.f_max or a positive source/mesh bandwidth"
            )
        return {"type": "gaussian", "f_max": frequency, "amplitude_ratio": 0.98}

    def to_fs(self, ctx=None, *, project_relative=False) -> dict:
        payload = super().to_fs(ctx, project_relative=project_relative)
        payload["Time"] = self._time_payload()
        if selection := self.time.selection_payload():
            payload["time_selection"] = selection
        return payload

    @classmethod
    def from_fs(cls, data, base_path=None, project_path=None):
        if data.get("workflow") != "transient" or any(
            k in data
            for k in ("f_list", "k_list", "k_weights", "k_units", "time_reconstruction")
        ):
            raise ValueError(
                "TransientJob requires workflow transient without frequency/wavenumber settings"
            )
        config = _fields(
            data["Time"],
            {
                "schema",
                "n_modes",
                "method",
                "start",
                "end",
                "time_step",
                "newmark",
                "gauss",
                "source",
                "output",
                "trace_output",
                "checkpoint",
                "restart",
                "point_source",
            },
            "Time",
        )
        n_modes = config.pop("n_modes", None)
        point_source_config = config.pop("point_source", {"type": "dirac"})
        point_source_config = _fields(
            point_source_config,
            {"type", "f_max", "amplitude_ratio"},
            "Time.point_source",
        )
        if point_source_config.get("type") not in {"gaussian", "dirac"}:
            raise ValueError("Unsupported Time.point_source type")
        if point_source_config["type"] == "gaussian":
            frequency = point_source_config.get("f_max", 0)
            ratio = point_source_config.get("amplitude_ratio", 0.98)
            if not np.isfinite(frequency) or frequency <= 0 or not 0 < ratio < 1:
                raise ValueError(
                    "Gaussian point_source requires f_max > 0 and 0 < amplitude_ratio < 1"
                )
        source = TimeFunction.from_fs(config.pop("source", {}))
        output = TimeOutput.from_fs(config.pop("output", {}))
        trace_output = config.pop("trace_output", None)
        trace_schedule = (
            None if trace_output is None else TimeOutput.from_fs(trace_output)
        )
        checkpoint = CheckpointSchedule.from_fs(config.pop("checkpoint", {}))
        restart = _fields(config.pop("restart", {}), {"path"}, "Time.restart")
        sim = cls._load_simulation_for_job(
            data["simulation"],
            base_path=base_path,
            project_path=project_path or data.get("project_path"),
            source_project=data.get("project_path"),
        )
        job = cls(
            data["name"],
            sim,
            time=TimeIntegration.from_fs(config, selection=data.get("time_selection")),
            source=source,
            point_source=(
                "auto" if point_source_config["type"] == "gaussian" else "dirac"
            ),
            n_modes=n_modes,
            outputs=JobOutputs.from_fs(data.get("Outputs")),
            output_schedule=output,
            trace_schedule=trace_schedule,
            checkpoint=checkpoint,
            restart=restart.get("path"),
        )
        job._job_id = data.get("job_id")
        job._imported_point_source = copy.deepcopy(point_source_config)
        return job

    def validate_outputs(self) -> None:
        """Validate the supported physical and output boundary before export."""
        self.output_schedule.validate(self.time)
        if self.trace_schedule is not None:
            self.trace_schedule.validate(self.time)
        sim = self.simulation
        sim._validate_azimuthal_mode()
        if self.n_modes is not None:
            if (
                isinstance(self.n_modes, bool)
                or not isinstance(self.n_modes, Integral)
                or not 1 <= self.n_modes <= 1000000
            ):
                raise ValueError(
                    "n_modes must be a positive integer no greater than 1000000"
                )
            self.n_modes = int(self.n_modes)
            if sim.azimuthal_mode is not None:
                raise ValueError(
                    "Set n_modes on TransientJob instead of simulation.azimuthal_mode"
                )
            if sim.dimension != 2 or sim.physics != "acoustic_axisym":
                raise ValueError("n_modes requires axisymmetric 2D acoustic physics")
        angular_order = (
            self.n_modes - 1 if self.n_modes is not None else sim.azimuthal_mode
        )
        physics = sim.physics.removesuffix("_axisym")
        axisymmetric = sim.axisymmetric or sim.physics.endswith("_axisym")
        reduced = [
            s
            for s in getattr(sim.model, "surfaces", []) or []
            if getattr(s, "is_reduced", False)
        ]
        if reduced:
            if sim.dimension not in {2, 3} or physics not in {"elastic", "coupled"}:
                raise ValueError(
                    "Native reduced fractures require Cartesian 2D/3D or axisymmetric 2D elastic or coupled physics"
                )
            if self.time.method != Newmark():
                raise ValueError(
                    "Native reduced fractures require Newmark(beta=0.25, gamma=0.5)"
                )
        if (
            physics not in {"acoustic", "elastic", "coupled"}
            or sim.dimension not in {2, 3}
            or (axisymmetric and sim.dimension != 2)
        ):
            raise ValueError(
                "TransientJob supports Cartesian 2D/3D or axisymmetric 2D acoustic, elastic and coupled physics"
            )
        if sim.discretization.to_fs().get("method", "").lower() != "galerkin":
            raise ValueError("TransientJob requires Discretization(method='Galerkin')")
        attenuation = _transient_attenuation_model(sim.model)
        if attenuation == "gsls":
            if axisymmetric:
                raise ValueError(
                    "gsls attenuation requires Cartesian 2D/3D coordinates"
                )
            if (
                not isinstance(self.time.method, Gauss)
                and self.time.method != Newmark()
            ):
                raise ValueError(
                    "gsls attenuation requires average-acceleration Newmark or Gauss"
                )
            # PML with gsls is composed in the solver (Sauce gsls-pml-qpqs change); no SDK gate.
        adapt = getattr(sim.mesh, "adapt", None)
        frequency = getattr(adapt, "f_adapt", None)
        if frequency is None or not np.isfinite(frequency) or frequency <= 0:
            raise ValueError(
                "TransientJob requires a positive mesh bandwidth: set "
                "TimeIntegration(..., f_max=<Hz>) or mesh.set_adapt(..., f_adapt=<Hz>)"
            )
        projection = _transient_projection(sim)
        if angular_order and projection:
            raise ValueError(
                "Azimuthal acoustic modes do not support Helmholtz projection"
            )
        projection = projection or sim.extra.get("helmholtz_projection", {}).get(
            "field", "velocity"
        )
        for output in self.outputs.vtk:
            _validate_paraview(
                output.to_fs(), projection=projection, azimuthal_mode=angular_order
            )
        if self.k_list is not None or self.f_list:
            raise ValueError("TransientJob does not accept frequencies or wavenumbers")
        for bc in sim.BCs:
            payload = bc.to_fs()
            if any(c.startswith("pml") for c in payload.get("conditions", [])):
                if (
                    not isinstance(self.time.method, Gauss)
                    and self.time.method != Newmark()
                ):
                    raise ValueError(
                        "Transient PML requires average-acceleration Newmark or Gauss"
                    )
        acquisition = getattr(sim, "acquisition", None)
        if acquisition is not None:
            if (
                acquisition.source_signature is not None
                or acquisition.boundary_loadings
            ):
                raise ValueError(
                    "TransientJob uses TimeFunction; spectral signatures and boundary loadings are unsupported"
                )
            count = acquisition.known_source_field_count()
            if (
                acquisition.max_batch is not None
                and count is not None
                and count > acquisition.max_batch
            ):
                raise ValueError(
                    "All transient encoded sources must fit in one acquisition batch"
                )
            geometry = acquisition.source_geometry
            kinds = set()
            if geometry is not None:
                kinds.add(geometry.kind)
                try:
                    kinds.update(
                        geometry.point(i).kind or geometry.kind
                        for i in range(geometry.point_count or 0)
                    )
                except ValueError:
                    pass  # External source catalogs are checked by Sauce after loading.
            if angular_order and kinds - {
                None,
                "scalar",
                "monopole",
                "volume_injection",
                "pressure_monopole",
                "isotropic_moment",
            }:
                raise ValueError(
                    "Azimuthal acoustic modes require scalar or volume-injection sources"
                )
            if self.source.kind == "step" and (
                "volume_injection" in kinds
                or (physics != "elastic" and kinds & {"isotropic_moment", "monopole"})
            ):
                raise ValueError(
                    "Acoustic moment and volume-injection sources require smooth time functions"
                )
            if self.restart is None:
                if "volume_injection" in kinds:
                    self.source.require_quiet_start(self.time.start)
                    self.source.require_quiet_start(self.time.start, derivative=1)
                if physics != "elastic" and kinds & {"isotropic_moment", "monopole"}:
                    self.source.require_quiet_start(self.time.start, derivative=2)
                if physics == "elastic" or kinds - {
                    "isotropic_moment",
                    "monopole",
                    "volume_injection",
                    None,
                }:
                    self.source.require_quiet_start(self.time.start)
            for group in acquisition.receiver_groups:
                _validate_fields(
                    group.to_fs(),
                    projection=projection,
                    azimuthal_mode=angular_order,
                )
        for field in self.outputs.wavefields:
            _validate_fields(
                field.to_fs(), projection=projection, azimuthal_mode=angular_order
            )

    def _restart_path(self) -> Path | None:
        if self.restart is None:
            return None
        path = Path(self.restart).expanduser()
        return path if path.is_absolute() else self._result_path / path

    def restart_step(self, task=1) -> int:
        """Validate the rank-zero time-integration header and return its completed step."""
        path = self._restart_path()
        if path is None:
            return 0
        mode = self.mode_for_task(task)
        if self.n_modes is not None:
            path = path / "modes" / f"mode_{mode:06d}"
            if int((path / "azimuthal_mode_R00000.txt").read_text()) != mode:
                raise ValueError("Checkpoint azimuthal_mode differs from task")
        with h5py.File(path / "newmark_manifest_R00000.h5", "r") as h5:
            layout = np.asarray(h5["layout"][()]).reshape(-1)
            scheme = np.asarray(h5["scheme"][()]).reshape(-1)
            theta_0 = (
                float(np.asarray(h5["theta_0"][()]).item()) if "theta_0" in h5 else 0.0
            )
        if (
            layout.shape != (4,)
            or scheme.shape != (6,)
            or layout[0] != 2
            or layout[1] != 0
            or layout[2] != (3 if isinstance(self.time.method, Gauss) else 2)
        ):
            raise ValueError("Invalid time-integration restart manifest")
        expected_theta = self.time.theta_0
        if not np.isclose(
            theta_0, expected_theta, rtol=64 * np.finfo(float).eps, atol=0
        ):
            raise ValueError("Restart theta_0 differs from this job")
        step = int(layout[3])
        expected = [
            self.time.start + step * self.time.dt,
            self.time.start,
            self.time.end,
            self.time.dt,
            self.time.method.beta if isinstance(self.time.method, Newmark) else 0,
            self.time.method.gamma if isinstance(self.time.method, Newmark) else 0,
        ]
        if not 0 <= step <= self.time.steps or not np.allclose(
            scheme, expected, rtol=64 * np.finfo(float).eps, atol=0
        ):
            raise ValueError(
                "Restart interval, step or integration method differs from this job"
            )
        return step

    def _external_input_fingerprint(self):
        try:
            return self._refresh_external_input_fingerprint(require_inputs=True)
        except FileNotFoundError:
            # A removed checkpoint copy cannot change committed output, so the
            # digest recorded at save keeps identifying the resumed run.
            recorded = getattr(self, "_external_input_fingerprint_cache", None)
            if recorded is None:
                raise
            return dict(recorded)

    def _refresh_external_input_fingerprint_for_save(self) -> None:
        try:
            self._external_input_fingerprint()
        except FileNotFoundError:
            self._external_input_fingerprint_cache = None

    def _input_fingerprint_payload(self) -> dict:
        path = self._restart_path()
        if path is None:
            return {}
        if self.checkpoint.stride:
            target = Path(self.checkpoint.path)
            if not target.is_absolute():
                target = self._result_path / target
            if target.resolve() == path.resolve():
                raise ValueError(
                    "Copy restart inputs or choose another checkpoint path; restart inputs must remain immutable"
                )
        for task in range(1, self.n_tasks + 1):
            self.restart_step(task)
        return {"restart": self._path_content_fingerprint(path)}

    def with_restart(
        self, path: str | Path, *, name: str | None = None
    ) -> TransientJob:
        """Return a job resuming this interval from an existing checkpoint copy."""
        path = Path(_path(path, "restart")).expanduser()
        if not path.is_absolute():
            path = self._result_path / path
        if not path.is_dir():
            raise FileNotFoundError(path)
        job = copy.copy(self)
        job.outputs = copy.deepcopy(self.outputs)
        job.restart = str(path.resolve())
        if name is not None:
            job.name = name
            job._file = None
            job._job_id = None
        job._external_input_fingerprint_cache = None
        job.validate_outputs()
        job._input_fingerprint_payload()
        return job

    @property
    def result_manifest_file(self) -> Path:
        return self.task_manifest_file(1)

    @staticmethod
    def _payload_for_layout(payload, *, source, target, source_projects=()):
        data = BaseJob._payload_for_layout(
            payload, source=source, target=target, source_projects=source_projects
        )
        if payload.get("Time", {}).get("restart"):
            data["Time"]["restart"]["path"] = str(
                target.job_dir / "inputs" / "transient_restart"
            )
        return data

    def remote_input_files(self, remote_project):
        files = super().remote_input_files(remote_project)
        restart = self._restart_path()
        if restart is not None:
            for task in range(1, self.n_tasks + 1):
                self.restart_step(task)
            target = (
                self._saved_layout().with_project(remote_project).job_dir
                / "inputs"
                / "transient_restart"
            )
            files.extend(
                (path, target / path.relative_to(restart))
                for path in sorted(restart.rglob("*"))
                if path.is_file()
            )
        return files

    def operation_result(self, *, result_path=None, task=1):
        """Read a committed result and require the current saved input digests."""
        saved_inputs = (
            json.loads(self.job_file.read_text())
            .get("artifact_contract", {})
            .get("external_inputs")
        )
        if saved_inputs != self._external_input_fingerprint():
            raise ValueError("Restart inputs changed since the job was saved")
        root = self._result_path if result_path is None else result_path
        mode = self.mode_for_task(task)
        if self.n_modes is None:
            result = load_operation_result(root, "transient")
        else:
            result = TaskResult.read(
                self.task_manifest_file(task, result_path=root), result_path=root
            )
            partition = result.partition
            if (
                partition.task != task
                or partition.task_count != self.n_tasks
                or partition.azimuthal_mode != mode
            ):
                raise ValueError(
                    "Transient result angular partition does not match this task"
                )
        expected = self._artifact_contract_fingerprints()
        candidates = [expected] if expected is not None else []
        candidates.extend(self.downloaded_task_fingerprints())
        if not result.successful or not any(
            all(
                k in p and result.fingerprints.get(k) == p[k]
                for k in ("job", "simulation", "outputs")
            )
            for p in candidates
        ):
            raise ValueError(
                "Transient result is failed or does not match the saved job inputs"
            )
        return result

    @property
    def traces(self):
        from frequensolve.seismic.transient import TransientOutputHandle

        return TransientOutputHandle(self, "simulated_traces")

    @property
    def wavefields(self):
        from frequensolve.seismic.transient import TransientOutputHandle

        return TransientOutputHandle(self, "sampled_wavefield")

    def results_exist(self) -> bool:
        return all(
            self._task_results_exist(task) for task in range(1, self.n_tasks + 1)
        )

    def _task_results_exist(self, task) -> bool:
        try:
            result = self.operation_result(task=task)
            for artifact in result.artifacts:
                if (
                    not artifact.path.is_file()
                    or artifact.path.stat().st_size != artifact.bytes
                ):
                    return False
            expected_series = {
                f"visualization-series:{output.name}"
                for output in self.outputs.vtk
                if output.extra.get("execute_on", "solve") != "none"
            }
            if not expected_series.issubset({a.id for a in result.artifacts}):
                return False
            expected_roles = set()
            if (
                self.outputs.traces is not None
                and self.simulation.acquisition is not None
                and self.simulation.acquisition.receiver_groups
            ):
                expected_roles.add("simulated_traces")
            if self.outputs.wavefields:
                expected_roles.add("sampled_wavefield")
            actual_roles = {
                a.role for a in result.artifacts if a.schema == "fs-time-step-1"
            }
            if not expected_roles.issubset(actual_roles):
                return False
            for handle in (self.traces, self.wavefields):
                if any(
                    a.schema == "fs-time-step-1" and a.role == handle.role
                    for a in result.artifacts
                ):
                    data = handle.open(
                        mode=(
                            self.mode_for_task(task)
                            if self.n_modes is not None
                            else None
                        )
                    )
                    if handle.role == "sampled_wavefield" and not {
                        f.name for f in self.outputs.wavefields
                    }.issubset(data.groups):
                        return False
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def is_task_current(self, task, *, state=None, manifest=None) -> bool:
        state = self.run_state() if state is None else state
        if self.n_modes is not None:
            return 1 <= task <= self.n_tasks and self._task_results_exist(task)
        return (
            task == 1
            and state.get("fingerprint") == self.fingerprint()
            and state.get("status") in {"completed", "skipped"}
            and self.results_exist()
        )

    def current_tasks(self):
        return [
            task for task in range(1, self.n_tasks + 1) if self.is_task_current(task)
        ]

    def is_run_current(self):
        return len(self.current_tasks()) == self.n_tasks

    def task_policy_fingerprint(self, task, policy=None):
        return self.task_fingerprint(task)

    def plan_tasks(
        self,
        *,
        skip_policy=None,
        reuse=False,
        force=False,
        apply=False,
        residual=None,
        ignore_solver_options=None,
    ):
        policy = SkipPolicy.from_value(
            skip_policy,
            residual=residual,
            ignore_solver_options=ignore_solver_options,
            reuse=False,
        )
        if policy.accept_failed or policy.ignore_solver_options:
            raise ValueError(
                "TransientJob supports strict or forced reruns; partial trajectories cannot be reused"
            )
        current = [] if force or policy.force else self.current_tasks()
        pending = [task for task in range(1, self.n_tasks + 1) if task not in current]
        removed = False
        if apply:
            for task in pending:
                path = self.task_manifest_file(task)
                path.with_name("progress.json").unlink(missing_ok=True)
                if path.exists():
                    path.unlink()
                    removed = True
        return TaskRunPlan(
            pending_indices=[task - 1 for task in pending],
            current_tasks=current,
            strict_current_tasks=current,
            reused_tasks=[],
            accepted_tasks=[],
            accepted_failed_tasks=[],
            skipped_task_records=[
                {
                    "task": task,
                    "status": "current",
                    "fingerprint": self.task_fingerprint(task),
                }
                for task in current
            ],
            skip_policy=policy,
            removed_stale_outputs=removed,
        )

    def task_run_plan(self, **kwargs):
        return self.plan_tasks(apply=True, **kwargs)

    def write_run_state(self, status="completed", **extra):
        if self.n_modes is not None:
            return self._write_modal_run_state(status, **extra)
        complete = status in {"completed", "skipped"} and self.results_exist()
        if status in {"completed", "skipped"} and not complete:
            status = "failed"
        failed = status in {"failed", "cancelled", "timeout"}
        payload = {
            "schema": "frequensolve-python-run-1",
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": self.fingerprint(),
            "fingerprint_payload": self.fingerprint_payload(),
            "tasks": [
                {
                    "task": 1,
                    "frequency": None,
                    "status": (
                        "succeeded" if complete else ("failed" if failed else "not_run")
                    ),
                    "complete": complete,
                    "fingerprint": self.task_fingerprint(1),
                    "path": str(self.result_manifest_file),
                }
            ],
            "task_summary": {
                "total": 1,
                "complete": int(complete),
                "succeeded": int(complete),
                "failed": int(failed),
                "not_run": int(not complete and not failed),
            },
        }
        records = extra.get("tasks") or []
        if records:
            payload["tasks"][0] = {**records[0], **payload["tasks"][0]}
        payload.update(
            {
                k: v
                for k, v in extra.items()
                if k not in {"status", "tasks", "fingerprint", "task_summary"}
            }
        )
        self._write_json_file(self.run_state_file, payload)
        return self.run_state_file

    def _write_modal_run_state(self, status, **extra):
        previous = {
            r.get("task", r.get("task_id", -1) + 1): r for r in extra.pop("tasks", [])
        }
        tasks = []
        for task in range(1, self.n_tasks + 1):
            complete = self._task_results_exist(task)
            record = previous.get(task, {})
            failed = not complete and (
                record.get("status") in {"error", "failed", "cancelled", "timeout"}
            )
            tasks.append(
                {
                    **record,
                    "task": task,
                    "azimuthal_mode": task - 1,
                    "frequency": None,
                    "status": (
                        "succeeded" if complete else ("failed" if failed else "not_run")
                    ),
                    "complete": complete,
                    "fingerprint": self.task_fingerprint(task),
                    "path": str(self.task_manifest_file(task)),
                }
            )
        count = sum(t["complete"] for t in tasks)
        failed = sum(t["status"] == "failed" for t in tasks)
        if status in {"completed", "skipped"} and count != self.n_tasks:
            status = "failed"
        payload = {
            **extra,
            "schema": "frequensolve-python-run-1",
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": self.fingerprint(),
            "fingerprint_payload": self.fingerprint_payload(),
            "tasks": tasks,
            "task_summary": {
                "total": self.n_tasks,
                "complete": count,
                "succeeded": count,
                "failed": failed,
                "not_run": self.n_tasks - count - failed,
            },
        }
        self._write_json_file(self.run_state_file, payload)
        return self.run_state_file

    @property
    def results(self):
        return (
            {
                task - 1: self.operation_result(task=task)
                for task in range(1, self.n_tasks + 1)
            }
            if self.n_modes is not None
            else self.operation_result()
        )

    def frequency_status(self):
        """Report trajectories through the common task-status API."""
        if self.n_modes is not None:
            run = self.run_state()
            state = (
                {row["task"]: row for row in run.get("tasks", [])}
                if run.get("fingerprint") == self.fingerprint()
                else {}
            )
            rows = []
            for task in range(1, self.n_tasks + 1):
                current = self.is_task_current(task)
                metadata = state.get(task, {})
                status = "succeeded" if current else metadata.get("status", "not_run")
                if not current and status in {"succeeded", "skipped", "complete"}:
                    status = "not_run"
                rows.append(
                    dict(
                        task=task,
                        azimuthal_mode=task - 1,
                        frequency=None,
                        status=status,
                        current=current,
                        trace_exists=current,
                        trace_file=None,
                        duration_seconds=metadata.get("duration_seconds"),
                        metadata=metadata,
                    )
                )
            return rows
        current = self.is_run_current()
        state = self.run_state()
        metadata = {}
        status = "succeeded" if current else "not_run"
        if state.get("fingerprint") == self.fingerprint():
            records = state.get("tasks") or []
            metadata = records[0] if records else {}
            if not current and state.get("status") in {
                "failed",
                "cancelled",
                "timeout",
            }:
                status = "failed"
        return [
            {
                "task": 1,
                "frequency": None,
                "status": status,
                "current": current,
                "trace_exists": current,
                "trace_file": None,
                "duration_seconds": metadata.get("duration_seconds"),
                "metadata": metadata,
            }
        ]


_FIELD_NAME_KEYS = frozenset(
    {"field", "quantity", "component", "components", "operator", "expression"}
)


def _validate_fields(value, *, projection=None, azimuthal_mode=None):
    """Reject harmonic-only receiver operations, including nested expressions."""
    if isinstance(value, Mapping):
        if value.get("_type") == "ReceiverNode":
            for component in value.get("components", []):
                if (
                    component.get("field")
                    in {
                        "displacement",
                        "velocity",
                        "p_velocity",
                        "s_velocity",
                        "p_displacement",
                        "s_displacement",
                    }
                    and component.get("direction") is None
                ):
                    raise ValueError(
                        "Vector receiver components require a measurement direction; use direction=[1, 0] in 2D"
                    )
        if any(value.get(key) is not None for key in ("transfer", "response")):
            raise ValueError("Transient receiver transfer functions are unsupported")
        # Only field-name strings are checked; nested mappings are reached by
        # recursion, so component names and units never match. Receiver
        # expression field nodes carry their field under "name"; ParaView
        # items use "name" as a label beside "field".
        names = (
            [value["name"]]
            if value.get("kind") == "field"
            and "field" not in value
            and isinstance(value.get("name"), str)
            else []
        )
        for key, child in value.items():
            if key not in _FIELD_NAME_KEYS:
                continue
            if key == "field" and isinstance(child, str) and projection:
                name = child.removeprefix("elastic:")
                allowed = {projection, f"p_{projection}", f"s_{projection}"}
                if projection == "displacement" and (
                    name == "velocity" or name.startswith("velocity_")
                ):
                    raise ValueError(
                        "Native velocity outputs require velocity Helmholtz projection"
                    )
                if name.startswith(
                    ("p_velocity", "s_velocity", "p_displacement", "s_displacement")
                ) and not any(name == f or name.startswith(f + "_") for f in allowed):
                    raise ValueError(
                        "Transient Helmholtz outputs must match helmholtz_projection.field"
                    )
            if isinstance(child, str):
                names.append(child)
            elif isinstance(child, (list, tuple)):
                names.extend(item for item in child if isinstance(item, str))
        for text in (name.lower() for name in names):
            if azimuthal_mode and "velocity" in text:
                raise ValueError(
                    "Positive azimuthal modes support pressure, not vector velocity output"
                )
            if any(word in text for word in ("acceleration", "frequency", "helmholtz")):
                raise ValueError(
                    "Transient receivers require primary or constitutive fields; harmonic velocity/acceleration and spectral recovery are unsupported"
                )
        for child in value.values():
            _validate_fields(
                child, projection=projection, azimuthal_mode=azimuthal_mode
            )
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_fields(
                child, projection=projection, azimuthal_mode=azimuthal_mode
            )


def _validate_paraview(output, *, projection=None, azimuthal_mode=None):
    """Validate native time snapshots of fields, material properties and mesh info."""
    if output.get("writer", {}).get("format", "vtu") not in {"vtu", "xdmf", "xmf"}:
        raise ValueError("Transient ParaView output supports VTU and XDMF only")
    if output.get("target", {}).get("kind", "volume") not in {"volume", "surface"}:
        raise ValueError(
            "Transient ParaView output requires a volume or surface target"
        )
    if output.get("source", {}).get("kind", "solution") != "solution":
        raise ValueError("Transient ParaView output requires source.kind='solution'")
    if output.get("post_process"):
        raise ValueError("Transient ParaView output does not support post_process")
    if output.get("execute_on", "solve") not in {"solve", "final", "initial", "none"}:
        raise ValueError("Transient ParaView output uses the TimeOutput schedule")
    items = output.get("items", [])
    fields = output.get("fields", [])
    if not items and not fields and not output.get("properties"):
        raise ValueError(
            "Transient ParaView output requires explicit fields or properties"
        )
    for item in items:
        if any(part in {"im", "imag", "imaginary"} for part in item.get("parts") or []):
            raise ValueError(
                "Transient ParaView fields are real; imaginary output is unsupported"
            )
    _validate_fields(items, projection=projection, azimuthal_mode=azimuthal_mode)
    for field in fields:
        _validate_fields(
            {"field": field}, projection=projection, azimuthal_mode=azimuthal_mode
        )


def _transient_attenuation_model(model) -> str | None:
    """Return the transient attenuation model, rejecting frequency-domain laws."""
    if model is None:
        return None
    name = getattr(model, "attenuation_model", None)
    normalized = name.strip().casefold() if isinstance(name, str) else name
    if normalized not in {"none", "gsls"}:
        shown = "kjartansson (solver default)" if name is None else repr(name)
        raise ValueError(
            f"TransientJob does not support model attenuation {shown}; "
            "transient attenuation needs model.attenuation_model='gsls' "
            "(or 'none' for lossless)"
        )
    if normalized == "gsls":
        build = getattr(model, "_attenuation_config", None)
        config = build() if callable(build) else None
        if config is not None:
            config.validate()
    return normalized


def _transient_projection(simulation):
    """Validate the native elastic Helmholtz configuration used by time outputs."""
    config = simulation.extra.get("helmholtz_projection", {})
    if not isinstance(config, Mapping):
        raise ValueError("helmholtz_projection must be a mapping")
    if not config:
        return None
    if "part" in config:
        raise ValueError(
            "TransientJob requires helmholtz_projection.parts to preserve the native field"
        )
    field = config.get("field", "displacement")
    if field not in {"displacement", "velocity"}:
        raise ValueError("helmholtz_projection.field must be displacement or velocity")
    if (
        field == "displacement"
        and simulation.physics.removesuffix("_axisym") != "elastic"
    ):
        raise ValueError(
            "Transient displacement Helmholtz projection requires standalone elastic physics"
        )
    if config.get("s_projection", "complement") != "complement":
        raise ValueError(
            "Transient Helmholtz output supports s_projection='complement' only"
        )
    if not (config.get("enabled", False) or config.get("parts")):
        return None
    parts = config.get("parts")
    if (
        not isinstance(parts, (list, tuple))
        or not parts
        or len(set(parts)) != len(parts)
        or set(parts) - {"p", "s"}
    ):
        raise ValueError("helmholtz_projection.parts must select unique p/s parts")
    return field
