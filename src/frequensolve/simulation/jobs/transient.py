"""Direct physical-time wave propagation through Sauce's transient workflow."""

from __future__ import annotations

import copy
import json
import h5py
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

from frequensolve.simulation.artifact_contract import (
    load_operation_result,
    operation_result_path,
)
from frequensolve.simulation.jobs.base import BaseJob
from frequensolve.simulation.jobs.run_state import SkipPolicy, TaskRunPlan
from frequensolve.simulation.outputs import JobOutputs
from frequensolve.simulation.time import (
    CheckpointSchedule,
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
    """A single fixed-step acoustic, elastic or coupled wave simulation.

    The simulation must explicitly select lossless Galerkin physics in Cartesian
    2D/3D or axisymmetric 2D and a positive mesh ``f_adapt``. Source geometry and strengths remain
    in its acquisition; ``source`` supplies their shared physical-time function.
    ``output_schedule`` selects both receiver and wavefield samples. The existing
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
        outputs=None,
        output_schedule: TimeOutput | None = None,
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
        self.time = time
        self.source = source
        self.output_schedule = output_schedule or TimeOutput(
            stride=output_stride,
            write_initial=True if write_initial is None else write_initial,
        )
        self.checkpoint = checkpoint or CheckpointSchedule()
        if not isinstance(self.output_schedule, TimeOutput) or not isinstance(
            self.checkpoint, CheckpointSchedule
        ):
            raise TypeError("Expected TimeOutput and CheckpointSchedule")
        self.restart = None if restart is None else _path(restart, "restart")
        super().__init__(name, simulation, "transient", [], JobOutputs(outputs))
        self.validate_outputs()

    def _time_payload(self) -> dict:
        payload = {
            **self.time.to_fs(),
            "source": self.source.to_fs(),
            "output": self.output_schedule.to_fs(),
            "checkpoint": self.checkpoint.to_fs(),
        }
        if self.restart is not None:
            payload["restart"] = {"path": self.restart}
        return payload

    def to_fs(self, ctx=None, *, project_relative=False) -> dict:
        payload = super().to_fs(ctx, project_relative=project_relative)
        payload["Time"] = self._time_payload()
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
                "method",
                "start",
                "end",
                "time_step",
                "newmark",
                "source",
                "output",
                "checkpoint",
                "restart",
            },
            "Time",
        )
        source = TimeFunction.from_fs(config.pop("source", {}))
        output = TimeOutput.from_fs(config.pop("output", {}))
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
            time=TimeIntegration.from_fs(config),
            source=source,
            outputs=JobOutputs.from_fs(data.get("Outputs")),
            output_schedule=output,
            checkpoint=checkpoint,
            restart=restart.get("path"),
        )
        job._job_id = data.get("job_id")
        return job

    def validate_outputs(self) -> None:
        """Validate the supported physical and output boundary before export."""
        self.output_schedule.validate(self.time)
        sim = self.simulation
        physics = sim.physics.removesuffix("_axisym")
        axisymmetric = sim.axisymmetric or sim.physics.endswith("_axisym")
        reduced = [s for s in getattr(sim.model, "surfaces", []) or [] if getattr(s, "is_reduced", False)]
        if reduced:
            if sim.dimension not in {2, 3} or physics not in {"elastic", "coupled"}:
                raise ValueError("Native reduced fractures require Cartesian 2D/3D or axisymmetric 2D elastic or coupled physics")
            if self.time.method != Newmark():
                raise ValueError("Native reduced fractures require Newmark(beta=0.25, gamma=0.5)")
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
        if (
            sim.model is not None
            and getattr(sim.model, "attenuation_model", None) != "none"
        ):
            raise ValueError("TransientJob requires model.attenuation_model='none'")
        adapt = getattr(sim.mesh, "adapt", None)
        frequency = getattr(adapt, "f_adapt", None)
        if frequency is None or not np.isfinite(frequency) or frequency <= 0:
            raise ValueError(
                "TransientJob requires mesh.set_adapt(..., f_adapt=<positive Hz>)"
            )
        projection = _transient_projection(sim)
        for output in self.outputs.vtk:
            _validate_paraview(output.to_fs(), projection=projection)
        if self.k_list is not None or self.f_list:
            raise ValueError("TransientJob does not accept frequencies or wavenumbers")
        for bc in sim.BCs:
            payload = bc.to_fs()
            if any(c.startswith("pml") for c in payload.get("conditions", [])):
                if self.time.method != Newmark():
                    raise ValueError(
                        "Transient PML requires Newmark(beta=0.25, gamma=0.5)"
                    )
                if payload.get("pml_cfs_frequency", 0) != 0:
                    raise ValueError("Transient PML requires pml_cfs_frequency=0")
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
            if self.source.kind == "step" and (
                "volume_injection" in kinds
                or (sim.physics != "elastic" and kinds & {"scalar", "monopole"})
            ):
                raise ValueError(
                    "Acoustic moment and volume-injection sources require smooth time functions"
                )
            if self.restart is None:
                if "volume_injection" in kinds:
                    self.source.require_quiet_start(self.time.start)
                    self.source.require_quiet_start(self.time.start, derivative=1)
                if physics != "elastic" and kinds & {"scalar", "monopole"}:
                    self.source.require_quiet_start(self.time.start, derivative=2)
                if physics == "elastic" or kinds - {
                    "scalar",
                    "monopole",
                    "volume_injection",
                    None,
                }:
                    self.source.require_quiet_start(self.time.start)
            for group in acquisition.receiver_groups:
                _validate_fields(group.to_fs(), projection=projection)
        for field in self.outputs.wavefields:
            _validate_fields(field.to_fs(), projection=projection)

    def _restart_path(self) -> Path | None:
        if self.restart is None:
            return None
        path = Path(self.restart).expanduser()
        return path if path.is_absolute() else self._result_path / path

    def restart_step(self) -> int:
        """Validate the rank-zero Newmark header and return its completed step."""
        path = self._restart_path()
        if path is None:
            return 0
        with h5py.File(path / "newmark_manifest_R00000.h5", "r") as h5:
            layout = np.asarray(h5["layout"][()]).reshape(-1)
            scheme = np.asarray(h5["scheme"][()]).reshape(-1)
        if (
            layout.shape != (4,)
            or scheme.shape != (6,)
            or layout[0] != 2
            or layout[1] != 0
            or layout[2] != 2
        ):
            raise ValueError("Invalid Newmark restart manifest")
        step = int(layout[3])
        expected = [
            self.time.start + step * self.time.dt,
            self.time.start,
            self.time.end,
            self.time.dt,
            self.time.method.beta,
            self.time.method.gamma,
        ]
        if not 0 <= step <= self.time.steps or not np.allclose(
            scheme, expected, rtol=64 * np.finfo(float).eps, atol=0
        ):
            raise ValueError(
                "Restart interval, step or Newmark method differs from this job"
            )
        return step

    def _external_input_fingerprint(self):
        return self._refresh_external_input_fingerprint(require_inputs=True)

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
        self.restart_step()
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
        return operation_result_path(self._result_path, "transient")

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
            self.restart_step()
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

    def operation_result(self, *, result_path=None):
        """Read a committed result and require the current saved input digests."""
        saved_inputs = (
            json.loads(self.job_file.read_text())
            .get("artifact_contract", {})
            .get("external_inputs")
        )
        if saved_inputs != self._compute_external_input_fingerprint():
            raise ValueError("Restart inputs changed since the job was saved")
        result = load_operation_result(
            self._result_path if result_path is None else result_path, "transient"
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
        try:
            result = self.operation_result()
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
                    data = handle.open()
                    if handle.role == "sampled_wavefield" and not {
                        f.name for f in self.outputs.wavefields
                    }.issubset(data.groups):
                        return False
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def is_task_current(self, task, *, state=None, manifest=None) -> bool:
        state = self.run_state() if state is None else state
        return (
            task == 1
            and state.get("fingerprint") == self.fingerprint()
            and state.get("status") in {"completed", "skipped"}
            and self.results_exist()
        )

    def current_tasks(self):
        return [1] if self.is_task_current(1) else []

    def is_run_current(self):
        return self.is_task_current(1)

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
        current = not (force or policy.force) and self.is_task_current(1)
        records = (
            [{"task": 1, "status": "current", "fingerprint": self.task_fingerprint(1)}]
            if current
            else []
        )
        removed = False
        if apply and not current and self.result_manifest_file.exists():
            self.result_manifest_file.unlink()
            removed = True
        return TaskRunPlan(
            pending_indices=[] if current else [0],
            current_tasks=[1] if current else [],
            strict_current_tasks=[1] if current else [],
            reused_tasks=[],
            accepted_tasks=[],
            accepted_failed_tasks=[],
            skipped_task_records=records,
            skip_policy=policy,
            removed_stale_outputs=removed,
        )

    def task_run_plan(self, **kwargs):
        return self.plan_tasks(apply=True, **kwargs)

    def write_run_state(self, status="completed", **extra):
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

    @property
    def results(self):
        return self.operation_result()

    def frequency_status(self):
        """Report the single trajectory through the common task-status API."""
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


def _validate_fields(value, *, projection=None):
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
        for key, child in value.items():
            if key in {
                "field",
                "quantity",
                "component",
                "components",
                "operator",
                "expression",
            }:
                text = str(child).lower()
                if key == "field" and isinstance(child, str) and projection:
                    name = child.removeprefix("elastic:")
                    allowed = {projection, f"p_{projection}", f"s_{projection}"}
                    if not any(name == f or name.startswith(f + "_") for f in allowed):
                        raise ValueError(
                            "Transient Helmholtz outputs must match helmholtz_projection.field"
                        )
                if any(
                    word in text for word in ("acceleration", "frequency", "helmholtz")
                ) or (any(word in text for word in ("p_velocity", "s_velocity")) and projection != "velocity"):

                    raise ValueError(
                        "Transient receivers require primary or constitutive fields; harmonic velocity/acceleration and spectral recovery are unsupported"
                    )
            _validate_fields(child, projection=projection)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_fields(child, projection=projection)


def _validate_paraview(output, *, projection=None):
    """Restrict native time snapshots to explicitly requested physical fields."""
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
    if output.get("execute_on", "solve") not in {"solve", "final", "none"}:
        raise ValueError("Transient ParaView output uses the TimeOutput schedule")
    items = output.get("items", [])
    fields = output.get("fields", [])
    if not items and not fields:
        raise ValueError("Transient ParaView output requires explicit fields")
    if output.get("properties") or any(
        i.get("kind", "field") != "field" for i in items
    ):
        raise ValueError(
            "Transient ParaView output accepts fields only, without properties or mesh info"
        )
    for item in items:
        if any(part in {"im", "imag", "imaginary"} for part in item.get("parts") or []):
            raise ValueError(
                "Transient ParaView fields are real; imaginary output is unsupported"
            )
    _validate_fields(items, projection=projection)
    for field in fields:
        _validate_fields({"field": field}, projection=projection)


def _transient_projection(simulation):
    """Validate the native elastic Helmholtz configuration used by time outputs."""
    config = simulation.extra.get("helmholtz_projection", {})
    if not config:
        return None
    if not isinstance(config, Mapping):
        raise ValueError("helmholtz_projection must be a mapping")
    if not (config.get("enabled", False) or config.get("parts") or config.get("part")):
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
    parts = config.get("parts")
    if (
        not isinstance(parts, (list, tuple))
        or not parts
        or len(set(parts)) != len(parts)
        or set(parts) - {"p", "s"}
    ):
        raise ValueError("helmholtz_projection.parts must select unique p/s parts")
    return field
