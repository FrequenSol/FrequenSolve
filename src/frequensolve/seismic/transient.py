"""Native physical-time receiver and sampled-wavefield output access."""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

__all__ = ["TransientDataset"]


def _strings(value):
    return [
        v.decode().strip() if isinstance(v, bytes) else str(v).strip()
        for v in np.asarray(value).reshape(-1)
    ]


def _scalar(h5, key):
    value = np.asarray(h5[key][()])
    if value.size != 1:
        raise ValueError(f"Expected scalar {key}")
    return value.item()


class TransientDataset:
    """Committed native time shards, read one requested group at a time.

    ``read(group, steps=[...])`` returns an xarray DataArray with physical
    ``time`` and integer ``step`` coordinates. Dense data use source/component/
    receiver axes; sparse data retain their trace axis and trace-table IDs.
    This reader never reconstructs signals from frequencies or globs payloads.
    """

    def __init__(self, job, role, *, result_path=None):
        self.job = job
        self.role = role
        self.result = job.operation_result(result_path=result_path)
        self._groups = {}
        metadata = {
            a.id: a
            for a in self.result.artifacts
            if a.representation == "hdf5_metadata"
        }
        families = {}
        for artifact in self.result.artifacts:
            if artifact.role != role or artifact.schema != "fs-time-step-1":
                continue
            if (
                not artifact.path.is_file()
                or artifact.path.stat().st_size != artifact.bytes
            ):
                raise ValueError(f"Missing or truncated time shard: {artifact.path}")
            deps = [metadata[d] for d in artifact.dependencies if d in metadata]
            if len(deps) != 1:
                raise ValueError(
                    "Time shard requires exactly one committed metadata dependency"
                )
            meta = deps[0]
            if not meta.path.is_file() or meta.path.stat().st_size != meta.bytes:
                raise ValueError(f"Missing or truncated time metadata: {meta.path}")
            with h5py.File(artifact.path, "r") as h5:
                if _strings(h5["sample_axis"][()]) != ["time"]:
                    raise ValueError("Expected a physical-time shard")
                step, time = _scalar(h5, "time_step"), float(_scalar(h5, "time"))
                if not isinstance(step, int) or step < 0 or step > job.time.steps:
                    raise ValueError("Invalid completed time step")
                expected = job.time.start + step * job.time.dt
                if not np.isfinite(time) or abs(time - expected) > 64 * np.finfo(
                    float
                ).eps * max(1, abs(expected)):
                    raise ValueError("Time shard does not match the integration grid")
                reference = _strings(h5["trace_metadata_file"][()])
                if reference != [meta.relative_path]:
                    raise ValueError(
                        "Time shard metadata reference differs from its committed dependency"
                    )
                if artifact.path.name != f"step_{step:012d}.h5":
                    raise ValueError(
                        "Time shard filename does not match its completed step"
                    )
            rows = families.setdefault(meta.id, {"metadata": meta, "steps": {}})
            if step in rows["steps"]:
                raise ValueError("Duplicate output for a completed time step")
            rows["steps"][step] = artifact.path
        expected_steps = set(job.output_schedule.steps(job.time))
        try:
            first_step = job.restart_step()
        except FileNotFoundError:
            # Without the removed checkpoint copy, the committed series starts
            # at the restart step; coverage from there on is still required.
            first_step = min(
                (step for family in families.values() for step in family["steps"]),
                default=0,
            )
        expected_steps = {s for s in expected_steps if s >= first_step}
        for family in families.values():
            if set(family["steps"]) != expected_steps:
                raise ValueError(
                    "Committed time output does not cover the requested schedule"
                )
            with h5py.File(family["metadata"].path, "r") as h5:
                base = "survey/receiver_groups/_catalog/"
                names = _strings(h5[base + "group_name"][()])
                paths = _strings(h5[base + "dataset_path"][()])
                if len(names) != len(paths):
                    raise ValueError("Invalid time-output group catalog")
                for name, path in zip(names, paths):
                    if name in self._groups:
                        raise ValueError(f"Ambiguous time-output group: {name}")
                    self._groups[name] = (family, path)
        if not self._groups:
            raise ValueError("No committed time outputs for this family")
        self.steps = np.array(sorted(expected_steps), dtype=np.int64)
        self.times = job.time.start + self.steps * job.time.dt

    @property
    def groups(self):
        """Available receiver or wavefield group names."""
        return tuple(self._groups)

    def read(self, group, *, steps=None):
        """Read one group and optional completed steps as a physical xarray."""
        import xarray as xr

        from frequensolve.seismic.trace_store import TraceStore

        family, path = self._groups[group]
        selected = self.steps if steps is None else np.asarray(steps)
        if (
            selected.ndim != 1
            or len(selected) == 0
            or any(
                isinstance(s, (bool, np.bool_))
                or not isinstance(s, (int, np.integer))
                or s not in family["steps"]
                for s in selected
            )
        ):
            raise ValueError(
                "steps must be a nonempty sequence of available integer steps"
            )
        arrays = []
        with h5py.File(family["metadata"].path, "r") as meta:
            attrs = meta[path].attrs
            dims = list(reversed(_strings(attrs["dims"])))
            if dims not in (["source", "component", "receiver"], ["trace"]):
                raise ValueError(f"Unsupported transient payload dimensions: {dims}")
            shape = None
            for step in selected:
                with h5py.File(family["steps"][int(step)], "r") as h5:
                    data = np.asarray(h5[path][()])
                    if data.ndim != len(dims) + 1 or data.shape[-1] != 2:
                        raise ValueError(
                            "Invalid transient complex-pair storage layout"
                        )
                    if shape is not None and data.shape != shape:
                        raise ValueError("Transient group shape changed between steps")
                    shape = data.shape
                    if not np.isfinite(data).all() or np.any(data[..., 1] != 0):
                        raise ValueError("Native time output must be finite and real")
                    arrays.append(data[..., 0])
            coords = TraceStore._trace_coordinates(
                meta, meta[path], group, dims, shape[:-1]
            )
            coords.update(
                time=self.job.time.start + selected * self.job.time.dt,
                step=("time", selected.astype(np.int64)),
            )
            output = xr.DataArray(
                np.stack(arrays), dims=["time", *dims], coords=coords, name=group
            )
            for key in ("units", "dimension"):
                if key in attrs:
                    values = _strings(attrs[key])
                    if "component" in dims and len(values) == output.sizes["component"]:
                        output = output.assign_coords({key: ("component", values)})
                    else:
                        output.attrs[key] = values
            table = f"survey/receiver_groups/{group}/traces"
            if dims == ["trace"] and table in meta:
                for key in ("source_id", "receiver_id", "component", "weight"):
                    if key in meta[table]:
                        values = np.asarray(meta[table][key][()]).reshape(-1)
                        if len(values) != output.sizes["trace"]:
                            raise ValueError(
                                "Sparse trace table does not match payload"
                            )
                        output = output.assign_coords({key: ("trace", values)})
            receiver_path = f"survey/receiver_groups/{group}/receivers/coordinates"
            if "receiver" in dims and receiver_path in meta:
                xyz = np.asarray(meta[receiver_path][()]).T
                if xyz.ndim == 2 and xyz.shape[0] == output.sizes["receiver"]:
                    axes = ("x", "z") if xyz.shape[1] == 2 else ("x", "y", "z")
                    output = output.assign_coords(
                        {axis: ("receiver", xyz[:, i]) for i, axis in enumerate(axes)}
                    )
            output.attrs.update(sample_axis="time", time_units="s")
            return output

    def time_domain(self, group, component=None, source=None, *, steps=None):
        """Read native time samples, optionally selecting dense labels."""
        data = self.read(group, steps=steps)
        selectors = {
            key: value
            for key, value in (("component", component), ("source", source))
            if value is not None
        }
        return data.sel(**selectors)


@dataclass(frozen=True)
class TransientOutputHandle:
    job: object
    role: str

    def __call__(self, *args, **kwargs):
        from frequensolve.simulation.jobs.artifacts import (
            TraceOutputHandle,
            WavefieldOutputHandle,
        )

        handle = (
            TraceOutputHandle
            if self.role == "simulated_traces"
            else WavefieldOutputHandle
        )
        return handle(self.job)(*args, **kwargs)

    def open(self, upscale=1, project_path=None):
        """Open committed native samples without a frequency-domain pack step."""
        if upscale != 1:
            raise ValueError("Transient output uses native samples; upscale must be 1")
        result_path = (
            None
            if project_path is None
            else self.job._saved_layout().with_project(project_path).result_dir
        )
        return TransientDataset(self.job, self.role, result_path=result_path)
