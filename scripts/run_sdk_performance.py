#!/usr/bin/env python3
"""Collect reproducible Python-side FrequenSolve performance evidence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np

from frequensolve import Acquisition, SeismicSimulation, SourceGeometry
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.seismic.receivers import ReceiverNode
from frequensolve.seismic.trace_store import TraceStore
from frequensolve.simulation.jobs import FrequencyDomainJob
from frequensolve.simulation.jobs.artifacts import RunMetadata
from frequensolve.simulation.simulation import CustomJSONEncoder

SCHEMA = "frequensolve-sdk-performance/v1"
BASELINE_SCHEMA = "frequensolve-sdk-performance-baseline/v1"
DEPENDENCIES = (
    "dask",
    "frequensolve",
    "h5py",
    "jsonschema",
    "numpy",
    "pint",
    "xarray",
)
COMPARISON_RUNNER_FIELDS = (
    "system",
    "release",
    "machine",
    "processor",
    "cpuCount",
    "pythonImplementation",
    "pythonVersion",
    "runnerImageOs",
    "runnerImageVersion",
)
COMPARISON_DEPENDENCIES = tuple(
    dependency for dependency in DEPENDENCIES if dependency != "frequensolve"
)


@dataclass(frozen=True)
class Scenario:
    """Prepared benchmark scenario with deterministic validation metadata."""

    name: str
    category: str
    size: str
    operation: Callable[[], Mapping[str, Any]]


def _cpu_model() -> str:
    model = platform.processor().strip()
    if model:
        return model
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return "unknown"


def runner_identity() -> dict[str, Any]:
    """Return fields that must stay stable for threshold comparisons."""

    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": _cpu_model(),
        "cpuCount": os.cpu_count(),
        "pythonImplementation": platform.python_implementation(),
        "pythonVersion": platform.python_version(),
        "runnerName": os.getenv("RUNNER_NAME"),
        "runnerImageOs": os.getenv("ImageOS"),
        "runnerImageVersion": os.getenv("ImageVersion"),
    }


def repository_identity(root: Path) -> dict[str, Any]:
    """Bind evidence to an exact repository state."""

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def dependency_versions() -> dict[str, str]:
    """Return exact versions for the runtime dependencies used by scenarios."""

    versions: dict[str, str] = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def comparison_runner_identity(runner: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the ephemeral job name from a recorded runner identity."""

    return {field: runner.get(field) for field in COMPARISON_RUNNER_FIELDS}


def comparison_dependency_versions(
    dependencies: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep environment versions while the measured package is commit-bound."""

    return {name: dependencies.get(name) for name in COMPARISON_DEPENDENCIES}


def _acquisition_scenario(size: str, count: int) -> Scenario:
    coordinates = [
        [float(index) / max(count - 1, 1), float(index % 17) / 17.0]
        for index in range(count)
    ]
    names = [f"source-{index:05d}" for index in range(count)]
    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar",
            coords=coordinates,
            names=names,
        )
    )

    def operation() -> Mapping[str, Any]:
        payload = acquisition.to_fs()
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        roundtrip = Acquisition.from_fs(payload)
        if roundtrip.source_point_count() != count:
            raise RuntimeError("acquisition roundtrip changed the source count")
        return {"sourcePoints": count, "jsonBytes": len(encoded)}

    return Scenario(
        name=f"acquisition-serialization-{size}",
        category="acquisition-serialization",
        size=size,
        operation=operation,
    )


def _job_scenario(root: Path, size: str, frequency_count: int) -> Scenario:
    project = root / f"job-{size}"
    simulation = SeismicSimulation(
        name=f"performance-{size}",
        physics="acoustic",
        dimension=2,
        project_path=project,
    )
    simulation.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0.0, 0.0], u_bound=[1.0, 1.0], n=[2, 2])
    )
    acquisition = Acquisition()
    acquisition.add_sources(kind="scalar", coords=[[0.5, 0.25]])
    receiver = ReceiverNode()
    receiver.add_component(name="pressure", field="pressure")
    acquisition.add_receiver_group(
        "surface",
        receiver,
        [[index / 31.0, 0.75] for index in range(32)],
    )
    simulation.acquisition = acquisition
    simulation.save()
    frequencies = np.linspace(1.0, 100.0, frequency_count).tolist()
    job = FrequencyDomainJob(
        name="frequency-sweep",
        simulation=simulation,
        f_list=frequencies,
    )

    def operation() -> Mapping[str, Any]:
        plan = job.plan_tasks(force=True)
        payload = job.to_fs(project_relative=True)
        packed = json.dumps(
            payload,
            cls=CustomJSONEncoder,
            separators=(",", ":"),
            sort_keys=True,
        )
        pending = plan["pending_indices"]
        if len(pending) != frequency_count:
            raise RuntimeError("job planning skipped an unexecuted frequency")
        return {
            "frequencyTasks": frequency_count,
            "pendingTasks": len(pending),
            "packedJsonBytes": len(packed),
        }

    return Scenario(
        name=f"job-planning-packing-{size}",
        category="job-planning-packing",
        size=size,
        operation=operation,
    )


def _write_trace_product(path: Path, *, frequencies: int, receivers: int) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "frequency", data=np.linspace(1.0, float(frequencies), frequencies)
        )
        h5.create_dataset("laplace", data=np.zeros(frequencies, dtype=np.float64))
        h5.create_dataset(
            "survey/packed_layout_kind",
            data=np.array(["packed_frequency_trace_v1"], dtype=string_dtype),
        )
        data = np.arange(
            frequencies * receivers * 2,
            dtype=np.float32,
        ).reshape(frequencies, 1, 1, receivers, 2)
        dataset = h5.create_dataset("surface", data=data)
        dataset.attrs["dims"] = ["receiver", "component", "shot", "frequency"]
        dataset.attrs["layout_kind"] = ["dense_trace_v1"]
        dataset.attrs["receiver"] = np.arange(1, receivers + 1, dtype=np.int32)
        dataset.attrs["component"] = np.array(["p"], dtype=string_dtype)
        dataset.attrs["shot"] = np.array([1], dtype=np.int32)


def _trace_scenario(
    root: Path,
    size: str,
    *,
    frequencies: int,
    receivers: int,
) -> Scenario:
    path = root / f"traces-{size}.h5"
    _write_trace_product(path, frequencies=frequencies, receivers=receivers)
    expected_frequencies = {index: float(index) for index in range(1, frequencies + 1)}

    def operation() -> Mapping[str, Any]:
        store = TraceStore(
            metadata={
                "groups": ["surface"],
                "f_map": expected_frequencies,
                "f_max": float(frequencies),
                "df": 1.0,
            },
            files=[path],
        )
        store._consolidated = path
        try:
            data = store.read_h5("surface")
            materialized = np.asarray(data.values)
            if materialized.shape != (frequencies, 1, 1, receivers, 2):
                raise RuntimeError("trace access returned an unexpected shape")
            return {
                "frequencies": frequencies,
                "receivers": receivers,
                "values": int(materialized.size),
            }
        finally:
            store.close()

    return Scenario(
        name=f"trace-store-access-{size}",
        category="trace-store-access",
        size=size,
        operation=operation,
    )


def _validation_scenario(root: Path, size: str, coordinate_count: int) -> Scenario:
    simulation = SeismicSimulation(
        name=f"validation-{size}",
        physics="acoustic",
        dimension=2,
        project_path=root / f"validation-{size}",
    )
    simulation.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0.0, 0.0], u_bound=[1.0, 1.0], n=[4, 4])
    )
    coordinates = [
        [float(index) / max(coordinate_count - 1, 1), 0.5]
        for index in range(coordinate_count)
    ]
    acquisition = Acquisition()
    acquisition.add_sources(kind="scalar", coords=coordinates)
    receiver = ReceiverNode()
    receiver.add_component(name="pressure", field="pressure")
    acquisition.add_receiver_group("surface", receiver, coordinates)
    simulation.acquisition = acquisition
    job = FrequencyDomainJob(name="validation", simulation=simulation, f_list=[10.0])

    def operation() -> Mapping[str, Any]:
        report = job.validate()
        if not report.ok:
            codes = [issue.code for issue in report.issues]
            raise RuntimeError(f"performance validation fixture is invalid: {codes}")
        return {"coordinates": coordinate_count, "issues": len(report.issues)}

    return Scenario(
        name=f"validation-{size}",
        category="validation",
        size=size,
        operation=operation,
    )


def _write_run_metadata(result_path: Path, artifact_count: int) -> None:
    run_dir = result_path / "_fs_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-run-manifest-1",
                "exit_status": {"status": "success"},
                "job_file_sha256": "a" * 64,
                "simulation_file_sha256": "b" * 64,
            }
        )
    )
    (run_dir / "outputs.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": f"ParaView/pv_{index:05d}.vtu",
                        "kind": "vtk",
                        "schema": "fs-output-artifact-1",
                    }
                    for index in range(artifact_count)
                ]
            }
        )
    )
    (run_dir / "timings.json").write_text(
        json.dumps(
            {
                "frequency_tasks": [
                    {"task": index + 1, "total_seconds": float(index + 1) / 10.0}
                    for index in range(artifact_count)
                ]
            }
        )
    )
    (result_path / "_fs_python_run.json").write_text(
        json.dumps({"status": "completed"})
    )


def _result_metadata_scenario(
    root: Path,
    size: str,
    artifact_count: int,
) -> Scenario:
    result_path = root / f"result-metadata-{size}"
    _write_run_metadata(result_path, artifact_count)

    def operation() -> Mapping[str, Any]:
        metadata = RunMetadata.read(result_path)
        artifacts = metadata.artifacts
        if not metadata.successful or len(artifacts) != artifact_count:
            raise RuntimeError("result metadata did not roundtrip expected artifacts")
        return {
            "artifacts": len(artifacts),
            "timingRows": len(metadata.timings.get("frequency_tasks", [])),
        }

    return Scenario(
        name=f"result-metadata-loading-{size}",
        category="result-metadata-loading",
        size=size,
        operation=operation,
    )


def build_scenarios(root: Path) -> list[Scenario]:
    """Build the reviewed small/large scenario matrix."""

    return [
        _acquisition_scenario("small", 16),
        _acquisition_scenario("large", 2048),
        _job_scenario(root, "small", 8),
        _job_scenario(root, "large", 1024),
        _trace_scenario(root, "small", frequencies=4, receivers=64),
        _trace_scenario(root, "large", frequencies=32, receivers=4096),
        _validation_scenario(root, "small", 8),
        _validation_scenario(root, "large", 1024),
        _result_metadata_scenario(root, "small", 16),
        _result_metadata_scenario(root, "large", 4096),
    ]


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("performance statistics require at least one sample")
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "median": statistics.median(values),
        "stdev": stdev,
        "coefficientOfVariation": stdev / mean if mean else 0.0,
    }


def measure_scenario(
    scenario: Scenario,
    *,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    """Measure wall time and peak Python heap for one prepared scenario."""

    if warmups < 0 or samples < 1:
        raise ValueError("warmups must be >= 0 and samples must be >= 1")
    for _ in range(warmups):
        scenario.operation()

    raw: list[dict[str, Any]] = []
    expected_result: Mapping[str, Any] | None = None
    for index in range(samples):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter_ns()
        try:
            result = scenario.operation()
            elapsed_ns = time.perf_counter_ns() - started
            _current, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        if not result:
            raise RuntimeError(f"scenario {scenario.name!r} returned empty evidence")
        if expected_result is None:
            expected_result = dict(result)
        elif dict(result) != dict(expected_result):
            raise RuntimeError(f"scenario {scenario.name!r} changed across samples")
        raw.append(
            {
                "sample": index + 1,
                "wallTimeSeconds": elapsed_ns / 1_000_000_000.0,
                "pythonHeapPeakBytes": int(peak_bytes),
            }
        )

    wall = [float(sample["wallTimeSeconds"]) for sample in raw]
    memory = [float(sample["pythonHeapPeakBytes"]) for sample in raw]
    return {
        "name": scenario.name,
        "category": scenario.category,
        "size": scenario.size,
        "warmups": warmups,
        "samples": raw,
        "wallTimeSeconds": _stats(wall),
        "pythonHeapPeakBytes": _stats(memory),
        "result": dict(expected_result or {}),
    }


def validate_evidence(
    evidence: Mapping[str, Any], expected_names: Sequence[str]
) -> None:
    """Fail closed on missing scenarios or empty measurements."""

    if evidence.get("schema") != SCHEMA:
        raise ValueError("unsupported SDK performance evidence schema")
    scenarios = evidence.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("SDK performance evidence contains no scenarios")
    names = [str(item.get("name")) for item in scenarios if isinstance(item, Mapping)]
    if names != list(expected_names):
        raise ValueError(
            f"SDK performance scenarios differ: expected {list(expected_names)}, got {names}"
        )
    for scenario in scenarios:
        samples = scenario.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"scenario {scenario.get('name')!r} has no measurements")
        for sample in samples:
            if float(sample.get("wallTimeSeconds", 0.0)) <= 0.0:
                raise ValueError("scenario contains an empty wall-time measurement")
            if int(sample.get("pythonHeapPeakBytes", 0)) <= 0:
                raise ValueError("scenario contains an empty peak-memory measurement")


def compare_to_baseline(
    evidence: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> list[str]:
    """Return threshold violations after strict runner/scenario validation."""

    if baseline.get("schema") != BASELINE_SCHEMA:
        raise ValueError("unsupported SDK performance baseline schema")
    expected_runner = baseline.get("runner")
    actual_runner = comparison_runner_identity(evidence.get("runner", {}))
    if expected_runner != actual_runner:
        raise ValueError("runner identity drifted from the reviewed baseline")
    actual_dependencies = comparison_dependency_versions(
        evidence.get("dependencies", {})
    )
    if baseline.get("dependencies") != actual_dependencies:
        raise ValueError("dependency versions drifted from the reviewed baseline")
    thresholds = baseline.get("thresholds")
    if not isinstance(thresholds, Mapping) or not thresholds:
        raise ValueError("performance baseline contains no thresholds")
    scenarios = {
        str(item["name"]): item
        for item in evidence.get("scenarios", [])
        if isinstance(item, Mapping) and "name" in item
    }
    if set(scenarios) != set(thresholds):
        raise ValueError("baseline scenario set differs from measured evidence")

    violations: list[str] = []
    for name, limits in thresholds.items():
        scenario = scenarios[str(name)]
        if not isinstance(limits, Mapping):
            raise ValueError(f"baseline thresholds for {name!r} are invalid")
        wall_limit = float(limits["maxMedianWallTimeSeconds"])
        memory_limit = int(limits["maxMedianPythonHeapBytes"])
        wall = float(scenario["wallTimeSeconds"]["median"])
        memory = int(scenario["pythonHeapPeakBytes"]["median"])
        if wall > wall_limit:
            violations.append(
                f"{name}: median wall time {wall:.6f}s exceeds {wall_limit:.6f}s"
            )
        if memory > memory_limit:
            violations.append(
                f"{name}: median Python heap {memory} exceeds {memory_limit} bytes"
            )
    return violations


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="frequensolve-sdk-performance-") as temp:
        scenario_root = Path(temp)
        scenarios = build_scenarios(scenario_root)
        measured = [
            measure_scenario(
                scenario,
                warmups=args.warmups,
                samples=args.samples,
            )
            for scenario in scenarios
        ]

    evidence = {
        "schema": SCHEMA,
        "createdAtUnixSeconds": time.time(),
        "repository": repository_identity(root),
        "runner": runner_identity(),
        "dependencies": dependency_versions(),
        "configuration": {"warmups": args.warmups, "samples": args.samples},
        "scenarios": measured,
    }
    validate_evidence(evidence, [scenario.name for scenario in scenarios])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")

    violations: list[str] = []
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        violations = compare_to_baseline(evidence, baseline)
    summary = {
        "output": str(args.output),
        "sha256": _artifact_sha256(args.output),
        "commit": evidence["repository"]["commit"],
        "scenarioCount": len(measured),
        "violations": violations,
    }
    print(json.dumps(summary, sort_keys=True))
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
