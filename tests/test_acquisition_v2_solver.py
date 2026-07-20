import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from frequensolve import JobLayout

GENERATOR = Path(__file__).parents[1] / "scripts" / "generate_acquisition_v2_probe.py"
CONTRACT_ROOT = (
    Path(__file__).parent / "contracts" / "sauce-a54bdda" / "trunk" / "contracts"
)


def _build_probe(project_root: Path) -> Path:
    spec = importlib.util.spec_from_file_location("acquisition_v2_probe", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.build_probe(project_root)


def _validate_against_sauce(acquisition: dict) -> None:
    registry = Registry()
    for schema_file in CONTRACT_ROOT.rglob("*.json"):
        contents = json.loads(schema_file.read_text())
        registry = registry.with_resource(
            contents["$id"], Resource.from_contents(contents)
        )
    schema_file = CONTRACT_ROOT / "inputs" / "fs-acquisition-2" / "schema.json"
    schema = json.loads(schema_file.read_text())
    Draft202012Validator(schema, registry=registry).validate(acquisition)


def test_probe_generator_writes_exact_five_point_four_field_artifact(tmp_path):
    project_root = (tmp_path / "project").resolve()
    job_file = _build_probe(project_root)
    job_payload = json.loads(job_file.read_text())
    simulation_file = project_root / job_payload["simulation"]
    simulation_payload = json.loads(simulation_file.read_text())
    acquisition = simulation_payload["Acquisition"]

    _validate_against_sauce(acquisition)
    assert acquisition["schema"] == "fs-acquisition-2"
    assert "source_groups" not in acquisition
    assert len(acquisition["source_geometry"]["sources"]) == 5
    assert len(acquisition["source_encoding"]["fields"]) == 4
    assert acquisition["source_encoding"]["fields"][-1]["terms"] == [
        {"source": "pair_pos", "coefficient": 1.0},
        {"source": "pair_neg", "coefficient": -1.0},
    ]


@pytest.mark.integration
def test_probe_passes_local_solver_init_and_sizing_unchanged(tmp_path):
    executable = os.environ.get("LOCAL_SOLVER_EXECUTABLE")
    if not executable:
        pytest.skip("LOCAL_SOLVER_EXECUTABLE is required")
    executable_path = Path(executable).expanduser()
    if not executable_path.exists():
        pytest.skip(f"LOCAL_SOLVER_EXECUTABLE does not exist: {executable_path}")

    project_root = (tmp_path / "project").resolve()
    job_file = _build_probe(project_root)
    job_payload = json.loads(job_file.read_text())
    layout = JobLayout.from_payload(job_payload, job_file=job_file).with_project(
        project_root
    )
    mesh_file = layout.simulation_dir / "mesh.gmp"
    sizing_file = layout.job_dir / "FS_sizing.json"

    command = [
        str(executable_path),
        "-nthreads",
        "1",
        "-j",
        str(job_file.relative_to(project_root)),
        "--work-directory",
        str(project_root),
        "--init",
        f"--sizing={sizing_file}",
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, (
        f"solver command failed: {' '.join(command)}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert mesh_file.is_file(), f"solver did not create {mesh_file}"
    assert sizing_file.is_file(), f"solver did not create {sizing_file}"
