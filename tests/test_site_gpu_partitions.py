"""GPU partitions: one MPI rank per GPU device, distinct devices for concurrent tasks.

The Stampede3 ``pvc`` partition has eight GPU stacks per node. Runs there
default to eight ranks per node, tasks are sized by device memory as well as
host memory, and every rank of a task selects the device under its allocation
slot (``FS_GPU_DEVICE``), because concurrent tasks are separate MPI jobs whose
ranks all start at local rank 0.
"""

import importlib.util
import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from frequensolve.orchestrator.sites.config_file import load_site_presets
from frequensolve.orchestrator.sites.hpc import site as hpc
from frequensolve.orchestrator.sites.hpc.config import Stampede3Config
from tests.test_site_curvature_slurm import FakeSlurmSite
from tests.test_slurm_site_refactor import DummySSHClientClass

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]


def _scheduler_module():
    path = Path(hpc.__file__).parent / "templates" / "sweep" / "adaptive_scheduler.py"
    spec = importlib.util.spec_from_file_location("adaptive_scheduler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stampede3_preset_describes_the_pvc_partition():
    pvc = load_site_presets()["presets"]["stampede3"]["partitions"]["pvc"]

    assert pvc["gpus_per_node"] == 8
    assert pvc["gpu_memory_per_gpu"] == 65536
    assert pvc["cores_per_node"] == 96
    assert Stampede3Config("pvc").gpus_per_node == 8
    assert Stampede3Config("spr").gpus_per_node == 0


def test_stampede3_pvc_jobs_default_to_gpu_aware_mpi():
    pvc = Stampede3Config("pvc").partition_environment

    assert pvc["I_MPI_OFFLOAD"] == "1"
    assert pvc["I_MPI_OFFLOAD_IPC_MODE"] == "drmfd"
    runtime = pvc["I_MPI_ROOT"]
    assert pvc["PATH"] == f"{runtime}/bin:${{PATH}}"
    # The compiler's Unified Runtime loader precedes module copies such as python/3.12's.
    assert (
        pvc["LD_LIBRARY_PATH"]
        == f"{runtime}/lib:${{CMPLR_ROOT}}/lib:${{LD_LIBRARY_PATH}}"
    )
    assert "FS_CUDA_AWARE_MPI" not in pvc
    assert Stampede3Config("spr").partition_environment == {}


@pytest.mark.parametrize(
    ("queue", "overrides", "expected"),
    [
        ("pvc", {}, 8),
        ("pvc", {"ranks_per_node": 16}, 16),
        ("pvc", {"procs_per_node": 4}, 4),
        ("spr", {}, 2),
    ],
)
def test_gpu_partitions_default_to_one_rank_per_device(queue, overrides, expected):
    profile = hpc.SlurmRunConfig(ranks_per_node=2)

    config, _ = profile.resolved(Stampede3Config(), queue=queue, **overrides)

    assert config.ranks_per_node == expected


def _site(monkeypatch, tmp_path, site_environment=None, **config):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    return FakeSlurmSite(
        config=hpc.SlurmSiteConfig(
            hostname="login.example.edu",
            queue="debug",
            mpi_wrapper="ibrun",
            partitions={
                "debug": {
                    "max_nodes": 4,
                    "cores_per_node": 96,
                    "memory_per_node": 1031000,
                    **config,
                }
            },
        ),
        solver="/remote/FS_seismic",
        work_dir=str(tmp_path / "work"),
        solver_policy="off",
        environment=site_environment,
    )


def _scheduler_config(script):
    assigned = script.split("printf '%s\\n' ", 1)[1].split(' > "$scheduler_config"')[0]
    return json.loads(shlex.split(assigned)[0])


@pytest.mark.parametrize(
    ("gpus", "gpu_memory", "ranks_per_node", "memory_per_rank"),
    [
        (8, 65536, 8, 64.0),
        (8, 0, 8, 1031000 / 8 / 1024),
        (0, 65536, 8, 1031000 / 8 / 1024),
        # Each rank binds one device; the four unused stacks add no memory.
        (8, 65536, 4, 64.0),
        (8, 65536, 16, 32.0),
    ],
)
def test_batch_sweep_sizes_tasks_by_device_memory_and_passes_the_gpu_layout(
    monkeypatch, tmp_path, gpus, gpu_memory, ranks_per_node, memory_per_rank
):
    site = _site(
        monkeypatch, tmp_path, gpus_per_node=gpus, gpu_memory_per_gpu=gpu_memory
    )

    script = site._sweep_SLURM_script(
        n_tasks=2,
        n_nodes=1,
        ranks_per_node=ranks_per_node,
        stdout=str(tmp_path / "logs"),
        queue="debug",
    )

    config = _scheduler_config(script)
    assert config["mem_per_rank_gib"] == pytest.approx(memory_per_rank)
    assert config["ranks_per_node"] == ranks_per_node
    assert config["gpus_per_node"] == gpus
    assert config["omp_threads"] == 96 // ranks_per_node


def test_batch_sweep_exports_the_partition_environment_under_site_overrides(
    monkeypatch, tmp_path
):
    site = _site(
        monkeypatch,
        tmp_path,
        site_environment={"FS_CUDA_AWARE_MPI": "0"},
        gpus_per_node=8,
        environment={
            "I_MPI_OFFLOAD": "1",
            "PATH": "/mpi/bin:${PATH}",
            "FS_CUDA_AWARE_MPI": "1",
        },
    )

    lines = site._sweep_SLURM_script(
        n_tasks=1,
        n_nodes=1,
        ranks_per_node=8,
        stdout=str(tmp_path / "logs"),
        queue="debug",
    ).splitlines()

    assert "export I_MPI_OFFLOAD=1" in lines
    assert 'export PATH="/mpi/bin:${PATH}"' in lines
    # The site environment opts this run out of GPU-aware MPI.
    assert "export FS_CUDA_AWARE_MPI=0" in lines
    assert "export FS_CUDA_AWARE_MPI=1" not in lines


def test_site_search_paths_extend_the_partition_runtime(monkeypatch, tmp_path):
    site = _site(
        monkeypatch,
        tmp_path,
        site_environment={
            "LD_LIBRARY_PATH": "/site/lib:${LD_LIBRARY_PATH}",
            "PATH": "/site/bin",
        },
        gpus_per_node=8,
        environment={
            "LD_LIBRARY_PATH": "/mpi/lib:${LD_LIBRARY_PATH}",
            "PATH": "/mpi/bin:${PATH}",
        },
    )

    lines = site._sweep_SLURM_script(
        n_tasks=1,
        n_nodes=1,
        ranks_per_node=8,
        stdout=str(tmp_path / "logs"),
        queue="debug",
    ).splitlines()

    # A self-referencing site value composes on top of the partition runtime.
    partition = lines.index('export LD_LIBRARY_PATH="/mpi/lib:${LD_LIBRARY_PATH}"')
    assert (
        lines.index('export LD_LIBRARY_PATH="/site/lib:${LD_LIBRARY_PATH}"') > partition
    )
    # A plain site value still replaces the partition value.
    assert "export PATH=/site/bin" in lines
    assert 'export PATH="/mpi/bin:${PATH}"' not in lines


def test_cpu_partition_jobs_export_no_gpu_mpi_environment(monkeypatch, tmp_path):
    site = _site(monkeypatch, tmp_path)

    script = site._sweep_SLURM_script(
        n_tasks=1,
        n_nodes=1,
        ranks_per_node=8,
        stdout=str(tmp_path / "logs"),
        queue="debug",
    )

    assert "I_MPI_OFFLOAD" not in script


def _scheduler(tmp_path, launcher, **options):
    return _scheduler_module().AdaptiveScheduler(
        {
            "executable": "/remote/bin/solver",
            "mpi": launcher,
            "total_ranks": 16,
            "omp_threads": 12,
            "mem_per_rank_gib": 64,
            "job_task_count": 1,
            "task_indices": [1],
            "sizing_json": str(tmp_path / "sizing.json"),
            **options,
        },
        job_file="job.json",
        output=str(tmp_path),
        status=str(tmp_path / "status.json"),
    )


def test_gpu_tasks_select_the_device_under_their_slots(tmp_path):
    scheduler = _scheduler_module()
    instance = _scheduler(tmp_path, "ibrun", ranks_per_node=8, gpus_per_node=8)

    command = instance._launch_command(task_id=1, offset=6, ranks=4)

    place = str(tmp_path / "fs-gpu-place")
    start = command.index(place)
    assert command[:start] == ["ibrun", "-n", "4", "-o", "6", "task_affinity"]
    assert command[start : start + 5] == [place, "6", "8", "8", "/remote/bin/solver"]
    assert Path(place).read_text() == f"#!/bin/sh\n{scheduler._GPU_PLACE}\n"
    assert os.access(place, os.X_OK)
    # TACC's ibrun re-splits its arguments, so the launch must be words only.
    assert " ".join(command).split() == command
    assert scheduler._GPU_PLACE == hpc.GPU_PLACE_SNIPPET


@pytest.mark.parametrize("launcher", ["ibrun", "mpirun"])
def test_cpu_and_whole_allocation_tasks_keep_the_solver_device_choice(
    tmp_path, launcher
):
    gpu = _scheduler(tmp_path, launcher, ranks_per_node=8, gpus_per_node=8)
    cpu = _scheduler(tmp_path, "ibrun", ranks_per_node=8, gpus_per_node=0)

    def placed(command):
        return any(arg.endswith("fs-gpu-place") for arg in command)

    assert not placed(cpu._launch_command(task_id=1, offset=6, ranks=4))
    if launcher == "mpirun":
        assert not placed(gpu._launch_command(task_id=1, offset=0, ranks=16))


@pytest.mark.parametrize(
    ("offset", "ranks_per_node", "gpus", "expected"),
    [
        (6, 8, 8, ["6", "7", "0", "1"]),  # a task crossing into the next node
        (0, 16, 8, ["0", "0", "1", "1"]),  # two ranks share each device
        (4, 4, 8, ["0", "2", "4", "6"]),  # one rank per GPU card
    ],
)
def test_gpu_placement_maps_slots_to_node_devices(
    tmp_path, offset, ranks_per_node, gpus, expected
):
    place = _scheduler_module().write_gpu_place_script(tmp_path)
    devices = []
    for rank in range(4):
        env = {
            name: value
            for name, value in os.environ.items()
            if name not in {"PMI_RANK", "SLURM_PROCID"}
        }
        env["PMI_RANK"] = str(rank)
        completed = subprocess.run(
            [
                place,
                str(offset),
                str(ranks_per_node),
                str(gpus),
                "/bin/sh",
                "-c",
                'printf %s "$FS_GPU_DEVICE"',
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        devices.append(completed.stdout)
    assert devices == expected


def test_attached_gpu_sweep_places_every_task_rank(monkeypatch, tmp_path):
    site = _site(monkeypatch, tmp_path, gpus_per_node=8)
    site.pool.nproc = 16
    site.pool.ncore = 192
    site.pool.nhost = 2
    job = SimpleNamespace(
        n_tasks=4,
        _remote_path=lambda work_dir: tmp_path,
        requires_postprocess=lambda: False,
    )

    script = site._sweep_script(job, procs_per_task=4, pack=False)

    assert 'gpu_place=("$gpu_place_script" "$off" 8 8)' in script
    assert 'task_affinity ${gpu_place[@]+"${gpu_place[@]}"} $executable' in script

    site.config = hpc.SlurmSiteConfig(hostname="login.example.edu", queue="cpu")
    assert "fs-gpu-place" not in site._sweep_script(job, procs_per_task=4, pack=False)


def test_attached_sweep_uses_the_allocation_partition(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    shape = {"max_nodes": 4, "cores_per_node": 96, "memory_per_node": 1031000}
    site = FakeSlurmSite(
        config=hpc.SlurmSiteConfig(
            hostname="login.example.edu",
            queue="debug",
            mpi_wrapper="ibrun",
            partitions={
                "debug": shape,
                "gpu": {
                    **shape,
                    "gpus_per_node": 8,
                    "environment": {"I_MPI_OFFLOAD": "1"},
                },
            },
        ),
        solver="/remote/FS_seismic",
        work_dir=str(tmp_path / "work"),
        solver_policy="off",
    )
    monkeypatch.setattr(
        site,
        "run_login",
        lambda cmd, **kwargs: (
            "JobId=42 Partition=gpu BatchHost=c001 NumNodes=2 NumCPUs=192 "
            "NumTasks=16 StartTime=2026-10-13T08:00:00 "
            "EndTime=2026-10-13T10:00:00 TimeLimit=02:00:00 RunTime=00:10:00"
        ),
    )
    site.pool.id = 42
    site._set_pool_info()
    job = SimpleNamespace(
        n_tasks=4,
        _remote_path=lambda work_dir: tmp_path,
        requires_postprocess=lambda: False,
    )

    # The site default is a CPU partition; the allocation runs on GPU nodes.
    assert site.pool.partition == "gpu"
    script = site._sweep_script(job, procs_per_task=4, pack=False)
    assert 'gpu_place=("$gpu_place_script" "$off" 8 8)' in script
    assert "export I_MPI_OFFLOAD=1" in script.splitlines()

    site.pool.partition = "unlisted"
    script = site._sweep_script(job, procs_per_task=4, pack=False)
    assert "fs-gpu-place" not in script
    assert "I_MPI_OFFLOAD" not in script
