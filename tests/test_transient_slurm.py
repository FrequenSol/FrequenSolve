"""Transient initialization and rank selection in SLURM batch jobs."""

import pytest

from tests.test_site_gpu_partitions import _site

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]


def test_transient_batch_initialization_skips_frequency_sizing(monkeypatch, tmp_path):
    from tests.test_site_gpu_partitions import _scheduler_config

    site = _site(monkeypatch, tmp_path)
    script = site._sweep_SLURM_script(
        n_tasks=1,
        n_nodes=1,
        ranks_per_node=8,
        stdout=str(tmp_path / "logs"),
        queue="debug",
        mesh_only_init=True,
        skip_sizing=False,
    )
    assert _scheduler_config(script)["skip_sizing"] is True
    init = [line for line in script.splitlines() if '> "$dir_out/init.log"' in line]
    assert "--mesh " in init[0]


def test_transient_modes_pin_ranks_without_sizing(monkeypatch, tmp_path):
    from tests.test_site_gpu_partitions import _scheduler_config

    site = _site(monkeypatch, tmp_path)
    script = site._sweep_SLURM_script(
        n_tasks=3,
        n_nodes=1,
        ranks_per_node=8,
        stdout=str(tmp_path / "logs"),
        queue="debug",
        mesh_only_init=True,
        task_ranks={2: 4},
    )
    config = _scheduler_config(script)
    assert config["skip_sizing"] is True
    assert {int(k): v for k, v in config["task_ranks"].items()} == {1: 8, 2: 4, 3: 8}
