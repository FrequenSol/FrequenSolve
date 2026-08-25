from pathlib import Path
from types import SimpleNamespace

import pytest
from paramiko import RSAKey

from frequensolve.orchestrator.sites.hpc.enterprise import EnterpriseHPCProfile
from frequensolve.orchestrator.sites.hpc.live_canary import (
    CANARY_ACKNOWLEDGEMENT,
    CANARY_POLICY_SCHEMA,
    SYNTHETIC_FIXTURE_ID,
    LiveSlurmCanaryError,
    LiveSlurmCanaryPolicy,
    _owned_remote_paths,
    build_synthetic_live_slurm_job,
    load_live_slurm_canary_policy,
    require_live_slurm_acknowledgement,
    validate_live_slurm_local_inputs,
    validate_live_slurm_site,
)


def policy_values(tmp_path: Path) -> dict:
    return {
        "schema": CANARY_POLICY_SCHEMA,
        "policyId": "synthetic-slurm-policy",
        "fixtureId": SYNTHETIC_FIXTURE_ID,
        "allowedHost": "login.example.invalid",
        "knownHostsFile": str(tmp_path / "known_hosts"),
        "knownHostsName": "login.example.invalid",
        "allowedPartition": "synthetic",
        "allowedWorkDir": "/approved/staging",
        "allowedScratchDir": "/approved/scratch",
        "maxNodes": 1,
        "maxRanks": 4,
        "maxThreadsPerRank": 1,
        "maxWallTime": "00:20:00",
        "completionTimeoutSeconds": 1300,
        "cancelTimeoutSeconds": 120,
        "cleanupOwner": "synthetic-test-operator",
        "logRetention": "sanitized-json-only",
        "emergencyCancelCommand": "scancel {job_id}",
    }


def profile() -> EnterpriseHPCProfile:
    digest = "a" * 64
    return EnterpriseHPCProfile(
        profile_id="synthetic-profile",
        host="login.example.invalid",
        bundle_manifest="/bundle/manifests/bundle.json",
        compatibility_manifest="/bundle/manifests/compatibility.json",
        bundle_schema_path="/bundle/manifests/bundle-schema.json",
        compatibility_schema_path="/bundle/manifests/compatibility-schema.json",
        bundle_root="/bundle",
        solver_path="/bundle/bin/fs2d",
        work_dir="/approved/staging",
        scratch_dir="/approved/scratch",
        bundle_version="0.1.0-test",
        bundle_content_sha256=digest,
        bundle_manifest_sha256=digest,
        bundle_schema_sha256=digest,
        compatibility_row_id="synthetic-row",
        compatibility_document_sha256=digest,
        compatibility_schema_sha256=digest,
        solver_version="v0.0.0-test",
        solver_source_commit="b" * 40,
        solver_build_id="synthetic build",
        solver_build_identity_sha256=digest,
        frequensolve_version="0.0.0-test",
        frequensolve_artifact_sha256=digest,
        allowed_partitions=("synthetic",),
        max_nodes=1,
        max_ranks=4,
        max_threads_per_rank=1,
        max_wall_time="00:20:00",
        support_tier="experimental",
        mpi_launcher="mpirun",
    )


def test_canary_policy_is_closed_and_requires_manual_acknowledgement(tmp_path):
    values = policy_values(tmp_path)
    policy = LiveSlurmCanaryPolicy.from_mapping(values)

    assert policy.fixture_id == SYNTHETIC_FIXTURE_ID
    require_live_slurm_acknowledgement(CANARY_ACKNOWLEDGEMENT)
    with pytest.raises(LiveSlurmCanaryError, match="disabled"):
        require_live_slurm_acknowledgement(None)
    with pytest.raises(ValueError, match="unsupported"):
        LiveSlurmCanaryPolicy.from_mapping({**values, "subscription": "forbidden"})
    with pytest.raises(ValueError, match="completion timeout"):
        LiveSlurmCanaryPolicy.from_mapping({**values, "completionTimeoutSeconds": 60})


def test_canary_policy_loader_rejects_duplicate_keys(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text('{"schema":"one","schema":"two"}')

    with pytest.raises(LiveSlurmCanaryError, match="invalid JSON"):
        load_live_slurm_canary_policy(policy_path)


def test_local_inputs_require_explicit_key_and_approved_known_host(tmp_path):
    private_key = tmp_path / "id_ed25519"
    private_key.write_text("synthetic-key-material")
    private_key.chmod(0o600)
    host_key = RSAKey.generate(1024)
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text(
        "login.example.invalid " f"{host_key.get_name()} {host_key.get_base64()}\n"
    )
    config = tmp_path / "site.toml"
    config.write_text(
        'default = "approved"\n'
        "[sites.approved]\n"
        'type = "slurm"\n'
        'hostname = "login.example.invalid"\n'
        'username = "synthetic-user"\n'
        f"ssh_key = {str(private_key)!r}\n"
    )
    policy = LiveSlurmCanaryPolicy.from_mapping(policy_values(tmp_path))

    validate_live_slurm_local_inputs(
        site_config_path=config,
        profile="approved",
        policy=policy,
    )

    private_key.chmod(0o644)
    with pytest.raises(LiveSlurmCanaryError, match="group/world"):
        validate_live_slurm_local_inputs(
            site_config_path=config,
            profile="approved",
            policy=policy,
        )


def test_site_and_bundle_must_remain_inside_policy(tmp_path):
    policy = LiveSlurmCanaryPolicy.from_mapping(policy_values(tmp_path))
    site = SimpleNamespace(
        enterprise_hpc=profile(),
        config=SimpleNamespace(hostname=policy.allowed_host, queue="synthetic"),
        work_dir=Path(policy.allowed_work_dir),
        scratch_dir=Path(policy.allowed_scratch_dir),
        run_config=SimpleNamespace(nodes=1, duration="00:20:00"),
    )

    assert (
        validate_live_slurm_site(site, policy, ranks=4, threads_per_rank=1)
        is site.enterprise_hpc
    )

    site.config.queue = "unapproved"
    with pytest.raises(LiveSlurmCanaryError, match="partition"):
        validate_live_slurm_site(site, policy, ranks=4, threads_per_rank=1)


def test_synthetic_fixture_uses_unique_owned_paths(tmp_path):
    project_path = tmp_path / "canary-project"
    project, success, cancellation = build_synthetic_live_slurm_job(
        project_path, "run-123456"
    )
    fake_site = SimpleNamespace(
        work_dir=Path("/approved/staging"),
        _remote_job_dir=lambda job: Path("/approved/staging/jobs")
        / job.simulation.name
        / job.name,
    )

    paths = _owned_remote_paths(fake_site, project, [success, cancellation])

    assert len(paths) == 4
    assert all(Path(path).name.startswith("fs_canary_") for path in paths)
    assert success.simulation is cancellation.simulation
    assert success.f_list == [20.0]
    with pytest.raises(LiveSlurmCanaryError, match="must not already exist"):
        build_synthetic_live_slurm_job(project_path, "run-654321")
