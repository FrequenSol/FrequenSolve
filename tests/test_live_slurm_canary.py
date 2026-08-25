from pathlib import Path
from types import SimpleNamespace

import pytest
from paramiko import RSAKey

from frequensolve.orchestrator.sites.hpc import live_canary as canary
from frequensolve.orchestrator.sites.hpc.enterprise import EnterpriseHPCProfile
from frequensolve.orchestrator.sites.hpc.live_canary import (
    CANARY_ACKNOWLEDGEMENT,
    CANARY_POLICY_SCHEMA,
    SYNTHETIC_FIXTURE_ID,
    LiveSlurmCanaryError,
    LiveSlurmCanaryPolicy,
    _cancel_active_scheduler_job,
    _cancel_canary_handles,
    _owned_remote_paths,
    _require_remote_paths_absent,
    build_synthetic_live_slurm_job,
    load_live_slurm_canary_policy,
    require_live_slurm_acknowledgement,
    validate_live_slurm_local_inputs,
    validate_live_slurm_site,
)

pytestmark = pytest.mark.hpc_hermetic


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
    with pytest.raises(ValueError, match="invalid port"):
        LiveSlurmCanaryPolicy.from_mapping(
            {**values, "knownHostsName": "[127.0.0.1]:65536"}
        )
    with pytest.raises(ValueError, match="knownHostsFile must be absolute"):
        LiveSlurmCanaryPolicy.from_mapping(
            {**values, "knownHostsFile": "relative-known-hosts"}
        )


def test_canary_policy_loader_rejects_duplicate_keys(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text('{"schema":"one","schema":"two"}')

    with pytest.raises(LiveSlurmCanaryError, match="invalid JSON"):
        load_live_slurm_canary_policy(policy_path)


def test_local_inputs_require_explicit_key_and_approved_known_host(tmp_path):
    private_key = tmp_path / "id_ed25519"
    private_key_object = RSAKey.generate(1024)
    private_key_object.write_private_key_file(str(private_key))
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
        "ssh_port = 22\n"
        f"known_hosts_file = {str(known_hosts)!r}\n"
        'known_hosts_name = "login.example.invalid"\n'
        "allow_ssh_agent = false\n"
        "allow_keyboard_interactive = false\n"
        'username = "synthetic-user"\n'
        f"ssh_key = {str(private_key)!r}\n"
    )
    policy = LiveSlurmCanaryPolicy.from_mapping(policy_values(tmp_path))

    validate_live_slurm_local_inputs(
        site_config_path=config,
        profile="approved",
        policy=policy,
    )

    unbound_config = config.read_text().replace(
        'known_hosts_name = "login.example.invalid"',
        'known_hosts_name = "other.example.invalid"',
    )
    config.write_text(unbound_config)
    with pytest.raises(LiveSlurmCanaryError, match="lookup is not bound"):
        validate_live_slurm_local_inputs(
            site_config_path=config,
            profile="approved",
            policy=policy,
        )
    config.write_text(
        unbound_config.replace(
            'known_hosts_name = "other.example.invalid"',
            'known_hosts_name = "login.example.invalid"',
        )
    )

    private_key.write_text("synthetic-invalid-private-key")
    with pytest.raises(LiveSlurmCanaryError, match="cannot be loaded"):
        validate_live_slurm_local_inputs(
            site_config_path=config,
            profile="approved",
            policy=policy,
        )
    private_key_object.write_private_key_file(str(private_key))
    private_key.chmod(0o600)

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
        config=SimpleNamespace(
            hostname=policy.allowed_host,
            ssh_port=22,
            known_hosts_file=policy.known_hosts_file,
            known_hosts_name=policy.known_hosts_name,
            allow_ssh_agent=False,
            allow_keyboard_interactive=False,
            queue="synthetic",
        ),
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

    assert len(paths) == 5
    assert all(Path(path).name.startswith("fs_canary_") for path in paths)
    assert "/approved/staging/jobs/fs_canary_run_123456_simulation" in paths
    assert success.simulation is cancellation.simulation
    assert success.f_list == [20.0]
    with pytest.raises(LiveSlurmCanaryError, match="must not already exist"):
        build_synthetic_live_slurm_job(project_path, "run-654321")


def test_scheduler_cleanup_cancels_job_when_public_handle_is_terminal():
    commands = []
    responses = iter(["50", "", ""])
    site = SimpleNamespace(
        run_login=lambda command: commands.append(command) or next(responses)
    )
    cancellations = []
    handle = SimpleNamespace(id="50", cancel=lambda: cancellations.append("50"))

    _cancel_active_scheduler_job(
        site,
        handle,
        timeout=10,
        poll_interval=0,
    )

    assert cancellations == ["50"]
    assert commands == [
        "squeue -h -j 50 -o %i",
        "squeue -h -j 50 -o %i",
    ]


def test_scheduler_cleanup_surfaces_actionable_emergency_cancellation(tmp_path):
    policy = LiveSlurmCanaryPolicy.from_mapping(policy_values(tmp_path))
    site = SimpleNamespace(
        run_config=SimpleNamespace(poll_interval=0),
        run_login=lambda command: (_ for _ in ()).throw(
            RuntimeError("synthetic transport loss")
        ),
    )
    handle = SimpleNamespace(id="50", cancel=lambda: None)

    with pytest.raises(
        LiveSlurmCanaryError,
        match=r"scheduler spend may remain active; run job 50: scancel 50",
    ):
        _cancel_canary_handles(site, (handle,), policy=policy)


def test_preexisting_remote_canary_paths_are_never_claimed():
    site = SimpleNamespace(run_login=lambda command: "1")

    with pytest.raises(LiveSlurmCanaryError, match="already exist"):
        _require_remote_paths_absent(
            site,
            ("/approved/staging/fs_canary_existing",),
        )


def test_live_canary_polls_slurm_until_cancellation_is_complete(monkeypatch, tmp_path):
    policy = LiveSlurmCanaryPolicy.from_mapping(policy_values(tmp_path))
    success_job = SimpleNamespace(name="fs_canary_success")
    cancel_job = SimpleNamespace(name="fs_canary_cancel")
    project = SimpleNamespace()
    success_result = SimpleNamespace(
        status=SimpleNamespace(state="complete"),
        output_files=lambda existing: [Path("synthetic-output")],
    )
    cancel_result = SimpleNamespace(status=SimpleNamespace(state="cancelled"))
    scheduler = {"cancel_requested": False, "post_cancel_probes": 0}

    success_handle = SimpleNamespace(
        id="10",
        watch=lambda **kwargs: [
            SimpleNamespace(state="pending"),
            SimpleNamespace(state="running"),
            SimpleNamespace(state="complete"),
        ],
        wait=lambda **kwargs: success_result,
        cancel=lambda: None,
    )

    def cancel():
        scheduler["cancel_requested"] = True

    cancel_handle = SimpleNamespace(
        id="20",
        wait=lambda **kwargs: cancel_result,
        cancel=cancel,
    )
    handles = iter([success_handle, cancel_handle])

    def run_login(command):
        if "squeue -h -n " in command:
            return ""
        if "squeue -h -j 10" in command:
            return ""
        if "squeue -h -j 20" in command:
            if not scheduler["cancel_requested"]:
                return "20"
            scheduler["post_cancel_probes"] += 1
            return "20" if scheduler["post_cancel_probes"] == 1 else ""
        raise AssertionError(f"unexpected command: {command}")

    site = SimpleNamespace(
        run_config=SimpleNamespace(poll_interval=0),
        enterprise_hpc_preflight=lambda: SimpleNamespace(to_evidence=lambda: {}),
        submit=lambda *args, **kwargs: next(handles),
        run_login=run_login,
    )
    monkeypatch.setattr(
        canary, "validate_live_slurm_site", lambda *args, **kwargs: profile()
    )
    monkeypatch.setattr(
        canary,
        "build_synthetic_live_slurm_job",
        lambda *args, **kwargs: (project, success_job, cancel_job),
    )
    monkeypatch.setattr(canary, "_trace_evidence", lambda result: {"finite": True})
    monkeypatch.setattr(
        canary,
        "_manifest_evidence",
        lambda *args, **kwargs: {"mpiRanks": 4, "threadsPerRank": 1},
    )
    monkeypatch.setattr(canary, "_process_count", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        canary, "_require_remote_paths_absent", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        canary,
        "_owned_remote_paths",
        lambda *args, **kwargs: ["/approved/staging/fs_canary_owned"],
    )
    monkeypatch.setattr(canary, "_cleanup_remote_paths", lambda *args, **kwargs: None)

    evidence = canary.run_live_slurm_canary(
        site=site,
        policy=policy,
        project_path=tmp_path / "project",
        run_token="run-123456",
        ranks=4,
        threads_per_rank=1,
    )

    assert scheduler["cancel_requested"] is True
    assert scheduler["post_cancel_probes"] >= 2
    assert evidence["cancellation"]["schedulerJobsRemaining"] == 0


def test_live_canary_recovers_job_id_when_submit_raises(monkeypatch, tmp_path):
    policy = LiveSlurmCanaryPolicy.from_mapping(policy_values(tmp_path))
    success_job = SimpleNamespace(name="fs_canary_recover_success", _job_id=None)
    cancel_job = SimpleNamespace(name="fs_canary_recover_cancel", _job_id=None)
    project = SimpleNamespace()
    scheduler = {"active": False, "cancelled": [], "cleaned": False}

    def submit(job, **kwargs):
        job._job_id = "30"
        scheduler["active"] = True
        raise RuntimeError("synthetic post-submit transport loss")

    def run_login(command):
        if "squeue -h -n fs_canary_recover_success" in command:
            return "30" if scheduler["active"] else ""
        if "squeue -h -n fs_canary_recover_cancel" in command:
            return ""
        if "squeue -h -j 30" in command:
            return "30" if scheduler["active"] else ""
        raise AssertionError(f"unexpected command: {command}")

    def cancel_scheduler_job(job_id):
        scheduler["cancelled"].append(job_id)
        scheduler["active"] = False
        return True

    site = SimpleNamespace(
        run_config=SimpleNamespace(poll_interval=0),
        enterprise_hpc_preflight=lambda: SimpleNamespace(to_evidence=lambda: {}),
        submit=submit,
        run_login=run_login,
        cancel_job=cancel_scheduler_job,
    )
    monkeypatch.setattr(
        canary, "validate_live_slurm_site", lambda *args, **kwargs: profile()
    )
    monkeypatch.setattr(
        canary,
        "build_synthetic_live_slurm_job",
        lambda *args, **kwargs: (project, success_job, cancel_job),
    )
    monkeypatch.setattr(
        canary,
        "_owned_remote_paths",
        lambda *args, **kwargs: ["/approved/staging/fs_canary_owned"],
    )
    monkeypatch.setattr(
        canary, "_require_remote_paths_absent", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        canary,
        "_cleanup_remote_paths",
        lambda *args, **kwargs: scheduler.__setitem__("cleaned", True),
    )

    with pytest.raises(RuntimeError, match="post-submit transport loss"):
        canary.run_live_slurm_canary(
            site=site,
            policy=policy,
            project_path=tmp_path / "project",
            run_token="run-654321",
            ranks=4,
            threads_per_rank=1,
        )

    assert scheduler["cancelled"] == ["30"]
    assert scheduler["active"] is False
    assert scheduler["cleaned"] is True
