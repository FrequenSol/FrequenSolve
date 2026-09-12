import base64
import hashlib
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from frequensolve import __version__ as frequensolve_version
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.orchestrator.sites.hpc import (
    SlurmLoginCredentials,
    SlurmRunConfig,
    SlurmSite,
    SlurmSiteConfig,
)
from frequensolve.orchestrator.sites.hpc import auth as hpc_auth
from frequensolve.orchestrator.sites.hpc import enterprise as enterprise_contract
from frequensolve.orchestrator.sites.hpc import site as hpc
from frequensolve.orchestrator.sites.hpc.auth import SlurmAuthenticator
from frequensolve.orchestrator.sites.hpc.enterprise import (
    BUNDLE_MANIFEST_RELATIVE,
    BUNDLE_SCHEMA_RELATIVE,
    COMPATIBILITY_MANIFEST_RELATIVE,
    COMPATIBILITY_SCHEMA_RELATIVE,
    EnterpriseHPCPreflightError,
    installed_frequensolve_artifact_sha256,
)
from frequensolve.project.project import Project
from frequensolve.simulation.jobs import FrequencyDomainJob

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]

BUNDLE_CONTENT_SHA = "a" * 64
BUNDLE_MANIFEST_SHA = "e" * 64
COMPATIBILITY_SHA = "b" * 64
BUNDLE_SCHEMA_SHA = "7" * 64
COMPATIBILITY_SCHEMA_SHA = "8" * 64
SOLVER_SHA = "9" * 64
SDK_SHA = "d" * 64
COMMIT = "1" * 40
SOLVER_IDENTITY = {
    "schema": "frequensolver-identity-1",
    "product": "FrequenSolver",
    "version": "1.2.3",
    "build_id": "synthetic-build",
    "git_commit": COMMIT,
}
SOLVER_IDENTITY_SHA = hashlib.sha256(
    (json.dumps(SOLVER_IDENTITY, sort_keys=True, separators=(",", ":")) + "\n").encode()
).hexdigest()


def profile_values():
    return {
        "schema": "frequensolve-enterprise-hpc-profile/v1",
        "profile_id": "synthetic-private-slurm",
        "bundle_root": "/opt/frequensolve",
        "bundle_manifest_sha256": BUNDLE_MANIFEST_SHA,
    }


def _installed_file(path, sha256, *, mode="0644"):
    return {
        "path": path,
        "kind": "file",
        "mode": mode,
        "ownership": {"user": "install-owner", "group": "install-group"},
        "size": 123,
        "sha256": sha256,
    }


def contract_pair():
    bundle = {
        "schemaVersion": "frequensolve-enterprise-hpc/v1",
        "bundleName": "frequensolve-enterprise-hpc",
        "bundleVersion": "1.0.0-synthetic",
        "releaseState": "candidate",
        "createdAt": "2026-08-24T12:00:00Z",
        "stateChangedAt": "2026-08-24T12:00:00Z",
        "contentSha256": BUNDLE_CONTENT_SHA,
        "deployment": {
            "repository": "FrequenSol/FrequenSolveHPCDeployment",
            "version": "1.0.0-synthetic",
            "sourceCommit": "3" * 40,
            "sourceArchiveSha256": "4" * 64,
        },
        "supportTier": "experimental",
        "upstreamArtifacts": {
            "sauce": {
                "repository": "FrequenSol/Sauce",
                "name": "synthetic-native.tar",
                "version": "1.2.3",
                "sourceCommit": COMMIT,
                "producerIdentitySha256": SOLVER_IDENTITY_SHA,
                "locator": "https://example.invalid/sauce.tar?versionId=synthetic-1",
                "sha256": "5" * 64,
            },
            "frequensolve": {
                "repository": "FrequenSol/FrequenSolve",
                "name": "frequensolve-synthetic.whl",
                "version": frequensolve_version,
                "sourceCommit": "2" * 40,
                "producerIdentitySha256": "6" * 64,
                "locator": "https://example.invalid/sdk.whl?versionId=synthetic-1",
                "sha256": SDK_SHA,
            },
        },
        "installedFiles": [
            _installed_file("bin/FS_seismic", SOLVER_SHA, mode="0755"),
            _installed_file(COMPATIBILITY_MANIFEST_RELATIVE, COMPATIBILITY_SHA),
            _installed_file(BUNDLE_SCHEMA_RELATIVE, BUNDLE_SCHEMA_SHA),
            _installed_file(
                COMPATIBILITY_SCHEMA_RELATIVE,
                COMPATIBILITY_SCHEMA_SHA,
            ),
        ],
        "executables": [
            {
                "path": "bin/FS_seismic",
                "variant": "double-mpi",
                "precision": "double",
                "features": ["mpi"],
                "publicIdentityCommand": "--identity-json",
            }
        ],
        "environment": {
            "modulefile": "modulefiles/frequensolve.lua",
            "wrapper": "wrappers/frequensolve-env",
            "doctor": "bin/frequensolve-doctor",
            "pythonArtifact": "python/frequensolve-synthetic.whl",
        },
        "compatibility": {
            "schemaVersion": "frequensolve-enterprise-hpc-compatibility/v1",
            "rowId": "synthetic-private-slurm-row",
            "documentSha256": COMPATIBILITY_SHA,
        },
        "evidence": [],
        "knownLimitations": ["Synthetic fixture only"],
        "upgradeFrom": [],
        "rollbackTo": [],
    }
    compatibility = {
        "schemaVersion": "frequensolve-enterprise-hpc-compatibility/v1",
        "rowId": "synthetic-private-slurm-row",
        "observedAt": "2026-08-24T12:00:00Z",
        "supportTier": "experimental",
        "platform": {
            "provider": "azure",
            "region": "synthetic-region",
            "os": {"name": "Synthetic Linux", "version": "1.0"},
            "baseImage": {
                "publisher": "SyntheticPublisher",
                "offer": "synthetic-linux",
                "sku": "synthetic-hpc",
                "version": "1.2.3",
            },
            "architecture": "x86_64",
            "cpu": {
                "family": "Synthetic CPU",
                "nodeShape": "Synthetic_HPC_v1",
                "coresPerNode": 8,
                "memoryGiB": 32,
            },
        },
        "toolchain": {
            "compiler": {"name": "Synthetic Compiler", "version": "1.0"},
            "mpi": {
                "name": "Synthetic MPI",
                "version": "1.0",
                "abi": "synthetic-mpi-abi-1",
            },
            "hdf5": {"version": "1.14.0", "parallel": True},
            "numericalLibraries": [{"name": "Synthetic BLAS", "version": "1.0"}],
            "python": "3.12.12",
            "slurm": "24.05.5",
            "launcher": "srun",
        },
        "storage": {
            "sharedFilesystem": "synthetic-shared",
            "scratchFilesystem": "synthetic-local",
            "scratchPattern": "job-local-then-owned-results",
        },
        "solver": {
            "version": "1.2.3",
            "sourceCommit": COMMIT,
            "buildIdentitySha256": SOLVER_IDENTITY_SHA,
            "variants": ["double-mpi"],
        },
        "frequensolve": {
            "version": frequensolve_version,
            "artifactSha256": SDK_SHA,
        },
        "limits": {
            "maxNodes": 2,
            "maxRanks": 16,
            "maxThreadsPerRank": 8,
        },
        "evidence": [],
        "knownLimitations": ["Synthetic fixture only"],
        "recertificationTriggers": ["Any identity change"],
    }
    return bundle, compatibility


def _required_synthetic_shape(value):
    if isinstance(value, dict):
        return {
            "type": "object",
            "required": list(value),
            "additionalProperties": False,
            "properties": {
                key: _required_synthetic_shape(item) for key, item in value.items()
            },
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "items": _required_synthetic_shape(value[0]) if value else {},
        }
    return {}


def contract_schemas(bundle=None, compatibility=None):
    if bundle is None or compatibility is None:
        bundle, compatibility = contract_pair()
    draft = "https://json-schema.org/draft/2020-12/schema"
    return (
        {
            **_required_synthetic_shape(bundle),
            "$schema": draft,
            "$id": enterprise_contract.BUNDLE_SCHEMA_ID,
        },
        {
            **_required_synthetic_shape(compatibility),
            "$schema": draft,
            "$id": enterprise_contract.COMPATIBILITY_SCHEMA_ID,
        },
    )


def bundle_snapshot(bundle=None, compatibility=None):
    if bundle is None or compatibility is None:
        bundle, compatibility = contract_pair()
    bundle_schema, compatibility_schema = contract_schemas(bundle, compatibility)
    return enterprise_contract._BundleSnapshot(
        bundle=bundle,
        bundle_sha256=BUNDLE_MANIFEST_SHA,
        compatibility=compatibility,
        compatibility_sha256=COMPATIBILITY_SHA,
        bundle_schema=bundle_schema,
        bundle_schema_sha256=BUNDLE_SCHEMA_SHA,
        compatibility_schema=compatibility_schema,
        compatibility_schema_sha256=COMPATIBILITY_SCHEMA_SHA,
    )


def runtime_observation(**overrides):
    values = {
        "host": "login.example.invalid",
        "scheduler_version": "slurm 24.05.5",
        "cores_per_node": 8,
        "mpi_launcher": "srun",
        "solver_path": "/opt/frequensolve/bin/FS_seismic",
        "solver_identity": SOLVER_IDENTITY,
        "frequensolve_version": frequensolve_version,
        "frequensolve_artifact_sha256": SDK_SHA,
    }
    values.update(overrides)
    return enterprise_contract._RuntimeObservation(**values)


def validate_contract(*, bundle=None, compatibility=None, runtime=None, profile=None):
    return enterprise_contract._validate_bundle_snapshot(
        profile
        or enterprise_contract._EnterpriseHPCProfile.from_mapping(profile_values()),
        bundle_snapshot(bundle, compatibility),
        runtime or runtime_observation(),
    )


def test_generated_profile_is_small_closed_and_shell_safe():
    values = profile_values()
    profile = enterprise_contract._EnterpriseHPCProfile.from_mapping(values)

    assert set(values) == {
        "schema",
        "profile_id",
        "bundle_root",
        "bundle_manifest_sha256",
    }
    assert profile.installed_path(BUNDLE_MANIFEST_RELATIVE) == (
        "/opt/frequensolve/manifests/bundle-manifest.json"
    )

    for name, value in (
        ("bundle_root", "/opt/frequensolve/../customer"),
        ("bundle_root", "/opt/frequensolve;touch-pwned"),
    ):
        with pytest.raises(ValueError, match="enterprise_hpc"):
            enterprise_contract._EnterpriseHPCProfile.from_mapping(
                {**values, name: value}
            )

    with pytest.raises(ValueError, match="Unsupported enterprise_hpc"):
        enterprise_contract._EnterpriseHPCProfile.from_mapping(
            {**values, "azure_subscription": "fake"}
        )


def test_profile_contains_no_provider_credential_or_duplicated_runtime_contract():
    serialized = json.dumps(profile_values(), sort_keys=True)
    for prohibited in (
        "azure",
        "password",
        "license",
        "solver_path",
        "host",
        "partition",
        "module",
        "max_nodes",
    ):
        assert prohibited not in serialized.lower()


def test_anchored_bundle_produces_only_sanitized_exact_evidence():
    validated = validate_contract()
    evidence = validated.result.to_evidence()

    assert evidence["bundle_content_sha256"] == BUNDLE_CONTENT_SHA
    assert evidence["bundle_manifest_sha256"] == BUNDLE_MANIFEST_SHA
    assert evidence["solver_build_identity_sha256"] == SOLVER_IDENTITY_SHA
    assert evidence["solver_source_commit"] == COMMIT
    assert evidence["frequensolve_artifact_sha256"] == SDK_SHA
    assert "bundle_root" not in evidence
    assert "account" not in evidence
    assert validated.limits.max_nodes == 2


@pytest.mark.parametrize(
    ("runtime_override", "message"),
    [
        ({"scheduler_version": "slurm 24.05.4"}, "compatibility scheduler"),
        ({"mpi_launcher": "mpirun"}, "compatibility launcher"),
        ({"cores_per_node": 64}, "compatibility cores per node"),
        ({"frequensolve_artifact_sha256": "0" * 64}, "FrequenSolve artifact"),
        (
            {"solver_path": "/opt/other/bin/FS_seismic"},
            "outside the immutable bundle root",
        ),
    ],
)
def test_runtime_must_match_the_anchored_compatibility_row(runtime_override, message):
    with pytest.raises(EnterpriseHPCPreflightError, match=message):
        validate_contract(runtime=runtime_observation(**runtime_override))


def test_public_solver_identity_must_match_the_anchored_producer_identity():
    identity = {**SOLVER_IDENTITY, "build_id": "wrong-build"}

    with pytest.raises(EnterpriseHPCPreflightError, match="solver build identity"):
        validate_contract(runtime=runtime_observation(solver_identity=identity))


def test_bundle_manifest_digest_is_the_single_profile_trust_anchor():
    profile = enterprise_contract._EnterpriseHPCProfile.from_mapping(profile_values())
    snapshot = bundle_snapshot()
    snapshot = enterprise_contract._BundleSnapshot(
        **{**snapshot.__dict__, "bundle_sha256": "0" * 64}
    )

    with pytest.raises(EnterpriseHPCPreflightError, match="does not match the profile"):
        enterprise_contract._validate_bundle_snapshot(
            profile,
            snapshot,
            runtime_observation(),
        )


def test_bundle_manifest_binds_all_other_installed_contracts():
    bundle, compatibility = contract_pair()
    bundle["installedFiles"][1]["sha256"] = "0" * 64

    with pytest.raises(EnterpriseHPCPreflightError, match="contract digest mismatch"):
        validate_contract(bundle=bundle, compatibility=compatibility)


def test_schema_valid_withdrawn_bundle_is_rejected():
    bundle, compatibility = contract_pair()
    bundle["releaseState"] = "withdrawn"
    bundle["withdrawal"] = {"reason": "Synthetic withdrawal test"}

    with pytest.raises(EnterpriseHPCPreflightError, match="withdrawn"):
        validate_contract(bundle=bundle, compatibility=compatibility)


def test_selected_solver_path_and_variant_are_bundle_declared():
    bundle, compatibility = contract_pair()
    bundle["executables"][0]["path"] = "bin/other-solver"

    with pytest.raises(EnterpriseHPCPreflightError, match="not uniquely declared"):
        validate_contract(bundle=bundle, compatibility=compatibility)


@pytest.mark.parametrize(
    ("nodes", "ranks_per_node", "cores", "message"),
    [
        (3, 4, 8, "max_nodes"),
        (2, 9, 16, "max_ranks"),
        (1, 9, 8, "node cores"),
        (1, 1, 16, "max_threads_per_rank"),
    ],
)
def test_compatibility_limits_bound_scheduler_requests(
    nodes, ranks_per_node, cores, message
):
    limits = validate_contract().limits

    with pytest.raises(EnterpriseHPCPreflightError, match=message):
        limits.validate_resources(
            nodes=nodes,
            ranks_per_node=ranks_per_node,
            cores_per_node=cores,
        )


class _ExitChannel:
    def __init__(self, status=0):
        self.status = status

    def recv_exit_status(self):
        return self.status


class _Stream:
    def __init__(self, text="", status=0):
        self.text = text
        self.channel = _ExitChannel(status)

    def read(self):
        return self.text.encode()


class _RawLogin:
    def exec_command(self, command, environment=None):
        if command == "echo $WORK":
            return None, _Stream("/synthetic/work"), _Stream("")
        if command == "echo $HOSTNAME":
            return None, _Stream("login"), _Stream("")
        return None, _Stream(""), _Stream("")

    def close(self):
        pass


class _WrappedLogin:
    def __init__(self, client):
        self.client = client
        self.hostname = "login"

    def close(self):
        self.client.close()

    def is_proxy(self):
        return False

    def get_transport(self):
        return None


class _Credentials(SlurmLoginCredentials):
    user_env = "SYNTHETIC_HPC_USERNAME"
    pw_env = "SYNTHETIC_HPC_PASSWORD"
    ssh_key_env = "SYNTHETIC_SSH_PASSPHRASE"


class _Site(SlurmSite):
    site_name = "Synthetic"
    credentials_cls = _Credentials

    def authenticate(self, host=None):
        return _RawLogin()


def _enterprise_site(monkeypatch):
    config = SlurmSiteConfig(
        hostname="login.example.invalid",
        queue="synthetic",
        mpi_wrapper="srun",
        account="synthetic-account",
        known_hosts_file="/synthetic/known_hosts",
        enterprise_hpc=profile_values(),
        max_nodes=2,
        cores_per_node=8,
        memory_per_node=32768,
    )
    run_config = SlurmRunConfig(
        queue="synthetic",
        nodes=1,
        ranks_per_node=4,
        duration="00:30:00",
        account="synthetic-account",
    )
    monkeypatch.setattr(hpc, "SSHClientClass", _WrappedLogin)
    monkeypatch.setattr(
        hpc,
        "installed_frequensolve_artifact_sha256",
        lambda: SDK_SHA,
    )
    return _Site(
        config=config,
        run_config=run_config,
        solver="/opt/frequensolve/bin/FS_seismic",
        work_dir="/synthetic/work",
        modules=["frequensolve/1.0.0-synthetic"],
    )


def test_generic_site_preserves_existing_configuration_and_submission_defaults(
    monkeypatch,
):
    run_config = SlurmRunConfig(
        queue="cpu,gpu",
        account="project.team+shared",
        notify_email="first.last%tag@example.com",
    )
    monkeypatch.setattr(hpc, "SSHClientClass", _WrappedLogin)

    site = _Site(
        config=SlurmSiteConfig(
            hostname="login.example.invalid",
            queue="cpu,gpu",
            mpi_wrapper="srun",
        ),
        run_config=run_config,
        solver="/existing/bin/FS_seismic",
        work_dir="/existing/work",
    )

    assert site.enterprise_hpc is None
    assert site.run_config is run_config
    assert site.run_config.queue == "cpu,gpu"
    assert site.run_config.account == "project.team+shared"
    assert site.run_config.notify_email == "first.last%tag@example.com"
    assert site.modules == []


def test_enterprise_profile_does_not_override_generic_site_values(monkeypatch):
    site = _enterprise_site(monkeypatch)

    assert site.executable == "/opt/frequensolve/bin/FS_seismic"
    assert site.work_dir == Path("/synthetic/work")
    assert site.modules == ["frequensolve/1.0.0-synthetic"]
    assert site.config.queue == "synthetic"
    assert site.config.account == "synthetic-account"


def test_enterprise_profile_requires_explicit_host_trust(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", _WrappedLogin)

    with pytest.raises(ValueError, match="explicit known-hosts"):
        _Site(
            config=SlurmSiteConfig(
                hostname="login.example.invalid",
                queue="synthetic",
                mpi_wrapper="srun",
                enterprise_hpc=profile_values(),
            ),
            solver="/opt/frequensolve/bin/FS_seismic",
            work_dir="/synthetic/work",
        )


def test_enterprise_profile_requires_configured_key_authentication(monkeypatch):
    configured_key = object()
    site = SimpleNamespace(
        enterprise_hpc=object(),
        config=SimpleNamespace(
            ssh_port=22,
            known_hosts_file=None,
        ),
        credentials=SimpleNamespace(username="scientist", ssh_key=configured_key),
    )
    authentication_methods = []

    class _Transport:
        def __init__(self, sock):
            self.authenticated = False

        def start_client(self, timeout):
            pass

        def auth_publickey(self, username, key):
            authentication_methods.append(("key", key))
            self.authenticated = True

        def auth_interactive(self, username, handler):
            authentication_methods.append(("interactive", None))

        def is_authenticated(self):
            return self.authenticated

        def set_keepalive(self, seconds):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        hpc_auth.socket,
        "create_connection",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(hpc_auth, "Transport", _Transport)
    monkeypatch.setattr(
        hpc_auth,
        "_verify_server_host_key",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "paramiko.agent.Agent",
        lambda: pytest.fail("Enterprise profile must not use the SSH agent"),
    )

    SlurmAuthenticator(site)._interactive_authentication("login.example.invalid")

    assert authentication_methods == [("key", configured_key)]


def _manifest_snapshot(payload, digest):
    raw = json.dumps(payload).encode()
    return digest + "\n" + base64.b64encode(raw).decode()


def _successful_remote_runner(site, *, identity=None, calls=None):
    profile = site.enterprise_hpc
    bundle, compatibility = contract_pair()
    bundle_schema, compatibility_schema = contract_schemas()
    snapshots = {
        profile.installed_path(BUNDLE_MANIFEST_RELATIVE): _manifest_snapshot(
            bundle, BUNDLE_MANIFEST_SHA
        ),
        profile.installed_path(COMPATIBILITY_MANIFEST_RELATIVE): _manifest_snapshot(
            compatibility, COMPATIBILITY_SHA
        ),
        profile.installed_path(BUNDLE_SCHEMA_RELATIVE): _manifest_snapshot(
            bundle_schema, BUNDLE_SCHEMA_SHA
        ),
        profile.installed_path(COMPATIBILITY_SCHEMA_RELATIVE): _manifest_snapshot(
            compatibility_schema, COMPATIBILITY_SCHEMA_SHA
        ),
    }
    solver_identity = identity or SOLVER_IDENTITY

    def run_login_cmd(command, timeout=None):
        if calls is not None:
            calls.append(command)
        if command == "scontrol --version":
            output = "slurm 24.05.5"
        elif "base64" in command:
            output = next(
                snapshot for path, snapshot in snapshots.items() if path in command
            )
        elif "--identity-json" in command:
            output = "\n".join(
                [
                    "frequensolve-frequensolver-identity-begin",
                    json.dumps(solver_identity),
                    "frequensolve-frequensolver-identity-ok",
                ]
            )
        elif "/opt/frequensolve/bin/FS_seismic" in command and "hashlib" in command:
            output = SOLVER_SHA
        else:
            output = ""
        return None, _Stream(output), _Stream("")

    return run_login_cmd


def test_preflight_observes_bundle_scheduler_and_public_solver_identity(monkeypatch):
    site = _enterprise_site(monkeypatch)
    calls = []
    monkeypatch.setattr(
        site, "run_login_cmd", _successful_remote_runner(site, calls=calls)
    )

    result = site._enterprise_hpc_preflight()

    assert result.scheduler_version == "slurm 24.05.5"
    assert result.solver_source_commit == COMMIT
    assert sum("--identity-json" in command for command in calls) == 1
    assert all(not command.lstrip().startswith("sbatch ") for command in calls)
    runtime_commands = [command for command in calls if "python3" in command]
    assert runtime_commands
    assert all(
        command.startswith("module load frequensolve/1.0.0-synthetic && ")
        for command in runtime_commands
    )
    assert all("/bin/activate" not in command for command in runtime_commands)


def test_preflight_rejects_unproven_sdk_before_remote_use(monkeypatch):
    site = _enterprise_site(monkeypatch)
    calls = []
    monkeypatch.setattr(hpc, "installed_frequensolve_artifact_sha256", lambda: None)
    monkeypatch.setattr(
        site,
        "run_login_cmd",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(EnterpriseHPCPreflightError, match="artifact provenance"):
        site._enterprise_hpc_preflight()

    assert calls == []


def test_installed_sdk_artifact_uses_pip_direct_url_and_record(monkeypatch, tmp_path):
    package = tmp_path / "frequensolve" / "module.py"
    metadata = tmp_path / "frequensolve-1.0.dist-info"
    package.parent.mkdir()
    metadata.mkdir()
    package.write_text("synthetic installed package\n")
    cache = package.parent / "__pycache__" / "module.cpython-312.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"synthetic generated cache")
    direct_url = json.dumps(
        {
            "archive_info": {
                "hash": "sha256=" + SDK_SHA,
                "hashes": {"sha256": SDK_SHA},
            },
            "url": "file:///synthetic/frequensolve.whl",
        }
    )
    direct_url_path = metadata / "direct_url.json"
    direct_url_path.write_text(direct_url)

    def record_hash(path):
        digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
        return digest.decode("ascii").rstrip("=")

    record = "\n".join(
        [
            f"frequensolve/module.py,sha256={record_hash(package)},{package.stat().st_size}",
            "frequensolve/__pycache__/module.cpython-312.pyc,,",
            "frequensolve-1.0.dist-info/direct_url.json,"
            f"sha256={record_hash(direct_url_path)},{direct_url_path.stat().st_size}",
            "frequensolve-1.0.dist-info/RECORD,,",
        ]
    )

    class _Distribution:
        def read_text(self, name):
            return {"direct_url.json": direct_url, "RECORD": record}[name]

        def locate_file(self, relative):
            return tmp_path / relative

    monkeypatch.setattr(
        enterprise_contract,
        "distribution",
        lambda name: _Distribution(),
    )

    assert installed_frequensolve_artifact_sha256() == SDK_SHA
    package.write_text("synthetic modified package\n")
    assert installed_frequensolve_artifact_sha256() is None


def test_attached_submit_validates_observed_pool_before_transfer(monkeypatch):
    site = _enterprise_site(monkeypatch)
    site.pool.id = "24680"
    site.pool.nhost = 3
    site.pool.nproc = 8
    site.pool.ncore = 16
    transferred = []
    monkeypatch.setattr(_Site, "provisioned", property(lambda self: True))

    def preflight(**kwargs):
        validated = validate_contract()
        site._enterprise_hpc_limits = validated.limits
        return validated.result

    monkeypatch.setattr(site, "_enterprise_hpc_preflight", preflight)
    monkeypatch.setattr(site, "prepare_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        site,
        "_submit_attached",
        lambda *args, **kwargs: transferred.append(args),
    )

    with pytest.raises(EnterpriseHPCPreflightError, match="max_nodes"):
        site.submit(object(), mode="attached", force=True)

    assert transferred == []


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"slurm_args": ["--nodes=999"]}, "raw slurm_args"),
        ({"run_path": "/synthetic/work; touch /tmp/pwned"}, "run_path overrides"),
    ],
)
def test_enterprise_preflight_rejects_untyped_overrides_before_remote_use(
    monkeypatch, override, error
):
    site = _enterprise_site(monkeypatch)
    config = site.run_config.merged(**override)
    calls = []
    monkeypatch.setattr(
        site,
        "run_login_cmd",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(EnterpriseHPCPreflightError, match=error):
        site._enterprise_hpc_preflight(run_config=config)

    assert calls == []


def test_enterprise_provision_preflights_before_transfer_or_scheduler_spend(
    monkeypatch,
):
    site = _enterprise_site(monkeypatch)
    side_effects = []

    def reject_preflight(*args, **kwargs):
        side_effects.append("preflight")
        raise EnterpriseHPCPreflightError("synthetic stop")

    monkeypatch.setattr(site, "_enterprise_hpc_preflight", reject_preflight)
    monkeypatch.setattr(site, "put", lambda *args, **kwargs: side_effects.append("put"))
    monkeypatch.setattr(
        site,
        "_submit_sbatch",
        lambda *args, **kwargs: side_effects.append("sbatch"),
    )

    with pytest.raises(EnterpriseHPCPreflightError, match="synthetic stop"):
        site.provision(nodes=1, tasks=4, duration="00:30:00")

    assert side_effects == ["preflight"]


def test_enterprise_provision_preserves_the_existing_sbatch_shape(monkeypatch):
    site = _enterprise_site(monkeypatch)
    events = []
    submissions = []
    monkeypatch.setattr(
        site,
        "_enterprise_hpc_preflight",
        lambda *args, **kwargs: events.append("preflight"),
    )
    monkeypatch.setattr(site, "put", lambda *args, **kwargs: events.append("put"))
    monkeypatch.setattr(
        site,
        "_submit_sbatch",
        lambda command: (
            events.append("sbatch") or submissions.append(command) or "24680"
        ),
    )
    monkeypatch.setattr(site, "_allocation_handle", lambda job_id: job_id)

    assert site.provision(nodes=1, tasks=4, duration="00:30:00") == "24680"
    assert events == ["preflight", "put", "preflight", "sbatch"]
    submission = shlex.split(submissions[0])
    assert submission[0] == "sbatch"
    assert len(submission) == 2


def test_run_record_contains_sanitized_enterprise_identities(monkeypatch, tmp_path):
    site = _enterprise_site(monkeypatch)
    site._enterprise_hpc_preflight_result = validate_contract().result
    project = Project(name="synthetic-project", path=tmp_path / "project")
    simulation = project.new_simulation(
        name="synthetic-simulation",
        physics="acoustic",
        dimension=2,
    )
    simulation.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1])
    )
    job = FrequencyDomainJob(
        name="synthetic-job",
        simulation=simulation,
        f_list=[10.0],
    )
    job.save()
    monkeypatch.setattr(site, "_store_remote_run_records", lambda job, record: None)

    record = site._record_site_run(job, scheduler_id="24680")

    evidence = record.metadata["enterprise_hpc"]
    assert record.scheduler_id == "24680"
    assert evidence["bundle_manifest_sha256"] == BUNDLE_MANIFEST_SHA
    assert evidence["compatibility_document_sha256"] == COMPATIBILITY_SHA
    serialized = json.dumps(evidence, sort_keys=True)
    assert "/opt/" not in serialized
    assert "account" not in serialized
    assert "credential" not in serialized
