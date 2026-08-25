import base64
import hashlib
import json

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
from frequensolve.orchestrator.sites.hpc import enterprise as enterprise_contract
from frequensolve.orchestrator.sites.hpc import site as hpc
from frequensolve.orchestrator.sites.hpc.enterprise import (
    EnterpriseHPCPreflightError,
    EnterpriseHPCProfile,
    installed_frequensolve_artifact_sha256,
    validate_bundle_pair,
)
from frequensolve.project.project import Project
from frequensolve.simulation.jobs import FrequencyDomainJob

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SOLVER_SHA = "9" * 64
COMMIT = "1" * 40


def profile_values():
    return {
        "schema": "frequensolve-enterprise-hpc-profile/v1",
        "profile_id": "synthetic-private-slurm",
        "host": "login.example.invalid",
        "bundle_manifest": "/opt/frequensolve/manifests/bundle.json",
        "compatibility_manifest": "/opt/frequensolve/manifests/compatibility.json",
        "bundle_schema_path": "/opt/frequensolve/schemas/bundle-manifest.schema.json",
        "compatibility_schema_path": "/opt/frequensolve/schemas/compatibility-row.schema.json",
        "bundle_root": "/opt/frequensolve",
        "solver_path": "/opt/frequensolve/bin/FS_seismic",
        "work_dir": "/synthetic/work",
        "scratch_dir": None,
        "bundle_version": "1.0.0-synthetic",
        "bundle_content_sha256": SHA_A,
        "bundle_manifest_sha256": "e" * 64,
        "bundle_schema_sha256": "7" * 64,
        "compatibility_row_id": "synthetic-private-slurm-row",
        "compatibility_document_sha256": SHA_B,
        "compatibility_schema_sha256": "8" * 64,
        "solver_version": "1.2.3",
        "solver_source_commit": COMMIT,
        "solver_build_id": "synthetic-build",
        "solver_build_identity_sha256": SHA_C,
        "frequensolve_version": frequensolve_version,
        "frequensolve_artifact_sha256": "d" * 64,
        "allowed_partitions": ["synthetic"],
        "max_nodes": 2,
        "max_ranks": 16,
        "max_threads_per_rank": 8,
        "max_wall_time": "00:30:00",
        "support_tier": "experimental",
        "module": "frequensolve/1.0.0-synthetic",
        "python_environment": "/opt/frequensolve/python-env",
        "account": "synthetic-account",
        "qos": "synthetic-qos",
        "mpi_launcher": "srun",
    }


def contract_pair():
    bundle = {
        "schemaVersion": "frequensolve-enterprise-hpc/v1",
        "bundleName": "frequensolve-enterprise-hpc",
        "bundleVersion": "1.0.0-synthetic",
        "releaseState": "candidate",
        "createdAt": "2026-08-24T12:00:00Z",
        "stateChangedAt": "2026-08-24T12:00:00Z",
        "contentSha256": SHA_A,
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
                "producerIdentitySha256": SHA_C,
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
                "sha256": "d" * 64,
            },
        },
        "installedFiles": [
            {
                "path": "bin/FS_seismic",
                "kind": "file",
                "mode": "0755",
                "ownership": {"user": "install-owner", "group": "install-group"},
                "size": 123,
                "sha256": SOLVER_SHA,
            }
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
            "documentSha256": SHA_B,
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
            "buildIdentitySha256": SHA_C,
            "variants": ["double-mpi"],
        },
        "frequensolve": {
            "version": frequensolve_version,
            "artifactSha256": "d" * 64,
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
    draft = "https://json-schema.org/draft/2020-12/schema"
    if bundle is None or compatibility is None:
        bundle, compatibility = contract_pair()
    return (
        {
            **_required_synthetic_shape(bundle),
            "$schema": draft,
            "$id": (
                "https://schemas.frequensol.com/frequensolve-enterprise-hpc/v1/"
                "bundle-manifest.schema.json"
            ),
            "title": "Synthetic test projection of the Deployment bundle schema",
        },
        {
            **_required_synthetic_shape(compatibility),
            "$schema": draft,
            "$id": (
                "https://schemas.frequensol.com/frequensolve-enterprise-hpc/v1/"
                "compatibility-row.schema.json"
            ),
            "title": "Synthetic test projection of the Deployment compatibility schema",
        },
    )


def schema_arguments():
    bundle_schema, compatibility_schema = contract_schemas()
    return {
        "bundle_schema": bundle_schema,
        "compatibility_schema": compatibility_schema,
    }


def test_profile_is_closed_and_rejects_unsafe_paths_and_shell_values():
    values = profile_values()
    profile = EnterpriseHPCProfile.from_mapping(values)

    assert profile.allowed_partitions == ("synthetic",)
    assert profile.solver_path == "/opt/frequensolve/bin/FS_seismic"

    for name, value in (
        ("solver_path", "/opt/frequensolve/../customer/solver"),
        ("solver_path", "/opt/frequensolve/bin/FS_seismic;touch-pwned"),
        ("bundle_manifest", "/opt/frequensolve/manifests/bundle.json;touch-pwned"),
        ("work_dir", "/synthetic/work;touch-pwned"),
        ("host", "login.example.invalid;touch /tmp/pwned"),
        ("module", "frequensolve/good\nmodule load hostile"),
        ("qos", "good;scancel 1"),
    ):
        invalid = {**values, name: value}
        with pytest.raises(ValueError, match="enterprise_hpc"):
            EnterpriseHPCProfile.from_mapping(invalid)

    with pytest.raises(ValueError, match="Unsupported enterprise_hpc"):
        EnterpriseHPCProfile.from_mapping({**values, "azure_subscription": "fake"})


def test_profile_rejects_contradictions_and_unbounded_resources():
    profile = EnterpriseHPCProfile.from_mapping(profile_values())

    profile.validate_site(
        host="login.example.invalid",
        partition="synthetic",
        account="synthetic-account",
        qos="synthetic-qos",
        mpi_launcher="srun",
    )
    profile.validate_resources(
        nodes=2,
        ranks_per_node=8,
        cores_per_node=64,
        duration_seconds=1800,
        max_wall_time_seconds=1800,
    )

    with pytest.raises(ValueError, match="not allowlisted"):
        profile.validate_site(
            host="login.example.invalid",
            partition="unapproved",
            account="synthetic-account",
            qos="synthetic-qos",
            mpi_launcher="srun",
        )
    unrestricted_account = EnterpriseHPCProfile.from_mapping(
        {**profile_values(), "account": None}
    )
    unrestricted_account.validate_site(
        host="login.example.invalid",
        partition="synthetic",
        account="",
        qos="synthetic-qos",
        mpi_launcher="srun",
    )
    with pytest.raises(ValueError, match="enterprise_hpc.account"):
        unrestricted_account.validate_site(
            host="login.example.invalid",
            partition="synthetic",
            account="safe;scancel 1",
            qos="synthetic-qos",
            mpi_launcher="srun",
        )
    with pytest.raises(ValueError, match="max_ranks"):
        profile.validate_resources(
            nodes=2,
            ranks_per_node=16,
            cores_per_node=64,
            duration_seconds=1800,
            max_wall_time_seconds=1800,
        )
    with pytest.raises(ValueError, match="max_nodes"):
        profile.validate_resources(
            nodes=3,
            ranks_per_node=1,
            cores_per_node=64,
            duration_seconds=1800,
            max_wall_time_seconds=1800,
        )
    with pytest.raises(ValueError, match="max_wall_time"):
        profile.validate_resources(
            nodes=1,
            ranks_per_node=8,
            cores_per_node=64,
            duration_seconds=1801,
            max_wall_time_seconds=1800,
        )


def test_bundle_pair_records_only_exact_sanitized_identities():
    profile = EnterpriseHPCProfile.from_mapping(profile_values())
    bundle, compatibility = contract_pair()

    result = validate_bundle_pair(
        profile,
        bundle,
        compatibility,
        bundle_sha256="e" * 64,
        compatibility_sha256=SHA_B,
        **schema_arguments(),
        scheduler_version="slurm 24.05.5",
        cores_per_node=8,
    )

    evidence = result.to_evidence()
    assert evidence["bundle_content_sha256"] == SHA_A
    assert evidence["solver_source_commit"] == COMMIT
    assert evidence["frequensolve_artifact_sha256"] == "d" * 64
    assert "bundle_manifest" not in evidence
    assert "account" not in evidence


@pytest.mark.parametrize(
    "mutation",
    [
        ("bundle", ("contentSha256",), "e" * 64),
        ("bundle", ("upstreamArtifacts", "sauce", "sourceCommit"), "2" * 40),
        (
            "compatibility",
            ("solver", "buildIdentitySha256"),
            "f" * 64,
        ),
        ("compatibility", ("frequensolve", "artifactSha256"), "0" * 64),
    ],
)
def test_bundle_pair_fails_closed_on_any_identity_mismatch(mutation):
    profile = EnterpriseHPCProfile.from_mapping(profile_values())
    bundle, compatibility = contract_pair()
    document_name, path, value = mutation
    document = bundle if document_name == "bundle" else compatibility
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(EnterpriseHPCPreflightError, match="compatibility mismatch"):
        validate_bundle_pair(
            profile,
            bundle,
            compatibility,
            bundle_sha256="e" * 64,
            compatibility_sha256=SHA_B,
            **schema_arguments(),
            scheduler_version="slurm 24.05.5",
            cores_per_node=8,
        )


def test_bundle_pair_rejects_schema_valid_withdrawn_bundle():
    profile = EnterpriseHPCProfile.from_mapping(profile_values())
    bundle, compatibility = contract_pair()
    bundle["releaseState"] = "withdrawn"
    bundle["withdrawal"] = {"reason": "Synthetic withdrawal test"}
    bundle_schema, compatibility_schema = contract_schemas(bundle, compatibility)

    with pytest.raises(EnterpriseHPCPreflightError, match="withdrawn"):
        validate_bundle_pair(
            profile,
            bundle,
            compatibility,
            bundle_sha256="e" * 64,
            compatibility_sha256=SHA_B,
            bundle_schema=bundle_schema,
            compatibility_schema=compatibility_schema,
            scheduler_version="slurm 24.05.5",
            cores_per_node=8,
        )


@pytest.mark.parametrize(
    "document_name,path",
    [
        ("compatibility", ("platform",)),
        ("compatibility", ("toolchain", "launcher")),
        ("compatibility", ("storage",)),
        ("compatibility", ("limits", "maxNodes")),
        ("compatibility", ("evidence",)),
        ("bundle", ("installedFiles",)),
        ("bundle", ("executables",)),
    ],
)
def test_bundle_pair_rejects_incomplete_deployment_v1_contracts(document_name, path):
    profile = EnterpriseHPCProfile.from_mapping(profile_values())
    bundle, compatibility = contract_pair()
    document = bundle if document_name == "bundle" else compatibility
    target = document
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]

    with pytest.raises(EnterpriseHPCPreflightError, match="Enterprise HPC"):
        validate_bundle_pair(
            profile,
            bundle,
            compatibility,
            bundle_sha256="e" * 64,
            compatibility_sha256=SHA_B,
            **schema_arguments(),
            scheduler_version="slurm 24.05.5",
            cores_per_node=8,
        )


def test_bundle_pair_rejects_malformed_nested_metadata_without_raw_attribute_error():
    profile = EnterpriseHPCProfile.from_mapping(profile_values())
    bundle, compatibility = contract_pair()
    compatibility["toolchain"] = {"launcher": None}

    with pytest.raises(EnterpriseHPCPreflightError, match="Enterprise HPC"):
        validate_bundle_pair(
            profile,
            bundle,
            compatibility,
            bundle_sha256="e" * 64,
            compatibility_sha256=SHA_B,
            **schema_arguments(),
            scheduler_version="slurm 24.05.5",
            cores_per_node=8,
        )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (("toolchain", "slurm", "24.05.4"), "compatibility scheduler"),
        (("toolchain", "launcher", "mpirun"), "compatibility launcher"),
        (("limits", "maxNodes", 99), "compatibility max nodes"),
        (("platform", "cpu", "coresPerNode", 64), "compatibility cores per node"),
    ],
)
def test_bundle_pair_binds_certified_scheduler_launcher_and_limits(mutation, error):
    profile = EnterpriseHPCProfile.from_mapping(profile_values())
    bundle, compatibility = contract_pair()
    *path, value = mutation
    target = compatibility
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(EnterpriseHPCPreflightError, match=error):
        validate_bundle_pair(
            profile,
            bundle,
            compatibility,
            bundle_sha256="e" * 64,
            compatibility_sha256=SHA_B,
            **schema_arguments(),
            scheduler_version="slurm 24.05.5",
            cores_per_node=8,
        )


def test_bundle_pair_binds_selected_solver_path_and_variant():
    profile = EnterpriseHPCProfile.from_mapping(profile_values())
    bundle, compatibility = contract_pair()
    bundle["executables"][0]["path"] = "bin/other-solver"

    with pytest.raises(EnterpriseHPCPreflightError, match="not uniquely declared"):
        validate_bundle_pair(
            profile,
            bundle,
            compatibility,
            bundle_sha256="e" * 64,
            compatibility_sha256=SHA_B,
            **schema_arguments(),
            scheduler_version="slurm 24.05.5",
            cores_per_node=8,
        )


def test_profile_fixture_contains_no_provider_or_credential_contract():
    serialized = json.dumps(profile_values(), sort_keys=True)
    assert "azure" not in serialized.lower()
    assert "password" not in serialized.lower()
    assert "license" not in serialized.lower()


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
    values = profile_values()
    profile = EnterpriseHPCProfile.from_mapping(values)
    config = SlurmSiteConfig(
        hostname=profile.host,
        queue="synthetic",
        mpi_wrapper="srun",
        account="synthetic-account",
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
        qos="synthetic-qos",
    )
    monkeypatch.setattr(hpc, "SSHClientClass", _WrappedLogin)
    monkeypatch.setattr(
        hpc,
        "installed_frequensolve_artifact_sha256",
        lambda: "d" * 64,
    )
    return _Site(
        config=config,
        run_config=run_config,
        enterprise_hpc=profile,
        solver=profile.solver_path,
        work_dir="/synthetic/work",
    )


def _manifest_snapshot(payload, digest):
    raw = json.dumps(payload).encode()
    return digest + "\n" + base64.b64encode(raw).decode()


def test_preflight_observes_scheduler_manifests_and_public_solver_identity(monkeypatch):
    site = _enterprise_site(monkeypatch)
    profile = site.enterprise_hpc
    bundle, compatibility = contract_pair()
    bundle_schema, compatibility_schema = contract_schemas()
    calls = []

    def run_login_cmd(command, timeout=None):
        calls.append(command)
        if command == "scontrol --version":
            output = "slurm 24.05.5"
        elif "base64" in command and profile.bundle_manifest in command:
            output = _manifest_snapshot(bundle, "e" * 64)
        elif "base64" in command and profile.compatibility_manifest in command:
            output = _manifest_snapshot(compatibility, SHA_B)
        elif "base64" in command and profile.bundle_schema_path in command:
            output = _manifest_snapshot(bundle_schema, "7" * 64)
        elif "base64" in command and profile.compatibility_schema_path in command:
            output = _manifest_snapshot(compatibility_schema, "8" * 64)
        elif profile.solver_path in command and "hashlib" in command:
            output = SOLVER_SHA
        elif "--identity-json" in command:
            output = "\n".join(
                [
                    "frequensolve-frequensolver-identity-begin",
                    json.dumps(
                        {
                            "schema": "frequensolver-identity-1",
                            "product": "FrequenSolver",
                            "version": "1.2.3",
                            "build_id": "synthetic-build",
                            "git_commit": COMMIT,
                        }
                    ),
                    "frequensolve-frequensolver-identity-ok",
                ]
            )
        else:
            output = ""
        return None, _Stream(output), _Stream("")

    monkeypatch.setattr(site, "run_login_cmd", run_login_cmd)

    result = site.enterprise_hpc_preflight()
    refreshed = site.enterprise_hpc_preflight()

    assert result == refreshed
    assert result.scheduler_version == "slurm 24.05.5"
    assert result.solver_source_commit == COMMIT
    assert sum("--identity-json" in command for command in calls) == 2
    launcher_probes = [command for command in calls if "command -v srun" in command]
    assert len(launcher_probes) == 2
    assert all(
        "module load frequensolve/1.0.0-synthetic" in command
        for command in launcher_probes
    )
    assert all(not command.startswith(("cat ", "sha256sum ")) for command in calls)
    assert all(not command.lstrip().startswith("sbatch ") for command in calls)


def test_preflight_rejects_multiline_scheduler_evidence(monkeypatch):
    site = _enterprise_site(monkeypatch)

    def run_login_cmd(command, timeout=None):
        output = (
            "slurm 24.05.5\nsynthetic-host-secret"
            if command == "scontrol --version"
            else ""
        )
        return None, _Stream(output), _Stream("")

    monkeypatch.setattr(site, "run_login_cmd", run_login_cmd)

    with pytest.raises(EnterpriseHPCPreflightError, match="unexpected scheduler"):
        site.enterprise_hpc_preflight()


def test_preflight_rejects_unproven_local_sdk_artifact_before_remote_use(monkeypatch):
    site = _enterprise_site(monkeypatch)
    calls = []
    monkeypatch.setattr(hpc, "installed_frequensolve_artifact_sha256", lambda: None)
    monkeypatch.setattr(
        site, "run_login_cmd", lambda *args, **kwargs: calls.append(args)
    )

    with pytest.raises(EnterpriseHPCPreflightError, match="artifact provenance"):
        site.enterprise_hpc_preflight()

    assert calls == []


def test_installed_sdk_artifact_uses_pip_direct_url_and_record(monkeypatch, tmp_path):
    package = tmp_path / "frequensolve" / "module.py"
    metadata = tmp_path / "frequensolve-1.0.dist-info"
    package.parent.mkdir()
    metadata.mkdir()
    package.write_text("synthetic installed package\n")
    direct_url = json.dumps(
        {
            "archive_info": {
                "hash": "sha256=" + "d" * 64,
                "hashes": {"sha256": "d" * 64},
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
        enterprise_contract, "distribution", lambda name: _Distribution()
    )

    assert installed_frequensolve_artifact_sha256() == "d" * 64

    package.write_text("synthetic modified package\n")
    assert installed_frequensolve_artifact_sha256() is None


@pytest.mark.parametrize(
    ("nodes", "ranks", "cores", "message"),
    [
        (3, 8, 16, "max_nodes"),
        (1, 17, 32, "max_ranks"),
        (1, 4, 10, "divide evenly"),
        (1, 2, 18, "max_threads_per_rank"),
    ],
)
def test_enterprise_profile_rejects_unbounded_attached_allocation(
    nodes, ranks, cores, message
):
    profile = EnterpriseHPCProfile.from_mapping(profile_values())

    with pytest.raises(ValueError, match=message):
        profile.validate_allocation(nodes=nodes, ranks=ranks, cores=cores)


def test_attached_submit_validates_actual_pool_before_transfer(monkeypatch):
    site = _enterprise_site(monkeypatch)
    site.pool.id = "24680"
    site.pool.nhost = 3
    site.pool.nproc = 8
    site.pool.ncore = 16
    transferred = []
    monkeypatch.setattr(_Site, "provisioned", property(lambda self: True))
    monkeypatch.setattr(site, "enterprise_hpc_preflight", lambda **kwargs: None)
    monkeypatch.setattr(site, "prepare_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        site, "_submit_attached", lambda *args, **kwargs: transferred.append(args)
    )

    with pytest.raises(ValueError, match="max_nodes"):
        site.submit(object(), mode="attached", force=True)

    assert transferred == []


def test_preflight_fails_before_submission_when_solver_identity_differs(monkeypatch):
    site = _enterprise_site(monkeypatch)
    profile = site.enterprise_hpc
    bundle, compatibility = contract_pair()
    bundle_schema, compatibility_schema = contract_schemas()
    submitted = []

    def run_login_cmd(command, timeout=None):
        if command == "scontrol --version":
            output = "slurm 24.05.5"
        elif "base64" in command and profile.bundle_manifest in command:
            output = _manifest_snapshot(bundle, "e" * 64)
        elif "base64" in command and profile.compatibility_manifest in command:
            output = _manifest_snapshot(compatibility, SHA_B)
        elif "base64" in command and profile.bundle_schema_path in command:
            output = _manifest_snapshot(bundle_schema, "7" * 64)
        elif "base64" in command and profile.compatibility_schema_path in command:
            output = _manifest_snapshot(compatibility_schema, "8" * 64)
        elif profile.solver_path in command and "hashlib" in command:
            output = SOLVER_SHA
        elif "--identity-json" in command:
            output = "\n".join(
                [
                    "frequensolve-frequensolver-identity-begin",
                    json.dumps(
                        {
                            "schema": "frequensolver-identity-1",
                            "product": "FrequenSolver",
                            "version": "9.9.9",
                            "build_id": "wrong-build",
                            "git_commit": "9" * 40,
                        }
                    ),
                    "frequensolve-frequensolver-identity-ok",
                ]
            )
        else:
            output = ""
        return None, _Stream(output), _Stream("")

    monkeypatch.setattr(site, "run_login_cmd", run_login_cmd)
    monkeypatch.setattr(
        site, "_submit_sbatch", lambda command: submitted.append(command)
    )

    with pytest.raises(
        EnterpriseHPCPreflightError,
        match="public identity does not match",
    ):
        site.enterprise_hpc_preflight()

    assert submitted == []


@pytest.mark.parametrize(
    "override,error",
    [
        ({"slurm_args": ["--nodes=999"]}, "raw slurm_args"),
        ({"run_path": "/synthetic/work; touch /tmp/pwned"}, "run_path overrides"),
    ],
)
def test_enterprise_preflight_rejects_untyped_submission_overrides_before_remote_use(
    monkeypatch, override, error
):
    site = _enterprise_site(monkeypatch)
    config = site.run_config.merged(**override)
    calls = []
    monkeypatch.setattr(
        site, "run_login_cmd", lambda *args, **kwargs: calls.append(args)
    )

    with pytest.raises(EnterpriseHPCPreflightError, match=error):
        site.enterprise_hpc_preflight(run_config=config)

    assert calls == []


def test_enterprise_provision_preflights_before_transfer_or_scheduler_spend(
    monkeypatch,
):
    site = _enterprise_site(monkeypatch)
    side_effects = []

    def reject_preflight(*args, **kwargs):
        side_effects.append("preflight")
        raise EnterpriseHPCPreflightError("synthetic stop")

    monkeypatch.setattr(site, "enterprise_hpc_preflight", reject_preflight)
    monkeypatch.setattr(site, "put", lambda *args, **kwargs: side_effects.append("put"))
    monkeypatch.setattr(
        site, "_submit_sbatch", lambda *args, **kwargs: side_effects.append("sbatch")
    )

    with pytest.raises(EnterpriseHPCPreflightError, match="synthetic stop"):
        site.provision(nodes=1, tasks=4, duration="00:30:00")

    assert side_effects == ["preflight"]


def test_enterprise_provision_rechecks_after_transfer_immediately_before_sbatch(
    monkeypatch,
):
    site = _enterprise_site(monkeypatch)
    events = []
    monkeypatch.setattr(
        site,
        "enterprise_hpc_preflight",
        lambda *args, **kwargs: events.append("preflight"),
    )
    monkeypatch.setattr(site, "put", lambda *args, **kwargs: events.append("put"))
    monkeypatch.setattr(
        site,
        "_submit_sbatch",
        lambda *args, **kwargs: events.append("sbatch") or "24680",
    )
    monkeypatch.setattr(site, "_allocation_handle", lambda job_id: job_id)

    assert site.provision(nodes=1, tasks=4, duration="00:30:00") == "24680"
    assert events == ["preflight", "put", "preflight", "sbatch"]


@pytest.mark.parametrize("field", ["queue", "account", "qos", "notify_email"])
def test_slurm_directive_values_reject_newline_injection(field):
    with pytest.raises(ValueError, match="safe SLURM directive"):
        SlurmRunConfig(**{field: "safe\n#SBATCH --nodes=999"})


def test_safe_existing_slurm_partition_list_and_email_characters_remain_supported():
    config = SlurmRunConfig(
        queue="cpu,gpu",
        notify_email="first.last%tag@example.com",
    )

    assert config.queue == "cpu,gpu"
    assert config.notify_email == "first.last%tag@example.com"


def test_enterprise_qos_is_rendered_as_a_bounded_scheduler_directive(monkeypatch):
    site = _enterprise_site(monkeypatch)

    script = site._generate_provision_script(
        n_nodes=1,
        ranks_per_node=4,
        duration="00:30:00",
        queue="synthetic",
        account="synthetic-account",
        qos="synthetic-qos",
    )

    assert "#SBATCH --qos=synthetic-qos" in script
    assert script.count("synthetic-qos") == 1


def test_run_record_contains_sanitized_enterprise_identities(monkeypatch, tmp_path):
    site = _enterprise_site(monkeypatch)
    profile = site.enterprise_hpc
    bundle, compatibility = contract_pair()
    site._enterprise_hpc_preflight_result = validate_bundle_pair(
        profile,
        bundle,
        compatibility,
        bundle_sha256="e" * 64,
        compatibility_sha256=SHA_B,
        **schema_arguments(),
        scheduler_version="slurm 24.05.5",
        cores_per_node=8,
    )
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
    assert evidence["bundle_manifest_sha256"] == "e" * 64
    assert evidence["compatibility_document_sha256"] == SHA_B
    assert evidence["scheduler_version"] == "slurm 24.05.5"
    serialized = json.dumps(evidence, sort_keys=True)
    assert "/opt/" not in serialized
    assert "account" not in serialized
    assert "credential" not in serialized
