from __future__ import annotations

import json
import subprocess

import pytest

from frequensolve import solver
from frequensolve.orchestrator.sites.hpc import site as hpc_site_module
from frequensolve.orchestrator.sites.hpc.site import SlurmSite
from frequensolve.orchestrator.sites.local import site as local_site_module
from frequensolve.orchestrator.sites.local.site import LocalSite
from frequensolve.solver import (
    IDENTITY_QUERY_TIMEOUT_SECONDS,
    PreferredSolver,
    SolverCompatibility,
    SolverCompatibilityError,
    SolverCompatibilityManifest,
    SolverCompatibilityWarning,
    SolverIdentity,
    SolverIdentityQuery,
    check_solver_compatibility,
    load_solver_compatibility,
    query_local_solver_identity,
    query_remote_solver_identity,
)

COMMIT = "a" * 40


def _manifest() -> SolverCompatibilityManifest:
    return SolverCompatibilityManifest(
        package_release="0.3.0",
        preferred_solver=PreferredSolver(
            release="v0.1.0",
            git_commit=COMMIT,
            release_url=("https://github.com/FrequenSol/Sauce/releases/tag/v0.1.0"),
        ),
        evidence_run_id=456,
        evidence_url=(
            "https://github.com/FrequenSol/FrequenSolveDockerImage/actions/runs/456"
        ),
    )


def _standard_manifest() -> SolverCompatibilityManifest:
    return SolverCompatibilityManifest(
        package_release="0.3.0",
        preferred_solver=PreferredSolver(
            release="v0.1.0",
            git_commit=COMMIT,
            release_url=("https://github.com/FrequenSol/Sauce/releases/tag/v0.1.0"),
        ),
        evidence_run_id=123,
        evidence_url=("https://github.com/FrequenSol/FrequenSolve/actions/runs/123"),
        validation_profile="standard",
    )


def _identity(*, version: str = "v0.1.0", commit: str = COMMIT):
    return SolverIdentity(
        version=version,
        build_id="release-v0.1.0",
        git_commit=commit,
    )


def _identity_json(**overrides) -> str:
    payload = {
        "schema": "fs-solver-identity-1",
        "product": "FS_solver",
        "version": "v0.1.0",
        "build_id": "release-v0.1.0",
        "git_commit": COMMIT,
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_packaged_development_manifest_loads_without_query_or_warning():
    with warnings_not_emitted():
        manifest = load_solver_compatibility()

    assert manifest.schema == "frequensolve-solver-compatibility/v2"
    assert manifest.preferred_solver is None
    assert manifest.validation_profile is None
    assert not manifest.solver_backed


def test_manifest_preserves_legacy_positional_schema_argument():
    manifest = SolverCompatibilityManifest(
        "0.3.0",
        None,
        None,
        None,
        "frequensolve-solver-compatibility/v1",
    )

    assert manifest.schema == "frequensolve-solver-compatibility/v1"


class warnings_not_emitted:
    """Fail when compatibility warnings are unexpectedly emitted."""

    def __enter__(self):
        import warnings

        self._manager = warnings.catch_warnings(record=True)
        self._caught = self._manager.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._manager.__exit__(exc_type, exc, traceback)
        assert not [
            warning
            for warning in self._caught
            if issubclass(warning.category, SolverCompatibilityWarning)
        ]


def test_loader_validates_release_manifest(tmp_path):
    path = tmp_path / "compatibility.json"
    path.write_text(
        json.dumps(
            {
                "schema": "frequensolve-solver-compatibility/v1",
                "package_release": "0.3.0",
                "preferred_solver": {
                    "release": "v0.1.0",
                    "git_commit": COMMIT,
                    "release_url": (
                        "https://github.com/FrequenSol/Sauce/releases/tag/v0.1.0"
                    ),
                },
                "evidence": {
                    "run_id": 456,
                    "url": (
                        "https://github.com/FrequenSol/"
                        "FrequenSolveDockerImage/actions/runs/456"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_solver_compatibility(path)

    assert loaded.package_release == "0.3.0"
    assert loaded.preferred_solver.release == "v0.1.0"
    assert loaded.evidence_run_id == 456
    assert loaded.validation_profile == "solver-backed"
    assert loaded.solver_backed
    assert loaded.schema == "frequensolve-solver-compatibility/v1"


def test_loader_validates_standard_v2_manifest(tmp_path):
    path = tmp_path / "compatibility.json"
    path.write_text(
        json.dumps(
            {
                "schema": "frequensolve-solver-compatibility/v2",
                "package_release": "0.3.0",
                "preferred_solver": {
                    "release": "v0.1.0",
                    "git_commit": COMMIT,
                    "release_url": (
                        "https://github.com/FrequenSol/Sauce/releases/tag/v0.1.0"
                    ),
                },
                "validation": {
                    "profile": "standard",
                    "solver_backed": False,
                    "run_id": 123,
                    "url": (
                        "https://github.com/FrequenSol/FrequenSolve/actions/runs/123"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_solver_compatibility(path)

    assert loaded.validation_profile == "standard"
    assert not loaded.solver_backed
    assert loaded.evidence_run_id == 123


def test_loader_rejects_caller_run_url_for_downstream_evidence(tmp_path):
    path = tmp_path / "compatibility.json"
    path.write_text(
        json.dumps(
            {
                "schema": "frequensolve-solver-compatibility/v1",
                "package_release": "0.3.0",
                "preferred_solver": {
                    "release": "v0.1.0",
                    "git_commit": COMMIT,
                    "release_url": (
                        "https://github.com/FrequenSol/Sauce/releases/tag/v0.1.0"
                    ),
                },
                "evidence": {
                    "run_id": 456,
                    "url": (
                        "https://github.com/FrequenSol/FrequenSolve/actions/runs/456"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evidence.url"):
        load_solver_compatibility(path)


def test_loader_rejects_release_url_for_another_solver_release(tmp_path):
    path = tmp_path / "compatibility.json"
    path.write_text(
        json.dumps(
            {
                "schema": "frequensolve-solver-compatibility/v1",
                "package_release": "0.3.0",
                "preferred_solver": {
                    "release": "v0.1.0",
                    "git_commit": COMMIT,
                    "release_url": (
                        "https://github.com/FrequenSol/Sauce/releases/tag/v0.2.0"
                    ),
                },
                "evidence": {
                    "run_id": 456,
                    "url": (
                        "https://github.com/FrequenSol/"
                        "FrequenSolveDockerImage/actions/runs/456"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="immutable Solver release"):
        load_solver_compatibility(path)


@pytest.mark.parametrize(
    "schema,product",
    [
        ("fs-solver-identity-1", "FS_solver"),
    ],
)
def test_local_identity_query_calls_executable_directly(monkeypatch, schema, product):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command, 0, _identity_json(schema=schema, product=product), ""
        )

    monkeypatch.setattr(solver.subprocess, "run", fake_run)

    result = query_local_solver_identity(
        "/opt/Frequen Solver/fs3d",
        environment={"PATH": "/opt/bin"},
    )

    assert result.identity == SolverIdentity(
        version="v0.1.0",
        build_id="release-v0.1.0",
        git_commit=COMMIT,
        schema=schema,
        product=product,
    )
    assert seen["command"] == ["/opt/Frequen Solver/fs3d", "--identity-json"]
    assert seen["kwargs"]["env"] == {"PATH": "/opt/bin"}


def test_identity_query_accepts_current_sauce_identity_additively(monkeypatch):
    payload = json.loads(_identity_json())
    payload.update(schema="fs-solver-identity-1", product="FS_solver")
    monkeypatch.setattr(
        solver.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps(payload), ""
        ),
    )

    result = query_local_solver_identity("/solver")

    assert result.identity == SolverIdentity(
        version="v0.1.0",
        build_id="release-v0.1.0",
        git_commit=COMMIT,
        schema="fs-solver-identity-1",
        product="FS_solver",
    )


@pytest.mark.parametrize(
    ("schema", "product"),
    [
        ("fs-solver-identity-1", "Solver"),
        ("solver-identity-1", "FS_solver"),
        ("unknown-identity-1", "unknown"),
    ],
)
def test_identity_query_rejects_unknown_or_mixed_identity_pairs(
    monkeypatch, schema, product
):
    payload = json.loads(_identity_json())
    payload.update(schema=schema, product=product)
    monkeypatch.setattr(
        solver.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps(payload), ""
        ),
    )

    result = query_local_solver_identity("/solver")

    assert result.identity is None
    assert "schema/product pair is unsupported" in result.error


def test_remote_identity_query_quotes_command_and_runs_setup_directly():
    commands = []

    result = query_remote_solver_identity(
        "/work/Frequen Solver/FS_seismic",
        lambda command: (
            commands.append(command)
            or "module setup output\n"
            + "frequensolve-solver-identity-begin\n"
            + _identity_json()
            + "\nfrequensolve-solver-identity-ok\n"
        ),
        setup_commands=["module load intel/25.1"],
    )

    assert result.identity == _identity()
    assert commands == [
        "set -e\n"
        "module load intel/25.1\n"
        "printf '%s\\n' frequensolve-solver-identity-begin\n"
        "'/work/Frequen Solver/FS_seismic' --identity-json\n"
        "printf '%s\\n' frequensolve-solver-identity-ok"
    ]
    assert "mpirun" not in commands[0]
    assert "srun" not in commands[0]


def test_remote_identity_query_parses_pretty_json_after_setup_output():
    pretty_identity = json.dumps(json.loads(_identity_json()), indent=2)

    result = query_remote_solver_identity(
        "/work/FS_seismic",
        lambda command: (
            "module setup output\n"
            "frequensolve-solver-identity-begin\n"
            f"{pretty_identity}\n"
            "frequensolve-solver-identity-ok\n"
        ),
        setup_commands=["module load intel/25.1"],
    )

    assert result.identity == _identity()


def test_remote_identity_timeout_warns_without_starting_a_job():
    commands = []

    def time_out(command):
        commands.append(command)
        raise TimeoutError(
            f"SSH login command timed out after {IDENTITY_QUERY_TIMEOUT_SECONDS} seconds"
        )

    with pytest.warns(SolverCompatibilityWarning, match="timed out"):
        result = check_solver_compatibility(
            "/remote/FS_seismic",
            manifest=_manifest(),
            remote_runner=time_out,
        )

    assert result.status == "unknown"
    assert len(commands) == 1
    assert "--identity-json" in commands[0]
    assert all(token not in commands[0] for token in ("mpirun", "srun", "sbatch"))


def test_remote_identity_timeout_fails_strict_policy():
    commands = []

    def time_out(command):
        commands.append(command)
        raise TimeoutError(
            f"SSH login command timed out after {IDENTITY_QUERY_TIMEOUT_SECONDS} seconds"
        )

    with pytest.raises(SolverCompatibilityError, match="timed out"):
        check_solver_compatibility(
            "/remote/FS_seismic",
            manifest=_manifest(),
            policy="strict",
            remote_runner=time_out,
        )

    assert len(commands) == 1
    assert all(token not in commands[0] for token in ("mpirun", "srun", "sbatch"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("build_id"),
        lambda payload: payload.update(extra="unexpected"),
    ],
)
def test_identity_query_rejects_noncanonical_object(monkeypatch, mutation):
    payload = json.loads(_identity_json())
    mutation(payload)
    monkeypatch.setattr(
        solver.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps(payload), ""
        ),
    )

    result = query_local_solver_identity("/solver")

    assert result.identity is None
    assert "keys" in result.error


@pytest.mark.parametrize(
    "schema,product",
    [
        ("fs-solver-identity-1", "Solver"),
        ("solver-identity-1", "FS_solver"),
        ("fs-solver-identity-2", "FS_solver"),
        ({}, "FS_solver"),
    ],
)
def test_identity_query_rejects_mixed_or_unknown_identity(monkeypatch, schema, product):
    monkeypatch.setattr(
        solver.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, _identity_json(schema=schema, product=product), ""
        ),
    )
    result = query_local_solver_identity("/solver")
    assert result.identity is None
    assert "schema/product pair" in result.error


def test_identity_query_rejects_multiline_build_id(monkeypatch):
    monkeypatch.setattr(
        solver.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            _identity_json(build_id="release-v0.1.0\nmalicious-output"),
            "",
        ),
    )

    result = query_local_solver_identity("/solver")

    assert result.identity is None
    assert result.error == "identity.build_id must be a single-line string"


def test_warn_policy_reports_preferred_solver_without_sauce(monkeypatch):
    monkeypatch.setattr(
        solver,
        "query_local_solver_identity",
        lambda *args, **kwargs: SolverIdentityQuery(
            _identity(version="v0.2.0", commit="b" * 40)
        ),
    )

    with pytest.warns(SolverCompatibilityWarning) as caught:
        result = check_solver_compatibility(
            "/solver",
            manifest=_manifest(),
        )

    message = str(caught[0].message)
    assert result.status == "untested"
    assert "Preferred Solver: v0.1.0" in message
    assert "may result in unexpected behavior" in message
    assert "Sauce" not in message


def test_strict_policy_fails_unknown_identity(monkeypatch):
    monkeypatch.setattr(
        solver,
        "query_local_solver_identity",
        lambda *args, **kwargs: SolverIdentityQuery(
            None, "identity option is unavailable"
        ),
    )

    with pytest.raises(SolverCompatibilityError, match="Preferred Solver"):
        check_solver_compatibility(
            "/solver",
            manifest=_manifest(),
            policy="strict",
        )


def test_off_policy_does_not_query_solver(monkeypatch):
    def fail_query(*args, **kwargs):
        raise AssertionError("off policy queried the solver")

    monkeypatch.setattr(
        solver,
        "query_local_solver_identity",
        fail_query,
    )

    result = check_solver_compatibility(
        "/solver",
        manifest=_manifest(),
        policy="off",
    )

    assert result.status == "off"


def test_exact_release_and_commit_are_confirmed(monkeypatch):
    monkeypatch.setattr(
        solver,
        "query_local_solver_identity",
        lambda *args, **kwargs: SolverIdentityQuery(_identity()),
    )

    result = check_solver_compatibility(
        "/solver",
        manifest=_manifest(),
        policy="strict",
    )

    assert result.confirmed
    assert "matches preferred Solver" in result.message


def test_standard_profile_exact_identity_remains_untested(monkeypatch):
    monkeypatch.setattr(
        solver,
        "query_local_solver_identity",
        lambda *args, **kwargs: SolverIdentityQuery(_identity()),
    )

    with pytest.warns(
        SolverCompatibilityWarning,
        match="did not run solver-backed validation",
    ):
        result = check_solver_compatibility(
            "/solver",
            manifest=_standard_manifest(),
        )

    assert result.status == "untested"
    assert not result.confirmed


def test_standard_profile_exact_identity_fails_strict_policy(monkeypatch):
    monkeypatch.setattr(
        solver,
        "query_local_solver_identity",
        lambda *args, **kwargs: SolverIdentityQuery(_identity()),
    )

    with pytest.raises(
        SolverCompatibilityError,
        match="did not run solver-backed validation",
    ):
        check_solver_compatibility(
            "/solver",
            manifest=_standard_manifest(),
            policy="strict",
        )


def _confirmed_result():
    return SolverCompatibility(
        status="compatible",
        message="confirmed",
        manifest=_manifest(),
        identity=_identity(),
    )


def _bare_local_site():
    site = object.__new__(LocalSite)
    site.solver_policy = "warn"
    site._solver_compatibility_result = None
    site._solver_compatibility_policy = None
    site.executable = "/solver"
    site.solver = "/solver"
    site.env = {}
    return site


def _bare_slurm_site():
    site = object.__new__(SlurmSite)
    site.solver_policy = "warn"
    site._solver_compatibility_result = None
    site._solver_compatibility_policy = None
    site._executable = "/remote/solver"
    site.modules = []
    site.environment = {}
    site.run_login = lambda command, **kwargs: (
        _identity_json() + "\nfrequensolve-solver-identity-ok\n"
    )
    return site


def test_slurm_login_timeout_closes_paramiko_channel():
    class Channel:
        closed = False

        def close(self):
            self.closed = True

    class TimedOutStream:
        def __init__(self, channel):
            self.channel = channel

        def read(self):
            raise TimeoutError("timed out")

    class LoginClient:
        hostname = "login"

        def __init__(self, stdout):
            self.stdout = stdout
            self.calls = []

        def exec_command(self, command, *, timeout=None):
            self.calls.append((command, timeout))
            return None, self.stdout, None

    channel = Channel()
    client = LoginClient(TimedOutStream(channel))
    site = object.__new__(SlurmSite)
    site._login_client = client

    with pytest.raises(TimeoutError, match="15.0 seconds"):
        site.run_login(
            "FS_seismic --identity-json",
            timeout=IDENTITY_QUERY_TIMEOUT_SECONDS,
        )

    assert client.calls == [
        ("FS_seismic --identity-json", IDENTITY_QUERY_TIMEOUT_SECONDS)
    ]
    assert channel.closed


def test_slurm_compatibility_probe_uses_bounded_login_timeout(monkeypatch):
    site = _bare_slurm_site()
    calls = []

    def run_login(command, *, timeout=None):
        calls.append((command, timeout))
        return _identity_json() + "\nfrequensolve-solver-identity-ok\n"

    def check(executable, **kwargs):
        kwargs["remote_runner"]("identity probe")
        return _confirmed_result()

    site.run_login = run_login
    monkeypatch.setattr(hpc_site_module, "check_solver_compatibility", check)

    result = site.check_solver_compatibility()

    assert result.confirmed
    assert calls == [("identity probe", IDENTITY_QUERY_TIMEOUT_SECONDS)]


def test_local_site_checks_once_before_submission(monkeypatch):
    site = _bare_local_site()
    events = []
    monkeypatch.setattr(
        local_site_module,
        "check_solver_compatibility",
        lambda *args, **kwargs: events.append("compatibility") or _confirmed_result(),
    )
    site.prepare_job = lambda job, **kwargs: events.append("prepare") or job

    class CurrentJob:
        name = "current"

        def is_run_current(self):
            return True

        def write_run_state(self, **kwargs):
            return None

    site.submit(CurrentJob())
    site.submit(CurrentJob())

    assert events == ["compatibility", "prepare", "prepare"]


@pytest.mark.parametrize(
    ("site_factory", "site_class"),
    [(_bare_local_site, LocalSite), (_bare_slurm_site, SlurmSite)],
)
def test_strict_site_failure_prevents_prepare_and_submission(
    monkeypatch, site_factory, site_class
):
    site = site_factory()
    events = []

    def fail_check(*, policy=None, force=False):
        events.append("compatibility")
        raise SolverCompatibilityError("strict pair rejected")

    site.check_solver_compatibility = fail_check
    site.prepare_job = lambda job, **kwargs: events.append("prepare") or job

    with pytest.raises(SolverCompatibilityError, match="strict pair rejected"):
        site_class.submit(site, object(), solver_policy="strict")

    assert events == ["compatibility"]


def test_slurm_site_caches_one_check_per_policy(monkeypatch):
    site = _bare_slurm_site()
    policies = []
    monkeypatch.setattr(
        hpc_site_module,
        "check_solver_compatibility",
        lambda *args, **kwargs: (
            policies.append(kwargs["policy"]) or _confirmed_result()
        ),
    )

    site.check_solver_compatibility()
    site.check_solver_compatibility()
    site.check_solver_compatibility(policy="strict")

    assert policies == ["warn", "strict"]


@pytest.mark.parametrize(
    "explicit,current,previous,legacy,expected",
    [
        (None, None, None, None, "warn"),
        (None, None, None, "strict", "strict"),
        (None, None, None, "off", "off"),
        (None, None, None, " STRICT ", "strict"),
        (None, None, "off", "strict", "off"),
        (None, None, "strict", "invalid", "strict"),
        (None, "off", "strict", "strict", "off"),
        (None, "strict", "invalid", "invalid", "strict"),
        ("warn", "strict", "off", "off", "warn"),
        ("off", "invalid", "invalid", "invalid", "off"),
    ],
)
def test_solver_policy_environment_alias_precedence(
    monkeypatch, explicit, current, previous, legacy, expected
):
    for key, value in (
        ("FS_SOLVER_POLICY", current),
        ("FREQUENSOLVE_SOLVER_POLICY", previous),
        ("FREQUENSOLVE_FREQUENSOLVER_POLICY", legacy),
    ):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    assert solver.resolve_solver_policy(explicit) == expected


@pytest.mark.parametrize(
    "current,previous,legacy",
    [(None, None, "invalid"), (None, "", "strict"), ("", "off", "strict")],
)
def test_invalid_selected_solver_policy_does_not_silently_fall_back(
    monkeypatch, current, previous, legacy
):
    for key, value in (
        ("FS_SOLVER_POLICY", current),
        ("FREQUENSOLVE_SOLVER_POLICY", previous),
        ("FREQUENSOLVE_FREQUENSOLVER_POLICY", legacy),
    ):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="compatibility policy"):
        solver.resolve_solver_policy()
