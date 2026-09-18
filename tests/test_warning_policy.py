import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "category", ["RuntimeWarning", "ResourceWarning", "DeprecationWarning"]
)
def test_new_project_warning_fails_the_configured_pytest_lane(tmp_path, category):
    test_file = tmp_path / "test_unexpected_warning.py"
    test_file.write_text(
        "import warnings\n"
        "def test_behavior():\n"
        f"    warnings.warn_explicit('synthetic new warning', {category}, "
        "filename='synthetic.py', lineno=1, module='frequensolve.synthetic')\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(Path(__file__).parents[1] / "pyproject.toml"),
            "-o",
            "addopts=",
            "-q",
            str(test_file),
        ],
        env={**os.environ, "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert category in result.stdout


def test_third_party_warning_also_requires_an_explicit_narrow_exception(tmp_path):
    test_file = tmp_path / "test_vendor_warning.py"
    test_file.write_text(
        "import warnings\n"
        "def test_behavior():\n"
        "    warnings.warn_explicit('synthetic vendor warning', RuntimeWarning, "
        "filename='vendor.py', lineno=1, module='synthetic_vendor')\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(Path(__file__).parents[1] / "pyproject.toml"),
            "-o",
            "addopts=",
            "-q",
            str(test_file),
        ],
        env={**os.environ, "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "synthetic vendor warning" in result.stdout


def test_real_sdk_warning_attributed_to_its_caller_fails(tmp_path):
    test_file = tmp_path / "test_caller_warning.py"
    test_file.write_text(
        "from frequensolve.orchestrator.utils import credentials\n"
        "def test_behavior(monkeypatch):\n"
        "    def unavailable_store():\n"
        "        raise credentials.CredentialStoreError('synthetic unavailable store')\n"
        "    monkeypatch.setattr(credentials, 'KeyringCredentialStore', unavailable_store)\n"
        "    credentials.Credentials(username='synthetic-user')\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(Path(__file__).parents[1] / "pyproject.toml"),
            "-o",
            "addopts=",
            "-q",
            str(test_file),
        ],
        env={**os.environ, "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "RuntimeWarning: synthetic unavailable store" in result.stdout
