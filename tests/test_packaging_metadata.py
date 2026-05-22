from pathlib import Path

import toml

ROOT = Path(__file__).resolve().parents[1]


def load_pyproject():
    return toml.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_project_metadata_uses_pep_639_license_expression():
    project = load_pyproject()["project"]

    assert project["license"] == "MIT"
    assert "LICENSE" in project["license-files"]
    assert not any(
        classifier.startswith("License ::") for classifier in project["classifiers"]
    )


def test_build_backend_requires_setuptools_with_pep_639_support():
    build_requires = load_pyproject()["build-system"]["requires"]

    assert "setuptools>=77.0.3" in build_requires


def test_base_dependencies_exclude_optional_execution_features():
    dependencies = "\n".join(load_pyproject()["project"]["dependencies"])

    for optional_dependency in [
        "boto3",
        "dask",
        "paramiko",
        "pyasdf",
        "pyfftw",
        "pyvista",
        "segyio",
    ]:
        assert optional_dependency not in dependencies


def test_expected_extras_are_available_for_non_base_workflows():
    extras = load_pyproject()["project"]["optional-dependencies"]

    assert {
        "cloud",
        "dev",
        "docs",
        "fast-fft",
        "hpc",
        "parallel",
        "seismic-io",
        "visual",
    }.issubset(extras)
