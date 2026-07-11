import runpy
from pathlib import Path

import pytest

import frequensolve

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sphinx_version_metadata_comes_from_the_public_package_version() -> None:
    config = runpy.run_path(str(REPO_ROOT / "docs/source/conf.py"))

    assert config["release"] == frequensolve.__version__
    assert config["version"] == config["_short_version"](config["release"])


@pytest.mark.parametrize(
    ("release", "expected"),
    [
        ("2.4.1", "2.4"),
        ("2.4.1.dev5+g1234567", "2.4"),
        ("0+untagged.1.g1234567", "0+untagged.1.g1234567"),
    ],
)
def test_sphinx_short_version_handles_release_and_untagged_builds(
    release: str, expected: str
) -> None:
    config = runpy.run_path(str(REPO_ROOT / "docs/source/conf.py"))

    assert config["_short_version"](release) == expected
