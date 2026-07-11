import runpy
from pathlib import Path

from packaging.version import Version

import frequensolve

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sphinx_release_matches_the_public_package_version() -> None:
    config = runpy.run_path(str(REPO_ROOT / "docs/source/conf.py"))
    package_version = Version(frequensolve.__version__)

    assert config["release"] == frequensolve.__version__
    assert config["version"] == f"{package_version.major}.{package_version.minor}"
