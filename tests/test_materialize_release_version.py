from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.materialize_release_version import materialize_release_version

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40


def versioneer_identity(source: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, versioneer; print(json.dumps(versioneer.get_versions()))",
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    ("version", "archived_refnames", "archive_version"),
    [
        (
            "0.3.0rc2",
            "(HEAD, tag: v0.3.0rc1, tag: v0.3.0rc2)",
            "0.3.0rc1",
        ),
        (
            "0.3.0",
            "(HEAD, tag: v0.3.0rc1, tag: v0.3.0)",
            "0.3.0",
        ),
    ],
)
def test_materializes_requested_version_in_multi_tag_git_free_archive(
    tmp_path: Path,
    version: str,
    archived_refnames: str,
    archive_version: str,
) -> None:
    source = tmp_path / f"frequensolve-{version}"
    version_file = source / "src/frequensolve/_version.py"
    version_file.parent.mkdir(parents=True)
    shutil.copy(ROOT / "versioneer.py", source / "versioneer.py")
    shutil.copy(ROOT / "setup.cfg", source / "setup.cfg")

    archived_version = (ROOT / "src/frequensolve/_version.py").read_text(
        encoding="utf-8"
    )
    archived_version = archived_version.replace(
        "$Format:%d$", archived_refnames
    ).replace("$Format:%H$", REVISION)
    version_file.write_text(archived_version, encoding="utf-8")

    assert versioneer_identity(source)["version"] == archive_version

    materialize_release_version(
        version=version,
        release_tag=f"v{version}",
        revision=REVISION,
        output=version_file,
    )

    assert versioneer_identity(source) == {
        "version": version,
        "full-revisionid": REVISION,
        "dirty": False,
        "error": None,
        "date": None,
    }


def test_rejects_release_tag_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "_version.py"

    with pytest.raises(ValueError, match="release tag must be v0.3.0rc2"):
        materialize_release_version(
            version="0.3.0rc2",
            release_tag="v0.3.0rc1",
            revision=REVISION,
            output=output,
        )

    assert not output.exists()
