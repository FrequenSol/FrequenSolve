from __future__ import annotations

import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from frequensolve.util import atomic as atomic_module
from frequensolve.util.atomic import (
    atomic_output_path,
    atomic_write_json,
    atomic_write_text,
)


def _temporary_outputs(destination):
    return list(
        destination.parent.glob(f".{destination.name}.*.tmp{destination.suffix}")
    )


def test_atomic_write_orders_file_sync_replace_and_directory_sync(
    monkeypatch, tmp_path
):
    destination = tmp_path / "state.json"
    events = []
    real_replace = atomic_module.os.replace

    monkeypatch.setattr(
        atomic_module,
        "_fsync_file",
        lambda path: events.append(("file", path)),
    )

    def replace(source, target):
        events.append(("replace", target))
        real_replace(source, target)

    monkeypatch.setattr(atomic_module.os, "replace", replace)
    monkeypatch.setattr(
        atomic_module,
        "_fsync_directory",
        lambda path: events.append(("directory", path)),
    )

    atomic_write_json(destination, {"generation": 1})

    assert [event[0] for event in events] == ["file", "replace", "directory"]
    assert json.loads(destination.read_text()) == {"generation": 1}


def test_atomic_output_preserves_existing_destination_mode(tmp_path):
    destination = tmp_path / "state.json"
    destination.write_text('{"generation": 1}')
    destination.chmod(0o640)

    atomic_write_json(destination, {"generation": 2})

    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


def test_new_atomic_output_uses_shared_metadata_mode_subject_to_umask(tmp_path):
    reference = tmp_path / "reference"
    descriptor = os.open(reference, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    os.close(descriptor)
    destination = tmp_path / "state.json"

    atomic_write_json(destination, {"generation": 1})

    assert stat.S_IMODE(destination.stat().st_mode) == stat.S_IMODE(
        reference.stat().st_mode
    )


def test_atomic_output_preserves_previous_file_when_writer_fails(tmp_path):
    destination = tmp_path / "state.json"
    destination.write_text('{"generation": 1}')

    with pytest.raises(RuntimeError, match="injected writer failure"):
        with atomic_output_path(destination) as temporary:
            temporary.write_text('{"generation": 2}')
            raise RuntimeError("injected writer failure")

    assert json.loads(destination.read_text()) == {"generation": 1}
    assert _temporary_outputs(destination) == []


def test_atomic_output_cleans_temporary_when_writer_fails_before_write(tmp_path):
    destination = tmp_path / "state.json"
    destination.write_text('{"generation": 1}')

    with pytest.raises(RuntimeError, match="injected early writer failure"):
        with atomic_output_path(destination):
            raise RuntimeError("injected early writer failure")

    assert json.loads(destination.read_text()) == {"generation": 1}
    assert _temporary_outputs(destination) == []


def test_atomic_output_preserves_previous_file_when_replace_fails(
    monkeypatch, tmp_path
):
    destination = tmp_path / "state.json"
    destination.write_text('{"generation": 1}')

    def fail_replace(_source, _target):
        raise OSError("injected replace failure")

    monkeypatch.setattr(atomic_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        atomic_write_text(destination, '{"generation": 2}')

    assert json.loads(destination.read_text()) == {"generation": 1}
    assert _temporary_outputs(destination) == []


def test_atomic_output_preserves_previous_file_when_file_sync_fails(
    monkeypatch, tmp_path
):
    destination = tmp_path / "state.json"
    destination.write_text('{"generation": 1}')

    def fail_file_sync(_path):
        raise OSError("injected file sync failure")

    monkeypatch.setattr(atomic_module, "_fsync_file", fail_file_sync)

    with pytest.raises(OSError, match="injected file sync failure"):
        atomic_write_text(destination, '{"generation": 2}')

    assert json.loads(destination.read_text()) == {"generation": 1}
    assert _temporary_outputs(destination) == []


def test_atomic_output_is_complete_if_directory_sync_fails_after_replace(
    monkeypatch, tmp_path
):
    destination = tmp_path / "state.json"
    destination.write_text('{"generation": 1}')

    def fail_directory_sync(_path):
        raise OSError("injected directory sync failure")

    monkeypatch.setattr(atomic_module, "_fsync_directory", fail_directory_sync)

    with pytest.raises(OSError, match="injected directory sync failure"):
        atomic_write_text(destination, '{"generation": 2}')

    assert json.loads(destination.read_text()) == {"generation": 2}
    assert _temporary_outputs(destination) == []


def test_concurrent_writers_use_unique_temporaries_and_publish_complete_json(
    tmp_path,
):
    destination = tmp_path / "state.json"
    writer_count = 8
    barrier = threading.Barrier(writer_count)
    temporary_paths = []
    paths_lock = threading.Lock()

    def publish(generation):
        payload = {"generation": generation, "values": [generation] * 4096}
        with atomic_output_path(destination) as temporary:
            temporary.write_text(json.dumps(payload))
            with paths_lock:
                temporary_paths.append(temporary)
            barrier.wait()

    with ThreadPoolExecutor(max_workers=writer_count) as executor:
        list(executor.map(publish, range(writer_count)))

    assert len(set(temporary_paths)) == writer_count
    published = json.loads(destination.read_text())
    assert published["generation"] in range(writer_count)
    assert published["values"] == [published["generation"]] * 4096
    assert _temporary_outputs(destination) == []
