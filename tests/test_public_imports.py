import importlib
import importlib.util
import subprocess
import sys


def test_top_level_import_has_no_optional_backend_side_effects():
    code = """
import sys
import frequensolve
forbidden = {'boto3', 'paramiko', 'dask', 'distributed', 'matplotlib'}
loaded = sorted(name for name in forbidden if name in sys.modules)
if loaded:
    raise SystemExit('unexpected optional imports: ' + ', '.join(loaded))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_public_package_imports_smoke():
    for name in [
        "frequensolve",
        "frequensolve.geometry",
        "frequensolve.mesh",
        "frequensolve.model",
        "frequensolve.project",
        "frequensolve.seismic",
        "frequensolve.simulation",
        "frequensolve.util",
        "frequensolve.orchestrator",
        "frequensolve.orchestrator.sites",
    ]:
        importlib.import_module(name)


def test_removed_legacy_public_names_are_not_exported():
    trace_record = importlib.import_module("frequensolve.seismic.trace_record")
    traces = importlib.import_module("frequensolve.seismic.traces")
    seismic = importlib.import_module("frequensolve.seismic")

    assert not hasattr(trace_record, "ShotRecord")
    assert "TraceRecord" not in getattr(trace_record, "__all__", [])
    assert not hasattr(seismic, "TraceRecord")
    assert not hasattr(seismic, "ShotRecord")
    assert importlib.util.find_spec("frequensolve.seismic.record_database") is None
    assert importlib.util.find_spec("frequensolve.orchestrator.file_manager") is None
    assert importlib.util.find_spec("frequensolve.project.workflows") is None
    assert not hasattr(traces.TraceDataset, "record_db")
    assert not hasattr(traces.TraceDataset, "read_FD")
    assert not hasattr(traces.TraceDataset, "read_TD")
