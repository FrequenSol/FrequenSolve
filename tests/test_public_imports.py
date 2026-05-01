import importlib
import importlib.util
import subprocess
import sys

from frequensolve._optional import missing_optional_class, optional_class


def test_top_level_import_smoke_in_clean_process():
    code = """
import frequensolve
for name in ['Project', 'LayeredModel', 'LocalSite', 'Stampede3Site', 'AWSSite']:
    getattr(frequensolve, name)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_top_level_import_does_not_load_optional_backends():
    code = """
import sys
import frequensolve
forbidden = {
    'boto3',
    'dask',
    'distributed',
    'dotenv',
    'matplotlib',
    'paramiko',
    'pint',
    'pylops',
    'pyfftw',
    'pyvista',
    'segyio',
}
loaded = sorted(name for name in forbidden if name in sys.modules)
if loaded:
    raise SystemExit('unexpected optional imports: ' + ', '.join(loaded))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_top_level_authoring_exports_are_available():
    import frequensolve as fs

    expected = [
        "Project",
        "SeismicSimulation",
        "LayeredModel",
        "ModelSubdomain",
        "Property",
        "prop",
        "CartesianGrid",
        "CoordinateSystem",
        "MeshManager",
        "HexMeshGenerator",
        "Acquisition",
        "PointSource",
        "ReceiverNode",
        "SparseSurvey",
        "RickerWavelet",
        "OrmsbyWavelet",
        "KlauderWavelet",
        "JobOutputs",
        "ParaviewOutput",
        "FrequencyDomainJob",
        "TimeDomainJob",
        "TraceDataset",
        "plot_gather",
        "read_vtu",
        "vtu_fields",
        "plot_vtu",
        "configure_fft",
    ]

    for name in expected:
        assert getattr(fs, name) is not None, name
    assert fs.ureg is not None


def test_optional_backend_exports_are_part_of_public_sdk_surface():
    import frequensolve as fs

    assert "LocalSite" in dir(fs)
    assert "Stampede3Site" in dir(fs)
    assert "AWSSite" in dir(fs)
    assert "LocalSite" in fs.__all__
    assert "Stampede3Site" in fs.__all__
    assert "AWSSite" in fs.__all__


def test_optional_dependency_placeholder_raises_install_hint():
    Missing = missing_optional_class(
        "MissingBackend",
        extra="parallel",
        error=ModuleNotFoundError("No module named 'distributed'"),
        module=__name__,
    )

    try:
        Missing()
    except ImportError as exc:
        message = str(exc)
    else:
        raise AssertionError("optional placeholder did not raise")

    assert "pip install frequensolve[parallel]" in message


def test_lazy_optional_class_raises_install_hint():
    Missing = optional_class(
        "MissingBackend",
        "frequensolve.missing_backend.MissingBackend",
        extra="parallel",
        dependencies=("missing-backend",),
        module=__name__,
    )

    try:
        Missing()
    except ImportError as exc:
        message = str(exc)
    else:
        raise AssertionError("lazy optional class did not raise")

    assert "pip install frequensolve[parallel]" in message
    assert "missing-backend" in message


def test_units_registry_is_lazy_but_usable():
    code = """
import sys
import frequensolve as fs
if 'pint' in sys.modules:
    raise SystemExit('pint imported before unit registry use')
if str(1 * fs.ureg.meter) != '1 meter':
    raise SystemExit('lazy unit registry produced an unexpected quantity')
if 'pint' not in sys.modules:
    raise SystemExit('pint was not imported when unit registry was used')
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_public_package_imports_smoke():
    for name in [
        "frequensolve",
        "frequensolve.geometry",
        "frequensolve.mesh",
        "frequensolve.model",
        "frequensolve.model.layered",
        "frequensolve.plotting",
        "frequensolve.plotting.analysis",
        "frequensolve.plotting.traces",
        "frequensolve.plotting.vtu",
        "frequensolve.project",
        "frequensolve.seismic",
        "frequensolve.simulation",
        "frequensolve.simulation.outputs",
        "frequensolve.util",
        "frequensolve.orchestrator",
        "frequensolve.orchestrator.sites",
    ]:
        importlib.import_module(name)


def test_removed_legacy_public_names_are_not_exported():
    trace_record = importlib.import_module("frequensolve.seismic.trace_record")
    traces = importlib.import_module("frequensolve.seismic.traces")
    fs = importlib.import_module("frequensolve")
    seismic = importlib.import_module("frequensolve.seismic")

    assert not hasattr(trace_record, "ShotRecord")
    assert "TraceRecord" not in getattr(trace_record, "__all__", [])
    assert not hasattr(seismic, "TraceRecord")
    assert not hasattr(seismic, "ShotRecord")
    assert not hasattr(seismic, "read_vtu_wavefield")
    assert not hasattr(seismic, "plot_vtu_wavefield")
    assert not hasattr(seismic, "LayeredModel")
    assert not hasattr(fs, "OutputManager")
    assert importlib.util.find_spec("frequensolve.simulation.output_manager") is None
    assert importlib.util.find_spec("frequensolve.seismic.layered_model") is None
    assert importlib.util.find_spec("frequensolve.seismic.layered_plotting") is None
    assert importlib.util.find_spec("frequensolve.seismic.plotting") is None
    assert importlib.util.find_spec("frequensolve.seismic.vtu") is None
    assert not hasattr(fs, "read_vtu_wavefield")
    assert not hasattr(fs, "plot_vtu_wavefield")
    assert importlib.util.find_spec("frequensolve.seismic.record_database") is None
    assert importlib.util.find_spec("frequensolve.orchestrator.file_manager") is None
    assert importlib.util.find_spec("frequensolve.project.workflows") is None
    assert importlib.util.find_spec("frequensolve.util.data_file") is None
    assert not hasattr(traces.TraceDataset, "record_db")
    assert not hasattr(traces.TraceDataset, "read_FD")
    assert not hasattr(traces.TraceDataset, "read_TD")
