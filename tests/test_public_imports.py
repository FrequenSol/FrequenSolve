import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import toml

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
        "BoreholeAnnularPadding",
        "Fracture",
        "ModelSubdomain",
        "Property",
        "coord",
        "prop",
        "ref",
        "remap",
        "var",
        "CartesianGrid",
        "CoordinateSystem",
        "MeshManager",
        "HexMeshGenerator",
        "Acquisition",
        "PointSource",
        "SourceGeometry",
        "SourceEncoding",
        "DistributedSource",
        "ReceiverNode",
        "SparseSurvey",
        "RickerWavelet",
        "OrmsbyWavelet",
        "KlauderWavelet",
        "JobOutputs",
        "OutputUnits",
        "AxisAlignedPlane",
        "ParaViewOutput",
        "ParaviewOutput",
        "outputs",
        "paraview",
        "wavefield",
        "field",
        "info",
        "output_property",
        "BaseJob",
        "FrequencyDomainJob",
        "TimeDomainJob",
        "TraceDataset",
        "plot_gather",
        "read_vtu",
        "vtu_fields",
        "plot_vtu",
        "configure_fft",
        "load",
    ]

    for name in expected:
        assert getattr(fs, name) is not None, name
    assert fs.ureg is not None


def test_top_level_colormap_exports_are_lazy_and_discoverable():
    code = """
import sys
import frequensolve as fs

if 'BuGrOr' not in fs.__all__:
    raise SystemExit('BuGrOr missing from __all__')
if 'BuGrOr' not in dir(fs):
    raise SystemExit('BuGrOr missing from dir(fs)')
if 'matplotlib' in sys.modules:
    raise SystemExit('matplotlib loaded before colormap access')

cmap = fs.BuGrOr
if getattr(cmap, 'N', 0) <= 0:
    raise SystemExit('invalid colormap')
if 'matplotlib' not in sys.modules:
    raise SystemExit('matplotlib was not loaded when colormap was resolved')
"""
    subprocess.run([sys.executable, "-c", code], check=True)


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
        extra="hpc",
        error=ModuleNotFoundError("No module named 'distributed'"),
        module=__name__,
    )

    try:
        Missing()
    except ImportError as exc:
        message = str(exc)
    else:
        raise AssertionError("optional placeholder did not raise")

    assert "pip install frequensolve[hpc]" in message


def test_lazy_optional_class_raises_install_hint():
    Missing = optional_class(
        "MissingBackend",
        "frequensolve.missing_backend.MissingBackend",
        extra="hpc",
        dependencies=("missing-backend",),
        module=__name__,
    )

    try:
        Missing()
    except ImportError as exc:
        message = str(exc)
    else:
        raise AssertionError("lazy optional class did not raise")

    assert "pip install frequensolve[hpc]" in message
    assert "missing-backend" in message


def test_parallel_extra_remains_hpc_alias():
    project_root = Path(__file__).resolve().parents[1]
    pyproject = toml.load(project_root / "pyproject.toml")
    extras = pyproject["project"]["optional-dependencies"]

    assert set(extras["parallel"]) == set(extras["hpc"])


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
        "frequensolve.expr",
        "frequensolve.model.layered",
        "frequensolve.plotting",
        "frequensolve.plotting.analysis",
        "frequensolve.plotting.traces",
        "frequensolve.plotting.vtu",
        "frequensolve.project",
        "frequensolve.seismic",
        "frequensolve.simulation",
        "frequensolve.simulation.jobs",
        "frequensolve.simulation.jobs.base",
        "frequensolve.simulation.jobs.fwi",
        "frequensolve.simulation.jobs.imaging",
        "frequensolve.simulation.jobs.forward",
        "frequensolve.simulation.outputs",
        "frequensolve.util",
        "frequensolve.orchestrator",
        "frequensolve.orchestrator.sites",
        "frequensolve.orchestrator.utils",
        "frequensolve.orchestrator.utils.pool",
        "frequensolve.orchestrator.utils.progress",
    ]:
        importlib.import_module(name)


def test_removed_legacy_public_names_are_not_exported():
    trace_record = importlib.import_module("frequensolve.seismic.trace_record")
    traces = importlib.import_module("frequensolve.seismic.traces")
    fs = importlib.import_module("frequensolve")
    project = importlib.import_module("frequensolve.project")
    seismic = importlib.import_module("frequensolve.seismic")
    jobs = importlib.import_module("frequensolve.simulation.jobs")

    assert not hasattr(jobs, "SimulationJob")
    assert not hasattr(jobs, "JobRunRecord")
    assert not hasattr(trace_record, "ShotRecord")
    assert "TraceRecord" not in getattr(trace_record, "__all__", [])
    assert not hasattr(seismic, "TraceRecord")
    assert not hasattr(seismic, "ShotRecord")
    assert not hasattr(seismic, "read_vtu_wavefield")
    assert not hasattr(seismic, "plot_vtu_wavefield")
    assert not hasattr(seismic, "LayeredModel")
    assert not hasattr(fs, "OutputManager")
    assert not hasattr(fs, "SimulationJob")
    assert not hasattr(fs, "JobRunRecord")
    assert importlib.util.find_spec("frequensolve.simulation.output_manager") is None
    assert importlib.util.find_spec("frequensolve.simulation.artifacts") is None
    assert importlib.util.find_spec("frequensolve.simulation.fwi") is None
    assert importlib.util.find_spec("frequensolve.simulation.imaging") is None
    assert importlib.util.find_spec("frequensolve.simulation.jobs.jobs") is None
    assert (
        importlib.util.find_spec("frequensolve.simulation.jobs.artifact_access") is None
    )
    assert importlib.util.find_spec("frequensolve.seismic.layered_model") is None
    assert importlib.util.find_spec("frequensolve.seismic.layered_plotting") is None
    assert importlib.util.find_spec("frequensolve.seismic.plotting") is None
    assert importlib.util.find_spec("frequensolve.seismic.signals") is None
    assert importlib.util.find_spec("frequensolve.seismic.vtu") is None
    assert not hasattr(seismic, "Signal")
    assert not hasattr(seismic, "AnalyticalSignal")
    assert not hasattr(seismic, "SignalFromFile")
    assert not hasattr(fs, "Signal")
    assert not hasattr(fs, "AnalyticalSignal")
    assert not hasattr(fs, "SignalFromFile")
    assert not hasattr(fs, "read_vtu_wavefield")
    assert not hasattr(fs, "plot_vtu_wavefield")
    assert importlib.util.find_spec("frequensolve.seismic.record_database") is None
    assert importlib.util.find_spec("frequensolve.orchestrator.credentials") is None
    assert importlib.util.find_spec("frequensolve.orchestrator.file_manager") is None
    assert importlib.util.find_spec("frequensolve.orchestrator.pool") is None
    assert importlib.util.find_spec("frequensolve.orchestrator.progress") is None
    assert importlib.util.find_spec("frequensolve.orchestrator.ssh") is None
    assert importlib.util.find_spec("frequensolve.project.migrate_version") is None
    assert importlib.util.find_spec("frequensolve.project.workflows") is None
    assert importlib.util.find_spec("frequensolve.util.data_file") is None
    assert importlib.util.find_spec("frequensolve.util.input_parser") is None
    assert importlib.util.find_spec("frequensolve.util.memoization") is None
    assert importlib.util.find_spec("frequensolve.util.paraview_wrapper") is None
    assert importlib.util.find_spec("frequensolve.util.registry") is None
    assert importlib.util.find_spec("frequensolve.util.report_builder") is None
    assert importlib.util.find_spec("frequensolve.util.serialize") is None
    assert not hasattr(fs, "Report")
    assert not hasattr(fs, "Figure")
    assert not hasattr(fs, "Section")
    assert not hasattr(fs, "Version")
    assert not hasattr(project, "Version")
    assert not hasattr(traces.TraceDataset, "record_db")
    assert not hasattr(traces.TraceDataset, "read_FD")
    assert not hasattr(traces.TraceDataset, "read_TD")
