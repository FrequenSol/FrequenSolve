"""Soft cased-borehole PML stability; opt in with FS_TEST_TRANSIENT_EXECUTABLE."""

import os

import numpy as np
import pytest

import frequensolve as fs
from frequensolve.orchestrator.sites.local import LocalSite

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(600),
    pytest.mark.skipif(
        not os.environ.get("FS_TEST_TRANSIENT_EXECUTABLE"),
        reason="requires a local Sauce fs2d executable with MUMPS",
    ),
]


def test_compact_pml_soft_axisymmetric_borehole_decays(tmp_path):
    """A thin PML at the band floor must activate its stability safeguard."""
    u = fs.ureg
    project = fs.Project(name="compact_pml", path=tmp_path / "project")
    sim = project.new_simulation("soft_cement", physics="coupled_axisym", dimension=2)
    model = fs.LayeredModel(dimension=2, x_limits=fs.Q_([0, 5], "m"))
    model.add_surface(name="surface", depth=0 * u.m)
    model.add_layer(
        name="formation",
        physics="elastic",
        properties={
            "Vp": 3000 * u.m / u.s,
            "Vs": 1000 * u.m / u.s,
            "Rho": 2400 * u.kg / u.m**3,
            "Qp": 60.0,
            "Qs": 60.0,
        },
    )
    model.add_surface(name="bottom", depth=20 * u.m)
    well = model.add_borehole(
        name="well",
        x=0 * u.m,
        top="surface",
        bottom="bottom",
        grading=fs.BoreholeGrading(wall_size=0.05 * u.m, growth=1.5, axial_factor=1.5),
    )
    for name, width, physics, vp, vs, rho, quality in (
        ("fluid", 3 * u.inch, "acoustic", 1500, None, 1000, None),
        ("casing", 1 * u.cm, "elastic", 2600, 1000, 1400, None),
        ("cement", 4 * u.inch, "elastic", 3000, 100, 2000, 300),
    ):
        properties = {"Vp": vp * u.m / u.s, "Rho": rho * u.kg / u.m**3}
        if vs is not None:
            properties["Vs"] = vs * u.m / u.s
        if quality is not None:
            properties.update(Qp=quality, Qs=quality)
        well.add_layer(name, width=width, physics=physics, properties=properties)
    model.attenuation = fs.AttenuationConfig(
        model="gsls",
        reference_frequency=10 * u.Hz,
        frequency_min=10 * u.Hz,
        frequency_max=200 * u.Hz,
    )
    sim += model
    sim += model.hex_mesh_generator([4, 1])
    sim.mesh.set_adapt(
        order=6, elems_per_wave=1.5, f_adapt=200, adapt_sources=2, guided_anisotropy=True
    )
    sim += fs.BoundaryCondition(boundaries=["x_min"], conditions=["axis"])
    sim += fs.BoundaryCondition(boundaries=["z_min"], conditions=["free"])
    # No explicit multiaxial setting: this must exercise the automatic policy,
    # including the actual width after the domain-size cap clips the layer.
    sim += fs.BoundaryCondition(
        boundaries=["x_max", "z_max"],
        conditions=["pml:impedance"],
        pml_wavelengths=0.05,
    )
    sim += fs.SolverConfig(
        solver="direct", grids=1, coarse_solver="mumps", precision="double"
    )
    source = fs.SourceGeometry.inline(
        kind="vector",
        sources=[
            fs.PointSource(
                "surface", coords=fs.Q_([2.5, 0], "m"), direction=[0, 1], amplitude=1 * u.MN
            )
        ],
    )
    survey = fs.Acquisition(sources=source)
    sensor = fs.ReceiverNode(name="hydrophone")
    sensor.add_component("pressure", "pressure", units="kPa")
    survey.add_receiver_group(
        "pressure",
        device=sensor,
        coords=fs.Q_([[0.0381, z] for z in (2, 10, 18)], "m"),
    )
    sim += survey
    job = fs.TransientJob(
        "pulse",
        sim,
        time=fs.TimeIntegration(-0.0125, 0.3125, 0.000125, method=fs.Radau(), f_max=200),
        source=fs.TimeFunction.ricker(80, delay=0.0125),
        trace_schedule=fs.TimeOutput(stride=8),
    )
    job += fs.TraceOutput()
    job += fs.OutputUnits(geometry="m", fields={"pressure": "kPa"})
    site = LocalSite(
        solver=os.environ["FS_TEST_TRANSIENT_EXECUTABLE"],
        solver_policy="off",
        n_workers=1,
        threads_per_worker=int(os.environ.get("FS_TEST_THREADS", "2")),
        local_directory=tmp_path / "workers",
        environment={"OMPI_MCA_btl": "self", "VECLIB_MAXIMUM_THREADS": "1"},
    )
    assert site.run(job, check=True).successful
    pressure = job.traces.open().read("pressure")
    assert np.isfinite(pressure.values).all()
    early = pressure.sel(time=slice(0.0125, 0.1125)).values.astype(float)
    late = pressure.sel(time=slice(0.2125, 0.3125)).values.astype(float)
    assert np.linalg.norm(early) > 1e-8
    # The unstabilized reproducer grows by many orders of magnitude here.
    assert np.linalg.norm(late) < 0.01 * np.linalg.norm(early)
