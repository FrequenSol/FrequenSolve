"""Direct acoustic pressure forcing preserves source kind and physical units."""

import json

import pytest

import frequensolve as fs


def test_pressure_monopole_roundtrip():
    source = fs.SourceGeometry.pressure_monopole(
        fs.Q_([[0.0, 2.0]], "m"), amplitude=fs.Q_(2.0, "m^3/s^2")
    )
    data = source.to_fs()
    restored = fs.SourceGeometry.from_fs(data)
    assert restored.kind == "pressure_monopole"
    assert restored.to_fs() == data
    assert data["defaults"]["amplitude"]["value"] == 2.0
    assert fs.Q_(1.0, data["defaults"]["amplitude"]["units"]).check(
        "[length] ** 3 / [time] ** 2"
    )


@pytest.mark.parametrize("physics", ["acoustic", "coupled"])
def test_pressure_monopole_transient_export(tmp_path, physics):
    project = fs.Project(name="p", path=tmp_path / "p")
    sim = project.new_simulation(
        name="wave", physics=physics, dimension=2, axisymmetric=True
    )
    sim.discretization = fs.Discretization(method="Galerkin")
    sim.model.attenuation_model = "none"
    sim.mesh.set_adapt(elems_per_wave=2, order=4, f_adapt=200)
    acq = fs.Acquisition()
    acq.add_sources(kind="pressure_monopole", coords=fs.Q_([[0.0, 2.0]], "m"))
    sim += acq
    job = fs.TransientJob(
        "pulse",
        sim,
        time=fs.TimeIntegration(-0.02, 0.1, 0.0001),
        source=fs.TimeFunction.ricker(50, delay=0.02),
    )
    job.save()
    serialized = (
        json.loads(sim._simulation_file.read_text())
        if hasattr(sim, "_simulation_file")
        else sim.to_fs()
    )
    assert serialized["Acquisition"]["source_geometry"]["kind"] == "pressure_monopole"


@pytest.mark.parametrize(
    "kind", ["scalar", "monopole", "volume_injection", "isotropic_moment"]
)
def test_legacy_source_kinds_preserved(kind):
    source = fs.SourceGeometry.points(coords=[[0.0, 2.0]], kind=kind)
    assert fs.SourceGeometry.from_fs(source.to_fs()).kind == kind


@pytest.mark.parametrize("kind", ["scalar", "pressure_monopole"])
def test_direct_forcing_accepts_quiet_step(tmp_path, kind):
    project = fs.Project(name="p", path=tmp_path / "p")
    sim = project.new_simulation(name="wave", physics="acoustic", dimension=2)
    sim.discretization = fs.Discretization(method="Galerkin")
    sim.model.attenuation_model = "none"
    sim.mesh.set_adapt(elems_per_wave=2, f_adapt=200)
    sim.acquisition.add_sources(kind=kind, coords=[[0.0, 2.0]])
    job = fs.TransientJob(
        "pulse",
        sim,
        time=fs.TimeIntegration(0.0, 0.1, 0.001),
        source=fs.TimeFunction.step(delay=0.02),
    )
    job.validate_outputs()
    sim.acquisition.source_geometry.kind = "isotropic_moment"
    with pytest.raises(ValueError, match="smooth time functions"):
        job.validate_outputs()
