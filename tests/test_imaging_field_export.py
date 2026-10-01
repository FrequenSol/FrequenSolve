"""Native field conversion defaults and optimizer/field separation."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.geometry.grids import CartesianGrid
from tests.imaging_fakes import FakeImagingSite
from tests.test_imaging_jobs import _assert_valid, _saved_simulation
from tests.test_imaging_workflows import _problem

pytestmark = pytest.mark.unit


def test_gradient_retains_raw_values_and_explicit_field_roles(tmp_path):
    lin = _problem(tmp_path, FakeImagingSite(seed=7)).linearize()
    raw = lin.gradient.values.copy()
    assert lin.gradient._field_factory == lin.field
    for gradient in (
        lin.gradient.copy(),
        -lin.gradient,
        2 * lin.gradient,
        lin.gradient / 2,
    ):
        assert gradient._field_factory == lin.field
    # An array-valued inverse diagonal produces a primal update, not a new covector.
    update = -lin.gradient / np.ones(lin.space.size)
    assert update._field_factory is None
    assert lin.field(update, input_role="primal").input_role == "primal"
    assert lin.field().input_role == "dual"
    np.testing.assert_array_equal(lin.gradient.values, raw)


def test_invalid_input_roles_and_blocks_are_rejected(tmp_path):
    lin = _problem(tmp_path, FakeImagingSite(seed=7)).linearize()
    with pytest.raises(ValueError, match="input_role"):
        lin.field(input_role="automatic")
    with pytest.raises(ValueError, match="mesh material"):
        lin.field()._coefficients(lin.space.blocks[0])


def test_persisted_mass_identity_units_and_material_slice(tmp_path):
    from frequensolve.imaging.property_mesh import read_lumped_mass

    path = tmp_path / "basis.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("property_space")
        group["identity"] = "basis-1"
        group["material_ranges"] = [[0, 2], [2, 3]]
        group["lumped_mass"] = [10, 20, 0.25, 0.5, 1.25]
        group["lumped_mass_units"] = "km**D"
    mass = read_lumped_mass(path, material=2, identity="basis-1")
    np.testing.assert_array_equal(mass, [0.25, 0.5, 1.25])
    assert not mass.flags.writeable
    with pytest.raises(ValueError, match="different property basis"):
        read_lumped_mass(path, material=2, identity="basis-2")


def test_cached_mapping_samples_density_locally_and_preserves_primal(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from frequensolve.imaging import statistics
    from frequensolve.imaging._field_export import NativeControlField

    path = tmp_path / "basis.h5"
    mass = np.array([0.25, 0.5, 1.25])
    with h5py.File(path, "w") as h5:
        g = h5.create_group("property_space")
        g["identity"] = "basis-1"
        g["material_ranges"] = [[0, 3]]
        g["lumped_mass"] = mass
        g["lumped_mass_units"] = "km**D"
    mapping = tmp_path / "map.h5"
    with h5py.File(mapping, "w") as h5:
        h5["offsets"] = [0, 1, 3, 5, 5]
        h5["indices"] = [0, 0, 1, 1, 2]
        h5["weights"] = [1, 0.5, 0.5, 0.25, 0.75]
        h5["valid"] = [1, 1, 1, 0]
    block = SimpleNamespace(kind="mesh", name="model.vp", size=3, prop="Vp")
    space = SimpleNamespace(
        block=lambda key: block,
        slices={block.name: slice(0, 3)},
        to_sauce_vector=lambda v: v.values,
        _mask_of=lambda b: np.ones(3, dtype=bool),
    )
    vector = SimpleNamespace(space=space, values=3 * mass)
    lin = SimpleNamespace(problem=SimpleNamespace(_shared=SimpleNamespace()))
    descriptor = dict(path=path, identity="basis-1", material=1)
    monkeypatch.setattr(statistics, "mesh_descriptor", lambda *a: descriptor)
    monkeypatch.setattr(NativeControlField, "_sampling_map", lambda *a: mapping)
    grid = CartesianGrid(n=[2, 2], x0=[0, 0], x1=[1, 1], dims=["x", "z"], units="km")
    dual = NativeControlField(lin, vector)
    for _ in range(2):
        result = dual.to_grid(grid, "vp")
        np.testing.assert_allclose(result.values.ravel()[:3], 3)
        assert np.isnan(result.values[1, 1])
    assert len(lin.problem._shared._field_export_cache) == 1
    np.testing.assert_array_equal(vector.values, 3 * mass)
    primal = NativeControlField(lin, vector, input_role="primal").to_grid(grid, "vp")
    np.testing.assert_allclose(primal.values.ravel()[:3], [0.75, 1.125, 3.1875])
    assert dual.to_property(grid, "vp").data is not None


def test_mapping_launches_locally_once_never_through_remote_site(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    from frequensolve.imaging._field_export import NativeControlField

    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("Field conversion must not submit an HPC job")

    def run(command, **kwargs):
        import json

        calls.append(command)
        request = json.loads(Path(command[command.index("-j") + 1]).read_text())
        _assert_valid(request)
        export = request["control_sensitivities"]["FieldExport"]
        assert export["input_role"] == "primal"
        assert export["sampling_map"] is True
        with h5py.File(export["output"], "w") as h5:
            h5["offsets"] = [0, 1]
        return SimpleNamespace(returncode=0)

    site = SimpleNamespace(run=forbidden, submit=forbidden, executable="remote-only")
    sim = _saved_simulation(tmp_path)
    source = im.FWIOperatorJob(
        "linearize",
        sim,
        [1],
        action="linearize",
        active=["model.vp"],
        state="state.json",
        covector="gradient.h5",
    )
    lin = SimpleNamespace(
        job=source,
        problem=SimpleNamespace(
            _shared=SimpleNamespace(), backend=SimpleNamespace(site=site, run=forbidden)
        ),
    )
    block = SimpleNamespace(name="model.vp", size=1, basis_identity="control-identity")
    grid = CartesianGrid(n=[2, 2], x0=[0, 0], x1=[1, 1], dims=["x", "z"], units="km")
    field = NativeControlField(lin, None, local_solver="local-sauce")
    monkeypatch.setattr(
        "frequensolve.imaging._field_export.shutil.which", lambda value: "/local/fs2d_s"
    )
    monkeypatch.setattr(subprocess, "run", run)
    descriptor = dict(identity="basis-1", material=1)
    path = field._sampling_map(block, descriptor, grid)
    assert field._sampling_map(block, descriptor, grid) == path
    assert len(calls) == 1
    assert calls[0][:3] == ["/local/fs2d_s", "-nthreads", "1"]
