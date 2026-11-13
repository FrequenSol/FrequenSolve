"""Quadrature selections survive SDK serialization without changing defaults."""

import pytest

from frequensolve import Discretization, SeismicSimulation


@pytest.mark.parametrize("rule", ["gauss", "ainsworth"])
def test_quadrature_simulation_roundtrip(rule, tmp_path):
    simulation = SeismicSimulation(
        name="quadrature", physics="acoustic", dimension=2, project_path=tmp_path
    )
    simulation.discretization = Discretization(method="Galerkin", quadrature=rule)
    payload = simulation.to_fs()["Discretization"]
    assert payload == {"method": "Galerkin", "quadrature": rule}
    assert Discretization.from_fs(payload).to_fs() == payload
    assert Discretization().to_fs() == {}
