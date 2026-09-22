import numpy as np

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.model.parameterization import BSplineControl, HatControl
from frequensolve.model.representation import (
    CartesianGridRepresentation,
    ControlRepresentation,
    EvaluationContext,
)


def test_hat_representation_has_local_partition_and_exact_pullback():
    control = HatControl(
        axis="below",
        coordinate_system="top_relative",
        origin=0.0,
        spacing=1.0,
        coefficients=np.zeros(4),
    )
    representation = ControlRepresentation(control)
    context = EvaluationContext(
        {"below": [-1.0, 0.0, 0.25, 1.5, 3.0, 4.0]},
        coordinate_system="top_relative",
    )
    operator = representation.sampling_operator(context)

    np.testing.assert_allclose(operator.toarray().sum(axis=1), [0, 1, 1, 1, 1, 0])
    assert np.max(np.diff(operator.indptr)) <= 2

    direction = np.array([0.2, -0.4, 0.5, 0.1])
    dual = np.array([0.7, -0.2, 0.3, 0.9, -0.1, 0.6])
    lhs = np.dot(representation.evaluate(direction, context, reshape=False), dual)
    rhs = np.dot(direction, representation.pullback(dual, context))
    np.testing.assert_allclose(lhs, rhs, rtol=1.0e-14, atol=1.0e-14)


def test_bspline_representation_reproduces_linear_polynomial():
    control = BSplineControl(
        axis="z",
        degree=2,
        knots=[0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
        coefficients=np.zeros(4),
    )
    representation = ControlRepresentation(control)
    context = EvaluationContext({"z": np.linspace(0.0, 2.0, 21)})
    coefficients = 1.0 + 2.0 * control.coordinates

    values = representation.evaluate(coefficients, context)

    np.testing.assert_allclose(values, 1.0 + 2.0 * np.linspace(0.0, 2.0, 21))


def test_cartesian_grid_representation_interpolates_and_pullback_is_exact():
    grid = CartesianGrid(
        n=[3, 2],
        x0=[0.0, 0.0],
        x1=[2.0, 1.0],
        dims=["x", "z"],
    )
    representation = CartesianGridRepresentation(grid)
    context = EvaluationContext({"x": [0.5, 1.5], "z": [0.25, 0.75]})
    nodes = representation.node_context
    coefficients = 2.0 * nodes.coordinate("x") - nodes.coordinate("z")
    dual = np.array([0.3, -0.7])

    values = representation.evaluate(coefficients, context, reshape=False)
    lhs = np.dot(values, dual)
    rhs = np.dot(coefficients, representation.pullback(dual, context))

    np.testing.assert_allclose(values, [0.75, 2.25])
    np.testing.assert_allclose(lhs, rhs, rtol=1.0e-14, atol=1.0e-14)


def test_cross_representation_transfer_projects_to_cartesian_nodes():
    control = ControlRepresentation(
        HatControl(axis="z", origin=0.0, spacing=0.5, coefficients=np.zeros(5))
    )
    grid = CartesianGrid(n=[9], x0=[0.0], x1=[2.0], dims=["z"])
    image = CartesianGridRepresentation(grid)
    coefficients = np.array([0.0, 1.0, 0.0, -1.0, 0.0])

    transferred = control.transfer_to(
        image, coefficients, image.node_context, damping=1.0e-14
    )

    np.testing.assert_allclose(
        image.evaluate(transferred, image.node_context),
        control.evaluate(coefficients, image.node_context),
        atol=1.0e-12,
    )
