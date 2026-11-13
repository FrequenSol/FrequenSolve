import numpy as np
import pytest

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.model.parameterization import (
    BSplineControl,
    HatControl,
    TensorBSplineControl,
    TensorHatControl,
)
from frequensolve.model.representation import (
    CartesianGridRepresentation,
    ControlRepresentation,
    EvaluationContext,
    FieldRepresentation,
    TensorControlRepresentation,
)


def test_sampling_operator_reflection_preserves_sparse_matrix_type():
    from typing import get_type_hints

    from scipy.sparse import csr_matrix

    for representation in (
        FieldRepresentation,
        ControlRepresentation,
        CartesianGridRepresentation,
    ):
        hints = get_type_hints(representation.sampling_operator)
        assert hints["return"] is csr_matrix


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


def test_held_hat_representation_clamps_coordinates_into_the_span():
    control = HatControl(
        axis="below",
        origin=0.0,
        spacing=1.0,
        coefficients=np.zeros(4),
        extrapolate="hold",
    )
    representation = ControlRepresentation(control)
    context = EvaluationContext({"below": [-7.0, 0.0, 0.25, 1.5, 3.0, 40.0]})
    operator = representation.sampling_operator(context)

    # Every point is active and the rows outside the span repeat the end rows.
    np.testing.assert_allclose(operator.toarray().sum(axis=1), np.ones(6))
    dense = operator.toarray()
    np.testing.assert_allclose(dense[0], dense[1])
    np.testing.assert_allclose(dense[5], dense[4])
    coefficients = np.array([1.0, 2.0, 0.5, -1.0])
    np.testing.assert_allclose(
        representation.evaluate(coefficients, context),
        [1.0, 1.0, 1.25, 1.25, -1.0, -1.0],
    )
    # The clamped coordinate map is constant outside the span: derivative rows vanish there.
    slope = representation.sampling_operator(context, derivative_order=1).toarray()
    np.testing.assert_allclose(slope[[0, 5]], 0.0)
    np.testing.assert_allclose(slope[2] @ coefficients, 1.0)
    # JVP/VJP stay a transpose pair with clamping.
    direction = np.array([0.2, -0.4, 0.5, 0.1])
    dual = np.array([0.7, -0.2, 0.3, 0.9, -0.1, 0.6])
    lhs = np.dot(representation.evaluate(direction, context, reshape=False), dual)
    rhs = np.dot(direction, representation.pullback(dual, context))
    np.testing.assert_allclose(lhs, rhs, rtol=1.0e-14, atol=1.0e-14)
    # Projection onto a held basis recovers the held profile exactly.
    recovered = representation.project(
        representation.evaluate(coefficients, context), context
    )
    np.testing.assert_allclose(recovered, coefficients, atol=1.0e-12)


def test_held_bspline_representation_holds_the_end_values():
    control = BSplineControl(
        axis="z",
        degree=2,
        knots=[0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
        coefficients=[1.0, 2.0, 3.0, 4.0],
        extrapolate="hold",
    )
    representation = ControlRepresentation(control)
    context = EvaluationContext({"z": [-3.0, 0.0, 2.0, 5.5]})
    np.testing.assert_allclose(
        representation.evaluate(np.asarray(control.coefficients), context),
        [1.0, 1.0, 4.0, 4.0],
    )
    zero = ControlRepresentation(
        control.__class__.from_fs({**control.to_fs(), "extrapolate": "zero"})
    )
    np.testing.assert_allclose(
        zero.evaluate(np.asarray(control.coefficients), context), [0.0, 1.0, 4.0, 0.0]
    )


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


def _land_tensor(**kwargs):
    return TensorBSplineControl.from_nodes(
        ["x", "z"],
        [[0.45, 0.9, 1.27, 1.6], np.linspace(0.0, 0.15, 6)],
        degree=[1, 3],
        **kwargs,
    )


def test_tensor_representation_matches_the_hat_lattice_and_pulls_back_exactly():
    hat = TensorHatControl(
        axes=["x", "z"],
        shape=[3, 4],
        origin=[0.0, 1.0],
        spacing=[0.5, 2.0],
        coefficients=np.zeros(12),
    )
    spline = TensorBSplineControl.from_nodes(["x", "z"], hat.axis_coordinates)
    rng = np.random.default_rng(4)
    x, z = rng.uniform(-0.5, 1.5, 60), rng.uniform(0.0, 8.0, 60)
    context = EvaluationContext({"x": x, "z": z})
    lattice = TensorControlRepresentation(hat)
    representation = TensorControlRepresentation(spline)
    assert representation.shape == (3, 4) and representation.size == 12

    dense = representation.sampling_operator(context).toarray()
    np.testing.assert_allclose(
        dense, lattice.sampling_operator(context).toarray(), rtol=0.0, atol=1e-14
    )
    # A partition of unity inside the closed box, no update outside it.
    inside = (x >= 0.0) & (x <= 1.0) & (z >= 1.0) & (z <= 7.0)
    np.testing.assert_allclose(dense.sum(axis=1), inside.astype(float), atol=1e-14)
    direction = rng.normal(size=12)
    dual = rng.normal(size=60)
    lhs = np.dot(representation.evaluate(direction, context, reshape=False), dual)
    rhs = np.dot(direction, representation.pullback(dual, context))
    np.testing.assert_allclose(lhs, rhs, rtol=1.0e-14, atol=1.0e-14)


def test_tensor_representation_orders_columns_first_axis_fastest():
    control = _land_tensor()
    representation = TensorControlRepresentation(control)
    nodes = control.coordinates
    # At a tensor node with open knots, only that node's x hat is nonzero.
    context = EvaluationContext({"x": [0.9], "z": [0.0]})
    row = representation.sampling_operator(context).toarray()[0]
    assert np.flatnonzero(row).tolist() == [1]
    np.testing.assert_allclose(nodes[1], [0.9, 0.0])
    # Column i0 + 4 * i1 multiplies the x basis i0 by the z basis i1.
    context = EvaluationContext({"x": [0.9, 1.0], "z": [0.05, 0.11]})
    dense = representation.sampling_operator(context).toarray()
    factors = [
        rep.sampling_operator(EvaluationContext({axis: values})).toarray()
        for rep, axis, values in zip(
            representation.axes_1d, ("x", "z"), ([0.9, 1.0], [0.05, 0.11])
        )
    ]
    np.testing.assert_allclose(
        dense, np.einsum("pi,pj->pji", *factors).reshape(2, -1), atol=1e-15
    )


def test_tensor_representation_differentiates_per_axis_and_projects_exactly():
    control = _land_tensor()
    representation = TensorControlRepresentation(control)
    nodes = control.coordinates
    coefficients = (1.0 + 2.0 * nodes[:, 0]) * (1.0 + 3.0 * nodes[:, 1])
    rng = np.random.default_rng(8)
    x, z = rng.uniform(0.46, 1.59, 30), rng.uniform(0.001, 0.149, 30)
    context = EvaluationContext({"x": x, "z": z})
    np.testing.assert_allclose(
        representation.evaluate(coefficients, context), (1 + 2 * x) * (1 + 3 * z)
    )
    np.testing.assert_allclose(
        representation.evaluate(coefficients, context, derivative_order=(0, 1)),
        3.0 * (1.0 + 2.0 * x),
    )
    np.testing.assert_allclose(
        representation.evaluate(coefficients, context, derivative_order=(1, 0)),
        2.0 * (1.0 + 3.0 * z),
    )
    with pytest.raises(ValueError, match="per axis"):
        representation.sampling_operator(context, derivative_order=1)
    with pytest.raises(ValueError, match="one value per axis"):
        representation.sampling_operator(context, derivative_order=(0, 1, 0))
    grid_x, grid_z = np.meshgrid(
        np.linspace(0.45, 1.6, 13), np.linspace(0.0, 0.15, 31), indexing="ij"
    )
    samples = EvaluationContext({"x": grid_x, "z": grid_z})
    recovered = representation.project(
        representation.evaluate(coefficients, samples), samples, tolerance=1e-14
    )
    np.testing.assert_allclose(recovered, coefficients, atol=1e-9)


def test_held_tensor_representation_clamps_every_axis():
    control = _land_tensor(extrapolate="hold")
    representation = TensorControlRepresentation(control)
    assert representation.hold
    outside = EvaluationContext({"x": [0.0, 2.0, 1.0], "z": [0.05, -0.3, 0.4]})
    clamped = EvaluationContext({"x": [0.45, 1.6, 1.0], "z": [0.05, 0.0, 0.15]})
    np.testing.assert_allclose(
        representation.sampling_operator(outside).toarray(),
        representation.sampling_operator(clamped).toarray(),
    )
    # Derivative rows vanish where the differentiated axis is clamped.
    slope = representation.sampling_operator(outside, derivative_order=(1, 0))
    assert slope.toarray()[:2].any(axis=1).tolist() == [False, False]
    zero = TensorControlRepresentation(_land_tensor())
    assert zero.sampling_operator(outside).nnz == 0
    with pytest.raises(TypeError, match="tensor-hat or tensor B-spline"):
        TensorControlRepresentation(HatControl("z", 1.0, [0.0, 0.0]))
