"""Opt-in Python composition of native receiver jets and synthetic attributes."""

import numpy as np

from .misfit import Misfit, Normalization, ObjectiveTerm
from .operators import ModelOperator
from .spectral_attributes import SpectralAttribute, SpectralJetLinearization


def receiver_coordinates(problem):
    """Resolve point-channel coordinates without guessing expanded device geometry."""
    from frequensolve import Q_
    from frequensolve.seismic.receivers import ReceiverNode

    result = {}
    for term in problem.misfit.objective_terms(
        [g.name for g in problem.observed_groups]
    ):
        group = problem.simulation.acquisition.receiver_groups[term.receiver_group]
        if not isinstance(group.device, ReceiverNode):
            raise ValueError(
                "automatic synthetic-array geometry currently requires ReceiverNode; "
                "use Comparison.bind(jets, coordinates=...) for expanded devices, "
                "DAS channel positions or cable arc length"
            )
        coords = group.coordinates
        units = getattr(coords, "units", None)
        if units is None:
            units = problem.simulation.units.defaults.get("length")
        if units is None:
            raise ValueError(
                "synthetic array geometry needs explicit coordinate length units"
            )
        result[term.id] = Q_(coords.get(), units)
    return result


def selected(problem):
    return any(
        t.comparison.reliability is not None
        for t in problem.misfit.objective_terms(
            [g.name for g in problem.observed_groups]
        )
    )


def _validate(problem, terms):
    if problem.patches is not None or problem.kernel_derivative is not None:
        raise ValueError(
            "synthetic comparisons cannot be combined with patches or kernel_derivative"
        )
    misfit = problem.misfit
    if (
        misfit.preprocess
        or misfit.include_default_preprocess
        or misfit.group_preprocess
        or any(
            misfit.projection_for(t.receiver_group).kind != "identity" for t in terms
        )
        or any(t.preprocess for t in terms)
    ):
        raise ValueError(
            "synthetic comparisons currently require unprocessed, unprojected receiver traces"
        )
    if problem.smoothing is not None:
        raise ValueError(
            "synthetic comparisons return raw gradients; apply smoothing explicitly afterwards"
        )
    for term in terms:
        if (
            term.comparison.kind != "phase_derivative"
            or term.comparison.source_derivative != "frozen"
        ):
            raise ValueError(
                "synthetic comparison terms currently require frozen-source phase_derivative"
            )
        norm = term.normalization
        if norm.kind != "explicit" or norm.components is not None:
            raise ValueError(
                "synthetic comparisons currently need scalar explicit normalization"
            )
        if term.loss.kind not in {"l2", "student_t"}:
            raise ValueError(
                "synthetic comparisons currently support L2 or Student-t loss"
            )


def _definition(term):
    scale = term.normalization.value
    if isinstance(scale, dict):
        from frequensolve import Q_

        scale = Q_(scale["value"], scale["units"]).to("s").magnitude
    loss = term.loss
    return SpectralAttribute(
        "phase_slope",
        time_scale=float(scale),
        relative_floor=term.comparison.relative_amplitude_floor,
        confidence_floor=term.comparison.confidence_floor or None,
        loss=loss.kind,
        loss_scale=np.sqrt(loss.nu * loss.c2) if loss.kind == "student_t" else 1,
    )


class _Jacobian(ModelOperator):
    def __init__(self, linearization):
        self.linearization = linearization
        super().__init__(
            linearization.space,
            None,
            shape=(
                sum(b.data.residual.size for _, b in linearization.parts),
                linearization.space.size,
            ),
        )

    def _matvec(self, x):
        return np.concatenate(
            [np.sqrt(w) * b.jvp(x).ravel().real for w, b in self.linearization.parts]
        )

    def _rmatvec(self, y):
        result = self.linearization.space.zeros()
        start = 0
        for weight, bound in self.linearization.parts:
            end = start + bound.data.residual.size
            result = result + bound.vjp(
                np.sqrt(weight)
                * np.asarray(y[start:end]).reshape(bound.data.residual.shape)
            )
            start = end
        return result.values


class SyntheticComparisonLinearization:
    """Saved attribute objective; ``trial`` retains fixed reliability and arrays.

    This composition uses existing native modeled JVP/VJP actions. It does not
    alter native receiver evaluation. ``gradient`` is a raw model covector;
    ``field`` retains the native material basis and mass convention.
    """

    def __init__(self, problem, parts, identities, *, gradient):
        self.problem, self.parts, self.identities = problem, parts, identities
        first = parts[0][1].jets.planes[0]
        self.space, self.state, self.point = first.space, first.state, first.point
        self.frequencies = problem.frequencies
        self.value = sum(weight * bound.value for weight, bound in parts)
        self.report = {}
        for (weight, bound), (_, term_id) in zip(parts, identities):
            self.report[term_id] = self.report.get(term_id, 0.0) + weight * bound.value
        self.gradient = None
        if gradient:
            self.gradient = self.space.zeros()
            for weight, bound in parts:
                if weight:
                    self.gradient = self.gradient + weight * bound.gradient()
            self.gradient._field_factory = self.field
        self.jacobian = _Jacobian(self)
        self.normal = self.jacobian.H @ self.jacobian

    def objective_residual(self):
        return np.concatenate(
            [np.sqrt(w) * b.data.residual.ravel().real for w, b in self.parts]
        )

    @property
    def jobs(self):
        return tuple(lin.job for _, b in self.parts for lin in b.jets.planes)

    @property
    def regularization_job(self):
        return self.parts[0][1].jets.planes[0].regularization_job

    def field(self, vector=None, **kwargs):
        return (
            self.parts[0][1]
            .jets.planes[0]
            .field(self.gradient if vector is None else vector, **kwargs)
        )

    def trial(self, v, *, gradient=False):
        """Evaluate candidate values/gradients without refreshing reliability."""
        return linearize(self.problem, v, gradient=gradient, reference=self)


def linearize(problem, v=None, *, gradient=True, reference=None):
    terms = problem.misfit.objective_terms([g.name for g in problem.observed_groups])
    _validate(problem, terms)
    if reference is not None and reference.problem is not problem:
        raise ValueError("reliability reference must belong to this exact problem view")
    frequencies = problem.frequencies
    weights = (
        np.ones(len(frequencies))
        if problem.weights is None
        else np.asarray(problem.weights)
    )
    parts, identities = [], []
    for frequency, weight in zip(frequencies, weights):
        for term in terms:
            raw = problem.with_misfit(
                Misfit.terms(
                    ObjectiveTerm(
                        term.receiver_group,
                        id=term.id,
                        normalization=Normalization.explicit(1),
                    )
                )
            )
            raw = raw.restrict(frequencies=[frequency], weights=[1.0])
            coordinates = (
                receiver_coordinates(raw) if term.comparison.array is not None else None
            )
            jets = SpectralJetLinearization.from_problem(raw, order=1, state=v)
            if parts and not parts[0][1].jets.space.equivalent(jets.space):
                raise ValueError(
                    "synthetic comparisons require a frozen control basis across terms/frequencies"
                )
            identity = frequency, term.id
            if reference is None:
                factor = term.weight
                if term.loss.kind == "student_t":
                    factor *= (term.loss.nu + 1) / (term.loss.nu * term.loss.c2)
                if term.normalization.reduction == "sum":
                    factor *= len(jets.keys)
                bound = term.comparison.bind(
                    jets,
                    coordinates=coordinates,
                    definition=_definition(term),
                    row_weights=np.full(len(jets.keys), factor),
                )
            else:
                index = reference.identities.index(identity)
                bound = reference.parts[index][1].trial(jets)
            parts.append((float(weight), bound))
            identities.append(identity)
    return SyntheticComparisonLinearization(
        problem, parts, identities, gradient=gradient
    )
