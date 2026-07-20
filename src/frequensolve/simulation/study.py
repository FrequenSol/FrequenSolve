"""Study-based authoring of related seismic simulations."""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Iterable, Mapping
from string import Formatter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Optional, Union

from frequensolve.simulation.simulation import SeismicSimulation

if TYPE_CHECKING:
    from frequensolve.project.project import Project

__all__ = ["SimulationCase", "SimulationStudy"]


_CaseInput = Union["SimulationCase", Mapping[str, str]]


class SimulationCase:
    """One selection of named values from a :class:`SimulationStudy`.

    Case attributes expose independent copies of the selected values. For
    example, a ``model`` parameter is available as ``case.model`` inside the
    study's simulation builder. ``selections`` retains the stable choice labels
    used for naming and provenance.
    """

    def __init__(
        self,
        study: "SimulationStudy",
        selections: Mapping[str, str],
        values: Mapping[str, Any],
        *,
        index: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        self._study = study
        self._selections = MappingProxyType(dict(selections))
        self._values = dict(values)
        self.index = index
        self.name = name

    @property
    def selections(self) -> Mapping[str, str]:
        """Selected choice label for each study parameter."""

        return self._selections

    def __getattr__(self, name: str) -> Any:
        values = self.__dict__.get("_values", {})
        if name in values:
            return values[name]
        raise AttributeError(
            f"{type(self).__name__!s} has no parameter or attribute {name!r}"
        )

    def new_simulation(
        self,
        *,
        physics: str,
        dimension: int | float | str,
        **kwargs: Any,
    ) -> SeismicSimulation:
        """Create a detached simulation named for this case.

        The simulation is attached to the project only after every requested
        study case has been built successfully.
        """

        self._require_materialization_context()
        return self._study.project._create_simulation(
            name=self.name,
            physics=physics,
            dimension=dimension,
            attach=False,
            **kwargs,
        )

    def clone(self, base: SeismicSimulation) -> SeismicSimulation:
        """Return an independent, detached copy of ``base`` for this case."""

        self._require_materialization_context()
        if not isinstance(base, SeismicSimulation):
            raise TypeError(
                "SimulationCase.clone() requires a SeismicSimulation; "
                f"got {type(base).__name__}"
            )

        memo = {}
        base_project = getattr(base, "_project", None)
        if base_project is not None:
            # Keep the owning project out of the cloned simulation graph. The
            # clone is rebound to this study's project below.
            memo[id(base_project)] = base_project
        try:
            simulation = copy.deepcopy(base, memo)
        except Exception as exc:
            raise TypeError(
                f"Could not clone base simulation {base.name!r}: {exc}"
            ) from exc

        simulation.name = self.name
        simulation._file = None
        simulation._project = self._study.project
        simulation.relocate(self._study.project.path)
        return simulation

    def _require_materialization_context(self) -> None:
        if self.index is None or self.name is None:
            raise RuntimeError(
                "This case is only a selection specification. Use it in "
                "study.materialize(cases=[...]); case.new_simulation() and "
                "case.clone() are available inside the @study.simulation builder."
            )


class SimulationStudy:
    """Define and materialize a family of related seismic simulations.

    Parameters are mappings from stable choice labels to authoring values. A
    builder registered with :meth:`simulation` receives a
    :class:`SimulationCase` exposing the selected values as attributes.
    """

    _RESERVED_PARAMETERS = {
        "clone",
        "index",
        "name",
        "new_simulation",
        "selections",
        "study",
    }

    def __init__(
        self,
        project: "Project",
        name: str,
        parameters: Mapping[str, Mapping[str, Any]],
        *,
        name_template: Optional[str] = None,
        max_cases: Optional[int] = 1000,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("A simulation study requires a non-empty name")
        if not parameters:
            raise ValueError("A simulation study requires at least one parameter")
        if max_cases is not None and (not isinstance(max_cases, int) or max_cases < 1):
            raise ValueError("max_cases must be a positive integer or None")

        self.project = project
        self.name = name
        self.max_cases = max_cases
        self._parameters = self._normalize_parameters(parameters)
        self.name_template = name_template or self._default_name_template()
        self._builder: Optional[Callable[[SimulationCase], SeismicSimulation]] = None
        self._validate_name_template()

    @property
    def parameters(self) -> Mapping[str, tuple[str, ...]]:
        """Parameter names mapped to their available choice labels."""

        return MappingProxyType(
            {name: tuple(choices) for name, choices in self._parameters.items()}
        )

    def simulation(
        self, builder: Callable[[SimulationCase], SeismicSimulation]
    ) -> Callable[[SimulationCase], SeismicSimulation]:
        """Register the function that builds one simulation from a case.

        This method is intended to be used as ``@study.simulation``. Repeating
        the decorator replaces the builder, which is convenient when rerunning
        notebook cells.
        """

        if not callable(builder):
            raise TypeError("The simulation builder must be callable")
        self._builder = builder
        return builder

    def case(self, **selections: str) -> SimulationCase:
        """Create a validated explicit case selection.

        The returned object is a lightweight selection for
        ``materialize(cases=[...])``. The same class is populated with a name,
        index, and independent parameter values inside the simulation builder.
        """

        labels = self._validate_selections(selections)
        return SimulationCase(self, labels, {})

    def preview(
        self, cases: Optional[Union[_CaseInput, Iterable[_CaseInput]]] = None
    ) -> list[dict[str, Any]]:
        """Return names and choice labels without invoking the builder."""

        resolved = self._resolve_cases(cases)
        self._preflight_names(resolved)
        return [
            {"name": name, "index": index, **labels} for index, labels, name in resolved
        ]

    def materialize(
        self, cases: Optional[Union[_CaseInput, Iterable[_CaseInput]]] = None
    ) -> list[SeismicSimulation]:
        """Build and attach either all combinations or explicit ``cases``.

        With no argument, the Cartesian product of every parameter choice is
        materialized in declaration order. Simulations are attached atomically:
        if any builder fails, the project's simulation list is left unchanged.
        """

        if self._builder is None:
            raise RuntimeError(
                "No simulation builder is registered; decorate a function with "
                "@study.simulation before materializing the study"
            )

        resolved = self._resolve_cases(cases)
        self._preflight_names(resolved)
        original_simulations = list(self.project.simulations)
        materialized: list[SeismicSimulation] = []
        simulation_ids: set[int] = set()

        try:
            for index, labels, name in resolved:
                case = SimulationCase(
                    self,
                    labels,
                    self._copy_values(labels),
                    index=index,
                    name=name,
                )
                simulation = self._builder(case)
                if not self._same_objects(
                    self.project.simulations, original_simulations
                ):
                    raise RuntimeError(
                        "The study builder changed project.simulations. Use "
                        "case.new_simulation() or case.clone() so study "
                        "materialization can remain atomic."
                    )
                if not isinstance(simulation, SeismicSimulation):
                    raise TypeError(
                        "The study builder must return a SeismicSimulation; "
                        f"got {type(simulation).__name__} for case {name!r}"
                    )
                if any(simulation is existing for existing in original_simulations):
                    raise ValueError(
                        "The study builder returned an existing project simulation. "
                        "Return case.clone(base) to create an independent variant."
                    )
                if id(simulation) in simulation_ids:
                    raise ValueError(
                        "The study builder returned the same simulation object for "
                        "more than one case"
                    )

                simulation_ids.add(id(simulation))
                simulation.name = name
                simulation._file = None
                simulation._project = self.project
                simulation.relocate(self.project.path)
                materialized.append(simulation)
        except Exception:
            self.project.simulations.clear()
            self.project.simulations.extend(original_simulations)
            raise

        self.project.simulations.extend(materialized)
        return materialized

    def _normalize_parameters(
        self, parameters: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for parameter, choices in parameters.items():
            if not isinstance(parameter, str) or not parameter.isidentifier():
                raise ValueError(
                    f"Study parameter names must be valid Python identifiers; got {parameter!r}"
                )
            if parameter in self._RESERVED_PARAMETERS:
                raise ValueError(f"Study parameter name {parameter!r} is reserved")
            if not isinstance(choices, Mapping) or not choices:
                raise ValueError(
                    f"Study parameter {parameter!r} requires a non-empty mapping "
                    "of choice labels to values"
                )
            normalized_choices: dict[str, Any] = {}
            for label, value in choices.items():
                if not isinstance(label, str) or not label:
                    raise ValueError(
                        f"Choice labels for parameter {parameter!r} must be non-empty strings"
                    )
                normalized_choices[label] = value
            normalized[parameter] = normalized_choices
        return normalized

    def _default_name_template(self) -> str:
        fields = ["{study}"]
        fields.extend(f"{parameter}-{{{parameter}}}" for parameter in self._parameters)
        return "__".join(fields)

    def _validate_name_template(self) -> None:
        if not isinstance(self.name_template, str) or not self.name_template:
            raise ValueError("name_template must be a non-empty string")
        allowed = {"study", "index", *self._parameters}
        try:
            parsed = list(Formatter().parse(self.name_template))
        except ValueError as exc:
            raise ValueError(f"Invalid simulation study name_template: {exc}") from exc

        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if not field_name:
                raise ValueError("name_template cannot contain an empty '{}' field")
            if field_name not in allowed:
                raise ValueError(
                    f"Unknown name_template field {field_name!r}; expected one of "
                    f"{sorted(allowed)}"
                )
            if conversion is not None:
                raise ValueError(
                    "name_template conversions such as !r are not supported"
                )
            if "{" in format_spec or "}" in format_spec:
                raise ValueError(
                    "Nested fields in name_template format specs are not supported"
                )

    def _validate_selections(self, selections: Mapping[str, Any]) -> dict[str, str]:
        unknown = set(selections).difference(self._parameters)
        missing = set(self._parameters).difference(selections)
        if unknown:
            raise ValueError(
                f"Unknown study parameter selection(s): {', '.join(sorted(unknown))}"
            )
        if missing:
            raise ValueError(
                f"Missing study parameter selection(s): {', '.join(sorted(missing))}"
            )

        labels: dict[str, str] = {}
        for parameter, choices in self._parameters.items():
            label = selections[parameter]
            if not isinstance(label, str) or label not in choices:
                raise ValueError(
                    f"Unknown choice {label!r} for parameter {parameter!r}; "
                    f"expected one of {list(choices)}"
                )
            labels[parameter] = label
        return labels

    def _copy_values(self, labels: Mapping[str, str]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for parameter, label in labels.items():
            try:
                values[parameter] = copy.deepcopy(self._parameters[parameter][label])
            except Exception as exc:
                raise TypeError(
                    f"Could not copy choice {label!r} for study parameter "
                    f"{parameter!r}: {exc}"
                ) from exc
        return values

    def _resolve_cases(
        self, cases: Optional[Union[_CaseInput, Iterable[_CaseInput]]]
    ) -> list[tuple[int, dict[str, str], str]]:
        if cases is None:
            parameter_names = list(self._parameters)
            label_sets = [list(choices) for choices in self._parameters.values()]
            labels_list = [
                dict(zip(parameter_names, combination))
                for combination in itertools.product(*label_sets)
            ]
        else:
            if isinstance(cases, (SimulationCase, Mapping)):
                case_items = [cases]
            else:
                case_items = list(cases)
            labels_list = []
            for item in case_items:
                if isinstance(item, SimulationCase):
                    if item._study is not self:
                        raise ValueError("Explicit cases must belong to this study")
                    selections = item.selections
                elif isinstance(item, Mapping):
                    selections = item
                else:
                    raise TypeError(
                        "Explicit cases must be study.case(...) objects or mappings; "
                        f"got {type(item).__name__}"
                    )
                labels_list.append(self._validate_selections(selections))

        if self.max_cases is not None and len(labels_list) > self.max_cases:
            raise ValueError(
                f"Study {self.name!r} would materialize {len(labels_list)} cases, "
                f"exceeding max_cases={self.max_cases}. Increase max_cases or pass "
                "explicit cases."
            )

        return [
            (index, labels, self._format_name(index, labels))
            for index, labels in enumerate(labels_list)
        ]

    def _format_name(self, index: int, labels: Mapping[str, str]) -> str:
        context: dict[str, Any] = {"study": self.name, "index": index, **labels}
        try:
            return self.name_template.format_map(context)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Could not format simulation name for study {self.name!r}: {exc}"
            ) from exc

    def _preflight_names(
        self, resolved: Iterable[tuple[int, Mapping[str, str], str]]
    ) -> None:
        names: list[str] = []
        for _, _, name in resolved:
            self._validate_simulation_name(name)
            names.append(name)

        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "The study name_template produces duplicate simulation names: "
                + ", ".join(repr(name) for name in duplicates)
            )

        existing = {str(simulation.name) for simulation in self.project.simulations}
        collisions = sorted(existing.intersection(names))
        if collisions:
            raise ValueError(
                "Study simulation name(s) already exist in the project: "
                + ", ".join(repr(name) for name in collisions)
            )

    @staticmethod
    def _validate_simulation_name(name: str) -> None:
        if not isinstance(name, str) or not name or name in {".", ".."}:
            raise ValueError(f"Invalid materialized simulation name: {name!r}")
        if any(character in name for character in ("/", "\\", "\x00")):
            raise ValueError(
                f"Materialized simulation name {name!r} contains an unsafe path character"
            )

    @staticmethod
    def _same_objects(left: Iterable[Any], right: Iterable[Any]) -> bool:
        left_items = list(left)
        right_items = list(right)
        return len(left_items) == len(right_items) and all(
            left_item is right_item
            for left_item, right_item in zip(left_items, right_items)
        )
