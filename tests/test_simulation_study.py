import pytest

from frequensolve.project import Project
from frequensolve.simulation import SimulationCase, SimulationStudy


def test_project_exposes_only_study_factory():
    assert hasattr(Project, "study")
    assert not hasattr(Project, "simulation_study")


def test_study_materializes_cartesian_product_with_custom_names(tmp_path):
    project = Project(name="project", path=tmp_path)
    base = project.new_simulation(
        name="base",
        physics="acoustic",
        dimension=2,
    )
    base.extra["base_marker"] = []
    seen = []
    receivers = {"coarse": [0.0, 1.0], "dense": [0.0, 0.5, 1.0]}
    models = {"reference": {"vp": 1.5}, "smoothed": {"vp": 1.4}}
    study = project.study(
        "survey_design",
        name_template="base__{receiver}__{source}__{model}",
        receiver=receivers,
        source={"explosive": "scalar", "vertical_force": "vector"},
        model=models,
    )

    @study.simulation
    def build(case):
        assert isinstance(case, SimulationCase)
        seen.append((case.index, case.name, dict(case.selections)))
        simulation = case.clone(base)
        simulation.extra["receiver"] = case.receiver
        simulation.extra["source"] = case.source
        simulation.extra["model"] = case.model
        simulation.extra["base_marker"].append(case.name)
        return simulation

    simulations = study.materialize()

    assert isinstance(study, SimulationStudy)
    assert [simulation.name for simulation in simulations] == [
        "base__coarse__explosive__reference",
        "base__coarse__explosive__smoothed",
        "base__coarse__vertical_force__reference",
        "base__coarse__vertical_force__smoothed",
        "base__dense__explosive__reference",
        "base__dense__explosive__smoothed",
        "base__dense__vertical_force__reference",
        "base__dense__vertical_force__smoothed",
    ]
    assert list(project.simulations) == [base, *simulations]
    assert seen[0] == (
        0,
        "base__coarse__explosive__reference",
        {"receiver": "coarse", "source": "explosive", "model": "reference"},
    )

    simulations[0].extra["receiver"].append(2.0)
    simulations[0].extra["model"]["vp"] = 9.9
    assert simulations[1].extra["receiver"] == [0.0, 1.0]
    assert simulations[2].extra["model"] == {"vp": 1.5}
    assert receivers["coarse"] == [0.0, 1.0]
    assert models["reference"] == {"vp": 1.5}
    assert base.extra["base_marker"] == []


def test_study_previews_and_materializes_only_explicit_cases(tmp_path):
    project = Project(name="project", path=tmp_path)
    study = project.study(
        "selected",
        name_template="{study}__{index:03d}__{receiver}__{model}",
        receiver={"coarse": [0], "dense": [0, 1]},
        model={"reference": 1, "smoothed": 2},
    )
    cases = [
        study.case(receiver="dense", model="smoothed"),
        {"receiver": "coarse", "model": "reference"},
    ]

    assert study.preview(cases=cases) == [
        {
            "name": "selected__000__dense__smoothed",
            "index": 0,
            "receiver": "dense",
            "model": "smoothed",
        },
        {
            "name": "selected__001__coarse__reference",
            "index": 1,
            "receiver": "coarse",
            "model": "reference",
        },
    ]

    @study.simulation
    def build(case):
        simulation = case.new_simulation(physics="acoustic", dimension=2)
        simulation.extra["receiver"] = case.receiver
        simulation.extra["model"] = case.model
        return simulation

    simulations = study.materialize(cases=cases)

    assert [simulation.name for simulation in simulations] == [
        "selected__000__dense__smoothed",
        "selected__001__coarse__reference",
    ]
    assert simulations[0].extra == {"receiver": [0, 1], "model": 2}


def test_study_default_name_includes_parameter_names(tmp_path):
    project = Project(name="project", path=tmp_path)
    study = project.study(
        "experiment",
        receiver={"coarse": object()},
        model={"reference": object()},
    )

    assert study.preview() == [
        {
            "name": "experiment__receiver-coarse__model-reference",
            "index": 0,
            "receiver": "coarse",
            "model": "reference",
        }
    ]


@pytest.mark.parametrize(
    ("selections", "message"),
    [
        ({"receiver": "coarse"}, "Missing study parameter"),
        (
            {"receiver": "coarse", "model": "reference", "source": "x"},
            "Unknown study parameter",
        ),
        (
            {"receiver": "missing", "model": "reference"},
            "Unknown choice 'missing'",
        ),
    ],
)
def test_study_rejects_invalid_explicit_cases(tmp_path, selections, message):
    project = Project(name="project", path=tmp_path)
    study = project.study(
        "experiment",
        receiver={"coarse": 1},
        model={"reference": 2},
    )

    with pytest.raises(ValueError, match=message):
        study.case(**selections)


def test_study_validates_name_template_and_rendered_names(tmp_path):
    project = Project(name="project", path=tmp_path)

    with pytest.raises(ValueError, match="Unknown name_template field 'missing'"):
        project.study(
            "bad",
            name_template="{missing}",
            receiver={"coarse": 1},
        )

    duplicate_study = project.study(
        "duplicate",
        name_template="{receiver}",
        receiver={"coarse": 1},
        model={"reference": 1, "smoothed": 2},
    )
    with pytest.raises(ValueError, match="duplicate simulation names"):
        duplicate_study.preview()

    unsafe_study = project.study(
        "unsafe",
        name_template="{receiver}",
        receiver={"surface/dense": 1},
    )
    with pytest.raises(ValueError, match="unsafe path character"):
        unsafe_study.preview()


def test_study_rejects_existing_project_name_before_building(tmp_path):
    project = Project(name="project", path=tmp_path)
    project.new_simulation(name="variant", physics="acoustic", dimension=2)
    study = project.study(
        "collision",
        name_template="variant",
        receiver={"coarse": 1},
    )
    builder_calls = []

    @study.simulation
    def build(case):
        builder_calls.append(case.name)
        return case.new_simulation(physics="acoustic", dimension=2)

    with pytest.raises(ValueError, match="already exist in the project"):
        study.materialize()

    assert builder_calls == []
    assert len(project.simulations) == 1


def test_study_materialization_is_atomic_when_builder_fails(tmp_path):
    project = Project(name="project", path=tmp_path)
    base = project.new_simulation(name="base", physics="acoustic", dimension=2)
    study = project.study(
        "atomic",
        receiver={"coarse": 1, "dense": 2},
    )

    @study.simulation
    def build(case):
        if case.selections["receiver"] == "dense":
            raise RuntimeError("builder failed")
        return case.clone(base)

    with pytest.raises(RuntimeError, match="builder failed"):
        study.materialize()

    assert list(project.simulations) == [base]


def test_study_rolls_back_builder_project_attachment(tmp_path):
    project = Project(name="project", path=tmp_path)
    study = project.study("atomic", receiver={"coarse": 1})

    @study.simulation
    def build(case):
        return project.new_simulation(
            name=case.name,
            physics="acoustic",
            dimension=2,
        )

    with pytest.raises(RuntimeError, match="case.new_simulation"):
        study.materialize()

    assert project.simulations == []


def test_study_enforces_case_guard_and_builder_registration(tmp_path):
    project = Project(name="project", path=tmp_path)
    study = project.study(
        "guarded",
        max_cases=1,
        receiver={"coarse": 1, "dense": 2},
    )

    with pytest.raises(ValueError, match="exceeding max_cases=1"):
        study.preview()

    one_case = study.case(receiver="coarse")
    with pytest.raises(RuntimeError, match="inside the @study.simulation builder"):
        one_case.new_simulation(physics="acoustic", dimension=2)

    with pytest.raises(RuntimeError, match="No simulation builder"):
        study.materialize(cases=[one_case])
