"""Behavior tests for the packaged simulation-knowledge catalog."""

from __future__ import annotations

import copy
import inspect
import json
from importlib.resources import files
from pathlib import Path

import jsonschema
import numpy as np
import pytest

import frequensolve as fs


def _resource_payload(name: str) -> dict:
    return json.loads(
        files("frequensolve.knowledge").joinpath(name).read_text(encoding="utf-8")
    )


def _build_starter_job(scenario: fs.StarterScenario, project_path):
    setup = scenario.setup
    project_config = dict(setup["project"])
    project = fs.Project(path=project_path, **project_config)

    simulation_config = dict(setup["simulation"])
    simulation = project.new_simulation(**simulation_config)

    model_config = dict(setup["model"])
    surfaces = model_config.pop("surfaces")
    layers = model_config.pop("layers")
    assert model_config.pop("type") == "LayeredModel"
    model = fs.LayeredModel(**model_config)
    for index, surface in enumerate(surfaces):
        model.add_surface(**surface)
        if index < len(layers):
            model.add_layer(**layers[index])
    simulation += model

    mesh_config = dict(setup["mesh"])
    assert mesh_config.pop("type") == "HexMeshGenerator"
    adapt = mesh_config.pop("adapt")
    source_grading = mesh_config.pop("source_grading")
    simulation += model.hex_mesh_generator(**mesh_config)
    simulation.mesh.set_adapt(**adapt)
    simulation.mesh.set_source_grading(**source_grading)

    for boundary in setup["boundary_conditions"]:
        simulation += fs.BoundaryCondition(**boundary)

    acquisition_config = setup["acquisition"]
    acquisition = fs.Acquisition()
    acquisition.add_sources(**acquisition_config["source"])
    receiver_config = acquisition_config["receiver_group"]
    receiver = fs.ReceiverNode(name=receiver_config["device_name"])
    receiver.add_component(**receiver_config["component"])
    line = receiver_config["coordinate_line"]
    assert line["axis"] == "x"
    coordinates = [
        [x, line["fixed"]["z"]]
        for x in np.linspace(line["start"], line["stop"], line["count"])
    ]
    acquisition.add_receiver_group(
        name=receiver_config["name"],
        device=receiver,
        coords=coordinates,
    )
    simulation += acquisition

    simulation += fs.Discretization(**setup["discretization"])
    simulation += fs.SolverConfig(**setup["solver"])

    job_config = dict(setup["job"])
    assert job_config.pop("type") == "FrequencyDomainJob"
    outputs = job_config.pop("outputs")
    vtk_outputs = [fs.VtkOutput.domain(**vtk_config) for vtk_config in outputs["vtk"]]
    return project, fs.FrequencyDomainJob(
        simulation=simulation,
        outputs=vtk_outputs,
        **job_config,
    )


def test_packaged_catalog_matches_its_json_schema():
    schema = _resource_payload(fs.CATALOG_SCHEMA_RESOURCE)
    payload = _resource_payload(fs.CATALOG_RESOURCE)

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(payload, schema)


def test_catalog_exposes_exact_installed_release_identities():
    catalog = fs.load_simulation_knowledge()
    compatibility = fs.load_frequensolver_compatibility()

    assert catalog.identities.package_version == fs.__version__
    assert catalog.identities.declared_package_release == compatibility.package_release
    assert catalog.identities.catalog_schema == fs.CATALOG_SCHEMA
    assert catalog.identities.catalog_version == "1.0.0"
    assert catalog.identities.authoring_rules_schema == fs.AUTHORING_RULES_SCHEMA
    assert catalog.identities.compatibility_schema == compatibility.schema
    preferred = compatibility.preferred_frequensolver
    assert catalog.identities.preferred_frequensolver_release == (
        preferred.release if preferred is not None else None
    )
    assert catalog.identities.preferred_frequensolver_commit == (
        preferred.git_commit if preferred is not None else None
    )
    assert tuple(
        (
            contract.name,
            contract.identity,
            contract.owner,
            contract.source_revision,
        )
        for contract in catalog.identities.contracts
    ) == (
        (
            "simulation",
            "fs-simulation-1",
            "Sauce",
            "a54bdda81c98780fb4b805b92cf6df6c6e8bd29a",
        ),
        (
            "acquisition",
            "fs-acquisition-2",
            "Sauce",
            "a54bdda81c98780fb4b805b92cf6df6c6e8bd29a",
        ),
        (
            "job",
            "fs-job-1",
            "Sauce",
            "a54bdda81c98780fb4b805b92cf6df6c6e8bd29a",
        ),
    )


def test_catalog_physics_lookup_matches_public_registries():
    catalog = fs.load_simulation_knowledge()

    assert (
        tuple(entry.id for entry in catalog.physics_entries) == fs.supported_physics()
    )
    for entry in catalog.physics_entries:
        assert entry.aliases == fs.physics_aliases(entry.id)
        assert entry.supported_dimensions == fs.supported_dimensions_for_physics(
            entry.id
        )
        assert entry.output_components == tuple(
            fs.components_for_physics(entry.id).allowed_components()
        )
        for dimension in entry.supported_dimensions:
            config = fs.SimulationConfig(
                name=f"{entry.id}_{dimension}",
                physics=entry.id,
                dimension=dimension,
            )
            assert config.dimension == dimension
        if entry.property_requirements == "cataloged":
            assert entry.material_profile
            assert entry.required_properties
            assert all(
                fs.canonical_property_name(name) == name
                for name in entry.required_properties
            )
        else:
            assert entry.material_profile is None
            assert entry.required_properties == ()
        for alias in entry.aliases:
            assert catalog.lookup_physics(alias) == entry

    assert catalog.lookup_physics("biot").id == "poroelastic"
    assert catalog.lookup_physics("maxwell").id == "em"
    with pytest.raises(ValueError, match="require dimension=2"):
        fs.SimulationConfig(
            name="invalid_axisymmetric_dimension",
            physics="acoustic_axisym",
            dimension=3,
        )
    with pytest.raises(KeyError):
        catalog.lookup_physics("not-a-physics")


def test_public_api_catalog_matches_top_level_package_exports():
    catalog = fs.load_simulation_knowledge()

    assert catalog.public_api
    for entry in catalog.public_api:
        exported = getattr(fs, entry.symbol)
        assert entry.import_path == f"frequensolve.{entry.symbol}"
        assert (
            inspect.isclass(exported)
            if entry.kind == "class"
            else inspect.isfunction(exported)
        )
        assert catalog.lookup_public_api(entry.id) == entry
        assert catalog.lookup_public_api(entry.symbol.swapcase()) == entry
        assert catalog.lookup_public_api(entry.import_path) == entry
        for alias in entry.aliases:
            assert getattr(fs, alias) is exported
            assert catalog.lookup_public_api(alias) == entry

    assert catalog.lookup_public_api("ParaViewOutput").symbol == "VtkOutput"
    with pytest.raises(KeyError):
        catalog.lookup_public_api("RemovedSimulationJob")


def test_glossary_is_searchable_and_cross_referenced():
    catalog = fs.load_simulation_knowledge()

    assert catalog.glossary
    for entry in catalog.glossary:
        assert catalog.lookup_glossary(entry.id) == entry
        assert catalog.lookup_glossary(entry.term.swapcase()) == entry
        for alias in entry.aliases:
            assert catalog.lookup_glossary(alias) == entry
        for public_api_id in entry.related_api:
            assert catalog.lookup_public_api(public_api_id)

    for entry in catalog.public_api:
        for glossary_id in entry.related_glossary:
            assert catalog.lookup_glossary(glossary_id)

    dense = catalog.lookup_glossary("HDF5Dense source encoding")
    assert "real/imaginary axis" in dense.definition
    assert "native complex storage" in dense.definition
    remote = catalog.lookup_glossary("cluster-resident file")
    assert "missing absolute path outside the resolved local project" in (
        remote.definition
    )
    surfaces = catalog.lookup_glossary("surface selector")
    assert "synthetic bottom" in surfaces.definition
    assert "borehole names" in surfaces.definition
    with pytest.raises(KeyError):
        catalog.lookup_glossary("unknown simulation term")


def test_authoring_rules_match_public_boundaries_and_numerics():
    catalog = fs.load_simulation_knowledge()
    boundary_rules = catalog.lookup_authoring_rule("boundary_conditions")
    assert isinstance(boundary_rules, fs.BoundaryAuthoringRules)

    for condition in boundary_rules.documented_conditions:
        boundary = fs.BoundaryCondition(
            conditions=condition,
            boundaries="x_min",
        )
        assert boundary.has_condition(condition)

    pml = fs.BoundaryCondition(conditions="pml", boundaries="x_min").to_fs()
    assert pml["pml_wavelengths"] == boundary_rules.pml.pml_wavelengths
    assert pml["pml_exponent"] == boundary_rules.pml.pml_exponent
    assert pml["pml_reflectivity"] == boundary_rules.pml.pml_reflectivity
    free = fs.BoundaryCondition(conditions="free", boundaries="z_min").to_fs()
    assert not set(pml).intersection(free) - {"conditions", "boundaries"}
    with pytest.raises(ValueError, match="requires `conditions`"):
        fs.BoundaryCondition(boundaries="x_min")

    discretization_rules = catalog.lookup_authoring_rule("discretization")
    assert isinstance(discretization_rules, fs.DiscretizationAuthoringRules)
    assert fs.Discretization().to_fs() == discretization_rules.default_payload
    with pytest.raises(ValueError, match="mesh adaptivity"):
        fs.Discretization(order=4)

    solver_rules = catalog.lookup_authoring_rule("solver")
    assert isinstance(solver_rules, fs.SolverAuthoringRules)
    assert fs.SolverConfig().to_fs() == {
        "solve_on": solver_rules.default_solve_on,
        "max_iter": solver_rules.default_max_iter,
        "tolerance": solver_rules.default_tolerance,
        "precision": solver_rules.default_precision,
    }
    for solve_on in solver_rules.solve_on_values:
        assert fs.SolverConfig(solve_on=solve_on).to_fs()["solve_on"] == solve_on
    for precision in solver_rules.precision_values:
        assert fs.SolverConfig(precision=precision).to_fs()["precision"] == precision

    with pytest.raises(KeyError):
        catalog.lookup_authoring_rule("unknown-area")


def test_authoring_rules_match_frequency_and_output_behavior(tmp_path):
    catalog = fs.load_simulation_knowledge()
    _, starter_job = _build_starter_job(
        catalog.get_starter_scenario(),
        tmp_path / "authoring-rules",
    )
    simulation = starter_job.simulation

    frequency_rules = catalog.lookup_authoring_rule("frequencies")
    assert isinstance(frequency_rules, fs.FrequencyAuthoringRules)
    damped = fs.FrequencyDomainJob(
        name="damped",
        simulation=simulation,
        f_list=[10.0 + 2.0j],
    )
    assert damped.f_list[0].imag <= 0
    zero_frequency = fs.FrequencyDomainJob(
        name="zero",
        simulation=simulation,
        f_list=[0.0],
    )
    assert {issue.code for issue in fs.validate_job(zero_frequency).errors} == {
        "job.frequency.nonpositive"
    }

    valid_time = fs.TimeDomainJob(
        name="time",
        simulation=simulation,
        f_min=0.0,
        f_max=2.0,
        df=1.0,
    )
    assert all(complex(value).real > 0 for value in valid_time.f_list)
    with pytest.raises(ValueError, match="either df or T_max"):
        fs.TimeDomainJob(name="missing", simulation=simulation, f_max=2.0)
    with pytest.raises(ValueError, match="only one of df or T_max"):
        fs.TimeDomainJob(
            name="spacing",
            simulation=simulation,
            f_max=2.0,
            df=1.0,
            T_max=1.0,
        )
    with pytest.raises(ValueError, match="greater than or equal to"):
        fs.TimeDomainJob(
            name="damping",
            simulation=simulation,
            f_max=2.0,
            df=1.0,
            damping_factor=frequency_rules.damping_factor_minimum - 0.1,
        )

    output_rules = catalog.lookup_authoring_rule("outputs")
    assert isinstance(output_rules, fs.OutputAuthoringRules)
    assert starter_job.outputs.traces is not None
    expected_writer_formats = {
        "vtu": "vtu",
        "xdmf": "xdmf",
        "xmf": "xdmf",
        "vtr": "vtr",
    }
    for output_format in output_rules.vtk_formats:
        assert fs.VtkOutput(format=output_format).to_fs()["writer"]["format"] == (
            expected_writer_formats[output_format]
        )
    for item_kind in output_rules.item_kinds:
        assert fs.VtkItem(item_kind, "pressure").to_fs()["kind"] == item_kind
    for part in output_rules.complex_parts:
        item = fs.VtkItem("field", "pressure", parts=part).to_fs()
        assert item["parts"] == [part]

    grid = fs.CartesianGrid(n=[2, 2], x0=[0.0, 0.0], x1=[1.0, 1.0])
    targets = (
        fs.VtkOutput.domain(),
        fs.VtkOutput.surface(),
        fs.VtkOutput.grid(grid),
    )
    assert tuple(output.to_fs()["target"]["kind"] for output in targets) == (
        output_rules.vtk_targets
    )
    for upscale in (
        output_rules.vtk_upscale_minimum,
        output_rules.vtk_upscale_maximum,
    ):
        assert (
            fs.VtkOutput.domain(upscale=upscale).to_fs()["target"]["mesh"]["upscale"]
            == upscale
        )
    with pytest.raises(ValueError, match="integer from 0 to 2"):
        fs.VtkOutput.domain(upscale=output_rules.vtk_upscale_maximum + 1)
    with pytest.raises(ValueError, match="grid targets do not support upscale"):
        fs.VtkOutput.grid(
            grid,
            upscale=1,
        )

    invalid_source = fs.FrequencyDomainJob(
        name="invalid-source",
        simulation=simulation,
        f_list=[10.0],
        outputs=fs.VtkOutput.domain(fields=["pressure"], sources=[0]),
    )
    assert "outputs.source_id.invalid" in {
        issue.code for issue in fs.validate_job(invalid_source).errors
    }
    missing_wavefield_grid = fs.FrequencyDomainJob(
        name="missing-wavefield-grid",
        simulation=simulation,
        f_list=[10.0],
        outputs=fs.WavefieldOutput(fields=["pressure"]),
    )
    assert "outputs.wavefield.grid.missing" in {
        issue.code for issue in fs.validate_job(missing_wavefield_grid).errors
    }

    multiple_frequencies = fs.FrequencyDomainJob(
        name="multiple",
        simulation=simulation,
        f_list=[10.0, 20.0],
        outputs=fs.VtkOutput.domain(fields=["pressure"]),
    )
    assert {issue.code for issue in fs.validate_job(multiple_frequencies).errors} == {
        "outputs.vtk.frequency_count"
    }


def test_authoring_rules_capture_acquisition_file_and_surface_contracts():
    catalog = fs.load_simulation_knowledge()

    acquisition = catalog.lookup_authoring_rule("acquisition")
    assert isinstance(acquisition, fs.AcquisitionAuthoringRules)
    dense = acquisition.hdf5_dense_source_encoding
    assert isinstance(dense, fs.Hdf5DenseEncodingAuthoringRules)
    assert dense.coefficient_dataset_rank == 3
    assert dense.coefficient_axis_order == (
        "encoded_field",
        "source",
        "real_imag",
    )
    assert dense.real_imag_pair_size == 2
    assert dense.real_imag_order == ("real", "imag")
    assert dense.storage_kinds == ("integer", "floating")
    assert dense.native_complex_storage_allowed is False
    assert dense.all_dimensions_non_empty is True

    file_rules = catalog.lookup_authoring_rule("file_references")
    assert isinstance(file_rules, fs.FileReferenceAuthoringRules)
    assert file_rules.relative_paths_resolve_from == "project_root"
    assert file_rules.missing_project_local_severity == "error"
    assert file_rules.existing_external_files_validate_locally is True
    assert file_rules.remote_unverified_policy == "slurm-or-explicit-opt-in"
    assert file_rules.explicit_opt_in_parameter == "allow_unverified_remote_files"
    assert file_rules.remote_unverified_requires_absolute_path is True
    assert file_rules.remote_unverified_requires_outside_project is True
    assert file_rules.remote_warning_includes_concrete_path is False
    remote_diagnostic = catalog.explain_validation(file_rules.remote_unverified_code)
    assert remote_diagnostic.severity == file_rules.remote_unverified_severity
    assert remote_diagnostic.code == "files.remote_unverified"

    outputs = catalog.lookup_authoring_rule("outputs")
    surfaces = outputs.model_surfaces
    assert isinstance(surfaces, fs.ModelSurfaceAuthoringRules)
    assert surfaces.named_aliases == ("top",)
    assert surfaces.authored_and_expanded_horizon_names is True
    assert surfaces.indexed_prefix == "surface_"
    assert surfaces.index_base == 1
    assert surfaces.case_sensitive is False
    assert surfaces.bottom_alias_allowed is False
    assert surfaces.borehole_surface_names_allowed is False


def test_validation_explanations_and_vetted_examples_are_actionable():
    catalog = fs.load_simulation_knowledge()

    explanation = catalog.explain_validation("field.unsupported")
    assert explanation.severity == "error"
    assert explanation.path
    assert explanation.explanation
    assert explanation.remediation

    example = catalog.get_example("quickstart-2d-acoustic")
    assert example.source_path == "docs/source/quickstart.rst"
    assert "tests/test_simulation_knowledge.py" in example.tested_by
    assert example.scenario_id == "known-small-2d-acoustic"

    with pytest.raises(KeyError):
        catalog.explain_validation("unknown.code")
    with pytest.raises(KeyError):
        catalog.get_example("unknown-example")


def test_package_validation_reports_enforce_the_catalog_registry():
    catalog = fs.load_simulation_knowledge()
    report = fs.ValidationReport.for_package_validators()

    for entry in catalog.validation_codes:
        report.add(
            entry.severity,
            entry.code,
            entry.explanation,
            path=entry.path,
            hint=entry.remediation,
        )
        assert catalog.explain_validation(entry.code) == entry

    assert {issue.code for issue in report.issues} == {
        entry.code for entry in catalog.validation_codes
    }
    with pytest.raises(RuntimeError, match="uncataloged diagnostic code"):
        report.error("package.validator.new_code", "Not cataloged yet.")
    with pytest.raises(RuntimeError, match="catalog declares 'error'"):
        report.warning("simulation.name.missing", "Wrong severity.")

    package_report = fs.validate_job(object())
    assert package_report.errors[0].code == "job.simulation.missing"
    with pytest.raises(RuntimeError, match="uncataloged diagnostic code"):
        package_report.error("package.validator.new_code", "Not cataloged yet.")

    custom_report = fs.ValidationReport()
    custom_report.warning("application.custom", "Application-defined diagnostic.")
    assert custom_report.warnings[0].code == "application.custom"


def test_vetted_example_sources_and_tests_exist():
    catalog = fs.load_simulation_knowledge()
    repository_root = Path(__file__).resolve().parents[1]

    for example in catalog.examples:
        assert (repository_root / example.source_path).is_file()
        for test_path in example.tested_by:
            assert (repository_root / test_path).is_file()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["physics"][0].update(
                {"guided_scenario_id": "missing-scenario"}
            ),
            "unknown guided scenario",
        ),
        (
            lambda payload: payload["validation_codes"].append(
                copy.deepcopy(payload["validation_codes"][0])
            ),
            "validation codes must be unique",
        ),
        (
            lambda payload: payload["public_api"][0]["related_glossary"].append(
                "missing-glossary"
            ),
            "references unknown glossary entries",
        ),
        (
            lambda payload: payload["glossary"][0]["related_api"].append(
                "missing-public-api"
            ),
            "references unknown public API entries",
        ),
        (
            lambda payload: payload["authoring_rules"]["acquisition"][
                "hdf5_dense_source_encoding"
            ].update({"coefficient_dataset_rank": 2}),
            "paired real/imag layout",
        ),
        (
            lambda payload: payload["authoring_rules"]["file_references"].update(
                {"remote_unverified_requires_outside_project": False}
            ),
            "local/remote validation policy",
        ),
        (
            lambda payload: payload["authoring_rules"]["outputs"][
                "model_surfaces"
            ].update({"bottom_alias_allowed": True}),
            "solver-visible horizons",
        ),
        (
            lambda payload: payload["physics"][0]["aliases"].append("sound"),
            "must match physics_aliases",
        ),
        (
            lambda payload: payload["authoring_rules"]["solver"]["defaults"].update(
                {"precision": "half"}
            ),
            "default solver settings are invalid",
        ),
        (
            lambda payload: payload["physics"][0].update(
                {"required_properties": ["vp"]}
            ),
            "material requirements for 'acoustic'",
        ),
        (
            lambda payload: payload["contracts"][0].update({"owner": "FrequenSolve"}),
            "pinned Sauce",
        ),
        (
            lambda payload: payload["starter_scenarios"][0]["setup"][
                "simulation"
            ].update({"physics": "elastic"}),
            "simulation.physics must match",
        ),
        (
            lambda payload: payload["starter_scenarios"][0]["setup"][
                "simulation"
            ].update({"dimension": 3}),
            "simulation.dimension must match",
        ),
        (
            lambda payload: payload["starter_scenarios"][0].update(
                {"physics": "acoustic_axisym", "dimension": 3}
            ),
            "unsupported for physics",
        ),
        (
            lambda payload: payload["starter_scenarios"][0].update(
                {"example_id": "saved-project-job-workflow"}
            ),
            "must reference each other",
        ),
        (
            lambda payload: payload["starter_scenarios"][0]["setup"]["job"].update(
                {"type": "TimeDomainJob"}
            ),
            "job.type must be 'FrequencyDomainJob'",
        ),
        (
            lambda payload: payload["starter_scenarios"][0]["setup"]["job"].pop(
                "f_list"
            ),
            "setup.job is missing keys: f_list",
        ),
        (
            lambda payload: payload["starter_scenarios"][0]["setup"]["model"].update(
                {"dimension": 3}
            ),
            "model.dimension must match",
        ),
    ],
)
def test_runtime_model_rejects_invalid_catalog_entries(mutate, message):
    payload = _resource_payload(fs.CATALOG_RESOURCE)
    mutate(payload)

    with pytest.raises(fs.CatalogValidationError, match=message):
        fs.SimulationKnowledgeCatalog.from_mapping(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["starter_scenarios"][0]["setup"]["simulation"].pop(
            "physics"
        ),
        lambda payload: payload["starter_scenarios"][0]["setup"]["job"].update(
            {"type": "TimeDomainJob"}
        ),
        lambda payload: payload["starter_scenarios"][0]["setup"]["job"].pop("outputs"),
        lambda payload: payload["starter_scenarios"][0]["setup"]["model"].update(
            {"type": "UncatalogedModel"}
        ),
        lambda payload: payload["public_api"][0].update(
            {"import_path": "frequensolve.project.Project"}
        ),
        lambda payload: payload["glossary"][0].update({"related_api": []}),
        lambda payload: payload["authoring_rules"]["acquisition"][
            "hdf5_dense_source_encoding"
        ].update({"native_complex_storage_allowed": True}),
        lambda payload: payload["authoring_rules"]["outputs"]["model_surfaces"].update(
            {"named_aliases": ["top", "bottom"]}
        ),
    ],
)
def test_json_schema_rejects_unsupported_starter_structure(mutate):
    schema = _resource_payload(fs.CATALOG_SCHEMA_RESOURCE)
    payload = _resource_payload(fs.CATALOG_RESOURCE)
    mutate(payload)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, schema)


def test_known_small_acoustic_scenario_builds_valid_contracts(tmp_path):
    catalog = fs.load_simulation_knowledge()
    scenario = catalog.get_starter_scenario()
    required_properties = set(
        catalog.lookup_physics(scenario.physics).required_properties
    )
    assert all(
        required_properties.issubset(
            fs.canonical_property_name(name) for name in layer["properties"]
        )
        for layer in scenario.setup["model"]["layers"]
    )
    project, job = _build_starter_job(scenario, tmp_path / "catalog-project")

    report = fs.validate_job(job)
    assert report.ok
    assert report.issues == []

    project_file = project.save()
    job_file = job.save()
    simulation_file = job.simulation._file

    assert project_file.exists()
    assert json.loads(simulation_file.read_text())["schema"] == "fs-simulation-1"
    assert json.loads(job_file.read_text())["schema"] == "fs-job-1"
    assert (
        job.simulation.acquisition.to_fs(job.simulation.export_context())["schema"]
        == "fs-acquisition-2"
    )
    assert fs.BaseJob.load(job_file).__class__ is fs.FrequencyDomainJob
