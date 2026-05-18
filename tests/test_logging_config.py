import json
import logging

from frequensolve.project import Project


def test_project_configures_package_logging_file(tmp_path):
    log_file = tmp_path / "frequensolve.log"

    Project(
        name="logging",
        path=tmp_path / "project",
        log_level="DEBUG",
        log_file=log_file,
    )

    logger = logging.getLogger("frequensolve.test")
    logger.debug("project logging configured")
    for handler in logging.getLogger("frequensolve").handlers:
        handler.flush()

    assert "project logging configured" in log_file.read_text()


def test_project_quiets_noisy_dependency_loggers_by_default(tmp_path):
    Project(name="logging", path=tmp_path / "project", log_level="DEBUG")

    assert logging.getLogger("frequensolve").level == logging.DEBUG
    assert logging.getLogger("distributed").level == logging.WARNING


def test_project_save_load_preserves_logging_preferences(tmp_path):
    project = Project(
        name="logging",
        path=tmp_path / "project",
        log_level="DEBUG",
        dependency_log_level="ERROR",
    )

    project_file = project.save()
    payload = json.loads(project_file.read_text())
    loaded = Project.load(project_file)

    assert payload["logging"] == {
        "level": "DEBUG",
        "dependency_level": "ERROR",
    }
    assert loaded.log_level == "DEBUG"
    assert loaded.dependency_log_level == "ERROR"
    assert logging.getLogger("frequensolve").level == logging.DEBUG
    assert logging.getLogger("distributed").level == logging.ERROR
