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
