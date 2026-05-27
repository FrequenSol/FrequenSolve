import logging

from frequensolve.orchestrator.sites.base import BaseSite


def test_emit_does_not_info_log_messages_it_prints(capsys, caplog):
    site = BaseSite(verbose=True)

    with caplog.at_level(logging.INFO, logger=site.__class__.__module__):
        site._emit("clean status")

    assert capsys.readouterr().out == "clean status\n"
    assert not [
        record
        for record in caplog.records
        if record.levelno == logging.INFO and record.message == "clean status"
    ]


def test_emit_still_logs_warnings_when_verbose(capsys, caplog):
    site = BaseSite(verbose=True)

    with caplog.at_level(logging.WARNING, logger=site.__class__.__module__):
        site._emit("important warning", level=logging.WARNING)

    assert capsys.readouterr().out == "important warning\n"
    assert any(
        record.levelno == logging.WARNING and record.message == "important warning"
        for record in caplog.records
    )
