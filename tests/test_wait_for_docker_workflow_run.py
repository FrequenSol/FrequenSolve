from __future__ import annotations

import pytest

from scripts.wait_for_docker_workflow_run import (
    completion_state,
    wait_for_completion,
)


def test_waits_across_pending_states_until_success():
    runs = iter(
        [
            {"status": "queued", "conclusion": None},
            {"status": "in_progress", "conclusion": None},
            {"status": "completed", "conclusion": "success"},
        ]
    )
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    assert wait_for_completion(
        lambda: next(runs),
        timeout_seconds=30,
        interval_seconds=5,
        monotonic=lambda: now[0],
        sleep=sleep,
    )
    assert now[0] == 10


def test_returns_pending_when_bounded_window_expires():
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    assert not wait_for_completion(
        lambda: {"status": "queued", "conclusion": None},
        timeout_seconds=10,
        interval_seconds=6,
        monotonic=lambda: now[0],
        sleep=sleep,
    )
    assert now[0] == 10


@pytest.mark.parametrize("conclusion", ["cancelled", "failure", "timed_out"])
def test_rejects_unsuccessful_completion(conclusion):
    with pytest.raises(ValueError, match="completed unsuccessfully"):
        completion_state({"status": "completed", "conclusion": conclusion})


def test_rejects_unknown_or_inconsistent_state():
    with pytest.raises(ValueError, match="unexpected state"):
        completion_state({"status": "in_progress", "conclusion": "success"})
