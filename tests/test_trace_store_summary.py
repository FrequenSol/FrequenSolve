import h5py
import numpy as np
import pytest

from frequensolve.seismic.trace_store import TraceStore


def test_trace_summary_reports_group_with_no_matching_frequencies(tmp_path):
    trace_file = tmp_path / "traces.h5"
    with h5py.File(trace_file, "w") as h5:
        h5.create_dataset("frequency", data=np.array([30.0]))
        traces = h5.create_dataset(
            "surface",
            data=np.zeros((2, 1, 1), dtype=np.float32),
        )
        traces.attrs["dims"] = ["receiver", "component", "shot"]
        traces.attrs["receiver"] = np.array([1, 2])
        traces.attrs["component"] = np.array(["p"], dtype="S")
        traces.attrs["shot"] = np.array([1])

    store = TraceStore(
        metadata={"groups": ["surface"], "f_map": {1: 10.0}},
        files=[trace_file],
    )
    store._consolidated = trace_file

    with pytest.raises(
        ValueError,
        match=(
            r"Cannot summarize trace group 'surface': no frequencies are available"
            r".*Expected frequencies from the job metadata: \[10\.0\] Hz"
            r".*inspect the job logs and fetched trace artifacts"
        ),
    ):
        store.format_summary()
