import numpy as np
import pytest
import xarray as xr

from frequensolve.seismic.trace_store import TraceStore


@pytest.mark.parametrize("string_keys", [False, True])
@pytest.mark.parametrize("subset", [False, True])
def test_repeated_real_frequencies_retain_distinct_laplace_samples(string_keys, subset):
    frequencies = {1: 2.5, 2: 3.0, 3: 3.5, 4: 2.5, 5: 3.0, 6: 3.5}
    laplace = {1: -0.5, 2: -0.5, 3: -0.5, 4: -0.15, 5: -0.15, 6: -0.15}
    requested = [4, 5, 6] if subset else list(frequencies)
    key = str if string_keys else int
    store = TraceStore(
        metadata={
            "f_map": {key(k): frequencies[k] for k in requested},
            "laplace_map": {key(k): laplace[k] for k in requested},
            "laplace_map_keys": requested,
        },
        files=[],
    )
    # Deliberately permuted storage; values identify the original samples.
    order = [6, 1, 4, 3, 2, 5]
    data = xr.DataArray(
        order,
        dims="frequency",
        coords={
            "frequency": [frequencies[k] for k in order],
            "laplace": ("frequency", [laplace[k] for k in order]),
        },
    )
    selected = store._filter_expected_frequency_data(data)
    expected = sorted(requested, key=lambda k: frequencies[k])
    np.testing.assert_array_equal(selected.values, expected)
    np.testing.assert_array_equal(selected.laplace, [laplace[k] for k in expected])
