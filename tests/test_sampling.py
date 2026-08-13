import numpy as np

from frequensolve.simulation.sampling import UniformSweepSampling


def test_uniform_sweep_sampling_values_and_cutoff_contract():
    sampling = UniformSweepSampling(
        f_min=1.0,
        f_max=4.0,
        df=1.0,
        t_shift=0.25,
        upscale=2,
    )

    assert sampling.T == 1.0
    assert sampling.nfreq == 5
    assert sampling.ntime == 8
    assert sampling.nFreq == 9
    assert sampling.nTime == 16
    assert np.array_equal(sampling.f_list, np.linspace(0.0, 4.0, 5))
    assert np.array_equal(sampling.F_list, np.linspace(0.0, 8.0, 9))

    n_samples, final_time = sampling.cutoff(0.25)

    assert isinstance(n_samples, int)
    assert isinstance(final_time, float)
    assert (n_samples, final_time) == (9, 0.3125)


def test_uniform_sweep_sampling_serialization_roundtrip():
    sampling = UniformSweepSampling(f_min=1.0, f_max=4.0, df=1.0, upscale=2)

    payload = sampling.to_fs()
    restored = UniformSweepSampling.from_fs(payload)

    assert restored == sampling
