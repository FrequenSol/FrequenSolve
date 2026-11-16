"""Native robust-array diagonal regression at a deterministic one-row boundary."""

import json

import numpy as np
import pytest

import frequensolve as fs
from frequensolve import imaging as im
from frequensolve.orchestrator.sites.local import LocalSite
from tests.test_imaging_integration import _executable, _simulation


@pytest.mark.integration
def test_array_diagonal_matches_columns_without_probe_variance(tmp_path):
    """One receiver per shot makes one Rademacher probe an exact diagonal."""
    project = fs.Project(name="array_diagonal", path=tmp_path / "project")
    truth = _simulation(project, "truth", 2.5)
    initial = _simulation(project, "initial", 2.2)
    for simulation in (truth, initial):
        acquisition = fs.Acquisition()
        acquisition.add_sources(
            kind="scalar", coords=[[0.2, 0.08], [0.4, 0.08], [0.6, 0.08], [0.8, 0.08]]
        )
        receiver = fs.ReceiverNode(name="hydrophone")
        receiver.add_component(name="p", field="pressure")
        acquisition.add_receiver_group("surface", receiver, [[0.45, 0.05]])
        simulation.acquisition = acquisition
        simulation.save()
    with LocalSite(solver=_executable(), n_workers=1, threads_per_worker=1) as site:
        observed = fs.FrequencyDomainJob(
            "observed",
            truth,
            [4 - 0.1j],
            phase_derivatives=1,
            source_derivative="frozen",
        )
        site.run(observed, check=True)
        problem = im.ImagingProblem(
            initial,
            controls=im.ControlSpace(
                vp=im.DepthProfile("vp", "layer_2", count=4, transform="identity")
            ),
            observed=im.ObservedData(observed, derivatives={"df": observed}),
            site=site,
            misfit=im.Misfit.student_t(
                nu=2,
                c2=0.01,
                comparison=im.Comparison.phase_derivative(
                    array=im.SyntheticArray.gaussian(sigma=75 * fs.ureg.m),
                    relative_amplitude_floor=0.005,
                    confidence_floor=0.05,
                ),
                normalization=im.Normalization.explicit(4),
                preprocess=[
                    im.Preprocess.trace_weight(
                        np.array([1, 0.7, 0.4, 0.2])[:, None, None]
                    )
                ],
                include_default_preprocess=False,
            ),
        )
        lin = problem.linearize(
            receiver_diagonal={"probes": 1, "source_samples": 4, "seed": 17}
        )
        expected = []
        columns = []
        for direction in np.eye(lin.space.size):
            column = lin.jvp(direction)
            expected.append(column.norm() ** 2)
            columns.append(column.values)
        np.testing.assert_allclose(
            lin.receiver_diagonal.values, expected, rtol=2e-3, atol=1e-10
        )

        sampled = problem.linearize(
            receiver_diagonal={"probes": 1, "source_samples": 2, "seed": 17}
        )
        record = json.loads(
            (
                sampled.job.covector_file(1).parent
                / "_fs_run/tasks/task_000001/result.json"
            ).read_text()
        )
        selected = [record["misc"][f"receiver_probe_source_{i}"] - 1 for i in (1, 2)]
        assert 0 <= selected[0] < 2 <= selected[1] < 4
        expected_sampled = 2 * np.sum(
            np.abs(np.asarray(columns)[:, selected]) ** 2, axis=1
        )
        np.testing.assert_allclose(
            sampled.receiver_diagonal.values, expected_sampled, rtol=2e-3, atol=1e-10
        )
        np.testing.assert_allclose(
            sampled.gradient.values, lin.gradient.values, rtol=1e-6, atol=1e-12
        )
