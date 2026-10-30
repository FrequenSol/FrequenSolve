"""Selected native covariance columns verify shared cross-parameter probes."""

import json
from copy import deepcopy

import h5py
import numpy as np
import pytest

import frequensolve as fs
from frequensolve import imaging as im
from frequensolve.orchestrator.sites.local import LocalSite
from tests.test_imaging_integration import _executable, _simulation


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["array", "reliable", "tukey"])
@pytest.mark.parametrize("probes", [1, 4])
def test_receiver_blocks_match_selected_columns(tmp_path, mode, probes):
    """One receiver per shot removes probe variance, including cross products."""
    project = fs.Project(name="receiver_blocks", path=tmp_path / "project")
    reliable = mode != "array"
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
        simulation += fs.SolverConfig(
            solve_on="final",
            max_iter=300,
            tolerance=1e-4,
            precision="single",
            mumps_precision="single",
        )
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
            site=site,
            controls=im.ControlSpace(
                **{
                    key: im.MeshParameters(
                        key,
                        "layer_2",
                        frequency=2,
                        epw=1,
                        artifact=str(tmp_path / "material.h5"),
                        transform="identity",
                    )
                    for key in ("vp", "rho")
                }
            ),
            observed=im.ObservedData(observed, derivatives={"df": observed}),
            misfit=im.Misfit.student_t(
                nu=2,
                c2=0.01,
                comparison=im.Comparison.phase_derivative(
                    array=(
                        None
                        if reliable
                        else im.SyntheticArray.gaussian(sigma=75 * fs.ureg.m)
                    ),
                    reliability="predicted" if reliable else None,
                    relative_amplitude_floor=0.2 if reliable else 0.005,
                    confidence_floor=0 if reliable else 0.05,
                ),
                normalization=im.Normalization.explicit(4),
                preprocess=[
                    im.Preprocess.trace_weight(
                        np.array([1, 0.7, 0.4, 0.2])[:, None, None]
                    )
                ],
                include_default_preprocess=False,
            ),
            min_support=0,
        )
        if mode == "tukey":
            problem.misfit.loss = im.Loss.tukey(delta=0.006)
        options = {
            "controls": ["vp", "rho"],
            "probes": probes,
            "source_samples": 4,
            "seed": 17,
        }
        lin = problem.linearize(receiver_blocks=options)
        assert problem.linearize(receiver_blocks=options) is lin
        if reliable:
            # Check the stored weight itself, independently of the JVP/probe pair.
            with h5py.File(
                lin.job.state_file(1).with_name("state_1_cache.h5")
            ) as cache:
                term = cache["terms/0"]
                keys = np.asarray(term["coordinate_keys"])
                predicted = np.asarray(term["simulated"])
                power = np.sum(predicted**2, axis=1)
                floor = np.asarray(term["phase_floor"])
                floor2 = floor[keys[:, 2] - 1, keys[:, 0] - 1] ** 2
                reliability = power / (power + floor2)
                expected_weight = (
                    np.array([1, 0.7, 0.4, 0.2])[keys[:, 0] - 1] * reliability
                )
                np.testing.assert_allclose(
                    term["static_weight"], expected_weight, rtol=1e-6
                )
                assert np.min(reliability) < 0.99
                robust_weight = np.asarray(term["robust_weight"])
        if mode == "tukey":
            student = deepcopy(problem.misfit)
            student.loss = im.Loss.student_t(nu=2, c2=0.01)
            baseline = problem.with_misfit(student).linearize(
                receiver_diagonal={"probes": 1}
            )
            with h5py.File(
                baseline.job.state_file(1).with_name("state_1_cache.h5")
            ) as cache:
                student_weight = np.asarray(cache["terms/0/robust_weight"])
            q = np.maximum(3 / student_weight - 0.02, 0)
            expected = np.maximum(1 - q / 0.006**2, 0) ** 2
            np.testing.assert_allclose(robust_weight, expected, atol=3e-4)
            assert np.any(robust_weight == 0) and np.any(robust_weight > 0)
        blocks = lin.receiver_blocks
        diagonal = lin.receiver_diagonal.to_file(native=True)
        for i, name in enumerate(blocks.controls):
            np.testing.assert_array_equal(
                blocks.matrices[:, i, i], diagonal.blocks[name]
            )
        assert np.linalg.eigvalsh(blocks.matrices).min() >= -1e-12
        # Select the most illuminated coefficient; one column for each parameter.
        node = int(np.argmax(np.trace(blocks.matrices, axis1=1, axis2=2)))
        directions = []
        columns = []
        for name in blocks.controls:
            direction = lin.space.zeros()
            direction.values[lin.space.slices["model." + name].start + node] = 1
            directions.append(direction)
            columns.append(lin.jvp(direction).values)
        columns = np.asarray(columns)
        exact = (columns.conj() @ columns.T).real
        np.testing.assert_allclose(blocks.matrices[node], exact, rtol=3e-3, atol=1e-10)
        sampled = problem.linearize(receiver_blocks={**options, "source_samples": 2})
        record = json.loads(
            (
                sampled.job.covector_file(1).parent
                / "_fs_run/tasks/task_000001/result.json"
            ).read_text()
        )
        selected = [record["misc"][f"receiver_probe_source_{i}"] - 1 for i in (1, 2)]
        expected = 2 * (columns[:, selected].conj() @ columns[:, selected].T).real
        np.testing.assert_allclose(
            sampled.receiver_blocks.matrices[node], expected, rtol=3e-3, atol=1e-10
        )
        np.testing.assert_allclose(
            sampled.gradient.values, lin.gradient.values, rtol=1e-6, atol=1e-12
        )
