"""Freeze point-attribute confidence into native, frequency-specific weights."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

from ._backend import fingerprint
from .misfit import Preprocess


def point_weights(
    keys, simulated, floor, *, pool=None, relative_floor=None, attribute_pool=None
):
    """Return source/component/receiver weights from native cache coordinates."""
    keys = np.asarray(keys, dtype=np.int64) - 1
    values = np.asarray(simulated, dtype=np.float64)
    floor = np.asarray(floor, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite confidence prediction")
    if keys.ndim != 2 or keys.shape[1] != 3 or not len(keys):
        raise ValueError("Invalid confidence coordinate keys")
    if np.any(keys < 0) or len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("Confidence coordinate keys must be unique and positive")
    source, receiver, component = keys.T
    if values.shape != (len(keys), 2) or floor.shape != (
        component.max() + 1,
        source.max() + 1,
    ):
        raise ValueError("Confidence cache shapes disagree")
    power = np.sum(values**2, axis=1)
    shape = (source.max() + 1, component.max() + 1, receiver.max() + 1)
    dense_power = np.zeros(shape)
    dense_power[source, component, receiver] = power
    if attribute_pool is not None:
        if len(keys) != np.prod(shape):
            raise ValueError("Array confidence requires a complete receiver layout")
        dense_power = attribute_pool.apply(
            dense_power.reshape(-1, shape[-1]).T
        ).T.reshape(shape)
        power = dense_power[source, component, receiver]
    if pool is not None:
        if (
            relative_floor is None
            or not np.isfinite(relative_floor)
            or relative_floor <= 0
        ):
            raise ValueError("Local confidence requires a positive relative floor")
        envelope = pool.apply(dense_power.reshape(-1, shape[-1]).T).T.reshape(shape)
        epsilon2 = relative_floor**2 * envelope[source, component, receiver]
    else:
        epsilon2 = floor[component, source] ** 2
    denominator = power + epsilon2
    weight = np.divide(
        power, denominator, out=np.zeros_like(power), where=denominator > 0
    )
    if (
        not np.isfinite(weight).all()
        or not np.isfinite(floor).all()
        or np.any(floor < 0)
    ):
        raise ValueError("Nonfinite or negative confidence cache values")
    dense = np.zeros((source.max() + 1, component.max() + 1, receiver.max() + 1))
    dense[source, component, receiver] = weight
    return dense


def freeze(problem, v=None, *, relative_floor=None, window=None):
    if relative_floor is not None and (
        not np.isfinite(relative_floor) or relative_floor <= 0
    ):
        raise ValueError("Confidence relative_floor must be positive and finite")
    if window is not None and relative_floor is None:
        relative_floor = 0.1
    terms = problem.misfit.objective_terms([g.name for g in problem.observed_groups])
    selected = [i for i, t in enumerate(terms) if t.comparison.reliability is not None]
    if not selected:
        return problem
    if problem.patches is not None or problem.kernel_derivative is not None:
        raise ValueError(
            "Frozen confidence currently requires an unwindowed global problem"
        )
    # Keep the same comparison, stabilization, projections and preprocessing;
    # only remove the model-dependent weight while preparing its snapshot.
    raw_misfit = copy.copy(problem.misfit)
    raw_misfit._terms = tuple(
        replace(t, comparison=replace(t.comparison, reliability=None)) for t in terms
    )
    raw = problem.with_misfit(raw_misfit)
    pools, attribute_pools = {}, {}
    if window is not None or any(
        terms[i].comparison.array is not None for i in selected
    ):
        from ._synthetic_comparison import receiver_coordinates

        coordinates = receiver_coordinates(raw)
        if window is not None:
            pools = {i: window.bind(coordinates[terms[i].id]) for i in selected}
        attribute_pools = {
            i: terms[i].comparison.array.bind(coordinates[terms[i].id])
            for i in selected
            if terms[i].comparison.array is not None
        }
    reference = raw.linearize(v, gradient=False)
    weights = {i: [] for i in selected}
    for task, frequency in enumerate(reference.frequencies, 1):
        manifest = reference.job.state_file(task)
        # The native manifest owns the cache filename (do not guess a job path).
        metadata = json.loads(manifest.read_text())
        by_id = {t["id"]: t for t in metadata["terms"]}
        for i in selected:
            descriptor = by_id[terms[i].id]["cache"]
            cache = Path(descriptor["file"])
            if not cache.is_absolute():
                cache = manifest.parent / cache
            with h5py.File(cache) as h:
                group = h[descriptor["group"]]
                floor = group["phase_floor"][:].astype(float)
                if relative_floor is not None and window is None:
                    floor *= (
                        relative_floor / terms[i].comparison.relative_amplitude_floor
                    )
                weights[i].append(
                    point_weights(
                        group["coordinate_keys"][:],
                        group["simulated"][:],
                        floor,
                        pool=pools.get(i),
                        relative_floor=relative_floor,
                        attribute_pool=attribute_pools.get(i),
                    )
                )
    identity = fingerprint(frequencies=reference.frequencies, weights=weights)
    path = problem.workdir / "reliability" / (identity.split(":")[-1] + ".h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".tmp.h5")
        with h5py.File(temporary, "w") as h:
            h.attrs["reference_job"] = str(reference.job.job_file)
            h.attrs["policy"] = (
                "predicted power / (predicted power + relative_floor squared * local predicted power); frozen; power uses each term's attribute array"
                if window is not None
                else "predicted power / (predicted power + observed reference floor squared); frozen"
            )
            h.attrs["window"] = json.dumps(None if window is None else window.to_dict())
            h.attrs["attribute_arrays"] = json.dumps(
                {str(i): terms[i].comparison.array.to_dict() for i in attribute_pools}
            )
            h.attrs["relative_floor"] = -1 if relative_floor is None else relative_floor
            h.create_dataset("reference_model", data=reference.point.values)
            for i in selected:
                for task, weight in enumerate(weights[i]):
                    h.create_dataset(f"terms/{i}/frequency_{task}", data=weight)
        temporary.replace(path)
    frozen_terms = list(raw_misfit._terms)
    for i in selected:
        rows = [
            {
                "frequency_hz": complex(f).real,
                "laplace_hz": complex(f).imag,
                "weights": {
                    "_type": "HDF5Dense",
                    "file": str(path),
                    "dataset": f"terms/{i}/frequency_{task}",
                },
            }
            for task, f in enumerate(reference.frequencies)
        ]
        hook = Preprocess(
            "trace_weight",
            "residual",
            {
                "layout": "source_component_receiver",
                "frequency_weights": rows,
            },
            name="frozen_predicted_confidence",
        )
        frozen_terms[i] = replace(
            frozen_terms[i], preprocess=(*terms[i].preprocess, hook)
        )
    frozen_misfit = copy.copy(raw_misfit)
    frozen_misfit._terms = tuple(frozen_terms)
    result = problem.with_misfit(frozen_misfit)
    result.reliability_snapshot = path
    return result
