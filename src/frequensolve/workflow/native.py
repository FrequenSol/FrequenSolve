"""Bounded native receiver snapshots with explicit pressure units and survey identity."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import _json_array


def pressure_snapshot(
    payload: bytes, metadata: bytes, row: dict, evidence: dict
) -> dict:
    import h5py

    if (
        len(payload) + len(metadata) > 16 * 1024 * 1024
        or (row.get("scientific") or {}).get("converged") is not True
    ):
        raise ValueError("Only bounded, converged native traces are supported")

    def dataset(file: Any, path: str) -> Any:
        group = file
        for part in path.strip("/").split("/"):
            if not isinstance(group.get(part, getlink=True), h5py.HardLink):
                raise ValueError("Native analysis does not follow HDF5 links")
            group = group[part]
        if (
            not isinstance(group, h5py.Dataset)
            or group.is_virtual
            or group.external
            or group.size > 4096
        ):
            raise ValueError("Unsupported native dataset or size")
        return group

    def strings(value: Any) -> list[str]:
        return [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in np.asarray(value).ravel()
        ]

    def scalar_attr(node: Any, name: str) -> str:
        values = strings(node.attrs.get(name, []))
        if len(values) != 1:
            raise ValueError("Native value representation is incomplete")
        return values[0]

    with tempfile.TemporaryDirectory(prefix="fs-pressure-") as temp:
        a, b = Path(temp) / "payload.h5", Path(temp) / "metadata.h5"
        a.write_bytes(payload)
        b.write_bytes(metadata)
        with h5py.File(a, "r") as data, h5py.File(b, "r") as survey:
            groups = strings(
                dataset(survey, "survey/receiver_groups/_catalog/group_name")[()]
            )
            if len(groups) != 1 or groups[0] != "reciprocal":
                raise ValueError(
                    "The pilot supports the qualified reciprocal receiver group"
                )
            group = groups[0]
            trace = dataset(data, group)
            values = np.asarray(trace[()])
            if values.shape != (2, 1, 2, 2) or not np.issubdtype(
                values.dtype, np.floating
            ):
                raise ValueError(
                    "Native traces differ from the qualified 3D acquisition"
                )
            if "value_convention_representation" in trace.attrs:
                prefix_attr = "value_convention_"
                expected_representation = "physical"
                units_attr = "primal_units"
                amplitude_scale_attr = "rhs_amplitude_scale_dataset"
                amplitude_scale_dataset = group + "_rhs_amplitude_scale"
            else:
                prefix_attr = "value_frame_"
                expected_representation = "coordinate"
                units_attr = "coordinate_units"
                amplitude_scale_attr = "rhs_norm_dataset"
                amplitude_scale_dataset = group + "_rhs_norm"
            if (
                scalar_attr(trace, prefix_attr + "representation")
                != expected_representation
                or scalar_attr(trace, prefix_attr + units_attr) != "Pa"
                or scalar_attr(trace, prefix_attr + "components") != "p"
                or scalar_attr(trace, prefix_attr + "roles") != "primal"
            ):
                raise ValueError(
                    "Pressure units or value representation are unsupported; no implicit conversion is performed"
                )
            frequency, laplace = map(float, row["frequency_value"])
            if (
                not np.isclose(
                    dataset(data, "frequency")[()], frequency, rtol=1e-12, atol=1e-10
                )
                or not np.isclose(
                    dataset(data, "laplace")[()], laplace, rtol=1e-12, atol=1e-10
                )
                or int(dataset(data, "task_id")[()]) != row["frequency_index"] + 1
            ):
                raise ValueError("Native task/frequency identity changed")
            for key in (
                "fs_solver_version",
                "fs_build_id",
                "fs_physics",
                "fs_dimension",
                "fs_workflow",
            ):
                if (
                    data.attrs.get(key) != survey.attrs.get(key)
                    or key not in data.attrs
                ):
                    raise ValueError("Native solver/survey identity differs")
            if any(
                strings(data.attrs[key]) != [expected]
                for key, expected in (
                    ("fs_dimension", "3"),
                    ("fs_physics", "acoustic"),
                    ("fs_workflow", "forward"),
                )
            ):
                raise ValueError(
                    "Native results differ from the qualified acoustic workflow"
                )
            prefix = f"survey/receiver_groups/{group}/receivers/"
            receivers = dataset(survey, prefix + "coordinates")
            sources = dataset(survey, "survey/source_encoding/reference_coordinates")
            if (
                scalar_attr(receivers, "units") != "m"
                or scalar_attr(sources, "units") != "m"
            ):
                raise ValueError("Survey coordinate units are unsupported")
            source_ids = dataset(survey, "survey/source_encoding/field_id")[()].tolist()
            receiver_ids = dataset(survey, prefix + "receiver_id")[()].tolist()
            scales = dataset(survey, "survey/source_encoding/source_scale")[()].tolist()
            encoding = {}
            for name in ("counts", "encoding_kind", "source_scale"):
                value = dataset(survey, "survey/source_encoding/" + name)[()]
                encoding[name] = (
                    strings(value)
                    if np.asarray(value).dtype.kind in "SO"
                    else value.tolist()
                )
            if (
                scalar_attr(trace, prefix_attr + amplitude_scale_attr)
                != "/" + amplitude_scale_dataset
            ):
                raise ValueError("Native source normalization reference is unsupported")
            encoding["rhs_normalization"] = dataset(data, amplitude_scale_dataset)[
                ()
            ].tolist()
            if encoding["encoding_kind"] != ["identity"] or encoding["counts"] != [
                2,
                2,
            ]:
                raise ValueError(
                    "Native source encoding differs from the qualified acquisition"
                )
            attrs = {
                key: strings(value)
                for key, value in trace.attrs.items()
                if key.startswith(prefix_attr)
            }
            complex_values = values[:, 0, :, 0] + 1j * values[:, 0, :, 1]
            result = {
                "version": "pressure-snapshot.v1",
                "real": _json_array(complex_values.real),
                "imaginary": _json_array(complex_values.imag),
                "metadata": {
                    "source_ids": source_ids,
                    "source_coordinates": sources[()].T.tolist(),
                    "receiver_ids": receiver_ids,
                    "receiver_coordinates": receivers[()].T.tolist(),
                    "coordinate_units": "m",
                    "component": "pressure",
                    "units": "Pa",
                    "frequency": frequency,
                    "laplace": laplace,
                    "source_scale": scales,
                    "source_encoding": encoding,
                    "selection": {
                        "group": group,
                        "component": "p",
                        "sources": "all",
                        "receivers": "all",
                    },
                    "value_frame": attrs,
                    "solver_version": strings(data.attrs["fs_solver_version"])[0],
                },
                "evidence": {
                    **evidence,
                    "converged": True,
                    "value_representation": "producer_declared_"
                    + expected_representation,
                    "scientific_qualification": "curated 3D acoustic case; no general physical-validity claim",
                },
            }
            from .analysis import analyze_pressure

            analyze_pressure(result)
            return result
