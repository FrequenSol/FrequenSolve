"""Standalone exports of explicitly indexed native trace segments."""

from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np

from .trace_store import TraceStore


def _filter_survey(target, groups):
    """Keep the selected receiver groups and matching rows of the survey catalog."""
    base = "survey/receiver_groups"
    if base not in target:
        return
    catalog_path = base + "/_catalog"
    if catalog_path in target:
        catalog = target[catalog_path]
        names = catalog["group_name"].asstr()[()].reshape(-1)
        paths = catalog["dataset_path"].asstr()[()].reshape(-1)
        selected = np.array(
            [
                name in groups or path.strip("/").removeprefix("trace_data/") in groups
                for name, path in zip(names, paths)
            ]
        )
        for name in list(catalog):
            dataset = catalog[name]
            if isinstance(dataset, h5py.Dataset) and dataset.shape == names.shape:
                values, attrs, dtype = (
                    dataset[()][selected],
                    dict(dataset.attrs),
                    dataset.dtype,
                )
                del catalog[name]
                catalog.create_dataset(name, data=values, dtype=dtype)
                catalog[name].attrs.update(attrs)
        keep = set(names[selected])
    else:
        keep = set(groups)
    for name in list(target[base]):
        if name != "_catalog" and name not in keep:
            del target[base + "/" + name]


def export_packed(entries, destination, *, groups=None):
    """Copy an exact trace selection into a single native indexed HDF5 file.

    Source segments are immutable. Survey layouts must match, and dataset
    indexes and RHS amplitude scales are preserved through renumbering.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if not entries or any(e["family"] != "traces" for e in entries):
        raise ValueError("Export requires an explicit native traces selection")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with ExitStack() as stack:
            sources = {
                path: stack.enter_context(h5py.File(path, "r"))
                for path in dict.fromkeys(str(e["path"]) for e in entries)
            }
            first = next(iter(sources.values()))
            groups = tuple(groups or TraceStore._h5_trace_groups(first))
            wanted_paths = {
                TraceStore._indexed_source_path_for_group(first, group)
                for group in groups
            }
            for group in groups:
                TraceStore._indexed_trace_rows(first, group)
            survey = {}
            first["survey"].visititems(
                lambda name, obj: (
                    survey.update({name: obj[()]})
                    if isinstance(obj, h5py.Dataset)
                    else None
                )
            )
            for source in sources.values():
                other = {}
                source["survey"].visititems(
                    lambda name, obj: (
                        other.update({name: obj[()]})
                        if isinstance(obj, h5py.Dataset)
                        else None
                    )
                )
                if survey.keys() != other.keys() or any(
                    not np.array_equal(value, other[name])
                    for name, value in survey.items()
                ):
                    raise ValueError("Packed trace surveys differ")
            with h5py.File(temporary, "w") as target:
                first.copy("survey", target)
                _filter_survey(target, groups)
                if "metadata/schema_version" in first:
                    target.require_group("metadata")
                    first.copy("metadata/schema_version", target["metadata"])
                numbers = np.arange(1, len(entries) + 1, dtype=np.int32)
                for name, values in {
                    "frequency": [e["frequency"] for e in entries],
                    "laplace": [e.get("laplace", 0.0) for e in entries],
                    "task_id": numbers,
                }.items():
                    target.create_dataset(name, data=values)
                    target.create_dataset(f"trace_index/{name}", data=values)
                target.create_dataset("trace_index/dataset_number", data=numbers)
                for name in (
                    "layout_kind",
                    "data_root",
                    "schema_version",
                    "dataset_number_format",
                ):
                    first.copy(f"trace_index/{name}", target["trace_index"])
                paths, originals, row_numbers, provenance = [], [], [], []
                for number, entry in zip(numbers, entries):
                    source = sources[str(entry["path"])]
                    old_number = entry["dataset_number"]
                    index = source["trace_index"]
                    selected = np.flatnonzero(index["dataset_number"][()] == old_number)
                    if len(selected) != 1:
                        raise ValueError("Missing or ambiguous packed frequency index")
                    i = selected[0]
                    if float(index["frequency"][i]) != entry["frequency"] or float(
                        index["laplace"][i]
                    ) != entry.get("laplace", 0.0):
                        raise ValueError(
                            "Packed selection frequency differs from source"
                        )
                    rows = np.flatnonzero(
                        index["datasets/dataset_number"][()] == old_number
                    )
                    source_paths = index["datasets/source_path"].asstr()[()]
                    rows = [
                        row
                        for row in rows
                        if "/" + source_paths[row].strip("/") in wanted_paths
                    ]
                    if not len(rows):
                        raise ValueError("Packed selection has no trace datasets")
                    if {
                        "/" + source_paths[row].strip("/") for row in rows
                    } != wanted_paths:
                        raise ValueError(
                            "Packed selection is missing a requested group"
                        )
                    for row in rows:
                        old_path = index["datasets/packed_path"].asstr()[row]
                        original = index["datasets/source_path"].asstr()[row]
                        parent = old_path.rsplit("/", 1)[0]
                        new_path = f"{parent}/{number:06d}"
                        source.copy(
                            old_path, target.require_group(parent), name=f"{number:06d}"
                        )
                        rhs_attribute = "value_convention_rhs_amplitude_scale_dataset"
                        reference = source[old_path].attrs.get(rhs_attribute)
                        scale = old_path + "_rhs_amplitude_scale"
                        if reference is not None:
                            values = np.asarray(reference).reshape(-1)
                            if values.size != 1:
                                raise ValueError("Ambiguous RHS companion reference")
                            scale = values[0]
                            if isinstance(scale, bytes):
                                scale = scale.decode("utf-8")
                            if (
                                not isinstance(scale, str)
                                or not scale
                                or scale not in source
                            ):
                                raise ValueError("Missing RHS companion reference")
                        if scale in source:
                            source.copy(
                                scale,
                                target[parent],
                                name=f"{number:06d}_rhs_amplitude_scale",
                            )
                            if reference is not None:
                                # HDF5 copy preserves attributes verbatim, including
                                # paths into the old numbering. Rebind the reference
                                # to the companion copied into this export.
                                del target[new_path].attrs[rhs_attribute]
                                target[new_path].attrs.create(
                                    rhs_attribute,
                                    np.full(
                                        np.asarray(reference).shape,
                                        new_path + "_rhs_amplitude_scale",
                                        dtype=object,
                                    ),
                                    dtype=h5py.string_dtype("utf-8"),
                                )
                        paths.append(new_path)
                        originals.append(original)
                        row_numbers.append(number)
                    provenance.append(f"{entry['path']}#{old_number}")
                string = h5py.string_dtype("utf-8")
                for name, values in {
                    "datasets/packed_path": paths,
                    "datasets/source_path": originals,
                    "shard_file": provenance,
                }.items():
                    target.create_dataset(
                        f"trace_index/{name}", data=values, dtype=string
                    )
                target.create_dataset(
                    "trace_index/datasets/dataset_number", data=row_numbers
                )
                target.attrs["export_provenance"] = (
                    "TraceDataset.export_packed; immutable source selection"
                )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
