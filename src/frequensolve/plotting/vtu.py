"""PyVista helpers for FrequenSolve VTU outputs."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from frequensolve._optional import optional_dependency_error

__all__ = [
    "read_vtu",
    "vtu_fields",
    "plot_vtu",
    "plot_vtu_slice",
]

_DEFAULT_SCALAR_BAR_ARGS = {
    "vertical": False,
    "position_x": 0.22,
    "position_y": 0.06,
    "width": 0.56,
    "height": 0.08,
}


def _pyvista():
    try:
        import pyvista as pv
    except ModuleNotFoundError as exc:
        raise optional_dependency_error(
            "VTU plotting",
            extra="visual",
            dependencies=("pyvista",),
            error=exc,
        ) from exc
    return pv


def read_vtu(path: str | Path):
    """Read a solver ``.vtu`` file with PyVista.

    PyVista is imported lazily so the core FrequenSolve Python API remains
    usable without VTK.
    """

    path = Path(path)
    if path.suffix.lower() != ".vtu":
        raise ValueError(f"Expected a .vtu file, got {path}")
    mesh = _pyvista().read(path)
    _attach_vtu_metadata(mesh, _read_vtu_metadata(path))
    return mesh


def _as_vtu_mesh(vtu):
    if isinstance(vtu, (str, Path)):
        return read_vtu(vtu)
    return vtu


def _attach_vtu_metadata(mesh, metadata: Mapping[str, Any] | None):
    try:
        mesh._frequensolve_vtu_metadata = dict(metadata or {})
    except Exception:
        pass
    return mesh


def _read_vtu_metadata(path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError:
        return {"point": {}, "cell": {}}

    appended_start = raw.find(b"<AppendedData")
    xml_bytes = raw if appended_start < 0 else raw[:appended_start] + b"</VTKFile>"
    try:
        root = ET.fromstring(xml_bytes.decode("utf-8", errors="ignore"))
    except ET.ParseError:
        return {"point": {}, "cell": {}}

    metadata: dict[str, dict[str, dict[str, Any]]] = {"point": {}, "cell": {}}
    for association, tag in (("point", "PointData"), ("cell", "CellData")):
        for array in root.findall(f".//{tag}/DataArray"):
            name = array.attrib.get("Name")
            if not name:
                continue
            entry = _vtu_array_metadata_from_attrs(array.attrib)
            if entry:
                metadata[association][name] = entry
    return metadata


def _vtu_array_metadata_from_attrs(attrs: Mapping[str, str]) -> dict[str, Any]:
    display_name = _vtu_attr(
        attrs,
        "fs_display_name",
        "fs_pretty_name",
        "pretty_name",
        "display_name",
        "long_name",
    )
    units = _vtu_attr(
        attrs,
        "fs_units",
        "fs_unit",
        "units",
        "unit",
        "unit_label",
    )
    components = _vtu_attr(
        attrs,
        "fs_components",
        "fs_component_names",
        "component_names",
        "components",
    )
    metadata = {}
    if display_name:
        metadata["display_name"] = _format_vtu_display_name(display_name)
    if units:
        metadata["units"] = units
    component_names = _parse_component_names(components)
    if component_names:
        metadata["components"] = component_names
    return metadata


def _parse_component_names(value: str | None) -> list[str]:
    if not value:
        return []
    text = str(value).strip()
    if not text:
        return []
    for sep in (",", ";", "|"):
        text = text.replace(sep, " ")
    return [part.strip() for part in text.split() if part.strip()]


def _vtu_attr(attrs: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = attrs.get(name)
        if value:
            return value
    normalized = {str(key).lower(): value for key, value in attrs.items()}
    for name in names:
        value = normalized.get(name.lower())
        if value:
            return value
    return None


def _format_vtu_display_name(name: str) -> str:
    name = str(name).replace("_", " ").strip()
    if not name:
        return name
    return name[:1].upper() + name[1:]


def _vtu_metadata(mesh) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    return getattr(mesh, "_frequensolve_vtu_metadata", {}) or {}


def _vtu_array_metadata(mesh, association: str, field: str) -> Mapping[str, Any]:
    metadata = _vtu_metadata(mesh)
    if association == "auto":
        return (
            metadata.get("point", {}).get(field)
            or metadata.get("cell", {}).get(field)
            or {}
        )
    return metadata.get(association, {}).get(field, {})


def _vtu_data(mesh, association: str):
    if association == "point":
        return mesh.point_data
    if association == "cell":
        return mesh.cell_data
    raise ValueError("association must be 'auto', 'point', or 'cell'")


def _vtu_field_names(mesh, association: str) -> list[str]:
    if association == "auto":
        names = list(mesh.point_data.keys())
        names.extend(name for name in mesh.cell_data.keys() if name not in names)
        return names
    return list(_vtu_data(mesh, association).keys())


VTU_FIELD_LABELS = {
    "rho": "Rho",
    "vp": "Vp",
    "vs": "Vs",
    "qp": "Qp",
    "qs": "Qs",
}


def vtu_fields(
    vtu,
    *,
    association: str = "auto",
) -> list[str]:
    """Return available point/cell data arrays in a VTU file.

    Args:
        vtu: VTU file path or mesh object accepted by ``plot_vtu`` helpers.
        association: Data association to inspect: ``"auto"``, ``"point"``, or
            ``"cell"``.

    Returns:
        Field names in display order.
    """

    mesh = _as_vtu_mesh(vtu)
    return _vtu_field_names(mesh, association)


def _select_vtu_data(mesh, field: str, association: str):
    if association == "auto":
        if field in mesh.point_data:
            return mesh.point_data, "point"
        if field in mesh.cell_data:
            return mesh.cell_data, "cell"
        available = ", ".join(_vtu_field_names(mesh, "auto")) or "<none>"
        raise KeyError(f"VTU field '{field}' not found. Available fields: {available}")

    data = _vtu_data(mesh, association)
    if field not in data:
        available = ", ".join(data.keys()) or "<none>"
        raise KeyError(
            f"VTU {association} field '{field}' not found. Available fields: {available}"
        )
    return data, association


def _normalize_component_name(component: str) -> str:
    return (
        str(component)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def _component_index(
    component: str | int,
    width: int,
    names: Sequence[str] | None = None,
) -> int:
    if isinstance(component, str):
        key = _normalize_component_name(component)
        if key.isdigit():
            index = int(key)
        else:
            metadata_names = list(names or [])
            metadata_lookup = {
                _normalize_component_name(name): i
                for i, name in enumerate(metadata_names[:width])
            }
            if key in metadata_lookup:
                index = metadata_lookup[key]
            else:
                index = _fallback_component_index(key, width)
    else:
        index = int(component)
    if index < 0 or index >= width:
        raise ValueError(f"component index {index} is outside field width {width}")
    return index


def _fallback_component_index(component: str, width: int) -> int:
    names = _fallback_component_names(width)
    if component in names:
        return names[component]
    accepted = sorted(names)
    raise ValueError(
        "component must be an integer or one of the supported component names "
        f"for field width {width}: {', '.join(accepted)}"
    )


def _fallback_component_names(width: int) -> dict[str, int]:
    aliases: dict[str, int] = {}
    if width == 2:
        aliases.update({"x": 0, "r": 0, "y": 1, "z": 1, "theta": 1, "t": 1})
    elif width >= 3:
        aliases.update({"x": 0, "r": 0, "y": 1, "theta": 1, "t": 1, "z": 2})

    if width == 3:
        aliases.update({"xx": 0, "rr": 0, "zz": 1, "xz": 2, "zx": 2, "rz": 2, "zr": 2})
    elif width == 4:
        aliases.update(
            {
                "rr": 0,
                "xx": 0,
                "zz": 1,
                "rz": 2,
                "zr": 2,
                "xz": 2,
                "zx": 2,
                "tt": 3,
                "thetatheta": 3,
                "yy": 3,
            }
        )
    elif width == 6:
        aliases.update(
            {
                "xx": 0,
                "rr": 0,
                "zz": 1,
                "xz": 2,
                "zx": 2,
                "rz": 2,
                "zr": 2,
                "xy": 3,
                "yx": 3,
                "rtheta": 3,
                "thetar": 3,
                "yy": 4,
                "tt": 4,
                "thetatheta": 4,
                "yz": 5,
                "zy": 5,
                "ztheta": 5,
                "thetaz": 5,
            }
        )
    return aliases


def _validate_vtu_part(part: str) -> str:
    aliases = {
        "re": "real",
        "real": "real",
        "im": "imag",
        "imag": "imag",
        "imaginary": "imag",
        "abs": "abs",
        "mag": "abs",
        "magnitude": "abs",
        "amplitude": "abs",
    }
    part = str(part).lower()
    part = aliases.get(part, part)
    if part not in {"real", "imag", "abs"}:
        raise ValueError("part must be 'real'/'re', 'imag'/'im', or 'abs'/'mag'")
    return part


def _part_array(values: np.ndarray, part: str) -> np.ndarray:
    part = _validate_vtu_part(part)
    if np.iscomplexobj(values):
        if part == "real":
            return values.real
        if part == "imag":
            return values.imag
        return np.abs(values)
    if part == "imag":
        raise ValueError("part='imag' requires a complex field or *_imag VTU array")
    if part == "abs":
        return np.abs(values)
    return values


def _strip_vtu_part_suffix(field: str) -> str:
    for suffix in ("_real", "_imag", "_re", "_im", "_abs", "_mag"):
        if field.endswith(suffix):
            return field[: -len(suffix)]
    return field


def _strip_vtu_index(field: str) -> str:
    head, sep, tail = field.rpartition("_")
    if sep and tail.isdigit():
        return head
    return field


def _vtu_source_index(field: str) -> int | None:
    base = _strip_vtu_part_suffix(field)
    _, sep, tail = base.rpartition("_")
    if sep and tail.isdigit():
        return int(tail)
    return None


def _source_indexed_vtu_base(field: str, source: int | str | None) -> str:
    if source is None:
        return field
    base = _strip_vtu_index(_strip_vtu_part_suffix(field))
    return f"{base}_{int(source)}"


def _vtu_field_key(field: str) -> str:
    return _strip_vtu_index(_strip_vtu_part_suffix(field)).lower()


def _default_vtu_label(field: str) -> str:
    key = _vtu_field_key(field)
    if key in VTU_FIELD_LABELS:
        return VTU_FIELD_LABELS[key]
    return " ".join(part.capitalize() for part in key.split("_") if part)


def _vtu_units(field: str, units: Any) -> str | None:
    key = _vtu_field_key(field)
    if isinstance(units, Mapping):
        value = units.get(field, units.get(key))
        return None if value in {None, ""} else str(value)
    if units is not None:
        return None if units == "" else str(units)
    return None


def _vtu_part_label(part: str) -> str:
    return {
        "real": "Real",
        "imag": "Imaginary",
        "abs": "Magnitude",
    }[part]


def _vtu_label_has_part(label: str | None, part: str) -> bool:
    if not label:
        return False
    text = label.lower()
    aliases = {
        "real": ("real",),
        "imag": ("imaginary", "imag"),
        "abs": ("magnitude", "amplitude", "absolute", "abs"),
    }[part]
    return any(alias in text for alias in aliases)


def _vtu_scalar_label(
    field: str,
    *,
    part: str,
    component: str | int | None,
    include_part: bool,
    label: str | None,
    units: Any,
) -> str:
    base = label or _default_vtu_label(field)
    if component is not None:
        base = f"{base} {str(component).upper()}"
    if include_part:
        base = f"{base} ({_vtu_part_label(part)})"
    unit_label = _vtu_units(field, units)
    if unit_label:
        base = f"{base} [{unit_label}]"
    return base


def _select_vtu_component(
    values: np.ndarray,
    *,
    component: str | int | None,
    metadata: Mapping[str, Any] | None,
) -> np.ndarray:
    values = np.asarray(values)
    if component is not None:
        if values.ndim != 2:
            raise ValueError("component can only be used with vector/tensor VTU fields")
        names = None if metadata is None else metadata.get("components")
        values = values[:, _component_index(component, values.shape[1], names)]
    elif values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    elif values.ndim > 1:
        raise ValueError("component is required for vector/tensor VTU fields")
    return values


def _source_indexed_vtu_part_fields(
    mesh,
    field: str,
    association: str,
    part: str,
    source: int | str | None = None,
) -> list[str]:
    key = _vtu_field_key(field)
    source_index = None if source is None else int(source)
    matches = []
    for name in _vtu_field_names(mesh, association):
        if _vtu_field_key(name) != key:
            continue
        if source_index is not None and _vtu_source_index(name) != source_index:
            continue
        if _resolved_vtu_part(name, "") == part:
            matches.append(name)
    return matches


def _ambiguous_source_indexed_field_error(
    field: str,
    part: str,
    matches: list[str],
) -> KeyError:
    bases = sorted({_strip_vtu_part_suffix(name) for name in matches})
    return KeyError(
        f"VTU field '{field}' is ambiguous for part='{part}'. Matching fields: "
        f"{', '.join(matches)}. Specify one source-indexed field base instead, "
        f"for example '{bases[0]}'."
    )


def _resolve_vtu_scalar(
    mesh,
    field: str,
    *,
    association: str,
    component: str | int | None,
    part: str,
    source: int | str | None,
    label: str | None,
    units: Any,
) -> tuple[str, str]:
    part = _validate_vtu_part(part)
    lookup_field = _source_indexed_vtu_base(field, source)
    candidate_fields = [lookup_field]
    if part == "real":
        candidate_fields.extend([f"{lookup_field}_real", f"{lookup_field}_re"])
    elif part == "imag":
        candidate_fields.extend([f"{lookup_field}_imag", f"{lookup_field}_im"])
    elif part == "abs":
        candidate_fields.extend([f"{lookup_field}_abs", f"{lookup_field}_mag"])

    data = None
    resolved_association = association
    resolved_field = None
    for name in candidate_fields:
        try:
            data, resolved_association = _select_vtu_data(mesh, name, association)
            resolved_field = name
            break
        except KeyError:
            continue
    if data is None or resolved_field is None:
        matches = _source_indexed_vtu_part_fields(
            mesh,
            field,
            association,
            part,
            source=source,
        )
        if len(matches) == 1:
            resolved_field = matches[0]
            data, resolved_association = _select_vtu_data(
                mesh,
                resolved_field,
                association,
            )
        elif len(matches) > 1:
            raise _ambiguous_source_indexed_field_error(field, part, matches)
    if data is None or resolved_field is None:
        if part == "abs":
            data, resolved_association = _paired_abs_vtu_data(
                mesh,
                lookup_field,
                association,
            )
            resolved_field = f"{lookup_field}_abs"
            values = data
            metadata = _vtu_array_metadata(
                mesh,
                resolved_association,
                resolved_field,
            ) or _paired_vtu_metadata(mesh, lookup_field, resolved_association)
            effective_label = label or metadata.get("display_name")
            effective_units = (
                units
                if units is not None
                else metadata.get("units")
                or _paired_vtu_units(mesh, lookup_field, resolved_association)
            )
            values = _select_vtu_component(
                values,
                component=component,
                metadata=metadata,
            )
            scalar_name = _vtu_scalar_label(
                field,
                part=part,
                component=component,
                include_part=not _vtu_label_has_part(effective_label, part),
                label=effective_label,
                units=effective_units,
            )
            _vtu_data(mesh, resolved_association)[scalar_name] = np.asarray(values)
            return scalar_name, resolved_association
        _select_vtu_data(mesh, lookup_field, association)

    values = np.asarray(data[resolved_field])

    metadata = _vtu_array_metadata(mesh, resolved_association, resolved_field)
    effective_label = label or metadata.get("display_name")
    effective_units = units if units is not None else metadata.get("units")
    if effective_units is None and part == "abs":
        effective_units = _paired_vtu_units(mesh, lookup_field, resolved_association)
    resolved_part = _resolved_vtu_part(resolved_field, field)
    include_part = (
        resolved_part == part
        and _resolved_vtu_part(resolved_field, "")
        and not _vtu_label_has_part(effective_label, part)
    )
    if resolved_part == part:
        values = np.asarray(values)
    elif part == "abs" and not np.iscomplexobj(values):
        try:
            values, resolved_association = _paired_abs_vtu_data(
                mesh, lookup_field, association
            )
            include_part = not _vtu_label_has_part(effective_label, part)
        except KeyError:
            values = np.abs(values)
            include_part = not _vtu_label_has_part(effective_label, part)
    else:
        values = _part_array(values, part)
        include_part = np.iscomplexobj(
            np.asarray(data[resolved_field])
        ) and not _vtu_label_has_part(effective_label, part)
    values = _select_vtu_component(
        values,
        component=component,
        metadata=metadata,
    )

    scalar_name = _vtu_scalar_label(
        field,
        part=part,
        component=component,
        include_part=bool(include_part),
        label=effective_label,
        units=effective_units,
    )
    _vtu_data(mesh, resolved_association)[scalar_name] = np.asarray(values)
    return scalar_name, resolved_association


def _resolved_vtu_part(resolved_field: str, base_field: str) -> str | None:
    suffix = resolved_field.removeprefix(base_field)
    for candidate in (suffix, resolved_field):
        if candidate.endswith(("_real", "_re")):
            return "real"
        if candidate.endswith(("_imag", "_im")):
            return "imag"
        if candidate.endswith(("_abs", "_mag")):
            return "abs"
    return None


def _paired_abs_vtu_data(mesh, field: str, association: str) -> tuple[np.ndarray, str]:
    for real_suffix, imag_suffix in (("_real", "_imag"), ("_re", "_im")):
        real_name = f"{field}{real_suffix}"
        imag_name = f"{field}{imag_suffix}"
        try:
            real_data, real_assoc = _select_vtu_data(mesh, real_name, association)
            imag_data, imag_assoc = _select_vtu_data(mesh, imag_name, association)
        except KeyError:
            continue
        if real_assoc != imag_assoc:
            raise ValueError(
                f"VTU fields '{real_name}' and '{imag_name}' use different associations"
            )
        return (
            np.hypot(
                np.asarray(real_data[real_name]), np.asarray(imag_data[imag_name])
            ),
            real_assoc,
        )
    available = ", ".join(_vtu_field_names(mesh, association)) or "<none>"
    raise KeyError(
        f"Could not form absolute VTU field '{field}'. Expected '{field}_real'/"
        f"'{field}_imag' or '{field}_re'/'{field}_im'. Available fields: {available}"
    )


def _paired_vtu_units(mesh, field: str, association: str) -> str | None:
    for suffix in ("_real", "_re", "_imag", "_im"):
        name = f"{field}{suffix}"
        try:
            _, resolved_association = _select_vtu_data(mesh, name, association)
        except KeyError:
            continue
        units = _vtu_array_metadata(mesh, resolved_association, name).get("units")
        if units:
            return units
    return None


def _paired_vtu_metadata(mesh, field: str, association: str) -> Mapping[str, Any]:
    for suffix in ("_real", "_re", "_imag", "_im"):
        name = f"{field}{suffix}"
        try:
            _, resolved_association = _select_vtu_data(mesh, name, association)
        except KeyError:
            continue
        metadata = _vtu_array_metadata(mesh, resolved_association, name)
        if metadata:
            return {
                key: value
                for key, value in metadata.items()
                if key in {"components", "units"}
            }
    return {}


def _apply_vtu_slice(mesh, slice_: str | Mapping[str, Any] | bool | None):
    if slice_ is None or slice_ is False:
        return mesh
    if slice_ is True:
        return mesh.slice(normal="z")
    if isinstance(slice_, str):
        return mesh.slice(normal=slice_)
    normal = slice_.get("normal", "z")
    origin = slice_.get("origin")
    return mesh.slice(normal=normal, origin=origin)


def _set_vtu_view(plotter, view: str | None) -> None:
    if view is None:
        return
    normalized = view.lower()
    if normalized in {"x_depth", "x-depth", "depth", "xy_depth"}:
        plotter.view_vector((0, 0, -1), viewup=(0, -1, 0), render=False)
    elif normalized == "xy":
        plotter.view_xy(render=False)
    elif normalized in {"xy_negative", "-xy"}:
        plotter.view_xy(negative=True, render=False)
    else:
        raise ValueError("view must be 'x_depth', 'xy', 'xy_negative', or None")
    plotter.reset_camera()


def _scalar_bar_args(
    scalar_bar: bool | Mapping[str, Any],
    scalar_bar_args: Mapping[str, Any] | None,
    *,
    title: str,
) -> tuple[bool, dict[str, Any] | None]:
    if isinstance(scalar_bar, Mapping):
        args = {**_DEFAULT_SCALAR_BAR_ARGS, **scalar_bar, **(scalar_bar_args or {})}
        show_scalar_bar = True
    else:
        show_scalar_bar = bool(scalar_bar)
        args = {**_DEFAULT_SCALAR_BAR_ARGS, **(scalar_bar_args or {})}
    if not show_scalar_bar:
        return False, None
    args.setdefault("title", title)
    return True, args


def plot_vtu(
    vtu,
    field: str | None = None,
    *,
    component: str | int | None = None,
    part: str = "real",
    source: int | str | None = None,
    association: str = "auto",
    label: str | None = None,
    units: str | Mapping[str, str] | None = None,
    slice: str | Mapping[str, Any] | bool | None = None,
    cmap: str = "RdBu_r",
    clim: tuple[float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_edges: bool = False,
    edge_color: str = "black",
    edge_width: float = 1.0,
    scalar_bar: bool | Mapping[str, Any] = True,
    background: str = "white",
    view: str | None = "x_depth",
    zoom: float | None = None,
    window_size: tuple[int, int] | None = None,
    screenshot: str | Path | None = None,
    show: bool = True,
    plotter=None,
    return_mesh: bool = False,
    **add_mesh_kwargs: Any,
):
    """Plot a scalar field from a solver ``.vtu`` file.

    Args:
        vtu: Path to a ``.vtu`` file or a PyVista dataset.
        field: VTU data array to plot. If omitted, the first available field is
            used.
        component: Optional component for vector/tensor fields. Integers use
            zero-based component indices. String names use VTU component
            metadata when present and otherwise support FrequenSolve component
            names such as ``"x"``, ``"z"``, ``"xx"``, ``"xz"``, ``"rr"``,
            ``"rz"``, and ``"tt"``.
        part: One of ``"real"``, ``"imag"``, or ``"abs"``. The solver suffix
            aliases ``"re"``, ``"im"``, and ``"mag"`` are accepted.
        source: Optional one-based source number for solver outputs named like
            ``strain_1_im`` or ``pressure_2_re``. When supplied,
            ``field="strain", source=1, part="im"`` resolves to
            ``strain_1_im``.
        association: Data association to search: ``"auto"``, ``"point"``, or
            ``"cell"``.
        label: Optional display name for the field. Part and units are appended.
        units: Optional unit label, or a mapping from raw/base field name to
            unit label. If omitted, FrequenSolve uses unit metadata from the VTU
            header.
        slice: Optional slice specification passed through to PyVista before
            plotting.
        cmap: Matplotlib/PyVista colormap name.
        clim: Explicit color limits.
        vmin: Lower color limit when ``clim`` is not provided.
        vmax: Upper color limit when ``clim`` is not provided.
        show_edges: If true, overlay edges from ``extract_all_edges()`` on the
            scalar plot. This is useful for inspecting higher-order quad/hex
            topology without relying on VTK/PyVista's rendered surface edges.
        edge_color: Color used when ``show_edges=True``.
        edge_width: Line width used when ``show_edges=True``.
        scalar_bar: Whether to show a scalar bar, or scalar-bar keyword
            arguments.
        background: Plotter background color.
        view: Camera view. The default ``"x_depth"`` orients +x to the right and
            +y/depth downward for 2D x-depth solver outputs.
        zoom: Optional camera zoom factor applied after the view is reset.
            Values greater than 1 fill more of the screenshot/window.
        window_size: Optional PyVista window size.
        screenshot: Optional output path for a screenshot.
        show: Whether to display the PyVista window.
        plotter: Existing PyVista plotter to draw into.
        return_mesh: Whether to return ``(plotter, mesh)``.
        **add_mesh_kwargs: Extra keyword arguments forwarded to
            ``plotter.add_mesh``.

    Returns:
        ``None`` by default. When ``return_mesh=True``, returns ``(plotter,
        mesh)`` so callers can continue customizing or inspecting the plot.
    """

    pv = _pyvista()
    source_mesh = _as_vtu_mesh(vtu)
    mesh = source_mesh.copy(deep=True)
    _attach_vtu_metadata(mesh, _vtu_metadata(source_mesh))

    fields = _vtu_field_names(mesh, association)
    if not fields:
        raise ValueError("VTU file does not contain point or cell data arrays")
    field = field or fields[0]
    scalar_name, _ = _resolve_vtu_scalar(
        mesh,
        field,
        association=association,
        component=component,
        part=part,
        source=source,
        label=label,
        units=units,
    )
    display_mesh = _apply_vtu_slice(mesh, slice)
    if clim is None and (vmin is not None or vmax is not None):
        scalar_data, _ = _select_vtu_data(display_mesh, scalar_name, "auto")
        scalar_values = np.asarray(scalar_data[scalar_name])
        clim = (
            float(np.nanmin(scalar_values)) if vmin is None else float(vmin),
            float(np.nanmax(scalar_values)) if vmax is None else float(vmax),
        )

    show_scalar_bar, scalar_bar_args = _scalar_bar_args(
        scalar_bar,
        add_mesh_kwargs.pop("scalar_bar_args", None),
        title=scalar_name,
    )

    mesh_kwargs = {
        "scalars": scalar_name,
        "cmap": cmap,
        "clim": clim,
        "show_scalar_bar": show_scalar_bar,
        **add_mesh_kwargs,
    }
    if show_scalar_bar:
        mesh_kwargs["scalar_bar_args"] = scalar_bar_args

    if plotter is None:
        plotter = pv.Plotter(window_size=window_size, notebook=True)
    plotter.set_background(background)
    plotter.add_mesh(display_mesh, **mesh_kwargs)
    _set_vtu_view(plotter, view)

    if show_edges:
        edge_mesh = display_mesh.extract_all_edges()
        mesh_kwargs = {
            "color": edge_color,
            "line_width": edge_width,
            **add_mesh_kwargs,
        }
        plotter.add_mesh(edge_mesh, **mesh_kwargs)

    if zoom is not None:
        if zoom <= 0:
            raise ValueError("zoom must be positive")
        plotter.camera.zoom(float(zoom))

    if screenshot is not None:
        plotter.screenshot(str(screenshot))
    if show:
        plotter.show()
    if return_mesh:
        return plotter, display_mesh
    if not show:
        plotter.close()
    return


def plot_vtu_slice(
    vtu,
    field: str | None = None,
    *,
    normal: str = "z",
    origin: Sequence[float] | None = None,
    **kwargs: Any,
):
    """Plot a planar slice through a solver ``.vtu`` file.

    Args:
        vtu: VTU file path or mesh object.
        field: Optional field name to color by.
        normal: Slice normal axis/name.
        origin: Optional slice origin.
        **kwargs: Additional ``plot_vtu`` keyword arguments.

    Returns:
        The value returned by ``plot_vtu``.
    """

    slice_spec: dict[str, Any] = {"normal": normal}
    if origin is not None:
        slice_spec["origin"] = origin
    return plot_vtu(vtu, field, slice=slice_spec, **kwargs)
