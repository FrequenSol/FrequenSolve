"""Shared mixins for serialization, extra fields, paths, and change tracking."""

import copy
import json
import warnings
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Mapping, Optional

__all__ = [
    "ExportContext",
    "ExtraFieldsMixin",
    "FSSerializableMixin",
    "MaterializeMixin",
    "TypeTaggedMixin",
    "merge_extra",
    "warn_deprecated_path_api",
]


class ExportContext:
    """Context used while exporting solver-facing JSON.

    Args:
        project_path: Optional project root used to compute project-relative
            file references.
        rel_path: Optional path below ``project_path`` where the current object
            owns generated artifacts.
        store: Optional shared backing store for HDF5-serialized arrays.
        default_length_units: Optional length units used by exporters that need
            a project-wide default.
    """

    def __init__(
        self,
        project_path: Optional[Path] = None,
        rel_path: Optional[Path] = None,
        store: Optional[Any] = None,
        default_length_units: Optional[Any] = None,
    ):
        self.project_path = (
            Path(project_path).resolve() if project_path is not None else None
        )
        self.rel_path = Path(rel_path) if rel_path is not None else Path()
        self.store = store
        self.default_length_units = default_length_units

    @property
    def path(self) -> Optional[Path]:
        """Return the current artifact directory for this export context.

        Returns:
            ``project_path / rel_path`` when a project path is available;
            otherwise ``None``.
        """

        if self.project_path is None:
            return None
        return self.project_path / self.rel_path

    def child(self, rel_path: Path) -> "ExportContext":
        """Create a child context below the current relative path.

        Args:
            rel_path: Relative path segment owned by the child object.

        Returns:
            New ``ExportContext`` sharing project path, store, and unit
            defaults.
        """

        return ExportContext(
            self.project_path,
            self.rel_path / rel_path,
            store=self.store,
            default_length_units=self.default_length_units,
        )

    def relative_to_project(self, path: Path) -> Path:
        """Return ``path`` relative to the project root when possible.

        Args:
            path: Local filesystem path to normalize.

        Returns:
            Project-relative path if ``path`` is inside ``project_path``;
            otherwise the original path.
        """

        path = Path(path)
        if self.project_path is None:
            return path
        try:
            return path.resolve().relative_to(self.project_path)
        except Exception:
            return path


class FSSerializableMixin:
    """Mixin for objects that can export FrequenSolve solver JSON.

    Dataclass fields or instance attributes that are public and non-``None``
    are recursively serialized. Subclasses may provide ``schema`` and
    ``merged_extra`` for standard schema tags and pass-through fields.
    """

    schema: ClassVar[Optional[str]] = None

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        """Serialize this object to a solver-facing payload.

        Args:
            ctx: Optional export context for nested values.

        Returns:
            JSON-compatible mapping.
        """

        payload = {}
        if self.schema is not None:
            payload["schema"] = self.schema

        if is_dataclass(self):
            for field in fields(self):
                if field.name.startswith("_") or field.name == "extra":
                    continue
                value = getattr(self, field.name)
                if value is not None:
                    payload[field.name] = fs_serialize(value, ctx)
        else:
            for key, value in vars(self).items():
                if key.startswith("_") or key == "extra":
                    continue
                if value is not None:
                    payload[key] = fs_serialize(value, ctx)

        if hasattr(self, "merged_extra"):
            payload.update(self.merged_extra(payload))
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]):
        """Deserialize this object from a solver payload.

        Args:
            data: Serialized mapping accepted by the class constructor.

        Returns:
            New instance of ``cls``.
        """

        return cls(**copy.deepcopy(dict(data)))

    def to_json(self, **kwargs) -> str:
        """Serialize this object to a JSON string.

        Args:
            **kwargs: Keyword arguments forwarded to ``json.dumps``.

        Returns:
            JSON string encoded with FrequenSolve's custom encoder.
        """

        from frequensolve.util.encoders import CustomJSONEncoder

        return json.dumps(self.to_fs(), cls=CustomJSONEncoder, **kwargs)


class ExtraFieldsMixin:
    """Mixin for standardized advanced/pass-through solver fields.

    Classes using this mixin keep unknown but intentionally supported solver
    fields in ``extra`` and merge them after typed fields during export.
    """

    extra: Dict[str, Any]

    def _init_extra(self, extra: Optional[Mapping[str, Any]] = None, **kwargs) -> None:
        merged = {}
        if extra:
            merged.update(copy.deepcopy(dict(extra)))
        merged.update(kwargs)
        self.extra = merged

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return extra fields; backward-compatible alias for ``extra``."""
        return self.extra

    @kwargs.setter
    def kwargs(self, value: Mapping[str, Any]) -> None:
        """Replace extra fields through the legacy ``kwargs`` alias.

        Args:
            value: Mapping of extra serialized fields.
        """

        self.extra = copy.deepcopy(dict(value))

    def merged_extra(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Return extra fields after checking typed-field collisions.

        Args:
            payload: Typed payload being exported.

        Returns:
            Deep copy of ``extra``.

        Raises:
            ValueError: If an extra field would overwrite a typed field.
        """

        collisions = set(payload).intersection(getattr(self, "extra", {}))
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"Extra field(s) collide with typed field(s): {names}")
        return copy.deepcopy(getattr(self, "extra", {}))


class TypeTaggedMixin(FSSerializableMixin):
    """Mixin for ``_type``-tagged polymorphic solver objects."""

    type_key: ClassVar[str] = "_type"

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        """Serialize this object with its concrete class name.

        Args:
            ctx: Optional export context for nested values.

        Returns:
            JSON-compatible mapping including the configured type key.
        """

        payload = super().to_fs(ctx)
        payload[self.type_key] = self.__class__.__name__
        return payload

    @classmethod
    def dispatch_from_fs(cls, data: Mapping[str, Any], registry: Mapping[str, type]):
        """Deserialize a type-tagged payload through a class registry.

        Args:
            data: Serialized payload containing ``type_key``.
            registry: Mapping from class names to concrete classes.

        Returns:
            Concrete class instance.

        Raises:
            ValueError: If the type tag is not registered.
        """

        payload = copy.deepcopy(dict(data))
        class_name = payload.pop(cls.type_key)
        try:
            target_cls = registry[class_name]
        except KeyError:
            raise ValueError(f"Unknown {cls.__name__} class: {class_name}") from None
        return target_cls.from_fs(payload)


class MaterializeMixin:
    """Marker for objects that need to write local artifacts before export."""

    def materialize(self, ctx: Optional[ExportContext] = None) -> None:
        """Materialize export artifacts.

        Args:
            ctx: Optional export context.

        Returns:
            ``None``. Subclasses override this when they write artifacts.
        """

        return None


def fs_serialize(value: Any, ctx: Optional[ExportContext] = None) -> Any:
    """Serialize nested values into JSON-compatible solver payloads.

    Args:
        value: Object, collection, path, NumPy value, or xarray value to
            serialize.
        ctx: Optional export context passed to objects with ``to_fs``.

    Returns:
        JSON-compatible representation of ``value``.
    """
    if hasattr(value, "to_fs"):
        return value.to_fs(ctx)
    try:
        import numpy as np
        from xarray import DataArray

        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        if isinstance(value, np.complexfloating):
            return [value.real.item(), value.imag.item()]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, DataArray):
            return value.values.tolist()
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {k: fs_serialize(v, ctx) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [fs_serialize(v, ctx) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def merge_extra(
    payload: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]],
    owner: str = "object",
) -> Dict[str, Any]:
    """Merge pass-through fields after checking typed-field collisions.

    Args:
        payload: Typed fields produced by the owner.
        extra: Advanced or pass-through fields to append.
        owner: Human-readable owner name for error messages.

    Returns:
        Merged payload copy.

    Raises:
        ValueError: If ``extra`` would overwrite a typed field.
    """
    merged = dict(payload)
    extra = dict(extra or {})
    collisions = set(merged).intersection(extra)
    if collisions:
        names = ", ".join(sorted(collisions))
        raise ValueError(f"{owner} extra field(s) collide with typed field(s): {names}")
    merged.update(copy.deepcopy(extra))
    return merged


def warn_deprecated_path_api(name: str) -> None:
    """Warn that an object's legacy path API was used."""

    warnings.warn(
        f"{name} is deprecated; export paths are now supplied by ExportContext.",
        DeprecationWarning,
        stacklevel=3,
    )


class ChangedMixin:
    """Mixin for tracking whether public attributes have changed."""

    def __init__(self, *args, **kwargs):
        self._changed = False
        super().__init__(*args, **kwargs)

    def __setattr__(self, name, value):
        if name[0] != "_changed":
            self.__dict__["_changed"] = True
        super().__setattr__(name, value)

    @property
    def is_changed(self):
        """Return whether this object has been modified since reset."""

        return self.__dict__.get("_changed", False)

    def reset_changed(self):
        """Clear the changed flag."""

        self.__dict__["_changed"] = False


class ParentMixin:
    """Mixin for lightweight parent/ancestor tracking."""

    def set_parent(self, parent):
        """Set this object's parent and propagate to child ``ParentMixin`` values.

        Args:
            parent: Parent object to attach.
        """
        self.parent = parent

        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, ParentMixin):
                attr_value.set_parent(self)

    def get_parents(self):
        """Return all ancestors up to the root object.

        Returns:
            List of parent objects, nearest first.
        """
        parents = []
        current = self
        while hasattr(current, "parent") and current.parent:
            parents.append(current.parent)
            current = current.parent
        return parents

    def __repr__(self):
        parent_info = (
            f" (Parents: {[p.__class__.__name__ for p in self.get_parents()]})"
            if hasattr(self, "parent")
            else ""
        )
        return f"{self.__class__.__name__}{parent_info}"
