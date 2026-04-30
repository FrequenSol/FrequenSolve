import copy
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Mapping, Optional


class ExportContext:
    """Context used while exporting solver-facing JSON."""

    def __init__(
        self,
        project_path: Optional[Path] = None,
        rel_path: Optional[Path] = None,
        store: Optional[Any] = None,
    ):
        self.project_path = (
            Path(project_path).resolve() if project_path is not None else None
        )
        self.rel_path = Path(rel_path) if rel_path is not None else Path()
        self.store = store

    @property
    def path(self) -> Optional[Path]:
        if self.project_path is None:
            return None
        return self.project_path / self.rel_path

    def child(self, rel_path: Path) -> "ExportContext":
        return ExportContext(
            self.project_path, self.rel_path / rel_path, store=self.store
        )

    def relative_to_project(self, path: Path) -> Path:
        path = Path(path)
        if self.project_path is None:
            return path
        try:
            return path.resolve().relative_to(self.project_path)
        except Exception:
            return path


class FSSerializableMixin:
    """Mixin for objects that can export FrequenSolve solver JSON."""

    schema: ClassVar[Optional[str]] = None

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
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
        return cls(**copy.deepcopy(dict(data)))

    def to_json(self, **kwargs) -> str:
        from frequensolve.util.encoders import CustomJSONEncoder

        return json.dumps(self.to_fs(), cls=CustomJSONEncoder, **kwargs)


class ExtraFieldsMixin:
    """Mixin for standardized advanced/pass-through solver fields."""

    extra: Dict[str, Any]

    def _init_extra(self, extra: Optional[Mapping[str, Any]] = None, **kwargs) -> None:
        merged = {}
        if extra:
            merged.update(copy.deepcopy(dict(extra)))
        merged.update(kwargs)
        self.extra = merged

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Backward-compatible alias for older code that used `.kwargs`."""
        return self.extra

    @kwargs.setter
    def kwargs(self, value: Mapping[str, Any]) -> None:
        self.extra = copy.deepcopy(dict(value))

    def merged_extra(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        collisions = set(payload).intersection(getattr(self, "extra", {}))
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"Extra field(s) collide with typed field(s): {names}")
        return copy.deepcopy(getattr(self, "extra", {}))


class TypeTaggedMixin(FSSerializableMixin):
    """Mixin for `_type` tagged polymorphic solver objects."""

    type_key: ClassVar[str] = "_type"

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        payload = super().to_fs(ctx)
        payload[self.type_key] = self.__class__.__name__
        return payload

    @classmethod
    def dispatch_from_fs(cls, data: Mapping[str, Any], registry: Mapping[str, type]):
        payload = copy.deepcopy(dict(data))
        class_name = payload.pop(cls.type_key)
        try:
            target_cls = registry[class_name]
        except KeyError:
            raise ValueError(f"Unknown {cls.__name__} class: {class_name}") from None
        return target_cls.from_fs(payload)


class PathContextMixin:
    """Mixin for objects that carry a project path and relative path."""

    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def export_context(self) -> ExportContext:
        return ExportContext(self._proj_path, self._rel_path)

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path


class MaterializeMixin:
    """Marker for objects that need to write local artifacts before export."""

    def materialize(self, ctx: Optional[ExportContext] = None) -> None:
        return None


def fs_serialize(value: Any, ctx: Optional[ExportContext] = None) -> Any:
    """Serialize nested values into JSON-compatible solver payloads."""
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
    """Merge advanced/pass-through fields after checking typed-field collisions."""
    merged = dict(payload)
    extra = dict(extra or {})
    collisions = set(merged).intersection(extra)
    if collisions:
        names = ", ".join(sorted(collisions))
        raise ValueError(f"{owner} extra field(s) collide with typed field(s): {names}")
    merged.update(copy.deepcopy(extra))
    return merged


class ChangedMixin:
    """Mixin for tracking changes to an object."""

    def __init__(self, *args, **kwargs):
        self._changed = False
        super().__init__(*args, **kwargs)

    def __setattr__(self, name, value):
        if name[0] != "_changed":
            self.__dict__["_changed"] = True
        super().__setattr__(name, value)

    @property
    def is_changed(self):
        return self.__dict__.get("_changed", False)

    def reset_changed(self):
        self.__dict__["_changed"] = False


class ParentMixin:
    def set_parent(self, parent):
        """Recursively sets the parent and propagates it to children."""
        self.parent = parent

        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, ParentMixin):
                attr_value.set_parent(self)

    def get_parents(self):
        """Returns a list of all ancestors up to the root."""
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
