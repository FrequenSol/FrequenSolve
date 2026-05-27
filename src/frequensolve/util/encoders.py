import json
from pathlib import Path

try:
    import toml
except ImportError:
    toml = None


class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder for FrequenSolve serialization helpers."""

    def default(self, obj):
        """Encode FrequenSolve, NumPy, xarray, and path-like values.

        Args:
            obj: Object to encode.

        Returns:
            JSON-compatible representation, or the superclass result for
            unsupported objects.
        """

        from frequensolve.util.mixins import fs_serialize

        serialized = fs_serialize(obj)
        if serialized is not obj:
            return serialized
        return super().default(obj)


if toml is not None:

    class CustomTOMLEncoder(toml.TomlEncoder):
        """TOML encoder for FrequenSolve serialization helpers."""

        def dump_value(self, obj):
            from numpy import bool_, floating, integer, ndarray
            from xarray import DataArray

            if isinstance(obj, (integer, floating, bool_)):
                return obj.item()
            if isinstance(obj, ndarray):
                return obj.tolist()
            if isinstance(obj, DataArray):
                return obj.values.tolist()
            if isinstance(obj, Path):
                return str(obj)
            if hasattr(obj, "tolist"):
                return obj.tolist()
            return str(obj)

else:

    class CustomTOMLEncoder:
        """Placeholder that fails clearly when TOML support is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError("toml is required to use CustomTOMLEncoder")
