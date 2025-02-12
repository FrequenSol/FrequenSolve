import json
from pathlib import Path

import toml

from frequensolve.util.class_registry import class_registry


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Simulation objects."""

    def default(self, obj):
        import numpy as np
        from numpy import ndarray
        from xarray import DataArray

        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, ndarray):
            return obj.tolist()
        if isinstance(obj, DataArray):
            return obj.values.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__()
        return super().default(obj)


def custom_json_decoder(obj):
    if "_type" in obj:
        class_name = obj["_type"]
        if class_name in class_registry:
            model_class = class_registry[class_name]
            return model_class.from_dict(obj)
    return obj


class CustomTOMLEncoder(toml.TomlEncoder):
    """Custom TOML encoder for Simulation objects."""

    def __init__(self):
        super().__init__()

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
        try:
            return str(obj)
        except:
            print(f"Cannot encode object of type {type(obj)}")
            return None
