import json
from pathlib import Path

try:
    import toml
except ImportError:
    toml = None


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Simulation objects."""

    def default(self, obj):
        import numpy as np
        from numpy import ndarray
        from xarray import DataArray

        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, (np.complex64, np.complex128)):
            return [obj.real.item(), obj.imag.item()]
        if isinstance(obj, ndarray):
            return obj.tolist()
        if isinstance(obj, DataArray):
            return obj.values.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__()
        return super().default(obj)


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


# import yaml
# class CustomYAMLEncoder(yaml.Dumper):
#     def numpy_representer(dumper, data):
#         """Convert numpy values to native Python types."""
#         return dumper.represent_float(float(data))

#     indent = kwargs.get("indent", 3)
#     try:
#         import numpy as np

#         yaml.add_representer(np.float64, numpy_representer)
#         yaml.add_representer(np.float32, numpy_representer)
#         yaml.add_representer(
#             np.int64, lambda dumper, data: dumper.represent_int(int(data))
#         )
#         yaml.add_representer(
#             np.int32, lambda dumper, data: dumper.represent_int(int(data))
#         )

#         return yaml.dump(
#             self.__dict__(),
#             indent=indent,
#             default_flow_style=False,
#             sort_keys=False,
#             **kwargs,
#         )
#     except Exception as e:
#         print(f"Failed to convert to YAML: {e}")
#         return self.__repr__()
