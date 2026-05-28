"""Simple class registry used for solver ``_type`` dispatch."""

class_registry = {}


def register_class(cls):
    """Register ``cls`` by class name and return it unchanged."""

    class_registry[cls.__name__] = cls
    return cls
