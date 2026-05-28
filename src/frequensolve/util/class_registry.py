"""Simple class registry used for solver ``_type`` dispatch."""

class_registry = {}


def register_class(cls):
    """Register a class for type-tagged deserialization.

    Args:
        cls: Class object to register by ``cls.__name__``.

    Returns:
        The same class, allowing use as a decorator.
    """

    class_registry[cls.__name__] = cls
    return cls
