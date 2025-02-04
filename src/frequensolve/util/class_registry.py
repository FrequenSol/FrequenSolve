class_registry = {}


def register_class(cls):
    class_registry[cls.__name__] = cls
    return cls
