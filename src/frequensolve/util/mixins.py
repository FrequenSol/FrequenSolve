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
