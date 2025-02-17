from typing import Union

__all__ = ["NamedList"]


class NamedList(list):
    """A list of objects with `name` attribute that can be indexed by name or index."""

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, str):
            for idx, s in enumerate(self):
                if s.name == key:
                    return super().__getitem__(idx)
            raise ValueError(f"Item '{key}' not found")
        elif isinstance(key, int):
            return super().__getitem__(key)
        else:
            raise ValueError(f"Invalid key type: {type(key)}")

    def __setitem__(self, key: Union[str, int], value):
        """Allows assignment via string key by searching for an object with matching name.

        Args:
            key (str or int): The name or index of the item.
            value: The value to assign.

        Raises:
            ValueError: If item with the given name is not found.
        """
        if isinstance(key, str):
            for idx, item in enumerate(self):
                if item.name == key:
                    return super().__setitem__(idx, value)
            raise ValueError(f"Item '{key}' not found for assignment")
        elif isinstance(key, int):
            return super().__setitem__(key, value)
        else:
            raise ValueError(f"Invalid key type: {type(key)}")
