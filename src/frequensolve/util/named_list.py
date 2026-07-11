"""List subclass that supports lookup by item ``name``."""

from typing import Union

__all__ = ["NamedList"]


class NamedList(list):
    """List of named objects addressable by name or index.

    Items are expected to expose a ``name`` attribute. String indexing searches
    for the first item with that name; integer indexing keeps normal list
    behavior.
    """

    def __getitem__(self, key: Union[str, int]):
        """Return an item by integer index or by its ``name`` attribute.

        Args:
            key: Integer list index or item name.

        Returns:
            Matching item.

        Raises:
            ValueError: If a named item is not found or the key type is invalid.
        """

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

    def __iadd__(self, other):
        """Append ``other`` and return the append result.

        This keeps legacy ``named_list += item`` behavior, which mirrors
        ``list.append`` rather than returning ``self``.
        """

        return self.append(other)
