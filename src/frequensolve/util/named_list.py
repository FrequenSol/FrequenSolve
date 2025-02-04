from typing import Union

__all__ = ['NamedList']

class NamedList(list):
   """A list of objects with `name` attribute that can be indexed by name or index."""

   def __getitem__(self, key: Union[str, int]):
      if isinstance(key, str):
         for s in self:
            if s.name == key:
               return s
         raise ValueError(f"Item '{key}' not found")
      elif isinstance(key, int):
         return super().__getitem__(key)
      else:
         raise ValueError(f"Invalid key type: {type(key)}")
      