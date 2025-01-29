
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
      while hasattr(current, 'parent') and current.parent:
         parents.append(current.parent)
         current = current.parent
      return parents

   def __repr__(self):
      parent_info = f" (Parents: {[p.__class__.__name__ for p in self.get_parents()]})" if hasattr(self, 'parent') else ""
      return f"{self.__class__.__name__}{parent_info}"
