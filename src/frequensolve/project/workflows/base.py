from abc         import ABC, abstractmethod
from typing      import Callable

__all__ = ['BaseWorkflow']

class BaseWorkflow(ABC):
   """Base class for workflow components.
   
   A workflow is a sequence of operations that can be executed on project components.
   It can be a single callable or a chain of callables.

   Attributes:
      name (str): Name of the workflow
      description (str): Description of what the workflow does
      callables (List[Callable]): List of callables to execute in sequence
   """

   def __init__(self, name: str, description: str = ""):
      """Initialize workflow.
      
      Args:
         name (str): Name of the workflow
         description (str): Description of what the workflow does
      """
      self.name = name
      self.description = description
      self.callables = []

   def add(self, func: Callable) -> None:
      """Add a callable to the workflow sequence.
      
      Args:
         func (Callable): Function to add to workflow
      """
      self.callables.append(func)

   @abstractmethod
   def execute(self, *args, **kwargs):
      """Execute the workflow.
      
      Must be implemented by subclasses to define workflow execution.
      """
      pass



# TODO: Make workitem or something that is a single function with specified output and input
# TODO: Make dependency graph of workitems
# TODO: Make events for workflows? (like on_start, on_finish, on_error)
# TODO: Make concurrency/synchronization for workflows? 
