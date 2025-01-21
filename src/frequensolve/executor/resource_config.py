from dataclasses import dataclass
from typing import Optional

__all__ = ['ResourceConfig']

@dataclass
class ResourceConfig:
   """Configuration for computational resource requirements.
   
   This class defines resource requirements and constraints for job execution
   on computational resources like clusters, local machines, or cloud instances.
   
   Attributes:
      name (str): Name identifier for the computational resource.
      ranks_per_job (Optional[int]): Number of MPI ranks to use per job.
      max_duration_in_seconds (Optional[int]): Maximum wall time per job in seconds.
      memory_per_rank_in_MB (Optional[int]): Memory allocation per MPI rank in megabytes.
   """

   def __init__(
      self,
      name: str,
      ranks_per_job: Optional[int] = None,
      max_duration_in_seconds: Optional[int] = None,
      memory_per_rank_in_MB: Optional[int] = None
   ):
      """Initialize a new ResourceConfig instance.
      
      Args:
         name: Name identifier for the computational resource.
         ranks_per_job: Number of MPI ranks to use per job.
         max_duration_in_seconds: Maximum wall time per job in seconds.
         memory_per_rank_in_MB: Memory allocation per MPI rank in megabytes.
      
      Raises:
         ValueError: If validation fails.
      """
      self.name = name
      self.ranks_per_job = ranks_per_job
      self.max_duration_in_seconds = max_duration_in_seconds
      self.memory_per_rank_in_MB = memory_per_rank_in_MB
      self._validate()

   def _validate(self) -> None:
      """Validate the configuration parameters.
      
      Checks that:
         1. name is not empty
         2. numeric parameters are positive when specified
      
      Raises:
         ValueError: If any validation checks fail.
      """
      if not self.name:
         raise ValueError("name cannot be empty")

      if self.ranks_per_job is not None:
         if self.ranks_per_job <= 0:
            raise ValueError("ranks_per_job must be positive")

      if self.max_duration_in_seconds is not None:
         if self.max_duration_in_seconds <= 0:
            raise ValueError("max_duration_in_seconds must be positive")

      if self.memory_per_rank_in_MB is not None:
         if self.memory_per_rank_in_MB <= 0:
            raise ValueError("memory_per_rank_in_MB must be positive")

   def to_dict(self) -> dict:
      """Convert the configuration to a dictionary.
      
      Returns:
         dict: Dictionary containing all non-None configuration values.
      """
      return {
         "name": self.name,
         "ranks_per_job": self.ranks_per_job,
         "max_duration_in_seconds": self.max_duration_in_seconds,
         "memory_per_rank_in_MB": self.memory_per_rank_in_MB
      }

   @classmethod
   def from_dict(cls, data: dict) -> "ResourceConfig":
      """Create a ResourceConfig instance from a dictionary.
      
      Args:
         data: Dictionary containing configuration parameters.
      
      Returns:
         ResourceConfig: New instance initialized with the dictionary values.
      """
      return cls(**data)

   def __str__(self) -> str:
      """Create a human-readable string representation.
      
      Returns:
         str: Multi-line description of the configuration.
      """
      lines = [f"ResourceConfig: {self.name}"]
      if self.ranks_per_job is not None:
         lines.append(f"   Ranks per job: {self.ranks_per_job}")
      if self.max_duration_in_seconds is not None:
         lines.append(f"   Max duration: {self.max_duration_in_seconds} seconds")
      if self.memory_per_rank_in_MB is not None:
         lines.append(f"   Memory per rank: {self.memory_per_rank_in_MB} MB")
      return "\n".join(lines) 