from abc          import ABC, abstractmethod
from dataclasses  import dataclass
from typing       import Optional

__all__ = ['BaseSiteConfig', 'BaseSite', 'SiteStatus']


@dataclass
class SiteStatus:
   """Status and result information for a command execution.
   
   This class tracks both immediate execution results (return code, output) 
   and ongoing job status information for batch/queued jobs.
   
   Attributes:
      status (str):      
         Current status of the execution:
            "pending":   Job is queued/waiting to start
            "running":   Job is currently executing
            "completed": Job finished successfully
            "failed":    Job failed or was cancelled
            "unknown":   Status cannot be determined
      return_code (int): 
         Exit code from the command (0 typically indicates success)
      stdout (str):      
         Standard output captured from the command
      stderr (str):      
         Standard error output from the command
      job_id (Optional[str]):       
         Job identifier for batch/queued jobs
      start_time (Optional[float]): 
         Unix timestamp when job started
   """
   status:        str = "unknown"
   return_code:   int = -1
   stdout:        str = ""
   stderr:        str = ""
   job_id:        Optional[str]   = None
   hostname:      Optional[str]   = None
   start_time:    Optional[float] = None

   @property
   def is_complete(self) -> bool:
      return self.status in ["completed", "failed"]

   @property
   def is_successful(self) -> bool:
      return self.status == "completed" and self.return_code == 0



class BaseSiteConfig(ABC):
   """Site configuration for job execution."""

   @abstractmethod
   def load(self, name: str) -> "BaseSiteConfig":
      pass

   @abstractmethod
   def save(self, name: str) -> None:
      pass



class BaseSite(ABC):
   """Base class for site configuration."""

   @abstractmethod
   def __init__(self, config: BaseSiteConfig):
      pass

   # @abstractmethod
   # def __enter__(self):
   #    pass

   # @abstractmethod
   # def __exit__(self, exc_type, exc_value, traceback):
   #    pass

   @abstractmethod
   def provision(self):
      pass

   @abstractmethod
   def deprovision(self):
      pass

   @abstractmethod
   def check_status(self):
      """Check the status of the site."""
      pass

   @abstractmethod
   def wait_provisioned(self):
      """Wait for the site to be provisioned."""
      pass
