from pathlib     import Path
from typing      import List, Union, Literal
from dataclasses import dataclass

from .._version import get_versions

__all__ = ['Version']

@dataclass
class Version:
   major: int
   minor: int
   patch: int

   def __lt__(self, other: "Version") -> bool:
      """Compare two versions.

      Args:
         other (Version): The other version to compare to.

      Returns:
         bool: True if this version is less than the other version, False otherwise.
      """
      if self.major < other.major:
         return True
      elif self.major == other.major:
         if self.minor < other.minor:
            return True
         elif self.minor == other.minor:
            return self.patch < other.patch
      return False


def __init__(self, major: int = 0, minor: int = 0, patch: int = 0):
   """Initialize Version object.
   
   Args:
      major (int): Major version number
      minor (int): Minor version number 
      patch (int): Patch version number
   """
   self.major = major
   self.minor = minor 
   self.patch = patch


@classmethod
def from_string(cls, version_str: str) -> "Version":
   """Create Version from string.
   
   Args:
      version_str (str): Version string in format "major.minor.patch"
      
   Returns:
      Version: New Version object
   """
   major, minor, patch = map(int, version_str.split("."))
   return cls(major, minor, patch)


def current() -> Version:
   """Get the current version.
   
   Returns:
      Version: The current version.
   """
   return Version.from_string(get_versions()['version'])


def __str__(self) -> str:
   """Convert Version to string.
   
   Returns:
      str: Version string in format "major.minor.patch"
   """
   return f"{self.major}.{self.minor}.{self.patch}"