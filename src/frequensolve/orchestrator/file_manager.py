
import fsspec
import hashlib

from typing import List, Optional
from pathlib import Path

__all__ = ["BaseFile", "FileManager"]

class BaseFile:
   """Stores information about a file and its location on multiple file systems."""
   
   def __init__(self, name: str, local_path: str):
      """
      Args:
         name: Name of the file
         local_path: Local path to the file
      """
      self.name = name
      self.local_path = local_path
      self.remote_paths = {}
   
   def add_remote_path(self, filesystem: str, remote_path: str):
      if not self.exists(filesystem):
         self.remote_paths[filesystem] = remote_path
   
   def get_remote_path(self, filesystem: str) -> Optional[str]:
      return self.remote_paths.get(filesystem)
   
   def list_paths(self) -> List[str]:
      return list(self.remote_paths.keys())
   
   def exists(self, filesystem: str) -> bool:
      if filesystem in self.remote_paths:
         return True
      return False


# TODO: Methods to add and remove symlinks to files (for files defined outside of the project directory)
# TODO: Methods to compress and decompress files before transfer
# TODO: Methods to add temporary files to remote directories and manage their lifespan
# TODO: Methods to compare metadata of local and remote files
# TODO: Make a SyncFile with __enter__ and __exit__ methods for context managers; after 
#       updating a file using with, the file will be synced to the remote file system.
# TODO: Make a special class that restricts on which systems a file can exist, be written to, etc.
# TODO: Add a lifetime manager. For files like models on persistent resources, we might want to 
#       keep the files permenantly. 
# TODO: to_dict and from_dict methods to load project files.

class FileManager:
   """Manages a list of files across multiple file systems."""
   
   def __init__(self, filesystems: Dict[str, str]):
      """
      Args:
         filesystems: List of file system names
      """
      self.filesystems = filesystems
      self.files = []
      self.fs_map = {}
      for key, fs in filesystems.items():
         self.fs_map[key] = fsspec.filesystem(fs)
   
   def add_file(self, file: str, local_path: str):
      """Add a file to the manager."""
      self.files.append(BaseFile(file, local_path))
      for fs in self.filesystems:
         if self.exists(fs, local_path):
            self.files[-1].add_remote_path(fs, local_path)
   
   def get_file(self, name: str) -> Optional[BaseFile]:  
      """Get a file by name."""
      for file in self.files:
         if file.name == name:
            return file
      return None
   
   def remove_file(self, name: str) -> bool:
      """Remove a file by name."""
      for i, file in enumerate(self.files):
         if file.name == name:
            del self.files[i]
            return True
      return False
   
   def exists(self, filesystem: str, path: str) -> bool:
      """Check if a file exists on a given filesystem."""
      fs = self.fs_map[filesystem]
      return fs.exists(path)
   
   def listdir(self, filesystem: str, path: str) -> List[str]:
      """List files in a directory on a given filesystem."""
      fs = self.fs_map[filesystem]
      return fs.ls(path)
   
   def mkdir(self, filesystem: str, path: str, parents: bool = False) -> bool:
      """Create a directory on a given filesystem. """
      try:
         fs = self.fs_map[filesystem]
         fs.mkdir(path, create_parents=parents)
         return True
      except Exception:
         return False
   
   def rmdir(self, filesystem: str, path: str) -> bool:
      """Remove a directory on a given filesystem."""
      try:
         fs = self.fs_map[filesystem]
         # fs.rm(path, recursive=True)
         return True
      except Exception:
         return False
   
   def safe_remote_rmdir(self, filesystem: str, path: str) -> bool:
      """Safely remove a remote directory on a given filesystem. """
      fs = self.fs_map[filesystem]
      try:
         # Check if path exists and is directory
         if not self.exists(filesystem, path):
            return False
            
         # Get absolute path
         abs_path = fs.info(path)["path"]
         
         if not abs_path.startswith(self.work_dir):
            return False

         if not fs.exists(path):
            return True
         if len(fs.ls(path)) > 0:
            return False
         fs.rmdir(path)
         return True
      except Exception:
         return False
   
   def remote_put(self, filesystem: str, local_path: str, remote_path: str) -> bool:
      """Put a local file to a remote filesystem."""
      try:
         fs = self.fs_map[filesystem]
         fs.put(local_path, remote_path)
         return True
      except Exception:
         return False
   
   def remote_get(self, filesystem: str, remote_path: str, local_path: str) -> bool:
      """Get a remote file from a filesystem to a local path."""
      try:
         fs = self.fs_map[filesystem]
         fs.get(remote_path, local_path)
         return True
      except Exception:
         return False
   
   def is_folder_writeable(self, filesystem: str, path: str) -> bool:
      """Check if a folder is writeable on a given filesystem."""
      fs = self.fs_map[filesystem]
      try:
         test_file = Path(path) / ".write_test"
         fs.touch(str(test_file))
         fs.rm(str(test_file))
         return True
      except Exception:
         return False

# def get_checksum(file_path):
#    """Calculate a file's checksum."""
#    hasher = hashlib.md5()
#    with open(file_path, "rb") as f:
#       for chunk in iter(lambda: f.read(4096), b""):
#          hasher.update(chunk)
#    return hasher.hexdigest()


# def needs_transfer(local_file, remote_path, fs):
#    """Check if a file needs to be transferred."""
#    local_file = Path(local_file)
   
#    # Get local file metadata
#    if not local_file.exists():
#       return True
   
#    local_checksum = get_checksum(local_file)
#    local_size = local_file.stat().st_size
   
#    try:
#       remote_info = fs.info(remote_path)
#       remote_checksum = remote_info.get("checksum", None)
#       remote_size = remote_info["size"]
#    except FileNotFoundError:
#       return True
#    return remote_size != local_size or (remote_checksum and remote_checksum != local_checksum)


# def compare_metadata(local_file, remote_path, fs):
#     """Compare metadata of local and remote files."""
#     local_file = Path(local_file)
#     if not local_file.exists():
#         return True  # Local file doesn't exist

#     local_mtime = local_file.stat().st_mtime
#     local_size = local_file.stat().st_size

#     try:
#         remote_info = fs.info(remote_path)
#         remote_mtime = remote_info["mtime"]
#         remote_size = remote_info["size"]
#     except FileNotFoundError:
#         return True  # Remote file doesn't exist

#     # Compare modification time and size
#     return local_mtime > remote_mtime or local_size != remote_size