#!/usr/bin/env python3

"""
system_info.py

Gathers detailed information about the operating system, hardware (CPU, GPU),
and versions of commonly installed software.

Usage:
    python system_info.py
"""

import logging
import platform
import subprocess
import sys
from typing import Dict, Optional, Union

try:
    import psutil
except ImportError:
    psutil = None

# Some distros might need an external library to detect distribution info
# On many Linux systems, 'distro' can provide more precise version info
try:
    import distro
except ImportError:
    distro = None

__all__ = ["SystemInfo"]

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# You can configure logging handlers/formatters as needed:
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class SystemInfo:
    """Collect operating system, CPU, GPU, and software version information."""

    def __init__(self):
        """Initialize a system information collector."""

        logger.debug("Initializing SystemInfo...")

    def get_os_info(self) -> Dict[str, str]:
        """
        Retrieve information about the operating system.
        Tries to capture details like distro name, version, etc.

        Returns:
            A dictionary with keys like 'os_system', 'os_release',
            'os_version', 'distro_name', 'distro_version' (if available).
        """
        os_info = {}

        # platform.system() often returns 'Linux', 'Windows', or 'Darwin'
        os_info["os_system"] = platform.system()
        os_info["os_release"] = platform.release()
        os_info["os_version"] = platform.version()

        # If the optional 'distro' package is available and we're on Linux,
        # we can get more granularity.
        if distro and os_info["os_system"].lower() == "linux":
            os_info["distro_name"] = distro.name(pretty=True)
            os_info["distro_version"] = distro.version(pretty=True)

        return os_info

    def get_cpu_info(self) -> Dict[str, Union[str, int]]:
        """
        Retrieve information about the CPU, such as the architecture,
        number of physical and logical cores, and memory details.

        Returns:
            A dictionary with CPU and memory related details.
        """
        cpu_info = {}
        cpu_info["machine"] = platform.machine()  # e.g., 'x86_64'
        cpu_info["processor"] = platform.processor()  # e.g., 'Intel(R) Core(TM) i7-...'

        # psutil can give more detailed info if installed
        if psutil:
            cpu_info["physical_cores"] = psutil.cpu_count(logical=False)
            cpu_info["logical_cores"] = psutil.cpu_count(logical=True)
            cpu_info["cpu_freq"] = (
                psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
            )
            cpu_info["memory"] = psutil.virtual_memory().total / 1024**2
        else:
            # Fallback if psutil is not installed
            logger.warning("psutil not found. CPU and memory details might be limited.")
            cpu_info["physical_cores"] = None
            cpu_info["logical_cores"] = None
            cpu_info["cpu_freq"] = {}
            cpu_info["memory"] = {}

        return cpu_info

    def get_memory_info(self) -> Dict[str, str]:
        """
        Retrieve information about the memory, such as the total, available, used, and percent of memory used.
        """
        memory_info = {}
        memory_info["total"] = psutil.virtual_memory().total
        memory_info["available"] = psutil.virtual_memory().available
        return memory_info

    def get_gpu_info(self) -> Dict[str, str]:
        """
        Attempt to retrieve GPU details by running `nvidia-smi`.
        If `nvidia-smi` is not found or fails, it logs a warning
        and returns partial/empty info.

        Returns:
            A dictionary describing the GPU(s), if found.
        """
        gpu_info = {}
        # Try to detect NVIDIA GPU with nvidia-smi
        try:
            cmd = ["nvidia-smi", "-L"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for idx, line in enumerate(lines):
                    gpu_info[f"GPU_{idx}"] = line
            else:
                logger.debug("nvidia-smi returned non-zero exit code.")
        except FileNotFoundError:
            logger.debug(
                "nvidia-smi not found. No NVIDIA GPU detected or driver not installed."
            )
        except Exception as e:
            logger.debug(f"Failed to run nvidia-smi: {e}")

        return gpu_info

    def get_software_versions(
        self, software_list: Optional[list] = None
    ) -> Dict[str, str]:
        """
        Attempt to retrieve version info for a list of software tools.

        By default, checks common tools like 'gcc', 'g++', 'python', 'git', 'cmake'.

        Args:
            software_list: list of software names (executables) to check.

        Returns:
            A dictionary mapping each software to its reported version string
            or 'Not found' if unsuccessful.
        """
        if software_list is None:
            software_list = ["gcc", "g++", "python", "git", "cmake"]

        versions = {}
        for sw in software_list:
            sw_version = self._get_version_of_command(sw)
            versions[sw] = sw_version
        return versions

    def _get_version_of_command(self, command_name: str) -> str:
        """
        Helper method that tries to run `<command_name> --version`
        to get the version. If that fails, returns 'Not found'.

        Args:
            command_name (str): The executable's name or path.

        Returns:
            str: The version string or 'Not found'.
        """
        # Some programs put version info on stderr, some on stdout
        # We can capture both
        try:
            result = subprocess.run(
                [command_name, "--version"], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                # Many commands show version in stdout, but some in stderr
                output = result.stdout.strip() or result.stderr.strip()
                return output.split("\n", 1)[0]  # take first line
            else:
                return "Not found"
        except FileNotFoundError:
            return "Not found"
        except Exception as e:
            logger.warning(f"Error obtaining version for {command_name}: {e}")
            return "Not found"

    def gather_all_info(self) -> Dict[str, Dict]:
        """
        Collect all information into a single dictionary.

        Returns:
            A dictionary with sub-dictionaries for OS, CPU, GPU, and
            some default software versions.
        """
        info = {
            "os": self.get_os_info(),
            "cpu": self.get_cpu_info(),
            "gpu": self.get_gpu_info(),
            "software": self.get_software_versions(),
        }
        return info


def main():
    """
    Example usage: gather all system info and print in a user-friendly manner.
    """
    system_info = SystemInfo()
    all_info = system_info.gather_all_info()

    # Display results
    print("=== System Information ===")
    for category, details in all_info.items():
        print(f"\n[{category}]")
        if isinstance(details, dict):
            for k, v in details.items():
                print(f"  {k}: {v}")
        else:
            print(details)


if __name__ == "__main__":
    main()
