from dataclasses import dataclass

from frequensolve._version import get_versions

__all__ = ["Version"]


@dataclass
class Version:
    major: int
    minor: int
    patch: int

    def __init__(
        self,
        major: int | None = None,
        minor: int | None = None,
        patch: int | None = None,
    ):
        """Initialize Version object.

        Args:
           major (int, optional): Major version number. Defaults to 0.
           minor (int, optional): Minor version number. Defaults to 0.
           patch (int, optional): Patch version number. Defaults to 0.
        """
        self.major = major if major is not None else 0
        self.minor = minor if minor is not None else 0
        self.patch = patch if patch is not None else 0

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

    @classmethod
    def from_string(cls, version_str: str) -> "Version":
        """Create Version from string.

        Args:
           version_str (str): Version string in format "major.minor.patch"

        Returns:
           Version: New Version object
        """
        try:
            major, minor, patch = map(int, version_str.split("."))
        except Exception as e:
            print(major, minor, patch)
            raise ValueError(e)
        return cls(major, minor, patch)

    def current() -> "Version":
        """Get the current version.

        Returns:
           Version: The current version.
        """
        return Version.from_string(get_versions()["version"])

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
