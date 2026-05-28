"""Structured validation diagnostics and exception helpers.

Validation reports collect stable issue codes, human-readable messages, object
paths, and optional remediation hints so callers can display diagnostics or
raise a single exception after preflight validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

__all__ = [
    "ValidationError",
    "ValidationIssue",
    "ValidationReport",
]

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    """One validation diagnostic.

    Args:
        severity: ``"error"`` for blocking issues or ``"warning"`` for
            suspicious but potentially intentional configuration.
        code: Stable machine-readable issue code.
        message: Human-readable explanation.
        path: Optional dotted path identifying the offending object.
        hint: Optional remediation guidance.
    """

    severity: Severity
    code: str
    message: str
    path: str = ""
    hint: Optional[str] = None

    def __str__(self) -> str:
        location = f"{self.path}: " if self.path else ""
        hint = f" Hint: {self.hint}" if self.hint else ""
        return f"{self.severity.upper()} {self.code}: {location}{self.message}{hint}"


@dataclass
class ValidationReport:
    """Collection of validation diagnostics.

    Args:
        issues: Initial diagnostics.
    """

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Return blocking validation issues.

        Returns:
            Diagnostics whose severity is ``"error"``.
        """

        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Return non-blocking validation issues.

        Returns:
            Diagnostics whose severity is ``"warning"``.
        """

        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        """Return whether the report has no blocking errors.

        Returns:
            ``True`` when :attr:`errors` is empty.
        """

        return not self.errors

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        *,
        path: str = "",
        hint: Optional[str] = None,
    ) -> None:
        """Append a diagnostic to the report.

        Args:
            severity: Diagnostic severity.
            code: Stable machine-readable issue code.
            message: Human-readable explanation.
            path: Optional dotted object path.
            hint: Optional remediation guidance.
        """

        self.issues.append(
            ValidationIssue(
                severity=severity,
                code=code,
                message=message,
                path=path,
                hint=hint,
            )
        )

    def error(
        self,
        code: str,
        message: str,
        *,
        path: str = "",
        hint: Optional[str] = None,
    ) -> None:
        """Append a blocking validation error.

        Args:
            code: Stable machine-readable issue code.
            message: Human-readable explanation.
            path: Optional dotted object path.
            hint: Optional remediation guidance.
        """

        self.add("error", code, message, path=path, hint=hint)

    def warning(
        self,
        code: str,
        message: str,
        *,
        path: str = "",
        hint: Optional[str] = None,
    ) -> None:
        """Append a non-blocking validation warning.

        Args:
            code: Stable machine-readable issue code.
            message: Human-readable explanation.
            path: Optional dotted object path.
            hint: Optional remediation guidance.
        """

        self.add("warning", code, message, path=path, hint=hint)

    def extend(self, other: "ValidationReport") -> None:
        """Append diagnostics from another report.

        Args:
            other: Report whose issues should be appended in order.
        """

        self.issues.extend(other.issues)

    def raise_for_errors(self) -> "ValidationReport":
        """Raise ``ValidationError`` if the report contains blocking issues.

        Returns:
            This report when no blocking errors are present.

        Raises:
            ValidationError: If any diagnostic has ``"error"`` severity.
        """

        if self.errors:
            raise ValidationError(self)
        return self

    def format(self, *, include_warnings: bool = True) -> str:
        """Return a human-readable multi-line report.

        Args:
            include_warnings: Whether to include warnings in the formatted text.
        """

        issues = self.issues if include_warnings else self.errors
        return "\n".join(str(issue) for issue in issues)


class ValidationError(ValueError):
    """Raised when a validation report contains blocking errors."""

    def __init__(self, report: ValidationReport):
        """Create an exception from a validation report.

        Args:
            report: Report containing one or more blocking validation errors.
        """

        self.report = report
        count = len(report.errors)
        plural = "" if count == 1 else "s"
        super().__init__(
            f"Validation failed with {count} error{plural}:\n"
            f"{report.format(include_warnings=False)}"
        )
