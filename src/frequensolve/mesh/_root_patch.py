# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Immutable native root selection; IDs belong to an authored parent mesh."""

import re
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping


@dataclass(frozen=True)
class RootPatchDescriptor:
    parent_fingerprint: str
    roots: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parent_fingerprint, str) or not re.fullmatch(
            r"[0-9a-fA-F]{16}", self.parent_fingerprint
        ):
            raise ValueError("parent_fingerprint must be a 16-digit hexadecimal string")
        roots = tuple(self.roots)
        if not roots or any(
            isinstance(root, bool) or not isinstance(root, Integral) or root < 1
            for root in roots
        ):
            raise ValueError("patch roots must be positive integer parent IDs")
        if len(set(roots)) != len(roots):
            raise ValueError("patch roots must be unique")
        object.__setattr__(self, "roots", tuple(sorted(int(root) for root in roots)))
        object.__setattr__(self, "parent_fingerprint", self.parent_fingerprint.upper())

    def to_fs(self) -> dict[str, Any]:
        return {
            "schema": "fs-root-patch-1",
            "parent_fingerprint": self.parent_fingerprint,
            "roots": list(self.roots),
        }

    @classmethod
    def from_fs(cls, value: Mapping[str, Any]) -> "RootPatchDescriptor":
        if set(value) != {"schema", "parent_fingerprint", "roots"}:
            raise ValueError(
                "root patch descriptor requires schema, parent_fingerprint and roots"
            )
        if value["schema"] != "fs-root-patch-1":
            raise ValueError("unsupported root patch descriptor schema")
        return cls(value["parent_fingerprint"], tuple(value["roots"]))
