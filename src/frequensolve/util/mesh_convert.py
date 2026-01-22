"""Utilities for converting FrequenSolve mesh formats."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from frequensolve.mesh.mesh import Mesh


def _infer_mesh_format(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix == ".gmp":
        return "gmp"
    if suffix == ".msh":
        return "gmsh"
    if suffix in {".e", ".exo", ".exodus"}:
        return "exodus"
    return None


def _read_gmp_dimension(path: Path) -> int:
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                return int(stripped.split()[0])
    raise ValueError(f"Unable to read mesh dimension from {path}")


def convert_gmp(
    input_path: Path,
    output_path: Path,
    output_format: Optional[str] = None,
) -> Path:
    output_format = output_format or _infer_mesh_format(output_path)
    if output_format not in {"gmp", "gmsh", "exodus"}:
        raise ValueError(
            "Output format must be one of: gmp, gmsh, exodus "
            "(or inferred from file extension)."
        )

    dimension = _read_gmp_dimension(input_path)
    mesh = Mesh(dimension=dimension)
    mesh.read_mesh(input_path, format="gmp")
    mesh.write_mesh(output_path, format=output_format)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a GMP mesh to another mesh format."
    )
    parser.add_argument("--input", required=True, help="Input GMP file path")
    parser.add_argument("--output", required=True, help="Output mesh file path")
    parser.add_argument(
        "--output-format",
        choices=["gmp", "gmsh", "exodus"],
        default=None,
        help="Output format (optional; inferred from extension if omitted)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    convert_gmp(input_path, output_path, args.output_format)


if __name__ == "__main__":
    main()
