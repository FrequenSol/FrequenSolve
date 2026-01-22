"""Render basic ParaView images from solver outputs."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from frequensolve.util.mesh_convert import convert_gmp

try:
    import pyvista as pv

    HAS_PYVISTA = True
except Exception:
    HAS_PYVISTA = False

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    paraview_dir: Path
    output_dir: Path
    fields: List[str]
    frequencies: Optional[List[float]] = None
    resolution: Tuple[int, int] = (1280, 720)
    mesh_filename: str = "mesh.png"
    mesh_gmp: Optional[Path] = None


def _find_xmf_files(paraview_dir: Path) -> List[Path]:
    return sorted(paraview_dir.glob("*.xmf"))


def _extract_index(file_name: str) -> Optional[int]:
    match = re.search(r"_([0-9]{5})\\.xmf$", file_name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\\d+)\\.xmf$", file_name)
    if match:
        return int(match.group(1))
    return None


def _render_xmf(
    xmf_path: Path,
    output_file: Path,
    field: Optional[str] = None,
    show_edges: bool = False,
    resolution: Tuple[int, int] = (1280, 720),
) -> None:
    if not HAS_PYVISTA:
        raise RuntimeError("PyVista is not available in this environment.")

    mesh = pv.read(str(xmf_path))

    plotter = pv.Plotter(off_screen=True, window_size=list(resolution))
    if field and field in mesh.array_names:
        plotter.add_mesh(
            mesh,
            scalars=field,
            show_edges=show_edges,
            lighting=False,
        )
    else:
        if field:
            logger.warning(
                "Field %s not found in %s. Rendering geometry only.",
                field,
                xmf_path,
            )
        plotter.add_mesh(mesh, show_edges=show_edges, color="white", lighting=False)

    plotter.camera_position = "xy"
    plotter.show(screenshot=str(output_file), auto_close=True)


def _render_mesh_file(
    mesh_path: Path,
    output_file: Path,
    show_edges: bool = True,
    resolution: Tuple[int, int] = (1280, 720),
) -> None:
    if not HAS_PYVISTA:
        raise RuntimeError("PyVista is not available in this environment.")

    plotter = pv.Plotter(off_screen=True, window_size=list(resolution))
    mesh = pv.read(str(mesh_path))
    plotter.add_mesh(mesh, show_edges=show_edges, color="white", lighting=False)
    plotter.camera_position = "xy"
    plotter.show(screenshot=str(output_file), auto_close=True)


def _prepare_mesh_for_render(mesh_path: Path, output_dir: Path) -> Path:
    if mesh_path.suffix.lower() == ".gmp":
        output_dir.mkdir(parents=True, exist_ok=True)
        converted = output_dir / f"{mesh_path.stem}.msh"
        convert_gmp(mesh_path, converted, output_format="gmsh")
        return converted
    return mesh_path


def render_results(config: RenderConfig) -> Dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if not config.paraview_dir.exists() and not config.mesh_gmp:
        raise FileNotFoundError(f"ParaView directory not found: {config.paraview_dir}")

    xmf_files = (
        _find_xmf_files(config.paraview_dir) if config.paraview_dir.exists() else []
    )

    if xmf_files:
        logger.info("Rendering from %d XMF files", len(xmf_files))
    elif config.mesh_gmp:
        logger.info("No XMF files found; rendering mesh only from GMP.")
    else:
        raise FileNotFoundError(f"No XMF files found in {config.paraview_dir}")

    manifest = {
        "mesh": None,
        "fields": config.fields,
        "frequencies": [],
        "resolution": {"width": config.resolution[0], "height": config.resolution[1]},
    }

    mesh_output = config.output_dir / config.mesh_filename
    if config.mesh_gmp:
        render_mesh_path = _prepare_mesh_for_render(config.mesh_gmp, config.output_dir)
        _render_mesh_file(
            render_mesh_path,
            mesh_output,
            show_edges=True,
            resolution=config.resolution,
        )
    else:
        _render_xmf(
            xmf_files[0],
            mesh_output,
            field=None,
            show_edges=True,
            resolution=config.resolution,
        )
    manifest["mesh"] = {"file": str(mesh_output)}

    # Frequency renders
    for xmf_path in xmf_files:
        index = _extract_index(xmf_path.name)
        freq_value = None
        if config.frequencies and index is not None:
            if 1 <= index <= len(config.frequencies):
                freq_value = config.frequencies[index - 1]

        image_entry = {
            "index": index,
            "frequency": freq_value,
            "images": {},
        }

        for field in config.fields:
            output_file = config.output_dir / f"freq_{index or 0:05d}" / f"{field}.png"
            _render_xmf(
                xmf_path,
                output_file,
                field=field,
                show_edges=False,
                resolution=config.resolution,
            )
            image_entry["images"][field] = str(output_file)

        manifest["frequencies"].append(image_entry)

    return manifest


def _parse_args() -> RenderConfig:
    import argparse

    parser = argparse.ArgumentParser(description="Render ParaView XMF outputs to PNG.")
    parser.add_argument(
        "--paraview-dir", required=True, help="Directory with XMF files"
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write images")
    parser.add_argument(
        "--mesh-gmp",
        default="",
        help="Optional GMP mesh file to convert and render for mesh image",
    )
    parser.add_argument(
        "--fields",
        default="disp_1_abs",
        help="Comma-separated list of fields to render",
    )
    parser.add_argument(
        "--frequencies",
        default="",
        help="JSON array of frequency values (optional)",
    )
    parser.add_argument(
        "--resolution",
        default="1280x720",
        help="Resolution as WIDTHxHEIGHT (e.g. 1280x720)",
    )
    parser.add_argument(
        "--mesh-filename", default="mesh.png", help="Output filename for mesh image"
    )

    args = parser.parse_args()
    width, height = args.resolution.lower().split("x")
    frequencies = json.loads(args.frequencies) if args.frequencies else None
    mesh_gmp = Path(args.mesh_gmp) if args.mesh_gmp else None

    return RenderConfig(
        paraview_dir=Path(args.paraview_dir),
        output_dir=Path(args.output_dir),
        fields=[f.strip() for f in args.fields.split(",") if f.strip()],
        frequencies=frequencies,
        resolution=(int(width), int(height)),
        mesh_filename=args.mesh_filename,
        mesh_gmp=mesh_gmp,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    config = _parse_args()
    manifest = render_results(config)
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Render manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
