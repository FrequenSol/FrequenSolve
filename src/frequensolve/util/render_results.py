"""Render basic ParaView images from solver outputs."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from paraview.simple import (  # noqa
        ColorBy,
        Delete,
        GetActiveViewOrCreate,
        GetColorTransferFunction,
        Hide,
        ResetCamera,
        SaveScreenshot,
        Show,
        XDMFReader,
    )

    HAS_PARAVIEW = True
except Exception:
    HAS_PARAVIEW = False

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    paraview_dir: Path
    output_dir: Path
    fields: List[str]
    frequencies: Optional[List[float]] = None
    resolution: Tuple[int, int] = (1280, 720)
    mesh_filename: str = "mesh.png"


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
    representation: str = "Surface",
    resolution: Tuple[int, int] = (1280, 720),
) -> None:
    if not HAS_PARAVIEW:
        raise RuntimeError("ParaView Python modules not available in this environment.")

    reader = XDMFReader(FileNames=[str(xmf_path)])
    view = GetActiveViewOrCreate("RenderView")
    display = Show(reader, view)
    display.Representation = representation

    if field:
        try:
            ColorBy(display, ("POINTS", field))
            field_lut = GetColorTransferFunction(field)
            display.LookupTable = field_lut
            display.RescaleTransferFunctionToDataRange(True)
        except Exception as exc:
            logger.warning("Failed to apply field %s for %s: %s", field, xmf_path, exc)
            ColorBy(display, None)
    else:
        ColorBy(display, None)

    ResetCamera(view)
    view.Update()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    SaveScreenshot(str(output_file), view, ImageResolution=list(resolution))

    Hide(reader, view)
    Delete(reader)


def render_results(config: RenderConfig) -> Dict:
    if not config.paraview_dir.exists():
        raise FileNotFoundError(f"ParaView directory not found: {config.paraview_dir}")

    xmf_files = _find_xmf_files(config.paraview_dir)
    if not xmf_files:
        raise FileNotFoundError(f"No XMF files found in {config.paraview_dir}")

    logger.info("Rendering from %d XMF files", len(xmf_files))

    manifest = {
        "mesh": None,
        "fields": config.fields,
        "frequencies": [],
        "resolution": {"width": config.resolution[0], "height": config.resolution[1]},
    }

    # Mesh render (first available XMF)
    mesh_output = config.output_dir / config.mesh_filename
    _render_xmf(
        xmf_files[0],
        mesh_output,
        field=None,
        representation="Surface With Edges",
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
                representation="Surface",
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

    return RenderConfig(
        paraview_dir=Path(args.paraview_dir),
        output_dir=Path(args.output_dir),
        fields=[f.strip() for f in args.fields.split(",") if f.strip()],
        frequencies=frequencies,
        resolution=(int(width), int(height)),
        mesh_filename=args.mesh_filename,
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
