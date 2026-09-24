"""Render named camera views as one labeled contact sheet."""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .renderer import render
from .scene_loader import load_scene

CAMERA_KEYS = {"eye", "target", "fov", "far", "camera_size", "up", "orientation"}


def _views(source_views, views_file, selected):
    if views_file is None:
        data = source_views
        if data is None:
            raise ValueError("define VIEWS in the scene or pass --views cameras.json")
    else:
        data = json.loads(views_file.read_text())
    if not isinstance(data, dict):
        raise ValueError("views must be a mapping of names to camera objects")
    defaults = data.get("defaults", {}) if "views" in data else {}
    views = data.get("views", data)
    if not isinstance(defaults, dict) or not isinstance(views, dict) or not views:
        raise ValueError("views must be a nonempty mapping of names to camera objects")
    if not set(defaults) <= CAMERA_KEYS:
        raise ValueError(
            f"unknown camera defaults: {', '.join(sorted(set(defaults) - CAMERA_KEYS))}"
        )
    names = selected or list(views)
    if len(names) != len(set(names)):
        raise ValueError("each --view name may appear only once")
    missing = [name for name in names if name not in views]
    if missing:
        raise ValueError(f"unknown view: {', '.join(missing)}")
    cameras = {}
    for name in names:
        camera = views[name]
        if not isinstance(name, str) or not name.strip() or not isinstance(camera, dict):
            raise ValueError("each view needs a nonempty name and a camera object")
        if not set(camera) <= CAMERA_KEYS:
            raise ValueError(
                f"unknown camera fields in {name}: {', '.join(sorted(set(camera) - CAMERA_KEYS))}"
            )
        merged = {**defaults, **camera}
        if "eye" not in merged:
            raise ValueError(f"view {name} needs an eye position")
        cameras[name] = {
            "eye": merged["eye"],
            "target": merged.get("target", (0, 0, 0)),
            "fov": merged.get("fov", 40),
            "far": merged.get("far", 20),
            "camera_size": merged.get("camera_size", 2),
            "up": merged.get("up"),
            "orientation": merged.get("orientation"),
        }
    return cameras, data if "views" in data else {}


def _contact_sheet(frames, width, height, columns, gap):
    rows = ceil(len(frames) / columns)
    label_height = 28
    sheet = Image.new(
        "RGB",
        (columns * width + (columns + 1) * gap, rows * (height + label_height) + (rows + 1) * gap),
        "#f0f0ec",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    for index, (name, frame) in enumerate(frames):
        x = gap + (index % columns) * (width + gap)
        y = gap + (index // columns) * (height + label_height + gap)
        sheet.paste(frame, (x, y))
        label = name
        if draw.textlength(label, font=font) > width - 4:
            while label and draw.textlength(label + "…", font=font) > width - 4:
                label = label[:-1]
            if label:
                label += "…"
        draw.text((x + 2, y + height + 6), label, fill="#202020", font=font)
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Python scene defining SCENE or MODEL")
    parser.add_argument("--views", type=Path, help="JSON mapping named views to copied cameras")
    parser.add_argument("--view", action="append", help="render this name (repeat to set order)")
    parser.add_argument("-o", "--output", type=Path, default=Path("contact-sheet.png"))
    parser.add_argument("--width", type=int, help="width of each view (default: 320)")
    parser.add_argument("--height", type=int, help="height of each view (default: 240)")
    parser.add_argument("--columns", type=int, help="number of columns (default: 3)")
    parser.add_argument("--gap", type=int, help="pixel gap (default: 12)")
    parser.add_argument("--final", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bvh", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--aa", choices=("auto", "none", "fxaa", "smaa", "ssaa", "adaptive"))
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_file():
        parser.error(f"source file does not exist: {source}")
    if args.views is not None and not args.views.is_file():
        parser.error(f"views file does not exist: {args.views}")
    try:
        loaded = load_scene(source)
        cameras, saved = _views(loaded.views, args.views, args.view)
        size = saved.get("size", [320, 240])
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("saved size must contain width and height")
        width = args.width if args.width is not None else size[0]
        height = args.height if args.height is not None else size[1]
        columns = args.columns if args.columns is not None else saved.get("columns", 3)
        gap = args.gap if args.gap is not None else saved.get("gap", 12)
        final = args.final if args.final is not None else saved.get("quality") == "final"
        bvh = args.bvh if args.bvh is not None else saved.get("bvh", False)
        aa = args.aa if args.aa is not None else saved.get("aa", "auto")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (width, height, columns, gap)
        ):
            raise ValueError("size, columns, and gap must be integers")
        if not 1 <= width <= 2048 or not 1 <= height <= 2048:
            raise ValueError("width and height must be from 1 to 2048")
        if columns < 1 or gap < 0:
            raise ValueError("columns must be positive and gap must be nonnegative")
        frames = []
        for name, camera in cameras.items():
            print(f"Rendering {name} ({len(frames) + 1}/{len(cameras)})", flush=True)
            frame = render(
                loaded.compiled,
                width,
                height,
                materials=loaded.materials,
                final=final,
                aa=aa,
                bvh=bvh,
                **camera,
            )
            frames.append((name, frame))
        sheet = _contact_sheet(frames, width, height, columns, gap)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    metadata = {
        "source": str(source),
        "views": cameras,
        "size": [width, height],
        "columns": columns,
        "gap": gap,
        "quality": "final" if final else "preview",
        "bvh": bvh,
        "aa": ("ssaa" if final else "none") if aa == "auto" else aa,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {args.output} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
