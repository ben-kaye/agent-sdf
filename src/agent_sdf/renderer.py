"""Render named views of a scene to one PNG and summarize its geometry."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import atan, ceil, isfinite, radians, sin, tan
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from .camera import _orientation, _orientation_up
from .field import min_
from .glsl import compile_glsl
from .material import material_arrays, scene_materials
from .model import CompiledScene, compile_scene
from .scene_loader import load_scene
from .smaa import smaa
from .transform import _vector3


def _chrome():
    candidates = (
        os.environ.get("SDF_CHROME"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise ValueError("PNG rendering requires Chrome or Chromium (or set SDF_CHROME)")


def _fxaa(image):
    """Apply edge-aware antialiasing to one rendered RGB frame."""
    rgb = image.astype(np.float32)
    luma = rgb @ np.array((0.299, 0.587, 0.114), dtype=np.float32)
    padded = np.pad(luma, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    north, south = padded[:-2, 1:-1], padded[2:, 1:-1]
    west, east = padded[1:-1, :-2], padded[1:-1, 2:]
    northwest, northeast = padded[:-2, :-2], padded[:-2, 2:]
    southwest, southeast = padded[2:, :-2], padded[2:, 2:]
    minimum = np.minimum.reduce((center, north, south, west, east))
    maximum = np.maximum.reduce((center, north, south, west, east))
    edge = maximum - minimum > np.maximum(21.25, maximum * 0.166)

    dx = -((northwest + northeast) - (southwest + southeast))
    dy = (northwest + southwest) - (northeast + southeast)
    reduction = np.maximum((northwest + northeast + southwest + southeast) * 0.03125, 1e-3)
    reciprocal = 1 / (np.minimum(np.abs(dx), np.abs(dy)) + reduction)
    dx = np.clip(dx * reciprocal, -8, 8)
    dy = np.clip(dy * reciprocal, -8, 8)
    y, x = np.indices(luma.shape, dtype=np.float32)

    def sample(offset):
        sx = np.clip(x + dx * offset, 0, image.shape[1] - 1)
        sy = np.clip(y + dy * offset, 0, image.shape[0] - 1)
        x0, y0 = np.floor(sx).astype(np.int32), np.floor(sy).astype(np.int32)
        x1 = np.minimum(x0 + 1, image.shape[1] - 1)
        y1 = np.minimum(y0 + 1, image.shape[0] - 1)
        wx, wy = (sx - x0)[..., None], (sy - y0)[..., None]
        top = rgb[y0, x0] * (1 - wx) + rgb[y0, x1] * wx
        bottom = rgb[y1, x0] * (1 - wx) + rgb[y1, x1] * wx
        return top * (1 - wy) + bottom * wy

    inner = (sample(-1 / 6) + sample(1 / 6)) * 0.5
    outer = inner * 0.5 + (sample(-0.5) + sample(0.5)) * 0.25
    outer_luma = outer @ np.array((0.299, 0.587, 0.114), dtype=np.float32)
    blended = np.where(((outer_luma < minimum) | (outer_luma > maximum))[..., None], inner, outer)
    return np.where(edge[..., None], blended, rgb).astype(np.uint8)


def _scene_bounds(scene):
    bounds = None
    try:
        box = scene.bounds()
        if not box.is_empty and all(isfinite(x) for x in (*box.minimum, *box.maximum)):
            bounds = (box.minimum, box.maximum)
    except ValueError:
        pass
    return bounds


def _config(scene, materials, eye, target, fov, camera_size, far, final, up, orientation):
    palette, metallic, roughness, opacity = material_arrays(materials)
    return {
        "name": "render",
        "materials": materials,
        "eye": eye,
        "target": target,
        "up": up,
        "orientation": orientation,
        "fov": fov,
        "cameraSize": camera_size,
        "far": far,
        "autoFar": False,
        "final": final,
        "bounds": _scene_bounds(scene),
        "palette": palette.tolist(),
        "metallic": metallic.tolist(),
        "roughness": roughness.tolist(),
        "opacity": opacity.tolist(),
    }


def _render_webgl(shader, config, width, height):
    html = Path(__file__).with_name("viewer.html").read_bytes()
    resources = {
        "/": (html, "text/html; charset=utf-8"),
        "/config": (json.dumps(config).encode(), "application/json"),
        "/scene.glsl": (shader.encode(), "text/plain; charset=utf-8"),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            resource = resources.get(self.path.split("?", 1)[0])
            if resource is None:
                self.send_error(404)
                return
            body, content_type = resource
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    executable_path=_chrome(),
                    headless=True,
                    args=["--enable-unsafe-swiftshader", "--disable-gpu-sandbox"],
                )
                try:
                    page = browser.new_page(viewport={"width": width, "height": height})
                    page.goto(f"http://127.0.0.1:{server.server_port}/?embed", wait_until="load")
                    page.wait_for_function(
                        """([width, height]) => {
                          const status = document.querySelector('#status');
                          const canvas = document.querySelector('#image');
                          return status.classList.contains('error') ||
                            (canvas.dataset.renderComplete !== undefined &&
                             canvas.width === width && canvas.height === height);
                        }""",
                        arg=[width, height],
                        timeout=300000,
                    )
                    if "error" in (page.locator("#status").get_attribute("class") or ""):
                        raise ValueError(page.locator("#status-text").inner_text())
                    data_url = page.locator("#image").evaluate(
                        "canvas => canvas.toDataURL('image/png')"
                    )
                    image = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
                    if image.size != (width, height):
                        raise ValueError(
                            f"headless WebGL returned {image.size}, expected {(width, height)}"
                        )
                    return image.convert("RGBA" if config.get("captureMode") == 1 else "RGB")
                finally:
                    browser.close()
        except PlaywrightError as exc:
            raise ValueError(f"headless WebGL rendering failed: {exc}") from exc
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _node_count(scene):
    seen = set()
    pending = [scene]
    while pending:
        node = pending.pop()
        if id(node) not in seen:
            seen.add(id(node))
            pending.extend(node.children)
    return len(seen)


def _render_chunked(compiled, config, width, height, *, bvh):
    parts = compiled.parts
    if len(parts) < 2:
        return _render_webgl(compile_glsl(compiled, bvh=bvh), config, width, height)
    layers = []
    depths = []
    start = 0
    while start < len(parts):
        end = min(start + 100, len(parts))
        field = min_(*(part.field for part in parts[start:end]))
        while _node_count(field) > 900 and end > start + 1:
            end = start + (end - start) // 2
            field = min_(*(part.field for part in parts[start:end]))
        group = CompiledScene(field, compiled.materials, parts[start:end])
        shader = compile_glsl(group, bvh=bvh)
        part_config = {**config, "bounds": _scene_bounds(field)}
        color = _render_webgl(shader, {**part_config, "captureMode": 1}, width, height)
        depth = _render_webgl(shader, {**part_config, "captureMode": 2}, width, height)
        layers.append(np.asarray(color, dtype=np.uint8))
        rgb = np.asarray(depth, dtype=np.uint32)
        depths.append(rgb[:, :, 0] * 65536 + rgb[:, :, 1] * 256 + rgb[:, :, 2])
        start = end

    order = np.argsort(np.stack(depths), axis=0)
    colors = np.stack(layers)
    sky = (np.arange(height, dtype=np.float32) + 0.5) / height
    background = np.array((244, 244, 240), dtype=np.float32) - sky[:, None] * np.array(
        (24, 21, 16), dtype=np.float32
    )
    composed = np.broadcast_to(background[:, None, :], (height, width, 3)).copy()
    for index in range(len(layers) - 1, -1, -1):
        selected = np.take_along_axis(colors, order[index][None, :, :, None], axis=0)[0]
        alpha = selected[:, :, 3:4].astype(np.float32) / 255
        composed = selected[:, :, :3].astype(np.float32) * alpha + composed * (1 - alpha)
    return Image.fromarray(np.rint(composed).astype(np.uint8), "RGB")


def render(
    scene,
    width,
    height,
    eye,
    target,
    fov=40,
    far=20.0,
    *,
    camera_size=2.0,
    materials=None,
    final=False,
    aa="auto",
    bvh=True,
    up=None,
    orientation=None,
):
    """Render a scene with the same GLSL program used by the browser viewer."""
    compiled = compile_scene(scene)
    scene = compiled.field
    materials = scene_materials(compiled.materials, materials)
    if scene.dim != 3:
        raise ValueError("render needs a 3D scene")
    if not 1 <= width <= 2048 or not 1 <= height <= 2048:
        raise ValueError("width and height must be from 1 to 2048")
    if aa not in ("auto", "none", "fxaa", "smaa", "ssaa", "adaptive"):
        raise ValueError("invalid antialiasing mode")
    if fov != 0 and not 1 <= fov <= 120:
        raise ValueError("fov must be zero or between 1 and 120 degrees")
    eye, target = _vector3(eye, "eye"), _vector3(target, "target")
    if up is not None:
        up = _vector3(up, "up")
    if orientation is not None:
        orientation = _orientation(orientation)
        orientation_up = _orientation_up(orientation)
        if up is not None and sum(a * b for a, b in zip(up, orientation_up, strict=True)) <= 0:
            raise ValueError("up and orientation must point to the same camera hemisphere")
    if not all(isfinite(v) for v in (fov, far, camera_size)):
        raise ValueError("camera values must be finite")
    if sum((a - b) ** 2 for a, b in zip(eye, target, strict=True)) < 1e-8:
        raise ValueError("eye and target must be distinct 3D points")
    if up is not None:
        forward = tuple(b - a for a, b in zip(eye, target, strict=True))
        cross = (
            forward[1] * up[2] - forward[2] * up[1],
            forward[2] * up[0] - forward[0] * up[2],
            forward[0] * up[1] - forward[1] * up[0],
        )
        if sum(component * component for component in cross) < 1e-16:
            raise ValueError("up must not be parallel to the viewing direction")
    if far <= 0 or camera_size <= 0:
        raise ValueError("far and camera size must be positive")
    mode = ("ssaa" if final else "none") if aa == "auto" else aa
    scale = 2 if mode in ("ssaa", "adaptive") else 1
    config = _config(scene, materials, eye, target, fov, camera_size, far, final, up, orientation)
    if _node_count(scene) > 1000:
        image = _render_chunked(compiled, config, width * scale, height * scale, bvh=bvh)
    else:
        shader = compile_glsl(compiled, bvh=bvh)
        image = _render_webgl(shader, config, width * scale, height * scale)
    if scale > 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    elif mode == "smaa":
        image = Image.fromarray(smaa(np.asarray(image)), "RGB")
    elif mode == "fxaa":
        image = Image.fromarray(_fxaa(np.asarray(image)), "RGB")
    return image


CAMERA_KEYS = {"eye", "target", "fov", "far", "camera_size", "up", "orientation"}

# Direction from the target to the eye, and the screen up vector.
STANDARD_VIEWS = {
    "iso": ((1, -1, 1), (0, 0, 1)),
    "front": ((0, -1, 0), (0, 0, 1)),
    "back": ((0, 1, 0), (0, 0, 1)),
    "left": ((-1, 0, 0), (0, 0, 1)),
    "right": ((1, 0, 0), (0, 0, 1)),
    "top": ((0, 0, 1), (0, 1, 0)),
    "bottom": ((0, 0, -1), (0, 1, 0)),
}


def _standard_view(name, bounds, width, height):
    """Frame the scene bounds from a standard direction; axis views are orthographic."""
    direction, up = STANDARD_VIEWS[name]
    minimum, maximum = bounds or ((-1, -1, -1), (1, 1, 1))
    center = tuple((a + b) / 2 for a, b in zip(minimum, maximum, strict=True))
    half = tuple(max((b - a) / 2, 1e-3) for a, b in zip(minimum, maximum, strict=True))
    radius = sum(h * h for h in half) ** 0.5
    length = sum(d * d for d in direction) ** 0.5
    direction = tuple(d / length for d in direction)
    aspect = width / height
    if name == "iso":
        fov = 40
        half_angle = min(radians(fov) / 2, atan(tan(radians(fov) / 2) * aspect))
        distance = radius / sin(half_angle) * 1.05
        size = 2.0
    else:
        fov = 0
        right = tuple(
            u * d - w * e
            for u, d, w, e in zip(
                (up[1], up[2], up[0]),
                (direction[2], direction[0], direction[1]),
                (up[2], up[0], up[1]),
                (direction[1], direction[2], direction[0]),
                strict=True,
            )
        )
        screen_up = sum(abs(u) * h for u, h in zip(up, half, strict=True))
        screen_right = sum(abs(r) * h for r, h in zip(right, half, strict=True))
        size = 2.2 * max(screen_up, screen_right / aspect)
        distance = 2 * radius + 1
    return {
        "eye": [c + d * distance for c, d in zip(center, direction, strict=True)],
        "target": list(center),
        "fov": fov,
        "far": distance + 2 * radius + 1,
        "camera_size": size,
        "up": list(up),
        "orientation": None,
    }


def _camera(name, camera, defaults):
    if not isinstance(name, str) or not name.strip() or not isinstance(camera, dict):
        raise ValueError("each view needs a nonempty name and a camera object")
    if not set(camera) <= CAMERA_KEYS:
        unknown = ", ".join(sorted(set(camera) - CAMERA_KEYS))
        raise ValueError(f"unknown camera fields in {name}: {unknown}")
    merged = {**defaults, **camera}
    if "eye" not in merged:
        raise ValueError(f"view {name} needs an eye position")
    return {
        "eye": merged["eye"],
        "target": merged.get("target", (0, 0, 0)),
        "fov": merged.get("fov", 40),
        "far": merged.get("far", 20),
        "camera_size": merged.get("camera_size", 2),
        "up": merged.get("up"),
        "orientation": merged.get("orientation"),
    }


def _saved_views(data):
    """Read VIEWS or a views file: a name-to-camera mapping, or prior render metadata."""
    if data is None:
        return {}, {}
    if not isinstance(data, dict):
        raise ValueError("views must be a mapping of names to camera objects")
    saved = data if "views" in data else {}
    defaults = data.get("defaults", {}) if saved else {}
    views = data.get("views", data)
    if not isinstance(defaults, dict) or not isinstance(views, dict):
        raise ValueError("views must be a mapping of names to camera objects")
    if not set(defaults) <= CAMERA_KEYS:
        unknown = ", ".join(sorted(set(defaults) - CAMERA_KEYS))
        raise ValueError(f"unknown camera defaults: {unknown}")
    return {name: _camera(name, camera, defaults) for name, camera in views.items()}, saved


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


def _box_summary(bounds):
    if bounds is None:
        return None
    minimum, maximum = bounds
    return {
        "min": list(minimum),
        "max": list(maximum),
        "extents": [b - a for a, b in zip(minimum, maximum, strict=True)],
    }


def _format_vector(values):
    return "(" + ", ".join(f"{value:.4g}" for value in values) + ")"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Python scene defining SCENE or MODEL")
    parser.add_argument(
        "--view",
        nargs="+",
        action="extend",
        help=f"views to render, in order: {', '.join(STANDARD_VIEWS)}, or names from VIEWS",
    )
    parser.add_argument("--views", type=Path, help="JSON views, or a previous render's metadata")
    parser.add_argument("-o", "--output", type=Path, default=Path("render.png"))
    parser.add_argument("--width", type=int, help="width of each view (default: 320)")
    parser.add_argument("--height", type=int, help="height of each view (default: 240)")
    parser.add_argument("--columns", type=int, help="views per row (default: 3)")
    parser.add_argument("--gap", type=int, help="pixel gap between views (default: 12)")
    camera = parser.add_argument_group("custom camera", "add a view named 'camera'")
    camera.add_argument("--eye", nargs=3, type=float)
    camera.add_argument("--target", nargs=3, type=float, help="default: scene center")
    camera.add_argument("--up", nargs=3, type=float)
    camera.add_argument("--orientation", nargs=4, type=float, help="quaternion (w x y z)")
    camera.add_argument("--fov", type=float, default=40, help="degrees; 0 is orthographic")
    camera.add_argument("--camera-size", type=float, default=2.0, help="orthographic height")
    camera.add_argument("--far", type=float, default=20)
    parser.add_argument("--final", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--bvh",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="accelerate bounded geometry on GPU (default: enabled)",
    )
    parser.add_argument("--aa", choices=("auto", "none", "fxaa", "smaa", "ssaa", "adaptive"))
    args = parser.parse_args()
    source = args.source.resolve()
    try:
        loaded = load_scene(source)
        data = loaded.views if args.views is None else json.loads(args.views.read_text())
        saved_views, saved = _saved_views(data)
        size = saved.get("size", [320, 240])
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("saved size must contain width and height")
        width = args.width if args.width is not None else size[0]
        height = args.height if args.height is not None else size[1]
        columns = args.columns if args.columns is not None else saved.get("columns", 3)
        gap = args.gap if args.gap is not None else saved.get("gap", 12)
        final = args.final if args.final is not None else saved.get("quality") == "final"
        bvh = args.bvh if args.bvh is not None else saved.get("bvh", True)
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

        bounds = _scene_bounds(loaded.scene)
        names = args.view or ([] if args.eye else list(saved_views) or ["iso"])
        if len(names) != len(set(names)):
            raise ValueError("each view may appear only once")
        cameras = {}
        for name in names:
            if name in saved_views:
                cameras[name] = saved_views[name]
            elif name in STANDARD_VIEWS:
                cameras[name] = _standard_view(name, bounds, width, height)
            else:
                raise ValueError(f"unknown view: {name}")
        if args.eye:
            if "camera" in cameras:
                raise ValueError("the custom camera view is named 'camera'; rename that view")
            center = [(a + b) / 2 for a, b in zip(*bounds, strict=True)] if bounds else [0, 0, 0]
            cameras["camera"] = {
                "eye": args.eye,
                "target": args.target or center,
                "fov": args.fov,
                "far": args.far,
                "camera_size": args.camera_size,
                "up": args.up,
                "orientation": args.orientation,
            }

        frames = []
        for name, view in cameras.items():
            if len(cameras) > 1:
                print(f"Rendering {name} ({len(frames) + 1}/{len(cameras)})", flush=True)
            frame = render(
                loaded.compiled,
                width,
                height,
                materials=loaded.materials,
                final=final,
                aa=aa,
                bvh=bvh,
                **view,
            )
            frames.append((name, frame))
        image = (
            frames[0][1]
            if len(frames) == 1
            else _contact_sheet(frames, width, height, columns, gap)
        )
        components = [
            {"path": "/".join(part.path), "bounds": _box_summary(_scene_bounds(part.field))}
            for part in loaded.compiled.parts
            if part.path
        ]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    metadata = {
        "source": str(source),
        "views": cameras,
        "size": [width, height],
        "columns": columns,
        "gap": gap,
        "quality": "final" if final else "preview",
        "bvh": bvh,
        "aa": ("ssaa" if final else "none") if aa == "auto" else aa,
        "bounds": _box_summary(bounds),
    }
    if components:
        metadata["components"] = components
    if loaded.materials:
        metadata["materials"] = loaded.materials
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {args.output} and {metadata_path}")
    if bounds is None:
        print("Bounds: unbounded")
    else:
        summary = _box_summary(bounds)
        print(
            f"Bounds: {_format_vector(summary['min'])} to {_format_vector(summary['max'])}, "
            f"extents {_format_vector(summary['extents'])}"
        )
    if components:
        print(f"Components: {len(components)} (bounds in {metadata_path})")


if __name__ == "__main__":
    main()
