"""Render GLSL signed distance scenes to PNG through headless WebGL2."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import isfinite, sqrt
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from .bvh import _union_components
from .dsl import Shape, min_
from .glsl import compile_glsl
from .materials import material_arrays, scene_materials
from .model import Model
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


def _orientation(value):
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("orientation must be four finite numbers (w, x, y, z)") from exc
    if len(result) != 4 or not all(isfinite(component) for component in result):
        raise ValueError("orientation must be four finite numbers (w, x, y, z)")
    length = sqrt(sum(component * component for component in result))
    if length < 1e-8:
        raise ValueError("orientation must be a nonzero quaternion")
    return tuple(component / length for component in result)


def _orientation_up(orientation):
    w, x, y, z = orientation
    return (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))


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
    html = Path(__file__).with_name("glsl_viewer.html").read_bytes()
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


def _render_chunked(scene, config, width, height):
    components = _union_components(scene)
    if len(components) < 2:
        return _render_webgl(compile_glsl(scene), config, width, height)
    layers = []
    depths = []
    start = 0
    while start < len(components):
        end = min(start + 100, len(components))
        part = min_(*components[start:end])
        while _node_count(part) > 900 and end > start + 1:
            end = start + (end - start) // 2
            part = min_(*components[start:end])
        shader = compile_glsl(part)
        part_config = {**config, "bounds": _scene_bounds(part)}
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
    bvh=False,
    up=None,
    orientation=None,
):
    """Render a scene with the same GLSL program used by the browser viewer."""
    if isinstance(scene, Model):
        materials = scene_materials(scene, materials)
        scene = scene.scene()
    else:
        materials = scene_materials(None, materials)
    if not isinstance(scene, Shape) or scene.dim != 3:
        raise ValueError("render needs a 3D Shape or Model")
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
        image = _render_chunked(scene, config, width * scale, height * scale)
    else:
        shader = compile_glsl(scene, bvh=bvh)
        image = _render_webgl(shader, config, width * scale, height * scale)
    if scale > 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    elif mode == "smaa":
        image = Image.fromarray(smaa(np.asarray(image)), "RGB")
    elif mode == "fxaa":
        image = Image.fromarray(_fxaa(np.asarray(image)), "RGB")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("-o", "--output", default="render.png")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--eye", nargs=3, type=float, default=(4, -6, 3))
    parser.add_argument("--target", nargs=3, type=float, default=(0, 0, 0))
    parser.add_argument("--up", nargs=3, type=float, help="camera up vector")
    parser.add_argument("--orientation", nargs=4, type=float, help="camera quaternion (w x y z)")
    parser.add_argument("--fov", type=float, default=40)
    parser.add_argument("--camera-size", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=20)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--bvh", action="store_true", help="prune bounded union components on GPU")
    parser.add_argument(
        "--aa", choices=("auto", "none", "fxaa", "smaa", "ssaa", "adaptive"), default="auto"
    )
    args = parser.parse_args()
    try:
        loaded = load_scene(args.source)
        image = render(
            loaded.scene,
            args.width,
            args.height,
            args.eye,
            args.target,
            args.fov,
            args.far,
            camera_size=args.camera_size,
            materials=loaded.materials,
            final=args.final,
            aa=args.aa,
            bvh=args.bvh,
            up=args.up,
            orientation=args.orientation,
        )
    except ValueError as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    metadata = {
        "source": str(loaded.source),
        "eye": args.eye,
        "target": args.target,
        "up": args.up,
        "orientation": args.orientation,
        "fov": args.fov,
        "projection": "orthographic" if args.fov == 0 else "perspective",
        "camera_size": args.camera_size,
        "far": args.far,
        "size": [args.width, args.height],
        "quality": "final" if args.final else "preview",
        "bvh": args.bvh,
        "aa": ("ssaa" if args.final else "none") if args.aa == "auto" else args.aa,
    }
    if loaded.model is not None:
        metadata["features"] = loaded.model.history()
    if loaded.materials:
        metadata["materials"] = loaded.materials
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {output} and camera metadata")


if __name__ == "__main__":
    main()
