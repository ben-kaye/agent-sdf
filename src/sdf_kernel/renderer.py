"""Render GLSL signed distance scenes to PNG through headless WebGL2."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import runpy
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import isfinite
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from .dsl import Model, Shape
from .glsl import compile_glsl
from .materials import material_arrays, scene_materials
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


def _config(scene, materials, eye, target, fov, camera_size, far, final):
    bounds = None
    try:
        box = scene.bounds()
        if not box.is_empty and all(isfinite(x) for x in (*box.minimum, *box.maximum)):
            bounds = (box.minimum, box.maximum)
    except ValueError:
        pass
    palette, metallic, roughness, opacity = material_arrays(materials)
    return {
        "name": "render",
        "materials": materials,
        "eye": eye,
        "target": target,
        "fov": fov,
        "cameraSize": camera_size,
        "far": far,
        "autoFar": False,
        "final": final,
        "bounds": bounds,
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
                        "() => !document.querySelector('#status').classList.contains('busy')",
                        timeout=90000,
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
                    return image.convert("RGB")
                finally:
                    browser.close()
        except PlaywrightError as exc:
            raise ValueError(f"headless WebGL rendering failed: {exc}") from exc
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


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
    if not all(isfinite(v) for v in (fov, far, camera_size)):
        raise ValueError("camera values must be finite")
    if sum((a - b) ** 2 for a, b in zip(eye, target, strict=True)) < 1e-8:
        raise ValueError("eye and target must be distinct 3D points")
    if far <= 0 or camera_size <= 0:
        raise ValueError("far and camera size must be positive")
    mode = ("ssaa" if final else "none") if aa == "auto" else aa
    scale = 2 if mode in ("ssaa", "adaptive") else 1
    shader = compile_glsl(scene, bvh=bvh)
    image = _render_webgl(
        shader,
        _config(scene, materials, tuple(eye), tuple(target), fov, camera_size, far, final),
        width * scale,
        height * scale,
    )
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
    parser.add_argument("--fov", type=float, default=40)
    parser.add_argument("--camera-size", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=20)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--bvh", action="store_true", help="prune bounded union components on GPU")
    parser.add_argument(
        "--aa", choices=("auto", "none", "fxaa", "smaa", "ssaa", "adaptive"), default="auto"
    )
    args = parser.parse_args()
    source = Path(args.source).resolve()
    if not source.is_file():
        parser.error(f"source file does not exist: {source}")
    sys.path.insert(0, str(source.parent))
    try:
        namespace = runpy.run_path(str(source))
    finally:
        sys.path.pop(0)
    scene = namespace.get("SCENE")
    model = namespace.get("MODEL")
    if scene is None and isinstance(model, Model):
        scene = model.scene()
    if not isinstance(scene, Shape) or scene.dim != 3:
        parser.error(f"{source} must define a 3D SCENE or MODEL")
    try:
        materials = scene_materials(
            model if isinstance(model, Model) else None, namespace.get("MATERIALS")
        )
        image = render(
            scene,
            args.width,
            args.height,
            args.eye,
            args.target,
            args.fov,
            args.far,
            camera_size=args.camera_size,
            materials=materials,
            final=args.final,
            aa=args.aa,
            bvh=args.bvh,
        )
    except ValueError as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    metadata = {
        "source": str(source),
        "eye": args.eye,
        "target": args.target,
        "fov": args.fov,
        "projection": "orthographic" if args.fov == 0 else "perspective",
        "camera_size": args.camera_size,
        "far": args.far,
        "size": [args.width, args.height],
        "quality": "final" if args.final else "preview",
        "bvh": args.bvh,
        "aa": ("ssaa" if args.final else "none") if args.aa == "auto" else args.aa,
    }
    if isinstance(model, Model):
        metadata["features"] = model.history()
    if materials:
        metadata["materials"] = materials
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {output} and camera metadata")


if __name__ == "__main__":
    main()
