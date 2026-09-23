"""Local browser viewer for JAX signed distance scenes."""

import argparse
import io
import json
import runpy
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import isfinite
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from .bvh import compile_bvh_sample
from .dsl import Model, Shape, _custom_count
from .materials import material_arrays, scene_materials


def load_scene(source):
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError(f"scene file does not exist: {source}")
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
        raise ValueError("scene must define a 3D SCENE or MODEL")
    return (
        source,
        scene,
        scene_materials(model if isinstance(model, Model) else None, namespace.get("MATERIALS")),
    )


class ViewerKernel:
    """Keep one compiled ray marcher per image size; camera values remain dynamic."""

    def __init__(self, scene, materials, *, bvh=True):
        self.scene = scene
        self.sample = (
            compile_bvh_sample(scene) if bvh else lambda points, active=None: scene.sample(points)
        )
        self.compiled = {}
        self.palette, self.metal, self.rough = material_arrays(materials)
        self.has_materials = bool(materials)

    def _compile(self, width, height):
        scene = self.scene
        sample = self.sample
        palette, metal, rough = self.palette, self.metal, self.rough
        has_materials = self.has_materials
        tile_limit = 16 if _custom_count(scene) >= 8 else 32
        tile_height, tile_width = min(height, tile_limit), min(width, tile_limit)
        rows, cols = jnp.meshgrid(jnp.arange(tile_height), jnp.arange(tile_width), indexing="ij")
        light = jnp.asarray((0.55, -0.45, 0.7), dtype=jnp.float32)
        light /= jnp.linalg.norm(light)

        def unit(v):
            return v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

        def frame(eye, target, fov, camera_size, far, row_start, col_start):
            forward = unit(target - eye)
            horizontal = jnp.linalg.norm(forward[:2])
            right = jnp.where(
                horizontal > 1e-8,
                jnp.asarray((forward[1], -forward[0], 0.0)) / jnp.maximum(horizontal, 1e-8),
                jnp.asarray((1.0, 0.0, 0.0)),
            )
            up = jnp.cross(right, forward)
            orthographic = fov == 0
            spread = jnp.where(orthographic, camera_size / 2, jnp.tan(jnp.deg2rad(fov) / 2))
            sx = (2 * (cols + col_start + 0.5) / width - 1) * spread * width / height
            sy = (1 - 2 * (rows + row_start + 0.5) / height) * spread
            displacement = sx[..., None] * right + sy[..., None] * up
            origin = jnp.where(orthographic, eye + displacement, eye)
            ray = jnp.where(orthographic, forward, unit(forward + displacement))
            travel = jnp.zeros((tile_height, tile_width), dtype=jnp.float32)
            hit = jnp.zeros((tile_height, tile_width), dtype=bool)
            material = jnp.zeros((tile_height, tile_width), dtype=jnp.int32)
            active = jnp.ones((tile_height, tile_width), dtype=bool)

            def step(state):
                count, travel, hit, material, active = state
                p = origin + travel[..., None] * ray
                distance, sampled_material = sample(p, active)
                found = active & (distance < 0.001)
                hit = hit | found
                material = jnp.where(found, sampled_material.astype(jnp.int32), material)
                travel = jnp.where(
                    active & ~found, travel + jnp.maximum(distance * 0.75, 0.0005), travel
                )
                active = active & ~found & (travel <= far)
                return count + 1, travel, hit, material, active

            def marching(state):
                count, _, _, _, active = state
                return (count < 96) & jnp.any(active)

            _, travel, hit, material, _ = jax.lax.while_loop(
                marching, step, (0, travel, hit, material, active)
            )
            shade = (rows + row_start) / max(height - 1, 1)
            background = jnp.stack((244 - 24 * shade, 244 - 21 * shade, 240 - 16 * shade), axis=-1)

            def shade_hits():
                p = origin + travel[..., None] * ray
                offsets = jnp.eye(3, dtype=jnp.float32) * 0.002
                gradient = jnp.stack(
                    [
                        sample(p + offsets[i], hit)[0] - sample(p - offsets[i], hit)[0]
                        for i in range(3)
                    ],
                    axis=-1,
                )
                normal = unit(gradient)
                diffuse = jnp.maximum(0, jnp.sum(normal * light, axis=-1))
                rim = jnp.maximum(0, jnp.sum(normal * -ray, axis=-1))
                level = jnp.minimum(1, 0.24 + 0.66 * diffuse + 0.10 * rim)
                finish = material % len(palette)
                base = palette[finish]
                shaded = base * level[..., None]
                if has_materials:
                    half_vector = unit(light - ray)
                    gloss = jnp.maximum(0, jnp.sum(normal * half_vector, axis=-1)) ** (
                        2 / jnp.maximum(rough[finish] ** 2, 0.025)
                    )
                    highlight = gloss * (0.10 + 0.28 * metal[finish]) * (1 - 0.35 * rough[finish])
                    specular = (
                        255 * (1 - metal[finish][..., None]) + base * metal[finish][..., None]
                    )
                    shaded = jnp.minimum(255, shaded + highlight[..., None] * specular)
                return jnp.where(hit[..., None], shaded, background).astype(jnp.uint8)

            return jax.lax.cond(jnp.any(hit), shade_hits, lambda: background.astype(jnp.uint8))

        return jax.jit(frame)

    def render_png(self, width, height, eye, target, fov, camera_size, far):
        if not 1 <= width <= 640 or not 1 <= height <= 480:
            raise ValueError("image size must be at most 640 × 480")
        if len(eye) != 3 or len(target) != 3:
            raise ValueError("eye and target must have three coordinates")
        if any(not isfinite(float(x)) for x in (*eye, *target, fov, camera_size, far)):
            raise ValueError("camera values must be finite")
        if sum((a - b) ** 2 for a, b in zip(eye, target, strict=True)) < 1e-8:
            raise ValueError("eye and target must be distinct 3D points")
        if fov != 0 and not 1 <= fov <= 120:
            raise ValueError("fov must be zero or between 1 and 120 degrees")
        if camera_size <= 0 or far <= 0:
            raise ValueError("camera size and far distance must be positive")
        size = (width, height)
        if size not in self.compiled:
            self.compiled[size] = self._compile(width, height)
        frame = self.compiled[size]
        eye = jnp.asarray(eye, dtype=jnp.float32)
        target = jnp.asarray(target, dtype=jnp.float32)
        fov, camera_size, far = jnp.float32(fov), jnp.float32(camera_size), jnp.float32(far)
        tile_limit = 16 if _custom_count(self.scene) >= 8 else 32
        tile_height, tile_width = min(height, tile_limit), min(width, tile_limit)
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        for row in range(0, height, tile_height):
            for col in range(0, width, tile_width):
                tile = np.asarray(frame(eye, target, fov, camera_size, far, row, col))
                pixels[row : row + tile_height, col : col + tile_width] = tile[
                    : min(tile_height, height - row), : min(tile_width, width - col)
                ]
        output = io.BytesIO()
        Image.fromarray(pixels, "RGB").save(output, format="PNG")
        return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--eye", nargs=3, type=float, default=(4, -6, 3))
    parser.add_argument("--target", nargs=3, type=float, default=(0, 0, 0))
    parser.add_argument("--fov", type=float, default=40)
    parser.add_argument("--camera-size", type=float, default=2)
    parser.add_argument("--far", type=float, default=20)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument(
        "--no-bvh", dest="bvh", action="store_false", help="disable BVH render pruning"
    )
    args = parser.parse_args()
    try:
        source, scene, materials = load_scene(args.source)
        kernel = ViewerKernel(scene, materials, bvh=args.bvh)
        if args.port not in range(1, 65536):
            raise ValueError("port must be between 1 and 65535")
        if any(
            not isfinite(value)
            for value in (*args.eye, *args.target, args.fov, args.camera_size, args.far)
        ):
            raise ValueError("camera values must be finite")
        if sum((a - b) ** 2 for a, b in zip(args.eye, args.target, strict=True)) < 1e-8:
            raise ValueError("eye and target must be distinct")
        if args.fov != 0 and not 1 <= args.fov <= 120:
            raise ValueError("fov must be zero or between 1 and 120 degrees")
        if args.camera_size <= 0 or args.far <= 0:
            raise ValueError("camera size and far distance must be positive")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    config = {
        "name": source.stem.replace("_", " "),
        "materials": materials,
        "eye": args.eye,
        "target": args.target,
        "fov": args.fov,
        "cameraSize": args.camera_size,
        "far": args.far,
    }
    html_path = Path(__file__).with_name("viewer.html")
    render_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                self.send_bytes(200, html_path.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/config":
                self.send_bytes(200, json.dumps(config).encode(), "application/json")
            elif self.path == "/favicon.ico":
                self.send_bytes(204, b"", "image/x-icon")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/render":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(length))
                with render_lock:
                    png = kernel.render_png(
                        int(request["width"]),
                        int(request["height"]),
                        request["eye"],
                        request["target"],
                        float(request["fov"]),
                        float(request["cameraSize"]),
                        float(request["far"]),
                    )
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                self.send_bytes(400, str(exc).encode(), "text/plain; charset=utf-8")
                return
            except Exception as exc:
                self.log_error("render failed: %s", exc)
                self.send_bytes(500, str(exc).encode(), "text/plain; charset=utf-8")
                return
            self.send_bytes(200, png, "image/png")

        def log_message(self, format, *args):
            if len(args) < 2 or args[1] != "200":
                super().log_message(format, *args)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        parser.error(str(exc))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Viewing {source}\n{url}\nPress Ctrl-C to stop", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
