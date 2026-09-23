"""Serve an analytic GLSL scene in a local WebGL2 viewport."""

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import isfinite
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

from .bounds import as_aabb
from .glsl import compile_glsl
from .materials import material_arrays
from .viewer import load_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--eye", nargs=3, type=float, default=(4, -6, 3))
    parser.add_argument("--target", nargs=3, type=float, default=(0, 0, 0))
    parser.add_argument("--fov", type=float, default=40)
    parser.add_argument("--camera-size", type=float, default=2)
    parser.add_argument("--far", type=float, help="fixed ray distance (default: auto)")
    parser.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="conservative scene bounds for fields without analytic bounds",
    )
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--bvh", action="store_true", help="prune bounded union components on GPU")
    args = parser.parse_args()
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not all(isfinite(v) for v in (*args.eye, *args.target, args.fov, args.camera_size)) or (
            args.far is not None and not isfinite(args.far)
        ):
            raise ValueError("camera values must be finite")
        if sum((a - b) ** 2 for a, b in zip(args.eye, args.target, strict=True)) < 1e-8:
            raise ValueError("eye and target must be distinct")
        if args.fov != 0 and not 1 <= args.fov <= 120:
            raise ValueError("fov must be zero or between 1 and 120 degrees")
        if args.camera_size <= 0 or (args.far is not None and args.far <= 0):
            raise ValueError("camera size and far distance must be positive")
        source, scene, materials = load_scene(args.source)
        if args.bounds is not None:
            bounds = as_aabb((args.bounds[:3], args.bounds[3:]), 3)
        else:
            try:
                bounds = scene.bounds()
            except ValueError:
                bounds = None
        if bounds is not None and (
            bounds.is_empty or not all(isfinite(v) for v in (*bounds.minimum, *bounds.maximum))
        ):
            bounds = None
        started = perf_counter()
        shader = compile_glsl(scene, bvh=args.bvh).encode()
        elapsed = perf_counter() - started
        colors, metal, rough, opacity = material_arrays(materials)
        config = json.dumps(
            {
                "name": source.stem.replace("_", " "),
                "materials": materials,
                "eye": args.eye,
                "target": args.target,
                "fov": args.fov,
                "cameraSize": args.camera_size,
                "far": 50 if args.far is None else args.far,
                "autoFar": args.far is None,
                "bounds": [bounds.minimum, bounds.maximum] if bounds is not None else None,
                "palette": colors.tolist(),
                "metallic": metal.tolist(),
                "roughness": rough.tolist(),
                "opacity": opacity.tolist(),
                "translationMs": round(elapsed * 1000),
            }
        ).encode()
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    html = Path(__file__).with_name("glsl_viewer.html")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/":
                body, content_type = html.read_bytes(), "text/html; charset=utf-8"
            elif route == "/config":
                body, content_type = config, "application/json"
            elif route == "/scene.glsl":
                body, content_type = shader, "text/plain; charset=utf-8"
            elif route == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            if len(args) < 2 or args[1] != "200":
                super().log_message(format, *args)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        parser.error(str(exc))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(
        f"GLSL viewer: {source}\nTranslated in {elapsed:.2f}s ({len(shader):,} bytes)"
        f"\n{url}\nPress Ctrl-C to stop",
        flush=True,
    )
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
