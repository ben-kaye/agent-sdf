"""Serve an analytic GLSL scene in a local WebGL2 viewport."""

import argparse
import importlib
import json
import sys
import threading
import traceback
import webbrowser
from contextlib import suppress
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import isfinite
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

from .bounds import as_aabb
from .glsl import compile_glsl, feature_names
from .materials import material_arrays
from .query import ray_cast
from .scene_loader import load_scene


def _compile_scene(source_path, args):
    loaded = load_scene(source_path)
    source, scene, materials = loaded.source, loaded.scene, loaded.materials
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
    shader = compile_glsl(loaded.compiled, bvh=args.bvh)
    elapsed = perf_counter() - started
    colors, metal, rough, opacity = material_arrays(materials)
    config = {
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
        "featureNames": feature_names(loaded.compiled),
    }
    return config, shader, loaded.compiled, elapsed


def _local_modules(source_dir, excluded):
    modules = {}
    for name, module in tuple(sys.modules.items()):
        if name in excluded:
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and path.is_relative_to(source_dir):
            modules[name] = path
    return modules


def _file_stamp(path):
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def _format_error(exc):
    return "".join(traceback.format_exception(exc, limit=8))[-4000:]


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
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    source = Path(args.source).resolve()
    if not source.is_file():
        parser.error(f"scene file does not exist: {source}")

    state = {
        "revision": 0,
        "config": None,
        "shader": None,
        "query_scene": None,
        "error": None,
        "compiling": False,
    }
    state_lock = threading.Lock()
    stop_watching = threading.Event()
    excluded_modules = set(sys.modules)
    local_modules = {}
    watched_paths = {source}

    def recompile():
        nonlocal local_modules, watched_paths
        with state_lock:
            state["compiling"] = True
        try:
            for name in local_modules:
                module = sys.modules.pop(name, None)
                cached = getattr(module, "__cached__", None)
                if cached:
                    with suppress(OSError):
                        Path(cached).unlink(missing_ok=True)
            importlib.invalidate_caches()
            config, shader, query_scene, elapsed = _compile_scene(source, args)
            # Validate the payload before publishing it to HTTP clients.
            json.dumps(config)
        except Exception as exc:
            error = _format_error(exc)
            with state_lock:
                state["error"] = error
            print(f"GLSL viewer reload failed:\n{error}", file=sys.stderr, flush=True)
        else:
            with state_lock:
                state.update(
                    revision=state["revision"] + 1,
                    config=config,
                    shader=shader,
                    query_scene=query_scene,
                    error=None,
                )
            print(
                f"Translated revision {state['revision']} in {elapsed:.2f}s "
                f"({len(shader.encode()):,} bytes)",
                flush=True,
            )
        finally:
            loaded = _local_modules(source.parent, excluded_modules)
            if state["error"] is None:
                local_modules = loaded
                watched_paths = {source, *loaded.values()}
            else:
                local_modules.update(loaded)
                watched_paths.update(loaded.values())
            with state_lock:
                state["compiling"] = False

    recompile()

    def watch_source():
        stamps = {path: _file_stamp(path) for path in watched_paths}
        while not stop_watching.wait(0.3):
            current = {path: _file_stamp(path) for path in watched_paths}
            if current != stamps:
                recompile()
                stamps = {
                    path: current[path] if path in current else _file_stamp(path)
                    for path in watched_paths
                }

    watcher = threading.Thread(target=watch_source, name="glsl-source-watcher", daemon=True)
    watcher.start()

    html = Path(__file__).with_name("glsl_viewer.html")

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status_code, payload):
            body = json.dumps(payload).encode()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if urlsplit(self.path).path != "/pick":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("invalid request length")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                origin = request["origin"]
                direction = request["direction"]
                maximum = request["max_distance"]
                if (
                    len(origin) != 3
                    or len(direction) != 3
                    or not all(
                        isinstance(v, (int, float)) and isfinite(v)
                        for v in (*origin, *direction, maximum)
                    )
                    or maximum <= 0
                ):
                    raise ValueError("origin, direction and max_distance must be finite")
                if sum(v * v for v in direction) < 1e-16:
                    raise ValueError("direction must be nonzero")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            with state_lock:
                revision = state["revision"]
                query_scene = state["query_scene"]
            if query_scene is None:
                self._json(503, {"error": "scene has not compiled successfully"})
                return
            if request.get("revision") != revision:
                self._json(409, {"error": "scene changed; pick again", "revision": revision})
                return
            try:
                hit = ray_cast(query_scene, origin, direction, max_distance=maximum)
            except Exception as exc:
                self._json(500, {"error": str(exc)})
                return
            self._json(200, {"revision": revision, "hit": asdict(hit) if hit else None})

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/":
                body, content_type = html.read_bytes(), "text/html; charset=utf-8"
            elif route == "/state":
                with state_lock:
                    body = json.dumps(
                        {key: value for key, value in state.items() if key != "query_scene"}
                    ).encode()
                content_type = "application/json"
            elif route == "/config":
                with state_lock:
                    config = state["config"]
                if config is None:
                    self.send_error(503, "scene has not compiled successfully")
                    return
                body, content_type = json.dumps(config).encode(), "application/json"
            elif route == "/scene.glsl":
                with state_lock:
                    shader = state["shader"]
                if shader is None:
                    self.send_error(503, "scene has not compiled successfully")
                    return
                body, content_type = shader.encode(), "text/plain; charset=utf-8"
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
    print(f"GLSL viewer: {source}\n{url}\nPress Ctrl-C to stop", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_watching.set()
        watcher.join(timeout=1)
        server.server_close()


if __name__ == "__main__":
    main()
