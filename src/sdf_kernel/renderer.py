"""JAX ray marcher for signed distance scenes."""

import argparse
import json
import runpy
import sys
from math import isfinite
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from .bvh import compile_bvh_sample
from .dsl import Model, Shape, _custom_count
from .materials import material_arrays, scene_materials
from .smaa import smaa


def _unit(v):
    length = jnp.linalg.norm(v, axis=-1, keepdims=True)
    if bool(jnp.any(length == 0)):
        raise ValueError("camera vectors cannot have zero length")
    return v / length


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


def _adaptive_edges(image, hit, travel, material):
    """Select both sides of visible color, silhouette, material, and depth edges."""
    edges = np.zeros(hit.shape, dtype=bool)

    def mark(a, b, first, second):
        color = np.max(np.abs(a.astype(np.int16) - b.astype(np.int16)), axis=-1) > 12
        silhouette = first[0] != second[0]
        surfaces = first[0] & second[0]
        materials = surfaces & (first[2] != second[2])
        depth = surfaces & (
            np.abs(first[1] - second[1]) > np.maximum(0.02, 0.05 * np.minimum(first[1], second[1]))
        )
        return color | silhouette | materials | depth

    horizontal = mark(
        image[:, :-1],
        image[:, 1:],
        (hit[:, :-1], travel[:, :-1], material[:, :-1]),
        (hit[:, 1:], travel[:, 1:], material[:, 1:]),
    )
    edges[:, :-1] |= horizontal
    edges[:, 1:] |= horizontal
    vertical = mark(
        image[:-1],
        image[1:],
        (hit[:-1], travel[:-1], material[:-1]),
        (hit[1:], travel[1:], material[1:]),
    )
    edges[:-1] |= vertical
    edges[1:] |= vertical
    return edges


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
):
    """Render a 3D scene with full, adaptive, or postprocess antialiasing."""
    if isinstance(scene, Model):
        materials = scene_materials(scene, materials)
        scene = scene.scene()
    if not isinstance(scene, Shape) or scene.dim != 3:
        raise ValueError("render needs a 3D Shape or Model")
    if not 1 <= width <= 2048 or not 1 <= height <= 2048:
        raise ValueError("width and height must be from 1 to 2048")
    if not isfinite(camera_size) or camera_size <= 0:
        raise ValueError("camera_size must be finite and positive")
    if not isfinite(far) or far <= 0:
        raise ValueError("far must be finite and positive")
    if aa not in ("auto", "none", "fxaa", "smaa", "ssaa", "adaptive"):
        raise ValueError("aa must be auto, none, fxaa, smaa, ssaa, or adaptive")
    aa_mode = ("ssaa" if final else "none") if aa == "auto" else aa
    if not isfinite(fov) or (fov != 0 and not 1 <= fov <= 120):
        raise ValueError("fov must be zero or between 1 and 120 degrees")
    orthographic = fov == 0

    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if eye.shape != (3,) or target.shape != (3,):
        raise ValueError("eye and target must have three coordinates")
    if not np.all(np.isfinite(eye)) or not np.all(np.isfinite(target)):
        raise ValueError("eye and target must be finite")
    if np.array_equal(eye, target):
        raise ValueError("eye and target must be distinct")
    eye = jnp.asarray(eye)
    target = jnp.asarray(target)
    forward = _unit(target - eye)
    horizontal = float(jnp.linalg.norm(forward[:2]))
    right = (
        jnp.asarray((forward[1] / horizontal, -forward[0] / horizontal, 0))
        if horizontal > 1e-8
        else jnp.asarray((1.0, 0.0, 0.0))
    )
    up = jnp.cross(right, forward)
    spread = camera_size / 2 if orthographic else jnp.tan(jnp.deg2rad(fov) / 2)
    scale = 2 if aa_mode == "ssaa" else 1
    tile_limit = (
        96
        if final or aa_mode in ("fxaa", "smaa", "adaptive")
        else 16
        if _custom_count(scene) >= 8
        else 32
    )
    tile_height, tile_width = min(height, tile_limit), min(width, tile_limit)
    raster_height, raster_width = tile_height * scale, tile_width * scale
    light = jnp.asarray((0.55, -0.45, 0.7), dtype=jnp.float32)
    light = light / jnp.linalg.norm(light)
    palette, metal, rough = material_arrays(materials)
    sample = (
        compile_bvh_sample(scene, np.asarray(target))
        if bvh
        else lambda points, active=None: scene.sample(points)
    )

    def trace(rows, cols, shade_scale):
        sample_shape = rows.shape
        sx = (2 * (cols + 0.5) / width - 1) * spread * width / height
        sy = (1 - 2 * (rows + 0.5) / height) * spread
        displacement = sx[..., None] * right + sy[..., None] * up
        origin = eye + displacement if orthographic else jnp.broadcast_to(eye, sample_shape + (3,))
        direction = forward + displacement
        ray = (
            jnp.broadcast_to(forward, origin.shape)
            if orthographic
            else direction / jnp.maximum(jnp.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
        )
        travel = jnp.zeros(sample_shape, dtype=jnp.float32)
        hit = jnp.zeros(sample_shape, dtype=bool)
        material = jnp.zeros(sample_shape, dtype=jnp.int32)
        active = jnp.ones(sample_shape, dtype=bool)

        def step(state):
            count, travel, hit, material, active = state
            p = origin + travel[..., None] * ray
            distance, sampled_material = sample(p, active)
            found = active & (distance < 0.001)
            hit = hit | found
            material = jnp.where(found, sampled_material.astype(jnp.int32), material)
            next_travel = travel + jnp.maximum(distance * 0.75, 0.0005)
            travel = jnp.where(active & ~found, next_travel, travel)
            active = active & ~found & (travel <= far)
            return count + 1, travel, hit, material, active

        def marching(state):
            count, _, _, _, active = state
            return (count < 96) & jnp.any(active)

        _, travel, hit, material, _ = jax.lax.while_loop(
            marching, step, (0, travel, hit, material, active)
        )
        shade = ((rows + 0.5) * shade_scale - 0.5) / max(height * shade_scale - 1, 1)
        background = jnp.stack((244 - 24 * shade, 244 - 21 * shade, 240 - 16 * shade), axis=-1)

        def shade_hits():
            p = origin + travel[..., None] * ray
            offsets = jnp.eye(3, dtype=jnp.float32) * 0.002
            gradients = []
            for axis in range(3):
                positive, _ = sample(p + offsets[axis], hit)
                negative, _ = sample(p - offsets[axis], hit)
                gradients.append(positive - negative)
            normal = jnp.stack(gradients, axis=-1)
            normal = normal / jnp.maximum(jnp.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
            diffuse = jnp.maximum(0, jnp.sum(normal * light, axis=-1))
            rim = jnp.maximum(0, jnp.sum(normal * -ray, axis=-1))
            if final:
                occlusion = jnp.zeros_like(diffuse)
                for radius, weight in ((0.06, 0.5), (0.16, 0.3), (0.35, 0.2)):
                    distance, _ = sample(p + normal * radius, hit)
                    occlusion += weight * jnp.maximum(radius - distance, 0) / radius
                ambient = jnp.clip(1 - 0.7 * occlusion, 0.45, 1)

                shadow_origin = p + normal * 0.01

                def shadow_step(_, state):
                    distance_along, visibility = state
                    distance, _ = sample(shadow_origin + distance_along[..., None] * light, hit)
                    visibility = jnp.minimum(
                        visibility,
                        jnp.clip(10 * distance / jnp.maximum(distance_along, 0.01), 0, 1),
                    )
                    distance_along += jnp.maximum(distance * 0.75, 0.025)
                    return distance_along, visibility

                _, shadow = jax.lax.fori_loop(
                    0, 16, shadow_step, (jnp.full_like(diffuse, 0.03), jnp.ones_like(diffuse))
                )
                shadow = jnp.clip(shadow, 0.3, 1)
                level = jnp.minimum(1, 0.24 * ambient + 0.66 * diffuse * shadow + 0.10 * rim)
            else:
                shadow = 1
                level = jnp.minimum(1, 0.24 + 0.66 * diffuse + 0.10 * rim)
            finish = material % len(palette)
            base = palette[finish]
            shaded = base * level[..., None]
            if materials:
                half_vector = light - ray
                half_vector = half_vector / jnp.maximum(
                    jnp.linalg.norm(half_vector, axis=-1, keepdims=True), 1e-12
                )
                gloss = jnp.maximum(0, jnp.sum(normal * half_vector, axis=-1)) ** (
                    2 / jnp.maximum(rough[finish] ** 2, 0.025)
                )
                highlight = (
                    gloss * (0.10 + 0.28 * metal[finish]) * (1 - 0.35 * rough[finish]) * shadow
                )
                specular_color = (
                    255 * (1 - metal[finish][..., None]) + base * metal[finish][..., None]
                )
                shaded = jnp.minimum(255, shaded + highlight[..., None] * specular_color)
            return jnp.where(hit[..., None], shaded, background).astype(jnp.uint8)

        rgb = jax.lax.cond(jnp.any(hit), shade_hits, lambda: background.astype(jnp.uint8))
        return rgb, hit, travel, material

    def trace_tile(row_start, col_start):
        rows, cols = jnp.meshgrid(
            row_start + (jnp.arange(raster_height) + 0.5) / scale - 0.5,
            col_start + (jnp.arange(raster_width) + 0.5) / scale - 0.5,
            indexing="ij",
        )
        return trace(rows, cols, scale)

    trace_tile = jax.jit(trace_tile)
    image = np.empty((height, width, 3), dtype=np.uint8)
    if aa_mode == "adaptive":
        hits = np.empty((height, width), dtype=bool)
        depths = np.empty((height, width), dtype=np.float32)
        material_ids = np.empty((height, width), dtype=np.int32)
    for row in range(0, height, tile_height):
        for col in range(0, width, tile_width):
            samples, hit, travel, material = trace_tile(row, col)
            samples = np.asarray(samples)
            pixels = (
                samples.reshape(tile_height, scale, tile_width, scale, 3).mean(axis=(1, 3))
                if scale == 2
                else samples
            )
            valid_height = min(tile_height, height - row)
            valid_width = min(tile_width, width - col)
            region = np.s_[row : row + valid_height, col : col + valid_width]
            image[region] = pixels[:valid_height, :valid_width]
            if aa_mode == "adaptive":
                hits[region] = np.asarray(hit)[:valid_height, :valid_width]
                depths[region] = np.asarray(travel)[:valid_height, :valid_width]
                material_ids[region] = np.asarray(material)[:valid_height, :valid_width]
    if aa_mode == "adaptive":
        edge_rows, edge_cols = np.nonzero(_adaptive_edges(image, hits, depths, material_ids))
        trace_edges = jax.jit(lambda rows, cols: trace(rows, cols, 2)[0])
        offsets_row = np.array((-0.25, -0.25, 0.25, 0.25), dtype=np.float32)
        offsets_col = np.array((-0.25, 0.25, -0.25, 0.25), dtype=np.float32)
        batch_size = 512
        for start in range(0, len(edge_rows), batch_size):
            end = min(start + batch_size, len(edge_rows))
            count = end - start
            rows = np.repeat(edge_rows[start:end].astype(np.float32), 4) + np.tile(
                offsets_row, count
            )
            cols = np.repeat(edge_cols[start:end].astype(np.float32), 4) + np.tile(
                offsets_col, count
            )
            if count < batch_size:
                rows = np.pad(rows, (0, 4 * (batch_size - count)), mode="edge")
                cols = np.pad(cols, (0, 4 * (batch_size - count)), mode="edge")
            samples = trace_edges(jnp.asarray(rows), jnp.asarray(cols))
            image[edge_rows[start:end], edge_cols[start:end]] = (
                np.asarray(samples)[: 4 * count].reshape(count, 4, 3).mean(axis=1)
            )
    if aa_mode == "fxaa":
        image = _fxaa(image)
    elif aa_mode == "smaa":
        image = smaa(image)
    return Image.fromarray(image, "RGB")


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
    parser.add_argument(
        "--final",
        action="store_true",
        help="ambient occlusion, soft shadows, and 2x antialiasing by default",
    )
    parser.add_argument(
        "--aa",
        choices=("auto", "none", "fxaa", "smaa", "ssaa", "adaptive"),
        default="auto",
        help="antialiasing; smaa is single-frame, adaptive traces edge rays, auto uses 2x SSAA with --final",
    )
    parser.add_argument(
        "--no-bvh", dest="bvh", action="store_false", help="disable BVH render pruning"
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
