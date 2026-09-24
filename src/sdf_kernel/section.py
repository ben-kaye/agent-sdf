"""Sample a signed distance field on a plane and trace its zero contours."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot, isfinite

import numpy as np

from .bounds import as_aabb
from .dsl import Shape
from .model import Model


@dataclass(frozen=True)
class SectionContour:
    """One world-space polyline, with its sampled arc length."""

    points: tuple[tuple[float, float, float], ...]
    closed: bool
    length: float


@dataclass(frozen=True)
class Section:
    """Zero contours and the plane coordinates used to sample them."""

    origin: tuple[float, float, float]
    normal: tuple[float, float, float]
    u_axis: tuple[float, float, float]
    v_axis: tuple[float, float, float]
    bounds: tuple[tuple[float, float], tuple[float, float]]
    contours: tuple[SectionContour, ...]


def _vector(value, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have three finite coordinates") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have three finite coordinates")
    return result


def _resolution(bounds, resolution, step):
    width = bounds.maximum[0] - bounds.minimum[0]
    height = bounds.maximum[1] - bounds.minimum[1]
    if width <= 0 or height <= 0:
        raise ValueError("section bounds must have positive width and height")
    if step is not None:
        if resolution != 128:
            raise ValueError("provide either resolution or step")
        try:
            step_size = float(step) if np.isscalar(step) else float("nan")
        except (TypeError, ValueError) as exc:
            raise ValueError("step must be a positive finite number") from exc
        if not isfinite(step_size) or step_size <= 0:
            raise ValueError("step must be a positive finite number")
        nx, ny = ceil(width / step_size), ceil(height / step_size)
    else:
        if isinstance(resolution, int):
            nx = ny = resolution
        else:
            try:
                nx, ny = resolution
            except (TypeError, ValueError) as exc:
                raise ValueError("resolution must be an integer or two integers") from exc
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (nx, ny)):
            raise ValueError("resolution must contain positive integers")
    if nx * ny > 4_000_000:
        raise ValueError("section grid exceeds four million cells")
    return nx, ny


def section(shape_or_model, origin, normal, bounds, *, resolution=128, step=None) -> Section:
    """Trace the zero level set in a finite plane rectangle.

    ``bounds`` are ((u_min, v_min), (u_max, v_max)) in the returned plane
    basis. For a +Z normal, u and v point along world +X and +Y.
    Resolution counts cells along each plane axis. Contours touching the
    sampling rectangle are open; closed contours repeat their first point.
    """
    shape = shape_or_model.scene() if isinstance(shape_or_model, Model) else shape_or_model
    if not isinstance(shape, Shape) or shape.dim != 3:
        raise ValueError("section requires a 3D Shape or Model")
    center = _vector(origin, "origin")
    n = _vector(normal, "normal")
    magnitude = np.linalg.norm(n)
    if magnitude == 0:
        raise ValueError("normal must be nonzero")
    n /= magnitude
    reference = np.array((0.0, 1.0, 0.0) if abs(n[1]) < 0.9 else (0.0, 0.0, 1.0))
    u = np.cross(reference, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    plane_bounds = as_aabb(bounds, 2)
    nx, ny = _resolution(plane_bounds, resolution, step)
    xs = np.linspace(plane_bounds.minimum[0], plane_bounds.maximum[0], nx + 1)
    ys = np.linspace(plane_bounds.minimum[1], plane_bounds.maximum[1], ny + 1)
    xx, yy = np.meshgrid(xs, ys)
    sample_points = center + xx[..., None] * u + yy[..., None] * v
    distances = np.asarray(shape.sample(sample_points)[0])
    if distances.shape != xx.shape or np.any(np.isnan(distances)):
        raise ValueError("shape returned invalid section distances")
    inside = distances <= 0

    locations = {}
    segments = []

    def edge_point(key, i, j, side):
        if key in locations:
            return
        if side == 0:
            a, b = distances[j, i], distances[j, i + 1]
            first, second = (xs[i], ys[j]), (xs[i + 1], ys[j])
        elif side == 1:
            a, b = distances[j, i + 1], distances[j + 1, i + 1]
            first, second = (xs[i + 1], ys[j]), (xs[i + 1], ys[j + 1])
        elif side == 2:
            a, b = distances[j + 1, i], distances[j + 1, i + 1]
            first, second = (xs[i], ys[j + 1]), (xs[i + 1], ys[j + 1])
        else:
            a, b = distances[j, i], distances[j + 1, i]
            first, second = (xs[i], ys[j]), (xs[i], ys[j + 1])
        if np.isinf(a):
            t = 1.0
        elif np.isinf(b):
            t = 0.0
        else:
            t = float(a / (a - b))
        planar = np.asarray(first) + t * (np.asarray(second) - first)
        locations[key] = tuple(float(x) for x in center + planar[0] * u + planar[1] * v)

    for j in range(ny):
        for i in range(nx):
            signs = (
                inside[j, i],
                inside[j, i + 1],
                inside[j + 1, i + 1],
                inside[j + 1, i],
            )
            keys = (("h", i, j), ("v", i + 1, j), ("h", i, j + 1), ("v", i, j))
            crossing = [k for k in range(4) if signs[k] != signs[(k + 1) % 4]]
            if not crossing:
                continue
            for side in crossing:
                edge_point(keys[side], i, j, side)
            if len(crossing) == 2:
                segments.append((keys[crossing[0]], keys[crossing[1]]))
            else:
                # Four crossings: connect around corners opposite the center.
                midpoint = center + ((xs[i] + xs[i + 1]) / 2) * u + ((ys[j] + ys[j + 1]) / 2) * v
                center_inside = float(shape.sample(midpoint)[0]) <= 0
                for corner in range(4):
                    if signs[corner] != center_inside:
                        segments.append((keys[(corner - 1) % 4], keys[corner]))

    adjacent = {}
    for index, (a, b) in enumerate(segments):
        adjacent.setdefault(a, []).append(index)
        adjacent.setdefault(b, []).append(index)
    unused = set(range(len(segments)))
    contours = []

    def trace(start):
        path = [start]
        current = start
        while True:
            candidates = [index for index in adjacent[current] if index in unused]
            if not candidates:
                break
            index = candidates[0]
            unused.remove(index)
            a, b = segments[index]
            current = b if current == a else a
            path.append(current)
            if current == start:
                break
        points = tuple(locations[key] for key in path)
        length = sum(
            hypot(hypot(b[0] - a[0], b[1] - a[1]), b[2] - a[2])
            for a, b in zip(points, points[1:], strict=False)
        )
        contours.append(SectionContour(points, path[0] == path[-1], length))

    for key, connected in adjacent.items():
        if len(connected) == 1 and any(index in unused for index in connected):
            trace(key)
    while unused:
        trace(segments[min(unused)][0])

    return Section(
        tuple(map(float, center)),
        tuple(map(float, n)),
        tuple(map(float, u)),
        tuple(map(float, v)),
        (plane_bounds.minimum, plane_bounds.maximum),
        tuple(contours),
    )
