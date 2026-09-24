"""Trace zero contours in a profile or on a solid's plane section."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot, isfinite

import numpy as np

from .bounds import as_aabb
from .dsl import Shape
from .model import compile_scene
from .plane import Plane


@dataclass(frozen=True)
class Contour:
    """One local or world-space polyline, with its sampled arc length."""

    points: tuple[tuple[float, ...], ...]
    closed: bool
    length: float


@dataclass(frozen=True)
class Section:
    """World-space contours traced inside the given plane bounds."""

    plane: Plane
    bounds: tuple[tuple[float, float], tuple[float, float]]
    contours: tuple[Contour, ...]


def _resolution(bounds, resolution, step):
    width = bounds.maximum[0] - bounds.minimum[0]
    height = bounds.maximum[1] - bounds.minimum[1]
    if width <= 0 or height <= 0:
        raise ValueError("contour bounds must have positive width and height")
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


def contours(profile: Shape, bounds, *, resolution=128, step=None) -> tuple[Contour, ...]:
    """Trace a 2D profile's zero level set inside finite local bounds."""
    if not isinstance(profile, Shape) or profile.dim != 2:
        raise ValueError("contours requires a 2D Shape")
    plane_bounds = as_aabb(bounds, 2)
    nx, ny = _resolution(plane_bounds, resolution, step)
    xs = np.linspace(plane_bounds.minimum[0], plane_bounds.maximum[0], nx + 1)
    ys = np.linspace(plane_bounds.minimum[1], plane_bounds.maximum[1], ny + 1)
    xx, yy = np.meshgrid(xs, ys)
    sample_points = np.stack((xx, yy), axis=-1)
    distances = np.asarray(profile.sample(sample_points)[0])
    if distances.shape != xx.shape or np.any(np.isnan(distances)):
        raise ValueError("profile returned invalid contour distances")
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
        locations[key] = tuple(float(x) for x in planar)

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
                midpoint = ((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2)
                center_inside = float(profile.sample(np.asarray(midpoint))[0]) <= 0
                for corner in range(4):
                    if signs[corner] != center_inside:
                        segments.append((keys[(corner - 1) % 4], keys[corner]))

    adjacent = {}
    for index, (a, b) in enumerate(segments):
        adjacent.setdefault(a, []).append(index)
        adjacent.setdefault(b, []).append(index)
    unused = set(range(len(segments)))
    traced = []

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
            hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:], strict=False)
        )
        traced.append(Contour(points, path[0] == path[-1], length))

    for key, connected in adjacent.items():
        if len(connected) == 1 and any(index in unused for index in connected):
            trace(key)
    while unused:
        trace(segments[min(unused)][0])

    return tuple(traced)


def section(shape_or_scene, plane: Plane, bounds, *, resolution=128, step=None) -> Section:
    """Trace a solid's zero contours in a Plane's exact local axes."""
    shape = compile_scene(shape_or_scene).shape
    if shape.dim != 3:
        raise ValueError("section requires a 3D scene")
    if not isinstance(plane, Plane):
        raise ValueError("section requires a Plane")
    plane_bounds = as_aabb(bounds, 2)
    local = shape.slice(plane).contours(plane_bounds, resolution=resolution, step=step)
    world = tuple(
        Contour(
            tuple(plane.local_to_world(point) for point in contour.points),
            contour.closed,
            contour.length,
        )
        for contour in local
    )
    return Section(plane, (plane_bounds.minimum, plane_bounds.maximum), world)
