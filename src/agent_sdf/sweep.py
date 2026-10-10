"""Sweep a 2D field along a sampled spatial curve."""

from __future__ import annotations

from itertools import product
from math import isfinite, sqrt
from typing import NamedTuple

import numpy as np

from .curve import Curve
from .field import Field, _point
from .nodes import Sweep
from .transform import Transform

_ORIENTATIONS = ("perpendicular", "parallel")


class _Span(NamedTuple):
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    # Profile-local coordinates are (dot(r, u), dot(r, v)) for r = p - start.
    u: tuple[float, float, float]
    v: tuple[float, float, float]
    # The profile embeds as start + x * axes[0] + y * axes[1] + s * axes[2].
    axes: tuple
    start_normal: tuple[float, float, float]
    end_normal: tuple[float, float, float]
    # Scales profile distances so sheared sampling stays 1-Lipschitz.
    scale: float
    start_pad: float
    end_pad: float


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(a):
    length = sqrt(_dot(a, a))
    if length < 1e-12:
        raise ValueError("sweep curve has a zero-length or unresolved segment")
    return tuple(x / length for x in a)


def _lift(point):
    return tuple(point) if len(point) == 3 else (point[0], point[1], 0.0)


def _rotate_minimal(vector, before, after):
    cosine = max(-1.0, min(1.0, _dot(before, after)))
    if cosine <= -0.95:
        raise ValueError("sweep curve turns too sharply for a finite miter")
    axis = _cross(before, after)
    sine = sqrt(_dot(axis, axis))
    if sine < 1e-12:
        return vector
    axis = tuple(x / sine for x in axis)
    along = _cross(axis, vector)
    return tuple(
        vector[i] * cosine + along[i] * sine + axis[i] * _dot(axis, vector) * (1 - cosine)
        for i in range(3)
    )


def _tangents(points):
    return tuple(
        _unit(tuple(b[i] - a[i] for i in range(3)))
        for a, b in zip(points, points[1:], strict=False)
    )


def _transported(points, up, closed):
    tangents = _tangents(points)
    if len(tangents) > 1:
        for a, b in zip(tangents, tangents[1:], strict=False):
            if _dot(a, b) <= -0.95:
                raise ValueError("sweep curve turns too sharply for a finite miter")
        if closed and _dot(tangents[-1], tangents[0]) <= -0.95:
            raise ValueError("closed sweep turns too sharply at its seam")
    first = tangents[0]
    if up is None:
        candidate = next(
            axis
            for axis in ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
            if abs(_dot(axis, first)) < 0.9
        )
    else:
        candidate = _point(up, 3)
    projection = tuple(candidate[i] - _dot(candidate, first) * first[i] for i in range(3))
    if _dot(projection, projection) < 1e-24:
        raise ValueError("up must not be parallel to the initial curve tangent")
    # ``up`` specifies the profile's local Y axis; local X completes a
    # right-handed (X, Y, tangent) frame.
    axes = [_unit(_cross(projection, first))]
    for before, after in zip(tangents, tangents[1:], strict=False):
        axes.append(_unit(_rotate_minimal(axes[-1], before, after)))
    if closed:
        # Holonomy requires a gradual twist along the spans. A fixed frame per
        # span cannot represent that without opening a seam in arbitrary profiles.
        returned = _rotate_minimal(axes[-1], tangents[-1], tangents[0])
        if _dot(returned, axes[0]) < 1 - 1e-8:
            raise ValueError("closed sweep has frame holonomy; use a planar path")
    return tuple(
        (axis, _cross(tangent, axis), tangent) for axis, tangent in zip(axes, tangents, strict=True)
    )


def _miters(frames, closed):
    tangents = [frame[2] for frame in frames]
    joints = [
        _unit(tuple(a + b for a, b in zip(tangents[i], tangents[i + 1], strict=True)))
        for i in range(len(tangents) - 1)
    ]
    if closed:
        seam = _unit(tuple(a + b for a, b in zip(tangents[-1], tangents[0], strict=True)))
        return (seam, *joints), (*joints, seam)
    return (tangents[0], *joints), (*joints, tangents[-1])


def _parallel(points, x_axis, y_axis):
    normal = _cross(x_axis, y_axis)
    tangents = _tangents(points)
    if any(abs(_dot(tangent, normal)) < 1e-6 for tangent in tangents):
        raise ValueError("a parallel sweep path must not run along the profile plane")
    return tuple((x_axis, y_axis, tangent) for tangent in tangents)


def _path(curve, tolerance, closed):
    points = []
    for sample in curve.sample(tolerance):
        point = _lift(sample)
        if not points or sqrt(sum((point[i] - points[-1][i]) ** 2 for i in range(3))) > 1e-10:
            points.append(point)
    if len(points) < 2:
        raise ValueError("sweep curve has no length")
    if closed:
        gap = sqrt(sum((points[0][i] - points[-1][i]) ** 2 for i in range(3)))
        if gap > max(1e-8, tolerance * 1e-3):
            points.append(points[0])
        else:
            points[-1] = points[0]
        if len(points) < 4:
            raise ValueError("closed sweep needs at least three spans")
    elif points[0] == points[-1]:
        raise ValueError("a looped sweep path requires closed=True")
    if len(points) - 1 > 8192:
        raise ValueError("sweep curve needs more than 8192 spans")
    return tuple(points)


def sweep(
    profile: Field,
    curve: Curve,
    *,
    tolerance=0.01,
    up=None,
    closed=False,
    orientation="perpendicular",
    plane=None,
) -> Field:
    """Approximate a profile sweep with straight spans.

    Perpendicular sweeps transport the profile frame and miter the joins;
    parallel sweeps translate the profile without turning it. Without a plane,
    the profile origin follows the curve; with one, the profile stays in place
    and the curve must start on that plane. The returned field preserves solid
    sign and materials, but its value is an implicit distance estimate.
    """
    if not isinstance(profile, Field) or profile.dim != 2:
        raise ValueError("sweep needs a 2D profile")
    if not isinstance(curve, Curve) or not curve.segments:
        raise ValueError("sweep needs a nonempty Curve")
    tolerance = float(tolerance)
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("sweep tolerance must be positive and finite")
    if not isinstance(closed, bool):
        raise ValueError("closed must be a boolean")
    if orientation not in _ORIENTATIONS:
        raise ValueError("sweep orientation must be perpendicular or parallel")
    if orientation == "parallel" and up is not None:
        raise ValueError("up only applies to perpendicular sweeps")
    points = _path(curve, tolerance, closed)
    x_axis, y_axis = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    if plane is not None:
        try:
            u, v = plane.world_to_local(points[0], tolerance=max(1e-8, tolerance * 1e-3))
        except ValueError as exc:
            raise ValueError("sweep curve must start on the sketch plane") from exc
        profile = profile.translate(-u, -v)
        x_axis, y_axis = plane.x_direction, plane.y_direction
        if orientation == "perpendicular":
            leaving = _dot(_tangents(points[:2])[0], plane.normal)
            if abs(leaving) < 1e-6:
                raise ValueError("sweep curve must leave the sketch plane")
            if leaving < 0:
                # The transported frame flips local X behind the plane.
                profile = profile.transform(Transform.scaling(-1, 1))
            up = y_axis
    if orientation == "perpendicular":
        frames = _transported(points, up, closed)
    else:
        frames = _parallel(points, x_axis, y_axis)
    from .bounds import _analytic, clip_boxes

    data = Sweep(points, frames, tolerance, closed, orientation)
    bounds = clip_boxes(_span_boxes(_spans(data), _analytic(profile), 0.0), 8)
    return Field("profile_sweep", 3, data._replace(bounds=bounds), (profile,))


def _spans(data):
    points, frames, closed = data.points, data.frames, data.closed
    overlap = data.tolerance * 0.01
    count = len(frames)
    if data.orientation == "perpendicular":
        start_normals, end_normals = _miters(frames, closed)
        samplers = [(x, y, 1.0) for x, y, _ in frames]
    else:
        normal = _cross(frames[0][0], frames[0][1])
        start_normals, end_normals, samplers = [], [], []
        for x, y, tangent in frames:
            slope = _dot(tangent, normal)
            # Shear along the span so every cross-section samples the same profile.
            u = tuple(x[i] - normal[i] * _dot(tangent, x) / slope for i in range(3))
            v = tuple(y[i] - normal[i] * _dot(tangent, y) / slope for i in range(3))
            facing = tuple(value if slope > 0 else -value for value in normal)
            start_normals.append(facing)
            end_normals.append(facing)
            samplers.append((u, v, abs(slope)))
    return tuple(
        _Span(
            a,
            b,
            u,
            v,
            frame,
            start_normal,
            end_normal,
            scale,
            overlap if closed or i else 0.0,
            overlap if closed or i + 1 < count else 0.0,
        )
        for i, (a, b, frame, start_normal, end_normal, (u, v, scale)) in enumerate(
            zip(
                points[:-1],
                points[1:],
                frames,
                start_normals,
                end_normals,
                samplers,
                strict=True,
            )
        )
    )


def evaluate_sweep(node, q, evaluate):
    """Evaluate the union of capped swept spans for a NumPy point array."""
    spans = _spans(node.data)

    def column(name):
        return np.asarray([getattr(span, name) for span in spans])

    relative = q[..., None, :] - column("start")
    local = np.stack(
        (np.sum(relative * column("u"), axis=-1), np.sum(relative * column("v"), axis=-1)),
        axis=-1,
    )
    profile_distance, material = evaluate(node.children[0], local)
    profile_distance = profile_distance * column("scale")
    start = -np.sum(relative * column("start_normal"), axis=-1) - column("start_pad")
    end = np.sum((q[..., None, :] - column("end")) * column("end_normal"), axis=-1) - column(
        "end_pad"
    )
    cap = np.maximum(start, end)
    # Exact prism combination for a straight perpendicular span. At bends and
    # sheared spans the same formula keeps the solid sign; magnitude is an estimate.
    outside = np.sqrt(np.maximum(profile_distance, 0) ** 2 + np.maximum(cap, 0) ** 2)
    distance = outside + np.minimum(np.maximum(profile_distance, cap), 0)
    selected = np.argmin(distance, axis=-1)
    minimum = np.min(distance, axis=-1)
    chosen_material = np.take_along_axis(material, selected[..., None], axis=-1)[..., 0]
    return minimum, chosen_material


def _span_boxes(spans, profile_box, padding):
    """Bound each capped span's sublevel set for a profile sublevel box."""
    from .bounds import AABB

    boxes = []
    for span in spans:
        lower = [float("inf")] * 3
        upper = [-float("inf")] * 3
        a, b = span.start, span.end
        x_axis, y_axis, direction = span.axes
        length = _dot(tuple(b[i] - a[i] for i in range(3)), direction)
        start_slope = _dot(direction, span.start_normal)
        end_slope = _dot(direction, span.end_normal)
        for x, y in product(
            (profile_box.minimum[0], profile_box.maximum[0]),
            (profile_box.minimum[1], profile_box.maximum[1]),
        ):
            radial = tuple(x * x_axis[i] + y * y_axis[i] for i in range(3))
            start_s = (-padding - span.start_pad - _dot(radial, span.start_normal)) / start_slope
            end_s = length + (padding + span.end_pad - _dot(radial, span.end_normal)) / end_slope
            for s in (start_s, end_s):
                for i in range(3):
                    coordinate = a[i] + radial[i] + s * direction[i]
                    lower[i] = min(lower[i], coordinate)
                    upper[i] = max(upper[i], coordinate)
        boxes.append(AABB(tuple(lower), tuple(upper)))
    return boxes


def sweep_bounds(node, level, analytic):
    """Conservative bounds for the sweep evaluator's sublevel set."""
    from .bounds import AABB, clip_bounds

    spans = _spans(node.data)
    padding = max(0.0, level)
    profile_box = analytic(node.children[0], padding / min(span.scale for span in spans))
    if profile_box.is_empty:
        return AABB((float("inf"),) * 3, (-float("inf"),) * 3)
    if not all(isfinite(x) for x in profile_box.minimum + profile_box.maximum):
        return AABB((-float("inf"),) * 3, (float("inf"),) * 3)
    box = AABB((float("inf"),) * 3, (-float("inf"),) * 3)
    for span_box in _span_boxes(spans, profile_box, padding):
        box = box.union(span_box)
    return clip_bounds(box, node.data.bounds, padding)
