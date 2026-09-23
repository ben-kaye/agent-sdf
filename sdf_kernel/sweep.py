"""Sweep a 2D field along a sampled spatial curve."""

from __future__ import annotations

from itertools import product
from math import isfinite, sqrt

import jax.numpy as jnp

from .dsl import Curve, Shape


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


def _frames(points, up, closed):
    tangents = tuple(
        _unit(tuple(b[i] - a[i] for i in range(3)))
        for a, b in zip(points, points[1:], strict=False)
    )
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
        candidate = tuple(float(x) for x in up)
        if len(candidate) != 3 or not all(isfinite(x) for x in candidate):
            raise ValueError("up must have three finite coordinates")
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


def sweep(profile: Shape, curve: Curve, *, tolerance=0.01, up=None, closed=False) -> Shape:
    """Approximate a transported profile sweep with mitered straight spans.

    The returned field preserves solid sign and materials. At bends its value is
    an implicit distance estimate, rather than an exact Euclidean distance.
    Near-reversals and closed paths requiring extra frame twist are rejected.
    """
    if not isinstance(profile, Shape) or profile.dim != 2:
        raise ValueError("sweep needs a 2D profile")
    if not isinstance(curve, Curve) or not curve.segments:
        raise ValueError("sweep needs a nonempty Curve")
    tolerance = float(tolerance)
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("sweep tolerance must be positive and finite")
    if not isinstance(closed, bool):
        raise ValueError("closed must be a boolean")
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
    points = tuple(points)
    frames = _frames(points, up, closed)
    return Shape("profile_sweep", 3, (points, frames, tolerance, closed), (profile,))


def evaluate_sweep(node, q, evaluate):
    """Evaluate the union of mitered swept spans for a JAX point array."""
    points, frames = node.data[:2]
    tolerance = node.data[2]
    closed = node.data[3]
    start_normals, end_normals = _miters(frames, closed)
    origins = jnp.asarray(points[:-1])
    basis_u = jnp.asarray([frame[0] for frame in frames])
    basis_v = jnp.asarray([frame[1] for frame in frames])
    relative = q[..., None, :] - origins
    local = jnp.stack(
        (jnp.sum(relative * basis_u, axis=-1), jnp.sum(relative * basis_v, axis=-1)), axis=-1
    )
    profile_distance, material = evaluate(node.children[0], local)
    overlap = tolerance * 0.01
    start_pad = jnp.asarray([overlap if closed or i else 0.0 for i in range(len(frames))])
    end_pad = jnp.asarray(
        [overlap if closed or i + 1 < len(frames) else 0.0 for i in range(len(frames))]
    )
    start = -jnp.sum(relative * jnp.asarray(start_normals), axis=-1) - start_pad
    end = (
        jnp.sum((q[..., None, :] - jnp.asarray(points[1:])) * jnp.asarray(end_normals), axis=-1)
        - end_pad
    )
    cap = jnp.maximum(start, end)
    # Exact prism combination for a straight span. At bends the same formula
    # keeps the correct solid sign, though its magnitude is only an estimate.
    outside = jnp.sqrt(jnp.maximum(profile_distance, 0) ** 2 + jnp.maximum(cap, 0) ** 2)
    distance = outside + jnp.minimum(jnp.maximum(profile_distance, cap), 0)
    selected = jnp.argmin(distance, axis=-1)
    minimum = jnp.min(distance, axis=-1)
    chosen_material = jnp.take_along_axis(material, selected[..., None], axis=-1)[..., 0]
    return minimum, chosen_material


def sweep_bounds(node, level, analytic):
    """Conservative bounds for the sweep evaluator's sublevel set."""
    from .bounds import AABB

    points, frames = node.data[:2]
    tolerance = node.data[2]
    closed = node.data[3]
    start_normals, end_normals = _miters(frames, closed)
    padding = max(0.0, level)
    profile_box = analytic(node.children[0], padding)
    if profile_box.is_empty:
        return AABB((float("inf"),) * 3, (-float("inf"),) * 3)
    if not all(isfinite(x) for x in profile_box.minimum + profile_box.maximum):
        return AABB((-float("inf"),) * 3, (float("inf"),) * 3)
    lower = [float("inf")] * 3
    upper = [-float("inf")] * 3
    for index, (a, b, (u, v, tangent), start_normal, end_normal) in enumerate(
        zip(points[:-1], points[1:], frames, start_normals, end_normals, strict=True)
    ):
        length = sqrt(sum((b[i] - a[i]) ** 2 for i in range(3)))
        start_slope = _dot(tangent, start_normal)
        end_slope = _dot(tangent, end_normal)
        start_pad = tolerance * 0.01 if closed or index else 0.0
        end_pad = tolerance * 0.01 if closed or index + 1 < len(frames) else 0.0
        for x, y in product(
            (profile_box.minimum[0], profile_box.maximum[0]),
            (profile_box.minimum[1], profile_box.maximum[1]),
        ):
            radial = tuple(x * u[i] + y * v[i] for i in range(3))
            start_s = (-padding - start_pad - _dot(radial, start_normal)) / start_slope
            end_s = length + (padding + end_pad - _dot(radial, end_normal)) / end_slope
            for s in (start_s, end_s):
                for i in range(3):
                    coordinate = a[i] + radial[i] + s * tangent[i]
                    lower[i] = min(lower[i], coordinate)
                    upper[i] = max(upper[i], coordinate)
    return AABB(tuple(lower), tuple(upper))
