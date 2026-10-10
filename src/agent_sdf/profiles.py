"""Profile lofts, field morphs, and spatially varying extrusion fields."""

import numpy as np

from .field import Field, LoftStation, _number, _positive
from .nodes import Extrude, FieldMorph, Loft, VariableExtrude

_Z_AXES = (2, (0, 1))


def _positions(values, operation):
    if len(values) < 2 or any(a >= b for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(f"{operation} needs at least two strictly increasing station positions")


def field_morph(sections) -> Field:
    """Interpolate 2D signed fields at (position, profile) sections along Z, with flat caps."""
    sections = tuple(sections)
    positions, profiles = [], []
    for position, profile in sections:
        if not isinstance(profile, Field) or profile.dim != 2:
            raise ValueError("field_morph sections must be 2D fields")
        positions.append(_number(position))
        profiles.append(profile)
    _positions(positions, "field_morph")
    return Field("field_morph", 3, FieldMorph(tuple(positions), _Z_AXES), tuple(profiles))


def loft_sketches(sketches) -> Field:
    from .plane import Plane, _dot
    from .transform import Transform

    first = sketches[0].plane
    normal = first.normal
    for sketch in sketches[1:]:
        if abs(abs(_dot(sketch.plane.normal, normal)) - 1) > 1e-9:
            raise ValueError("loft sketches must lie on parallel planes")
    if (
        _dot(
            tuple(b - a for a, b in zip(first.origin, sketches[-1].plane.origin, strict=True)),
            normal,
        )
        < 0
    ):
        normal = tuple(-value for value in normal)
    frame = Plane(first.origin, normal, x_direction=first.x_direction)
    sections = []
    for sketch in sketches:
        plane = sketch.plane
        delta = tuple(b - a for a, b in zip(frame.origin, plane.origin, strict=True))
        axes = (frame.x_direction, frame.y_direction)
        matrix = tuple(
            (_dot(plane.x_direction, axis), _dot(plane.y_direction, axis)) for axis in axes
        )
        profile = sketch.profile.transform(
            Transform(matrix, tuple(_dot(delta, axis) for axis in axes))
        )
        sections.append((_dot(delta, frame.normal), profile))
    _positions([position for position, _ in sections], "loft")
    return field_morph(sections).place(frame.as_placement())


def loft(profile, stations) -> Field:
    """Linearly interpolate profile offsets and positive scales between Z stations."""
    if not isinstance(profile, Field) or profile.dim != 2:
        raise ValueError("loft requires a 2D profile")
    stations = tuple(stations)
    if any(not isinstance(station, LoftStation) for station in stations):
        raise ValueError("loft stations must be LoftStation values")
    _positions([station.position for station in stations], "loft")
    stations = tuple((station.position, *station.offset, *station.scale) for station in stations)
    return Field("loft", 3, Loft(stations, _Z_AXES), (profile,))


def extrude(
    profile, height, *, offset=0.0, bounds=None, height_lipschitz=None, offset_lipschitz=None
):
    from .bounds import as_aabb

    height = height if callable(height) else _positive(height)
    offset = offset if callable(offset) else _number(offset)
    height_lipschitz = None if height_lipschitz is None else _number(height_lipschitz)
    offset_lipschitz = None if offset_lipschitz is None else _number(offset_lipschitz)
    if (height_lipschitz is not None and height_lipschitz < 0) or (
        offset_lipschitz is not None and offset_lipschitz < 0
    ):
        raise ValueError("extrusion gradient bounds must be nonnegative")
    if not callable(height) and not callable(offset) and bounds is None:
        return Field("extrude", 3, Extrude(height, offset), (profile,))
    if bounds is None:
        raise ValueError("callable extrusion requires finite bounds to cap the geometry")
    bounds = as_aabb(bounds, 3)
    return Field(
        "variable_extrude",
        3,
        VariableExtrude(height, offset, bounds, height_lipschitz, offset_lipschitz),
        (profile,),
    )


def _capped(planar, axial):
    return np.hypot(np.maximum(planar, 0), np.maximum(axial, 0)) + np.minimum(
        np.maximum(planar, axial), 0
    )


def evaluate_operation(node, q, evaluate):
    if node.kind == "variable_extrude":
        height, offset = node.data.height, node.data.offset
        box = node.data.bounds
        xy = np.clip(q[..., :2], box.minimum[:2], box.maximum[:2])
        x, y, z = xy[..., 0], xy[..., 1], q[..., 2]
        height = np.asarray(height(x, y) if callable(height) else height)
        offset = np.asarray(offset(x, y) if callable(offset) else offset)
        distance, material = evaluate(node.children[0], q[..., :2])
        distance = _capped(distance, np.abs(z - offset) - height / 2)
    else:
        stations, (axis, transverse) = node.data.stations, node.data.axes
        z = q[..., axis]
        xy = q[..., np.asarray(transverse)]
        if node.kind == "loft":
            table = np.asarray(stations)
            positions = table[:, 0]
            ox, oy, sx, sy = (np.interp(z, positions, table[:, i]) for i in range(1, 5))
            scale = np.stack((sx, sy), axis=-1)
            offset = np.stack((ox, oy), axis=-1)
            distance, material = evaluate(node.children[0], (xy - offset) / scale)
            distance *= np.minimum(sx, sy)
        else:
            positions = np.asarray(stations)
            index = np.clip(np.searchsorted(positions, z, side="right") - 1, 0, len(stations) - 2)
            weight = np.clip(
                (z - positions[index]) / (positions[index + 1] - positions[index]), 0, 1
            )
            samples = [evaluate(child, xy) for child in node.children]
            fields = np.stack([sample[0] for sample in samples], axis=-1)
            materials = np.stack([sample[1] for sample in samples], axis=-1)
            a = np.take_along_axis(fields, index[..., None], axis=-1)[..., 0]
            b = np.take_along_axis(fields, (index + 1)[..., None], axis=-1)[..., 0]
            distance = (1 - weight) * a + weight * b
            nearest = index + (weight >= 0.5)
            material = np.take_along_axis(materials, nearest[..., None], axis=-1)[..., 0]
        distance = _capped(distance, np.maximum(positions[0] - z, z - positions[-1]))
    return distance, material


def operation_bounds(node, level, analytic):
    from .bounds import AABB, _empty, _unknown

    padding = max(0.0, level)
    if node.kind == "variable_extrude":
        height, offset, box = node.data.height, node.data.offset, node.data.bounds
        bounded = box.expanded(padding)
        planar = analytic(node.children[0], level)
        if planar.is_empty:
            return _empty(3)
        if callable(height) or callable(offset):
            return bounded.intersection(
                AABB((*planar.minimum, -float("inf")), (*planar.maximum, float("inf")))
            )
        return bounded.intersection(
            AABB(
                (*planar.minimum, offset - height / 2 - padding),
                (*planar.maximum, offset + height / 2 + padding),
            )
        )
    stations, (axis, transverse) = node.data.stations, node.data.axes
    if node.kind == "field_morph":
        planar = _empty(2)
        for profile in node.children:
            planar = planar.union(analytic(profile, level))
        low, high = stations[0], stations[-1]
    else:
        # A common sublevel bound covers every interpolated scale/offset.
        minimum_scale = min(min(s[3:]) for s in stations)
        base = analytic(node.children[0], padding / minimum_scale)
        if base.is_empty:
            return _empty(3)
        planar = _empty(2)
        for _, ox, oy, sx, sy in stations:
            planar = planar.union(
                AABB(
                    (ox + sx * base.minimum[0], oy + sy * base.minimum[1]),
                    (ox + sx * base.maximum[0], oy + sy * base.maximum[1]),
                )
            )
        low, high = stations[0][0], stations[-1][0]
    if planar.is_empty:
        return _empty(3)
    lower, upper = list(_unknown(3).minimum), list(_unknown(3).maximum)
    lower[axis], upper[axis] = low - padding, high + padding
    for i, target in enumerate(transverse):
        lower[target], upper[target] = planar.minimum[i], planar.maximum[i]
    return AABB(tuple(lower), tuple(upper))
