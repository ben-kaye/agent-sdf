"""Profile lofts, field morphs, and spatially varying extrusion fields."""

from math import hypot, isfinite

import numpy as np

from .field import Field, LoftStation, _number, _positive
from .nodes import Extrude, FieldMorph, Loft, VariableExtrude

_Z_AXES = (2, (0, 1))


def _positions(values, operation):
    if len(values) < 2 or any(a >= b for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(f"{operation} needs at least two strictly increasing station positions")


def _section_boxes(sections, axes):
    """Enclose each station segment's zero set from (position, planar box) pairs."""
    from .bounds import AABB, clip_boxes

    axis, transverse = axes

    def lift(low, high, planar):
        lower, upper = [0.0] * 3, [0.0] * 3
        lower[axis], upper[axis] = low, high
        for i, target in enumerate(transverse):
            lower[target], upper[target] = planar.minimum[i], planar.maximum[i]
        return AABB(tuple(lower), tuple(upper))

    return clip_boxes(
        (
            lift(a, b, first.union(second))
            for (a, first), (b, second) in zip(sections, sections[1:], strict=False)
        ),
        4,
    )


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
    from .bounds import _analytic

    bounds = _section_boxes(
        [
            (position, _analytic(profile))
            for position, profile in zip(positions, profiles, strict=True)
        ],
        _Z_AXES,
    )
    return Field("field_morph", 3, FieldMorph(tuple(positions), _Z_AXES, bounds), tuple(profiles))


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
    from .bounds import AABB, _analytic

    base = _analytic(profile)
    bounds = _section_boxes(
        [
            (
                position,
                AABB(
                    (ox + sx * base.minimum[0], oy + sy * base.minimum[1]),
                    (ox + sx * base.maximum[0], oy + sy * base.maximum[1]),
                ),
            )
            for position, ox, oy, sx, sy in stations
        ],
        _Z_AXES,
    )
    return Field("loft", 3, Loft(stations, _Z_AXES, bounds), (profile,))


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
    from .bounds import clip_bounds

    return clip_bounds(AABB(tuple(lower), tuple(upper)), node.data.bounds, padding)


def _loft_scale_spans(first, last, ratio=2.0, limit=24):
    """Split a loft segment so each step box stays within a scale ratio.

    The march certificate divides by the smallest scale in the box. One box
    over a taper to a point stalls before the tip is reached.
    """

    def at(t):
        return tuple(a + t * (b - a) for a, b in zip(first, last, strict=True))

    def step(t):
        dt = 1.0 - t
        for start, end in zip(first[3:], last[3:], strict=True):
            origin = start + t * (end - start)
            slope = end - start
            if origin <= 0.0 or slope == 0.0:
                continue
            limit_scale = origin * ratio if slope > 0.0 else origin / ratio
            dt = min(dt, (limit_scale - origin) / slope)
        return dt

    spans = []
    t = 0.0
    for _ in range(limit):
        if t >= 1.0 - 1e-12:
            break
        dt = step(t)
        nxt = 1.0 if dt <= 1e-8 else min(1.0, t + dt)
        spans.append((at(t), at(nxt)))
        t = nxt
    if not spans:
        spans.append((first, last))
    elif t < 1.0 - 1e-12:
        spans.append((at(t), at(1.0)))
    return spans


def _section_span_box(node, first, last):
    from .bounds import AABB

    profile = node.children[0].bounds()
    if profile.is_empty or not all(isfinite(x) for x in profile.minimum + profile.maximum):
        raise ValueError("GLSL loft needs finite profile bounds for safe stepping")
    axis, transverse = node.data.axes
    low, high = [0.0] * 3, [0.0] * 3
    low[axis], high[axis] = first[0], last[0]
    for i, world_axis in enumerate(transverse):
        values = [
            station[1 + i] + station[3 + i] * edge
            for station in (first, last)
            for edge in (profile.minimum[i], profile.maximum[i])
        ]
        low[world_axis], high[world_axis] = min(values), max(values)
    return AABB(tuple(low), tuple(high))


def section_step_boxes(node):
    """Return padded (box, pad) pairs for each scale-limited loft span."""
    result = []
    for first, last in zip(node.data.stations, node.data.stations[1:], strict=False):
        for span in _loft_scale_spans(first, last):
            box = _section_span_box(node, *span)
            pad = max(
                1e-4,
                1e-6 * hypot(*(b - a for a, b in zip(box.minimum, box.maximum, strict=True))),
            )
            result.append((box.expanded(pad), pad))
    return result


def _field_magnitude(node, box, bound):
    center = tuple((a + b) / 2 for a, b in zip(box.minimum, box.maximum, strict=True))
    radius = hypot(*(b - a for a, b in zip(center, box.maximum, strict=True)))
    value = float(node.sample(np.asarray(center, dtype=float))[0])
    if not isfinite(value):
        raise ValueError("section profile has a nonfinite field value")
    return abs(value) + bound * radius + 1e-5


def _section_at(a, b, position):
    t = (position - a[0]) / (b[0] - a[0])
    return (position, *(x + t * (y - x) for x, y in zip(a[1:], b[1:], strict=True)))


def section_lipschitz(node, box, recurse):
    """Bound a loft or field morph's slope over a 3D box."""
    from .bounds import AABB

    stations, (axial, transverse) = node.data.stations, node.data.axes
    planar = AABB(
        tuple(box.minimum[i] for i in transverse), tuple(box.maximum[i] for i in transverse)
    )
    if node.kind == "field_morph":
        bounds = [recurse(profile, planar) for profile in node.children]
        magnitudes = [
            _field_magnitude(profile, planar, bound)
            for profile, bound in zip(node.children, bounds, strict=True)
        ]
        slope = max(
            (a + b) / (z1 - z0)
            for a, b, z0, z1 in zip(
                magnitudes[:-1], magnitudes[1:], stations[:-1], stations[1:], strict=True
            )
        )
        return hypot(max(bounds), slope, 1.0)
    active = []
    for a, b in zip(stations, stations[1:], strict=False):
        start = max(a[0], box.minimum[axial])
        end = min(b[0], box.maximum[axial])
        if end > start:
            active.append((_section_at(a, b, start), _section_at(a, b, end)))
    if not active:
        active = [(stations[0], stations[1])]
    rows = tuple(row for pair in active for row in pair)
    minimum_scale = [min(row[3 + i] for row in rows) for i in range(2)]
    maximum_scale = min(max(row[3] for row in rows), max(row[4] for row in rows))
    offset_low = [min(row[1 + i] for row in rows) for i in range(2)]
    offset_high = [max(row[1 + i] for row in rows) for i in range(2)]
    displacement = [
        max(abs(planar.minimum[i] - offset_high[i]), abs(planar.maximum[i] - offset_low[i]))
        for i in range(2)
    ]
    reach = [displacement[i] / minimum_scale[i] for i in range(2)]
    profile_box = AABB(tuple(-x for x in reach), tuple(reach))
    profile = node.children[0]
    profile_bound = recurse(profile, profile_box)
    magnitude = _field_magnitude(profile, profile_box, profile_bound)
    transverse_bound = maximum_scale * profile_bound / min(minimum_scale)
    axial_bound = 0.0
    for a, b in active:
        dz = b[0] - a[0]
        offset_slope = [abs(b[1 + i] - a[1 + i]) / dz for i in range(2)]
        scale_slope = [abs(b[3 + i] - a[3 + i]) / dz for i in range(2)]
        mapped_slope = hypot(
            *(
                offset_slope[i] / minimum_scale[i]
                + displacement[i] * scale_slope[i] / minimum_scale[i] ** 2
                for i in range(2)
            )
        )
        axial_bound = max(
            axial_bound,
            max(scale_slope) * magnitude + maximum_scale * profile_bound * mapped_slope,
        )
    return hypot(transverse_bound, axial_bound, 1.0)
