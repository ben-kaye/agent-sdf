"""Section lofts and spatially varying extrusion fields."""

import jax.numpy as jnp

from .dsl import Shape, _number, _point, _positive


def _axes(axis):
    if axis not in ("x", "y", "z"):
        raise ValueError("loft axis must be x, y, or z")
    axial = "xyz".index(axis)
    return axial, tuple(i for i in range(3) if i != axial)


def _positions(values):
    if len(values) < 2 or any(a >= b for a, b in zip(values, values[1:], strict=False)):
        raise ValueError("loft needs at least two strictly increasing station positions")


def loft(sections, *, axis="z") -> Shape:
    """Interpolate 2D signed fields at (position, profile) sections, with flat caps.

    This implicit morph is a distance estimate; use profile.loft for interpolated
    offsets/scales of one profile. Transverse coordinates follow world axis order.
    """
    axes = _axes(axis)
    sections = tuple(sections)
    positions, profiles = [], []
    for position, profile in sections:
        if not isinstance(profile, Shape) or profile.dim != 2:
            raise ValueError("loft sections must be 2D shapes")
        positions.append(_number(position))
        profiles.append(profile)
    _positions(positions)
    return Shape("loft", 3, (tuple(positions), axes), tuple(profiles))


def section_loft(profile, stations, *, axis="z") -> Shape:
    """Linearly interpolate profile offsets and positive axis scales between stations."""
    if not isinstance(profile, Shape) or profile.dim != 2:
        raise ValueError("loft requires a 2D profile")
    axes = _axes(axis)
    stations = tuple(_point(station, 5) for station in stations)
    _positions([station[0] for station in stations])
    for station in stations:
        _positive(station[3])
        _positive(station[4])
    return Shape("section_loft", 3, (stations, axes), (profile,))


def extrude(profile, height, *, offset=0.0, bounds=None):
    from .bounds import as_aabb

    height = height if callable(height) else _positive(height)
    offset = offset if callable(offset) else _number(offset)
    if not callable(height) and not callable(offset) and bounds is None:
        return Shape("extrude", 3, (height, offset), (profile,))
    bounds = None if bounds is None else as_aabb(bounds, 3)
    return Shape("variable_extrude", 3, (height, offset, bounds), (profile,))


def _capped(planar, axial):
    return jnp.hypot(jnp.maximum(planar, 0), jnp.maximum(axial, 0)) + jnp.minimum(
        jnp.maximum(planar, axial), 0
    )


def evaluate_operation(node, q, evaluate):
    if node.kind == "variable_extrude":
        height, offset, _ = node.data
        x, y, z = q[..., 0], q[..., 1], q[..., 2]
        height = jnp.asarray(height(x, y) if callable(height) else height)
        offset = jnp.asarray(offset(x, y) if callable(offset) else offset)
        distance, material = evaluate(node.children[0], q[..., :2])
        distance = _capped(distance, jnp.abs(z - offset) - height / 2)
    else:
        stations, (axis, transverse) = node.data
        z = q[..., axis]
        xy = q[..., jnp.asarray(transverse)]
        if node.kind == "section_loft":
            table = jnp.asarray(stations)
            positions = table[:, 0]
            ox, oy, sx, sy = (jnp.interp(z, positions, table[:, i]) for i in range(1, 5))
            scale = jnp.stack((sx, sy), axis=-1)
            offset = jnp.stack((ox, oy), axis=-1)
            distance, material = evaluate(node.children[0], (xy - offset) / scale)
            distance *= jnp.minimum(sx, sy)
        else:
            positions = jnp.asarray(stations)
            index = jnp.clip(jnp.searchsorted(positions, z, side="right") - 1, 0, len(stations) - 2)
            weight = jnp.clip(
                (z - positions[index]) / (positions[index + 1] - positions[index]), 0, 1
            )
            samples = [evaluate(child, xy) for child in node.children]
            fields = jnp.stack([sample[0] for sample in samples], axis=-1)
            materials = jnp.stack([sample[1] for sample in samples], axis=-1)
            a = jnp.take_along_axis(fields, index[..., None], axis=-1)[..., 0]
            b = jnp.take_along_axis(fields, (index + 1)[..., None], axis=-1)[..., 0]
            distance = (1 - weight) * a + weight * b
            nearest = index + (weight >= 0.5)
            material = jnp.take_along_axis(materials, nearest[..., None], axis=-1)[..., 0]
        distance = _capped(distance, jnp.maximum(positions[0] - z, z - positions[-1]))
    return distance, material


def operation_bounds(node, level, analytic):
    from .bounds import AABB, _empty, _unknown

    padding = max(0.0, level)
    if node.kind == "variable_extrude":
        height, offset, box = node.data
        if box is not None and level <= 0:
            return box
        planar = analytic(node.children[0], level)
        if planar.is_empty:
            return _empty(3)
        if callable(height) or callable(offset):
            return AABB((*planar.minimum, -float("inf")), (*planar.maximum, float("inf")))
        return AABB(
            (*planar.minimum, offset - height / 2 - padding),
            (*planar.maximum, offset + height / 2 + padding),
        )
    stations, (axis, transverse) = node.data
    if node.kind == "loft":
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
