"""Conservative analytic sublevel bounds."""

from collections import namedtuple
from itertools import product
from math import hypot, inf, isfinite


def _coordinate(value):
    try:
        iter(value)
    except TypeError:
        return float(value)
    raise ValueError("AABB coordinates must be scalar")


class AABB(namedtuple("_AABB", "minimum maximum")):
    __slots__ = ()

    def __new__(cls, minimum, maximum):
        try:
            minimum = tuple(_coordinate(value) for value in minimum)
            maximum = tuple(_coordinate(value) for value in maximum)
        except (TypeError, ValueError) as exc:
            raise ValueError("AABB corners must be coordinate vectors") from exc
        if len(minimum) not in (2, 3) or len(maximum) != len(minimum):
            raise ValueError("AABB corners must have matching 2D or 3D shapes")
        return super().__new__(cls, minimum, maximum)

    @property
    def is_empty(self):
        return any(a > b for a, b in zip(self.minimum, self.maximum, strict=True))

    @property
    def extents(self):
        if self.is_empty:
            return (0.0,) * len(self.minimum)
        return tuple(b - a for a, b in zip(self.minimum, self.maximum, strict=True))

    def union(self, other):
        return AABB(
            tuple(map(min, self.minimum, other.minimum, strict=True)),
            tuple(map(max, self.maximum, other.maximum, strict=True)),
        )

    def intersection(self, other):
        result = AABB(
            tuple(map(max, self.minimum, other.minimum, strict=True)),
            tuple(map(min, self.maximum, other.maximum, strict=True)),
        )
        return _empty(len(self.minimum)) if result.is_empty else result

    def expanded(self, amount):
        if self.is_empty:
            return self
        return AABB(
            tuple(x - amount for x in self.minimum), tuple(x + amount for x in self.maximum)
        )


def _empty(dim):
    return AABB((inf,) * dim, (-inf,) * dim)


def _unknown(dim):
    return AABB((-inf,) * dim, (inf,) * dim)


def as_aabb(value, dim):
    try:
        lower, upper = (tuple(_coordinate(x) for x in p) for p in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("bounds must be a pair of finite coordinate vectors") from exc
    if (
        len(lower) != dim
        or len(upper) != dim
        or not all(isfinite(x) for x in lower + upper)
        or any(a > b for a, b in zip(lower, upper, strict=True))
    ):
        raise ValueError(f"bounds need {dim} finite ordered coordinates per corner")
    return AABB(lower, upper)


def _points_bounds(points):
    axes = tuple(zip(*points, strict=True))
    return AABB(tuple(min(axis) for axis in axes), tuple(max(axis) for axis in axes))


def _centered(half):
    return AABB(tuple(-x for x in half), tuple(half))


def _affine(box, matrix, translation):
    if box.is_empty:
        return box
    if not all(isfinite(x) for x in box.minimum + box.maximum):
        return _unknown(len(box.minimum))
    return _points_bounds(
        tuple(
            sum(a * b for a, b in zip(row, corner, strict=True)) + offset
            for row, offset in zip(matrix, translation, strict=True)
        )
        for corner in product(*zip(box.minimum, box.maximum, strict=True))
    )


def _slice_bounds(box, plane):
    if box.is_empty:
        return _empty(2)
    if not all(isfinite(x) for x in box.minimum + box.maximum):
        return _unknown(2)
    corners = tuple(product(*zip(box.minimum, box.maximum, strict=True)))
    offsets = [tuple(p[i] - plane.origin[i] for i in range(3)) for p in corners]
    distances = [sum(a * b for a, b in zip(p, plane.normal, strict=True)) for p in offsets]
    if min(distances) > 0 or max(distances) < 0:
        return _empty(2)
    return _points_bounds(
        tuple(
            sum(a * b for a, b in zip(p, axis, strict=True))
            for axis in (plane.x_direction, plane.y_direction)
        )
        for p in offsets
    )


def _analytic(field, level=0.0):
    """Bound {p: field(p) <= level}; carry offsets through distance scaling."""
    kind, data, dim = field.kind, field.data, field.dim
    child = field.children[0] if field.children else None
    padding = max(0.0, level)
    if kind in ("sphere", "circle"):
        return _centered((data.radius + padding,) * dim)
    if kind in ("box", "rectangle"):
        return _centered(
            tuple(
                x / 2 + padding
                for x in (
                    (data.width, data.height) if dim == 2 else (data.width, data.depth, data.height)
                )
            )
        )
    if kind == "cylinder":
        return _centered((data.radius + padding, data.radius + padding, data.height / 2 + padding))
    if kind == "torus":
        major, tube = data.major_radius, data.tube_radius
        return _centered((major + tube + padding, major + tube + padding, tube + padding))
    if kind == "capsule":
        return _points_bounds((data.start, data.end)).expanded(data.radius + padding)
    if kind == "tube":
        return _points_bounds(data.points).expanded(data.radius + padding)
    if kind == "polygon":
        return _points_bounds(data.vertices).expanded(padding)
    if kind == "curve_profile":
        result = _empty(dim)
        for segment in data.segments:
            if segment.kind == "arc":
                center, start, _ = segment.points
                radius = hypot(*(a - b for a, b in zip(start, center, strict=True)))
                box = AABB(center, center).expanded(radius)
            else:
                box = _points_bounds(segment.points)
            result = result.union(box)
        return result.expanded(padding)
    if kind == "custom":
        if level <= 0 and data.bounds is not None:
            return data.bounds
        return _unknown(dim)
    if kind in ("min", "max", "subtract", "smooth_union"):
        threshold = level + (data.radius / 4 if kind == "smooth_union" else 0)
        a = _analytic(child, threshold)
        if kind == "subtract":
            return a
        b = _analytic(field.children[1], threshold)
        return a.intersection(b) if kind == "max" else a.union(b)
    if kind == "offset":
        return _analytic(child, level + data.amount)
    if kind == "shell":
        return (
            _empty(dim)
            if level < -data.thickness / 2
            else _analytic(child, level + data.thickness / 2)
        )
    if kind in ("material", "feature"):
        return _analytic(child, level)
    if kind == "linear_pattern":
        box = _analytic(child, level)
        if box.is_empty:
            return box
        displacement = tuple((data.count - 1) * value for value in data.step)
        return AABB(
            tuple(a + min(0.0, d) for a, d in zip(box.minimum, displacement, strict=True)),
            tuple(b + max(0.0, d) for b, d in zip(box.maximum, displacement, strict=True)),
        )
    if kind == "radial_pattern":
        box = _analytic(child, level)
        if box.is_empty or data.count == 1:
            return box
        a, b = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[data.axis]
        radius = hypot(
            max(abs(box.minimum[a]), abs(box.maximum[a])),
            max(abs(box.minimum[b]), abs(box.maximum[b])),
        )
        low, high = list(box.minimum), list(box.maximum)
        low[a] = low[b] = -radius
        high[a] = high[b] = radius
        return AABB(low, high)
    if kind == "transform":
        transform = data.transform
        return _affine(
            _analytic(child, level / transform.distance_scale), transform.matrix, transform.offset
        )
    if kind == "slice":
        return _slice_bounds(_analytic(child, level), data.plane)
    if kind == "warp":
        return data.bounds if level <= 0 and data.bounds is not None else _unknown(dim)
    if kind in ("field_morph", "loft", "variable_extrude"):
        from .operations import operation_bounds

        return operation_bounds(field, level, _analytic)
    if kind == "profile_sweep":
        from .sweep import sweep_bounds

        return sweep_bounds(field, level, _analytic)
    if kind == "extrude":
        box = _analytic(child, level)
        if box.is_empty:
            return _empty(dim)
        offset = data.offset
        return AABB(
            (*box.minimum, offset - data.height / 2 - padding),
            (*box.maximum, offset + data.height / 2 + padding),
        )
    if kind == "revolve":
        box = _analytic(child, level)
        if box.is_empty or box.maximum[0] < 0:
            return _empty(dim)
        radius = box.maximum[0]
        return AABB((-radius, -radius, box.minimum[1]), (radius, radius, box.maximum[1]))
    return _unknown(dim)


def field_bounds(field, scene_bounds=None, *, tolerance=0.01, max_cells=65536):
    """Return analytic bounds, clipped to a supplied region when needed."""
    tolerance = float(tolerance)
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("bounds tolerance must be positive and finite")
    if not isinstance(max_cells, int) or isinstance(max_cells, bool) or max_cells < 1:
        raise ValueError("max_cells must be a positive integer")
    analytic = _analytic(field)
    if analytic.is_empty or all(isfinite(x) for x in analytic.minimum + analytic.maximum):
        return analytic
    if scene_bounds is None:
        raise ValueError("unresolved geometry needs explicit custom bounds or scene_bounds")
    return as_aabb(scene_bounds, field.dim).intersection(analytic)
