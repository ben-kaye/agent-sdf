"""Conservative analytic sublevel bounds."""

from collections import namedtuple
from itertools import product
from math import cos, hypot, inf, isfinite, sin


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


def _rotation(dim, angle, axis):
    c, s = cos(angle), sin(angle)
    matrix = [[float(i == j) for j in range(dim)] for i in range(dim)]
    a, b = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    matrix[a][a] = matrix[b][b] = c
    matrix[a][b], matrix[b][a] = -s, s
    return matrix


def _analytic(shape, level=0.0):
    """Bound {p: field(p) <= level}; carry offsets through distance scaling."""
    kind, data, dim = shape.kind, shape.data, shape.dim
    child = shape.children[0] if shape.children else None
    padding = max(0.0, level)
    if kind in ("sphere", "circle"):
        return _centered((data[0] + padding,) * dim)
    if kind in ("box", "rectangle"):
        return _centered(tuple(x / 2 + padding for x in data[:dim]))
    if kind == "cylinder":
        return _centered((data[0] + padding, data[0] + padding, data[1] / 2 + padding))
    if kind == "torus":
        major, tube = data
        return _centered((major + tube + padding, major + tube + padding, tube + padding))
    if kind == "capsule":
        return _points_bounds(data[:2]).expanded(data[2] + padding)
    if kind == "tube":
        return _points_bounds(data[0]).expanded(data[1] + padding)
    if kind == "polygon":
        return _points_bounds(data[0]).expanded(padding)
    if kind == "curve_profile":
        result = _empty(dim)
        for segment in data[0]:
            if segment.kind == "arc":
                center, start, _ = segment.points
                radius = hypot(*(a - b for a, b in zip(start, center, strict=True)))
                box = AABB(center, center).expanded(radius)
            else:
                box = _points_bounds(segment.points)
            result = result.union(box)
        return result.expanded(padding)
    if kind in ("custom", "python"):
        if level <= 0 and len(data) > 2 and data[2] is not None:
            return data[2]
        return _unknown(dim)
    if kind in ("min", "max", "subtract", "smooth_union"):
        threshold = level + (data[0] / 4 if kind == "smooth_union" else 0)
        a = _analytic(child, threshold)
        if kind == "subtract":
            return a
        b = _analytic(shape.children[1], threshold)
        return a.intersection(b) if kind == "max" else a.union(b)
    if kind == "offset":
        return _analytic(child, level + data[0])
    if kind == "shell":
        return _empty(dim) if level < -data[0] / 2 else _analytic(child, level + data[0] / 2)
    if kind == "material":
        return _analytic(child, level)
    if kind == "translate":
        box = _analytic(child, level)
        return AABB(
            tuple(a + b for a, b in zip(box.minimum, data, strict=True)),
            tuple(a + b for a, b in zip(box.maximum, data, strict=True)),
        )
    if kind == "scale":
        box = _analytic(child, level / data[0])
        return AABB(
            tuple(x * data[0] for x in box.minimum), tuple(x * data[0] for x in box.maximum)
        )
    if kind == "rotate":
        return _affine(_analytic(child, level), _rotation(dim, *data), (0.0,) * dim)
    if kind == "transform":
        transform = data[0]
        if transform.forward_affine is None:
            return _unknown(dim)
        return _affine(
            _analytic(child, level / transform.distance_scale), *transform.forward_affine
        )
    if kind in ("loft", "section_loft", "variable_extrude"):
        from .operations import operation_bounds

        return operation_bounds(shape, level, _analytic)
    if kind == "profile_sweep":
        from .sweep import sweep_bounds

        return sweep_bounds(shape, level, _analytic)
    if kind == "extrude":
        box = _analytic(child, level)
        if box.is_empty:
            return _empty(dim)
        offset = data[1] if len(data) > 1 else 0.0
        return AABB(
            (*box.minimum, offset - data[0] / 2 - padding),
            (*box.maximum, offset + data[0] / 2 + padding),
        )
    if kind == "revolve":
        box = _analytic(child, level)
        if box.is_empty or box.maximum[0] < 0:
            return _empty(dim)
        radius = box.maximum[0]
        return AABB((-radius, -radius, box.minimum[1]), (radius, radius, box.maximum[1]))
    return _unknown(dim)


def shape_bounds(shape, scene_bounds=None, *, tolerance=0.01, max_cells=65536):
    """Return analytic bounds, clipped to a supplied region when needed."""
    tolerance = float(tolerance)
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("bounds tolerance must be positive and finite")
    if not isinstance(max_cells, int) or isinstance(max_cells, bool) or max_cells < 1:
        raise ValueError("max_cells must be a positive integer")
    analytic = _analytic(shape)
    if analytic.is_empty or all(isfinite(x) for x in analytic.minimum + analytic.maximum):
        return analytic
    if scene_bounds is None:
        raise ValueError("unresolved geometry needs explicit custom bounds or scene_bounds")
    return as_aabb(scene_bounds, shape.dim).intersection(analytic)
