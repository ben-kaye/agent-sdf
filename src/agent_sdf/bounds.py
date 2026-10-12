"""Per-kind bounding rules: solid extents, value and ray-step certificates, and Lipschitz bounds."""

from collections import namedtuple
from itertools import product
from math import cos, hypot, inf, isfinite, pi, prod, sin

import numpy as np


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


def clip_boxes(boxes, limit):
    """Merge boxes enclosing a zero set to at most limit, padded for float32 clipping.

    Returns None when any box is empty or unbounded, leaving the field unclipped.
    """
    boxes = list(boxes)
    if not boxes or any(
        box.is_empty or not all(isfinite(v) for v in box.minimum + box.maximum) for box in boxes
    ):
        return None
    while len(boxes) > limit:
        merged = [a.union(b) for a, b in zip(boxes, boxes[1:], strict=False)]
        growth = [
            prod(m.extents) - prod(a.extents) - prod(b.extents)
            for m, a, b in zip(merged, boxes, boxes[1:], strict=False)
        ]
        i = growth.index(min(growth))
        boxes[i : i + 2] = [merged[i]]
    return tuple(
        box.expanded(1e-6 * (1 + max(map(abs, box.minimum + box.maximum)))) for box in boxes
    )


def clip_bounds(box, clip, padding):
    """Intersect a sublevel bound with the enclosure of a field's clip boxes."""
    if clip is None:
        return box
    enclosure = clip[0]
    for item in clip[1:]:
        enclosure = enclosure.union(item)
    # The clip caps every sublevel set at its box distance.
    return box.intersection(enclosure.expanded(padding))


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


def _project(box, plane):
    """Bound a 3D box projected onto plane-local (x, y); projection never lengthens."""
    offsets = [
        tuple(c - o for c, o in zip(corner, plane.origin, strict=True))
        for corner in product(*zip(box.minimum, box.maximum, strict=True))
    ]
    return _points_bounds(
        tuple(
            sum(a * b for a, b in zip(p, axis, strict=True))
            for axis in (plane.x_direction, plane.y_direction)
        )
        for p in offsets
    )


def _slice_bounds(box, plane):
    if box.is_empty:
        return _empty(2)
    if not all(isfinite(x) for x in box.minimum + box.maximum):
        return _unknown(2)
    corners = product(*zip(box.minimum, box.maximum, strict=True))
    distances = [sum((p[i] - plane.origin[i]) * plane.normal[i] for i in range(3)) for p in corners]
    if min(distances) > 0 or max(distances) < 0:
        return _empty(2)
    return _project(box, plane)


def revolve_bounds(box, angle):
    """Bound a (radial, axial) profile box revolved about Z from +X through angle."""
    (u0, z0), (u1, z1) = box.minimum, box.maximum
    if not all(isfinite(x) for x in (u0, u1)):
        return AABB((-inf, -inf, z0), (inf, inf, z1))
    if angle >= 2 * pi:
        radius = max(-u0, u1, 0.0)
        return AABB((-radius, -radius, z0), (radius, radius, z1))
    points = []
    # Profile points at negative radial coordinates sweep from the opposite side.
    for low, high, start in ((max(u0, 0.0), u1, 0.0), (max(-u1, 0.0), -u0, pi)):
        if high < 0:
            continue
        turns = (k * pi / 2 for k in range(1, 8) if start < k * pi / 2 < start + angle)
        for a in (start, start + angle, *turns):
            points += [(radius * cos(a), radius * sin(a)) for radius in (low, high)]
    xs, ys = zip(*points, strict=True)
    return AABB((min(xs), min(ys), z0), (max(xs), max(ys), z1))


def enclose(boxes):
    boxes = list(boxes)
    result = boxes[0]
    for box in boxes[1:]:
        result = result.union(box)
    return result


def _swept(box, displacement):
    """Bound a box copied along a linear displacement."""
    if box.is_empty:
        return box
    return AABB(
        tuple(a + min(0.0, d) for a, d in zip(box.minimum, displacement, strict=True)),
        tuple(b + max(0.0, d) for b, d in zip(box.maximum, displacement, strict=True)),
    )


def _ring(box, data):
    """Bound a box copied around a radial pattern's axis."""
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


def _extruded(box, data, padding):
    if box.is_empty:
        return _empty(3)
    return AABB(
        (*box.minimum, data.offset - data.height / 2 - padding),
        (*box.maximum, data.offset + data.height / 2 + padding),
    )


def _mapped(box, transform, condition):
    """Map a box forward, padded for float32 error amplified by the condition number."""
    box = _affine(box, transform.matrix, transform.offset)
    return box.expanded(1e-5 * (1 + max(map(abs, box.minimum + box.maximum))) * condition)


def _expanded(certificate, amount):
    if certificate is None:
        return None
    box, scale = certificate
    return box.expanded(amount / scale), scale


def step_box(node):
    """Return the padded (box, pad) used for box-entry stepping, or None."""
    box = node.bounds()
    if box.is_empty or not all(isfinite(x) for x in box.minimum + box.maximum):
        return None
    pad = (
        0.0
        if node.kind == "variable_extrude"
        else max(
            1e-4, 1e-6 * hypot(*(b - a for a, b in zip(box.minimum, box.maximum, strict=True)))
        )
    )
    return box.expanded(pad), pad


class Rule:
    """Bounding rules for one field kind.

    solid bounds {p: field(p) <= level}. value returns (box, scale) with field
    values >= scale * distance to box outside it, certifying children with
    recurse. step does the same for ray-march steps; the caller unions it with
    value. lipschitz bounds the field's slope over a convex box, certifying
    children over the boxes they are sampled in with recurse.
    """

    def solid(self, field, level):
        return _unknown(field.dim)

    def value(self, field, recurse):
        return None

    def step(self, field, recurse):
        try:
            padded = step_box(field)
        except ValueError:
            return None
        return None if padded is None else (padded[0], 1.0)

    def lipschitz(self, field, box, recurse):
        raise ValueError(f"field kind '{field.kind}' has no Lipschitz bound")


RULES = {}
_DEFAULT = Rule()


def _rule(*kinds):
    def register(cls):
        RULES.update(dict.fromkeys(kinds, cls()))
        return cls

    return register


PRIMITIVES = (
    "circle",
    "sphere",
    "rectangle",
    "box",
    "cylinder",
    "torus",
    "capsule",
    "tube",
    "polygon",
    "curve_profile",
)


def _primitive_box(field, padding=0.0):
    kind, data, dim = field.kind, field.data, field.dim
    if kind in ("sphere", "circle"):
        return _centered((data.radius + padding,) * dim)
    if kind in ("box", "rectangle"):
        sizes = (data.width, data.height) if dim == 2 else (data.width, data.depth, data.height)
        return _centered(tuple(x / 2 + padding for x in sizes))
    if kind == "cylinder":
        return _centered((data.radius + padding, data.radius + padding, data.height / 2 + padding))
    if kind == "torus":
        outer = data.major_radius + data.tube_radius + padding
        return _centered((outer, outer, data.tube_radius + padding))
    if kind == "capsule":
        return _points_bounds((data.start, data.end)).expanded(data.radius + padding)
    if kind == "tube":
        return _points_bounds(data.points).expanded(data.radius + padding)
    if kind == "polygon":
        return _points_bounds(data.vertices).expanded(padding)
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


@_rule(*PRIMITIVES)
class _Primitive(Rule):
    def solid(self, field, level):
        return _primitive_box(field, max(0.0, level))

    def value(self, field, recurse):
        return _primitive_box(field), 1.0

    step = value

    def lipschitz(self, field, box, recurse):
        return 1.0


def _steepest(field, box, recurse):
    """Bound operations whose gradient is a convex combination of their children's."""
    return max(recurse(child, box) for child in field.children)


@_rule("custom")
class _Clipped(Rule):
    # Lowered with max(d, sdBox) against the declared bounds.
    def solid(self, field, level):
        return field.data.bounds.expanded(max(0.0, level))

    def value(self, field, recurse):
        return field.data.bounds, 1.0

    def lipschitz(self, field, box, recurse):
        if field.data.lipschitz is None:
            raise ValueError("custom fields need a declared Lipschitz bound")
        return max(1.0, field.data.lipschitz)


@_rule("warp")
class _Warp(_Clipped):
    def lipschitz(self, field, box, recurse):
        warp = field.data.warp
        center = np.asarray(box.minimum) / 2 + np.asarray(box.maximum) / 2
        radius = np.linalg.norm(np.asarray(box.maximum) - center) * warp.lipschitz
        mapped = warp.map_points(center)
        child_box = AABB(tuple(mapped - radius), tuple(mapped + radius))
        return max(1.0, recurse(field.children[0], child_box))


@_rule("field_morph")
class _Sections(Rule):
    def solid(self, field, level):
        from .profiles import operation_bounds

        return operation_bounds(field, level, _analytic)

    def value(self, field, recurse):
        # Clipped to the distance of boxes enclosing the zero set.
        bounds = field.data.bounds
        return None if bounds is None else (enclose(bounds), 1.0)

    def lipschitz(self, field, box, recurse):
        from .profiles import section_lipschitz

        return section_lipschitz(field, box, recurse)


@_rule("loft")
class _Loft(_Sections):
    def step(self, field, recurse):
        from .profiles import section_step_boxes

        return enclose(box for box, _ in section_step_boxes(field)), 1.0


@_rule("variable_extrude")
class _VariableExtrude(_Sections):
    def value(self, field, recurse):
        return field.data.bounds, 1.0

    def lipschitz(self, field, box, recurse):
        data = field.data
        height = data.height_lipschitz if callable(data.height) else 0.0
        offset = data.offset_lipschitz if callable(data.offset) else 0.0
        if height is None or offset is None:
            raise ValueError(
                "variable extrusion needs height_lipschitz and offset_lipschitz for callable fields"
            )
        child = recurse(field.children[0], _planar(box))
        return hypot(child, hypot(1.0, offset + height * 0.5))


@_rule("profile_sweep")
class _Sweep(_Sections):
    def solid(self, field, level):
        from .sweep import sweep_bounds

        return sweep_bounds(field, level, _analytic)

    def lipschitz(self, field, box, recurse):
        from .sweep import _spans

        # A sweep samples profiles at offsets from span origins; parallel
        # sweeps shear those offsets by up to the inverse span scale.
        spans = _spans(field.data)
        distance = max(
            hypot(*(b - a for a, b in zip(span.start, q, strict=True)))
            for span in spans
            for q in (box.minimum, box.maximum)
        ) / min(span.scale for span in spans)
        child = recurse(field.children[0], AABB((-distance, -distance), (distance, distance)))
        return hypot(child, 1.0)


@_rule("material", "feature")
class _Label(Rule):
    def solid(self, field, level):
        return _analytic(field.children[0], level)

    def value(self, field, recurse):
        return recurse(field.children[0])

    step = value
    lipschitz = staticmethod(_steepest)


@_rule("subtract")
class _Subtract(_Label):
    """A difference lies inside its first operand."""


@_rule("min")
class _Union(Rule):
    def solid(self, field, level):
        a, b = (_analytic(child, level) for child in field.children)
        return a.union(b)

    def value(self, field, recurse):
        a, b = (recurse(child) for child in field.children)
        if a is None or b is None:
            return None
        return a[0].union(b[0]), min(a[1], b[1])

    step = value
    lipschitz = staticmethod(_steepest)


@_rule("smooth_union")
class _SmoothUnion(_Union):
    def solid(self, field, level):
        return super().solid(field, level + field.data.radius / 4)

    def value(self, field, recurse):
        return _expanded(super().value(field, recurse), field.data.radius / 4)

    step = Rule.step


@_rule("max")
class _Intersection(Rule):
    def solid(self, field, level):
        a, b = (_analytic(child, level) for child in field.children)
        return a.intersection(b)

    def value(self, field, recurse):
        # An intersection's enclosing box alone does not certify a Euclidean
        # distance bound. Either operand's certificate does.
        available = [item for item in map(recurse, field.children) if item is not None]
        return min(available, key=lambda item: prod(item[0].extents)) if available else None

    step = value
    lipschitz = staticmethod(_steepest)


@_rule("offset")
class _Offset(Rule):
    def solid(self, field, level):
        return _analytic(field.children[0], level + field.data.amount)

    def value(self, field, recurse):
        return _expanded(recurse(field.children[0]), max(0.0, field.data.amount))

    lipschitz = staticmethod(_steepest)


@_rule("shell")
class _Shell(Rule):
    def solid(self, field, level):
        half = field.data.thickness / 2
        if level < -half:
            return _empty(field.dim)
        return _analytic(field.children[0], level + half)

    def value(self, field, recurse):
        return _expanded(recurse(field.children[0]), field.data.thickness / 2)

    lipschitz = staticmethod(_steepest)


@_rule("transform")
class _Transform(Rule):
    def solid(self, field, level):
        transform = field.data.transform
        box = _analytic(field.children[0], level / transform.distance_scale)
        return _affine(box, transform.matrix, transform.offset)

    def value(self, field, recurse):
        child = recurse(field.children[0])
        if child is None:
            return None
        transform = field.data.transform
        stretch = float(np.linalg.norm(np.asarray(transform.matrix), 2))
        inverse = np.asarray(transform.inverse_matrix, dtype=np.float32).astype(float)
        if np.linalg.cond(inverse) > 1e4:
            return None
        condition = stretch * float(np.linalg.norm(inverse, 2))
        box = _mapped(child[0], transform, condition)
        return box, child[1] * transform.distance_scale / stretch

    def step(self, field, recurse):
        child = recurse(field.children[0])
        if child is None:
            return None
        # Box entry and distance steps map back through the ray speed;
        # the condition number bounds both directions of the stretch.
        transform = field.data.transform
        condition = float(np.linalg.norm(np.asarray(transform.matrix, dtype=float), 2)) * float(
            np.linalg.norm(np.asarray(transform.inverse_matrix, dtype=float), 2)
        )
        return _mapped(child[0], transform, condition), child[1] / condition

    def lipschitz(self, field, box, recurse):
        transform = field.data.transform
        inverse = np.asarray(transform.inverse_matrix)
        child_box = _affine(box, inverse, -inverse @ np.asarray(transform.offset))
        stretch = float(np.linalg.norm(inverse, 2))
        return recurse(field.children[0], child_box) * transform.distance_scale * stretch


@_rule("slice")
class _Slice(Rule):
    def solid(self, field, level):
        return _slice_bounds(_analytic(field.children[0], level), field.data.plane)

    def value(self, field, recurse):
        child = recurse(field.children[0])
        if child is None:
            return None
        return _project(child[0], field.data.plane), child[1]

    def lipschitz(self, field, box, recurse):
        plane = field.data.plane
        matrix = tuple(zip(plane.x_direction, plane.y_direction, strict=True))
        return recurse(field.children[0], _affine(box, matrix, plane.origin))


def _displacement(data, sign=1.0):
    return tuple(sign * (data.count - 1) * value for value in data.step)


@_rule("linear_pattern")
class _LinearPattern(Rule):
    def solid(self, field, level):
        return _swept(_analytic(field.children[0], level), _displacement(field.data))

    def value(self, field, recurse):
        child = recurse(field.children[0])
        return None if child is None else (_swept(child[0], _displacement(field.data)), child[1])

    step = value

    def lipschitz(self, field, box, recurse):
        # Every copy samples the child at p minus a multiple of the step.
        return recurse(field.children[0], _swept(box, _displacement(field.data, -1.0)))


@_rule("radial_pattern")
class _RadialPattern(Rule):
    def solid(self, field, level):
        return _ring(_analytic(field.children[0], level), field.data)

    def value(self, field, recurse):
        child = recurse(field.children[0])
        return None if child is None else (_ring(child[0], field.data), child[1])

    step = value

    def lipschitz(self, field, box, recurse):
        return recurse(field.children[0], _ring(box, field.data))


@_rule("extrude")
class _Extrude(Rule):
    def solid(self, field, level):
        box = _analytic(field.children[0], level)
        return _extruded(box, field.data, max(0.0, level))

    def value(self, field, recurse):
        child = recurse(field.children[0])
        if child is None:
            return None
        return _extruded(child[0], field.data, 0.0), min(child[1], 1.0)

    def lipschitz(self, field, box, recurse):
        return max(recurse(field.children[0], _planar(box)), 1.0)


def _planar(box):
    return AABB(box.minimum[:2], box.maximum[:2])


def _revolve_sides(profile):
    """Whether a revolve profile reaches the +u and -u sides of its axis."""
    box = _analytic(profile)
    if box.is_empty:
        return True, False
    return box.maximum[0] > 0 or box.minimum[0] >= 0, box.minimum[0] < 0


@_rule("revolve")
class _Revolve(Rule):
    def solid(self, field, level):
        box = _analytic(field.children[0], level)
        if box.is_empty:
            return _empty(3)
        box = revolve_bounds(box, field.data.angle)
        return box if field.data.angle >= 2 * pi else box.expanded(max(0.0, level))

    def value(self, field, recurse):
        child = recurse(field.children[0])
        if child is None:
            return None
        box, scale = child
        if field.data.angle < 2 * pi:
            scale = min(scale, 1.0)
        return revolve_bounds(box, field.data.angle), scale

    def lipschitz(self, field, box, recurse):
        radius = max(
            hypot(x, y)
            for x in (box.minimum[0], box.maximum[0])
            for y in (box.minimum[1], box.maximum[1])
        )
        near, far = _revolve_sides(field.children[0])
        child_box = AABB(
            (-radius if far else 0.0, box.minimum[2]), (radius if near else 0.0, box.maximum[2])
        )
        child = recurse(field.children[0], child_box)
        return child if field.data.angle >= 2 * pi else max(child, 1.0)


def _analytic(field, level=0.0):
    """Bound {p: field(p) <= level}; carry offsets through distance scaling."""
    return RULES.get(field.kind, _DEFAULT).solid(field, level)


def _checked(result):
    if result is None:
        return None
    box, scale = result
    if box.is_empty or not all(isfinite(v) for v in box.minimum + box.maximum):
        return None
    return result if isfinite(scale) and scale > 0 else None


def certificate(field, recurse=None):
    """Return (box, scale) with field values >= scale * distance to box outside it.

    recurse certifies children; it defaults to this value certificate.
    """
    return _checked(RULES.get(field.kind, _DEFAULT).value(field, recurse or certificate))


def step_certificate(field, recurse):
    """Return (box, scale) with ray steps and values >= scale * box distance.

    recurse certifies children's steps.
    """
    step = _checked(RULES.get(field.kind, _DEFAULT).step(field, recurse))
    value = certificate(field)
    if step is None or value is None:
        return None
    return step[0].union(value[0]), min(step[1], value[1])


def lipschitz(field, box):
    """Bound the field's slope over a convex box containing every march point and surface."""
    result = RULES.get(field.kind, _DEFAULT).lipschitz(field, box, lipschitz)
    if not isfinite(result) or result <= 0:
        raise ValueError("field Lipschitz bound must be finite and positive")
    return result


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
