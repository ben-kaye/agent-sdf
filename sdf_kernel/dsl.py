"""Small CAD-like DSL evaluated with JAX."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, pi, sin, sqrt

import jax.numpy as jnp

from .hit import Hit
from .materials import PALETTE, finish, validate_material_id
from .transform import Transform, _Pose


def _number(value: float) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError("coordinates and dimensions must be finite")
    return value


def _positive(value: float) -> float:
    value = _number(value)
    if value <= 0:
        raise ValueError("dimension must be positive")
    return value


def _point(values: Iterable[float], dim: int) -> tuple[float, ...]:
    result = tuple(_number(v) for v in values)
    if len(result) != dim:
        raise ValueError(f"expected {dim} coordinates")
    return result


@dataclass(frozen=True)
class Shape:
    kind: str
    dim: int
    data: tuple = ()
    children: tuple[Shape, ...] = ()

    def sample(self, points):
        from .backend import evaluate_shape

        return evaluate_shape(self, points)

    def __call__(self, *point):
        if len(point) != self.dim:
            raise ValueError(f"expected {self.dim} coordinates")
        distance, material = self.sample(jnp.asarray(point))
        return Hit(float(distance), int(material))

    def sdf(self, *point):
        return self(*point).distance

    def bounds(self, scene_bounds=None, *, tolerance=0.01, max_cells=65536):
        """Conservative AABB; a fallback scene_bounds must enclose the entire shape.

        Custom fields need a supplied Lipschitz bound for adaptive refinement.
        Otherwise the scene box is retained. Budget exhaustion keeps uncertain cells.
        """
        from .bounds import shape_bounds

        return shape_bounds(self, scene_bounds, tolerance=tolerance, max_cells=max_cells)

    def extents(self, scene_bounds=None, *, tolerance=0.01, max_cells=65536):
        """Return axis lengths of the conservative bounds (zero for empty shapes)."""
        return self.bounds(scene_bounds, tolerance=tolerance, max_cells=max_cells).extents

    def bvh(self, scene_bounds=None, *, tolerance=0.01, max_cells=65536, leaf_size=1):
        """Build a spatial hierarchy of union components, preserving CSG groups."""
        from .bvh import build_bvh

        return build_bvh(
            self, scene_bounds, tolerance=tolerance, max_cells=max_cells, leaf_size=leaf_size
        )

    def __or__(self, other: Shape) -> Shape:
        return union(self, other)

    def __and__(self, other: Shape) -> Shape:
        return _binary("intersect", self, other)

    def __sub__(self, other: Shape) -> Shape:
        return _binary("subtract", self, other)

    def smooth_union(self, other: Shape, radius: float) -> Shape:
        _same_dim(self, other)
        return Shape("smooth_union", self.dim, (_positive(radius),), (self, other))

    def translate(self, *offset: float) -> Shape:
        return Shape("translate", self.dim, _point(offset, self.dim), (self,))

    def rotate(self, angle: float, axis: str = "z") -> Shape:
        if self.dim == 2 and axis != "z":
            raise ValueError("2D sketches rotate around z")
        if self.dim == 3 and axis not in ("x", "y", "z"):
            raise ValueError("axis must be x, y, or z")
        return Shape("rotate", self.dim, (_number(angle), axis), (self,))

    def scale(self, factor: float) -> Shape:
        return Shape("scale", self.dim, (_positive(factor),), (self,))

    def transform(self, transform: Transform) -> Shape:
        if not isinstance(transform, Transform) or transform.dim != self.dim:
            raise ValueError("shape and transform dimensions must match")
        return Shape("transform", self.dim, (transform,), (self,))

    def place(
        self,
        translation=None,
        *,
        rotation_matrix=None,
        axis_angle=None,
        quaternion=None,
        rpy=None,
    ) -> Shape | PlacedProfile:
        """Place geometry in 3D; rpy angles are applied as Rz @ Ry @ Rx."""
        pose = _Pose.from_components(
            translation, R=rotation_matrix, aa=axis_angle, q=quaternion, rpy=rpy
        )
        if self.dim == 2:
            return PlacedProfile(self, pose)
        return Shape("transform", 3, (pose.as_transform(),), (self,))

    def offset(self, amount: float) -> Shape:
        return Shape("offset", self.dim, (_number(amount),), (self,))

    def shell(self, thickness: float) -> Shape:
        if self.dim != 3:
            raise ValueError("shell requires a 3D solid")
        return Shape("shell", 3, (_positive(thickness),), (self,))

    def linear_pattern(self, count: int, *step: float) -> Shape:
        if not isinstance(count, int) or count < 1:
            raise ValueError("pattern count must be a positive integer")
        delta = _point(step, self.dim)
        return union(*(self.translate(*(i * x for x in delta)) for i in range(count)))

    def radial_pattern(self, count: int, axis: str = "z") -> Shape:
        if self.dim != 3 or not isinstance(count, int) or count < 1:
            raise ValueError("radial pattern requires a 3D solid and positive count")
        return union(*(self.rotate(2 * pi * i / count, axis) for i in range(count)))

    def with_material(self, material_id: int) -> Shape:
        if self.dim != 3:
            raise ValueError("with_material requires a 3D shape")
        return Shape("material", 3, (validate_material_id(material_id),), (self,))

    def extrude(self, height, *, offset=0.0, bounds=None) -> Shape:
        """Extrude along Z; constant height spans offset +/- height/2 (Z=0 by default).

        Height and offset may be JAX functions of (x, y). Heights are full
        thicknesses; nonpositive values have no interior. Callable fields need
        enclosing 3D bounds for automatic bounds computation.
        Variable extrusion produces an implicit distance estimate.
        """
        if self.dim != 2:
            raise ValueError("only 2D sketches can be extruded")
        from .operations import extrude

        return extrude(self, height, offset=offset, bounds=bounds)

    def loft(self, stations, *, axis="z") -> Shape:
        """Loft through (position, offset_u, offset_v, scale_u, scale_v).

        Scales must be positive. Returns a capped implicit distance estimate;
        transverse coordinates follow world axis order (Y/Z for axis='x').
        """
        from .operations import section_loft

        return section_loft(self, stations, axis=axis)

    def sweep(self, curve, *, tolerance=0.01, up=None, closed=False) -> Shape:
        """Sweep along a 2D/3D curve with transported frames and mitered joins.

        Near-reversals and closed paths requiring extra frame twist are rejected.
        """
        from .sweep import sweep

        return sweep(self, curve, tolerance=tolerance, up=up, closed=closed)

    def revolve(self) -> Shape:
        if self.dim != 2:
            raise ValueError("only 2D sketches can be revolved")
        return Shape("revolve", 3, (), (self,))


@dataclass(frozen=True)
class PlacedProfile:
    """A 2D profile embedded in a 3D plane, ready for a solid operation."""

    profile: Shape
    pose: _Pose
    post_transform: Transform | None = None

    def transform(self, transform: Transform) -> PlacedProfile:
        if not isinstance(transform, Transform) or transform.dim != 3:
            raise ValueError("placed profile and transform dimensions must match")
        next_transform = (
            transform if self.post_transform is None else self.post_transform.then(transform)
        )
        return PlacedProfile(self.profile, self.pose, next_transform)

    def place(
        self,
        translation=None,
        *,
        rotation_matrix=None,
        axis_angle=None,
        quaternion=None,
        rpy=None,
    ) -> PlacedProfile:
        next_pose = _Pose.from_components(
            translation, R=rotation_matrix, aa=axis_angle, q=quaternion, rpy=rpy
        )
        if self.post_transform is not None:
            return PlacedProfile(
                self.profile, self.pose, self.post_transform.then(next_pose.as_transform())
            )
        return PlacedProfile(self.profile, self.pose.then(next_pose))

    def extrude(self, height, *, offset=0.0, bounds=None) -> Shape:
        solid = self.profile.extrude(height, offset=offset, bounds=bounds).transform(
            self.pose.as_transform()
        )
        return solid if self.post_transform is None else solid.transform(self.post_transform)

    def revolve(self) -> Shape:
        solid = self.profile.revolve().transform(self.pose.as_transform())
        return solid if self.post_transform is None else solid.transform(self.post_transform)


def _same_dim(a: Shape, b: Shape) -> None:
    if not isinstance(a, Shape) or not isinstance(b, Shape) or a.dim != b.dim:
        raise ValueError("Boolean operands must be shapes of the same dimension")


def _binary(kind: str, a: Shape, b: Shape) -> Shape:
    _same_dim(a, b)
    return Shape(kind, a.dim, (), (a, b))


def union(*shapes: Shape) -> Shape:
    if not shapes:
        raise ValueError("union needs at least one shape")
    if not isinstance(shapes[0], Shape):
        raise ValueError("union needs Shape operands")
    result = shapes[0]
    for shape in shapes[1:]:
        result = _binary("union", result, shape)
    return result


def _custom_count(shape: Shape) -> int:
    return (shape.kind in ("custom", "python")) + sum(
        _custom_count(child) for child in shape.children
    )


# Sketch primitives use the XY plane. Extrusion is centered on Z.
def circle(radius: float) -> Shape:
    return Shape("circle", 2, (_positive(radius),))


def rectangle(width: float, height: float, corner_radius: float = 0) -> Shape:
    w, h, r = _positive(width), _positive(height), _number(corner_radius)
    if r < 0 or r > min(w, h) / 2:
        raise ValueError("corner radius must fit inside the rectangle")
    return Shape("rectangle", 2, (w, h, r))


def polygon(points: Iterable[Iterable[float]]) -> Shape:
    pts = tuple(_point(p, 2) for p in points)
    if len(pts) < 3 or len(set(pts)) < 3:
        raise ValueError("polygon needs three distinct points")
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    if any(pts[i] == pts[(i + 1) % len(pts)] for i in range(len(pts))):
        raise ValueError("polygon has a zero-length edge")
    return Shape("polygon", 2, (pts,))


def loft(sections, *, axis="z") -> Shape:
    """Morph between (position, 2D profile) sections with flat end caps."""
    from .operations import loft as make_loft

    return make_loft(sections, axis=axis)


# Solid primitives use Z as their main axis and are centered at the origin.
def sphere(radius: float) -> Shape:
    return Shape("sphere", 3, (_positive(radius),))


def box(width: float, depth: float, height: float, corner_radius: float = 0) -> Shape:
    w, d, h, r = _positive(width), _positive(depth), _positive(height), _number(corner_radius)
    if r < 0 or r > min(w, d, h) / 2:
        raise ValueError("corner radius must fit inside the box")
    return Shape("box", 3, (w, d, h, r))


def cylinder(radius: float, height: float) -> Shape:
    return Shape("cylinder", 3, (_positive(radius), _positive(height)))


def torus(major_radius: float, tube_radius: float) -> Shape:
    major, tube = _positive(major_radius), _positive(tube_radius)
    return Shape("torus", 3, (major, tube))


def capsule(start: Iterable[float], end: Iterable[float], radius: float) -> Shape:
    a, b = _point(start, 3), _point(end, 3)
    if a == b:
        raise ValueError("capsule endpoints must differ")
    return Shape("capsule", 3, (a, b, _positive(radius)))


def custom_sdf(
    fn, dim: int = 3, material_id: int = 0, *, bounds=None, lipschitz: float | None = None
) -> Shape:
    """Compose a JAX callable; bounds enclose f <= 0 in local coordinates.

    lipschitz promises |f(p)-f(q)| <= L*|p-q|; use 1 for a true SDF.
    """
    if dim not in (2, 3):
        raise ValueError("custom SDF dimension must be 2 or 3")
    if not callable(fn):
        raise ValueError("custom SDF needs a JAX-compatible callable")
    material_id = validate_material_id(material_id)
    from .bounds import as_aabb

    bounds = None if bounds is None else as_aabb(bounds, dim)
    lipschitz = None if lipschitz is None else _positive(lipschitz)
    return Shape("custom", dim, (fn, material_id, bounds, lipschitz))


class Model:
    """Named, ordered CSG history. Every apply step depends on earlier fields."""

    def __init__(self):
        self._shapes: dict[str, Shape] = {}
        self._history: list[dict] = []
        self._current: str | None = None
        self._materials: dict[str, dict] = {}

    def material(
        self, name: str, *, base_color: str, metallic: float = 0.0, roughness: float = 0.8
    ) -> int:
        """Define a named finish and return its ID for Shape geometry."""
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("material name must be a Python identifier")
        if name in self._materials:
            raise ValueError(f"material already exists: {name}")
        index = len(PALETTE) + len(self._materials)
        if index > 4095:
            raise ValueError("model cannot define more than 4087 materials")
        self._materials[name] = {"id": index, **finish(base_color, metallic, roughness)}
        return index

    def materials(self) -> dict[str, dict]:
        """Return JSON-compatible material definitions."""
        return {name: definition.copy() for name, definition in self._materials.items()}

    def _check_name(self, name: str) -> None:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("feature name must be a Python identifier")
        if name in self._shapes:
            raise ValueError(f"feature already exists: {name}")

    def define(self, name: str, shape: Shape) -> Shape:
        """Register an independent field without changing the active result."""
        self._check_name(name)
        if not isinstance(shape, Shape):
            raise ValueError("feature must be a Shape")
        self._shapes[name] = shape
        self._history.append({"name": name, "op": "define", "depends_on": []})
        return shape

    def start(self, name: str, shape: Shape) -> Shape:
        """Register the initial field and make it the active result."""
        if not isinstance(shape, Shape) or shape.dim != 3:
            raise ValueError("model result must be 3D")
        result = self.define(name, shape)
        self._current = name
        return result

    def transform(self, name: str, source: str, transform: Transform) -> Shape:
        """Define a transformed feature from a previously named field."""
        self._check_name(name)
        if source not in self._shapes:
            raise ValueError("source must name a previously defined field")
        result = self._shapes[source].transform(transform)
        self._shapes[name] = result
        self._history.append({"name": name, "op": "transform", "depends_on": [source]})
        return result

    def place(
        self,
        name: str,
        source: str,
        translation=None,
        *,
        rotation_matrix=None,
        axis_angle=None,
        quaternion=None,
        rpy=None,
    ) -> Shape:
        """Define a placed 3D feature from a previously named solid."""
        self._check_name(name)
        if source not in self._shapes:
            raise ValueError("source must name a previously defined field")
        result = self._shapes[source].place(
            translation,
            rotation_matrix=rotation_matrix,
            axis_angle=axis_angle,
            quaternion=quaternion,
            rpy=rpy,
        )
        if not isinstance(result, Shape):
            raise ValueError("place a 2D profile into a solid before adding it to a model")
        self._shapes[name] = result
        self._history.append({"name": name, "op": "place", "depends_on": [source]})
        return result

    def apply(
        self, name: str, op: str, tool: str, *, base: str | None = None, radius: float | None = None
    ) -> Shape:
        """Apply a named field to the active result (or an explicit earlier base)."""
        self._check_name(name)
        base = self._current if base is None else base
        if base not in self._shapes or tool not in self._shapes:
            raise ValueError("base and tool must name previously defined fields")
        if op not in ("union", "intersect", "subtract", "smooth_union"):
            raise ValueError("op must be union, intersect, subtract, or smooth_union")
        if op != "smooth_union" and radius is not None:
            raise ValueError("radius is only valid for smooth_union")
        a, b = self._shapes[base], self._shapes[tool]
        if a.dim != b.dim:
            raise ValueError("base and tool must have the same dimension")
        if a.dim != 3:
            raise ValueError("model result must be 3D")
        if op == "union":
            result = a | b
        elif op == "intersect":
            result = a & b
        elif op == "subtract":
            result = a - b
        else:
            if radius is None:
                raise ValueError("smooth_union requires radius")
            result = a.smooth_union(b, _positive(radius))
        self._shapes[name] = result
        self._history.append(
            {
                "name": name,
                "op": op,
                "depends_on": [base, tool],
                **({"radius": radius} if op == "smooth_union" else {}),
            }
        )
        self._current = name
        return result

    def scene(self, name: str | None = None) -> Shape:
        key = self._current if name is None else name
        if key not in self._shapes or self._shapes[key].dim != 3:
            raise ValueError("scene must name a 3D model field")
        return self._shapes[key]

    def history(self) -> list[dict]:
        """JSON-compatible feature order and direct dependencies."""
        return [dict(step, depends_on=list(step["depends_on"])) for step in self._history]


@dataclass(frozen=True)
class _CurveSegment:
    kind: str
    points: tuple[tuple[float, ...], ...]
    steps: int = 1
    sweep: float = 0.0

    def point(self, t: float) -> tuple[float, ...]:
        p = self.points
        if self.kind == "arc":
            center, start, _ = p
            radius = hypot(start[0] - center[0], start[1] - center[1])
            angle = atan2(start[1] - center[1], start[0] - center[0]) + t * self.sweep
            return (center[0] + radius * cos(angle), center[1] + radius * sin(angle))
        if self.kind == "arc3":
            center, start, _, normal = p
            radial = tuple(start[i] - center[i] for i in range(3))
            side = _cross(normal, radial)
            return tuple(
                center[i] + radial[i] * cos(t * self.sweep) + side[i] * sin(t * self.sweep)
                for i in range(3)
            )
        if self.kind == "line":
            return tuple((1 - t) * a + t * b for a, b in zip(*p, strict=True))
        if self.kind == "quad":
            a, b, c = p
            return tuple(
                (1 - t) ** 2 * a[i] + 2 * (1 - t) * t * b[i] + t * t * c[i] for i in range(len(a))
            )
        a, b, c, d = p
        return tuple(
            (1 - t) ** 3 * a[i]
            + 3 * (1 - t) ** 2 * t * b[i]
            + 3 * (1 - t) * t * t * c[i]
            + t**3 * d[i]
            for i in range(len(a))
        )

    def derivative(self, t: float) -> tuple[float, ...]:
        p = self.points
        if self.kind == "arc":
            center, start, _ = p
            radius = hypot(start[0] - center[0], start[1] - center[1])
            angle = atan2(start[1] - center[1], start[0] - center[0]) + t * self.sweep
            return (-radius * self.sweep * sin(angle), radius * self.sweep * cos(angle))
        if self.kind == "arc3":
            center, start, _, normal = p
            radial = tuple(start[i] - center[i] for i in range(3))
            side = _cross(normal, radial)
            return tuple(
                self.sweep * (-radial[i] * sin(t * self.sweep) + side[i] * cos(t * self.sweep))
                for i in range(3)
            )
        if self.kind == "line":
            return tuple(b - a for a, b in zip(*p, strict=True))
        if self.kind == "quad":
            a, b, c = p
            return tuple(2 * ((1 - t) * (b[i] - a[i]) + t * (c[i] - b[i])) for i in range(len(a)))
        a, b, c, d = p
        return tuple(
            3
            * (
                (1 - t) ** 2 * (b[i] - a[i])
                + 2 * (1 - t) * t * (c[i] - b[i])
                + t * t * (d[i] - c[i])
            )
            for i in range(len(a))
        )


def _cross(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


class Curve:
    """2D or 3D piecewise line, Bézier, and 2D circular arc curve."""

    def __init__(self, *start: float):
        self.dim = len(start)
        if self.dim not in (2, 3):
            raise ValueError("curve start must have two or three coordinates")
        self.start = _point(start, self.dim)
        self._end = self.start
        self.segments: list[_CurveSegment] = []

    @staticmethod
    def _steps(value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 128:
            raise ValueError("curve steps must be an integer from 1 to 128")
        return value

    def line_to(self, *end: float) -> Curve:
        point = _point(end, self.dim)
        if point != self._end:
            self.segments.append(_CurveSegment("line", (self._end, point)))
            self._end = point
        return self

    def quad_to(self, control: Iterable[float], end: Iterable[float], steps: int = 12) -> Curve:
        control, end = _point(control, self.dim), _point(end, self.dim)
        self.segments.append(_CurveSegment("quad", (self._end, control, end), self._steps(steps)))
        self._end = end
        return self

    def cubic_to(
        self, c1: Iterable[float], c2: Iterable[float], end: Iterable[float], steps: int = 16
    ) -> Curve:
        c1, c2, end = _point(c1, self.dim), _point(c2, self.dim), _point(end, self.dim)
        self.segments.append(_CurveSegment("cubic", (self._end, c1, c2, end), self._steps(steps)))
        self._end = end
        return self

    def arc_to(
        self,
        center: Iterable[float],
        end: Iterable[float],
        *,
        clockwise: bool = False,
        normal: Iterable[float] | None = None,
        steps: int = 12,
    ) -> Curve:
        """Append a circular arc; 3D arcs require a plane normal."""
        center, end = _point(center, self.dim), _point(end, self.dim)
        radial = tuple(a - b for a, b in zip(self._end, center, strict=True))
        target = tuple(a - b for a, b in zip(end, center, strict=True))
        r0 = sqrt(sum(v * v for v in radial))
        r1 = sqrt(sum(v * v for v in target))
        if r0 == 0 or end == self._end or abs(r0 - r1) > 1e-7 * r0:
            raise ValueError("arc endpoints must differ and lie on the same circle")
        if self.dim == 2:
            if normal is not None:
                raise ValueError("2D arcs do not take a plane normal")
            a0 = atan2(radial[1], radial[0])
            a1 = atan2(target[1], target[0])
            angle = a1 - a0
            kind = "arc"
            points = (center, self._end, end)
        else:
            if normal is None:
                raise ValueError("3D arcs require a plane normal")
            axis = _point(normal, 3)
            magnitude = sqrt(sum(v * v for v in axis))
            if magnitude == 0:
                raise ValueError("arc plane normal must be nonzero")
            axis = tuple(v / magnitude for v in axis)
            if (
                abs(sum(a * b for a, b in zip(axis, radial, strict=True))) > 1e-7 * r0
                or abs(sum(a * b for a, b in zip(axis, target, strict=True))) > 1e-7 * r1
            ):
                raise ValueError("arc endpoints must lie in the plane normal to the axis")
            side = _cross(axis, radial)
            angle = atan2(
                sum(a * b for a, b in zip(side, target, strict=True)),
                sum(a * b for a, b in zip(radial, target, strict=True)),
            )
            kind = "arc3"
            points = (center, self._end, end, axis)
        sweep = angle % (2 * pi)
        if clockwise:
            sweep -= 2 * pi
        if abs(sweep) < 1e-12:
            raise ValueError("arc sweep must be nonzero")
        self.segments.append(_CurveSegment(kind, points, self._steps(steps), sweep))
        self._end = end
        return self

    def _locate(self, t: float) -> tuple[_CurveSegment, float]:
        if not self.segments:
            raise ValueError("curve has no segments")
        t = _number(t)
        if not 0 <= t <= 1:
            raise ValueError("curve parameter must be between 0 and 1")
        u = t * len(self.segments)
        index = min(int(u), len(self.segments) - 1)
        return self.segments[index], u - index

    def point_at(self, t: float) -> tuple[float, ...]:
        """Evaluate at normalized segment parameter (each command spans 1/N)."""
        segment, u = self._locate(t)
        if u == 1:
            return segment.points[-2] if segment.kind == "arc3" else segment.points[-1]
        return segment.point(u)

    def tangent_at(self, t: float) -> tuple[float, ...]:
        segment, u = self._locate(t)
        derivative = segment.derivative(u)
        length = sqrt(sum(v * v for v in derivative))
        if length == 0:
            raise ValueError("curve tangent is undefined at this parameter")
        return tuple(v / length for v in derivative)

    def sample(self, tolerance: float = 0.01) -> tuple[tuple[float, ...], ...]:
        """Flatten segments, subdividing until curve deviation is within tolerance."""
        tolerance = _positive(tolerance)
        result = [self.start]

        def append_segment(segment: _CurveSegment, left: float, right: float, depth: int) -> None:
            a, b = segment.point(left), segment.point(right)

            def deviation(t: float) -> float:
                point = segment.point(t)
                u = (t - left) / (right - left)
                return sqrt(
                    sum((point[i] - ((1 - u) * a[i] + u * b[i])) ** 2 for i in range(self.dim))
                )

            error = (
                max(deviation(left + (right - left) * u) for u in (0.25, 0.5, 0.75))
                if segment.kind != "line"
                else 0
            )
            if error > tolerance and depth >= 12:
                raise ValueError("curve tolerance requires more subdivision")
            if error > tolerance:
                mid = (left + right) / 2
                append_segment(segment, left, mid, depth + 1)
                append_segment(segment, mid, right, depth + 1)
            elif b != result[-1]:
                result.append(b)

        for segment in self.segments:
            for i in range(segment.steps):
                append_segment(segment, i / segment.steps, (i + 1) / segment.steps, 0)
            result[-1] = segment.points[-2] if segment.kind == "arc3" else segment.points[-1]
        return tuple(result)

    def length(self, tolerance: float = 0.01) -> float:
        points = self.sample(tolerance)
        return sum(
            sqrt(sum((b[i] - a[i]) ** 2 for i in range(self.dim)))
            for a, b in zip(points, points[1:], strict=False)
        )

    def close(self) -> Shape:
        if self.dim != 2:
            raise ValueError("only 2D curves can be closed into a profile")
        if not self.segments:
            raise ValueError("a profile needs at least one curve segment")
        self.line_to(*self.start)
        return Shape("curve_profile", 2, (tuple(self.segments),))

    def tube(self, radius: float, tolerance: float | None = None, *, closed: bool = False) -> Shape:
        if self.dim != 3 or not self.segments:
            raise ValueError("a tube needs a 3D curve with at least one segment")
        radius = _positive(radius)
        points = self.sample(min(0.01, radius / 8) if tolerance is None else tolerance)
        if closed and points[-1] != points[0]:
            points += (points[0],)
        if len(points) < 2:
            raise ValueError("tube curve has no length")
        return Shape("tube", 3, (points, radius))

    def sweep(self, profile: Shape, *, tolerance=0.01, up=None, closed=False) -> Shape:
        """Sweep a 2D profile; up selects its initial local Y direction.

        Near-reversals and closed paths requiring extra frame twist are rejected.
        """
        from .sweep import sweep

        return sweep(profile, self, tolerance=tolerance, up=up, closed=closed)
