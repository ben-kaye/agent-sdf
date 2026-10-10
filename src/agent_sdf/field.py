"""Small CAD-like signed distance field language."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import cos, hypot, isfinite, pi, sin

import numpy as np

from .material import validate_material_id
from .nodes import (
    Box,
    Capsule,
    Custom,
    Cylinder,
    Empty,
    Feature,
    LinearPattern,
    Material,
    Offset,
    Payload,
    Polygon,
    RadialPattern,
    Radius,
    Rectangle,
    Revolve,
    Shell,
    Slice,
    SmoothUnion,
    Torus,
    TransformPayload,
    WarpPayload,
    validate_field,
)
from .plane import Plane, _cross
from .sample import Sample
from .transform import Placement, Transform, _vector_args
from .warp import Warp


def _number(value: float) -> float:
    if np.ndim(value) != 0:
        raise ValueError("coordinates and dimensions must be scalar")
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
    try:
        result = tuple(_number(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected {dim} finite coordinates") from exc
    if len(result) != dim:
        raise ValueError(f"expected {dim} coordinates")
    return result


@dataclass(frozen=True, init=False)
class LoftStation:
    """Profile position, transverse offset, and positive scalar or pair scale."""

    position: float
    offset: tuple[float, float]
    scale: tuple[float, float]

    def __init__(self, position, *, offset=(0, 0), scale=1):
        position = _number(position)
        offset = _point(offset, 2)
        scale = (
            (_positive(scale),) * 2
            if np.ndim(scale) == 0
            else tuple(_positive(value) for value in _point(scale, 2))
        )
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "scale", scale)


@dataclass(frozen=True)
class Field:
    kind: str
    dim: int
    data: Payload = Empty()
    children: tuple[Field, ...] = ()

    def __post_init__(self):
        validate_field(self.kind, self.dim, self.data, self.children)

    def sample(self, points):
        from .numpy_eval import evaluate_field

        return evaluate_field(self, points)

    def __call__(self, *point):
        point = np.asarray(_vector_args(point), dtype=np.float32)
        if point.shape != (self.dim,):
            raise ValueError(f"expected {self.dim} coordinates")
        distance, material = self.sample(point)
        return Sample(float(distance), int(material))

    def sdf(self, *point):
        return self(*point).distance

    def bounds(self, scene_bounds=None, *, tolerance=0.01, max_cells=65536):
        """Return analytic bounds, clipped to scene_bounds when unresolved."""
        from .bounds import field_bounds

        return field_bounds(self, scene_bounds, tolerance=tolerance, max_cells=max_cells)

    def extents(self, scene_bounds=None, *, tolerance=0.01, max_cells=65536):
        """Return axis lengths of the conservative bounds (zero for empty fields)."""
        return self.bounds(scene_bounds, tolerance=tolerance, max_cells=max_cells).extents

    def bvh(self, scene_bounds=None, *, tolerance=0.01, max_cells=65536, leaf_size=1):
        """Build a spatial hierarchy of union components, preserving CSG groups."""
        from .bvh import build_bvh

        return build_bvh(
            self, scene_bounds, tolerance=tolerance, max_cells=max_cells, leaf_size=leaf_size
        )

    def __or__(self, other: Field) -> Field:
        return min_(self, other)

    def __and__(self, other: Field) -> Field:
        return max_(self, other)

    def __sub__(self, other: Field) -> Field:
        return _binary("subtract", self, other)

    def __xor__(self, other: Field) -> Field:
        _same_dim(self, other)
        return (self - other) | (other - self)

    def smooth_union(self, other: Field, radius: float) -> Field:
        _same_dim(self, other)
        return Field("smooth_union", self.dim, SmoothUnion(_positive(radius)), (self, other))

    def translate(self, *offset: float) -> Field:
        return self.transform(Transform.translation(*_point(_vector_args(offset), self.dim)))

    def rotate(self, angle: float, axis: str = "z") -> Field:
        return self.transform(Transform.rotation(angle, axis, dim=self.dim))

    def scale(self, factor: float) -> Field:
        return self.transform(Transform.scaling(*((_positive(factor),) * self.dim)))

    def transform(self, transform: Transform) -> Field:
        if not isinstance(transform, Transform) or transform.dim != self.dim:
            raise ValueError("field and transform dimensions must match")
        return Field("transform", self.dim, TransformPayload(transform), (self,))

    def place(self, placement: Placement) -> Field:
        """Apply a reusable rigid placement to a 3D solid."""
        if self.dim != 3:
            raise ValueError("place requires a 3D solid; use profile.on(plane) for a 2D sketch")
        if not isinstance(placement, Placement):
            raise ValueError("place requires a Placement")
        return self.transform(placement.as_transform())

    def warp(self, warp: Warp, *, bounds) -> Field:
        """Apply a nonlinear map, capped by finite bounds in the output coordinates."""
        from .bounds import as_aabb

        if not isinstance(warp, Warp) or warp.dim != self.dim:
            raise ValueError("field and warp dimensions must match")
        bounds = as_aabb(bounds, self.dim)
        return Field("warp", self.dim, WarpPayload(warp, bounds), (self,))

    def slice(self, plane: Plane) -> Field:
        """Restrict a 3D field exactly to plane-local (x, y), without redistancing."""
        if self.dim != 3 or not isinstance(plane, Plane):
            raise ValueError("slice requires a 3D field and a Plane")
        return Field("slice", 2, Slice(plane), (self,))

    def contours(self, bounds, *, resolution=128, step=None):
        """Sample a 2D field's zero contours in its local coordinates."""
        from .section import contours

        return contours(self, bounds, resolution=resolution, step=step)

    def on(self, plane: Plane) -> Sketch:
        """Place this 2D profile's (x, y) coordinates on a plane."""
        return Sketch(self, plane)

    def offset(self, amount: float) -> Field:
        return Field("offset", self.dim, Offset(_number(amount)), (self,))

    def shell(self, thickness: float) -> Field:
        if self.dim != 3:
            raise ValueError("shell requires a 3D solid")
        return Field("shell", 3, Shell(_positive(thickness)), (self,))

    def linear_pattern(self, count: int, *step: float) -> Field:
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("pattern count must be a positive integer")
        delta = _point(_vector_args(step), self.dim)
        return Field("linear_pattern", self.dim, LinearPattern(count, delta), (self,))

    def radial_pattern(self, count: int, axis: str = "z") -> Field:
        if self.dim != 3 or not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("radial pattern requires a 3D solid and positive count")
        return Field("radial_pattern", 3, RadialPattern(count, axis), (self,))

    def with_material(self, material_id: int) -> Field:
        if self.dim != 3:
            raise ValueError("with_material requires a 3D field")
        return Field("material", 3, Material(validate_material_id(material_id)), (self,))

    def named(self, name: str) -> Field:
        """Add a feature scope, preserving the names of constituent features."""
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("feature name must be a Python identifier")
        return Field("feature", self.dim, Feature(name), (self,))

    def ray_cast(self, origin, direction, **kwargs):
        from .query import ray_cast

        return ray_cast(self, origin, direction, **kwargs)

    def pick(self, x, y, width, height, **kwargs):
        from .query import pick

        return pick(self, x, y, width, height, **kwargs)

    def section(self, plane: Plane, bounds, **kwargs):
        from .section import section

        return section(self, plane, bounds, **kwargs)

    def extrude(
        self, height, *, offset=0.0, bounds=None, height_lipschitz=None, offset_lipschitz=None
    ) -> Field:
        """Extrude along Z; constant height spans offset +/- height/2 (Z=0 by default).

        Height and offset may be symbolic functions of (x, y). Heights are full
        thicknesses; nonpositive values have no interior. Callable fields need
        finite 3D bounds, which cap the resulting solid. Height and offset
        coordinates clamp to those bounds outside their domain.
        Variable extrusion produces an implicit distance estimate.
        """
        if self.dim != 2:
            raise ValueError("only 2D sketches can be extruded")
        from .profiles import extrude

        return extrude(
            self,
            height,
            offset=offset,
            bounds=bounds,
            height_lipschitz=height_lipschitz,
            offset_lipschitz=offset_lipschitz,
        )

    def loft(self, stations: Iterable[LoftStation]) -> Field:
        """Loft along Z, interpolating offsets and scales between LoftStations.

        Returns a flat-capped implicit distance estimate. Use .on(plane).loft for other axes.
        """
        from .profiles import loft

        return loft(self, stations)

    def sweep(
        self, curve, *, tolerance=0.01, up=None, closed=False, orientation="perpendicular"
    ) -> Field:
        """Sweep the profile origin along a 2D/3D curve.

        Perpendicular sweeps transport the profile frame with mitered joins; up
        selects its initial local Y. Near-reversals and closed paths requiring
        extra frame twist are rejected. Parallel sweeps keep the profile in the
        world XY orientation, and the path must never run along that plane.
        """
        from .sweep import sweep

        return sweep(
            self, curve, tolerance=tolerance, up=up, closed=closed, orientation=orientation
        )

    def revolve(self, *, axis, angle: float = 2 * pi) -> Field:
        """Revolve on the default XY plane; see Sketch.revolve."""
        return self.on(Plane.xy()).revolve(axis=axis, angle=angle)


@dataclass(frozen=True)
class Sketch:
    """A 2D profile embedded in a 3D plane, ready for a solid operation."""

    profile: Field
    plane: Plane

    def __post_init__(self):
        if not isinstance(self.profile, Field) or self.profile.dim != 2:
            raise ValueError("sketch requires a 2D profile")
        if not isinstance(self.plane, Plane):
            raise ValueError("sketch requires a Plane")

    def extrude(
        self, height, *, offset=0.0, bounds=None, height_lipschitz=None, offset_lipschitz=None
    ) -> Field:
        """Extrude along the plane normal, centered at offset +/- height/2.

        Fields receive profile-local (x, y); bounds use (x, y, normal) coordinates.
        """
        return self.profile.extrude(
            height,
            offset=offset,
            bounds=bounds,
            height_lipschitz=height_lipschitz,
            offset_lipschitz=offset_lipschitz,
        ).transform(self.plane.as_transform())

    def loft(self, stations: Iterable[LoftStation]) -> Field:
        """Loft along the plane normal; station offsets and scales use profile-local (x, y)."""
        return self.profile.loft(stations).place(self.plane.as_placement())

    def sweep(self, curve, *, tolerance=0.01, closed=False, orientation="perpendicular") -> Field:
        """Sweep the profile, in place, along a world curve starting on the plane.

        The profile point at the curve start follows the path. Perpendicular
        sweeps keep the profile square to the path, starting from the plane's
        local Y; parallel sweeps keep the plane orientation.
        """
        from .sweep import sweep

        return sweep(
            self.profile,
            curve,
            tolerance=tolerance,
            closed=closed,
            orientation=orientation,
            plane=self.plane,
        )

    def revolve(self, *, axis, angle: float = 2 * pi) -> Field:
        """Revolve the profile around a sketch-local axis, starting from the plane.

        axis is "x", "y", or a line through two profile points ((x0, y0), (x1, y1)).
        angle sweeps by the right-hand rule about the axis direction; negative
        angles sweep the other way. Profile points on both sides of the axis revolve.
        """
        start, direction = _revolve_axis(axis)
        angle = _number(angle)
        if not 0 < abs(angle) <= 2 * pi:
            raise ValueError("revolve angle must be nonzero and at most 2*pi in magnitude")
        normal = (direction[1], -direction[0])
        profile = self.profile
        if (start, direction) != ((0.0, 0.0), (0.0, 1.0)):
            profile = profile.transform(
                Transform(
                    (normal, direction),
                    (-_dot2(normal, start), -_dot2(direction, start)),
                )
            )
        plane = self.plane

        def world(u, v):
            return tuple(
                u * x + v * y for x, y in zip(plane.x_direction, plane.y_direction, strict=False)
            )

        radial, axial = world(*normal), world(*direction)
        tangent = _cross(axial, radial)
        if angle < 0:
            c, s = cos(angle), sin(angle)
            radial, tangent = (
                tuple(c * r + s * t for r, t in zip(radial, tangent, strict=False)),
                tuple(c * t - s * r for r, t in zip(radial, tangent, strict=False)),
            )
        origin = tuple(o + w for o, w in zip(plane.origin, world(*start), strict=False))
        rotation = tuple((radial[i], tangent[i], axial[i]) for i in range(3))
        solid = Field("revolve", 3, Revolve(min(abs(angle), 2 * pi)), (profile,))
        return solid.place(Placement(origin, rotation_matrix=rotation))


def _revolve_axis(axis):
    if axis == "x":
        return (0.0, 0.0), (1.0, 0.0)
    if axis == "y":
        return (0.0, 0.0), (0.0, 1.0)
    try:
        (x0, y0), (x1, y1) = axis
        start, end = (_number(x0), _number(y0)), (_number(x1), _number(y1))
    except (TypeError, ValueError) as exc:
        raise ValueError('revolution axis must be "x", "y", or two 2D points') from exc
    length = hypot(end[0] - start[0], end[1] - start[1])
    if length == 0:
        raise ValueError("revolution axis points must differ")
    return start, ((end[0] - start[0]) / length, (end[1] - start[1]) / length)


def _dot2(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _same_dim(a: Field, b: Field) -> None:
    if not isinstance(a, Field) or not isinstance(b, Field) or a.dim != b.dim:
        raise ValueError("Boolean operands must be fields of the same dimension")


def _binary(kind: str, a: Field, b: Field) -> Field:
    _same_dim(a, b)
    return Field(kind, a.dim, Empty(), (a, b))


def _field_reduce(kind: str, fields):
    if len(fields) == 1 and isinstance(fields[0], (list, tuple)):
        fields = tuple(fields[0])
    if not fields:
        raise ValueError(f"{kind}_ needs at least one field")
    if any(isinstance(field, Field) for field in fields):
        if not all(isinstance(field, Field) for field in fields):
            raise TypeError("cannot combine distance fields and scalar fields")
        leaves = []

        def collect(field):
            if field.kind == kind:
                for child in field.children:
                    collect(child)
            else:
                leaves.append(field)

        for field in fields:
            collect(field)

        def balance(first, last):
            if last - first == 1:
                return leaves[first]
            middle = (first + last) // 2
            return _binary(kind, balance(first, middle), balance(middle, last))

        return balance(0, len(leaves))
    from .expr import Scalar

    if not any(isinstance(field, Scalar) for field in fields):
        reducer = np.minimum if kind == "min" else np.maximum
        result = fields[0]
        for field in fields[1:]:
            result = reducer(result, field)
        return result
    from .expr import max_ as scalar_max
    from .expr import min_ as scalar_min

    return (scalar_min if kind == "min" else scalar_max)(*fields)


def min_(*fields):
    """Field-wise minimum (union for signed distance fields)."""
    return _field_reduce("min", fields)


def max_(*fields):
    """Field-wise maximum (intersection for signed distance fields)."""
    return _field_reduce("max", fields)


def _custom_count(field: Field) -> int:
    return (field.kind == "custom") + sum(_custom_count(child) for child in field.children)


# Sketch primitives use the XY plane. Extrusion is centered on Z.
def circle(radius: float) -> Field:
    return Field("circle", 2, Radius(_positive(radius)))


def rectangle(width: float, height: float, corner_radius: float = 0) -> Field:
    w, h, r = _positive(width), _positive(height), _number(corner_radius)
    if r < 0 or r > min(w, h) / 2:
        raise ValueError("corner radius must fit inside the rectangle")
    return Field("rectangle", 2, Rectangle(w, h, r))


def polygon(points: Iterable[Iterable[float]]) -> Field:
    pts = tuple(_point(p, 2) for p in points)
    if len(pts) < 3 or len(set(pts)) < 3:
        raise ValueError("polygon needs three distinct points")
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    if any(pts[i] == pts[(i + 1) % len(pts)] for i in range(len(pts))):
        raise ValueError("polygon has a zero-length edge")
    return Field("polygon", 2, Polygon(pts))


def loft(*sketches: Sketch) -> Field:
    """Blend sketch profiles on parallel planes, in order, into a flat-capped solid.

    Profiles interpolate as distance fields, so disjoint neighbors can pinch or
    vanish between sections. Returns an implicit distance estimate.
    """
    if len(sketches) == 1 and not isinstance(sketches[0], Sketch):
        sketches = tuple(sketches[0])
    if len(sketches) < 2 or not all(isinstance(sketch, Sketch) for sketch in sketches):
        raise ValueError("loft needs at least two sketches")
    from .profiles import loft_sketches

    return loft_sketches(sketches)


# Solid primitives use Z as their main axis and are centered at the origin.
def sphere(radius: float) -> Field:
    return Field("sphere", 3, Radius(_positive(radius)))


def box(width: float, depth: float, height: float, corner_radius: float = 0) -> Field:
    w, d, h, r = _positive(width), _positive(depth), _positive(height), _number(corner_radius)
    if r < 0 or r > min(w, d, h) / 2:
        raise ValueError("corner radius must fit inside the box")
    return Field("box", 3, Box(w, d, h, r))


def cylinder(radius: float, height: float) -> Field:
    return Field("cylinder", 3, Cylinder(_positive(radius), _positive(height)))


def torus(major_radius: float, tube_radius: float) -> Field:
    major, tube = _positive(major_radius), _positive(tube_radius)
    return Field("torus", 3, Torus(major, tube))


def capsule(start: Iterable[float], end: Iterable[float], radius: float) -> Field:
    a, b = _point(start, 3), _point(end, 3)
    if a == b:
        raise ValueError("capsule endpoints must differ")
    return Field("capsule", 3, Capsule(a, b, _positive(radius)))


def custom_sdf(fn, dim: int = 3, material_id: int = 0, *, lipschitz: float, bounds) -> Field:
    """Capture a renderable scalar field capped by finite local bounds.

    The callable receives symbolic coordinates once. Its result must be a
    scalar distance or Sample(distance, material), using field-aware math.
    lipschitz promises |f(p)-f(q)| <= L*|p-q|; use 1 for a true SDF.
    """
    if dim not in (2, 3):
        raise ValueError("custom SDF dimension must be 2 or 3")
    if not callable(fn):
        raise ValueError("custom SDF needs a callable")
    material_id = validate_material_id(material_id)
    from .bounds import as_aabb
    from .expr import as_scalar, compile_glsl, coordinates, to_glsl

    bounds = as_aabb(bounds, dim)
    lipschitz = _positive(lipschitz)
    try:
        value = fn(*coordinates()[:dim])
        distance, material = (
            (value.distance, value.material) if isinstance(value, Sample) else (value, material_id)
        )
        distance, material = as_scalar(distance), as_scalar(material)
        if distance.kind != "float" or material.kind != "float":
            raise TypeError("distance and material must be numeric scalar fields")
        compile_glsl(distance)
        to_glsl(material)
    except Exception as exc:
        raise ValueError(
            "custom_sdf callable must return a renderable scalar field; "
            "use symbolic coordinates and agent_sdf.math operations"
        ) from exc
    return Field("custom", dim, Custom(distance, material, bounds, lipschitz))
