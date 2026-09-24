"""Small CAD-like signed distance field language."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

import numpy as np

from .hit import Hit
from .materials import validate_material_id
from .plane import Plane, _cross
from .shape_payloads import (
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
    Rotation,
    Scale,
    Shell,
    SmoothUnion,
    Torus,
    TransformPayload,
    Translation,
    validate_shape,
)
from .transform import Transform, _Pose, _vector_args


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


@dataclass(frozen=True)
class Shape:
    kind: str
    dim: int
    data: Payload = Empty()
    children: tuple[Shape, ...] = ()

    def __post_init__(self):
        validate_shape(self.kind, self.dim, self.data, self.children)

    def sample(self, points):
        from .backend import evaluate_shape

        return evaluate_shape(self, points)

    def __call__(self, *point):
        point = np.asarray(_vector_args(point), dtype=np.float32)
        if point.shape != (self.dim,):
            raise ValueError(f"expected {self.dim} coordinates")
        distance, material = self.sample(point)
        return Hit(float(distance), int(material))

    def sdf(self, *point):
        return self(*point).distance

    def bounds(self, scene_bounds=None, *, tolerance=0.01, max_cells=65536):
        """Return analytic bounds, clipped to scene_bounds when unresolved."""
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
        return min_(self, other)

    def __and__(self, other: Shape) -> Shape:
        return max_(self, other)

    def __sub__(self, other: Shape) -> Shape:
        return _binary("subtract", self, other)

    def __xor__(self, other: Shape) -> Shape:
        _same_dim(self, other)
        return (self - other) | (other - self)

    def smooth_union(self, other: Shape, radius: float) -> Shape:
        _same_dim(self, other)
        return Shape("smooth_union", self.dim, SmoothUnion(_positive(radius)), (self, other))

    def translate(self, *offset: float) -> Shape:
        return Shape(
            "translate", self.dim, Translation(_point(_vector_args(offset), self.dim)), (self,)
        )

    def rotate(self, angle: float, axis: str = "z") -> Shape:
        if self.dim == 2 and axis != "z":
            raise ValueError("2D sketches rotate around z")
        if self.dim == 3 and axis not in ("x", "y", "z"):
            raise ValueError("axis must be x, y, or z")
        return Shape("rotate", self.dim, Rotation(_number(angle), axis), (self,))

    def scale(self, factor: float) -> Shape:
        return Shape("scale", self.dim, Scale(_positive(factor)), (self,))

    def transform(self, transform: Transform) -> Shape:
        if not isinstance(transform, Transform) or transform.dim != self.dim:
            raise ValueError("shape and transform dimensions must match")
        return Shape("transform", self.dim, TransformPayload(transform), (self,))

    def place(
        self,
        translation=None,
        *,
        rotation_matrix=None,
        axis_angle=None,
        quaternion=None,
        rpy=None,
    ) -> Shape:
        """Place a 3D solid; rpy angles are applied as Rz @ Ry @ Rx."""
        if self.dim != 3:
            raise ValueError("place requires a 3D solid; use profile.on(plane) for a 2D sketch")
        pose = _Pose.from_components(
            translation, R=rotation_matrix, aa=axis_angle, q=quaternion, rpy=rpy
        )
        return Shape("transform", 3, TransformPayload(pose.as_transform()), (self,))

    def on(self, plane: Plane) -> Sketch:
        """Place this 2D profile's (x, y) coordinates on a plane."""
        return Sketch(self, plane)

    def offset(self, amount: float) -> Shape:
        return Shape("offset", self.dim, Offset(_number(amount)), (self,))

    def shell(self, thickness: float) -> Shape:
        if self.dim != 3:
            raise ValueError("shell requires a 3D solid")
        return Shape("shell", 3, Shell(_positive(thickness)), (self,))

    def linear_pattern(self, count: int, *step: float) -> Shape:
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("pattern count must be a positive integer")
        delta = _point(_vector_args(step), self.dim)
        return Shape("linear_pattern", self.dim, LinearPattern(count, delta), (self,))

    def radial_pattern(self, count: int, axis: str = "z") -> Shape:
        if self.dim != 3 or not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("radial pattern requires a 3D solid and positive count")
        return Shape("radial_pattern", 3, RadialPattern(count, axis), (self,))

    def with_material(self, material_id: int) -> Shape:
        if self.dim != 3:
            raise ValueError("with_material requires a 3D shape")
        return Shape("material", 3, Material(validate_material_id(material_id)), (self,))

    def named(self, name: str) -> Shape:
        """Name this field as one surface feature for geometry queries."""
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("feature name must be a Python identifier")
        return Shape("feature", self.dim, Feature(name), (self,))

    def ray_cast(self, origin, direction, **kwargs):
        from .query import ray_cast

        return ray_cast(self, origin, direction, **kwargs)

    def pick(self, x, y, width, height, **kwargs):
        from .query import pick

        return pick(self, x, y, width, height, **kwargs)

    def section(self, origin, normal, bounds, **kwargs):
        from .section import section

        return section(self, origin, normal, bounds, **kwargs)

    def extrude(
        self, height, *, offset=0.0, bounds=None, height_lipschitz=None, offset_lipschitz=None
    ) -> Shape:
        """Extrude along Z; constant height spans offset +/- height/2 (Z=0 by default).

        Height and offset may be symbolic functions of (x, y). Heights are full
        thicknesses; nonpositive values have no interior. Callable fields need
        enclosing 3D bounds for automatic bounds computation.
        Variable extrusion produces an implicit distance estimate.
        """
        if self.dim != 2:
            raise ValueError("only 2D sketches can be extruded")
        from .operations import extrude

        return extrude(
            self,
            height,
            offset=offset,
            bounds=bounds,
            height_lipschitz=height_lipschitz,
            offset_lipschitz=offset_lipschitz,
        )

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

    def revolve(self, *, axis: str) -> Shape:
        """Revolve around the profile's X or Y axis on the default XY plane."""
        return self.on(Plane.xy()).revolve(axis=axis)


@dataclass(frozen=True)
class Sketch:
    """A 2D profile embedded in a 3D plane, ready for a solid operation."""

    profile: Shape
    plane: Plane

    def __post_init__(self):
        if not isinstance(self.profile, Shape) or self.profile.dim != 2:
            raise ValueError("sketch requires a 2D profile")
        if not isinstance(self.plane, Plane):
            raise ValueError("sketch requires a Plane")

    def extrude(
        self, height, *, offset=0.0, bounds=None, height_lipschitz=None, offset_lipschitz=None
    ) -> Shape:
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

    def revolve(self, *, axis: str) -> Shape:
        """Revolve around the plane's local X or Y axis through its origin."""
        if axis not in ("x", "y"):
            raise ValueError("revolution axis must be sketch-local x or y")
        profile = self.profile
        if axis == "x":
            profile = profile.transform(
                Transform(
                    2,
                    lambda p: p[::-1],
                    forward_affine=(((0.0, 1.0), (1.0, 0.0)), (0.0, 0.0)),
                )
            )
        profile = profile | profile.transform(Transform.scaling(-1, 1))
        solid = Shape("revolve", 3, Empty(), (profile,))
        radial, axial = (
            (self.plane.y_direction, self.plane.x_direction)
            if axis == "x"
            else (self.plane.x_direction, self.plane.y_direction)
        )
        tangent = _cross(axial, radial)
        rotation = tuple((radial[i], tangent[i], axial[i]) for i in range(3))
        return solid.transform(_Pose(self.plane.origin, rotation).as_transform())


def _same_dim(a: Shape, b: Shape) -> None:
    if not isinstance(a, Shape) or not isinstance(b, Shape) or a.dim != b.dim:
        raise ValueError("Boolean operands must be shapes of the same dimension")


def _binary(kind: str, a: Shape, b: Shape) -> Shape:
    _same_dim(a, b)
    return Shape(kind, a.dim, Empty(), (a, b))


def _field_reduce(kind: str, fields):
    if len(fields) == 1 and isinstance(fields[0], (list, tuple)):
        fields = tuple(fields[0])
    if not fields:
        raise ValueError(f"{kind}_ needs at least one field")
    if any(isinstance(field, Shape) for field in fields):
        if not all(isinstance(field, Shape) for field in fields):
            raise TypeError("cannot combine shapes and scalar fields")
        leaves = []

        def collect(shape):
            if shape.kind == kind:
                for child in shape.children:
                    collect(child)
            else:
                leaves.append(shape)

        for field in fields:
            collect(field)

        def balance(first, last):
            if last - first == 1:
                return leaves[first]
            middle = (first + last) // 2
            return _binary(kind, balance(first, middle), balance(middle, last))

        return balance(0, len(leaves))
    from .field import Scalar

    if not any(isinstance(field, Scalar) for field in fields):
        reducer = np.minimum if kind == "min" else np.maximum
        result = fields[0]
        for field in fields[1:]:
            result = reducer(result, field)
        return result
    from .field import max_ as scalar_max
    from .field import min_ as scalar_min

    return (scalar_min if kind == "min" else scalar_max)(*fields)


def min_(*fields):
    """Field-wise minimum (union for signed distance fields)."""
    return _field_reduce("min", fields)


def max_(*fields):
    """Field-wise maximum (intersection for signed distance fields)."""
    return _field_reduce("max", fields)


def _custom_count(shape: Shape) -> int:
    return (shape.kind == "custom") + sum(_custom_count(child) for child in shape.children)


# Sketch primitives use the XY plane. Extrusion is centered on Z.
def circle(radius: float) -> Shape:
    return Shape("circle", 2, Radius(_positive(radius)))


def rectangle(width: float, height: float, corner_radius: float = 0) -> Shape:
    w, h, r = _positive(width), _positive(height), _number(corner_radius)
    if r < 0 or r > min(w, h) / 2:
        raise ValueError("corner radius must fit inside the rectangle")
    return Shape("rectangle", 2, Rectangle(w, h, r))


def polygon(points: Iterable[Iterable[float]]) -> Shape:
    pts = tuple(_point(p, 2) for p in points)
    if len(pts) < 3 or len(set(pts)) < 3:
        raise ValueError("polygon needs three distinct points")
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    if any(pts[i] == pts[(i + 1) % len(pts)] for i in range(len(pts))):
        raise ValueError("polygon has a zero-length edge")
    return Shape("polygon", 2, Polygon(pts))


def loft(sections, *, axis="z") -> Shape:
    """Morph between (position, 2D profile) sections with flat end caps."""
    from .operations import loft as make_loft

    return make_loft(sections, axis=axis)


# Solid primitives use Z as their main axis and are centered at the origin.
def sphere(radius: float) -> Shape:
    return Shape("sphere", 3, Radius(_positive(radius)))


def box(width: float, depth: float, height: float, corner_radius: float = 0) -> Shape:
    w, d, h, r = _positive(width), _positive(depth), _positive(height), _number(corner_radius)
    if r < 0 or r > min(w, d, h) / 2:
        raise ValueError("corner radius must fit inside the box")
    return Shape("box", 3, Box(w, d, h, r))


def cylinder(radius: float, height: float) -> Shape:
    return Shape("cylinder", 3, Cylinder(_positive(radius), _positive(height)))


def torus(major_radius: float, tube_radius: float) -> Shape:
    major, tube = _positive(major_radius), _positive(tube_radius)
    return Shape("torus", 3, Torus(major, tube))


def capsule(start: Iterable[float], end: Iterable[float], radius: float) -> Shape:
    a, b = _point(start, 3), _point(end, 3)
    if a == b:
        raise ValueError("capsule endpoints must differ")
    return Shape("capsule", 3, Capsule(a, b, _positive(radius)))


def custom_sdf(
    fn, dim: int = 3, material_id: int = 0, *, bounds=None, lipschitz: float | None = None
) -> Shape:
    """Compose a symbolic distance field; bounds enclose f <= 0 in local coordinates.

    lipschitz promises |f(p)-f(q)| <= L*|p-q|; use 1 for a true SDF.
    """
    if dim not in (2, 3):
        raise ValueError("custom SDF dimension must be 2 or 3")
    if not callable(fn):
        raise ValueError("custom SDF needs a callable")
    material_id = validate_material_id(material_id)
    from .bounds import as_aabb

    bounds = None if bounds is None else as_aabb(bounds, dim)
    lipschitz = None if lipschitz is None else _positive(lipschitz)
    return Shape("custom", dim, Custom(fn, material_id, bounds, lipschitz))
