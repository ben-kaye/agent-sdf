"""Typed payloads and construction rules for Shape nodes."""

from __future__ import annotations

from math import isfinite
from typing import Any, NamedTuple


class Empty(NamedTuple):
    pass


class Radius(NamedTuple):
    radius: float


class Rectangle(NamedTuple):
    width: float
    height: float
    corner_radius: float


class Box(NamedTuple):
    width: float
    depth: float
    height: float
    corner_radius: float


class Polygon(NamedTuple):
    vertices: tuple[tuple[float, float], ...]


class CurveProfile(NamedTuple):
    segments: tuple[Any, ...]


class Cylinder(NamedTuple):
    radius: float
    height: float


class Torus(NamedTuple):
    major_radius: float
    tube_radius: float


class Capsule(NamedTuple):
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius: float


class Tube(NamedTuple):
    points: tuple[tuple[float, float, float], ...]
    radius: float


class Custom(NamedTuple):
    fn: Any
    material_id: int
    bounds: Any
    lipschitz: float | None


class SmoothUnion(NamedTuple):
    radius: float


class Translation(NamedTuple):
    offset: tuple[float, ...]


class LinearPattern(NamedTuple):
    count: int
    step: tuple[float, ...]


class RadialPattern(NamedTuple):
    count: int
    axis: str


class Rotation(NamedTuple):
    angle: float
    axis: str


class Scale(NamedTuple):
    factor: float


class TransformPayload(NamedTuple):
    transform: Any


class Offset(NamedTuple):
    amount: float


class Shell(NamedTuple):
    thickness: float


class Material(NamedTuple):
    material_id: int


class Feature(NamedTuple):
    name: str


class Extrude(NamedTuple):
    height: float
    offset: float


class VariableExtrude(NamedTuple):
    height: Any
    offset: Any
    bounds: Any
    height_lipschitz: float | None = None
    offset_lipschitz: float | None = None


class Loft(NamedTuple):
    stations: tuple[float, ...]
    axes: tuple[int, tuple[int, int]]


class SectionLoft(NamedTuple):
    stations: tuple[tuple[float, float, float, float, float], ...]
    axes: tuple[int, tuple[int, int]]


class Sweep(NamedTuple):
    points: tuple[tuple[float, float, float], ...]
    frames: tuple[Any, ...]
    tolerance: float
    closed: bool


Payload = (
    Empty
    | Radius
    | Rectangle
    | Box
    | Polygon
    | CurveProfile
    | Cylinder
    | Torus
    | Capsule
    | Tube
    | Custom
    | SmoothUnion
    | Translation
    | LinearPattern
    | RadialPattern
    | Rotation
    | Scale
    | TransformPayload
    | Offset
    | Shell
    | Material
    | Feature
    | Extrude
    | VariableExtrude
    | Loft
    | SectionLoft
    | Sweep
)

_SPECS = {
    "circle": (Radius, 2, 0),
    "sphere": (Radius, 3, 0),
    "rectangle": (Rectangle, 2, 0),
    "box": (Box, 3, 0),
    "polygon": (Polygon, 2, 0),
    "curve_profile": (CurveProfile, 2, 0),
    "cylinder": (Cylinder, 3, 0),
    "torus": (Torus, 3, 0),
    "capsule": (Capsule, 3, 0),
    "tube": (Tube, 3, 0),
    "custom": (Custom, None, 0),
    "min": (Empty, None, 2),
    "max": (Empty, None, 2),
    "subtract": (Empty, None, 2),
    "smooth_union": (SmoothUnion, None, 2),
    "translate": (Translation, None, 1),
    "linear_pattern": (LinearPattern, None, 1),
    "radial_pattern": (RadialPattern, 3, 1),
    "rotate": (Rotation, None, 1),
    "scale": (Scale, None, 1),
    "transform": (TransformPayload, None, 1),
    "offset": (Offset, None, 1),
    "shell": (Shell, 3, 1),
    "material": (Material, 3, 1),
    "feature": (Feature, None, 1),
    "extrude": (Extrude, 3, 1),
    "variable_extrude": (VariableExtrude, 3, 1),
    "loft": (Loft, 3, None),
    "section_loft": (SectionLoft, 3, 1),
    "profile_sweep": (Sweep, 3, 1),
    "revolve": (Empty, 3, 1),
}


def _finite(value: Any, *, positive: bool = False) -> bool:
    try:
        number = float(value)
    except TypeError, ValueError, OverflowError:
        return False
    return isfinite(number) and (not positive or number > 0)


def _vector(value: Any, dim: int) -> bool:
    return isinstance(value, tuple) and len(value) == dim and all(_finite(x) for x in value)


def validate_shape(kind: str, dim: int, data: Payload, children: tuple[Any, ...]) -> None:
    """Reject malformed nodes before they reach evaluation, bounds, or GLSL."""
    if kind not in _SPECS:
        raise ValueError(f"unknown shape kind: {kind}")
    payload_type, expected_dim, child_count = _SPECS[kind]
    if dim not in (2, 3) or (expected_dim is not None and dim != expected_dim):
        raise ValueError(f"{kind} has an invalid dimension")
    if type(data) is not payload_type:
        raise TypeError(f"{kind} requires {payload_type.__name__} payload")
    if not isinstance(children, tuple) or (
        child_count is not None and len(children) != child_count
    ):
        raise ValueError(f"{kind} has an invalid number of children")
    if any(getattr(child, "dim", None) not in (2, 3) for child in children):
        raise ValueError(f"{kind} children must be shapes")
    if kind in ("min", "max", "subtract", "smooth_union") and any(
        child.dim != dim for child in children
    ):
        raise ValueError(f"{kind} children must match shape dimension")
    if kind in ("extrude", "variable_extrude", "revolve", "section_loft", "profile_sweep"):
        if children[0].dim != 2:
            raise ValueError(f"{kind} requires a 2D profile")
    elif kind == "loft":
        if len(children) < 2 or any(child.dim != 2 for child in children):
            raise ValueError("loft requires at least two 2D profiles")
    elif any(child.dim != dim for child in children):
        raise ValueError(f"{kind} children must match shape dimension")

    if kind in ("circle", "sphere") and not _finite(data.radius, positive=True):
        raise ValueError("radius must be positive and finite")
    if kind in ("rectangle", "box"):
        sizes = (
            (data.width, data.height)
            if kind == "rectangle"
            else (data.width, data.depth, data.height)
        )
        if not all(_finite(size, positive=True) for size in sizes) or not _finite(
            data.corner_radius
        ):
            raise ValueError("box dimensions must be finite and positive")
        if not 0 <= data.corner_radius <= min(sizes) / 2:
            raise ValueError("corner radius must fit inside the box")
    if kind == "cylinder" and not (
        _finite(data.radius, positive=True) and _finite(data.height, positive=True)
    ):
        raise ValueError("cylinder dimensions must be positive and finite")
    if kind == "torus" and not (
        _finite(data.major_radius, positive=True) and _finite(data.tube_radius, positive=True)
    ):
        raise ValueError("torus radii must be positive and finite")
    if kind == "capsule" and not (
        _vector(data.start, 3)
        and _vector(data.end, 3)
        and data.start != data.end
        and _finite(data.radius, positive=True)
    ):
        raise ValueError("capsule needs distinct endpoints and a positive radius")
    if kind == "tube" and not (
        isinstance(data.points, tuple)
        and len(data.points) >= 2
        and all(_vector(point, 3) for point in data.points)
        and _finite(data.radius, positive=True)
    ):
        raise ValueError("tube needs 3D points and a positive radius")
    if kind == "polygon" and not (
        isinstance(data.vertices, tuple)
        and len(data.vertices) >= 3
        and all(_vector(point, 2) for point in data.vertices)
        and len(set(data.vertices)) >= 3
        and all(
            a != b
            for a, b in zip(data.vertices, data.vertices[1:] + data.vertices[:1], strict=True)
        )
    ):
        raise ValueError("polygon needs three distinct finite vertices")
    if kind == "curve_profile" and not (
        isinstance(data.segments, tuple)
        and data.segments
        and all(
            getattr(segment, "kind", None) in ("line", "quad", "arc") for segment in data.segments
        )
    ):
        raise ValueError("curve profile needs valid 2D segments")
    if kind == "custom":
        from .bounds import AABB
        from .materials import validate_material_id

        if not callable(data.fn):
            raise ValueError("custom SDF needs a callable")
        validate_material_id(data.material_id)
        if data.bounds is not None and not isinstance(data.bounds, AABB):
            raise ValueError("custom bounds must be an AABB")
        if data.lipschitz is not None and not _finite(data.lipschitz, positive=True):
            raise ValueError("custom Lipschitz bound must be positive and finite")
    if kind == "smooth_union" and not _finite(data.radius, positive=True):
        raise ValueError("smooth_union requires a positive finite radius")
    if kind == "scale" and not _finite(data.factor, positive=True):
        raise ValueError(f"{kind} requires a positive finite value")
    if kind == "translate" and not _vector(data.offset, dim):
        raise ValueError("translation must match shape dimension")
    if kind == "linear_pattern" and not (
        isinstance(data.count, int)
        and not isinstance(data.count, bool)
        and data.count >= 1
        and _vector(data.step, dim)
        and all(_finite((data.count - 1) * value) for value in data.step)
    ):
        raise ValueError("linear pattern needs a positive count and finite step")
    if kind == "radial_pattern" and not (
        isinstance(data.count, int)
        and not isinstance(data.count, bool)
        and data.count >= 1
        and data.axis in ("x", "y", "z")
    ):
        raise ValueError("radial pattern needs a positive count and valid axis")
    if kind == "rotate" and not (
        _finite(data.angle) and data.axis in (("z",) if dim == 2 else ("x", "y", "z"))
    ):
        raise ValueError("rotation requires a finite angle and valid axis")
    if kind == "transform" and getattr(data.transform, "dim", None) != dim:
        raise ValueError("transform dimension must match shape")
    if kind == "offset" and not _finite(data.amount):
        raise ValueError("offset requires a finite value")
    if kind == "shell" and not _finite(data.thickness, positive=True):
        raise ValueError("shell requires a positive finite thickness")
    if kind == "feature" and not (isinstance(data.name, str) and data.name.isidentifier()):
        raise ValueError("feature name must be a Python identifier")
    if kind == "material":
        from .materials import validate_material_id

        validate_material_id(data.material_id)
    if kind == "extrude" and not (_finite(data.height, positive=True) and _finite(data.offset)):
        raise ValueError("extrude requires a positive height and finite offset")
    if kind == "variable_extrude" and not all(
        callable(value) or _finite(value, positive=field == "height")
        for field, value in (("height", data.height), ("offset", data.offset))
    ):
        raise ValueError("variable extrusion fields must be callable or finite")
    if kind == "loft" and len(data.stations) != len(children):
        raise ValueError("loft stations must match profiles")
    if kind == "section_loft" and not all(
        isinstance(station, tuple)
        and len(station) == 5
        and all(_finite(value) for value in station)
        and station[3] > 0
        and station[4] > 0
        for station in data.stations
    ):
        raise ValueError("section loft stations need finite offsets and positive scales")
    positions = (
        tuple(station[0] for station in data.stations)
        if kind == "section_loft"
        else data.stations
        if kind == "loft"
        else ()
    )
    if kind in ("loft", "section_loft") and not (
        len(positions) >= 2
        and all(_finite(position) for position in positions)
        and all(a < b for a, b in zip(positions, positions[1:], strict=False))
        and data.axes[0] in (0, 1, 2)
        and tuple(sorted((*data.axes[1], data.axes[0]))) == (0, 1, 2)
    ):
        raise ValueError("loft needs ordered stations and valid axes")
    if kind == "profile_sweep" and not (
        len(data.points) >= 2
        and len(data.frames) == len(data.points) - 1
        and all(_vector(point, 3) for point in data.points)
        and all(
            len(frame) == 3 and all(_vector(axis, 3) for axis in frame) for frame in data.frames
        )
        and _finite(data.tolerance, positive=True)
        and isinstance(data.closed, bool)
    ):
        raise ValueError("sweep needs matching points and frames")
