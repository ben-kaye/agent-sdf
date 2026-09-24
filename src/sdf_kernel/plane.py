"""Oriented planes for placing 2D profiles in world coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from .transform import Transform, _finite, _Pose, _vector3


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _unit(vector, message):
    scale = max(abs(value) for value in vector)
    if scale < 1e-12:
        raise ValueError(message)
    scaled = tuple(value / scale for value in vector)
    length = hypot(*scaled)
    return tuple(value / length for value in scaled)


def _local_point(point):
    try:
        result = tuple(_finite(value) for value in point)
    except (TypeError, ValueError) as exc:
        raise ValueError("local point must have two finite coordinates") from exc
    if len(result) != 2:
        raise ValueError("local point must have two finite coordinates")
    return result


@dataclass(frozen=True)
class Plane:
    """Local (x, y) axes with x cross y = normal; directions are world vectors."""

    origin: tuple[float, float, float]
    normal: tuple[float, float, float]
    x_direction: tuple[float, float, float] = field(kw_only=True)
    y_direction: tuple[float, float, float] = field(init=False)

    def __post_init__(self):
        origin = _vector3(self.origin, "origin")
        normal = _unit(_vector3(self.normal, "normal"), "normal must be nonzero")
        reference = _unit(_vector3(self.x_direction, "x_direction"), "x_direction must be nonzero")
        projection = tuple(reference[i] - _dot(reference, normal) * normal[i] for i in range(3))
        x_direction = _unit(projection, "x_direction must not be parallel to normal")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "x_direction", x_direction)
        object.__setattr__(self, "y_direction", _cross(normal, x_direction))

    @classmethod
    def xy(cls, *, z=0.0) -> Plane:
        """Local (x, y) maps to world (X, Y); normal is +Z."""
        return cls((0, 0, z), (0, 0, 1), x_direction=(1, 0, 0))

    @classmethod
    def xz(cls, *, y=0.0) -> Plane:
        """Local (x, y) maps to world (X, Z); normal is -Y."""
        return cls((0, y, 0), (0, -1, 0), x_direction=(1, 0, 0))

    @classmethod
    def yz(cls, *, x=0.0) -> Plane:
        """Local (x, y) maps to world (Y, Z); normal is +X."""
        return cls((x, 0, 0), (1, 0, 0), x_direction=(0, 1, 0))

    def local_to_world(self, point):
        u, v = _local_point(point)
        return tuple(
            self.origin[i] + u * self.x_direction[i] + v * self.y_direction[i] for i in range(3)
        )

    def world_to_local(self, point, *, tolerance=1e-8):
        world = _vector3(point, "world point")
        tolerance = _finite(tolerance)
        if tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        delta = tuple(world[i] - self.origin[i] for i in range(3))
        if abs(_dot(delta, self.normal)) > tolerance:
            raise ValueError("world point must lie on the plane")
        return _dot(delta, self.x_direction), _dot(delta, self.y_direction)

    def as_transform(self) -> Transform:
        """Map local (x, y, normal) coordinates into world coordinates."""
        rotation = tuple(
            (self.x_direction[i], self.y_direction[i], self.normal[i]) for i in range(3)
        )
        return _Pose(self.origin, rotation).as_transform()
