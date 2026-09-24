"""Reusable affine transforms and rigid placements."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin, sqrt

import numpy as np


def _finite(value: float) -> float:
    if np.ndim(value) != 0:
        raise ValueError("transform values must be scalar")
    result = float(value)
    if not isfinite(result):
        raise ValueError("transform values must be finite")
    return result


def _vector_args(values):
    if len(values) == 1 and not isinstance(values[0], (str, bytes)):
        try:
            return tuple(values[0])
        except TypeError:
            pass
    return values


def _vector3(value, name):
    try:
        result = tuple(_finite(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have three finite coordinates") from exc
    if len(result) != 3:
        raise ValueError(f"{name} must have three finite coordinates")
    return result


def _matrix3(value):
    try:
        rows = tuple(_vector3(row, "rotation row") for row in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("rotation_matrix must be a 3x3 rotation matrix") from exc
    if len(rows) != 3:
        raise ValueError("rotation_matrix must be a 3x3 rotation matrix")
    columns = tuple(zip(*rows, strict=True))
    for i, col in enumerate(columns):
        for j, other in enumerate(columns):
            dot = sum(a * b for a, b in zip(col, other, strict=True))
            if abs(dot - (1.0 if i == j else 0.0)) > 1e-12:
                raise ValueError("rotation_matrix must be a proper rotation matrix")
    cross = (
        columns[0][1] * columns[1][2] - columns[0][2] * columns[1][1],
        columns[0][2] * columns[1][0] - columns[0][0] * columns[1][2],
        columns[0][0] * columns[1][1] - columns[0][1] * columns[1][0],
    )
    if abs(sum(a * b for a, b in zip(cross, columns[2], strict=True)) - 1.0) > 1e-12:
        raise ValueError("rotation_matrix must be a proper rotation matrix")
    return rows


def _quaternion_matrix(quaternion):
    try:
        w, x, y, z = tuple(_finite(item) for item in quaternion)
    except (TypeError, ValueError) as exc:
        raise ValueError("quaternion must be (w, x, y, z)") from exc
    scale = max(abs(value) for value in (w, x, y, z))
    if scale == 0:
        raise ValueError("quaternion must be nonzero")
    scaled = tuple(item / scale for item in (w, x, y, z))
    length = sqrt(sum(item * item for item in scaled))
    w, x, y, z = (item / length for item in scaled)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


@dataclass(frozen=True, init=False)
class Transform:
    """Invertible local-to-world affine map in two or three dimensions.

    distance_scale is the smallest singular value: exact for similarity maps
    and conservative for other affine maps. Composition recomputes it from
    the combined matrix.
    """

    matrix: tuple[tuple[float, ...], ...]
    offset: tuple[float, ...]
    inverse_matrix: tuple[tuple[float, ...], ...]
    distance_scale: float
    dim: int

    def __init__(self, matrix, translation=None):
        try:
            rows = tuple(tuple(_finite(value) for value in row) for row in matrix)
        except (TypeError, ValueError) as exc:
            raise ValueError("matrix must be a finite 2x2 or 3x3 matrix") from exc
        dim = len(rows)
        if dim not in (2, 3) or any(len(row) != dim for row in rows):
            raise ValueError("matrix must be a finite 2x2 or 3x3 matrix")
        if translation is None:
            offset = (0.0,) * dim
        else:
            try:
                offset = tuple(_finite(value) for value in translation)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"translation must have {dim} finite coordinates") from exc
            if len(offset) != dim:
                raise ValueError(f"translation must have {dim} finite coordinates")
        array = np.asarray(rows, dtype=float)
        try:
            singular_values = np.linalg.svd(array, compute_uv=False)
            inverse = np.linalg.inv(array)
        except np.linalg.LinAlgError as exc:
            raise ValueError("matrix must be nonsingular") from exc
        scale = float(singular_values[-1])
        if scale <= 0 or not isfinite(scale) or not np.isfinite(inverse).all():
            raise ValueError("matrix must be nonsingular")
        object.__setattr__(self, "matrix", rows)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "inverse_matrix", tuple(tuple(row) for row in inverse.tolist()))
        object.__setattr__(self, "distance_scale", scale)
        object.__setattr__(self, "dim", dim)

    @classmethod
    def translation(cls, *offset: float) -> Transform:
        values = tuple(_finite(value) for value in _vector_args(offset))
        if len(values) not in (2, 3):
            raise ValueError("translation needs two or three coordinates")
        dim = len(values)
        return cls(tuple(tuple(float(i == j) for j in range(dim)) for i in range(dim)), values)

    @classmethod
    def scaling(cls, *factors: float) -> Transform:
        values = tuple(_finite(value) for value in _vector_args(factors))
        if len(values) not in (2, 3) or any(value == 0 for value in values):
            raise ValueError("scaling needs two or three nonzero factors")
        dim = len(values)
        return cls(
            tuple(tuple(values[i] if i == j else 0.0 for j in range(dim)) for i in range(dim))
        )

    @classmethod
    def rotation(cls, angle: float, axis: str = "z", *, dim: int = 3) -> Transform:
        angle = _finite(angle)
        if dim not in (2, 3) or axis not in ("x", "y", "z") or (dim == 2 and axis != "z"):
            raise ValueError("rotation axis must be valid for the dimension")
        c, s = cos(angle), sin(angle)
        if dim == 2:
            return cls(((c, -s), (s, c)))
        if axis == "x":
            return cls(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))
        if axis == "y":
            return cls(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))
        return cls(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))

    def then(self, other: Transform) -> Transform:
        """Apply this transform, then other, in world coordinates."""
        if not isinstance(other, Transform) or self.dim != other.dim:
            raise ValueError("composed transforms must have the same dimension")
        first, second = np.asarray(self.matrix), np.asarray(other.matrix)
        matrix = second @ first
        offset = second @ np.asarray(self.offset) + np.asarray(other.offset)
        return Transform(matrix, offset)

    def map_points(self, points):
        """Map world points to local points, preserving any batch dimensions."""
        points = np.asarray(points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != self.dim:
            raise ValueError(f"points must end in {self.dim} coordinates")
        return (points - np.asarray(self.offset)) @ np.asarray(self.inverse_matrix).T

    def map_python(self, *point: float) -> tuple[float, ...]:
        point = np.asarray(_vector_args(point), dtype=float)
        if point.shape != (self.dim,):
            raise ValueError(f"point must have {self.dim} coordinates")
        values = tuple(float(value) for value in self.map_points(point))
        if not all(isfinite(value) for value in values):
            raise ValueError("mapped point contains nonfinite coordinates")
        return values

    def __call__(self, primitive):
        return primitive.transform(self)


@dataclass(frozen=True, init=False)
class Placement:
    """Rigid local-to-world placement; angles are radians."""

    translation: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]

    def __init__(
        self,
        translation=(0, 0, 0),
        *,
        rotation_matrix=None,
        axis_angle=None,
        quaternion=None,
        rpy=None,
    ):
        if sum(value is not None for value in (rotation_matrix, axis_angle, quaternion, rpy)) > 1:
            raise ValueError("provide only one rotation representation")
        location = _vector3(translation, "translation")
        if rotation_matrix is not None:
            rotation = _matrix3(rotation_matrix)
        elif axis_angle is not None:
            try:
                axis, angle = axis_angle
            except (TypeError, ValueError) as exc:
                raise ValueError("axis_angle must be (axis, angle)") from exc
            axis = _vector3(axis, "axis")
            angle = _finite(angle)
            scale = max(abs(item) for item in axis)
            if scale == 0:
                raise ValueError("axis must be nonzero")
            scaled = tuple(item / scale for item in axis)
            length = sqrt(sum(item * item for item in scaled))
            x, y, z = (item / length for item in scaled)
            half = angle / 2
            rotation = _quaternion_matrix((cos(half), x * sin(half), y * sin(half), z * sin(half)))
        elif quaternion is not None:
            rotation = _quaternion_matrix(quaternion)
        elif rpy is not None:
            roll, pitch, yaw = _vector3(rpy, "rpy")
            cr, sr = cos(roll), sin(roll)
            cp, sp = cos(pitch), sin(pitch)
            cy, sy = cos(yaw), sin(yaw)
            rotation = (
                (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                (-sp, cp * sr, cp * cr),
            )
        else:
            rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        object.__setattr__(self, "translation", location)
        object.__setattr__(self, "rotation", rotation)

    def as_transform(self) -> Transform:
        return Transform(self.rotation, self.translation)

    def then(self, other: Placement) -> Placement:
        """Apply this placement, then other, in world coordinates."""
        if not isinstance(other, Placement):
            raise ValueError("composed placements must both be Placement values")
        first, second = np.asarray(self.rotation), np.asarray(other.rotation)
        rotation = second @ first
        translation = second @ np.asarray(self.translation) + np.asarray(other.translation)
        return Placement(translation, rotation_matrix=rotation)

    def __call__(self, primitive):
        return primitive.transform(self.as_transform())
