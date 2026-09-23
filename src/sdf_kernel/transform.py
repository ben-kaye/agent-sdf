"""Reusable coordinate transforms for signed distance fields."""

from collections.abc import Callable
from dataclasses import dataclass, field
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
        rows = tuple(_vector3(row, "R row") for row in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("R must be a 3x3 rotation matrix") from exc
    if len(rows) != 3:
        raise ValueError("R must be a 3x3 rotation matrix")
    columns = tuple(zip(*rows, strict=True))
    for i, col in enumerate(columns):
        for j, other in enumerate(columns):
            dot = sum(a * b for a, b in zip(col, other, strict=True))
            if abs(dot - (1.0 if i == j else 0.0)) > 1e-5:
                raise ValueError("R must be a proper rotation matrix")
    cross = (
        columns[0][1] * columns[1][2] - columns[0][2] * columns[1][1],
        columns[0][2] * columns[1][0] - columns[0][0] * columns[1][2],
        columns[0][0] * columns[1][1] - columns[0][1] * columns[1][0],
    )
    if sum(a * b for a, b in zip(cross, columns[2], strict=True)) < 0:
        raise ValueError("R must be a proper rotation matrix")
    return rows


def _quaternion_matrix(quaternion):
    try:
        w, x, y, z = tuple(_finite(item) for item in quaternion)
    except (TypeError, ValueError) as exc:
        raise ValueError("q must be (w, x, y, z)") from exc
    length = sqrt(w * w + x * x + y * y + z * z)
    if length == 0:
        raise ValueError("q must be nonzero")
    w, x, y, z = (item / length for item in (w, x, y, z))
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


@dataclass(frozen=True)
class _Pose:
    """Rigid local-to-world placement; angles are radians."""

    translation: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]

    @classmethod
    def from_components(cls, t=None, *, R=None, aa=None, q=None, rpy=None):
        if sum(value is not None for value in (R, aa, q, rpy)) > 1:
            raise ValueError("provide only one of R, aa, q, or rpy")
        if t is None or (isinstance(t, (int, float)) and t == 0):
            translation = (0.0, 0.0, 0.0)
        else:
            translation = _vector3(t, "t")
        if R is not None:
            rotation = _matrix3(R)
        elif aa is not None:
            try:
                axis, angle = aa
            except (TypeError, ValueError) as exc:
                raise ValueError("aa must be (axis, angle)") from exc
            axis = _vector3(axis, "aa axis")
            angle = _finite(angle)
            length = sqrt(sum(item * item for item in axis))
            if length == 0:
                raise ValueError("aa axis must be nonzero")
            x, y, z = (item / length for item in axis)
            half = angle / 2
            rotation = _quaternion_matrix((cos(half), x * sin(half), y * sin(half), z * sin(half)))
        elif q is not None:
            rotation = _quaternion_matrix(q)
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
        return cls(translation, rotation)

    def as_transform(self):
        translation = np.asarray(self.translation, dtype=float)
        rotation = np.asarray(self.rotation, dtype=float)
        return Transform(
            3,
            lambda point: rotation.T @ (point - translation),
            forward_affine=(self.rotation, self.translation),
        )

    def then(self, other):
        """Apply this pose, then another pose in world coordinates."""
        a, b = self.rotation, other.rotation
        rotation = tuple(
            tuple(sum(b[i][k] * a[k][j] for k in range(3)) for j in range(3)) for i in range(3)
        )
        translation = tuple(
            sum(b[i][k] * self.translation[k] for k in range(3)) + other.translation[i]
            for i in range(3)
        )
        return _Pose(translation, rotation)


@dataclass(frozen=True)
class Transform:
    """A point map from world coordinates into primitive local coordinates.

    `inverse` receives one point of shape `(dim,)` and returns the same
    shape. `distance_scale` converts local distances to world distances; a
    nonlinear map needs a conservative scale for safe sphere tracing.
    `forward_affine`, when supplied, promises the matching local-to-world
    (matrix, translation) map for bounds propagation.
    """

    dim: int
    inverse: Callable
    distance_scale: float = 1.0
    forward_affine: tuple[tuple[tuple[float, ...], ...], tuple[float, ...]] | None = None
    _single: Callable = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.dim not in (2, 3):
            raise ValueError("transform dimension must be 2 or 3")
        if not callable(self.inverse):
            raise ValueError("inverse map must be callable")
        scale = _finite(self.distance_scale)
        if scale <= 0:
            raise ValueError("distance scale must be positive")
        object.__setattr__(self, "distance_scale", scale)
        if self.forward_affine is not None:
            try:
                matrix, translation = self.forward_affine
                matrix = tuple(tuple(_finite(value) for value in row) for row in matrix)
                translation = tuple(_finite(value) for value in translation)
            except (TypeError, ValueError) as exc:
                raise ValueError("forward affine map must contain finite values") from exc
            if (
                len(matrix) != self.dim
                or any(len(row) != self.dim for row in matrix)
                or len(translation) != self.dim
            ):
                raise ValueError(
                    f"forward affine map must have shape ({self.dim}, {self.dim}) and ({self.dim},)"
                )
            object.__setattr__(self, "forward_affine", (matrix, translation))

        try:
            result = np.asarray(self.inverse(np.zeros(self.dim, dtype=float)), dtype=float)
        except Exception as exc:
            raise ValueError(f"inverse map must accept a numeric point: {exc}") from exc
        if result.shape != (self.dim,):
            raise ValueError(f"inverse map must return a point of shape ({self.dim},)")
        object.__setattr__(self, "_single", self.inverse)

    @classmethod
    def translation(cls, *offset: float) -> Transform:
        values = tuple(_finite(value) for value in _vector_args(offset))
        if len(values) not in (2, 3):
            raise ValueError("translation needs two or three coordinates")
        delta = np.asarray(values, dtype=float)
        identity = tuple(
            tuple(float(i == j) for j in range(len(values))) for i in range(len(values))
        )
        return cls(len(values), lambda point: point - delta, forward_affine=(identity, values))

    @classmethod
    def scaling(cls, *factors: float) -> Transform:
        values = tuple(_finite(value) for value in _vector_args(factors))
        if len(values) not in (2, 3) or any(value == 0 for value in values):
            raise ValueError("scaling needs two or three nonzero factors")
        factors_array = np.asarray(values, dtype=float)
        return cls(
            len(values),
            lambda point: point / factors_array,
            distance_scale=min(abs(value) for value in values),
            forward_affine=(
                tuple(
                    tuple(values[i] if i == j else 0.0 for j in range(len(values)))
                    for i in range(len(values))
                ),
                (0.0,) * len(values),
            ),
        )

    @classmethod
    def rotation(cls, angle: float, axis: str = "z", *, dim: int = 3) -> Transform:
        angle = _finite(angle)
        if dim not in (2, 3) or axis not in ("x", "y", "z") or (dim == 2 and axis != "z"):
            raise ValueError("rotation axis must be valid for the dimension")
        c, s = cos(angle), sin(angle)
        if dim == 2:
            return cls(
                2,
                lambda p: np.stack((c * p[0] + s * p[1], -s * p[0] + c * p[1])),
                forward_affine=(((c, -s), (s, c)), (0.0, 0.0)),
            )
        if axis == "x":
            return cls(
                3,
                lambda p: np.stack((p[0], c * p[1] + s * p[2], -s * p[1] + c * p[2])),
                forward_affine=(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)), (0.0, 0.0, 0.0)),
            )
        if axis == "y":
            return cls(
                3,
                lambda p: np.stack((c * p[0] - s * p[2], p[1], s * p[0] + c * p[2])),
                forward_affine=(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), (0.0, 0.0, 0.0)),
            )
        return cls(
            3,
            lambda p: np.stack((c * p[0] + s * p[1], -s * p[0] + c * p[1], p[2])),
            forward_affine=(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0)),
        )

    def then(self, other: Transform) -> Transform:
        """Apply this transform, then `other`, in world space."""
        if not isinstance(other, Transform) or self.dim != other.dim:
            raise ValueError("composed transforms must have the same dimension")
        affine = None
        if self.forward_affine is not None and other.forward_affine is not None:
            first_matrix, first_offset = self.forward_affine
            second_matrix, second_offset = other.forward_affine
            matrix = tuple(
                tuple(
                    sum(second_matrix[i][k] * first_matrix[k][j] for k in range(self.dim))
                    for j in range(self.dim)
                )
                for i in range(self.dim)
            )
            offset = tuple(
                sum(second_matrix[i][k] * first_offset[k] for k in range(self.dim))
                + second_offset[i]
                for i in range(self.dim)
            )
            affine = (matrix, offset)
        return Transform(
            self.dim,
            lambda point: self.inverse(other.inverse(point)),
            self.distance_scale * other.distance_scale,
            forward_affine=affine,
        )

    def map_points(self, points):
        """Map one point or any batch with a final coordinate axis."""
        points = np.asarray(points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != self.dim:
            raise ValueError(f"points must end in {self.dim} coordinates")
        if self.forward_affine is not None:
            matrix, offset = self.forward_affine
            inverse = np.linalg.inv(np.asarray(matrix, dtype=float))
            return (points - np.asarray(offset, dtype=float)) @ inverse.T
        if points.ndim == 1:
            return np.asarray(self._single(points), dtype=float)
        return np.asarray(
            [self._single(point) for point in points.reshape(-1, self.dim)], dtype=float
        ).reshape(points.shape)

    def map_python(self, *point: float) -> tuple[float, ...]:
        point = np.asarray(_vector_args(point), dtype=float)
        if point.shape != (self.dim,):
            raise ValueError(f"point must have {self.dim} coordinates")
        result = self.map_points(point)
        values = tuple(float(value) for value in result)
        if not all(isfinite(value) for value in values):
            raise ValueError("inverse map returned nonfinite coordinates")
        return values

    def __call__(self, primitive):
        return primitive.transform(self)
