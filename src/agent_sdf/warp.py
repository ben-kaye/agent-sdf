"""Nonlinear coordinate maps with declared field guarantees."""

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from .transform import _finite, _vector_args


@dataclass(frozen=True)
class Warp:
    """Map world points to local points, evaluating f(inverse(p)) / lipschitz.

    ``inverse`` accepts one coordinate vector. ``lipschitz`` must bound its
    global Euclidean Lipschitz constant; the resulting field inherits the
    child's Lipschitz bound, but need not be an exact signed distance.
    GLSL requires the same callback to accept symbolic coordinates.
    """

    dim: int
    inverse: Callable
    lipschitz: float = field(kw_only=True)

    def __post_init__(self):
        if not isinstance(self.dim, int) or self.dim not in (2, 3):
            raise ValueError("warp dimension must be 2 or 3")
        if not callable(self.inverse):
            raise ValueError("warp inverse must be callable")
        bound = _finite(self.lipschitz)
        if bound <= 0 or not isfinite(1 / bound):
            raise ValueError("warp Lipschitz bound must be positive with a finite reciprocal")
        object.__setattr__(self, "lipschitz", bound)
        self.map_points(np.zeros(self.dim))

    @property
    def distance_scale(self):
        return 1 / self.lipschitz

    def map_points(self, points):
        """Apply the inverse map to points with a final coordinate axis."""
        points = np.asarray(points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != self.dim:
            raise ValueError(f"points must end in {self.dim} coordinates")
        if not points.size:
            return np.empty_like(points)
        try:
            result = np.asarray(
                [self.inverse(point) for point in points.reshape(-1, self.dim)], dtype=float
            )
        except Exception as exc:
            raise ValueError("warp inverse must return finite coordinate vectors") from exc
        if result.shape != (points.size // self.dim, self.dim) or not np.all(np.isfinite(result)):
            raise ValueError(f"warp inverse must return {self.dim} finite coordinates")
        return result.reshape(points.shape)

    def map_python(self, *point):
        point = np.asarray(_vector_args(point), dtype=float)
        if point.shape != (self.dim,):
            raise ValueError(f"point must have {self.dim} coordinates")
        return tuple(float(value) for value in self.map_points(point))

    def __call__(self, field, *, bounds):
        return field.warp(self, bounds=bounds)
