"""Piecewise curves and derived profile geometry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin, sqrt

from .field import Field, _number, _point, _positive
from .nodes import CurveProfile, Tube
from .transform import _vector_args


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
        raise ValueError(f"unknown curve segment kind: {self.kind}")

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
        raise ValueError(f"unknown curve segment kind: {self.kind}")


def _cross(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


class Curve:
    """2D or 3D piecewise line, quadratic Bézier, and circular arc curve."""

    def __init__(self, *start: float):
        start = _vector_args(start)
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
        point = _point(_vector_args(end), self.dim)
        if point != self._end:
            self.segments.append(_CurveSegment("line", (self._end, point)))
            self._end = point
        return self

    def quad_to(self, control: Iterable[float], end: Iterable[float], steps: int = 12) -> Curve:
        control, end = _point(control, self.dim), _point(end, self.dim)
        self.segments.append(_CurveSegment("quad", (self._end, control, end), self._steps(steps)))
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

    def close(self) -> Field:
        if self.dim != 2:
            raise ValueError("only 2D curves can be closed into a profile")
        if not self.segments:
            raise ValueError("a profile needs at least one curve segment")
        self.line_to(*self.start)
        return Field("curve_profile", 2, CurveProfile(tuple(self.segments)))

    def tube(self, radius: float, tolerance: float | None = None, *, closed: bool = False) -> Field:
        if self.dim != 3 or not self.segments:
            raise ValueError("a tube needs a 3D curve with at least one segment")
        radius = _positive(radius)
        points = self.sample(min(0.01, radius / 8) if tolerance is None else tolerance)
        if closed and points[-1] != points[0]:
            points += (points[0],)
        if len(points) < 2:
            raise ValueError("tube curve has no length")
        return Field("tube", 3, Tube(points, radius))

    def sweep(
        self,
        profile: Field,
        *,
        tolerance=0.01,
        up=None,
        closed=False,
        orientation="perpendicular",
    ) -> Field:
        """Sweep a 2D profile; see Field.sweep."""
        from .sweep import sweep

        return sweep(
            profile, self, tolerance=tolerance, up=up, closed=closed, orientation=orientation
        )
