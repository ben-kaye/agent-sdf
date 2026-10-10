"""Surface queries for signed distance geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin, sqrt, tan

import numpy as np

from .camera import _orientation, _orientation_up
from .field import Field
from .model import compile_scene


@dataclass(frozen=True)
class SurfaceHit:
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    distance: float
    material_id: int
    material_name: str | None
    feature_name: str | None
    instance_path: tuple[str, ...] = ()
    contributors: tuple[str, ...] = ()


def _vector(value, label):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain three finite coordinates")
    return vector


def _distance(field, point):
    return float(field.sample(point)[0])


def _sample(field, point):
    distance, material = field.sample(point)
    return float(distance), int(material)


def _rotate_back(point, angle, axis, dim):
    c, s = cos(angle), sin(angle)
    x, y = point[:2]
    if dim == 2:
        return np.array((c * x + s * y, -s * x + c * y))
    z = point[2]
    if axis == "x":
        return np.array((x, c * y + s * z, -s * y + c * z))
    if axis == "y":
        return np.array((c * x - s * z, y, s * x + c * z))
    return np.array((c * x + s * y, -s * x + c * y, z))


def _contributors(field, point):
    """Return primary and optional blend contributor at a world-space point."""
    kind, data = field.kind, field.data
    children = field.children
    if kind == "feature":
        inner = _contributors(children[0], point)
        return tuple(f"{data.name}/{name}" for name in inner) if inner else (data.name,)
    if kind in ("min", "max", "subtract", "smooth_union"):
        a, b = children
        da, db = _distance(a, point), _distance(b, point)
        if kind == "min":
            return _contributors(a if da < db else b, point)
        if kind == "max":
            return _contributors(a if da > db else b, point)
        if kind == "subtract":
            return _contributors(a if da > -db else b, point)
        primary = a if da < db else b
        secondary = b if da < db else a
        names = _contributors(primary, point)
        if abs(da - db) < data.radius:
            names += _contributors(secondary, point)
        return tuple(dict.fromkeys(names))
    if kind == "linear_pattern":
        child = children[0]
        step = np.asarray(data.step)
        local = point
        best = _distance(child, local)
        for index in range(1, data.count):
            candidate = point - index * step
            distance = _distance(child, candidate)
            if distance <= best:
                best, local = distance, candidate
        return _contributors(child, local)
    if kind == "radial_pattern":
        child = children[0]
        local = point
        best = _distance(child, local)
        for index in range(1, data.count):
            candidate = _rotate_back(point, 2 * np.pi * index / data.count, data.axis, 3)
            distance = _distance(child, candidate)
            if distance <= best:
                best, local = distance, candidate
        return _contributors(child, local)
    if not children:
        return ()
    child = children[0]
    if kind == "transform":
        point = data.transform.map_points(point)
    elif kind == "slice":
        plane = data.plane
        point = (
            np.asarray(plane.origin)
            + point[0] * np.asarray(plane.x_direction)
            + point[1] * np.asarray(plane.y_direction)
        )
    elif kind == "warp":
        point = data.warp.map_points(point)
    elif kind in ("extrude", "variable_extrude"):
        point = point[:2]
    elif kind == "revolve":
        x, y, z = point
        r = np.hypot(x, y)
        phi = np.arctan2(y, x) % (2 * np.pi)
        c, s = np.cos(data.angle), np.sin(data.angle)
        candidates = [
            (_distance(child, np.array((u, z))), np.array((u, z)))
            for u, swept in (
                (r, phi <= data.angle),
                (-r, (phi + np.pi) % (2 * np.pi) <= data.angle),
            )
            if swept or data.angle >= 2 * np.pi
        ]
        if data.angle < 2 * np.pi:
            for u, w in ((x, y), (c * x + s * y, c * y - s * x)):
                distance = _distance(child, np.array((u, z)))
                candidates.append((np.hypot(max(distance, 0.0), w), np.array((u, z))))
        point = min(candidates, key=lambda item: item[0])[1]
    elif kind == "loft":
        stations, (axis, transverse) = data.stations, data.axes
        table = np.asarray(stations)
        section = np.array([np.interp(point[axis], table[:, 0], table[:, i]) for i in range(1, 5)])
        point = (point[list(transverse)] - section[:2]) / section[2:]
    elif kind == "field_morph":
        stations, (axis, transverse) = data.stations, data.axes
        index = int(
            np.clip(np.searchsorted(stations, point[axis], side="right") - 1, 0, len(stations) - 2)
        )
        weight = np.clip(
            (point[axis] - stations[index]) / (stations[index + 1] - stations[index]), 0, 1
        )
        return _contributors(children[index + (weight >= 0.5)], point[list(transverse)])
    elif kind == "profile_sweep":
        # A named sweep is normally wrapped at the 3D level; without one,
        # its profile name is shared across every swept span.
        return _contributors(child, np.zeros(2))
    return _contributors(child, point)


def _clip_ray(field, origin, direction, max_distance, tolerance):
    near, far = 0.0, max_distance
    try:
        bounds = field.bounds()
    except ValueError:
        return near, far
    if bounds.is_empty:
        return 1.0, 0.0
    for low, high, o, d in zip(bounds.minimum, bounds.maximum, origin, direction, strict=True):
        if not isfinite(low) or not isfinite(high):
            continue
        pad = max(tolerance * 4, 1e-4 * (high - low))
        low -= pad
        high += pad
        if abs(d) < 1e-12:
            if o < low or o > high:
                return 1.0, 0.0
        else:
            a, b = (low - o) / d, (high - o) / d
            near, far = max(near, min(a, b)), min(far, max(a, b))
    return near, far


def _cast_field(field, origin, direction, max_distance, tolerance):
    """Cast one independent solid so another solid cannot hide its surface."""
    near, far = _clip_ray(field, origin, direction, max_distance, tolerance)
    if near > far:
        return None
    travel = near
    distance = _distance(field, origin + travel * direction)
    if not isfinite(distance):
        return None
    hit_t = None
    for _ in range(1024):
        if abs(distance) <= tolerance:
            hit_t = travel
            break
        next_t = min(travel + max(abs(distance) * 0.75, tolerance), far)
        if next_t <= travel:
            break
        next_distance = _distance(field, origin + next_t * direction)
        if not isfinite(next_distance):
            break
        if (distance < 0) != (next_distance < 0):
            low, high = travel, next_t
            low_d = distance
            for _ in range(32):
                middle = (low + high) * 0.5
                middle_d = _distance(field, origin + middle * direction)
                if (low_d < 0) == (middle_d < 0):
                    low, low_d = middle, middle_d
                else:
                    high = middle
            hit_t = (low + high) * 0.5
            break
        travel, distance = next_t, next_distance
    if hit_t is None:
        return None
    point = origin + hit_t * direction
    epsilon = max(tolerance * 2, 1e-5)
    gradient = np.array(
        [
            _distance(field, point + np.eye(3)[axis] * epsilon)
            - _distance(field, point - np.eye(3)[axis] * epsilon)
            for axis in range(3)
        ]
    )
    normal_length = np.linalg.norm(gradient)
    normal = gradient / normal_length if normal_length > 1e-12 else -direction
    return (
        tuple(float(v) for v in point),
        tuple(float(v) for v in normal),
        float(hit_t),
        _sample(field, point)[1],
        _contributors(field, point),
    )


def _local_feature(name, path):
    if not path:
        return name
    prefix = "/".join(path)
    if name == prefix:
        return None
    if name.startswith(prefix + "/"):
        return name[len(prefix) + 1 :]
    return name


def ray_cast(field_or_scene, origin, direction, *, max_distance=100.0, tolerance=1e-4):
    """Return the first surface crossing and its instance identity."""
    compiled = compile_scene(field_or_scene)
    if not isinstance(compiled.field, Field) or compiled.field.dim != 3:
        raise ValueError("ray_cast needs a 3D Field, Part, Assembly, or Scene")
    origin, direction = _vector(origin, "origin"), _vector(direction, "direction")
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("ray direction must be nonzero")
    direction /= length
    max_distance, tolerance = float(max_distance), float(tolerance)
    if not isfinite(max_distance) or max_distance <= 0 or not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("max_distance and tolerance must be finite and positive")
    material_names = {value["id"]: name for name, value in compiled.materials.items()}
    nearest = None
    for part in compiled.parts:
        hit = _cast_field(part.field, origin, direction, max_distance, tolerance)
        if hit is None:
            continue
        point, normal, distance, material_id, names = hit
        if nearest is not None and distance > nearest.distance:
            continue
        local_names = tuple(
            name for value in names if (name := _local_feature(value, part.path)) is not None
        )
        material_name = material_names.get(material_id)
        if material_name is not None and part.part is not None:
            material_name = material_name.rsplit("/", 1)[-1]
        nearest = SurfaceHit(
            point,
            normal,
            distance,
            material_id,
            material_name,
            local_names[0] if local_names else None,
            part.path,
            tuple(dict.fromkeys(local_names)),
        )
    return nearest


def pick(
    field_or_scene,
    x,
    y,
    width,
    height,
    *,
    eye,
    target,
    up=None,
    fov=40.0,
    camera_size=2.0,
    max_distance=100.0,
    tolerance=1e-4,
    orientation=None,
):
    """Pick a top-left-origin pixel; orientation controls camera roll around the view direction."""
    eye, target = _vector(eye, "eye"), _vector(target, "target")
    if up is not None:
        up = _vector(up, "up")
    if orientation is not None:
        orientation_up = np.asarray(_orientation_up(_orientation(orientation)))
        if up is not None and np.dot(up, orientation_up) <= 0:
            raise ValueError("up and orientation must point to the same camera hemisphere")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
        or not isfinite(x)
        or not isfinite(y)
        or not 0 <= x < width
        or not 0 <= y < height
    ):
        raise ValueError("pixel must lie inside a positive image size")
    if not isfinite(fov) or (fov != 0 and not 1 <= fov <= 120):
        raise ValueError("fov must be zero or between 1 and 120 degrees")
    if not isfinite(camera_size) or camera_size <= 0:
        raise ValueError("camera_size must be finite and positive")
    forward = target - eye
    forward_length = np.linalg.norm(forward)
    if forward_length == 0:
        raise ValueError("eye and target must differ")
    forward /= forward_length
    if up is not None and np.linalg.norm(np.cross(forward, up)) < 1e-12:
        raise ValueError("up must not be parallel to the view direction")
    if orientation is not None:
        camera_up_hint = orientation_up
    elif up is not None:
        camera_up_hint = up
    else:
        back = -forward
        horizontal = np.linalg.norm(back[:2])
        if horizontal > 1e-8:
            right = np.array((-back[1] / horizontal, back[0] / horizontal, 0.0))
        else:
            right = np.array((1.0, 0.0, 0.0))
        camera_up_hint = np.cross(right, forward)
    right = np.cross(forward, camera_up_hint)
    right_length = np.linalg.norm(right)
    if right_length < 1e-12:
        raise ValueError("camera up must not be parallel to the view direction")
    right /= right_length
    camera_up = np.cross(right, forward)
    spread = camera_size * 0.5 if fov == 0 else tan(radians(fov) * 0.5)
    uv = np.array((2 * (x + 0.5) / width - 1, 1 - 2 * (y + 0.5) / height)) * spread
    uv[0] *= width / height
    offset = uv[0] * right + uv[1] * camera_up
    origin = eye + offset if fov == 0 else eye
    direction = forward if fov == 0 else forward + offset
    return ray_cast(
        field_or_scene, origin, direction, max_distance=max_distance, tolerance=tolerance
    )


def distance_between(first, second):
    """Measure straight-line distance between two hits or 3D points."""
    a = first.point if isinstance(first, SurfaceHit) else first
    b = second.point if isinstance(second, SurfaceHit) else second
    return sqrt(
        sum((x - y) ** 2 for x, y in zip(_vector(a, "first"), _vector(b, "second"), strict=True))
    )
