"""NumPy evaluation for signed distance geometry."""

from math import atan2, pi, sqrt

import numpy as np

from .hit import Hit


def _length(value):
    return np.linalg.norm(value, axis=-1)


def _round_box(p, half_size, radius):
    q = np.abs(p) - np.asarray(half_size) + radius
    return _length(np.maximum(q, 0)) + np.minimum(np.max(q, axis=-1), 0) - radius


def _segment(p, a, b):
    a, b = np.asarray(a), np.asarray(b)
    edge = b - a
    t = np.clip(np.sum((p - a) * edge, axis=-1) / np.sum(edge * edge), 0, 1)
    return _length(p - a - t[..., None] * edge)


def _curve_xy(segment, t):
    points = segment.points
    if segment.kind == "arc":
        center, start, _ = points
        angle = atan2(start[1] - center[1], start[0] - center[0]) + t * segment.sweep
        radius = sqrt(sum((start[i] - center[i]) ** 2 for i in range(2)))
        return center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)
    if segment.kind == "line":
        a, b = points
        return tuple((1 - t) * a[i] + t * b[i] for i in range(2))
    if segment.kind == "quad":
        a, b, c = points
        return tuple((1 - t) ** 2 * a[i] + 2 * (1 - t) * t * b[i] + t**2 * c[i] for i in range(2))
    raise ValueError(f"unknown curve segment kind: {segment.kind}")


def _curve_y_breaks(segment):
    """Parameter intervals on which the curve's Y coordinate is monotone."""
    points = segment.points
    roots = []
    if segment.kind == "quad":
        a, b, c = (point[1] for point in points)
        denominator = a - 2 * b + c
        if denominator:
            roots.append((a - b) / denominator)
    elif segment.kind == "arc":
        center, start, _ = points
        start_angle = atan2(start[1] - center[1], start[0] - center[0])
        roots.extend((pi / 2 + k * pi - start_angle) / segment.sweep for k in range(-4, 5))
    return (0.0, *sorted(t for t in roots if 0 < t < 1), 1.0)


def _quad_distance(p, points):
    a, control, end = (np.asarray(point, dtype=float) for point in points)
    b = 2 * (control - a)
    c = a - 2 * control + end
    curvature2 = np.dot(c, c)
    if curvature2 <= 1e-10 * np.dot(b, b):
        if not np.any(b + c):
            return _length(p - a)
        return _segment(p, a, end)

    offset = a - p
    cubic = 2 * curvature2
    quadratic = 3 * np.dot(b, c)
    linear = np.dot(b, b) + 2 * np.sum(offset * c, axis=-1)
    constant = np.sum(offset * b, axis=-1)
    shift = quadratic / (3 * cubic)
    depressed = linear / cubic - quadratic * quadratic / (3 * cubic * cubic)
    free = (
        2 * quadratic**3 / (27 * cubic**3)
        - quadratic * linear / (3 * cubic * cubic)
        + constant / cubic
    )
    discriminant = (free / 2) ** 2 + (depressed / 3) ** 3
    best = np.minimum(np.sum((p - a) ** 2, axis=-1), np.sum((p - end) ** 2, axis=-1))

    def consider(root):
        nonlocal best
        t = np.clip(root, 0, 1)
        delta = a + t[..., None] * (b + t[..., None] * c) - p
        squared = np.sum(delta * delta, axis=-1)
        best = np.where((root >= 0) & (root <= 1), np.minimum(best, squared), best)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        radical = np.sqrt(np.maximum(discriminant, 0))
        one_root = np.cbrt(-free / 2 + radical) + np.cbrt(-free / 2 - radical) - shift
        consider(np.where(discriminant > 0, one_root, np.nan))

        scale = 2 * np.sqrt(np.maximum(-depressed / 3, 0))
        ratio = np.clip(-free / np.maximum(scale**3 / 4, np.finfo(float).tiny), -1, 1)
        angle = np.arccos(ratio) / 3
        for k in range(3):
            root = scale * np.cos(angle - 2 * pi * k / 3) - shift
            consider(np.where(discriminant <= 0, root, np.nan))
    return np.sqrt(best)


def _quad_x_at_y(py, points, lo, hi):
    a, control, end = points
    bx = 2 * (control[0] - a[0])
    cx = a[0] - 2 * control[0] + end[0]
    by = 2 * (control[1] - a[1])
    cy = a[1] - 2 * control[1] + end[1]
    if cy == 0:
        t = (py - a[1]) / by
    else:
        discriminant = np.maximum(by * by - 4 * cy * (a[1] - py), 0)
        sign = 1 if by + 2 * cy * (lo + hi) / 2 > 0 else -1
        numerator = -by + sign * np.sqrt(discriminant)
        alternate = -by - sign * np.sqrt(discriminant)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(
                np.abs(numerator) >= np.abs(alternate),
                numerator / (2 * cy),
                2 * (a[1] - py) / alternate,
            )
    return a[0] + t * (bx + t * cx)


def _curve_distance(p, segment):
    if segment.kind == "line":
        return _segment(p, *segment.points)
    if segment.kind == "quad":
        return _quad_distance(p, segment.points)
    if segment.kind == "arc":
        center, start, end = segment.points
        dx, dy = p[..., 0] - center[0], p[..., 1] - center[1]
        angle = np.arctan2(dy, dx)
        start_angle = atan2(start[1] - center[1], start[0] - center[0])
        progress = np.mod((angle - start_angle) * (1 if segment.sweep > 0 else -1), 2 * pi)
        radius = sqrt(sum((start[i] - center[i]) ** 2 for i in range(2)))
        radial = np.abs(np.sqrt(dx * dx + dy * dy) - radius)
        ends = np.minimum(_length(p - np.asarray(start)), _length(p - np.asarray(end)))
        return np.where(progress <= abs(segment.sweep), radial, ends)

    raise ValueError(f"unknown curve segment kind: {segment.kind}")


def _curve_profile(p, segments):
    distance = np.full(p.shape[:-1], np.inf)
    inside = np.zeros(p.shape[:-1], dtype=bool)
    px, py = p[..., 0], p[..., 1]
    for segment in segments:
        distance = np.minimum(distance, _curve_distance(p, segment))
        if segment.kind == "line":
            a, b = segment.points
            if a[1] != b[1]:
                crossing = (a[1] > py) != (b[1] > py)
                x = a[0] + (py - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                inside = np.logical_xor(inside, crossing & (px < x))
            continue
        breaks = _curve_y_breaks(segment)
        for lo, hi in zip(breaks, breaks[1:], strict=False):
            y0 = segment.point(lo)[1]
            y1 = segment.point(hi)[1]
            if abs(y1 - y0) < 1e-14:
                continue
            crossing = (y0 > py) != (y1 > py)
            if segment.kind == "quad":
                x = _quad_x_at_y(np.where(crossing, py, (y0 + y1) / 2), segment.points, lo, hi)
                inside = np.logical_xor(inside, crossing & (px < x))
                continue
            parameter_dtype = np.result_type(py, np.float32)
            lower = np.full(py.shape, lo, dtype=parameter_dtype)
            upper = np.full(py.shape, hi, dtype=parameter_dtype)

            def bisect(_, interval, curve_segment=segment, increasing=y1 > y0):
                a, b = interval
                mid = (a + b) / 2
                _, y = _curve_xy(curve_segment, mid)
                move_lower = (y < py) if increasing else (y > py)
                return np.where(move_lower, mid, a), np.where(move_lower, b, mid)

            for _ in range(28):
                lower, upper = bisect(None, (lower, upper))
            x, _ = _curve_xy(segment, (lower + upper) / 2)
            inside = np.logical_xor(inside, crossing & (px < x))
    return np.where(inside, -distance, distance)


def evaluate_shape(shape, points):
    """Return distance and material arrays for points with a final coordinate axis."""
    p = np.asarray(points)
    if p.shape[-1] != shape.dim:
        raise ValueError(f"expected points with {shape.dim} coordinates")

    def evaluate(node, q):
        kind, data = node.kind, node.data
        child = node.children[0] if node.children else None
        zero = np.zeros(q.shape[:-1], dtype=np.int32)
        if kind in ("circle", "sphere"):
            return _length(q) - data[0], zero
        if kind == "rectangle":
            w, h, radius = data
            return _round_box(q, (w / 2, h / 2), radius), zero
        if kind == "box":
            w, depth, h, radius = data
            return _round_box(q, (w / 2, depth / 2, h / 2), radius), zero
        if kind == "polygon":
            vertices = data[0]
            distance = np.full(q.shape[:-1], np.inf)
            inside = np.zeros(q.shape[:-1], dtype=bool)
            for a, b in zip(vertices, vertices[1:] + vertices[:1], strict=False):
                distance = np.minimum(distance, _segment(q, a, b))
                ax, ay = a
                bx, by = b
                crossing = (ay > q[..., 1]) != (by > q[..., 1])
                x_edge = ax + (q[..., 1] - ay) * (bx - ax) / (by - ay if by != ay else 1)
                inside = np.logical_xor(inside, crossing & (q[..., 0] < x_edge))
            return np.where(inside, -distance, distance), zero
        if kind == "curve_profile":
            return _curve_profile(q, data[0]), zero
        if kind == "cylinder":
            radius, height = data
            v = np.stack((_length(q[..., :2]) - radius, np.abs(q[..., 2]) - height / 2), axis=-1)
            return _length(np.maximum(v, 0)) + np.minimum(np.max(v, axis=-1), 0), zero
        if kind == "torus":
            major, tube = data
            v = np.stack((_length(q[..., :2]) - major, q[..., 2]), axis=-1)
            return _length(v) - tube, zero
        if kind in ("capsule", "tube"):
            segments = (
                [(data[0], data[1])]
                if kind == "capsule"
                else list(zip(data[0], data[0][1:], strict=False))
            )
            radius = data[2] if kind == "capsule" else data[1]
            distance = np.full(q.shape[:-1], np.inf)
            for a, b in segments:
                distance = np.minimum(distance, _segment(q, a, b))
            return distance - radius, zero
        if kind in ("custom", "python"):
            fn, material = data[:2]
            value = fn(*[q[..., axis] for axis in range(node.dim)])
            distance, material = (
                (value.distance, value.material) if isinstance(value, Hit) else (value, material)
            )
            return (
                np.broadcast_to(np.asarray(distance), q.shape[:-1]),
                np.broadcast_to(np.asarray(material, dtype=np.int32), q.shape[:-1]),
            )
        if kind in ("min", "max", "subtract", "smooth_union"):
            da, ma = evaluate(node.children[0], q)
            db, mb = evaluate(node.children[1], q)
            if kind == "min":
                return np.minimum(da, db), np.where(da < db, ma, mb)
            if kind == "max":
                return np.maximum(da, db), np.where(da > db, ma, mb)
            if kind == "subtract":
                return np.maximum(da, -db), ma
            radius = data[0]
            h = np.maximum(radius - np.abs(da - db), 0) / radius
            return np.minimum(da, db) - h * h * radius * 0.25, np.where(da < db, ma, mb)
        if kind == "translate":
            return evaluate(child, q - np.asarray(data))
        if kind == "scale":
            distance, material = evaluate(child, q / data[0])
            return distance * data[0], material
        if kind == "transform":
            transform = data[0]
            mapped = transform.map_points(q)
            distance, material = evaluate(child, mapped)
            return distance * transform.distance_scale, material
        if kind == "rotate":
            angle, axis = data
            c, s = np.cos(angle), np.sin(angle)
            x, y = q[..., 0], q[..., 1]
            if node.dim == 2:
                mapped = np.stack((c * x + s * y, -s * x + c * y), axis=-1)
            else:
                z = q[..., 2]
                if axis == "x":
                    mapped = np.stack((x, c * y + s * z, -s * y + c * z), axis=-1)
                elif axis == "y":
                    mapped = np.stack((c * x - s * z, y, s * x + c * z), axis=-1)
                else:
                    mapped = np.stack((c * x + s * y, -s * x + c * y, z), axis=-1)
            return evaluate(child, mapped)
        if kind == "offset":
            distance, material = evaluate(child, q)
            return distance - data[0], material
        if kind == "shell":
            distance, material = evaluate(child, q)
            return np.abs(distance) - data[0] / 2, material
        if kind == "material":
            distance, _ = evaluate(child, q)
            return distance, zero + data[0]
        if kind in ("loft", "section_loft", "variable_extrude"):
            from .operations import evaluate_operation

            return evaluate_operation(node, q, evaluate)
        if kind == "profile_sweep":
            from .sweep import evaluate_sweep

            return evaluate_sweep(node, q, evaluate)
        if kind == "extrude":
            distance, material = evaluate(child, q[..., :2])
            offset = data[1] if len(data) > 1 else 0.0
            v = np.stack((distance, np.abs(q[..., 2] - offset) - data[0] / 2), axis=-1)
            return _length(np.maximum(v, 0)) + np.minimum(np.max(v, axis=-1), 0), material
        if kind == "revolve":
            profile = np.stack((_length(q[..., :2]), q[..., 2]), axis=-1)
            return evaluate(child, profile)
        raise ValueError(f"unknown shape kind: {kind}")

    return evaluate(shape, p)
