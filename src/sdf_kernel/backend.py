"""JAX evaluation for signed distance geometry."""

from math import atan2, pi, sqrt

import jax
import jax.numpy as jnp

from .hit import Hit


def _length(value):
    return jnp.linalg.norm(value, axis=-1)


def _round_box(p, half_size, radius):
    q = jnp.abs(p) - jnp.asarray(half_size) + radius
    return _length(jnp.maximum(q, 0)) + jnp.minimum(jnp.max(q, axis=-1), 0) - radius


def _segment(p, a, b):
    a, b = jnp.asarray(a), jnp.asarray(b)
    edge = b - a
    t = jnp.clip(jnp.sum((p - a) * edge, axis=-1) / jnp.sum(edge * edge), 0, 1)
    return _length(p - a - t[..., None] * edge)


def _curve_xy(segment, t):
    points = segment.points
    if segment.kind == "arc":
        center, start, _ = points
        angle = atan2(start[1] - center[1], start[0] - center[0]) + t * segment.sweep
        radius = sqrt(sum((start[i] - center[i]) ** 2 for i in range(2)))
        return center[0] + radius * jnp.cos(angle), center[1] + radius * jnp.sin(angle)
    if segment.kind == "line":
        a, b = points
        return tuple((1 - t) * a[i] + t * b[i] for i in range(2))
    if segment.kind == "quad":
        a, b, c = points
        return tuple((1 - t) ** 2 * a[i] + 2 * (1 - t) * t * b[i] + t**2 * c[i] for i in range(2))
    a, b, c, d = points
    return tuple(
        (1 - t) ** 3 * a[i] + 3 * (1 - t) ** 2 * t * b[i] + 3 * (1 - t) * t**2 * c[i] + t**3 * d[i]
        for i in range(2)
    )


def _curve_y_breaks(segment):
    """Parameter intervals on which the curve's Y coordinate is monotone."""
    points = segment.points
    roots = []
    if segment.kind == "quad":
        a, b, c = (point[1] for point in points)
        denominator = a - 2 * b + c
        if denominator:
            roots.append((a - b) / denominator)
    elif segment.kind == "cubic":
        a, b, c, d = (point[1] for point in points)
        aa = 3 * (-a + 3 * b - 3 * c + d)
        bb = 6 * (a - 2 * b + c)
        cc = 3 * (b - a)
        if abs(aa) < 1e-14:
            if abs(bb) >= 1e-14:
                roots.append(-cc / bb)
        else:
            discriminant = bb * bb - 4 * aa * cc
            if discriminant >= 0:
                roots.extend(
                    ((-bb - sqrt(discriminant)) / (2 * aa), (-bb + sqrt(discriminant)) / (2 * aa))
                )
    elif segment.kind == "arc":
        center, start, _ = points
        start_angle = atan2(start[1] - center[1], start[0] - center[0])
        roots.extend((pi / 2 + k * pi - start_angle) / segment.sweep for k in range(-4, 5))
    return (0.0, *sorted(t for t in roots if 0 < t < 1), 1.0)


def _curve_distance(p, segment):
    if segment.kind == "line":
        return _segment(p, *segment.points)
    if segment.kind == "arc":
        center, start, end = segment.points
        dx, dy = p[..., 0] - center[0], p[..., 1] - center[1]
        angle = jnp.arctan2(dy, dx)
        start_angle = atan2(start[1] - center[1], start[0] - center[0])
        progress = jnp.mod((angle - start_angle) * (1 if segment.sweep > 0 else -1), 2 * pi)
        radius = sqrt(sum((start[i] - center[i]) ** 2 for i in range(2)))
        radial = jnp.abs(jnp.sqrt(dx * dx + dy * dy) - radius)
        ends = jnp.minimum(_length(p - jnp.asarray(start)), _length(p - jnp.asarray(end)))
        return jnp.where(progress <= abs(segment.sweep), radial, ends)

    # Cubic nearest-point equations are quintic. Minimize the original curve
    # numerically instead of replacing it with a polygon.
    count = 16
    grid = jnp.linspace(0, 1, count + 1).reshape((-1,) + (1,) * (p.ndim - 1))
    x, y = _curve_xy(segment, grid)
    coarse = (x - p[..., 0]) ** 2 + (y - p[..., 1]) ** 2
    index = jnp.argmin(coarse, axis=0)
    left = jnp.maximum(index - 1, 0) / count
    right = jnp.minimum(index + 1, count) / count
    ratio = (sqrt(5) - 1) / 2

    def squared(t):
        x, y = _curve_xy(segment, t)
        return (x - p[..., 0]) ** 2 + (y - p[..., 1]) ** 2

    def narrow(_, interval):
        lo, hi = interval
        a = hi - ratio * (hi - lo)
        b = lo + ratio * (hi - lo)
        choose_left = squared(a) < squared(b)
        return jnp.where(choose_left, lo, a), jnp.where(choose_left, b, hi)

    left, right = jax.lax.fori_loop(0, 24, narrow, (left, right))
    refined = squared((left + right) / 2)
    return jnp.sqrt(jnp.minimum(refined, jnp.min(coarse, axis=0)))


def _curve_profile(p, segments):
    distance = jnp.full(p.shape[:-1], jnp.inf)
    inside = jnp.zeros(p.shape[:-1], dtype=bool)
    px, py = p[..., 0], p[..., 1]
    for segment in segments:
        distance = jnp.minimum(distance, _curve_distance(p, segment))
        if segment.kind == "line":
            a, b = segment.points
            if a[1] != b[1]:
                crossing = (a[1] > py) != (b[1] > py)
                x = a[0] + (py - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                inside = jnp.logical_xor(inside, crossing & (px < x))
            continue
        breaks = _curve_y_breaks(segment)
        for lo, hi in zip(breaks, breaks[1:], strict=False):
            y0 = segment.point(lo)[1]
            y1 = segment.point(hi)[1]
            if abs(y1 - y0) < 1e-14:
                continue
            crossing = (y0 > py) != (y1 > py)
            parameter_dtype = jnp.result_type(py, jnp.float32)
            lower = jnp.full(py.shape, lo, dtype=parameter_dtype)
            upper = jnp.full(py.shape, hi, dtype=parameter_dtype)

            def bisect(_, interval, curve_segment=segment, increasing=y1 > y0):
                a, b = interval
                mid = (a + b) / 2
                _, y = _curve_xy(curve_segment, mid)
                move_lower = (y < py) if increasing else (y > py)
                return jnp.where(move_lower, mid, a), jnp.where(move_lower, b, mid)

            lower, upper = jax.lax.fori_loop(0, 28, bisect, (lower, upper))
            x, _ = _curve_xy(segment, (lower + upper) / 2)
            inside = jnp.logical_xor(inside, crossing & (px < x))
    return jnp.where(inside, -distance, distance)


def evaluate_shape(shape, points):
    """Return distance and material arrays for points with a final coordinate axis."""
    p = jnp.asarray(points)
    if p.shape[-1] != shape.dim:
        raise ValueError(f"expected points with {shape.dim} coordinates")

    def evaluate(node, q):
        kind, data = node.kind, node.data
        child = node.children[0] if node.children else None
        zero = jnp.zeros(q.shape[:-1], dtype=jnp.int32)
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
            distance = jnp.full(q.shape[:-1], jnp.inf)
            inside = jnp.zeros(q.shape[:-1], dtype=bool)
            for a, b in zip(vertices, vertices[1:] + vertices[:1], strict=False):
                distance = jnp.minimum(distance, _segment(q, a, b))
                ax, ay = a
                bx, by = b
                crossing = (ay > q[..., 1]) != (by > q[..., 1])
                x_edge = ax + (q[..., 1] - ay) * (bx - ax) / (by - ay if by != ay else 1)
                inside = jnp.logical_xor(inside, crossing & (q[..., 0] < x_edge))
            return jnp.where(inside, -distance, distance), zero
        if kind == "curve_profile":
            return _curve_profile(q, data[0]), zero
        if kind == "cylinder":
            radius, height = data
            v = jnp.stack((_length(q[..., :2]) - radius, jnp.abs(q[..., 2]) - height / 2), axis=-1)
            return _length(jnp.maximum(v, 0)) + jnp.minimum(jnp.max(v, axis=-1), 0), zero
        if kind == "torus":
            major, tube = data
            v = jnp.stack((_length(q[..., :2]) - major, q[..., 2]), axis=-1)
            return _length(v) - tube, zero
        if kind in ("capsule", "tube"):
            segments = (
                [(data[0], data[1])]
                if kind == "capsule"
                else list(zip(data[0], data[0][1:], strict=False))
            )
            radius = data[2] if kind == "capsule" else data[1]
            distance = jnp.full(q.shape[:-1], jnp.inf)
            for a, b in segments:
                distance = jnp.minimum(distance, _segment(q, a, b))
            return distance - radius, zero
        if kind in ("custom", "python"):
            fn, material = data[:2]
            value = fn(*[q[..., axis] for axis in range(node.dim)])
            distance, material = (
                (value.distance, value.material) if isinstance(value, Hit) else (value, material)
            )
            return (
                jnp.broadcast_to(jnp.asarray(distance), q.shape[:-1]),
                jnp.broadcast_to(jnp.asarray(material, dtype=jnp.int32), q.shape[:-1]),
            )
        if kind in ("union", "intersect", "subtract", "smooth_union"):
            da, ma = evaluate(node.children[0], q)
            db, mb = evaluate(node.children[1], q)
            if kind == "union":
                return jnp.minimum(da, db), jnp.where(da < db, ma, mb)
            if kind == "intersect":
                return jnp.maximum(da, db), jnp.where(da > db, ma, mb)
            if kind == "subtract":
                return jnp.maximum(da, -db), ma
            radius = data[0]
            h = jnp.maximum(radius - jnp.abs(da - db), 0) / radius
            return jnp.minimum(da, db) - h * h * radius * 0.25, jnp.where(da < db, ma, mb)
        if kind == "translate":
            return evaluate(child, q - jnp.asarray(data))
        if kind == "scale":
            distance, material = evaluate(child, q / data[0])
            return distance * data[0], material
        if kind == "transform":
            transform = data[0]
            mapped = transform.map_jax(q)
            distance, material = evaluate(child, mapped)
            return distance * transform.distance_scale, material
        if kind == "rotate":
            angle, axis = data
            c, s = jnp.cos(angle), jnp.sin(angle)
            x, y = q[..., 0], q[..., 1]
            if node.dim == 2:
                mapped = jnp.stack((c * x + s * y, -s * x + c * y), axis=-1)
            else:
                z = q[..., 2]
                if axis == "x":
                    mapped = jnp.stack((x, c * y + s * z, -s * y + c * z), axis=-1)
                elif axis == "y":
                    mapped = jnp.stack((c * x - s * z, y, s * x + c * z), axis=-1)
                else:
                    mapped = jnp.stack((c * x + s * y, -s * x + c * y, z), axis=-1)
            return evaluate(child, mapped)
        if kind == "offset":
            distance, material = evaluate(child, q)
            return distance - data[0], material
        if kind == "shell":
            distance, material = evaluate(child, q)
            return jnp.abs(distance) - data[0] / 2, material
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
            v = jnp.stack((distance, jnp.abs(q[..., 2] - offset) - data[0] / 2), axis=-1)
            return _length(jnp.maximum(v, 0)) + jnp.minimum(jnp.max(v, axis=-1), 0), material
        if kind == "revolve":
            profile = jnp.stack((_length(q[..., :2]), q[..., 2]), axis=-1)
            return evaluate(child, profile)
        raise ValueError(f"unknown shape kind: {kind}")

    return evaluate(shape, p)


def compile_jax(scene):
    """Compile a 3D scene into a JAX callable accepting (..., 3) points."""
    from .dsl import Model, Shape

    if isinstance(scene, Model):
        scene = scene.scene()
    if not isinstance(scene, Shape) or scene.dim != 3:
        raise ValueError("scene must be a 3D Shape or Model")
    return jax.jit(lambda points: scene.sample(points))
