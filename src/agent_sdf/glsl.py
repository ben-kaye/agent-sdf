"""Lower the immutable Field tree directly to GLSL ES 3.00."""

from __future__ import annotations

import re
from math import cos, hypot, isfinite, pi, sin

import numpy as np

from .model import compile_scene

_MAX_NODES = 4096
_MAX_SWEEP_SPANS = 128
_MAX_PROFILE_SEGMENTS = 2048


def _f(value):
    value = float(value)
    if not isfinite(value):
        raise ValueError("GLSL geometry requires finite constants")
    result = repr(value)
    return result if "." in result or "e" in result else result + ".0"


def _v(values):
    values = tuple(values)
    return f"vec{len(values)}(" + ", ".join(map(_f, values)) + ")"


def _bounds_distance(bounds):
    center = tuple(a / 2 + b / 2 for a, b in zip(bounds.minimum, bounds.maximum, strict=True))
    half = tuple(b / 2 - a / 2 for a, b in zip(bounds.minimum, bounds.maximum, strict=True))
    return f"sdBox{len(center)}({_minus('p', center)}, {_v(half)}, 0.0)"


def _clip(boxes, result):
    """Cap result.x below by the distance to boxes enclosing the zero set."""
    if boxes is None:
        return []
    clip = _bounds_distance(boxes[0])
    for box in boxes[1:]:
        clip = f"min({clip}, {_bounds_distance(box)})"
    return [f"{result}.x = max({result}.x, {clip});"]


def _group(expression):
    return expression if re.fullmatch(r"[\w.]+", expression) else f"({expression})"


def _minus(p, offset):
    return f"{p} - {_v(offset)}" if any(offset) else p


def _over(expression, scale):
    return expression if scale == 1.0 else f"{expression} / {_f(scale)}"


def _linear(matrix, q):
    """GLSL for matrix @ q, eliding zero terms, unit scales and permutations."""
    rows = [tuple(float(value) for value in row) for row in np.asarray(matrix, dtype=float)]
    size = max(abs(value) for row in rows for value in row)
    rows = [tuple(0.0 if abs(value) <= 1e-15 * size else value for value in row) for row in rows]
    inputs = len(rows[0])
    picks = [[j for j, value in enumerate(row) if value] for row in rows]
    if all(len(pick) == 1 for pick in picks):
        swizzle = "".join("xyz"[pick[0]] for pick in picks)
        scale = [row[pick[0]] for row, pick in zip(rows, picks, strict=True)]
        base = q if swizzle == "xyz"[:inputs] else f"{_group(q)}.{swizzle}"
        if all(value == 1.0 for value in scale):
            return base
        factor = _f(scale[0]) if len(set(scale)) == 1 else _v(scale)
        return f"{_group(base)} * {factor}"
    shape = f"mat{inputs}" if inputs == len(rows) else f"mat{inputs}x{len(rows)}"
    columns = ", ".join(_f(value) for column in zip(*rows, strict=True) for value in column)
    return f"{shape}({columns}) * {_group(q)}"


def _rotation_inverse(c, s, axis):
    a, b = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    matrix = np.eye(3)
    matrix[a, a] = matrix[b, b] = c
    matrix[a, b], matrix[b, a] = s, -s
    return matrix


class _Functions:
    """Emit each distinct GLSL function once, in dependency order."""

    def __init__(self):
        self.source = {}
        self.names = {}

    def __len__(self):
        return len(self.source)

    def add(self, prefix, result, params, body):
        key = (result, params, body)
        if key not in self.names:
            name = f"{prefix}{len(self.source)}"
            self.names[key] = name
            self.source[name] = f"{result} {name}({params}) {{\n{body}\n}}"
        return self.names[key]


def _reachable(functions, roots):
    """Keep generated functions referenced from the root source, in order."""
    words = re.compile(r"[A-Za-z_]\w*")
    keep, pending = set(), [roots]
    while pending:
        for word in words.findall(pending.pop()):
            if word in functions and word not in keep:
                keep.add(word)
                pending.append(functions[word])
    return [source for name, source in functions.items() if name in keep]


def _inverse_rotation(p, c, s, axis):
    if axis == "x":
        return f"vec3({p}.x, {c} * {p}.y + {s} * {p}.z, -{s} * {p}.y + {c} * {p}.z)"
    if axis == "y":
        return f"vec3({c} * {p}.x - {s} * {p}.z, {p}.y, {s} * {p}.x + {c} * {p}.z)"
    return f"vec3({c} * {p}.x + {s} * {p}.y, -{s} * {p}.x + {c} * {p}.y, {p}.z)"


def _box_map(box, matrix, translation):
    from itertools import product

    corners = product(*zip(box.minimum, box.maximum, strict=True))
    points = [
        tuple(
            sum(a * p for a, p in zip(row, corner, strict=True)) + t
            for row, t in zip(matrix, translation, strict=True)
        )
        for corner in corners
    ]
    from .bounds import AABB

    return AABB(
        tuple(min(p[i] for p in points) for i in range(len(translation))),
        tuple(max(p[i] for p in points) for i in range(len(translation))),
    )


def _field_magnitude(node, box, bound):
    center = tuple((a + b) / 2 for a, b in zip(box.minimum, box.maximum, strict=True))
    radius = hypot(*(b - a for a, b in zip(center, box.maximum, strict=True)))
    value = float(node.sample(np.asarray(center, dtype=float))[0])
    if not isfinite(value):
        raise ValueError("GLSL profile has a nonfinite field value")
    return abs(value) + bound * radius + 1e-5


def _lipschitz(node, box):
    """Certify a bound over a convex box containing every march point and surface."""
    from .bounds import AABB

    kind = node.kind
    data = node.data
    if kind == "linear_pattern":
        displacement = tuple((data.count - 1) * value for value in data.step)
        child_box = AABB(
            tuple(a - max(0.0, d) for a, d in zip(box.minimum, displacement, strict=True)),
            tuple(b - min(0.0, d) for b, d in zip(box.maximum, displacement, strict=True)),
        )
    elif kind == "radial_pattern":
        a, b = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[data.axis]
        radius = hypot(
            max(abs(box.minimum[a]), abs(box.maximum[a])),
            max(abs(box.minimum[b]), abs(box.maximum[b])),
        )
        low, high = list(box.minimum), list(box.maximum)
        low[a] = low[b] = -radius
        high[a] = high[b] = radius
        child_box = AABB(low, high)
    elif kind == "transform":
        transform = data.transform
        inverse = np.asarray(transform.inverse_matrix)
        offset = -inverse @ np.asarray(transform.offset)
        child_box = _box_map(box, inverse, offset)
    elif kind == "slice":
        plane = data.plane
        matrix = tuple(zip(plane.x_direction, plane.y_direction, strict=True))
        child_box = _box_map(box, matrix, plane.origin)
    elif kind == "warp":
        center = np.asarray(box.minimum) / 2 + np.asarray(box.maximum) / 2
        radius = np.linalg.norm(np.asarray(box.maximum) - center) * data.warp.lipschitz
        mapped = data.warp.map_points(center)
        child_box = AABB(tuple(mapped - radius), tuple(mapped + radius))
    elif kind in ("extrude", "variable_extrude", "profile_sweep"):
        if kind == "profile_sweep":
            # A sweep samples profiles at offsets from span origins; parallel
            # sweeps shear those offsets by up to the inverse span scale.
            from .sweep import _spans

            spans = _spans(data)
            distance = max(
                hypot(*(b - a for a, b in zip(span.start, q, strict=True)))
                for span in spans
                for q in (box.minimum, box.maximum)
            ) / min(span.scale for span in spans)
            child_box = AABB((-distance, -distance), (distance, distance))
        else:
            child_box = AABB(box.minimum[:2], box.maximum[:2])
    elif kind == "revolve":
        radius = max(
            hypot(x, y)
            for x in (box.minimum[0], box.maximum[0])
            for y in (box.minimum[1], box.maximum[1])
        )
        near, far = _revolve_sides(node.children[0])
        child_box = AABB(
            (-radius if far else 0.0, box.minimum[2]), (radius if near else 0.0, box.maximum[2])
        )
    elif kind in ("field_morph", "loft"):
        axis, transverse = data.axes
        child_box = AABB(
            tuple(box.minimum[i] for i in transverse),
            tuple(box.maximum[i] for i in transverse),
        )
    else:
        child_box = box
    if kind == "loft":
        stations = data.stations
        axial = data.axes[0]

        def section_at(a, b, position):
            t = (position - a[0]) / (b[0] - a[0])
            return (position, *(x + t * (y - x) for x, y in zip(a[1:], b[1:], strict=True)))

        active = []
        for a, b in zip(stations, stations[1:], strict=False):
            start = max(a[0], box.minimum[axial])
            end = min(b[0], box.maximum[axial])
            if end <= start:
                continue
            active.append((section_at(a, b, start), section_at(a, b, end)))
        if not active:
            active = [(stations[0], stations[1])]
        active_rows = tuple(row for pair in active for row in pair)
        minimum_scale = [min(row[3 + i] for row in active_rows) for i in range(2)]
        maximum_scale = min(max(row[3] for row in active_rows), max(row[4] for row in active_rows))
        offset_low = [min(row[1 + i] for row in active_rows) for i in range(2)]
        offset_high = [max(row[1 + i] for row in active_rows) for i in range(2)]
        displacement = [
            max(
                abs(child_box.minimum[i] - offset_high[i]),
                abs(child_box.maximum[i] - offset_low[i]),
            )
            for i in range(2)
        ]
        qmax = [displacement[i] / minimum_scale[i] for i in range(2)]
        profile_box = AABB(tuple(-x for x in qmax), tuple(qmax))
        profile = node.children[0]
        profile_bound = _lipschitz(profile, profile_box)
        magnitude = _field_magnitude(profile, profile_box, profile_bound)
        transverse_bound = maximum_scale * profile_bound / min(minimum_scale)
        axial_bound = 0.0
        for a, b in active:
            dz = b[0] - a[0]
            offset_slope = [abs(b[1 + i] - a[1 + i]) / dz for i in range(2)]
            scale_slope = [abs(b[3 + i] - a[3 + i]) / dz for i in range(2)]
            mapped_slope = hypot(
                *(
                    offset_slope[i] / minimum_scale[i]
                    + displacement[i] * scale_slope[i] / minimum_scale[i] ** 2
                    for i in range(2)
                )
            )
            axial_bound = max(
                axial_bound,
                max(scale_slope) * magnitude + maximum_scale * profile_bound * mapped_slope,
            )
        result = hypot(transverse_bound, axial_bound, 1.0)
        children = ()
    elif kind == "field_morph":
        profiles = node.children
        bounds = [_lipschitz(profile, child_box) for profile in profiles]
        magnitudes = [
            _field_magnitude(profile, child_box, bound)
            for profile, bound in zip(profiles, bounds, strict=True)
        ]
        axial = max(
            (a + b) / (z1 - z0)
            for a, b, z0, z1 in zip(
                magnitudes[:-1], magnitudes[1:], data.stations[:-1], data.stations[1:], strict=True
            )
        )
        result = hypot(max(bounds), axial, 1.0)
        children = ()
    else:
        children = tuple(_lipschitz(child, child_box) for child in node.children)
    if kind in (
        "circle",
        "sphere",
        "rectangle",
        "box",
        "cylinder",
        "torus",
        "capsule",
        "tube",
        "curve_profile",
        "polygon",
    ):
        result = 1.0
    elif kind == "custom":
        result = node.data.lipschitz
        if result is None:
            raise ValueError("GLSL custom fields need a declared Lipschitz bound")
        result = max(1.0, result)
    elif kind in ("min", "max", "subtract", "smooth_union"):
        # The smooth minimum's gradient is a convex combination of its inputs.
        result = max(children)
    elif kind in (
        "linear_pattern",
        "radial_pattern",
        "offset",
        "shell",
        "material",
        "feature",
        "slice",
    ):
        result = children[0]
    elif kind == "warp":
        result = max(1.0, children[0])
    elif kind == "transform":
        transform = node.data.transform
        inverse = np.asarray(transform.inverse_matrix)
        result = children[0] * transform.distance_scale * float(np.linalg.norm(inverse, 2))
    elif kind == "extrude":
        result = max(children[0], 1.0)
    elif kind == "profile_sweep":
        result = hypot(children[0], 1.0)
    elif kind == "revolve":
        result = children[0] if data.angle >= 2 * pi else max(children[0], 1.0)
    elif kind == "variable_extrude":
        height = data.height_lipschitz if callable(data.height) else 0.0
        offset = data.offset_lipschitz if callable(data.offset) else 0.0
        if height is None or offset is None:
            raise ValueError(
                "GLSL variable extrusion needs height_lipschitz and offset_lipschitz "
                "for callable fields"
            )
        result = hypot(children[0], hypot(1.0, offset + height * 0.5))
    elif kind in ("field_morph", "loft"):
        pass
    else:
        raise ValueError(f"GLSL lowering does not support field kind '{kind}'")
    if not isfinite(result) or result <= 0:
        raise ValueError("GLSL field Lipschitz bound must be finite and positive")
    return result


def _revolve_sides(profile):
    """Whether a revolve profile reaches the +u and -u sides of its axis."""
    from .bounds import _analytic

    box = _analytic(profile)
    if box.is_empty:
        return True, False
    return box.maximum[0] > 0 or box.minimum[0] >= 0, box.minimum[0] < 0


def _field_kinds(node):
    yield node.kind
    for child in node.children:
        yield from _field_kinds(child)


def _loft_scale_spans(first, last, ratio=2.0, limit=24):
    """Split a loft segment so each step box stays within a scale ratio.

    The march certificate divides by the smallest scale in the box. One box
    over a taper to a point stalls before the tip is reached.
    """

    def at(t):
        return tuple(a + t * (b - a) for a, b in zip(first, last, strict=True))

    def step(t):
        dt = 1.0 - t
        for start, end in zip(first[3:], last[3:], strict=True):
            origin = start + t * (end - start)
            slope = end - start
            if origin <= 0.0 or slope == 0.0:
                continue
            limit_scale = origin * ratio if slope > 0.0 else origin / ratio
            dt = min(dt, (limit_scale - origin) / slope)
        return dt

    spans = []
    t = 0.0
    for _ in range(limit):
        if t >= 1.0 - 1e-12:
            break
        dt = step(t)
        nxt = 1.0 if dt <= 1e-8 else min(1.0, t + dt)
        spans.append((at(t), at(nxt)))
        t = nxt
    if not spans:
        spans.append((first, last))
    elif t < 1.0 - 1e-12:
        spans.append((at(t), at(1.0)))
    return spans


def _section_span_box(node, first, last):
    from .bounds import AABB

    profile = node.children[0].bounds()
    if profile.is_empty or not all(isfinite(x) for x in profile.minimum + profile.maximum):
        raise ValueError("GLSL loft needs finite profile bounds for safe stepping")
    axis, transverse = node.data.axes
    low, high = [0.0] * 3, [0.0] * 3
    low[axis], high[axis] = first[0], last[0]
    for i, world_axis in enumerate(transverse):
        values = [
            station[1 + i] + station[3 + i] * edge
            for station in (first, last)
            for edge in (profile.minimum[i], profile.maximum[i])
        ]
        low[world_axis], high[world_axis] = min(values), max(values)
    return AABB(tuple(low), tuple(high))


def _section_step_boxes(node):
    """Return padded (box, pad) pairs for each scale-limited loft span."""
    result = []
    for first, last in zip(node.data.stations, node.data.stations[1:], strict=False):
        for span in _loft_scale_spans(first, last):
            box = _section_span_box(node, *span)
            pad = max(
                1e-4,
                1e-6 * hypot(*(b - a for a, b in zip(box.minimum, box.maximum, strict=True))),
            )
            result.append((box.expanded(pad), pad))
    return result


def _step_box(node):
    """Return the padded (box, pad) used for box-entry stepping, or None."""
    box = node.bounds()
    if box.is_empty or not all(isfinite(x) for x in box.minimum + box.maximum):
        return None
    pad = (
        0.0
        if node.kind == "variable_extrude"
        else max(
            1e-4, 1e-6 * hypot(*(b - a for a, b in zip(box.minimum, box.maximum, strict=True)))
        )
    )
    return box.expanded(pad), pad


_PRIMITIVES = {
    "circle",
    "sphere",
    "rectangle",
    "box",
    "cylinder",
    "torus",
    "capsule",
    "tube",
    "curve_profile",
    "polygon",
}


class _QueryLowerer:
    """Lower fields to vec2(distance, safe ray step) functions."""

    def __init__(self, raw_names, raw_steps):
        self.raw_names = raw_names
        self.raw_steps = raw_steps
        self.names = {}
        self.functions = _Functions()
        self.certificates = {}

    def _certificate(self, node):
        """Return (box, scale) with step and distance >= scale * box distance."""
        key = id(node)
        if key not in self.certificates:
            self.certificates[key] = self._step_certificate(node)
        return self.certificates[key]

    def _step_certificate(self, node):
        from .bounds import _affine
        from .bvh import _enclose, certificate

        if id(node) in self.raw_steps:
            return certificate(node)
        kind = node.kind
        if kind in {"min", "max", "subtract", "linear_pattern", "radial_pattern"} or kind in {
            "material",
            "feature",
        }:
            step = certificate(node, self._certificate)
        elif kind == "transform":
            child = self._certificate(node.children[0])
            if child is None:
                return None
            # Box entry and distance steps map back through the ray speed;
            # the condition number bounds both directions of the stretch.
            transform = node.data.transform
            condition = float(np.linalg.norm(np.asarray(transform.matrix, dtype=float), 2)) * float(
                np.linalg.norm(np.asarray(transform.inverse_matrix, dtype=float), 2)
            )
            box = _affine(child[0], transform.matrix, transform.offset)
            box = box.expanded(1e-5 * (1 + max(map(abs, box.minimum + box.maximum))) * condition)
            step = (box, child[1] / condition)
        elif kind == "loft":
            step = (_enclose([box for box, _ in _section_step_boxes(node)]), 1.0)
        elif kind in _PRIMITIVES:
            step = certificate(node)
        else:
            try:
                padded = _step_box(node)
            except ValueError:
                padded = None
            step = None if padded is None else (padded[0], 1.0)
        if step is None:
            return None
        value = certificate(node)
        if value is None:
            return None
        return step[0].union(value[0]), min(step[1], value[1])

    def _add(self, prefix, dim, body, result="vec2"):
        return self.functions.add(prefix, result, f"vec{dim} p, vec{dim} ray", body)

    def _union(self, node):
        """Skip union components whose certified query exceeds the current best."""
        from .bvh import _build
        from .glsl_pruning import _rounded_box

        components, pending = [], [node]
        while pending:
            current = pending.pop()
            if current.kind == "min":
                pending.extend(reversed(current.children))
            else:
                components.append(current)
        certificates = [self._certificate(component) for component in components]
        if sum(cert is not None for cert in certificates[1:]) < 4:
            return None
        names = [self.lower(component) for component in components]
        dim = node.dim

        def update(index):
            return (
                f"{{ vec2 h = {names[index]}(p, ray); "
                "best = vec2(min(best.x, h.x), min(best.y, h.y)); }"
            )

        entries = []
        for index, cert in enumerate(certificates):
            if index and cert is not None:
                box = _rounded_box(cert[0])
                center = tuple(a / 2 + b / 2 for a, b in zip(box.minimum, box.maximum, strict=True))
                entries.append((index, box, center))

        def emit(tree):
            emitted = [emit(child) for child in tree.children]
            scale = min([s for _, s in emitted] + [certificates[i][1] for i in tree.fields])
            rounded = float(np.nextafter(np.float32(scale), np.float32(0)))
            lines = [
                f"vec{dim} lo = {_v(tree.bounds.minimum)}, hi = {_v(tree.bounds.maximum)};",
                f"vec{dim} gap = max(max(lo - p, p - hi), 0.0);",
                "float bound = max(best.x, best.y);",
                f"if (any(greaterThan(gap, vec{dim}(0.0))) && length(gap) * {_f(rounded)} > bound + 1e-5 * (1.0 + abs(bound))) return best;",
                *(f"best = {child}(p, ray, best);" for child, _ in emitted),
                *(update(index) for index in tree.fields),
                "return best;",
            ]
            name = self.functions.add(
                "queryUnion", "vec2", f"vec{dim} p, vec{dim} ray, vec2 best", " ".join(lines)
            )
            return name, scale

        root, _ = emit(_build(entries, dim, 4))
        lines = [f"vec2 best = {names[0]}(p, ray);"]
        lines += [update(i) for i, cert in enumerate(certificates[1:], 1) if cert is None]
        lines += [f"best = {root}(p, ray, best);", "return best;"]
        name = self._add("query", dim, " ".join(lines))
        self.names[id(node)] = name
        return name

    def lower(self, node):
        key = id(node)
        if key in self.names:
            return self.names[key]
        dim = node.dim
        if key in self.raw_steps:
            name = self._add("query", dim, f"return vec2({self.raw_names[key]}(p).x);")
            self.names[key] = name
            return name
        if node.kind == "min":
            union = self._union(node)
            if union is not None:
                return union
        children = [self.lower(child) for child in node.children]
        kind = node.kind
        child = children[0] if children else None
        if kind in ("material", "feature"):
            self.names[key] = child
            return child
        raw = self.raw_names[key]
        if kind in ("min", "max"):
            comparison = "<" if kind == "min" else ">"
            body = (
                f"vec2 a = {children[0]}(p, ray), b = {children[1]}(p, ray); "
                f"return vec2(a.x {comparison} b.x ? a.x : b.x, {kind}(a.y, b.y));"
            )
        elif kind == "subtract":
            body = f"return max({children[0]}(p, ray), -{children[1]}(p, ray));"
        elif kind == "linear_pattern":
            sample = f"{child}(p - float(i) * {_v(node.data.step)}, ray)"
            body = (
                "vec2 best = vec2(1e30); "
                f"for (int i = 0; i < {node.data.count}; ++i) {{ "
                + self._pattern_sample(sample)
                + "} return best;"
            )
        elif kind == "radial_pattern":
            axis = node.data.axis
            local = _inverse_rotation("p", "c", "s", axis)
            local_ray = _inverse_rotation("ray", "c", "s", axis)
            body = (
                "vec2 best = vec2(1e30); "
                f"for (int i = 0; i < {node.data.count}; ++i) {{ "
                f"float angle = float(i) * {_f(2 * pi / node.data.count)}; "
                "float c = cos(angle), s = sin(angle); "
                + self._pattern_sample(f"{child}({local}, {local_ray})")
                + "} return best;"
            )
        elif kind == "transform":
            transform = node.data.transform
            inverse = np.asarray(transform.inverse_matrix, dtype=float)
            mapped = _linear(inverse, _minus("p", transform.offset))
            local_ray = _linear(inverse, "ray")
            scale = transform.distance_scale
            if np.allclose(np.linalg.svd(inverse, compute_uv=False), 1.0, rtol=1e-12, atol=0):
                if scale == 1.0:
                    if mapped == "p" and local_ray == "ray":
                        self.names[key] = child
                        return child
                    body = f"return {child}({mapped}, {local_ray});"
                else:
                    body = (
                        f"vec2 h = {child}({mapped}, {local_ray}); "
                        f"return vec2(h.x * {_f(scale)}, h.y);"
                    )
            else:
                body = (
                    f"vec{dim} localRay = {local_ray}; float speed = length(localRay); "
                    f"vec2 h = {child}({mapped}, localRay / speed); "
                    f"return vec2(h.x * {_f(scale)}, h.y / speed);"
                )
        else:
            advance = self._section_step(node) if kind == "loft" else self._raw_step(node, raw)
            advance_name = self.functions.add(
                "queryAdvance", "float", f"vec{dim} p, vec{dim} ray, float distance", advance
            )
            body = (
                f"float distance = {raw}(p).x; "
                f"return vec2(distance, {advance_name}(p, ray, distance));"
            )
        name = self._add("query", dim, body)
        self.names[key] = name
        return name

    def _pattern_sample(self, sample):
        return f"vec2 h = {sample}; if (h.x <= best.x) best.x = h.x; best.y = min(best.y, h.y); "

    def _section_step(self, node):
        lines = ["float best = 1e30;"]
        for box, pad in _section_step_boxes(node):
            bound = max(1.0, _lipschitz(node, box))
            lines.extend(
                [
                    "{",
                    f"vec3 gap = max(max({_v(box.minimum)} - p, p - {_v(box.maximum)}), 0.0);",
                    "float candidate;",
                    "if (any(greaterThan(gap, vec3(0.0))))",
                    f"  candidate = rayBoxEntry(p, ray, {_v(box.minimum)}, {_v(box.maximum)}) + {_f(pad * 0.5)};",
                    "else",
                    f"  candidate = {_over('distance', bound)};",
                    "best = min(best, candidate);",
                    "}",
                ]
            )
        lines.append("return best;")
        return " ".join(lines)

    def _raw_step(self, node, raw):
        from .bounds import AABB

        if node.kind in _PRIMITIVES:
            return "return distance;"

        box = node.bounds()
        if box.is_empty:
            return "return 1e30;"
        padded = _step_box(node)
        finite = padded is not None
        if finite:
            box, pad = padded
        elif node.kind in ("field_morph", "loft", "variable_extrude"):
            raise ValueError(f"GLSL {node.kind} needs finite bounds for conservative stepping")
        else:
            box = AABB((0.0,) * node.dim, (0.0,) * node.dim)
        bound = _lipschitz(node, box)
        if finite:
            lo, hi = _v(box.minimum), _v(box.maximum)
            if node.kind == "variable_extrude":
                advance = (
                    f"float entry = rayBoxEntry(p, ray, {lo}, {hi}); "
                    "if (entry >= 1e29) return entry; "
                    f"vec{node.dim} boundary = clamp(p + entry * ray, {lo}, {hi}); "
                    f"return entry + max({raw}(boundary).x, 0.0) / {_f(2 * max(1.0, bound))}; "
                )
            else:
                advance = f"return rayBoxEntry(p, ray, {lo}, {hi}) + {_f(pad * 0.5)}; "
            outside = (
                f"vec{node.dim} gap = max(max({lo} - p, p - {hi}), 0.0); "
                "if (any(greaterThan(gap, vec"
                f"{node.dim}(0.0)))) {{ {advance} }} "
            )
        else:
            outside = ""
        return f"{outside}return {_over('distance', max(1.0, bound))};"


_HELPERS = """
float rayBoxEntry(vec2 p, vec2 ray, vec2 lo, vec2 hi) {
  float near = 0.0, far = 1e30;
  for (int axis = 0; axis < 2; ++axis) {
    if (ray[axis] == 0.0) {
      if (p[axis] < lo[axis] || p[axis] > hi[axis]) return 1e30;
    } else {
      float a = (lo[axis] - p[axis]) / ray[axis];
      float b = (hi[axis] - p[axis]) / ray[axis];
      near = max(near, min(a, b));
      far = min(far, max(a, b));
    }
  }
  return far < near ? 1e30 : near;
}
float rayBoxEntry(vec3 p, vec3 ray, vec3 lo, vec3 hi) {
  float near = 0.0, far = 1e30;
  for (int axis = 0; axis < 3; ++axis) {
    if (ray[axis] == 0.0) {
      if (p[axis] < lo[axis] || p[axis] > hi[axis]) return 1e30;
    } else {
      float a = (lo[axis] - p[axis]) / ray[axis];
      float b = (hi[axis] - p[axis]) / ray[axis];
      near = max(near, min(a, b));
      far = min(far, max(a, b));
    }
  }
  return far < near ? 1e30 : near;
}
float cap2(float a, float b) {
  return length(max(vec2(a, b), 0.0)) + min(max(a, b), 0.0);
}
float sdBox2(vec2 p, vec2 halfSize, float radius) {
  vec2 q = abs(p) - halfSize + radius;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}
float sdBox3(vec3 p, vec3 halfSize, float radius) {
  vec3 q = abs(p) - halfSize + radius;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - radius;
}
float sdSegment2(vec2 p, vec2 a, vec2 b) {
  vec2 e = b - a;
  return length(p - a - clamp(dot(p - a, e) / dot(e, e), 0.0, 1.0) * e);
}
float sdSegment3(vec3 p, vec3 a, vec3 b) {
  vec3 e = b - a;
  return length(p - a - clamp(dot(p - a, e) / dot(e, e), 0.0, 1.0) * e);
}
float sdArc2(vec2 p, vec2 center, vec2 start, vec2 end, float sweep) {
  vec2 radial = p - center;
  float startAngle = atan(start.y - center.y, start.x - center.x);
  float progress = mod((atan(radial.y, radial.x) - startAngle) * sign(sweep), 6.283185307179586);
  float circleDistance = abs(length(radial) - length(start - center));
  float endpointDistance = min(length(p - start), length(p - end));
  return progress <= abs(sweep) ? circleDistance : endpointDistance;
}
float quadDistance2At(vec2 p, vec2 a, vec2 b, vec2 c, float t) {
  vec2 delta = a + t * (b + t * c) - p;
  return dot(delta, delta);
}
float signedCbrt(float x) {
  return sign(x) * pow(abs(x), 0.3333333333333333);
}
float sdQuadratic2(vec2 p, vec2 a, vec2 b, vec2 c) {
  float b2 = dot(b, b);
  float c2 = dot(c, c);
  if (c2 <= 1e-10 * b2) {
    vec2 edge = b + c;
    float edge2 = dot(edge, edge);
    if (edge2 == 0.0) return length(p - a);
    float t = clamp(dot(p - a, edge) / edge2, 0.0, 1.0);
    return length(p - a - t * edge);
  }
  float best = min(quadDistance2At(p, a, b, c, 0.0),
                   quadDistance2At(p, a, b, c, 1.0));
  vec2 offset = a - p;
  float A = 1.5 * dot(b, c) / c2;
  float B = (0.5 * b2 + dot(offset, c)) / c2;
  float C = 0.5 * dot(offset, b) / c2;
  float thirdP = (B - A * A / 3.0) / 3.0;
  float halfQ = 0.5 * (2.0 * A * A * A / 27.0 - A * B / 3.0 + C);
  float discriminant = halfQ * halfQ + thirdP * thirdP * thirdP;
  if (discriminant >= 0.0) {
    float root = signedCbrt(-halfQ + sqrt(discriminant))
               + signedCbrt(-halfQ - sqrt(discriminant)) - A / 3.0;
    if (root > 0.0 && root < 1.0)
      best = min(best, quadDistance2At(p, a, b, c, root));
    if (discriminant == 0.0) {
      root = -0.5 * (root + A / 3.0) - A / 3.0;
      if (root > 0.0 && root < 1.0)
        best = min(best, quadDistance2At(p, a, b, c, root));
    }
  } else {
    float radius = sqrt(-thirdP);
    float angle = acos(clamp(-halfQ / (radius * radius * radius), -1.0, 1.0)) / 3.0;
    for (int i = 0; i < 3; ++i) {
      float root = 2.0 * radius * cos(angle - 2.0943951023931953 * float(i)) - A / 3.0;
      if (root > 0.0 && root < 1.0)
        best = min(best, quadDistance2At(p, a, b, c, root));
    }
  }
  return sqrt(best);
}
float quadraticXAtY(float y, vec2 a, vec2 b, vec2 c, float lo, float hi) {
  float t;
  float offset = a.y - y;
  if (abs(c.y) <= 1e-7 * abs(b.y)) {
    t = (y - a.y) / b.y;
  } else {
    float radical = sqrt(max(b.y * b.y - 4.0 * c.y * offset, 0.0));
    float q = -0.5 * (b.y + (b.y >= 0.0 ? radical : -radical));
    float first = q / c.y;
    float second = abs(q) > 1e-30 ? offset / q : -b.y / (2.0 * c.y);
    t = first >= lo && first <= hi ? first : second;
  }
  t = clamp(t, lo, hi);
  return a.x + t * (b.x + t * c.x);
}
"""


def _quadratic_coefficients(segment):
    a, control, end = segment.points
    return (
        a,
        tuple(2 * (control[i] - a[i]) for i in range(2)),
        tuple(a[i] - 2 * control[i] + end[i] for i in range(2)),
    )


def feature_names(scene):
    """Return the feature ID table used by generated GLSL (index zero is unnamed)."""
    scene = compile_scene(scene).field
    names = [None]
    seen = set()

    def visit(node):
        if node.kind == "feature" and node.data.name not in seen:
            seen.add(node.data.name)
            names.append(node.data.name)
        for child in node.children:
            visit(child)

    visit(scene)
    return tuple(names)


class _Lowerer:
    def __init__(self, feature_ids, *, accelerated=False):
        from .glsl_pruning import _ValueHierarchy

        self.functions = _Functions()
        self.names = {}
        self.feature_ids = feature_ids
        self.accelerated = accelerated
        self.hierarchy = _ValueHierarchy(self.functions)
        self.raw_steps = set()

    def lower(self, node, *, union_child=False):
        key = id(node)
        if key in self.names:
            return self.names[key]
        if len(self.functions) >= _MAX_NODES:
            raise ValueError(f"GLSL scene exceeds {_MAX_NODES} distinct field functions")
        children = [self.lower(child, union_child=node.kind == "min") for child in node.children]
        if self._raw_step(node):
            self.raw_steps.add(key)
        if node.kind == "transform" and node.data.transform.distance_scale == 1.0:
            transform = node.data.transform
            if _linear(transform.inverse_matrix, _minus("p", transform.offset)) == "p":
                self.names[key] = children[0]
                return children[0]
        lines = (
            self._accelerated_body(node, children)
            if self.accelerated and not (union_child and node.kind == "min")
            else None
        )
        if lines is None:
            lines = self._body(node, children)
        body = "\n".join(f"  {line}" for line in lines)
        self.names[key] = self.functions.add("field", "vec3", f"vec{node.dim} p", body)
        return self.names[key]

    def _raw_step(self, node):
        if node.kind in {
            "circle",
            "sphere",
            "rectangle",
            "box",
            "cylinder",
            "torus",
            "capsule",
            "tube",
            "curve_profile",
            "polygon",
        }:
            return True
        if node.kind in {
            "min",
            "max",
            "subtract",
            "linear_pattern",
            "radial_pattern",
            "material",
            "feature",
        }:
            return all(id(child) in self.raw_steps for child in node.children)
        if node.kind == "transform" and id(node.children[0]) in self.raw_steps:
            stretch = np.linalg.svd(np.asarray(node.data.transform.matrix), compute_uv=False)
            return bool(
                np.allclose(stretch, node.data.transform.distance_scale, rtol=1e-10, atol=0)
            )
        return False

    def _accelerated_body(self, node, children):
        from .bounds import AABB, _affine, _points_bounds
        from .bvh import certificate as _certificate

        if node.kind == "min":
            components, pending = [], [node]
            while pending:
                current = pending.pop()
                if current.kind == "min":
                    pending.extend(reversed(current.children))
                else:
                    components.append(current)
            entries = [(f"{self.names[id(field)]}(p)", _certificate(field)) for field in components]
        elif node.kind in {"linear_pattern", "radial_pattern"}:
            certificate = _certificate(node.children[0])
            if certificate is None or not 4 <= node.data.count <= 256:
                return None
            box, scale = certificate
            entries = []
            for index in range(node.data.count):
                if node.kind == "linear_pattern":
                    delta = tuple(index * value for value in node.data.step)
                    instance_box = AABB(
                        tuple(a + d for a, d in zip(box.minimum, delta, strict=True)),
                        tuple(b + d for b, d in zip(box.maximum, delta, strict=True)),
                    )
                    point = _minus("p", delta)
                else:
                    angle = index * 2 * pi / node.data.count
                    inverse = _rotation_inverse(cos(angle), sin(angle), node.data.axis)
                    instance_box = _affine(box, inverse.T, (0, 0, 0))
                    point = _linear(inverse, "p")
                entries.append((f"{children[0]}({point})", (instance_box, scale)))
        elif node.kind == "tube":
            entries = [
                (
                    f"vec3(sdSegment3(p, {_v(a)}, {_v(b)}) - {_f(node.data.radius)}, 0.0, 0.0)",
                    (_points_bounds((a, b)).expanded(node.data.radius), 1.0),
                )
                for a, b in zip(node.data.points, node.data.points[1:], strict=False)
            ]
        else:
            return None
        return self.hierarchy.lower(entries, node.dim)

    def _body(self, node, children):
        kind, data = node.kind, node.data
        child = children[0] if children else None
        if kind in ("circle", "sphere"):
            return [f"return vec3(length(p) - {_f(data.radius)}, 0.0, 0.0);"]
        if kind in ("rectangle", "box"):
            sizes = (
                (data.width, data.height)
                if node.dim == 2
                else (data.width, data.depth, data.height)
            )
            radius = data.corner_radius
            return [
                f"return vec3(sdBox{node.dim}(p, {_v(s / 2 for s in sizes)}, {_f(radius)}), 0.0, 0.0);"
            ]
        if kind == "cylinder":
            radius, height = data.radius, data.height
            return [
                f"return vec3(cap2(length(p.xy) - {_f(radius)}, abs(p.z) - {_f(height / 2)}), 0.0, 0.0);"
            ]
        if kind == "torus":
            major, tube = data.major_radius, data.tube_radius
            return [
                f"return vec3(length(vec2(length(p.xy) - {_f(major)}, p.z)) - {_f(tube)}, 0.0, 0.0);"
            ]
        if kind == "capsule":
            a, b, radius = data.start, data.end, data.radius
            return [f"return vec3(sdSegment3(p, {_v(a)}, {_v(b)}) - {_f(radius)}, 0.0, 0.0);"]
        if kind == "tube":
            points, radius = data.points, data.radius
            lines = ["float d = 1e30;"]
            for a, b in zip(points, points[1:], strict=False):
                lines.append(f"d = min(d, sdSegment3(p, {_v(a)}, {_v(b)}));")
            return [*lines, f"return vec3(d - {_f(radius)}, 0.0, 0.0);"]
        if kind == "curve_profile":
            return self._curve_profile(data.segments)
        if kind == "polygon":
            vertices = data.vertices
            if len(vertices) > _MAX_PROFILE_SEGMENTS:
                raise ValueError(f"GLSL profiles support at most {_MAX_PROFILE_SEGMENTS} edges")
            lines = ["float d = 1e30;", "bool inside = false;"]
            for a, b in zip(vertices, vertices[1:] + vertices[:1], strict=False):
                lines.append(f"d = min(d, sdSegment2(p, {_v(a)}, {_v(b)}));")
                if a[1] != b[1]:
                    lines.append(
                        f"inside = inside != (({_f(a[1])} > p.y) != ({_f(b[1])} > p.y) && p.x < {_f(a[0])} + (p.y - {_f(a[1])}) * {_f((b[0] - a[0]) / (b[1] - a[1]))});"
                    )
            return [*lines, "return vec3(inside ? -d : d, 0.0, 0.0);"]
        if kind in ("min", "max", "subtract", "smooth_union"):
            lines = [f"vec3 a = {children[0]}(p);", f"vec3 b = {children[1]}(p);"]
            if kind == "min":
                return [*lines, "return a.x < b.x ? a : b;"]
            if kind == "max":
                return [*lines, "return a.x > b.x ? a : b;"]
            if kind == "subtract":
                return [*lines, "return vec3(max(a.x, -b.x), a.y, a.x > -b.x ? a.z : b.z);"]
            radius = _f(data.radius)
            return [
                *lines,
                f"float h = max({radius} - abs(a.x - b.x), 0.0) / {radius};",
                f"return vec3(min(a.x, b.x) - h * h * {radius} * 0.25, a.x < b.x ? a.yz : b.yz);",
            ]
        if kind == "linear_pattern":
            return [
                "vec3 best = vec3(1e30, 0.0, 0.0);",
                f"for (int i = 0; i < {data.count}; ++i) {{",
                f"  vec3 h = {child}(p - float(i) * {_v(data.step)});",
                "  if (h.x <= best.x) best = h;",
                "}",
                "return best;",
            ]
        if kind == "radial_pattern":
            local = _inverse_rotation("p", "c", "s", data.axis)
            return [
                "vec3 best = vec3(1e30, 0.0, 0.0);",
                f"for (int i = 0; i < {data.count}; ++i) {{",
                f"  float angle = float(i) * {_f(2 * pi / data.count)};",
                "  float c = cos(angle), s = sin(angle);",
                f"  vec3 h = {child}({local});",
                "  if (h.x <= best.x) best = h;",
                "}",
                "return best;",
            ]
        if kind == "transform":
            transform = data.transform
            mapped = _linear(transform.inverse_matrix, _minus("p", transform.offset))
            if transform.distance_scale == 1.0:
                return [f"return {child}({mapped});"]
            return [
                f"vec3 h = {child}({mapped});",
                f"return vec3(h.x * {_f(transform.distance_scale)}, h.yz);",
            ]
        if kind == "slice":
            plane = data.plane
            directions = tuple(zip(plane.x_direction, plane.y_direction, strict=True))
            mapped = _linear(directions, "p")
            if any(plane.origin):
                mapped = f"{mapped} + {_v(plane.origin)}"
            return [f"return {child}({mapped});"]
        if kind == "warp":
            from .expr import coordinates, to_glsl

            try:
                mapped = tuple(data.warp.inverse(coordinates()[: node.dim]))
                if len(mapped) != node.dim:
                    raise ValueError("wrong coordinate count")
                expressions = tuple(to_glsl(value) for value in mapped)
            except Exception as exc:
                raise ValueError("GLSL warp inverse must return symbolic coordinates") from exc
            mapped = f"vec{node.dim}(" + ", ".join(expressions) + ")"
            return [
                *(f"float {axis} = p.{axis};" for axis in "xyz"[: node.dim]),
                f"vec3 h = {child}({mapped});",
                f"return vec3(max(h.x * {_f(data.warp.distance_scale)}, {_bounds_distance(data.bounds)}), h.yz);",
            ]
        if kind == "offset":
            return [f"vec3 h = {child}(p);", f"return vec3(h.x - {_f(data.amount)}, h.yz);"]
        if kind == "shell":
            return [
                f"vec3 h = {child}(p);",
                f"return vec3(abs(h.x) - {_f(data.thickness / 2)}, h.yz);",
            ]
        if kind == "material":
            return [f"vec3 h = {child}(p);", f"return vec3(h.x, {_f(data.material_id)}, h.z);"]
        if kind == "feature":
            return [
                f"vec3 h = {child}(p);",
                f"return vec3(h.xy, {_f(self.feature_ids[data.name])});",
            ]
        if kind == "extrude":
            height, offset = data.height, data.offset
            z = f"p.z - {_f(offset)}" if offset else "p.z"
            return [
                f"vec3 h = {child}(p.xy);",
                f"return vec3(cap2(h.x, abs({z}) - {_f(height / 2)}), h.yz);",
            ]
        if kind == "variable_extrude":
            height, offset = data.height, data.offset
            from .expr import coordinates, to_glsl

            x, y = coordinates()[:2]
            try:
                height = height(x, y) if callable(height) else height
                offset = offset(x, y) if callable(offset) else offset
                height_glsl, offset_glsl = to_glsl(height), to_glsl(offset)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "GLSL extrusion callables must return symbolic scalar fields"
                ) from exc
            return [
                f"float x = clamp(p.x, {_f(data.bounds.minimum[0])}, {_f(data.bounds.maximum[0])});",
                f"float y = clamp(p.y, {_f(data.bounds.minimum[1])}, {_f(data.bounds.maximum[1])});",
                f"vec3 h = {child}(p.xy);",
                f"return vec3(max(cap2(h.x, abs(p.z - ({offset_glsl})) - ({height_glsl}) * 0.5), {_bounds_distance(data.bounds)}), h.yz);",
            ]
        if kind == "revolve":
            near, far = _revolve_sides(node.children[0])
            miss = "vec3(1e30, 0.0, 0.0)"
            lines = [
                "float r = length(p.xy);",
                f"vec3 h = {f'{child}(vec2(r, p.z))' if near else miss};",
                f"vec3 o = {f'{child}(vec2(-r, p.z))' if far else miss};",
            ]
            if data.angle >= 2 * pi:
                return [*lines, "return h.x <= o.x ? h : o;"]
            c, s = cos(data.angle), sin(data.angle)
            return [
                *lines,
                f"float phi = mod(atan(p.y, p.x), {_f(2 * pi)});",
                f"if (phi > {_f(data.angle)}) h.x = 1e30;",
                f"if (mod(phi + {_f(pi)}, {_f(2 * pi)}) > {_f(data.angle)}) o.x = 1e30;",
                "if (o.x < h.x) h = o;",
                f"vec2 e = vec2({_f(c)} * p.x + {_f(s)} * p.y, {_f(c)} * p.y - {_f(s)} * p.x);",
                "if (h.x < 0.0) return vec3(max(h.x, -min(abs(p.y), abs(e.y))), h.yz);",
                f"vec3 a = {child}(p.xz);",
                f"vec3 b = {child}(vec2(e.x, p.z));",
                "a.x = length(vec2(max(a.x, 0.0), p.y));",
                "b.x = length(vec2(max(b.x, 0.0), e.y));",
                "if (a.x < h.x) h = a;",
                "if (b.x < h.x) h = b;",
                "return h;",
            ]
        if kind in ("field_morph", "loft"):
            return self._sections(node, children)
        if kind == "profile_sweep":
            return self._sweep(node, child)
        if kind == "custom":
            from .expr import compile_glsl as compile_field_glsl
            from .expr import to_glsl

            coordinate_lines = [f"float {axis} = p.{axis};" for axis in "xyz"[: node.dim]]
            declarations, result = compile_field_glsl(data.distance)
            material = to_glsl(data.material)
            return [
                *coordinate_lines,
                *declarations,
                f"return vec3(max({result}, {_bounds_distance(data.bounds)}), {material}, 0.0);",
            ]
        raise ValueError(f"GLSL lowering does not support field kind '{kind}'")

    def _curve_profile(self, segments):
        from .numpy_eval import _curve_y_breaks

        if len(segments) > _MAX_PROFILE_SEGMENTS:
            raise ValueError(f"GLSL profiles support at most {_MAX_PROFILE_SEGMENTS} segments")
        simple_distances = []
        bezier_distances = []
        crossings = []
        for segment in segments:
            if segment.kind == "line":
                a, b = segment.points
                simple_distances.append(f"d = min(d, sdSegment2(p, {_v(a)}, {_v(b)}));")
                if a[1] != b[1]:
                    x = f"{_f(a[0])} + (p.y - {_f(a[1])}) * {_f((b[0] - a[0]) / (b[1] - a[1]))}"
                    crossings.append(
                        f"inside = inside != ((({_f(a[1])} > p.y) != ({_f(b[1])} > p.y)) && p.x < {x});"
                    )
                continue

            if segment.kind == "arc":
                center, start, end = segment.points
                radius = hypot(start[0] - center[0], start[1] - center[1])
                simple_distances.append(
                    f"d = min(d, sdArc2(p, {_v(center)}, {_v(start)}, {_v(end)}, {_f(segment.sweep)}));"
                )
            elif segment.kind == "quad":
                coefficients = _quadratic_coefficients(segment)
                quad_args = ", ".join(_v(value) for value in coefficients)
                low = tuple(min(point[i] for point in segment.points) for i in range(2))
                high = tuple(max(point[i] for point in segment.points) for i in range(2))
                bezier_distances.append(
                    f"if (length(max(max({_v(low)} - p, p - {_v(high)}), 0.0)) < d) "
                    f"d = min(d, sdQuadratic2(p, {quad_args}));"
                )
            else:
                raise ValueError(f"unknown curve segment kind: {segment.kind}")

            breaks = _curve_y_breaks(segment)
            for lo, hi in zip(breaks, breaks[1:], strict=False):
                y0, y1 = segment.point(lo)[1], segment.point(hi)[1]
                if abs(y1 - y0) < 1e-14:
                    continue
                if segment.kind == "arc":
                    side = 1 if segment.point((lo + hi) / 2)[0] >= center[0] else -1
                    x = (
                        f"{_f(center[0])} + {_f(side)} * "
                        f"sqrt(max(0.0, {_f(radius * radius)} - "
                        f"(p.y - {_f(center[1])}) * (p.y - {_f(center[1])})))"
                    )
                else:
                    x = f"quadraticXAtY(p.y, {quad_args}, {_f(lo)}, {_f(hi)})"
                crossings.append(
                    f"inside = inside != ((({_f(y0)} > p.y) != ({_f(y1)} > p.y)) && p.x < {x});"
                )
        return [
            "float d = 1e30;",
            *simple_distances,
            *bezier_distances,
            "bool inside = false;",
            *crossings,
            "return vec3(inside ? -d : d, 0.0, 0.0);",
        ]

    def _sections(self, node, children):
        stations, (axis, transverse) = node.data.stations, node.data.axes
        ax = "xyz"[axis]
        u, v = ("xyz"[i] for i in transverse)
        lines = [f"float z = p.{ax};", f"vec2 uv = p.{u}{v};"]
        if node.kind == "loft":
            lines += ["vec4 section = vec4(0.0);", "float t;"]
            for i, (a, b) in enumerate(zip(stations, stations[1:], strict=False)):
                condition = "if" if i == 0 else "else if"
                bound = f"z < {_f(b[0])}" if i < len(stations) - 2 else "true"
                lines += [
                    f"{condition} ({bound}) {{",
                    f"  t = clamp((z - {_f(a[0])}) / {_f(b[0] - a[0])}, 0.0, 1.0);",
                    f"  section = mix({_v(a[1:])}, {_v(b[1:])}, t);",
                    "}",
                ]
            lines += [
                f"vec3 h = {children[0]}((uv - section.xy) / section.zw);",
                "h.x *= min(section.z, section.w);",
            ]
            first, last = stations[0][0], stations[-1][0]
        else:
            lines += ["vec3 h = vec3(1e30, 0.0, 0.0);", "float t;"]
            for i, (a, b) in enumerate(zip(stations, stations[1:], strict=False)):
                condition = "if" if i == 0 else "else if"
                bound = f"z < {_f(b)}" if i < len(stations) - 2 else "true"
                lines += [
                    f"{condition} ({bound}) {{",
                    f"  t = clamp((z - {_f(a)}) / {_f(b - a)}, 0.0, 1.0);",
                    f"  vec3 left = {children[i]}(uv);",
                    f"  vec3 right = {children[i + 1]}(uv);",
                    "  h = vec3(mix(left.x, right.x, t), t < 0.5 ? left.yz : right.yz);",
                    "}",
                ]
            first, last = stations[0], stations[-1]
        lines.append(f"h.x = cap2(h.x, max({_f(first)} - z, z - {_f(last)}));")
        lines += _clip(node.data.bounds, "h")
        lines.append("return h;")
        return lines

    def _sweep(self, node, child):
        from .sweep import _spans

        if len(node.data.frames) > _MAX_SWEEP_SPANS:
            raise ValueError(f"GLSL sweeps support at most {_MAX_SWEEP_SPANS} sampled spans")
        lines = ["vec3 best = vec3(1e30, 0.0, 0.0);"]
        for span in _spans(node.data):
            ns, ne = span.start_normal, span.end_normal
            lines += [
                "{",
                f"  vec3 r = p - {_v(span.start)};",
                f"  vec3 h = {child}(vec2(dot(r, {_v(span.u)}), dot(r, {_v(span.v)})));",
                f"  float cap = max(-dot(r, {_v(ns)}) - {_f(span.start_pad)}, dot(p - {_v(span.end)}, {_v(ne)}) - {_f(span.end_pad)});",
                f"  float d = cap2(h.x * {_f(span.scale)}, cap);",
                "  if (d < best.x) best = vec3(d, h.yz);",
                "}",
            ]
        lines += _clip(node.data.bounds, "best")
        lines.append("return best;")
        return lines


def compile_glsl(scene, *, bvh=True):
    """Return GLSL ES 3.00 for the union field and visible part surfaces.

    Set bvh to prune certified union components on the GPU.
    """
    from .glsl_trace import _surface_tracer

    traverse_parts = bvh
    compiled = compile_scene(scene)
    scene = compiled.field
    if scene.dim != 3:
        raise ValueError("GLSL scene must be 3D")
    from .bounds import AABB

    box = scene.bounds()
    if box.is_empty or not all(isfinite(x) for x in box.minimum + box.maximum):
        if any(kind in {"field_morph", "loft"} for kind in _field_kinds(scene)):
            raise ValueError(
                "GLSL loft and field_morph scenes need finite bounds for safe ray marching"
            )
        box = AABB((0.0,) * 3, (0.0,) * 3)
    else:
        padding = max(
            0.01, 0.0001 * hypot(*(b - a for a, b in zip(box.minimum, box.maximum, strict=True)))
        )
        box = box.expanded(padding)
    names = feature_names(compiled)
    compiler = _Lowerer(
        {name: index for index, name in enumerate(names) if name is not None},
        accelerated=traverse_parts,
    )
    part_roots = tuple(compiler.lower(part.field) for part in compiled.parts)
    root = compiler.lower(scene)
    query = _QueryLowerer(compiler.names, compiler.raw_steps)
    part_queries = tuple(query.lower(part.field) for part in compiled.parts)
    query_source = [f"vec2 mapQuery(vec3 p, vec3 ray) {{ return {query.lower(scene)}(p, ray); }}"]
    if len(part_roots) == 1:
        surface_source = [
            f"vec3 mapSurface(vec3 p) {{ return {part_roots[0]}(p); }}",
            f"vec2 mapSurfaceQuery(vec3 p, vec3 ray) {{ vec2 h = {part_queries[0]}(p, ray); return vec2(h.x, abs(h.y)); }}",
        ]
    else:
        surface_lines = [
            f"vec3 best = {part_roots[0]}(p);",
            "float distance = abs(best.x);",
            "vec3 h;",
        ]
        surface_lines.extend(
            f"h = {name}(p); if (abs(h.x) <= distance) {{ best = h; distance = abs(h.x); }}"
            for name in part_roots[1:]
        )
        surface_lines.append("return best;")
        query_lines = [
            f"vec2 best = {part_queries[0]}(p, ray);",
            "best.y = abs(best.y);",
            "float distance = abs(best.x);",
            "vec2 h;",
        ]
        query_lines.extend(
            f"h = {name}(p, ray); if (abs(h.x) <= distance) {{ best.x = h.x; distance = abs(h.x); }} best.y = min(best.y, abs(h.y));"
            for name in part_queries[1:]
        )
        query_lines.append("return best;")
        surface_source = [
            "vec3 mapSurface(vec3 p) {\n"
            + "\n".join(f"  {line}" for line in surface_lines)
            + "\n}",
            "vec2 mapSurfaceQuery(vec3 p, vec3 ray) {\n"
            + "\n".join(f"  {line}" for line in query_lines)
            + "\n}",
        ]
    surface_source.append(
        _surface_tracer(compiled, part_roots, part_queries, enabled=traverse_parts)
    )

    def assemble(map_scene, *emitters):
        functions = {}
        for emitter in (compiler.functions, query.functions, *emitters):
            functions.update(emitter.source)
        roots = "\n".join([map_scene, *query_source, *surface_source])
        return "\n".join([_HELPERS, *_reachable(functions, roots), roots])

    if not bvh:
        return assemble(f"vec3 mapScene(vec3 p) {{ return {root}(p); }}")

    from .bvh import _build, _union_components, certificate

    # Gap pruning only skips a component whose certificate bounds its value
    # above the current best, so the union stays exact for any field mix.
    components = _union_components(scene)
    certified = [
        (index, component, certificate(component)) for index, component in enumerate(components)
    ]
    certified = [(index, component, cert) for index, component, cert in certified if cert]
    if len(certified) < 4:
        return assemble(f"vec3 mapScene(vec3 p) {{ return {root}(p); }}")

    # Evaluate the first component unconditionally so the initial best value
    # is exact even when a field exceeds any finite GLSL sentinel.
    first = compiler.lower(components[0])
    certified = [(i, field, box) for i, field, box in certified if i != 0]
    certified_ids = {i for i, _, _ in certified}
    ordinary = [
        (i, compiler.lower(field))
        for i, field in enumerate(components[1:], 1)
        if i not in certified_ids
    ]
    indexed = {}
    entries = []
    for index, field, (box, scale) in certified:
        name = compiler.lower(field)
        indexed[id(field)] = (index, name, scale)
        center = tuple((a + b) / 2 for a, b in zip(box.minimum, box.maximum, strict=True))
        entries.append((field, box, center))
    tree = _build(entries, 3, 8)
    bvh_functions = _Functions()

    def emit(node):
        emitted = [emit(child) for child in node.children]
        child_names = [child for child, _ in emitted]
        scale = min([s for _, s in emitted] + [indexed[id(field)][2] for field in node.fields])
        low = _v(
            float(np.nextafter(np.float32(v), np.float32(-np.inf))) for v in node.bounds.minimum
        )
        high = _v(
            float(np.nextafter(np.float32(v), np.float32(np.inf))) for v in node.bounds.maximum
        )
        lines = [
            f"vec3 lo = {low};",
            f"vec3 hi = {high};",
            "if (any(lessThan(p, lo)) || any(greaterThan(p, hi))) {",
            "  vec3 gap = max(max(lo - p, p - hi), 0.0);",
            f"  if (length(gap) * {_f(np.nextafter(np.float32(scale), np.float32(0)))} > best.x + 1e-5 * (1.0 + abs(best.x))) return best;",
            "}",
        ]
        if child_names:
            lines.extend(f"best = {child}(p, best);" for child in child_names)
        else:
            for field in node.fields:
                index, field_name, _ = indexed[id(field)]
                lines.extend(
                    [
                        "{",
                        f"vec3 h = {field_name}(p);",
                        f"if (h.x < best.x || (h.x == best.x && {_f(index)} > best.w)) best = vec4(h, {_f(index)});",
                        "}",
                    ]
                )
        lines.append("return best;")
        body = "\n".join(f"  {line}" for line in lines)
        return bvh_functions.add("bvh", "vec4", "vec3 p, vec4 best", body), scale

    bvh_root = emit(tree)[0] if entries else None
    map_lines = [f"vec3 h = {first}(p);", "vec4 best = vec4(h, 0.0);"]
    for index, field_name in ordinary:
        map_lines.extend(
            [
                f"h = {field_name}(p);",
                f"if (h.x < best.x || (h.x == best.x && {_f(index)} > best.w)) best = vec4(h, {_f(index)});",
            ]
        )
    if bvh_root is not None:
        map_lines.append(f"best = {bvh_root}(p, best);")
    map_lines.append("return best.xyz;")
    map_scene = "vec3 mapScene(vec3 p) {\n" + "\n".join(f"  {line}" for line in map_lines) + "\n}"
    return assemble(map_scene, bvh_functions)
