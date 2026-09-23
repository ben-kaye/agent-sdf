"""Lower the immutable Shape tree directly to GLSL ES 3.00."""

from __future__ import annotations

from math import cos, hypot, isfinite, sin

import numpy as np

from .dsl import Model, Shape

_MAX_NODES = 1024
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


def _mat_inverse(matrix):
    try:
        result = np.linalg.inv(np.asarray(matrix, dtype=float))
    except np.linalg.LinAlgError as exc:
        raise ValueError("GLSL affine transform must be invertible") from exc
    return tuple(tuple(float(x) for x in row) for row in result)


def _affine_point(matrix, translation, p):
    coordinates = "xy" if len(translation) == 2 else "xyz"
    return (
        f"vec{len(translation)}("
        + ", ".join(
            " + ".join(
                f"{_f(value)} * ({p}.{axis} - {_f(translation[j])})"
                for j, (axis, value) in enumerate(zip(coordinates, row, strict=True))
            )
            for row in matrix
        )
        + ")"
    )


_HELPERS = """
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


class _Lowerer:
    def __init__(self):
        self.functions = []
        self.names = {}

    def lower(self, node):
        key = id(node)
        if key in self.names:
            return self.names[key]
        if len(self.names) >= _MAX_NODES:
            raise ValueError(f"GLSL scene exceeds {_MAX_NODES} shape nodes")
        children = [self.lower(child) for child in node.children]
        name = f"shape{len(self.names)}"
        self.names[key] = name
        dim = node.dim
        ptype = f"vec{dim}"
        lines = self._body(node, children)
        self.functions.append(
            f"vec2 {name}({ptype} p) {{\n" + "\n".join(f"  {line}" for line in lines) + "\n}"
        )
        return name

    def _body(self, node, children):
        kind, data = node.kind, node.data
        child = children[0] if children else None
        if kind in ("circle", "sphere"):
            return [f"return vec2(length(p) - {_f(data[0])}, 0.0);"]
        if kind in ("rectangle", "box"):
            *sizes, radius = data
            return [
                f"return vec2(sdBox{node.dim}(p, {_v(s / 2 for s in sizes)}, {_f(radius)}), 0.0);"
            ]
        if kind == "cylinder":
            radius, height = data
            return [
                f"return vec2(cap2(length(p.xy) - {_f(radius)}, abs(p.z) - {_f(height / 2)}), 0.0);"
            ]
        if kind == "torus":
            major, tube = data
            return [
                f"return vec2(length(vec2(length(p.xy) - {_f(major)}, p.z)) - {_f(tube)}, 0.0);"
            ]
        if kind == "capsule":
            a, b, radius = data
            return [f"return vec2(sdSegment3(p, {_v(a)}, {_v(b)}) - {_f(radius)}, 0.0);"]
        if kind == "tube":
            points, radius = data
            lines = ["float d = 1e30;"]
            for a, b in zip(points, points[1:], strict=False):
                lines.append(f"d = min(d, sdSegment3(p, {_v(a)}, {_v(b)}));")
            return [*lines, f"return vec2(d - {_f(radius)}, 0.0);"]
        if kind == "curve_profile":
            return self._curve_profile(data[0])
        if kind == "polygon":
            vertices = data[0]
            if len(vertices) > _MAX_PROFILE_SEGMENTS:
                raise ValueError(f"GLSL profiles support at most {_MAX_PROFILE_SEGMENTS} edges")
            lines = ["float d = 1e30;", "bool inside = false;"]
            for a, b in zip(vertices, vertices[1:] + vertices[:1], strict=False):
                lines.append(f"d = min(d, sdSegment2(p, {_v(a)}, {_v(b)}));")
                if a[1] != b[1]:
                    lines.append(
                        f"inside = inside != (({_f(a[1])} > p.y) != ({_f(b[1])} > p.y) && p.x < {_f(a[0])} + (p.y - {_f(a[1])}) * {_f((b[0] - a[0]) / (b[1] - a[1]))});"
                    )
            return [*lines, "return vec2(inside ? -d : d, 0.0);"]
        if kind in ("min", "max", "subtract", "smooth_union"):
            lines = [f"vec2 a = {children[0]}(p);", f"vec2 b = {children[1]}(p);"]
            if kind == "min":
                return [*lines, "return a.x < b.x ? a : b;"]
            if kind == "max":
                return [*lines, "return a.x > b.x ? a : b;"]
            if kind == "subtract":
                return [*lines, "return vec2(max(a.x, -b.x), a.y);"]
            radius = _f(data[0])
            return [
                *lines,
                f"float h = max({radius} - abs(a.x - b.x), 0.0) / {radius};",
                f"return vec2(min(a.x, b.x) - h * h * {radius} * 0.25, a.x < b.x ? a.y : b.y);",
            ]
        if kind == "translate":
            return [f"return {child}(p - {_v(data)});"]
        if kind == "scale":
            factor = _f(data[0])
            return [f"vec2 h = {child}(p / {factor});", f"return vec2(h.x * {factor}, h.y);"]
        if kind == "rotate":
            angle, axis = data
            c, s = _f(cos(angle)), _f(sin(angle))
            if node.dim == 2:
                mapped = f"vec2({c} * p.x + {s} * p.y, -{s} * p.x + {c} * p.y)"
            elif axis == "x":
                mapped = f"vec3(p.x, {c} * p.y + {s} * p.z, -{s} * p.y + {c} * p.z)"
            elif axis == "y":
                mapped = f"vec3({c} * p.x - {s} * p.z, p.y, {s} * p.x + {c} * p.z)"
            else:
                mapped = f"vec3({c} * p.x + {s} * p.y, -{s} * p.x + {c} * p.y, p.z)"
            return [f"return {child}({mapped});"]
        if kind == "transform":
            transform = data[0]
            if transform.forward_affine is None:
                raise ValueError("GLSL lowering requires forward_affine for custom transforms")
            matrix, translation = transform.forward_affine
            mapped = _affine_point(_mat_inverse(matrix), translation, "p")
            return [
                f"vec2 h = {child}({mapped});",
                f"return vec2(h.x * {_f(transform.distance_scale)}, h.y);",
            ]
        if kind == "offset":
            return [f"vec2 h = {child}(p);", f"return vec2(h.x - {_f(data[0])}, h.y);"]
        if kind == "shell":
            return [f"vec2 h = {child}(p);", f"return vec2(abs(h.x) - {_f(data[0] / 2)}, h.y);"]
        if kind == "material":
            return [f"vec2 h = {child}(p);", f"return vec2(h.x, {_f(data[0])});"]
        if kind == "extrude":
            height, offset = data
            return [
                f"vec2 h = {child}(p.xy);",
                f"return vec2(cap2(h.x, abs(p.z - {_f(offset)}) - {_f(height / 2)}), h.y);",
            ]
        if kind == "variable_extrude":
            height, offset = data[:2]
            from .field import coordinates, to_glsl

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
                "float x = p.x;",
                "float y = p.y;",
                f"vec2 h = {child}(p.xy);",
                f"return vec2(cap2(h.x, abs(p.z - ({offset_glsl})) - ({height_glsl}) * 0.5), h.y);",
            ]
        if kind == "revolve":
            return [f"return {child}(vec2(length(p.xy), p.z));"]
        if kind in ("loft", "section_loft"):
            return self._loft(node, children)
        if kind == "profile_sweep":
            return self._sweep(node, child)
        if kind in ("custom", "python"):
            from .field import compile_glsl as compile_field_glsl
            from .field import coordinates, to_glsl
            from .hit import Hit

            fn, material = data[:2]
            value = fn(*coordinates()[: node.dim])
            if isinstance(value, Hit):
                distance, material = value.distance, value.material
            else:
                distance = value
            coordinate_lines = [f"float {axis} = p.{axis};" for axis in "xyz"[: node.dim]]
            declarations, result = compile_field_glsl(distance)
            return [
                *coordinate_lines,
                *declarations,
                f"return vec2({result}, {to_glsl(material)});",
            ]
        raise ValueError(f"GLSL lowering does not support shape kind '{kind}'")

    def _curve_profile(self, segments):
        from .backend import _curve_y_breaks

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
            "return vec2(inside ? -d : d, 0.0);",
        ]

    def _loft(self, node, children):
        stations, (axis, transverse) = node.data
        ax = "xyz"[axis]
        u, v = ("xyz"[i] for i in transverse)
        lines = [f"float z = p.{ax};", f"vec2 uv = p.{u}{v};"]
        if node.kind == "section_loft":
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
                f"vec2 h = {children[0]}((uv - section.xy) / section.zw);",
                "h.x *= min(section.z, section.w);",
            ]
            first, last = stations[0][0], stations[-1][0]
        else:
            lines += ["vec2 h = vec2(1e30, 0.0);", "float t;"]
            for i, (a, b) in enumerate(zip(stations, stations[1:], strict=False)):
                condition = "if" if i == 0 else "else if"
                bound = f"z < {_f(b)}" if i < len(stations) - 2 else "true"
                lines += [
                    f"{condition} ({bound}) {{",
                    f"  t = clamp((z - {_f(a)}) / {_f(b - a)}, 0.0, 1.0);",
                    f"  vec2 left = {children[i]}(uv);",
                    f"  vec2 right = {children[i + 1]}(uv);",
                    "  h = vec2(mix(left.x, right.x, t), t < 0.5 ? left.y : right.y);",
                    "}",
                ]
            first, last = stations[0], stations[-1]
        lines += [f"h.x = cap2(h.x, max({_f(first)} - z, z - {_f(last)}));", "return h;"]
        return lines

    def _sweep(self, node, child):
        from .sweep import _miters

        points, frames, tolerance, closed = node.data
        if len(frames) > _MAX_SWEEP_SPANS:
            raise ValueError(f"GLSL sweeps support at most {_MAX_SWEEP_SPANS} sampled spans")
        start_normals, end_normals = _miters(frames, closed)
        lines = ["vec2 best = vec2(1e30, 0.0);"]
        for i, (a, b, frame, ns, ne) in enumerate(
            zip(points[:-1], points[1:], frames, start_normals, end_normals, strict=True)
        ):
            u, v, _ = frame
            start_pad = tolerance * 0.01 if closed or i else 0.0
            end_pad = tolerance * 0.01 if closed or i + 1 < len(frames) else 0.0
            lines += [
                "{",
                f"  vec3 r = p - {_v(a)};",
                f"  vec2 h = {child}(vec2(dot(r, {_v(u)}), dot(r, {_v(v)})));",
                f"  float cap = max(-dot(r, {_v(ns)}) - {_f(start_pad)}, dot(p - {_v(b)}, {_v(ne)}) - {_f(end_pad)});",
                "  float d = cap2(h.x, cap);",
                "  if (d < best.x) best = vec2(d, h.y);",
                "}",
            ]
        lines.append("return best;")
        return lines


def compile_glsl(scene, *, bvh=False):
    """Return GLSL ES 3.00 defining vec2 mapScene(vec3 p).

    Set bvh to prune certified union components on the GPU.
    """
    if isinstance(scene, Model):
        scene = scene.scene()
    if not isinstance(scene, Shape) or scene.dim != 3:
        raise ValueError("GLSL scene must be a 3D Shape or Model")
    compiler = _Lowerer()
    if not bvh:
        root = compiler.lower(scene)
        return "\n".join(
            [_HELPERS, *compiler.functions, f"vec2 mapScene(vec3 p) {{ return {root}(p); }}"]
        )

    from .bvh import _build, _certified_box, _union_components

    components = _union_components(scene)
    certified = [
        (index, component, _certified_box(component)) for index, component in enumerate(components)
    ]
    certified = [(index, component, box) for index, component, box in certified if box is not None]
    if len(certified) < 4:
        root = compiler.lower(scene)
        return "\n".join(
            [_HELPERS, *compiler.functions, f"vec2 mapScene(vec3 p) {{ return {root}(p); }}"]
        )

    # Evaluate the first component unconditionally so the initial best value
    # is exact even when a field exceeds any finite GLSL sentinel.
    first = compiler.lower(components[0])
    certified = [(i, shape, box) for i, shape, box in certified if i != 0]
    certified_ids = {i for i, _, _ in certified}
    ordinary = [
        (i, compiler.lower(shape))
        for i, shape in enumerate(components[1:], 1)
        if i not in certified_ids
    ]
    indexed = {}
    entries = []
    for index, shape, box in certified:
        name = compiler.lower(shape)
        indexed[id(shape)] = (index, name)
        center = tuple((a + b) / 2 for a, b in zip(box.minimum, box.maximum, strict=True))
        entries.append((shape, box, center))
    tree = _build(entries, 3, 1)
    bvh_functions = []

    def emit(node):
        child_names = [emit(child) for child in node.children]
        name = f"bvh{len(bvh_functions)}"
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
            "  if (length(gap) > best.x + 1e-5 * (1.0 + abs(best.x))) return best;",
            "}",
        ]
        if child_names:
            lines.extend(f"best = {child}(p, best);" for child in child_names)
        else:
            for shape in node.shapes:
                index, shape_name = indexed[id(shape)]
                lines.extend(
                    [
                        f"vec2 h = {shape_name}(p);",
                        f"if (h.x < best.x || (h.x == best.x && {_f(index)} > best.z)) best = vec3(h, {_f(index)});",
                    ]
                )
        lines.append("return best;")
        bvh_functions.append(
            f"vec3 {name}(vec3 p, vec3 best) {{\n"
            + "\n".join(f"  {line}" for line in lines)
            + "\n}"
        )
        return name

    bvh_root = emit(tree) if entries else None
    map_lines = [f"vec2 h = {first}(p);", "vec3 best = vec3(h, 0.0);"]
    for index, shape_name in ordinary:
        map_lines.extend(
            [
                f"h = {shape_name}(p);",
                f"if (h.x < best.x || (h.x == best.x && {_f(index)} > best.z)) best = vec3(h, {_f(index)});",
            ]
        )
    if bvh_root is not None:
        map_lines.append(f"best = {bvh_root}(p, best);")
    map_lines.append("return best.xy;")
    return "\n".join(
        [
            _HELPERS,
            *compiler.functions,
            *bvh_functions,
            "vec2 mapScene(vec3 p) {\n" + "\n".join(f"  {line}" for line in map_lines) + "\n}",
        ]
    )
