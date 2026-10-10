"""Generate bounded part traversal for primary surface rays."""

from math import hypot, isfinite

import numpy as np

from .bounds import _analytic
from .bvh import _build

_MARCH = """
bool marchPart(int part, vec3 origin, vec3 ray, float start, float end,
               bool escaping, out float travel, out vec3 hit) {
  travel = start;
  hit = vec3(0.0);
  for (int i = 0; i < 256; ++i) {
    if (travel > end) return false;
    vec3 p = origin + travel * ray;
    hit = mapHitSurface(p, part);
    if (isnan(hit.x) || isinf(hit.x)) return false;
    bool skip = escaping && abs(hit.x) < 0.004;
    if (!skip && abs(hit.x) < 0.001) return true;
    float step = mapHitStep(p, ray, part);
    if (isnan(step) || isinf(step)) return false;
    if (skip) step = max(step * 0.75, 0.002);
    else escaping = false;
    if (step <= 0.0) return false;
    float nextTravel = travel + step;
    if (nextTravel <= travel) return false;
    travel = nextTravel;
  }
  return false;
}
"""

_INTERVAL = """
vec2 partRayInterval(vec3 origin, vec3 ray, vec3 lo, vec3 hi,
                     float start, float end) {
  for (int axis = 0; axis < 3; ++axis) {
    if (ray[axis] == 0.0) {
      if (origin[axis] < lo[axis] || origin[axis] > hi[axis]) return vec2(1.0, 0.0);
    } else {
      float a = (lo[axis] - origin[axis]) / ray[axis];
      float b = (hi[axis] - origin[axis]) / ray[axis];
      start = max(start, min(a, b));
      end = min(end, max(a, b));
      if (start > end) return vec2(1.0, 0.0);
    }
  }
  return vec2(start, end);
}
"""


def _part_box(field):
    # Enclose the hit tolerance, not just the zero set. An arbitrary field
    # can be close to zero far outside its declared surface bounds.
    box = _analytic(field, 0.001)
    if box.is_empty:
        return box
    if not all(isfinite(v) for v in box.minimum + box.maximum):
        return None
    padding = max(1e-5, 1e-6 * hypot(*box.extents))
    return box.expanded(padding)


def _surface_tracer(compiled, roots, steps, *, enabled):
    from .glsl import _v

    if not enabled:
        return "\n".join(
            [
                "vec3 mapHitSurface(vec3 p, int part) { return mapSurface(p); }",
                "float mapHitStep(vec3 p, vec3 ray, int part) { return mapSurfaceStep(p, ray); }",
                _MARCH,
                """bool traceSurface(vec3 origin, vec3 ray, float start, float end,
                    bool escaping, out float travel, out vec3 hit, out int part) {
                  part = -1;
                  return marchPart(part, origin, ray, start, end, escaping, travel, hit);
                }""",
            ]
        )

    source = [
        "vec3 mapHitSurface(vec3 p, int part) { switch (part) {",
        *(f"case {i}: return {root}(p);" for i, root in enumerate(roots)),
        "} return vec3(1e30, 0.0, 0.0); }",
        "float mapHitStep(vec3 p, vec3 ray, int part) { switch (part) {",
        *(f"case {i}: return abs({step}(p, ray));" for i, step in enumerate(steps)),
        "} return 1e30; }",
        _MARCH,
        _INTERVAL,
    ]
    entries, ordinary = [], []
    for index, part in enumerate(compiled.parts):
        box = _part_box(part.field)
        if box is None:
            ordinary.append(index)
        elif not box.is_empty:
            center = tuple(a / 2 + b / 2 for a, b in zip(box.minimum, box.maximum, strict=True))
            entries.append((index, box, center))

    nodes = []

    def flatten(node):
        index = len(nodes)
        nodes.append(None)
        if node.children:
            children = [flatten(child) for child in node.children]
            links = (children[0][0], children[1][0], -1)
            depth = 1 + max(child[1] for child in children)
        else:
            links = (-1, -1, node.fields[0])
            depth = 1
        nodes[index] = (node.bounds, links)
        return index, depth

    if entries:
        _, depth = flatten(_build(entries, 3, 1))
        for name, side, direction in (("Lo", "minimum", -np.inf), ("Hi", "maximum", np.inf)):
            values = [
                _v(
                    float(np.nextafter(np.float32(v), np.float32(direction)))
                    for v in getattr(box, side)
                )
                for box, _ in nodes
            ]
            source.append(
                f"const vec3 partBounds{name}[{len(nodes)}] = vec3[]({', '.join(values)});"
            )
        links = [f"ivec3({a}, {b}, {part})" for _, (a, b, part) in nodes]
        source.append(f"const ivec3 partLinks[{len(nodes)}] = ivec3[]({', '.join(links)});")

    source.append("""
bool traceSurface(vec3 origin, vec3 ray, float start, float end, bool escaping,
                  out float travel, out vec3 hit, out int part) {
  travel = end;
  hit = vec3(0.0);
  part = -1;
  float candidateTravel;
  vec3 candidateHit;
""")
    for index in ordinary:
        source.append(f"""
  if (marchPart({index}, origin, ray, start, travel, escaping, candidateTravel, candidateHit)) {{
    if (part < 0 || candidateTravel < travel || (candidateTravel == travel && {index} > part)) {{
      travel = candidateTravel; hit = candidateHit; part = {index};
    }}
  }}
""")
    if entries:
        source.append(f"""
  int stack[{depth}];
  int count = 1;
  stack[0] = 0;
  for (int visited = 0; visited < {len(nodes)}; ++visited) {{
    if (count == 0) break;
    int node = stack[--count];
    vec2 interval = partRayInterval(origin, ray, partBoundsLo[node], partBoundsHi[node], start, travel);
    if (interval.x > interval.y) continue;
    ivec3 links = partLinks[node];
    if (links.z >= 0) {{
      // A jump to a later box has already left the previous transparent surface.
      bool skipSurface = escaping && interval.x == start;
      if (marchPart(links.z, origin, ray, interval.x, interval.y, skipSurface, candidateTravel, candidateHit)) {{
        if (part < 0 || candidateTravel < travel || (candidateTravel == travel && links.z > part)) {{
          travel = candidateTravel; hit = candidateHit; part = links.z;
        }}
      }}
    }} else {{
      vec2 left = partRayInterval(origin, ray, partBoundsLo[links.x], partBoundsHi[links.x], start, travel);
      vec2 right = partRayInterval(origin, ray, partBoundsLo[links.y], partBoundsHi[links.y], start, travel);
      bool useLeft = left.x <= left.y;
      bool useRight = right.x <= right.y;
      if (useLeft && useRight) {{
        bool leftFirst = left.x <= right.x;
        stack[count++] = leftFirst ? links.y : links.x;
        stack[count++] = leftFirst ? links.x : links.y;
      }} else if (useLeft) stack[count++] = links.x;
      else if (useRight) stack[count++] = links.y;
    }}
  }}
""")
    source.append("return part >= 0; }")
    return "\n".join(source)
