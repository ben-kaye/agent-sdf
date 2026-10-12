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
    vec2 query = mapHitQuery(p, ray, part);
    if (isnan(query.x) || isinf(query.x)) return false;
    bool skip = escaping && abs(query.x) < 0.004;
    if (!skip && abs(query.x) < 0.001) {
      hit = mapHitSurface(p, part);
      return true;
    }
    float step = query.y;
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


def _surface_tracer(compiled, roots, queries, *, enabled):
    from .glsl import _v

    if not enabled:
        return "\n".join(
            [
                "vec3 mapHitSurface(vec3 p, int part) { return mapSurface(p); }",
                "vec2 mapHitQuery(vec3 p, vec3 ray, int part) { return mapSurfaceQuery(p, ray); }",
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
        "vec2 mapHitQuery(vec3 p, vec3 ray, int part) { vec2 h; switch (part) {",
        *(
            f"case {i}: h = {query}(p, ray); return vec2(h.x, abs(h.y));"
            for i, query in enumerate(queries)
        ),
        "} return vec2(1e30); }",
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

    part_boxes = {index: box for index, box, _ in entries}
    for name, side, direction, fallback in (
        ("Lo", "minimum", -np.inf, -1e30),
        ("Hi", "maximum", np.inf, 1e30),
    ):
        values = [
            _v(
                float(np.nextafter(np.float32(v), np.float32(direction)))
                for v in getattr(part_boxes[index], side)
            )
            if index in part_boxes
            else _v((fallback if index in ordinary else -fallback,) * 3)
            for index in range(len(compiled.parts))
        ]
        source.append(f"const vec3 partBox{name}[{len(values)}] = vec3[]({', '.join(values)});")

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
    # Pick each march interval first so the loop inlines marchPart once.
    ordinary_march = f"""
    if (visited < {len(ordinary)}) {{
      candidate = ordinaryParts[visited];
      interval = vec2(start, travel);
      skipSurface = escaping;
    }}"""
    bvh_march = """
    {
      if (count == 0) break;
      int node = stack[--count];
      interval = partRayInterval(origin, ray, partBoundsLo[node], partBoundsHi[node], start, travel);
      if (interval.x > interval.y) continue;
      ivec3 links = partLinks[node];
      if (links.z < 0) {
        vec2 left = partRayInterval(origin, ray, partBoundsLo[links.x], partBoundsHi[links.x], start, travel);
        vec2 right = partRayInterval(origin, ray, partBoundsLo[links.y], partBoundsHi[links.y], start, travel);
        bool useLeft = left.x <= left.y;
        bool useRight = right.x <= right.y;
        if (useLeft && useRight) {
          bool leftFirst = left.x <= right.x;
          stack[count++] = leftFirst ? links.y : links.x;
          stack[count++] = leftFirst ? links.x : links.y;
        } else if (useLeft) stack[count++] = links.x;
        else if (useRight) stack[count++] = links.y;
        continue;
      }
      candidate = links.z;
      // A jump to a later box has already left the previous transparent surface.
      skipSurface = escaping && interval.x == start;
    }"""
    if ordinary:
        source.append(
            f"  const int ordinaryParts[{len(ordinary)}] = int[]({', '.join(map(str, ordinary))});"
        )
    if entries:
        source.append(f"  int stack[{depth}];\n  int count = 1;\n  stack[0] = 0;")
    if ordinary or entries:
        branches = ((ordinary_march, ordinary), (bvh_march, entries))
        select = " else".join(branch for branch, used in branches if used)
        source.append(f"""
  for (int visited = 0; visited < {len(ordinary) + len(nodes)}; ++visited) {{
    int candidate;
    vec2 interval;
    bool skipSurface;{select}
    if (marchPart(candidate, origin, ray, interval.x, interval.y, skipSurface, candidateTravel, candidateHit)) {{
      if (part < 0 || candidateTravel < travel || (candidateTravel == travel && candidate > part)) {{
        travel = candidateTravel; hit = candidateHit; part = candidate;
      }}
    }}
  }}
""")
    source.append(f"""
  if (part < 0) return false;
  // Separately marched parts can meet on one union surface. Choose its part
  // as mapSurface does: smallest absolute value, later parts winning ties.
  vec3 p = origin + travel * ray;
  float distance = abs(hit.x);
  int chosen = part;
  for (int i = 0; i < {len(compiled.parts)}; ++i) {{
    if (i == part || any(lessThan(p, partBoxLo[i])) || any(greaterThan(p, partBoxHi[i]))) continue;
    vec3 h = mapHitSurface(p, i);
    if (abs(h.x) < distance || (abs(h.x) == distance && i > chosen)) {{
      hit = h; distance = abs(h.x); chosen = i;
    }}
  }}
  part = chosen;
  return true; }}""")
    return "\n".join(source)
