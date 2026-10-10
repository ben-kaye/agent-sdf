"""Conservative value pruning inside field expressions."""

from math import isfinite

import numpy as np

from .bounds import AABB, _affine, _analytic
from .bvh import _build


def _certificate(field):
    """Bound positive field values below by scaled distance to an enclosing box."""
    kind, data = field.kind, field.data
    if kind in {
        "circle",
        "sphere",
        "rectangle",
        "box",
        "cylinder",
        "torus",
        "capsule",
        "tube",
        "polygon",
        "curve_profile",
    }:
        result = (_analytic(field), 1.0)
    elif kind in {"custom", "warp", "variable_extrude"}:
        # These lowered expressions are explicitly clipped with max(d, sdBox).
        result = (data.bounds, 1.0)
    elif kind in {"material", "feature"}:
        return _certificate(field.children[0])
    elif kind in {"min", "max", "subtract", "smooth_union"}:
        a = _certificate(field.children[0])
        if kind == "subtract":
            return a
        b = _certificate(field.children[1])
        if kind == "max":
            # An intersection's enclosing box alone does not certify a
            # Euclidean distance bound. Either operand's certificate does.
            available = [item for item in (a, b) if item is not None]
            return min(available, key=lambda item: np.prod(item[0].extents)) if available else None
        if a is None or b is None:
            return None
        scale = min(a[1], b[1])
        box = a[0].union(b[0])
        if kind == "smooth_union":
            box = box.expanded(data.radius / (4 * scale))
        result = (box, scale)
    elif kind in {"offset", "shell"}:
        child = _certificate(field.children[0])
        if child is None:
            return None
        amount = max(0.0, data.amount) if kind == "offset" else data.thickness / 2
        result = (child[0].expanded(amount / child[1]), child[1])
    elif kind == "transform":
        child = _certificate(field.children[0])
        if child is None:
            return None
        transform = data.transform
        stretch = float(np.linalg.norm(np.asarray(transform.matrix), 2))
        inverse = np.asarray(transform.inverse_matrix, dtype=np.float32)
        if np.linalg.cond(inverse.astype(float)) > 1e4:
            return None
        box = _affine(child[0], transform.matrix, transform.offset)
        condition = stretch * float(np.linalg.norm(inverse.astype(float), 2))
        box = box.expanded(1e-5 * (1 + max(map(abs, box.minimum + box.maximum))) * condition)
        result = (
            box,
            child[1] * transform.distance_scale / stretch,
        )
    elif kind == "linear_pattern":
        child = _certificate(field.children[0])
        if child is None:
            return None
        displacement = tuple((data.count - 1) * value for value in data.step)
        box = AABB(
            tuple(a + min(0.0, d) for a, d in zip(child[0].minimum, displacement, strict=True)),
            tuple(b + max(0.0, d) for b, d in zip(child[0].maximum, displacement, strict=True)),
        )
        result = (box, child[1])
    elif kind == "radial_pattern":
        child = _certificate(field.children[0])
        if child is None:
            return None
        a, b = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[data.axis]
        radius = np.hypot(
            max(abs(child[0].minimum[a]), abs(child[0].maximum[a])),
            max(abs(child[0].minimum[b]), abs(child[0].maximum[b])),
        )
        low, high = list(child[0].minimum), list(child[0].maximum)
        low[a] = low[b] = -radius
        high[a] = high[b] = radius
        result = (AABB(low, high), child[1])
    elif kind in {"extrude", "revolve"}:
        child = _certificate(field.children[0])
        if child is None:
            return None
        box, scale = child
        if kind == "extrude":
            box = AABB(
                (*box.minimum, data.offset - data.height / 2),
                (*box.maximum, data.offset + data.height / 2),
            )
            scale = min(scale, 1.0)
        else:
            radius = max(0.0, box.maximum[0])
            box = AABB((-radius, -radius, box.minimum[1]), (radius, radius, box.maximum[1]))
        result = (box, scale)
    else:
        return None
    box, scale = result
    if box.is_empty or not all(isfinite(v) for v in box.minimum + box.maximum):
        return None
    return (box, scale) if isfinite(scale) and scale > 0 else None


def _rounded_box(box):
    # Enclose float32 point maps, trigonometry, and center/half box clipping.
    box = box.expanded(1e-5 * (1 + max(map(abs, box.minimum + box.maximum))))
    return AABB(
        tuple(float(np.nextafter(np.float32(v), -np.float32(np.inf))) for v in box.minimum),
        tuple(float(np.nextafter(np.float32(v), np.float32(np.inf))) for v in box.maximum),
    )


class _ValueHierarchy:
    def __init__(self, functions, prefix="field"):
        self.functions = functions
        self.count = 0
        self.prefix = prefix

    def lower(self, entries, dim, *, later_wins=True):
        """Return exact union evaluation, including original-order label ties.

        Entries contain (expression, certificate). Uncertified operands are
        evaluated normally, and CSG operands remain indivisible expressions.
        """
        from .glsl import _f, _v

        certified = [(i, expr, cert) for i, (expr, cert) in enumerate(entries) if cert is not None]
        if len(certified) < 4:
            return None
        indexed = {i: (expr, cert[1]) for i, expr, cert in certified}
        boxes = {i: _rounded_box(cert[0]) for i, _, cert in certified}
        tree = _build(
            [
                (
                    i,
                    box,
                    tuple(a / 2 + b / 2 for a, b in zip(box.minimum, box.maximum, strict=True)),
                )
                for i, box in boxes.items()
                if i != 0
            ],
            dim,
            4,
        )
        order_test = ">" if later_wins else "<"

        def evaluate(index, expression):
            return [
                f"vec3 h = {expression};",
                f"if (h.x < best.x || (h.x == best.x && {_f(index)} {order_test} best.w)) best = vec4(h, {_f(index)});",
            ]

        def emit(node):
            children = [emit(child) for child in node.children]
            name = f"{self.prefix}Hierarchy{self.count}"
            self.count += 1
            indices = []

            def collect(current):
                indices.extend(current.fields)
                for child in current.children:
                    collect(child)

            collect(node)
            # Lower-bound scales also round down; geometry boxes round out.
            scale = float(
                np.nextafter(np.float32(min(indexed[i][1] for i in indices)), np.float32(0))
            )
            lines = [
                f"vec{dim} lo = {_v(node.bounds.minimum)}, hi = {_v(node.bounds.maximum)};",
                f"vec{dim} gap = max(max(lo - p, p - hi), 0.0);",
                f"if (any(greaterThan(gap, vec{dim}(0.0))) && length(gap) * {_f(scale)} > best.x + 1e-5 * (1.0 + abs(best.x))) return best;",
            ]
            if children:
                # Near-first traversal quickly establishes a useful upper bound.
                centers = [
                    tuple(
                        a / 2 + b / 2
                        for a, b in zip(child.bounds.minimum, child.bounds.maximum, strict=True)
                    )
                    for child in node.children
                ]
                lines.extend(
                    [
                        f"vec{dim} leftGap = p - {_v(centers[0])}, rightGap = p - {_v(centers[1])};",
                        "if (dot(leftGap, leftGap) <= dot(rightGap, rightGap)) {",
                        f"best = {children[0]}(p, best); best = {children[1]}(p, best);",
                        "} else {",
                        f"best = {children[1]}(p, best); best = {children[0]}(p, best);",
                        "}",
                    ]
                )
            else:
                for index in node.fields:
                    lines.extend(["{", *evaluate(index, indexed[index][0]), "}"])
            lines.append("return best;")
            self.functions.append(
                f"vec4 {name}(vec{dim} p, vec4 best) {{\n" + "\n".join(lines) + "\n}"
            )
            return name

        root = emit(tree)
        lines = [f"vec4 best = vec4({entries[0][0]}, 0.0);"]
        for index, (expr, cert) in enumerate(entries[1:], 1):
            if cert is None:
                lines.extend(["{", *evaluate(index, expr), "}"])
        lines.extend([f"best = {root}(p, best);", "return best.xyz;"])
        return lines
