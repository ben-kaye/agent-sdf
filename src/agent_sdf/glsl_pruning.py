"""Conservative value pruning inside field expressions."""

import numpy as np

from .bounds import AABB
from .bvh import _build


def _rounded_box(box):
    # Enclose float32 point maps, trigonometry, and center/half box clipping.
    box = box.expanded(1e-5 * (1 + max(map(abs, box.minimum + box.maximum))))
    return AABB(
        tuple(float(np.nextafter(np.float32(v), -np.float32(np.inf))) for v in box.minimum),
        tuple(float(np.nextafter(np.float32(v), np.float32(np.inf))) for v in box.maximum),
    )


class _ValueHierarchy:
    def __init__(self, functions):
        self.functions = functions

    def lower(self, entries, dim):
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

        def emit(node):
            children = [emit(child) for child in node.children]
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
                # Call each child once: GLSL inlines calls, so branch-dependent
                # orderings duplicate every subtree per level.
                lines.extend(f"best = {child}(p, best);" for child in children)
            else:
                lines.extend(
                    f"best = pick(best, {indexed[index][0]}, {_f(index)});" for index in node.fields
                )
            lines.append("return best;")
            return self.functions.add(
                "fieldHierarchy", "vec4", f"vec{dim} p, vec4 best", "\n".join(lines)
            )

        root = emit(tree)
        lines = [f"vec4 best = vec4({entries[0][0]}, 0.0);"]
        for index, (expr, cert) in enumerate(entries[1:], 1):
            if cert is None:
                lines.append(f"best = pick(best, {expr}, {_f(index)});")
        lines.extend([f"best = {root}(p, best);", "return best.xyz;"])
        return lines
