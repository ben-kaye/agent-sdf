"""Spatial hierarchy for independently bounded union components."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from typing import TYPE_CHECKING

import numpy as np

from .bounds import AABB

if TYPE_CHECKING:
    from .field import Field


@dataclass(frozen=True)
class BVHNode:
    bounds: AABB
    children: tuple[BVHNode, ...] = ()
    fields: tuple[Field, ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children


def _union_components(field: Field) -> list[Field]:
    """Flatten unions through point maps and labels, preserving other CSG."""
    from .field import Field

    distributive = {"transform", "slice", "warp", "material", "feature"}
    components = []
    pending = [(field, ())]
    while pending:
        current, wrappers = pending.pop()
        if current.kind == "min":
            pending.extend((child, wrappers) for child in reversed(current.children))
        elif current.kind in distributive:
            pending.append((current.children[0], (*wrappers, current)))
        else:
            for wrapper in reversed(wrappers):
                current = Field(wrapper.kind, wrapper.dim, wrapper.data, (current,))
            components.append(current)
    return components


def _certified_box(field: Field):
    """Return a box when distance to it is a lower bound outside it.

    This deliberately excludes custom fields and distance estimates: a bound
    on their zero set does not bound their field value at a query point.
    """
    if field.kind in {"sphere", "box", "cylinder", "torus", "capsule", "tube"}:
        pass
    elif field.kind in {"linear_pattern", "radial_pattern", "material", "feature"}:
        if _certified_box(field.children[0]) is None:
            return None
    elif field.kind == "transform":
        transform = field.data.transform
        matrix = getattr(transform, "matrix", None)
        if matrix is None or _certified_box(field.children[0]) is None:
            return None
        singular_values = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
        if not np.allclose(singular_values, singular_values[0], rtol=1e-10, atol=0):
            return None
    else:
        return None
    try:
        box = field.bounds()
    except ValueError:
        return None
    if box.is_empty or not all(isfinite(x) for x in box.minimum + box.maximum):
        return None
    return box


def _enclose(boxes: list[AABB]) -> AABB:
    result = boxes[0]
    for box in boxes[1:]:
        result = result.union(box)
    return result


def _build(components, dim, leaf_size):
    if not components:
        return BVHNode(AABB((inf,) * dim, (-inf,) * dim))
    box = _enclose([item[1] for item in components])
    if len(components) <= leaf_size:
        return BVHNode(box, fields=tuple(item[0] for item in components))
    spread = [
        max(item[2][axis] for item in components) - min(item[2][axis] for item in components)
        for axis in range(dim)
    ]
    axis = max(range(dim), key=spread.__getitem__)
    ordered = sorted(components, key=lambda item: item[2][axis])
    middle = len(ordered) // 2
    return BVHNode(
        box,
        children=(
            _build(ordered[:middle], dim, leaf_size),
            _build(ordered[middle:], dim, leaf_size),
        ),
    )


def build_bvh(
    field: Field,
    scene_bounds: AABB | tuple | None = None,
    *,
    tolerance: float = 0.01,
    max_cells: int = 65536,
    leaf_size: int = 1,
) -> BVHNode:
    """Group union components by median center along their widest spread."""
    if isinstance(leaf_size, bool) or not isinstance(leaf_size, int) or leaf_size < 1:
        raise ValueError("leaf_size must be a positive integer")

    components = []
    for component in _union_components(field):
        box = component.bounds(scene_bounds, tolerance=tolerance, max_cells=max_cells)
        if box.is_empty:
            continue
        center = tuple(lo / 2 + hi / 2 for lo, hi in zip(box.minimum, box.maximum, strict=True))
        if not all(isfinite(value) for value in center):
            raise ValueError("BVH components must have finite bounds")
        components.append((component, box, center))

    return _build(components, field.dim, leaf_size)


def compile_bvh_sample(field: Field, reference_point=None):
    """Return a NumPy sampler that prunes certified union components."""
    import numpy as np

    components = _union_components(field)
    certified = [
        (i, component, _certified_box(component)) for i, component in enumerate(components)
    ]
    certified = [(i, component, box) for i, component, box in certified if box is not None]
    if len(certified) < 4:
        return lambda points, active=None: field.sample(points)

    indexed = {id(component): i for i, component, _ in certified if i != 0}
    entries = []
    for i, component, box in certified:
        if i == 0:
            continue
        center = tuple((a + b) / 2 for a, b in zip(box.minimum, box.maximum, strict=True))
        entries.append((component, box, center))
    tree = _build(entries, field.dim, 1)
    indexed_ids = set(indexed.values())
    ordinary = [
        (i, component) for i, component in enumerate(components[1:], 1) if i not in indexed_ids
    ]

    def sample(points, active=None):
        p = np.asarray(points)
        if p.shape[-1] != field.dim:
            raise ValueError(f"expected points with {field.dim} coordinates")
        flat = p.reshape(-1, field.dim)
        distance, material = components[0].sample(flat)
        best = np.array(distance, copy=True)
        materials = np.array(material, copy=True)
        order = np.zeros(flat.shape[0], dtype=np.int32)

        def evaluate(index, component, selected):
            if not selected.size:
                return
            d, m = component.sample(flat[selected])
            improve = (d < best[selected]) | ((d == best[selected]) & (index > order[selected]))
            selected = selected[improve]
            best[selected] = d[improve]
            materials[selected] = m[improve]
            order[selected] = index

        all_points = np.arange(flat.shape[0])
        for index, component in ordinary:
            evaluate(index, component, all_points)

        def visit(node, selected):
            if not selected.size:
                return
            lo = np.asarray(node.bounds.minimum)
            hi = np.asarray(node.bounds.maximum)
            q = flat[selected]
            outside = np.any((q < lo) | (q > hi), axis=-1)
            gap = np.maximum(np.maximum(lo - q, q - hi), 0)
            bound = np.linalg.norm(gap, axis=-1)
            selected = selected[~outside | (bound <= best[selected])]
            if not selected.size:
                return
            if node.children:
                for child in node.children:
                    visit(child, selected)
            else:
                for component in node.fields:
                    evaluate(indexed[id(component)], component, selected)

        visit(tree, all_points)
        return best.reshape(p.shape[:-1]), materials.reshape(p.shape[:-1])

    return sample
