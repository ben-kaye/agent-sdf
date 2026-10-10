"""Spatial hierarchy for independently bounded union components."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import inf, isfinite, pi
from typing import TYPE_CHECKING

import numpy as np

from .bounds import AABB, _affine, _analytic, _points_bounds, revolve_bounds

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


def _enclose(boxes: list[AABB]) -> AABB:
    result = boxes[0]
    for box in boxes[1:]:
        result = result.union(box)
    return result


def certificate(field, recurse=None):
    """Return (box, scale) with field values >= scale * distance to box outside it.

    recurse certifies children; it defaults to this value certificate.
    """
    recurse = recurse or certificate
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
    elif kind in {"loft", "field_morph", "profile_sweep"}:
        # Clipped to the distance of boxes enclosing the zero set.
        if data.bounds is None:
            return None
        result = (_enclose(list(data.bounds)), 1.0)
    elif kind in {"material", "feature"}:
        return recurse(field.children[0])
    elif kind == "slice":
        # Projection onto the plane never lengthens distances to the child box.
        child = recurse(field.children[0])
        if child is None:
            return None
        plane = data.plane
        corners = product(*zip(child[0].minimum, child[0].maximum, strict=True))
        offsets = [
            tuple(c - o for c, o in zip(corner, plane.origin, strict=True)) for corner in corners
        ]
        box = _points_bounds(
            tuple(
                sum(a * b for a, b in zip(p, axis, strict=True))
                for axis in (plane.x_direction, plane.y_direction)
            )
            for p in offsets
        )
        result = (box, child[1])
    elif kind in {"min", "max", "subtract", "smooth_union"}:
        a = recurse(field.children[0])
        if kind == "subtract":
            return a
        b = recurse(field.children[1])
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
        child = recurse(field.children[0])
        if child is None:
            return None
        amount = max(0.0, data.amount) if kind == "offset" else data.thickness / 2
        result = (child[0].expanded(amount / child[1]), child[1])
    elif kind == "transform":
        child = recurse(field.children[0])
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
        child = recurse(field.children[0])
        if child is None:
            return None
        displacement = tuple((data.count - 1) * value for value in data.step)
        box = AABB(
            tuple(a + min(0.0, d) for a, d in zip(child[0].minimum, displacement, strict=True)),
            tuple(b + max(0.0, d) for b, d in zip(child[0].maximum, displacement, strict=True)),
        )
        result = (box, child[1])
    elif kind == "radial_pattern":
        child = recurse(field.children[0])
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
        child = recurse(field.children[0])
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
            box = revolve_bounds(box, data.angle)
            if data.angle < 2 * pi:
                scale = min(scale, 1.0)
        result = (box, scale)
    else:
        return None
    box, scale = result
    if box.is_empty or not all(isfinite(v) for v in box.minimum + box.maximum):
        return None
    return (box, scale) if isfinite(scale) and scale > 0 else None


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
    certified = [(i, component, certificate(component)) for i, component in enumerate(components)]
    certified = [(i, component, cert) for i, component, cert in certified if cert is not None]
    if len(certified) < 4:
        return lambda points, active=None: field.sample(points)

    indexed = {id(component): i for i, component, _ in certified if i != 0}
    leaf_scales = {id(component): cert[1] for _, component, cert in certified}
    entries = []
    for i, component, (box, _) in certified:
        if i == 0:
            continue
        center = tuple((a + b) / 2 for a, b in zip(box.minimum, box.maximum, strict=True))
        entries.append((component, box, center))
    tree = _build(entries, field.dim, 1)
    scales = {}

    def collect(node):
        values = [collect(child) for child in node.children]
        values += [leaf_scales[id(component)] for component in node.fields]
        scales[id(node)] = min(values)
        return scales[id(node)]

    collect(tree)
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
            selected = selected[~outside | (bound * scales[id(node)] <= best[selected])]
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
