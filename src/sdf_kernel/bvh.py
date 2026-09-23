"""Spatial hierarchy for independently bounded union components."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .bounds import AABB, _analytic, _sublevel_growth

if TYPE_CHECKING:
    from .dsl import Shape


@dataclass(frozen=True)
class BVHNode:
    bounds: AABB
    children: tuple[BVHNode, ...] = ()
    shapes: tuple[Shape, ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children


def _union_components(shape: Shape) -> list[Shape]:
    """Flatten unions through affine and material wrappers, preserving other CSG."""
    from .dsl import Shape

    distributive = {"translate", "rotate", "scale", "transform", "material"}
    components = []
    pending = [(shape, ())]
    while pending:
        current, wrappers = pending.pop()
        if current.kind == "union":
            pending.extend((child, wrappers) for child in reversed(current.children))
        elif current.kind in distributive:
            pending.append((current.children[0], (*wrappers, current)))
        else:
            for wrapper in reversed(wrappers):
                current = Shape(wrapper.kind, wrapper.dim, wrapper.data, (current,))
            components.append(current)
    return components


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
        return BVHNode(box, shapes=tuple(item[0] for item in components))
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
    shape: Shape,
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
    for component in _union_components(shape):
        box = component.bounds(scene_bounds, tolerance=tolerance, max_cells=max_cells)
        if box.is_empty:
            continue
        center = tuple(lo / 2 + hi / 2 for lo, hi in zip(box.minimum, box.maximum, strict=True))
        if not all(isfinite(value) for value in center):
            raise ValueError("BVH components must have finite bounds")
        components.append((component, box, center))

    return _build(components, shape.dim, leaf_size)


def compile_bvh_sample(shape: Shape, reference_point=None):
    """Exact union sampling with tile-wide conditional subtree evaluation."""
    components = _union_components(shape)
    bounded, fallback = [], []
    ranks, growths = {}, {}
    for rank, component in enumerate(components):
        ranks[id(component)] = rank
        box = _analytic(component)
        growth = _sublevel_growth(component)
        if (
            box.is_empty
            or growth is None
            or not all(isfinite(x) for x in box.minimum + box.maximum)
            or not all(isfinite(x) and x > 0 for x in growth)
        ):
            fallback.append(component)
            continue
        growths[id(component)] = growth
        center = tuple(lo / 2 + hi / 2 for lo, hi in zip(box.minimum, box.maximum, strict=True))
        bounded.append((component, box, center))

    if not bounded or len(components) == 1:
        return lambda points, active=None: shape.sample(points)

    root = _build(bounded, shape.dim, 1)
    reference = (0.0,) * shape.dim if reference_point is None else tuple(reference_point)

    def prepare(node):
        children = tuple(
            prepare(child)
            for child in sorted(
                node.children,
                key=lambda child: sum(
                    max(lo - p, p - hi, 0.0) ** 2
                    for lo, hi, p in zip(
                        child.bounds.minimum, child.bounds.maximum, reference, strict=True
                    )
                ),
            )
        )
        growth = (
            growths[id(node.shapes[0])]
            if node.is_leaf
            else tuple(max(child[1][axis] for child in children) for axis in range(shape.dim))
        )
        return node, growth, children

    tree = prepare(root)

    def sample(points, active=None):
        points = jnp.asarray(points)
        active = jnp.ones(points.shape[:-1], dtype=bool) if active is None else active
        state = (
            jnp.full(points.shape[:-1], jnp.inf, dtype=points.dtype),
            jnp.zeros(points.shape[:-1], dtype=jnp.int32),
            jnp.full(points.shape[:-1], -1, dtype=jnp.int32),
        )

        def evaluate(component, state, mask):
            best, material, rank = state
            distance, sampled_material = component.sample(points)
            index = ranks[id(component)]
            chosen = mask & ((distance < best) | ((distance == best) & (index > rank)))
            return (
                jnp.where(mask, jnp.minimum(best, distance), best),
                jnp.where(chosen, sampled_material, material),
                jnp.where(chosen, index, rank),
            )

        for component in fallback:
            state = evaluate(component, state, active)

        def visit(entry, state, mask):
            node, growth, children = entry
            # Bounds of f <= t grow by at most t * growth on each axis.
            # Inside the zero-level box no negative-distance bound is assumed.
            lo, hi = jnp.asarray(node.bounds.minimum), jnp.asarray(node.bounds.maximum)
            magnitude = jnp.maximum(
                jnp.max(jnp.abs(points), axis=-1),
                jnp.maximum(jnp.max(jnp.abs(lo)), jnp.max(jnp.abs(hi))),
            )
            margin = 32 * jnp.finfo(points.dtype).eps * (1 + magnitude)
            gap = jnp.maximum(lo - points, points - hi) - margin[..., None]
            lower = jnp.max(gap / jnp.asarray(growth), axis=-1)
            lower = jnp.where(lower > 0, lower, -jnp.inf)
            needed = mask & ((lower <= state[0]) | jnp.isnan(state[0]))

            def descend(current):
                if node.is_leaf:
                    return evaluate(node.shapes[0], current, needed)
                for child in children:
                    current = visit(child, current, needed)
                return current

            # A scalar cond really skips field execution for a whole tile.
            return jax.lax.cond(jnp.any(needed), descend, lambda current: current, state)

        distance, material, _ = visit(tree, state, active)
        return distance, material

    return sample
