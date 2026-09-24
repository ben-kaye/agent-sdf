"""Named CSG history and material definitions."""

from __future__ import annotations

from .dsl import Shape, _positive, max_, min_
from .materials import PALETTE, finish
from .transform import Transform


class Model:
    """Named, ordered CSG history. Every apply step depends on earlier fields."""

    def __init__(self):
        self._shapes: dict[str, Shape] = {}
        self._history: list[dict] = []
        self._current: str | None = None
        self._materials: dict[str, dict] = {}

    def material(
        self,
        name: str,
        *,
        base_color: str,
        metallic: float = 0.0,
        roughness: float = 0.8,
        opacity: float = 1.0,
    ) -> int:
        """Define a named finish and return its ID for Shape geometry."""
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("material name must be a Python identifier")
        if name in self._materials:
            raise ValueError(f"material already exists: {name}")
        index = len(PALETTE) + len(self._materials)
        if index > 4095:
            raise ValueError("model cannot define more than 4087 materials")
        self._materials[name] = {"id": index, **finish(base_color, metallic, roughness, opacity)}
        return index

    def materials(self) -> dict[str, dict]:
        """Return JSON-compatible material definitions."""
        return {name: definition.copy() for name, definition in self._materials.items()}

    def _check_name(self, name: str) -> None:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("feature name must be a Python identifier")
        if name in self._shapes:
            raise ValueError(f"feature already exists: {name}")

    def define(self, name: str, shape: Shape) -> Shape:
        """Register an independent field without changing the active result."""
        self._check_name(name)
        if not isinstance(shape, Shape):
            raise ValueError("feature must be a Shape")
        result = shape.named(name)
        self._shapes[name] = result
        self._history.append({"name": name, "op": "define", "depends_on": []})
        return result

    def start(self, name: str, shape: Shape) -> Shape:
        """Register the initial field and make it the active result."""
        if not isinstance(shape, Shape) or shape.dim != 3:
            raise ValueError("model result must be 3D")
        result = self.define(name, shape)
        self._current = name
        return result

    def transform(self, name: str, source: str, transform: Transform) -> Shape:
        """Define a transformed feature from a previously named field."""
        self._check_name(name)
        if source not in self._shapes:
            raise ValueError("source must name a previously defined field")
        result = self._shapes[source].transform(transform).named(name)
        self._shapes[name] = result
        self._history.append({"name": name, "op": "transform", "depends_on": [source]})
        return result

    def place(
        self,
        name: str,
        source: str,
        translation=None,
        *,
        rotation_matrix=None,
        axis_angle=None,
        quaternion=None,
        rpy=None,
    ) -> Shape:
        """Define a placed 3D feature from a previously named solid."""
        self._check_name(name)
        if source not in self._shapes:
            raise ValueError("source must name a previously defined field")
        result = self._shapes[source].place(
            translation,
            rotation_matrix=rotation_matrix,
            axis_angle=axis_angle,
            quaternion=quaternion,
            rpy=rpy,
        )
        result = result.named(name)
        self._shapes[name] = result
        self._history.append({"name": name, "op": "place", "depends_on": [source]})
        return result

    def apply(
        self, name: str, op: str, tool: str, *, base: str | None = None, radius: float | None = None
    ) -> Shape:
        """Apply a named field to the active result (or an explicit earlier base)."""
        self._check_name(name)
        base = self._current if base is None else base
        if base not in self._shapes or tool not in self._shapes:
            raise ValueError("base and tool must name previously defined fields")
        if op not in ("min", "max", "subtract", "smooth_union"):
            raise ValueError("op must be min, max, subtract, or smooth_union")
        if op != "smooth_union" and radius is not None:
            raise ValueError("radius is only valid for smooth_union")
        a, b = self._shapes[base], self._shapes[tool]
        if a.dim != b.dim:
            raise ValueError("base and tool must have the same dimension")
        if a.dim != 3:
            raise ValueError("model result must be 3D")
        if op == "min":
            result = min_(a, b)
        elif op == "max":
            result = max_(a, b)
        elif op == "subtract":
            result = a - b
        else:
            if radius is None:
                raise ValueError("smooth_union requires radius")
            result = a.smooth_union(b, _positive(radius))
        self._shapes[name] = result
        self._history.append(
            {
                "name": name,
                "op": op,
                "depends_on": [base, tool],
                **({"radius": radius} if op == "smooth_union" else {}),
            }
        )
        self._current = name
        return result

    def scene(self, name: str | None = None) -> Shape:
        key = self._current if name is None else name
        if key not in self._shapes or self._shapes[key].dim != 3:
            raise ValueError("scene must name a 3D model field")
        return self._shapes[key]

    def history(self) -> list[dict]:
        """JSON-compatible feature order and direct dependencies."""
        return [dict(step, depends_on=list(step["depends_on"])) for step in self._history]

    def ray_cast(self, origin, direction, **kwargs):
        from .query import ray_cast

        return ray_cast(self, origin, direction, **kwargs)

    def pick(self, x, y, width, height, **kwargs):
        from .query import pick

        return pick(self, x, y, width, height, **kwargs)

    def section(self, origin, normal, bounds, **kwargs):
        from .section import section

        return section(self, origin, normal, bounds, **kwargs)
