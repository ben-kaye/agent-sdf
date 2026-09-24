"""Named points on an oriented sketch plane."""

from dataclasses import dataclass, field

from .plane import Plane, _local_point


@dataclass(frozen=True)
class AnchorPlane(Plane):
    """Right-handed local XY plane with named points stored in local coordinates."""

    _points: dict[str, tuple[float, float]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def define_point(self, name, point, *, space="local"):
        """Name a local (u, v) point or a coplanar world (x, y, z) point."""
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("point name must be a Python identifier")
        if name in self._points:
            raise ValueError(f"point already exists: {name}")
        if space == "local":
            local = _local_point(point)
        elif space == "world":
            local = self.world_to_local(point)
        else:
            raise ValueError("space must be local or world")
        self._points[name] = local
        return self.point(name, space=space)

    def point(self, name, *, space="local"):
        """Get a named point in plane-local or world coordinates."""
        local = self._points[name]
        if space == "local":
            return local
        if space == "world":
            return self.local_to_world(local)
        raise ValueError("space must be local or world")

    def points(self, *, space="local"):
        """Return a copy of the named points in the requested coordinates."""
        return {name: self.point(name, space=space) for name in self._points}
