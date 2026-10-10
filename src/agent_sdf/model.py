"""Reusable parts, nested assemblies, and independently placed scene objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .field import Field, min_
from .material import PALETTE, Material, validate_material_id
from .nodes import Feature
from .nodes import Material as MaterialPayload
from .plane import AnchorPlane, Plane
from .transform import Placement, _vector3


def _name(value, label):
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{label} must be a Python identifier")
    return value


def _mapping(value, label):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _place_datum(value, placement):
    rotation = np.asarray(placement.rotation)

    def point(p):
        return tuple(float(v) for v in rotation @ p + placement.translation)

    if isinstance(value, Placement):
        return value.then(placement)
    if isinstance(value, Plane):
        plane_type = AnchorPlane if isinstance(value, AnchorPlane) else Plane
        result = plane_type(
            point(value.origin),
            tuple(rotation @ value.normal),
            x_direction=tuple(rotation @ value.x_direction),
        )
        if isinstance(value, AnchorPlane):
            object.__setattr__(result, "_points", MappingProxyType(value.points()))
        return result
    return point(value)


def _datums(values):
    result = {}
    for name, value in _mapping(values, "datums").items():
        _name(name, "datum name")
        if not isinstance(value, (Plane, Placement)):
            value = _vector3(value, "point datum")
        result[name] = _place_datum(value, Placement())
    return MappingProxyType(result)


class _Object:
    def to_field(self) -> Field:
        """Explicitly combine the placed part fields into one union field."""
        return compile_scene(self).field

    def bounds(self, scene_bounds=None, **kwargs):
        return self.to_field().bounds(scene_bounds, **kwargs)

    def extents(self, scene_bounds=None, **kwargs):
        return self.bounds(scene_bounds, **kwargs).extents

    def bvh(self, scene_bounds=None, **kwargs):
        return self.to_field().bvh(scene_bounds, **kwargs)

    def ray_cast(self, origin, direction, **kwargs):
        from .query import ray_cast

        return ray_cast(self, origin, direction, **kwargs)

    def pick(self, x, y, width, height, **kwargs):
        from .query import pick

        return pick(self, x, y, width, height, **kwargs)

    def slice(self, plane):
        return self.to_field().slice(plane)

    def section(self, plane, bounds, **kwargs):
        from .section import section

        return section(self, plane, bounds, **kwargs)


class _Placeable(_Object):
    def at(self, placement: Placement) -> Instance:
        """Create an occurrence without changing this definition."""
        return Instance(self, placement)


@dataclass(frozen=True, init=False, eq=False)
class Part(_Placeable):
    """One local geometry expression, its materials, and exported datums."""

    field: Field
    materials: Mapping[int, Material]
    datums: Mapping
    metadata: Mapping

    def __init__(self, field, *, material=None, materials=None, datums=None, metadata=None):
        if not isinstance(field, Field) or field.dim != 3:
            raise ValueError("a Part requires a 3D Field")
        definitions = _mapping(materials, "materials")
        if material is not None:
            if not isinstance(material, Material):
                raise ValueError("part material must be a Material")
            if definitions:
                raise ValueError("provide material or local materials, not both")
            field = field.with_material(0)
            definitions = {0: material}
        named_materials = {}
        for key, value in definitions.items():
            validate_material_id(key)
            if not isinstance(value, Material):
                raise ValueError("part materials must map local IDs to Material values")
            if value.name in named_materials and named_materials[value.name] != value:
                raise ValueError(f"part material name has conflicting definitions: {value.name}")
            named_materials[value.name] = value
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "materials", MappingProxyType(definitions))
        object.__setattr__(self, "datums", _datums(datums))
        object.__setattr__(self, "metadata", MappingProxyType(_mapping(metadata, "metadata")))

    def datum(self, name):
        """Read an exported datum in part-local coordinates."""
        return self.datums[name]


@dataclass(frozen=True, eq=False)
class Instance(_Placeable):
    """A part or assembly definition with a placement relative to its parent."""

    definition: Part | Assembly
    placement: Placement = Placement()

    def __post_init__(self):
        if not isinstance(self.definition, (Part, Assembly)):
            raise ValueError("an instance requires a Part or Assembly definition")
        if not isinstance(self.placement, Placement):
            raise ValueError("instance placement must be a Placement")

    def at(self, placement: Placement) -> Instance:
        """Apply an additional placement in parent coordinates."""
        return Instance(self.definition, self.placement.then(placement))

    @property
    def datums(self):
        return MappingProxyType(
            {
                name: _place_datum(value, self.placement)
                for name, value in self.definition.datums.items()
            }
        )

    def datum(self, path):
        """Resolve a datum into this instance's parent coordinates."""
        return _place_datum(self.definition.datum(path), self.placement)


class _Composition(_Object):
    def __init__(self, components=None, *, datums=None, metadata=None, **named_components):
        self._components: dict[str, Instance] = {}
        self._datums = _datums(datums)
        self._metadata = MappingProxyType(_mapping(metadata, "metadata"))
        for name, value in _mapping(components, "components").items():
            self.add(name, value)
        for name, value in named_components.items():
            self.add(name, value)

    @property
    def components(self):
        return MappingProxyType(self._components)

    @property
    def datums(self):
        return self._datums

    @property
    def metadata(self):
        return self._metadata

    def __getitem__(self, name):
        return self._components[name]

    def add(self, name, value) -> Instance:
        """Add a named component, preserving its definition and local placement."""
        _name(name, "component name")
        if name in self._components:
            raise ValueError(f"component already exists: {name}")
        instance = value if isinstance(value, Instance) else Instance(value)
        visited = set()

        def contains(definition):
            if definition is self:
                return True
            if not isinstance(definition, Assembly) or id(definition) in visited:
                return False
            visited.add(id(definition))
            return any(contains(child.definition) for child in definition.components.values())

        if contains(instance.definition):
            raise ValueError("assembly components cannot form a cycle")
        self._components[name] = instance
        return instance

    def datum(self, path):
        """Resolve an exported datum, optionally through a component/name path."""
        if not isinstance(path, str) or not path:
            raise ValueError("datum path must be a nonempty string")
        first, separator, rest = path.partition("/")
        if not separator:
            return self.datums[first]
        return self.components[first].datum(rest)


class Assembly(_Composition, _Placeable):
    """A reusable local composition of named Part and Assembly instances."""


class Scene(_Composition):
    """The world root of independently placed Part and Assembly instances."""


@dataclass(frozen=True)
class CompiledPart:
    path: tuple[str, ...]
    field: Field
    part: Part | None
    placement: Placement


@dataclass(frozen=True)
class CompiledScene:
    field: Field
    materials: dict
    parts: tuple[CompiledPart, ...]


def _prepare_features(field, prefix=()):
    """Lower nested feature scopes to leaf labels without losing constituent names."""
    cache = {}

    def visit(node, scope):
        key = (id(node), scope)
        if key in cache:
            return cache[key]
        if node.kind == "feature":
            result = visit(node.children[0], (*scope, node.data.name))
        elif node.children:
            result = Field(
                node.kind, node.dim, node.data, tuple(visit(c, scope) for c in node.children)
            )
        elif scope:
            result = Field("feature", node.dim, Feature("/".join(scope)), (node,))
        else:
            result = node
        cache[key] = result
        return result

    return visit(field, tuple(prefix))


def _remap_materials(field, mapping):
    cache = {}

    def material_id(value):
        return mapping.get(value, value % len(PALETTE))

    def remap_custom_material(original):
        from .math import floor, where

        result = original - floor(original / len(PALETTE)) * len(PALETTE)
        for local, global_id in mapping.items():
            result = where(original == local, global_id, result)
        return result

    def visit(node):
        key = id(node)
        if key in cache:
            return cache[key]
        data = node.data
        if node.kind == "material":
            data = MaterialPayload(material_id(data.material_id))
            children = node.children
        else:
            children = tuple(visit(c) for c in node.children)
        if node.kind == "custom":
            data = data._replace(material=remap_custom_material(data.material))
        result = Field(node.kind, node.dim, data, children)
        if not children and node.kind != "custom" and 0 in mapping:
            result = Field("material", node.dim, MaterialPayload(mapping[0]), (result,))
        cache[key] = result
        return result

    return visit(field)


def compile_scene(value) -> CompiledScene:
    """Resolve instance placements, material bindings, and qualified feature labels."""
    if isinstance(value, CompiledScene):
        return value
    if isinstance(value, Field):
        if value.dim != 3:
            raise ValueError("a scene requires 3D geometry")
        field = _prepare_features(value)
        return CompiledScene(field, {}, (CompiledPart((), field, None, Placement()),))
    if not isinstance(value, (Part, Assembly, Scene, Instance)):
        raise ValueError("expected a 3D Field, Part, Assembly, Instance, or Scene")
    parts = []
    materials = {}
    material_ids = {}
    active = set()
    prepared = {}

    def visit(obj, path, placement):
        if isinstance(obj, Instance):
            visit(obj.definition, path, obj.placement.then(placement))
        elif isinstance(obj, Part):
            if id(obj) not in prepared:
                mapping = {}
                for local, material in obj.materials.items():
                    if material not in material_ids:
                        index = len(PALETTE) + len(material_ids)
                        validate_material_id(index)
                        material_ids[material] = index
                        name = "/".join((*path, material.name))
                        materials[name] = {"id": index, **material.finish()}
                    mapping[local] = material_ids[material]
                prepared[id(obj)] = _remap_materials(obj.field, mapping)
            field = _prepare_features(prepared[id(obj)], path).place(placement)
            parts.append(CompiledPart(path, field, obj, placement))
        else:
            key = id(obj)
            if key in active:
                raise ValueError("assembly components cannot form a cycle")
            active.add(key)
            for name, instance in obj.components.items():
                visit(instance, (*path, name), placement)
            active.remove(key)

    visit(value, (), Placement())
    if not parts:
        raise ValueError("cannot evaluate a composition containing no parts")
    field = min_(*(part.field for part in parts))
    return CompiledScene(field, materials, tuple(parts))
