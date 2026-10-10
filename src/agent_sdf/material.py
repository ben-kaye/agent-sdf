"""Scene finishes shared by the still renderer and browser viewer."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import numpy as np

PALETTE = (
    (220, 219, 211),
    (220, 174, 112),
    (105, 167, 201),
    (178, 140, 193),
    (130, 189, 150),
    (35, 39, 44),
    (48, 87, 112),
    (190, 52, 45),
    (178, 184, 187),
)


@dataclass(frozen=True)
class Material:
    """A reusable named finish, independent of renderer material IDs."""

    name: str
    base_color: str
    metallic: float = 0.0
    roughness: float = 0.8
    opacity: float = 1.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("material name must be a Python identifier")
        for key, value in finish(
            self.base_color, self.metallic, self.roughness, self.opacity
        ).items():
            object.__setattr__(self, key, value)

    def finish(self):
        return finish(self.base_color, self.metallic, self.roughness, self.opacity)


def validate_material_id(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4095:
        raise ValueError("material id must be an integer from 0 to 4095")
    return value


def finish(base_color, metallic=0.0, roughness=0.8, opacity=1.0):
    """Validate and normalize one JSON-compatible material finish."""
    if not isinstance(base_color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", base_color):
        raise ValueError("material base_color must be a #RRGGBB hex color")
    values = []
    for label, value in (("metallic", metallic), ("roughness", roughness), ("opacity", opacity)):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"material {label} must be between 0 and 1") from exc
        if not isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"material {label} must be between 0 and 1")
        values.append(number)
    return {
        "base_color": base_color.lower(),
        "metallic": values[0],
        "roughness": values[1],
        "opacity": values[2],
    }


def normalize_materials(materials):
    if materials is None:
        return {}
    if not isinstance(materials, Mapping):
        raise ValueError("materials must map names to finish definitions")
    result = {}
    used_ids = set()
    for name, definition in materials.items():
        if not isinstance(name, str) or not all(part.isidentifier() for part in name.split("/")):
            raise ValueError("material name must contain identifier path segments")
        if not isinstance(definition, Mapping):
            raise ValueError(f"material {name} must be a finish definition")
        index = validate_material_id(definition.get("id"))
        if index in used_ids:
            raise ValueError(f"material id {index} is already assigned")
        used_ids.add(index)
        result[name] = {
            "id": index,
            **finish(
                definition.get("base_color"),
                definition.get("metallic", 0.0),
                definition.get("roughness", 0.8),
                definition.get("opacity", 1.0),
            ),
        }
    return result


def material_arrays(materials):
    definitions = normalize_materials(materials)
    count = max((len(PALETTE), *(item["id"] + 1 for item in definitions.values())))
    colors = [PALETTE[i % len(PALETTE)] for i in range(count)]
    metallic = [0.0] * count
    roughness = [0.8] * count
    opacity = [1.0] * count
    for item in definitions.values():
        index = item["id"]
        color = item["base_color"][1:]
        colors[index] = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
        metallic[index] = item["metallic"]
        roughness[index] = item["roughness"]
        opacity[index] = item["opacity"]
    return (
        np.asarray(colors, dtype=np.float32),
        np.asarray(metallic, dtype=np.float32),
        np.asarray(roughness, dtype=np.float32),
        np.asarray(opacity, dtype=np.float32),
    )


def scene_materials(base=None, external=None):
    result = normalize_materials(base)
    result.update(normalize_materials(external))
    return normalize_materials(result)
