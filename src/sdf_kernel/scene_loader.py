"""Load a Python scene for the viewer and renderers."""

import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

from .dsl import Shape
from .materials import scene_materials
from .model import Model


@dataclass(frozen=True)
class LoadedScene:
    source: Path
    scene: Shape
    model: Model | None
    materials: dict
    views: object


def load_scene(source: str | Path) -> LoadedScene:
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError(f"scene file does not exist: {source}")

    sys.path.insert(0, str(source.parent))
    try:
        namespace = runpy.run_path(str(source))
    finally:
        sys.path.pop(0)

    model = namespace.get("MODEL")
    model = model if isinstance(model, Model) else None
    scene = namespace.get("SCENE")
    if scene is None and model is not None:
        scene = model.scene()
    if not isinstance(scene, Shape) or scene.dim != 3:
        raise ValueError(f"{source} must define a 3D SCENE or MODEL")

    materials = scene_materials(model, namespace.get("MATERIALS"))
    return LoadedScene(source, scene, model, materials, namespace.get("VIEWS"))
