"""Load a Python scene for the viewer and renderers."""

import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

from .dsl import Shape
from .materials import scene_materials
from .model import Assembly, CompiledScene, Instance, Part, Scene, compile_scene


@dataclass(frozen=True)
class LoadedScene:
    source: Path
    scene: Shape
    object: Shape | Part | Assembly | Scene | Instance | CompiledScene
    compiled: CompiledScene
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

    object = namespace.get("SCENE")
    if object is None:
        model = namespace.get("MODEL")
        if isinstance(model, (Part, Assembly, Instance)):
            object = model
    if object is None:
        raise ValueError(f"{source} must define a 3D SCENE or MODEL")

    compiled = compile_scene(object)
    if compiled.shape.dim != 3:
        raise ValueError(f"{source} must define a 3D SCENE or MODEL")
    materials = scene_materials(compiled.materials, namespace.get("MATERIALS"))
    compiled = CompiledScene(compiled.shape, materials, compiled.parts)
    return LoadedScene(source, compiled.shape, object, compiled, materials, namespace.get("VIEWS"))
