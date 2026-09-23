"""Load SDF scenes and launch their WebGL2 viewer."""

import runpy
import sys
from pathlib import Path

from .dsl import Model, Shape
from .materials import scene_materials


def load_scene(source):
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError(f"scene file does not exist: {source}")
    sys.path.insert(0, str(source.parent))
    try:
        namespace = runpy.run_path(str(source))
    finally:
        sys.path.pop(0)
    scene = namespace.get("SCENE")
    model = namespace.get("MODEL")
    if scene is None and isinstance(model, Model):
        scene = model.scene()
    if not isinstance(scene, Shape) or scene.dim != 3:
        raise ValueError("scene must define a 3D SCENE or MODEL")
    return (
        source,
        scene,
        scene_materials(model if isinstance(model, Model) else None, namespace.get("MATERIALS")),
    )


def main():
    from .glsl_viewer import main as run_glsl_viewer

    run_glsl_viewer()


if __name__ == "__main__":
    main()
