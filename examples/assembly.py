"""A small fixture built from repeated parts and nested placements."""

from sdf_kernel import Assembly, Material, Part, Placement, Scene, box, cylinder

steel = Material("steel", base_color="#9ca8b0", metallic=0.8, roughness=0.3)
base = Part(box(1.6, 1.6, 0.1), material=steel, datums={"center": (0, 0, 0)})
pin = Part(cylinder(0.1, 0.5), material=steel, datums={"tip": (0, 0, 0.25)})

pair = Assembly()
pair.add("left", pin.at(Placement((-0.5, 0, 0.3))))
pair.add("right", pin.at(Placement((0.5, 0, 0.3))))

MODEL = Assembly()
MODEL.add("base", base)
MODEL.add("front", pair.at(Placement((0, -0.5, 0))))
MODEL.add("back", pair.at(Placement((0, 0.5, 0))))

SCENE = Scene(fixture=MODEL)
FRONT_LEFT_TIP = SCENE.datum("fixture/front/left/tip")
