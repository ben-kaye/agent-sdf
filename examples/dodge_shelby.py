"""1987 Shelby Charger GLH-S with curved pressings and detailed body trim."""

import math

from sdf_kernel import Curve, Model, Shape, box, capsule, cylinder, min_, polygon, rectangle, torus

FRONT_AXLE = 1.25
REAR_AXLE = FRONT_AXLE - 2.451
WHEEL_Z = 0.293

SIDE_GLASS = ((-0.45, 0.945), (0.56, 0.945), (0.105, 1.230), (-0.45, 1.230))
QUARTER_PANEL = ((-1.71, 0.925), (-0.50, 0.925), (-0.51, 1.225), (-0.91, 1.225))


def block(x, y, z, sx, sy, sz, radius=0):
    return box(sx, sy, sz, radius).translate(x, y, z)


def xz_profile(points, depth, y=0):
    sketch = points if isinstance(points, Shape) else polygon(points)
    return sketch.extrude(depth).rotate(math.pi / 2, "x").translate(0, y, 0)


def curved_body_profile():
    return (
        Curve(-2.218, 0.33)
        .line_to(-2.218, 0.72)
        .quad_to((-2.17, 0.82), (-1.96, 0.86), steps=5)
        .quad_to((-1.590312, 0.895750), (-1.233750, 0.904500), steps=4)
        .quad_to((-0.877188, 0.913250), (-0.65, 0.916), steps=4)
        .line_to(0.75, 0.925)
        .quad_to((1.205313, 0.927938), (1.486250, 0.914000), steps=3)
        .quad_to((1.767188, 0.900062), (1.90, 0.87), steps=3)
        .quad_to((2.12, 0.79), (2.218, 0.66), steps=5)
        .line_to(2.218, 0.30)
        .line_to(1.92, 0.22)
        .line_to(-1.92, 0.22)
        .close()
    )


def curved_roof_profile():
    return (
        Curve(-2.04, 0.855)
        .quad_to((-2.00, 0.93), (-1.94, 0.945), steps=4)
        .quad_to((-1.678438, 1.038437), (-1.351250, 1.130000), steps=5)
        .quad_to((-1.024063, 1.221562), (-0.89, 1.225), steps=5)
        .quad_to((-0.80, 1.27), (-0.68, 1.268), steps=4)
        .line_to(0.10, 1.275)
        .quad_to((0.19, 1.27), (0.24, 1.247), steps=4)
        .line_to(0.76, 0.925)
        .close()
    )


def y_cylinder(x, y, z, radius, depth):
    return cylinder(radius, depth).rotate(math.pi / 2, "x").translate(x, y, z)


def x_cylinder(x, y, z, radius, depth):
    return cylinder(radius, depth).rotate(math.pi / 2, "y").translate(x, y, z)


def line(a, b, radius):
    return capsule(a, b, radius)


def side_panel(points, side, depth=0.018, protrusion=0):
    angle = side * 0.397
    expanded = tuple((x, z / math.cos(angle)) for x, z in points)
    return (
        polygon(expanded)
        .extrude(depth)
        .rotate(math.pi / 2 + angle, "x")
        .translate(0, side * (1.14 + protrusion), 0)
    )


def sloped_screen(low, high, depth, thickness, outward):
    dx, dz = high[0] - low[0], high[1] - low[1]
    angle = math.atan2(dx, dz)
    length = math.hypot(dx, dz)
    shift_x = outward * dz / length
    shift_z = abs(outward) * abs(dx) / length
    return (
        box(thickness, depth, length)
        .rotate(angle, "y")
        .translate((low[0] + high[0]) / 2 + shift_x, 0, (low[1] + high[1]) / 2 + shift_z)
    )


MODEL = Model()
paint = MODEL.material("gloss_black_paint", base_color="#252d32", metallic=0.42, roughness=0.20)
trim = MODEL.material("black_polymer", base_color="#11171b", metallic=0.05, roughness=0.61)
glass_id = MODEL.material("smoked_glass", base_color="#1d3542", metallic=0.13, roughness=0.13)
rubber = MODEL.material("gatorback_rubber", base_color="#17191b", metallic=0.0, roughness=0.87)
alloy = MODEL.material("centurion_alloy", base_color="#aeb4b5", metallic=0.74, roughness=0.28)
silver = MODEL.material("silver_graphics", base_color="#9ca4a7", metallic=0.42, roughness=0.38)
white = MODEL.material("sealed_beam_headlamp", base_color="#dce2dc", metallic=0.04, roughness=0.18)
red = MODEL.material("rear_lamp_red", base_color="#ba211c", metallic=0.0, roughness=0.24)
amber = MODEL.material("front_marker_amber", base_color="#db6a23", metallic=0.0, roughness=0.28)


def attach(name, shape, material):
    MODEL.define(name, shape.with_material(material))
    MODEL.apply(name + "_attached", "min", name)


# Side silhouette is an extruded sketch. A section loft gives the body its
# measured changing width, and four cylinders open the wheel arches.
width_stations = (
    (-2.218, 0, 0.5, 0.740, 1),
    (-1.97, 0, 0.5, 0.810, 1),
    (-1.40, 0, 0.5, 0.825, 1),
    (0.75, 0, 0.5, 0.815, 1),
    (1.73, 0, 0.5, 0.807, 1),
    (2.218, 0, 0.5, 0.690, 1),
)
body_envelope = rectangle(2, 3).loft(width_stations, axis="x")
body_shell = xz_profile(curved_body_profile(), 1.9) & body_envelope
arches = min_(*(y_cylinder(axle, 0, WHEEL_Z, 0.320, 1.9) for axle in (FRONT_AXLE, REAR_AXLE)))
MODEL.start("hatchback_body_and_wheel_openings", (body_shell - arches).with_material(paint))

roof_width = rectangle(6, 2).loft(
    ((0.85, 0, 0, 1, 0.76), (0.94, 0, 0, 1, 0.735), (1.15, 0, 0, 1, 0.66), (1.275, 0, 0, 1, 0.60)),
    axis="z",
)
attach("sloping_hatch_roof", xz_profile(curved_roof_profile(), 1.7) & roof_width, paint)

painted = [
    block(1.46, 0, 0.933, 0.56, 0.46, 0.044, 0.009),
    block(-1.95, 0, 0.885, 0.35, 1.52, 0.046, 0.008),
    block(-2.087, 0, 0.858, 0.17, 1.44, 0.09, 0.015),
    # The hood center swells gently into the offset scoop.
    xz_profile(
        Curve(0.93, 0.924)
        .quad_to((1.19, 0.977), (1.48, 0.990), steps=6)
        .quad_to((1.72, 0.981), (1.88, 0.912), steps=6)
        .close(),
        0.46,
    ),
    block(2.204, 0, 0.386, 0.072, 1.48, 0.15, 0.025),
    block(2.245, 0, 0.278, 0.12, 1.57, 0.048, 0.012),
    block(-2.235, 0, 0.267, 0.11, 1.45, 0.065, 0.014),
]
flare_profile = torus(0.331, 0.027).rotate(math.pi / 2, "x")
upper_flare = flare_profile & block(0, 0, 0.307, 0.74, 0.16, 0.646)
for axle in (FRONT_AXLE, REAR_AXLE):
    for side in (-1, 1):
        painted.append(upper_flare.translate(axle, side * 0.807, WHEEL_Z))
for side in (-1, 1):
    painted.extend(
        (
            block(-0.02, side * 0.805, 0.259, 1.52, 0.076, 0.072, 0.008),
            block(0.652, side * 0.83, 0.925, 0.236, 0.14, 0.092, 0.022),
            block(-2.025, side * 0.59, 0.864, 0.25, 0.065, 0.12, 0.012),
            block(-1.998, side * 0.58, 0.941, 0.36, 0.25, 0.026, 0.007),
        )
    )
attach("shelby_spoiler_hood_air_dam_and_flares", min_(*painted), paint)

# Shared wheel prototypes keep the GLSL scene compact.
tire_profile = y_cylinder(0, 0, 0, 0.293, 0.205) - y_cylinder(0, 0, 0, 0.194, 0.24)
barrel_profile = y_cylinder(0, 0, 0, 0.199, 0.022)
rim_profile = y_cylinder(0, 0, 0, 0.201, 0.024) - y_cylinder(0, 0, 0, 0.181, 0.04)
spoke_outline = (
    (0.035, -0.028),
    (0.120, -0.040),
    (0.185, -0.024),
    (0.183, 0.026),
    (0.122, 0.040),
    (0.035, 0.028),
)
spokes = xz_profile(spoke_outline, 0.018).rotate(0.25, "y").radial_pattern(5, "y")
cap_profile = y_cylinder(0, 0, 0, 0.052, 0.018)
tires, barrels, wheels = [], [], []
sidewalls = []
for axle in (FRONT_AXLE, REAR_AXLE):
    for side in (-1, 1):
        center = (axle, side * 0.733, WHEEL_Z)
        tires.append(tire_profile.translate(*center))
        sidewalls.append(
            torus(0.267, 0.026).rotate(math.pi / 2, "x").translate(axle, side * 0.838, WHEEL_Z)
        )
        barrels.append(barrel_profile.translate(axle, side * 0.839, WHEEL_Z))
        wheels.extend(
            (
                rim_profile.translate(axle, side * 0.846, WHEEL_Z),
                spokes.translate(axle, side * 0.854, WHEEL_Z),
                cap_profile.translate(axle, side * 0.861, WHEEL_Z),
            )
        )
attach("rounded_gatorback_tires", min_(*tires, *sidewalls), rubber)
attach("wheel_recesses", min_(*barrels), trim)
attach("five_spoke_centurion_wheels", min_(*wheels), alloy)

# The side panes lean inward with the roof. Separate outer and inset panels
# retain the painted B pillar and narrow black gasket.
inner_side = ((-0.43, 0.957), (0.535, 0.957), (0.095, 1.218), (-0.43, 1.218))
window_trim = [
    sloped_screen((0.76, 0.95), (0.22, 1.24), 1.36, 0.024, 0.005),
    sloped_screen((-1.92, 0.975), (-0.95, 1.215), 1.35, 0.024, -0.005),
    block(2.223, 0, 0.626, 0.028, 0.76, 0.082, 0.011),
    block(2.215, 0, 0.432, 0.036, 0.98, 0.15, 0.013),
    block(2.180, 0, 0.255, 0.17, 1.32, 0.042, 0.010),
    block(-2.226, 0, 0.484, 0.028, 1.40, 0.096, 0.011),
    block(1.465, 0, 0.956, 0.26, 0.306, 0.028, 0.006),
    block(1.48, 0.08, 0.970, 0.23, 0.27, 0.012, 0.003),
    block(2.305, 0, 0.353, 0.018, 1.05, 0.075, 0.009),
    block(-2.258, 0, 0.594, 0.026, 0.86, 0.24, 0.010),
]
glazing = [
    sloped_screen((0.723, 0.97), (0.248, 1.225), 1.28, 0.012, 0.026),
    sloped_screen((-1.859, 0.99), (-0.990, 1.205), 1.27, 0.012, -0.026),
    block(-0.29, 0, 1.275, 0.46, 0.64, 0.006, 0.002),
]
for side in (-1, 1):
    shoulder = (
        Curve(-1.92, side * 0.806, 0.786)
        .quad_to(
            (-1.465938, side * 0.823719, 0.810969),
            (-0.788750, side * 0.825125, 0.818875),
            steps=5,
        )
        .quad_to(
            (-0.111562, side * 0.826531, 0.826781),
            (0.56, side * 0.824, 0.824),
            steps=5,
        )
        .quad_to(
            (1.043438, side * 0.829250, 0.822469),
            (1.431250, side * 0.815750, 0.798625),
            steps=4,
        )
        .quad_to(
            (1.819062, side * 0.802250, 0.774781),
            (2.07, side * 0.770, 0.729),
            steps=4,
        )
    )
    window_trim.extend(
        (
            side_panel(SIDE_GLASS, side, protrusion=0.005),
            side_panel(QUARTER_PANEL, side, protrusion=0.005),
            block(2.203, side * 0.565, 0.685, 0.046, 0.306, 0.178, 0.012),
            block(2.212, side * 0.58, 0.427, 0.038, 0.28, 0.126, 0.012),
            block(-2.229, side * 0.563, 0.680, 0.032, 0.336, 0.166, 0.010),
            line((-0.46, side * 0.826, 0.25), (-0.46, side * 0.82, 0.915), 0.005),
            line((-1.97, side * 0.822, 0.72), (2.18, side * 0.79, 0.72), 0.004),
            block(0.635, side * 0.84, 0.927, 0.146, 0.09, 0.040, 0.008),
            block(-0.06, side * 0.815, 0.283, 1.42, 0.052, 0.052, 0.009),
            shoulder.tube(0.0045, tolerance=0.006),
            block(2.305, side * 0.468, 0.355, 0.024, 0.31, 0.086, 0.008),
            block(-2.258, side * 0.645, 0.604, 0.031, 0.17, 0.22, 0.007),
        )
    )
    glazing.extend(
        (
            side_panel(inner_side, side, protrusion=0.020),
            block(0.620, side * 0.900, 0.935, 0.120, 0.018, 0.07, 0.006),
        )
    )
attach("fascias_and_window_seals", min_(*window_trim), trim)
attach("windscreen_hatch_and_side_glass", min_(*glazing), glass_id)

headlamps, taillamps, markers, brightwork = [], [], [], []
for side in (-1, 1):
    headlamps.extend(
        (
            block(2.239, side * 0.566, 0.708, 0.019, 0.175, 0.121, 0.007),
            block(2.241, side * 0.327, 0.708, 0.019, 0.175, 0.121, 0.007),
        )
    )
    taillamps.extend(
        (
            block(-2.244, side * 0.555, 0.685, 0.016, 0.25, 0.13, 0.005),
            block(-2.276, side * 0.336, 0.685, 0.014, 0.173, 0.13, 0.004),
            block(-2.239, side * 0.72, 0.405, 0.020, 0.12, 0.04, 0.004),
        )
    )
    markers.extend(
        (
            block(2.229, side * 0.558, 0.420, 0.018, 0.182, 0.03, 0.004),
            block(1.88, side * 0.815, 0.450, 0.142, 0.016, 0.044, 0.004),
        )
    )
    brightwork.extend(
        (
            line((-1.93, side * 0.836, 0.605), (2.10, side * 0.811, 0.605), 0.005),
            block(-0.15, side * 0.837, 0.881, 0.164, 0.022, 0.02, 0.004),
            block(0.63, side * 0.857, 0.955, 0.058, 0.018, 0.024, 0.004),
            x_cylinder(-2.238, side * 0.51, 0.305, 0.039, 0.05)
            - x_cylinder(-2.238, side * 0.51, 0.305, 0.027, 0.08),
            block(2.255, side * 0.445, 0.786, 0.016, 0.015, 0.11, 0.003),
            block(-2.281, side * 0.443, 0.680, 0.014, 0.016, 0.145, 0.003),
        )
    )
for k in range(-3, 4):
    brightwork.append(line((2.248, k * 0.089, 0.594), (2.248, k * 0.089, 0.654), 0.003))
for z in (0.331, 0.36, 0.389):
    brightwork.append(line((2.323, -0.49, z), (2.323, 0.49, z), 0.003))
attach("square_sealed_beam_headlamps", min_(*headlamps), white)
attach("rear_tail_lamps", min_(*taillamps), red)
attach("amber_front_markers", min_(*markers), amber)
attach("silver_stripe_handles_and_exhaust", min_(*brightwork), silver)

GLYPH_PATHS = {
    "S": ((1, 1), (0, 1), (0, 0.5), (1, 0.5), (1, 0), (0, 0)),
    "H": ((0, 0), (0, 1), (0, 0.5), (1, 0.5), (1, 1), (1, 0)),
    "E": ((1, 1), (0, 1), (0, 0.5), (0.8, 0.5), (0, 0.5), (0, 0), (1, 0)),
    "L": ((0, 1), (0, 0), (1, 0)),
    "B": (
        (0, 0),
        (0, 1),
        (0.75, 1),
        (1, 0.75),
        (0.75, 0.5),
        (0, 0.5),
        (0.75, 0.5),
        (1, 0.25),
        (0.75, 0),
        (0, 0),
    ),
    "Y": ((0, 1), (0.5, 0.5), (1, 1), (0.5, 0.5), (0.5, 0)),
    "G": ((1, 1), (0, 1), (0, 0), (1, 0), (1, 0.5), (0.5, 0.5)),
}
graphics = []
for side in (-1, 1):
    for word, left_x, right_x, z0, height, width, surface in (
        ("SHELBY", 0.28, 0.74, 0.280, 0.072, 0.061, 0.845),
        ("GLHS", -1.55, -0.99, 0.995, 0.045, 0.042, 0.738),
    ):
        direction = -side
        for index, letter in enumerate(word):
            origin = (left_x if side < 0 else right_x) + direction * index * width * 1.25
            path = GLYPH_PATHS[letter]
            points = tuple(
                (origin + direction * u * width, side * surface, z0 + v * height) for u, v in path
            )
            curve = Curve(*points[0])
            for point in points[1:]:
                curve.line_to(*point)
            graphics.append(curve.tube(0.0035))
attach("shelby_and_glhs_graphics", min_(*graphics), silver)

SCENE = MODEL.scene()
