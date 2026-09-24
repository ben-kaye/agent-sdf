"""1969 Dodge Charger R/T coupe, in metres with the nose along +X and Z up."""

import math

from sdf_kernel import Model, box, capsule, cylinder, min_, polygon, rectangle, torus

WHEELBASE = 2.9718  # 117 in
AXLES = (-WHEELBASE / 2, WHEELBASE / 2)
WHEEL_Z = 0.348
ROOF_HEIGHT = 1.3513  # 53.2 in overall

STATIONS = (-2.642, -2.42, -1.88, -1.49, -1.03, -0.20, 0.65, 1.49, 2.30, 2.642)
WIDTHS = (0.76, 0.88, 0.94, 0.963, 0.87, 0.825, 0.84, 0.955, 0.90, 0.78)
TOPS = (0.956, 1.005, 1.035, 1.055, 1.045, 1.04, 1.047, 1.025, 0.955, 0.90)
BOTTOMS = (
    (-2.642, 0.39),
    (-2.25, 0.28),
    (-1.75, 0.205),
    (1.90, 0.205),
    (2.26, 0.34),
    (2.642, 0.46),
)
ROOF_OUTLINE = (
    (-1.82, 1.005),
    (-1.67, 1.075),
    (-1.115, 1.318),
    (-0.965, ROOF_HEIGHT),
    (-0.15, ROOF_HEIGHT),
    (0.04, 1.321),
    (0.70, 1.047),
    (0.63, 1.025),
    (-1.72, 1.005),
)
FRONT_SIDE_GLASS = ((-0.25, 1.073), (0.565, 1.073), (0.022, 1.316), (-0.25, 1.316))
REAR_SIDE_GLASS = ((-1.505, 1.074), (-0.302, 1.074), (-0.302, 1.316), (-1.03, 1.316))


def block(x, y, z, hx, hy, hz, radius=0):
    return box(2 * hx, 2 * hy, 2 * hz, min(radius, hx, hy, hz)).translate(x, y, z)


def xz_profile(points, depth, y=0):
    return polygon(points).extrude(depth).rotate(math.pi / 2, "x").translate(0, y, 0)


def y_cylinder(x, y, z, radius, half_depth):
    return cylinder(radius, 2 * half_depth).rotate(math.pi / 2, "x").translate(x, y, z)


def x_cylinder(x, y, z, radius, half_depth):
    return cylinder(radius, 2 * half_depth).rotate(math.pi / 2, "y").translate(x, y, z)


def line(a, b, radius):
    return capsule(a, b, radius)


def side_panel(points, side, depth=0.024):
    # The pane plane follows the inward taper of the roof pillars.
    angle = math.atan(0.65)
    raised = tuple(((x if side > 0 else -x), z / math.cos(angle)) for x, z in points)
    panel = xz_profile(raised, depth).rotate(angle, "x").translate(0, 1.473, 0)
    return panel if side > 0 else panel.rotate(math.pi, "z")


def roof_width_envelope(margin=0):
    return rectangle(6, 2).loft(
        (
            (1.0, 0, 0, 1, 0.79 - margin),
            (1.08, 0, 0, 1, 0.765 - margin),
            (1.32, 0, 0, 1, 0.61 - margin),
            (1.36, 0, 0, 1, 0.59 - margin),
        ),
        axis="z",
    )


def screen(points, width_margin=0):
    return xz_profile(points, 1.65) & roof_width_envelope(width_margin)


def front_screen(margin):
    low, high = 1.058 + margin, 1.319 - margin
    slope = (0.043 - 0.682) / (1.319 - 1.058)
    xlow = 0.682 + slope * (low - 1.058)
    xhigh = 0.682 + slope * (high - 1.058)
    return screen(
        ((xlow - 0.015, low), (xlow + 0.015, low), (xhigh + 0.015, high), (xhigh - 0.015, high)),
        margin,
    )


def rear_screen(margin):
    low, high = 1.048 + margin, 1.313 - margin
    slope = (-1.07 + 1.505) / (1.315 - 1.035)
    xlow = -1.505 + slope * (low - 1.035)
    xhigh = -1.505 + slope * (high - 1.035)
    profile = xz_profile(
        ((xlow - 0.014, low), (xlow + 0.014, low), (xhigh + 0.014, high), (xhigh - 0.014, high)),
        2 * (0.525 - margin),
    )
    return profile


MODEL = Model()
paint = MODEL.material("r_t_dark_green", base_color="#28544a", metallic=0.53, roughness=0.24)
glass_id = MODEL.material(
    "period_tinted_glass", base_color="#1f3437", metallic=0.10, roughness=0.12
)
black = MODEL.material("grille_and_seals", base_color="#171a19", metallic=0.05, roughness=0.68)
rubber = MODEL.material("bias_ply_rubber", base_color="#151615", metallic=0.0, roughness=0.92)
chrome = MODEL.material("bright_chrome", base_color="#d0d2cc", metallic=0.93, roughness=0.17)
red = MODEL.material("red_tail_lens", base_color="#a51817", metallic=0.02, roughness=0.21)
amber = MODEL.material("amber_marker_lens", base_color="#df7e22", metallic=0.01, roughness=0.25)


def attach(name, shape, material):
    MODEL.define(name, shape.with_material(material))
    MODEL.apply(name + "_attached", "min", name)


# A side profile and width loft form the coke-bottle body. Cylinders open
# the wheel arches before the visible wheel geometry is attached.
body_profile = xz_profile(tuple(zip(STATIONS, TOPS, strict=True)) + tuple(reversed(BOTTOMS)), 2.1)
width_envelope = rectangle(2, 3).loft(
    tuple((x, 0, 0.7, width, 1) for x, width in zip(STATIONS, WIDTHS, strict=True)),
    axis="x",
)
body = body_profile & width_envelope
arches = min_(*(y_cylinder(axle, 0, WHEEL_Z, 0.403, 1.05) for axle in AXLES))
MODEL.start("coke_bottle_body_and_wheel_openings", (body - arches).with_material(paint))

roof = xz_profile(ROOF_OUTLINE, 1.7) & roof_width_envelope()
recess = xz_profile(((-2, 1.045), (-1.489, 1.045), (-1.065, 1.318), (-2, 1.318)), 1.07)
attach("flying_buttress_roof", roof - recess, paint)

painted = []
flare = torus(0.418, 0.018).rotate(math.pi / 2, "x")
upper_flare = flare & block(0, 0, 0.2, 0.46, 0.06, 0.238)
for axle in AXLES:
    for side in (-1, 1):
        painted.append(upper_flare.translate(axle, side * 0.936, WHEEL_Z))
painted.extend(
    (
        block(1.30, 0, 1.026, 0.66, 0.335, 0.032, 0.025),
        block(-2.27, 0, 0.997, 0.34, 0.73, 0.018, 0.012),
    )
)
for side in (-1, 1):
    painted.append(block(-0.01, side * 0.807, 0.285, 1.04, 0.034, 0.055, 0.013))
attach("hood_fenders_and_sills", min_(*painted), paint)

tire = y_cylinder(0, 0, 0, 0.348, 0.132) - y_cylinder(0, 0, 0, 0.244, 0.15)
wheel_dark = y_cylinder(0, 0, 0, 0.240, 0.018)
wheel_ring = y_cylinder(0, 0, 0, 0.247, 0.013) - y_cylinder(0, 0, 0, 0.220, 0.021)
spokes = min_(
    *(
        line(
            (0.048 * math.cos(a), 0, 0.048 * math.sin(a)),
            (0.215 * math.cos(a), 0, 0.215 * math.sin(a)),
            0.023,
        )
        for a in (2 * math.pi * i / 5 + math.pi / 2 for i in range(5))
    )
)
cap = y_cylinder(0, 0, 0, 0.061, 0.012)
tires, dark_wheels, chrome_wheels = [], [], []
for axle in AXLES:
    for side in (-1, 1):
        tires.append(tire.translate(axle, side * 0.823, WHEEL_Z))
        dark_wheels.append(wheel_dark.translate(axle, side * 0.937, WHEEL_Z))
        cy = side * 0.954
        chrome_wheels.extend(
            (
                wheel_ring.translate(axle, cy, WHEEL_Z),
                spokes.translate(axle, cy, WHEEL_Z),
                cap.translate(axle, cy + side * 0.009, WHEEL_Z),
            )
        )
attach("four_period_tires", min_(*tires), rubber)
attach("magnum_wheel_insets", min_(*dark_wheels), black)
attach("magnum_five_spoke_wheels", min_(*chrome_wheels), chrome)

side_glass, side_seals = [], []
for side in (-1, 1):
    for profile in (FRONT_SIDE_GLASS, REAR_SIDE_GLASS):
        side_glass.append(side_panel(profile, side))
        expanded = tuple(
            (px + (0.014 if px > -0.3 else -0.014), pz + (-0.014 if pz < 1.2 else 0.014))
            for px, pz in profile
        )
        side_seals.append(side_panel(expanded, side))

trim = [
    *side_seals,
    front_screen(0.008),
    rear_screen(0.008),
    block(2.638, 0, 0.783, 0.025, 0.793, 0.132, 0.027),
    block(-2.638, 0, 0.775, 0.023, 0.771, 0.145, 0.015),
    block(2.56, 0, 0.405, 0.06, 0.64, 0.075, 0.018),
]
for side in (-1, 1):
    for px, py, low in ((0.625, 0.842, 0.48), (-0.92, 0.867, 0.37), (-2.04, 0.921, 0.65)):
        trim.append(line((px, side * py, low), (px, side * py, 1.055), 0.004))
    trim.extend(
        (
            line((-0.277, side * 0.772, 1.077), (-0.277, side * 0.626, 1.327), 0.012),
            block(-0.205, side * 0.836, 0.963, 0.105, 0.015, 0.019, 0.007),
            block(-2.33, side * 0.864, 0.79, 0.066, 0.010, 0.019, 0.006),
            block(2.245, side * 0.885, 0.790, 0.061, 0.010, 0.019, 0.006),
            line((0.72, side * 0.64, 1.054), (2.48, side * 0.64, 0.933), 0.003),
            line((-2.53, side * 0.68, 0.970), (-1.86, side * 0.68, 1.034), 0.003),
            block(1.63, side * 0.245, 1.063, 0.18, 0.065, 0.006, 0.004),
        )
    )
for row in range(7):
    trim.append(block(2.668, 0, 0.683 + 0.032 * row, 0.006, 0.705, 0.003, 0.001))
attach("grille_window_seals_and_panel_gaps", min_(*trim), black)
attach(
    "windscreens_and_side_glass",
    min_(*side_glass, front_screen(0.027), rear_screen(0.026)),
    glass_id,
)

red_lamps, amber_lamps, chrome_details, mirrors = [], [], [], []
for side in (-1, 1):
    red_lamps.extend(
        (
            block(-2.669, side * 0.386, 0.797, 0.012, 0.340, 0.074, 0.019),
            block(-2.50, side * 0.85, 0.49, 0.015, 0.045, 0.022, 0.008),
        )
    )
    amber_lamps.extend(
        (
            block(2.385, side * 0.78, 0.43, 0.048, 0.018, 0.020, 0.007),
            block(2.54, side * 0.66, 0.575, 0.023, 0.064, 0.015, 0.007),
        )
    )
    chrome_details.extend(
        (
            line((2.628, side * 0.78, 0.934), (2.628, side * 0.78, 0.658), 0.011),
            line((-2.654, side * 0.753, 0.876), (-2.654, side * 0.753, 0.706), 0.009),
            block(-0.205, side * 0.851, 0.968, 0.077, 0.010, 0.008, 0.004),
            x_cylinder(-2.653, side * 0.45, 0.355, 0.043, 0.032),
        )
    )
    for a, b in zip(ROOF_OUTLINE[1:6], ROOF_OUTLINE[2:7], strict=True):
        width_a = 1.469 - 0.65 * a[1]
        width_b = 1.469 - 0.65 * b[1]
        chrome_details.append(
            line((a[0], side * width_a, a[1]), (b[0], side * width_b, b[1]), 0.007)
        )
    mirrors.append(block(0.575, side * 0.855, 1.075, 0.112, 0.068, 0.038, 0.026))
chrome_details.extend(
    (
        block(2.622, 0, 0.602, 0.045, 0.83, 0.035, 0.020),
        block(-2.629, 0, 0.573, 0.045, 0.82, 0.035, 0.020),
        block(2.672, 0, 0.785, 0.007, 0.019, 0.129, 0.003),
        block(-2.682, 0, 0.793, 0.008, 0.023, 0.092, 0.003),
    )
)
attach("rectangular_1969_tail_lamps", min_(*red_lamps), red)
attach("front_turn_and_side_markers", min_(*amber_lamps), amber)
attach("split_grille_bumpers_and_trim", min_(*chrome_details), chrome)
attach("fender_mirrors", min_(*mirrors), chrome)
attach("shadowed_chassis", block(0, 0, 0.255, 1.685, 0.68, 0.0575, 0.025), black)
SCENE = MODEL.scene()
