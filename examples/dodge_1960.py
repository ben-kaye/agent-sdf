"""1969 Dodge Charger R/T coupe, in metres with the nose along +X and Z up."""

import math

from sdf_kernel import Model, custom_sdf, max_, min_
from sdf_kernel import box as sdk_box
from sdf_kernel import math as field_math

WHEELBASE = 2.9718  # 117 in
AXLES = (-WHEELBASE / 2, WHEELBASE / 2)
HALF_LENGTH = 2.6416  # 208 in overall
HALF_WIDTH = 0.9728  # 76.6 in overall
ROOF_HEIGHT = 1.3513  # 53.2 in overall
WHEEL_Z = 0.348


def rounded_box(x, y, z, cx, cy, cz, hx, hy, hz, radius=0.0):
    qx = field_math.abs(x - cx) - hx + radius
    qy = field_math.abs(y - cy) - hy + radius
    qz = field_math.abs(z - cz) - hz + radius
    outside = field_math.sqrt(max_(qx, 0) ** 2 + max_(qy, 0) ** 2 + max_(qz, 0) ** 2)
    return outside + min_(max_(qx, max_(qy, qz)), 0) - radius


def capped_cylinder_y(x, y, z, cx, cy, cz, radius, half_depth):
    radial = field_math.hypot(x - cx, z - cz) - radius
    axial = field_math.abs(y - cy) - half_depth
    return field_math.hypot(max_(radial, 0), max_(axial, 0)) + min_(max_(radial, axial), 0)


def capped_cylinder_x(x, y, z, cx, cy, cz, radius, half_depth):
    radial = field_math.hypot(y - cy, z - cz) - radius
    axial = field_math.abs(x - cx) - half_depth
    return field_math.hypot(max_(radial, 0), max_(axial, 0)) + min_(max_(radial, axial), 0)


def segment(x, y, z, a, b, radius):
    v = tuple(b[i] - a[i] for i in range(3))
    t = field_math.clip(
        sum((p - a[i]) * v[i] for i, p in enumerate((x, y, z))) / sum(n * n for n in v), 0, 1
    )
    return (
        field_math.sqrt(sum((p - a[i] - t * v[i]) ** 2 for i, p in enumerate((x, y, z)))) - radius
    )


def polygon(x, z, vertices):
    d = field_math.full_like(x, 1e3)
    inside = field_math.zeros_like(x, dtype=bool)
    for (ax, az), (bx, bz) in zip(vertices, vertices[1:] + vertices[:1], strict=True):
        vx, vz = bx - ax, bz - az
        t = field_math.clip(((x - ax) * vx + (z - az) * vz) / (vx * vx + vz * vz), 0, 1)
        d = min_(d, field_math.hypot(x - ax - t * vx, z - az - t * vz))
        crossing = (az > z) != (bz > z)
        inside = field_math.logical_xor(
            inside, crossing & (x < ax + (z - az) * vx / (vz if vz else 1))
        )
    return field_math.where(inside, -d, d)


def profile_prism(x, y, z, vertices, center_y, half_width):
    profile = polygon(x, z, vertices)
    depth = field_math.abs(y - center_y) - half_width
    return field_math.hypot(max_(profile, 0), max_(depth, 0)) + min_(max_(profile, depth), 0)


# The width table makes the pinched doors and pronounced rear hips part of the
# main volume. The shoulder and hood heights are independent of the lower sill.
STATIONS = field_math.array((-2.642, -2.42, -1.88, -1.49, -1.03, -0.20, 0.65, 1.49, 2.30, 2.642))
WIDTHS = field_math.array((0.76, 0.88, 0.94, 0.963, 0.87, 0.825, 0.84, 0.955, 0.90, 0.78))
TOPS = field_math.array((0.956, 1.005, 1.035, 1.055, 1.045, 1.04, 1.047, 1.025, 0.955, 0.90))


def body(x, y, z):
    shoulder = field_math.interp(x, STATIONS, WIDTHS)
    width = shoulder - 0.055 * field_math.clip((0.82 - z) / 0.55, 0, 1)
    top = field_math.interp(x, STATIONS, TOPS)
    bottom = field_math.interp(
        x,
        field_math.array((-2.642, -2.25, -1.75, 1.90, 2.26, 2.642)),
        field_math.array((0.39, 0.28, 0.205, 0.205, 0.34, 0.46)),
    )
    center_z = (top + bottom) * 0.5
    shell = rounded_box(x, y, z, 0, 0, center_z, HALF_LENGTH, width, (top - bottom) * 0.5, 0.042)
    # A recessed door center breaks the otherwise flat slab sided cross section.
    scallop_radius = field_math.sqrt(((x + 0.12) / 0.90) ** 2 + ((z - 0.752) / 0.19) ** 2)
    scallop_depth = 0.018 * field_math.clip(1 - scallop_radius**2, 0, 1)
    scallop = max_((scallop_radius - 1) * 0.12, width - scallop_depth - field_math.abs(y))
    shell = max_(shell, -scallop)
    for axle in AXLES:
        # The wheel openings are cut in the sheet metal before wheel insertion.
        shell = max_(shell, -capped_cylinder_y(x, y, z, axle, 0, WHEEL_Z, 0.403, 1.05))
    return shell


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


def roof_width(z):
    return field_math.interp(
        z, field_math.array((1.0, 1.08, 1.32, 1.36)), field_math.array((0.79, 0.765, 0.61, 0.59))
    )


def rear_glass_x(z):
    return field_math.interp(z, field_math.array((1.035, 1.315)), field_math.array((-1.505, -1.07)))


def roof(x, y, z):
    outline = polygon(x, z, ROOF_OUTLINE)
    lateral = field_math.abs(y) - roof_width(z)
    shell = field_math.hypot(max_(outline, 0), max_(lateral, 0)) + min_(max_(outline, lateral), 0)
    # The standard Charger has a tunneled backlight. Only the center is cut
    # back; the wide sail panels continue all the way down to the deck.
    recess = max_(x - rear_glass_x(z), field_math.abs(y) - 0.535, 1.045 - z, z - 1.318)
    return max_(shell, -recess)


def painted_pressings(x, y, z):
    ay = field_math.abs(y)
    d = field_math.full_like(x, 1e3)
    # Raised wheel lips and the R/T power bulge are painted sheet metal.
    for axle in AXLES:
        radial = field_math.hypot(x - axle, z - WHEEL_Z)
        lip = field_math.hypot(radial - 0.418, ay - 0.936) - 0.018
        d = min_(d, max_(lip, 0.31 - z))
    d = min_(
        d,
        rounded_box(x, y, z, 1.30, 0, 1.026, 0.66, 0.335, 0.032, 0.025),
        rounded_box(x, y, z, -2.27, 0, 0.997, 0.34, 0.73, 0.018, 0.012),
    )
    # Long rocker panels make the car read as a low two-door B-body.
    for side in (-1, 1):
        d = min_(d, rounded_box(x, y, z, -0.01, side * 0.807, 0.285, 1.04, 0.034, 0.055, 0.013))
    return d


def tires(x, y, z):
    d = field_math.full_like(x, 1e3)
    for axle in AXLES:
        for side in (-1, 1):
            outer = capped_cylinder_y(x, y, z, axle, side * 0.823, WHEEL_Z, 0.348, 0.132)
            bore = capped_cylinder_y(x, y, z, axle, side * 0.823, WHEEL_Z, 0.244, 0.15)
            d = min_(d, max_(outer, -bore))
    return d


def wheels_dark(x, y, z):
    d = field_math.full_like(x, 1e3)
    for axle in AXLES:
        for side in (-1, 1):
            d = min_(d, capped_cylinder_y(x, y, z, axle, side * 0.937, WHEEL_Z, 0.240, 0.018))
    return d


def wheels_chrome(x, y, z):
    d = field_math.full_like(x, 1e3)
    for axle in AXLES:
        for side in (-1, 1):
            cy = side * 0.954
            outer = capped_cylinder_y(x, y, z, axle, cy, WHEEL_Z, 0.247, 0.013)
            inner = capped_cylinder_y(x, y, z, axle, cy, WHEEL_Z, 0.220, 0.021)
            d = min_(d, max_(outer, -inner))
            for spoke in range(5):
                a = 2 * math.pi * spoke / 5 + math.pi / 2
                c, s = math.cos(a), math.sin(a)
                d = min_(
                    d,
                    segment(
                        x,
                        y,
                        z,
                        (axle + 0.048 * c, cy, WHEEL_Z + 0.048 * s),
                        (axle + 0.215 * c, cy, WHEEL_Z + 0.215 * s),
                        0.023,
                    ),
                )
            d = min_(d, capped_cylinder_y(x, y, z, axle, cy + side * 0.009, WHEEL_Z, 0.061, 0.012))
    return d


FRONT_SIDE_GLASS = ((-0.25, 1.073), (0.565, 1.073), (0.022, 1.316), (-0.25, 1.316))
REAR_SIDE_GLASS = ((-1.505, 1.074), (-0.302, 1.074), (-0.302, 1.316), (-1.03, 1.316))


def side_glass(x, y, z, border=False):
    profiles = (FRONT_SIDE_GLASS, REAR_SIDE_GLASS)
    surface = roof_width(z)
    d = field_math.full_like(x, 1e3)
    for side in (-1, 1):
        for profile in profiles:
            if border:
                shifted = tuple(
                    (px + (0.014 if px > -0.3 else -0.014), pz + (-0.014 if pz < 1.2 else 0.014))
                    for px, pz in profile
                )
                shape = shifted
            else:
                shape = profile
            d = min_(d, profile_prism(x, y, z, shape, side * (surface + 0.004), 0.012))
    return d


def windscreen(x, y, z, border=False):
    margin = 0.008 if border else 0.027
    low, high = 1.058 + margin, 1.319 - margin
    surface = field_math.interp(
        z, field_math.array((1.058, 1.319)), field_math.array((0.682, 0.043))
    )
    width = roof_width(z) - margin
    return max_(field_math.abs(x - surface) - 0.015, field_math.abs(y) - width, low - z, z - high)


def backlight(x, y, z, border=False):
    margin = 0.008 if border else 0.026
    width = 0.525 - margin
    return max_(
        field_math.abs(x - rear_glass_x(z)) - 0.014,
        field_math.abs(y) - width,
        1.048 + margin - z,
        z - 1.313 + margin,
    )


def glass(x, y, z):
    return min_(side_glass(x, y, z), windscreen(x, y, z), backlight(x, y, z))


def dark_details(x, y, z):
    d = min_(
        side_glass(x, y, z, border=True),
        windscreen(x, y, z, border=True),
        backlight(x, y, z, border=True),
        # Deeply recessed, full-width hidden-headlamp grille and rear panel.
        rounded_box(x, y, z, 2.638, 0, 0.783, 0.025, 0.793, 0.132, 0.027),
        rounded_box(x, y, z, -2.638, 0, 0.775, 0.023, 0.771, 0.145, 0.015),
        rounded_box(x, y, z, 2.56, 0, 0.405, 0.06, 0.64, 0.075, 0.018),
    )
    for side in (-1, 1):
        # Door, rear-quarter, and rocker shut lines.
        for px, py, low in ((0.625, 0.842, 0.48), (-0.92, 0.867, 0.37), (-2.04, 0.921, 0.65)):
            d = min_(d, segment(x, y, z, (px, side * py, low), (px, side * py, 1.055), 0.004))
        d = min_(
            d, segment(x, y, z, (-0.277, side * 0.772, 1.077), (-0.277, side * 0.626, 1.327), 0.012)
        )
        d = min_(d, rounded_box(x, y, z, -0.205, side * 0.836, 0.963, 0.105, 0.015, 0.019, 0.007))
        d = min_(d, rounded_box(x, y, z, -2.33, side * 0.864, 0.79, 0.066, 0.010, 0.019, 0.006))
        d = min_(d, rounded_box(x, y, z, 2.245, side * 0.885, 0.790, 0.061, 0.010, 0.019, 0.006))
        d = min_(
            d,
            segment(x, y, z, (0.72, side * 0.64, 1.054), (2.48, side * 0.64, 0.933), 0.003),
        )
        d = min_(
            d,
            segment(x, y, z, (-2.53, side * 0.68, 0.970), (-1.86, side * 0.68, 1.034), 0.003),
        )
        d = min_(d, rounded_box(x, y, z, 1.63, side * 0.245, 1.063, 0.18, 0.065, 0.006, 0.004))
    for row in range(7):
        h = 0.683 + 0.032 * row
        d = min_(d, rounded_box(x, y, z, 2.668, 0, h, 0.006, 0.705, 0.003, 0.001))
    return d


def red_lamps(x, y, z):
    d = field_math.full_like(x, 1e3)
    for side in (-1, 1):
        d = min_(d, rounded_box(x, y, z, -2.669, side * 0.386, 0.797, 0.012, 0.340, 0.074, 0.019))
        d = min_(d, rounded_box(x, y, z, -2.50, side * 0.85, 0.49, 0.015, 0.045, 0.022, 0.008))
    return d


def amber_lamps(x, y, z):
    d = field_math.full_like(x, 1e3)
    for side in (-1, 1):
        d = min_(d, rounded_box(x, y, z, 2.385, side * 0.78, 0.43, 0.048, 0.018, 0.020, 0.007))
        d = min_(d, rounded_box(x, y, z, 2.54, side * 0.66, 0.575, 0.023, 0.064, 0.015, 0.007))
    return d


def chrome_details(x, y, z):
    d = field_math.full_like(x, 1e3)
    # Front bumper wraps around the recessed grille; rear bumper sits below lamps.
    for zc, xc, half_y in ((0.602, 2.622, 0.83), (0.573, -2.629, 0.82)):
        d = min_(d, rounded_box(x, y, z, xc, 0, zc, 0.045, half_y, 0.035, 0.020))
    for side in (-1, 1):
        d = min_(
            d,
            segment(x, y, z, (2.628, side * 0.78, 0.934), (2.628, side * 0.78, 0.658), 0.011),
            segment(x, y, z, (-2.654, side * 0.753, 0.876), (-2.654, side * 0.753, 0.706), 0.009),
            rounded_box(x, y, z, -0.205, side * 0.851, 0.968, 0.077, 0.010, 0.008, 0.004),
            capped_cylinder_x(x, y, z, -2.653, side * 0.45, 0.355, 0.043, 0.032),
        )
        # Polished drip rail follows the roof edge and sail panel.
        for a, b in zip(ROOF_OUTLINE[1:6], ROOF_OUTLINE[2:7], strict=True):
            d = min_(
                d,
                segment(
                    x,
                    y,
                    z,
                    (a[0], side * roof_width(a[1]), a[1]),
                    (b[0], side * roof_width(b[1]), b[1]),
                    0.007,
                ),
            )
    d = min_(
        d,
        rounded_box(x, y, z, 2.672, 0, 0.785, 0.007, 0.019, 0.129, 0.003),
        rounded_box(x, y, z, -2.682, 0, 0.793, 0.008, 0.023, 0.092, 0.003),
    )
    return d


def mirrors(x, y, z):
    d = field_math.full_like(x, 1e3)
    for side in (-1, 1):
        d = min_(d, rounded_box(x, y, z, 0.575, side * 0.855, 1.075, 0.112, 0.068, 0.038, 0.026))
    return d


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


def attach(name, fn, material):
    MODEL.define(name, custom_sdf(fn, material_id=material))
    MODEL.apply(name + "_attached", "min", name)


MODEL.start("coke_bottle_body_and_wheel_openings", custom_sdf(body, material_id=paint))
attach("flying_buttress_roof", roof, paint)
attach("hood_fenders_and_sills", painted_pressings, paint)
attach("four_period_tires", tires, rubber)
attach("magnum_wheel_insets", wheels_dark, black)
attach("magnum_five_spoke_wheels", wheels_chrome, chrome)
attach("grille_window_seals_and_panel_gaps", dark_details, black)
attach("windscreens_and_side_glass", glass, glass_id)
attach("rectangular_1969_tail_lamps", red_lamps, red)
attach("front_turn_and_side_markers", amber_lamps, amber)
attach("split_grille_bumpers_and_trim", chrome_details, chrome)
attach("fender_mirrors", mirrors, chrome)

# A native SDK solid sets the recessed dark underbody behind the wheels.
MODEL.define(
    "shadowed_chassis",
    sdk_box(3.37, 1.36, 0.115, corner_radius=0.025).translate(0, 0, 0.255).with_material(black),
)
MODEL.apply("chassis_attached", "min", "shadowed_chassis")
SCENE = MODEL.scene()
