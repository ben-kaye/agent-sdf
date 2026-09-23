"""Detailed twin-engine fighter built entirely with the agent_sdf SDK.

Coordinates are metres: +X points toward the nose, Z is up.  Run with
`.venv/bin/python -m sdf_kernel.renderer jet_fighter.py -o renders/jet-fighter.png
 --width 320 --height 240 --eye 17 -22 10 --target 0 0 0 --far 45`.
"""

from sdf_kernel import Hit, Model, custom_sdf, max_, min_
from sdf_kernel import math as field_math

# Renderer palette indices: warm titanium, charcoal composite, blue glass,
# warning red, and bright alloy.  The explicit finish data also documents the
# intended real-world material treatment independently of preview lighting.
MATERIALS = {
    "airframe": {"id": 8, "base_color": "#929a9d", "metallic": 0.72, "roughness": 0.43},
    "radome": {"id": 5, "base_color": "#292e32", "metallic": 0.08, "roughness": 0.72},
    "canopy": {"id": 6, "base_color": "#263e4a", "metallic": 0.15, "roughness": 0.11},
    "engine": {"id": 4, "base_color": "#353839", "metallic": 0.75, "roughness": 0.62},
    "missile": {"id": 0, "base_color": "#dedbd2", "metallic": 0.26, "roughness": 0.4},
    "marking": {"id": 7, "base_color": "#a84237", "metallic": 0.1, "roughness": 0.58},
}


def box(x, y, z, cx, cy, cz, hx, hy, hz):
    qx, qy, qz = (
        field_math.abs(x - cx) - hx,
        field_math.abs(y - cy) - hy,
        field_math.abs(z - cz) - hz,
    )
    ox, oy, oz = max_(qx, 0), max_(qy, 0), max_(qz, 0)
    return field_math.sqrt(ox * ox + oy * oy + oz * oz) + min_(max_(qx, max_(qy, qz)), 0)


def rounded_box(x, y, z, cx, cy, cz, hx, hy, hz, radius):
    return box(x, y, z, cx, cy, cz, hx - radius, hy - radius, hz - radius) - radius


def extruded_polygon(a, b, c, vertices, center, half_thickness):
    """Exact signed polygon distance, extruded along its third coordinate."""
    d = field_math.full_like(a, 1e4)
    inside = field_math.zeros_like(a, dtype=bool)
    for (ax, ay), (bx, by) in zip(vertices, vertices[1:] + vertices[:1], strict=True):
        ex, ey = bx - ax, by - ay
        t = field_math.clip(((a - ax) * ex + (b - ay) * ey) / (ex * ex + ey * ey), 0, 1)
        d = min_(d, field_math.hypot(a - ax - t * ex, b - ay - t * ey))
        cross = (ay > b) != (by > b)
        xcross = ax + (b - ay) * ex / (ey if ey else 1)
        inside = field_math.logical_xor(inside, cross & (a < xcross))
    planar = field_math.where(inside, -d, d)
    vertical = field_math.abs(c - center) - half_thickness
    return field_math.hypot(max_(planar, 0), max_(vertical, 0)) + min_(max_(planar, vertical), 0)


def loft(x, y, z, stations):
    """Elliptical fuselage/nacelle sections with smoothly changing proportions."""
    xp = field_math.array([s[0] for s in stations])
    yc = field_math.interp(x, xp, field_math.array([s[1] for s in stations]))
    zc = field_math.interp(x, xp, field_math.array([s[2] for s in stations]))
    ry = field_math.interp(x, xp, field_math.array([s[3] for s in stations]))
    rz = field_math.interp(x, xp, field_math.array([s[4] for s in stations]))
    radial = (field_math.hypot((y - yc) / ry, (z - zc) / rz) - 1) * min_(ry, rz)
    axial = max_(xp[0] - x, x - xp[-1])
    return field_math.hypot(max_(radial, 0), max_(axial, 0)) + min_(max_(radial, axial), 0)


def segment(x, y, z, a, b, radius):
    vx, vy, vz = (b[i] - a[i] for i in range(3))
    t = field_math.clip(
        ((x - a[0]) * vx + (y - a[1]) * vy + (z - a[2]) * vz) / (vx * vx + vy * vy + vz * vz), 0, 1
    )
    return (
        field_math.sqrt(
            (x - a[0] - t * vx) ** 2 + (y - a[1] - t * vy) ** 2 + (z - a[2] - t * vz) ** 2
        )
        - radius
    )


def x_cylinder(x, y, z, cx, cy, cz, length, radius):
    axial = field_math.abs(x - cx) - length / 2
    radial = field_math.hypot(y - cy, z - cz) - radius
    return field_math.hypot(max_(axial, 0), max_(radial, 0)) + min_(max_(axial, radial), 0)


FUSELAGE = (
    (-8.55, 0, 0.12, 0.38, 0.41),
    (-7.4, 0, 0.12, 0.78, 0.58),
    (-5.6, 0, 0.10, 1.24, 0.70),
    (-3.6, 0, 0.16, 1.38, 0.75),
    (-1.1, 0, 0.20, 1.28, 0.78),
    (1.5, 0, 0.25, 1.04, 0.68),
    (3.7, 0, 0.24, 0.78, 0.53),
    (5.3, 0, 0.20, 0.56, 0.41),
    (7.1, 0, 0.12, 0.37, 0.28),
    (8.7, 0, 0.08, 0.12, 0.12),
    (9.45, 0, 0.06, 0.013, 0.016),
)


def fuselage(x, y, z):
    d = loft(x, y, z, FUSELAGE)
    material = field_math.where(x > 7.45, 5, 8)
    return Hit(d, material)


WING = ((2.0, 1.05), (1.15, 1.95), (-1.6, 6.65), (-2.55, 6.65), (-3.55, 3.25), (-4.55, 1.45))
LEX = ((4.42, 0.55), (3.54, 0.83), (1.55, 2.17), (1.13, 2.03), (1.88, 1.01))
FLAP = ((-2.95, 2.65), (-3.50, 3.25), (-2.55, 6.65), (-2.22, 6.55))
TAILPLANE = ((-5.38, 0.75), (-6.65, 1.42), (-7.42, 3.55), (-8.18, 3.55), (-8.10, 0.75))


def lifting_surfaces(x, y, z):
    ay = field_math.abs(y)
    # Anhedral, span taper and chordwise lens section form an actual airfoil.
    wing_z = -0.04 - 0.034 * (ay - 1.4)
    wing_t = max_(0.045, 0.225 - 0.028 * (ay - 1.4))
    leading = field_math.interp(
        ay, field_math.array((1.05, 1.95, 6.65)), field_math.array((2.0, 1.15, -1.6))
    )
    trailing = field_math.interp(
        ay, field_math.array((1.45, 3.25, 6.65)), field_math.array((-4.55, -3.55, -2.55))
    )
    chord_fraction = (x - trailing) / max_(leading - trailing, 0.2)
    wing_profile = field_math.sqrt(max_(0.012, 1 - (2 * chord_fraction - 1) ** 2))
    main = extruded_polygon(x, ay, z, WING, wing_z, wing_t * wing_profile)
    root_extension = extruded_polygon(
        x, ay, z, LEX, 0.07, 0.095 * field_math.sqrt(max_(0.1, 1 - ((x - 2.7) / 1.8) ** 2))
    )
    flap = extruded_polygon(x, ay, z, FLAP, wing_z - 0.025, wing_t * 0.62 * wing_profile)
    tail_z = 0.22 + 0.065 * (ay - 0.75)
    tail_leading = -5.38 - 0.58 * (ay - 0.75)
    tail_trailing = -8.10 - 0.025 * (ay - 0.75)
    tail_fraction = (x - tail_trailing) / max_(tail_leading - tail_trailing, 0.2)
    tail_profile = field_math.sqrt(max_(0.02, 1 - (2 * tail_fraction - 1) ** 2))
    tail = extruded_polygon(
        x, ay, z, TAILPLANE, tail_z, max_(0.032, 0.105 - 0.012 * ay) * tail_profile
    )
    return min_(min_(main, root_extension), min_(flap, tail))


FIN = ((-5.65, 0.40), (-7.10, 2.72), (-8.10, 2.93), (-8.34, 2.72), (-7.75, 0.45))
RUDDER = ((-7.77, 0.65), (-7.15, 2.57), (-8.10, 2.76), (-8.31, 2.61))


def vertical_tails(x, y, z):
    # Canted fins rise from the two engine shoulders.
    fin_y = field_math.abs(y) - (1.09 + 0.23 * (z - 0.4))
    taper = max_(0.045, 0.145 - 0.043 * (z - 0.4))
    main = extruded_polygon(x, z, fin_y, FIN, 0, taper)
    rudder = extruded_polygon(x, z, fin_y, RUDDER, 0.015, taper * 0.65)
    return min_(main, rudder)


VENTRAL_FIN = ((-6.10, -0.25), (-7.28, -1.32), (-8.13, -1.29), (-7.76, -0.31))


def ventral_fins(x, y, z):
    d = field_math.full_like(x, 1e4)
    for side in (-1, 1):
        d = min_(d, extruded_polygon(x, z, y, VENTRAL_FIN, side * 1.30, 0.085))
    return d


def engines(x, y, z):
    d = field_math.full_like(x, 1e4)
    for side in (-1, 1):
        nacelle = loft(
            x,
            y,
            z,
            (
                (-8.72, side * 1.04, -0.06, 0.47, 0.43),
                (-7.6, side * 1.04, -0.05, 0.58, 0.53),
                (-5.5, side * 1.04, -0.03, 0.64, 0.57),
                (-3.0, side * 1.03, 0.03, 0.64, 0.55),
                (-1.55, side * 1.03, 0.08, 0.55, 0.45),
            ),
        )
        d = min_(d, nacelle)
    return d


def inlet_housings(x, y, z):
    d = field_math.full_like(x, 1e4)
    for side in (-1, 1):
        # The front aperture is carved from this fairing in the model history.
        fairing = rounded_box(x, y, z, -0.75, side * 1.55, -0.14, 1.42, 0.48, 0.43, 0.075)
        chin = segment(x, y, z, (1.05, side * 1.12, -0.17), (-2.35, side * 1.55, -0.28), 0.19)
        d = min_(d, min_(fairing, chin))
    return d


def inlet_cutters(x, y, z):
    d = field_math.full_like(x, 1e4)
    for side in (-1, 1):
        d = min_(d, box(x, y, z, -0.12, side * 1.57, -0.16, 1.05, 0.33, 0.29))
    return d


def inlet_depth(x, y, z):
    d = field_math.full_like(x, 1e4)
    for side in (-1, 1):
        d = min_(d, box(x, y, z, -0.62, side * 1.57, -0.16, 0.07, 0.32, 0.28))
    return d


def exhausts(x, y, z):
    d = field_math.full_like(x, 1e4)
    for side in (-1, 1):
        yc = side * 1.04
        # Concentric outer shroud, recessed black throat and inner turbine cone.
        outer = x_cylinder(x, y, z, -8.73, yc, -0.06, 0.43, 0.45)
        bore = x_cylinder(x, y, z, -8.88, yc, -0.06, 0.47, 0.325)
        ring = max_(outer, -bore)
        throat = x_cylinder(x, y, z, -8.77, yc, -0.06, 0.025, 0.318)
        cone = loft(x, y, z, ((-9.01, yc, -0.06, 0.02, 0.02), (-8.74, yc, -0.06, 0.19, 0.19)))
        d = min_(d, min_(ring, min_(throat, cone)))
        # Faceted petals on the afterburner ring.
        for k in range(12):
            angle = 2 * field_math.pi * k / 12
            py = yc + 0.43 * field_math.cos(angle)
            pz = -0.06 + 0.43 * field_math.sin(angle)
            d = min_(d, segment(x, y, z, (-8.96, py, pz), (-8.55, py, pz), 0.019))
    return d


CANOPY = (
    (1.50, 0, 0.80, 0.52, 0.12),
    (1.83, 0, 1.04, 0.69, 0.34),
    (2.55, 0, 1.16, 0.72, 0.48),
    (3.62, 0, 1.19, 0.65, 0.48),
    (4.60, 0, 1.04, 0.50, 0.33),
    (5.45, 0, 0.76, 0.13, 0.08),
)


def canopy(x, y, z):
    glass = loft(x, y, z, CANOPY)
    lower = 0.80 - z
    return max_(glass, lower)


def canopy_frames(x, y, z):
    shell = field_math.abs(loft(x, y, z, CANOPY)) - 0.022
    lower = 0.79 - z
    bands = min_(
        field_math.abs(x - 1.75) - 0.045,
        min_(field_math.abs(x - 3.02) - 0.026, field_math.abs(x - 5.07) - 0.035),
    )
    center = field_math.abs(y) - 0.018
    return max_(max_(shell, lower), min_(bands, center))


def wing_hardpoints(x, y, z):
    d = field_math.full_like(x, 1e4)
    for side in (-1, 1):
        for station_y in (2.75, 4.75):
            cx = -0.88 - 0.52 * (station_y - 2.75)
            cy = side * station_y
            pylon = extruded_polygon(
                x,
                z,
                y,
                ((cx - 0.55, -0.29), (cx + 0.38, -0.29), (cx + 0.17, -0.72), (cx - 0.27, -0.72)),
                cy,
                0.055,
            )
            d = min_(d, pylon)
    return d


def weapons(x, y, z):
    d = field_math.full_like(x, 1e4)
    for side in (-1, 1):
        for station_y in (2.75, 4.75):
            cx = -0.88 - 0.52 * (station_y - 2.75)
            cy = side * station_y
            body = x_cylinder(x, y, z, cx, cy, -0.79, 3.10, 0.125)
            nose = loft(
                x,
                y,
                z,
                ((cx + 1.54, cy, -0.79, 0.125, 0.125), (cx + 2.18, cy, -0.79, 0.004, 0.004)),
            )
            aft = field_math.full_like(x, 1e4)
            for offset in (-1, 1):
                aft = min_(
                    aft,
                    extruded_polygon(
                        x,
                        y,
                        z,
                        (
                            (cx - 1.62, cy + offset * 0.10),
                            (cx - 1.05, cy + offset * 0.10),
                            (cx - 1.42, cy + offset * 0.43),
                        ),
                        -0.79,
                        0.018,
                    ),
                )
            d = min_(d, min_(body, min_(nose, aft)))
    # Wingtip missile rails and their smaller short-range rounds.
    for side in (-1, 1):
        cy = side * 6.57
        rail = box(x, y, z, -2.10, cy, -0.26, 1.42, 0.065, 0.07)
        round_body = x_cylinder(x, y, z, -2.18, cy, -0.38, 2.35, 0.085)
        tip = loft(x, y, z, ((-0.99, cy, -0.38, 0.085, 0.085), (-0.52, cy, -0.38, 0.004, 0.004)))
        d = min_(d, min_(rail, min_(round_body, tip)))
    return d


def surface_detail(x, y, z):
    d = field_math.full_like(x, 1e4)
    for side in (-1, 1):
        # Wing panel seams, hinge line, formation light and fin rudder hinge.
        traces = (
            ((0.35, side * 2.03, 0.168), (-2.44, side * 5.82, 0.005), 0.013),
            ((-3.04, side * 2.63, 0.12), (-2.45, side * 6.41, -0.005), 0.012),
            ((-7.77, side * 1.40, 0.75), (-7.54, side * 1.62, 2.58), 0.014),
            ((-5.88, side * 1.53, 0.30), (-7.55, side * 3.29, 0.39), 0.011),
        )
        for a, b, radius in traces:
            d = min_(d, segment(x, y, z, a, b, radius))
        light = box(x, y, z, -2.30, side * 6.65, -0.18, 0.13, 0.033, 0.028)
        d = min_(d, light)
    # Dorsal refuelling spine and a small pitot probe.
    d = min_(d, segment(x, y, z, (-1.25, 0, 0.99), (1.25, 0, 0.96), 0.035))
    d = min_(d, segment(x, y, z, (9.35, 0, 0.06), (9.90, 0, 0.06), 0.016))
    return d


def sensors_and_antennas(x, y, z):
    # Forward electro-optical fairing and dorsal blade aerials.
    pod = field_math.sqrt(((x - 5.58) / 0.42) ** 2 + (y / 0.25) ** 2 + ((z + 0.42) / 0.17) ** 2) - 1
    pod = pod * 0.17
    d = pod
    for px, height in ((0.65, 0.35), (-2.25, 0.29), (-4.92, 0.24)):
        blade = extruded_polygon(
            x, z, y, ((px - 0.16, 0.88), (px + 0.06, 0.88), (px - 0.05, 0.88 + height)), 0, 0.015
        )
        d = min_(d, blade)
    for side in (-1, 1):
        # Shoulder formation lights sit flush above the inlet fairings.
        d = min_(d, box(x, y, z, -1.18, side * 1.91, 0.17, 0.29, 0.015, 0.012))
    return d


def rescue_markings(x, y, z):
    d = field_math.full_like(x, 1e4)
    for side in (-1, 1):
        # Small service triangles on the upper wing near the inboard pylon.
        mark = extruded_polygon(
            x,
            y,
            z,
            ((-0.72, side * 3.02), (-0.91, side * 3.32), (-1.08, side * 3.04)),
            0.083,
            0.014,
        )
        d = min_(d, mark)
    return d


MODEL = Model()
MODEL.start("lofted_fuselage", custom_sdf(fuselage))
MODEL.define("swept_wings_and_tailplanes", custom_sdf(lifting_surfaces, material_id=8))
MODEL.apply("lifting_surfaces_attached", "min", "swept_wings_and_tailplanes")
MODEL.define("canted_vertical_tails", custom_sdf(vertical_tails, material_id=8))
MODEL.apply("tails_attached", "min", "canted_vertical_tails")
MODEL.define("ventral_stability_fins", custom_sdf(ventral_fins, material_id=8))
MODEL.apply("ventral_fins_attached", "min", "ventral_stability_fins")
MODEL.define("twin_engine_nacelles", custom_sdf(engines, material_id=8))
MODEL.apply("engines_attached", "min", "twin_engine_nacelles")
MODEL.define("intake_fairings", custom_sdf(inlet_housings, material_id=8))
MODEL.apply("intake_housings_attached", "min", "intake_fairings")
MODEL.define("intake_duct_cutters", custom_sdf(inlet_cutters))
MODEL.apply("open_intake_ducts", "subtract", "intake_duct_cutters")
MODEL.define("shadowed_intake_depth", custom_sdf(inlet_depth, material_id=5))
MODEL.apply("intake_depth_attached", "min", "shadowed_intake_depth")
MODEL.define("afterburner_nozzles", custom_sdf(exhausts, material_id=4))
MODEL.apply("nozzles_attached", "min", "afterburner_nozzles")
MODEL.define("smoked_canopy", custom_sdf(canopy, material_id=6))
MODEL.apply("canopy_attached", "min", "smoked_canopy")
MODEL.define("canopy_structural_frames", custom_sdf(canopy_frames, material_id=5))
MODEL.apply("frames_attached", "min", "canopy_structural_frames")
MODEL.define("underwing_pylons", custom_sdf(wing_hardpoints, material_id=8))
MODEL.apply("pylons_attached", "min", "underwing_pylons")
MODEL.define("wing_stores_and_tip_rails", custom_sdf(weapons, material_id=0))
MODEL.apply("stores_attached", "min", "wing_stores_and_tip_rails")
MODEL.define("engraved_seams_and_details", custom_sdf(surface_detail, material_id=5))
MODEL.apply("detailed_airframe", "min", "engraved_seams_and_details")
MODEL.define("sensors_antennae_and_lights", custom_sdf(sensors_and_antennas, material_id=5))
MODEL.apply("flight_hardware_attached", "min", "sensors_antennae_and_lights")
MODEL.define("rescue_and_service_markings", custom_sdf(rescue_markings, material_id=7))
MODEL.apply("painted_markings_attached", "min", "rescue_and_service_markings")
SCENE = MODEL.scene()
