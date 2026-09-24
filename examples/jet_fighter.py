"""Twin-engine fighter assembled from SDK sketches and solids."""

import math

from sdf_kernel import (
    Model,
    Plane,
    Transform,
    box,
    capsule,
    circle,
    cylinder,
    max_,
    min_,
    polygon,
    sphere,
)
from sdf_kernel import math as field_math


def block(x, y, z, hx, hy, hz, radius=0):
    return box(2 * hx, 2 * hy, 2 * hz, radius).translate(x, y, z)


def x_cylinder(x, y, z, length, radius):
    return cylinder(radius, length).rotate(math.pi / 2, "y").translate(x, y, z)


def xz_polygon(points, thickness, y=0):
    return polygon(points).on(Plane.xz(y=y)).extrude(thickness)


def ellipse_loft(stations):
    return circle(1).loft(stations, axis="x")


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
WING = ((2.0, 1.05), (1.15, 1.95), (-1.6, 6.65), (-2.55, 6.65), (-3.55, 3.25), (-4.55, 1.45))
LEX = ((4.42, 0.55), (3.54, 0.83), (1.55, 2.17), (1.13, 2.03), (1.88, 1.01))
FLAP = ((-2.95, 2.65), (-3.50, 3.25), (-2.55, 6.65), (-2.22, 6.55))
TAILPLANE = ((-5.38, 0.75), (-6.65, 1.42), (-7.42, 3.55), (-8.18, 3.55), (-8.10, 0.75))
FIN = ((-5.65, 0.40), (-7.10, 2.72), (-8.10, 2.93), (-8.34, 2.72), (-7.75, 0.45))
RUDDER = ((-7.77, 0.65), (-7.15, 2.57), (-8.10, 2.76), (-8.31, 2.61))
VENTRAL_FIN = ((-6.10, -0.25), (-7.28, -1.32), (-8.13, -1.29), (-7.76, -0.31))
CANOPY = (
    (1.50, 0, 0.80, 0.52, 0.12),
    (1.83, 0, 1.04, 0.69, 0.34),
    (2.55, 0, 1.16, 0.72, 0.48),
    (3.62, 0, 1.19, 0.65, 0.48),
    (4.60, 0, 1.04, 0.50, 0.33),
    (5.45, 0, 0.76, 0.13, 0.08),
)


def mirrored_polygon(points, side):
    return polygon((x, side * y) for x, y in points)


def wing_height(x, y):
    span = field_math.abs(y)
    leading = field_math.interp(span, (1.05, 1.95, 6.65), (2.0, 1.15, -1.6))
    trailing = field_math.interp(span, (1.45, 3.25, 6.65), (-4.55, -3.55, -2.55))
    fraction = (x - trailing) / max_(leading - trailing, 0.2)
    profile = field_math.sqrt(max_(0.012, 1 - (2 * fraction - 1) ** 2))
    return 2 * max_(0.045, 0.225 - 0.028 * (span - 1.4)) * profile


def wing_offset(x, y):
    return -0.04 - 0.034 * (field_math.abs(y) - 1.4)


def lex_height(x, y):
    return 0.19 * field_math.sqrt(max_(0.1, 1 - ((x - 2.7) / 1.8) ** 2))


def tail_height(x, y):
    span = field_math.abs(y)
    leading = -5.38 - 0.58 * (span - 0.75)
    trailing = -8.10 - 0.025 * (span - 0.75)
    fraction = (x - trailing) / max_(leading - trailing, 0.2)
    profile = field_math.sqrt(max_(0.02, 1 - (2 * fraction - 1) ** 2))
    return 2 * max_(0.032, 0.105 - 0.012 * span) * profile


def tail_offset(x, y):
    return 0.22 + 0.065 * (field_math.abs(y) - 0.75)


def lifting_surfaces():
    parts = []
    for side in (-1, 1):
        parts.append(
            mirrored_polygon(WING, side).extrude(
                wing_height,
                offset=wing_offset,
                bounds=((-4.6, -6.7, -0.5), (2.1, 6.7, 0.5)),
                height_lipschitz=128,
                offset_lipschitz=0.034,
            )
        )
        parts.append(
            mirrored_polygon(LEX, side).extrude(
                lex_height,
                offset=0.07,
                bounds=((1.1, -2.2, -0.2), (4.5, 2.2, 0.3)),
                height_lipschitz=1,
            )
        )
        parts.append(
            mirrored_polygon(FLAP, side).extrude(
                lambda x, y: 0.62 * wing_height(x, y),
                offset=wing_offset,
                bounds=((-3.6, -6.7, -0.5), (-2.1, 6.7, 0.5)),
                height_lipschitz=80,
                offset_lipschitz=0.034,
            )
        )
        parts.append(
            mirrored_polygon(TAILPLANE, side).extrude(
                tail_height,
                offset=tail_offset,
                bounds=((-8.2, -3.6, -0.1), (-5.3, 3.6, 0.6)),
                height_lipschitz=8,
                offset_lipschitz=0.065,
            )
        )
    return min_(*parts)


def fin_height(x, z):
    return 2 * max_(0.045, 0.145 - 0.043 * (z - 0.4))


def vertical_tails():
    fins = []
    for side in (-1, 1):

        def fin_offset(x, z, side=side):
            return -side * (1.09 + 0.23 * (z - 0.4))

        main = (
            polygon(FIN)
            .on(Plane.xz())
            .extrude(
                fin_height,
                offset=fin_offset,
                bounds=((-8.4, 0.3, -2), (-5.6, 3, 2)),
                height_lipschitz=0.1,
                offset_lipschitz=0.23,
            )
        )
        rudder = (
            polygon(RUDDER)
            .on(Plane.xz())
            .extrude(
                lambda x, z: 0.65 * fin_height(x, z),
                offset=lambda x, z, side=side: -side * (1.105 + 0.23 * (z - 0.4)),
                bounds=((-8.4, 0.6, -2), (-7.1, 2.8, 2)),
                height_lipschitz=0.1,
                offset_lipschitz=0.23,
            )
        )
        fins.extend((main, rudder))
    return min_(*fins)


def engines():
    return min_(
        *(
            ellipse_loft(
                (
                    (-8.72, side * 1.04, -0.06, 0.47, 0.43),
                    (-7.6, side * 1.04, -0.05, 0.58, 0.53),
                    (-5.5, side * 1.04, -0.03, 0.64, 0.57),
                    (-3.0, side * 1.03, 0.03, 0.64, 0.55),
                    (-1.55, side * 1.03, 0.08, 0.55, 0.45),
                )
            )
            for side in (-1, 1)
        )
    )


def inlet_housings():
    return min_(
        *(
            block(-0.75, side * 1.55, -0.14, 1.42, 0.48, 0.43, 0.075)
            | capsule((1.05, side * 1.12, -0.17), (-2.35, side * 1.55, -0.28), 0.19)
            for side in (-1, 1)
        )
    )


def exhausts():
    parts = []
    for side in (-1, 1):
        yc = side * 1.04
        outer = x_cylinder(-8.73, yc, -0.06, 0.43, 0.45)
        bore = x_cylinder(-8.88, yc, -0.06, 0.47, 0.325)
        throat = x_cylinder(-8.77, yc, -0.06, 0.025, 0.318)
        cone = ellipse_loft(((-9.01, yc, -0.06, 0.02, 0.02), (-8.74, yc, -0.06, 0.19, 0.19)))
        parts.extend((outer - bore, throat, cone))
        for index in range(12):
            angle = 2 * math.pi * index / 12
            py = yc + 0.43 * math.cos(angle)
            pz = -0.06 + 0.43 * math.sin(angle)
            parts.append(capsule((-8.96, py, pz), (-8.55, py, pz), 0.019))
    return min_(*parts)


def canopy_frames(glass):
    bands = min_(
        block(1.75, 0, 1.2, 0.045, 1, 1),
        block(3.02, 0, 1.2, 0.026, 1, 1),
        block(5.07, 0, 1.2, 0.035, 1, 1),
    )
    center = block(3.5, 0, 1.2, 3, 0.018, 1)
    return glass.shell(0.044) & block(3.5, 0, 2.29, 3, 1, 1.5) & (bands | center)


def wing_hardpoints():
    pylons = []
    for side in (-1, 1):
        for station_y in (2.75, 4.75):
            cx = -0.88 - 0.52 * (station_y - 2.75)
            pylons.append(
                xz_polygon(
                    (
                        (cx - 0.55, -0.29),
                        (cx + 0.38, -0.29),
                        (cx + 0.17, -0.72),
                        (cx - 0.27, -0.72),
                    ),
                    0.11,
                    side * station_y,
                )
            )
    return min_(*pylons)


def weapons():
    stores = []
    for side in (-1, 1):
        for station_y in (2.75, 4.75):
            cx = -0.88 - 0.52 * (station_y - 2.75)
            cy = side * station_y
            stores.append(x_cylinder(cx, cy, -0.79, 3.10, 0.125))
            stores.append(
                ellipse_loft(
                    (
                        (cx + 1.54, cy, -0.79, 0.125, 0.125),
                        (cx + 2.18, cy, -0.79, 0.004, 0.004),
                    )
                )
            )
            for offset in (-1, 1):
                stores.append(
                    polygon(
                        (
                            (cx - 1.62, cy + offset * 0.10),
                            (cx - 1.05, cy + offset * 0.10),
                            (cx - 1.42, cy + offset * 0.43),
                        )
                    ).extrude(0.036, offset=-0.79)
                )
        cy = side * 6.57
        stores.extend(
            (
                block(-2.10, cy, -0.26, 1.42, 0.065, 0.07),
                x_cylinder(-2.18, cy, -0.38, 2.35, 0.085),
                ellipse_loft(((-0.99, cy, -0.38, 0.085, 0.085), (-0.52, cy, -0.38, 0.004, 0.004))),
            )
        )
    return min_(*stores)


def surface_detail():
    details = []
    for side in (-1, 1):
        details.extend(
            capsule(a, b, radius)
            for a, b, radius in (
                ((0.35, side * 2.03, 0.168), (-2.44, side * 5.82, 0.005), 0.013),
                ((-3.04, side * 2.63, 0.12), (-2.45, side * 6.41, -0.005), 0.012),
                ((-7.77, side * 1.40, 0.75), (-7.54, side * 1.62, 2.58), 0.014),
                ((-5.88, side * 1.53, 0.30), (-7.55, side * 3.29, 0.39), 0.011),
            )
        )
        details.append(block(-2.30, side * 6.65, -0.18, 0.13, 0.033, 0.028))
    details.extend(
        (
            capsule((-1.25, 0, 0.99), (1.25, 0, 0.96), 0.035),
            capsule((9.35, 0, 0.06), (9.90, 0, 0.06), 0.016),
        )
    )
    return min_(*details)


def sensors_and_antennas():
    parts = [sphere(1).transform(Transform.scaling(0.42, 0.25, 0.17)).translate(5.58, 0, -0.42)]
    for px, height in ((0.65, 0.35), (-2.25, 0.29), (-4.92, 0.24)):
        parts.append(
            xz_polygon(((px - 0.16, 0.88), (px + 0.06, 0.88), (px - 0.05, 0.88 + height)), 0.03)
        )
    for side in (-1, 1):
        parts.append(block(-1.18, side * 1.91, 0.17, 0.29, 0.015, 0.012))
    return min_(*parts)


def rescue_markings():
    return min_(
        *(
            mirrored_polygon(((-0.72, 3.02), (-0.91, 3.32), (-1.08, 3.04)), side).extrude(
                0.028, offset=0.083
            )
            for side in (-1, 1)
        )
    )


MODEL = Model()
airframe = MODEL.material("airframe", base_color="#929a9d", metallic=0.72, roughness=0.43)
radome = MODEL.material("radome", base_color="#292e32", metallic=0.08, roughness=0.72)
glass_id = MODEL.material("canopy", base_color="#263e4a", metallic=0.15, roughness=0.11)
engine_id = MODEL.material("engine", base_color="#353839", metallic=0.75, roughness=0.62)
missile = MODEL.material("missile", base_color="#dedbd2", metallic=0.26, roughness=0.40)
marking = MODEL.material("marking", base_color="#a84237", metallic=0.10, roughness=0.58)


def attach(name, shape, material):
    MODEL.define(name, shape.with_material(material))
    MODEL.apply(name + "_attached", "min", name)


fuselage = ellipse_loft(FUSELAGE)
MODEL.start("lofted_fuselage", fuselage.with_material(airframe))
attach("radome", fuselage & block(8.95, 0, 0, 1.5, 2, 2), radome)
attach("swept_wings_and_tailplanes", lifting_surfaces(), airframe)
attach("canted_vertical_tails", vertical_tails(), airframe)
attach(
    "ventral_stability_fins",
    min_(*(xz_polygon(VENTRAL_FIN, 0.17, side * 1.30) for side in (-1, 1))),
    airframe,
)
attach("twin_engine_nacelles", engines(), airframe)
attach("intake_fairings", inlet_housings(), airframe)
MODEL.define(
    "intake_duct_cutters",
    min_(*(block(-0.12, side * 1.57, -0.16, 1.05, 0.33, 0.29) for side in (-1, 1))),
)
MODEL.apply("open_intake_ducts", "subtract", "intake_duct_cutters")
attach(
    "shadowed_intake_depth",
    min_(*(block(-0.62, side * 1.57, -0.16, 0.07, 0.32, 0.28) for side in (-1, 1))),
    radome,
)
attach("afterburner_nozzles", exhausts(), engine_id)
glass = ellipse_loft(CANOPY)
attach("smoked_canopy", glass & block(3.5, 0, 2.30, 3, 1, 1.5), glass_id)
attach("canopy_structural_frames", canopy_frames(glass), radome)
attach("underwing_pylons", wing_hardpoints(), airframe)
attach("wing_stores_and_tip_rails", weapons(), missile)
attach("engraved_seams_and_details", surface_detail(), radome)
attach("sensors_antennae_and_lights", sensors_and_antennas(), radome)
attach("rescue_and_service_markings", rescue_markings(), marking)
SCENE = MODEL.scene()
