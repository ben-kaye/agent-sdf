"""SWR-inspired electric train. Metres, Z up, leading cab toward -X."""

import math
import os

from sdf_kernel import (
    LoftStation,
    Material,
    Part,
    Plane,
    Scene,
    box,
    capsule,
    cylinder,
    min_,
    polygon,
    rectangle,
    torus,
)

VIEW = os.environ.get("SWR_VIEW", "exterior")
if VIEW not in {"exterior", "cutaway", "interior"}:
    raise ValueError("SWR_VIEW must be exterior, cutaway, or interior")

NAVY = Material("swr_midnight_blue", base_color="#14243d", metallic=0.36, roughness=0.29)
SILVER = Material("satin_silver_body", base_color="#c4cbd0", metallic=0.58, roughness=0.32)
YELLOW = Material("cab_safety_yellow", base_color="#f7d52f", roughness=0.34)
CYAN = Material("swr_blue_handles", base_color="#009dd1", metallic=0.2, roughness=0.3)
GLASS = Material("blue_smoked_glazing", base_color="#172d3d", metallic=0.52, roughness=0.12)
SEAL = Material("window_rubber_and_bellows", base_color="#121820", roughness=0.77)
STEEL = Material("machined_steel", base_color="#89969d", metallic=0.88, roughness=0.27)
DARK_STEEL = Material("bogie_steel", base_color="#303b44", metallic=0.65, roughness=0.58)
WHITE = Material("warm_white", base_color="#f4f3df", roughness=0.31)
RED = Material("red_marker_lights", base_color="#b42b32", metallic=0.15, roughness=0.2)
AMBER = Material("destination_leds", base_color="#ffb64d", roughness=0.5)
SHELL = Material("pale_seat_shells", base_color="#d8dcd8", roughness=0.6)
FABRIC = Material("navy_seat_upholstery", base_color="#293747", roughness=0.96)
FLOOR = Material("warm_grey_saloon_floor", base_color="#8b8d86", roughness=0.94)
SLEEPER = Material("concrete_sleepers", base_color="#777d7e", roughness=0.98)
BALLAST = Material("display_track_bed", base_color="#3b424b", roughness=0.97)


PART_MATERIALS = dict(
    enumerate(
        (
            NAVY,
            SILVER,
            YELLOW,
            CYAN,
            GLASS,
            SEAL,
            STEEL,
            DARK_STEEL,
            WHITE,
            RED,
            AMBER,
            SHELL,
            FABRIC,
            FLOOR,
            SLEEPER,
            BALLAST,
        )
    )
)
MATERIAL_IDS = {material: index for index, material in PART_MATERIALS.items()}


def block(x, y, z, length, width, height, radius=0):
    return box(length, width, height, min(radius, length / 2, width / 2, height / 2)).translate(
        x, y, z
    )


def colored(shape, material):
    return shape.with_material(MATERIAL_IDS[material])


def side_panel(x, y, z, width, height, depth, radius=0):
    return rectangle(width, height, radius).on(Plane.xz(y=y)).extrude(depth).translate(x, 0, z)


def front_panel(points, depth=2):
    return polygon(points).on(Plane.yz(x=-9.2)).extrude(depth)


def axle_cylinder(x, y, z, radius, length):
    return cylinder(radius, length).rotate(math.pi / 2, "x").translate(x, y, z)


def rail_line(a, b, radius=0.018, material=STEEL):
    return colored(capsule(a, b, radius), material)


GLYPHS = {
    "S": ("111", "100", "100", "111", "001", "001", "111"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "R": ("1110", "1001", "1001", "1110", "1010", "1001", "1001"),
    "0": ("111", "101", "101", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "010", "010", "111"),
    "2": ("111", "001", "001", "111", "100", "100", "111"),
    "4": ("101", "101", "101", "111", "001", "001", "001"),
    "7": ("111", "001", "001", "010", "010", "010", "010"),
}


def lettering(text, pixel=0.035):
    strokes = []
    cursor = 0
    for letter in text:
        if letter == " ":
            cursor += 3 * pixel
            continue
        rows = GLYPHS[letter]
        for row, cells in enumerate(rows):
            start = None
            for col, cell in enumerate(cells + "0"):
                if cell == "1" and start is None:
                    start = col
                if cell == "0" and start is not None:
                    strokes.append(
                        block(
                            cursor + (start + col) * pixel / 2,
                            0,
                            (6 - row) * pixel,
                            (col - start) * pixel,
                            0.014,
                            pixel,
                        )
                    )
                    start = None
        cursor += (len(rows[0]) + 1) * pixel
    return min_(*strokes).translate(-cursor / 2, 0, 0)


def seat():
    parts = [
        colored(block(-0.18, 0, 1.32, 0.12, 0.11, 0.42, 0.03), DARK_STEEL),
        colored(block(0, 0, 1.52, 0.51, 0.45, 0.13, 0.055), SHELL),
        colored(block(0.025, 0, 1.60, 0.48, 0.425, 0.115, 0.05), FABRIC),
    ]
    back = rectangle(0.46, 0.89, 0.085).on(Plane.yz()).extrude(0.11)
    padding = rectangle(0.408, 0.79, 0.085).on(Plane.yz()).extrude(0.075)
    parts.extend(
        (
            colored(back.rotate(-0.11, "y").translate(-0.225, 0, 2.045), SHELL),
            colored(padding.rotate(-0.11, "y").translate(-0.153, 0, 2.075), FABRIC),
            colored(block(-0.207, 0, 2.424, 0.14, 0.42, 0.13, 0.055), FABRIC),
        )
    )
    for side in (-1, 1):
        parts.extend(
            (
                rail_line((-0.17, side * 0.236, 1.76), (0.23, side * 0.236, 1.76), 0.025, SEAL),
                rail_line((-0.18, side * 0.22, 1.60), (-0.18, side * 0.236, 1.75), 0.016),
            )
        )
    handle = (
        (rectangle(0.24, 0.29, 0.085) - rectangle(0.155, 0.195, 0.055))
        .on(Plane.yz())
        .extrude(0.048)
    )
    parts.append(colored(handle.translate(-0.245, 0, 2.455), CYAN))
    return min_(*parts)


def interior():
    pair = seat().linear_pattern(2, 0, 0.49, 0).translate(0, -1.035, 0)
    row = pair.linear_pattern(2, 0, 1.58, 0)
    parts = [colored(block(0.62, 0, 1.10, 15.65, 2.52, 0.12), FLOOR)]
    for x, count in ((-3.40, 7), (4.25, 4)):
        parts.append(row.linear_pattern(count, 0.80, 0, 0).translate(x, 0, 0))
    parts.append(row.rotate(math.pi, "z").translate(-6.20, 0, 0))
    for x in (-4.78, 2.77):
        for side in (-1, 1):
            parts.append(rail_line((x, side * 0.40, 1.16), (x, side * 0.40, 3.48), 0.022))
            parts.append(
                colored(side_panel(x + 0.15, side * 0.88, 1.97, 0.07, 1.65, 0.66, 0.025), SHELL)
            )
            parts.append(rail_line((x + 0.10, side * 0.53, 1.35), (x + 0.10, side * 0.53, 2.75)))
    for side in (-1, 1):
        parts.extend(
            (
                colored(block(0.65, side * 1.12, 3.11, 15.5, 0.46, 0.055, 0.02), SHELL),
                rail_line((-7.05, side * 0.865, 3.17), (8.40, side * 0.865, 3.17), 0.025),
                colored(block(0.65, side * 0.72, 3.49, 15.45, 0.065, 0.024, 0.01), WHITE),
            )
        )
        bracket = rail_line((0, side * 0.90, 3.12), (0, side * 1.26, 3.45), 0.016)
        parts.append(bracket.linear_pattern(9, 1.75, 0, 0).translate(-6.50, 0, 0))
    parts.extend(
        (
            colored(block(-7.15, 0, 2.25, 0.07, 2.49, 2.25), SHELL),
            colored(
                side_panel(0, 0, 0, 0.65, 1.7, 0.07, 0.03)
                .rotate(math.pi / 2, "z")
                .translate(-7.19, 0, 2.12),
                SEAL,
            ),
            colored(block(8.37, 0, 2.60, 0.065, 0.87, 0.20, 0.035), SEAL),
            colored(block(8.326, 0, 2.60, 0.015, 0.66, 0.025), AMBER),
        )
    )
    return min_(*parts)


def bogie(x):
    parts = [colored(block(x, 0, 0.65, 3.18, 1.70, 0.25, 0.08), DARK_STEEL)]
    for offset in (-1.05, 1.05):
        xx = x + offset
        parts.append(colored(axle_cylinder(xx, 0, 0.51, 0.10, 2.12), DARK_STEEL))
        for side in (-1, 1):
            yy = side * 0.785
            wheel = axle_cylinder(xx, yy, 0.51, 0.415, 0.135)
            flange = axle_cylinder(xx, side * 0.704, 0.51, 0.446, 0.032)
            hub = axle_cylinder(xx, side * 0.889, 0.51, 0.14, 0.09)
            recess = axle_cylinder(xx, side * 0.858, 0.51, 0.295, 0.035)
            parts.extend(
                (
                    colored(wheel | flange, STEEL),
                    colored(recess, DARK_STEEL),
                    colored(hub, STEEL),
                    colored(block(xx, side * 1.02, 0.53, 0.42, 0.24, 0.27, 0.035), DARK_STEEL),
                )
            )
            spring = min_(
                *(torus(0.105, 0.027).translate(0, 0, zz) for zz in (0, 0.066, 0.132, 0.198))
            )
            parts.append(colored(spring.translate(xx, side * 1.04, 0.73), STEEL))
    for side in (-1, 1):
        parts.extend(
            (
                colored(block(x, side * 1.035, 0.68, 2.25, 0.14, 0.22, 0.06), DARK_STEEL),
                colored(cylinder(0.29, 0.20).translate(x, side * 0.75, 0.89), SEAL),
                rail_line((x - 0.62, side * 1.14, 0.59), (x + 0.58, side * 1.14, 0.90), 0.055),
            )
        )
    return min_(*parts)


def cab():
    stations = (
        (0.76, -9.13, -7.0, 0.91),
        (1.06, -9.63, -7.0, 1.23),
        (1.70, -9.70, -7.0, 1.39),
        (2.48, -9.48, -7.0, 1.38),
        (3.14, -9.12, -7.0, 1.31),
        (3.53, -8.71, -7.0, 1.13),
        (3.74, -8.16, -7.0, 0.84),
    )
    outer = rectangle(2, 2, 0.23).loft(
        tuple(
            LoftStation(z, offset=((front + back) / 2, 0), scale=((back - front) / 2, half_width))
            for z, front, back, half_width in stations
        )
    )
    parts = [colored(outer, NAVY)]
    yellow_mask = front_panel(
        (
            (-0.98, 1.25),
            (-1.22, 1.50),
            (-1.24, 2.72),
            (-1.02, 3.27),
            (-0.77, 3.41),
            (0.77, 3.41),
            (1.02, 3.27),
            (1.24, 2.72),
            (1.22, 1.50),
            (0.98, 1.25),
        )
    )
    windscreen = (
        (-0.80, 1.98),
        (-1.00, 2.13),
        (-1.03, 3.02),
        (-0.87, 3.21),
        (0.87, 3.21),
        (1.03, 3.02),
        (1.00, 2.13),
        (0.80, 1.98),
    )
    glass_mask = front_panel(windscreen)
    skin = outer.offset(0.014) - outer.offset(-0.015)
    parts.extend(
        (
            colored(skin & yellow_mask, YELLOW),
            colored((outer.offset(0.028) - outer) & glass_mask.offset(0.055), SEAL),
            colored((outer.offset(0.039) - outer.offset(0.015)) & glass_mask, GLASS),
        )
    )
    for side in (-1, 1):
        lamp_mask = front_panel(
            ((side * 0.82, 1.42), (side * 1.08, 1.54), (side * 1.08, 1.90), (side * 0.93, 1.83))
        )
        parts.append(colored((outer.offset(0.044) - outer) & lamp_mask, SEAL))
        for z, radius in ((1.57, 0.065), (1.75, 0.040)):
            xx = -9.72 if z < 1.7 else -9.70
            parts.append(
                colored(
                    cylinder(radius, 0.065).rotate(math.pi / 2, "y").translate(xx, side * 0.95, z),
                    WHITE if z < 1.7 else RED,
                )
            )
        parts.append(colored(side_panel(-7.70, side * 1.375, 2.70, 0.68, 0.77, 0.035, 0.12), SEAL))
        parts.append(colored(side_panel(-7.69, side * 1.399, 2.70, 0.56, 0.64, 0.015, 0.08), GLASS))
        parts.extend(
            (
                rail_line((-9.55, side * 0.65, 2.08), (-9.34, side * 0.52, 2.61), 0.016, SEAL),
                rail_line((-9.39, side * 0.53, 2.50), (-9.09, side * 0.74, 2.98), 0.020, SEAL),
                colored(block(-7.41, side * 1.47, 3.0, 0.18, 0.16, 0.28, 0.045), NAVY),
            )
        )
    parts.extend(
        (
            colored(block(-9.05, 0, 3.28, 0.09, 0.67, 0.18, 0.04), SEAL),
            colored(block(-9.102, 0, 3.28, 0.015, 0.45, 0.032), AMBER),
            colored(block(-9.55, 0, 1.02, 0.49, 1.23, 0.37, 0.095), SEAL),
            colored(block(-9.86, 0, 0.99, 0.72, 0.19, 0.17, 0.04), DARK_STEEL),
            colored(block(-10.20, 0, 1.0, 0.20, 0.49, 0.33, 0.045), STEEL),
            colored(
                cylinder(0.135, 0.045).rotate(math.pi / 2, "y").translate(-10.32, 0, 1.0), SEAL
            ),
            rail_line((-9.88, -0.37, 1.12), (-9.92, -0.39, 0.77), 0.035, SEAL),
            rail_line((-9.88, 0.37, 1.12), (-9.92, 0.39, 0.77), 0.035, SEAL),
        )
    )
    serial = lettering("701 042", 0.025).rotate(math.pi / 2, "z").translate(-9.745, 0, 1.52)
    parts.append(colored(serial, NAVY))
    return min_(*parts)


WINDOWS = (
    (-6.00, 1.18),
    (-2.85, 1.40),
    (-1.28, 1.40),
    (0.29, 1.40),
    (1.85, 1.40),
    (5.26, 1.40),
    (6.84, 1.40),
)
DOORS = (-4.55, 3.63)


def carriage(cutaway=False):
    outer = rectangle(2.78, 2.84, 0.34).on(Plane.yz()).extrude(15.75).translate(0.65, 0, 2.32)
    hollow = rectangle(2.51, 2.56, 0.27).on(Plane.yz()).extrude(15.95).translate(0.65, 0, 2.32)
    apertures = min_(*(side_panel(x, 0, 2.56, w, 1.04, 3.1, 0.09) for x, w in WINDOWS))
    body = (outer - hollow) - apertures
    parts = [colored(body, NAVY)]
    silver_upper = outer & block(0.65, 0, 3.44, 16, 3, 0.59)
    silver_lower = outer & block(1.2, 0, 1.44, 14.7, 3, 0.49)
    parts.extend((colored(silver_upper, SILVER), colored(silver_lower, SILVER)))
    side_details = []
    for side in (-1, 1):
        yy = side * 1.392
        for x, width in WINDOWS:
            frame = side_panel(x, yy, 2.56, width + 0.085, 1.125, 0.038, 0.105)
            cut = side_panel(x, yy, 2.56, width - 0.025, 1.015, 0.08, 0.075)
            side_details.extend(
                (
                    colored(frame - cut, SEAL),
                    colored(
                        side_panel(x, side * 1.386, 2.56, width - 0.04, 1.0, 0.016, 0.075), GLASS
                    ),
                    colored(block(x, side * 1.402, 2.94, width - 0.22, 0.011, 0.018), STEEL),
                )
            )
        for x in DOORS:
            side_details.append(colored(side_panel(x, yy, 2.22, 1.55, 2.40, 0.04, 0.08), SEAL))
            for leaf in (-1, 1):
                xx = x + leaf * 0.374
                side_details.extend(
                    (
                        colored(
                            side_panel(xx, side * 1.425, 2.22, 0.728, 2.34, 0.024, 0.065), SILVER
                        ),
                        colored(side_panel(xx, side * 1.442, 2.64, 0.35, 1.23, 0.02, 0.135), SEAL),
                        colored(
                            side_panel(xx, side * 1.455, 2.64, 0.288, 1.145, 0.018, 0.112), GLASS
                        ),
                        colored(block(xx, side * 1.464, 1.40, 0.64, 0.015, 0.025), STEEL),
                    )
                )
            side_details.extend(
                (
                    colored(block(x, side * 1.49, 1.045, 1.55, 0.23, 0.09, 0.016), STEEL),
                    colored(axle_cylinder(x + 0.87, side * 1.43, 1.94, 0.055, 0.032), YELLOW),
                    colored(axle_cylinder(x + 0.87, side * 1.453, 1.94, 0.031, 0.018), CYAN),
                )
            )
        side_details.extend(
            (
                colored(block(0.65, side * 1.345, 1.02, 15.72, 0.03, 0.036), YELLOW),
                colored(block(0.65, side * 1.39, 3.27, 15.5, 0.025, 0.035), STEEL),
                colored(side_panel(-0.3, side * 1.408, 1.59, 3.9, 0.55, 0.021, 0.10), NAVY),
            )
        )
        logo = lettering("SWR", 0.06)
        if side > 0:
            logo = logo.rotate(math.pi, "z")
        side_details.append(colored(logo.translate(-0.20, side * 1.439, 1.43), WHITE))
        for shift in (0, 0.21):
            xx = -1.48 - shift
            arrow = (
                polygon(
                    (
                        (xx - 0.12, 1.48),
                        (xx + 0.03, 1.67),
                        (xx - 0.12, 1.86),
                        (xx + 0.02, 1.86),
                        (xx + 0.18, 1.67),
                        (xx + 0.02, 1.48),
                    )
                )
                .on(Plane.xz(y=side * 1.446))
                .extrude(0.02)
            )
            side_details.append(colored(arrow, CYAN))
    roof_details = []
    for x in (-2.6, 4.8):
        roof_details.extend(
            (
                colored(block(x, 0, 3.80, 2.15, 1.65, 0.20, 0.08), SILVER),
                colored(block(x, 0, 3.912, 1.54, 1.25, 0.022, 0.01), DARK_STEEL),
                colored(
                    block(0, 0, 0, 0.04, 1.24, 0.025)
                    .linear_pattern(13, 0.112, 0, 0)
                    .translate(x - 0.68, 0, 3.936),
                    STEEL,
                ),
            )
        )
    for side in (-1, 1):
        roof_details.append(rail_line((-6.9, side * 1.05, 3.62), (8.30, side * 1.05, 3.62), 0.022))
    shell_parts = min_(*parts, *side_details, *roof_details)
    if cutaway:
        opening = block(0.90, -0.87, 3.0, 14.5, 2.90, 3.0)
        shell_parts = shell_parts - opening
    underfloor = [
        colored(block(0.65, 0, 0.99, 15.5, 2.32, 0.21, 0.055), DARK_STEEL),
        bogie(-5.70),
        bogie(6.05),
    ]
    for x, length in ((-2.60, 1.75), (-0.15, 2.4), (2.54, 1.75)):
        underfloor.append(colored(block(x, 0, 0.64, length, 1.89, 0.54, 0.055), SILVER))
        for side in (-1, 1):
            underfloor.append(
                colored(
                    block(0, 0, 0, 0.032, 0.02, 0.31)
                    .linear_pattern(10, 0.11, 0, 0)
                    .translate(x - 0.50, side * 0.955, 0.65),
                    DARK_STEEL,
                )
            )
    return min_(shell_parts, interior(), cab(), *underfloor)


def gangway():
    ring = rectangle(1.86, 2.49, 0.20) - rectangle(1.47, 2.15, 0.13)
    folds = (
        ring.on(Plane.yz()).extrude(0.055).linear_pattern(8, 0.10, 0, 0).translate(8.55, 0, 2.30)
    )
    return min_(
        colored(folds, SEAL),
        colored(block(8.92, 0, 1.10, 1.0, 1.48, 0.14), STEEL),
        colored(block(8.92, 0, 0.78, 1.0, 0.24, 0.23), DARK_STEEL),
    )


def track():
    mid, length = 8.9, 43.7
    parts = [colored(block(mid, 0, -0.29, length, 3.85, 0.26, 0.12), BALLAST)]
    sleeper = colored(block(0, 0, -0.09, 0.24, 2.58, 0.16, 0.035), SLEEPER)
    parts.append(sleeper.linear_pattern(69, 0.63, 0, 0).translate(-12.52, 0, 0))
    for side in (-1, 1):
        y = side * 0.755
        parts.extend(
            (
                colored(block(mid, y, 0.005, length, 0.15, 0.025), DARK_STEEL),
                colored(block(mid, y, 0.049, length, 0.028, 0.077), DARK_STEEL),
                colored(block(mid, y, 0.092, length, 0.068, 0.042, 0.012), STEEL),
            )
        )
    return min_(*parts)


if VIEW == "interior":
    SCENE = min_(
        interior(),
        colored(block(0.65, 1.32, 2.30, 15.75, 0.13, 2.50, 0.04), SHELL),
        colored(block(0.65, 0, 3.57, 15.75, 2.55, 0.10, 0.04), SHELL),
    )
else:
    leading_car = carriage(cutaway=VIEW == "cutaway")
    trailing_car = (
        (leading_car if VIEW == "exterior" else carriage())
        .rotate(math.pi, "z")
        .translate(17.8, 0, 0)
    )
    SCENE = min_(leading_car, trailing_car, gangway(), track())

MODEL = Part(SCENE, materials=PART_MATERIALS)
SCENE = Scene(south_western_railway_train=MODEL)
