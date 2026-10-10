"""A sea-green semi-hollow electric guitar, built entirely from public geometry."""

from math import pi, sin

from agent_sdf import (
    Curve,
    Material,
    Part,
    Scene,
    box,
    capsule,
    circle,
    cylinder,
    min_,
    polygon,
    torus,
)


def make_guitar():
    groups = {}

    def add(material, *shapes):
        groups.setdefault(material, []).extend(shapes)

    outline = (
        Curve(0, -2.35)
        .quad_to((-1.65, -2.4), (-1.76, -1.28))
        .quad_to((-1.82, -0.65), (-1.30, 0.05))
        .quad_to((-1.01, 0.48), (-1.31, 1.10))
        .quad_to((-1.56, 1.87), (-0.72, 2.16))
        .quad_to((-0.91, 1.24), (-0.36, 1.22))
        .line_to(0.36, 1.22)
        .quad_to((0.80, 1.25), (0.68, 1.91))
        .quad_to((1.53, 1.61), (1.26, 0.96))
        .quad_to((0.96, 0.40), (1.34, -0.04))
        .quad_to((1.95, -0.86), (1.66, -1.56))
        .quad_to((1.35, -2.4), (0, -2.35))
        .sample(0.04)
    )
    outline = polygon(outline)
    # Three inset rounded layers produce the rim, binding, and carved-looking top.
    back = outline.offset(-0.07).extrude(0.29).offset(0.075).translate(0, 0, -0.03)
    binding = outline.offset(-0.035).extrude(0.045).translate(0, 0, 0.168)
    top = outline.offset(-0.13).extrude(0.055).offset(0.065).translate(0, 0, 0.22)

    def soundhole(x, mirror):
        points = []
        for side, indices in ((1, range(17)), (-1, range(16, -1, -1))):
            for i in indices:
                t = i / 16
                center_x = x + mirror * (0.14 * sin(2 * pi * t) + 0.11 * (t - 0.5))
                points.append((center_x + side * 0.047, -0.82 + 1.3 * t))
        slot = polygon(points)
        for t in (0, 1):
            slot |= circle(0.105).translate(x + mirror * 0.11 * (t - 0.5), -0.82 + 1.3 * t)
        return slot.extrude(0.34).translate(0, 0, 0.25)

    holes = soundhole(-0.99, -1) | soundhole(0.99, 1)
    add("teal", top - holes)
    add("dark_teal", back)
    add("ivory", binding - holes)
    add("black", outline.offset(-0.20).extrude(0.015).translate(0, 0, 0.085))

    neck_profile = polygon([(-0.35, 1.03), (0.35, 1.03), (0.235, 6.30), (-0.235, 6.30)])
    add("wood", neck_profile.extrude(0.22).offset(0.035).translate(0, 0, 0.16))
    add("ivory", neck_profile.extrude(0.07).translate(0, 0, 0.31))
    add("ebony", neck_profile.offset(-0.024).extrude(0.07).translate(0, 0, 0.355))

    head = (
        Curve(-0.235, 6.24)
        .quad_to((-0.27, 6.54), (-0.43, 6.68))
        .line_to(-0.38, 7.91)
        .quad_to((-0.23, 8.14), (0, 8.02))
        .quad_to((0.23, 8.14), (0.38, 7.91))
        .line_to(0.43, 6.68)
        .quad_to((0.27, 6.54), (0.235, 6.24))
        .line_to(-0.235, 6.24)
        .sample(0.025)
    )
    head = polygon(head)
    add("wood", head.extrude(0.20).translate(0, 0, 0.16))
    add("ivory", head.offset(-0.025).extrude(0.025).translate(0, 0, 0.271))
    add("ebony", head.offset(-0.060).extrude(0.025).translate(0, 0, 0.289))
    diamond = polygon([(0, -0.12), (0.08, 0), (0, 0.12), (-0.08, 0)])
    add("pearl", diamond.extrude(0.01).translate(0, 7.58, 0.309))

    nut_y, bridge_y = 6.27, -0.84
    length = nut_y - bridge_y
    fret_positions = [nut_y - length * (1 - 2 ** (-i / 12)) for i in range(23)]
    for i, y in enumerate(fret_positions[1:], 1):
        half_width = 0.235 + (6.30 - y) / 5.27 * 0.115 - 0.031
        add("nickel", capsule((-half_width, y, 0.401), (half_width, y, 0.401), 0.013))
        if i in (3, 5, 7, 9, 12, 15, 17, 19, 21):
            midpoint = (y + fret_positions[i - 1]) / 2
            for x in (-0.095, 0.095) if i == 12 else (0,):
                add("pearl", cylinder(0.035, 0.009).translate(x, midpoint, 0.394))
    add("ivory", box(0.49, 0.075, 0.08, corner_radius=0.009).translate(0, nut_y, 0.404))

    for y in (0.96, -0.01):
        add("black", box(0.94, 0.46, 0.065, corner_radius=0.025).translate(0, y, 0.340))
        add("ivory", box(0.83, 0.35, 0.075, corner_radius=0.035).translate(0, y, 0.385))
        add("gold", box(0.78, 0.16, 0.075, corner_radius=0.035).translate(0, y - 0.08, 0.412))
        add("black", box(0.78, 0.12, 0.075, corner_radius=0.025).translate(0, y + 0.09, 0.412))
        for x in (-0.29, -0.174, -0.058, 0.058, 0.174, 0.29):
            add("nickel", cylinder(0.027, 0.01).translate(x, y + 0.09, 0.456))
        for x in (-0.419, 0.419):
            screw = cylinder(0.025, 0.018) - box(0.036, 0.009, 0.02).translate(0, 0, 0.009)
            add("nickel", screw.translate(x, y, 0.382))

    for y, width in ((bridge_y, 0.87), (-1.29, 0.95)):
        add("gold", box(width, 0.15, 0.12, corner_radius=0.04).translate(0, y, 0.405))
        for x in (-width / 2 + 0.035, width / 2 - 0.035):
            add("nickel", cylinder(0.043, 0.055).translate(x, y, 0.475))

    for i in range(6):
        bridge_x = (i - 2.5) * 0.11
        nut_x = (i - 2.5) * 0.076
        radius = 0.0065 + (5 - i) * 0.0009
        add("nickel", capsule((bridge_x, -1.29, 0.475), (bridge_x, bridge_y, 0.51), radius))
        add("nickel", capsule((bridge_x, bridge_y, 0.51), (nut_x, nut_y, 0.46), radius))
        add(
            "gold",
            box(0.070, 0.075, 0.045, corner_radius=0.009).translate(bridge_x, bridge_y, 0.48),
        )
        side = -1 if i < 3 else 1
        post_y = 6.79 + (i if i < 3 else 5 - i) * 0.42
        post_x = side * 0.27
        add("nickel", capsule((nut_x, nut_y, 0.46), (post_x, post_y, 0.365), radius))
        add("gold", cylinder(0.075, 0.05).translate(post_x, post_y, 0.325))
        add("nickel", cylinder(0.033, 0.105).translate(post_x, post_y, 0.355))
        add("gold", capsule((post_x, post_y, 0.17), (side * 0.52, post_y, 0.17), 0.03))
        add(
            "ivory", box(0.16, 0.20, 0.10, corner_radius=0.045).translate(side * 0.57, post_y, 0.17)
        )

    for x, y in ((0.88, -1.15), (1.16, -1.53), (0.64, -1.74)):
        add("gold", cylinder(0.12, 0.095).translate(x, y, 0.39))
        add("black", cylinder(0.096, 0.028).translate(x, y, 0.45))
        add("ivory", box(0.009, 0.058, 0.005).translate(x, y + 0.023, 0.467))
        add("gold", torus(0.107, 0.009).translate(x, y, 0.45))
    add("gold", cylinder(0.105, 0.025).translate(-0.73, 1.22, 0.335))
    add("nickel", capsule((-0.73, 1.22, 0.345), (-0.70, 1.27, 0.49), 0.021))
    add("ivory", capsule((-0.70, 1.27, 0.47), (-0.68, 1.30, 0.55), 0.035))
    add("gold", cylinder(0.065, 0.12).rotate(pi / 2, axis="x").translate(0, -2.40, -0.025))

    finishes = {
        "teal": ("#277e79", 0.30, 0.25),
        "dark_teal": ("#123c3b", 0.15, 0.33),
        "ivory": ("#eee1bd", 0.05, 0.36),
        "wood": ("#75472e", 0.05, 0.42),
        "ebony": ("#201c1b", 0.03, 0.40),
        "black": ("#101619", 0.15, 0.35),
        "pearl": ("#e2efdf", 0.20, 0.27),
        "gold": ("#d7b366", 0.75, 0.28),
        "nickel": ("#d4dddf", 0.82, 0.23),
    }
    scene = Scene()
    for name, shapes in groups.items():
        color, metallic, roughness = finishes[name]
        scene.add(
            name,
            Part(
                min_(*shapes),
                material=Material(name, base_color=color, metallic=metallic, roughness=roughness),
            ),
        )
    return scene


SCENE = make_guitar()
