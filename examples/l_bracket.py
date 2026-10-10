"""A sliding V-jaw fixture; dimensions are millimetres before display scaling."""

import math
import os

from sdf_kernel import (
    AABB,
    Assembly,
    Material,
    Part,
    Placement,
    Plane,
    Scene,
    circle,
    custom_sdf,
    cylinder,
    max_,
    min_,
    polygon,
    rectangle,
)

SCALE = 0.025
SCREW_SPACING = 36
JAW_HEIGHT = 60
TRAVEL_LIMIT = 8
MOUNT_X = 46
MOUNT_Y = -26
JAW_BORE_Y = 11

BLUE = Material("anodized_blue", base_color="#326b8a", metallic=0.65, roughness=0.33)
ORANGE = Material("anodized_copper", base_color="#da823b", metallic=0.63, roughness=0.3)
BLACK = Material("black_oxide_screws", base_color="#343b43", metallic=0.75, roughness=0.3)
STEEL = Material("machined_steel", base_color="#bdc7ce", metallic=0.85, roughness=0.25)


def front_plane(y=0, z=0):
    return Plane(origin=(0, y, z), normal=(0, -1, 0), x_direction=(1, 0, 0))


def paired(shape):
    return shape.translate(-SCREW_SPACING / 2, 0, 0).linear_pattern(2, SCREW_SPACING, 0, 0)


def bracket_shape():
    foot = rectangle(108, 82, corner_radius=6).extrude(10).translate(0, -4, 5)
    upright = rectangle(84, 77, corner_radius=4).on(front_plane(24, 47.5)).extrude(12)

    rib_profile = polygon([(-22, 9), (24, 9), (24, 74), (18, 74), (-22, 16)])
    rib_window = polygon([(-9, 21), (13, 21), (13, 52)]).offset(1.5)
    ribs = (rib_profile - rib_window).on(Plane.yz(x=-36)).extrude(7)
    ribs = ribs.linear_pattern(2, 72, 0, 0)
    blank = foot | upright | ribs

    mount_slot = rectangle(7, 19, corner_radius=3.5).extrude(14)
    mount_slots = mount_slot.translate(-MOUNT_X, MOUNT_Y, 5).linear_pattern(2, 2 * MOUNT_X, 0, 0)
    mount_slots = mount_slots.linear_pattern(2, 0, 46, 0)
    tray = rectangle(51, 29, corner_radius=5).extrude(4).translate(0, -18, 9)
    drain = cylinder(3, 14).translate(0, -18, 5)

    adjustment_slot = rectangle(6.6, 36, corner_radius=3.3)
    adjustment_slots = paired(adjustment_slot.on(front_plane(24, JAW_HEIGHT)).extrude(16))
    guide = rectangle(10.5, 50, corner_radius=1.5).on(front_plane(19.8, 59)).extrude(4)
    nut_track = rectangle(10.4, 43, corner_radius=1).on(front_plane(29, 60)).extrude(4)
    nut_tracks = paired(nut_track)
    accessory_bores = circle(2.5).on(front_plane(24, 78)).extrude(16)
    accessory_bores = accessory_bores.translate(-30, 0, 0).linear_pattern(2, 60, 0, 0)

    return (
        blank - mount_slots - tray - drain - adjustment_slots - guide - nut_tracks - accessory_bores
    )


def jaw_shape():
    body = rectangle(60, 36, corner_radius=3).on(front_plane(JAW_BORE_Y)).extrude(14)
    tongue = rectangle(10, 30, corner_radius=1.5).on(front_plane(19.45)).extrude(3.9)
    bores = paired(circle(3.3).on(front_plane(JAW_BORE_Y)).extrude(18))
    counterbores = paired(circle(6.5).on(front_plane(5.5)).extrude(5))
    face_pocket = rectangle(12, 17, corner_radius=2).on(front_plane(5)).extrude(4)
    v_saddle = polygon([(5, 19), (17, 19), (11, 12)]).on(Plane.yz()).extrude(64)
    return (body | tongue) - bores - counterbores - face_pocket - v_saddle


def hexagon(across_flats):
    radius = across_flats / math.sqrt(3)
    return polygon(
        [
            (
                radius * math.cos(math.pi / 6 + i * math.pi / 3),
                radius * math.sin(math.pi / 6 + i * math.pi / 3),
            )
            for i in range(6)
        ]
    )


def socket_screw(length):
    # Local +Z points from the bearing face into the receiving part.
    turning_profile = polygon(
        [
            (0, -6),
            (4.4, -6),
            (5, -5.4),
            (5, -0.4),
            (4.6, 0),
            (3, 0),
            (3, 5),
            (2.45, 5),
            (2.45, length - 1),
            (1.8, length),
            (0, length),
        ]
    )
    turned = turning_profile.revolve(axis="y").rotate(math.pi / 2, axis="x")
    socket = hexagon(5).extrude(3.8).translate(0, 0, -4.5)
    socket_lead = circle(3.05).extrude(0.5).translate(0, 0, -5.95)

    # A one-millimetre pitch with rounded crests. Polynomial trig keeps this field
    # expressible using scalar arithmetic for both CPU evaluation and GPU rendering.
    def thread_field(x, y, z):
        angle = min_(max_(z, -0.5), 0.5) * (2 * math.pi)
        a2 = angle * angle
        sine = angle * (1 + a2 * (-1 / 6 + a2 * (1 / 120 + a2 * (-1 / 5040 + a2 / 362880))))
        cosine = 1 + a2 * (
            -1 / 2 + a2 * (1 / 24 + a2 * (-1 / 720 + a2 * (1 / 40320 - a2 / 3628800)))
        )
        radius = (x * x + y * y) ** 0.5
        crest = (x * cosine + y * sine) / max_(radius, 2)
        radial = (radius - 2.725 - 0.275 * crest) / 2.4
        return max_(radial, abs(z) - 0.5)

    thread = custom_sdf(
        thread_field,
        bounds=AABB((-3, -3, -0.5), (3, 3, 0.5)),
        lipschitz=1,
    )
    thread = thread.linear_pattern(int(length) - 6, 0, 0, 1).translate(0, 0, 5.5)
    return (turned | thread) - socket - socket_lead


def washer_shape():
    return (circle(6.2) - circle(3.3)).extrude(1.2)


def nut_shape():
    return hexagon(10).extrude(5) - cylinder(2.6, 8)


def make_fixture(explode=0, jaw_travel=0):
    if not 0 <= explode <= 1:
        raise ValueError("explode must be between 0 and 1")
    if not -TRAVEL_LIMIT <= jaw_travel <= TRAVEL_LIMIT:
        raise ValueError(f"jaw_travel must be between {-TRAVEL_LIMIT} and {TRAVEL_LIMIT} mm")

    def component(shape, material, **datums):
        return Part(
            shape.scale(SCALE),
            material=material,
            datums={name: tuple(v * SCALE for v in point) for name, point in datums.items()},
        )

    def put(assembly, name, part, position=(0, 0, 0)):
        assembly.add(name, part.at(Placement(tuple(v * SCALE for v in position))))

    fixture = Assembly()
    bracket = component(
        bracket_shape(),
        BLUE,
        guide_center=(0, 18, JAW_HEIGHT),
        floor=(0, 0, 0),
        left_mount=(-MOUNT_X, MOUNT_Y, 0),
        right_mount=(MOUNT_X, MOUNT_Y, 0),
    )
    jaw = component(
        jaw_shape(),
        ORANGE,
        mating_face=(0, 18, 0),
        saddle=(0, 11, 12),
        left_bore=(-SCREW_SPACING / 2, JAW_BORE_Y, 0),
        right_bore=(SCREW_SPACING / 2, JAW_BORE_Y, 0),
    )
    put(fixture, "bracket", bracket)
    guide_center = fixture.datum("bracket/guide_center")
    mating_face = jaw.datum("mating_face")
    jaw_position = tuple(
        guide - mating for guide, mating in zip(guide_center, mating_face, strict=True)
    )
    jaw_position = (
        jaw_position[0],
        jaw_position[1] - 34 * explode * SCALE,
        jaw_position[2] + jaw_travel * SCALE,
    )
    fixture.add("sliding_jaw", jaw.at(Placement(jaw_position)))

    jaw_fasteners = Assembly()
    jaw_screw = component(socket_screw(30).rotate(-math.pi / 2, axis="x"), BLACK)
    front_washer = component(washer_shape().rotate(math.pi / 2, axis="x"), STEEL)
    rear_nut = component(nut_shape().rotate(math.pi / 2, axis="x"), STEEL)
    for label in ("left", "right"):
        bore = fixture.datum(f"sliding_jaw/{label}_bore")
        x, jaw_z = bore[0] / SCALE, bore[2] / SCALE
        put(jaw_fasteners, f"{label}_screw", jaw_screw, (x, 6.8 - 63 * explode, jaw_z))
        put(jaw_fasteners, f"{label}_washer", front_washer, (x, 7.4 - 47 * explode, jaw_z))
        put(jaw_fasteners, f"{label}_captive_nut", rear_nut, (x, 29.5 + 19 * explode, jaw_z))
    fixture.add("jaw_hardware", jaw_fasteners)

    mounting = Assembly()
    mounting_screw = component(socket_screw(22).rotate(math.pi, axis="x"), BLACK)
    flat_washer = component(washer_shape(), STEEL)
    for label in ("left", "right"):
        mount = fixture.datum(f"bracket/{label}_mount")
        x, y = mount[0] / SCALE, mount[1] / SCALE
        put(mounting, f"{label}_screw", mounting_screw, (x, y, 11.2 + 27 * explode))
        put(mounting, f"{label}_washer", flat_washer, (x, y, 10.6 + 12 * explode))
    fixture.add("mounting_hardware", mounting)
    return fixture


EXPLODE = float(os.environ.get("SDF_FIXTURE_EXPLODE", "0"))
JAW_TRAVEL = float(os.environ.get("SDF_FIXTURE_TRAVEL", "0"))
MODEL = make_fixture(EXPLODE, JAW_TRAVEL)
SCENE = Scene(fixture=MODEL)
