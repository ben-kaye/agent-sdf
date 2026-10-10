"""A two-cheek forked link with a shared, hollow center sleeve.

Dimensions are proportional: the reference drawings do not specify a scale.
X follows the link, Y crosses the fork, and Z is up.
"""

from sdf_kernel import Assembly, Curve, Material, Part, Plane, Scene, circle

# The broad arched top and shallow V at the bottom follow the supplied section.
CHEEK_OUTLINE = (
    Curve(-5.22, 0)
    .quad_to((-5.22, 0.59), (-4.42, 0.59), 168)
    .line_to(-3.56, 0.59)
    .quad_to((-3.38, 0.59), (-2.95, 0.79), 164)
    .quad_to((-1.12, 1.64), (0, 1.64), 168)
    .quad_to((1.12, 1.64), (2.95, 0.79), 168)
    .quad_to((3.38, 0.59), (3.56, 0.59), 164)
    .line_to(4.42, 0.59)
    .quad_to((5.22, 0.59), (5.22, 0), 168)
    .quad_to((5.22, -0.59), (4.42, -0.59), 168)
    .line_to(3.56, -0.59)
    .line_to(2.95, -0.69)
    .quad_to((1.00, -1.02), (0, -1.02), 168)
    .quad_to((-1.00, -1.02), (-2.95, -0.69), 168)
    .line_to(-3.56, -0.59)
    .line_to(-4.42, -0.59)
    .quad_to((-5.22, -0.59), (-5.22, 0), 168)
    .close()
)

# The openings narrow toward the end eyes and leave a web around the sleeve.
LEFT_WINDOW = (
    Curve(-2.74, -0.03)
    .quad_to((-2.74, 0.14), (-2.55, 0.19), 164)
    .line_to(-1.41, 0.60)
    .quad_to((-1.15, 0.71), (-1.23, 0.38), 165)
    .quad_to((-1.30, 0.10), (-1.18, -0.27), 164)
    .quad_to((-1.15, -0.42), (-1.36, -0.39), 164)
    .line_to(-2.55, -0.33)
    .quad_to((-2.74, -0.30), (-2.74, -0.03), 164)
    .close()
)
RIGHT_WINDOW = (
    Curve(1.25, 0.57)
    .quad_to((1.16, 0.81), (1.42, 0.67), 165)
    .line_to(2.60, 0.27)
    .quad_to((2.85, 0.18), (2.85, -0.02), 164)
    .quad_to((2.85, -0.28), (2.65, -0.31), 164)
    .line_to(1.34, -0.41)
    .quad_to((1.12, -0.45), (1.19, -0.27), 164)
    .quad_to((1.30, 0.12), (1.25, 0.57), 164)
    .close()
)

CENTER = (0, 0.24)
BORE_RADIUS = 0.61
SLEEVE_RADIUS = 0.84
END_EYE_X = 4.42
END_BORE_RADIUS = 0.36

CHEEK_PROFILE = (
    CHEEK_OUTLINE
    - LEFT_WINDOW
    - RIGHT_WINDOW
    - circle(SLEEVE_RADIUS).translate(*CENTER)
    - circle(END_BORE_RADIUS).translate(-END_EYE_X, 0)
    - circle(END_BORE_RADIUS).translate(END_EYE_X, 0)
)


def make_link():
    cheek_thickness = 0.23
    cheek_gap = 1.35
    cheek_center_y = (cheek_gap + cheek_thickness) / 2
    outer_face_y = cheek_gap / 2 + cheek_thickness

    # Extrusion is centered on the sketch plane.
    rear_cheek = CHEEK_PROFILE.on(Plane.xz(y=cheek_center_y)).extrude(cheek_thickness)
    front_cheek = CHEEK_PROFILE.on(Plane.xz(y=-cheek_center_y)).extrude(cheek_thickness)

    sleeve_profile = (circle(SLEEVE_RADIUS) - circle(BORE_RADIUS)).translate(*CENTER)
    sleeve = sleeve_profile.on(Plane.xz()).extrude(2 * outer_face_y)

    link = Assembly()
    link.add(
        "cheeks",
        Part(
            front_cheek | rear_cheek,
            material=Material(
                "satin_aluminum", base_color="#d3dddf", metallic=0.12, roughness=0.68
            ),
        ),
    )
    link.add(
        "sleeve",
        Part(
            sleeve,
            material=Material("steel_bushing", base_color="#b9c4c7", metallic=0.3, roughness=0.48),
        ),
    )
    return link


MODEL = make_link()
SCENE = Scene(link=MODEL)
