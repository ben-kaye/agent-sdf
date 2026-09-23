"""Single-frame morphological antialiasing for RGB renders.

This CPU SMAA variant uses local-contrast edge detection, searches for
staircase patterns, estimates their coverage analytically, and blends only
across those edges. It does not use the reference implementation's area and
search textures or temporal/subpixel modes.
"""

import numpy as np


def _edges(luma):
    """Detect top and left boundaries with SMAA's local contrast rejection."""
    padded = np.pad(luma, 2, mode="edge")
    center = padded[2:-2, 2:-2]
    top = np.abs(center - padded[1:-3, 2:-2])
    left = np.abs(center - padded[2:-2, 1:-3])
    right = np.abs(center - padded[2:-2, 3:-1])
    bottom = np.abs(center - padded[3:-1, 2:-2])
    two_top = np.abs(padded[1:-3, 2:-2] - padded[:-4, 2:-2])
    two_left = np.abs(padded[2:-2, 1:-3] - padded[2:-2, :-4])
    vertical_max = np.maximum.reduce((top, left, right, bottom, two_top))
    horizontal_max = np.maximum.reduce((top, left, right, bottom, two_left))
    top_edge = (top > 0.08) & (top >= 0.5 * vertical_max)
    left_edge = (left > 0.08) & (left >= 0.5 * horizontal_max)
    top_edge[0, :] = False
    left_edge[:, 0] = False
    return top_edge, left_edge


def _crossing(vertical, row, boundary):
    """Return which side of a horizontal edge a crossing edge continues on."""
    if not 0 < boundary < vertical.shape[1]:
        return 0
    above = vertical[row - 1, boundary]
    below = vertical[row, boundary]
    if above == below:
        return 0  # A two-pixel crossing is a corner, not a staircase.
    return -1 if above else 1


def _blend_weights(horizontal, vertical):
    """Find staircase spans and return weights on both sides of each edge."""
    height, width = horizontal.shape
    upper = np.zeros((height, width), dtype=np.float32)
    lower = np.zeros_like(upper)
    for row in range(1, height):
        active = np.flatnonzero(horizontal[row])
        if not active.size:
            continue
        breaks = np.flatnonzero(np.diff(active) > 1) + 1
        for run in np.split(active, breaks):
            start, end = int(run[0]), int(run[-1]) + 1
            crossings = [
                (x, sign) for x in range(start, end + 1) if (sign := _crossing(vertical, row, x))
            ]
            for (left, left_sign), (right, right_sign) in zip(
                crossings, crossings[1:], strict=False
            ):
                span = right - left
                if not 1 <= span <= 32 or left_sign == right_sign:
                    continue
                # Integrate the triangular coverage at four positions per pixel.
                positions = np.arange(span, dtype=np.float32)[:, None] + np.array(
                    (0.125, 0.375, 0.625, 0.875), dtype=np.float32
                )
                offsets = 0.5 * left_sign + (right_sign - left_sign) * 0.5 * positions / span
                upper[row, left:right] = np.maximum(
                    upper[row, left:right], np.maximum(-offsets, 0).mean(axis=1)
                )
                lower[row, left:right] = np.maximum(
                    lower[row, left:right], np.maximum(offsets, 0).mean(axis=1)
                )
    return upper, lower


def smaa(image):
    """Apply a conservative, single-frame SMAA variant to a uint8 RGB image."""
    rgb = image.astype(np.float32)
    luma = (rgb @ np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)) / 255
    top, left = _edges(luma)
    upper, lower = _blend_weights(top, left)
    left_side, right_side = _blend_weights(left.T, top.T)
    left_side, right_side = left_side.T, right_side.T

    up = np.zeros_like(upper)
    down = np.zeros_like(upper)
    up[1:] = lower[1:]
    down[:-1] = upper[1:]
    left_weight = np.zeros_like(upper)
    right_weight = np.zeros_like(upper)
    left_weight[:, 1:] = right_side[:, 1:]
    right_weight[:, :-1] = left_side[:, 1:]

    vertical = up + down
    horizontal = left_weight + right_weight
    use_vertical = vertical >= horizontal
    weights = np.where(use_vertical, vertical, horizontal)
    weights = np.minimum(weights, 0.5)
    output = rgb * (1 - weights[..., None])
    output[1:] += rgb[:-1] * (up[1:] * use_vertical[1:])[..., None]
    output[:-1] += rgb[1:] * (down[:-1] * use_vertical[:-1])[..., None]
    output[:, 1:] += rgb[:, :-1] * (left_weight[:, 1:] * ~use_vertical[:, 1:])[..., None]
    output[:, :-1] += rgb[:, 1:] * (right_weight[:, :-1] * ~use_vertical[:, :-1])[..., None]
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)
