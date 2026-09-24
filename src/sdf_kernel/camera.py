"""Shared camera orientation helpers."""

from math import isfinite, sqrt


def _orientation(value):
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("orientation must be four finite numbers (w, x, y, z)") from exc
    if len(result) != 4 or not all(isfinite(component) for component in result):
        raise ValueError("orientation must be four finite numbers (w, x, y, z)")
    length = sqrt(sum(component * component for component in result))
    if length < 1e-8:
        raise ValueError("orientation must be a nonzero quaternion")
    return tuple(component / length for component in result)


def _orientation_up(orientation):
    w, x, y, z = orientation
    return (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
