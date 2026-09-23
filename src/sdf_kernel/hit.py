"""Distance and material returned by a custom signed distance function."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    distance: object
    material: object = 0
