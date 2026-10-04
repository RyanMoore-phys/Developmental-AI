"""Adapter-declared axis conventions, as DATA.

Nothing in the geometry core reads these; an environment adapter picks one
and attaches it to its frames (`FrameGraph.add_frame(..., convention=...)`).
Each declaration carries known-answer cases and is checked by `verify()` at
registration, because a convention table "is exactly the kind of thing that
looks right and is off by a sign" (sensors/heading.py, which this mirrors —
the parity is pinned in tests/_foundation_geometry_smoke.py).

Minecraft: x east, y up, z south (right-handed). Yaw 0 faces +z (south),
yaw 90 faces -x (west) — "yaw grows clockwise seen from above". The SIGN is
derived from the known answers, not from the word "clockwise" (see the note
in heading.py and contract F of tests/_oracle_isolation_smoke.py). Pitch +90
looks straight down.
"""

from __future__ import annotations

from typing import Dict

from .errors import DegenerateError
from .frames import AxisConvention

_REGISTRY: Dict[str, AxisConvention] = {}


def declare_convention(conv: AxisConvention) -> AxisConvention:
    bad = conv.verify()
    if bad:
        raise DegenerateError("convention fails its own known answers: " + "; ".join(bad))
    if conv.name in _REGISTRY:
        raise DegenerateError(f"convention {conv.name!r} already declared")
    _REGISTRY[conv.name] = conv
    return conv


def get_convention(name: str) -> AxisConvention:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown axis convention {name!r}; declared: {sorted(_REGISTRY)}")


def declared_conventions():
    return dict(_REGISTRY)


MINECRAFT = declare_convention(AxisConvention(
    "minecraft", up=(0.0, 1.0, 0.0), forward_at_zero=(0.0, 0.0, 1.0),
    yaw_sign=-1.0, pitch_sign=+1.0,
    declared_by="minecraft/minerl adapter (mirrors sensors/heading.py)",
    cases=[(0.0, 0.0, (0.0, 0.0, 1.0)),      # south, +z
           (90.0, 0.0, (-1.0, 0.0, 0.0)),    # west,  -x
           (180.0, 0.0, (0.0, 0.0, -1.0)),   # north, -z
           (270.0, 0.0, (1.0, 0.0, 0.0)),    # east,  +x
           (0.0, 90.0, (0.0, -1.0, 0.0)),    # straight down
           (0.0, -90.0, (0.0, 1.0, 0.0))]))  # straight up

# A second, unrelated declaration so the class is demonstrably not
# Minecraft-shaped: ROS REP-103 body frame (x forward, y left, z up).
ROS_FLU = declare_convention(AxisConvention(
    "ros_flu", up=(0.0, 0.0, 1.0), forward_at_zero=(1.0, 0.0, 0.0),
    yaw_sign=+1.0, pitch_sign=+1.0,
    declared_by="ROS REP-103",
    cases=[(0.0, 0.0, (1.0, 0.0, 0.0)),
           (90.0, 0.0, (0.0, 1.0, 0.0)),     # yaw CCW from above: left
           (0.0, 90.0, (0.0, 0.0, -1.0))]))  # positive pitch: nose down
