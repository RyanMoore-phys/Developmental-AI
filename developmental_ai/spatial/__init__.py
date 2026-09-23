"""Representations that outlive the frame they came from.

A LIGHT PACKAGE ON PURPOSE. `core/__init__` imports the whole developmental
loop, which imports gymnasium — so anything living under `core/` is
unimportable on a machine without the Minecraft stack, which is exactly the
machine these contracts most need to run on. The only dependency here is
numpy.
"""
from .memory import (
    EgocentricOccupancy, SuccessorMap, WalkabilityMap, rotate_translate)

__all__ = ["EgocentricOccupancy", "SuccessorMap", "WalkabilityMap",
           "rotate_translate"]
