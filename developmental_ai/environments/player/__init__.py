"""The Player Adapter — SkyBot plays any game through a screen, speakers,
a keyboard and a mouse (docs/UNIVERSAL_GAMES.md §11)."""
from developmental_ai.environments.player.adapter import PlayerAdapter, RealBackends  # noqa: F401
from developmental_ai.environments.player.controller import (  # noqa: F401
    ControllerError, Slot, parse_controller, validate)
