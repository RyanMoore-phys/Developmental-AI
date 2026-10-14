"""SkyBot's controller for the Player Adapter: slots -> keys, buttons, mouse.

docs/UNIVERSAL_GAMES.md §11 (Player Adapter). A game YAML lists controller
slots; each slot is what SkyBot HOLDS while that slot is its decision:

    - {}                                   # slot 0: release everything
    - {keys: [Up]}                         # hold the Up arrow
    - {keys: [Up], buttons: [1]}           # walk + left mouse button
    - {mouse: {dx_per_s: 400}}             # turn: move the mouse at +400 px/s

Mouse is a held VELOCITY, not a jump: turning speed is the same whether
SkyBot decides 35x/s or 2x/s, so the consequences of an action scale with the
game time it is held — what the `tempo` sense assumes.

SAFETY, enforced here and again at compose time:
  * slot 0 is empty (release-all) in every game;
  * only names that appear in the YAML are ever sent (the whitelist);
  * OS / window-manager keys and chords are refused outright (Super, Print,
    XF86*, Alt+F4, Ctrl+Alt+anything) — a seat is its own X display, but a
    game or the window manager can still act on them (Alt+F4 closes the game);
  * ONE-WAY DOORS (CLAUDE.md §4.5): a `doors` entry names keys that open a
    state and keys that close it; if any slot can open it, some slot must be
    able to close it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Sequence, Tuple

# keys no slot may ever hold
FORBIDDEN_KEYS = frozenset({
    "Super_L", "Super_R", "Meta_L", "Meta_R", "Hyper_L", "Hyper_R",
    "Multi_key", "Print", "Sys_Req", "Break", "Menu", "Mode_switch",
    "ISO_Level3_Shift", "Num_Lock", "Caps_Lock", "Scroll_Lock",
})
FORBIDDEN_PREFIXES = ("XF86",)
_ALT = frozenset({"Alt_L", "Alt_R"})
_CTRL = frozenset({"Control_L", "Control_R"})
_FKEYS = frozenset(f"F{i}" for i in range(1, 25))
BUTTONS = frozenset({1, 2, 3})          # left, middle, right (no wheel yet)
_SLOT_FIELDS = frozenset({"keys", "buttons", "mouse"})
_MOUSE_FIELDS = frozenset({"dx_per_s", "dy_per_s"})


class ControllerError(ValueError):
    pass


@dataclass(frozen=True)
class Slot:
    keys: FrozenSet[str] = field(default_factory=frozenset)
    buttons: FrozenSet[int] = field(default_factory=frozenset)
    mouse_vx: float = 0.0               # pixels per second, held
    mouse_vy: float = 0.0

    @property
    def empty(self) -> bool:
        return (not self.keys and not self.buttons
                and self.mouse_vx == 0.0 and self.mouse_vy == 0.0)


def _check_key(name: Any, where: str) -> str:
    if not isinstance(name, str) or not name:
        raise ControllerError(f"{where}: key names are X keysym strings, got {name!r}")
    if name in FORBIDDEN_KEYS or name.startswith(FORBIDDEN_PREFIXES):
        raise ControllerError(f"{where}: {name!r} is an OS/window-manager key and is never sent")
    return name


def parse_controller(entries: Sequence[Dict[str, Any]]) -> List[Slot]:
    if not entries:
        raise ControllerError("controller is empty")
    slots: List[Slot] = []
    for i, e in enumerate(entries):
        where = f"controller slot {i}"
        e = dict(e or {})
        bad = set(e) - _SLOT_FIELDS
        if bad:
            raise ControllerError(f"{where}: unknown field(s) {sorted(bad)} "
                                  f"(allowed: keys, buttons, mouse)")
        keys = frozenset(_check_key(k, where) for k in (e.get("keys") or []))
        buttons = frozenset(int(b) for b in (e.get("buttons") or []))
        if not buttons <= BUTTONS:
            raise ControllerError(f"{where}: mouse buttons must be in {sorted(BUTTONS)}")
        m = dict(e.get("mouse") or {})
        if set(m) - _MOUSE_FIELDS:
            raise ControllerError(f"{where}: mouse takes dx_per_s / dy_per_s only")
        if keys & _ALT and keys & _FKEYS:
            raise ControllerError(f"{where}: Alt+F-key chords (Alt+F4 closes the game) are refused")
        if keys & _CTRL and keys & _ALT:
            raise ControllerError(f"{where}: Ctrl+Alt chords are refused")
        slots.append(Slot(keys, buttons, float(m.get("dx_per_s", 0.0)),
                          float(m.get("dy_per_s", 0.0))))
    if not slots[0].empty:
        raise ControllerError("controller slot 0 must be {} (release everything): "
                              "every game's slot 0 is its no-op")
    return slots


def whitelist(slots: Iterable[Slot]) -> Tuple[FrozenSet[str], FrozenSet[int]]:
    """Every key and button SkyBot can ever send in this game."""
    keys, buttons = set(), set()
    for s in slots:
        keys |= s.keys
        buttons |= s.buttons
    return frozenset(keys), frozenset(buttons)


def door_problems(slots: Sequence[Slot], doors: Sequence[Dict[str, Any]]) -> List[str]:
    """One-way doors: a state some slot can OPEN that no slot can CLOSE."""
    held = whitelist(slots)[0]
    out = []
    for d in doors or []:
        opens = set(d.get("opens") or [])
        closes = set(d.get("closes") or [])
        if not opens:
            out.append(f"door {d!r}: `opens` is empty")
            continue
        if opens & held and not (closes & held):
            out.append(f"one-way door: {sorted(opens & held)} can be pressed but "
                       f"none of its closers {sorted(closes)} is in any slot")
    return out


def validate(game: Dict[str, Any], action_slots: int) -> List[Slot]:
    """Parse + every compose-time check for a player game. Raises on failure."""
    slots = parse_controller(game.get("controller") or [])
    if len(slots) > int(action_slots):
        raise ControllerError(f"{len(slots)} controller slots > the brain's "
                              f"universal.action_slots={action_slots}")
    probs = door_problems(slots, game.get("doors") or [])
    if probs:
        raise ControllerError("; ".join(probs))
    return slots
