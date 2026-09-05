"""Mainhand-from-evidence smoke (2026-08-17).

THE BUG THIS ENCODES
    The agent could not perceive its own hand. Three layers stacked:

      1. `equipped_items` is NOT populated by MineRL 1.0's MCP-Reborn
         bridge — it reads "none" whatever is held (already documented in
         minerl_env.py, and it cost a day once).
      2. So a fallback derived the hand FROM THE INVENTORY — but it only
         ever asked "is the SPAWN TOOL still in the bag?"
         (`tool = self._start_tool`).
      3. The shipped skybot config sets `start_tool: null`. With `tool`
         None the guard `if isinstance(inv, dict) and tool:` is False, the
         whole branch is skipped, and `mainhand` can never be anything but
         "none".

    MEASURED on the live pod: `hand=none` in 160/160 samples across a
    7-hour run, while the agent carried 8+ items and spent 11% of its
    actions on `use`. Every consumer — the proprio has_tool sense, the
    chop-budget diagnosis, `tool_worn` — was reading a frozen constant.
    That is the "constant predicate survives for weeks" failure class this
    project keeps paying for.

THE FIX
    A flat inventory (item -> count, NO slot indices — see
    FlatInventoryObservation) cannot name the selected slot. The one ground
    truth available: an item that a `use` actually CONSUMED was, at that
    instant, in the hand. Nothing in TREECHOP_MACROS changes the selected
    slot, so it stays there while any remains. Retrospective and honest,
    exactly like the reach sense's "evidence" mode.

Run: PYTHONPATH=. python tests/_mainhand_evidence_smoke.py
"""
import os
import sys

sys.path.insert(0, ".")

import numpy as np

SRC = os.path.join("developmental_ai", "environments", "minerl_env.py")


def _derive(mainhand_in, inv, start_tool, last_placed):
    """The derivation, verbatim in behaviour from minerl_env._world_events."""
    out = {}
    if mainhand_in is not None:
        out["mainhand"] = mainhand_in
    if not out.get("mainhand") or out.get("mainhand") in ("none", "air"):
        tool = start_tool
        if isinstance(inv, dict) and tool:
            qty = inv.get(tool)
            if qty is not None:
                n = float(np.asarray(qty).flatten()[0])
                if n > 0:
                    out["mainhand"] = str(tool)
                    out["mainhand_derived"] = True
        if (not out.get("mainhand")
                or out.get("mainhand") in ("none", "air")):
            if last_placed and isinstance(inv, dict):
                q = inv.get(last_placed)
                if q is not None:
                    if float(np.asarray(q).flatten()[0]) > 0:
                        out["mainhand"] = str(last_placed)
                        out["mainhand_derived"] = True
    return out


def test_the_original_bug():
    """start_tool None + a full bag == blind, under the OLD rule."""
    inv = {"dirt": 42, "log": 3}
    # old behaviour == new behaviour with no placement evidence yet
    out = _derive("none", inv, None, None)
    assert out.get("mainhand") in (None, "none"), out
    print("  1. no start_tool + no evidence -> still 'none' (honest, not a "
          "guess): the old code could reach ONLY this state")


def test_evidence_names_the_hand():
    inv = {"dirt": 41, "log": 3}
    out = _derive("none", inv, None, "dirt")
    assert out["mainhand"] == "dirt", out
    assert out["mainhand_derived"] is True
    print("  2. a `use` that consumed dirt -> hand reads 'dirt' "
          "(the 160/160 blind spot is closed)")


def test_exhausted_stack_stops_claiming():
    """Placed the last one: the claim must lapse, not latch."""
    out = _derive("none", {"dirt": 0, "log": 3}, None, "dirt")
    assert out.get("mainhand") in (None, "none"), out
    out2 = _derive("none", {"log": 3}, None, "dirt")
    assert out2.get("mainhand") in (None, "none"), out2
    print("  3. stack exhausted / item gone -> reverts to 'none' "
          "(evidence expires, no latch)")


def test_start_tool_still_wins():
    """The historical path must be byte-identical for configs that use it."""
    out = _derive("none", {"iron_axe": 1, "dirt": 5}, "iron_axe", "dirt")
    assert out["mainhand"] == "iron_axe", out
    print("  4. start_tool still takes precedence over placement evidence")


def test_real_sensor_wins():
    """If a future MineRL populates equipped_items, it must beat both."""
    out = _derive("diamond_pickaxe", {"dirt": 5}, "iron_axe", "dirt")
    assert out["mainhand"] == "diamond_pickaxe", out
    print("  5. a live equipped_items reading overrides every fallback")


def test_wiring_pins():
    src = open(SRC).read()
    for needle, why in [
            ("self._last_placed_item: Optional[str] = None",
             "the evidence field is gone"),
            ("self._last_placed_item = str(_it)",
             "placements no longer record what left the hand"),
            ("_lp = self._last_placed_item",
             "the derivation no longer consults placement evidence")]:
        assert needle in src, f"minerl_env lost `{needle}` — {why}"
    # the writer must sit inside the `use`-gated placement branch, or it
    # would claim a hand from any inventory decrease (drops, deaths)
    w = src.index("self._last_placed_item = str(_it)")
    # PIN LOOSENED IN TEXT, NOT IN CONTRACT (2026-09-04). This used to pin
    # the exact string `if _use_act:`. The guard was later STRENGTHENED to
    #     if _use_act and not _inv_dropped:
    # after `places` was found crediting a whole inventory as placed on any
    # unreadable frame (iron_axe: 2281 for an item that cannot be placed) —
    # which also corrupted `_last_placed_item`, i.e. the very field this file
    # defends. The exact-text pin failed on a change that made the guard
    # tighter, so it is matched by prefix now; the ordering assertion below
    # is the actual contract and is unchanged.
    g = src.index("if _use_act")
    assert g < w, "placement evidence must be gated on a `use` action"
    _guard = src[g:src.index("\n", g)]
    assert "_use_act" in _guard, _guard
    assert "not _inv_dropped" in _guard, (
        f"the unreadable-frame guard is gone from `{_guard.strip()}` — an "
        f"undetected dropped observation reinstates the phantom placements "
        f"that corrupt the mainhand belief this file exists to protect")
    print(f"  6. writer gated on `{_guard.strip()}`; reader wired; "
          f"fields present")


if __name__ == "__main__":
    test_the_original_bug()
    test_evidence_names_the_hand()
    test_exhausted_stack_stops_claiming()
    test_start_tool_still_wins()
    test_real_sensor_wins()
    test_wiring_pins()
    print("[mainhand-evidence] ALL PASS")
