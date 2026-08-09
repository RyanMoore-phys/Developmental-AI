"""Smoke: the two DEAD SENSORS that blocked tool- and creature-meaning.

Both subsystems were built, tested, live — and had never produced a single
fact. Neither was a logic bug. Both were sensors that silently reported a
default forever, which is indistinguishable from "the thing never happened".

  1. `equipped_items.mainhand` is NOT populated by MineRL 1.0's MCP-Reborn
     bridge: it reads "none" no matter what is held. Read as ground truth it
     said "the agent is barehanded", which produced a ~60-tick break estimate,
     an option-budget change made for the wrong reason, and a confident wrong
     conclusion that the axe was unreachable. The agent has in fact been
     HOLDING the iron axe since spawn (SimpleInventoryAgentStart writes it to
     inventory index 0, index 0 IS hotbar slot 0, `currentItem` starts at 0,
     and no macro in TREECHOP_MACROS can change the selected slot).

  2. `life` is pinned at MAX_LIFE by a MineRL key-path bug, so the life-derived
     `damage`/`died` events could never fire — even though a skeleton has
     demonstrably KILLED this agent (difficulty is hard-coded HARD). With no
     felt consequence there is nothing for "creature" to MEAN.

Run: PYTHONPATH=. python tests/_perception_unblock_smoke.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np

from developmental_ai.environments.minerl_env import (
    START_TOOL, MineRLEnvAdapter)

ENV = "developmental_ai/environments/minerl_env.py"


def _adapter():
    """Bare adapter with only the state _world_events touches."""
    e = MineRLEnvAdapter.__new__(MineRLEnvAdapter)
    e._prev_tool_damage = {}
    e._prev_mainhand = None
    # SEPARATE history for the derived hand — sharing _prev_mainhand with the
    # dead equipped_items path caused the per-step false-loss loop (#58).
    # NOTE: omitting this made the derivation raise AttributeError, which the
    # broad try/except in _world_events SWALLOWS — so the symptom was "the
    # sensor silently does nothing", identical to the class of bug this file
    # exists to catch. The real __init__ sets it; only this stub can forget.
    e._prev_derived_hand = None
    e._prev_life = None
    e._prev_damage_taken = None
    e._prev_deaths = None
    return e


def main() -> None:
    src = open(ENV).read()

    # ---- A. ONE SOURCE OF TRUTH FOR THE HELD TOOL -----------------------
    # The agent-start spec and the sensor fallback must never disagree about
    # what the agent holds — that disagreement IS the original confusion.
    assert 'START_TOOL: str = "iron_axe"' in src
    assert "dict(type=START_TOOL" in src, "agent start does not use START_TOOL"
    assert "tool = START_TOOL" in src, "derivation does not use START_TOOL"
    # Count only in CODE. Comments legitimately quote the literal when
    # explaining a bug (e.g. the per-step false-loss loop of #58), and a
    # whole-file count fails on prose — which is failing for the wrong reason,
    # the exact trap that a fixed-window assertion hit earlier tonight.
    _code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    _n = sum(ln.count('"iron_axe"') for ln in _code)
    assert _n == 1, (
        f"iron_axe appears in CODE {_n} times, expected only the START_TOOL "
        f"constant — the agent-start spec and the sensor can drift apart again")

    # ---- B. NESTED-GROUP STAT READER ------------------------------------
    # ObserveFromFullStats nests as raw_obs["deaths"]["deaths"]; _read_scalar
    # would miss it and return None, which reads as "never happened".
    R = MineRLEnvAdapter._read_stat
    assert R({"damage_taken": {"damage_taken": np.array([7.0])}},
             "damage_taken") == 7.0, "nested stat group not read"
    assert R({"deaths": np.array([2.0])}, "deaths") == 2.0, "flat form not read"
    assert R({"other": 1}, "deaths") is None
    assert R("not a dict", "deaths") is None      # must not raise

    # ---- C. MAINHAND IS DERIVED WHEN THE SENSOR IS DEAD -----------------
    e = _adapter()
    raw = {"equipped_items": {"mainhand": {"type": "none",
                                           "damage": np.array([0])}},
           "inventory": {START_TOOL: np.array([1])}}
    out = e._world_events(raw)
    assert out.get("mainhand") == START_TOOL, (
        f"mainhand={out.get('mainhand')!r} — the dead sensor still wins, so "
        f"the tool-meaning subsystem stays blocked")
    assert out.get("mainhand_derived") is True, \
        "a derived reading must be labelled as derived, not passed off as measured"

    # ---- D. A REAL SENSOR READING MUST STILL WIN ------------------------
    # If a future MineRL populates equipped_items, measurement beats inference.
    e2 = _adapter()
    out2 = e2._world_events({
        "equipped_items": {"mainhand": {"type": "stone_axe",
                                        "damage": np.array([3])}},
        "inventory": {START_TOOL: np.array([1])}})
    assert out2.get("mainhand") == "stone_axe", \
        "derivation overrode a real reading"
    assert not out2.get("mainhand_derived"), "real reading mislabelled as derived"

    # ---- E. LOSING THE TOOL IS AN EVENT ---------------------------------
    e3 = _adapter()
    e3._world_events({"inventory": {START_TOOL: np.array([1])}})
    gone = e3._world_events({"inventory": {START_TOOL: np.array([0])}})
    assert gone.get("tool_lost") == START_TOOL, \
        "the tool leaving the inventory mints no event — 'it broke' is unlearnable"
    assert gone.get("mainhand") == "none"

    # ---- F. HURT AND DEATH ARE GROUND TRUTH, NOT LIFE-DERIVED -----------
    assert 'ObserveFromFullStats("damage_taken")' in src
    assert 'ObserveFromFullStats("deaths")' in src
    e4 = _adapter()
    base = {"damage_taken": {"damage_taken": np.array([0.0])},
            "deaths": {"deaths": np.array([0.0])}}
    first = e4._world_events(base)
    assert "damage" not in first, "first observation must not invent an event"
    hurt = e4._world_events({"damage_taken": {"damage_taken": np.array([4.0])},
                             "deaths": {"deaths": np.array([0.0])}})
    assert hurt.get("damage") == 4.0, "a damage delta produced no event"
    assert hurt.get("damage_source") == "stat", "event not attributed to the counter"
    assert "died" not in hurt, "damage alone must not read as death"
    dead = e4._world_events({"damage_taken": {"damage_taken": np.array([4.0])},
                             "deaths": {"deaths": np.array([1.0])}})
    assert dead.get("died") is True, "a deaths delta produced no death event"
    # counters are monotonic: no delta => no event (never a phantom)
    same = e4._world_events({"damage_taken": {"damage_taken": np.array([4.0])},
                             "deaths": {"deaths": np.array([1.0])}})
    assert "damage" not in same and "died" not in same, \
        "an unchanged counter manufactured an event"

    # ---- G. THE LIFE PATH MUST NOT BE TRUSTED SILENTLY ------------------
    assert "pinned at MAX_LIFE" in src, \
        "nothing warns a future reader that life-derived damage is dead"

    # ---- H. NO PER-STEP FALSE TOOL LOSS (bug #58) -----------------------
    # The derived-hand branch once wrote the DEAD sensor's `_prev_mainhand`,
    # so the dead path saw "had an axe, now none" EVERY STEP and emitted
    # tool_lost ~7000 times, flooding the KG with a fact that never happened.
    # The two paths must keep separate history.
    e5 = _adapter()
    raw5 = {"equipped_items": {"mainhand": {"type": "none",
                                            "damage": np.array([0])}},
            "inventory": {START_TOOL: np.array([1])}}
    spurious = sum(1 for _ in range(200)
                   if e5._world_events(raw5).get("tool_lost"))
    assert spurious == 0, (
        f"{spurious}/200 steps emitted a FALSE tool_lost while the tool was "
        f"present — the per-step false-loss loop is back and will pollute the "
        f"knowledge graph at ~1 bogus fact per step")
    # ...and a REAL loss must still fire exactly once, not zero, not forever
    gone5 = {"equipped_items": {"mainhand": {"type": "none",
                                             "damage": np.array([0])}},
             "inventory": {START_TOOL: np.array([0])}}
    real = sum(1 for _ in range(50)
               if e5._world_events(gone5).get("tool_lost"))
    assert real == 1, f"a genuine tool loss fired {real}x, expected exactly 1"
    assert "_prev_derived_hand" in src, \
        "derived hand shares state with the dead sensor again"

    print(f"[perception-unblock-smoke] ALL PASS: {START_TOOL} is a single "
          f"constant shared by spec and sensor; mainhand DERIVES when "
          f"equipped_items is dead (and is labelled derived) while a real "
          f"reading still wins; tool loss mints an event; hurt/death come from "
          f"monotonic ground-truth counters with no phantom events")


if __name__ == "__main__":
    main()
