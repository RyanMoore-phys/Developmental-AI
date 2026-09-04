"""Inventory-derived counters: missing is not zero (Mac-ok; no MineRL).

    PYTHONPATH=. ./venv/bin/python tests/_inventory_counter_smoke.py

THE TWO LIVE LIES
-----------------
1. `places` listed **iron_axe: 2281** and **acacia_door: 3004**. An iron axe
   cannot be placed at all. CLAUDE.md §5 already names this counter as
   garbage — "don't build an argument on a counter you haven't validated" —
   because a session did exactly that.
2. `log_pickup: 1547` against **9 logs ever broken**. Both cannot be true.

ONE MECHANISM BEHIND BOTH: an absent reading was silently treated as ZERO.

  * `_inv_counts` keeps only items with count > 0 AND returns `{}` on any
    exception, so one dropped/malformed frame makes every carried item look
    like it hit zero. On a step holding `use`, the whole inventory is
    credited as placed — a 64-stack scores +64 in a single tick.
  * `_ach = _inf.get("achievements", {}) or {}` then `_ach.get("log", 0)`, so
    carrying 3 logs through a dropped frame reads 3 -> 0 -> 3 and the
    recovery counts as a PICKUP.

NEITHER IS COSMETIC. `places` sets `_last_placed_item`, which is how mainhand
is derived in this fork (no equip action; `equipped_items` is dead in MineRL
1.0) — so a phantom placement corrupts what the agent believes it is holding.
And `log_pickup` is a GROUNDED EFFECT KEY, so every phantom pickup fed a real
goal slot and a real reward event.

CONTRACTS
  1. A dropped (empty) inventory frame credits NOTHING and does not clobber
     the last known counts.
  2. One `use` places at most ONE block — a stack cannot vanish in one tick.
  3. An item vanishing entirely is confirmed on a second reading before being
     credited (placed-my-last-one is real; a one-frame flicker is not).
  4. A genuine single placement still counts, and still sets the mainhand
     belief.
  5. The `log_pickup` path treats a missing key as UNKNOWN, not as zero.
  6. Both fixes are present in BOTH stepping bodies where they are duplicated
     (§4.2).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "developmental_ai"
ENV_SRC = (ROOT / "environments" / "minerl_env.py").read_text()
LOOP_SRC = (ROOT / "core" / "developmental_loop.py").read_text()


class _Placer:
    """Replays the shipped placement-detection rules over fake readings."""

    def __init__(self, prev):
        self._inv_prev_counts = dict(prev)
        self._places_by_type = {}
        self._inv_gone_pending = {}
        self._last_placed_item = None
        self.dropped = 0

    def step(self, inv_now, use=True):
        """inv_now is None for an UNREADABLE frame, {} for genuinely empty —
        the distinction the shipped `_inv_counts` now makes."""
        placed = []
        dropped = inv_now is None
        if dropped:
            inv_now = {}
        if use and not dropped:
            for it, pc in self._inv_prev_counts.items():
                present = it in inv_now
                dd = int(pc) - int(inv_now.get(it, 0))
                if dd <= 0:
                    self._inv_gone_pending.pop(it, None)
                    continue
                if dd > 1:
                    self._inv_gone_pending.pop(it, None)
                    continue
                if not present and self._inv_gone_pending.get(it, 0) < 1:
                    self._inv_gone_pending[it] = 1
                    continue
                self._inv_gone_pending.pop(it, None)
                self._places_by_type[it] = (
                    self._places_by_type.get(it, 0) + dd)
                placed.append(it)
                self._last_placed_item = str(it)
        if not dropped:
            nxt = dict(inv_now)
            for pit in self._inv_gone_pending:
                if pit not in nxt and pit in self._inv_prev_counts:
                    nxt[pit] = self._inv_prev_counts[pit]
            self._inv_prev_counts = nxt
        else:
            self.dropped += 1
        return placed


def test_dropped_frame_credits_nothing():
    """Contract 1 — the iron_axe: 2281 mechanism, reproduced."""
    p = _Placer({"iron_axe": 1, "dirt": 64, "acacia_door": 12})
    for _ in range(200):                    # 200 flickering frames
        p.step(None, use=True)              # UNREADABLE frame
    assert p._places_by_type == {}, (
        f"a dropped frame was credited as placements: {p._places_by_type}. "
        f"This is exactly how `places` reached iron_axe: 2281.")
    assert p._last_placed_item is None, "mainhand belief must not be corrupted"
    assert p._inv_prev_counts["dirt"] == 64, (
        "a dropped frame must not clobber the last KNOWN counts")
    assert p.dropped == 200
    print(f"[inv-counter] 1. 200 dropped frames -> 0 placements "
          f"(pre-fix this scored 77 per frame = the whole inventory)")


def test_a_stack_cannot_be_placed_in_one_tick():
    """Contract 2 — one `use` press places at most one block."""
    p = _Placer({"dirt": 64})
    p.step({"dirt": 1})                     # 63 gone in a single tick
    assert p._places_by_type == {}, (
        f"credited {p._places_by_type} for a 63-block drop in one tick — a "
        f"single `use` cannot place a stack; that is a death-wipe or a bad "
        f"read")
    print("[inv-counter] 2. a 64 -> 1 jump in one tick credits nothing")


def test_vanishing_item_is_confirmed_before_crediting():
    """Contract 3 — placed-my-last-one is real; a flicker is not."""
    flick = _Placer({"torch": 1})
    flick.step({})                          # vanished (unconfirmed)
    assert flick._places_by_type == {}, "must wait for confirmation"
    # ...and it comes back: it was a flicker, never a placement
    flick.step({"torch": 1})
    assert flick._places_by_type == {}, (
        f"a one-frame flicker was credited: {flick._places_by_type}")

    real = _Placer({"torch": 1})
    real.step({})                           # vanished
    real.step({})                           # still gone -> confirmed
    assert real._places_by_type == {"torch": 1}, real._places_by_type
    print("[inv-counter] 3. flicker -> 0; genuine disappearance confirmed on "
          "the second reading -> 1")


def test_a_real_placement_still_counts():
    """Contract 4 — the fix must not silence the true signal."""
    p = _Placer({"crafting_table": 2, "dirt": 64})
    p.step({"crafting_table": 1, "dirt": 64})
    assert p._places_by_type == {"crafting_table": 1}, p._places_by_type
    assert p._last_placed_item == "crafting_table", (
        "a real placement must still set the mainhand belief — it is the "
        "only evidence of what is held in this fork")
    p.step({"crafting_table": 1, "dirt": 64}, use=False)
    assert p._places_by_type == {"crafting_table": 1}, "no use -> no place"
    print("[inv-counter] 4. a genuine 2->1 placement counts once and sets "
          "mainhand; a no-use step counts nothing")


def test_log_pickup_treats_missing_as_unknown():
    """Contract 5 — log_pickup: 1547 against 9 logs broken."""
    # Strip comments: the fix's own note quotes the OLD expression to explain
    # what it replaced, and a naive substring check trips on that sentence.
    code = "\n".join(ln for ln in LOOP_SRC.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert '_ach.get("log", 0)' not in code, (
        "a missing achievements key still reads as zero logs, so carrying "
        "logs through a dropped frame counts as a fresh pickup")
    assert 'if isinstance(_ach, dict) and "log" in _ach:' in LOOP_SRC
    assert "_lg = None" in LOOP_SRC, "absent must be UNKNOWN, not 0"
    assert "_lg is not None and _lg_prev is not None" in LOOP_SRC, (
        "the comparison must be skipped entirely on an unknown reading")
    print("[inv-counter] 5. log_pickup: a missing key is UNKNOWN — the "
          "comparison is skipped and the last known count is preserved")


def test_both_bodies_and_the_env_are_patched():
    """Contract 6 — §4.2 duplicated-body drift."""
    n = LOOP_SRC.count('if isinstance(_ach, dict) and "log" in _ach:')
    assert n == 2, (
        f"log_pickup guard present in {n}/2 stepping bodies; "
        f"_collect_segment is the one SkyBot actually runs")
    assert ENV_SRC.count("_inv_dropped = _inv_now is None") == 1
    assert "def _inv_counts(raw_obs) -> Optional[Dict[str, int]]:" in ENV_SRC, (
        "_inv_counts must distinguish unreadable (None) from empty ({}) — "
        "empty is the normal barehanded state and must stay usable")
    assert "_inv_gone_pending" in ENV_SRC
    # the old unguarded arithmetic must be gone
    assert not re.search(r"_dd = int\(_pc\) - int\(_inv_now\.get\(_it, 0\)\)"
                         r"\s*\n\s*if _dd > 0:", ENV_SRC), \
        "the original unguarded placement arithmetic is still present"
    print(f"[inv-counter] 6. log_pickup guard in {n}/2 bodies; placement "
          f"guards present in minerl_env")


if __name__ == "__main__":
    for fn in (test_dropped_frame_credits_nothing,
               test_a_stack_cannot_be_placed_in_one_tick,
               test_vanishing_item_is_confirmed_before_crediting,
               test_a_real_placement_still_counts,
               test_log_pickup_treats_missing_as_unknown,
               test_both_bodies_and_the_env_are_patched):
        fn()
    print("[inv-counter] ALL PASS")
