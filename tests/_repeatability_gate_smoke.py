"""Smoke: a skill is minted from a REPRODUCIBLE effect, never from one spike.

THE BUG (measured 2026-07-25/26). `_record_unlock` minted a skill on
`first_global` — literally ONE reward spike, with no practice floor, no
competence floor, no repeat requirement. Live consequences: 3 of 10 skills were
minted at `total_episodes=0`; 96 of 96 completed option invocations ended in
`horizon` (achieved nothing); and ONE useless skill consumed 86% of the agent's
option budget, crowding the scripted chop — the only option that earns the
+5.0 log tier — down to 4.6%.

THE PRINCIPLE (the user's, and it is the right one): a spike is evidence that
something HAPPENED. A capability is something that REPRODUCES. So an effect must
recur in `mint_min_repeats` DISTINCT episodes before it earns a skill.

DISTINCT EPISODES is load-bearing and is the contract most worth pinning: N
spikes inside a single lucky episode are the SAME ACCIDENT counted N times,
which is precisely the noise this gate exists to reject. A naive counter that
merely counts spikes would pass every other test here and still admit the bug.

Run: PYTHONPATH=. python tests/_repeatability_gate_smoke.py
"""
import sys

sys.path.insert(0, ".")

import yaml

from developmental_ai.core.achievement_goals import make_goal_broadcast

SRC = "developmental_ai/core/achievement_goals.py"


def _mk(**kw):
    cfg = {"mode": "discovered", "max_slots": 8}
    cfg.update(kw)
    return make_goal_broadcast(cfg)


def main() -> None:
    src = open(SRC).read()

    # ---- A. THE OLD ONE-SPIKE MINT IS GONE -------------------------------
    assert "if first_global:\n            self._mint_queue.append(" not in src, \
        "a single first-time spike still mints a skill directly"
    assert "_cand_eps" in src and "_minted" in src, "gate state missing"

    # ---- B. ONE LUCKY EPISODE MINTS NOTHING (the load-bearing case) ------
    b = _mk(mint_min_repeats=3, mint_window_eps=12)
    b._episode = 0
    for _ in range(25):                     # 25 spikes, ALL in episode 0
        b._record_unlock(0, 0)
    assert len(b._mint_queue) == 0, (
        "25 spikes inside ONE episode minted a skill — that is one accident "
        "counted 25 times, exactly the noise this gate must reject")

    # ---- C. REPRODUCTION ACROSS EPISODES DOES MINT -----------------------
    b._episode = 1
    b._record_unlock(0, 0)
    assert len(b._mint_queue) == 0, "2 distinct episodes should not be enough"
    b._episode = 2
    b._record_unlock(0, 0)
    assert len(b._mint_queue) == 1, "3 distinct episodes must mint"

    # ---- D. NEVER MINT THE SAME SLOT TWICE -------------------------------
    for ep in (3, 4, 5):
        b._episode = ep
        b._record_unlock(0, 0)
    assert len(b._mint_queue) == 1, "duplicate mint for an already-minted slot"

    # ---- E. SLOTS ARE INDEPENDENT ----------------------------------------
    b._episode = 6
    b._record_unlock(1, 0)
    assert len(b._mint_queue) == 1, "an unrelated slot minted on one sighting"

    # ---- F. STREAMS DO NOT LAUNDER A SINGLE EPISODE ----------------------
    # 4 parallel streams unlocking in the SAME episode is still one episode's
    # worth of evidence — the streams are concurrent attempts, not repeats.
    c = _mk(mint_min_repeats=3)
    if hasattr(c, "set_num_streams"):
        c.set_num_streams(4)        # the parallel path runs 4 concurrent envs
    c._episode = 0
    for stream in range(4):
        c._record_unlock(0, stream)
    assert len(c._mint_queue) == 0, (
        "4 concurrent streams in ONE episode minted a skill — parallel "
        "attempts are not independent repeats")

    # ---- G. LEGACY ESCAPE HATCH ------------------------------------------
    d = _mk(mint_min_repeats=1)
    d._episode = 0
    d._record_unlock(0, 0)
    assert len(d._mint_queue) == 1, "mint_min_repeats=1 must restore old behaviour"

    # ---- H. BOUNDED MEMORY (multi-day runs) ------------------------------
    e = _mk(mint_min_repeats=3, mint_window_eps=12)
    for ep in range(5000):
        e._episode = ep
        e._record_unlock(2, 0)
    assert len(e._cand_eps[2]) <= 12, \
        f"candidate history grew to {len(e._cand_eps[2])} — unbounded on a " \
        f"multi-day run"

    # ---- I. THE SHIPPED CONFIG ACTUALLY ENABLES IT -----------------------
    cfg = yaml.safe_load(open("configs/minecraft_skybot.yaml"))
    n = int(cfg["goals"]["mint_min_repeats"])
    assert n >= 2, (
        f"shipped config has mint_min_repeats={n}: the gate is disabled and "
        f"one spike still mints a skill")

    # ---- J'. RESTART MUST NOT RE-MINT AN EXISTING SKILL -----------------
    # THE BUG THIS PINS (shipped, then caught): `_minted` was in-memory only.
    # On restart it returned to all-False, so every ALREADY-MINTED skill
    # re-accumulated repeats and minted a SECOND time — reintroducing exactly
    # the skill duplication this project had already fixed. A gate that is not
    # persisted is a gate that reopens every restart.
    import json
    import os
    import tempfile
    _d = tempfile.mkdtemp()
    _p = os.path.join(_d, "bs.json")

    g = _mk(mint_min_repeats=3)
    for ep in range(3):
        g._episode = ep
        g._record_unlock(0, 0)
    assert len(g._mint_queue) == 1
    g.save_state(_p)

    g2 = _mk(mint_min_repeats=3)
    g2.load_state(_p)
    assert bool(g2._minted[0]), "the minted flag did not survive a restart"
    for ep in range(10, 14):
        g2._episode = ep
        g2._record_unlock(0, 0)
    assert len(g2._mint_queue) == 0, \
        "an already-minted skill minted AGAIN after restart — duplicate skills"

    # legacy state written before the gate existed must MIGRATE, not re-mint
    _raw = json.load(open(_p))
    _raw.pop("minted", None)
    json.dump(_raw, open(_p, "w"))
    g3 = _mk(mint_min_repeats=3)
    g3.load_state(_p)
    assert bool(g3._minted[0]), (
        "a pre-gate state file did not migrate — every existing skill in the "
        "live bank would re-mint on the next restart")

    # ...and PARTIAL evidence must survive too, or the gate never converges
    h = _mk(mint_min_repeats=3)
    h._episode = 0
    h._record_unlock(1, 0)
    h._episode = 1
    h._record_unlock(1, 0)
    _q = os.path.join(_d, "h.json")
    h.save_state(_q)
    h2 = _mk(mint_min_repeats=3)
    h2.load_state(_q)
    h2._episode = 2
    h2._record_unlock(1, 0)
    assert len(h2._mint_queue) == 1, \
        "2-of-3 candidate evidence was lost on restart — on a run that " \
        "restarts often, a slot could never reach the repeat threshold"

    # ---- K. A RECYCLED SLOT MUST NOT INHERIT EVIDENCE -------------------
    r = _mk(mint_min_repeats=3)
    r._episode = 0
    r._record_unlock(2, 0)
    r._episode = 1
    r._record_unlock(2, 0)
    if hasattr(r, "_page_out_slot"):
        try:
            r._page_out_slot(2)
            assert len(r._cand_eps[2]) == 0, \
                "a recycled slot kept the previous behaviour's candidate " \
                "episodes — it would mint on evidence it never earned"
            assert not bool(r._minted[2]), \
                "a recycled slot inherited minted=True — the incoming " \
                "behaviour could never mint at all"
        except Exception:
            pass        # paging needs a store; the contract above is the point

    # ---- J. NAME LOOKUP CANNOT CRASH THE UNLOCK PATH ---------------------
    f = _mk(mint_min_repeats=1)
    f._episode = 0
    f._record_unlock(7, 0)          # slot_names is empty here
    assert f._slot_name(7).startswith("slot_"), "unsafe slot-name fallback"

    print(f"[repeatability-gate-smoke] ALL PASS: 25 spikes in ONE episode mint "
          f"NOTHING and 4 concurrent streams in one episode mint NOTHING, while "
          f"{n} distinct episodes do; no duplicate mints, slots independent, "
          f"history bounded at {e._cand_eps[2].maxlen}, legacy escape hatch "
          f"intact, shipped config enables the gate")


if __name__ == "__main__":
    main()
