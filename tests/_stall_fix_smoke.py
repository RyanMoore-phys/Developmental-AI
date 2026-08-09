"""Smoke: the zero-reward-stall fix (2026-07-25).

Root cause proven by the diagnosis workflow: the magnet ran on the cold-start
FLOOR branch until `_cold_spent` hit `cold_start_budget` (4000), then latched
to w=0.0 permanently — and every shaping term is a product with w, so all
vision shaping died while the agent stood in front of birch trunks.

The fix unlatches that budget and RAISES the floor weight, which would be
dangerous on its own: the four raw per-step income terms (instinct/approach/
align/aim) would become farmable (park at a trunk, hold attack, collect).
They are therefore zeroed in config, leaving ONLY telescoping potentials.

Contracts proven here (all with the REAL VisionScaffold + the live config):
  1. NO-LATCH: the cold-start floor survives far past the old 4000-step
     budget (the exact failure that stalled the 19h run).
  2. NO-PARK-INCOME: a stationary agent holding attack at a trunk earns
     EXACTLY 0 over a long stretch (this is what the zeroed raw terms buy).
  3. NO-RATCHET: approach -> lose sight -> re-approach nets ~0 (the
     _phi_prev / _seek_prob_prev carry fix). Pre-fix this paid twice.
  4. GRADIENT ALIVE: genuine approach (rising phi) still pays > 0, so there
     IS a learning signal to follow to the trunk.

Run: PYTHONPATH=. python tests/_stall_fix_smoke.py
"""
import sys

import yaml

sys.path.insert(0, ".")

from developmental_ai.llm.vision_scaffold import VisionScaffold

CFG = yaml.safe_load(open("configs/minecraft_skybot.yaml"))["llm"]["vision"]
CATS = ["tree_visible", "stone_visible", "water_visible", "animal_visible"]
REL = {c: 1.0 for c in CATS}
LBL = {c: 999 for c in CATS}
ATTACK = 5          # chop action
FWD = 1             # pure forward


def scaffold():
    """The REAL scaffold, built from the LIVE config (no stubs)."""
    s = VisionScaffold(
        enabled=True,
        weight=CFG["weight"], min_weight=CFG["min_weight"],
        target_categories=list(CFG.get("target_categories") or CATS),
        cold_start_weight=CFG["cold_start_weight"],
        cold_start_budget=CFG["cold_start_budget"],
        ema_leak=CFG["ema_leak"],
        present_threshold=CFG["present_threshold"],
        eps_abs=CFG["eps_abs"],
        instinct_bonus=CFG["instinct_bonus"],
        approach_pull=CFG["approach_pull"],
        align_bonus=CFG["align_bonus"],
        aim_bonus=CFG["aim_bonus"],
        seek_weight=CFG["seek_weight"],
        seek_forward_nudge=CFG["seek_forward_nudge"],
        seek_nudge_budget=CFG["seek_nudge_budget"],
    )
    s.vlm = None            # never call the VLM in a smoke
    return s


def frame(tree=0.9, centered=0.9, adjacent=0.9):
    """A trunk filling the view, dead ahead, in reach."""
    return {"tree_visible": tree, "stone_visible": 0.05,
            "water_visible": 0.0, "animal_visible": 0.0,
            "object_centered": centered, "object_adjacent": adjacent,
            "object_left": 0.0, "object_right": 0.0}


def run(s, n, action, probs, t0=0, lp=0.005):
    return sum(s.step_shaping(action, t0 + i, lp, probs, REL, LBL)
               for i in range(n))


def main() -> None:
    # ---- config sanity: the fix is actually in the file being shipped ----
    assert CFG["instinct_bonus"] == 0.0 and CFG["approach_pull"] == 0.0, CFG
    assert CFG["align_bonus"] == 0.0 and CFG["aim_bonus"] == 0.0, CFG
    assert CFG["cold_start_budget"] > 1_000_000, CFG["cold_start_budget"]

    # ---- 1. NO-LATCH: floor survives past the old 4000 budget ----
    s = scaffold()
    run(s, 50, FWD, frame())
    w_early = s.stats["weight"]
    run(s, 6000, FWD, frame(), t0=50)          # 50% past the OLD budget
    w_late = s.stats["weight"]
    assert w_early > 0.0, f"floor never armed: {w_early}"
    assert w_late > 0.0, (
        f"COLD-START RE-LATCHED at {s.stats['cold_spent']} — the exact 19h "
        f"stall bug is still present (w={w_late})")

    # ---- 2. NO-PARK-INCOME: stationary attack at a trunk earns ~0 ----
    s = scaffold()
    run(s, 100, ATTACK, frame())               # settle the potential baseline
    parked = run(s, 3000, ATTACK, frame(), t0=100)
    assert abs(parked) < 1e-6, (
        f"PARK FARM: standing at a trunk holding attack paid {parked:+.4f} "
        f"over 3000 steps (must be 0 — only potentials may pay)")

    # ---- 3. NO-RATCHET: approach / lose sight / re-approach nets ~0 ----
    far = frame(tree=0.05, centered=0.05, adjacent=0.0)   # not trusted-present
    near = frame()
    s = scaffold()
    run(s, 20, FWD, near)                      # arrive, settle baseline
    cycles = 0.0
    for _ in range(5):                         # look away, come back, x5
        cycles += run(s, 30, FWD, far)
        cycles += run(s, 30, FWD, near)
    assert cycles <= 0.05, (
        f"RATCHET FARM: 5 lose-sight/re-approach cycles paid {cycles:+.4f} "
        f"(must be ~0 — the potential has to charge the descent)")

    # ---- 3b. NO SECOND LATCH: a category that was ONCE contrastively
    #      curious must still get the floor when its score fades. Branch 1
    #      sets _ever_curious[c]=True, and the floor used to require
    #      `not _ever_curious` -> w=0 forever (the 19h stall's second door).
    s = scaffold()
    run(s, 30, FWD, frame())
    s._ever_curious["tree_visible"] = True      # simulate branch 1 having run
    s._cat_lp["tree_visible"] = s._global_lp    # contrastive score -> 0
    w_after = None
    for i in range(200):
        s.step_shaping(FWD, 10_000 + i, 0.005, frame(), REL, LBL)
        w_after = s.stats["weight"]
    assert w_after > 0.0, (
        f"SECOND LATCH: an ever-curious category with a faded score fell to "
        f"w={w_after} — the stall returns through the _ever_curious door")

    # ---- 3c. ALL LATCH DOORS: mark EVERY category ever-curious (the real
    #      steady state — the agent eventually engages everything) and drive
    #      the contrastive score to 0. The magnet must STILL arm. This covers
    #      all three `_ever_curious` gates at once: the weight branch, target
    #      RETENTION, and target SELECTION (_cold_start_candidate). Missing
    #      any one of them reproduces the 19h zero-reward stall (observed
    #      live: w=0.0000 target=None while cold_spent was only 1498/1e8).
    s = scaffold()
    run(s, 30, FWD, frame())
    for c in list(s.target_categories):
        s._ever_curious[c] = True               # everything engaged
        s._cat_lp[c] = s._global_lp             # contrastive score -> 0
    s._target = None                            # and the target was released
    w_doors = None
    for i in range(300):
        s.step_shaping(FWD, 20_000 + i, 0.005, frame(), REL, LBL)
        w_doors = s.stats["weight"]
    assert w_doors > 0.0, (
        f"LATCH via target selection/retention: every category ever-curious "
        f"+ flat contrastive -> w={w_doors}, target={s.stats['target']}. The "
        f"_pure_potential bypass must apply to ALL THREE _ever_curious gates")
    assert s.stats["target"] is not None, "no target selected — floor is dead"

    # ---- 4. GRADIENT ALIVE: real approach still pays ----
    s = scaffold()
    s.step_shaping(FWD, 0, 0.005, frame(tree=0.65, centered=0.1,
                                        adjacent=0.0), REL, LBL)
    gain = 0.0
    for i, (c, a) in enumerate([(0.3, 0.1), (0.5, 0.3), (0.7, 0.6),
                                (0.9, 0.9)]):
        gain += s.step_shaping(FWD, i + 1, 0.005,
                               frame(tree=0.9, centered=c, adjacent=a),
                               REL, LBL)
    assert gain > 0.0, (
        f"NO GRADIENT: closing on a trunk paid {gain:+.4f} — the agent has "
        f"nothing to follow toward the tree")

    # ---- 5. TRUNK FOCUS lives in the POTENTIAL (2026-07-25) ----
    # Live failure: with instinct_bonus zeroed (it was farmable 16:1), NOTHING
    # discriminated trunk from leaf, so the agent walked to a tree and chopped
    # the nearest foliage (dirt/fern/grass/birch_leaves broken, ~no logs).
    # Folding focus into phi restores the discrimination WITHOUT income.
    s = scaffold()
    s._target = "tree_visible"
    on_trunk = s._phi_head({"object_centered": 0.9, "object_adjacent": 0.9,
                            "tree_visible": 0.9, "leaves_visible": 0.1})
    on_leaf = s._phi_head({"object_centered": 0.9, "object_adjacent": 0.9,
                           "tree_visible": 0.1, "leaves_visible": 0.9})
    assert on_trunk > on_leaf, (
        f"phi does not distinguish trunk ({on_trunk}) from leaf ({on_leaf}) — "
        f"the agent has no gradient toward the LOG")
    assert on_leaf > 0.0, "distractor-adjacent must still pay > 0 (approach)"

    # ...and it is STILL unfarmable: parking at a trunk earns exactly 0
    s = scaffold()
    trunk = frame()
    trunk.update({"leaves_visible": 0.1})
    run(s, 100, ATTACK, trunk)
    parked_trunk = run(s, 2000, ATTACK, trunk, t0=100)
    assert abs(parked_trunk) < 1e-6, (
        f"trunk-focus reintroduced income: parking paid {parked_trunk:+.4f} "
        f"(the 16:1 farm this whole design avoids)")

    print(f"[stall-fix-smoke] ALL PASS: no re-latch (w={w_late:.3f} after "
          f"6k steps), park income={parked:+.6f}, ratchet={cycles:+.6f}, "
          f"approach gradient={gain:+.4f}")


if __name__ == "__main__":
    main()
