"""Unstuck wave smoke (2026-08-16).

THE LIVE FINDINGS THIS ENCODES (vast.ai pod, run of 2026-08-16)
    1. The agent stood at ONE coordinate for 2+ hours re-emitting help
       requests nobody read: the help channel was write-only (#51 emit,
       #46 deferred).
    2. Gaze pinned at +90; every vision predicate went constant-LOW and was
       flagged DEGENERATE despite abundant historical positives; the flag
       set is the magnet's exclusion list, so the one drive that would have
       raised the gaze was forbidden from steering (5th guard-becomes-
       latch — the min-positives gate fixed cold start, not warm
       starvation).
    3. The pitch clamp is an attractor with no exit tax: the gaze-bucket
       bonus saturates 1/sqrt(n), so nothing durable prices leaning on the
       clamp.
    4. (ops) runlogs/ollama.log: 581 MB in 5 days — capped in the launch
       scripts, not tested here.

Contracts:
    A. UnstuckAdvisor: valid advice passes; unknown remedies, hallucinated
       categories, garbage JSON and query explosions all collapse to None;
       the cooldown holds; exchanges land in help_responses.jsonl.
    B. Signal-health split: constant-LOW with proven positives is STARVED
       (still steerable, reported via actions["gaze_starved"]); constant-
       HIGH stays excluded; no evidence dict keeps the old strict law.
    C. Gaze-leveling potential: telescopes to ~0 over any closed pitch
       cycle, charges ~nothing for mining tilts, full gradient at clamps.
    D. Wiring pins (source-grepped, the _position_alias_smoke pattern):
       the loop stashes the advisor frame, consults the advisor at L3,
       applies the gaze-starved remedy, and pays gaze_level into the
       ledger; the skybot config carries the new keys.

Run: PYTHONPATH=. python tests/_unstuck_wave_smoke.py
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, ".")

from developmental_ai.infra.advisor import UnstuckAdvisor
from developmental_ai.infra.stack import InfraStack

N_OBS = 500  # > SignalMonitor's min_obs=400 judgement gate


# ---------------------------------------------------------------- A: advisor
def test_advisor():
    tmp = tempfile.mkdtemp(prefix="advisor_smoke_")
    try:
        answers = {}

        def q(prompt, images):
            q.calls += 1
            q.last_images = images
            return answers.get("raw")
        q.calls = 0

        adv = UnstuckAdvisor(q, min_gap=100, log_dir=tmp)
        cats = ["tree_visible", "water_visible"]
        sit = {"situation": "stuck seg_extrinsic=0"}

        # 1. valid prime advice passes, category validated
        answers["raw"] = json.dumps({"remedy": "prime",
                                     "category": "tree_visible",
                                     "why": "trees on the horizon"})
        a = adv.advise(sit, cats, b"png", 1000)
        assert a == {"remedy": "prime", "category": "tree_visible",
                     "why": "trees on the horizon"}, a
        assert q.last_images == [b"png"]
        print("  A1. valid prime advice accepted with its category")

        # 2. cooldown: a second ask inside min_gap is refused WITHOUT a query
        c0 = q.calls
        assert adv.advise(sit, cats, None, 1050) is None
        assert q.calls == c0, "cooldown must not spend a VLM query"
        print("  A2. cooldown holds (and costs no query)")

        # 3. hallucinated category -> None (junk-goal factory lesson)
        answers["raw"] = json.dumps({"remedy": "prime",
                                     "category": "unicorn_visible"})
        assert adv.advise(sit, cats, None, 2000) is None
        print("  A3. hallucinated category refused")

        # 4. unknown remedy, garbage JSON, empty reply, exploding query
        answers["raw"] = json.dumps({"remedy": "teleport_home"})
        assert adv.advise(sit, cats, None, 3000) is None
        answers["raw"] = "the agent should simply try harder"
        assert adv.advise(sit, cats, None, 4000) is None
        answers["raw"] = None
        assert adv.advise(sit, cats, None, 5000) is None

        def boom(prompt, images):
            raise RuntimeError("ollama fell over")
        adv2 = UnstuckAdvisor(boom, min_gap=1, log_dir=tmp)
        assert adv2.advise(sit, cats, None, 6000) is None
        print("  A4. unknown remedy / prose / empty / exploding query -> None")

        # 5. non-prime advice strips any category the model volunteered
        answers["raw"] = json.dumps({"remedy": "look_around",
                                     "category": "tree_visible",
                                     "why": "view is all dirt"})
        a = adv.advise(sit, cats, None, 7000)
        assert a is not None and a["remedy"] == "look_around"
        assert a["category"] is None
        print("  A5. category only travels with prime")

        # 6. the dialogue is on disk
        path = os.path.join(tmp, "help_responses.jsonl")
        rows = [json.loads(l) for l in open(path)]
        assert len(rows) >= 5, rows
        assert any(r["advice"] and r["advice"]["remedy"] == "prime"
                   for r in rows)
        assert any(r["advice"] is None for r in rows)
        print(f"  A6. {len(rows)} exchanges logged to help_responses.jsonl")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------- B: starved-vs-broken split
def test_signal_split():
    tmp = tempfile.mkdtemp(prefix="split_smoke_")
    try:
        stack = InfraStack({"enabled": True,
                            "degenerate_min_positives": 10,
                            "gaze_starved_frac": 0.6,
                            "stuck_floors": {}}, 12,
                           log_dir=tmp)
        assert stack.signals is not None, "signal monitor must be on"
        # constant-LOW (honest absence while staring at dirt) x2,
        # constant-HIGH (asserts presence regardless of scene) x1
        for _ in range(N_OBS):
            stack.signal_observe({"tree_visible": 0.02,
                                  "water_visible": 0.03,
                                  "stone_high": 0.97})
        ctx = {"step": 1000, "stuck_eligible": False,
               "signal_evidence": {"tree_visible": 50,
                                   "water_visible": 50,
                                   "stone_high": 50}}
        lines, actions = stack.segment(ctx)
        deg = stack.degenerate_signals()
        assert "tree_visible" not in deg and "water_visible" not in deg, (
            f"starved-low signals must stay steerable, got {deg}")
        assert "stone_high" in deg, (
            f"stuck-high must stay excluded, got {deg}")
        assert actions.get("gaze_starved") == ["tree_visible",
                                               "water_visible"], actions
        assert any("GAZE-STARVED" in l for l in lines), lines
        print("  B1. constant-low+positives steerable; constant-high "
              "excluded; gaze_starved raised at 2/3 starved")

        # no evidence dict -> the old strict law (everything flat excluded)
        stack2 = InfraStack({"enabled": True, "stuck_floors": {}}, 12,
                            log_dir=tmp)
        for _ in range(N_OBS):
            stack2.signal_observe({"tree_visible": 0.02})
        _, actions2 = stack2.segment({"step": 1000,
                                      "stuck_eligible": False})
        assert "tree_visible" in stack2.degenerate_signals()
        assert "gaze_starved" not in actions2
        print("  B2. absent evidence keeps the old strict behaviour")

        # below the mass fraction: starved individually, no gaze verdict
        stack3 = InfraStack({"enabled": True,
                             "degenerate_min_positives": 10,
                             "gaze_starved_frac": 0.6,
                             "stuck_floors": {}}, 12, log_dir=tmp)
        for i in range(N_OBS):
            stack3.signal_observe({"tree_visible": 0.02,
                                   "alive_a": 0.9 if i % 2 else 0.1,
                                   "alive_b": 0.8 if i % 3 else 0.2})
        _, actions3 = stack3.segment({"step": 1000,
                                      "stuck_eligible": False,
                                      "signal_evidence": {"tree_visible": 50,
                                                          "alive_a": 50,
                                                          "alive_b": 50}})
        assert "gaze_starved" not in actions3, actions3
        assert "tree_visible" not in stack3.degenerate_signals()
        print("  B3. one starved signal among live ones: steerable, "
              "but no mass-gaze verdict")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------ C: gaze-leveling potential
def test_pitch_potential():
    # the exact formula from developmental_loop (gaze leveling)
    def phi(pitch):
        return -((min(90.0, abs(float(pitch))) / 90.0) ** 4)

    g, w = 0.99, 0.02
    # closed cycle horizon -> clamp -> horizon telescopes to ~0
    path = [0, 30, 60, 90, 90, 90, 60, 30, 0]
    total, prev = 0.0, phi(path[0])
    for p in path[1:]:
        cur = phi(p)
        total += w * (g * cur - prev)
        prev = cur
    assert abs(total) < 0.005, f"cycle must telescope, got {total}"
    print(f"  C1. closed clamp cycle nets {total:+.5f} (~0, no farm)")

    # mining tilt is ~free, the last 30 degrees carry the gradient
    assert abs(phi(60)) < 0.2, phi(60)
    assert phi(90) == -1.0
    exit_pay = w * (g * phi(45) - phi(90))
    assert exit_pay > 0.015, f"leaving the clamp must be audible: {exit_pay}"
    print(f"  C2. phi(60deg)={phi(60):.3f} (~free), clamp exit pays "
          f"{exit_pay:+.4f}")


# ------------------------------------------- E: boring-view base discount
def test_boring_base_discount():
    # the exact multiplier from the loop's application site
    def mult(factor, w=0.85):
        return 1.0 - w * (1.0 - factor)

    # fully boring view (factor floored at 0.15 by _boring_view_factor):
    # the measured 0.0147/step cloud wage must fall below honest work
    floor_mult = mult(0.15)
    assert 0.25 < floor_mult < 0.30, floor_mult
    assert 0.0147 * floor_mult < 0.005, "sky wage must stop dominating"
    # a horizon view keeps full pay; the discount must never flip the sign
    assert mult(1.0) == 1.0
    assert all(mult(f) > 0.0 for f in (0.15, 0.5, 1.0))
    print(f"  E1. cloud wage 0.0147 -> {0.0147 * floor_mult:.4f}/step at "
          f"the clamp; horizon pay untouched")

    loop_src = open(os.path.join("developmental_ai", "core",
                                 "developmental_loop.py")).read()
    # the discount must land BEFORE the census accumulator, or the census
    # lies about the base again (the 2026-08-03 lesson, in reverse)
    apply_at = loop_src.index("_bvw * (1.0 - _bf)")
    census_at = loop_src.index('self._cen_base = getattr(self, "_cen_base"')
    assert apply_at < census_at, "discount must precede the census"
    assert "icm_boring_discount" in loop_src
    print("  E2. applied before the census accumulator; config-gated")


# --------------------------------------------------------- D: wiring pins
def test_wiring_pins():
    loop_src = open(os.path.join("developmental_ai", "core",
                                 "developmental_loop.py")).read()
    for needle, why in [
            ("self._advisor_frame = _frame",
             "the advisor never sees the current view"),
            ("self._advise_unstuck(_acts)",
             "L3 never consults the advisor"),
            ('_acts.get("gaze_starved")',
             "the gaze-starved verdict has no consumer"),
            ('("gaze_level", "_pitch_level_sum")',
             "gaze leveling invisible to the ledger/farm detector"),
            ("pitch_level_weight", "the potential is unconfigurable")]:
        assert needle in loop_src, f"loop lost `{needle}` — {why}"
    stack_src = open(os.path.join("developmental_ai", "infra",
                                  "stack.py")).read()
    assert "gaze_starved" in stack_src
    cfg = open(os.path.join("configs", "minecraft_skybot.yaml")).read()
    for key in ("pitch_level_weight", "advisor_enabled",
                "gaze_starved_frac", "icm_boring_discount"):
        assert key in cfg, f"skybot config lost `{key}`"
    print("  D1. loop/stack/config wiring pinned")


if __name__ == "__main__":
    test_advisor()
    test_signal_split()
    test_pitch_potential()
    test_boring_base_discount()
    test_wiring_pins()
    print("[unstuck-wave] ALL PASS")
