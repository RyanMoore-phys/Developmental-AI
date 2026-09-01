"""GUI-occlusion farm smoke (2026-08-17).

THE LIVE FAILURE THIS ENCODES
    SkyBot right-clicked a WANDERING VILLAGER, opened its trade screen, and
    sat inside it for 10,149+ CONSECUTIVE steps: position frozen, gui 66% of
    the segment, `magnet_seek` paying 77% of ALL income.

    Two independent faults, both of which had to be fixed:

    1. THE LEAK. The gui_open guard zeroes `intrinsic` — but the magnet's
       shaping is added to prim_EXTRINSIC (`prim_extrinsic = rewards[0] +
       _sr`), which that guard never touched. So the single largest income
       source kept paying through an occluded camera. This is the
       2026-08-02 occlusion farm repeating one channel over.

    2. NO WAY OUT, AND THE EXIT TRAINED AWAY. Zeroing income gives no
       gradient toward the door. Worse: pressing `inventory` OPENS a screen
       that pays 0, so the policy drove that action to zero probability —
       measured 0 presses across an entire run at 92%-of-max entropy. When
       a villager opened a GUI via `use`, the agent had already unlearned
       the only button that closes one. Guard-becomes-latch.

    3. AND THE FIX ITSELF WAS WRONG FIRST TIME. A dwell COST written as the
       usual `gamma*Phi' - Phi` pays `w*(1-gamma)` on every step Phi is
       PINNED — turning "sitting here costs" into "sitting here pays". The
       shipped gaze-level term had the same flaw and the log showed it:
       "Gaze level: +0.00014/step", a wage for staying at the pitch clamp.

Contracts:
    A. A state-cost potential must net ~0 over enter/dwell/exit AND pay
       exactly 0 while pinned. (gamma<1 fails this; plain difference passes.)
    B. Magnet shaping is not added to prim_extrinsic while gui_open, and the
       seek/centring potentials re-adopt so closing pays no windfall.
    C. The dwell cost lands in the EXTRINSIC channel — the intrinsic one is
       zeroed inside a GUI, which would erase the cost entirely.

Run: PYTHONPATH=. python tests/_gui_farm_smoke.py
"""
import os
import sys

sys.path.insert(0, ".")

SRC = os.path.join("developmental_ai", "core", "developmental_loop.py")


def _cycle(weight, gamma, steps, floor_n=200.0):
    """enter -> dwell `steps` -> exit, and the per-step pay while pinned."""
    def phi(run):
        return -min(1.0, run / floor_n)
    total, prev = 0.0, phi(0)
    for run in list(range(1, steps + 1)) + [0]:
        cur = phi(run)
        total += weight * (gamma * cur - prev)
        prev = cur
    pinned = weight * (gamma * -1.0 - -1.0)
    return total, pinned


def test_state_cost_must_not_pay_while_pinned():
    # the WRONG form, kept as a regression witness
    tot_bad, pinned_bad = _cycle(0.05, 0.99, 600)
    assert pinned_bad > 0.0, "the flawed form should pay while pinned"
    assert tot_bad > 0.2, tot_bad
    print(f"  A1. gamma=0.99 form: cycle {tot_bad:+.4f}, pinned "
          f"{pinned_bad:+.6f}/step -> a WAGE for dwelling (rejected)")

    tot_ok, pinned_ok = _cycle(0.05, 1.0, 600)
    assert abs(tot_ok) < 1e-9, tot_ok
    assert abs(pinned_ok) < 1e-12, pinned_ok
    print(f"  A2. plain difference: cycle {tot_ok:+.6f}, pinned "
          f"{pinned_ok:+.6f}/step -> costs on entry, refunds on exit, "
          f"flat while stuck")

    # entering genuinely costs, leaving genuinely refunds
    def phi(r):
        return -min(1.0, r / 200.0)
    enter = 0.05 * (phi(200) - phi(0))
    leave = 0.05 * (phi(0) - phi(200))
    assert enter < 0 < leave and abs(enter + leave) < 1e-12
    print(f"  A3. entry {enter:+.4f} / exit {leave:+.4f} — equal and "
          f"opposite, so open/close cannot be farmed either way")


def test_source_contracts():
    src = open(SRC).read()
    # the leak: magnet shaping must be gui-gated
    assert 'if bool((step_infos[0] or {}).get("gui_open")):' in src
    assert "self.vision_scaffold._seek_prob_prev = None" in src, (
        "seek potential must re-adopt in a GUI or closing pays a windfall")
    # both duplicated bodies must carry it (this codebase's recurring bug)
    assert src.count("_gui_now2 = bool(") == 2, (
        "the gui gate must exist in BOTH stepping bodies")
    assert src.count("self._gui_dwell_phi = _gphi") == 2
    print("  B1. magnet shaping gui-gated in BOTH bodies; seek re-adopts")

    # the dwell cost must use the plain difference, in BOTH bodies
    assert "_gdw * (\n                            _gphi - float(_gprev))" in src \
        or "_gphi - float(_gprev)" in src
    assert "_gg * _gphi" not in src, (
        "gui dwell reverted to the gamma form — it would pay to dwell")
    assert "_gp * _php" not in src, (
        "gaze level reverted to the gamma form — it paid +0.00014/step to "
        "sit at the clamp")
    print("  B2. both state-cost potentials use the plain difference")

    # the cost must go to the EXTRINSIC channel
    i_cost = src.index("prim_extrinsic = prim_extrinsic + _gdw")
    i_zero = src.index("intrinsic[0] = intrinsic[0] * 0.0")
    assert i_cost < i_zero, "cost must not be placed after/into the zeroing"
    assert "intrinsic[0] = intrinsic[0] + _gdw" not in src, (
        "a dwell cost in the intrinsic channel is erased by the gui guard")
    print("  B3. dwell cost paid into prim_extrinsic (intrinsic is zeroed "
          "inside a GUI, which would erase it)")

    cfg = open(os.path.join("configs", "minecraft_skybot.yaml")).read()
    for k in ("gui_dwell_weight", "gui_dwell_steps"):
        assert k in cfg, f"skybot config lost `{k}`"
    print("  B4. config keys present")


def test_no_one_way_doors():
    """An action that OPENS a screen may not be offered without the one
    that CLOSES it.

    This is the trap that produced the villager incident: `disable_macros:
    [10]` removed `inventory` — the only close — while `use` (11), which
    opens a villager/chest screen on right-click, stayed available. The
    agent then had no exit in its action space at all and sat in the trade
    menu for 10,149+ consecutive steps. Disabling 10 was defensible when
    the menu PAID (a reward leak, since fixed); it was never defensible
    while an entrance remained open.
    """
    import yaml
    cfg = yaml.safe_load(open(os.path.join("configs",
                                           "minecraft_skybot.yaml")))
    dis = set(int(x) for x in
              (cfg.get("environment", {}).get("disable_macros") or []))
    OPENS, CLOSES = 11, 10          # use (right-click) / inventory (toggle)
    if OPENS not in dis:
        assert CLOSES not in dis, (
            f"macro {OPENS} (`use`, opens villager/chest screens) is offered "
            f"while macro {CLOSES} (`inventory`, the ONLY close) is disabled "
            f"— that is a one-way door; disable both or neither")
    print(f"  C1. no one-way door: disable_macros={sorted(dis) or '[]'} — "
          f"an entrance is never offered without its exit")


if __name__ == "__main__":
    test_state_cost_must_not_pay_while_pinned()
    test_source_contracts()
    test_no_one_way_doors()
    print("[gui-farm] ALL PASS")
