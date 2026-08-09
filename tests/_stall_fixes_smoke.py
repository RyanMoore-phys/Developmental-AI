"""Smoke: the five fixes from the 2026-07-27 stall assessment.

THE STALL (60 segments, 0 logs felled, 0 mints, declining mastery):
six independent blockers, two of them introduced this session. These are
the five that were fixed in code; the sixth (the tool circularity) is a
design decision left to the user.

  1. GUI PARALYSIS — the agent spent 50-60% of every episode frozen in the
     inventory screen (measured: 0.0% GUI-open across all 20 videos from
     07-24/25/26 at action_dim=10; 32-92% on every 07-27 video with the
     `inventory` button). Swings there hit nothing, yet still counted as
     attack ticks — and toggling the menu is the biggest pixel delta
     available, i.e. a CURIOSITY FARM.
  2. DIGGING PAID BEST — dirt/grass_block/gravel/sand fell through
     `_break_reward`'s 1.0 default, the same tier as any solid block, so
     the best-paid reachable behaviour was sinking a shaft. All four
     earning segments were ground breaks. The agent buried itself.
  3. COMPETENCE RATCHET (latch #6) — all 10 bound slots sat at competence
     0.0004-0.041 against a 0.25 floor; slot 25 at logit -7.78 needed ~134
     consecutive successes to be offered again. That is a latch wearing a
     probability's clothes.
  4. MINT LATCH — `_minted` was True for ALL 48 slots against a 10-skill
     bank, because the migration equated "ever unlocked" with "ever
     minted". The repeatability gate was reached and PASSED four times in
     the stalled run and blocked every time.
  5. FOSSIL MINTS — the naive fix for (4) re-opens 38 slots of which 37 are
     `discovered_N` and one is `break_air`: re-minting exactly the fossils
     a tool deliberately purged on 07-25. The guard belongs at the MINT
     SITE, not in a cleanup script that runs afterwards.

Run: PYTHONPATH=. python tests/_stall_fixes_smoke.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np

ENV = "developmental_ai/environments/minerl_env.py"
LOOP = "developmental_ai/core/developmental_loop.py"


def main() -> None:
    from developmental_ai.environments.minerl_env import MineRLEnvAdapter as E

    # `_gui_open` became an INSTANCE method on 2026-08-02 (it now consults
    # ground-truth `isGuiOpen` and logs its source once), so calling it off
    # the class raised TypeError and this whole file has been dying on its
    # first assertion ever since — taking the reward-tier, GUI-suppression and
    # RewardMixer checks below down with it, unnoticed, for the exact days the
    # run was misbehaving. Bind it to a bare instance: __new__ skips __init__,
    # which would otherwise try to launch a real Minecraft client.
    _probe = E.__new__(E)
    _probe._gui_src_logged = True          # silence the one-time source log
    _gui_open = _probe._gui_open

    # ---- 1. GUI DETECTION separates a menu from the world ---------------
    # mid-grey achromatic centre = GUI; the brightness floor is load-bearing
    # because pure black is achromatic too and a dark night must NOT read as
    # a menu (that false-positived a 07-24 video at 100%).
    gui = np.full((64, 64, 3), 160, np.uint8)          # inventory panel
    world = np.zeros((64, 64, 3), np.uint8)
    world[..., 1] = 140                                 # green grass
    world[..., 0] = 60
    night = np.zeros((64, 64, 3), np.uint8)             # pitch dark
    blue_sky = np.zeros((64, 64, 3), np.uint8)
    blue_sky[..., 2] = 200
    assert _gui_open(gui) is True, "menu not detected"
    assert _gui_open(world) is False, "grass read as a menu"
    assert _gui_open(night) is False, \
        "a dark night reads as GUI-open — the brightness floor is missing"
    assert _gui_open(blue_sky) is False, "sky read as a menu"
    assert _gui_open(None) is False and _gui_open("junk") is False
    # ground truth WINS over the pixel heuristic when the bridge supplies it
    assert _gui_open({"isGuiOpen": np.array([1]), "pov": world}) is True
    assert _gui_open({"isGuiOpen": np.array([0]), "pov": gui}) is False

    src = open(ENV).read()
    assert "_gui = self._gui_open(" in src, "GUI never evaluated in step()"
    # the GATE, not just the assignment — a mutant that neuters `if _gui:`
    # leaves the string `_atk = False` sitting there untouched.
    assert "if _gui:\n            _atk = False" in src, \
        "attack ticks are still counted while the world cannot hear them"
    assert 'info["gui_frac"]' in src, "GUI paralysis is not reported"

    # ---- 2. THE GROUND MUST NOT OUT-PAY THE GOAL ------------------------
    assert E._break_reward("oak_log") == 5.0, "log tier changed"
    for ground in ("dirt", "grass_block", "gravel", "sand", "stone"):
        assert E._break_reward(ground) == 0.15, (
            f"{ground} still pays {E._break_reward(ground)} — digging "
            f"out-pays chopping and the agent buries itself")
    assert E._break_reward("oak_leaves") == 0.3, "leaf tier changed"
    assert E._break_reward("grass") == 0.15, "trivial tier changed"
    # a real solid block the agent must SEEK still pays the middle tier
    assert E._break_reward("crafting_table") == 1.0
    # ...and the ground now sits BELOW the goal-discovery spike threshold,
    # so digging can no longer mint a goal
    assert E._break_reward("dirt") < 0.9

    # ---- 3. COMPETENCE IS RECOVERABLE, still discriminating -------------
    from developmental_ai.core.self_model import CompetencePredictor
    c = CompetencePredictor(n_tasks=4)
    for _ in range(400):
        c.update(2, 0.0)
    assert float(c.head.weight[0, 2]) >= -2.0001, "ratchet not clamped"
    assert c.predict(2) < 0.25, "clamp broke the gate's discrimination"
    n = 0
    while c.predict(2) < 0.25 and n < 500:
        c.update(2, 1.0)
        n += 1
    assert n < 60, f"recovery still takes {n} successes — effectively a latch"

    # ---- 4/5. THE LATCH REPAIR MUST NOT REBUILD THE JUNK FACTORY --------
    from developmental_ai.core.achievement_goals import is_fossil_slot_name
    for fossil in ("discovered_0", "discovered_38", "break_air", "air", ""):
        assert is_fossil_slot_name(fossil), f"{fossil!r} not seen as fossil"
    for real in ("break_oak_log", "break_grass_block", "craft_planks"):
        assert not is_fossil_slot_name(real), f"{real!r} wrongly seen as fossil"

    loop = open(LOOP).read()
    i = loop.index("_name, _desc = self._grounded_skill_name(slot, _pre)")
    guard = loop[i:i + 2000]   # the guard's rationale comment is long
    # the CONDITION, not just the import: a mutant can replace the `if` and
    # leave `from ... import is_fossil_slot_name` above it.
    assert ("if is_fossil_slot_name(slot_name) or \\\n"
            "                            is_fossil_slot_name(_name):") in guard, \
        "no fossil guard CONDITION at the mint site — re-opening the latch " \
        "would re-mint the 37 discovered_N + break_air slots the retire " \
        "tool purged on 07-25"
    assert "continue" in guard, "fossil guard does not actually skip the mint"

    # the migration derives from the BANK, and the repair is ONE-SHOT
    ag = open("developmental_ai/core/achievement_goals.py").read()
    assert "minted_probe" in ag, "migration still cannot consult the bank"
    assert "minted_reconciled_v1" in ag, \
        "no one-shot flag — an unconditional every-load rewrite of a private " \
        "latch is itself a standing override"
    assert ag.count("minted_reconciled_v1") >= 2, \
        "flag is read or written but not both (must persist)"
    # ...and the probe is keyed by INDEX PREFIX, not by drifting slot names
    assert 'pfx = f"ach_{int(idx):02d}_"' in loop, \
        "bank probe keys on slot NAME — names drift (slot 0 is now " \
        "discovered_0 while its skill is ach_00_break_birch_leaves), so a " \
        "name-keyed probe answers 'no skill' for every renamed slot"

    # ---- reward normalization exists but is OFF by default --------------
    import inspect

    from developmental_ai.policy.actor_critic import RewardMixer
    # THE DEFAULT VALUE, checked directly. Asserting mix()==0.7 is not
    # enough: with no extrinsic seen the cap is inert, so a default of 3.0
    # still returns 0.7 and the check passes while the project's "never
    # hard-fix the intrinsic/extrinsic ratio" rule is silently broken.
    _dflt = inspect.signature(RewardMixer.__init__).parameters[
        "return_ratio_cap"].default
    assert float(_dflt) == 0.0, (
        f"return_ratio_cap defaults to {_dflt} — return normalization must "
        f"be OPT-IN; it caps a ratio, which this project forbids by default")
    assert abs(RewardMixer().mix(1.0, 0.0) - 0.7) < 1e-9, \
        "default RewardMixer behaviour CHANGED"
    m = RewardMixer(intrinsic_weight=0.5, extrinsic_weight=0.5,
                    return_ratio_cap=3.0, ret_ema_alpha=1e-3,
                    min_intrinsic_scale=0.1)
    for i in range(20000):
        m.mix(0.3, 5.0 if i % 5000 == 0 else 0.0)
    out = m.mix(0.3, 0.0)
    assert 0 < out < 0.5 * 0.3, "cap did not damp a dense intrinsic stream"
    assert out >= 0.5 * 0.3 * 0.1 * 0.999, \
        "intrinsic driven below the floor — that annihilates the " \
        "exploration engine in the name of making the task visible"

    # ---- 6. OCCLUSION: no world change, no world curiosity -------------
    # MEASURED live before this fix: the menu paid +0.18 -> +0.30 intrinsic
    # per step vs +0.09 for the world (2.0x -> 3.1x, WIDENING). The agent was
    # optimal under its signal and sat in the inventory ~50% of every episode.
    # The ICM scores prediction error on the OBSERVATION; an overlay changes
    # the observation without changing the world, so paying for it is a
    # category error — the same one already fixed for the attack counter.
    assert loop.count("intrinsic[0] = intrinsic[0] * 0.0") >= 2, (
        "curiosity is still paid while the world is occluded, at one or "
        "both parallel loops — the menu remains a farm")
    _oc = loop.index("intrinsic[0] = intrinsic[0] * 0.0")
    assert 'gui_open' in loop[max(0, _oc - 400):_oc], \
        "the suppression is not conditioned on the world being occluded"
    # ...and the menu is NOT forbidden: crafting there still pays extrinsic
    assert "craft" in open(ENV).read(), "craft economy removed"

    # ---- 7. RAW SURPRISE IS EXPOSED BY EVERY CURIOSITY CLASS -----------
    # `wm_fidelity` (skill mastery) and the occlusion split both read
    # `curiosity.last_pred_error`. The BASE ICM set it — but the live class
    # is LearningProgressCuriosity, which OVERRIDES compute_intrinsic_reward
    # entirely, so the parent's assignment never ran and the field sat at its
    # 0.0 init forever. Both consumers failed SILENTLY: the split printed
    # `+0.0000 vs +0.0000 (ratio nan)` and mastery's predictability half was
    # fed a constant zero. An overridden method orphans every side effect the
    # parent performed — so assert it on EVERY class, not just the base.
    import torch as _t

    from developmental_ai.curiosity.icm import IntrinsicCuriosityModule
    from developmental_ai.curiosity.learning_progress import (
        LearningProgressCuriosity)
    for _cls in (IntrinsicCuriosityModule, LearningProgressCuriosity):
        _m = _cls(obs_dim=32, action_dim=4, feature_dim=16, hidden_dim=16)
        assert _m.last_pred_error == 0.0
        _m.compute_intrinsic_reward(
            _t.rand(4, 32),
            _t.nn.functional.one_hot(_t.randint(0, 4, (4,)), 4).float(),
            _t.rand(4, 32))
        assert _m.last_pred_error > 0.0, (
            f"{_cls.__name__} never sets last_pred_error — wm_fidelity and "
            f"the occlusion split silently read a constant 0")

    print("[stall-fixes-smoke] ALL PASS: GUI detected (menu yes, grass/night/"
          "sky no) and swings into a menu no longer count; ground pays 0.15 "
          "so digging cannot out-pay the 5.0 log; competence clamps at -2.0 "
          f"(still gated, recovers in {n}) instead of needing ~134; fossil "
          "guard sits AT the mint site so the one-shot bank-derived latch "
          "repair cannot re-mint discovered_N/break_air; return "
          "normalization present, bounded by a floor, and OFF by default")


if __name__ == "__main__":
    main()
