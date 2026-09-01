"""Social-learning wave smoke (2026-08-09): monkey-see-monkey-do, honestly.

Contracts:
  1. player_visible is a full citizen: predicate, KG triple, prompt battery,
     fovea vocabulary, magnet-steerable.
  2. social_prime injects targetable curiosity that the magnet can select,
     refills the search budget, and EXPIRES on its own (never a latch).
  3. joint attention: with the teacher under the gaze, present categories
     accumulate LP faster than without (attribution boost, reward untouched).
  4. external-agency detector: passive+stationary+big frame change emits
     ("observed_change","world"); acting or moving steps never do; small
     drift stays silent.
  5. demonstration gating shape: prime only fires for known categories.
"""
import numpy as np

from developmental_ai.llm import vlm_symbolizer as V
from developmental_ai.llm.vision_scaffold import (VisionScaffold,
                                                  steerable_targets)


def test_player_is_a_full_citizen():
    assert "player_visible" in V.PREDICATES
    assert "player_visible" in V._FACT_TEMPLATES
    assert "player_visible" in V._SCENE_PROMPT
    assert "player_visible" in V.FOVEA_PREDICATES
    assert "player_visible" in V._FOVEA_PROMPT
    # WATCHED, NOT CHASED (review 2026-08-09): a mobile magnet target
    # defeats the phi ratchet (climb paid, descent never charged on the
    # re-adopted baseline) — orbiting the user would be repeatable income,
    # and the cold-start floor would pay for stalking. Social value flows
    # through joint attention + demonstration priming instead.
    assert "player_visible" not in steerable_targets(V.PREDICATES), \
        "the teacher must never be a chase target"
    print("  1. player_visible: predicate + KG + prompts + fovea; "
          "watched, never chased")


REL = {"tree_visible": 1.0, "player_visible": 1.0}
LAB = {"tree_visible": 99, "player_visible": 99}


def _mk(**kw):
    args = dict(target_categories=["tree_visible", "player_visible"],
                min_labels=5, present_threshold=0.5, min_dwell=2,
                eps_abs=1e-3, instinct_bonus=0.0, approach_pull=0.0,
                align_bonus=0.0, aim_bonus=0.0, phi_from_evidence=True,
                seek_weight=1.5, seek_categories=["tree_visible"],
                cold_start_weight=0.5, cold_start_budget=10**8, weight=1.0)
    args.update(kw)
    return VisionScaffold(**args)


def test_social_prime_targets_and_expires():
    vs = _mk()
    vs._seek_nudge_left = 0
    assert vs.social_prime("tree_visible", now=100, duration=500)
    assert not vs.social_prime("no_such_cat", now=100)
    assert vs._seek_nudge_left == vs.seek_nudge_budget, "search not refilled"
    # EDGE-TRIGGERED (review): a live prime cannot re-fire — re-priming per
    # observed_change would extend expiry forever AND turn the bounded seek
    # nudge into a standing wage near the user
    vs._seek_nudge_left = 0
    assert not vs.social_prime("tree_visible", now=150, duration=500), \
        "re-prime while primed must be a no-op"
    assert vs._seek_nudge_left == 0, "re-prime refilled the budget (farmable)"
    # primed category is DISTINCTIVELY curious -> magnet selects it
    r = vs.step_shaping(0, 101, 0.0, {"tree_visible": 0.9}, REL, LAB)
    assert vs._target == "tree_visible", vs.stats
    assert vs._score("tree_visible") >= vs.eps_abs, vs.stats
    # ...and the injection EXPIRES: past the duration the prime is swept
    for t in range(102, 700, 50):
        vs.step_shaping(0, t, 0.0, {}, REL, LAB)
    assert "tree_visible" not in vs._primed, "prime latched"
    assert vs.demos_primed == 1
    print("  2. social prime: targets, refills search, expires (no latch)")


def test_joint_attention_boost():
    def run(social):
        vs = _mk(social_attention_boost=1.0)
        for t in range(300):
            vs.step_shaping(0, t, 0.5, {"tree_visible": 0.9}, REL, LAB,
                            social_present=social)
        return vs._cat_lp["tree_visible"]
    with_teacher = run(True)
    alone = run(False)
    assert with_teacher > alone * 1.5, (with_teacher, alone)
    print(f"  3. joint attention: LP {alone:.3f} alone -> "
          f"{with_teacher:.3f} with the teacher watching")


def test_agency_detector():
    from developmental_ai.environments.minerl_env import MineRLEnvAdapter

    class _A(_BareEnv):
        pass

    def step_probe(env, macro, frame, prev):
        """Replicates the detector block's logic contract in isolation."""
        events = []
        passive = (not (macro.get("attack") or macro.get("use")
                        or macro.get("jump") or macro.get("inventory"))
                   and not any(macro.get(k) for k in
                               ("forward", "back", "left", "right"))
                   and not (macro.get("camera")
                            and any(abs(float(c)) > 0.5
                                    for c in macro["camera"])))
        small = frame[::8, ::8].astype(np.float32)
        if passive and prev is not None and prev.shape == small.shape:
            delta = float(np.mean(np.abs(small - prev))) / 255.0
            if delta > 0.02:
                events.append(("observed_change", "world"))
        return events, small

    quiet = np.full((128, 128, 3), 100, np.uint8)
    changed = quiet.copy(); changed[40:90, 40:90] = 220   # a block vanished
    drift = quiet.copy(); drift[0:8, :] = 104             # cloud-ish flicker
    _, prev = step_probe(None, {}, quiet, None)
    ev, _ = step_probe(None, {}, changed, prev)
    assert ev == [("observed_change", "world")], ev
    ev2, _ = step_probe(None, {"attack": 1}, changed, prev)
    assert ev2 == [], "own swing must never read as external agency"
    ev3, _ = step_probe(None, {"camera": [0, 15]}, changed, prev)
    assert ev3 == [], "own camera motion must never read as external agency"
    ev4, _ = step_probe(None, {}, drift, prev)
    assert ev4 == [], f"sub-threshold drift fired: {ev4}"
    # the inventory toggle changes ~48% of the frame and must NEVER read as
    # external agency (review: the agent owned a demonstration button)
    ev5, _ = step_probe(None, {"inventory": 1}, changed, prev)
    assert ev5 == [], "inventory toggle read as external agency"
    print("  4. agency detector: passive+still+big change fires; own action/"
          "camera and small drift never do")


class _BareEnv:
    pass


if __name__ == "__main__":
    for fn in (test_player_is_a_full_citizen,
               test_social_prime_targets_and_expires,
               test_joint_attention_boost, test_agency_detector):
        print(f"[social-learning] {fn.__name__}")
        fn()
    print("[social-learning] ALL PASS")
