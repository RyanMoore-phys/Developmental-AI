"""Semantic novelty smoke (2026-08-11).

THE OBSERVATION THIS ENCODES
    SkyBot was placed directly in front of a tree. It ignored the tree,
    looked down, dug ONE dirt block, and sat in the hole. Perceptual novelty
    explains that exactly: it counts a hash of the VISUAL LATENT, so a
    freshly dug hole is a genuinely never-before-seen view and paid like a
    discovery — while being 34% of all income. Digging MANUFACTURES novelty.

THE CONTRACT
    Novelty is now priced by how new the MEANING in view is:
      * unseen symbols in view (a first village, a first tree) -> ~1.0, so
        novelty SPIKES on genuine discovery;
      * long-familiar named things (dirt seen thousands of times) -> ~0, so
        digging a hole stops paying;
      * nothing nameable in view -> a small non-zero floor, because the
        vocabulary's silence may mean new territory;
      * familiar-NAMED decays BELOW the unnamed floor (asymmetry is the
        point: naming something you've seen 4000 times is less interesting
        than not being able to name it at all);
      * a STALE reading (dream-actor steps skip _symbol_novelty) falls back
        to the floor rather than mispricing by an old view;
      * the whole thing is OFF by default, so other configs are unchanged.

Run: PYTHONPATH=. python tests/_semantic_novelty_smoke.py
"""
import os
import sys

sys.path.insert(0, ".")


class _Bare:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _factor(**kw):
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    s = _Bare(_nov_semantic=True, _nov_unnamed_floor=0.1,
              total_timesteps=1000, _sym_rarity_step=1000, **kw)
    return DevelopmentalAI._semantic_novelty_factor(s)


def test_discovery_spikes():
    # a symbol never seen before -> rarity 1/sqrt(1+0) = 1.0
    f = _factor(_sym_rarity_now=1.0, _sym_named_now=1)
    assert f == 1.0, f
    # a village-ish scene: several symbols, the RAREST governs
    f2 = _factor(_sym_rarity_now=1.0, _sym_named_now=4)
    assert f2 == 1.0, f2
    print(f"  1. first sighting of a nameable thing pays FULL novelty "
          f"({f:.2f}) — discovery spikes")


def test_dug_hole_stops_paying():
    # `dirt` named 4444 times (the real lifetime count from the training host)
    rarity = 1.0 / ((1.0 + 4444) ** 0.5)
    f = _factor(_sym_rarity_now=rarity, _sym_named_now=1)
    assert f < 0.02, f
    # ...and it is BELOW the unnamed floor: familiar-named is less
    # interesting than un-nameable
    assert f < 0.1, f
    print(f"  2. a view of 4444x-seen `dirt` pays {f:.4f} of novelty "
          f"(<2%, and below the {0.1} unnamed floor) — digging defunded")


def test_unnamed_keeps_a_floor():
    # nothing the vocabulary can name -> small but non-zero pull
    f = _factor(_sym_rarity_now=0.0, _sym_named_now=0)
    assert f == 0.1, f
    print(f"  3. an UNNAMEABLE view keeps the {f:.2f} floor — silence from "
          f"the vocabulary may mean new territory, not nothing")


def test_stale_reading_falls_back():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    # _symbol_novelty is skipped on dream-actor steps: the stash is from an
    # older step and must NOT be used to price this one
    s = _Bare(_nov_semantic=True, _nov_unnamed_floor=0.1,
              total_timesteps=1000, _sym_rarity_step=930,
              _sym_rarity_now=1.0, _sym_named_now=3)
    f = DevelopmentalAI._semantic_novelty_factor(s)
    assert f == 0.1, f
    # and a total absence of the stash is equally safe
    s2 = _Bare(_nov_semantic=True, _nov_unnamed_floor=0.1,
               total_timesteps=1000)
    assert DevelopmentalAI._semantic_novelty_factor(s2) == 0.1
    print("  4. a STALE or missing reading falls back to the floor, never "
          "prices this step by an old view")


def test_off_by_default_and_never_raises():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    # absent config -> exactly the old behaviour (factor 1.0 = no change)
    assert DevelopmentalAI._semantic_novelty_factor(_Bare()) == 1.0
    # a broken internal state must not kill the step
    bad = _Bare(_nov_semantic=True, _nov_unnamed_floor="not-a-number",
                total_timesteps=1, _sym_rarity_step=1,
                _sym_named_now=1, _sym_rarity_now=None)
    assert DevelopmentalAI._semantic_novelty_factor(bad) == 1.0
    print("  5. OFF by default (factor 1.0, other configs byte-identical) "
          "and a malformed state degrades to 1.0 rather than raising")


def test_config_wired():
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "..", "configs",
                           "minecraft_skybot.yaml")) as f:
        cur = yaml.safe_load(f)["curiosity"]
    assert cur["novelty_semantic"] is True
    assert float(cur["novelty_weight"]) <= 0.10, cur["novelty_weight"]
    assert 0.0 < float(cur["novelty_unnamed_floor"]) < 0.5
    # the ledger line must report the new name
    src = open(os.path.join(os.path.dirname(__file__), "..",
                            "developmental_ai", "core",
                            "developmental_loop.py")).read()
    assert '("novelty", "_nov_sum")' in src
    assert '("view_novelty", "_nov_sum")' not in src, \
        "the old ledger name survives — the viewer would show two terms"
    print(f"  6. skybot config: weight={cur['novelty_weight']} "
          f"semantic=on floor={cur['novelty_unnamed_floor']}; ledger "
          f"renamed view_novelty -> novelty")


def test_activity_never_outweighs_learning():
    """THE STANDING PRINCIPLE (user, 2026-08-12): general activity must not
    be weighted anywhere near the actual learning process.

    Three terms pay for ACTIVITY regardless of meaning — coverage (distance
    travelled), novelty (unseen views) and gaze (where the camera points).
    Two pay for LEARNING — symbol_weight (naming a new set of things) and
    new_symbol_bonus (acquiring a concept). Each activity term must sit
    strictly below every learning term.

    progress_weight USED to be the third learning term. Since 2026-10-05
    (user-accepted) it is MEASUREMENT ONLY and pays nothing: a world-model
    improvement does not establish credit for the action taken now, and the
    per-step rate was an ambient wage. Leaving it in the comparison made
    this test pass trivially on a weight that buys nothing, so the contract
    for it is now the opposite one: measured, never paid —
      * no reward path: the loop never calls record_reward("progress"...)
        and never multiplies _progress_weight into anything (it is read
        only as the on/off switch for the measurement);
      * ProbeSetProgress.rate() is 0 even right after a real improvement.

    This is a REGRESSION guard, not a style check. Coverage has been retuned
    four times (0.45 -> 0.10 -> 0.30 -> 0.07) and each raise was locally
    justified — "something has to make the body move" — while producing an
    agent paid mostly to wander (measured: coverage 65% of the ledger).
    """
    import re
    import yaml
    import torch
    from developmental_ai.infra.progress_curiosity import ProbeSetProgress
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "..", "configs",
                           "minecraft_skybot.yaml")) as f:
        cur = yaml.safe_load(f)["curiosity"]
    activity = {k: float(cur[k]) for k in
                ("coverage_weight", "novelty_weight", "gaze_weight")}
    learning = {k: float(cur[k]) for k in
                ("symbol_weight", "new_symbol_bonus")}
    worst_learn = min(learning.values())
    assert worst_learn > 0, learning
    for name, w in sorted(activity.items(), key=lambda kv: -kv[1]):
        assert w < worst_learn, (
            f"{name}={w} >= the weakest learning term ({worst_learn}) — "
            f"activity is being paid like discovery again")
    # and coverage specifically must not dominate the activity block either
    assert activity["coverage_weight"] <= 0.10, activity

    # progress: measured, never paid
    with open(os.path.join(here, "..", "developmental_ai", "core",
                           "developmental_loop.py")) as f:
        src = f.read()
    assert not re.search(r"record_reward\(\s*[\"']progress[\"']", src), \
        "a progress payment path is back in the loop"
    uses = [ln.strip() for ln in src.splitlines() if "_progress_weight" in ln]
    mult = re.compile(r"\*\s*(self\.)?_progress_weight|_progress_weight\s*\*")
    for ln in uses:
        assert not mult.search(ln.split("#")[0]), \
            f"_progress_weight used as a multiplier: {ln}"
    assert any("self._progress_weight > 0.0" in ln for ln in uses), uses
    p = ProbeSetProgress(capacity=3, eval_every=1)
    for i in range(3):
        p.maybe_add_probe({"t": torch.tensor([[float(i)]])})
    p.evaluate(lambda b: 1.0, 0, model_version="v0")
    assert p.evaluate(lambda b: 0.25, 1, model_version="v1") > 0  # measured
    assert p.rate() == 0.0 and p.stats["rate"] == 0.0             # unpaid
    print(f"  7. economy shape: activity {activity} all strictly below "
          f"learning {learning}; progress_weight={cur['progress_weight']} "
          f"is measurement only (measured > 0, paid 0)")


if __name__ == "__main__":
    for fn in (test_discovery_spikes, test_dug_hole_stops_paying,
               test_unnamed_keeps_a_floor, test_stale_reading_falls_back,
               test_off_by_default_and_never_raises, test_config_wired,
               test_activity_never_outweighs_learning):
        print(f"[semantic-novelty] {fn.__name__}")
        fn()
    print("[semantic-novelty] ALL PASS")
