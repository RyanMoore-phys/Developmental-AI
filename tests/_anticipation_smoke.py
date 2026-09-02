"""
_anticipation_smoke — the 2026-09-02 rise-and-reset intrinsic reward
(infra/anticipation.py), plus the VLM 3B downsize + reliability hardening
it shipped alongside.

Run:  PYTHONPATH=. ./venv/bin/python tests/_anticipation_smoke.py

THE DESIGN, restated as the contract this file checks
================================================================================
Per grounded predicate: repeated SIGHTING without the associated caused
effect builds a bounded anticipation level (`wait`); the step the effect
actually fires, the level is paid out — scaled by how long it had built —
and reset to zero; and the payout shrinks toward nothing as the agent gets
independently better at producing that effect (measured via a lifetime event
count, not assumed).

WHY THIS IS NOT THE SAME SHAPE AS EVERY PRIOR FARM
    Sighting is THROTTLED (`sight_throttle_steps`), not per-step. A per-step
    counter pays more the longer the agent stares at something — the exact
    shape of the sky (96% of drive), the menu (77% of income), and holding
    attack on an unreachable trunk (96% of option activity, 0 logs). This
    can only rise on genuine RE-encounters.

WHY THE PAYOUT IS IN `intrinsic`, NEVER `prim_extrinsic`
    It is derived from the grounding head's own inference on the agent's own
    latent. `prim_extrinsic` is the world model's reward LABEL; feeding it a
    quantity derived from the world model's own latent is a closed
    self-feeding loop — the exact contamination infra/consequence.py's
    docstring names as the reason its deficit is ranking-only. Verified here
    at the grep level: both call sites of `_infra_step` add its return
    directly to `intrinsic[0]`.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

from developmental_ai.infra.anticipation import AnticipationMap  # noqa: E402
from developmental_ai.infra.stack import familiarity_curve       # noqa: E402

LOOP = os.path.join("developmental_ai", "core", "developmental_loop.py")


def _map(**over):
    cfg = dict(enabled=True, weight=1.0, sight_threshold=0.5,
              sight_throttle_steps=10, wait_cap=8, assoc_beta=0.3,
              assoc_floor=0.1, base_beta=0.01, max_keys=64)
    cfg.update(over)
    return AnticipationMap(cfg)


def _train_association(m, predicate="tree_visible", key=("break", "oak_log"),
                       rounds=30, step0=0, absent_steps=300):
    """Repeatedly co-occur a sighting with the event so assoc[p][key] clears
    the floor — the earned-linkage substrate every other test depends on.

    ---- ABSENT PERIOD FIRST (added 2026-09-02, after this failed on the pod)
    Association is LIFT: how much MORE present the predicate is when the
    effect fires than it is normally. `_base` initialises to the first
    observed probability, so a helper that only ever shows the predicate
    PRESENT pins the baseline at that same value, lift collapses to ~0, and
    nothing can ever pay.

    That was the module behaving CORRECTLY on an input shape that cannot
    occur in a real run — `tree_visible` is false most of the time — and the
    test asserting it must pay anyway. Measured on the pod: without the
    absent period `assoc` converged to 0.038 (below any floor) and payout
    was 0; with it, `assoc` = 0.749 and payout = 0.748.
    """
    step = step0
    for _ in range(absent_steps):
        m.observe({predicate: 0.05}, [], {}, 1000.0, step)
        step += 1
    for _ in range(rounds):
        m.observe({predicate: 0.95}, [key], {"break:oak_log": 0}, 1000.0, step)
        step += 1
    return step


# ------------------------------------------------------------------ 1 -----
def test_rise_then_reset():
    m = _map()
    step = 0
    # sight WITHOUT the event, throttle-spaced, enough times to hit the cap
    for _ in range(12):
        m.observe({"tree_visible": 0.9}, [], {}, 1000.0, step)
        step += m.sight_throttle_steps
    assert m._wait.get("tree_visible", 0) == m.wait_cap, m._wait
    print(f"  1. {12} throttled sightings without the event -> wait capped "
          f"at {m.wait_cap} (never unbounded)")

    # now build a real association and let the event fire — must pay AND reset
    step = _train_association(m, step0=step)
    before = dict(m._wait)
    payout = m.observe({"tree_visible": 0.95}, [("break", "oak_log")],
                       {"break:oak_log": 0}, 1000.0, step)
    assert payout > 0.0, "trained association + capped wait must pay"
    assert m._wait.get("tree_visible", 0) == 0, (
        "the counter must reset to 0 the step the effect fires")
    print(f"  2. event fires -> payout {payout:.4f} > 0, wait reset "
          f"{before.get('tree_visible')} -> 0")


# ------------------------------------------------------------------ 2 -----
def test_payout_scales_with_wait():
    m1, m2 = _map(), _map()
    _train_association(m1)
    _train_association(m2)
    # m1: only ONE sighting before payout (wait=1). m2: capped wait.
    m1.observe({"tree_visible": 0.9}, [], {}, 1000.0, 10_000)
    for i in range(m2.wait_cap):
        m2.observe({"tree_visible": 0.9}, [], {},
                  1000.0, 10_000 + i * m2.sight_throttle_steps)
    p1 = m1.observe({"tree_visible": 0.95}, [("break", "oak_log")],
                    {"break:oak_log": 0}, 1000.0, 99_999)
    p2 = m2.observe({"tree_visible": 0.95}, [("break", "oak_log")],
                    {"break:oak_log": 0}, 1000.0, 199_999)
    assert p2 > p1 > 0.0, (p1, p2)
    print(f"  3. longer anticipation pays more at the same association: "
          f"wait=1 -> {p1:.4f}, wait={m2.wait_cap} -> {p2:.4f}")


# ------------------------------------------------------------------ 3 -----
def test_familiarity_decay():
    """The user's core requirement: decreases as SkyBot masters it."""
    m = _map()
    _train_association(m)
    m._wait["tree_visible"] = m.wait_cap
    p_novel = m.observe({"tree_visible": 0.95}, [("break", "oak_log")],
                        {"break:oak_log": 0}, 1000.0, 50_000)
    m._wait["tree_visible"] = m.wait_cap
    p_mastered = m.observe({"tree_visible": 0.95}, [("break", "oak_log")],
                           {"break:oak_log": 50_000}, 1000.0, 60_000)
    assert p_mastered < p_novel * 0.05, (p_novel, p_mastered)
    print(f"  4. event_counts 0 -> {p_novel:.4f}, event_counts 50000 (>> "
          f"habituation_scale 1000) -> {p_mastered:.6f} — the channel fades "
          f"as the agent no longer needs the hint, not a bug")

    # exact formula match, not just "it went down"
    fam0 = familiarity_curve(0, 1000.0)
    fam5 = familiarity_curve(50_000, 1000.0)
    assert fam0 == 0.0
    assert abs(fam5 - (1.0 - 1.0 / 51.0)) < 1e-9
    print("  5. familiarity_curve matches the exact formula "
          "category_boringness uses — one mastery clock, not two")


# ------------------------------------------------------------------ 4 -----
def test_no_association_never_pays():
    m = _map()
    for i in range(50):
        m.observe({"sky_visible": 0.99}, [], {}, 1000.0, i * 100)
    assert m._wait.get("sky_visible", 0) == m.wait_cap
    p = m.observe({"sky_visible": 0.99}, [("break", "oak_log")],
                 {"break:oak_log": 0}, 1000.0, 99_999)
    assert p == 0.0, (
        "a predicate with no earned association must never pay, however "
        "often it is sighted — nothing is 'gotten' from seeing the sky")
    print("  6. sky_visible: capped wait, zero association -> pays exactly "
          "0.0 regardless of sighting count")


# ------------------------------------------------------------------ 5 -----
def test_throttle_blocks_farming():
    m = _map(sight_throttle_steps=100)
    for step in range(0, 50):                 # every step, well under throttle
        m.observe({"tree_visible": 0.9}, [], {}, 1000.0, step)
    assert m._wait.get("tree_visible", 0) <= 1, (
        f"wait={m._wait.get('tree_visible')} after 50 UNTHROTTLED steps of "
        f"continuous presence — this is exactly the stare-and-farm shape "
        f"the throttle exists to block")
    print("  7. 50 steps of continuous presence inside one throttle window "
          "raises wait at most once — staring cannot inflate the counter")


# ------------------------------------------------------------------ 6 -----
def test_never_raises_on_malformed_input():
    m = _map()
    for probs, events, counts in [
        (None, None, None), ({}, [], {}), ({"x": float("nan")}, [], {}),
        ({"tree_visible": 0.9}, [("break",)], {}),   # malformed event tuple
        ({"tree_visible": 0.9}, [("break", "oak_log")], None),
    ]:
        try:
            r = m.observe(probs, events, counts, 1000.0, 0)
            assert isinstance(r, float)
        except Exception as e:
            raise AssertionError(f"observe() raised on malformed input: {e}")
    print("  8. observe() never raises on None/NaN/malformed input "
          "(same defensive contract as ConsequenceMap.observe)")


# ------------------------------------------------------------------ 7 -----
def test_persistence_roundtrip():
    m = _map()
    _train_association(m)
    m._wait["tree_visible"] = 5
    st = m.state()
    m2 = _map()
    assert m2.load_state(st)
    assert m2._wait.get("tree_visible") == 5
    assert abs(m2._assoc["tree_visible"]["break:oak_log"]
              - m._assoc["tree_visible"]["break:oak_log"]) < 1e-12
    print("  9. state()/load_state() round-trips wait + earned associations "
          "exactly — a restart must not erase the bootstrap")

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "anticipation_state.json")
        assert m.save(p)
        m3 = _map()
        assert m3.load(p)
        assert m3._wait.get("tree_visible") == 5
    print("  10. save()/load() file round-trip matches (mirrors "
          "ConsequenceMap's save/load contract exactly)")


# ------------------------------------------------------------------ 8 -----
def test_wired_into_intrinsic_not_extrinsic():
    src = open(LOOP).read()
    assert src.count("_ei = self._infra_step(") == 2, (
        "_infra_step must be called from exactly the two waking stepping "
        "bodies documented in its own docstring")
    i_def = src.index("def _infra_step(")
    i_next = src.index("def _scout_mixed_reward(")
    body = src[i_def:i_next]
    assert "self.anticipation.observe(" in body, (
        "the anticipation payout must be computed INSIDE _infra_step, the "
        "single shared method, not duplicated per stepping body")
    assert "prim_extrinsic" not in body.split("self.anticipation.observe(")[1][:400], (
        "the anticipation payout must never be added to prim_extrinsic — "
        "that is the world model's reward label")
    for i in (src.index("_ei = self._infra_step(", 0),
              src.index("_ei = self._infra_step(",
                        src.index("_ei = self._infra_step(") + 1)):
        window = src[i:i + 400]
        assert "intrinsic[0] = intrinsic[0] + _ei" in window, (
            "_infra_step's return (which includes the anticipation payout) "
            "must be added to intrinsic[0] at every call site")
    print("  11. anticipation lives inside the one shared _infra_step, and "
          "both call sites route its return into intrinsic[0] — never "
          "prim_extrinsic")


# ------------------------------------------------------------------ 9 -----
def test_vlm_downsize_and_reprobe_config():
    import yaml
    c = yaml.safe_load(open(os.path.join(
        "configs", "minecraft_skybot.yaml")))["symbolic_grounding"]
    assert c["model"] == "qwen2.5vl:3b-q4", c["model"]
    assert int(c.get("reprobe_after_failures", 0)) > 0
    print(f"  12. symbolic_grounding.model={c['model']}, "
          f"reprobe_after_failures={c['reprobe_after_failures']}")

    llm_src = open(os.path.join(
        "developmental_ai", "llm", "llm_module.py")).read()
    assert "def probe_ollama_model(model: str, deep: bool = True)" in llm_src
    assert "_synthetic_probe_png" in llm_src, (
        "the stage-2 smoke check must exist: tag presence alone cannot "
        "catch a truncated import or a missing vision head")
    print("  13. probe_ollama_model has a deep (real generate-call) stage, "
          "on by default")

    vlm_src = open(os.path.join(
        "developmental_ai", "llm", "vlm_symbolizer.py")).read()
    assert "reprobe_after_failures" in vlm_src
    assert "_consec_failures" in vlm_src
    print("  14. _ollama_scene_query tracks consecutive failures and "
          "re-probes mid-run — a crashed Ollama is no longer silent")


if __name__ == "__main__":
    print("F1. rise-then-reset")
    test_rise_then_reset()
    print("F2. payout scales with anticipation, not just association")
    test_payout_scales_with_wait()
    print("F3. familiarity decay — the user's core 'gets it on its own' spec")
    test_familiarity_decay()
    print("F4. no earned correlate -> never pays")
    test_no_association_never_pays()
    print("F5. throttle blocks the stare-and-farm shape")
    test_throttle_blocks_farming()
    print("F6. defensive contract")
    test_never_raises_on_malformed_input()
    print("F7. persistence")
    test_persistence_roundtrip()
    print("F8. channel correctness (intrinsic, never prim_extrinsic)")
    test_wired_into_intrinsic_not_extrinsic()
    print("F9. VLM downsize + reliability hardening")
    test_vlm_downsize_and_reprobe_config()
    print("[anticipation] ALL PASS")
