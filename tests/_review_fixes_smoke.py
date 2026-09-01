"""Four confirmed code-review findings (2026-08-23) — regression contracts.

WHAT THIS FILE DEFENDS

A. A PARTIAL WORLD-MODEL RESTORE REPORTED AS CLEAN, AND OPENED THE GATE.
   `_load_wm` called `load_state_dict(..., strict=False)` — necessary, because
   the reward head must keep its fresh init — and then re-implemented only the
   `unexpected` half of the check by hand. `_missing` was bound and never
   read. So a checkpoint missing whole submodules (arch drift, an older brain,
   or a TRUNCATED file: world_model.pt is written non-atomically) loaded
   silently, `world_model` was appended to `_ok`, and the dependency gate that
   asks `"world_model" not in _ok` therefore PASSED — restoring the symbol
   head and the LSH novelty counters onto a half-fresh encoder. That is the
   exact pairing the gate exists to prevent, and its own comment states the
   consequence: "the head reads noise, and the novelty counters mark unseen
   views as already-visited, which zeroes exploration income exactly when it
   is needed most." A silent zero-exploration run that logs as a good resume.

B. THE MAGNET WAS GATED ON A DEPENDENCY IT DOES NOT HAVE.
   `magnet` sat in that same tuple by association. VisionScaffold.state() is
   five category-NAME-keyed scalar containers — learning progress for "tree",
   not a latent anything. Gating it meant that whenever the world model failed
   to restore, the curiosity memory was ALSO discarded and the agent booted at
   `w=0.0000, target=None`: the full cold-start tax magnet.pt exists to
   remove, paid silently, in the one situation where it hurts most. The log
   line even gave an untrue reason ("needs world_model").

C. A QUARTER OF THE ADVISOR'S VOCABULARY WAS DEAD CODE.
   The stuck ladder applies its L2 remedy (`if _lvl >= 2 and
   _stuck_boost_orig is None`) BEFORE calling the advisor (`if _lvl >= 3`) in
   the same segment hook. Since L3 satisfies `>= 2`, `_stuck_boost_orig` was
   always non-None by the time the advisor ran, so its branch — `elif _r ==
   "explore_wider" and self._stuck_boost_orig is None` — could never be taken.
   The VLM's answer was printed and written to help_responses.jsonl as
   ACCEPTED ADVICE while nothing happened, corrupting the only record of
   whether the advisor helps at all.

D. THE ENV TIMER WAS DUPLICATED BUT ITS ACCUMULATOR WAS NOT.
   `_t_env0 = time.time()` sits in both stepping bodies under a comment
   reading "Set in BOTH duplicated bodies on purpose: letting these two drift
   is how the last six same-shape bugs were born." The matching
   `_env_wait_sum` accumulation existed only in `_collect_segment` — so
   `_run_episode_parallel` had an unused local and never emitted the `Loop
   timing:` line at all. Seventh instance of the class the comment warns
   about, which is why this is now a mechanical guard and not another careful
   edit.

Run: PYTHONPATH=. python tests/_review_fixes_smoke.py
"""
import os
import sys
import types

sys.path.insert(0, ".")

SRC = os.path.join("developmental_ai", "core", "developmental_loop.py")


# ---------------------------------------------------------------- A ------
def test_A_world_model_restore_must_not_silently_partial():
    from developmental_ai.core.developmental_loop import (
        _wm_restore_absent, _WM_ECONOMY_PREFIXES)

    # A0 — THE WITNESS. Demonstrate the old behaviour against real torch
    #      rather than asserting it from memory: strict=False accepts a
    #      payload that cannot fill the model, returns the shortfall in
    #      `missing`, and RAISES NOTHING. Code that ignores that return value
    #      has no way to know the restore was partial.
    import torch.nn as nn
    m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    full = m.state_dict()
    partial = {k: v for k, v in full.items() if not k.startswith("0.")}
    missing, unexpected = m.load_state_dict(partial, strict=False)
    assert list(unexpected) == [], unexpected
    assert set(missing) == {"0.weight", "0.bias"}, missing
    print(f"  A0. WITNESS: strict=False loaded a payload missing "
          f"{sorted(missing)} without raising — layer 0 is silently FRESH. "
          f"Ignoring `missing` is what made a partial restore report clean")

    MODEL = ["encoder.0.weight", "encoder.0.bias", "rssm.gru.weight",
             "rssm.gru.bias", "decoder.0.weight",
             "reward_predictor.0.weight", "reward_bins"]
    ECON = {k for k in MODEL if k.startswith(_WM_ECONOMY_PREFIXES)}
    BODY = [k for k in MODEL if k not in ECON]

    # A1 — the NORMAL restore: full checkpoint, reward head dropped on purpose
    assert _wm_restore_absent(MODEL, BODY, ECON) == [], (
        "the ordinary reward-head-dropped restore must stay clean")
    print("  A1. full payload, reward head deliberately dropped -> no "
          "complaint (the normal path is not broken)")

    # A2 — THE REGRESSION: a whole submodule absent from the checkpoint
    partial = [k for k in BODY if not k.startswith("encoder.")]
    absent = _wm_restore_absent(MODEL, partial, ECON)
    assert absent == ["encoder.0.bias", "encoder.0.weight"], absent
    print(f"  A2. checkpoint missing a submodule -> flags {absent} "
          f"(this is what loaded SILENTLY and opened the gate)")

    # A3 — absent reward head that we did NOT drop is a real defect, not an
    #      excuse. This is why exemption is keyed to what we removed rather
    #      than to a prefix match on what came back missing.
    absent3 = _wm_restore_absent(MODEL, BODY, set())
    assert set(absent3) == ECON, absent3
    print("  A3. reward head absent with resume_reward_head=true (nothing "
          "dropped) -> flagged, NOT excused by prefix")

    # A4 — ...and the same keys ARE excused when we did drop them
    assert _wm_restore_absent(MODEL, BODY, ECON) == []
    print("  A4. the same keys, excused when we removed them ourselves")

    # A5 — a superset payload is the `unexpected` check's job, not this one
    assert _wm_restore_absent(MODEL, MODEL + ["ghost.weight"], set()) == []
    print("  A5. extra keys are not this check's business (unexpected-keys "
          "guard still raises on them)")

    # A6 — truncated write: only the first tensors made it to disk
    trunc = BODY[:2]
    absent6 = _wm_restore_absent(MODEL, trunc, ECON)
    assert len(absent6) == len(BODY) - 2, absent6
    print(f"  A6. truncated checkpoint (2 of {len(BODY)} tensors) -> flags "
          f"the other {len(absent6)} (non-atomic write makes this real)")

    # A7 — the check must run BEFORE any tensor is copied, or the model is
    #      left a hybrid: torch copies matching tensors and raises only at
    #      the end.
    src = open(SRC).read()
    i_pre = src.index("_absent = _wm_restore_absent(")
    i_load = src.index("_missing, _unexpected = self.world_model.load_state_dict")
    assert i_pre < i_load, (
        "pre-flight must precede load_state_dict — a check made afterwards "
        "leaves a partial hybrid behind")
    print("  A7. pre-flight precedes load_state_dict (refusal leaves the "
          "world model pristine, so it stays out of _ok and the gate holds)")

    # A8 — `_missing` may never be bound and ignored again
    assert "_unexplained" in src and "_unexplained:" in src, (
        "the post-load _missing result must still be examined")
    print("  A8. post-load _missing is examined, not discarded")

    # A9 — a clean refusal must not be reported as a possible hybrid. The
    #      generic handler's "torch copies matching tensors BEFORE raising"
    #      warning is true of a shape mismatch and FALSE of a pre-flight
    #      refusal; this repo's rule that a degraded outcome must never read
    #      as a clean one holds in the other direction too.
    assert "class _ResumePreflightRefusal" in src
    assert "_skip.append(f\"{_n}(refused:CLEAN)\")" in src
    i_clean = src.index("refused:CLEAN")
    i_partial = src.index('_skip.append(f"{_n}({type(_e).__name__}:PARTIAL?)")')
    assert i_clean < i_partial, (
        "the clean-refusal branch must be tested before the generic "
        "partial-hybrid warning, or a refusal is reported as corruption")
    print("  A9. a pre-flight refusal reports as CLEAN/fresh, not as a "
          "possible partial hybrid (no false corruption alarm)")


# ---------------------------------------------------------------- B ------
def test_B_magnet_is_not_encoder_keyed():
    from developmental_ai.core.developmental_loop import (
        _RESUME_ENCODER_KEYED)

    # B1 — membership
    assert "perception" in _RESUME_ENCODER_KEYED
    assert "familiarity" in _RESUME_ENCODER_KEYED
    assert "magnet" not in _RESUME_ENCODER_KEYED, (
        "magnet was re-added to the encoder-keyed gate — its state is "
        "category-name-keyed and valid against any encoder; gating it throws "
        "away curiosity memory on exactly the resume that can least afford it")
    print(f"  B1. _RESUME_ENCODER_KEYED = {_RESUME_ENCODER_KEYED} — magnet "
          f"exempt, the two genuinely latent-keyed components are not")

    # B2 — THE GUARD ON THE EXEMPTION. The exemption is only sound while the
    #      magnet's persisted state stays name-keyed. Pin the key set so a
    #      future encoder-derived field cannot be added invisibly.
    from developmental_ai.llm.vision_scaffold import VisionScaffold
    vs = VisionScaffold()
    st = vs.state()
    EXPECTED = {"cat_lp", "global_lp", "lp_scale", "ever_curious",
                "cat_absent"}
    assert set(st) == EXPECTED, (
        f"VisionScaffold.state() key set changed: got {sorted(st)}, expected "
        f"{sorted(EXPECTED)}.\nDECIDE ON PURPOSE: if the new field is derived "
        f"from the world-model encoder (a latent centroid, an LSH bucket, any "
        f"tensor), `magnet` must go BACK into _RESUME_ENCODER_KEYED — it "
        f"would otherwise be restored onto a fresh encoder. If it is another "
        f"name-keyed scalar, add it to EXPECTED here.")
    print(f"  B2. state() is exactly {sorted(EXPECTED)} — every entry keyed "
          f"by CATEGORY NAME, nothing derived from the encoder")

    # B3 — and every value is a name-keyed container or a plain scalar, not a
    #      tensor/array smuggled in under an innocent name
    for k, v in st.items():
        assert isinstance(v, (dict, float, int, bool)), (
            f"state()['{k}'] is {type(v).__name__} — a non-scalar payload is "
            f"how an encoder dependency would sneak in")
        if isinstance(v, dict):
            assert all(isinstance(kk, str) for kk in v), (
                f"state()['{k}'] has non-string keys — category names only")
    print("  B3. all values are name-keyed dicts or scalars (no tensors)")

    # B4 — round-trip is meaningful with no encoder in sight
    vs._cat_lp["tree"] = 0.42
    vs._global_lp = 0.11
    payload = vs.state()
    fresh = VisionScaffold()
    fresh.load_state(payload)
    assert abs(fresh._cat_lp.get("tree", 0.0) - 0.42) < 1e-9
    assert abs(fresh._global_lp - 0.11) < 1e-9
    print("  B4. state -> load_state carries learning progress into a NEW "
          "scaffold with no world model involved at all")

    # B5 — the 19-hour-stall latch stays un-persisted. This is the exact
    #      place someone would "helpfully" add it.
    assert "_cold_spent" not in st, (
        "_cold_spent is back in the persisted state — carrying a SPENT "
        "cold-start budget across processes is the latch behind the 19-hour "
        "zero-reward stall, promoted from run-scoped to permanent")
    print("  B5. _cold_spent still excluded (the 19h zero-reward latch stays "
          "run-scoped)")

    # B6 — the gate reads the named constant, not an inline tuple
    src = open(SRC).read()
    assert "if (_n in _RESUME_ENCODER_KEYED" in src
    assert '_n in ("perception", "familiarity", "magnet")' not in src
    print("  B6. the resume gate reads _RESUME_ENCODER_KEYED (the reasoning "
          "has one home, and this test has something to assert on)")


# ---------------------------------------------------------------- C ------
def _stub(t=1000, nov=0.10, cov=0.20):
    return types.SimpleNamespace(
        total_timesteps=t, _novelty_weight=nov, _coverage_weight=cov,
        _stuck_boost_orig=None, _stuck_boost_until=-1,
        _stuck_boost_started=-1)


def test_C_explore_boost_is_reachable_bounded_and_shared():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    boost = DevelopmentalAI._apply_explore_boost

    # C0 — THE WITNESS, kept as a regression exhibit: the OLD branch under the
    #      OLD ordering. L2 sets _stuck_boost_orig, then the advisor's
    #      `elif _r == "explore_wider" and self._stuck_boost_orig is None`
    #      tests it and finds it non-None. Every time. Forever.
    s0 = _stub()
    s0._stuck_boost_orig = (s0._novelty_weight, s0._coverage_weight)  # L2 ran
    fired = s0._stuck_boost_orig is None                # the old precondition
    assert fired is False, (
        "the old precondition could not be satisfied at L3 — that was the bug")
    print("  C0. WITNESS: under the live L2-then-L3 order the old "
          "`_stuck_boost_orig is None` precondition is False on every "
          "reachable path -> explore_wider was unreachable by construction")

    # C1 — fresh application
    s = _stub()
    st = boost(s, 2048)
    assert s._stuck_boost_orig == (0.10, 0.20)
    assert abs(s._novelty_weight - 0.20) < 1e-12
    assert abs(s._coverage_weight - 0.40) < 1e-12
    assert s._stuck_boost_until == 1000 + 2048
    assert s._stuck_boost_started == 1000
    print(f"  C1. fresh apply -> {st!r}; weights exactly x2, originals saved")

    # C2 — THE REGRESSION. Replay the live order: L2 fires, then L3 reaches
    #      the advisor in the SAME segment hook. Under the old code the
    #      second call was unreachable and did nothing.
    s = _stub()
    boost(s, 2048)                       # L2 remedy
    st2 = boost(s, 4096)                 # advisor: explore_wider
    assert "declined" not in st2, st2
    assert s._stuck_boost_until == 1000 + 4096, s._stuck_boost_until
    print(f"  C2. L2 then advisor (the live order) -> {st2!r}; window "
          f"extended 2048 -> 4096. This call previously did NOTHING while "
          f"being logged as accepted advice")

    # C3 — weights must not compound across calls
    assert abs(s._novelty_weight - 0.20) < 1e-12, s._novelty_weight
    assert abs(s._coverage_weight - 0.40) < 1e-12
    print("  C3. still x2 after two calls, never x4 (extension re-times the "
          "boost, it does not re-multiply)")

    # C4 — extension is CAPPED: a rate-limited advisor answering
    #      explore_wider forever must not hold the weights up forever
    s = _stub(t=0)
    boost(s, 2048)
    seen = []
    for k in range(1, 9):
        s.total_timesteps = 4096 * k     # advisor_min_gap apart
        seen.append(boost(s, 4096))
    assert s._stuck_boost_until <= 0 + 16384, s._stuck_boost_until
    assert any("declined" in x for x in seen), seen
    print(f"  C4. 8 consecutive explore_wider calls -> expiry capped at "
          f"{s._stuck_boost_until} (started+16384); later calls return "
          f"{seen[-1]!r}. A remedy that never expires is guard-becomes-latch "
          f"wearing the other hat")

    # C5 — a shorter window must never pull an existing expiry IN
    s = _stub(t=0)
    boost(s, 4096)
    st5 = boost(s, 100)
    assert s._stuck_boost_until == 4096, s._stuck_boost_until
    assert st5 == "declined (boost already runs longer)", (
        f"a redundant short window must be distinguishable in the log from a "
        f"CAPPED one — an operator reads these to tell a bounded system "
        f"working from a remedy arriving late; got {st5!r}")
    print(f"  C5. a shorter window is declined, not applied -> {st5!r} "
          f"(expiry stays {s._stuck_boost_until})")

    # C6 — expiry restores the EXACT originals (bit-for-bit, not recomputed
    #      by halving: a doubled-then-halved float is not always the original)
    s = _stub(nov=0.1234567, cov=0.7654321)
    boost(s, 10)
    s._novelty_weight, s._coverage_weight = s._stuck_boost_orig
    assert s._novelty_weight == 0.1234567 and s._coverage_weight == 0.7654321
    print("  C6. expiry restores the exact original floats from the saved "
          "tuple (never by halving)")

    # C7 — source: the dead precondition is gone, both call sites share the
    #      one bounded implementation, and the advisor LOGS its status so a
    #      declined extension is visibly declined rather than silently
    #      reported as accepted
    src = open(SRC).read()
    assert '_r == "explore_wider" and self._stuck_boost_orig is None' \
        not in src, (
            "the unreachable precondition is back — L2 always fires first, "
            "so this makes explore_wider dead code again")
    assert src.count("self._apply_explore_boost(") == 2, (
        "both the L2 ladder and the advisor must use the shared helper")
    assert 'print("  infra/advisor: explore_wider -> "' in src, (
        "the advisor must print the returned status; logging advice as "
        "accepted while it was declined is the reporting bug this fixes")
    print("  C7. dead precondition removed; L2 + advisor share one bounded "
          "helper; the advisor prints what actually happened")

    # C8 — THE FINDING, GENERALISED: no remedy in the menu may be dead. The
    #      real bug was that one could be, with nothing noticing.
    from developmental_ai.infra.advisor import REMEDIES
    import inspect
    body = inspect.getsource(DevelopmentalAI._advise_unstuck)
    NOOP_BY_DESIGN = {"conserve"}        # documented: doing nothing is an answer
    for r in REMEDIES:
        if r in NOOP_BY_DESIGN:
            assert r in body, f"{r} is not even mentioned in _advise_unstuck"
            continue
        assert f'_r == "{r}"' in body, (
            f"remedy '{r}' is offered in the VLM's menu but has no handler in "
            f"_advise_unstuck — it would be logged as accepted advice and do "
            f"nothing, which is exactly the explore_wider bug")
    print(f"  C8. every remedy in {list(REMEDIES)} has a live handler "
          f"(conserve is a documented no-op) — the NEXT dead remedy fails "
          f"here too")


# ---------------------------------------------------------------- D ------
def test_D_env_timing_present_in_both_bodies():
    src = open(SRC).read()

    n_t0 = src.count("_t_env0 = time.time()")
    n_sum = src.count("self._env_wait_sum = (getattr")
    n_n = src.count("self._env_wait_n = getattr")
    assert n_t0 == 2, f"_t_env0 set in {n_t0} bodies, expected 2"
    assert n_sum == 2, (
        f"_env_wait_sum accumulated in {n_sum} bodies, expected 2 — this is "
        f"the drift: the timer was duplicated and the accumulator was not, "
        f"so one body had an unused local and never emitted `Loop timing:`")
    assert n_n == 2, f"_env_wait_n incremented in {n_n} bodies, expected 2"
    print(f"  D1. timer x{n_t0}, sum x{n_sum}, count x{n_n} — present in "
          f"both stepping bodies")

    # D2 — counting is not enough: both halves could sit in the SAME body.
    #      Require strict alternation timer -> accumulator -> timer -> ...
    marks = []
    for tok, tag in (("_t_env0 = time.time()", "timer"),
                     ("self._env_wait_sum = (getattr", "accum")):
        i = 0
        while True:
            j = src.find(tok, i)
            if j < 0:
                break
            marks.append((j, tag))
            i = j + 1
    order = [t for _, t in sorted(marks)]
    assert order == ["timer", "accum", "timer", "accum"], (
        f"timer/accumulator order is {order} — each _t_env0 must be consumed "
        f"before the next one is set, or both halves are in one body while "
        f"the counts still look right")
    print(f"  D2. source order is {order} — each timer is consumed by the "
          f"accumulator in its own body")

    # D3 — the diagnostic still has a consumer
    assert "Loop timing:" in src and "self._env_wait_sum = 0.0" in src
    print("  D3. `Loop timing:` print + reset still present (the line this "
          "instrumentation exists to produce)")


if __name__ == "__main__":
    print("A. world-model restore must not be silently partial")
    test_A_world_model_restore_must_not_silently_partial()
    print("B. the magnet is not keyed to the encoder")
    test_B_magnet_is_not_encoder_keyed()
    print("C. explore_wider is reachable, bounded, and shared")
    test_C_explore_boost_is_reachable_bounded_and_shared()
    print("D. env timing lives in both stepping bodies")
    test_D_env_timing_present_in_both_bodies()
    print("[review-fixes] ALL PASS")
