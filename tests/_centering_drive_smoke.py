"""Smoke: the 2026-08-02 "seeing and aiming are one drive" wave (A-D).

THE SYMPTOM: SkyBot sat with its pitch clamped at +90 (straight down),
never looking up (lookUP 1-3%), while the symbol reward that was supposed
to pull its view toward interesting things paid +0.008/step and did not
care WHERE anything was.

  A. The approach potential was a THRESHOLD LADDER
     (0.85 if adj>=0.5 else 0.6 if cen>=0.5 else 0.35). As a potential that
     is useless for aiming: p(centered) 0.10 -> 0.49 paid EXACTLY ZERO and
     0.49 -> 0.51 paid the whole step. No gradient to follow.
  B. A DEAD ZONE between seek and approach: _seek_shaping returned 0 the
     instant the goal became trusted-present, and _phi_head only ran when
     the magnet weight was non-zero AND the target was present. "Goal
     visible but off-centre" — the state that needs a turn — fell between.
  C. The symbol drive is position-blind by construction: it scores the SET
     of predicates above the trust threshold, so centring cannot pay.
  D. policy_entropy and the magnet turn/other split were recorded ONLY in
     _run_episode_parallel; skybot runs _collect_segment, so the COLLAPSED
     detector and the "Magnet pay" readout never printed in the mode that
     matters. The action histogram walked attack 11% -> 55% -> 70% unseen.

Run: PYTHONPATH=. python tests/_centering_drive_smoke.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np

LOOP = "developmental_ai/core/developmental_loop.py"
CFG = "configs/minecraft_skybot.yaml"


def _scaffold(**kw):
    from developmental_ai.llm.vision_scaffold import VisionScaffold
    base = dict(enabled=True, weight=0.6, seek_weight=0.5,
                seek_categories=["tree_visible"], seek_forward_nudge=0.0,
                min_labels=1, reliability_floor=0.0, present_threshold=0.6)
    base.update(kw)
    return VisionScaffold(**base)


def main() -> None:
    import yaml
    loop_src = open(LOOP).read()
    cfg = yaml.safe_load(open(CFG))

    # ---- A. the approach potential is CONTINUOUS ------------------------
    vs = _scaffold()
    vs._target = None                      # isolate from the focus factor
    phis = [vs._phi_head({"object_centered": c, "object_adjacent": 0.0})
            for c in (0.0, 0.1, 0.3, 0.49, 0.51, 0.8, 1.0)]
    assert all(b > a + 1e-9 for a, b in zip(phis, phis[1:])), (
        f"phi is not strictly increasing in p(centered): {phis} — a flat "
        "region means re-centring pays nothing over that range")
    # the sub-threshold range that used to be dead must now pay
    assert phis[3] - phis[1] > 1e-3, (
        "0.10 -> 0.49 still pays ~nothing; this is the dead ladder")
    # landmarks preserved so existing tuning still means what it meant
    assert abs(vs._phi_head({"object_centered": 0.0,
                             "object_adjacent": 0.0}) - 0.35) < 1e-9
    assert abs(vs._phi_head({"object_centered": 1.0,
                             "object_adjacent": 1.0}) - 1.00) < 1e-9
    # bounded => Ng telescoping still applies
    for c in np.linspace(0, 1, 11):
        for a in np.linspace(0, 1, 11):
            v = vs._phi_head({"object_centered": float(c),
                              "object_adjacent": float(a)})
            assert 0.0 <= v <= 1.0, f"phi left [0,1] at ({c},{a}): {v}"

    # ---- B. no dead zone: a VISIBLE but off-centre goal still pays ------
    vs2 = _scaffold()
    P = {"tree_visible": 0.9, "object_centered": 0.0}
    R, L = {"tree_visible": 1.0}, {"tree_visible": 99}
    present = {"tree_visible"}
    vs2._seek_shaping(0, P, R, L, present)          # adopt the baseline
    r_centre = vs2._seek_shaping(
        0, {"tree_visible": 0.9, "object_centered": 1.0}, R, L, present)
    assert r_centre > 1e-6, (
        f"centring a VISIBLE goal paid {r_centre} — the seek->approach dead "
        "zone is still open")
    # ...and it telescopes: turning back gives it all up again
    r_back = vs2._seek_shaping(
        0, {"tree_visible": 0.9, "object_centered": 0.0}, R, L, present)
    assert abs(r_centre + r_back) < 1e-6, (
        f"look-toward/look-away did not net ~0 ({r_centre} + {r_back}) — the "
        "potential is farmable")
    # a goal that is NOT in frame must not pay for centring anything
    vs3 = _scaffold()
    vs3._seek_shaping(0, {"tree_visible": 0.0, "object_centered": 0.0},
                      R, L, set())
    r_none = vs3._seek_shaping(
        0, {"tree_visible": 0.0, "object_centered": 1.0}, R, L, set())
    assert abs(r_none) < 1e-9, (
        f"centring paid {r_none} with the goal off-screen — this would become "
        "a generic 'point at anything' bonus")

    # ---- C. the symbol drive now has a centring potential ---------------
    from developmental_ai.core.developmental_loop import DevelopmentalAI

    class _Sym:
        fact_threshold, min_labels, reliability_floor = 0.7, 1, 0.0

    class _Stub:
        symbolizer = _Sym()
        _known_symbols = set()
        _symbol_counts = {}
        _new_symbol_bonus = 0.0
        _symbol_weight = 0.0
        def __init__(self, probs): self._p = probs
        def _grounded_object_probs(self, _):
            return self._p, {k: 1.0 for k in self._p}, {k: 9 for k in self._p}

    def phi_for(probs, repeats=1):
        s = _Stub(probs)
        for _ in range(repeats):
            DevelopmentalAI._symbol_novelty(s, object())
        return getattr(s, "_sym_center_phi", None)

    fresh = phi_for({"tree_visible": 0.9, "object_centered": 1.0})
    assert fresh is not None, (
        "_sym_center_phi was never set — _symbol_novelty swallows exceptions, "
        "so a silent failure here looks like a pass")
    assert fresh > 0.9, f"centring an unfamiliar symbol scored only {fresh}"
    # off-centre must score strictly lower for the SAME symbol
    off = phi_for({"tree_visible": 0.9, "object_centered": 0.0})
    assert off < fresh, f"off-centre {off} did not score below centred {fresh}"
    # familiarity decays the pull (1/sqrt(n)), so it cannot be farmed forever
    worn = phi_for({"tree_visible": 0.9, "object_centered": 1.0}, repeats=50)
    assert worn < 0.25 * fresh, (
        f"pull did not decay with familiarity ({worn} vs {fresh})")
    # a bare affordance is not a "thing" to centre on
    afford = phi_for({"object_centered": 1.0, "object_adjacent": 1.0})
    assert afford == 0.0, (
        f"centring on affordance predicates alone scored {afford} — that pays "
        "for pointing at anything")
    # wired as a TELESCOPING potential, never as raw income
    assert "_sym_center_prev" in loop_src and "_gc * _phi_c - _prev_c" in loop_src, \
        "the centring term is not applied as gamma*Phi' - Phi"
    assert float(cfg["curiosity"]["symbol_center_weight"]) > 0.0, \
        "skybot does not enable the centring drive"

    # ---- E. GAZE COVERAGE unsticks a clamped pitch ----------------------
    class _G:
        _gaze_weight = 0.25
        _gaze_bucket_deg = 15.0
        def __init__(self): self._gaze_counts = {}
    g = _G()
    gb = DevelopmentalAI._gaze_bonus
    # a fresh direction pays ~the full weight
    first = gb(g, 0.0)
    assert abs(first - 0.25) < 1e-9, f"a new pitch bucket must pay: {first}"
    # STARING decays as 1/sqrt(n): being pinned at a clamp must go to ~0
    for _ in range(400):
        gb(g, 90.0)
    pinned = gb(g, 90.0)
    assert pinned < 0.02, (
        f"a pitch dwelt in 400 times still pays {pinned} — being pinned must "
        "decay to ~nothing or this becomes an incentive to stare")
    # ...while a direction it has NOT looked in still pays a lot more
    fresh = gb(g, -45.0)
    assert fresh > 10 * pinned, (
        f"an unvisited direction ({fresh}) must out-pay the pinned one "
        f"({pinned}) by a wide margin or the clamp is not unstuck")
    # SWEEPING cannot be farmed: every bucket has its own decaying count
    g2 = _G()
    sweep = [gb(g2, p) for _ in range(200) for p in (-30.0, 30.0)]
    assert sum(sweep[-20:]) / 20 < 0.03, (
        "oscillating between two directions still pays — gaze novelty is "
        "farmable by sweeping, which is the spinning failure mode again")
    # a missing sense reads neutral, never fatal
    g3 = _G()
    assert gb(g3, None) == 0.0 and gb(g3, "junk") == 0.0
    assert g3._gaze_counts == {}, "a bad reading must not pollute the counts"
    # bucketing is coarse enough that jitter is the same direction
    g4 = _G()
    gb(g4, 89.0); gb(g4, 90.0)
    assert len(g4._gaze_counts) == 1, (
        f"89deg and 90deg landed in different buckets ({g4._gaze_counts}) — "
        "camera jitter would read as endless new directions")
    assert float(cfg["curiosity"]["gaze_weight"]) > 0.0, \
        "skybot does not enable gaze coverage"
    # yaw must NOT be included (spinning is the known cheap-novelty failure)
    src_gz = loop_src[loop_src.index("def _gaze_bonus"):]
    assert "yaw" not in src_gz[:src_gz.index("def _view_key")].lower(), \
        "gaze coverage reads yaw — that re-creates the spin farm"

    # ---- F. STANDING STILL MUST PAY ZERO --------------------------------
    # MEASURED live on a stationary agent: the ICM/LP base paid +0.1321/step,
    # 95% of the whole intrinsic drive, while running_std had collapsed to
    # 1.42e-05 — i.e. the view WAS mastered and the normalizer was
    # re-inflating residual jitter ~100x into an income for existing. Doing
    # nothing was the best-paid behaviour available and no shaping term
    # (gaze + centring + coverage ~ 0.03 combined) could outbid it.
    from developmental_ai.curiosity.learning_progress import (
        LearningProgressCuriosity)

    def gate(abs_frac, errs):
        """The two-gate decision, as applied per bucket in the real loop."""
        h = np.asarray(errs, dtype=np.float32)
        half = len(h) // 2
        older, recent = float(h[:half].mean()), float(h[half:].mean())
        drop, noise = older - recent, float(h.std())
        return drop if (drop > 0.5 * noise
                        and drop > abs_frac * max(recent, 1e-12)) else 0.0

    mastered = [2.64e-3, 2.61e-3, 2.66e-3, 2.63e-3,
                2.60e-3, 2.62e-3, 2.59e-3, 2.61e-3]   # flat + jitter
    learning = [8.0e-3, 7.5e-3, 7.0e-3, 6.5e-3,
                4.0e-3, 3.5e-3, 3.0e-3, 2.5e-3]       # genuinely improving
    frac = float(cfg["curiosity"]["lp_abs_frac"])
    assert frac > 0.0, "skybot does not enable the absolute progress floor"
    assert gate(0.0, mastered) > 0.0, (
        "the OLD relative-only gate should pass this jitter — if it does not, "
        "this test no longer reproduces the bug it guards")
    assert gate(frac, mastered) == 0.0, (
        f"a MASTERED bucket still pays {gate(frac, mastered)} — standing "
        "still remains profitable")
    assert abs(gate(frac, learning) - gate(0.0, learning)) < 1e-12, (
        "the absolute floor suppressed REAL learning progress — it must only "
        "remove jitter, never genuine error reduction")
    # off by default so no other config's curiosity signal changes
    assert LearningProgressCuriosity(
        obs_dim=4, action_dim=2).lp_abs_frac == 0.0, \
        "the progress floor is ON by default — that changes every experiment"

    # ---- G. ACTION-CONDITIONAL CURIOSITY: a no-op must earn NOTHING -----
    # MEASURED: the policy converged deterministically on `noop` (80% of
    # actions, entropy 0.00) while still collecting +0.2951/step. Curiosity is
    # forward-model prediction error, and on a live server the world moves
    # without the agent — day/night, mobs, other players — so ambient change
    # paid an income for merely existing. The env asks for frozen time but a
    # foreign Paper server ignores that handler (POV frames from one run show
    # both daylight and night).
    import torch as _t
    _t.manual_seed(0)
    _cc = LearningProgressCuriosity(
        obs_dim=8, action_dim=4, feature_dim=16, hidden_dim=32,
        action_conditional=True, null_action=0)
    _o, _n = _t.randn(6, 8), _t.randn(6, 8)

    def _a(i):
        v = _t.zeros(6, 4); v[:, i] = 1.0; return v

    _r_noop = _cc.compute_intrinsic_reward(_o, _a(0), _n, update_state=False)
    assert float(_r_noop.abs().max()) == 0.0, (
        f"a NO-OP still earns {float(_r_noop.abs().max())} — the whole point "
        "is that comparing the forward model against itself gives exactly 0")
    assert _cc.last_action_attribution == 0.0
    # a real action is compared against a DIFFERENT prediction, so the gate is
    # free to be positive (magnitude depends on how action-aware the forward
    # model is — see the ceiling warning below)
    _cc.compute_intrinsic_reward(_o, _a(2), _n, update_state=False)
    assert _cc.last_action_attribution >= 0.0

    # THE FAILURE MODE TO WATCH: if the forward model ignores its action
    # input, err_null == err_a for EVERY action, the gate is ~0 everywhere and
    # curiosity is suppressed entirely rather than selectively. That is not
    # detectable from a random-init unit test — it is why the segment log
    # prints "Action attribution" every segment. Pin that the readout exists.
    assert "last_action_attribution" in open(LOOP).read(), (
        "the action-attribution readout is missing — without it, a gate that "
        "suppresses ALL curiosity looks identical to one that works")
    # gates REWARD, never perception: training must be untouched
    _src_lp = open("developmental_ai/curiosity/learning_progress.py").read()
    _src_lp2 = open("developmental_ai/llm/vision_scaffold.py").read()
    assert "_gate is not None" in _src_lp and "reward = reward * _gate" in _src_lp
    assert "def train_step" not in _src_lp.split("_gate = torch.clamp")[1][:400], \
        "the gate leaked into the training path — it must scale reward only"
    # OFF by default
    assert LearningProgressCuriosity(
        obs_dim=4, action_dim=2).action_conditional is False, \
        "action-conditioning is ON by default — that changes every experiment"
    assert bool(cfg["curiosity"]["action_conditional"]) is True, \
        "skybot does not enable action-conditional curiosity"

    # ---- H. NO MORE POINT-SAMPLE READOUTS -------------------------------
    # Three separate times this session a diagnostic sampled the hot loop at
    # ONE step and was then reasoned off as if it were a segment statistic:
    #   * the reward census (sat inside a wants_step(every=50) block, ~20 of
    #     1024 steps, understating the ICM base ~10x)
    #   * action attribution (printed the final step -> read 0.000 and looked
    #     like a dead gate)
    #   * reach sense (printed the final step -> read 0.00 in a segment that
    #     contained 142 breaks)
    # A diagnostic that samples unrepresentatively is worse than none, because
    # it looks like data. Every per-step quantity must be reported as a mean.
    for _name, _acc in (("reach", "_reachval_n"),
                        ("pitch", "_pitch_n"),
                        ("action attribution", "_attr_n"),
                        ("reward census base", "_cen_n")):
        assert _acc in loop_src, (
            f"the {_name} readout has no per-step accumulator — it is a "
            "point sample again")
    # and the aggregates must actually be divided by their counts
    for _num, _den in (("_reachval_sum", "_rvn"), ("_pitch_sum", "_pn"),
                       ("_attr_sum", "_an"), ("_cen_base", "_cn2")):
        assert f"{_num}\", 0.0) / {_den}" in loop_src or \
               f"{_num}\", 0.0) / max(1, {_den})" in loop_src or \
               f"self, \"{_num}\", 0.0) / {_den}" in loop_src, \
            f"{_num} is not divided by its own step count"
    # the surviving last-step values must SAY they are last-step
    assert "Reach sense (last step)" in loop_src, (
        "a last-step reach value is still printed without saying so — that "
        "is what made 0.00 read as 'the trace is broken'")

    # ---- I. SATURATION IS A ONE-WAY DOOR --------------------------------
    # The fact that explains every failed entropy_coef attempt: at a saturated
    # softmax the entropy GRADIENT vanishes, so the bonus can prevent collapse
    # but can never reverse it.
    from developmental_ai.policy.actor_critic import StandaloneActorCritic
    _g = {}
    for _mp, _lv in ((2.0, "healthy"), (20.0, "saturated")):
        _lo = _t.zeros(1, 36, requires_grad=True)
        with _t.no_grad():
            _lo[0, 0] = _mp
        _d = _t.distributions.Categorical(logits=_lo)
        _d.entropy().sum().backward()
        _g[_lv] = float(_lo.grad.abs().max())
    assert _g["saturated"] < _g["healthy"] / 1000.0, (
        f"entropy gradient at saturation ({_g['saturated']:.2e}) is not "
        f"vanishing vs healthy ({_g['healthy']:.2e}) — the premise behind the "
        "hard logit bound would be wrong")

    # the bound: order preserved, spread capped, identity when off
    _p = StandaloneActorCritic(obs_dim=4, action_dim=6, hidden_dim=8,
                               logit_range=5.0)
    _raw = _t.tensor([[12.0, 3.0, -8.0, 0.5, -1.0, 2.0]])
    _b = _p._bound_logits(_raw)
    _spread = float(_b.max() - _b.min())
    assert _spread <= 5.0 + 1e-4, f"spread {_spread} exceeds the bound"
    assert _t.equal(_raw.argsort(), _b.argsort()), (
        "the bound reordered the logits — it must rescale, never reorder, or "
        "it would change WHICH action is preferred")
    # a policy already inside the bound is untouched
    _small = _t.tensor([[0.4, -0.3, 0.1, 0.0, -0.2, 0.2]])
    assert _t.allclose(_p._bound_logits(_small),
                       _small - _small.mean(), atol=1e-5), \
        "the bound is not a no-op on an already-modest policy"
    # OFF by default -> exact identity, so no other config changes
    _off = StandaloneActorCritic(obs_dim=4, action_dim=6, hidden_dim=8)
    assert _off.logit_range == 0.0
    assert _t.equal(_off._bound_logits(_raw), _raw), \
        "logit bounding is not an exact identity when disabled"
    assert float(cfg["policy"]["logit_range"]) > 0.0, \
        "skybot does not enable the exploration floor"
    # applied when ACTING and when TRAINING, or PPO's ratios would be wrong
    _ac = open("developmental_ai/policy/actor_critic.py").read()
    assert _ac.count("_bound_logits(") >= 4, (
        "the bound is not applied at every logit site — acting and training "
        "must match or every importance ratio is computed against a "
        "distribution the agent never sampled from")

    # ---- J. THE MAGNET MUST BE ABLE TO RANK AN ALWAYS-PRESENT CATEGORY --
    # _score was cat_lp[c] - global_lp, but cat_lp is EMA'd with the same beta
    # and the same lp as global_lp on every step c is present. For a category
    # present on EVERY step the two series are identical by construction and
    # the score is exactly 0 forever. tree_visible is that case — the head
    # reported it present even while the agent stood in unrenderable void.
    # MEASURED: 144 of 144 segments at the cold-start floor w=0.3500,
    # cold_spent[tree]=144827, curiosity-ranked branch never fired once.
    _cats = ["tree_visible", "stone_visible", "water_visible"]

    def _scored(peer):
        _v = _scaffold(contrast_vs_peers=peer, target_categories=_cats)
        _v._cat_lp = {"tree_visible": 0.20, "stone_visible": 0.02,
                      "water_visible": 0.01}
        _v._global_lp = 0.20          # tree present every step -> tracks global
        # peers are now restricted to what is CURRENTLY VISIBLE (see K below:
        # frozen absent categories would otherwise outrank the live target).
        _v._present_now = set(_cats)
        return {c: _v._score(c) for c in _cats}

    _off, _on = _scored(False), _scored(True)
    assert _off["tree_visible"] == 0.0, (
        "the old form no longer reproduces the bug — this test has stopped "
        "guarding anything")
    assert _on["tree_visible"] > 0.1, (
        f"peer-relative scoring still cannot rank an always-present category "
        f"({_on['tree_visible']})")
    # it must still RANK, not just inflate: a genuinely dull category stays 0
    assert _on["water_visible"] == 0.0, "everything scores > 0 — not a ranking"
    assert _on["tree_visible"] > _on["stone_visible"], "ordering lost"
    # OFF by default so no other config's magnet behaviour changes
    from developmental_ai.llm.vision_scaffold import VisionScaffold as _VS
    assert _VS(enabled=True).contrast_vs_peers is False, \
        "peer-relative scoring is ON by default"
    assert bool(cfg["llm"]["vision"]["contrast_vs_peers"]) is True, \
        "skybot does not enable peer-relative scoring"
    # and the inertness alarm must exist, or this stays invisible next time
    assert "MAGNET INERT" in loop_src, (
        "no alarm for a magnet pinned at its cold-start floor — finding this "
        "required grepping cold_spent by hand across 144 segments")

    # ---- K. PEERS MUST BE THINGS IT CAN SEE -----------------------------
    # ema_leak is 0.0 by design, so an ABSENT category's LP FREEZES. Mobs and
    # water glimpsed once early keep stale HIGH values forever while an
    # always-present category tracks the declining global — so comparing
    # against ALL peers ranks ghosts above the live target. Measured live:
    # tree=0.201 (== global) vs frozen peers ~0.29 -> score 0 -> the magnet
    # fell back to its cold-start floor and started paying for ROTATION.
    _vp = _scaffold(contrast_vs_peers=True,
                    target_categories=["tree_visible", "stone_visible",
                                       "water_visible", "zombie_visible"])
    _vp._cat_lp = {"tree_visible": 0.201, "stone_visible": 0.150,
                   "water_visible": 0.290, "zombie_visible": 0.326}
    _vp._global_lp = 0.2009
    _vp._present_now = set()                       # stale ghosts included
    _stale = _vp._score("tree_visible")
    _vp._present_now = {"tree_visible", "stone_visible"}   # only what is in view
    _live = _vp._score("tree_visible")
    assert _stale < 0.001, (
        "the all-peers form no longer reproduces the bug — this test has "
        "stopped guarding anything")
    assert _live > 0.02, (
        f"restricting peers to what is visible still cannot rank the target "
        f"({_live})")
    assert _vp._score("stone_visible") == 0.0, "ordering lost among peers"
    # and step_shaping must actually publish the present set each step
    assert "self._present_now = set(present)" in _src_lp2, \
        "_present_now is never updated, so _score reads a stale/empty set"

    # ---- L. PHI FROM MEASURED CONTACT, NOT FROM A CAPTION ---------------
    # object_centered / object_adjacent are VLM-taught, and llava cannot tell
    # from a still frame whether something sits under the crosshair. The head
    # agreed with the VLM 96% of the time and faithfully learned "nothing is
    # ever centred" — measured Centring Phi 0.003 and Reach 2.0% for a whole
    # run. No amount of training fixes a label that carries no information.
    _pv2 = {"tree_visible": 0.9, "object_centered": 0.0, "object_adjacent": 0.0}

    def _phi(ev, contact):
        _v = _scaffold(phi_from_evidence=ev)
        _v._target = None                      # isolate from the focus factor
        _v._reach_measured = contact
        return _v._phi_head(_pv2)

    # with the VLM saying nothing is centred, the old form is pinned flat
    assert abs(_phi(False, 1.0) - _phi(False, 0.0)) < 1e-9, (
        "the caption-based form is no longer flat under a blind head — this "
        "test has stopped reproducing the bug it guards")
    # evidence-based must move with real contact, giving an actual gradient
    _lo, _hi = _phi(True, 0.0), _phi(True, 1.0)
    assert _hi > _lo + 0.5, f"phi barely moves with contact: {_lo} -> {_hi}"
    _mid = _phi(True, 0.5)
    assert _lo < _mid < _hi, f"phi is not monotone in contact: {_lo},{_mid},{_hi}"
    assert 0.0 <= _hi <= 1.0, "phi left [0,1] — Ng telescoping needs it bounded"
    # the seek scalar must use the same source, or the two disagree per step
    assert "self._reach_measured) if self.phi_from_evidence" in _src_lp2, \
        "seek still keys on the VLM's object_centered while phi does not"
    # the loop must actually SUPPLY the measurement each step
    assert "reach_measured=float(getattr(self, \"_reach_now\", 0.0))" in loop_src, \
        "_reach_now is never passed to step_shaping — phi would read a stale 0"
    # OFF by default so no other config's magnet changes
    from developmental_ai.llm.vision_scaffold import VisionScaffold as _VS2
    assert _VS2(enabled=True).phi_from_evidence is False
    assert bool(cfg["llm"]["vision"]["phi_from_evidence"]) is True, \
        "skybot does not enable evidence-based phi"

    # ---- D. both dead diagnostics now live in the LIFELONG path ---------
    seg = loop_src[loop_src.index("def _collect_segment"):
                   loop_src.index("def _extract_and_store_facts")]
    assert "policy_entropy" in seg, (
        "_collect_segment still drops PPO entropy — the COLLAPSED detector "
        "cannot fire in the mode skybot runs")
    # 2026-08-07: the magnet block (including the turn/other split) was
    # deduplicated into _magnet_step_shaping — the FIFTH copy-drift incident
    # made the three inline copies themselves the hazard this test guards
    # against. The contract is unchanged: the split must be recorded on the
    # path _collect_segment actually runs. Accept either the inline split or
    # a call into the shared helper that contains it.
    if "_mag_turn" not in seg:
        assert "_magnet_step_shaping" in seg, (
            "_collect_segment neither records the magnet turn/other split "
            "inline nor calls the shared helper")
        helper = loop_src[loop_src.index("def _magnet_step_shaping"):
                          loop_src.index("def _grounded_skill_name")]
        assert "_mag_turn" in helper and "_mag_other" in helper, (
            "the shared magnet helper lost the turn/other split")

    print("[centering-drive-smoke] ALL PASS: logits bounded so saturation "
          "cannot latch (entropy gradient vanishes at max_prob=1, so the "
          "bonus can only prevent, never cure); a no-op earns EXACTLY zero "
          "(action-conditional gate, off by default, reward-only); "
          "a mastered view now pays ZERO "
          "(jitter gated, real learning untouched, off by default); "
          "gaze coverage pays a fresh "
          "direction and decays a pinned one to ~0 (sweep-proof, pitch-only); "
          "approach potential is continuous "
          "and strictly increasing below the old 0.5 threshold (landmarks and "
          "[0,1] bound preserved); a visible-but-off-centre goal now pays, "
          "telescopes to ~0 on look-away, and pays nothing when the goal is "
          "off-screen; the symbol drive has a centring potential that prefers "
          "unfamiliar symbols, decays with familiarity, ignores bare "
          "affordances and is applied as gamma*Phi'-Phi; policy_entropy and "
          "the magnet split are recorded in _collect_segment")


if __name__ == "__main__":
    main()
