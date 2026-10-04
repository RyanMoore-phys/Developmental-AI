"""Contract test: curiosity as EXPERIMENT SELECTION (plan Stage 10).

WHY THIS EXISTS
    CLAUDE.md §9: most historical "progress" was the agent finding a way to
    get paid for nothing — staring at the sky (96% of drive), sitting in a
    menu (77% of income). A prediction-error drive is maximal on whatever is
    least predictable, which is exactly a noisy TV, a flickering display or
    a model's own hallucination. Stage 10 replaces "how surprised am I" with
    "which experiment would resolve a question I actually have", and this
    test is the argument that it does — written to catch it being wrong.

WHAT IS CLAIMED (contracts)
    A  NOISY TV. On the cue->response fixture with a coin-flip TV action,
       I(H;O) of watching is EXACTLY 0 while every response scores > 0; the
       novelty witness (predictive entropy) ranks the TV FIRST. With an
       ensemble (posterior samples per outcome key, fitted on REAL adapter
       outcomes) the BALD score of a much-watched TV falls below 0.01 nats
       while its prediction error stays at the coin's 0.25, and an
       unresolved learnable key keeps > 0.03 nats.
    B  STANDING STILL. On the lever box, "rest" carries 0 information about
       the hidden regime (realized info over 300 real rest steps == 0), and
       a model trained on those idle observations produces no persistent
       dev-probe progress: the NET tracker's tail progress is < 10% of what
       a clipped learning-progress measure (max(0, drop)) books on the same
       losses — the clipped one grows without bound on pure noise.
    C  USEFUL PASSIVE OBSERVATION. The same "rest" IS informative when the
       question is about the display itself (dropout 0.1 vs 0.5): MI > 0,
       real updates identify the truth (> 0.95), realized info > 0.
    D  REPEATED EXPERIMENTS DECAY. Repeating one test (and passive watching
       in C) drives its value below 5% (C: 10%) of its initial value once
       the question it asks is resolved.
    E  FALSE MODEL DISAGREEMENT. Members disagreeing only on an output the
       question is not about score 0 on the declared relevant output; the
       witness with every output included scores > 0.5 nats.
    F  MODEL-GENERATED JACKPOT. One wild member: raw > 3 nats, robust
       (leave-one-out) < 0.01, flagged; spread-out disagreement is not
       flagged; enormous genuine disagreement is capped at DEFAULT_INFO_CAP.
    G  FORGETTING LOOP. 10 learn/forget cycles over 4 dev probes (each
       phase SUSTAINED for 8 evaluations) book the same raw gain as 40
       probes learned once, but the loop's net progress is <= 0, most of
       its gain is reported as discounted relearning, and every looping
       probe is flagged; the genuine discovery gets full credit and no
       flags. A change-detector reset makes relearning count again.
       (Revised 2026-10-03: the old loop alternated EVERY evaluation, which
       a noise-aware tracker cannot tell from two-point noise; see
       forgetting.py.)
    G2 STATIONARY NOISE IS NOT FORGETTING (verifier finding B2). 16 probes
       of i.i.d. N(0.5 | 0.3, 0.2) for 2000 evaluations x 3 seeds: 0 flags
       (single-crossing tracker: 16/16) and net progress within 5 of the
       telescoped first-minus-last value (was ~ -3000). A NOISY genuine
       loop (0.9 <-> 0.1 every 12 evaluations, sd 0.05) still flags 16/16.
    H  ACTION-DEPENDENT EVIDENCE. The same hit after commands 2 and 3 at cue
       0 moves the posterior to shift=2 and shift=3 respectively, and the
       update is tied to info["executed_action"].
    I  COST ACCOUNTING. Equal information, durations 1 vs 4: score
       difference is exactly w_effort * 3 and the cheaper one is chosen; a
       real run's account books ticks, forecast and realized nats apart,
       and TV picks cost ticks but earn 0 realized nats.
    J  HELD-OUT GUARD. A held-out probe is refused at add() and at measure()
       even when smuggled in behind the API.
    K  TERM METADATA. A term missing its recovery condition is refused.
    L  IMAGINED OUTCOMES. An "imagined" hit is refused and moves nothing.
    N  TARGETED INFORMATION (verifier F6/E10c). On the joint shift x 4-bit
       lamp set the lamp query is worth log 2 in I(H;O) (witness) but 0 in
       I(shift;O), its nuisance information is logged apart, and a
       response's targeted value equals the shift-only MI. Through the
       library fixture: targeted selection spends < 5% of interactions on
       the lamp (joint > 25%) and identifies >= 2 interactions sooner; on
       the original noisy-TV fixture targeted and joint choices are
       IDENTICAL scenario by scenario (so M cannot get worse).
    O  MISSPECIFICATION (verifier F6/E10d). Truth not in the set: the
       posterior still goes confident (100%), but a resolved conclusion is
       REPORTED wrong in <= 15% (was 100% confident-wrong, unflagged) and
       the set is flagged in >= 80%; well-specified: >= 95% resolved, none
       wrong, <= 5% false flags. A flagged question marks the Experiment
       conclusion unreliable, requests revision and re-opens targeted
       info; TV noise moves the check neither way; the flag clears when a
       hypothesis fits again, and restarts when the set is revised.
    M  THE STAGE 10 COMPARISON (A/B harness, preregistered below BEFORE
       running): info-gain selection vs random vs ICM-style prediction error
       (alternative) with the predictive-entropy ablation, on held-out
       scenarios with the noisy TV present, 5 seeds. Asserted: the harness
       ran and decided; info-gain is not WORSE than random. The verdict is
       reported, not forced.

Run: PYTHONPATH=. python tests/_foundation_selection_smoke.py   (< 60 s CPU)
"""
import dataclasses
import math
import sys
import tempfile
import time

sys.path.insert(0, ".")

import numpy as np

from developmental_ai.foundation.adapters import NonspatialAdapter
from developmental_ai.foundation.contracts import Action, Experiment, Sentinel
from developmental_ai.foundation.experiments import (
    DEFAULT_INFO_CAP, VERDICTS, CostAccount, CueShiftHypotheses, DevProbeSet,
    DistractedCueAdapter, ExperimentQuestion, ExperimentSelector, HeldOutAccessError,
    ImaginedEvidenceError, Preregistration, Probe, ProgressMeter, RetentionTracker,
    TermMetadataError, check_terms, default_terms, effort_of, ensemble_information,
    enumerate_candidates, hypothesis_information, make_arm, make_split,
    run_ab, run_identification, slip_probability, targeted_information,
    update_from_real_outcome)
from developmental_ai.foundation.inference import HypothesisSet
from developmental_ai.foundation.mechanisms.api import MixedOutcome, OutcomeLayout

T0 = time.time()
N = [0]


def ok(msg):
    N[0] += 1
    print(f"  {N[0]:2d}. {msg}")


def _val(obs, ch):
    for o in obs:
        if o.channel == ch:
            return o.value
    return None


def _act(env, obs, cmd):
    return Action(env.ENVIRONMENT, env.stream, obs[0].episode, obs[0].seq,
                  env.action_spec().spec_id, cmd, 1.0, time.time())


def _two(names):
    hs = HypothesisSet(floor=0.0)
    for n in names:
        hs.add_hypothesis(n)
    return hs


# --------------------------------------------------------------------- A
def test_noisy_tv():
    K = 6
    m = CueShiftHypotheses(K, slip_probability(0.35))
    hs = m.new_set()
    tv = hypothesis_information(hs, m.outcome_table(0, K))
    resp = [hypothesis_information(hs, m.outcome_table(0, k)) for k in range(K)]
    assert tv["info"] == 0.0 and min(r["info"] for r in resp) > 0.05, (tv, resp[0])
    assert tv["predictive_entropy"] > max(r["predictive_entropy"] for r in resp)
    ok(f"MI(watch_tv)=0 exactly, MI(respond)>={min(r['info'] for r in resp):.3f}; "
       f"novelty witness ranks TV first (H={tv['predictive_entropy']:.3f})")

    # Ensemble of posterior samples per key, fitted on REAL adapter outcomes.
    env = DistractedCueAdapter(K, shift=1, seed=3, max_steps=400)
    obs = env.reset(3)
    counts = {}
    err = {}
    target = None
    for t in range(330):
        cue = int(np.argmax(_val(obs, "cue")))
        cmd = K if t < 300 else (cue + 4) % K          # 300 TV, then a few responses
        obs, _, _, _ = env.step(_act(env, obs, cmd))
        if cmd == K:
            y, key = int(_val(obs, "tv")), "tv"
        else:
            y, key = int(float(_val(obs, "reward")) > 0.5), ("resp", cue, cmd)
            if target is None:
                target = key
            if key != target:
                continue
        a, b = counts.get(key, (0, 0))
        counts[key] = (a + y, b + 1 - y)
        e = err.get(key, (0.5, 0.25, 0))
        mean, ema, n = e
        err[key] = (mean + (y - mean) / (n + 1), 0.7 * ema + 0.3 * (y - mean) ** 2, n + 1)
    lay = OutcomeLayout((), ("y",), ("0", "1"))

    def members(key, M=8):
        a, b = counts[key]
        rng = np.random.default_rng(11)
        ps = rng.beta(1 + a, 1 + b, size=M)
        return [MixedOutcome.gaussian_categorical(
            lay, np.zeros((1, 0)), np.ones((1, 0)), np.array([[[1 - p, p]]]))
            for p in ps]
    b_tv = ensemble_information(members("tv"), disc_index=[0])["info"][0]
    b_key = ensemble_information(members(target), disc_index=[0])["info"][0]
    assert sum(counts["tv"]) == 300 and sum(counts[target]) <= 5
    assert b_tv < 0.01 and b_key > 0.03, (b_tv, b_key, counts[target])
    assert err["tv"][1] > 0.15, err["tv"]
    ok(f"BALD(TV after 300 real watches)={b_tv:.4f} < 0.01 while its prediction "
       f"error stays {err['tv'][1]:.2f}; unresolved key ({sum(counts[target])} obs) "
       f"BALD={b_key:.3f}")

    g = OutcomeLayout(("y",))
    noisy = [MixedOutcome.gaussian_categorical(g, np.array([[0.5]]), np.array([[0.25]]))
             for _ in range(5)]
    learn = [MixedOutcome.gaussian_categorical(g, np.array([[mu]]), np.array([[0.01]]))
             for mu in (0.1, 0.3, 0.5, 0.7, 0.9)]
    gn = ensemble_information(noisy, cont_index=[0])["info"][0]
    gl = ensemble_information(learn, cont_index=[0])["info"][0]
    assert gn < 1e-12 and gl > 0.5, (gn, gl)
    ok(f"Gaussian ensemble: pure aleatoric {gn:.1e}, disagreeing members {gl:.2f} nats")


# --------------------------------------------------------------------- B
def test_standing_still():
    env = NonspatialAdapter(seed=5, max_steps=400, dropout=0.1)
    obs = env.reset(5)
    regime = _two(["r0", "r1"])
    rest_table = np.array([[1.0 - 1e-6, 1e-6], [1.0 - 1e-6, 1e-6]])   # delta==0 under both
    lever_table = np.array([[1e-3, 1 - 1e-3], [1 - 1e-3, 1e-3]])
    assert hypothesis_information(regime, rest_table)["info"] == 0.0
    assert hypothesis_information(regime, lever_table)["info"] > 0.6
    realized = 0.0
    xs = []
    for _ in range(300):
        obs, _, _, info = env.step(_act(env, obs, 0))           # rest
        c = _val(obs, "count")
        xs.append(1 if isinstance(c, Sentinel) else 0)
        realized += update_from_real_outcome(
            regime, np.log(rest_table[:, 0]), executed_action=info["executed_action"],
            evidence=[o for o in obs if o.provenance == "sensor"])
    assert realized == 0.0 and np.allclose(regime.probs(), 0.5)
    ok("rest: MI about the hidden regime = 0; 300 REAL idle steps realized 0.000 nats")

    # Dev probes: rest observations from dev episodes of other lever runs.
    pe = NonspatialAdapter(seed=9, max_steps=10, dropout=0.1)
    ep_obs = {}
    for _ in range(40):
        o = pe.reset()
        ep = o[0].episode
        ep_obs[ep] = []
        for _ in range(5):
            o, term, trunc, _ = pe.step(_act(pe, o, 0))
            ep_obs[ep].append(1 if isinstance(_val(o, "count"), Sentinel) else 0)
    split = make_split("lever:idle", list(ep_obs), 0.25)
    probes = DevProbeSet(split)
    for e in split.dev:
        for i, x in enumerate(ep_obs[e]):
            probes.add(f"{e}#{i}", "lever:idle", e, x)
    q = [0.5]

    def loss(p: Probe):
        return -math.log(q[0] if p.payload else 1.0 - q[0])
    tracker = RetentionTracker(known_below=1e-3, forgotten_above=1e3)
    meter = ProgressMeter(probes, loss, tracker)
    prev, clipped_tail, net_tail = None, 0.0, 0.0
    for t, x in enumerate(xs):
        q[0] += 0.2 * (x - q[0])                                  # learner on idle data
        q[0] = min(max(q[0], 1e-3), 1 - 1e-3)
        r = meter.measure(t)
        if prev is not None and t >= 50:
            clipped_tail += max(0.0, prev - r["mean_loss"]) * len(probes)
            net_tail += r["progress"]
        prev = r["mean_loss"]
    assert clipped_tail > 5.0, clipped_tail
    assert abs(net_tail) < 0.1 * clipped_tail, (net_tail, clipped_tail)
    assert not tracker.flags()
    ok(f"idle learner over 250 tail evals: clipped LP books {clipped_tail:.1f} "
       f"(grows on noise), NET dev-probe progress {net_tail:+.2f} (no persistence)")


# --------------------------------------------------------------------- C, D
def test_passive_observation_and_decay():
    env = NonspatialAdapter(seed=6, max_steps=400, dropout=0.1)
    obs = env.reset(6)
    disp = _two(["dropout=0.1", "dropout=0.5"])
    table = np.array([[0.9, 0.1], [0.5, 0.5]])                  # cols: present, absent
    mi0 = hypothesis_information(disp, table)["info"]
    realized = 0.0
    for t in range(120):
        obs, _, _, info = env.step(_act(env, obs, 0))           # rest = pure watching
        x = 1 if isinstance(_val(obs, "count"), Sentinel) else 0
        realized += update_from_real_outcome(
            disp, np.log(table[:, x]), executed_action=info["executed_action"],
            evidence=[o for o in obs if o.provenance == "sensor"])
    mi1 = hypothesis_information(disp, table)["info"]
    assert mi0 > 0.05 and disp.probs()[0] > 0.95 and realized > 0.3, (mi0, disp.probs())
    assert mi1 < 0.1 * mi0, (mi0, mi1)
    ok(f"useful passive observation: MI={mi0:.3f} > 0, P(truth)={disp.probs()[0]:.3f}, "
       f"realized {realized:.2f} nats; then MI decays to {mi1:.4f} (<10%)")

    K = 6
    m = CueShiftHypotheses(K, slip_probability(0.35))
    hs = m.new_set()
    env = DistractedCueAdapter(K, shift=1, seed=4)
    obs = env.reset(4)

    def test_value():
        return hypothesis_information(hs, m.outcome_table(0, 2))["info"]  # asks "shift=2?"
    v0 = test_value()
    for _ in range(25):
        cue = int(np.argmax(_val(obs, "cue")))
        cmd = (cue + 2) % K                                      # the same experiment
        obs, _, _, info = env.step(_act(env, obs, cmd))
        y = int(float(_val(obs, "reward")) > 0.5)
        update_from_real_outcome(hs, m.logliks(cue, cmd, y),
                                 executed_action=info["executed_action"],
                                 evidence=[o for o in obs if o.channel == "reward"])
    v1 = test_value()
    assert v1 < 0.05 * v0, (v0, v1)
    ok(f"repeating one experiment: its value {v0:.3f} -> {v1:.4f} once resolved")


# --------------------------------------------------------------------- E, F
def test_false_disagreement_and_jackpots():
    lay = OutcomeLayout(("relevant", "irrelevant"))
    ms = [MixedOutcome.gaussian_categorical(lay, np.array([[0.0, d]]), np.full((1, 2), 0.01))
          for d in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    rel = ensemble_information(ms, cont_index=[0])["info"][0]
    both = ensemble_information(ms, cont_index=[0, 1])["info"][0]
    assert rel < 1e-12 and both > 0.5, (rel, both)
    ok(f"false disagreement on an irrelevant output: relevant-only {rel:.1e}, "
       f"witness with it included {both:.2f} nats")

    g = OutcomeLayout(("y",))

    def ens(means, var=0.01):
        return [MixedOutcome.gaussian_categorical(g, np.array([[x]]), np.array([[var]]))
                for x in means]
    j = ensemble_information(ens([0, 0, 0, 0, 100.0]), cont_index=[0])
    s = ensemble_information(ens([-1, -0.5, 0, 0.5, 1]), cont_index=[0])
    h = ensemble_information(ens([-1e3, -5e2, 0, 5e2, 1e3]), cont_index=[0])
    assert j["raw"][0] > 3 and j["robust"][0] < 0.01 and j["jackpot"][0], j
    assert j["info"][0] < 0.01
    assert not s["jackpot"][0] and s["info"][0] > 1.0, s
    assert h["capped"][0] and h["info"][0] == DEFAULT_INFO_CAP, h
    ok(f"jackpot: raw {j['raw'][0]:.1f} -> robust {j['robust'][0]:.1e} (flagged); "
       f"spread {s['info'][0]:.2f} unflagged; huge genuine disagreement capped at "
       f"{DEFAULT_INFO_CAP:.2f}")


# --------------------------------------------------------------------- G
LOOP_HOLD = 8          # each learn / forget phase is sustained for 8 evaluations


def _tracker(**kw):
    return RetentionTracker(known_below=0.2, forgotten_above=0.8, relearn_discount=0.25,
                            loop_flag=2, stable_evals=5, **kw)


def test_forgetting_loop():
    split = make_split("forget", [f"e{i}" for i in range(80)], 0.25)

    def run(n_probes, schedule):
        ps = DevProbeSet(split)
        for i in range(n_probes):
            ps.add(f"p{i}", "forget", split.dev[i % len(split.dev)])
        cur = {}
        tr = _tracker()
        meter = ProgressMeter(ps, lambda p: cur[p.probe_id], tr)
        for t, losses in enumerate(schedule):
            cur.clear()
            cur.update(losses)
            meter.measure(t)
        return tr
    # 10 learn phases, 9 forget phases, each held LOOP_HOLD evaluations
    loop = [{f"p{i}": (1.0 if (t // LOOP_HOLD) % 2 == 0 else 0.05) for i in range(4)}
            for t in range(20 * LOOP_HOLD)]
    tl = run(4, loop)
    disc = []                                                   # 40 probes, one each
    for t in range(41):
        disc.append({f"p{i}": (0.05 if i < t else 1.0) for i in range(40)})
    td = run(40, disc)
    close_raw = abs(tl.totals["raw_gain"] - td.totals["raw_gain"]) < 1e-9
    assert close_raw, (tl.totals, td.totals)
    assert tl.totals["progress"] <= 0.0 and tl.flags() == [f"p{i}" for i in range(4)], \
        (tl.totals, tl.flags())
    assert tl.totals["relearning_progress"] > 0.5 * tl.totals["raw_gain"], tl.totals
    assert abs(td.totals["progress"] - td.totals["raw_gain"]) < 1e-9 and not td.flags()
    ok(f"same raw gain {tl.totals['raw_gain']:.1f}: sustained forget/relearn loop nets "
       f"{tl.totals['progress']:+.2f} (relearning {tl.totals['relearning_progress']:.1f} "
       f"discounted, 4/4 flagged); genuine discovery nets {td.totals['progress']:.1f}")
    tl.reset_probe("p0", "PredictiveCUSUM: rule change")
    t0 = 1000
    for k in range(LOOP_HOLD):                                  # everything forgotten
        tl.observe(t0 + k, {f"p{i}": 1.0 for i in range(4)})
    h = tl.observe(t0 + LOOP_HOLD, {f"p{i}": 0.05 for i in range(4)})   # relearned
    assert abs(h["progress"] - 0.95) < 1e-3 and abs(h["relearning_progress"] - 2.85) < 1e-3, h
    assert tl.state["p1"] == "forgotten" and tl.learned["p0"] == 0
    ok("after a change-detector reset, relearning p0 earns full credit "
       f"({h['progress']:.3f}); p1-p3 relearning ({h['relearning_progress']:.2f}) "
       "stays discounted")


def test_stationary_noise_is_not_forgetting():
    """Verifier finding B2 (stage9_11/RESULTS.md): i.i.d. N(0.5, 0.2) probe
    losses straddling the (0.45, 0.55) band netted ~ -3000 'progress' and
    flagged 16/16 probes as forgetting loops on the single-crossing tracker.
    Falsification: a stationary model must net ~ its telescoped first-minus-
    last change and flag nothing; a NOISY genuine loop must still flag."""
    flagged = probe_runs = 0
    worst = 0.0
    for center in (0.5, 0.3):
        for seed in range(3):
            rng = np.random.default_rng(seed)
            tr = RetentionTracker(known_below=0.45, forgotten_above=0.55)
            first = last = None
            for t in range(2000):
                losses = {f"p{i}": float(center + 0.2 * rng.normal()) for i in range(16)}
                tr.observe(t, losses)
                first = first or losses
                last = losses
            tele = sum(first[k] - last[k] for k in first)
            worst = max(worst, abs(tr.totals["progress"] - tele))
            flagged += len(tr.flags())
            probe_runs += 16
    assert flagged == 0, f"stationary noise flagged {flagged}/{probe_runs} probes"
    assert worst < 5.0, f"net progress strays {worst:.1f} from the telescoped value"
    ok(f"stationary noise N(0.5|0.3, 0.2), 16 probes x 2000 evals x 3 seeds: flagged "
       f"{flagged}/{probe_runs} (single-crossing tracker: 16/16 at 0.5), |net - "
       f"telescoped| <= {worst:.2f} (was ~3000)")
    rng = np.random.default_rng(7)
    tr = RetentionTracker(known_below=0.45, forgotten_above=0.55)
    for t in range(240):
        lvl = 0.1 if (t // 12) % 2 else 0.9
        tr.observe(t, {f"p{i}": float(lvl + 0.05 * rng.normal()) for i in range(16)})
    assert len(tr.flags()) == 16 and tr.totals["progress"] <= 0.0 and \
        tr.totals["relearning_progress"] > 0, (tr.flags(), tr.totals)
    ok(f"a NOISY genuine loop (0.9 <-> 0.1 every 12 evals, sd 0.05) still flags "
       f"16/16, nets {tr.totals['progress']:+.2f}, relearning "
       f"{tr.totals['relearning_progress']:.1f} discounted")


# --------------------------------------------------------------------- H, I, L
def test_action_dependence_costs_imagination():
    K = 6
    m = CueShiftHypotheses(K, slip_probability(0.35))
    a, b = m.new_set(), m.new_set()
    a.update(m.logliks(0, 2, 1))
    b.update(m.logliks(0, 3, 1))
    assert a.map() == "shift=2" and b.map() == "shift=3"
    env = DistractedCueAdapter(K, shift=1, seed=8)
    obs = env.reset(8)
    cue = int(np.argmax(_val(obs, "cue")))
    hs = m.new_set()
    intended = (cue + 1) % K
    obs, _, _, info = env.step(_act(env, obs, intended))
    ex = info["executed_action"]
    y = int(float(_val(obs, "reward")) > 0.5)
    update_from_real_outcome(hs, m.logliks(cue, ex.command, y), executed_action=ex,
                             evidence=[o for o in obs if o.channel == "reward"])
    assert ex.command == intended and ex.t_complete is not None
    ok("same hit after cmd 2 vs 3 -> MAP shift=2 vs shift=3; real update keyed on "
       "info['executed_action']")

    spec = env.action_spec()
    cs = enumerate_candidates(spec, durations=(1, 4), max_candidates=16, max_horizon=4)
    pair = [c for c in cs if c.command == 0]
    sel = ExperimentSelector(default_terms(effort=0.01), epsilon=0.0)
    vals = {c.candidate_id: {"info_gain": 0.4, "usefulness": 0.0, "task": 0.0,
                             "effort": effort_of(c), "risk": 0.0} for c in pair}
    s = sel.select(pair, vals)
    tots = sorted(v["total"] for v in s.table.values())
    assert s.chosen.duration_ticks == 1 and abs((tots[1] - tots[0]) - 0.03) < 1e-12
    r = run_identification("random", shift=2, seed=1, budget=40, threshold=0.999)
    assert r["ticks"] == r["interactions"] and r["forecast_nats"] >= 0
    tv = DistractedCueAdapter(K, shift=1, seed=2)
    o = tv.reset(2)
    o, _, _, inf = tv.step(_act(tv, o, K))
    hs2 = m.new_set()
    acct = CostAccount()
    kl = update_from_real_outcome(hs2, m.logliks(0, K, int(_val(o, "tv"))),
                                  executed_action=inf["executed_action"],
                                  evidence=[x for x in o if x.channel == "tv"])
    acct.charge(s, 1, kl)
    assert kl == 0.0 and acct.ticks == 1 and acct.realized_nats == 0.0
    ok(f"cost: equal info, 1 vs 4 ticks -> score gap {tots[1] - tots[0]:.2f} = "
       f"0.01x3, cheaper chosen; TV pick costs 1 tick, earns {kl:.1f} nats")

    imagined = [x.replace(provenance="imagined") for x in o if x.channel == "tv"]
    p_before = hs2.probs().copy()
    try:
        update_from_real_outcome(hs2, m.logliks(0, 1, 1),
                                 executed_action=inf["executed_action"], evidence=imagined)
        raise AssertionError("imagined evidence accepted")
    except ImaginedEvidenceError:
        pass
    assert np.array_equal(p_before, hs2.probs())
    ok("an imagined outcome is refused (ImaginedEvidenceError) and moves nothing")


# --------------------------------------------------------------------- J, K
def test_guards():
    split = make_split("guard", [f"e{i}" for i in range(30)], 0.25)
    ps = DevProbeSet(split)
    ps.add("d", "guard", split.dev[0])
    try:
        ps.add("h", "guard", split.heldout[0])
        raise AssertionError("held-out probe accepted")
    except HeldOutAccessError:
        pass
    ps._probes["smuggled"] = Probe("smuggled", "guard", split.heldout[0])
    meter = ProgressMeter(ps, lambda p: 1.0, RetentionTracker(0.1, 0.9))
    try:
        meter.measure(0)
        raise AssertionError("smuggled held-out probe measured")
    except HeldOutAccessError:
        pass
    ok("held-out episode refused at add() and at measure() (smuggled past the API)")
    bad = tuple(dataclasses.replace(t, recovery="") if t.name == "info_gain" else t
                for t in default_terms())
    try:
        check_terms(bad)
        raise AssertionError("term without recovery accepted")
    except TermMetadataError as e:
        assert "recovery" in str(e)
    ok("a term with no reachable recovery condition is refused (TermMetadataError)")


# --------------------------------------------------------------------- N
def _misspec_map(K, shift, rng):
    """The verifier's E10d world: half the cues follow `shift`, the rest are
    re-targeted, and the map is guaranteed not to be any shift."""
    resp = [(c + shift) % K for c in range(K)]
    half = sorted(rng.choice(K, K // 2, replace=False).tolist())
    for c in half:
        resp[c] = (resp[c] + 1 + int(rng.integers(K - 1))) % K
    for s_ in range(K):
        if all(resp[c] == (c + s_) % K for c in range(K)):
            resp[half[0]] = (resp[half[0]] + 1) % K
    return resp


def test_targeted_information():
    """Verifier finding F6/E10c: with a deterministic 4-bit lamp the task
    does not need, I(H;O) over the JOINT (shift x lamp) set pays log 2 per
    lamp query and joint info-gain spent 44% of interactions on the lamp.
    Targeted information I(shift; O) must value the lamp at 0, waste < 5%,
    identify faster — and change NOTHING on the original noisy-TV fixture."""
    K = 6
    m = CueShiftHypotheses(K, slip_probability(0.35), lamp_bits=4)
    hs = m.new_set(floor=1e-4)
    labels = [m.answer(n) for n in hs.names]
    lamp = targeted_information(hs, m.outcome_table(0, K + 1), labels)
    resp = targeted_information(hs, m.outcome_table(0, 2), labels)
    tv = targeted_information(hs, m.outcome_table(0, K), labels)
    assert lamp["joint"] > 0.6 > resp["joint"], (lamp, resp)        # the bug, kept as witness
    assert lamp["info"] == 0.0 and abs(lamp["nuisance"] - math.log(2)) < 1e-2
    assert tv["info"] == 0.0 and tv["nuisance"] == 0.0
    plain = CueShiftHypotheses(K, slip_probability(0.35)).new_set()
    ref = hypothesis_information(plain, CueShiftHypotheses(K, slip_probability(0.35))
                                 .outcome_table(0, 2))["info"]
    assert abs(resp["info"] - ref) < 1e-3 and resp["info"] > 0.05, (resp, ref)
    ok(f"joint I(H;O): lamp {lamp['joint']:.3f} > response {resp['joint']:.3f} (witness); "
       f"targeted: lamp {lamp['info']:.1f} (nuisance {lamp['nuisance']:.3f} logged apart), "
       f"response {resp['info']:.3f} = shift-only MI")
    res = {a: [run_identification(a, i % K, 500 + i, lamp_bits=4, lamp=(7 * i) % 16,
                                  floor=1e-4) for i in range(16)]
           for a in ("info_gain", "info_joint")}
    lf = {a: float(np.mean([r["lamp_fraction"] for r in v])) for a, v in res.items()}
    it = {a: float(np.mean([r["interactions_to_identify"] for r in v])) for a, v in res.items()}
    assert lf["info_gain"] < 0.05 and lf["info_joint"] > 0.25, lf
    assert it["info_gain"] <= it["info_joint"] - 2.0, it
    ok(f"lamp world (16 scenarios): targeted spends {lf['info_gain']:.1%} on the lamp, "
       f"identifies in {it['info_gain']:.1f}; joint {lf['info_joint']:.1%}, "
       f"{it['info_joint']:.1f}")
    same = [(run_identification("info_gain", i % K, 900 + i)["interactions_to_identify"],
             run_identification("info_joint", i % K, 900 + i)["interactions_to_identify"])
            for i in range(12)]
    assert all(a == b for a, b in same), same
    ok("original noisy-TV fixture: targeted and joint make identical choices on 12/12 "
       "scenarios (Q = H there), so the Stage 10 result cannot get worse")


# --------------------------------------------------------------------- O
def test_misspecification():
    """Verifier finding F6/E10d: with the truth NOT in the hypothesis set,
    selection reached > 0.9 posterior on a wrong shift in 100% of scenarios
    and nothing flagged it. The question must (i) flag the set, (ii) not
    report a resolved conclusion, (iii) re-open probing, (iv) not flag a
    well-specified set, and (v) let the flag clear (§4.1)."""
    K, N = 6, 20
    mis = [run_identification("info_gain", i % K, 3000 + i, stop="report",
                              response_map=_misspec_map(K, i % K, np.random.default_rng(i)))
           for i in range(N)]
    well = [run_identification("info_gain", i % K, 2000 + i, stop="report")
            for i in range(N)]
    conf = np.mean([r["confident"] for r in mis])
    wrong = np.mean([r["reported_wrong"] for r in mis])
    flag = np.mean([r["misspec_flagged"] for r in mis])
    assert conf == 1.0, conf                     # the posterior still goes confident...
    assert wrong <= 0.15 and flag >= 0.8, (wrong, flag)   # ...but it is not REPORTED
    rep = np.mean([r["reported"] for r in well])
    ff = np.mean([r["misspec_flagged"] for r in well])
    assert rep >= 0.95 and not any(r["reported_wrong"] for r in well) and ff <= 0.05, (rep, ff)
    ok(f"misspecified set ({N} scenarios): confident {conf:.0%} as before, but reported "
       f"resolved-wrong {wrong:.0%} (was 100% confident-wrong, unflagged), flagged "
       f"{flag:.0%}; well-specified: resolved {rep:.0%}, wrong 0, false flag {ff:.0%}, "
       f"resolved at {np.mean([r['interactions_to_resolved'] for r in well]):.1f} "
       f"interactions")

    # (iii) + conclusion: a flagged question re-opens info and marks the record
    hs = HypothesisSet(floor=1e-3)
    for n in ("a", "b"):
        hs.add_hypothesis(n)
    answers = {"a": "A", "b": "B"}
    q = ExperimentQuestion(hs, answers, experiment=Experiment(
        "exp1", "which?", ("A", "B"), {"cmd": 0}, {}, {"ticks": 100}))
    sharp = np.array([[0.95, 0.05], [0.05, 0.95]])     # a predicts 0, b predicts 1
    env = DistractedCueAdapter(K, shift=1, seed=3, max_steps=1000)
    obs = env.reset(3)

    def real(outcome):
        nonlocal obs
        obs, _, _, info = env.step(_act(env, obs, K))
        q.observe(sharp if outcome is not None else np.full((2, 2), 0.5),
                  outcome if outcome is not None else 0,
                  executed_action=info["executed_action"],
                  evidence=[x for x in obs if x.channel == "tv"])
    for _ in range(12):
        real(0)
    assert q.status()["state"] == "resolved"
    for k in range(30):              # a fits 2 of 3, b 1 of 3: neither is the truth
        real(int(k % 3 == 2))
    st = q.status()
    raw_info = hypothesis_information(hs, sharp)["info"]
    c = q.conclusion().payload["conclusion"]
    assert st["state"] == "misspecified" and st["request_revision"] and not c["reliable"]
    assert c["answer"] is None and "misspecified" in c["why_unreliable"]
    assert st["p_leading"] > 0.9 and q.information(sharp)["info"] > 5 * max(raw_info, 1e-3), \
        (st, q.information(sharp), raw_info)
    for _ in range(200):                                 # TV noise: no evidence either way
        real(None)
    assert q.misspecified(), "noise must neither confirm nor refute"
    for _ in range(60):                                  # the world fits `a` again
        real(0)
    assert not q.misspecified() and q.status()["state"] == "resolved"
    ok("flagged: state misspecified, revision requested, Experiment conclusion marked "
       "unreliable, info re-opened; TV noise moves nothing; flag clears once a "
       "hypothesis fits again (no latch)")
    for _ in range(10):
        q.monitor.observe(["a", "b"], np.array([[0.05, 0.95], [0.95, 0.05]]), 0)
    assert q.misspecified()
    answers["c"] = "A"
    hs.add_hypothesis("c")
    assert q.misspecified() is False and q.monitor.flagged() is False
    ok("a structural revision (new hypothesis) restarts the model check on the new set")


# --------------------------------------------------------------------- M
def test_stage10_ab():
    split = make_split("stage10:cuetv", [f"scenario{i}" for i in range(40)], 0.25)
    # PREREGISTRATION — written and frozen before any arm runs.
    prereg = Preregistration(
        name="stage10-infogain-vs-random-noisy-tv",
        hypothesis=("selecting the experiment with the highest expected information "
                    "about the hidden shift (I(H;O)) identifies the mechanism in "
                    "fewer real interactions than uniform-random experiments, with a "
                    "coin-flip TV action available to both"),
        primary_metric="interactions_to_identify", direction="lower", delta=1.0,
        baseline_arm="random", candidate_arm="info_gain",
        ablation_arm="entropy", alternative_arm="icm",
        seeds=(0, 1, 2, 3, 4), basis="final",
        resource_budget={"max_wall_seconds": 20.0},
        failure_conditions=(
            "any arm raises or exceeds 20 s per (seed, split)",
            "metric = mean over held-out scenarios of interactions until posterior "
            "mass on the true shift first exceeds 0.9; never reached -> budget+1=61 "
            "(censored, counted, not dropped)",
            "ICM/LP/entropy are reported; only info_gain vs random is decisive"),
        split_fingerprints=(split.fingerprint,),
        notes=("fixture: DistractedCueAdapter K=6, reward_noise=0.35 (slip ~0.077), "
               "epsilon=0.05, budget=60; scenarios = held-out episode ids; all arms "
               "share one Bayesian learner and differ only in selection")).freeze()
    arms = {a: make_arm(a, budget=60) for a in ("info_gain", "random", "icm", "entropy", "lp")}
    with tempfile.TemporaryDirectory() as tmp:
        rep = run_ab(prereg, arms, splits=[split], measure_memory=False, out_dir=tmp)
        assert rep.json_path and rep.md_path
    v = rep.verdict
    assert v["verdict"] in VERDICTS
    assert all(r.status == "ok" for r in rep.results.runs), \
        [(r.arm, r.status, r.error) for r in rep.results.runs if r.status != "ok"]
    mean = {a: float(np.mean([r.primary for r in rep.results.runs if r.arm == a]))
            for a in arms}
    tvf = {a: float(np.mean([r.metrics["tv_fraction"]
                             for r in rep.results.runs if r.arm == a])) for a in arms}
    npi = {a: float(np.mean([r.metrics["realized_nats_per_interaction"]
                             for r in rep.results.runs if r.arm == a])) for a in arms}
    assert mean["info_gain"] <= mean["random"], mean
    c = v["comparisons"]["final"]
    ok(f"A/B ran ({len(rep.results.runs)} runs, {len(split.heldout)} held-out "
       f"scenarios x 5 seeds): VERDICT {v['label']} — mean diff "
       f"{c['mean_diff']:.2f} CI [{c['ci_low']:.2f}, {c['ci_high']:.2f}]")
    for a in arms:
        print(f"      {a:9s} interactions={mean[a]:5.2f}  tv_fraction={tvf[a]:.2f}  "
              f"realized nats/interaction={npi[a]:.3f}")
    for role, comp in v["secondary"].items():
        print(f"      secondary {role}: mean diff {comp['mean_diff']:.2f} "
              f"CI [{comp['ci_low']:.2f}, {comp['ci_high']:.2f}]")
    ok("info_gain is not worse than random (asserted); full acceptance reported above")


if __name__ == "__main__":
    tests = [test_noisy_tv, test_standing_still, test_passive_observation_and_decay,
             test_false_disagreement_and_jackpots, test_forgetting_loop,
             test_stationary_noise_is_not_forgetting,
             test_action_dependence_costs_imagination, test_guards,
             test_targeted_information, test_misspecification, test_stage10_ab]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  FAIL {fn.__name__}: {e!r}")
    print(f"[foundation_selection_smoke] {time.time() - T0:.1f}s")
    if failed:
        print(f"[foundation_selection_smoke] {failed} FAILED")
        sys.exit(1)
    print("[foundation_selection_smoke] ALL PASS")
