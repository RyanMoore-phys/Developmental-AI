"""Stage 11 contracts: planning, skills and consolidation judged on REAL outcomes.

WHAT IS CLAIMED (plan Stage 11; foundation/planning/)

    The completion gate is "improved real behavior with documented costs and
    retained capabilities. Planning success in imagination alone does not
    qualify." Every number below is measured in the point-mass ADAPTER
    (records with sensor provenance), never in the model. Each contract is
    written so a permissive implementation fails it.

    A. CLOSED LOOP. With a model fitted on 150 random real transitions, the
       planner beats a random controller on real late-episode distance to
       the goal, and is the decision source for most steps (the gate opened
       through real validation, not by default).
    B. MODEL EXPLOITATION IS CAUGHT (falsification). A wrong-but-confident
       model (gain sign flipped, variance 1e-6) with the reliability gate
       DISABLED drives the mass away — worse than random — while its own
       imagined return looks fine: the exploitation witness. With the gate,
       the same model never controls a step after its first real
       validations; real outcome equals the fallback's.
    C. DEADLINE. A model too slow for the deadline: the first call is
       ABANDONED at the deadline (the model call runs on the planner's
       worker thread; verifier finding B3 — it used to overrun by one whole
       synchronous chunk) and discarded; every later call is pre-empted
       and returns the existing PPO actor-critic's action
       (ActorCriticFallback over a random-init StandaloneActorCritic) within
       the deadline, logged "fallback_timeout". When the model becomes fast
       again (and the abandoned call has returned) the reprobe escape
       returns control to the planner.
    D. ACTION REMAPPING (transfer.ControlRemap, a 90-degree rotation of the
       five pushes). The validated planner is in control; within 3 real
       steps of the remap the gate closes, the CUSUM raises a dynamics
       alarm and the consolidated mechanism goes needs_revalidation. A refit
       on post-change real transitions (version + 1) passes validation,
       revalidates the mechanism and returns control to the planner, whose
       real performance recovers. Knowledge is re-earned, not latched shut.
    E. OPTION DURATION AND CREDIT. SMDP-Q with gamma^tau prefers a 1-step
       reward of 0.6 over a 10-step option paying 1.0 (correct: 0.6 >
       0.9^9); the naive one-step discount prefers the option (witness).
    F. SKILL DISCRIMINATION. Skills are random-init actor-critics. A byte
       copy under another name, and a copy with 1e-7 weight noise (different
       weight hash), are flagged DUPLICATES; a same-NAME different-seed
       policy and a head-permuted copy (same weight multiset) are NOT. The
       verdict is on a probe set built from every skill's own evidenced
       states (build_probe_set) and says what it covers; the same pairs on
       the bare shared probes are only "indistinguishable_on_probes" (F8).
    G. COMPETENCE CALIBRATION. 1500 skills with true p ~ U(0.05, 0.95) and
       3-40 real trials: Beta posterior predictive ECE < 0.05 and 90%
       interval coverage within [0.86, 0.94]. Imagined successes raise and
       change nothing.
    H. LEARNED INITIATION/TERMINATION from real executions of a reach skill
       (PD controller to the origin, 12-step budget): the quadratic
       initiation set predicts held-out success better than the base rate
       (accuracy >= 0.85) and the termination set fires near the subgoal and
       not far from it.
    I. RETENTION. Competence on retention probes is re-measured after an
       interference (gains overwritten); significant forgetting is flagged
       for the damaged skill and NOT for an untouched control skill.
    J. SAVE / RESUME. Skills, mechanism records, knowledge statuses,
       retention, planner + reliability state survive a JSON round trip with
       identity and version intact; the resumed controller makes the SAME
       next decisions as the original (rng and warm start restored).
    K. STALE MODEL RECOVERY after a representation MAJOR bump: the gated
       controller falls back ("knowledge: needs_revalidation"), a store
       loaded after the bump also comes back needs_revalidation, and
       revalidation on new real data returns control to the planner.
    L. SKILL-BANK BRIDGE reads rows written from the REAL skill_bank.Skill
       dataclass, read-only (directory byte-identical), name is not identity.

    Verifier findings (experiments/foundation_ab/stage9_11/RESULTS.md; the
    post-fix numbers are in POSTFIX_planning.md there):
    M. DEADLINE UNDER HEAVY-TAILED LATENCY (B3). 2% of model calls sleep
       30 ms, deadline 10 ms, 2 x 150 decisions: < 2% of decide() calls
       exceed 1.25x the deadline (the verifier measured 8.5% over 2x before
       the fix). Witness: the SAME fixture with synchronous calls
       (isolate_calls=False, the old behaviour) exceeds 1.25x on >= 3%.
    N. REGION-AWARE RELIABILITY (F7). Point mass with a MUD disc the
       model is confidently wrong about (it claims a boost), wrong nowhere
       else. With the plain global gate (regions=False) the planner keeps
       re-entering the mud each time the window refills (witness, >= 3
       entries); the region gate at least halves that. Escape (CLAUDE.md
       §4.1): after a refit to the true dynamics (version bump) the planner
       controls steps inside the mud again and the region is no longer
       marked — knowledge is re-earned, not latched shut.
    O. DUPLICATES NEED COVERAGE (F8). Skill pairs that differ only where
       some |s_i| > 7: on the bare N(0, 2^2) probes they are
       "indistinguishable_on_probes" (witness of the probe gap); on a set built
       from each skill's own evidence (deployment states N(0, 4^2)) no pair
       is a duplicate. A byte copy under another name stays a duplicate
       and a same-name different policy stays distinct.
    Q. HUNG MODEL IS NOT A LATCH (known limit closed 2026-10-03; CLAUDE.md
       §4.1). Q1, DEFAULT watchdog (hang_after_s = max(1 s, 20 x deadline)):
       the first model call never returns on its own. Decisions go to the
       fallback while it hangs; after ~1 s the worker is poisoned, a fresh
       one re-probes, and the PLANNER controls again within 2 s — while
       the wait `observe` does before touching the model (it waited on the
       in-flight call with NO timeout, blocking the caller for as long as
       the call hung) is bounded by hang_after_s.
       BEFORE THE FIX: the planner never regained control and observe
       blocked until the call was released externally (here at 6 s).
       Q2, a model on which EVERY call hangs (hang_after 50 ms,
       max_respawns 2): over 1 s of decisions the fallback has every step,
       respawns stop at 2, and model-call threads never exceed 1 + 2; once
       calls return again the planner is back.
    R. GRADED SKILL VERDICT (verifier E11e, closed 2026-10-03). A copy whose
       logits are x3 (same argmax on every state: the same deterministic
       behaviour) was reported DISTINCT (JS 0.10 bits, never flagged). On
       evidence-built probe sets (the E11 policies, deployment states
       N(0, 4^2)), 20 trials: tempered copy -> near_duplicate >= 19/20
       (modal action agrees on every probe both skills resolve with 64
       executions); byte copy under another name -> duplicate 20/20; same
       name, different policy -> distinct 20/20; off-probe pair -> never
       duplicate or near_duplicate. Bare probes never yield either positive
       verdict (indistinguishable_on_probes). Reported, not asserted: weight
       drift 0.1 (8.8% argmax disagreement on deployment states, all at
       near-ties) mostly grades near_duplicate — its differences are not
       resolvable by 64 executions.

    TIMING. The wall-clock contracts (C, M and Q) go through _timed: the whole
    measurement is retried up to 3 times and passes if ANY attempt holds
    the bound (a scheduler stall on a loaded runner is not the code's
    fault; each was checked to miss on ALL attempts against the bug it
    guards). The attempt used is printed.

Run: PYTHONPATH=. python tests/_foundation_planning_smoke.py   (< 60 s)
"""
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, ".")

import numpy as np

try:                                   # actor_critic imports gymnasium at module
    import gymnasium                   # level but never uses it; CI has it. On a
except ImportError:                    # dev box without it, stub the name only.
    sys.modules["gymnasium"] = types.ModuleType("gymnasium")

from developmental_ai.foundation.experience.errors import ProvenanceError
from developmental_ai.foundation.perception.versioning import RepresentationRegistry
from developmental_ai.foundation.planning import (
    ActorCriticFallback, BetaCompetence, BoundedPlanner, EpisodeMemory, KnowledgeStore,
    LearnedSkill, PlanningController, ReliabilityMonitor, RetentionMonitor,
    behaviour_signature, build_probe_set, calibration, find_duplicates, interval_coverage,
    load_planning_state, read_skill_bank, save_planning_state, smdp_target)
from developmental_ai.foundation.planning.fixtures import (
    DT, GoalReward, PointMassAdapter, PointMassMechanism, ProportionalController,
    encode_discrete, make_action, policy_vector, run_episode, state_from_obs, transitions)
from developmental_ai.foundation.transfer import ControlRemap

EVAL_SEEDS = list(range(500, 508))


def _timed(measure, ok, what, attempts=3):
    """Wall-clock contracts claim a bound the CODE holds; a scheduler stall
    on a loaded runner is not the code's fault. Re-run the whole measurement
    up to `attempts` times and pass if ANY attempt meets the bound (a code
    bug misses it every time); fail with every attempt's numbers. Returns
    (result, attempts used)."""
    seen = []
    for k in range(1, attempts + 1):
        r = measure()
        if ok(r):
            return r, k
        seen.append(r)
    raise AssertionError(f"{what}: bound missed on all {attempts} attempts: {seen}")


def collect_random(env, n_eps, seed):
    rng = np.random.default_rng(seed)
    rows = []

    def on(obs, cmd, nxt):
        s, _, p = state_from_obs(obs)
        assert p == "sensor"
        rows.append((s, encode_discrete(np.array([[cmd]]))[0], DT, state_from_obs(nxt)[0]))
    for e in range(n_eps):
        run_episode(env, lambda o: int(rng.integers(5)), seed=seed + e, on_step=on)
    return rows


def make_controller(model, env, gated=True, fallback=None, **kw):
    gr = GoalReward()
    args = dict(n_entities=1, action_dim=2, encode=encode_discrete, horizon=8,
                n_candidates=64, n_iters=3, deadline_s=1.0, seed=1)
    args.update(kw)
    pl = BoundedPlanner(model, gr, env.action_spec(), **args)
    rel = ReliabilityMonitor(window=10, min_count=3, max_z2=9.0) if gated else \
        ReliabilityMonitor(window=1, min_count=1, max_z2=1e300)   # gate disabled
    if fallback is None:
        pc = ProportionalController(seed=3)
        fallback = lambda o: pc(policy_vector(o))
    return PlanningController(pl, fallback, rel), gr


def run_controlled(ctl, gr, env, seeds, on_extra=None, gaps=None):
    """Real late distance; optionally the IMAGINATION GAP per planner
    decision: imagined discounted return minus the discounted return of the
    real distances that followed (same horizon, same gamma)."""
    out, imagined = [], []
    H, gam = ctl.planner.H, ctl.planner.gamma
    for sd in seeds:
        imag_t, real_d = [], []

        def choose(obs):
            s, g, _ = state_from_obs(obs)
            gr.goal = g
            d = ctl.decide(s, DT, obs)
            imag_t.append(d.plan.imagined_value if d.source == "planner" else None)
            if d.source == "planner":
                imagined.append(d.plan.imagined_value)
            return d.command

        def on(obs, cmd, nxt):
            s, _, p = state_from_obs(obs)
            info = ctl.observe(s, cmd, DT, state_from_obs(nxt)[0], provenance=p)
            s2, g2, _ = state_from_obs(nxt)
            real_d.append(float(np.linalg.norm(s2.pos[0, 0] - g2)))
            if on_extra is not None:
                on_extra(obs, cmd, nxt, info)
        out.append(run_episode(env, choose, seed=sd, on_step=on)["late_dist"])
        if gaps is not None:
            disc = gam ** np.arange(H)
            for t, v in enumerate(imag_t):
                if v is not None and t + H <= len(real_d):
                    gaps.append((v - float(disc @ -np.asarray(real_d[t:t + H]))) / disc.sum())
    return float(np.mean(out)), (float(np.mean(imagined)) if imagined else None)


def baseline(env, policy, seeds):
    return float(np.mean([run_episode(env, policy, seed=s)["late_dist"] for s in seeds]))


# ---------------------------------------------------------------------------
def test_closed_loop_and_exploitation(model):
    env = PointMassAdapter("discrete", seed=0)
    rng = np.random.default_rng(9)
    rnd = baseline(env, lambda o: int(rng.integers(5)), EVAL_SEEDS)
    pc = ProportionalController(seed=3)
    fb = baseline(env, lambda o: pc(policy_vector(o)), EVAL_SEEDS)
    ctl, gr = make_controller(model, env)
    gap_ok = []
    plan, _ = run_controlled(ctl, gr, env, EVAL_SEEDS, gaps=gap_ok)
    share = ctl.counts["planner"] / sum(ctl.counts.values())
    assert plan < 0.5 * rnd, f"planner {plan:.3f} vs random {rnd:.3f}"
    assert share > 0.9, f"planner controlled only {share:.2f} of steps"
    print(f"  A. real late distance: planner {plan:.3f}, fallback {fb:.3f}, random "
          f"{rnd:.3f}; planner controlled {share:.0%} of steps; imagination gap "
          f"{np.mean(gap_ok):+.3f} m/step")

    wrong = PointMassMechanism("pm.wrong")
    wrong.set_params(model.damping, -model.gain, [1e-6] * 2, [1e-8] * 2)
    c_open, g_open = make_controller(wrong, env, gated=False)
    gap_bad = []
    exploit, imag = run_controlled(c_open, g_open, env, EVAL_SEEDS, gaps=gap_bad)
    c_gate, g_gate = make_controller(wrong, env, gated=True)
    gated, _ = run_controlled(c_gate, g_gate, env, EVAL_SEEDS)
    assert exploit > rnd, f"wrong model should be worse than random: {exploit} vs {rnd}"
    # the exploitation witness: imagination promises far more than reality pays
    assert np.mean(gap_bad) > 0.3 and abs(np.mean(gap_ok)) < 0.05, (np.mean(gap_bad),
                                                                    np.mean(gap_ok))
    assert c_gate.counts["planner"] == 0, c_gate.counts
    assert gated < 0.5 * exploit and abs(gated - fb) < 1e-9, (gated, fb, exploit)
    print(f"  B. wrong-but-confident model, gate disabled: real {exploit:.3f} (> random); "
          f"imagination gap {np.mean(gap_bad):+.2f} m/step (it believes it is "
          f"winning). Gated: real {gated:.3f} == "
          f"fallback, planner steps {c_gate.counts['planner']}")


def test_deadline():
    import torch
    from developmental_ai.policy.actor_critic import StandaloneActorCritic
    torch.manual_seed(0)
    ac = StandaloneActorCritic(obs_dim=4, action_dim=5, hidden_dim=32)
    acf = ActorCriticFallback(ac, deterministic=True)

    class Slow(PointMassMechanism):
        delay = 0.004

        def _predict(self, state, action, dt):
            time.sleep(self.delay)
            return super()._predict(state, action, dt)
    m = Slow()
    m.set_params([0.8, 0.8], 4 * np.eye(2), [1e-4] * 2, [1e-6] * 2)
    env = PointMassAdapter("discrete", seed=0)
    deadline = 0.010
    obs = env.reset(seed=1)
    s, g, _ = state_from_obs(obs)
    prev = []

    def measure():
        if prev:                            # the last attempt's abandoned call holds m
            prev[-1].planner.wait_idle(1.0)
        ctl, gr = make_controller(m, env, fallback=lambda o: acf(policy_vector(o)),
                                  horizon=4, n_iters=2, deadline_s=deadline,
                                  reprobe_after=5)
        prev.append(ctl)
        for _ in range(3):                  # open the reliability gate
            ctl.reliability.validate([0], [1], [0], provenance="sensor",
                                     model_version=m.version)
        gr.goal = g
        first = ctl.decide(s, DT, obs)
        chunk = ctl.planner.chunk_cost
        later = [ctl.decide(s, DT, obs) for _ in range(5)]
        return ctl, first, chunk, later, max(d.elapsed for d in later)

    # the planner's own time; was deadline + one whole synchronous chunk
    # (4 x 4 ms sleeps, 25 ms measured). M is the statistical contract.
    (ctl, first, chunk, later, worst), att = _timed(
        measure, lambda r: (r[1].source == "fallback_timeout" and
                            r[1].plan.elapsed <= deadline + 0.006 and r[4] < deadline),
        f"first call <= deadline + 6 ms and pre-empted calls < deadline ({deadline})")
    assert all(d.source == "fallback_timeout" for d in later)
    assert all(d.command == acf(policy_vector(obs)) for d in later), "not the PPO action"
    Slow.delay = 0.0                         # the slowness was transient
    assert ctl.planner.wait_idle(1.0)        # the abandoned call returns
    resumed = [ctl.decide(s, DT, obs).source for _ in range(3)]
    # THE CONTRACT IS "NOT LATCHED OUT": once the slowness passes the planner
    # takes control again. Whether it gets there through a forced re-probe or
    # because its cost estimate already fits is RUNNER-SPEED dependent — the
    # same commit (bcc846e) passed on one CI run and failed `probes >= 1` on
    # the next. The re-probe escape itself is pinned on a fake clock in
    # tests/unit/test_foundation_planning_unit.py
    # (deadline_discard_preempt_and_reprobe), so it is not re-asserted here.
    assert "planner" in resumed, (resumed, ctl.planner.stats)
    p = acf.action_probs(policy_vector(obs))
    assert int(np.argmax(p)) == later[0].command and abs(p.sum() - 1) < 1e-6
    print(f"  C. deadline {deadline * 1e3:.0f} ms, chunk >= {chunk * 1e3:.1f} ms: first call "
          f"{first.elapsed * 1e3:.1f} ms (abandoned at the deadline), pre-empted calls "
          f"<= {worst * 1e3:.2f} ms -> PPO action (attempt {att}); after the model sped "
          f"up: {resumed}")


def test_remap(model, reg, store):
    inner = PointMassAdapter("discrete", seed=0)
    remapped = ControlRemap(inner, [0, 3, 4, 2, 1])
    m = PointMassMechanism("pm.linear")
    m.set_params(model.damping, model.gain, model.var_vel, model.var_pos)
    store.add_candidate("mechanism", "pm.linear", m.to_record(), "pm.state")
    ctl, gr = make_controller(m, inner)
    before, _ = run_controlled(ctl, gr, inner, EVAL_SEEDS[:3])
    for _ in range(12):
        store.record_validation("pm.linear", True, provenance="sensor")
    assert store.promote("pm.linear")[0]
    ctl.knowledge_gate = store.gate("pm.linear")
    sources, post_rows, alarm_at = [], [], []
    step = [0]

    def on(obs, cmd, nxt, info):
        step[0] += 1
        s, _, _ = state_from_obs(obs)
        post_rows.append((s, encode_discrete(np.array([[cmd]]))[0], DT, state_from_obs(nxt)[0]))
        if info["alarm"] and not alarm_at:
            alarm_at.append(step[0])
            store.on_dynamics_change(["pm.linear"], f"CUSUM {info['kind']}")
    n0 = len(ctl.log)
    broken, _ = run_controlled(ctl, gr, remapped, EVAL_SEEDS[:2], on_extra=on)
    srcs = [x["source"] for x in list(ctl.log)[n0:]]
    first_fb = next(i for i, s in enumerate(srcs) if s != "planner")
    assert first_fb <= 3, f"planner kept control {first_fb} steps after the remap"
    assert alarm_at and store.status("pm.linear") == "needs_revalidation"
    assert srcs[first_fb + 1:].count("planner") == 0, "gate reopened without a refit"
    m.fit(transitions(post_rows))                    # version + 1
    rate = []
    run_controlled(ctl, gr, remapped, EVAL_SEEDS[2:3],
                   on_extra=lambda o, c, n, i: rate.append(i["z2"] < 9.0))
    assert np.mean(rate) > 0.9
    assert store.revalidate("pm.linear", True, provenance="sensor",
                            new_record=m.to_record()) == "consolidated"
    n1 = len(ctl.log)
    after, _ = run_controlled(ctl, gr, remapped, EVAL_SEEDS[3:6])
    share = [x["source"] for x in list(ctl.log)[n1:]].count("planner") / (len(ctl.log) - n1)
    assert share > 0.9 and after < 0.6 * broken, (share, after, broken)
    rot = np.array([[0, -1], [1, 0]]) * 4
    # wall contacts (unmodelled) bias the least-squares fit, so require only
    # that the refit is nearer the true rotated gain than the stale identity
    assert np.linalg.norm(m.gain - rot) < 0.5 * np.linalg.norm(m.gain - 4 * np.eye(2)), m.gain
    print(f"  D. remap: planner lost control after {first_fb} real step(s), CUSUM alarm at "
          f"step {alarm_at[0]}, mechanism needs_revalidation; refit v{m.version} learned "
          f"the rotation; real late dist before {before:.3f} / during {broken:.3f} / after "
          f"{after:.3f}, planner share {share:.0%}")
    return m, ctl, gr


def test_smdp():
    gamma = 0.9
    for naive in (False, True):
        q = {"opt": 0.0, "prim": 0.0}
        for _ in range(200):
            rew = [0.0] * 9 + [1.0]
            tgt = smdp_target(rew, gamma, 0.0, True) if not naive else 1.0
            q["opt"] += 0.1 * (tgt - q["opt"])
            q["prim"] += 0.1 * (smdp_target([0.6], gamma, 0.0, True) - q["prim"])
        best = max(q, key=q.get)
        if naive:
            assert best == "opt", "witness: one-step discount over-credits long options"
        else:
            assert best == "prim" and abs(q["opt"] - gamma ** 9) < 1e-3, q
    print(f"  E. SMDP: Q(option)={gamma ** 9:.3f} < Q(primitive)=0.6 -> primitive; naive "
          f"one-step discount picks the 10-step option")


def test_discrimination():
    import torch
    from developmental_ai.policy.actor_critic import StandaloneActorCritic

    def ac(seed):
        torch.manual_seed(seed)
        return StandaloneActorCritic(obs_dim=4, action_dim=5, hidden_dim=32)
    a = ac(1)
    b = ac(2)
    b.actor.load_state_dict(a.actor.state_dict())              # byte copy
    c = ac(3)
    c.actor.load_state_dict(a.actor.state_dict())
    with torch.no_grad():
        for p in c.actor.parameters():
            p.add_(1e-7 * torch.randn_like(p))
    d = ac(4)                                                   # different policy
    e = ac(5)
    e.actor.load_state_dict(a.actor.state_dict())
    with torch.no_grad():                                       # permute head rows
        perm = torch.tensor([4, 0, 1, 2, 3])
        e.actor.action_head.weight.copy_(a.actor.action_head.weight[perm])
        e.actor.action_head.bias.copy_(a.actor.action_head.bias[perm])
    rng = np.random.default_rng(0)
    shared = rng.normal(size=(32, 4)) * 2
    skills = {"skill:chop#1": a, "skill:reach#7": b, "skill:noise#3": c,
              "skill:chop#2": d, "skill:perm#9": e}             # names deliberately reused
    fbs = {k: ActorCriticFallback(v) for k, v in skills.items()}
    bare = {k: behaviour_signature(shared, probs_fn=f.action_probs) for k, f in fbs.items()}
    assert {f.verdict for f in find_duplicates(bare, threshold=1e-3)} == \
        {"indistinguishable_on_probes"}
    evidence = {k: rng.normal(size=(24, 4)) * 2 for k in skills}   # where each one ran
    probes = build_probe_set(evidence, shared=shared)
    sigs = {k: behaviour_signature(probes, probs_fn=f.action_probs) for k, f in fbs.items()}
    found = find_duplicates(sigs, threshold=1e-3, probe_set=probes)
    assert all(f.verdict == "duplicate" for f in found), [f.statement for f in found]
    dup = {frozenset((f.a, f.b)) for f in found}
    from developmental_ai.foundation.planning import behavioural_divergence
    ids = sorted(sigs)
    divs = {frozenset((x, y)): behavioural_divergence(sigs[x], sigs[y])
            for i, x in enumerate(ids) for y in ids[i + 1:]}
    want = {frozenset(p) for p in [("skill:chop#1", "skill:reach#7"),
                                   ("skill:chop#1", "skill:noise#3"),
                                   ("skill:reach#7", "skill:noise#3")]}
    assert dup == want, dup
    assert fbs["skill:noise#3"].weights_hash() != fbs["skill:chop#1"].weights_hash()
    hi_dup = max(divs[p] for p in want)
    lo_other = min(v for p, v in divs.items() if p not in want)
    assert lo_other > 10 * 1e-3 > 1e-3 > 100 * hi_dup, (hi_dup, lo_other)
    print(f"  F. duplicates by behaviour: {sorted(tuple(sorted(p)) for p in dup)} (max JS "
          f"{hi_dup:.1e} bits; '{found[0].statement}'); same-name/different-seed and "
          f"head-permuted copies not flagged (min JS {lo_other:.3f} bits; threshold 1e-3)")


def test_calibration():
    rng = np.random.default_rng(0)
    comps, truth, preds, outs = [], [], [], []
    for _ in range(1500):
        p = rng.uniform(0.05, 0.95)
        c = BetaCompetence()
        for _ in range(int(rng.integers(3, 41))):
            c.record(bool(rng.random() < p), provenance="sensor")
        n0 = c.n
        try:
            c.record(True, provenance="imagined")
            raise AssertionError("imagined success accepted")
        except ProvenanceError:
            pass
        assert c.n == n0
        comps.append(c)
        truth.append(p)
        preds.append(c.mean())
        outs.append(int(rng.random() < p))                     # next REAL attempt
    cal = calibration(preds, outs)
    cov = interval_coverage(comps, truth, 0.9)
    assert cal["ece"] < 0.05 and 0.86 <= cov <= 0.94, (cal["ece"], cov)
    print(f"  G. competence: ECE {cal['ece']:.3f}, Brier {cal['brier']:.3f}, 90% interval "
          f"coverage {cov:.3f}; 1500 imagined successes rejected")


def reach_attempt(env, start, kp, kd, budget=12, radius=0.25):
    """Real execution of the reach skill from `start` toward the origin."""
    env.start_override = (np.asarray(start, float), np.zeros(2))
    obs = env.reset(seed=int(abs(start[0]) * 1e6) % 100000)
    env.start_override = None
    visited = []
    for t in range(budget):
        ch = {o.channel: o for o in obs}
        pos, vel = ch["pos"].value, ch["vel"].value
        visited.append(pos.copy())
        if np.linalg.norm(pos) < radius:
            return visited, True, ch["pos"].provenance
        f = np.clip(-kp * pos - kd * vel, -1, 1)
        obs, _, _, _ = env.step(make_action(obs, env.action_spec(), f))
    pos = {o.channel: o for o in obs}["pos"].value
    visited.append(pos.copy())
    return visited, bool(np.linalg.norm(pos) < radius), "sensor"


def test_initiation_termination_retention(retention):
    env = PointMassAdapter("box", noise=0.01, limit=4.0, episode_len=40, seed=0)
    rng = np.random.default_rng(1)
    sk = LearnedSkill("skill:reach-origin", "fixture:pd(kp=1.5,kd=0.4)", state_dim=2,
                      representation="pm.state", representation_version="1.0.0",
                      max_duration=12)
    for i in range(160):
        st = rng.uniform(-3, 3, 2)
        vis, ok, prov = reach_attempt(env, st, 1.5, 0.4)
        sk.record_attempt(st, vis, ok, provenance=prov, evidence_id=f"reach/{i}")
        if ok:
            retention.add_probe(sk.skill_id, st)
    sk.refit()
    test = rng.uniform(-3, 3, (80, 2))
    truth = np.array([reach_attempt(env, s, 1.5, 0.4)[1] for s in test])
    pred = sk.initiation.prob(test) > 0.5
    acc, base = float(np.mean(pred == truth)), float(max(truth.mean(), 1 - truth.mean()))
    assert acc >= 0.85 and acc > base + 0.1, (acc, base)
    near, far = sk.termination.prob(np.array([[0.05, 0.0]]))[0], \
        sk.termination.prob(np.array([[2.0, 0.0]]))[0]
    assert near > 0.5 > far, (near, far)
    print(f"  H. learned initiation accuracy {acc:.2f} (base rate {base:.2f}, success "
          f"rate {truth.mean():.2f}); termination p(near)={near:.2f}, p(far)={far:.2f}; "
          f"competence {sk.competence.mean():.2f} from {sk.competence.n} real attempts")

    ctrl = LearnedSkill("skill:reach-origin-ctrl", "fixture:pd(kp=1.5,kd=0.4)#2",
                        state_dim=2, representation="pm.state",
                        representation_version="1.0.0")
    for p in retention.probe_states(sk.skill_id):
        retention.add_probe(ctrl.skill_id, p)

    def measure(sid, t, kp):
        res = [reach_attempt(env, p, kp, 0.4)[1] for p in retention.probe_states(sid)]
        return retention.measure(sid, t, res, provenance="sensor")
    measure(sk.skill_id, 0, 1.5)
    measure(ctrl.skill_id, 0, 1.5)
    measure(sk.skill_id, 1, 0.15)                 # interference overwrote the gains
    measure(ctrl.skill_id, 1, 1.5)                # untouched
    f, fc = retention.forgetting(sk.skill_id), retention.forgetting(ctrl.skill_id)
    assert f["significant"] and not fc["significant"], (f, fc)
    print(f"  I. retention on {len(retention.probe_states(sk.skill_id))} probes: damaged "
          f"{f['best']['rate']:.2f}->{f['current']['rate']:.2f} (forgetting flagged), "
          f"control {fc['best']['rate']:.2f}->{fc['current']['rate']:.2f} (not flagged)")
    return sk, ctrl


def test_resume(path, skills, m, ctl, gr, reg, store, retention):
    env = ControlRemap(PointMassAdapter("discrete", seed=0), [0, 3, 4, 2, 1])
    mem = EpisodeMemory(max_episodes=3)
    for i in range(5):
        mem.add(f"ep{i}", [f"ev{i}.0"])
    save_planning_state(path, skills=skills, mechanisms=[m.to_record()], knowledge=store,
                        retention=retention, controller=ctl, memory=mem,
                        extra={"goal": gr.goal.copy()})
    reg2 = RepresentationRegistry()
    reg2.declare("pm.state", str(reg.current("pm.state")))
    store2, ret2, mem2 = KnowledgeStore(reg2), RetentionMonitor(), EpisodeMemory()
    out = None
    m2 = None
    out = load_planning_state(path, knowledge=store2, retention=ret2, memory=mem2)
    m2 = PointMassMechanism.from_record(out["mechanisms"][0])
    ctl2, gr2 = make_controller(m2, env)
    ctl2.knowledge_gate = store2.gate("pm.linear")
    load_planning_state(path, controller=ctl2)
    gr2.goal = out["extra"]["goal"]
    assert [s.to_record() for s in out["skills"]] == [s.to_record() for s in skills]
    assert out["mechanisms"][0] == m.to_record() and m2.version == m.version
    assert store2.status("pm.linear") == store.status("pm.linear") == "consolidated"
    assert len(mem2) == 3 and mem2.evicted == 2
    assert np.allclose(ret2.probe_states(skills[0].skill_id),
                       retention.probe_states(skills[0].skill_id))
    for s_orig, s_new in zip(skills, out["skills"]):
        assert s_new.competence.state_dict() == s_orig.competence.state_dict()
    obs = env.reset(seed=77)
    st, _, _ = state_from_obs(obs)
    a = [ctl.decide(st, DT, obs) for _ in range(4)]
    b = [ctl2.decide(st, DT, obs) for _ in range(4)]
    assert [x.source for x in a] == [x.source for x in b] == ["planner"] * 4
    assert [x.command for x in a] == [x.command for x in b], "resume changed decisions"
    print(f"  J. round trip: {len(skills)} skills, mechanism {m.mechanism_id} v{m.version}, "
          f"store/retention/memory/planner state restored; next 4 decisions identical")
    return store2, reg2


def test_stale_recovery(m, ctl, gr, reg, store, path):
    env = ControlRemap(PointMassAdapter("discrete", seed=0), [0, 3, 4, 2, 1])
    reg.bump("pm.state", "2.0.0", "positions re-expressed relative to the goal")
    obs = env.reset(seed=5)
    st, g, _ = state_from_obs(obs)
    gr.goal = g
    d = ctl.decide(st, DT, obs)
    assert d.source == "fallback_unreliable" and "needs_revalidation" in d.reason, d.reason
    reg3 = RepresentationRegistry()
    reg3.declare("pm.state", "2.0.0")
    s3 = KnowledgeStore(reg3)
    load_planning_state(path, knowledge=s3)
    assert s3.status("pm.linear") == "needs_revalidation", "loaded stale knowledge as valid"
    rows = collect_random(env, 3, 900)
    m.fit(transitions(rows))
    store.revalidate("pm.linear", True, provenance="sensor", new_record=m.to_record())
    late, _ = run_controlled(ctl, gr, env, EVAL_SEEDS[:2])
    tail = [x["source"] for x in list(ctl.log)[-30:]]
    assert tail.count("planner") > 20, tail
    print(f"  K. major bump -> fallback ({d.reason}); store loaded after the bump is "
          f"needs_revalidation; refit v{m.version} + revalidation -> planner again "
          f"(real late dist {late:.3f})")


def test_bridge():
    from developmental_ai.skill_bank.skill_bank import Skill as LegacySkill
    with tempfile.TemporaryDirectory() as d:
        rows = [dataclasses.asdict(LegacySkill(skill_id="break_oak_log_01", name="break_oak_log",
                                               asked_log=[1, 1, 0, 1], success_rate=0.97,
                                               preconditions={"tree_visible": True})),
                dataclasses.asdict(LegacySkill(skill_id="break_oak_log_02",
                                               name="break_oak_log"))]
        with open(os.path.join(d, "registry.json"), "w") as f:
            json.dump({"skills": rows, "last_updated": 0.0}, f)

        def snap():
            out = {}
            for root, dirs, files in os.walk(d):
                for fn in files:
                    p = os.path.join(root, fn)
                    out[p] = (os.path.getmtime(p), hashlib.sha1(open(p, "rb").read()).hexdigest())
                out[root] = tuple(sorted(dirs + files))
            return out
        before = snap()
        sk = read_skill_bank(d)
        assert snap() == before, "bridge modified the bank"
        assert [s.skill_id for s in sk] == ["break_oak_log_01", "break_oak_log_02"]
        assert sk[0].payload["legacy_competence"]["a"] == 4.0
        assert all(s.payload["display_name_is_identity"] is False for s in sk)
    try:
        read_skill_bank(os.path.join(os.getcwd(), "skill_bank_mc_curiosity"))
        raise AssertionError("live bank path accepted")
    except Exception as e:
        assert "refusing" in str(e)
    print("  L. bridge: 2 rows from real skill_bank.Skill dataclass, same name -> 2 skills, "
          "competence from asked_log only, directory unchanged, live bank refused")


# ---------------------------------------------------------------- verifier findings
def test_deadline_heavy_tail():
    class Spiky(PointMassMechanism):
        def __init__(self, seed):
            super().__init__("pm.spiky")
            self.rng = np.random.default_rng(seed)

        def _predict(self, state, action, dt):
            if self.rng.random() < 0.02:
                time.sleep(0.030)
            return super()._predict(state, action, dt)
    deadline, frac = 0.010, {}

    def arm(isolate):
        lat = []
        for seed in (0, 1):
            m = Spiky(seed)
            m.set_params([0.8, 0.8], 4 * np.eye(2), [1e-4] * 2, [1e-6] * 2)
            env = PointMassAdapter("discrete", seed=0)
            ctl, gr = make_controller(m, env, horizon=4, n_iters=2, deadline_s=deadline,
                                      reprobe_after=5, seed=seed, isolate_calls=isolate)
            for _ in range(3):
                ctl.reliability.validate([0], [1], [0], provenance="sensor",
                                         model_version=m.version)
            obs = env.reset(seed=seed)
            st, g, _ = state_from_obs(obs)
            gr.goal = g
            for _ in range(150):
                t0 = time.perf_counter()
                ctl.decide(st, DT, obs)
                lat.append((time.perf_counter() - t0) / deadline)
            ctl.planner.wait_idle(1.0)
        return (float(np.mean(np.array(lat) > 1.25)), float(np.max(lat)),
                float(np.quantile(lat, 0.99)))
    # load only ADDS overruns, so the synchronous witness needs no retry
    frac[False] = arm(False)
    assert frac[False][0] >= 0.03, f"witness: the fixture must produce overruns {frac}"
    frac[True], att = _timed(lambda: arm(True), lambda r: r[0] < 0.02,
                             "deadline held under heavy tails (< 2% over 1.25x)")
    print(f"  M. heavy-tailed model (2% x 30 ms), deadline 10 ms: > 1.25x on "
          f"{frac[True][0]:.1%} of decisions (p99 {frac[True][2]:.2f}x, max "
          f"{frac[True][1]:.2f}x); synchronous witness {frac[False][0]:.1%} (max "
          f"{frac[False][1]:.1f}x); attempt {att}")


MUD_R = 0.6


def _in_mud(pos):
    return np.linalg.norm(np.asarray(pos).reshape(-1, 2), axis=1) < MUD_R


class MudAdapter(PointMassAdapter):
    """Inside the disc: gain x0.15, damping 0.5 (after the verifier's E11a)."""

    def step(self, action):
        if _in_mud(self.pos)[0]:
            g, self.gain = self.gain, self.gain * 0.15
            self.vel = self.vel * (0.5 / 0.8)
            try:
                return super().step(action)
            finally:
                self.gain = g
        return super().step(action)


class MudModel(PointMassMechanism):
    """Correct outside the disc; inside it uses (in_gain, in_damp)."""

    def __init__(self, in_gain, in_damp):
        super().__init__("pm.mud")
        self.in_gain, self.in_damp = in_gain, in_damp

    def _means(self, state, action, dt):
        vel, pos, f = state.vel[:, 0, :], state.pos[:, 0, :], action[:, 0, :]
        m = _in_mud(pos)[:, None]
        damp = np.where(m, self.in_damp, self.damping[None])
        v2 = damp * vel + np.where(m, self.in_gain, 1.0) * (f @ self.gain.T) * dt[:, None]
        return pos + v2 * dt[:, None], v2


def _mud_run(ctl, gr, env, seeds):
    entries = planner_in_mud = 0
    for sd in seeds:
        obs = env.reset(seed=sd)
        while True:
            st, g, prov = state_from_obs(obs)
            gr.goal = g
            here = bool(_in_mud(st.pos[0, 0])[0])
            d = ctl.decide(st, DT, obs)
            nxt, term, trunc, _ = env.step(make_action(obs, env.action_spec(), d.command))
            s2 = state_from_obs(nxt)[0]
            if d.source == "planner":
                planner_in_mud += here
                entries += bool(_in_mud(s2.pos[0, 0])[0] and not here)
            ctl.observe(st, d.command, DT, s2, provenance=prov)
            obs = nxt
            if term or trunc:
                break
    return entries, planner_in_mud


def test_region_gate():
    seeds = list(range(700, 730))
    out = {}
    for regions in (False, True):
        env = MudAdapter("discrete", seed=0, episode_len=30)
        m = MudModel(3.0, 0.8)                                  # claims a boost
        m.set_params([0.8, 0.8], 4 * np.eye(2), [1e-4] * 2, [1e-6] * 2)
        gr = GoalReward()
        pc = ProportionalController(seed=3)
        pl = BoundedPlanner(m, gr, env.action_spec(), n_entities=1, action_dim=2,
                            encode=encode_discrete, horizon=8, n_candidates=64, n_iters=3,
                            deadline_s=1.0, seed=1)
        ctl = PlanningController(pl, lambda o: pc(policy_vector(o)),
                                 ReliabilityMonitor(window=10, min_count=3, max_z2=9.0,
                                                    regions=regions))
        out[regions] = (_mud_run(ctl, gr, env, seeds), ctl, m, gr, env)
    (e_glob, _), (e_reg, _) = out[False][0], out[True][0]
    assert e_glob >= 3, f"witness: the global gate should keep re-entering ({e_glob})"
    assert e_reg <= e_glob / 2, f"region gate re-entries {e_reg} vs global {e_glob}"
    _, ctl, m, gr, env = out[True]
    centre = ctl._key(state_from_obs(env.reset(seed=0))[0],
                      np.zeros((1, 4)))                         # (0, 0) position key
    assert ctl.reliability.region_unreliable(centre)[0], "the mud was not marked"
    m.in_gain, m.in_damp = 0.15, 0.5                             # refit to the truth
    m.set_params([0.8, 0.8], 4 * np.eye(2), [1e-4] * 2, [1e-6] * 2)   # version + 1
    _, after_in_mud = _mud_run(ctl, gr, env, list(range(800, 815)))
    assert after_in_mud > 0, "the refitted model never controlled a step in the mud again"
    assert not ctl.reliability.region_unreliable(centre)[0], "region still marked"
    print(f"  N. mud fixture, 30 episodes: planner entries into the mud the model is wrong "
          f"about: global gate {e_glob}, region gate {e_reg}; after a refit to the true "
          f"dynamics the planner controlled {after_in_mud} steps inside the mud and the "
          f"region is no longer marked")


def test_offprobe_duplicates():
    perm = np.array([1, 2, 3, 4, 0])

    def probs(p, offprobe=False):
        def f(x):
            z = p["W2"] @ np.tanh(p["W1"] @ x + p["b1"]) + p["b2"]
            if offprobe and np.max(np.abs(x)) > 7.0:
                z = z[perm]
            e = np.exp(z - z.max())
            return e / e.sum()
        return f

    def pol(r):
        return {"W1": r.normal(0, 0.5, (16, 4)), "b1": r.normal(0, 0.1, 16),
                "W2": r.normal(0, 0.5, (5, 16)), "b2": r.normal(0, 0.1, 5)}
    n, flags = 10, {"bare_offprobe": 0, "offprobe": 0, "copy": 0, "samename": 0}
    for tr in range(n):
        r = np.random.default_rng(900 + tr)
        A, B = pol(r), pol(r)
        shared = r.normal(size=(32, 4)) * 2.0
        fa = probs(A)
        pairs = {"offprobe": probs(A, True), "copy": probs({k: v.copy() for k, v in A.items()}),
                 "samename": probs(B)}
        bare = find_duplicates({"a": behaviour_signature(shared, probs_fn=fa),
                                "b": behaviour_signature(shared, probs_fn=pairs["offprobe"])})
        flags["bare_offprobe"] += int(len(bare) == 1 and
                                      bare[0].verdict == "indistinguishable_on_probes")
        for k, fb in pairs.items():
            ps = build_probe_set({"a": r.normal(size=(64, 4)) * 4.0,
                                  "b": r.normal(size=(64, 4)) * 4.0}, shared=shared)
            found = find_duplicates({"a": behaviour_signature(ps, probs_fn=fa),
                                     "b": behaviour_signature(ps, probs_fn=fb)},
                                    probe_set=ps)
            flags[k] += int(any(f.verdict == "duplicate" for f in found))
    assert flags["bare_offprobe"] >= 0.8 * n, f"witness: bare probes miss the gap {flags}"
    assert flags["offprobe"] == 0 and flags["samename"] == 0 and flags["copy"] == n, flags
    print(f"  O. off-probe pairs: bare 32-probe set calls {flags['bare_offprobe']}/{n} "
          f"indistinguishable; evidence-built probe sets: duplicate verdicts offprobe "
          f"{flags['offprobe']}/{n}, byte copy {flags['copy']}/{n}, same-name other policy "
          f"{flags['samename']}/{n}")


def test_graded_verdict():
    perm = np.array([1, 2, 3, 4, 0])

    def pol(r):
        return {"W1": r.normal(0, 0.5, (16, 4)), "b1": r.normal(0, 0.1, 16),
                "W2": r.normal(0, 0.5, (5, 16)), "b2": r.normal(0, 0.1, 5)}

    def probs(p, temp=1.0, offprobe=False):
        def f(x):
            z = (p["W2"] @ np.tanh(p["W1"] @ x + p["b1"]) + p["b2"]) * temp
            if offprobe and np.max(np.abs(x)) > 7.0:
                z = z[perm]
            e = np.exp(z - z.max())
            return e / e.sum()
        return f
    n = 20
    tally = {k: {} for k in ("tempered", "copy", "samename", "offprobe", "drift0.1",
                             "bare_tempered")}
    js_t = []
    for tr in range(n):
        r = np.random.default_rng(1300 + tr)
        A, B = pol(r), pol(r)
        shared = r.normal(size=(32, 4)) * 2.0
        fa = probs(A)
        pairs = {"tempered": probs(A, 3.0), "copy": probs({k: v.copy() for k, v in A.items()}),
                 "samename": probs(B), "offprobe": probs(A, offprobe=True),
                 "drift0.1": probs({k: v + 0.1 * r.normal(size=v.shape) * np.abs(v).mean()
                                    for k, v in A.items()})}
        ev = {"a": r.normal(size=(64, 4)) * 4.0, "b": r.normal(size=(64, 4)) * 4.0}
        ps = build_probe_set(ev, shared=shared)
        for k, fb in pairs.items():
            f = find_duplicates({"a": behaviour_signature(ps, probs_fn=fa),
                                 "b": behaviour_signature(ps, probs_fn=fb)},
                                probe_set=ps, include_distinct=True)
            assert len(f) == 1
            tally[k][f[0].verdict] = tally[k].get(f[0].verdict, 0) + 1
            if k == "tempered":
                js_t.append(f[0].divergence)
        fb = find_duplicates({"a": behaviour_signature(shared, probs_fn=fa),
                              "b": behaviour_signature(shared, probs_fn=pairs["tempered"])},
                             include_distinct=True)
        tally["bare_tempered"][fb[0].verdict] = tally["bare_tempered"].get(fb[0].verdict, 0) + 1
    assert tally["tempered"].get("near_duplicate", 0) >= n - 1, tally
    assert tally["copy"] == {"duplicate": n} and tally["samename"] == {"distinct": n}, tally
    assert not {"duplicate", "near_duplicate"} & set(tally["offprobe"]), tally
    assert set(tally["bare_tempered"]) <= {"indistinguishable_on_probes", "distinct"}, tally
    print(f"  R. graded verdicts over {n} trials (evidence-built probes): tempered x3 "
          f"{tally['tempered']} (JS {np.mean(js_t):.3f} bits: distinct by JS alone), byte "
          f"copy {tally['copy']}, same-name other policy {tally['samename']}, off-probe "
          f"{tally['offprobe']}; bare probes, tempered {tally['bare_tempered']}; reported: "
          f"drift 0.1 {tally['drift0.1']}")


def _model_threads():
    import threading
    return sum(1 for t in threading.enumerate()
               if t.name == "foundation-planner-model-call" and t.is_alive())


def test_hung_model():
    import threading

    class Hanging(PointMassMechanism):
        mode, release, rollouts = "once", None, 0

        def rollout(self, *a, **k):
            n = self.rollouts
            self.rollouts += 1
            if self.mode == "all" or n == 0:
                self.release.wait(30.0)
            return super().rollout(*a, **k)

    env = PointMassAdapter("discrete", seed=0)
    obs = env.reset(seed=1)
    s, g, _ = state_from_obs(obs)
    def run_mode(mode, kw, span):
        m = Hanging()
        m.mode, m.release = mode, threading.Event()
        m.set_params([0.8, 0.8], 4 * np.eye(2), [1e-4] * 2, [1e-6] * 2)
        safety = threading.Timer(6.0, m.release.set)      # old code would block forever
        safety.daemon = True
        safety.start()
        base = _model_threads()
        ctl, gr = make_controller(m, env, horizon=4, n_iters=2, deadline_s=0.010,
                                  n_candidates=32, **kw)
        for _ in range(3):
            ctl.reliability.validate([0], [1], [0], provenance="sensor", model_version=m.version)
        gr.goal = g
        t0, first_planner, srcs, peak, obs_worst = time.perf_counter(), None, [], 0, 0.0
        while time.perf_counter() - t0 < span:
            d = ctl.decide(s, DT, obs)
            srcs.append(d.source)
            if d.source == "planner" and first_planner is None:
                first_planner = time.perf_counter() - t0
                if mode == "once":
                    break
            o0 = time.perf_counter()          # observe()'s wait for an in-flight call
            ctl.planner.wait_usable() if hasattr(ctl.planner, "wait_usable") else \
                ctl.planner.wait_idle()
            obs_worst = max(obs_worst, time.perf_counter() - o0)
            peak = max(peak, _model_threads() - base)
            time.sleep(0.005)
        snap = (dict(ctl.planner.stats), getattr(ctl.planner, "live_poisoned", lambda: 0)())
        back = None
        if mode == "all":                     # snapshot taken BEFORE the release
            m.release.set()
            t1 = time.perf_counter()
            while time.perf_counter() - t1 < 2.0:
                if ctl.decide(s, DT, obs).source == "planner":
                    back = time.perf_counter() - t1
                    break
                time.sleep(0.005)
        res = (first_planner, srcs, peak, obs_worst, snap[0], snap[1],
               getattr(ctl.planner, "hang_after_s", None), back)
        m.release.set()
        safety.cancel()
        return res

    (fp, srcs, _, ow, st, _, hang, _), k1 = _timed(
        lambda: run_mode("once", {}, 2.5),
        lambda r: (r[0] is not None and r[0] < 2.0 and r[3] < 1.2 and
                   all(x != "planner" for x in r[1][:-1]) and r[4]["respawns"] == 1),
        "hang once: planner back < 2 s, observe wait < 1.2 s, exactly 1 respawn "
        "(first_planner, sources, -, observe wait, stats, ...)")
    (fa, srca, peak, owa, sta, poisoned, _, back), k2 = _timed(
        lambda: run_mode("all", dict(hang_after_s=0.05, max_respawns=2,
                                     respawn_backoff_s=0.05), 1.0),
        lambda r: (r[0] is None and all(x == "fallback_timeout" for x in r[1]) and
                   r[4]["respawns"] == 2 and r[5] == 2 and r[2] <= 3 and r[7] is not None),
        "always hangs: all fallback, 2 respawns, <= 3 threads, back after release")
    print(f"  Q. hang once (default watchdog, hang_after {hang:.1f} s): planner back after "
          f"{fp:.2f} s ({len(srcs) - 1} fallback decisions, respawns {st['respawns']}), worst "
          f"observe {ow * 1e3:.0f} ms; always hangs: {len(srca)} decisions all fallback, "
          f"respawns {sta['respawns']} (cap 2), peak model-call threads +{peak}, planner back "
          f"{back * 1e3:.0f} ms after calls return; attempts {k1}, {k2}")


if __name__ == "__main__":
    t0 = time.time()
    env0 = PointMassAdapter("discrete", seed=0)
    model = PointMassMechanism("pm.fit")
    model.fit(transitions(collect_random(env0, 5, 100)))
    test_closed_loop_and_exploitation(model)
    test_deadline()
    reg = RepresentationRegistry()
    reg.declare("pm.state", "1.0.0")
    store = KnowledgeStore(reg, promote_min_n=10)
    m, ctl, gr = test_remap(model, reg, store)
    test_smdp()
    test_discrimination()
    test_calibration()
    retention = RetentionMonitor(max_probes=24, seed=0)
    sk, ctrl = test_initiation_termination_retention(retention)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "planning_state.json")
        test_resume(path, [sk, ctrl], m, ctl, gr, reg, store, retention)
        test_stale_recovery(m, ctl, gr, reg, store, path)
    test_bridge()
    test_deadline_heavy_tail()
    test_region_gate()
    test_offprobe_duplicates()
    test_graded_verdict()
    test_hung_model()
    print(f"  ({time.time() - t0:.1f}s)")
    print("[foundation-planning] ALL PASS")
