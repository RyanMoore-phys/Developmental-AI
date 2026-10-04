"""Foundation discovery smoke (2026-10-03) — plan Stage 9 test list + gate.

WHAT IS CLAIMED

    "improvement plan.md" Stage 9: "change explanations when fitting
    existing parameters is inadequate." Tests: "known hidden rules;
    spurious correlation; insufficient evidence; confounding fixtures;
    regime changes; mechanism proliferation; catastrophic replacement;
    recovery from rejected candidates." Gate: "structural revision improves
    held-out predictions or adaptation compared with parameter-only updates
    under a stated budget."

WHY IT IS WORTH A CONTRACT

    Every one of these has a cheap fake, and this repo has shipped the
    pattern behind most of them. A structure search scored on the data it
    was fitted to accepts noise (so: fit and validation are different
    EPISODES, and pure-noise data must yield nothing). A correlation that
    fits well gets read as a cause (the `places` counter "proved" wasted
    no-ops and listed iron_axe: 2281 — CLAUDE.md §5), so the causal label
    may only come from rows an agent ACTION set. A replacement rule that
    trusts one good window installs a worse model; a retirement that
    deletes makes a returning regime cost a full relearn; a guard on
    revision that nothing re-opens is a latch (§4.1). And "structural
    revision helps" is meaningless unless the parameter-only update gets
    the SAME budget and the comparison is on episodes neither saw — and
    unless structure does NOT win when only parameters changed.

Contracts:
    A. Known hidden rules are recovered: hidden switch (y = 2x iff h == 1,
       else -1), threshold (y = 3 iff temp >= 0.5, else -1) and lever/door
       (y = 2*lever iff door == 1): probe-point predictions within 0.15 of
       the rule, threshold within 0.02, held-out NLL within 0.15 nats of the
       noise floor, distractor variables absent from the best mechanism.
       FALSIFICATION: the same tables with y shuffled across rows yield ZERO
       accepted candidates that ADD a dependency (dropping x there is a
       correct simplification and allowed — measured in 4/30 seeds); every
       fit sees fit episodes only; validation rejects at least one candidate
       that passed the dev BIC test (so it is load-bearing, not decoration).
       Swept offline over 20 seeds x 3 worlds: 0 failures.
    B. Spurious correlation: S predicts y observationally (C confounds
       both, C hidden) and IS accepted as a predictive dependency — but
       labelled `observational`, reason intervention_null, a contradiction
       recorded, interventional effect |.| < 0.4 vs observational slope > 1. With
       no interventions and 3x more data the label stays `observational`
       (no_intervention): correlation never auto-upgrades.
    C. Confounding: X -> y with hidden U driving both. X is labelled
       `interventional`, and the label carries the INTERVENTIONAL effect
       (within 0.3 of the true 1.0) while the pooled fit's slope is biased
       (> 1.6). door in lever/door — modulating but never set by an action
       — stays observational while lever is interventional.
    D. Insufficient evidence: a search with too few validation episodes
       accepts nothing; a genuinely better candidate judged on windows too
       small to decide stays `candidate` (never promoted) and expires to
       retired("stale") after max_inconclusive windows — then comes back by
       revival once a decisive window arrives (escape path, §4.1).
    E. Regime A -> B -> A through ChangeTriggeredReviser: no alarm and no
       search in A; the change to B fires the CUSUM and triggers a bounded
       STRUCTURAL search whose winner (not the refit) is promoted and uses
       h; the return to A REVIVES the original mechanism — same ref,
       parameters bit-identical, ZERO searches during the return.
    F. Proliferation: 1 search accepts many candidates; admitting all of
       them into a registry with max_live 3 / max_retired 4 never exceeds
       either cap; the search itself stops at max_candidates.
    G. Catastrophic replacement prevented: a challenger that fits its own
       dev data better but is worse on validation is never promoted — over
       10 windows, and also when its windows ALTERNATE good/bad (sustained
       advantage required); the incumbent stays bit-identical.
    H. Recovery from rejected candidates: admitting then rejecting a
       candidate leaves the incumbent bit-identical and snapshot/restore
       reproduces the registry exactly; a fitter that raises on half the
       candidates neither aborts the search nor alters the parent.
    I. Budgets honoured with a slow fitter: 0.15 s/fit under a 0.5 s budget
       returns by 0.6 s ("wall"); a fitter that hangs for 30 s is abandoned
       and the search returns by budget + 0.1 s.
    J. COMPLETION GATE (3 seeds): after the structural rule change A -> B,
       a reviser allowed the full proposal language beats one allowed only
       REFIT (parameter-only), under the SAME SearchBudget, on final
       held-out episodes neither ever saw, by > 0.5 nats/row on every seed.
       CONTROL: after a parameter-only change (slope 2 -> 3) the structural
       reviser is NOT worse than parameter-only (within 0.05 nats) — it does
       not invent structure that is not there.

    K. (verifier B1, stage9_11 E9f) the wall budget holds on a LARGE, WIDE
       table: 150 continuous variables x 60 000 rows (600 episodes), pure
       noise, wall_s 0.25. MEASURED BEFORE THE FIX: 4.1-4.7 s (16-19x),
       evaluated = 1 — proposal enumeration (a quantile, an argsort and
       three np.unique sorts per variable) ran OUTSIDE the deadline guard.
       Contract: returns within 1.5 x wall_s with the default fitter AND
       with a slow fitter (0.05 s/fit), stops 'wall', and (default fitter)
       gets to evaluate more than the baseline (enumeration no longer eats
       the budget). The
       enumeration is lazy and deadline-checked; its ORDER is unchanged
       (checked against the eager list on a mixed discrete/continuous
       table), so budgets that end by max_candidates pick the same set.

    L. (verifier F6) the reviser's revision trigger also fires on
       MISSPECIFICATION, not only on a change alarm. Root y = 1 + 2x (sd
       0.2, fitted on 40 episodes); the world adds +-0.2 by a binary h the
       root ignores (zero mean, variance ratio ~2). With the change detector
       BLINDED (thresholds 1e9) the MisspecificationMonitor on the
       incumbent's predictive opens a revision on >= 7/8 seeds and the
       incumbent ends up using h on >= 7/8; with the monitor off as well,
       0/8 (no search ever runs — the pre-fix state of a quiet detector).
       The report carries the evidence count. FALSIFICATION: well-specified
       data, both detectors live, same 8 seeds x 60 episodes: 0
       misspecification triggers and 0 searches. Reported, not asserted:
       with the detector live the fixed CUSUM usually fires first, so on
       Gaussian predictives the monitor is mostly a second line.

    P. (Known limit "an abandoned fit keeps running in its thread", closed
       2026-10-03) ABANDONED FITS STOP, AND THEIR NUMBER IS CAPPED.
       P1 cooperative: an iterative fitter (25 refits through the library
       form, 30 ms apart; it never checks a clock itself) on the lever/door
       world, searched under wall_s 0.35 / 0.8 / 1.25 / 1.7 — the deadline
       lands in the baseline fit, in a candidate fit and inside the causal
       annotation. After each search returns, the number of live
       "discovery-fit" threads must be back to its pre-search value within
       2 x the iteration time (60 ms). MEASURED BEFORE THE FIX: the
       annotation's refits ran with NO deadline, so a search abandoned
       there left its thread running ~0.3-0.75 s after returning (the
       cancel now reaches every refit through forms.CancelToken).
       P2 non-cooperative: a fitter that blocks until released cannot be
       cancelled. With max_abandoned 2, four searches leave at most 2 such
       threads alive; searches 3 and 4 start no fit (evaluated 0) and stop
       "budget_exhausted". Before the fix every search leaked one more
       thread, without bound. ESCAPE (CLAUDE.md §4.1): once the hung fits
       return, the very next search runs normally — nothing is reset.

    TIMING. The wall-clock contracts (I, K and P1) go through _timed: the whole
    measurement is retried up to 3 times and passes if ANY attempt holds
    the bound (a scheduler stall on a loaded runner is not the code's
    fault; each was checked to miss on ALL attempts against the bug it
    guards). The attempt used is printed.

Run: PYTHONPATH=. python tests/_foundation_discovery_smoke.py
"""
import sys
import threading
import time

sys.path.insert(0, ".")

import numpy as np

from developmental_ai.foundation.discovery import (
    REFIT, ChangeTriggeredReviser, EvidenceTable, MechanismRegistry, SearchBudget,
    fit_mechanism, initial_mechanism, search)
from developmental_ai.foundation.discovery.fixtures import make_split_episodes

T0 = time.time()
NOISE_FLOOR = 0.5 * np.log(2 * np.pi * 0.2 ** 2) + 0.5     # NLL of N(0, 0.2^2) truth


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


def _cat(eps):
    return EvidenceTable.concat(eps)


def _probe(mech, **cols):
    n = len(next(iter(cols.values())))
    t = EvidenceTable({**{k: np.asarray(v, float) for k, v in cols.items()},
                       "y": np.zeros(n)}, "y", ["p"] * n, "probe")
    return mech.predict(t)[0]


def _nll(mech, tab):
    return float(-np.nanmean(mech.loglik(tab)))


def _shuffled(tab, seed):
    y = np.random.default_rng(seed).permutation(tab.y)
    cols = dict(tab.columns)
    cols["y"] = y
    return EvidenceTable(cols, "y", tab.episodes, tab.scope, tab.intervened)


def _grown(rep, parent):
    """Accepted candidates that ADD a dependency the parent did not have.
    (On shuffled data, dropping x is a legitimate simplification.)"""
    return [str(c.proposal) for c in rep.structural()
            if not set(c.mechanism.dependencies()) <= set(parent.dependencies())]


def _thresholds(mech):
    out = []
    for b in mech.branches:
        out += [c.value for c in b.predicate if c.var == "temp" and c.op in ("<", ">=")]
        if b.form == "piecewise" and b.pivot == "temp":
            out.append(b.params["t"])
    return out


# --------------------------------------------------------------------- A
def test_hidden_rules():
    lines = []
    # hidden switch
    dev, held = make_split_episodes("hidden_switch", "fx:hs", 0, 20, 8)
    seen = set()

    def recording(m, t, deadline=None):
        seen.update(t.unique_episodes())
        return fit_mechanism(m, t, deadline)

    rep = search(initial_mechanism("y", "linear", ("x",)), _cat(dev), SearchBudget(wall_s=10),
                 fitter=recording)
    assert rep.val_episodes and seen.isdisjoint(rep.val_episodes), "a fit saw validation"
    assert seen <= set(rep.fit_episodes), "a fit saw a non-fit episode"
    val_rejects = sum(v for k, v in rep.counts.items() if k.startswith("rejected_validation"))
    b = rep.best.mechanism
    deps = set(b.dependencies())
    assert {"h", "x"} <= deps and "d" not in deps, deps
    p = _probe(b, x=[0.5, -0.5, 0.5, -0.5], h=[1, 1, 0, 0], d=[0, 0, 0, 0])
    assert np.allclose(p, [1.0, -1.0, -1.0, -1.0], atol=0.15), p
    nll = _nll(b, _cat(held))
    assert nll < NOISE_FLOOR + 0.15, nll
    lines.append(f"switch: {rep.best.proposal.op} deps {sorted(deps)} held-out NLL {nll:.3f} "
                 f"(floor {NOISE_FLOOR:.3f}; root refit {_nll(rep.baseline, _cat(held)):.3f})")
    root = initial_mechanism("y", "linear", ("x",))
    null = search(root, _shuffled(_cat(dev), 1), SearchBudget(wall_s=10))
    assert not _grown(null, root), _grown(null, root)
    n_null = null.evaluated
    # threshold
    dev, held = make_split_episodes("threshold", "fx:th", 1, 20, 8)
    rep = search(initial_mechanism("y"), _cat(dev), SearchBudget(wall_s=10))
    b = rep.best.mechanism
    th = _thresholds(b)
    assert th and all(abs(t - 0.5) < 0.02 for t in th), th
    assert "x" not in b.dependencies(), b.dependencies()
    p = _probe(b, temp=[0.2, 0.8], x=[0, 0])
    assert np.allclose(p, [-1, 3], atol=0.15), p
    ll = b.loglik(_cat(held))
    med = float(-np.median(ll))
    lines.append(f"threshold: {rep.best.proposal.op} at {th[0]:.4f} (true 0.5), probes "
                 f"{np.round(p, 2).tolist()}, median held-out NLL {med:.3f}")
    null = search(initial_mechanism("y"), _shuffled(_cat(dev), 2), SearchBudget(wall_s=10))
    assert not _grown(null, initial_mechanism("y")), _grown(null, initial_mechanism("y"))
    n_null += null.evaluated
    # lever / door
    dev, held = make_split_episodes("lever_door", "fx:ld", 2, 20, 8)
    rep = search(initial_mechanism("y"), _cat(dev), SearchBudget(wall_s=10))
    b = rep.best.mechanism
    assert {"lever", "door"} <= set(b.dependencies()) and "z" not in b.dependencies()
    p = _probe(b, lever=[1, 0, 1, 0], door=[1, 1, 0, 0], z=[0, 0, 0, 0])
    assert np.allclose(p, [2, 0, 0, 0], atol=0.15), p
    nll = _nll(b, _cat(held))
    assert nll < NOISE_FLOOR + 0.15, nll
    lines.append(f"lever/door: depth {rep.best.depth}, {len(b.branches)} branches, probes "
                 f"{np.round(p, 2).tolist()}, held-out NLL {nll:.3f}")
    val_rejects += sum(v for k, v in rep.counts.items() if k.startswith("rejected_validation"))
    null = search(initial_mechanism("y"), _shuffled(_cat(dev), 3), SearchBudget(wall_s=10))
    assert not _grown(null, initial_mechanism("y")), _grown(null, initial_mechanism("y"))
    n_null += null.evaluated
    assert val_rejects >= 1, "validation never rejected a BIC-passing candidate: untested gate"
    print("  A. " + "; ".join(lines))
    print(f"     y shuffled (3 worlds, {n_null} candidates fitted): 0 accepted candidates add a dependency; "
          f"fits never saw a validation episode; {val_rejects} candidates passed dev BIC "
          f"but failed validation")
    return rep


# --------------------------------------------------------------------- B
def test_spurious():
    dev, _ = make_split_episodes("spurious", "fx:sp", 4, 40)
    rep = search(initial_mechanism("y"), _cat(dev), SearchBudget(wall_s=10))
    c = [c for c in rep.structural() if "S" in c.mechanism.dependencies()]
    assert c, "S is predictive observationally and should be accepted as a dependency"
    lab = c[0].mechanism.causal["S"]
    assert lab["status"] == "observational" and lab["reason"] == "intervention_null", lab
    assert abs(lab["effect_interventional"]) < 0.4 and lab["effect_observational"] > 1.0, lab
    from developmental_ai.foundation.discovery import contradictions
    assert contradictions(c[0].mechanism.causal)[0]["var"] == "S"
    dev2, _ = make_split_episodes("spurious", "fx:sp2", 5, 60, intervene_frac=0.0)
    rep2 = search(initial_mechanism("y"), _cat(dev2), SearchBudget(wall_s=10))
    c2 = [c for c in rep2.structural() if "S" in c.mechanism.dependencies()][0]
    lab2 = c2.mechanism.causal["S"]
    assert lab2["status"] == "observational" and lab2["reason"] == "no_intervention", lab2
    print(f"  B. spurious S accepted (val +{c[0].val['mean']:.2f} nats) but OBSERVATIONAL: "
          f"intervention_null on {lab['n_intervened']} intervened rows, effect "
          f"{lab['effect_interventional']:+.2f} vs observational slope "
          f"{lab['effect_observational']:+.2f}; 1800 purely observational rows "
          f"(slope {lab2['effect_observational']:+.2f}): still observational (no_intervention)")


# --------------------------------------------------------------------- C
def test_confounding(lever_rep):
    dev, _ = make_split_episodes("confounded", "fx:cf", 6, 60)
    rep = search(initial_mechanism("y"), _cat(dev), SearchBudget(wall_s=10))
    c = [c for c in rep.structural() if c.mechanism.dependencies() == ("X",)][0]
    lab = c.mechanism.causal["X"]
    assert lab["status"] == "interventional", lab
    assert abs(lab["effect_interventional"] - 1.0) < 0.3, lab
    assert lab["effect_observational"] > 1.6, lab
    ld = lever_rep.best.mechanism.causal
    assert ld["lever"]["status"] == "interventional", ld["lever"]
    assert ld["door"]["status"] == "observational" and ld["door"]["reason"] == "no_intervention"
    print(f"  C. confounded X: interventional, effect {lab['effect_interventional']:.2f} "
          f"(true 1.00) vs pooled slope {lab['effect_observational']:.2f}; lever/door: "
          f"lever interventional (dBIC {ld['lever']['delta_bic']:.0f}), door observational "
          f"(never set by an action)")


# --------------------------------------------------------------------- D
def test_insufficient_evidence():
    dev, _ = make_split_episodes("hidden_switch", "fx:ie", 7, 4)
    rep = search(initial_mechanism("y", "linear", ("x",)), _cat(dev), SearchBudget(wall_s=5))
    assert rep.stopped == "insufficient_validation" and not rep.accepted, rep.stopped
    tr, _ = make_split_episodes("hidden_switch", "fx:ie2", 8, 30)
    root = fit_mechanism(initial_mechanism("y", "linear", ("x",)), _cat(tr[:10]))
    better = search(initial_mechanism("y", "linear", ("x",)), _cat(tr[:20]),
                    SearchBudget(wall_s=5)).best.mechanism
    reg = MechanismRegistry(promote_windows=2, max_inconclusive=6, min_episodes=3)
    reg.install(root)
    assert reg.admit(better, {"val": {"mean": 1.0}})[0]
    states = []
    for w in range(6):                        # windows of 2 episodes: undecidable
        reg.evaluate_window("y", _cat(tr[20 + (w % 5): 22 + (w % 5)]))
        states.append(reg.entry(better.ref()).state)
    assert states[:5] == ["candidate"] * 5 and states[5] == "retired", states
    assert reg.entry(better.ref()).retired_reason == "stale"
    assert reg.incumbent("y").ref() == root.ref()
    ev = reg.consider_revival("y", _cat(tr[20:30]))
    assert ev and ev[0]["key"] == better.ref() and reg.entry(better.ref()).state == "probationary"
    print(f"  D. 4 dev episodes -> {len(rep.val_episodes)} validation: nothing accepted; "
          f"better candidate on 2-episode windows: {states[:5].count('candidate')} windows "
          f"'candidate', then retired('stale'); revived by a 10-episode window "
          f"(t={ev[0]['t']:.1f})")


# --------------------------------------------------------------------- E
def test_regime_change_and_revival():
    scope = "fx:regime"
    A1, _ = make_split_episodes("regime", scope, 11, 20, prefix="a", regime="A")
    B, _ = make_split_episodes("regime", scope, 12, 30, prefix="b", regime="B")
    A2, _ = make_split_episodes("regime", scope, 13, 30, prefix="c", regime="A")
    root = fit_mechanism(initial_mechanism("y", "linear", ("x",)), _cat(A1[:6]))
    root_json = root.to_json()
    reg = MechanismRegistry(promote_windows=2, revival_windows=2)
    reg.install(root)
    rv = ChangeTriggeredReviser(reg, "y", SearchBudget(wall_s=2.0))
    for ep in A1[6:]:
        rv.observe_episode(ep)
    assert rv.alarms == 0 and not rv.searches, (rv.alarms, len(rv.searches))
    t_promo = t_alarm = None
    for i, ep in enumerate(B):
        for e in rv.observe_episode(ep):
            if e["event"] == "promoted" and t_promo is None:
                t_promo = i
            if e["event"] == "alarm" and t_alarm is None:
                t_alarm, kind = i, e["kind"]
    assert rv.alarms > 0 and rv.searches, "the change to B was not acted on"
    inc = reg.incumbent("y")
    ent = reg.incumbent_entry("y")
    op = ent.provenance[0]["proposal"]["op"]
    assert op != REFIT and "h" in inc.dependencies(), (op, inc.dependencies())
    assert reg.entry(root.ref()).state == "retired"
    assert reg.entry(root.ref()).retired_reason == "replaced"
    assert root.ref() in reg.chain(inc.ref())
    n_search = len(rv.searches)
    t_rev = t_back = None
    for i, ep in enumerate(A2):
        for e in rv.observe_episode(ep):
            if e["event"] == "revived" and t_rev is None:
                t_rev = i
            if e["event"] == "promoted" and t_back is None:
                t_back = i
    back = reg.incumbent("y")
    assert back.ref() == root.ref() and back.to_json() == root_json, "not a bit-identical revival"
    assert len(rv.searches) == n_search, "revival should not need a search"
    assert reg.entry(root.ref()).revived
    print(f"  E. A: 0 alarms/0 searches; B: first alarm in B episode {t_alarm} ({kind}), "
          f"{n_search} bounded search(es), {op} winner using {list(inc.dependencies())} "
          f"promoted at B episode {t_promo}; A again: original {root.ref()} revived at "
          f"episode {t_rev}, re-promoted at {t_back}, params bit-identical, 0 new searches")


# --------------------------------------------------------------------- F
def test_proliferation():
    dev, _ = make_split_episodes("lever_door", "fx:pf", 14, 20)
    rep = search(initial_mechanism("y"), _cat(dev), SearchBudget(wall_s=10, max_candidates=25))
    assert rep.evaluated <= 25 and rep.stopped == "candidates", (rep.evaluated, rep.stopped)
    big = search(initial_mechanism("y"), _cat(dev), SearchBudget(wall_s=10))
    root = fit_mechanism(initial_mechanism("y"), _cat(dev))
    reg = MechanismRegistry(max_live=3, max_retired=4)
    reg.install(root)
    peak_live = peak_ret = 0
    outcomes = {}
    for c in big.accepted:
        ok, why = reg.admit(c.mechanism, c.provenance())
        outcomes[why] = outcomes.get(why, 0) + 1
        peak_live = max(peak_live, len(reg.live("y")))
        peak_ret = max(peak_ret, len(reg.retired("y")))
    assert len(big.accepted) >= 8, len(big.accepted)
    assert peak_live <= 3 and peak_ret <= 4, (peak_live, peak_ret)
    print(f"  F. max_candidates 25 -> {rep.evaluated} fitted, stopped '{rep.stopped}'; "
          f"{len(big.accepted)} accepted candidates offered to max_live 3: peak live "
          f"{peak_live}, peak retired {peak_ret} (cap 4), admissions {outcomes}")


# --------------------------------------------------------------------- G
def test_catastrophic_replacement():
    A, _ = make_split_episodes("regime", "fx:cr", 15, 50, regime="A")
    B, _ = make_split_episodes("regime", "fx:cr", 16, 30, prefix="b", regime="B")
    inc = fit_mechanism(initial_mechanism("y", "linear", ("x",)), _cat(A[:10]))
    bad = search(initial_mechanism("y", "linear", ("x",)), _cat(B),
                 SearchBudget(wall_s=5)).best.mechanism
    tb = _cat(B)
    assert bad.bic(tb) < inc.bic(tb), "challenger should look better on its own data"
    reg = MechanismRegistry(promote_windows=2, retire_windows=3)
    reg.install(inc)
    before = inc.to_json()
    reg.admit(bad, {"val": {"mean": 5.0}})
    for w in range(10):
        reg.evaluate_window("y", _cat(A[10 + 4 * w: 14 + 4 * w]))
        assert reg.incumbent("y").ref() == inc.ref()
    e = reg.entry(bad.ref())
    assert e.state == "retired" and e.retired_reason == "disadvantage" and not e.was_promoted
    assert reg.incumbent("y").to_json() == before
    # alternating evidence: never two advantages in a row -> never promoted
    B2, _ = make_split_episodes("regime", "fx:cr", 17, 24, prefix="bb", regime="B")
    reg2 = MechanismRegistry(promote_windows=2, retire_windows=5)
    reg2.install(inc)
    reg2.admit(bad, {"val": {"mean": 5.0}})
    verdicts = []
    for w in range(6):
        win = _cat(B2[4 * (w // 2): 4 * (w // 2) + 4]) if w % 2 == 0 else \
            _cat(A[10 + 4 * w: 14 + 4 * w])
        reg2.evaluate_window("y", win)
        verdicts.append(reg2.entry(bad.ref()).windows[-1]["verdict"][0])
    assert reg2.incumbent("y").ref() == inc.ref() and "".join(verdicts) == "adadad", verdicts
    print(f"  G. challenger with better own-data BIC ({bad.bic(tb):.0f} vs {inc.bic(tb):.0f}) "
          f"but worse on validation: 10 windows, never promoted, retired('disadvantage'); "
          f"alternating windows {''.join(verdicts)}: never promoted; incumbent bit-identical")


# --------------------------------------------------------------------- H
def test_recovery():
    import json
    A, _ = make_split_episodes("regime", "fx:rc", 18, 30, regime="A")
    inc = fit_mechanism(initial_mechanism("y", "linear", ("x",)), _cat(A[:10]))
    reg = MechanismRegistry(retire_windows=1)
    reg.install(inc)
    snap = json.loads(json.dumps(reg.snapshot()))
    pred0 = inc.predict(_cat(A[20:]))[0]
    bad = fit_mechanism(initial_mechanism("y"), _cat(A[:10]))
    reg.admit(bad, {"val": {"mean": 0.1}})
    reg.evaluate_window("y", _cat(A[10:20]))
    assert reg.entry(bad.ref()).state == "retired"
    assert np.array_equal(reg.incumbent("y").predict(_cat(A[20:]))[0], pred0)
    r2 = MechanismRegistry.restore(snap)
    assert r2.snapshot() == snap and r2.incumbent("y").to_json() == inc.to_json()
    assert not any(e.key == bad.ref() for e in r2.entries("y"))
    calls = {"n": 0}

    def flaky(m, t, deadline=None):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("fitter crashed")
        return fit_mechanism(m, t, deadline)

    parent = fit_mechanism(initial_mechanism("y", "linear", ("x",)), _cat(A[:10]))
    pj = parent.to_json()
    rep = search(parent, _cat(A), SearchBudget(wall_s=5), fitter=flaky)
    assert rep.counts.get("fit_exception", 0) > 0 and rep.stopped == "exhausted", rep.counts
    assert parent.to_json() == pj
    print(f"  H. rejected candidate: incumbent predictions bit-identical, snapshot/restore "
          f"exact; crashing fitter: {rep.counts['fit_exception']} exceptions absorbed, search "
          f"'{rep.stopped}', parent unchanged")


# --------------------------------------------------------------------- I
def test_budget_slow_fitter():
    dev, _ = make_split_episodes("hidden_switch", "fx:bs", 19, 20)
    tab = _cat(dev)

    def slow(m, t, deadline=None):
        time.sleep(0.15)
        return fit_mechanism(m, t, deadline)

    def run(wall, fitter):
        def measure():
            t0 = time.monotonic()
            r = search(initial_mechanism("y", "linear", ("x",)), tab,
                       SearchBudget(wall_s=wall), fitter=fitter)
            return r, time.monotonic() - t0
        return measure

    (rep, dt), k1 = _timed(run(0.5, slow), lambda r: r[0].stopped == "wall" and r[1] < 0.6,
                           "0.15 s/fit under wall_s 0.5")
    release = threading.Event()

    def hang(m, t, deadline=None):           # ignores its deadline; released at the end
        release.wait(30)

    try:
        (rep2, dt2), k2 = _timed(run(0.3, hang), lambda r: r[0].stopped == "wall" and
                                 r[1] < 0.4 and r[0].counts["abandoned_timeout"] == 1,
                                 "30 s hang under wall_s 0.3")
    finally:
        release.set()
    print(f"  I. 0.15 s/fit under 0.5 s: returned in {dt:.2f} s after {rep.evaluated} fits "
          f"('wall'; attempt {k1}); 30 s hang under 0.3 s: returned in {dt2:.2f} s, fit "
          f"abandoned (attempt {k2})")


# --------------------------------------------------------------------- J
def _run_reviser(world, pre, post, seed, ops):
    scope = f"fx:gate-{world}-{seed}"
    pre_dev, _ = make_split_episodes(world, scope, 100 + seed, 10, prefix="p", regime=pre)
    post_dev, held = make_split_episodes(world, scope, 200 + seed, 24, 10, prefix="q",
                                         regime=post)
    root = fit_mechanism(initial_mechanism("y", "linear", ("x",)), _cat(pre_dev[:6]))
    reg = MechanismRegistry(promote_windows=2)
    reg.install(root)
    budget = SearchBudget(wall_s=2.0, max_candidates=150)
    rv = ChangeTriggeredReviser(reg, "y", budget, ops=ops)
    for ep in pre_dev[6:] + post_dev:
        rv.observe_episode(ep)
    H = _cat(held)
    used = sum(r.evaluated for r in rv.searches)
    secs = sum(r.elapsed for r in rv.searches)
    return _nll(reg.incumbent("y"), H), _nll(root, H), used, secs, budget


def test_gate():
    rows = []
    for seed in range(3):
        s, root_s, us, ts, budget = _run_reviser("regime", "A", "B", seed, ops=(
            "ADD_DEPENDENCY", "SPLIT_APPLICABILITY", "COMPOSE", "REPLACE_TRANSITION", REFIT))
        p, root_p, up, tp, _ = _run_reviser("regime", "A", "B", seed, ops=(REFIT,))
        cs, _, _, _, _ = _run_reviser("param_change", "A", "A2", seed, ops=(
            "ADD_DEPENDENCY", "SPLIT_APPLICABILITY", "COMPOSE", "REPLACE_TRANSITION", REFIT))
        cp, croot, _, _, _ = _run_reviser("param_change", "A", "A2", seed, ops=(REFIT,))
        rows.append((seed, s, p, root_s, us, ts, up, tp, cs, cp, croot))
        assert s < p - 0.5, (seed, s, p)
        assert cs <= cp + 0.05 and cp < croot, (seed, cs, cp, croot)
    print(f"  J. GATE  budget {budget.to_dict()} per search, both arms; held-out NLL "
          f"(nats/row, 10 final-held-out episodes x 30 rows):")
    print("     seed | structural | param-only | no update | cand/secs (struct, param) || "
          "CONTROL slope change: structural | param-only | no update")
    for seed, s, p, r, us, ts, up, tp, cs, cp, cr in rows:
        print(f"      {seed}   |   {s:7.3f}  |  {p:7.3f}   |  {r:7.3f}  | {us:3d}/{ts:.2f}s, "
              f"{up:3d}/{tp:.2f}s       ||   {cs:7.3f}  |  {cp:7.3f}   |  {cr:7.3f}")
    gap = np.mean([p - s for _, s, p, *_ in rows])
    print(f"     mean structural advantage after the structural change: {gap:.2f} nats/row; "
          f"control gap {np.mean([cp - cs for *_, cs, cp, _ in rows]):+.3f}")


# --------------------------------------------------------------------- K
def _wide_table(n_vars=150, n_eps=600, rows=100, seed=0):
    from developmental_ai.foundation.discovery import is_final_heldout
    eids = [e for e in (f"w{i}" for i in range(4 * n_eps))
            if not is_final_heldout("fx:wide", e)][:n_eps]
    rng = np.random.default_rng(seed)
    n = rows * len(eids)
    cols = {f"v{i}": rng.uniform(-1, 1, n) for i in range(n_vars)}
    cols["y"] = rng.normal(0, 1, n)
    return EvidenceTable(cols, "y", np.repeat(eids, rows), "fx:wide")


def test_budget_wide_table():
    tab = _wide_table()
    wall = 0.25
    budget = SearchBudget(wall_s=wall, max_candidates=10_000, max_bytes=100_000,
                          max_rows=10 ** 7)

    def slow(m, t, deadline=None):
        time.sleep(0.05)
        return fit_mechanism(m, t, deadline)

    out = []
    for name, kw in (("default", {}), ("slow 0.05s/fit", {"fitter": slow})):
        def measure(kw=kw):
            t0 = time.perf_counter()
            rep = search(initial_mechanism("y"), tab, budget, **kw)
            return time.perf_counter() - t0, rep.evaluated, rep.stopped

        # enumeration no longer eats the budget before the 2nd fit (default fitter)
        (dt, ev, st), k = _timed(measure, lambda r, d=not kw: r[0] <= 1.5 * wall and
                                 r[2] == "wall" and (r[1] > 1 or not d), name)
        out.append((name, dt, ev, st, k))
    print(f"  K. 150 vars x {len(tab)} rows, wall_s {wall}: " + "; ".join(
        f"{n} {dt:.3f} s ({dt / wall:.2f}x, {ev} evaluated, '{st}', attempt {k})"
        for n, dt, ev, st, k in out))


# --------------------------------------------------------------------- L
def _mis_eps(seed, n, mis, prefix, rows=20, scope="fx:mis"):
    from developmental_ai.foundation.discovery import is_final_heldout
    rng = np.random.default_rng(seed)
    ids = [e for e in (f"{prefix}{i}" for i in range(4 * n))
           if not is_final_heldout(scope, e)][:n]
    out = []
    for e in ids:
        x, h = rng.uniform(-1, 1, rows), rng.integers(0, 2, rows).astype(float)
        d = rng.uniform(-1, 1, rows)
        y = 1 + 2 * x + mis * (2 * h - 1) + rng.normal(0, 0.2, rows)
        out.append(EvidenceTable({"x": x, "h": h, "d": d, "y": y}, "y", [e] * rows, scope))
    return out


def _mis_run(seed, mis, **kw):
    pre = _mis_eps(100 + seed, 40, 0.0, "p")      # an accurate sd: ratio is ~1.64
    root = fit_mechanism(initial_mechanism("y", "linear", ("x",)), _cat(pre))
    reg = MechanismRegistry(promote_windows=2)
    reg.install(root)
    rv = ChangeTriggeredReviser(reg, "y", SearchBudget(wall_s=2.0, max_candidates=150), **kw)
    evid = []
    for ep in _mis_eps(200 + seed, 60, mis, "q"):
        evid += [e["evidence"] for e in rv.observe_episode(ep) if e["event"] == "misspecified"]
    return rv, "h" in reg.incumbent("y").dependencies(), evid


def test_misspecification_trigger():
    from developmental_ai.foundation.inference import PredictiveCUSUM
    blind = lambda: PredictiveCUSUM(h_mean=1e9, h_disp=1e9)        # change detector off
    trig = uses_h = uses_h_off = live_trig = live_h = 0
    evid = []
    for seed in range(8):
        rv, h, ev = _mis_run(seed, 0.2, detector_factory=blind)
        trig += rv.misspec_triggers > 0
        uses_h += h
        evid += ev
        assert rv.misspecification()["triggers"] == rv.misspec_triggers
        rv_off, h_off, _ = _mis_run(seed, 0.2, detector_factory=blind, misspec_factory=None)
        uses_h_off += h_off
        assert not rv_off.searches
        rv_live, hl, _ = _mis_run(seed, 0.2)
        live_trig += rv_live.misspec_triggers > 0
        live_h += hl
    null_trig = null_search = 0
    for seed in range(8):
        rv, _, _ = _mis_run(seed, 0.0)
        null_trig += rv.misspec_triggers
        null_search += len(rv.searches)
    print(f"  L. misfit +-0.2 by ignored h (variance ratio ~2, mean 0), 8 seeds x 60 episodes, "
          f"CHANGE DETECTOR BLINDED: misspecification opened a revision on {trig}/8 seeds "
          f"(evidence {min(evid) if evid else '-'}-{max(evid) if evid else '-'} rows), incumbent "
          f"uses h on {uses_h}/8 vs {uses_h_off}/8 with the monitor off too; with the detector "
          f"live it uses h on {live_h}/8 and the monitor opened {live_trig}/8 (the fixed CUSUM "
          f"usually fires first); well-specified: {null_trig} triggers, {null_search} searches")
    assert trig >= 7 and uses_h >= 7 and uses_h_off == 0, (trig, uses_h, uses_h_off)
    assert evid and min(evid) > 0
    assert null_trig == 0 and null_search == 0, (null_trig, null_search)


# --------------------------------------------------------------------- P
def _fit_threads():
    return sum(1 for t in threading.enumerate() if t.name == "discovery-fit" and t.is_alive())


def _settle(base, window):
    """Seconds until live fit threads are back to `base` (None: not within window)."""
    t0 = time.perf_counter()
    while True:
        dt = time.perf_counter() - t0
        if _fit_threads() <= base:
            return dt
        if dt > window:
            return None
        time.sleep(0.001)


def test_abandoned_fits():
    dev, _ = make_split_episodes("lever_door", "fx:abandon", 31, 20)
    tab = _cat(dev)
    IT, N = 0.03, 10

    def iterative(m, t, deadline=None):
        out = None
        for _ in range(N):                       # e.g. restarts: refit, keep the last
            time.sleep(IT)
            out = fit_mechanism(m, t, deadline)  # the library form polls the deadline
        return out

    rows = []
    for wall in (0.15, 0.45, 1.1, 1.35):
        def measure(wall=wall):
            base = _fit_threads()
            rep = search(initial_mechanism("y"), tab, SearchBudget(wall_s=wall),
                         fitter=iterative)
            settle = _settle(base, 2 * IT)
            late = _settle(base, 3.0) if settle is None else settle
            return (wall, rep.evaluated, rep.counts.get("abandoned_timeout", 0), settle,
                    late, rep.stopped)
        r, k = _timed(measure, lambda r: r[3] is not None and r[5] == "wall",
                      f"P1 wall_s {wall}: abandoned fit outlived its search by more than "
                      f"2 x {IT * 1e3:.0f} ms (settle, late = items 3, 4)")
        rows.append(r[:5] + (k,))
    print("  P1. cooperative iterative fitter (30 ms/iteration): wall_s -> evaluated, "
          "abandoned, live fit threads back to baseline after: " + "; ".join(
              f"{w} -> {ev}, {ab}, {s * 1e3:.0f} ms (attempt {k})"
              for w, ev, ab, s, _, k in rows))
    assert sum(r[2] for r in rows) >= 1, rows

    release = threading.Event()

    def blocking(m, t, deadline=None):          # never checks: cannot be cancelled
        release.wait(30)
        return fit_mechanism(m, t, deadline)

    # Quiesce first: the cap counts LIVE fit threads, so an earlier contract's
    # abandoned (already-released) fit that exits DURING this check frees a
    # slot and the third search runs ('wall') instead of being refused. That
    # flaked 2/6 runs on a loaded machine — a test-ordering artefact, not the cap.
    assert _settle(0, 5.0) is not None, "earlier fit threads never exited"
    base = _fit_threads()
    reps = [search(initial_mechanism("y"), tab, SearchBudget(wall_s=0.1), fitter=blocking,
                   max_abandoned=2 + base) for _ in range(4)]
    alive = _fit_threads() - base
    try:
        assert alive <= 2, f"{alive} hung fit threads alive (cap 2)"
        assert [r.stopped for r in reps] == ["wall", "wall", "budget_exhausted",
                                             "budget_exhausted"], [r.stopped for r in reps]
        assert [r.evaluated for r in reps[2:]] == [0, 0]
        assert all(r.counts.get("refused_abandoned_cap") == 1 for r in reps[2:])
    finally:
        release.set()
    assert _settle(base, 2.0) is not None, "released fits never exited"
    rep = search(initial_mechanism("y"), tab, SearchBudget(wall_s=2.0), max_abandoned=2 + base)
    assert rep.stopped == "exhausted" and rep.evaluated > 1, (rep.stopped, rep.evaluated)
    print(f"  P2. blocking fitter, cap 2: 4 searches -> stopped {[r.stopped for r in reps]}, "
          f"{alive} hung threads alive (not 4); after release the next search "
          f"'{rep.stopped}' with {rep.evaluated} evaluated (slots free themselves)")


if __name__ == "__main__":
    lever = test_hidden_rules()
    test_spurious()
    test_confounding(lever)
    test_insufficient_evidence()
    test_regime_change_and_revival()
    test_proliferation()
    test_catastrophic_replacement()
    test_recovery()
    test_budget_slow_fitter()
    test_budget_wide_table()
    test_abandoned_fits()
    test_misspecification_trigger()
    test_gate()
    print(f"  ({time.time() - T0:.1f}s)")
    print("[foundation-discovery] ALL PASS")
