"""Unit cases for developmental_ai.foundation.discovery (plan Stage 9).

Parts only: table construction (interventions ONLY from agent Action
records, target never intervened, final held-out refused, fit/validation
split disjoint by episode), conditions (NaN satisfies neither P nor NOT P),
form fits recover known parameters, complexity/BIC arithmetic, proposal
validation + JSON round trip + stale-parent refusal + per-op semantics,
composition, contracts.Mechanism round trip, paired verdicts (coverage
cheat refused, one episode is inconclusive), the timeout runner, budget and
registry parameter validation, ablation, cap/state bookkeeping. The design
claims (rule recovery, causal labels, revival, gate) live in
tests/_foundation_discovery_smoke.py.

Run: PYTHONPATH=. python tests/unit/test_foundation_discovery_unit.py
"""
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from tests.unit._runner import case, close, raises, run_all
from developmental_ai.foundation.contracts import Action, from_json, to_json
from developmental_ai.foundation.runtime.splits import SplitLeakError
from developmental_ai.foundation.discovery import (
    ADD_DEPENDENCY, COMPOSE, REFIT, REPLACE_TRANSITION, SPLIT_APPLICABILITY, Branch,
    ComposedMechanism, Condition, DiscoveryError, EvidenceTable, FitError,
    MechanismRegistry, Proposal, ProposalError, SearchBudget, StructuredMechanism, ablate,
    enumerate_proposals, fit_mechanism, initial_mechanism, is_final_heldout,
    mechanism_from_record, per_episode_paired, verdict)
from developmental_ai.foundation.discovery.search import _TIMEOUT, _run_with_timeout
from developmental_ai.foundation.discovery.fixtures import make_episode, make_split_episodes


def _tab(n=200, seed=0, eps=10, scope="u:t", **extra):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    h = rng.integers(0, 2, n).astype(float)
    cols = {"x": x, "h": h, "y": 1 + 2 * x + rng.normal(0, 0.05, n)}
    cols.update(extra)
    ids = [f"e{i % eps}" for i in range(n)]
    return EvidenceTable(cols, "y", ids, scope)


def _timed(measure, ok, what, attempts=3):
    """Wall-clock bound: re-run the measurement up to `attempts` times, pass
    if ANY attempt holds it (a scheduler stall on a loaded runner is not the
    code's fault; a code bug misses every time). Returns (result, attempt)."""
    seen = []
    for k in range(1, attempts + 1):
        r = measure()
        if ok(r):
            if k > 1:
                print(f"  (timing: '{what}' held on attempt {k} of {attempts})")
            return r, k
        seen.append(r)
    raise AssertionError(f"{what}: bound missed on all {attempts} attempts: {seen}")


def _dev_ids(scope, k):
    out, i = [], 0
    while len(out) < k:
        if not is_final_heldout(scope, f"d{i}"):
            out.append(f"d{i}")
        i += 1
    return out


# ---- table ----------------------------------------------------------------------
@case
def table_validation():
    raises(lambda: EvidenceTable({"x": [1.0]}, "y", ["e"], "s"), DiscoveryError, "no target")
    raises(lambda: EvidenceTable({"y": [1.0, 2.0]}, "y", ["e"], "s"), DiscoveryError, "len")
    raises(lambda: EvidenceTable({"y": [np.inf]}, "y", ["e"], "s"), DiscoveryError, "inf")
    raises(lambda: EvidenceTable({"y": [1.0]}, "y", ["e"], ""), DiscoveryError, "scope")
    raises(lambda: EvidenceTable({"y": [1.0]}, "y", ["e"], "s", {"y": [True]}),
           DiscoveryError, "target intervened")


@case
def interventions_only_from_agent_actions():
    rows = [{"episode": "e", "seq": i, "a": float(i), "y": 0.0} for i in range(4)]
    acts = [Action(environment="f", stream="s", episode="e", seq=0, spec_id="x", command=0,
                   duration=0.0, t_dispatch=0.0, payload={"sets": {"a": 0.0}}),
            Action(environment="f", stream="s", episode="e", seq=1, spec_id="x", command=0,
                   duration=0.0, t_dispatch=1.0, payload={"sets": {"a": 1.0},
                                                          "actor": "evaluator"}),
            Action(environment="f", stream="s", episode="e", seq=2, spec_id="x", command=0,
                   duration=0.0, t_dispatch=2.0, payload={"sets": {"a": 99.0}}),
            Action(environment="f", stream="s", episode="e", seq=3, spec_id="x", command=0,
                   duration=0.0, t_dispatch=3.0, payload={"sets": {"y": 0.0}})]
    t = EvidenceTable.from_records(rows, "y", "s", acts)
    # seq 0 agent sets a: yes; seq1 evaluator: no; seq2 value mismatch: no; target: never
    assert t.intervened_mask("a").tolist() == [True, False, False, False]
    assert "y" not in t.intervened
    raises(lambda: EvidenceTable.from_records(rows, "y", "s", ["nope"]), DiscoveryError)
    r2 = [{"episode": "e", "seq": 0, "a": None, "y": 1.0}]
    assert np.isnan(EvidenceTable.from_records(r2, "y", "s").col("a")[0])


@case
def heldout_refused_and_split_disjoint():
    scope = "u:split"
    held = next(f"d{i}" for i in range(1000) if is_final_heldout(scope, f"d{i}"))
    t = EvidenceTable({"y": np.zeros(3)}, "y", [held] * 3, scope)
    raises(t.require_dev, SplitLeakError, "final held-out offered")
    ids = _dev_ids(scope, 40)
    t = EvidenceTable({"y": np.zeros(400)}, "y", [ids[i % 40] for i in range(400)], scope)
    f, v = t.fit_validation_split()
    assert set(f.unique_episodes()).isdisjoint(v.unique_episodes())
    assert len(f) + len(v) == 400 and 0 < len(v.unique_episodes()) < 40
    f2, v2 = t.fit_validation_split()
    assert f2.unique_episodes() == f.unique_episodes(), "split not deterministic"


# ---- conditions / forms -----------------------------------------------------------
@case
def condition_semantics():
    t = EvidenceTable({"a": [0.0, 1.0, np.nan], "y": [0.0, 0.0, 0.0]}, "y", ["e"] * 3, "s")
    c = Condition("a", "<", 0.5)
    assert c.holds(t).tolist() == [True, False, False]
    assert c.negate().holds(t).tolist() == [False, True, False], "NaN must satisfy neither"
    assert Condition.from_dict(c.to_dict()) == c and c.negate().negate() == c
    raises(lambda: Condition("a", "<=", 1), DiscoveryError)
    raises(lambda: Condition("a", "<", np.nan), DiscoveryError)


@case
def branch_validation():
    raises(lambda: Branch((), "constant", ("x",)), DiscoveryError, "constant w/ inputs")
    raises(lambda: Branch((), "piecewise", ("x",)), DiscoveryError, "no pivot")
    raises(lambda: Branch((), "linear", ("x",), "x"), DiscoveryError, "pivot on linear")
    raises(lambda: Branch((), "lookup"), DiscoveryError, "lookup no inputs")
    raises(lambda: Branch((), "cubic"), DiscoveryError, "unknown form")
    raises(lambda: StructuredMechanism("m", "y", [Branch((), "linear", ("y",))]),
           DiscoveryError, "own target as input")


@case
def forms_recover_parameters():
    t = _tab()
    lin = fit_mechanism(initial_mechanism("y", "linear", ("x",)), t)
    close(lin.branches[0].params["w"][0], 2.0, 0.03, "slope")
    close(lin.branches[0].params["b"], 1.0, 0.03, "intercept")
    close(lin.branches[0].sigma, 0.05, 0.01, "sigma")
    con = fit_mechanism(initial_mechanism("y"), t)
    close(con.branches[0].params["b"], t.y.mean(), 1e-12, "const")
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 300)
    tp = EvidenceTable({"p": p, "y": np.where(p < 0.3, -1.0, 2.0) + rng.normal(0, .05, 300)},
                       "y", ["e"] * 300, "s")
    pw = fit_mechanism(StructuredMechanism("m", "y", [Branch((), "piecewise", (), "p")]), tp)
    close(pw.branches[0].params["t"], 0.3, 0.03, "threshold")
    lk = fit_mechanism(StructuredMechanism("m", "y", [Branch((), "lookup", ("h",))]), t)
    assert set(lk.branches[0].params["cells"]) == {"0", "1"}
    unseen = EvidenceTable({"x": [0.0], "h": [5.0], "y": [0.0]}, "y", ["e"], "s")
    mu, sd = lk.predict(unseen)
    assert sd[0] > lk.branches[0].sigma, "unseen lookup key must widen, not pretend"
    raises(lambda: fit_mechanism(StructuredMechanism("m", "y", [Branch((), "lookup", ("x",))]),
                                 t), FitError, "continuous lookup")
    tiny = t.select(np.arange(len(t)) < 4)
    raises(lambda: fit_mechanism(initial_mechanism("y"), tiny), FitError, "too few rows")


@case
def complexity_and_bic():
    t = _tab()
    assert initial_mechanism("y").complexity() == 2
    assert initial_mechanism("y", "linear", ("x", "h")).complexity() == 4
    two = StructuredMechanism("m", "y", [Branch((Condition("h", "==", 1),), "linear", ("x",)),
                                         Branch((Condition("h", "!=", 1),), "constant")])
    assert two.complexity() == 3 + 2 + 1
    f = fit_mechanism(two, t)
    nll, n = f.nll(t)
    close(f.bic(t), 2 * nll + 6 * np.log(n), 1e-9, "BIC")
    assert f.loglik(t).shape == (len(t),)


@case
def fits_are_pure():
    t = _tab()
    m = initial_mechanism("y", "linear", ("x",))
    f = fit_mechanism(m, t)
    assert not m.fitted and f.fitted and f is not m


@case
def missing_rows_not_scored():
    t = _tab()
    cols = dict(t.columns)
    cols["x"] = t.col("x").copy()
    cols["x"][:10] = np.nan
    t2 = EvidenceTable(cols, "y", t.episodes, t.scope)
    f = fit_mechanism(initial_mechanism("y", "linear", ("x",)), t2)
    ll = f.loglik(t2)
    assert np.isnan(ll[:10]).all() and np.isfinite(ll[10:]).all()


# ---- proposals ------------------------------------------------------------------
@case
def proposal_validation_and_roundtrip():
    raises(lambda: Proposal("DELETE", ("m@1",)), ProposalError, "unknown op")
    raises(lambda: Proposal(ADD_DEPENDENCY, ("m@1",), {}), ProposalError, "missing var")
    raises(lambda: Proposal(ADD_DEPENDENCY, ("m",), {"var": "x"}), ProposalError, "no version")
    raises(lambda: Proposal(COMPOSE, ("m@1",)), ProposalError, "compose 1 parent")
    raises(lambda: Proposal(REPLACE_TRANSITION, ("m@1",), {"form": "piecewise"}),
           ProposalError, "piecewise w/o pivot")
    raises(lambda: Proposal(ADD_DEPENDENCY, ("m@1",), {"var": "x", "branch": -1}),
           ProposalError, "neg branch")
    p = Proposal(SPLIT_APPLICABILITY, ("m@1",), {"condition": Condition("h", "==", 1),
                                                 "branch": 0})
    q = Proposal.from_json(p.to_json())
    assert q == p and q.proposal_id == p.proposal_id
    d = p.to_dict()
    d["version"] = 99
    raises(lambda: Proposal.from_dict(d), ProposalError, "future schema")


@case
def proposal_semantics():
    m = initial_mechanism("y", mechanism_id="m")
    pm = {"m": m}
    a = Proposal(ADD_DEPENDENCY, ("m@1",), {"var": "x"}).apply(pm)
    assert a.branches[0].form == "linear" and a.branches[0].inputs == ("x",)
    assert a.parents == ("m@1",) and a.mechanism_id != "m"
    s = Proposal(SPLIT_APPLICABILITY, ("m@1",), {"condition": {"var": "h", "op": "==",
                                                               "value": 1}}).apply(pm)
    t = _tab()
    masks = [b.mask(t) for b in s.branches]
    assert len(masks) == 2 and np.array_equal(masks[0] ^ masks[1], np.ones(len(t), bool))
    r = Proposal(REPLACE_TRANSITION, ("m@1",), {"form": "piecewise", "pivot": "x"}).apply(pm)
    assert r.branches[0].form == "piecewise" and r.branches[0].pivot == "x"
    raises(lambda: Proposal(REPLACE_TRANSITION, ("m@1",), {"form": "lookup"}).apply(pm),
           ProposalError, "lookup without inputs")
    raises(lambda: Proposal(ADD_DEPENDENCY, ("m@2",), {"var": "x"}).apply(pm),
           ProposalError, "stale parent")
    raises(lambda: Proposal(ADD_DEPENDENCY, ("m@1",), {"var": "y"}).apply(pm),
           ProposalError, "own target")
    raises(lambda: Proposal(ADD_DEPENDENCY, ("m@1",), {"var": "x", "branch": 3}).apply(pm),
           ProposalError, "branch range")
    f = Proposal(REFIT, ("m@1",)).apply(pm)
    assert f.mechanism_id == "m" and f.version == 2 and f.parents == ("m@1",)


@case
def compose_predicts_through_unobserved_intermediate():
    dev, _ = make_split_episodes("chain", "u:chain", 0, 10, z_missing=0.3)
    t = EvidenceTable.concat(dev)
    zt = EvidenceTable(t.columns, "z", t.episodes, t.scope)
    m1 = fit_mechanism(initial_mechanism("z", "linear", ("x",), "m1"), zt)
    m2 = fit_mechanism(initial_mechanism("y", "linear", ("z",), "m2"), t)
    c = Proposal(COMPOSE, (m1.ref(), m2.ref())).apply({"m1": m1, "m2": m2})
    assert isinstance(c, ComposedMechanism) and c.dependencies() == ("x",)
    cf = fit_mechanism(c, t)
    p = EvidenceTable({"x": [0.5], "z": [np.nan], "y": [0.0]}, "y", ["e"], "s")
    close(cf.predict(p)[0][0], 3.0, 0.15, "y = 3*(2*0.5)")
    assert np.isnan(m2.predict(p)[0][0]), "m2 alone cannot predict without z"
    raises(lambda: Proposal(COMPOSE, (m2.ref(), m1.ref())).apply({"m1": m1, "m2": m2}),
           ProposalError, "non-chaining compose")


@case
def mechanism_record_roundtrip():
    t = _tab()
    two = StructuredMechanism("m", "y", [Branch((Condition("h", "==", 1),), "linear", ("x",)),
                                         Branch((Condition("h", "!=", 1),), "lookup", ("h",))],
                              causal={"x": {"status": "observational"}})
    f = fit_mechanism(two, t)
    rec = from_json(to_json(f.to_record()))
    g = mechanism_from_record(rec)
    assert np.array_equal(f.predict(t)[0], g.predict(t)[0]) and g.causal == f.causal
    assert rec.inputs == ("h", "x") and rec.outputs == ("y",)
    dev, _ = make_split_episodes("chain", "u:chain2", 1, 6, z_missing=0.0)
    ct = EvidenceTable.concat(dev)
    m1 = initial_mechanism("z", "linear", ("x",), "m1")
    m2 = initial_mechanism("y", "linear", ("z",), "m2")
    c = fit_mechanism(ComposedMechanism("c", m1, m2), ct)
    c2 = mechanism_from_record(from_json(c.to_json()))
    assert np.array_equal(c.predict(ct)[0], c2.predict(ct)[0])


@case
def enumerate_respects_ops_and_target():
    t = _tab()
    m = initial_mechanism("y", "linear", ("x",), "m")
    ps = enumerate_proposals(m, t, ops=(ADD_DEPENDENCY,))
    assert ps and all(p.op == ADD_DEPENDENCY for p in ps)
    assert all(p.args["var"] != "y" for p in ps)
    allp = enumerate_proposals(m, t)
    assert not any(p.op == REFIT for p in allp), "REFIT is the baseline, not a proposal"
    assert {p.op for p in allp} >= {ADD_DEPENDENCY, SPLIT_APPLICABILITY, REPLACE_TRANSITION}


# ---- scoring / verdicts --------------------------------------------------------
@case
def paired_verdicts():
    eps = np.array([f"e{i // 10}" for i in range(50)], dtype=object)
    a = np.full(50, -0.5)
    b = np.full(50, -1.0) + np.linspace(0, 0.01, 50)
    st = per_episode_paired(a, b, eps)
    assert st["n_episodes"] == 5 and verdict(st) == "advantage"
    assert verdict(per_episode_paired(b, a, eps)) == "disadvantage"
    one = per_episode_paired(a[:10], b[:10], eps[:10])
    assert verdict(one) == "inconclusive", "one episode cannot decide"
    cheat = a.copy()
    cheat[:20] = np.nan                                   # declines to predict 40%
    assert verdict(per_episode_paired(cheat, b, eps)) == "disadvantage", "coverage cheat"


@case
def timeout_runner():
    import threading
    release = threading.Event()

    def measure():
        t0 = time.monotonic()
        r = _run_with_timeout(lambda: release.wait(5), 0.05)
        return r, time.monotonic() - t0
    try:
        _timed(measure, lambda r: r[0] is _TIMEOUT and r[1] < 0.5, "timeout runner returns")
    finally:
        release.set()
    assert _run_with_timeout(lambda: 3, 1.0) == 3
    raises(lambda: _run_with_timeout(lambda: 1 / 0, 1.0), ZeroDivisionError)


@case
def abandoned_fit_is_cancelled_and_counted():
    """Known limit closed 2026-10-03: an abandoned fit kept running. The
    runner now cancels the evaluation's CancelToken (still a float deadline,
    so old fitters work) and registers the thread until it exits."""
    import threading
    from developmental_ai.foundation.discovery import (BudgetExceeded, CancelToken,
                                                       check_deadline, live_abandoned_fits)
    tok = CancelToken(time.monotonic() + 60)
    assert isinstance(tok, float) and tok > time.monotonic() and not tok.expired()
    check_deadline(tok)
    check_deadline(None)
    tok.cancel()
    assert tok.cancelled and tok.expired()
    raises(lambda: check_deadline(tok), BudgetExceeded, "cancelled token")
    raises(lambda: check_deadline(time.monotonic() - 1), BudgetExceeded, "past float")
    base = live_abandoned_fits()

    def measure():
        stopped = threading.Event()
        tok2 = CancelToken(time.monotonic() + 60)      # deadline far away: only cancel stops it

        def loop():
            try:
                while True:
                    time.sleep(0.01)
                    check_deadline(tok2)
            finally:
                stopped.set()
        r = _run_with_timeout(loop, 0.03, tok2)
        return r is _TIMEOUT, tok2.cancelled, stopped.wait(0.05)
    _timed(measure, lambda r: all(r), "cooperative fn stops within ~2 iterations of cancel")
    t0 = time.monotonic()
    while live_abandoned_fits() > base and time.monotonic() - t0 < 0.5:
        time.sleep(0.002)
    assert live_abandoned_fits() == base, "dead thread still counted"
    tab = _tab()
    raises(lambda: __import__("developmental_ai.foundation.discovery", fromlist=["search"])
           .search(initial_mechanism("y"), tab, max_abandoned=0), DiscoveryError, "cap >= 1")


@case
def budget_validation():
    raises(lambda: SearchBudget(wall_s=0), DiscoveryError)
    raises(lambda: SearchBudget(wall_s=float("inf")), DiscoveryError)
    raises(lambda: SearchBudget(max_candidates=0), DiscoveryError)
    raises(lambda: SearchBudget(depth=True), DiscoveryError)
    assert SearchBudget().to_dict()["beam"] == 2


# ---- causal ----------------------------------------------------------------------
@case
def ablate_merges_regions():
    two = StructuredMechanism("m", "y", [Branch((Condition("h", "==", 1),), "linear", ("x",)),
                                         Branch((Condition("h", "!=", 1),), "constant")])
    a = ablate(two, "h")
    assert len(a.branches) == 1 and a.branches[0].inputs == ("x",)
    b = ablate(two, "x")
    assert [br.form for br in b.branches] == ["constant", "constant"]
    pw = StructuredMechanism("m", "y", [Branch((), "piecewise", ("x",), "h")])
    assert ablate(pw, "h").branches[0].form == "linear"


# ---- lifecycle -------------------------------------------------------------------
@case
def registry_validation_and_install():
    raises(lambda: MechanismRegistry(max_live=1), DiscoveryError)
    raises(lambda: MechanismRegistry(promote_windows=0), DiscoveryError)
    reg = MechanismRegistry()
    t = _tab()
    raises(lambda: reg.install(initial_mechanism("y")), DiscoveryError, "unfitted install")
    m = fit_mechanism(initial_mechanism("y"), t)
    reg.install(m)
    raises(lambda: reg.install(fit_mechanism(initial_mechanism("y", "linear", ("x",)), t)),
           DiscoveryError, "second incumbent")
    assert reg.admit(m, {})[1] == "duplicate"
    raises(lambda: reg.incumbent("other"), DiscoveryError)


@case
def registry_cap_and_snapshot():
    t = _tab()
    reg = MechanismRegistry(max_live=2, max_retired=1)
    reg.install(fit_mechanism(initial_mechanism("y"), t))
    c1 = fit_mechanism(initial_mechanism("y", "linear", ("x",), "c1"), t)
    c2 = fit_mechanism(initial_mechanism("y", "linear", ("h",), "c2"), t)
    c3 = fit_mechanism(initial_mechanism("y", "linear", ("x", "h"), "c3"), t)
    assert reg.admit(c1, {"val": {"mean": 0.5}}) == (True, "admitted")
    assert reg.admit(c2, {"val": {"mean": 0.1}}) == (False, "cap_full")
    assert reg.admit(c3, {"val": {"mean": 0.9}}) == (True, "admitted")   # evicts c1
    assert reg.entry(c1.ref()).state == "retired"
    assert reg.entry(c1.ref()).retired_reason == "evicted_cap"
    assert len(reg.live("y")) == 2 and len(reg.retired("y")) <= 1
    snap = reg.snapshot()
    r2 = MechanismRegistry.restore(snap)
    assert r2.snapshot() == snap
    raises(lambda: MechanismRegistry.restore({"schema": "x"}), DiscoveryError)


@case
def fixture_routes_interventions():
    ep = make_episode("lever_door", "e0", "u:ld", np.random.default_rng(0), 20)
    assert ep.intervened_mask("lever").all() and not ep.intervened_mask("door").any()
    dev, held = make_split_episodes("regime", "u:r", 0, 5, 2, regime="A")
    assert len(dev) == 5 and len(held) == 2
    assert all(is_final_heldout("u:r", h.unique_episodes()[0]) for h in held)


@case
def table_level_cache_and_lazy_enumeration():
    # verifier B1: levels()/is_discrete() were re-sorted ~3x per variable per
    # enumeration; now cached per table and equal to the uncached answer.
    rng = np.random.default_rng(3)
    n = 300
    t = EvidenceTable({"c": rng.uniform(-1, 1, n), "k": rng.integers(0, 3, n).astype(float),
                       "w": rng.integers(0, 40, n).astype(float), "h": np.full(n, 0.5),
                       "m": np.where(rng.random(n) < 0.5, np.nan, 1.0),
                       "y": rng.normal(size=n)}, "y", ["e"] * n, "u:t")
    for v in ("c", "k", "w", "h", "m"):
        col = t.col(v)
        lv = np.unique(col[np.isfinite(col)])
        assert np.array_equal(t.levels(v), lv) and t.levels(v) is t.levels(v)
        assert t.is_discrete(v) == (0 < len(lv) <= 8 and bool(np.all(lv == np.round(lv)))), v
        for k in (0, 1, 2, 3):
            assert t.n_distinct_at_least(v, k) == (len(lv) >= k), (v, k)
    raises(lambda: t.levels(v).__setitem__(0, 9.0), ValueError, "cached levels read-only")
    from developmental_ai.foundation.discovery.forms import BudgetExceeded
    from developmental_ai.foundation.discovery.search import iter_proposals
    m = fit_mechanism(initial_mechanism("y", "linear", ("c",)), t)
    eager = enumerate_proposals(m, t)
    assert [p.to_dict() for p in iter_proposals(m, t)] == [p.to_dict() for p in eager]
    assert len(eager) > 10
    gen = iter_proposals(m, t, deadline=time.monotonic() - 1.0)
    raises(lambda: list(gen), BudgetExceeded, "past deadline stops enumeration")


if __name__ == "__main__":
    sys.exit(run_all("foundation-discovery-unit"))
