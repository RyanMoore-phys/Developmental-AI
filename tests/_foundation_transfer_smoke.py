"""Stage 12 contracts: adapters, variations, transfer, parity, profiling.

WHAT IS CLAIMED (plan Stage 12; foundation/transfer/)

    The plan's Stage 12 claim is bounded: "documented transfer under
    specified conditions ... The claim remains bounded by tested
    environments." These contracts specify those conditions on fixtures, and
    each is written so that a permissive implementation fails it:

    A. CONFORMANCE ACROSS ADAPTERS. The lever box (discrete and box) and the
       cue-response env — two different nonspatial adapters — and EVERY
       variation wrapper pass the same adapter contract. A wrapper that
       leaks its remapped command into executed_action is caught (the
       suite is not vacuous).
    B. ONE FACTOR PER VARIATION. Under a matched rollout, appearance changes
       only the cue values (invertibly), dynamics only the reward rule,
       controls only which command does what, dropout only availability
       (ABSENT, never 0.0), embodiment only the action space. Everything
       else is bit-identical.
    C. TRANSFER HELPS when nothing changed: on matched target episodes and
       exploration seeds, the transferred learner beats the fresh one, and
       the applicability probe says "transfer".
    D. NEGATIVE TRANSFER IS DETECTED under an action permutation (the
       transferred learner is worse than fresh), and abstention turns the
       deployed curve back into the fresh one.
    E. ABSTENTION FIRES ON LOW APPLICABILITY — permuted controls, rescaled
       appearance (no state is recognised), an added actuator (structurally
       INAPPLICABLE) — and does NOT fire when the environment is unchanged.
    F. THE PARITY HELPER IS NOT A RUBBER STAMP: a loop and a vectorised
       implementation pass; a 1e-4 perturbation, a moved NaN, a dtype-kind
       change, a different input mutation and a mismatched exception are
       each caught.
    G. THE PROFILER RANKS AND COUNTS: a deliberately slow function is the
       top own-time hotspot, and per-element work shows builtin-call
       crossings that grow with N while the batched version's stay flat.

Run: PYTHONPATH=. python tests/_foundation_transfer_smoke.py   (< 60 s)
"""
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from developmental_ai.foundation.adapters import NonspatialAdapter
from developmental_ai.foundation.contracts import ABSENT, UNKNOWN, Action
from developmental_ai.foundation.transfer import (
    AddActuators, AppearanceChange, ChannelDropout, ControlRemap,
    CueResponseAdapter, DynamicsShift, NativeParityError, TabularQLearner,
    TransferBudget, assert_native_parity, assert_suite_conforms,
    profile_callable, run_conformance_suite, transfer_eval)
from developmental_ai.foundation.transfer.variations import _Wrapper

PERM = [1, 2, 3, 4, 5, 0]
BUDGET = TransferBudget(source_steps=400, target_steps=120, probe_steps=40)


def cue(s):
    return CueResponseAdapter(seed=s)


def learner(spec, seed):
    return TabularQLearner(spec, seed=seed)


class LeakyRemap(_Wrapper):
    """BROKEN on purpose: reports the remapped (inner) command as executed."""

    def step(self, action):
        cmd = self.action_spec().check_action(action)
        inner = action.replace(command=(cmd + 1) % self.action_spec().n)
        obs, t, tr, info = self.inner.step(inner)
        return obs, t, tr, info


def test_conformance():
    fs = {
        "lever_discrete": lambda s: NonspatialAdapter(seed=s),
        "lever_box": lambda s: NonspatialAdapter("box", seed=s),
        "cue_response": cue,
        "appearance": lambda s: AppearanceChange(cue(s), "cue", PERM, scale=2.0, offset=0.5),
        "dynamics": lambda s: DynamicsShift(cue(s), shift=3),
        "controls": lambda s: ControlRemap(cue(s), PERM),
        "controls_box": lambda s: ControlRemap(NonspatialAdapter("box", seed=s), [0], signs=[-1]),
        "dropout": lambda s: ChannelDropout(cue(s), ["cue"], 0.3, seed=s),
        "embodiment": lambda s: AddActuators(cue(s), (3,)),
        "embodiment_box": lambda s: AddActuators(NonspatialAdapter("box", seed=s), (2, 2)),
    }
    res = run_conformance_suite(fs, seeds=(0, 1), n_steps=60)
    assert_suite_conforms(res)
    assert all(v["steps"] > 0 for v in res.values())
    bad = run_conformance_suite({"leaky": lambda s: LeakyRemap(cue(s))}, seeds=(0,))
    assert not bad["leaky"]["ok"], "broken wrapper passed conformance"
    assert any("executed_action command differs" in p for p in bad["leaky"]["problems"])
    print(f"  A. {len(fs)} adapters/wrappers conform over 2 seeds; a wrapper "
          f"leaking its remapped command is caught")


def rollout(env, cmds, seed=3):
    """obs values per step (dict channel->value) under a fixed command list."""
    obs = env.reset(seed=seed)
    trace = [{o.channel: o.value for o in obs}]
    for c in cmds:
        h = obs[0]
        a = Action(h.environment, h.stream, h.episode, h.seq,
                   env.action_spec().spec_id, c, UNKNOWN, time.time())
        obs, term, trunc, _ = env.step(a)
        trace.append({o.channel: o.value for o in obs})
        if term or trunc:
            break
    return trace


def same(a, b):
    if a is ABSENT or b is ABSENT:
        return a is b
    return np.array_equal(np.asarray(a), np.asarray(b)) and \
        np.asarray(a).dtype == np.asarray(b).dtype


def test_one_factor_each():
    rng = np.random.default_rng(0)
    cmds = [int(x) for x in rng.integers(6, size=19)]
    base = rollout(cue(0), cmds)
    n = len(base)

    app = AppearanceChange(cue(0), "cue", PERM, scale=2.0, offset=0.5)
    t = rollout(app, cmds)
    assert all(same(app.invert(t[i]["cue"]), base[i]["cue"]) and
               not same(t[i]["cue"], base[i]["cue"]) for i in range(n))
    assert all(same(t[i]["reward"], base[i]["reward"]) and
               same(t[i]["rule"], base[i]["rule"]) for i in range(n))
    assert app.action_spec() == cue(0).action_spec()

    dyn = DynamicsShift(cue(0), shift=4)
    t = rollout(dyn, cmds)
    assert all(same(t[i]["cue"], base[i]["cue"]) for i in range(n))
    assert any(not same(t[i]["reward"], base[i]["reward"]) for i in range(n))
    assert dyn.observation_spec() == cue(0).observation_spec()
    assert dyn.action_spec() == cue(0).action_spec()

    ctl = ControlRemap(cue(0), PERM)
    t = rollout(ctl, [ctl.inverse_command(c) for c in cmds])
    assert all(all(same(t[i][k], base[i][k]) for k in base[i]) for i in range(n)), \
        "inverse-mapped commands did not reproduce the base rollout"
    t2 = rollout(ctl, cmds)
    assert any(not same(t2[i]["reward"], base[i]["reward"]) for i in range(n))
    assert ctl.observation_spec() == cue(0).observation_spec()

    drop = ChannelDropout(cue(0), ["cue"], 0.5, seed=1)
    t = rollout(drop, cmds)
    absent = sum(t[i]["cue"] is ABSENT for i in range(n))
    assert 0 < absent < n
    assert all(t[i]["cue"] is ABSENT or same(t[i]["cue"], base[i]["cue"]) for i in range(n))
    assert all(same(t[i]["reward"], base[i]["reward"]) for i in range(n))
    assert not any(isinstance(t[i]["cue"], np.ndarray) and not t[i]["cue"].any()
                   for i in range(n)), "a dropped reading was written as zeros"

    emb = AddActuators(cue(0), (3,))
    t = rollout(emb, [(c, int(k % 3)) for k, c in enumerate(cmds)])
    assert all(all(same(t[i][k], base[i][k]) for k in base[i]) for i in range(n))
    assert emb.action_spec().shape == (2,) and cue(0).action_spec().shape == ()
    assert emb.observation_spec() == cue(0).observation_spec()
    print(f"  B. over a {n}-step matched rollout each wrapper changed only its "
          f"factor (dropout: {absent} ABSENT, 0 zero-filled)")


def target(wrap):
    return lambda s: wrap(CueResponseAdapter(seed=s + 50))


def test_transfer_helps_when_unchanged():
    r = transfer_eval(learner, cue, target(lambda e: e), BUDGET)
    assert r["decision"] == "transfer", r["mean_applicability"]
    assert r["positive_transfer"], (r["effect"], r["mean_diff"])
    assert all(x["diff"] > 0 for x in r["per_seed"])
    early = slice(0, 30)
    assert np.nanmean(r["transfer_curve"][early]) > np.nanmean(r["fresh_curve"][early])
    print(f"  C. unchanged target: transfer +{r['mean_diff']:.3f} mean reward over "
          f"fresh on all 3 seeds, applicability {r['mean_applicability']:.2f} -> transfer")
    return r


def test_negative_transfer_and_abstention():
    perm = transfer_eval(learner, cue, target(lambda e: ControlRemap(e, PERM)), BUDGET)
    assert perm["negative_transfer"], (perm["effect"], perm["mean_diff"])
    assert perm["decision"] == "abstain" and perm["abstention_avoided_negative"]
    assert np.array_equal(perm["deployed_curve"], perm["fresh_curve"], equal_nan=True)
    print(f"  D. action permutation: negative transfer detected "
          f"({perm['mean_diff']:+.3f}); abstained, deployed curve = fresh")
    cases = {
        "controls": perm,
        "appearance_rescale": transfer_eval(
            learner, cue, target(lambda e: AppearanceChange(e, "cue", scale=2.0)), BUDGET),
        "embodiment": transfer_eval(
            learner, cue, target(lambda e: AddActuators(e, (2,))), BUDGET),
    }
    for k, r in cases.items():
        assert r["decision"] == "abstain", (k, r["mean_applicability"])
        assert r["mean_applicability"] < 0.5
    assert cases["embodiment"]["effect"] == "inapplicable"
    assert cases["embodiment"]["transfer_curve"] is None
    print("  E. abstained on " + ", ".join(
        f"{k} (applicability {r['mean_applicability']:.2f})" for k, r in cases.items())
          + "; did not abstain when unchanged (C)")


def py_softmax_loop(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        e = np.exp(x[i] - x[i].max())
        out[i] = e / e.sum()
    return out


def np_softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_parity_helper():
    rng = np.random.default_rng(0)
    inputs = [rng.normal(size=(8, 5)) for _ in range(6)]
    rep = assert_native_parity(py_softmax_loop, np_softmax, inputs, rtol=1e-12, atol=1e-15)
    assert rep.n_cases == 6 and rep.max_abs_err < 1e-12

    def caught(fn, what, ins=inputs):
        try:
            assert_native_parity(py_softmax_loop if what != "mutation" else mut_py,
                                 fn, ins, rtol=1e-6)
        except NativeParityError:
            return
        raise AssertionError(f"parity helper missed: {what}")

    caught(lambda x: np_softmax(x) * (1 + 1e-4), "perturbation")

    def nan_moved(x):
        y = np_softmax(x)
        y[0, 0] = np.nan
        return y
    caught(nan_moved, "NaN placement")
    caught(lambda x: (np_softmax(x) * 1e6).astype(np.int64), "dtype kind")

    def mut_py(x):
        r = np_softmax(x)
        x[...] = 0.0
        return r
    caught(lambda x: np_softmax(x), "mutation")

    def raiser(x):
        raise FloatingPointError("native overflow")
    caught(raiser, "exception mismatch")
    print(f"  F. loop vs vectorised softmax at parity (max abs err "
          f"{rep.max_abs_err:.1e}, speedup x{rep.speedup:.1f}); perturbation, NaN, "
          f"dtype, mutation and exception mismatches all caught")


def slow_spot(n=20000):
    s = 0.0
    for i in range(n):
        s += i * 0.5
    return s


def fast_spot():
    return sum(range(100))


def workload():
    for _ in range(3):
        slow_spot()
        fast_spot()


def test_profiler():
    rep = profile_callable(workload)
    top = rep["hotspots"][0]["function"]
    assert "slow_spot" in top, f"top hotspot is {top}"
    counts = {}
    for n in (1000, 4000):
        x = np.ones(n)
        loop = profile_callable(lambda: [abs(float(v)) for v in x])["crossings"]
        batch = profile_callable(lambda: float(np.abs(x).sum()))["crossings"]
        counts[n] = (loop["builtin_calls"], batch["builtin_calls"])
    assert counts[4000][0] >= 3.5 * counts[1000][0], f"loop crossings flat? {counts}"
    assert counts[4000][1] == counts[1000][1], f"batched crossings grew: {counts}"
    print(f"  G. top hotspot {top}; builtin crossings loop {counts[1000][0]}->"
          f"{counts[4000][0]} for 4x data, batched {counts[1000][1]}->{counts[4000][1]}")


if __name__ == "__main__":
    t0 = time.time()
    test_conformance()
    test_one_factor_each()
    test_transfer_helps_when_unchanged()
    test_negative_transfer_and_abstention()
    test_parity_helper()
    test_profiler()
    print(f"  ({time.time() - t0:.1f}s)")
    print("[foundation-transfer] ALL PASS")
