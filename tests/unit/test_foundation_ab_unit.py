"""Unit cases for developmental_ai.foundation.experiments.ab (plan §7.2).

Parts only: the t table, the paired bootstrap, curve lookup, prereg
validation / hashing / serialisation, episode-level splits and metric
validation. The design claims (tamper refusal, false-positive control,
failure recording, reports) live in tests/_foundation_ab_smoke.py.

Run: PYTHONPATH=. python tests/unit/test_foundation_ab_unit.py
"""
import dataclasses
import sys

sys.path.insert(0, ".")

import numpy as np

from tests.unit._runner import case, run_all, close, raises
from developmental_ai.foundation.experiments import ab


def _p(**kw):
    base = dict(name="u", hypothesis="h", primary_metric="m", direction="higher",
                baseline_arm="A", candidate_arm="B", seeds=(0, 1, 2))
    base.update(kw)
    return ab.Preregistration(**base)


@case
def t_table_known_values_and_monotone():
    close(ab.t_quantile_975(2), 4.303, 1e-9)
    close(ab.t_quantile_975(10), 2.228, 1e-9)
    vals = [ab.t_quantile_975(d) for d in range(1, 400)]
    assert all(a >= b for a, b in zip(vals, vals[1:])), "t quantile not monotone"
    assert 1.96 <= vals[-1] < 1.97
    raises(lambda: ab.t_quantile_975(0), ValueError, "df 0")


@case
def bootstrap_deterministic_and_brackets_mean():
    d = [0.3, -0.1, 0.5, 0.2]
    a = ab.paired_bootstrap_ci(d, 1000, seed=7)
    b = ab.paired_bootstrap_ci(d, 1000, seed=7)
    assert a == b, "bootstrap not deterministic in its seed"
    m, lo, hi = a
    close(m, np.mean(d))
    assert lo < m < hi
    m2, lo2, hi2 = ab.paired_bootstrap_ci(d, 1000, seed=7, small_sample_correction=False)
    assert lo < lo2 and hi > hi2, "small-sample expansion must widen the CI"


@case
def bootstrap_single_pair_has_no_interval():
    m, lo, hi = ab.paired_bootstrap_ci([1.0], 500, 0)
    assert lo is None and hi is None
    raises(lambda: ab.paired_bootstrap_ci([], 500, 0), ValueError, "empty")
    raises(lambda: ab.paired_bootstrap_ci([np.nan, 1.0], 500, 0), ValueError, "nan")


@case
def value_at_is_a_step_function():
    c = ((10, 0.1, 1.0), (20, 0.2, 2.0), (30, 0.3, 3.0))
    assert ab.value_at(c, 0, 5) is None
    assert ab.value_at(c, 0, 10) == 1.0
    assert ab.value_at(c, 0, 25) == 2.0
    assert ab.value_at(c, 1, 0.31) == 3.0


def _run(arm, primary, curve, inter):
    return ab.RunOutcome(arm, 0, "s", "fp", "ok", primary, {}, tuple(curve), inter, None,
                         0.0, None, None, None, 0)


@case
def equal_interactions_on_fixed_data_reads_the_reported_value():
    # Regression (RESULTS_stage6_8.md bug 3, E1c): every point of MLP-128's
    # curve had the SAME interaction count (longer training on fixed data), and
    # equal_interactions read the LAST one — the 480-epoch overfit model (+118)
    # instead of the 60-epoch value the run reports. A run that used exactly the
    # common budget must contribute its primary.
    a = _run("A", 0.62, [(100, 1.0, 0.62), (100, 2.0, 5.0), (100, 8.0, 120.0)], 100)
    b = _run("B", 0.17, [(100, 6.0, 0.17)], 100)
    assert ab._pair_value(a, b, "equal_interactions") == (0.62, 0.17, 100)
    # equal_time is untouched: the common budget is 6 s -> A's 2 s point
    assert ab._pair_value(a, b, "equal_time") == (5.0, 0.17, 6.0)
    # a run that used MORE interactions is truncated by the step function
    c = _run("C", 0.3, [(50, 0.5, 2.0), (100, 1.0, 1.0), (200, 2.0, 0.3)], 200)
    assert ab._pair_value(c, b, "equal_interactions") == (1.0, 0.17, 100)
    # strictly increasing curves: unchanged (primary == last point anyway)
    d = _run("D", 0.4, [(50, 0.5, 0.9), (100, 1.0, 0.4)], 100)
    assert ab._pair_value(d, b, "equal_interactions") == (0.4, 0.17, 100)


@case
def prereg_rejects_malformed():
    E = ab.PreregistrationError
    raises(lambda: _p(direction="up"), E, "direction")
    raises(lambda: _p(basis="vibes"), E, "basis")
    raises(lambda: _p(seeds=()), E, "no seeds")
    raises(lambda: _p(seeds=(1, 1)), E, "dup seeds")
    raises(lambda: _p(seeds=(True,)), E, "bool seed")
    raises(lambda: _p(delta=-1.0), E, "negative delta")
    raises(lambda: _p(delta=float("nan")), E, "nan delta")
    raises(lambda: _p(ablation_arm="A"), E, "role clash")
    raises(lambda: _p(resource_budget={"gpu_hours": 1}), E, "unknown budget key")
    raises(lambda: _p(resource_budget={"max_wall_seconds": 0}), E, "zero budget")
    raises(lambda: _p(hypothesis=" "), E, "blank hypothesis")
    raises(lambda: _p(n_bootstrap=10), E, "tiny bootstrap")


@case
def freeze_hash_covers_every_field():
    p = _p().freeze(clock_ns=lambda: 100)
    p.verify()
    raises(lambda: p.freeze(), ab.PreregistrationError, "double freeze")
    for k, v in (("delta", 0.5), ("hypothesis", "other"), ("seeds", (0, 1, 3)),
                 ("failure_conditions", ("x",)), ("frozen_at_ns", 99)):
        q = dataclasses.replace(p, **{k: v})
        raises(q.verify, ab.PreregistrationViolation, f"edit of {k} undetected")
    raises(_p().verify, ab.PreregistrationViolation, "unfrozen verify")


@case
def prereg_round_trips():
    p = _p(resource_budget={"max_wall_seconds": 3}, failure_conditions=("c",)).freeze()
    q = ab.Preregistration.from_dict(p.to_dict())
    assert q == p and q.frozen_hash == p.frozen_hash
    q.verify()


@case
def make_split_is_episode_level_and_deterministic():
    s1 = ab.make_split("sc", range(50), 0.3)
    s2 = ab.make_split("sc", range(50), 0.3)
    assert s1 == s2 and s1.fingerprint == s2.fingerprint
    assert not set(s1.dev) & set(s1.heldout)
    assert len(s1.dev) + len(s1.heldout) == 50
    raises(lambda: ab.make_split("sc", [1], 0.3), ValueError, "empty side")
    raises(lambda: ab.make_split("sc", [1, 1, 2], 0.3), ValueError, "dup ids")
    raises(lambda: setattr(s1, "dev", ()), dataclasses.FrozenInstanceError, "mutable")


@case
def metrics_validation():
    v = ab.validate_metrics
    raises(lambda: v([1], "m"), ValueError, "non-dict")
    raises(lambda: v({}, "m"), ValueError, "missing primary")
    raises(lambda: v({"m": float("inf")}, "m"), ValueError, "inf primary")
    raises(lambda: v({"m": True}, "m"), ValueError, "bool primary")
    raises(lambda: v({"m": 1.0, "curve": [{"interactions": 2, "seconds": 0, "value": 1},
                                          {"interactions": 1, "seconds": 0, "value": 1}]},
                     "m"), ValueError, "decreasing curve")
    raises(lambda: v({"m": 1.0, "other": object()}, "m"), ValueError, "non-JSON extra")
    pv, curve, inter, upd, other = v({"m": np.float32(2.0), "x": np.int64(3),
                                      "curve": [{"interactions": 4, "seconds": 0.1,
                                                 "value": 1.0}]}, "m")
    assert pv == 2.0 and inter == 4 and upd is None and other == {"x": 3}


if __name__ == "__main__":
    sys.exit(1 if run_all("foundation-ab-unit") else 0)
