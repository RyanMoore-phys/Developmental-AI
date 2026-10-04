"""Fresh vs transferred learning, negative transfer, abstention (Stage 12.3).

transfer_eval(learner_factory, source_env, target_env, budget, seeds)
    source_env / target_env are FACTORIES, callable(seed) -> adapter, so
    every arm of every seed gets its own instance. Per seed:

      1. a source learner trains on the source for budget.source_steps;
      2. a FRESH learner trains on the target for budget.target_steps;
      3. a TRANSFERRED learner (a clone of 1, given the fresh learner's
         exploration seed) trains on the same target episodes for the same
         number of steps. Matched: same target seed, same exploration seed,
         same interaction count — the only difference is the starting state.
      4. before 3, a frozen greedy PROBE of budget.probe_steps measures the
         transferred model's APPLICABILITY on the target:
             coverage  = fraction of probe states the model has seen
             fit       = 1 - mean|TD error| / reward range (clipped [0,1])
             applicability = coverage * fit
         An action space the model cannot act in at all (embodiment change)
         is INAPPLICABLE: applicability 0 and no transferred run.

    decision = "abstain" when mean applicability < abstain_threshold (or the
    transfer is inapplicable): the agent then learns fresh and the
    transferred knowledge is not used. The forced-transfer curve is still
    reported, so a reader can see what abstention avoided.

    negative_transfer: transferred mean reward over the target budget is
    below fresh by more than `margin` on average AND on a majority of seeds.
    positive_transfer: the mirror image. Neither: "neutral".

The probe's interactions are reported (probe_interactions) because they are
real target interactions spent on deciding, not learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence

import numpy as np

from ..contracts import INAPPLICABLE
from .learners import run_learning


@dataclass(frozen=True)
class TransferBudget:
    source_steps: int
    target_steps: int
    probe_steps: int = 40

    def __post_init__(self):
        for k in ("source_steps", "target_steps", "probe_steps"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{k} must be an int >= 1, got {v!r}")


def applicability(model, target_env, seed: int, probe_steps: int) -> Dict[str, Any]:
    """Frozen greedy probe of a transferred model on the target."""
    if not model.compatible(target_env.action_spec()):
        return {"applicability": 0.0, "coverage": INAPPLICABLE,
                "fit": INAPPLICABLE, "reason": "action space incompatible"}
    probe = model.clone()
    out = run_learning(probe, target_env, probe_steps, seed, learn=False)
    coverage = float(out["known"].mean()) if out["known"].size else 0.0
    td = out["td_errors"]
    if td.size == 0:
        return {"applicability": 0.0, "coverage": coverage, "fit": INAPPLICABLE,
                "reason": "no reward observed during the probe"}
    rng = model.r_max - model.r_min if np.isfinite(model.r_max - model.r_min) else 0.0
    scale = max(rng, 1e-9) if rng > 0 else 1.0
    fit = float(np.clip(1.0 - np.mean(np.abs(td)) / scale, 0.0, 1.0))
    return {"applicability": coverage * fit, "coverage": coverage, "fit": fit,
            "reason": ""}


def _mean_reward(r: np.ndarray) -> float:
    return float(np.nanmean(r)) if np.isfinite(r).any() else float("nan")


def transfer_eval(learner_factory: Callable, source_env: Callable,
                  target_env: Callable, budget: TransferBudget,
                  seeds: Sequence[int] = (0, 1, 2), abstain_threshold: float = 0.5,
                  margin: float = 0.05) -> Dict[str, Any]:
    if not isinstance(budget, TransferBudget):
        raise TypeError("budget must be a TransferBudget")
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("need at least one seed")
    per_seed = []
    for s in seeds:
        src = source_env(s)
        src_learner = learner_factory(src.action_spec(), s)
        run_learning(src_learner, src, budget.source_steps, seed=s)
        explore_seed, target_seed = s + 7919, s + 104729
        tgt_aspec = target_env(s).action_spec()
        fresh = learner_factory(tgt_aspec, explore_seed)
        fr = run_learning(fresh, target_env(s), budget.target_steps, seed=target_seed)
        app = applicability(src_learner, target_env(s), s + 15485863, budget.probe_steps)
        row = {"seed": s, "applicability": app,
               "fresh_rewards": fr["rewards"], "fresh_mean": _mean_reward(fr["rewards"]),
               "transfer_rewards": None, "transfer_mean": None, "diff": None}
        if app["coverage"] is not INAPPLICABLE:
            tr_learner = src_learner.clone()
            tr_learner.reseed(explore_seed)
            tr = run_learning(tr_learner, target_env(s), budget.target_steps,
                              seed=target_seed)
            row["transfer_rewards"] = tr["rewards"]
            row["transfer_mean"] = _mean_reward(tr["rewards"])
            row["diff"] = row["transfer_mean"] - row["fresh_mean"]
        per_seed.append(row)
    apps = [r["applicability"]["applicability"] for r in per_seed]
    inapplicable = any(r["applicability"]["coverage"] is INAPPLICABLE for r in per_seed)
    diffs = [r["diff"] for r in per_seed if r["diff"] is not None]
    if diffs:
        md = float(np.mean(diffs))
        neg_frac = float(np.mean([d < 0 for d in diffs]))
        pos_frac = float(np.mean([d > 0 for d in diffs]))
        if md < -margin and neg_frac > 0.5:
            effect = "negative"
        elif md > margin and pos_frac > 0.5:
            effect = "positive"
        else:
            effect = "neutral"
    else:
        md, effect = None, "inapplicable"
    mean_app = float(np.mean(apps))
    decision = "abstain" if (inapplicable or mean_app < abstain_threshold) else "transfer"

    def curve(key):
        rows = [r[key] for r in per_seed if r[key] is not None]
        return np.nanmean(np.stack(rows), axis=0) if rows else None

    fresh_curve, transfer_curve = curve("fresh_rewards"), curve("transfer_rewards")
    return {
        "decision": decision, "effect": effect,
        "negative_transfer": effect == "negative",
        "positive_transfer": effect == "positive",
        "mean_applicability": mean_app, "abstain_threshold": abstain_threshold,
        "mean_diff": md, "margin": margin,
        "fresh_curve": fresh_curve, "transfer_curve": transfer_curve,
        "deployed_curve": transfer_curve if decision == "transfer" else fresh_curve,
        "probe_interactions": budget.probe_steps * len(seeds),
        "abstention_avoided_negative": decision == "abstain" and effect == "negative",
        "per_seed": per_seed, "budget": budget,
    }
