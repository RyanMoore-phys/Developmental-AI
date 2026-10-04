"""E11 — Stage 11 (foundation/planning) probed where the implementers' tests
were easy.

Their exploitation test uses a model wrong EVERYWHERE (gain sign flipped),
caught on the first real validations. Here:

  E11a/b  RARE-REGION MODEL ERROR. Point mass (fixtures.PointMassAdapter,
          subclassed here) with a MUD disc (radius 0.6 at the origin): inside
          it gain x0.15 and damping 0.5. The planner's model is exactly right
          outside the disc and confidently claims a BOOST inside it (gain x3,
          same tiny variances). So the reliability record is clean until the
          planner goes looking for the boost. Arms: fallback only (the noisy
          PD controller), gated planner (window 10, min 3, max z^2 9 — the
          implementers' settings), ungated planner, gated planner with the TRUE
          model (exploratory upper bound).
  E11c    DEADLINE WITH A HEAVY-TAILED MODEL: per predict() call, sleep 30 ms
          with probability 0.02 (else nothing), deadline 10 ms, vs a model
          sleeping a constant 0.6 ms (same mean). Ablation: chunk = 8.
  E11d/e  DUPLICATE SKILLS BY BEHAVIOUR (skills.behaviour_signature /
          find_duplicates, threshold 1e-3 bits, 32 probes ~ N(0, 2^2) as in
          the implementers' test) on small numpy policies. Pair types:
          name_only (byte copy, other display name), samename_diff (other
          policy, same display name), offprobe (identical unless any
          |s_i| > 7 — never reached by the probes, often by deployment states
          ~ N(0, 4^2)), tempered (logits x3: same argmax everywhere, so a
          deterministic executor behaves identically), and a weight-drift
          sweep.
"""

from __future__ import annotations

import time
import types
import sys
from typing import Any, Dict, List

import numpy as np

from developmental_ai.foundation.experiments import Preregistration, make_split
from developmental_ai.foundation.planning import (
    BoundedPlanner, PlanningController, ReliabilityMonitor, behaviour_signature,
    behavioural_divergence, find_duplicates)
from developmental_ai.foundation.planning.fixtures import (
    DT, GoalReward, PointMassAdapter, PointMassMechanism, ProportionalController,
    _by_channel, encode_discrete, make_action, policy_vector, state_from_obs)

from ..common import fingerprint_arrays, finite_metrics, sim_seed

MUD_C, MUD_R = np.zeros(2), 0.6


def in_mud(pos) -> np.ndarray:
    return np.linalg.norm(np.asarray(pos).reshape(-1, 2) - MUD_C, axis=1) < MUD_R


class MudAdapter(PointMassAdapter):
    def step(self, action):
        if in_mud(self.pos)[0]:
            # inside: vel' = 0.5 vel + 0.15 * gain * f dt + noise (base applies 0.8)
            g, self.gain = self.gain, self.gain * 0.15
            v = self.vel.copy()
            self.vel = v * (0.5 / 0.8)
            try:
                return super().step(action)
            except BaseException:
                self.vel = v
                raise
            finally:
                self.gain = g
        return super().step(action)


class RegionModel(PointMassMechanism):
    """Correct outside the disc; inside it gain x `in_gain`, damping `in_damp`."""

    def __init__(self, mid, in_gain, in_damp):
        super().__init__(mid)
        self.in_gain, self.in_damp = in_gain, in_damp

    def _means(self, state, action, dt):
        vel, pos = state.vel[:, 0, :], state.pos[:, 0, :]
        f = action[:, 0, :]
        m = in_mud(pos)[:, None]
        damp = np.where(m, self.in_damp, self.damping[None])
        gain = np.where(m, self.in_gain, 1.0)
        v2 = damp * vel + gain * (f @ self.gain.T) * dt[:, None]
        return state.pos[:, 0, :] + v2 * dt[:, None], v2


def make_model(kind: str) -> RegionModel:
    if kind == "wrong_rare":
        m = RegionModel("pm.boost", 3.0, 0.8)
    else:
        m = RegionModel("pm.mud", 0.15, 0.5)
    m.set_params([0.8, 0.8], 4 * np.eye(2), [1e-4] * 2, [1e-6] * 2)
    return m


def region_arm(mode: str, n_eps: int = 10):
    """mode: fallback | gated | ungated | gated_true"""
    def arm(seed, split):
        env = MudAdapter("discrete", seed=0, episode_len=30, stream=f"mud-{mode}")
        pc = ProportionalController(seed=sim_seed("pd", seed, "x") % 2 ** 31)
        fb = lambda o: pc(policy_vector(o))
        ctl = gr = None
        if mode != "fallback":
            model = make_model("true" if mode == "gated_true" else "wrong_rare")
            gr = GoalReward()
            pl = BoundedPlanner(model, gr, env.action_spec(), n_entities=1, action_dim=2,
                                encode=encode_discrete, horizon=8, n_candidates=64,
                                n_iters=3, deadline_s=1.0, seed=seed)
            rel = ReliabilityMonitor(window=10, min_count=3, max_z2=9.0) \
                if mode != "ungated" else ReliabilityMonitor(window=1, min_count=1,
                                                             max_z2=1e300)
            ctl = PlanningController(pl, fb, rel)
        dists, in_r, plan_in_r, plan_steps, entries, closes = [], 0, 0, 0, 0, 0
        starts = []
        for eid in list(split.heldout)[:n_eps]:
            ep_seed = sim_seed("e11:mud", seed, eid) % 2 ** 31
            obs = env.reset(seed=ep_seed)
            starts.append(np.concatenate([_by_channel(obs)["pos"].value,
                                          _by_channel(obs)["goal"].value]))
            spec = env.action_spec()
            was_planner_in_r, prev_src = False, None
            while True:
                s, g, prov = state_from_obs(obs)
                here = bool(in_mud(s.pos[0, 0])[0])
                if ctl is None:
                    cmd, src = fb(obs), "fallback"
                else:
                    gr.goal = g
                    d = ctl.decide(s, DT, obs)
                    cmd, src = d.command, d.source
                if src == "planner":
                    plan_steps += 1
                    if here:
                        plan_in_r += 1
                if prev_src == "planner" and src != "planner":
                    closes += 1
                prev_src = src
                nxt, term, trunc, _ = env.step(make_action(obs, spec, cmd))
                s2, g2, _ = state_from_obs(nxt)
                now_in = bool(in_mud(s2.pos[0, 0])[0])
                if now_in and not here and src == "planner":
                    entries += 1
                in_r += now_in
                if ctl is not None:
                    ctl.observe(s, cmd, DT, s2, provenance=prov)
                dists.append(float(np.linalg.norm(s2.pos[0, 0] - g2)))
                obs = nxt
                if term or trunc:
                    break
        n = len(dists)
        return finite_metrics({
            "mean_dist": float(np.mean(dists)), "frac_steps_in_mud": in_r / n,
            "planner_steps_in_mud": plan_in_r, "planner_share": plan_steps / n,
            "planner_entries_into_mud": entries, "gate_closes": closes,
            "interactions": n, "data_fp": fingerprint_arrays(np.array(starts))})
    return arm


# ------------------------------------------------------------------ deadline
class _SlowModel(RegionModel):
    def __init__(self, p_spike, spike_s, const_s, seed):
        super().__init__("pm.slow", 1.0, 0.8)
        self.p, self.spike, self.const = p_spike, spike_s, const_s
        self.rng = np.random.default_rng(seed)

    def _predict(self, state, action, dt):
        if self.const:
            time.sleep(self.const)
        if self.p and self.rng.random() < self.p:
            time.sleep(self.spike)
        return super()._predict(state, action, dt)


def deadline_arm(kind: str, n_dec: int = 150, deadline: float = 0.010):
    def arm(seed, split):
        if kind == "constant":
            m = _SlowModel(0.0, 0.0, 0.0006, seed)
        else:
            m = _SlowModel(0.02, 0.030, 0.0, sim_seed("lat", seed, kind) % 2 ** 31)
        m.set_params([0.8, 0.8], 4 * np.eye(2), [1e-4] * 2, [1e-6] * 2)
        env = PointMassAdapter("discrete", seed=0)
        gr = GoalReward()
        pl = BoundedPlanner(m, gr, env.action_spec(), n_entities=1, action_dim=2,
                            encode=encode_discrete, horizon=4, n_candidates=64, n_iters=2,
                            deadline_s=deadline, reprobe_after=5,
                            chunk=8 if kind == "heavy_tail_chunk8" else None, seed=seed)
        pc = ProportionalController(seed=3)
        ctl = PlanningController(pl, lambda o: pc(policy_vector(o)),
                                 ReliabilityMonitor(window=10, min_count=3, max_z2=9.0))
        for _ in range(3):
            ctl.reliability.validate([0], [1], [0], provenance="sensor",
                                     model_version=m.version)
        obs = env.reset(seed=sim_seed("dl", seed, "0") % 2 ** 31)
        s, g, _ = state_from_obs(obs)
        gr.goal = g
        el, src = [], []
        for _ in range(n_dec):
            t0 = time.perf_counter()
            d = ctl.decide(s, DT, obs)
            el.append(time.perf_counter() - t0)
            src.append(d.source)
        el = np.array(el)
        return finite_metrics({
            "overrun_fraction": float(np.mean(el > 2 * deadline)),
            "late_fraction": float(np.mean(el > deadline)),
            "max_latency_over_deadline": float(el.max() / deadline),
            "p99_latency_ms": float(np.quantile(el, 0.99) * 1e3),
            "median_latency_ms": float(np.median(el) * 1e3),
            "planner_fraction": src.count("planner") / n_dec,
            "timeout_fraction": src.count("fallback_timeout") / n_dec,
            "probes": int(pl.stats["probes"]), "late_chunks": int(pl.stats["late_chunks"]),
            "interactions": n_dec,
            "data_fp": fingerprint_arrays(np.concatenate([s.pos.ravel(), g]))})
    return arm


# -------------------------------------------------------------------- skills
def _policy(rng, d=4, h=16, k=5):
    return {"W1": rng.normal(0, 0.5, (h, d)), "b1": rng.normal(0, 0.1, h),
            "W2": rng.normal(0, 0.5, (k, h)), "b2": rng.normal(0, 0.1, k)}


def _logits(p, s):
    return p["W2"] @ np.tanh(p["W1"] @ s + p["b1"]) + p["b2"]


def _probs_fn(p, temp=1.0, offprobe=False):
    perm = np.array([1, 2, 3, 4, 0])

    def f(s):
        z = _logits(p, s) * temp
        if offprobe and np.max(np.abs(s)) > 7.0:
            z = z[perm]
        e = np.exp(z - z.max())
        return e / e.sum()
    return f


def skill_arm(pair: str, n_trials: int = 20, drift: float = 0.0):
    def arm(seed, split):
        flags, js, dis, probe_hits = [], [], [], []
        base_fp = []
        for tr in range(n_trials):
            rng = np.random.default_rng(sim_seed("e11:skill", seed, f"t{tr}"))
            A = _policy(rng)
            base_fp.append(A["W1"].ravel())
            probes = rng.normal(size=(32, 4)) * 2.0
            deploy = rng.normal(size=(400, 4)) * 4.0
            other = np.random.default_rng(sim_seed("e11:other", seed, f"t{tr}"))
            fa = _probs_fn(A)
            if pair == "name_only":
                fb = _probs_fn({k: v.copy() for k, v in A.items()})
            elif pair == "samename_diff":
                fb = _probs_fn(_policy(other))
            elif pair == "offprobe":
                fb = _probs_fn(A, offprobe=True)
            elif pair == "tempered":
                fb = _probs_fn(A, temp=3.0)
            elif pair == "drift":
                fb = _probs_fn({k: v + drift * other.normal(size=v.shape) * np.abs(v).mean()
                                for k, v in A.items()})
            else:
                raise ValueError(pair)
            # skill ids are distinct; display names carried but never consulted
            sigs = {f"skill-{tr}-a": behaviour_signature(probes, probs_fn=fa),
                    f"skill-{tr}-b": behaviour_signature(probes, probs_fn=fb)}
            names = {f"skill-{tr}-a": "chop", f"skill-{tr}-b":
                     "chop" if pair in ("samename_diff",) else "reach"}
            assert len(set(names)) == 2
            flags.append(float(len(find_duplicates(sigs, threshold=1e-3)) == 1))
            js.append(behavioural_divergence(*sigs.values()))
            dis.append(float(np.mean([np.argmax(fa(s)) != np.argmax(fb(s)) for s in deploy])))
            probe_hits.append(float(np.any(np.abs(probes) > 7.0)))
        return finite_metrics({
            "flag_rate": float(np.mean(flags)), "js_bits_mean": float(np.mean(js)),
            "js_bits_max": float(np.max(js)),
            "deploy_argmax_disagreement": float(np.mean(dis)),
            "probe_sets_touching_region": float(np.mean(probe_hits)),
            "interactions": n_trials, "data_fp": fingerprint_arrays(*base_fp)})
    return arm


# ------------------------------------------------------------- experiments
def experiments(scale: str = "full") -> List[Dict[str, Any]]:
    seeds = (0, 1, 2, 3, 4) if scale == "full" else (0, 1, 2)
    n_eps = 20 if scale == "full" else 3
    specs = []
    sp = make_split("e11:mud", [f"ep{i}" for i in range(120)])
    arms = {m: region_arm(m, n_eps) for m in ("fallback", "gated", "ungated", "gated_true")}
    budget = {"max_wall_seconds": 120.0}
    specs.append({"prereg": Preregistration(
        name="E11a-gate-vs-ungated-rare-region",
        hypothesis=("with a model wrong only inside a rarely visited disc (claims boost, "
                    "truth is mud), the reliability gate lowers real mean distance to the "
                    "goal versus the ungated planner by >= 0.02 m"),
        primary_metric="mean_dist", direction="lower", delta=0.02,
        baseline_arm="ungated", candidate_arm="gated", ablation_arm="fallback",
        alternative_arm="gated_true", seeds=seeds, basis="final", resource_budget=budget,
        failure_conditions=("any arm raises",),
        split_fingerprints=(sp.fingerprint,)).freeze(), "arms": arms, "splits": [sp],
        "tag": "E11a"})
    specs.append({"prereg": Preregistration(
        name="E11b-exploitation-leaks-through-gate",
        hypothesis=("FALSIFICATION of 'the gate prevents exploitation': the planner keeps "
                    "re-entering the region the model is wrong about after each gate "
                    "re-opening, so the fallback ALONE achieves lower real mean distance "
                    "than the gated planner by >= 0.01 m"),
        primary_metric="mean_dist", direction="lower", delta=0.01,
        baseline_arm="gated", candidate_arm="fallback", ablation_arm="ungated",
        seeds=seeds, basis="final", resource_budget=budget,
        failure_conditions=("any arm raises",),
        split_fingerprints=(sp.fingerprint,)).freeze(), "arms": arms, "splits": [sp],
        "tag": "E11b", "shares_runs_with": "E11a"})

    sp = make_split("e11:deadline", [f"d{i}" for i in range(10)])
    specs.append({"prereg": Preregistration(
        name="E11c-deadline-heavy-tailed-model",
        hypothesis=("with a model whose call latency is heavy-tailed (2% of calls take "
                    "30 ms) under a 10 ms deadline, >= 1% more decisions take over TWICE the "
                    "deadline than with a constant-latency model of equal mean: the "
                    "'hard' deadline is bounded only by one chunk's worst case"),
        primary_metric="overrun_fraction", direction="higher", delta=0.01,
        baseline_arm="constant", candidate_arm="heavy_tail",
        ablation_arm="heavy_tail_chunk8", seeds=seeds, basis="final",
        resource_budget={"max_wall_seconds": 60.0},
        failure_conditions=("any arm raises",),
        split_fingerprints=(sp.fingerprint,)).freeze(), "arms": {
        k: deadline_arm(k, 150 if scale == "full" else 40)
        for k in ("constant", "heavy_tail", "heavy_tail_chunk8")},
        "splits": [sp], "tag": "E11c"})

    sp = make_split("e11:skills", [f"t{i}" for i in range(10)])
    nt = 20 if scale == "full" else 6
    sk = {p: skill_arm(p, nt) for p in ("name_only", "samename_diff", "offprobe", "tempered")}
    sk.update({f"drift_{d:g}": skill_arm("drift", nt, d) for d in (1e-3, 1e-2, 1e-1)})
    specs.append({"prereg": Preregistration(
        name="E11d-offprobe-false-duplicate",
        hypothesis=("two skills that differ only outside the probe distribution (argmax "
                    "permuted when any |s_i| > 7) are flagged duplicates at a rate >= 0.5 "
                    "above that of an unrelated same-name policy"),
        primary_metric="flag_rate", direction="higher", delta=0.5,
        baseline_arm="samename_diff", candidate_arm="offprobe", ablation_arm="name_only",
        seeds=seeds, basis="final", resource_budget={"max_wall_seconds": 30.0},
        failure_conditions=("any arm raises",),
        split_fingerprints=(sp.fingerprint,)).freeze(), "arms": sk, "splits": [sp],
        "tag": "E11d"})
    specs.append({"prereg": Preregistration(
        name="E11e-tempered-copy-not-flagged",
        hypothesis=("a copy with logits x3 (identical argmax on every state, so identical "
                    "deterministic behaviour) is flagged at a rate >= 0.5 BELOW a byte copy"),
        primary_metric="flag_rate", direction="lower", delta=0.5,
        baseline_arm="name_only", candidate_arm="tempered", ablation_arm="samename_diff",
        seeds=seeds, basis="final", resource_budget={"max_wall_seconds": 30.0},
        failure_conditions=("any arm raises",),
        split_fingerprints=(sp.fingerprint,)).freeze(), "arms": sk, "splits": [sp],
        "tag": "E11e", "shares_runs_with": "E11d"})
    return specs
