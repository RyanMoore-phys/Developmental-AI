"""The Stage 10 comparison fixture: a hidden mechanism next to a noisy TV.

    DistractedCueAdapter  transfer.CueResponseAdapter (the nonspatial
                          cue -> response box whose hidden `shift` IS the
                          mechanism to identify), plus ONE extra action,
                          "watch_tv", whose outcome is a fair coin on an
                          optional "tv" channel. Watching does not step the
                          inner box (cue unchanged, reward ABSENT — no
                          response was made, so absent, not zero).
    CueShiftHypotheses    the K hypotheses shift = 0..K-1 and the outcome
                          model P(hit | shift, cue, command); hit = reward >
                          0.5, so Gaussian reward noise sigma becomes a slip
                          probability 0.5 * erfc(0.5 / (sigma * sqrt 2)).
                          The TV row is the same coin under every hypothesis.
    Optional, for the verifier's stage9_11 findings (E10c, E10d):
      lamp_bits / lamp   a LEARNABLE BUT USELESS distractor: `lamp_bits`
                         extra actions, lamp j returns bit j of the hidden
                         `lamp` deterministically on a "lamp" channel.
                         CueShiftHypotheses(lamp_bits=...) is then the JOINT
                         set shift x lamp; the question is still the shift.
      response_map       a MISSPECIFIED world: hit iff command ==
                         response_map[cue] (the inner box's reward is
                         remapped per cue), which need not be any shift —
                         the truth is then not in the hypothesis set.
    run_identification    ONE learner, six selectors. Every arm updates the
                          same HypothesisSet through the same real-evidence
                          path (question.ExperimentQuestion.observe ->
                          information.update_from_real_outcome), so arms
                          differ ONLY in which experiment they run:
          info_gain   candidates.enumerate_candidates + ExperimentSelector,
                      info term = TARGETED I(shift; O) from the question
                      (= I(H; O) when the set is shift-only)
          info_joint  ABLATION: I(H; O) over the joint set (the pre-fix
                      behaviour; identical to info_gain without a lamp)
          entropy     ABLATION: the same selector with the info term replaced
                      by the predictive entropy alone (no expected-conditional
                      -entropy subtraction) — the noisy-TV-blind version
          random      uniform over the same candidates
          icm         prediction-error curiosity (the principle of
                      curiosity/icm.py): EMA squared error of a running-mean
                      outcome predictor per (cue, command), optimistic init,
                      argmax
          lp          learning progress (the principle of
                      curiosity/learning_progress.py): drop in mean error
                      between the older and newer half of a key's history
    The truth (`rule`, an EVALUATOR channel) is read only by the metric.

Why re-implementations and not curiosity/icm.py itself: those modules are
torch feature-space models over pixel observations (CNN/MLP encoders,
forward/inverse heads); on a K-symbol fixture they reduce to exactly the
statistics above, and wiring them would compare encoders, not selection
rules. The principle (raw error vs error reduction) is what is compared.

Metric (preregistered in tests/_foundation_selection_smoke.py): interactions
until the posterior mass on the true shift first exceeds 0.9, censored at
`budget + 1` if it never does (`stop="truth"`). `stop="report"` instead runs
until the question REPORTS a resolved conclusion (or the budget) and also
returns the misspecification flag, the reported answer and its timing. Scenarios are the HELD-OUT episodes of the
split; selection never sees a held-out id except as the world it acts in.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..contracts import (ABSENT, Action, ActionSpec, ChannelSpec, ContractError,
                         INAPPLICABLE, Observation, ObservationSpec,
                         RecordValidationError, Sentinel)
from ..inference import HypothesisSet
from ..transfer.envs import CueResponseAdapter
from .ab import DataSplit
from .candidates import enumerate_candidates
from .information import hypothesis_information
from .question import RESOLVED, ExperimentQuestion
from .selection import CostAccount, ExperimentSelector, default_terms, effort_of

ARMS = ("info_gain", "entropy", "random", "icm", "lp", "info_joint")


class DistractedCueAdapter:
    ENVIRONMENT = "nonspatial-cue-response+tv"

    def __init__(self, n_cues: int = 6, shift: int = 1, reward_noise: float = 0.35,
                 max_steps: int = 200, stream: str = "cuetv-0", seed: int = 0,
                 clock: Callable[[], float] = time.time, lamp_bits: int = 0,
                 lamp: int = 0, response_map: Optional[List[int]] = None):
        self.n_cues, self.max_steps = int(n_cues), int(max_steps)
        self.lamp_bits, self.lamp = int(lamp_bits), int(lamp)
        if self.lamp_bits < 0 or not 0 <= self.lamp < 2 ** self.lamp_bits:
            raise ContractError("need lamp_bits >= 0 and 0 <= lamp < 2**lamp_bits")
        self.shift = int(shift) % self.n_cues
        self.response_map = None
        if response_map is not None:
            rm = [int(x) for x in response_map]
            if len(rm) != self.n_cues or any(not 0 <= x < self.n_cues for x in rm):
                raise ContractError("response_map must give one response per cue")
            self.response_map = rm
        self.stream, self._clock = stream, clock
        self.inner = CueResponseAdapter(n_cues=n_cues, shift=shift,
                                        episode_len=max_steps, reward_noise=reward_noise,
                                        stream=stream, seed=seed, clock=clock)
        self._seed = int(seed)
        self._tv_rng = np.random.default_rng(seed + 7_777_777)
        self.TV = self.n_cues
        self._aspec = ActionSpec.discrete(
            "cuetv.act", self.n_cues + 1 + self.lamp_bits, units=("symbol",),
            payload={"labels": [f"respond{k}" for k in range(self.n_cues)] + ["watch_tv"]
                     + [f"lamp{j}" for j in range(self.lamp_bits)]})
        extra = (ChannelSpec("tv", "scalar", (), "float64", "sensor", True,
                             required=False, units="noise", frame=INAPPLICABLE),)
        if self.lamp_bits:
            extra += (ChannelSpec("lamp", "scalar", (), "float64", "sensor", True,
                                  required=False, units="bit", frame=INAPPLICABLE),)
        self._ospec = ObservationSpec(self.ENVIRONMENT, tuple(
            self.inner.observation_spec().channels) + extra)
        self._done, self._seq, self._episode = True, 0, None
        self._inner_obs: List[Observation] = []

    def observation_spec(self) -> ObservationSpec:
        return self._ospec

    def action_spec(self) -> ActionSpec:
        return self._aspec

    def reset(self, seed: Optional[int] = None) -> List[Observation]:
        if seed is not None:
            self._tv_rng = np.random.default_rng(seed + 7_777_777)
        self._inner_obs = self.inner.reset(seed)
        self._episode = self._inner_obs[0].episode
        self._seq, self._done = 0, False
        return self._relabel(self._inner_obs, None)

    def step(self, action: Action):
        if self._done:
            raise ContractError("step() before reset() or after an episode ended")
        if not isinstance(action, Action):
            raise RecordValidationError("step() takes an Action record")
        cmd = self._aspec.check_action(action)
        if (action.environment, action.stream, action.episode, action.seq) != (
                self.ENVIRONMENT, self.stream, self._episode, self._seq):
            raise RecordValidationError(f"stale action for {action.episode!r}@{action.seq}")
        tv = lamp = None
        if cmd < self.n_cues:
            icmd = int(cmd)
            if self.response_map is not None:
                # the inner box rewards (cue + shift); swap that response with
                # the one this world really rewards, so a hit <=> cmd == map[cue]
                cue = self._cue()
                want, tgt = self.response_map[cue], (cue + self.shift) % self.n_cues
                icmd = tgt if icmd == want else (want if icmd == tgt else icmd)
            ia = Action(self.inner.ENVIRONMENT, self.stream, self._episode,
                        self._inner_obs[0].seq, self.inner.action_spec().spec_id,
                        icmd, action.duration, action.t_dispatch)
            self._inner_obs, _, _, _ = self.inner.step(ia)
        elif cmd == self.n_cues:
            tv = np.float64(float(self._tv_rng.integers(2)))
        else:
            lamp = np.float64(float((self.lamp >> (cmd - self.n_cues - 1)) & 1))
        self._seq += 1
        truncated = self._seq >= self.max_steps
        self._done = truncated
        return self._relabel(self._inner_obs, tv, lamp), False, truncated, {
            "client_recovery": False,
            "executed_action": action.replace(t_complete=max(float(self._clock()),
                                                             action.t_dispatch))}

    def _cue(self) -> int:
        for o in self._inner_obs:
            if o.channel == "cue":
                return int(np.argmax(o.value))
        raise ContractError("no cue channel")

    def _relabel(self, inner_obs, tv, lamp=None) -> List[Observation]:
        out = []
        for o in inner_obs:
            o = o.replace(environment=self.ENVIRONMENT, seq=self._seq,
                          t_env=float(self._seq))
            if o.channel == "reward" and (tv is not None or lamp is not None):
                o = o.replace(payload={"value": ABSENT}, missingness={})
            out.append(o)
        for ch, v in (("tv", tv), ("lamp", lamp)):
            if v is not None:
                out.append(out[0].replace(channel=ch, provenance="sensor",
                                          payload={"value": v}, missingness={}))
        return out


def slip_probability(reward_noise: float) -> float:
    if reward_noise <= 0:
        return 1e-3                       # never a hard zero (HypothesisSet refuses -inf)
    return max(1e-3, 0.5 * math.erfc(0.5 / (reward_noise * math.sqrt(2.0))))


class CueShiftHypotheses:
    """shift = 0..K-1, optionally crossed with a `lamp_bits`-bit lamp: the
    JOINT set a learner carries when it models everything it can see. The
    QUESTION is the shift (`answer`); the lamp is a nuisance factor."""

    def __init__(self, n_cues: int, slip: float, lamp_bits: int = 0,
                 lamp_eps: float = 1e-3):
        if not 0 < slip < 0.5:
            raise ContractError("slip must be in (0, 0.5)")
        if int(lamp_bits) < 0 or not 0 < lamp_eps < 0.5:
            raise ContractError("need lamp_bits >= 0 and lamp_eps in (0, 0.5)")
        self.K, self.slip = int(n_cues), float(slip)
        self.lamp_bits, self.lamp_eps = int(lamp_bits), float(lamp_eps)
        D = 2 ** self.lamp_bits
        self.shift_of = np.repeat(np.arange(self.K), D)
        self.lamp_of = np.tile(np.arange(D), self.K)
        self.names = [f"shift={s}" if not self.lamp_bits else f"shift={s},lamp={L}"
                      for s, L in zip(self.shift_of, self.lamp_of)]

    @staticmethod
    def answer(name: str) -> str:
        """The question's answer a hypothesis gives: its shift."""
        return name.split(",")[0]

    def new_set(self, floor: float = 1e-3) -> HypothesisSet:
        hs = HypothesisSet(floor=floor, max_hypotheses=max(16, len(self.names)))
        for n in self.names:
            hs.add_hypothesis(n, prior=None)
        return hs

    def outcome_table(self, cue: int, command: int) -> np.ndarray:
        """(H, 2): columns P(0), P(1). A response: miss/hit; watch_tv: the
        TV's coin (identical rows); lamp j: bit j of the lamp (rows differ
        only by the nuisance factor)."""
        if command < self.K:
            hit = np.where((cue + self.shift_of) % self.K == command,
                           1.0 - self.slip, self.slip)
        elif command == self.K:
            hit = np.full(len(self.names), 0.5)
        else:
            j = int(command) - self.K - 1
            if not 0 <= j < self.lamp_bits:
                raise ContractError(f"command {command} is not a known action")
            hit = np.where((self.lamp_of >> j) & 1 == 1, 1.0 - self.lamp_eps,
                           self.lamp_eps)
        return np.stack([1.0 - hit, hit], 1)

    def logliks(self, cue: int, command: int, outcome: int) -> np.ndarray:
        return np.log(self.outcome_table(cue, command)[:, int(outcome)])


def _reading(obs, channel):
    for o in obs:
        if o.channel == channel and o.provenance == "sensor":
            v = o.value
            return None if isinstance(v, Sentinel) else v
    return None


def _truth(obs) -> int:
    for o in obs:
        if o.channel == "rule":
            return int(o.value)
    raise ContractError("no evaluator rule channel")


class _ErrorStats:
    """Per-key running-mean predictor and its squared-error history."""

    def __init__(self, init_err: float = 0.25, beta: float = 0.3):
        self.mean: Dict[Any, float] = {}
        self.n: Dict[Any, int] = {}
        self.ema: Dict[Any, float] = {}
        self.hist: Dict[Any, List[float]] = {}
        self.init_err, self.beta = init_err, beta

    def update(self, key, y: float) -> None:
        m = self.mean.get(key, 0.5)
        e = (y - m) ** 2
        n = self.n.get(key, 0) + 1
        self.n[key] = n
        self.mean[key] = m + (y - m) / n
        self.ema[key] = (1 - self.beta) * self.ema.get(key, self.init_err) + self.beta * e
        self.hist.setdefault(key, []).append(e)
        del self.hist[key][:-30]

    def error(self, key) -> float:
        return self.ema.get(key, self.init_err)

    def progress(self, key, min_samples: int = 4) -> float:
        h = self.hist.get(key, [])
        if len(h) < min_samples:
            return 0.0
        k = len(h) // 2
        return max(0.0, float(np.mean(h[:k]) - np.mean(h[k:])))


def _key(cue: int, cmd: int, K: int):
    if cmd == K:
        return ("tv",)
    return ("lamp", cmd) if cmd > K else (cue, cmd)


def run_identification(arm: str, shift: int, seed: int, n_cues: int = 6,
                       reward_noise: float = 0.35, budget: int = 60,
                       threshold: float = 0.9, epsilon: float = 0.05,
                       lamp_bits: int = 0, lamp: int = 0,
                       response_map: Optional[List[int]] = None,
                       floor: float = 1e-3, confirm: int = 5,
                       stop: str = "truth") -> Dict[str, Any]:
    """One identification run. `stop="truth"` (the preregistered Stage 10
    metric) ends when the posterior on the true shift first exceeds
    `threshold` (evaluator-side). `stop="report"` ends when the QUESTION
    reports a resolved conclusion (question.ExperimentQuestion: confident AND
    `confirm` surviving checks AND not flagged) or at the budget — the right
    rule when the truth may not be in the set (`response_map`)."""
    if arm not in ARMS:
        raise ContractError(f"unknown arm {arm!r}; known {ARMS}")
    if stop not in ("truth", "report"):
        raise ContractError("stop must be 'truth' or 'report'")
    if stop == "truth" and response_map is not None:
        raise ContractError("a response_map world has no true shift: use stop='report'")
    env = DistractedCueAdapter(n_cues, shift, reward_noise, max_steps=budget + 1,
                               stream=f"cuetv-{arm}", seed=seed, lamp_bits=lamp_bits,
                               lamp=lamp, response_map=response_map)
    spec = env.action_spec()
    model = CueShiftHypotheses(n_cues, slip_probability(reward_noise), lamp_bits)
    hs = model.new_set(floor=floor)
    q = ExperimentQuestion(hs, model.answer, threshold=threshold, confirm=confirm)
    rng = np.random.default_rng(seed + 101)
    terms = default_terms(info_gain=1.0, effort=0.01)
    sel = ExperimentSelector(terms, epsilon=epsilon, seed=seed + 202)
    stats = _ErrorStats()
    acct = CostAccount()
    obs = env.reset(seed)
    truth = None if response_map is not None else f"shift={_truth(obs)}"
    first = conf_t = conf_ans = flag_t = None
    tv_picks = lamp_picks = 0
    nuisance = 0.0
    n_act = len(spec.payload["labels"])
    for t in range(1, budget + 1):
        cue = int(np.argmax(_reading(obs, "cue")))
        cs = enumerate_candidates(spec, durations=(1,), max_candidates=min(64, max(16, n_act)),
                                  max_horizon=1)
        vals, nuis = {}, {}
        for c in cs:
            T = model.outcome_table(cue, c.command)
            if arm == "info_gain":                # TARGETED: I(shift; O)
                hi = q.information(T)
                ig, nuis[c.candidate_id] = hi["info"], hi["nuisance"]
            elif arm == "info_joint":             # ablation: I(H; O) on the joint set
                ig = hypothesis_information(hs, T)["info"]
            elif arm == "entropy":
                ig = hypothesis_information(hs, T)["predictive_entropy"]
            elif arm == "icm":
                ig = stats.error(_key(cue, c.command, n_cues))
            elif arm == "lp":
                ig = stats.progress(_key(cue, c.command, n_cues))
            else:
                ig = 0.0
            vals[c.candidate_id] = {"info_gain": ig, "usefulness": 0.0, "task": 0.0,
                                    "effort": effort_of(c), "risk": 0.0}
        status = q.status()
        if arm == "random":
            chosen = cs.candidates[int(rng.integers(len(cs)))]
            s = sel.select([chosen], {chosen.candidate_id: vals[chosen.candidate_id]},
                           status=status)
        else:
            s = sel.select(cs, vals, status=status)
        nuisance += nuis.get(s.chosen.candidate_id, 0.0)
        cmd = int(s.chosen.command)
        tv_picks += cmd == n_cues
        lamp_picks += cmd > n_cues
        a = Action(env.ENVIRONMENT, env.stream, obs[0].episode, obs[0].seq, spec.spec_id,
                   cmd, 1.0, float(time.time()))
        obs, _, _, info = env.step(a)
        if cmd < n_cues:
            outcome = int(float(_reading(obs, "reward")) > 0.5)
        elif cmd == n_cues:
            outcome = int(float(_reading(obs, "tv")))
        else:
            outcome = int(float(_reading(obs, "lamp")))
        stats.update(_key(cue, cmd, n_cues), float(outcome))
        ev = [o for o in obs if o.channel in ("reward", "tv", "lamp")
              and o.provenance == "sensor"]
        realized = q.observe(model.outcome_table(cue, cmd), outcome,
                             executed_action=info["executed_action"], evidence=ev)
        acct.charge(s, 1, realized)
        st = q.status()
        if flag_t is None and st["misspecified"]:
            flag_t = t
        if conf_t is None and st["p_leading"] > threshold:
            conf_t, conf_ans = t, st["leading"]
        if first is None and truth is not None and q.answer_probs()[truth] > threshold:
            first = t
            if stop == "truth":
                break
        if stop == "report" and st["state"] == RESOLVED:
            break
    st = q.status()
    n = float(acct.interactions)
    reported = st["state"] == RESOLVED
    return {"interactions_to_identify": first if first is not None else budget + 1,
            "censored": first is None, "tv_fraction": tv_picks / n,
            "lamp_fraction": lamp_picks / n,
            "map_correct": truth is not None and st["leading"] == truth,
            "final_state": st["state"], "reported": reported,
            "reported_wrong": reported and st["answer"] != truth,
            "confident": conf_t is not None, "confident_answer": conf_ans,
            "time_to_confidence": conf_t if conf_t is not None else budget + 1,
            "misspec_flagged": flag_t is not None,
            "flag_time": flag_t if flag_t is not None else budget + 1,
            "interactions_to_resolved": q.resolved_at if reported else budget + 1,
            "chosen_nuisance_nats": nuisance, **acct.summary()}


def scenario(scenario_id: str, seed: int, n_cues: int) -> Tuple[int, int]:
    h = hashlib.sha256(f"{scenario_id}\x1f{seed}".encode()).digest()
    return int.from_bytes(h[:4], "big") % n_cues, int.from_bytes(h[4:8], "big") % (2 ** 31)


def make_arm(arm: str, **kw) -> Callable[[int, DataSplit], Dict[str, Any]]:
    n_cues = int(kw.get("n_cues", 6))

    def run(seed: int, split: DataSplit) -> Dict[str, Any]:
        res = []
        for sid in split.heldout:
            shift, s = scenario(sid, seed, n_cues)
            res.append(run_identification(arm, shift, s, **kw))
        x = [r["interactions_to_identify"] for r in res]
        out = {"interactions_to_identify": float(np.mean(x)),
               "per_scenario": x,
               "censored": int(sum(r["censored"] for r in res)),
               "interactions": int(sum(r["interactions"] for r in res))}
        for k in ("tv_fraction", "lamp_fraction", "map_correct",
                  "realized_nats_per_interaction", "misspec_flagged", "reported",
                  "interactions_to_resolved"):
            out[k] = float(np.mean([float(r[k]) for r in res]))
        return out
    return run
