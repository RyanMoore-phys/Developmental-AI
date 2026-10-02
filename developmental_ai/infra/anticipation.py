"""Anticipation — a rise-and-reset intrinsic reward for VLM-grounded
predicates, earned toward whatever caused effect they turn out to precede.

THE SHAPE (specified 2026-09-02, before this module existed)
    The first time the agent sees a predicate (say `tree_visible`) it has
    not yet converted into a caused effect, that predicate carries almost no
    weight. Each further sighting without the effect raises how much weight
    it carries — bounded, never unbounded — and the step the effect finally
    happens, the accumulated weight is PAID OUT and the counter resets to
    near-zero. As the agent gets better at producing that effect ON ITS OWN
    (measured by how often it already happens, not assumed), the whole
    channel's payout for that effect shrinks toward nothing. The VLM stops
    being needed as a guide exactly as it stops being useful.

WHY THIS IS NOT A NEW MECHANISM BOLTED ON FROM NOTHING
    Two pieces already existed and already solve half of this:
      * `infra/consequence.py` (ConsequenceMap) established the channel
        discipline this module inherits: contrastive EMA-learned association
        between a perceptual category and a caused event, defensive
        (never raises), bounded (pruned), and PERSISTED — restart amnesia on
        a paying channel recreates exactly the bootstrap latch that module's
        own docstring names as the reason nine earlier mechanisms died.
      * `infra/stack.py`'s `category_boringness` already computes a
        familiarity/mastery curve, `1 - 1/(1 + n/scale)`, from a lifetime
        event count. This module calls the IDENTICAL formula (see
        `familiarity_curve` in stack.py) rather than inventing a second
        mastery clock that could drift from the first.

WHY THE SIGHTING COUNTER IS THROTTLED, NOT PER-STEP
    A per-step counter would pay MORE the longer the agent stares at
    something — which is the exact shape of every farm this project has
    ever produced (sky 96% of drive, a menu 77% of income, holding attack on
    an unreachable trunk 96% of option activity, 0 logs). `wait[p]` only
    increments once per `sight_throttle_steps`, so continuous presence
    cannot inflate it past one tick per throttle window; only genuinely
    REPEATED encounters (walk away, come back) build anticipation.

ASSOCIATION IS *LIFT*, AND THAT HAS A COUNTERINTUITIVE CONSEQUENCE
    Credit is `max(0, p - base(p))` against the predicate's OWN slow
    baseline, so what is learned is how much MORE present a predicate is
    when the effect fires than it is normally. A predicate that is
    *always* true therefore earns NOTHING, however often the effect
    follows it — that is the anti-farm property, not a defect, and it is
    what stops a stuck-true head output banking credit for every event in
    the world.

    The consequence to know about (found 2026-09-02, when the smoke test
    asserted otherwise and failed on the training host): a predicate held
    CONTINUOUSLY present builds no association, so a very long
    uninterrupted sighting streak erodes the association it is
    accumulating toward. Measured: feeding only present-frames drove
    `assoc` to 0.038 — under every floor — while a realistic base rate
    (absent most of the time, present near the effect) gave 0.749.
    In practice `base_beta` is 0.001 and real viewing is intermittent, so
    this does not bite. Written down because the obvious "fix" — dropping
    the contrastive term — would trade a non-problem for the farm this
    mechanism exists to prevent.

WHY THE PAYOUT NEVER TOUCHES `prim_extrinsic`
    The payout is derived from the grounding head's own inference on the
    agent's OWN latent. `prim_extrinsic` is written to the replay buffer as
    the world model's reward LABEL — training the world model to predict a
    reward that is itself a function of the world model's latent is a closed
    self-feeding loop, the same failure ConsequenceMap's docstring names as
    the reason its deficit is ranking-only. This module's `observe()` return
    value must be added to `intrinsic` and nowhere else.

DEFENSIVE CONTRACT (mirrors infra/consequence.py): never raises, bounded
memory, every accumulator persisted, and it only ever RETURNS a number — the
caller decides what channel to put it in and when to save the state.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from developmental_ai.infra.stack import familiarity_curve


class AnticipationMap:
    """Per-predicate anticipation: rises on repeated sighting, pays out and
    resets on the associated caused effect, damped by earned mastery."""

    def __init__(self, cfg: Optional[dict] = None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.weight = float(cfg.get("weight", 0.5))
        # a predicate counts as "seen" this step at or above this probability
        self.sight_threshold = float(cfg.get("sight_threshold", 0.5))
        # minimum steps between two sightings that both bump `wait` — the
        # anti-farm gate; see module docstring
        self.sight_throttle_steps = int(cfg.get("sight_throttle_steps", 50))
        # ceiling on the anticipation counter (bounded, never unbounded)
        self.wait_cap = int(cfg.get("wait_cap", 8))
        # EMA rate for the predicate<->event association (slower = more
        # conservative about crediting a coincidence)
        self.assoc_beta = float(cfg.get("assoc_beta", 0.01))
        # an association must clear this before it can ever pay
        self.assoc_floor = float(cfg.get("assoc_floor", 0.3))
        # EMA rate for each predicate's own baseline presence (the
        # contrastive term: credit is presence ABOVE this baseline)
        self.base_beta = float(cfg.get("base_beta", 0.001))
        self.max_keys = int(cfg.get("max_keys", 512))

        self._base: Dict[str, float] = {}                    # p -> baseline
        self._assoc: Dict[str, Dict[str, float]] = {}         # p -> {key: a}
        self._wait: Dict[str, int] = {}                       # p -> counter
        self._last_sight_step: Dict[str, int] = {}            # p -> step

        self.last_error: Optional[str] = None
        self.errors: List[str] = []
        # forensics: how much was paid, how many times, and at what ramp —
        # the loop accumulates these into its own segment-log counters, but
        # keeping a lifetime total here means a restart doesn't lose the
        # "has this ever paid anything" question.
        self.total_paid = 0.0
        self.total_payouts = 0

    # ------------------------------------------------------------------ #
    def _note(self, msg: str) -> None:
        self.last_error = str(msg)
        if len(self.errors) < 64:
            self.errors.append(str(msg))

    def _prune(self, d: Dict, cap: int) -> None:
        while len(d) > cap:
            d.pop(next(iter(d)))

    # ------------------------------------------------------------------ #
    def observe(self, probs: Optional[Dict[str, float]],
                events: Optional[List[Tuple[str, str]]],
                event_counts: Optional[Dict[str, int]],
                habituation_scale: float,
                step: int) -> float:
        """One step of evidence. Returns THIS STEP's intrinsic payout.

        `probs`: predicate -> P(true) from the grounding head, THIS step.
        `events`: the adapter's typed CAUSED-effect stream for this step
            (e.g. [("break", "oak_log")]) — pass an empty list on a step
            with no effect, never skip the call, or `wait` sighting-throttle
            timing drifts against the real step clock.
        `event_counts`: lifetime count per `"kind:sub"` key — the SAME
            counts infra/stack.py already threads through for
            `category_boringness`, so mastery is measured once, consistently.
        `habituation_scale`: pass `self._habituation_scale` from the loop —
            the SAME timescale the base-view habituation curve uses.
        """
        if not self.enabled or not probs:
            return 0.0
        try:
            # ---- 1. sighting: bounded, throttled ------------------------
            for p, prob in probs.items():
                prob = float(prob)
                if prob != prob:                      # NaN
                    continue
                b = self._base.get(p, prob)
                self._base[p] = b + self.base_beta * (prob - b)
                if prob < self.sight_threshold:
                    continue
                last = self._last_sight_step.get(p, -10**9)
                if step - last >= self.sight_throttle_steps:
                    self._last_sight_step[p] = step
                    self._wait[p] = min(
                        self.wait_cap, self._wait.get(p, 0) + 1)

            if not events:
                self._prune(self._base, self.max_keys)
                self._prune(self._wait, self.max_keys)
                return 0.0

            # ---- 2. earned association: which predicate precedes which
            #         caused effect? Contrastive against each predicate's
            #         OWN baseline, so an always-on predicate cannot bank
            #         credit for effects it had nothing to do with. -------
            for kind, sub in events:
                key = f"{kind}:{sub}"
                for p, prob in probs.items():
                    prob = float(prob)
                    if prob != prob or prob < self.sight_threshold:
                        continue
                    credit = max(0.0, prob - self._base.get(p, 0.0))
                    if credit <= 0.0:
                        continue
                    d = self._assoc.setdefault(p, {})
                    d[key] = (1 - self.assoc_beta) * d.get(key, 0.0) \
                        + self.assoc_beta * credit
                    self._prune(d, self.max_keys)

            # ---- 3. payout on the step the effect actually happens ------
            total = 0.0
            counts = event_counts or {}
            for kind, sub in events:
                key = f"{kind}:{sub}"
                n = float(counts.get(key, 0))
                familiarity = familiarity_curve(n, habituation_scale)
                for p, assoc_by_key in list(self._assoc.items()):
                    a = assoc_by_key.get(key, 0.0)
                    w = self._wait.get(p, 0)
                    if a < self.assoc_floor or w <= 0:
                        continue
                    ramp = w / float(self.wait_cap)
                    payout = self.weight * a * ramp * (1.0 - familiarity)
                    if payout > 0.0:
                        total += payout
                        self.total_paid += payout
                        self.total_payouts += 1
                    # RESET: "that counter goes back down" — every time the
                    # effect fires, whether or not this predicate's own
                    # association was strong enough to pay, because the
                    # agent DID just get the effect and the wait it was
                    # accumulating toward is now stale evidence either way.
                    self._wait[p] = 0

            self._prune(self._base, self.max_keys)
            self._prune(self._wait, self.max_keys)
            return float(total)
        except Exception as exc:                          # pragma: no cover
            self._note(f"observe failed: {exc!r}")
            return 0.0

    # ------------------------------------------------------------------ #
    def state(self) -> Dict:
        return {
            "base": dict(self._base),
            "assoc": {p: dict(d) for p, d in self._assoc.items()},
            "wait": dict(self._wait),
            "last_sight_step": dict(self._last_sight_step),
            "total_paid": float(self.total_paid),
            "total_payouts": int(self.total_payouts),
        }

    def load_state(self, st: Dict) -> bool:
        try:
            self._base = {str(k): float(v)
                          for k, v in (st.get("base") or {}).items()}
            self._assoc = {str(p): {str(k): float(v) for k, v in d.items()}
                          for p, d in (st.get("assoc") or {}).items()}
            self._wait = {str(k): int(v)
                         for k, v in (st.get("wait") or {}).items()}
            self._last_sight_step = {
                str(k): int(v)
                for k, v in (st.get("last_sight_step") or {}).items()}
            self.total_paid = float(st.get("total_paid", 0.0))
            self.total_payouts = int(st.get("total_payouts", 0))
            return True
        except Exception as exc:
            self._note(f"load_state failed: {exc!r}")
            return False

    def save(self, path: str) -> bool:
        try:
            tmp = str(path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.state(), f)
            os.replace(tmp, path)
            return True
        except Exception as exc:
            self._note(f"save failed: {exc!r}")
            return False

    def load(self, path: str) -> bool:
        try:
            if not os.path.exists(path):
                return False
            with open(path) as f:
                return self.load_state(json.load(f))
        except Exception as exc:
            self._note(f"load failed: {exc!r}")
            return False
