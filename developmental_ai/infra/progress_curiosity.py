"""Compression progress: pay for LEARNING, not for surprise.

WHY THIS EXISTS.  The base intrinsic reward was ICM prediction ERROR —
the agent is paid wherever its forward model is currently wrong.  On
the long runs that signal was maximised by anything visually dramatic,
and it funded five distinct reward farms: scenes the model could NEVER
compress (high-motion backdrops, destruction spectacle, flicker) kept
paying forever precisely because they are unlearnable.  This is the
classic noisy-TV pathology: a strobe light is maximally surprising and
zero per cent learnable, so an error-seeker parks in front of it.

The principled replacement is COMPRESSION PROGRESS (Schmidhuber): pay
for the world model measurably IMPROVING — the derivative of competence
— rather than the level of incompetence.  Irreducible noise then pays
nothing (the model never gets better at it), a mastered scene pays
nothing (no room left to improve), and a scene the model is actively
learning pays — exactly the gradient attention should follow.

MEASUREMENT DISCIPLINE.  Improvement is scored on a FIXED probe set,
never on the current stream.  Scoring on whatever the agent happens to
be looking at would re-open the farm (drift toward easy frames and call
the falling loss "progress").  Probes are reservoir-sampled from replay
batches the caller offers, so the set stays representative of lifetime
experience without early experience monopolising it — but it is STABLE
between evaluations: the same exam is re-sat every interval, and only a
reservoir draw ever swaps a single question, never a wholesale rotate
(a rotating exam would let probe-set churn masquerade as learning, in
either direction).

DEFENSIVE CONTRACT.  This is reward machinery riding on a live world
model whose shapes, devices and failure modes change across arch
versions; it must never be able to kill a run.  A probe whose loss call
raises is skipped and counted in ``eval_errors`` (last repr parked on
``last_error`` for telemetry); an evaluation where every probe fails
returns None and pays nothing new; and a dead evaluator's payout decays
to zero within three missed evaluation windows instead of paying a
stale rate forever.  This codebase has hit the guard-becomes-latch
anti-pattern four times; a REWARD that latches is the same disease with
the sign flipped, so staleness decay is built in rather than left to
the caller.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional

import torch

__all__ = ["ProbeSetProgress"]


class ProbeSetProgress:
    """Reward = normalised improvement of world-model loss on fixed probes.

    Lifecycle: the caller offers replay batches to :meth:`maybe_add_probe`
    whenever convenient, calls :meth:`tick` once per environment step, and
    calls :meth:`evaluate` with a loss closure on the same step clock; the
    per-step intrinsic payout is read from :meth:`rate`.  All state is
    plain Python floats plus small detached CPU tensors, so the object is
    checkpoint-friendly and holds no autograd graphs.
    """

    def __init__(self, capacity: int = 12, eval_every: int = 512,
                 ema_beta: float = 0.3, norm_decay: float = 0.999) -> None:
        self._capacity = max(1, int(capacity))
        self._eval_every = max(1, int(eval_every))
        self._ema_beta = float(ema_beta)
        self._norm_decay = float(norm_decay)
        # Private, constant-seeded RNG: the reservoir must not consume
        # from (or be perturbed by) the global RNG stream the trainer
        # seeds, and identical runs should build identical probe sets so
        # progress curves are comparable across reruns.
        self._rng = random.Random(0x5EED)
        self._probes: List[Dict[str, torch.Tensor]] = []
        self._offers = 0
        # EMA of probe-set mean loss.  None until the first evaluation:
        # you cannot have improved before you have a baseline, so the
        # first evaluation only SETS this and pays 0.0.
        self._ema: Optional[float] = None
        # Running-max normaliser, exactly like EmpowermentPotential's:
        # raw progress is in loss units, which shrink by orders of
        # magnitude as the model converges; a fixed scale would make the
        # payout vanish mid-run.  The decay before absorption is the
        # standing latch antidote — one early giant improvement must not
        # pin the denominator and flatten every later payout to ~0.
        self._running_max = 0.0
        self._last_eval_step: Optional[int] = None     # gates due-ness
        self._last_success_step: Optional[int] = None  # gates staleness
        self._step = 0                                 # advanced by tick()
        self._rate = 0.0
        self.last_mean_loss: Optional[float] = None
        self.last_progress = 0.0
        self.eval_errors = 0
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # probe management                                                   #
    # ------------------------------------------------------------------ #
    def maybe_add_probe(self, batch: Dict[str, Any]) -> bool:
        """Offer a replay batch; reservoir-keep at most ``capacity`` probes.

        Algorithm R over the number of offers seen: the first ``capacity``
        offers are kept outright, offer *i* thereafter replaces a random
        held probe with probability capacity/i.  Every batch ever offered
        therefore has EQUAL probability of being in the set — early
        experience cannot monopolise the exam — yet between two
        evaluations with no interleaved keep the set is byte-identical.

        Only the first sequence of the batch (slice ``[0:1]``) is stored,
        as a detached CPU clone per tensor: a probe must not pin a full
        replay batch, a GPU allocation, or an autograd graph for the
        lifetime of the run.  Malformed offers (no tensors, unsliceable
        values) are declined with the error parked on ``last_error`` —
        a reward monitor never crashes the run over a bad batch.
        """
        try:
            probe = {k: v[0:1].detach().cpu().clone()
                     for k, v in batch.items() if torch.is_tensor(v)}
            if not probe:
                return False
            self._offers += 1
            if len(self._probes) < self._capacity:
                self._probes.append(probe)
                return True
            j = self._rng.randrange(self._offers)
            if j < self._capacity:
                self._probes[j] = probe
                return True
            return False
        except Exception as exc:                       # noqa: BLE001
            self.last_error = repr(exc)
            return False

    # ------------------------------------------------------------------ #
    # evaluation                                                         #
    # ------------------------------------------------------------------ #
    def evaluate(self, loss_fn: Callable[[Dict[str, torch.Tensor]], Any],
                 step: int, device: Any = None) -> Optional[float]:
        """Re-sit the fixed exam; return normalised improvement (or None).

        None means "no verdict": not due yet, fewer than 3 probes held
        (a 1-probe exam is pure variance), or every probe's loss call
        failed.  Otherwise the return is progress in [0, 1]-ish units:
        ``max(0, ema_prev - mean_loss)`` run through running-max
        normalisation.  Improvement only — a WORSENING model pays 0.0,
        never negative: intrinsic reward steers attention, and punishing
        the agent for its trainer's bad gradient step would teach it to
        avoid being present when learning happens.

        Per-probe failures are skipped and counted in ``eval_errors``;
        the surviving probes still produce a verdict, because one probe
        with a stale shape must not blind the whole signal.  On an
        all-fail evaluation ``_last_eval_step`` still advances so a
        broken loss_fn is retried once per window, not hammered every
        step — but ``_last_success_step`` does not, so :meth:`rate`
        decays the payout of a dead evaluator to zero.
        """
        try:
            step = int(step)
            if (self._last_eval_step is not None
                    and step - self._last_eval_step < self._eval_every):
                return None
            if len(self._probes) < 3:
                return None

            losses: List[float] = []
            with torch.no_grad():
                for probe in self._probes:
                    try:
                        batch = (probe if device is None else
                                 {k: v.to(device) for k, v in probe.items()})
                        val = float(loss_fn(batch))
                        if not math.isfinite(val):
                            raise ValueError(
                                f"non-finite probe loss {val!r}")
                        losses.append(val)
                    except Exception as exc:           # noqa: BLE001
                        self.eval_errors += 1
                        self.last_error = repr(exc)

            if not losses:
                self._last_eval_step = step
                return None

            mean_loss = sum(losses) / len(losses)
            self._last_eval_step = step
            self._last_success_step = step
            self.last_mean_loss = mean_loss

            if self._ema is None:
                # Baseline-setting evaluation: no prior, no progress.
                self._ema = mean_loss
                self.last_progress = 0.0
                self._rate = 0.0
                return 0.0

            progress_raw = max(0.0, self._ema - mean_loss)
            self._ema = ((1.0 - self._ema_beta) * self._ema
                         + self._ema_beta * mean_loss)
            self._running_max = max(self._running_max * self._norm_decay,
                                    progress_raw)
            out = (progress_raw / self._running_max
                   if self._running_max > 0.0 else 0.0)
            self.last_progress = out
            # Amortise: the improvement was earned over eval_every steps
            # of experience, so it is paid back at out/eval_every per
            # step rather than as a lump the policy could time-farm.
            self._rate = out / self._eval_every
            return out
        except Exception as exc:                       # noqa: BLE001
            self.last_error = repr(exc)
            return None

    # ------------------------------------------------------------------ #
    # per-step payout                                                    #
    # ------------------------------------------------------------------ #
    def tick(self, step: int) -> None:
        """Advance the internal step clock (O(1); call once per env step).

        rate() judges staleness against this clock rather than taking a
        step argument, so a caller that stops evaluating (crashed trainer
        thread, wedged model) still sees the payout die: reward paths
        must degrade to silence, not to a frozen last-known-good value.
        """
        try:
            self._step = int(step)
        except (TypeError, ValueError):
            pass  # a garbage step must not kill the run; clock just holds

    def rate(self) -> float:
        """Current per-step payout; full for one window, then fades to 0.

        Within ``eval_every`` steps of the last successful evaluation the
        amortised rate is paid in full (that IS the interval it was
        earned over).  Beyond that it fades linearly, hitting exactly 0
        at ``3 * eval_every``: an evaluator that has missed three windows
        is dead, and a dead evaluator must not pay forever — the reward
        mirror of the guard-becomes-latch bug this repo has hit four
        times.
        """
        if self._rate <= 0.0 or self._last_success_step is None:
            return 0.0
        age = self._step - self._last_success_step
        if age >= 3 * self._eval_every:
            return 0.0
        if age <= self._eval_every:
            return self._rate
        fade = 1.0 - (age - self._eval_every) / float(2 * self._eval_every)
        return self._rate * fade

    @property
    def stats(self) -> Dict[str, Any]:
        """Telemetry snapshot; safe to call at any time, never raises."""
        return {"probes": len(self._probes),
                "last_mean_loss": self.last_mean_loss,
                "last_progress": self.last_progress,
                "eval_errors": self.eval_errors,
                "rate": self.rate()}
