"""Empowerment: how many distinct futures are reachable from HERE?

WHY THIS EXISTS.  On a multi-day run the agent twice parked itself in
states from which almost no futures were reachable — once at the bottom
of a self-dug pit, once balanced on a self-built column for six hours.
Nothing in the reward function knew those states were traps, because
every hand-written signal we had was about *content* ("this observation
is novel", "that goal paid off"), never about *option space*.  Any rule
we could have written ("up is good", "don't wall yourself in") would
have been domain knowledge, and this project's whole premise is that
the agent earns its priors instead of inheriting ours.

Empowerment fixes this with zero domain vocabulary: roll the agent's OWN
world model forward from the current state under several random action
sequences and measure how far the imagined futures spread apart.  In an
open state, different actions lead to genuinely different places, so the
imagined latents diverge.  In a trap, every action collapses onto the
same few successors, the rollouts stay bunched, and the score falls —
regardless of what the trap is made of.  The loop turns the (bounded,
normalised) score into a *potential* and rewards its telescoping delta,
so parking in a low-empowerment state is intrinsically unattractive
while no absolute reward is farmable by oscillating (potential-based
shaping is policy-invariant; we relearned the hard way, via a 16:1
reward farm, why raw-score bonuses are not).

DEFENSIVE CONTRACT.  This is monitoring/shaping machinery riding on top
of a live world model whose shapes and device placement change across
architecture versions.  It must NEVER be able to kill a run: any failure
inside :func:`empowerment_score` returns 0.0 and parks the repr on
``empowerment_score.last_error`` for the telemetry layer to surface.  A
0.0 score simply means "no empowerment signal this tick" — the shaping
delta it produces is bounded and the run continues.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List

import torch

__all__ = ["empowerment_score", "EmpowermentPotential"]


def _accepts_knowledge(imagine_fn: Any) -> bool:
    """True when the model's imagine_trajectory has a ``knowledge`` param.

    The project's RSSM takes ``knowledge=None`` (symbolic conditioning
    from the knowledge graph); older forks and test fakes may not.  We
    inspect rather than try/except so a genuine TypeError raised INSIDE
    the rollout is never mistaken for a signature mismatch and silently
    retried against a half-run, contaminated harvest.
    """
    try:
        return "knowledge" in inspect.signature(imagine_fn).parameters
    except (TypeError, ValueError):
        # Uninspectable callables (C extensions, exotic wrappers): the
        # in-repo models all accept the kwarg, so default to passing it.
        return True


def empowerment_score(world_model: Any, start_state: Dict[str, torch.Tensor],
                      action_dim: int, n_seq: int = 8,
                      horizon: int = 8) -> float:
    """Divergence of imagined futures from one state; higher = freer.

    ``start_state`` is a single latent state ``{"h": [1,H], "z": [1,Z]}``.
    Both tensors are tiled to ``n_seq`` rows so one imagine_trajectory
    call rolls all sequences in parallel — n_seq separate calls would
    multiply the wall-clock cost of a signal that runs inside the live
    loop's tick budget.

    HARVESTING TRICK: the world model calls ``policy_fn(latent)`` once
    per imagined step, so a policy that appends every latent it is shown
    to a local list receives the full per-step latent trajectory without
    depending on the model's return format (which has changed shape
    across arch versions; the dream-consolidation code harvests the same
    way for the same reason).  The policy IGNORES the latent it is given
    and emits uniform-random one-hot actions per row — empowerment is a
    property of the state, so the probe must not be biased by whatever
    the current policy happens to prefer.

    Score = mean over feature dims of the std across the ``n_seq``
    rollouts of the LAST harvested latent.  In a trap every action maps
    to the same successor, the rows never separate, and the std is ~0;
    in an open state the random action sequences fan out and the std
    grows with the reachable-set diameter.

    Any exception returns 0.0 with the repr recorded on
    ``empowerment_score.last_error`` (cleared to ``None`` on success) —
    a shaping probe must never be able to crash a multi-day run.
    """
    try:
        harvested: List[torch.Tensor] = []

        def _policy_fn(latent: torch.Tensor) -> torch.Tensor:
            # Keep only a detached copy: the caller runs under no_grad,
            # but a model that leaks grad-enabled tensors must not pin
            # its autograd graph in our list for the rollout's lifetime.
            harvested.append(latent.detach())
            rows = latent.shape[0]
            idx = torch.randint(action_dim, (rows,), device=latent.device)
            one_hot = torch.zeros(rows, action_dim, dtype=torch.float32,
                                  device=latent.device)
            one_hot[torch.arange(rows, device=latent.device), idx] = 1.0
            return one_hot

        start = {k: start_state[k].detach().clone().repeat(n_seq, 1)
                 for k in ("h", "z")}

        imagine = world_model.imagine_trajectory
        # eval() during the probe: dropout noise would inject divergence
        # even in a fully trapped state, silently reporting freedom where
        # there is none.  Restore afterwards — the trainer thread owns
        # the model's mode and we must hand it back exactly as found.
        was_training = bool(getattr(world_model, "training", False))
        if was_training:
            world_model.eval()
        try:
            with torch.no_grad():
                if _accepts_knowledge(imagine):
                    imagine(start, _policy_fn, horizon=horizon,
                            knowledge=None)
                else:
                    imagine(start, _policy_fn, horizon=horizon)
        finally:
            if was_training:
                world_model.train()

        if not harvested:
            raise RuntimeError(
                "imagine_trajectory never called policy_fn "
                f"(horizon={horizon}); no latents harvested")

        last = harvested[-1].detach().cpu().float()   # [n_seq, features]
        score = float(last.std(dim=0).mean().item())
        empowerment_score.last_error = None           # type: ignore[attr-defined]
        return score
    except Exception as exc:                          # noqa: BLE001 — contain ALL
        empowerment_score.last_error = repr(exc)      # type: ignore[attr-defined]
        return 0.0


# The attribute exists from import time so telemetry can read it before
# the first probe without an AttributeError branch of its own.
empowerment_score.last_error = None                   # type: ignore[attr-defined]


class EmpowermentPotential:
    """Turns raw empowerment scores into a bounded potential phi ∈ [0,1].

    Raw scores are in world-model latent units, which drift as the model
    trains (early-run latents are near-random and huge-variance; a
    converged model's are tight).  A fixed normaliser would make the
    shaping term vanish or explode across the run, so we track a
    RUNNING MAX that decays slightly each update before absorbing the
    new score.  The decay matters: without it the max is a latch — one
    early outlier (e.g. a pre-training variance spike) would pin the
    denominator forever and flatten phi to ~0 for the rest of the run.
    This codebase has hit the guard-becomes-latch failure four times;
    the decay is the standing antidote.

    The loop consumes phi as a state potential (reward += phi' - phi),
    so only *changes* in relative empowerment pay out and no absolute
    level can be farmed by sitting still.
    """

    def __init__(self, decay: float = 0.999, mode: str = "relative",
                 baseline_beta: float = 0.002, spread_k: float = 6.0) -> None:
        """
        ``mode="running_max"`` is the original: phi = score / decaying max.
        ``mode="relative"`` (default since 2026-09-04) centres phi on a SLOW
        baseline so that "better than usual" pays. See :meth:`update`.

        ``baseline_beta`` is the EMA rate for that baseline. 0.002 means the
        "usual" is an average over roughly the last ~500 readings; at the
        shipped ``empowerment_interval: 25`` that is ~12,500 env steps, so it
        adapts over hours, not seconds. Deliberately slow: the baseline must
        track a genuinely changed environment (a new biome, a different game)
        WITHOUT chasing the moment-to-moment score, because chasing it is
        exactly what flattened the old normaliser.
        """
        self._decay = float(decay)
        self._running_max = 0.0
        self._mode = str(mode)
        self._beta = float(baseline_beta)
        self._mu: float = 0.0            # slow "usual" empowerment
        self._mad: float = 0.0           # slow mean-absolute-deviation (scale)
        self._k = float(spread_k)
        self._n = 0

    def update(self, score: float) -> float:
        """Fold in a new score; return phi ∈ [0,1].

        WHY THE DEFAULT CHANGED (2026-09-04, measured on a live run).
        The original returns ``score / running_max`` where
        ``running_max = max(running_max*decay, score)`` — the score RE-RAISES
        its own denominator every update. So while empowerment is steady,
        ``running_max == score`` and phi is pinned at 1.0 by construction.
        Measured live: score 0.43 with phi 0.97-1.00 across the whole run, and
        the loop rewards ``phi' - phi``, so a potential at its ceiling paid
        almost nothing — empowerment was 0.7% of all income.
        That made it a TRAP DETECTOR (phi falls when options are lost, and
        recovering pays it back) but never an ACQUISITION drive: there was no
        gradient for gaining options, only for not losing them.

        ``relative`` mode centres phi on a slow baseline instead:

            phi = 0.5 + 0.5 * (score - mu) / (k * mad)

        so phi is 0.5 at "usual", rises above when the agent reaches a
        genuinely more open state, and falls below when it is closing options
        off. Headroom exists in BOTH directions, which is what a telescoping
        potential needs to reward improvement.

        It remains a POTENTIAL, so the policy-invariance argument is
        unchanged: no absolute level is farmable by sitting still, because
        only the delta pays and returning to a state returns the potential.

        `mu`/`mad` adapt slowly (see ``baseline_beta``), so a changed
        environment re-centres what "usual" means rather than leaving the term
        permanently saturated — the property that lets this work in a
        different biome, or a different game, with no retuning.

        NO LATCH. The old decay existed because a single early outlier could
        pin `running_max` forever (the guard-becomes-latch failure, four
        occurrences). `mu` and `mad` are EMAs with no max operator, so no
        single reading can pin them; an outlier is absorbed and forgotten at
        the baseline rate. `running_max` is still maintained in this mode so
        `mode` can be switched back without a cold start.
        """
        if not isinstance(score, (int, float)) or score != score \
                or score in (float("inf"), float("-inf")) or score < 0.0:
            score = 0.0
        score = float(score)
        self._running_max = max(self._running_max * self._decay, score)

        if self._mode != "relative":
            if self._running_max <= 0.0:
                # All-zero history (probe erroring, or a genuinely degenerate
                # model): phi is constant, so the telescoping delta is 0 and
                # the shaping term is inert rather than divide-by-zero dead.
                return 1.0
            return min(1.0, score / self._running_max)

        # ---- relative mode ------------------------------------------------
        self._n += 1
        if self._n == 1:
            # First reading defines "usual"; phi starts NEUTRAL at 0.5, not
            # 1.0. Starting at the ceiling would hand the first few steps a
            # large free negative delta as it settled.
            self._mu = score
            self._mad = 0.0
            return 0.5
        _dev = abs(score - self._mu)
        self._mu += self._beta * (score - self._mu)
        self._mad += self._beta * (_dev - self._mad)
        # SPREAD_K CALIBRATED, NOT GUESSED (2026-09-04). Swept against the
        # scores measured live on the training host (0.337-0.443) plus realistic jitter,
        # scoring each k by how well a genuine improvement (0.43 -> 0.55)
        # stands out from step-to-step noise:
        #     k=2  33% of readings railed at 0/1, signal:noise 0.92x
        #     k=3  25%                            1.22x
        #     k=4  20%                            1.47x
        #     k=6  12%                            1.92x   <- chosen
        #     k=8   8%                            1.82x
        # At k=2 the real improvement was INDISTINGUISHABLE from noise (<1x) —
        # a tighter scale looks more responsive and is strictly worse, because
        # a telescoping potential that swings on noise adds reward variance
        # carrying no information. k=6 keeps phi off the rails most of the
        # time while still using its range for genuinely unusual states.
        _scale = self._k * self._mad
        if _scale <= 1e-9:
            # No variation seen yet (or a genuinely constant score): report
            # neutral rather than dividing by ~0 and producing a huge delta.
            return 0.5
        return max(0.0, min(1.0, 0.5 + 0.5 * (score - self._mu) / _scale))
