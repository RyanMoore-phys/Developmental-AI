"""Imagination-derived curiosity — the drive that does not die.

MOTIVATION (2026-07-25). Learning-Progress curiosity is a DERIVATIVE of
prediction error, so once the world model masters a stationary world it goes
to zero *by definition* and the organism has no drive left (observed: 141
consecutive segments at exactly 0.00 reward after WM loss fell 1.26 -> 0.058).
A lifelong agent living in ONE world will always reach that point.

This module adds two intrinsic drives sourced from the agent's own world
model, so the signal never flatlines:

  1. DISAGREEMENT (durable, never dies). Imagine K rollouts from the CURRENT
     latent and measure how much they disagree. High spread = "my model does
     not know what happens here" = genuinely worth visiting. This is
     Plan2Explore-style epistemic drive. It cannot be wireheaded: imagining a
     jackpot pays nothing, only reaching genuinely-uncertain states pays, and
     once the agent goes there and learns, that spot stops paying (it
     self-extinguishes LOCALLY but never GLOBALLY, as long as anything in the
     world remains unmodelled).

  2. IMAGINED REWARD (small, DECAYING — "daydreaming"). A bounded bonus when
     the world model imagines high-payoff futures from here. This is the
     user's requested "smaller reward for imagining scenarios that would have
     given a stronger reward". It decays with wall-clock experience so it can
     never become the point of living.

BOREDOM GATE: both terms are scaled by how dead EXTERNAL curiosity is
(`boredom = 1 - min(1, global_lp / lp_reference)`). While the world is still
teaching the agent things, imagination contributes ~nothing; it takes over
only when the environment has gone quiet. That is the intended ordering — real
novelty first, mind-wandering when bored.

WIREHEADING SAFETY (load-bearing — do not "simplify" this):
these bonuses go into the INTRINSIC channel ONLY. In this codebase the replay
buffer's reward LABEL is `prim_extrinsic` (extrinsic + vision shaping), and
the WM's reward head trains on that label. Intrinsic reward never reaches it.
So the agent's daydreams can NEVER train the model that generates the
daydreams — the self-reinforcing loop is structurally impossible. If anyone
ever routes these into `prim_extrinsic`/`shaped_extrinsic`, that loop closes
and the agent can learn to hallucinate its own reward.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ImaginationCuriosity:
    """Intrinsic drive computed by rolling the world model forward."""

    def __init__(
        self,
        enabled: bool = False,
        every: int = 50,             # steps between imagination probes
        rollouts: int = 8,           # K parallel imagined futures
        horizon: int = 10,           # H latent steps each
        disagreement_weight: float = 0.30,
        imagined_reward_weight: float = 0.10,
        imagined_reward_halflife: int = 300_000,   # steps; "decreases"
        lp_reference: float = 0.05,  # global_lp at which boredom = 0
        max_bonus: float = 0.5,      # hard cap on the TOTAL intrinsic add
        # ---- PAST-STATE BASELINE (2026-09-25) ---------------------------
        # See the block above self._past below for why these exist.
        past_states: int = 32,       # ring buffer of remembered latents
        baseline_every: int = 10,    # recompute the baseline every N probes
        baseline_samples: int = 4,   # past states imagined per recompute
    ):
        self.enabled = bool(enabled)
        self.every = max(1, int(every))
        self.rollouts = max(2, int(rollouts))     # need >=2 to disagree
        self.horizon = max(1, int(horizon))
        self.disagreement_weight = float(disagreement_weight)
        self.imagined_reward_weight = float(imagined_reward_weight)
        self.imagined_reward_halflife = max(1, int(imagined_reward_halflife))
        self.lp_reference = max(1e-6, float(lp_reference))
        self.max_bonus = float(max_bonus)
        # running scale so disagreement is comparable across training (the raw
        # latent variance shrinks as the WM sharpens; a fixed threshold would
        # repeat the eps_abs mistake that latched the vision magnet off)
        self._dis_scale = 1e-6
        # ---- "HERE" IS JUDGED AGAINST "WHERE I HAVE BEEN" (2026-09-25) ----
        # WHY: both SkyBots dug into a cave and stopped acting. 196k steps
        # since a block broke, and 69% of ALL income was this module. The
        # bonus is computed from the CURRENT state and needs no action, so
        # standing still somewhere unpredictable paid every step -- and it is
        # gated on learning progress being dead, so acting would have switched
        # the income off. Structurally CLAUDE.md 9 again (sky 96%, menu 77%).
        #
        # The docstring at the top of this file already promised the cure:
        # "once the agent goes there and learns, that spot stops paying (it
        # self-extinguishes LOCALLY but never GLOBALLY)". That was FALSE as
        # implemented -- see the normalisation note in bonus().
        #
        # The fix: remember past latents and ask "is here more uncertain than
        # I am USED TO?" rather than "is here uncertain?". A learned cave
        # converges to the baseline and stops paying; a genuinely unmodelled
        # region still spikes above it. It also makes the comparison one the
        # agent can only win by MOVING.
        self._past: "deque[Dict[str, torch.Tensor]]" = deque(
            maxlen=max(2, int(past_states)))
        self._baseline_every = max(1, int(baseline_every))
        self._baseline_samples = max(1, int(baseline_samples))
        # None until the buffer has anything to say -> cold start keeps the
        # ORIGINAL behaviour. An agent with no memory should still be curious;
        # paying 0 here would be a latch of exactly the kind this file warns
        # about two lines up.
        self._dis_baseline: Optional[float] = None
        self._last: Dict[str, float] = {
            "disagreement": 0.0, "imagined": 0.0,
            "boredom": 0.0, "bonus": 0.0}
        self._probes = 0

    # ---- helpers ---------------------------------------------------------
    def wants_step(self, step: int) -> bool:
        return self.enabled and (step % self.every == 0)

    def _boredom(self, global_lp: float) -> float:
        """1.0 when external curiosity is dead, 0.0 while it is healthy."""
        return float(np.clip(1.0 - (global_lp / self.lp_reference), 0.0, 1.0))

    def _imagined_decay(self, timestep: int) -> float:
        """Halflife decay — the daydream bonus must shrink over a life."""
        return float(0.5 ** (timestep / self.imagined_reward_halflife))

    def _imagine(self, world_model, rssm_state, policy_fn, knowledge):
        """K rollouts from ONE state. train() is restored even on failure."""
        K = self.rollouts
        was_training = world_model.training
        world_model.eval()
        try:
            with torch.no_grad():
                s0 = {k: rssm_state[k][0:1].detach().clone().repeat(K, 1)
                      for k in ("h", "z")}
                kn = (knowledge.unsqueeze(0).expand(K, -1)
                      if knowledge is not None and knowledge.dim() == 1
                      else knowledge)
                return world_model.imagine_trajectory(
                    s0, policy_fn=policy_fn, horizon=self.horizon,
                    knowledge=kn)
        finally:
            world_model.train(was_training)

    @staticmethod
    def _spread(imagined) -> float:
        """Std across rollouts, averaged over horizon and features."""
        lat = imagined["latents"]              # (K,H,D) or list
        if isinstance(lat, (list, tuple)):
            lat = torch.stack(list(lat), dim=1)
        return float(lat.std(dim=0).mean().item())

    def _sample_past(self, rssm_state) -> None:
        """Remember this state, SPARSELY.

        Stored every `baseline_every` probes, not every probe, so the buffer
        spans TIME rather than the last few minutes. At the defaults (probe
        every 50 steps, stride 10, 32 slots) that is ~16k steps of history.
        A buffer holding only the recent past would make the baseline agree
        with wherever the agent currently is -- exactly the comparison this
        is meant to break.
        """
        if self._probes % self._baseline_every:
            return
        try:
            self._past.append({k: rssm_state[k][0:1].detach().clone()
                               for k in ("h", "z")})
        except Exception:
            pass          # a memory that will not store must not kill a run

    def _refresh_baseline(self, world_model, policy_fn, knowledge) -> None:
        """Mean disagreement over a sample of REMEMBERED states.

        Recomputed every `baseline_every` probes and cached in between. This
        is the only cost this change adds; paying it every probe would
        multiply imagination cost by `baseline_samples` on a GPU that has
        only just stopped running out of memory.
        """
        # A baseline needs enough states to be a BASELINE. With one or two,
        # "typical uncertainty" is just "uncertainty at that one place", and
        # a single unlucky sample would silence the drive entirely.
        if len(self._past) < max(2, self._baseline_samples):
            return
        if self._probes % self._baseline_every:
            return
        try:
            n = min(self._baseline_samples, len(self._past))
            # Evenly spaced across the buffer, NOT the newest n -- the newest
            # are the most likely to be the very place we are comparing
            # against, which would drive the baseline toward `here`.
            idx = np.linspace(0, len(self._past) - 1, n).astype(int)
            vals = [self._spread(self._imagine(world_model, self._past[i],
                                               policy_fn, knowledge))
                    for i in idx]
            if vals:
                self._dis_baseline = float(np.mean(vals))
        except Exception:
            pass          # keep the previous baseline; never break the run

    # ---- main ------------------------------------------------------------
    def bonus(self, world_model, rssm_state: Dict[str, torch.Tensor],
              policy_fn, timestep: int, global_lp: float,
              knowledge: Optional[torch.Tensor] = None,
              symexp_fn=None) -> float:
        """Intrinsic bonus for the CURRENT state. Returns 0.0 when disabled,
        untrained, bored=0, or on any failure (never breaks a run)."""
        if not self.enabled:
            return 0.0
        boredom = self._boredom(global_lp)
        if boredom <= 0.0:
            self._last.update(boredom=0.0, bonus=0.0)
            return 0.0          # world is still teaching — do not daydream
        try:
            imagined = self._imagine(world_model, rssm_state, policy_fn,
                                     knowledge)
            dis_raw = self._spread(imagined)

            # --- 1. DISAGREEMENT, RELATIVE TO THE AGENT'S OWN HISTORY -----
            # WHAT WAS HERE, and why it failed (2026-09-25):
            #     self._dis_scale = max(self._dis_scale * 0.999, dis_raw)
            #     dis = clip(dis_raw / (self._dis_scale + 1e-8), 0, 1)
            # A running maximum that DECAYS. When the model learned the cave
            # and dis_raw fell to a low constant, _dis_scale decayed down to
            # meet it and `dis` climbed back to ~1.0. "Nothing left to
            # imagine" therefore read as MAXIMUM NOVELTY, and the bonus could
            # never self-extinguish the way this module docstring claims.
            # Same shape as the degenerate-signal gate that counted labels
            # instead of positives: a measure blind to its own exhaustion.
            #
            # NOT replaced with a fixed threshold -- the original note in
            # __init__ is right that raw latent variance shrinks as the WM
            # sharpens, and a constant would repeat the eps_abs latch that
            # killed the vision magnet. Normalising by the BASELINE stays
            # adaptive AND can reach zero, which a decaying scale cannot.
            # ORDER MATTERS. Refresh from the PAST first, then remember the
            # present: appending first made `here` part of its own baseline,
            # so the very first probe scored (dis_raw - dis_raw) = 0 and the
            # cold-start path could never run.
            self._refresh_baseline(world_model, policy_fn, knowledge)
            self._sample_past(rssm_state)
            if self._dis_baseline is None:
                # COLD START: nothing remembered yet, so keep the original
                # behaviour. An agent with no past should still be curious.
                self._dis_scale = max(self._dis_scale * 0.999, dis_raw)
                dis = float(np.clip(dis_raw / (self._dis_scale + 1e-8),
                                    0.0, 1.0))
            else:
                # "How much MORE uncertain is here than I am used to?", as a
                # fraction of usual. here == usual -> 0. Twice as uncertain
                # as usual -> 1. No separate scale that can latch.
                base = self._dis_baseline
                dis = float(np.clip((dis_raw - base) / (base + 1e-8),
                                    0.0, 1.0))

            # --- 2. IMAGINED REWARD (small, decaying) ---
            rew = imagined.get("rewards")
            img = 0.0
            if rew is not None:
                r = symexp_fn(rew) if symexp_fn is not None else rew
                # mean best-case payoff the model dreams up from here
                img_raw = float(r.squeeze(-1).sum(dim=-1).mean().item())
                img = float(np.tanh(max(0.0, img_raw)))   # bounded, >=0

            decay = self._imagined_decay(timestep)
            bonus = boredom * (self.disagreement_weight * dis
                               + self.imagined_reward_weight * decay * img)
            bonus = float(np.clip(bonus, 0.0, self.max_bonus))
            self._probes += 1
            self._last = {"disagreement": round(dis, 4),
                          # baseline VISIBLE: without it, "dis fell to 0" and
                          # "the baseline is wrong" look identical in a log.
                          "dis_raw": round(dis_raw, 5),
                          "dis_baseline": (round(self._dis_baseline, 5)
                                           if self._dis_baseline is not None
                                           else -1.0),
                          "past_states": len(self._past),
                          "imagined": round(img, 4),
                          "boredom": round(boredom, 3),
                          "decay": round(decay, 3),
                          "bonus": round(bonus, 4)}
            return bonus
        except Exception:
            logger.exception("imagination curiosity failed — returning 0")
            return 0.0

    @property
    def stats(self) -> Dict[str, float]:
        return dict(self._last, probes=self._probes, enabled=self.enabled)
