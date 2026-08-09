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
            K, H = self.rollouts, self.horizon
            was_training = world_model.training
            world_model.eval()
            try:
                with torch.no_grad():
                    s0 = {k: rssm_state[k][0:1].detach().clone().repeat(K, 1)
                          for k in ("h", "z")}
                    kn = (knowledge.unsqueeze(0).expand(K, -1)
                          if knowledge is not None and knowledge.dim() == 1
                          else knowledge)
                    imagined = world_model.imagine_trajectory(
                        s0, policy_fn=policy_fn, horizon=H, knowledge=kn)
            finally:
                world_model.train(was_training)

            # --- 1. DISAGREEMENT: spread across the K imagined futures ---
            lat = imagined["latents"]              # (K,H,D) or list
            if isinstance(lat, (list, tuple)):
                lat = torch.stack(list(lat), dim=1)
            # std across rollouts, averaged over horizon+features
            dis_raw = float(lat.std(dim=0).mean().item())
            self._dis_scale = max(self._dis_scale * 0.999, dis_raw)
            dis = float(np.clip(dis_raw / (self._dis_scale + 1e-8), 0.0, 1.0))

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
