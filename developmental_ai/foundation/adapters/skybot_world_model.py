"""The existing RSSM behind the StateBelief / Prediction records.

Plan Stage 2 item 4: "Provide wrappers around existing state and prediction
APIs." This wraps `world_model.rssm.RSSM` WITHOUT changing it:

    belief(state, belief_id, support)    RSSM state {"h", "z"} -> StateBelief
    predict(state, belief_id, commands)  RSSM.imagine_step rolled over the
                                         candidate commands -> Prediction

The outcome distribution is the RSSM's own prior at each imagined step,
recomputed from the imagined h with `rssm.prior_net` (the same network
imagine_step sampled z from): outcome["probs"] is (horizon, S, C). The
sampled z and h are recorded alongside.

SHADOW-SAFE (plan §8). The wrapper runs under no_grad, never toggles
train/eval mode, and forks the CPU RNG so the live learner's random stream
is not advanced by a shadow prediction. Whether z was sampled by Gumbel
(module in training mode) or argmax (eval mode) is recorded, not changed.
Every Prediction names its model version and snapshot id, so it can be
scored against later evidence and never mistaken for evidence itself.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from ..contracts import (INAPPLICABLE, UNKNOWN, ActionSpec, ContractError,
                         ObsRef, Prediction, StateBelief)


class RSSMPredictor:
    def __init__(self, rssm, action_spec: ActionSpec, model_version: str,
                 snapshot_id: str, model_name: str = "world_model.rssm"):
        self.rssm = rssm
        self.spec = action_spec
        self.model_version = str(model_version)
        self.snapshot_id = str(snapshot_id)
        self.model_name = model_name
        ad = int(rssm.action_dim)
        if action_spec.kind == "discrete" and action_spec.n != ad:
            raise ContractError(f"discrete spec n={action_spec.n} != RSSM "
                                f"action_dim={ad}")
        if action_spec.kind == "box" and int(np.prod(action_spec.shape or (1,))) != ad:
            raise ContractError(f"box spec {action_spec.shape} != RSSM "
                                f"action_dim={ad}")
        if action_spec.kind == "multi_discrete" and sum(action_spec.nvec) != ad:
            raise ContractError(f"multi_discrete one-hots sum to "
                                f"{sum(action_spec.nvec)} != action_dim={ad}")
        self.representation = (f"rssm_latent[h{rssm.deterministic_size},"
                               f"z{rssm.stochastic_size}x{rssm.stochastic_classes}]")

    # ---- helpers ---------------------------------------------------------
    def _check_state(self, state):
        h, z = state.get("h"), state.get("z")
        r = self.rssm
        if h is None or z is None or tuple(h.shape) != (1, r.deterministic_size) \
                or tuple(z.shape) != (1, r.stoch_dim):
            raise ContractError(
                f"RSSM state must be batch-1 {{'h': (1, {r.deterministic_size}), "
                f"'z': (1, {r.stoch_dim})}}")

    def _encode(self, cmd):
        """Canonical command -> the action vector the RSSM was trained on."""
        ad = int(self.rssm.action_dim)
        v = np.zeros(ad, np.float32)
        if self.spec.kind == "discrete":
            v[cmd] = 1.0
        elif self.spec.kind == "box":
            v[:] = np.asarray(cmd, np.float32).reshape(-1)
        else:
            off = 0
            for x, n in zip(cmd, self.spec.nvec):
                v[off + x] = 1.0
                off += n
        return v

    # ---- records ---------------------------------------------------------
    def belief(self, state, belief_id: str, support: Sequence[ObsRef] = ()) -> StateBelief:
        self._check_state(state)
        return StateBelief(
            belief_id, self.representation, 1,
            {"h": state["h"].detach().cpu().numpy()[0].astype(np.float32),
             "z": state["z"].detach().cpu().numpy()[0].astype(np.float32)},
            # h is deterministic given the history; the posterior logits that
            # would quantify z are not part of the RSSM state dict.
            {"h": INAPPLICABLE, "z": UNKNOWN},
            tuple(support),
            {"model": self.model_name, "model_version": self.model_version,
             "snapshot_id": self.snapshot_id})

    def predict(self, state, belief_id: str, commands: Sequence, prediction_id: str,
                t_wall: Optional[float] = None, seed: Optional[int] = None) -> Prediction:
        import torch
        self._check_state(state)
        if len(commands) < 1:
            raise ContractError("predict needs at least one command")
        canon = [self.spec.validate_command(c) for c in commands]
        r = self.rssm
        dev = next(r.parameters()).device
        S, C = r.stochastic_size, r.stochastic_classes
        probs, hs, zs = [], [], []
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(int(seed))
            st = {"h": state["h"].to(dev), "z": state["z"].to(dev)}
            for c in canon:
                a = torch.as_tensor(self._encode(c), device=dev).unsqueeze(0)
                st = r.imagine_step(st, a)
                logits = r.prior_net(st["h"]).view(1, S, C)
                probs.append(torch.softmax(logits, -1)[0].cpu().numpy())
                hs.append(st["h"][0].cpu().numpy())
                zs.append(st["z"][0].cpu().numpy())
        return Prediction(
            prediction_id, belief_id, {self.model_name: self.model_version},
            self.snapshot_id, self.spec.spec_id, tuple(canon), len(canon),
            {"kind": "rssm_prior_categorical",
             "probs": np.stack(probs).astype(np.float32),
             "h": np.stack(hs).astype(np.float32),
             "z_sample": np.stack(zs).astype(np.float32)},
            float(time.time() if t_wall is None else t_wall),
            {"sampling": "gumbel" if r.training else "argmax"})
