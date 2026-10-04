"""Residual model: what a mechanism leaves unexplained (plan Stage 6.5).

`ResidualMechanism(base)` keeps the base mechanism FROZEN and learns, with a
small MLP over the same flattened inputs plus the base's own predicted mean,
a Gaussian over  r = observed next state - base mean.  Its prediction is

    mean = base mean + residual mean,   var = residual variance

(the residual net's variance replaces the base's: it is fitted to what is
actually left over). The discrete part is the base's, untouched.

Why keep it separate instead of retraining the base: the residual is the
EVIDENCE of a missing effect. `unexplained(batch)` reports per-dimension
how much systematic error the base leaves and how much the residual
removes; a large, structured residual is the cue structural discovery
(Stage 9) needs, and a residual near zero says the base is adequate there.

STALENESS IS AN ERROR, NOT A DRIFT. The residual is fitted against one
base version. If the base is refitted afterwards, the residual describes a
model that no longer exists; predict() raises until the residual is
refitted (fit() is the escape path — this guard is not a latch, §4.1).

STATIC DIMS AND RESUMABLE TRAINING: as for every torch mechanism
(_nets module docstring). A dim whose residual is identically zero in the
training data (the base already gets it exactly, e.g. a wall's position)
is passed through as the base mean and excluded from the loss; one Adam
lives across fits; training_state_dict() carries it (and the base version
the residual was fitted against — the base itself is saved separately).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from ..contracts import ContractError
from ._nets import (ResumableTraining, Standardizer, count_params, head_loss, mlp,
                    pass_static, seeded, split_head, t, train, update_static_mask)
from .api import BaseMechanism, MixedOutcome, TransitionBatch


class ResidualMechanism(ResumableTraining, BaseMechanism):
    _TRAIN_ATTRS = ResumableTraining._TRAIN_ATTRS + ("base_version_at_fit",)

    def __init__(self, base: BaseMechanism, hidden: int = 64, seed: int = 0,
                 mechanism_id: str = None):
        if not isinstance(base, BaseMechanism):
            raise ContractError("ResidualMechanism wraps a BaseMechanism")
        super().__init__(mechanism_id or f"residual[{base.mechanism_id}]", base.N, base.D,
                         base.A, base.Ad, base.layout.classes or ("none", "event"),
                         assumptions={"base": base.mechanism_id, "base_frozen": True})
        self.layout = base.layout
        self.base = base
        self.uses_absolute_position = True
        self.seed = int(seed)
        self.base_version_at_fit = None
        Dc = self.layout.Dc
        self.in_dim = self.N * (2 * self.D + self.A + self.Ad) + 1 + Dc
        with seeded(self.seed):
            m = nn.Module()
            m.in_std = Standardizer(self.in_dim)
            m.out_std = Standardizer(Dc)
            m.net = mlp(self.in_dim, int(hidden), 2 * Dc, layers=2)
        m.register_buffer("static", torch.zeros(Dc, dtype=torch.bool))
        self.module = m

    def _trainable_params(self):
        return list(self.module.net.parameters())

    def model_versions(self) -> Dict[str, int]:
        return {self.mechanism_id: int(self.version),
                self.base.mechanism_id: int(self.base.version)}

    def _x(self, state, action, dt, base_mean):
        B = state.B
        x = np.concatenate([state.pos.reshape(B, -1), state.vel.reshape(B, -1),
                            state.attrs.reshape(B, -1), action.reshape(B, -1),
                            dt[:, None], base_mean], 1)
        return t(x)

    def _fit(self, batch: TransitionBatch, epochs: int = 60, batch_size: int = 128,
             lr: float = 2e-3, seed=None, **kw):
        if self.base.version < 1:
            raise ContractError("fit the base mechanism before its residual")
        bm = self.base.predict(batch.state, batch.action, batch.dt).mean()
        x = self._x(batch.state, batch.action, batch.dt, bm)
        rr = batch.next_state.continuous() - bm
        m = self.module
        static = update_static_mask(rr, None if self.version == 0 else m.static.numpy(),
                                    np.zeros(self.layout.Dc, bool), self.mechanism_id)
        m.static.copy_(torch.from_numpy(static))
        st = m.static.clone()
        r = t(rr)
        m.in_std.fit(x)
        m.out_std.fit(r)
        rn = m.out_std(r)
        xn = m.in_std(x)
        m.train()

        def loss_fn(idx):
            mu, var, _ = split_head(m.net(xn[idx]), self.layout.Dc, 0, 0)
            return head_loss(mu, var, None, rn[idx], None, st)

        hist = train(self._trainable_params(), loss_fn, len(batch), epochs=epochs,
                     batch_size=batch_size, lr=lr,
                     seed=self.seed + 1000 * (self.version + 1) if seed is None else seed,
                     optimizer=self._optimizer(lr))
        m.eval()
        self.base_version_at_fit = int(self.base.version)
        return {"loss_first": hist[0], "loss_last": hist[-1]}

    def _predict(self, state, action, dt) -> MixedOutcome:
        if self.base_version_at_fit is None:
            raise ContractError(f"{self.mechanism_id} has not been fitted")
        if self.base.version != self.base_version_at_fit:
            raise ContractError(
                f"residual fitted against {self.base.mechanism_id} v"
                f"{self.base_version_at_fit}, base is now v{self.base.version}; "
                f"refit the residual")
        bd = self.base.predict(state, action, dt)
        bm = bd.mean()
        m = self.module
        with torch.no_grad():
            mu, var, _ = split_head(m.net(m.in_std(self._x(state, action, dt, bm))),
                                    self.layout.Dc, 0, 0)
        sd = m.out_std.sd.double().numpy()
        mean = bm + mu.double().numpy() * sd + m.out_std.mu.double().numpy()
        mean, v = pass_static(mean, var.double().numpy() * sd ** 2, bm, m.static.numpy(), sd)
        return MixedOutcome.gaussian_categorical(self.layout, mean, v, bd.probs_marginal())

    def unexplained(self, batch: TransitionBatch) -> Dict[str, Any]:
        """Per-dim RMS of what the base leaves and of what remains after the
        residual, plus the residual's mean correction (bias the base misses)."""
        y = batch.next_state.continuous()
        bm = self.base.predict(batch.state, batch.action, batch.dt).mean()
        rm = self.predict(batch.state, batch.action, batch.dt).mean()
        base_rms = np.sqrt(((y - bm) ** 2).mean(0))
        after_rms = np.sqrt(((y - rm) ** 2).mean(0))
        return {"base_rms": base_rms, "after_rms": after_rms,
                "mean_correction": (rm - bm).mean(0),
                "explained_fraction": 1.0 - (after_rms ** 2).sum() / max((base_rms ** 2).sum(), 1e-12)}

    def state_dict(self):
        return self.module.state_dict()

    def resources(self, *a, **k):
        return {"params": count_params(self.module)}
