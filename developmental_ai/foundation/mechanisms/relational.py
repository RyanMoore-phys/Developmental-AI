"""A modest relational dynamics model: an Interaction Network (plan Stage 6.2).

Battaglia et al. 2016 shape, kept small:

    node encoder   phi_o(entity features)                 -> h_i
    edge model     phi_R(h_i, h_j, relative geometry_ij)  -> e_ij   (all i != j)
    aggregation    a_i = sum_j e_ij
    node update    phi_U(h_i, a_i, features_i)            -> outcome head_i

Entity features: velocity, static attributes, the action applied to that
entity, and dt (duration is an INPUT, plan Stage 6 test "duration
sensitivity"). Relative geometry: p_j - p_i, v_j - v_i, |p_j - p_i|.

Outcome head per entity: Gaussian over the change of (pos, vel) — mean and
softplus variance — and a categorical over event classes (contact type).

DECLARED SYMMETRY (plan §2.1, Stage 6.4 "apply symmetries only where their
assumptions are declared"). `translation_invariant=True` means: the model
never sees an absolute position, only relative ones, and predicts position
CHANGES — so translating every entity's position by c translates the
predicted positions by exactly c and changes nothing else. That is true by
construction, not learned, and is only CORRECT for a world whose physics has
no preferred place; in the box fixture that holds because the walls are
entities that move with the translation. With `translation_invariant=False`
the encoder also reads absolute positions and nothing guarantees
equivariance — the smoke test checks that it indeed is NOT equivariant, so
the symmetry claim is shown to come from the declaration, not for free.

No rotation symmetry is declared: gravity fixes a direction in the world
(a physical rotation changes the outcome; geometry.constraints).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ._nets import Standardizer, TorchMechanism, mlp, split_head, t


class InteractionNetwork(TorchMechanism):
    def __init__(self, n_entities: int, dim: int, attr_dim: int, action_dim: int,
                 classes: Sequence[str], translation_invariant: bool = True,
                 hidden: int = 64, seed: int = 0,
                 mechanism_id: str = "relational.interaction_network"):
        self.ti = bool(translation_invariant)
        super().__init__(mechanism_id, n_entities, dim, attr_dim, action_dim, classes,
                         assumptions={"translation_invariant": self.ti,
                                      "rotation_invariant": False,
                                      "pairwise_additive_interactions": True},
                         seed=seed, hidden=hidden)
        self.uses_absolute_position = not self.ti

    @property
    def node_in(self) -> int:
        return self.D + self.A + self.Ad + 1 + (0 if self.ti else self.D)

    def _build(self) -> nn.Module:
        H, D, K = self.hidden, self.D, self.layout.K
        m = nn.Module()
        m.node_std = Standardizer(self.node_in)
        m.rel_std = Standardizer(2 * D + 1, center=False)
        m.enc = mlp(self.node_in, H, H, layers=1)
        m.edge = mlp(2 * H + 2 * D + 1, H, H, layers=2)
        m.upd = mlp(2 * H + self.node_in, H, 4 * D + K, layers=2)
        return m

    def _prepare(self, state, action, dt):
        B, N = state.B, state.N
        f = [t(state.vel), t(state.attrs), t(action), t(dt)[:, None, None].expand(B, N, 1)]
        if not self.ti:
            f.append(t(state.pos))
        pos, vel = t(state.pos), t(state.vel)
        rp = pos[:, None, :, :] - pos[:, :, None, :]        # [b, i, j] = p_j - p_i
        rv = vel[:, None, :, :] - vel[:, :, None, :]
        dist = torch.sqrt((rp ** 2).sum(-1, keepdim=True) + 1e-12)
        return {"node": torch.cat(f, -1), "rel": torch.cat([rp, rv, dist], -1)}

    def _fit_stats(self, prep):
        self.module.node_std.fit(prep["node"])
        N = prep["rel"].shape[1]
        off = ~torch.eye(N, dtype=torch.bool)
        self.module.rel_std.fit(prep["rel"][:, off])

    def _forward(self, prep):
        m = self.module
        x = m.node_std(prep["node"])                         # (B, N, F)
        B, N, _ = x.shape
        h = m.enc(x)
        r = m.rel_std(prep["rel"])                           # (B, N, N, 2D+1)
        H = h.shape[-1]
        e = m.edge(torch.cat([h[:, :, None].expand(B, N, N, H),
                              h[:, None].expand(B, N, N, H), r], -1))
        mask = (~torch.eye(N, dtype=torch.bool)).to(e.dtype)[None, :, :, None]
        agg = (e * mask).sum(2)
        out = m.upd(torch.cat([h, agg, x], -1))              # (B, N, 4D + K)
        D2 = 2 * self.D
        mu, var, _ = split_head(torch.cat([out[..., :D2].reshape(B, -1),
                                           out[..., D2:2 * D2].reshape(B, -1)], -1),
                                N * D2, 0, 0)
        logits = out[..., 2 * D2:]
        return mu, var, logits
