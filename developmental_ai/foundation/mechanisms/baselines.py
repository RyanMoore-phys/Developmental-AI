"""Comparison models for the relational mechanism (plan Stage 6.3).

MLPMechanism  — "an ordinary model receiving the same structured inputs".
    It reads EXACTLY the EntityBatch the Interaction Network reads (every
    entity's pos, vel, attrs, action, plus dt), flattened, and predicts the
    same outcome (Gaussian over the change of every entity's pos/vel +
    per-entity event logits) with the same loss. So a difference between
    the two is architecture, not information. It does receive absolute
    positions — the IN with a declared translation symmetry deliberately
    does not, which is LESS input, never more. Compute is not equal by
    default: report `resources()` (params, FLOPs) next to any comparison
    and size `hidden` to match when compute is the question ("extra inputs
    and extra compute must not be confused with architectural gains").

ConstantVelocityMechanism — declared kinematics with no learned dynamics:
    pos' = pos + vel*dt, vel' = vel; fit() estimates only a per-dim residual
    variance and per-entity event frequencies. A floor any learned model
    must clear, and a convenient imperfect base for residual.py.

RSSMMechanism — the existing world_model.rssm.RSSM where its outputs are
    comparable. WHAT IS COMPARABLE: the next continuous entity state, read
    out of the RSSM's imagined latent by the repo's own ObservationDecoder
    (as the live agent reads observations back out of latents), scored by the
    same NLL/CRPS/coverage on the same observable target. WHAT IS NOT:
      * events     the RSSM has no event head; its layout has NO discrete
                   part (comparison on events is INAPPLICABLE, not zero)
      * dt         the RSSM has no duration input; dt is appended to the
                   action vector (an input the live RSSM never receives)
      * memory     trained here with one-step context (posterior on obs_t,
                   imagine one step) because the fixture state is Markov;
                   its recurrent memory is therefore not exercised
      * latents    KL / latent distances have no counterpart and are never
                   compared (plan §7.3)
    `native_prediction` routes a single state through
    adapters.RSSMPredictor so the contract record is the RSSM's own, and
    returns the decoded outcome so a test can prove both paths agree.

Static outcome dims and resumable training follow _nets (module docstring)
for every learned model here: never-moving dims are passed through and
excluded from the NLL; one Adam per mechanism persists across fits.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..contracts import ActionSpec, ContractError, Prediction
from ._nets import (ResumableTraining, Standardizer, TorchMechanism, count_flops,
                    count_params, head_loss, mlp, pass_static, seeded, split_head, t,
                    train, update_static_mask)
from .api import (BaseMechanism, EntityBatch, MixedOutcome, OutcomeLayout,
                  TransitionBatch)


class MLPMechanism(TorchMechanism):
    def __init__(self, n_entities: int, dim: int, attr_dim: int, action_dim: int,
                 classes: Sequence[str], hidden: int = 128, layers: int = 2,
                 seed: int = 0, mechanism_id: str = "baseline.mlp"):
        self.layers = int(layers)
        super().__init__(mechanism_id, n_entities, dim, attr_dim, action_dim, classes,
                         assumptions={"translation_invariant": False,
                                      "same_inputs_as": "relational.interaction_network"},
                         seed=seed, hidden=hidden)
        self.uses_absolute_position = True

    @property
    def in_dim(self) -> int:
        return self.N * (2 * self.D + self.A + self.Ad) + 1

    def _build(self):
        m = nn.Module()
        m.in_std = Standardizer(self.in_dim)
        Dc, N, K = self.layout.Dc, self.N, self.layout.K
        m.net = mlp(self.in_dim, self.hidden, 2 * Dc + N * K, layers=self.layers)
        return m

    def _prepare(self, state, action, dt):
        B = state.B
        x = torch.cat([t(state.pos), t(state.vel), t(state.attrs), t(action)], -1)
        return {"x": torch.cat([x.reshape(B, -1), t(dt)[:, None]], -1)}

    def _fit_stats(self, prep):
        self.module.in_std.fit(prep["x"])

    def _forward(self, prep):
        raw = self.module.net(self.module.in_std(prep["x"]))
        mu, var, logits = split_head(raw, self.layout.Dc, self.N, self.layout.K)
        return mu, var, logits


class ConstantVelocityMechanism(BaseMechanism):
    def __init__(self, n_entities, dim, attr_dim, action_dim, classes,
                 mechanism_id: str = "baseline.constant_velocity"):
        super().__init__(mechanism_id, n_entities, dim, attr_dim, action_dim, classes,
                         assumptions={"translation_invariant": True,
                                      "dynamics": "declared constant velocity, no forces"})
        self._var = np.ones(self.layout.Dc)
        self._freq = np.full((self.N, self.layout.K), 1.0 / self.layout.K)

    def _mean(self, state, dt):
        return np.concatenate([state.pos + state.vel * dt[:, None, None], state.vel],
                              -1).reshape(state.B, -1)

    def _predict(self, state, action, dt):
        B = state.B
        return MixedOutcome.gaussian_categorical(
            self.layout, self._mean(state, dt), np.broadcast_to(self._var, (B, self.layout.Dc)),
            np.broadcast_to(self._freq, (B, self.N, self.layout.K)))

    def _fit(self, batch, **kw):
        r = batch.next_state.continuous() - self._mean(batch.state, batch.dt)
        self._var = np.maximum((r ** 2).mean(0), 1e-8)
        cnt = np.ones((self.N, self.layout.K))                   # Laplace
        for k in range(self.layout.K):
            cnt[:, k] += (batch.events == k).sum(0)
        self._freq = cnt / cnt.sum(1, keepdims=True)
        return {"residual_rms": float(np.sqrt(self._var.mean()))}


class RSSMMechanism(ResumableTraining, BaseMechanism):
    """See module docstring for what is and is not comparable."""

    def __init__(self, n_entities, dim, attr_dim, action_dim, classes,
                 stochastic_size: int = 8, stochastic_classes: int = 8,
                 deterministic_size: int = 64, hidden: int = 64, seed: int = 0,
                 mechanism_id: str = "baseline.rssm"):
        from ...world_model.rssm import RSSM, ObservationDecoder
        super().__init__(mechanism_id, n_entities, dim, attr_dim, action_dim, classes,
                         assumptions={"translation_invariant": False,
                                      "dt_as_action_dim": True, "context_steps": 1,
                                      "events": "INAPPLICABLE (no event head)"})
        self.uses_absolute_position = True
        self.event_classes = tuple(classes)
        self.layout = OutcomeLayout(self.layout.cont_names)       # no discrete part
        self.seed = int(seed)
        Dc = self.layout.Dc
        self.obs_dim = Dc + self.N * self.A
        self.act_dim = self.N * self.Ad + 1
        with seeded(self.seed):
            m = nn.Module()
            m.rssm = RSSM(self.obs_dim, self.act_dim, stochastic_size, stochastic_classes,
                          deterministic_size, hidden)
            m.dec = ObservationDecoder(m.rssm.latent_dim, 2 * Dc, hidden_dim=hidden)
            m.obs_std = Standardizer(self.obs_dim)
            m.act_std = Standardizer(self.act_dim)
            m.out_std = Standardizer(Dc)
        m.register_buffer("static", torch.zeros(Dc, dtype=torch.bool))
        self.module = m
        self.module.eval()

    def _trainable_params(self):
        return list(self.module.parameters())

    def _vecs(self, state, action, dt):
        B = state.B
        obs = t(np.concatenate([state.continuous(), state.attrs.reshape(B, -1)], 1))
        act = t(np.concatenate([action.reshape(B, -1), dt[:, None]], 1))
        return obs, act

    def _posterior(self, obs):
        r = self.module.rssm
        s0 = r.initial_state(obs.shape[0], torch.device("cpu"))
        post, _ = r.observe_step(s0, torch.zeros(obs.shape[0], self.act_dim),
                                 self.module.obs_std(obs))
        return post

    def _head(self, latent):
        return split_head(self.module.dec(latent), self.layout.Dc, 0, 0)[:2]

    def _out(self, mu, var, state):
        sd = self.module.out_std.sd.double().numpy()
        m = self.module.out_std.mu.double().numpy()
        cur = state.continuous()
        mean, v = pass_static(cur + mu.double().numpy() * sd + m,
                              var.double().numpy() * sd ** 2, cur,
                              self.module.static.numpy(), sd)
        return MixedOutcome.gaussian_categorical(self.layout, mean, v)

    def _predict(self, state, action, dt):
        self.module.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            obs, act = self._vecs(state, action, dt)
            post = self._posterior(obs)
            prior = self.module.rssm.imagine_step(post, self.module.act_std(act))
            mu, var = self._head(self.module.rssm.get_latent(prior))
        return self._out(mu, var, state)

    def _fit(self, batch: TransitionBatch, epochs: int = 60, batch_size: int = 128,
             lr: float = 2e-3, seed=None, kl_weight: float = 0.1, **kw):
        m, r = self.module, self.module.rssm
        obs, act = self._vecs(batch.state, batch.action, batch.dt)
        obs1, _ = self._vecs(batch.next_state, batch.action, batch.dt)
        m.obs_std.fit(obs)
        m.act_std.fit(act)
        change = batch.next_state.continuous() - batch.state.continuous()
        static = update_static_mask(change, None if self.version == 0 else m.static.numpy(),
                                    np.zeros(self.layout.Dc, bool), self.mechanism_id)
        m.static.copy_(torch.from_numpy(static))
        st = m.static.clone()
        y = t(change)
        m.out_std.fit(y)
        y = m.out_std(y)
        m.train()

        def loss_fn(idx):
            post = self._posterior(obs[idx])
            a = m.act_std(act[idx])
            prior = r.imagine_step(post, a)
            mu, var = self._head(r.get_latent(prior))
            _, info = r.observe_step(post, a, m.obs_std(obs1[idx]))
            kl = r.compute_kl_loss(info["prior_logits"], info["posterior_logits"])
            return head_loss(mu, var, None, y[idx], None, st) + kl_weight * kl

        hist = train(self._trainable_params(), loss_fn, len(batch), epochs=epochs,
                     batch_size=batch_size, lr=lr,
                     seed=self.seed + 1000 * (self.version + 1) if seed is None else seed,
                     optimizer=self._optimizer(lr))
        m.eval()
        return {"loss_first": hist[0], "loss_last": hist[-1], "epochs": len(hist)}

    # ---- the RSSM's own record path -------------------------------------
    def action_spec(self) -> ActionSpec:
        n = self.act_dim
        return ActionSpec.box(f"{self.mechanism_id}.std_action", -1e3 * np.ones(n),
                              1e3 * np.ones(n), tuple(["standardised"] * n))

    def native_prediction(self, state: EntityBatch, action, dt, *, prediction_id: str,
                          belief_id: str, snapshot_id: str):
        """(Prediction from adapters.RSSMPredictor, decoded MixedOutcome) for a
        batch-1 state. The decode reads the h and z the RSSM itself recorded."""
        from ..adapters.skybot_world_model import RSSMPredictor
        action, dt = self._check(state, action, dt)
        if state.B != 1:
            raise ContractError("native_prediction is batch-1 (RSSMPredictor contract)")
        self.module.eval()
        with torch.no_grad():
            obs, act = self._vecs(state, action, dt)
            post = self._posterior(obs)
            cmd = self.module.act_std(act)[0].double().numpy()
        rp = RSSMPredictor(self.module.rssm, self.action_spec(), str(self.version),
                           snapshot_id, model_name=self.mechanism_id)
        pred = rp.predict(post, belief_id, [cmd], prediction_id)
        lat = torch.cat([t(pred.outcome["h"][-1:]), t(pred.outcome["z_sample"][-1:])], -1)
        with torch.no_grad():
            mu, var = self._head(lat)
        return pred, self._out(mu, var, state)

    def state_dict(self):
        return self.module.state_dict()

    def resources(self, state=None, action=None, dt=None) -> Dict[str, Any]:
        out = {"params": count_params(self.module)}
        if state is not None:
            out["flops_per_batch"] = count_flops(
                self.module, lambda: self._predict(state, np.asarray(action, float),
                                                   np.asarray(dt, float)))
            out["flops_per_sample"] = out["flops_per_batch"] // max(state.B, 1)
        return out

    comparable = {"continuous_next_state": True, "events": False,
                  "dt": "appended to action", "latent_kl": False}
