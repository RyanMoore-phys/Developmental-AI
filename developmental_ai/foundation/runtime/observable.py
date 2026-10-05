"""Deterministic observable diagnostics. No sampling, mode changes or reward.

Callers of the torch functions hold the model lock. Argmax latent rollouts are
a diagnostic decoding rule, not a claim to be the mean of a nonlinear
stochastic decoder.

`pool_observation` is the ONE observable target used by the shadow recorder
for forecast, persistence baseline and outcome alike. It is plain numpy on
host arrays (2026-10-05): pooling on the device and calling `.cpu()` on the
acting thread forced a sync with the async trainer's queued kernels and was
charged to the shadow budget, and the forecast/persistence pooling ran
inside the world-model lock where only weight reads belong.
"""
import numpy as np
import torch
import torch.nn.functional as F

PROBE_SIDE = 8


def transition(rssm, state, action):
    """h_{t+1}: `RSSM.imagine_step` (no knowledge) up to its prior. No RNG."""
    film = getattr(rssm, 'film', None)
    base = (film(action, state['z']) if film is not None
            else torch.cat([state['z'], action], -1))
    return rssm.gru_norm(rssm.gru(rssm.gru_input_proj(base), state['h']))


def prior_step(rssm, state, action):
    """(h_{t+1}, prior logits of z_{t+1}) — ONE GRU transition, reused by the
    categorical prediction and the observable forecast alike."""
    h = transition(rssm, state, action)
    return h, rssm.prior_net(h)


def categorical_mode(rssm, logits):
    logits = logits.reshape(-1, rssm.stochastic_size, rssm.stochastic_classes)
    return F.one_hot(logits.argmax(-1), rssm.stochastic_classes).float().flatten(1)


@torch.no_grad()
def decode_prior_mode(model, h, prior_logits):
    """Decoded observation at the prior's categorical mode, in observation
    units (symexp for vector models, raw decoder output for pixels)."""
    z = categorical_mode(model.rssm, prior_logits)
    decoded = model.decoder(model.rssm.get_latent({'h': h, 'z': z}))
    if not model.pixel_obs:
        from developmental_ai.world_model.rssm import symexp
        decoded = symexp(decoded)
    return decoded


@torch.no_grad()
def predict_observation(model, state, actions):
    h, logits = prior_step(model.rssm, state, actions)
    return decode_prior_mode(model, h, logits)


def probe_meta(model):
    """The pooling parameters a forecast is filed with (so the outcome is
    pooled identically without touching the model again)."""
    return {"pixel": bool(model.pixel_obs),
            "channels": int(getattr(model, "image_channels", 0) or 0),
            "size": int(getattr(model, "image_size", 0) or 0)}


def _bins(n_in, n_out):
    # torch adaptive_avg_pool semantics: [floor(i*n/o), ceil((i+1)*n/o))
    return [((i * n_in) // n_out, -((-(i + 1) * n_in) // n_out))
            for i in range(n_out)]


def pool_observation(values, pixel, channels, size, side=PROBE_SIDE):
    """(k, D) host array -> (k, d) float32 observable target, same units.

    Pixels (rows laid out C,H,W) are average-pooled to side x side with the
    exact bins of torch's adaptive_avg_pool2d (a plain block mean when size
    is a multiple of side, which every power-of-two image_size >= 16 is).
    Vector observations are returned flattened. Never modifies `values`.
    """
    x = np.asarray(values, dtype=np.float32)
    x = x.reshape(x.shape[0] if x.ndim > 1 else 1, -1)
    if not pixel:
        return x.astype(np.float32, copy=True)
    channels, size, side = int(channels), int(size), int(side)
    if x.shape[1] != channels * size * size:
        raise ValueError(f"pixel observation width {x.shape[1]} != "
                         f"{channels}x{size}x{size}")
    k = x.shape[0]
    img = x.reshape(k, channels, size, size).astype(np.float64)
    if size % side == 0:
        b = size // side
        out = img.reshape(k, channels, side, b, side, b).mean(axis=(3, 5))
    else:
        out = np.empty((k, channels, side, side), np.float64)
        bins = _bins(size, side)
        for i, (r0, r1) in enumerate(bins):
            for j, (c0, c1) in enumerate(bins):
                out[:, :, i, j] = img[:, :, r0:r1, c0:c1].mean(axis=(2, 3))
    return out.reshape(k, -1).astype(np.float32)


@torch.no_grad()
def sequence_loss(model, batch):
    """One-step observable MSE, deterministic posterior state, fixed targets.

    Full frames are scored for pixels. Invalid boundary pairs are excluded.
    A restart-marked sequence is rejected instead of interpreting reset data.
    """
    obs, actions = batch['observations'], batch['actions']
    if obs.shape[1] < 2:
        raise ValueError('observable probes need at least two frames')
    # The replay column is `restarts` (replay_buffer.py); `restarted` was the
    # only key checked here, so the guard could never fire. Accept either.
    for _rk in ('restarts', 'restarted'):
        if _rk in batch and bool(batch[_rk].any()):
            raise ValueError('recovery sequence cannot be a progress probe')
    state = model.rssm.initial_state(len(obs), obs.device)
    encoded = model.embed(obs.reshape(-1, obs.shape[-1]),
                          None if batch.get('proprio') is None else
                          batch['proprio'].reshape(-1, model.proprio_dim))
    encoded = encoded.reshape(len(obs), obs.shape[1], -1)
    total, count = obs.new_zeros(()), 0
    for t in range(obs.shape[1] - 1):
        prev = torch.zeros_like(actions[:, t]) if t == 0 else actions[:, t-1]
        h = transition(model.rssm, state, prev)
        z = categorical_mode(model.rssm, model.rssm.posterior_net(
            torch.cat([h, encoded[:, t]], -1)))
        state = {'h': h, 'z': z}
        pred = predict_observation(model, state, actions[:, t])
        errors = (pred - obs[:, t+1]).square().flatten(1).mean(1)
        mask = torch.ones_like(errors, dtype=torch.bool)
        if 'dones' in batch:
            mask &= ~batch['dones'][:, t].reshape(-1).bool()
        if 'continues' in batch:
            mask &= batch['continues'][:, t].reshape(-1) > 0
        if mask.any():
            total += errors[mask].sum()
            count += int(mask.sum())
    if not count:
        raise ValueError('no valid observable transitions')
    return float(total / count)
