"""Torch helpers shared by the learned mechanisms (private).

    mlp                 Linear/ELU stack
    Standardizer        per-feature affine normaliser; fitted ONCE (first
                        fit) and frozen, so a later fit cannot silently change
                        what an input means to an already-trained network;
                        constant features get scale 1 (not a tiny floor)
    GaussCatHead        splits a raw output into Gaussian mean / variance and
                        categorical logits; variance = softplus + floor
    head_loss           Gaussian NLL over the DYNAMIC dims (+ cross-entropy on
                        observed events); static dims are masked out
    update_static_mask  which continuous outcome dims are STATIC (see below)
    pass_static         predict static dims as identity with floor variance
    train               Adam over minibatches with an explicit numpy RNG; takes
                        a persistent optimizer so training can RESUME
    ResumableTraining   per-mechanism persistent Adam + training_state_dict /
                        load_training_state_dict (weights, optimizer, version)
    count_linear_flops  forward-hook FLOP count over Linear and GRUCell layers

STATIC DIMS (fixed 2026-10-03; found by the independent A/B verifier,
experiments/foundation_ab/RESULTS_stage6_8.md, bug 1). The learned
mechanisms predict the CHANGE of every entity's pos/vel. In the box fixture
the 4 walls are entities whose 16 change dims are EXACTLY zero, so a
Gaussian head can drive their variance to VAR_FLOOR and each such dim pays
0.5*log(1e-4) = -4.6 nats: about -66 of a training NLL whose floor was
-73.7, against -8.4 (IN) / -1.8 (MLP-128) from the 12 ball dims that are
actually being learned. Chasing that unbounded-below term made training
non-monotone and, on small data, divergent: per-epoch loss
[9.5, 4.2, -2.1, -10.5, -15.9, 35.8, 283.1, 332.1] (seed 11, 360
transitions, lr 1e-3). Now a dim is STATIC when its change is identically
zero over the training data (|next - current| <= STATIC_ATOL, i.e. it
never moved; a nonzero-but-constant change is NOT treated as static,
because "always moved by c" does not generalise the way "never moved"
does), or when the caller DECLARES it static. A static dim
    * stays an INPUT (the network still reads wall positions),
    * is excluded from the loss (no gradient flows into its head slot),
    * is predicted as the identity: mean = current value exactly,
      variance = VAR_FLOOR * out_std.sd**2 (what a perfectly converged
      head would have emitted; only the scored dims ever enter metrics).
ESCAPE PATH (CLAUDE.md §4.1 — the mask must not latch): the head keeps an
output slot for every dim, and the observed mask can only SHRINK — a later
fit on data where a "static" dim moves makes it dynamic for good and it is
learned from then on, with no rebuild. A DECLARED static dim is a caller
assertion; data that contradicts it raises ContractError instead of being
silently learned or silently ignored, and `undeclare_static` withdraws it
(so the refusal is not a latch either).

RESUMABLE TRAINING (bug 2). `fit` used to build a fresh Adam on every call,
so chunked or continual fitting reset the moment estimates each time (the
FLOP-matched MLP's loss rose 9.3 -> 310 across resets at toy scale). The
optimizer is now created on the FIRST fit and reused by every later fit (lr
and weight_decay are updated in place to the values that fit asked for),
and `training_state_dict()` / `load_training_state_dict()` carry weights,
optimizer moments, version, support range and static mask, so a restored
mechanism continues bit-for-bit where the saved one would have.
`state_dict()` is unchanged: it is the WEIGHTS-only view that
runtime.SnapshotRegistry publishes for predictions.

Seeds: model construction and training run inside torch.random.fork_rng, so
fitting a shadow model never advances the live learner's torch RNG.
"""

from __future__ import annotations

import contextlib
import copy
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VAR_FLOOR = 1e-4          # in standardised units
STATIC_ATOL = 1e-9        # |change| at or below this on every row = never moved
TRAIN_STATE_SCHEMA = 1


def mlp(i: int, h: int, o: int, layers: int = 2) -> nn.Sequential:
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, h), nn.ELU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class Standardizer(nn.Module):
    def __init__(self, dim: int, center: bool = True):
        super().__init__()
        self.center = center
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    def fit(self, x: torch.Tensor) -> None:
        if bool(self.fitted):
            return
        x = x.reshape(-1, x.shape[-1]).double()
        if self.center:
            self.mu.copy_(x.mean(0).float())
        sd = x.std(0) if x.shape[0] > 1 else torch.ones(x.shape[-1], dtype=x.dtype)
        # A feature constant in training (a wall's position, a static wall's
        # zero change) gets scale 1, NOT a tiny floor: a 1e-6 floor turns any
        # later deviation into a 1e6-sized input (measured: an MLP's
        # prediction moved by 1.5e5 m when the walls were translated 3.7 m).
        self.sd.copy_(torch.where(sd < 1e-6, torch.ones_like(sd), sd).float())
        self.fitted.fill_(True)

    def forward(self, x):
        return (x - self.mu) / self.sd


def split_head(raw: torch.Tensor, dc: int, g: int, k: int):
    """raw (..., 2*dc + g*k) -> mean (..., dc), var (..., dc), logits (..., g, k)."""
    mu = raw[..., :dc]
    var = F.softplus(raw[..., dc:2 * dc]) + VAR_FLOOR
    logits = raw[..., 2 * dc:].reshape(raw.shape[:-1] + (g, k)) if g else None
    return mu, var, logits


def head_loss(mu, var, logits, y, ev, static: Optional[torch.Tensor] = None
              ) -> torch.Tensor:
    """Gaussian NLL summed over continuous dims, mean over rows, plus event
    cross-entropy. `static` (bool, (Dc,)) masks dims out of the NLL entirely
    (module docstring: a never-moving dim would otherwise dominate)."""
    per = torch.log(var) + (y - mu) ** 2 / var
    if static is not None:
        per = per * (~static).to(per.dtype)
    nll = 0.5 * per.sum(-1).mean()
    if logits is not None and ev is not None:
        obs = ev >= 0
        if bool(obs.any()):
            ce = F.cross_entropy(logits[obs], ev[obs], reduction="sum") / mu.shape[0]
            nll = nll + ce
    return nll


def update_static_mask(change: np.ndarray, prev: Optional[np.ndarray],
                       declared: np.ndarray, who: str = "mechanism") -> np.ndarray:
    """-> bool (Dc,) static mask after a fit on rows `change` (B, Dc).

    prev None (first fit): static = never moved in this data, OR declared.
    Later fits: static = (prev AND never moved in this data) OR declared —
    a dim seen moving once is dynamic for good (the mask only shrinks; the
    escape path of module docstring). Declared dims that moved -> error."""
    change = np.asarray(change, dtype=np.float64)
    if change.ndim != 2 or change.shape[1] != declared.shape[0]:
        raise ContractError(f"{who}: change {change.shape} vs {declared.shape[0]} dims")
    still = np.all(np.abs(change) <= STATIC_ATOL, 0)
    bad = np.flatnonzero(declared & ~still)
    if bad.size:
        raise ContractError(f"{who}: dims {bad.tolist()} were DECLARED static but "
                            f"change in the training data (max |change| "
                            f"{np.abs(change[:, bad]).max():.3g}); fix the data or "
                            f"undeclare_static({bad.tolist()})")
    obs = still if prev is None else (np.asarray(prev, bool) & still)
    return obs | declared


def pass_static(mean: np.ndarray, var: np.ndarray, current: np.ndarray,
                static: np.ndarray, sd: np.ndarray):
    """Static dims: mean = current value exactly, var = VAR_FLOOR * sd**2."""
    if static.any():
        mean = np.array(mean, dtype=np.float64, copy=True)
        var = np.array(var, dtype=np.float64, copy=True)
        mean[..., static] = current[..., static]
        var[..., static] = VAR_FLOOR * sd[static] ** 2
    return mean, var


def check_static_dims(dims: Sequence[int], n: int, who: str) -> np.ndarray:
    idx = np.asarray(list(dims), dtype=np.int64).reshape(-1)
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise ContractError(f"{who}: static dims {idx.tolist()} out of range [0, {n})")
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


@contextlib.contextmanager
def seeded(seed: Optional[int]):
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(int(seed))
        yield


def train(params, loss_fn: Callable[[np.ndarray], torch.Tensor], n: int, *,
          epochs: int, batch_size: int, lr: float, seed: int,
          weight_decay: float = 0.0,
          optimizer: Optional[torch.optim.Optimizer] = None) -> List[float]:
    """loss_fn(index array) -> scalar loss. Returns per-epoch mean losses.
    `optimizer` (over exactly `params`) is reused — its moments carry over —
    with lr / weight_decay set to this call's values; None = a fresh Adam."""
    rng = np.random.default_rng(seed)
    if optimizer is None:
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        opt = optimizer
        for g in opt.param_groups:
            g["lr"], g["weight_decay"] = float(lr), float(weight_decay)
    hist = []
    with seeded(seed):
        for ep in range(int(epochs)):
            perm = rng.permutation(n)
            tot, cnt = 0.0, 0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                loss = loss_fn(idx)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch {ep}")
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()
                tot += float(loss.detach()) * len(idx)
                cnt += len(idx)
            hist.append(tot / max(cnt, 1))
    return hist


def count_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def count_flops(module: nn.Module, run: Callable[[], None]) -> int:
    """Multiply-adds x2 of every Linear / GRUCell call during `run()`."""
    total = [0]

    def lin(m, inp, out):
        total[0] += 2 * int(np.prod(inp[0].shape[:-1])) * m.in_features * m.out_features

    def gru(m, inp, out):
        rows = int(np.prod(inp[0].shape[:-1]))
        total[0] += 2 * rows * 3 * m.hidden_size * (m.input_size + m.hidden_size)

    hooks = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin))
        elif isinstance(m, nn.GRUCell):
            hooks.append(m.register_forward_hook(gru))
    try:
        with torch.no_grad():
            run()
    finally:
        for h in hooks:
            h.remove()
    return total[0]


def t(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


# ======================================================================
# Shared learned-mechanism base
# ======================================================================

from ..contracts import ContractError  # noqa: E402
from .api import BaseMechanism, MixedOutcome, TransitionBatch  # noqa: E402


class ResumableTraining:
    """Mixin for torch mechanisms: ONE optimizer for the mechanism's life,
    and a full training state that can be saved and restored.

    Subclasses provide `self.module` and `_trainable_params()` (a fixed,
    ordered parameter list — the optimizer state is keyed by position).
    Attributes named in `_TRAIN_ATTRS` are carried in the training state."""

    _TRAIN_ATTRS = ("version", "_lo", "_hi", "last_fit")
    _opt: Optional[torch.optim.Optimizer] = None

    def _trainable_params(self) -> List[torch.nn.Parameter]:   # pragma: no cover
        raise NotImplementedError

    def _optimizer(self, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
        if self._opt is None:
            self._opt = torch.optim.Adam(self._trainable_params(), lr=float(lr),
                                         weight_decay=float(weight_decay))
        return self._opt

    def reset_optimizer(self) -> None:
        """Explicit opt-out: the next fit starts a fresh Adam."""
        self._opt = None

    def training_state_dict(self) -> Dict[str, Any]:
        return {"schema": TRAIN_STATE_SCHEMA, "mechanism_id": self.mechanism_id,
                "class": type(self).__name__,
                "module": {k: v.detach().clone() for k, v in self.module.state_dict().items()},
                "optimizer": None if self._opt is None else copy.deepcopy(self._opt.state_dict()),
                "attrs": {a: copy.deepcopy(getattr(self, a)) for a in self._TRAIN_ATTRS}}

    def load_training_state_dict(self, d: Dict[str, Any]) -> None:
        if not isinstance(d, dict) or d.get("schema") != TRAIN_STATE_SCHEMA:
            raise ContractError(f"{self.mechanism_id}: unsupported training state schema "
                                f"{d.get('schema') if isinstance(d, dict) else type(d)}")
        if d["mechanism_id"] != self.mechanism_id or d["class"] != type(self).__name__:
            raise ContractError(f"training state of {d['class']}:{d['mechanism_id']} cannot "
                                f"load into {type(self).__name__}:{self.mechanism_id}")
        self.module.load_state_dict(d["module"], strict=True)
        self._opt = None
        if d["optimizer"] is not None:
            opt = torch.optim.Adam(self._trainable_params())
            opt.load_state_dict(copy.deepcopy(d["optimizer"]))
            self._opt = opt
        for a, v in d["attrs"].items():
            if a not in self._TRAIN_ATTRS:
                raise ContractError(f"unknown training attribute {a!r}")
            setattr(self, a, copy.deepcopy(v))


class TorchMechanism(ResumableTraining, BaseMechanism):
    """A learned mechanism predicting the standardised CHANGE of the
    continuous state (next - current) plus per-entity event logits.

    Subclasses build `self.module` (an nn.Module holding every parameter and
    standardiser) in `_build`, and implement:
        _prepare(state, action, dt) -> dict of float tensors, batch-first
        _fit_stats(prep)            -> fit input standardisers (first fit only)
        _forward(prep)              -> mu_n (B, Dc), var_n (B, Dc), logits (B, N, K)

    Static outcome dims (module docstring) live in two module buffers, so
    they travel with state_dict / snapshots: `static_declared` (set by
    `declare_static`) and `static` (the mask in force, refreshed per fit).
    """

    def __init__(self, mechanism_id, n_entities, dim, attr_dim, action_dim, classes,
                 assumptions=None, seed: int = 0, hidden: int = 64):
        super().__init__(mechanism_id, n_entities, dim, attr_dim, action_dim, classes,
                         assumptions)
        self.seed, self.hidden = int(seed), int(hidden)
        with seeded(self.seed):
            self.module = self._build()
            self.module.add_module("out_std", Standardizer(self.layout.Dc))
        Dc = self.layout.Dc
        self.module.register_buffer("static", torch.zeros(Dc, dtype=torch.bool))
        self.module.register_buffer("static_declared", torch.zeros(Dc, dtype=torch.bool))

    def _build(self) -> nn.Module:                         # pragma: no cover
        raise NotImplementedError

    def _trainable_params(self):
        return list(self.module.parameters())

    def declare_static(self, dims: Sequence[int]) -> None:
        """Declare continuous outcome dims (indices into layout.cont_names)
        that never change: predicted as identity, never trained. Adds to
        any earlier declaration; takes effect at once and on every fit."""
        m = check_static_dims(dims, self.layout.Dc, self.mechanism_id)
        self.module.static_declared |= torch.from_numpy(m)
        self.module.static |= torch.from_numpy(m)

    def undeclare_static(self, dims: Sequence[int]) -> None:
        """Withdraw a declaration (the escape path from the ContractError a
        contradicted declaration raises). The dims stay passed through until
        the next fit, which re-derives them from the data."""
        m = check_static_dims(dims, self.layout.Dc, self.mechanism_id)
        self.module.static_declared &= ~torch.from_numpy(m)

    @property
    def static_dims(self) -> np.ndarray:
        """Indices of the continuous outcome dims currently passed through."""
        return np.flatnonzero(self.module.static.numpy())

    def _out(self, mu_n, var_n, logits, state):
        sd = self.module.out_std.sd.double().numpy()
        m = self.module.out_std.mu.double().numpy()
        cur = state.continuous()
        mean = cur + mu_n.double().numpy() * sd + m
        var = var_n.double().numpy() * sd ** 2
        mean, var = pass_static(mean, var, cur, self.module.static.numpy(), sd)
        probs = (torch.softmax(logits.double(), -1).numpy() if logits is not None
                 else np.zeros((state.B, 0, 0)))
        return MixedOutcome.gaussian_categorical(self.layout, mean, var, probs)

    def _predict(self, state, action, dt):
        # An unfitted model still returns a valid distribution; it is
        # prediction_record that refuses to file it (version 0).
        self.module.eval()
        with torch.no_grad():
            prep = self._prepare(state, action, dt)
            mu_n, var_n, logits = self._forward(prep)
        return self._out(mu_n, var_n, logits, state)

    def _fit(self, batch: TransitionBatch, epochs: int = 60, batch_size: int = 128,
             lr: float = 2e-3, seed: Optional[int] = None, weight_decay: float = 0.0):
        prep = self._prepare(batch.state, batch.action, batch.dt)
        self._fit_stats(prep)
        change = batch.next_state.continuous() - batch.state.continuous()
        static = update_static_mask(
            change, None if self.version == 0 else self.module.static.numpy(),
            self.module.static_declared.numpy(), self.mechanism_id)
        self.module.static.copy_(torch.from_numpy(static))
        st = self.module.static.clone()
        y = t(change)
        self.module.out_std.fit(y)
        y = self.module.out_std(y)
        ev = torch.as_tensor(batch.events, dtype=torch.long)
        self.module.train()

        def loss_fn(idx):
            sub = {k: v[idx] for k, v in prep.items()}
            mu, var, logits = self._forward(sub)
            return head_loss(mu, var, logits, y[idx], ev[idx], st)

        hist = train(self._trainable_params(), loss_fn, len(batch), epochs=epochs,
                     batch_size=batch_size, lr=lr,
                     seed=self.seed + 1000 * (self.version + 1) if seed is None else seed,
                     weight_decay=weight_decay,
                     optimizer=self._optimizer(lr, weight_decay))
        self.module.eval()
        return {"loss_first": hist[0], "loss_last": hist[-1], "epochs": len(hist),
                "loss_hist": [float(h) for h in hist], "static_dims": int(static.sum())}

    def state_dict(self):
        """Weights (+ standardisers, static masks) only — the snapshot view.
        The resumable view with optimizer moments is training_state_dict()."""
        return self.module.state_dict()

    def resources(self, state=None, action=None, dt=None) -> Dict[str, int]:
        r = {"params": count_params(self.module)}
        if state is not None:
            prep = self._prepare(state, action, dt)
            r["flops_per_batch"] = count_flops(self.module, lambda: self._forward(prep))
            r["flops_per_sample"] = r["flops_per_batch"] // max(state.B, 1)
        return r
