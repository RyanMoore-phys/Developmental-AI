"""Bounded bootstrapped ensemble of mechanism predictors (plan Stage 7.1-7.2).

K <= 8 members, each built by `factory(i)` and fitted on a bootstrap
resample of the training transitions. The predictive distribution is the
equal-weight MIXTURE of the members' predictives (a MixedOutcome with
M = K x member components), and its variance splits exactly into

    aleatoric = mean over members of each member's predictive variance
                (noise the members agree is irreducible — includes
                observation noise; a member cannot tell the two apart)
    epistemic = variance over members of the members' predictive means
                (disagreement: what more data in this region would reduce)

`decompose(...)` returns both. The split is only as good as the members'
diversity: bootstrap + independent seeds is a cheap approximation to a
posterior over weights, not one. In particular AGREEMENT IS NOT
CORRECTNESS (plan Stage 7 gate): members that share an inductive bias agree
confidently on data whose rule has changed. Use change.py on the predictive
log-likelihood to catch that; never treat low epistemic variance as proof.

Bootstrap unit: i.i.d. transitions by default. Adjacent transitions of one
episode are correlated, so i.i.d. resampling UNDERSTATES disagreement; pass
`groups` (e.g. episode ids, one per transition) to resample whole groups.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..contracts import UNKNOWN, ContractError, is_missing
from ..mechanisms.api import MixedOutcome, TransitionBatch

MAX_MEMBERS = 8


class BootstrapEnsemble:
    def __init__(self, factory: Callable[[int], Any], k: int = 5, seed: int = 0,
                 bootstrap: bool = True, mechanism_id: Optional[str] = None):
        k = int(k)
        if not 2 <= k <= MAX_MEMBERS:
            raise ContractError(f"ensemble size must be in [2, {MAX_MEMBERS}], got {k} "
                                f"(bounded: each member costs a full model)")
        self.members = [factory(i) for i in range(k)]
        ids = [m.mechanism_id for m in self.members]
        if len(set(ids)) != k:
            raise ContractError(f"members need distinct mechanism_ids, got {ids}")
        lays = {m.layout for m in self.members}
        if len(lays) != 1:
            raise ContractError("members predict different outcome layouts")
        self.layout = lays.pop()
        self.mechanism_id = mechanism_id or f"ensemble[{ids[0]}x{k}]"
        self.seed, self.bootstrap = int(seed), bool(bootstrap)
        self.version = 0
        self.assumptions = {"members": k, "bootstrap": self.bootstrap,
                            **{f"member.{a}": v for a, v in self.members[0].assumptions.items()
                               if isinstance(v, (bool, int, float, str))}}
        self.last_fit: Dict[str, Any] = {}

    def model_versions(self) -> Dict[str, int]:
        out = {self.mechanism_id: int(self.version)}
        out.update({m.mechanism_id: int(m.version) for m in self.members})
        return out

    def fit(self, batch: TransitionBatch, groups: Optional[Sequence] = None, **kw):
        n = len(batch)
        rng = np.random.default_rng(self.seed + 7919 * (self.version + 1))
        reps, secs = [], 0.0
        g = None if groups is None else np.asarray(groups)
        if g is not None and len(g) != n:
            raise ContractError("groups must have one entry per transition")
        for m in self.members:
            if not self.bootstrap:
                idx = np.arange(n)
            elif g is None:
                idx = rng.integers(0, n, n)
            else:
                u = np.unique(g)
                pick = rng.choice(u, len(u), replace=True)
                idx = np.concatenate([np.flatnonzero(g == p) for p in pick])
            r = m.fit(batch.subset(idx), **kw)
            reps.append(r)
            secs += r.get("train_seconds", 0.0)
        self.version += 1
        self.last_fit = {"members": reps, "train_seconds": secs, "version": self.version}
        return self.last_fit

    def predict(self, state, action, dt) -> MixedOutcome:
        return MixedOutcome.mixture([m.predict(state, action, dt) for m in self.members])

    def decompose(self, state, action, dt) -> Dict[str, np.ndarray]:
        d = self.predict(state, action, dt)
        a, e = d.decompose_variance()
        return {"aleatoric": a, "epistemic": e, "total": a + e}

    def applicability(self, state):
        vals = [m.applicability(state) for m in self.members]
        if any(is_missing(v) for v in vals):
            return UNKNOWN
        return np.mean(vals, 0)

    def rollout(self, state, actions, dts, mode: str = "mean", n_samples: int = 1,
                rng: Optional[np.random.Generator] = None) -> List[MixedOutcome]:
        """Each member rolls out on its own (its own beliefs compound); step h
        is the equal-weight mixture of the members' step-h predictives."""
        per = [m.rollout(state, actions, dts, mode=mode, n_samples=n_samples, rng=rng)
               for m in self.members]
        return [MixedOutcome.mixture([p[h] for p in per]) for h in range(len(per[0]))]

    def state_dict(self):
        out = {}
        for i, m in enumerate(self.members):
            for k, v in m.state_dict().items():
                out[f"member{i}.{k}"] = v
        return out

    def training_state_dict(self) -> Dict[str, Any]:
        """Resumable state: every member's weights + optimizer moments
        (mechanisms._nets.ResumableTraining) and the ensemble version (which
        seeds the next bootstrap draw)."""
        for m in self.members:
            if not hasattr(m, "training_state_dict"):
                raise ContractError(f"member {m.mechanism_id} has no resumable training state")
        return {"schema": 1, "mechanism_id": self.mechanism_id, "version": int(self.version),
                "members": [m.training_state_dict() for m in self.members]}

    def load_training_state_dict(self, d: Dict[str, Any]) -> None:
        if d.get("schema") != 1 or d.get("mechanism_id") != self.mechanism_id \
                or len(d.get("members", ())) != len(self.members):
            raise ContractError(f"{self.mechanism_id}: incompatible ensemble training state")
        for m, s in zip(self.members, d["members"]):
            m.load_training_state_dict(s)
        self.version = int(d["version"])

    def resources(self, *a, **k) -> Dict[str, Any]:
        rs = [m.resources(*a, **k) for m in self.members]
        return {key: sum(r[key] for r in rs) for key in rs[0]
                if all(isinstance(r.get(key), (int, float)) for r in rs)}
