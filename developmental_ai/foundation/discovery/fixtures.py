"""Synthetic worlds with KNOWN hidden rules for Stage 9 tests.

Every world returns per-row variable values and which variables an agent
ACTION set on that row; `make_episode` turns that into rows + contracts
Action records (payload {"sets": ...}) and builds the EvidenceTable through
EvidenceTable.from_records — so interventions reach discovery only by the
same route real ones would.

    hidden_switch  y = 2x if h == 1 else -1                (h observed, unused
                                                            by the root mechanism)
    threshold      y = 3 if temp >= 0.5 else -1
    lever_door     y = 2*lever if door == 1 else 0          (lever set by the
                                                            agent; door is not)
    spurious       y = 2C, S = C + noise observationally; C hidden; on
                   intervened rows the agent sets S at random -> no effect
    confounded     y = 1.0*X + 2U, X = U + noise observationally, U hidden;
                   intervened rows set X at random -> effect 1.0, while the
                   observational slope is ~2.6
    regime         "A": y = 1 + 2x;  "B": y = 1 + 2x if h == 1 else 1 - 2x
                   (a STRUCTURAL change: x's effect now depends on h)
    param_change   "A": y = 1 + 2x;  "A2": y = 1 + 3x  (parameter-only
                   change: the control for the Stage 9 gate)
    chain          z = 2x, y = 3z; z unobserved on `z_missing` of rows
Noise is Gaussian, sd 0.2 unless stated.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from ..contracts import Action
from .table import EvidenceTable, is_final_heldout

NOISE = 0.2


def _hidden_switch(rng, n, **kw):
    x, h, d = rng.uniform(-1, 1, n), rng.integers(0, 2, n).astype(float), rng.uniform(-1, 1, n)
    y = np.where(h == 1, 2 * x, -1.0) + rng.normal(0, NOISE, n)
    return {"x": x, "h": h, "d": d, "y": y}, {}


def _threshold(rng, n, **kw):
    t, x = rng.uniform(0, 1, n), rng.uniform(-1, 1, n)
    y = np.where(t >= 0.5, 3.0, -1.0) + rng.normal(0, NOISE, n)
    return {"temp": t, "x": x, "y": y}, {}


def _lever_door(rng, n, **kw):
    lever = rng.integers(0, 2, n).astype(float)
    door = rng.integers(0, 2, n).astype(float)
    z = rng.uniform(-1, 1, n)
    y = np.where(door == 1, 2 * lever, 0.0) + rng.normal(0, NOISE, n)
    return {"lever": lever, "door": door, "z": z, "y": y}, {"lever": np.ones(n, bool)}


def _spurious(rng, n, intervene_frac=0.3, **kw):
    c = rng.normal(0, 1, n)
    iv = rng.uniform(size=n) < intervene_frac
    s = np.where(iv, np.round(rng.normal(0, 1, n), 3), c + rng.normal(0, 0.3, n))
    d = rng.uniform(-1, 1, n)
    y = 2 * c + rng.normal(0, NOISE, n)
    return {"S": s, "d": d, "y": y}, {"S": iv}


def _confounded(rng, n, intervene_frac=0.4, **kw):
    u = rng.normal(0, 1, n)
    iv = rng.uniform(size=n) < intervene_frac
    x = np.where(iv, np.round(rng.normal(0, 1.1, n), 3), u + rng.normal(0, 0.5, n))
    y = 1.0 * x + 2 * u + rng.normal(0, NOISE, n)
    return {"X": x, "y": y}, {"X": iv}


def _regime(rng, n, regime="A", **kw):
    x, h = rng.uniform(-1, 1, n), rng.integers(0, 2, n).astype(float)
    if regime == "A":
        mu = 1 + 2 * x
    elif regime == "B":
        mu = np.where(h == 1, 1 + 2 * x, 1 - 2 * x)
    else:
        raise ValueError(f"regime {regime!r}")
    return {"x": x, "h": h, "y": mu + rng.normal(0, NOISE, n)}, {}


def _param_change(rng, n, regime="A", **kw):
    x, h = rng.uniform(-1, 1, n), rng.integers(0, 2, n).astype(float)
    slope = {"A": 2.0, "A2": 3.0}[regime]
    return {"x": x, "h": h, "y": 1 + slope * x + rng.normal(0, NOISE, n)}, {}


def _chain(rng, n, z_missing=0.5, **kw):
    x = rng.uniform(-1, 1, n)
    z = 2 * x + rng.normal(0, 0.1, n)
    y = 3 * z + rng.normal(0, NOISE, n)
    zobs = np.where(rng.uniform(size=n) < z_missing, np.nan, z)
    return {"x": x, "z": zobs, "y": y}, {}


WORLDS: Dict[str, Callable] = {
    "hidden_switch": _hidden_switch, "threshold": _threshold, "lever_door": _lever_door,
    "spurious": _spurious, "confounded": _confounded, "regime": _regime,
    "param_change": _param_change, "chain": _chain,
}


def make_episode(world: str, episode: str, scope: str, rng: np.random.Generator,
                 n_rows: int = 30, target: str = "y", **kw) -> EvidenceTable:
    cols, iv = WORLDS[world](rng, n_rows, **kw)
    rows, actions = [], []
    for i in range(n_rows):
        r = {"episode": episode, "seq": i}
        r.update({k: (None if np.isnan(v[i]) else float(v[i])) for k, v in cols.items()})
        rows.append(r)
        sets = {k: float(cols[k][i]) for k, m in iv.items() if m[i]}
        if sets:
            actions.append(Action(environment="fixture", stream=world, episode=episode,
                                  seq=i, spec_id="fixture-set", command=0, duration=0.0,
                                  t_dispatch=float(i), payload={"sets": sets,
                                                                "actor": "agent"}))
    return EvidenceTable.from_records(rows, target, scope, actions)


def make_split_episodes(world: str, scope: str, seed: int, n_dev: int, n_heldout: int = 0,
                        prefix: str = "e", start: int = 0, n_rows: int = 30, **kw
                        ) -> Tuple[List[EvidenceTable], List[EvidenceTable]]:
    """Generate episodes in id order, routing each by runtime.splits (final
    held-out vs dev) until both quotas are filled."""
    rng = np.random.default_rng(seed)
    dev: List[EvidenceTable] = []
    held: List[EvidenceTable] = []
    i = start
    while len(dev) < n_dev or len(held) < n_heldout:
        eid = f"{prefix}{i}"
        i += 1
        ho = is_final_heldout(scope, eid)
        if (ho and len(held) >= n_heldout) or (not ho and len(dev) >= n_dev):
            continue
        t = make_episode(world, eid, scope, rng, n_rows, **kw)
        (held if ho else dev).append(t)
    return dev, held


__all__ = ["WORLDS", "make_episode", "make_split_episodes", "NOISE"]
