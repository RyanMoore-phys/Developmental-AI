"""EvidenceTable — tabular evidence for mechanism discovery, with episode
identity and an explicit record of which values an AGENT ACTION set.

Why a table and not the EvidenceStore directly: discovery fits small
closed-form transition forms over named scalar variables, and needs three
things the raw stream does not make cheap — (1) column access by variable,
(2) the episode of every row (splits are by episode, runtime.splits), and
(3) for every variable, a boolean mask of rows whose value was SET BY AN
ACTION the agent took (contracts.Action records whose payload declares
`sets`). (3) is the only route to an `interventional` causal label
(discovery.causal); nothing else can mark a row intervened.

Missing values are NaN in the float columns and are never scored as zero:
every consumer drops rows whose needed variables are NaN and reports it.
UNKNOWN / ABSENT / INAPPLICABLE / None in `from_records` all become NaN —
they stay distinct in the source records; here they are all "no reading".

Splits (plan §7.2.2, runtime.splits):
    * FINAL HELD-OUT episodes (runtime.splits.assign_split with the default
      salt) are refused by discovery outright (`require_dev`): selecting
      structure on them would make them no longer held out.
    * Within dev, a second deterministic episode-level split with its own
      salt (VALIDATION_SALT) separates FIT episodes from VALIDATION
      episodes; `assert_no_leak` is checked on every split.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..contracts import Action, ContractError, is_missing
from ..runtime.splits import (DEV, HELDOUT, SplitLeakError, assert_no_leak,
                              assign_split)

VALIDATION_SALT = "skybot-foundation-discovery-validation-v1"
DEFAULT_VAL_FRACTION = 0.4
MAX_DISCRETE_LEVELS = 8


class DiscoveryError(ContractError):
    """Invalid input to discovery (shape, type, missing column...)."""


def _as_float(v) -> float:
    if v is None or is_missing(v):
        return float("nan")
    if isinstance(v, bool):
        return float(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise DiscoveryError(f"value {v!r} is not numeric")
    if np.isinf(f):
        raise DiscoveryError(f"infinite value {v!r}")
    return f


class EvidenceTable:
    """Columns of float64 (NaN = no reading), one target, per-row episode."""

    def __init__(self, columns: Mapping[str, Any], target: str,
                 episodes: Sequence[Any], scope: str,
                 intervened: Optional[Mapping[str, Any]] = None):
        if not isinstance(scope, str) or not scope.strip():
            raise DiscoveryError("scope must be a non-empty string")
        if target not in columns:
            raise DiscoveryError(f"target {target!r} not among columns")
        eps = np.asarray([str(e) for e in episodes], dtype=object)
        n = len(eps)
        cols: Dict[str, np.ndarray] = {}
        for k, v in columns.items():
            if not isinstance(k, str) or not k or "|" in k:
                raise DiscoveryError(f"bad column name {k!r}")
            a = np.asarray(v, dtype=np.float64)
            if a.shape != (n,):
                raise DiscoveryError(f"column {k!r} shape {a.shape} != ({n},)")
            if np.any(np.isinf(a)):
                raise DiscoveryError(f"column {k!r} has inf")
            cols[k] = a
        iv: Dict[str, np.ndarray] = {}
        for k, m in (intervened or {}).items():
            if k not in cols:
                raise DiscoveryError(f"intervened mask for unknown column {k!r}")
            if k == target:
                raise DiscoveryError("the target cannot be intervened on: an action "
                                     "setting the outcome makes it no outcome")
            m = np.asarray(m, dtype=bool)
            if m.shape != (n,):
                raise DiscoveryError(f"intervened[{k!r}] shape {m.shape} != ({n},)")
            iv[k] = m
        self.columns = cols
        # Per-column caches for levels()/is_discrete(): both are np.unique
        # sorts, and enumeration asked for them ~3x per variable per call
        # (verifier B1: 1.4 s of sorting on 150 vars x 60k rows). Columns are
        # never mutated after construction — select/with_column/concat build
        # NEW tables — so a cache keyed by column name is exact.
        self._levels: Dict[str, np.ndarray] = {}
        self._discrete: Dict[str, bool] = {}
        self.target = target
        self.episodes = eps
        self.scope = scope
        self.intervened = iv

    # ---- construction ------------------------------------------------------
    @classmethod
    def from_records(cls, rows: Sequence[Mapping[str, Any]], target: str,
                     scope: str, actions: Iterable[Action] = ()) -> "EvidenceTable":
        """rows: dicts with 'episode', 'seq' and variable values. An Action
        record (same episode, same seq) whose payload {"sets": {var: value}}
        names a variable AND whose value equals the row's reading marks that
        row intervened for that variable. payload["actor"] other than
        "agent" is refused as a source of interventions (evaluator/scripted
        settings are not the agent's experiments)."""
        if not rows:
            raise DiscoveryError("no rows")
        names = sorted({k for r in rows for k in r if k not in ("episode", "seq")})
        sets: Dict[Tuple[str, int], Dict[str, float]] = {}
        for a in actions:
            if not isinstance(a, Action):
                raise DiscoveryError(f"actions must be contracts.Action, got {type(a).__name__}")
            if a.payload.get("actor", "agent") != "agent":
                continue
            s = a.payload.get("sets")
            if not s:
                continue
            if not isinstance(s, Mapping):
                raise DiscoveryError("Action.payload['sets'] must be a mapping")
            sets[(a.episode, a.seq)] = {k: _as_float(v) for k, v in s.items()}
        cols = {k: np.full(len(rows), np.nan) for k in names}
        iv = {k: np.zeros(len(rows), bool) for k in names if k != target}
        eps = []
        for i, r in enumerate(rows):
            if "episode" not in r or "seq" not in r:
                raise DiscoveryError(f"row {i} lacks episode/seq")
            eps.append(str(r["episode"]))
            for k in names:
                if k in r:
                    cols[k][i] = _as_float(r[k])
            act = sets.get((str(r["episode"]), int(r["seq"])))
            if act:
                for k, v in act.items():
                    if k == target or k not in iv:
                        continue
                    if np.isfinite(v) and cols[k][i] == v:
                        iv[k][i] = True
        return cls(cols, target, eps, scope,
                   {k: m for k, m in iv.items() if m.any()})

    # ---- access ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def y(self) -> np.ndarray:
        return self.columns[self.target]

    @property
    def variables(self) -> List[str]:
        return sorted(k for k in self.columns if k != self.target)

    def col(self, name: str) -> np.ndarray:
        if name not in self.columns:
            raise DiscoveryError(f"unknown variable {name!r}")
        return self.columns[name]

    def intervened_mask(self, name: str) -> np.ndarray:
        return self.intervened.get(name, np.zeros(len(self), bool))

    def unique_episodes(self) -> List[str]:
        seen, out = set(), []
        for e in self.episodes:
            if e not in seen:
                seen.add(e)
                out.append(e)
        return out

    def levels(self, name: str) -> np.ndarray:
        """Sorted distinct finite values (cached; a read-only array)."""
        lv = self._levels.get(name)
        if lv is None:
            v = self.col(name)
            lv = np.unique(v[np.isfinite(v)])
            lv.setflags(write=False)
            self._levels[name] = lv
        return lv

    def n_distinct_at_least(self, name: str, k: int) -> bool:
        """True iff the column has >= k distinct finite values (k <= 2 is
        answered in O(n) without sorting)."""
        if k <= 2 and name not in self._levels:
            v = self.col(name)
            v = v[np.isfinite(v)]
            if k <= 0:
                return True
            if k == 1:
                return len(v) > 0
            return len(v) > 0 and bool(v.min() < v.max())
        return len(self.levels(name)) >= k

    def is_discrete(self, name: str) -> bool:
        """<= MAX_DISCRETE_LEVELS distinct finite values, all integers
        (cached). A non-integer column is rejected in O(n) without the sort."""
        d = self._discrete.get(name)
        if d is None:
            if name in self._levels:
                lv = self._levels[name]
                d = 0 < len(lv) <= MAX_DISCRETE_LEVELS and bool(np.all(lv == np.round(lv)))
            else:
                v = self.col(name)
                v = v[np.isfinite(v)]
                if len(v) == 0 or not bool(np.all(v == np.round(v))):
                    d = False
                else:
                    lv = self.levels(name)
                    d = 0 < len(lv) <= MAX_DISCRETE_LEVELS
            self._discrete[name] = d
        return d

    def select(self, mask) -> "EvidenceTable":
        m = np.asarray(mask)
        return EvidenceTable({k: v[m] for k, v in self.columns.items()}, self.target,
                             self.episodes[m], self.scope,
                             {k: v[m] for k, v in self.intervened.items()})

    def with_column(self, name: str, values) -> "EvidenceTable":
        cols = dict(self.columns)
        cols[name] = np.asarray(values, float)
        iv = {k: v for k, v in self.intervened.items() if k != name}
        return EvidenceTable(cols, self.target, self.episodes, self.scope, iv)

    def by_episodes(self, eps: Iterable[str]) -> "EvidenceTable":
        s = set(eps)
        return self.select(np.array([e in s for e in self.episodes], bool))

    @staticmethod
    def concat(tables: Sequence["EvidenceTable"]) -> "EvidenceTable":
        if not tables:
            raise DiscoveryError("concat of no tables")
        t0 = tables[0]
        for t in tables[1:]:
            if t.target != t0.target or t.scope != t0.scope or set(t.columns) != set(t0.columns):
                raise DiscoveryError("concat: tables differ in target/scope/columns")
        names = set().union(*[set(t.intervened) for t in tables])
        return EvidenceTable(
            {k: np.concatenate([t.columns[k] for t in tables]) for k in t0.columns},
            t0.target, np.concatenate([t.episodes for t in tables]), t0.scope,
            {k: np.concatenate([t.intervened_mask(k) for t in tables]) for k in names})

    # ---- splits ------------------------------------------------------------
    def require_dev(self) -> "EvidenceTable":
        """Refuse any FINAL held-out episode (runtime.splits default salt)."""
        bad = [e for e in self.unique_episodes()
               if assign_split(self.scope, e) == HELDOUT]
        if bad:
            raise SplitLeakError(
                f"{len(bad)} final held-out episode(s) offered to discovery "
                f"(e.g. {self.scope}/{bad[0]}); structure selected on them would "
                f"leave no held-out evaluation (plan §7.2.2)")
        return self

    def fit_validation_split(self, val_fraction: float = DEFAULT_VAL_FRACTION,
                             salt: str = VALIDATION_SALT
                             ) -> Tuple["EvidenceTable", "EvidenceTable"]:
        """Deterministic episode-level fit / validation split of DEV data."""
        self.require_dev()
        fit_e, val_e = [], []
        for e in self.unique_episodes():
            (val_e if assign_split(self.scope, e, val_fraction, salt) == HELDOUT
             else fit_e).append(e)
        assert_no_leak([(self.scope, e) for e in fit_e], [(self.scope, e) for e in val_e])
        return self.by_episodes(fit_e), self.by_episodes(val_e)


def is_final_heldout(scope: str, episode: str) -> bool:
    return assign_split(scope, episode) == HELDOUT


__all__ = ["EvidenceTable", "DiscoveryError", "VALIDATION_SALT",
           "DEFAULT_VAL_FRACTION", "is_final_heldout", "DEV", "HELDOUT"]
