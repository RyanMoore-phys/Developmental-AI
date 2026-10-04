"""Mixed retrieval with a representative floor, and its measured effect.

COMPONENTS (each draws refs of one kind, default "evidence" = outcomes)
    representative  uniform over a reservoir (Algorithm R) fed every record
                    of the view in write order — ordinary data, as it came
    failure         ∝ current prediction_score error (UNKNOWN = unscored is
                    EXCLUDED, never treated as zero error)
    contradiction   uniform over outcomes whose current score contradicts
                    its prediction
    coverage        uniform over bins, then uniform within the bin, so an
                    under-visited bin is drawn as often as a crowded one
                    (bin_fn over index metadata; default: stream + action)
    context         uniform over view.query(**context)

THE FLOOR. Plan §9: "Keep representative evidence when prioritizing unusual
transitions." Each batch reserves ceil(representative_fraction * n) draws
for the representative component, and a priority component with no
candidates (nothing scored yet, no contradictions) FALLS BACK TO
REPRESENTATIVE, never to another priority. So the representative share is
>= the declared fraction in every batch whatever the priority skew, and the
declared fraction itself may not go below MIN_REPRESENTATIVE (0.25).

MEASURABLE. `stats()` reports, beside the drawn/declared fractions and
fallbacks, what the mix DID to the data: the total-variation distance
between the sampled and population bin distributions, the share taken by
the most-drawn record, and mean error sampled vs population. A sampling
change is documented by these numbers, not by its intent.

The reservoir forgets records the store evicts; after eviction it is
uniform over the records that were admitted AND survived.
"""

from __future__ import annotations

import collections
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..contracts import UNKNOWN
from .errors import ExperienceError

MIN_REPRESENTATIVE = 0.25
PRIORITY_COMPONENTS = ("failure", "contradiction", "coverage", "context")
REPRESENTATIVE = "representative"


def default_bin(meta: Dict[str, Any]) -> Tuple:
    return (meta.get("env"), meta.get("stream"),
            tuple(meta.get("specs", ())), tuple(meta.get("commands", ())))


class Reservoir:
    """Algorithm R: after n offers, each offered item is held with
    probability capacity/n."""

    def __init__(self, capacity: int, rng: np.random.Generator):
        if int(capacity) < 1:
            raise ValueError("reservoir capacity must be >= 1")
        self.capacity = int(capacity)
        self.rng = rng
        self.items: List[str] = []
        self.seen = 0

    def offer(self, item: str) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        j = int(self.rng.integers(0, self.seen))
        if j < self.capacity:
            self.items[j] = item

    def retain(self, alive) -> None:
        self.items = [x for x in self.items if x in alive]


class MixedSampler:
    def __init__(self, view, priorities: Optional[Dict[str, float]] = None,
                 representative_fraction: float = MIN_REPRESENTATIVE,
                 kind: str = "evidence",
                 bin_fn: Optional[Callable[[Dict], Any]] = None,
                 context: Optional[Dict[str, Any]] = None,
                 reservoir_size: int = 4096, seed: int = 0):
        rf = float(representative_fraction)
        if not (MIN_REPRESENTATIVE <= rf <= 1.0):
            raise ValueError(
                f"representative_fraction {rf} below the floor "
                f"{MIN_REPRESENTATIVE}: prioritisation may not starve "
                f"ordinary data (plan §9)")
        pr = dict(priorities or {})
        for k, w in pr.items():
            if k not in PRIORITY_COMPONENTS:
                raise ValueError(f"unknown component {k!r}; known "
                                 f"{PRIORITY_COMPONENTS}")
            if not (isinstance(w, (int, float)) and w >= 0 and math.isfinite(w)):
                raise ValueError(f"weight for {k!r} must be finite >= 0")
        if "context" in pr and not context:
            raise ValueError("the context component needs a context filter")
        tot = sum(pr.values())
        if rf < 1.0 and tot <= 0:
            raise ValueError("representative_fraction < 1 needs at least one "
                             "positive priority weight")
        self.view = view
        self.kind = kind
        self.bin_fn = bin_fn or default_bin
        self.context = dict(context or {})
        self.rng = np.random.default_rng(seed)
        self.declared = {REPRESENTATIVE: rf}
        for k, w in pr.items():
            self.declared[k] = (1.0 - rf) * w / tot if tot > 0 else 0.0
        self.reservoir = Reservoir(reservoir_size, self.rng)
        self._offered: set = set()
        self._score_cache: Dict[str, Tuple[Tuple, Any]] = {}
        self.drawn = collections.Counter()
        self.fallbacks = collections.Counter()
        self.freq = collections.Counter()
        self.bin_freq = collections.Counter()
        self._err_sampled: List[float] = []

    # population ------------------------------------------------------------
    def _sync(self) -> List[str]:
        pop = self.view.refs(self.kind)
        alive = set(pop)
        for r in pop:
            if r not in self._offered:
                self._offered.add(r)
                self.reservoir.offer(r)
        if len(self._offered) != len(alive):
            self._offered &= alive
            self.reservoir.retain(alive)
            for r in list(self._score_cache):
                if r not in alive:
                    del self._score_cache[r]
        return pop

    def _score(self, ref: str):
        key = self.view.interp_key(ref)
        hit = self._score_cache.get(ref)
        if hit is not None and hit[0] == key:
            return hit[1]
        s = self.view.current_score(ref) if key else UNKNOWN
        self._score_cache[ref] = (key, s)
        return s

    def _quotas(self, n: int) -> Dict[str, int]:
        q = {REPRESENTATIVE: int(math.ceil(self.declared[REPRESENTATIVE] * n - 1e-9))}
        rest = n - q[REPRESENTATIVE]
        pri = [(k, w) for k, w in self.declared.items() if k != REPRESENTATIVE]
        tot = sum(w for _, w in pri)
        if rest <= 0 or tot <= 0:
            q[REPRESENTATIVE] = n
            return q
        raw = [(k, rest * w / tot) for k, w in pri]
        base = {k: int(math.floor(x)) for k, x in raw}
        left = rest - sum(base.values())
        for k, x in sorted(raw, key=lambda t: -(t[1] - math.floor(t[1])))[:left]:
            base[k] += 1
        q.update(base)
        return q

    # drawing ---------------------------------------------------------------
    def _draw(self, comp: str, k: int, pop: List[str]) -> List[str]:
        if k <= 0:
            return []
        rng = self.rng
        if comp == REPRESENTATIVE:
            items = self.reservoir.items or pop
            return [items[i] for i in rng.integers(0, len(items), k)]
        if comp == "failure":
            cand, w = [], []
            for r in pop:
                s = self._score(r)
                if s is not UNKNOWN:
                    cand.append(r)
                    w.append(float(s["error"]))
            w = np.asarray(w, dtype=float)
            if not cand or not np.isfinite(w).all() or w.sum() <= 0:
                return []
            idx = rng.choice(len(cand), size=k, p=w / w.sum())
            return [cand[i] for i in idx]
        if comp == "contradiction":
            cand = [r for r in pop
                    if self._score(r) is not UNKNOWN and self._score(r)["contradicts"]]
        elif comp == "coverage":
            bins: Dict[Any, List[str]] = {}
            for r in pop:
                bins.setdefault(self.bin_fn(self.view.meta(r)), []).append(r)
            if not bins:
                return []
            keys = list(bins)
            out = []
            for b in rng.integers(0, len(keys), k):
                members = bins[keys[b]]
                out.append(members[int(rng.integers(0, len(members)))])
            return out
        elif comp == "context":
            cand = self.view.query(kind=self.kind, **self.context)
        else:                                       # pragma: no cover
            raise ValueError(comp)
        if not cand:
            return []
        return [cand[i] for i in rng.integers(0, len(cand), k)]

    def sample(self, n: int) -> List[Tuple[str, str]]:
        """-> [(ref, component)] of length n (with replacement)."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be an int >= 1")
        pop = self._sync()
        if not pop:
            raise ExperienceError(f"no {self.kind!r} records in the view")
        out: List[Tuple[str, str]] = []
        extra_rep = 0
        for comp, k in self._quotas(n).items():
            if comp == REPRESENTATIVE:
                continue
            got = self._draw(comp, k, pop)
            if len(got) < k:
                self.fallbacks[comp] += k - len(got)
                extra_rep += k - len(got)
            out.extend((r, comp) for r in got)
        rep_k = self._quotas(n)[REPRESENTATIVE] + extra_rep
        out.extend((r, REPRESENTATIVE) for r in self._draw(REPRESENTATIVE, rep_k, pop))
        for r, comp in out:
            self.drawn[comp] += 1
            self.freq[r] += 1
            self.bin_freq[self.bin_fn(self.view.meta(r))] += 1
            s = self._score(r)
            if s is not UNKNOWN:
                self._err_sampled.append(float(s["error"]))
        self.rng.shuffle(out)
        return out

    # measurement -----------------------------------------------------------
    def frequencies(self) -> Dict[str, int]:
        return dict(self.freq)

    def stats(self) -> Dict[str, Any]:
        pop = self._sync()
        draws = sum(self.drawn.values())
        pop_bins = collections.Counter(self.bin_fn(self.view.meta(r)) for r in pop)
        tv = UNKNOWN
        if draws and pop:
            keys = set(pop_bins) | set(self.bin_freq)
            tv = 0.5 * sum(abs(self.bin_freq[b] / draws - pop_bins[b] / len(pop))
                           for b in keys)
        errs = [self._score(r) for r in pop]
        errs = [float(s["error"]) for s in errs if s is not UNKNOWN]
        return {
            "draws": draws,
            "declared": dict(self.declared),
            "drawn": dict(self.drawn),
            "fraction": {k: v / draws for k, v in self.drawn.items()} if draws else {},
            "fallbacks": dict(self.fallbacks),
            "population": len(pop),
            "reservoir": len(self.reservoir.items),
            "unique_refs": len(self.freq),
            "max_ref_share": (max(self.freq.values()) / draws) if draws else UNKNOWN,
            "bin_tv_distance": tv,
            "mean_error_sampled": (float(np.mean(self._err_sampled))
                                   if self._err_sampled else UNKNOWN),
            "mean_error_population": float(np.mean(errs)) if errs else UNKNOWN,
        }
