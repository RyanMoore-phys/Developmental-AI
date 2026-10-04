"""Explicit mechanisms for discovery: gated branches over a SMALL transition
library, closed-form fits, penalised scores.

A StructuredMechanism predicts one scalar target as a Gaussian. It is a list
of BRANCHES whose predicates (conjunctions of Conditions) partition the
input space — partition holds by construction, because the only way to get
more than one branch is SPLIT_APPLICABILITY, which replaces a branch by
(branch AND P, branch AND NOT P). Each branch has one transition FORM:

    constant   mu = b                                   k = 1 (+1 sigma)
    linear     mu = w.x + b                             k = d+1 (+1)
    piecewise  mu = wL.x+bL if pivot <  t else wR.x+bR  k = 2(d+1)+1 (+1)
    lookup     mu = table[discrete key(x)]              k = cells+1 (+1)

A ComposedMechanism m2(m1): m1 predicts an intermediate variable z that m2
uses as an input; the composite predicts m2's target from m1's inputs,
substituting m1's mean for z (so it works where z is never observed).

Scores (all in nats, per row unless stated): NLL of the Gaussian with a
1% wide outlier component (_robust_ll explains why one boundary row must
not decide an episode); complexity k =
sum of branch parameters + 1 per split condition; BIC = 2*NLL_sum +
k*ln(n). A row whose needed variable is NaN, or that no branch covers, is
NOT scored (ll = NaN) — missing is not zero error — and coverage is
reported so a candidate cannot win by declining to predict.

Fits are pure: `fit_mechanism` returns a NEW mechanism and never mutates
its argument (a rejected candidate therefore cannot damage its parent).
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contracts import Mechanism
from .table import DiscoveryError, EvidenceTable

FORMS = ("constant", "linear", "piecewise", "lookup")
OPS = ("<", ">=", "==", "!=")
_NEG = {"<": ">=", ">=": "<", "==": "!=", "!=": "=="}
SIGMA_MIN = 1e-3
MIN_BRANCH_ROWS = 6
MAX_LOOKUP_CELLS = 64
PIECEWISE_QUANTILES = tuple(float(q) for q in np.linspace(0.025, 0.975, 39))
LOG2PI = math.log(2 * math.pi)
ROBUST_EPS = 0.01


class FitError(DiscoveryError):
    """A structure cannot be fitted to this evidence (too few rows, ...)."""


class BudgetExceeded(DiscoveryError):
    """A cooperative deadline passed during a fit."""


class CancelToken(float):
    """A fit deadline that can also be CANCELLED (cooperative cancellation).

    It IS a float — the time.monotonic() deadline — so every fitter written
    against the `fitter(mech, table, deadline)` signature keeps working
    unchanged (comparisons, arithmetic, passing it on to fit_mechanism).
    What it adds is `cancel()`: search() cancels the token of a fit it
    ABANDONS, and every library form polls it between iterations (per
    branch, per piecewise threshold, per lookup chunk) through
    check_deadline, so an abandoned fit stops within one iteration instead
    of running on in its thread (verifier B1 follow-up: "an abandoned fit's
    thread cannot be killed in Python"). A fitter that never calls
    check_deadline cannot be stopped this way; search() caps how many of
    those may be alive at once instead (search.MAX_ABANDONED_FITS)."""

    def __new__(cls, deadline: float):
        obj = float.__new__(cls, deadline)
        obj._event = threading.Event()
        return obj

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def expired(self) -> bool:
        return self._event.is_set() or time.monotonic() > float(self)


def check_deadline(deadline: Optional[float]) -> None:
    """Raise BudgetExceeded if `deadline` (a monotonic float, a CancelToken,
    or None = no limit) has passed or been cancelled. Custom fitters call
    this between their own iterations to be cancellable."""
    if deadline is None:
        return
    if getattr(deadline, "cancelled", False):
        raise BudgetExceeded("fit cancelled (abandoned by the search)")
    if time.monotonic() > deadline:
        raise BudgetExceeded("fit deadline passed")


_check_deadline = check_deadline          # historical private name


# --------------------------------------------------------------- conditions
@dataclass(frozen=True)
class Condition:
    var: str
    op: str
    value: float

    def __post_init__(self):
        if not isinstance(self.var, str) or not self.var:
            raise DiscoveryError("Condition.var must be a non-empty string")
        if self.op not in OPS:
            raise DiscoveryError(f"Condition.op {self.op!r} not in {OPS}")
        v = float(self.value)
        if not math.isfinite(v):
            raise DiscoveryError("Condition.value must be finite")
        object.__setattr__(self, "value", v)

    def holds(self, table: EvidenceTable) -> np.ndarray:
        x = table.col(self.var)
        ok = np.isfinite(x)
        with np.errstate(invalid="ignore"):
            r = {"<": x < self.value, ">=": x >= self.value,
                 "==": x == self.value, "!=": x != self.value}[self.op]
        return r & ok          # NaN satisfies neither P nor NOT P

    def negate(self) -> "Condition":
        return Condition(self.var, _NEG[self.op], self.value)

    def to_dict(self) -> Dict[str, Any]:
        return {"var": self.var, "op": self.op, "value": self.value}

    @classmethod
    def from_dict(cls, d) -> "Condition":
        return cls(str(d["var"]), str(d["op"]), float(d["value"]))

    def __str__(self):
        return f"{self.var}{self.op}{self.value:g}"


# ------------------------------------------------------------------ branches
@dataclass(frozen=True)
class Branch:
    predicate: Tuple[Condition, ...] = ()
    form: str = "constant"
    inputs: Tuple[str, ...] = ()
    pivot: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    sigma: float = float("nan")
    n_fit: int = 0

    def __post_init__(self):
        if self.form not in FORMS:
            raise DiscoveryError(f"form {self.form!r} not in {FORMS}")
        if len(set(self.inputs)) != len(self.inputs):
            raise DiscoveryError(f"duplicate inputs {self.inputs}")
        if self.form == "constant" and self.inputs:
            raise DiscoveryError("constant form takes no inputs")
        if self.form == "piecewise" and not self.pivot:
            raise DiscoveryError("piecewise form needs a pivot variable")
        if self.form != "piecewise" and self.pivot is not None:
            raise DiscoveryError("only piecewise takes a pivot")
        if self.form == "lookup" and not self.inputs:
            raise DiscoveryError("lookup form needs at least one input")

    @property
    def fitted(self) -> bool:
        return bool(self.params) and math.isfinite(self.sigma)

    def variables(self) -> Tuple[str, ...]:
        vs = list(self.inputs)
        if self.pivot and self.pivot not in vs:
            vs.append(self.pivot)
        return tuple(vs)

    def n_params(self) -> int:
        d = len(self.inputs)
        if self.form == "constant":
            k = 1
        elif self.form == "linear":
            k = d + 1
        elif self.form == "piecewise":
            k = 2 * (d + 1) + 1
        else:
            k = len(self.params.get("cells", {})) + 1 if self.params else 2
        return k + 1                                         # + sigma

    def mask(self, table: EvidenceTable) -> np.ndarray:
        m = np.ones(len(table), bool)
        for c in self.predicate:
            m &= c.holds(table)
        return m

    def usable(self, table: EvidenceTable) -> np.ndarray:
        m = self.mask(table)
        for v in self.variables():
            m &= np.isfinite(table.col(v))
        return m

    def to_dict(self) -> Dict[str, Any]:
        return {"predicate": [c.to_dict() for c in self.predicate], "form": self.form,
                "inputs": list(self.inputs), "pivot": self.pivot,
                "params": _jsonable(self.params), "sigma": float(self.sigma),
                "n_fit": int(self.n_fit)}

    @classmethod
    def from_dict(cls, d) -> "Branch":
        return cls(tuple(Condition.from_dict(c) for c in d["predicate"]), d["form"],
                   tuple(d["inputs"]), d.get("pivot"), dict(d.get("params") or {}),
                   float(d.get("sigma", float("nan"))), int(d.get("n_fit", 0)))

    def structure(self) -> Tuple:
        return (tuple(sorted(str(c) for c in self.predicate)), self.form,
                tuple(sorted(self.inputs)), self.pivot)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    return x


def _key(vals: np.ndarray) -> str:
    return "|".join(f"{float(v):g}" for v in vals)


def _design(table: EvidenceTable, inputs: Sequence[str], m: np.ndarray) -> np.ndarray:
    cols = [table.col(v)[m] for v in inputs]
    return np.column_stack(cols + [np.ones(int(m.sum()))])


def _lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # tiny ridge keeps a collinear design finite; negligible otherwise
    A = X.T @ X + 1e-8 * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def _branch_mean(b: Branch, table: EvidenceTable, m: np.ndarray):
    """(mu, sigma) on rows m (all assumed usable)."""
    n = int(m.sum())
    p = b.params
    if b.form == "constant":
        return np.full(n, p["b"]), np.full(n, b.sigma)
    if b.form == "linear":
        X = _design(table, b.inputs, m)
        return X @ np.asarray(p["w"] + [p["b"]], float), np.full(n, b.sigma)
    if b.form == "piecewise":
        X = _design(table, b.inputs, m)
        left = table.col(b.pivot)[m] < p["t"]
        mu = np.where(left, X @ np.asarray(p["wL"] + [p["bL"]], float),
                      X @ np.asarray(p["wR"] + [p["bR"]], float))
        return mu, np.full(n, b.sigma)
    # lookup
    keys = np.column_stack([table.col(v)[m] for v in b.inputs])
    cells = p["cells"]
    mu = np.empty(n)
    sd = np.full(n, b.sigma)
    unseen_sd = math.sqrt(b.sigma ** 2 + p["var_y"])
    for i in range(n):
        k = _key(keys[i])
        if k in cells:
            mu[i] = cells[k]
        else:
            mu[i] = p["default"]
            sd[i] = unseen_sd
    return mu, sd


def fit_branch(b: Branch, table: EvidenceTable, m: np.ndarray,
               deadline: Optional[float] = None) -> Branch:
    """Fit branch b on rows m (usable rows with a finite target). The
    iterative forms poll `deadline` (check_deadline) between iterations."""
    y = table.y[m]
    n = len(y)
    if n < MIN_BRANCH_ROWS:
        raise FitError(f"branch {b.structure()} has {n} rows < {MIN_BRANCH_ROWS}")
    if b.form == "constant":
        params = {"b": float(y.mean())}
        resid = y - y.mean()
    elif b.form == "linear":
        X = _design(table, b.inputs, m)
        if n < X.shape[1] + 2:
            raise FitError("too few rows for linear form")
        w = _lstsq(X, y)
        params = {"w": [float(v) for v in w[:-1]], "b": float(w[-1])}
        resid = y - X @ w
    elif b.form == "piecewise":
        X = _design(table, b.inputs, m)
        piv = table.col(b.pivot)[m]
        best = None
        for t in np.unique(np.quantile(piv, PIECEWISE_QUANTILES)):
            check_deadline(deadline)
            L = piv < t
            nl, nr = int(L.sum()), int((~L).sum())
            if min(nl, nr) < max(MIN_BRANCH_ROWS, X.shape[1] + 1):
                continue
            wl, wr = _lstsq(X[L], y[L]), _lstsq(X[~L], y[~L])
            r = np.where(L, y - X @ wl, y - X @ wr)
            sse = float(r @ r)
            if best is None or sse < best[0]:
                best = (sse, float(t), wl, wr, r)
        if best is None:
            raise FitError("piecewise: no threshold leaves both sides fittable")
        _, t, wl, wr, resid = best
        params = {"t": t, "wL": [float(v) for v in wl[:-1]], "bL": float(wl[-1]),
                  "wR": [float(v) for v in wr[:-1]], "bR": float(wr[-1])}
    else:
        keys = np.column_stack([table.col(v)[m] for v in b.inputs])
        if np.any(keys != np.round(keys)):
            raise FitError("lookup inputs must be discrete (integer-valued)")
        groups: Dict[str, List[float]] = {}
        for i in range(n):
            if not i % 4096:
                check_deadline(deadline)
            groups.setdefault(_key(keys[i]), []).append(y[i])
        if len(groups) > MAX_LOOKUP_CELLS:
            raise FitError(f"lookup has {len(groups)} cells > {MAX_LOOKUP_CELLS}")
        cells = {k: float(np.mean(v)) for k, v in groups.items()}
        params = {"cells": cells, "default": float(y.mean()), "var_y": float(y.var())}
        resid = y - np.array([cells[_key(keys[i])] for i in range(n)])
    sigma = max(SIGMA_MIN, float(np.sqrt(np.mean(resid ** 2))))
    return replace(b, params=params, sigma=sigma, n_fit=n)


# ------------------------------------------------------------- mechanisms
def _gauss_ll(y, mu, sd):
    return -0.5 * (LOG2PI + 2 * np.log(sd) + ((y - mu) / sd) ** 2)


def _robust_ll(y, mu, sd, outlier_sd):
    """log[(1-eps) N(y; mu, sd) + eps N(y; mu, max(sd, outlier_sd))].

    WHY: a hard rule fitted to finite data puts its boundary slightly off
    (threshold 0.504 for a true 0.5). A row in that sliver costs a pure
    Gaussian with sd 0.2 ~200 nats — one row then decides a whole
    validation episode, and the CORRECT structure loses to a blurrier wrong
    one (measured: threshold fixture seed 5). The eps component (outlier_sd
    = target spread on the fit rows) caps a single row's cost at ~8 nats
    and costs ~eps nats/row when the model is right."""
    g = _gauss_ll(y, mu, sd)
    if outlier_sd is None:
        return g
    w = _gauss_ll(y, mu, np.maximum(sd, outlier_sd))
    return np.logaddexp(math.log1p(-ROBUST_EPS) + g, math.log(ROBUST_EPS) + w)


class _MechBase:
    kind = "base"
    mechanism_id: str
    target: str
    version: int
    parents: Tuple[str, ...]
    causal: Dict[str, Any]

    # subclasses: predict(table) -> (mu, sd) with NaN where not predicted
    def loglik(self, table: EvidenceTable) -> np.ndarray:
        if table.target != self.target:
            raise DiscoveryError(f"table target {table.target!r} != mechanism "
                                 f"target {self.target!r}")
        mu, sd = self.predict(table)
        y = table.y
        ll = np.full(len(table), np.nan)
        ok = np.isfinite(mu) & np.isfinite(y)
        ll[ok] = _robust_ll(y[ok], mu[ok], sd[ok], self.outlier_sd)
        return ll

    def nll(self, table: EvidenceTable) -> Tuple[float, int]:
        ll = self.loglik(table)
        ok = np.isfinite(ll)
        return float(-ll[ok].sum()), int(ok.sum())

    def bic(self, table: EvidenceTable) -> float:
        nll, n = self.nll(table)
        if n == 0:
            return float("inf")
        return 2.0 * nll + self.complexity() * math.log(n)

    def ref(self) -> str:
        return f"{self.mechanism_id}@{self.version}"

    def to_json(self) -> str:
        from ..contracts import to_json
        return to_json(self.to_record())

    def nbytes(self) -> int:
        return len(self.to_json())


class StructuredMechanism(_MechBase):
    kind = "structured"

    def __init__(self, mechanism_id: str, target: str, branches: Sequence[Branch],
                 version: int = 1, parents: Sequence[str] = (),
                 causal: Optional[Dict[str, Any]] = None,
                 outlier_sd: Optional[float] = None):
        if not isinstance(mechanism_id, str) or not mechanism_id:
            raise DiscoveryError("mechanism_id must be a non-empty string")
        if not branches:
            raise DiscoveryError("a mechanism needs at least one branch")
        self.mechanism_id = mechanism_id
        self.target = target
        self.branches = tuple(branches)
        for b in self.branches:
            if target in b.variables() or any(c.var == target for c in b.predicate):
                raise DiscoveryError(f"target {target!r} used as its own input")
        self.version = int(version)
        self.parents = tuple(parents)
        self.causal = dict(causal or {})
        if outlier_sd is not None and not (math.isfinite(outlier_sd) and outlier_sd > 0):
            raise DiscoveryError(f"outlier_sd must be finite and > 0, got {outlier_sd}")
        self.outlier_sd = None if outlier_sd is None else float(outlier_sd)

    @property
    def fitted(self) -> bool:
        return all(b.fitted for b in self.branches)

    def dependencies(self) -> Tuple[str, ...]:
        s = set()
        for b in self.branches:
            s.update(b.variables())
            s.update(c.var for c in b.predicate)
        return tuple(sorted(s))

    def complexity(self) -> int:
        return sum(b.n_params() for b in self.branches) + (len(self.branches) - 1)

    def structure_key(self) -> Tuple:
        return ("structured", self.target, tuple(sorted(b.structure() for b in self.branches)))

    def with_branches(self, branches, mechanism_id=None, parents=None, version=None):
        return StructuredMechanism(mechanism_id or self.mechanism_id, self.target,
                                   branches, self.version if version is None else version,
                                   self.parents if parents is None else parents,
                                   self.causal, self.outlier_sd)

    def predict(self, table: EvidenceTable):
        if not self.fitted:
            raise DiscoveryError(f"{self.mechanism_id} is not fitted")
        mu = np.full(len(table), np.nan)
        sd = np.full(len(table), np.nan)
        for b in self.branches:
            m = b.usable(table)
            if m.any():
                mu[m], sd[m] = _branch_mean(b, table, m)
        return mu, sd

    def fit(self, table: EvidenceTable, deadline: Optional[float] = None
            ) -> "StructuredMechanism":
        y_ok = np.isfinite(table.y)
        new = []
        for b in self.branches:
            check_deadline(deadline)
            new.append(fit_branch(b, table, b.usable(table) & y_ok, deadline))
        out_sd = max(SIGMA_MIN, float(np.std(table.y[y_ok])))
        return StructuredMechanism(self.mechanism_id, self.target, new, self.version,
                                   self.parents, self.causal, out_sd)

    # ---- contracts.Mechanism ----------------------------------------------
    def to_record(self, evidence_refs: Sequence[str] = ()) -> Mechanism:
        return Mechanism(
            mechanism_id=self.mechanism_id, version=self.version,
            inputs=self.dependencies(), outputs=(self.target,),
            applicability={"branches": [[c.to_dict() for c in b.predicate]
                                        for b in self.branches]},
            parameters={"branches": [b.to_dict() for b in self.branches]},
            evidence_refs=tuple(evidence_refs),
            payload={"kind": self.kind, "parents": list(self.parents),
                     "outlier_sd": self.outlier_sd,
                     "complexity": self.complexity(), "causal": _jsonable(self.causal)})


class ComposedMechanism(_MechBase):
    """m2 o m1: m1.target (the intermediate) is an input/dependency of m2."""
    kind = "composed"

    def __init__(self, mechanism_id: str, first: StructuredMechanism,
                 second: StructuredMechanism, version: int = 1,
                 parents: Sequence[str] = (), causal=None):
        if first.target not in second.dependencies():
            raise DiscoveryError(f"cannot compose: {first.target!r} is not a "
                                 f"dependency of {second.mechanism_id}")
        if second.target in first.dependencies():
            raise DiscoveryError("cannot compose: cycle through the outer target")
        self.mechanism_id = mechanism_id
        self.first, self.second = first, second
        self.target = second.target
        self.intermediate = first.target
        self.version = int(version)
        self.parents = tuple(parents)
        self.causal = dict(causal or {})

    @property
    def fitted(self) -> bool:
        return self.first.fitted and self.second.fitted

    @property
    def outlier_sd(self) -> Optional[float]:
        return self.second.outlier_sd

    def dependencies(self) -> Tuple[str, ...]:
        s = set(self.first.dependencies()) | (set(self.second.dependencies())
                                              - {self.intermediate})
        return tuple(sorted(s))

    def complexity(self) -> int:
        return self.first.complexity() + self.second.complexity()

    def structure_key(self) -> Tuple:
        return ("composed", self.first.structure_key(), self.second.structure_key())

    def _substituted(self, table: EvidenceTable, first=None) -> EvidenceTable:
        mu1, _ = (first or self.first).predict(table)     # predict ignores the target
        return table.with_column(self.intermediate, mu1)

    def predict(self, table: EvidenceTable):
        return self.second.predict(self._substituted(table))

    def fit(self, table: EvidenceTable, deadline=None) -> "ComposedMechanism":
        if self.intermediate not in table.columns:
            raise FitError(f"intermediate {self.intermediate!r} never observed; "
                           f"the first stage cannot be fitted")
        inner = EvidenceTable(table.columns, self.intermediate, table.episodes, table.scope)
        f1 = self.first.fit(inner, deadline)
        check_deadline(deadline)
        f2 = self.second.fit(self._substituted(table, f1), deadline)
        return ComposedMechanism(self.mechanism_id, f1, f2, self.version, self.parents,
                                 self.causal)

    def to_record(self, evidence_refs: Sequence[str] = ()) -> Mechanism:
        r1, r2 = self.first.to_record(), self.second.to_record()
        return Mechanism(
            mechanism_id=self.mechanism_id, version=self.version,
            inputs=self.dependencies(), outputs=(self.target,),
            applicability={"first": r1.applicability, "second": r2.applicability},
            parameters={"first": r1.parameters, "second": r2.parameters},
            evidence_refs=tuple(evidence_refs),
            payload={"kind": self.kind, "parents": list(self.parents),
                     "intermediate": self.intermediate,
                     "first_id": self.first.mechanism_id, "first_target": self.first.target,
                     "second_id": self.second.mechanism_id,
                     "first_outlier_sd": self.first.outlier_sd,
                     "outlier_sd": self.second.outlier_sd,
                     "complexity": self.complexity(), "causal": _jsonable(self.causal)})


def mechanism_from_record(rec: Mechanism) -> _MechBase:
    kind = rec.payload.get("kind")
    if kind == "structured":
        return StructuredMechanism(
            rec.mechanism_id, rec.outputs[0],
            [Branch.from_dict(d) for d in rec.parameters["branches"]], rec.version,
            tuple(rec.payload.get("parents", ())), dict(rec.payload.get("causal") or {}),
            rec.payload.get("outlier_sd"))
    if kind == "composed":
        p = rec.payload
        f = StructuredMechanism(p["first_id"], p["first_target"],
                                [Branch.from_dict(d) for d in rec.parameters["first"]["branches"]],
                                outlier_sd=p.get("first_outlier_sd"))
        s = StructuredMechanism(p["second_id"], rec.outputs[0],
                                [Branch.from_dict(d) for d in rec.parameters["second"]["branches"]],
                                outlier_sd=p.get("outlier_sd"))
        return ComposedMechanism(rec.mechanism_id, f, s, rec.version,
                                 tuple(p.get("parents", ())), dict(p.get("causal") or {}))
    raise DiscoveryError(f"unknown mechanism kind {kind!r}")


def fit_mechanism(mech: _MechBase, table: EvidenceTable,
                  deadline: Optional[float] = None) -> _MechBase:
    """The default fitter: pure, closed-form, cooperative deadline."""
    return mech.fit(table, deadline)


def initial_mechanism(target: str, form: str = "constant", inputs: Sequence[str] = (),
                      mechanism_id: Optional[str] = None) -> StructuredMechanism:
    mid = mechanism_id or f"{target}:root-{form}-" + hashlib.sha256(
        "|".join(inputs).encode()).hexdigest()[:6]
    return StructuredMechanism(mid, target, [Branch((), form, tuple(inputs))])


def per_episode_paired(ll_a: np.ndarray, ll_b: np.ndarray, episodes: np.ndarray,
                       min_coverage: float = 0.95) -> Dict[str, Any]:
    """Paired comparison of per-row log-likelihoods a vs b, blocked by
    EPISODE (rows within an episode are not independent). Rows b scores
    but a does not count as a COVERAGE failure for a."""
    a_ok, b_ok = np.isfinite(ll_a), np.isfinite(ll_b)
    cov = float(a_ok[b_ok].mean()) if b_ok.any() else (1.0 if a_ok.any() else 0.0)
    both = a_ok & b_ok
    diffs = []
    for e in dict.fromkeys(episodes[both].tolist()):
        m = both & (episodes == e)
        diffs.append(float(np.mean(ll_a[m] - ll_b[m])))
    d = np.asarray(diffs)
    n = len(d)
    mean = float(d.mean()) if n else float("nan")
    if n >= 2:
        se = float(d.std(ddof=1) / math.sqrt(n))
        t = mean / se if se > 0 else (float("inf") * np.sign(mean) if mean else 0.0)
    else:
        se, t = float("nan"), float("nan")
    return {"mean": mean, "se": se, "t": float(t), "n_episodes": n,
            "coverage": cov, "n_rows": int(both.sum())}


def verdict(stats: Dict[str, Any], t_crit: float = 2.0, margin: float = 0.0,
            min_episodes: int = 3) -> str:
    """'advantage' | 'disadvantage' | 'inconclusive' for a over b."""
    if stats["coverage"] < 0.95 and stats["n_rows"] > 0:
        return "disadvantage"
    if stats["n_episodes"] < min_episodes or math.isnan(stats["t"]):
        return "inconclusive"
    if stats["t"] >= t_crit and stats["mean"] > margin:
        return "advantage"
    if stats["t"] <= -t_crit and stats["mean"] < -margin:
        return "disadvantage"
    return "inconclusive"


__all__ = ["FORMS", "Condition", "Branch", "StructuredMechanism", "ComposedMechanism",
           "fit_mechanism", "fit_branch", "initial_mechanism", "mechanism_from_record",
           "per_episode_paired", "verdict", "FitError", "BudgetExceeded",
           "CancelToken", "check_deadline",
           "MIN_BRANCH_ROWS", "MAX_LOOKUP_CELLS", "SIGMA_MIN"]
