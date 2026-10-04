"""Causal status of mechanism dependencies (plan Stage 9 item 6).

Every dependency of a mechanism (an input, a piecewise pivot, or a variable
in a branch predicate) is labelled

    interventional  the evidence supporting it includes rows where an AGENT
                    ACTION SET that variable (EvidenceTable.intervened, built
                    only from contracts.Action records with payload "sets"),
                    and on THOSE ROWS ALONE removing the dependency costs at
                    least `delta_bic` (BIC of the ablated structure minus the
                    full one, both refitted on the intervened rows)
    observational   everything else

The reason is recorded: "no_intervention" (never set by an action),
"insufficient_intervention" (too few intervened rows to fit), or
"intervention_null" — the agent DID set the variable and the outcome did
not follow. That last one is a contradiction of a correlational pattern and
is reported as such.

THE RULE THAT MATTERS: there is no path from observational to
interventional except intervened rows. More observational data, a better
fit, or a stronger correlation never upgrades a label (a confounded
predictor fits beautifully). And on intervened rows the label carries the
INTERVENTIONAL effect estimate, which under confounding differs from the
pooled (observational) coefficient — both are reported, side by side.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .forms import (Branch, ComposedMechanism, FitError, StructuredMechanism,
                    _MechBase, check_deadline, fit_mechanism)
from .table import DiscoveryError, EvidenceTable

OBSERVATIONAL = "observational"
INTERVENTIONAL = "interventional"
CAUSAL_STATUSES = (OBSERVATIONAL, INTERVENTIONAL)
MIN_INTERVENED = 20
DELTA_BIC = 6.0


def ablate(mech: StructuredMechanism, var: str) -> StructuredMechanism:
    """The structure with every use of `var` removed (unfitted)."""
    out: List[Branch] = []
    for b in mech.branches:
        pred = tuple(c for c in b.predicate if c.var != var)
        inputs = tuple(v for v in b.inputs if v != var)
        form, pivot = b.form, b.pivot
        if pivot == var:
            form, pivot = "linear", None
        if form in ("linear", "lookup") and not inputs:
            form = "constant"
        if form == "piecewise" and pivot is None:
            form = "linear" if inputs else "constant"
        nb = Branch(pred, form, inputs, pivot)
        for i, o in enumerate(out):                         # merge equal regions
            if set(o.predicate) == set(nb.predicate):
                ins = tuple(dict.fromkeys(o.inputs + nb.inputs))
                f = o.form if o.form != "constant" else ("linear" if ins else "constant")
                out[i] = Branch(o.predicate, f, ins, o.pivot)
                break
        else:
            out.append(nb)
    return StructuredMechanism(mech.mechanism_id + "~" + var, mech.target, out)


def _effect(fitted: StructuredMechanism, var: str) -> Optional[float]:
    for b in fitted.branches:
        if b.form == "linear" and var in b.inputs:
            return float(b.params["w"][b.inputs.index(var)])
    return None


def annotate_causal(mech: _MechBase, table: EvidenceTable,
                    min_intervened: int = MIN_INTERVENED,
                    delta_bic: float = DELTA_BIC, fitter=fit_mechanism,
                    deadline: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    """{dependency: {"status", "reason", "n_intervened", "delta_bic",
    "effect_interventional", "effect_observational"}}

    `deadline` (a monotonic float or forms.CancelToken) is checked before
    each dependency and handed to every refit, so an annotation running
    inside an ABANDONED search evaluation stops instead of refitting on in
    its thread with no limit (before 2026-10-03 these fits ran with no
    deadline at all). None = unlimited, the historical behaviour; a fitter
    is then called with two arguments, exactly as before."""
    if table.target != mech.target:
        raise DiscoveryError("annotate_causal: table target differs from mechanism's")
    labels: Dict[str, Dict[str, Any]] = {}
    fit = (lambda m, t: fitter(m, t)) if deadline is None else \
        (lambda m, t: fitter(m, t, deadline))
    for var in mech.dependencies():
        check_deadline(deadline)
        lab = {"status": OBSERVATIONAL, "reason": "no_intervention",
               "n_intervened": 0, "delta_bic": None,
               "effect_interventional": None, "effect_observational": None}
        labels[var] = lab
        if isinstance(mech, ComposedMechanism):
            lab["reason"] = "composite_not_tested"
            continue
        if mech.fitted:
            lab["effect_observational"] = _effect(mech, var)
        rows = table.intervened_mask(var) & np.isfinite(table.y)
        for d in mech.dependencies():
            rows &= np.isfinite(table.col(d))
        n = int(rows.sum())
        lab["n_intervened"] = n
        if n == 0:
            continue
        if n < min_intervened or np.ptp(table.col(var)[rows]) == 0:
            lab["reason"] = "insufficient_intervention"
            continue
        sub = table.select(rows)
        try:
            full = fit(mech, sub)
            abl = fit(ablate(mech, var), sub)
        except FitError:
            lab["reason"] = "insufficient_intervention"
            continue
        d = float(abl.bic(sub) - full.bic(sub))
        lab["delta_bic"] = d
        lab["effect_interventional"] = _effect(full, var)
        if d >= delta_bic:
            lab["status"], lab["reason"] = INTERVENTIONAL, "intervention_supported"
        else:
            lab["reason"] = "intervention_null"
    return labels


def with_causal(mech: _MechBase, labels: Dict[str, Dict[str, Any]]) -> _MechBase:
    """A copy carrying `labels` (mechanisms are treated as immutable)."""
    import copy
    m = copy.copy(mech)
    m.causal = {k: dict(v) for k, v in labels.items()}
    return m


def contradictions(labels: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"kind": "intervention_null", "var": k, "n_intervened": v["n_intervened"],
             "delta_bic": v["delta_bic"]}
            for k, v in labels.items() if v["reason"] == "intervention_null"]


__all__ = ["OBSERVATIONAL", "INTERVENTIONAL", "CAUSAL_STATUSES", "ablate",
           "annotate_causal", "with_causal", "contradictions", "MIN_INTERVENED",
           "DELTA_BIC"]
