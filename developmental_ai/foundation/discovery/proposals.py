"""The restricted proposal language (plan Stage 9 item 1).

A Proposal is a typed, serialisable record naming its PARENT mechanisms
(as "id@version", so a proposal against a stale version is refused) and
one operation from a closed set:

    ADD_DEPENDENCY      (parent, var[, branch])  branch inputs += var
                        (constant -> linear(var); other forms keep theirs)
    SPLIT_APPLICABILITY (parent, condition[, branch])  branch -> (b AND P),
                        (b AND NOT P), each refitted; P is var<thr, var>=thr,
                        var==value or var!=value
    COMPOSE             (first, second)  second o first, first's target an
                        input of second (ComposedMechanism)
    REPLACE_TRANSITION  (parent, form[, pivot][, branch])  form from the
                        library: constant, linear, piecewise(pivot), lookup
    REFIT               (parent)  the PARAMETER-ONLY update — same structure,
                        new parameters. Not structural; it is here so the
                        parameter-only baseline goes through the very same
                        budgeted, validated path (the Stage 9 gate compares
                        against it under the same budget).

Nothing outside this set can be proposed: there is no free-form program
search, by design (plan: "restricted proposal language").
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from .forms import (FORMS, Branch, ComposedMechanism, Condition, StructuredMechanism,
                    _MechBase)
from .table import DiscoveryError

ADD_DEPENDENCY = "ADD_DEPENDENCY"
SPLIT_APPLICABILITY = "SPLIT_APPLICABILITY"
COMPOSE = "COMPOSE"
REPLACE_TRANSITION = "REPLACE_TRANSITION"
REFIT = "REFIT"
STRUCTURAL_OPS = (ADD_DEPENDENCY, SPLIT_APPLICABILITY, COMPOSE, REPLACE_TRANSITION)
ALL_OPS = STRUCTURAL_OPS + (REFIT,)
PROPOSAL_SCHEMA_VERSION = 1

_ARGS = {
    ADD_DEPENDENCY: ({"var"}, {"branch"}),
    SPLIT_APPLICABILITY: ({"condition"}, {"branch"}),
    COMPOSE: (set(), set()),
    REPLACE_TRANSITION: ({"form"}, {"pivot", "branch"}),
    REFIT: (set(), set()),
}
_NPARENTS = {COMPOSE: 2}


class ProposalError(DiscoveryError):
    """A malformed proposal, or one that does not apply to its parents."""


def _split_ref(ref: str) -> Tuple[str, int]:
    if not isinstance(ref, str) or "@" not in ref:
        raise ProposalError(f"parent ref must be 'id@version', got {ref!r}")
    mid, v = ref.rsplit("@", 1)
    try:
        return mid, int(v)
    except ValueError:
        raise ProposalError(f"bad version in parent ref {ref!r}")


@dataclass(frozen=True)
class Proposal:
    op: str
    parents: Tuple[str, ...]
    args: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.op not in _ARGS:
            raise ProposalError(f"unknown op {self.op!r}; the language is {ALL_OPS}")
        parents = tuple(self.parents)
        if len(parents) != _NPARENTS.get(self.op, 1):
            raise ProposalError(f"{self.op} takes {_NPARENTS.get(self.op, 1)} parent(s), "
                                f"got {len(parents)}")
        for r in parents:
            _split_ref(r)
        object.__setattr__(self, "parents", parents)
        req, opt = _ARGS[self.op]
        keys = set(self.args)
        if not req <= keys or not keys <= req | opt:
            raise ProposalError(f"{self.op} args must be {sorted(req)} (+ optional "
                                f"{sorted(opt)}), got {sorted(keys)}")
        a = dict(self.args)
        if "branch" in a and a["branch"] is not None:
            if isinstance(a["branch"], bool) or not isinstance(a["branch"], int) \
                    or a["branch"] < 0:
                raise ProposalError(f"branch must be a non-negative int, got {a['branch']!r}")
        if "var" in a and (not isinstance(a["var"], str) or not a["var"]):
            raise ProposalError("var must be a non-empty string")
        if "condition" in a:
            c = a["condition"]
            a["condition"] = c if isinstance(c, Condition) else Condition.from_dict(c)
        if "form" in a:
            if a["form"] not in FORMS:
                raise ProposalError(f"form {a['form']!r} not in the library {FORMS}")
            if (a["form"] == "piecewise") != bool(a.get("pivot")):
                raise ProposalError("piecewise needs a pivot; other forms take none")
        object.__setattr__(self, "args", a)

    # ---- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        args = {k: (v.to_dict() if isinstance(v, Condition) else v)
                for k, v in sorted(self.args.items())}
        return {"schema": "discovery.proposal", "version": PROPOSAL_SCHEMA_VERSION,
                "op": self.op, "parents": list(self.parents), "args": args}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Proposal":
        if d.get("schema") != "discovery.proposal":
            raise ProposalError(f"not a proposal record: {d.get('schema')!r}")
        if d.get("version") != PROPOSAL_SCHEMA_VERSION:
            raise ProposalError(f"proposal schema version {d.get('version')!r} "
                                f"!= {PROPOSAL_SCHEMA_VERSION}")
        return cls(d["op"], tuple(d["parents"]), dict(d["args"]))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Proposal":
        return cls.from_dict(json.loads(s))

    @property
    def proposal_id(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    def __str__(self):
        a = ",".join(f"{k}={v}" for k, v in sorted(self.args.items()))
        return f"{self.op}({'+'.join(self.parents)}{';' + a if a else ''})"

    # ---- application -------------------------------------------------------
    def apply(self, parents: Mapping[str, _MechBase]) -> _MechBase:
        """The UNFITTED child structure. `parents` maps mechanism_id -> mech;
        each must exist at exactly the version this proposal names."""
        ms = []
        for r in self.parents:
            mid, v = _split_ref(r)
            if mid not in parents:
                raise ProposalError(f"parent {mid!r} not supplied")
            m = parents[mid]
            if m.version != v:
                raise ProposalError(f"stale proposal: parent {mid} is at version "
                                    f"{m.version}, proposal names {v}")
            ms.append(m)
        new_id = f"{ms[-1].target}:{self.op.lower()[:6]}-{self.proposal_id[:10]}"
        refs = tuple(m.ref() for m in ms)
        if self.op == COMPOSE:
            first, second = ms
            if not isinstance(first, StructuredMechanism) or \
                    not isinstance(second, StructuredMechanism):
                raise ProposalError("COMPOSE takes two structured mechanisms")
            try:
                return ComposedMechanism(new_id, first, second, parents=refs)
            except DiscoveryError as e:
                raise ProposalError(str(e))
        m = ms[0]
        if self.op == REFIT:            # same mechanism, next version
            if isinstance(m, StructuredMechanism):
                return m.with_branches(m.branches, parents=refs, version=m.version + 1)
            return ComposedMechanism(m.mechanism_id, m.first, m.second, m.version + 1,
                                     refs, m.causal)
        if not isinstance(m, StructuredMechanism):
            raise ProposalError(f"{self.op} applies to structured mechanisms only")
        bi = self.args.get("branch")
        if bi is not None and bi >= len(m.branches):
            raise ProposalError(f"branch {bi} out of range ({len(m.branches)} branches)")
        idx = range(len(m.branches)) if bi is None else [bi]
        out = []
        for i, b in enumerate(m.branches):
            if i not in idx:
                out.append(b)
                continue
            out.extend(self._edit(b, m.target))
        if all(o.structure() == b.structure() for o, b in zip(out, m.branches)) and \
                len(out) == len(m.branches):
            raise ProposalError(f"{self} does not change the structure")
        return StructuredMechanism(new_id, m.target, out, version=1, parents=refs)

    def _edit(self, b: Branch, target: str):
        a = self.args
        if self.op == ADD_DEPENDENCY:
            v = a["var"]
            if v == target:
                raise ProposalError("a mechanism cannot depend on its own target")
            if v in b.inputs:
                return [b]
            form = "linear" if b.form == "constant" else b.form
            return [Branch(b.predicate, form, b.inputs + (v,), b.pivot)]
        if self.op == SPLIT_APPLICABILITY:
            c = a["condition"]
            if c.var == target:
                raise ProposalError("cannot split on the target")
            if c in b.predicate or c.negate() in b.predicate:
                raise ProposalError(f"branch already split on {c}")
            return [Branch(b.predicate + (c,), b.form, b.inputs, b.pivot),
                    Branch(b.predicate + (c.negate(),), b.form, b.inputs, b.pivot)]
        # REPLACE_TRANSITION
        form = a["form"]
        if form == "constant":
            return [Branch(b.predicate, "constant")]
        if form == "piecewise":
            if a["pivot"] == target:
                raise ProposalError("pivot cannot be the target")
            return [Branch(b.predicate, "piecewise", b.inputs, a["pivot"])]
        if form == "lookup" and not b.inputs:
            raise ProposalError("lookup needs inputs; the branch has none")
        return [Branch(b.predicate, form, b.inputs)]


__all__ = ["Proposal", "ProposalError", "ADD_DEPENDENCY", "SPLIT_APPLICABILITY",
           "COMPOSE", "REPLACE_TRANSITION", "REFIT", "STRUCTURAL_OPS", "ALL_OPS",
           "PROPOSAL_SCHEMA_VERSION"]
