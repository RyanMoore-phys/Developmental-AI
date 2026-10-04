"""The eight core records of plan §5.

    Observation   stream, episode, seq, timestamps, channel, payload,
                  missingness, provenance
    Action        control spec ref, command, duration, dispatch/completion
    StateBelief   representation + version, variables, uncertainty, support
    Mechanism     inputs, outputs, applicability, parameters, uncertainty,
                  evidence refs, version
    Prediction    source belief + model versions/snapshot, candidate action,
                  horizon, outcome distribution
    Experiment    question, alternatives, intervention, predicted evidence,
                  cost budget
    Evidence      observations, executed actions, prediction refs, validity,
                  provenance
    Skill         controller ref, initiation, termination, competence evidence

Common fields are typed and validated here; everything domain-specific goes
in `payload` (and, for Observation, the reading lives in payload["value"]).

ORDERING CONVENTION. Observation seq t is what the agent saw before acting;
the Action with seq t was dispatched in response to it; the observations
with seq t+1 are its consequence.

PROVENANCE. "sensor" (measured), "inferred" (derived from measurements),
"imagined" (a model's output), "evaluator" (privileged truth; never a
learning input). An Evidence record must agree with every observation it
contains, so imagined outcomes cannot be filed as sensor evidence (plan
§7.4: "a model cannot certify its own imagined outcomes as real evidence").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from .base import Record, register_record
from .ids import ObsRef, Scope
from .sentinels import UNKNOWN, Sentinel
from .values import (PROVENANCES, check_value, fail, req_enum, req_finite,
                     req_int, req_mapping, req_str, req_str_tuple, req_tuple)

VALIDITIES = ("valid", "partial", "invalid")


def _scope_fields(o, r):
    for f in ("environment", "stream", "episode"):
        req_str(o, f, getattr(r, f))
    r._set("seq", req_int(o, "seq", r.seq, minimum=0))


def _command(o, v):
    """Commands are int, a numeric array/tuple of numbers; never bool/str."""
    if isinstance(v, (bool, np.bool_, str, Sentinel)) or v is None:
        fail(o, f"command must be numeric, got {v!r}")
    check_value(o, "command", v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, list):
        return tuple(v)
    return v


@register_record
@dataclass(frozen=True, eq=False)
class Observation(Record):
    SCHEMA = "observation"
    VERSION = 1

    environment: str
    stream: str
    episode: str
    seq: int
    t_env: float
    t_wall: float
    channel: str
    provenance: str
    payload: Dict[str, Any]
    missingness: Dict[str, Sentinel] = field(default_factory=dict)

    def _validate(self):
        o = "Observation"
        _scope_fields(o, self)
        self._set("t_env", req_finite(o, "t_env", self.t_env))
        self._set("t_wall", req_finite(o, "t_wall", self.t_wall))
        req_str(o, "channel", self.channel)
        req_enum(o, "provenance", self.provenance, PROVENANCES)
        payload = req_mapping(o, "payload", self.payload)
        miss = req_mapping(o, "missingness", self.missingness)
        for k, s in miss.items():
            if not isinstance(s, Sentinel):
                fail(o, f"missingness[{k!r}] must be a sentinel, got {s!r}")
            if k in payload and payload[k] is not s:
                fail(o, f"payload[{k!r}] holds a value but missingness "
                        f"declares it {s!r}")
        # A sentinel in the payload IS a missingness declaration; record it
        # so consumers have one place to look.
        for k, v in payload.items():
            if isinstance(v, Sentinel):
                miss.setdefault(k, v)
        self._set("payload", payload)
        self._set("missingness", miss)

    @property
    def scope(self) -> Scope:
        return Scope(self.environment, self.stream, self.episode)

    def ref(self) -> ObsRef:
        return ObsRef(self.scope, self.seq, self.channel)

    @property
    def value(self):
        """payload["value"], or ABSENT if the payload has none."""
        from .sentinels import ABSENT
        return self.payload.get("value", ABSENT)


@register_record
@dataclass(frozen=True, eq=False)
class Action(Record):
    SCHEMA = "action"
    VERSION = 1

    environment: str
    stream: str
    episode: str
    seq: int
    spec_id: str
    command: Any
    duration: Any          # seconds >= 0, or UNKNOWN
    t_dispatch: float
    t_complete: Any = UNKNOWN   # finite >= t_dispatch, or UNKNOWN (pending)
    payload: Dict[str, Any] = field(default_factory=dict)

    def _validate(self):
        o = "Action"
        _scope_fields(o, self)
        req_str(o, "spec_id", self.spec_id)
        self._set("command", _command(o, self.command))
        self._set("duration", req_finite(o, "duration", self.duration,
                                         allow=(UNKNOWN,)))
        if self.duration is not UNKNOWN and self.duration < 0:
            fail(o, f"duration must be >= 0, got {self.duration}")
        self._set("t_dispatch", req_finite(o, "t_dispatch", self.t_dispatch))
        self._set("t_complete", req_finite(o, "t_complete", self.t_complete,
                                           allow=(UNKNOWN,)))
        if self.t_complete is not UNKNOWN and self.t_complete < self.t_dispatch:
            fail(o, f"t_complete {self.t_complete} precedes t_dispatch "
                    f"{self.t_dispatch}")
        self._set("payload", req_mapping(o, "payload", self.payload))

    @property
    def scope(self) -> Scope:
        return Scope(self.environment, self.stream, self.episode)


@register_record
@dataclass(frozen=True, eq=False)
class StateBelief(Record):
    """Variables a representation inferred, each with an uncertainty entry
    (UNKNOWN unless supplied — a belief never claims certainty by
    omission) and the observations that support it."""

    SCHEMA = "state_belief"
    VERSION = 1

    belief_id: str
    representation: str
    representation_version: int
    variables: Dict[str, Any]
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    support: Tuple[ObsRef, ...] = ()
    payload: Dict[str, Any] = field(default_factory=dict)

    def _validate(self):
        o = "StateBelief"
        req_str(o, "belief_id", self.belief_id)
        req_str(o, "representation", self.representation)
        self._set("representation_version",
                  req_int(o, "representation_version",
                          self.representation_version, minimum=1))
        var = req_mapping(o, "variables", self.variables)
        unc = req_mapping(o, "uncertainty", self.uncertainty)
        extra = set(unc) - set(var)
        if extra:
            fail(o, f"uncertainty for undeclared variables {sorted(extra)}")
        for k in var:
            unc.setdefault(k, UNKNOWN)
        self._set("variables", var)
        self._set("uncertainty", unc)
        self._set("support", req_tuple(o, "support", self.support, elem=ObsRef))
        self._set("payload", req_mapping(o, "payload", self.payload))


@register_record
@dataclass(frozen=True, eq=False)
class Mechanism(Record):
    SCHEMA = "mechanism"
    VERSION = 1

    mechanism_id: str
    version: int
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    applicability: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: Tuple[str, ...] = ()
    payload: Dict[str, Any] = field(default_factory=dict)

    def _validate(self):
        o = "Mechanism"
        req_str(o, "mechanism_id", self.mechanism_id)
        self._set("version", req_int(o, "version", self.version, minimum=1))
        self._set("inputs", req_str_tuple(o, "inputs", self.inputs, unique=True))
        self._set("outputs", req_str_tuple(o, "outputs", self.outputs,
                                           nonempty=True, unique=True))
        self._set("applicability", req_mapping(o, "applicability", self.applicability))
        par = req_mapping(o, "parameters", self.parameters)
        unc = req_mapping(o, "uncertainty", self.uncertainty)
        extra = set(unc) - set(par)
        if extra:
            fail(o, f"uncertainty for undeclared parameters {sorted(extra)}")
        for k in par:
            unc.setdefault(k, UNKNOWN)
        self._set("parameters", par)
        self._set("uncertainty", unc)
        self._set("evidence_refs", req_str_tuple(o, "evidence_refs", self.evidence_refs))
        self._set("payload", req_mapping(o, "payload", self.payload))


@register_record
@dataclass(frozen=True, eq=False)
class Prediction(Record):
    """An IMAGINED outcome. Carries the belief and model snapshot that made
    it, so it can be scored later and can never be mistaken for evidence."""

    SCHEMA = "prediction"
    VERSION = 1

    prediction_id: str
    source_belief: str
    model_versions: Dict[str, Any]
    snapshot_id: str
    action_spec_id: str
    candidate_action: Tuple[Any, ...]     # one command per horizon step
    horizon: int
    outcome: Dict[str, Any]               # must contain "kind"
    t_wall: float
    payload: Dict[str, Any] = field(default_factory=dict)

    def _validate(self):
        o = "Prediction"
        for f in ("prediction_id", "source_belief", "snapshot_id", "action_spec_id"):
            req_str(o, f, getattr(self, f))
        mv = req_mapping(o, "model_versions", self.model_versions)
        if not mv:
            fail(o, "model_versions must name at least one model")
        self._set("model_versions", mv)
        self._set("horizon", req_int(o, "horizon", self.horizon, minimum=1))
        ca = req_tuple(o, "candidate_action", self.candidate_action)
        if len(ca) != self.horizon:
            fail(o, f"candidate_action has {len(ca)} commands for horizon "
                    f"{self.horizon}")
        self._set("candidate_action", tuple(_command(o, c) for c in ca))
        out = req_mapping(o, "outcome", self.outcome)
        if not isinstance(out.get("kind"), str) or not out["kind"]:
            fail(o, "outcome must name its distribution in outcome['kind']")
        self._set("outcome", out)
        self._set("t_wall", req_finite(o, "t_wall", self.t_wall))
        self._set("payload", req_mapping(o, "payload", self.payload))


@register_record
@dataclass(frozen=True, eq=False)
class Experiment(Record):
    SCHEMA = "experiment"
    VERSION = 1

    experiment_id: str
    question: str
    alternatives: Tuple[str, ...]
    intervention: Dict[str, Any]
    predicted_evidence: Dict[str, Any]    # alternative -> prediction ids
    cost_budget: Dict[str, Any]           # resource -> finite amount >= 0
    payload: Dict[str, Any] = field(default_factory=dict)

    def _validate(self):
        o = "Experiment"
        req_str(o, "experiment_id", self.experiment_id)
        req_str(o, "question", self.question)
        alts = req_str_tuple(o, "alternatives", self.alternatives, unique=True)
        if len(alts) < 2:
            fail(o, "an experiment needs at least two alternatives to "
                    "distinguish")
        self._set("alternatives", alts)
        self._set("intervention", req_mapping(o, "intervention", self.intervention))
        pe = req_mapping(o, "predicted_evidence", self.predicted_evidence)
        bad = set(pe) - set(alts)
        if bad:
            fail(o, f"predicted_evidence for unknown alternatives {sorted(bad)}")
        self._set("predicted_evidence",
                  {k: req_str_tuple(o, f"predicted_evidence[{k!r}]", v)
                   for k, v in pe.items()})
        cb = req_mapping(o, "cost_budget", self.cost_budget)
        if not cb:
            fail(o, "cost_budget must bound at least one resource")
        for k, v in cb.items():
            v = req_finite(o, f"cost_budget[{k!r}]", v)
            if v < 0:
                fail(o, f"cost_budget[{k!r}] must be >= 0")
            cb[k] = v
        self._set("cost_budget", cb)
        self._set("payload", req_mapping(o, "payload", self.payload))


@register_record
@dataclass(frozen=True, eq=False)
class Evidence(Record):
    """What actually happened: observations and executed actions from ONE
    scope, the predictions it bears on, and whether it is usable."""

    SCHEMA = "evidence"
    VERSION = 1

    evidence_id: str
    observations: Tuple[Observation, ...]
    actions: Tuple[Action, ...]
    prediction_refs: Tuple[str, ...]
    validity: str
    provenance: str
    validity_reason: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def _validate(self):
        o = "Evidence"
        req_str(o, "evidence_id", self.evidence_id)
        obs = req_tuple(o, "observations", self.observations,
                        elem=Observation, nonempty=True)
        acts = req_tuple(o, "actions", self.actions, elem=Action)
        self._set("observations", obs)
        self._set("actions", acts)
        self._set("prediction_refs",
                  req_str_tuple(o, "prediction_refs", self.prediction_refs))
        req_enum(o, "validity", self.validity, VALIDITIES)
        req_enum(o, "provenance", self.provenance, PROVENANCES)
        req_str(o, "validity_reason", self.validity_reason, allow_empty=True)
        if self.validity != "valid" and not self.validity_reason:
            fail(o, f"validity {self.validity!r} needs a validity_reason")
        scope = obs[0].scope
        for r in obs + acts:
            if r.scope != scope:
                fail(o, f"mixes scopes {scope} and {r.scope}; one Evidence "
                        f"record never spans streams, episodes or "
                        f"environments")
        for ob in obs:
            if ob.provenance != self.provenance:
                fail(o, f"{self.provenance!r} evidence contains a "
                        f"{ob.provenance!r} observation ({ob.channel!r}); a "
                        f"model cannot certify its own imagined outcome")
        self._set("payload", req_mapping(o, "payload", self.payload))

    @property
    def scope(self) -> Scope:
        return self.observations[0].scope


@register_record
@dataclass(frozen=True, eq=False)
class Skill(Record):
    """A learned controller. Identity is `skill_id` + `version`, never a
    display name (CLAUDE.md §4.6: names carry no weight). Both initiation
    and termination must be declared: a skill with no way to end is a
    latch (§4.1)."""

    SCHEMA = "skill"
    VERSION = 1

    skill_id: str
    version: int
    controller_ref: str
    initiation: Dict[str, Any]
    termination: Dict[str, Any]
    competence_evidence: Tuple[str, ...] = ()
    payload: Dict[str, Any] = field(default_factory=dict)

    def _validate(self):
        o = "Skill"
        req_str(o, "skill_id", self.skill_id)
        self._set("version", req_int(o, "version", self.version, minimum=1))
        req_str(o, "controller_ref", self.controller_ref)
        for f in ("initiation", "termination"):
            m = req_mapping(o, f, getattr(self, f))
            if not m:
                fail(o, f"{f} must be declared (non-empty)")
            self._set(f, m)
        self._set("competence_evidence",
                  req_str_tuple(o, "competence_evidence", self.competence_evidence))
        self._set("payload", req_mapping(o, "payload", self.payload))
