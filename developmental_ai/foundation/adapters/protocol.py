"""The EnvironmentAdapter contract and its reusable conformance check.

An adapter turns one environment into records. It DECLARES its channels
(ObservationSpec) and its control (ActionSpec), and then:

    reset(seed=None) -> list[Observation]
        A NEW episode id that no earlier observation of this adapter has
        used, every observation at seq 0. Identities never carry across it.

    step(action) -> (list[Observation], terminated, truncated, info)
        `action` must target the CURRENT scope and seq; a stale action (from
        a previous episode, or a seq already consumed) and an out-of-spec
        command are rejected with a ContractError and change nothing.

        Three different endings, never conflated:
          terminated       the environment's own end (goal reached, death).
                           No bootstrap past it. Next call: reset().
          truncated        an external cut (time limit). The state was not
                           terminal, so value bootstrap is legitimate. Next
                           call: reset().
          client_recovery  info["client_recovery"] — the client/connection
                           was rebuilt mid-stream. Not an environment end:
                           terminated and truncated are both False, and the
                           returned observations ALREADY belong to a fresh
                           episode id at seq 0, so no consumer can splice
                           across the rebuild (transition_valid rejects it
                           on the episode rule, with no flag to forget).
        info["executed_action"] is the Action as executed, with t_complete
        filled in (>= t_dispatch).

    Every observation carries payload["value"] (a reading, ABSENT or
    UNKNOWN) matching its ChannelSpec's shape and dtype, and the channel's
    declared provenance. Required channels appear on every step.

`policy_observations` is the consumer-side gate: learning reads only what
it returns, and it never returns an evaluator channel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                   # py3.8+: typing.Protocol
    from typing import Protocol, runtime_checkable
except ImportError:                    # pragma: no cover
    Protocol = object

    def runtime_checkable(c):
        return c

from ..contracts import (UNKNOWN, Action, ActionSpec, AdapterConformanceError,
                         ContractError, Observation, ObservationSpec,
                         deep_equal, transition_valid)


@runtime_checkable
class EnvironmentAdapter(Protocol):
    def observation_spec(self) -> ObservationSpec: ...

    def action_spec(self) -> ActionSpec: ...

    def reset(self, seed: Optional[int] = None) -> List[Observation]: ...

    def step(self, action: Action) -> Tuple[List[Observation], bool, bool, Dict]: ...


def policy_observations(observations: Sequence[Observation],
                        spec: ObservationSpec) -> List[Observation]:
    """The observations learning may consume. Decided by the SPEC's channel
    declaration AND the record's own provenance, so neither a mislabelled
    record nor a mislabelled channel can let evaluator data through."""
    allowed = set(spec.policy_channels())
    return [o for o in observations
            if o.channel in allowed and o.provenance != "evaluator"]


@dataclass
class ConformanceReport:
    steps: int = 0
    episodes: int = 0
    terminations: int = 0
    truncations: int = 0
    recoveries: int = 0
    channels_seen: set = field(default_factory=set)
    action_kind: str = ""
    problems: List[str] = field(default_factory=list)


def observation_batch_problems(obs, spec: ObservationSpec, seq: int,
                               where: str = "batch") -> List[str]:
    """Why one step's observations break the contract against `spec`
    ([] if they do not). Reusable for producers that are not full adapters,
    such as SkyBotRecordAdapter."""
    p = []
    if not isinstance(obs, list) or not obs:
        return [f"{where}: must return a non-empty list of Observation"]
    if not all(isinstance(o, Observation) for o in obs):
        return [f"{where}: non-Observation in batch"]
    declared = {c.name: c for c in spec.channels}
    scopes = {o.scope for o in obs}
    if len(scopes) != 1:
        p.append(f"{where}: one step spans scopes {scopes}")
    chans = [o.channel for o in obs]
    if len(set(chans)) != len(chans):
        p.append(f"{where}: duplicate channels {chans}")
    for c in spec.channels:
        if c.required and c.name not in chans:
            p.append(f"{where}: required channel {c.name!r} missing")
    for o in obs:
        if o.environment != spec.environment:
            p.append(f"{where}: environment {o.environment!r} != spec "
                     f"{spec.environment!r}")
        if o.seq != seq:
            p.append(f"{where}: {o.channel!r} seq {o.seq}, expected {seq}")
        c = declared.get(o.channel)
        if c is None:
            p.append(f"{where}: undeclared channel {o.channel!r}")
            continue
        if o.provenance != c.provenance:
            p.append(f"{where}: {o.channel!r} provenance {o.provenance!r} != "
                     f"declared {c.provenance!r}")
        if "value" not in o.payload:
            p.append(f"{where}: {o.channel!r} payload has no 'value'")
        else:
            p += [f"{where}: {o.channel!r} {m}" for m in c.value_problems(o.value)]
    return p


def _out_of_spec(spec: ActionSpec):
    if spec.kind == "discrete":
        return spec.n
    if spec.kind == "multi_discrete":
        return tuple(spec.nvec)
    hi = np.where(np.isfinite(spec.high), spec.high + 1.0, np.nan)
    if np.isnan(hi).all():
        return np.full(spec.shape, np.nan)       # rejected as non-finite
    return np.where(np.isnan(hi), 0.0, hi)


def _expect_reject(adapter, act, what, problems):
    try:
        adapter.step(act)
    except ContractError:
        return
    except Exception as e:                       # wrong error type
        problems.append(f"{what}: raised {type(e).__name__}, expected ContractError")
        return
    problems.append(f"{what}: was accepted")


def check_adapter_conformance(adapter, n_steps: int = 50, seed: int = 0,
                              raise_on_failure: bool = True) -> ConformanceReport:
    """Drive `adapter` for n_steps of random in-spec actions and check every
    clause of the contract above. Raises AdapterConformanceError listing ALL
    problems (or returns them in the report's `problems` attribute when
    raise_on_failure=False)."""
    rng = np.random.default_rng(seed)
    problems: List[str] = []
    rep = ConformanceReport()
    ospec, aspec = adapter.observation_spec(), adapter.action_spec()
    if not isinstance(ospec, ObservationSpec):
        problems.append("observation_spec() must return an ObservationSpec")
    if not isinstance(aspec, ActionSpec):
        problems.append("action_spec() must return an ActionSpec")
    if problems:
        raise AdapterConformanceError(problems)
    rep.action_kind = aspec.kind
    episodes_seen = set()

    def begin(obs, where):
        problems.extend(observation_batch_problems(obs, ospec, 0, where))
        ep = obs[0].episode if obs and isinstance(obs[0], Observation) else None
        if ep in episodes_seen:
            problems.append(f"{where}: episode id {ep!r} reused")
        episodes_seen.add(ep)
        rep.episodes += 1
        return obs

    prev = begin(adapter.reset(seed=seed), "reset")
    rep.channels_seen.update(o.channel for o in prev)
    rejected_once = False
    for i in range(n_steps):
        if problems:
            break
        head = prev[0]
        mk = lambda cmd, spec_id=aspec.spec_id, ep=head.episode, seq=head.seq: Action(
            head.environment, head.stream, ep, seq, spec_id, cmd, UNKNOWN,
            time.time())
        if not rejected_once:
            rejected_once = True
            _expect_reject(adapter, mk(_out_of_spec(aspec)), "out-of-spec command", problems)
            _expect_reject(adapter, mk(aspec.sample(rng), spec_id=aspec.spec_id + "#other"),
                           "action for another spec", problems)
            _expect_reject(adapter, mk(aspec.sample(rng), ep=head.episode + "#stale"),
                           "action from another episode", problems)
            _expect_reject(adapter, mk(aspec.sample(rng), seq=head.seq + 7),
                           "action with a wrong seq", problems)
            if problems:          # a wrongly accepted action moved its state
                break
        cmd = aspec.sample(rng)
        act = mk(cmd)
        try:
            out = adapter.step(act)
        except Exception as e:
            problems.append(f"step {i}: in-spec action raised "
                            f"{type(e).__name__}: {e}")
            break
        rep.steps += 1
        where = f"step {i}"
        if not (isinstance(out, tuple) and len(out) == 4):
            problems.append(f"{where}: must return (obs, terminated, truncated, info)")
            break
        obs, term, trunc, info = out
        if not isinstance(term, bool) or not isinstance(trunc, bool):
            problems.append(f"{where}: terminated/truncated must be bool")
        if not isinstance(info, dict) or not isinstance(info.get("client_recovery"), bool):
            problems.append(f"{where}: info['client_recovery'] must be a bool")
            break
        ex = info.get("executed_action")
        if not isinstance(ex, Action):
            problems.append(f"{where}: info['executed_action'] must be an Action")
        else:
            if ex.scope != act.scope or ex.seq != act.seq or ex.spec_id != act.spec_id:
                problems.append(f"{where}: executed_action scope/seq/spec differs")
            if not deep_equal(aspec.validate_command(ex.command),
                              aspec.validate_command(act.command)):
                problems.append(f"{where}: executed_action command differs")
            if ex.t_complete is UNKNOWN:
                problems.append(f"{where}: executed_action.t_complete not set")
        recovery = info["client_recovery"]
        if recovery:
            rep.recoveries += 1
            if term or trunc:
                problems.append(f"{where}: client_recovery reported together "
                                f"with terminated/truncated")
            new = begin(obs, where + " (recovery)")
            if new and new[0].episode == head.episode:
                problems.append(f"{where}: recovery kept the old episode id")
            _check_pairs(prev, new, False, where, problems)
            prev = new
            continue
        problems.extend(observation_batch_problems(obs, ospec, head.seq + 1, where))
        if obs and obs[0].episode != head.episode:
            problems.append(f"{where}: episode changed without reset/recovery")
        _check_pairs(prev, obs, True, where, problems)
        rep.channels_seen.update(o.channel for o in obs)
        if term or trunc:
            rep.terminations += int(term)
            rep.truncations += int(trunc)
            new = begin(adapter.reset(), f"reset after {where}")
            _check_pairs(obs, new, False, f"reset after {where}", problems)
            prev = new
        else:
            prev = obs
    rep.problems = problems
    if problems and raise_on_failure:
        raise AdapterConformanceError(problems)
    return rep


def _check_pairs(prev, nxt, same_episode, where, problems):
    """Within an episode, sensor channels must chain; across a boundary,
    nothing may. Evaluator channels never form a trainable transition."""
    before = {o.channel: o for o in prev}
    for o in nxt:
        p = before.get(o.channel)
        if p is None:
            continue
        ok = transition_valid(p, o)
        trainable = same_episode and o.provenance == "sensor" and p.provenance == "sensor"
        if ok != trainable:
            problems.append(
                f"{where}: transition on {o.channel!r} valid={ok}, expected "
                f"{trainable}")
