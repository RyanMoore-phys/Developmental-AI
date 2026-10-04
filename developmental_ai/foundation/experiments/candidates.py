"""Experiment candidates drawn ONLY from what the adapter can actually do
(plan Stage 10 items 1 and 4).

A candidate is "do this, for this long": a primitive command from the
adapter's ActionSpec held for `duration_ticks`, or a learned Skill record run
for its declared maximum. Nothing here invents an action:

  * primitive commands are enumerated from (discrete, multi_discrete) or
    sampled inside (box) the ActionSpec, and every one is passed through
    `ActionSpec.validate_command` before it becomes a candidate;
  * a skill is admitted only if its record says which ActionSpec it drives
    (`payload["spec_id"]`) and that spec is the adapter's, and only if it
    declares a bounded length (`payload["max_ticks"]`). A skill with no
    declared end is a latch (CLAUDE.md §4.1) and is REJECTED, with the reason
    returned — never silently dropped;
  * durations are only offered where the spec's `duration_mode` lets the
    caller choose them ("tick" / "held"); an "until_complete" action runs to
    its own end, so it is offered once with duration 1 (one invocation).

SHORT HORIZONS, CAPPED COUNTS (item 4). `max_horizon` (ticks) and
`max_candidates` are required to sit under hard ceilings MAX_HORIZON and
MAX_CANDIDATES. When the full set is larger than `max_candidates`, a uniform
random subset is drawn — FRESHLY ON EVERY CALL, which needs an rng. A fixed
subset would be a permanent exclusion of the remainder (guard-becomes-latch,
§4.1); a re-drawn one gives every admissible candidate a positive chance on
every step, which is the stated recovery path. `truncated` says it happened.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ..contracts import ActionSpec, ContractError, INAPPLICABLE, Skill

MAX_CANDIDATES = 64      # hard ceiling on candidates scored per decision
MAX_HORIZON = 32         # hard ceiling on one candidate's length, in ticks
PRIMITIVE, SKILL = "primitive", "skill"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    kind: str                       # PRIMITIVE | SKILL
    spec_id: str                    # the adapter ActionSpec it executes on
    command: Any                    # canonical command; INAPPLICABLE for a skill
    duration_ticks: int
    skill_id: Optional[str] = None
    skill_version: Optional[int] = None

    def __post_init__(self):
        if self.kind not in (PRIMITIVE, SKILL):
            raise ContractError(f"candidate kind {self.kind!r} not in {(PRIMITIVE, SKILL)}")
        if isinstance(self.duration_ticks, bool) or not isinstance(
                self.duration_ticks, (int, np.integer)) or self.duration_ticks < 1:
            raise ContractError(f"duration_ticks must be an int >= 1, got "
                                f"{self.duration_ticks!r}")
        if (self.kind == SKILL) != (self.skill_id is not None):
            raise ContractError("skill_id is set iff kind == 'skill'")


@dataclass(frozen=True)
class CandidateSet:
    candidates: Tuple[Candidate, ...]
    rejected: Tuple[Tuple[str, str], ...]     # (what, why) — never silent
    truncated: bool
    n_admissible: int

    def __len__(self):
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)

    def by_id(self, cid: str) -> Candidate:
        for c in self.candidates:
            if c.candidate_id == cid:
                return c
        raise KeyError(cid)


def _cmd_label(cmd) -> str:
    if isinstance(cmd, np.ndarray):
        return "[" + ",".join(f"{x:.4g}" for x in cmd.ravel()) + "]"
    return str(cmd)


def _primitive_commands(spec: ActionSpec, box_samples: int,
                        rng: Optional[np.random.Generator], cap: int) -> List[Any]:
    if spec.kind == "discrete":
        return list(range(int(spec.n)))
    if spec.kind == "multi_discrete":
        total = int(np.prod([int(n) for n in spec.nvec]))
        if total <= 4 * cap:
            return [tuple(c) for c in itertools.product(*[range(int(n)) for n in spec.nvec])]
        if rng is None:
            raise ContractError(f"multi_discrete spec has {total} commands; sampling "
                                f"them needs an rng (a fixed subset is a latch)")
        seen, out = set(), []
        for _ in range(8 * cap):
            c = spec.sample(rng)
            if c not in seen:
                seen.add(c)
                out.append(c)
            if len(out) >= 2 * cap:
                break
        return out
    # box: the centre of the finite box, plus uniform samples inside it
    if rng is None:
        raise ContractError("a box ActionSpec has no finite command list; "
                            "sampling it needs an rng")
    lo, hi = spec.low, spec.high
    out = [np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi),
                    np.clip(0.0, lo, hi)).reshape(spec.shape)]
    out += [spec.sample(rng) for _ in range(int(box_samples))]
    return out


def enumerate_candidates(action_spec: ActionSpec, durations: Sequence[int] = (1,),
                         skills: Sequence[Skill] = (), max_candidates: int = 16,
                         max_horizon: int = 8, box_samples: int = 8,
                         rng: Optional[np.random.Generator] = None) -> CandidateSet:
    if not isinstance(action_spec, ActionSpec):
        raise ContractError("candidates come from the adapter's ActionSpec")
    if isinstance(max_candidates, bool) or not 1 <= int(max_candidates) <= MAX_CANDIDATES:
        raise ContractError(f"max_candidates must be in [1, {MAX_CANDIDATES}]")
    if isinstance(max_horizon, bool) or not 1 <= int(max_horizon) <= MAX_HORIZON:
        raise ContractError(f"max_horizon must be in [1, {MAX_HORIZON}] ticks "
                            f"(Stage 10 begins with short horizons)")
    max_candidates, max_horizon = int(max_candidates), int(max_horizon)
    rejected: List[Tuple[str, str]] = []
    durs: List[int] = []
    for d in durations:
        if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 1:
            raise ContractError(f"duration {d!r} must be an int >= 1 tick")
        if d > max_horizon:
            rejected.append((f"duration={d}", f"exceeds max_horizon={max_horizon}"))
            continue
        if action_spec.duration_mode == "until_complete" and d != 1:
            rejected.append((f"duration={d}", "spec duration_mode is until_complete: "
                             "the action ends itself, its length is not chosen"))
            continue
        if int(d) not in durs:
            durs.append(int(d))
    if not durs and action_spec.duration_mode == "until_complete":
        durs = [1]
    admissible: List[Candidate] = []
    for cmd in _primitive_commands(action_spec, box_samples, rng, max_candidates):
        cmd = action_spec.validate_command(cmd)          # never an invented action
        for d in durs:
            admissible.append(Candidate(f"{action_spec.spec_id}:{_cmd_label(cmd)}x{d}",
                                        PRIMITIVE, action_spec.spec_id, cmd, d))
    for sk in skills:
        if not isinstance(sk, Skill):
            raise ContractError(f"skills must be contracts.Skill records, got "
                                f"{type(sk).__name__}")
        what = f"skill {sk.skill_id}@v{sk.version}"
        sid = sk.payload.get("spec_id")
        if sid != action_spec.spec_id:
            rejected.append((what, f"drives spec {sid!r}; this adapter offers "
                                   f"{action_spec.spec_id!r}"))
            continue
        mt = sk.payload.get("max_ticks")
        if isinstance(mt, bool) or not isinstance(mt, (int, np.integer)) or mt < 1:
            rejected.append((what, "no declared max_ticks: a skill with no bounded "
                                   "end is a latch (CLAUDE.md §4.1)"))
            continue
        if mt > max_horizon:
            rejected.append((what, f"max_ticks={mt} exceeds max_horizon={max_horizon}"))
            continue
        admissible.append(Candidate(f"skill:{sk.skill_id}@v{sk.version}", SKILL,
                                    action_spec.spec_id, INAPPLICABLE, int(mt),
                                    sk.skill_id, int(sk.version)))
    ids = [c.candidate_id for c in admissible]
    if len(set(ids)) != len(ids):
        raise ContractError("duplicate candidate ids")
    n = len(admissible)
    truncated = n > max_candidates
    if truncated:
        if rng is None:
            raise ContractError(f"{n} admissible candidates > max_candidates="
                                f"{max_candidates}: truncation needs an rng so the "
                                f"subset is re-drawn each call (a fixed subset is a "
                                f"permanent exclusion, §4.1)")
        keep = np.sort(rng.choice(n, max_candidates, replace=False))
        admissible = [admissible[i] for i in keep]
    return CandidateSet(tuple(admissible), tuple(rejected), truncated, n)
