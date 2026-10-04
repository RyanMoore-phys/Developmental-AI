"""Representation versioning: dependents are invalidated or translated, never
silently reused (plan Stage 8 item 6).

WHAT IS CLAIMED
    1. Every perception representation has a SEMANTIC version "MAJOR.MINOR.
       PATCH". MAJOR changes when the MEANING of a variable changes (a
       bearing in [-1, 1] becomes a bearing in radians; slot k stops meaning
       the same thing); MINOR/PATCH are declared compatible additions/fixes.
       `StateBelief.representation_version` (an int, contracts) carries the
       MAJOR number; the full semver rides in `payload["representation_
       semver"]`.
    2. Anything built ON a representation — a cache of beliefs, a fitted
       mechanism, a skill's initiation set — registers as a DEPENDENT bound
       to the version it was built against.
    3. A MAJOR bump visits every dependent bound to an older major: if a
       translator chain old->new is registered, the dependent's data is
       passed through it and the dependent stays usable (status
       "translated"); otherwise (or if the translator raises) it is
       INVALIDATED. `use()` on an invalidated dependent RAISES
       StaleDependentError. MINOR/PATCH bumps leave dependents valid.
    4. `check_belief` raises on a StateBelief whose major is not current.

WHY
    CLAUDE.md §4.6: skills are byte copies that re-derive from goal
    slot_keys — so a slot whose meaning changed under a skill does not fail,
    it quietly means something else. The only safe default for a semantic
    change is "stop using it until someone translates or rebuilds it".

ESCAPE PATH (CLAUDE.md §4.1 — every guard needs one)
    An invalidated dependent is not a latch: `rebuild(dep_id, data)` rebinds
    it to the CURRENT version with freshly derived data (refit the mechanism,
    recollect the cache). What reopens it is re-deriving it, which is the
    caller's own action — not a human noticing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .errors import PerceptionError, StaleDependentError

_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
DEPENDENT_KINDS = ("cache", "mechanism", "skill", "belief_consumer")
DEPENDENT_STATUSES = ("valid", "translated", "invalidated")


@dataclass(frozen=True, order=True)
class SemVer:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, s) -> "SemVer":
        if isinstance(s, SemVer):
            return s
        m = _SEMVER.match(s) if isinstance(s, str) else None
        if not m:
            raise PerceptionError(
                f"version must be 'MAJOR.MINOR.PATCH', got {s!r}")
        v = cls(*(int(g) for g in m.groups()))
        if v.major < 1:
            raise PerceptionError(
                f"major version must be >= 1 (StateBelief.representation_"
                f"version is the major and must be >= 1), got {s!r}")
        return v

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass
class Dependent:
    dep_id: str
    kind: str
    representation: str
    bound_version: SemVer
    data: Any = None
    status: str = "valid"
    history: List[str] = field(default_factory=list)


class RepresentationRegistry:
    """Versions, translators and dependents for perception representations."""

    def __init__(self):
        self._versions: Dict[str, SemVer] = {}
        self._descriptions: Dict[str, str] = {}
        self._translators: Dict[Tuple[str, int], Tuple[int, Callable]] = {}
        self._deps: Dict[str, Dependent] = {}

    # ---------------------------------------------------------- versions
    def declare(self, name: str, version: str = "1.0.0",
                description: str = "") -> SemVer:
        if not isinstance(name, str) or not name:
            raise PerceptionError("representation name must be a non-empty str")
        v = SemVer.parse(version)
        if name in self._versions:
            if self._versions[name] != v:
                raise PerceptionError(
                    f"{name!r} already declared at {self._versions[name]}; "
                    f"use bump() to change it")
            return v
        self._versions[name] = v
        self._descriptions[name] = description
        return v

    def known(self, name: str) -> bool:
        return name in self._versions

    def current(self, name: str) -> SemVer:
        try:
            return self._versions[name]
        except KeyError:
            raise PerceptionError(
                f"representation {name!r} was never declared; known: "
                f"{sorted(self._versions)}") from None

    def major(self, name: str) -> int:
        return self.current(name).major

    # ------------------------------------------------------- translators
    def register_translator(self, name: str, from_major: int, to_major: int,
                            fn: Callable[[Any, Dependent], Any]) -> None:
        """fn(data, dependent) -> translated data. One hop per major step."""
        self.current(name)
        if int(to_major) <= int(from_major):
            raise PerceptionError("a translator must go to a newer major")
        if not callable(fn):
            raise PerceptionError("translator must be callable")
        self._translators[(name, int(from_major))] = (int(to_major), fn)

    def _chain(self, name: str, from_major: int, to_major: int
               ) -> Optional[List[Callable]]:
        fns, m = [], from_major
        while m < to_major:
            hop = self._translators.get((name, m))
            if hop is None or hop[0] > to_major:
                return None
            fns.append(hop[1])
            m = hop[0]
        return fns if m == to_major else None

    # -------------------------------------------------------- dependents
    def register_dependent(self, dep_id: str, kind: str, representation: str,
                           data: Any = None) -> Dependent:
        if kind not in DEPENDENT_KINDS:
            raise PerceptionError(f"kind must be one of {DEPENDENT_KINDS}, "
                                  f"got {kind!r}")
        if not isinstance(dep_id, str) or not dep_id:
            raise PerceptionError("dep_id must be a non-empty str")
        if dep_id in self._deps:
            raise PerceptionError(f"dependent {dep_id!r} already registered; "
                                  f"use rebuild() to rebind it")
        d = Dependent(dep_id, kind, representation,
                      self.current(representation), data)
        d.history.append(f"registered@{d.bound_version}")
        self._deps[dep_id] = d
        return d

    def dependent(self, dep_id: str) -> Dependent:
        try:
            return self._deps[dep_id]
        except KeyError:
            raise PerceptionError(f"unknown dependent {dep_id!r}") from None

    def use(self, dep_id: str) -> Any:
        """The dependent's data, or StaleDependentError if it no longer
        describes the current representation."""
        d = self.dependent(dep_id)
        cur = self.current(d.representation)
        if d.status == "invalidated" or d.bound_version.major != cur.major:
            raise StaleDependentError(
                f"{d.kind} {dep_id!r} was built on {d.representation} "
                f"{d.bound_version}, current is {cur} (status {d.status}); "
                f"rebuild() it against the current version")
        return d.data

    def rebuild(self, dep_id: str, data: Any) -> Dependent:
        """Escape path: rebind a dependent to the current version with
        freshly derived data."""
        d = self.dependent(dep_id)
        d.bound_version = self.current(d.representation)
        d.data = data
        d.status = "valid"
        d.history.append(f"rebuilt@{d.bound_version}")
        return d

    # -------------------------------------------------------------- bump
    def bump(self, name: str, new_version: str, reason: str
             ) -> Dict[str, List[str]]:
        """Advance a representation. Returns {"translated", "invalidated",
        "unchanged"} lists of dependent ids."""
        if not isinstance(reason, str) or not reason.strip():
            raise PerceptionError("a version bump must state its reason")
        old = self.current(name)
        new = SemVer.parse(new_version)
        if new <= old:
            raise PerceptionError(f"{name}: {new} is not newer than {old}")
        self._versions[name] = new
        report: Dict[str, List[str]] = {"translated": [], "invalidated": [],
                                        "unchanged": []}
        for d in self._deps.values():
            if d.representation != name or d.status == "invalidated":
                continue
            if d.bound_version.major == new.major:
                d.bound_version = new           # declared compatible
                d.history.append(f"compatible@{new}")
                report["unchanged"].append(d.dep_id)
                continue
            chain = self._chain(name, d.bound_version.major, new.major)
            if chain is None:
                d.status = "invalidated"
                d.history.append(f"invalidated@{new}: no translator "
                                 f"{d.bound_version.major}->{new.major} ({reason})")
                report["invalidated"].append(d.dep_id)
                continue
            try:
                data = d.data
                for fn in chain:
                    data = fn(data, d)
            except Exception as e:              # translator refused
                d.status = "invalidated"
                d.history.append(f"invalidated@{new}: translator raised {e!r}")
                report["invalidated"].append(d.dep_id)
                continue
            d.data = data
            d.bound_version = new
            d.status = "translated"
            d.history.append(f"translated@{new}")
            report["translated"].append(d.dep_id)
        return report

    # ------------------------------------------------------------ beliefs
    def check_belief(self, belief) -> None:
        """Raise unless a StateBelief was produced by the current major of a
        declared representation."""
        cur = self.current(belief.representation)
        if belief.representation_version != cur.major:
            raise StaleDependentError(
                f"belief {belief.belief_id!r} is {belief.representation} "
                f"v{belief.representation_version}, current major is "
                f"{cur.major}")
        sv = belief.payload.get("representation_semver")
        if isinstance(sv, str) and SemVer.parse(sv).major != cur.major:
            raise StaleDependentError(
                f"belief {belief.belief_id!r} payload semver {sv} disagrees "
                f"with current {cur}")

    def stamp(self, name: str) -> Tuple[int, str]:
        """(major for StateBelief.representation_version, full semver)."""
        v = self.current(name)
        return v.major, str(v)
