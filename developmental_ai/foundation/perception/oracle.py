"""PRIVILEGED-STATE DIAGNOSTICS — evaluator only; never a learning input.

WHAT IS CLAIMED (plan Stage 8 completion gate: "Separate privileged-state
diagnostics from pixel-based results")
    1. `oracle_diagnostics(track_log, truth)` compares an IdentityTracker's
       exported log with EVALUATOR truth (which detection came from which
       true entity) and reports ID switches, fragmentation, track purity,
       confirmed clutter tracks and CONFIDENT-WRONG associations.
    2. It accepts truth ONLY as an `EvaluatorTruth` whose provenance is
       "evaluator"; anything else raises PrivilegedInputError. Truth labelled
       "sensor" is exactly the relabelling the contracts smoke (contract B)
       refuses, refused here too.
    3. It cannot feed back: it reads a deep-copied log (`export_log`), returns
       a frozen `OracleReport` stamped provenance="evaluator", and the
       tracker refuses any non-ndarray input (so neither the truth nor the
       report can be passed to `step`). No other perception module imports
       this one (asserted by the unit test, by source scan).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .errors import PerceptionError, PrivilegedInputError


@dataclass(frozen=True)
class EvaluatorTruth:
    """Per frame: `det_entity[t]` = tuple of true entity ids, one per
    detection in the order given to the tracker (-1 = clutter)."""

    det_entity: Tuple[Tuple[int, ...], ...]
    provenance: str

    def __post_init__(self):
        if self.provenance != "evaluator":
            raise PrivilegedInputError(
                f"EvaluatorTruth provenance must be 'evaluator', got "
                f"{self.provenance!r}")
        object.__setattr__(self, "det_entity",
                           tuple(tuple(int(e) for e in fr)
                                 for fr in self.det_entity))


@dataclass(frozen=True)
class OracleReport:
    metrics: Mapping[str, Any]
    provenance: str = "evaluator"


def oracle_diagnostics(track_log: Sequence[Dict[str, Any]],
                       truth: EvaluatorTruth,
                       confident: float = 0.8) -> OracleReport:
    if not isinstance(truth, EvaluatorTruth):
        raise PrivilegedInputError(
            f"truth must be an EvaluatorTruth (provenance 'evaluator'), got "
            f"{type(truth).__name__}")
    if len(track_log) != len(truth.det_entity):
        raise PerceptionError(f"log has {len(track_log)} frames, truth has "
                              f"{len(truth.det_entity)}")
    aliases = dict(track_log[-1].get("aliases", {})) if track_log else {}

    def res(l):
        seen = set()
        while l in aliases and l not in seen:
            seen.add(l)
            l = aliases[l]
        return l

    ent_seq: Dict[int, List[str]] = {}
    track_ents: Dict[str, List[int]] = {}
    track_first: Dict[str, int] = {}
    confident_wrong = 0
    confirmed_dets = 0
    for rec, ents in zip(track_log, truth.det_entity):
        assign = rec["assign"]
        if len(assign) != len(ents):
            raise PerceptionError(f"frame {rec['frame']}: {len(assign)} "
                                  f"detections logged, {len(ents)} in truth")
        for loc, e in zip(assign, ents):
            if loc is None or rec["states"].get(loc) != "confirmed":
                continue
            r = res(loc)
            confirmed_dets += 1
            track_ents.setdefault(r, []).append(e)
            track_first.setdefault(r, e)
            if e >= 0:
                ent_seq.setdefault(e, []).append(r)
            if e != track_first[r] and rec["confidence"].get(loc, 1.0) > confident:
                confident_wrong += 1
    id_switches = {e: sum(1 for a, b in zip(s[:-1], s[1:]) if a != b)
                   for e, s in ent_seq.items()}
    fragments = {e: len(set(s)) for e, s in ent_seq.items()}
    purity, clutter_tracks = {}, 0
    for t, es in track_ents.items():
        c = Counter(es)
        top, n = c.most_common(1)[0]
        purity[t] = n / len(es)
        if top < 0:
            clutter_tracks += 1
    m = {"id_switches": id_switches,
         "total_id_switches": int(sum(id_switches.values())),
         "fragments": fragments,
         "purity": purity,
         "false_confirmed_tracks": int(clutter_tracks),
         "confident_wrong": int(confident_wrong),
         "confirmed_detections": int(confirmed_dets),
         "entity_tracks": {e: sorted(set(s)) for e, s in ent_seq.items()}}
    return OracleReport(MappingProxyType(m))
