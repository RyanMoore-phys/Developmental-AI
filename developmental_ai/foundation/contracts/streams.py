"""Which pairs of observations may form a training transition.

Plan §7.4: "Reset splices and unrelated streams cannot become valid training
transitions." The replay buffer learned this the hard way — a window that
crossed the write head taught the model a splice between two unrelated
worlds as if it were dynamics (tests/_replay_sampling_smoke.py). Here the
rule is stated once, on records, for every consumer:

    same environment, same stream, same episode     (no cross-stream, no
                                                     reset splice)
    same channel                                    (no cross-modal pair)
    next.seq == prev.seq + 1                        (adjacent; no gap)
    t_env and t_wall do not run backwards
    both provenances in `allow`, default ("sensor",) — imagined and
                                                     evaluator data are not
                                                     real transitions

A client recovery or a reset always starts a new episode id (adapter
contract), so it is caught by the episode rule rather than by a flag the
consumer could forget to read.
"""

from __future__ import annotations

from typing import List, Sequence

from .records import Observation

TRAINABLE_PROVENANCE = ("sensor",)


def transition_problems(prev: Observation, nxt: Observation,
                        allow: Sequence[str] = TRAINABLE_PROVENANCE) -> List[str]:
    """Every reason (prev -> nxt) is not a valid transition; [] if valid."""
    if not isinstance(prev, Observation) or not isinstance(nxt, Observation):
        return ["both ends must be Observation records"]
    p = []
    if prev.environment != nxt.environment:
        p.append(f"cross-environment: {prev.environment!r} -> {nxt.environment!r}")
    if prev.stream != nxt.stream:
        p.append(f"cross-stream: {prev.stream!r} -> {nxt.stream!r}")
    if prev.episode != nxt.episode:
        p.append(f"cross-episode (reset splice): {prev.episode!r} -> {nxt.episode!r}")
    if prev.channel != nxt.channel:
        p.append(f"cross-channel: {prev.channel!r} -> {nxt.channel!r}")
    if nxt.seq != prev.seq + 1:
        p.append(f"non-adjacent seq: {prev.seq} -> {nxt.seq}")
    if nxt.t_env < prev.t_env:
        p.append(f"t_env runs backwards: {prev.t_env} -> {nxt.t_env}")
    if nxt.t_wall < prev.t_wall:
        p.append(f"t_wall runs backwards: {prev.t_wall} -> {nxt.t_wall}")
    for end, ob in (("prev", prev), ("next", nxt)):
        if ob.provenance not in allow:
            p.append(f"{end} provenance {ob.provenance!r} not trainable "
                     f"(allowed {tuple(allow)})")
    return p


def transition_valid(prev: Observation, nxt: Observation,
                     allow: Sequence[str] = TRAINABLE_PROVENANCE) -> bool:
    return not transition_problems(prev, nxt, allow)
