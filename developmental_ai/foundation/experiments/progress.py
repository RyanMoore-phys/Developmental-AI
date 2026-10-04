"""Progress measured on a DEVELOPMENT probe set only (plan Stage 10 item 6).

Selection signals may be steered by progress; final evaluation data may
never steer anything. So the only way to build a progress measurement here
is through a `DevProbeSet`, and a DevProbeSet refuses any probe whose
episode is on the held-out side of the split:

  * with a `DataSplit` (experiments.ab.make_split): the probe's scope must be
    the split's scope and its episode must be listed in `split.dev`; an
    episode in `split.heldout` raises HeldOutAccessError, an episode on
    neither side raises ContractError (unknown is not dev);
  * without one: runtime.splits.assign_split(scope, episode, fraction, salt)
    is computed and HELDOUT raises.

The unit is the episode (runtime.splits): a probe built from frame t of a
held-out episode is refused exactly like the episode itself. `verify()`
re-checks every probe and runs before every measurement, so a probe set
that was mutated behind the API's back cannot feed a signal either.

`ProgressMeter.measure(t)` evaluates the caller's loss on every probe and
hands the losses to a forgetting.RetentionTracker, whose NET, relearning-
discounted progress is the number a usefulness term may use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..contracts import ContractError
from ..runtime.splits import (DEFAULT_HELDOUT_FRACTION, HELDOUT, SPLIT_SALT,
                              assign_split, episode_key)
from .ab import DataSplit
from .forgetting import RetentionTracker


class HeldOutAccessError(ContractError):
    """Held-out evaluation data was offered to a selection/progress signal."""


@dataclass(frozen=True)
class Probe:
    probe_id: str
    scope: str
    episode_id: str
    payload: Any = None


def guard_dev_only(scope: str, episode_ids, split: Optional[DataSplit] = None,
                   heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
                   salt: str = SPLIT_SALT) -> None:
    """Raise HeldOutAccessError if any episode is held out."""
    for e in episode_ids:
        k = episode_key(scope, e)
        if split is not None:
            if k.scope != split.scope:
                raise ContractError(f"probe scope {k.scope!r} is not the split's "
                                    f"scope {split.scope!r}")
            if k.episode_id in split.heldout:
                raise HeldOutAccessError(
                    f"episode {k.scope}/{k.episode_id} is HELD OUT: final "
                    f"evaluation data may never compute a selection signal")
            if k.episode_id not in split.dev:
                raise ContractError(f"episode {k.episode_id!r} is on neither side "
                                    f"of split {split.split_id!r}; unknown is not dev")
        elif assign_split(k.scope, k.episode_id, heldout_fraction, salt) == HELDOUT:
            raise HeldOutAccessError(
                f"episode {k.scope}/{k.episode_id} hashes to the HELD-OUT side "
                f"(fraction {heldout_fraction}, salt {salt!r})")


class DevProbeSet:
    def __init__(self, split: Optional[DataSplit] = None,
                 heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
                 salt: str = SPLIT_SALT, max_probes: int = 256):
        if split is not None and not isinstance(split, DataSplit):
            raise ContractError("split must be an experiments.ab.DataSplit")
        self.split, self.heldout_fraction, self.salt = split, float(heldout_fraction), salt
        self.max_probes = int(max_probes)
        self._probes: Dict[str, Probe] = {}

    def add(self, probe_id: str, scope: str, episode_id, payload: Any = None) -> Probe:
        if not isinstance(probe_id, str) or not probe_id:
            raise ContractError("probe_id must be a non-empty string")
        if probe_id in self._probes:
            raise ContractError(f"duplicate probe {probe_id!r}")
        if len(self._probes) >= self.max_probes:
            raise ContractError(f"probe set is bounded at {self.max_probes}")
        guard_dev_only(scope, [episode_id], self.split, self.heldout_fraction, self.salt)
        p = Probe(probe_id, scope, str(episode_id), payload)
        self._probes[probe_id] = p
        return p

    def verify(self) -> None:
        for p in self._probes.values():
            guard_dev_only(p.scope, [p.episode_id], self.split,
                           self.heldout_fraction, self.salt)

    def probes(self) -> List[Probe]:
        return list(self._probes.values())

    def __len__(self):
        return len(self._probes)


class ProgressMeter:
    def __init__(self, probes: DevProbeSet, evaluate: Callable[[Probe], float],
                 tracker: RetentionTracker):
        if not isinstance(probes, DevProbeSet):
            raise ContractError("progress is measured on a DevProbeSet only")
        if not isinstance(tracker, RetentionTracker):
            raise ContractError("a ProgressMeter needs a RetentionTracker")
        self.probes, self.evaluate, self.tracker = probes, evaluate, tracker

    def measure(self, t: int) -> Dict[str, Any]:
        self.probes.verify()
        if not len(self.probes):
            raise ContractError("empty probe set: nothing to measure progress on")
        losses = {}
        for p in self.probes.probes():
            v = float(self.evaluate(p))
            if not math.isfinite(v):
                raise ContractError(f"probe {p.probe_id} loss is not finite")
            losses[p.probe_id] = v
        out = self.tracker.observe(t, losses)
        out["mean_loss"] = sum(losses.values()) / len(losses)
        return out
