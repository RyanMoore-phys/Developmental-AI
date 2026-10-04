"""One conformance suite across many adapters (plan Stage 12.1, "the same
learning core through multiple adapters").

run_conformance_suite({name: factory(seed) -> adapter}, seeds, n_steps)
runs adapters.check_adapter_conformance on a FRESH instance per (adapter,
seed) and collects every problem rather than stopping at the first adapter
that fails. A wrapper is an adapter too, so every environment variation is
checked by the same clauses as the environment it wraps.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

from ..adapters import check_adapter_conformance
from ..contracts import AdapterConformanceError


def run_conformance_suite(factories: Mapping[str, Callable[[int], Any]],
                          seeds: Sequence[int] = (0, 1),
                          n_steps: int = 60) -> Dict[str, Dict[str, Any]]:
    if not factories:
        raise ValueError("no adapters given")
    out = {}
    for name, make in factories.items():
        problems, steps, episodes, kinds, channels = [], 0, 0, set(), set()
        for s in seeds:
            try:
                rep = check_adapter_conformance(make(s), n_steps=n_steps, seed=s,
                                                raise_on_failure=False)
            except AdapterConformanceError as e:     # spec-level failure
                problems += [f"seed {s}: {p}" for p in e.problems]
                continue
            except Exception as e:                   # construction / crash
                problems.append(f"seed {s}: {type(e).__name__}: {e}")
                continue
            problems += [f"seed {s}: {p}" for p in rep.problems]
            steps += rep.steps
            episodes += rep.episodes
            kinds.add(rep.action_kind)
            channels |= set(rep.channels_seen)
        out[name] = {"ok": not problems, "problems": problems, "steps": steps,
                     "episodes": episodes, "action_kinds": sorted(kinds),
                     "channels_seen": sorted(channels)}
    return out


def assert_suite_conforms(results: Mapping[str, Dict[str, Any]]) -> None:
    bad = {k: v["problems"] for k, v in results.items() if not v["ok"]}
    if bad:
        raise AdapterConformanceError(
            [f"{k}: {p}" for k, ps in bad.items() for p in ps])
