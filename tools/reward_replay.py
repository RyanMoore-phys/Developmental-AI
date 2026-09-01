"""Offline counterfactual reward replay: score trajectories through the REAL
reward primitives without booting an environment.

WHY THIS EXISTS (two measured failures):

(a) COST OF THE LOOP. Every reward-design change used to cost a live restart
    plus hours of wall-clock before we learned what the agent would prefer
    under the new economy. This tool answers the same question in seconds:
    hand it candidate trajectories ("harvest the target", "grind the familiar
    thing", "stare at the sky") and a config, and it says which one the
    reward stack pays best.

(b) GREEN SMOKES, DEAD AGENT. Thirteen mechanism smokes were all green while
    the assembled agent stood on a pillar doing nothing — each mechanism was
    individually correct and the STACK still preferred the wrong behaviour.
    A behavioural claim ("chopping out-pays pillaring by 3x") can only be
    asserted by running whole trajectories through the assembled economy,
    which is what this module makes cheap. tests/_behavior_suite.py does
    exactly that on every run.

THE PRIMITIVES ARE IMPORTED, NEVER REIMPLEMENTED. A private copy of the
reward math would rot the day someone tunes the live code, and the suite
would then be asserting preferences of an economy that no longer exists —
the exact drift this tool was built to catch. The real functions are invoked
unbound on bare shim objects carrying only the attributes they read (the
same pattern the smoke tests use), so a change to the live economy is picked
up here on the next run with zero maintenance.

ADMITTED APPROXIMATION. The scores are NOT the live reward signal: the live
intrinsic is an ICM/novelty stack over learned latents, replay reduces it to
`novelty_weight * new_view` shaped by the real boring-view and habituation
factors; the live extrinsic includes shaping terms replay omits. That is
deliberate — the tool exists to compare TRAJECTORY PREFERENCES under a
config (is A paid more than B, and by how much), not to reproduce absolute
magnitudes. Any conclusion of the form "the total was 23.4" is outside its
warranty; "A out-pays B 10x" is inside it.

DEFENSIVE BY CONTRACT: this is monitoring/analysis code and must never be
able to take down anything that calls it. Every per-step computation is
caught and contained; failures come back as data in the result's "errors"
list (and a zeroed per_step row) instead of an exception.

Runnable standalone:  ./venv/bin/python tools/reward_replay.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Real primitives — see module docstring for why these are imported rather
# than copied. developmental_loop is a heavy import (torch etc.); that is
# accepted here exactly as it is in the existing smoke tests.
from developmental_ai.core.developmental_loop import DevelopmentalAI
from developmental_ai.environments.minerl_env import MineRLEnvAdapter

# Event kinds the running-count bookkeeper tracks, and the per-kind info
# keys _event_count_map / _habituation_factor expect. Kept as one table so
# the flat "kind:subtype" vocabulary here can never drift from the loop's.
_KIND_TO_INFO_KEY: Dict[str, str] = {
    "break": "breaks_by_type",
    "place": "places_by_type",
    "craft": "crafts_by_type",
    "pickup": "pickups_by_type",
}

_MISSING = object()   # sentinel: distinguishes "class attr absent" from None


class _DecayShim:
    """Bare object for the unbound MineRLEnvAdapter._break_decay call.

    Carries exactly the two attributes the method reads. `_breaks_by_type`
    is a LIVE reference to the replayer's running break-count map, so the
    decay always sees the same counts habituation does — one bookkeeper,
    no chance of the two multipliers disagreeing about familiarity.
    """

    def __init__(self, scale: float, breaks_by_type: Dict[str, int]) -> None:
        self._break_decay_scale = float(scale)
        self._breaks_by_type = breaks_by_type


class _LoopShim:
    """Bare object exposing the loop's intrinsic-side primitives unbound.

    The methods are the REAL ones off DevelopmentalAI; only the state they
    read is synthetic. `_event_count_map` must be re-wrapped in
    staticmethod(): accessing it on the class yields the plain function,
    and assigning a plain function into a class body would turn it back
    into an instance method that swallows `self` as the info argument.
    """

    _boring_view_factor = DevelopmentalAI._boring_view_factor
    _habituation_factor = DevelopmentalAI._habituation_factor
    _event_count_map = staticmethod(DevelopmentalAI._event_count_map)

    def __init__(self, habituation_scale: float) -> None:
        self._habituation_scale = float(habituation_scale)
        self._habit_prev: Dict[str, int] = {}
        # symbolizer/infra None = cold-start behaviour of the live loop:
        # min_labels falls back to 5, no learned category-boringness. The
        # safe direction, and the only honest one offline.
        self.symbolizer = None
        self.infra = None
        self._last_fovea_probs: Dict[str, float] = {}
        self._last_fovea_counts: Dict[str, int] = {}
        self._last_env_info: Dict[str, Any] = {}
        self._last_world_info: Dict[str, Any] = {}


def _normalise_events(step: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(kind, subtype) tuples from a step, tolerating JSON round-trips.

    Logged trajectories arrive with lists where synthetic ones have tuples;
    both must score identically or the tool would rank the same behaviour
    differently depending on where the trajectory came from.
    """
    out: List[Tuple[str, str]] = []
    for ev in (step.get("events") or ()):
        kind, sub = ev  # malformed entries raise; the caller contains it
        out.append((str(kind), str(sub)))
    return out


def score_trajectory(steps: "List[dict]", *,
                     habituation_scale: float = 50.0,
                     log_break_reward: float = 20.0,
                     break_decay_scale: float = 25.0,
                     novelty_weight: float = 0.15,
                     novelty_sky_discount: bool = True) -> dict:
    """Score one trajectory through the real reward primitives.

    Each step dict may contain (all optional):
      "events":       [("break", "<type>"), ("place", "<type>"), ...] —
                      what the agent caused this step.
      "counts":       {"break:<type>": n, "place:<type>": n, ...} lifetime
                      counts BEFORE this step; seeds/overrides the running
                      tally for those keys (typically set once, on the
                      first step that needs a familiarity level).
      "fovea":        {"<category>_visible": prob, ...} fovea-head probs.
      "fovea_counts": per-category label counts; defaults to 10 for every
                      category in "fovea" (past the trust gate) because a
                      synthetic trajectory asserts what IS in view, not how
                      long the head took to learn it.
      "pitch":        view tilt in degrees (drives the geometric clamp).
      "new_view":     True when this step's view is novel (the intrinsic
                      proxy's trigger).
      "achievements": raw lifetime counter dict for logged trajectories
                      that predate the typed event stream; synthesised from
                      events otherwise so _habituation_factor's legacy path
                      sees increments either way.

    Per step: extrinsic = max over break events of the real
    _break_reward * _break_decay (0.0 when nothing broke — mirroring the
    live adapter's max-not-sum aggregation); intrinsic proxy =
    novelty_weight * new_view * _boring_view_factor * _habituation_factor.
    Habituation multiplies only the intrinsic, as in the live loop — the
    extrinsic already carries its own familiarity decay, and stacking both
    would double-punish exactly the behaviours being compared. Familiarity
    counts use the value BEFORE the step (a first-ever event keeps full
    surprise) and accumulate across steps.

    Returns {"extrinsic", "intrinsic_proxy", "total", "per_step", "errors"}
    — an admitted approximation of the live stack (see module docstring):
    compare trajectory PREFERENCES under a config, never absolute
    magnitudes. Never raises for malformed steps; those surface as entries
    in "errors" with a zeroed per_step row.
    """
    loop = _LoopShim(habituation_scale)
    running: Dict[str, Dict[str, int]] = {k: {} for k in _KIND_TO_INFO_KEY}
    decay = _DecayShim(break_decay_scale, running["break"])
    per_step: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    ext_total = 0.0
    intr_total = 0.0

    # The classmethod consults a CLASS attribute (an instance attr would be
    # invisible to it — see the adapter's own comment), so the override is
    # installed for the duration of scoring and restored in finally: two
    # interleaved configs, or a live adapter constructed later in the same
    # process, must not inherit this call's economy.
    prev_lbr = getattr(MineRLEnvAdapter, "LOG_BREAK_REWARD", _MISSING)
    MineRLEnvAdapter.LOG_BREAK_REWARD = float(log_break_reward)
    try:
        for i, step in enumerate(steps):
            try:
                # -- seed lifetime counts (BEFORE-step semantics) ---------
                for key, val in (step.get("counts") or {}).items():
                    kind, _, sub = str(key).partition(":")
                    if kind in running and sub:
                        running[kind][sub] = int(val)

                events = _normalise_events(step)

                # -- extrinsic: real whitelist tier x real familiarity ----
                ext = 0.0
                for kind, sub in events:
                    if kind != "break":
                        continue
                    ext = max(ext,
                              MineRLEnvAdapter._break_reward(sub)
                              * MineRLEnvAdapter._break_decay(decay, sub))

                # -- perception state for the boring-view factor ----------
                pitch = step.get("pitch")
                loop._last_world_info = (
                    {"pitch": float(pitch)} if pitch is not None else {})
                fps = dict(step.get("fovea") or {})
                loop._last_fovea_probs = fps
                loop._last_fovea_counts = dict(
                    step.get("fovea_counts")
                    or {cat: 10 for cat in fps})

                # -- habituation info: event-stream form when the step has
                # events (the general path), legacy achievements otherwise.
                # An explicitly-provided achievements dict with no events
                # must NOT be shadowed by an empty events list — the event
                # path short-circuits on `is not None` and would return 1.0
                # without ever reading the increments a logged trajectory
                # carries.
                info: Dict[str, Any] = {
                    key: running[kind]
                    for kind, key in _KIND_TO_INFO_KEY.items()}
                if events or "achievements" not in step:
                    info["events"] = events
                ach = step.get("achievements")
                if ach is None:
                    ach = {"mine_" + sub: running["break"].get(sub, 0) + 1
                           for kind, sub in events if kind == "break"}
                info["achievements"] = ach

                hab = loop._habituation_factor(info)
                view = (loop._boring_view_factor()
                        if novelty_sky_discount else 1.0)
                intr = (novelty_weight
                        * (1.0 if step.get("new_view") else 0.0)
                        * view * hab)

                # -- commit: counts move AFTER scoring so this step saw the
                # world as it was when the behaviour happened -------------
                for kind, sub in events:
                    if kind in running:
                        running[kind][sub] = running[kind].get(sub, 0) + 1

                ext_total += ext
                intr_total += intr
                per_step.append({
                    "i": i, "extrinsic": ext, "intrinsic_proxy": intr,
                    "habituation": hab, "view_factor": view,
                    "events": events,
                })
            except Exception as exc:  # contain: errors are data, not crashes
                errors.append({"i": i, "error": repr(exc)})
                per_step.append({
                    "i": i, "extrinsic": 0.0, "intrinsic_proxy": 0.0,
                    "habituation": 1.0, "view_factor": 1.0,
                    "events": [], "error": repr(exc),
                })
    finally:
        if prev_lbr is _MISSING:
            try:
                del MineRLEnvAdapter.LOG_BREAK_REWARD
            except AttributeError:
                pass
        else:
            MineRLEnvAdapter.LOG_BREAK_REWARD = prev_lbr

    return {
        "extrinsic": ext_total,
        "intrinsic_proxy": intr_total,
        "total": ext_total + intr_total,
        "per_step": per_step,
        "errors": errors,
    }


# --------------------------------------------------------------------------
# Standalone demo: four behavioural archetypes, ~60 steps each. Subtype
# names are deliberately generic — the archetypes are universal (approach a
# paying target / grind a familiar freebie / stare at the sky / stack the
# same object), only the economy that prices them is domain-specific.
# --------------------------------------------------------------------------

def _demo_trajectories() -> "Dict[str, List[dict]]":
    """The four canonical failure/success shapes replay was built to rank."""
    approach: List[dict] = []
    for i in range(50):   # target grows in the fovea as the agent closes in
        approach.append({"fovea": {"tree_visible": 0.2 + 0.014 * i},
                         "pitch": 5.0, "new_view": (i % 2 == 0)})
    approach.append({"events": [("break", "target_log")],
                     "counts": {"break:target_log": 1},
                     "fovea": {"tree_visible": 0.9}, "pitch": 10.0,
                     "new_view": True})
    approach += [{"fovea": {"tree_visible": 0.6}, "pitch": 5.0,
                  "new_view": (i % 2 == 0)} for i in range(9)]

    grind = [dict({"events": [("break", "terrain_a")], "pitch": 70.0,
                   "new_view": True},
                  **({"counts": {"break:terrain_a": 400}} if i == 0 else {}))
             for i in range(60)]

    stare = [{"pitch": -85.0, "new_view": True,
              "fovea": {"sky_visible": 0.95}} for _ in range(60)]

    tower = [dict({"events": [("place", "terrain_a")], "pitch": -70.0,
                   "new_view": True},
                  **({"counts": {"place:terrain_a": 60}} if i == 0 else {}))
             for i in range(60)]

    return {"approach_target": approach, "grind_familiar": grind,
            "sky_stare": stare, "tower_build": tower}


if __name__ == "__main__":
    print("reward_replay: scoring demo trajectories under the default "
          "economy\n")
    header = f"{'trajectory':<18} {'extrinsic':>10} {'intrinsic':>10} " \
             f"{'total':>10} {'errors':>7}"
    print(header)
    print("-" * len(header))
    for name, traj in _demo_trajectories().items():
        r = score_trajectory(traj)
        print(f"{name:<18} {r['extrinsic']:>10.3f} "
              f"{r['intrinsic_proxy']:>10.3f} {r['total']:>10.3f} "
              f"{len(r['errors']):>7d}")
    print("\n(compare rows, not magnitudes — see module docstring)")
