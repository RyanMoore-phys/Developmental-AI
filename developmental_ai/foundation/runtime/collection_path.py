"""Which collection body a config selects, decided BEFORE the env is built.

WHY THIS EXISTS (plan Stage 3, CLAUDE.md §4.2)
    `developmental_loop.py` has three bodies that step the environment:

        _collect_segment         lifelong stream (what SkyBot runs)
        _run_episode_parallel    episodic, N envs in lockstep
        _run_episode             episodic, ONE env — the pre-Wave-1 agent

    `_run_episode` has none of Waves 1-2 or Phases 1-7 (no info["sensors"],
    no _augment_proprio, no flow senses, no spatial step). It was reached
    SILENTLY: `parallel_envs.num_envs <= 1` quietly set
    `_use_parallel_envs = False`, so a "smaller" SkyBot run was a different,
    older agent with nothing in the log saying so. Plan Stage 3's completion
    gate is exactly this: "A nominally smaller run must not silently select a
    different agent."

THE RULE (least disruptive correct rule; see docs/foundation/BASELINE_AUDIT.md)
    1. lifelong.enabled            -> needs the parallel path. No single-env
                                      lifelong body exists, so there is no
                                      escape flag: raise (as the loop always
                                      did, only now before the env boots).
    2. parallel effective (enabled and num_envs >= 2) -> as configured.
    3. otherwise the single-env legacy body is selected, and it is REFUSED if
         a. the config ASKED for parallel (enabled: true) but num_envs <= 1
            — the silent downgrade itself; or
         b. the environment is Minecraft (MineRL/MineDojo/"minecraft"), where
            the single-env body is known not to carry the live wiring;
       unless `parallel_envs.allow_legacy_single_env: true`.
    Non-Minecraft configs that never asked for parallel (crafter, cartpole,
    minigrid smokes: ~20 tests) keep the historical single-env default, which
    is the path they were written and validated against.

THE ESCAPE PATH (CLAUDE.md §4.1 — a guard needs a reachable re-opener)
    `parallel_envs.allow_legacy_single_env: true` re-opens rule 3, and the
    error text names it. It is a config key a human sets deliberately; the
    selected path is also returned so it can be logged and put in a manifest.

Import-light by design: the loop imports this at construction time.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple

LIFELONG_SEGMENT = "lifelong_segment"      # _collect_segment
PARALLEL_EPISODE = "parallel_episode"      # _run_episode_parallel
SINGLE_ENV_LEGACY = "single_env_legacy"    # _run_episode

PATHS = (LIFELONG_SEGMENT, PARALLEL_EPISODE, SINGLE_ENV_LEGACY)
ESCAPE_KEY = "parallel_envs.allow_legacy_single_env"

_MINECRAFT_MARKERS = ("minerl", "minedojo", "minecraft")


class UnsupportedCollectionPath(ValueError):
    """A config selects a collection body that is not supported for it.

    Subclasses ValueError so the loop's historical contract (lifelong without
    parallel raises ValueError mentioning 'parallel') still holds.
    """


class CollectionPathDecision(NamedTuple):
    path: str
    reason: str
    num_envs: int


def _get(cfg: Any, key: str) -> Dict:
    try:
        v = cfg.get(key, None)
    except AttributeError:
        return {}
    return v if v is not None else {}


def is_minecraft_env(env_name: Any) -> bool:
    n = str(env_name or "").lower()
    return any(m in n for m in _MINECRAFT_MARKERS)


def select_collection_path(config: Any) -> CollectionPathDecision:
    """Return the collection body `config` selects, or raise explicitly.

    Mirrors the loop's own reading of the config exactly:
    `parallel_envs.enabled` (default False), `num_envs` (default 1, floored at
    1), `lifelong.enabled` (default False), `environment.name` (default
    CartPole-v1).
    """
    par = _get(config, "parallel_envs")
    life = _get(config, "lifelong")
    env = _get(config, "environment")
    requested = bool(par.get("enabled", False))
    n = max(1, int(par.get("num_envs", 1) or 1))
    parallel = requested and n > 1
    lifelong = bool(life.get("enabled", False))
    allow = bool(par.get("allow_legacy_single_env", False))
    env_name = env.get("name", "CartPole-v1")

    if lifelong:
        if not parallel:
            raise UnsupportedCollectionPath(
                "lifelong.enabled requires parallel_envs.enabled with "
                f"num_envs>1 (got enabled={requested}, num_envs={n}); the "
                "single-env path is not covered by the continuous loop and "
                "has no lifelong body, so there is no override — set "
                "parallel_envs.enabled: true and num_envs: 2 (CLAUDE.md "
                "§4.2).")
        return CollectionPathDecision(
            LIFELONG_SEGMENT, "lifelong.enabled with parallel envs", n)
    if parallel:
        return CollectionPathDecision(
            PARALLEL_EPISODE, f"parallel_envs.enabled, num_envs={n}", n)

    if allow:
        return CollectionPathDecision(
            SINGLE_ENV_LEGACY,
            f"{ESCAPE_KEY}: true (explicit opt-in to the pre-Wave-1 body)", n)
    if requested:
        why = (f"parallel_envs.enabled is true but num_envs={n}, which "
               "silently downgrades to the single-env body")
    elif is_minecraft_env(env_name):
        why = (f"environment {env_name!r} is Minecraft and the single-env "
               "body lacks the live wiring")
    else:
        return CollectionPathDecision(
            SINGLE_ENV_LEGACY,
            "non-Minecraft config without parallel_envs (historical default)",
            n)
    raise UnsupportedCollectionPath(
        f"refusing the single-env collection path: {why}. _run_episode is "
        "the pre-Wave-1 agent (no info['sensors'], no _augment_proprio, no "
        "flow senses, no spatial step — CLAUDE.md §4.2), so this would be a "
        "DIFFERENT agent, not a smaller one. Fix: set parallel_envs.enabled: "
        "true with num_envs >= 2. Escape path: set "
        f"{ESCAPE_KEY}: true to run the legacy single-env body deliberately.")
