"""Collection-path smoke (2026-10-03): a smaller run is not a different agent.

THE FAILURE THIS ENCODES (CLAUDE.md §4.2, plan Stage 3)

    developmental_loop.py steps the environment in three bodies:
    _collect_segment (lifelong, live SkyBot), _run_episode_parallel
    (episodic, N envs) and _run_episode (one env). _run_episode is the
    pre-Wave-1 agent. It has no info["sensors"], no _augment_proprio, no flow
    senses and no spatial step. It used to be selected SILENTLY:
    `parallel_envs.num_envs <= 1` set `_use_parallel_envs = False`, so
    "SkyBot with one client" quietly ran an older agent with nothing in the
    log saying so. The plan's Stage 3 gate is "a nominally smaller run must
    not silently select a different agent".

THE RULE (foundation/runtime/collection_path.py)

    The single-env body is REFUSED when:
      * the config asked for parallel but got num_envs <= 1, or
      * the env is Minecraft.
    The escape is `parallel_envs.allow_legacy_single_env: true`, and the
    error text names it. A guard with no reachable re-opener is a latch
    (§4.1).
    Non-Minecraft configs that never asked for parallel keep the historical
    single-env default. About 20 crafter/minigrid/cartpole smokes depend on
    it, and that is the path they were validated against.

Contracts:
    A. SkyBot config with num_envs 1 raises in every variant:
       - as shipped (lifelong on): ValueError mentioning parallel;
       - lifelong off, parallel requested;
       - lifelong off, parallel off.
       The non-lifelong errors name the escape key.
    B. The escape flag re-opens the non-lifelong variants (and only those:
       lifelong has no single-env body, so no flag can select one).
    C. The parallel/lifelong paths are unaffected. SkyBot as shipped selects
       lifelong_segment; num_envs 2 without lifelong selects
       parallel_episode. Every config in configs/ selects exactly what the
       OLD rule selected, unless it is Minecraft-single-env. So no
       non-Minecraft config changed path.
    D. Source contract on the loop:
       - train()'s dispatch has exactly three branches (lifelong, parallel,
         else), and each body is called from exactly one place;
       - the decision is made before the first make_env() in __init__;
       - _use_parallel_envs is cross-checked against it.
    E. Integration (needs gymnasium, otherwise reported as SKIPPED):
       - DevelopmentalAI refuses before it builds an env;
       - with the flag it gets as far as building one.

Run: PYTHONPATH=. python tests/_collection_path_smoke.py
"""
import copy
import importlib.util
import os
import re
import sys

sys.path.insert(0, ".")

import yaml

from developmental_ai.foundation.runtime.collection_path import (
    ESCAPE_KEY, LIFELONG_SEGMENT, PARALLEL_EPISODE, SINGLE_ENV_LEGACY,
    UnsupportedCollectionPath, is_minecraft_env, select_collection_path)

LOOP = os.path.join("developmental_ai", "core", "developmental_loop.py")
SKY = yaml.safe_load(open("configs/minecraft_skybot.yaml"))


def _sky(**par):
    c = copy.deepcopy(SKY)
    life = par.pop("lifelong", None)
    if life is not None:
        c["lifelong"]["enabled"] = life
    c["parallel_envs"].update(par)
    return c


def _raises(cfg, *needles):
    try:
        d = select_collection_path(cfg)
    except ValueError as e:
        for n in needles:
            assert n in str(e), f"error text lacks {n!r}: {e}"
        return e
    raise AssertionError(f"not refused; selected {d.path}")


def test_skybot_single_env_refused():
    _raises(_sky(num_envs=1), "parallel")
    e1 = _raises(_sky(num_envs=1, lifelong=False), ESCAPE_KEY,
                 "pre-Wave-1")
    e2 = _raises(_sky(num_envs=1, enabled=False, lifelong=False), ESCAPE_KEY,
                 "Minecraft")
    assert isinstance(e1, UnsupportedCollectionPath)
    assert isinstance(e2, UnsupportedCollectionPath)
    print("  A. skybot num_envs 1 refused (lifelong / requested / plain), "
          "escape key named")


def test_flag_reopens():
    for c in (_sky(num_envs=1, lifelong=False),
              _sky(num_envs=1, enabled=False, lifelong=False)):
        c["parallel_envs"]["allow_legacy_single_env"] = True
        assert select_collection_path(c).path == SINGLE_ENV_LEGACY
    # Lifelong has no single-env body: no flag may select one.
    c = _sky(num_envs=1, allow_legacy_single_env=True)
    _raises(c, "parallel")
    print("  B. allow_legacy_single_env re-opens the episodic single-env "
          "body; lifelong stays refused")


def _old_rule(cfg):
    par = cfg.get("parallel_envs") or {}
    par_on = bool(par.get("enabled", False)) and max(
        1, int(par.get("num_envs", 1))) > 1
    if (cfg.get("lifelong") or {}).get("enabled", False):
        return LIFELONG_SEGMENT if par_on else "raise"
    return PARALLEL_EPISODE if par_on else SINGLE_ENV_LEGACY


def test_parallel_unaffected_and_no_other_config_moved():
    assert select_collection_path(SKY).path == LIFELONG_SEGMENT
    assert select_collection_path(
        _sky(lifelong=False)).path == PARALLEL_EPISODE
    moved = []
    n = 0
    for fn in sorted(os.listdir("configs")):
        if not fn.endswith(".yaml"):
            continue
        cfg = yaml.safe_load(open(os.path.join("configs", fn)))
        n += 1
        old = _old_rule(cfg)
        try:
            new = select_collection_path(cfg).path
        except ValueError:
            new = "raise"
        if new != old:
            env = (cfg.get("environment") or {}).get("name")
            assert is_minecraft_env(env) and old == SINGLE_ENV_LEGACY, (
                f"{fn}: path moved {old} -> {new} for a non-Minecraft "
                f"config")
            moved.append(fn)
    print(f"  C. skybot -> {LIFELONG_SEGMENT}; num_envs 2 -> "
          f"{PARALLEL_EPISODE}; {n} configs checked, moved: {moved or 'none'}")


def test_source_dispatch():
    src = open(LOOP).read()
    m = re.search(
        r"\n(\s+)if self\._lifelong:\n\s+episode_metrics = "
        r"self\._collect_segment\(\)\n\1elif self\._use_parallel_envs:\n"
        r"\s+episode_metrics = self\._run_episode_parallel\(\n[^\n]*\n"
        r"\1else:\n\s+episode_metrics = self\._run_episode\(\)\n", src)
    assert m, "train() dispatch no longer has the expected three branches"
    tail = src[m.end():m.end() + 400]
    assert not tail.lstrip().startswith(("elif", "else")), \
        "an extra dispatch branch was added"
    assert src.count("self._collect_segment()") == 1
    assert src.count("self._run_episode()") == 1
    assert src.count("self._run_episode_parallel(") == 1
    assert src.count("select_collection_path(self.config)") == 1
    init = src[src.index("def __init__(self, config"):]
    assert init.index("select_collection_path(self.config)") < \
        init.index("make_env("), "decision must precede the env build"
    assert "bool(self._use_parallel_envs) != (" in src, \
        "the _use_parallel_envs cross-check was removed"
    print("  D. dispatch = {lifelong, parallel, else}; each body called once;"
          " decision precedes make_env; cross-check present")


class _ReachedEnvBuild(Exception):
    pass


def test_integration():
    if importlib.util.find_spec("gymnasium") is None:
        print("  E. SKIPPED (gymnasium absent here; runs on CI / host) — "
              "A-D are the local evidence")
        return
    import developmental_ai.core.developmental_loop as dl

    def _boom(*a, **k):
        raise _ReachedEnvBuild()

    saved = dl.make_env
    dl.make_env = _boom
    try:
        c = _sky(num_envs=1, lifelong=False)
        c["llm"] = {"enabled": False}
        try:
            dl.DevelopmentalAI(config=copy.deepcopy(c))
            raise AssertionError("constructed a single-env SkyBot")
        except UnsupportedCollectionPath as e:
            assert ESCAPE_KEY in str(e)
        c["parallel_envs"]["allow_legacy_single_env"] = True
        try:
            dl.DevelopmentalAI(config=copy.deepcopy(c))
            raise AssertionError("make_env was never reached")
        except _ReachedEnvBuild:
            pass
    finally:
        dl.make_env = saved
    print("  E. DevelopmentalAI refuses before make_env; the flag reaches it")


if __name__ == "__main__":
    test_skybot_single_env_refused()
    test_flag_reopens()
    test_parallel_unaffected_and_no_other_config_moved()
    test_source_dispatch()
    test_integration()
    print("[collection-path] ALL PASS")
