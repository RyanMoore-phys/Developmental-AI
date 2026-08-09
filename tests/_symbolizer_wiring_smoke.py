"""Symbolizer WIRING smoke — drives the REAL parallel loop on Crafter.

_symbolizer_smoke.py tests the subsystem in isolation. This one tests the
integration into developmental_loop.py, which is where wiring bugs live:

  1. The loop constructs the symbolizer from the config block.
  2. maybe_label/collect actually fire in the waking parallel path, and the
     head trains from the agent's own latent (labels > 0, head_steps > 0).
  3. Perceptual facts reach the KNOWLEDGE GRAPH as real SymbolicFacts.
  4. A ground-truth break event (mine_* counter increase) mints the CAUSAL
     fact ("attack_held","breaks",X) at confidence 1.0.
  5. The break diff is EDGE-triggered: an unchanged counter must not re-mint
     the same fact every step (that would flood the KG).
  6. _prev_mine clears per episode — MineRL's mine_block stat restarts at 0
     in a fresh world, so a stale high-water mark would swallow every break.
  7. _log_progress renders the grounding line without throwing.
  8. Dream episodes do NOT query the VLM (waking-only channel).
  9. Facts carry a real timestep, not the default 0.
 10. A predicate whose reliability collapses has its already-minted triple
     RETRACTED from the KG by the loop — the other half of the gates.
"""
import copy
import json

import numpy as np
import yaml

SB = "/tmp/symbolizer_wiring_sb"
_STUB = json.dumps({"tree_visible": True, "grass_visible": True,
                    "stone_visible": False, "breakable_in_reach": True})


def _install_stub_vlm():
    """Make the symbolizer think Ollama is up and answer deterministically,
    exercising the real submit -> parse -> train path with no server."""
    import developmental_ai.llm.vlm_symbolizer as V
    V._check_ollama_server = lambda *a, **k: True
    V.VLMSymbolizer._ollama_scene_query = lambda self, png: _STUB


def _cfg():
    cfg = yaml.safe_load(open("configs/crafter.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({"stochastic_size": 8, "stochastic_classes": 8,
                               "deterministic_size": 64, "encoder_hidden": 64,
                               "batch_size": 4, "sequence_length": 8,
                               "train_iters": 1, "buffer_capacity": 4000})
    cfg["environment"]["max_episode_steps"] = 40
    cfg["parallel_envs"] = {"enabled": True, "num_envs": 2}
    cfg["skill_bank"]["storage_dir"] = SB
    cfg["loop"]["verbose"] = 0
    cfg["llm"]["enabled"] = False
    cfg["llm"]["vision"] = {"enabled": False}
    # the subsystem under test; interval=1 so it fires inside a short smoke
    cfg["symbolic_grounding"] = {
        "enabled": True, "interval": 1, "max_interval": 600,
        "hidden_dim": 32, "lr": 1e-2, "fact_threshold": 0.7,
        "min_labels": 2, "reliability_floor": 0.35, "anneal_agreement": 0.9,
        # decoupled from the vision scaffold (which is OFF here) — this is
        # the contract that causal grounding survives without the magnet
        "chop_actions": [0, 1, 2, 3, 4],
    }
    return cfg


def main():
    _install_stub_vlm()
    from developmental_ai.core.developmental_loop import (
        DevelopmentalAI, _to_fact)

    # ---- 1. construction from config -------------------------------------
    ai = DevelopmentalAI(_cfg())
    assert ai.symbolizer is not None, "loop did not construct the symbolizer"
    assert ai.symbolizer._available, "stub VLM not wired"
    lat_dim = ai.world_model.rssm.latent_dim
    assert ai.symbolizer.head.net[0].in_features == lat_dim, (
        "head latent_dim mismatch — would explode on the first real latent")
    print(f"  1. constructed from config (latent_dim={lat_dim}) ok")

    # ---- inject fake mine_* achievements into env 0's info ----------------
    # Crafter reports collect_*, not mine_*; forge a MineRL-shaped counter so
    # the ground-truth causal path is exercised end to end.
    ai._ensure_parallel_envs()
    env0 = ai._parallel_envs[0]
    counter = {"n": 0}
    _raw_step = env0.step

    def _step(a):
        o, r, te, tr, info = _raw_step(a)
        info = dict(info) if isinstance(info, dict) else {}
        ach = dict(info.get("achievements", {}) or {})
        counter["n"] += 1
        # counter goes 0,0,0,...,1,1,1,...,2 -> few EDGES over many steps
        ach["mine_oak_log"] = counter["n"] // 8
        info["achievements"] = ach
        return o, r, te, tr, info
    env0.step = _step

    # count ground-truth event firings (total_facts mixes in the perceptual
    # channel, so it cannot distinguish edge- from level-triggering)
    events = {"n": 0}
    _raw_observe = ai.symbolizer.observe_event

    def _observe(kind, detail, **kw):
        events["n"] += 1
        return _raw_observe(kind, detail, **kw)
    ai.symbolizer.observe_event = _observe

    # ---- 2/3/4/5. run a waking episode ------------------------------------
    kg_before = len(ai.knowledge_graph.query())
    ai._run_episode_parallel(use_dream_actor=False)

    sym = ai.symbolizer
    assert sym.total_labels > 0, "VLM never labelled during the waking path"
    assert sym.total_head_steps > 0, "head never trained from the latent"
    print(f"  2. label channel live: labels={sym.total_labels}, "
          f"head_steps={sym.total_head_steps}")

    facts = ai.knowledge_graph.query()
    assert len(facts) > kg_before, "no facts reached the knowledge graph"
    srcs = {f.source for f in facts}
    assert "vlm_grounded" in srcs, f"no perceptual facts in KG (sources={srcs})"
    print(f"  3. perceptual facts in KG: +{len(facts) - kg_before} "
          f"(sources={sorted(srcs)})")

    causal = [f for f in facts
              if f.subject == "attack_held" and f.relation == "breaks"]
    assert causal, "ground-truth break event minted no causal fact"
    assert all(f.confidence == 1.0 for f in causal), "causal facts must be 1.0"
    assert any(f.obj == "oak_log" for f in causal), (
        f"wrong object: {[f.obj for f in causal]}")
    print(f"  4. causal fact from ground truth: "
          f"('attack_held','breaks','oak_log') @ conf 1.0")

    # 5. EDGE-triggered: the counter rises once every 8 steps, so
    #    observe_event must fire once per RISE, not once per step.
    edges = counter["n"] // 8
    assert events["n"] == edges, (
        f"break diff is level-triggered, not edge-triggered: "
        f"observe_event fired {events['n']}x for {edges} real breaks "
        f"across {counter['n']} steps")
    print(f"  5. edge-triggered: {events['n']} event firings for {edges} "
          f"counter rises over {counter['n']} steps")

    # ---- 6. per-episode reset of the high-water marks ---------------------
    ai._prev_mine["oak_log"] = 999
    ai._run_episode_parallel(use_dream_actor=False)
    assert ai._prev_mine.get("oak_log", 999) < 999, (
        "_prev_mine not cleared at episode start — a fresh world's counters "
        "restart at 0 and every break would be swallowed")
    print("  6. _prev_mine cleared per episode ok")

    # ---- 7. progress line renders ----------------------------------------
    ai._log_progress()
    print("  7. _log_progress renders the grounding line ok")

    # ---- 8. dream episodes must not query the VLM ------------------------
    before = sym.total_labels
    ai._run_episode_parallel(use_dream_actor=True)
    assert sym.total_labels == before, (
        f"VLM queried during a DREAM episode ({before}->{sym.total_labels}): "
        "grounding must be a waking-only channel")
    print("  8. dream path does not query the VLM ok")

    # ---- 9. facts carry a REAL timestep, not 0 ---------------------------
    # _to_fact used to omit timestamp, so every symbolizer fact was stamped 0
    # and the max(timestamp) update inside add_fact was a permanent no-op.
    perceptual = [f for f in ai.knowledge_graph.query()
                  if f.source == "vlm_grounded"]
    assert perceptual, "no perceptual facts to check timestamps on"
    assert any(f.timestamp > 0 for f in perceptual), (
        "symbolizer facts all stamped timestep 0 — timestamps unusable for "
        "ordering, decay or staleness")
    print(f"  9. facts carry a real timestep "
          f"(max={max(f.timestamp for f in perceptual)})")

    # ---- 10. the retraction channel is wired into the loop ---------------
    assert hasattr(ai, "_symbolizer_retractions"), "retraction counter missing"
    _t = ("object", "is", "breakable")
    ai.knowledge_graph.add_fact(_to_fact((*_t, 0.9), "vlm_grounded", 5))
    # Stop injecting breaks first: observe_event would EMA reliability back
    # up mid-episode, re-arm the predicate and let perceptual_facts re-assert
    # the very triple we are checking got retracted.
    env0.step = _raw_step
    for p_ in sym.reliability:
        sym.reliability[p_] = 0.0          # world discredits everything
    sym._retracted.clear()
    before = ai._symbolizer_retractions
    ai._run_episode_parallel(use_dream_actor=False)
    assert ai._symbolizer_retractions > before, (
        "reliability collapsed but the loop retracted nothing")
    assert _t not in ai.knowledge_graph.facts, (
        "discredited fact still resident in the KG after an episode")
    print(f"  10. loop retracted {ai._symbolizer_retractions - before} "
          f"discredited fact(s)")

    ai.shutdown() if hasattr(ai, "shutdown") else ai.symbolizer.close()
    print("[symbolizer-wiring] ALL PASS")


if __name__ == "__main__":
    main()
