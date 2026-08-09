"""VLM-symbolizer smoke (Mac-ok, no Ollama/MineRL needed).

Contracts for the "knows how -> knows why" subsystem:
  1. VLM labels train the GroundedSymbolHead, and the head LEARNS to predict
     predicates from the latent alone (the borrowed->owned conversion).
  2. Partial labels (VLM omits a key) never teach a false negative (masking).
  3. The VLM steps back: as head agreement rises, the query interval widens.
  4. Ground-truth events mint CAUSAL facts (attack_held breaks X) at
     confidence 1.0 — never taken on the VLM's word.
  5. Self-testing abstractions: a predicate the world keeps contradicting
     loses reliability and STOPS emitting perceptual facts.
  6. Perceptual facts are gated by head confidence AND earned reliability.
"""
import json

import numpy as np
import torch


def _stub(labels):
    """Returns a query_fn that always answers with `labels`."""
    return lambda png: json.dumps(labels)


def test_head_learns_from_vlm():
    from developmental_ai.llm.vlm_symbolizer import (
        VLMSymbolizer, PREDICATES)
    torch.manual_seed(0)
    LAT = 32
    sym = VLMSymbolizer(latent_dim=LAT, hidden_dim=64, lr=5e-3,
                        query_fn=_stub({"tree_visible": True,
                                        "grass_visible": True,
                                        "stone_visible": False}))
    # a FIXED latent should become predictable: train it repeatedly
    lat = torch.randn(1, LAT)
    labels = {"tree_visible": True, "grass_visible": True,
              "stone_visible": False}
    for _ in range(150):
        sym._train_head(lat, labels)
    p = sym.head.predict(lat).squeeze(0)
    i_tree = PREDICATES.index("tree_visible")
    i_stone = PREDICATES.index("stone_visible")
    assert p[i_tree] > 0.8, f"head didn't learn tree_visible: {p[i_tree]:.2f}"
    assert p[i_stone] < 0.2, f"head didn't learn NOT stone: {p[i_stone]:.2f}"
    print(f"  head learns from VLM labels ok "
          f"(tree={p[i_tree]:.2f}, stone={p[i_stone]:.2f})")


def test_partial_labels_masked():
    """A predicate the VLM did NOT report must not be ACTIVELY TAUGHT as
    false. (It can still drift via the shared trunk — inherent to a
    multi-label head — but must drift far less than an explicit negative.)"""
    from developmental_ai.llm.vlm_symbolizer import (
        VLMSymbolizer, PREDICATES)
    i_water = PREDICATES.index("water_visible")
    LAT = 32

    def run(explicit_negative: bool) -> float:
        torch.manual_seed(0)
        sym = VLMSymbolizer(latent_dim=LAT, hidden_dim=64, lr=5e-3,
                            query_fn=_stub({}))
        torch.manual_seed(1)
        lat = torch.randn(1, LAT)
        labels = {"tree_visible": True}
        if explicit_negative:                 # VLM explicitly says "no water"
            labels["water_visible"] = False
        for _ in range(100):
            sym._train_head(lat, labels)
        return float(sym.head.predict(lat).squeeze(0)[i_water])

    masked = run(False)      # water unreported -> masked out of the loss
    taught = run(True)       # water reported False -> actively taught
    assert taught < masked, (
        f"mask had no effect: masked={masked:.3f} taught={taught:.3f}")
    assert taught < 0.1, f"explicit negative should drive ~0, got {taught:.3f}"
    print(f"  masking ok: unreported={masked:.3f} vs explicitly-false="
          f"{taught:.3f} (mask spares the direct negative)")


def test_vlm_steps_back():
    """Once the head agrees with the VLM, queries get rarer (annealing)."""
    from developmental_ai.llm.vlm_symbolizer import VLMSymbolizer
    torch.manual_seed(0)
    sym = VLMSymbolizer(latent_dim=16, hidden_dim=32, lr=1e-2,
                        interval=60, max_interval=600,
                        anneal_agreement=0.9, query_fn=_stub({}))
    start = sym.interval
    lat = torch.randn(1, 16)
    labels = {"tree_visible": True, "grass_visible": False}
    for _ in range(400):                      # long enough to master + anneal
        sym._train_head(lat, labels)
    assert sym.interval > start, (
        f"VLM never stepped back: interval {start}->{sym.interval}")
    print(f"  annealing ok: VLM interval {start} -> {sym.interval} "
          f"as head learned")


def test_causal_facts_from_ground_truth():
    from developmental_ai.llm.vlm_symbolizer import VLMSymbolizer
    sym = VLMSymbolizer(latent_dim=16, hidden_dim=32, query_fn=_stub({}))
    facts = sym.observe_event("block_break", "oak_log", was_chopping=True)
    triples = {(s, r, o) for s, r, o, _c in facts}
    assert ("attack_held", "breaks", "oak_log") in triples, triples
    assert ("oak_log", "is", "breakable") in triples, triples
    assert all(c == 1.0 for *_x, c in facts), "ground-truth facts must be 1.0"
    # a non-chop break mints the property but not the causal claim
    facts2 = sym.observe_event("block_break", "dirt", was_chopping=False)
    t2 = {(s, r, o) for s, r, o, _c in facts2}
    assert ("attack_held", "breaks", "dirt") not in t2
    print("  causal facts ok: ground-truth only, confidence 1.0")


def test_self_testing_abstractions():
    """A predicate the world repeatedly contradicts must lose reliability and
    stop emitting facts — abstractions that test themselves."""
    from developmental_ai.llm.vlm_symbolizer import (
        VLMSymbolizer, PREDICATES)
    torch.manual_seed(0)
    sym = VLMSymbolizer(latent_dim=16, hidden_dim=32, lr=1e-2,
                        reliability_floor=0.35, fact_threshold=0.5,
                        query_fn=_stub({}))
    lat = torch.randn(1, 16)
    # make the head confidently assert breakable_in_reach
    for _ in range(200):
        sym._train_head(lat, {"breakable_in_reach": True})
    i = PREDICATES.index("breakable_in_reach")
    assert float(sym.head.predict(lat).squeeze(0)[i]) > 0.8
    # ...but the VLM's last claim keeps being WRONG when breaks happen
    sym.last_labels = {"breakable_in_reach": False}
    rel_before = sym.reliability["breakable_in_reach"]
    for _ in range(40):
        sym.observe_event("block_break", "stone", was_chopping=True)
    rel_after = sym.reliability["breakable_in_reach"]
    assert rel_after < rel_before, f"reliability didn't fall: {rel_after}"
    assert rel_after < sym.reliability_floor, (
        f"contradicted predicate still above floor: {rel_after:.2f}")
    facts = sym.perceptual_facts(lat)
    emitted = {o for _s, _r, o, _c in facts}
    assert "breakable" not in emitted, (
        "a discredited predicate is still emitting facts")
    print(f"  self-testing ok: reliability {rel_before:.2f}->{rel_after:.2f}, "
          f"predicate muted")


def test_perceptual_facts_gated():
    from developmental_ai.llm.vlm_symbolizer import (
        VLMSymbolizer, PREDICATES)
    torch.manual_seed(0)
    sym = VLMSymbolizer(latent_dim=16, hidden_dim=32, lr=1e-2,
                        fact_threshold=0.7, query_fn=_stub({}))
    lat = torch.randn(1, 16)
    for _ in range(200):
        sym._train_head(lat, {"tree_visible": True, "water_visible": False})
    facts = sym.perceptual_facts(lat)
    objs = {o for _s, _r, o, _c in facts}
    assert "tree" in objs, f"confident predicate not emitted: {objs}"
    assert "water" not in objs, f"low-prob predicate emitted: {objs}"
    assert all(0.0 <= c <= 1.0 for *_x, c in facts)
    print(f"  fact gating ok: emitted {sorted(objs)}")


def test_no_evidence_no_assertion():
    """A predicate the VLM has never labelled must NOT emit facts, even if
    shared-trunk drift pushes its probability over the threshold. Otherwise
    an untrained head floods the knowledge graph with fabricated facts."""
    from developmental_ai.llm.vlm_symbolizer import VLMSymbolizer
    torch.manual_seed(0)
    sym = VLMSymbolizer(latent_dim=16, hidden_dim=32, lr=1e-2,
                        fact_threshold=0.7, min_labels=5, query_fn=_stub({}))
    lat = torch.randn(1, 16)
    # only ever label tree_visible
    for _ in range(200):
        sym._train_head(lat, {"tree_visible": True})
    facts = sym.perceptual_facts(lat)
    objs = {o for _s, _r, o, _c in facts}
    assert objs == {"tree"}, (
        f"unlabelled predicates fabricated facts: {sorted(objs)}")
    assert sym.label_counts["animal_visible"] == 0
    print(f"  evidence gate ok: only labelled predicates assert {sorted(objs)}")


def test_discredited_predicate_is_retracted():
    """The other half of self-testing abstractions: a predicate the world
    discredits must not merely STOP EMITTING — its already-minted triple must
    LEAVE the knowledge graph, or it feeds the graph embedding for the rest
    of the run."""
    from developmental_ai.llm.vlm_symbolizer import VLMSymbolizer
    from developmental_ai.knowledge_graph.knowledge_graph import (
        InMemoryKnowledgeGraph, SymbolicFact)

    def _fact(triple, ts, source="vlm_grounded"):
        s_, r_, o_, c_ = triple
        return SymbolicFact(subject=s_, relation=r_, obj=o_,
                            confidence=float(c_), source=source, timestamp=ts)

    torch.manual_seed(0)
    kg = InMemoryKnowledgeGraph()
    sym = VLMSymbolizer(latent_dim=16, hidden_dim=32, lr=1e-2,
                        reliability_floor=0.35, fact_threshold=0.5,
                        min_labels=1, query_fn=_stub({}))
    lat = torch.randn(1, 16)
    for _ in range(200):
        sym._train_head(lat, {"breakable_in_reach": True})

    T = ("object", "is", "breakable")
    for f in sym.perceptual_facts(lat):
        kg.add_fact(_fact(f, 7), refresh=True)
    assert T in kg.facts, "perceptual fact never reached the KG"
    assert kg.facts[T].timestamp == 7, (
        f"fact stamped {kg.facts[T].timestamp}, not the real timestep")

    # confidence must be REVISABLE, not a high-water ratchet
    kg.add_fact(_fact((*T, 0.01), 8), refresh=True)
    assert kg.facts[T].confidence < 0.1, (
        f"confidence ratcheted to a high-water mark: {kg.facts[T].confidence}")

    # now let the world discredit it
    sym.last_labels = {"breakable_in_reach": False}
    for _ in range(40):
        sym.observe_event("block_break", "stone", was_chopping=True)
    stale = sym.stale_facts()
    assert T in stale, f"discredited predicate not handed back: {stale}"
    assert sym.stale_facts() == [], "retraction is not edge-triggered"

    assert kg.remove_fact(*T, source="vlm_grounded") is True
    assert T not in kg.facts, "fact still resident after retraction"
    assert T not in kg.get_triples_for_embedding(), (
        "retracted fact still feeds the graph embedding")
    print(f"  retraction ok: {T} left the KG and the embedding input")


def test_retraction_is_source_scoped():
    """One subsystem must never delete a triple another one asserted."""
    from developmental_ai.knowledge_graph.knowledge_graph import (
        InMemoryKnowledgeGraph, SymbolicFact)
    kg = InMemoryKnowledgeGraph()
    kg.add_fact(SymbolicFact("object", "is", "breakable", 1.0,
                             source="grounded_event"))
    assert kg.remove_fact("object", "is", "breakable",
                          source="vlm_grounded") is False, (
        "source scoping failed — deleted another subsystem's fact")
    assert ("object", "is", "breakable") in kg.facts
    print("  source scoping ok: foreign facts survive retraction")


def test_teacher_returns_when_head_degrades():
    """Annealing used to be a one-way ratchet: a head that degraded could
    never recall its teacher."""
    from developmental_ai.llm.vlm_symbolizer import VLMSymbolizer
    torch.manual_seed(0)
    sym = VLMSymbolizer(latent_dim=16, hidden_dim=32, lr=1e-2,
                        interval=60, max_interval=600,
                        anneal_agreement=0.9, query_fn=_stub({}))
    lat = torch.randn(1, 16)
    for _ in range(400):                       # master it -> teacher retires
        sym._train_head(lat, {"tree_visible": True, "grass_visible": False})
    widened = sym.interval
    assert widened > 60, "never annealed in the first place"
    # The world becomes genuinely UNPREDICTABLE from this latent (the real
    # degradation case: distribution shift the head cannot fit). Flipping to
    # a second fixed mapping would NOT test this — the head just relearns it
    # in a few steps and agreement recovers before the window fills.
    import random as _rnd
    _rnd.seed(0)
    for _ in range(400):
        sym._train_head(lat, {"tree_visible": bool(_rnd.getrandbits(1)),
                              "grass_visible": bool(_rnd.getrandbits(1))})
    assert sym.interval < widened, (
        f"teacher never came back: interval stuck at {sym.interval} "
        f"after agreement collapsed")
    assert sym.interval >= sym.base_interval, "de-annealed below base"
    print(f"  de-anneal ok: interval {widened} -> {sym.interval} when the "
          f"head degraded")


def test_pending_latent_survives_inflight_polls():
    """collect() runs EVERY step while a ~1s query is in flight. It must not
    drop the latent paired with the frame being labelled, or the head trains
    on nothing for the whole run."""
    from developmental_ai.llm.vlm_symbolizer import VLMSymbolizer
    import threading
    gate = threading.Event()

    def _slow(png):
        gate.wait(timeout=5)
        return json.dumps({"tree_visible": True})

    sym = VLMSymbolizer(latent_dim=16, hidden_dim=32, query_fn=_slow)
    lat = torch.randn(1, 16)
    sym.maybe_label(np.zeros((8, 8, 3), dtype=np.uint8), lat, 0)
    for _ in range(50):                        # poll while still in flight
        assert sym.collect() is None
    assert sym._pending_latent is not None, (
        "in-flight polling destroyed the pending latent")
    gate.set()
    import time as _t
    for _ in range(200):                       # let the worker actually finish
        if sym.collect() is not None:
            break
        _t.sleep(0.02)
    assert sym.total_head_steps > 0, "head never trained from the paired latent"
    sym.close()
    print("  latent pairing survives in-flight polling ok")


if __name__ == "__main__":
    for fn in (test_head_learns_from_vlm, test_partial_labels_masked,
               test_vlm_steps_back, test_causal_facts_from_ground_truth,
               test_self_testing_abstractions, test_perceptual_facts_gated,
               test_no_evidence_no_assertion,
               test_discredited_predicate_is_retracted,
               test_retraction_is_source_scoped,
               test_teacher_returns_when_head_degrades,
               test_pending_latent_survives_inflight_polls):
        print(f"[symbolizer-smoke] {fn.__name__}")
        fn()
    print("[symbolizer-smoke] ALL PASS")
