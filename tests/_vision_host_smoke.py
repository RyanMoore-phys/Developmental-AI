"""Curiosity-magnet grounding-input smoke (Mac- or pod-runnable, no Ollama).

RETIRED July 2026: the old version drove llava on recorded MineRL frames to
prove the vision scaffold's VLM could see trunks. The magnet redesign deleted
that VLM channel entirely — the magnet's "vision" is now the phase-2 GROUNDING
HEAD's per-object probabilities predicted from the agent's OWN RSSM latent, and
its competence is measured LIVE in the run (the "Symbol grounding" panel's
agreement/reliability), not by a separate llava probe.

What remains worth checking as a standalone artifact is the INPUT CONTRACT the
magnet depends on: the grounding head yields a probability for each of the
magnet's target categories, and the loop's `_grounded_object_probs` extraction
hands the magnet exactly the dict shape it consumes. That is what this file now
verifies, using a real GroundedSymbolHead (no VLM, no env, no Ollama).
"""
import numpy as np
import torch

from developmental_ai.llm.vlm_symbolizer import (
    VLMSymbolizer, GroundedSymbolHead, PREDICATES)
from developmental_ai.llm.vision_scaffold import VisionScaffold, DEFAULT_TARGETS


def _stub_query(*_a, **_k):
    return None            # never call a real VLM in this smoke


def main():
    latent_dim = 32
    sym = VLMSymbolizer(latent_dim=latent_dim, hidden_dim=48,
                        min_labels=1, query_fn=_stub_query)
    assert isinstance(sym.head, GroundedSymbolHead)

    # 1. the head yields a probability for EVERY predicate, and in particular
    #    for each of the magnet's target interactables.
    lat = torch.randn(1, latent_dim)
    probs = sym.head.predict(lat).squeeze(0)
    assert probs.shape[0] == len(PREDICATES), (
        f"head width {probs.shape[0]} != {len(PREDICATES)} predicates")
    for cat in DEFAULT_TARGETS:
        assert cat in PREDICATES, f"magnet target {cat} not a grounded predicate"
        p = float(probs[PREDICATES.index(cat)])
        assert 0.0 <= p <= 1.0, f"{cat} prob out of range: {p}"
    print(f"[pod-smoke] grounding head covers all {len(DEFAULT_TARGETS)} magnet "
          f"targets with valid probabilities")

    # 2. the extraction the loop performs (mirrors _grounded_object_probs)
    #    hands the magnet a {predicate: float} dict + reliability + labels, and
    #    the magnet consumes them without error and produces finite shaping.
    object_probs = {p: float(probs[i]) for i, p in enumerate(PREDICATES)}
    reliability = sym.reliability
    label_counts = {p: 5 for p in PREDICATES}     # pretend all are grounded
    mag = VisionScaffold(target_categories=DEFAULT_TARGETS, min_labels=1,
                         present_threshold=0.5)
    r = mag.step_shaping(action=0, timestep=0, lp_scalar=0.5,
                         object_probs=object_probs, reliability=reliability,
                         label_counts=label_counts)
    assert np.isfinite(r), f"magnet produced non-finite shaping: {r}"
    st = mag.stats
    assert set(st["cat_lp"].keys()) == set(DEFAULT_TARGETS)
    print(f"[pod-smoke] loop extraction -> magnet plumbing ok "
          f"(shaping={r:.4f}, stats={st})")

    # 3. an untrusted category (too few labels / low reliability) never becomes
    #    the target even if the head is confident — the trust gate holds.
    mag2 = VisionScaffold(target_categories=DEFAULT_TARGETS, min_labels=5,
                          reliability_floor=0.35, present_threshold=0.5)
    hot = {c: 0.99 for c in DEFAULT_TARGETS}
    for t in range(50):
        mag2.step_shaping(0, t, 1.0, hot,
                          reliability={c: 0.0 for c in DEFAULT_TARGETS},   # untrusted
                          label_counts={c: 0 for c in DEFAULT_TARGETS})    # no labels
    assert mag2._target is None, (
        f"magnet targeted an untrusted category: {mag2.stats}")
    print("[pod-smoke] trust gate holds (no labels / zero reliability -> inert)")
    print("[pod-smoke] ALL PASS")


if __name__ == "__main__":
    main()
