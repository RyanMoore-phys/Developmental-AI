"""Lifelong RSSM cross-segment carry fix + real liveness metric (needs torch).

The bug: rssm.observe_step RETURNS A NEW dict, so the loop's local rssm_state
diverges from self._ll.rssm_state (which stayed the startup seed). Each segment
then restarted the "continuous" deterministic state from the seed — a silent
no-op that defeated multi-day continuity. The ||h|| instrument couldn't catch it
(gru_norm LayerNorm pins ||h|| at ~sqrt(H) for ANY h).

Contracts:
  1. The one-line loop fix (self._ll.rssm_state = rssm_state after observe_step)
     propagates the EVOLVED state; without it, _ll stays frozen at the seed.
  2. h_liveness() reads ~0 for a frozen (unchanged) h and clearly >0 once h
     evolves — the signal ||h|| could never provide.
"""
import math

import torch

from developmental_ai.core.lifelong_state import LifelongState


def test_carry_propagates_evolved_state():
    seed = {"h": torch.zeros(2, 4), "z": torch.zeros(2, 4)}
    ls = LifelongState(seed, state_decay=1.0)
    local = ls.rssm_state
    for _ in range(3):                       # mimic observe_step: NEW dict/step
        local = {"h": local["h"] + 1.0, "z": local["z"]}
        ls.rssm_state = local                # THE FIX (loop L~2931)
    assert ls.rssm_state is local, "carry did not track the evolved local state"
    assert torch.allclose(ls.rssm_state["h"], torch.full((2, 4), 3.0)), \
        "evolved state not propagated into LifelongState"
    assert not torch.allclose(ls.rssm_state["h"], seed["h"]), \
        "carried state still equals the frozen seed"

    # WITHOUT the sync, _ll stays pinned at the seed (documents the bug)
    ls2 = LifelongState({"h": torch.zeros(2, 4), "z": torch.zeros(2, 4)},
                        state_decay=1.0)
    seed2 = ls2.rssm_state
    l2 = ls2.rssm_state
    for _ in range(3):
        l2 = {"h": l2["h"] + 1.0, "z": l2["z"]}   # no write-back
    assert ls2.rssm_state is seed2 and torch.allclose(
        ls2.rssm_state["h"], torch.zeros(2, 4)), "bug repro broke"
    print("  1. cross-segment carry propagates evolved h (frozen without the fix)")


def test_h_liveness_detects_freeze():
    h0 = torch.randn(3, 6)
    ls = LifelongState({"h": h0.clone(), "z": torch.zeros(3, 6)}, state_decay=1.0)
    assert math.isnan(ls.h_liveness()), "first call has no prior segment -> NaN"
    # FROZEN: same h across segments -> cosine distance ~0
    ls.rssm_state = {"h": h0.clone(), "z": torch.zeros(3, 6)}
    d_frozen = ls.h_liveness()
    assert d_frozen < 1e-5, f"frozen h should read ~0 evolution: {d_frozen}"
    # EVOLVED: h moves -> cosine distance clearly > 0
    ls.rssm_state = {"h": h0 + 0.7 * torch.randn(3, 6), "z": torch.zeros(3, 6)}
    d_live = ls.h_liveness()
    assert d_live > 1e-3, f"evolved h should read >0: {d_live}"
    # ||h|| by contrast is LayerNorm-blind: near-constant regardless
    print(f"  2. h_liveness: frozen={d_frozen:.2e} evolved={d_live:.3f} "
          "(||h|| could not tell these apart)")


if __name__ == "__main__":
    for fn in (test_carry_propagates_evolved_state, test_h_liveness_detects_freeze):
        print(f"[lifelong-carry-smoke] {fn.__name__}")
        fn()
    print("[lifelong-carry-smoke] ALL PASS")
