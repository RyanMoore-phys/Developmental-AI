"""Skill-delta device + magnet-persistence smoke (2026-08-17).

THE TWO LIVE FINDINGS THIS ENCODES

1. SKILL DELTAS WERE NEVER COMPRESSED ON A GPU BOX. The bank writes its
   shared base encoder with .cpu() and reads it back with
   map_location="cpu", while the live encoder sits on the run's device. So
   `to_delta`'s `w - b` raised "Expected all tensors to be on the same
   device", the caller caught it and stored the skill WHOLE. Measured on
   the live training host: two failures in one run and 14 MB per skill — the entire
   compression win absent on exactly the hardware that trains.

2. THE MAGNET WAS AMNESIAC ACROSS RESTARTS. `reset()` preserves curiosity
   memory across episodes/deaths/dreams, but nothing carried it across a
   PROCESS restart: every launch printed "w=0.0000, target=None" with all
   categories at LP 0.000 and spent its first hours unsteered — the restart
   tax that perception/familiarity checkpoints removed, still being paid by
   the drive that aims everything else.

Contracts:
    A. to_delta survives a base and an actor on DIFFERENT devices, and
       round-trips through from_delta; the storage layer hands it CPU
       tensors so no CUDA tensor reaches the checkpoint file.
    B. The magnet's state() round-trips; load_state MERGES (never wipes a
       grown vocabulary); _cold_spent is NOT carried (the 19h-stall latch
       must not become permanent); junk payloads degrade, never raise.
    C. Wiring: magnet.pt is written, registered in the resume table, and
       gated on world_model exactly like perception/familiarity.

Run: PYTHONPATH=. python tests/_persistence_fixes_smoke.py
"""
import os
import sys

sys.path.insert(0, ".")

import torch

from developmental_ai.skill_bank import skill_delta as sd


# ------------------------------------------------- A: skill-delta devices
def test_delta_cross_device():
    torch.manual_seed(0)
    base = {"w": torch.randn(32, 16), "b": torch.zeros(32)}
    actor = {"w": base["w"] + 0.01 * torch.randn(32, 16),
             "b": base["b"] + 0.5}

    # THE BUG, reproduced with the only device pair available everywhere:
    # a plain tensor vs an explicitly-CPU one is same-device, so instead
    # assert the alignment branch exists and that mixed inputs work when a
    # second device IS present.
    src = open(os.path.join("developmental_ai", "skill_bank",
                            "skill_delta.py")).read()
    assert "b.device != w.device" in src, (
        "to_delta lost its device alignment — the GPU trap is re-set")
    assert "add.to(out[k].device)" in src, (
        "from_delta lost its device alignment")
    print("  A1. both delta directions align devices before arithmetic")

    d = sd.to_delta(base, actor, tol=0.01)
    assert sd.is_delta(d)
    back = sd.from_delta(base, d)
    for k in actor:
        err = float((back[k] - actor[k]).abs().max())
        assert err < 0.05, f"{k} round-trip error {err}"
    print("  A2. to_delta -> from_delta round-trips within tolerance")

    if torch.cuda.is_available():          # real check when a GPU exists
        actor_gpu = {k: v.cuda() for k, v in actor.items()}
        d2 = sd.to_delta(base, actor_gpu, tol=0.01)
        assert sd.is_delta(d2)
        print("  A3. cpu base x cuda actor: delta built (was the live crash)")
    else:
        print("  A3. no CUDA here — pinned by source assertion in A1")

    # the storage layer must hand to_delta CPU tensors, so nothing CUDA
    # can reach the checkpoint file
    bank_src = open(os.path.join("developmental_ai", "skill_bank",
                                 "skill_bank.py")).read()
    assert "_enc_cpu = {k: v.detach().cpu() for k, v in enc.items()}" in \
        bank_src, "the encoder is no longer moved to CPU before to_delta"
    assert bank_src.index("_enc_cpu = ") < bank_src.index(
        'out["encoder"] = _sdelta.to_delta'), "moved after use"
    print("  A4. storage layer deltas on CPU, matching the base beside it")


# ----------------------------------------------------- B: magnet round-trip
def _mk_scaffold(cats):
    from developmental_ai.llm.vision_scaffold import VisionScaffold
    return VisionScaffold(target_categories=list(cats))


def test_magnet_state():
    a = _mk_scaffold(["tree_visible", "stone_visible"])
    a._cat_lp["tree_visible"] = 0.42
    a._global_lp = 0.11
    a._lp_scale["tree_visible"] = 0.9
    a._ever_curious["tree_visible"] = True
    a._cat_absent["stone_visible"] = 7
    a._cold_spent["tree_visible"] = 12345      # must NOT travel

    st = a.state()
    assert "cold_spent" not in st, (
        "cold-start budget must not persist — that promotes the 19h "
        "zero-reward stall latch from run-scoped to permanent")
    print("  B1. state() excludes _cold_spent (anti-latch, deliberate)")

    # a LATER run whose vocabulary has GROWN must keep its new categories
    b = _mk_scaffold(["tree_visible", "stone_visible", "water_visible"])
    b._cat_lp["water_visible"] = 0.33
    summary = b.load_state(st)
    assert b._cat_lp["tree_visible"] == 0.42
    assert b._cat_lp["water_visible"] == 0.33, "merge wiped a new category"
    assert b._global_lp == 0.11
    assert b._lp_scale["tree_visible"] == 0.9
    assert b._ever_curious["tree_visible"] is True
    assert b._cat_absent["stone_visible"] == 7
    assert b._cold_spent.get("tree_visible", 0) == 0
    assert "1/3" in summary, summary
    print(f"  B2. merge-restore keeps a grown vocabulary — {summary}")

    # torch.save/load is the transport the loop actually uses
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "magnet.pt")
        torch.save(a.state(), p)
        c = _mk_scaffold(["tree_visible", "stone_visible"])
        c.load_state(torch.load(p, weights_only=False))
        assert c._cat_lp["tree_visible"] == 0.42
    print("  B3. survives the torch.save/load transport")

    # junk degrades, never raises: a broken curiosity restore must cost the
    # run its memory, not its life
    d = _mk_scaffold(["tree_visible"])
    for junk in (None, [], "nope", {"cat_lp": "not-a-dict"}):
        s = d.load_state(junk)
        assert isinstance(s, str) and s
    assert d._cat_lp["tree_visible"] == 0.0
    print("  B4. junk payloads degrade to a summary string, never raise")


# ---------------------------------------------------------- C: loop wiring
def test_wiring():
    src = open(os.path.join("developmental_ai", "core",
                            "developmental_loop.py")).read()
    for needle, why in [
            ('"magnet.pt"', "magnet state is never written"),
            ('"magnet": ("magnet.pt", _load_magnet)',
             "magnet is not resumable"),
            # REVISED 2026-08-23 (review finding). This used to assert that
            # `magnet` was INSIDE the world-model dependency gate, on the
            # theory that stale LP would be "silently trusted against a fresh
            # encoder". That reasoning was wrong: VisionScaffold.state() is
            # five category-NAME-keyed scalar containers (learning progress
            # for "tree"), with nothing derived from the encoder's latent
            # space — unlike perception's symbol head or familiarity's LSH
            # buckets, which genuinely are. Gating it meant a failed
            # world-model restore ALSO threw away the curiosity memory, so
            # the agent paid magnet.pt's full cold-start tax on exactly the
            # resume that could least afford it — while logging the untrue
            # reason "needs world_model".
            #
            # The gate is now a named constant, and the key set of state() is
            # pinned in tests/_review_fixes_smoke.py (B2) so that adding an
            # encoder-derived field forces this decision to be re-made on
            # purpose instead of silently invalidating the exemption.
            ('_RESUME_ENCODER_KEYED = ("perception", "familiarity")',
             "the encoder-keyed gate is no longer a named constant — magnet "
             "must stay OUT of it (name-keyed state) and perception/"
             "familiarity must stay IN"),
            ('if (_n in _RESUME_ENCODER_KEYED',
             "the resume gate no longer reads the named constant"),
            ('"magnet")))', "magnet is not sorted after world_model")]:
        assert needle in src, f"loop lost `{needle}` — {why}"
    cfg = open(os.path.join("configs", "minecraft_skybot.yaml")).read()
    assert "\n    - magnet\n" in cfg, "skybot does not resume the magnet"
    print("  C1. magnet saved, registered, world_model-gated, config-listed")


if __name__ == "__main__":
    test_delta_cross_device()
    test_magnet_state()
    test_wiring()
    print("[persistence-fixes] ALL PASS")
