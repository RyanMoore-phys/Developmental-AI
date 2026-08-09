"""Smoke: arch v3 conv trunk — build, train, save/load, and the arch guard.

WHY THIS ARCH EXISTS (measured): under flat, 30.1 of a stored skill's 30.2M
params were THREE copies of a (·, 49250) matrix reading raw pixels — 116 MB
per skill, near-duplicate across the bank (cosine 0.869-1.000 between whole
skills). The conv trunk encodes pixels once, shared by actor/critic/gate.

Pins:
  1. DEFAULT UNCHANGED — arch="flat" builds the exact legacy stack (no
     encoder, no new state-dict keys) so every existing env/test/skill is
     byte-compatible.
  2. Conv policy: select/masked-select/train all run on flat CHW pixels;
     the ENCODER'S WEIGHTS MOVE under training (it is in the optimizer).
  3. ~10x+ param shrink at pixel scale, conditioner included.
  4. ARCH GUARD: a flat sd loaded into a conv policy (and vice versa)
     raises BEFORE any parameter is copied — torch >= 2.6 partial-copies
     then raises, which is how franken-policies happen.
  5. Conv on non-pixel obs refuses at construction.

Run: PYTHONPATH=. python tests/_conv_encoder_smoke.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from developmental_ai.policy.actor_critic import StandaloneActorCritic

DEV = torch.device("cpu")
OBS = 3 * 32 * 32       # small pixel obs so the suite runs in seconds


def n_params(pol):
    mods = [pol.actor, pol.critic]
    if pol.encoder is not None:
        mods.append(pol.encoder)
    if pol.conditioner is not None:
        mods.append(pol.conditioner)
    return sum(p.numel() for m in mods for p in m.parameters())


def main() -> None:
    # ---- 1. flat default is the legacy stack, bit-for-bit shape-wise -----
    flat = StandaloneActorCritic(obs_dim=12, action_dim=4, hidden_dim=16,
                                 device=DEV)
    assert flat.encoder is None and flat.arch == "flat"
    sd = flat.get_state_dict()
    assert "encoder" not in sd and "arch" not in sd, \
        "flat state dict grew new keys — legacy skills would stop loading"
    a, info = flat.select_action(np.zeros(12, np.float32))
    assert 0 <= a < 4 and "log_prob" in info

    # ---- 2. conv policy: forward, masked, train, encoder learns ----------
    conv = StandaloneActorCritic(obs_dim=OBS, action_dim=6, hidden_dim=32,
                                 knowledge_dim=10, arch="conv", enc_dim=64,
                                 device=DEV)
    assert conv.encoder is not None and conv.feat_dim == 64
    obs = np.random.rand(OBS).astype(np.float32)
    kv = np.random.rand(10).astype(np.float32)
    a, info = conv.select_action(obs, knowledge=kv)
    assert 0 <= a < 6
    mask = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    for _ in range(40):
        a, _ = conv.select_action(obs, knowledge=kv, action_mask=mask)
        assert mask[a], f"masked select emitted masked action {a}"
    # train: encoder weights must MOVE (it is in the optimizer + clip set)
    enc_before = conv.encoder.conv[0].weight.detach().clone()
    for i in range(66):
        conv.store_transition(np.random.rand(OBS).astype(np.float32),
                              int(i % 6), float(i % 3 == 0), i % 33 == 32,
                              -1.0, 0.1, knowledge=kv)
    m = conv.train_step(n_epochs=2)
    assert np.isfinite(m["policy_loss"])
    delta = (conv.encoder.conv[0].weight.detach() - enc_before).abs().max()
    assert float(delta) > 0, \
        "encoder never trained — it is not reached by the PPO loss"

    # ---- 3. the shrink, measured AT PRODUCTION SCALE ---------------------
    # THE SHRINK IS SCALE-DEPENDENT and a first draft of this test asserted
    # it at 32px, where it DOES NOT EXIST: there the encoder's own head
    # (Linear(2048, enc)) costs about what the flat input layer costs, so
    # the two archs tie (~0.23M each). The win comes from the flat input
    # layer growing linearly with pixels while the conv encoder stays
    # ~constant. Assert it where production runs: 128px, obs 49152.
    flat_px = StandaloneActorCritic(obs_dim=OBS, action_dim=6, hidden_dim=32,
                                    knowledge_dim=10, device=DEV)
    n_flat_small, n_conv_small = n_params(flat_px), n_params(conv)
    PROD = 3 * 128 * 128
    big_flat = StandaloneActorCritic(obs_dim=PROD, action_dim=34,
                                     hidden_dim=256, knowledge_dim=98,
                                     device=DEV)
    big_conv = StandaloneActorCritic(obs_dim=PROD, action_dim=34,
                                     hidden_dim=256, knowledge_dim=98,
                                     arch="conv", enc_dim=256, device=DEV)
    n_flat, n_conv = n_params(big_flat), n_params(big_conv)
    assert n_conv * 8 < n_flat, (
        f"conv {n_conv/1e6:.2f}M vs flat {n_flat/1e6:.2f}M at 128px — "
        f"expected >=8x; the whole point is per-skill 116MB -> ~7MB")
    # ...and the conditioner specifically stops reading raw pixels
    n_cond_flat = sum(p.numel() for p in big_flat.conditioner.parameters())
    n_cond_conv = sum(p.numel() for p in big_conv.conditioner.parameters())
    assert n_cond_conv * 20 < n_cond_flat, (
        f"conditioner {n_cond_conv} vs {n_cond_flat}: it should read "
        f"encoder features (98*354), not pixels (98*49250)")

    # ---- 4. arch guard fires BEFORE any copy -----------------------------
    conv_sd = conv.get_state_dict()
    assert conv_sd.get("arch") == "conv" and "encoder" in conv_sd
    probe = flat_px.actor.action_head.weight.detach().clone()
    try:
        flat_px.load_state_dict(conv_sd)
        raise AssertionError("conv sd loaded into a flat policy")
    except ValueError:
        pass
    assert torch.equal(probe, flat_px.actor.action_head.weight), \
        "failed load MUTATED the policy — the franken guard is a lie"
    try:
        conv.load_state_dict(flat_px.get_state_dict())
        raise AssertionError("flat sd loaded into a conv policy")
    except ValueError:
        pass
    # same-arch roundtrip works
    conv2 = StandaloneActorCritic(obs_dim=OBS, action_dim=6, hidden_dim=32,
                                  knowledge_dim=10, arch="conv", enc_dim=64,
                                  device=DEV)
    conv2.load_state_dict(conv_sd)
    assert torch.equal(conv2.encoder.conv[0].weight,
                       conv.encoder.conv[0].weight)

    # ---- 5. conv refuses non-pixel obs at construction -------------------
    try:
        StandaloneActorCritic(obs_dim=100, action_dim=4, arch="conv",
                              device=DEV)
        raise AssertionError("conv accepted obs_dim=100 (not 3*s*s)")
    except ValueError:
        pass

    # ---- 6. END-TO-END: a conv skill BINDS as an option and executes -----
    # _bind_conv/_materialize/skill_action were reachable only through the
    # live loop; the nesting suite stubs the bank out, so this path had
    # never actually run. Bind a real conv skill through the real bank.
    import shutil

    from developmental_ai.policy.options import SkillOptionBank, SlotRefused
    from developmental_ai.skill_bank.skill_bank import SkillBank
    SB = "/tmp/conv_bind_smoke"
    shutil.rmtree(SB, ignore_errors=True)
    sb = SkillBank(storage_dir=SB)
    P_, K_ = 6, 4
    live = StandaloneActorCritic(obs_dim=OBS, action_dim=P_ + K_,
                                 hidden_dim=32, knowledge_dim=0,
                                 arch="conv", enc_dim=64, device=DEV)
    sb.save_skill("ach_00_break_x", "break x", live.get_state_dict(),
                  obs_dim=OBS, action_dim=P_, arch="conv", enc_dim=64,
                  head_dim=P_ + K_, slot_map={"1": "ach_00_break_x"})
    ob = SkillOptionBank(sb, OBS, P_, K_, DEV)
    ob.refresh_slots()
    b = ob.slots[0]
    assert b is not None and b["arch"] == "conv", "conv skill failed to bind"
    assert b["head_dim"] == P_ + K_ and b["p_own"] == P_, \
        "conv head was TRUNCATED at bind — that is what removes the ability "\
        "to invoke other skills at all"
    assert b["slot_map"] == {"1": "ach_00_break_x"}

    # no mask supplied: a conv skill must STILL never emit a raw slot row
    for _ in range(60):
        a = ob.skill_action(0, obs)
        assert 0 <= a < P_, f"unmasked conv skill emitted slot row {a}"
    # with a mask opening a slot row, that row is reachable
    hm = np.zeros(P_ + K_, dtype=bool)
    hm[P_ + 1] = True
    assert ob.skill_action(0, obs, head_mask=hm) == P_ + 1, \
        "masked conv head cannot reach an opened slot row"

    # metadata that contradicts the weights is REFUSED, never guessed
    sb.save_skill("ach_01_break_y", "break y", live.get_state_dict(),
                  obs_dim=OBS, action_dim=P_, arch="conv", enc_dim=64,
                  head_dim=999, slot_map=None)
    ob2 = SkillOptionBank(sb, OBS, P_, K_, DEV)
    try:
        ob2._bind(0, sb.skills["ach_01_break_y"])
        raise AssertionError("head_dim 999 vs a real head bound anyway")
    except SlotRefused:
        pass
    # ---- 7. a re-save that OMITS arch kwargs must not corrupt the row ----
    # This is the CRITICAL review finding: the re-distill site called
    # save_skill without arch/enc_dim/head_dim/slot_map, so conv weights got
    # a `arch='flat', enc_dim=0, slot_map=None` registry row -> _bind_conv
    # refused -> `refused` never cleared -> the skill deleted itself from the
    # option bank BY IMPROVING.
    sb.save_skill("ach_00_break_x", "break x", live.get_state_dict(),
                  success_rate=0.95, total_episodes=99,
                  obs_dim=OBS, action_dim=P_, dedup=True)   # NO arch kwargs
    r = sb.skills["ach_00_break_x"]
    assert r.arch == "conv" and r.enc_dim == 64, (
        f"a caller that omitted the arch kwargs relabelled conv weights as "
        f"{r.arch}/enc_dim={r.enc_dim} — metadata must be derived from the "
        f"state dict, never trusted from a silent caller")
    assert r.head_dim == P_ + K_, f"head_dim lost on re-save: {r.head_dim}"
    assert r.slot_map == {"1": "ach_00_break_x"}, (
        f"slot_map nulled on re-save ({r.slot_map}) — it is recorded nowhere "
        f"else, so every nested-invocation row would go dark forever")
    # ...and it still binds (the whole point)
    ob3 = SkillOptionBank(sb, OBS, P_, K_, DEV)
    ob3.refresh_slots()
    assert ob3.slots[0] is not None and ob3.slots[0]["arch"] == "conv", \
        "conv skill no longer binds after a metadata-less re-save"
    # DIRECTIONALITY, stated honestly: `arch="flat"` is the DEFAULT, so an
    # explicit "flat" is indistinguishable from a caller that said nothing.
    # Believing the WEIGHTS in that direction is precisely what repairs the
    # critical bug (the re-distill caller passes nothing). The contradiction
    # that CAN be detected is the other way round — claiming conv when there
    # is no encoder to back it — and that must raise.
    flat_sd = StandaloneActorCritic(obs_dim=OBS, action_dim=P_ + K_,
                                    hidden_dim=32, device=DEV).get_state_dict()
    try:
        sb.save_skill("ach_02_break_z", "z", flat_sd, obs_dim=OBS,
                      action_dim=P_, arch="conv", enc_dim=64)
        raise AssertionError(
            "arch='conv' accepted for FLAT weights — the registry would "
            "describe an encoder that does not exist and _bind would route "
            "to _bind_conv on a flat state dict")
    except ValueError:
        pass

    # ---- 7. a re-save that OMITS arch kwargs must not corrupt the row ----
    # This is the CRITICAL review finding: the re-distill site called
    # save_skill without arch/enc_dim/head_dim/slot_map, so conv weights got
    # a `arch='flat', enc_dim=0, slot_map=None` registry row -> _bind_conv
    # refused -> `refused` never cleared -> the skill deleted itself from the
    # option bank BY IMPROVING.
    sb.save_skill("ach_00_break_x", "break x", live.get_state_dict(),
                  success_rate=0.95, total_episodes=99,
                  obs_dim=OBS, action_dim=P_, dedup=True)   # NO arch kwargs
    r = sb.skills["ach_00_break_x"]
    assert r.arch == "conv" and r.enc_dim == 64, (
        f"a caller that omitted the arch kwargs relabelled conv weights as "
        f"{r.arch}/enc_dim={r.enc_dim} — metadata must be derived from the "
        f"state dict, never trusted from a silent caller")
    assert r.head_dim == P_ + K_, f"head_dim lost on re-save: {r.head_dim}"
    assert r.slot_map == {"1": "ach_00_break_x"}, (
        f"slot_map nulled on re-save ({r.slot_map}) — it is recorded nowhere "
        f"else, so every nested-invocation row would go dark forever")
    # ...and it still binds (the whole point)
    ob3 = SkillOptionBank(sb, OBS, P_, K_, DEV)
    ob3.refresh_slots()
    assert ob3.slots[0] is not None and ob3.slots[0]["arch"] == "conv", \
        "conv skill no longer binds after a metadata-less re-save"
    # DIRECTIONALITY, stated honestly: `arch="flat"` is the DEFAULT, so an
    # explicit "flat" is indistinguishable from a caller that said nothing.
    # Believing the WEIGHTS in that direction is precisely what repairs the
    # critical bug (the re-distill caller passes nothing). The contradiction
    # that CAN be detected is the other way round — claiming conv when there
    # is no encoder to back it — and that must raise.
    flat_sd = StandaloneActorCritic(obs_dim=OBS, action_dim=P_ + K_,
                                    hidden_dim=32, device=DEV).get_state_dict()
    try:
        sb.save_skill("ach_02_break_z", "z", flat_sd, obs_dim=OBS,
                      action_dim=P_, arch="conv", enc_dim=64)
        raise AssertionError(
            "arch='conv' accepted for FLAT weights — the registry would "
            "describe an encoder that does not exist and _bind would route "
            "to _bind_conv on a flat state dict")
    except ValueError:
        pass

    shutil.rmtree(SB, ignore_errors=True)

    print(f"[conv-encoder-smoke] ALL PASS: flat default unchanged (no new "
          f"sd keys); conv select/masked/train ok with encoder learning "
          f"(delta={float(delta):.2e}); AT 128px {n_flat/1e6:.1f}M -> "
          f"{n_conv/1e6:.1f}M params ({n_flat/max(1,n_conv):.1f}x, "
          f"conditioner {n_cond_flat/1e6:.1f}M -> {n_cond_conv/1e3:.0f}k) "
          f"while at 32px they tie ({n_flat_small/1e6:.2f}M vs "
          f"{n_conv_small/1e6:.2f}M — the win is scale-dependent); arch "
          f"guard raises pre-copy both directions; non-pixel conv refused")


if __name__ == "__main__":
    main()
