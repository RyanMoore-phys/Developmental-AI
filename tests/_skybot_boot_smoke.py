"""BOOT TEST: build the agent from the REAL shipped config and exercise the
full skill lifecycle, without needing Minecraft.

WHY THIS EXISTS
  Every subsystem here has its own passing unit test, and on 2026-08-01 the
  COMPOSITION was still broken in two places that no unit test could see:

    1. `save_skill` hardcoded "conv" as the only encoded arch, so every
       arch="wm" skill was refused outright — nothing could ever be minted.
    2. the option executor's child-mask branch tested `arch == "conv"`, so
       wm skills silently lost skill-invokes-skill.

  Both were found by accident while building something else. Either would
  have surfaced hours into a multi-day run on a live server, after the first
  skill mint. A unit test proves a part works; this proves the parts fit.

WHAT IT COVERS
  1. the shipped config parses and its cross-key invariants hold
  2. the policy builds at the config's arch and shares the WM encoder
  3. a skill MINTS through the real SkillBank (the arch-guard trap)
  4. it BINDS into an option slot and produces an action (the encoder-class
     and child-mask traps)
  5. practice runs on a successful invocation and changes the weights
  6. the practised skill persists and reloads with behaviour intact
  7. delta-form storage round-trips through the same path

Run: PYTHONPATH=. python tests/_skybot_boot_smoke.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.policy.actor_critic import StandaloneActorCritic
from developmental_ai.policy.options import SkillOptionBank, OptionExecutor
from developmental_ai.skill_bank.skill_bank import SkillBank
from developmental_ai.skill_bank.skill_practice import SkillPractice
from developmental_ai.world_model.rssm import WorldModel

CFG_PATH = "configs/minecraft_skybot.yaml"


def _cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def test_config_invariants():
    c = _cfg()
    env, par, ll = c["environment"], c["parallel_envs"], c["lifelong"]

    # lifelong REQUIRES >1 env: num_envs<=1 disables parallel mode, and the
    # loop then raises at construction. A config that cannot boot is worse
    # than one that boots badly.
    assert ll["enabled"] and par["enabled"] and par["num_envs"] > 1, (
        "lifelong needs parallel_envs.enabled with num_envs>1 — this config "
        "would raise in DevelopmentalLoop.__init__")

    # options require the parallel discrete path or they silently vanish
    assert c["skills_as_options"]["enabled"]
    # k_slots is INERT under lifelong; active_skills is the real knob
    assert ll["active_skills"] == c["skills_as_options"]["k_slots"], (
        "k_slots disagrees with lifelong.active_skills — the latter wins and "
        "the former is a lie to whoever reads it")

    # arch=wm derives its width from the world model; they must agree
    if c["policy"].get("arch") == "wm":
        assert "enc_dim" not in c["policy"], (
            "policy.enc_dim is derived from world_model.encoder_hidden under "
            "arch=wm; setting it invites the two to drift apart")

    # the scripted-skill contract (tests/_no_scripted_skills_smoke.py)
    assert not c["skills_as_options"].get("scripted_options"), (
        "scripted option reintroduced")

    # dream: control=true would make the ENTIRE dream block dead in lifelong
    dt = c["dream_training"]
    if dt.get("enabled"):
        assert not dt.get("control", True), (
            "dream_training.control=true is a no-op under lifelong AND "
            "disables augment mode — every dream key would be dead")
        # the anneal's "solved" bar must exceed incidental segment reward, or
        # distillation anneals itself off on a dirt break
        assert float(dt.get("solve_threshold", 0.1)) >= 1.0, (
            "solve_threshold is calibrated for EPISODE reward; in lifelong a "
            "1024-step segment crosses 0.1 on one dirt break")
    print("  1. shipped config invariants ok (lifelong/options/arch/dream)")


def _build(tmp, c):
    """The real objects, wired the way developmental_loop wires them."""
    env_c, wm_c, pol_c = c["environment"], c["world_model"], c["policy"]
    side = int(env_c["image_size"])
    obs_dim = 3 * side * side
    P = 12
    K = int(c["lifelong"]["active_skills"])

    wm = WorldModel(obs_dim=obs_dim, action_dim=P, pixel_obs=True,
                    image_size=side, hidden_dim=int(wm_c["encoder_hidden"]),
                    stochastic_size=int(wm_c["stochastic_size"]),
                    stochastic_classes=int(wm_c["stochastic_classes"]),
                    deterministic_size=int(wm_c["deterministic_size"]))
    arch = pol_c.get("arch", "flat")
    enc_dim = (int(wm_c["encoder_hidden"]) if arch == "wm"
               else int(pol_c.get("enc_dim", 256)))
    pol = StandaloneActorCritic(
        obs_dim=obs_dim, action_dim=P + K, hidden_dim=256, arch=arch,
        enc_dim=enc_dim, knowledge_dim=98, proprio_dim=9, device="cpu")
    if arch == "wm":
        pol.attach_shared_encoder(wm.encoder)
    bank = SkillBank(storage_dir=tmp)
    ob = SkillOptionBank(bank, obs_dim, P, K, "cpu",
                         c["skills_as_options"].get("cache", {}))
    pr_c = c["skills_as_options"].get("practice", {}) or {}
    if pr_c.get("enabled"):
        ob.practice = SkillPractice(
            enabled=True, lr=float(pr_c.get("lr", 1e-4)),
            kl_max=float(pr_c.get("kl_max", 0.02)),
            min_steps=int(pr_c.get("min_steps", 4)), device="cpu")
    ex = OptionExecutor(ob, int(c["parallel_envs"]["num_envs"]),
                        dict(c["skills_as_options"], spike_threshold=0.9))
    return wm, pol, bank, ob, ex, obs_dim, P, K


def test_boot_mint_bind_invoke_practise_persist():
    c = _cfg()
    tmp = tempfile.mkdtemp()
    try:
        wm, pol, bank, ob, ex, obs_dim, P, K = _build(tmp, c)
        assert pol.arch == c["policy"].get("arch", "flat")
        print(f"  2. policy built arch={pol.arch} feat_dim={pol.feat_dim} "
              f"(shared encoder attached)")

        # ---- 3. MINT through the real bank (the arch-guard trap) --------
        sid = "ach_00_break_oak_log"
        sk = bank.save_skill(
            skill_id=sid, name="break_oak_log",
            policy_state_dict=pol.get_state_dict(),
            success_rate=0.6, total_episodes=5,
            obs_dim=obs_dim, action_dim=P,
            preconditions={"tree_visible": True})
        assert sk is not None and os.path.exists(sk.policy_path)
        assert sk.arch == pol.arch, (
            f"registry recorded arch={sk.arch!r} for {pol.arch!r} weights — "
            f"binding reads the registry and would refuse this skill")
        print(f"  3. mint ok (arch={sk.arch}, enc_dim={sk.enc_dim}, "
              f"{os.path.getsize(sk.policy_path)/1e6:.2f} MB)")

        # ---- 4. BIND + ACT (encoder-class and child-mask traps) ---------
        ob.refresh_slots()
        slot = ob.slot_of_skill(sid)
        assert slot is not None, f"skill refused at bind: {ob.refused}"
        obs = np.random.rand(obs_dim).astype(np.float32)
        acts = {ob.skill_action(slot, obs, env=0) for _ in range(24)}
        assert acts, "bound skill produced no action"
        assert max(acts) < P + K
        # the encoded family keeps its FULL head, so nesting is reachable
        b = ob.slots[slot]
        assert b["arch"] in ("conv", "wm"), b["arch"]
        assert int(b["head_dim"]) == P + K, (
            "stored head was truncated — skill-invokes-skill is dead")
        print(f"  4. bind + act ok (slot {slot}, head {b['head_dim']}, "
              f"actions sampled: {len(acts)} distinct)")

        # ---- 4b. SYMBOLIC CONDITIONING + PROPRIOCEPTION ARE LIVE --------
        # Both were structurally present and semantically dead: the gate was
        # dropped on a shape error and zero-padded, and proprio was never
        # passed at all. "Present" is not the property that matters —
        # LOAD-BEARING is. So assert that changing each input changes the
        # skill's output distribution. A regression that silently zeroes
        # either one fails here instead of in month three of a run.
        # 2026-08-09: _materialize grew a 4th element (the owned DeltaHead)
        actor, cond, enc, _delta = ob._materialize(slot)
        assert ob.cond_load_failures == 0, (
            f"symbolic conditioning DISABLED for {ob.cond_failed_ids} — the "
            f"skill would act on zero knowledge")
        assert cond is not None and b["ctx"] is not None, (
            "conditioner or its context vector is missing — the skill has no "
            "symbolic input at all")
        assert b["kdim"] > 0 and b["proprio_dim"] > 0, (
            f"kdim={b['kdim']} proprio_dim={b['proprio_dim']}: the input "
            f"blocks must both be non-empty for this config")
        assert b["enc_dim"] + b["proprio_dim"] + b["kdim"] == b["in_dim"], (
            "input widths do not partition the actor's input — the old "
            "subtraction bug, where kdim absorbed proprio's slots")

        def _logits(pr):
            torch.manual_seed(0)
            f = enc(torch.from_numpy(obs).reshape(1, -1))
            p = torch.as_tensor(pr, dtype=torch.float32).reshape(1, -1)
            tail = cond(f, torch.from_numpy(b["ctx"]).reshape(1, -1))
            return actor.action_head(actor.shared(
                torch.cat([f, p, tail], dim=-1)))

        starving = np.zeros(b["proprio_dim"], dtype=np.float32)
        fed = np.ones(b["proprio_dim"], dtype=np.float32)
        assert not torch.allclose(_logits(starving), _logits(fed)), (
            "body state does not change the skill's action distribution — "
            "proprio is being ignored")

        f0 = enc(torch.from_numpy(obs).reshape(1, -1))
        t_real = cond(f0, torch.from_numpy(b["ctx"]).reshape(1, -1))
        assert float(t_real.abs().sum()) > 0.0, (
            "the knowledge gate emits all zeros — conditioning is inert")
        print(f"  4b. conditioning LIVE (kdim={b['kdim']}, gate sum="
              f"{float(t_real.abs().sum()):.3f}) and proprio LOAD-BEARING "
              f"(dim={b['proprio_dim']}); widths partition in_dim")

        # ---- 5. PRACTICE on a successful invocation ---------------------
        if ob.practice is not None and ob.practice.enabled:
            actor = ob.resident_actor(slot)
            before = [p.detach().clone() for p in actor.parameters()]
            feats = torch.randn(1, b["in_dim"])
            for _ in range(12):
                ob.practice.record(0, slot, feats[0], 3)
            info = ob.practice.on_close(0, slot, actor, produced_effect=True)
            assert info and not info.get("reverted"), f"practice no-op: {info}"
            moved = any(not torch.equal(a, p)
                        for a, p in zip(before, actor.parameters()))
            assert moved, "practice reported an update but weights are frozen"
            assert info["kl"] <= ob.practice.kl_max + 1e-9, "trust region breached"
            ob.writeback_actor(slot, actor)
            print(f"  5. practice ok (kl={info['kl']:.2e} <= "
                  f"{ob.practice.kl_max}, strength={info['strength']:.2f})")

        # ---- 6. PERSIST + RELOAD ----------------------------------------
        sd = bank.load_skill_policy(sid)
        sd["actor"] = {k: v.to(torch.float32)
                       for k, v in ob.slots[slot]["actor_sd"].items()}
        bank.save_skill(skill_id=sid, name=sk.name, policy_state_dict=sd,
                        success_rate=0.7, total_episodes=6,
                        obs_dim=obs_dim, action_dim=P, dedup=False)
        bank2 = SkillBank(storage_dir=tmp)
        got = bank2.load_skill_policy(sid)
        assert got is not None, "practised skill unloadable after restart"
        assert set(got["actor"]) == set(sd["actor"])
        for k in sd["actor"]:
            assert torch.allclose(got["actor"][k].float(),
                                  sd["actor"][k].float(), atol=1e-3), k
        print("  6. persist + reload ok (practised weights survive restart)")

        # ---- 7. delta storage round-trips through the same path ---------
        from developmental_ai.skill_bank import skill_delta as sdelta
        enc = torch.load(sk.policy_path, map_location="cpu").get("encoder")
        assert enc is not None
        if sdelta.is_delta(enc):
            assert os.path.exists(os.path.join(tmp, SkillBank.BASE_ENCODER_FILE))
            print("  7. encoder stored as a DELTA against the bank base ok")
        else:
            print("  7. encoder stored in full (no base yet) — ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for fn in (test_config_invariants,
               test_boot_mint_bind_invoke_practise_persist):
        print(f"[skybot-boot-smoke] {fn.__name__}")
        fn()
    print("[skybot-boot-smoke] ALL PASS")
