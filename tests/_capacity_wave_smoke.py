"""
_capacity_wave_smoke — the 2026-09-01 sizing wave: five capacity increases,
landed together, and what they cost.

Run:  PYTHONPATH=. ./venv/bin/python tests/_capacity_wave_smoke.py

WHY THESE FIVE, AND WHY TOGETHER
================================================================================
  1. world_model.batch_size            8 -> 16
  2. curiosity.feature_dim           256 -> 512
  3. policy.hidden_dim               256 -> 512
  4. prospection.horizon              15 -> 24
  5. world_model.encoder/decoder_hidden 512 -> 768

They are together because there was NOTHING TO INVALIDATE: no pod, no live
brain, and skill_bank.storage_dir was already repointed to an empty directory
by the arch: rssm change. Two of the five are checkpoint-breaking (3 orphans
every stored skill, 5 orphans world_model.pt), so doing them now is free and
doing them after a run would destroy accumulated state.

THE COST IS REAL AND IS NAMED: five capacity changes in one Minecraft run
cannot be told apart. Attribution is bought on Crafter, where each is a
one-key ablation and a 1M-step run is a weekend on a Mac Mini.

WHAT THIS FILE CAN AND CANNOT CHECK
================================================================================
This machine has no GPU, no pod and no MineRL. So this asserts CONTRACTS and
SHAPES and prints MEASURED parameter counts — it does not and cannot measure
VRAM high-water, WM block time, or step rate. Those have instruments now
(`AsyncWM ... block ms/iter`, `Phase timing:`, `Curiosity (mean/step)`) and
must be read on the first cluster run. Saying so here is the point: a smoke
test that implied it had validated a capacity change would be worse than no
test at all.

THE FREE WIN inside change (1), which is the one most worth having: the
world-model batch quotas are int(fraction x batch_size). At 8 that gave
int(0.15*8)=1 goal-replay window and int(0.05*8)=0 terminal. At 16 it gives
TWO goal windows. Goal-prioritised replay doubles for nothing, and that is the
mechanism most starved of the rare log break the whole scoreboard turns on.
"""
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

CFG = os.path.join("configs", "minecraft_skybot.yaml")
C = yaml.safe_load(open(CFG))
W, E, CUR, POL = C["world_model"], C["environment"], C["curiosity"], C["policy"]
OBS = 3 * E["image_size"] ** 2
LAT = W["deterministic_size"] + W["stochastic_size"] * W["stochastic_classes"]
A = 13


def _mb(m):
    return sum(p.numel() for p in m.parameters()) * 4 / 1e6


# ------------------------------------------------------------------ 1 -----
def test_config_values():
    assert int(W["batch_size"]) == 16, W["batch_size"]
    assert int(CUR["feature_dim"]) == 512, CUR["feature_dim"]
    assert int(POL["hidden_dim"]) == 512, POL["hidden_dim"]
    assert int(C["prospection"]["horizon"]) == 24, C["prospection"]["horizon"]
    assert int(W["encoder_hidden"]) == 768 == int(W["decoder_hidden"])
    print("  1. all five config values are what the wave says they are")

    # DEAD KEY, FOUND BY MEASURING THE WAVE (2026-09-01). `decoder_hidden` is
    # read by nothing, and CNNDecoder ignores the hidden_dim it is handed —
    # its size comes from latent_dim and the image_size channel ladder. The
    # 512 -> 768 change moved the decoder by exactly 0 bytes. The loop must
    # SAY so at boot rather than let the key imply capacity that is not
    # there; this is the second instance of the policy.batch_size class.
    loop_src = open(os.path.join(
        "developmental_ai", "core", "developmental_loop.py")).read()
    assert "decoder_hidden=%s is NOT read by anything" in loop_src, (
        "decoder_hidden is unread and the boot warning is missing")
    import subprocess
    _hits = subprocess.run(
        ["grep", "-rn", "decoder_hidden", "developmental_ai"],
        capture_output=True, text=True).stdout
    assert "wm_cfg[\"decoder_hidden\"]" in _hits or "decoder_hidden" in _hits
    assert "hidden_dim=wm_cfg.get(\"decoder_hidden\"" not in loop_src, (
        "if decoder_hidden ever becomes real, this contract must be rewritten")
    print("     ...and decoder_hidden is flagged as unread (it moved the "
          "decoder by 0 bytes; see test 3)")

    # image_size and remote_server_scope were explicitly ruled OUT of this
    # wave by the user; a later edit quietly changing them would invalidate
    # every comparison made against this config.
    assert int(E["image_size"]) == 128, "image_size was ruled out of scope"
    assert E["remote_server_scope"] == "all", "scouts must stay on the server"
    print("  2. the two things ruled OUT (image_size, scout worlds) unchanged")


# ------------------------------------------------------------------ 2 -----
def test_wm_capacity_and_a_real_gradient_step():
    from developmental_ai.world_model.rssm import WorldModel
    wm = WorldModel(
        obs_dim=OBS, action_dim=A, stochastic_size=W["stochastic_size"],
        stochastic_classes=W["stochastic_classes"],
        deterministic_size=W["deterministic_size"],
        hidden_dim=W["encoder_hidden"], pixel_obs=True,
        image_size=E["image_size"],
        film_conditioning=W.get("film_conditioning", False),
        inverse_dynamics=W.get("inverse_dynamics", False))
    print(f"  3. world model {_mb(wm):.1f} MB fp32 "
          f"(enc {_mb(wm.encoder):.1f} / dec {_mb(wm.decoder):.1f} / "
          f"rssm {_mb(wm.rssm):.1f}) -> x3 with Adam = {_mb(wm)*3:.0f} MB")
    assert wm.rssm.latent_dim == LAT, (wm.rssm.latent_dim, LAT)

    # A REAL forward+backward at the NEW batch size. Small seq so this stays
    # a shape-and-finiteness proof on CPU rather than a benchmark — the
    # timing question belongs to the cluster, not to a Mac.
    B, L = int(W["batch_size"]), 4
    obs = torch.rand(B, L, OBS)
    act = torch.nn.functional.one_hot(
        torch.randint(0, A, (B, L)), A).float()
    rew = torch.randn(B, L)
    cont = torch.ones(B, L)
    losses = wm.compute_loss(obs, act, rew, cont)
    total = losses["total"] if isinstance(losses, dict) else losses[0]
    assert torch.isfinite(total), total
    total.backward()
    _g = [p.grad for p in wm.parameters() if p.grad is not None]
    assert _g and all(torch.isfinite(g).all() for g in _g)
    print(f"  4. batch {B} x seq {L} forward+backward: total loss "
          f"{float(total):.4f}, all gradients finite (shapes hold at 16)")


# ------------------------------------------------------------------ 3 -----
def test_batch_quota_free_win():
    """The reason change (1) is the best of the five."""
    tf, gf = float(W["terminal_fraction"]), float(W["goal_replay_fraction"])
    old_goal, new_goal = int(8 * gf), int(16 * gf)
    assert old_goal == 1 and new_goal == 2, (old_goal, new_goal)
    assert int(16 * tf) == 0, "terminal quota should still be 0 — deaths are "
    print(f"  5. goal-replay windows per batch: {old_goal} -> {new_goal} "
          f"(terminal stays {int(16*tf)}). The rarest and most valuable "
          f"windows in the buffer double, for free")


# ------------------------------------------------------------------ 4 -----
def test_policy_trunk():
    from developmental_ai.policy.actor_critic import StandaloneActorCritic
    kd = 2 * C["goals"]["max_slots"] + 2
    pro = 17
    p = StandaloneActorCritic(
        obs_dim=OBS, action_dim=A + C["lifelong"]["active_skills"],
        hidden_dim=int(POL["hidden_dim"]), arch="rssm", enc_dim=LAT,
        knowledge_dim=kd, proprio_dim=pro)
    want_in = LAT + pro + kd
    assert p.feat_dim == LAT + pro, (p.feat_dim, LAT + pro)
    first = p.actor.shared[0]
    assert first.in_features == want_in, (first.in_features, want_in)
    assert first.out_features == 512, first.out_features
    print(f"  6. policy trunk {want_in} -> {first.out_features} "
          f"(was -> 256, a {want_in/256:.1f}:1 squeeze; now "
          f"{want_in/first.out_features:.1f}:1)")
    print(f"     actor+critic {_mb(p.actor)+_mb(p.critic):.1f} MB fp32 "
          f"-> {(_mb(p.actor)+_mb(p.critic))*3:.1f} MB with Adam")


# ------------------------------------------------------------------ 5 -----
def test_icm_feature_space():
    from developmental_ai.curiosity.learning_progress import (
        LearningProgressCuriosity)
    icm = LearningProgressCuriosity(
        obs_dim=OBS, action_dim=A, feature_dim=int(CUR["feature_dim"]),
        hidden_dim=CUR["hidden_dim"], discrete_actions=True,
        pixel_obs=True, image_size=E["image_size"],
        action_conditional=CUR.get("action_conditional", False),
        null_action=CUR.get("null_action", 0))
    o = torch.rand(2, OBS)
    n = torch.rand(2, OBS)
    a = torch.nn.functional.one_hot(torch.randint(0, A, (2,)), A).float()
    r = icm.compute_intrinsic_reward(o, a, n)
    assert r.shape == (2,) and torch.isfinite(r).all(), r
    feats = icm.encoder(o)
    assert feats.shape[1] == 512, feats.shape
    print(f"  7. ICM feature space {feats.shape[1]}-d (was 256) — the space "
          f"learning progress is measured in; {_mb(icm):.1f} MB fp32")


# ------------------------------------------------------------------ 6 -----
def test_prospection_horizon_in_ticks():
    """The horizon is a claim about GAME TIME, not decisions."""
    rep = int(E["action_repeat"])
    h = int(C["prospection"]["horizon"])
    assert h * rep >= 90, (h, rep, h * rep)
    print(f"  8. prospection horizon {h} x action_repeat {rep} = {h*rep} game "
          f"ticks (a barehanded oak log is ~60; at the old 15 it was exactly "
          f"60 — no margin to tell 'fells a log' from 'does nothing')")


# ------------------------------------------------------------------ 7 -----
def test_lp_prototype_cap_and_cache():
    """The 2026-09-02 exploration lever, and the cache that makes it affordable.

    LP is the error DROP within a prototype's own history. Evicting a
    prototype drops that history (`_bucket_err.pop`), so a mastered scene
    returns reading as brand-new — the drive that decides what is
    interesting manufacturing fake novelty out of its own forgetting. Hence
    the cap, and hence the cache: `_bucket_key` used to np.stack the WHOLE
    prototype set once per env step, which at 4096 would have been an 8x
    allocate-and-copy on the hot path.
    """
    from developmental_ai.curiosity.learning_progress import (
        LearningProgressCuriosity)
    assert int(CUR["lp_max_protos"]) == 4096, CUR["lp_max_protos"]
    pool = int(CUR["lp_pixel_pool"])
    dims = 3 * pool * pool
    kb = 4096 * dims * 4 / 1e3
    assert kb < 1000, kb
    print(f"  10. lp_max_protos=4096 x {dims} dims x 4 B = {kb:.0f} KB of "
          f"HOST ram (not VRAM) — a future bump must restate this cost")

    icm = LearningProgressCuriosity(
        obs_dim=OBS, action_dim=A, feature_dim=CUR["feature_dim"],
        hidden_dim=CUR["hidden_dim"], discrete_actions=True, pixel_obs=True,
        image_size=E["image_size"], lp_max_protos=8, lp_proto_thresh=1e-9)
    rng = np.random.RandomState(0)

    def mint(k):
        return icm._bucket_key(rng.rand(OBS).astype(np.float32),
                              update_state=True)

    # cache invalidated by WRITER 2 (insert)
    mint(0)
    assert icm._proto_arr is None, "insert must invalidate the cache"
    icm._bucket_key(rng.rand(OBS).astype(np.float32), update_state=True)
    print("  11. cache invalidated on insert (writer 2)")

    # fill past the cap to force WRITER 1 (LRU eviction)
    for _ in range(12):
        mint(0)
    assert len(icm._protos) == 8, len(icm._protos)
    assert icm._proto_arr is None, "eviction must invalidate the cache"
    print(f"  12. cache invalidated on evict (writer 1); prototype set held "
          f"at the cap ({len(icm._protos)})")

    # CACHE CORRECTNESS: a stale matrix would silently match the WRONG
    # bucket and read downstream as "LP got worse" with nothing pointing at
    # the cause. Compare against an uncached reference over the same inputs.
    probe = rng.rand(OBS).astype(np.float32)
    icm._bucket_key(probe, update_state=False)          # populates the cache
    assert icm._proto_arr is not None
    ids_ref = list(icm._protos.keys())
    arr_ref = np.stack([icm._protos[i] for i in ids_ref])
    assert list(icm._proto_ids) == ids_ref
    assert np.array_equal(icm._proto_arr, arr_ref), (
        "cached prototype matrix diverged from the live dict")
    print("  13. cached matrix identical to an uncached rebuild (a stale one "
          "would match the wrong bucket, silently)")

    # the read-only (dream/imagination) path must neither mint nor evict
    n_before = len(icm._protos)
    icm._bucket_key(rng.rand(OBS).astype(np.float32), update_state=False)
    assert len(icm._protos) == n_before, (
        "update_state=False minted a prototype — the read-only path must "
        "never mutate the set the waking loop depends on")
    print("  14. update_state=False neither mints nor evicts")


# ------------------------------------------------------------------ 8 -----
def test_dream_distill_is_visible():
    """The distill path ran every segment while the log said 'warmup'."""
    loop = open(os.path.join(
        "developmental_ai", "core", "developmental_loop.py")).read()
    assert "elif getattr(self, \"dream_augment_active\", False):" in loop, (
        "the augment/distill path needs its own log branch — the other one "
        "tests dream_training_active, the CONTROL flag, permanently False")
    assert "Dream distill:" in loop
    assert "dream_distill_eff_weight" in loop
    i = loop.index("Dream policy:     warmup")
    assert "dream_activate_after" in loop[i - 900:i], (
        "the warmup line must be gated on the AUGMENT path's own counter; "
        "using dream_warmup/total_episodes is what let it claim 'warmup "
        "(0 remaining)' for an entire run while distillation was live")
    print("  15. dream distillation has a log branch, prints eff_weight, and "
          "the 'warmup' line can no longer outlive warmup")

    # the row my VRAM chart got wrong: measured at hidden_dim 256, the loop
    # wires dream_training.hidden_dim (400). Sourced from config here so the
    # accounting cannot drift from the run again.
    from developmental_ai.policy.actor_critic import DreamActorCritic
    dh = int((C.get("dream_training") or {}).get("hidden_dim", 400))
    assert "hidden_dim=dream_cfg.get(\"hidden_dim\", 400)" in loop
    da = DreamActorCritic(latent_dim=LAT, action_dim=A, hidden_dim=dh)
    print(f"  16. dream actor-critic at the CONFIGURED hidden_dim={dh}: "
          f"{_mb(da):.1f} MB fp32 -> {_mb(da)*3:.1f} MB with Adam "
          f"(the earlier chart said 43 MB, measured at 256 — it was wrong)")


# ------------------------------------------------------------------ 9 -----
def test_what_this_cannot_check():
    """Named explicitly so nobody reads ALL PASS as 'the wave is validated'."""
    print("  17. NOT CHECKED HERE (no GPU/pod/MineRL on this machine) — read "
          "on the first cluster run:")
    for line in ("VRAM high-water",
                 "`AsyncWM ... block ms/iter` at batch 16",
                 "`Curiosity (mean/step)` under the 512-d feature space",
                 "`Phase timing: act` under prospection horizon 24",
                 "`Policy: entropy` / max_prob under the 512 trunk",
                 "whether 4096 prototypes actually reduces LP churn",
                 "the live `Dream distill: eff_weight` — near 0 means the "
                 "dream actor is inert and its capacity is not the limit"):
        print(f"       - {line}")


if __name__ == "__main__":
    print("capacity wave: five sizing changes, landed together")
    test_config_values()
    test_wm_capacity_and_a_real_gradient_step()
    test_batch_quota_free_win()
    test_policy_trunk()
    test_icm_feature_space()
    test_prospection_horizon_in_ticks()
    print("LP prototype cap + matrix cache (2026-09-02)")
    test_lp_prototype_cap_and_cache()
    print("dream distillation observability (2026-09-02)")
    test_dream_distill_is_visible()
    test_what_this_cannot_check()
    print("[capacity-wave] ALL PASS")
