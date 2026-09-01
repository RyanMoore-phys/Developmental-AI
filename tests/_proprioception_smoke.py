"""Smoke: PROPRIOCEPTION — the agent can feel its own state (rung 0).

WHY THIS EXISTS. The policy saw ONLY pixels plus a goal vector. Its hunger,
health, depth, whether it held anything, and whether it was standing in a
menu were all invisible to it. Measured consequences of that blindness:
  * the agent spent 50-60% of every episode frozen in an inventory screen —
    it could not perceive being in one, so it could not learn to leave;
  * it dug itself into a pit with no tree in frame — it could not perceive
    depth;
  * `food_level` was registered by the env and READ BY NOTHING, so starving
    was unlearnable in advance even though starving to death is a real,
    already-observed consequence.

An agent cannot learn to eat when hungry if it cannot feel hunger. This is a
missing SENSE, not a missing rule — nothing here says hunger is bad or that
menus should be closed. Consequences already exist in the world; meaning is
earned from experience, as `iron_axe enables break_spruce_log` was.

Pins:
  1. The vector is FIXED-WIDTH, NORMALIZED and TOTAL — unknown fields report
     a neutral value instead of raising, so a body sense never crashes a run.
  2. It is ENVIRONMENT-AGNOSTIC: depth (not raw y) and "carrying something"
     (not "carrying oak_log") transfer to any world, incl. a live server.
  3. It reaches the POLICY: select_action / store_transition / train_step all
     carry it, and it survives a real PPO update.
  4. It is UNGATED — proprio does NOT go through the KnowledgeConditioner,
     whose gate starts near-closed. A body is not a hypothesis to admit.
  5. proprio_dim=0 is byte-identical to the old pixels-only policy.

Run: PYTHONPATH=. python tests/_proprioception_smoke.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from developmental_ai.environments.minerl_env import MineRLEnvAdapter as E
from developmental_ai.policy.actor_critic import StandaloneActorCritic

DEV = torch.device("cpu")
OBS = 3 * 32 * 32


def main() -> None:
    # ---- 1/2. the vector: total, normalized, world-agnostic -------------
    e = E.__new__(E)
    # 8 -> 9 (2026-08-01): `swing` added, the current attack streak.
    # 10 -> 13 (2026-08-23): `moved`, `head_sin`, `head_cos`. The body could
    # feel hunger, health, depth, whether a menu covered the screen and where
    # it was looking — but NOT whether it was actually going anywhere, for an
    # agent whose every recorded failure is a failure to move (sky-staring,
    # 10,149 steps in a villager's trade menu, attacking an unreachable
    # trunk). The adapter already read xpos/zpos every step for coverage, so
    # the displacement was there the whole time and simply never handed over.
    # Still proprioception, not meaning: it describes the body and names
    # nothing in the world.
    # The invariant that matters is DIM == len(KEYS), not the literal count;
    # the count is asserted too so adding a sense stays a deliberate act.
    assert E.PROPRIO_DIM == len(E.PROPRIO_KEYS) == 13

    neutral = e._proprio({})
    assert neutral.shape == (13,) and np.isfinite(neutral).all()
    # heading is the one sense with no meaningful "fine" value, so unknown
    # reads the CENTRE — the same convention `pitch` uses
    assert neutral[11] == 0.5 and neutral[12] == 0.5
    # an ABSENT sense reads as "fine", never as "starving" — a missing
    # sensor must not look like an emergency
    assert neutral[0] == 1.0 and neutral[1] == 1.0 and neutral[2] == 1.0
    assert neutral[3] == 0.0, "unknown position must not read as buried"

    bad = e._proprio({"food": 2, "saturation": 0, "life": 6, "ypos": 11,
                      "mainhand": "none", "gui_open": True, "damage": 3.0,
                      "carrying": 0})
    good = e._proprio({"food": 20, "saturation": 20, "life": 20, "ypos": 70,
                       "mainhand": "iron_axe", "carrying": 4})
    assert bad[0] < good[0], "hunger not distinguished"
    assert bad[3] > 0.5 and good[3] == 0.0, "depth not distinguished"
    assert bad[4] == 0.0 and good[4] == 1.0, "holding-something not sensed"
    assert bad[5] == 1.0 and good[5] == 0.0, \
        "being in a MENU is not sensed — the agent cannot learn to leave one"
    assert bad[6] == 1.0 and good[6] == 0.0, "being hurt not sensed"
    assert (0.0 <= np.concatenate([bad, good])).all() and \
           (np.concatenate([bad, good]) <= 1.0).all(), "not normalized"
    # TOTAL: garbage in never raises
    for junk in ({"food": "x"}, {"ypos": None}, {"carrying": float("nan")}):
        v = e._proprio(junk)
        assert v.shape == (E.PROPRIO_DIM,)

    # ---- 3. it reaches the policy and survives a PPO update -------------
    p = StandaloneActorCritic(obs_dim=OBS, action_dim=6, hidden_dim=32,
                              knowledge_dim=10, arch="conv", enc_dim=64,
                              proprio_dim=E.PROPRIO_DIM, device=DEV)
    assert p.feat_dim == 64 + E.PROPRIO_DIM, (
        f"proprio not in the feature width ({p.feat_dim})")
    kv = np.zeros(10, np.float32)
    a, _ = p.select_action(np.random.rand(OBS).astype(np.float32),
                           knowledge=kv, proprio=bad)
    assert 0 <= a < 6
    for i in range(70):
        p.store_transition(np.random.rand(OBS).astype(np.float32), i % 6,
                           0.0, False, -1.0, 0.1, knowledge=kv, proprio=bad)
    assert len(p.rollout_proprio) == 70, "proprio rows not stored"
    m = p.train_step(n_epochs=2)
    assert np.isfinite(m["policy_loss"])
    assert not p.rollout_proprio, "rollout not cleared"
    # a step with NO self-state still stores a row — arrays must stay aligned
    p.store_transition(np.random.rand(OBS).astype(np.float32), 0, 0.0, False,
                       -1.0, 0.1, knowledge=kv, proprio=None)
    assert len(p.rollout_proprio) == 1 and not p.rollout_proprio[0].any()

    # ---- 4. proprio is UNGATED; the gate still reads perception only ----
    assert p.conditioner.gate_net[0].in_features == 64 + 10, (
        "the knowledge gate's input width changed with proprio — adding a "
        "sense must not reshape the gate, and a body must not be gated")
    # ...and it genuinely CHANGES the policy input (not silently dropped)
    with torch.no_grad():
        o = torch.zeros(1, OBS)
        f = p._encode(o)
        a1 = p._augment(f, p._prep_knowledge(kv), p._prep_proprio(bad))
        a2 = p._augment(f, p._prep_knowledge(kv), p._prep_proprio(good))
    assert a1.shape[-1] == 64 + E.PROPRIO_DIM + 10
    assert not torch.equal(a1, a2), "proprio does not affect the policy input"

    # ---- 4b. A CALLER THAT FORGETS THE BODY MUST NOT CRASH --------------
    # THE LIVE FAILURE (2026-07-27, 4 occurrences in the first 4 segments):
    #   RuntimeError: mat1 and mat2 shapes cannot be multiplied
    #                 widths track E.PROPRIO_DIM, not a literal
    # Prospection / imagination / dream-distill build the policy input
    # themselves and passed no proprio, so the vector was exactly PROPRIO_DIM
    # columns too narrow. All three live inside try/except, so this did not
    # stop the run — it SILENTLY KILLED those subsystems, which is worse.
    with torch.no_grad():
        f8 = p._encode(torch.zeros(8, OBS))          # a BATCH, as they use
        k8 = torch.zeros(8, 10)
        a_missing = p._augment(f8, k8)                # no proprio at all
        assert a_missing.shape == (8, 64 + E.PROPRIO_DIM + 10), (
            f"forgetting the body channel yields {tuple(a_missing.shape)} — "
            f"this is the (8x354 vs 362x256) crash")
        assert not a_missing[:, 64:72].any(), "pad must be NEUTRAL zeros"
        # a single self-state broadcast across an imagined batch
        a_bcast = p._augment(f8, k8, p._prep_proprio(bad))
        assert a_bcast.shape == (8, 64 + E.PROPRIO_DIM + 10)
        assert torch.equal(a_bcast[0, 64:72], a_bcast[7, 64:72])
    # and the three real call sites pass the CURRENT body, not a blank
    loop_src = open("developmental_ai/core/developmental_loop.py").read()
    assert loop_src.count("self._cur_proprio_t()") >= 3, (
        "prospection/imagination/dream-distill do not all pass the agent's "
        "actual self-state")

    # ---- 5. no body sense -> byte-identical old policy -------------------
    q = StandaloneActorCritic(obs_dim=12, action_dim=4, hidden_dim=16,
                              device=DEV)
    assert q.proprio_dim == 0 and q.feat_dim == 12
    q.select_action(np.zeros(12, np.float32))
    assert not q.rollout_proprio

    # ---- the env emits it every step, and the loop looks for it ---------
    env_src = open("developmental_ai/environments/minerl_env.py").read()
    assert 'info["proprio"] = self._proprio(' in env_src, \
        "env never emits proprio"
    assert '_w["gui_open"] = bool(_gui)' in env_src, \
        "gui state never reaches the body sense"
    assert 'self._read_scalar(raw_obs, "food_level", "food")' in env_src, \
        "hunger still unread — it was registered but never consumed"
    loop = open("developmental_ai/core/developmental_loop.py").read()
    assert "proprio_per_env=_props" in loop, "loop never feeds proprio to act()"
    assert 'proprio=_closed.get("proprio")' in loop, \
        "the SMDP row is stored without the self-state it acted on"
    assert "self._proprio_source" in loop, "proprio width not taken from the env"

    print("[proprioception-smoke] ALL PASS: 13-dim normalized body sense "
          "(hunger/saturation/health/depth/holding/in-menu/hurt/carrying/"
          "swing/pitch/moved/heading); "
          "absent senses read neutral not alarming; world-agnostic (depth "
          "not y, carrying not oak_log); reaches select_action + rollout + "
          "PPO update and changes the policy input; UNGATED with the "
          "knowledge gate's width unchanged; proprio_dim=0 unchanged")


if __name__ == "__main__":
    main()
