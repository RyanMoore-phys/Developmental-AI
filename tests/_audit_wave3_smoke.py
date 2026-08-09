"""Lightweight smokes for audit fix-wave-3 (fast: tiny tensors, no MineRL/Ollama).

Each test targets ONE fix and runs sub-second. The runner isolates failures so a
bug in one fix documents rather than hiding the rest. NOT covered here (need a
live MineRL client / full loop): minerl_env _log_count re-baseline &
clear_break_marks, the log-pickup effect-key guard, the crash-reset ADVANCE
block — those are compile-verified + code-reviewed only.
"""
import tempfile

import numpy as np
import torch

# ---------------------------------------------------------------- seek / magnet
from developmental_ai.llm.vision_scaffold import VisionScaffold

REL = {"tree_visible": 1.0, "stone_visible": 1.0}
LAB = {"tree_visible": 5, "stone_visible": 5}
ADJ = {"object_adjacent": 0.9, "object_centered": 0.9}


def test_nudge_pure_forward_only():
    vs = VisionScaffold(target_categories=["tree_visible"], min_labels=1,
                        present_threshold=0.6, seek_weight=0.0,
                        seek_forward_nudge=0.02, seek_nudge_budget=50)
    # tree fully absent + grounded -> nudge eligible; action 6 = attack+forward
    r6 = vs.step_shaping(6, 0, 0.0, {"tree_visible": 0.0}, REL, LAB)
    assert abs(r6) < 1e-9, f"attack+forward (6) must NOT collect the nudge: {r6}"
    r1 = vs.step_shaping(1, 1, 0.0, {"tree_visible": 0.0}, REL, LAB)
    assert abs(r1 - 0.02) < 1e-9, f"pure forward (1) should get the nudge: {r1}"


def test_cold_start_budget_wanes():
    vs = VisionScaffold(target_categories=["stone_visible"], min_labels=1,
                        present_threshold=0.5, cold_start_weight=0.2,
                        cold_start_budget=5, eps_abs=1e-3)
    # stone present, never distinctively curious (lp=0) -> cold-start floor,
    # but only for `budget` steps, then it wanes to 0
    ws = []
    for t in range(12):
        vs.step_shaping(0, t, 0.0, {"stone_visible": 0.9, **ADJ}, REL, LAB)
        ws.append(vs.current_weight())
    assert ws[0] == 0.2, f"floor should engage at cold start: {ws[0]}"
    assert ws[-1] == 0.0, f"floor must wane to 0 after budget spent: {ws}"


def test_seek_streak_refill():
    vs = VisionScaffold(target_categories=["tree_visible"], min_labels=1,
                        present_threshold=0.6, seek_weight=0.0,
                        seek_forward_nudge=0.02, seek_nudge_budget=4)
    for t in range(4):                       # burn the whole budget (tree gone)
        vs.step_shaping(1, t, 0.0, {"tree_visible": 0.0}, REL, LAB)
    assert vs._seek_nudge_left == 0
    for t in range(4, 7):                    # 3 consecutive gp>=0.3 sightings
        vs.step_shaping(0, t, 0.0, {"tree_visible": 0.45}, REL, LAB)
    assert vs._seek_nudge_left == 4, (
        f"3-sighting streak should refill the budget: {vs._seek_nudge_left}")


def test_nudge_grounding_gated():
    vs = VisionScaffold(target_categories=["tree_visible"], min_labels=5,
                        present_threshold=0.6, seek_weight=0.0,
                        seek_forward_nudge=0.02, seek_nudge_budget=50)
    # grounding immature (labels < min_labels) -> gp path ungrounded -> no burn
    r = vs.step_shaping(1, 0, 0.0, {"tree_visible": 0.0}, {"tree_visible": 1.0},
                        {"tree_visible": 1})
    assert abs(r) < 1e-9, f"nudge must not burn before grounding is trusted: {r}"
    assert vs._seek_nudge_left == 50, "budget burned pre-grounding"


def test_eps_abs_handoff_to_tree():
    # a sub-eps stone score must NOT block the cold-start handoff to a virgin
    # tree that is present and higher priority
    vs = VisionScaffold(target_categories=["tree_visible", "stone_visible"],
                        min_labels=1, present_threshold=0.5,
                        cold_start_weight=0.2, eps_abs=1e-3, ema_beta=0.05)
    both = {"tree_visible": 0.9, "stone_visible": 0.9, **ADJ}
    for t in range(10):
        vs.step_shaping(0, t, 0.05, both, REL, LAB)   # uniform tiny lp
    assert vs._target == "tree_visible", (
        f"cold-start should prefer the higher-priority tree: {vs._target}")


# ------------------------------------------------------------ vlm reliability
from developmental_ai.llm.vlm_symbolizer import VLMSymbolizer


def _sym():
    return VLMSymbolizer(enabled=False, latent_dim=8, hidden_dim=8,
                         interval=60, min_labels=1)


def test_reliability_freshness_gate():
    s = _sym()
    s.last_labels = {"tree_visible": False}          # a MISS claim
    s._last_submit = 0
    base = s.reliability["tree_visible"]
    s.observe_event("block_break", "oak_log", was_chopping=True, now=10_000)
    assert s.reliability["tree_visible"] == base, (
        "stale label (age >> 2*base_interval) must NOT score reliability")
    s.observe_event("block_break", "oak_log", was_chopping=True, now=100)
    assert s.reliability["tree_visible"] < base, (
        "a FRESH label should score (miss -> reliability drops)")


def test_reliability_regresses_to_prior():
    s = _sym()
    s.reliability["tree_visible"] = 0.30             # collapsed below floor
    for t in range(400):
        s.maybe_label(None, None, t)                 # frame None -> regression only
    assert s.reliability["tree_visible"] > 0.31, (
        f"reliability should drift back toward 0.7 prior: "
        f"{s.reliability['tree_visible']}")


# ------------------------------------------------------------- exploration_ratio
from developmental_ai.curiosity.icm import IntrinsicCuriosityModule


def test_exploration_ratio_composes():
    icm = IntrinsicCuriosityModule(obs_dim=8, action_dim=2, feature_dim=8,
                                   hidden_dim=8, reward_scale=1.0,
                                   discrete_actions=True)
    icm.exploration_decay_rate = 0.02
    icm.update_exploration_ratio(0)                  # 0 mastered -> full curiosity
    assert abs(icm.reward_scale - 1.0) < 1e-6, (
        f"0 mastered skills must not quench curiosity: {icm.reward_scale}")
    icm.update_exploration_ratio(20)
    assert abs(icm.reward_scale - np.exp(-0.02 * 20)) < 1e-6, (
        f"ratio should compose with the base scale: {icm.reward_scale}")


# --------------------------------------------------------------- competence
from developmental_ai.core.self_model import CompetencePredictor


def test_competence_bias_frozen_and_decoupled():
    cp = CompetencePredictor(n_tasks=4)
    assert cp.head.bias.requires_grad is False, "shared bias must be frozen"
    p1_before = cp.predict(1)
    for _ in range(30):
        cp.update(0, 1.0)                            # only task 0 succeeds
    p0 = cp.predict(0)
    p1_after = cp.predict(1)
    assert p0 > 0.6, f"task 0 competence should rise: {p0}"
    assert abs(p1_after - p1_before) < 1e-6, (
        f"task 1 must be decoupled from task 0's updates: "
        f"{p1_before}->{p1_after}")


# ----------------------------------------------------------- goal broadcaster
from developmental_ai.core.achievement_goals import AchievementGoalBroadcast


def test_clear_stream_banks_success():
    b = AchievementGoalBroadcast(max_slots=4)
    b.slot_names = ["a", "b", "c", "d"]
    b.set_num_streams(2)
    b.target = 0
    c_before = float(b.competence.predict(0))
    b._achieved_now[1, 0] = 1.0                      # stream 1 achieved target
    b.clear_stream(1)                                # death mid-window
    c_after = float(b.competence.predict(0))
    assert c_after > c_before, (
        f"in-window success must be banked, not erased: {c_before}->{c_after}")
    # reset() must NOT re-score the already-scored stream as a failure
    c_pre_reset = float(b.competence.predict(0))
    b.reset()
    assert float(b.competence.predict(0)) >= c_pre_reset - 1e-3, (
        "reset re-scored a banked stream as failure")


def test_estimated_dag_ragged_widths():
    b = AchievementGoalBroadcast(max_slots=4)
    b.unlock_log = [
        {"slot": 0, "achieved_before": [1.0, 0.0]},           # 2-wide (legacy)
        {"slot": 0, "achieved_before": [0.0, 1.0, 0.0, 1.0]}, # 4-wide (post-widen)
    ]
    dag = b.estimated_dag()                          # must not raise
    assert 0 in dag and dag[0].shape == (4,), f"ragged rows not padded: {dag}"


# ----------------------------------------------------------- evict-for-mint
from developmental_ai.skill_bank.skill_bank import SkillBank
from developmental_ai.policy.options import SkillOptionBank


def _pol(obs_dim=4, hidden=8, p=6):
    return {"actor": {"shared.0.weight": torch.zeros(hidden, obs_dim),
                      "shared.0.bias": torch.zeros(hidden),
                      "action_head.weight": torch.zeros(p, hidden),
                      "action_head.bias": torch.zeros(p)}}


def _save(sb, sid, sr):
    sb.save_skill(skill_id=sid, name=sid, policy_state_dict=_pol(),
                  success_rate=sr, total_episodes=sb.min_practice_episodes + 3,
                  obs_dim=4, action_dim=6, dedup=True)


def test_evict_for_mint_when_full():
    with tempfile.TemporaryDirectory() as d:
        sb = SkillBank(storage_dir=d)
        _save(sb, "ach_00_a", 0.5)
        _save(sb, "ach_01_b", 0.3)
        bank = SkillOptionBank(sb, obs_dim=4, P=6, K=2, device=torch.device("cpu"))
        bank.refresh_slots()                         # fills both slots
        assert all(s is not None for s in bank.slots), "K=2 should be full"
        inv_slot = next(i for i, s in enumerate(bank.slots)
                        if s["skill_id"] == "ach_00_a")
        bank.note_invoked(inv_slot)                  # A is practiced; B is not
        _save(sb, "ach_02_log", 0.4)
        bank.notify_minted("ach_02_log")
        bank.refresh_slots()                         # must evict never-invoked B
        ids = [s["skill_id"] for s in bank.slots if s is not None]
        assert "ach_02_log" in ids, f"fresh mint never bound: {ids}"
        assert "ach_01_b" not in ids, f"never-invoked B should be evicted: {ids}"
        assert "ach_00_a" in ids, f"practiced A must be kept: {ids}"


# ----------------------------------------------------------- LP significance
from developmental_ai.curiosity.learning_progress import LearningProgressCuriosity


def test_lp_quenches_on_mastered_scene():
    lp = LearningProgressCuriosity(obs_dim=8, action_dim=2, feature_dim=8,
                                   hidden_dim=8, discrete_actions=True,
                                   lp_history=8, lp_min_samples=4,
                                   novelty_window=64, reward_scale=1.0)
    obs = torch.randn(1, 8)
    nxt = torch.randn(1, 8)
    act = torch.zeros(1, 2)
    vals = []
    for _ in range(50):                              # same scene repeatedly
        r = lp.compute_intrinsic_reward(obs, act, nxt)
        lp.train_step(obs, act, nxt)
        vals.append(float(np.asarray(r).reshape(-1)[0]))
    tail = float(np.mean(vals[-10:]))
    assert tail < 1.0, (
        f"mastered scene should quench (significance gate + std floor); "
        f"tail-mean intrinsic={tail:.3f}")


TESTS = [
    test_nudge_pure_forward_only, test_cold_start_budget_wanes,
    test_seek_streak_refill, test_nudge_grounding_gated,
    test_eps_abs_handoff_to_tree, test_reliability_freshness_gate,
    test_reliability_regresses_to_prior, test_exploration_ratio_composes,
    test_competence_bias_frozen_and_decoupled, test_clear_stream_banks_success,
    test_estimated_dag_ragged_widths, test_evict_for_mint_when_full,
    test_lp_quenches_on_mastered_scene,
]

if __name__ == "__main__":
    import traceback
    passed, failed = 0, []
    for fn in TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            failed.append((fn.__name__, traceback.format_exc()))
    print(f"\n[wave3-smoke] {passed}/{len(TESTS)} passed")
    if failed:
        print("\n===== FAILURE DETAIL =====")
        for name, tb in failed:
            print(f"\n--- {name} ---\n{tb}")
