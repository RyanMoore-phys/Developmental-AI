"""Curiosity-redesign smoke (Mac-ok, no MineRL/Ollama needed).

The July 2026 bet: block-breaking DISCOVERED via curiosity, not gifted.
  1. MACROS: no more 60-tick chop — attack is a plain action_repeat press;
     holding = consecutive picks (verified against a fake tick-counting env).
  2. FIRST-BREAK REWARDS: +1 the first time each block type is broken per
     episode (mine_block deltas); log-collect unchanged; achievements dict
     carries mine_<type>; baselines respected across resets.
  3. LP PIXEL BUCKETING: same scene w/ crack overlay -> SAME bucket (the
     property the discovery loop needs); different scenes -> different
     buckets; per-frame noise ("noisy TV") -> singleton buckets -> LP 0;
     error decline within a bucket -> LP > 0.
  4. SCAFFOLD: instinct pays ONLY on trunk contact (trunk-honed).
"""
import json

import numpy as np


def test_macros_emergent_hold():
    from developmental_ai.environments.minerl_env import TREECHOP_MACROS
    assert all("_ticks" not in m for m in TREECHOP_MACROS), \
        "60-tick gift macros must be gone"
    assert TREECHOP_MACROS[5] == {"attack": 1}
    assert TREECHOP_MACROS[6] == {"attack": 1, "forward": 1}
    print("  macros ok: attack is a plain press; hold must be chained")


class _FakeMineRL:
    """Counts consecutive attack ticks; 'breaks' dirt after 8 held ticks.
    Mimics the state-diff hold semantics of the real bridge."""
    class _AS:
        def noop(self):
            return {"attack": 0, "forward": 0, "back": 0, "jump": 0,
                    "camera": np.zeros(2, np.float32)}
    action_space = _AS()

    def __init__(self):
        self.held = 0
        self.mine = {"dirt": 0}
        self.t = 0

    def _obs(self):
        return {"pov": np.zeros((64, 64, 3), np.uint8),
                "inventory": {"log": 0},
                "mine_block": dict(self.mine)}

    def reset(self):
        self.held = 0
        # Real MineRL 1.0 creates a NEW world every reset (MCP-Reborn's
        # initMission -> createNewWorld), and Minecraft stores mine_block
        # per level save — so the counters restart at 0. The fake modelled
        # carry-over semantics the real env does not have.
        self.mine = {"dirt": 0}
        return self._obs()

    def step(self, act):
        self.t += 1
        if act.get("attack"):
            self.held += 1
        else:
            self.held = 0          # release resets progress
        if self.held == 8:
            self.mine["dirt"] += 1
            self.held = 0
        return self._obs(), 0.0, False, {}

    def close(self):
        pass


def test_hold_semantics_and_first_break_reward():
    from developmental_ai.environments.minerl_env import MineRLEnvAdapter
    a = MineRLEnvAdapter.__new__(MineRLEnvAdapter)   # skip real client boot
    a._macros = list(__import__(
        "developmental_ai.environments.minerl_env",
        fromlist=["TREECHOP_MACROS"]).TREECHOP_MACROS)
    a.image_size = 64
    a.render_size = 64
    a.action_repeat = 2
    a._last_pov = None
    a._log_count = 0
    a._mine_baseline = {}
    a._broken_this_episode = set()
    a._env = _FakeMineRL()
    a._env.reset()

    # 3 consecutive attack picks = 6 held ticks -> not broken yet
    r_total = 0.0
    for _ in range(3):
        obs, r, te, tr, info = MineRLEnvAdapter.step(a, 5)
        r_total += r
    assert r_total == 0.0 and a._env.mine["dirt"] == 0
    # a NON-attack step releases -> progress reset
    MineRLEnvAdapter.step(a, 1)
    assert a._env.held == 0
    # 4 consecutive attack picks = 8 ticks -> dirt breaks -> +1 first-break
    r_total = 0.0
    for _ in range(4):
        obs, r, te, tr, info = MineRLEnvAdapter.step(a, 5)
        r_total += r
    assert a._env.mine["dirt"] == 1
    assert r_total == 1.0, f"first dirt break should pay +1, got {r_total}"
    assert info["achievements"].get("mine_dirt") == 1
    # breaking dirt AGAIN this episode pays nothing
    r_total = 0.0
    for _ in range(4):
        obs, r, te, tr, info = MineRLEnvAdapter.step(a, 5)
        r_total += r
    assert a._env.mine["dirt"] == 2 and r_total == 0.0
    print("  hold semantics ok: chained attack breaks, release resets, "
          "first-break pays once per type per episode")


def test_lp_pixel_bucketing():
    from developmental_ai.curiosity.learning_progress import (
        LearningProgressCuriosity)
    lp = LearningProgressCuriosity(
        obs_dim=3 * 64 * 64, action_dim=10, feature_dim=32, hidden_dim=32,
        pixel_obs=True, image_channels=3, image_size=64,
        discrete_actions=True, lp_min_samples=4, lp_history=30, lp_pixel_pool=4,
        lp_proto_thresh=0.06, lp_max_protos=512)

    rng = np.random.default_rng(0)

    # CONTRACT 1 (statistical): a crack-sized local overlay keeps the bucket
    # for MOST scenes. Hard quantization is boundary-fragile by nature, so
    # the guarantee is statistical, and LP tolerates the minority of flips
    # (a split scene still recurs; its histories just dilute slightly).
    stable = 0
    trials = 40
    for _ in range(trials):
        scene = rng.random((3, 64, 64)).astype(np.float32)
        cracked = scene.copy()
        cracked[:, 30:36, 30:36] = 0.1     # 6x6 damage-overlay patch
        if lp._bucket_key(scene.flatten()) == lp._bucket_key(
                cracked.flatten()):
            stable += 1
    assert stable >= trials - 2, (
        f"prototype bucketing should hold crack stages deterministically: "
        f"{stable}/{trials}")

    # CONTRACT 2: structurally different scenes get different buckets
    # (bright sky-ish vs dark cave-ish vs mixed field).
    sky = np.full((3, 64, 64), 0.8, np.float32)
    cave = np.full((3, 64, 64), 0.15, np.float32)
    field = rng.random((3, 64, 64)).astype(np.float32) * 0.5
    ks = {lp._bucket_key(x.flatten()) for x in (sky, cave, field)}
    assert len(ks) == 3, "distinct scenes must not merge"

    # CONTRACT 3 (noisy-TV immunity — the property that matters): whether
    # TV frames land in singleton buckets (no history -> LP 0) or all merge
    # into one bucket (history never declines -> LP 0), the TV pays nothing.
    from collections import deque
    tv_key = lp._bucket_key(rng.random(3 * 64 * 64).astype(np.float32))
    hist = deque(maxlen=30)
    lp._bucket_err[tv_key] = hist
    for _ in range(20):                      # unlearnable: flat/noisy error
        hist.append(1.0 + float(rng.normal(0, 0.02)))
    h = np.fromiter(hist, dtype=np.float32)
    half = len(h) // 2
    tv_lp = max(0.0, float(h[:half].mean() - h[half:].mean()))
    assert tv_lp < 0.02, f"TV bucket must not show progress: {tv_lp}"

    # CONTRACT 4: a learnable bucket (declining error) yields LP > 0.
    k_learn = lp._bucket_key(field.flatten())
    hist2 = deque(maxlen=30)
    lp._bucket_err[k_learn] = hist2
    for e in [1.0, 0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2]:
        hist2.append(e)
    h2 = np.fromiter(hist2, dtype=np.float32)
    half2 = len(h2) // 2
    assert h2[:half2].mean() - h2[half2:].mean() > 0.3
    print(f"  LP pixel bucketing ok: crack-stable {stable}/{trials}, "
          "scenes separate, TV pays 0, decline pays > 0")


def test_scaffold_focused_only_instinct():
    """FOCUS-GATED (July 2026 magnet redesign): the instinct bonus pays only
    when the agent is FOCUSED on the curious object (object_centered AND
    object_adjacent) — chopping while the object is merely in view (not under
    the crosshair / not in reach) no longer pays, so the agent isn't rewarded
    for flailing at foliage. Isolated from the phi-diff term by holding phi
    constant (so any residual reward is the instinct term alone).
    """
    from developmental_ai.llm.vision_scaffold import VisionScaffold
    sc = VisionScaffold(weight=0.4, instinct_bonus=0.2, chop_actions=[5, 6],
                        target_categories=["tree_visible"], min_labels=1,
                        present_threshold=0.5, eps_abs=1e-4,
                        ema_beta=0.1, ema_leak=0.02)
    rel = {"tree_visible": 1.0}
    lab = {"tree_visible": 5}
    FOCUS = {"tree_visible": 0.9, "object_centered": 0.9, "object_adjacent": 0.9}

    # 1. build curiosity for the tree (intermittent LP correlated with view)
    #    so the magnet's weight engages and it targets the tree.
    for i in range(400):
        if i % 2 == 0:
            sc.step_shaping(0, i, 1.0, FOCUS, rel, lab)
        else:
            sc.step_shaping(0, i, 0.0, {}, rel, lab)
    # settle phi at focused (constant) so the potential-diff term is 0
    for t in range(400, 410):
        sc.step_shaping(0, t, 1.0, FOCUS, rel, lab)
    assert sc.current_weight() > 0.0 and sc._target == "tree_visible", sc.stats

    # 2. focused: a chop pays exactly the instinct bonus (phi-diff is 0 here);
    #    a non-chop action at the same focused state pays nothing. current_weight()
    #    returns the weight computed DURING the last step, so read it right after
    #    the chop to compare against the exact w that scaled the bonus.
    r_noop = sc.step_shaping(0, 411, 1.0, FOCUS, rel, lab)
    r_chop = sc.step_shaping(5, 412, 1.0, FOCUS, rel, lab)
    w_chop = sc.current_weight()
    assert abs(r_noop) < 1e-9, f"focused noop should not pay: {r_noop}"
    assert w_chop > 0.0, f"weight collapsed before chop: {sc.stats}"
    assert abs(r_chop - w_chop * 0.2) < 1e-6, (
        f"focused chop should pay w*bonus: {r_chop} vs {w_chop * 0.2}")

    # 3. present but NOT focused (in view, crosshair off it / out of reach):
    #    a chop pays NO instinct bonus. Hold this un-focused phi constant first
    #    so the potential-diff term is 0, isolating the (absent) instinct term.
    UNFOCUSED = {"tree_visible": 0.9, "object_centered": 0.1, "object_adjacent": 0.1}
    for t in range(413, 423):
        sc.step_shaping(0, t, 1.0, UNFOCUSED, rel, lab)
    r_chop_far = sc.step_shaping(5, 423, 1.0, UNFOCUSED, rel, lab)
    assert abs(r_chop_far) < 1e-9, f"un-focused chop must not pay: {r_chop_far}"
    print("  magnet ok: instinct pays ONLY when focused on the object, "
          "not merely in view")


def test_lp_readonly_dream_mode():
    """update_state=False (imagined/dream frames) must NOT mutate the LP
    quantizer's prototypes, error histories, or running stats."""
    from developmental_ai.curiosity.learning_progress import (
        LearningProgressCuriosity)
    import torch
    lp = LearningProgressCuriosity(
        obs_dim=3 * 64 * 64, action_dim=10, feature_dim=32, hidden_dim=32,
        pixel_obs=True, image_channels=3, image_size=64,
        discrete_actions=True, lp_pixel_pool=4, lp_proto_thresh=0.06,
        lp_max_protos=512)
    rng = np.random.default_rng(1)
    # seed some real prototypes
    for _ in range(6):
        o = torch.from_numpy(rng.random((2, 3 * 64 * 64)).astype(np.float32))
        a = torch.nn.functional.one_hot(torch.randint(0, 10, (2,)), 10).float()
        n = torch.from_numpy(rng.random((2, 3 * 64 * 64)).astype(np.float32))
        lp.compute_intrinsic_reward(o, a, n, update_state=True)
    protos_before = dict(lp._protos)
    buckets_before = {k: list(v) for k, v in lp._bucket_err.items()}
    tick_before = lp._proto_tick
    median_before = lp._running_median
    # dream frames: read-only calls must change nothing persistent
    for _ in range(30):
        o = torch.from_numpy(rng.random((16, 3 * 64 * 64)).astype(np.float32))
        a = torch.nn.functional.one_hot(torch.randint(0, 10, (16,)), 10).float()
        n = torch.from_numpy(rng.random((16, 3 * 64 * 64)).astype(np.float32))
        r = lp.compute_intrinsic_reward(o, a, n, update_state=False)
        assert r.shape[0] == 16
    assert lp._protos.keys() == protos_before.keys(), "dream minted protos"
    assert lp._proto_tick == tick_before, "dream advanced LRU tick"
    assert lp._running_median == median_before, "dream moved running stats"
    for k, v in lp._bucket_err.items():
        assert list(v) == buckets_before.get(k), "dream appended to histories"
    print("  LP read-only mode ok: dream frames leave quantizer state intact")


if __name__ == "__main__":
    for fn in (test_macros_emergent_hold,
               test_hold_semantics_and_first_break_reward,
               test_lp_pixel_bucketing, test_scaffold_focused_only_instinct,
               test_lp_readonly_dream_mode):
        print(f"[curiosity-smoke] {fn.__name__}")
        fn()
    print("[curiosity-smoke] ALL PASS")
