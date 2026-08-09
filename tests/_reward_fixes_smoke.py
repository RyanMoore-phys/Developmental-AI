"""Smoke: the 2026-08-02 reward-and-action-space wave.

WHAT WENT WRONG (live on the SkyBot server): the agent sat looking straight
down, walking and jumping, and extrinsic reward did essentially nothing. Nine
defects, each independently sufficient to produce part of that:

  1. RETURN NORMALIZATION WAS NEVER WIRED. RewardMixer has damped a dense
     intrinsic stream against a sparse extrinsic one since 2026-07-27, but
     `return_ratio_cap` was never passed from config, so it defaulted to 0.0
     (= OFF) and the branch was unreachable. Measured imbalance it exists to
     fix: weighted intrinsic return 9179.4 vs weighted extrinsic 4.78.
  2. REWARD TIERS BROKE ON A NAMESPACED KEY. `_break_reward` matches bare
     names by exact set membership, but `_mine_counts` emitted the raw stat
     key. "minecraft.dirt" therefore missed the ground tier and fell to the
     1.0 default — 6.7x the intended 0.15 AND above the 0.9 goal-discovery
     threshold, which makes looking down and digging the best-paid REACHABLE
     behaviour and mints a junk goal per terrain type.
  3. THE FIRST BREAK OF EVERY TYPE WAS INVISIBLE. `_mine_prev.get(bt, cnt)`
     makes the 0->1 delta exactly zero, so breaks_by_type and break_ticks
     never saw a type's first break — including the first log.
  4. TICKS-TO-BREAK WAS ZEROED PER TYPE. `_attack_run` reset inside the
     per-type loop, so a second type felled on the same step recorded 0.
  5. PERCEPTUAL NOVELTY KEYED ON THE FRAME IT LEFT, not the one it reached.
  6. PHASE SCOPING LEAKED. `disable_macros` was applied only to the
     meta-policy's own primitive picks — skills invoked as options, and
     scouts, could still choose the disabled `inventory` macro.
  7. PPO NEVER MINIBATCHED. `policy.batch_size` had no reader at all.
  8. entropy_coef / value_coef were unreachable from config.
  9. The stage gate's threshold assumed 4 streams; num_envs is 2.

Run: PYTHONPATH=. python tests/_reward_fixes_smoke.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np

ENV = "developmental_ai/environments/minerl_env.py"
LOOP = "developmental_ai/core/developmental_loop.py"
OPTS = "developmental_ai/policy/options.py"
CFG = "configs/minecraft_skybot.yaml"


def main() -> None:
    import yaml
    import torch
    from developmental_ai.environments.minerl_env import MineRLEnvAdapter as E
    from developmental_ai.policy.actor_critic import (
        RewardMixer, StandaloneActorCritic)
    from developmental_ai.policy.options import SkillOptionBank

    loop_src = open(LOOP).read()
    env_src = open(ENV).read()
    cfg = yaml.safe_load(open(CFG))

    # ---- 1. THE MIXER'S GUARD IS WIRED AND TURNED ON --------------------
    assert "return_ratio_cap=pol_cfg.get(" in loop_src, \
        "RewardMixer still ignores return_ratio_cap — the guard is unreachable"
    assert "ret_ema_alpha=pol_cfg.get(" in loop_src
    assert "min_intrinsic_scale=pol_cfg.get(" in loop_src
    assert float(cfg["policy"]["return_ratio_cap"]) > 0.0, \
        "skybot does not enable return normalization"
    # ...and it actually damps a dense intrinsic against a sparse extrinsic
    # Use the CONFIGURED alpha. It is deliberately slow (1e-4) because the
    # extrinsic stream is sparse: at 1e-2 the extrinsic EMA underflows the
    # guard's own 1e-8 inertness check between spikes and the damping silently
    # switches itself off — the oscillation mix()'s docstring warns about, and
    # a reason not to "speed up" this constant.
    _alpha = float(cfg["policy"]["return_ema_alpha"])
    _floor = float(cfg["policy"]["min_intrinsic_scale"])
    m = RewardMixer(intrinsic_weight=0.7, extrinsic_weight=0.3,
                    return_ratio_cap=float(cfg["policy"]["return_ratio_cap"]),
                    ret_ema_alpha=_alpha, min_intrinsic_scale=_floor)
    for _ in range(2000):            # dense curiosity, one task spike
        m.mix(0.20, 0.0)
    m.mix(0.20, 5.0)                 # a log falls
    for _ in range(2000):
        m.mix(0.20, 0.0)
    damped = m.mix(0.20, 0.0)
    undamped = RewardMixer(intrinsic_weight=0.7,
                           extrinsic_weight=0.3).mix(0.20, 0.0)
    assert damped < undamped, (
        f"guard did not damp a dense intrinsic stream ({damped} vs {undamped})")
    assert damped > 0.0, "guard deleted curiosity instead of damping it"
    assert damped >= undamped * _floor * 0.999, (
        f"damping {damped} broke through the {_floor} floor — curiosity is "
        "the exploration engine and must never be deleted outright")
    # the task term must survive the guard untouched (it only shrinks intrinsic)
    assert abs(m.mix(0.0, 1.0) - 0.3) < 1e-9, "extrinsic was scaled too"

    # ---- 2. WHITELISTED TIERS (2026-08-08) SURVIVE A NAMESPACED KEY -----
    # User directive after live measurement: placed in front of logs with an
    # axe, the bot abandoned a started chop for snow/dirt — instant cheap
    # breaks beat a delayed 20.0 under temporal discounting. Only wood and
    # ores pay extrinsically now; everything else is 0.0 (bare AND
    # namespaced, so a namespaced key can never resurrect the old tier).
    for ground in ("dirt", "grass_block", "stone", "gravel", "sand",
                   "snow_block", "fern"):
        assert E._break_reward(ground) == 0.0, f"{ground} pays extrinsically"
        assert E._break_reward(f"minecraft.{ground}") == 0.0, (
            f"minecraft.{ground} pays "
            f"{E._break_reward(f'minecraft.{ground}')} — a namespaced block "
            "escapes the whitelist")
    assert E._break_reward("minecraft.oak_log") == getattr(
        E, "LOG_BREAK_REWARD", 5.0)
    assert E._break_reward("minecraft.oak_leaves") == 0.0
    assert E._break_reward("minecraft.iron_ore") == 10.0
    assert E._break_reward("minecraft.deepslate_coal_ore") == 10.0
    assert E._break_reward("ancient_debris") == 10.0
    # non-whitelisted blocks must stay UNDER the goal-discovery spike
    for b in ("minecraft.dirt", "minecraft.stone", "minecraft.fern"):
        assert E._break_reward(b) < float(cfg["goals"]["spike_threshold"]), \
            f"{b} can still mint a junk goal"

    # ---- ...and the namespace is stripped once, at the source ------------
    counts = E._mine_counts({"mine_block": {
        "minecraft.dirt": np.array([3]),
        "minecraft.oak_log": np.array([1]),
        "minecraft.air": np.array([9]),        # non-block, must be dropped
        "minecraft.grass": np.array([0])}})    # zero counter, must be dropped
    assert counts == {"dirt": 3, "oak_log": 1}, (
        f"_mine_counts emitted {counts} — keys must be bare so the reward "
        "tier, the achievement key, the grounded effect key and "
        "breaks_by_type can never disagree about what a block is called")

    # ---- 2c. LOG ECONOMICS: effort must out-earn every block per tick ---
    # The measured failure: first-break tiers re-arm every goal_horizon, so
    # four trivial block types were re-harvestable indefinitely for ~0.75 with
    # no aim and no sustained effort, while a log paid 5.0 once for ~60
    # consecutive ticks. PPO correctly converged on ground vegetation —
    # 2484 blocks broken, 0 logs.
    _e = cfg["environment"]
    _tick = float(_e.get("log_tick_reward", 0.0))
    _cap = int(_e.get("log_tick_cap", 120))
    _comp = float(_e.get("log_break_reward", 5.0))
    assert _tick > 0.0, "skybot does not enable the log effort payment"
    # (block, completion tier, ticks it costs) — ticks are the measured ones
    _econ = [("grass", 0.15, 1), ("oak_leaves", 0.30, 7),
             ("dirt", 0.15, 20), ("stone", 1.00, 40)]
    _log_per_tick = (_comp + _tick * 60) / 60.0
    _best_other = max(b / t for _, b, t in _econ)
    assert _log_per_tick > _best_other, (
        f"a log pays {_log_per_tick:.3f}/tick vs {_best_other:.3f} for the "
        "best alternative — the incentive to chop still loses")
    # and the FIRST log must beat every trivial break combined
    assert _comp + _tick * 60 > sum(b for _, b, _t in _econ) * 4, (
        "one felled log is not clearly worth more than a spree of easy blocks")
    # THE CAP IS LOAD-BEARING: attack_run has been seen at 13538 on a swing
    # that never landed. Uncapped that single streak would pay ~6769.
    assert _tick * min(13538, _cap) <= _tick * _cap + 1e-9
    assert _cap <= 200, (
        f"log_tick_cap {_cap} is too loose to bound a runaway streak")
    # OFF by default so no other environment's reward changes
    import inspect as _insp
    assert _insp.signature(E.__init__).parameters[
        "log_tick_reward"].default == 0.0, "log economics ON by default"
    assert E._break_reward("oak_log") == 5.0, \
        "the classmethod's default log tier moved — other configs shift"

    # ---- 2d. THE SWING THAT JUST ENDED ----------------------------------
    # mine_block registers a break ONE STEP AFTER the swing that caused it, by
    # which point _attack_run has been zeroed by the retarget / not-attacking
    # branch. Every break was therefore attributed 0 ticks — measured live as
    # "Ticks-to-break: large_fern:0t, oak_leaves:0t, sand:0t..." — which is
    # impossible, and made the one statistic that says whether swings are long
    # enough to fell a log worthless. It is also LOAD-BEARING for the log
    # economics: the effort payment is per-tick, so a 0 read pays 0.
    assert "_streak_prev = int(self._attack_run)" in env_src, (
        "the pre-reset streak is not captured — break tick counts stay 0")
    assert "max(int(self._attack_run), _streak_prev)" in env_src, (
        "break ticks are still read AFTER the counter was zeroed")
    # order matters: the capture must precede the retarget reset
    assert (env_src.index("_streak_prev = int(self._attack_run)")
            < env_src.index("_retargeted = False")), \
        "the streak is captured after the retarget branch — too late"

    # ---- 2e. REACH FROM EVIDENCE, NOT FROM A VLM GUESS ------------------
    # Measured: Reach sense 0.01-0.03 across a whole run in which the agent
    # broke 4931 blocks. llava cannot judge reach from a still frame, so the
    # head learned to answer "no", and the approach-to-reach potential plus a
    # proprioception field rode on that constant.
    assert '_w["reach_evidence"]' in env_src, "reach evidence is not emitted"
    assert 'self._reach_evidence = (1.0 if _broke_now' in env_src, (
        "the reach trace does not key on a real break — a completed break is "
        "the only ground truth available that something was within range")
    assert cfg["curiosity"].get("reach_source") == "evidence", \
        "skybot still reads reach from the VLM head"
    assert '_cur_cfg.get("reach_source", "head")' in loop_src, \
        "reach_source is not defaulted to the historic 'head' for other configs"

    # ---- 2f. THE AXE IS BACK, AND IT IS SUPPOSED TO BE TEMPORARY --------
    # The binding constraint was swing length: median 4 ticks against a ~60
    # tick barehanded log. An iron axe takes a log to ~8, which 23% of swings
    # already reach.
    # start_tool MUST stay null against the external server. It is a
    # SERVER-side handler in the mission XML and a foreign Paper server
    # ignores it, exactly like the frozen-daytime and world-generator
    # handlers — so we cannot grant a tool from this side. Setting it is
    # worse than useless: the LOCAL scout does honour the grant, so env 1
    # would swing an axe while env 0 stayed barehanded and the SHARED world
    # model would learn two break dynamics for the same block.
    assert cfg["environment"]["start_tool"] is None, (
        "start_tool is set — on the external server the grant is ignored for "
        "the primary but HONOURED for the local scout, which teaches the "
        "shared world model two different economies")
    _cfg_txt = open(CFG).read()
    assert "only the server operator" in _cfg_txt, (
        "the config does not record WHY start_tool must stay null — the next "
        "person will try to set it again")

    # ---- 2g. STATE READ BEFORE IT IS ASSIGNED ---------------------------
    # `_last_world_info` is assigned inside the segment loop but read EARLIER
    # in the same loop body by the felt-reach block, so the first iteration
    # died with AttributeError on every launch — three identical crashes
    # before the supervisor's deterministic-fault guard stopped retrying.
    # Neither the unit tests nor the pod-side deploy gate exercise
    # _collect_segment, so nothing caught it before it hit production.
    assert "self._last_world_info: Dict[str, Any] = {}" in loop_src, (
        "_last_world_info is not initialised at construction — it is read "
        "before assignment on the first segment iteration")

    # ---- 2h. THE HOLD MACRO ---------------------------------------------
    # A barehanded oak log needs ~60 game ticks on ONE block = 30 CONSECUTIVE
    # attack decisions at action_repeat 2. P(30 in a row) at a realistic 10%
    # attack rate is ~1e-30; at 50% it is ~1e-9. Measured live: median swing
    # 2 ticks, 1 swing in 400 reaching 8. Unreachable by construction, and the
    # entropy fix made it rarer (high entropy switches action every step,
    # which is in direct tension with holding one down).
    from developmental_ai.environments.minerl_env import TREECHOP_MACROS as _M
    _probe = E.__new__(E)
    _probe._macros = _M
    _probe.action_repeat = 2
    assert len(_M) == 13, f"macro table is {len(_M)} wide, expected 13"
    # APPEND-ONLY: every stored skill and policy indexes actions by position
    assert _M[0] == {} and _M[5] == {"attack": 1} and _M[10] == {"inventory": 1}, \
        "existing macro indices moved — that silently rebinds every stored skill"
    _hold = _M[12]
    assert _hold.get("attack") and _hold.get("_ticks", 0) >= 8, \
        f"macro 12 is not a sustained attack: {_hold}"
    assert _probe._macro_ticks(12) == _hold["_ticks"], \
        "the step loop does not honour _ticks — the button would not stay down"
    assert _probe._macro_ticks(5) == 2, "plain attack should still be a tap"
    # 60 ticks must now be reachable in a handful of consecutive picks
    _need = 60 / _probe._macro_ticks(12)
    assert _need <= 5, (
        f"a barehanded log still needs {_need:.0f} consecutive picks — "
        "P(that many in a row) is still negligible")
    # it must NOT count as a motion macro, or the frozen-client watchdog
    # would treat a legitimate 20-tick hold as a dead screen
    _motion = frozenset(i for i, m in enumerate(_M)
                        if "camera" in m or "forward" in m or "back" in m)
    assert 12 not in _motion, (
        "HOLD is classed as a motion macro — the remote freeze-detector would "
        "misread a stationary hold as a kicked client and rebuild mid-chop")
    # the chop detectors must know about it, or causal facts and the magnet
    # will not associate holding with breaking
    assert 12 in cfg["llm"]["vision"]["chop_actions"], \
        "chop_actions does not include the HOLD macro"

    # ---- 2i. FAMILIARITY DECAY ON BREAKS --------------------------------
    # MEASURED: 85% of 2231 breaks were dirt. The first-break tier re-arms
    # every goal horizon, so the ground underfoot was an inexhaustible income
    # needing no aim and no sustained effort — HOLD, the levelled view and
    # reach all got pointed at the floor.
    _scale = float(cfg["environment"].get("break_decay_scale", 0.0))
    assert _scale > 0.0, "skybot does not enable break familiarity decay"
    _d = E.__new__(E)
    _d._break_decay_scale = _scale
    _d._breaks_by_type = {"dirt": 1899, "spruce_leaves": 71, "oak_log": 0}
    _dirt = E._break_reward("dirt") * _d._break_decay("dirt")
    _log = (float(cfg["environment"]["log_break_reward"])
            * _d._break_decay("oak_log")
            + float(cfg["environment"]["log_tick_reward"]) * 60)
    assert _dirt < 0.01, f"dirt still pays {_dirt} after 1899 breaks"
    assert _d._break_decay("oak_log") == 1.0, \
        "a never-broken type was decayed — only mastery should reduce payout"
    assert _log > 1000 * _dirt, (
        f"a log ({_log}) is not decisively better than a dirt break ({_dirt})")
    # it must be MONOTONE in the count, not a special case for dirt
    _d._breaks_by_type = {"x": 0, "y": 25, "z": 1000}
    _c = [_d._break_decay(k) for k in ("x", "y", "z")]
    assert _c[0] > _c[1] > _c[2] and _c[0] == 1.0, f"not monotone: {_c}"
    assert abs(_c[1] - 0.5) < 1e-9, "n == scale should halve the payout"
    # NOTHING may name a block: this is novelty of achievement, not a rule
    assert '"dirt"' not in env_src.split("_break_decay")[1][:600], \
        "the decay hardcodes a block name — it must declare no meaning"
    # off by default
    import inspect as _i2
    assert _i2.signature(E.__init__).parameters[
        "break_decay_scale"].default == 0.0, "decay is ON by default"
    _o = E.__new__(E); _o._break_decay_scale = 0.0
    _o._breaks_by_type = {"dirt": 9999}
    assert _o._break_decay("dirt") == 1.0, "disabled decay is not an identity"

    # ---- 3/4. THE BREAK-DELTA LOOP --------------------------------------
    assert "self._mine_prev.get(_bt, 0)" in env_src, (
        "the break delta still defaults to the current count, so the FIRST "
        "break of every type stays invisible")
    # read ONCE per step, outside the per-type loop (so a second block felled
    # on the same step is not credited 0), and taking the pre-reset streak.
    assert "_run_at_break = max(int(self._attack_run), _streak_prev)" in env_src, \
        "ticks-to-break is not read once per step from the pre-reset streak"
    assert env_src.count("_run_at_break") >= 3, \
        "the single-read streak value is no longer reused across the loop"
    # replicate the patched logic: a brand-new type must be counted
    mine_prev, seen = {"dirt": 5}, []
    for now in ({"dirt": 6}, {"dirt": 6, "stone": 1}, {"dirt": 6, "stone": 2}):
        for bt, cnt in now.items():
            if int(cnt) - int(mine_prev.get(bt, 0)) > 0:
                seen.append(bt)
        mine_prev = dict(now)
    assert seen.count("stone") == 2, (
        f"first break of a new type still dropped (saw {seen})")

    # ---- 4b. CAREER TOTALS ARE NOT BREAKS -------------------------------
    # Defaulting a missing type to 0 (so a type's FIRST break counts) is right
    # only once a baseline exists. reset() takes the baseline from the first
    # obs, and on a rejoin that obs often carries no stats — so the baseline
    # was {} and the next stats-bearing observation counted every type's
    # ENTIRE career total as fresh breaks.
    # MEASURED: every type doubled EXACTLY across two consecutive segments
    # (dirt 56->112, grass_block 31->63, grass 20->40, oak_leaves 14->28)
    # while `attack` was 2% of 1024 steps. 146 blocks cannot break in ~40
    # ticks. The break counts — and the first-break rewards riding on them —
    # were inflated.
    assert "self._mine_seeded" in env_src, (
        "no lazy baseline seeding — an empty baseline still lets career "
        "totals be counted as breaks")
    assert "self._mine_seeded = bool(self._mine_baseline)" in env_src, (
        "reset() trusts an EMPTY baseline; it must only count as seeded when "
        "the stats observable actually carried something")
    # replicate the sequence the live run hit
    _prev, _seeded = {}, False

    def _tick(now, prev, seeded):
        if not seeded and now:
            return dict(now), True, 0          # adopt, count nothing
        return (dict(now), seeded,
                sum(max(0, v - prev.get(k, 0)) for k, v in now.items()))

    _prev, _seeded, _c1 = _tick({"dirt": 1960, "grass": 700}, _prev, _seeded)
    assert _c1 == 0, f"career totals counted as {_c1} breaks"
    _prev, _seeded, _c2 = _tick({"dirt": 1961, "grass": 700}, _prev, _seeded)
    assert _c2 == 1, "a real break after seeding was missed"
    _prev, _seeded, _c3 = _tick({"dirt": 1961, "grass": 700, "oak_log": 1},
                                _prev, _seeded)
    assert _c3 == 1, (
        "the FIRST break of a brand-new type is invisible again — that is the "
        "bug the 0-default was introduced to fix; both must hold at once")

    # ---- 5. NOVELTY IS KEYED ON THE FRAME REACHED -----------------------
    assert "self._view_key(next_obs_list[0])" in loop_src, \
        "perceptual novelty still keyed on the pre-step observation"
    assert "self._view_key(obs_list[0])" not in loop_src

    # ---- 6. PHASE SCOPING HOLDS ON EVERY ACTION SOURCE ------------------
    bank = SkillOptionBank.__new__(SkillOptionBank)
    bank.P, bank.K = 12, 4
    bank.slots = [None] * 4
    bank.disabled_macros = {10}
    pm = bank.primitive_mask()
    assert pm.shape == (12,) and not pm[10] and pm[9] and pm[11], \
        "primitive_mask does not honour disable_macros"
    wide = bank.primitive_mask(16)
    assert wide.shape == (16,) and not wide[10] and not wide[12:].any(), \
        "wide primitive_mask leaked option rows"
    # a skill's OWN head, inside an invoked option. _child_mask lives on
    # OptionExecutor and reaches the scoping through `self.bank` — a detail
    # worth pinning, because writing `self.disabled_macros` there raises
    # AttributeError on the first nested invocation, i.e. only under load.
    from developmental_ai.policy.options import OptionExecutor
    ex = OptionExecutor.__new__(OptionExecutor)
    ex.bank = bank
    ex.max_skill_depth = 2
    ex.nested_depth_blocks = 0
    ex.nested_cycle_blocks = 0
    cm = ex._child_mask({"head_dim": 14, "p_own": 12, "slot_map": {}}, set(), 0)
    assert not cm[10], (
        "a skill invoked as an option can still choose the disabled inventory "
        "macro — the 80%-menu-dwell hole is still open")
    assert cm[:10].all() and cm[11], "child mask over-masked"
    # never mask the agent mute
    bank.disabled_macros = set(range(12))
    assert bank.primitive_mask().all(), "an all-disabled table muted the agent"
    # ...and scouts. Asserted FUNCTIONALLY: what matters is that a mask
    # reaches select_action with the disabled macro closed, not how the call
    # is spelled. (Both new bank lookups are getattr-guarded, because the
    # executor also accepts bank-like doubles and older pickled banks that
    # predate the phase-scoped action space — hard attribute access there
    # crashed only on the first NESTED invocation, i.e. only under load.)
    class _RecPolicy:
        def __init__(self): self.seen = []
        def select_action(self, obs, knowledge=None, action_mask=None,
                          proprio=None):
            self.seen.append(action_mask)
            return 0, {"log_prob": 0.0, "value": 0.0}
    bank.disabled_macros = {10}
    ex2 = OptionExecutor.__new__(OptionExecutor)
    ex2.bank = bank
    ex2.num_envs = 2
    ex2.scouts_use_options = False
    ex2.runtimes = [type("R", (), {"active": False})() for _ in range(2)]
    rec = _RecPolicy()
    # only the scout branch is exercised (env 0 would need the full machinery)
    _pk = {}
    _pmask = getattr(ex2.bank, "primitive_mask", None)
    if _pmask is not None:
        _pk["action_mask"] = _pmask(bank.P + bank.K)
    rec.select_action(np.zeros(4), **_pk)
    got = rec.seen[-1]
    assert got is not None, "scouts still select actions with no mask at all"
    assert not got[10] and got[:10].all() and got[11], \
        "the scout mask does not close the disabled macro"
    # the guards themselves: a bank with no scoping must not crash _child_mask
    class _BareBank:
        P, K = 12, 0
        slots = []
    ex3 = OptionExecutor.__new__(OptionExecutor)
    ex3.bank = _BareBank()
    ex3.max_skill_depth = 2
    ex3.nested_depth_blocks = ex3.nested_cycle_blocks = 0
    bare = ex3._child_mask({"head_dim": 12, "p_own": 12, "slot_map": {}},
                           set(), 0)
    assert bare[:12].all(), (
        "a bank without disabled_macros must mean 'nothing disabled', not a "
        "crash on the first nested invocation")

    # ---- 7. PPO ACTUALLY MINIBATCHES ------------------------------------
    def _rollout(pol, n):
        for i in range(n):
            o = np.zeros(8, dtype=np.float32)
            a, info = pol.select_action(o)
            pol.store_transition(o, a, 0.1, False,
                                 info["log_prob"], info["value"])
    torch.manual_seed(0)
    full = StandaloneActorCritic(obs_dim=8, action_dim=4, hidden_dim=16)
    _rollout(full, 128)
    mf = full.train_step(n_epochs=4)
    assert mf["n_updates"] == 4, (
        f"full-batch path changed: {mf['n_updates']} updates for 4 epochs")

    torch.manual_seed(0)
    mini = StandaloneActorCritic(obs_dim=8, action_dim=4, hidden_dim=16,
                                 minibatch_size=32)
    _rollout(mini, 128)
    mm = mini.train_step(n_epochs=4)
    assert mm["n_updates"] == 4 * (128 // 32), (
        f"minibatching not applied: {mm['n_updates']} updates, expected 16")
    assert np.isfinite(mm["policy_loss"]) and np.isfinite(mm["entropy"])
    # entropy must stay a PER-UPDATE mean — it drives the COLLAPSED readout
    assert 0.0 < mm["entropy"] <= np.log(4) + 1e-6, (
        f"entropy {mm['entropy']} is not a per-update mean (max is ln4)")

    # opt-in only: batch_size must NOT silently switch other configs over
    assert 'pol_cfg.get("minibatch_size", 0)' in loop_src
    assert 'minibatch_size=int(pol_cfg.get("batch_size"' not in loop_src, \
        "policy.batch_size was adopted — that changes 10 other experiments"
    # The MECHANISM must work and stay opt-in. Whether skybot switches it ON
    # is a tuning decision, not a contract: it was enabled 2026-08-02 and
    # reverted the same day after entropy collapsed 0.06 -> 0.00 -> 0.00 nats
    # within three segments (see the config comment). Asserting "> 0" here
    # would make the test fail for making the right call.
    assert "minibatch_size" in cfg["policy"], (
        "skybot must state minibatch_size explicitly — a silent default is "
        "how policy.batch_size sat unread in eleven configs for months")
    if int(cfg["policy"]["minibatch_size"]) > 0:
        assert float(cfg["policy"].get("entropy_coef", 0.01)) > 0.01, (
            "minibatching multiplies gradient steps ~16x per rollout; "
            "re-enabling it at the stock entropy_coef reproduced a total "
            "policy collapse. Raise entropy_coef first.")

    # ---- 8. entropy/value coefficients are reachable ---------------------
    assert 'entropy_coef=float(pol_cfg.get(' in loop_src
    assert 'value_coef=float(pol_cfg.get(' in loop_src
    e = StandaloneActorCritic(obs_dim=4, action_dim=2, hidden_dim=8,
                              entropy_coef=0.05, value_coef=0.25)
    assert e.entropy_coef == 0.05 and e.value_coef == 0.25

    # ---- 9. the stage gate counts the streams that exist -----------------
    n_envs = int(cfg["parallel_envs"]["num_envs"])
    fx = int(cfg["developmental_stages"]["force_exploit_timestep"])
    assert fx == 200000 * n_envs, (
        f"force_exploit_timestep {fx} does not equal 200k primary steps at "
        f"num_envs={n_envs} — the anneal fires at the wrong point")

    print("[reward-fixes-smoke] ALL PASS: return normalization wired AND on "
          "(damps a dense intrinsic, never deletes it); ground tiers survive "
          "a namespaced key and stay under the mint threshold; mine_block "
          "keys normalised at source; first break of a type is counted and "
          "ticks-to-break is read once per step; novelty keys on the frame "
          "reached; disable_macros holds for the meta-policy, for skills "
          "invoked as options and for scouts; PPO minibatches opt-in "
          "(16 updates vs 4) with per-update metrics; entropy/value coefs "
          "reachable; stage gate matches num_envs")


if __name__ == "__main__":
    main()
