"""GUI-occlusion farm smoke (2026-08-17).

THE LIVE FAILURE THIS ENCODES
    SkyBot right-clicked a WANDERING VILLAGER, opened its trade screen, and
    sat inside it for 10,149+ CONSECUTIVE steps: position frozen, gui 66% of
    the segment, `magnet_seek` paying 77% of ALL income.

    Two independent faults, both of which had to be fixed:

    1. THE LEAK. The gui_open guard zeroes `intrinsic` — but the magnet's
       shaping is added to prim_EXTRINSIC (`prim_extrinsic = rewards[0] +
       _sr`), which that guard never touched. So the single largest income
       source kept paying through an occluded camera. This is the
       2026-08-02 occlusion farm repeating one channel over.

    2. NO WAY OUT, AND THE EXIT TRAINED AWAY. Zeroing income gives no
       gradient toward the door. Worse: pressing `inventory` OPENS a screen
       that pays 0, so the policy drove that action to zero probability —
       measured 0 presses across an entire run at 92%-of-max entropy. When
       a villager opened a GUI via `use`, the agent had already unlearned
       the only button that closes one. Guard-becomes-latch.

    3. AND THE FIX ITSELF WAS WRONG FIRST TIME. A dwell COST written as the
       usual `gamma*Phi' - Phi` pays `w*(1-gamma)` on every step Phi is
       PINNED — turning "sitting here costs" into "sitting here pays". The
       shipped gaze-level term had the same flaw and the log showed it:
       "Gaze level: +0.00014/step", a wage for staying at the pitch clamp.

Contracts:
    A. A state-cost potential must net ~0 over enter/dwell/exit AND pay
       exactly 0 while pinned. (gamma<1 fails this; plain difference passes.)
    B. Magnet shaping is not added to prim_extrinsic while gui_open, and the
       seek/centring potentials re-adopt so closing pays no windfall.
    C. The dwell cost lands in the EXTRINSIC channel — the intrinsic one is
       zeroed inside a GUI, which would erase the cost entirely.
    D. THE TERM RUNS ON STREAM 0 WITHOUT THE VISION SCAFFOLD (2026-10-03,
       docs/foundation/BASELINE_AUDIT.md §3). The stream-0 dwell block sat
       INSIDE `if not use_dream_actor and self.vision_scaffold is not None:`
       in BOTH live bodies, and the live config runs `llm.vision.enabled:
       false` — so the one stream PPO trains on paid NO dwell cost (measured
       on the old code: every recorded stream-0 extrinsic 0.0 through a
       30-step menu), while every scout paid it via _scout_mixed_reward, and
       the yaml's "a menu now pays strictly <= 0" rested on a term that
       never ran. Drives the REAL _collect_segment (lifelong) and
       _run_episode_parallel (episodic) on a zero-reward fake env whose
       stream 0 opens a GUI for 30 steps, and records the extrinsic handed
       to the primary reward_mixer.mix call:
         D1 scaffold None: entry costs (-w/N per step, -w in total), pinned
            pays EXACTLY 0 per step, the close step refunds +w, the cycle
            nets 0, and the sequence equals the plain-difference reference.
         D2 scaffold present (stub, magnet paying 0): the IDENTICAL sequence
            — applied exactly once per step, never twice.
         D3 control: gui_dwell_weight 0 records all zeros, so D1 is
            measuring the dwell term and nothing else in the channel.
       Needs gymnasium (legacy tier); A-C run without it.
    E. The episodic body re-adopts the cost potentials at episode start.
    F. THE FARM DAMP REACHES STREAM-0 INTRINSIC (2026-10-04): with the loop
       detector reporting damp 0.5, the intrinsic handed to the primary mix
       is exactly 0.5x the undamped run in BOTH real bodies; damp 1.0 is
       byte-identical; a GUI step still pays exactly 0. See the test.
    G. A GENUINE PER-STEP DWELL COST after a grace period (2026-10-05): the
       potential pays 0 while pinned, so it cannot discourage dwelling, and
       the menu became a refuge (GUI 19% -> 70%). G drives both real bodies
       and the scout path; see the test for G1-G6.

Run: PYTHONPATH=. python tests/_gui_farm_smoke.py
"""
import os
import re
import sys

sys.path.insert(0, ".")

SRC = os.path.join("developmental_ai", "core", "developmental_loop.py")


def _cycle(weight, gamma, steps, floor_n=200.0):
    """enter -> dwell `steps` -> exit, and the per-step pay while pinned."""
    def phi(run):
        return -min(1.0, run / floor_n)
    total, prev = 0.0, phi(0)
    for run in list(range(1, steps + 1)) + [0]:
        cur = phi(run)
        total += weight * (gamma * cur - prev)
        prev = cur
    pinned = weight * (gamma * -1.0 - -1.0)
    return total, pinned


def test_state_cost_must_not_pay_while_pinned():
    # the WRONG form, kept as a regression witness
    tot_bad, pinned_bad = _cycle(0.05, 0.99, 600)
    assert pinned_bad > 0.0, "the flawed form should pay while pinned"
    assert tot_bad > 0.2, tot_bad
    print(f"  A1. gamma=0.99 form: cycle {tot_bad:+.4f}, pinned "
          f"{pinned_bad:+.6f}/step -> a WAGE for dwelling (rejected)")

    tot_ok, pinned_ok = _cycle(0.05, 1.0, 600)
    assert abs(tot_ok) < 1e-9, tot_ok
    assert abs(pinned_ok) < 1e-12, pinned_ok
    print(f"  A2. plain difference: cycle {tot_ok:+.6f}, pinned "
          f"{pinned_ok:+.6f}/step -> costs on entry, refunds on exit, "
          f"flat while stuck")

    # entering genuinely costs, leaving genuinely refunds
    def phi(r):
        return -min(1.0, r / 200.0)
    enter = 0.05 * (phi(200) - phi(0))
    leave = 0.05 * (phi(0) - phi(200))
    assert enter < 0 < leave and abs(enter + leave) < 1e-12
    print(f"  A3. entry {enter:+.4f} / exit {leave:+.4f} — equal and "
          f"opposite, so open/close cannot be farmed either way")


def test_source_contracts():
    src = open(SRC).read()
    # the leak: magnet shaping must be gui-gated
    assert 'if bool((step_infos[0] or {}).get("gui_open")):' in src
    assert "self.vision_scaffold._seek_prob_prev = None" in src, (
        "seek potential must re-adopt in a GUI or closing pays a windfall")
    # both duplicated bodies must carry it (this codebase's recurring bug)
    assert src.count("_gui_now2 = bool(") == 2, (
        "the gui gate must exist in BOTH stepping bodies")
    assert src.count("prim_extrinsic += self._gui_reward(0, step_infos[0])") == 2
    print("  B1. magnet shaping gui-gated in BOTH bodies; seek re-adopts")

    # the dwell cost must use the plain difference, in BOTH bodies
    from developmental_ai.core.reward_components import gui_costs
    assert gui_costs(True, 200, -1.0, weight=0.05).total == 0.0
    assert "_gg * _gphi" not in src, (
        "gui dwell reverted to the gamma form — it would pay to dwell")
    assert "_gp * _php" not in src, (
        "gaze level reverted to the gamma form — it paid +0.00014/step to "
        "sit at the clamp")
    print("  B2. both state-cost potentials use the plain difference")

    # the cost must go to the EXTRINSIC channel
    i_cost = src.index("prim_extrinsic += self._gui_reward(0, step_infos[0])")
    i_zero = src.index("intrinsic[0] = intrinsic[0] * 0.0")
    assert i_cost < i_zero, "cost must not be placed after/into the zeroing"
    assert "intrinsic[0] = intrinsic[0] + _gdw" not in src, (
        "a dwell cost in the intrinsic channel is erased by the gui guard")
    print("  B3. dwell cost paid into prim_extrinsic (intrinsic is zeroed "
          "inside a GUI, which would erase it)")

    cfg = open(os.path.join("configs", "minecraft_skybot.yaml")).read()
    for k in ("gui_dwell_weight", "gui_dwell_steps"):
        assert k in cfg, f"skybot config lost `{k}`"
    print("  B4. config keys present")


def test_no_one_way_doors():
    """An action that OPENS a screen may not be offered without the one
    that CLOSES it.

    This is the trap that produced the villager incident: `disable_macros:
    [10]` removed `inventory` — the only close — while `use` (11), which
    opens a villager/chest screen on right-click, stayed available. The
    agent then had no exit in its action space at all and sat in the trade
    menu for 10,149+ consecutive steps. Disabling 10 was defensible when
    the menu PAID (a reward leak, since fixed); it was never defensible
    while an entrance remained open.
    """
    import yaml
    cfg = yaml.safe_load(open(os.path.join("configs",
                                           "minecraft_skybot.yaml")))
    dis = set(int(x) for x in
              (cfg.get("environment", {}).get("disable_macros") or []))
    OPENS, CLOSES = 11, 10          # use (right-click) / inventory (toggle)
    if OPENS not in dis:
        assert CLOSES not in dis, (
            f"macro {OPENS} (`use`, opens villager/chest screens) is offered "
            f"while macro {CLOSES} (`inventory`, the ONLY close) is disabled "
            f"— that is a one-way door; disable both or neither")
    print(f"  C1. no one-way door: disable_macros={sorted(dis) or '[]'} — "
          f"an entrance is never offered without its exit")


# ---- D. drive the REAL collection bodies -----------------------------------
GUI_OPEN_AT, GUI_LEN = 5, 30          # stream 0: closed, 30 steps open, closed
DWELL_W, DWELL_N = 0.05, 10.0         # saturates after 10 -> 20 pinned steps


def _gui_env(idx):
    """Zero-reward, never-terminating (truncates at 70) 4-d env. Stream 0 reports
    info["gui_open"] on steps [GUI_OPEN_AT, GUI_OPEN_AT + GUI_LEN); scouts
    never do. Every step's flag is logged so the recorded rewards can be
    aligned against what the env actually said."""
    import gymnasium as gym
    import numpy as np

    class GuiEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        action_space = gym.spaces.Discrete(2)

        def __init__(self):
            self.idx, self.t = idx, 0
            self.rng = np.random.default_rng(idx)
            GUI_LOG[idx] = []

        def _obs(self):
            return self.rng.uniform(-1, 1, 4).astype(np.float32)

        def reset(self, seed=None, options=None):
            return self._obs(), {"gui_open": False}

        def step(self, a):
            g = (self.idx in GUI_STREAMS
                 and any(a <= self.t < b for a, b in GUI_WINDOWS))
            self.t += 1
            GUI_LOG[self.idx].append(bool(g))
            # truncates at 70 so the EPISODIC body ends; the lifelong
            # segment (60 steps) never reaches it, so no reset intervenes
            return (self._obs(), 0.0, False, self.t >= 70,
                    {"gui_open": bool(g)})

    return GuiEnv()


GUI_LOG = {}
# Which streams open a GUI, and over which step windows. The defaults are the
# D/F fixture (stream 0 only, one 30-step stay); contract G widens them.
GUI_STREAMS = {0}
GUI_WINDOWS = [(GUI_OPEN_AT, GUI_OPEN_AT + GUI_LEN)]


INTRINSIC = {}   # the last _drive's stream-0 intrinsic handed to the mix
SCOUT = {}       # the last _drive's scout (stream 1) extrinsic handed to mix


def _drive(lifelong, weight, scaffold, damp=None, step_cost=None,
           grace=None):
    """Run one real body; return (stream-0 gui flags, stream-0 extrinsic
    handed to the PRIMARY reward_mixer.mix) step-aligned. The matching
    INTRINSIC argument of that same call lands in INTRINSIC["rec"].
    damp: None leaves the infra stack alone; a float makes its loop
    detector report that damp on every step (the REAL on_step still runs
    first, so everything else it publishes is unchanged)."""
    import tempfile
    import yaml
    import developmental_ai.core.developmental_loop as dl
    from developmental_ai.environments.wrappers import DevelopmentalEnvWrapper

    tmp = tempfile.mkdtemp(prefix="gui_farm_")
    state = {"n": 0}

    def fake(*a, **k):
        i = state["n"]
        state["n"] += 1
        return DevelopmentalEnvWrapper(_gui_env(i), normalize_obs=False), None
    _orig = dl.make_env
    dl.make_env = fake
    GUI_LOG.clear()
    cfg = yaml.safe_load(open(os.path.join("configs", "default.yaml")))
    cfg["seed"] = 0
    cfg["world_model"].update({
        "stochastic_size": 8, "stochastic_classes": 8,
        "deterministic_size": 32, "encoder_hidden": 32,
        "decoder_hidden": 32, "batch_size": 4, "sequence_length": 8,
        "train_iters": 1, "buffer_capacity": 2000})
    cfg["parallel_envs"] = {"enabled": True, "num_envs": 2}
    cfg["policy"].update({"n_steps": 200, "minibatch_size": 16,
                          "n_epochs": 1})
    cfg["environment"]["max_episode_steps"] = 70
    if lifelong:
        cfg["lifelong"] = {"enabled": True, "forever": False,
                           "segment_len": 60, "wm_train_every": 10 ** 6,
                           "goal_horizon": 10 ** 9, "state_decay": 1.0,
                           "reset_on_death_only": True,
                           "stop_file": os.path.join(tmp, "STOP")}
    cfg.setdefault("loop", {})["verbose"] = 0
    cfg.setdefault("llm", {})["enabled"] = False
    cfg.setdefault("skill_bank", {})["storage_dir"] = os.path.join(tmp, "sb")
    ai = dl.DevelopmentalAI(config=cfg)
    try:
        assert ai.vision_scaffold is None, "fixture must start scaffold-less"
        ai._gui_dwell_weight = float(weight)
        ai._gui_dwell_steps = DWELL_N
        if step_cost is not None:      # None: leave the loop's own default
            ai._gui_dwell_step_cost = float(step_cost)
        if grace is not None:
            ai._gui_dwell_grace_steps = int(grace)
        if scaffold:
            ai.vision_scaffold = _StubScaffold()
            ai._magnet_step_shaping = lambda *a, **k: 0.0
        if damp is not None:
            assert ai.infra is not None, "fixture needs the infra stack"
            _on_step = ai.infra.on_step

            def damped_on_step(*a, **k):
                _on_step(*a, **k)
                ai.infra.last_loop_damp = float(damp)
            ai.infra.on_step = damped_on_step
        rec, irec, srec = [], [], []
        INTRINSIC["rec"] = irec
        SCOUT["rec"] = srec
        _mix = ai.reward_mixer.mix

        def spy(i, e, update_stats=True):
            if update_stats:              # the PRIMARY call; scouts pass False
                rec.append(float(e))
                irec.append(float(i))
            else:
                srec.append(float(e))
            return _mix(i, e, update_stats=update_stats)
        ai.reward_mixer.mix = spy
        if lifelong:
            ai._collect_segment()
        else:
            ai._run_episode_parallel(use_dream_actor=False)
        if scaffold:
            ai.vision_scaffold = None     # the stub has nothing to close
        flags = list(GUI_LOG[0])
        assert len(flags) == len(rec) >= GUI_OPEN_AT + GUI_LEN + 3, (
            len(flags), len(rec))
        return flags, rec
    finally:
        dl.make_env = _orig
        ai.close()


class _StubScaffold:
    """Enough of VisionScaffold for the scaffold block to RUN (its magnet
    call is patched to pay 0) — so D2 isolates 'is the dwell term applied
    once or twice', not magnet arithmetic."""
    stats = {}
    target_categories = []
    _seek_cats = []
    cold_start_weight = 0.0
    seek_nudge_budget = 0
    _phi_prev = None
    _seek_prob_prev = None

    def reset(self):
        pass

    def wants_step(self, *_a):
        return False

    def social_prime(self, *_a, **_k):
        return False

    def propose_category(self, *_a, **_k):
        return None

    def set_deficit_source(self, *_a, **_k):
        pass


def _reference(flags, w, n):
    """The plain-difference potential, independently: Phi=-min(1, run/n),
    pay w*(Phi' - Phi), first sample charges nothing."""
    out, run, prev = [], 0, None
    for g in flags:
        run = run + 1 if g else 0
        phi = -min(1.0, run / n)
        out.append(0.0 if prev is None else w * (phi - prev))
        prev = phi
    return out


def test_stream0_dwell_without_scaffold():
    import importlib.util
    if importlib.util.find_spec("gymnasium") is None:
        raise ImportError("No module named 'gymnasium'")   # runner: SKIP
    for lifelong, label in ((True, "_collect_segment"),
                            (False, "_run_episode_parallel")):
        flags, pay = _drive(lifelong, DWELL_W, scaffold=False)
        o = flags.index(True)
        c = o + flags[o:].index(False)
        assert c - o == GUI_LEN, (o, c)
        n = int(DWELL_N)
        entry = pay[o:o + n]
        pinned = pay[o + n:c]
        exit_ = pay[c]
        assert all(abs(x + DWELL_W / DWELL_N) < 1e-9 for x in entry), (
            f"{label}: stream 0 paid no dwell cost on entry with the vision "
            f"scaffold OFF (the live config) — entry steps {entry}")
        assert pinned and all(x == 0.0 for x in pinned), (
            f"{label}: pinned in a menu must pay EXACTLY 0/step: {pinned}")
        assert abs(exit_ - DWELL_W) < 1e-9, (label, exit_)
        assert abs(sum(pay)) < 1e-9, (label, sum(pay))
        assert all(x == 0.0 for k, x in enumerate(pay)
                   if k < o or k > c), (label, pay)
        ref = _reference(flags, DWELL_W, DWELL_N)
        assert max(abs(a - b) for a, b in zip(pay, ref)) < 1e-9, label
        print(f"  D1. {label}, scaffold None: entry {sum(entry):+.4f} "
              f"({len(entry)} x {entry[0]:+.4f}), pinned {len(pinned)} "
              f"steps at exactly 0.0, exit {exit_:+.4f}, cycle "
              f"{sum(pay):+.2e}")

        flags2, pay2 = _drive(lifelong, DWELL_W, scaffold=True)
        assert flags2 == flags
        assert max(abs(a - b) for a, b in zip(pay2, pay)) < 1e-12, (
            f"{label}: with the scaffold present the dwell term must be "
            f"applied exactly ONCE per step (double-charge or missing): "
            f"{pay2[o:o + 3]} vs {pay[o:o + 3]}")
        print(f"  D2. {label}, scaffold present: identical sequence — "
              f"applied once per step, entry {sum(pay2[o:o + n]):+.4f}")

        flags0, pay0 = _drive(lifelong, 0.0, scaffold=False)
        assert all(x == 0.0 for x in pay0), (label, pay0)
        print(f"  D3. {label}, weight 0: all {len(pay0)} stream-0 "
              f"extrinsics 0.0 — D1 measures the dwell term alone")



def test_farm_damp_reaches_intrinsic():
    """F. THE FARM DAMP MUST REACH THE CHANNEL THAT PAYS (2026-10-04).
    `last_loop_damp` (infra/ledger.py loop_damp) multiplied only `_sr`, the
    vision magnet's shaping — and the live config runs llm.vision.enabled:
    false, so `_sr` is always 0 there and a revisit loop's INTRINSIC income
    (the curiosity a farm actually farms) was never damped: the response to
    a detected farm was a multiply on a dead channel. Drives BOTH real
    bodies with the loop detector reporting a damp of 0.5:
      F1 stream-0 intrinsic handed to the primary mix is EXACTLY 0.5x the
         undamped run on every open-world step (and nonzero somewhere, so
         the check is not vacuous) — fails on the pre-fix code, where the
         two runs are identical;
      F2 damp 1.0 leaves the whole sequence byte-identical to a run whose
         detector is untouched (the fix costs nothing when no farm exists);
      F3 inside a GUI the intrinsic is still EXACTLY 0 (the occlusion
         zeroing runs after the damp, so the damp cannot resurrect it).
    The damp is not a latch: it is read fresh every step from loop_damp(),
    which returns 1.0 again once the agent leaves the cell for more than
    revisit_horizon steps (contract 3 of _farm_damping_smoke)."""
    import importlib.util
    if importlib.util.find_spec("gymnasium") is None:
        raise ImportError("No module named 'gymnasium'")   # runner: SKIP
    for lifelong, label in ((True, "_collect_segment"),
                            (False, "_run_episode_parallel")):
        flags, _ = _drive(lifelong, 0.0, scaffold=False)
        base = list(INTRINSIC["rec"])
        _, _ = _drive(lifelong, 0.0, scaffold=False, damp=1.0)
        one = list(INTRINSIC["rec"])
        f2, _ = _drive(lifelong, 0.0, scaffold=False, damp=0.5)
        half = list(INTRINSIC["rec"])
        assert f2 == flags and len(base) == len(half) == len(one)
        assert one == base, (
            f"{label}: damp 1.0 must leave intrinsic byte-identical")
        world = [k for k, g in enumerate(flags) if not g]
        assert any(base[k] != 0.0 for k in world), (
            f"{label}: undamped intrinsic all zero — F1 would be vacuous")
        bad = [(k, base[k], half[k]) for k in world
               if half[k] != 0.5 * base[k]]
        assert not bad, (
            f"{label}: loop damp 0.5 did not scale stream-0 INTRINSIC "
            f"(first mismatches {bad[:3]}) — the damp is on a dead channel")
        gui = [half[k] for k, g in enumerate(flags) if g]
        assert gui and all(x == 0.0 for x in gui), (label, gui)
        print(f"  F. {label}: loop damp 0.5 -> intrinsic exactly 0.5x on "
              f"{len(world)} open-world steps; damp 1.0 byte-identical; "
              f"{len(gui)} GUI steps still exactly 0.0")


STEP_COST, GRACE = 0.002, 10      # 30-step stay -> 20 charged steps


def _expected_step_cost(flags, cost, grace):
    """The rule, independently: run = consecutive GUI steps so far; a GUI
    step whose run exceeds `grace` costs `cost`; anything else costs 0."""
    out, run = [], 0
    for g in flags:
        run = run + 1 if g else 0
        out.append(-cost if (g and run > grace) else 0.0)
    return out


def test_gui_dwell_step_cost():
    """G. A GENUINE PER-STEP GUI DWELL COST AFTER A GRACE PERIOD (2026-10-05).

    THE LIVE INCIDENT. 11.4 h of shadow data after the LP fix: GUI-open
    share rose 19% -> 70% on stream 0 and 50% on stream 1, the longest stay
    ~1040 steps, and action 10 (toggle inventory) became the TOP action,
    chosen mostly by the shared policy. Pay per sampled step: moving outside
    -0.0057 intrinsic, standing still outside +0.0004, inside a GUI ~0. The
    menu became a REFUGE from a world that, on average, cost money.

    WHY THE EXISTING TERM COULD NOT STOP IT. `gui_dwell_weight` is a
    plain-difference potential (correctly — CLAUDE.md 4.3): it charges once
    on entry and refunds on exit, so it pays EXACTLY 0 per step while
    pinned. A telescoping potential structurally cannot discourage
    dwelling; only a genuine per-step cost can. Hence `gui_dwell_step_cost`
    charged on every GUI step past `gui_dwell_grace_steps` — into
    prim_EXTRINSIC (the intrinsic channel is zeroed in a GUI, 4.4), with no
    refund and no potential, so nothing about it can be farmed. Closing the
    GUI (action 10, always offered — contract C) ends it on the next step.

    Drives the REAL bodies (vision scaffold None, the live config) exactly
    as contract D does, isolating the new term as the difference between a
    run with the cost and the identical run without it:
      G1 a stay of grace+N steps costs exactly N*cost beyond the potential,
         on precisely the GUI steps whose run exceeds the grace, in BOTH
         bodies, the same per-step numbers in each; and with the potential
         OFF (weight 0) the cost still runs (it must not depend on the
         potential's run counter being ticked).
      G2 a stay <= grace (grace = stay length, and grace > stay) costs
         nothing extra.
      G3 closing stops the cost on the very next step, and the run counter
         RESETS: a second, shorter stay restarts the grace (a counter that
         did not reset would charge its whole length).
      G4 cost 0 is byte-identical to a run with the loop's defaults and to
         the plain-difference reference — contract D's numbers unchanged.
      G5 scouts (_scout_mixed_reward, per-stream gui_run) follow the same
         rule on their own stay.
      G6 source: since the refactor into reward_components.gui_costs /
         `_gui_reward` (2026-10-05) the cost is computed in ONE place and
         called from exactly THREE sites — once in each live body (each
         before the intrinsic-zeroing line) and once in the scout reward.
         Inside `_gui_reward`, gui_costs is called once and the potential
         is assigned once per holder (stream 0 / scout state); everywhere
         else in the loop the stream-0 potential is only ever RESET to None
         (boundary re-adoption), never computed — a second computing site
         is exactly the duplicated-body drift (CLAUDE.md 4.2) this guards.
    Fails on the pre-change code: the attributes are ignored, so the
    with-cost and without-cost runs are identical and G1 sees 0, not N*cost.
    """
    import importlib.util
    global GUI_STREAMS, GUI_WINDOWS
    if importlib.util.find_spec("gymnasium") is None:
        raise ImportError("No module named 'gymnasium'")   # runner: SKIP
    w0 = (GUI_STREAMS, GUI_WINDOWS)
    try:
        per_body = {}
        for lifelong, label in ((True, "_collect_segment"),
                                (False, "_run_episode_parallel")):
            flags, base = _drive(lifelong, DWELL_W, False,
                                 step_cost=0.0, grace=GRACE)
            f1, paid = _drive(lifelong, DWELL_W, False,
                              step_cost=STEP_COST, grace=GRACE)
            assert f1 == flags
            diff = [b - a for a, b in zip(base, paid)]
            want = _expected_step_cost(flags, STEP_COST, GRACE)
            n_charged = GUI_LEN - GRACE
            assert max(abs(d - x) for d, x in zip(diff, want)) < 1e-12, (
                f"{label}: per-step dwell cost does not follow the rule "
                f"(got {sum(diff):+.6f}, want {sum(want):+.6f}) — the term "
                f"is missing from this body or charges the wrong steps")
            assert abs(sum(diff) + n_charged * STEP_COST) < 1e-12, (
                label, sum(diff))
            o = flags.index(True)
            per_body[label] = diff[o:o + GUI_LEN + 3]
            print(f"  G1. {label}: {GUI_LEN}-step stay, grace {GRACE} -> "
                  f"{sum(diff):+.4f} beyond the potential "
                  f"({n_charged} x {-STEP_COST:+.4f})")

            fz, pz = _drive(lifelong, 0.0, False, step_cost=0.0,
                            grace=GRACE)
            fc, pc = _drive(lifelong, 0.0, False, step_cost=STEP_COST,
                            grace=GRACE)
            assert all(x == 0.0 for x in pz), (label, pz)
            assert max(abs(a - b) for a, b in zip(pc, want)) < 1e-12, (
                f"{label}: with the potential OFF the step cost must still "
                f"run (it may not borrow a counter nobody ticks): "
                f"{sum(pc):+.6f}")
            print(f"  G1. {label}: potential off (weight 0) -> step cost "
                  f"alone {sum(pc):+.4f}, same rule")

            for g in (GUI_LEN, GUI_LEN + 10):
                _, pg = _drive(lifelong, DWELL_W, False,
                               step_cost=STEP_COST, grace=g)
                assert pg == base, (
                    f"{label}: stay {GUI_LEN} <= grace {g} must cost nothing "
                    f"extra: {sum(pg) - sum(base):+.6f}")
            print(f"  G2. {label}: stay {GUI_LEN} with grace {GUI_LEN} and "
                  f"{GUI_LEN + 10} -> exactly the potential, nothing extra")

            # G3: two stays; the second (15 steps) must restart the grace
            GUI_WINDOWS = [(5, 35), (37, 52)]
            f3, b3 = _drive(lifelong, DWELL_W, False, step_cost=0.0,
                            grace=GRACE)
            _, p3 = _drive(lifelong, DWELL_W, False, step_cost=STEP_COST,
                           grace=GRACE)
            d3 = [b - a for a, b in zip(b3, p3)]
            w3 = _expected_step_cost(f3, STEP_COST, GRACE)
            assert max(abs(a - b) for a, b in zip(d3, w3)) < 1e-12, (
                f"{label}: two stays mis-charged ({sum(d3):+.6f} vs "
                f"{sum(w3):+.6f}) — the run counter did not reset on close")
            closes = [k for k in range(1, len(f3)) if f3[k - 1] and not f3[k]]
            assert len(closes) == 2 and all(d3[k] == 0.0 for k in closes)
            assert all(d3[k] == 0.0 for k, g in enumerate(f3) if not g)
            second = sum(d3[37:])
            assert abs(second + 5 * STEP_COST) < 1e-12, (label, second)
            GUI_WINDOWS = w0[1]
            print(f"  G3. {label}: close step and every open-world step pay "
                  f"0 extra; second 15-step stay {second:+.4f} (5 charged — "
                  f"the counter reset, grace restarted)")

            # G4: cost 0 == the loop's defaults == contract D reference
            fd, pd = _drive(lifelong, DWELL_W, False)
            assert fd == flags and pd == base, (
                f"{label}: cost 0 is not byte-identical to the default run")
            ref = _reference(flags, DWELL_W, DWELL_N)
            assert max(abs(a - b) for a, b in zip(base, ref)) < 1e-9
            print(f"  G4. {label}: cost 0 byte-identical to the default run "
                  f"and to contract D's plain-difference reference")
        a, b = per_body.values()
        assert a == b, "the two bodies charge different numbers"
        print("  G1. both bodies: identical per-step cost sequence")

        # G5: scouts. Only the lifelong body trains PPO on scouts.
        GUI_STREAMS = {0, 1}
        _drive(True, DWELL_W, False, step_cost=0.0, grace=GRACE)
        s0 = list(SCOUT["rec"])
        _drive(True, DWELL_W, False, step_cost=STEP_COST, grace=GRACE)
        s1 = list(SCOUT["rec"])
        sflags = list(GUI_LOG[1])
        assert len(s0) == len(s1) == len(sflags) and any(sflags)
        sd = [b - a for a, b in zip(s0, s1)]
        sw = _expected_step_cost(sflags, STEP_COST, GRACE)
        assert max(abs(x - y) for x, y in zip(sd, sw)) < 1e-12, (
            f"scout dwell step cost wrong: {sum(sd):+.6f} vs {sum(sw):+.6f}")
        assert abs(sum(sd) + (GUI_LEN - GRACE) * STEP_COST) < 1e-12
        print(f"  G5. scout (_scout_mixed_reward): same rule, "
              f"{sum(sd):+.4f} over its {GUI_LEN}-step stay")
    finally:
        GUI_STREAMS, GUI_WINDOWS = w0

    src = open(SRC).read()

    def _method(name):
        i0 = src.index(f"    def {name}(")
        return src[i0:src.index("\n    def ", i0 + 10)]

    key = "prim_extrinsic += self._gui_reward(0, step_infos[0])"
    assert src.count(key) == 2, (
        f"stream-0 step cost must exist in BOTH bodies: {src.count(key)}")
    assert src.count("    def _gui_reward(") == 1
    # exactly three call sites: two live bodies + the scout reward
    assert src.count("self._gui_reward(") == 3, src.count("self._gui_reward(")
    for name in ("_run_episode_parallel", "_collect_segment"):
        body = _method(name)
        assert body.count("self._gui_reward(") == 1, name
        assert body.index(key) < body.index(
            "intrinsic[0] = intrinsic[0] * 0.0"), name
    assert "_gui_reward(" not in _method("_run_episode"), (
        "the unwired legacy body must not grow a third copy")
    scout = _method("_scout_mixed_reward")
    assert scout.count("self._gui_reward(") == 1
    assert "_ext += self._gui_reward(e_i, info)" in scout
    # one computation, one assignment per holder, all inside _gui_reward
    gr = _method("_gui_reward")
    assert src.count("gui_costs(") == 1 and gr.count("gui_costs(") == 1, (
        "gui_costs must be called from _gui_reward only")
    s0 = "self._gui_run, self._gui_dwell_phi = costs.run, costs.potential"
    sc = 'state["gui_run"], state["gui"] = costs.run, costs.potential'
    assert gr.count(s0) == 1 and gr.count(sc) == 1, (
        "potential must be assigned exactly once per holder in _gui_reward")
    assert src.count(s0) == 1 and src.count(sc) == 1
    rhs = re.findall(r"self\._gui_dwell_phi\s*=(?!=)\s*([^\n]*)", src)
    computed = [r for r in rhs if r.split("#")[0].strip() != "None"]
    assert computed == ["costs.run, costs.potential"], (
        f"stream-0 potential written with a non-reset value outside "
        f"_gui_reward: {computed}")
    print(f"  G6. _gui_reward called at exactly 3 sites (one per live body, "
          f"each before the intrinsic zeroing; one scout on _ext; none in "
          f"_run_episode); gui_costs called once; potential assigned once per "
          f"holder inside _gui_reward, elsewhere only reset to None "
          f"({len(rhs) - 1} resets)")


def test_episodic_boundary_readopts_cost_potentials():
    """The episodic body must reset the stream-0 cost potentials at each
    episode start (2026-10-03). Without it, an episode ending inside a menu
    (Phi = -1) pays the next episode a phantom +w refund on step one — the
    same boundary bug _collect_segment already fixes on a client rebuild."""
    src = open(os.path.join("developmental_ai", "core",
                            "developmental_loop.py")).read()
    body = src[src.index("    def _run_episode_parallel("):
               src.index("    def _start_stream(")]
    head = body[:body.index("self._parallel_reset(envs)") + 2000]
    for line in ("self._gui_dwell_phi = None", "self._gui_run = 0",
                 "self._pitch_level_phi = None"):
        assert line in head, f"episodic body does not re-adopt: {line}"
    print("  E. episodic body re-adopts the gui-dwell and pitch-level cost "
          "potentials at episode start (no phantom refund across a reset)")


if __name__ == "__main__":
    test_state_cost_must_not_pay_while_pinned()
    test_source_contracts()
    test_no_one_way_doors()
    test_stream0_dwell_without_scaffold()
    test_farm_damp_reaches_intrinsic()
    test_episodic_boundary_readopts_cost_potentials()
    test_gui_dwell_step_cost()
    print("[gui-farm] ALL PASS")
