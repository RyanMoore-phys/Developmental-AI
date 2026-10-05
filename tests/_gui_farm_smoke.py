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

Run: PYTHONPATH=. python tests/_gui_farm_smoke.py
"""
import os
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
    assert src.count("self._gui_dwell_phi = _gphi") == 2
    print("  B1. magnet shaping gui-gated in BOTH bodies; seek re-adopts")

    # the dwell cost must use the plain difference, in BOTH bodies
    assert "_gdw * (\n                            _gphi - float(_gprev))" in src \
        or "_gphi - float(_gprev)" in src
    assert "_gg * _gphi" not in src, (
        "gui dwell reverted to the gamma form — it would pay to dwell")
    assert "_gp * _php" not in src, (
        "gaze level reverted to the gamma form — it paid +0.00014/step to "
        "sit at the clamp")
    print("  B2. both state-cost potentials use the plain difference")

    # the cost must go to the EXTRINSIC channel
    i_cost = src.index("prim_extrinsic = prim_extrinsic + _gdw")
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
            g = (self.idx == 0
                 and GUI_OPEN_AT <= self.t < GUI_OPEN_AT + GUI_LEN)
            self.t += 1
            GUI_LOG[self.idx].append(bool(g))
            # truncates at 70 so the EPISODIC body ends; the lifelong
            # segment (60 steps) never reaches it, so no reset intervenes
            return (self._obs(), 0.0, False, self.t >= 70,
                    {"gui_open": bool(g)})

    return GuiEnv()


GUI_LOG = {}


INTRINSIC = {}   # the last _drive's stream-0 intrinsic handed to the mix


def _drive(lifelong, weight, scaffold, damp=None):
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
        rec, irec = [], []
        INTRINSIC["rec"] = irec
        _mix = ai.reward_mixer.mix

        def spy(i, e, update_stats=True):
            if update_stats:              # the PRIMARY call; scouts pass False
                rec.append(float(e))
                irec.append(float(i))
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
    print("[gui-farm] ALL PASS")
