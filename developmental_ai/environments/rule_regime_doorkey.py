"""
Rule-Regime DoorKey — a *costly perceptual-aliasing* env for the Rung-6
(Global-Workspace) decisive test.

WHY THIS EXISTS
---------------
The cross-task transfer version of Rung 6 kept hitting the MEASUREMENT WALL: a
DoorKey task is reactively solvable (key/door are visible, "approach salient
object + toggle" works), so a feed-forward policy routes around the symbolic
broadcast and the lesion effect stays small. To answer *decisively* whether the
broadcast can be made load-bearing, the task must make the correct action
**provably not a function of the observation**, AND make probing for the hidden
state **costly** — otherwise the agent substitutes exploration for the broadcast
and the wall returns.

THE DESIGN (affordance "Heaven/Hell")
-------------------------------------
Each episode samples a hidden AFFORDANCE REGIME, 50/50:

  * KEY   regime — the door is locked; you must pick up the key, then toggle.
  * NOKEY regime — the door opens by toggling directly; **picking up the key is
                   a trap** (episode ends, no reward).

Two enforced properties remove every reactive escape:

  1. REGIME HIDDEN FROM OBSERVATION. The door is rendered identically in both
     regimes (a DisguisedDoor encodes "closed" as the *locked* state 2, never the
     unlocked-closed state 1), and the key is present in both. No observation
     feature predicts the regime — the start-state obs distribution is identical
     across regimes (verified in the __main__ self-test).
  2. PROBING IS COSTLY BOTH WAYS. The two regime-disambiguating actions are
     fatal in the wrong regime:
        * KEY  regime: toggling the (locked-looking) door WITHOUT the key
                       -> trap (terminate, trap_reward).
        * NOKEY regime: picking up the key -> trap (terminate, trap_reward).

So no single reactive policy can win both regimes: "always fetch key then toggle"
fails NOKEY (key trap); "always toggle directly" fails KEY (no-key toggle trap).
The obs-only ceiling is provably ~0.5 (commit to one regime). The ONLY safe
disambiguator is a signal that *tells* the agent the active rule every step —
the symbolic broadcast (see core/rule_regime_broadcast.py). The broadcast carries
task-structure KNOWLEDGE (the active affordance rule), is always present (nothing
to remember -> not a memory test), and is the env's `regime` attribute, exposed to
the broadcaster exactly as `carrying` is.

FALSIFICATION
-------------
If a policy WITH the true regime broadcast exceeds the 0.5 aliasing ceiling
(~1.0) while zero/constant/noise broadcasts stay capped at ~0.5, the broadcast is
decisively load-bearing. If even the intact arm caps at ~0.5, it is not — clean
fail. The mandatory validity gate is that the zero/constant/noise arms DO cap at
~0.5 (otherwise the env leaks the regime and the test is void).
"""

from __future__ import annotations

import gymnasium

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from minigrid.minigrid_env import MiniGridEnv

KEY_REGIME = "KEY"
NOKEY_REGIME = "NOKEY"


class DisguisedDoor(Door):
    """A door whose *encoded* state never reveals the regime: it renders as
    OPEN (0) when open and LOCKED (2) when closed — never the closed-unlocked
    state (1). In the NOKEY regime its toggle opens it without a key, but the
    encoding still shows "locked when closed", so the observation is identical
    to a genuinely-locked door. Behaviour is regime-aware; appearance is not."""

    def encode(self):
        state = 0 if self.is_open else 2  # open=0 else locked-looking=2
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def toggle(self, env, pos):
        if getattr(env, "regime", KEY_REGIME) == NOKEY_REGIME:
            # Directly openable — no key required. (is_locked stays True only for
            # the disguised ENCODING; can_overlap() gates on is_open, so an open
            # door is passable regardless.)
            self.is_open = not self.is_open
            return True
        # KEY regime: standard locked behaviour (needs the matching key).
        return super().toggle(env, pos)


class RuleRegimeDoorKeyEnv(MiniGridEnv):
    """DoorKey with a hidden affordance regime and costly probes. See module
    docstring. `regime` is resampled every episode and exposed for the symbolic
    broadcast; it is NEVER encoded into the observation."""

    def __init__(
        self,
        size: int = 5,
        regime_prob: float = 0.5,   # P(KEY regime)
        trap_reward: float = 0.0,   # reward when a wrong-regime probe is taken
        max_steps: int | None = None,
        **kwargs,
    ):
        self.regime_prob = float(regime_prob)
        self.trap_reward = float(trap_reward)
        self.regime = KEY_REGIME  # set per-episode in _gen_grid
        if max_steps is None:
            max_steps = 10 * size**2
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "follow the active rule to open the door and reach the goal"

    def _gen_grid(self, width, height):
        # Sample the hidden regime for THIS episode from the env RNG (seeded ->
        # reproducible). The regime is the ONLY thing that differs between arms
        # of the experiment; it is not placed in the grid / observation.
        self.regime = (
            KEY_REGIME if self._rand_float(0.0, 1.0) < self.regime_prob else NOKEY_REGIME
        )

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(), width - 2, height - 2)

        splitIdx = self._rand_int(2, width - 2)
        self.grid.vert_wall(splitIdx, 0)
        self.place_agent(size=(splitIdx, height))

        doorIdx = self._rand_int(1, height - 2)
        # Always created locked (so KEY regime is standard); the DisguisedDoor
        # makes the NOKEY regime visually indistinguishable.
        self.put_obj(DisguisedDoor("yellow", is_locked=True), splitIdx, doorIdx)

        # The key is present in BOTH regimes (decoy/trap in NOKEY).
        self.place_obj(obj=Key("yellow"), top=(0, 0), size=(splitIdx, height))

        self.mission = self._gen_mission()

    def step(self, action):
        action = int(action)
        fwd_cell = self.grid.get(*self.front_pos)
        carrying_key = isinstance(self.carrying, Key)

        # Detect the two costly probes BEFORE the base env processes the action.
        toggle_locked_door_no_key = (
            action == int(self.actions.toggle)
            and isinstance(fwd_cell, Door)
            and not fwd_cell.is_open
            and self.regime == KEY_REGIME
            and not carrying_key
        )
        pickup_key_in_nokey = (
            action == int(self.actions.pickup)
            and isinstance(fwd_cell, Key)
            and self.regime == NOKEY_REGIME
        )

        obs, reward, terminated, truncated, info = super().step(action)

        if toggle_locked_door_no_key or pickup_key_in_nokey:
            reward = self.trap_reward
            terminated = True
            info["trap"] = (
                "toggle_no_key" if toggle_locked_door_no_key else "pickup_key_nokey"
            )
        info["regime"] = self.regime
        return obs, reward, terminated, truncated, info


def _register():
    """Register a few sizes under MiniGrid-* IDs so the existing make_env path
    (flatten wrapper + minigrid handling keys off the 'MiniGrid' prefix) applies
    unchanged. Idempotent.

    Two trap regimes per size:
      * default IDs (trap_reward=0.0): a wrong-regime probe just ends the episode
        at 0 — leaves a SAFE 'commit to one regime' strategy (~0.5 ceiling) that
        the broadcast gives no advantage over, so it is NOT decisive.
      * '-Pen-' IDs (trap_reward=-1.0): a wrong-regime probe is PUNISHED, so
        committing has negative expected value and the ONLY route to positive
        reward is to READ the regime from the broadcast and avoid the wrong
        action. This removes the free escape hatch -> makes the broadcast
        necessary, the decisive form of the test.
    """
    entry = "developmental_ai.environments.rule_regime_doorkey:RuleRegimeDoorKeyEnv"
    for size in (5, 6, 8):
        specs = [
            (f"MiniGrid-RuleRegimeDoorKey-{size}x{size}-v0", {"size": size}),
            (f"MiniGrid-RuleRegimeDoorKey-Pen-{size}x{size}-v0",
             {"size": size, "trap_reward": -1.0}),
        ]
        for env_id, kw in specs:
            if env_id in gymnasium.registry:
                continue
            gymnasium.register(id=env_id, entry_point=entry, kwargs=kw)


_register()


if __name__ == "__main__":
    # ---- validity self-test: the aliasing must be real and the traps must bite ----
    import numpy as np

    env = RuleRegimeDoorKeyEnv(size=5)

    # (1) Observation is independent of the regime: flip the regime WITHOUT
    #     regenerating the grid; the encoded image must be byte-identical.
    obs, _ = env.reset(seed=0)
    img_a = env.grid.encode().copy()
    saved = env.regime
    env.regime = NOKEY_REGIME if saved == KEY_REGIME else KEY_REGIME
    img_b = env.grid.encode().copy()
    env.regime = saved
    assert np.array_equal(img_a, img_b), "LEAK: grid encoding depends on regime!"
    # The door cell must encode as locked(2) while closed in either regime.
    door_states = [img_a[x, y, 2] for x in range(5) for y in range(5)
                   if img_a[x, y, 0] == OBJECT_TO_IDX["door"]]
    assert door_states and all(s == 2 for s in door_states), \
        f"door not disguised as locked: {door_states}"
    print("[ok] obs is independent of regime; door disguised as locked")

    # (2) Regime distribution over many resets ~ regime_prob.
    n = 400
    keys = sum(1 for _ in range(n) if (env.reset()[0] is not None and env.regime == KEY_REGIME))
    print(f"[ok] regime KEY fraction over {n} resets: {keys / n:.2f} (~0.50 expected)")

    # (3) NOKEY toggle opens the door WITHOUT a key.
    found = False
    for s in range(50):
        env.reset(seed=1000 + s)
        if env.regime != NOKEY_REGIME:
            continue
        # find the door, stand in front of it, face it, toggle
        door_pos = None
        for x in range(env.grid.width):
            for y in range(env.grid.height):
                c = env.grid.get(x, y)
                if isinstance(c, Door):
                    door_pos = (x, y)
        dx, dy = door_pos
        env.agent_pos = (dx - 1, dy)        # cell left of the door
        env.agent_dir = 0                    # facing right -> toward the door
        env.carrying = None
        _, _, term, _, info = env.step(int(env.actions.toggle))
        door = env.grid.get(dx, dy)
        assert door.is_open, "NOKEY toggle did not open the door"
        assert not term and "trap" not in info, "NOKEY toggle wrongly trapped"
        found = True
        break
    assert found, "no NOKEY episode sampled in 50 tries (unlucky seed?)"
    print("[ok] NOKEY: toggling opens the door with no key, no trap")

    # (4) KEY-regime toggle WITHOUT key -> trap (terminate).
    found = False
    for s in range(50):
        env.reset(seed=2000 + s)
        if env.regime != KEY_REGIME:
            continue
        door_pos = None
        for x in range(env.grid.width):
            for y in range(env.grid.height):
                if isinstance(env.grid.get(x, y), Door):
                    door_pos = (x, y)
        dx, dy = door_pos
        env.agent_pos = (dx - 1, dy)
        env.agent_dir = 0
        env.carrying = None
        _, r, term, _, info = env.step(int(env.actions.toggle))
        assert term and info.get("trap") == "toggle_no_key", \
            f"KEY no-key toggle did not trap: term={term} info={info}"
        found = True
        break
    assert found, "no KEY episode sampled in 50 tries"
    print("[ok] KEY: toggling a locked door with no key -> trap (terminate)")

    # (5) NOKEY-regime key pickup -> trap (terminate).
    found = False
    for s in range(80):
        env.reset(seed=3000 + s)
        if env.regime != NOKEY_REGIME:
            continue
        key_pos = None
        for x in range(env.grid.width):
            for y in range(env.grid.height):
                if isinstance(env.grid.get(x, y), Key):
                    key_pos = (x, y)
        if key_pos is None:
            continue
        kx, ky = key_pos
        env.agent_pos = (kx - 1, ky)
        env.agent_dir = 0
        env.carrying = None
        front = env.grid.get(*env.front_pos)
        if not isinstance(front, Key):
            continue
        _, r, term, _, info = env.step(int(env.actions.pickup))
        assert term and info.get("trap") == "pickup_key_nokey", \
            f"NOKEY key pickup did not trap: term={term} info={info}"
        found = True
        break
    assert found, "could not position in front of key in NOKEY in 80 tries"
    print("[ok] NOKEY: picking up the key -> trap (terminate)")

    # (6) Penalty variant: a wrong-regime probe yields the negative trap_reward.
    penv = RuleRegimeDoorKeyEnv(size=5, trap_reward=-1.0)
    found = False
    for s in range(60):
        penv.reset(seed=4000 + s)
        if penv.regime != KEY_REGIME:
            continue
        door_pos = next(((x, y) for x in range(penv.grid.width)
                         for y in range(penv.grid.height)
                         if isinstance(penv.grid.get(x, y), Door)), None)
        dx, dy = door_pos
        penv.agent_pos = (dx - 1, dy); penv.agent_dir = 0; penv.carrying = None
        _, r, term, _, info = penv.step(int(penv.actions.toggle))
        assert term and info.get("trap") == "toggle_no_key" and r == -1.0, \
            f"penalty trap wrong: r={r} term={term} info={info}"
        found = True
        break
    assert found, "no KEY episode for penalty check"
    print("[ok] Pen variant: wrong-regime probe -> reward -1.0 (commit is unprofitable)")

    print("\nRULE-REGIME ENV SELF-TEST OK")
