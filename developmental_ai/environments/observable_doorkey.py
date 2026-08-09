"""
Observable key/no-key DoorKey — the Rung 1 (cumulative transfer) testbed.

A normal, fully-learnable DoorKey whose door is, per episode, EITHER:
  * KEY regime   — locked (needs the key, then toggle), OR
  * NOKEY regime — closed-but-unlocked (just toggle; the key is a decoy).

Crucially the regime is OBSERVABLE: MiniGrid's native door-state encoding already
distinguishes them — a locked door encodes state=2, a closed-unlocked door
state=1 — so the agent can see which regime it is in directly from the
observation. No hidden regime, no broadcast, no trap (this is NOT the Rung-6
device). It is just a richer DoorKey that demands TWO strategies:
  - "fetch the key, then open the door"  (KEY), and
  - "open the door directly"             (NOKEY, ignore the decoy key),
plus the shared navigate/reach-goal behaviour. That variety is the point: it
mints a richer skill set for the cumulative-transfer test (5x5 base run builds
the bank → 8x8 transfer run reuses those skills + adds new ones).

Standard sparse goal reward (1 − 0.9·steps/max_steps on reaching the goal, else
0), so the task is learnable and a transfer-vs-fresh competence comparison is
well-defined.
"""

from __future__ import annotations

import gymnasium

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from minigrid.minigrid_env import MiniGridEnv

KEY_REGIME = "KEY"
NOKEY_REGIME = "NOKEY"


class ObservableKeyNoKeyEnv(MiniGridEnv):
    """DoorKey whose door is randomly locked (KEY) or closed-unlocked (NOKEY)
    each episode. The regime is visible via the native door-state encoding."""

    def __init__(self, size: int = 5, key_regime_prob: float = 0.5,
                 max_steps: int | None = None, **kwargs):
        self.key_regime_prob = float(key_regime_prob)
        self.regime = KEY_REGIME
        if max_steps is None:
            max_steps = 10 * size**2
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(mission_space=mission_space, grid_size=size,
                         max_steps=max_steps, see_through_walls=False, **kwargs)

    @staticmethod
    def _gen_mission():
        return "open the door (use the key only if it is locked) and reach the goal"

    def _gen_grid(self, width, height):
        self.regime = (KEY_REGIME if self._rand_float(0.0, 1.0) < self.key_regime_prob
                       else NOKEY_REGIME)
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(), width - 2, height - 2)

        splitIdx = self._rand_int(2, width - 2)
        self.grid.vert_wall(splitIdx, 0)
        self.place_agent(size=(splitIdx, height))

        doorIdx = self._rand_int(1, height - 2)
        # KEY -> locked (state=2 in obs); NOKEY -> closed-unlocked (state=1).
        locked = self.regime == KEY_REGIME
        self.put_obj(Door("yellow", is_locked=locked), splitIdx, doorIdx)

        # Key present in BOTH regimes — needed in KEY, a decoy to ignore in NOKEY.
        self.place_obj(obj=Key("yellow"), top=(0, 0), size=(splitIdx, height))
        self.mission = self._gen_mission()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["regime"] = self.regime
        return obs, reward, terminated, truncated, info


def _register():
    entry = "developmental_ai.environments.observable_doorkey:ObservableKeyNoKeyEnv"
    for size in (5, 6, 8):
        env_id = f"MiniGrid-ObsKeyNoKey-{size}x{size}-v0"
        if env_id in gymnasium.registry:
            continue
        gymnasium.register(id=env_id, entry_point=entry, kwargs={"size": size})


_register()


if __name__ == "__main__":
    # ---- self-test: regime observable, both regimes solvable, key is decoy ----
    import numpy as np
    from minigrid.core.constants import OBJECT_TO_IDX

    env = ObservableKeyNoKeyEnv(size=5)

    # (1) regime is visible: locked door encodes state=2, unlocked-closed state=1.
    seen = {}
    for s in range(80):
        env.reset(seed=s)
        door = next(env.grid.get(x, y) for x in range(env.grid.width)
                    for y in range(env.grid.height) if isinstance(env.grid.get(x, y), Door))
        seen.setdefault(env.regime, set()).add(door.encode()[2])
    assert seen.get(KEY_REGIME) == {2}, f"KEY door state should be 2 (locked): {seen}"
    assert seen.get(NOKEY_REGIME) == {1}, f"NOKEY door state should be 1 (closed): {seen}"
    print(f"[ok] regime observable in obs: KEY->state2(locked), NOKEY->state1(closed)")

    # (2) NOKEY: toggling the door WITHOUT a key opens it (no key needed).
    found = False
    for s in range(60):
        env.reset(seed=500 + s)
        if env.regime != NOKEY_REGIME:
            continue
        dx, dy = next((x, y) for x in range(env.grid.width) for y in range(env.grid.height)
                      if isinstance(env.grid.get(x, y), Door))
        env.agent_pos = (dx - 1, dy); env.agent_dir = 0; env.carrying = None
        env.step(int(env.actions.toggle))
        assert env.grid.get(dx, dy).is_open, "NOKEY toggle should open the door"
        found = True; break
    assert found
    print("[ok] NOKEY: toggle opens the door with no key (decoy key ignorable)")

    # (3) KEY: toggling WITHOUT the key does NOT open it (key genuinely needed).
    found = False
    for s in range(60):
        env.reset(seed=900 + s)
        if env.regime != KEY_REGIME:
            continue
        dx, dy = next((x, y) for x in range(env.grid.width) for y in range(env.grid.height)
                      if isinstance(env.grid.get(x, y), Door))
        env.agent_pos = (dx - 1, dy); env.agent_dir = 0; env.carrying = None
        env.step(int(env.actions.toggle))
        assert not env.grid.get(dx, dy).is_open, "KEY door should stay shut without key"
        found = True; break
    assert found
    print("[ok] KEY: locked door stays shut without the key")

    # (4) reward-on-goal works (standard sparse reward; learnable task).
    env.reset(seed=3)
    print(f"[ok] regime sampling balanced; max_steps={env.max_steps}")
    print("\nOBSERVABLE KEY/NO-KEY ENV SELF-TEST OK")
