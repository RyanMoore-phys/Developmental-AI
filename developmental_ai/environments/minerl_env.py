"""MineRL (Minecraft) → Gymnasium adapter.

Real Minecraft, via MineRL 1.0 (master). Verified on the pod against
MineRLTreechop-v0:
  * obs: dict with "pov" = (64, 64, 3) uint8 HWC — IDENTICAL shape to
    Crafter, so the validated pixel pipeline (CNN encoder/decoder, uint8
    replay, CNN-ICM) applies unchanged. We surface flat CHW float [0,1].
  * action: a DICT space (attack/forward/jump/left/right/back/sneak/sprint
    each Discrete(2), plus camera Box(2,) pitch/yaw in degrees).
  * reward: +1 per log collected (Treechop).
  * ~14 steps/s headless (Minecraft ticks at 20Hz, so ~0.7x realtime).

WHY AN ADAPTER
  1. LEGACY GYM: MineRL uses old gym — reset() -> obs, step() -> 4-tuple,
     old gym.spaces. The project is gymnasium (5-tuple). Same bridge as the
     Crafter adapter.
  2. DICT ACTIONS -> DISCRETE MACROS: this project's policy emits a single
     discrete action; Minecraft wants a dict of simultaneous keypresses +
     continuous camera. We expose a Discrete(N) space of hand-designed MACRO
     actions (the standard MineRL practice), each expanding to a full action
     dict. Macros are chosen for tree-chopping: look around, approach, and
     attack-while-moving (chopping requires holding attack on a log).
  3. Minecraft has no achievements dict. Treechop's reward spikes (+1 per
     log) are exactly what the DISCOVERED goal mode consumes — the earned
     goal space (which BEAT the oracle one on Crafter, rung 10) is the
     natural fit here.

ACTION REPEAT: each macro is held for `action_repeat` game ticks (standard
MineRL practice). Camera macros are scaled by 1/repeat so the NET turn per
macro stays the same regardless of repeat. Halves the decision rate, doubles
game-time per policy step and per buffer slot, and makes "chop" actually
hold attack long enough to progress a log break.

CRASH RECOVERY: MineRL's Java clients die/hang after hours (socket timeouts
on reset — killed a 320k-step run at 80% complete). Any exception from the
underlying env now rebuilds the Minecraft client in place: reset() retries
on a fresh client; step() truncates the episode (the agent didn't fail —
the simulator did) and the caller's autoreset lands on the fresh client.

HEADLESS: launch under `xvfb-run -a` (Minecraft needs an X display even
headless). First reset compiles/launches the Java client (~90s).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

# Macro action set. Each entry is a partial action dict merged onto noop().
# Camera is [pitch, yaw] in degrees (net, across the held ticks).
#
# HOLDING IS EMERGENT (July 2026 curiosity redesign): "attack" is a plain
# action_repeat-tick press, exactly like walking. A sustained hold — what
# block-breaking requires — exists ONLY as a sequence the agent chooses:
# picking attack again next step keeps the button down (the MCP-Reborn
# bridge diffs consecutive key states, so back-to-back attack steps emit no
# release edge and breaking progress carries over). The intended discovery
# loop: random attack near a block -> crack pixels appear -> curiosity
# (learning progress) fires -> longer chains -> block breaks -> first-break
# reward -> goal slot -> skill. The old 60-tick "_ticks" chop bursts handed
# the hold to the agent for free; that scaffold is gone. ("_ticks" remains
# supported by the step loop for probes/tests.)
# The tool granted at spawn. SINGLE SOURCE OF TRUTH: the agent-start spec and
# the inventory-derived mainhand fallback must never disagree about what the
# agent is holding — a mismatch would reintroduce exactly the "is it holding
# anything?" confusion that cost a day (see the mainhand derivation below).
# NOTE this item is HELD from spawn, not merely carried: SimpleInventoryAgentStart
# writes it to inventory index 0, which IS hotbar slot 0, and `currentItem`
# starts at 0. Nothing in TREECHOP_MACROS can change the selected slot.
# DEFAULT ONLY — overridable per-config via `environment.start_tool`.
# Set it to null/empty to spawn genuinely EMPTY-HANDED (see _tool below).
START_TOOL: str = "iron_axe"

TREECHOP_MACROS: list = [
    {},                                                  # 0 noop
    {"forward": 1},                                      # 1 walk
    {"forward": 1, "jump": 1},                           # 2 jump-walk (obstacles)
    {"camera": [0.0, -15.0]},                            # 3 turn left
    {"camera": [0.0, 15.0]},                             # 4 turn right
    {"attack": 1},                                       # 5 attack (hold = re-pick)
    {"attack": 1, "forward": 1},                         # 6 attack while walking
    {"camera": [-15.0, 0.0]},                            # 7 look up
    {"camera": [15.0, 0.0]},                             # 8 look down
    {"back": 1},                                         # 9 back off
    # ---- PLAYER-LIKE BUTTONS (2026-07-26, crafting phase 2) -------------
    # APPEND-ONLY, never reorder: every saved skill/policy/config indexes
    # actions by position, and indices 0-9 must stay exactly what they were.
    # These two are the MINIMAL set for GUI crafting and placement, chosen
    # over hotbar.1-9 to keep the exploration burden small:
    #   inventory — toggles the crafting screen (probe-verified: it opens,
    #               renders into the agent's own frame at 47.9% pixel change,
    #               and camera/click drive it);
    #   use       — right-click: places a held block (the crafting table),
    #               interacts. Slot selection stays at hotbar 0 (currentItem
    #               initialises to 0), so an item moved to the first hotbar
    #               slot in the GUI becomes the held item — reachable purely
    #               by cursor + click, no hotbar keys needed.
    # NOTHING here encodes a recipe. The buttons are capabilities; what they
    # do must be discovered by using them, like everything else.
    {"inventory": 1},                                    # 10 toggle inventory
    {"use": 1},                                          # 11 right-click/place
    # ---- HOLD ATTACK (2026-08-05) ---------------------------------------
    # THE ARITHMETIC THAT FORCED THIS. A barehanded oak log needs ~60 game
    # ticks of sustained attack on ONE block. At action_repeat 2 that is 30
    # CONSECUTIVE `attack` decisions, stationary (any move >0.05 blocks resets
    # the streak). A step-wise stochastic policy cannot produce that: at a
    # realistic 10% attack rate P(30 in a row) ~ 1e-30, and even at 50% it is
    # 1e-9. Measured live: median swing 2 ticks, ONE swing in 400 reaching 8.
    # It is not a tuning problem — it is unreachable by construction, and
    # fixing the entropy collapse made it rarer still (a high-entropy policy
    # switches action every step, which is in direct tension with holding).
    #
    # 20 ticks in ONE decision makes a barehanded log 3 consecutive picks
    # instead of 30 — P(3 in a row) ~ 0.1%, i.e. reachable. The step loop
    # already supported this: `_macro_ticks` honours `_ticks` and step()
    # applies the SAME action dict for every tick, so the button genuinely
    # stays down rather than being re-pressed.
    #
    # A CAPABILITY, NOT A SCRIPT — the same argument the table already makes
    # for `inventory` and `use`: "the buttons are capabilities; what they do
    # must be discovered by using them". Nothing here says when to hold, what
    # to aim at, or that trees matter. Compare a human, who holds the mouse
    # button down rather than clicking thirty times.
    # APPEND-ONLY: indices 0-11 are untouched, as every stored skill and
    # policy indexes actions by position.
    # `_ticks` is ABSOLUTE — _macro_ticks returns m.get("_ticks",
    # action_repeat), so this does NOT scale with action_repeat. That matters:
    # at repeat 2 this macro was 10x a plain attack (20 vs 2 ticks), and the
    # 10x is what made a barehanded log reachable in 3 consecutive picks
    # instead of 30. At repeat 4 a plain attack is 4 ticks, so leaving this at
    # 20 would quietly halve the advantage to 5x and put the log back out of
    # reach. 40 preserves the ratio and keeps the arithmetic in the note above
    # true: ~60 ticks for a barehanded oak log is still ~2 consecutive picks.
    # Editing the VALUE is safe; only the INDEX is append-only.
    {"attack": 1, "_ticks": 40},                         # 12 HOLD attack
]


class MineRLEnvAdapter(gym.Env):
    """Gymnasium-API wrapper over a MineRL env with discrete macro actions."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_name: str = "MineRLTreechop-v0",
                 macros: Optional[list] = None, image_size: int = 64,
                 action_repeat: int = 1, render_size: int = 0,
                 lifelong: bool = False, remote_server: Optional[str] = None,
                 remote_step_delay_s: float = 0.0,
                 start_tool: Optional[str] = START_TOOL,
                 log_tick_reward: float = 0.0,
                 log_tick_cap: int = 120,
                 log_break_reward: float = 5.0,
                 break_decay_scale: float = 0.0,
                 break_memory_path: Optional[str] = None,
                 agent_name: Optional[str] = None):
        super().__init__()
        # LIFELONG: drop the 8000-tick time-up quit + TimeLimit backstop so the
        # world NEVER ends on time (death = 64-log success only). The Java
        # server manufactures the boundary regardless of loop logic, so this
        # env-level change is load-bearing. Off -> byte-identical.
        self._lifelong = bool(lifelong)
        # REMOTE SERVER ("ip:port" or None): join an external multiplayer
        # server instead of generating a local world. The mission XML carries
        # <RemoteServer> inside ServerInitialConditions; the MCP-Reborn client
        # sees it in loadOrCreateWorld() and calls connectToServer() (vanilla
        # multiplayer join) instead of createNewWorld(). Server-side handlers
        # (world gen, time-up quit, time/spawning conditions) are ignored by a
        # foreign server — reward/termination must be client-observation-based
        # (ours already are: inventory deltas + mine_block stats). None -> off,
        # byte-identical local behaviour.
        self.remote_server = str(remote_server) if remote_server else None
        # ---- ACTION THROTTLE (remote mode only) --------------------------
        # 0.0 = off, and off is usually CORRECT: with a real server the client
        # is tick-synced to the server's 20 TPS, so the agent already runs at
        # wall-clock speed and cannot out-pace a human. This exists for the
        # case where a server's anti-cheat/anti-spam still objects (rapid
        # camera slew reads as erratic to some plugins) — raising it trades
        # samples-per-hour for a lower kick rate. Applied ONLY in remote mode
        # so local generated-world runs stay exactly as fast as before.
        self.remote_step_delay_s = float(remote_step_delay_s or 0.0)
        # server-visible identity; None keeps the historical single-client
        # behaviour byte-for-byte (see create_agent_start)
        self.agent_name = str(agent_name) if agent_name else None
        # ---- START TOOL (None/'' = spawn EMPTY-HANDED) --------------------
        # Handing the agent an iron axe makes a log ~8 ticks instead of ~80,
        # which is the difference between a chop it can stumble into and one
        # it has to sustain. It also made the two streams DISAGREE: on the
        # external server the grant was re-applied on every rejoin while the
        # local scout kept its own axe, so the world model was being taught
        # two different break dynamics for the same block. Empty-handed is
        # both the honest test and the consistent one.
        self._start_tool = (str(start_tool) if start_tool else None)
        # ---- LOG ECONOMICS (2026-08-03) ----------------------------------
        # MEASURED cause of "breaks everything except trees": the first-break
        # tiers re-arm every goal horizon, so four trivial block types can be
        # re-harvested indefinitely for ~0.75/horizon requiring no aim and no
        # sustained effort, while a log pays once and costs ~60 consecutive
        # ticks on one specific target. Reliable-and-easy beat large-and-hard
        # and PPO correctly converged on clearing ground vegetation.
        #
        # `log_tick_reward` pays for the EFFORT, per attack tick invested,
        # for logs only — so the process of felling a log out-earns any other
        # block per tick, not merely at completion.
        #
        # PAID AT COMPLETION, and it cannot be otherwise: nothing in the
        # observation says "I am currently mining a log". Only the completed
        # break names the block (mine_block counters). Retroactive payment
        # proportional to ticks invested carries the same gradient once the
        # value function propagates it back over the swing — which is exactly
        # what GAE does — but be clear that it is retroactive.
        self._log_tick_reward = float(log_tick_reward or 0.0)
        # HARD CAP, load-bearing. `_attack_run` has been observed at 13538
        # ticks on a swing that never landed; without a bound a single
        # mis-attributed streak would pay thousands and wreck the run. 120 =
        # 2x a barehanded log, so real chops are never clipped.
        self._log_tick_cap = int(log_tick_cap or 120)
        # instance override consulted by the _break_reward classmethod
        self.LOG_BREAK_REWARD = float(log_break_reward)
        # _break_reward is a CLASSMETHOD: an instance attribute is invisible
        # to it, so the configured value was silently ignored and every log
        # paid the 5.0 fallback — the first felled log under the new stack
        # printed "(+5.0 completion tier)" against a configured 20.0
        # (2026-08-08). The class attribute is what the classmethod actually
        # reads; every env in a process shares one config, so setting it
        # class-wide is correct.
        MineRLEnvAdapter.LOG_BREAK_REWARD = float(log_break_reward)
        # ---- REACH, FROM EVIDENCE (2026-08-04) ---------------------------
        # `breakable_in_reach` is a VLM-taught predicate, and llava cannot
        # judge reach from a still frame — it learned to answer "no" almost
        # always. Measured: Reach sense 0.01-0.03 for the entire run while the
        # agent broke 4931 blocks, i.e. things were demonstrably in reach the
        # whole time. Two shaping terms (approach-to-reach) and one
        # proprioception field ride on that number and were inert because of it.
        # A break is GROUND TRUTH that something was within reach. This trace
        # rises to 1.0 on a break and decays otherwise — evidence rather than
        # a guess. It is retrospective and says "I recently could touch
        # something", not "I can right now"; that is weaker than the wireframe
        # affordance it stands in for, and honest about being a proxy.
        self._reach_evidence = 0.0
        self._reach_decay = 0.93
        # ---- BASELINE SEEDING (fix 2026-08-04) ---------------------------
        # The break delta defaults a missing type to 0 (so a type's FIRST
        # break is counted). That is right once a baseline exists — and
        # catastrophic before one does: reset() captures the baseline from the
        # first obs, and on a rejoin that obs often carries NO stats yet, so
        # the baseline is {} and the next stats-bearing observation counts
        # every type's ENTIRE career total as fresh breaks.
        # MEASURED: every block type doubled EXACTLY between two consecutive
        # segments (dirt 56->112, grass_block 31->63, grass 20->40,
        # oak_leaves 14->28) while `attack` was 2% of 1024 steps — 146 blocks
        # cannot break in ~40 ticks. The reported break counts, and therefore
        # the first-break rewards riding on them, were inflated.
        # Seed lazily instead: the first observation that actually CARRIES
        # stats becomes the baseline, and nothing is counted from it.
        self._mine_seeded = False
        # ---- FAMILIARITY DECAY ON BREAKS (2026-08-06) --------------------
        # MEASURED: 85% of 2231 breaks were dirt. The first-break tier
        # re-arms every goal horizon, so the ground under the agent's feet is
        # an inexhaustible 0.15 income requiring no aim, no target selection
        # and no sustained effort. Every capability built for chopping — the
        # HOLD macro, the levelled view, reach — got pointed at the floor.
        #
        # DELIBERATELY NOT "dirt pays less". This declares nothing about
        # which blocks matter: the payout for a block TYPE decays with how
        # often the agent has already broken THAT type, as
        #     tier / (1 + n_type / scale)
        # Dirt at 1899 breaks decays to ~1% of its tier; a log at zero breaks
        # is untouched at full value. Novelty of ACHIEVEMENT, the same
        # 1/(1+n) shape the curiosity terms already use — the agent stops
        # being paid for what it has thoroughly mastered, and what it has
        # never managed stays worth the most.
        # 0 = off = byte-identical for every other config.
        self._break_decay_scale = float(break_decay_scale or 0.0)
        # REMOTE DISCONNECT WATCHDOG: a kick/timeout on a real server leaves
        # the client on a STATIC screen (Disconnected / death GUI) while
        # step() keeps "succeeding" on frozen pixels — no exception, so the
        # crash-recovery path never fires and the stream would train on a
        # dead screen. Signature: byte-identical POV across many consecutive
        # steps THAT COMMANDED MOTION (camera/walk always changes a live
        # frame). Counter only advances while actively stepping, so training
        # pauses can't false-positive. On trigger: rebuild client -> fresh
        # mission init -> rejoins the server (auto-rejoin).
        self._frozen_limit = 150          # ~10-15s of active motion steps
        # steps to ignore after a reset/rebuild while the client boots and
        # connects; below this the frozen-POV test cannot fire at all.
        self._frozen_grace = 600
        self._steps_since_reset = 0
        self._deaths_rejoined = 0         # respawn-rejoins performed
        # log ONCE which gui-detection source is in use: silently
        # falling back to pixels is exactly how the stone false
        # positive went unnoticed for a whole run.
        self._gui_src_logged = False
        # WHAT IS IT ACTUALLY CHOOSING. Every diagnosis so far has inferred
        # behaviour from its consequences; this measures the decision itself,
        # which is the thing that has been guessed at repeatedly.
        self._action_hist: Dict[int, int] = {}
        self._frozen_count = 0
        self._prev_pov_digest: Optional[bytes] = None
        # macros that must change a live frame (camera moves + locomotion);
        # computed from the source list — self._macros is assigned below.
        self._motion_macros = frozenset(
            i for i, m in enumerate(macros or TREECHOP_MACROS)
            if "camera" in m or "forward" in m or "back" in m)
        self._env_name = env_name
        self._macros = list(macros or TREECHOP_MACROS)
        self.image_size = int(image_size)
        # render_size > image_size: the game renders NATIVELY at render_size
        # (real-Minecraft-quality frames for videos + the vision LLM via
        # render()/_last_pov), and the agent's obs is block-mean downsampled
        # to image_size — the agent's world is unchanged. 0 = render at
        # image_size (legacy behaviour). Must be an integer multiple.
        self.render_size = int(render_size) if render_size else self.image_size
        if self.render_size % self.image_size != 0:
            raise ValueError(
                f"render_size {self.render_size} must be a multiple of "
                f"image_size {self.image_size}")
        self.action_repeat = max(1, int(action_repeat))
        # "Full speed" in blocks per AGENT STEP, which is action_repeat game
        # ticks long. MOVE_SCALE is quoted per 2 ticks (its original
        # calibration), so this keeps `moved` spanning [0,1] across a normal
        # walk at any repeat instead of pinning at 1.0.
        self._move_scale = self.MOVE_SCALE * self.action_repeat / 2.0
        self._last_pov: Optional[np.ndarray] = None
        self._log_count = 0
        self._mine_baseline: Dict[str, int] = {}
        self._broken_this_episode: set = set()
        # external-curiosity state (territory coverage + damage/death)
        self.coverage_cell = 8          # world blocks per novelty cell
        self._visits: Dict[tuple, int] = {}
        self._prev_food = None
        self._prev_life: Optional[float] = None
        # ticks-to-break measurement (attack ticks -> break)
        self._attack_run = 0
        self._attack_run_max = 0
        # completed swing sequences that ended with NO block breaking, and the
        # per-type tally of what DID break — the two halves of the aim /
        # timing / target-selection question (see the note at the reset site).
        from collections import deque as _dq
        self._runs_nobreak = _dq(maxlen=400)
        self._breaks_by_type: Dict[str, int] = {}
        # LIFELONG BREAK MEMORY (2026-08-08). breaks_by_type is what the
        # familiarity decay AND the new intrinsic habituation key on — but it
        # lived only in-process, so every restart re-opened the dirt tier at
        # full worth and reset "mastered" to "novel". Mastery is not
        # per-process. Loaded here, saved on update (atomic tmp+rename),
        # None = off (old behaviour).
        self._break_memory_path = (str(break_memory_path)
                                   if break_memory_path else None)
        self._break_mem_dirty = 0
        # PLACEMENT MEMORY (2026-08-08): with break/sky/ground income
        # defunded the bot's next farm was PLACING blocks (use was its most
        # frequent action at 18%, pillar at y=71, 1 territory cell/hour) —
        # placement makes the same big frame change a break does and was the
        # one un-habituated way left to manufacture novelty. Same mastery
        # counters, same decay, same file.
        self._places_by_type: Dict[str, int] = {}
        # WHAT A `use` ACTUALLY CONSUMED — the only ground truth about what
        # is in the hand that this env can obtain (see the mainhand
        # derivation): `equipped_items` is dead in MineRL 1.0 and the
        # inventory observation is FLAT (item -> count, no slot indices), so
        # nothing else can name the held item.
        self._last_placed_item: Optional[str] = None
        # crafts and pickups joined the event memory with the typed event
        # stream (infra #44): mastery/habituation is about EVENTS the agent
        # causes, whatever their kind — the four kinds this adapter emits.
        self._crafts_by_type: Dict[str, int] = {}
        self._pickups_by_type: Dict[str, int] = {}
        self._inv_prev_counts: Dict[str, int] = {}
        if self._break_memory_path:
            try:
                import json as _json
                import os as _os
                if _os.path.exists(self._break_memory_path):
                    with open(self._break_memory_path) as _f:
                        _m = _json.load(_f)
                    if "breaks" in _m or "places" in _m:
                        self._breaks_by_type = {
                            str(k): int(v)
                            for k, v in (_m.get("breaks") or {}).items()}
                        self._places_by_type = {
                            str(k): int(v)
                            for k, v in (_m.get("places") or {}).items()}
                        self._crafts_by_type = {
                            str(k): int(v)
                            for k, v in (_m.get("crafts") or {}).items()}
                        self._pickups_by_type = {
                            str(k): int(v)
                            for k, v in (_m.get("pickups") or {}).items()}
                    else:       # legacy flat format = breaks only
                        self._breaks_by_type = {str(k): int(v)
                                                for k, v in _m.items()}
                    for _ck, _cv in (_m.get("cells") or {}).items():
                        try:
                            _x, _z = _ck.split(",")
                            self._visits[(int(_x), int(_z))] = int(_cv)
                        except Exception:
                            continue
                    logger.info(
                        "break memory: restored %d block types "
                        "(%d lifetime breaks, %d placements, %d crafts, "
                        "%d pickups) from %s",
                        len(self._breaks_by_type),
                        sum(self._breaks_by_type.values()),
                        sum(self._places_by_type.values()),
                        sum(self._crafts_by_type.values()),
                        sum(self._pickups_by_type.values()),
                        self._break_memory_path)
            except Exception as _e:
                logger.warning("break memory: restore failed (%s) — "
                               "starting fresh", _e)
        # GUI paralysis accounting (2026-07-27): what fraction of the
        # episode the agent spends frozen in a menu, and the longest
        # unbroken stretch. Printed per segment so a repeat is visible
        # immediately instead of after a frame-by-frame video audit.
        self._gui_steps = 0
        self._steps_total = 0
        self._gui_run = 0
        self._gui_run_max = 0
        self._break_ticks = _dq(maxlen=200)
        self._mine_prev: Dict[str, int] = {}   # per-step break deltas
        # craft economy state (mirrors the break economy exactly)
        self._craft_prev: Dict[str, int] = {}
        self._crafted_this_episode: set = set()
        # Monotonic ground-truth counters (life is pinned at MAX_LIFE by a
        # MineRL key-path bug, so it can never report being hurt or dying).
        self._prev_damage_taken: Optional[float] = None
        self._prev_deaths: Optional[float] = None
        # tool state, for LEARNED tool semantics (wear / loss / enablement)
        self._prev_tool_damage: Dict[str, float] = {}
        self._prev_mainhand: Optional[str] = None
        # separate history for the INVENTORY-DERIVED hand: sharing
        # _prev_mainhand with the dead equipped_items path created a
        # per-step false tool_lost loop (~7000 bogus KG facts).
        self._prev_derived_hand: Optional[str] = None
        self._env = self._make_underlying()

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3 * self.image_size * self.image_size,),
            dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self._macros))

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _block_mean(img: np.ndarray, k: int) -> np.ndarray:
        """Exact k-x antialiased downsample of an HWC uint8 image."""
        h, w, c = img.shape
        return img.reshape(h // k, k, w // k, k, c).mean(
            axis=(1, 3)).astype(np.uint8)

    def _pov_to_obs(self, obs: Dict) -> np.ndarray:
        """dict obs -> flat CHW float32 [0,1] (the project's pixel contract).

        With render_size > image_size, _last_pov keeps the FULL-RESOLUTION
        frame (videos + vision LLM see real Minecraft) while the returned
        obs is the block-mean downsample the agent has always seen."""
        pov = np.asarray(obs["pov"], dtype=np.uint8)
        self._last_pov = pov                     # kept raw for video/render
        if self.render_size != self.image_size:
            pov = self._block_mean(pov, self.render_size // self.image_size)
        return (pov.astype(np.float32) / 255.0).transpose(2, 0, 1).flatten()

    def _last_obs(self) -> np.ndarray:
        """Best-available observation after a client crash (for the truncated
        terminal transition)."""
        if self._last_pov is not None:
            pov = self._last_pov
            if self.render_size != self.image_size:
                pov = self._block_mean(pov,
                                       self.render_size // self.image_size)
            return (pov.astype(np.float32) / 255.0
                    ).transpose(2, 0, 1).flatten()
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _macro_ticks(self, idx: int) -> int:
        """How many game ticks this macro is held for."""
        return max(1, int(self._macros[int(idx)].get(
            "_ticks", self.action_repeat)))

    def _macro_to_action(self, idx: int) -> Dict:
        act = self._env.action_space.noop()
        ticks = self._macro_ticks(idx)
        for k, v in self._macros[int(idx)].items():
            if k == "_ticks":
                continue
            if k == "camera":
                # Net turn is spread across the held ticks.
                act[k] = np.array(v, dtype=np.float32) / ticks
            else:
                act[k] = v
        return act

    def _make_underlying(self):
        """Construct the raw MineRL env — a FIXED Treechop.

        Stock MineRLTreechop-v0 is BROKEN in MineRL 1.0: scripted contact
        chopping (probes 1-2, July 2026: 8k+ ticks pinned against trunks
        with attack held, iron axe in inventory) never breaks a block, never
        collects a log, never pays reward. The HumanControl/HumanSurvival
        env family — the exact pathway VPT used, sharing the same MCP-Reborn
        input bridge — is the working one. So we build Treechop's SETTING on
        that family: forest world (Treechop's generator options), iron axe,
        frozen daytime, 8000-tick server timer — and compute reward OURSELVES
        in step() from the inventory's log-count delta, independent of the
        (dead) RewardForCollectingItems handler.
        """
        import minerl  # noqa: F401  (registers + pulls herobraine)
        from minerl.herobraine.env_specs.human_survival_specs import (
            HumanSurvival)
        from minerl.herobraine.env_specs.human_controls import (
            HumanControlEnvSpec)
        from minerl.herobraine.env_specs.treechop_specs import (
            TREECHOP_WORLD_GENERATOR_OPTIONS)
        from minerl.herobraine.hero.mc import MS_PER_STEP
        import minerl.herobraine.hero.handlers as handlers

        render_size = self.render_size

        _lifelong = self._lifelong
        _remote = self.remote_server
        # closure for the spec class below: None => spawn empty-handed
        _tool = self._start_tool
        _agent_name = self.agent_name
        class TreechopFixed(HumanSurvival):
            def __init__(self):
                super().__init__(
                    name="MineRLTreechopFixed-v0",
                    resolution=(render_size, render_size),
                    # informational to spec.make (no TimeLimit applied); huge
                    # under lifelong so nothing honors it as a limit.
                    max_episode_steps=(10 ** 9 if _lifelong else 8000),
                )
                # Rendered into <Name> by mission.xml.j2 (the template reads
                # `agent_username | default("SkyBot")`, so an unpatched
                # template and every other env spec are unaffected). The
                # Malmo agent name and the multiplayer username are separate
                # fields and BOTH have to be unique — the first identifies
                # the agent to Malmo, the second to the Minecraft server.
                self.agent_username = _agent_name or "SkyBot"

            def create_observables(self):
                # SLIM observation (July 2026 — 16-env stall fix): stock
                # HumanSurvival adds ObserveFromFullStats(None) which
                # serializes the ENTIRE stats dict (thousands of fields) every
                # step — crippling at 16 parallel clients (a 1-env probe
                # tolerated it; the fleet stalled on it). We only need POV +
                # inventory (log count) + the mine_block stat (first-break
                # rewards), so drop straight to the HumanControl base
                # observables (POV + inventory) and add mine_block only.
                # EXTERNAL CURIOSITY (2026-07-25): the agent was BLIND to
                # where it was and to being hurt — so a lifelong agent in one
                # world had nothing that could surprise it once the WM
                # mastered the view, and mobs/death could not influence it at
                # all. Location + life stats restore that. These land in the
                # obs DICT and are surfaced through `info` only; the agent's
                # obs TENSOR stays pure pixels, so obs_dim is unchanged and
                # every saved skill policy stays loadable.
                from minerl.herobraine.hero.handlers.agent.observations \
                    .location_stats import ObservationFromCurrentLocation
                from minerl.herobraine.hero.handlers.agent.observations \
                    .lifestats import ObservationFromLifeStats
                # WHAT IS IN THE AGENT'S HAND (2026-07-25) — the decisive
                # measurement for the chop-completion question. A chop option
                # holds attack for max_option_steps(25) x action_repeat(2) =
                # 50 game ticks; a log takes ~8 ticks with an iron axe but
                # ~60 BAREHANDED. The env grants the axe to the INVENTORY,
                # which is not the same as holding it, and the macro action
                # space has no equip action — so if it starts barehanded every
                # chop times out just short, forever. Observe it rather than
                # guess (also surfaced via `info`, never the obs tensor).
                from minerl.herobraine.hero.handlers.agent.observations \
                    .equipped_item import EquippedItemObservation
                # GROUND TRUTH FOR "IS A MENU COVERING THE SCREEN"
                # (2026-08-02). The pixel heuristic this replaces asked "is
                # the centre of the frame achromatic mid-grey?" — which is
                # true of the inventory panel AND of STONE, COBBLESTONE and
                # ANDESITE (all score 1.000 on it). Underground that misfired
                # on ~84% of frames, and because a "GUI" frame forces
                # `_atk = False` and RESETS the attack-run counter, the agent
                # could never accumulate the ~80 ticks a barehanded log needs.
                # The whole chop diagnosis was reading that corrupted flag.
                from minerl.herobraine.hero.handlers.agent.observations \
                    .is_gui_open import IsGuiOpen
                return HumanControlEnvSpec.create_observables(self) + [
                    handlers.ObserveFromFullStats("mine_block"),
                    IsGuiOpen(),
                    ObservationFromCurrentLocation(),
                    # NOTE: life is pinned at MAX_LIFE by a MineRL key-path
                    # bug — do NOT derive damage or death from it. The two
                    # ground-truth counters below are what actually move.
                    ObservationFromLifeStats(),
                    # GROUND TRUTH FOR BEING HURT AND DYING (2026-07-26).
                    # The creature-perception vocabulary has never produced a
                    # fact — not because mobs are absent (a skeleton has
                    # already KILLED the agent in a live run; difficulty is
                    # hard-coded HARD) but because nothing measurable ever
                    # told the agent it was hurt. Without a felt consequence
                    # there is nothing for "creature" to MEAN. These give the
                    # causal-fact path a real event to attach meaning to.
                    handlers.ObserveFromFullStats("damage_taken"),
                    handlers.ObserveFromFullStats("deaths"),
                    # GROUND TRUTH THAT A CRAFT HAPPENED (2026-07-26). The
                    # symbolic craft actions are unusable in this fork (the
                    # Java backend closes the socket on unknown commands), so
                    # crafting is GUI-only — and a GUI action that silently
                    # no-ops is indistinguishable from the agent never trying
                    # unless the OUTCOME is observable. 976 per-item counters;
                    # a craft either increments one or it did not happen.
                    handlers.ObserveFromFullStats("craft_item"),
                    EquippedItemObservation(
                        items=(["air", "stone_axe", "wooden_axe"]
                               + ([_tool] if _tool else [])),
                        mainhand=True, offhand=False, armor=False)]

            def create_agent_start(self):
                # NO TOOL -> no SimpleInventoryAgentStart at all, so nothing
                # is re-granted on rejoin. The agent starts with empty hands
                # and whatever it later holds it must have obtained itself.
                _start = super().create_agent_start()
                # MULTIPLAYER IDENTITY (2026-08-17). Every client used to
                # join the server as the same player, because provisioning
                # rewrote the template's per-agent
                # `<Name>MineRLAgent{{agent_index}}` to a fixed "SkyBot".
                # That is invisible while only env 0 is remote, and fatal the
                # moment a second one is: Minecraft resolves a duplicate
                # login by kicking the incumbent, so N clients would evict
                # each other in a loop forever. EnvServer.java reads exactly
                # this handler (getMultiplayerUsername -> setUsername), so it
                # is the supported way to give each client its own identity.
                if _agent_name:
                    _start = _start + [
                        handlers.MultiplayerUsername(name=_agent_name)]
                if not _tool:
                    return _start
                return _start + [
                    handlers.SimpleInventoryAgentStart([
                        dict(type=_tool, quantity=1)])]

            def create_server_world_generators(self):
                return [handlers.DefaultWorldGenerator(
                    force_reset="true",
                    generator_options=TREECHOP_WORLD_GENERATOR_OPTIONS)]

            def create_server_quit_producers(self):
                if _lifelong:
                    # 64-log success only — no time-up quit (never-ending world)
                    return [handlers.ServerQuitWhenAnyAgentFinishes()]
                return [
                    handlers.ServerQuitFromTimeUp(8000 * MS_PER_STEP),
                    handlers.ServerQuitWhenAnyAgentFinishes()]

            def create_server_initial_conditions(self):
                # Frozen daytime: constant lighting for the pixel WM + VLM.
                conds = [
                    handlers.TimeInitialCondition(
                        allow_passage_of_time=False)]
                if _remote:
                    # <RemoteServer>ip:port</RemoteServer> — the client joins
                    # this multiplayer server instead of creating a world
                    # (EnvServer.loadOrCreateWorld -> connectToServer). JAXB
                    # schema order is Time, Weather, RemoteServer,
                    # AllowSpawning — keep it between Time and Spawning.
                    from minerl.herobraine.hero.handlers.server.world import (
                        RemoteServer)
                    conds.append(RemoteServer(_remote))
                conds.append(
                    handlers.SpawningInitialCondition(allow_spawning=True))
                return conds

        spec_env = TreechopFixed().make()
        if self._lifelong:
            # no TimeLimit backstop -> the world genuinely never truncates
            return spec_env
        # spec.make() applies no TimeLimit; the mission XML's server timer is
        # the binding terminator — this backstop is parity insurance only.
        from gym.wrappers import TimeLimit
        return TimeLimit(spec_env, max_episode_steps=8000)

    # Non-solid "blocks" Minecraft still reports in mine_block. Breaking into
    # air, or displacing a fluid, is not an achievement — see _mine_counts.
    _NON_BLOCKS = frozenset({
        "air", "cave_air", "void_air", "water", "flowing_water",
        "lava", "flowing_lava", "fire", "bubble_column",
    })

    # ---- EXTERNAL CURIOSITY: territory coverage + damage/death events -----
    @staticmethod
    def _craft_counts(raw_obs) -> Dict[str, int]:
        """Per-item craft counters from ObserveFromFullStats("craft_item").

        Same nested-group shape as mine_block: raw_obs["craft_item"][item].
        Returns only nonzero counters — 976 zero entries per step would be
        noise. Monotonic within a world, restarts at 0 on a new world (the
        reset()-side re-baseline handles that, as with _mine_prev).
        """
        out: Dict[str, int] = {}
        g = raw_obs.get("craft_item") if isinstance(raw_obs, dict) else None
        if not isinstance(g, dict):
            return out
        for k, v in g.items():
            try:
                n = int(np.asarray(v).flatten()[0])
            except Exception:
                continue
            if n > 0:
                out[str(k)] = n
        return out

    @staticmethod
    def _read_stat(raw_obs, key: str) -> Optional[float]:
        """Read an ObserveFromFullStats counter, which nests as a GROUP.

        `_read_scalar` walks raw_obs[k] -> location_stats -> life_stats and so
        will NOT find these: the handler nests them as
        `raw_obs["damage_taken"]["damage_taken"]`. Silently returning None here
        would look exactly like "the agent was never hurt", which is the class
        of false reading that has repeatedly cost this project days.
        """
        if not isinstance(raw_obs, dict):
            return None
        g = raw_obs.get(key)
        if isinstance(g, dict):
            g = g.get(key)
        if g is None:
            return None
        try:
            return float(np.asarray(g).flatten()[0])
        except Exception:
            return None


    # ---- PROPRIOCEPTION (2026-07-27, rung 0) ------------------------------
    # WHAT IT IS LIKE TO BE ME RIGHT NOW, as a small fixed normalized vector.
    #
    # WHY THIS EXISTS: the policy saw ONLY pixels plus a goal vector. Its own
    # hunger, health, depth, whether it was holding anything, and whether it
    # was standing in a menu were all invisible to it. An agent cannot learn
    # to eat when hungry, flee when hurt, or close a screen it cannot feel.
    # Every survival behaviour the project wants was unlearnable for want of
    # a sense, not for want of training.
    #
    # DELIBERATELY SENSES, NOT RULES. Nothing here says hunger is bad, or
    # that menus should be closed, or that being underground is dangerous.
    # It reports state; consequences already exist in the world (starving
    # kills, damage kills, a menu wastes the only resource the agent
    # actually has, which is time). Meaning is earned from experience, the
    # same way the tool-meaning subsystem earned "iron_axe enables
    # break_spruce_log".
    #
    # ENVIRONMENT-AGNOSTIC by construction: every field is either a [0,1]
    # normalized scalar or a 0/1 flag, and a field the env cannot supply
    # reports its neutral value rather than crashing. The same vector shape
    # works on a fresh world, on a live multiplayer server, and in Crafter.
    # `swing` (2026-08-01): HOW LONG HAVE I BEEN SWINGING — the current
    # unbroken attack streak, normalized. Added because breaking a block is
    # the one thing the agent does that takes SUSTAINED effort, and it was
    # completely invisible to the policy: attack_run was computed and printed
    # as a diagnostic but never entered any observation. Holding attack on a
    # log is ~60 ticks barehanded (the granted iron axe cannot be selected —
    # TREECHOP_MACROS has no hotbar keys), while the agent's measured best is
    # 10-21 consecutive attack macros = 20-42 ticks. It was being asked to
    # hold a button through ~60 ticks of ZERO feedback, with no way to sense
    # that anything was accumulating.
    #
    # A SENSE, NOT A REWARD, and deliberately so. It enters proprioception,
    # never the reward stream, so it CANNOT be farmed: swinging at nothing
    # raises this number and pays exactly zero. What it buys is that the
    # world model can now predict "the streak was long and then a block
    # broke", which makes the break PREDICTABLE — and prediction is what
    # curiosity is paid for. Nothing here says trees matter, or that swinging
    # is good; that still has to be earned.
    ATTACK_RUN_SCALE = 100.0        # neutral scale, not a threshold
    # `pitch` (2026-08-02): WHERE MY HEAD IS POINTING, normalized from
    # Minecraft's [-90, +90] to [0, 1] (0 = straight up, 0.5 = level,
    # 1 = straight down). Head orientation is proprioception in the most
    # literal sense — humans know where they are looking without seeing
    # themselves — and the agent had no access to it at all, which is why it
    # could sit clamped at -90 swinging at sky with nothing to tell it so.
    # MOVED + HEADING added 2026-08-23. The body could feel hunger, health,
    # depth, whether a menu was covering the screen, how long it had been
    # swinging and where it was looking — but NOT whether it was actually
    # going anywhere. Every recorded failure of this agent is a failure to
    # move: staring at the sky, sitting in a villager's trade menu for 10,149
    # steps, holding attack against a trunk it could not reach. "Am I moving"
    # is the most conspicuous missing proprioceptive fact, and the adapter
    # already reads xpos/zpos every step for coverage.
    #
    # STILL PROPRIOCEPTION, NOT MEANING: these describe the body, name nothing
    # in the world, and so leave the earned-meaning principle intact.
    PROPRIO_KEYS = ("food", "saturation", "life", "depth",
                    "has_tool", "gui_open", "hurt_recent", "carrying",
                    "swing", "pitch", "moved", "head_sin", "head_cos")
    # blocks of horizontal travel per agent step that reads as "full speed".
    # Minecraft walks ~4.3 blocks/s; at action_repeat 2 (0.1s) that is ~0.43,
    # sprinting ~0.56 — so 0.5 puts a normal walk near the top of the range
    # without pinning it there.
    # DERIVED FROM action_repeat SINCE 2026-09-01 (see _move_scale). A fixed
    # 0.5 at repeat 4 saturates on every ordinary walk (~0.86 blocks/step),
    # so `moved` reads 1.0 whether the agent is strolling or sprinting and
    # stops discriminating — the exact fate of a normalized sense whose
    # denominator is calibrated for a different step length.
    MOVE_SCALE = 0.5            # per 2 ticks; scaled by action_repeat/2
    PROPRIO_DIM = len(PROPRIO_KEYS)

    def _proprio(self, world: Dict[str, Any]) -> np.ndarray:
        """Fixed-order normalized self-state. Never raises; unknown -> neutral."""
        v = np.zeros(self.PROPRIO_DIM, dtype=np.float32)
        try:
            f = world.get("food")
            v[0] = 1.0 if f is None else float(np.clip(f / 20.0, 0.0, 1.0))
            sat = world.get("saturation")
            v[1] = 1.0 if sat is None else float(np.clip(sat / 20.0, 0.0, 1.0))
            lf = world.get("life")
            v[2] = 1.0 if lf is None else float(np.clip(lf / 20.0, 0.0, 1.0))
            # DEPTH, not raw y: "how far underground am I" transfers across
            # worlds where sea level differs. 0 = surface or above.
            y = world.get("ypos")
            v[3] = 0.0 if y is None else float(
                np.clip((63.0 - float(y)) / 63.0, 0.0, 1.0))
            mh = str(world.get("mainhand") or "none")
            v[4] = 0.0 if mh in ("none", "air", "") else 1.0
            v[5] = 1.0 if world.get("gui_open") else 0.0
            v[6] = 1.0 if world.get("damage") else 0.0
            # SOMETHING IN THE BAG — the difference between having gathered
            # and not having gathered, without naming any particular item.
            v[7] = float(np.clip(float(world.get("carrying", 0.0)) / 8.0,
                                 0.0, 1.0))
            # SUSTAINED EFFORT, made feelable. See ATTACK_RUN_SCALE above.
            v[8] = float(np.clip(float(world.get("attack_run", 0.0))
                                 / self.ATTACK_RUN_SCALE, 0.0, 1.0))
            _pt = world.get("pitch")
            v[9] = (0.5 if _pt is None else
                    float(np.clip((float(_pt) + 90.0) / 180.0, 0.0, 1.0)))
            # AM I MOVING (2026-08-23)
            _mv = world.get("moved")
            # getattr, NOT attribute access. `_proprio` is wrapped in a broad
            # `except Exception: pass`, so an AttributeError here does not
            # raise — it silently leaves v[10:] at zero, i.e. `moved`,
            # `pitch` and both heading fields all read as neutral-but-wrong
            # with nothing logged. Anything reaching this method without a
            # full __init__ (a test double, an older pickled adapter) would
            # lose four senses at once and look merely idle. The class
            # constant is the correct fallback: it IS _move_scale at
            # action_repeat 2, the calibration it was written for.
            _ms = getattr(self, "_move_scale", None) or self.MOVE_SCALE
            v[10] = (0.0 if _mv is None else
                     float(np.clip(float(_mv) / _ms, 0.0, 1.0)))
            # WHICH WAY AM I FACING. sin/cos rather than raw degrees so the
            # wrap at 360->0 is continuous rather than a cliff.
            #
            # RESCALED TO [0,1] like every other entry: this vector's standing
            # invariant is that all of it is normalized to [0,1] (asserted in
            # _proprioception_smoke), and a raw sine would have been the only
            # member on a different scale. Unknown reads 0.5/0.5 — the exact
            # convention `pitch` above already uses for "no reading".
            _yw = world.get("yaw")
            v[11] = v[12] = 0.5
            if _yw is not None:
                _r = float(np.radians(float(_yw)))
                v[11] = float(np.sin(_r)) * 0.5 + 0.5
                v[12] = float(np.cos(_r)) * 0.5 + 0.5
        except Exception:
            pass
        return v

    @staticmethod
    def _read_scalar(raw_obs, *keys) -> Optional[float]:
        """Pull a scalar out of the (possibly nested) MineRL obs dict."""
        if not isinstance(raw_obs, dict):
            return None
        for k in keys:
            v = raw_obs.get(k)
            if v is None and isinstance(raw_obs.get("location_stats"), dict):
                v = raw_obs["location_stats"].get(k)
            if v is None and isinstance(raw_obs.get("life_stats"), dict):
                v = raw_obs["life_stats"].get(k)
            if v is not None:
                try:
                    return float(np.asarray(v).flatten()[0])
                except Exception:
                    continue
        return None

    def _world_events(self, raw_obs) -> Dict[str, Any]:
        """Territory novelty + damage/death, from the new location/life stats.

        COVERAGE is count-based novelty over discretised (x,z) cells: a cell
        the agent has rarely stood in pays ~1, a well-trodden one pays ~0.
        This is the drive that keeps the WORLD able to surprise a lifelong
        agent — prediction-error curiosity dies once the current view is
        mastered, and nothing else pushed the agent to new ground.

        Returned via `info` and consumed as INTRINSIC reward by the loop, so
        it never becomes a replay reward label.
        """
        out: Dict[str, Any] = {}
        x = self._read_scalar(raw_obs, "xpos")
        z = self._read_scalar(raw_obs, "zpos")
        y = self._read_scalar(raw_obs, "ypos")
        life = self._read_scalar(raw_obs, "life")
        if x is not None and z is not None:
            cell = (int(x // self.coverage_cell), int(z // self.coverage_cell))
            n = self._visits.get(cell, 0)
            self._visits[cell] = n + 1
            if len(self._visits) > 50000:            # bound the memory
                self._visits.pop(next(iter(self._visits)))
            # territory changes EVERY step, so it needs a step-paced flush of
            # its own — the event counter can sit still for hours (review
            # finding 2026-08-11). Self-throttling; event=False so this does
            # not consume the event budget.
            self._visit_writes = int(getattr(self, "_visit_writes", 0)) + 1
            self._save_break_memory(event=False)
            out["coverage"] = float(1.0 / np.sqrt(1.0 + n))
            out["cell"] = cell
            out["cells_seen"] = len(self._visits)
            # WHERE IS IT LOOKING. Minecraft clamps pitch to [-90, +90];
            # -90 is straight up, +90 straight down. A value pinned at a
            # clamp means the camera cannot travel further that way, which
            # from outside looks exactly like "it is stuck looking up".
            _p = self._read_scalar(raw_obs, "pitch")
            if _p is not None:
                out["pitch"] = float(_p)
            # heading too (infra #38): the episodic-memory bearing sense is
            # RELATIVE to where the agent faces, which needs yaw
            _yw = self._read_scalar(raw_obs, "yaw")
            if _yw is not None:
                out["yaw"] = float(_yw)
            if y is not None:
                out["ypos"] = y
            # ABSOLUTE POSITION (2026-08-02). Only the derived `cell` and
            # `ypos` escaped this function, so nothing downstream could say
            # WHERE the agent actually is — and "it is standing inside the
            # server's spawn-protection radius, where breaking is forbidden
            # for non-ops" is indistinguishable from "its swings miss" unless
            # you can read x/z. Two floats, already computed above.
            out["xpos"] = float(x)
            out["zpos"] = float(z)
            # KEY-NAME ALIAS — repairs FIVE silently dead consumers
            # (2026-08-11). This function has always published `xpos`/`ypos`/
            # `zpos`, but every downstream reader was written against
            # `x`/`y`/`z`, so each got None and failed CLOSED without a word:
            #   * the episodic sighting gate (infra/stack.py ~319 requires a
            #     non-None position) — which is why the run reported
            #     "sighting never" for 336k steps for ALL NINE categories,
            #     including `dirt` at a 42.6% fovea-positive rate. MEASURED:
            #     zero " @(x,z)" markers in the entire run log, while the
            #     status line printed "Position: x=+36.3 ..." on the same
            #     step because THAT reader uses xpos/ypos/zpos.
            #   * the episodic BEARING proprio sense (loop _augment_proprio)
            #   * _memory_pull_phi (so that term could never pay at all)
            #   * the demonstration landmark (demos were witnessed, none
            #     stored)
            #   * the live viewer's position payload
            # Aliasing at the SOURCE fixes every consumer at once, including
            # any not yet found, and cannot disturb the xpos/ypos/zpos
            # readers. A pure measurement repair: no new reward term.
            out["x"] = out["xpos"]
            out["z"] = out["zpos"]
            # HOW FAR DID THE BODY ACTUALLY TRAVEL since the last step. Kept
            # here rather than in _proprio so that stays a pure function of
            # `world`; this method is already the stateful one (it owns the
            # coverage visit counts).
            _pxz = getattr(self, "_prev_xz", None)
            out["moved"] = (0.0 if _pxz is None else
                            float(np.hypot(float(x) - _pxz[0],
                                           float(z) - _pxz[1])))
            self._prev_xz = (float(x), float(z))
            out["y"] = float(out.get("ypos", 0.0) or 0.0)
        # WHAT IS IN HAND — decides the chop-completion question (see
        # create_observables). "air"/none => barehanded => a log needs ~60
        # ticks but the chop option only holds attack for 50.
        try:
            eq = raw_obs.get("equipped_items") if isinstance(raw_obs, dict) \
                else None
            if isinstance(eq, dict):
                mh = eq.get("mainhand")
                dmg = None
                if isinstance(mh, dict):
                    dmg = mh.get("damage")
                    mh = mh.get("type")
                if mh is not None:
                    name = str(mh.decode() if isinstance(mh, bytes) else mh)
                    out["mainhand"] = name
                    # TOOL WEAR — the ground-truth EXPERIENCE behind the
                    # visual `tool_worn` predicate. The agent sees a shrinking
                    # durability bar; this is what actually happened. Meaning
                    # ("using it wears it out", "it eventually breaks") is
                    # learned from these events, never declared.
                    if dmg is not None:
                        try:
                            d = float(np.asarray(dmg).flatten()[0])
                            out["tool_damage"] = d
                            prev = self._prev_tool_damage.get(name)
                            if prev is not None and d > prev:
                                out["tool_wore"] = d - prev
                            self._prev_tool_damage[name] = d
                        except Exception:
                            pass
                    # the tool the agent HAD is gone from its hand
                    if (self._prev_mainhand
                            and self._prev_mainhand not in ("none", "air")
                            and name in ("none", "air")):
                        out["tool_lost"] = self._prev_mainhand
                    self._prev_mainhand = name
        except Exception:
            pass

        # ---- MAINHAND, DERIVED FROM INVENTORY (2026-07-26) ---------------
        # `equipped_items` above is NOT POPULATED by MineRL 1.0's MCP-Reborn
        # bridge — it reads "none" no matter what the agent is holding. That
        # single unpopulated field cost a full day: it was read as "the agent
        # chops BAREHANDED", which produced a ~60-tick break estimate, an
        # option-budget change made for the wrong reason, and a confident
        # (wrong) conclusion that the axe was unreachable.
        #
        # The truth, verified from the installed sources and from the agent's
        # own rendered POV (HUD shows the axe in the HIGHLIGHTED hotbar slot,
        # with a used durability bar): `SimpleInventoryAgentStart` enumerates
        # items into slots {0: item, ...} (start.py:78-80), Java writes index 0
        # to PlayerInventory.mainInventory[0], index 0 IS hotbar slot 0, and
        # `currentItem` initialises to 0. Nothing in this env ever changes the
        # selected slot: TREECHOP_MACROS has no hotbar.N and no mouse wheel.
        # So the granted tool is HELD from spawn.
        #
        # We therefore derive the hand from the inventory rather than trusting
        # the dead sensor — but ONLY as a fallback, so that if a future MineRL
        # does populate equipped_items, the real reading wins.
        if not out.get("mainhand") or out.get("mainhand") in ("none", "air"):
            try:
                inv = raw_obs.get("inventory") if isinstance(raw_obs, dict) \
                    else None
                tool = self._start_tool
                if isinstance(inv, dict) and tool:
                    qty = inv.get(tool)
                    if qty is not None:
                        n = float(np.asarray(qty).flatten()[0])
                        if n > 0:
                            out["mainhand"] = str(tool)
                            out["mainhand_derived"] = True
                            # DO NOT write self._prev_mainhand here. That is
                            # the DEAD sensor's state, and feeding it this
                            # value created a per-step false-loss loop:
                            #   step N: dead path sees prev="iron_axe",
                            #           current="none" -> emits tool_lost,
                            #           sets prev="none"
                            #   step N: this branch sets prev="iron_axe" again
                            #   step N+1: identical -> tool_lost AGAIN
                            # It minted `iron_axe BROKE from use` ~7000 times
                            # and polluted the knowledge graph with a fact that
                            # never happened. The derived hand keeps its OWN
                            # history so the two paths cannot interfere.
                            if (self._prev_derived_hand
                                    in (None, "none", "air")):
                                self._prev_derived_hand = str(tool)
                        elif self._prev_derived_hand not in (None, "none",
                                                            "air"):
                            # the tool left the inventory: a REAL loss, and
                            # judged against the DERIVED history only.
                            out["tool_lost"] = self._prev_derived_hand
                            out["mainhand"] = "none"
                            self._prev_derived_hand = "none"
                # ---- NO START TOOL -> THE HAND WAS UNKNOWABLE -------------
                # (2026-08-17) The block above answers only "is the SPAWN
                # tool still in the bag?". With `start_tool: null` — the
                # shipped skybot setting — `tool` is None, so the whole
                # branch was skipped and `mainhand` could never be anything
                # but "none". Measured: hand=none in 160/160 samples across
                # a 7-hour run while the agent carried 8+ items and spent
                # 11% of its actions on `use`. It was not empty-handed; it
                # was blind to its own hand, which is worse — every
                # consumer (the proprio has_tool sense, the chop-budget
                # diagnosis, tool_worn) was reading a constant.
                #
                # A flat inventory (item -> count, no slots) cannot name the
                # selected slot, so this uses the one ground truth that
                # exists: an item a `use` actually CONSUMED was, at that
                # moment, in the hand. Nothing in TREECHOP_MACROS changes
                # the selected slot, so it stays there while any remains.
                # Retrospective and honest about it, exactly like the reach
                # sense's "evidence" mode: it says "I recently placed this
                # and still have some", never "I am definitely holding it".
                if (not out.get("mainhand")
                        or out.get("mainhand") in ("none", "air")):
                    _lp = self._last_placed_item
                    if _lp and isinstance(inv, dict):
                        _q = inv.get(_lp)
                        if _q is not None:
                            try:
                                if float(np.asarray(_q).flatten()[0]) > 0:
                                    out["mainhand"] = str(_lp)
                                    out["mainhand_derived"] = True
                            except Exception:
                                pass
            except Exception:
                pass
        # ---- HUNGER (2026-07-27, rung-0 survival perception) --------------
        # `_read_scalar` already searches life_stats, so these cost one call
        # each — but NOTHING read them, so the agent could not perceive
        # itself starving. It cannot learn to eat what it cannot feel needing.
        # Surfaced as PERCEPTION only: no hunger reward, no "eat when low"
        # rule. Starving to death is already a real consequence (deaths is
        # observed and ends the episode); what was missing was the sense that
        # makes that consequence learnable in advance. Meaning is earned here
        # exactly as `iron_axe enables break_spruce_log` was.
        food = self._read_scalar(raw_obs, "food_level", "food")
        if food is not None:
            out["food"] = food
            if self._prev_food is not None and food < self._prev_food - 0.5:
                out["hunger_fell"] = float(self._prev_food - food)
            self._prev_food = food
        sat = self._read_scalar(raw_obs, "saturation")
        if sat is not None:
            out["saturation"] = sat
        if life is not None:
            out["life"] = life
            if self._prev_life is not None:
                d = life - self._prev_life
                if d < -0.5:
                    out["damage"] = float(-d)        # mobs/fall/etc. HURT
                if life <= 0.0 < self._prev_life:
                    out["died"] = True
            self._prev_life = life
        # GROUND-TRUTH hurt/death, from monotonic stat counters (2026-07-26).
        # The life-derived path above is kept but is UNRELIABLE: life reads
        # pinned at MAX_LIFE, so `damage`/`died` above have never fired even
        # though a skeleton has demonstrably killed this agent. These counters
        # only ever increase, so a positive delta is an unambiguous event.
        _dmg = self._read_stat(raw_obs, "damage_taken")
        if _dmg is not None:
            if self._prev_damage_taken is not None and _dmg > self._prev_damage_taken:
                out["damage"] = float(_dmg - self._prev_damage_taken)
                out["damage_source"] = "stat"
            self._prev_damage_taken = _dmg
        _dea = self._read_stat(raw_obs, "deaths")
        if _dea is not None:
            if self._prev_deaths is not None and _dea > self._prev_deaths:
                # STAT-SYNC GUARD (2026-08-08): a fresh join reads 0 until
                # the server pushes the player's PERSISTENT totals — on a
                # server whose SkyBot has 73 lifetime deaths that sync is a
                # +73 jump, not a death. A real death increments by 1
                # (rarely 2 across a slow read). MEASURED: the unguarded
                # check turned the sync into a ~57s die->rejoin->sync loop,
                # 16 rejoins in 30 minutes with hp pinned at 1.00.
                if (_dea - self._prev_deaths) <= 2.5:
                    out["died"] = True
            self._prev_deaths = _dea
        return out

    @staticmethod
    def _inv_counts(raw_obs) -> Dict[str, int]:
        """Per-item inventory counts (bare names), for placement detection."""
        try:
            inv = raw_obs.get("inventory", {}) if isinstance(raw_obs, dict) \
                else {}
            out = {}
            for k, v in inv.items():
                c = int(np.asarray(v).flatten()[0])
                if c > 0:
                    out[str(k).split(".")[-1]] = c
            return out
        except Exception:
            return {}

    @staticmethod
    def _count_logs(raw_obs) -> int:
        """Total log items in the inventory observation (oak_log, birch_log,
        ... — any '*log*' key)."""
        try:
            inv = raw_obs.get("inventory", {}) if isinstance(raw_obs, dict) \
                else {}
            total = 0
            for k, v in inv.items():
                if "log" in str(k):
                    total += int(np.asarray(v).flatten()[0])
            return total
        except Exception:
            return 0

    # Instant-break (1-tick) vegetation — no hold required, so it's not a
    # real block-breaking achievement. Kept BELOW the goal-discovery spike
    # threshold so it never mints a skill.
    _TRIVIAL_BREAKS = frozenset({
        "grass", "tall_grass", "fern", "large_fern", "vine", "seagrass",
        "tall_seagrass", "kelp", "kelp_plant", "dead_bush", "sugar_cane",
        "poppy", "dandelion", "oxeye_daisy", "blue_orchid", "allium",
        "azure_bluet", "cornflower", "lily_of_the_valley", "sunflower",
        "lilac", "rose_bush", "peony", "sweet_berry_bush", "wheat",
        "snow", "torch", "wall_torch",
    })

    # GROUND the agent is STANDING ON (2026-07-27, stall assessment). These
    # fell through `_break_reward`'s 1.0 default, which made DIGGING the
    # best-paid reachable behaviour: measured over the stalled run, all four
    # earning segments were ground breaks and one episode scored 6.00 by
    # sinking a vertical shaft through six new block types. The agent buried
    # itself in a pit and there was no tree in frame for the rest of the run.
    # This is the same argument the file already makes for leaves below —
    # "the easy thing to hit" must not out-pay the goal — except terrain is
    # EASIER than leaves: it requires no aim at all, only looking down.
    _GROUND_BREAKS = frozenset({
        "dirt", "coarse_dirt", "grass_block", "podzol", "mycelium",
        "gravel", "sand", "red_sand", "clay", "soul_sand",
        "stone", "cobblestone", "sandstone", "andesite", "diorite",
        "granite", "netherrack", "snow_block", "moss_block",
    })

    @classmethod
    def _break_reward(cls, btype) -> float:
        """First-break reward, WHITELISTED to wood and ores (2026-08-08).
        Logs (the goal) pay big, ores pay well, everything else pays 0.0 —
        see the note at the bottom for why the old token tiers had to go.

        NAMESPACE-ROBUST (fix 2026-08-02). The tier sets below hold BARE block
        names, and the membership tests are exact — so a namespaced stat key
        ("minecraft.dirt") missed every set and fell through to the 1.0
        default. That is 6.7x the intended ground tier AND above the 0.9
        goal-discovery spike threshold, which makes digging the best-paid
        REACHABLE behaviour (it needs no aim, only looking down, and can be
        done while walking) and mints a junk `discovered_N` goal for every
        terrain type. _mine_counts already strips the namespace at source; this
        is the second line of defence, because a tier table matched by exact
        string must never depend on an upstream caller having normalised first.
        """
        b = str(btype).split(".")[-1]
        if "log" in b:
            # class attribute so a per-instance override (set from config in
            # __init__) is picked up without changing this classmethod's shape
            return float(getattr(cls, "LOG_BREAK_REWARD", 5.0))
        # ORES: the other block family whose breaks are genuinely goal-
        # relevant (mining progression). Sub-log tier, still well above the
        # goal-discovery spike so a first ore mints a real goal.
        if "ore" in b or b == "ancient_debris":
            return 10.0
        # EVERYTHING ELSE PAYS ZERO (2026-08-08, user directive). The tier
        # ladder (0.15 ground / 0.3 leaves / 1.0 default) was built to keep a
        # little learning signal on non-goal blocks — and MEASURED live, that
        # little signal was still enough: placed directly in front of logs
        # with an axe, the bot abandoned a started chop for snow and dirt,
        # because instant cheap breaks beat a delayed 20.0 under temporal
        # discounting every time. Non-wood/ore breaks keep their INTRINSIC
        # novelty (first-of-type surprise, already mastery-habituated) but
        # earn no extrinsic income at all: wood and ore are the only breaks
        # the WORLD now rewards.
        return 0.0

    def _gui_open(self, obs) -> bool:
        """True when a menu is covering the world.

        GROUND TRUTH FIRST (2026-08-02). MineRL exposes `IsGuiOpen`, and it is
        now requested in create_observables, so the honest answer comes from
        the game rather than from the frame.

        WHY THE OLD PIXEL TEST HAD TO GO. It asked "is the centre of the frame
        achromatic and mid-grey?". Measured against real block colours:

            inventory panel 1.000 | stone 1.000 | cobblestone 1.000
            andesite        1.000 | dirt  0.000 | grass       0.000

        i.e. it could not distinguish the inventory panel from a stone wall.
        Underground it reported "in a menu" on ~84% of steps, and since a GUI
        frame forces `_atk = False` and RESETS the attack-run counter, the
        agent could never accumulate the ~80 ticks a barehanded log needs.
        Every "TIMING"/"AIM" verdict from the chop diagnosis was reading that.

        THE FALLBACK IS STILL NEEDED, and is deliberately conservative.
        `equipped_items` is precedent: the handler exists, MineRL's MCP-Reborn
        bridge never populates it, and reading its empty value as truth cost a
        day and produced a wrong conclusion. So if `isGuiOpen` is absent we
        fall back to pixels — but with the discriminator the old test lacked:
        FLATNESS. The inventory panel is a near-uniform fill; stone is
        textured. Requiring low per-channel variance separates them, and when
        the region is textured we answer False (the safe direction: a missed
        menu costs some wasted curiosity, a false menu silently destroys the
        agent's ability to break anything at all).
        """
        # 1. ground truth, when the bridge actually supplies it
        if isinstance(obs, dict) and "isGuiOpen" in obs:
            try:
                v = np.asarray(obs["isGuiOpen"]).flatten()
                if v.size:
                    if not self._gui_src_logged:
                        self._gui_src_logged = True
                        logger.info("gui detection: ground-truth isGuiOpen")
                    return bool(int(v[0]))
            except Exception:
                pass
        # 2. pixel fallback: achromatic + mid-grey + FLAT
        try:
            pov = obs.get("pov") if isinstance(obs, dict) else obs
            a = np.asarray(pov)
            if a.ndim != 3 or a.shape[2] < 3:
                return False
            if not self._gui_src_logged:
                self._gui_src_logged = True
                logger.warning(
                    "gui detection: isGuiOpen NOT supplied — using the pixel "
                    "fallback (achromatic+midgrey+flat)")
            h, w = a.shape[0], a.shape[1]
            box = a[int(0.50 * h):int(0.68 * h), int(0.34 * w):int(0.66 * w)]
            if box.size == 0:
                return False
            r = box[..., 0].astype(np.int16)
            g = box[..., 1].astype(np.int16)
            b = box[..., 2].astype(np.int16)
            achromatic = ((np.abs(b - g) < 10) & (np.abs(g - r) < 10)
                          & (np.abs(b - r) < 10))
            midgrey = (b > 110) & (b < 225)
            if float((achromatic & midgrey).mean()) <= 0.5:
                return False
            # THE NEW TEST: a GUI panel is a flat fill; stone is speckled.
            flat = max(float(r.std()), float(g.std()), float(b.std()))
            return flat < 6.0
        except Exception:
            return False

    @staticmethod
    def _mine_counts(raw_obs) -> Dict[str, int]:
        """Per-block-type break counts from the mine_block stats observation
        (only types with count > 0, so the dict stays small)."""
        try:
            mb = raw_obs.get("mine_block", {}) if isinstance(raw_obs, dict) \
                else {}
            out = {}
            for k, v in mb.items():
                c = int(np.asarray(v).flatten()[0])
                if c <= 0:
                    continue
                # NON-BLOCK FILTER (2026-07-25): Minecraft reports mine_block
                # entries for non-solid "blocks" (air is logged when a break
                # completes into open space, fluids when they're displaced).
                # These are not achievements: they minted the junk `break_air`
                # goal AND paid a first-break bonus, which pushed the step
                # reward over the 0.9 spike threshold and manufactured a
                # `discovered_N` goal + skill. Filter at the EARLIEST point so
                # reward, achievements, effect_key and goal all die together.
                name = str(k).split(".")[-1]
                if name in MineRLEnvAdapter._NON_BLOCKS:
                    continue
                # EMIT THE BARE NAME (fix 2026-08-02). The namespace was
                # stripped for the _NON_BLOCKS test and then thrown away, so
                # every downstream consumer — the reward tier, the achievement
                # key, the grounded effect key that names a skill, and
                # breaks_by_type — saw the raw key. One normalisation, at the
                # single point the counts enter the system, so those four can
                # never disagree about what a block is called.
                out[name] = c
            return out
        except Exception:
            return {}

    def _break_decay(self, btype) -> float:
        """Familiarity multiplier for this block type, in (0, 1].

        1.0 for a type never broken; falls as 1/(1 + n/scale) with the count
        of that type already felled. Off (returns 1.0) when scale <= 0.
        """
        if self._break_decay_scale <= 0.0:
            return 1.0
        _n = int(self._breaks_by_type.get(str(btype), 0))
        return 1.0 / (1.0 + (_n / self._break_decay_scale))

    def _save_break_memory(self, force: bool = False,
                           event: bool = True) -> None:
        """Persist the lifelong event + territory memory. Atomic tmp+rename
        so a crash mid-write can never corrupt it.

        THROTTLE SCOPE FIX (2026-08-11, review finding): the every-20 counter
        ticks only on break/place/craft/pickup EVENTS, which was right when
        the payload held only event counts. The payload now also carries
        `cells` (territory visits), and those change EVERY STEP — so hours of
        pure navigation with no events flushed nothing at all, and the newly
        added territory memory was silently lost on restart. Events still
        force a flush every 20; a step-count floor now also flushes a run
        that is exploring but not breaking. `force=True` (shutdown) always
        writes."""
        if not self._break_memory_path:
            return
        if not force:
            # `event` callers bump the event counter; the per-step caller
            # must NOT, or "every 20 events" would silently become
            # "every 20 steps" and write the file constantly.
            if event:
                self._break_mem_dirty += 1
            _steps = int(getattr(self, "_visit_writes", 0))
            _due = (self._break_mem_dirty >= 20
                    or _steps - int(getattr(self, "_last_mem_flush_step", 0))
                    >= 2000)
            if not _due:
                return
            self._last_mem_flush_step = _steps
        self._break_mem_dirty = 0
        try:
            import json as _json
            import os as _os
            _tmp = self._break_memory_path + ".tmp"
            with open(_tmp, "w") as _f:
                _json.dump({"breaks": self._breaks_by_type,
                            "places": self._places_by_type,
                            "crafts": self._crafts_by_type,
                            "pickups": self._pickups_by_type,
                            # territory familiarity (2026-08-10): coverage
                            # paid a fresh windfall per restart because the
                            # visit counts died with the process. getattr:
                            # a missing dict must cost the cells, never the
                            # whole break-memory flush
                            "cells": {f"{k[0]},{k[1]}": int(v)
                                      for k, v in (getattr(self, "_visits",
                                                           None)
                                                   or {}).items()}},
                           _f)
            _os.replace(_tmp, self._break_memory_path)
        except Exception as _e:
            logger.debug("break memory: save failed: %s", _e)

    def clear_break_marks(self) -> None:
        """Re-open the first-break reward tiers (audit fix). In LIFELONG mode
        reset() may not run for days, so `_broken_this_episode` silently became
        once-per-LIFETIME — after the first dirt/leaf/stone of each type,
        re-breaking paid 0 forever while the broadcaster re-armed its goal
        masks every goal_horizon, so re-armed slots could never re-unlock and
        non-log competence decayed toward zero. The loop calls this on the SAME
        goal-horizon cadence as broadcaster.reset(), aligning the env's reward
        horizon with the goal system's evidence horizon. (_mine_baseline is
        per-world and stays untouched.)"""
        self._broken_this_episode = set()
        # craft first-of-type tiers re-arm on the same cadence — a craft the
        # goal system re-targets must be re-earnable, like a re-armed break.
        self._crafted_this_episode = set()

    def _remote_frozen(self, obs: Optional[Dict], action: int) -> bool:
        """Disconnect detector for remote-server mode (see __init__ note).

        Returns True when POV has been byte-identical for _frozen_limit
        consecutive MOTION steps — the static disconnect/death-screen
        signature. Local missions: always False (never even hashes).
        """
        if not self.remote_server or not isinstance(obs, dict):
            return False
        # GRACE PERIOD AFTER A REBUILD (fix 2026-08-05). A booting client sits
        # on a static loading/connecting screen for ~2 minutes, and while it
        # does, position and look are unavailable — which this detector reads
        # as "POV frozen AND state frozen", i.e. a disconnect. It then rebuilds
        # the client, restarting the boot, and the run never gets past it:
        # MEASURED 27 rebuilds at a ~3 minute cadence with the agent never
        # once appearing on the server. The watchdog was consuming the very
        # connection it exists to protect.
        self._steps_since_reset = getattr(self, "_steps_since_reset", 0) + 1
        if self._steps_since_reset < self._frozen_grace:
            return False
        pov = obs.get("pov")
        if pov is None:
            return False
        import hashlib
        digest = hashlib.md5(np.ascontiguousarray(pov)).digest()
        # A FROZEN FRAME IS NOT A DISCONNECT (fix 2026-08-02). Staring at a
        # featureless wall, or being wedged against a block while commanding
        # forward, produces a byte-identical POV indefinitely — and the old
        # test called that a kick and REBUILT THE CLIENT, which is why the
        # agent kept "randomly leaving" the server after standing still.
        #
        # The discriminator: on a real disconnect the server stops applying
        # input entirely, so the agent's OWN STATE freezes too. If it commands
        # a camera move and its pitch/yaw still change, the server is talking
        # to it and the static frame just means there is nothing to see.
        _look = None
        try:
            _look = (self._read_scalar(obs, "pitch"),
                     self._read_scalar(obs, "yaw"),
                     self._read_scalar(obs, "xpos"),
                     self._read_scalar(obs, "zpos"))
        except Exception:
            pass
        _state_frozen = True
        _prev_look = getattr(self, "_prev_look_state", None)
        if _look is not None and _prev_look is not None:
            try:
                _state_frozen = all(
                    (a is None or b is None or abs(float(a) - float(b)) < 1e-6)
                    for a, b in zip(_look, _prev_look))
            except Exception:
                _state_frozen = True
        self._prev_look_state = _look
        if digest == self._prev_pov_digest:
            # only count it when the WORLD is not responding to us at all
            if action in self._motion_macros and _state_frozen:
                self._frozen_count += 1
            elif not _state_frozen:
                self._frozen_count = 0
        else:
            self._frozen_count = 0
        self._prev_pov_digest = digest
        return self._frozen_count >= self._frozen_limit

    def _select_tool_slot(self, obs):
        """DISABLED — PROVEN INEFFECTIVE 2026-07-25, kept as a record.

        Pressing hotbar.1 does NOT equip the granted axe: the item is in
        the inventory but in no hotbar slot (verified live — all 9 hotbar
        keys leave mainhand=none, and slot 36 is out of range so the item
        is not placed at all). Equipping it would need inventory-GUI
        interaction the agent does not have. The chop budget was widened
        instead (max_option_steps 25->40 = 80 ticks > the ~60 a
        barehanded log needs). Do not re-enable without re-measuring
        `Chop budget: mainhand=`.

        Original intent:

        MEASURED 2026-07-25: `mainhand=none` — the agent had been chopping
        BAREHANDED its whole life. `SimpleInventoryAgentStart` places the iron
        axe in inventory slot 0, but a Minecraft item is only *held* once its
        hotbar slot is SELECTED, and the agent's macro action space has no
        hotbar/equip action — so it could never take the axe out. Barehanded a
        log needs ~60 ticks while a chop option holds attack for only 50, so
        every chop timed out a few ticks short, forever. With the axe it is
        ~8 ticks.

        Fix: press `hotbar.1` (select slot 0) a few times at reset. This uses
        the UNDERLYING MineRL action space directly, so the agent's own macro
        space is untouched — action_dim stays 10 and every saved skill policy
        keeps loading. The agent is not being taught to equip; it is being
        handed the tool the environment always intended it to hold.
        """
        try:
            act = self._env.action_space.noop()
            if "hotbar.1" not in act:
                return obs
            act["hotbar.1"] = 1
            for _ in range(2):          # press + settle
                obs, _r, _d, _i = self._env.step(act)
            self._log_count = self._count_logs(obs)
        except Exception as e:
            logger.warning("tool-slot select failed (%s) — agent may be "
                           "barehanded; watch `Chop budget: mainhand=`", e)
        return obs

    def _rebuild(self) -> None:
        """Replace a dead/hung Minecraft client with a fresh one (~90s)."""
        try:
            self._env.close()
        except Exception as e:
            # The old Java process may leak; nothing more we can do from here.
            logger.warning("MineRL close() during rebuild failed: %s", e)
        # the agency detector's previous-frame must not survive a rebuild:
        # diffing the fresh world against a pre-rebuild frame would read the
        # whole scene change as "someone else did something" (2026-08-09)
        self._prev_small_pov = None
        self._env = self._make_underlying()

    # ---- gymnasium API ----------------------------------------------------
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        # agency-detector state never survives a world boundary (review
        # 2026-08-09): diffing a fresh world against the previous one reads
        # the whole scene as "someone else did something"
        self._prev_small_pov = None
        self._prev_agency_state = None
        # a fresh world teleports the body: the first displacement after a
        # reset is meaningless and must read 0, not "sprinted 200 blocks"
        self._prev_xz = None
        if seed is not None:
            try:
                self._env.seed(int(seed))
            except Exception:
                pass  # not all MineRL envs support seeding
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                obs = self._env.reset()
                # a fresh mission means a fresh boot: re-arm the grace period
                # so the frozen-POV watchdog cannot fire on the loading screen
                self._steps_since_reset = 0
                self._frozen_count = 0
                self._prev_pov_digest = None
                self._log_count = self._count_logs(obs)
                # Defensive baseline only. MineRL 1.0 builds a NEW world
                # every reset (MCP-Reborn initMission -> createNewWorld) and
                # Minecraft keeps mine_block per level save, so these counts
                # restart at 0 and this always evaluates to {} in practice.
                # The real once-per-type-per-episode guard is
                # _broken_this_episode below. Kept so the adapter stays
                # correct if a future backend ever reuses a world.
                self._mine_baseline = self._mine_counts(obs)
                self._broken_this_episode: set = set()
                # per-step break deltas: a new world restarts mine_block at 0,
                # so stale high counts must not survive the boundary.
                self._mine_prev = dict(self._mine_baseline)
                # only trust a baseline that actually contains stats; an empty
                # one means the observable was not populated yet.
                self._mine_seeded = bool(self._mine_baseline)
                self._craft_prev = self._craft_counts(obs)
                self._crafted_this_episode = set()
                self._attack_run = 0
                self._frozen_count = 0
                self._prev_pov_digest = None
                return self._pov_to_obs(obs), {
                    "logs": self._log_count,
                    "achievements": {"log": self._log_count}}
            except Exception as e:  # dead client, socket timeout, ...
                last_err = e
                logger.warning(
                    "MineRL reset failed (attempt %d/3: %s) — rebuilding "
                    "Minecraft client", attempt + 1, e)
                self._rebuild()
        raise RuntimeError(
            f"MineRL reset failed 3x even after client rebuilds: {last_err}")

    def step(self, action: int
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        act = self._macro_to_action(action)
        total_reward = 0.0
        obs, done, info = None, False, {}
        try:
            for _ in range(self._macro_ticks(action)):
                obs, reward, done, info = self._env.step(act)
                total_reward += float(reward)  # spec pays none; kept if fixed
                if done:
                    break
        except Exception as e:  # client died mid-episode: salvage the episode
            logger.warning(
                "MineRL step failed (%s) — rebuilding client and truncating "
                "episode", e)
            self._rebuild()
            return self._last_obs(), total_reward, False, True, {
                "env_restarted": True}
        # AUTO-REJOIN (remote mode): a kick/timeout freezes the client on a
        # static screen with step() still "succeeding" — detect and rebuild
        # (fresh mission init -> the client rejoins the server).
        if self._remote_frozen(obs, action):
            logger.warning(
                "remote server: POV frozen for %d motion steps — client "
                "kicked/disconnected; rebuilding to rejoin %s",
                self._frozen_count, self.remote_server)
            self._frozen_count = 0
            self._prev_pov_digest = None
            self._rebuild()
            return self._last_obs(), total_reward, False, True, {
                "env_restarted": True}
        # ---- ACTION THROTTLE ---------------------------------------------
        # Applied AFTER the step so it paces the agent's command rate against
        # the server without distorting anything the agent observes.
        if self.remote_server and self.remote_step_delay_s > 0.0:
            import time as _t
            _t.sleep(self.remote_step_delay_s)
        # ---- AUTO-RESPAWN (remote mode, 2026-08-02) ----------------------
        # The frozen watchdog above already recovers a death eventually — the
        # death GUI is static, so after `_frozen_limit` motion steps it
        # rebuilds and rejoins. But it cannot tell a DEATH from a KICK, so it
        # makes the agent stare at the death screen for ~150 steps first, and
        # every one of those steps is fed to the world model as real
        # experience of a world that is not responding.
        #
        # The death itself is GROUND TRUTH and available immediately: the
        # monotonic `deaths` stat increments the moment it happens. Use it to
        # rejoin AT ONCE instead of waiting for the frozen heuristic. On a
        # real server a fresh mission init respawns the player, so this IS
        # the respawn path — no GUI button-hunting required.
        if self.remote_server:
            try:
                _d = self._read_stat(obs, "deaths")
                if (_d is not None and self._prev_deaths is not None
                        and _d > self._prev_deaths):
                    if (_d - self._prev_deaths) > 2.5:
                        # STAT SYNC, NOT A DEATH — see the guard in
                        # _life_stats. Adopt as baseline; do NOT rejoin
                        # (the unguarded version rejoin-looped every ~57s).
                        logger.info(
                            "remote server: deaths stat sync %.0f -> %.0f "
                            "(persistent-total baseline, not a death)",
                            self._prev_deaths, _d)
                        self._prev_deaths = _d
                    else:
                        logger.warning(
                            "remote server: SkyBot DIED (deaths %.0f -> "
                            "%.0f) — rejoining immediately to respawn",
                            self._prev_deaths, _d)
                        self._prev_deaths = _d
                        self._frozen_count = 0
                        self._prev_pov_digest = None
                        self._deaths_rejoined += 1
                        self._rebuild()
                        return self._last_obs(), total_reward, False, True, {
                            "env_restarted": True, "respawned": True}
            except Exception:
                pass        # a diagnostic must never take the run down
        # ADAPTER-SIDE REWARDS (env handler is dead in the 1.0 port):
        # 1. +1 per log gained in inventory (the Treechop task reward).
        #    High-water mark: drops never pay negative, re-collects never
        #    pay twice.
        new_count = self._count_logs(obs)
        # ---- TYPED EVENT STREAM (infra #44, 2026-08-08) ------------------
        # The event-centric environment contract: every effect the agent
        # CAUSES this step is emitted as (kind, subtype). Habituation, goal
        # admission, affordance mapping, episodic memory and farm detection
        # all key off this stream, which is what keeps THEM domain-agnostic —
        # the taxonomy lives here in the adapter, nowhere else.
        _events: list = []
        # DEATH RE-BASELINE (audit fix): the spec allows survival deaths
        # (respawn wipes the inventory) and lifelong mode may not reset() for
        # days — after a death at N logs the monotone high-water silenced the
        # +1/log signal until the agent re-exceeded N. A full collapse to 0
        # while holding logs is the respawn signature (the action space has no
        # way to toss a whole stack); small drops still never re-pay.
        if new_count == 0 and self._log_count > 0:
            self._log_count = 0
        if new_count > self._log_count:
            _events.append(("pickup", "log"))
            self._pickups_by_type["log"] = (
                self._pickups_by_type.get("log", 0)
                + int(new_count - self._log_count))
            self._save_break_memory()
        total_reward += float(max(0, new_count - self._log_count))
        self._log_count = max(new_count, self._log_count)
        # 2. FIRST-BREAK ACHIEVEMENTS — now TIERED by block value (July 2026):
        #    a flat +1 made breaking a fern worth as much as felling a log, so
        #    the agent farmed the abundant instant-break plants and the
        #    discovered-goal machinery filled with trivial signatures. Tiers:
        #      * LOG (the Treechop goal)         -> +5.0  (dominant spike)
        #      * solid block needing a HOLD      -> +1.0  (leaves, dirt, stone)
        #      * instant-break plant (grass/fern)-> +0.15 (BELOW the 0.9 goal-
        #        discovery spike threshold, so plants no longer mint skills —
        #        only real, hold-requiring blocks become goals).
        # ---- TICKS-TO-BREAK, MEASURED (2026-07-26) -----------------------
        # Two days were lost to INFERRING how long a break takes from proxies:
        # first a dead `mainhand` sensor (read as "barehanded", giving ~60
        # ticks), then option durations. Both were wrong. This measures the
        # actual quantity: how many consecutive ATTACK ticks precede a break.
        #   ~8   -> holding an axe and CONNECTING (the axe is real)
        #   ~60  -> connecting barehanded
        #   never breaking while the counter climbs high -> the swing is NOT
        #          LANDING, i.e. an AIM problem, not a timing one.
        # Live evidence pointing here: every option ran its FULL budget (96/96
        # horizon, zero spikes, duration variance ZERO) while the agent holds
        # an axe and swings for ~80 ticks — 10x the margin a log needs.
        # ---- GUI PARALYSIS GATE (2026-07-27 stall assessment) -------------
        # A swing delivered into an open inventory screen hits nothing: the
        # menu swallows the click. Counting those ticks made ticks-to-break
        # meaningless (a 154-tick "attack run" that never broke anything was
        # mostly the agent standing in a menu) and made the AIM-vs-TIMING
        # diagnostic lie. Measure only swings that could actually land.
        _gui = self._gui_open(obs)
        self._steps_total += 1
        if _gui:
            self._gui_steps += 1
            self._gui_run += 1
            self._gui_run_max = max(self._gui_run_max, self._gui_run)
        else:
            self._gui_run = 0
        info["gui_open"] = bool(_gui)
        info["gui_frac"] = (self._gui_steps / max(1, self._steps_total))
        info["gui_run_max"] = int(self._gui_run_max)
        self._action_hist[int(action)] = self._action_hist.get(int(action), 0) + 1
        _atk = bool(self._macros[int(action)].get("attack")) \
            if 0 <= int(action) < len(self._macros) else False
        if _gui:
            _atk = False           # the world cannot hear this swing
        # ---- PLACEMENT DETECTION (2026-08-08, see _places_by_type) -------
        # A placed block is an inventory count DECREASING on a step whose
        # macro holds `use`. The use-gate keeps death-wipes and drops from
        # counting (they happen without `use`); a wipe landing on the same
        # tick as a use press is the rare bounded error we accept.
        _use_act = (bool(self._macros[int(action)].get("use"))
                    if 0 <= int(action) < len(self._macros) else False)
        if _gui:
            _use_act = False
        # ---- EFFORT SIGNAL (2026-08-09) ----------------------------------
        # Which steps SPENT the body: attack/use/jump are metabolically
        # effortful in a way looking and walking are not. Emitted as a
        # domain-agnostic contract field so the loop can price effort
        # (observed live: sustained attack swings at CLOUDS — futile effort
        # is free under a pure-novelty economy, so nothing extinguished it;
        # a human stops punching air because effort costs).
        _mac = (self._macros[int(action)]
                if 0 <= int(action) < len(self._macros) else {})
        # _atk, not the raw macro key: a swing the GUI swallows is not
        # effort spent on the world (symmetric with _use_act)
        info["effort"] = float(bool(_atk) or _use_act
                               or bool(_mac.get("jump")))
        # ---- EXTERNAL-AGENCY DETECTION (2026-08-09, social learning) -----
        # "The world changed and I did not do it." When the agent is PASSIVE
        # (no attack/use, no movement or camera macro) and STATIONARY, its
        # own POV should be nearly static — a large frame change in that
        # state is something ELSE acting (another player felling a block,
        # a mob, physics). Emitted as a typed event so the loop can do
        # observational learning; never credited as a self-caused effect.
        # Cheap: mean |delta| on an 8x-downsampled frame, only computed on
        # passive-stationary steps (rare), thresholded well above cloud
        # drift / sun flicker.
        try:
            # PASSIVE means the AGENT issued nothing frame-changing (review
            # 2026-08-09: the inventory toggle changes ~48% of the frame and
            # passed as "passive" — the agent owned a button that
            # manufactured demonstrations; any GUI step is likewise self-
            # caused screen change)
            _passive = (info["effort"] == 0.0
                        and not _gui
                        and not any(_mac.get(k) for k in
                                    ("forward", "back", "left", "right",
                                     "jump", "inventory", "sneak",
                                     "sprint"))
                        and not (_mac.get("camera")
                                 and any(abs(float(c)) > 0.5
                                         for c in _mac["camera"])))
            # ...and STATIONARY is MEASURED, not asserted from the action
            # (review: falls, knockback, water drift and momentum move the
            # camera on noop steps and read as external agency). Position
            # AND look must both be still vs the previous step.
            _ax = self._read_scalar(obs, "xpos")
            _ay = self._read_scalar(obs, "ypos")
            _az = self._read_scalar(obs, "zpos")
            _ap = self._read_scalar(obs, "pitch")
            _aw = self._read_scalar(obs, "yaw")
            _cur_state = (_ax, _ay, _az, _ap, _aw)
            _prev_state = getattr(self, "_prev_agency_state", None)
            _still = (_prev_state is not None
                      and None not in _cur_state
                      and None not in _prev_state
                      and abs(_cur_state[0] - _prev_state[0]) < 0.05
                      and abs(_cur_state[1] - _prev_state[1]) < 0.05
                      and abs(_cur_state[2] - _prev_state[2]) < 0.05
                      and abs(_cur_state[3] - _prev_state[3]) < 1.0
                      and abs(_cur_state[4] - _prev_state[4]) < 1.0)
            self._prev_agency_state = _cur_state
            _pov = obs.get("pov") if isinstance(obs, dict) else None
            if _pov is not None:
                _small = np.asarray(_pov, dtype=np.float32)[::8, ::8]
                _prev_small = getattr(self, "_prev_small_pov", None)
                if (_passive and _still and _prev_small is not None
                        and _prev_small.shape == _small.shape):
                    _delta = float(np.mean(np.abs(
                        _small - _prev_small))) / 255.0
                    if _delta > float(getattr(
                            self, "_observed_change_thresh", 0.02)):
                        _events.append(("observed_change", "world"))
                        info["observed_change_mag"] = round(_delta, 4)
                self._prev_small_pov = _small
        except Exception:
            pass
        _inv_now = self._inv_counts(obs)
        _placed_now: list = []
        if _use_act:
            for _it, _pc in self._inv_prev_counts.items():
                _dd = int(_pc) - int(_inv_now.get(_it, 0))
                if _dd > 0:
                    self._places_by_type[_it] = (
                        self._places_by_type.get(_it, 0) + _dd)
                    _placed_now.append(_it)
                    # A `use` that consumed this item PROVES it was in the
                    # selected slot at that moment. Nothing here can change
                    # the selected slot, so it stays there while any remains.
                    self._last_placed_item = str(_it)
                    self._save_break_memory()
        self._inv_prev_counts = _inv_now
        info["placed_now"] = _placed_now
        info["places_by_type"] = dict(self._places_by_type)
        # ---- #5: AN HONEST SWING COUNTER (2026-08-02) --------------------
        # Break progress in Minecraft is PER-BLOCK and resets the moment the
        # crosshair moves to a different block. The old counter incremented on
        # any attack macro regardless, so walking forward while swinging
        # reported a long, growing streak while the game was resetting
        # progress every step — the `swing` sense was telling the agent it was
        # making progress when it was not, which is worse than no sense at
        # all. Retarget is detected from GROUND TRUTH (position + look
        # direction), not inferred from the macro.
        # ---- THE SWING THAT JUST ENDED (fix 2026-08-04) ------------------
        # mine_block registers a break ONE STEP AFTER the swing that caused
        # it. By the time the break-delta loop below runs, `_attack_run` has
        # already been zeroed — either by the retarget branch or by the
        # not-attacking branch — so every break was attributed 0 ticks.
        # Measured consequence: "Ticks-to-break: large_fern:0t, oak_leaves:0t,
        # sand:0t, spruce_leaves:0t, wheat:0t" — every break at 0 ticks, which
        # is impossible, and the one statistic that says whether swings are
        # long enough to fell a log was therefore garbage.
        # LOAD-BEARING FOR THE LOG ECONOMICS TOO: the effort payment is
        # per-tick, so a 0 read pays 0 no matter how long the chop took.
        # LAZY BASELINE: if reset() saw no stats, adopt the first observation
        # that has them and count nothing from it.
        if not self._mine_seeded:
            _seed = self._mine_counts(obs)
            if _seed:
                self._mine_baseline = dict(_seed)
                self._mine_prev = dict(_seed)
                self._mine_seeded = True
                logger.info(
                    "mine_block baseline seeded from the first stats-bearing "
                    "observation (%d types) — counts before this point are "
                    "career totals, not breaks", len(_seed))
        _streak_prev = int(self._attack_run)
        _retargeted = False
        try:
            _px = self._read_scalar(obs, "xpos")
            _pz = self._read_scalar(obs, "zpos")
            _pi = self._read_scalar(obs, "pitch")
            _ya = self._read_scalar(obs, "yaw")
            _now = (_px, _pz, _pi, _ya)
            _prev = getattr(self, "_prev_aim", None)
            if _prev is not None and None not in _now and None not in _prev:
                # ---- THRESHOLD SCALES WITH action_repeat (2026-09-01) -----
                # This measures displacement BETWEEN AGENT STEPS, and a step
                # now covers `action_repeat` game ticks. At repeat 2 a
                # stationary agent drifts well under 0.05; at repeat 4 the
                # same physical stillness accumulates twice the drift
                # (knockback, water, slope, server rubber-banding), so a
                # fixed 0.05 would read as a RETARGET and zero `attack_run`
                # on a swing that never moved. The streak counter is the one
                # signal telling the agent that persistence accumulates —
                # spuriously resetting it is the quiet way to make chopping
                # unlearnable again.
                _mv_eps = 0.025 * float(self.action_repeat)
                _moved = (abs(_now[0] - _prev[0]) > _mv_eps
                          or abs(_now[1] - _prev[1]) > _mv_eps)
                _looked = (abs(_now[2] - _prev[2]) > 1.0
                           or abs(_now[3] - _prev[3]) > 1.0)
                _retargeted = bool(_moved or _looked)
            self._prev_aim = _now
        except Exception:
            pass
        if _atk and _retargeted:
            # still swinging, but at something else now: the streak that
            # matters (progress on ONE block) genuinely restarted.
            if self._attack_run > 0:
                self._runs_nobreak.append(int(self._attack_run))
            self._attack_run = 0
        if _atk:
            # the macro's OWN tick count, not a flat action_repeat: a probe
            # macro carrying `_ticks: 60` holds attack for 60 game ticks, and
            # counting 2 for it would undercount ticks-to-break 30x on any
            # scripted-hold path. Production macros carry no _ticks, so this
            # is identical there (_macro_ticks falls back to action_repeat).
            self._attack_run += self._macro_ticks(int(action))
            self._attack_run_max = max(self._attack_run_max, self._attack_run)
        else:
            # ---- CHOP FORENSICS: a swing SEQUENCE just ended without a
            # break. This is the measurement that separates the two failure
            # modes people keep arguing about from a third nobody names:
            #   AIM      — long runs that end with NOTHING breaking
            #   TIMING   — runs that end SHORTER than ticks-to-break
            #   TARGET   — runs that DO break things, but the wrong things
            # Recording only `attack_run_max` (a single scalar) could not tell
            # these apart, which is why the question stayed open for weeks.
            if self._attack_run > 0:
                self._runs_nobreak.append(int(self._attack_run))
                # A SWING THAT ACHIEVED NOTHING, exported per-step
                # (2026-08-14). "I tried this and nothing happened" is
                # evidence about a thing just as much as breaking it is —
                # and without it a category the agent CANNOT affect (stone,
                # barehanded) never closes, so it keeps maximum consequence
                # deficit forever and holds the magnet's attention
                # permanently. Measured live: 400 fruitless swings, 182 of
                # them >=8 ticks, max streak 202, zero breaks.
                self._swing_failed = int(self._attack_run)
            self._attack_run = 0

        mine_now = self._mine_counts(obs)
        _new_breaks: list = []
        for btype, cnt in mine_now.items():
            if (btype not in self._broken_this_episode
                    and cnt > self._mine_baseline.get(btype, 0)):
                self._broken_this_episode.add(btype)
                _new_breaks.append(btype)
        if _new_breaks:
            # MAX, NOT SUM (2026-07-25): summing tiers defeated the tiering
            # itself — six trivial plants at 0.15 sum to 0.90 and cross the
            # goal-discovery spike threshold, manufacturing exactly the junk
            # `discovered_N` goal the sub-0.9 tier was designed to prevent.
            # One step = one achievement, valued at its best block.
            # tier x familiarity decay: a type broken thousands of times
            # stops paying, one never managed keeps its full worth.
            total_reward += max(self._break_reward(b) * self._break_decay(b)
                                for b in _new_breaks)
        # THE MEASUREMENT records EVERY break, not just the first of each type
        # (2026-07-26). Hanging it off `_new_breaks` — which is first-break-
        # per-type-per-episode, because that is what the reward tier needs —
        # gave THREE lifetime samples and could say nothing about how OFTEN or
        # how RELIABLY the agent breaks anything. A measurement that
        # accumulates one sample per block type is not a distribution.
        # ONE STREAK, ONE RESET (fix 2026-08-02). `_attack_run` was zeroed
        # INSIDE this loop, so a second block type felled on the same step
        # recorded a ticks-to-break of 0 — poisoning the exact distribution
        # this measurement exists to produce. The streak that just ended is the
        # same streak for every block that fell this step, so read it once.
        # the larger of "still accumulating" and "what it was when this step
        # began" — the break belongs to the swing that preceded it.
        _run_at_break = max(int(self._attack_run), _streak_prev)
        _broke_now = False
        for _bt, _cnt in mine_now.items():
            # DEFAULT 0, NOT _cnt (fix 2026-08-02). `.get(_bt, _cnt)` makes the
            # delta exactly zero the first time a type is ever seen, so the
            # FIRST break of every block type was invisible here — including
            # the first log, which is the single sample the chop diagnosis most
            # needs. _mine_counts drops zero counters, so a type absent from
            # _mine_prev genuinely stood at 0.
            _d = int(_cnt) - int(self._mine_prev.get(_bt, 0))
            if _d > 0:
                _events.append(("break", str(_bt)))
                # ---- EFFORT PAYMENT, LOGS ONLY ----------------------------
                # per tick actually invested in the swing that felled it,
                # bounded so a runaway streak cannot pay thousands.
                if self._log_tick_reward > 0.0 and "log" in str(_bt):
                    _ticks = max(0, min(int(_run_at_break),
                                        int(self._log_tick_cap)))
                    _eff = self._log_tick_reward * _ticks * int(_d)
                    total_reward += _eff
                    logger.info(
                        "LOG FELLED: %s x%d after %d ticks -> effort %+.2f "
                        "(+%.1f completion tier)", _bt, _d, _ticks, _eff,
                        self._break_reward(_bt))
                self._break_ticks.append((_bt, _run_at_break))
                # WHAT is it actually felling? The target-selection half of
                # the diagnosis: a healthy chop run and a run that only ever
                # digs dirt have identical attack-run statistics.
                self._breaks_by_type[_bt] = (
                    self._breaks_by_type.get(_bt, 0) + _d)
                self._save_break_memory()
                _broke_now = True
        # ground-truth reach: a break proves something was within range
        self._reach_evidence = (1.0 if _broke_now
                                else self._reach_evidence * self._reach_decay)
        if _broke_now:
            self._attack_run = 0
        self._mine_prev = dict(mine_now)
        info = dict(info) if isinstance(info, dict) else {}
        # territory novelty + damage/death (consumed as INTRINSIC by the loop)
        try:
            info["world"] = self._world_events(obs)
        except Exception:
            info["world"] = {}
        # ---- SELF-STATE the policy can actually condition on --------------
        # `world` is assembled from the raw obs; two facts the agent needs are
        # only known HERE: whether a menu is covering the screen, and how much
        # it is carrying. Fold them in, then emit the fixed proprio vector.
        try:
            _w = info["world"]
            _w["gui_open"] = bool(_gui)
            _w["gui_frac"] = info.get("gui_frac")
            _w["gui_run_max"] = info.get("gui_run_max")
            _inv = obs.get("inventory") if isinstance(obs, dict) else None
            if isinstance(_inv, dict):
                _w["carrying"] = float(sum(
                    float(np.asarray(q).flatten()[0])
                    for q in _inv.values()
                    if q is not None))
            # the swing streak lives on the adapter, not in `world` — fold it
            # in so the agent can FEEL sustained effort (see PROPRIO_KEYS)
            _w["attack_run"] = float(self._attack_run)
            _w["reach_evidence"] = float(self._reach_evidence)
            info["proprio"] = self._proprio(_w)
        except Exception:
            info["proprio"] = np.zeros(self.PROPRIO_DIM, dtype=np.float32)
        # TICKS-TO-BREAK: the measured quantity, carried on info so the loop
        # can print it. `attack_run` is the CURRENT unbroken attack streak —
        # a large streak with no breaks is the signature of swinging at
        # nothing (aim), as opposed to swinging too briefly (timing).
        try:
            info["break_ticks"] = list(self._break_ticks)[-8:]
            info["attack_run"] = int(self._attack_run)
            info["attack_run_max"] = int(self._attack_run_max)
            info["runs_nobreak"] = list(self._runs_nobreak)
            info["breaks_by_type"] = dict(self._breaks_by_type)
            info["action_hist"] = dict(self._action_hist)
            # IS THE BREAK SENSOR EVEN ALIVE? Every break-derived signal in
            # this project — the first-break reward tiers, the goal unlocks,
            # the skills that mint from them, the whole chop diagnosis —
            # comes from the `mine_block` STATISTIC. Minecraft tracks stats
            # SERVER-side in multiplayer and only pushes them to a client on
            # request, so on a foreign server this dict can stay permanently
            # empty no matter what the agent breaks. That would make the
            # agent blind to its own achievements while looking, from the
            # logs, exactly like "it never breaks anything" — which is what a
            # 10068-tick attack streak with zero recorded breaks looks like.
            # Report the sensor's own health so the two are distinguishable.
            _mb = obs.get("mine_block") if isinstance(obs, dict) else None
            info["mine_block_present"] = _mb is not None
            info["mine_block_entries"] = (len(_mb) if isinstance(_mb, dict)
                                          else -1)
            info["mine_block_nonzero"] = (
                sum(1 for v in _mb.values()
                    if float(np.asarray(v).flatten()[0]) > 0)
                if isinstance(_mb, dict) else -1)
        except Exception:
            pass
        info["logs"] = new_count
        ach = {"log": new_count}
        for btype, cnt in mine_now.items():
            ach[f"mine_{btype}"] = cnt
        # ---- CRAFT ECONOMY (2026-07-26): the break economy, mirrored -----
        # A craft_item counter increment is a REAL effect the agent caused,
        # exactly like a block break: first-craft-of-type this episode pays
        # the TOP tier (making something is at least as significant as
        # felling a log), the achievement key rides the SAME path breaks do
        # (achievements -> goal discovery -> repeatability gate -> skill),
        # and clear_break_marks() re-arms it on the same goal-horizon
        # cadence. NOTHING here names a recipe — the reward is for an
        # OUTCOME the agent produced, however it found it.
        _crafted_now = self._craft_counts(obs)
        for _it, _cnt in _crafted_now.items():
            _d = int(_cnt) - int(self._craft_prev.get(_it, _cnt))
            if _d > 0:
                ach[f"craft_{_it}"] = int(_cnt)
                _events.append(("craft", str(_it)))
                self._crafts_by_type[str(_it)] = (
                    self._crafts_by_type.get(str(_it), 0) + _d)
                self._save_break_memory()
                if _it not in self._crafted_this_episode:
                    self._crafted_this_episode.add(_it)
                    total_reward += 5.0
        self._craft_prev = dict(_crafted_now)
        info["achievements"] = ach
        # typed event stream, assembled from every effect detected above
        # (placements were detected earlier in the step; a death this step
        # is read off the world info the life-stats section produced)
        for _it in _placed_now:
            _events.append(("place", str(_it)))
        if bool((info.get("world") or {}).get("died")):
            _events.append(("death", "self"))
        info["events"] = _events
        info["crafts_by_type"] = dict(self._crafts_by_type)
        info["pickups_by_type"] = dict(self._pickups_by_type)
        # WHAT IS HELD (2026-08-13). `_inv_counts` was computed every step for
        # placement detection but never published, so nothing downstream could
        # ask "what do I have?" — and possession is the second axis of "where
        # I am" for the consequence frontier. Already computed above; this
        # only exports it.
        info["inventory"] = dict(_inv_now)
        # ticks of the swing that just ended fruitlessly (0 = none this step)
        info["swing_failed"] = int(getattr(self, "_swing_failed", 0))
        self._swing_failed = 0
        # Task success: 64 logs = Treechop's completion condition.
        terminated = new_count >= 64
        # Otherwise MineRL's `done` is the step limit -> TRUNCATION (the
        # agent didn't fail — time ran out), keeping GAE bootstrapping right.
        return (self._pov_to_obs(obs), total_reward, terminated,
                bool(done) and not terminated, info)

    def render(self):
        return self._last_pov

    def close(self):
        # FINAL FLUSH (2026-08-11, review finding): nothing wrote the lifelong
        # memory on shutdown, so everything since the last throttled flush —
        # up to 2000 steps of territory — died with the process.
        try:
            self._save_break_memory(force=True)
        except Exception as _e:
            # never block shutdown — but never lose the memory SILENTLY
            logger.warning("break memory: final flush FAILED (%s) — "
                           "territory since the last flush is lost", _e)
        try:
            self._env.close()
        except Exception:
            pass
