"""
Environment Wrappers — Gymnasium Integration Layer
=====================================================
Wraps Gymnasium environments with additional functionality needed
by the developmental AI framework:

  1. Observation normalization: Keeps observations in a consistent range
  2. Action formatting: Converts between discrete/continuous as needed
  3. Curiosity reward injection: Adds intrinsic reward to the environment reward
  4. Fact extraction hooks: Triggers symbolic fact extraction each step
  5. Curriculum support: Enables dynamic difficulty adjustment

Data flow (per step):
  Agent selects action
  → Environment wrapper formats action
  → Gymnasium env executes step
  → Wrapper normalizes observation
  → Wrapper triggers fact extraction for knowledge graph
  → Wrapper computes curiosity reward via ICM
  → Wrapper returns (obs, mixed_reward, done, info)

The wrapper follows Gymnasium's standard API, so it's compatible with
Stable-Baselines3 and any other Gymnasium-compatible library.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Optional, Tuple, Any, List
import logging

logger = logging.getLogger(__name__)


class DevelopmentalEnvWrapper(gym.Wrapper):
    """
    Main environment wrapper for the developmental AI framework.

    Adds curiosity reward injection, observation normalization, and
    hooks for fact extraction and skill detection.

    Usage:
        env = gym.make("CartPole-v1")
        wrapped = DevelopmentalEnvWrapper(env)
        obs, info = wrapped.reset()
        obs, reward, terminated, truncated, info = wrapped.step(action)
        # info now contains 'intrinsic_reward', 'extrinsic_reward', etc.
    """

    def __init__(
        self,
        env: gym.Env,
        normalize_obs: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
    ):
        super().__init__(env)

        self.normalize_obs = normalize_obs
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        # Cap for the per-episode obs/action/reward history (only the last ~20
        # are consumed; keep a small margin). Bounds RAM on long episodes.
        self._history_cap = 64
        # Latest step's info dict — read by broadcasters that need env info
        # (e.g. the achievement goal channel reads info["achievements"]).
        self.last_info: Dict = {}

        # Running observation statistics for normalization
        self._obs_mean = None
        self._obs_var = None
        self._obs_count = 0

        # Episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_count = 0
        self.total_steps = 0

        # History for fact extraction
        self.obs_history: List[np.ndarray] = []
        self.action_history: List[Any] = []
        self.reward_history: List[float] = []

    @property
    def obs_dim(self) -> int:
        """Dimension of the observation space."""
        if isinstance(self.observation_space, gym.spaces.Box):
            return int(np.prod(self.observation_space.shape))
        elif isinstance(self.observation_space, gym.spaces.Discrete):
            return self.observation_space.n
        return 0

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (number of discrete actions or continuous dims)."""
        if isinstance(self.action_space, gym.spaces.Discrete):
            return self.action_space.n
        elif isinstance(self.action_space, gym.spaces.Box):
            return int(np.prod(self.action_space.shape))
        return 0

    @property
    def is_discrete(self) -> bool:
        """Whether the action space is discrete."""
        return isinstance(self.action_space, gym.spaces.Discrete)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment and tracking state."""
        obs, info = self.env.reset(**kwargs)
        obs = self._process_obs(obs)

        # Reset episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        self.obs_history = [obs.copy()]
        self.action_history = []
        self.reward_history = []
        self.last_info = dict(info) if isinstance(info, dict) else {}

        info["episode_count"] = self.episode_count

        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Returns enriched info dict with:
          - extrinsic_reward: Raw reward from the environment
          - raw_obs: Unnormalized observation
          - episode_reward: Cumulative reward so far
          - episode_length: Steps taken so far
          - total_steps: Total steps across all episodes
        """
        # Execute step in underlying environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Process observation
        raw_obs = obs.copy() if isinstance(obs, np.ndarray) else obs
        obs = self._process_obs(obs)

        # Clip reward for stability
        clipped_reward = float(np.clip(reward, -self.clip_reward, self.clip_reward))

        # Update tracking
        self.episode_reward += clipped_reward
        self.episode_length += 1
        self.total_steps += 1

        # Store recent history for fact extraction. Capped (only the last ~20
        # are ever read, at developmental_loop.py:2142) so LONG episodes —
        # Crafter runs to 10k steps — can't balloon RAM: each obs is the full
        # observation (12288 floats on pixels ≈ 49 KB), so an uncapped history
        # would hold ~0.5 GB per long episode.
        self.obs_history.append(obs.copy())
        self.action_history.append(action)
        self.reward_history.append(clipped_reward)
        if len(self.obs_history) > self._history_cap:
            self.obs_history = self.obs_history[-self._history_cap:]
            self.action_history = self.action_history[-self._history_cap:]
            self.reward_history = self.reward_history[-self._history_cap:]

        # Expose the latest info to broadcasters (e.g. achievement goals)
        self.last_info = info if isinstance(info, dict) else {}

        # Enrich info dict
        info["extrinsic_reward"] = clipped_reward
        info["raw_obs"] = raw_obs
        info["episode_reward"] = self.episode_reward
        info["episode_length"] = self.episode_length
        info["total_steps"] = self.total_steps

        # Track episode completion
        done = terminated or truncated
        if done:
            self.episode_count += 1
            info["episode_complete"] = True
            info["final_episode_reward"] = self.episode_reward
            info["final_episode_length"] = self.episode_length

        return obs, clipped_reward, terminated, truncated, info

    def _process_obs(self, obs: Any) -> np.ndarray:
        """Flatten and optionally normalize observations."""
        obs = np.asarray(obs, dtype=np.float32).flatten()

        if self.normalize_obs:
            # Frozen during evaluation (July 2026 audit H4): eval rollouts must
            # neither perturb the statistics training resumes with, nor be
            # scored under drifting input statistics.
            if not getattr(self, "_stats_frozen", False):
                self._update_obs_stats(obs)
            if self._obs_var is not None and self._obs_count > 10:
                obs = (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

        obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        return obs

    def freeze_obs_stats(self, frozen: bool = True) -> None:
        """Freeze/unfreeze the running observation-normalization statistics.
        Call with True before evaluation episodes and False after."""
        self._stats_frozen = bool(frozen)

    def _update_obs_stats(self, obs: np.ndarray) -> None:
        """Update running mean/variance of observations (Welford's algorithm)."""
        self._obs_count += 1
        if self._obs_mean is None:
            self._obs_mean = obs.copy()
            self._obs_var = np.zeros_like(obs)
        else:
            delta = obs - self._obs_mean
            self._obs_mean += delta / self._obs_count
            delta2 = obs - self._obs_mean
            self._obs_var += (delta * delta2 - self._obs_var) / self._obs_count

    def get_episode_summary(self) -> Dict[str, float]:
        """Summary of the current/last episode."""
        return {
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
        }


# ---------------------------------------------------------------------------
# Curriculum Wrapper — Dynamic difficulty adjustment
# ---------------------------------------------------------------------------

class CurriculumWrapper(gym.Wrapper):
    """
    Dynamically adjusts environment difficulty based on agent performance.

    As the agent masters easy challenges, the environment gets harder.
    This implements a simple curriculum learning strategy inspired by
    the developmental psychology concept of "zone of proximal development" —
    challenges should be just beyond current ability, not too easy or too hard.

    For Gymnasium environments, difficulty adjustment depends on the environment:
      - CartPole: Increase max episode length, add wind/noise
      - MountainCar: Increase hill steepness
      - Custom envs: Override adjust_difficulty() method

    The curriculum tracks performance over a sliding window and increases
    difficulty when the agent consistently succeeds.
    """

    def __init__(
        self,
        env: gym.Env,
        difficulty_increase_threshold: float = 0.8,
        check_interval: int = 50,
        max_difficulty: int = 10,
    ):
        super().__init__(env)

        self.difficulty_increase_threshold = difficulty_increase_threshold
        self.check_interval = check_interval
        self.max_difficulty = max_difficulty

        self.current_difficulty = 0
        self.episode_rewards = []
        self.episode_successes = []
        self.episodes_since_check = 0

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["difficulty_level"] = self.current_difficulty
        return obs, reward, terminated, truncated, info

    def record_episode(self, reward: float, success: bool) -> bool:
        """
        Record an episode outcome and check if difficulty should increase.

        Returns True if difficulty was increased.
        """
        self.episode_rewards.append(reward)
        self.episode_successes.append(success)
        self.episodes_since_check += 1

        if self.episodes_since_check >= self.check_interval:
            return self._check_and_adjust()

        return False

    def _check_and_adjust(self) -> bool:
        """Check recent performance and potentially increase difficulty."""
        if not self.episode_successes:
            return False

        recent = self.episode_successes[-self.check_interval:]
        success_rate = np.mean(recent)

        self.episodes_since_check = 0

        if success_rate >= self.difficulty_increase_threshold:
            if self.current_difficulty < self.max_difficulty:
                self.current_difficulty += 1
                self._apply_difficulty()
                logger.info(
                    f"Difficulty increased to {self.current_difficulty} "
                    f"(success rate: {success_rate:.2f})"
                )
                return True

        return False

    def _apply_difficulty(self) -> None:
        """
        Apply the current difficulty level to the environment.

        Override this method for custom environments.
        Default implementation adds observation noise proportional to difficulty.
        """
        # Base implementation: the difficulty level is available via
        # info["difficulty_level"] for the developmental loop to use
        pass

    @property
    def difficulty_info(self) -> Dict[str, Any]:
        return {
            "current_difficulty": self.current_difficulty,
            "max_difficulty": self.max_difficulty,
            "recent_success_rate": (
                np.mean(self.episode_successes[-self.check_interval:])
                if self.episode_successes else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Pixel Observation Wrapper
# ---------------------------------------------------------------------------

class PixelObservationWrapper(gym.Wrapper):
    """
    Wraps any environment to use pixel (RGB image) observations.

    Uses Gymnasium's render_mode="rgb_array" to capture frames, then
    resizes and normalizes them. The original vector observation is
    preserved in info["vector_obs"] for symbolic fact extraction.
    """

    def __init__(
        self,
        env: gym.Env,
        image_size: int = 64,
        grayscale: bool = False,
    ):
        super().__init__(env)
        self.image_size = image_size
        self.grayscale = grayscale
        self.channels = 1 if grayscale else 3

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.channels * image_size * image_size,),
            dtype=np.float32,
        )

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        try:
            from PIL import Image

            img = Image.fromarray(frame)
            img = img.resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
            if self.grayscale:
                img = img.convert("L")
            pixels = np.array(img, dtype=np.float32) / 255.0
        except ImportError:
            h, w = frame.shape[:2]
            row_idx = np.linspace(0, h - 1, self.image_size).astype(int)
            col_idx = np.linspace(0, w - 1, self.image_size).astype(int)
            pixels = frame[np.ix_(row_idx, col_idx)]
            if self.grayscale and pixels.ndim == 3:
                pixels = np.mean(pixels, axis=-1)
            pixels = pixels.astype(np.float32) / 255.0

        if pixels.ndim == 3:
            pixels = pixels.transpose(2, 0, 1)  # HWC -> CHW
        elif pixels.ndim == 2:
            pixels = pixels[np.newaxis]  # HW -> 1HW

        return pixels.flatten()

    def _get_pixel_obs(self, vector_obs) -> np.ndarray:
        frame = self.env.render()
        if frame is not None:
            return self._process_frame(frame)
        return np.zeros(
            self.channels * self.image_size * self.image_size,
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["vector_obs"] = obs
        return self._get_pixel_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["vector_obs"] = obs
        return self._get_pixel_obs(obs), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# MiniGrid Observation Wrapper (Dict -> flat symbolic vector)
# ---------------------------------------------------------------------------

class NativePixelObsWrapper(gym.ObservationWrapper):
    """For environments whose observation ALREADY IS an image (e.g. Crafter's
    (H, W, 3) uint8 render): convert HWC uint8 [0,255] → flat CHW float32
    [0,1], the layout every pixel consumer in this codebase expects
    (CNNEncoder reshapes flat → (B, C, H, W); CNNDecoder emits flat CHW;
    recon compares them directly, no symlog).

    This is deliberately NOT PixelObservationWrapper, which re-renders via
    env.render() and would discard the env's native observation (and return
    zeros when render yields None). No Welford normalization — /255 is the
    whole story for images.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        assert h == w, f"square images only, got {(h, w)}"
        self.image_size = h
        self.channels = c
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(c * h * w,), dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return (np.asarray(obs, dtype=np.float32) / 255.0
                ).transpose(2, 0, 1).flatten()


class MiniGridFlattenWrapper(gym.Wrapper):
    """
    Flatten a MiniGrid Dict observation into a 1-D float32 Box vector so the
    rest of the framework (which assumes array obs) can consume it unchanged.

    MiniGrid returns {'image': (7,7,3) uint8 symbolic grid, 'direction': int,
    'mission': str}. The 7x7x3 grid is MiniGrid's *symbolic* partial view —
    each cell is (object_idx, color_idx, state_idx) — which is exactly the kind
    of structured observation the neurosymbolic layer is meant to exploit. We
    flatten that grid (147) and append a one-hot of the agent's facing
    direction (4) -> 151-d vector. The constant mission string is dropped from
    the policy input but preserved (with the raw dict) in info['minigrid_obs']
    so a richer fact extractor can use it later.

    Applied as the INNERMOST wrapper in make_env (CurriculumWrapper does not
    override reset(), so without this the raw Dict would reach
    DevelopmentalEnvWrapper._process_obs and crash np.asarray(dict)).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        spaces = env.observation_space.spaces
        img_shape = spaces["image"].shape
        self._img_size = int(np.prod(img_shape))
        self._n_dir = int(spaces["direction"].n)
        dim = self._img_size + self._n_dir
        self.observation_space = gym.spaces.Box(
            low=0.0, high=255.0, shape=(dim,), dtype=np.float32
        )

    def _flatten(self, obs: Dict) -> np.ndarray:
        img = np.asarray(obs["image"], dtype=np.float32).flatten()
        direction = np.zeros(self._n_dir, dtype=np.float32)
        d = int(obs["direction"])
        if 0 <= d < self._n_dir:
            direction[d] = 1.0
        return np.concatenate([img, direction]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["minigrid_obs"] = obs
        return self._flatten(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["minigrid_obs"] = obs
        return self._flatten(obs), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Environment Factory
# ---------------------------------------------------------------------------

def make_env(
    env_name: str = "CartPole-v1",
    normalize: bool = True,
    curriculum: bool = True,
    render_mode: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    pixel_obs: bool = False,
    image_size: int = 64,
    grayscale: bool = False,
    agent_view_size: Optional[int] = None,
    action_repeat: int = 1,
    render_size: int = 0,
    sensors_cfg: Optional[Dict] = None,
    lifelong: bool = False,
    remote_server: Optional[str] = None,
    remote_step_delay_s: float = 0.0,
    start_tool: Any = "__default__",
    log_tick_reward: float = 0.0,
    log_tick_cap: int = 120,
    log_break_reward: float = 5.0,
    break_decay_scale: float = 0.0,
    break_memory_path: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> Tuple[DevelopmentalEnvWrapper, Optional[CurriculumWrapper]]:
    """
    Create a wrapped Gymnasium environment ready for the developmental AI framework.

    Args:
        env_name: Gymnasium environment ID
        normalize: Whether to normalize observations
        curriculum: Whether to wrap with curriculum learning
        render_mode: "human" for visual, None for headless
        max_episode_steps: Override default max steps
        pixel_obs: Use rendered pixel observations instead of vector
        image_size: Pixel observation resolution (square)
        grayscale: Convert pixel observations to grayscale

    Returns:
        (wrapped_env, curriculum_wrapper) — curriculum_wrapper is None if curriculum=False
    """
    # Crafter envs (CrafterReward-v1 / CrafterNoReward-v1) use the legacy gym
    # API and register only with old `gym`, so they bypass gym.make entirely:
    # construct the gymnasium adapter directly, pass its NATIVE image through
    # (CHW float [0,1] flat — no re-render, no Welford), and wrap. The config
    # must set environment.pixel_obs=true so the WM selects its CNN path.
    if env_name.startswith("Crafter"):
        from developmental_ai.environments.crafter_env import CrafterEnvAdapter
        env = CrafterEnvAdapter(
            reward="NoReward" not in env_name,
            length=max_episode_steps or 10000,
            size=image_size,
        )
        env = NativePixelObsWrapper(env)
        curriculum_wrapper = None
        if curriculum:
            curriculum_wrapper = CurriculumWrapper(env)
            env = curriculum_wrapper
        wrapped = DevelopmentalEnvWrapper(env, normalize_obs=False)
        return wrapped, curriculum_wrapper

    # MineRL (real Minecraft): legacy-gym dict-obs / dict-action env bridged
    # by the adapter, which emits the project's flat CHW float [0,1] pixel
    # contract and a Discrete macro-action space. Config sets pixel_obs=true
    # so the WM takes its CNN path. No re-render, no Welford (obs is [0,1]).
    if env_name.startswith("MineRL"):
        from developmental_ai.environments.minerl_env import MineRLEnvAdapter
        env = MineRLEnvAdapter(env_name=env_name, image_size=image_size,
                               action_repeat=action_repeat,
                               render_size=render_size, lifelong=lifelong,
                               sensors_cfg=sensors_cfg,
                               remote_server=remote_server,
                               remote_step_delay_s=remote_step_delay_s,
                               log_tick_reward=log_tick_reward,
                               log_tick_cap=log_tick_cap,
                               log_break_reward=log_break_reward,
                               break_decay_scale=break_decay_scale,
                               break_memory_path=break_memory_path,
                               agent_name=agent_name,
                               **({} if start_tool == "__default__"
                                  else {"start_tool": start_tool}))
        curriculum_wrapper = None
        if curriculum:
            curriculum_wrapper = CurriculumWrapper(env)
            env = curriculum_wrapper
        wrapped = DevelopmentalEnvWrapper(env, normalize_obs=False)
        return wrapped, curriculum_wrapper

    # Craftax (Craftax-*-v1): JAX-functional env, bridged to gymnasium by the
    # adapter. We use the SYMBOLIC variant (flat factored-state vector) → the
    # MLP world-model path (pixel_obs stays false in config), so the adapter's
    # flat float32 obs flows straight through DevelopmentalEnvWrapper.
    if env_name.startswith("Craftax"):
        from developmental_ai.environments.craftax_env import CraftaxEnvAdapter
        env = CraftaxEnvAdapter(env_name=env_name)
        curriculum_wrapper = None
        if curriculum:
            curriculum_wrapper = CurriculumWrapper(env)
            env = curriculum_wrapper
        # normalize_obs=False: the factored obs is already [0,1] with many
        # sparse binary (tile-type) dims — Welford standardization would turn
        # rare 1s into large z-scores and destabilize reconstruction.
        wrapped = DevelopmentalEnvWrapper(env, normalize_obs=False)
        return wrapped, curriculum_wrapper

    # MiniGrid / BabyAI envs ship Dict observations and need their package
    # imported to register with Gymnasium. They use the built-in symbolic grid
    # view, so the rendered-pixel path is bypassed.
    is_minigrid = env_name.startswith("MiniGrid") or env_name.startswith("BabyAI")
    if is_minigrid:
        import minigrid  # noqa: F401  (registers MiniGrid/BabyAI env IDs)
        # Custom envs ship under MiniGrid-* IDs; importing their module registers
        # those IDs with Gymnasium (idempotent).
        if "RuleRegime" in env_name:
            import developmental_ai.environments.rule_regime_doorkey  # noqa: F401
        if "NoisyTV" in env_name:
            import developmental_ai.environments.noisy_tv_gridworld  # noqa: F401
        if "ObsKeyNoKey" in env_name:
            import developmental_ai.environments.observable_doorkey  # noqa: F401
        pixel_obs = False

    kwargs = {}
    if pixel_obs:
        kwargs["render_mode"] = "rgb_array"
    elif render_mode:
        kwargs["render_mode"] = render_mode
    if max_episode_steps:
        kwargs["max_episode_steps"] = max_episode_steps
    # Restrict the agent's egocentric view to force partial observability (Rung 6:
    # makes the within-episode memory broadcast load-bearing). MiniGrid only;
    # must be odd and >= 3. The flatten wrapper reads the view shape dynamically,
    # so obs_dim adapts automatically.
    if is_minigrid and agent_view_size:
        kwargs["agent_view_size"] = int(agent_view_size)

    env = gym.make(env_name, **kwargs)

    # Flatten MiniGrid's Dict obs to a Box vector BEFORE any other wrapper, so
    # everything downstream (curriculum, developmental) sees an array obs.
    if is_minigrid:
        env = MiniGridFlattenWrapper(env)

    # Apply curriculum wrapper first (innermost)
    curriculum_wrapper = None
    if curriculum:
        curriculum_wrapper = CurriculumWrapper(env)
        env = curriculum_wrapper

    # Apply pixel observation wrapper if requested
    if pixel_obs:
        env = PixelObservationWrapper(
            env, image_size=image_size, grayscale=grayscale
        )

    # Apply developmental wrapper (outermost)
    wrapped = DevelopmentalEnvWrapper(
        env, normalize_obs=normalize and not pixel_obs
    )

    return wrapped, curriculum_wrapper


# ---------------------------------------------------------------------------
# Environment observation label helpers
# ---------------------------------------------------------------------------
# These provide human-readable labels for common Gymnasium environments,
# used by the FactExtractor to produce meaningful symbolic facts.

ENV_OBS_LABELS = {
    "CartPole-v1": [
        "cart_position",
        "cart_velocity",
        "pole_angle",
        "pole_angular_velocity",
    ],
    "MountainCar-v0": [
        "car_position",
        "car_velocity",
    ],
    "Acrobot-v1": [
        "cos_joint1",
        "sin_joint1",
        "cos_joint2",
        "sin_joint2",
        "angular_vel_joint1",
        "angular_vel_joint2",
    ],
    "LunarLander-v3": [
        "x_position",
        "y_position",
        "x_velocity",
        "y_velocity",
        "angle",
        "angular_velocity",
        "left_leg_contact",
        "right_leg_contact",
    ],
    "Pendulum-v1": [
        "cos_angle",
        "sin_angle",
        "angular_velocity",
    ],
    # --- MuJoCo Continuous Control ---
    "HalfCheetah-v4": [
        "rootx", "rootz", "rooty",
        "bthigh_angle", "bshin_angle", "bfoot_angle",
        "fthigh_angle", "fshin_angle", "ffoot_angle",
        "rootx_vel", "rootz_vel", "rooty_vel",
        "bthigh_vel", "bshin_vel", "bfoot_vel",
        "fthigh_vel", "fshin_vel",
    ],
    "Hopper-v4": [
        "rootz", "rooty",
        "thigh_angle", "leg_angle", "foot_angle",
        "rootx_vel", "rootz_vel", "rooty_vel",
        "thigh_vel", "leg_vel", "foot_vel",
    ],
    "Walker2d-v4": [
        "rootz", "rooty",
        "thigh_angle", "leg_angle", "foot_angle",
        "thigh_left_angle", "leg_left_angle", "foot_left_angle",
        "rootx_vel", "rootz_vel", "rooty_vel",
        "thigh_vel", "leg_vel", "foot_vel",
        "thigh_left_vel", "leg_left_vel", "foot_left_vel",
    ],
    "Ant-v4": [
        "torso_z", "torso_orient_w", "torso_orient_x",
        "torso_orient_y", "torso_orient_z",
        "hip1_angle", "ankle1_angle",
        "hip2_angle", "ankle2_angle",
        "hip3_angle", "ankle3_angle",
        "hip4_angle", "ankle4_angle",
        "torso_xvel", "torso_yvel", "torso_zvel",
        "torso_wxvel", "torso_wyvel", "torso_wzvel",
        "hip1_vel", "ankle1_vel",
        "hip2_vel", "ankle2_vel",
        "hip3_vel", "ankle3_vel",
        "hip4_vel", "ankle4_vel",
    ],
    "Swimmer-v4": [
        "angle_front", "angle_back",
        "vel_x", "vel_y",
        "ang_vel_front", "ang_vel_back",
        "slider_pos_x", "slider_pos_y",
    ],
    "Reacher-v4": [
        "cos_joint0", "cos_joint1",
        "sin_joint0", "sin_joint1",
        "target_x", "target_y",
        "angular_vel_joint0", "angular_vel_joint1",
        "fingertip_target_dist_x", "fingertip_target_dist_y",
        "fingertip_target_dist_z",
    ],
}

ENV_ACTION_LABELS = {
    "CartPole-v1": ["push_left", "push_right"],
    "MountainCar-v0": ["push_left", "no_push", "push_right"],
    "Acrobot-v1": ["torque_negative", "torque_zero", "torque_positive"],
    "LunarLander-v3": ["noop", "fire_left", "fire_main", "fire_right"],
    # --- MuJoCo (continuous torques) ---
    "HalfCheetah-v4": [
        "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot",
    ],
    "Hopper-v4": ["thigh", "leg", "foot"],
    "Walker2d-v4": [
        "thigh", "leg", "foot", "thigh_left", "leg_left", "foot_left",
    ],
    "Ant-v4": [
        "hip1", "ankle1", "hip2", "ankle2",
        "hip3", "ankle3", "hip4", "ankle4",
    ],
    "Swimmer-v4": ["torque_front", "torque_back"],
    "Reacher-v4": ["joint0_torque", "joint1_torque"],
}


# MiniGrid/BabyAI share a fixed 7-action discrete space (Actions enum). The
# obs is a flattened 7x7x3 symbolic grid + direction one-hot, which has no
# compact per-feature label set, so obs labels fall back to None (the loop
# guards `if self.obs_labels` everywhere).
MINIGRID_ACTION_LABELS = [
    "turn_left", "turn_right", "move_forward",
    "pickup", "drop", "toggle", "done",
]


def get_env_labels(env_name: str) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Get observation and action labels for a known environment.

    Falls back to version variants (e.g., HalfCheetah-v5 → HalfCheetah-v4)
    so labels don't need to be duplicated for every Gymnasium version bump.
    """
    if env_name.startswith("MiniGrid") or env_name.startswith("BabyAI"):
        return None, MINIGRID_ACTION_LABELS

    obs = ENV_OBS_LABELS.get(env_name)
    act = ENV_ACTION_LABELS.get(env_name)
    if obs is None and "-v" in env_name:
        base = env_name.rsplit("-v", 1)[0]
        for suffix in ("v4", "v5", "v3", "v2", "v1"):
            key = f"{base}-{suffix}"
            if key in ENV_OBS_LABELS:
                obs = ENV_OBS_LABELS[key]
                act = ENV_ACTION_LABELS.get(key)
                break
    return obs, act
