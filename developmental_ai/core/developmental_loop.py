"""
Developmental AI Loop — Core Orchestrator
============================================
This is the MAIN LOOP that ties all components together into a unified
developmental learning system. It implements the core cycle from the
architecture diagram:

  Environment → Curiosity Engine → Set Own Goal → Explore & Act →
  Observe Result → Prediction Error → Update World Model → Add Skill →
  (repeat, no endpoint)

Key difference from standard RL training loops:
  - There is NO fixed task or reward function
  - The agent sets its OWN goals based on curiosity
  - Learning never "finishes" — skills compound indefinitely
  - The knowledge graph accumulates symbolic facts alongside neural learning
  - Skills are detected, saved, and reused automatically

Data flow between components (per step):
  1. Gymnasium sends raw observations to ICM + Policy
  2. ICM computes curiosity/intrinsic reward from prediction error
  3. Policy selects action using blended (intrinsic + extrinsic) reward
  4. World model (RSSM) encodes experience into latent space
  5. Fact extractor converts observations to symbolic facts → Knowledge Graph
  6. PyKEEN trains graph embeddings → feeds structured knowledge back to RSSM
  7. Skill Bank detects mastery and saves/retrieves learned policies

This module is the "orchestra conductor" — it doesn't implement any
algorithms itself, but coordinates all the components in the right order.

NOTE: The "glue layer" (Section 3.2 in the reference doc) that deeply
connects DreamerV3's latent space to the skill bank is intentionally
left as a placeholder interface. The user will implement this custom
component later. What we build here is the orchestration framework
that the glue layer plugs into.
"""

import math

import torch
import numpy as np
import gymnasium as gym
import yaml
import json
import os
import time
import random
import logging
from typing import Dict, Optional, Any, List, Tuple
from collections import deque

# Framework components
from developmental_ai.world_model.rssm import WorldModel, symexp
from developmental_ai.world_model.replay_buffer import (
    ReplayBuffer,
    MultiStreamReplayBuffer,
)
from developmental_ai.curiosity.icm import IntrinsicCuriosityModule
from developmental_ai.curiosity.learning_progress import LearningProgressCuriosity
from developmental_ai.knowledge_graph.knowledge_graph import (
    InMemoryKnowledgeGraph,
    Neo4jKnowledgeGraph,
    create_knowledge_graph,
    FactExtractor,
    SymbolicFact,
)
from developmental_ai.knowledge_graph.graph_embeddings import (
    KGEmbeddingTrainer,
    KnowledgeGraphGNN,
    SymbolicNeuralGate,
)
from developmental_ai.skill_bank.skill_bank import (
    SkillBank,
    MasteryDetector,
    SkillComposer,
    CompositeSkillExecutor,
)
from developmental_ai.policy.actor_critic import (
    StandaloneActorCritic,
    RewardMixer,
    DreamActorCritic,
)
from developmental_ai.environments.wrappers import (
    make_env,
    get_env_labels,
    DevelopmentalEnvWrapper,
)
from developmental_ai.core.episodic_memory import EpisodicWorkingMemory
from developmental_ai.core.affordance_broadcast import AffordanceBroadcast
from developmental_ai.core.rule_regime_broadcast import RuleRegimeBroadcast
from developmental_ai.world_model.symbolic_decoder import SymbolicDecoderManager
from developmental_ai.core.glue_layer import (
    GlueLayer,
    DevelopmentalStageController,
)
from developmental_ai.llm.llm_module import LLMModule

logger = logging.getLogger(__name__)


# Confidence for EARNED-MEANING facts. These four paths had NEVER executed —
# both sensors that trigger them were dead — so a missing confidence field
# (3-tuple into a 4-tuple unpack) crashed the run the moment they first fired.
# The values encode measured-vs-inferred: the damage/death EVENT is ground
# truth but WHICH creature caused it is a VLM identification and can be wrong;
# the tool facts are measured on both halves (inventory-derived hand + a real
# block break), so they are trusted more.
_CREATURE_FACT_CONF = 0.7
_TOOL_FACT_CONF = 0.9


# ---- resume contracts (2026-08-23 review) --------------------------------
# Weights fit to a REWARD ECONOMY rather than to how the world works. They are
# dropped from a world-model restore on purpose (see _load_wm), so their
# absence from a checkpoint payload is expected, not a defect.
_WM_ECONOMY_PREFIXES = ("reward_predictor", "reward_bins")

# Components expressed IN THE WORLD MODEL'S LATENT SPACE. Restoring one of
# these onto a FRESH encoder pairs trained weights with a representation they
# were never fit to: the symbol head reads noise, and the familiarity counters
# mark unseen views as already-visited, zeroing exploration income exactly when
# it is needed most.
#
# `magnet` is deliberately NOT a member. VisionScaffold.state() is
# category-NAME-keyed scalars (learning progress per label like "tree"), which
# mean the same thing against any encoder; gating it here made a failed
# world-model restore ALSO pay the magnet's full cold-start tax, silently, in
# the one situation where that hurts most. If you add an encoder-derived field
# to that state (a latent centroid, an LSH bucket), put `magnet` back —
# tests/_review_fixes_smoke.py asserts the state's key set precisely so this
# decision cannot be made by accident.
_RESUME_ENCODER_KEYED = ("perception", "familiarity")


class _ResumePreflightRefusal(RuntimeError):
    """A restore refused BEFORE any tensor was copied.

    Distinct from a mid-load failure on purpose: the generic handler warns
    that torch may have left a partial hybrid behind, which is true of a
    shape mismatch and false here. Reporting a clean refusal as a possible
    hybrid would send the next operator hunting for corruption that does not
    exist — this project's rule is that a degraded outcome must never read as
    a clean one, and the converse holds too.
    """


def _wm_restore_absent(model_keys, payload_keys, dropped_keys):
    """Tensors the model NEEDS that this restore will not supply.

    `load_state_dict(..., strict=False)` is required here (the reward head must
    keep its fresh init), but strict=False silences BOTH halves of the check —
    and only the `unexpected` half was re-implemented by hand. A checkpoint
    missing whole submodules therefore loaded silently, `world_model` was
    reported as restored, and the dependency gate it guards then admitted
    perception/familiarity onto a half-fresh encoder: the exact pairing that
    gate exists to prevent. (world_model.pt is written non-atomically, so a
    truncated file makes this reachable, not merely theoretical.)

    A key is excused ONLY if we removed it ourselves (`dropped_keys`), never by
    prefix-matching what came back missing: if a checkpoint never carried a
    reward head and `resume_reward_head` is on, nothing was dropped and the
    absence is a real defect that must be reported.
    """
    return sorted(set(model_keys) - set(payload_keys) - set(dropped_keys))


def _to_fact(triple, source: str, timestep: int = 0) -> "SymbolicFact":
    """(subject, relation, object, confidence) -> SymbolicFact.

    `timestep` must be threaded in: SymbolicFact.timestamp defaults to 0, so
    without it every symbolizer fact is stamped 0 and the max(timestamp)
    update inside add_fact is a permanent no-op. FactExtractor already passes
    a real timestep; this makes the symbolizer channel consistent with it.
    """
    subj, rel, obj, conf = triple
    return SymbolicFact(subject=subj, relation=rel, obj=obj,
                        confidence=float(conf), source=source,
                        timestamp=int(timestep))


class _NullSymbolicDecoder:
    """No-op symbolic decoder for PIXEL envs (July 2026, Crafter).

    The real SymbolicDecoderManager allocates one classification head per
    observation DIMENSION and extracts per-dimension facts — sensible for a
    151-d symbolic grid vector, meaningless (and ~16M params + a per-step
    softmax over 12k heads) for raw pixels. This stub satisfies every call
    site: update_discretizer/extract_facts/train_step no-op, stats carries
    the keys the episode printout reads, avg_confidence=0 keeps the glue's
    uncertainty term inert, and decoder=None (the checkpoint writer guards).
    """

    decoder = None
    avg_confidence = 0.0
    stats = {"symbolic_decoder_confidence": 0.0}

    def update_discretizer(self, obs) -> None:
        pass

    def extract_facts(self, latent, timestep: int = 0) -> list:
        return []

    def train_step(self, latent_states=None, raw_observations=None, **kw) -> dict:
        return {}


class DevelopmentalAI:
    """
    The complete developmental AI agent.

    This class creates and coordinates all components:
      - Environment (Gymnasium)
      - World Model (RSSM)
      - Curiosity Engine (ICM)
      - Knowledge Graph (in-memory or Neo4j)
      - Graph Embeddings (PyKEEN)
      - Skill Bank
      - Policy (actor-critic)
      - Reward Mixer

    Usage:
        agent = DevelopmentalAI(config_path="configs/default.yaml")
        agent.run(total_timesteps=100000)  # Or run indefinitely

    The agent can be stopped and resumed — the knowledge graph and
    skill bank persist to disk, and the world model can be checkpointed.
    """

    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None):
        """
        Initialize all components of the developmental AI agent.

        Args:
            config: Configuration dictionary (overrides config_path)
            config_path: Path to YAML configuration file
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            # Use sensible defaults
            self.config = self._default_config()

        # ---- EFFECTIVE-CONFIG TRACKING (infra #26, 2026-08-08) -----------
        # A configured log reward of 20.0 was silently ignored for its whole
        # life (the code read a different attribute) and keys have shipped in
        # every config with nothing reading them. Wrap the config so every
        # .get() is recorded; the first segment log reports what was NEVER
        # read. Failure to wrap must never block a run.
        self._config_path = config_path
        try:
            from developmental_ai.infra.config_echo import TrackedConfig
            if isinstance(self.config, dict):
                self.config = TrackedConfig(self.config)
        except Exception as _e:
            logger.warning("infra: config tracking unavailable (%s)", _e)

        # ---- Deterministic seeding (optional) ----
        # When `seed` is set in the config, fix every RNG that affects training so
        # that two runs differing ONLY in a single config flag (e.g. the symbolic
        # ablation: symbolic.enabled on vs off) start from identical weights and
        # follow the same environment stream. Without this, run-to-run RNG noise
        # could masquerade as a treatment effect. Default None preserves the prior
        # non-deterministic behavior. The env's RNG is seeded once on the first
        # reset (see `_seeded_reset`); subsequent resets advance deterministically.
        self.seed = self.config.get("seed", None)
        self._env_seeded = False
        if self.seed is not None:
            self.seed = int(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            # Single-threaded BLAS removes a major source of float nondeterminism
            # on CPU; harmless if already set by the launch environment.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
            logger.info(f"Deterministic seed set: {self.seed}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # ---- Create environment ----
        env_cfg = self.config.get("environment", {})
        self.env_name = env_cfg.get("name", "CartPole-v1")
        self.pixel_obs = env_cfg.get("pixel_obs", False)
        # Determine lifelong EARLY: env[0] (self.env) is built next and must
        # carry the flag (in lifelong mode the world must never truncate).
        # The full LifelongController + parallel-requirement check comes later.
        self._lifelong = bool(
            self.config.get("lifelong", {}).get("enabled", False))
        # ---- COLLECTION PATH, DECIDED BEFORE THE ENV BOOTS (2026-10-03) ----
        # num_envs <= 1 used to drop SILENTLY into _run_episode, the unwired
        # pre-Wave-1 body (CLAUDE.md §4.2) — a "smaller" SkyBot run was a
        # different agent. select_collection_path raises for Minecraft
        # configs and for a requested-but-downgraded parallel config; the
        # escape is parallel_envs.allow_legacy_single_env: true. Non-Minecraft
        # configs that never asked for parallel keep the historical default.
        # tests/_collection_path_smoke.py holds the rule.
        from developmental_ai.foundation.runtime.collection_path import (
            select_collection_path)
        self._collection_path = select_collection_path(self.config)
        logger.info("collection path: %s (%s)", self._collection_path.path,
                    self._collection_path.reason)
        self.image_size = env_cfg.get("image_size", 64)
        # Game ticks per agent decision. Cached so anything whose MEANING is
        # in game time (not decision count) can be expressed in ticks and
        # stay invariant when this knob moves — it went 2 -> 4 on 2026-09-01
        # and silently doubled the world-time span of every step-denominated
        # window in the loop.
        self.env_action_repeat = max(1, int(env_cfg.get("action_repeat", 1)))
        self.image_channels = 1 if env_cfg.get("grayscale", False) else 3
        # Live viewer: when enabled, force rgb_array rendering so env.render()
        # yields frames to stream. Config-gated, default off (existing runs
        # unchanged). See developmental_ai/core/viewer_server.py.
        viewer_cfg = self.config.get("viewer", {})
        self._viewer_enabled = bool(viewer_cfg.get("enabled", False))
        _render_mode = "rgb_array" if self._viewer_enabled else env_cfg.get("render_mode")
        self.env, self.curriculum = make_env(
            env_name=self.env_name,
            normalize=True,
            curriculum=self.config.get("loop", {}).get("curriculum_enabled", True),
            render_mode=_render_mode,
            max_episode_steps=env_cfg.get("max_episode_steps"),
            pixel_obs=self.pixel_obs,
            image_size=self.image_size,
            grayscale=env_cfg.get("grayscale", False),
            agent_view_size=env_cfg.get("agent_view_size"),
            action_repeat=env_cfg.get("action_repeat", 1),
            render_size=env_cfg.get("render_size", 0),
            # THE SENSOR BUS. Absent from config -> None -> the adapter
            # emits info["proprio"] and nothing else, i.e. exactly the
            # pre-bus behaviour for every other config in the tree.
            sensors_cfg=(self.config.get("sensors") or None),
            lifelong=self._lifelong,
            remote_server=env_cfg.get("remote_server"),
            # this site builds the PRIMARY env, so it keeps the bare prefix
            # (see _agent_name_for: renaming slot 0 would orphan its player)
            agent_name=self._agent_name_for(0),
            remote_step_delay_s=env_cfg.get("remote_step_delay_s", 0.0),
            # LOG ECONOMICS (2026-08-03): pays per attack tick invested in
            # the swing that fells a log, so the PROCESS of chopping out-earns
            # any other block per tick — not just the completion event.
            # 0.0 = off = byte-identical for every other config. BOTH call
            # sites get it: the scouts feed the same shared world model, so
            # different reward rules per stream would teach it two economies.
            log_tick_reward=env_cfg.get("log_tick_reward", 0.0),
            log_tick_cap=env_cfg.get("log_tick_cap", 120),
            log_break_reward=env_cfg.get("log_break_reward", 5.0),
            break_decay_scale=env_cfg.get("break_decay_scale", 0.0),
            # lifelong break-mastery memory (2026-08-08): survives restarts
            break_memory_path=env_cfg.get("break_memory_path", None),
            start_tool=env_cfg.get("start_tool", "__default__"),
        )

        # ---- Live viewer server (optional) ----
        self.viewer = None
        if self._viewer_enabled:
            try:
                from developmental_ai.core.viewer_server import ViewerServer
                self.viewer = ViewerServer(port=int(viewer_cfg.get("port", 8000)))
                self.viewer.start()
                logger.info(
                    f"Live viewer streaming on port {viewer_cfg.get('port', 8000)} "
                    f"(SSH-tunnel it, then open http://localhost:{viewer_cfg.get('port', 8000)})"
                )
            except Exception as e:
                logger.warning(f"Live viewer disabled (init failed): {e}")
                self.viewer = None

        # Get environment dimensions
        self.obs_dim = self.env.obs_dim
        # PROPRIOCEPTION SOURCE: the env decides whether the agent has a body
        # sense and how wide it is. Duck-typed on purpose — any env exposing
        # PROPRIO_DIM + emitting info["proprio"] gets one for free, and any
        # env that does not yields 0 and the unchanged pixels-only policy.
        # Unwrapped so a wrapper chain cannot hide it.
        _e = self.env
        for _ in range(6):
            if hasattr(_e, "PROPRIO_DIM"):
                break
            _e = getattr(_e, "env", None) or getattr(_e, "unwrapped", None)
            if _e is None:
                break
        self._proprio_source = _e if hasattr(_e, "PROPRIO_DIM") else None
        # last-seen self-state per env; act() consumes this (it runs before
        # the step, so there is no "this step" proprio yet)
        self._proprio_per_env = None
        # RAW env proprio as of the CURRENT observation, one slot per body.
        # Separate from _proprio_per_env (which is the loop-AUGMENTED vector
        # the policy reads): the world model gets the env's own body sense
        # only, because the augmented one contains the flow senses the world
        # model itself produces. None when the world model was built without
        # a body (world_model.proprio absent), so nothing allocates.
        self._wm_proprio_prev = None
        # ---- PERSPECTIVE FROM MOTION: per-step sense state ---------------
        self._flow_now = None
        self._flow_prev_latent = None
        self._flow_prev_valid = None
        self._flow_resid_now = 0.0
        # ---- SPATIAL SURPRISE (2026-09-19, roadmap P4) -------------------
        # 8x8 to match the encoder's own spatial grid, so a surprise cell and
        # a feature-map cell describe the same patch of world. Curiosity has
        # been SCENE-LEVEL since the project began: the agent could be
        # surprised, but never surprised BY A PLACE IN THE FRAME. This is 64
        # numbers that were already being computed and then averaged away.
        self._surprise_grid = int(
            (self.config.get("sensors") or {}).get("surprise_grid", 8))
        self._surprise_now = None
        # ---- SPATIAL MEMORY (2026-09-19, roadmap G2/P2/P3) --------------
        # The first representations in this system that outlive the frame
        # they came from. All three are built from the agent's OWN estimates
        # — the world model's forward-probe magnitude and the body's sensed
        # ego-motion — never from the engine's truth, which exists in this
        # codebase only as the RED oracle and only to measure them.
        _sp = (self.config.get("spatial") or {})
        self._spatial_on = bool(_sp.get("enabled", False))
        self._occ_grid = int(_sp.get("occupancy_grid", 16))
        self._sr_grid = int(_sp.get("successor_grid", 8))
        self._walk_bins = int(_sp.get("walk_bins", 8))
        self._probe_grid = int(_sp.get("probe_grid", 8))
        self._occupancy = self._successor = self._walkable = None
        self._probe_map_now = None
        self._occ_now = self._sr_now = self._walk_now = None
        if self._spatial_on:
            from developmental_ai.spatial import (
                EgocentricOccupancy, SuccessorMap, WalkabilityMap)
            self._occupancy = EgocentricOccupancy(
                grid=self._occ_grid,
                cell_blocks=float(_sp.get("occupancy_cell_blocks", 1.0)),
                decay=float(_sp.get("occupancy_decay", 0.93)),
                fov_rad=math.radians(float(
                    (self.config.get("world_model") or {}).get(
                        "camera_fov_deg", 70.0))),
                max_range=float(_sp.get("max_range_blocks", 12.0)))
            self._successor = SuccessorMap(
                grid=self._sr_grid,
                cell_blocks=float(_sp.get("successor_cell_blocks", 8.0)),
                gamma=float(_sp.get("successor_gamma", 0.95)),
                lr=float(_sp.get("successor_lr", 0.1)))
            self._walkable = WalkabilityMap(
                bins=self._walk_bins, lr=float(_sp.get("walk_lr", 0.1)))
            logger.info(
                "spatial memory ON: occupancy %dx%d, successor %dx%d, "
                "walkability %d bearings, probe map %dx%d",
                self._occ_grid, self._occ_grid, self._sr_grid, self._sr_grid,
                self._walk_bins, self._probe_grid, self._probe_grid)
        # ---- THE ORACLE (2026-09-19, roadmap O1) -------------------------
        # RED sensors exist to make the project measurable, never to be used.
        # Enabled purely from the sensor config; there is no world-model or
        # policy switch for it because there is no world-model or policy
        # PATH for it.
        _oc = (self.config.get("sensors") or {})
        self._oracle_enabled = "true_position" in list(_oc.get("enabled") or [])
        self._oracle_every = int(_oc.get("oracle_report_every", 500))
        self._oracle_drift = deque(maxlen=2000)
        self._oracle_steps = 0
        self._oracle_origin = None
        self._oracle_last = None
        self._oracle_travel = 0.0
        # THE COUNTERFACTUAL PROBE: "if I walked forward from here, how would
        # the scene sweep past me?" Reuses the vision scaffold's EXISTING
        # declaration of which actions are forward motion rather than adding a
        # second, driftable copy of that fact. First entry is plain walk.
        # THE KEY IS `llm.vision`, NOT `vision_scaffold` (fixed 2026-09-20).
        # This read a top-level key that does not exist and fell back to the
        # literal [1, 2, 6] — which happens to equal the configured value, so
        # nothing misbehaved and nobody noticed. That is the trap: change
        # llm.vision.forward_actions and the probe silently keeps aiming at
        # the old action. Found by AST-auditing every config.get() literal
        # against the live YAML; see tests/_config_keys_smoke.py.
        _fwd = list(((self.config.get("llm", {}) or {}).get("vision", {})
                     or {}).get("forward_actions", [1, 2, 6]) or [1])
        self._flow_probe_idx = int(
            (self.config.get("world_model", {}) or {}).get(
                "flow_probe_action", _fwd[0]))
        # THE CONFIG KEY IS `symbolic_grounding`, NOT `symbol_grounding`.
        # This read the wrong key from 2026-09-18 until it was caught on
        # 2026-09-20. It happened to be harmless — the fallback 0.4 equals
        # the configured value — but a silent fallback to a default is
        # exactly how a setting stops meaning anything, and the next person
        # to change fovea_frac would have found the flow senses ignoring it.
        self._flow_fovea_frac = float(
            (self.config.get("symbolic_grounding", {}) or {}).get(
                "fovea_frac", 0.4))
        if self._proprio_source is not None:
            logger.info("proprioception: %d dims from %s (%s)",
                        int(self._proprio_source.PROPRIO_DIM),
                        type(self._proprio_source).__name__,
                        ", ".join(self._proprio_source.PROPRIO_KEYS))
        self.action_dim = self.env.action_dim
        self.is_discrete = self.env.is_discrete

        # Per-DIMENSION symbolic machinery (the symbolic decoder builds one
        # classification head per obs dim; the fact extractor bins per obs
        # dim) only makes sense for LOW-dim structured observations. Skip it
        # for pixels (12288-d) AND high-dim structured vectors like
        # Craftax-Symbolic (8268-d) — otherwise it is ~thousands of heads of
        # per-element nonsense and a large silent per-step tax. Low-dim
        # symbolic envs (MiniGrid 151-d, CartPole 4-d) are unaffected.
        self._skip_perdim_symbolic = self.pixel_obs or self.obs_dim > 2000
        # Cadence for the symbolic-decoder read in the step loop when it IS
        # live (see the block in _collect_segment). 1 = every step = the
        # pre-2026-09-01 behaviour, so any config that wants it back can have
        # it; the KG consumers read at 25-50, which is where 25 comes from.
        self._sd_every = max(1, int(
            self.config.get("symbolic", {}).get("decoder_interval", 25)))
        # ---- DO SCOUT STREAMS FEED THE POLICY? (2026-09-01) --------------
        # LIFELONG ONLY, deliberately. The episodic body (_run_episode_parallel)
        # ends on the primary's terminal, so its scouts are cut mid-trajectory
        # at an arbitrary point with no bootstrap — storing those rows would
        # add truncation bias rather than data. The lifelong body has no
        # episode boundary at all, which is exactly what makes a scout's
        # trajectory as well-formed as the primary's.
        # Per-stream potential state for _scout_mixed_reward lives here too;
        # it is cleared per stream on that stream's death.
        self._scouts_in_ppo = bool(
            self.config.get("lifelong", {}).get("enabled", False)
            and self.config.get("policy", {}).get("scouts_in_ppo", True))
        self._sc_phi: Dict[int, Dict[str, float]] = {}
        # World-model gradient steps per block. Seeded from config; adapted
        # from the measured async skip rate only when train_iters_auto is on
        # (off by default — it changes the WM's data diet).
        _wmc = self.config.get("world_model", {}) or {}
        self._wm_train_iters = int(_wmc.get("train_iters", 4))
        self._train_iters_auto = dict(_wmc.get("train_iters_auto") or {})

        # Get human-readable labels for this environment
        self.obs_labels, self.action_labels = get_env_labels(self.env_name)

        logger.info(
            f"Environment: {self.env_name} | "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"discrete={self.is_discrete}"
        )

        # ---- Create World Model (RSSM) ----
        wm_cfg = self.config.get("world_model", {})
        _slot_cfg = self.config.get("slots") or {}
        # WIDTH OF THE BODY SENSE THE WORLD MODEL RECEIVES. Opt-in: a config
        # without `world_model.proprio` gets 0 and the model is built exactly
        # as before. Gated on the env actually HAVING a body vector, so
        # turning the flag on in an env with no proprioception is a no-op
        # rather than a shape error at the first forward pass.
        # ---- WIDTH COMES FROM THE SENSOR BUS (2026-09-19) ----------------
        # Was PROPRIO_DIM. The bus is the single declaration of what the
        # transport vector contains and in what order; reading the env's
        # proprio width here instead would silently truncate every sensor
        # registered after the first one.
        #
        # The bus is rebuilt here rather than read off the adapter because
        # the world model is constructed BEFORE any env exists, and the
        # layout is a pure function of config — so the two cannot disagree.
        self._sensor_bus = None
        self._sensor_image_specs = []
        _sen_cfg = self.config.get("sensors") or {}
        if (bool(wm_cfg.get("proprio", False))
                and getattr(self, "_proprio_source", None) is not None):
            if _sen_cfg.get("enabled") is not None:
                from developmental_ai.sensors import build_default_bus
                self._sensor_bus = build_default_bus(
                    int(self._proprio_source.PROPRIO_DIM),
                    lambda ctx: None,          # widths only; the ENV reads
                    enabled=list(_sen_cfg.get("enabled") or []),
                    fovea_size=int(_sen_cfg.get("fovea_size", 32)))
                self._wm_proprio_dim = int(self._sensor_bus.width)
                self._sensor_image_specs = [
                    (off, shp[0], shp[1], shp[2])
                    for _n, off, _w, shp in self._sensor_bus.image_layout()]
                # image_layout offsets are relative to the IMAGE block, which
                # begins after all vector fields.
                _vw = int(self._sensor_bus.vector_width)
                self._sensor_image_specs = [
                    (off + _vw, c, h, w)
                    for (off, c, h, w) in self._sensor_image_specs]
                self._sensor_layout_hash = self._sensor_bus.layout_hash()
                logger.info("sensor bus (world model): %s",
                            self._sensor_bus.describe())
            else:
                self._wm_proprio_dim = int(
                    self._proprio_source.PROPRIO_DIM)
                self._sensor_layout_hash = "proprio-only"
        else:
            self._wm_proprio_dim = 0
            self._sensor_layout_hash = "none"
        # WHERE EACH BODY FIELD LIVES, derived from the env's own key order
        # so it cannot drift from PROPRIO_KEYS. Empty when the env has no
        # body, which disables the depth parameterization rather than letting
        # it read someone else's vector by index.
        _pk = tuple(getattr(self._proprio_source, "PROPRIO_KEYS", ())
                    if getattr(self, "_proprio_source", None) is not None
                    else ())
        self._proprio_layout = {k: i for i, k in enumerate(_pk)}
        if self._wm_proprio_dim:
            logger.info(
                "world model receives PROPRIOCEPTION: %d dims (%s). It can "
                "now tell 'I turned my head' from 'the world spun' — the "
                "discrimination the flow head needs.",
                self._wm_proprio_dim,
                ", ".join(self._proprio_source.PROPRIO_KEYS))
        self.world_model = WorldModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            stochastic_size=wm_cfg.get("stochastic_size", 32),
            stochastic_classes=wm_cfg.get("stochastic_classes", 32),
            deterministic_size=wm_cfg.get("deterministic_size", 512),
            # NOTE: `world_model.decoder_hidden` is NOT passed, and cannot be
            # — WorldModel has one `hidden_dim`, and CNNDecoder accepts a
            # hidden_dim it never uses (its size comes from latent_dim and
            # the image_size channel ladder). Measured 2026-09-01: raising
            # decoder_hidden 512 -> 768 changed the decoder by 0 bytes. The
            # key is flagged at boot rather than left to imply a capacity
            # that does not exist — same treatment policy.batch_size gets.
            hidden_dim=wm_cfg.get("encoder_hidden", 256),
            learning_rate=wm_cfg.get("learning_rate", 1e-4),
            pixel_obs=self.pixel_obs,
            image_channels=self.image_channels,
            image_size=self.image_size,
            film_conditioning=bool(wm_cfg.get("film_conditioning", False)),
            inverse_dynamics=bool(wm_cfg.get("inverse_dynamics", False)),
            inverse_dynamics_scale=float(wm_cfg.get("inverse_dynamics_scale", 1.0)),
            discrete_actions=self.is_discrete,
            # ---- PERSPECTIVE FROM MOTION (2026-09-18) --------------------
            # See world_model/rssm.py's FlowHead block for the argument. All
            # six default to the historical behaviour, so the ten non-Minecraft
            # configs build exactly the model they built before.
            min_grid=int(wm_cfg.get("spatial_grid", 4)),
            coord_channels=bool(wm_cfg.get("coord_channels", False)),
            readout_channels=int(wm_cfg.get("readout_channels", 0)),
            flow_head=bool(wm_cfg.get("flow_head", False)),
            flow_size=int(wm_cfg.get("flow_size", 32)),
            flow_photo_size=int(wm_cfg.get("flow_photo_size", 0)),
            flow_weight=float(wm_cfg.get("flow_weight", 1.0)),
            flow_smooth_weight=float(wm_cfg.get("flow_smooth_weight", 0.05)),
            recon_residual_lambda=float(
                wm_cfg.get("recon_residual_lambda", 0.0)),
            latent_horizons=wm_cfg.get("latent_horizons") or [],
            latent_horizon_weight=float(
                wm_cfg.get("latent_horizon_weight", 1.0)),
            # ---- OBJECT SLOTS (roadmap R1-R4) -----------------------------
            # Gated on the flow head by construction (WorldModel refuses to
            # build them without it) AND shipped off, because slots decode
            # the flow field and a flow field that has not been shown to
            # learn on live frames is noise to decompose.
            slots=bool(_slot_cfg.get("enabled", False)),
            num_slots=int(_slot_cfg.get("num_slots", 6)),
            slot_dim=int(_slot_cfg.get("slot_dim", 64)),
            slot_iters=int(_slot_cfg.get("iters", 3)),
            slot_weight=float(_slot_cfg.get("weight", 1.0)),
            # ---- WAVE 2 (2026-09-18) -------------------------------------
            # Every default below reproduces Wave 1 term for term, which is
            # what makes the "every switch reverts" contract assertable.
            flow_mode=str(wm_cfg.get("flow_mode", "raw")),
            flow_automask=bool(wm_cfg.get("flow_automask", False)),
            flow_scales=wm_cfg.get("flow_scales") or [1],
            flow_strides=wm_cfg.get("flow_strides") or [1],
            camera_fov_deg=float(wm_cfg.get("camera_fov_deg", 70.0)),
            # The body's field layout comes from the ENV, which owns what its
            # proprio vector means — not from a constant in the world model
            # that would silently mis-read a different body.
            proprio_layout=self._proprio_layout,
            move_scale=float(getattr(
                getattr(self, "_proprio_source", None), "MOVE_SCALE", 1.0)),
            # THE ENV'S RAW BODY VECTOR, not the loop-augmented one. The
            # augmented vector carries the flow senses themselves (see
            # _augment_proprio), and feeding a model's own output back into
            # its input is circular. The raw env vector is also the only one
            # available at buffer-add time, which is where the column is
            # written.
            proprio_dim=self._wm_proprio_dim,
            sensor_image_specs=self._sensor_image_specs,
            sensor_feature_dim=int(_sen_cfg.get("feature_dim", 64)),
            # VRAM only — see the AMP block in rssm.py. CUDA-gated inside, so
            # these are inert on CPU and cannot move a single CPU-side test.
            amp_dtype=wm_cfg.get("amp_dtype"),
            grad_checkpoint=bool(wm_cfg.get("grad_checkpoint", False)),
        ).to(self.device)
        # Causal action alignment (ON by default since the July 2026 audit —
        # CODE_AUDIT_2026-07.md §A). When on, the WM learns true
        # action->next-state dynamics (state_t built from the action that
        # CAUSED o_t), fixing the off-by-one that made the prior predict the
        # next obs from the NEXT action (policy-correlation leak).
        self.world_model.action_shift = bool(wm_cfg.get("causal_align", True))

        # ---- Parallel environments (opt-in, OFF by default) ----
        # When enabled, multiple env copies are stepped in lockstep and the
        # neural ops are batched across them, while the replay buffer keeps one
        # independent stream per env. The single-env path is preserved exactly.
        par_cfg = self.config.get("parallel_envs", {})
        self._use_parallel_envs = par_cfg.get("enabled", False)
        self._num_envs = max(1, int(par_cfg.get("num_envs", 1)))
        if self._num_envs <= 1:
            self._use_parallel_envs = False
        # The early decision and this reading must agree, or the two readers
        # of parallel_envs have drifted and the guard above guards nothing.
        if bool(self._use_parallel_envs) != (
                self._collection_path.path != "single_env_legacy"):
            raise RuntimeError(
                "collection-path decision disagrees with _use_parallel_envs "
                f"({self._collection_path.path} vs {self._use_parallel_envs})")
        self._parallel_envs: List[Any] = []
        self._parallel_curricula: List[Any] = []

        # ---- FOUNDATION SHADOW RECORDER (plan §8 step 2, 2026-10-03) ----
        # OFF unless foundation.shadow.enabled (from_config returns None).
        # When on, it files each live step as foundation records (sensor
        # transport, executed action with its real timing, the world model's
        # prediction logged BEFORE env.step, episode endings) in an
        # EvidenceStore. It READS ONLY and draws no random numbers, so it
        # cannot move actions, rewards, replay, normalisation or RNG; it
        # never raises into the loop (one warning, then self-disabled for
        # the run; relaunch to re-enable). Wired into _run_episode_parallel
        # and _collect_segment with identical call text; NOT into
        # _run_episode. tests/_foundation_shadow_smoke.py holds all of this.
        from developmental_ai.foundation.runtime.shadow import ShadowRecorder
        self._shadow = ShadowRecorder.from_config(
            self.config, bus=self._sensor_bus, action_dim=self.action_dim,
            is_discrete=self.is_discrete, num_streams=self._num_envs,
            environment=self.env_name,
            layout_hash=getattr(self, "_sensor_layout_hash", None),
            action_repeat=int(env_cfg.get("action_repeat", 1) or 1))
        if (self._shadow is not None
                and self._collection_path.path == "single_env_legacy"):
            logger.warning("foundation.shadow is enabled but the single-env "
                           "body (_run_episode) is not wired to it: nothing "
                           "will be recorded on this path")

        # ---- LEARNING TELEMETRY + EXPLORATION (2026-10-07) ---------------
        # MEASUREMENT ONLY: runlogs/learning.jsonl (one record per segment,
        # written from _emit_metrics) and the evaluator-only position trace.
        # Built from config `telemetry:`; a failure here leaves both None and
        # the agent unchanged — a monitor must never stop the run starting.
        self._ltel = None
        self._explore_tracker = None
        try:
            _tcfg = dict(self.config.get("telemetry", {}) or {})
            if _tcfg.get("enabled", False):
                from developmental_ai.infra.learning_telemetry import \
                    LearningTelemetry
                from developmental_ai.foundation.runtime.shadow import \
                    action_sources as _lt_as
                self._lt_action_sources = _lt_as
                self._ltel = LearningTelemetry(
                    dict(_tcfg.get("learning", {}) or {}))
                _xcfg = dict(_tcfg.get("exploration", {}) or {})
                if _xcfg.get("enabled", True):
                    from developmental_ai.infra.exploration import \
                        ExplorationTracker
                    self._explore_tracker = ExplorationTracker(
                        cell_size=float(_xcfg.get("cell_size", 1.0)),
                        trace_every_steps=int(
                            _xcfg.get("trace_every_steps", 16)),
                        trace_path=_xcfg.get(
                            "trace_path", "runlogs/position_trace.jsonl"),
                        stationary_eps=float(
                            _xcfg.get("stationary_eps", 0.25)),
                        max_cells=_xcfg.get("max_cells"),
                        run_id=self._ltel.run_id)
                logger.info("learning telemetry ON -> %s | exploration %s",
                            getattr(self._ltel.sink, "path", None),
                            "ON" if self._explore_tracker else "off")
        except Exception as exc:
            self._ltel = None
            self._explore_tracker = None
            logger.warning("learning telemetry unavailable: %r", exc)

        # Replay buffer for world model training
        per_cfg = self.config.get("prioritized_replay", {})
        # uint8 obs storage (step 8): pixel obs are [0,1] -> quantize x255 for
        # a 4x RAM cut; vector envs keep float32 (obs may be negative).
        _obs_uint8 = bool(self.pixel_obs and wm_cfg.get("obs_uint8", True))

        # ---- Lifelong / continuous-stream mode (default OFF) ----
        from developmental_ai.core.lifelong import LifelongController
        self._ll_ctrl = LifelongController(self.config.get("lifelong", {}))
        assert self._lifelong == self._ll_ctrl.enabled  # set early, re-affirmed
        if self._lifelong and not self._use_parallel_envs:
            raise ValueError(
                "lifelong.enabled requires parallel_envs.enabled with "
                "num_envs>1; the single-env path is not covered by the "
                "continuous loop.")
        self._ll = None                 # the LifelongState, built at stream start
        self._stream_started = False
        self._stream_obs = None
        self._last_wm_metrics = None

        if self._use_parallel_envs:
            self.replay_buffer = MultiStreamReplayBuffer(
                num_streams=self._num_envs,
                capacity=wm_cfg.get("buffer_capacity", 100000),
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                per_alpha=per_cfg.get("alpha", 0.6),
                per_beta=per_cfg.get("beta", 0.4),
                per_epsilon=per_cfg.get("epsilon", 1e-2),
                obs_uint8=_obs_uint8,
                growth=wm_cfg.get("buffer_growth"),
                proprio_dim=self._wm_proprio_dim,
                sensor_layout=self._sensor_layout_hash,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=wm_cfg.get("buffer_capacity", 100000),
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                per_alpha=per_cfg.get("alpha", 0.6),
                per_beta=per_cfg.get("beta", 0.4),
                per_epsilon=per_cfg.get("epsilon", 1e-2),
                obs_uint8=_obs_uint8,
                growth=wm_cfg.get("buffer_growth"),
                proprio_dim=self._wm_proprio_dim,
                sensor_layout=self._sensor_layout_hash,
            )
        # Prioritized + asynchronous replay are opt-in. When both are off, the
        # buffer behaves exactly like the original uniform/synchronous sampler.
        self._use_per = per_cfg.get("enabled", False)
        self._use_async_replay = self.config.get("async_replay", {}).get(
            "enabled", False
        )
        self._bg_sampler = None  # lazily created BackgroundSampler
        # ---- Async world-model trainer ("retrospective consolidation") ----
        # Acting-while-learning: the WM's 128-iter gradient block runs on a
        # background thread so the primary stream never freezes at the
        # wm_train_every cadence (on a real multiplayer server the world keeps
        # ticking — a 20-50s inline block leaves the bot statue-still and
        # historically got it kicked). One-slot ticket queue, skip-if-busy:
        # tickets never backlog; a skipped tick reposts at the next cadence
        # point on fresher data. PPO stays synchronous (on-policy). Params are
        # guarded by _wm_param_lock at per-gradient-step granularity against
        # the main thread's per-step observe forwards, checkpoint saves, and
        # prospection rollouts. Default OFF: existing configs byte-identical.
        import queue as _queue
        import threading as _threading
        self._use_async_wm = self.config.get("async_wm", {}).get(
            "enabled", False)
        self._wm_trainer_thread = None
        self._wm_ticket = _queue.Queue(maxsize=1)
        self._wm_trainer_stop = _threading.Event()
        self._wm_busy = _threading.Event()
        self._wm_param_lock = _threading.RLock()
        self._wm_metrics_box = None  # trainer publishes by atomic ref-swap
        self._wm_metrics_fresh = False  # consume-once gate for run()'s users
        # replay-ratio instrumentation (equal-learning verdict): count cadence
        # points reached vs training blocks actually EXECUTED. skip_ratio =
        # 1 - executed/cadence; near 0 => async trains on ~every datum sync
        # would, i.e. equal effective replay ratio.
        self._wm_cadence_hits = 0
        self._wm_blocks_run = 0
        # ---- Goal-level prospection config (hook wired at broadcaster) ----
        self._prospection_cfg = dict(self.config.get("prospection", {}))
        # ---- Imagination curiosity (the drive that does not die) ----
        # LP curiosity is a DERIVATIVE of prediction error, so it necessarily
        # dies once a stationary world is mastered. These world-model-sourced
        # drives take over as external curiosity fades (boredom gate), and go
        # into the INTRINSIC channel ONLY so they can never become a replay
        # reward LABEL — the WM reward head must never learn from the agent's
        # own daydreams (that loop is how an agent learns to hallucinate its
        # own reward). See curiosity/imagination_curiosity.py.
        from developmental_ai.curiosity.imagination_curiosity import (
            ImaginationCuriosity)
        # territory-coverage weight (external curiosity; intrinsic channel)
        # #1/#3 persistence shaping (potential-based; see the note at its
        # application site). ticks = what a barehanded break actually costs.
        _cur_cfg = self.config.get("curiosity", {})
        self._persist_weight = float(_cur_cfg.get("persistence_weight", 0.0))
        self._persist_ticks = float(_cur_cfg.get("persistence_ticks", 80.0))
        self._persist_phi = 0.0
        # #7 approach-to-reach potential (telescoping; see its use site)
        self._reach_weight = float(_cur_cfg.get("reach_weight", 0.0))
        # GAZE LEVELING potential (telescoping; see its use site). The pitch
        # clamp is an attractor with no exit tax: pushing into it is a free
        # no-op and the gaze-bucket novelty that unsticks it SATURATES
        # (1/sqrt(n)), so a long life at the clamp reads as the cheapest
        # place to be. Quartic in |pitch|/90 so scanning and legitimate
        # mining tilts (<=60 deg, phi >= -0.2) cost ~nothing while the last
        # 30 degrees into a clamp carry the whole gradient.
        self._pitch_level_weight = float(_cur_cfg.get("pitch_level_weight",
                                                      0.0))
        self._pitch_level_phi = None
        # GUI-DWELL cost (plain difference; see its use site). Zeroing income
        # inside a menu stops the FARM but supplies no gradient OUT, and it
        # trained the one escape button to ~zero probability.
        self._gui_dwell_weight = float(_cur_cfg.get("gui_dwell_weight", 0.0))
        self._gui_dwell_steps = float(_cur_cfg.get("gui_dwell_steps", 200.0))
        self._gui_dwell_phi = None
        self._gui_run = 0
        # Validated HERE, not first inside the step loop (gui_costs raises
        # there too): a negative weight would flip the entry charge into an
        # entry wage, and a bad value should fail at boot, not mid-run.
        if (not math.isfinite(self._gui_dwell_weight)
                or self._gui_dwell_weight < 0.0
                or not math.isfinite(self._gui_dwell_steps)):
            raise ValueError(
                "curiosity.gui_dwell_weight must be finite and >= 0 and "
                "gui_dwell_steps finite: "
                f"{self._gui_dwell_weight}, {self._gui_dwell_steps}")
        # GUI-DWELL STEP COST (2026-10-05; see its use site). A genuine
        # per-step charge after a grace period, because the potential above
        # pays exactly 0 while pinned and so cannot discourage dwelling.
        # Negative values would be a WAGE for sitting in a menu — refused.
        self._gui_dwell_step_cost = float(_cur_cfg.get("gui_dwell_step_cost",
                                                       0.0))
        self._gui_dwell_grace_steps = int(_cur_cfg.get("gui_dwell_grace_steps",
                                                       40))
        if (not math.isfinite(self._gui_dwell_step_cost)
                or self._gui_dwell_step_cost < 0.0
                or self._gui_dwell_grace_steps < 0):
            raise ValueError(
                "curiosity.gui_dwell_step_cost and gui_dwell_grace_steps "
                "must be >= 0 (a negative cost pays the agent to sit in a "
                f"menu): {self._gui_dwell_step_cost}, "
                f"{self._gui_dwell_grace_steps}")
        # ---- PERCEPTUAL NOVELTY (see _perceptual_cell) --------------------
        # Count-based novelty over WHAT IS SEEN rather than WHERE THE BODY IS.
        self._novelty_weight = float(_cur_cfg.get("novelty_weight", 0.0))
        self._nov_bits = int(_cur_cfg.get("novelty_bits", 14))
        self._nov_counts: Dict[int, int] = {}
        self._nov_proj = None
        # SKY DISCOUNT (2026-08-07): clouds drift even under frozen time, so
        # a sky-filled gaze mints fresh view buckets forever — measured as
        # the largest itemised intrinsic term (+0.022/step) while the agent
        # dwelt at the -90 pitch clamp. Scale view-novelty by how little of
        # the FOVEA is sky (trust-gated; 1.0 when the fovea is ungated, so
        # the term is inert until the sense has earned labels). Floor keeps a
        # first-ever sky glance worth something — it is the FARM that dies,
        # not the sight.
        self._nov_sky_discount = bool(_cur_cfg.get(
            "novelty_sky_discount", False))
        # SEMANTIC NOVELTY (2026-08-11): scale this term by how new the
        # MEANING in view is (see _semantic_novelty_factor). Default OFF so
        # every other config keeps its exact old pricing.
        self._nov_semantic = bool(_cur_cfg.get("novelty_semantic", False))
        self._nov_unnamed_floor = float(_cur_cfg.get(
            "novelty_unnamed_floor", 0.1))
        # FROZEN HASH ENCODER (2026-08-07): _view_key hashed through the LIVE
        # shared encoder, which keeps training — so the bucket map itself
        # drifts and OLD sights slowly re-key as "new" (novelty that
        # replenishes without the world changing). Hash through a frozen
        # snapshot refreshed every N steps instead: stable buckets between
        # refreshes, representation still allowed to improve at refresh.
        # 0 = live encoder (old behaviour).
        # DECISIONS -> FRAMES (see _frames_per): compared against
        # total_timesteps, which advances by num_envs per decision.
        self._nov_enc_refresh = self._frames_per(int(_cur_cfg.get(
            "novelty_encoder_refresh", 0)))
        self._nov_enc = None
        self._nov_enc_stamp = -1
        # MASTERY HABITUATION (2026-08-08, see _habituation_factor): scale
        # of the familiarity decay applied to the step intrinsic when a
        # well-mastered block type breaks. 0 = off.
        self._habituation_scale = float(_cur_cfg.get(
            "break_habituation_scale", 0.0))
        self._habit_prev: Dict[str, int] = {}
        # ---- SYMBOLIC NOVELTY (see _symbol_novelty) ----------------------
        self._symbol_weight = float(_cur_cfg.get("symbol_weight", 0.0))
        self._new_symbol_bonus = float(_cur_cfg.get("new_symbol_bonus", 0.0))
        # ---- SEEING AND AIMING ARE ONE DRIVE (2026-08-02) ----------------
        # The symbol reward scores the SET of predicates currently above the
        # trust threshold, decaying as 1/sqrt(n) per distinct set. It is
        # POSITION-BLIND — a tree at the edge of frame and a tree dead-centre
        # pay identically — and it had decayed to ~0.01/step. So nothing in
        # the intrinsic channel rewarded TURNING TOWARD a symbol, and the
        # agent sat with its pitch clamped at +90 while the drive that was
        # supposed to pull it back paid the same either way.
        # This adds a POTENTIAL on centring the least-familiar thing the
        # agent can currently name. 0.0 = off (every other config).
        self._symbol_center_weight = float(
            _cur_cfg.get("symbol_center_weight", 0.0))
        # ---- GAZE COVERAGE (2026-08-02) ----------------------------------
        # Territory coverage counts the (x,z) CELLS the agent has stood in, so
        # standing still decays to ~0 and new ground pays. Nothing did the
        # same for WHERE IT LOOKS — and pitch is CLAMPED to [-90,+90], so an
        # agent that drifts to +90 finds that `lookDOWN` is a perfectly free
        # no-op (the world does not change, which also means zero prediction
        # error and zero curiosity cost) while `lookUP` is the only action
        # that moves. Measured: pitch pinned at +90 for 100k+ steps with
        # lookDOWN chosen ~2x more often than lookUP, so the net pressure was
        # downward into a wall that costs nothing to push against.
        # Same count-based form as territory: a pitch bucket seen often pays
        # ~0, an unvisited one pays ~1, decaying 1/sqrt(n).
        # PITCH ONLY, DELIBERATELY. Yaw is cyclic and unbounded, and spinning
        # is this project's known cheap-novelty failure mode (turnL 56% once
        # ate a whole run). Pitch spans ~12 buckets and SATURATES quickly, so
        # the term does its job and then gets out of the way.
        # "head" = the VLM-taught predicate (historic default); "evidence" =
        # ground truth from real break events. See _reach_sense for why the
        # head fails on this task.
        self._reach_source = str(_cur_cfg.get("reach_source", "head"))
        # INITIALISED HERE (fix 2026-08-04). This is assigned inside the
        # segment loop, but the FELT-REACH block reads it EARLIER in the same
        # loop body — so on the very first iteration it did not exist yet and
        # every launch died with AttributeError before completing one segment.
        # Three identical crashes in a row; the supervisor's deterministic-
        # fault guard correctly refused to keep restarting. Owning it at
        # construction removes the ordering dependency entirely rather than
        # guarding each read.
        self._last_world_info: Dict[str, Any] = {}
        self._gaze_weight = float(_cur_cfg.get("gaze_weight", 0.0))
        self._gaze_bucket_deg = float(_cur_cfg.get("gaze_bucket_deg", 15.0))
        self._gaze_counts: Dict[int, int] = {}
        self._known_symbols: set = set()
        self._symbol_counts: Dict[tuple, int] = {}
        self._reach_phi = 0.0
        self._coverage_weight = float(
            self.config.get("curiosity", {}).get("coverage_weight", 0.0))
        self._deaths_seen = 0
        _imag = dict(self.config.get("imagination_curiosity", {}))
        self.imagination_curiosity = ImaginationCuriosity(
            enabled=_imag.get("enabled", False),
            every=_imag.get("every", 50),
            rollouts=_imag.get("rollouts", 8),
            horizon=_imag.get("horizon", 10),
            disagreement_weight=_imag.get("disagreement_weight", 0.30),
            imagined_reward_weight=_imag.get("imagined_reward_weight", 0.10),
            imagined_reward_halflife=_imag.get(
                "imagined_reward_halflife", 300_000),
            lp_reference=_imag.get("lp_reference", 0.05),
            max_bonus=_imag.get("max_bonus", 0.5),
            # past-state baseline (2026-09-25) — see imagination_curiosity.py
            past_states=_imag.get("past_states", 32),
            baseline_every=_imag.get("baseline_every", 10),
            baseline_samples=_imag.get("baseline_samples", 4),
        )
        # Goal-prioritized replay: fraction of each WM/dream batch drawn from
        # windows that contain a nonzero reward, so the world model sees the rare
        # goal transition far more often (attacks the WM-accuracy bottleneck on
        # sparse-reward tasks). 0.0 = off (original behaviour).
        self._goal_replay_fraction = float(
            wm_cfg.get("goal_replay_fraction", 0.0))
        # HOW BIG A REWARD COUNTS AS "THE GOAL" (2026-09-18). The stratum
        # above used the sampler's 1e-3 default, i.e. "did anything pay at
        # all". On a SCOUT stream that is a sharp question — it stores the
        # raw env reward and _break_reward pays exactly 0.0 for everything
        # that is not a log or an ore. On the PRIMARY stream it is almost
        # vacuous: what gets stored is `prim_extrinsic`, which carries magnet
        # shaping, approach and goal-dwell on top of the env reward, and
        # those are nonzero on most steps. So the mechanism that exists
        # BECAUSE this run has ~35 log breaks in its entire history was
        # selecting from approximately every window.
        #
        # 5.0 is log-sized by construction (a log pays 20.0 + 0.5/tick, an
        # ore 10.0, and no shaping term comes close), so the stratum means
        # what its name says regardless of how the channels are mixed.
        # 1e-3 restores the old behaviour exactly.
        self._goal_replay_threshold = float(
            wm_cfg.get("goal_replay_threshold", 1e-3))
        self._wm_reward_pool = -1
        if self._goal_replay_fraction > 0.0:
            logger.info(
                "goal-prioritized replay: %.0f%% of each WM batch from "
                "windows paying > %.3g (log=20+0.5/tick, ore=10, all else 0)",
                100.0 * self._goal_replay_fraction,
                self._goal_replay_threshold)

        # ---- Create Curiosity Engine ----
        # mode: "novelty" (default; ICM raw prediction error) or
        #       "learning_progress" (reward the REDUCTION of prediction error per
        #       state bucket — ignores the unlearnable "noisy TV"). Same forward
        #       model + training either way; only the reward shape differs. This
        #       is the Rung-3 (autotelic) lever. Default keeps prior runs unchanged.
        cur_cfg = self.config.get("curiosity", {})
        self.curiosity_mode = cur_cfg.get("mode", "novelty")
        _cur_kwargs = dict(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            feature_dim=cur_cfg.get("feature_dim", 256),
            hidden_dim=cur_cfg.get("hidden_dim", 256),
            forward_loss_weight=cur_cfg.get("forward_loss_weight", 0.8),
            inverse_loss_weight=cur_cfg.get("inverse_loss_weight", 0.2),
            reward_scale=cur_cfg.get("curiosity_reward_scale", 1.0),
            reward_clip=cur_cfg.get("curiosity_reward_clip", 5.0),
            learning_rate=cur_cfg.get("learning_rate", 1e-4),
            # was silently ignored (audit): the YAML knob controlling the
            # normalization-pool length never reached the constructor.
            novelty_window=cur_cfg.get("novelty_window", 1000),
            discrete_actions=self.is_discrete,
            # CNN feature encoder on pixel envs (step 8; MLP-on-pixels had no
            # spatial prior and ~3.1M params in its first layer).
            pixel_obs=bool(self.pixel_obs and cur_cfg.get("cnn_encoder", True)),
            image_channels=self.image_channels,
            image_size=self.image_size,
        )
        if self.curiosity_mode == "learning_progress":
            self.curiosity = LearningProgressCuriosity(
                **_cur_kwargs,
                lp_sig_z=cur_cfg.get("lp_sig_z", 2.33),
                lp_rho_max=cur_cfg.get("lp_rho_max", 0.9),
                lp_visit_collapse=cur_cfg.get("lp_visit_collapse", True),
                lp_history=cur_cfg.get("lp_history", 30),
                lp_min_samples=cur_cfg.get("lp_min_samples", 4),
                lp_bucket_round=cur_cfg.get("lp_bucket_round", 1),
                lp_pixel_pool=cur_cfg.get("lp_pixel_pool", 4),
                lp_proto_thresh=cur_cfg.get("lp_proto_thresh", 0.025),
                lp_max_protos=cur_cfg.get("lp_max_protos", 512),
                # ABSOLUTE PROGRESS FLOOR: a mastered world must pay 0. See
                # LearningProgressCuriosity.__init__ for the measurement that
                # forced this (base term paid 95% of the drive for standing
                # still). 0.0 = off = byte-identical for every other config.
                lp_abs_frac=cur_cfg.get("lp_abs_frac", 0.0),
                # ACTION-CONDITIONAL: pay only for surprise the agent's own
                # action accounts for. False = off = byte-identical.
                action_conditional=cur_cfg.get("action_conditional", False),
                null_action=cur_cfg.get("null_action", 0),
                # PERSPECTIVE FROM MOTION: weight of the world model's flow
                # residual against the forward-model error when the two are
                # summed for BUCKETING. Only has an effect when a flow head
                # exists to supply one, so this is inert in every config that
                # has not opted in. 0.0 disables the channel without
                # rebuilding the world model, and is the revert.
                flow_error_weight=float(
                    cur_cfg.get("flow_error_weight", 1.0)),
            ).to(self.device)
            logger.info("LP gate: z=%s rho_max=%s visit_collapse=%s",
                        self.curiosity.lp_sig_z, self.curiosity.lp_rho_max,
                        self.curiosity.lp_visit_collapse)
            if cur_cfg.get("action_conditional", False):
                logger.info(
                    "ACTION-CONDITIONAL CURIOSITY ACTIVE: intrinsic is scaled "
                    "by how much of each transition the agent's OWN action "
                    "accounts for (null action=%d). Ambient change — day/night, "
                    "mobs, other players — no longer pays, and a no-op earns "
                    "EXACTLY zero by construction.",
                    int(cur_cfg.get("null_action", 0)))
            if float(cur_cfg.get("lp_abs_frac", 0.0)) > 0.0:
                logger.info(
                    "LP ABSOLUTE PROGRESS FLOOR ACTIVE: an error drop must "
                    "also exceed %.1f%% of the bucket's outstanding error to "
                    "count as learning — a mastered view now pays exactly 0",
                    100.0 * float(cur_cfg["lp_abs_frac"]))
        else:
            self.curiosity = IntrinsicCuriosityModule(**_cur_kwargs).to(self.device)

        # ---- BATCHED CURIOSITY TRAINING (2026-08-23) ---------------------
        # MEASURED: `curiosity.train_step` is 1519 aten ops per call and runs
        # EVERY env step at batch num_envs (2). With GPU utilisation at 2%,
        # that device is idle — the cost is ~2350 tiny kernel launches per
        # step, not arithmetic. Accumulating K steps and issuing ONE update
        # on the concatenated batch carries the same gradient information
        # with K-fold fewer launches (measured 1519 -> 95 ops/step at K=16).
        #
        # WHY BATCH AND NEVER SKIP: forward-model prediction error IS the
        # curiosity reward. Training the ICM less often makes novelty decay
        # more slowly, which is a live change to the reward economy — the
        # thing this project has been burned by repeatedly. Batching changes
        # only WHEN the optimizer steps.
        #
        # SAFE HERE because LearningProgressCuriosity does NOT override
        # train_step: the LP bucket/history bookkeeping lives entirely in
        # compute_intrinsic_reward, which stays per-step and untouched.
        # ICM.train_step is a pure forward/backward/step with no state.
        #
        # Default 1 = byte-identical to the previous behaviour.
        self._curiosity_train_every = max(
            1, int((self.config.get("curiosity", {}) or {}).get(
                "train_every", 1)))
        self._cur_train_buf: List[Any] = []
        self._last_icm_metrics: Dict[str, float] = {}

        # ---- A2: REUSE THE ENCODER FORWARD (2026-08-23) -------------------
        # arch='wm' means the policy encodes through world_model.encoder — the
        # SAME module the loop already ran on this frame one step earlier (as
        # `encoded_next`, for the RSSM). Handing that result forward removes a
        # duplicate forward per env step. See _feats_for_act/_set_enc_carry:
        # explicit dataflow, invalidated by an IDENTITY test on the observation
        # arrays, never a cache.
        #
        # verify_encoder_feats recomputes and compares on every use. It costs
        # the saving while on — that is the point: the first live run PROVES
        # the carry is correct before the check is turned off. Tolerance is
        # deliberate (the shared encoder trains asynchronously, so one
        # optimizer step of drift is legitimate; a wrong frame is O(1)).
        _pcfg = self.config.get("policy", {}) or {}
        self._reuse_enc_feats = bool(_pcfg.get("reuse_encoder_feats", False))
        self._verify_enc_feats = bool(
            self._reuse_enc_feats
            and _pcfg.get("verify_encoder_feats", True))
        self._enc_carry = None
        if self._reuse_enc_feats:
            logger.info(
                "policy: REUSING the world-model encoder forward across the "
                "step boundary (self-check %s)",
                "ON — proving correctness, saving disabled until it is off"
                if self._verify_enc_feats else "OFF")
        if self._curiosity_train_every > 1:
            logger.info(
                "curiosity: BATCHED training — one update per %d steps on "
                "the concatenated batch (same gradients, ~%dx fewer kernel "
                "launches)", self._curiosity_train_every,
                self._curiosity_train_every)

        # ---- VLM symbolic grounding ("knows how" -> "knows why") ----
        # Turns pixels into PREDICATES so the dormant symbolic stack (KG,
        # rule induction) has input. The VLM teaches a head that predicts
        # symbols from the agent's own latent, then steps back.
        sg_cfg = self.config.get("symbolic_grounding", {}) or {}
        self.symbolizer = None
        self._neg_evidence_ticks = 0
        self._symbol_clock = 0
        self._prev_mine: Dict[str, int] = {}
        self._symbolizer_retractions = 0
        # slot -> RSSM latent captured at the unlock moment. Becomes the
        # skill's context_embedding at mint: "what the world looked like
        # when this was first achieved" — the real when-does-this-apply
        # signal (the old one-hot said nothing).
        self._unlock_context: Dict[int, np.ndarray] = {}
        # slot -> POV frame at the unlock moment (discovery.png provenance
        # + the image the VLM names the skill from)
        self._unlock_frame: Dict[int, Any] = {}
        # GROUNDED SKILL NAMING (July 2026, replaces VLM hallucination):
        # per-env running block-break high-water, and the newest block each
        # env broke THIS step. A discovered achievement's grounded identity
        # is the block whose break caused its reward spike -> name it after
        # that block ("break_oak_log"), not the VLM's optimistic guess
        # ("craft_wooden_planks" / "fish_blue_water").
        self._mine_by_env: Dict[int, Dict[str, int]] = {}
        self._new_break_by_env: Dict[int, str] = {}
        self._unlock_block: Dict[int, str] = {}
        if sg_cfg.get("enabled", False) and self.pixel_obs:
            from developmental_ai.llm.vlm_symbolizer import VLMSymbolizer
            # Say which sensor is actually about to label the world, before a
            # single fact is minted from it (see probe_ollama_model).
            try:
                from developmental_ai.llm.llm_module import (
                    probe_ollama_model, set_ollama_endpoint)
                # Point the labeller at whatever node serves it. None keeps
                # the shipped resolution (OLLAMA_HOST, else localhost).
                set_ollama_endpoint(
                    (self.config.get("llm", {}) or {}).get("endpoint"))
                probe_ollama_model(sg_cfg.get("model", "llava:7b"))
            except Exception:
                pass
            self.symbolizer = VLMSymbolizer(
                model=sg_cfg.get("model", "llava:7b"),
                # KV-cache ceiling. Config-driven so the OOM fix is tunable
                # without an edit; see the VLMSymbolizer constructor for why
                # shrinking it is a SENSOR change, not just a memory knob.
                num_ctx=sg_cfg.get("num_ctx", 4096),
                interval=sg_cfg.get("interval", 60),
                max_interval=sg_cfg.get("max_interval", 600),
                latent_dim=self.world_model.rssm.latent_dim,
                hidden_dim=sg_cfg.get("hidden_dim", 256),
                lr=sg_cfg.get("lr", 1e-3),
                fact_threshold=sg_cfg.get("fact_threshold", 0.7),
                min_labels=sg_cfg.get("min_labels", 5),
                reliability_floor=sg_cfg.get("reliability_floor", 0.35),
                anneal_agreement=sg_cfg.get("anneal_agreement", 0.9),
                device=self.device,
                fovea=sg_cfg.get("fovea", False),
                fovea_frac=sg_cfg.get("fovea_frac", 0.4),
                fovea_interval=sg_cfg.get("fovea_interval", None),
                input_max_side=int(sg_cfg.get("input_max_side", 0) or 0),
                # ---- QUALITY OVER SPEED (2026-09-01) ------------------
                # Labels are the scarce resource, not parameters: replay
                # first, then capacity, then the ensemble that says when the
                # head does not know. Defaults reproduce the old behaviour
                # exactly (no replay, one member, hidden_dim, 3 layers).
                head_hidden=sg_cfg.get("head_hidden"),
                head_layers=int(sg_cfg.get("head_layers", 3)),
                head_members=int(sg_cfg.get("head_members", 1)),
                replay_capacity=int(sg_cfg.get("replay_capacity", 0)),
                replay_batch=int(sg_cfg.get("replay_batch", 64)),
                replay_steps=int(sg_cfg.get("replay_steps", 0)),
                max_disagreement=float(
                    sg_cfg.get("max_disagreement", 1.0)),
                reprobe_after_failures=int(
                    sg_cfg.get("reprobe_after_failures", 5)),
                # CROP-CONSISTENT FOVEA (2026-08-12): put the fovea head on
                # ENCODER FEATURES OF THE CROP rather than the whole-frame
                # RSSM latent, so its input describes the same region its
                # labels do. Width is the encoder's, not the RSSM's.
                #
                # ---- WAS `rssm.obs_dim` (fixed 2026-09-18) ---------------
                # That read was correct only while the identity
                # `rssm.obs_dim == encoder width` held. Feeding
                # proprioception into the embed BREAKS it: the RSSM is now
                # built with `hidden_dim + proprio_dim`, while _encode_crop
                # still returns bare encoder features. The head would have
                # been constructed 13 wider than its own input and thrown on
                # the first crop. `encoder.vec_dim` is the encoder's own
                # declaration of its width and cannot drift from it.
                fovea_latent_dim=(
                    int(getattr(self.world_model.encoder, "vec_dim",
                                self.world_model.rssm.obs_dim))
                    if sg_cfg.get("fovea_crop_input", True) else None))
            if sg_cfg.get("fovea_crop_input", True):
                self.symbolizer.set_crop_encoder(self._encode_crop)
            # Which actions count as "attacking" for the causal claim
            # ("attack_held" breaks X). Defaults to the vision scaffold's
            # chop actions, but is settable independently: causal grounding
            # must not silently die if the scaffold is turned off.
            _chop = sg_cfg.get("chop_actions")
            if _chop is None:
                _chop = (self.config.get("llm", {}) or {}).get(
                    "vision", {}).get("chop_actions", [])
            self._chop_action_set = frozenset(int(a) for a in (_chop or []))
            if not self._chop_action_set:
                logger.warning(
                    "symbolic_grounding: no chop_actions configured — break "
                    "events will mint properties but no causal facts")
            # NEGATIVE EVIDENCE gate: attack ticks with no break before the
            # claimed affordance predicates are docked. 0 = off.
            self._neg_evidence_ticks = int(sg_cfg.get(
                "neg_evidence_ticks", 0))
            self._neg_evt_scored = False
            logger.info("VLM symbolic grounding active (interval=%d, "
                        "chop_actions=%s, neg_evidence_ticks=%d)",
                        sg_cfg.get("interval", 60),
                        sorted(self._chop_action_set),
                        self._neg_evidence_ticks)

        # ---- Brain-state emitter (live skill-graph telemetry) ----
        # Writes brain_state.json for viewer/brain_viewer.html: the skill
        # graph (earned edges only) + the live firing feed. The user's
        # window into the agent's memory system.
        bv_cfg = self.config.get("brain_viewer", {}) or {}
        self.brain_emitter = None
        self._brain_clock = 0
        # NOTE: the emitter is CONSTRUCTED later (after skill_bank +
        # broadcaster + option_executor exist) — see the block after the
        # policy/mixer setup. It references those organs, so it cannot be
        # built here.

        # Configurable curiosity-vs-skill-count damper (both modes; icm.py note).
        self.curiosity.exploration_decay_rate = cur_cfg.get(
            "exploration_decay_rate", 0.1)

        # ---- Create Knowledge Graph (Neo4j if available, else in-memory) ----
        kg_cfg = self.config.get("knowledge_graph", {})
        self.knowledge_graph = create_knowledge_graph(kg_cfg)
        self.fact_extractor = FactExtractor(
            obs_labels=self.obs_labels,
            min_confidence=kg_cfg.get("min_confidence", 0.7),
        )

        # Graph embeddings (PyKEEN)
        self.kg_embedder = KGEmbeddingTrainer(
            embedding_dim=kg_cfg.get("embedding_dim", 64),
            model_name=kg_cfg.get("embedding_model", "TransE"),
            num_epochs=kg_cfg.get("embedding_epochs", 100),
        )

        # ---- Create Symbolic Decoder (Approach 2: latent → discrete categories) ----
        # This is the trained neural bridge between the RSSM's latent space
        # and the symbolic knowledge graph. Instead of decoding to continuous
        # observations and then discretizing (lossy), it learns sharp category
        # boundaries directly in latent space.
        sd_cfg = self.config.get("symbolic_decoder", {})
        # Pixel gate (July 2026, Crafter integration): the symbolic decoder
        # builds a per-OBS-DIMENSION head bank and mints per-dimension facts.
        # On a 64x64x3 pixel env that is ~12288 heads (~16M params) extracting
        # "facts" about individual pixels — semantically meaningless and a
        # large silent per-step tax. Pixel envs get a no-op stub; symbols for
        # such envs come from higher-level channels (achievements, rulebook).
        if self._skip_perdim_symbolic:
            self.symbolic_decoder = _NullSymbolicDecoder()
        else:
            self.symbolic_decoder = SymbolicDecoderManager(
                latent_dim=self.world_model.rssm.latent_dim,
                obs_dim=self.obs_dim,
                num_categories=sd_cfg.get("num_categories", 5),
                hidden_dim=sd_cfg.get("hidden_dim", 128),
                learning_rate=sd_cfg.get("learning_rate", 1e-4),
                confidence_threshold=sd_cfg.get("confidence_threshold", 0.6),
                obs_labels=self.obs_labels,
                device=self.device,
            )

        # ---- Create Glue Layer (neural-symbolic bridge) ----
        # Orchestrates: KnowledgeIntegrator (GNN+Gate), GoalGenerator,
        # AdvancedMasteryDetector, AdvancedCurriculumManager, SkillSelector
        glue_cfg = self.config.get("glue", {})
        self.glue = GlueLayer(
            latent_dim=self.world_model.rssm.latent_dim,
            obs_dim=self.obs_dim,
            embedding_dim=kg_cfg.get("embedding_dim", 64),
            gnn_hidden_dim=glue_cfg.get("gnn_hidden_dim", 128),
            gnn_output_dim=glue_cfg.get("gnn_output_dim", 64),
            goal_embedding_dim=glue_cfg.get("goal_embedding_dim", 16),
            max_difficulty=10,
            device=self.device,
            # ---- DON'T BUILD THE GATE ON PIXELS (2026-09-01) -------------
            # Same flag that installs `_NullSymbolicDecoder`, for the same
            # reason. The gate is sized off latent_dim, so at the RSSM's 4352
            # it is ~154 MB of VRAM for a module that is in no optimizer,
            # runs under no_grad, and feeds a decoder that returns []. Its
            # job — letting symbolic knowledge modulate what the agent
            # computes — is done by KnowledgeConditioner in the policy, at
            # 1.7 MB, trained by the PPO loss, on a path that can actually
            # change behaviour.
            build_gate=not self._skip_perdim_symbolic,
            # Env-relative mastery competence band. Defaults (475/0) keep
            # CartPole behavior identical; negative-reward envs override these
            # (e.g. Acrobot: target=-100, floor=-500) so a strong negative
            # return still reads as high competence and can mint skills.
            reward_target=self.config.get("environment", {}).get(
                "mastery_reward_target", 475.0
            ),
            reward_floor=self.config.get("environment", {}).get(
                "mastery_reward_floor", 0.0
            ),
        )

        # ---- Create Skill Bank ----
        sb_cfg = self.config.get("skill_bank", {})
        self.skill_bank = SkillBank(
            storage_dir=sb_cfg.get("storage_dir", "./skill_bank_data"),
            min_practice_episodes=sb_cfg.get("min_practice_episodes", 3),
            policy_versions_kept=sb_cfg.get("policy_versions_kept", 5),
            mastery_milestone=sb_cfg.get("mastery_threshold", 0.8),
            # retrieval policy — how many prior skills a new goal may
            # consider, and how relevant one must be to seed the live policy.
            # Both keys shipped in every config with no reader at all; the
            # warm-start below hardcoded 3 / 0.7 instead.
            max_skills_loaded=sb_cfg.get("max_skills_loaded", 5),
            similarity_threshold=sb_cfg.get("similarity_threshold", 0.5),
        )
        self.mastery_detector = MasteryDetector(
            mastery_threshold=sb_cfg.get("mastery_threshold", 0.8),
            mastery_window=sb_cfg.get("mastery_window", 50),
            min_episodes=sb_cfg.get("min_episodes_before_save", 20),
        )

        # ---- Symbolic feature-conditioning (Path A) ----
        # When enabled, the KG knowledge vector is fed into the policy as a
        # gated input feature (the gate is trained by the policy loss), so the
        # symbolic layer can actually shape behavior instead of only feeding a
        # self-referential fact-extractor. Set symbolic.enabled=false to run the
        # obs-only baseline (the other arm of the load-bearing ablation).
        sym_cfg = self.config.get("symbolic", {})
        self.symbolic_enabled = sym_cfg.get("enabled", True)
        # Auto-disable the symbolic/KG layer on CONTINUOUS action spaces. It was
        # built for discrete gridworlds; on continuous actions it only produces
        # vacuous `action_continuous` facts and feeds the policy a noisy knowledge
        # vector — a 5-seed Hopper ablation showed it *significantly hurts* return
        # (curiosity-only 957 vs full-organism 706, p<0.01). Set
        # `symbolic.force_on_continuous: true` to override (e.g. to reproduce that
        # ablation). Discrete envs are unaffected.
        if (self.symbolic_enabled and not self.is_discrete
                and not sym_cfg.get("force_on_continuous", False)):
            logger.info(
                "Continuous action space detected — auto-disabling the symbolic "
                "layer (it taxes continuous control; set "
                "symbolic.force_on_continuous=true to override)."
            )
            self.symbolic_enabled = False
        # Rung 6 (Global-Workspace) lesion. When True the symbolic layer runs
        # IDENTICALLY to the intact arm (KG builds, the knowledge vector is still
        # computed, the policy keeps the same input shape + gate params) but the
        # vector's CONTENT is severed to zeros before it reaches the policy. This
        # isolates the broadcast *channel* from its *content*: if behaviour
        # degrades vs intact, information genuinely flows KG -> action. Default
        # False so every prior run is byte-for-byte unchanged.
        self.symbolic_broadcast_lesion = bool(sym_cfg.get("broadcast_lesion", False))
        # Lesion CONTENT mode (Rung-6 content-vs-presence controls). The plain
        # "zero" lesion hands the policy an all-zeros vector — clean, but OOD
        # under force-concat training (the policy never saw zeros), so a drop vs
        # intact could be dismissed as the policy breaking on a never-seen input.
        # The in-distribution controls remove that confound:
        #   "none"     — intact (no lesion).
        #   "zero"     — all-zeros (presence severed; may be OOD).
        #   "scramble" — a VALID canonical phase vector for the WRONG phase
        #                (in-distribution, wrong info) -> tests CONTENT use.
        #   "constant" — a FIXED valid phase vector regardless of state
        #                (in-distribution, no variation) -> tests need for VARIATION.
        # Back-compat: if broadcast_lesion=True but no mode given, default "zero";
        # if a mode in {zero,scramble,constant} is given, the lesion is active even
        # without broadcast_lesion=True. verify_broadcast_live.py flips
        # self.symbolic_broadcast_lesion directly with mode "none" -> treat as zero.
        self.broadcast_lesion_mode = str(
            sym_cfg.get("broadcast_lesion_mode",
                        "zero" if self.symbolic_broadcast_lesion else "none")
        ).lower()
        if self.broadcast_lesion_mode in ("zero", "scramble", "constant", "noise"):
            self.symbolic_broadcast_lesion = True
        # Dedicated RNG for the scramble control so the wrong-phase draw is
        # reproducible per seed and never perturbs the policy / env RNG streams.
        self._lesion_rng = np.random.RandomState(
            int(self.seed) if self.seed is not None else 0
        )
        # Which channel feeds the policy broadcast:
        #   "kg"       — GNN mean-pool over the knowledge graph (original; a global
        #                vector recomputed every 50 episodes -> constant within an
        #                episode, which is why the gate never opened on DoorKey).
        #   "episodic"  — within-episode working memory of last-seen key/door/goal
        #                 locations (varies every step, carries info the partial
        #                 observation lacks). Same gate / lesion / off plumbing.
        #   "affordance"— compact symbolic TASK-PHASE / rule vector (the
        #                 "locked-door => needs-key" affordance), encoding WHAT
        #                 the task state is, not WHERE anything is. Grid-size /
        #                 layout invariant, so it is the natural unit of CROSS-TASK
        #                 TRANSFER: a phase signal learned on an easy env transfers
        #                 to a harder one. (Rung 6 redesign — knowledge, not memory.)
        #   "none"     — no broadcast channel (DEFAULT since the July 2026
        #                audit, H5): the KG->GNN path was never trained (frozen
        #                gate at sigmoid(-2), no optimizer over the GNN) and its
        #                PyKEEN embeddings rotate arbitrarily every retrain, so
        #                it injected cost + noise, not knowledge. Set "kg"
        #                explicitly to reproduce the old behavior; the KG store,
        #                fact extraction, and symbolic decoder are unaffected.
        self.knowledge_source = sym_cfg.get("knowledge_source", "none")
        # Unified broadcaster handle (episodic OR affordance); kept as
        # self.broadcaster so the reset/update/feature hooks are source-agnostic.
        self.broadcaster = None
        if self.symbolic_enabled and self.knowledge_source == "episodic":
            self.broadcaster = EpisodicWorkingMemory()
            knowledge_dim = EpisodicWorkingMemory.DIM
        elif self.symbolic_enabled and self.knowledge_source == "affordance":
            self.broadcaster = AffordanceBroadcast()
            knowledge_dim = AffordanceBroadcast.DIM
        elif self.symbolic_enabled and self.knowledge_source == "rule_regime":
            # Hidden-affordance-regime knowledge for the Rung-6 DECISIVE test
            # (costly perceptual aliasing). Carries the active rule, which the
            # observation provably lacks. See core/rule_regime_broadcast.py.
            self.broadcaster = RuleRegimeBroadcast()
            knowledge_dim = RuleRegimeBroadcast.DIM
        elif self.symbolic_enabled and self.knowledge_source == "goal":
            # Achievement GOAL channel (rich-env program, step 1): a REAL goal
            # representation the policy consumes — target slot + achieved mask,
            # IMGEP frontier targeting via the rung-5 CompetencePredictor.
            # goals.mode: "given" (oracle achievements — privileged arm) or
            # "discovered" (reward-spike + obs-delta clustering — earned arm).
            # See core/achievement_goals.py.
            from developmental_ai.core.achievement_goals import (
                make_goal_broadcast)
            _goals_cfg = dict(self.config.get("goals", {}))
            # LIFELONG working-set width: active_goals is the single knob for
            # the goal registry size (the resident set). It OVERRIDES
            # goals.max_slots so raising one variable widens the organism (the
            # 24/48 widening). Paging keeps TOTAL goals unbounded regardless.
            if self._lifelong:
                _goals_cfg["max_slots"] = int(self._ll_ctrl.active_goals)
            self.broadcaster = make_goal_broadcast(_goals_cfg)
            # GOAL-LEVEL PROSPECTION: before committing to a target at each
            # goal horizon, imagine pursuing the frontier's top-M candidates
            # from the current stream state and blend the imagined payoff into
            # the selection (see AchievementGoalBroadcast._select_target).
            if self._prospection_cfg.get("enabled", False):
                self.broadcaster.prospection_hook = self._prospective_goal_score
                self.broadcaster.prospection_weight = float(
                    self._prospection_cfg.get("weight", 0.5))
                self.broadcaster.prospection_top_m = int(
                    self._prospection_cfg.get("top_m", 4))
            # STABLE SKILL IDENTITY (July 2026): restore the slot registry,
            # cluster signatures and competence self-model from the skill
            # bank's directory. Without this every process relabelled
            # discovered_N from 0 in spike order, so the same name meant
            # different behaviours across runs — and skill upserts clobbered
            # the wrong directories.
            self._broadcaster_state_path = os.path.join(
                self.skill_bank.storage_dir, "broadcaster_state.json")
            # THE PROBE ANSWERS "does slot i already have a skill?" from the
            # BANK, which is the only authority on it. Match by the stable
            # `ach_<NN>_` index prefix, never by slot NAME: names drift as
            # goals re-key (slot 0 is `discovered_0` today but its skill is
            # `ach_00_break_birch_leaves`), and a name-keyed lookup silently
            # answers "no skill" for every renamed slot.
            def _slot_has_skill(idx: int, _name: str) -> bool:
                pfx = f"ach_{int(idx):02d}_"
                return any(sid.startswith(pfx)
                           for sid in self.skill_bank.skills)
            self.broadcaster.load_state(self._broadcaster_state_path,
                                        minted_probe=_slot_has_skill)
            # LIFELONG working-set paging (opt-in via lifelong.paging.enabled):
            # attach an unbounded on-disk store so a FULL goal registry pages
            # the stalest low-value slot OUT (evict-on-disuse) and recalls a
            # re-encountered behaviour back IN (recall-on-cue) instead of
            # refusing new discoveries at the max_slots cap. Without this the
            # registry keeps its fixed-cap behaviour, byte-identical.
            self._goal_ltm = None
            self._page_events_total = {"out": 0, "in": 0}
            _pcfg = (self.config.get("lifelong", {}).get("paging", {}) or {})
            if (_pcfg.get("enabled", False)
                    and hasattr(self.broadcaster, "attach_long_term_store")):
                from developmental_ai.memory.long_term_store import (
                    LongTermStore)
                _cue_dim = int(getattr(self.broadcaster, "pool", 192))
                self._goal_ltm = LongTermStore(
                    os.path.join(self.skill_bank.storage_dir, "ltm_goals"),
                    cue_dim=_cue_dim)
                self.broadcaster.attach_long_term_store(
                    self._goal_ltm,
                    min_idle_episodes=int(_pcfg.get("min_idle_episodes", 8)),
                    recall_threshold=float(
                        _pcfg.get("recall_threshold", 0.8)))
                logger.info(
                    "Working-set paging ON: goal registry <-> ltm_goals "
                    "(cue_dim=%d, min_idle=%d)", _cue_dim,
                    int(_pcfg.get("min_idle_episodes", 8)))
            knowledge_dim = self.broadcaster.DIM
        elif self.symbolic_enabled and self.knowledge_source == "kg":
            knowledge_dim = self.glue.knowledge_integrator.gnn_output_dim
        else:
            # "none" (default) or symbolic off: no policy broadcast channel.
            knowledge_dim = 0
        # Back-compat alias (older code paths reference self.episodic_memory).
        self.episodic_memory = (
            self.broadcaster
            if isinstance(self.broadcaster, EpisodicWorkingMemory)
            else None
        )
        # Broadcast gate mode: "learned" (default; the gate decides how much
        # knowledge to admit) or "open" (force-concat; gate fixed fully open so
        # the knowledge reaches the policy unattenuated). Force-concat is used by
        # the cross-task transfer test so the source policy wires the broadcast in
        # unconditionally and lesioning its content at transfer time genuinely
        # bites. Default "learned" keeps prior ablations unchanged.
        self.broadcast_gate_mode = sym_cfg.get("broadcast_gate", "learned")

        # ---- Create Policy ----
        pol_cfg = self.config.get("policy", {})
        # SKILLS AS OPTIONS (July 2026, Layer 4): widen the discrete head by
        # K invocation slots. self.action_dim stays P EVERYWHERE ELSE — the
        # world model, curiosity, replay and dream actor only ever see
        # executed primitives (transition-semantics invariant, E7).
        opt_cfg = self.config.get("skills_as_options", {}) or {}
        # LIFELONG: active_skills is the single knob for the resident skill
        # (option) set — it OVERRIDES k_slots so the 24/48 widening is one
        # variable. Paging keeps TOTAL skills unbounded on disk regardless.
        _k_default = (int(self._ll_ctrl.active_skills) if self._lifelong
                      else int(opt_cfg.get("k_slots", 8)))
        _k_slots = _k_default if opt_cfg.get("enabled", False) else 0
        if _k_slots and not (self._use_parallel_envs and self.is_discrete):
            logger.warning("skills_as_options requires the parallel discrete "
                           "path — disabled for this run")
            _k_slots = 0
        self.meta_action_dim = self.action_dim + _k_slots
        # ---- policy arch (v3): conv trunk for pixel envs -------------------
        # `arch: conv` shares one CNN across actor/critic/conditioner (~16x
        # smaller skills, spatial prior). Guarded: conv REQUIRES flat-CHW
        # pixel obs; anything else falls back to flat with a warning rather
        # than crashing a run on a config copied between environments.
        _arch = str(pol_cfg.get("arch", "flat"))
        if _arch in ("conv", "wm"):
            _side = int(round((self.obs_dim / 3) ** 0.5))
            if 3 * _side * _side != int(self.obs_dim):
                logger.warning("policy.arch=%s needs 3*s*s pixel obs "
                               "(obs_dim=%d) — falling back to flat",
                               _arch, self.obs_dim)
                _arch = "flat"
        # arch="wm" reads the WORLD MODEL's encoder, so its feature width is
        # not a free parameter — it IS world_model.encoder_hidden. Deriving it
        # here (instead of trusting policy.enc_dim) makes the two impossible
        # to set inconsistently; attach_shared_encoder still verifies with a
        # real forward pass.
        _enc_dim = int(pol_cfg.get("enc_dim", 256))
        if _arch == "wm":
            _wm_hidden = int(self.config.get("world_model", {}).get(
                "encoder_hidden", 256))
            if _enc_dim != _wm_hidden:
                logger.info(
                    "policy.arch=wm: enc_dim %d -> %d (world_model."
                    "encoder_hidden) — perception width is the world model's",
                    _enc_dim, _wm_hidden)
            _enc_dim = _wm_hidden
            if not self.pixel_obs:
                logger.warning("policy.arch=wm needs pixel obs — the world "
                               "model encoder is a CNN. Falling back to flat.")
                _arch = "flat"
        elif _arch == "rssm":
            # ---- THE POLICY SEES THE RSSM (2026-09-01) -------------------
            # Width is the world model's LATENT, not a free parameter and not
            # the encoder's: h (deterministic) concatenated with the
            # flattened categorical z. Derived here so it cannot be set
            # inconsistently with the world model that produces it.
            _enc_dim = int(self.world_model.rssm.latent_dim)
            logger.info(
                "policy.arch=rssm: the actor reads the world model's latent "
                "(%d = deterministic + stochastic*classes), not encoder "
                "features of one frame. The policy is no longer memoryless.",
                _enc_dim)
            # The carried-encoder-features optimisation and its self-check are
            # wm-only: the latent is produced fresh per env per step and is
            # never carried across a step boundary, so there is nothing to
            # verify and _encode would raise if asked.
            self._reuse_enc_feats = False
            self._verify_enc_feats = False
        self.policy = StandaloneActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.meta_action_dim,
            # ---- WAS HARDCODED (2026-09-01) ------------------------------
            # 256 was chosen when arch='wm' fed the actor 512-d encoder
            # features — a 2:1 squeeze into the trunk. arch='rssm' made the
            # input 4352 + 17 proprio + 98 knowledge = 4467, i.e. a 17:1
            # compression through one Linear, and nobody revisited it. Left
            # at 256 so this change is a no-op; raised only with a
            # measurement behind it (Crafter is where that is cheap).
            hidden_dim=int(pol_cfg.get("hidden_dim", 256)),
            continuous=not self.is_discrete,
            learning_rate=pol_cfg.get("learning_rate", 3e-4),
            gamma=pol_cfg.get("gamma", 0.99),
            gae_lambda=pol_cfg.get("gae_lambda", 0.95),
            clip_range=pol_cfg.get("clip_range", 0.2),
            # ---- PPO KNOBS THAT HAD NO READER (2026-08-02) ---------------
            # train_step did `n_epochs` FULL-batch passes and nothing else, so
            # a 1024-step rollout bought ~10 gradient steps where standard PPO
            # takes ~160. entropy_coef and value_coef were likewise pinned at
            # their signature defaults and could not be tuned at all — on a run
            # whose failure mode is action-distribution collapse.
            #
            # DELIBERATELY A NEW KEY, NOT `policy.batch_size`. batch_size is
            # set in all eleven configs and has never been read, so honouring
            # it now would silently switch MiniGrid, Crafter, Craftax and every
            # control task to a different learning regime mid-project and
            # invalidate their recorded results. `minibatch_size` is opt-in;
            # absent -> 0 -> the full-batch path, unchanged. The stale
            # batch_size key is reported at boot instead of quietly lying.
            minibatch_size=int(pol_cfg.get("minibatch_size", 0) or 0),
            # THE MISSING TRUST REGION — see StandaloneActorCritic.__init__.
            # 0.0 = off = byte-identical for every config that omits it.
            target_kl=float(pol_cfg.get("target_kl", 0.0) or 0.0),
            # HARD exploration floor — see StandaloneActorCritic.__init__ for
            # why a soft entropy bonus provably cannot do this job.
            logit_range=float(pol_cfg.get("logit_range", 0.0) or 0.0),
            entropy_coef=float(pol_cfg.get("entropy_coef", 0.01)),
            value_coef=float(pol_cfg.get("value_coef", 0.5)),
            knowledge_dim=knowledge_dim,
            knowledge_gate_mode=self.broadcast_gate_mode,
            device=self.device,
            arch=_arch,
            enc_dim=_enc_dim,
            # PROPRIOCEPTION: width comes from the ENV, so an env with no
            # body sense yields 0 and the policy is byte-identical to before.
            # +1 for the loop-computed REACH sense (see _reach_sense). The
            # env supplies the body; this one field is perceptual and needs
            # the RSSM latent, which only the loop has.
            # +4 loop-side senses when grounding exists: reach + the three
            # episodic-bearing fields (validity*proximity, sin, cos) — see
            # _augment_proprio (2026-08-09; was +1 reach-only)
            # +4 when a symbolizer exists (reach + the three episodic-bearing
            # fields) and +4 more when the flow head exists (flow_fovea,
            # flow_edge, flow_ratio, mover). MUST mirror _augment_proprio
            # field-for-field: that function is the only writer, this is the
            # only declaration of the width, and a mismatch is silent — the
            # policy would read four senses shifted by four positions.
            proprio_dim=(int(getattr(
                getattr(self, "_proprio_source", None), "PROPRIO_DIM", 0))
                + (4 if getattr(self, "_proprio_source", None) is not None
                   and self.symbolizer is not None else 0)
                + ((4 + self._surprise_grid * self._surprise_grid)
                   if getattr(self, "_proprio_source", None) is not None
                   and bool(getattr(self.world_model, "flow_enabled", False))
                   else 0)
                + self._spatial_width()),
            # ROW-AWARE UPDATE TRIGGER (2026-09-01) — see should_update().
            # 0/0 keeps the historical env-step-only behaviour.
            min_rows_per_update=int(
                pol_cfg.get("min_rows_per_update", 0) or 0),
            max_env_steps_per_update=int(
                pol_cfg.get("max_env_steps_per_update", 0) or 0),
            max_rows_per_update=int(
                pol_cfg.get("max_rows_per_update", 0) or 0),
            # Same gate and same scheme as the replay buffer's obs_uint8:
            # pixels only, x255 round-trip. Defaults to whatever the buffer
            # does so the two cannot silently disagree about how a frame is
            # carried.
            rollout_uint8=bool(
                self.pixel_obs
                and pol_cfg.get("rollout_uint8",
                                wm_cfg.get("obs_uint8", True))),
        )
        # A CONFIG KEY THAT DOES NOTHING MUST SAY SO — second instance
        # (2026-09-01). `world_model.decoder_hidden` is read by nothing in the
        # tree, and CNNDecoder ignores the hidden_dim it is handed anyway.
        # Found by MEASURING a capacity change rather than trusting it:
        # raising it 512 -> 768 moved the decoder by exactly 0 bytes.
        if (wm_cfg.get("decoder_hidden") is not None
                and int(wm_cfg["decoder_hidden"])
                != int(wm_cfg.get("encoder_hidden", 256))):
            logger.warning(
                "world_model.decoder_hidden=%s is NOT read by anything (it "
                "never has been) — WorldModel takes ONE hidden_dim, sourced "
                "from encoder_hidden=%s, and CNNDecoder's size comes from "
                "latent_dim + the image_size channel ladder. Raising it "
                "changes no parameters. Set encoder_hidden to size the "
                "world model.",
                wm_cfg["decoder_hidden"], wm_cfg.get("encoder_hidden", 256))
        # A CONFIG KEY THAT DOES NOTHING MUST SAY SO. `policy.batch_size` has
        # been present and unread since the first config; silence made it look
        # honoured. See minibatch_size above for why it is not simply adopted.
        if pol_cfg.get("batch_size") and not pol_cfg.get("minibatch_size"):
            logger.warning(
                "policy.batch_size=%s is NOT read by the PPO update (it never "
                "has been) — PPO runs %s full-batch passes per rollout. Set "
                "policy.minibatch_size to enable real minibatching.",
                pol_cfg.get("batch_size"), pol_cfg.get("n_epochs", 10))
        if float(pol_cfg.get("logit_range", 0.0) or 0.0) > 0.0:
            logger.info(
                "LOGIT RANGE BOUNDED at %.1f: the action distribution can "
                "become confident but never CERTAIN (max_prob is capped), so "
                "the entropy gradient never vanishes and a collapsed policy "
                "can always recover. Saturation is a one-way door: at "
                "max_prob=1.000 the entropy gradient is 9.5e-07, five orders "
                "of magnitude below healthy — no entropy_coef can reverse it.",
                float(pol_cfg["logit_range"]))
        if int(pol_cfg.get("minibatch_size", 0) or 0) > 0:
            logger.info(
                "PPO MINIBATCHING ACTIVE: minibatch=%d over %d-step rollouts "
                "x %d epochs — ~%d gradient steps per update instead of %d",
                int(pol_cfg["minibatch_size"]),
                int(pol_cfg.get("n_steps", 2048)),
                int(pol_cfg.get("n_epochs", 10)),
                int(pol_cfg.get("n_epochs", 10)) * max(1, int(
                    pol_cfg.get("n_steps", 2048))
                    // int(pol_cfg["minibatch_size"])),
                int(pol_cfg.get("n_epochs", 10)))

        # ---- ONE VENTRAL STREAM (arch="wm") -------------------------------
        # Hand the policy the world model's encoder. From here the agent has
        # a SINGLE visual system: trained by prediction on every frame of
        # every stream, read by the world model, the dream actor and now the
        # waking policy alike. Before this, the policy carried its own 50 MB
        # pixel layer trained only by PPO gradients — measured at 86% of
        # weights still inside their init bound, i.e. it was acting on a
        # near-random projection while a trained one sat unused next door.
        if _arch == "wm":
            self.policy.attach_shared_encoder(self.world_model.encoder)
            logger.info(
                "SHARED PERCEPTION: policy reads world_model.encoder "
                "(%d features, detached — PPO cannot reshape it)", _enc_dim)
        # STATE THE ARCH AT BOOT. Verifying "is the conv trunk actually live"
        # otherwise means inferring it from the config plus the absence of a
        # fallback warning, or waiting for a checkpoint to measure its size —
        # both indirect, on a run that lasts weeks. One line, greppable.
        _npar = sum(p.numel() for m in (self.policy.actor, self.policy.critic,
                                        self.policy.encoder,
                                        self.policy.conditioner)
                    if m is not None for p in m.parameters())
        logger.info("Policy arch=%s enc_dim=%s params=%.2fM (obs_dim=%d, "
                    "head=%d)", self.policy.arch,
                    # enc_dim is meaningful for the whole ENCODED family, not
                    # just conv — printing "-" under arch=wm would report the
                    # shared-perception width as absent, i.e. a diagnostic
                    # that lies about the thing it exists to show.
                    self.policy.enc_dim
                    if self.policy.arch in ("conv", "wm", "rssm") else "-",
                    _npar / 1e6, self.obs_dim, self.meta_action_dim)

        # PPO updates on accumulated rollouts (~n_steps), NOT one short episode.
        # Per-episode training on ~9 samples gave a frozen, zero-gradient policy;
        # batching across episodes restores stable gradients and real learning.
        self.policy_update_steps = pol_cfg.get("n_steps", 2048)

        if bv_cfg.get("enabled", False):
            from developmental_ai.skill_bank.brain_state import (
                BrainStateEmitter)
            _bv_dir = bv_cfg.get("out_dir", "runlogs/brain")
            os.makedirs(_bv_dir, exist_ok=True)
            self.brain_emitter = BrainStateEmitter(
                self.skill_bank,
                broadcaster=self.broadcaster,
                knowledge_graph=self.knowledge_graph,
                symbolizer=self.symbolizer,
                long_term_store=getattr(self, "_goal_ltm", None),
                out_path=os.path.join(_bv_dir, "brain_state.json"))
            self._brain_interval = int(bv_cfg.get("interval", 400))
            # make the served directory self-contained: viewer page + a
            # link to the skill notes (discovery.png in note panels)
            try:
                import shutil as _sh
                _src = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__)))),
                    "viewer", "brain_viewer.html")
                if os.path.exists(_src):
                    _sh.copy(_src, os.path.join(_bv_dir,
                                                "brain_viewer.html"))
                _notes = os.path.join(_bv_dir, "notes")
                if not os.path.exists(_notes):
                    os.symlink(os.path.abspath(
                        self.skill_bank.storage_dir), _notes)
            except OSError:
                pass
            # Legacy skills predate notes: give them thin, honest notes so
            # the brain's first render is not a field of anonymous nodes.
            # Their real names/preconditions arrive at their next unlock.
            try:
                from developmental_ai.skill_bank.skill_notes import (
                    backfill_notes, reground_names)
                # strip stale VLM-hallucinated names from prior runs FIRST,
                # so the brain shows honest names from the first render
                reground_names(self.skill_bank)
                backfill_notes(self.skill_bank, self.broadcaster,
                               self.knowledge_graph)
            except Exception as _e:
                logger.warning("note backfill skipped: %s", _e)
            logger.info("Brain viewer emitter active -> %s", _bv_dir)

        # Options executor (None when disabled). Brain-emitter firing hooks
        # are wired after the emitter exists (post-construction, below).
        self.option_executor = None
        if _k_slots:
            from developmental_ai.policy.options import (
                OptionExecutor, SkillOptionBank)
            _n_envs = int(self.config.get("parallel_envs", {}).get(
                "num_envs", 1))
            _ob = SkillOptionBank(
                self.skill_bank, self.obs_dim, self.action_dim, _k_slots,
                self.device, opt_cfg.get("cache", {}))
            # ---- SKILLS AS MEMORIES ---------------------------------------
            # Off by default: with practice disabled a bound skill stays
            # frozen for the run exactly as before, so every other config is
            # unaffected. Enabled, an invocation that produces the skill's own
            # grounded effect trains that skill toward what it just did, under
            # a KL trust region, with consolidation slowing proven skills and
            # decay weakening unrehearsed ones.
            # PHASE-SCOPED ACTION SPACE: indices the policy may not pick.
            _dis = [int(x) for x in
                    (self.config.get("environment", {}).get(
                        "disable_macros") or [])]
            if _dis:
                _ob.disabled_macros = set(_dis)
                logger.info(
                    "ACTION SPACE SCOPED: macros %s disabled for this phase "
                    "(indices preserved; the actions are simply not offered)",
                    sorted(_dis))
            _pr_cfg = dict(opt_cfg.get("practice", {}) or {})
            if _pr_cfg.get("enabled", False):
                from developmental_ai.skill_bank.skill_practice import (
                    SkillPractice)
                _ob.practice = SkillPractice(
                    enabled=True,
                    lr=float(_pr_cfg.get("lr", 1e-4)),
                    kl_max=float(_pr_cfg.get("kl_max", 0.02)),
                    min_steps=int(_pr_cfg.get("min_steps", 4)),
                    epochs=int(_pr_cfg.get("epochs", 1)),
                    strength_up=float(_pr_cfg.get("strength_up", 0.10)),
                    strength_down=float(_pr_cfg.get("strength_down", 0.02)),
                    decay_per_segment=float(
                        _pr_cfg.get("decay_per_segment", 0.01)),
                    stale_below=float(_pr_cfg.get("stale_below", 0.15)),
                    max_traj=int(_pr_cfg.get("max_traj", 256)),
                    device=str(self.device))
                logger.info(
                    "SKILL PRACTICE ACTIVE: bound skills are no longer frozen "
                    "(lr=%.1e, KL trust region %.3f, decay %.3f/segment)",
                    _ob.practice.lr, _ob.practice.kl_max,
                    _ob.practice.decay_per_segment)
            # SCRIPTED bootstrap options (e.g. chop_trunk): reserved BEFORE
            # skill binding so they hold a slot the policy can invoke to
            # perform a behaviour the emergent policy can't stumble into.
            for _spec in (opt_cfg.get("scripted_options") or []):
                _ob.reserve_scripted(_spec)
            self.option_executor = OptionExecutor(
                _ob, _n_envs, dict(
                    opt_cfg,
                    spike_threshold=self.config.get("goals", {}).get(
                        "spike_threshold", 0.9),
                    # DECISIONS -> FRAMES (see _frames_per): probation
                    # compares `_last_inv_t` against the timestep handed to
                    # act(), which is total_timesteps — so at num_envs 2 a
                    # gated slot was re-offered twice as often as the config
                    # said. Scaled here rather than by giving the executor a
                    # different clock: `_start_event` and `_close` are handed
                    # their t from different call sites, and mixing time
                    # bases between them would corrupt every recorded option
                    # duration.
                    gate_retry_after=self._frames_per(
                        int(opt_cfg.get("gate_retry_after", 3000)))),
                gamma=pol_cfg.get("gamma", 0.99))
            logger.info("Skills-as-options ACTIVE: head %d = %d primitives "
                        "+ %d slots", self.meta_action_dim, self.action_dim,
                        _k_slots)
            # ---- FOVEA CONTACT GATE (2026-08-07, change 5) --------------
            # break_*log* options only offer when the trunk is under the
            # gaze or there is fresh contact evidence — so the option's
            # 80-tick budget starts at the work, not during the walk
            # (measured: 61 break_spruce_log invocations, 0 successes).
            # Fail-open on every uncertainty: sense ungated -> no veto;
            # non-log skill -> no veto. A veto can therefore never lock a
            # skill out longer than the fovea itself stays empty.
            if bool(opt_cfg.get("fovea_precondition", False)):
                _min_lab = int(sg_cfg.get("min_labels", 5))

                def _log_contact_gate(binding, _self=self,
                                      _ml=_min_lab) -> bool:
                    try:
                        name = str(binding.get("name")
                                   or binding.get("skill_id") or "")
                        if "log" not in name:
                            return True
                        fcs = getattr(_self, "_last_fovea_counts",
                                      None) or {}
                        if fcs.get("tree_visible", 0) < _ml:
                            return True     # sense not yet earned -> no veto
                        if float(getattr(_self, "_reach_now", 0.0)) > 0.5:
                            return True     # measured contact
                        fps = getattr(_self, "_last_fovea_probs",
                                      None) or {}
                        return float(fps.get("tree_visible", 0.0)) >= 0.35
                    except Exception:
                        return True         # never let the gate kill an offer
                self.option_executor.contact_gate = _log_contact_gate
                logger.info("options: fovea contact gate ACTIVE for "
                            "*log* skills (threshold 0.35, fail-open)")
            if self.brain_emitter is not None:
                # live "pathways of thought": option start/end events light
                # nodes up in the brain viewer as they fire
                self.option_executor.on_start = (
                    self.brain_emitter.option_started)
                self.option_executor.on_end = (
                    lambda env, sid, t, outcome:
                    self.brain_emitter.option_ended(env, sid, t, outcome))

        # ---- Create Reward Mixer ----
        # gated=True hands anneal control to the developmental stage controller:
        # intrinsic weight is held high through EXPLORE and only begins its
        # curiosity->task drift once the controller fires EXPLORE->EXPLOIT.
        self.reward_mixer = RewardMixer(
            intrinsic_weight=pol_cfg.get("intrinsic_reward_weight", 0.7),
            extrinsic_weight=pol_cfg.get("extrinsic_reward_weight", 0.3),
            decay_rate=pol_cfg.get("reward_decay_rate", 5e-6),
            # FLOOR, previously un-passed and therefore stuck at its 0.1
            # default: curiosity annealed down to 0.1 and stayed there
            # forever. Setting it to 0 lets the drive genuinely reach zero on
            # a world the agent has mastered, which is the point of an anneal.
            min_intrinsic=pol_cfg.get("min_intrinsic", 0.1),
            gated=True,
            reentry_bump=self.config.get("developmental_stages", {}).get(
                "reentry_bump", 0.2
            ),
            adaptive_gating=pol_cfg.get("adaptive_gating", False),
            adaptive_rate=pol_cfg.get("adaptive_gating_rate", 5e-5),
            # ---- WIRED 2026-09-01, AND IT HAD TO BE ----------------------
            # `solve_threshold` decides what counts as "the task is being
            # solved" and therefore how fast curiosity is annealed away. It
            # has never been passed, so it sat at the 0.1 default — and on
            # this env ANY dirt break clears 0.1. Enabling adaptive gating
            # without this would have read "solved" on the ground-clearing
            # behaviour the whole reward economy exists to defund, and
            # annealed the exploration drive to its floor while the agent had
            # still never felled a log. The config sets it to a LOG-sized
            # return for that reason.
            solve_threshold=float(pol_cfg.get("solve_threshold", 0.1)),
            solve_ema_beta=float(pol_cfg.get("solve_ema_beta", 0.05)),
            # ---- RETURN NORMALIZATION, ACTUALLY WIRED (2026-08-02) --------
            # RewardMixer has carried this guard since 2026-07-27 and NOTHING
            # ever passed it, so `return_ratio_cap` sat at its 0.0 default and
            # the damping branch inside mix() was unreachable — the fix existed
            # in source and was inert in every run ever made.
            #
            # It exists because the WEIGHTS said 0.7/0.3 while the RETURNS were
            # 1922:1 curiosity (measured on the stalled run: weighted intrinsic
            # return 9179.4 vs weighted extrinsic 4.78). Annealing a weight
            # cannot matter when one stream fires every step and the other
            # fires four times in sixty segments — PPO optimises curiosity and
            # never sees the task. That is what "reward does not work" IS.
            #
            # 0.0 still means OFF, which remains the default for every config
            # that does not ask for it (tests/_stall_fixes_smoke.py pins that
            # default, and the guard only ever SHRINKS intrinsic, never
            # amplifies it, so it cannot detonate a task-free warmup).
            return_ratio_cap=pol_cfg.get("return_ratio_cap", 0.0),
            ret_ema_alpha=pol_cfg.get("return_ema_alpha", 1e-4),
            min_intrinsic_scale=pol_cfg.get("min_intrinsic_scale", 0.1),
        )
        if float(pol_cfg.get("return_ratio_cap", 0.0)) > 0.0:
            logger.info(
                "REWARD RETURN NORMALIZATION ACTIVE: intrinsic is damped "
                "whenever its return EMA exceeds %.1fx the extrinsic one "
                "(floor %.2f, alpha %.0e) — the task signal can no longer be "
                "buried by a dense per-step curiosity stream",
                self.reward_mixer.target_ratio,
                self.reward_mixer.min_intrinsic_scale,
                self.reward_mixer._ret_ema_alpha)

        # ---- Create Dream Actor-Critic (latent-space policy) ----
        dream_cfg = self.config.get("dream_training", {})
        self.dream_actor = DreamActorCritic(
            latent_dim=self.world_model.rssm.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=dream_cfg.get("hidden_dim", 400),
            continuous=not self.is_discrete,
            actor_lr=dream_cfg.get("actor_lr", 3e-5),
            critic_lr=dream_cfg.get("critic_lr", 3e-5),
            gamma=dream_cfg.get("gamma", 0.997),
            lambda_=dream_cfg.get("lambda_", 0.95),
            entropy_scale=dream_cfg.get("entropy_scale", 3e-4),
            target_ema=dream_cfg.get("target_ema", 0.98),
        ).to(self.device)
        self.dream_training_active = False
        self.dream_warmup = dream_cfg.get("warmup_episodes", 50)
        self.dream_enabled = dream_cfg.get("enabled", True)
        # ---- Imagination-AUGMENTED learning (Rung 4) ----
        # dream_control True (default) = original dream-CONTROL (the dream actor
        # drives real actions in the IMAGINE stage — the destabilizing path).
        # dream_control False = AUGMENT mode: the dream actor trains in imagination
        # and is DISTILLED into the real policy, but the REAL PPO policy stays in
        # control and learns normally (dream_training_active is never set, so the
        # PPO path is byte-for-byte unaffected). This is the proper sample-
        # efficiency lever: extra learning from accurate dreams, no control handover.
        self.dream_control = dream_cfg.get("control", True)
        self.dream_augment = bool(self.dream_enabled) and not self.dream_control
        self.dream_activate_after = dream_cfg.get("activate_after_episodes", self.dream_warmup)
        self.dream_distill_weight = float(dream_cfg.get("distill_weight", 1.0))
        self.dream_augment_active = False
        self._distill_opt = (
            torch.optim.Adam(self.policy.actor.parameters(),
                             lr=dream_cfg.get("distill_lr", 1e-4))
            if self.dream_augment else None
        )
        # ---- Distillation shaping (Rung 4 fix) ----
        # The original distill pulled the real policy toward the dream actor at
        # EVERY replay state, with a FIXED weight, via forward KL (mode-covering).
        # That violated "anneal naturally — never fix the ratio": late in training
        # PPO has already surpassed the dream actor, yet the fixed-weight pull kept
        # re-injecting the dream's softness and dragged final AUC/frac_solved down
        # (confirmed in both seeds even after the reward head was fixed). Three
        # coordinated levers fix it — distill only good dreams, only while they help:
        #   value_gate    : per-state mask — distill only where the dream critic
        #                   values the state ABOVE the real critic (the dream sees a
        #                   better path there); elsewhere PPO is already >= so no pull.
        #   natural_anneal: scale the distill weight by (1 - solve_rate_EMA) so the
        #                   dream's influence fades to zero as extrinsic reward
        #                   becomes reliably findable — kills the late-training drag.
        #   teacher_temp  : sharpen the teacher (<1) so the real policy adopts the
        #                   dream's PREFERRED action, not its uncertainty.
        self.dream_distill_value_gate = bool(dream_cfg.get("value_gate", True))
        self.dream_distill_anneal = bool(dream_cfg.get("natural_anneal", True))
        self.dream_distill_temp = float(dream_cfg.get("teacher_temp", 0.5))
        self._dream_solve_ema = 0.0
        self._dream_solve_beta = float(dream_cfg.get("solve_ema_beta", 0.05))
        self._dream_solve_threshold = float(dream_cfg.get("solve_threshold", 0.1))
        # ---- WM-trust gating (July 2026 audit, H2 — capstone collapse fix) ----
        # The value gate alone compares INCOMMENSURABLE critics (dream critic:
        # extrinsic-only WM-predicted symexp'd rewards; real critic: mixed
        # intrinsic+extrinsic GAE), so on a weak-WM env it opens widest exactly
        # where the teacher hallucinates; and the natural anneal is MAXIMAL when
        # the policy fails — a positive feedback loop into collapse. Fix: only
        # distill at states where the WM's own surprise (posterior-vs-prior KL)
        # is low, and scale the global pull by an EMA of that trust fraction, so
        # on envs the WM doesn't understand the pull fades no matter how badly
        # the real policy is doing. Signal-driven — no fixed ratio.
        self.dream_distill_trust_gate = bool(dream_cfg.get("trust_gate", True))
        self.dream_distill_trust_kl_max = float(dream_cfg.get("trust_kl_max", 6.0))
        self._wm_trust_ema = 0.0
        self._wm_trust_beta = float(dream_cfg.get("trust_ema_beta", 0.1))

        # ---- Developmental Stage Controller (closed-loop EXPLORE<->EXPLOIT) ----
        # Drives the curiosity->task handoff (and the optional, trust-gated
        # Stage-2 dream-control) from the agent's own learning signals — WM
        # prediction-error plateau as the spine, skill-acquisition rate as a
        # forward-only confirm. Replaces the old wall-clock anneal + episode-
        # count dream latch. IMAGINE (dream-control) stays dormant unless dream
        # training is enabled AND the world model earns trust.
        stage_cfg = self.config.get("developmental_stages", {})
        self.stage_controller = DevelopmentalStageController(
            window=stage_cfg.get("window", 30),
            patience=stage_cfg.get("patience", 5),
            min_episodes=stage_cfg.get("min_episodes", 50),
            flat_slope_threshold=stage_cfg.get("flat_slope_threshold", 1e-3),
            skill_rate_threshold=stage_cfg.get("skill_rate_threshold", 0.02),
            spike_factor=stage_cfg.get("spike_factor", 1.5),
            trust_level=stage_cfg.get("trust_level", 0.0),
            force_exploit_timestep=stage_cfg.get("force_exploit_timestep", None),
            force_imagine_timestep=stage_cfg.get("force_imagine_timestep", None),
            dream_enabled=self.dream_enabled,
        )

        # ---- Create LLM Module (optional — requires Ollama) ----
        llm_cfg = self.config.get("llm", {})
        shape_cfg = llm_cfg.get("reward_shaping", {}) or {}
        if llm_cfg.get("enabled", True):
            self.llm = LLMModule(
                model=llm_cfg.get("model", "llama3.1:8b"),
                env_description=llm_cfg.get(
                    "env_description",
                    f"Gymnasium {self.env_name} environment",
                ),
                obs_labels=self.obs_labels,
                perception_interval=llm_cfg.get("perception_interval", 50),
                goal_interval=llm_cfg.get("goal_interval", 25),
                reasoning_interval=llm_cfg.get("reasoning_interval", 100),
                shaping_interval=shape_cfg.get("interval", 25),
                async_mode=llm_cfg.get("async_mode", True),
                enabled=True,
            )
            if self.llm.is_available:
                logger.info(
                    "LLM module active — Ollama will assist with "
                    "perception and goal generation "
                    f"(async={self.llm.async_mode})"
                )
        else:
            # enabled=False forces every sub-channel offline even if a local
            # Ollama server is running, so no LLM calls fire and all should_*/
            # submit_* short-circuit (poll_* -> None).
            self.llm = LLMModule(enabled=False, async_mode=False)

        # Most recent goal produced by the (async) LLM goal generator. The loop
        # reuses it across episodes until a fresh one arrives, so a slow Ollama
        # call never blocks goal selection.
        self._pending_llm_goal: Optional[Dict[str, Any]] = None

        # ---- LLM goal-progress reward shaping (Path B) ----
        # The LLM scores how much each episode advanced the current goal; that
        # progress estimate Phi is consumed as a POTENTIAL-BASED shaping term:
        #   F = w * (Phi_now - Phi_prev)
        # applied once per episode onto the last stored transition. Difference-
        # form (no per-step constant) => telescopes to w*(Phi_final-Phi_initial),
        # so it CANNOT be farmed by stalling and provably preserves the task's
        # optimal policy. The weight w anneals toward zero on the same per-
        # timestep clock as curiosity — a fading guide, never a fixed ratio.
        self.llm_shaping_enabled = (
            shape_cfg.get("enabled", True) and llm_cfg.get("enabled", True)
        )
        self.llm_shape_w0 = shape_cfg.get("weight", 0.2)
        self.llm_shape_w = self.llm_shape_w0
        self.llm_shape_min = shape_cfg.get("min_weight", 0.0)
        self.llm_shape_decay = shape_cfg.get("decay", 5e-6)
        self.llm_shape_clip = shape_cfg.get("clip", 1.0)
        self.llm_potential = 0.0       # Phi: latest LLM goal-progress estimate
        self.llm_potential_prev = 0.0  # Phi at last shaping application
        self._recent_ep_rewards: deque = deque(maxlen=20)

        # ---- Curiosity-ranked vision magnet (INSTINCT channel, redesigned
        #      July 2026) ----
        # No longer a second VLM. The magnet consumes (a) the GROUNDING HEAD's
        # per-object presence probabilities (which the symbolizer already
        # predicts from the agent's own RSSM latent each step) and (b) the
        # primary stream's own learning-progress reward (intrinsic[0]), keeps a
        # contrastive per-category LP-EMA, and steers the agent toward whatever
        # it is currently most CURIOUS about — retargeting with hysteresis and
        # setting its OWN weight so it FADES as curiosity quenches and RE-ARMS
        # on a novel object. It never selects actions, and the env's raw reward
        # stream — what spike-based goal discovery and all grading read — is
        # never modified. Condition-based (curiosity magnitude), not a clock,
        # so it is correct in a never-ending lifelong stream.
        vis_cfg = llm_cfg.get("vision", {}) or {}
        self.vision_scaffold = None
        # Waking-step counter: retarget-hysteresis clock (min_dwell). Advances
        # ONLY on serial waking steps (where the magnet actually runs), NOT
        # during dream/parallel phases.
        self._vision_clock = 0
        if vis_cfg.get("enabled", False) and self.pixel_obs:
            from developmental_ai.llm.vision_scaffold import VisionScaffold
            self.vision_scaffold = VisionScaffold(
                weight=vis_cfg.get("weight", 0.15),
                min_weight=vis_cfg.get("min_weight", 0.0),
                cold_start_weight=vis_cfg.get("cold_start_weight", 0.10),
                promise_weight=vis_cfg.get("promise_weight", 0.0),
                cold_start_budget=vis_cfg.get("cold_start_budget", 4000),
                focus_distractors=vis_cfg.get("focus_distractors", None),
                target_categories=vis_cfg.get("target_categories", None),
                ema_beta=vis_cfg.get("ema_beta", 0.01),
                ema_leak=vis_cfg.get("ema_leak", 0.005),
                present_threshold=vis_cfg.get("present_threshold", 0.6),
                reliability_floor=vis_cfg.get("reliability_floor", 0.35),
                min_labels=vis_cfg.get("min_labels", 5),
                retarget_margin=vis_cfg.get("retarget_margin", 0.20),
                min_dwell=vis_cfg.get("min_dwell", 25),
                absent_tolerance=vis_cfg.get("absent_tolerance", 3),
                eps_abs=vis_cfg.get("eps_abs", 1.0e-3),
                scale_decay=vis_cfg.get("scale_decay", 0.999),
                instinct_bonus=vis_cfg.get("instinct_bonus", 0.05),
                approach_pull=vis_cfg.get("approach_pull", 0.0),
                align_bonus=vis_cfg.get("align_bonus", 0.0),
                aim_bonus=vis_cfg.get("aim_bonus", 0.0),
                seek_weight=vis_cfg.get("seek_weight", 0.0),
                seek_categories=vis_cfg.get("seek_categories", None),
                seek_forward_nudge=vis_cfg.get("seek_forward_nudge", 0.0),
                seek_nudge_budget=vis_cfg.get("seek_nudge_budget", 200),
                seek_nudge_regen=vis_cfg.get("seek_nudge_regen", 0),
                lp_stale_tau=vis_cfg.get("lp_stale_tau", 0.0),
                chop_actions=vis_cfg.get("chop_actions", [5, 6]),
                forward_actions=vis_cfg.get("forward_actions", [1, 2, 6]),
                turn_left_actions=vis_cfg.get("turn_left_actions", [3]),
                turn_right_actions=vis_cfg.get("turn_right_actions", [4]),
                pitch_actions=vis_cfg.get("pitch_actions", [7, 8]),
                contrast_vs_peers=vis_cfg.get("contrast_vs_peers", False),
                phi_from_evidence=vis_cfg.get("phi_from_evidence", False),
                seek_pitch_level=vis_cfg.get("seek_pitch_level", False),
                social_attention_boost=vis_cfg.get(
                    "social_attention_boost", 0.0),
            )
            logger.info(
                "Curiosity-ranked vision magnet active: w0=%.3f targets=%s "
                "(contrastive LP-EMA, condition-based fade)",
                vis_cfg.get("weight", 0.15),
                vis_cfg.get("target_categories") or "default-interactables")

        # ---- Create Skill Composer ----
        self.skill_composer = SkillComposer(
            min_mastery=sb_cfg.get("mastery_threshold", 0.8),
        )
        self.composite_executor = CompositeSkillExecutor(self.skill_bank)

        # ---- GENERAL INFRASTRUCTURE STACK (2026-08-08) -------------------
        # One facade for the domain-agnostic monitors (gates, heartbeats,
        # reward ledger, farm detector, signal health, drift, stuck
        # escalation, episodic memory, affordance map, empowerment, traces).
        # See docs/GENERAL_INFRASTRUCTURE.md and infra/stack.py. Constructed
        # defensively: a broken monitor disables itself, never the run.
        # ---- CONSEQUENCE FRONTIER (2026-08-13) ---------------------------
        # "Bored of what you have already caused; curious about what you have
        # seen but never affected" — plus possession as a second axis of
        # location. Feeds the magnet's SELECTION only (never its weight, which
        # would become a world-model reward label) and one count-bounded
        # income for crossing into a possession set never held before.
        self.consequence = None
        try:
            _cq_cfg = dict((self.config.get("infra", {}) or {}).get(
                "consequence", {}) or {})
            if _cq_cfg.get("enabled", False):
                from developmental_ai.infra.consequence import ConsequenceMap
                self.consequence = ConsequenceMap(_cq_cfg)
                _cqp = os.path.join(
                    str((self.config.get("infra", {}) or {}).get(
                        "log_dir", "runlogs")), "consequence_state.json")
                self._consequence_path = _cqp
                if self.consequence.load(_cqp):
                    logger.info(
                        "consequence: restored banked evidence from %s — "
                        "this state IS the bootstrap (a fresh map would "
                        "re-latch the very problem it exists to solve)",
                        _cqp)
                if self.vision_scaffold is not None:
                    self.vision_scaffold.set_deficit_source(
                        self.consequence.deficit)
                logger.info(
                    "CONSEQUENCE FRONTIER ACTIVE: magnet ranking weighted by "
                    "unconsummated consequence (promise_weight=%.2f), "
                    "possession-frontier income=%.3f (one-shot per set)",
                    float(vis_cfg.get("promise_weight", 0.0)),
                    float(_cq_cfg.get("possession_weight", 0.0)))
        except Exception as _e:
            logger.warning("consequence map unavailable (%s) — magnet falls "
                           "back to pure learning-progress ranking", _e)
            self.consequence = None

        # ---- ANTICIPATION (2026-09-02) -----------------------------------
        # Rise-and-reset intrinsic reward for a grounded predicate: builds
        # while the agent keeps SEEING something it hasn't yet produced the
        # effect of, pays out and resets the moment it does, and the whole
        # channel decays toward zero as the agent gets independently better
        # at producing that effect (infra/anticipation.py; same persistence
        # and "never in prim_extrinsic" discipline as ConsequenceMap above).
        self.anticipation = None
        try:
            _an_cfg = dict((self.config.get("infra", {}) or {}).get(
                "anticipation", {}) or {})
            if _an_cfg.get("enabled", False):
                from developmental_ai.infra.anticipation import (
                    AnticipationMap)
                self.anticipation = AnticipationMap(_an_cfg)
                _anp = os.path.join(
                    str((self.config.get("infra", {}) or {}).get(
                        "log_dir", "runlogs")), "anticipation_state.json")
                self._anticipation_path = _anp
                if self.anticipation.load(_anp):
                    logger.info(
                        "anticipation: restored banked evidence from %s — "
                        "a fresh map would erase every predicate<->effect "
                        "association already earned this run", _anp)
                logger.info(
                    "ANTICIPATION REWARD ACTIVE: weight=%.2f, wait_cap=%d, "
                    "sight_throttle=%d steps",
                    float(_an_cfg.get("weight", 0.5)),
                    int(_an_cfg.get("wait_cap", 8)),
                    int(_an_cfg.get("sight_throttle_steps", 50)))
        except Exception as _e:
            logger.warning("anticipation map unavailable (%s) — no "
                           "anticipation shaping this run", _e)
            self.anticipation = None

        # ---- live metrics tracker (structured, durable) --------------------
        # Off unless configured. Built BEFORE the infra stack so a failure
        # here cannot leave `infra` half-constructed, and wrapped because a
        # monitoring sink must never prevent the agent from starting.
        self._run_started_at = time.time()
        self._metrics_sink = None
        self._heartbeat_sink = None
        self._hb_every_s = 0.0
        self._hb_last = 0.0
        self._hb_last_steps = 0
        try:
            _mcfg = dict(self.config.get("metrics", {}) or {})
            if _mcfg.get("enabled"):
                from developmental_ai.infra.metrics_sink import MetricsSink
                self._metrics_sink = MetricsSink(_mcfg)
                # ---- HEARTBEAT: a fast, CHEAP second feed -------------------
                # The segment record is emitted once per segment — ~7 minutes
                # at 1024 steps x action_repeat 4 — which is far too coarse to
                # watch a run live. Shortening segment_len is NOT the fix: it
                # is PPO's batch and GAE horizon, so it would change training
                # to improve a dashboard.
                # This is a separate, deliberately thin record on a wall-clock
                # throttle. Separate FILE and TABLE so the two cadences never
                # mix in one series, and so a heartbeat can never be mistaken
                # for a segment observation.
                _hb = float(_mcfg.get("heartbeat_seconds", 15) or 0)
                if _hb > 0:
                    _hcfg = dict(_mcfg)
                    _hcfg["path"] = _mcfg.get(
                        "heartbeat_path", "runlogs/heartbeat.jsonl")
                    # fsync EVERY 15s would be gratuitous — a lost heartbeat
                    # costs one dashboard point, unlike a lost segment record.
                    _hcfg["fsync"] = bool(_mcfg.get("heartbeat_fsync", False))
                    self._heartbeat_sink = MetricsSink(_hcfg)
                    self._hb_every_s = _hb
                    logger.info("metrics heartbeat every %.0fs -> %s",
                                _hb, self._heartbeat_sink.path)
                logger.info(
                    "metrics tracker ON -> %s (fsync=%s, rotate at %.0f MB, "
                    "keep %d) | resuming at seq %d",
                    self._metrics_sink.path, self._metrics_sink.fsync,
                    self._metrics_sink.max_bytes / 1e6,
                    self._metrics_sink.keep, self._metrics_sink.seq)
        except Exception as exc:
            logger.warning("metrics tracker unavailable: %r", exc)

        self.infra = None
        try:
            from developmental_ai.infra.stack import InfraStack
            self.infra = InfraStack(self.config.get("infra", {}) or {},
                                    action_dim=int(self.action_dim))
            # the stack consumes its section via dict() copy (untrackable
            # C-level reads) — mark it read so the echo stays truthful
            if hasattr(self.config, "mark_all_read"):
                self.config.mark_all_read("infra")
            logger.info(InfraStack.provenance_line(
                getattr(self, "_config_path", None), dict(self.config)))
            # expected cadences for the subsystems whose SILENT death has
            # already cost this project weeks (proof-of-life, infra #25).
            # Registered ONLY for subsystems that exist in this run — a
            # vector-env run has no symbolizer/magnet, and a heartbeat for a
            # subsystem that was never built is a permanent false alarm
            # (measured on the CartPole integration probe).
            # expected cadences in TOTAL-timestep units: subsystems fire per
            # PRIMARY step but total_timesteps advances num_envs per step —
            # register in the clock domain the report reads or every cadence
            # is off by the fleet size (caught live: false "magnet OVERDUE")
            _fleet = max(1, int(getattr(self, "_num_envs", 1) or 1))
            if self.symbolizer is not None:
                _sgi = int(sg_cfg.get("max_interval", 600) or 600)
                self.infra.register_heartbeat("vlm_label",
                                              3 * _sgi * _fleet, 0)
                if getattr(self.symbolizer, "fovea_enabled", False):
                    self.infra.register_heartbeat(
                        "fovea_label",
                        6 * _fleet * int(getattr(self.symbolizer,
                                                 "fovea_interval", 60)
                                         or 60), 0)
            if self.vision_scaffold is not None:
                self.infra.register_heartbeat("magnet", 64 * _fleet, 0)
            self.infra.register_heartbeat("consolidation", 0, 0)
            self.infra.register_heartbeat("viewer", 0, 0)
            self.infra.register_heartbeat("option_offer", 0, 0)
        except Exception as _e:
            logger.warning("infra: stack unavailable (%s) — running without",
                           _e)
        # ---- COMPRESSION-PROGRESS CURIOSITY (infra #13, 2026-08-09) ------
        # The base intrinsic is ICM prediction ERROR — maximized by anything
        # visually dramatic, which funded five distinct reward farms. This
        # block MEASURES the world model improving on matched probes (same
        # probe, same objective, across model versions). Since 2026-10-05 it
        # PAYS NOTHING: model improvement does not establish credit for the
        # action being taken now, and the old per-step rate was an ambient
        # wage. progress_weight > 0 only switches the measurement on;
        # icm_base_scale still damps the raw-error term independently.
        self._progress = None
        self._progress_weight = float(_cur_cfg.get("progress_weight", 0.0))
        self._icm_base_scale = float(_cur_cfg.get("icm_base_scale", 1.0))
        # boring-view discount on the BASE term (see its use site: the sky
        # discount never covered the base, and clouds paid 96% of the drive
        # at the -90 clamp). 0 = off (every other config keeps old pricing).
        self._icm_boring_discount = float(_cur_cfg.get(
            "icm_boring_discount", 0.0))
        # metabolic effort pricing (see _infra_step); 0 = off
        self._effort_cost = float(_cur_cfg.get("effort_cost", 0.0))
        # memory-pull potential (point 3, see _memory_pull_phi); 0 = off
        self._memory_pull_weight = float(
            _cur_cfg.get("memory_pull_weight", 0.0))
        if self._progress_weight > 0.0:
            try:
                from developmental_ai.infra.progress_curiosity import (
                    ProbeSetProgress)
                self._progress_every = int(_cur_cfg.get(
                    "progress_eval_every", 512))
                self._progress = ProbeSetProgress(
                    eval_every=self._progress_every)
                logger.info(
                    "paired progress SHADOW ONLY (no reward): legacy weight=%.3f "
                    "eval_every=%d icm_base_scale=%.2f",
                    self._progress_weight, self._progress_every,
                    self._icm_base_scale)
            except Exception as _e:
                logger.warning("progress curiosity unavailable (%s)", _e)
        # stuck-escalation state (level-2 temporarily boosts exploration
        # weights; originals restored a segment later — never a latch)
        self._stuck_boost_until = -1
        self._stuck_boost_orig = None
        # when the CURRENT boost first applied — the cap on extension is
        # measured from here, so a remedy that keeps being requested cannot
        # quietly become permanent (see _apply_explore_boost)
        self._stuck_boost_started = -1
        self._seg_extrinsic_sum = 0.0
        self._seg_cells_prev = 0

        # ---- Training state ----
        self.total_timesteps = 0
        self.total_episodes = 0
        self.episode_rewards = deque(maxlen=100)
        self.training_metrics: Dict[str, deque] = {
            "episode_reward": deque(maxlen=100),
            "episode_length": deque(maxlen=100),
            "intrinsic_reward": deque(maxlen=100),
            "world_model_loss": deque(maxlen=100),
            "curiosity_loss": deque(maxlen=100),
            "policy_loss": deque(maxlen=100),
            "kl_divergence": deque(maxlen=100),
            "new_facts": deque(maxlen=100),
            "symbolic_decoder_loss": deque(maxlen=100),
            "symbolic_decoder_accuracy": deque(maxlen=100),
            "symbolic_decoder_facts": deque(maxlen=100),
            "glue_mastery_score": deque(maxlen=100),
            "glue_kg_density": deque(maxlen=100),
            "dream_actor_loss": deque(maxlen=100),
            "dream_distill_loss": deque(maxlen=100),
            "dream_distill_gate_frac": deque(maxlen=100),
            "dream_distill_eff_weight": deque(maxlen=100),
            "dream_critic_loss": deque(maxlen=100),
            "dream_returns_mean": deque(maxlen=100),
            "inverse_dynamics_loss": deque(maxlen=100),
            # PERSPECTIVE FROM MOTION. `flow_loss` is the photometric warp
            # error — the number that has to go DOWN for any of this to be
            # real, and the go/no-go for the object-slot wave. `flow_residual`
            # is what the warp could not explain, i.e. the curiosity channel's
            # input; watching it against `moved` is the falsifier (income at
            # moved~0 means another wage for standing still).
            "flow_loss": deque(maxlen=100),
            "flow_residual": deque(maxlen=100),
            # Auto-mask retained fraction. 1.0 = nothing dropped. Trending
            # toward 0 means the photometric loss is starving itself, which
            # is the one way auto-masking can go wrong quietly.
            "flow_mask": deque(maxlen=100),
            # P1. If this does not fall, imagined trajectories are fiction
            # past step one — and the policy trains on 15-step dreams.
            "horizon_loss": deque(maxlen=100),
            # R1. If this does not fall below what one undifferentiated field
            # achieves, the decomposition is not buying anything.
            "slot_loss": deque(maxlen=100),
            # The stage controller's SPINE (it drives every stage transition
            # via prediction_error) and, until 2026-09-04, never recorded
            # anywhere a human could read. Added alongside the WM-loss
            # telemetry fix so the signal the curriculum turns on is visible.
            "reconstruction_error": deque(maxlen=100),
        }

        # ---- SELECTIVE RESUME (2026-08-03) -------------------------------
        # MEASURED: load_checkpoint() exists but NOTHING in the launch path
        # calls it — not run_minecraft.py, not launch_skybot.sh. Its only
        # callers are the demo/eval/transfer scripts. So every restart threw
        # away the world model, policy, curiosity, dream actor and glue layer
        # and relearned from scratch, while the skill bank (goals, minted
        # skills) persisted. The system retained SKILLS better than LEARNING,
        # which is why entropy started ~90% and the reward curve
        # 0.02 -> 0.14 -> 0.38 -> 0.53 was re-run from zero after every deploy.
        _rc = [str(x) for x in (self.config.get("loop", {}).get(
            "resume_components") or [])]
        if _rc:
            self._resume_components(
                self.config.get("loop", {}).get(
                    "log_dir", "./logs") + "/checkpoints", _rc)

    def _resume_components(self, checkpoint_dir: str, names) -> None:
        """Restore ONLY the named components from a checkpoint.

        DELIBERATELY NOT `load_checkpoint`. That one restores the policy and
        dream actor too, and those are the ones that must NOT come back right
        now: they encode a policy converged under a reward economics that has
        since changed, and they resume at ~0 entropy with no way to explore
        out of a now-wrong optimum. Discarding the policy each launch is
        currently the only reason the run ever shows a healthy exploration
        phase. This method makes that a CHOICE rather than an accident.

        DIRECTIONAL, and the direction matters: keeping the world model while
        resetting the policy is safe, because under arch="wm" the policy reads
        world_model.encoder and simply relearns its action mapping on a better
        perception. The reverse — a resumed policy on a fresh encoder — would
        have its perception change underneath it, and is never valid here.

        Every load is shape-guarded. `action_dim` moved 10 -> 12 historically,
        and a stale checkpoint with the wrong head width must fail LOUD and
        fall back to fresh weights, never load partially.
        """
        import os as _os
        if not _os.path.isdir(checkpoint_dir):
            logger.info("resume: no checkpoint dir at %s — starting fresh",
                        checkpoint_dir)
            return
        # SHAPES MUST MATCH THE WRITER, not the attribute name. _save_checkpoint
        # writes symbolic_decoder.pt from `self.symbolic_decoder.DECODER` (the
        # inner module, which is None under the pixel gate) and glue_layer.pt
        # as a TWO-KEY DICT {"gnn", "gate"} — loading either into the outer
        # object would raise, or worse, silently no-op.
        def _load_wm(m):
            # SPLIT THE RESTORE BY WHAT THE WEIGHTS MEAN (2026-08-13).
            #
            # `policy`/`dream_actor`/`curiosity` are refused by name because
            # they encode an optimum converged under reward economics that
            # has since changed. The world model's REWARD PREDICTOR is the
            # same category and was being restored anyway: it is trained on
            # `prim_extrinsic` labels (rewards[0] + magnet shaping), i.e. on
            # exactly the economy in force when the checkpoint was written.
            # This one was written when breaking dirt paid handsomely
            # (view-novelty 34%, coverage 65%), so a restored reward head
            # still BELIEVES dirt pays — and dream_actor, prospection and
            # empowerment all plan against that belief. Old dirt-grinding
            # therefore keeps influencing behaviour through imagination even
            # though the policy itself is fresh every run.
            #
            # The rest of the model is economy-INDEPENDENT and worth keeping:
            # the encoder and RSSM model how the world WORKS (pixels,
            # dynamics), not what is worth doing. Splitting the restore keeps
            # hard-won perception and dynamics while letting values be
            # relearned under the current economy.
            _drop = {k: v for k, v in m.items()
                     if not str(k).startswith(_WM_ECONOMY_PREFIXES)}
            _n_dropped = len(m) - len(_drop)
            if _n_dropped and bool(self.config.get("loop", {}).get(
                    "resume_reward_head", False)):
                _drop = m                      # explicit opt-in to the old
                _n_dropped = 0                 # behaviour, off by default
            # PRE-FLIGHT, BEFORE ANY MUTATION (2026-08-23 review). torch copies
            # matching tensors as it goes and only raises at the END, so a
            # check made after the load leaves a partial hybrid behind. The
            # payload's key set is knowable up front, so a checkpoint that
            # cannot fill this model is refused while the model is still
            # pristine — and, because the refusal keeps `world_model` out of
            # `_ok`, the dependency gate below correctly withholds perception
            # and familiarity instead of pairing them with a half-fresh
            # encoder.
            _dropped_keys = ({str(k) for k in m
                              if str(k).startswith(_WM_ECONOMY_PREFIXES)}
                             if _n_dropped else set())
            _absent = _wm_restore_absent(self.world_model.state_dict().keys(),
                                         _drop.keys(), _dropped_keys)
            if _absent:
                raise _ResumePreflightRefusal(
                    f"world_model checkpoint cannot fill this model: "
                    f"{len(_absent)} tensor(s) absent, e.g. {_absent[:4]} — "
                    f"loading it would leave those submodules silently FRESH "
                    f"while the resume reported success, which then admits "
                    f"perception/familiarity onto a half-fresh encoder. "
                    f"Usually arch drift or a truncated write. Refusing "
                    f"before any tensor is copied.")
            # strict=False: the reward head keeps its FRESH init rather than
            # being overwritten; every other tensor is restored exactly.
            _missing, _unexpected = self.world_model.load_state_dict(
                _drop, strict=False)
            # Invariant, not a duplicate of the pre-flight: by construction
            # `_missing` can now only contain keys we dropped on purpose. If
            # it ever holds anything else, the two views of the model
            # disagree and the restore is not trustworthy.
            _unexplained = [str(k) for k in _missing
                            if str(k) not in _dropped_keys]
            if _unexplained:
                raise RuntimeError(
                    f"world_model restore left {len(_unexplained)} tensor(s) "
                    f"un-restored despite passing pre-flight, e.g. "
                    f"{_unexplained[:4]} — state_dict() and load_state_dict() "
                    f"disagree about this model's keys")
            if _n_dropped:
                logger.info(
                    "resume: world_model restored WITHOUT its reward head "
                    "(%d tensors dropped, %d params) — that head was fit to "
                    "a reward economy that has since changed, and "
                    "imagination plans against it. Encoder + RSSM (how the "
                    "world works) are restored in full.",
                    _n_dropped,
                    sum(int(v.numel()) for k, v in m.items()
                        if str(k).startswith(("reward_predictor",
                                              "reward_bins"))
                        and hasattr(v, "numel")))
            if _unexpected:
                raise RuntimeError(
                    f"world_model checkpoint has unexpected keys "
                    f"{list(_unexpected)[:4]} — shape/arch drift")

        def _load_sd(m):
            _dec = getattr(self.symbolic_decoder, "decoder", None)
            if _dec is None:
                raise RuntimeError(
                    "symbolic_decoder.decoder is None (pixel gate / null "
                    "decoder) — nothing to restore into")
            _dec.load_state_dict(m)

        def _load_glue(m):
            _ki = self.glue.knowledge_integrator
            if not isinstance(m, dict) or "gnn" not in m:
                raise RuntimeError(
                    f"glue_layer.pt is not the expected dict with a 'gnn' "
                    f"key (got {type(m).__name__})")
            _ki.gnn.load_state_dict(m["gnn"])
            # The gate is absent on the pixel path (build_gate=False, see
            # KnowledgeIntegrator). A checkpoint written before 2026-09-01
            # still carries one; skipping it is correct rather than lossy,
            # because that module was never trained — it is frozen at its
            # random init, so a "restored" gate and a fresh one are the same
            # distribution. Refuse only the genuine mismatch: a live gate
            # with no saved weights to put in it.
            if _ki.gate is not None:
                if "gate" not in m:
                    raise RuntimeError(
                        "glue_layer.pt has no 'gate' but this run built one "
                        "(non-pixel obs) — refusing to leave it fresh while "
                        "reporting a successful restore")
                _ki.gate.load_state_dict(m["gate"])

        def _load_perception(m):
            if self.symbolizer is None:
                raise RuntimeError("no symbolizer in this run")
            summary = self.symbolizer.load_state(m)
            _n = sum(1 for v in self.symbolizer.label_counts.values() if v > 0)
            # A degraded restore must not read as a clean one (review
            # finding): load_state swallows its own errors, so without this
            # the operator sees "restored" at INFO for a partial load.
            if "partial" in summary or "FRESH" in summary or "DROPPED" in summary:
                logger.warning("resume: perception restored DEGRADED (%s) — "
                               "%d/%d predicates carry prior evidence",
                               summary, _n, len(self.symbolizer.label_counts))
            else:
                logger.info("resume: perception restored (%s) — %d/%d "
                            "predicates carry prior evidence", summary, _n,
                            len(self.symbolizer.label_counts))

        def _load_familiarity(m):
            # merge, never replace: a fresh dict is valid too
            for attr, key in (("_nov_counts", "nov_counts"),
                              ("_gaze_counts", "gaze_counts"),
                              ("_symbol_counts", "symbol_counts"),
                              ("_symbol_sight_counts",
                               "symbol_sight_counts")):
                cur = getattr(self, attr, None)
                if isinstance(cur, dict):
                    cur.update(m.get(key) or {})
                else:
                    setattr(self, attr, dict(m.get(key) or {}))
            ks = m.get("known_symbols")
            if ks:
                cur = getattr(self, "_known_symbols", None)
                if isinstance(cur, set):
                    cur.update(ks)
                else:
                    self._known_symbols = set(ks)

        def _load_magnet(m):
            if self.vision_scaffold is None:
                raise RuntimeError("no vision scaffold in this run")
            summary = self.vision_scaffold.load_state(m)
            if "partial" in summary or "FRESH" in summary:
                logger.warning("resume: magnet curiosity restored DEGRADED "
                               "(%s)", summary)
            else:
                logger.info("resume: magnet curiosity restored — %s", summary)

        # Replay buffer: EXPERIENCE, not policy — restoring it cannot resume a
        # converged optimum, only spare the world model re-collecting hours of
        # the world. It is also the component whose absence was invisible: the
        # buffer had no persistence at all, so every crash-relaunch silently
        # restarted world-model training from empty.
        #
        # NOT keyed to the encoder's latent space (it stores raw observations
        # and actions), so unlike perception/familiarity it does NOT belong in
        # _RESUME_ENCODER_KEYED — a fresh encoder can be trained on old frames
        # perfectly well; that is what a replay buffer IS.
        def _load_replay(_ignored):
            _on, _dir, _ = self._replay_persist_cfg()
            if not _on:
                raise RuntimeError(
                    "world_model.buffer_persist is false — remove "
                    "'replay_buffer' from loop.resume_components or enable it")
            if self.replay_buffer is None:
                raise RuntimeError("no replay buffer in this run")
            _n = self.replay_buffer.load(_dir)
            logger.info(
                "resume: replay buffer restored — %d transitions (~%.1f h of "
                "experience at 3.2 steps/s per stream); the world model does "
                "NOT restart from empty", _n, _n / max(1.0, 3.2 * 3600.0))

        _loaders = {
            "world_model": ("world_model.pt", _load_wm),
            "symbolic_decoder": ("symbolic_decoder.pt", _load_sd),
            "glue_layer": ("glue_layer.pt", _load_glue),
            # perception + familiarity (2026-08-10): experience, not policy —
            # restoring them cannot resume a converged optimum, only spare
            # the agent re-learning what the world looks like
            "perception": ("symbolizer.pt", _load_perception),
            "familiarity": ("familiarity.pt", _load_familiarity),
            "magnet": ("magnet.pt", _load_magnet),
        }
        _ok, _skip = [], []
        # DEPENDENCY ORDER, not config order: `perception`/`familiarity` are
        # gated below on world_model having actually restored, and that test
        # reads `_ok` as it fills. Trusting the order someone happened to
        # write in YAML would make the gate silently wrong (it would skip a
        # perfectly good perception restore merely for being listed first).
        names = sorted(names, key=lambda n: (n in ("perception",
                                                   "familiarity",
                                                   "magnet")))
        for _n in names:
            if _n == "knowledge_graph":
                _kp = _os.path.join(checkpoint_dir, "knowledge_graph.json")
                if not _os.path.exists(_kp):
                    _skip.append(f"{_n}(absent)"); continue
                try:
                    self.knowledge_graph.load(_kp)
                    _ok.append(_n)
                except Exception as _e:
                    _skip.append(f"{_n}({type(_e).__name__})")
                    logger.warning("resume: knowledge_graph FAILED (%s) — "
                                   "continuing with an empty graph", _e)
                continue
            if _n == "replay_buffer":
                # a DIRECTORY of .npy arrays, not a torch file — special-cased
                # like knowledge_graph rather than forced through the
                # torch.load loader table
                _on, _bd, _ = self._replay_persist_cfg()
                if not _on or not _os.path.isdir(_bd):
                    _skip.append(f"{_n}(absent)"); continue
                try:
                    _load_replay(None)
                    _ok.append(_n)
                except Exception as _e:
                    _skip.append(f"{_n}({type(_e).__name__})")
                    logger.warning(
                        "resume: replay_buffer FAILED (%s: %s) — continuing "
                        "with an EMPTY buffer, so the world model relearns "
                        "the world from scratch this run", type(_e).__name__,
                        _e)
                continue
            if _n in ("policy", "dream_actor", "curiosity"):
                # refused by name, so it can never happen by a config typo
                logger.warning(
                    "resume: REFUSING '%s' — restoring a converged policy "
                    "resumes its entropy collapse under a reward function "
                    "that has since changed. Remove it from "
                    "loop.resume_components.", _n)
                _skip.append(f"{_n}(refused)")
                continue
            _spec = _loaders.get(_n)
            if _spec is None:
                _skip.append(f"{_n}(unknown)"); continue
            # DEPENDENCY: perception and familiarity are both expressed in the
            # WORLD MODEL's latent space — the symbol head is built with
            # latent_dim from world_model.rssm, and the familiarity counts are
            # keyed by an LSH of the encoder's output. Restoring either onto a
            # DIFFERENT (fresh) encoder pairs trained weights with a
            # representation they were never fit to: the head reads noise, and
            # the novelty counters mark unseen views as already-visited, which
            # zeroes exploration income exactly when it is needed most.
            # (2026-08-11 review finding; world_model.pt is also written
            # non-atomically, so a truncated file makes this reachable.)
            #
            # MEMBERSHIP IS A NAMED CONTRACT (2026-08-23 review), not an inline
            # tuple: `magnet` was here by association and does not belong. Its
            # state is category-NAME-keyed learning progress, valid against any
            # encoder, so gating it made a failed world-model restore also
            # throw away the curiosity memory — while logging the untrue reason
            # "needs world_model". See _RESUME_ENCODER_KEYED.
            if (_n in _RESUME_ENCODER_KEYED
                    and "world_model" not in _ok):
                _skip.append(f"{_n}(needs world_model)")
                logger.warning(
                    "resume: SKIPPING '%s' because world_model was not "
                    "restored — it is keyed to that encoder's latent space "
                    "and would be meaningless (worse: silently trusted) "
                    "against a fresh one.", _n)
                continue
            _fn, _apply = _spec
            _fp = _os.path.join(checkpoint_dir, _fn)
            if not _os.path.exists(_fp):
                _skip.append(f"{_n}(absent)"); continue
            try:
                _apply(torch.load(_fp, map_location=self.device))
                _ok.append(_n)
            except Exception as _e:
                # LOUD — but NOT clean. The old comment here claimed
                # load_state_dict "raises before mutating, so the module is
                # intact". That is FALSE and was verified false (2026-08-11):
                # Module._load_from_state_dict collects shape mismatches into
                # error_msgs and CONTINUES, raising only at the end — so every
                # parameter before the mismatched one has already been copied.
                # A component reported here as skipped may therefore be a
                # HYBRID (e.g. restored encoder + fresh action head), which is
                # exactly the action_dim 10->12 case this guard was written
                # for. Say so, rather than implying a clean fallback.
                if isinstance(_e, _ResumePreflightRefusal):
                    # refused before copying: this component IS cleanly fresh,
                    # and saying "may be a hybrid" here would be a false alarm
                    _skip.append(f"{_n}(refused:CLEAN)")
                    logger.warning(
                        "resume: '%s' REFUSED before loading — %s The module "
                        "is cleanly FRESH (nothing was copied), and it is "
                        "held out of the restored set, so anything gated on "
                        "it is correctly withheld too.", _n, _e)
                    continue
                _skip.append(f"{_n}({type(_e).__name__}:PARTIAL?)")
                logger.warning(
                    "resume: '%s' FAILED mid-load (%s: %s) — usually a shape "
                    "change (action_dim, enc_dim). WARNING: torch copies "
                    "matching tensors BEFORE raising, so this module may now "
                    "be a partial hybrid, not fresh. If this is the world "
                    "model, prefer restarting with that component removed "
                    "from loop.resume_components.",
                    _n, type(_e).__name__, _e)
        logger.info("RESUMED %s from %s%s", _ok or "nothing", checkpoint_dir,
                    f" | skipped: {', '.join(_skip)}" if _skip else "")
        if "world_model" in _ok:
            logger.info(
                "  perception carried over: the policy reads "
                "world_model.encoder, so a fresh policy starts on a TRAINED "
                "visual system and only relearns its action mapping")

    def _drop_curiosity_batch(self) -> None:
        """Discard any part-filled ICM training batch.

        Called on shutdown and on a stream death. The transitions in it are
        individually valid, so this is hygiene rather than a correctness fix:
        it keeps a batch from outliving the run (or the world) that produced
        it, which is the rule the perception/familiarity work settled on —
        evidence must not outlive its head.
        """
        try:
            self._cur_train_buf.clear()
        except Exception:
            pass

    def close(self) -> None:
        """Clean up resources (Neo4j connections, LLM threads, etc.)."""
        if getattr(self, "_shadow", None) is not None:
            self._shadow.close()      # never raises; logs its summary
        if getattr(self, "_explore_tracker", None) is not None:
            self._explore_tracker.flush()   # position trace tail; never raises
        if hasattr(self.knowledge_graph, "close"):
            self.knowledge_graph.close()
        if hasattr(self, "llm") and self.llm is not None:
            self.llm.close()
        if getattr(self, "vision_scaffold", None) is not None:
            self.vision_scaffold.close()
        if getattr(self, "symbolizer", None) is not None:
            self.symbolizer.close()
        self._drop_curiosity_batch()
        # Stop the async WM trainer FIRST (before the sampler it consumes and
        # before env/CUDA teardown): a daemon thread killed at interpreter
        # exit mid-optimizer-step segfaults, and closing the sampler first
        # would strand a mid-block trainer in sampler.get() (review fix).
        if getattr(self, "_wm_trainer_thread", None) is not None:
            self._wm_trainer_stop.set()
            self._wm_trainer_thread.join(timeout=90)
            if self._wm_trainer_thread.is_alive():
                logger.warning("async WM trainer did not stop within 90s")
        if hasattr(self, "_bg_sampler") and self._bg_sampler is not None:
            self._bg_sampler.close()
        # Close any extra parallel envs (index 0 is self.env, closed below).
        for e in getattr(self, "_parallel_envs", [])[1:]:
            try:
                e.close()
            except Exception:
                pass
        self.env.close()

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def run(
        self,
        total_timesteps: Optional[int] = None,
        log_interval: int = 10,
        save_interval: Optional[int] = None,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """
        Run the developmental AI loop.

        This is the core training loop that implements:
          Environment → Curiosity → Goal → Explore → Observe →
          Prediction Error → Update World Model → Add Skill → Repeat

        Args:
            total_timesteps: Max steps to run (None = run forever)
            log_interval: Episodes between logging
            save_interval: Timesteps between checkpoints
            verbose: 0=silent, 1=info, 2=debug

        Returns:
            Training summary with metrics
        """
        if total_timesteps is None:
            _cfg_budget = self.config.get("loop", {}).get(
                "total_timesteps", 100000)
            # forever-mode -> None (run until stop-file/SIGTERM); else the
            # configured budget, exactly as before.
            total_timesteps = self._ll_ctrl.budget(_cfg_budget)

        # `loop.checkpoint_interval` had NO reader: run_minecraft.py calls
        # run() without save_interval, so the config value was inert and the
        # hardcoded 10000 default applied silently. An explicit argument still
        # wins (harnesses that pass one keep their behaviour).
        if save_interval is None:
            save_interval = int(self.config.get("loop", {}).get(
                "checkpoint_interval", 10000))
        logger.info("checkpointing every %d timesteps", save_interval)

        logger.info(f"Starting developmental loop for {total_timesteps} timesteps")
        logger.info(f"Components: RSSM world model, ICM curiosity, "
                     f"knowledge graph, skill bank, PPO policy")

        start_time = time.time()

        # Warm-start from prior skills happens exactly ONCE, even if run() is
        # called repeatedly in blocks (e.g. the Rung-5 multi-task harness swaps
        # self.env and re-enters run() each block). Repeatedly overwriting the
        # live policy would clobber in-progress learning, so the latch persists
        # across run() calls — only the first call seeds.
        if not hasattr(self, "_warm_start_done"):
            self._warm_start_done = False

        # ---- Env-swap detection (July 2026 audit, H2) ----
        # Block-driven harnesses (capstone, rung 5) swap self.env and re-enter
        # run(). On a swap: (a) restart the dream-augment warmup so distillation
        # doesn't fire immediately from a teacher trained on the PREVIOUS env's
        # dynamics, and (b) flush the PPO rollout buffer so the first (and
        # formative) update on the new env isn't contaminated with up to a full
        # window of old-env transitions whose rewards and optimal actions differ.
        if not hasattr(self, "_env_token"):
            self._env_token = id(self.env)
            self._episodes_this_env = self.total_episodes
        elif self._env_token != id(self.env):
            self._env_token = id(self.env)
            self._episodes_this_env = 0
            n_stale = len(self.policy.rollout_obs)
            if n_stale:
                for lst in (self.policy.rollout_obs,
                            self.policy.rollout_actions,
                            self.policy.rollout_rewards,
                            self.policy.rollout_dones,
                            self.policy.rollout_log_probs,
                            self.policy.rollout_values,
                            self.policy.rollout_knowledge,
                            self.policy.rollout_taus,
                            self.policy.rollout_masks):
                    lst.clear()
            logger.info(
                "Env swap detected: dream-augment warmup restarted, "
                "%d stale rollout transitions flushed.", n_stale)

        # STOP-FILE APPLIES TO BUDGETED RUNS TOO (2026-07-25). It used to be
        # consulted only on the `total_timesteps is None` (forever) branch,
        # while `should_stop()` itself returned False unless `forever` was set
        # — so for a budgeted run (`forever: false`, the mode actually used on
        # the training host) touching `lifelong.stop_file` did NOTHING. The documented
        # graceful-stop escape hatch silently no-op'd, leaving SIGKILL mid-write
        # as the only way to end a run. Both halves are fixed: `should_stop()`
        # now honours the stop-file whenever lifelong is enabled, and it is
        # checked on BOTH branches here.
        while (not self._ll_ctrl.should_stop()
               and ((total_timesteps is None)
                    or (self.total_timesteps < total_timesteps))):
            # ---- Generate goal for this episode ----
            # LLM goal generation is ASYNC: we poll for any goal that finished
            # on the background thread, and periodically submit a new request.
            # The (possibly slow) Ollama call never blocks this loop.
            ready_goal = self.llm.poll_goal()
            if ready_goal:
                self._pending_llm_goal = ready_goal
                logger.info(
                    f"LLM goal: {ready_goal.get('description', '?')} "
                    f"[{ready_goal.get('strategy', '?')}]"
                )

            if self.llm.should_generate_goal(self.total_episodes):
                kg_stats = self.knowledge_graph.get_stats()
                sk_stats = self.skill_bank.get_stats()
                self.llm.submit_goal(
                    knowledge_summary=(
                        f"{kg_stats['num_facts']} facts, "
                        f"{kg_stats['num_entities']} entities, "
                        f"{kg_stats['num_action_rules']} rules"
                    ),
                    skill_summary=", ".join(sk_stats["skill_names"]) or "None",
                    recent_performance=(
                        f"Avg reward: "
                        f"{np.mean(list(self.training_metrics['episode_reward'])) if self.training_metrics['episode_reward'] else 0:.2f}"
                    ),
                )

            llm_goal = self._pending_llm_goal
            goal = self.glue.generate_goal(
                self.knowledge_graph,
                self.symbolic_decoder,
                skill_bank=self.skill_bank,
                llm_subgoals=llm_goal.get("sub_goals") if llm_goal else None,
                stage=self.stage_controller.stage,
            )

            # ---- Warm-start from prior skills (ONCE, at run start) ----
            # Cumulative learning across runs: a relevant skill saved by a
            # *previous* run seeds the PPO policy so practice compounds. We do
            # this exactly once per run. Overwriting the live policy mid-run
            # (the old per-episode behavior) repeatedly clobbered in-progress
            # learning — the Q1 instability — so it is now latched.
            if not self._warm_start_done:
                self._warm_start_done = True
                relevant_skills = self.glue.select_skills(
                    goal, self.skill_bank, self.knowledge_graph,
                    max_skills=self.skill_bank.max_skills_loaded,
                )
                # relevant_skills is sorted by relevance (desc). Take the best
                # skill that is BOTH relevant enough AND shape-compatible with
                # this env's policy network. A high-similarity match to a
                # different-env skill (different obs/action dims) must NOT be
                # loaded — load_state_dict would raise a shape mismatch and
                # crash the run — so we skip it and try the next candidate.
                #
                # The cut is `skill_bank.similarity_threshold` (was a
                # hardcoded 0.7). Scale is the normalized [0,1] relevance from
                # SkillSelector, further scaled by competence — so it is the
                # bar for "relevant AND competent enough to seed the policy",
                # and how many candidates get this far is max_skills_loaded.
                _sim_floor = float(self.skill_bank.similarity_threshold)
                for best_skill, score in relevant_skills:
                    if score <= _sim_floor:
                        break  # sorted desc → nothing further qualifies

                    dims_known = (
                        best_skill.obs_dim is not None
                        and best_skill.action_dim is not None
                    )
                    if dims_known and (
                        best_skill.obs_dim != self.obs_dim
                        or best_skill.action_dim != self.action_dim
                    ):
                        logger.info(
                            f"Skipping warm-start from '{best_skill.name}' "
                            f"(relevance={score:.2f}) — shape mismatch "
                            f"(skill obs/act="
                            f"{best_skill.obs_dim}/{best_skill.action_dim}, "
                            f"env obs/act={self.obs_dim}/{self.action_dim})"
                        )
                        continue

                    policy_state = self.skill_bank.load_skill_policy(
                        best_skill.skill_id
                    )
                    if not policy_state:
                        continue

                    # SNAPSHOT-AND-RESTORE (2026-07-26). torch >=2.6 copies
                    # every MATCHING parameter into the module BEFORE raising
                    # on the first mismatched one — so the except-branch's
                    # "keep the fresh policy" was a lie: the run continued on
                    # a FRANKEN-policy (foreign trunk + fresh head + fresh
                    # critic) while logging "training from scratch". The only
                    # honest failure mode is all-or-nothing.
                    import copy as _copy
                    _fresh_sd = _copy.deepcopy(self.policy.get_state_dict())
                    try:
                        self.policy.load_state_dict(policy_state)
                    except (RuntimeError, ValueError) as e:
                        self.policy.load_state_dict(_fresh_sd)   # undo partial
                        logger.info(
                            f"Warm-start from '{best_skill.name}' failed "
                            f"(incompatible policy: {e}); restored the fresh "
                            f"policy — training from scratch, genuinely"
                        )
                        continue

                    logger.info(
                        f"Warm-started policy from skill "
                        f"'{best_skill.name}' (relevance={score:.2f}) "
                        f"— seeding this run from prior experience"
                    )
                    break

            # ---- Run one episode ----
            # Parallel collection only activates once dream training is active.
            # Before that, the on-policy PPO learner is collecting real rollouts
            # (whose GAE assumes a single temporal stream), so we keep the
            # proven single-env path. After warmup the real policy is dormant
            # and the rollout's only job is to feed the world model + buffer —
            # which parallelizes safely across N envs.
            # Parallel collection now runs in BOTH phases (July 2026 curiosity
            # program): waking uses the REAL policy across N envs (15 "scout"
            # streams feed the world model + curiosity + discovery; PPO learns
            # on-policy from the primary stream only, so GAE stays single-
            # stream-correct), dream uses the latent dream actor. This is what
            # makes the fleet accelerate the CURIOSITY phase, not just dreams.
            if self._lifelong:
                episode_metrics = self._collect_segment()
            elif self._use_parallel_envs:
                episode_metrics = self._run_episode_parallel(
                    use_dream_actor=self.dream_training_active)
            else:
                episode_metrics = self._run_episode()

            # Track metrics
            for key, value in episode_metrics.items():
                if key in self.training_metrics:
                    self.training_metrics[key].append(value)

            self.total_episodes += 1
            self._episodes_this_env += 1

            # ---- Achievement-grounded skill minting (rich-env program,
            # step 2) ---- One skill PER ACHIEVEMENT at its first unlock,
            # instead of one-per-env at an unreachable 0.8 mastery bar (both
            # 1M rich-env runs minted ZERO skills under the old rule). The
            # context embedding is the CANONICAL goal vector for that slot,
            # so future goal-conditioned retrieval matches by construction.
            if (
                self.broadcaster is not None
                and hasattr(self.broadcaster, "pop_skill_mint_events")
                and self.config.get("goals", {}).get("mint_skills", True)
            ):
                for slot, slot_name in self.broadcaster.pop_skill_mint_events():
                    emb = self._unlock_context.get(slot)
                    if emb is None:  # unlock predates latent pairing
                        emb = np.zeros(self.broadcaster.DIM, dtype=np.float32)
                        emb[slot] = 1.0
                        emb[2 * self.broadcaster.max_slots] = 1.0
                    comp = float(
                        self.broadcaster.competence.predict_all()[slot])
                    # GROUNDED naming — from the block that actually broke
                    # at unlock, never the VLM (which hallucinated
                    # "fish_blue_water" etc. from frames). Ground truth only.
                    from developmental_ai.skill_bank import skill_notes
                    _frame = self._unlock_frame.get(slot)
                    _pre = self._preconditions_from_latent(slot)
                    _name, _desc = self._grounded_skill_name(slot, _pre)
                    # ---- FOSSIL GUARD (2026-07-27 stall assessment) -------
                    # Refuse to mint a slot whose NAME encodes no grounded
                    # effect (`discovered_N`, `break_air`). Those are evidence
                    # that SOMETHING happened, not evidence of a capability;
                    # minting them is what filled this bank with 48 slots and
                    # 0 usable skills. scripts/retire_fossil_skills.py cleaned
                    # them once, but nothing stopped them coming back — so the
                    # guard belongs HERE, at the moment of creation, not in a
                    # tool that runs afterwards.
                    from developmental_ai.core.achievement_goals import (
                        is_fossil_slot_name)
                    if is_fossil_slot_name(slot_name) or \
                            is_fossil_slot_name(_name):
                        logger.info(
                            "mint REFUSED for slot %d: fossil name "
                            "(slot_name=%r, grounded=%r) — an unnamed effect "
                            "is not a capability", slot, slot_name, _name)
                        continue
                    # ---- COMPOSITION METADATA (arch v3) -----------------
                    # slot_map: what each option-slot row of the SAVED head
                    # meant at this instant, by IDENTITY. Slots rebind, so a
                    # frozen skill can only ever re-resolve children through
                    # this map — a row whose skill has left the working set
                    # goes dark instead of silently invoking a stranger.
                    # parent_skills: skills observed RUNNING when this
                    # slot's effect unlocked (earned provenance, additive).
                    _slot_map, _parents = None, None
                    if self.option_executor is not None:
                        _slot_map = {
                            str(_j): _sb["skill_id"]
                            for _j, _sb in enumerate(
                                self.option_executor.bank.slots)
                            if _sb is not None
                            and not _sb.get("scripted")}
                        _parents = sorted({
                            _co["skill_id"]
                            for _co in self.option_executor.co_log
                            if int(_co.get("unlocked_slot", -1)) == slot})
                    _skill = self.skill_bank.save_skill(
                        skill_id=f"ach_{slot:02d}_{slot_name}"[:64],
                        name=_name,
                        policy_state_dict=self.policy.get_state_dict(),
                        description=_desc,
                        success_rate=comp,
                        total_episodes=self.total_episodes,
                        avg_reward=float(
                            episode_metrics.get("episode_reward", 0.0)),
                        context_embedding=emb,
                        preconditions=_pre or None,
                        obs_dim=self.obs_dim,
                        action_dim=self.action_dim,
                        dedup=True,
                        parent_skills=_parents or None,
                        **self._arch_save_kwargs(),
                    )
                    self._write_skill_note(_skill, slot, _frame)
                    if self.option_executor is not None:
                        self.option_executor.bank.notify_minted(
                            _skill.skill_id)
                    logger.info(
                        "Minted achievement skill '%s' -> '%s' (slot %d, "
                        "ep %d)", slot_name, _name, slot,
                        self.total_episodes)

            # Persist slot identity + competence once per episode: a few-KB
            # atomic write, and the thing that makes skills survive ANY stop
            # (there is no signal handler on this path — durability comes
            # from writing at every boundary, not from a graceful exit).
            if (self.broadcaster is not None
                    and hasattr(self.broadcaster, "save_state")):
                self.broadcaster.save_state(self._broadcaster_state_path)
            # Consolidate working-set paging: drain page events (bounds the
            # queues on a multi-day run) + persist the long-term index.
            self._drain_goal_paging()
            if self.option_executor is not None:
                # earned prerequisite edges from this episode's practice,
                # then one registry flush for all invocation stats
                _slot_to_sid = {
                    i: f"ach_{i:02d}_{nm}"[:64]
                    for i, nm in enumerate(
                        getattr(self.broadcaster, "slot_names", []))}
                self.skill_bank.mine_cooccurrences(
                    list(self.option_executor.co_log), _slot_to_sid)
                self.option_executor.co_log.clear()
                # RE-DISTILL improved skills (audit critical): a skill mints
                # ONCE at its first-ever unlock, when the policy could barely do
                # it, and nothing ever captures the IMPROVED policy as competence
                # grows. Re-save the current policy for any slot whose competence
                # now clearly beats its stored skill (dedup -> challenger_wins
                # keeps the strict improvement + archives the incumbent). Gated
                # by a competence margin + minimum practice so it neither
                # thrashes disk nor re-saves noise. Within-run BOUND slots stay
                # frozen (refresh only fills empty slots), so meta-values stay
                # stationary; the better weights are picked up at the next
                # empty-slot bind or restart.
                if (self.broadcaster is not None
                        and hasattr(self.broadcaster, "competence")
                        and self.config.get("goals", {}).get(
                            "mint_skills", True)):
                    _comp = self.broadcaster.competence.predict_all()
                    _att = getattr(self.broadcaster, "_attempts", None)
                    for _sl, _sid in _slot_to_sid.items():
                        _sk = self.skill_bank.skills.get(_sid)
                        if _sk is None or _sl >= len(_comp):
                            continue
                        _c = float(_comp[_sl])
                        _at = (int(_att[_sl]) if _att is not None
                               and _sl < len(_att) else 0)
                        if (_c >= 0.3 and _at >= 10
                                and _c > float(_sk.success_rate) + 0.15):
                            self.skill_bank.save_skill(
                                skill_id=_sid, name=_sk.name,
                                policy_state_dict=self.policy.get_state_dict(),
                                description=_sk.description, success_rate=_c,
                                total_episodes=self.total_episodes,
                                context_embedding=_sk.context_embedding,
                                preconditions=_sk.preconditions or None,
                                obs_dim=self.obs_dim,
                                action_dim=self.action_dim, dedup=True,
                                # arch metadata MUST travel with the weights:
                                # omitting it wrote conv tensors under a
                                # 'flat' registry row and the skill was then
                                # permanently SlotRefused (review CRITICAL).
                                **self._arch_save_kwargs())
                            logger.info(
                                "Re-distilled skill %s: success_rate %.2f -> "
                                "%.2f (attempts=%d)", _sid,
                                _sk.success_rate, _c, _at)
                # ---- MEMORY MAINTENANCE (skills as memories) --------------
                # Time passes: unrehearsed skills weaken. Practised ones have
                # improved weights sitting in their slot bindings, so persist
                # those to disk here — the same boundary every other durable
                # write uses, since there is no signal handler on this path.
                _pr = getattr(self.option_executor, "bank", None)
                _pr = getattr(_pr, "practice", None) if _pr else None
                if _pr is not None and _pr.enabled:
                    _stale = _pr.decay()
                    _n = self._persist_practised_skills()
                    _st = _pr.stats()
                    logger.info(
                        "skill memory: %d practised (%d updates, %d "
                        "trust-clipped, %d abstained) | mean strength %.2f "
                        "drift %.4f | %d persisted | %d stale",
                        _st["slots_practised"], _st["updates_total"],
                        _st["clipped_total"], _st["abstained"],
                        _st["mean_strength"], _st["mean_drift"], _n,
                        len(_stale))
                self.skill_bank.flush_if_dirty()
                # BIND newly-minted (and re-distilled) skills into option slots.
                # Without this the lifelong loop only ever binds at
                # _start_stream, so every skill discovered DURING the run (a
                # future break_oak_log included) stays queued in _pending_minted
                # forever and is never invocable (audit critical). refresh_slots
                # fills EMPTY slots only — bound slots stay frozen, preserving
                # meta-value stationarity.
                self.option_executor.bank.refresh_slots()

            # UNRESOLVED LINKS -> ATTENTION. Concepts the grounded facts
            # reference but no skill achieves are visible gaps; the slots
            # that are stepping stones toward them get a frontier boost.
            # Bias only — ghosts never become goal slots of their own (an
            # unachievable slot would be targeted forever).
            if (self.broadcaster is not None
                    and self.config.get("brain_viewer", {}).get(
                        "ghost_frontier", True)
                    and hasattr(self.broadcaster, "set_frontier_bonus")):
                try:
                    from developmental_ai.skill_bank.brain_state import (
                        compute_ghost_frontier)
                    self.broadcaster.set_frontier_bonus(
                        compute_ghost_frontier(
                            self.skill_bank, self.knowledge_graph,
                            self.broadcaster))
                except Exception as _e:
                    logger.warning("ghost frontier skipped: %s", _e)

            if self.brain_emitter is not None:
                self.brain_emitter.emit(self.total_episodes,
                                        self.total_timesteps)

            # ---- Periodic world model + symbolic decoder training ----
            # Lifelong trains the WM on a STEP cadence inside _collect_segment
            # and reuses those metrics here (no re-training); the episodic
            # path trains once per episode, exactly as before.
            if self._lifelong:
                # consume-once (review fix): with the async trainer a block
                # can outlast a segment; re-feeding the SAME reconstruction
                # value would flatten the stage controller's error slope.
                _wm_ready = (self._last_wm_metrics is not None
                             and self._wm_metrics_fresh)
            else:
                _wm_ready = self.replay_buffer.is_ready
            if _wm_ready:
                wm_metrics = (self._last_wm_metrics if self._lifelong
                              else self._train_world_model())
                if self._lifelong:
                    self._wm_metrics_fresh = False
                # producer site 3 of 3: non-lifelong, where _train_world_model
                # was just called inline above. In lifelong mode the metrics
                # were already recorded at their producer (trainer thread or
                # the synchronous branch), so recording again here would
                # double-count. The glue/stage-controller consumption below
                # deliberately STAYS on the consume-once path: re-feeding a
                # stale reconstruction value would flatten its error slope.
                if not self._lifelong:
                    self._record_wm_telemetry(wm_metrics)
                # Feed training signals to glue layer
                self.glue.record_training_signals(
                    kl_value=wm_metrics.get("kl"),
                    prediction_error=wm_metrics.get("reconstruction"),
                )

                # ---- Developmental stage controller (closed-loop) ----
                # WM reconstruction error (learning progress) is the spine;
                # skill count feeds the forward-only confirm. The controller
                # decides when curiosity has done its job and the agent should
                # tip toward task reward — and, with dream enabled, when the
                # world model is trustworthy enough for dream-control.
                stage_info = self.stage_controller.update(
                    prediction_error=wm_metrics.get("reconstruction"),
                    skill_count=self.skill_bank.get_skill_count(),
                    timestep=self.total_timesteps,
                )
                if stage_info["transitioned"]:
                    self._on_stage_transition(stage_info)

                # ---- Dream-based policy training (Stage 2 / IMAGINE only) ----
                # Dream-control is active only in the trust-gated IMAGINE stage,
                # which is unreachable unless dream training is enabled in config
                # AND the world model earns trust. On cheap/dense envs (CartPole)
                # it never engages, keeping PPO in control as intended.
                # Dream-CONTROL only when explicitly enabled (default). In augment
                # mode this stays False so the PPO control/learning path is intact.
                self.dream_training_active = (
                    (not self._lifelong) and self.dream_control and (
                        self.stage_controller.stage
                        == DevelopmentalStageController.IMAGINE))
                if self.dream_training_active:
                    try:
                        dream_metrics = self._dream_train_policy()
                        for k in (
                            "dream_actor_loss",
                            "dream_critic_loss",
                            "dream_returns_mean",
                        ):
                            if k in dream_metrics:
                                self.training_metrics[k].append(
                                    dream_metrics[k]
                                )
                    except ValueError:
                        pass  # Not enough data for dream training yet

                # ---- Imagination-AUGMENTED learning (Rung 4) ----
                # PPO stays in control; the dream actor trains in imagination and
                # is distilled into the real policy => extra learning per real step.
                # Warmup counts episodes on THIS env (audit H2): the global
                # counter meant that after an env swap the condition was
                # already true, so distillation fired from episode 1 on the
                # new env from a stale dream actor and old-env replay.
                self.dream_augment_active = (
                    self.dream_augment
                    and getattr(self, "_episodes_this_env", self.total_episodes)
                    >= self.dream_activate_after
                )
                if self.dream_augment_active:
                    try:
                        dm = self._dream_train_policy()
                        for k in ("dream_actor_loss", "dream_critic_loss",
                                  "dream_returns_mean"):
                            if k in dm:
                                self.training_metrics[k].append(dm[k])
                        dd = self._distill_dream_to_real()
                        # SLEEP CONSOLIDATION for skills. Placed AFTER the
                        # distill so `_wm_trust_ema` reflects this segment's
                        # measured trust — consolidating on last segment's
                        # trust estimate would let a WM that just went bad get
                        # one more round of teaching in.
                        _dc = self._dream_consolidate_skills()
                        if _dc["slots"]:
                            logger.info(
                                "dream consolidation: %d skills rehearsed, "
                                "%d updated (WM trust %.2f)",
                                int(_dc["slots"]), int(_dc["updates"]),
                                _dc["trust"])
                        self.training_metrics["dream_distill_loss"].append(
                            dd["dream_distill_loss"])
                        self.training_metrics["dream_distill_gate_frac"].append(
                            dd["dream_distill_gate_frac"])
                        self.training_metrics["dream_distill_eff_weight"].append(
                            dd["dream_distill_eff_weight"])
                    except (ValueError, RuntimeError) as e:
                        # Don't fail silently — a persistent error here means a
                        # composition bug (e.g. distill input dim vs symbolic), not
                        # a transient. Log the first occurrence so it can't hide.
                        if not getattr(self, "_dream_augment_warned", False):
                            self._dream_augment_warned = True
                            logger.warning(
                                "dream-augment step failed (%s): %r — dreaming is "
                                "NOT training the policy this run.", type(e).__name__, e)

            # ---- Record episode reward for glue mastery detector ----
            self.glue.record_training_signals(
                episode_reward=episode_metrics.get("episode_reward", 0)
            )

            # ---- Step hierarchical goal manager ----
            goal_status = self.glue.step_goal(
                episode_metrics.get("episode_reward", 0)
            )
            if goal_status.get("goal_completed"):
                logger.info("Root goal completed — new goal next episode")
            elif goal_status.get("replan_needed"):
                logger.info("Sub-goal failed — replanning")

            # ---- Check skill mastery (multi-signal via glue layer) ----
            # Success is reward-relative so negative-reward envs (e.g. Acrobot,
            # MountainCar) can still register success / mastery and mint skills.
            # Threshold defaults to 0.0 (i.e. reward > 0), which preserves the
            # original CartPole behavior EXACTLY. Set
            # `environment.success_reward_threshold` per-env for negative-reward
            # tasks (e.g. -150 for Acrobot = "reached the goal in < 150 steps").
            success_threshold = self.config.get("environment", {}).get(
                "success_reward_threshold", 0.0
            )
            episode_success = (
                episode_metrics.get("episode_reward", 0) > success_threshold
            )
            self.mastery_detector.record_episode(
                episode_metrics.get("episode_reward", 0),
                episode_success,
            )
            is_mastered, mastery_signals = self.glue.detect_mastery(
                self.knowledge_graph
            )
            self.training_metrics["glue_mastery_score"].append(
                mastery_signals.get("combined_score", 0)
            )
            if is_mastered:
                self._save_skill(mastery_signals.get("combined_score", 0.8))
                self.glue.mastery_detector.reset()
                self.mastery_detector.reset()

                # ---- Check for skill composition after mastery ----
                if self.skill_bank.get_skill_count() >= 2:
                    pairs = self.skill_composer.find_composable_pairs(
                        self.skill_bank, self.knowledge_graph
                    )
                    for skill_a, skill_b, score, comp_type in pairs:
                        self.skill_composer.compose_skills(
                            skill_a, skill_b, comp_type, self.skill_bank
                        )

            # ---- Update curriculum (multi-signal via glue layer) ----
            if self.curriculum is not None:
                self.curriculum.record_episode(
                    episode_metrics.get("episode_reward", 0),
                    episode_success,
                )
                recommended, curr_signals = self.glue.recommend_difficulty(
                    self.skill_bank,
                    self.knowledge_graph,
                    self.curriculum.current_difficulty,
                )
                self.training_metrics["glue_kg_density"].append(
                    curr_signals.get("kg_density", 0)
                )
                if recommended != self.curriculum.current_difficulty:
                    self.curriculum.current_difficulty = recommended
                    logger.info(
                        f"Difficulty adjusted to {recommended} "
                        f"(depth={curr_signals['skill_tree_depth']:.0f}, "
                        f"density={curr_signals['kg_density']:.1f})"
                    )

            # ---- Periodic graph embedding + knowledge vector update ----
            # Only when the "kg" channel is actually consumed (audit H5): the
            # from-scratch PyKEEN retrain blocks the main loop and scales with
            # the unboundedly growing fact set — pure waste when no consumer.
            if self.knowledge_source == "kg" and self.total_episodes % 50 == 0:
                self._update_graph_embeddings()
                self.glue.update_knowledge(
                    self.kg_embedder, self.knowledge_graph
                )

            # ---- LLM fact extraction (async submit + poll) ----
            # Consume any result that finished on the background thread, then
            # periodically submit a fresh snapshot. Polling every episode keeps
            # results flowing without ever blocking the loop on Ollama.
            self._llm_consume_facts()
            if self.llm.should_extract_facts(self.total_episodes):
                self._llm_submit_facts()

            # ---- LLM knowledge analysis (async submit + poll) ----
            self._llm_consume_analysis()
            if self.llm.should_analyze(self.total_episodes):
                self._llm_submit_analysis()

            # ---- Decay curiosity over time (per-timestep, not per-episode) ----
            # Advance the anneal by the number of timesteps just experienced so
            # the high-curiosity -> high-task shift progresses on a consistent
            # clock even as episodes lengthen.
            # Adaptive gating: feed the per-episode extrinsic return so the anneal
            # tracks how findable the task reward is (no-op unless enabled).
            if getattr(self.reward_mixer, "adaptive_gating", False):
                self.reward_mixer.observe_episode(
                    episode_metrics.get("episode_reward", 0.0))
            self.reward_mixer.decay(
                steps=episode_metrics.get("episode_length", 1)
            )
            # Dream-distillation natural anneal: track how reliably the REAL policy
            # is now solving so the dream's pull can fade as competence takes over
            # (updated unconditionally in augment mode, independent of the mixer's
            # adaptive-gating switch above).
            if self.dream_augment:
                _solved = (1.0 if episode_metrics.get("episode_reward", 0.0)
                           > self._dream_solve_threshold else 0.0)
                self._dream_solve_ema += self._dream_solve_beta * (
                    _solved - self._dream_solve_ema)
            # Anneal the LLM goal-progress shaping weight on the SAME per-timestep
            # clock, so the LLM's guidance fades as the agent's own competence
            # takes over — natural drift, not a fixed shaping ratio (Path B).
            if self.llm_shaping_enabled:
                self.llm_shape_w = max(
                    self.llm_shape_min,
                    self.llm_shape_w
                    - self.llm_shape_decay * episode_metrics.get("episode_length", 1),
                )
            # Keyed to MASTERED skills, not raw registry size (audit fix): the
            # persisted bank held 24 skills at boot — minted largely by trivial
            # first-breaks in prior runs — which quenched curiosity 38% before
            # this run learned anything. Competence should quench curiosity;
            # a monotone counter should not impersonate competence.
            self.curiosity.update_exploration_ratio(
                self.skill_bank.get_stats().get("mastered_skills", 0)
            )

            # ---- Logging ----
            # Telemetry timing snapshot FIRST: _log_progress resets the phase
            # accumulators it reads (measurement only, never raises).
            self._lt_snapshot_timing()
            if verbose >= 1 and self.total_episodes % log_interval == 0:
                self._log_progress()

            # ---- STRUCTURED METRICS: ONE EMISSION SITE ----------------------
            # AFTER _log_progress, and that ordering is load-bearing.
            # The reward ledger's income statement is produced inside
            # InfraStack.segment(), which _log_progress calls — and
            # RewardLedger.segment() is a CONSUMING read, so the sink reads
            # the stash rather than calling it again. Emitting BEFORE this
            # point (as the first version did) meant every record carried the
            # PREVIOUS segment's provenance and the first carried none at all:
            # measured live, seq=1 had reward_total=None while the log printed
            # `infra/ledger: total=+30.98 hhi=0.46 | symbols=62%...`.
            #
            # UNCONDITIONAL on purpose. It sits outside the `if verbose` gate
            # so metrics do not silently stop when logging is turned down —
            # a tracker that disappears with a verbosity flag is exactly the
            # kind of invisible degradation this project keeps paying for.
            # (With the shipped verbose=1/log_interval=1 the stash is fresh
            # every segment; if logging is ever throttled the record carries
            # the last statement produced, which is still true, just older.)
            #
            # Placed at the point where all three stepping bodies have already
            # converged, so it cannot drift between duplicated bodies (§4.2).
            # tests/_metrics_sink_smoke.py asserts this appears exactly ONCE.
            self._emit_metrics(episode_metrics)

            # ---- Checkpointing ----
            # MONOTONE INTERVAL, not a modulo window (fix 2026-08-01, audit
            # #31). The old condition was
            #     total_timesteps % save_interval < env.episode_length + 1
            # which assumed episode_length approximates how far the counter
            # moved since the last check. In LIFELONG that assumption breaks
            # completely: env 0's episode_length grows without bound (it
            # resets only on 64-log success — crash restarts happen inside the
            # adapter and never touch it), so once it exceeds save_interval
            # the predicate is PERMANENTLY TRUE and a checkpoint fired after
            # EVERY 1024-step segment: a full knowledge-graph JSON dump plus
            # six torch.save calls, serially blocking the stream for the rest
            # of a multi-day run. Comparing against the last checkpoint's step
            # is correct in both modes and cannot be fooled by episode length.
            if (self.total_timesteps
                    - getattr(self, "_last_checkpoint_step", 0)
                    >= save_interval):
                self._last_checkpoint_step = self.total_timesteps
                self._save_checkpoint()

        # FINAL CHECKPOINT (2026-08-11, review finding). The only save site is
        # the interval check above, so a graceful stop discarded up to
        # `checkpoint_interval` (25k) steps — including the perception heads
        # and familiarity counts this wave added precisely so they would stop
        # dying with the process. Saving on the way out is what makes the
        # persistence real. Never let a save failure mask the run's result.
        try:
            self._last_checkpoint_step = self.total_timesteps
            self._save_checkpoint()
            logger.info("final checkpoint written at %d timesteps",
                        self.total_timesteps)
        except Exception as _e:
            logger.warning("final checkpoint FAILED (%s) — the run's learned "
                           "state since the last interval save is lost", _e)
        elapsed = time.time() - start_time
        return self._training_summary(elapsed)

    # -----------------------------------------------------------------------
    # Developmental stage transitions
    # -----------------------------------------------------------------------

    def _on_stage_transition(self, stage_info: Dict[str, Any]) -> None:
        """React to a developmental stage change emitted by the controller.

        The reward mixer is re-anchored here so the curiosity->task drift is
        gated by the agent's learning state rather than a wall clock:
          - EXPLORE->EXPLOIT: begin the anneal from the current (high) anchor.
          - EXPLOIT->EXPLORE: re-ignite curiosity (reset intrinsic high, pause).
          - *->IMAGINE: dream-control takes over action selection; the mixer
            continues from its current annealed weights.
        """
        to_stage = stage_info["to"]
        C = DevelopmentalStageController
        if to_stage == C.EXPLOIT:
            self.reward_mixer.start_anneal()
        elif to_stage == C.EXPLORE:
            self.reward_mixer.restore_exploration()

        # The active goal's intent is tied to the regime it was born in
        # (frontier goal in EXPLORE, consolidation goal in EXPLOIT). Reset it so
        # the next generate_goal() builds a stage-appropriate root — otherwise a
        # sticky goal whose threshold is unreachable in the new regime persists
        # indefinitely and the stage-gating never engages.
        self.glue.goal_manager.reset_active_goal()

        logger.info(
            f"[developmental stage] {stage_info['from']} -> {to_stage} "
            f"| reason={stage_info['reason']} "
            f"| WM-error level={stage_info['level']:.4f} "
            f"slope={stage_info['slope']:.5f} "
            f"skill_rate={stage_info['skill_rate']:.3f} "
            f"| intrinsic={self.reward_mixer.weights['intrinsic']:.3f}"
        )

    # -----------------------------------------------------------------------
    # Single episode execution
    # -----------------------------------------------------------------------

    def _run_episode(self) -> Dict[str, float]:
        """
        Run one complete episode of the developmental loop.

        This is where the core cycle happens:
          1. Reset environment
          2. For each step:
             a. Policy selects action
             b. Environment executes action
             c. ICM computes curiosity reward
             d. Fact extractor generates symbolic facts
             e. Replay buffer stores transition
             f. Policy stores experience for training
          3. After episode: update policy, extract action rules
        """
        self._curiosity_boundary()
        obs, info = self._seeded_reset()
        # Rung 6 broadcast: clear the broadcaster and fold in the starting view so
        # the first action is conditioned on what is visible at reset.
        if self.broadcaster is not None:
            self.broadcaster.reset()
            self._broadcaster_update(self.env, 0)   # reset priming (no break)
        # Clear the vision scaffold's per-episode belief (potential anchor +
        # aimed window + any in-flight assessment) so guidance never leaks
        # across the env reset. See VisionScaffold.reset().
        if self.vision_scaffold is not None:
            self.vision_scaffold.reset()
        if self.infra is not None:
            # Re-adopt the approach reference at the boundary: carrying the
            # old distance across a world change would pay (or charge) for a
            # move the agent never made. Same discipline the scaffold uses.
            self.infra.approach_reset()
        done = False
        episode_reward = 0.0
        episode_intrinsic = 0.0
        episode_length = 0
        new_facts_count = 0
        symbolic_decoder_facts_count = 0
        symbolizer_facts_count = 0

        # RSSM state tracking — encode initial obs for dream policy
        rssm_state = self.world_model.rssm.initial_state(1, self.device)
        with torch.no_grad():
            _init_obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            _init_encoded = self.world_model.embed(_init_obs_t)
            _zero_act = torch.zeros(1, self.action_dim, device=self.device)
            rssm_state, _ = self.world_model.rssm.observe_step(
                rssm_state, _zero_act, _init_encoded
            )

        # Feed initial observation to the symbolic decoder's discretizer
        self.symbolic_decoder.update_discretizer(obs)

        while not done:
            # ---- 1. SELECT ACTION ----
            # Snapshot the current KG knowledge feature (detached) so it can
            # condition the policy AND be stored with the transition for the
            # PPO update. None when symbolic conditioning is disabled.
            kv_np = self._current_knowledge_feature()
            if self.dream_training_active:
                with torch.no_grad():
                    _latent = self.world_model.rssm.get_latent(rssm_state)
                action, policy_info = self.dream_actor.select_action(_latent)
            else:
                # arch='rssm' reads the world model's belief, which this path
                # already carries in `rssm_state`. Computed BEFORE the step,
                # so it is the state the decision is actually made from.
                if self.policy.arch == "rssm":
                    with torch.no_grad():
                        self._act_latent = self.world_model.rssm.get_latent(
                            rssm_state).detach()
                else:
                    self._act_latent = None
                action, policy_info = self.policy.select_action(
                    obs, knowledge=kv_np,
                    feats=(None if self._act_latent is None
                           else self._act_latent[0:1]))

            # ---- 2. EXECUTE IN ENVIRONMENT ----
            next_obs, extrinsic_reward, terminated, truncated, step_info = self.env.step(action)
            done = terminated or truncated

            # Rung 6 broadcast: fold the post-step pose + partial view into the
            # broadcaster so the NEXT action sees an up-to-date feature.
            # GROUNDED-EFFECT dedup on the serial path too: pass WHAT broke so
            # the same behaviour keys ONE slot (else viewpoint variation mints
            # a fresh slot + skill each time — the break_*_leaves x8 bug), and
            # pair fresh unlocks with the grounded block for stable skill names.
            if self.broadcaster is not None:
                _blk = self._grounded_break_for(step_info, 0)
                _ub = len(getattr(self.broadcaster, "unlock_log", []))
                self._broadcaster_update(self.env, 0, effect_key=_blk)
                if hasattr(self.broadcaster, "unlock_log"):
                    with torch.no_grad():
                        _lat0 = self.world_model.rssm.get_latent(rssm_state)
                    for _ev in self.broadcaster.unlock_log[_ub:]:
                        _slot = int(_ev["slot"])
                        self._unlock_context[_slot] = (
                            _lat0[0].detach().cpu().numpy())
                        if _blk:
                            self._unlock_block[_slot] = _blk

            # ---- 2b. shaping seed (magnet applied after LP is known) ----
            # `shaped_extrinsic` feeds LEARNING (replay buffer -> WM reward
            # head, and the PPO mix below); `extrinsic_reward` itself stays
            # raw so episode metrics, spike-based goal discovery, and all
            # grading remain uncontaminated.
            shaped_extrinsic = extrinsic_reward

            # ---- 3. COMPUTE CURIOSITY REWARD ----
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)

            # Encode action for ICM input: one-hot (discrete) or the raw action
            # VECTOR (continuous). For continuous, `action` is already a (act_dim,)
            # array, so wrap to (1, act_dim) — NOT [action] which would give 3-D.
            if self.is_discrete:
                action_tensor = torch.zeros(1, self.action_dim).to(self.device)
                action_tensor[0, int(action)] = 1.0
            else:
                action_tensor = torch.FloatTensor(
                    np.asarray(action, dtype=np.float32)
                ).unsqueeze(0).to(self.device)

            intrinsic_reward = self._curiosity_reward(
                obs_tensor, action_tensor, next_obs_tensor,
                ended=[terminated or truncated]).item()
            # ICM BASE SCALE (infra #13): the raw-error term is damped by
            # config so it seasons rather than dominates. Since 2026-10-05
            # paired progress is MEASUREMENT ONLY and pays nothing, so this
            # damped term (plus the itemised novelty terms) IS the base drive
            # — nothing "leads" it any more. 1.0 = old behaviour.
            intrinsic_reward *= float(getattr(self, "_icm_base_scale", 1.0))

            # ---- 3b. CURIOSITY-RANKED MAGNET (waking only) ----
            # Runs after LP is known: the magnet reads intrinsic_reward (this
            # step's learning progress) + the grounding head's per-object probs
            # on the current belief, and steers toward the most-curious in-view
            # object. Guidance only during waking control (dream runs on what
            # was internalized).
            if (self.vision_scaffold is not None
                    and not self.dream_training_active):
                _scaffold_r = self._magnet_step_shaping(
                    action, intrinsic_reward,
                    self.world_model.rssm.get_latent(rssm_state).detach())
                if _scaffold_r:
                    shaped_extrinsic = extrinsic_reward + _scaffold_r

            # ---- 4. MIX REWARDS ----
            mixed_reward = self.reward_mixer.mix(intrinsic_reward, shaped_extrinsic)

            # ---- 5. EXTRACT SYMBOLIC FACTS (two sources) ----
            # Source A: Direct observation discretization (existing FactExtractor).
            # Pixel gate (July 2026, Crafter): the extractor bins per-dimension
            # values — on 12288-d raw pixels it would mint thousands of
            # per-pixel "facts" per step into an unboundedly growing KG.
            new_facts = 0
            if not self._skip_perdim_symbolic:
                new_facts = self._extract_and_store_facts(
                    obs, action, next_obs, extrinsic_reward
                )
            new_facts_count += new_facts

            # Source B: Symbolic decoder on knowledge-augmented RSSM latent
            # Step the RSSM forward to get the latent encoding of this transition
            # ---- WM one-step PREDICTION (prior) for the live accuracy panel ----
            # Imagine forward from the PRE-update state under the action taken,
            # so this is a genuine forecast of next_obs (not a reconstruction
            # that already saw it). Decoder targets are symlog(obs), so invert.
            wm_pred_obs = None
            if self.viewer is not None:
                try:
                    with torch.no_grad():
                        _prior = self.world_model.rssm.imagine_step(
                            rssm_state, action_tensor)
                        _dec = self.world_model.decoder(
                            self.world_model.rssm.get_latent(_prior))
                        wm_pred_obs = (torch.sign(_dec)
                                       * torch.expm1(torch.abs(_dec))
                                       ).squeeze(0).cpu().numpy()
                except Exception:
                    wm_pred_obs = None

            encoded_obs = self.world_model.embed(next_obs_tensor)
            rssm_state, _ = self.world_model.rssm.observe_step(
                rssm_state, action_tensor, encoded_obs
            )
            latent = self.world_model.rssm.get_latent(rssm_state)

            # ---- LIVE VIEWER (optional, best-effort, never breaks training) ----
            if self.viewer is not None:
                self._viewer_push(action, extrinsic_reward, intrinsic_reward,
                                  rssm_state, latent, episode_length,
                                  wm_pred_obs, next_obs)

            # Augment latent with symbolic knowledge from GNN + gate
            augmented_latent = self.glue.augment_latent(latent)

            # Extract high-confidence symbolic facts from augmented latent
            sd_facts = self.symbolic_decoder.extract_facts(
                augmented_latent.squeeze(0), timestep=self.total_timesteps
            )
            for fact in sd_facts:
                if self.knowledge_graph.add_fact(fact):
                    symbolic_decoder_facts_count += 1

            # Update the discretizer with the new observation
            self.symbolic_decoder.update_discretizer(next_obs)

            # Capture scene snapshot periodically for goal generation
            # (gated: per-dimension scene graphs are meaningless on pixels /
            # high-dim structured obs)
            if episode_length % 50 == 0 and not self._skip_perdim_symbolic:
                scene = self.fact_extractor.extract_scene_graph(
                    next_obs, timestep=self.total_timesteps
                )
                self.knowledge_graph.save_scene_snapshot(scene)

            # ---- 6. STORE EXPERIENCE ----
            # Replay buffer (for world model training). Stores the SHAPED
            # extrinsic so the WM reward head — and therefore the dream
            # actor — inherits the (annealing) instinct signal.
            self.replay_buffer.add(obs, action, shaped_extrinsic, done)

            # Policy buffer (skipped during dream training)
            if not self.dream_training_active:
                self.policy.store_transition(
                    obs, action, mixed_reward, done,
                    policy_info["log_prob"], policy_info["value"],
                    knowledge=kv_np, feats=self._feats_row(0),
                )

            # ---- 7. TRAIN CURIOSITY MODULE ----
            icm_metrics = self._curiosity_train(
                obs_tensor, action_tensor, next_obs_tensor
            )

            # Update tracking
            episode_reward += extrinsic_reward
            episode_intrinsic += intrinsic_reward
            episode_length += 1
            self.total_timesteps += 1

            obs = next_obs

        # ---- LLM GOAL-PROGRESS REWARD SHAPING (Path B) ----
        # Applied here, before the PPO update, so the potential-based increment
        # lands in this episode's last transition and is used by the very next
        # train_step. Potential-based + annealed => guides without redefining
        # the task and fades over training (never a fixed ratio).
        self._apply_llm_goal_shaping(episode_reward, episode_length)

        # ---- END OF EPISODE: TRAIN POLICY ----
        # Only run a PPO update once enough transitions have accumulated across
        # episodes (~n_steps). Episodes always end on done=True, so the buffer
        # ends cleanly on an episode boundary and GAE bootstrapping stays valid.
        # This replaces the old per-episode update on ~9 samples, which produced
        # a frozen, zero-gradient policy (Policy loss ~0) that never learned.
        if (
            not self.dream_training_active
            and self.policy.should_update(self.policy_update_steps)
        ):
            # policy.n_epochs, not a hardcoded 10 (fix 2026-08-02): the two
            # other call sites read the config and this one did not, so the
            # knob silently meant different things depending on which loop
            # path a run took.
            policy_metrics = self.policy.train_step(
                n_epochs=int(self.config.get("policy", {}).get(
                    "n_epochs", 10)))
        else:
            policy_metrics = {"policy_loss": 0.0, "value_loss": 0.0}

        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "intrinsic_reward": episode_intrinsic / max(1, episode_length),
            "policy_loss": policy_metrics.get("policy_loss", 0),
            "curiosity_loss": icm_metrics.get("icm_total", 0),
            "new_facts": new_facts_count,
            "symbolic_decoder_facts": symbolic_decoder_facts_count,
        }

    # -----------------------------------------------------------------------
    # Parallel episode execution (vectorized data collection)
    # -----------------------------------------------------------------------

    def _ensure_parallel_envs(self) -> None:
        """Lazily create the extra env copies (env[0] is the primary self.env)."""
        if self._parallel_envs:
            return
        # Persistent pool for concurrent env stepping/resets. Real-game envs
        # (MineRL) block on Java with the GIL released, so threads genuinely
        # parallelize; fast envs see only microseconds of overhead.
        from concurrent.futures import ThreadPoolExecutor
        self._env_pool = ThreadPoolExecutor(
            max_workers=self._num_envs, thread_name_prefix="envstep")
        self._parallel_envs = [self.env]
        self._parallel_curricula = [self.curriculum]
        env_cfg = self.config.get("environment", {})
        if env_cfg.get("remote_server") and env_cfg.get(
                "remote_server_scope", "primary") == "all":
            # Say it out loud: a silent name collision here is invisible in
            # this log and only shows up as clients evicting each other in
            # the JAVA logs, which is where the first attempt hid.
            logger.info("server identities: %s (slot 0 keeps the bare name "
                        "so its player data survives)",
                        ", ".join(self._agent_name_for(i)
                                  for i in range(self._num_envs)))
        # ENUMERATED (2026-08-17): this loop discarded its index, so every
        # scout built here was anonymous — and with remote_server_scope
        # "all" they all joined the server under slot 0's name and evicted
        # each other. The rebuild path (_make_one_env) already knew its
        # slot; this one, which builds the ORIGINAL scouts, did not.
        for _sidx in range(1, self._num_envs):
            e, c = make_env(
                env_name=self.env_name,
                normalize=True,
                curriculum=self.config.get("loop", {}).get("curriculum_enabled", True),
                render_mode=None,
                max_episode_steps=env_cfg.get("max_episode_steps"),
                pixel_obs=self.pixel_obs,
                image_size=self.image_size,
                grayscale=env_cfg.get("grayscale", False),
                agent_view_size=env_cfg.get("agent_view_size"),
                action_repeat=env_cfg.get("action_repeat", 1),
                render_size=env_cfg.get("render_size", 0),
                # THE SENSOR BUS. Absent from config -> None -> the
                # adapter emits info["proprio"] and nothing else, i.e.
                # exactly the pre-bus behaviour for every other config.
                sensors_cfg=(self.config.get("sensors") or None),
                lifelong=getattr(self, "_lifelong", False),
                # remote_server_scope "primary" (default): only env 0 — the
                # continuous lifelong stream — joins the external server; the
                # scout envs explore local generated worlds. "all": every
                # stream joins (N bots on the server).
                remote_server=(env_cfg.get("remote_server")
                               if env_cfg.get("remote_server_scope",
                                              "primary") == "all" else None),
                agent_name=self._agent_name_for(_sidx),
            )
            self._parallel_envs.append(e)
            self._parallel_curricula.append(c)
        logger.info(
            f"Parallel collection active with {self._num_envs} environments"
        )

    def _agent_name_for(self, idx: int) -> str:
        """Server-visible player name for parallel slot `idx`.

        Slot 0 keeps the BARE prefix. An offline-mode server derives the
        player UUID from the name, so the lifelong stream's position,
        inventory and history live under "SkyBot" — renaming it would hand
        it a brand-new player at world spawn with empty hands and silently
        strand everything it has built. Scouts take a suffix, because a
        duplicate login is resolved by kicking the incumbent: without this
        N clients evict each other forever.
        """
        _pref = str((self.config.get("environment", {}) or {}).get(
            "agent_name_prefix", "SkyBot"))
        return _pref if int(idx) <= 0 else f"{_pref}{int(idx)}"

    def _make_one_env(self, idx: int = -1):
        """Build one fresh env (used to replace a hung/dead parallel env).
        See _agent_name_for for why slot 0's name is special.
        (helper defined just above)

        idx is the parallel-env slot being replaced: slot 0 is the primary
        lifelong stream, which must KEEP its remote_server binding across
        crash-rebuilds (else the visitor silently respawns into a local
        world). Scouts (idx != 0) follow remote_server_scope.
        """
        env_cfg = self.config.get("environment", {})
        _scope = env_cfg.get("remote_server_scope", "primary")
        _remote = env_cfg.get("remote_server") if (
            idx == 0 or _scope == "all") else None
        # SERVER IDENTITY (2026-08-17). Slot 0 KEEPS the bare prefix on
        # purpose: an offline-mode server derives the player UUID from the
        # name, so renaming slot 0 would hand the lifelong stream a brand
        # new player — world spawn, empty inventory, none of its history.
        # Scouts get a suffix so no two clients can evict each other.
        _name = self._agent_name_for(idx)
        e, _c = make_env(
            env_name=self.env_name, normalize=True,
            curriculum=self.config.get("loop", {}).get(
                "curriculum_enabled", True),
            render_mode=None,
            max_episode_steps=env_cfg.get("max_episode_steps"),
            pixel_obs=self.pixel_obs, image_size=self.image_size,
            grayscale=env_cfg.get("grayscale", False),
            agent_view_size=env_cfg.get("agent_view_size"),
            action_repeat=env_cfg.get("action_repeat", 1),
            render_size=env_cfg.get("render_size", 0),
            # THE SENSOR BUS. Absent from config -> None -> the adapter
            # emits info["proprio"] and nothing else, i.e. exactly the
            # pre-bus behaviour for every other config in the tree.
            sensors_cfg=(self.config.get("sensors") or None),
            lifelong=self._lifelong,
            remote_server=_remote,
            agent_name=_name,
            remote_step_delay_s=env_cfg.get("remote_step_delay_s", 0.0),
            # LOG ECONOMICS (2026-08-03): pays per attack tick invested in
            # the swing that fells a log, so the PROCESS of chopping out-earns
            # any other block per tick — not just the completion event.
            # 0.0 = off = byte-identical for every other config. BOTH call
            # sites get it: the scouts feed the same shared world model, so
            # different reward rules per stream would teach it two economies.
            log_tick_reward=env_cfg.get("log_tick_reward", 0.0),
            log_tick_cap=env_cfg.get("log_tick_cap", 120),
            log_break_reward=env_cfg.get("log_break_reward", 5.0),
            break_decay_scale=env_cfg.get("break_decay_scale", 0.0),
            # break-mastery memory is the PRIMARY stream's biography — scouts
            # write nothing (concurrent writers would race the same file, and
            # a scout's local-world grinding is not the visitor's history).
            break_memory_path=(env_cfg.get("break_memory_path", None)
                               if idx == 0 else None),
            **({} if env_cfg.get("start_tool", "__default__") == "__default__"
               else {"start_tool": env_cfg.get("start_tool")}))
        return e

    def _parallel_reset(self, envs):
        """Reset all N envs concurrently with a PER-ENV TIMEOUT. A MineRL
        client that hangs its reset (no exception, just a stuck socket) would
        otherwise block the whole lockstep barrier forever (observed: a 16-env
        run stalled with 0 episodes). On timeout we abandon the stuck client
        (its worker thread leaks but is harmless) and rebuild that env in
        place, so one bad client can't freeze the fleet.
        """
        from concurrent.futures import TimeoutError as _FTimeout
        timeout = float(self.config.get("parallel_envs", {}).get(
            "reset_timeout_s", 300))
        def _reap(old_env, i):
            """Close an ABANDONED client in the background.

            The docstring above says the leaked worker thread "is harmless".
            That was measured on the 125 GB GPU host. On `main` (15.3 GB, 2026-09-23)
            it is NOT: the abandoned process is a full Minecraft JAVA CLIENT
            holding ~0.5-1 GB, nothing ever reaped it, and `status` showed
            `java clients: 3 (want 2)` while the box sat at 99% swap with the
            OOM killer already firing. Every hung reset leaked another one.

            close() is exactly what may hang -- that is why this env was
            abandoned -- so it runs on a DAEMON thread that is never joined.
            If it succeeds the memory comes back; if it hangs we are no worse
            off than before. It must never delay the barrier this whole
            function exists to protect.
            """
            # LOCAL IMPORT, because `threading` is NOT imported at module
            # scope in this file -- only inside __init__ (755) and
            # _ensure_wm_trainer (7636). This cost 7 crashed runs:
            #   File "developmental_loop.py", line 4108, in _reap
            #     threading.Thread(target=_c, daemon=True,
            #   NameError: name 'threading' is not defined
            # I "verified" the import with ast.walk, which traverses the WHOLE
            # tree and happily matched those two FUNCTION-LOCAL imports. The
            # check passed and meant nothing. Module-level imports are
            # `tree.body`, not `ast.walk(tree)`.
            import threading

            def _c():
                try:
                    old_env.close()
                    logger.info("env %d: abandoned client closed", i)
                except Exception as ex:
                    logger.warning("env %d: abandoned client would not close "
                                   "(%s) — it will hold RAM until the run ends",
                                   i, ex)
            threading.Thread(target=_c, daemon=True,
                             name=f"reap-env{i}").start()

        def _bounded_rebuild(i):
            # Rebuild env i and reset it THROUGH THE POOL with a timeout, so a
            # rebuild that also hangs (the hung MineRL client won't die and the
            # new one won't come up) can't re-block the barrier forever — the
            # earlier bug that re-stalled the 8-env run. Give up to 2 tries,
            # then zero-fill and carry on with a dead stream for this episode.
            for _try in range(2):
                try:
                    envs[i] = self._make_one_env(i)
                    rf = self._env_pool.submit(envs[i].reset)
                    o = rf.result(timeout=timeout)[0]
                    return np.asarray(o, dtype=np.float32)
                except Exception as ex:
                    logger.warning("env %d rebuild try %d failed (%s)",
                                   i, _try + 1, ex)
            logger.error("env %d rebuild exhausted; zero-filling stream", i)
            return np.zeros(self.obs_dim, dtype=np.float32)

        obs_list = [None] * len(envs)
        futs = {i: self._env_pool.submit(e.reset) for i, e in enumerate(envs)}
        for i, f in futs.items():
            try:
                obs_list[i] = np.asarray(f.result(timeout=timeout)[0],
                                         dtype=np.float32)
            except _FTimeout:
                logger.warning("env %d reset exceeded %.0fs — rebuilding "
                               "(abandoning the hung one)", i, timeout)
                _reap(envs[i], i)          # or it leaks a whole java client
                obs_list[i] = _bounded_rebuild(i)
            except Exception as ex:
                logger.warning("env %d reset error (%s) — rebuilding", i, ex)
                _reap(envs[i], i)
                obs_list[i] = _bounded_rebuild(i)
        return obs_list

    def _drain_goal_paging(self) -> None:
        """Consume the broadcaster's working-set page events and persist the
        long-term index. Draining bounds the event queues on a multi-day run;
        the store's save keeps dormant knowledge across process restarts.

        A page-OUT never wipes a skill: skill_bank_data lives in its own dirs
        and is retained on disk. The freed goal slot simply stops being
        offered; a later unlock on the reused slot re-mints its skill through
        the normal mint path, and a RECALL rebinds the dormant behaviour. Full
        skill/option rebinding on recall is deferred (P3c) — logged here so the
        paging activity is never silent.
        """
        br = self.broadcaster
        if br is None or not hasattr(br, "drain_page_events"):
            return
        po, pi = br.drain_page_events()
        tot = getattr(self, "_page_events_total", None)
        if tot is not None:
            tot["out"] += len(po)
            tot["in"] += len(pi)
        for ev in po:
            logger.info("goal slot %d PAGED OUT ('%s' -> %s); skill data kept",
                        ev["slot"], ev["name"], ev["mem_id"])
        for ev in pi:
            logger.info("goal slot %d RECALLED ('%s' <- %s)",
                        ev["slot"], ev["name"], ev["mem_id"])
        ltm = getattr(self, "_goal_ltm", None)
        if ltm is not None:
            ltm.save()

    def _grounded_break_for(self, step_info, env_idx: int = 0):
        """Highest-tier newly-broken block from ONE step's achievements (the
        grounded behaviour identity for effect-keyed dedup). Mirrors the
        parallel-path _new_break_by_env extraction for the serial single-env
        loop; updates the per-env mine high-water marks. log > solid > plant."""
        if not isinstance(step_info, dict):
            return None
        ach = step_info.get("achievements", {}) or {}
        hw = self._mine_by_env.setdefault(env_idx, {})
        best, best_rank = None, -1
        for k, v in ach.items():
            if not k.startswith("mine_"):
                continue
            b = k[5:]
            try:
                c = int(v)
            except (TypeError, ValueError):
                continue
            if c > hw.get(b, 0):
                hw[b] = c
                rank = (2 if "log" in b else
                        0 if b in ("grass", "tall_grass", "fern", "vine",
                                   "seagrass") else 1)
                if rank > best_rank:
                    best, best_rank = b, rank
        return best

    def _broadcaster_update(self, env, stream: int, effect_key=None) -> None:
        """Feed one env into the broadcaster, tolerating both signatures.

        The achievement broadcasters accept a `stream` index (per-stream
        episode state); the rung-6 broadcasters (Episodic/Affordance/
        RuleRegime) are single-stream `update(env)`. Only the achievement
        family reaches the parallel path in practice, but stay signature-
        agnostic so a mis-paired config degrades to primary-stream-only
        rather than crashing.
        """
        # Detect the broadcaster's update() ARITY ONCE (achievement arms take
        # (env, stream, effect_key); rung-6 arms take (env)) and dispatch
        # directly, so a REAL TypeError from inside update() propagates instead
        # of being masked as a signature mismatch — a masked TypeError would
        # silently retry WITHOUT effect_key, downgrading grounded-effect dedup
        # back to the noisy-signature clustering this change fixes.
        mode = getattr(self, "_bc_update_mode", None)
        if mode is None:
            import inspect
            try:
                params = inspect.signature(self.broadcaster.update).parameters
            except (TypeError, ValueError):
                params = {}
            mode = ("effect" if "effect_key" in params
                    else "stream" if len(params) >= 2 else "env")
            self._bc_update_mode = mode
        if mode == "effect":
            self.broadcaster.update(env, stream, effect_key=effect_key)
        elif mode == "stream":
            self.broadcaster.update(env, stream)
        elif stream == 0:
            self.broadcaster.update(env)

    def _run_episode_parallel(self, use_dream_actor: bool = True
                              ) -> Dict[str, float]:
        """
        Collect experience from N environments stepped in lockstep.

        The PRIMARY env (index 0) defines the episode boundary returned to the
        main loop; the other envs autoreset and keep filling their own replay
        streams. ICM/RSSM ops are BATCHED across all N envs. Symbolic fact
        extraction and knowledge-graph writes are primary-env only.

        Two actor modes:
          * use_dream_actor=True (IMAGINE phase): the latent-space dream actor
            selects actions; the real PPO policy is dormant.
          * use_dream_actor=False (WAKING/curiosity phase): the REAL PPO policy
            selects actions per env. All N streams feed the world model +
            curiosity + discovery, but ONLY the primary stream (0) feeds the
            PPO rollout buffer + the vision scaffold shaping, so PPO's
            on-policy GAE stays single-stream-correct. The other N-1 envs are
            curiosity SCOUTS that widen world-model coverage and first-break
            discovery without perturbing the policy update.
        """
        self._ensure_parallel_envs()
        n = self._num_envs
        envs = self._parallel_envs

        # Keep every env at the primary env's curriculum difficulty.
        if self.curriculum is not None:
            for c in self._parallel_curricula:
                if c is not None:
                    c.current_difficulty = self.curriculum.current_difficulty

        # Reset all envs CONCURRENTLY (July 2026, Minecraft-parallel fix):
        # env.reset()/step() on real-game envs (MineRL) block on a Java
        # subprocess for 70ms-90s with the GIL released, so serial resets
        # would cost N x 90s at every episode boundary. Fast envs are
        # unaffected (thread overhead is microseconds).
        self._curiosity_boundary()
        obs_list = self._parallel_reset(envs)
        self.symbolic_decoder.update_discretizer(obs_list[0])

        # Goal broadcaster (July 2026): the parallel path predates the goal
        # channel — without these hooks, goal targeting / discovery / skill
        # minting silently die in parallel mode. Primary-stream only (an
        # accepted degradation: discovery sees 1/N of the experience).
        if self.broadcaster is not None:
            if hasattr(self.broadcaster, "set_num_streams"):
                self.broadcaster.set_num_streams(len(envs))
            self.broadcaster.reset()
            for _i, e in enumerate(envs):
                self._broadcaster_update(e, _i)
        # Clear the vision scaffold's per-episode belief (waking parallel path
        # shapes the primary stream, same as the single-env path).
        if not use_dream_actor and self.vision_scaffold is not None:
            self.vision_scaffold.reset()
        if self.infra is not None:
            # Re-adopt the approach reference at the boundary: carrying the
            # old distance across a world change would pay (or charge) for a
            # move the agent never made. Same discipline the scaffold uses.
            self.infra.approach_reset()
        # Re-adopt the stream-0 COST potentials at the boundary (2026-10-03).
        # _collect_segment does this on a client rebuild; this body never
        # did, so an episode that ended inside a menu (Phi = -1) handed the
        # next one a phantom +w refund on its first step. None = "first
        # sample, charge nothing" — the sentinel both cost potentials use.
        self._pitch_level_phi = None
        self._gui_run = 0
        self._gui_dwell_phi = None
        if self.symbolizer is not None:
            self._prev_mine.clear()   # counts restart at 0 in a fresh world
        # per-env break high-water also restarts (new world every reset)
        self._mine_by_env.clear()
        self._new_break_by_env.clear()
        if not use_dream_actor and self.option_executor is not None:
            # close any dangling options (telemetry only) and bind newly
            # minted skills into EMPTY slots — bound slots stay frozen.
            self.option_executor.clear_all(self.total_timesteps)
            self.option_executor.bank.refresh_slots()

        # Batched RSSM state, initialised exactly like the single-env path.
        rssm_state = self.world_model.rssm.initial_state(n, self.device)
        with torch.no_grad():
            obs_t = torch.from_numpy(np.stack(obs_list)).to(self.device)
            encoded = self.world_model.embed(obs_t)
            zero_act = torch.zeros(n, self.action_dim, device=self.device)
            rssm_state, _ = self.world_model.rssm.observe_step(
                rssm_state, zero_act, encoded
            )

        episode_reward = 0.0
        episode_intrinsic = 0.0
        episode_length = 0
        new_facts_count = 0
        symbolic_decoder_facts_count = 0
        symbolizer_facts_count = 0
        icm_metrics: Dict[str, float] = {}
        primary_done = False

        primary_kv = None       # knowledge feature stored with the PPO rollout
        primary_pol = None      # PPO policy_info for the primary stream
        while not primary_done:
            _pt = time.perf_counter()          # phase clock (see _phase_mark)
            obs_t = torch.from_numpy(np.stack(obs_list)).to(self.device)

            # ---- 1. SELECT ACTIONS ----
            if use_dream_actor:
                # Latent-space dream actor, batched across all N envs.
                with torch.no_grad():
                    latent = self.world_model.rssm.get_latent(rssm_state)
                    dist = self.dream_actor.get_action_dist(latent)
                    sampled = dist.sample()
                    if self.is_discrete:
                        action_idx = sampled.long().view(-1)
                        action_tensor = torch.nn.functional.one_hot(
                            action_idx, self.action_dim).float()
                        env_actions = [int(a) for a in action_idx.cpu().numpy()]
                    else:
                        action_tensor = sampled.view(n, self.action_dim).float()
                        env_actions = [a for a in action_tensor.cpu().numpy()]
            else:
                # WAKING: real PPO policy per env. Primary stream keeps its
                # log_prob/value for the on-policy update; scouts just act.
                primary_kv = self._current_knowledge_feature()
                # THE BELIEF THIS DECISION IS MADE FROM, hoisted ABOVE the
                # options branch (2026-09-01). arch='rssm' needs it on BOTH
                # paths — with options disabled the loop stores scout and
                # primary rows from the `else` branch below, and reading a
                # latent set only inside the options branch would have handed
                # the update last step's belief, silently.
                if self.policy.arch == "rssm":
                    with torch.no_grad():
                        self._act_latent = self.world_model.rssm.get_latent(
                            rssm_state).detach()
                else:
                    self._act_latent = None
                if self.option_executor is not None:
                    # options path: executor resolves primitives per env,
                    # opening/continuing skill invocations as sampled. Its
                    # decision records replace primary_pol for storage;
                    # keep primary_pol non-None so the dream-phase guard
                    # below still distinguishes waking from dreaming.
                    # THE BELIEF THE DECISION IS MADE FROM. Computed once,
                    # before the action, and reused by both consumers: the
                    # precondition gate and (arch='rssm') the policy input
                    # itself. This is the state the agent is actually in when
                    # it chooses — computing it after the step would score the
                    # initiation set against a world the choice already
                    # changed.
                    _preds = None
                    if self.option_executor.gate_by_preconditions:
                        with torch.no_grad():
                            _preds = self._predicates_batch(
                                self._act_latent
                                if self._act_latent is not None
                                else self.world_model.rssm.get_latent(
                                    rssm_state))
                    # live per-skill competence for the warmup gate:
                    # broadcaster slot N -> skill ach_NN_.. -> competence[N]
                    _comp = None
                    if (self.option_executor.competence_floor > 0.0
                            and self.broadcaster is not None
                            and hasattr(self.broadcaster, "competence")):
                        _cv = self.broadcaster.competence.predict_all()
                        _nm = getattr(self.broadcaster, "slot_names", [])
                        # Keyed by BOTH the reconstructed name AND the raw slot
                        # index. A skill id freezes the slot name it was minted
                        # under, so once a slot is re-keyed (grounded effect <->
                        # discovered_N) only the index still matches; the bank's
                        # gate falls back to it via options._slot_index_of.
                        _comp = {f"ach_{i:02d}_{n}"[:64]: float(_cv[i])
                                 for i, n in enumerate(_nm) if i < len(_cv)}
                        _comp.update({i: float(_cv[i])
                                      for i in range(len(_cv))})
                    # SELF-STATE THE AGENT CURRENTLY FEELS. act() runs
                    # BEFORE this step, so it must use the LAST observation's
                    # proprio — reading this step's step_infos here referenced
                    # it before assignment and crashed the episodic path on
                    # its very first iteration.
                    _props = getattr(self, "_proprio_per_env", None)
                    env_actions = self.option_executor.act(
                        obs_list, primary_kv, self.policy,
                        self.total_timesteps, predicates_per_env=_preds,
                        competence=_comp, proprio_per_env=_props,
                        feats_per_env=self._feats_for_act(n),
                        verify_feats=self._verify_enc_feats)
                    primary_pol = (self.option_executor._primary_primitive
                                   or {"log_prob": 0.0, "value": 0.0})
                else:
                    env_actions = []
                    # SCOUT DECISIONS ARE KEPT HERE TOO (2026-09-01). This
                    # branch runs when options are DISABLED, and it already
                    # computed `pinfo` for every env and then discarded it for
                    # e_i != 0. The scout-PPO storage below hangs off the
                    # option executor, which is None on this path — so with
                    # skills_as_options.enabled false, `_scouts_in_ppo` read
                    # true and not one scout row was ever stored. That is the
                    # _viewer_push / gui_open / felt-reach pattern exactly: a
                    # correct mechanism nobody calls.
                    self._noopt_pol = [None] * n
                    for e_i in range(n):
                        a, pinfo = self.policy.select_action(
                            obs_list[e_i],
                            knowledge=primary_kv if e_i == 0 else None,
                            feats=(None if self._act_latent is None
                                   else self._act_latent[e_i:e_i + 1]))
                        env_actions.append(int(a) if self.is_discrete else a)
                        self._noopt_pol[e_i] = pinfo
                        if e_i == 0:
                            primary_pol = pinfo
                if self.is_discrete:
                    action_tensor = torch.zeros(n, self.action_dim,
                                                device=self.device)
                    action_tensor[range(n),
                                  [int(a) for a in env_actions]] = 1.0
                else:
                    action_tensor = torch.from_numpy(
                        np.stack(env_actions).astype(np.float32)
                    ).to(self.device)

            _pt = self._phase_mark("act", _pt)
            self._lt_pre_step(env_actions, n, use_dream_actor)
            # FOUNDATION SHADOW (off unless foundation.shadow.enabled): obs t
            # and the world model's prediction for the chosen action, logged
            # BEFORE env.step so the store can prove it preceded the outcome.
            if self._shadow is not None:
                self._shadow.before_step(rssm_state, env_actions,
                                         self._wm_param_lock, self.world_model,
                                         executor=self.option_executor, observations=obs_list)
                _pt = self._phase_mark("shadow", _pt)

            # ---- 2. STEP ALL ENVS (concurrently — see reset note) ----
            from concurrent.futures import TimeoutError as _FTimeout
            _step_to = float(self.config.get("parallel_envs", {}).get(
                "step_timeout_s", 120))
            # WHERE THE WALL CLOCK GOES (2026-08-17). The run holds ~3.4
            # steps/s against a 10 steps/s ceiling (20 server ticks/s over
            # action_repeat 2) while the GPU sits idle, so two thirds of
            # every step is unaccounted for. Timing the env round-trip is
            # what separates "MineRL is slow" from "our own per-step Python
            # is slow" — those call for completely different work, and
            # guessing which has already cost this project weeks. Set in
            # BOTH duplicated bodies on purpose: letting these two drift is
            # how the last six same-shape bugs were born.
            _t_env0 = time.time()
            futs = [self._env_pool.submit(envs[e_i].step, env_actions[e_i])
                    for e_i in range(n)]
            next_obs_list, rewards, dones = [], [], []
            _sh_ends = []   # shadow: (terminated, truncated, t_result) per env
            # Whether this step's `done` is a CLIENT REBUILD rather than a
            # real terminal. The episodic body needs it for the same reason
            # the lifelong one does: the world model must not be taught the
            # splice across a rebuild as dynamics (see ReplayBuffer.add).
            restarted: List[bool] = []
            prim_info: Dict[str, Any] = {}
            step_infos: List[Dict[str, Any]] = [None] * n
            # keep the body sense fresh for the NEXT act() (see _props)
            if self._proprio_source is not None:
                self._proprio_per_env = [None] * n
            # PERSISTS across steps (it is a one-step memory), so it is sized
            # once and never cleared here — unlike _proprio_per_env above,
            # which is this step's fresh reading.
            if (self._wm_proprio_dim
                    and (self._wm_proprio_prev is None
                         or len(self._wm_proprio_prev) != n)):
                self._wm_proprio_prev = [None] * n
            for e_i, f in enumerate(futs):
                _rst = False
                try:
                    nobs, rew, term, trunc, _inf = f.result(timeout=_step_to)
                    if isinstance(_inf, dict):
                        step_infos[e_i] = _inf
                        if (self._proprio_per_env is not None
                                and e_i < len(self._proprio_per_env)):
                            # loop-side senses (reach + episodic bearing) —
                            # ONE shared assembly for every body, see
                            # _augment_proprio
                            self._proprio_per_env[e_i] = \
                                self._augment_proprio(
                                    (_inf or {}).get("proprio"), e_i)
                    if e_i == 0 and isinstance(_inf, dict):
                        prim_info = _inf   # symbolizer reads mine_* from here
                except (_FTimeout, Exception) as ex:
                    # Hung/failed client: end its episode (done) so the next
                    # boundary rebuilds it via _parallel_reset; carry the last
                    # obs, zero reward. One straggler can't freeze the fleet.
                    logger.warning("env %d step hung/failed (%s) — truncating",
                                   e_i, ex)
                    nobs = (obs_list[e_i]
                            if obs_list[e_i] is not None
                            else np.zeros(self.obs_dim, dtype=np.float32))
                    rew, term, trunc, _rst = 0.0, False, True, True
                next_obs_list.append(np.asarray(nobs, dtype=np.float32))
                rewards.append(float(rew))
                _sh_ends.append((bool(term), bool(trunc), time.time()))
                restarted.append(_rst)
                dones.append(bool(term or trunc))
            # envs step CONCURRENTLY, so this is the wait for the SLOWEST
            # client, which is exactly what the serial loop pays. Present in
            # BOTH bodies: the timer above was duplicated but this accumulator
            # was not, so `_t_env0` here was an unused local and this path
            # never produced the `Loop timing:` line at all (7th instance of
            # the drift the comment above warns about).
            self._env_wait_sum = (getattr(self, "_env_wait_sum", 0.0)
                                  + (time.time() - _t_env0))
            self._env_wait_n = getattr(self, "_env_wait_n", 0) + 1
            self._env_parts_add(step_infos)
            _pt = self._phase_mark("env", _pt)
            # ---- SET HERE, NOT IN THE COVERAGE BLOCK (2026-09-01) ----
            # `_boring_view_factor()` reads `_last_world_info["pitch"]`, and
            # it is called from the ICM base discount — which runs EARLIER in
            # this body than the coverage block that used to assign it. So a
            # term scaling up to 85% of the base drive was computed from the
            # PREVIOUS decision's pitch, and action_repeat 2 -> 4 doubled that
            # staleness from 2 game ticks to 4. Assigned immediately after the
            # env results land, which is the first moment it is knowable.
            # ALSO ADDED TO THE EPISODIC BODY, which never set it at all: it
            # was reading whatever the last lifelong segment left behind.
            # ...the FULL info too: ticks-to-break lives at top level, not
            # under "world", and reading the wrong dict would silently print
            # nothing — which is how a measurement quietly becomes a
            # non-measurement (this has happened repeatedly here).
            if step_infos:
                self._last_world_info = (step_infos[0] or {}).get("world") or {}
                self._last_env_info = step_infos[0] or {}
            # per-env newly-broken block this step (highest tier wins:
            # log > solid > plant, mirroring the tiered break reward, so the
            # naming picks the block that actually spiked the reward)
            self._new_break_by_env = {}
            for _e in range(n):
                _inf = step_infos[_e]
                if not isinstance(_inf, dict):
                    continue
                _ach = _inf.get("achievements", {}) or {}
                _hw = self._mine_by_env.setdefault(_e, {})
                _best, _best_rank = None, -1
                for _k, _v in _ach.items():
                    # CRAFTS ARE GROUNDED EFFECTS TOO (2026-07-26): a
                    # craft_item increment is as real as a block break and
                    # outranks everything — making something is the rarest,
                    # most significant event in this world. The effect key
                    # KEEPS its craft_ prefix (`craft_planks`), so it can
                    # never collide with a block of the same name and the
                    # goal slot is legibly a crafting goal.
                    if _k.startswith("craft_"):
                        _b, _rank_base = _k, 3
                    elif _k.startswith("mine_"):
                        _b, _rank_base = _k[5:], None
                    else:
                        continue
                    try:
                        _c = int(_v)
                    except (TypeError, ValueError):
                        continue
                    if _c > _hw.get(_b, 0):
                        _hw[_b] = _c
                        # MASTERY GROUND TRUTH (task #39): every open option
                        # frame on this env witnesses the effect key in the
                        # SAME vocabulary skills are named in (break_X /
                        # craft_Y) — at close, each frame is asked whether
                        # its OWN key is among what it witnessed.
                        if self.option_executor is not None:
                            self.option_executor.note_effect(
                                _e, _k if _k.startswith("craft_")
                                else f"break_{_b}")
                        _rank = (_rank_base if _rank_base is not None else
                                 (2 if "log" in _b else
                                  0 if _b in ("grass", "tall_grass", "fern",
                                              "vine", "seagrass") else 1))
                        if _rank > _best_rank:
                            _best, _best_rank = _b, _rank
                # LOG-PICKUP ATTRIBUTION GUARD (audit fix): the +1/log
                # inventory reward has no mine_* counterpart (a pickup breaks
                # nothing), so a pickup spike landing on the same step as an
                # incidental leaf/grass break used to be grounded-keyed to the
                # VEGETATION — minting exactly the foliage goals the tiered
                # rewards were built to suppress, and blending leaf signatures
                # into log-goal provenance. If the log count rose this step and
                # no mine_*log break explains it, pass NO effect key (ungrounded
                # signature clustering handles that spike correctly).
                if not hasattr(self, "_loginv_by_env"):
                    self._loginv_by_env = {}
                # ---- MISSING IS NOT ZERO (fix 2026-09-04) ----------------
                # MEASURED: the live run reported log_pickup=1547 against
                # NINE logs ever broken. Those cannot both be true, and the
                # counter was the liar.
                # CAUSE: `_ach` is `_inf.get("achievements", {}) or {}`, so
                # on any step whose info lacks that dict (or lacks the "log"
                # key) `_ach.get("log", 0)` returned 0 — indistinguishable
                # from "the agent is carrying zero logs". Carrying 3 logs
                # through a single dropped observation therefore reads
                # 3 -> 0 -> 3, and the recovery is counted as a PICKUP. With
                # an intermittent key that manufactures hundreds of pickups
                # out of one real one, which is exactly the 1547-vs-9 gap.
                # This is the counter class CLAUDE.md §5 warns about by name
                # (`places` listing iron_axe: 2281) — and it is not cosmetic:
                # log_pickup is a GROUNDED EFFECT KEY, so every phantom
                # pickup fed a real goal slot and a real reward event.
                # An absent reading is UNKNOWN. Skip the comparison and do
                # not overwrite the last KNOWN count, so a dropped frame is
                # simply not evidence rather than being evidence of a gain.
                _lg = None
                if isinstance(_ach, dict) and "log" in _ach:
                    try:
                        _lg = max(0, int(_ach["log"]))
                    except (TypeError, ValueError):
                        _lg = None
                _lg_prev = self._loginv_by_env.get(_e)
                if _lg is not None:
                    self._loginv_by_env[_e] = _lg
                else:
                    self._loginv_unknown = getattr(
                        self, "_loginv_unknown", 0) + 1
                if (_lg is not None and _lg_prev is not None
                        and _lg > _lg_prev
                        and (_best is None or "log" not in _best)):
                    # STABLE PICKUP KEY (2026-07-25) — was `_best = None`.
                    # Refusing to mis-ground a pickup was right, but None
                    # routed it into UNGROUNDED signature clustering, so every
                    # novel viewpoint of picking up a log minted a fresh
                    # `discovered_N` goal + skill. The core task reward was
                    # the dominant junk producer (48/48 goal slots were
                    # `discovered_N`). A stable canonical key collapses ALL
                    # pickups into ONE grounded slot instead.
                    _best = "log_pickup"
                if _best is not None:
                    self._new_break_by_env[_e] = _best

            next_obs_t = torch.from_numpy(np.stack(next_obs_list)).to(self.device)

            if (not use_dream_actor and self.option_executor is not None
                    and not self._scouts_in_ppo):
                # scout option terminations (raw reward; no credit records)
                #
                # SKIPPED WHEN SCOUTS FEED PPO (2026-09-01). Accumulating a
                # scout option's SMDP return needs the MIXED reward, which
                # does not exist until the shaping block further down, so
                # that path calls observe_scouts there instead. Calling it in
                # both places would advance `steps_done` twice per env step
                # and terminate every scout option at half its horizon — the
                # two sites are mutually exclusive by construction rather
                # than by convention. `_scouts_in_ppo` is false on the
                # episodic path, so this body is unchanged there.
                self.option_executor.observe_scouts(
                    rewards, dones, self.total_timesteps)

            # Goal broadcaster ingests ALL streams (July 2026 pivot, part 2).
            # Discovery previously watched only the primary env, so 7/8 of
            # the fleet's experience could never mint a goal slot — with
            # first-discovery being THE bottleneck in Minecraft, that threw
            # away 8x the odds of catching the first grounded event. Each
            # wrapper keeps its own reward/obs history; the signature
            # clustering in DiscoveredAchievementGoals dedups events found
            # independently by different streams. Each env passes its STREAM
            # INDEX: per-stream episode state keeps the self-model learning
            # P(success in one episode) instead of P(any-of-N succeeds).
            _unlocks_before = (len(self.broadcaster.unlock_log)
                               if (self.broadcaster is not None and hasattr(
                                   self.broadcaster, "unlock_log")) else 0)
            if self.broadcaster is not None:
                for _i, e in enumerate(envs):
                    # grounded-effect dedup: pass WHAT broke on this stream so
                    # the same behaviour keys one slot (kills break_*_leaves x8)
                    self._broadcaster_update(
                        e, _i, effect_key=self._new_break_by_env.get(_i))

            _pt = self._phase_mark("goals", _pt)

            # ---- 3. CURIOSITY (batched) ----
            # PERSPECTIVE FROM MOTION joins the drive HERE, as a second
            # prediction-error channel feeding learning progress — NOT as a
            # new reward term. LP pays for error going DOWN, so a new channel
            # changes what the agent can make progress ON, never how much it
            # is paid for standing anywhere. And the residual is a ratio
            # against how much the frame changed at all: no movement, no
            # change, no residual, no income.
            # EVALUATION SINK. Reads info["oracle"] and nothing else writes
            # to the agent from here — see _oracle_observe.
            self._oracle_observe(step_infos, n)
            self._lt_post_step(step_infos, dones, n)
            _fr = self._flow_residual_batch(
                obs_t, next_obs_t, action_tensor, n,
                ego=self._flow_ego_batch(step_infos, n))
            self._flow_resid_now = (
                float(_fr[0]) if _fr is not None else 0.0)
            if _fr is not None:
                self.training_metrics["flow_residual"].append(
                    self._flow_resid_now)
            intrinsic = self._curiosity_reward(
                obs_t, action_tensor, next_obs_t, extra_error=_fr,
                ended=dones, invalid=[bool(restarted[i]) or bool(
                    (step_infos[i] or {}).get("env_restarted")) for i in range(n)])
            # ICM BASE SCALE (infra #13): the raw-error term is damped by
            # config so it seasons rather than dominates. Since 2026-10-05
            # paired progress is MEASUREMENT ONLY and pays nothing, so this
            # damped term (plus the itemised novelty terms) IS the base drive
            # — nothing "leads" it any more. 1.0 = old behaviour.
            intrinsic = intrinsic * float(getattr(self, "_icm_base_scale", 1.0))
            self._lt_begin(intrinsic)
            icm_metrics = self._curiosity_train(
                obs_t, action_tensor, next_obs_t
            )
            _pt = self._phase_mark("curiosity", _pt)

            # ---- 4. RSSM observe (batched) + primary-only symbolic facts ----
            # _wm_param_lock (review fix): with the async trainer these
            # forwards would otherwise read params MID-optimizer-step; the
            # torn latents flow into the permanently carried lifelong state.
            # Uncontended (sync mode) the RLock costs ~1us.
            with self._wm_param_lock, torch.no_grad():
                # SAME BODY THE TRAINER SEES. The world model is trained on
                # (frame, proprio) pairs from the buffer; encoding the live
                # frame with a zero body here would make every acting latent
                # come from a distribution the model never trained on.
                encoded_next = self.world_model.embed(
                    next_obs_t, self._wm_proprio_batch(step_infos, n))
                rssm_state, _ = self.world_model.rssm.observe_step(
                    rssm_state, action_tensor, encoded_next
                )
                latent = self.world_model.rssm.get_latent(rssm_state)
            # LIFELONG carry fix: observe_step RETURNS A NEW dict, so the local
            # rssm_state diverges from self._ll.rssm_state (which otherwise stays
            # the startup seed) — silently making the "continuous" cross-segment
            # carry a no-op AND making death_reset zero the wrong (stale) tensor.
            # Sync the LifelongState every step so the next segment resumes from
            # the evolved state and death_reset/decay mutate the LIVE object.
            # No-op off the lifelong path (self._ll is None).
            if self._ll is not None:
                self._ll.rssm_state = rssm_state
            # arch='rssm': the belief the NEXT decision will be made from, kept
            # for the segment-end GAE bootstrap. Cheap (a reference), and it
            # has to be the post-observe latent — that is the state the next
            # segment resumes at, which is exactly what V(s_last) means.
            if self.policy.arch == "rssm":
                self._bootstrap_latent = latent.detach()

            # pair fresh unlocks with the latent of the stream that earned
            # them (context provenance for skill minting / the brain graph)
            if (self.broadcaster is not None
                    and hasattr(self.broadcaster, "unlock_log")):
                for _ev in self.broadcaster.unlock_log[_unlocks_before:]:
                    _st = int(_ev.get("stream", 0))
                    if 0 <= _st < latent.shape[0]:
                        self._unlock_context[int(_ev["slot"])] = (
                            latent[_st].detach().cpu().numpy())
                    try:  # provenance frame (adapters cache last POV: cheap)
                        self._unlock_frame[int(_ev["slot"])] = (
                            envs[_st].render())
                    except Exception:
                        pass
                    # grounded name source: the block that broke on THIS
                    # stream this step is what the reward spike (= the
                    # unlock) was caused by
                    if _st in self._new_break_by_env:
                        self._unlock_block[int(_ev["slot"])] = (
                            self._new_break_by_env[_st])
                    # co-occurrence: unlock landed while (or just as) an
                    # option ran on that stream -> earned evidence that the
                    # SKILL causes the ACHIEVEMENT (fed to the prerequisite
                    # miner at episode end; same-step closes included).
                    if (not use_dream_actor
                            and self.option_executor is not None):
                        _rt = self.option_executor.runtimes[_st]
                        _sid = (_rt.skill_id if _rt.active else
                                self.option_executor
                                .last_closed_this_step[_st])
                        if _sid is not None:
                            self.option_executor.record_cooccurrence(
                                _sid, int(_ev["slot"]), _st,
                                self.total_timesteps)

            # Gate: per-dimension fact extraction is meaningless (and
            # unboundedly expensive) on raw pixels / high-dim structured obs.
            if not self._skip_perdim_symbolic:
                new_facts_count += self._extract_and_store_facts(
                    obs_list[0], env_actions[0], next_obs_list[0], rewards[0]
                )
            # sd heads + discretizer are trained on the async trainer thread;
            # lock the reads/writes here (review fix — write-write race on the
            # discretizer, torn sd-head reads)
            #
            # ---- GATED BY THE SAME FLAG AS THE BLOCK ABOVE (2026-09-01) ----
            # MEASURE FIRST, AND THE FIRST MEASUREMENT WAS WRONG. This was
            # flagged as "Welford over a 49,152-dim frame every step"; it is
            # not. `_skip_perdim_symbolic` already swaps in
            # `_NullSymbolicDecoder`, whose update_discretizer/extract_facts
            # are `pass`/`[]`, so on a pixel env the decoder arithmetic here
            # has always been free. Recording the correction rather than the
            # theory, because §5 is explicit that this project keeps paying
            # for plausible stories.
            #
            # WHAT IS ACTUALLY BOUGHT. Two things that are NOT free and have
            # no reachable value on a pixel env:
            #   * `_wm_param_lock` is ACQUIRED EVERY STEP. The async WM
            #     trainer holds it per gradient step and runs 384 back to
            #     back, so the acting thread can block a full optimizer step
            #     here — for a decoder that returns [].
            #   * `glue.augment_latent` is a GNN-gate forward whose only
            #     consumer is `extract_facts`, which on this path returns [].
            # So the gate removes a per-step lock handshake and a wasted
            # forward, and changes nothing about what is learned.
            #
            # AND ON A CADENCE WHEN IT IS ON: the KG consumers of `sd_facts`
            # read at `perception_interval`/`goal_interval` (50/25), so paying
            # for a fresh extraction every step bought nothing. `_sd_every`
            # keeps step 0 of every window so a cold start still primes the
            # discretizer and the heads on the first frame they see.
            _sd_now = (not self._skip_perdim_symbolic
                       and (self.total_timesteps % self._sd_every == 0))
            if _sd_now:
                with self._wm_param_lock:
                    with torch.no_grad():
                        augmented_primary = self.glue.augment_latent(
                            latent[0:1])
                    sd_facts = self.symbolic_decoder.extract_facts(
                        augmented_primary.squeeze(0),
                        timestep=self.total_timesteps
                    )
                    self.symbolic_decoder.update_discretizer(
                        next_obs_list[0])
                for fact in sd_facts:
                    if self.knowledge_graph.add_fact(fact):
                        symbolic_decoder_facts_count += 1

            if episode_length % 50 == 0 and not self._skip_perdim_symbolic:
                scene = self.fact_extractor.extract_scene_graph(
                    next_obs_list[0], timestep=self.total_timesteps
                )
                self.knowledge_graph.save_scene_snapshot(scene)

            _pt = self._phase_mark("world", _pt)

            # ---- 5. STORE EXPERIENCE (one stream per env) ----
            # Primary stream (0) gets vision-scaffold shaping folded into its
            # extrinsic (waking only); scouts store the raw env reward. The
            # WM reward head thus learns the true reward from 15 streams and
            # the shaped one from the primary — matching the single-env path.
            prim_extrinsic = rewards[0]
            self._lt_mark("shaping", intrinsic, prim_extrinsic, "env")
            _frame = None
            if not use_dream_actor and self.symbolizer is not None:
                try:
                    _frame = envs[0].render()   # symbolizer's VLM label frame
                except Exception:
                    _frame = None
            # Curiosity-ranked magnet: reads the grounding head's per-object
            # probs on the primary latent + this step's learning progress
            # (intrinsic[0]) and steers toward the most-curious in-view object.
            # FELT REACH, every step (cheap: one small MLP forward on a
            # latent the loop already has). Computed here so the value is
            # fresh for the NEXT act(), matching how proprio is refreshed.
            if not use_dream_actor and self.symbolizer is not None:
                self._reach_now = self._reach_sense(latent[0:1])
            # PERSPECTIVE FROM MOTION, same cadence and same cost profile as
            # reach. Deliberately NOT gated on the symbolizer: this is
            # geometry read off the world model, and it exists in an env that
            # never grounds a single symbol.
            if getattr(self.world_model, "flow_enabled", False):
                if not use_dream_actor:
                    self._flow_now = self._flow_senses(
                        latent[0:1], self._flow_resid_now)
                    self._spatial_step(
                        latent[0:1], step_infos,
                        # DID THE AGENT ASK TO MOVE. Walkability is learned
                        # from the GAP between the command and the outcome,
                        # so the command half has to come from the action
                        # actually issued to the primary body.
                        self._is_move_action(int(env_actions[0])))
                # THE MEMORY IS REFRESHED EVEN WHILE DREAMING, deliberately.
                # The senses above are for ACTING and a dreaming agent is not
                # acting, but the cache is a fact about which frame the next
                # residual will be measured against. Leaving it stale through
                # a dream block would make the first waking residual compare
                # the current frame to one from before the dream — a splice
                # across time that reads as enormous unexplained motion.
                # THE ONE-STEP MEMORY, refreshed AFTER it has been consumed.
                # `valid` marks the streams whose next frame will still belong
                # to THIS world: a stream that just ended or was rebuilt has a
                # previous frame from a different world entirely, and warping
                # across that splice would report a near-total residual and
                # read as the most interesting event in the run.
                self._flow_prev_latent = latent.detach()
                self._flow_prev_valid = torch.tensor(
                    [not (bool(dones[i]) or bool(restarted[i]))
                     for i in range(n)],
                    dtype=torch.bool, device=self.device)
            # THE ORACLE'S FRAME RESETS WITH THE WORLD. The dead reckoner's
            # origin is the episode start; truth must be re-anchored at the
            # same instant or the drift measurement becomes a constant offset.
            if bool(dones[0]) or bool(restarted[0]):
                self._oracle_reset()
                self._spatial_reset()
            # MASTERY HABITUATION: a break of a long-mastered type damps this
            # step's whole intrinsic (see _habituation_factor) — applied
            # BEFORE the magnet so its LP attribution habituates too.
            if not use_dream_actor:
                _hab = self._habituation_factor(prim_info)
                if _hab < 1.0:
                    intrinsic[0] = intrinsic[0] * _hab
                    self._lt_mark("damp_habituation", intrinsic, prim_extrinsic, None, "habituation", _hab)
            if not use_dream_actor and self.vision_scaffold is not None:
                _sr = self._magnet_step_shaping(
                    env_actions[0], float(intrinsic[0].item()), latent[0:1])
                # OCCLUSION APPLIES TO THIS CHANNEL TOO (2026-08-17). The
                # gui_open guard further down zeroes `intrinsic` — but the
                # magnet's shaping is added to prim_EXTRINSIC, which that
                # guard never touches, so it kept paying through a menu.
                # MEASURED LIVE: SkyBot right-clicked a wandering villager,
                # opened its trade screen, and sat there for 9,125
                # CONSECUTIVE steps (gui 64% of the segment, position
                # frozen) while magnet_seek paid 77% of ALL income. This is
                # the 2026-08-02 occlusion farm exactly, one channel over:
                # inside a GUI the agent cannot move, look or swing, so
                # anything that pays there is paying for blindness.
                # Still CALLED while occluded — heartbeats, the fovea latent
                # refresh and signal-health observation must not stall —
                # just not PAID.
                _gui_now2 = bool((step_infos[0] or {}).get("gui_open"))
                if _gui_now2:
                    # seek/centring are POTENTIALS: left holding a menu
                    # frame's phi, the step the menu closes would collect a
                    # windfall for closing it. Re-adopt with no delta,
                    # exactly as reset() does at an episode boundary.
                    self.vision_scaffold._phi_prev = None
                    self.vision_scaffold._seek_prob_prev = None
                elif _sr:
                    # ---- FARM DAMPING (2026-09-04) -----------------------
                    # infra/farm named a live farm — 2,213 laps in one cell
                    # at +0.026/step — and NOTHING consumed it. Detection
                    # without response is how 77% of income once went to a
                    # villager menu and 96% of drive to the sky.
                    # SHAPING ONLY. rewards[0] is real environment income and
                    # is never damped: an agent that actually breaks a block
                    # in a cell it happens to have circled must still be paid
                    # in full, or this becomes a new way to be wrong about
                    # the scoreboard (§9).
                    # Not a latch — the multiplier is a pure function of lap
                    # count and income EMA, both of which decay once the
                    # agent stays away longer than revisit_horizon. See
                    # BehaviouralLoopDetector.loop_damp.
                    _damp = 1.0
                    if self.infra is not None:
                        _damp = float(getattr(
                            self.infra, "last_loop_damp", 1.0) or 1.0)
                    _sr = _sr * _damp
                    prim_extrinsic = rewards[0] + _sr
                    self._lt_mark("magnet", intrinsic, prim_extrinsic)
                # ---- APPROACH A REMEMBERED PLACE (2026-09-04) -----------
                # infra/episodic has recorded `sighting:tree_visible @(x,z)`
                # for the whole project and driven NOTHING, so the agent
                # could only pursue what was ON SCREEN — look away and the
                # tree stopped existing. See InfraStack._approach_step.
                # PLAIN DIFFERENCE (§4.3): pays w*(d_prev - d_now), which is
                # exactly 0.0 while stationary. The telescoping form would
                # pay a wage for holding still (`Gaze level: +0.00014/step`).
                # prim_EXTRINSIC, before the gui zeroing line (§4.4), and
                # not paid at all while occluded — inside a menu the agent
                # cannot walk, so anything paid there is paying for
                # blindness (the villager incident, 77% of income).
                if not _gui_now2 and self.infra is not None:
                    _ar = float(getattr(
                        self.infra, "last_approach_reward", 0.0) or 0.0)
                    if _ar:
                        prim_extrinsic = prim_extrinsic + _ar
                        self._lt_mark("approach", intrinsic, prim_extrinsic)
            # ---- GETTING OUT MUST PAY -----------------------------------
            # Zeroing income inside a menu removes the FARM but leaves
            # no gradient toward the exit — and it did something worse:
            # because pressing `inventory` OPENS a screen that pays 0,
            # the policy trained that action to ~zero probability
            # (measured: 0 presses in an entire run at 92%-of-max
            # entropy). So when a wandering villager's trade screen
            # opened via `use`, the agent had already unlearned the only
            # button that closes one, and sat there 10,149+ consecutive
            # steps. Guard-becomes-latch, again.
            # Plain-difference state cost on dwell: Phi = -min(1, run/N).
            # Sitting accrues the cost once; leaving collects it back;
            # an open/close cycle nets 0, so this cannot be farmed in
            # either direction. Paid into prim_EXTRINSIC on purpose —
            # the intrinsic channel is zeroed while a GUI is open, so a
            # cost placed there would be erased by the very guard it is
            # meant to complement.
            # NOT INSIDE THE VISION-SCAFFOLD BLOCK (2026-10-03, see
            # docs/foundation/BASELINE_AUDIT.md §3). It used to be, and the
            # live config runs `llm.vision.enabled: false` — so the scaffold
            # was None, stream 0 (the ONLY stream PPO trains on) paid NO
            # dwell cost, every scout (_scout_mixed_reward) did, and the
            # yaml's "a menu now pays strictly <= 0" rested on a term that
            # never ran. The gui flag is read from env info here — the same
            # source as the scout path and the intrinsic-zeroing guard — so
            # this runs exactly once per waking step, scaffold or not.
            if not use_dream_actor:
                prim_extrinsic += self._gui_reward(0, step_infos[0])
                self._lt_mark("gui_dwell", intrinsic, prim_extrinsic)
            # general infra: event monitors + empowerment (shared helper)
            if not use_dream_actor:
                _ei = self._infra_step(
                    prim_info, env_actions[0], rssm_state,
                    float(intrinsic[0].item()) + float(prim_extrinsic),
                    float(rewards[0]))
                if _ei:
                    intrinsic[0] = intrinsic[0] + _ei
                    self._lt_mark("empowerment", intrinsic, prim_extrinsic)
                # ---- FARM DAMP ON THE CHANNEL THAT PAYS (2026-10-04) -----
                # last_loop_damp used to multiply only the magnet's `_sr`,
                # which is always 0 on the live config (llm.vision.enabled
                # false) — so a detected revisit loop kept its full
                # INTRINSIC income. Read AFTER _infra_step: on_step has just
                # scored THIS step's cell, and it was fed the UNDAMPED
                # income, so the damp cannot release itself by its own
                # effect. BEFORE the gui zeroing at the mix, which still
                # yields exactly 0 in a menu. Not a latch: see
                # _loop_damp_factor (leave the cell -> 1.0 again).
                _ldi = self._loop_damp_factor()
                if _ldi < 1.0:
                    intrinsic[0] = intrinsic[0] * _ldi
                    self._lt_mark("damp_farm", intrinsic, prim_extrinsic, None, "farm_damp", _ldi)

            # ---- VLM SYMBOLIC GROUNDING (primary stream, waking only) ----
            # The VLM names what the agent is looking at; a head learns to
            # predict those names from the agent's OWN latent (borrowed ->
            # owned). Perceptual facts are emitted only when a fresh label
            # lands, so KG traffic anneals with the VLM itself. Causal facts
            # come from real break events, never from the VLM's word.
            if not use_dream_actor and self.symbolizer is not None:
                _lat0 = latent[0:1].detach()
                # last frame, kept for the unstuck advisor: at stuck L3 the
                # VLM is shown what the agent currently sees (a reference
                # assignment per step, only read at L3 segments)
                self._advisor_frame = _frame
                self.symbolizer.maybe_label(_frame, _lat0, self._symbol_clock)
                # SCORE THE HEAD AGAINST TELEMETRY (2026-09-01). pitch,
                # gui_open and mainhand settle four predicates exactly, every
                # step, and were never compared — see observe_truth. This is
                # the half of the verifier that does not need the VLM.
                self.symbolizer.observe_truth(
                    (prim_info.get("world") or {}), _lat0)
                _labels = self.symbolizer.collect()
                if _labels is not None and self.infra is not None:
                    self.infra.beat("vlm_label", self.total_timesteps)
                # LAYER 3: route a VLM-PROPOSED category to the instinct. The
                # proposal is attention only — it goes to the magnet's
                # probation set, never to the head and never to the KG. If
                # attending to it yields no learning progress it is evicted.
                if (_labels and self.vision_scaffold is not None
                        and _labels.get("__novel__")):
                    try:
                        self.vision_scaffold.propose_category(
                            str(_labels["__novel__"]), self.total_timesteps)
                    except Exception as _e:     # never kill a run over instinct
                        logger.warning("category proposal failed: %s", _e)
                if _labels is not None:
                    for _sf in self.symbolizer.perceptual_facts(_lat0):
                        # refresh=True: this channel may REVISE its own
                        # confidence downward. Without it add_fact ratchets
                        # to a high-water mark and a discredited predicate
                        # keeps its most-confident moment forever.
                        if self.knowledge_graph.add_fact(
                                _to_fact(_sf, "vlm_grounded",
                                         self.total_timesteps),
                                refresh=True):
                            symbolizer_facts_count += 1
                # ground truth: which block types were newly broken this step
                _ach = prim_info.get("achievements", {}) or {}
                _chopping = (self.is_discrete
                             and int(env_actions[0]) in self._chop_action_set)
                _broke_now_p = []
                for _k, _v in _ach.items():
                    if not _k.startswith("mine_"):
                        continue
                    _btype = _k[5:]
                    if int(_v) > int(self._prev_mine.get(_btype, 0)):
                        self._prev_mine[_btype] = int(_v)
                        _broke_now_p.append(_btype)
                        for _cf in self.symbolizer.observe_event(
                                "block_break", _btype,
                                was_chopping=_chopping,
                                now=self._symbol_clock):
                            if self.knowledge_graph.add_fact(_to_fact(
                                    _cf, "grounded_event",
                                    self.total_timesteps)):
                                symbolizer_facts_count += 1
                # negative evidence — same contract as the lifelong body
                # (see the comment there): a full break-worth of held attack
                # with no break docks the claimed affordance predicates.
                if self._neg_evidence_ticks > 0:
                    _arun = float(prim_info.get("attack_run", 0.0) or 0.0)
                    if _broke_now_p or _arun <= 0.0:
                        self._neg_evt_scored = False
                    elif (_arun >= self._neg_evidence_ticks
                          and not getattr(self, "_neg_evt_scored", False)):
                        self._neg_evt_scored = True
                        self._neg_evt_total = getattr(
                            self, "_neg_evt_total", 0) + \
                            self.symbolizer.observe_negative_event(
                                now=self._symbol_clock)
                # ---- RETRACTION: the other half of the gates ----
                # min_labels/reliability_floor only stop NEW facts. A
                # predicate the world has discredited must also stop feeding
                # its triple to the graph embedding. Placed after the break
                # events above so a collapse caused by THIS step retracts
                # immediately. stale_facts() is edge-triggered, so this is a
                # 16-entry scan per step and at most one KG write per
                # collapse; source-scoped, so it can only delete facts this
                # channel minted.
                for _stale in self.symbolizer.stale_facts():
                    try:
                        if self.knowledge_graph.remove_fact(
                                *_stale, source="vlm_grounded"):
                            self._symbolizer_retractions += 1
                            logger.info(
                                "Symbol grounding: retracted %s "
                                "(reliability fell below floor)", _stale)
                    except Exception as _e:   # never kill a multi-day run
                        logger.warning(
                            "KG retraction failed for %s: %s", _stale, _e)
                self._symbol_clock += 1

            _pt = self._phase_mark("vlm", _pt)

            if self.brain_emitter is not None:
                self._brain_clock += 1
                if self._brain_clock % self._brain_interval == 0:
                    self.brain_emitter.emit(self.total_episodes,
                                            self.total_timesteps)
            for e_i in range(n):
                self.replay_buffer.add(
                    obs_list[e_i], env_actions[e_i],
                    prim_extrinsic if e_i == 0 else rewards[e_i],
                    dones[e_i], stream=e_i,
                    # WHICH KIND OF `done` THIS IS (2026-09-01). Both the
                    # loop-level step timeout (`restarted`) and the
                    # adapter-caught crash (`env_restarted`) rebuild the
                    # CLIENT, so the frames either side are different worlds.
                    # The advantage trace wants them severed exactly like a
                    # terminal — which is why `dones` folds them together —
                    # but the world model must not learn the splice as
                    # dynamics, and must certainly not OVERSAMPLE it.
                    restart=bool(restarted[e_i]) or bool(
                        (step_infos[e_i] or {}).get("env_restarted")),
                    # THE BODY AS IT WAS WHEN obs_list[e_i] WAS SEEN, not as
                    # it is now. `step_infos` describes the state AFTER this
                    # step, so writing it here would pair o_t with proprio
                    # from t+1 and hand the world model a one-step lookahead
                    # on its own motion — the same off-by-one the causal
                    # action alignment exists to prevent.
                    proprio=(self._wm_proprio_prev[e_i]
                             if (self._wm_proprio_prev is not None
                                 and e_i < len(self._wm_proprio_prev))
                             else None),
                )
            # Advance the one-step body memory AFTER the write.
            if self._wm_proprio_prev is not None:
                for e_i in range(n):
                    _si = step_infos[e_i] or {}
                    self._wm_proprio_prev[e_i] = (
                        _si.get("sensors")
                        if _si.get("sensors") is not None
                        else _si.get("proprio"))
            # ---- SUB-PHASING `store` (2026-09-20) --------------------
            # The measured profile put 287 ms/step (21.6%) in `store`, which
            # is not explicable from reading this window: it writes one
            # replay row per env and appends a few numpy arrays to the PPO
            # rollout lists. Rather than guess (CLAUDE.md 5), split the
            # bucket and let the next profile name the cost.
            _pt = self._phase_mark("store_buf", _pt)

            # Primary stream feeds the on-policy PPO rollout (waking only).
            if not use_dream_actor and primary_pol is not None:
                # ---- OCCLUSION SUPPRESSION MUST PRECEDE THE MIX ----------
                # (fix 2026-08-02, found live on the server.) The gui_open
                # zeroing sat ~40 lines BELOW this call — AFTER `mixed` was
                # computed and AFTER the transition was stored — so it only
                # ever changed the logged `episode_intrinsic` and NEVER
                # reached the PPO reward. The guard was inert from the day it
                # was written, in BOTH loop paths.
                #
                # Measured live: the inventory screen paid +0.0410/step vs
                # +0.0196 for the world (2.09x), and the agent sat in the menu
                # 89% of a segment with an unbroken 1491-step dwell —
                # motionless, because inside a GUI it cannot move, look or
                # swing. Curiosity was paying for the act of occluding the
                # camera, which is exactly what the note below forbids.
                #
                # The menu stays fully available and crafting still earns its
                # extrinsic tier; only payment for the occlusion is removed.
                if bool((step_infos[0] or {}).get("gui_open")):
                    intrinsic[0] = intrinsic[0] * 0.0
                    self._lt_mark("gui_zero", intrinsic, prim_extrinsic, None, "gui_zero", 0.0)
                mixed = self.reward_mixer.mix(
                    float(intrinsic[0].item()), prim_extrinsic)
                self._lt_commit(mixed, intrinsic, prim_extrinsic)
                # running "typical surprise" baseline — the denominator of
                # mastery's WM-fidelity ratio. Updated EVERY primary step
                # (not just during options), or the baseline would be biased
                # toward exactly the moments options select for.
                #
                # THE RAW FORWARD-MODEL ERROR, not `intrinsic` (review HIGH,
                # flagged independently by four reviewers). `intrinsic` is
                # the SHAPED curiosity reward: mean-centred, clipped, and
                # carrying learning-progress / imagination / coverage
                # bonuses. Feeding that to "can the world model predict this
                # skill" measured a different variable — a skill could look
                # unpredictable purely because it was novel, or predictable
                # because its LP had flattened. `last_pred_error` is the
                # pre-normalization MSE the ICM already computes.
                _err_now = float(getattr(self.curiosity, "last_pred_error",
                                         float(intrinsic[0].item())))
                # ---- OCCLUSION: NO WORLD CHANGE, NO WORLD CURIOSITY ------
                # MEASURED (2026-07-27): the inventory screen paid the agent
                # +0.18 -> +0.30 intrinsic per step against +0.09 for the
                # world — a 2.0x -> 3.1x advantage that WIDENED over the run.
                # The agent was not failing to learn; it was optimal under
                # the signal it was given, and spent ~50% of every episode
                # in a menu.
                #
                # This is a CATEGORY ERROR, not a preference. The ICM scores
                # prediction error on the OBSERVATION. When a menu covers the
                # screen the observation changes hugely while the WORLD does
                # not change at all, so curiosity — which exists to reward
                # learning about world dynamics — is paying for an overlay
                # drawn over the world. It is the same argument already
                # accepted for the attack counter: a swing into a menu never
                # reached the world, so counting it was a measurement error.
                # Applying it to actions but not to perception was
                # inconsistent.
                #
                # NOT a rule about what the agent should want: the menu stays
                # fully available, and CRAFTING in it still pays the +5.0
                # extrinsic first-craft tier. What is removed is payment for
                # the act of occluding the camera.
                #
                # (2026-08-02) The zeroing now happens BEFORE the mix, above.
                # By the time control reaches here intrinsic[0] is already 0 on
                # a GUI frame, so re-multiplying by 0.0 was a no-op left behind
                # by that fix. Removed; the reasoning above is the load-bearing
                # part and stays.
                # ---- IS THE MENU A CURIOSITY FARM? (2026-07-27) ----------
                # The agent spends ~50% of every episode in the inventory
                # screen. I have been calling that "a learning result" with
                # no evidence. It is a MEASURABLE question: split the
                # intrinsic reward by whether the world was covered by a
                # menu at the time. If menu frames pay MORE, the agent is
                # behaving optimally under the signal it is given and the
                # signal is what needs looking at -- not the behaviour.
                try:
                    _gui_now = bool((step_infos[0] or {}).get("gui_open"))
                    # THE RAW forward-model surprise, NOT the post-
                    # suppression intrinsic. Measuring the value we just
                    # zeroed makes the readout tautological (`in-menu=0.0000`
                    # by construction) and destroys the falsification test
                    # this line exists to provide. What we want to know is
                    # whether the menu WOULD still pay better — i.e. whether
                    # the underlying category error is still there — which
                    # only the pre-suppression error can answer.
                    _iv = _err_now
                    if not hasattr(self, "_gui_intr"):
                        self._gui_intr = [0.0, 0, 0.0, 0]   # sum,n | sum,n
                    if _gui_now:
                        self._gui_intr[0] += _iv
                        self._gui_intr[1] += 1
                    else:
                        self._gui_intr[2] += _iv
                        self._gui_intr[3] += 1
                except Exception:
                    pass
                self._wm_err_ema = (
                    _err_now if not hasattr(self, "_wm_err_ema")
                    else 0.999 * self._wm_err_ema + 0.001 * _err_now)
                if self.option_executor is not None:
                    _rt0 = self.option_executor.runtimes[0]
                    if _rt0.active:
                        # inside an option: accumulate; store ONE SMDP
                        # decision when it closes
                        _closed = self.option_executor.observe_primary(
                            float(rewards[0]), bool(dones[0]), mixed,
                            self.total_timesteps, wm_err=_err_now)
                        if _closed is not None:
                            self.policy.store_transition(
                                _closed["obs"], _closed["meta_action"],
                                _closed["reward"], dones[0],
                                _closed["log_prob"], _closed["value"],
                                knowledge=_closed["kv"],
                                tau=_closed["tau"],
                                action_mask=_closed["mask"],
                                proprio=_closed.get("proprio"),
                                feats=_closed.get("feats"))
                            self.skill_bank.record_invocation(
                                _closed["skill_id"], _closed["outcome"],
                                _closed["reward"])
                            self._competence_from_option(_closed)
                            # WM fidelity (task #39): this invocation's mean
                            # surprise vs the agent's typical surprise. The
                            # asked/success half is recorded by the executor
                            # itself at every frame close (incl. scouts and
                            # nested frames) — only the fidelity ratio needs
                            # the loop's global baseline.
                            _wem = _closed.get("wm_err_mean")
                            if (_wem is not None
                                    and getattr(self, "_wm_err_ema", 0) > 1e-9):
                                self.skill_bank.record_wm_fidelity(
                                    _closed["skill_id"],
                                    float(_wem) / float(self._wm_err_ema))
                    else:
                        _pp = self.option_executor._primary_primitive
                        if _pp is not None:
                            self.policy.store_transition(
                                obs_list[0], _pp["action"], mixed,
                                dones[0], _pp["log_prob"], _pp["value"],
                                knowledge=primary_kv, tau=1,
                                action_mask=_pp["mask"],
                                proprio=_pp.get("proprio"),
                                feats=_pp.get("feats"))
                else:
                    self.policy.store_transition(
                        obs_list[0], env_actions[0], mixed, dones[0],
                        primary_pol["log_prob"], primary_pol["value"],
                        knowledge=primary_kv, feats=self._feats_row(0))

            _pt = self._phase_mark("store_ppo", _pt)
            # FOUNDATION SHADOW: action t, obs t+1 and the evidence linking
            # them, AFTER reward assembly (final values) and BEFORE the
            # advance below reassigns obs/resets envs. Episodic: a primary
            # done re-resets the WHOLE fleet at the next call, so every
            # stream's episode ends here.
            _sh_fleet_reset = bool(dones[0])
            if self._shadow is not None:
                self._shadow.after_step(
                    step_infos, env_actions, rewards, dones, restarted,
                    _sh_ends, _t_env0, prim_extrinsic, intrinsic,
                    _sh_fleet_reset, use_dream_actor, observations=next_obs_list)
                _pt = self._phase_mark("shadow", _pt)

            # ---- 6. ADVANCE / AUTORESET ----
            primary_done = dones[0]
            for e_i in range(n):
                if dones[e_i]:
                    if e_i == 0 or primary_done:
                        # Primary done ends the episode; every env is then
                        # concurrently re-reset at the next episode's start —
                        # resetting non-primaries here too was a SERIAL,
                        # REDUNDANT double reset (on MineRL, minutes per
                        # episode boundary: fixed step limits make all envs
                        # finish together). Just carry the obs; it's discarded.
                        obs_list[e_i] = next_obs_list[e_i]
                    else:
                        o, _ = envs[e_i].reset()
                        obs_list[e_i] = np.asarray(o, dtype=np.float32)
                        # Reset that env's latent row to the initial state.
                        rssm_state["h"][e_i] = 0.0
                        rssm_state["z"][e_i] = 0.0
                        # That stream just started a NEW episode: clear its
                        # achieved mask so unlocks re-count and its baseline
                        # re-establishes (else a mid-episode autoreset makes
                        # the stream's row a union over several episodes).
                        if (self.broadcaster is not None
                                and hasattr(self.broadcaster, "clear_stream")):
                            self.broadcaster.clear_stream(e_i)
                        if (not use_dream_actor
                                and self.option_executor is not None):
                            self.option_executor.clear(
                                e_i, self.total_timesteps)
                else:
                    obs_list[e_i] = next_obs_list[e_i]

            _pt = self._phase_mark("store_reset", _pt)
            # A2: hand this step's encoder output to the next act(). MUST be
            # after the advance loop above — validity is decided by whether
            # obs_list[e] IS next_obs_list[e] (same object = the env kept
            # running), which is only true once that loop has assigned.
            self._set_enc_carry(encoded_next, obs_list, next_obs_list)

            episode_reward += rewards[0]
            episode_intrinsic += float(intrinsic[0].item())
            episode_length += 1
            self.total_timesteps += n  # all N transitions are real experience
            _pt = self._phase_mark("store", _pt)

        # On-policy PPO update from the primary stream's rollout (waking only;
        # single-stream GAE, exactly as the serial path does at episode end).
        policy_loss = 0.0
        if (not use_dream_actor
                and self.policy.should_update(self.policy_update_steps)):
            _pt_ppo = time.perf_counter()
            pm = self.policy.train_step(n_epochs=self.config.get(
                "policy", {}).get("n_epochs", 10))
            policy_loss = pm.get("policy_loss", 0.0)
            # PPO already computes entropy; it was returned and then dropped,
            # so "has the policy collapsed onto one action" was unanswerable
            # from the logs — the exact question a turn-dominated action
            # histogram raises.
            if "entropy" in pm:
                self.training_metrics.setdefault(
                    "policy_entropy", deque(maxlen=100)).append(
                        float(pm["entropy"]))
            self._phase_mark("ppo", _pt_ppo)

        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "intrinsic_reward": episode_intrinsic / max(1, episode_length),
            "policy_loss": policy_loss,
            "curiosity_loss": icm_metrics.get("icm_total", 0),
            "new_facts": new_facts_count,
            "symbolic_decoder_facts": symbolic_decoder_facts_count,
        }

    def _start_stream(self) -> None:
        """ONE-TIME continuous-stream setup: the ONLY place the full reset
        block (env reset, RSSM init, broadcaster.reset, scaffold reset) runs
        in lifelong mode. Copies _run_episode_parallel's reset block, then
        seeds LifelongState + carries obs on self. The episodic function is
        left untouched (duplication is the deliberate price of not editing a
        live-critical path)."""
        from developmental_ai.core.lifelong_state import LifelongState
        use_dream_actor = False
        self._ensure_parallel_envs()
        n = self._num_envs
        envs = self._parallel_envs

        # Keep every env at the primary env's curriculum difficulty.
        if self.curriculum is not None:
            for c in self._parallel_curricula:
                if c is not None:
                    c.current_difficulty = self.curriculum.current_difficulty

        # Reset all envs CONCURRENTLY (July 2026, Minecraft-parallel fix):
        # env.reset()/step() on real-game envs (MineRL) block on a Java
        # subprocess for 70ms-90s with the GIL released, so serial resets
        # would cost N x 90s at every episode boundary. Fast envs are
        # unaffected (thread overhead is microseconds).
        self._curiosity_boundary()
        obs_list = self._parallel_reset(envs)
        self.symbolic_decoder.update_discretizer(obs_list[0])

        # Goal broadcaster (July 2026): the parallel path predates the goal
        # channel — without these hooks, goal targeting / discovery / skill
        # minting silently die in parallel mode. Primary-stream only (an
        # accepted degradation: discovery sees 1/N of the experience).
        if self.broadcaster is not None:
            if hasattr(self.broadcaster, "set_num_streams"):
                self.broadcaster.set_num_streams(len(envs))
            self.broadcaster.reset()
            for _i, e in enumerate(envs):
                self._broadcaster_update(e, _i)
        # Clear the vision scaffold's per-episode belief (waking parallel path
        # shapes the primary stream, same as the single-env path).
        if not use_dream_actor and self.vision_scaffold is not None:
            self.vision_scaffold.reset()
        if self.infra is not None:
            # Re-adopt the approach reference at the boundary: carrying the
            # old distance across a world change would pay (or charge) for a
            # move the agent never made. Same discipline the scaffold uses.
            self.infra.approach_reset()
        if self.symbolizer is not None:
            self._prev_mine.clear()   # counts restart at 0 in a fresh world
        # per-env break high-water also restarts (new world every reset)
        self._mine_by_env.clear()
        self._new_break_by_env.clear()
        if not use_dream_actor and self.option_executor is not None:
            # close any dangling options (telemetry only) and bind newly
            # minted skills into EMPTY slots — bound slots stay frozen.
            self.option_executor.clear_all(self.total_timesteps)
            self.option_executor.bank.refresh_slots()

        # Batched RSSM state, initialised exactly like the single-env path.
        rssm_state = self.world_model.rssm.initial_state(n, self.device)
        with torch.no_grad():
            obs_t = torch.from_numpy(np.stack(obs_list)).to(self.device)
            encoded = self.world_model.embed(obs_t)
            zero_act = torch.zeros(n, self.action_dim, device=self.device)
            rssm_state, _ = self.world_model.rssm.observe_step(
                rssm_state, zero_act, encoded
            )
        self._ll = self._ll_ctrl.make_state(
            rssm_state, broadcaster=self.broadcaster,
            scaffold=self.vision_scaffold)
        self._stream_obs = obs_list

    def _collect_segment(self) -> Dict[str, float]:
        """CONTINUOUS waking collection (lifelong mode). Steps the unbroken
        world for `lifelong.segment_len` steps, carrying RSSM/mask/belief
        across segments with a soft decay (self._ll.decay), hard-zeroing a
        stream ONLY on a true terminal. Never calls env.reset except on death.
        Duplicates _run_episode_parallel's step body verbatim except the two
        marked transforms, so the episodic path stays byte-identical."""
        ll = self.config.get("lifelong", {})
        segment_len = int(ll.get("segment_len", self.policy_update_steps))
        wm_train_every = int(ll.get("wm_train_every", 250))
        goal_horizon = int(ll.get("goal_horizon", 8000))
        n = self._num_envs
        if not self._stream_started:
            self._stream_started = True
            self._start_stream()
        envs = self._parallel_envs
        rssm_state = self._ll.rssm_state          # carried; observe_step mutates
        obs_list = self._stream_obs               # carried between segments
        use_dream_actor = False

        episode_reward = 0.0
        episode_intrinsic = 0.0
        episode_length = 0
        new_facts_count = 0
        symbolic_decoder_facts_count = 0
        symbolizer_facts_count = 0
        icm_metrics: Dict[str, float] = {}
        primary_kv = None
        primary_pol = None

        for _seg_i in range(segment_len):
            _pt = time.perf_counter()          # phase clock (see _phase_mark)
            obs_t = torch.from_numpy(np.stack(obs_list)).to(self.device)

            # ---- 1. SELECT ACTIONS ----
            if use_dream_actor:
                # Latent-space dream actor, batched across all N envs.
                with torch.no_grad():
                    latent = self.world_model.rssm.get_latent(rssm_state)
                    dist = self.dream_actor.get_action_dist(latent)
                    sampled = dist.sample()
                    if self.is_discrete:
                        action_idx = sampled.long().view(-1)
                        action_tensor = torch.nn.functional.one_hot(
                            action_idx, self.action_dim).float()
                        env_actions = [int(a) for a in action_idx.cpu().numpy()]
                    else:
                        action_tensor = sampled.view(n, self.action_dim).float()
                        env_actions = [a for a in action_tensor.cpu().numpy()]
            else:
                # WAKING: real PPO policy per env. Primary stream keeps its
                # log_prob/value for the on-policy update; scouts just act.
                primary_kv = self._current_knowledge_feature()
                # THE BELIEF THIS DECISION IS MADE FROM, hoisted ABOVE the
                # options branch (2026-09-01). arch='rssm' needs it on BOTH
                # paths — with options disabled the loop stores scout and
                # primary rows from the `else` branch below, and reading a
                # latent set only inside the options branch would have handed
                # the update last step's belief, silently.
                if self.policy.arch == "rssm":
                    with torch.no_grad():
                        self._act_latent = self.world_model.rssm.get_latent(
                            rssm_state).detach()
                else:
                    self._act_latent = None
                if self.option_executor is not None:
                    # options path: executor resolves primitives per env,
                    # opening/continuing skill invocations as sampled. Its
                    # decision records replace primary_pol for storage;
                    # keep primary_pol non-None so the dream-phase guard
                    # below still distinguishes waking from dreaming.
                    # THE BELIEF THE DECISION IS MADE FROM. Computed once,
                    # before the action, and reused by both consumers: the
                    # precondition gate and (arch='rssm') the policy input
                    # itself. This is the state the agent is actually in when
                    # it chooses — computing it after the step would score the
                    # initiation set against a world the choice already
                    # changed.
                    _preds = None
                    if self.option_executor.gate_by_preconditions:
                        with torch.no_grad():
                            _preds = self._predicates_batch(
                                self._act_latent
                                if self._act_latent is not None
                                else self.world_model.rssm.get_latent(
                                    rssm_state))
                    # live per-skill competence for the warmup gate:
                    # broadcaster slot N -> skill ach_NN_.. -> competence[N]
                    _comp = None
                    if (self.option_executor.competence_floor > 0.0
                            and self.broadcaster is not None
                            and hasattr(self.broadcaster, "competence")):
                        _cv = self.broadcaster.competence.predict_all()
                        _nm = getattr(self.broadcaster, "slot_names", [])
                        # Keyed by BOTH the reconstructed name AND the raw slot
                        # index. A skill id freezes the slot name it was minted
                        # under, so once a slot is re-keyed (grounded effect <->
                        # discovered_N) only the index still matches; the bank's
                        # gate falls back to it via options._slot_index_of.
                        _comp = {f"ach_{i:02d}_{n}"[:64]: float(_cv[i])
                                 for i, n in enumerate(_nm) if i < len(_cv)}
                        _comp.update({i: float(_cv[i])
                                      for i in range(len(_cv))})
                    # SELF-STATE THE AGENT CURRENTLY FEELS. act() runs
                    # BEFORE this step, so it must use the LAST observation's
                    # proprio — reading this step's step_infos here referenced
                    # it before assignment and crashed the episodic path on
                    # its very first iteration.
                    _props = getattr(self, "_proprio_per_env", None)
                    env_actions = self.option_executor.act(
                        obs_list, primary_kv, self.policy,
                        self.total_timesteps, predicates_per_env=_preds,
                        competence=_comp, proprio_per_env=_props,
                        feats_per_env=self._feats_for_act(n),
                        verify_feats=self._verify_enc_feats)
                    primary_pol = (self.option_executor._primary_primitive
                                   or {"log_prob": 0.0, "value": 0.0})
                else:
                    env_actions = []
                    # SCOUT DECISIONS ARE KEPT HERE TOO (2026-09-01). This
                    # branch runs when options are DISABLED, and it already
                    # computed `pinfo` for every env and then discarded it for
                    # e_i != 0. The scout-PPO storage below hangs off the
                    # option executor, which is None on this path — so with
                    # skills_as_options.enabled false, `_scouts_in_ppo` read
                    # true and not one scout row was ever stored. That is the
                    # _viewer_push / gui_open / felt-reach pattern exactly: a
                    # correct mechanism nobody calls.
                    self._noopt_pol = [None] * n
                    for e_i in range(n):
                        a, pinfo = self.policy.select_action(
                            obs_list[e_i],
                            knowledge=primary_kv if e_i == 0 else None,
                            feats=(None if self._act_latent is None
                                   else self._act_latent[e_i:e_i + 1]))
                        env_actions.append(int(a) if self.is_discrete else a)
                        self._noopt_pol[e_i] = pinfo
                        if e_i == 0:
                            primary_pol = pinfo
                if self.is_discrete:
                    action_tensor = torch.zeros(n, self.action_dim,
                                                device=self.device)
                    action_tensor[range(n),
                                  [int(a) for a in env_actions]] = 1.0
                else:
                    action_tensor = torch.from_numpy(
                        np.stack(env_actions).astype(np.float32)
                    ).to(self.device)

            _pt = self._phase_mark("act", _pt)
            self._lt_pre_step(env_actions, n, use_dream_actor)
            # FOUNDATION SHADOW (off unless foundation.shadow.enabled): obs t
            # and the world model's prediction for the chosen action, logged
            # BEFORE env.step so the store can prove it preceded the outcome.
            if self._shadow is not None:
                self._shadow.before_step(rssm_state, env_actions,
                                         self._wm_param_lock, self.world_model,
                                         executor=self.option_executor, observations=obs_list)
                _pt = self._phase_mark("shadow", _pt)

            # ---- 2. STEP ALL ENVS (concurrently — see reset note) ----
            from concurrent.futures import TimeoutError as _FTimeout
            _step_to = float(self.config.get("parallel_envs", {}).get(
                "step_timeout_s", 120))
            # WHERE THE WALL CLOCK GOES (2026-08-17). The run holds ~3.4
            # steps/s against a 10 steps/s ceiling (20 server ticks/s over
            # action_repeat 2) while the GPU sits idle, so two thirds of
            # every step is unaccounted for. Timing the env round-trip is
            # what separates "MineRL is slow" from "our own per-step Python
            # is slow" — those call for completely different work, and
            # guessing which has already cost this project weeks. Set in
            # BOTH duplicated bodies on purpose: letting these two drift is
            # how the last six same-shape bugs were born.
            _t_env0 = time.time()
            futs = [self._env_pool.submit(envs[e_i].step, env_actions[e_i])
                    for e_i in range(n)]
            next_obs_list, rewards, dones = [], [], []
            _sh_ends = []   # shadow: (terminated, truncated, t_result) per env
            terminateds, restarted = [], []   # LIFELONG: terminal vs restart
            prim_info: Dict[str, Any] = {}
            step_infos: List[Dict[str, Any]] = [None] * n
            # keep the body sense fresh for the NEXT act() (see _props)
            if self._proprio_source is not None:
                self._proprio_per_env = [None] * n
            # PERSISTS across steps (it is a one-step memory), so it is sized
            # once and never cleared here — unlike _proprio_per_env above,
            # which is this step's fresh reading.
            if (self._wm_proprio_dim
                    and (self._wm_proprio_prev is None
                         or len(self._wm_proprio_prev) != n)):
                self._wm_proprio_prev = [None] * n
            for e_i, f in enumerate(futs):
                _rst = False
                try:
                    nobs, rew, term, trunc, _inf = f.result(timeout=_step_to)
                    if isinstance(_inf, dict):
                        step_infos[e_i] = _inf
                        if (self._proprio_per_env is not None
                                and e_i < len(self._proprio_per_env)):
                            # loop-side senses appended here too — this body
                            # previously stored the RAW env vector and the
                            # reach feeling was silently zero-padded away in
                            # every lifelong run (6th duplicated-body bug)
                            self._proprio_per_env[e_i] = \
                                self._augment_proprio(
                                    (_inf or {}).get("proprio"), e_i)
                    if e_i == 0 and isinstance(_inf, dict):
                        prim_info = _inf   # symbolizer reads mine_* from here
                except (_FTimeout, Exception) as ex:
                    # Hung/failed client: the wrapper rebuilt its own world, so
                    # this is a crash-RESTART (a true terminal), not a time
                    # truncation. Carry last obs, zero reward.
                    logger.warning("env %d step hung/failed (%s) — restart",
                                   e_i, ex)
                    nobs = (obs_list[e_i]
                            if obs_list[e_i] is not None
                            else np.zeros(self.obs_dim, dtype=np.float32))
                    rew, term, trunc, _rst = 0.0, False, True, True
                next_obs_list.append(np.asarray(nobs, dtype=np.float32))
                rewards.append(float(rew))
                _sh_ends.append((bool(term), bool(trunc), time.time()))
                terminateds.append(bool(term))
                restarted.append(_rst)
                # LIFELONG: the STORED done is the TERM-FLAG (real terminal or
                # crash-restart). A plain 8000-tick time-TRUNCATION is NOT a
                # boundary and must not sever the advantage trace, so it is
                # deliberately excluded here.
                dones.append(bool(term) or _rst)
            # envs step CONCURRENTLY, so this is the wait for the SLOWEST
            # client, which is exactly what the serial loop pays.
            self._env_wait_sum = (getattr(self, "_env_wait_sum", 0.0)
                                  + (time.time() - _t_env0))
            self._env_wait_n = getattr(self, "_env_wait_n", 0) + 1
            self._env_parts_add(step_infos)
            _pt = self._phase_mark("env", _pt)
            # ---- SET HERE, NOT IN THE COVERAGE BLOCK (2026-09-01) ----
            # `_boring_view_factor()` reads `_last_world_info["pitch"]`, and
            # it is called from the ICM base discount — which runs EARLIER in
            # this body than the coverage block that used to assign it. So a
            # term scaling up to 85% of the base drive was computed from the
            # PREVIOUS decision's pitch, and action_repeat 2 -> 4 doubled that
            # staleness from 2 game ticks to 4. Assigned immediately after the
            # env results land, which is the first moment it is knowable.
            # ALSO ADDED TO THE EPISODIC BODY, which never set it at all: it
            # was reading whatever the last lifelong segment left behind.
            # ...the FULL info too: ticks-to-break lives at top level, not
            # under "world", and reading the wrong dict would silently print
            # nothing — which is how a measurement quietly becomes a
            # non-measurement (this has happened repeatedly here).
            if step_infos:
                self._last_world_info = (step_infos[0] or {}).get("world") or {}
                self._last_env_info = step_infos[0] or {}
            # per-env newly-broken block this step (highest tier wins:
            # log > solid > plant, mirroring the tiered break reward, so the
            # naming picks the block that actually spiked the reward)
            self._new_break_by_env = {}
            for _e in range(n):
                _inf = step_infos[_e]
                if not isinstance(_inf, dict):
                    continue
                _ach = _inf.get("achievements", {}) or {}
                _hw = self._mine_by_env.setdefault(_e, {})
                _best, _best_rank = None, -1
                for _k, _v in _ach.items():
                    # CRAFTS ARE GROUNDED EFFECTS TOO (2026-07-26): a
                    # craft_item increment is as real as a block break and
                    # outranks everything — making something is the rarest,
                    # most significant event in this world. The effect key
                    # KEEPS its craft_ prefix (`craft_planks`), so it can
                    # never collide with a block of the same name and the
                    # goal slot is legibly a crafting goal.
                    if _k.startswith("craft_"):
                        _b, _rank_base = _k, 3
                    elif _k.startswith("mine_"):
                        _b, _rank_base = _k[5:], None
                    else:
                        continue
                    try:
                        _c = int(_v)
                    except (TypeError, ValueError):
                        continue
                    if _c > _hw.get(_b, 0):
                        _hw[_b] = _c
                        # MASTERY GROUND TRUTH (task #39): every open option
                        # frame on this env witnesses the effect key in the
                        # SAME vocabulary skills are named in (break_X /
                        # craft_Y) — at close, each frame is asked whether
                        # its OWN key is among what it witnessed.
                        if self.option_executor is not None:
                            self.option_executor.note_effect(
                                _e, _k if _k.startswith("craft_")
                                else f"break_{_b}")
                        _rank = (_rank_base if _rank_base is not None else
                                 (2 if "log" in _b else
                                  0 if _b in ("grass", "tall_grass", "fern",
                                              "vine", "seagrass") else 1))
                        if _rank > _best_rank:
                            _best, _best_rank = _b, _rank
                # LOG-PICKUP ATTRIBUTION GUARD (audit fix): the +1/log
                # inventory reward has no mine_* counterpart (a pickup breaks
                # nothing), so a pickup spike landing on the same step as an
                # incidental leaf/grass break used to be grounded-keyed to the
                # VEGETATION — minting exactly the foliage goals the tiered
                # rewards were built to suppress, and blending leaf signatures
                # into log-goal provenance. If the log count rose this step and
                # no mine_*log break explains it, pass NO effect key (ungrounded
                # signature clustering handles that spike correctly).
                if not hasattr(self, "_loginv_by_env"):
                    self._loginv_by_env = {}
                # ---- MISSING IS NOT ZERO (fix 2026-09-04) ----------------
                # MEASURED: the live run reported log_pickup=1547 against
                # NINE logs ever broken. Those cannot both be true, and the
                # counter was the liar.
                # CAUSE: `_ach` is `_inf.get("achievements", {}) or {}`, so
                # on any step whose info lacks that dict (or lacks the "log"
                # key) `_ach.get("log", 0)` returned 0 — indistinguishable
                # from "the agent is carrying zero logs". Carrying 3 logs
                # through a single dropped observation therefore reads
                # 3 -> 0 -> 3, and the recovery is counted as a PICKUP. With
                # an intermittent key that manufactures hundreds of pickups
                # out of one real one, which is exactly the 1547-vs-9 gap.
                # This is the counter class CLAUDE.md §5 warns about by name
                # (`places` listing iron_axe: 2281) — and it is not cosmetic:
                # log_pickup is a GROUNDED EFFECT KEY, so every phantom
                # pickup fed a real goal slot and a real reward event.
                # An absent reading is UNKNOWN. Skip the comparison and do
                # not overwrite the last KNOWN count, so a dropped frame is
                # simply not evidence rather than being evidence of a gain.
                _lg = None
                if isinstance(_ach, dict) and "log" in _ach:
                    try:
                        _lg = max(0, int(_ach["log"]))
                    except (TypeError, ValueError):
                        _lg = None
                _lg_prev = self._loginv_by_env.get(_e)
                if _lg is not None:
                    self._loginv_by_env[_e] = _lg
                else:
                    self._loginv_unknown = getattr(
                        self, "_loginv_unknown", 0) + 1
                if (_lg is not None and _lg_prev is not None
                        and _lg > _lg_prev
                        and (_best is None or "log" not in _best)):
                    # STABLE PICKUP KEY (2026-07-25) — was `_best = None`.
                    # Refusing to mis-ground a pickup was right, but None
                    # routed it into UNGROUNDED signature clustering, so every
                    # novel viewpoint of picking up a log minted a fresh
                    # `discovered_N` goal + skill. The core task reward was
                    # the dominant junk producer (48/48 goal slots were
                    # `discovered_N`). A stable canonical key collapses ALL
                    # pickups into ONE grounded slot instead.
                    _best = "log_pickup"
                if _best is not None:
                    self._new_break_by_env[_e] = _best

            next_obs_t = torch.from_numpy(np.stack(next_obs_list)).to(self.device)

            if (not use_dream_actor and self.option_executor is not None
                    and not self._scouts_in_ppo):
                # scout option terminations (raw reward; no credit records)
                #
                # SKIPPED WHEN SCOUTS FEED PPO (2026-09-01). Accumulating a
                # scout option's SMDP return needs the MIXED reward, which
                # does not exist until the shaping block further down, so
                # that path calls observe_scouts there instead. Calling it in
                # both places would advance `steps_done` twice per env step
                # and terminate every scout option at half its horizon — the
                # two sites are mutually exclusive by construction rather
                # than by convention. `_scouts_in_ppo` is false on the
                # episodic path, so this body is unchanged there.
                self.option_executor.observe_scouts(
                    rewards, dones, self.total_timesteps)

            # Goal broadcaster ingests ALL streams (July 2026 pivot, part 2).
            # Discovery previously watched only the primary env, so 7/8 of
            # the fleet's experience could never mint a goal slot — with
            # first-discovery being THE bottleneck in Minecraft, that threw
            # away 8x the odds of catching the first grounded event. Each
            # wrapper keeps its own reward/obs history; the signature
            # clustering in DiscoveredAchievementGoals dedups events found
            # independently by different streams. Each env passes its STREAM
            # INDEX: per-stream episode state keeps the self-model learning
            # P(success in one episode) instead of P(any-of-N succeeds).
            _unlocks_before = (len(self.broadcaster.unlock_log)
                               if (self.broadcaster is not None and hasattr(
                                   self.broadcaster, "unlock_log")) else 0)
            if self.broadcaster is not None:
                for _i, e in enumerate(envs):
                    # grounded-effect dedup: pass WHAT broke on this stream so
                    # the same behaviour keys one slot (kills break_*_leaves x8)
                    self._broadcaster_update(
                        e, _i, effect_key=self._new_break_by_env.get(_i))

            _pt = self._phase_mark("goals", _pt)

            # ---- 3. CURIOSITY (batched) ----
            # PERSPECTIVE FROM MOTION joins the drive HERE, as a second
            # prediction-error channel feeding learning progress — NOT as a
            # new reward term. LP pays for error going DOWN, so a new channel
            # changes what the agent can make progress ON, never how much it
            # is paid for standing anywhere. And the residual is a ratio
            # against how much the frame changed at all: no movement, no
            # change, no residual, no income.
            # EVALUATION SINK. Reads info["oracle"] and nothing else writes
            # to the agent from here — see _oracle_observe.
            self._oracle_observe(step_infos, n)
            self._lt_post_step(step_infos, dones, n)
            _fr = self._flow_residual_batch(
                obs_t, next_obs_t, action_tensor, n,
                ego=self._flow_ego_batch(step_infos, n))
            self._flow_resid_now = (
                float(_fr[0]) if _fr is not None else 0.0)
            if _fr is not None:
                self.training_metrics["flow_residual"].append(
                    self._flow_resid_now)
            intrinsic = self._curiosity_reward(
                obs_t, action_tensor, next_obs_t, extra_error=_fr,
                ended=dones, invalid=[bool(restarted[i]) or bool(
                    (step_infos[i] or {}).get("env_restarted")) for i in range(n)])
            # ICM BASE SCALE (infra #13): the raw-error term is damped by
            # config so it seasons rather than dominates. Since 2026-10-05
            # paired progress is MEASUREMENT ONLY and pays nothing, so this
            # damped term (plus the itemised novelty terms) IS the base drive
            # — nothing "leads" it any more. 1.0 = old behaviour.
            intrinsic = intrinsic * float(getattr(self, "_icm_base_scale", 1.0))
            self._lt_begin(intrinsic)
            # BORING-VIEW BASE DISCOUNT (2026-08-16). The sky discount only
            # ever covered the itemised novelty term; the BASE was left at
            # full pay. Measured live at the -90 clamp: ICM/LP base
            # +0.0147/step = 96% of the whole drive, earned by watching
            # clouds drift — the census's own caption said it: "prediction
            # error alone pays this much for EXISTING, so standing still is
            # already profitable and no shaping term can outbid it". Reuse
            # the SAME measured judgement (_boring_view_factor: geometry
            # first, fovea second, learned mastery-boringness third, floored
            # at 0.15 so first glances pay) on the primary stream's base.
            # Reward-side only — the WM/curiosity modules still train on
            # every frame, so perception keeps learning from the sky; it is
            # the WAGE that dies, not the sight. Applied BEFORE the census
            # accumulator below, so the census cannot lie about it.
            _bvw = float(getattr(self, "_icm_boring_discount", 0.0))
            if _bvw > 0.0:
                _bf = float(self._boring_view_factor())
                intrinsic[0] = intrinsic[0] * (1.0 - _bvw * (1.0 - _bf))
                self._lt_mark("damp_boring_view", intrinsic, 0.0, None, "boring_view", 1.0 - _bvw * (1.0 - _bf))
                self._bv_sum = getattr(self, "_bv_sum", 0.0) + _bf
                self._bv_n = getattr(self, "_bv_n", 0) + 1

            # EVERY STEP, UNCONDITIONALLY (fix 2026-08-03). This accumulator
            # was first placed inside the imagination-curiosity block, which
            # is gated by wants_step() with `every: 50` — so it sampled ~20
            # of every 1024 steps, and not representatively. The census then
            # reported base +0.0073/step against a "+0.0189 total drive"
            # while the segment's ACTUAL mean intrinsic was +0.1958/step: a
            # ~10x understatement that made the progress floor look far more
            # effective than it was. A diagnostic that samples a hot loop
            # conditionally is a diagnostic that lies.
            self._cen_base = getattr(self, "_cen_base", 0.0) + float(
                intrinsic[0].item())
            self._cen_n = getattr(self, "_cen_n", 0) + 1
            # SEGMENT MEAN (fix 2026-08-04). `last_action_attribution` is
            # overwritten every step, so printing it directly showed whatever
            # the FINAL step of the segment happened to be — a 2-sample
            # reading from one timestep. It printed 0.000 and read as "the
            # gate is dead", which is indistinguishable from "the last step
            # was a no-op". Same sampling mistake as the census accumulator,
            # made twice: a diagnostic that samples a hot loop unrepresenta-
            # tively is worse than no diagnostic, because it looks like data.
            _at = getattr(self.curiosity, "last_action_attribution", None)
            if _at is not None:
                self._attr_sum = getattr(self, "_attr_sum", 0.0) + float(_at)
                self._attr_n = getattr(self, "_attr_n", 0) + 1
                if float(_at) > 1e-6:
                    self._attr_nz = getattr(self, "_attr_nz", 0) + 1
            icm_metrics = self._curiosity_train(
                obs_t, action_tensor, next_obs_t
            )
            _pt = self._phase_mark("curiosity", _pt)

            # ---- 4. RSSM observe (batched) + primary-only symbolic facts ----
            # _wm_param_lock (review fix): with the async trainer these
            # forwards would otherwise read params MID-optimizer-step; the
            # torn latents flow into the permanently carried lifelong state.
            # Uncontended (sync mode) the RLock costs ~1us.
            with self._wm_param_lock, torch.no_grad():
                # SAME BODY THE TRAINER SEES. The world model is trained on
                # (frame, proprio) pairs from the buffer; encoding the live
                # frame with a zero body here would make every acting latent
                # come from a distribution the model never trained on.
                encoded_next = self.world_model.embed(
                    next_obs_t, self._wm_proprio_batch(step_infos, n))
                rssm_state, _ = self.world_model.rssm.observe_step(
                    rssm_state, action_tensor, encoded_next
                )
                latent = self.world_model.rssm.get_latent(rssm_state)
            # LIFELONG carry fix: observe_step RETURNS A NEW dict, so the local
            # rssm_state diverges from self._ll.rssm_state (which otherwise stays
            # the startup seed) — silently making the "continuous" cross-segment
            # carry a no-op AND making death_reset zero the wrong (stale) tensor.
            # Sync the LifelongState every step so the next segment resumes from
            # the evolved state and death_reset/decay mutate the LIVE object.
            # No-op off the lifelong path (self._ll is None).
            if self._ll is not None:
                self._ll.rssm_state = rssm_state
            # arch='rssm': the belief the NEXT decision will be made from, kept
            # for the segment-end GAE bootstrap. Cheap (a reference), and it
            # has to be the post-observe latent — that is the state the next
            # segment resumes at, which is exactly what V(s_last) means.
            if self.policy.arch == "rssm":
                self._bootstrap_latent = latent.detach()

            # ---- IMAGINATION CURIOSITY (primary stream, INTRINSIC only) ----
            # Rolls the WM forward from the live latent: pays for reaching
            # states the model is UNCERTAIN about (disagreement — never dies),
            # plus a small DECAYING daydream bonus. Gated on boredom, so it
            # contributes ~nothing while the environment is still teaching.
            # Added to intrinsic[0] — NOT to prim_extrinsic — so it can never
            # become a replay reward LABEL and train the WM reward head on the
            # agent's own daydreams (that loop = self-hallucinated reward).
            if (not use_dream_actor
                    and self.imagination_curiosity.enabled
                    and self.imagination_curiosity.wants_step(
                        self.total_timesteps)
                    and (self._last_wm_metrics or self._wm_metrics_box)):
                _vs = (self.vision_scaffold.stats
                       if self.vision_scaffold is not None else {})
                _glp = float(_vs.get("global_lp", 0.0) or 0.0)
                _kn = self.glue.knowledge_integrator.knowledge_vector
                _P = self.action_dim

                # the actor input is [obs, gated_knowledge]; feeding raw obs
                # is a shape error (obs_dim vs obs_dim+DIM). Mirror the
                # prospection path and augment with the CURRENT goal broadcast.
                _ikv = self.policy._prep_knowledge(
                    self._current_knowledge_feature())

                def _imag_policy(_lat, _P=_P, _ikv=_ikv):
                    # arch='rssm' SHORT-CIRCUITS THE ROUND TRIP. The policy's
                    # input IS the latent, so decoding it to pixels and
                    # re-encoding them is not merely wasted compute — it
                    # pushes the imagined state through a lossy
                    # decode/encode pair and asks the actor about the
                    # result. Feeding `_lat` straight in is both faster and
                    # strictly closer to what the actor sees when awake.
                    if self.policy.arch == "rssm":
                        _k = (_ikv.expand(_lat.shape[0], -1)
                              if _ikv is not None else None)
                        _aug = self.policy._augment(
                            _lat, _k, self._cur_proprio_t())
                        _lg = self.policy.actor.action_head(
                            self.policy.actor.shared(_aug))
                        _a = torch.distributions.Categorical(
                            logits=_lg[..., :_P]).sample()
                        return torch.nn.functional.one_hot(_a, _P).float()
                    _obs = self.world_model.decoder(_lat)
                    _obs = (_obs.flatten(1) if self.pixel_obs
                            else symexp(_obs))
                    _k = (_ikv.expand(_obs.shape[0], -1)
                          if _ikv is not None else None)
                    # conv: the MLP stack consumes ENCODER FEATURES, not
                    # raw pixels (review HIGH — these three imagination/
                    # prospection/distill sites bypassed the trunk).
                    _aug = self.policy._augment(
                        self.policy._encode(_obs), _k, self._cur_proprio_t())
                    _lg = self.policy.actor.action_head(
                        self.policy.actor.shared(_aug))
                    _a = torch.distributions.Categorical(
                        logits=_lg[..., :_P]).sample()
                    return torch.nn.functional.one_hot(_a, _P).float()

                with self._wm_param_lock:
                    _ib = self.imagination_curiosity.bonus(
                        self.world_model, rssm_state, _imag_policy,
                        self.total_timesteps, _glp,
                        knowledge=_kn, symexp_fn=symexp)
                # REWARD CENSUS (2026-08-02). The segment log itemised
                # coverage, gaze, centring, symbols, novelty, reach and
                # persistence — together ~2% of a +0.19/step drive. The other
                # ~98% is the raw ICM/learning-progress term and it was NEVER
                # printed, so "what is the agent paid for while standing
                # still?" was unanswerable from the log. Captured here,
                # BEFORE any shaping is added, so it is the base in isolation.
                if _ib:
                    intrinsic[0] = intrinsic[0] + _ib
                    self._cen_imag = getattr(self, "_cen_imag", 0.0) + float(_ib)

            # FELT REACH — computed HERE, in the lifelong loop. It was only
            # ever computed in `_run_episode_parallel` (the episodic path),
            # which this run never enters, so `_reach_now` stayed 0.0 and both
            # the reach proprioception field and the approach-to-reach
            # potential were silently inert the whole time.
            if not use_dream_actor and self.symbolizer is not None:
                self._reach_now = self._reach_sense(latent[0:1])
            # PERSPECTIVE FROM MOTION, same cadence and same cost profile as
            # reach. Deliberately NOT gated on the symbolizer: this is
            # geometry read off the world model, and it exists in an env that
            # never grounds a single symbol.
            if getattr(self.world_model, "flow_enabled", False):
                if not use_dream_actor:
                    self._flow_now = self._flow_senses(
                        latent[0:1], self._flow_resid_now)
                    self._spatial_step(
                        latent[0:1], step_infos,
                        # DID THE AGENT ASK TO MOVE. Walkability is learned
                        # from the GAP between the command and the outcome,
                        # so the command half has to come from the action
                        # actually issued to the primary body.
                        self._is_move_action(int(env_actions[0])))
                # THE MEMORY IS REFRESHED EVEN WHILE DREAMING, deliberately.
                # The senses above are for ACTING and a dreaming agent is not
                # acting, but the cache is a fact about which frame the next
                # residual will be measured against. Leaving it stale through
                # a dream block would make the first waking residual compare
                # the current frame to one from before the dream — a splice
                # across time that reads as enormous unexplained motion.
                # THE ONE-STEP MEMORY, refreshed AFTER it has been consumed.
                # `valid` marks the streams whose next frame will still belong
                # to THIS world: a stream that just ended or was rebuilt has a
                # previous frame from a different world entirely, and warping
                # across that splice would report a near-total residual and
                # read as the most interesting event in the run.
                self._flow_prev_latent = latent.detach()
                self._flow_prev_valid = torch.tensor(
                    [not (bool(dones[i]) or bool(restarted[i]))
                     for i in range(n)],
                    dtype=torch.bool, device=self.device)
            # THE ORACLE'S FRAME RESETS WITH THE WORLD. The dead reckoner's
            # origin is the episode start; truth must be re-anchored at the
            # same instant or the drift measurement becomes a constant offset.
            if bool(dones[0]) or bool(restarted[0]):
                self._oracle_reset()
                self._spatial_reset()
                # PREFER MEASURED OVER PREDICTED (2026-08-04). The VLM-taught
                # `breakable_in_reach` head sat at 0.01-0.03 for an entire run
                # in which the agent broke 4931 blocks — llava cannot judge
                # reach from a still frame, so the head learned to answer "no"
                # and two shaping terms plus a proprioception field rode on it.
                # The env now emits ground truth: a completed break PROVES
                # something was within range. Use it when present; fall back
                # to the head otherwise.
                if self._reach_source == "evidence":
                    _re = (getattr(self, "_last_world_info", None)
                           or {}).get("reach_evidence")
                    if _re is not None:
                        self._reach_now = float(_re)
                        self._reach_gate = "measured"
                # SEGMENT AGGREGATE (fix 2026-08-04). `Reach sense` printed
                # `_reach_now` — whatever the FINAL step of the segment
                # happened to be. It read 0.00 in a segment containing 142
                # breaks, which says nothing about whether the trace works.
                # Third time this session I sampled a hot loop at one point
                # and reasoned off it (after the reward census and the action
                # attribution), so every such readout is now a mean.
                # getattr, LIKE EVERY OTHER READ OF THIS ATTRIBUTE (fixed
                # 2026-10-01). `_reach_now` is only ASSIGNED inside
                # conditionals (4860, 6035, 6087) -- and the comment 60 lines
                # up already says "which this run never enters, so
                # `_reach_now` stayed 0.0". When none of those paths run the
                # attribute does not exist, and this bare read killed the
                # whole run:
                #   AttributeError: 'DevelopmentalAI' object has no attribute
                #   '_reach_now'. Did you mean: '_reach_n'?
                # Lines 1874, 6209 and 10673 all use getattr with a 0.0
                # default; this was the only bare one.
                _rv = float(getattr(self, "_reach_now", 0.0))
                self._reachval_sum = getattr(self, "_reachval_sum", 0.0) + _rv
                self._reachval_n = getattr(self, "_reachval_n", 0) + 1
                if _rv > 0.5:
                    self._reachval_hi = getattr(self, "_reachval_hi", 0) + 1

            # ---- SYMBOLIC NOVELTY: new THINGS, not new pixels ----------
            # Pays for acquiring a symbol and for seeing combinations it has
            # not seen, decaying as the vocabulary becomes familiar. Staring
            # at the ground produces one unchanging signature, so it goes to
            # ~0 within seconds rather than paying forever.
            if (float(self._symbol_weight) > 0.0
                    or float(self._new_symbol_bonus) > 0.0) \
                    and not use_dream_actor:
                _sb, _nk, _sig = self._symbol_novelty(latent[0:1])
                if _sb:
                    intrinsic[0] = intrinsic[0] + _sb
                self._sym_sum = getattr(self, "_sym_sum", 0.0) + float(_sb)
                self._sym_n = getattr(self, "_sym_n", 0) + 1
                self._sym_known = _nk
                # ---- C: pay for BRINGING it to the middle of the frame ----
                # Potential-based (Ng): F = gamma*Phi(s") - Phi(s). Optimal
                # policy unchanged; what it supplies is the per-step gradient
                # that "turn toward the thing" never had. Intrinsic channel
                # only, so it can never become a replay reward LABEL.
                _scw = float(self._symbol_center_weight)
                if _scw > 0.0:
                    _phi_c = float(getattr(self, "_sym_center_phi", 0.0))
                    _prev_c = float(getattr(self, "_sym_center_prev", 0.0))
                    _gc = float(self.config.get("policy", {}).get(
                        "gamma", 0.99))
                    _fc = _scw * (_gc * _phi_c - _prev_c)
                    self._sym_center_prev = _phi_c
                    intrinsic[0] = intrinsic[0] + _fc
                    self._symc_sum = getattr(self, "_symc_sum", 0.0) + _fc
                    self._symc_abs = getattr(self, "_symc_abs", 0.0) + abs(_fc)
                    self._symc_n = getattr(self, "_symc_n", 0) + 1

            # ---- NOVELTY OF EXPERIENCE, NOT OF PLACE -------------------
            # Same count-based form as territory coverage (new -> ~1, revisits
            # decay as 1/sqrt(n)), but keyed on WHAT THE AGENT SEES instead of
            # where its feet are. Walking in a straight line staring at the
            # floor maximises distance and sees almost nothing, so it earns
            # almost nothing here; turning to look at something it has not
            # seen before earns, and keeps earning only while it is genuinely
            # new. Intrinsic channel only -> never a replay reward label.
            _nw = float(self._novelty_weight)
            if _nw > 0.0 and not use_dream_actor:
                # DETERMINISTIC KEY. The RSSM's z is a SAMPLED categorical
                # (StraightThroughCategorical), redrawn every step — so an
                # identical frame hashed to a DIFFERENT bucket each time and
                # staring at unchanging ground still read as 369 "new views"
                # per segment. Novelty keyed on a stochastic variable is
                # novelty that can never be exhausted. The encoder output is
                # a pure function of the frame, so the same sight is the same
                # bucket and repetition genuinely decays.
                # NEXT_obs, not obs (fix 2026-08-02). Keyed on `obs_list[0]`
                # this paid for the view the agent was LEAVING, so the action
                # that actually brought a new sight into frame was credited to
                # the previous state and the term supplied almost no gradient
                # toward seeking anything. Every other latent-derived term in
                # this loop (`latent`, the reach sense, symbol novelty) is
                # computed from next_obs; this one was the odd man out.
                _pc = self._perceptual_cell(self._view_key(next_obs_list[0]))
                if _pc >= 0:
                    _n = self._nov_counts.get(_pc, 0)
                    self._nov_counts[_pc] = _n + 1
                    if len(self._nov_counts) > 200000:
                        self._nov_counts.pop(next(iter(self._nov_counts)))
                    _nv = _nw * (1.0 / ((1.0 + _n) ** 0.5))
                    # BORING-VIEW DISCOUNT (see _boring_view_factor): sky
                    # (drifting clouds mint buckets forever) and MASTERED
                    # materials (a fresh dirt hole is a new view, and digging
                    # manufactures them endlessly) pay reduced view-novelty,
                    # in proportion to how much of the FOVEA they fill.
                    if self._nov_sky_discount:
                        _nv *= self._boring_view_factor()
                    # SEMANTIC GATE (2026-08-11): price this sight by how new
                    # its MEANING is, not by how new its pixels are. This is
                    # what stops a self-dug hole from reading as a discovery.
                    _sf = self._semantic_novelty_factor()
                    _nv *= _sf
                    self._nov_sem_sum = (getattr(self, "_nov_sem_sum", 0.0)
                                         + float(_sf))
                    self._nov_sem_n = getattr(self, "_nov_sem_n", 0) + 1
                    intrinsic[0] = intrinsic[0] + _nv
                    self._nov_sum = getattr(self, "_nov_sum", 0.0) + _nv
                    self._nov_n = getattr(self, "_nov_n", 0) + 1

            # ---- EXTERNAL CURIOSITY: territory coverage (INTRINSIC only) ----
            # Count-based novelty over discretised world cells, emitted by the
            # env adapter from the newly-restored position stats. This is what
            # keeps the WORLD able to surprise a lifelong agent: standing still
            # pays ~0, new ground pays ~1, and revisiting decays as 1/sqrt(n).
            # Also intrinsic-only => never a replay label.
            _cw = float(self._coverage_weight)

            # ---- #7: NOTHING PAID FOR GETTING WITHIN REACH -------------
            # The stall fix zeroed instinct/approach/align/aim because they
            # were RAW PER-STEP INCOME and a parked agent farmed them
            # (~0.14/step for standing at a trunk). That was correct, but it
            # left `seek` paying only for GLIMPSING a target and nothing at
            # all for CLOSING on one — so the agent was rewarded for looking
            # at trees and never for reaching them.
            #
            # This restores the missing half in the telescoping form:
            # a potential on the grounded probability that something is
            # BREAKABLE IN REACH. Approaching pays; standing still pays
            # exactly 0 (Phi stops changing); backing off charges back what
            # approaching earned. Unfarmable by construction, unlike the term
            # it replaces. Uses the same head+trust gate as the reach sense.
            _rw = float(self._reach_weight)
            if _rw > 0.0 and not use_dream_actor:
                _rp = float(getattr(self, "_reach_now", 0.0))
                _rprev = float(getattr(self, "_reach_phi", 0.0))
                _g2 = float(self.config.get("policy", {}).get("gamma", 0.99))
                _fr = _rw * (_g2 * _rp - _rprev)
                self._reach_phi = _rp
                intrinsic[0] = intrinsic[0] + _fr
                self._reach_sum = getattr(self, "_reach_sum", 0.0) + _fr
                self._reach_n = getattr(self, "_reach_n", 0) + 1

            # ---- #1/#3: PERSISTENCE IS OTHERWISE INVISIBLE ------------
            # ICM pays for prediction error, i.e. for the frame CHANGING.
            # Holding attack on one block is the lowest-change action
            # available (the crack overlay is a few pixels at 128px), while
            # turning repaints everything. So the dominant drive paid MOST
            # for the behaviours that prevent breaking and LEAST for the one
            # that achieves it — and every earlier fix only moved which cheap
            # novelty it farmed.
            #
            # POTENTIAL-BASED, so it cannot be farmed. With
            #     Phi = min(1, attack_run / ticks_needed)
            #     F   = gamma * Phi(s') - Phi(s)
            # Ng's theorem says the optimal policy is UNCHANGED: any cycle of
            # swing-then-stop telescopes to ~0, so there is no income in
            # starting swings and abandoning them. What it does supply is a
            # per-step GRADIENT for continuing one, which is exactly the
            # signal a step-wise stochastic policy needs to hold an action
            # for the ~8 consecutive macros a barehanded break requires.
            # Intrinsic channel only -> never a replay reward label.
            _pw = float(self._persist_weight)
            if _pw > 0.0 and step_infos:
                _ar = float((getattr(self, "_last_env_info", None) or {})
                            .get("attack_run", 0.0))
                _phi = min(1.0, _ar / max(1.0, float(self._persist_ticks)))
                _phi_prev = float(getattr(self, "_persist_phi", 0.0))
                _g = float(self.config.get("policy", {}).get("gamma", 0.99))
                _f = _pw * (_g * _phi - _phi_prev)
                self._persist_phi = _phi
                intrinsic[0] = intrinsic[0] + _f
                self._persist_sum = getattr(self, "_persist_sum", 0.0) + _f
                self._persist_n = getattr(self, "_persist_n", 0) + 1

            # ---- GAZE LEVELING: the clamp costs nothing to lean on ------
            # Measured live (2026-08-16): pitch pinned at +90, at a clamp on
            # 21% of steps, every vision predicate starved into DEGENERATE.
            # The gaze-bucket bonus below unsticks a clamp ONCE and then
            # saturates, so nothing durable makes leaning on the clamp cost.
            # POTENTIAL-BASED (Ng): Phi = -(|pitch|/90)^4, F = g*Phi' - Phi.
            # Any look-down/look-back cycle telescopes to ~0 — no farm, no
            # tax on transient mining tilts — but SITTING at the clamp means
            # having paid Phi: -1 and never collecting it back, so the exit
            # gradient is always live, unlike a count that decays. Intrinsic
            # channel only -> never a replay reward label.
            _plw = float(self._pitch_level_weight)
            if _plw > 0.0 and step_infos:
                _pvl = (getattr(self, "_last_world_info", None) or {}
                        ).get("pitch")
                if _pvl is not None:
                    _php = -((min(90.0, abs(float(_pvl))) / 90.0) ** 4)
                    _php_prev = getattr(self, "_pitch_level_phi", None)
                    if _php_prev is not None:
                        # PLAIN DIFFERENCE (fixed 2026-08-17). This shipped
                        # as gamma*Phi' - Phi, which pays w*(1-gamma) every
                        # step that Phi is PINNED — and Phi is pinned at -1
                        # exactly when the agent is stuck at a clamp. The
                        # log proved it: "Gaze level: +0.00014/step", a
                        # small WAGE for the behaviour the term was added to
                        # discourage. Phi' - Phi costs on entry, refunds on
                        # exit, and is flat while nothing changes.
                        _fpl = _plw * (_php - float(_php_prev))
                        intrinsic[0] = intrinsic[0] + _fpl
                        self._pitch_level_sum = getattr(
                            self, "_pitch_level_sum", 0.0) + _fpl
                        self._pitch_level_n = getattr(
                            self, "_pitch_level_n", 0) + 1
                    self._pitch_level_phi = _php

            # ---- GAZE COVERAGE: novelty of VIEW DIRECTION ---------------
            # Count-based over pitch buckets, so being pinned at a clamp is
            # the single most-visited direction there is and pays ~nothing,
            # while any direction it has not looked in pays. Declares nothing
            # about WHAT to look at — only that an unexplored gaze direction
            # is novel, exactly the principle already accepted for territory.
            # Intrinsic channel only => never a replay reward label.
            # ---- LIVE VIEWER (2026-08-02) -------------------------------
            # _viewer_push was called ONLY from _run_episode, the serial
            # episodic path. skybot runs _collect_segment, so the viewer has
            # never received a single frame in the mode that matters — the
            # THIRD subsystem found dead this way, after the gui_open guard
            # and the entropy/magnet diagnostics. _viewer_push swallows every
            # exception, so this cannot interrupt the run.
            if self.viewer is not None and not use_dream_actor:
                self._viewer_push(
                    env_actions[0], float(rewards[0]),
                    float(intrinsic[0].item()), rssm_state, latent[0:1],
                    episode_length)
                if self.infra is not None:
                    self.infra.beat("viewer", self.total_timesteps)

            # WHERE IT LOOKED, ALL SEGMENT — not just on the last step. A
            # single-step pitch cannot distinguish "pinned at the clamp" from
            # "swept through and happened to end there".
            _pv = (getattr(self, "_last_world_info", None) or {}).get("pitch")
            if _pv is not None:
                self._pitch_sum = getattr(self, "_pitch_sum", 0.0) + float(_pv)
                self._pitch_n = getattr(self, "_pitch_n", 0) + 1
                if abs(abs(float(_pv)) - 90.0) < 2.0:
                    self._pitch_clamped = getattr(
                        self, "_pitch_clamped", 0) + 1

            if float(self._gaze_weight) > 0.0 and step_infos:
                _gv = self._gaze_bonus(
                    (self._last_world_info or {}).get("pitch"))
                if _gv:
                    intrinsic[0] = intrinsic[0] + _gv
                    self._gaze_sum = getattr(self, "_gaze_sum", 0.0) + _gv
                    self._gaze_n = getattr(self, "_gaze_n", 0) + 1

            if _cw > 0.0 and step_infos:
                _wi = self._last_world_info
                _cov = float(_wi.get("coverage", 0.0) or 0.0)
                # DISCOVERY, NOT EXCAVATION (2026-08-09, live finding): a new
                # cell reached within a few seconds of SELF-EXCAVATION pays
                # nothing. Coverage credits "walked somewhere new"; the agent
                # had priced tunnels — ~2 dirt of effort (−0.06) bores a
                # fresh cell (+0.30), a 5x profit — and went back to eating
                # terrain. Fourth instance of MANUFACTURED NOVELTY (sky
                # views, pillar views, placement views, now tunnel cells):
                # novelty created by modifying the world is not discovery.
                # EXPRESSED IN GAME TICKS (2026-09-01). This was 40 STEPS,
                # which at action_repeat 2 meant 80 ticks — but the claim it
                # makes ("this cell was dug out a moment ago, so reaching it
                # is not discovery") is about world time, not decision count,
                # so at repeat 4 it silently covered 160 ticks. 160 // repeat
                # keeps the original 80-tick window whatever the knob says.
                if (_cov > 0.0 and self.total_timesteps
                        - getattr(self, "_last_excav_step", -10**9)
                        < (160 // self.env_action_repeat) * max(
                            1, int(getattr(self, "_num_envs", 1) or 1))):
                    self._cov_suppressed = getattr(
                        self, "_cov_suppressed", 0) + 1
                    _cov = 0.0
                if _cov > 0.0:
                    intrinsic[0] = intrinsic[0] + _cw * _cov
                # MEASURE THE TERRITORY DRIVE AT ITS POINT OF APPLICATION.
                # Whether coverage actually competes with the cheap novelty
                # of spinning was unanswerable: the value was applied and
                # never reported, so "coverage pays 0 because it is broken"
                # and "coverage pays 0 because the agent never walks" looked
                # identical from outside.
                self._cov_sum = getattr(self, "_cov_sum", 0.0) + _cw * _cov
                self._cov_n = getattr(self, "_cov_n", 0) + 1
                self._cov_cells = int(_wi.get("cells_seen", 0) or 0)
                if _wi.get("died"):
                    self._deaths_seen = getattr(self, "_deaths_seen", 0) + 1


            # pair fresh unlocks with the latent of the stream that earned
            # them (context provenance for skill minting / the brain graph)
            if (self.broadcaster is not None
                    and hasattr(self.broadcaster, "unlock_log")):
                for _ev in self.broadcaster.unlock_log[_unlocks_before:]:
                    _st = int(_ev.get("stream", 0))
                    if 0 <= _st < latent.shape[0]:
                        self._unlock_context[int(_ev["slot"])] = (
                            latent[_st].detach().cpu().numpy())
                    try:  # provenance frame (adapters cache last POV: cheap)
                        self._unlock_frame[int(_ev["slot"])] = (
                            envs[_st].render())
                    except Exception:
                        pass
                    # grounded name source: the block that broke on THIS
                    # stream this step is what the reward spike (= the
                    # unlock) was caused by
                    if _st in self._new_break_by_env:
                        self._unlock_block[int(_ev["slot"])] = (
                            self._new_break_by_env[_st])
                    # co-occurrence: unlock landed while (or just as) an
                    # option ran on that stream -> earned evidence that the
                    # SKILL causes the ACHIEVEMENT (fed to the prerequisite
                    # miner at episode end; same-step closes included).
                    if (not use_dream_actor
                            and self.option_executor is not None):
                        _rt = self.option_executor.runtimes[_st]
                        _sid = (_rt.skill_id if _rt.active else
                                self.option_executor
                                .last_closed_this_step[_st])
                        if _sid is not None:
                            self.option_executor.record_cooccurrence(
                                _sid, int(_ev["slot"]), _st,
                                self.total_timesteps)

            # Gate: per-dimension fact extraction is meaningless (and
            # unboundedly expensive) on raw pixels / high-dim structured obs.
            if not self._skip_perdim_symbolic:
                new_facts_count += self._extract_and_store_facts(
                    obs_list[0], env_actions[0], next_obs_list[0], rewards[0]
                )
            # sd heads + discretizer are trained on the async trainer thread;
            # lock the reads/writes here (review fix — write-write race on the
            # discretizer, torn sd-head reads)
            #
            # ---- GATED BY THE SAME FLAG AS THE BLOCK ABOVE (2026-09-01) ----
            # MEASURE FIRST, AND THE FIRST MEASUREMENT WAS WRONG. This was
            # flagged as "Welford over a 49,152-dim frame every step"; it is
            # not. `_skip_perdim_symbolic` already swaps in
            # `_NullSymbolicDecoder`, whose update_discretizer/extract_facts
            # are `pass`/`[]`, so on a pixel env the decoder arithmetic here
            # has always been free. Recording the correction rather than the
            # theory, because §5 is explicit that this project keeps paying
            # for plausible stories.
            #
            # WHAT IS ACTUALLY BOUGHT. Two things that are NOT free and have
            # no reachable value on a pixel env:
            #   * `_wm_param_lock` is ACQUIRED EVERY STEP. The async WM
            #     trainer holds it per gradient step and runs 384 back to
            #     back, so the acting thread can block a full optimizer step
            #     here — for a decoder that returns [].
            #   * `glue.augment_latent` is a GNN-gate forward whose only
            #     consumer is `extract_facts`, which on this path returns [].
            # So the gate removes a per-step lock handshake and a wasted
            # forward, and changes nothing about what is learned.
            #
            # AND ON A CADENCE WHEN IT IS ON: the KG consumers of `sd_facts`
            # read at `perception_interval`/`goal_interval` (50/25), so paying
            # for a fresh extraction every step bought nothing. `_sd_every`
            # keeps step 0 of every window so a cold start still primes the
            # discretizer and the heads on the first frame they see.
            _sd_now = (not self._skip_perdim_symbolic
                       and (self.total_timesteps % self._sd_every == 0))
            if _sd_now:
                with self._wm_param_lock:
                    with torch.no_grad():
                        augmented_primary = self.glue.augment_latent(
                            latent[0:1])
                    sd_facts = self.symbolic_decoder.extract_facts(
                        augmented_primary.squeeze(0),
                        timestep=self.total_timesteps
                    )
                    self.symbolic_decoder.update_discretizer(
                        next_obs_list[0])
                for fact in sd_facts:
                    if self.knowledge_graph.add_fact(fact):
                        symbolic_decoder_facts_count += 1

            if episode_length % 50 == 0 and not self._skip_perdim_symbolic:
                scene = self.fact_extractor.extract_scene_graph(
                    next_obs_list[0], timestep=self.total_timesteps
                )
                self.knowledge_graph.save_scene_snapshot(scene)

            _pt = self._phase_mark("world", _pt)

            # ---- 5. STORE EXPERIENCE (one stream per env) ----
            # Primary stream (0) gets vision-scaffold shaping folded into its
            # extrinsic (waking only); scouts store the raw env reward. The
            # WM reward head thus learns the true reward from 15 streams and
            # the shaped one from the primary — matching the single-env path.
            prim_extrinsic = rewards[0]
            self._lt_mark("shaping", intrinsic, prim_extrinsic, "env")
            _frame = None
            if not use_dream_actor and self.symbolizer is not None:
                try:
                    _frame = envs[0].render()   # symbolizer's VLM label frame
                except Exception:
                    _frame = None
            # MASTERY HABITUATION: a break of a long-mastered type damps this
            # step's whole intrinsic (see _habituation_factor) — applied
            # BEFORE the magnet so its LP attribution habituates too.
            if not use_dream_actor:
                _hab = self._habituation_factor(prim_info)
                if _hab < 1.0:
                    intrinsic[0] = intrinsic[0] * _hab
                    self._lt_mark("damp_habituation", intrinsic, prim_extrinsic, None, "habituation", _hab)
            # Curiosity-ranked magnet: reads the grounding head's per-object
            # probs on the primary latent + this step's learning progress
            # (intrinsic[0]) and steers toward the most-curious in-view object.
            if not use_dream_actor and self.vision_scaffold is not None:
                _sr = self._magnet_step_shaping(
                    env_actions[0], float(intrinsic[0].item()), latent[0:1])
                # OCCLUSION APPLIES TO THIS CHANNEL TOO (2026-08-17). The
                # gui_open guard further down zeroes `intrinsic` — but the
                # magnet's shaping is added to prim_EXTRINSIC, which that
                # guard never touches, so it kept paying through a menu.
                # MEASURED LIVE: SkyBot right-clicked a wandering villager,
                # opened its trade screen, and sat there for 9,125
                # CONSECUTIVE steps (gui 64% of the segment, position
                # frozen) while magnet_seek paid 77% of ALL income. This is
                # the 2026-08-02 occlusion farm exactly, one channel over:
                # inside a GUI the agent cannot move, look or swing, so
                # anything that pays there is paying for blindness.
                # Still CALLED while occluded — heartbeats, the fovea latent
                # refresh and signal-health observation must not stall —
                # just not PAID.
                _gui_now2 = bool((step_infos[0] or {}).get("gui_open"))
                if _gui_now2:
                    # seek/centring are POTENTIALS: left holding a menu
                    # frame's phi, the step the menu closes would collect a
                    # windfall for closing it. Re-adopt with no delta,
                    # exactly as reset() does at an episode boundary.
                    self.vision_scaffold._phi_prev = None
                    self.vision_scaffold._seek_prob_prev = None
                elif _sr:
                    # ---- FARM DAMPING (2026-09-04) -----------------------
                    # infra/farm named a live farm — 2,213 laps in one cell
                    # at +0.026/step — and NOTHING consumed it. Detection
                    # without response is how 77% of income once went to a
                    # villager menu and 96% of drive to the sky.
                    # SHAPING ONLY. rewards[0] is real environment income and
                    # is never damped: an agent that actually breaks a block
                    # in a cell it happens to have circled must still be paid
                    # in full, or this becomes a new way to be wrong about
                    # the scoreboard (§9).
                    # Not a latch — the multiplier is a pure function of lap
                    # count and income EMA, both of which decay once the
                    # agent stays away longer than revisit_horizon. See
                    # BehaviouralLoopDetector.loop_damp.
                    _damp = 1.0
                    if self.infra is not None:
                        _damp = float(getattr(
                            self.infra, "last_loop_damp", 1.0) or 1.0)
                    _sr = _sr * _damp
                    prim_extrinsic = rewards[0] + _sr
                    self._lt_mark("magnet", intrinsic, prim_extrinsic)
                # ---- APPROACH A REMEMBERED PLACE (2026-09-04) -----------
                # infra/episodic has recorded `sighting:tree_visible @(x,z)`
                # for the whole project and driven NOTHING, so the agent
                # could only pursue what was ON SCREEN — look away and the
                # tree stopped existing. See InfraStack._approach_step.
                # PLAIN DIFFERENCE (§4.3): pays w*(d_prev - d_now), which is
                # exactly 0.0 while stationary. The telescoping form would
                # pay a wage for holding still (`Gaze level: +0.00014/step`).
                # prim_EXTRINSIC, before the gui zeroing line (§4.4), and
                # not paid at all while occluded — inside a menu the agent
                # cannot walk, so anything paid there is paying for
                # blindness (the villager incident, 77% of income).
                if not _gui_now2 and self.infra is not None:
                    _ar = float(getattr(
                        self.infra, "last_approach_reward", 0.0) or 0.0)
                    if _ar:
                        prim_extrinsic = prim_extrinsic + _ar
                        self._lt_mark("approach", intrinsic, prim_extrinsic)
            # ---- GETTING OUT MUST PAY -----------------------------------
            # Zeroing income inside a menu removes the FARM but leaves
            # no gradient toward the exit — and it did something worse:
            # because pressing `inventory` OPENS a screen that pays 0,
            # the policy trained that action to ~zero probability
            # (measured: 0 presses in an entire run at 92%-of-max
            # entropy). So when a wandering villager's trade screen
            # opened via `use`, the agent had already unlearned the only
            # button that closes one, and sat there 10,149+ consecutive
            # steps. Guard-becomes-latch, again.
            # Plain-difference state cost on dwell: Phi = -min(1, run/N).
            # Sitting accrues the cost once; leaving collects it back;
            # an open/close cycle nets 0, so this cannot be farmed in
            # either direction. Paid into prim_EXTRINSIC on purpose —
            # the intrinsic channel is zeroed while a GUI is open, so a
            # cost placed there would be erased by the very guard it is
            # meant to complement.
            # NOT INSIDE THE VISION-SCAFFOLD BLOCK (2026-10-03, see
            # docs/foundation/BASELINE_AUDIT.md §3). It used to be, and the
            # live config runs `llm.vision.enabled: false` — so the scaffold
            # was None, stream 0 (the ONLY stream PPO trains on) paid NO
            # dwell cost, every scout (_scout_mixed_reward) did, and the
            # yaml's "a menu now pays strictly <= 0" rested on a term that
            # never ran. The gui flag is read from env info here — the same
            # source as the scout path and the intrinsic-zeroing guard — so
            # this runs exactly once per waking step, scaffold or not.
            if not use_dream_actor:
                prim_extrinsic += self._gui_reward(0, step_infos[0])
                self._lt_mark("gui_dwell", intrinsic, prim_extrinsic)
            # general infra: event monitors + empowerment shaping (shared
            # helper — see _infra_step)
            if not use_dream_actor:
                _ei = self._infra_step(
                    prim_info, env_actions[0], rssm_state,
                    float(intrinsic[0].item()) + float(prim_extrinsic),
                    float(rewards[0]))
                if _ei:
                    intrinsic[0] = intrinsic[0] + _ei
                    self._lt_mark("empowerment", intrinsic, prim_extrinsic)
                # ---- FARM DAMP ON THE CHANNEL THAT PAYS (2026-10-04) -----
                # last_loop_damp used to multiply only the magnet's `_sr`,
                # which is always 0 on the live config (llm.vision.enabled
                # false) — so a detected revisit loop kept its full
                # INTRINSIC income. Read AFTER _infra_step: on_step has just
                # scored THIS step's cell, and it was fed the UNDAMPED
                # income, so the damp cannot release itself by its own
                # effect. BEFORE the gui zeroing at the mix, which still
                # yields exactly 0 in a menu. Not a latch: see
                # _loop_damp_factor (leave the cell -> 1.0 again).
                _ldi = self._loop_damp_factor()
                if _ldi < 1.0:
                    intrinsic[0] = intrinsic[0] * _ldi
                    self._lt_mark("damp_farm", intrinsic, prim_extrinsic, None, "farm_damp", _ldi)

            # ---- VLM SYMBOLIC GROUNDING (primary stream, waking only) ----
            # The VLM names what the agent is looking at; a head learns to
            # predict those names from the agent's OWN latent (borrowed ->
            # owned). Perceptual facts are emitted only when a fresh label
            # lands, so KG traffic anneals with the VLM itself. Causal facts
            # come from real break events, never from the VLM's word.
            if not use_dream_actor and self.symbolizer is not None:
                _lat0 = latent[0:1].detach()
                # last frame, kept for the unstuck advisor: at stuck L3 the
                # VLM is shown what the agent currently sees (a reference
                # assignment per step, only read at L3 segments)
                self._advisor_frame = _frame
                self.symbolizer.maybe_label(_frame, _lat0, self._symbol_clock)
                # SCORE THE HEAD AGAINST TELEMETRY (2026-09-01). pitch,
                # gui_open and mainhand settle four predicates exactly, every
                # step, and were never compared — see observe_truth. This is
                # the half of the verifier that does not need the VLM.
                self.symbolizer.observe_truth(
                    (prim_info.get("world") or {}), _lat0)
                _labels = self.symbolizer.collect()
                if _labels is not None and self.infra is not None:
                    self.infra.beat("vlm_label", self.total_timesteps)
                if (self.infra is not None and self.symbolizer is not None
                        and getattr(self.symbolizer, "total_fovea_labels", 0)
                        != getattr(self, "_hb_fovea_seen", 0)):
                    self._hb_fovea_seen = self.symbolizer.total_fovea_labels
                    self.infra.beat("fovea_label", self.total_timesteps)
                # LAYER 3: route a VLM-PROPOSED category to the instinct. The
                # proposal is attention only — it goes to the magnet's
                # probation set, never to the KG. If
                # attending to it yields no learning progress it is evicted.
                if (_labels and self.vision_scaffold is not None
                        and _labels.get("__novel__")):
                    try:
                        self.vision_scaffold.propose_category(
                            str(_labels["__novel__"]), self.total_timesteps)
                    except Exception as _e:     # never kill a run over instinct
                        logger.warning("category proposal failed: %s", _e)
                if _labels is not None:
                    for _sf in self.symbolizer.perceptual_facts(_lat0):
                        # refresh=True: this channel may REVISE its own
                        # confidence downward. Without it add_fact ratchets
                        # to a high-water mark and a discredited predicate
                        # keeps its most-confident moment forever.
                        if self.knowledge_graph.add_fact(
                                _to_fact(_sf, "vlm_grounded",
                                         self.total_timesteps),
                                refresh=True):
                            symbolizer_facts_count += 1
                # ground truth: which block types were newly broken this step
                _ach = prim_info.get("achievements", {}) or {}
                _chopping = (self.is_discrete
                             and int(env_actions[0]) in self._chop_action_set)
                _broke_now = []
                for _k, _v in _ach.items():
                    if not _k.startswith("mine_"):
                        continue
                    _btype = _k[5:]
                    if int(_v) > int(self._prev_mine.get(_btype, 0)):
                        self._prev_mine[_btype] = int(_v)
                        _broke_now.append(_btype)
                        for _cf in self.symbolizer.observe_event(
                                "block_break", _btype,
                                was_chopping=_chopping,
                                now=self._symbol_clock):
                            if self.knowledge_graph.add_fact(_to_fact(
                                    _cf, "grounded_event",
                                    self.total_timesteps)):
                                symbolizer_facts_count += 1
                # ---- NEGATIVE EVIDENCE (2026-08-07): a full break-worth of
                # held attack with NOTHING breaking contradicts the claimed
                # affordance predicates. Edge-triggered once per streak; any
                # break (or letting go) re-arms the exam. This closes the
                # verifier's false-positive blind spot the module docstring
                # documented (a stuck-true predicate used to drift to
                # reliability 1.0 unopposed — llava's tree_visible did).
                if self._neg_evidence_ticks > 0:
                    _arun = float(prim_info.get("attack_run", 0.0) or 0.0)
                    if _broke_now or _arun <= 0.0:
                        self._neg_evt_scored = False
                    elif (_arun >= self._neg_evidence_ticks
                          and not getattr(self, "_neg_evt_scored", False)):
                        self._neg_evt_scored = True
                        self._neg_evt_total = getattr(
                            self, "_neg_evt_total", 0) + \
                            self.symbolizer.observe_negative_event(
                                now=self._symbol_clock)
                # ---- CREATURE MEANING, EARNED (2026-07-25) ----
                # The VLM only ever supplies a NAME from pixels. What a
                # creature MEANS is learned HERE, from what actually happens
                # to the agent: a creature in view when health drops becomes
                # causally linked to being hurt. Nothing declares "zombie =
                # hostile" — a cow that never hurts the agent never earns the
                # link; a creeper that does earns it fast. Uses the
                # damage/death signal the restored life-stats provide.
                _wi = (prim_info.get("world") or {})
                _dmg = float(_wi.get("damage", 0.0) or 0.0)
                if _dmg > 0.0 or _wi.get("died"):
                    from developmental_ai.llm.vlm_symbolizer import (
                        CREATURE_PREDICATES)
                    _seen = set()
                    try:
                        for _pf in self.symbolizer.perceptual_facts(_lat0):
                            _seen.add(str(_pf[2]))     # ("scene","contains",X)
                    except Exception:
                        pass
                    for _cp in CREATURE_PREDICATES:
                        _who = _cp.replace("_visible", "")
                        if _who not in _seen:
                            continue
                        _rel = "killed" if _wi.get("died") else "hurt"
                        if self.knowledge_graph.add_fact(_to_fact(
                                (_who, _rel, "agent", _CREATURE_FACT_CONF), "grounded_event",
                                self.total_timesteps), refresh=True):
                            symbolizer_facts_count += 1
                        logger.warning(
                            "creature meaning learned: %s %s agent "
                            "(damage=%.1f)", _who, _rel, _dmg)
                # ---- TOOL MEANING, EARNED (2026-07-25) ----
                # The agent SEES what it holds (holding_tool) and that the
                # thing wears out (tool_worn). What a tool is FOR is learned
                # HERE, from use:
                #   * a block falls while a tool is in hand -> that tool
                #     ENABLES that effect
                #   * the durability value rises as it works -> it WEARS DOWN
                #   * the tool leaves its hand entirely     -> it BREAKS
                # Nothing declares "an axe chops wood". If the agent fells a
                # log barehanded the axe earns no credit for it, and a tool
                # that never wears never earns `wears_down`.
                _tool = str(_wi.get("mainhand") or "")
                if _tool and _tool not in ("none", "air"):
                    for _bt in _broke_now:
                        if True:
                            if self.knowledge_graph.add_fact(_to_fact(
                                    (_tool, "enables", f"break_{_bt}", _TOOL_FACT_CONF),
                                    "grounded_event", self.total_timesteps),
                                    refresh=True):
                                symbolizer_facts_count += 1
                            logger.warning(
                                "tool meaning learned: %s enables break_%s",
                                _tool, _bt)
                    if _wi.get("tool_wore"):
                        if self.knowledge_graph.add_fact(_to_fact(
                                (_tool, "wears_down", "with_use", _TOOL_FACT_CONF),
                                "grounded_event", self.total_timesteps),
                                refresh=True):
                            symbolizer_facts_count += 1
                if _wi.get("tool_lost"):
                    _lost = str(_wi["tool_lost"])
                    if self.knowledge_graph.add_fact(_to_fact(
                            (_lost, "breaks_from", "use", _TOOL_FACT_CONF),
                            "grounded_event", self.total_timesteps),
                            refresh=True):
                        symbolizer_facts_count += 1
                    logger.warning("tool meaning learned: %s BROKE from use",
                                   _lost)

                # ---- RETRACTION: the other half of the gates ----
                # min_labels/reliability_floor only stop NEW facts. A
                # predicate the world has discredited must also stop feeding
                # its triple to the graph embedding. Placed after the break
                # events above so a collapse caused by THIS step retracts
                # immediately. stale_facts() is edge-triggered, so this is a
                # 16-entry scan per step and at most one KG write per
                # collapse; source-scoped, so it can only delete facts this
                # channel minted.
                for _stale in self.symbolizer.stale_facts():
                    try:
                        if self.knowledge_graph.remove_fact(
                                *_stale, source="vlm_grounded"):
                            self._symbolizer_retractions += 1
                            logger.info(
                                "Symbol grounding: retracted %s "
                                "(reliability fell below floor)", _stale)
                    except Exception as _e:   # never kill a multi-day run
                        logger.warning(
                            "KG retraction failed for %s: %s", _stale, _e)
                self._symbol_clock += 1

            _pt = self._phase_mark("vlm", _pt)

            if self.brain_emitter is not None:
                self._brain_clock += 1
                if self._brain_clock % self._brain_interval == 0:
                    self.brain_emitter.emit(self.total_episodes,
                                            self.total_timesteps)
            for e_i in range(n):
                self.replay_buffer.add(
                    obs_list[e_i], env_actions[e_i],
                    prim_extrinsic if e_i == 0 else rewards[e_i],
                    dones[e_i], stream=e_i,
                    # WHICH KIND OF `done` THIS IS (2026-09-01). Both the
                    # loop-level step timeout (`restarted`) and the
                    # adapter-caught crash (`env_restarted`) rebuild the
                    # CLIENT, so the frames either side are different worlds.
                    # The advantage trace wants them severed exactly like a
                    # terminal — which is why `dones` folds them together —
                    # but the world model must not learn the splice as
                    # dynamics, and must certainly not OVERSAMPLE it.
                    restart=bool(restarted[e_i]) or bool(
                        (step_infos[e_i] or {}).get("env_restarted")),
                    # THE BODY AS IT WAS WHEN obs_list[e_i] WAS SEEN, not as
                    # it is now. `step_infos` describes the state AFTER this
                    # step, so writing it here would pair o_t with proprio
                    # from t+1 and hand the world model a one-step lookahead
                    # on its own motion — the same off-by-one the causal
                    # action alignment exists to prevent.
                    proprio=(self._wm_proprio_prev[e_i]
                             if (self._wm_proprio_prev is not None
                                 and e_i < len(self._wm_proprio_prev))
                             else None),
                )
            # Advance the one-step body memory AFTER the write.
            if self._wm_proprio_prev is not None:
                for e_i in range(n):
                    _si = step_infos[e_i] or {}
                    self._wm_proprio_prev[e_i] = (
                        _si.get("sensors")
                        if _si.get("sensors") is not None
                        else _si.get("proprio"))
            # ---- SUB-PHASING `store` (2026-09-20) --------------------
            # The measured profile put 287 ms/step (21.6%) in `store`, which
            # is not explicable from reading this window: it writes one
            # replay row per env and appends a few numpy arrays to the PPO
            # rollout lists. Rather than guess (CLAUDE.md 5), split the
            # bucket and let the next profile name the cost.
            _pt = self._phase_mark("store_buf", _pt)

            # Primary stream feeds the on-policy PPO rollout (waking only).
            if not use_dream_actor and primary_pol is not None:
                # ---- OCCLUSION SUPPRESSION MUST PRECEDE THE MIX ----------
                # (fix 2026-08-02, found live on the server.) The gui_open
                # zeroing sat ~40 lines BELOW this call — AFTER `mixed` was
                # computed and AFTER the transition was stored — so it only
                # ever changed the logged `episode_intrinsic` and NEVER
                # reached the PPO reward. The guard was inert from the day it
                # was written, in BOTH loop paths.
                #
                # Measured live: the inventory screen paid +0.0410/step vs
                # +0.0196 for the world (2.09x), and the agent sat in the menu
                # 89% of a segment with an unbroken 1491-step dwell —
                # motionless, because inside a GUI it cannot move, look or
                # swing. Curiosity was paying for the act of occluding the
                # camera, which is exactly what the note below forbids.
                #
                # The menu stays fully available and crafting still earns its
                # extrinsic tier; only payment for the occlusion is removed.
                if bool((step_infos[0] or {}).get("gui_open")):
                    intrinsic[0] = intrinsic[0] * 0.0
                    self._lt_mark("gui_zero", intrinsic, prim_extrinsic, None, "gui_zero", 0.0)
                mixed = self.reward_mixer.mix(
                    float(intrinsic[0].item()), prim_extrinsic)
                self._lt_commit(mixed, intrinsic, prim_extrinsic)
                # running "typical surprise" baseline — the denominator of
                # mastery's WM-fidelity ratio. Updated EVERY primary step
                # (not just during options), or the baseline would be biased
                # toward exactly the moments options select for.
                #
                # THE RAW FORWARD-MODEL ERROR, not `intrinsic` (review HIGH,
                # flagged independently by four reviewers). `intrinsic` is
                # the SHAPED curiosity reward: mean-centred, clipped, and
                # carrying learning-progress / imagination / coverage
                # bonuses. Feeding that to "can the world model predict this
                # skill" measured a different variable — a skill could look
                # unpredictable purely because it was novel, or predictable
                # because its LP had flattened. `last_pred_error` is the
                # pre-normalization MSE the ICM already computes.
                _err_now = float(getattr(self.curiosity, "last_pred_error",
                                         float(intrinsic[0].item())))
                # ---- OCCLUSION: NO WORLD CHANGE, NO WORLD CURIOSITY ------
                # MEASURED (2026-07-27): the inventory screen paid the agent
                # +0.18 -> +0.30 intrinsic per step against +0.09 for the
                # world — a 2.0x -> 3.1x advantage that WIDENED over the run.
                # The agent was not failing to learn; it was optimal under
                # the signal it was given, and spent ~50% of every episode
                # in a menu.
                #
                # This is a CATEGORY ERROR, not a preference. The ICM scores
                # prediction error on the OBSERVATION. When a menu covers the
                # screen the observation changes hugely while the WORLD does
                # not change at all, so curiosity — which exists to reward
                # learning about world dynamics — is paying for an overlay
                # drawn over the world. It is the same argument already
                # accepted for the attack counter: a swing into a menu never
                # reached the world, so counting it was a measurement error.
                # Applying it to actions but not to perception was
                # inconsistent.
                #
                # NOT a rule about what the agent should want: the menu stays
                # fully available, and CRAFTING in it still pays the +5.0
                # extrinsic first-craft tier. What is removed is payment for
                # the act of occluding the camera.
                #
                # (2026-08-02) The zeroing now happens BEFORE the mix, above.
                # By the time control reaches here intrinsic[0] is already 0 on
                # a GUI frame, so re-multiplying by 0.0 was a no-op left behind
                # by that fix. Removed; the reasoning above is the load-bearing
                # part and stays.
                # ---- IS THE MENU A CURIOSITY FARM? (2026-07-27) ----------
                # The agent spends ~50% of every episode in the inventory
                # screen. I have been calling that "a learning result" with
                # no evidence. It is a MEASURABLE question: split the
                # intrinsic reward by whether the world was covered by a
                # menu at the time. If menu frames pay MORE, the agent is
                # behaving optimally under the signal it is given and the
                # signal is what needs looking at -- not the behaviour.
                try:
                    _gui_now = bool((step_infos[0] or {}).get("gui_open"))
                    # THE RAW forward-model surprise, NOT the post-
                    # suppression intrinsic. Measuring the value we just
                    # zeroed makes the readout tautological (`in-menu=0.0000`
                    # by construction) and destroys the falsification test
                    # this line exists to provide. What we want to know is
                    # whether the menu WOULD still pay better — i.e. whether
                    # the underlying category error is still there — which
                    # only the pre-suppression error can answer.
                    _iv = _err_now
                    if not hasattr(self, "_gui_intr"):
                        self._gui_intr = [0.0, 0, 0.0, 0]   # sum,n | sum,n
                    if _gui_now:
                        self._gui_intr[0] += _iv
                        self._gui_intr[1] += 1
                    else:
                        self._gui_intr[2] += _iv
                        self._gui_intr[3] += 1
                except Exception:
                    pass
                self._wm_err_ema = (
                    _err_now if not hasattr(self, "_wm_err_ema")
                    else 0.999 * self._wm_err_ema + 0.001 * _err_now)
                if self.option_executor is not None:
                    _rt0 = self.option_executor.runtimes[0]
                    if _rt0.active:
                        # inside an option: accumulate; store ONE SMDP
                        # decision when it closes
                        _closed = self.option_executor.observe_primary(
                            float(rewards[0]), bool(dones[0]), mixed,
                            self.total_timesteps, wm_err=_err_now)
                        if _closed is not None:
                            self.policy.store_transition(
                                _closed["obs"], _closed["meta_action"],
                                _closed["reward"], dones[0],
                                _closed["log_prob"], _closed["value"],
                                knowledge=_closed["kv"],
                                tau=_closed["tau"],
                                action_mask=_closed["mask"],
                                proprio=_closed.get("proprio"),
                                feats=_closed.get("feats"))
                            self.skill_bank.record_invocation(
                                _closed["skill_id"], _closed["outcome"],
                                _closed["reward"])
                            self._competence_from_option(_closed)
                            # WM fidelity (task #39): this invocation's mean
                            # surprise vs the agent's typical surprise. The
                            # asked/success half is recorded by the executor
                            # itself at every frame close (incl. scouts and
                            # nested frames) — only the fidelity ratio needs
                            # the loop's global baseline.
                            _wem = _closed.get("wm_err_mean")
                            if (_wem is not None
                                    and getattr(self, "_wm_err_ema", 0) > 1e-9):
                                self.skill_bank.record_wm_fidelity(
                                    _closed["skill_id"],
                                    float(_wem) / float(self._wm_err_ema))
                    else:
                        _pp = self.option_executor._primary_primitive
                        if _pp is not None:
                            self.policy.store_transition(
                                obs_list[0], _pp["action"], mixed,
                                dones[0], _pp["log_prob"], _pp["value"],
                                knowledge=primary_kv, tau=1,
                                action_mask=_pp["mask"],
                                proprio=_pp.get("proprio"),
                                feats=_pp.get("feats"))
                else:
                    self.policy.store_transition(
                        obs_list[0], env_actions[0], mixed, dones[0],
                        primary_pol["log_prob"], primary_pol["value"],
                        knowledge=primary_kv, feats=self._feats_row(0))

            # ---- 5b. SCOUT STREAMS FEED PPO TOO (2026-09-01) -------------
            # Every scout decision is the SHARED policy choosing under a
            # known mask, so it is on-policy experience and there was never a
            # reason for it to be invisible to the update — it was simply
            # never plumbed. Two things it fixes at once: the fleet's whole
            # wall clock now produces policy gradient (on a 2-client fleet
            # that is 2x the data for 0 extra Minecraft clients, and the
            # server's client count is the binding constraint), and the row
            # starvation that made the update degenerate to a single
            # full-batch step goes away with it.
            # GAE segments rows by stream, so interleaving is safe; see
            # _compute_gae.
            if (not use_dream_actor and self._scouts_in_ppo and n > 1):
                _sc_mixed = [0.0] * n
                for _e in range(1, n):
                    _sc_mixed[_e] = self._scout_mixed_reward(
                        _e, step_infos[_e] or {}, float(rewards[_e]),
                        float(intrinsic[_e].item()))
            if (not use_dream_actor and self._scouts_in_ppo and n > 1
                    and self.option_executor is None):
                # OPTIONS DISABLED: every scout decision is a plain primitive
                # and its policy_info was kept in the act branch above.
                for _e in range(1, n):
                    _pi = (getattr(self, "_noopt_pol", None) or [None] * n)[_e]
                    if _pi is None:
                        continue
                    self.policy.store_transition(
                        obs_list[_e], env_actions[_e], _sc_mixed[_e],
                        dones[_e], _pi["log_prob"], _pi["value"],
                        knowledge=None, tau=1, stream=_e,
                        feats=self._feats_row(_e))
            elif (not use_dream_actor and self.option_executor is not None
                    and self._scouts_in_ppo and n > 1):
                _sc_closed = self.option_executor.observe_scouts(
                    rewards, dones, self.total_timesteps,
                    mixed_rewards=_sc_mixed)
                for _e in range(1, n):
                    _cl = _sc_closed.get(_e)
                    if _cl is not None:
                        # SMDP row: one decision spanning tau env steps
                        self.policy.store_transition(
                            _cl["obs"], _cl["meta_action"], _cl["reward"],
                            dones[_e], _cl["log_prob"], _cl["value"],
                            knowledge=_cl["kv"], tau=_cl["tau"],
                            action_mask=_cl["mask"],
                            proprio=_cl.get("proprio"), stream=_e,
                            feats=_cl.get("feats"))
                        # stream=_e keeps `option_spike_ema` primary-only, so
                        # paging and the competence gate keep reading the
                        # same quantity they always have (see
                        # SkillBank.record_invocation)
                        self.skill_bank.record_invocation(
                            _cl["skill_id"], _cl["outcome"], _cl["reward"],
                            stream=_e)
                        continue
                    _sp = self.option_executor._scout_primitive[_e]
                    if _sp is not None:
                        # a record is present only when a decision was MADE
                        # this step (act() clears them), so a scout inside an
                        # option stores nothing here — its reward is being
                        # accumulated into the option's return instead
                        self.policy.store_transition(
                            obs_list[_e], _sp["action"], _sc_mixed[_e],
                            dones[_e], _sp["log_prob"], _sp["value"],
                            knowledge=None, tau=1,
                            action_mask=_sp["mask"],
                            proprio=_sp.get("proprio"), stream=_e,
                            feats=_sp.get("feats"))

            # ---- 6. ADVANCE (lifelong: DEATH-ONLY reset; no episode) ----
            # A time-truncation carries on unbroken. Only a true terminal
            # (64-log success) or a crash-restart zeroes that stream. NOTE: in
            # lifelong the MineRL adapter emits NO time-truncation (TimeLimit is
            # removed), so a truncated=True from a MineRL step is ALWAYS a
            # crash-restart the adapter caught internally (it rebuilt the CLIENT
            # but the mission only launches inside reset()) — that stream must be
            # reset here or it wedges on a stale frame forever (audit critical).
            _pt = self._phase_mark("store_ppo", _pt)
            # FOUNDATION SHADOW (see the episodic body). Lifelong: a stream's
            # episode ends only on its OWN boundary; there is no fleet reset.
            _sh_fleet_reset = False
            if self._shadow is not None:
                self._shadow.after_step(
                    step_infos, env_actions, rewards, dones, restarted,
                    _sh_ends, _t_env0, prim_extrinsic, intrinsic,
                    _sh_fleet_reset, use_dream_actor, observations=next_obs_list)
                _pt = self._phase_mark("shadow", _pt)
            _reset_to = float(self.config.get("parallel_envs", {}).get(
                "reset_timeout_s", 300))
            for e_i in range(n):
                _crashed = bool(step_infos[e_i]) and step_infos[e_i].get(
                    "env_restarted")
                if dones[e_i] or _crashed:
                    self._ll.death_reset(e_i)        # zero h/z row + mask row
                    self._drop_curiosity_batch()
                    # A POTENTIAL MUST NOT CROSS A WORLD BOUNDARY. Every
                    # episode is a fresh world here, so a phi carried over
                    # pays a phantom delta on the first step of the new life
                    # — e.g. a swing streak that "ended" because the world
                    # was rebuilt would be charged back as if the agent had
                    # let go. Same reason the vision scaffold re-adopts its
                    # phi with no delta at a boundary.
                    self._sc_phi.pop(e_i, None)
                    if e_i == 0:
                        # STREAM 0'S POTENTIALS ARE PLAIN SCALARS, not rows
                        # in `_sc_phi`, so the pop above misses them entirely
                        # — the same boundary bug the line above fixes for
                        # scouts, left in place for the one stream that
                        # matters most. Re-adopt with NO delta: 0.0 for the
                        # two progress potentials, and the None sentinel each
                        # cost potential already uses to mean "first sample,
                        # charge nothing" (see _pitch_level_phi / the gui
                        # dwell block). Without this the first step of a new
                        # life collects a phantom refund for a swing streak
                        # that only ended because the client rebuilt.
                        self._persist_phi = 0.0
                        self._reach_phi = 0.0
                        self._pitch_level_phi = None
                        self._gui_run = 0
                        self._gui_dwell_phi = None
                        # Same class, one dict over: mine_* counters restart
                        # at 0 in a new world, so a stale high-water mark
                        # makes `v > prev` never fire and habituation
                        # silently stops damping. Latent on MineRL (the
                        # adapter emits the typed `events` stream, so the
                        # legacy branch never runs) — wrong anywhere else.
                        self._habit_prev.clear()
                    self._mine_by_env.pop(e_i, None)
                    self._new_break_by_env.pop(e_i, None)
                    if hasattr(self, "_loginv_by_env"):
                        self._loginv_by_env.pop(e_i, None)
                    # fresh world: mine_* counts restart at 0, so the primary's
                    # symbolizer high-water must restart too — else no causal
                    # block_break facts (and no tree_visible reliability hits)
                    # until the new life re-exceeds the old life's totals
                    # (audit fix; mirrors the episodic paths' clears).
                    if e_i == 0 and self.symbolizer is not None:
                        self._prev_mine.clear()
                    if self.option_executor is not None:
                        self.option_executor.clear(e_i, self.total_timesteps)
                    if restarted[e_i] and not _crashed:
                        # loop-level step TIMEOUT: the step future may still be
                        # in flight, so a reset here could race it on the same
                        # client — carry the last obs (rare; the adapter-caught
                        # crash below is the near-certain path).
                        obs_list[e_i] = next_obs_list[e_i]
                    else:
                        # 64-log SUCCESS or adapter-caught CRASH-REBUILD: the
                        # client needs a fresh mission (launched in reset()).
                        # BOUNDED so a hung reset cannot freeze the whole fleet.
                        try:
                            _rf = self._env_pool.submit(envs[e_i].reset)
                            o = _rf.result(timeout=_reset_to)[0]
                            obs_list[e_i] = np.asarray(o, dtype=np.float32)
                        except Exception as _re:
                            logger.warning(
                                "env %d bounded reset failed (%s) — carrying "
                                "obs; stream may lag one segment", e_i, _re)
                            obs_list[e_i] = next_obs_list[e_i]
                else:
                    obs_list[e_i] = next_obs_list[e_i]  # (no truncation in LL)

            _pt = self._phase_mark("store_reset", _pt)
            # A2: hand this step's encoder output to the next act(). MUST be
            # after the advance loop above — validity is decided by whether
            # obs_list[e] IS next_obs_list[e] (same object = the env kept
            # running), which is only true once that loop has assigned.
            self._set_enc_carry(encoded_next, obs_list, next_obs_list)

            episode_reward += rewards[0]
            episode_intrinsic += float(intrinsic[0].item())
            episode_length += 1
            self.total_timesteps += n  # all N transitions are real experience
            _pt = self._phase_mark("store", _pt)

            # ---- lifelong cadences (step-based, not episode-based) ----
            self._ll.wm_steps += 1
            if (self._ll.wm_steps >= wm_train_every
                    and self.replay_buffer.is_ready):
                self._ll.wm_steps = 0
                if self._use_async_wm:
                    # RETROSPECTIVE-ASYNC: consolidation runs on the trainer
                    # thread while acting continues. Skip-if-busy; the
                    # busy-check/put window can let AT MOST one extra ticket
                    # slip through (trainer sets busy after get) — harmless:
                    # one extra back-to-back block on fresh data.
                    self._ensure_wm_trainer()
                    self._wm_cadence_hits += 1
                    if (not self._wm_busy.is_set()
                            and self._wm_ticket.empty()):
                        try:
                            self._wm_ticket.put_nowait(1)
                            self._wm_blocks_run += 1
                        except Exception:
                            pass
                    if self._wm_metrics_box is not None:
                        # consume-once: each finished block feeds run()'s
                        # stage-controller/glue consumers exactly once
                        # (review fix — stale-metrics slope flattening)
                        self._last_wm_metrics = self._wm_metrics_box
                        self._wm_metrics_box = None
                        self._wm_metrics_fresh = True
                else:
                    self._last_wm_metrics = self._train_world_model()
                    self._wm_metrics_fresh = True
                    # producer site 2 of 3: lifelong, synchronous trainer
                    self._record_wm_telemetry(self._last_wm_metrics)
            self._ll.goal_steps += 1
            if (self._ll.goal_steps >= goal_horizon
                    and self.broadcaster is not None):
                # goal-horizon: competence Bernoulli update + IMGEP retarget +
                # mask hard-clear (the mask has been softly fading each segment)
                self._ll.goal_steps = 0
                self.broadcaster.reset()
                _ls = getattr(self.broadcaster, "last_selection", None)
                if _ls:
                    print(f"  Prospection re-rank: {_ls}", flush=True)
                    self.broadcaster.last_selection = None
                if self._use_async_wm and self._wm_cadence_hits:
                    _skip = 1.0 - self._wm_blocks_run / self._wm_cadence_hits
                    _bms = float(getattr(self, "_wm_block_ms", 0.0))
                    _bit = int(getattr(self, "_wm_block_iters", 0) or 0)
                    _pool = int(getattr(self, "_wm_reward_pool", -1))
                    print(f"  AsyncWM replay-ratio: {self._wm_blocks_run}"
                          f"/{self._wm_cadence_hits} blocks run "
                          f"(skip {_skip:.1%})"
                          + (f" | block {_bms:.0f} ms / {_bit} iters "
                             f"({_bms / max(1, _bit):.1f} ms/iter)"
                             if _bit else "")
                          # THE GOAL POOL. Printed next to the block timing
                          # because that is where someone reading the log is
                          # already looking, and because a pool of ~0 means
                          # goal_replay_fraction is buying nothing no matter
                          # what it is set to.
                          + (f" | reward pool {_pool} windows"
                             if _pool >= 0 else ""), flush=True)
                    # ---- train_iters FROM MEASURED SKIP (2026-09-01) ------
                    # `train_iters: 384` is a NOMINAL figure. With the async
                    # trainer's skip-if-busy, the ratio the GPU actually
                    # sustains is 384 x (1 - skip) — so the config has been
                    # stating an intent while the hardware decided the real
                    # number, and nothing reconciled the two. This closes the
                    # loop: high skip means the block is longer than the
                    # cadence, so shorten it and run MORE blocks on FRESHER
                    # data; low skip means there is headroom.
                    #
                    # SHIPPED OFF. It changes the world model's data diet,
                    # which is the one thing in this wave that cannot be
                    # judged from a smoke test — same discipline as
                    # curiosity.train_every. Turn it on with a cluster to
                    # watch, alone, and read `wm/loss` for three horizons.
                    _ai = self._train_iters_auto
                    if _ai.get("enabled") and self._wm_cadence_hits >= 4:
                        _tgt = float(_ai.get("target_skip", 0.2))
                        _cur = int(self._wm_train_iters)
                        # proportional, and deliberately gentle: a 2x lurch in
                        # replay ratio between horizons would confound every
                        # other measurement in the run.
                        _new = _cur * (1.0 - 0.25 * (_skip - _tgt) /
                                       max(0.05, 1.0 - _tgt))
                        _new = int(max(int(_ai.get("min_iters", 32)),
                                       min(int(_ai.get("max_iters", 512)),
                                           round(_new))))
                        if _new != _cur:
                            print(f"  AsyncWM train_iters: {_cur} -> {_new} "
                                  f"(skip {_skip:.1%} vs target {_tgt:.0%})",
                                  flush=True)
                            self._wm_train_iters = _new
                    self._wm_cadence_hits = 0
                    self._wm_blocks_run = 0
                # Re-open the env-side first-break tiers on the SAME horizon
                # (audit fix): without this the broadcaster re-armed goal masks
                # that could never re-unlock (first-break rewards had become
                # once-per-LIFETIME), driving non-log competence to zero.
                for _e in envs:
                    if hasattr(_e, "clear_break_marks"):
                        _e.clear_break_marks()

        self._stream_obs = obs_list
        self._ll.decay()                          # per-segment soft leak

        # PPO update: bootstrap with V(s_last) because the segment ends
        # MID-trajectory (the rollout is not a terminated episode).
        policy_loss = 0.0
        if self.policy.should_update(self.policy_update_steps):
            _pt_ppo = time.perf_counter()
            # ---- ONE BOOTSTRAP PER STREAM (2026-09-01) -------------------
            # The segment ends MID-trajectory for every body, not just the
            # primary, so each stream needs V(s_last) from ITS OWN last
            # observation. Giving the scouts 0.0 would tell the estimator
            # their future is worthless and bias every advantage on those
            # rows downward — the same truncation bias `last_value` was
            # introduced to remove for stream 0.
            # knowledge=None for scouts: that is what they ACTED under, and
            # the bootstrap has to be computed on the same input as the
            # values it is being compared against.
            # arch='rssm': bootstrap from the CURRENT belief (the RSSM state
            # the next segment will resume from), not from a re-encode of the
            # last observation — which for a recurrent state is not even
            # defined. `_bootstrap_latent` is set right after the last
            # observe_step below.
            _blat = getattr(self, "_bootstrap_latent", None)
            last_v = {0: self.policy.compute_last_value(
                obs_list[0], knowledge=self._current_knowledge_feature(),
                feats=(None if _blat is None
                       else _blat[0].detach().cpu().numpy()))}
            if self._scouts_in_ppo:
                for _e in range(1, n):
                    last_v[_e] = self.policy.compute_last_value(
                        obs_list[_e], knowledge=None,
                        feats=(None if _blat is None or _e >= _blat.shape[0]
                               else _blat[_e].detach().cpu().numpy()))
            pm = self.policy.train_step(
                n_epochs=self.config.get("policy", {}).get("n_epochs", 10),
                last_value=last_v)
            policy_loss = pm.get("policy_loss", 0.0)
            self._last_ppo = dict(pm)      # for the entropy forensics line
            # COLLAPSE DETECTOR, in the path that actually runs (2026-08-02).
            # PPO returns entropy and the episodic path recorded it; this one
            # dropped it, so the "<-- COLLAPSED" warning could never fire for
            # skybot. It went unseen while the action distribution walked from
            # attack:11% to attack:70% over three segments.
            if "entropy" in pm:
                self.training_metrics.setdefault(
                    "policy_entropy", deque(maxlen=100)).append(
                        float(pm["entropy"]))
            self._phase_mark("ppo", _pt_ppo)

        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "intrinsic_reward": episode_intrinsic / max(1, episode_length),
            "policy_loss": policy_loss,
            "curiosity_loss": icm_metrics.get("icm_total", 0),
            "new_facts": new_facts_count,
            "symbolic_decoder_facts": symbolic_decoder_facts_count,
        }

    # -----------------------------------------------------------------------
    # Symbolic fact extraction
    # -----------------------------------------------------------------------

    def _extract_and_store_facts(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        reward: float,
    ) -> int:
        """
        Extract symbolic facts from a transition and store in knowledge graph.

        Returns the number of NEW (novel) facts added.
        """
        # Extract transition facts (state changes + action rules)
        facts, action_rule = self.fact_extractor.extract_transition_facts(
            obs=obs,
            action=action,
            next_obs=next_obs,
            reward=reward,
            timestep=self.total_timesteps,
            action_labels=self.action_labels,
        )

        # Store facts in knowledge graph
        novel_count = 0
        for fact in facts:
            if self.knowledge_graph.add_fact(fact):
                novel_count += 1

        # Store action rule if discovered
        if action_rule is not None:
            self.knowledge_graph.add_action_rule(action_rule)

        return novel_count

    # -----------------------------------------------------------------------
    # World model training
    # -----------------------------------------------------------------------

    # Mapping VERIFIED against rssm.compute_loss's returned dict
    # {total, reconstruction, kl, reward, continue, inverse} and against
    # symbolic_decoder.train_step's keys — read, not assumed. Four counters
    # shipped broken this week from guessed names hidden by a bare `except`.
    _WM_TELEMETRY = (
        ("world_model_loss", "total"),
        ("kl_divergence", "kl"),
        ("reconstruction_error", "reconstruction"),
        ("inverse_dynamics_loss", "inverse"),
        ("flow_loss", "flow"),
        ("flow_mask", "flow_mask"),
        ("horizon_loss", "horizon"),
        ("slot_loss", "slot"),
        ("symbolic_decoder_loss", "symbolic_decoder_loss"),
        ("symbolic_decoder_accuracy", "symbolic_decoder_accuracy"),
    )

    def _record_wm_telemetry(self, m: Optional[Dict[str, float]]) -> None:
        """Record world-model losses for the logs/metrics sink.

        WHY THIS EXISTS (2026-09-04). Telemetry used to be appended by the
        SEGMENT CONSUMER in run(), which reads the losses through a
        consume-once relay (_wm_metrics_box -> _last_wm_metrics ->
        _wm_metrics_fresh). That relay exists for the stage controller, which
        must never see the same reconstruction value twice or its error slope
        flattens. But the main thread only checks the box inside
        `wm_steps >= wm_train_every AND _use_async_wm` — one instant every 250
        steps — while a training block takes ~71.6s against a ~73.5s cadence.
        The gate routinely missed, `_wm_metrics_fresh` was cleared by whoever
        did see it, and the deque stayed EMPTY.

        MEASURED CONSEQUENCE: `_log_progress` printed its `.get(..., 0)`
        default — `WM loss: 0.0000` in 24 of 24 readings — and the metrics
        sink emitted None, while the trainer was demonstrably running 32/32
        blocks over a 150k-transition buffer. A learning world model was
        indistinguishable from a dead one, and the dream, prospection, the
        policy's latents and empowerment all sit downstream of it.

        Telemetry must not ride on a relay whose job is something else. This
        is called at each of the three sites where metrics are PRODUCED —
        async trainer thread, lifelong-synchronous, and non-lifelong — so
        there is exactly one writer per producer and no double counting.
        Safe from the trainer thread: deque.append is atomic under the GIL.
        """
        if not m:
            return
        for _key, _mkey in self._WM_TELEMETRY:
            _v = m.get(_mkey)
            if _v is None:
                continue
            try:
                _v = float(_v)
            except (TypeError, ValueError):
                continue
            if _v != _v:                      # NaN: recording it would make
                continue                      # every downstream mean NaN
            # setdefault, NOT direct indexing: training_metrics is a plain
            # dict literal, so a key added to _WM_TELEMETRY without being
            # declared there would raise KeyError ON THE TRAINER THREAD,
            # where the traceback is invisible and the effect is a silently
            # dead trainer. Every key above is declared; this makes the
            # failure mode impossible rather than merely absent today.
            self.training_metrics.setdefault(
                _key, deque(maxlen=100)).append(_v)

    def _train_world_model(self) -> Dict[str, float]:
        """
        Train the RSSM world model AND the symbolic decoder on replay buffer data.

        Runs multiple gradient steps per call (train_iters) to give the world
        model enough signal to learn dynamics. DreamerV3 typically does a 1:1
        ratio of real steps to gradient steps; with our episodic training loop
        we compensate by doing multiple batches per episode.

        The symbolic decoder piggybacks on the last batch — we reuse
        the latent states the RSSM just computed.
        """
        wm_cfg = self.config.get("world_model", {})
        batch_size = wm_cfg.get("batch_size", 16)
        seq_len = wm_cfg.get("sequence_length", 50)
        # LIVE value, not the config literal: `_wm_train_iters` is seeded from
        # config and may be adapted from the measured async skip rate (see the
        # goal-horizon block). Reading wm_cfg here would have made that
        # adaptation a no-op that still printed as if it worked.
        train_iters = self._wm_train_iters
        prioritized = self._use_per
        # ---- TERMINAL OVERSAMPLING IS NOW A KNOB (2026-09-01) ------------
        # This call never passed `terminal_fraction`, so it silently used the
        # sampler's 0.25 default — tuned for EPISODIC runs where terminals are
        # common. In lifelong mode `reset_on_death_only` is true, so a
        # terminal is a death: rare by design. A quarter of every batch was
        # being spent on them (see the sampler for the 768-draws-per-block
        # arithmetic). The sampler now also refuses to duplicate a small pool,
        # so this fraction is a CEILING rather than a quota.
        terminal_fraction = float(wm_cfg.get("terminal_fraction", 0.25))
        # ---- GROW THE BUFFER HERE, NOT ON THE ACTING THREAD --------------
        # This is the "async learning period between intervals": the WM block
        # is already off the step loop (async trainer) or already a pause in
        # it (sync), so a block allocation costs nothing that is not already
        # being spent. add() only ever sets a flag; this is where it lands.
        try:
            self.replay_buffer.maybe_grow()
        except Exception as _ge:      # never let sizing kill a training block
            logger.warning("replay growth check failed: %s", _ge)

        def _do_train(batch: Dict[str, Any]) -> Dict[str, float]:
            """Run one gradient step on a sampled batch, updating PER
            priorities from the per-sequence reconstruction error when PER
            is enabled.

            _wm_param_lock at PER-GRADIENT-STEP granularity: with the async
            trainer, the main thread's per-step observe forwards / prospection
            rollouts / checkpoint saves wait at most ONE optimizer step, and
            never see torn parameters mid-update. Uncontended (sync mode) the
            RLock costs ~1us per iteration.
            """
            with self._wm_param_lock:
                if prioritized:
                    m, errs = self.world_model.train_step(
                        observations=batch["observations"],
                        actions=batch["actions"],
                        rewards=batch["rewards"],
                        continues=batch["continues"],
                        importance_weights=batch["weights"],
                        return_per_sample=True,
                        # ABSENT when the buffer has no body column, which is
                        # exactly what a model built with proprio_dim=0
                        # expects; a model built WITH one reads neutral zeros
                        # rather than raising (see WorldModel.embed).
                        proprio=batch.get("proprio"),
                    )
                    self.replay_buffer.update_priorities(
                        batch["start_indices"], seq_len, errs
                    )
                    self._lt_wm_step(batch)     # telemetry: read-only
                    return m
                m = self.world_model.train_step(
                    observations=batch["observations"],
                    actions=batch["actions"],
                    rewards=batch["rewards"],
                    continues=batch["continues"],
                    proprio=batch.get("proprio"),
                )
                self._lt_wm_step(batch)         # telemetry: read-only
                return m

        try:
            metrics = {}
            batch = None

            # ---- HOW LONG A BLOCK ACTUALLY TAKES (2026-09-01) ------------
            # The replay buffer became block-storage this wave so long runs
            # stop dying on memory. Block indexing is on this exact path
            # (batch_size x train_iters gathers per block), so the change
            # could have paid for the memory fix with throughput — the thing
            # the wave existed to improve. Timed, not assumed: if ms/iter
            # rises against the pre-growth run, the fix is FEWER, LARGER
            # blocks (raise world_model.buffer_growth.block_transitions),
            # and the knob already exists.
            _t_wm = time.perf_counter()
            if self._use_async_replay:
                # Prefetch all batches for this call on a background thread so
                # sampling overlaps with the (CPU/GPU-bound) gradient steps.
                sampler = self._get_bg_sampler(batch_size, seq_len, prioritized)
                sampler.request(train_iters)
                for _ in range(train_iters):
                    batch = sampler.get()
                    metrics = _do_train(batch)
            else:
                for _ in range(train_iters):
                    batch = self.replay_buffer.sample_sequences(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        device=self.device,
                        prioritized=prioritized,
                        reward_fraction=self._goal_replay_fraction,
                        reward_threshold=self._goal_replay_threshold,
                        terminal_fraction=terminal_fraction,
                    )
                    metrics = _do_train(batch)
            self._wm_block_ms = 1000.0 * (time.perf_counter() - _t_wm)
            self._wm_block_iters = int(train_iters)
            # HOW MANY GOAL WINDOWS THE STRATUM ACTUALLY HAD. This is the
            # measurement that says whether goal_replay_threshold changed
            # anything: under the old 1e-3 default on the primary stream this
            # was approximately "every window in the buffer"; log-sized it
            # should be a countable number of real breaks.
            if batch is not None:
                self._wm_reward_pool = int(batch.get("reward_pool", -1))

            # Train the symbolic decoder on the last batch (same lock: the
            # sd heads are read on the main thread every step via the glue)
            with self._wm_param_lock:
                with torch.no_grad():
                    states, _ = self.world_model.observe_sequence(
                        batch["observations"], batch["actions"],
                        batch.get("proprio"),
                    )
                    latent_states = torch.cat(
                        [states["h"], states["z"]], dim=-1)

                sd_metrics = self.symbolic_decoder.train_step(
                    latent_states=latent_states,
                    raw_observations=batch["observations"],
                )
            metrics.update(sd_metrics)

            # An empty dict here means the gradient loop above never executed
            # its body — train_iters <= 0, or the sampler yielded nothing —
            # and returns EXACTLY what the exception path returns. Those are
            # different failures and the caller (`if m:`) cannot tell them
            # apart, so say which one this is. Without this, "no metrics" has
            # two silent causes and diagnosing it means guessing.
            if not metrics:
                self._wm_empty_count = getattr(self, "_wm_empty_count", 0) + 1
                if (self._wm_empty_count <= 3
                        or self._wm_empty_count % 50 == 0):
                    logger.warning(
                        "World model block produced NO metrics without "
                        "raising (block #%d): train_iters=%d, batch=%s. The "
                        "gradient loop body never ran — this is not the "
                        "exception path.",
                        self._wm_empty_count, int(train_iters),
                        "None" if batch is None else "present")
            return metrics

        except ValueError as e:
            # ---- THIS WAS logger.debug UNTIL 2026-09-04 --------------------
            # MEASURED: 158,000 steps, 9 hours, and this handler fired on
            # EVERY block while emitting nothing a human could see, because
            # debug is below the host's log level. Downstream, the entire
            # developmental engine was blocked by it and said so in a way
            # that read like data rather than absence:
            #   stage: explore (WM-error=inf, slope=-inf)   <- the ONLY
            #   value of WM-error in the whole log
            # With no reconstruction error the stage controller can never
            # leave `explore`, so IMAGINE is unreachable, so the dream never
            # runs ("Dream policy: warmup (0 segments remaining)" forever),
            # so no goals unlock, so nothing is minted. One silent `except`
            # at DEBUG held the whole ladder down.
            # It also cost a diagnosis: grepping the log for this message
            # returned 0 hits, which was read as "this path is not taken"
            # when it actually meant "this path is not printed".
            # A swallowed exception must be LOUD. exc_info names the raising
            # call, which is the one thing needed to tell "training never
            # started" (sample_sequences) apart from "training ran and its
            # metrics were then discarded" (observe_sequence, after the
            # gradient steps) — opposite bugs with opposite fixes.
            self._wm_skip_count = getattr(self, "_wm_skip_count", 0) + 1
            if self._wm_skip_count <= 3 or self._wm_skip_count % 50 == 0:
                logger.warning(
                    "World model training SKIPPED (block #%d): %s — no "
                    "reconstruction error this block, so the stage "
                    "controller stays blind and the dream cannot activate.",
                    self._wm_skip_count, e, exc_info=True)
            return {}

    def _get_bg_sampler(self, batch_size: int, seq_len: int, prioritized: bool):
        """Lazily create the background prefetch sampler (async replay).

        batch_size/seq_len/prioritized come from config and are constant across
        the run, so a single long-lived sampler is correct. It is shut down in
        close().
        """
        if self._bg_sampler is None:
            from developmental_ai.world_model.replay_buffer import BackgroundSampler

            self._bg_sampler = BackgroundSampler(
                buffer=self.replay_buffer,
                batch_size=batch_size,
                seq_len=seq_len,
                device=self.device,
                prioritized=prioritized,
                reward_fraction=self._goal_replay_fraction,
                reward_threshold=self._goal_replay_threshold,
                terminal_fraction=float(self.config.get(
                    "world_model", {}).get("terminal_fraction", 0.25)),
            )
        return self._bg_sampler

    # -----------------------------------------------------------------------
    # Async world-model trainer (retrospective consolidation off-thread)
    # -----------------------------------------------------------------------
    def _ensure_wm_trainer(self) -> None:
        """Lazily start the background WM trainer (async_wm.enabled)."""
        if self._wm_trainer_thread is not None:
            return
        import queue
        import threading

        def _loop() -> None:
            while not self._wm_trainer_stop.is_set():
                try:
                    self._wm_ticket.get(timeout=0.25)
                except queue.Empty:
                    continue
                self._wm_busy.set()
                try:
                    m = self._train_world_model()
                    if m:
                        # atomic ref-swap under the GIL; the main thread
                        # copies the reference into _last_wm_metrics
                        self._wm_metrics_box = m
                        # producer site 1 of 3: async trainer thread. Recorded
                        # HERE, at the source, rather than downstream of the
                        # consume-once box — see _record_wm_telemetry for the
                        # measured failure this fixes.
                        self._record_wm_telemetry(m)
                except Exception:
                    # a crashed consolidation must log loudly and keep the
                    # trainer alive for the next ticket — never kill the run
                    logger.exception("async WM trainer iteration failed")
                finally:
                    self._wm_busy.clear()

        self._wm_trainer_thread = threading.Thread(
            target=_loop, name="wm-trainer", daemon=True)
        self._wm_trainer_thread.start()
        logger.warning("async WM trainer started (acting-while-learning)")

    # -----------------------------------------------------------------------
    # Goal-level prospection (imagine-before-committing)
    # -----------------------------------------------------------------------
    def _dream_consolidate_skills(self) -> Dict[str, float]:
        """SLEEP CONSOLIDATION — practise skills inside the world model.

        Waking practice can only fire when a skill is actually invoked AND
        produces its grounded effect. On a Minecraft client at ~10 steps/s
        those events are rare, so most skills would consolidate at a crawl.
        Here each bound skill is rolled forward THROUGH THE WORLD MODEL using
        its own actor, the resulting imagined trajectories are ranked by
        imagined return, and the skill is trained toward its ELITE rollouts —
        the ones where its own behaviour led somewhere good.

        This is deliberately the same mechanism as waking practice (same
        self-imitation, same KL trust region, same consolidation curve), only
        the evidence is dreamed. Three gates keep dreamed evidence honest:

          * WM TRUST. Gated on `_wm_trust_ema`, the same signal the dream
            distiller uses. A world model whose prior does not anticipate what
            actually happens produces rollouts that are confabulation, and
            training a skill on those is how an agent gets better at
            hallucinations. Below the floor, nothing consolidates.
          * DISCOUNTED EVIDENCE. `lr_scale` < 1: an imagined success is
            weaker evidence than a real one and is weighted as such.
          * ELITE SELECTION. Only the best fraction of rollouts teach. The
            median rollout of a mediocre skill is not worth imitating.

        Returns telemetry; never raises (a multi-day run must survive a bad
        batch).
        """
        out = {"slots": 0.0, "updates": 0.0, "trust": 0.0}
        ex = self.option_executor
        if ex is None or self.world_model is None:
            return out
        pr = getattr(ex.bank, "practice", None)
        if pr is None or not pr.enabled:
            return out
        cs = dict(self.config.get("dream_training", {}).get(
            "consolidate_skills", {}) or {})
        if not cs.get("enabled", False):
            return out

        trust = float(getattr(self, "_wm_trust_ema", 0.0))
        out["trust"] = trust
        min_trust = float(cs.get("min_trust", 0.2))
        if trust < min_trust:
            # Not a failure — the correct behaviour. Say so, because a silent
            # no-op here is indistinguishable from the feature being broken.
            self._dream_consolidate_skipped = getattr(
                self, "_dream_consolidate_skipped", 0) + 1
            return out

        n_roll = int(cs.get("rollouts", 8))
        horizon = int(cs.get("horizon", 12))
        elite_frac = float(cs.get("elite_frac", 0.25))
        per_seg = int(cs.get("slots_per_segment", 3))
        lr_scale = float(cs.get("lr_scale", 0.5))
        gamma = float(self.config.get("policy", {}).get("gamma", 0.99))

        cand = [i for i, b in enumerate(ex.bank.slots)
                if b is not None and not b.get("scripted")]
        if not cand:
            return out
        # DELIBERATE PRACTICE (infra #37, 2026-08-09; replaces round-robin).
        # Rehearse the skills nearest the LEARNING EDGE: success probability
        # ~0.5 is where one rehearsal teaches the most (the zone of proximal
        # development — arguably THE mechanism of human skill acquisition).
        # p is Laplace-smoothed from the executor's real invocation tally;
        # a never-tried skill scores 0.4 — worth a look, but behind anything
        # actually at the edge — so the tail of the bank still cannot starve
        # (the round-robin's virtue, kept without its blindness).
        _stats = getattr(ex, "slot_stats", {}) or {}
        # AGING + JITTER (review finding, 2026-08-09): a pure stable sort on
        # the ZPD score starved the tail — with all slots untried every score
        # ties at 0.4 and the SAME lowest-indexed picks win every segment
        # (the round-robin this replaced guaranteed coverage; its virtue is
        # restored here as a hunger term). Each unpicked segment adds
        # priority; being picked resets it; ties break randomly. A chronic
        # failure (score ~0.06) therefore still dreams once its hunger
        # accumulates — rehearsal is the only path by which it can change.
        if not hasattr(self, "_zpd_hunger"):
            self._zpd_hunger = {}

        def _zpd(slot: int) -> float:
            st = _stats.get(slot) or {}
            n = int(st.get("invocations", 0) or 0)
            if n <= 0:
                base = 0.4
            else:
                p = (int(st.get("spikes", 0) or 0) + 1.0) / (n + 2.0)
                base = 1.0 - 2.0 * abs(p - 0.5)
            return (base + 0.05 * self._zpd_hunger.get(slot, 0)
                    + 0.01 * random.random())

        picks = sorted(cand, key=_zpd, reverse=True)[:min(per_seg,
                                                          len(cand))]
        for _c in cand:
            self._zpd_hunger[_c] = (0 if _c in picks
                                    else self._zpd_hunger.get(_c, 0) + 1)

        for slot in picks:
            try:
                info = self._consolidate_one_slot(
                    slot, n_roll, horizon, elite_frac, gamma, lr_scale)
            except Exception as e:
                logger.warning("dream consolidation failed on slot %d: %s",
                               slot, e)
                continue
            out["slots"] += 1.0
            if info:
                out["updates"] += 1.0
            if self.infra is not None:
                # proof-of-life: consolidation ran END-TO-END (this exact
                # subsystem silently crashed every segment for weeks once)
                self.infra.beat("consolidation", self.total_timesteps)
        return out

    def _consolidate_one_slot(self, slot: int, n_roll: int, horizon: int,
                              elite_frac: float, gamma: float,
                              lr_scale: float) -> Optional[Dict[str, float]]:
        """Roll ONE skill through the world model and consolidate its elites."""
        ex = self.option_executor
        pr = ex.bank.practice
        b = ex.bank.slots[slot]
        actor = ex.bank.resident_actor(slot)
        if actor is None or b is None:
            return None

        # start states: real replay states, so the skill dreams from places
        # it could actually be — not from arbitrary latents
        wm_cfg = self.config.get("world_model", {})
        batch = self.replay_buffer.sample_sequences(
            batch_size=n_roll, seq_len=wm_cfg.get("sequence_length", 16),
            device=self.device,
            reward_fraction=self._goal_replay_fraction)
        with torch.no_grad():
            states, _ = self.world_model.observe_sequence(
                batch["observations"], batch["actions"],
                batch.get("proprio"))
            t_idx = torch.randint(0, states["h"].shape[1], (n_roll,))
            start = {"h": states["h"][torch.arange(n_roll), t_idx],
                     "z": states["z"][torch.arange(n_roll), t_idx]}

        P = int(self.action_dim)
        kdim = int(b.get("kdim") or 0)
        ctx = b.get("ctx")
        ctx_t = (torch.as_tensor(ctx, dtype=torch.float32,
                                 device=self.device).reshape(1, -1)
                 if (ctx is not None and kdim > 0) else None)
        _cond = ex.bank._materialize(slot)[1]
        feat_log: List[torch.Tensor] = []
        act_log: List[torch.Tensor] = []

        def _skill_policy(latent):
            # THE SKILL MUST SEE WHAT IT SEES WHEN AWAKE. Its actor consumes
            # encoder features, so the imagined latent is decoded back to an
            # observation and re-encoded through the SAME shared encoder —
            # the identical path `_add_dream_curiosity` uses to score
            # imagined frames with the ICM.
            with torch.no_grad():
                obs = self.world_model.decoder(latent)
                obs = obs.flatten(1) if self.pixel_obs else symexp(obs)
                feats = (self.policy._shared_encoder(obs)
                         if getattr(self.policy, "_shared_encoder", None)
                         is not None else obs)
                feats = feats.reshape(obs.shape[0], -1)
                # FIELD ORDER (review finding, 2026-08-09): the waking input
                # is [features | PROPRIO | knowledge]. This path used to
                # build [features | knowledge] and let the end-pad absorb the
                # difference — which parked the KNOWLEDGE values in the
                # proprio COLUMNS and zeros where knowledge belongs, so
                # consolidation rehearsed a systematically misaligned input.
                # A dream has no body-state: neutral zeros IN THE RIGHT
                # COLUMNS, the same missing-sense convention as everywhere.
                _pdim_c = int(b.get("proprio_dim", 0) or 0)
                if _pdim_c > 0:
                    feats = torch.cat(
                        [feats, torch.zeros(feats.shape[0], _pdim_c,
                                            device=feats.device)], dim=-1)
                if kdim > 0:
                    tail = (_cond(feats, ctx_t.expand(feats.shape[0], -1))
                            if (_cond is not None and ctx_t is not None)
                            else torch.zeros(feats.shape[0], kdim,
                                             device=feats.device))
                    feats = torch.cat([feats, tail], dim=-1)
                # DIMENSION RECONCILIATION (2026-08-07). The waking feature
                # vector carries loop-appended senses (proprio/knowledge
                # tail) this dream path does not rebuild, so an actor minted
                # on the wider vector crashed here EVERY segment with a
                # shape-mismatch ("8x354 vs 365x256") and those skills never
                # slept once. Pad the missing tail with zeros — the same
                # "absent sense reads neutral 0" convention every other
                # missing sense uses — or truncate a stale-wider vector.
                # Logged once per slot so a mismatch is visible, not fatal.
                _want = int(actor.shared[0].in_features)
                if feats.shape[-1] != _want:
                    if slot not in getattr(self, "_consol_dim_warned", set()):
                        self._consol_dim_warned = getattr(
                            self, "_consol_dim_warned", set()) | {slot}
                        logger.warning(
                            "dream consolidation slot %d: feature dim %d != "
                            "actor's %d — %s the difference (senses absent "
                            "in dreams read 0)", slot, feats.shape[-1],
                            _want, "zero-padding"
                            if feats.shape[-1] < _want else "truncating")
                    if feats.shape[-1] < _want:
                        feats = torch.cat(
                            [feats, torch.zeros(
                                feats.shape[0], _want - feats.shape[-1],
                                device=feats.device)], dim=-1)
                    else:
                        feats = feats[:, :_want]
                logits = actor.action_head(actor.shared(feats))
                # the dreamed skill must be the skill it actually IS: base +
                # owned delta (infra #35) — rehearsing the base alone would
                # practise a behaviour the waking skill no longer produces
                _dh = ex.bank.resident_delta(slot)
                if _dh is not None:
                    try:
                        _dd = _dh(feats)
                        if _dd.shape == logits.shape:
                            logits = logits + _dd
                    except Exception:
                        pass
                # primitives only: a dreamed NESTED invocation cannot be
                # executed inside the world model (there is no executor in
                # there), so slot rows are masked off rather than sampled and
                # silently reinterpreted as a primitive.
                logits = logits[:, :P]
                a = torch.distributions.Categorical(logits=logits).sample()
            feat_log.append(feats.detach().cpu())
            act_log.append(a.detach().cpu())
            return torch.nn.functional.one_hot(a, P).float()

        with self._wm_param_lock:
            imagined = self.world_model.imagine_trajectory(
                start, _skill_policy, horizon=horizon,
                knowledge=None)
        if not feat_log:
            return None

        # imagined discounted return per rollout
        rew = symexp(imagined["rewards"]).reshape(n_roll, -1)
        disc = torch.tensor([gamma ** t for t in range(rew.shape[1])],
                            device=rew.device).reshape(1, -1)
        ret = (rew * disc).sum(dim=1).cpu()

        k = max(1, int(round(elite_frac * n_roll)))
        elite = torch.topk(ret, k).indices.tolist()
        # A skill whose elite rollouts are no better than its median ones has
        # learned nothing from this dream — imitating them would just inject
        # noise. Require the elites to actually stand out.
        if float(ret.max() - ret.median()) <= 1e-6:
            return None

        traj: List[Tuple[torch.Tensor, int]] = []
        for t, (f, a) in enumerate(zip(feat_log, act_log)):
            for e in elite:
                if e < f.shape[0]:
                    traj.append((f[e].reshape(1, -1), int(a[e])))
        # sleep-practice trains the skill's OWNED delta (infra #35); the
        # base stays the frozen mint-time record. Written back so the
        # individuation survives LRU eviction.
        _delta = ex.bank.resident_delta(slot)
        _info = pr.consolidate(slot, actor, traj, lr_scale=lr_scale,
                               delta=_delta)
        if _info and not _info.get("reverted") and _delta is not None:
            ex.bank.writeback_delta(slot, _delta, _info.get("delta_mag"))
        return _info

    def _persist_practised_skills(self) -> int:
        """Write practised skill weights back to disk.

        A practised skill lives in its slot binding (so it survives LRU
        eviction) but the bank on disk still holds the weights it was minted
        with. Without this the improvement would not survive a restart — the
        agent would re-learn the same skill every run and never accumulate,
        which is exactly the "recording, not a memory" failure this feature
        exists to fix.

        Writes only slots whose `practised` counter moved since the last
        flush, so a segment in which nothing was practised costs no I/O.
        """
        if self.option_executor is None:
            return 0
        bank = self.option_executor.bank
        written = 0
        for slot, b in enumerate(bank.slots):
            if b is None or b.get("scripted"):
                continue
            n = int(b.get("practised", 0))
            if n <= int(b.get("practised_flushed", 0)):
                continue
            sid = b.get("skill_id")
            sk = self.skill_bank.skills.get(sid)
            if sk is None:
                continue
            try:
                sd = self.skill_bank.load_skill_policy(sid) or {}
                # replace ONLY the actor: the critic/conditioner/encoder are
                # untouched by practice, and the encoder in particular is the
                # world model's — rewriting it from here would be a lie about
                # what changed.
                sd["actor"] = {k: v.to(torch.float32)
                               for k, v in b["actor_sd"].items()}
                # the OWNED individuation travels with the skill (infra #35):
                # without this line every restart silently zeroed all
                # accumulated deltas — a recording again, not a memory
                # (review finding, 2026-08-09)
                if b.get("delta_sd"):
                    sd["delta"] = {k: v.to(torch.float32)
                                   for k, v in b["delta_sd"].items()}
                self.skill_bank.save_skill(
                    skill_id=sid, name=sk.name,
                    policy_state_dict=sd,
                    description=sk.description,
                    success_rate=float(sk.success_rate),
                    total_episodes=self.total_episodes,
                    context_embedding=sk.context_embedding,
                    preconditions=sk.preconditions or None,
                    obs_dim=self.obs_dim, action_dim=self.action_dim,
                    dedup=False, **self._arch_save_kwargs())
                b["practised_flushed"] = n
                written += 1
            except Exception as e:      # never kill a multi-day run
                logger.warning("persisting practised skill %s failed: %s",
                               sid, e)
        return written

    def _arch_save_kwargs(self) -> Dict[str, Any]:
        """Arch metadata that MUST accompany every saved policy.

        ONE builder for every save_skill site (mint, re-distill, _save_skill).
        The re-distill and _save_skill sites originally omitted these, so
        conv weights were written under a `arch='flat'` registry row; `_bind`
        sniffs the family from the state dict, routes to `_bind_conv`, reads
        enc_dim=0 from the registry, and raises SlotRefused — permanently,
        because `refused` was never cleared. A skill deleted itself from the
        option bank precisely BY IMPROVING. Keep this the single source.
        """
        # "wm" is an ENCODED family exactly like "conv" — it stores an encoder
        # and binds through _bind_conv — so it must carry enc_dim too. Omitting
        # it reproduces the documented failure above verbatim: enc_dim=0 in the
        # registry -> SlotRefused -> permanently un-invocable skill.
        _pa = getattr(self.policy, "arch", "flat")
        out: Dict[str, Any] = {
            "arch": _pa,
            # "rssm" belongs here too (2026-09-02): it reads features, so it
            # HAS an enc_dim, and _bind_conv refuses a zero one. It only
            # survived because the bind path can recover the value from the
            # state dict — a warning-logged fallback, not the intended route,
            # and exactly the "un-invocable skill" this comment warns about.
            "enc_dim": (int(getattr(self.policy, "enc_dim", 0))
                        if _pa in ("conv", "wm", "rssm") else 0),
            "head_dim": int(getattr(self, "meta_action_dim",
                                    self.action_dim)),
        }
        if self.option_executor is not None:
            out["slot_map"] = {
                str(_j): _sb["skill_id"]
                for _j, _sb in enumerate(self.option_executor.bank.slots)
                if _sb is not None and not _sb.get("scripted")} or None
        return out


    def _cur_proprio_t(self):
        """The primary stream's LAST-SEEN self-state, as the policy wants it.

        Imagination and prospection ask "what if I did X from here" — the
        body they should reason with is the one the agent currently has, not
        a blank. Returns None when the env has no body sense, in which case
        _augment is a no-op for this channel.
        """
        try:
            pp = (getattr(self, "_proprio_per_env", None) or [None])[0]
            return self.policy._prep_proprio(pp)
        except Exception:
            return None

    def _competence_from_option(self, closed: Dict[str, Any]) -> None:
        """Feed an option's OUTCOME back into the competence gate.

        THE GAP THIS CLOSES (measured 2026-07-25): competence was updated in
        exactly two places, both goal-attempt boundaries in achievement_goals.
        NOTHING in the option path ever touched it, so the gate was
        structurally blind to whether an option WORKS. Live consequence: 96 of
        96 completed options ended in `horizon` (full 80-tick budget, nothing
        achieved), one useless skill took 86% of all invocations, and its
        competence never moved — so it kept being offered forever, crowding
        out the scripted chop (down to ~3% of the budget) which is the only
        option that earns the +5.0 log tier.

        `spike` = the option produced a real reward spike -> success.
        Anything else (`horizon`, `episode_end`) -> failure. That is the same
        Bernoulli signal the predictor already consumes for goals, so a skill
        that never delivers self-gates within a few dozen invocations.

        MUST be paired with the re-offer probation in OptionExecutor.act():
        gating on failure ALONE would make this the 5th one-way latch in this
        project — gated -> never offered -> never invoked -> no new outcomes ->
        never reopens. See [competence probation] there.
        """
        bc = getattr(self, "broadcaster", None)
        if bc is None or not hasattr(bc, "competence"):
            return
        try:
            from developmental_ai.policy.options import _slot_index_of
            slot = _slot_index_of(closed.get("skill_id", ""))
            if slot < 0 or slot >= int(bc.competence.n_tasks):
                return          # scripted/unslotted option: nothing to update
            bc.competence.update(
                slot, 1.0 if closed.get("outcome") == "spike" else 0.0)
        except Exception as e:      # never kill a multi-day run over telemetry
            logger.warning("competence-from-option update failed: %s", e)

    def _prospective_goal_score(self, candidates) -> Dict[int, float]:
        """Imagine pursuing each candidate goal from the CURRENT stream state.

        For each candidate slot g: build the counterfactual goal-broadcast
        vector for g, roll the REAL (goal-conditioned) waking policy through
        the world model's imagination for K rollouts x H steps, and score by
        the discounted, continue-weighted, symexp'd reward-head return. The
        broadcaster blends these scores into its frontier selection — so what
        the organism imagines directly changes what it commits to next.

        Empty dict = "no opinion" (frontier-only fallback). Guards: never
        mutates the live rssm_state (detach/clone), read-only WM (eval +
        no_grad, mode restored), holds _wm_param_lock so the async trainer
        can't shift dynamics mid-rollout, requires a trained WM.
        """
        try:
            if (self._ll is None or getattr(self._ll, "rssm_state", None)
                    is None or not candidates):
                return {}
            # truthiness, not identity (review fix): a skipped first training
            # returns {} — that must NOT satisfy "WM has trained at least once"
            if not self._last_wm_metrics and not self._wm_metrics_box:
                return {}   # WM never trained -> imagining would hallucinate
            import torch.nn.functional as F
            b = self.broadcaster
            cfg = self._prospection_cfg
            K = int(cfg.get("rollouts", 16))
            H = int(cfg.get("horizon", 15))
            P = self.action_dim
            gamma = float(cfg.get("gamma", 0.997))
            disc = torch.pow(
                torch.tensor(gamma, device=self.device),
                torch.arange(H, device=self.device).float())
            knowledge = self.glue.knowledge_integrator.knowledge_vector
            if knowledge is not None:
                knowledge = knowledge.unsqueeze(0).expand(K, -1)
            was_training = self.world_model.training
            scores: Dict[int, float] = {}
            sems: Dict[int, float] = {}
            with self._wm_param_lock:
                self.world_model.eval()
                try:
                    with torch.no_grad():
                        # snapshot the primary stream's live latent (row 0);
                        # clone so decay()'s in-place mul can never touch it
                        s0 = {k: self._ll.rssm_state[k][0:1].detach()
                              .clone().repeat(K, 1) for k in ("h", "z")}
                        comp = b.competence.predict_all()
                        for g in candidates:
                            g = int(g)
                            f = np.zeros(b.DIM, dtype=np.float32)
                            f[g] = 1.0
                            f[b.max_slots:2 * b.max_slots] = b.achieved_union
                            f[2 * b.max_slots] = 1.0
                            f[2 * b.max_slots + 1] = float(comp[g])
                            kv = self.policy._prep_knowledge(f)
                            kvK = (kv.expand(K, -1)
                                   if kv is not None else None)

                            def policy_fn(latent, _kvK=kvK):
                                # arch='rssm': no decode/encode round trip —
                                # the latent IS the policy's input, so the
                                # imagined state goes straight in instead of
                                # through a lossy decoder/encoder pair.
                                if self.policy.arch == "rssm":
                                    aug = self.policy._augment(
                                        latent, _kvK, self._cur_proprio_t())
                                else:
                                    obs_hat = self.world_model.decoder(latent)
                                    if self.pixel_obs:
                                        obs_hat = obs_hat.flatten(1)
                                    else:
                                        obs_hat = symexp(obs_hat)
                                    aug = self.policy._augment(
                                        self.policy._encode(obs_hat), _kvK,
                                        self._cur_proprio_t())
                                feats = self.policy.actor.shared(aug)
                                # options-aware: primitive slice only (the WM
                                # was built P-wide; slot logits cancel)
                                logits = self.policy.actor.action_head(
                                    feats)[..., :P]
                                a = torch.distributions.Categorical(
                                    logits=logits).sample()
                                return F.one_hot(a, P).float()

                            imagined = self.world_model.imagine_trajectory(
                                {k: v.clone() for k, v in s0.items()},
                                policy_fn=policy_fn, horizon=H,
                                knowledge=knowledge)
                            ext = symexp(imagined["rewards"]).squeeze(-1)
                            w = disc
                            cont = imagined.get("continues")
                            if cont is not None:
                                # SHIFTED continue product (review fix):
                                # r_t is weighted by survival THROUGH t-1
                                # (r_t + gamma*c_t*future convention — c_t
                                # gates the NEXT step, not its own reward)
                                c = cont.squeeze(-1)
                                c_shift = torch.cat(
                                    [torch.ones_like(c[:, :1]), c[:, :-1]],
                                    dim=1)
                                w = disc * c_shift.cumprod(dim=-1)
                            rets = (ext * w).sum(-1)          # (K,)
                            scores[g] = float(rets.mean().item())
                            sems[g] = float(
                                (rets.std(unbiased=False)
                                 / max(1.0, float(K) ** 0.5)).item())
                finally:
                    self.world_model.train(was_training)
            # NOISE FLOOR (review fix): range-normalizing pure Monte-Carlo
            # noise would make selection half-random. If the spread across
            # candidates is within ~2 standard errors, prospection has no
            # real opinion — return {} and let the frontier decide alone.
            if scores:
                vals = list(scores.values())
                spread = max(vals) - min(vals)
                noise = 2.0 * (sum(sems.values()) / max(1, len(sems)))
                if spread < noise:
                    print(f"  Prospection: no signal (spread {spread:.4f} < "
                          f"noise {noise:.4f}) — frontier-only", flush=True)
                    return {}
            print(f"  Prospection scores (K={K},H={H}): "
                  + ", ".join(f"slot{g}={v:+.3f}"
                              for g, v in scores.items()), flush=True)
            return scores
        except Exception:
            logger.exception("prospection scoring failed — frontier-only")
            return {}

    # -----------------------------------------------------------------------
    # Symbolic feature-conditioning (Path A)
    # -----------------------------------------------------------------------

    def _seeded_reset(self):
        """Reset the environment, seeding its RNG exactly once (on the first
        reset) when a deterministic seed is configured. Gymnasium seeds the env
        RNG when `seed=` is passed; later resets without a seed continue the same
        deterministic stream, so episodes still vary but the whole run is
        reproducible. With no configured seed this is a plain reset()."""
        if self.seed is not None and not self._env_seeded:
            self._env_seeded = True
            try:
                return self.env.reset(seed=self.seed)
            except TypeError:
                # Older wrapper signature without seed support — fall back.
                return self.env.reset()
        return self.env.reset()

    def _current_knowledge_feature(self) -> Optional[np.ndarray]:
        """Current KG knowledge vector as a detached numpy feature for policy
        conditioning, or None when symbolic conditioning is off. Before the
        first embedding update the vector is None -> return zeros so the policy
        input keeps a fixed shape (the gate sees a null feature)."""
        if not self.symbolic_enabled or self.policy.knowledge_dim == 0:
            return None
        if self.symbolic_broadcast_lesion:
            # Rung 6 lesion: the channel (KG / episodic / affordance) is built and
            # updated identically, but we hand the policy a content-destroyed
            # feature instead of the real one. Same shape, same gate. The MODE
            # selects HOW content is destroyed (presence vs content vs variation):
            mode = getattr(self, "broadcast_lesion_mode", "zero")
            if mode == "none":
                # External toggle (e.g. verify_broadcast_live.py) with no mode set
                # -> treat as the plain zero lesion.
                mode = "zero"
            if mode == "scramble" and hasattr(self.broadcaster, "scrambled_feature"):
                # VALID-but-WRONG affordance vector (in-distribution, wrong info).
                return self.broadcaster.scrambled_feature(self._lesion_rng).astype(np.float32)
            if mode == "noise" and hasattr(self.broadcaster, "noise_feature"):
                # RANDOM valid vector each step, uncorrelated with the true state
                # (in-distribution, present + varies, but carries no information).
                return self.broadcaster.noise_feature(self._lesion_rng).astype(np.float32)
            if mode == "constant" and hasattr(self.broadcaster, "constant_feature"):
                # FIXED valid affordance vector (in-distribution, no variation).
                return self.broadcaster.constant_feature().astype(np.float32)
            # "zero" (or fallback when the broadcaster lacks a control method):
            # all-zeros — presence severed, clean control, possibly OOD.
            return np.zeros(self.policy.knowledge_dim, dtype=np.float32)
        if self.broadcaster is not None:
            # Within-episode broadcast (episodic working memory or symbolic
            # affordance/task-phase vector) — Rung 6 redesign.
            return self.broadcaster.feature().astype(np.float32)
        kv = self.glue.knowledge_integrator.knowledge_vector
        if kv is None:
            return np.zeros(self.policy.knowledge_dim, dtype=np.float32)
        return kv.detach().cpu().numpy().astype(np.float32).reshape(-1)

    # -----------------------------------------------------------------------
    # LLM goal-progress reward shaping (Path B)
    # -----------------------------------------------------------------------

    def _trajectory_summary(self, episode_reward: float, episode_length: int) -> str:
        """Compact textual summary of recent performance for the LLM to judge
        goal progress against. Kept short so the Ollama call stays cheap."""
        recent = list(self._recent_ep_rewards)
        avg = float(np.mean(recent)) if recent else episode_reward
        best = float(np.max(recent)) if recent else episode_reward
        trend = "improving" if len(recent) >= 2 and recent[-1] > recent[0] else "flat/declining"
        return (
            f"Latest episode return: {episode_reward:.1f} over {episode_length} steps.\n"
            f"Recent (last {len(recent)} eps) avg return: {avg:.1f}, best: {best:.1f}.\n"
            f"Trend: {trend}."
        )

    def _apply_llm_goal_shaping(
        self, episode_reward: float, episode_length: int
    ) -> None:
        """Poll the async LLM goal-progress score, apply a potential-based
        shaping increment onto the episode's last stored transition, and queue
        the next scoring job. No-op when shaping is disabled or during dream
        training (the dream actor trains on imagined rollouts, not this buffer)."""
        if not self.llm_shaping_enabled or self.dream_training_active:
            return

        # Consume any finished progress score -> update potential Phi. EMA
        # smooths LLM-to-LLM noise so the shaping doesn't jerk the reward around.
        p = self.llm.poll_progress()
        if p is not None:
            self.llm_potential = 0.5 * self.llm_potential + 0.5 * float(p)

        # Potential-based increment F = w * (Phi_now - Phi_prev) onto the last
        # transition. Difference-form telescopes => can't be farmed; clipped and
        # weight-annealed => bounded, fading guide.
        if self.policy.rollout_rewards:
            shaping = self.llm_shape_w * (self.llm_potential - self.llm_potential_prev)
            shaping = float(np.clip(shaping, -self.llm_shape_clip, self.llm_shape_clip))
            if shaping != 0.0:
                self.policy.rollout_rewards[-1] += shaping
            self.llm_potential_prev = self.llm_potential

        # Queue the next async scoring job against the current goal.
        self._recent_ep_rewards.append(episode_reward)
        if self.llm.should_score_progress(self.total_episodes):
            goal = self.glue.goal_manager.current_goal
            summary = self._trajectory_summary(episode_reward, episode_length)
            self.llm.submit_progress(
                goal.get("description", "") if goal else "", summary
            )

    # -----------------------------------------------------------------------
    # Graph embedding updates
    # -----------------------------------------------------------------------

    def _update_graph_embeddings(self) -> None:
        """
        Retrain knowledge graph embeddings on accumulated facts.

        This periodically updates the PyKEEN embeddings so the world model
        has access to the latest symbolic knowledge. The embeddings convert
        symbolic facts into vectors the neural world model can process.
        """
        triples = self.knowledge_graph.get_triples_for_embedding()
        if len(triples) >= 3:
            result = self.kg_embedder.train(triples)
            logger.debug(f"Updated graph embeddings: {result}")

    # -----------------------------------------------------------------------
    # Skill detection and saving
    # -----------------------------------------------------------------------

    def _save_skill(self, mastery_level: float) -> None:
        """Save the current policy as a mastered skill."""
        # Deterministic identity: one canonical skill per environment.
        # Re-mastering the same env upserts (keeps the best policy +
        # accumulates practice) instead of spawning near-duplicate skills.
        skill_id = f"skill_{self.env_name}"
        skill_name = f"Behavior in {self.env_name} (difficulty {self._get_difficulty()})"

        # Get recent performance stats
        recent_rewards = list(self.training_metrics["episode_reward"])
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        success_rate = mastery_level

        # Extract goal facts from knowledge graph
        goal_facts = []
        for fact in self.knowledge_graph.query(relation="received_reward"):
            goal_facts.append({
                "subject": fact.subject,
                "relation": fact.relation,
                "object": fact.obj,
            })

        # Context embedding = the goal the agent was pursuing when it mastered
        # this behavior. Stored so a future run can RETRIEVE this skill by
        # goal similarity (select_skills cosine) and warm-start from it —
        # cumulative learning across runs. obs/action dims are recorded so the
        # warm-start path can verify the saved policy is shape-compatible
        # before loading it (a cross-env match must not be loaded blindly).
        context_embedding = self.glue.goal_manager.current_embedding

        self.skill_bank.save_skill(
            skill_id=skill_id,
            name=skill_name,
            policy_state_dict=self.policy.get_state_dict(),
            description=f"Mastered at difficulty {self._get_difficulty()} with {mastery_level:.2f} mastery",
            success_rate=success_rate,
            total_episodes=self.total_episodes,
            avg_reward=avg_reward,
            context_embedding=context_embedding,
            goal_facts=goal_facts[:10],  # Keep it manageable
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            dedup=True,
            **self._arch_save_kwargs(),
        )

        logger.info(
            f"Skill saved: '{skill_name}' "
            f"(mastery={mastery_level:.2f}, reward={avg_reward:.2f})"
        )

        # Auto-push to Hugging Face Hub if configured
        hub_cfg = self.config.get("hub", {})
        if hub_cfg.get("push_on_mastery") and hub_cfg.get("repo_id"):
            try:
                self.skill_bank.push_to_hub(
                    repo_id=hub_cfg["repo_id"],
                    skill_ids=[skill_id],
                    token=hub_cfg.get("token"),
                    private=hub_cfg.get("private", True),
                )
            except Exception as e:
                logger.warning(f"Hub push failed: {e}")

        # Reset mastery detector for next skill
        self.mastery_detector.reset()

    def _get_difficulty(self) -> int:
        """Get current curriculum difficulty level."""
        if self.curriculum is not None:
            return self.curriculum.current_difficulty
        return 0

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def _income_now(self, top: int = 7):
        """LIVE income statement: shares of |reward| by source, THIS segment
        so far — the running values of the same per-term sums the ledger
        harvests at segment cadence, plus raw env income and the ICM/LP base.
        Built for the viewer: the number-one question in every reward-hacking
        incident was "what is it being paid for RIGHT NOW", and the segment
        report only answers it every ~1024 steps, after the fact."""
        try:
            src = {}
            for name, attr in (("coverage", "_cov_sum"), ("gaze", "_gaze_sum"),
                               ("novelty", "_nov_sum"),
                               ("symbols", "_sym_sum"),
                               ("sym_center", "_symc_sum"),
                               ("persistence", "_persist_sum"),
                               ("reach", "_reach_sum"),
                               ("imagination", "_cen_imag"),
                               ("icm_base", "_cen_base"),
                               ("extrinsic", "_seg_extrinsic_sum")):
                v = float(getattr(self, attr, 0.0) or 0.0)
                if v:
                    src[name] = v
            mag = (float(getattr(self, "_mag_turn", 0.0) or 0.0)
                   + float(getattr(self, "_mag_other", 0.0) or 0.0))
            if mag:
                src["magnet_seek"] = mag
            tot = sum(abs(v) for v in src.values())
            if tot <= 0:
                return None
            ranked = sorted(src.items(), key=lambda kv: -abs(kv[1]))[:top]
            return [(n, abs(v) / tot, v) for n, v in ranked]
        except Exception:
            return None

    def _viewer_push(self, action, ext_r, int_r, rssm_state, latent, ep_step,
                     pred_obs=None, actual_obs=None):
        """Stream the live env frame + world-model state to the viewer.

        Best-effort and fully isolated: any failure is swallowed so the viewer
        can never interrupt training.
        """
        try:
            try:
                env_rgb = self.env.render()
            except Exception:
                env_rgb = None
            # World-model predicted reward for the current latent (a WM head).
            try:
                with torch.no_grad():
                    pred_r = float(self.world_model.predict_reward(latent).item())
                pred_r = f"{pred_r:+.3f}"
            except Exception:
                pred_r = "n/a"
            act = action
            if hasattr(action, "tolist"):
                act = np.round(np.asarray(action), 2).tolist()
            lines = [
                ("stage", str(getattr(self.stage_controller, "stage", "-"))),
                ("action", str(act)),
                ("ext reward", f"{float(ext_r):+.3f}"),
                ("int reward", f"{float(int_r):+.3f}"),
                ("WM pred reward", pred_r),
                ("curiosity wt", f"{self.reward_mixer.weights['intrinsic']:.3f}"),
                ("episode step", str(ep_step)),
                ("timestep", str(self.total_timesteps)),
            ]
            recon = None
            if pred_obs is not None and actual_obs is not None:
                a = np.asarray(actual_obs, dtype=np.float32).ravel()
                p = np.asarray(pred_obs, dtype=np.float32).ravel()
                n = min(a.size, p.size)
                if n:
                    err = float(np.mean(np.abs(a[:n] - p[:n])))
                    recon = {"env_name": self.env_name, "real": a[:n].tolist(),
                             "pred": p[:n].tolist(), "err": err}
                    lines.append(("WM pred err", f"{err:.4f}"))
            z = rssm_state["z"].detach().cpu().numpy().ravel()
            self.viewer.push(
                env_rgb, f"{self.env_name}", lines, z=z,
                ret_hist=list(self.episode_rewards), recon=recon,
                income=self._income_now(),
            )
        except Exception:
            pass

    def _emit_heartbeat(self) -> None:
        """A thin live record, at most once every `heartbeat_seconds`.

        DELIBERATELY CHEAP. This runs on the per-step path, so it must cost
        almost nothing when it is not writing: one attribute read and one
        float compare. Everything it reports is either already computed or a
        dict lookup — no nvidia-smi (a subprocess per heartbeat would be
        absurd), no state_dict walks, no locks.

        NOT A SUBSTITUTE FOR THE SEGMENT RECORD. It carries running totals so
        a dashboard can show movement between segments; the segment record
        remains the authoritative per-segment observation, and the two live in
        separate files and separate tables so they can never be confused for
        one another in a query.
        """
        sink = self._heartbeat_sink
        if sink is None or self._hb_every_s <= 0:
            return
        try:
            now = time.time()
            if now - self._hb_last < self._hb_every_s:
                return
            steps = int(getattr(self, "total_timesteps", 0))
            dt = max(1e-6, now - self._hb_last) if self._hb_last else 0.0
            sps = ((steps - self._hb_last_steps) / dt) if dt else None
            self._hb_last, self._hb_last_steps = now, steps

            _wi = getattr(self, "_last_env_info", None) or {}
            bbt = {k: int(v) for k, v in
                   (_wi.get("breaks_by_type") or {}).items()
                   if isinstance(k, str)}
            _logs = sum(v for k, v in bbt.items() if "log" in k)
            rec = {
                "total_timesteps": steps,
                "uptime_s": round(now - getattr(
                    self, "_run_started_at", now), 1),
                "steps_per_s": round(sps, 2) if sps else None,
                "breaks_total": sum(bbt.values()),
                "logs": _logs,
                "attack_run": _wi.get("attack_run"),
                "episodes": int(getattr(self, "total_episodes", 0)),
            }
            # ---- LIVE INCOME SHARES (2026-09-28) ---------------------
            # WHY HERE AND NOT THE SEGMENT RECORD. `reward_shares` already
            # ships in the segment record, but that is emitted once per
            # ~1024 steps -- so the dashboard panel updated every ~10 minutes
            # and could only ever show the PREVIOUS segment. _income_now()
            # was written for exactly this ("what is it being paid for RIGHT
            # NOW") and was wired only to the live viewer, so nothing that
            # persists ever saw it.
            # top=12, not the default 7: truncation is why a share graph
            # cannot sum to 1, and a hidden channel is precisely what we are
            # trying to catch. `imagination` reached 69% of income during the
            # cave incident and was not on the dashboard at all.
            # Cost: one dict build + sort of ~10 floats, and only when the
            # heartbeat actually writes. The "deliberately cheap" contract in
            # this docstring still holds.
            try:
                _inc = self._income_now(top=12)
                if _inc:
                    rec["income_now"] = {n: round(share, 4)
                                         for n, share, _v in _inc}
            except Exception:
                pass
            # torch's own allocator counter — a cheap read, unlike shelling
            # out to nvidia-smi on the hot path.
            try:
                if torch.cuda.is_available():
                    rec["gpu_mem_mb"] = round(
                        torch.cuda.memory_allocated() / 1e6, 1)
            except Exception:
                pass
            sink.write(rec)
        except Exception:                              # pragma: no cover
            pass          # a heartbeat must never disturb the step loop

    def _emit_metrics(self, episode_metrics: Dict[str, Any]) -> None:
        """Write one structured record per segment for the live tracker.

        MEASURES NOTHING NEW. Every value here is already computed for the
        segment log; this only serialises it so a dashboard can read it and a
        host death cannot take it. Called from exactly one site (see §4.2 note
        there) and asserted as such by tests/_metrics_sink_smoke.py.

        Defensive throughout: a monitoring path must never be able to kill the
        run it monitors, so every read is a .get()/getattr() and the whole
        body is wrapped. A missing field is an absent key, never an exception.
        """
        # ---- MEMORY CENSUS HOOK (2026-10-07): MEASUREMENT ONLY ------------
        # Rides THIS site so there is still exactly one emission site (§4.2;
        # tests/_metrics_sink_smoke.py). Placed BEFORE the sink check so it
        # does not depend on metrics being enabled. Throttled by
        # diagnostics.memory_census_every_s; `touch runlogs/MEMCENSUS` forces
        # one at the next segment. Read-only (infra/memory_census.py) and
        # can never raise into the loop: one warning, then silence.
        try:
            _mch = getattr(self, "_memory_census_hook", None)
            if _mch is None:
                from developmental_ai.infra.memory_census import \
                    MemoryCensusHook
                _mch = MemoryCensusHook(
                    self.config.get("diagnostics", {}) or {})
                self._memory_census_hook = _mch
            _mch.tick(self)
        except Exception as exc:                       # pragma: no cover
            if not getattr(self, "_memory_census_warned", False):
                self._memory_census_warned = True
                logger.warning("memory census hook failed: %r", exc)
        # ---- LEARNING TELEMETRY (2026-10-07): MEASUREMENT ONLY -----------
        # learning.jsonl rides THIS site too (one emission site, §4.2), and
        # before the sink check so it does not depend on metrics.enabled.
        try:
            self._lt_emit()
        except Exception as exc:                       # pragma: no cover
            if not getattr(self, "_ltel_warned", False):
                self._ltel_warned = True
                logger.warning("learning telemetry emit failed: %r", exc)
        sink = getattr(self, "_metrics_sink", None)
        if sink is None:
            return
        try:
            em = dict(episode_metrics or {})
            rec: Dict[str, Any] = {
                "total_timesteps": int(getattr(self, "total_timesteps", 0)),
                "episode": int(getattr(self, "total_episodes", 0)),
                "uptime_s": round(time.time() - getattr(
                    self, "_run_started_at", time.time()), 1),
            }

            # ---- reward provenance: READ THE STASH, never call segment() ----
            # RewardLedger.segment() is a consuming read already performed in
            # infra/stack.py. Calling it here would return an empty statement
            # and destroy the real one.
            _led = dict(getattr(self.infra, "last_ledger_segment", {}) or {}) \
                if getattr(self, "infra", None) is not None else {}
            rec["reward_total"] = _led.get("total")
            rec["reward_shares"] = _led.get("shares") or {}
            rec["reward_hhi"] = _led.get("hhi")
            rec["reward_alarms"] = _led.get("alarms") or []

            # ---- reward channels, straight from the segment ----
            for k in ("episode_reward", "intrinsic_reward", "curiosity",
                      "avg_episode_reward", "episode_length",
                      "elapsed_time_seconds"):
                if k in em:
                    rec[k] = em[k]

            # ---- LEARNING SIGNALS LIVE IN training_metrics, NOT HERE --------
            # (fix 2026-09-04) These were read from `episode_metrics` and were
            # therefore None in every record — the "Learning signals" panel was
            # empty for the whole first run. `_collect_segment` does not return
            # them; they are appended to `self.training_metrics` deques from
            # several places (the PPO update, the WM train step), so the latest
            # value is the tail of the deque. Same access pattern _log_progress
            # already uses: list(self.training_metrics[k])[-1].
            _tm = getattr(self, "training_metrics", None) or {}
            for k in ("policy_entropy", "world_model_loss", "kl_divergence",
                      # reconstruction_error is the stage controller's SPINE:
                      # every curriculum transition is driven by its slope.
                      # It was never emitted, so "why has the stage not
                      # advanced?" was unanswerable from the dashboard.
                      "reconstruction_error",
                      "symbolic_decoder_loss", "symbolic_decoder_accuracy",
                      "inverse_dynamics_loss", "dream_distill_eff_weight",
                      "glue_mastery_score", "glue_kg_density"):
                try:
                    _dq = _tm.get(k)
                    if _dq:
                        rec[k] = list(_dq)[-1]
                except Exception:
                    pass

            # Imagination: emit boredom and the probe count alongside the
            # bonus. Without boredom, a 0.0 bonus on the dashboard cannot be
            # told apart from a broken path — see the note in _log_progress.
            _ic = getattr(self, "imagination_curiosity", None)
            if _ic is not None:
                try:
                    _is = _ic.stats
                    rec["imagination_bonus"] = float(_is.get("bonus", 0.0))
                    rec["imagination_boredom"] = float(
                        _is.get("boredom", 0.0))
                    rec["imagination_probes"] = int(_is.get("probes", 0))
                except Exception:
                    pass

            # ---- THE SCOREBOARD ----
            # STRING KEYS ONLY: a display bug once printed integer keys here
            # (they were action indices), so "2 breaks | top: [(3, 570)]" read
            # as 570 of block-type 3. Accept only names, so a mis-wired dict
            # shows as empty rather than as confident nonsense.
            # `_last_env_info`, NOT `_last_world_info` (fix 2026-09-04).
            # Both attributes exist and are assigned on the same lines
            # (4160/5157), which is exactly why the wrong one went unnoticed:
            #   _last_env_info   = step_infos[0]            <- has breaks_by_type
            #   _last_world_info = step_infos[0]["world"]   <- does NOT
            # Reading the world sub-dict returned {} forever, so the tracker
            # reported breaks_total=0 for an entire run while the segment log
            # printed "blocks broken: 2642 total, 8 logs" from the SAME step
            # info. A metric that reads zero is indistinguishable from an agent
            # doing nothing — which is the conclusion it very nearly produced.
            # This is the counter-you-have-not-validated trap in CLAUDE.md §5,
            # and the validation is: these numbers must match the
            # `blocks broken:` line in _log_progress, which reads _last_env_info.
            _wi = getattr(self, "_last_env_info", None) or {}
            bbt = {k: int(v) for k, v in
                   (_wi.get("breaks_by_type") or {}).items()
                   if isinstance(k, str)}
            rec["breaks_by_type"] = bbt
            rec["breaks_total"] = sum(bbt.values())
            _logs = sum(v for k, v in bbt.items() if "log" in k)
            rec["logs"] = _logs
            # The number this project is actually judged on: 399 breaks -> 1
            # log, historically. None (not 0) when no log has ever dropped,
            # so "no data" is distinguishable from "infinitely bad".
            rec["breaks_per_log"] = (
                round(rec["breaks_total"] / _logs, 1) if _logs else None)

            _cbt = {k: int(v) for k, v in
                    (_wi.get("crafts_by_type") or {}).items()
                    if isinstance(k, str)}
            rec["crafts_by_type"] = _cbt
            rec["crafts_total"] = sum(_cbt.values())

            _nb = [r for r in (_wi.get("runs_nobreak") or [])
                   if isinstance(r, (int, float))]
            if _nb:
                rec["attack_run_max"] = max(_nb)
                rec["attack_run_mean"] = round(sum(_nb) / len(_nb), 2)
                rec["attack_runs"] = len(_nb)

            # ---- skills: minted vs actually invoked ----
            # The competence gate once froze at 16 skills / 0 invocations
            # FOREVER, invisible for days. Both numbers, side by side.
            ob = getattr(self, "skill_option_bank", None) or getattr(
                getattr(self, "option_executor", None), "bank", None)
            if ob is not None:
                rec["skills_bound"] = sum(
                    1 for s in (getattr(ob, "slots", None) or []) if s)
                rec["skills_refused"] = len(getattr(ob, "refused", {}) or {})
                rec["skill_disk_loads"] = getattr(ob, "disk_loads", None)
                rec["cond_live"] = getattr(ob, "cond_live", None)
                rec["cond_load_failures"] = getattr(
                    ob, "cond_load_failures", None)
                rec["rssm_no_latent"] = getattr(ob, "rssm_no_latent", 0)
            sb = getattr(self, "skill_bank", None)
            if sb is not None:
                # `get_stats()`, NOT `list_skills()` (fix 2026-09-04).
                # SkillBank has no list_skills — the call raised AttributeError
                # every segment and a bare `except: pass` swallowed it, so
                # `skills_minted` was simply ABSENT from every record and the
                # panel had nothing to plot. A silent except around a metric is
                # how a tracker ends up looking alive while storing nothing;
                # the failure is now logged rather than discarded.
                try:
                    _st = sb.get_stats() or {}
                    rec["skills_minted"] = int(_st.get("total_skills", 0))
                    rec["skills_mastered"] = int(_st.get("mastered_skills", 0))
                    rec["skills_composite"] = int(_st.get("composite_skills", 0))
                    rec["avg_success_rate"] = _st.get("avg_success_rate")
                except Exception as exc:
                    logger.debug("skill stats unavailable: %r", exc)

            sink.write(rec)
        except Exception as exc:                       # pragma: no cover
            # Never let the tracker take the run down with it.
            logger.debug("metrics emit failed: %r", exc)

    def _save_checkpoint(self) -> None:
        """Save all component states to disk.

        Holds _wm_param_lock: with the async trainer, a state_dict() walk
        mid-optimizer-step would persist TORN weights (mixed pre/post-update
        tensors across modules) — poisoning the training host-restore path. Waiting one
        gradient step (<1s) buys a coherent snapshot. No-op cost sync mode.
        """
        with self._wm_param_lock:
            self._save_checkpoint_locked()
        # DELIBERATELY OUTSIDE THE PARAM LOCK: the replay buffer can be many
        # GB, and holding _wm_param_lock across that write would stall the
        # ACTING thread (its per-step observe forwards take the same lock)
        # for the whole of a multi-second disk write. The buffer has its own
        # internal lock, so it is safe to snapshot concurrently.
        self._save_replay_buffer()

    def _replay_persist_cfg(self):
        """(enabled, path, max_transitions_per_stream) or (False, ..) if off."""
        _wm = self.config.get("world_model", {}) or {}
        if not bool(_wm.get("buffer_persist", False)):
            return False, "", None
        _dir = os.path.join(
            self.config.get("loop", {}).get("log_dir", "./logs"),
            "checkpoints", "replay_buffer")
        _gb = float(_wm.get("buffer_persist_max_gb", 8.0) or 0.0)
        if _gb <= 0:
            return True, _dir, None
        # bytes per stored transition: obs dominates (uint8 pixels), plus the
        # action row and three float32 scalars
        _ob = int(self.obs_dim) * (1 if bool(_wm.get("obs_uint8", True))
                                   and self.pixel_obs else 4)
        _per = _ob + int(self.action_dim) * 4 + 12
        return True, _dir, max(1000, int(_gb * 1e9 / max(1, _per)))

    def _save_replay_buffer(self) -> None:
        """Persist recent experience so a crash-relaunch does not restart the
        world model from an EMPTY buffer.

        Contained: a failed buffer save must never take down a run that is
        otherwise fine. Losing the buffer costs re-collection; losing the
        process costs the whole session.
        """
        _on, _dir, _max = self._replay_persist_cfg()
        if not _on or self.replay_buffer is None:
            return
        try:
            _t0 = time.time()
            _n = self.replay_buffer.save(_dir, max_transitions=_max)
            logger.info(
                "replay buffer saved: %d transitions in %.1fs -> %s "
                "(a relaunch now resumes with this experience instead of an "
                "empty buffer)", _n, time.time() - _t0, _dir)
        except Exception as _e:
            logger.warning(
                "replay buffer save FAILED (%s: %s) — continuing; the run is "
                "unaffected but a crash would lose this experience",
                type(_e).__name__, _e)

    def _save_checkpoint_locked(self) -> None:
        checkpoint_dir = os.path.join(
            self.config.get("loop", {}).get("log_dir", "./logs"),
            "checkpoints",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save knowledge graph
        self.knowledge_graph.save(
            os.path.join(checkpoint_dir, "knowledge_graph.json")
        )

        # Save world model
        torch.save(
            self.world_model.state_dict(),
            os.path.join(checkpoint_dir, "world_model.pt"),
        )

        # Save curiosity module
        torch.save(
            self.curiosity.state_dict(),
            os.path.join(checkpoint_dir, "curiosity.pt"),
        )

        # SIDECARS (2026-10-05): atomic (tmp + fsync + os.replace) so a
        # crash mid-write cannot leave a torn file for the next boot, and
        # guarded so a failed sidecar never costs the policy save below.
        # Brain-mirrored: scripts/brain_mirror/brain_mirror.py lists both.
        from developmental_ai.infra.progress_curiosity import (
            atomic_pickle_dump)
        if isinstance(self.curiosity, LearningProgressCuriosity):
            try:
                atomic_pickle_dump(
                    os.path.join(checkpoint_dir, "curiosity_visits.pkl"),
                    self.curiosity.visit_state_dict())
            except Exception as _e:
                logger.warning("curiosity_visits.pkl save failed: %s", _e)
        if getattr(self, "_progress", None) is not None:
            try:
                atomic_pickle_dump(
                    os.path.join(checkpoint_dir, "progress_probes.pkl"),
                    self._progress.state_dict())
            except Exception as _e:
                logger.warning("progress_probes.pkl save failed: %s", _e)
        # exploration's career cell set (telemetry, evaluator-only): a
        # restart must not re-count every visited cell as new.
        if getattr(self, "_explore_tracker", None) is not None:
            try:
                atomic_pickle_dump(
                    os.path.join(checkpoint_dir, "exploration_cells.pkl"),
                    self._explore_tracker.state_dict())
            except Exception as _e:
                logger.warning("exploration_cells.pkl save failed: %s", _e)

        # Save policy
        torch.save(
            self.policy.get_state_dict(),
            os.path.join(checkpoint_dir, "policy.pt"),
        )

        # options: record the head layout so a resumed run can fail LOUD on
        # an incompatible action head instead of loading silently wrong
        if getattr(self, "option_executor", None) is not None:
            try:
                # .get("mtime"): scripted (bootstrap) slots have no policy
                # file, hence no mtime — b["mtime"] raised KeyError and
                # aborted the WHOLE options_state save (2026-07-23 host run).
                _slot_tbl = [
                    None if b is None else {
                        "skill_id": b["skill_id"],
                        "mtime": b.get("mtime"),
                        "scripted": bool(b.get("scripted", False))}
                    for b in self.option_executor.bank.slots]
                with open(os.path.join(checkpoint_dir,
                                       "options_state.json"), "w") as _f:
                    # int() coerces np.int64 (action_dim comes from gym
                    # Discrete(n).n, which is np.int64 -> not JSON-serializable)
                    json.dump({"meta_action_dim": int(self.meta_action_dim),
                               "action_dim": int(self.action_dim),
                               "slots": _slot_tbl}, _f)
            except Exception as _e:
                logger.warning("options_state save failed: %s", _e)

        # Save symbolic decoder (absent on pixel envs — no-op stub)
        if getattr(self.symbolic_decoder, "decoder", None) is not None:
            torch.save(
                self.symbolic_decoder.decoder.state_dict(),
                os.path.join(checkpoint_dir, "symbolic_decoder.pt"),
            )

        # Save glue layer (GNN + gate). The gate is absent on the pixel path
        # (build_gate=False) — omit the key rather than writing a null, so an
        # older loader sees a payload it can refuse cleanly instead of one it
        # would silently half-restore.
        _ki_save = {"gnn": self.glue.knowledge_integrator.gnn.state_dict()}
        if self.glue.knowledge_integrator.gate is not None:
            _ki_save["gate"] = (
                self.glue.knowledge_integrator.gate.state_dict())
        torch.save(_ki_save, os.path.join(checkpoint_dir, "glue_layer.pt"))

        # Save dream actor-critic
        torch.save(
            self.dream_actor.state_dict(),
            os.path.join(checkpoint_dir, "dream_actor.pt"),
        )

        # PERCEPTION (2026-08-10): the grounded heads + reliability + label
        # evidence — dying with the process cost hours of re-grounding and a
        # DEGENERATE-gated magnet at every boot. Perception is experience.
        if self.symbolizer is not None:
            try:
                torch.save(self.symbolizer.state(),
                           os.path.join(checkpoint_dir, "symbolizer.pt"))
            except Exception as _e:
                logger.warning("symbolizer checkpoint failed: %s", _e)

        # FAMILIARITY (2026-08-10): the count-based "what have I already
        # seen" dicts (view/gaze/symbol novelty). Absent from every
        # state_dict, they reset each launch and paid a fresh novelty
        # WINDFALL per restart — measured as symbols at 55-65% of all income
        # after every boot of the restart-heavy waves.
        try:
            torch.save({
                "nov_counts": dict(getattr(self, "_nov_counts", {}) or {}),
                "gaze_counts": dict(getattr(self, "_gaze_counts", {}) or {}),
                "known_symbols": sorted(getattr(self, "_known_symbols",
                                                set()) or set()),
                "symbol_counts": dict(getattr(self, "_symbol_counts",
                                              {}) or {}),
                "symbol_sight_counts": dict(getattr(
                    self, "_symbol_sight_counts", {}) or {}),
            }, os.path.join(checkpoint_dir, "familiarity.pt"))
        except Exception as _e:
            logger.warning("familiarity checkpoint failed: %s", _e)

        # MAGNET curiosity memory (2026-08-17): what the agent is curious
        # ABOUT. familiarity.pt above carries what it has already seen; this
        # carries the learning-progress estimates that decide where to look
        # next. Without it every launch printed "w=0.0000, target=None" with
        # all categories at LP 0.000 and spent its first hours unsteered —
        # the same restart tax, on the drive that aims the other senses.
        if self.vision_scaffold is not None:
            try:
                torch.save(self.vision_scaffold.state(),
                           os.path.join(checkpoint_dir, "magnet.pt"))
            except Exception as _e:
                logger.warning("magnet checkpoint failed: %s", _e)

        # CONSEQUENCE state: the banked evidence that lets the frontier
        # bootstrap without a first success. Written next to the run's logs
        # (not into the checkpoint dir) so it survives a checkpoint wipe.
        try:
            if self.consequence is not None:
                self.consequence.save(getattr(self, "_consequence_path",
                                              "runlogs/consequence_state.json"))
        except Exception as _e:
            logger.warning("consequence checkpoint failed: %s", _e)

        # ANTICIPATION state: same "restart amnesia recreates the bootstrap
        # problem" argument as consequence above, and it matters MORE here
        # because this channel PAYS (consequence's deficit only ranks).
        try:
            if self.anticipation is not None:
                self.anticipation.save(getattr(
                    self, "_anticipation_path",
                    "runlogs/anticipation_state.json"))
        except Exception as _e:
            logger.warning("anticipation checkpoint failed: %s", _e)

        logger.debug(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load component states from a previous checkpoint."""
        # Load knowledge graph
        kg_path = os.path.join(checkpoint_dir, "knowledge_graph.json")
        if os.path.exists(kg_path):
            self.knowledge_graph.load(kg_path)

        # Load world model
        wm_path = os.path.join(checkpoint_dir, "world_model.pt")
        if os.path.exists(wm_path):
            self.world_model.load_state_dict(
                torch.load(wm_path, map_location=self.device)
            )

        # Load curiosity module
        cur_path = os.path.join(checkpoint_dir, "curiosity.pt")
        if os.path.exists(cur_path):
            self.curiosity.load_state_dict(
                torch.load(cur_path, map_location=self.device)
            )

        # SIDECARS: absent (pre-2026-10 checkpoints) is silent; unreadable
        # or rejected (truncated write, prototype size changed by config)
        # is a COLD START FOR THAT COMPONENT ONLY, one warning, the file
        # renamed aside — never a boot crash, which the supervisor would
        # relaunch into forever (CLAUDE.md 4.1). Both load methods validate
        # before mutating, so "cold" really is the fresh module.
        from developmental_ai.infra.progress_curiosity import (
            load_pickle_sidecar)
        if isinstance(self.curiosity, LearningProgressCuriosity):
            load_pickle_sidecar(
                os.path.join(checkpoint_dir, "curiosity_visits.pkl"),
                self.curiosity.load_visit_state_dict, "curiosity visits",
                log=logger)
        if getattr(self, "_progress", None) is not None:
            load_pickle_sidecar(
                os.path.join(checkpoint_dir, "progress_probes.pkl"),
                self._progress.load_state_dict, "paired progress probes",
                log=logger)
        if getattr(self, "_explore_tracker", None) is not None:
            load_pickle_sidecar(
                os.path.join(checkpoint_dir, "exploration_cells.pkl"),
                self._explore_tracker.load_state_dict, "exploration cells",
                log=logger)

        # Load policy
        pol_path = os.path.join(checkpoint_dir, "policy.pt")
        if os.path.exists(pol_path):
            self.policy.load_state_dict(
                torch.load(pol_path, map_location="cpu")
            )

        # Load dream actor-critic
        dream_path = os.path.join(checkpoint_dir, "dream_actor.pt")
        if os.path.exists(dream_path):
            self.dream_actor.load_state_dict(
                torch.load(dream_path, map_location=self.device)
            )

        logger.info(f"Checkpoint loaded from {checkpoint_dir}")

    # -----------------------------------------------------------------------
    # LLM integration helpers
    # -----------------------------------------------------------------------

    def _llm_submit_facts(self) -> None:
        """Snapshot recent experience and SUBMIT an async fact-extraction job.

        All reading of mutating environment state (obs/action/reward history)
        happens here on the main thread; only the Ollama call + parsing runs in
        the background, so there is no data race on env state.
        """
        # Gate (July 2026): this method serializes EVERY observation dimension
        # as a named feature. On a 12288-d pixel obs (or an 8268-d
        # Craftax-Symbolic vector) that is a per-element dump of hundreds of
        # thousands of entries per call — meaningless to an LLM and huge.
        if self._skip_perdim_symbolic:
            return
        recent_obs = []
        recent_actions = []
        recent_rewards = []

        for i, obs_arr in enumerate(self.env.obs_history[-20:]):
            obs_dict = {}
            for j, val in enumerate(obs_arr):
                label = (
                    self.obs_labels[j]
                    if self.obs_labels and j < len(self.obs_labels)
                    else f"dim_{j}"
                )
                obs_dict[label] = float(val)
            recent_obs.append(obs_dict)

            if i < len(self.env.action_history):
                act = self.env.action_history[-(20 - i)] if len(self.env.action_history) >= (20 - i) else 0
                act_name = (
                    self.action_labels[act]
                    if self.action_labels and isinstance(act, int) and act < len(self.action_labels)
                    else str(act)
                )
                recent_actions.append(act_name)
            else:
                recent_actions.append("unknown")

            if i < len(self.env.reward_history):
                recent_rewards.append(
                    self.env.reward_history[-(20 - i)]
                    if len(self.env.reward_history) >= (20 - i)
                    else 0.0
                )
            else:
                recent_rewards.append(0.0)

        min_len = min(len(recent_obs), len(recent_actions), len(recent_rewards))
        self.llm.submit_facts(
            recent_obs[:min_len],
            recent_actions[:min_len],
            recent_rewards[:min_len],
        )

    def _llm_consume_facts(self) -> None:
        """Consume a finished fact-extraction result (KG writes stay on the
        main thread) and add any novel facts to the knowledge graph."""
        facts = self.llm.poll_facts()
        if not facts:
            return

        from developmental_ai.knowledge_graph.knowledge_graph import (
            SymbolicFact)   # hoisted out of the loop (2026-08-02): re-imported
        #                     per fact AND shadowing the module-level name.
        novel_count = 0
        for fact_dict in facts:
            if not isinstance(fact_dict, dict):
                continue
            fact = SymbolicFact(
                subject=fact_dict.get("subject", "unknown"),
                relation=fact_dict.get("relation", "related_to"),
                obj=fact_dict.get("object", "unknown"),
                confidence=float(fact_dict.get("confidence", 0.8)),
                source="llm_perception",
                timestamp=self.total_timesteps,
            )
            if self.knowledge_graph.add_fact(fact):
                novel_count += 1

        if novel_count > 0:
            logger.info(f"LLM extracted {novel_count} novel facts")

    def _llm_submit_analysis(self) -> None:
        """Snapshot the knowledge graph and SUBMIT an async analysis job."""
        facts = [
            {"subject": f.subject, "relation": f.relation, "object": f.obj}
            for f in self.knowledge_graph.facts.values()
        ]
        rules = [
            {
                "preconditions": r.preconditions,
                "action": r.action,
                "effects": r.effects,
                "confidence": r.confidence,
            }
            for r in self.knowledge_graph.action_rules
        ]
        skills = [
            {
                "name": s.name,
                "description": s.description,
                "mastery_level": s.mastery_level,
                "goal_facts": s.goal_facts,
            }
            for s in self.skill_bank.skills.values()
        ]

        self.llm.submit_analysis(facts, rules, skills)

    def _llm_consume_analysis(self) -> None:
        """Consume a finished knowledge-analysis result and log suggestions."""
        analysis = self.llm.poll_analysis()
        if analysis:
            logger.info(
                f"LLM analysis: {analysis.get('reasoning', 'N/A')}"
            )

            # Use LLM composition suggestions
            for suggestion in analysis.get("composition_suggestions", []):
                if isinstance(suggestion, dict):
                    logger.info(
                        f"LLM suggests composing: {suggestion}"
                    )

    # -----------------------------------------------------------------------
    # Dream-based policy training
    # -----------------------------------------------------------------------

    def _dream_train_policy(self) -> Dict[str, float]:
        """
        Train dream actor-critic on imagined trajectories (DreamerV3-style).

        1. Sample posterior states from replay buffer via world model
        2. Imagine H-step trajectories using dream actor
        3. Compute lambda-returns from predicted rewards
        4. Train actor to maximize returns, critic to predict them
        """
        wm_cfg = self.config.get("world_model", {})
        dream_cfg = self.config.get("dream_training", {})
        batch_size = dream_cfg.get("batch_size", 16)
        horizon = dream_cfg.get("imagination_horizon", 15)
        n_batches = dream_cfg.get("n_dream_batches", 1)

        metrics: Dict[str, float] = {}

        for _ in range(n_batches):
            # 1. Get posterior states as starting points for imagination.
            #    Goal-prioritized so imagination roots near reward-relevant states
            #    (the dream actor then actually experiences imagined goals).
            batch = self.replay_buffer.sample_sequences(
                batch_size=batch_size,
                seq_len=wm_cfg.get("sequence_length", 50),
                device=self.device,
                reward_fraction=self._goal_replay_fraction,
            )

            with torch.no_grad():
                states, _ = self.world_model.observe_sequence(
                    batch["observations"], batch["actions"],
                    batch.get("proprio")
                )
                # Pick random timesteps along the sequences as starting points
                seq_len = states["h"].shape[1]
                t_idx = torch.randint(0, seq_len, (batch_size,))
                starting_states = {
                    "h": states["h"][torch.arange(batch_size), t_idx],
                    "z": states["z"][torch.arange(batch_size), t_idx],
                }

            # 2. Imagine trajectories from these starting states
            #    Pass knowledge vector so dreams respect symbolic facts
            knowledge = self.glue.knowledge_integrator.knowledge_vector
            if knowledge is not None:
                knowledge = knowledge.unsqueeze(0).expand(batch_size, -1)

            self.world_model.eval()
            with torch.no_grad():
                imagined = self.world_model.imagine_trajectory(
                    starting_states,
                    policy_fn=self.dream_actor.policy_fn,
                    horizon=horizon,
                    knowledge=knowledge,
                )
            self.world_model.train()

            # 3. Convert predicted rewards from symlog to real scale
            imagined["rewards"] = symexp(imagined["rewards"])

            # 4. Optionally add curiosity bonus to imagined rewards
            if dream_cfg.get("dream_curiosity", False):
                imagined = self._add_dream_curiosity(imagined)

            # 5. Train dream actor-critic on this batch
            metrics = self.dream_actor.dream_training_step(imagined)

        return metrics

    def _distill_dream_to_real(self, batch_size: int = 32, seq_len: int = 10
                               ) -> Dict[str, float]:
        """Imagination-augmented learning: distill the imagination-trained dream
        actor (latent-space) into the REAL obs-space policy — but ONLY from dreams
        that are actually better, and with an influence that ANNEALS to zero as the
        real policy masters the task (never a fixed ratio).

        Per replay state we (a) take the dream actor's action distribution + value
        (teacher) and the real actor's + value (student); (b) value-GATE: keep only
        states where the dream critic values the state above the real critic (the
        dream sees a better path there); (c) sharpen the teacher by ``teacher_temp``
        (<1) so the student adopts the dream's preferred action rather than its
        uncertainty; (d) scale the whole update by (1 - solve_rate_EMA) so the
        dream's pull fades as extrinsic reward becomes reliably findable.

        Composes with symbolic ON: the real actor is conditioned on the knowledge
        vector, so we pass the agent's current knowledge feature (the same one
        dream-training uses) to ``_augment`` for every replay state. With symbolic
        off this is None and reduces to the obs-only path. (Previously hardcoded
        None, which silently RuntimeError'd the whole distill whenever symbolic was
        on — the dream-augment never actually trained the policy.)"""
        data = self.replay_buffer.sample_sequences(batch_size, seq_len, device=self.device)
        obs = data["observations"]                       # (B, L, obs_dim)
        acts = data["actions"]                            # (B, L, A) one-hot
        ld = self.world_model.rssm.latent_dim
        with torch.no_grad():
            states, infos = self.world_model.observe_sequence(
                obs, acts, data.get("proprio"))
            latent = self.world_model.rssm.get_latent(
                {"h": states["h"], "z": states["z"]}).reshape(-1, ld)
            teacher_logits = self.dream_actor.get_action_dist(latent).logits  # (N, A)
            v_dream = self.dream_actor.critic_net(latent).reshape(-1)          # (N,)
            # Per-state WM surprise: KL(posterior || prior) in nats. High KL =
            # the WM's prior did not anticipate what actually happened there =
            # its dream rollouts through that state are not to be trusted.
            kl_ps = None
            if self.dream_distill_trust_gate:
                S = self.world_model.rssm.stochastic_size
                C = self.world_model.rssm.stochastic_classes
                lp_post = torch.log_softmax(
                    infos["posterior_logits"].reshape(-1, S, C), dim=-1)
                lp_prior = torch.log_softmax(
                    infos["prior_logits"].reshape(-1, S, C), dim=-1)
                kl_ps = (lp_post.exp() * (lp_post - lp_prior)).sum(-1).sum(-1)  # (N,)
        obs_flat = obs.reshape(-1, self.obs_dim)
        # Knowledge vector for the symbolic-conditioned actor — use the SAME source
        # the live policy uses (_current_knowledge_feature; zeros if not yet
        # computed). _prep_knowledge returns (1, kdim) or None (symbolic off), and
        # we broadcast it to every replay state.
        kv = self.policy._prep_knowledge(self._current_knowledge_feature())
        if kv is not None:
            kv = kv.to(obs_flat.device).expand(obs_flat.shape[0], -1)
        # Detach: the distill updates the ACTOR weights given the current
        # conditioning; it must not backprop into the conditioner (whose params
        # _distill_opt does not own — that would corrupt PPO's gradients).
        # arch='rssm': `latent` above is already the policy's input space —
        # computed from the SAME observe_sequence the teacher was scored on,
        # so student and teacher are read on identical states. Re-encoding
        # obs_flat would have handed the student a different one.
        aug = self.policy._augment(
            latent if self.policy.arch == "rssm"
            else self.policy._encode(obs_flat),
            kv, self._cur_proprio_t()).detach()
        with torch.no_grad():
            v_real = self.policy.critic(aug).reshape(-1)                       # (N,)
        # (b) value gate: distill only where the dream expects to do strictly better
        gate = ((v_dream > v_real).float() if self.dream_distill_value_gate
                else torch.ones_like(v_dream))
        # (b2) WM-trust gate (audit H2): additionally require low WM surprise at
        # the state — the value gate alone opens on hallucinated dream value.
        trust_frac = 1.0
        if kl_ps is not None:
            trust = (kl_ps <= self.dream_distill_trust_kl_max).float()
            trust_frac = float(trust.mean().item())
            self._wm_trust_ema += self._wm_trust_beta * (
                trust_frac - self._wm_trust_ema)
            gate = gate * trust
        gate_frac = float(gate.mean().item())
        # (c) sharpened teacher target (temperature < 1 -> preferred action, not softness)
        T = max(1e-3, self.dream_distill_temp)
        teacher_p = torch.softmax(teacher_logits / T, dim=-1).clamp_min(1e-8)
        if getattr(self, "meta_action_dim", self.action_dim) \
                > self.action_dim:
            # OPTIONS: the student head is P+K wide but the dream teacher is
            # P-way. log_softmax over the primitive SLICE is exactly the
            # renormalized conditional over primitives (slot logits cancel),
            # so distillation targets the right object and slot logits get
            # ZERO distill gradient — invoke/don't-invoke stays PPO's call.
            # Without this the (N,P)x(N,P+K) broadcast raises and the
            # warn-once handler silently kills dream-augment for the run.
            _feats = self.policy.actor.shared(aug)
            _logits = self.policy.actor.action_head(_feats)
            student_logp = torch.log_softmax(
                _logits[..., :self.action_dim], dim=-1)
        else:
            student_logp = self.policy.actor(aug).clamp_min(1e-8).log()
        # forward cross-entropy distill (== forward-KL student gradient), gated
        per_state = -(teacher_p * student_logp).sum(-1)                        # (N,)
        denom = gate.sum().clamp_min(1.0)
        distill = (gate * per_state).sum() / denom
        # (d) natural anneal: fade the pull as the real policy solves reliably
        eff_w = self.dream_distill_weight
        if self.dream_distill_anneal:
            eff_w = eff_w * (1.0 - max(0.0, min(1.0, self._dream_solve_ema)))
        # (e) trust-scaled anneal (audit H2): the (1 - solve_ema) anneal is
        # MAXIMAL when the policy fails — on a weak-WM env that was a positive
        # feedback loop into collapse (failing policy -> full-strength pull
        # from a hallucinating teacher -> more failure). Scaling by the trust
        # EMA fades the pull wherever the WM hasn't earned trust, regardless
        # of how badly the real policy is doing.
        if self.dream_distill_trust_gate:
            eff_w = eff_w * max(0.0, min(1.0, self._wm_trust_ema))
        self._distill_opt.zero_grad()
        (eff_w * distill).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 10.0)
        self._distill_opt.step()
        return {"dream_distill_loss": float(distill.item()),
                "dream_distill_gate_frac": gate_frac,
                "dream_distill_trust_frac": float(trust_frac),
                "dream_distill_eff_weight": float(eff_w)}

    def _add_dream_curiosity(
        self, imagined: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add ICM curiosity bonus to imagined trajectory rewards."""
        with torch.no_grad():
            latents = imagined["latents"]
            batch_size, horizon, _ = latents.shape

            # Decode imagined latents to approximate observations for ICM
            decoded = self.world_model.decoder(
                latents.reshape(-1, latents.shape[-1])
            ).reshape(batch_size, horizon, -1)

            # Undo symlog for vector observations
            if not self.pixel_obs:
                decoded = symexp(decoded)

            # Compute curiosity reward per imagined transition. READ-ONLY
            # (update_state=False): imagined decoder reconstructions must not
            # mint/evict prototypes, append to error histories, or move the
            # running normalization — that would pollute the state the WAKING
            # curiosity signal is computed from (decoded frames co-bucket with
            # real scenes and carry systematically lower error).
            curiosity_rewards = torch.zeros_like(imagined["rewards"])
            for t in range(horizon - 1):
                cr = self.curiosity.compute_intrinsic_reward(
                    decoded[:, t],
                    imagined["actions"][:, t],
                    decoded[:, t + 1],
                    update_state=False,
                )
                curiosity_rewards[:, t, 0] = cr

            # Mix: extrinsic predicted reward + intrinsic curiosity
            imagined["rewards"] = (
                self.reward_mixer.weights["extrinsic"] * imagined["rewards"]
                + self.reward_mixer.weights["intrinsic"] * curiosity_rewards
            )

        return imagined

    # -----------------------------------------------------------------------
    # Logging and metrics
    # -----------------------------------------------------------------------

    def _predicates_batch(self, latent) -> Optional[List[Dict[str, bool]]]:
        """Grounded-predicate readout for EVERY env from the current belief.

        Feeds initiation-set gating: a skill is only offered where its
        recorded preconditions hold. Returns None when grounding has no
        evidence yet (early run), which leaves every slot offered — gating
        degrades to today's behaviour instead of silently disabling skills.
        """
        if self.symbolizer is None or latent is None:
            return None
        try:
            from developmental_ai.llm.vlm_symbolizer import PREDICATES
            grounded = [
                (i, p) for i, p in enumerate(PREDICATES)
                if (self.symbolizer.label_counts.get(p, 0)
                    >= self.symbolizer.min_labels)]
            if not grounded:
                return None
            probs = self.symbolizer.head.predict(
                latent.to(self.symbolizer.device))
            return [{p: bool(probs[r, i] > 0.5) for i, p in grounded}
                    for r in range(probs.shape[0])]
        except Exception:
            return None

    def _symbol_novelty(self, latent_row):
        """Reward for seeing something the agent can NAME as new.

        Returns (bonus, n_grounded, signature) or (0.0, 0, None).

        THE LOOP THIS IMPLEMENTS
            sees new thing -> a symbol becomes grounded -> reward
            sees more of an already-known symbol       -> less reward

        Two components, both count-based so they decay by construction:

        1. NEW SYMBOL. The first time a predicate crosses the symbolizer's
           trust gate (enough VLM labels AND enough reliability against real
           events), the agent has genuinely acquired a concept it did not
           have. That is paid ONCE, and can never be paid for that symbol
           again — a symbol is only new the first time.

        2. NEW COMBINATION. Distinct sets of currently-true predicates
           ("tree+stone", "water alone", "nothing") are counted, and the
           payout decays as 1/sqrt(n) like all the other novelty here. Staring
           at the ground yields the SAME signature every step, so it decays to
           nothing within seconds — which is the behaviour this replaces.

        WHY SYMBOLS RATHER THAN PIXELS. The visual hash it supersedes keyed on
        raw latent geometry, so a change of lighting or camera angle read as
        "new" even though nothing new was there. A symbolic signature only
        changes when the agent's own grounded vocabulary says the SCENE
        changed. Nothing here declares which symbols matter; the vocabulary is
        whatever the VLM taught and the world confirmed.
        """
        try:
            if self.symbolizer is None or latent_row is None:
                return 0.0, 0, None
            probs, rel, labels = self._grounded_object_probs(latent_row)
            if not probs:
                return 0.0, 0, None
            thr = float(self.symbolizer.fact_threshold)
            minl = int(self.symbolizer.min_labels)
            floor = float(self.symbolizer.reliability_floor)
            bonus = 0.0
            active = []
            for k in sorted(probs):
                grounded = ((labels or {}).get(k, 0) >= minl
                            and (rel or {}).get(k, 0.0) >= floor)
                if not grounded:
                    continue
                if k not in self._known_symbols:
                    # a concept it did not have before: paid once, ever
                    self._known_symbols.add(k)
                    bonus += float(self._new_symbol_bonus)
                    logger.info("NEW SYMBOL GROUNDED: %s (%d known)",
                                k, len(self._known_symbols))
                if probs[k] >= thr:
                    active.append(k)
            sig = tuple(active)
            # ---- CENTRING POTENTIAL (2026-08-02) ------------------------
            # Phi = P(something is under the crosshair) x how UNFAMILIAR the
            # least-seen nameable thing in view is. Turning so that a rarely
            # seen symbol is what the crosshair rests on raises Phi; turning
            # away lowers it again, so the shaping telescopes to ~0 over any
            # look-away/look-back cycle and cannot be farmed by fidgeting.
            #
            # It declares NOTHING about which symbols matter. The weight is
            # the same 1/sqrt(n) familiarity decay the set term already uses,
            # so what the agent is pulled toward is simply whatever it has
            # named least often — and that pull fades as it becomes familiar.
            #
            # HONEST LIMITATION: the vocabulary has no PER-SYMBOL position
            # predicate (`object_centered` means "something is centred", not
            # "the tree is centred"), so this is a proxy: centre something
            # WHILE an unfamiliar symbol is in view. Making it exact needs the
            # symbolizer to emit position per category.
            _AFFORD = ("object_centered", "object_adjacent",
                       "object_left", "object_right")
            _cen = float(probs.get("object_centered", 0.0))
            _sc = getattr(self, "_symbol_sight_counts", None)
            if _sc is None:
                _sc = self._symbol_sight_counts = {}
            _wn = 0.0
            _named = 0
            for _k in active:
                if _k in _AFFORD:
                    continue          # an affordance is not a thing
                _c = _sc.get(_k, 0)
                _sc[_k] = _c + 1
                _named += 1
                _wn = max(_wn, 1.0 / ((1.0 + _c) ** 0.5))
            self._sym_center_phi = float(
                max(0.0, min(1.0, max(0.0, min(1.0, _cen)) * _wn)))
            # SEMANTIC NOVELTY SIGNAL (2026-08-11, user directive). `_wn` is
            # "how unfamiliar is the least-seen NAMEABLE thing in view" — the
            # honest measure of whether this sight is a DISCOVERY rather than
            # merely an unseen pixel arrangement. Stashed with the step it was
            # computed on so the perceptual-novelty term can consume it
            # without recomputing, and can tell a stale value from a fresh
            # one (this method is skipped on dream-actor steps).
            self._sym_rarity_now = float(_wn)
            self._sym_named_now = int(_named)
            self._sym_rarity_step = int(self.total_timesteps)
            n = self._symbol_counts.get(sig, 0)
            self._symbol_counts[sig] = n + 1
            bonus += float(self._symbol_weight) / ((1.0 + n) ** 0.5)
            return bonus, len(self._known_symbols), sig
        except Exception:
            return 0.0, 0, None

    def _semantic_novelty_factor(self) -> float:
        """How new is the MEANING in view — not how new the pixels are.

        WHY (2026-08-11, user directive, from a decisive live observation).
        The agent was placed directly in front of a tree, ignored it, dug a
        single dirt block and sat in the hole. Perceptual novelty explains
        that exactly: it is a count over a hash of the visual latent, so a
        freshly dug hole is a genuinely never-before-seen view and pays like
        one. Digging MANUFACTURES novelty — the fourth time this project has
        met that anti-pattern ("novelty created by modifying the world is not
        discovery"). It was also the single largest income line at 34%.

        The fix is to make novelty SEMANTIC. The payout is now scaled by how
        unfamiliar the least-seen NAMEABLE thing in view is (`_wn` from
        `_symbol_novelty`, the same 1/sqrt(1+n) familiarity decay used
        everywhere else here):

          * a first village / first tree -> its symbols are unseen, factor
            ~1.0, so novelty SPIKES exactly when something is genuinely
            discovered (and the symbol term and curiosity spike with it);
          * a dirt hole -> `dirt` has been named thousands of times, factor
            ~0.02, so digging stops paying;
          * a view it cannot NAME at all -> the `unnamed_floor`, a small but
            non-zero pull, because the symbolic layer's silence may mean
            genuinely new territory rather than nothing of interest. That
            floor is what keeps this exploration rather than pure exploitation
            of the existing vocabulary.

        Declares NOTHING about which symbols matter: the vocabulary is
        whatever the VLM taught and the world confirmed, and the pull is
        simply toward whatever has been named least.
        """
        try:
            if not getattr(self, "_nov_semantic", False):
                return 1.0
            _floor = float(getattr(self, "_nov_unnamed_floor", 0.1))
            # STALENESS: _symbol_novelty is skipped on dream-actor steps and
            # when the symbol weights are 0. A stale rarity would silently
            # price this step by a view several steps old, so fall back to
            # the floor rather than to a wrong number.
            if int(getattr(self, "_sym_rarity_step", -1)) != int(
                    self.total_timesteps):
                return _floor
            if int(getattr(self, "_sym_named_now", 0)) <= 0:
                return _floor          # nothing nameable in view
            _r = float(getattr(self, "_sym_rarity_now", 0.0) or 0.0)
            # NOTE: no floor on the named branch. A thing you can name and
            # have seen 4000 times is LESS interesting than one you cannot
            # name at all, so familiar-named views are allowed to decay below
            # the unnamed floor — that asymmetry is the whole point.
            return max(0.0, min(1.0, _r))
        except Exception:
            return 1.0                 # never let pricing kill the step

    def _gaze_bonus(self, pitch) -> float:
        """Count-based novelty over VIEW DIRECTION (pitch buckets).

        Territory coverage counts the (x,z) cells the agent has stood in;
        nothing did the same for where it LOOKS. Pitch is clamped to
        [-90,+90], so an agent that drifts to a clamp finds `lookDOWN` is a
        free no-op — the world does not change, which means zero prediction
        error and therefore zero curiosity cost — while `lookUP` is the only
        action that moves it. Measured: pitch pinned at +90 for 100k+ steps
        with lookDOWN chosen ~2x more than lookUP, i.e. steady downward
        pressure into a wall that costs nothing to push against.

        A well-dwelt direction pays ~0 and an unvisited one pays ~the weight,
        decaying 1/sqrt(n) — so this cannot be farmed by sweeping: every
        bucket the agent oscillates through has its own count, and they all
        decay. Pitch spans ~12 buckets and SATURATES quickly, which is the
        point: it unsticks a clamp and then gets out of the way.

        Declares nothing about what is worth looking at. Returns 0.0 when the
        env cannot supply pitch (a missing sense reads neutral, never fatal).
        """
        if pitch is None:
            return 0.0
        try:
            bkt = int(round(float(pitch)
                            / max(1.0, float(self._gaze_bucket_deg))))
        except (TypeError, ValueError):
            return 0.0
        n = self._gaze_counts.get(bkt, 0)
        self._gaze_counts[bkt] = n + 1
        return float(self._gaze_weight) * (1.0 / ((1.0 + n) ** 0.5))

    def _feats_row(self, e_i: int):
        """This env's decision features as a numpy row, or None.

        Only arch='rssm' stores features with a transition — every other arch
        recomputes them from the observation at update time. `_act_latent` is
        the belief BEFORE this step's action, which is the state the decision
        was actually made from; using the post-step latent would score the
        choice against a world the choice itself produced.
        """
        _lat = getattr(self, "_act_latent", None)
        if _lat is None or e_i >= int(_lat.shape[0]):
            return None
        return _lat[e_i].detach().cpu().numpy()

    def _feats_for_act(self, n: int):
        """Per-env encoder features carried from last step, or None.

        The loop already ran `world_model.encoder(next_obs_t)` for the RSSM,
        and one step later that IS this env's observation — so the policy's
        own `_encode` (arch='wm' shares that same encoder) is a duplicate
        forward on every env step. This hands the result over EXPLICITLY
        rather than caching it: only the loop knows whether a given env's
        frame actually carried, and a cache that guessed would eventually
        serve a pre-reset frame after a world rebuild, silently.
        """
        # arch='rssm': the policy's input IS the world model's latent, which
        # the loop holds for every env at this exact moment. Hand it over per
        # env. No carry, no validity mask, no self-check — it is produced
        # fresh from that env's own RSSM row this step, so the failure modes
        # the carry machinery below guards against (a stale frame after a
        # world rebuild, a wrong env index) cannot arise.
        # getattr on SELF too, not just on the policy: this helper is called
        # with test doubles that carry only the carry-state fields (see
        # _throughput_wave_smoke::A2), and `self.policy` does not exist on
        # them. A missing policy means "not rssm", which is the old path.
        if getattr(getattr(self, "policy", None), "arch", "") == "rssm":
            _lat = getattr(self, "_act_latent", None)
            if _lat is None or int(_lat.shape[0]) != int(n):
                return None
            return [_lat[e:e + 1] for e in range(n)]
        if not self._reuse_enc_feats:
            return None
        _c = getattr(self, "_enc_carry", None)
        if _c is None:
            return None
        _feats, _valid = _c
        if _feats is None or int(_feats.shape[0]) != int(n):
            return None            # env count changed (rebuild): recompute
        return [(_feats[e:e + 1] if _valid[e] else None) for e in range(n)]

    def _set_enc_carry(self, encoded_next, obs_list, next_obs_list) -> None:
        """Remember this step's encoded next_obs for the next act().

        INVALIDATION IS THE WHOLE POINT, and it is checked DIRECTLY rather
        than inferred. Call this AFTER the obs-advance loop: an env that kept
        running has had `obs_list[e] = next_obs_list[e]` assigned, so the two
        entries are THE SAME OBJECT, while an env that reset holds a fresh
        array from `envs[e].reset()`. An identity test therefore states the
        exact precondition — "the frame the loop just encoded is the frame
        the policy is about to act on" — with no dependence on the branch
        conditions above (`dones[e] or _crashed`, plus a timeout path that
        carries the obs anyway). Replicating those conditions here is how
        this would eventually go silently wrong.
        """
        if not self._reuse_enc_feats or encoded_next is None:
            self._enc_carry = None
            return
        try:
            _f = encoded_next.reshape(int(encoded_next.shape[0]), -1).detach()
        except Exception:
            self._enc_carry = None
            return
        _n = int(_f.shape[0])
        if (obs_list is None or next_obs_list is None
                or len(obs_list) < _n or len(next_obs_list) < _n):
            self._enc_carry = None
            return
        self._enc_carry = (
            _f, [obs_list[_e] is next_obs_list[_e] for _e in range(_n)])

    # ---- per-step wall-clock attribution (2026-09-01) ---------------------
    # WHY THIS EXISTS. The env round-trip has been timed since 2026-08-17 and
    # reported ~3.4 steps/s against a 10 steps/s ceiling with the GPU idle —
    # i.e. two thirds of every step spent somewhere in OUR OWN per-step
    # Python, with no way to say where. §5 of CLAUDE.md is explicit that this
    # project has burned days on plausible theories; "the loop is slow" is a
    # theory until the phases are numbers. These buckets make the remainder
    # attributable, and they are what lets the other four fixes in this wave
    # be measured rather than argued.
    #
    # Cost: one perf_counter() per phase per step (~50ns each), no locks, no
    # allocation beyond one dict. Reset by _log_progress each segment.
    # `store` is split into four since 2026-09-20: the measured profile put
    # 21.6% of the step there and nothing in that window accounts for it.
    # `store` now holds only what the sub-buckets do not, so the four plus
    # `store` still sum to the original bucket.
    # "shadow" (2026-10-04): the foundation shadow recorder marks its own
    # phase in both live bodies; without it here its cost was folded into
    # UNACCOUNTED and the one number that says what it costs live was hidden.
    _PHASES = ("act", "env", "goals", "curiosity", "world", "vlm",
               "store_buf", "store_ppo", "store_reset", "store", "ppo",
               "shadow")

    def _frames_per(self, decisions: int) -> int:
        """Convert a threshold expressed in DECISIONS into env FRAMES.

        ---- TWO CLOCKS, ONE COUNTER (2026-09-01) -------------------------
        `total_timesteps` advances by `n` per loop iteration — it counts env
        frames across the whole fleet, which is what heartbeats, the census
        and "how much experience exists" all correctly mean.

        But several schedules are about how much the POLICY has been through,
        not how many frames the fleet collected, and they were compared
        against `total_timesteps` anyway: the novelty encoder snapshot
        (`novelty_encoder_refresh`), the unstuck advisor's rate limit
        (`advisor_min_gap`), and option probation (`gate_retry_after`). At
        num_envs 2 each of those fired twice as often as its config said, and
        `num_envs` is precisely the knob that will move on a cluster — so the
        error scales with the fleet rather than staying a fixed factor.

        Multiplying the THRESHOLD keeps the config value meaning what it
        reads as ("this many decisions") without mixing clocks inside the
        objects that consume it — an option's `_start_event` t and its
        `_close` t come from different call sites, and giving them different
        time bases would corrupt every recorded option duration.

        NOT affected, checked rather than assumed: `lp_stale_tau` counts
        per-category ABSENT OBSERVATIONS (`_cat_absent`), incremented once
        per scaffold step, so it was already in decision units.
        """
        return int(decisions) * max(1, int(getattr(self, "_num_envs", 1) or 1))

    # Sub-buckets of `env`, reported by the ADAPTER on info["step_ms"].
    # `_ENV_PARTS` is in-thread time inside one client's step(); the clients
    # run concurrently, so these apportion the WAIT rather than adding to it.
    _ENV_PARTS = ("engine", "obs", "sense", "adapter")

    def _env_parts_add(self, step_infos) -> None:
        """Fold this step's adapter breakdown into the segment accumulator.

        THE SLOWEST CLIENT, NOT THE MEAN. The loop waits on `f.result()` for
        every future, so the wall cost of the fan-out is the max, and a mean
        would flatter a fleet with one straggler. `_env_slow_sum` is what the
        `env` phase should be if the pool were perfectly parallel; the gap
        between it and the measured phase IS the pool/GIL overhead, which is
        the number that says whether adding clients buys anything.
        """
        _acc = getattr(self, "_env_parts", None)
        if _acc is None:
            _acc = self._env_parts = {}
        _slow = 0.0
        _seen = False
        for _inf in (step_infos or ()):
            _p = (_inf or {}).get("step_ms") if isinstance(_inf, dict) else None
            if not isinstance(_p, dict):
                continue
            _seen = True
            for _k in self._ENV_PARTS:
                _acc[_k] = _acc.get(_k, 0.0) + float(_p.get(_k, 0.0))
            _slow = max(_slow, float(_p.get("total", 0.0)))
        if _seen:
            _acc["_n"] = _acc.get("_n", 0.0) + 1.0
            _acc["_slow"] = _acc.get("_slow", 0.0) + _slow

    def _phase_mark(self, name: str, t0: float) -> float:
        """Add elapsed wall clock to phase `name`; return a fresh timestamp.

        Returns the new mark so call sites read as a chain
        (`_pt = self._phase_mark(<phase>, _pt)`) and no phase can be silently
        omitted from the total — an unaccounted gap shows up as the printed
        phases failing to sum to the step time, which is exactly the signal
        this instrument exists to give.
        """
        _now = time.perf_counter()
        _acc = getattr(self, "_phase_acc", None)
        if _acc is None:
            _acc = self._phase_acc = {}
        _acc[name] = _acc.get(name, 0.0) + (_now - t0)
        return _now

    def _world_model_version(self):
        steps = [int(v.get("step", 0)) for v in self.world_model.optimizer.state.values()]
        return "adam_step=" + str(max(steps, default=0))

    def _curiosity_boundary(self, streams=None, valid=True):
        if isinstance(self.curiosity, LearningProgressCuriosity):
            self.curiosity.end_visits(streams, valid=valid)

    def _curiosity_reward(self, obs, actions, next_obs, extra_error=None,
                          ended=None, invalid=None):
        """One stream/boundary contract for all collection paths."""
        n = len(obs)
        bad = np.zeros(n, dtype=bool) if invalid is None else np.asarray(invalid, bool)
        self._curiosity_valid_rows = ~bad
        ids = [i for i in range(n) if not bad[i]]
        result = torch.zeros(n, device=obs.device)
        if ids:
            kw = {}
            if extra_error is not None:
                kw["extra_error"] = torch.as_tensor(extra_error, device=obs.device)[ids]
            if isinstance(self.curiosity, LearningProgressCuriosity):
                kw["stream_ids"] = ids
            result[ids] = self.curiosity.compute_intrinsic_reward(
                obs[ids], actions[ids], next_obs[ids], **kw)
        # A restart row is EXCLUDED above (its transition splices two
        # worlds), but the visit open before it was real experience in the
        # old world: close it as valid. Discarding it threw away the very
        # pre-crash evidence the splice exclusion exists to protect.
        for i in range(n):
            if bad[i] or (ended is not None and ended[i]):
                self._curiosity_boundary([i])
        return result

    def _curiosity_train(self, obs_t, action_t, next_obs_t) -> Dict[str, float]:
        """One curiosity gradient step, optionally BATCHED across env steps.

        ONE implementation for all three stepping bodies on purpose. The
        per-step reward machinery in this file has produced six-plus
        duplicated-body bugs; a shared helper cannot drift by construction,
        which is strictly better than three copies kept in sync by hand.

        `train_every: 1` (the default) calls straight through, so the
        behaviour is byte-identical to before this existed. Above 1, the
        transitions are accumulated and one update is issued on the
        concatenated batch: identical gradient information, K-fold fewer
        kernel launches. Returns the most recent metrics on the steps that
        do not train (they feed the `curiosity_loss` log line only).
        """
        valid = getattr(self, "_curiosity_valid_rows", None)
        if valid is not None and len(valid) == len(obs_t):
            mask = torch.as_tensor(valid, device=obs_t.device)
            obs_t, action_t, next_obs_t = obs_t[mask], action_t[mask], next_obs_t[mask]
        if len(obs_t) == 0:
            return self._last_icm_metrics
        _k = self._curiosity_train_every
        if _k <= 1:
            self._last_icm_metrics = self.curiosity.train_step(
                obs_t, action_t, next_obs_t)
            return self._last_icm_metrics
        # detach: holding the autograd graph across steps would leak memory
        # and tie this batch to graphs the optimizer has already consumed
        self._cur_train_buf.append((obs_t.detach(), action_t.detach(),
                                    next_obs_t.detach()))
        if len(self._cur_train_buf) < _k:
            return self._last_icm_metrics
        _o = torch.cat([b[0] for b in self._cur_train_buf], dim=0)
        _a = torch.cat([b[1] for b in self._cur_train_buf], dim=0)
        _n = torch.cat([b[2] for b in self._cur_train_buf], dim=0)
        self._cur_train_buf.clear()
        self._last_icm_metrics = self.curiosity.train_step(_o, _a, _n)
        return self._last_icm_metrics

    def _apply_explore_boost(self, window: int,
                             max_total: int = 16384) -> str:
        """Double the exploration weights for `window` steps, or EXTEND a
        boost that is already running. Returns a status the caller logs.

        ONE implementation for two callers (2026-08-23 review). The stuck
        ladder's L2 remedy and the VLM advisor's `explore_wider` remedy
        open-coded the same four lines, and the advisor's copy was guarded by
        `if self._stuck_boost_orig is None` — which, because L2 fires first in
        the same segment hook and L3 satisfies `>= 2`, could NEVER be true.
        The remedy was dead code: the exchange was printed and written to
        help_responses.jsonl as accepted advice while nothing happened, which
        corrupted the only record of whether the advisor helps.

        Two invariants, both of them scars:

        * WEIGHTS DOUBLE ONCE. A second call extends the window, it does not
          re-multiply — otherwise repeated advice compounds 2x, 4x, 8x.
        * EXTENSION IS CAPPED at `max_total` steps from when the boost
          STARTED. The advisor is rate-limited, not silenced, so an unbounded
          extension would hold the weights at 2x forever and the expiry line
          would never print. A remedy that never expires is this project's
          guard-becomes-latch failure wearing the other hat.
        """
        _t = int(self.total_timesteps)
        if self._stuck_boost_orig is None:
            self._stuck_boost_orig = (self._novelty_weight,
                                      self._coverage_weight)
            self._novelty_weight *= 2.0
            self._coverage_weight *= 2.0
            self._stuck_boost_started = _t
            self._stuck_boost_until = _t + int(window)
            return f"applied (x2 for {int(window)} steps)"
        _ceil = int(getattr(self, "_stuck_boost_started", _t)) + int(max_total)
        _new = min(_t + int(window), _ceil)
        if _new <= int(self._stuck_boost_until):
            # two different refusals, and an operator reading the log needs to
            # tell them apart: a capped remedy is a bounded system working,
            # a redundant one is just a shorter window arriving late
            return ("declined (at max duration)"
                    if _t + int(window) > _ceil
                    else "declined (boost already runs longer)")
        self._stuck_boost_until = _new
        return f"extended (x2 until +{_new - _t} steps)"

    def _advise_unstuck(self, acts: Dict) -> None:
        """Stuck L3 -> ask the local VLM for slight guidance (infra #46).

        The help request stops being write-only: the same model that
        teaches the agent's perception reads the request plus the CURRENT
        VIEW and picks one intervention from a bounded menu of existing
        drive levers (see infra/advisor.py for the philosophy: guidance
        biases what is INTERESTING; the agent's own body keeps discovering
        the how). Everything here fails to a no-op — the run must behave
        identically with the advisor absent, refusing, or wrong.
        """
        _icfg = self.config.get("infra", {}) or {}
        if not bool(_icfg.get("advisor_enabled", False)):
            return
        if self.symbolizer is None:
            return
        adv = getattr(self, "_unstuck_advisor", None)
        if adv is None:
            from developmental_ai.infra.advisor import UnstuckAdvisor
            from developmental_ai.llm.vlm_symbolizer import (
                _get_ollama_client)
            _model = getattr(self.symbolizer, "model", None)

            def _q(prompt, images):
                c = _get_ollama_client()
                if c is None or not _model:
                    return None
                r = c.generate(model=_model, prompt=prompt,
                               images=images or None, format="json",
                               keep_alive=-1,
                               # num_ctx HERE TOO. Unreachable while the VLM
                               # is disabled (guarded on self.symbolizer), but
                               # this is the advisor path and it would re-load
                               # the model with an UNCAPPED cache the moment
                               # symbolic_grounding is turned back on -- the
                               # exact 8.3 GB that OOM-killed this box.
                               options={"num_predict": 200,
                                        "temperature": 0.0,
                                        "num_ctx": getattr(
                                            self.symbolizer, "num_ctx", 4096)})
                return (r.get("response") or "").strip()

            adv = UnstuckAdvisor(
                _q,
                # DECISIONS -> FRAMES: `advise` is called with
                # total_timesteps (see _frames_per).
                min_gap=self._frames_per(
                    int(_icfg.get("advisor_min_gap", 4096))),
                log_dir=str(_icfg.get("log_dir", "runlogs")))
            self._unstuck_advisor = adv
        _cats = (list(getattr(self.vision_scaffold, "target_categories",
                              None) or [])
                 if self.vision_scaffold is not None else [])
        _png = None
        try:
            _fr = getattr(self, "_advisor_frame", None)
            if _fr is not None:
                _png = self.symbolizer._encode_png(_fr)
        except Exception:
            _png = None
        _wi = getattr(self, "_last_world_info", None) or {}
        _sit = {
            "situation": acts.get("stuck_reason"),
            "position": (_wi.get("x"), _wi.get("y"), _wi.get("z")),
            "recent_memory": (self.infra.episodic.summary(
                self.total_timesteps)
                if (self.infra is not None
                    and self.infra.episodic is not None) else None),
        }
        advice = adv.advise(_sit, _cats, _png, self.total_timesteps)
        if advice is None:
            return
        _r = advice["remedy"]
        print("  infra/advisor: " + _r
              + (f" [{advice['category']}]" if advice.get("category")
                 else "")
              + (f" — {advice['why']}" if advice.get("why") else ""))
        if _r == "look_around":
            # same lever as the gaze-starved remedy: re-open gaze novelty
            # and let the magnet act on whatever the sweep finds
            _gc = getattr(self, "_gaze_counts", None)
            if isinstance(_gc, dict):
                for _b in list(_gc):
                    _gc[_b] = min(_gc[_b], 25)
            if self.vision_scaffold is not None:
                self.vision_scaffold._seek_nudge_left = \
                    self.vision_scaffold.seek_nudge_budget
        elif _r == "prime" and self.vision_scaffold is not None:
            # the goal-emulation lever: category validated against the
            # agent's OWN vocabulary in the advisor's parser; social_prime
            # keeps its edge-trigger, so repeated advice cannot latch
            if not self.vision_scaffold.social_prime(
                    advice["category"], self.total_timesteps,
                    duration=int(_icfg.get("advisor_prime_steps", 4000))):
                print("  infra/advisor: prime declined "
                      "(already primed or unknown category)")
        elif _r == "explore_wider":
            # the L2 remedy on request, with a LONGER window than the ladder's
            # own. No `is None` precondition: L2 has always already fired by
            # the time L3 reaches here, so requiring an unboosted state made
            # this branch unreachable. _apply_explore_boost extends instead,
            # and caps the extension so repeated advice cannot latch.
            # "->" not "—": the line above already carries the VLM's REASON
            # for the remedy; this one reports what actually happened to it,
            # which is the half that used to be a lie
            print("  infra/advisor: explore_wider -> "
                  + self._apply_explore_boost(4096))
        # "conserve" is a deliberate no-op: the advisor judged the lull
        # transient, and doing nothing on advice is still an answer

    def _view_key(self, obs_row):
        """A DETERMINISTIC vector for what is on screen right now.

        Uses the shared world-model encoder, which is a pure function of the
        frame. Deliberately NOT the RSSM latent: its stochastic half is
        resampled every step, so hashing it made every repeat of the same
        sight look new and turned count-based novelty back into an
        inexhaustible farm — the exact failure it was meant to remove.
        """
        try:
            t = torch.as_tensor(np.asarray(obs_row, dtype=np.float32),
                                device=self.device).reshape(1, -1)
            with torch.no_grad():
                # arch='rssm' has no _shared_encoder (its input is the latent,
                # not encoder features), but this key has always wanted a
                # pure function of the FRAME — so read the world model's
                # encoder directly rather than losing the frozen-snapshot
                # behaviour below. The latent would be wrong here for the
                # reason the docstring gives.
                enc = (getattr(self.policy, "_shared_encoder", None)
                       or (self.world_model.encoder
                           if self.policy.arch == "rssm" else None))
                # FROZEN SNAPSHOT (see _nov_enc_refresh at init): the live
                # encoder trains every step, so its bucket map drifts and old
                # sights re-key as new. Hash through a periodic snapshot so
                # repetition genuinely decays between refreshes.
                if enc is not None and self._nov_enc_refresh > 0:
                    if (self._nov_enc is None
                            or self.total_timesteps - self._nov_enc_stamp
                            >= self._nov_enc_refresh):
                        import copy as _copy
                        self._nov_enc = _copy.deepcopy(enc).eval()
                        for _p in self._nov_enc.parameters():
                            _p.requires_grad_(False)
                        self._nov_enc_stamp = self.total_timesteps
                    return self._nov_enc(t).reshape(1, -1)
                if enc is not None:
                    return enc(t).reshape(1, -1)
                return self.policy._encode(t).reshape(1, -1)
        except Exception:
            return None

    def _perceptual_cell(self, latent_row) -> int:
        """A discrete id for WHAT THE AGENT IS CURRENTLY SEEING.

        Locality-sensitive hash of the agent's own world-model latent: a
        fixed random projection to `_nov_bits` sign bits. Similar views land
        in the same bucket, genuinely different ones do not.

        WHY NOT SPATIAL CELLS. Territory coverage counts (x, z) cells, so it
        pays for DISTANCE TRAVELLED and nothing else — and the cheapest way
        to maximise it is to walk in a straight line staring at the floor,
        which is exactly what the agent converged on (walk 40%, pitch pinned
        at +90). Novelty of PLACE is not novelty of EXPERIENCE.

        WHY NOT RAW PREDICTION ERROR. ICM never saturates on a rotation, so
        spinning farms it forever. A COUNT-based bonus does saturate: turning
        on the spot cycles through the same handful of buckets and their
        payout decays as 1/sqrt(n), while walking into genuinely new
        surroundings keeps producing unseen ones. That asymmetry is the whole
        point — it rewards seeing new THINGS, not covering ground.

        The key is the agent's OWN representation, so what counts as "a new
        thing" is defined by what its world model can already tell apart,
        not by a rule we impose.
        """
        try:
            v = latent_row.reshape(1, -1)
            if getattr(self, "_nov_proj", None) is None:
                g = torch.Generator(device="cpu").manual_seed(0)
                self._nov_proj = torch.randn(
                    v.shape[1], int(self._nov_bits), generator=g).to(v.device)
            bits = (v.float() @ self._nov_proj > 0).reshape(-1)
            out = 0
            for i, b in enumerate(bits.tolist()):
                if b:
                    out |= (1 << i)
            return int(out)
        except Exception:
            return -1

    def _reach_sense(self, latent_row) -> float:
        """CAN I TOUCH ANYTHING FROM HERE — the felt affordance, in [0,1].

        WHY THIS AND NOT A DEPTH NUMBER. Handing the policy a distance
        readout would be a rangefinder, not perception, and it is not what
        humans use. People do not estimate metres before swinging; Minecraft
        draws a wireframe outline around the block your crosshair is on ONCE
        IT IS IN REACH, and players key off that affordance directly. This is
        the same signal, read from the agent's own grounding head
        (`breakable_in_reach`), which the VLM taught and which the symbolizer
        scores against REAL break events — it is one of only four predicates
        with genuine ground truth behind it.

        WHY IT IS NEEDED. Swinging into air and swinging into a trunk are
        currently identical as far as the policy's inputs are concerned: it
        has no contact sense at all. That is why it can hold attack at a
        canopy six blocks up indefinitely — nothing it can feel distinguishes
        that from chopping. Depth ITSELF stays learnable from the pixels
        (occlusion, texture gradient, fog, and above all motion parallax now
        that it actually walks); this supplies the calibration signal that
        makes those cues mean something, the way reaching and missing is what
        teaches a child where their hand ends.

        DELIBERATELY A SENSE, NOT A RULE. It says nothing about sky being bad
        or trees being good, and it is not a reward — it enters
        proprioception only, so it can shape what the agent does without
        paying it to do anything. An absent or untrusted head reads 0.0,
        which is the same neutral convention every other missing sense uses.
        """
        try:
            if self.symbolizer is None or latent_row is None:
                return 0.0
            probs, rel, labels = self._grounded_object_probs(latent_row)
            if not probs:
                return 0.0
            key = "breakable_in_reach"
            if key not in probs:
                self._reach_gate = "absent"
                return 0.0
            # Record WHY the sense reads what it reads. A gated-off sense and
            # a genuinely-nothing-in-reach sense both print 0.00, and those
            # demand opposite responses: one means the feature is inert, the
            # other means the agent is aiming at nothing.
            self._reach_raw = float(probs[key])
            # Trust gate, mirroring the magnet and the symbolizer's own
            # perceptual-fact gate: an ungrounded predicate reports nothing
            # rather than noise the policy would learn to act on.
            if (labels or {}).get(key, 0) < self.symbolizer.min_labels:
                self._reach_gate = (
                    f"gated:labels={(labels or {}).get(key, 0)}"
                    f"<{self.symbolizer.min_labels}")
                return 0.0
            if (rel or {}).get(key, 0.0) < self.symbolizer.reliability_floor:
                self._reach_gate = (
                    f"gated:reliability={(rel or {}).get(key, 0.0):.2f}"
                    f"<{self.symbolizer.reliability_floor}")
                return 0.0
            self._reach_gate = "live"
            return float(max(0.0, min(1.0, probs[key])))
        except Exception:
            return 0.0

    def _grounded_object_probs(self, latent_row):
        """Per-object presence signal for the curiosity magnet, from the
        grounding head on ONE env's RSSM latent (the primary stream).

        Returns (object_probs, reliability, label_counts) — the head's float
        P(predicate) for every predicate, plus the symbolizer's earned
        reliability and label-count dicts (the magnet's own trust gate mirrors
        perceptual_facts). Returns (None, None, None) when grounding has no
        evidence yet, which leaves the magnet inert instead of steering on
        noise.
        """
        if self.symbolizer is None or latent_row is None:
            return None, None, None
        try:
            from developmental_ai.llm.vlm_symbolizer import PREDICATES
            probs = self.symbolizer.head.predict(
                latent_row.reshape(1, -1).to(self.symbolizer.device)
            ).squeeze(0)
            object_probs = {p: float(probs[i]) for i, p in enumerate(PREDICATES)}
            return (object_probs, self.symbolizer.reliability,
                    self.symbolizer.label_counts)
        except Exception:
            return None, None, None

    def _encode_crop(self, crop):
        """HWC uint8 centre crop -> [1, enc_dim] world-model encoder features.

        The symbolizer owns the CROP (it knows fovea_frac); this owns the
        OBSERVATION CONTRACT (resize to image_size, /255, CHW, flatten) and
        the encoder. Keeping the split here is why the symbolizer needs no
        knowledge of how pixels reach the model.

        DETACHED on purpose: the fovea head trains on these features, and
        gradient must never flow back into the SHARED encoder — reshaping
        the perception every other subsystem reads, to serve one head, is
        precisely the coupling the project has been burned by before.
        """
        try:
            enc = getattr(self.world_model, "encoder", None)
            if enc is None or crop is None:
                return None
            side = int(getattr(enc, "image_size", 0) or 0)
            if side <= 0:
                side = int(round((float(self.obs_dim) / 3.0) ** 0.5))
            a = np.asarray(crop)
            if a.ndim != 3 or a.shape[2] != 3:
                return None
            h, w = a.shape[:2]
            if h != side or w != side:
                # nearest-neighbour resample: no scipy/cv2 dependency, and
                # the head only needs coarse layout, not interpolation
                yi = np.clip((np.arange(side) * h) // side, 0, h - 1)
                xi = np.clip((np.arange(side) * w) // side, 0, w - 1)
                a = a[yi][:, xi]
            obs = (a.astype(np.float32) / 255.0).transpose(2, 0, 1)
            t = torch.as_tensor(obs.reshape(1, -1), device=self.device)
            with torch.no_grad():
                return enc(t).detach()
        except Exception:
            return None

    def _fovea_kwargs(self, latent_row) -> Dict[str, Any]:
        """Foveal probs + counts + measured pitch for the magnet, from the
        symbolizer's fovea head. Empty dict when the channel is off, so the
        magnet call sites can always `**` this without branching (and the
        THREE duplicated loop bodies cannot drift apart — the hazard that
        killed the gui_open guard, the magnet diagnostics and the viewer)."""
        out: Dict[str, Any] = {}
        try:
            if (self.symbolizer is not None
                    and getattr(self.symbolizer, "fovea_enabled", False)):
                fp = self.symbolizer.fovea_probs(latent_row)
                if fp is not None:
                    out["fovea_probs"] = fp
                    out["fovea_counts"] = dict(
                        self.symbolizer.fovea_label_counts)
                    # cached for other consumers this same step (the
                    # sky-novelty discount reads it; one head forward, N uses)
                    self._last_fovea_probs = fp
                    self._last_fovea_counts = out["fovea_counts"]
                    # POSITIVE sightings, cached separately: presence-claims
                    # (social gate, episodic landmarks, assoc) must gate on
                    # "actually seen", not "head was trained" (review)
                    self._last_fovea_pos = dict(getattr(
                        self.symbolizer, "fovea_pos_counts", {}) or {})
            _p = (getattr(self, "_last_world_info", None) or {}).get("pitch")
            if _p is not None:
                out["pitch"] = float(_p)
        except Exception:
            return {}
        return out

    @staticmethod
    def _event_count_map(info) -> Dict[str, int]:
        """Lifetime event counts as one flat "kind:subtype" map, from the
        adapter's per-kind exports. This is the shared vocabulary between
        habituation, the boring-view discount and the infra stack's
        category<->event association — one assembly point, no drift."""
        out: Dict[str, int] = {}
        if not isinstance(info, dict):
            return out
        for kind, key in (("break", "breaks_by_type"),
                          ("place", "places_by_type"),
                          ("craft", "crafts_by_type"),
                          ("pickup", "pickups_by_type")):
            for k, v in (info.get(key) or {}).items():
                try:
                    out[f"{kind}:{k}"] = int(v)
                except Exception:
                    continue
        return out

    def _boring_view_factor(self) -> float:
        """View-novelty multiplier in [0.15, 1]: how much of the fovea is
        filled by sky or by a MASTERED material (2026-08-08).

        Sky: clouds drift even under frozen time, so sky views mint fresh
        buckets forever. Mastered materials: digging manufactures genuinely
        new views (every hole is one), so view-novelty was bankrolling the
        dirt farm the extrinsic decay had already defunded. Both are the
        same statement — a view dominated by something the agent has fully
        habituated to is not worth orienting to. Trust-gated per category;
        trees/water/animals are never discounted; floored so first glances
        keep paying. The FARM dies, not the sight."""
        boring = 0.0
        # MEASURED PITCH FIRST (2026-08-08): the sky discount keyed only on
        # the fovea head's sky reading, and the head's confidence sags
        # exactly when the agent lives in the failure mode (staring at
        # clouds all segment -> agreement 0.80 and falling -> discount never
        # bites -> novelty re-funds the sky farm at 0.04/step from a dirt
        # pillar at y=72). Above ~55 degrees of tilt the view is sky or the
        # agent's own feet BY GEOMETRY — same prefer-measured-over-predicted
        # rule as the reach sense. SYMMETRIC since the same day: with the up
        # clamp defunded the bot swapped to the DOWN clamp (mean +76.7deg,
        # 48.5% pinned, staring at mastered ground) — the goals of a
        # terrestrial world live near the horizon in both directions.
        _pv = (getattr(self, "_last_world_info", None) or {}).get("pitch")
        if _pv is not None and abs(float(_pv)) > 55.0:
            boring = max(0.0, min(1.0, (abs(float(_pv)) - 55.0) / 35.0))
        fps = getattr(self, "_last_fovea_probs", None) or {}
        fcs = getattr(self, "_last_fovea_counts", None) or {}
        if not fps:
            return max(0.15, 1.0 - boring)
        ml = (getattr(self.symbolizer, "min_labels", 5)
              if self.symbolizer is not None else 5)
        if fcs.get("sky_visible", 0) >= ml:
            boring = max(boring, max(0.0, min(1.0, float(
                fps.get("sky_visible", 0.0)))))
        # MEASURED ASSOCIATION, NOT A NAME MAP (2026-08-08). This used to
        # consult a hand-written category->block-name table — domain
        # knowledge a general agent cannot ship. The infra stack instead
        # LEARNS which perceptual categories co-occur with which caused
        # events ("what was I looking at when things happened") and mastery
        # of those events makes the category boring. Cold start = no
        # associations = no discount, which is the safe direction.
        if self._habituation_scale > 0.0 and self.infra is not None:
            for cat, b in self.infra.category_boringness(
                    self._habituation_scale).items():
                if fcs.get(cat, 0) < ml:
                    continue
                boring = max(boring, max(0.0, min(1.0, float(
                    fps.get(cat, 0.0)))) * float(b))
        return max(0.15, 1.0 - boring)

    def _habituation_factor(self, info) -> float:
        """MASTERY HABITUATION (2026-08-08): familiarity multiplier for THIS
        step's intrinsic reward, keyed on what just BROKE.

        WHY. The extrinsic break tier already decays with familiarity
        (break_decay_scale), and it worked — dirt pays ~nothing. What kept
        the dirt farm alive was the INTRINSIC side: every break produces a
        large frame change, so the ICM/novelty stack re-pays the orienting
        response for the 300th dirt exactly as for the 1st. MEASURED: 272 +
        134 dirt across two runs, 0 logs, agent grinding the ground at its
        feet. Habituation is the human analogue — a stimulus repeated
        without consequence stops triggering orientation. The step's whole
        intrinsic is scaled by the LEAST-familiar type that broke (a novel
        block in the same step keeps its full surprise), floored at 0.1 so
        mastered breaks stay perceptible, 1.0 when nothing broke at all —
        sights, movement and exploration are never touched.
        """
        if self._habituation_scale <= 0.0 or not isinstance(info, dict):
            return 1.0
        # EVENT-STREAM FORM (infra #44/#12, 2026-08-08): adapters that emit
        # the typed event stream get fully general habituation — any event
        # KIND the domain has (break/place/craft/pickup/...), no taxonomy in
        # this file. The least-familiar event still governs; deaths are not
        # "caused effects" and never habituate. Legacy parsing below remains
        # for adapters without the stream.
        ev = info.get("events")
        if ev is not None:
            counts = self._event_count_map(info)
            facs = []
            for kind, sub in ev:
                # CAUSED kinds only (review 2026-08-09): an uncounted kind
                # (observed_change, death) reads count 0 -> factor 1.0, and
                # max() would let it lift a co-occurring MASTERED event back
                # to full surprise — un-defunding the very farms habituation
                # exists to kill. Non-caused changes are not this body's
                # repetitions and have no business in its habituation.
                if kind not in ("break", "place", "craft", "pickup"):
                    continue
                n = counts.get(f"{kind}:{sub}", 0)
                facs.append(1.0 / (1.0 + float(n) / self._habituation_scale))
            if not facs:
                return 1.0
            f = max(0.1, min(1.0, float(max(facs))))
            if f < 1.0:
                self._habit_hits = getattr(self, "_habit_hits", 0) + 1
                self._habit_sum = getattr(self, "_habit_sum", 0.0) + f
            return f
        ach = info.get("achievements") or {}
        broke = []
        for k, v in ach.items():
            k = str(k)
            if not k.startswith("mine_"):
                continue
            try:
                v = int(v)
            except Exception:
                continue
            prev = self._habit_prev.get(k)
            # first sighting of a counter is baseline, not a break (counters
            # restart at 0 on every mission rebuild — see MineRL stat note)
            if prev is not None and v > prev:
                broke.append(k[5:])
            self._habit_prev[k] = v
        # PLACEMENTS habituate exactly like breaks (2026-08-08): with breaks
        # defunded the bot pillar-built instead (use action 18%, 1 cell/hr) —
        # placing the 200th dirt makes the same unearned frame-change income
        # breaking it did. Same counters, same decay, least-familiar event
        # still governs so a genuinely novel placement keeps its surprise.
        placed = [str(t) for t in (info.get("placed_now") or [])]
        if not broke and not placed:
            return 1.0
        facs = []
        bbt = info.get("breaks_by_type") or {}
        for b in broke:
            facs.append(1.0 / (1.0 + float(bbt.get(b, 0))
                               / self._habituation_scale))
        pbt = info.get("places_by_type") or {}
        for t in placed:
            facs.append(1.0 / (1.0 + float(pbt.get(t, 0))
                               / self._habituation_scale))
        f = max(0.1, min(1.0, float(max(facs))))
        if f < 1.0:
            self._habit_hits = getattr(self, "_habit_hits", 0) + 1
            self._habit_sum = getattr(self, "_habit_sum", 0.0) + f
        return f

    # The neutral row: no flow, no mover, and a figure/ground ratio of 0.5
    # meaning "the attended region is at the same depth as its surround".
    # 0.5 rather than 0.0 because 0.0 is a CLAIM (the thing I am looking at is
    # further away than everything around it), and a body without the sense
    # must not make claims.
    _FLOW_NEUTRAL = {"flow_fovea": 0.0, "flow_edge": 0.0,
                     "flow_ratio": 0.5, "mover": 0.0}

    def _flow_ego_batch(self, step_infos, n: int):
        """(n, 3) body motion over THIS step, or None.

        Built from the two raw proprio readings the loop already holds:
        `_wm_proprio_prev` describes obs_t (it is advanced only after the
        buffer write, which happens later in the step) and `step_infos`
        describes next_obs_t. So at curiosity time both ends of the interval
        are in hand and no new state is needed.

        None unless the world model is in depth mode — every other mode
        ignores ego, and building it would be pure cost.
        """
        if (getattr(self.world_model, "flow_mode", "raw") != "depth"
                or not self._wm_proprio_dim
                or self._wm_proprio_prev is None):
            return None
        try:
            prev = self._wm_proprio_batch_from(self._wm_proprio_prev, n)
            now = self._wm_proprio_batch(step_infos, n)
            if prev is None or now is None:
                return None
            from developmental_ai.world_model.rssm import ego_from_proprio
            return ego_from_proprio(
                prev, now, self._proprio_layout,
                float(getattr(getattr(self, "_proprio_source", None),
                              "MOVE_SCALE", 1.0)))
        except Exception as _e:
            logger.debug("ego from proprio failed: %s", _e)
            return None

    def _wm_proprio_batch_from(self, rows, n: int):
        """(n, proprio_dim) from a list of raw vectors, missing -> zeros."""
        if not self._wm_proprio_dim:
            return None
        d = self._wm_proprio_dim
        out = np.zeros((n, d), dtype=np.float32)
        for e_i in range(min(n, len(rows) if rows is not None else 0)):
            v = rows[e_i]
            if v is None:
                continue
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            out[e_i, :min(d, v.shape[0])] = v[:d]
        return torch.from_numpy(out).to(self.device)

    def _flow_residual_batch(self, obs_t, next_obs_t, action_tensor, n: int,
                             ego=None):
        """(n,) unexplained-motion fraction for THIS step's transition.

        ALIGNMENT IS THE WHOLE POINT OF THIS BEING SEPARATE. Curiosity scores
        the transition (obs_t -> next_obs_t) under action a_t, so the residual
        must be measured on that same pair. The latent of obs_t is last step's
        POST-observe latent — this step's observe has not run yet — which is
        exactly what `_flow_prev_latent` holds. Reading `self._flow_now`
        instead (the version cached for proprioception) would score the
        transition one step stale.

        Returns None when there is no flow head or no usable previous latent,
        which the caller passes straight through as "no extra channel".
        """
        if not getattr(self.world_model, "flow_enabled", False):
            return None
        pl = self._flow_prev_latent
        if pl is None or pl.shape[0] != n:
            return None                       # first step, or the fleet resized
        try:
            with torch.no_grad():
                _r, _m = self.world_model.flow_residual(
                    obs_t, next_obs_t, pl, action_tensor,
                    valid=self._flow_prev_valid, ego=ego,
                    return_map=True, map_size=self._surprise_grid)
            # WHERE the world model is wrong, for the PRIMARY stream only —
            # the same cadence and the same cost profile every other loop-side
            # sense gets. Kept as a plain array; _augment_proprio appends it.
            self._surprise_now = _m[0, 0].reshape(-1).detach().cpu().numpy()
            return _r
        except Exception as _e:
            logger.debug("flow residual failed: %s", _e)
            return None

    def _flow_senses(self, latent_row, mover: float):
        """The three probe senses for the primary stream, plus `mover`.

        Cheap by construction: one FlowHead forward on a latent the loop has
        already computed. Nothing here touches the env, renders, or calls a
        VLM — the three things that have historically cost step rate.

        `mover` is passed in rather than recomputed: the residual was already
        measured against the correct frame pair when curiosity ran earlier in
        this same step (see _flow_residual_batch), and warping twice to get the
        same number would be pure cost.

        NOT A REWARD. These enter proprioception and nothing else, on the
        doctrine _reach_sense states: a felt affordance, not a rangefinder,
        and not a payment. tests/_perspective_smoke.py holds the contract that
        nothing downstream of them can pay while the agent stands still.
        """
        out = dict(self._FLOW_NEUTRAL)
        try:
            probe = torch.zeros(1, self.action_dim, device=self.device)
            probe[0, self._flow_probe_idx] = 1.0
            out.update(self.world_model.flow_probe(
                latent_row, probe, fovea_frac=self._flow_fovea_frac))
            out["mover"] = float(mover)
        except Exception as _e:                      # never stall the step
            logger.debug("flow senses failed: %s", _e)
        return out

    # ---- LEARNING TELEMETRY (2026-10-07): MEASUREMENT ONLY ---------------
    # Thin forwarders so each body carries ONE short line per stage (§4.2:
    # identical text in _run_episode_parallel and _collect_segment, count
    # 2). Off (self._ltel None) they cost one attribute test. Everything is
    # read-only: values the body already computed, the executor's state via
    # the shadow's read-only action_sources(), the producers' own pop_*
    # counters. Nothing here can change reward, actions, replay or training
    # (tests/_learning_telemetry_smoke.py proves it byte-for-byte), and the
    # telemetry object swallows its own errors.
    def _lt_begin(self, intrinsic) -> None:
        if self._ltel is not None:
            self._ltel.begin(intrinsic)

    def _lt_mark(self, label, intrinsic, extrinsic, ext_label=None,
                 mult_name=None, mult=None) -> None:
        if self._ltel is not None:
            self._ltel.mark(label, intrinsic, extrinsic, ext_label,
                            mult_name, mult)

    def _lt_commit(self, mixed, intrinsic, extrinsic) -> None:
        if self._ltel is not None:
            self._ltel.commit(0, mixed, intrinsic, extrinsic,
                              self.reward_mixer)

    def _lt_pre_step(self, env_actions, n, use_dream_actor) -> None:
        if self._ltel is None:
            return
        try:
            if use_dream_actor:
                src = ["dream_actor"] * int(n)
            else:
                src = [d.get("source", "unknown") for d in
                       self._lt_action_sources(self.option_executor, n)]
        except Exception:
            src = ["unknown"] * int(n)
        self._ltel.pre_step(env_actions, src)

    def _lt_post_step(self, step_infos, dones, n) -> None:
        if self._ltel is not None:
            self._ltel.post_step(step_infos, dones, n)

    def _lt_wm_step(self, batch) -> None:
        if self._ltel is not None:
            self._ltel.wm_step(getattr(self.world_model, "last_step_stats",
                                       None), batch)

    def _lt_snapshot_timing(self) -> None:
        """BEFORE _log_progress, which resets the phase accumulators."""
        if self._ltel is not None:
            self._ltel.snapshot_timing(
                getattr(self, "_phase_acc", None),
                getattr(self, "_env_wait_n", 0),
                getattr(self, "_env_wait_sum", 0.0))

    def _lt_emit(self) -> None:
        """Gather the learner-side sections and write learning.jsonl. Called
        ONLY from _emit_metrics (the single emission site). Every producer
        is optional (getattr): a missing one logs null, never raises."""
        lt = self._ltel
        if lt is None:
            return
        sec: Dict[str, Any] = {}
        try:
            xt = getattr(self, "_explore_tracker", None)
            if xt is not None:
                xt.flush()
                sec["exploration"] = xt.segment_summary(reset=True)
        except Exception as exc:
            lt._err("exploration", exc)
        try:
            ppo = getattr(self.policy, "last_update_stats", None)
            # null unless a NEW update landed since the last record
            if ppo is not None and id(ppo) != lt._last_ppo_id:
                lt._last_ppo_id = id(ppo)
                sec["ppo"] = dict(ppo)
            else:
                sec["ppo"] = None
        except Exception as exc:
            lt._err("ppo", exc)
        try:
            _pg = getattr(self.curiosity, "pop_gate_stats", None)
            sec["curiosity"] = dict(_pg()) if callable(_pg) else None
        except Exception as exc:
            lt._err("curiosity", exc)
        try:
            rb = self.replay_buffer
            _ps = getattr(rb, "pop_sample_stats", None)
            rp = dict(_ps()) if callable(_ps) else {}
            rp["size"] = int(len(rb))
            rp["capacity"] = getattr(rb, "capacity", None)
            sec["replay"] = rp
        except Exception as exc:
            lt._err("replay", exc)
        try:
            inf = getattr(self, "infra", None)
            sec["ledger_signed"] = dict((getattr(
                inf, "last_ledger_segment", {}) or {}).get("signed") or {}) \
                if inf is not None else None
        except Exception as exc:
            lt._err("ledger", exc)
        _ms = getattr(self, "_metrics_sink", None)
        lt.record(int(getattr(self, "total_timesteps", 0)), sec,
                  extra_dropped=int(getattr(_ms, "dropped", 0) or 0))

    def _oracle_observe(self, step_infos, n: int) -> None:
        """Consume the RED channel. EVALUATION ONLY — writes nothing the
        agent can read.

        WHAT THIS MEASURES TODAY. `dead_reckon` integrates the agent's own
        sensed body-relative displacement from an episode-local origin; the
        engine's true position says where it actually went. The difference is
        DRIFT, and until now it was an assumption. Reported as a running
        mean and as drift-per-block-travelled, which is the scale-free form —
        an agent that has walked 500 blocks should not look worse than one
        that walked 5 for the same absolute error.

        WHY THIS IS THE ONLY CONSUMER. `info["oracle"]` is a separate dict
        from `info["sensors"]`; the replay buffer never carries it, and
        SensorBus.read_policy never builds it. A meaning has no route into
        the world model, and this method is deliberately the single place
        that touches one — so the isolation test has exactly one thing to
        check rather than a habit to audit.

        EXPLORATION (2026-10-07, telemetry). The ExplorationTracker is fed
        HERE and nowhere else, for EVERY stream (the drift readout below is
        stream 0's only). It writes the evaluator-only position trace and
        the learning.jsonl "exploration" section — files, never a path back
        into the agent.
        """
        # ONE read of the RED key per stream (the isolation test counts the
        # literal): both the tracker and the drift readout use this list.
        _orc = [((_s or {}).get("oracle") or {}) for _s in
                (step_infos or [])[:max(0, int(n))]]
        _xt = getattr(self, "_explore_tracker", None)
        if _xt is not None and _orc:
            try:
                _tw = time.time()
                for _e in range(len(_orc)):
                    _sx = step_infos[_e] or {}
                    _px = _orc[_e].get("true_position")
                    if _px is None:
                        continue
                    _wx = _sx.get("world") or {}
                    _xt.observe(
                        _e, int(self.total_timesteps), _tw, _px,
                        yaw=_wx.get("yaw"), pitch=_wx.get("pitch"),
                        episode=(self._ltel.episode_id(_e)
                                 if self._ltel is not None else None))
            except Exception as exc:                   # pragma: no cover
                if not getattr(self, "_explore_warned", False):
                    self._explore_warned = True
                    logger.warning("exploration tracker failed: %r", exc)
        if not self._oracle_enabled:
            return
        _si = (step_infos[0] or {}) if step_infos else {}
        pos = (_orc[0] if _orc else {}).get("true_position")
        if pos is None:
            return
        pos = np.asarray(pos, dtype=np.float64)
        w = _si.get("world") or {}
        dr = (w.get("dr_x"), w.get("dr_z"))
        if dr[0] is None:
            return
        if self._oracle_origin is None:
            # The dead reckoner's origin is the episode start, so truth has
            # to be measured from the same instant or the comparison is an
            # offset rather than a drift.
            self._oracle_origin = pos.copy()
            self._oracle_travel = 0.0
            self._oracle_last = pos.copy()
            return
        self._oracle_travel += float(np.hypot(*(pos - self._oracle_last)[[0, 2]]))
        self._oracle_last = pos.copy()
        true_dx = float(pos[0] - self._oracle_origin[0])
        true_dz = float(pos[2] - self._oracle_origin[2])
        err = float(np.hypot(float(dr[0]) - true_dx, float(dr[1]) - true_dz))
        self._oracle_drift.append(err)
        self._oracle_steps += 1
        if self._oracle_steps % max(1, self._oracle_every) == 0:
            _m = float(np.mean(self._oracle_drift)) if self._oracle_drift else 0.0
            _rel = _m / max(1.0, self._oracle_travel)
            logger.info(
                "ORACLE (evaluation only): dead-reckoning drift mean %.2f "
                "blocks over %.0f blocks travelled (%.1f%% of path); "
                "estimate (%.1f, %.1f) vs true (%.1f, %.1f)",
                _m, self._oracle_travel, 100.0 * _rel,
                float(dr[0]), float(dr[1]), true_dx, true_dz)
            self.training_metrics.setdefault(
                "oracle_dr_drift", deque(maxlen=100)).append(_m)

    def _oracle_reset(self) -> None:
        """A new world is a new frame of reference for both sides."""
        self._oracle_origin = None
        self._oracle_last = None
        self._oracle_travel = 0.0

    def _spatial_width(self) -> int:
        """THE single declaration of how many fields the spatial maps add.

        Read by the policy's proprio_dim and written by _spatial_fields.
        Two call sites, one number — the arrangement that exists because the
        reach sense was once appended in one loop body and declared in
        neither, and was silently dead in every run.
        """
        if not self._spatial_on:
            return 0
        return (self._probe_grid * self._probe_grid        # G1 sweep map
                + self._occ_grid * self._occ_grid          # G2 occupancy
                + self._sr_grid * self._sr_grid            # P2 successor
                + self._walk_bins)                         # P3 walkability

    def _spatial_fields(self, e_i: int) -> List[float]:
        """The spatial maps as a flat list, or neutral when unavailable.

        PRIMARY STREAM ONLY carries real values, exactly like reach and the
        flow senses: the maps are built from the primary's latent once per
        step. A scout reads the neutral row, which for every one of these is
        zero — "I have no record of anything there" — except walkability,
        whose neutral is 0.5 because zero would be a CLAIM that the world is
        a solid block, and that is the belief that stops an agent trying.
        """
        n_probe = self._probe_grid * self._probe_grid
        n_occ = self._occ_grid * self._occ_grid
        n_sr = self._sr_grid * self._sr_grid
        if e_i != 0:
            return ([0.0] * (n_probe + n_occ + n_sr)
                    + [0.5] * self._walk_bins)
        out: List[float] = []
        for arr, n, neutral in (
                (self._probe_map_now, n_probe, 0.0),
                (self._occ_now, n_occ, 0.0),
                (self._sr_now, n_sr, 0.0),
                (self._walk_now, self._walk_bins, 0.5)):
            if arr is None:
                out.extend([neutral] * n)
                continue
            flat = np.asarray(arr, dtype=np.float32).reshape(-1)
            if flat.shape[0] != n:
                out.extend([neutral] * n)
            else:
                out.extend(float(x) for x in flat)
        return out

    def _is_move_action(self, action_idx: int) -> bool:
        """Does this macro command forward translation?

        Reuses the vision scaffold's EXISTING declaration of which actions
        are forward motion rather than adding a second, driftable copy of
        that fact — the same source `_flow_probe_idx` reads.
        """
        return int(action_idx) in set(
            ((self.config.get("llm", {}) or {}).get("vision", {}) or {}).get(
                "forward_actions", [1, 2, 6]) or [])

    def _spatial_step(self, latent_row, step_infos,
                      commanded_move: bool = False) -> None:
        """Advance the three spatial maps by one step. Primary stream only.

        ORDER MATTERS AND IS NOT ARBITRARY: the maps are re-registered by the
        ego-motion of the step that has JUST HAPPENED, then written with the
        frame that resulted from it. Writing first would place this frame's
        reading at the body's previous pose.

        Costs one extra FlowHead forward on a latent the loop already has —
        the same deal _reach_sense and the flow probe get. Nothing here
        touches the env, renders, or calls a VLM.
        """
        if not self._spatial_on:
            return
        try:
            w = ((step_infos[0] or {}).get("world") or {}) if step_infos else {}
            d_yaw = math.radians(float(w.get("yaw_delta", 0.0) or 0.0))
            fwd = float(w.get("move_fwd", 0.0) or 0.0)
            lat = float(w.get("move_lat", 0.0) or 0.0)

            probe = torch.zeros(1, self.action_dim, device=self.device)
            probe[0, self._flow_probe_idx] = 1.0
            pm = self.world_model.probe_map(
                latent_row, probe, map_size=self._probe_grid)
            self._probe_map_now = (
                None if pm is None
                else pm[0, 0].detach().cpu().numpy().astype(np.float32))

            self._occ_now = self._occupancy.step(
                self._probe_map_now, d_yaw, fwd, lat)
            self._sr_now = self._successor.step(
                float(w.get("dr_x", 0.0) or 0.0),
                float(w.get("dr_z", 0.0) or 0.0))
            # "Did I ASK to move" is the action, not the outcome — that is
            # the whole point: the label comes from the gap between them.
            self._walk_now = self._walkable.step(
                bool(commanded_move),
                float(w.get("moved", 0.0) or 0.0), d_yaw)
        except Exception as _e:
            logger.debug("spatial step failed: %s", _e)

    def _spatial_reset(self) -> None:
        """A new world is a new map. World persistence is impossible in this
        fork (measured), so carrying any of these across a reset would be
        remembering a place that no longer exists."""
        if not self._spatial_on:
            return
        self._occupancy.reset()
        self._successor.reset()
        self._walkable.reset()
        self._occ_now = self._sr_now = self._walk_now = None
        self._probe_map_now = None

    def _wm_proprio_batch(self, step_infos, n: int):
        """(n, proprio_dim) tensor of RAW env body vectors, or None.

        None when the world model was built without a body — which is the
        signal WorldModel.embed reads to skip the concat entirely, so a
        non-Minecraft config never allocates this.

        A body that is missing for one stream reads NEUTRAL ZEROS rather than
        failing the batch: a scout whose client is rebuilding still has to
        produce a latent this step.
        """
        if not self._wm_proprio_dim:
            return None
        d = self._wm_proprio_dim
        out = np.zeros((n, d), dtype=np.float32)
        for e_i in range(n):
            _si = (step_infos[e_i] or {}) if e_i < len(step_infos) else {}
            # `sensors` is the bus transport; `proprio` is the pre-bus
            # 13-field vector and remains the fallback so a config without a
            # `sensors:` block behaves exactly as it did.
            v = _si.get("sensors")
            if v is None:
                v = _si.get("proprio")
            if v is None:
                continue
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            out[e_i, :min(d, v.shape[0])] = v[:d]
        return torch.from_numpy(out).to(self.device)

    def _augment_proprio(self, pp, e_i: int):
        """LOOP-SIDE BODY SENSES, appended to the env's proprio vector in ONE
        place for every loop body (2026-08-09).

        WHY A SHARED HELPER: the reach append previously lived only in the
        parallel-episodic body — the lifelong body stored the env's raw
        vector, `_prep_proprio` zero-padded it, and the policy's reach
        FEELING was silently dead in every skybot run (the SIXTH
        duplicated-body casualty). One assembly point ends the class.

        Fields appended (primary stream carries real values; scouts read the
        neutral 0.0 every missing sense uses):
          [reach]                    P(something breakable within arm's reach)
          [memory validity*proximity, sin(bearing_rel), cos(bearing_rel)]
                                     the EPISODIC pull (infra #38): where, in
                                     body-relative terms, the goal object was
                                     last seen. This is what turns wandering
                                     into returning — acting on one's own
                                     remembered past instead of only the
                                     current frame. Fades with age (memory,
                                     not a beacon) and reads neutral when
                                     nothing has ever been sighted.
        """
        if pp is None:
            return pp
        # WIDTH IS UNCONDITIONAL ONCE A SENSE EXISTS. The symbolizer guard
        # below covers only the four symbolizer-DERIVED fields; the flow
        # senses come from the world model and must be appended whether or
        # not a symbolizer was built, or the policy's proprio_dim and this
        # vector disagree by four and every row is silently misread.
        _flow_on = bool(getattr(
            getattr(self, "world_model", None), "flow_enabled", False))
        if self.symbolizer is None and not _flow_on:
            return pp
        try:
            # THE FOUR SYMBOLIZER-DERIVED FIELDS EXIST ONLY IF IT DOES. Their
            # presence is what the policy's proprio_dim declaration keys on,
            # so appending four zeros in a symbolizer-less run would make this
            # vector four wider than the policy expects and shift every flow
            # sense into the wrong slot.
            ext = [0.0, 0.0, 0.0, 0.0] if self.symbolizer is not None else []
            if e_i == 0 and self.symbolizer is not None:
                ext[0] = float(getattr(self, "_reach_now", 0.0))
                if (self.infra is not None
                        and self.infra.episodic is not None
                        and self.vision_scaffold is not None):
                    wi = getattr(self, "_last_world_info", None) or {}
                    x, z, yw = wi.get("x"), wi.get("z"), wi.get("yaw")
                    cats = list(getattr(self.vision_scaffold, "_seek_cats",
                                        None) or [])
                    if x is not None and z is not None and cats:
                        rec = None
                        for kind in ("sighting", "break"):
                            r = self.infra.episodic.last(kind, cats[0])
                            if r and r.get("position") and (
                                    rec is None
                                    or r["step"] > rec["step"]):
                                rec = r
                        if rec is not None:
                            tx, _ty, tz = rec["position"]
                            dx, dz = float(tx) - float(x), \
                                float(tz) - float(z)
                            dist = (dx * dx + dz * dz) ** 0.5
                            age = max(0, self.total_timesteps
                                      - int(rec["step"]))
                            validity = math.exp(-age / 20000.0)
                            proximity = 1.0 / (1.0 + dist / 32.0)
                            bearing = math.atan2(dx, dz)
                            # MINECRAFT YAW SIGN: yaw 0 faces +z (south) and
                            # yaw 90 faces WEST (-x) — yaw grows CLOCKWISE
                            # while atan2(dx,dz) grows counter-clockwise, so
                            # body-relative bearing is bearing PLUS yaw.
                            # (bearing - yaw read "dead ahead" as "behind";
                            # the yaw=0 test case cannot see the sign.)
                            rel = bearing + (math.radians(float(yw))
                                             if yw is not None else 0.0)
                            ext[1] = float(validity * proximity)
                            ext[2] = float(math.sin(rel))
                            ext[3] = float(math.cos(rel))
            if _flow_on:
                # ---- PERSPECTIVE FROM MOTION: four geometric senses -----
                # PRIMARY STREAM ONLY carries real values, exactly like
                # reach: the senses are read off the primary's latent once
                # per step. A scout reads the neutral row — 0 flow, 0.5
                # ratio (neither figure nor ground), 0 mover — which is the
                # honest encoding of "this body has no such sense", not a
                # claim that nothing is moving.
                fn = (getattr(self, "_flow_now", None) if e_i == 0 else None)
                fn = fn or self._FLOW_NEUTRAL
                ext.extend([
                    float(fn.get("flow_fovea", 0.0)),
                    float(fn.get("flow_edge", 0.0)),
                    float(fn.get("flow_ratio", 0.5)),
                    float(fn.get("mover", 0.0)),
                ])
                # ---- P4: WHERE the model is wrong ----------------------
                # POLICY ONLY, deliberately. This is derived FROM the world
                # model, so feeding it back into the world model's own input
                # would be circular — the same argument that keeps the flow
                # senses out of the sensor bus and in this vector instead.
                _sm = (getattr(self, "_surprise_now", None)
                       if e_i == 0 else None)
                _n = self._surprise_grid * self._surprise_grid
                if _sm is not None and len(_sm) == _n:
                    ext.extend(float(x) for x in _sm)
                else:
                    ext.extend([0.0] * _n)
            if self._spatial_on:
                # ---- G1/G2/P2/P3: the maps that outlive the frame -------
                # POLICY ONLY, like every other world-model-derived sense:
                # feeding a model's own output back into its input is
                # circular. Widths here must mirror _spatial_width() exactly
                # — that function is the only declaration and this is the
                # only writer, and a mismatch is silent.
                ext.extend(self._spatial_fields(e_i))
            return np.concatenate([np.asarray(pp, dtype=np.float32),
                                   np.asarray(ext, dtype=np.float32)])
        except Exception:
            return pp

    def _memory_pull_phi(self):
        """Potential on closeness to the freshest remembered goal site
        (sighting/achievement/demo of a seek category), in [0,1] — and the
        step of the record it points at, so a NEW memory re-adopts the
        baseline without paying (the magnet's just-switched rule).

        WHY (2026-08-10): the episodic bearing sense INFORMED the policy but
        nothing MOTIVATED acting on it — memory was a map with no pull. This
        is the bridge from "I remember where trees were" to "I go back":
        phi = validity(age) * proximity(distance), telescoping, so returning
        pays once, loitering pays zero, and leaving charges back — with the
        pull fading as the memory ages (a memory, not a beacon)."""
        try:
            if (self.infra is None or self.infra.episodic is None
                    or self.vision_scaffold is None):
                return None, None
            wi = getattr(self, "_last_world_info", None) or {}
            x, z = wi.get("x"), wi.get("z")
            cats = list(getattr(self.vision_scaffold, "_seek_cats",
                                None) or [])
            if x is None or z is None or not cats:
                return None, None
            rec = None
            for kind in ("sighting", "break", "demo"):
                r = self.infra.episodic.last(kind, cats[0])
                if (r and r.get("position")
                        and (rec is None or r["step"] > rec["step"])):
                    rec = r
            if rec is None:
                return None, None
            tx, _ty, tz = rec["position"]
            dist = ((float(tx) - float(x)) ** 2
                    + (float(tz) - float(z)) ** 2) ** 0.5
            age = max(0, self.total_timesteps - int(rec["step"]))
            phi = math.exp(-age / 20000.0) / (1.0 + dist / 32.0)
            return float(phi), int(rec["step"])
        except Exception:
            return None, None

    def _loop_damp_factor(self) -> float:
        """The farm detector's multiplier for THIS step's stream-0 income,
        in (0, 1]; 1.0 when infra is off or the value is unusable.

        Applied to stream-0 INTRINSIC by both waking bodies (and, as before,
        to the magnet's `_sr`). Never to rewards[0]: real environment reward
        stays honest (_farm_damping_smoke contract 4).

        RE-OPEN CONDITION (CLAUDE.md §4.1). Nothing here stores state: the
        value is BehaviouralLoopDetector.loop_damp(cell), recomputed every
        step from the lap count and income EMA, both of which decay once the
        agent stays out of the cell for more than revisit_horizon steps — so
        leaving the loop restores full pay, by an action the agent can take.

        SCOUTS ARE NOT DAMPED, deliberately: the detector observes stream 0's
        cells only (on_step is a primary-only hook), so there is no per-scout
        loop evidence to damp on. _scout_mixed_reward lists it as omitted."""
        if self.infra is None:
            return 1.0
        try:
            d = float(getattr(self.infra, "last_loop_damp", 1.0) or 1.0)
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(d) or d <= 0.0:
            return 1.0
        return min(1.0, d)

    def _infra_step(self, info, action, rssm_state, step_income,
                    raw_extrinsic) -> float:
        """Per-step general-infrastructure hook, shared by both waking loop
        bodies (same dedup rationale as _magnet_step_shaping — the fifth
        copy-drift incident is why per-step concerns are single functions).
        Feeds the farm detector / drift / affordance / episodic / trace
        monitors with the adapter's typed event stream, and returns the
        EMPOWERMENT shaping delta to add to intrinsic (0.0 when off).
        step_income = everything the learner is being paid this step
        (intrinsic + shaped extrinsic) — the quantity a farm farms."""
        # ---- LIVE HEARTBEAT (throttled, cheap, never raises) ---------------
        # BEFORE the `infra is None` guard on purpose: the tracker must not
        # switch itself off because an unrelated subsystem is disabled.
        # Placed in _infra_step because it is the ONE per-step function shared
        # by both waking bodies — the same dedup rationale as the docstring
        # above, and the reason a per-step concern is never written twice.
        # Cost is a wall-clock compare on the common path; the write happens
        # roughly once every heartbeat_seconds, not once per step.
        self._emit_heartbeat()
        if self.infra is None:
            return 0.0
        try:
            # marks the stuck-metrics as GENUINELY FED this segment — a run
            # whose body never calls this hook (vector envs on the serial
            # path) must not be judged "stuck" on metrics nobody supplied
            self._infra_fed = True
            self._seg_extrinsic_sum = getattr(
                self, "_seg_extrinsic_sum", 0.0) + float(raw_extrinsic)
            wi = (info or {}).get("world") or {}
            pos = None
            if wi.get("x") is not None and wi.get("z") is not None:
                pos = (float(wi["x"]), float(wi.get("y", 0.0)),
                       float(wi["z"]))
            ev = (info or {}).get("events") or []
            # Computed ONCE, reused by both infra.on_step below AND
            # anticipation (2026-09-02) — was being built inline at the
            # on_step call only; a second caller needing it would otherwise
            # have to either duplicate the call or read a stale copy.
            _evc = self._event_count_map(info) if ev else None
            # excavation stamp for the coverage discovery-gate: breaking or
            # placing marks the near future's "new cells" as manufactured
            if any(k in ("break", "place") for k, _ in ev):
                self._last_excav_step = self.total_timesteps
            # ---- DEMONSTRATION -> GOAL EMULATION (2026-08-09) ------------
            # An observed_change (the world changed, not by this body) with
            # the other player recently under the gaze = a DEMONSTRATION.
            # The most co-present world category in the fovea becomes the
            # inferred goal of the demonstration and is socially primed:
            # curiosity injected, search refilled. Watching the teacher fell
            # a tree makes trees hot; the agent's own body still discovers
            # the swing (goal emulation, not motor mimicry).
            # window in the fleet-scaled clock (200 primary steps), with a
            # REFRACTORY period so a recurring change source can neither
            # refill the seek budget without bound nor flood the log
            # (review 2026-08-09)
            _fleet_w = 200 * max(1, int(getattr(self, "_num_envs", 1) or 1))
            if (self.vision_scaffold is not None
                    and any(k == "observed_change" for k, _ in ev)
                    and self.total_timesteps
                    - getattr(self, "_player_seen_step", -10**9) <= _fleet_w
                    and self._vision_clock
                    >= getattr(self, "_demo_cooldown_until", 0)):
                _fps_d = getattr(self, "_last_fovea_probs", None) or {}
                _fcs_d = getattr(self, "_last_fovea_counts", None) or {}
                # candidates filtered to PRIMEABLE categories FIRST (review:
                # a background like dirt under the crosshair used to win the
                # max() and the prime silently no-opped — the demonstration
                # was lost). Primeable = the magnet's own steerable targets.
                _primeable = set(self.vision_scaffold.target_categories)
                _cand = [(float(p), c) for c, p in _fps_d.items()
                         if c in _primeable
                         and c not in ("player_visible", "sky_visible")
                         and _fcs_d.get(c, 0) >= 5 and float(p) >= 0.25]
                if _cand:
                    _top = max(_cand)[1]
                    if self.vision_scaffold.social_prime(
                            _top, self._vision_clock):
                        self._demo_cooldown_until = self._vision_clock + 500
                        self._demos_seen = getattr(
                            self, "_demos_seen", 0) + 1
                        if (self.infra is not None
                                and self.infra.episodic is not None):
                            _wi_d = (info or {}).get("world") or {}
                            if _wi_d.get("x") is not None:
                                self.infra.episodic.record(
                                    "demo", _top, self.total_timesteps,
                                    (float(_wi_d["x"]),
                                     float(_wi_d.get("y", 0.0)),
                                     float(_wi_d.get("z", 0.0))))
                        logger.info(
                            "DEMONSTRATION witnessed (#%d): observed change "
                            "with the player in view — socially primed %r",
                            self._demos_seen, _top)
            g = float(self.config.get("policy", {}).get("gamma", 0.99))
            _extra = float(self.infra.empowerment_shaping(
                self.world_model, rssm_state, int(self.action_dim),
                self.total_timesteps, g,
                wm_lock=getattr(self, "_wm_param_lock", None)))
            # ---- compression-progress curiosity (infra #13) --------------
            # probes are offered and evaluated on the module's own cadence;
            # the WM lock is taken NON-BLOCKING (same acting-loop rule as
            # empowerment: a skipped reading costs nothing, a stall costs a
            # server kick)
            if self._progress is not None:
                self._progress.tick(self.total_timesteps)
                if (self.total_timesteps
                        - getattr(self, "_probe_last_offer", -10**9)
                        >= self._progress_every):
                    self._probe_last_offer = self.total_timesteps
                    try:
                        _pb = self.replay_buffer.sample_sequences(
                            batch_size=1,
                            seq_len=int(self.config.get(
                                "world_model", {}).get(
                                "sequence_length", 16)),
                            device=torch.device("cpu"))
                        self._progress.maybe_add_probe(_pb)
                    except Exception:
                        pass

                def _wm_loss(pb):
                    from developmental_ai.foundation.runtime.observable import sequence_loss
                    return sequence_loss(self.world_model, pb)
                _lk = getattr(self, "_wm_param_lock", None)
                # Cadence FIRST: the lock and the optimizer-state scan in
                # _world_model_version() are only paid on a due step.
                if self._progress.due(self.total_timesteps) and (
                        _lk is None or _lk.acquire(blocking=False)):
                    try:
                        self._progress.evaluate(_wm_loss,
                                                self.total_timesteps,
                                                device=self.device,
                                                model_version=self._world_model_version(),
                                                objective="deterministic-observation-mse-v1")
                    finally:
                        if _lk is not None:
                            _lk.release()
                    self.infra.progress_measurement = self._progress.stats
                # Measurement only: model improvement does not establish
                # credit for the action being taken now. No ambient payment
                # — there is deliberately no record_reward for it.
                event = self._progress.last_event
                if event is not None and event.event_id != getattr(self, "_last_progress_event", None):
                    self._last_progress_event = event.event_id
                    logger.info("paired-progress shadow event=%s model=%s improvement=%.6g "
                                "forgetting=%.6g se=%.6g probes=%d reward=0",
                                event.event_id, event.model_version, event.improvement,
                                event.forgetting, event.standard_error, len(event.probe_ids))
            # ---- MEMORY-PULL POTENTIAL (2026-08-10, point 3) -------------
            # gamma-potential on remembered-goal-site closeness; a CHANGED
            # memory record re-adopts the baseline unpaid (else every fresh
            # sighting would gift the jump in phi).
            _mw = float(getattr(self, "_memory_pull_weight", 0.0) or 0.0)
            if _mw > 0.0:
                if any(k == "death" for k, _s in (ev or [])):
                    # respawning NEARER the remembered site must not pay
                    # (die-to-travel would be a farm) — re-adopt unpaid
                    self._mem_pull_prev = None
                _mp, _mrec = self._memory_pull_phi()
                if _mp is not None:
                    if (getattr(self, "_mem_pull_rec", None) != _mrec
                            or getattr(self, "_mem_pull_prev", None)
                            is None):
                        self._mem_pull_rec = _mrec
                        self._mem_pull_prev = _mp
                    else:
                        _mf = _mw * (g * _mp - self._mem_pull_prev)
                        self._mem_pull_prev = _mp
                        if _mf:
                            _extra += _mf
                            self.infra.record_reward("memory_pull", _mf)
            # ---- CONSEQUENCE FRONTIER (2026-08-13) -----------------------
            if self.consequence is not None:
                # A: evidence. `caused` counts ONLY self-caused world events —
                # ambient change must never close a consequence set (watching
                # a tree sway is not affecting it), which is why
                # observed_change is excluded.
                _caused = any(k in ("break", "place", "craft", "pickup")
                              for k, _s in (ev or []))
                self.consequence.observe(
                    getattr(self, "_last_fovea_probs", None), caused=_caused,
                    # a swing that ended with nothing broken: evidence that
                    # this thing does not respond to me (see observe)
                    attempted=bool((info or {}).get("swing_failed", 0)))
                # respawn/rebuild hands back an inventory the agent did not
                # earn — and so does a human granting a tool
                if any(k in ("death",) for k, _s in (ev or [])) or \
                        (info or {}).get("env_restarted"):
                    self.consequence.suppress(self.total_timesteps + 2)
                # B: possession frontier — one-shot per never-held set
                _inv = (info or {}).get("inventory")
                if _inv:
                    _pi, _new = self.consequence.possession_income(
                        _inv, self.total_timesteps)
                    if _pi:
                        _extra += _pi
                        self.infra.record_reward("frontier", _pi)
                    if _new:
                        logger.info(
                            "POSSESSION FRONTIER: first time holding %s",
                            self.consequence._key(_inv))
            # ---- ANTICIPATION (2026-09-02) --------------------------------
            # Called EVERY step (not just when ev is non-empty): the
            # "sighting" half has to accumulate on ordinary presence, and
            # only the "payout" half needs an event. Placed here rather than
            # inside _magnet_step_shaping specifically because THIS method
            # already has `ev` and `_evc` as fresh locals from the exact
            # same step — no cache-on-self, no risk of the one-step
            # staleness that _last_world_info was found to have earlier.
            #
            # The probs come from a SEPARATE small forward through the
            # grounding head (~2ms measured for a 5-member 512-hidden
            # ensemble): _magnet_step_shaping already has this same head's
            # output as `_op`, but it and this method are not called from
            # the same place in every stepping body, so recomputing here is
            # the correctness-over-cheapness choice — the alternative
            # (threading `_op` across methods via `self`) is exactly the
            # cache-staleness class of bug this comment is deliberately
            # avoiding.
            if self.anticipation is not None:
                try:
                    with self._wm_param_lock, torch.no_grad():
                        _ant_latent = self.world_model.rssm.get_latent(
                            rssm_state)[0:1]
                    _ant_probs, _, _ = self._grounded_object_probs(
                        _ant_latent)
                    _ant_pay = self.anticipation.observe(
                        _ant_probs, ev, _evc, self._habituation_scale,
                        self.total_timesteps)
                    if _ant_pay:
                        _extra += _ant_pay
                        self.infra.record_reward("anticipation", _ant_pay)
                        self._anticip_sum = getattr(
                            self, "_anticip_sum", 0.0) + _ant_pay
                        self._anticip_payouts = getattr(
                            self, "_anticip_payouts", 0) + 1
                    self._anticip_n = getattr(self, "_anticip_n", 0) + 1
                except Exception as _ae:
                    logger.debug("anticipation step failed: %s", _ae)
            # ---- METABOLIC EFFORT COST (2026-08-09) ----------------------
            # Effortful actions (the adapter's "effort" contract field) cost
            # a small constant. Observed live: sustained attack at CLOUDS —
            # under a whitelisted economy futile swings were exactly free,
            # so entropy kept them common. Effort pricing is the general,
            # homeostatic extinguisher: air-punching now bleeds, while a
            # full barehanded chop (~60 effort ticks ≈ −0.09) is trivially
            # repaid by the +20 log. Intrinsic channel, ledger-visible.
            _ec = float(getattr(self, "_effort_cost", 0.0) or 0.0)
            if _ec > 0.0:
                _ef = float((info or {}).get("effort", 0.0) or 0.0)
                if _ef:
                    _extra -= _ec * _ef
                    self.infra.record_reward("effort", -_ec * _ef)
            # monitors see the WHOLE income including the shaping computed
            # just above (review finding: a farm running ON empowerment or
            # progress income would have been invisible to its own police)
            self.infra.on_step(
                step=self.total_timesteps, action=int(action), events=ev,
                cell=wi.get("cell"), position=pos, pitch=wi.get("pitch"),
                fovea_probs=getattr(self, "_last_fovea_probs", None),
                # POSITIVE counts: the stack's consumers make PRESENCE
                # claims (sighting landmarks, event association) — gate on
                # seen, not on trained (review 2026-08-09)
                fovea_counts=getattr(self, "_last_fovea_pos", None),
                step_reward=float(step_income) + float(_extra),
                event_counts=_evc)
            return _extra
        except Exception:
            return 0.0

    def _gui_reward(self, stream, info):
        """Extrinsic-only state costs; closing the GUI resets dwell immediately.

        Both primary collection paths and scouts use the same pure calculation.
        Existing boundary code resets these state holders without a refund.
        """
        from developmental_ai.core.reward_components import gui_costs
        state = {} if stream == 0 else self._sc_phi.setdefault(stream, {})
        run = getattr(self, "_gui_run", 0) if stream == 0 else state.get("gui_run", 0)
        prev = getattr(self, "_gui_dwell_phi", None) if stream == 0 else state.get("gui")
        costs = gui_costs(bool((info or {}).get("gui_open")), run, prev,
            weight=float(getattr(self, "_gui_dwell_weight", 0.0)),
            dwell_steps=float(getattr(self, "_gui_dwell_steps", 200.0)),
            step_cost=float(getattr(self, "_gui_dwell_step_cost", 0.0)),
            grace_steps=int(getattr(self, "_gui_dwell_grace_steps", 40)))
        if stream == 0:
            self._gui_run, self._gui_dwell_phi = costs.run, costs.potential
        else:
            state["gui_run"], state["gui"] = costs.run, costs.potential
        return costs.total

    def _scout_mixed_reward(self, e_i: int, info, raw_ext: float,
                            intrinsic_e: float) -> float:
        """The reward a SCOUT stream's PPO row carries (2026-09-01).

        ---- WHY THIS EXISTS, AND WHAT IT DELIBERATELY OMITS --------------
        Until now only stream 0 stored PPO rows. The scouts' curiosity was
        computed every step and thrown away, and their transitions reached
        the world model but never the policy — so on a 2-client fleet, half
        the wall clock (and half the server's client budget, which is the
        binding constraint) produced no policy gradient at all. It is also
        what starved the update of rows: at mean tau ~40 a segment yielded
        ~25 of them.
        ONE POLICY MUST NOT BE TRAINED ON TWO REWARD FUNCTIONS. That is a
        worse failure than a small batch, so the rule here is: a scout is
        paid by exactly those terms that are COMPUTABLE PER STREAM from that
        stream's own step_info, and by nothing else.

          included: env reward, base curiosity (ICM/LP), persistence,
                    GUI-dwell cost, gaze-level cost, GUI intrinsic zeroing
          omitted:  the vision magnet, infra/empowerment shaping,
                    the farm loop damp (the detector sees stream 0's cells
                    only — see _loop_damp_factor), habituation damping, imagination curiosity, symbol
                    novelty, symbol centring, view novelty, territory
                    coverage, the reach potential, the gaze-bucket bonus

        THE OMISSION LIST IS LONG AND THIS DOCSTRING PREVIOUSLY NAMED TWO OF
        IT (corrected 2026-09-01). The point of writing it down is to make
        "one policy, two reward functions" auditable; a list that understates
        the gap fourfold does the opposite. Every omitted term is
        primary-only for one of two reasons: it needs the VLM/fovea (magnet,
        symbol novelty/centring, reach), or it rides on single-stream shaping
        state the loop keeps for env 0 alone (habituation, imagination,
        coverage, gaze bucket, infra).

        WHAT THAT MEANS IN PRACTICE: a scout's reward is close to raw
        curiosity plus the env's own reward, while the primary's is heavily
        shaped. Rows from both go into one update, so the shaped and unshaped
        views of the same policy are being averaged. That is still better
        than discarding half the fleet's experience — but it is a real cost,
        and it is why `magnet` in the income census is labelled stream 0's
        alone. Closing it fully would mean running the VLM on every client,
        which the fovea work already measured as unaffordable; closing it
        PARTLY is cheap and is the obvious next move — habituation is
        computable from `step_infos[e]` and is the one omitted term that
        needs nothing this method does not already have.

        Potentials are kept PER STREAM (`_sc_phi`) and reset on that
        stream's death — a potential carried across a world boundary pays a
        phantom delta on the first step of the new world.
        """
        _w = (info or {}).get("world") or {}
        _st = self._sc_phi.setdefault(e_i, {})
        _g = float(self.config.get("policy", {}).get("gamma", 0.99))
        _ext = float(raw_ext)
        _int = float(intrinsic_e)
        # telemetry parts (measurement only; same additions, same order)
        _lpi = {"icm_base": _int}

        # PERSISTENCE (telescoping: progress, so gamma applies)
        _pw = float(self._persist_weight)
        if _pw > 0.0:
            _phi = min(1.0, float(info.get("attack_run", 0.0) or 0.0)
                       / max(1.0, float(self._persist_ticks)))
            _i0 = _int
            _int += _pw * (_g * _phi - float(_st.get("persist", 0.0)))
            _lpi["persistence"] = _int - _i0
            _st["persist"] = _phi

        # GAZE LEVEL (plain difference: a STATE COST must pay 0 while pinned
        # — see the 2026-08-17 gamma-discounting incident)
        _plw = float(self._pitch_level_weight)
        _pv = _w.get("pitch")
        if _plw > 0.0 and _pv is not None:
            _php = -((min(90.0, abs(float(_pv))) / 90.0) ** 4)
            if "pitch" in _st:
                _i0 = _int
                _int += _plw * (_php - float(_st["pitch"]))
                _lpi["gaze_level"] = _int - _i0
            _st["pitch"] = _php

        # OCCLUSION: no world change, no world curiosity. Same category
        # argument as the primary — inside a GUI the observation changes a
        # great deal and the world does not change at all.
        _gui = bool(info.get("gui_open"))
        _i_pre_gui = _int
        if _gui:
            _lpi["gui_zero"] = -_int
            _int = 0.0

        _e0 = _ext
        _ext += self._gui_reward(e_i, info)

        # update_stats=False: the return EMAs are a per-STEP clock tuned on
        # ONE stream; letting N bodies tick it would scale the anneal by N.
        _mixed = self.reward_mixer.mix(_int, _ext, update_stats=False)
        if self._ltel is not None:
            self._ltel.book_parts(
                e_i, _lpi, {"env": _e0, "gui_dwell": _ext - _e0}, _mixed,
                self.reward_mixer, after_damp=_i_pre_gui,
                mults={"gui_zero": 0.0} if _gui else None)
        return _mixed

    def _magnet_step_shaping(self, action, lp_scalar, latent_row) -> float:
        """THE one magnet invocation, shared by all three loop bodies.

        The serial, parallel-episodic and lifelong bodies each carried their
        own copy of this block, and the copies kept drifting: the rotation-
        pay split existed only in one body for a whole run, the episodic
        copy silently dropped `reach_measured`, and the fovea kwargs had to
        be added three times. Fifth documented instance of the duplicated-
        body hazard (gui guard, viewer, magnet diagnostics, reach) — so the
        block now lives here once. Callers keep their own dream-gating and
        apply the returned shaping to their own extrinsic variable."""
        if (self.vision_scaffold is None
                or not self.vision_scaffold.wants_step(self._vision_clock)):
            return 0.0
        _op, _rel, _lab = self._grounded_object_probs(latent_row)
        # SIGNAL HEALTH (infra #1): a predicate whose output carries no
        # information (a learned constant — the tree_visible failure) is
        # excluded from steering until it recovers. The monitor observes
        # every head output; the DEGENERATE set is refreshed each segment.
        _excl = None
        if self.infra is not None:
            self.infra.signal_observe(_op)
            _excl = self.infra.degenerate_signals() or None
        # ---- IS THERE A TREE IN VIEW? (2026-09-01) -----------------------
        # THE leading indicator for the CLAUDE.md section 9 scoreboard.
        # Everything the goal needs — aim, swing, hold — is conditional on a
        # trunk being on screen, and nothing in the segment log ever said
        # whether one was. Without it "still no logs" and "never saw a
        # trunk" are the same reading, which is exactly the ambiguity that
        # let 399 breaks / 1 log stand unexplained.
        #
        # REPORTED WITH ITS TRUST STATE, not as a bare number. An untrained
        # head sits at sigmoid ~0.5, so a mean of 0.5 means "no
        # information", NOT "half the time" — the label count is what tells
        # the two apart. `DEGENERATE` means the signal-health monitor has
        # excluded it from steering, which is the llava tree_visible failure
        # recurring and would make every other figure on the line moot.
        # Lives here rather than in the step loops because this method is
        # the ONE magnet invocation shared by all three bodies (see the
        # docstring), so it cannot drift between them.
        _tp = float((_op or {}).get("tree_visible", 0.0))
        self._tree_sum = getattr(self, "_tree_sum", 0.0) + _tp
        self._tree_n = getattr(self, "_tree_n", 0) + 1
        if _tp >= 0.5:
            self._tree_hi = getattr(self, "_tree_hi", 0) + 1
        self._tree_rel = float((_rel or {}).get("tree_visible", 0.0))
        self._tree_lab = int((_lab or {}).get("tree_visible", 0))
        self._tree_degen = bool(_excl and "tree_visible" in _excl)
        # SOCIAL PRESENCE (2026-08-09): is the other player under the gaze?
        # Drives joint attention in the scaffold and stamps the last-seen
        # step the demonstration detector (in _infra_step) checks against.
        # Gated on POSITIVE teacher sightings (an untrained fresh head sits
        # at sigmoid ~0.5 and the label-count gate is free after ~5 fovea
        # labels — review 2026-08-09) and a threshold above cold-start
        # noise.
        _fps_s = getattr(self, "_last_fovea_probs", None) or {}
        _fpp_s = getattr(self, "_last_fovea_pos", None) or {}
        _social = (_fpp_s.get("player_visible", 0) >= 2
                   and float(_fps_s.get("player_visible", 0.0)) >= 0.65)
        # ...OR the FULL-FRAME channel (2026-08-09, live finding): the fovea
        # only certifies the teacher when they happen to stand in the centre
        # crop at a label instant, so meeting them took ages — the user
        # stood in front of the agent and nothing happened. The full frame
        # sees them anywhere in view.
        if not _social and self.symbolizer is not None:
            _pos_full = getattr(self.symbolizer, "pos_counts", {}) or {}
            _social = (_pos_full.get("player_visible", 0) >= 2
                       and float((_op or {}).get("player_visible", 0.0))
                       >= 0.6)
        if _social:
            self._player_seen_step = self.total_timesteps
        # proof-of-life beat: UNCONDITIONAL and guarded (review: an
        # indentation slip nested this under the social branch, making the
        # magnet read permanently overdue on any player-free stretch — and
        # an unguarded deref would crash on infra=None)
        if self.infra is not None:
            # (same clock domain as the heartbeat report: total_timesteps)
            self.infra.beat("magnet", self.total_timesteps)
        _sr = self.vision_scaffold.step_shaping(
            action=int(action) if self.is_discrete else -1,
            timestep=self._vision_clock,
            lp_scalar=float(lp_scalar),
            object_probs=_op, reliability=_rel, label_counts=_lab,
            reach_measured=float(getattr(self, "_reach_now", 0.0)),
            excluded=_excl,
            social_present=_social,
            **self._fovea_kwargs(latent_row))
        # rotation-pay split: seek is a potential on visibility and turning
        # is what changes visibility — if look-away/look-back does not
        # telescope to 0, spinning becomes a farm. Measured, not argued.
        _a0 = int(action) if self.is_discrete else -1
        if _a0 in (3, 4):
            self._mag_turn = getattr(self, "_mag_turn", 0.0) + float(_sr or 0.0)
            self._mag_turn_n = getattr(self, "_mag_turn_n", 0) + 1
        else:
            self._mag_other = getattr(self, "_mag_other", 0.0) + float(_sr or 0.0)
            self._mag_other_n = getattr(self, "_mag_other_n", 0) + 1
        self._vision_clock += 1
        return float(_sr or 0.0)

    def _grounded_skill_name(self, slot: int, preconditions: Dict[str, bool]
                             ) -> Tuple[str, str]:
        """A skill's name from GROUND TRUTH, not the VLM.

        Priority: the block whose break caused this achievement's reward
        spike ("break_oak_log"). Fallback: the dominant grounded scene
        predicate the head asserts ("act_where_tree_visible"). Last resort:
        a neutral slot name. Never a hallucinated action.
        """
        block = self._unlock_block.get(slot)
        if block:
            return (f"break_{block}",
                    f"Breaks {block} (grounded; first unlocked at episode "
                    f"{self.total_episodes}).")
        true_preds = [k for k, v in (preconditions or {}).items() if v]
        if true_preds:
            # SPECIFICITY, NOT ALPHABET (2026-07-25). `sorted(...)[0]` picked
            # the alphabetically-first asserted predicate, which across 23
            # predicates is nearly always the same one — 14 skills became
            # `act_where_breakable_in_reach` and 8 `act_where_dirt_visible`,
            # splitting one behaviour's practice across many slots so
            # competence never accumulated and NOTHING ever reached mastery
            # (0 of 51). Rank by RARITY instead: the least-frequently-asserted
            # predicate is the most informative thing about this behaviour.
            # The slot suffix then guarantees uniqueness, so two genuinely
            # different behaviours can never silently share an identity.
            counts = {}
            try:
                counts = dict(getattr(self.symbolizer, "label_counts", {}) or {})
            except Exception:
                counts = {}
            key = min(true_preds, key=lambda p: (counts.get(p, 0), p))
            return (f"act_where_{key}_s{slot:02d}",
                    f"Behaviour reliable when {key} (slot {slot}, episode "
                    f"{self.total_episodes}).")
        return (f"skill_slot_{slot:02d}",
                f"Discovered behaviour (episode {self.total_episodes}).")

    def _preconditions_from_latent(self, slot: int) -> Dict[str, bool]:
        """Grounded-predicate readout of a slot's unlock latent. Only
        predicates with enough VLM evidence assert (same evidence gate as
        the knowledge graph — an ungrounded head must not define a skill's
        applicability)."""
        out: Dict[str, bool] = {}
        lat = self._unlock_context.get(slot)
        if self.symbolizer is None or lat is None:
            return out
        try:
            from developmental_ai.llm.vlm_symbolizer import PREDICATES
            probs = self.symbolizer.head.predict(
                torch.from_numpy(lat).reshape(1, -1).to(
                    self.symbolizer.device)).squeeze(0)
            for i, pred in enumerate(PREDICATES):
                if (self.symbolizer.label_counts.get(pred, 0)
                        >= self.symbolizer.min_labels):
                    out[pred] = bool(probs[i] > 0.5)
        except Exception:
            pass
        return out

    def _write_skill_note(self, skill, slot: int, frame) -> None:
        """Render a skill's note.md from EARNED evidence only.

        effects        primary-stream block types broken this episode
                       (ground truth; refined later by KG causal facts)
        preconditions  grounded-predicate readout of the unlock latent
                       (only predicates with enough VLM evidence assert)
        prerequisites  the achieved-before mask of this slot's unlock event
                       (observed precedence, not similarity)
        """
        try:
            from developmental_ai.skill_bank import skill_notes
            skill_dir = os.path.join(
                self.skill_bank.storage_dir, skill.skill_id)
            skill_notes.save_discovery_frame(frame, skill_dir)

            preconditions = self._preconditions_from_latent(slot)

            prerequisites = []
            if (self.broadcaster is not None
                    and hasattr(self.broadcaster, "unlock_log")):
                for ev in reversed(self.broadcaster.unlock_log):
                    if int(ev["slot"]) == slot:
                        mask = np.asarray(ev["achieved_before"])
                        names = self.broadcaster.slot_names
                        prerequisites = [
                            f"ach_{i:02d}_{names[i]}"[:64]
                            for i in np.nonzero(mask > 0)[0]
                            if i < len(names)]
                        break

            skill_notes.write_note(
                skill, skill_dir,
                effects=sorted(self._prev_mine.keys()),
                preconditions=preconditions,
                prerequisites=prerequisites,
                provenance={
                    "episode": int(self.total_episodes),
                    "timestep": int(self.total_timesteps),
                    "env": str(self.config.get("environment", {})
                               .get("name", "?")),
                },
                competence_history=[(time.time(),
                                     float(skill.success_rate))],
            )
        except Exception as e:  # notes are provenance, never fatal
            logger.warning("skill note write failed for %s: %s",
                           skill.skill_id, e)

    def _log_progress(self) -> None:
        """Print training progress to stdout."""
        metrics = {}
        for key, values in self.training_metrics.items():
            if values:
                metrics[key] = np.mean(list(values))

        # LEDGER HARVEST (infra #10) — snapshot the per-term shaping sums
        # BEFORE their own print sections reset them further down. This is
        # the income statement: what was the learner actually paid for?
        _ledger_snapshot = {}
        if self.infra is not None:
            for _src, _attr in (("coverage", "_cov_sum"),
                                ("gaze", "_gaze_sum"),
                                ("novelty", "_nov_sum"),
                                ("symbols", "_sym_sum"),
                                ("sym_center", "_symc_sum"),
                                ("persistence", "_persist_sum"),
                                ("reach", "_reach_sum"),
                                ("gaze_level", "_pitch_level_sum"),
                                ("imagination", "_cen_imag")):
                _v = float(getattr(self, _attr, 0.0) or 0.0)
                if _v:
                    _ledger_snapshot[_src] = _v
            _mag = (float(getattr(self, "_mag_turn", 0.0) or 0.0)
                    + float(getattr(self, "_mag_other", 0.0) or 0.0))
            if _mag:
                _ledger_snapshot["magnet_seek"] = _mag
            _ext = float(getattr(self, "_seg_extrinsic_sum", 0.0) or 0.0)
            if _ext:
                _ledger_snapshot["extrinsic_env"] = _ext

        kg_stats = self.knowledge_graph.get_stats()
        skill_stats = self.skill_bank.get_stats()
        curiosity_stats = self.curiosity.stats

        print(f"\n{'='*60}")
        print(f"Episode {self.total_episodes} | Timestep {self.total_timesteps}")
        print(f"{'='*60}")
        _ep_r = self.training_metrics.get('episode_reward')
        _last_r = _ep_r[-1] if _ep_r else 0.0
        # print BOTH: the running mean AND this segment's raw reward. The mean is
        # a cumulative deque average (=sum/N) that decays like 8.46/N after one
        # big early episode and reads like a "decline" even when nothing changed;
        # the raw per-episode value is the honest recent signal.
        print(f"  Reward (avg):     {metrics.get('episode_reward', 0):.2f}"
              f"   (this ep: {_last_r:.2f})")
        print(f"  Episode length:   {metrics.get('episode_length', 0):.0f}")
        print(f"  Intrinsic reward: {metrics.get('intrinsic_reward', 0):.4f}")
        # `n=` is not decoration: a bare 0.0000 could mean "converged" or
        # "no data", and for 24 consecutive readings it silently meant the
        # latter while the trainer ran 32/32 blocks. The sample count makes
        # an empty deque impossible to mistake for a trained model.
        _wmq = self.training_metrics.get('world_model_loss') or ()
        print(f"  WM loss:          {metrics.get('world_model_loss', 0):.4f}"
              f"  (n={len(_wmq)}, recon="
              f"{metrics.get('reconstruction_error', 0):.4f})")
        print(f"  KL divergence:    {metrics.get('kl_divergence', 0):.4f}")
        print(f"  Policy loss:      {metrics.get('policy_loss', 0):.4f}")
        print(f"  Curiosity loss:   {metrics.get('curiosity_loss', 0):.4f}")
        # NOT-APPLICABLE IS NOT ZERO (2026-09-04). On pixel observations
        # `_skip_perdim_symbolic` is True, so `_extract_and_store_facts` is
        # never called and `symbolic_decoder` is a _NullSymbolicDecoder that
        # returns []. Both counters are therefore structurally 0 for SkyBot —
        # by design, since per-obs-dimension "facts" about individual pixels
        # are meaningless. But printing 0.0 made the knowledge graph look
        # DEAD, and cost a session's investigation chasing a stall that was
        # a correct gate. On pixels the KG is fed by the VLM symbolizer, and
        # `KG density` below is the number that actually means something.
        if self._skip_perdim_symbolic:
            print("  New facts/ep:     n/a (pixel obs: per-dim symbolic off; "
                  "see KG density)")
        else:
            print(f"  New facts/ep:     {metrics.get('new_facts', 0):.1f}")
        sd_stats = self.symbolic_decoder.stats
        print(f"  Sym decoder:      loss={metrics.get('symbolic_decoder_loss', 0):.4f}, "
              f"acc={metrics.get('symbolic_decoder_accuracy', 0):.1%}, "
              f"conf={sd_stats['symbolic_decoder_confidence']:.2f}")
        if self.world_model.inverse_dynamics_enabled:
            print(f"  Inverse dynamics: loss={metrics.get('inverse_dynamics_loss', 0):.4f}")
        # ---- WHY IS IMAGINATION ZERO? (2026-09-04) -------------------------
        # `imagination +0.0000/step` in the reward census read like a dead
        # subsystem. It is not: the bonus is scaled by
        #   boredom = 1 - min(1, global_lp / lp_reference)
        # and returns EARLY at boredom <= 0, so while external curiosity is
        # still teaching, contributing nothing is the designed behaviour —
        # this drive exists to take over WHEN learning progress dies, which
        # on a persistent server it eventually will. A zero with no context
        # is indistinguishable from a broken path, and this project has now
        # lost time twice to exactly that (WM loss 0.0000; New facts/ep 0.0).
        # So print the reason next to the number: probes counts whether the
        # path RAN, boredom explains why it paid what it paid.
        _ic = getattr(self, "imagination_curiosity", None)
        if _ic is not None and getattr(_ic, "enabled", False):
            _is = _ic.stats
            print(f"  Imagination:      bonus={_is.get('bonus', 0.0):.4f}, "
                  f"boredom={_is.get('boredom', 0.0):.3f}, "
                  f"probes={_is.get('probes', 0)} "
                  f"(boredom 0 = external curiosity still teaching, "
                  f"so 0 bonus is CORRECT here)")
        if self._skip_perdim_symbolic:
            print("  SD facts/ep:      n/a (_NullSymbolicDecoder on pixels)")
        else:
            print(f"  SD facts/ep:      "
                  f"{metrics.get('symbolic_decoder_facts', 0):.1f}")
        print(f"  Knowledge graph:  {kg_stats['num_facts']} facts, "
              f"{kg_stats['num_entities']} entities, "
              f"{kg_stats['num_action_rules']} rules")
        # ---- BODY + BEHAVIOUR (2026-07-27, rung 0) --------------------
        # gui_frac is the number that made the stall diagnosable only by
        # frame-classifying 20 videos after the fact. Printed every segment
        # now, so paralysis is visible in seconds instead of a forensic pass.
        try:
            _w0 = (getattr(self, "_last_world_info", None) or {})
            _pp = (getattr(self, "_proprio_per_env", None) or [None])[0]
            _bits = []
            if _w0.get("gui_frac") is not None:
                _bits.append(f"gui={_w0['gui_frac']:.0%}"
                             f"(run{int(_w0.get('gui_run_max', 0))})")
            if _pp is not None and len(_pp) >= 8:
                _bits.append(f"food={_pp[0]:.2f} hp={_pp[2]:.2f} "
                             f"depth={_pp[3]:.2f} tool={int(_pp[4])} "
                             f"carry={_pp[7]:.2f}")
            if _w0.get("mainhand"):
                _bits.append(f"hand={_w0['mainhand']}")
            if _bits:
                print("  Body/behaviour:   " + " | ".join(_bits))
            _gi = getattr(self, "_gui_intr", None)
            if _gi and _gi[1] and _gi[3]:
                _in_gui, _in_world = _gi[0] / _gi[1], _gi[2] / _gi[3]
                print(f"  Curiosity split:  in-menu={_in_gui:+.4f}/step "
                      f"vs world={_in_world:+.4f}/step "
                      f"(ratio {(_in_gui / _in_world if abs(_in_world) > 1e-9 else float('nan')):+.2f}) "
                      f"— RAW surprise, pre-suppression: >1 means the menu would still out-pay the world if we paid for it")
        except Exception:
            pass
        print(f"  Skills learned:   {skill_stats['total_skills']} "
              f"({skill_stats['mastered_skills']} mastered)")
        # MASTERY LEDGER + NESTED COMPOSITION (arch v3): asked/success per
        # skill and the earned invocation graph. One line each, greppable.
        try:
            _ml = [(s.skill_id, sum(s.asked_log or []), len(s.asked_log or []),
                    s.wm_fidelity)
                   for s in self.skill_bank.skills.values()
                   if (s.asked_total or 0) > 0]
            if _ml:
                _ml.sort(key=lambda x: -x[2])
                _msg = ", ".join(
                    f"{sid.split('_', 2)[-1]}={hit}/{n}"
                    + (f" wm={fid:.2f}" if fid is not None else "")
                    for sid, hit, n, fid in _ml[:6])
                print(f"  Mastery ledger:   {_msg}")
            if self.option_executor is not None:
                _ns = self.option_executor
                if _ns.nested_pushes or _ns.max_skill_depth > 1:
                    print(f"  Nested options:   pushes={_ns.nested_pushes}, "
                          f"cycle_blocks={_ns.nested_cycle_blocks}, "
                          f"depth_blocks={_ns.nested_depth_blocks}, "
                          f"max_depth={_ns.max_skill_depth}")
        except Exception:
            pass    # telemetry must never kill the run
        print(f"  Exploration ratio: {curiosity_stats['exploration_ratio']:.3f}")
        print(f"  Reward weights:   intrinsic={self.reward_mixer.weights['intrinsic']:.3f}, "
              f"extrinsic={self.reward_mixer.weights['extrinsic']:.3f}")
        # `inf` here is NOT a measurement — DevelopmentalStageController._level
        # returns float("inf") when error_history is EMPTY, i.e. the controller
        # has never been given a reconstruction error at all. Printed as a
        # number it read like a very large error; in the live log
        # `WM-error=inf` was the ONLY value ever recorded across 158k steps,
        # and it silently meant "blind", not "bad". While it is inf the
        # controller can never leave `explore`, so IMAGINE is unreachable and
        # the dream never runs. Say that in words, and say for how long — a
        # system that cannot distinguish "no data" from "a number" has now
        # cost three separate investigations (WM loss 0.0000; New facts/ep
        # 0.0; this).
        _lvl = self.stage_controller._level()
        _n_err = len(getattr(self.stage_controller, "error_history", ()) or ())
        if _n_err == 0 or not np.isfinite(_lvl):
            self._stage_blind_segments = getattr(
                self, "_stage_blind_segments", 0) + 1
            print(f"  Dev stage:        {self.stage_controller.stage} "
                  f"(WM-error=n/a — NO world-model samples for "
                  f"{self._stage_blind_segments} segments; the stage cannot "
                  f"advance and the dream cannot activate until "
                  f"_train_world_model returns metrics)")
        else:
            self._stage_blind_segments = 0
            print(f"  Dev stage:        {self.stage_controller.stage} "
                  f"(WM-error={_lvl:.4f}, n={_n_err}, "
                  f"slope={self.stage_controller._slope():.5f})")
        if self._lifelong and self._ll is not None:
            # ||h||: THE stability signal for the continuous stream. If it
            # settles into a band (not monotonic growth over hours) the
            # never-resetting RSSM is stable — the interim run's core question.
            # h_evolve = mean cosine distance between successive segments' carried
            # h: ~0 => the cross-segment carry is frozen (the bug), >0 => the
            # continuous state genuinely evolves. ||h|| is a magnitude band only
            # (gru_norm LayerNorm pins it near sqrt(H), BLIND to evolution).
            print(f"  Lifelong stream:  h_evolve={self._ll.h_liveness():.4f}, "
                  f"||h||={self._ll.h_norm():.3f}, "
                  f"decay={self._ll.state_decay}, "
                  f"total_steps={self.total_timesteps}")
        if getattr(self, "_goal_ltm", None) is not None:
            ls = self._goal_ltm.stats()
            pe = getattr(self, "_page_events_total", {"out": 0, "in": 0})
            print(f"  Working set:      {ls['active']} active / "
                  f"{ls['dormant']} dormant goals (total known {ls['total']}), "
                  f"paged out={pe['out']}, recalled={pe['in']}")
        if self.vision_scaffold is not None:
            vs = self.vision_scaffold.stats
            _cat = ", ".join(f"{k.split('_')[0]}={v:.3f}"
                             for k, v in vs["cat_lp"].items())
            print(f"  Curiosity magnet: w={vs['weight']:.4f}, "
                  f"target={vs['target']}, "
                  f"global_lp={vs['global_lp']:.4f} | {_cat}")
            # THE GOAL'S LEADING INDICATOR (2026-09-01). Read `labels` first:
            # below symbolic_grounding.min_labels the head is untrained and
            # the percentage is noise, not a sighting rate. DEGENERATE means
            # the sensor has collapsed and the q4 VLM is the first thing to
            # revert. A healthy trunk-seeking run shows this RISING while
            # logs are still 0 — that is the state in which the chop fixes
            # (repeat 4, macro-12 at 40 ticks, the RSSM policy) are actually
            # under test rather than untested.
            _tn = int(getattr(self, "_tree_n", 0) or 0)
            if _tn:
                print(f"  Tree in view: "
                      f"{100.0 * getattr(self, '_tree_hi', 0) / _tn:.1f}% of "
                      f"steps (mean p="
                      f"{getattr(self, '_tree_sum', 0.0) / _tn:.3f}, "
                      f"reliability={getattr(self, '_tree_rel', 0.0):.2f}, "
                      f"labels={getattr(self, '_tree_lab', 0)}"
                      + (", DEGENERATE — excluded from steering"
                         if getattr(self, "_tree_degen", False) else "")
                      + ")")
                self._tree_sum, self._tree_n, self._tree_hi = 0.0, 0, 0
            # ANTICIPATION (2026-09-02): mean/step is the intrinsic drive it
            # actually contributed; payouts is how often the rise-then-reset
            # cycle actually completed this segment (0 for a long stretch
            # means either nothing is associated yet or nothing is being
            # achieved — the lifetime `self.anticipation.total_payouts` in
            # the saved state file distinguishes cold-start from stalled).
            _an_n = int(getattr(self, "_anticip_n", 0) or 0)
            if _an_n:
                _an_mean = getattr(self, "_anticip_sum", 0.0) / _an_n
                print(f"  Anticipation bonus: mean "
                      f"{_an_mean:+.5f}/step | "
                      f"{getattr(self, '_anticip_payouts', 0)} payouts this "
                      f"segment | lifetime "
                      f"{getattr(self.anticipation, 'total_payouts', 0)} "
                      f"payouts, {getattr(self.anticipation, 'total_paid', 0.0):.2f} total")
                self._anticip_sum = 0.0
                self._anticip_n = 0
                self._anticip_payouts = 0
            # OBSERVABILITY (2026-07-25): w=0 has THREE distinct causes and the
            # line above cannot tell them apart — the 19h zero-reward stall was
            # the cold-start BUDGET latching, which was invisible here. Print
            # the budget/seek internals the scaffold already tracks.
            _cs = vs.get("cold_spent") or {}
            _sk = vs.get("seek") or {}
            if _cs or _sk:
                _csx = ", ".join(f"{k.split('_')[0]}={v}"
                                 for k, v in sorted(_cs.items()))
                print(f"    magnet internals: cold_spent[{_csx}] "
                      f"budget={vs.get('cold_start_budget', '?')} "
                      f"seek_nudge_left={_sk.get('nudge_left', '?')} "
                      f"seek_cats={_sk.get('cats', '?')}")
            # FOVEA (2026-08-06): P(goal under the crosshair) per seek cat
            # (-1.0 = not yet grounded / channel off) + the gaze-righting
            # levelness. These are the two new gradients — if fovea stays -1
            # the crop labels are not landing; if pitch_level dwells at 0.65
            # the agent is still sky-clamped.
            # SOCIAL STATE (2026-08-09): teacher certification progress +
            # demonstrations — invisible before, which is why "I stood in
            # front of it and nothing happened" could not be diagnosed from
            # the log.
            _soc = vs.get("social") or {}
            _pos_f = (getattr(self.symbolizer, "pos_counts", {})
                      if self.symbolizer is not None else {}) or {}
            _pos_v = (getattr(self.symbolizer, "fovea_pos_counts", {})
                      if self.symbolizer is not None else {}) or {}
            print(f"    social: player_pos_labels full={_pos_f.get('player_visible', 0)} "
                  f"fovea={_pos_v.get('player_visible', 0)} | "
                  f"demos={getattr(self, '_demos_seen', 0)} "
                  f"primed={_soc.get('primed', [])} | coverage suppressed "
                  f"(excavated) {getattr(self, '_cov_suppressed', 0)} steps")
            _fv = vs.get("fovea") or {}
            _sym_fv = (self.symbolizer.stats.get("fovea")
                       if self.symbolizer is not None else None) or {}
            if _fv or _sym_fv:
                _fvx = ", ".join(f"{k.split('_')[0]}={v:.3f}"
                                 for k, v in _fv.items())
                print(f"    fovea: [{_fvx}] labels={_sym_fv.get('labels', 0)} "
                      f"agreement={_sym_fv.get('agreement')} "
                      f"grounded={_sym_fv.get('grounded', 0)}/"
                      f"{len(_sym_fv.get('last', {})) or 8} "
                      f"pitch_level={vs.get('pitch_level', '?')}")
        # OPTION INVOCATIONS (2026-07-25): with the raw per-step chop income
        # removed, shaping only walks the agent TO the trunk — the scripted
        # chop_trunk option is what holds attack, and the env's +5.0 log break
        # does the teaching. If that option is never sampled the chain is
        # broken and the agent approaches forever without chopping. This is
        # the number that distinguishes those two cases; nothing logged it.
        if getattr(self, "option_executor", None) is not None:
            _bank = self.option_executor.bank
            _inv = getattr(_bank, "invoked", {}) or {}
            _tot = sum(_inv.values())
            _named = []
            for _slot, _n in sorted(_inv.items(), key=lambda kv: -kv[1])[:4]:
                _b = _bank.slots[_slot] if _slot < len(_bank.slots) else None
                _nm = (_b or {}).get("name") or (_b or {}).get("skill_id") or _slot
                _named.append(f"{_nm}={_n}")
            # INDIVIDUATION (infra #35): mean |delta| per practised skill —
            # 0.000 = still the base policy (a recording); growth = practice
            # has made it its own thing (a memory). The number that was
            # previously unmeasurable because skills were byte copies.
            _dl = [f"s{_s:02d}={float(_b.get('delta_mag', 0.0)):.3f}"
                   for _s, _b in enumerate(_bank.slots)
                   if _b is not None and _b.get("delta_mag")]
            if _dl:
                print(f"  Skill deltas:     {', '.join(_dl[:8])}")
            _scripted = [s for s in range(len(_bank.slots))
                         if _bank.is_scripted(s)]
            _scr = ", ".join(
                f"slot{s}:{_inv.get(s, 0)}" for s in _scripted) or "none"
            print(f"  Option invocations: total={_tot} | scripted[{_scr}] | "
                  f"top: {', '.join(_named) if _named else '(none yet)'}")
            # OFFERED vs CHOSEN — the line that distinguishes "the gate never
            # offered it" from "it was offered and the policy never picked it".
            # Those are opposite bugs with opposite fixes, and without this
            # they look IDENTICAL from the logs (total= simply stops rising).
            try:
                # NOTE: the method is snapshot(), NOT stats(). A first cut
                # called stats() — which does not exist — and the broad
                # `except: pass` below swallowed the AttributeError, so the
                # line silently never printed while everything looked fine.
                # That is why the failure is now REPORTED rather than passed.
                _ovc = (self.option_executor.snapshot() or {}).get(
                    "offered_vs_chosen") or {}
                _d = int(_ovc.get("decisions", 0))
                _wo = int(_ovc.get("with_offer", 0))
                _pk = int(_ovc.get("picked", 0))
                _wl = int(_ovc.get("with_learned", 0))
                _ls = int(_ovc.get("learned_offered_sum", 0))
                _pr = int(_ovc.get("probation", 0))
                # Scout option-steps whose reward could not be attributed to
                # any storable decision (2026-09-01). 0 is the proof that
                # scout SMDP rows are being kept; anything climbing means a
                # runtime is accumulating a return that _close() discards,
                # and that is the next thing to look at.
                _srd = int(_ovc.get("scout_rows_dropped", 0))
                if _d:
                    print(f"  Option offers: {_wo}/{_d} decisions had a slot "
                          f"offered ({100.0*_wo/_d:.1f}%), picked {_pk} "
                          f"({100.0*_pk/max(1,_wo):.2f}% of offered)"
                          + (f"  <-- {_srd} scout option-steps DROPPED "
                             f"(reward accumulated, no decision to store it "
                             f"against)" if _srd else "")
                          + ("  <-- OFFERED BUT NEVER CHOSEN: meta-policy "
                             "collapse, not a gate problem"
                             if _wo > 200 and _pk == 0 else ""))
                    # LEARNED-only view: the ratio above includes the scripted
                    # slot (gate-exempt) and so reads ~100% even when every
                    # learned skill is gated. This line is what shows the
                    # competence gate actually closing, and probation
                    # re-opening it.
                    print(f"  Gate state: learned offered on {_wl}/{_d} "
                          f"({100.0*_wl/_d:.1f}%), mean {_ls/max(1,_d):.2f} "
                          f"slots/decision, probation re-offers {_pr}, "
                          f"fovea-vetoes "
                          f"{getattr(self.option_executor, 'contact_vetoes', 0)}")
                else:
                    print("  Option offers: (no open decisions recorded yet)")
            except Exception as _e:
                # Never take down a multi-day run — but NEVER fail silently
                # either: a telemetry line that vanishes without a trace is
                # indistinguishable from a metric that is legitimately zero.
                print(f"  Option offers: UNAVAILABLE ({type(_e).__name__}: "
                      f"{_e}) — telemetry broken, not a zero measurement")
        # CHOP-COMPLETION MEASUREMENT (2026-07-25): every motivational
        # subsystem verified healthy while extrinsic reward stayed ~0, so the
        # remaining suspect is the chop TIMING OUT. A chop holds attack for
        # max_option_steps x action_repeat ticks; a log needs ~8 with an axe
        # but ~60 barehanded. Print what is actually in hand and the budget,
        # so one segment settles it instead of another hypothesis.
        try:
            _w0 = (getattr(self, "_last_world_info", None) or {})
            _mh = _w0.get("mainhand", "?")
            # config path is skills_as_options.* (NOT options.*) — reading
            # the wrong key made this diagnostic report a stale default 25
            # while the real budget was 40, i.e. the measurement would have
            # lied about the very thing it exists to measure.
            _mos = int(self.config.get("skills_as_options", {}).get(
                "max_option_steps", 25))
            _ar = int(self.config.get("environment", {}).get(
                "action_repeat", 1))
            print(f"  Chop budget: mainhand={_mh} | {_mos} option-steps x "
                  f"action_repeat {_ar} = {_mos * _ar} ticks "
                  f"(log needs ~8 with an axe, ~60 barehanded)")
            # THE MEASURED QUANTITY (2026-07-26). Two days were lost inferring
            # break time from proxies — a dead `mainhand` sensor, then option
            # durations — and both inferences were wrong. This prints what
            # actually happened: attack ticks that produced each real break,
            # and the longest unbroken attack streak. A large streak with no
            # breaks means the swing is NOT LANDING (aim), not that it is too
            # short (timing). Never infer this again; read it here.
            _wi = getattr(self, "_last_env_info", None) or {}
            _bt = _wi.get("break_ticks") or []
            _ar_now = _wi.get("attack_run")
            _ar_max = _wi.get("attack_run_max")
            if _bt or _ar_max is not None:
                _s = ", ".join(f"{b}:{t}t" for b, t in _bt[-5:]) or "NO BREAKS YET"
                print(f"  Ticks-to-break: {_s} | attack streak now="
                      f"{_ar_now} max={_ar_max}")
            # ---- CHOP DIAGNOSIS: aim vs timing vs target selection --------
            # The old one-line hint ("long streaks, no breaks => AIM") could
            # only fire when the agent had broken NOTHING AT ALL, so it was
            # silent in every real run and the question was settled by
            # argument instead of data. These three statistics separate the
            # modes outright, and the verdict is stated rather than implied.
            _nb = _wi.get("runs_nobreak") or []
            # KEY-TYPE GUARD: breaks_by_type is keyed by BLOCK NAME. A
            # display bug printed INTEGER keys here — they were action
            # indices — so "blocks broken: 2 total | top: [(3, 570)]"
            # read as though 570 of block-type 3 had been broken.
            # Accept only string keys, so a mis-wired dict shows as
            # empty rather than as confident nonsense.
            _bbt = {k: v for k, v in
                    (_wi.get("breaks_by_type") or {}).items()
                    if isinstance(k, str)}
            if _nb or _bbt:
                _need = 8            # ticks for a log WITH the held iron axe
                _long = [r for r in _nb if r >= _need]
                _tot_breaks = sum(_bbt.values())
                _logs = sum(v for k, v in _bbt.items() if "log" in k)
                _top = sorted(_bbt.items(), key=lambda kv: -kv[1])[:4]
                _med = (sorted(_nb)[len(_nb) // 2] if _nb else 0)
                if _tot_breaks == 0 and _long:
                    _verdict = "AIM — long swings connect with nothing"
                elif _nb and not _long and _tot_breaks == 0:
                    _verdict = "TIMING — every swing ends short of a break"
                elif _tot_breaks and _logs == 0:
                    _verdict = ("TARGET SELECTION — it breaks things, just "
                                "not logs")
                elif _logs:
                    _verdict = f"CHOPPING — {_logs} log(s) felled"
                else:
                    _verdict = "insufficient data"
                # WHERE IT IS LOOKING + WHAT IT IS CHOOSING. Both were
                # inferred from consequences until now, which is how "stuck
                # looking up" and "cannot look down" stayed unanswerable.
                _pitch = _w0.get("pitch")
                _ah = _wi.get("action_hist") or {}
                if _ah:
                    _tot = max(1, sum(_ah.values()))
                    _names = {0: "noop", 1: "walk", 2: "jump", 3: "turnL",
                              4: "turnR", 5: "attack", 6: "atk+walk",
                              7: "lookUP", 8: "lookDOWN", 9: "back",
                              10: "INVENTORY", 11: "use", 12: "HOLD"}
                    # NOT `_top`: that name already holds the blocks-broken
                    # ranking computed above, and reusing it here silently
                    # overwrote it — which is why the "blocks broken" line
                    # kept printing ACTION indices long after its key-type
                    # guard was correct. The guard was fine; the variable was
                    # being clobbered between computing and printing it.
                    # FULL, not top-5 (fix 2026-08-06). The table is 13 wide
                    # and a top-5 view could not answer the simplest question
                    # about the newest action: "is HOLD ever chosen?" It never
                    # placed in the top 5, so its share was unknowable from
                    # the log — the exact blindness that makes a capability
                    # look broken when it may simply be unpreferred.
                    _atop = sorted(_ah.items(), key=lambda kv: -kv[1])
                    _s = " ".join(
                        f"{_names.get(int(k), k)}:{100.0*v/_tot:.0f}%"
                        for k, v in _atop)
                    _up = 100.0 * _ah.get(7, 0) / _tot
                    _dn = 100.0 * _ah.get(8, 0) / _tot
                    # WHERE IT IS STANDING (2026-08-02). 67,584 steps were
                    # spent with a frozen inventory, a 10,068-tick attack
                    # streak and ZERO blocks broken, and the log could not say
                    # whether the agent was somewhere it is ALLOWED to break
                    # blocks — server spawn-protection and adventure mode both
                    # look exactly like "the swing missed" from in here.
                    # Coordinates make that answerable in one glance instead of
                    # another week of inferring aim from consequences.
                    _px = _w0.get("xpos")
                    _py = _w0.get("ypos")
                    _pz = _w0.get("zpos")
                    if _px is not None and _pz is not None:
                        print(f"  Position: x={_px:+.1f} y="
                              f"{'?' if _py is None else f'{_py:.1f}'} "
                              f"z={_pz:+.1f}  cell={_w0.get('cell')}"
                              # ONLY flag when breaking is ALSO failing.
                              # Printed on position alone it accused the server
                              # of spawn-protection while the agent was happily
                              # breaking blocks 8 blocks from origin — a false
                              # lead left in the log forever. MEASURED
                              # 2026-08-02: the real cause of the 33-segment
                              # "0 breaks" was POSITION — the agent was perched
                              # on a tree canopy, pitch clamped down, with
                              # nothing within reach. A rejoin put it back on
                              # the ground and it broke 16 blocks in one
                              # segment at these very coordinates.
                              + ("   <-- near origin AND nothing broken: "
                                 "check spawn-protection / land claims"
                                 if (abs(float(_px)) <= 16
                                     and abs(float(_pz)) <= 16
                                     and not (_wi.get("breaks_by_type") or {}))
                                 else ""))
                    print(f"  Looking: pitch="
                          f"{'n/a' if _pitch is None else f'{_pitch:+.0f}'}"
                          f"  (-90=up +90=down)"
                          + ("  <-- PINNED" if _pitch is not None
                             and abs(abs(_pitch) - 90.0) < 2.0 else ""))
                    print(f"  Actions: {_s} | lookUP {_up:.0f}% vs "
                          f"lookDOWN {_dn:.0f}%")
                _cn = int(getattr(self, "_cov_n", 0) or 0)
                if _cn:
                    _cavg = float(getattr(self, "_cov_sum", 0.0)) / _cn
                    _icm = float(np.mean(
                        list(self.training_metrics["intrinsic_reward"])[-1:])
                        if self.training_metrics["intrinsic_reward"] else 0.0)
                    print(f"  Coverage: {_cavg:+.4f}/step from territory "
                          f"(weight {self._coverage_weight}) | cells seen "
                          f"{getattr(self, '_cov_cells', 0)} | total intrinsic "
                          f"{_icm:+.4f}/step -> territory is "
                          f"{100.0 * _cavg / _icm if _icm > 1e-9 else 0.0:.0f}% "
                          f"of the drive")
                    self._cov_sum, self._cov_n = 0.0, 0
                _ent = (list(self.training_metrics["policy_entropy"])[-1]
                        if self.training_metrics.get("policy_entropy")
                        else None)
                _mt = float(getattr(self, "_mag_turn", 0.0))
                _mtn = int(getattr(self, "_mag_turn_n", 0))
                _mo = float(getattr(self, "_mag_other", 0.0))
                _mon = int(getattr(self, "_mag_other_n", 0))
                if _ent is not None or _mtn or _mon:
                    _emax = float(np.log(max(2, self.action_dim)))
                    print(
                        "  Policy: entropy "
                        + ("n/a" if _ent is None else
                           f"{_ent:.2f}/{_emax:.2f} nats "
                           f"({100.0 * _ent / _emax:.0f}% of max)")
                        + ("  <-- COLLAPSED" if _ent is not None
                           and _ent < 0.5 * _emax else ""))
                    _pp = getattr(self, "_last_ppo", None)
                    if _pp:
                        _ras = _pp.get("raw_adv_std", float("nan"))
                        _osp = _pp.get("obs_spread", float("nan"))
                        _mxp = _pp.get("max_prob", float("nan"))
                        _kl = _pp.get("approx_kl")
                        _er = _pp.get("epochs_run")
                        if _kl is not None:
                            # TWO CAUSES, OPPOSITE MEANINGS (2026-08-23).
                            # This line used to call ANY reduction "good:
                            # the policy hit its movement budget" — that
                            # describes target_kl early-stopping, which is
                            # OFF (0.0, withdrawn 2026-08-05). So every
                            # reduction reported here was really the
                            # tiny-batch guard REMOVING updates, and a
                            # starved policy read as a healthy one.
                            _req = int(_pp.get("epochs_requested", 0) or
                                       self.config.get('policy', {}).get(
                                           'n_epochs', 10))
                            _capped = bool(_pp.get("epochs_capped", False))
                            # THREE CAUSES NOW, NOT TWO (2026-09-01). With
                            # target_kl back on (and minibatched, so it stops
                            # per-minibatch rather than after a whole
                            # full-batch pass) "stopped early" is reachable
                            # again — which is exactly the condition
                            # _throughput_wave_smoke::B1e said must be
                            # re-checked rather than assumed. The update now
                            # exports which one it was, so the log states it
                            # instead of inferring it:
                            #   epochs_capped -> the tiny-batch guard removed
                            #     updates (STARVED; full-batch path only)
                            #   kl_stopped    -> the actor spent its movement
                            #     budget and froze while the critic finished
                            #   n_updates >= max_updates -> the rollout's
                            #     gradient-step budget ran out (healthy)
                            _klstop = bool(_pp.get("kl_stopped", False))
                            _mxu = int(_pp.get("max_updates", 0) or 0)
                            _nup = int(_pp.get("n_updates", 0) or 0)
                            _budget = bool(_mxu and _nup >= _mxu)
                            print(f"  PPO trust region: approx_kl={_kl:.4f} | "
                                  f"epochs run {_er}/{_req} | "
                                  f"updates {_nup}"
                                  + (f"/{_mxu}" if _mxu else "")
                                  + ("  <-- STARVED: too few rows for more "
                                     "(collapse guard removed updates)"
                                     if _capped else
                                     "  <-- KL-STOPPED: actor froze at its "
                                     "movement budget, critic kept fitting"
                                     if _klstop else
                                     "  <-- budget: the rollout's gradient-"
                                     "step allowance was spent (healthy)"
                                     if _budget else ""))
                            if _capped:
                                print(f"  PPO update rate: "
                                      f"{_pp.get('env_steps', 0)} env steps -> "
                                      f"{_pp.get('rows', 0)} rows "
                                      f"({_pp.get('rows_option', 0)} option / "
                                      f"{_pp.get('rows_primitive', 0)} "
                                      f"primitive, mean tau "
                                      f"{_pp.get('tau_mean', 1.0):.1f}) -> "
                                      f"{_er} gradient pass(es). Options "
                                      f"consume rows: one option row spans "
                                      f"tau env steps.")
                        # THE ALARMS THIS WAVE ADDED AND NEVER PRINTED
                        # (2026-09-01). adv_clipped_frac / value_clamped_frac
                        # / nonfinite_skip / last_update_forced were all
                        # computed and surfaced nowhere — the "correct
                        # mechanism nobody calls" failure this file documents
                        # three times over (the gui_open guard, _viewer_push,
                        # the felt-reach sense). The two genuine alarms print
                        # ONLY when non-zero, so a healthy segment reads as
                        # before plus one adv_clip= field:
                        #   V-CLAMPED   -> the symlog critic is diverging and
                        #                  symexp is being bounded; the run
                        #                  is on its way to inf without it
                        #   UPDATE SKIPPED -> a whole rollout was dropped for
                        #                  non-finite advantages
                        _vcf = float(_pp.get("value_clamped_frac", 0.0) or 0.0)
                        print(f"  PPO forensics: raw_adv_std={_ras:.2e} "
                              f"(|adv|={_pp.get('raw_adv_absmean', 0.0):.2e}) "
                              f"| obs_spread={_osp:.4f} | max_prob={_mxp:.3f} "
                              f"| rows={_pp.get('rows', 0)} "
                              f"updates={_pp.get('n_updates', 0)}"
                              f" | adv_clip="
                              f"{_pp.get('adv_clipped_frac', 0.0):.1%}"
                              + (f" | V-CLAMPED {_vcf:.2%} <-- critic "
                                 f"diverging" if _vcf else "")
                              + (" | UPDATE SKIPPED (non-finite advantages)"
                                 if _pp.get("nonfinite_skip") else "")
                              + (" | forced (row/env-step ceiling)"
                                 if getattr(self.policy,
                                            "last_update_forced", False)
                                 else ""))
                        # Say which cause the numbers actually support, rather
                        # than leaving three hypotheses to argue about.
                        if _ras == _ras and _ras < 1e-4:
                            print("      <-- NOISE AMPLIFICATION: the raw "
                                  "advantage spread is ~0, so normalising it "
                                  "to unit variance feeds PPO pure noise at "
                                  "full gradient scale every update")
                        if _osp == _osp and _osp < 1e-3:
                            print("      <-- STATE COLLAPSE: the rollout is "
                                  "essentially ONE observation, so a "
                                  "deterministic policy is genuinely optimal "
                                  "for it — entropy 0 is correct here, and "
                                  "self-reinforcing")
                        if _mxp == _mxp and _mxp > 0.99:
                            print("      <-- SATURATED: softmax is one-hot; "
                                  "no bounded entropy bonus can pull the "
                                  "logits back from here")
                    # HAS THE MAGNET EVER LEFT ITS COLD-START FLOOR?
                    # Measured over this whole run: 144 of 144 segments at
                    # w=0.3500 with cold_spent[tree]=144827. It never once
                    # entered its curiosity-ranked branch, and finding that
                    # required grepping cold_spent by hand. Say it out loud.
                    try:
                        _vs2 = self.vision_scaffold.stats
                        _wnow = float(_vs2.get("weight", 0.0) or 0.0)
                        _cs = float(getattr(
                            self.vision_scaffold, "cold_start_weight", 0.0))
                        if abs(_wnow - _cs) < 1e-6:
                            self._magnet_cold = getattr(
                                self, "_magnet_cold", 0) + 1
                        else:
                            self._magnet_cold = 0
                        if self._magnet_cold >= 10:
                            print(f"      <-- MAGNET INERT: pinned at its "
                                  f"cold-start floor for {self._magnet_cold} "
                                  f"segments. Its contrastive score is 0, so "
                                  f"it has never ranked anything by curiosity "
                                  f"— it is a constant pull, not guidance.")
                    except Exception:
                        pass
                    print(f"  Magnet pay: turning {_mt/max(1,_mtn):+.4f}/step "
                          f"vs other {_mo/max(1,_mon):+.4f}/step"
                          + ("   <-- ROTATION IS BEING PAID"
                             if _mtn and _mon
                             and _mt/max(1,_mtn) > _mo/max(1,_mon) + 1e-6
                             else ""))
                    self._mag_turn = self._mag_other = 0.0
                    self._mag_turn_n = self._mag_other_n = 0
                _rvn = int(getattr(self, "_reachval_n", 0) or 0)
                if _rvn:
                    _rvm = getattr(self, "_reachval_sum", 0.0) / _rvn
                    _rvh = int(getattr(self, "_reachval_hi", 0) or 0)
                    print(f"  Reach (segment): mean {_rvm:.3f} | in-reach on "
                          f"{100.0 * _rvh / _rvn:.1f}% of steps "
                          f"[{getattr(self, '_reach_gate', 'n/a')}]")
                    self._reachval_sum, self._reachval_n = 0.0, 0
                    self._reachval_hi = 0
                _pn = int(getattr(self, "_pitch_n", 0) or 0)
                if _pn:
                    _pm = getattr(self, "_pitch_sum", 0.0) / _pn
                    _pc = int(getattr(self, "_pitch_clamped", 0) or 0)
                    print(f"  Pitch (segment): mean {_pm:+.1f}deg | at the "
                          f"+-90 clamp on {100.0 * _pc / _pn:.1f}% of steps "
                          f"(-90=up +90=down)")
                    self._pitch_sum, self._pitch_n = 0.0, 0
                    self._pitch_clamped = 0
                _ewn = int(getattr(self, "_env_wait_n", 0) or 0)
                if _ewn:
                    _now_w = time.time()
                    _prev_w = getattr(self, "_seg_wall_prev", None)
                    self._seg_wall_prev = _now_w
                    _ew_ms = 1000.0 * getattr(self, "_env_wait_sum",
                                              0.0) / max(1, _ewn)
                    if _prev_w:
                        _step_ms = 1000.0 * (_now_w - _prev_w) / max(1, _ewn)
                        # env% is the share of each step spent waiting on the
                        # Minecraft client. High => the world is the limit
                        # (ceiling: 1000/20*action_repeat ms). Low => the
                        # limit is our own per-step Python, and no amount of
                        # faster env plumbing would help.
                        print(f"  Loop timing: {_step_ms:.0f} ms/step total | "
                              f"env round-trip {_ew_ms:.0f} ms "
                              f"({100.0*_ew_ms/max(1e-6,_step_ms):.0f}%) | "
                              f"own {_step_ms-_ew_ms:.0f} ms | "
                              f"{1000.0/max(1e-6,_step_ms):.2f} steps/s")
                        # WHERE THE OWN-TIME GOES (2026-09-01). The line
                        # above has said "own = 2/3 of every step" since
                        # 2026-08-17 without ever naming a phase, which is
                        # exactly the kind of half-measurement §5 warns
                        # about. UNACCOUNTED is printed on purpose: if the
                        # phases do not sum to the step time, the gap is
                        # real work nobody is timing, and pretending the
                        # buckets are exhaustive would hide it.
                        _acc = getattr(self, "_phase_acc", None) or {}
                        if _acc:
                            _parts, _sum = [], 0.0
                            for _ph in self._PHASES:
                                _v = 1000.0 * _acc.get(_ph, 0.0) / max(1, _ewn)
                                _sum += _v
                                _parts.append(
                                    f"{_ph} {_v:.0f}ms "
                                    f"({100.0*_v/max(1e-6,_step_ms):.0f}%)")
                            _un = _step_ms - _sum
                            _parts.append(
                                f"UNACCOUNTED {_un:.0f}ms "
                                f"({100.0*_un/max(1e-6,_step_ms):.0f}%)")
                            print("  Phase timing: " + " | ".join(_parts))
                            # WHAT IS INSIDE `env` (2026-09-21). `env` is the
                            # largest phase left and was one opaque number.
                            # These are IN-THREAD means per client-step, so
                            # they do NOT sum to the `env` wall time — the
                            # comparison that matters is `slowest` (what env
                            # would cost with a perfect pool) against the
                            # measured `env` above. A large gap is pool/GIL
                            # overhead and means extra clients are being
                            # serialised rather than overlapped.
                            _ep = getattr(self, "_env_parts", None) or {}
                            _en = float(_ep.get("_n", 0.0))
                            if _en:
                                _eparts = [
                                    f"{_k} {_ep.get(_k, 0.0)/_en:.0f}ms"
                                    for _k in self._ENV_PARTS]
                                _slow = _ep.get("_slow", 0.0) / _en
                                _envw = 1000.0 * _acc.get("env", 0.0) / max(
                                    1, _ewn)
                                print("  Env breakdown (in-thread): "
                                      + " | ".join(_eparts)
                                      + f" | slowest client {_slow:.0f}ms"
                                      + f" vs env phase {_envw:.0f}ms"
                                      + f" -> pool overhead "
                                        f"{_envw - _slow:+.0f}ms")
                    self._env_parts = {}
                    self._phase_acc = {}
                    self._env_wait_sum = 0.0
                    self._env_wait_n = 0
                # A2 self-check readout: the largest |carried - recomputed|
                # feature delta this segment. ~0 = no async WM update landed
                # between encode and use; small but nonzero = one optimizer
                # step of drift (expected). Anything large would have RAISED
                # rather than printed, so this line is a proof of correctness
                # accumulating, not a warning.
                if getattr(self, "_verify_enc_feats", False):
                    _fd = float(getattr(self.policy, "last_feat_drift", 0.0))
                    print(f"  Feat drift (max): {_fd:.2e} — carried encoder "
                          f"features vs recomputed (self-check ON; set "
                          f"policy.verify_encoder_feats false to collect the "
                          f"speedup once this stays ~0)")
                    self.policy.last_feat_drift = 0.0
                print(f"  Reach sense (last step): {float(getattr(self, '_reach_now', 0.0)):.2f} "
                      f"[{getattr(self, '_reach_gate', 'n/a')}] raw="
                      f"{float(getattr(self, '_reach_raw', 0.0)):.2f} "
                      f"(P breakable_in_reach, 0=nothing touchable)")
                _pn = int(getattr(self, "_persist_n", 0) or 0)
                _rn2 = int(getattr(self, "_reach_n", 0) or 0)
                if _pn or _rn2:
                    print(f"  Shaping: persistence "
                          f"{getattr(self, '_persist_sum', 0.0)/max(1,_pn):+.4f}"
                          f"/step | approach-to-reach "
                          f"{getattr(self, '_reach_sum', 0.0)/max(1,_rn2):+.4f}"
                          f"/step  (both telescoping -> net ~0 per cycle)")
                    self._persist_sum = self._reach_sum = 0.0
                    self._persist_n = self._reach_n = 0
                _gln = int(getattr(self, "_pitch_level_n", 0) or 0)
                if _gln:
                    print(f"  Gaze level: "
                          f"{getattr(self, '_pitch_level_sum', 0.0)/max(1,_gln):+.5f}"
                          f"/step (telescoping; Phi now "
                          f"{float(getattr(self, '_pitch_level_phi', 0.0) or 0.0):+.3f}"
                          f", -1 = at the clamp)")
                    self._pitch_level_sum = 0.0
                    self._pitch_level_n = 0
                _sn = int(getattr(self, "_sym_n", 0) or 0)
                if _sn:
                    print(f"  Symbols: {getattr(self, '_sym_sum', 0.0)/max(1,_sn):+.4f}"
                          f"/step | grounded vocabulary "
                          f"{getattr(self, '_sym_known', 0)} "
                          f"| distinct scenes {len(self._symbol_counts)}")
                    _scn = int(getattr(self, "_symc_n", 0) or 0)
                    if _scn:
                        # MEAN |F|, NOT MEAN F. A telescoping potential sums to
                        # ~0 over any trajectory BY CONSTRUCTION, so printing
                        # the signed mean showed "-0.0003/step" and read as
                        # "this term is doing nothing" no matter how strongly
                        # it was steering. The magnitude is the informative
                        # number; Phi says where it currently sits.
                        print(f"  Centring: |{getattr(self, '_symc_abs', 0.0)/_scn:.4f}"
                              f"|/step magnitude, Phi now "
                              f"{float(getattr(self, '_sym_center_phi', 0.0)):.3f} "
                              f"(signed mean {getattr(self, '_symc_sum', 0.0)/_scn:+.5f}"
                              f" ~ 0 = telescoping correctly, NOT inert)")
                        self._symc_sum, self._symc_n = 0.0, 0
                        self._symc_abs = 0.0
                    # WHAT IS IT BEING PAID FOR WHILE IT STANDS STILL?
                    _cn2 = int(getattr(self, "_cen_n", 0) or 0)
                    if _cn2:
                        _b = getattr(self, "_cen_base", 0.0) / _cn2
                        _im = getattr(self, "_cen_imag", 0.0) / _cn2
                        _tot_now = float(np.mean(
                            list(self.training_metrics["intrinsic_reward"])[-1:])
                            if self.training_metrics["intrinsic_reward"] else 0.0)
                        _bvn = int(getattr(self, "_bv_n", 0) or 0)
                        print(f"  Reward census: ICM/LP base {_b:+.4f}/step "
                              f"({100.0 * abs(_b) / max(1e-9, abs(_tot_now)):.0f}% "
                              f"of the {_tot_now:+.4f} drive) | imagination "
                              f"{_im:+.4f}/step | everything itemised above is "
                              f"the remainder"
                              + (f" | boring-view factor "
                                 f"{getattr(self, '_bv_sum', 0.0)/max(1,_bvn):.2f}"
                                 f" mean (1=full pay, base already discounted)"
                                 if _bvn else ""))
                        self._bv_sum, self._bv_n = 0.0, 0
                        # WHY is the base what it is? LP is a DERIVATIVE with
                        # a significance gate, so on a mastered static view it
                        # should be ~0 — the user's point exactly: standing
                        # still ought to pay nothing. If it does not, the
                        # normalizer is the first suspect:
                        #     reward = (lp - median) / max(std, 1e-3)
                        # LP is mostly zeros, so when the pool's true spread
                        # falls under that 1e-3 FLOOR, residual jitter that
                        # squeaks past the significance gate is amplified
                        # toward O(0.1) — an income for existing.
                        _cu = self.curiosity
                        _an = int(getattr(self, "_attr_n", 0) or 0)
                        if _an and getattr(_cu, "action_conditional", False):
                            _amean = getattr(self, "_attr_sum", 0.0) / _an
                            _anz = int(getattr(self, "_attr_nz", 0) or 0)
                            print(f"      Action attribution: mean {_amean:.4f} "
                                  f"over {_an} steps | nonzero on "
                                  f"{100.0 * _anz / _an:.1f}% of them")
                            # The two ways this goes wrong look identical in a
                            # single number, so say which one the data shows.
                            if _anz == 0:
                                print("        <-- ZERO ON EVERY STEP: the "
                                      "forward model is ignoring its action "
                                      "input, so this gate suppresses ALL "
                                      "curiosity rather than selectively. "
                                      "Set curiosity.action_conditional=false")
                            elif _amean < 0.02:
                                print("        <-- very low: the agent's "
                                      "actions explain almost none of what it "
                                      "sees. Either the world is dominated by "
                                      "ambient change (intended target) or the "
                                      "forward model is barely action-aware")
                            self._attr_sum, self._attr_n = 0.0, 0
                            self._attr_nz = 0
                        _rs = getattr(_cu, "_running_std", None)
                        _rm = getattr(_cu, "_running_median", None)
                        if _rs is not None:
                            print(f"      LP internals: running_std={_rs:.2e} "
                                  f"median={_rm:.2e} raw_pred_err="
                                  f"{getattr(_cu, 'last_pred_error', 0.0):.2e}"
                                  # A low std alone is NOT a fault — on a
                                  # mastered view it is the correct reading.
                                  # It only matters if the BASE is still being
                                  # paid despite it, which is the amplification
                                  # signature. Warning on std alone cried wolf
                                  # the moment the progress floor started
                                  # working (std 3.6e-06 with the base down to
                                  # +0.0011/step is exactly right).
                                  + ("   <-- std under the 1e-3 floor AND the "
                                     "base is still paying: the normalizer is "
                                     "amplifying jitter into an income"
                                     if (_rs < 1e-3 and abs(_b) > 0.02)
                                     else ""))
                        if abs(_b) > 0.5 * abs(_tot_now):
                            print("      <-- THE BASE DOMINATES: prediction "
                                  "error alone pays this much for EXISTING, "
                                  "so standing still is already profitable "
                                  "and no shaping term can outbid it")
                        self._cen_base = self._cen_imag = 0.0
                        self._cen_n = 0
                    _gn2 = int(getattr(self, "_gaze_n", 0) or 0)
                    if _gn2:
                        _gb = sorted(self._gaze_counts.items(),
                                     key=lambda kv: -kv[1])[:3]
                        print(f"  Gaze: {getattr(self, '_gaze_sum', 0.0)/_gn2:+.4f}"
                              f"/step over {len(self._gaze_counts)} pitch "
                              f"buckets | most-dwelt "
                              + ", ".join(
                                  f"{int(b)*int(self._gaze_bucket_deg):+d}deg"
                                  f"x{c}" for b, c in _gb))
                        self._gaze_sum, self._gaze_n = 0.0, 0
                    self._sym_sum = 0.0
                    self._sym_n = 0
                _nn = int(getattr(self, "_nov_n", 0) or 0)
                if _nn:
                    print(f"  Novelty(seen): "
                          f"{getattr(self, '_nov_sum', 0.0)/max(1,_nn):+.4f}/step "
                          f"| distinct views {len(self._nov_counts)} "
                          f"(what it SEES, not distance travelled)")
                    self._nov_sum = 0.0
                    self._nov_n = 0
                _mbp = _wi.get("mine_block_present")
                if _mbp is not None:
                    print(f"  Break sensor: mine_block present={_mbp} "
                          f"entries={_wi.get('mine_block_entries')} "
                          f"nonzero={_wi.get('mine_block_nonzero')}"
                          + ("   <-- SENSOR DEAD: breaks cannot be seen"
                             if not _mbp or _wi.get('mine_block_entries', 0) == 0
                             else ""))
                print(f"  Chop diagnosis: {_verdict}")
                print(f"    swings ending with no break: {len(_nb)} "
                      f"(median {_med}t, >={_need}t: {len(_long)})")
                print(f"    blocks broken: {_tot_breaks} total, {_logs} logs"
                      + (f" | top: {_top}" if _top else ""))
        except Exception:
            pass
        if self.symbolizer is not None:
            ss = self.symbolizer.stats
            _agr = ss["agreement"]
            print(f"  Symbol grounding: "
                  f"labels={ss['labels']}, vlm_every={ss['vlm_interval']}, "
                  f"agreement={'n/a' if _agr is None else f'{_agr:.2f}'}, "
                  f"grounded={ss['grounded_predicates']}/"
                  f"{ss['n_predicates']}, facts={ss['facts']}, "
                  f"retracted={self._symbolizer_retractions}, "
                  f"parse_fail={ss['parse_failures']}, "
                  f"neg_evt={getattr(self, '_neg_evt_total', 0)}")
        if self.curriculum:
            print(f"  Difficulty:       {self.curriculum.current_difficulty}")
        glue_stats = self.glue.stats
        kv_status = "active" if glue_stats["has_knowledge_vector"] else "pending"
        print(f"  Knowledge vector: {kv_status}")
        print(f"  Goals generated:  {glue_stats['goals_generated']}")
        print(f"  Goal hierarchy:   depth={glue_stats['goal_stack_depth']}, "
              f"completed={glue_stats['completed_goals']}, "
              f"failed={glue_stats['failed_goals']}")
        print(f"  Mastery score:    {metrics.get('glue_mastery_score', 0):.3f}")
        print(f"  KG density:       {metrics.get('glue_kg_density', 0):.2f}")
        # Skill composition stats
        comp_count = self.skill_composer.compositions_created
        if comp_count > 0:
            print(f"  Compositions:     {comp_count} composite skills created")
        # LLM stats
        llm_stats = self.llm.stats
        if llm_stats["llm_available"]:
            print(f"  LLM calls:        perception={llm_stats['perception_calls']}, "
                  f"goals={llm_stats['goal_calls']}, "
                  f"reasoning={llm_stats['reasoning_calls']}")
        # Dream training stats
        if self.dream_training_active:
            print(f"  Dream policy:     actor_loss={metrics.get('dream_actor_loss', 0):.4f}, "
                  f"critic_loss={metrics.get('dream_critic_loss', 0):.4f}, "
                  f"returns={metrics.get('dream_returns_mean', 0):.2f}")
        elif getattr(self, "dream_augment_active", False):
            # ---- THE DISTILL PATH WAS INVISIBLE (2026-09-02) -------------
            # This branch used to be `elif self.dream_enabled:` printing
            # "warmup (0 episodes remaining)" FOREVER, because the only
            # other branch tests `dream_training_active` — the CONTROL flag,
            # which is permanently False at `dream_training.control: false`.
            # Meanwhile `dream_augment_active` flips True once
            # `_episodes_this_env >= dream_activate_after` (2), and that
            # counter increments once per SEGMENT in lifelong mode — so
            # distillation has been pulling the live PPO actor toward the
            # dream actor at distill_weight 1.0 every segment from segment 3
            # on, while the log said it had not started. Worse than silence:
            # it read as "not yet running" for the entire run.
            #
            # `eff_weight` is the number that answers "does this subsystem
            # matter at all": distill_weight x (1 - solve_ema) x trust_ema.
            # Near zero means the dream actor is being ignored no matter its
            # capacity — which is the reading that decides whether raising
            # dream_training.hidden_dim could ever help, and the reason that
            # capacity question is deliberately NOT being answered by
            # guesswork.
            _dw = metrics.get("dream_distill_eff_weight")
            if _dw is None:
                _dq = self.training_metrics.get("dream_distill_eff_weight")
                _dw = float(_dq[-1]) if _dq else 0.0
            _dg = metrics.get("dream_distill_gate_frac")
            if _dg is None:
                _gq = self.training_metrics.get("dream_distill_gate_frac")
                _dg = float(_gq[-1]) if _gq else 0.0
            print(f"  Dream distill:    eff_weight={float(_dw):.4f} "
                  f"(distill_weight x (1-solve_ema) x trust_ema) | "
                  f"value-gate pass {float(_dg):.1%} | "
                  f"loss={metrics.get('dream_distill_loss', 0.0):.4f}"
                  + ("   <-- INERT: the dream actor is being ignored; its "
                     "capacity cannot be the limiting factor"
                     if float(_dw) < 1e-3 else ""))
        elif self.dream_enabled:
            # Genuinely still in warmup. Gated on the AUGMENT path's own
            # counter, not dream_warmup/total_episodes — using the wrong one
            # is what let this claim "warmup" indefinitely above.
            remaining = max(0, int(self.dream_activate_after)
                            - int(getattr(self, "_episodes_this_env", 0)))
            print(f"  Dream policy:     warmup ({remaining} segments "
                  f"remaining before distillation activates)")
        # Pixel observation mode
        if self.pixel_obs:
            print(f"  Observation mode: pixel ({self.image_channels}x{self.image_size}x{self.image_size})")

        # ---- GENERAL INFRASTRUCTURE (2026-08-08) -------------------------
        # Segment-cadence evaluation of the domain-agnostic monitors, plus
        # the condition-fed gates and the stuck-escalation ladder. The stack
        # only RECOMMENDS; escalation actions are applied here, visibly.
        if self.infra is not None:
            try:
                _wi = getattr(self, "_last_world_info", None) or {}
                _cells = int(_wi.get("cells_seen", 0) or 0)
                _cells_delta = _cells - int(getattr(self, "_seg_cells_prev",
                                                    0) or 0)
                self._seg_cells_prev = _cells
                _vs_i = (self.vision_scaffold.stats
                         if self.vision_scaffold is not None else {})
                _seek_i = _vs_i.get("seek") or {}
                # condition-fed gates: every known suppressor states its
                # condition here each segment; the registry alarms if one
                # stays closed past its declared budget (infra #20)
                self.infra.gate_state(
                    "seek_nudge_budget",
                    closed=(int(_seek_i.get("nudge_left", 1) or 0) == 0),
                    step=self.total_timesteps,
                    reason="budget exhausted",
                    reopen="goal sighting or trickle regen",
                    max_closed=30000)
                # ---- GATE PREDICATE FIXED (2026-09-04) -------------------
                # This alarmed for 71,680 steps ("GATE OVERDUE:
                # 'magnet_weight' closed for 71680 steps, max 60000") while
                # magnet_seek was simultaneously paying 61% of ALL income
                # (5.97/segment, up 17x over the run). Both cannot be true.
                # CAUSE: `weight` is an INSTANTANEOUS value — vision_scaffold
                # recomputes self._w every step and it falls to 0.0 on any
                # "engaged-then-quiet" step. Sampling it once per segment and
                # calling that "closed for the whole segment" is a counter
                # that was never validated (CLAUDE.md §5: the `places`
                # counter lesson — iron_axe: 2281).
                # A false alarm is worse than no alarm: it trains the reader
                # to ignore the one channel that reports real latches, and
                # this repo has had NINE of those.
                # A gate is closed only if the magnet was inactive AND
                # actually paid nothing across the segment. The ledger share
                # may be one segment stale (it is stashed in
                # InfraStack.segment); that is immaterial against a
                # 60,000-step budget.
                _mag_paid = float((dict(getattr(
                    self.infra, "last_ledger_segment", {}) or {}).get(
                        "shares") or {}).get("magnet_seek", 0.0) or 0.0)
                self.infra.gate_state(
                    "magnet_weight",
                    closed=(float(_vs_i.get("weight", 1.0) or 0.0) == 0.0
                            and _mag_paid == 0.0),
                    step=self.total_timesteps,
                    reason="w=0 AND no magnet income this segment",
                    reopen="curiosity re-arm or cold-start candidate",
                    max_closed=60000)
                _ovc = {}
                if getattr(self, "option_executor", None) is not None:
                    _ovc = getattr(self.option_executor,
                                   "offered_vs_chosen", None)
                    _ovc = _ovc() if callable(_ovc) else {}
                    _lo = int(getattr(self.option_executor,
                                      "learned_offered_sum", 0) or 0)
                    self.infra.gate_state(
                        "learned_options",
                        closed=(_lo == getattr(self, "_hb_lo_prev", 0)),
                        step=self.total_timesteps,
                        reason="no learned option offered all segment",
                        reopen="competence/probation/contact gates",
                        max_closed=120000)
                    if _lo != getattr(self, "_hb_lo_prev", 0):
                        self.infra.beat("option_offer", self.total_timesteps)
                    self._hb_lo_prev = _lo
                _ss_i = (self.symbolizer.stats
                         if self.symbolizer is not None else {})
                _ctx = {
                    "step": self.total_timesteps,
                    # stuck metrics only when the per-step hook actually fed
                    # them this segment (see _infra_step) — an unfed monitor
                    # judging zeros produced L3 help-spam on vector envs
                    "stuck_eligible": bool(getattr(self, "_infra_fed",
                                                   False)),
                    "seg_extrinsic": float(getattr(self,
                                                   "_seg_extrinsic_sum",
                                                   0.0) or 0.0),
                    "cells_delta": float(_cells_delta),
                    "events": float(self.infra.pop_segment_events()),
                    "position": (_wi.get("x"), _wi.get("y"), _wi.get("z")),
                    "labels": _ss_i.get("labels"),
                    # Teacher evidence per predicate for the degenerate gate.
                    # POSITIVE counts, not total labels (2026-08-11): one VLM
                    # query labels every predicate, so label_counts measures
                    # elapsed queries and gates nothing. "Has this predicate
                    # ever been TRUE?" is the question that separates a broken
                    # head from an honest report about the world.
                    "signal_evidence": (dict(getattr(self.symbolizer,
                                                     "pos_counts", {}))
                                        if self.symbolizer is not None
                                        else None),
                    "intrinsic_per_step": metrics.get("intrinsic_reward"),
                    "ledger_sources": _ledger_snapshot,
                    # effects the goal system needs producible: caused-event
                    # kinds. "break" is what every current goal consumes;
                    # derived generically would read the goal ontology (#40).
                    "required_effects": ["break"],
                }
                _lines, _acts = self.infra.segment(_ctx)
                for _ln in _lines:
                    print(_ln)
                self._seg_extrinsic_sum = 0.0
                self._infra_fed = False
                # ---- stuck-escalation ladder (infra #50/#16) -------------
                _lvl = int(_acts.get("stuck_level", 0) or 0)
                if (self._stuck_boost_orig is not None
                        and self.total_timesteps
                        >= self._stuck_boost_until):
                    self._novelty_weight, self._coverage_weight = \
                        self._stuck_boost_orig
                    self._stuck_boost_orig = None
                    # clear the extension anchor too, so the NEXT boost gets a
                    # full max_total window rather than inheriting a spent one
                    self._stuck_boost_started = -1
                    print("  infra/stuck: exploration boost EXPIRED "
                          "(weights restored)")
                if _lvl >= 1 and self.vision_scaffold is not None:
                    self.vision_scaffold._seek_nudge_left = \
                        self.vision_scaffold.seek_nudge_budget
                    print("  infra/stuck: L1 remedy — search budget refilled")
                if _lvl >= 2 and self._stuck_boost_orig is None:
                    print("  infra/stuck: L2 remedy — "
                          + self._apply_explore_boost(2048))
                # ---- gaze starvation (2026-08-16): the stack judged the
                # DEGENERATE sweep to be a VIEW problem, not a sensor
                # problem. Remedy on the behaviour side: make looking around
                # pay again by capping the saturated gaze-bucket counts
                # (1/sqrt(246k) is not an incentive) and refill the seek
                # budget so the magnet can act on whatever the sweep finds.
                # Cooldown-gated so an agent cannot cycle starve->sweep for
                # income; each event re-opens a bounded, decaying purse.
                if (_acts.get("gaze_starved")
                        and self.total_timesteps
                        >= getattr(self, "_gaze_starve_cooldown", 0)):
                    self._gaze_starve_cooldown = self.total_timesteps + 8192
                    _gc = getattr(self, "_gaze_counts", None)
                    if isinstance(_gc, dict) and _gc:
                        for _b in list(_gc):
                            _gc[_b] = min(_gc[_b], 25)
                    if self.vision_scaffold is not None:
                        self.vision_scaffold._seek_nudge_left = \
                            self.vision_scaffold.seek_nudge_budget
                    print("  infra/signals: gaze-starved remedy — gaze "
                          "novelty re-opened (buckets capped at 25) + seek "
                          "budget refilled")
                # L3: the help request is emitted inside the stack (jsonl +
                # trace dump) and — new 2026-08-16 (#46) — also READ, by
                # the local VLM, which may answer with slight guidance
                # (see _advise_unstuck). Help remains a REQUEST: any
                # failure in the advisor path is a no-op and the run
                # continues regardless.
                if _lvl >= 3:
                    try:
                        self._advise_unstuck(_acts)
                    except Exception as _ae:
                        logger.debug("unstuck advisor contained: %s", _ae)
                # effective-config echo, once, at the first segment: what was
                # configured but never read (infra #26)
                if not getattr(self, "_config_echoed", False):
                    self._config_echoed = True
                    try:
                        from developmental_ai.infra.config_echo import (
                            effective_summary)
                        if hasattr(self.config, "accessed_paths"):
                            print("  infra/config: "
                                  + effective_summary(self.config))
                    except Exception:
                        pass
            except Exception as _e:
                print(f"  infra: segment hook error contained: {_e!r}")
        print(f"{'='*60}")

    def _training_summary(self, elapsed_time: float) -> Dict[str, Any]:
        """Generate a summary of the training run."""
        return {
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "elapsed_time_seconds": elapsed_time,
            "avg_episode_reward": (
                np.mean(list(self.training_metrics["episode_reward"]))
                if self.training_metrics["episode_reward"] else 0.0
            ),
            "knowledge_graph": self.knowledge_graph.get_stats(),
            "skill_bank": self.skill_bank.get_stats(),
            "curiosity": self.curiosity.stats,
            "reward_weights": self.reward_mixer.weights,
            "compositions_created": self.skill_composer.compositions_created,
            "llm": self.llm.stats,
            "pixel_obs": self.pixel_obs,
            "dream_training_active": self.dream_training_active,
            "developmental_stage": self.stage_controller.stage,
        }

    # -----------------------------------------------------------------------
    # Default configuration
    # -----------------------------------------------------------------------

    @staticmethod
    def _default_config() -> Dict:
        """Minimal default configuration for quick prototyping."""
        return {
            "environment": {
                "name": "CartPole-v1",
                "max_episode_steps": 500,
            },
            "world_model": {
                "stochastic_size": 16,
                "stochastic_classes": 16,
                "deterministic_size": 256,
                "encoder_hidden": 128,
                "buffer_capacity": 50000,
                "batch_size": 8,
                "sequence_length": 30,
                "learning_rate": 1e-4,
            },
            "curiosity": {
                "feature_dim": 128,
                "hidden_dim": 128,
                "forward_loss_weight": 0.8,
                "inverse_loss_weight": 0.2,
                "curiosity_reward_scale": 1.0,
                "curiosity_reward_clip": 5.0,
                "learning_rate": 1e-4,
            },
            "knowledge_graph": {
                "embedding_dim": 32,
                "embedding_model": "TransE",
                "embedding_epochs": 50,
                "min_confidence": 0.7,
            },
            "symbolic_decoder": {
                "num_categories": 5,
                "hidden_dim": 128,
                "learning_rate": 1e-4,
                "confidence_threshold": 0.6,
            },
            "skill_bank": {
                "storage_dir": "./skill_bank_data",
                "mastery_threshold": 0.8,
                "mastery_window": 50,
                "min_episodes_before_save": 20,
            },
            "policy": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "intrinsic_reward_weight": 0.7,
                "extrinsic_reward_weight": 0.3,
            },
            "glue": {
                "gnn_hidden_dim": 64,
                "gnn_output_dim": 32,
                "goal_embedding_dim": 16,
            },
            "llm": {
                "enabled": True,
                "model": "llama3.1:8b",
                "perception_interval": 50,
                "goal_interval": 25,
                "reasoning_interval": 100,
            },
            "dream_training": {
                "enabled": True,
                "warmup_episodes": 50,
                "imagination_horizon": 15,
                "batch_size": 16,
                "n_dream_batches": 1,
                "dream_curiosity": False,
                "hidden_dim": 400,
                "actor_lr": 3e-5,
                "critic_lr": 3e-5,
                "gamma": 0.997,
                "lambda_": 0.95,
                "entropy_scale": 3e-4,
                "target_ema": 0.98,
            },
            "loop": {
                "total_timesteps": 50000,
                "curriculum_enabled": True,
                "log_dir": "./logs",
                "verbose": 1,
            },
        }


# ---------------------------------------------------------------------------
# ============= GLUE LAYER INTERFACE =============
# ---------------------------------------------------------------------------
# This is the placeholder for the custom glue code you'll write later.
# It defines the INTERFACE that connects DreamerV3's latent world model
# to the skill bank and back into the policy layer.
#
# The glue layer is the novel research contribution (Section 3.2 in the
# reference doc). It handles:
#   1. Skill detection logic: when has the agent mastered a capability?
#   2. Skill retrieval: how to select relevant prior skills for new goals
#   3. Curriculum management: how to escalate difficulty as skills compound
#   4. Neo4j ↔ DreamerV3 bridge: extracting symbolic facts from latent data
# ---------------------------------------------------------------------------

class GlueLayerInterface:
    """
    Interface for the glue layer connecting RSSM latent space to symbolic reasoning.

    Concrete implementation: developmental_ai.core.glue_layer.GlueLayer
    which is wired into the DevelopmentalAI loop above.

    All four methods are now implemented:
      1. extract_symbolic_from_latent → via SymbolicDecoderManager
      2. select_skills_for_goal → via SkillSelector (cosine + graph + prereqs)
      3. detect_skill_mastery → via AdvancedMasteryDetector (KL + reward + rules)
      4. manage_curriculum → via AdvancedCurriculumManager (depth + density + error)
    """

    def extract_symbolic_from_latent(
        self,
        latent_state: torch.Tensor,
        world_model: WorldModel,
    ) -> List[SymbolicFact]:
        """Implemented via SymbolicDecoderManager — latent → category probabilities → facts."""
        raise NotImplementedError

    def select_skills_for_goal(
        self,
        goal_embedding: torch.Tensor,
        skill_bank: SkillBank,
        knowledge_graph: InMemoryKnowledgeGraph,
    ) -> List[str]:
        """Implemented via SkillSelector — cosine similarity + graph overlap + prereqs."""
        raise NotImplementedError

    def detect_skill_mastery(
        self,
        latent_trajectory: torch.Tensor,
        reward_history: List[float],
        knowledge_graph: InMemoryKnowledgeGraph,
    ) -> bool:
        """Implemented via AdvancedMasteryDetector — KL trend + reward stability + rule confidence."""
        raise NotImplementedError

    def manage_curriculum(
        self,
        skill_bank: SkillBank,
        knowledge_graph: InMemoryKnowledgeGraph,
        current_difficulty: int,
    ) -> int:
        """Implemented via AdvancedCurriculumManager — skill depth + KG density + pred error."""
        raise NotImplementedError
