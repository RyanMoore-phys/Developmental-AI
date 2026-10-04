"""Machine-readable assumption register (plan Stage 1 item 2, §2.1).

WHAT IT CLAIMS
    Every piece of structure SUPPLIED to SkyBot (rather than earned) is listed
    here under exactly one category:

        framework-math         coordinate algebra, probability bookkeeping,
                               model classes, shaping forms
        adapter-metadata       what the environment adapter declares: action
                               space, sensor shapes, normalisers, reward tiers
        learned-from-experience  what the agent must acquire (listed so that
                               a supplied version of it is recognisable as a
                               violation)
        evaluator-only         information that may be MEASURED but must never
                               reach observations, reward, memory or actions

    The sensor half of the register is checked against the LIVE sensor
    registry (`developmental_ai.sensors.build_default_bus`) and the proprio
    field list against the adapter source, by tests/_foundation_baseline_
    smoke.py and tests/unit/test_foundation_runtime_unit.py — so adding,
    removing or reclassifying a sensor without updating this file (and
    docs/foundation/ASSUMPTIONS.md) fails the build instead of going stale.

Policy-visible means: enabled in the config AND not RED. RED sensors are read
only by `SensorBus.read_oracle` into `info["oracle"]`, whose single consumer
is `DevelopmentalAI._oracle_observe` (evaluation sink).
"""

from __future__ import annotations

import ast
import os
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence

FRAMEWORK_MATH = "framework-math"
ADAPTER_METADATA = "adapter-metadata"
LEARNED = "learned-from-experience"
EVALUATOR_ONLY = "evaluator-only"
CATEGORIES = (FRAMEWORK_MATH, ADAPTER_METADATA, LEARNED, EVALUATOR_ONLY)

# ---------------------------------------------------------------------------
# Sensors: hand-written, CHECKED against the live registry. name -> class.
# ---------------------------------------------------------------------------
SENSOR_CLASSIFICATION: Dict[str, str] = {
    "proprio": "green",
    "screen_fx": "green",
    "fovea_native": "green",
    "fovea_delta": "green",
    "dead_reckon": "green",
    "motion": "green",
    "light": "green",
    "sky": "green",
    "audio": "green",
    "hud": "green",
    "true_position": "red",
}

# The 13 proprio fields, in wire order (environments/minerl_env.py
# PROPRIO_KEYS). Checked against the source by AST, because importing the
# adapter needs gymnasium.
PROPRIO_FIELDS = ("food", "saturation", "life", "depth", "has_tool",
                  "gui_open", "hurt_recent", "carrying", "swing", "pitch",
                  "moved", "head_sin", "head_cos")

# info keys: which carry policy input, which are evaluation sinks.
POLICY_INFO_KEYS = ("sensors", "proprio")
EVALUATOR_INFO_KEYS = ("oracle",)


class Assumption(NamedTuple):
    id: str
    category: str
    statement: str
    where: str
    policy_visible: Optional[bool]   # None = not an input channel
    note: str = ""


_REGISTER: List[Assumption] = [
    # ---- framework-math -------------------------------------------------
    Assumption("FM1", FRAMEWORK_MATH,
               "World model class: RSSM with categorical stochastic latent "
               "and deterministic recurrent state (DreamerV3-style).",
               "developmental_ai/world_model/rssm.py", None),
    Assumption("FM2", FRAMEWORK_MATH,
               "Policy learning is PPO with GAE; primary reward is a linear "
               "mix w_i*scaled_intrinsic + w_e*extrinsic with adaptive "
               "gating and a return-ratio cap.",
               "developmental_ai/policy/actor_critic.py (RewardMixer.mix)",
               None),
    Assumption("FM3", FRAMEWORK_MATH,
               "Shaping forms: progress potentials telescope (gamma*phi' - "
               "phi); state costs use the plain difference (gamma=1).",
               "CLAUDE.md §4.3; tests/_gui_farm_smoke.py", None),
    Assumption("FM4", FRAMEWORK_MATH,
               "Heading conventions and dead-reckoning integration at fixed "
               "scales (DR_SCALES) over sensed body motion.",
               "developmental_ai/sensors/heading.py, sensors/builtins.py",
               None),
    Assumption("FM5", FRAMEWORK_MATH,
               "Perspective flow uses a pinhole projection model (the FOV "
               "VALUE is adapter metadata, see AM3).",
               "developmental_ai/world_model/rssm.py (flow_*), "
               "docs/PERSPECTIVE_LEARNING.md", None),
    Assumption("FM6", FRAMEWORK_MATH,
               "Evaluation bookkeeping: content-hashed manifests and "
               "salted-hash episode splits.",
               "developmental_ai/foundation/runtime/{manifest,splits}.py",
               None),
    # ---- adapter-metadata -----------------------------------------------
    Assumption("AM1", ADAPTER_METADATA,
               "Action space: MineRL buttons + camera deltas, held for "
               "environment.action_repeat (4) game ticks per decision.",
               "developmental_ai/environments/minerl_env.py; "
               "configs/minecraft_skybot.yaml environment.action_repeat",
               True),
    Assumption("AM2", ADAPTER_METADATA,
               "Observation: 128px RGB POV (block-mean of a 384px native "
               "render); 32px native fovea at the crosshair.",
               "configs/minecraft_skybot.yaml environment.image_size, "
               "render_size, sensors.fovea_size", True),
    Assumption("AM3", ADAPTER_METADATA,
               "Camera field of view 70 degrees.",
               "configs/minecraft_skybot.yaml world_model.camera_fov_deg",
               None),
    Assumption("AM4", ADAPTER_METADATA,
               "Proprio: 13 fixed-order normalised body fields "
               "(PROPRIO_FIELDS) with adapter-chosen normalisers "
               "(life/20, MOVE_SCALE derived from action_repeat, ...).",
               "developmental_ai/environments/minerl_env.py PROPRIO_KEYS, "
               "_proprio", True),
    Assumption("AM5", ADAPTER_METADATA,
               "proprio.depth is computed from the ENGINE-TRUE y coordinate "
               "(clip((63 - ypos)/63)), with sea level 63 supplied.",
               "developmental_ai/environments/minerl_env.py _proprio", True,
               "A component of true position reaches the policy, while the "
               "full true position is RED. Inconsistent with the bus rule "
               "('true coordinates' are meanings); flagged, not changed."),
    Assumption("AM6", ADAPTER_METADATA,
               "proprio.has_tool / carrying / gui_open are derived from "
               "engine inventory and GUI state, without item names.",
               "developmental_ai/environments/minerl_env.py _proprio", True),
    Assumption("AM7", ADAPTER_METADATA,
               "Extrinsic reward tiers are keyed by engine block NAMES (log "
               "tier 20.0) and decayed by per-type break counts "
               "(break_decay_scale 25, persisted break memory).",
               "configs/minecraft_skybot.yaml environment.break_* ; "
               "developmental_ai/environments/minerl_env.py", None,
               "Block identity enters the REWARD (not the observation). "
               "Supplied structure; tier values not re-audited here."),
    Assumption("AM8", ADAPTER_METADATA,
               "Every enabled non-RED sensor-bus channel (see sensor table) "
               "declares width, shape and version; layout_hash covers order.",
               "developmental_ai/sensors/registry.py", True),
    # ---- learned-from-experience ----------------------------------------
    Assumption("LE1", LEARNED, "Action consequences / dynamics.",
               "world model training on replay", None),
    Assumption("LE2", LEARNED,
               "Goals: discovered from reward/novelty spikes, not declared.",
               "developmental_ai/core (goal broadcaster), goals.*", None),
    Assumption("LE3", LEARNED,
               "Skills: minted as policy copies; names carry no weight "
               "(CLAUDE.md §4.6).", "developmental_ai/skill_bank/", None),
    Assumption("LE4", LEARNED,
               "Object identity / slots and spatial memory.",
               "developmental_ai/slots/, developmental_ai/spatial/", None),
    Assumption("LE5", LEARNED,
               "Knowledge-graph facts (asserted and retracted from "
               "experience).", "developmental_ai/knowledge_graph/", None),
    Assumption("LE6", LEARNED, "Depth/flow from self-motion.",
               "docs/PERSPECTIVE_LEARNING.md", None),
    # ---- evaluator-only -------------------------------------------------
    Assumption("EO1", EVALUATOR_ONLY,
               "RED sensors (true_position) -> info['oracle'] -> "
               "_oracle_observe only; never replay, reward or policy.",
               "developmental_ai/sensors/registry.py read_oracle; "
               "environments/minerl_env.py; core/developmental_loop.py "
               "_oracle_observe", False),
    Assumption("EO2", EVALUATOR_ONLY,
               "Scoreboard outputs (breaks_per_log, logs, reward shares) in "
               "runlogs/metrics.jsonl and heartbeat.jsonl.",
               "core/developmental_loop.py _emit_metrics, _emit_heartbeat",
               False,
               "The OUTPUT is evaluator-only; the underlying engine break "
               "counts also drive AM7's reward decay."),
    Assumption("EO3", EVALUATOR_ONLY,
               "Held-out episode assignment (splits) must not influence "
               "training data selection.",
               "developmental_ai/foundation/runtime/splits.py", False),
]


def register() -> List[Assumption]:
    return list(_REGISTER)


def by_category() -> Dict[str, List[Assumption]]:
    out: Dict[str, List[Assumption]] = {c: [] for c in CATEGORIES}
    for a in _REGISTER:
        out[a.category].append(a)
    return out


def validate_register(entries: Iterable[Assumption] = None) -> None:
    seen = set()
    for a in (entries if entries is not None else _REGISTER):
        if a.category not in CATEGORIES:
            raise ValueError(f"{a.id}: unknown category {a.category!r}")
        if a.id in seen:
            raise ValueError(f"duplicate assumption id {a.id}")
        if a.category == EVALUATOR_ONLY and a.policy_visible:
            raise ValueError(f"{a.id}: evaluator-only cannot be "
                             f"policy-visible")
        if not a.statement or not a.where:
            raise ValueError(f"{a.id}: statement and where are required")
        seen.add(a.id)


# ---------------------------------------------------------------------------
# Derivation from code
# ---------------------------------------------------------------------------

def live_sensor_classification() -> Dict[str, str]:
    """{name: classification} from the live default bus."""
    from developmental_ai.sensors import build_default_bus
    bus = build_default_bus(len(PROPRIO_FIELDS), lambda ctx: None)
    return {n: bus.get(n).classification for n in bus.names()}


def policy_visible_sensors(enabled: Optional[Sequence[str]]) -> List[str]:
    """Names a config's bus feeds the policy: enabled and not RED (live)."""
    from developmental_ai.sensors import build_default_bus
    bus = build_default_bus(len(PROPRIO_FIELDS), lambda ctx: None,
                            enabled=list(enabled) if enabled else None)
    return [s.name for s in bus.policy_sensors()]


def evaluator_only_sensors(enabled: Optional[Sequence[str]]) -> List[str]:
    from developmental_ai.sensors import build_default_bus
    bus = build_default_bus(len(PROPRIO_FIELDS), lambda ctx: None,
                            enabled=list(enabled) if enabled else None)
    return [s.name for s in bus.oracle_sensors()]


def proprio_fields_from_source(repo_root: str = ".") -> tuple:
    """PROPRIO_KEYS from minerl_env.py, by AST (no gymnasium import)."""
    p = os.path.join(repo_root, "developmental_ai", "environments",
                     "minerl_env.py")
    tree = ast.parse(open(p).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "PROPRIO_KEYS":
                    return tuple(ast.literal_eval(node.value))
    raise LookupError("PROPRIO_KEYS not found in minerl_env.py")


def register_mismatches(repo_root: str = ".") -> List[str]:
    """Differences between this register and the live code. [] = in sync."""
    out = []
    live = live_sensor_classification()
    for n in sorted(set(live) | set(SENSOR_CLASSIFICATION)):
        a, b = SENSOR_CLASSIFICATION.get(n), live.get(n)
        if a != b:
            out.append(f"sensor {n}: register={a} live={b}")
    src = proprio_fields_from_source(repo_root)
    if src != PROPRIO_FIELDS:
        out.append(f"proprio fields: register={PROPRIO_FIELDS} source={src}")
    return out


def sensor_table(enabled: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    """Per-sensor rows for a config: class, enabled, policy-visible."""
    vis = set(policy_visible_sensors(enabled))
    en = set(enabled) if enabled else None
    rows = []
    for n, cls in live_sensor_classification().items():
        is_en = (n in en) if en is not None else (cls == "green")
        rows.append({"name": n, "classification": cls, "enabled": is_en,
                     "policy_visible": n in vis,
                     "category": (EVALUATOR_ONLY if cls == "red"
                                  else ADAPTER_METADATA)})
    return rows
