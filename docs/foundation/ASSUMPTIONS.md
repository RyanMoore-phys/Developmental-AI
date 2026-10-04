# Assumption register (Stage 1)

Machine-readable twin: `developmental_ai/foundation/runtime/assumptions.py`.
`tests/_foundation_baseline_smoke.py` checks that the sensor table below, the
Python register and the live sensor registry (`developmental_ai/sensors`) all
agree. If you add, remove or reclassify a sensor, update all three or the
build fails.

Categories:

- **framework-math**: math and bookkeeping the framework supplies.
- **adapter-metadata**: what the environment adapter declares.
- **learned-from-experience**: what the agent has to acquire. These are listed
  so that a supplied version of one is easy to spot.
- **evaluator-only**: may be measured, but must never reach observations,
  reward, memory or actions.

## Sensors (derived from code)

"Policy-visible" means the sensor is enabled and is not RED.
`SensorBus.read_policy` never visits a RED sensor. RED values go only to
`info["oracle"]`, and the only thing that reads that is
`DevelopmentalAI._oracle_observe`.

The last column is for `configs/minecraft_skybot.yaml` (`sensors.enabled`).

| sensor | class | category | skybot |
|---|---|---|---|
| `proprio` | green | adapter-metadata | policy |
| `screen_fx` | green | adapter-metadata | policy |
| `fovea_native` | green | adapter-metadata | policy |
| `fovea_delta` | green | adapter-metadata | policy |
| `dead_reckon` | green | adapter-metadata | policy |
| `motion` | green | adapter-metadata | policy |
| `light` | green | adapter-metadata | policy |
| `sky` | green | adapter-metadata | policy |
| `audio` | green | adapter-metadata | registered, not enabled |
| `hud` | green | adapter-metadata | registered, not enabled |
| `true_position` | red | evaluator-only | enabled, oracle only |

Proprio fields, in wire order (`minerl_env.py PROPRIO_KEYS`; the smoke test
checks this by AST): food, saturation, life, depth, has_tool, gui_open,
hurt_recent, carrying, swing, pitch, moved, head_sin, head_cos.

## Supplied structure

| id | category | statement | where |
|---|---|---|---|
| FM1 | framework-math | RSSM: categorical stochastic latent and a deterministic recurrent state | `world_model/rssm.py` |
| FM2 | framework-math | PPO with GAE. Reward = w_i·scaled intrinsic + w_e·extrinsic, with adaptive gating and a ratio cap | `policy/actor_critic.py` RewardMixer |
| FM3 | framework-math | Progress potentials telescope. State costs use the plain difference (γ=1) | CLAUDE.md §4.3 |
| FM4 | framework-math | Heading conventions. Dead reckoning at fixed `DR_SCALES` | `sensors/heading.py`, `sensors/builtins.py` |
| FM5 | framework-math | Perspective flow uses a pinhole projection | `world_model/rssm.py`, `docs/PERSPECTIVE_LEARNING.md` |
| FM6 | framework-math | Content-hashed manifests and salted-hash episode splits | `foundation/runtime/` |
| AM1 | adapter-metadata | MineRL buttons and camera, held for `action_repeat` (4) ticks | `environments/minerl_env.py` |
| AM2 | adapter-metadata | 128 px POV (downsampled from a 384 px render) and a 32 px native fovea | config `environment`, `sensors` |
| AM3 | adapter-metadata | Camera FOV of 70° | config `world_model.camera_fov_deg` |
| AM4 | adapter-metadata | 13 normalised proprio fields with normalisers the adapter chose | `minerl_env.py _proprio` |
| AM5 | adapter-metadata | **`proprio.depth` comes from the engine's true y** (`clip((63-ypos)/63)`) | `minerl_env.py _proprio` |
| AM6 | adapter-metadata | `has_tool`, `carrying` and `gui_open` come from engine inventory and GUI state, without item names | `minerl_env.py _proprio` |
| AM7 | adapter-metadata | **Extrinsic reward tiers are keyed by engine block names** (log = 20.0) and decay with per-type break counts | config `environment.break_*` |
| AM8 | adapter-metadata | Each bus channel declares its width, shape and version. `layout_hash` covers the order | `sensors/registry.py` |
| LE1–LE6 | learned | Dynamics, goals, skills (copies, §4.6), slots and identity, KG facts, depth from motion | various |
| EO1 | evaluator-only | RED sensors go to `info["oracle"]` and then only to `_oracle_observe` | `sensors/registry.py`, loop |
| EO2 | evaluator-only | Scoreboard outputs (`breaks_per_log`, `logs`, reward shares) in `metrics.jsonl` / `heartbeat.jsonl` | loop `_emit_metrics` |
| EO3 | evaluator-only | Held-out split assignment must not affect which data is trained on | `foundation/runtime/splits.py` |

## Flagged inconsistencies (recorded here, not changed)

- **AM5:** one component of the true position reaches the policy through
  proprio, but the full true position is RED. By the bus's own rule ("true
  coordinates" are meanings), `depth` should be reviewed.
- **AM7:** block identity (the engine's names) enters the extrinsic reward.
  It does not enter the observation. This is supplied structure in the reward
  channel, and the tier values have not been re-audited.
- **EO2:** the scoreboard output is evaluator-only, but the engine break
  counts behind it also drive the AM7 reward decay. So the counter is not
  evaluator-only at its source.
