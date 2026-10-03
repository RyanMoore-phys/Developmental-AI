# Architecture — the Developmental-AI organism

A curiosity-driven neurosymbolic RL agent that learns Minecraft Treechop with no
demonstrations. Runs as a **lifelong continuous stream**: 2 parallel MineRL envs,
stepped in 1024-step "segments" (what the logs call "episodes"). Code lives in
`developmental_ai/` in the parent repo.

## Stream topology (critical to understand)

- **env 0 = PRIMARY** — the ONLY stream PPO trains on (single-stream GAE). Gets
  the vision-magnet shaping. This is the policy that actually improves.
- **envs 1–3 = SCOUTS** — feed the world model, curiosity, and goal discovery,
  but NOT PPO gradients. They accelerate discovery, not policy learning.
- **Lifelong resets:** env resets ONLY on 64-log success or a client crash. The
  8000-tick time-truncation is deliberately removed (`create_server_quit_producers`
  under `_lifelong`). So a stream lives in ONE continuous world for a long time.

## Subsystems

| Subsystem | File(s) | Role |
|---|---|---|
| **Env adapter** | `environments/minerl_env.py` | Wraps MineRL 1.0 / MCP-Reborn. Adapter-side reward (the 1.0 handler is dead). Discrete→macro action space. Crash-rebuild recovery. |
| **World model (RSSM)** | `world_model/rssm.py` | DreamerV3-style recurrent state-space model. `observe_step` returns a NEW state dict each step. |
| **Curiosity (LP)** | `curiosity/learning_progress.py`, `icm.py` | Learning-Progress intrinsic reward: per-scene error-reduction, prototype-VQ pixel bucketing. The agent's central drive. |
| **Vision magnet** | `llm/vision_scaffold.py` | Curiosity-ranked shaping on the PRIMARY stream: steers toward the most-curious in-view object; cold-start floor; **tree-seeking drive** (potential on tree-visibility + budgeted forward nudge). No second VLM — reads the grounding head's probs + LP. |
| **VLM symbolic grounding** | `llm/vlm_symbolizer.py` | Local llava:7b (Ollama) labels frames; a head learns to predict labels from the agent's own latent; emits perceptual + causal facts into the KG; per-predicate reliability. |
| **Goal discovery** | `core/achievement_goals.py` | Reward spike ≥0.9 → a broadcaster slot keyed by the grounded block type. IMGEP frontier target selection. Working-set↔long-term paging. |
| **Competence self-model** | `core/self_model.py` | Online logistic per-goal competence estimate (Bernoulli updates). Gates which skills are offered as options. |
| **Skills-as-options** | `policy/options.py`, `skill_bank/` | Minted skills become invocable options (SMDP). Frozen-weight slots. Scripted `chop_trunk` bootstrap macro. |
| **PPO policy** | `policy/` | The base actor-critic. Meta-policy head widened P→P+K for option selection. |
| **Dream training** | dream actor/critic | DreamerV3 imagination (trust-gated). NOTE audit: largely unreachable under lifelong — see AUDIT_FINDINGS. |
| **Lifelong controller** | `core/lifelong.py`, `core/lifelong_state.py` | Forever-mode, cross-segment RSSM carry, death-only reset, working-set sizes. |
| **Orchestration** | `core/developmental_loop.py` | The main loop. `_collect_segment` is the lifelong step body; the mint block + per-segment maintenance run in `run()`. |
| **Brain viewer** | `skill_bank/brain_state.py`, `viewer/` | Emits `brain_state.json` for a live web view of goals/skills/firings. |

## Reward structure (env adapter, `minerl_env.py`)

- **+1 per log gained** in inventory (high-water; the Treechop task reward).
- **Tiered first-break** (once per block TYPE per episode-horizon): log **+5.0**,
  solid-needs-hold **+1.0**, leaves **+0.3**, trivial plants **+0.15**.
- Leaves priced at 0.3 (below the 0.9 goal-spike) so foliage does NOT mint goals.
- Terminated at 64 logs (Treechop success). No time-truncation in lifelong.

## Action space (discrete → MineRL macro)

`1,2` = forward · `3` = turn-left · `4` = turn-right · `5` = attack (hold =
re-pick) · `6` = attack+forward · `7` = look-up · `8` = look-down.
`action_repeat=2`, camera macros scaled so net turn is invariant.

## The reward mixer principle (a HARD constraint)

Intrinsic (curiosity) and extrinsic (env) rewards are combined by a mixer that
**anneals naturally** as competence grows. **Never hard-fix the intrinsic/
extrinsic ratio.** Any shaping (magnet, seek) must feed `shaped_extrinsic` so it
rides that anneal.

## Config

`configs/minecraft_lifelong.yaml` — the live run config. Key blocks:
`parallel_envs` (num_envs 4), `curiosity` (mode learning_progress),
`llm.vision` (the magnet + seek params), `skills_as_options`, `goals`,
`lifelong`, `developmental_stages` (force_exploit/force_imagine clocks).
