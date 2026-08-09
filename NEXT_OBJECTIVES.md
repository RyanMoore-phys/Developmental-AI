# Next Objectives — post-audit roadmap (July 2026)

Written after the H1–H5 audit packages landed. Prerequisite state: WM
action-causality fixed (rung 8 ✅), rung 9 law-induction ✅, curiosity
normalization settled, honest-eval machinery (H4) in place, compute moved to
the RunPod pod (48 vCPU + RTX 4000 Ada 20GB; see memory/vm notes — run CPU
mode until Objective 2 lands). Ordered by leverage; each has a falsifiable
deliverable, per house rules.

## 0. Close out the audit validation (IN FLIGHT on the pod)
Capstone H2 validation (4 seeds, LLM-off; pass = 0/4 collapses AND AUC ≥
vanilla's ~0.71), rung 4 full rerun (does dreaming help now that the WM is
causal — or only stop hurting?), rung 3 final artifact. When these land,
H1–H5 are all empirically satisfied, not just code-complete.

## 1. Richer environment — Crafter (the big one) — **DEVELOPMENT DONE (2026-07-14)**

> Integration complete and smoke-verified on real frames (`_crafter_smoke.py`,
> 5/5): `environments/crafter_env.py` (gymnasium adapter over crafter 1.8.3's
> legacy-gym API, honest terminated/truncated split via `info["discount"]`),
> `NativePixelObsWrapper` (flat CHW float32 [0,1], exact round-trip verified,
> no Welford, no re-render), `make_env` Crafter branch, `configs/crafter.yaml`
> (self-complete). Pixel gates added in the loop: `_NullSymbolicDecoder` (the
> real one would be ~16M params of per-pixel heads), per-pixel fact/scene
> extraction skipped. Probe: 8.7 steps/s Mac CPU at full size (12.8M-param CNN
> WM), achievements already unlocking under exploration. Vector-path
> regression fully green. DEFERRED (known, acceptable v1): ICM feature encoder
> is still an MLP on flat pixels; replay stores float32 (buffer_capacity
> capped at 20k ≈ 1 GB — uint8 storage is the unlock for bigger buffers);
> LP-curiosity hash bucketing untested on pixels (novelty mode is default).
> **REAL 1M-STEP RUN DONE (2026-07-14, GPU L4, 4.0h, 69 steps/s):** clean —
> no NaN/crash, 0 KG facts leaked (pixel gates hold), 5863 episodes. Learning
> curve: reward 2.24 → early curiosity dip 0.9 → steady climb → plateau ~3.4
> (peak 3.72). Final avg reward ~3.4 (Crafter reward = achievements/episode;
> for scale: random ~1, weak PPO ~4, DreamerV3 ~14) — a legitimate but modest
> first-run score, headroom in the v1 deferrals (MLP-ICM, 20k float buffer,
> no tuning). 0 skills minted (avg competence 3.4/8 ≈ 0.42 < 0.8 mastery
> threshold — expected, not a bug). The developmental arc (explore-dip then
> exploit-climb) is the reward-mixer/stage-controller working as designed.
>
> **Post-run adversarial code review (workflow, 8 agents) found + FIXED 4 real
> latent bugs** none of which the happy path hit: (1) `_llm_submit_facts` not
> pixel-gated (per-pixel LLM dump — gated); (2) per-episode obs/action/reward
> history unbounded (0.5 GB/long-episode risk — capped at 64); (3) eval never
> passed the knowledge feature (symbolic agents eval'd on zeros — fixed);
> (4) continuous-action eval built a rank-3 action tensor (latent crash —
> fixed). Remaining nits logged, not blocking (GAE truncation ~1/2048 bias,
> inert CurriculumWrapper, raw_obs naming, viewer per-pixel stream). All
> re-verified on CUDA + full regression green.
**Why:** MiniGrid is saturated: the stack masters it, and every remaining
question ("does dreaming pay?", "do symbols scale?") needs a world with more
structure than one door and one key. Crafter (Hafner) is purpose-built for
exactly this project: open-ended survival, pixel observations, 22
achievements forming a natural developmental ladder (collect wood → craft
pickaxe → mine iron → ...), partial observability, day/night dynamics.
It is the community-standard benchmark for exactly the
curiosity/world-model/skill-acquisition questions this project asks.
**First steps:** `pip install crafter`; wire a config (pixel_obs=true, CNN
encoder path already exists in rssm.py); baseline run = vanilla PPO;
pre-register achievement-count@1M-steps metrics for full-stack vs vanilla.
**Falsifiable claim:** the developmental machinery (curiosity + skills +
dreaming) beats vanilla PPO on achievement breadth in a world that actually
demands cumulative learning — the claim MiniGrid could never test.

## 1b. Next stage — Craftax (deeper) — **INTEGRATED + REAL RUN LAUNCHED (2026-07-14)**

> Craftax (Matthews 2024): a JAX expansion of Crafter — 9 dungeon floors, 60+
> achievements, 43 actions. Chose the **SYMBOLIC** variant (`Craftax-Symbolic-v1`,
> flat 8268-d factored-state vector) over pixels: it fits the structured/law-
> induction goal, reuses the validated MLP world-model path, and avoids the
> pixel variant's non-square 130×110 frame (our CNN is square-only) + its
> lossy downsampling. `environments/craftax_env.py` (JAX-functional →
> gymnasium OO adapter: holds key/state/params, jits step, JAX→numpy at the
> boundary, terminated split via info discount), make_env Craftax branch
> (normalize_obs=False — obs already [0,1]), self-complete `configs/craftax.yaml`.
> Principled refactor: the per-dimension symbolic machinery (decoder heads +
> fact extraction) is now gated by `_skip_perdim_symbolic = pixel_obs OR
> obs_dim>2000` — so it auto-skips on Craftax's 8268-d obs (would've been 8268
> heads) while low-dim MiniGrid (151-d) is unchanged. Smoke-verified
> (`_craftax_smoke.py`, 4/4 on the pod) + full regression green on Mac AND pod
> (CPU + CUDA). Probe: 84 steps/s GPU (10.4M-param MLP WM; JAX-on-CPU env +
> torch-on-GPU learning, clean split). **Real 1M-step run DONE (2026-07-14,
> GPU L4, 2.27h, ~123 st/s effective).** Clean — no NaN/crash, 0 KG facts (gate
> holds), 3675 episodes. Learning: reward 2.15 → steady climb → 4.32 (peak
> 4.59), NO collapse, NO early dip — a CLEANER, HIGHER curve than Crafter
> (final 4.23 vs 3.39, in 2.3h vs 4h). Why the deeper world learned BETTER:
> the factored 8268-d obs is far more informative than raw pixels, Craftax has
> 60+ achievements (denser reward), and the MLP-on-structured-obs is easier to
> fit than CNN-on-pixels — vindicating the symbolic-over-pixels choice. The
> stack SCALES to a deeper world. DEFERRED: Craftax-aware symbolic channel
> (achievements/inventory via the rulebook) instead of the auto-skipped per-dim
> path; the pixel variant (non-square 130×110, needs square-resize).

## 2. Fix the GPU device bug — **FIXED (2026-07-14); CUDA-verified on pod**
`StandaloneActorCritic` now takes `device` (threaded from the loop), moves
actor/critic/conditioner to it, and creates every input tensor on it;
`train_step` keeps a CPU numpy copy for GAE (a CUDA tensor's `.numpy()` would
raise). Verified on CPU: `_gpu_device_smoke.py` (placement, select_action,
train_step, dream actor, end-to-end distill graph) + full regression green.
**First step of the next pod session: run `_gpu_device_smoke.py` and
`_crafter_smoke.py` with the GPU visible — that's the CUDA proof — then
launch Crafter training** (1M steps; ~32h on Mac CPU at 8.7 steps/s, the
RTX 4000 should cut that several-fold).

## 3. Rung 9 → the organism (rules leave the harness)
Rung 9 proved intervention-verified schemas transfer — using a scripted
navigator. Integrate the RuleBook into the live loop: (a) uncertain rules
become GOALS for the existing goal stack (experiment-driven curiosity);
(b) verified rules gate/augment dream rollouts (the WALL-E-2.0 idea already
cited in code comments); (c) rules persist in the KG as the semantic-memory
channel H5 cleared space for. **Falsifiable:** rung-9's lesion protocol,
re-run with the LEARNED policy instead of the scripted navigator.

## 4. Rung 7 autonomous retry — rule anchors
The one honest negative left standing: autonomous cross-domain
correspondence discovery. New tool since then: shared verified RULES.
If both domains contain "something opens something," the rule instances are
correspondence anchors the 2026-06 attempt lacked (its failure mode was
exactly non-identifiability without anchors). **Falsifiable:** the existing
rung-7 5-check bar, unchanged — it refused a false pass once already.

## 5. Capstone at paper strength
If Objective 0's 4-seed capstone is stable: extend to 8 seeds, add the
vanilla arm, bootstrap CIs (stats.py), seeded argmax eval (H4) — turning
"H2 fixed the collapse" into a defensible claim. Optionally revive LLM
shaping on the pod (Ollama fits comfortably in 251GB RAM / 48 cores — use
llama3.2:3b, cadence-capped) to test whether it adds ANYTHING post-H2; if
yes, distill its judgments into a tiny reward MLP (amortized shaping).

## 6. Rung 5 completion — allocation utility
The self-model is calibrated + load-bearing but tied uniform under strong
transfer. Build the weak/negative-transfer task-set the 2026-06 writeup
prescribed (e.g. interference pairs: task B's optimal policy punishes task
A's habits). **Falsifiable:** intact allocation beats uniform where practice
is NOT fungible; scramble still craters.

## 7. Legacy-rung statistics upgrade (cheap, pod-parallel)
Re-run the decisive rungs (3, 6, 8, 9) at 6–8 seeds with H4 protocol
(seeded argmax eval, CIs, artifacts) — 48 cores make this an afternoon.
Turns every headline claim into a re-derivable artifact. Enable
`parallel_envs` for single-run speed while at it.

## 8. Law-world extensions (rung 9b) — toward "laws of physics"
Compositional laws (two rules must combine), probabilistic laws (effects
with p<1 — the imperfection principle meets stochastic worlds), and
law-families with functional structure (e.g. conserved quantities in a
resource world). This is the staircase from "key opens door" toward the
multi-environment invariance pressure that squeezes out genuine physics-like
abstractions — and where an EBM/sampling reformulation would first earn a
real test, if that thread is ever picked back up.

---
*Standing constraints unchanged: free/open-source only, anneal naturally,
never wipe skill_bank_data, pre-registered metrics, always a vanilla
baseline, report function not experience. New one from this audit: reward
changes require a full-run canary — the baseline arm IS the canary.*
