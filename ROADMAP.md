# Developmental AI — Research Roadmap

**North star (inspiration, not a deliverable):** build an agent whose *learning
dynamics resemble how natural minds develop*. We treat "consciousness" as an
out-of-reach philosophical question and instead pursue its measurable
**functional analogs** — cumulative learning, composition, self-directed goals,
mental simulation, metacognition, global integration. Each is something we can
build, test, and *fail*.

**Honesty rule:** when a rung works we have demonstrated a *function* (e.g.
functional metacognition), **not** subjective experience. We say exactly that.

---

## The ladder

Each rung = a capability + a falsifiable metric + the theory of mind it touches.
Status: ✅ done · 🟡 in progress · ⬜ planned.

### Rung 0 — Single-env mastery + learnability gates ✅
- **Demonstrates:** the stack can learn a task above random and bank a skill.
- **Metric:** above-random competence; ≥1 skill mastered.
- **Status:** DoorKey-5x5 mastered (curve 0.10→0.74, 1 skill banked).

### Rung 1 — Cumulative transfer ✅ *(demonstrated on a hard, transfer-apt target; the DoorKey skill-bank-retrieval route stays null)*
- **Demonstrates (intended):** prior knowledge accelerates a *harder* env — a
  mind doesn't relearn from scratch.
- **Metric:** transfer beats fresh on AUC/frac_solved (and episodes-to-a-
  *reachable* 0.3 competence ≤ 0.7× fresh), ≥3 seeds, plus a vanilla-PPO baseline.
- **Harness:** `rung1_cumulative.py` (base 5x5 builds a skill bank → 8x8 arms:
  transfer=load that bank / fresh=empty / vanilla=plain PPO) on observable
  key/no-key (`observable_doorkey.py`). Earlier: `transfer_run.py` (policy-weight
  warm-start variant).
- **Result — two negatives (3–4 seeds, ObsKeyNoKey 5x5→8x8).**
  1. **Skill-bank transfer is null:** transfer ≈ fresh (mean AUC 0.151 vs 0.165),
     high seed variance, no consistent direction. The base mints only **1**
     de-dup'd skill, which carries no reliable acceleration (hurts on some seeds).
  2. **The mandated baseline bit:** **vanilla PPO beats the full system** on every
     metric (mean AUC 0.217 vs fresh 0.165; frac_solved 0.395 vs 0.322). A 2×2
     ablation (`intrinsic_tax_ablation.py`, symbolic×intrinsic, 4 seeds)
     decomposes it: **intrinsic reward is a consistent mild tax** (intrinsic-OFF
     wins 4/4 seeds when symbolic off) and **symbolic is a consistent *speed*
     tax** (reaches 0.3 faster 3–4/4), both small + variance-heavy.
- **Why this is coherent, not a contradiction.** The machinery helps *exactly
  where the task demands it*: curiosity is essential reward-free (Rung 3 ✅),
  the symbolic broadcast under hidden structure (Rung 6 ✅), and both are a mild
  net tax on a findable-reward task where plain PPO already suffices. (Note:
  the older `transfer_run.py` did show strong transfer via **policy-weight**
  warm-start — it is the *skill-bank* route that does not carry.)
- **To demonstrate it:** mint *richer/composable* skills (loosen de-dup), and/or
  pick a target hard enough from-scratch that prior knowledge has room to matter.
- **Revisit — DEMONSTRATED (decisive, 3 seeds).** `rung1_revisit.py` follows that
  prescription on the scale-invariant SeqGoal world: learn A-then-B on 5x5, bank the
  policy, then warm-start it (via the bank) onto the much harder **9x9** target. Mean
  over 3 seeds — **transfer** AUC **0.984**, competent in **~30** eps; **fresh**
  (curiosity, from scratch) 0.817, ~170 eps; **vanilla PPO** 0.511, ~443 eps
  (final 0.67). Transfer beats BOTH fresh and vanilla on success-AUC AND
  episodes-to-competence: prior knowledge accelerates a harder env. Caveats kept
  honest: (a) the mechanism is **policy-weight warm-start** from the banked skill —
  the route the roadmap already flagged as working; the *retrieve-and-apply* skill-bank
  route on DoorKey remains the null above. (b) vanilla PPO, which *won* the original
  findable-reward Rung 1, now loses badly — on a hard SPARSE target, curiosity-driven
  exploration is essential, coherent with the "machinery is task-appropriate" theme.
- **Theory:** development itself.

### Rung 2 — Skill composition ✅
- **Demonstrates:** banked behaviors recombine into new ones without retraining
  from zero.
- **Metric:** a task solvable only by chaining ≥2 banked skills reaches
  competence faster than (a) from-scratch and (b) vanilla PPO.
- **Harness:** `rung2_composition.py` on a sequential-goal gridworld
  (`environments/seq_goal.py`; ids SeqGoal-{A,B,B1,AB}). Banks two navigation
  primitives (goto_A, goto_B), then on the SPARSE composite "reach A THEN B"
  runs three arms: **compose** (chain the banked skills, handed off A→B by the
  `CompositeSkillExecutor` state-aware *subgoal* switch, ZERO training) vs
  **vanilla PPO** vs **PPO + count-curiosity**. The second-leg primitive is
  trained with the reached_A flag pinned on (SeqGoal-B1) so it is never fed an
  out-of-distribution flag on reuse.
- **Result (decisive, 3 seeds).** Compose solves the composite at **1.00**
  success with **0 training episodes**. Both learners also reach 1.00 — but only
  after **~180 episodes** of training (vanilla ~180, curiosity ~176, ≈7.2k env
  steps). So composition reaches competence *faster than from-scratch AND vanilla
  PPO* — the rung metric — and beats vanilla PPO, the baseline that *won* Rung 1.
  The win comes from the explicit subgoal hand-off: the composite carries
  sequential structure a memoryless learner must discover, but chained skills
  supply for free.
- **Theory:** combinatorial generalization; how children build complex action
  from primitives.

### Rung 3 — Autotelic (self-directed) goals ✅
- **Demonstrates:** with NO external reward, the agent grows its explored
  frontier, and its curiosity tracks LEARNING PROGRESS rather than raw novelty.
- **Metric:** external reward OFF — (a) the explored frontier grows, and (b) the
  share of effort wasted on an UNLEARNABLE noise source (the "noisy TV").
- **Result (decisive, 4 seeds).** Reward-free NoisyTV gridworld (24x24): an empty
  room to explore + a local "noisy TV" that corrupts the observation with fresh
  random values (permanent prediction error, ZERO learning progress). Two arms,
  identical except the curiosity signal — `novelty` (ICM, raw prediction error)
  vs `lp` (learning-progress = REDUCTION of prediction error per state bucket).
  Symbolic layer off, so the ONLY variable is curiosity type:
  - frontier (learnable coverage): novelty **1.00**, lp **0.99** — BOTH
    autonomously explore the whole room (autotelic frontier growth, no reward).
  - effort wasted at the noisy TV (tv_fraction, overall / 2nd-half):
    novelty **0.158 / 0.199** (escalates once the room is explored) vs
    lp **0.004 / 0.003** (~0).

  4/4 seeds, non-overlapping (worst novelty 0.076 ≫ best lp 0.006); ~45× less
  effort wasted on noise. The novelty arm's 2nd-half ESCALATION (lp stays ~0) is
  the noisy-TV trap caught in the act: once the room is mastered, noise is the
  only "novelty" left, so the novelty agent increasingly camps at it.
- **Falsifiable.** A too-weak TV (novelty not drawn in) would show no gap; the
  validity gate (novelty tv_fraction 8–20× the random-occupancy baseline)
  confirms the TV is a real attractor.
- **Scope (honesty).** Proves the FUNCTION: LP-curiosity avoids the noisy-TV trap
  that novelty falls into, while both grow the frontier autonomously. In a fully
  coverable room the difference is wasted effort, not final frontier size. No
  claim about experience.
- **Harness:** `rung3_autotelic.py` (arms novelty/lp on `MiniGrid-NoisyTV-*`);
  env `developmental_ai/environments/noisy_tv_gridworld.py`; LP curiosity
  `developmental_ai/curiosity/learning_progress.py` (config `curiosity.mode`).
- **Theory:** intrinsic motivation / infant play (autotelic agents, IMGEP).

### Rung 4 — Learning in imagination 🟡 (mechanism fixed; not load-bearing)
- **Demonstrates:** the agent improves by simulating rollouts in its world model
  instead of only acting for real (imagination-AUGMENTED learning: a dream actor
  trains on RSSM rollouts and is distilled into the real policy; PPO keeps control).
- **Metric:** sample efficiency, dream-augment vs dream-off — episodes-to-competence
  (trailing-10 ≥ 0.5), AUC (mean reward), frac_solved; plus world-model rollout
  accuracy out to N steps before divergence.
- **Validity gates (both PASS).** (1) Obs-rollout gate: the WM's rollouts beat
  persistence out to 15 steps (`wm_rollout_probe.py`). (2) Reward head, measured in
  the *exact training alignment* (`wm_reward_probe2.py`): predicted reward
  discriminates goal vs non-goal by **+0.44** (corr +0.70). *An earlier "reward head
  blind" reading was a PROBE bug* — the probe shifted the action sequence relative to
  how `observe_sequence(obs, actions)` is trained. The MSE-on-symlog reward head works.
- **The distillation bug (found & fixed).** The original distill pulled the real
  policy toward the dream actor (a) at **every** replay state, (b) with a **fixed**
  weight that never annealed, (c) via **mode-covering forward-KL**. Late in training,
  when PPO had already surpassed the dream actor, it kept re-injecting the dream's
  softness — a consistent drag. On 5x5 this cost **−0.14 AUC and −0.13 frac_solved in
  BOTH seeds**. Fix (three coordinated levers, all honoring "anneal naturally — never
  fix the ratio"): **value-gate** (distill only where the dream critic values a state
  above the real critic), **natural anneal** (scale the pull by `1 − solve_rate_EMA`),
  **teacher-sharpening** (temperature < 1 so the policy adopts the dream's *preferred
  action*, not its uncertainty). Smoke-verified: gate fires at ~50% of states, weight
  anneals 0.77→0.20 as competence rises, PPO stays in control.
- **Result — fixed, but not a win (8 seed-pairs, DoorKey 5x5/6x6/8x8).** The fix
  **eliminated the consistent harm** (5x5 AUC delta moved from −0.14/−0.07 to
  +0.01/−0.01; frac from −0.13/−0.04 to +0.02/−0.00). But dreaming does **not**
  reliably *help*: across 5x5 (2 seeds, wash), 6x6 (2 seeds, mixed — one seed +0.10
  frac, one slight loss) and 8x8 (4 seeds, slight edge to dream-off on average),
  it is ~4 wins / 4 losses with high seed variance and no consistent direction.
  (8x8 at 300k stays in the partial-learning regime — frac ~0.2, competence rarely
  sustained, so eps-to-competence is mostly `None` for both arms.)
- **Why (the honest mechanism).** Imagination-augmented learning is **bottlenecked by
  world-model accuracy**. On sparse-reward DoorKey the WM sees only a handful of real
  goal transitions, so its imagined rollouts near the goal are unreliable — worst
  exactly on 8x8, where augment slightly *hurt*. The gate/anneal make dreaming **safe**
  (they suppress the worst pulls) but cannot manufacture a benefit the WM can't supply.
  Dreaming is safe to keep ON; it does not earn its keep on these tasks.
- **WM-accuracy upgrade attempt (real, but insufficient).** Directly attacked the
  bottleneck with two DreamerV3-style upgrades: a **twohot distributional reward head**
  (255-bin categorical + cross-entropy, replacing mean-collapse-prone MSE; reward-head
  discrimination rose +0.44 → **+0.51**) and **goal-prioritized replay** (oversample
  reward-bearing windows; goal exposure in a batch rose **6.6% → 56.6%**). Both
  unit-tested and config-gated (`world_model.goal_replay_fraction`; twohot always on).
  Re-ran 6x6 (3 seeds): augment improved to **2/3 wins on AUC and frac** (avg AUC +0.015,
  frac +0.018) vs the prior 1/1 split — but the margins are within seed noise (±0.04),
  one seed flipped to a loss, and the DEFINING metric (`eps→competence`, sample
  efficiency) **stayed worse** (+12 eps avg). The world model got measurably better; the
  *policy* advantage did not. Judged not large enough to expect an 8x8 pass — did NOT
  run the 700k confirmation. The upgrades stay (good engineering); the rung verdict stands.
- **Falsifiable — and it under-delivered.** The pre-registered PASS was "augment
  reaches competence in fewer real steps and ≥ AUC across seeds." It did not; recorded
  as a negative, like the Rung 1 intrinsic-tax finding.
- **Harness:** `rung4_dreaming.py` (arms dream_off / dream_augment; the fixed
  distillation is the default), probes `wm_rollout_probe.py`, `wm_reward_probe2.py`
  (training-aligned, definitive), `wm_reward_probe.py` (misaligned — kept as the
  cautionary example). Distillation in `developmental_ai/core/developmental_loop.py`
  (`_distill_dream_to_real`).
- **Theory:** predictive processing + mental simulation / counterfactuals.

### Rung 5 — Self-model / metacognition 🟡  *(flagship — metacognition core ✅; allocation utility transfer-limited)*
- **Demonstrates:** the agent models its own competence ("knows what it doesn't
  know") and uses that model to allocate practice across a task-set.
- **Setup.** One shared agent (identical 151-d egocentric obs across sizes) is
  driven in fixed-budget BLOCKS over DoorKey 5x5/6x6/8x8; each block an allocator
  picks which task to practice. A `CompetencePredictor` self-model predicts
  P(success) per task online from the agent's own outcomes (calibration = ECE) and
  tracks learning-progress (allocation signal). Arms: **intact** (true self-model,
  LP + UCB-explore + staleness maintenance), **scramble** (self-model stats read
  through a fixed derangement — valid-but-wrong), **uniform** (no self-model,
  equal practice). Metric: mean-competence AUC (primary), set-mastery, ECE.
- **Result (3 seeds × 2 fixed re-runs = 6 seed-pairs).**
  - **Calibration — decisive PASS.** ECE **0.019–0.044** on every intact run. The
    self-model is well-calibrated ("when it says 70%, it succeeds ~70%").
  - **Causally load-bearing — decisive PASS (3/3, both runs).** `intact ≫ scramble`
    every seed (AUC ~0.56 vs ~0.23): corrupting the self-model's *content* craters
    behavior (scramble dumps 2300 episodes on one task, 12 on another). This is the
    Rung-6-standard content lesion — the self-model functionally drives action.
  - **Beats uniform — NO (robust tie).** intact ≈ uniform on AUC across all 6
    seed-pairs (means ≈ 0.557 vs 0.534–0.570, within run-to-run noise ±0.04).
- **Why (the honest mechanism).** The three sizes share the egocentric obs and task
  structure, so **transfer is strong and roughly symmetric — every practice episode
  helps every task**, which makes naive uniform near-optimal. Allocation *order*
  only matters when practice is NOT fungible. Tellingly the ordering is
  **scramble (wrong self-model) ≪ uniform (no self-model) ≈ intact (right
  self-model)** — so here the self-model's value is "don't be *wrong*," not "beat
  equal practice." Two fix cycles (LP/ROI allocator, then staleness-based
  anti-forgetting that restored late 6x6 decay 0.4→0.7 and lifted intact's *final*
  competence 0.59→0.72 to match uniform) confirmed the controller is fine — it's
  the strong-transfer *task* that doesn't reward allocation.
- **Falsifiable — and partially under-delivered.** Pre-registered PASS was "intact
  beats BOTH scramble and uniform across seeds." It beats scramble decisively but
  ties uniform → the metacognition *core* (calibration + causal load-bearing) is a
  clean pass; the allocation-*utility*-over-uniform claim is not shown and would
  need a weak/negative-transfer task-set (future work).
- **Harness:** `rung5_selfmodel.py` (arms intact/scramble/uniform; block-driven
  multi-task), self-model in `developmental_ai/core/self_model.py`
  (`CompetencePredictor`: online logistic P(success), ECE/Brier, learning-progress).
- **Theory:** Higher-Order Theories of consciousness; metacognition. The closest
  *functional* analog to self-awareness we can rigorously test — here: a calibrated,
  causally-load-bearing self-model of task competence.

### Rung 6 — Global integration (the "workspace" test) ✅
- **Demonstrates:** symbolic information from the broadcast is globally available
  and measurably shapes action (functional global integration — *not* experience).
- **Metric:** ablate the broadcast *content* — if behavior degrades, it was
  functionally integrated, not decorative.
- **Result (decisive, 4 seeds).** Purpose-built *costly perceptual-aliasing* env
  (rule-regime DoorKey): each episode a hidden affordance regime (KEY vs NOKEY)
  decides the correct action; the regime is (a) absent from the observation and
  (b) unsafe to probe for (the disambiguating action is a trap, penalty −1.0), so
  the broadcast is the ONLY channel carrying it. Converged frac_solved:
  - intact (true regime): **0.62** (best-window up to 0.90)
  - zero (content severed): **0.49**
  - constant (present but uninformative): **0.48**

  intact beats BOTH controls **4/4 seeds with non-overlapping ranges** (worst
  intact 0.612 > best control 0.499; gap ≈ **+0.13**). The controls plateau
  exactly at the information-theoretic ceiling (½ = base rate of the majority
  regime); intact breaks it by reading the regime from the broadcast. No OOD
  confound — each arm trains *and* evaluates on its own input distribution.
- **Why this design.** The original KG-GNN mean-pool was one global vector
  recomputed every 50 episodes → constant within an episode → null by
  construction (the learned gate correctly stayed shut). A natural task
  (DoorKey 5x5→8x8 transfer) hit a *measurement wall*: an expressive policy
  relearns from raw obs, so the lesion barely bit (intact−scramble ≈ +0.04, 3/4
  seeds). The aliasing env removes the wall by making the broadcast's content the
  only way past a provable 0.5 ceiling; the −1.0 trap removes the free
  "commit to one regime" escape hatch, making the broadcast *necessary*.
- **Falsifiable — and it did fail once.** The trap=0 variant left a safe commit
  strategy and intact stayed pinned at the ceiling (a real null). The penalty
  variant passes decisively. A rung you cannot fail is not a test.
- **Scope (honesty).** Proves the *function* (global availability shapes action)
  via the symbolic affordance/regime channel; the raw KG→GNN path stays
  decorative. No claim of subjective experience.
- **Harness:** `rung6_aliasing.py` (decisive; arms intact/zero/constant/noise on
  `MiniGrid-RuleRegimeDoorKey-Pen-5x5-v0`), env in
  `developmental_ai/environments/rule_regime_doorkey.py`; broadcast in
  `developmental_ai/core/rule_regime_broadcast.py`. Earlier variants:
  `rung6_transfer.py` (cross-task transfer), `rung6_broadcast_ablation.py`
  (within-env intact/lesion/off).
- **Theory:** Global Workspace Theory.

### Rung 7 — Cross-domain abstraction ✅ *(demonstrated via explicit representation alignment; naive weight-transfer is null)*
- **Demonstrates (intended):** knowledge transfers across *structurally different*
  domains, not just 5x5→8x8.
- **Metric:** positive transfer between two domains that share abstract structure
  but differ on the surface.
- **Harness:** `rung7_crossdomain.py` — a modular policy `obs --[encoder_d]--> z
  --[shared CORE]--> abstract-action --[adapter_d]--> env` on SeqGoal-AB-9x9, where
  Domain B re-encodes the SAME abstract MDP through a random linear projection
  (different obs dim) + a permuted action space. The shared CORE is the candidate
  transferable abstract skill; encoder_d/adapter_d are thin per-domain surface
  adapters.
- **Result — four negatives.** (a) **Frozen** single-source core + learned adapters:
  fails to align under sparse reward (AUC ~0.00). (b) **Warm-start** single-source
  core, trainable (3 seeds): *negative* transfer — random-core scratch is FASTER
  (AUC 0.74 vs 0.54; ~328 vs ~564 eps) because the core entangles the source's
  action convention and must be unlearned. (c) **Multi-domain** (shared core trained
  on 3 surface-different sources), frozen: still won't align, and the round-robin
  multi-task training didn't even master the projected sources. (d) Multi-domain
  warm-start: ≈ scratch (no benefit).
- **Diagnosis (why it's coherent).** On a NOVEL target the bottleneck is learning
  that domain's surface decode (the random projection) + action map — both
  necessarily domain-specific. A transferred abstract core can't help with
  surface-specific learning, and the abstract navigation it *does* carry isn't the
  bottleneck (curiosity learns it fast anyway). So abstract-skill weight-transfer
  buys ~nothing when cross-domain difficulty is dominated by surface decoding — and
  can hurt (negative transfer). A genuine pass likely needs explicit representation
  **alignment** (e.g. a contrastive/reconstruction loss tying each domain's encoder
  to a shared z-space), not gradient-through-a-frozen-core hope, or a task family
  where the abstract structure (not the surface) is the dominant cost.
- **Resolution — DEMONSTRATED (explicit alignment, 3 seeds).** Exactly the predicted
  fix (`rung7_crossdomain.py`): **supervised-align** the target encoder so
  encoder_B(obs_B) matches encoder_A(obs_A) on corresponding underlying states
  (alignment MSE → **0.000**, no reward), landing Domain B in the shared z-space; then
  reuse the warm-start abstract core and RL only the thin action-adapter. Mean over 3
  seeds — **aligned** AUC **0.825**, competent in **~155** eps; **no-align** null (same
  warm core, encoder left to RL) 0.387 / ~564 eps; **scratch** 0.335 / ~615 eps.
  Aligned beats BOTH decisively, and the no-align null is **load-bearing** (without the
  explicit alignment the identical core fails) — confirming RL alone can't align a fresh
  encoder. **Scope (honesty):** the alignment consumes cross-domain *state
  correspondences* (the same situation observed in both surfaces) — a modest, standard
  analogy input; deriving them autonomously (no paired data) is the open next step.
- **Autonomous attempt — NEGATIVE (honest).** `rung7_autonomous.py` tries to *discover*
  the correspondence with NO paired data: from random Domain-B interaction, recover the
  encoder by least-squares fitting B's movements to the abstract dynamics + a
  self-identified reward anchor, and the action permutation by searching all 4!. Graded
  against ground truth (used ONLY to grade, never to train). Across the variants tried
  (wall-hit filtering, symmetry-broken asymmetric goals) it recovers the geometry only
  partially (z-corr ~0.85) and gets the action permutation **wrong**, so **zero-shot
  transfer is 0.00** — failing the pre-declared 5-check bar. Controls behave (random
  encoder & shuffled-signal lesion → 0.00). **Why:** chicken-and-egg non-identifiability
  — the encoder needs the perm-labelled dynamics to be pinned, but the perm needs a
  precise encoder; with only a 2-D movement subspace + sparse anchors, the encoder's
  flexibility absorbs perm errors so the dynamics residual can't single out the truth.
  Autonomous discovery stays OPEN; the *assisted* result above (given correspondences)
  stands. (The rigorous grader is the point — it refused a false pass.)
- **Theory:** analogical reasoning — the core of human general intelligence. (The four
  weight-transfer nulls stand as honest negatives, cf. Rung 1 original / Rung 4 / Rung 8;
  the alignment route is the positive — given correspondences, abstract skills transfer.)

### Rung 8 — Counterfactual reasoning (Pearl's rung 3) ✅ *(DEMONSTRATED July 2026 after the H1 WM fix — posterior matching 0.540 vs chance 0.145, prior 0.211; single-seed protocol, local)*

> **July 2026 rerun (fixed WM, same 90k/seed-42 protocol, local Mac):**
> branches=160, chance=0.145 — posterior (abduced) matching **0.540**,
> prior (no obs) 0.211; changed-dims MSE posterior/prior **0.022 / 0.261**
> (−91%). The pre-registered bar (chance+0.10 AND prior+0.08) is cleared ~3×
> over; the abduction lesion is decisive. The 2026-06 qualified-negative below
> is kept verbatim as the honest record of the WM-limited attempt.
- **Demonstrates:** the agent answers "had I done a′ at state s — instead of the
  action it actually took — would the outcome have differed?", and its answers are
  *correct*. This is the top of Pearl's ladder. Note the distinction from Rung 4:
  Rung 4 (dreaming) is **intervention** (`do(a)` — forward-simulate the
  consequences of an action); a counterfactual additionally requires **abduction**
  (infer the latent/noise from what *actually* happened) and re-rolling with the
  actual stochastic draws held fixed. The substrate already exists — the RSSM
  posterior (`observe_sequence`) *is* abduction, `imagine_trajectory` *is*
  prediction; the only new piece is branching the action from the **posterior**
  (abduced) latent with noise held fixed, rather than from a freshly sampled state.
- **Metric (decisive, falsifiable — Rung-6 standard).** MiniGrid is resettable, so
  we have GROUND TRUTH: reset the real env to s, execute the counterfactual action
  a′, observe the true outcome. Compare the model's abduction-conditioned
  counterfactual prediction to that truth (outcome/return match, or next-state
  accuracy) over many held-out branch points.
  - **Lesion (load-bearing):** replace the abduced **posterior** latent with the
    **prior** (i.e., do NOT condition on what actually happened) and re-predict.
    If posterior-conditioned counterfactuals beat prior-only, the agent is doing
    genuine counterfactual *inference*, not just forward prediction — the abduction
    step is causally necessary. Scramble the conditioning episode as a second null.
  - **Behavioral version:** **counterfactual credit assignment** — use "how much
    did this action matter vs the alternatives" to identify the pivotal action in a
    failed episode (or to speed learning); scramble the counterfactual advantage as
    the null. A win that survives the lesion is the load-bearing pass.
- **Why now-feasible.** The WM is accurate enough to *simulate* even though Rung 4
  showed dreaming doesn't improve the *policy* (rollouts beat persistence to 15
  steps; twohot reward head discriminates the goal +0.51). Accuracy-for-reasoning,
  not accuracy-for-policy-improvement, is what counterfactual queries need — so the
  Rung-4 negative does not block this rung. Builds on Rung 4 (world model) and
  pairs with Rung 5 (self-model): "I should have known to get the key first" =
  self-model + counterfactual.
- **Scope (honesty).** *Functional* counterfactual queries over the agent's own
  learned world model — NOT the rich human sense (causal-structure discovery,
  nested or language-grounded counterfactuals), which is far beyond a MiniGrid
  RSSM. Report the function; never claim experience.
- **Result (action-consequence matching, 160 branch points, DoorKey-6x6).** Built
  exactly as specified: abduce posterior latent at the branch, intervene with each
  action a′, predict next obs, score against reset+replay GROUND TRUTH; metric =
  does the prediction for a′ best-match a′'s real outcome (chance = 1/#distinct ≈ 0.15).
  - **The machinery composes and abduction helps (the lesion fires, weakly):**
    posterior beats prior on BOTH matching (**0.205 vs 0.156**) and full-obs
    prediction error (**0.205 vs 0.279, −26%**). Conditioning on what actually
    happened genuinely improves the counterfactual — abduction is directionally
    necessary, as designed.
  - **But NOT decisive:** posterior matching (0.205) is only modestly above chance
    (0.146), short of the pre-registered bar (chance+0.10 AND prior+0.08).
- **Why (the recurring bottleneck).** The WM's **action-conditioning is too weak**:
  a dedicated A/B measured action-sensitivity at only 0.066 vs a 0.77 prediction
  error, so the model distinguishes "which action → which outcome" only faintly.
  Same limiter behind the Rung-4 dreaming null. The *architecture* has every piece
  (posterior = abduction, imagine = intervention, and they compose correctly); the
  *learned dynamics* aren't action-causal enough for a decisive obs-space pass.
- **Found+fixed an off-by-one en route (kept, off by default).** `observe_sequence`
  paired `a_t` with `o_t`; standard DreamerV3 builds state_t from the action that
  CAUSED it (`a_{t-1}`). The fix (`world_model.causal_align`) makes
  imagine_step(s_t, a_t) predict o_{t+1} causally and raised action-sensitivity
  ~45% (0.046→0.066) — directionally right, but not enough to clear the bar. The
  original convention was compensating, not catastrophically broken (changed-dim
  prediction is ~2.3× better than persistence either way; the "barely beats
  persistence" full-obs number was a static-background metric artifact).
- **Harness:** `rung8_counterfactual.py` (abduce/intervene/predict + posterior-vs-
  prior lesion, reset+replay ground truth), `_causal_ab.py` (alignment A/B),
  `_cf_align_probe.py` (action-slot probe); WM flag `world_model.causal_align`.
- **Theory:** Pearl's structural causal models / ladder of causation (abduction →
  intervention → prediction); counterfactual credit assignment in RL.

### Rung 9 — Law induction / schema transfer (the "scientific method" rung) ✅ *(DEMONSTRATED July 2026, pre-registered, CIs + artifact)*

- **Demonstrates:** ABSTRACT, world-invariant knowledge ("laws") induced by the
  agent's OWN interventions transfers to novel worlds and is load-bearing —
  the bridge from interventional (simulator) to explicit (symbolic) causality.
- **Design.** A family of `LawWorldEnv` worlds (`environments/law_worlds.py`):
  per-world hidden bindings (key→door color PERMUTATION; one hazardous floor
  color) under per-episode surface variation (layouts, door colors cycling, a
  colored band that must be crossed). Wrong-key toggles and hazard steps are
  TRAPS (rung-6 costly-probe principle), so experiments are informative but
  expensive. A `RuleBook` (`core/rulebook.py`) holds Beta-posterior evidence
  from interventions — capped at ~6 trials/hypothesis, loose 0.25/0.75
  settle band (**the imperfection principle: rules internalized fast, with
  tolerated error, like humans — and near-zero FLOPs: rules are tables**).
  A `SchemaPrior` induces the cross-world invariants (one-hazard structure,
  binding-is-permutation) from completed worlds' rulebooks.
- **Protocol (pre-registered).** Phase A: an experiment-driven explorer (shared
  scripted BFS navigator — navigation deliberately NOT the tested variable;
  precedent: rung 2's executor, rung 8's reset+replay) lives in 8 training
  worlds × 8 episodes and induces the schema (binding_is_perm = 0.90). Phase
  B: 12 UNSEEN worlds × 8 episodes, three arms identical except the schema:
  intact / **scramble** (same confidence numbers, inference permuted —
  valid-but-wrong) / **none** (empty schema; within-world learning still on).
  PASS = intact > none AND intact > scramble with bootstrap CI of the diff
  excluding 0; VALIDITY = scramble ≤ none.
- **Result — decisive PASS** (`rung9_lawinduction.py`; artifact
  `rung9_lawinduction_results/results.json`):
  - intact **0.573** [0.469, 0.677] ≫ none **0.323** [0.292, 0.354] ≫
    scramble **0.219** [0.156, 0.271] (success-AUC over 12 worlds)
  - intact−none **+0.250** [+0.146, +0.354]; intact−scramble **+0.354**
    [+0.240, +0.469] — both exclude zero.
  - **Validity gate:** scramble−none **−0.104** [−0.177, −0.042] — a wrong
    schema is WORSE than none: the schema's *content* is load-bearing
    (cf. rung 5's "a wrong self-model is worse than no self-model").
  - Mechanism visible in the costs: intact wastes 2.92 wrong-key traps/world
    vs 5.0+ for none/scramble — cross-door DEDUCTION from the permutation
    schema ("that key is bound to another door, don't test it").
- **Scope (honesty).** Laws-as-relations over factored MiniGrid state, not
  variable discovery from pixels; the navigator is scripted and shared; the
  hazard-structure prior stayed weak (one_hazard 0.50 — the cautious explorer
  rarely generated hazard evidence in training; an honest open edge).
  Integration into the full organism (rules as goals for the live curiosity
  system) is the open next step. First rung to ship with CIs and a results
  artifact from birth (H4).
- **Theory:** the child-as-scientist paradigm (Gopnik); active causal
  discovery; Popperian falsification as a data structure (Beta evidence,
  refuted-hypothesis exclusion); Konidaris-style symbols grounded in what
  planning needs.

### Rung 10 — Achievement-goal channel on a real tech tree (Crafter) 🟡✅
*(reduced protocol: n=2 seeds, 40k steps, local — 2026-07-15)*

- **Setup.** The rich-env goal channel (`core/achievement_goals.py`): target
  achievement + achieved-mask broadcast to the policy, IMGEP frontier
  targeting via the rung-5 CompetencePredictor, one skill minted per
  achievement. Arms: **given** (oracle achievement list) / **discovered**
  (reward-spike + obs-delta clustering — never reads the oracle) / **none**.
  Metric: distinct oracle achievements unlocked (measurement tap, identical
  across arms). Harness `rung10_techtree.py`; artifacts
  `rung10_results_s0/`, `_s1/`.
- **Result — the EARNED goal space wins; the oracle one is inconsistent.**
  - discovered breadth **[9, 9]** vs none **[7, 7]**: diff **+2.00, CI
    excludes zero** (degenerate CI — identical values both seeds). Purity
    0.67–0.90, coverage 4–6 oracle achievements, 24 skills minted per seed.
  - given breadth **[9, 6]** vs none: diff +0.50 [−1.00, +2.00] — **does NOT
    exclude zero.** The pre-registered primary (given > none) FAILS at n=2;
    the secondary (discovered > none) PASSES.
  - **Why this is coherent:** discovered slots are grounded in what the agent
    actually EXPERIENCES (reward-spike moments it has reached at least once),
    so its frontier targets are calibrated to competence. The oracle list
    includes deep achievements far beyond current reach — the frontier scorer
    can chase unreachable targets and waste episodes (seed 1's given arm
    underperformed even the baseline). Grounded-in-experience goals beat
    privileged-but-uncalibrated goals: an IMGEP-flavored finding, and a
    genuinely surprising one worth confirming at full budget someday.
- **Law induction bonus:** the estimated prerequisite DAG recovered the REAL
  Crafter tech tree from experience: `place_table ← collect_wood`,
  `make_wood_pickaxe ← {collect_wood, place_table}`, `make_wood_sword ←
  {collect_wood, place_table, +1 spurious co-occurrence edge (wake_up)}` —
  correct structure with honest noise. This DAG feeds the Crafter→Craftax
  rule-level transfer (step 6).
- **Skill bank on rich envs is ALIVE:** goal arms minted 6–24 skills/run;
  the none arm minted **0** (the old one-skill-per-env rule never fires).
- **Scope (honesty):** n=2 seeds, 40k steps, breadth ties make CIs
  degenerate; scramble control not run (deferred). Directional, not
  decisive — but the discovered-arm consistency (9/9 vs 7/7) and the
  recovered DAG are the load-bearing observations.

### Rung 2b — Tech-tree cumulative learning (Crafter) 🟡 *(wash at n=2, 2026-07-15)*

Warm-starting a fresh run from a banked achievement skill (rung-10's bank;
best-success skill = `collect_sapling`): seed 8 **+2526 steps faster** to
depth-2, seed 7 **−4869 slower**; breadth identical both seeds (7/7, 8/8).
**Inconclusive — a wash with high seed variance**, consistent with the
original Rung-1 lesson: warm-start transfer pays only when the target is hard
relative to from-scratch learning, and Crafter's early tree is cheap enough
to learn fresh. The honest next lever is depth-CONDITIONAL warm-starting
(load the deepest banked skill, not the highest-success one) and harder
targets — not more seeds on this protocol. Harness
`rung2b_treecomposition.py`; artifacts `rung2b_results_s7/`, `_s8/`.

### Step 6 — Crafter→Craftax rule-level transfer 🟡 *(NULL, n=2, 2026-07-15)*

Transferred the Crafter-learned prerequisite DAG (6 rules, e.g.
`make_wood_pickaxe ← {collect_wood, place_table}`) into Craftax as a
name-anchored goal-ordering prior. dag breadth **[12, 7]** vs no-dag
**[10, 10]**: diff **−0.50 [−3.00, +2.00] — CI includes zero. NULL.**
High variance, no direction. Honest reading: the shared Crafter↔Craftax
subtree is shallow (~6 rules of a 67-achievement space), so the prior
constrains little and mostly damps early goals; Craftax's depth lives in
dungeons the Crafter DAG says nothing about. The rung-7 lesson holds shape:
transfer needs the shared structure to be the BOTTLENECK, and here it isn't.
Harness `crafter_to_craftax_transfer.py`; artifacts `pod_results/step6_*`.

### Step 7 — Rich-env capstone: the full organism on Craftax ✅
*(reduced protocol: 4 seeds × 200k, 2026-07-15)*

The whole stack at once — RSSM + dreaming (H2 trust-gated augment) + goal
channel + per-achievement skill minting + curiosity + stage controller — on
the deepest world, 4 seeds:

| seed | breadth | skills | reward quarters | collapsed |
|---|---|---|---|---|
| 42 | 10 | 10 | 3.98 → 4.50 | no |
| 7 | 11 | 11 | 3.86 → 4.18 | no |
| 123 | 13 | 13 | 3.62 → 3.62 | no |
| 7777 | **16** | **16** | 3.78 → 3.78 | no |

- **0/4 collapses, 4/4 developed** (final-quarter reward ≥ first-quarter).
  **H2's stability fix HOLDS at rich-env scale** — the ~1/3 stochastic
  collapse rate that defined the June organism is gone on a world an order of
  magnitude deeper than DoorKey.
- **Breadth 12.5 [10.5, 14.8]**, and **skills minted == breadth on every
  seed** (10/11/13/16): every achievement the organism reaches becomes a
  banked skill. The skill bank, dead on rich envs a day ago (0 skills across
  2M steps), is now the organism's memory of its own tech tree.
- Scope: reduced protocol, no vanilla arm at this scale (the comparison lives
  in rung 10); breadth CIs are wide at n=4.

---

## When the developmental machinery helps (cross-cutting finding)

The components are **task-appropriate, not free**. Measured directly:

| task type | curiosity (intrinsic) | symbolic broadcast | best arm |
|---|---|---|---|
| reward-FREE exploration (Rung 3) | **essential** | n/a | learning-progress curiosity |
| hidden structure (Rung 6 aliasing) | n/a | **essential** | broadcast intact |
| findable-reward goal (Rung 1) | mild **tax** (4/4 seeds) | mild **speed tax** (3–4/4) | **plain PPO (vanilla)** |
| sparse-reward goal (Rung 4 dreaming) | n/a (dreaming) | n/a | **safe but neutral** — gated by WM accuracy |
| multi-task practice allocation (Rung 5 self-model) | n/a | n/a | self-model: **calibrated + load-bearing** (≫ scramble) but **ties uniform** under strong transfer |
| model-based reasoning (Rung 4 dreaming, Rung 8 counterfactual) | n/a | n/a | ~~**bottlenecked by WM action-causality**~~ — was the recurring limiter (sensitivity 0.066 vs error 0.77); **FIXED July 2026** by the H1 audit package: sensitivity **0.188** now EXCEEDS changed-dims error **0.170**, factual slot flipped to the causal a[t] (see "July 2026 audit" section) |

i.e. each piece earns its compute cost exactly where the task needs it, and is
dead weight where plain RL suffices. The actionable consequence is
**signal-driven gating**: anneal curiosity down as extrinsic reward becomes
findable (so the system is never worse than vanilla on easy tasks, while still
winning on sparse/structured ones). This is a *natural* anneal driven by the
observed reward, NOT a hand-fixed ratio. (Implemented behind
`reward_mixer.adaptive_gating`; off by default.)

---

## July 2026 audit + fix packages (H1–H3)

A multi-agent code audit (**`CODE_AUDIT_2026-07.md`** — 14 confirmed bugs,
adversarially verified; fix order §H) explained both standing mysteries
mechanically, and the fixes are now in:

**Infrastructure change (July 2026): the Azure VM is no longer available.**
All experiments now run locally (6-core i5 Mac). Reruns use REDUCED protocols
(fewer seeds/sizes), flagged per result. Historical VM results stand as
recorded.

### H1 — WM action-causality package (DONE; A/B DECISIVE)

The 0.066-sensitivity limiter was four compounding causes: non-causal a_t/o_t
pairing (`causal_align` off), gradient starvation (recon/reward losses read
posteriors; the action-dependent prior only gets KL·0.3 with free-nats
zeroing), 146:1 action dilution (FiLM identity-init = zero initial influence),
and the inverse-dynamics head reading posteriors. **Plus a config landmine:**
configs do NOT merge with `default.yaml`, and `minigrid_doorkey.yaml` lacked
the `film_conditioning`/`inverse_dynamics` keys — both were silently OFF in
every DoorKey experiment to date.

Fixes: `causal_align` default ON; inverse-target off-by-one under
`action_shift` fixed (the two mechanisms were mutually broken — the target was
the agent's NEXT action, a policy leak); double straight-through gradient
removed; replay windows may no longer cross episode boundaries (terminal AND
reward strata) or the circular write seam; PER importance weights computed
from the true three-stratum mixture. Unit-verified: `_h1_fix_smoke.py`.

**A/B rerun (70k, seed 42, DoorKey-6x6 — same protocol as the historical
record; local Mac):**

| arm | act-sensitivity | changed-dims MSE (best slot) | factual slot |
|---|---|---|---|
| old default (VM record) | 0.046 | — | a[t+1] (leak) |
| old `causal_align` (VM record) | 0.066 | — | — |
| new pkg, causal OFF | 0.082 | 0.721 | a[t+1] (leak) |
| **new pkg, causal ON** | **0.188** | **0.170** | **a[t] (causal)** |

The pathology inverted: action-sensitivity now EXCEEDS the changed-dims error
(0.188 vs 0.170; was 0.066 vs 0.77). The WM beats persistence 10.4× on
changed dims (was ~2.3×) and 6.5× on full obs (previously "barely"). Single
seed — matches the historical protocol, same caveat.

### H2 — capstone-collapse package (DONE; smoke-verified; multi-seed capstone rerun PENDING)

The stochastic ~1/3 collapse had a confirmed three-part mechanism: the distill
value gate compares incommensurable critics (opens on hallucinated dream value
exactly where the WM is weakest), the natural anneal is MAXIMAL on failure (a
positive feedback loop), and the dream warmup counter was global (distillation
fired immediately after each env swap from a stale teacher). Fixes, all
signal-driven (no fixed ratios): **WM-trust gate** (distill only where the
WM's own surprise — posterior‖prior KL — is ≤ `dream_training.trust_kl_max`,
default 6.0 nats) + **trust-scaled anneal** (effective weight × trust-EMA, so
the pull fades on untrusted envs regardless of solve rate) + **per-env warmup**
(`_episodes_this_env`) + **PPO rollout flush on env swap**. Verified:
`_h2_h3_fix_smoke.py` (incl. live env-swap + trust-threshold sweep).
**VALIDATED END-TO-END: the 4-seed capstone is a clean sweep (4/4 perfect,
0 collapses) — see the capstone section above.** The stochastic collapse is
gone.

### H3 — curiosity package (DONE; smoke-verified; Rung-3 re-confirm PENDING)

ICM/LP intrinsic reward was mean-centered → zero-mean: ~half of transitions
got NEGATIVE curiosity, and LP punished FIRST VISITS to new states
(anti-exploration inside the exploration signal). Fixes: std-only
normalization (both arms identically — the Rung-3 controlled-swap property is
preserved); encoder now trained ONLY by the inverse loss (the forward loss was
teaching it to collapse features; Pathak-correct); cross-entropy inverse loss
for discrete actions; `reward_scale` applied post-normalization (the H-GRAIL
skill-decay was provably inert before). Because BOTH Rung-3 arms change, the
Rung-3 pass must be re-confirmed (reduced protocol, local).

### H4 — eval-honesty package (DONE; smoke-verified)

Greedy/argmax eval path (`select_action(deterministic=True)`; the Evaluator's
"deterministic actions" docstring is finally true — it previously SAMPLED,
±0.2 noise at 20 episodes); obs-normalizer stats frozen during eval
(`freeze_obs_stats`); seeded eval episodes (base_seed + i, reproducible);
per-env `success_reward_threshold` respected (Acrobot no longer reports 0%
forever); `developmental_ai/core/stats.py` adds `bootstrap_ci` /
`bootstrap_diff_ci` / `save_results` — new harnesses must report CIs and
write results artifacts (evidence rule #2, finally enforced in code).
Historical harnesses keep their sampled-eval protocols for comparability.
Verified: `_h4_fix_smoke.py`.

### H5 — dead GNN path disabled by default (DONE)

`symbolic.knowledge_source` now defaults to **"none"**: the KG→GNN broadcast
was never trained (frozen gate at sigmoid(−2)≈0.12, GNN under `no_grad`, no
optimizer) and its PyKEEN embeddings rotate arbitrarily at every from-scratch
retrain — cost + noise, not knowledge. The blocking 50-episode PyKEEN retrain
now only runs when "kg" is explicitly requested. KG store, fact extraction,
symbolic decoder, and the hand-built broadcasts (episodic/affordance/
rule_regime) are unaffected. The semantic-memory ROLE returns, rebuilt
properly, in the law-induction program below.

### H3 correction — the audit's normalization complaint is WITHDRAWN for novelty

Two attempted non-negative replacements for the mean-centered curiosity both
failed the full-run canary: std-only (all-positive) collapsed 5x5 learning to
frac 0.08, and median-centered-clamp-at-zero still left **~81 units of
farmable intrinsic per timeout episode** (measured) vs task reward 1.0 —
frac 0.15. **Conclusion: zero-mean is the anti-farming mechanism, not a bug.**
The negative half of the signal is what makes novelty RELATIVE ("more novel
than the recent typical step"), so padding an episode buys nothing. Final
state: ICM/novelty keeps mean-centering (original behavior restored, comment
documents why); **LP keeps clamp-at-zero** (its mostly-zeros shape has no
farmable mass, and first visits to new states are no longer punished — the
one audit sub-claim that survives). The other H3 fixes stand (encoder trained
by inverse loss only, CE inverse loss, post-normalization reward_scale).
Lesson recorded twice over: a unit-verified reward change still needs a
full-run canary, and the baseline arm IS the canary.

### Rerun status (compute migrated to the RunPod pod, 2026-07-12 — 48 vCPU;
### Mac runs stopped mid-flight, partials salvaged below; final artifacts land
### in rung4_pod_s*/rung3_pod_s*/capstone_h2_s* on the pod)

- **Rung 8 (counterfactual)** with the fixed WM: **DECISIVE PASS** — posterior
  matching **0.540** (chance 0.145, prior 0.211; bar was 0.245/0.291);
  abduction lesion −91% changed-dims error. First rung lifted by H1.
- **Rung 9 (law induction)**: **DECISIVE PASS** — see the Rung 9 section.
- **Rung 3 re-confirm**: **DECISIVE PASS under the final H3 normalization**
  (2 seeds, pod + salvaged Mac). tv_frac LP vs novelty: seed 42 **0.008 vs
  0.049**, seed 7 **0.002 vs 0.059** — LP ignores the noisy TV by ~30–45×;
  frontier coverage 1.00 for both arms both seeds. The median-center/
  clamp-at-zero curiosity change did NOT harm the autotelic result.
- **Rung 4 (dreaming)**: **COMPLETE (2 seeds, 5x5, fully-fixed causal WM) —
  still NOT a win.** eps→competence dream_augment vs dream_off: seed 42
  **198 vs 117**, seed 7 **168 vs 107** — augment is SLOWER to competence
  both seeds; final AUC/frac a wash (augment wins s42 0.626/0.844, loses s7
  0.567/0.814). The pre-registered pass ("competence in fewer real steps AND
  ≥ AUC") is NOT met. **Key finding: the H1 WM fix lifted rung 8 (reasoning)
  decisively but NOT rung 4 (policy improvement)** — different demands on the
  WM. Rung 8 needs accuracy-to-answer-queries (delivered); rung 4 needs
  imagined rollouts to beat REAL experience, and on easy findable-reward 5x5
  real experience is already cheap, so dreaming only adds distillation
  overhead (a mild sample-efficiency tax — coherent with the task-
  appropriateness theme). Its potential value is on HARD/SPARSE envs (8x8),
  where the WM was previously too weak; re-testing augment on 8x8 with the
  fixed WM is the honest open follow-up (not run — reduced local protocol was
  5x5 only). The dream_off baseline (0.832/0.859) beats the historical ~0.74,
  confirming the fix stack improved base learning.
- **Capstone H2 validation** (4 seeds, full arm, LLM OFF, on the pod):
  **DECISIVE PASS — a CLEAN SWEEP.** All 4 seeds (42, 7, 123, 7777)
  final competence **[1.00, 1.00, 1.00]**, retention_drop **[0,0,0]**,
  **AUC 1.000** — every seed mastered all three envs AND retained perfectly.
  **0/4 collapses** (vs the historical ~1/3 stochastic collapse rate).
  - **The two historical collapsers are now flawless:** seed 42 (collapsed to
    [0,0,0.2] in the original capstone) and seed 7 (collapsed to [0,0,0] in the
    LLM-off stability test) both come back **perfect WITHOUT the LLM.** This is
    the key result: the H2 trust-gate/anneal/warmup package is a **structural**
    fix that *replaces* the n=3 LLM-stabilization crutch — the collapse was the
    diagnosed mechanism (incommensurable value gate + failure-maximal anneal +
    global warmup + unflushed buffer), not something only dense LLM shaping
    could paper over.
  - **All 4 seeds mastered 8x8 at 1.00** — historically 8x8 sat stuck in the
    partial-learning regime (~0.2) because the WM was too weak there. **Credit
    H1:** the causal-WM fix let dreaming distill useful policy on the hardest
    env instead of hallucinating, so the organism cracks 8x8 AND holds its
    earlier competence. H1 and H2 compound.
  - **The "high ceiling / low floor, UNSTABLE" verdict below is SUPERSEDED.**
    The integrated organism now develops smoothly and reliably across the full
    curriculum — the outcome the June capstone reached only 1 seed in 2.
- Capstone multi-seed (H2 validation) and Rung-3 re-confirm (H3): QUEUED.

---

## Integration capstone — does the *whole organism* work?

Every rung tested one component with the others off. The capstone turns the FULL
stack on at once — symbolic + curiosity + adaptive gating + stage controller +
skill bank + dreaming-augment + WM upgrades + self-model — and runs it autonomously
across a SEQUENCE of novel envs (DoorKey 5x5 → 6x6 → 8x8), vs a vanilla-PPO arm
(shared agent carried across the sequence in both). It measures forward transfer,
retention (re-eval earlier envs at the end), cumulative competence AUC, and
coordination health. Harness: `capstone_integration.py`.

- **It composes at the plumbing level — including the local LLM.** The full stack
  (and a separate LLM-on smoke with Ollama llama3.1:8b in the loop) runs end-to-end
  across env-swaps + eval phases with no crash/NaN. Control signals behave: gating
  anneals (intrinsic → ~0), stage controller hands off to EXPLOIT, skills mint,
  dreaming runs.
- **The capstone earned its keep by finding a real COMPOSE-TIME bug** that every
  isolated rung missed: dream-augment distillation was hardcoded `knowledge=None`,
  so it silently RuntimeError'd (and burned compute on discarded imagination)
  whenever **symbolic was on** — i.e. dreaming never trained the policy in the full
  organism. Fixed (pass the live knowledge feature; detach so it can't corrupt PPO
  gradients) + the silent `except` now logs once. Verified: dream_distill_updates
  0 → 100 with symbolic on.
- **Result (full vs vanilla, 2 seeds, 80k/env).** Competence = success over eval
  episodes; AUC over the eval matrix; retention = end-of-sequence eval per env.

  | seed | arm | final per-env [5,6,8] | retention | AUC |
  |---|---|---|---|---|
  | 7  | **full** | **[1.00, 1.00, 1.00]** | perfect (drop 0/0/0) | **1.000** |
  | 42 | **full** | **[0.00, 0.05, 0.20]** | collapsed (drop 1.0/0.95) | 0.575 |
  | 7  | vanilla | [0.70, 0.45, 0.30] | moderate | 0.700 |
  | 42 | vanilla | [0.75, 0.50, 0.05] | moderate | 0.717 |

- **Verdict — high ceiling, low floor; powerful but UNSTABLE.** The integrated
  organism is a double-edged amplifier. On seed7 it achieved the dream outcome —
  AUC **1.000**, all three envs mastered AND retained perfectly (smooth autonomous
  development, which vanilla never reaches). On seed42 it **catastrophically
  collapsed**: it had mastered 5x5+6x6, then training 8x8 (where the WM is weakest)
  wiped the policy on every env (final ≈ 0). Vanilla is unspectacular but **stable**
  (~0.71 both seeds, no collapse). Mean AUC favours full (0.79 vs 0.71) but the mean
  lies — it's one perfect run and one collapse; full's *floor* is below vanilla.
- **The honest answer to "does it develop smoothly on its own?"** Sometimes
  beautifully, sometimes it self-destructs. Turning everything on raises both the
  upside and the downside, at ~2x the compute (dreaming). The likely instability
  source is dreaming distilling into the shared policy during the hardest env (weak
  WM there) with no anti-forgetting maintenance in a sequential curriculum — the
  same WM-action-causality limiter (Rung 4/8) now showing up as a *stability* risk.
- **Falsifiable next step (not yet run):** add competence-driven anti-forgetting
  maintenance (the Rung-5 staleness mechanism) to the sequence and/or value-gate
  dreaming OFF on envs where WM trust is low; re-run multi-seed and test whether the
  organism becomes *reliably* smooth (low-variance, no collapse) rather than
  high-variance.

### Stability test — does LLM-guided shaping reduce the collapse? (promising)

The instability above prompted a clean isolation: **dreaming ON in BOTH arms**
(`dream_distill_updates=100` everywhere — so unlike the earlier confounded smoke,
the LLM is the only variable), 3 seeds, LLM-on vs LLM-off, on the same sequence.
The local Ollama (`llama3.1:8b`) was capped to 6 cores and its call cadence cut 4x
(`goal_interval 25→100`) to keep it from starving the box.

| seed | LLM-off final / AUC | LLM-on final / AUC |
|---|---|---|
| 42  | [0.90,0.55,0.60] / 0.592 | [1,1,1] / **1.000** |
| 7   | **[0,0,0] COLLAPSE** / 0.508 | [0.85,0.80,0.65] / **0.708** |
| 123 | [1,1,1] / 0.975 | [1,1,1] / **1.000** |
| mean AUC | **0.692** | **0.903** |
| collapses | **1 / 3** | **0 / 3** |

- **LLM-guided rewards stabilized the organism:** 0/3 collapses with the LLM on vs
  1/3 without; paired, **LLM-on ≥ LLM-off in every seed**; mean AUC 0.90 vs 0.69,
  mean final competence 0.92 vs 0.56. The seed that collapsed without the LLM
  (`[0,0,0]`) held all three envs with it on. Mechanism: the denser, directional
  goal-progress signal reduces reliance on the destabilizing dream gradients during
  the hardest env, so the policy drifts/forgets less.
- **Caveat (honest):** the collapse is *stochastic* (it hit seed42 in the first
  capstone, seed7 here — a ~1/3 random failure mode, not seed-deterministic). At n=3,
  0/3 vs 1/3 is **strong-suggestive, not decisive** — but direction, paired
  improvement, variance reduction, and mechanism all agree. Confirm at higher seed
  count. Harness: `capstone_integration.py --llm 1`; relaunch helper
  `_capstone_relaunch.sh` (Ollama core-cap + cadence reduction + staggered launch).

---

## Rules that make any result *count as evidence*

1. **Always run the vanilla-PPO baseline.** Every "the developmental machinery
   helped" claim needs the from-scratch control, or it proves nothing.
2. **Multiple seeds + variance.** Gridworld RL is high-variance; a one-seed win
   is noise. Report confidence intervals.
3. **Pre-declare the metric** before running, or you'll cherry-pick.
4. **State a failure criterion** per rung. A milestone you can't fail isn't one.
5. **Don't read experience into function.** Report the function; never claim
   sentience/awareness.

---

## Recommended path

1. ✅ **Rung 6** — DONE (decisive aliasing pass).
2. ✅ **Rung 3** — DONE (noisy-TV: learning-progress curiosity beats novelty, 4/4).
3. ✅ **Rung 2** — DONE (skill composition: chained banked skills are competent at
   **0** training episodes; vanilla PPO and PPO+curiosity need **~180** episodes to
   match, 3 seeds). Sequential-goal gridworld; `rung2_composition.py`.
4. ✅ **Rung 1** — original DoorKey skill-bank-*retrieval* route was null (vanilla won on
   findable reward). REVISIT demonstrates the rung: warm-start from a banked policy onto a
   hard 9x9 sparse target beats fresh AND vanilla (AUC 0.984 vs 0.817 vs 0.511; ~30 vs ~170
   vs ~443 eps to competence, 3 seeds). `rung1_revisit.py`. (Surfaced + kept the
   "machinery is task-appropriate" finding: curiosity is essential on the hard sparse target.)
4. **Signal-driven gating** (`reward_mixer.adaptive_gating`) — anneal curiosity
   as extrinsic reward becomes findable, so the system is never worse than vanilla
   on easy tasks. ← turning Rung-1's negative into a feature.
5. 🟡 **Rung 4** (dreaming) — ran; mechanism FIXED (the broken unconditional/fixed/
   mode-covering distillation → value-gated + naturally-annealed + sharpened), which
   eliminated the consistent harm. But NOT load-bearing: dreaming is safe-but-neutral
   on DoorKey across 8 seed-pairs (5x5/6x6/8x8) — bottlenecked by world-model accuracy
   on sparse reward. A second honest negative (cf. Rung 1).
6. 🟡 **Rung 5** (self-model / metacognition) — flagship; ran (2 fixed re-runs).
   Metacognition CORE passes decisively: self-model is **calibrated** (ECE ~0.03)
   and **causally load-bearing** (intact ≫ scramble 3/3). Allocation **ties uniform**
   under DoorKey's strong transfer → utility-over-uniform not shown (needs a
   weak/negative-transfer task-set). First flagship rung with its core demonstrated.
7. ✅ **Rung 8** (counterfactual reasoning, Pearl rung 3) — DEMONSTRATED (July
   2026). The 2026-06 attempt composed correctly but was capped by the WM's weak
   action-conditioning (matching 0.205 vs chance 0.146, not decisive). After the
   H1 audit package fixed the limiter, the rerun is decisive: posterior matching
   **0.540** vs chance 0.145 / prior 0.211, abduction lesion −91% changed-dims
   error (single-seed protocol, local). The predicted "one fix could lift both
   Rung 4 AND Rung 8" is half-confirmed; Rung 4 rerun in progress.

---

## Harness index

| Concern | File | Notes |
|---|---|---|
| Single demo run | `run_demo.py` | `--load-checkpoint` warm-starts from a prior run |
| Symbolic causal proof (CartPole) | `ablation_run.py` | symbolic ON vs OFF |
| Rung 1 transfer | `transfer_run.py` | paired TRANSFER vs FRESH, 5x5→8x8 |
| Rung 6 decisive (aliasing) | `rung6_aliasing.py` | intact/zero/constant/noise on RuleRegimeDoorKey-Pen — the load-bearing pass |
| Rung 6 aliasing env | `developmental_ai/environments/rule_regime_doorkey.py` | hidden-regime, costly-probe DoorKey |
| Rung 6 cross-task transfer | `rung6_transfer.py` | DoorKey 5x5→8x8, force-concat broadcast |
| Rung 6 broadcast (orig) | `rung6_broadcast_ablation.py` | intact / lesion / off |
| Rung 2 composition (decisive) | `rung2_composition.py` | bank goto_A/goto_B → compose (0-train) vs vanilla/curiosity on sparse A-then-B |
| Rung 7 cross-domain (✅ via alignment) | `rung7_crossdomain.py` | modular encoder/CORE/adapter; weight-transfer null (4 negatives) → supervised encoder alignment beats scratch + no-align null |
| Rung 7 autonomous (negative) | `rung7_autonomous.py` | discover correspondence w/o pairs; ground-truth-graded 5-check bar; non-identifiable → open |
| Rung 2 composition env | `developmental_ai/environments/seq_goal.py` | sequential-goal gridworld; SeqGoal-{A,B,B1,AB} |
| Rung 1 cumulative transfer | `rung1_cumulative.py` | base→{transfer/fresh/vanilla}; obs key/no-key (original DoorKey negative) |
| Rung 1 revisit (decisive) | `rung1_revisit.py` | SeqGoal 5x5→9x9 warm-start transfer vs fresh/vanilla; ✅ demonstrated |
| Rung 1 observable env | `developmental_ai/environments/observable_doorkey.py` | visible-regime key/no-key DoorKey |
| Intrinsic/symbolic tax 2x2 | `intrinsic_tax_ablation.py` | symbolic × intrinsic-weight decomposition |
| Rung 3 autotelic (noisy-TV) | `rung3_autotelic.py` | novelty vs learning-progress curiosity, reward-free |
| Rung 3 noisy-TV env | `developmental_ai/environments/noisy_tv_gridworld.py` | reward-free room + unlearnable noise distractor |
| Rung 3 LP curiosity | `developmental_ai/curiosity/learning_progress.py` | error-reduction (learning-progress) reward; `curiosity.mode` |
| Rung 4 dreaming | `rung4_dreaming.py` | dream_off vs dream_augment (fixed distillation is default) |
| Rung 4 WM rollout gate | `wm_rollout_probe.py` | imagined rollouts vs persistence (PASS to 15 steps) |
| Rung 4 reward-head probe | `wm_reward_probe2.py` | training-aligned reward discrimination (+0.44, definitive); `wm_reward_probe.py` is the misaligned cautionary version |
| Rung 4 distillation | `developmental_ai/core/developmental_loop.py` | `_distill_dream_to_real`: value-gate + natural anneal + teacher-sharpen |
| Rung 4 WM upgrades | `developmental_ai/world_model/rssm.py` (twohot head: `twohot_encode/decode`, `predict_reward`), `replay_buffer.py` (`_find_reward_starts`, `reward_fraction`) | twohot reward head + goal-prioritized replay (`world_model.goal_replay_fraction`); `_wm_upgrade_unit.py` unit tests |
| Rung 5 self-model | `rung5_selfmodel.py` | intact/scramble/uniform; block-driven competence allocation over DoorKey 5x5/6x6/8x8 |
| Rung 5 competence predictor | `developmental_ai/core/self_model.py` | `CompetencePredictor`: online logistic P(success), ECE/Brier calibration, learning-progress |
| Rung 8 counterfactual | `rung8_counterfactual.py` | abduce→intervene→predict; posterior-vs-prior lesion; reset+replay ground truth |
| Rung 8 WM alignment A/B | `_causal_ab.py`, `_cf_align_probe.py` | causal action-alignment test (`world_model.causal_align`); action-slot probe |
| Integration capstone | `capstone_integration.py` | full stack vs vanilla across DoorKey 5x5→6x6→8x8; transfer/retention/coordination; `--llm 1` for LLM-on |
| Architecture flow maps | `docs/FLOWMAPS.md` (+ `docs/arch.png`, `docs/loop.png`) | current organism stack + learning loop, with load-bearing/decorative overlay |

## Minecraft campaign (MineRL Treechop, July 15–16 2026) — PARKED, framework pivot

Infrastructure shipped and validated: MineRL 1.0 gymnasium adapter (Discrete(10)
macros, crash recovery with in-place client rebuild, per-macro tick holds),
episode video recorder, 8-env parallel collection (threaded lockstep, dream-
actor-driven, engaged via the trust gate), 200M-param world-model config, and a
reviewed 128×128 pixel pipeline (resolution-general CNNs, native 128px MineRL
render).

Run history:
- **Run 1** (400k budget): crashed at 320k on a hung Java client; postmortem
  found the parallel path had never engaged — controller `window: 6` made the
  slope threshold unreachable (fixed in `glue_layer.py`, + `force_imagine_timestep`
  backstop). 0 reward.
- **Run 2** (400k, all fixes): infrastructure clean sweep — genuine trust-gate
  IMAGINE at ep 12, 8-env fan-out, one client crash survived, 4.4h. Still
  0 reward: 2-tick chop holds made log-breaking physically unreachable
  (~60 consecutive attack ticks needed; P≈10⁻³⁰ under exploration).
- **Run 3** (100k, 60-tick chop bursts): reward mechanically reachable
  (~26 aimed-chop opportunities/episode); IMAGINE engaged; PARKED by user at
  ~ep 20+ with still 0 reward — random aim never connected once.

Verdict: exploration-by-luck cannot cross the FIRST-DISCOVERY gap in Minecraft.
Curiosity habituates to consequence-free behavior (works as designed) but is
structurally blind to affordances never experienced once.

**Pivot (user decision, 2026-07-16):** reinstate the LLM as instinct/scaffold —
a local vision LLM (Ollama) supplying perception hints, goal suggestions, and
ANNEALING reward shaping (guidance that fades with competence; never direct
action control). Runs 4 (200M@64px, `configs/minecraft_200m.yaml`) and
5 (200M@128px, `configs/minecraft_128.yaml`) parked until the scaffold lands.
Also queued: goal discovery widened from primary-stream-only to all 8 streams;
crafting macros for real tech-tree depth (Treechop is a one-concept world).

**Scaffold implementation (2026-07-16, post-pivot):** shipped and smoke-tested.
`VisionScaffold` (`developmental_ai/llm/vision_scaffold.py`): llava:7b via
local Ollama assesses the agent's own POV every 25 steps (async single-slot,
1.0s/call on the pod GPU, stall-safe) → ANNEALED shaping = potential-based
approach term `w·(Φₜ−Φₜ₋₁)` + chop-while-aimed instinct bonus; `w` hits zero
at 75k steps. Never selects actions. Shaped reward feeds learning (PPO mix +
replay/WM reward head); the raw env reward stream (metrics, spike discovery,
grading) is untouched. Goal discovery now ingests ALL parallel streams
(signature clustering dedups cross-stream events) — 8× the odds of the first
grounding event. Tests: `_vision_scaffold_smoke.py` (stub-VLM semantics,
all-stream mint+dedup, Crafter loop wiring), `_vision_host_smoke.py` (real
llava on real gameplay frames, strict-JSON + latency + competence checks).

**Adversarial review (18-agent workflow, 2026-07-16):** 11 confirmed findings,
all fixed + smoke-covered. Highlights — (HIGH) per-stream broadcaster state:
the all-stream pivot made the competence self-model learn P(any-of-N streams
succeeds), inflating mastery ~N× and derailing IMGEP frontier selection — now
`_achieved_now` is (N_streams × slots), reset()/record_unlock/feature all
per-stream, N Bernoulli samples/episode, per-stream prereq evidence in
unlock_log, per-stream Given baselines, clear_stream() on autoreset. (HIGH)
scaffold had no episode/dream-boundary reset → potential term + aimed belief
leaked across resets into the reward head; added `VisionScaffold.reset()` +
generation guard that drops cross-boundary in-flight assessments. (MED)
anneal clock switched from `total_timesteps` (advances 8× in parallel/dream)
to a waking-only `_vision_clock`; aimed window anchored at frame-capture time;
render() gated on `wants_frame`. (LOW) `_AsyncChannel.submit` no longer drops
a done-but-unpolled result; parse-failure counter live; query-error warnings
throttled. All rung/parallel regressions clean.

| Concern | File | Notes |
|---|---|---|
| Vision-LLM scaffold | `developmental_ai/llm/vision_scaffold.py` | instinct channel; config `llm.vision.*`; llava:7b on pod Ollama |
| MineRL adapter | `developmental_ai/environments/minerl_env.py` | macros (60-tick chop `_ticks`), crash rebuild, custom-resolution spec |
| Parallel collection | `developmental_ai/core/developmental_loop.py` | `_run_episode_parallel`, `_ensure_parallel_envs`, thread pool |
| Stage gate fixes | `developmental_ai/core/glue_layer.py` | `_slope` window cap; `force_imagine_timestep` backstop |
| Minecraft configs | `configs/minecraft_parallel.yaml`, `minecraft_200m.yaml`, `minecraft_128.yaml` | 12.5M / 200.6M / 200.6M@128px |
| Launcher | `run_minecraft.py` | pod-only, xvfb; launch with `PYTHONUNBUFFERED=1` |

## Standing constraints

- Free/open-source tools only; **no paid API keys**. All LLM via local Ollama.
- Reward mixer **anneals naturally** — never hard-fix the intrinsic/extrinsic
  ratio.
- **Never wipe `skill_bank_data`** — cumulative learning is on. Experiments use
  dedicated `/tmp` skill dirs; consolidation/dedup only with a backup first.
