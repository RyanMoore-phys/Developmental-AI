# improvement plan

**SkyBot: a general learning foundation based on geometry, physical structure, uncertain beliefs, and experimentation**

Date: 2026-10-02  
Audience: a solo developer extending the Developmental-AI repository  
Status: proposed architecture and implementation roadmap; no implementation or training results are claimed

## 1. Purpose and scope

Develop a substantial new learning foundation for SkyBot. Minecraft remains the primary environment in which the agent develops. The learning core must also accommodate other environments through observation and action adapters.

The intended improvement is a learner that can represent persistent structure, infer reusable mechanisms, recognize uncertainty, select informative experiences, and revise inadequate explanations. This requires coordinated changes to perception, world modeling, memory, curiosity, planning, skills, and orchestration.

The project should remain manageable as a sequence of standalone development milestones. Each milestone must deliver useful functionality and evidence before the next depends on it.

### Intended outcomes

- More useful knowledge and competence per interaction.
- Better predictions about action consequences over multiple time steps.
- Reuse of learned mechanisms across appearances, situations, and environments.
- Appropriate uncertainty when observations are incomplete or rules change.
- Exploration directed toward learnable and useful questions.
- A common architecture that can learn unfamiliar controls and embodiments.
- A framework in which gradient learning, Bayesian inference, search, and direct solvers can coexist.

### Boundaries

This plan does not promise successful learning in every possible environment. Some environments provide insufficient evidence, arbitrary dynamics, or inaccessible controls. Universality is an architectural aspiration assessed through specified transfer experiments.

The scope does not require eliminating backpropagation, immediately replacing numerical backends, implementing a complete physics simulator, or supplying Minecraft recipes and skills. Building an independent GPU runtime is a later option requiring its own justification.

Minecraft interaction remains central. Small synthetic fixtures are appropriate for mathematical and inference tests; they are not substitutes for demonstrating useful behavior in Minecraft. Additional environments are eventually necessary to establish transfer.

## 2. Principles that govern the design

### 2.1 Supplied structure must be explicit

Maintain a register of assumptions. Distinguish mathematical structure supplied by the framework, sensor metadata supplied by an adapter, and knowledge acquired through experience.

Coordinate transformations, type checking, and probability bookkeeping are framework capabilities. Object identity, material properties, action meanings, and mechanism applicability generally require evidence.

Preserve the repository principle that meanings and skills are earned from experience. Evaluation labels, simulator internals, and privileged state must not silently enter the agent's observations, rewards, learned memory, or action selection.

### 2.2 Geometry is a family of representations

Support continuous quantities, discrete variables, sets, relations, and geometry. Do not require every environment to be a collection of rigid objects in Euclidean three-dimensional space.

Geometric algebra is useful for certain geometric objects and transformations. It does not automatically provide uncertainty, dimensional units, causal reasoning, conservation, or model discovery. Those require separate semantics and algorithms.

### 2.3 Physical laws have applicability conditions

A change of coordinates and a physical change to a scene are different operations. Predictions should remain consistent under coordinate changes. A physical rotation may change an outcome when gravity, boundaries, or other directional influences remain fixed.

Conservation depends on the system boundary, external interactions, and the law being modeled. Minecraft includes discrete creation and removal of blocks and game-specific rules. Hard constraints should apply only where justified. Approximate or uncertain laws need soft constraints and explicit exceptions.

### 2.4 Learning includes more than parameter updates

The system must be able to update state estimates, fit parameters, compare hypotheses, change model structure, and retire failed explanations.

Backpropagation remains appropriate for neural perception and differentiable models. Filtering, Bayesian updates, bounded structural search, and direct solvers address other parts of learning. These methods may overlap; Bayesian inference itself can use gradients.

### 2.5 Predictions are not observations

Keep actual experience, inferred state, imagined outcomes, and evaluator truth distinguishable. Imagined experience may train policies under explicit trust limits, but it cannot confirm the mechanism that generated it.

### 2.6 Complexity must earn its cost

Every added model, inference step, and planner has an interaction, memory, and compute cost. Evaluate both sample efficiency and wall-clock performance. A more accurate model that starves the agent of experience may reduce overall learning.

## 3. Repository starting point

The following is a map of relevant repository components, not a claim that every feature is enabled or validated in the current deployment.

| Existing area | Intended role in the redesign |
|---|---|
| `developmental_ai/world_model/rssm.py` | Baseline predictive model; later one implementation behind a common prediction interface |
| `developmental_ai/world_model/replay_buffer.py` | Starting point for sequence sampling and experience transport |
| `developmental_ai/sensors/` | Observation channels with declared metadata and provenance |
| `developmental_ai/slots/` and `developmental_ai/spatial/` | Starting points for identity, geometry, relations, and persistent state |
| `developmental_ai/curiosity/` | Baseline exploration methods and future experiment-selection integration |
| `developmental_ai/infra/progress_curiosity.py` | Existing progress measurement to evaluate before creating overlapping infrastructure |
| `developmental_ai/policy/` | Baseline control, options, and future planning integration |
| `developmental_ai/skill_bank/` | Skill persistence and future evidence-based applicability and competence |
| `developmental_ai/knowledge_graph/` | Derived knowledge view; mechanism evidence requires richer records |
| `developmental_ai/memory/` and episodic memory components | Existing persistence to reconcile with the evidence store |
| `developmental_ai/core/developmental_loop.py` | Orchestration to separate into explicit stages |
| `developmental_ai/core/glue_layer.py` | Existing integration responsibilities to audit and redistribute |
| `developmental_ai/environments/` | Environment adapters and transfer environments |
| `developmental_ai/infra/` | Telemetry, gates, accounting, and diagnostics to reuse |
| `tests/run_all.py` | Existing standalone test runner and CI registration conventions |

Repository documents contain historical descriptions that can conflict with later changes. Establish current behavior from code, resolved configuration, and run evidence. For example, `CLAUDE.md` records differences among collection paths and historical reward exploits. Treat those as migration risks requiring tests.

## 4. Language and tool choices

| Layer | Initial choice | Reason and later direction |
|---|---|---|
| Interfaces and orchestration | Python | Direct integration with the existing repository |
| Neural perception and dynamics | Python with PyTorch | Retain existing numerical and gradient infrastructure |
| Geometry and reference algorithms | NumPy/SciPy; PyTorch when needed | Straightforward reference implementations and numerical comparisons |
| Belief inference and mechanism search | Python | Fast iteration while the algorithm is still changing |
| Stable runtime components | Rust, only after evidence justifies a port | Strong types, memory control, and efficient CPU execution |
| Python/Rust integration | PyO3 | Keep a common Python-facing API |
| Custom GPU operations | Deferred | Add only for measured bottlenecks with correctness references |
| Tests | Existing standalone Python suites | Preserve repository conventions and runner/CI parity |

Use the repository virtual environment. Record exact dependency versions for reproducible experiments; select compatible versions during implementation rather than assuming the newest releases work with all environment dependencies.

Python typing does not enforce every contract at runtime. Validate frames, shapes, units, timestamps, and schema versions explicitly. Rust types can enforce some static relationships, but properties discovered at runtime still require dynamic checks.

## 5. Target architecture and package boundaries

The proposed loop is:

```text
Environment adapter
    -> observation records
    -> perception and state inference
    -> belief over state and mechanisms
    -> predicted outcomes for candidate actions
    -> experiment selection and control
    -> action execution
    -> recorded evidence
    -> parameter learning, model revision, and consolidation
```

All consumers use versioned state and mechanism interfaces. Background learners publish model snapshots at explicit boundaries. Planners record which snapshots produced their predictions.

Suggested package organization, introduced incrementally:

```text
developmental_ai/foundation/
    contracts/       # Backend-independent records and interfaces
    geometry/        # Types, frames, transformations, applicable constraints
    experience/      # Evidence records, persistence, retrieval
    mechanisms/      # Predictive mechanisms and their applicability
    inference/       # Beliefs, filtering, hypothesis management
    discovery/       # Bounded proposals and structural revision
    experiments/     # Information estimates and candidate evaluation
    runtime/         # Scheduling, snapshots, checkpoints, resource budgets
```

Existing perception and policy packages should implement these interfaces. Avoid building a second independent orchestration stack. A future native package can sit under `native/` once a specific port is justified.

### Core records

| Record | Minimum information |
|---|---|
| Observation | Stream, episode, sequence, timestamps, channel, payload, missingness, provenance |
| Action | Available control specification, command, duration, dispatch and completion timing |
| State belief | Representation version, inferred variables, uncertainty, observation support |
| Mechanism | Inputs, outputs, applicability, parameters, uncertainty, evidence references, version |
| Prediction | Source belief and model versions, candidate action, horizon, outcome distribution |
| Experiment | Question, alternatives, candidate intervention, predicted evidence, cost budget |
| Evidence | Actual observations, executed actions, prediction references, validity and provenance |
| Skill | Learned controller, initiation conditions, termination rule, competence evidence |

Unknown, absent, zero, and inapplicable must be separate states. Entity IDs and frame IDs need explicit scope; do not assume identities persist across unrelated resets or environments.

## 6. Implementation sequence

The stages below are solo projects. Finish their deliverables and review their evidence before expanding scope. Some iteration between adjacent stages is expected.

### Stage 1 — Baseline and assumptions

**Objective:** establish what improvement means and preserve a reproducible comparison.

**Implementation:**

1. Audit active configuration, collection paths, observation sources, reward channels, and model-update ownership.
2. Record which senses are available to the policy and which information is evaluator-only.
3. Create a reproducible run manifest containing revision or source snapshot identity, resolved config, dependencies, seeds, checkpoints, environment settings, and hardware.
4. Select development scenarios and reserve held-out scenarios before tuning.
5. Measure behavior, prediction error by horizon, throughput, memory, and existing failure modes.

**Language and scope:** Python measurement tools; existing telemetry where possible.

**Tests:** repeated baseline runs; validated outcome counters; evaluator isolation; checkpoint restoration; identical recorded-input evaluation where possible.

**Completion gate:** another run can reproduce the baseline within documented variation. Unknowns and skipped checks are listed. Historical README numbers are not treated as fresh measurements.

### Stage 2 — Shared interfaces and adapter contract

**Objective:** define how learning components communicate without encoding Minecraft semantics.

**Implementation:**

1. Introduce the core records from Section 5 with versioned serialization.
2. Specify required and optional sensor metadata, action shapes, timing, and reset semantics.
3. Separate domain payloads from common record fields.
4. Provide wrappers around existing state and prediction APIs.
5. Define migration rules for incompatible schemas and reject unsupported versions explicitly.

**Language:** Python protocols, data structures, and runtime validation.

**Tests:** serialization round trips; malformed and missing fields; continuous and discrete actions; mixed observation channels; cross-stream contamination; reset boundaries.

**Completion gate:** existing SkyBot data can travel through the interfaces, and a minimal nonspatial adapter can satisfy the same contract without fabricated spatial variables.

### Stage 3 — Orchestration migration

**Objective:** make algorithm replacement possible without duplicating behavior across execution paths.

**Implementation:**

1. Extract observation, state update, action selection, recording, and learning stages from the central loop behind interfaces.
2. Audit all reachable collection paths, including single-environment, parallel, and lifelong execution.
3. Preserve explicit stream ownership for policy updates and shared world-model training.
4. Introduce immutable inference snapshots and defined background publication points.
5. Keep the current implementation selectable as a baseline.

**Language:** Python.

**Tests:** recorded-input parity; resets; multiple streams; model publication during inference; save/resume; reward accounting; configuration switches.

**Completion gate:** no unexplained behavioral change from the refactor. Paths that remain unsupported fail explicitly. A nominally smaller run must not silently select a different agent.

### Stage 4 — Typed state and geometry library

**Objective:** make representation consistency executable.

**Implementation:**

1. Support scalars, vectors, discrete variables, relations, frames, timestamps, and uncertainty references.
2. Add transformation composition, inversion, and explicit frame conversion.
3. Represent supplied units and unknown scale separately; monocular estimates must not silently become metric ground truth.
4. Introduce geometric algebra operations only for selected use cases, with a declared algebra and conventions.
5. Add a constraint registry with scope, applicability, evidence, and violation diagnostics.

**Language:** Python/NumPy reference implementation; PyTorch-compatible operations where gradients are needed.

**Tests:** algebraic identities, randomized transforms, round trips, degeneracies, unit compatibility, coordinate covariance, and reference comparisons. Check gradients numerically for differentiable operations where meaningful.

**Completion gate:** coordinate changes preserve represented meaning within declared tolerances; invalid operations are detected; discrete and unknown structures remain expressible.

### Stage 5 — Experience and evidence infrastructure

**Objective:** make data quality and scientific comparison reliable.

**Implementation:**

1. Reconcile replay, episodic memory, and long-term storage responsibilities before introducing another store.
2. Preserve immutable observation/action records and attach revisable interpretations separately.
3. Record actual action timing and duration; distinguish environment termination from truncation and client recovery.
4. Save predictions before outcomes arrive, including source model versions.
5. Add retrieval by context, failure, coverage, and contradiction while retaining a representative sampling component.
6. Define retention, compression, disk limits, recovery, and migration behavior.

**Language:** Python; simple indexed manifests and chunked arrays are sufficient initially.

**Tests:** interrupted writes, recovery, sequence alignment, corrupted records, sampling frequencies, bounded growth, privacy of evaluator channels, and train/evaluation split leakage.

**Completion gate:** a selected interaction can be reconstructed with its original prediction, and sampling changes have measurable, documented effects.

### Stage 6 — Structured mechanism prediction

**Objective:** demonstrate that physical and relational structure improves learning on actual experience.

**Implementation:**

1. Define a common mechanism prediction API around state, action, elapsed time, and predicted outcome distribution.
2. Implement one modest relational dynamics model with continuous and discrete outputs.
3. Compare against the existing RSSM where outputs are comparable and against an ordinary model receiving the same structured inputs.
4. Apply symmetries and constraints only where their assumptions are declared.
5. Evaluate multiple horizons and distributional predictions; retain a residual model for unexplained effects.
6. Keep all architecture comparisons on identical data before changing exploration.

**Language:** Python/PyTorch.

**Tests:** held-out trajectories; contact and non-contact examples; duration sensitivity; reset exclusion; transformed inputs; law violations; stochastic observations; long-horizon drift.

**Completion gate:** predeclared improvement in prediction, calibration, or sample efficiency, with resource costs reported. Extra inputs and extra compute must not be confused with architectural gains.

### Stage 7 — Belief inference and uncertainty

**Objective:** maintain uncertainty about state and mechanisms instead of collapsing ambiguity prematurely.

**Implementation:**

1. Start with a small bounded ensemble or finite hypothesis set.
2. Separate observation noise from uncertainty due to limited knowledge where the model supports that distinction.
3. Add an update cycle for prediction, evidence incorporation, and posterior approximation.
4. Use particle filtering or other sampling only where multimodal state inference warrants it.
5. Track model mismatch and rule changes separately from ordinary measurement noise.
6. Define numerical stabilization, hypothesis limits, and recovery from overconfidence.

**Language:** Python with PyTorch/NumPy/SciPy as required.

**Tests:** ambiguous observations; missing data; known posterior fixtures; contradictory evidence; changed dynamics; calibration curves; interval coverage; particle degeneration if applicable.

**Completion gate:** uncertainty has demonstrated predictive value and can increase appropriately. Agreement among models alone is not accepted as proof of correctness.

### Stage 8 — Learned representation and persistent perception

**Objective:** infer useful structure from the agent's permitted observations.

**Implementation:**

1. Connect existing motion, slot, and spatial components through the state-belief interface.
2. Learn persistent identity and correspondence using temporal prediction and interaction evidence.
3. Represent uncertain identity, occlusion, appearance changes, and new or disappearing entities.
4. Allow alternative state families such as fields or discrete processes when object decomposition is unsuitable.
5. Add bounded split/merge proposals and retain residual features for unmodeled information.
6. Version representations and invalidate or translate dependent caches and mechanisms when semantics change.

**Language:** Python/PyTorch.

**Tests:** occlusion and reappearance; identity swaps; camera movement; texture changes; scale ambiguity; distractors; downstream prediction; representation migration; non-object fixtures.

**Completion gate:** learned state improves a downstream outcome, not just visual attractiveness. Separate privileged-state diagnostics from pixel-based results.

**Research limit:** early stages may use fixed representation families. Discovery of arbitrary useful representations is a long-term research objective, not an assumed capability.

### Stage 9 — Mechanism discovery and structural revision

**Objective:** change explanations when fitting existing parameters is inadequate.

**Implementation:**

1. Define a restricted proposal language for adding dependencies, splitting applicability, composing mechanisms, and replacing transitions.
2. Search under explicit time and memory budgets.
3. Evaluate candidates on development evidence with complexity penalties and separate validation episodes.
4. Preserve provenance, contradictions, and a probationary status for new mechanisms.
5. Support promotion, retirement, and revival when applicability changes.
6. Keep correlated observational patterns distinct from intervention-supported causal claims.

**Language:** Python; optimized search can be considered later.

**Tests:** known hidden rules; spurious correlation; insufficient evidence; confounding fixtures; regime changes; mechanism proliferation; catastrophic replacement; recovery from rejected candidates.

**Completion gate:** structural revision improves held-out predictions or adaptation compared with parameter-only updates under a stated budget.

### Stage 10 — Curiosity as experiment selection

**Objective:** choose experiences that resolve useful uncertainty.

**Implementation:**

1. Define experiment candidates using actual available actions, durations, and learned skills.
2. Estimate how outcomes could distinguish competing mechanisms or improve competence.
3. Combine expected learning, usefulness, task requirements, effort, and risk through explicit, logged terms.
4. Begin with short horizons and limited candidate counts.
5. Compare against existing learning-progress and imagination curiosity before retiring any baseline.
6. Use development probe sets for progress measurement; never use final held-out evaluation data to guide actions.
7. Track forgetting and model instability so repeated relearning cannot masquerade as unlimited discovery.

**Language:** Python; model calls may use PyTorch.

**Tests:** noisy distractors; standing still; useful passive observation; repeated experiments; false model disagreement; model-generated jackpots; forgetting loops; action-dependent evidence; cost accounting.

**Completion gate:** more validated knowledge or competence per interaction than the baseline across repeated runs. A larger intrinsic reward is not sufficient evidence.

Every reward or gate must state its channel, occlusion behavior, idle behavior, and reachable recovery condition. Reuse the repository's ledger and signal-health infrastructure.

### Stage 11 — Planning, skills, and consolidation

**Objective:** turn discoveries into reliable behavior and reusable competence.

**Implementation:**

1. Integrate predictions into bounded planning with uncertainty-aware horizon limits.
2. Fall back to the existing learned controller when planning times out or the model is unreliable.
3. Learn initiation and termination conditions for skills from experience.
4. Store competence as evidence-backed estimates; names and copied weights do not establish distinct skills.
5. Separate episode memory, candidate mechanisms, and consolidated knowledge.
6. Re-evaluate old skills after representation or dynamics changes.
7. Maintain retention samples and explicit forgetting measurements.

**Language:** Python/PyTorch.

**Tests:** closed-loop outcomes; model exploitation; deadline failures; action remapping; option duration and credit assignment; skill discrimination; competence calibration; retention; resume; recovery from stale models.

**Completion gate:** improved real behavior with documented costs and retained capabilities. Planning success in imagination alone does not qualify.

### Stage 12 — Transfer and runtime optimization

**Objective:** establish adaptation beyond the original environment and optimize a proven design.

**Implementation:**

1. Test the same learning core through multiple adapters, including a nonspatial environment.
2. Independently change appearance, dynamics, controls, observation availability, and embodiment.
3. Compare fresh learning with transferred state and mechanisms; allow abstention when knowledge does not apply.
4. Profile the full loop before porting code.
5. Select one stable CPU bottleneck for a Rust implementation, retaining the Python reference.
6. Preserve batched backend operations and avoid per-element Python/Rust crossings.
7. Consider custom GPU kernels only after profiling demonstrates a remaining need.

**Language:** Python first; Rust/PyO3 selectively.

**Tests:** adapter conformance; matched fresh/transfer runs; negative transfer; rule changes; serialization compatibility; Python/native numerical parity; resource benchmarks; full regression suite.

**Completion gate:** documented transfer under specified conditions and measured runtime gains without degraded correctness. The claim remains bounded by tested environments.

## 7. Testing and experimental procedures

### 7.1 Test levels

| Level | Purpose | Where to run |
|---|---|---|
| Unit | Algebra, types, shapes, boundary conditions, numerical routines | Local development environment |
| Contract | Assumptions, evidence isolation, recovery, reset behavior, interfaces | Local when dependencies permit |
| Integration | Connected components, snapshots, persistence, concurrency | Local or training host depending on requirements |
| Offline learning | Compare models using fixed data | Suitable compute host |
| Closed-loop learning | Measure behavior when decisions change collected data | Training host and environment server |
| Transfer | Test adaptation and knowledge reuse | Controlled evaluation environments |
| Regression | Prevent known failures and unintended losses | Appropriate local/host tiers |

Retain the existing standalone `__main__` test convention. Register new suites in `tests/run_all.py` and the relevant CI workflow together. A skipped suite is not a passed suite.

Existing commands, run from the repository root using its virtual environment:

```bash
PYTHONPATH=. ./venv/bin/python tests/run_all.py unit
PYTHONPATH=. ./venv/bin/python tests/run_all.py contract
PYTHONPATH=. ./venv/bin/python tests/run_all.py legacy
```

Run host-dependent suites where their dependencies and environment exist. Do not treat a missing package as evidence that behavior was tested. Run targeted suites during development and broader checks when integration or new evidence warrants them.

### 7.2 Procedure for a learning comparison

1. Write the hypothesis, primary metric, resource budget, failure conditions, and acceptance rule before running the final comparison.
2. Split by episodes, trajectories, worlds, or regimes as appropriate. Adjacent frames from one episode must not leak across training and evaluation.
3. Tune on development data; freeze settings before final evaluation.
4. Use identical observations and action availability across competing methods unless input quality is the variable being studied.
5. Compare fixed-data performance before comparing online exploration.
6. Run a small pilot to determine feasibility and variance. Use at least three independent seeds for an initial screening comparison; expand when uncertainty could change the decision. Three seeds are not a guarantee of statistical reliability.
7. Report per-run outcomes, aggregate variation, failures, environment interactions, model updates, wall-clock time, and peak resources.
8. Include an ablation removing the new component and a simple alternative where feasible.
9. Record negative results and preserve the baseline if the criterion is unmet.

Shared Minecraft worlds contain interacting agents and changing server state. Such runs are not automatically independent. Use isolated evaluation instances or explicitly model and report shared-world conditions.

### 7.3 Metrics

| Area | Useful evidence |
|---|---|
| Prediction | Error by horizon, event likelihood, distributional scores, applicability failures |
| Uncertainty | Calibration, interval coverage, response to missing or contradictory evidence |
| Representation | Identity continuity, downstream prediction/control, robustness to appearance changes |
| Exploration | Interactions needed to resolve a hidden mechanism, retained learning, useful competence |
| Behavior | Validated task outcomes, recovery, skill reliability, idle and exploit behavior |
| Transfer | Learning required after a change, retained competence, negative transfer |
| Runtime | Environment steps per second, planning latency, update time, RAM/VRAM/disk use |

Do not compare arbitrary latent distances across models as if they had the same meaning. Use common observable targets or validated downstream outcomes. Conservation error is relevant only where the constraint applies.

### 7.4 Required regression cases

- Repeated idle observations must not produce persistent fabricated learning progress.
- Genuine information from passive observation may still be useful; test it separately from idle reward farming.
- GUI occlusion and unavailable sensors cannot manufacture evidence or erase necessary recovery actions.
- Gates must have reachable recovery conditions.
- Every supported collection path must expose the same declared learning behavior.
- Reset splices and unrelated streams cannot become valid training transitions.
- Evaluator information cannot reach learning through rewards, caches, or diagnostic side channels.
- A model cannot certify its own imagined outcomes as real evidence.
- Skills and mechanisms must survive persistence with identity and version information intact.

## 8. Migration, checkpointing, and operational controls

For each new component, use the following progression:

1. **Offline:** evaluate on stored experience.
2. **Shadow:** compute predictions during a run without controlling actions or rewards.
3. **Limited integration:** enable through configuration for a bounded experiment.
4. **Default candidate:** broaden only after declared gates pass.

Shadow execution consumes resources; measure its effect on the baseline. Do not allow background experiments to silently alter the active learner's replay, normalization, or memory.

Checkpoint environment-independent model state, representation versions, belief state, optimizer state where applicable, random states, evidence indices, and scheduling counters. Document which environment state can and cannot be restored exactly.

Before migrations, preserve the original checkpoint and accumulated skill/memory data. Use new output locations and explicit conversion tools. Never overwrite a checkpoint merely because the new schema loads successfully.

Rollback selects the previous component configuration and compatible state. It must not depend on reversing irreversible memory mutations. A new model becoming default does not authorize deletion of its predecessor's evidence.

## 9. Resource strategy for a solo developer

The repository describes a constrained reference machine. Re-measure available resources before setting budgets; documentation is not a live hardware inventory.

- Use local development for contracts, data inspection, and small fixtures.
- Use the training host for actual Minecraft integration and sustained learning.
- Start with short planning horizons, few hypotheses, and bounded mechanism search.
- Separate learning cadence from environment stepping so expensive updates do not automatically stall interaction.
- Log data movement and serialization overhead alongside model computation.
- Keep representative evidence when prioritizing unusual transitions.
- Track disk growth and recovery time as first-class limits.

Do not assign fixed calendar promises to research stages. Estimate the next milestone after a pilot demonstrates its data and compute requirements. Perception discovery and causal mechanism revision are likely to require substantial iteration.

## 10. Solo development workflow

For each stage, maintain one concise milestone record containing:

1. Problem and hypothesis.
2. Supplied assumptions and permitted inputs.
3. Proposed interfaces and affected repository areas.
4. Smallest complete implementation.
5. Tests and experimental acceptance criteria.
6. Compute and storage budget.
7. Findings, limitations, and rollback procedure.

Keep one major algorithmic change active at a time. Separate refactoring from learning changes so observed gains have an interpretable cause. Reuse existing infrastructure after auditing it; avoid parallel stores or reward systems with overlapping ownership.

### Milestone groups

| Group | Stages | Deliverable |
|---|---|---|
| A: dependable foundation | 1–5 | Reproducible baseline, interfaces, modular execution, typed records, trustworthy experience |
| B: predictive understanding | 6–7 | Structured prediction with meaningful uncertainty |
| C: learned explanations | 8–9 | Learned representations and bounded mechanism revision |
| D: self-directed competence | 10–11 | Informative exploration connected to planning and reusable skills |
| E: demonstrated breadth | 12 | Transfer evidence and justified runtime optimization |

The first project to undertake is Stage 1. Its concrete outputs are a baseline manifest, an assumption register, an evaluation split, and a measurement report. The first major learning claim should come from Stages 6–7, before investment in a custom native runtime.

## 11. Risks and decision rules

| Risk | Response |
|---|---|
| Architecture becomes abstract before it predicts anything | Require every interface to serve an implemented producer, consumer, or test fixture |
| Geometric assumptions exclude valid environments | Support discrete/nonspatial state and test constraint violations |
| Better results come from additional privileged inputs | Equalize inputs and label oracle diagnostics explicitly |
| Better predictions do not improve behavior | Evaluate closed-loop decisions before promoting the component |
| Curiosity rewards noise or forgetting | Test distractors, retention, and evidence-backed progress |
| Mechanisms proliferate without benefit | Bound search and penalize unnecessary complexity |
| Representation changes invalidate accumulated knowledge | Version semantics, revalidate dependents, preserve old evidence |
| Ensemble agreement creates false confidence | Measure calibration and performance under distribution changes |
| Planning exploits an inaccurate model | Bound horizons and require real-outcome validation |
| Added computation reduces experience too much | Compare equal-time results alongside equal-interaction results |
| New implementation obscures old failures | Preserve baseline execution and incident-based regression cases |

If a stage fails its criterion, inspect whether the limitation is data quality, representation, inference, optimization, or evaluation. Do not automatically respond by enlarging the architecture. Keep useful infrastructure even when an algorithmic hypothesis is rejected.

## 12. Completion criteria for the overall program

The program has produced a useful new foundation when:

- Components share a versioned representation of evidence, uncertain state, and mechanisms.
- Geometry and physical constraints have explicit scope and can be rejected when inappropriate.
- Learned mechanisms improve prediction or adaptation under fair comparisons.
- Curiosity improves retained knowledge or competence rather than reward alone.
- Planning and skills improve actual behavior without unacceptable runtime costs.
- Representation and dynamics changes can be handled without silently corrupting memory.
- The same core supports Minecraft and meaningfully different environments through adapters.
- Transfer claims are backed by held-out results, including failures and negative transfer.
- Backpropagation is available where useful, without being a requirement for every learning operation.

This would establish a broader and more testable learning architecture. It would not prove unrestricted universality or guarantee human-like understanding.

## 13. Background references

These references motivate individual components. None validates the complete proposed SkyBot architecture.

- [Geometric Deep Learning](https://arxiv.org/abs/2104.13478): mathematical structures and symmetry-based inductive biases.
- [Geometric Algebra Transformer](https://papers.nips.cc/paper_files/paper/2023/hash/6f6dd92b03ff9be7468a6104611c9187-Abstract-Conference.html): geometric algebra representations within a learned architecture.
- [E(n) Equivariant Graph Neural Networks](https://proceedings.mlr.press/v139/satorras21a.html): a simpler structured baseline for geometric prediction.
- [Toward Causal Representation Learning](https://arxiv.org/abs/2102.11107): representation, mechanisms, and transfer questions.
- [Challenging Common Assumptions in Unsupervised Disentanglement](https://arxiv.org/abs/1811.12359): limits on identifying factors without additional assumptions.
- [Interaction Networks](https://arxiv.org/abs/1612.00222): learned dynamics over objects and relations.
- [PLATO](https://www.nature.com/articles/s41562-022-01394-8): object-centered physical prediction and its experimental scope.
- [VIME](https://arxiv.org/abs/1605.09674): information-gain exploration.
- [Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/abs/1808.04355): curiosity performance and stochastic-environment limitations.
- [Hamiltonian Neural Networks](https://arxiv.org/abs/1906.01563): physical structure compatible with gradient learning.
- [Differentiable Particle Filters](https://arxiv.org/abs/1805.11122): inference structure combined with learned components.
- [PyO3 documentation](https://pyo3.rs/main/doc/pyo3/): future Python/Rust integration.

Repository starting references: [README](README.md), [incident and working notes](CLAUDE.md), [perspective learning](docs/PERSPECTIVE_LEARNING.md), [general infrastructure](docs/GENERAL_INFRASTRUCTURE.md), and [test runner](tests/run_all.py). Validate historical claims against the active code and run configuration before implementation.
