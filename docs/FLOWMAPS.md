# Developmental AI — current flow maps (the real organism)

These redraw the two original concept diagrams against **how the system actually
works today**, after the rung ladder + integration capstone. Key differences from
the original aspirational stack:

- The "Stable-Baselines3 policy" is a **custom `StandaloneActorCritic` (PPO)**, with
  a **knowledge-conditioned actor** (symbolic feature-conditioning, Path A).
- "DreamerV3" is a **custom RSSM world model** — now with a **twohot reward head**,
  **goal-prioritized replay**, and an optional **causal action-alignment** fix.
- "Curiosity Baselines + ICM" is a **custom ICM**, with a **learning-progress** mode
  (rewards error *reduction*, ignores the noisy-TV trap).
- The **skill bank is local** (not HF Hub); the **KG uses PyKEEN embeddings** (no Neo4j),
  and the **GNN→graph path is largely decorative** — the load-bearing symbolic signal
  is the **affordance/regime broadcast** that conditions the policy.
- New organs the original didn't have: **reward mixer with adaptive gating**,
  **developmental stage controller**, **dream actor + distillation (augment)**, and a
  **self-model (competence predictor)**.

Legend (node colour = role): 🟩 perception · 🟧 curiosity/drive · 🟦 goals & world model ·
🟫 action · 🟥 error/learning signal · 🟩(dark) skills/memory.

---

## 1) Architecture — what the organism is made of

```mermaid
flowchart TD
  ENV["Gymnasium / MiniGrid<br/>DoorKey · RuleRegime · NoisyTV · classic-control<br/><i>obs out, actions in</i>"]:::perc

  CUR["Curiosity engine — custom ICM<br/>novelty OR <b>learning-progress</b><br/><i>→ intrinsic reward</i>"]:::cur
  MIX["Reward mixer<br/><b>adaptive gating</b>: intrinsic weight ↓<br/>as extrinsic reward becomes findable"]:::cur

  POL["PPO policy — StandaloneActorCritic<br/><b>knowledge-conditioned</b> actor-critic"]:::act

  RSSM["World model — RSSM (DreamerV3-style)<br/>posterior = abduction · prior = imagine<br/><b>twohot reward head</b> · <b>goal-prioritized replay</b><br/>continue head · PER · causal-align"]:::wm
  DREAM["Dream actor-critic (latent space)<br/>trains in imagination →<br/><b>distilled</b> into real policy<br/>value-gated · annealed · sharpened"]:::wm

  STAGE["Developmental stage controller<br/>EXPLORE ⇄ EXPLOIT (⇄ IMAGINE)<br/><i>from WM-error plateau + skill rate</i>"]:::goal
  SELF["Self-model — CompetencePredictor<br/>calibrated P(success) + learning-progress<br/><i>what to practice · notices forgetting</i>"]:::goal

  KG["Knowledge graph + PyKEEN embeddings<br/>facts · rules"]:::wm
  BCAST["Affordance / regime broadcast<br/><b>conditions the policy</b> (Global Workspace)"]:::wm

  SKILL["Skill bank (local)<br/>mint on mastery · warm-start transfer · dedup<br/><i>capabilities compound across envs</i>"]:::skill
  LLM["Local LLM — Ollama llama3.1:8b<br/>async goal generation + shaping<br/><i>stage-gated · offline</i>"]:::goal
  PT["PyTorch (CPU) — tensor foundation"]:::found

  ENV -->|observations| CUR
  ENV -->|observations| POL
  CUR --> MIX --> POL
  POL -->|actions| ENV
  ENV -->|transitions| RSSM
  RSSM -->|prediction error| CUR
  RSSM -->|imagined rollouts| DREAM
  DREAM -->|policy distillation| POL
  RSSM --> STAGE --> MIX
  STAGE --> DREAM
  POL --> SELF --> STAGE
  SELF --> SKILL
  RSSM -->|symbolic facts| KG --> BCAST -->|broadcast feature| POL
  SKILL -->|warm-start| POL
  LLM -->|goals + shaping| MIX
  PT -.-> RSSM
  PT -.-> POL
  PT -.-> DREAM

  classDef perc  fill:#1f5e51,stroke:#3da08a,color:#eafff8;
  classDef cur   fill:#7a4d12,stroke:#d39b3a,color:#fff3df;
  classDef goal  fill:#3b3a8c,stroke:#7b78e6,color:#eceaff;
  classDef act   fill:#7a2f23,stroke:#d97a63,color:#ffe9e3;
  classDef wm    fill:#2a2f6b,stroke:#6f78d6,color:#e6e9ff;
  classDef err   fill:#7a2330,stroke:#d96379,color:#ffe3e9;
  classDef skill fill:#2f5e1f,stroke:#7ad36b,color:#eaffe3;
  classDef found fill:#3a3a3a,stroke:#888,color:#ddd;
```

---

## 2) The developmental learning loop — how it runs

```mermaid
flowchart TD
  PERCEIVE["Perceive<br/>egocentric observation"]:::perc
  CURI["Curiosity engine<br/>novelty / <b>learning-progress</b><br/>→ intrinsic reward"]:::cur
  GOAL["Set goal<br/>autotelic / LLM goal +<br/><b>self-model</b> picks what to practice"]:::goal
  GATE["Reward mixer — adaptive gating<br/>intrinsic ↓ as task reward becomes findable"]:::cur
  STAGE["Stage controller<br/>EXPLORE ⇄ EXPLOIT"]:::goal
  ACT["Explore & act<br/>PPO policy (knowledge-conditioned)"]:::act
  WMOD["World model (RSSM)<br/>internal map: recon + reward + continue"]:::goal
  OBS["Observe result"]:::perc
  PE["Prediction error<br/>surprise signal"]:::err
  DREAMSTEP["Dream<br/>imagine rollouts → dream actor →<br/>distill into policy (augment)"]:::goal
  WMUP["Update world model"]:::goal
  SELFUP["Update self-model<br/>competence (calibrated) + learning-progress"]:::goal
  SKILLADD["Skill added / warm-start<br/>capability grows · transfers across envs"]:::skill

  PERCEIVE --> CURI --> GOAL --> GATE --> STAGE --> ACT
  ACT --> OBS --> PE
  ACT --> WMOD --> OBS
  PE -->|refine| WMUP --> DREAMSTEP --> SKILLADD
  PE -->|drives anneal| GATE
  PE --> SELFUP --> GOAL
  DREAMSTEP -->|distilled policy| ACT
  SKILLADD -->|open-ended, keep growing| PERCEIVE
  STAGE -.->|EXPLORE — curiosity leads| CURI
  STAGE -.->|EXPLOIT — task leads| ACT

  classDef perc  fill:#1f5e51,stroke:#3da08a,color:#eafff8;
  classDef cur   fill:#7a4d12,stroke:#d39b3a,color:#fff3df;
  classDef goal  fill:#3b3a8c,stroke:#7b78e6,color:#eceaff;
  classDef act   fill:#7a2f23,stroke:#d97a63,color:#ffe9e3;
  classDef err   fill:#7a2330,stroke:#d96379,color:#ffe3e9;
  classDef skill fill:#2f5e1f,stroke:#7ad36b,color:#eaffe3;
```

---

## 3) Honesty overlay — which paths are *proven* load-bearing

The map above is the **full intended organism**. What the falsifiable rung tests
actually established about each path:

| Path | Status | Evidence |
|---|---|---|
| Curiosity (learning-progress) — reward-free exploration | ✅ **essential** | Rung 3: LP ≫ novelty on noisy-TV (4/4 seeds) |
| Symbolic **broadcast** conditioning the policy | ✅ **load-bearing under hidden structure** | Rung 6: content lesion craters behaviour (4/4, decisive) |
| Self-model causally driving behaviour | ✅ **calibrated + load-bearing** | Rung 5: intact ≫ scramble (3/3); ECE ~0.03 |
| Adaptive gating (curiosity anneal) | ✅ **net-positive safeguard** | turns the Rung-1 intrinsic-tax into "never worse than vanilla" |
| Skill bank / cumulative transfer | 🟡 **works, modest** | mints + warm-starts; transfer strong but hard to beat plain policy-carry |
| Curiosity + symbolic on *findable-reward* tasks | 🟡 **mild tax** | Rung 1: plain PPO wins where reward is easy to find |
| Dreaming (imagination→policy) | 🟡 **safe, not load-bearing** | Rung 4: harm fixed; neutral — **bottlenecked by WM action-causality** |
| Counterfactual reasoning (abduction→intervene→predict) | 🟡 **machinery works, WM-limited** | Rung 8: posterior ≫ prior (abduction helps) but not decisive |
| Raw KG → GNN graph-vector path | ⬜ **decorative** | Rung 6: the load-bearing signal is the broadcast, not the GNN pool |

**The through-line:** every component earns its keep *where the task demands it* —
curiosity reward-free, the broadcast under hidden structure, the self-model for
metacognition — and is a mild tax where plain RL suffices. The recurring limiter for
the *model-based* organs (dreaming, counterfactuals) is the **world model's weak
action-conditioning**: it predicts observation dynamics well but only faintly
distinguishes *which action* causes *which outcome*.

**Integration capstone (whole-organism):** the full stack — including the local LLM —
runs end-to-end across a sequence of novel envs without falling over, and its control
signals behave (curiosity anneals, stage controller hands off, skills mint). It also
surfaced a real *compose-time* bug invisible to every isolated rung test
(dream-augment was silently incompatible with symbolic-on — since fixed). So: it
**coheres and develops autonomously**, but "smoothly" required fixing composition
failures that only appear when everything is on at once.
