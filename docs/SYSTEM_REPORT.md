# SkyBot — system report, September 2026

*One document: what was built, what it cost, what broke, what the machine
needs, and what to do next. Supersedes nothing; `PERSPECTIVE_LEARNING.md`,
`PLURALITY_ROADMAP.md` and `TESTING_PLAN.md` hold the detail this summarises.*

---

## 1. What was built

Four waves, from "the agent sees a flat 128×128 frame and has no reason to
recover the 3D world behind it" to a sensorium with instances, spatial
memory, and an action space that can express what it perceives.

| Wave | Shipped | Enabled |
|---|---|---|
| **1 — perspective from motion** | flow head + warp objective, 8×8 spatial encoder with coord channels, residual-weighted reconstruction, proprioception into the world model, 4 unpaid senses, residual→learning-progress | ✅ |
| **2 — a correct photometric loss** | auto-masking, multi-scale pyramid, `flow_mode: raw\|depth` with ego geometry from stored proprio, goal-replay threshold | ✅ (`depth` off) |
| **3–4 — plurality (phases 1–4)** | sensor bus, native-resolution crosshair fovea, screen effects, dead reckoning, motion, light, sky, spatial surprise map, multi-horizon prediction, egocentric occupancy, successor map, walkability, the RED oracle | ✅ |
| **5–7 — instances and reach (phases 5–7)** | SAVi object slots + relations + per-slot grounding, finer aim, strafe/sprint/sneak, hold durations, audio (Python half) | ⛔ gated |

**Central idea:** *depth is what your own movement reveals; objectness is what
moves together when you move.* Everything above is that claim, plus the
machinery to express it.

**The rule everything obeys:** a **sense** is a transducer reporting a
physical quantity without naming anything; a **meaning** names or classifies.
Senses may be policy input. Meanings are evaluation only, and are excluded
*structurally* — `SensorBus.read_policy` never visits a RED sensor, so there
is no filter to forget.

---

## 2. Validation status

All on an RTX 2000 Ada training host against real MineRL, 2026-09-19/20.

| | |
|---|---|
| **Unit** | 80 cases, 3 files — parts, bounds, neutrals, degenerate inputs |
| **Contract** | 5 suites — the design arguments, with measured numbers |
| **Legacy** | 6 suites — the pre-existing regression surface |
| **Stage 0** | ✅ GO — MineRL exposes a 24-key human action space; all macro keys real |
| **Stage 1** | ✅ 14/14 suites |
| **Stage 2** | ✅ 500 steps, bare adapter — **all 8 sensors vary**, 0 NaN, 500 oracle reads with no leak, 28/28 macros distinct |
| **Stage 3** | ✅ 300 steps, full 2-client lifelong loop — transport 4144=4144, policy proprio 477=477, maps advance, clamp holds, **dead-reckoning drift 4.9→6.2 blocks** |
| **Stage 4+** | ⛔ not run — the flow go/no-go needs hours |

**Nothing here proves the agent learns better.** It proves the machinery
carries real signal and does not pay for standing still.

---

## 3. Issues that arose

Every one was found by a test or a measurement, and every one would have been
silent. This is the most useful section in the document.

### 3.1 Bugs in the new code

| Bug | What it would have done |
|---|---|
| `flow_warp` used `align_corners=False` with a `linspace(-1,1,n)` grid | **Zero flow was not the identity** (max err 0.490). The loss charged a blur floor no field could remove; the head's cheapest escape was a constant offset — a bias on every depth reading. |
| `compute_kl_loss` free nats = 0.1 | The horizon measurement sat **pinned at its own floor** — 0.1100 for every configuration, all seeds. With an unclamped divergence, 16-step rollout error went **0.066 → 0.0002**. |
| `charbonnier(0) = 1e-3` | A **static** cell computed `1e-3/1e-3 ≈ 1.0` surprise — manufacturing curiosity out of stillness, the exact failure this project keeps paying for. |
| Occupancy re-registration translation sign | A remembered wall **receded** as the agent walked into it. |
| Bilinear rotation does not conserve mass | Confidence could compound above `1/decay` until every cell read solid — the agent would believe it was walled in. |
| Goal-replay threshold `1e-3` vs shaped rewards | The stratum selected **396 of 396 windows** — exactly uniform sampling, on a run whose history holds ~35 log breaks. |
| Unknown macro keys | Accepted silently by `act[k]=v` and dropped at the socket; the policy learns a button that does nothing. Now a boot-time crash. |
| NaN on degenerate frames | Four pixel sensors produced NaN on an empty/odd frame. A NaN in the RSSM posterior poisons every latent downstream, permanently. |

### 3.2 A feature removed by a pre-existing test

I added nine **hotbar** macros. `_action_widening_smoke.py` already carried
`# still no hotbar keys (deliberate: minimal exploration burden)` and failed.
I removed the feature rather than weakening the test — and **Stage 0.3 proved
the original decision right**: MineRL does not render the HUD (bottom-band
pixel variance **0.029 vs 0.290** mid-frame), so the agent cannot *see* its
selection change. Nine invisible actions over an inventory that has never
held two things is exploration burden bought with nothing.

### 3.3 Recommendations that did not survive measurement

- **Longer flow baselines (`flow_strides`)** — proposed on classical-SfM
  reasoning. Measured: near/far ratio `1.60/1.28/1.06 → 0.86/0.79/0.84`. Not
  merely unhelpful — **ordering inverted**. One head emits one field per
  `(latent, action)`; at stride *k* the displacement is *k*× larger while
  `max_flow` bounds it, so a long-stride term is unrepresentable and drags
  the shared head into a compromise. Shipped **off**, with the measurement.
- **Time-of-day phase (G6)** — a phase is *not derivable from one frame*;
  dawn and dusk look alike. Shipped as sky *appearance*, claiming nothing.
  The contract pins `dawn == dusk`.

### 3.4 Open alarm

**`flow_mask` retained fraction 0.26** (want 0.6–0.95). Auto-masking is
dropping 74% of pixels, so the loss trains only on what it already explains.
That is the starvation spiral the contract warned about — and the +40% gain I
measured for auto-masking was on a *synthetic* scene with one clean occluder,
which Minecraft frames are not. Only 3 world-model updates occurred, so the
sample is tiny. **Stage 4 decides**; `flow_automask: false` is the exact
revert and contract Q proves it reverts cleanly.

### 3.5 Harness defects (mine, not the code's)

`pgrep -f <pattern>` matches the SSH command doing the checking — the same
footgun CLAUDE.md records for `pkill -f "tailscale nc"`. Also: a distinctness
check that compared only the socket dict would have condemned the hold-duration
macros, which are distinguished by `_ticks`, not by keys.

---

## 4. Measured resource profile

Training host: RTX 2000 Ada (16 GB), 128 cores, 125 GB RAM, Ubuntu, Python 3.10.

### 4.1 Brain (world model, live config)

| | |
|---|---|
| parameters | **120.5 M** |
| fp32 weights | **482 MB** |
| resident with Adam (2 moments + grads) | **~1.45 GB** |
| encoder / RSSM / **decoder** / flow head / sensor encoders | 6.7 / 28.6 / **71.7** / 5.7 / 0.24 M |

> **The decoder is 59% of the world model** and its only job is pixel
> reconstruction. See §6.1.

### 4.2 Replay buffer

| | |
|---|---|
| row | 49,152 uint8 (obs) + 4,144 float32 (sensors) = **64.2 KB** |
| per 25k-transition block | **1.64 GB** |
| 1 M transitions | **~64 GB** |

Sensors added 16.6 KB/row — a **34% larger row** than pixels alone. This is
the single largest cost of the plurality work and is worth knowing before
sizing a long run.

### 4.3 Runtime

| | |
|---|---|
| VRAM | **5.4 GB / 16 GB** (≈4.7 GB of it is Ollama's `llama-server`) |
| RAM | python ≈4.9 GB + llama-server ≈4.7 GB ≈ **10 GB** |
| GPU utilisation | **0–2%** — the GPU is idle; the client and the step loop are the constraint |
| Disk | `venv_mc` 12 GB, `mc-build` 5.6 GB, repo+build **34 GB** |
| throughput, bare adapter | **2.64–2.74 steps/s** (1 client) |
| throughput, full loop | **0.24 steps/s** (2 clients) — an **11× gap** |

### 4.4 Recommended provisioning

| Resource | Minimum | Comfortable | Why |
|---|---|---|---|
| **VRAM** | 8 GB | **16 GB** | brain ~1.5 GB + Ollama ~5 GB + activations; 16 GB leaves room for `flow_mode: depth` and slots |
| **RAM** | 32 GB | **64 GB+** | scales with buffer blocks, not with the model |
| **Disk** | 60 GB | **250 GB** | 34 GB build + buffer at 1.64 GB/block + brain snapshots |
| **CPU** | 8 cores | **16+** | MineRL clients are CPU-bound; `taskset` pins one core per client |
| **Brain storage** | — | **10 GB per archived brain** | 482 MB world model + symbolizer + magnet + familiarity + KG + skill bank + options, with headroom for several generations |

**Back up the brain, not the buffer.** A training host has already died with ~11 days of
unbacked state. The buffer is large and re-collectable; the brain is small and
is not.

---

## 5. The 11× throughput gap

2.64 steps/s on the bare adapter, 0.24 in the full loop. At 0.24 a six-hour
go/no-go yields ~5,000 steps, which is thin for a question that needs hours of
trend.

PHASE_TIMING_PLACEHOLDER

---

## 6. Future design, with reasoning

### 6.1 Delete the decoder — the largest single win available

**71.7 M of 120.5 M parameters (59%) exist to reconstruct pixels.** Pixel
reconstruction is also the objective this project has already documented as
object-blind: a three-block trunk is rounding error against sky and texture,
which is why Wave 2 had to add residual weighting to compensate.

The JEPA line (V-JEPA, LeWorldModel, Delta-JEPA) predicts in *embedding*
space with no decoder at all, and reports better action-relevance precisely
because pixel losses spend capacity on task-irrelevant detail. Wave 3's
multi-horizon latent prediction already works this way and measured a **300×**
improvement in rollout accuracy.

Removing the decoder would cut the model ~60%, free ~900 MB of optimiser
state, and remove the loss term that most dilutes what the encoder attends
to. The cost is losing pixel dreams for the viewer — worth trading.

### 6.2 Resolve the auto-mask alarm before anything else

§3.4. One measurement, one config line either way.

### 6.3 The flow go/no-go gates real work

`flow_mode: depth` and all of Phase 5 wait on it. Do **not** enable slots to
"see if it helps" — without a working flow field they decompose noise, which
is SAVi's own stated failure mode.

### 6.4 Sub-step frames during attack runs

At `action_repeat: 4` the agent stores 1 frame per 4 ticks, so a ~60-tick log
break is 15 observations and the other 45 rendered frames are discarded.
Keeping sub-step frames **only during attack runs** would give ~4× the visual
detail of the one event the model is starved of, at near-zero step cost and
bounded buffer cost. This is the best remaining answer to data starvation.

### 6.5 Four unused buttons

MineRL exposes `drop`, `pickItem`, `swapHands`, `ESC`. `pickItem` and
`swapHands` are genuinely interesting later (offhand use; selecting the block
under the crosshair). Not added: an action nothing needs only dilutes
exploration — the lesson of §3.2.

### 6.6 A selection sense, if hotbar keys ever return

They cannot be learned while invisible. The honest route is grounding in
*consequence* — what changed when the agent used something after pressing a
key — not the highlight, which this engine does not draw.

### 6.7 Engine independence is already paid for

`sensors/`, `spatial/`, `slots/` and `world_model/` have **no MineRL
imports**. The heading convention is a declared value with a per-engine
known-answer table. When the external server returns, the adapter changes and
nothing above it does.

---

## 7. Honest limits

- No claim is made that any of this improves learning. Stage 5 asks that, over
  days.
- Slots have **no contract tests** — every claim worth asserting needs a
  working flow field first.
- Audio is untestable until the MineRL client is modded to expose it.
- The HUD sensor is **dead on this engine**, measured, not pending.
- The scoreboard has not moved: **13,305 blocks broken → 35 logs**, and
  chopping a tree has never been learned. That remains the only number that
  matters.
