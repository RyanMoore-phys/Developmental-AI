# The plurality roadmap — what shipped, Phases 1–7

*2026-09-19. Companion to `docs/PERSPECTIVE_LEARNING.md`, which covers the
two waves that came before and is the argument this builds on.*

---

## 0. The question this answers

> *Is SkyBot even close to capturing Minecraft's complexity? It seems quite
> binary, in perception and in action.*

It was, and the specifics were worse than the impression:

| | before | Minecraft |
|---|---|---|
| Symbols | 25 **boolean** predicates | ~1000 block types, states, relations |
| Actions | **13** macros, camera in ±15° quanta only | continuous mouse, ~20 keys, GUI, hotbar |
| Modalities | RGB 128² + 13 proprio scalars | vision, **audio**, HUD, damage, particles |
| Spatial memory | none — every representation per-frame | a running map of where things were |
| Instances | none | arbitrarily many |

The grounding head emitted `tree_visible=1, object_left=1`. That **cannot
express** *"two trees — one at 5 blocks partly behind leaves, one at 15."*
No quantity, no instance, no relation. Everything downstream inherited it.

**The fix was not more pixels.** It was more *structure per dimension*, plus
an action space able to express the nuance perception now carries.

---

## 1. The rule every item obeys

> **A SENSE is a transducer. It reports a physical quantity the agent's own
> body or its own screen makes available, without naming what anything IS.
> A MEANING names, classifies or identifies.**
>
> **Senses may be policy input. Meanings may only ever be evaluation.**

### "Player-perceivable" is necessary but NOT sufficient

A player *can* press F3 and read `minecraft:oak_log`. That string is
player-perceivable and still forbidden. **The thesis is not "only what a
player can see" — it is "meaning is earned, never declared."** A human
reading `oak_log` already knows English and already knows what a log is; the
string indexes knowledge earned elsewhere. SkyBot has no elsewhere.

F3 splits exactly along this line:

| F3 column | Contents | Ruling |
|---|---|---|
| **Left** | XYZ, facing, velocity, light, biome, time | mostly **SENSE** |
| **Right** | **targeted block name**, fluids, entity names | **MEANING** — evaluation only |

Two left-column exceptions, ruled explicitly:

- **Absolute XYZ is GPS.** A player without F3 has dead reckoning, not
  coordinates. Refused as input; used as the oracle that *measures* the
  dead reckoning (§4).
- **Biome ID is a free clustering** of terrain the agent should earn from
  pixels. AMBER, default off, and not enabled.

### Classification, enforced structurally

- **GREEN** — a pure sense. Policy input.
- **AMBER** — leaks structure the agent should earn. Opt-in, default off.
- **RED** — a meaning. **Evaluation only**, and `SensorBus.read_policy`
  never *visits* a RED sensor. There is no filter to forget.

---

## 2. The sensor bus (S0) — the enabling mechanism

`developmental_ai/sensors/`. Adding a sense is now a **registration**, not a
schema migration against the replay buffer, the world model and both loop
bodies.

| Property | Why |
|---|---|
| **Layout hash** over `(name, width, version, kind, shape)` in order | Width alone is not enough — two sensor sets can match in width and mean entirely different things field-for-field. Mismatch restores **neutral**, never misaligned. |
| **Optional on load** | Follows the `restarts` precedent. The host's buffer is the only copy of the agent's experience; a column that refuses to load bricks it. |
| **`proprio` stays first at offset 0** | Every row written before the bus existed still lines up. |
| **Neutral is not always zero** | A ratio whose midpoint is 0.5 reads 0.5 when absent. Zero would be a *claim*. |
| **Vector vs image kinds** | The foveal crop is 3,072 numbers against 13 body scalars. Flat concatenation would drown the body at 236:1, so image sensors get their own conv encoder. Live: **4,144 transport → 176 encoded.** |

---

## 3. Phase 1 — make the pixels carry what a player sees

| | Item | Cost | Note |
|---|---|---|---|
| **A1** | Crosshair fovea at **native** resolution (32×32 of the 384px frame, cut before the downsample) | 3,072 | The block-breaking **crack overlay**, the **in-reach wireframe** and break **particles** all live below the 128px floor. Measured: a small central feature survives the native crop at full contrast; the downsample leaves **13 of 120**. |
| **A3** | Foveal temporal difference | 1,024 | The crack overlay is an *animation*. |
| **A4** | Screen effects (damage vignette, underwater, fire) | 6 | Channel *excess*, not raw redness — otherwise it fires on a desert sunset. |
| **A5** | Finer camera quanta: ±5°, ±2° appended | — | See §6. |

---

## 4. Phase 2 — the body's own geometry, and the oracle

**G3 · dead reckoning** (16 floats). Integrates the agent's *own sensed*
body-relative displacement from an episode-local origin. Encoded at four
scales (4/16/64/256 blocks per cycle) — raw (x,z) is unbounded and gives a
linear layer no reason to treat nearby positions as similar. Grid-like
multi-scale codes *emerge* from training a network to path-integrate
(Banino et al., Nature 2018).

**G4 · motion** (5 floats). `moved` was a magnitude, so nothing could tell
walking forward from being shoved sideways or pressing `back`.

**O1 · the oracle** (RED). `true_position` registered so the project can
finally **measure itself**, never so the agent can use it. Reports
dead-reckoning drift in blocks *and as a percent of path travelled* — the
scale-free form.

> **The design decision that made this real.** My first instinct was to
> integrate the engine's position deltas. That would have been GPS with extra
> steps: zero drift, and an oracle measuring nothing. Integrating only what
> the body senses is what makes the measurement mean something. An unsensed
> 0.5-block/step sideways push shows as **5.0 blocks of divergence over 10
> steps**.

---

## 5. Phase 3 — structure from data already computed

**G5 · light** (4). p05/p50/p95 + unlit fraction. Deliberately *not* mean
brightness — a cave mouth in a sunlit field is a low p05 beside a high p95,
and their average is an ordinary afternoon.

**G6 · sky** (4). **A correction to the roadmap.** The item said
"time-of-day phase (sin, cos)". That is *not derivable from one frame* —
dawn and dusk look nearly identical, and a player separates them by which
horizon the sun is on plus memory. Shipping a phase would have been a
fabrication. It reports appearance and claims nothing; the clock is left for
the RSSM to integrate. The contract pins **dawn == dusk**.

**P4 · spatial surprise map** (64). Already computed inside `compute_loss`
and averaged into a scalar. Curiosity had been scene-level since the project
began: the agent could be surprised, never surprised **by a place in the
frame**.

**P1 · multi-horizon latent prediction** (`[4, 16]`). The model was trained
one step ahead while the policy trains on **15-step imagined rollouts** — the
horizon the dreams live at was never supervised.

---

## 6. Phase 6 — action plurality

**13 → 28 macros**, indices 0–20 frozen, vocabulary 8 → 12 buttons.

The arithmetic that forced A5, at Minecraft's 70° FOV:

| distance | target subtends | old 15° step | new 2° step |
|---|---|---|---|
| 2 blocks | 28.1° | 0.5× | 0.07× |
| 5 blocks | 11.4° | 1.3× | 0.18× |
| **10 blocks** | **5.7°** | **2.6× the target** | **0.35×** |
| 20 blocks | 2.9° | 5.2× | 0.70× |

Approach carried **±7.5° of unavoidable aim error** — ±1.3 blocks of lateral
drift at 10 blocks — and every close-range correction swung past the trunk.
A limit cycle, and a better mechanical account of *13,305 breaks → 35 logs*
than any perception story. VPT uses mu-law quantised camera bins: fine near
zero, coarse far out. 15/5/2 is that shape.

Also added: **strafe** left/right (orbiting a tree was not expressible at
all, since turning and moving are coupled through the heading), sprint,
sneak, standalone jump; **hold durations 10/40/120** (one fixed 40 was either
far too long for an axe or two thirds too short for bare hands).

**A9 (hotbar keys) was proposed, built, and then REMOVED** — 28 macros, not
37. A pre-existing contract already carried *"still no hotbar keys
(deliberate: minimal exploration burden)"*, and Stage 0 made that decision
**more** right rather than less: MineRL does not render the HUD, so the agent
cannot see its selection change, and it has never held two things at once.
Nine invisible actions over an empty inventory is exploration burden bought
with nothing. See §13 and the block in `minerl_env.py`.

**A boot-time guard was added with them.** `_macro_to_action` does
`act[k] = v` on `action_space.noop()`, so a key the engine lacks is accepted
silently and dropped at the socket — the macro looks like it fired and the
policy learns a button that does nothing. `_validate_macros` now crashes at
construction instead.

**A6 (factored camera head) was NOT built.** The plan gated it on A5's live
measurement, and it would replace the action head with a Dict head, breaking
the append-only action-index contract every saved skill depends on.

---

## 7. Phase 4 — representations that outlive the frame

`developmental_ai/spatial/`. Built from the agent's **own estimates** — the
world model's forward-probe magnitude and the body's sensed ego-motion.

| | Item | Cost | |
|---|---|---|---|
| **G1** | Forward-probe sweep map | 64 | Works in **both** flow modes, which is how Phase 4 shipped before the depth gate: under `depth` the magnitude *is* inverse depth; under `raw` it is what the head has learned. Either way the ordering is the useful part. |
| **G2** | Egocentric occupancy | 256 | A **memory, not a survey** — an unobserved cell fades rather than asserting emptiness it cannot vouch for. |
| **P2** | Successor map over **dead-reckoned** cells | 64 | Not egocentric: that frame re-centres every step, so every state would be its own successor. |
| **P3** | Walkability by bearing | 8 | **Annotation by interaction** — nobody labels a wall; the wall labels itself by stopping you. Prior 0.5, because zero would claim the world is solid, and that is the belief that stops an agent trying. |

---

## 8. Phase 5 — object slots (code only, gated)

`developmental_ai/slots/`. Three choices, each from a published failure:

1. **Slots decode the FLOW FIELD, not pixels.** DINOSAUR found pixel targets
   make slots latch onto colour and texture. Minecraft is maximally textured
   — a pixel-target slot model here would confidently segment grass noise.
   SAVi showed flow is the target that works, because *what moves together
   is an object* — this project's own thesis one level down.
2. **Slots are conditioned, not randomly initialised.** SAVi's central result
   is that one point per object suffices and unconditioned init does not
   scale. Two such points are already computed and neither is a label: the
   fovea centre, and the peak of the mover residual.
3. **Identity is separate from appearance**, updating only while present
   (Dual-State Slot Attention; TSA). Otherwise slots *swap* under occlusion —
   and an agent walking through a forest occludes constantly.

**R2 · relations** — 30 ordered pairs × 5 fields: Δbearing, Δdepth,
occlusion ordering, size ratio, co-motion. **This is the sentence that could
not be said.**

**Shipped off, structurally.** `WorldModel` refuses to build the module
without a flow head. Until `flow_loss` is shown to fall on live frames there
is no reconstruction target and this is a slot model trained on noise.

**R3** is in the symbolizer: the *same* head class and vocabulary, only the
input changes from "everything I can see" to "this one region that moved
together". Label routing is **deliberately unbuilt** — attributing a VLM
label to whichever slot owns the evidence needs live calibration.

---

## 9. Phase 7 — audio

**MineRL does not expose audio.** Its observation space is POV + inventory +
stats; there is public evidence of people trying to add it to MCP-Reborn and
no shipped handler. Making this real means **modding the Java client**.

The Python half is done and waiting: an image sensor (a spectrogram is a
picture of sound) that reads `info["audio"]` and returns neutral every frame
until the key exists. The contract the Java side must meet is written at the
sensor: `(2, 32, 16)` float32 in `[0,1]` — stereo because direction is half
the information, mel-scaled because raw 44.1 kHz is ~7,000 numbers per step.

---

## 10. What is enabled right now

```yaml
sensors.enabled: [proprio, screen_fx, fovea_native, fovea_delta,
                  dead_reckon, motion, true_position, light, sky]
# registered, NOT enabled:  hud (HUD-in-POV unverified), audio (no handler)
spatial.enabled: true          # G1 G2 P2 P3
slots.enabled:   false         # gated on the flow go/no-go
world_model.flow_mode: raw     # gated on the same
world_model.latent_horizons: [4, 16]
```

**Transport:** 4,144 floats → **176 encoded** at the RSSM.
**Policy proprio append:** 4 flow senses + 64 surprise + 392 spatial = **460**.

---

## 11. Bugs these changes surfaced

Recorded because each was found by a test rather than by reading, and each
would have been silent:

| Bug | Symptom it would have had |
|---|---|
| `flow_warp` used `align_corners=False` with a `linspace(-1,1,n)` grid | **Zero flow was not the identity** (max err 0.490). The loss charged a blur floor no field could remove; the head's cheapest escape was a constant offset — a bias on every depth reading. |
| `compute_kl_loss` free nats = 0.1 | The horizon measurement was **pinned at its own floor** (0.1100 for every config, all seeds). Fixed with an unclamped `prior_divergence`: 16-step rollout error **0.066 → 0.0002**. |
| `charbonnier(0) = 1e-3` | A **static** cell computed `1e-3/1e-3 ≈ 1.0` surprise. The scalar was sheltered by `motion_floor`; the per-cell map was not — manufacturing curiosity out of stillness. |
| Occupancy re-registration translation sign | A remembered wall **receded** as the agent walked into it. The intuitive phrasing ("the world slides backwards") gave the wrong sign; backward-mapping already carries the inversion. |
| Bilinear rotation does not conserve mass | Confidence could **compound above 1/decay** until every cell read solid — the worst failure for a walkability signal. Now clamped. |
| Goal replay threshold `1e-3` against shaped rewards | The stratum selected **396 of 396 windows** — exactly uniform sampling, on a run whose history holds ~35 log breaks. |
| Unknown macro keys | Accepted silently and dropped at the socket; the policy learns a button that does nothing. Now a boot-time crash. |

---

## 12. Engine independence

**MineRL is a stand-in**, in use only while the external Paper server is
offline. Nothing in the new code is allowed to assume it.

Audited 2026-09-19. The result:

| Module | MineRL coupling |
|---|---|
| `sensors/` | **none** — every sensor reads a plain context dict (`world`, `info`, `pov_native`, `prev_fovea`) and works on any RGB frame and any body dict |
| `spatial/` | **none** — numpy only |
| `slots/` | **none** — torch only |
| `world_model/` | **none** |
| `environments/minerl_env.py` | by definition; this is the adapter layer and the right place for it |

Two real couplings were found and removed:

1. **The heading convention was baked into the maths.** Which axis yaw 0
   points along, and which way the angle grows, are facts the *engine*
   chooses — and Minecraft's answer was written as a sign in three separate
   functions. It is now a **value** (`sensors/heading.py`), declared by the
   adapter, with a known-answer table per engine. Adding an engine is one
   table entry.

   > Reasoning from the phrase *"yaw grows clockwise"* produced the **wrong
   > sign** on the first attempt — yaw 90 came out facing east instead of
   > west. Whether a rotation reads clockwise depends on which way you draw
   > the axes. That is why `heading_cases()` exists and why the unit suite
   > runs it for every registered convention.

2. **`PROPRIO_LAYOUT_MINERL`** — an engine-named constant in the world
   model. Renamed `FALLBACK_PROPRIO_LAYOUT` and demoted to what it actually
   is: a convenience so a bare `WorldModel` can be built in a test. The live
   path derives the layout from the adapter's own `PROPRIO_KEYS`.

`ego_from_proprio` needs **no** convention argument, and that is not an
oversight: it returns the rotation *between* two stored headings, which is
the same number under any convention — the convention cancels. `DeadReckoner`
does take one, because it must turn a heading into a direction in the world.

---

## 13. Stage 0 results — MEASURED 2026-09-19 on an RTX 2000 Ada training host

Three of the five open questions are now closed, against real MineRL.

| | Question | Answer |
|---|---|---|
| 0.1 | Does MineRL boot and step? | **YES** — built in 123 s, 1.75 steps/s at one client |
| 0.2 | Are the new macro keys real? | **YES** — all 20 macro keys present in a 24-key space |
| 0.3 | Is the HUD in POV? | **NO** |
| 0.4 | Any audio handler? | **NO**, as expected |

**0.2 — the action space is the full human keyboard.** Measured:

```
ESC, attack, back, camera, drop, forward, hotbar.1 … hotbar.9,
inventory, jump, left, pickItem, right, sneak, sprint, swapHands, use
```

So A5/A7/A8/A9 ship whole — nothing is dropped at the socket. It also shows
**four buttons still unused**: `drop`, `pickItem`, `swapHands` and `ESC`.
`swapHands` and `pickItem` are the interesting ones (offhand use, and
selecting the block you are looking at); they are not added here because
nothing yet needs them, and an unused action only dilutes exploration.

**0.3 — A2 IS DEAD, and this is a permanent finding rather than a pending
question.** The bottom band's pixel variance is **0.029 against 0.290**
mid-frame: an order of magnitude *less* structure where a rendered HUD would
be an order of magnitude more. MineRL's POV is world-only.

Consequences, stated plainly:

- the `hud` sensor stays disabled forever on this engine;
- **A9's hotbar keys work, but the agent cannot SEE its selection change.**
  It can press `hotbar.4` and the game will honour it; the only evidence
  available is the indirect kind already in use (`EquippedItemObservation`
  and `_last_placed_item`). Deliberate equipping is therefore *possible* but
  not yet *observable*, and any skill built on it has to be grounded in
  consequence rather than in seeing the highlight move.

## 14. What is still unverified
1. **The flow go/no-go**: does `flow_loss` fall on live frames? Gates
   `flow_mode: depth` and all of Phase 5.
2. **Whether aim was the binding constraint.** The arithmetic is sound; the
   oracle answers whether it dominates.

See `docs/TESTING_PLAN.md` for how each of these gets settled.
