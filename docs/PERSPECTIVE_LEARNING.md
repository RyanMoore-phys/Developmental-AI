# Perspective from motion

*2026-09-18. Wave 1 shipped; Wave 2 (object slots) is specced and gated.*

> **Depth is what your own movement reveals. Objectness is what moves
> together when you move.**

---

## 1. The problem, stated honestly

SkyBot saw a flat 128×128 RGB frame and had never been given a reason to
recover the 3D world behind it. It could learn that a patch of pixels predicts
another patch of pixels. It had no mechanism by which "that trunk is four
blocks away and that one is twelve" could become a fact it represents.

The scoreboard is the argument: **13,305 blocks broken → 35 logs**, all log
skills at 0/20, and a history of income earned by staring at the sky (96% of
the drive), sitting in a villager's trade menu (77% of income) and holding
attack at an unreachable canopy (96% of option activity, 0 logs).

### 1.1 The diagnosis that was wrong the first time

Recorded here because it is exactly the kind of plausible theory this project
has burned days on, and because the fix that follows from it is different.

**The wrong claim:** *"the encoder flattens its bottleneck, so spatial identity
is destroyed."* False. `CNNEncoder.forward` does `reshape` then `Linear`, and a
flatten+Linear is **position-specific** — each bottleneck cell owns its own
weights. Position was representable before this wave.

**What was actually wrong,** three separate things that the single wrong claim
had blurred together:

1. **The bottleneck was 4×4.** `_cnn_channel_ladder` drove *every* `image_size`
   down to a 4×4 grid. At 128px one cell covers a **32×32 pixel slab**. A
   mid-distance trunk lives entirely inside one cell, indistinct from the dirt
   behind it.
2. **The objective did not care about objects.** Reconstruction was a flat
   per-pixel MSE over the whole frame. Sky, ground and repeated texture own the
   error mass; a three-block trunk is rounding error. This is a documented
   failure mode of autoencoder world models, not a hunch — small objects
   produce error signals too weak to train an encoder to perceive them.
3. **Nothing downstream ever indexed position.** `configs/minecraft_skybot.yaml`
   says it outright under `symbol_weight`: *"POSITION-BLIND: a tree at the edge
   of frame and a tree dead-centre pay exactly the same."* `vision_scaffold.py`
   admits *"no instance disambiguation of two trees in one frame."*

Raising the input resolution fixes none of these. **The loss had to change.**

---

## 2. What this wave built

### 2.1 The flow head — the centrepiece

From the latent and an action, predict a dense displacement field. Warp frame
*t* through it. Charge the photometric error against frame *t+1*.

Nothing labels the flow. **Novel-view synthesis is the supervision.** There is
no separate optical-flow estimator, and nothing new runs on the env-step
critical path — the loss lives inside world-model training, on replay sequences
that already hold consecutive frames.

The consequence that matters: **under a forward translation, predicted flow
magnitude is inverse depth.** Near things sweep fast.

### 2.2 Why this and not "walk at it and watch it grow"

The intuitive version of this idea is behavioural: pick a blob, centre it, step
forward, watch its radius grow, call it *converged* when the radius stops
growing, and label anything that never converges a mover. The underlying
intuition is right — and it is what the flow head implements — but the literal
algorithm fails in four specific ways this project has already paid for:

| The behavioural version | Why it fails here |
|---|---|
| Colour-cluster → ellipse segmentation | **Declares** what an object is. Grass is one enormous object, a canopy is fractal, a trunk against dark leaves may not separate at all. Forbidden in spirit by `_no_scripted_skills_smoke.py`, and it degrades *silently* — the `places`-counter lesson. |
| "Centre, step, check radius, retarget" | A hand-written **policy**. It would be the most competent behaviour in the system on day one and PPO would learn to let it drive: "96% of option activity, 0 logs" in a new costume. |
| Pay for radial growth | The cheapest source of monotone radial growth is **a wall**. Approach → converge → retarget → repeat, forever, zero contact with the scoreboard. |
| "Mover = failed to converge" | A residual bucket. Camera noise, fog, foliage sway, chunk load-in, another SkyBot and your own bump all land in it. A mover needs a **positive** test. |

The flow head keeps the insight and drops all four. The counterfactual query
*"if I stepped forward from here, how fast would this sweep past me?"* is one
head forward — the same fact the radius test measures, available **without
moving**, learned rather than declared. And a mover is defined positively: flow
the action-conditioned warp **cannot** explain.

### 2.3 The five changes

| # | Change | Where |
|---|---|---|
| 1 | **8×8 bottleneck** (one fewer conv stage at 128px) + **CoordConv** input planes + a 1×1 readout | `world_model/rssm.py` — `_cnn_channel_ladder`, `CNNEncoder` |
| 2 | **`FlowHead`** + warp photometric loss (Charbonnier) + edge-aware smoothness | `world_model/rssm.py` — `FlowHead`, `flow_warp`, `compute_loss` |
| 3 | **Residual-weighted reconstruction**: weight recon by where the warp failed | `world_model/rssm.py` — `compute_loss` |
| 4 | **Four body senses**, unpaid, into proprioception | `core/developmental_loop.py` — `_flow_senses`, `_augment_proprio` |
| 5 | **Flow residual → learning progress** as a second error channel | `curiosity/learning_progress.py` |

Plus the plumbing that makes (2) possible: the world model now receives the
env's raw proprio vector (`pitch`, `head_sin/cos`, `moved`, …) in its embed, so
it can tell *"I turned my head"* from *"the world spun around me"* — the
discrimination a flow head in a first-person game cannot do without.

**Measured parameter cost** at live sizes (`image_size` 128, `encoder_hidden`
768, `latent_dim` 4352), not estimated:

| | before | after |
|---|---|---|
| encoder total | 4.12M | **6.71M** (fc 3.15M → 6.29M) |
| flow head | — | **5.35M** |

Nearly all of the flow head is the two Linears in front of its deconv stack
(fc 3.3M, proj 1.6M, deconv only 0.4M), so `flow_size` is *not* the expensive
knob — halving it saves ~0.3M and costs half the field's resolution. Narrow its
`hidden_dim` instead if it ever has to shrink. Against a 54M-parameter world
model on a GPU measured at 2% utilisation, both numbers are noise.

### 2.4 The four senses

All four come from one `FlowHead` forward on a latent the loop already
computed, plus one warp. They enter **proprioception and nothing else**, on the
doctrine `_reach_sense` already states: a felt affordance, not a rangefinder,
and not a payment.

| Sense | Means |
|---|---|
| `flow_fovea` | If I stepped forward, how fast would **what I'm looking at** sweep past me? — inverse depth on the attended thing |
| `flow_edge` | The same for the periphery. The centre/edge gradient **is** the geometry of heading: content toward the direction of travel barely moves, edges sweep hardest |
| `flow_ratio` | fovea / (fovea + edge). Figure-ground — above 0.5 means the attended thing stands *nearer* than its surround, i.e. an object rather than backdrop |
| `mover` | The fraction of the last frame-change the action-conditioned warp could **not** explain |

`_reach_sense` is **kept, not replaced**: it is ground-truthed by real break
events, and these are geometric complements to it.

> **A note on what was dropped.** The plan named a fourth sense, `impedance`
> ("I commanded motion and the world didn't move"). It was cut during
> implementation: `moved` already measures realised displacement, so the policy
> could compute it from inputs it already has, and the version that survived
> review needed an EMA-normalised magic constant to be meaningful. `flow_edge`
> replaced it — free from the same forward pass, and it is the centre/periphery
> gradient the original proposal actually asked for.

---

## 3. The property everything rests on

**No ego-motion → no predicted flow → nothing to be wrong about → no income.**

The residual is a *ratio*: photometric error after warping, over the error of
not warping at all. If the frame did not change, the denominator is ~0, there
is nothing to explain, and it returns exactly **0** — not "small", zero.

This closes the sky exploit **structurally** rather than with a guard. Flat sky
has no parallax and no texture for the warp to fail on. Standing still pays
nothing. Nobody has to remember to re-open anything.

It also matters *where* this enters. The residual joins **learning progress**,
which pays for error **going down**, never for error. Adding a channel changes
*what the agent can make progress on* — not how much it is paid for standing
anywhere. `last_pred_error` is deliberately left uncontaminated: it is
documented as raw forward-model surprise and is read by skill `wm_fidelity` and
the occlusion split.

### 3.1 Escape paths (CLAUDE.md §4.1)

Every guard added here states what re-opens it.

| Guard | Escape |
|---|---|
| `recon_residual_lambda` weighting | Bounded in `[1, 1+λ]`, **never reaches zero** — no region of the frame can become permanently unlearnable. `λ=0` is the exact revert. |
| `motion_floor` on the residual | Gates an **instantaneous** measurement. Re-opens on the very next frame in which anything moves. No counter, no human. |
| `valid` mask across resets | Recomputed every step from this step's `dones`/`restarted`. Cannot persist. |
| `flow_error_weight` | `0.0` disables the curiosity channel **without rebuilding the world model**. |

---

## 4. Research grounding

The citations exist so that when something fails we know whether we broke it or
inherited it.

| Finding | Source | What it decided |
|---|---|---|
| Ego-motion is a sufficient supervisory signal for visual features — rivalling class-label supervision | Agrawal, Carreira & Malik, *Learning to See by Moving* (ICCV 2015); Jayaraman & Grauman, *Learning Image Representations Tied to Ego-Motion* (ICCV 2015) | The premise. The agent's own movement is the teacher; no labels needed. |
| Depth + ego-motion are jointly learnable from monocular video by **novel-view synthesis** | SfMLearner / monodepth2 lineage; *Forecasting depth and ego-motion with transformers and self-supervision* (arXiv 2206.07435) | The warp-photometric loss **is** the estimator. No flow network on the critical path. |
| Curiosity from **optical-flow prediction error** over two consecutive frames | Yu et al., *Flow-based Intrinsic Curiosity Module* (arXiv 1905.10071) | Validates the residual as a drive. Its stated limit — needs motion to carry novelty — is why it *augments* rather than replaces. |
| Flow-based curiosity is still farmable by aleatoric noise (foliage, rain, other agents) | *Beyond Noisy-TVs: Noise-Robust Exploration via Learning Progress Monitoring* (arXiv 2509.25438); *Aleatoric Mapping Agents* (arXiv 2102.04399) | Route through the existing `LearningProgressCuriosity`, never raw error. |
| Reconstruction loss in pixel space makes small objects invisible; their error is negligible against texture | *Dreaming* (arXiv 2007.14535) and the JEPA line | §1.1 defect 2, and the reason for residual-weighted reconstruction. |
| Slot attention over **optical flow** (not pixels) is what makes object decomposition work on video; slots conditioned on *a single point on the object* | Kipf et al., *Conditional Object-Centric Learning from Video* (SAVi, ICLR 2022) | Wave 2's design. The "centre of attention" is SAVi's conditioning cue. |
| Pixel targets make slots latch onto colour/texture; reconstructing **self-supervised features** is what scaled them to real scenes | Seitzer et al., *DINOSAUR* (ICLR 2023) | Minecraft is maximally textured. A pixel-target slot model would segment grass noise. |
| Fully-unsupervised decomposition "still fails to scale to diverse realistic data" | SAVi, stated limitation | **Wave 2 is gated, not shipped.** |
| Object-centric latent dynamics works on relational control, but "broader evidence across benchmarks remains limited" | SOLD (ICML 2025); Object-Centric Dreamer; FOCUS | Do not bet the run on a slot-RSSM. |
| Embodied agents improve perception by moving and enforcing 3D consistency | *SEAL* (arXiv 2112.01001) | The long-run shape: movement as the perception curriculum. |

---

## 5. Wave 2 — objectness, behind a go/no-go

**Do not start this until both hold on live host data:**

1. `flow_loss` is measurably **decreasing** on live frames (not synthetic);
2. `flow_fovea` correlates with real contact events (break events / `_reach_now`).

Otherwise it is a slot model trained on noise, which is precisely the failure
SAVi documents.

Then: **SAVi-style slots** (K ≈ 6) over the 8×8 map, **decoding the flow field,
not pixels**, conditioned on the fovea centre and the peak of the mover
residual. Slots feed the **symbolizer's grounding head per slot** — which is
what finally makes "two trees in one frame" expressible and retires the
position-blindness. Slots stay **out of the PPO trunk** in that wave; that is a
separate input change with its own blast radius.

---

## 6. Verification

```bash
PYTHONPATH=. ./venv/bin/python tests/_perspective_smoke.py
```

Nine contracts, all synthetic or static — **no host run required**:

| | Contract |
|---|---|
| A | Zero-motion, zero income — identical frames pay *exactly* 0 |
| B | A pinned sense pays 0/step; the telescoping form kept as a regression witness (§4.3) |
| C | Sky/aperture — an unidentifiable field must not manufacture residual |
| D | Reset splice — a frame pair spanning two worlds reads 0, not "most interesting event in the run" |
| E | Both loop bodies (§4.2: senses=2, residual=2, cache=2) and a proprio width that matches its writer |
| F | Position is representable — same patch, two positions, different output |
| G | **Depth ordering is learned** — near plane's predicted flow exceeds far plane's after training |
| H | Config coherence — you cannot enable half of this |
| I | Body column optional, index-aligned, width-guarded, multi-stream, grows |

G is the one that matters. Measured: photometric 0.273 → 0.078, near/far flow
ratio **1.40×** on a scene whose true shift ratio is 4×. Ordered correctly,
under-separated — which is the honest reading of a small head trained for 400
steps on eight frames.

### 6.1 On the training host

- `flow` loss must **fall** on live frames. That is the Wave 2 go/no-go.
- WM block wall-time: expect +10–15%. **Revert trigger** if worse with no loss
  improvement.
- Step rate: **unchanged**. Nothing was added to the env-step path.

### 6.2 The falsifier

New telemetry keys `flow_loss` and `flow_residual` exist for exactly this.

> Plot per-step flow-derived income against `moved` and against `pitch`.
> **Non-zero income at `moved ≈ 0` means another wage for standing still.
> Positive correlation with looking up means the sky was rebuilt.**

Write that plot before trusting any of this.

---

## 7. What this wave deliberately did not do

- No colour-cluster→ellipse segmentation.
- No approach controller, and **no new reward term of any kind**. Change 5
  replaces an *input* to `intrinsic`; it adds no payment.
- No slots in the policy trunk.
- No change to `_reach_sense`, the magnet, or any existing channel's gating.

---

# Wave 2 — a correct photometric loss, and one measured retreat

*2026-09-18, same day. Wave 1's warp loss was a first cut; this makes it
correct, adds the geometry that was being guessed, and fixes the batch.*

## 8. A bug in Wave 1, found by asking the obvious question

`flow_warp` shipped with `align_corners=False` against a `linspace(-1, 1, n)`
base grid. **Those two conventions do not pair.** Under `align_corners=False`,
±1 are the *outer edges* of the corner pixels rather than their centres, so the
identity grid resamples at a half-pixel offset.

> Measured on a random 16×16 image: `max |warp(img, 0) - img|` was **0.490**
> with `False` and **0.000001** with `True`.

Zero flow was not the identity. So the photometric loss charged a floor of blur
error that **no field could remove**, and the head's cheapest route to reducing
it was a constant compensating offset — a bias on every depth reading. The
residual inflated the same way, since its denominator uses no warp at all.

**Contract A did not catch it**, and the reason is worth keeping: the
zero-motion test gates on `motion_floor` *before* the warp runs, so identical
frames returned 0 by the gate and never exercised the path. A guard hid a bug
from the test whose job was to find it. Contract **J** now asserts the identity
directly.

This also moved Wave 1's headline number: the two-plane near/far ratio went
**1.40× → 1.20×**, because the blur floor had been inflating both sides. The
contract asserts the *margin* (near > far), not the value — pinning the value
would make every legitimate change to the warp read as a regression.

## 9. What shipped, and on what evidence

Each item was measured on a two-plane synthetic before being turned on. Three
were proposed; **one did not survive its own test.**

| | Item | Evidence | Shipped |
|---|---|---|---|
| W2.2 | **Auto-masking** — drop pixels the warp makes *worse* | near/far 1.11/1.18/1.15× → **1.59/1.51/1.64×**, 3 of 3 seeds, 87% of pixels retained | **on** |
| W2.3 | **Multi-scale pyramid** | pyramid beat single-scale in **5 of 6** (shift, seed) pairs, lost one | **on**, weak evidence stated |
| W2.4 | **Longer baselines** | **ordering inverted** — see below | **off** |
| W2.5 | **`flow_mode: depth`** | geometry is exact by construction (contract O) | built, defaulted **off** |
| W2.6 | **Goal-replay threshold** | **33× dilution** removed — see §11 | **on** |

### The retreat: longer baselines make it worse

I recommended warping t→t+k for k ∈ {1,2,4} on classical-SfM reasoning — depth
resolution scales with baseline, adjacent frames are the worst baseline
available, and the pairs are already in every batch. On the two-plane scene
(near band moving 4× the far, truth 4.0×), across three seeds:

```
strides [1]      1.60x  1.28x  1.06x
strides [1,2,4]  0.86x  0.79x  0.84x   <- ORDERING INVERTED
```

Below 1.0 the head reports the **near** plane as moving *slower* than the far
one. Worse than no depth signal at all.

**Diagnosis, and it is mechanical rather than a toy artifact.** One head emits
one field per `(latent, action)`. At stride *k* the true displacement is *k*×
larger, while `max_flow` bounds the field — so a long-stride term can be
literally *unrepresentable*. An unlearnable term does not sit quietly: it drags
the shared head into a compromise that destroys the stride-1 field the agent
actually acts on. This is the CLAUDE.md §4.1 failure — *the guard made the
thing impossible* — one level down, in the tensor math.

**The fix was tried and is not enough.** Scaling the bound with the baseline
(`FlowHead.range_scale`, kept because it is correct regardless) gave
`0.79× / 1.16× / 1.46×`: better than broken, still worse than stride [1] on two
of three seeds, and wildly variable.

The capability stays in the code, config-gated and correct, and **off**.
Turning it on needs a head conditioned on the stride as a first-class input,
not one asked to infer it from a summed action. Contract **M** pins both the
representability relation and the live config, so nobody re-enables it without
reading this.

## 10. Geometry that is given rather than guessed

`flow_mode: depth` is a **parameterization of the existing head**, not a new
one. Both modes emit a field the identical warp/loss path consumes.

In a driving paper ego-motion must be regressed. Here it is essentially
**known** — and, crucially, **needs no new buffer column**: Δyaw, Δpitch and
forward distance are a *finite difference of the proprio Wave 1 already
stores*.

- Rotation-induced flow carries **no** depth information and dominates in first
  person. Translation-induced flow carries **all** of it. Asked for a raw
  2-channel field, the head must learn that separation from scratch and will
  mostly learn the rotation, because that is where the error mass is.
  Factoring makes it structural (contract **O**: rotation is depth-independent
  to float precision; translation gives near/far 9×).
- Yaw comes from the stored `(sin, cos)` **pair**, never a difference of
  angles — heading wraps, and `atan2` differences jump by 2π once per
  revolution. Contract **N** pins this with a 350°→10° case *and* the sign,
  because this repo has already shipped exactly that sign error once, in the
  episodic bearing, where the yaw=0 case cannot see it.

**One declared constant: FOV** (70°, Minecraft's default). Honest escape: a
wrong FOV rescales every depth *uniformly and preserves ordering*, and every
consumer today reads only relative magnitude — so it is invisible downstream
until someone wants a distance in blocks.

**Shipped as `raw`.** The Wave 1 go/no-go has not run. One config line flips it.

## 11. The batch that actually contains a log

`_break_reward` already separates perfectly — **log 20.0 + 0.5/tick, ore 10.0,
everything else exactly 0.0** — so the raw env reward is genuinely sparse and
log-shaped.

But the buffer does not store the raw env reward for the stream that matters.
Both loop bodies store `prim_extrinsic` for stream 0, which carries **magnet
shaping, approach and goal-dwell** on top, and those are nonzero on most steps.
Meanwhile `_find_reward_starts` was called with its hardcoded `1e-3` default.

> Measured on a synthetic buffer with shaping-sized reward every step and three
> log breaks in 400: at `1e-3` the pool was **396 of 396 possible windows** —
> goal-prioritised replay was performing **exactly uniform sampling**. At `5.0`
> it was the 12 windows containing a break. A **33× dilution** of the one
> mechanism that exists *because* this run's entire history holds ~35 logs.

`goal_replay_threshold: 5.0` is log-sized by construction, so the stratum means
what its name says however the reward channels are mixed. The `AsyncWM` line
now prints `reward pool: N windows` — that number should fall by orders of
magnitude the moment this ships, which is the live confirmation.

## 12. Still outstanding

**W2.7 — throughput — was not done.** `_phase_mark`/`_PHASES` already
accumulates per-phase wall clock and prints `Phase timing: … | UNACCOUNTED …`,
and we are at ~3.4 steps/s against a 10 steps/s ceiling with the config's own
finding that *our* Python is the binding constraint. But **there are no
`runlogs` on this machine** and the training host is tailnet-only, so the read could not
be performed. It remains the largest available multiplier and it needs no new
ideas — just that line, off the running system.

## 13. Verification

17 contracts, all passing, no host required:

- **A–I** — Wave 1, unchanged, now serving as the regression suite.
- **J** — zero flow is the identity warp (the bug above).
- **K** — auto-mask retains 100% at init (the anti-latch) and masks exactly the
  pixels the warp makes worse.
- **L** — every requested scale contributes; scale 1 *is* single-scale.
- **M** — `range_scale` widens the field with the baseline, and the live config
  is pinned to `strides [1]`.
- **N** — ego from proprio survives wrap-around and the sign is pinned.
- **O** — rotation depth-independent, translation ∝ inverse depth, zero ego →
  exactly zero flow.
- **P** — the reward threshold removes the 33× dilution.
- **Q** — **every switch reverts.** Wave 2 defaults reproduce Wave 1's loss to
  float equality. This one caught a test bug of its own: `compute_loss` is
  stochastic (the RSSM's straight-through categorical samples), so the two
  models must be seeded immediately before each forward or identical models
  disagree.
