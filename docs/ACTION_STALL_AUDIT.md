# SkyBot action-stall audit (2026-08-18)

> ## ⚠ STATUS UPDATE (2026-08-19) — THE STALL BROKE ON ITS OWN
>
> The headline below is **no longer true** and is kept only as the record of
> what was measured on 2026-08-18. After a further ~20 hours (uptime 1d 08h,
> 309 segments) the agent recovered without any of the Tier-1 fixes being
> applied:
>
> | | 2026-08-18 (stalled) | 2026-08-19 (now) |
> |---|---|---|
> | blocks broken | 12,709 frozen 122 segments | **13,305 and rising every segment** |
> | ticks-to-break | `NO BREAKS YET` | `dirt:0t, dirt:2t, dirt:20t` |
> | at pitch clamp | 62.1% of steps | **7.5%** |
> | GUI dwell | 49–73% | **32%** |
> | hand | `none` 244/244 | **`dirt`** (140 samples) — the mainhand fix works |
> | position | parked, cell (−4, 9) | moved ~86 blocks, cell (−4, −2), 551 cells seen |
> | total drive | +0.0041/step | **+0.0297/step (7×)** |
> | ledger concentration | HHI 0.55–0.62, magnet_seek 73% | **HHI 0.18**, magnet_seek 10% |
> | option invocations | 290 | **2,833** |
>
> **What this means for the analysis.** The most likely trigger was §1.4 —
> re-enabling `inventory` let it leave the villager's trade screen, after
> which it moved away and resumed normal activity. So the stall was
> dominated by the **one-way door**, not by the γ-bleed.
>
> **What still stands.** The γ-discounting defect (§0) is real, still live in
> four terms, and `persistence` still prints negative (−0.0001/step) — but it
> is now demonstrated to be a *drag*, not the cause of the stall. Priorities
> in §6 should be re-read in that light.
>
> **What has NOT changed: still 35 logs felled, and every log skill reads
> 0/20 in the mastery ledger.** The agent is busy and mobile again, but the
> actual objective is untouched. Whatever prevents chopping is separate from
> whatever caused the stall.

**Headline measurement (2026-08-18, now superseded): `blocks broken: 12709`
is IDENTICAL in all 122 segments of the current run. `Ticks-to-break: NO
BREAKS YET`. Zero blocks of any kind — not a log, not dirt — across ~125,000
steps.** Meanwhile every subsystem repaired over the last two days reports
healthy.

All numbers below are measured from the live pod, not inferred.

---

## 0. THE UNIFYING DEFECT — γ-discounted potentials bleed in every good state

Most shaping terms are potential-based, applied as

```
F = w · (γ·Φ' − Φ)          with γ = 0.99
```

That is the textbook Ng form and it is policy-invariant **over a full
trajectory**. But it has a property that matters enormously here: when Φ is
**held constant** at value c, it pays

```
F = w · c · (γ − 1) = −0.01 · w · c   EVERY step
```

A sustained *positive* potential is therefore a **continuous drain**. And
the drain scales with the weight:

| Term | weight | bleed at Φ=1.0 | as % of the total +0.0041/step drive |
|---|---|---|---|
| `persistence_weight` | 1.00 | **−0.01000/step** | **244%** |
| `symbol_center_weight` | 0.50 | −0.00500/step | 122% |
| `reach_weight` | 0.25 | −0.00250/step | 61% |
| `empowerment_weight` | 0.05 | −0.00050/step | 12% |

**Read the consequence carefully.** Every state we actually want SkyBot in —
sustaining an attack on a block, centred on a symbol, within reach of
something breakable — holds a potential *high*, and therefore bleeds reward
for as long as it stays there. The single state in which **every one of
these potentials is zero, and the bleed is therefore also zero, is looking
at empty sky.**

The sky is not being paid. It is the only place that isn't being taxed.

Corroboration from the logs: the persistence term — the one added
specifically to make sustained attack possible — prints **negative** in
practice (`Shaping: persistence -0.0004/step`, `-0.0001/step`). It is
charging the agent for the behaviour it exists to create.

This is the third and fourth instance of the same defect class found in two
days. Already fixed: `gaze_level` (was paying +0.00014/step to sit at the
clamp) and `gui_dwell`. Still present: persistence, symbol-centring, reach,
empowerment.

**Fix:** use the plain difference `Φ' − Φ` for any potential that can be
*held* — 0 while static, negative on loss, positive on gain, exactly 0 over
a cycle. Reserve the γ form for potentials that are inherently transient.

### 0b. Potentials structurally cannot discourage dwelling

Related, and it invalidates one of my own fixes. A telescoping potential is
**zero while the state is unchanged, by construction**. So it can make
entering/leaving a state neutral, but it can *never* make *staying* cost
anything. `pitch_level_weight` was corrected from "pays to sit at the clamp"
to "indifferent to sitting at the clamp" — which is not what is needed.
Discouraging a dwell requires a genuine per-step cost (a pure negative is
not farmable in reverse; the agent simply avoids it), in the family of the
existing `effort_cost: 0.0015`.

---

## 1. MECHANICAL — the action space itself

### 1.1 The pitch clamp is a no-op sink (PRIMARY behavioural cause)
Pitch is clamped to [−90°, +90°]. **At a clamp, `lookUP` is a no-op** — the
camera cannot go further, so the action changes nothing. Exactly **one of
thirteen** actions (`lookDOWN`, 7%) moves the view off the clamp; the other
twelve leave pitch untouched.

- expected dwell at a clamp ≈ 1/0.07 ≈ **14 steps**
- expected dwell at any interior pitch ≈ 1/0.14 ≈ **7 steps**
- predicted pile-up ≈ 2×; **measured: −90° visited 12,103 times vs 6,875 for
  the neighbouring bucket = 1.76×**
- policy is *balanced* (`lookUP 7% vs lookDOWN 7%`), so this is not a
  preference — it is the geometry of the action space
- result: **at a clamp 62.1% of steps**, `Looking: pitch=+90 <-- PINNED`

### 1.2 Nothing can put an item in the selected hotbar slot
`hand=none` in **244/244** samples. Three stacked causes:
1. `equipped_items` is dead in MineRL 1.0 (documented in-repo).
2. The inventory fallback only asked "is the SPAWN TOOL still in the bag?".
3. `start_tool: null` ⇒ that branch never ran at all.

Fixed to derive the hand from placement evidence — but **inert**, because
no placement ever occurs. The real blocker: `FlatInventoryObservation` is
item→count with **no slot indices**, nothing in `TREECHOP_MACROS` can change
the selected slot, and on a remote server `SimpleInventoryAgentStart` likely
never applies because the server owns the inventory. **Placement — and
therefore pillaring, crafting tables, and every build behaviour — is
unreachable by construction.** `use` is 4–11% of all actions and can do
nothing.

### 1.3 No standalone jump
Macro 2 is `{forward, jump}`. Jumping always moves forward, so a pillar
cannot be built under oneself. Pillar-jump escape is unreachable.

### 1.4 One-way doors in the action space (FIXED, invariant now pinned)
`disable_macros: [10]` removed `inventory` — the only action that CLOSES a
screen — while leaving `use` (11), which OPENS villager/chest screens.
SkyBot entered a wandering villager's trade screen and sat there **10,149+
consecutive steps** with no exit in its action space. Re-enabled;
`_gui_farm_smoke.py` now fails if any GUI-opening action is offered without
its closing action.

### 1.5 Sustained attack vs the tick budget
A barehanded log needs ~60 ticks. `action_repeat: 2` and the HOLD macro's
20 ticks make this reachable in principle (3 consecutive picks), but the
agent is barehanded permanently (1.2), so every break is on the slow path.
Measured: `attack streak max=134`, `swings ending with no break: 400`.

---

## 2. THE REWARD-GUIDANCE DECISION MAKERS

### 2.1 The drive is vanishingly small
`total intrinsic +0.0041/step`, of which **ICM/LP base is 88%**. All the
carefully-designed shaping terms together are ~12% of a drive that is itself
near zero. **In a near-zero-gradient landscape, behaviour is decided by
action dynamics (§1.1), not by reward.** This is why the mechanically-sticky
state wins.

### 2.2 `magnet_seek` dominates when it pays at all
Observed between **6% and 78%** of the ledger depending on segment, with
concentration (HHI) swinging 0.17–0.62. A single term oscillating between
irrelevant and three-quarters of all income is not a stable economy.

### 2.3 The seek nudge pays for NOT finding the goal
```python
if seek_forward_nudge > 0 and grounded and gp < 0.05 and nudge_left > 0
   and action in _nudge_actions:
    r += seek_forward_nudge      # RAW income, not a potential
```
This is flat income conditioned on the goal being **invisible** (`gp < 0.05`)
— precisely the sky-staring state. Its budget is refilled by the stuck-L1
remedy *and* by the advisor's `look_around`, so a stuck agent gets the
not-finding wage topped up repeatedly. Currently `seek_nudge_left=0`, so it
is exhausted rather than active, but the incentive shape is wrong.

### 2.4 Base curiosity rewards visual change, and the sky changes
Prediction error is a *proxy* for learning, and a poor one in a world with
clouds, water and particles. Measured earlier: the base paid **96% of the
drive** for watching clouds at the −90 clamp. Mitigated by
`icm_boring_discount: 0.85` (factor now 0.28–0.42), not eliminated.

### 2.5 The gaze-bucket novelty has permanently saturated
Designed to unstick clamps, decaying 1/√n, with the explicit intent that it
"unsticks a clamp and then gets out of the way." At **12,103 visits** to
−90°, it now pays ≈0.0005/step. It got out of the way permanently, and the
clamp it was meant to break is the one it can no longer break.

### 2.6 Reach is circular in evidence mode
`reach` is derived from *completed breaks* ("a completed break proves
something was within range"). With zero breaks, reach reads **0.000, in-reach
on 0.0% of steps, forever** — and the approach-to-reach potential (w=0.25)
is therefore inert. No breaks ⇒ no reach evidence ⇒ no approach gradient ⇒
no breaks.

### 2.7 Extrinsic is structurally unreachable right now
`log_break_reward: 20.0` and `log_tick_reward: 0.5` are large and correct —
but `seg_extrinsic = 0` every segment, because nothing breaks. The entire
extrinsic economy (weight 0.3 of the mix) is dark.

### 2.8 Familiarity decay may have closed the cheap on-ramp
`break_decay_scale: 25` and `break_habituation_scale: 50` discount
already-mastered block types. With 9,966 dirt and 2,052 grass already
broken, the cheap, always-available blocks now pay near zero — removing the
practice surface on which sustained-attack skill was originally learned,
while logs remain too rare to substitute.

### 2.9 Learning progress dies on a mastered view
LP is a derivative; on a static, well-modelled scene it →0 by definition.
A motionless agent staring at an unchanging sky is exactly the regime where
LP-based curiosity has nothing to say.

---

## 3. PERCEPTION

### 3.1 `player_visible` is unvalidated and effectively blind
**2 positives in 1,833 full-frame queries** (0.1%); 18 in 13,032 fovea
queries. `reliability = 0.7` — *exactly* the untouched prior, so it has
never once been validated against ground truth. We cannot distinguish "no
player was present" from "the head is blind". This blocks the social/peer
learning experiment entirely.

### 3.2 Peers never share a view
A new username is a NEW player, so peers spawn at world spawn while slot 0
resumes at its saved coordinates. `player_pos_labels` was byte-identical
across 7 consecutive segments. Malmo `<Placement>` cannot fix it (a vanilla
server places the player). Needs RCON teleport or a moved world spawn.

### 3.3 Fovea agreement is mediocre and drifting
0.76–0.87 against the full channel's 0.87–0.96, and it is the head the
magnet reads **every step** to decide where to look.

### 3.4 The `places` counter is garbage
`places` lists `iron_axe: 2281` (unplaceable) and `acacia_door: 3004` (never
owned). The heuristic — a `use` press plus any inventory count decreasing —
is picking up spurious inventory fluctuation across dozens of item types. It
must not be trusted as evidence of placement.

---

## 4. OPTIONS / SKILLS

### 4.1 Skills invoke but never succeed
`Option invocations: total=290` (break_birch_log 234, break_log_pickup 50,
break_oak_log 6) with `Mastery ledger: break_birch_log=0/20,
break_oak_log=0/20`. The skill layer is firing hundreds of times and
succeeding zero times — because of §1.1 and §1.2, not because of the skill
machinery. `Skills learned: 3 (0 mastered)`.

### 4.2 The policy is discarded every launch (by design)
`resume: REFUSING 'policy'`. Correct for escaping converged optima, but it
means no behavioural competence accumulates across the many restarts this
work required — only the world model, perception, familiarity and (now) the
magnet's curiosity memory carry over.

---

## 5. OPERATIONAL / INFRASTRUCTURE

### 5.1 Throughput is under half the achievable ceiling
`356 ms/step = env 147 ms (41%) + own Python 208 ms`, giving **2.81
steps/s**. The live server ticks at 20/s and `action_repeat: 2`, so the hard
ceiling is **10 steps/s**; the env round-trip has been measured as low as
98 ms (essentially the tick floor). **All remaining headroom is our own
per-step Python.** Ruled out by measurement: OMP thread count (1/2/4/8/16 all
within noise; per-step torch CPU work is ~3% of a step) and
`remote_step_delay_s` (already 0).

### 5.2 Client rebuild churn
**10 rebuilds in 121 segments**, against 1-in-76 on the best run. Each
rebuild is a rejoin and a discontinuity in the lifelong stream.

### 5.3 Peer count is bounded by the Minecraft server, not the pod
4 clients ⇒ 0 segments in 17 minutes, 9 rebuilds, `TimeoutError` on the
trainer↔java socket — while the pod was idle (bridge 0.3% CPU, GPU 0–13%,
load 4/48, distinct core pins). 2 clients are stable. Also: remote peers are
*cheaper* than local scouts (generating a world costs more than receiving
chunks).

### 5.4 Graceful stop does not work during boot
The STOP file is checked once per segment, so it is inert while clients are
still booting. With several clients, boot outlasts any reasonable wait and a
SIGKILL is required, losing everything since the last checkpoint.

### 5.5 `pkill -f "tailscale nc <ip>"` also kills the bridge
socat's own command line contains `EXEC:tailscale nc <ip> 25565`, so that
pattern destroys the 25565 listener. Use `pkill -x socat` or match on
`envPort`.

---

## 6. RANKED FIX LIST

**Tier 1 — explains the stall directly**
1. **Convert held potentials to the plain difference** (persistence 1.00,
   symbol-centring 0.50, reach 0.25, empowerment 0.05). Removes a drain of
   up to 244% of the entire drive from exactly the productive states. (§0)
2. **Add a genuine per-step cost for sitting at a pitch clamp** — not a
   potential, which cannot discourage dwelling. ~0.001/step against a
   0.0041 drive. Verifiable against the gaze-bucket census: −90°'s share
   should fall toward ~7.7%. (§0b, §1.1)
3. **Give the agent a way to hold a block** — hotbar-select capability, or
   establish why picked-up items never reach the selected slot. Unblocks
   placement, pillaring and crafting simultaneously. (§1.2)

**Tier 2 — removes perverse incentives**
4. Make the seek nudge conditional on *progress* rather than on the goal
   being invisible. (§2.3)
5. Break the reach circularity — allow a non-evidence reach signal so the
   approach gradient can exist before the first break. (§2.6)
6. Re-open a cheap practice surface (revisit `break_decay_scale` /
   `break_habituation_scale`) so sustained-attack skill has somewhere to
   grow. (§2.8)

**Tier 3 — throughput and hygiene**
7. Profile the 208 ms of own-Python per step (the only remaining throughput
   headroom). (§5.1)
8. Validate `player_visible` against known-present frames before trusting
   any social result. (§3.1)
9. Co-locate peers (RCON teleport or move world spawn). (§3.2)
10. Investigate the rebuild churn. (§5.2)

---

## 7. WHAT IS ACTUALLY HEALTHY

Recorded so this is not read as a system-wide failure — these were measured
working in the same run:

- **The VLM unstuck advisor** (user's design): 32 interventions, every one
  validated, sensible varied reasoning (`prime [tree_visible] — The tree
  visible in the distance might offer resources and new areas to explore`).
- **Magnet curiosity persistence**: restores 11/11 steerable categories
  across restarts; first time in project history curiosity survives a
  process restart.
- **Magnet arming**: `w=1.0000, target=stone_visible` (was permanently
  `w=0 / target=None`).
- **GUI escape**: dwell 73% → 49% and falling, `INVENTORY` at 10% of actions,
  max dwell 10,149 → ~200 steps.
- **Skill-delta compression** on GPU (device-mismatch fixed; verified
  round-trip error 2e-4).
- **World model**: `h_evolve` 0.45–0.76, WM error ~0.004; symbol grounding
  agreement 0.87–0.96 with 29/29 predicates grounded.
- **Loop timing instrumentation**, so throughput is no longer invisible.
