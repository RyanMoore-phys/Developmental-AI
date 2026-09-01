# Current state (2026-08-10): POD TERMINATED — everything rescued locally

The pod was deliberately stopped and terminated on 2026-08-10. Everything
non-reproducible was pulled to `pod_repository/data/final_rescue_20260810/`
(see its README for contents, verification, and resurrection steps). The live
brain, all brain archives, final checkpoints, the lifelong event-mastery
memory, every run log and all videos are preserved and hash-verified.

To bring SkyBot back: docs/RECREATE.md + the final-rescue README's restore
block. The tailnet identity and ollama models are re-creatable; the RunPod
connection details in these docs are STALE by definition now.

---

# Current state (2026-07-22)

## Brain contents (rescued in `../data/`)

**Goals** — `broadcaster_state.json`, 29 slots, broadcaster episode counter 74.
7 grounded to real block types (the rest are ungrounded pixel-signature
discoveries `discovered_N`):

| Goal | slot | lifetime unlocks |
|---|---|---|
| break_dirt | 0 | 38 |
| **break_oak_log** | 6 | **18** |
| break_grass_block | 16 | 11 |
| break_jungle_log | 21 | 1 |
| break_gravel | 25 | 1 |
| break_spruce_log | 26 | 1 |
| break_birch_log | 27 | 1 |

**Skills** — `registry.json`, 32 records. Grounded: break_birch_leaves,
break_oak_leaves, break_dirt, break_grass_block, break_jungle_log, break_gravel,
break_spruce_log, break_birch_log (+ ~19 ungrounded `discovered_N`). **None
mastered** — success rates 0.0–0.5 (the log skills ~0.003). This is expected:
the agent has *discovered* log-breaking (4 tree species!) but not yet
*consolidated* it. Each skill dir holds a `policy.pt` PPO snapshot.

**Headline:** the developmental premise is working — curiosity + magnet led the
agent to discover and ground log-breaking across four tree species with zero
demonstrations. The open problem is consolidation (turning rare discoveries into
reliable, mastered skills), which the audit fixes target.

## Code / deployment status

- **Wave 1 — DEPLOYED & proven on the pod:** tree-seeking drive, the frozen-h
  cross-segment RSSM carry fix, `h_evolve` liveness metric, per-episode reward
  print. Verified live: `h_evolve` 0.49–0.64, primary stream chopped a log
  (ep-2 reward 6.15), 3 new tree species grounded.
- **Waves 2 + 3 — BUILT, smoke-tested, STAGED on the Mac, NOT yet deployed.**
  The pod run is stopped. These need: sync → single relaunch.

## Audit fixes staged (not deployed) — see `../../AUDIT_FINDINGS.md`

A full-system audit produced 38 findings (workflows died on account limits before
the verify stage; findings are code-cited). Fixed inline + smoke-tested
(`_audit_criticals_smoke.py`, `_audit_wave3_smoke.py`, all green):

**3 criticals:** (1) crash-rebuilt MineRL client wedged its stream on a frozen
frame forever → bounded reset in the ADVANCE block; (2) minted skills never bound
into option slots in lifelong → `refresh_slots` per-segment + evict-for-mint;
(3) minted skill = frozen first-unlock snapshot → competence-triggered re-distill.

**Wave-3 highs/mediums:** seek-nudge economics (pure-forward, cold-start budget
that wanes so static stone stops paying), tree-reliability collapse trap
(freshness-gated scoring + regression-to-prior), LP noise-farming (significance
gate + std floor), exploration_ratio keyed to MASTERED skills (was lifetime
count), log-pickup goal mis-keying guard, competence shared-bias frozen
(decouples slots), clear_stream evidence banking, force_exploit clock 200k→800k,
first-break horizon alignment, log-count death re-baseline, `_prev_mine` clear,
`estimated_dag` ragged-width guard, `novelty_window` wiring.

**Deliberately NOT fixed** (unverified / invasive — documented in AUDIT_FINDINGS):
PER sampling perf, terminal-fraction degeneracy, scout-blind Bernoulli
attribution, paging protection for log goals, dream-unreachable honesty, and a
few lows.

## What's next (recommended order)

1. **Deploy waves 2+3:** sync code → `bash scripts/launch_lifelong.sh 1000000`
   on a **GPU** pod → watch `h_evolve`, per-episode reward, `tree=` prob, log
   unlocks, and confirm no wedged streams (the critical-#1 fix).
2. Watch whether the seek/economics fixes stop the primary stranding off-tree.
3. Watch for the first *mastered* log skill (competence climbing on
   break_*_log), now that skills rebind + re-distill.
4. Later: the deferred audit findings, and the deferred lifelong items
   (P2 clock swaps, P3c policy reshape + skill rebind-on-recall).

## Key invariants (do not violate)

- NEVER wipe `skill_bank_mc_curiosity/`.
- The reward mixer anneals naturally — never hard-fix the intrinsic/extrinsic ratio.
- env 0's lifelong stream stays continuous.
- Free/open-source/local tools only; LLM via local Ollama (no paid APIs).
