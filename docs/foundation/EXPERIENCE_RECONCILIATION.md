# Experience stores: reconciliation (Stage 5, item 1)

Plan Stage 5 item 1 says to work out what replay, episodic memory and
long-term storage are each responsible for before adding another store. Code:
`developmental_ai/foundation/experience/`. Tests:
`tests/unit/test_foundation_experience_unit.py` and
`tests/_foundation_experience_smoke.py`.

## What already exists

| Store | File | Owner (constructed by) | What it holds | Retention | Training source? |
|---|---|---|---|---|---|
| Replay buffer | `world_model/replay_buffer.py` (`ReplayBuffer`, `MultiStreamReplayBuffer`) | `DevelopmentalAI.__init__` → `self.replay_buffer` (`core/developmental_loop.py` ~730) | Flat obs (pixels, optionally uint8), action, reward, done, restart flag, proprio, PER priority. All in RAM block arrays, one per stream. | Circular. Oldest rows are overwritten at `buffer_capacity`, and growth is capped by `buffer_growth` (2 GB on `main`). `save()` keeps the newest `max_transitions`. Priorities change in place. | **Yes. It is the only one.** World model and PPO sequences come from `sample_sequences`. |
| Long-term store | `memory/long_term_store.py` (`LongTermStore`) | Loop `self._goal_ltm` (~1487), only when `lifelong.paging.enabled`. It is attached to the broadcaster. | A recall index for skills, goals and facts: cue vector, value, usage, active/dormant flag. It holds no payloads. | Disk JSON index, written atomically. Unbounded: items are forgotten only explicitly. Mutable (touch, set_active). | No. It pages items in and out of working memory. That affects policy input, not gradients. |
| Episodic working memory | `core/episodic_memory.py` (`EpisodicWorkingMemory`) | Loop `self.broadcaster` (~1415), only when `knowledge_source == "episodic"` | A 14-d feature giving where the key, door and goal were last seen. MiniGrid only; on any other environment it is a no-op. | RAM. Reset every episode. | No. It is a policy-input feature (the Rung 6 broadcast). |
| Episodic event memory | `infra/episodic.py` (`EpisodicEventMemory`) | `infra/stack.py` `_mk_episodic` | Typed events (kind, subtype, step, position, salience) for recency, proximity and bearing queries and log reports. | RAM `deque(maxlen=4000)`, oldest evicted first. Lost on restart. | No. It records, answers queries and reports only. |
| **Evidence store (new)** | `foundation/experience/store.py` (`EvidenceStore`) | Offline tools and tests only. **The live loop never constructs it** (plan §8, Offline/Shadow). | Immutable Observation, Action (with actual timing), Prediction (written *before* its outcome), Evidence and episode-end records. Revisable interpretations are kept separately. | Disk: an fsynced journal plus gzipped sealed chunks with sha256 in a manifest, under a `max_bytes` budget. The oldest unpinned chunks are evicted first. Held-out data and promoted-mechanism evidence are pinned, up to a capped share of the budget. | **No.** It is the record of record for evaluation and scientific comparison. Offline fitting reads only `learning_view()`, which is the dev split. |

## Why a new store, and why it is not a second replay

None of the existing stores can serve as the record that results are checked
against:

- The replay buffer is built for gradient throughput. It overwrites in place,
  changes priorities in place, stores no prediction, no model version and no
  real action timing, and does not separate a reset from a client recovery
  except through a restart flag.
- The long-term store and the two episodic memories hold derived summaries,
  not raw experience.

The evidence store therefore takes exactly the job none of them does:
**an immutable, verifiable record of what was observed and done, what a
model predicted beforehand (and which snapshot made the prediction), and
what then happened.** Comparisons in plan §7, held-out evaluation, and the
Stage 5 completion gate (reconstructing a selected interaction with its
original prediction) all read from it.

Training ownership does not move. The replay buffer stays the only training
source for the live learner. Nothing in `core/developmental_loop.py` reads
or writes the evidence store, so it cannot change live replay, reward or
normalisation (plan §8). An offline consumer that fits parameters from
stored experience, such as a Stage 6 mechanism, reads through
`learning_view()`. That view holds the dev partition object and nothing
else, so it cannot reach held-out data (`SplitLeakError`) or evaluator data
(`PrivacyError`).

## Decisions worth knowing

| Decision | Reason |
|---|---|
| Partition is chosen at write time: `dev` / `heldout` by `runtime.splits` episode key (`"<env>:<stream>"`, episode id), and `evaluator` by provenance | Privacy then comes from structure, in the same way that `SensorBus.read_policy` never visits a RED sensor. A filter would be something a consumer could forget to apply. |
| The store's clock is the global write sequence, not `t_wall` | "Prediction before outcome" is checked against the order in which records reached disk. A clock the producer controls could be wrong. |
| Evidence may only cite observations and actions this store already recorded, and they must be byte-identical (content hash) | `log_observation` refuses imagined provenance. That makes it the only way an observation can enter evidence, so a model output cannot be filed as evidence by relabelling it (§7.4). |
| An unscored outcome has error `UNKNOWN` and is left out of failure sampling | Treating unscored as 0.0 error would make "never checked" look like "predicted perfectly". |
| The representative floor is at least 0.25 per batch. An empty priority component falls back to representative, never to another priority | Plan §9: "Keep representative evidence when prioritizing unusual transitions." |
| Pinned data is capped at `max_pinned_frac` of the budget. Explicit pins past the cap are refused, and held-out chunks past it can be evicted, oldest first | Avoids a guard-becomes-latch (CLAUDE.md §4.1). If everything were pinned, the only escape would be a human. |
| Migration writes a new root through the contracts registry and opens the source read-only | Plan §8: never overwrite the original. |
| Client recovery ends the old episode with reason `client_recovery`. Its unresolved prediction has no outcome | Nothing is spliced across a client rebuild, and the store does not invent an outcome. |

## Not done here

- Nothing writes to the store from the live loop. Moving to the Shadow stage
  (§8) means wiring an `AdapterRecorder`-style producer into
  `_collect_segment`, measured against the step-rate baseline. That has not
  been started.
- The index is rebuilt by scanning every chunk on open. That is fine at
  offline scale. A persistent index is the first thing to add if open time
  becomes a measured cost.
