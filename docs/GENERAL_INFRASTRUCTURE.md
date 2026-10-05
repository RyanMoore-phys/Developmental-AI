# General Infrastructure — problems, solutions, and reasoning

> **IMPLEMENTATION STATUS (updated 2026-08-08, same day):** a first wave of
> this programme is now BUILT, smoke-tested and deployed. See
> **Part 13 — What was implemented** at the bottom for the per-item ledger of
> what shipped, what is partial, and what remains deferred, plus the new
> `developmental_ai/infra/` package map. Items are marked in place through
> the document: ✅ implemented, 🟡 partial, ⬜ deferred.

**Written 2026-08-08.** Every problem below was *measured* on this project, in
the Minecraft/SkyBot embodiment. None of them are Minecraft problems. This
document exists to convert a week of domain-specific firefighting into a
domain-general engineering programme: each entry states the problem with the
evidence that produced it, proposes a solution, and argues why that solution
is the right one for a system meant to develop in **any** environment —
another game, a robot body, or a language stream.

**How to read it.** Part 0 is the meta-diagnosis: seven anti-patterns that
account for nearly every failure we have seen. Part 1 lists the
domain-specific patches currently in the codebase and the general mechanism
that should eventually replace each. Parts A–J are the catalogue: 57 numbered
items, each `Problem / Solution / Why this solution`. Part 11 ranks them.
Part 12 is the evidence index — the measured numbers behind the claims, so no
statement here has to be taken on trust.

**Status of every item: PROPOSED.** Nothing here is implemented. Items marked
🔴 are the ones I would build first.

---

## Part 0 — The seven anti-patterns

Nearly every bug in this project's history is an instance of one of these.
They are listed in descending order of measured cost.

**AP-1. Uninformative signals trusted as sensors.** A signal that never varies
carries zero information, but every downstream consumer treats it as
authoritative. The grounding head reported `tree_visible` at ~1.0 in every
frame — including in unrenderable void 65 blocks underground — because its
teacher (llava:7b) answered `true` for every image patch it was ever shown.
Head/teacher agreement measured 0.98, which is exactly what a faithfully
learned constant looks like. Consequence: every gradient built on that signal
(seek potential, centring, focus, target ranking) was flat for the lifetime of
the project, and the magnet sat pinned at its cold-start floor for 144
consecutive segments.

**AP-2. Guards that become latches.** A protective mechanism with no re-arming
path becomes an absorbing state. Nine occurrences: the cold-start budget, the
seek-nudge budget, the competence gate, the reliability floor, `_ever_curious`,
the annealing ratchet, the log-count high-water, `_broken_this_episode`, and
the anneal-only-widens interval. One of them caused a 19-hour zero-reward run.

**AP-3. Reward hacking as a structural inevitability.** Fixing one exploit
reveals the next-cheapest one. Measured sequence, all in 48 hours: dig dirt →
stare at sky → build a pillar and stare at sky from altitude → stare at the
ground → place blocks. Five exploits, five hand-written patches, no reason to
believe the sixth does not exist.

**AP-4. Silent inertness — declared ≠ running.** Features that are configured,
logged as active, and never execute. Confirmed instances: the live viewer
never received a frame in the mode that matters; the reach sense was computed
only in a code path the run never entered; the magnet's turn/other diagnostic
existed only in the episodic loop; `record_asked` had zero callers; dream
consolidation raised a shape error every segment for weeks.

**AP-5. Duplicated execution paths drifting apart.** Five incidents. The same
one-line fix had to be applied to three loop bodies, and was forgotten in at
least one of them twice.

**AP-6. Missing affordances.** The agent cannot equip a tool: its action space
contains no hotbar or inventory-manipulation action and its selected slot is
pinned at 0 forever. A granted iron axe therefore never reaches its hand, and
every chop it has ever attempted was barehanded (~60 ticks instead of ~8).
This was diagnosed as a *motivation* problem for weeks. A capability gap is
invisible to every motivational instrument we own.

**AP-7. Instant-small beats delayed-large.** Placed directly in front of logs
with an axe, the agent started a chop, abandoned it within a second, and dug
snow instead. Under per-step discounting a 60-tick payoff of 20.0 loses to an
immediate 0.15, and until 2026-08-08 the cheap blocks paid at all.

---

## Part 1 — Domain patches now in the code, and their general replacements

Being honest about what we built this week. Each of these works, and each is
Minecraft-shaped. The right long-term move is to keep the *behaviour* and
delete the *domain knowledge*.

| Current patch | What it hardcodes | General mechanism that should replace it |
|---|---|---|
| `_MASTERY_BLOCKS` map | Which block names count as "ground/foliage" | **#12** event-type habituation over adapter-emitted event kinds |
| Pitch-geometry boring-view rule (`abs(pitch) > 55°`) | That sky is up and feet are down | **#14** empowerment + **#13** learning-progress novelty |
| GUI predicates in `NON_STEERABLE` | Which percepts are screens | **#3/#33** percept typing: self-state vs world-object vs affordance |
| Wood/ore extrinsic whitelist | Which blocks are goal-relevant | **#40** goal ontology with learned admission criteria |
| Fovea = fixed 40% centre crop | That the target is centred in the frame | **#5** active perception with a learned/uncertainty-driven query policy |
| Contact gate matching `"log" in name` | Which skills need contact | **#43** learned initiation sets from per-option success statistics |
| `focus_distractors = [leaves, grass]` | What competes with a trunk | **#8** cross-channel contradiction + learned confusability |
| Hand-tuned `seek_*`, `coverage_weight` | The relative price of searching | **#10** reward accounting + **#11** farm detection closing the loop automatically |

---

## Part A — Epistemics: making the agent's beliefs answerable

### 🔴 1. Signal informativeness monitors

**Problem.** We had no way to distinguish "this sensor works" from "this
sensor is stuck." `tree_visible` sat at ~1.0 for the life of the project while
reporting 0.98 agreement with its teacher, and nothing in the system could
tell that a constant had been learned. Agreement measures whether the student
matches the teacher; it says nothing about whether either varies with the
world.

**Solution.** Wrap every learned predicate in a `SignalMonitor` that keeps a
windowed record of its output and reports, every segment: Shannon entropy of
the binarised signal, variance of the raw probability, and — where a
ground-truth event exists — mutual information with that event. Any predicate
below an entropy floor (say 0.1 bits over 1000 observations) is flagged
`DEGENERATE` and automatically demoted: excluded from gating and target
selection until it recovers, while still being trained. The segment log prints
a bits-per-predicate table.

**Why this solution.** Information content is the property we actually depend
on, so measure it directly instead of proxying it with accuracy. It is
model-agnostic, costs a histogram per predicate, and catches both failure
modes that agreement conflates (constant teacher, collapsed student). Demotion
rather than deletion keeps it honest: a sensor that recovers is used again.
For a general agent this is the minimum viable self-knowledge about perception
— *knowing which of your senses are currently telling you anything*.

### 🔴 2. Teacher calibration harness

**Problem.** We chose, trusted, and eventually replaced a teacher model by
hand. llava:7b answered `tree_visible=true` on eight of eight probe crops,
including canopy-only and pure-grass patches, and hallucinated freely at 128px
("a dog", "YOUR INVESTMENT"). We only discovered this because I manually built
a probe battery in the middle of a debugging session; qwen2.5vl:7b then scored
8/8 on the same battery at equal latency.

**Solution.** Make that battery permanent infrastructure. A `TeacherProbe`
holds a small set of *contrastive* items — pairs differing in exactly the
property being asked — with known answers. It runs at startup and on a slow
schedule thereafter, scoring each question by discrimination
(`P(yes|positive) − P(yes|negative)`), not accuracy. Below threshold, the
system logs loudly and can fail over to the next model in a configured
preference list. Results persist to a calibration file with model version and
date.

**Why this solution.** An oracle is a dependency, and dependencies get health
checks. Discrimination is the right metric because it is exactly what a
constant-answering model fails, and it is cheap: a dozen items, a few seconds.
It also turns model selection into a repeatable procedure instead of an
afternoon of manual probing — which matters more, not less, as teachers get
swapped for language, audio, or code domains.

### 🔴 3. Falsifiability as a precondition for belief

**Problem.** Our verifier is structurally blind to false positives. Only 4 of
28 predicates are ever scored, and only on block-break events, so a predicate
stuck at `true` is scored correct on every break and drifts to reliability
1.0 unopposed. That is precisely how llava's constant went undetected for
weeks while the module docstring cheerfully documented the blindness.

**Solution.** A registry mapping each assertible predicate to the event
classes that can *confirm* and *disconfirm* it. At startup, any predicate with
no disconfirming source is marked `UNFALSIFIABLE` and given a hard confidence
cap at its prior — it may never be used as evidence for anything. Adapters
gain a standard responsibility to emit negative events (we added exactly one
by hand: "attacked for 90 ticks and nothing broke"). The startup log prints
the falsifiable/unfalsifiable split.

**Why this solution.** Capping at the prior is the honest encoding of "no
evidence either way," and it makes the gap visible rather than letting
confidence accumulate from one-sided evidence. This is Popper as
infrastructure: a belief that no observation could refute is not a belief
about the world, and an agent that can only confirm will converge on
comfortable nonsense in any domain — vision, language, or science.

### 4. Provenance and trust typing on every signal

**Problem.** We independently invented "prefer measured over predicted" three
times: reach evidence over the VLM's spatial claim, break evidence over
`object_centered`, and pitch geometry over the learned sky head. Each time it
was a bespoke `if` and a long comment, and each time the logs could not say
which source had actually driven behaviour.

**Solution.** A `Signal` value type carrying `(value, source ∈ {measured,
derived, predicted, claimed}, confidence, timestamp)`. Consumers declare a
minimum trust level; an arbiter selects the highest-trust available source and
records which one won. The segment log reports, per consumer, the fraction of
steps served by each tier.

**Why this solution.** The precedence rule is already our house policy — make
it a type so the next sensor inherits it for free instead of re-deriving it
after a multi-day debug. Recording the winning source converts an invisible
policy into a measurable one; "the reach sense ran on predictions 98% of the
time" is the kind of sentence that ends an argument. Any embodied or tool-using
agent mixes ground truth with inference, so this is universal plumbing.

### 5. Active perception — attention as a costed action

**Problem.** A VLM cannot answer "is the trunk under my crosshair?" from a
full frame; it picked the far-left trunk over the near-right one when asked
for the nearest. We solved it by cropping the centre — converting an
unanswerable spatial question into an answerable presence question — but the
crop is a hardcoded 40% box.

**Solution.** Promote attention to an action type. The adapter exposes a set
of view transforms (centre crop, region crop, zoom, in language: a context
span or a re-read), each with a cost; a policy or an uncertainty heuristic
chooses which to issue within a budget. The fovea becomes one instance of a
general capability, and the agent can learn *where* to look rather than being
told.

**Why this solution.** Every real sensor has bounded resolution or context, so
the general repair is to change the question rather than upgrade the sensor —
which is what saccades, foveation, and re-reading all are. Making it costed
and learnable means the agent discovers that looking closer pays, instead of
inheriting our guess about which 40% of the frame matters. This transfers
directly to language agents choosing what to re-read and to any
retrieval-augmented system choosing what to fetch.

### 6. Uncertainty-driven teacher scheduling

**Problem.** Teacher calls were scheduled on a fixed clock with an anneal
driven by *full-frame* agreement. Because the fovea channel shared that
schedule, it inherited an interval that widened to 454 steps — one foveal
label per ~908 steps — precisely while the fovea head was the least trained
and most needed component in the system.

**Solution.** Replace clock-plus-anneal with an information-gain scheduler.
Score candidate moments by head entropy near 0.5, by cross-channel
disagreement, and by state novelty; spend a fixed query budget on the
highest-scoring moments, subject to a floor cadence per channel so nothing can
starve. Each channel anneals on *its own* competence, never on a sibling's.

**Why this solution.** Fixed clocks spend the budget uniformly over easy and
hard states; active learning spends it where the gradient is, which is the
whole reason to have an expensive teacher. The per-channel-competence rule is
the general lesson from our starvation bug: a scheduler must be driven by the
metric of the thing it schedules.

### 7. Self-reinforcing blind-spot detection

**Problem.** The fovea head's agreement sagged to 0.80 exactly in the failure
mode we were trying to diagnose — the agent spent all its time staring at
clouds, so the head got worst at judging the very states the agent occupied,
so the discount that would have moved it never fired. The loop is self-sealing:
the agent lives where it is blind, and stays blind because it lives there.

**Solution.** Track per-sensor reliability *conditioned on state clusters*
(reuse the existing perceptual-cell hash), and compute an occupancy-weighted
reliability alongside the global one. Alarm when occupancy-weighted reliability
falls materially below average — the signature of "the agent has moved into
its own blind spot" — and route extra teacher budget to those clusters.

**Why this solution.** A global accuracy number cannot see this; only a
conditioned one can. It is also the natural trigger for active learning, so
detection and remedy share a mechanism. Any system whose behaviour shapes its
own data distribution — every RL agent, every deployed recommender, every
self-training LLM — has this failure mode.

### 8. Cross-channel contradiction as evidence

**Problem.** Our channels silently `max()` over each other. When the fovea
says no trunk and the full-frame head says trunk present, the disagreement —
which is free information about *which* channel is wrong — is discarded.

**Solution.** When two channels' claims about the same proposition diverge
beyond a margin, emit a `contradiction` event: dock both channels' reliability
slightly, raise the teacher-query priority for that proposition, and log it.
Over time, a channel that is usually the loser of contradictions is
automatically down-weighted.

**Why this solution.** Disagreement is supervision you get without ground
truth, and it localises error to a proposition rather than a whole model. It
is the same principle behind ensemble disagreement in active learning and
behind LLM self-consistency checks, and it works in any domain with more than
one estimate of the same fact.

### 9. Probability calibration per predicate

**Problem.** Every threshold in the system — 0.6 present, 0.35 fovea, 0.5
focus, 0.7 fact — assumes a probability means the same thing across heads and
across time. Nobody has ever verified that, and a head trained on a
class-imbalanced label stream is systematically miscalibrated.

**Solution.** Fit a per-predicate calibration map (temperature or isotonic) on
held-out teacher labels, refresh it on a slow schedule, and apply it at the
head's output boundary so all downstream thresholds operate on calibrated
probabilities. Report calibration error per predicate.

**Why this solution.** It makes every hand-tuned threshold in the system mean
what it says, which retroactively repairs a dozen tuning decisions and makes
future ones portable across environments. Calibration is cheap, standard, and
a prerequisite for any principled use of probabilities — including
uncertainty-driven exploration (#6, #53).

---

## Part B — Motivation: an immune system instead of patches

### 🔴 10. Reward accounting as a first-class subsystem

**Problem.** For most of this project it was impossible to answer "what is the
agent being paid for right now?" The reward census I added mid-session
immediately revealed that ~98% of the drive came from a term nobody was
printing. Ad-hoc instrumentation found our three largest bugs; it should not
be ad hoc.

**Solution.** A `RewardLedger` that every reward-emitting site writes to with a
source tag. Per segment it reports total reward, per-source share and rate,
and a concentration index. Two automatic alarms: (a) any single source
exceeding a share threshold for several consecutive segments, (b) a
*consistency check* — the ledger's total must equal the reward actually handed
to the learner, so a code path that adds reward without declaring itself is
impossible.

**Why this solution.** The consistency check is the part that makes this
infrastructure rather than logging: it converts "we think we know the reward
function" into a verified invariant. Concentration alarms are the earliest
possible warning of reward hacking, because every farm shows up first as one
source dominating the income statement. Any agent with more than two reward
terms needs this permanently.

### 🔴 11. Automatic farm detection

**Problem.** Five reward exploits in 48 hours, each found by a human eyeballing
logs, each fixed by hand. Our detection latency was hours to weeks; the
agent's discovery latency was minutes.

**Solution.** A monitor over a rolling window of `(state-cluster, action)`
sequences that detects behavioural cycles — returns to the same state cluster
within *k* steps — and computes net reward per cycle. A cycle whose mean net
income is reliably positive over many instances is flagged with its action
sequence and its dominant reward source from the ledger (#10).

**Why this solution.** This is the general form of the thing we kept doing
manually, and it has a principled basis: potential-based shaping guarantees
any cycle nets ~0, so *income on a cycle is definitionally either a real
achievement or a farm* — and the ledger says which. It requires no list of
banned behaviours, so it catches the sixth exploit we have not thought of, in
any domain. In a language agent, the same detector finds degenerate loops and
self-reward hacking.

### 🔴 12. Universal habituation over self-caused event types

**Problem.** We implemented habituation twice by hand, for two special cases:
block breaks (after the agent broke 272 dirt and 0 logs) and block placements
(after it responded by pillar-building, `use` at 22% of actions). Both fixes
carry hardcoded Minecraft taxonomies. There is no reason to think placements
were the last novelty-manufacturing action.

**Solution.** The adapter emits typed events `(kind, subtype)` — break, place,
craft, utter, move-to-new-place, whatever the domain has. The core maintains
lifetime counts per event type, applies a familiarity decay to intrinsic
reward on steps where an event fires (least-familiar event governs), and
persists the counts across restarts. No domain taxonomy anywhere in the core.

**Why this solution.** It replaces `_MASTERY_BLOCKS` and the placement special
case with one rule that covers every event class the domain will ever emit,
including ones we have not met. Habituation is the most conserved learning
phenomenon in biology — it exists in *Aplysia* — precisely because "stop
responding to what repeats without consequence" is universally correct. In
language it is the repetition penalty; in music, motif fatigue; here, dirt.

### 🔴 13. Curiosity that measures learning, not change

**Problem.** Our dominant intrinsic term is ICM prediction error, which is
maximised by anything visually dramatic. That is why breaking blocks, placing
blocks, and drifting clouds all funded farms: they change many pixels while
teaching nothing. Holding attack on one block — the behaviour that actually
achieves the goal — is the *lowest-change* action available, so the drive paid
most for what prevents progress.

**Solution.** Shift the primary intrinsic signal from error to *progress*:
periodically re-evaluate the world model on a fixed probe set of stored states
and pay for measured improvement attributable to recent experience
(compression progress / information gain). Keep raw error only as a
short-horizon supplement, capped by the ledger's concentration alarm.

> **Superseded 2026-10-05: progress is measured, never paid.** The paired
> probe measurement (`infra/progress_curiosity.py`) still runs when
> `progress_weight > 0`, but it records no reward: an improvement in the
> world model does not establish credit for the action being taken now, and
> the per-step rate was an ambient wage. The drive is the damped raw-error
> term (`icm_base_scale`) plus the itemised novelty terms. Paying for progress
> again needs causal attribution (`EvidenceCredit`, deliberately unwired) and
> a validated live integration. See `docs/foundation/LEARNING_PROCESS_IMPROVEMENTS.md`.

**Why this solution.** Error is maximised by noise — the noisy-TV problem, of
which drifting clouds are a textbook instance — while progress is maximised by
*learnable* structure, which is what a developmental agent should be
attracted to. We already have learning-progress machinery; the failure is that
raw error out-weighed it. This is domain-independent: it is the difference
between being fascinated by static and being fascinated by a language you are
starting to understand.

### 🔴 14. Empowerment / controllability as an intrinsic drive

**Problem.** The agent dug itself into a pit, then built a dirt pillar and
stood on it for six hours, one territory cell, ~26% movement actions, going
nowhere. Nothing in the reward function knew that a hole and a tower are bad
places to be. We patched the symptom with a pitch rule that encodes "sky is
up."

**Solution.** Add an empowerment term: from the current state, sample action
sequences through the world model and measure the diversity (entropy) of
reachable latent states over an n-step horizon. States with few reachable
futures — pits, towers, corners, dead ends — score low; open ground scores
high. Add as an intrinsic drive and as a term in stuck-detection (#50).

**Why this solution.** It replaces two of this week's domain patches with a
single principle that requires no knowledge of gravity, sky, or terrain:
*prefer states from which you can do more*. It is also a natural safety and
robustness prior (avoid irreversible corners), it uses the world model we
already train, and it transfers to any domain with a notion of reachable
futures — including conversation states in a dialogue agent.

### 15. Homeostatic drives with natural satiation

**Problem.** Our drives are budgets that expire (`cold_start_budget`,
`seek_nudge_budget`), and expiry is exactly the latch pathology of AP-2: once
spent, the drive is gone forever, so the agent lost its search drive
permanently while parked in a self-dug pit.

**Solution.** Model drives as depletable reservoirs with set points rather
than one-shot budgets: an exploration need that grows with time-since-novelty
and quenches on discovery; a contact need that grows with time-since-effect.
The reservoir formulation cannot latch because replenishment is intrinsic to
the mechanism, not a separately-remembered special case.

**Why this solution.** Biology solved "guard becomes latch" with homeostasis:
hunger does not expire. Structurally, a bounded self-restoring set point gives
the same protection a budget was meant to give (no unbounded farming) while
being immune to the failure mode that has bitten this project nine times. It
also produces natural behavioural rhythms — explore, consolidate, explore —
without a scheduler.

### 16. Explicit boredom → strategy switch

**Problem.** When intrinsic reward dries up, our agent does *more of the same
sampling*, just with lower value. It has no notion of "this is not working,
try a different kind of thing," which is why it can spend six hours in one
cell.

**Solution.** A metacontroller with explicit named modes — exploit, explore,
seek-novelty, practise, seek-help, rest — each with entry and exit conditions
and its own reward weighting or policy. Boredom (sustained low learning
progress) is an entry condition for exploration/help-seeking, not merely a
scalar that scales reward down.

**Why this solution.** Scalar drives can only change *how much* an agent wants
something, never *what kind* of thing it tries. Mode switching is how animals
and people escape local minima, and it makes the agent's strategy legible in
the logs ("entered SEEK-HELP at step N"), which is worth as much for debugging
as for behaviour.

### 🔴 17. Patience infrastructure (delayed gratification)

**Problem.** The decisive behavioural failure: placed in front of logs with an
axe, the agent began a chop and abandoned it within a second for snow. Under
per-step discounting, a 60-tick payoff of 20.0 is worth less than an immediate
0.15, and until this week the cheap blocks paid at all. Symmetrically, it has
also held attack for 340 ticks at an out-of-reach block — both are the same
missing mechanism.

**Solution.** Four coordinated pieces: (a) a completion bonus paid when a
committed macro finishes, making abandonment costly; (b) reward smoothing
across option boundaries so intra-option progress is visible to the learner;
(c) an option-level value function that is not shredded by per-step γ; (d) a
patience curriculum that lengthens the action-to-payoff gap as competence
grows. Termination remains progress-based (#42), so commitment never becomes
stubbornness.

**Why this solution.** The problem is structural, not motivational: no reward
tuning fixes γ^60. Options are the standard fix and we already have the SMDP
machinery; what is missing is the commitment economics on top of it. Delayed
gratification is a core developmental milestone in humans and the single
biggest determinant of whether an agent can pursue any multi-step goal — in
Minecraft, in a codebase, or in a conversation.

### 18. Potential-based shaping as an enforced type

**Problem.** Every shaping term in this system has been hand-verified for
telescoping — in comments, in smoke tests, in review — and we still shipped
farmable raw income twice (the 16:1 park-at-trunk incentive, the leaf-chop
bonus). The guarantee lives in prose, not in the type system.

**Solution.** A shaping API that accepts *only* a state-potential function;
the framework computes `γφ(s′) − φ(s)` itself. Raw per-step income requires a
different, loudly-named API that demands an explicit budget and registers with
the ledger (#10). Existing shaping is migrated to the potential form.

**Why this solution.** Ng's policy-invariance theorem gives us a guarantee for
free *if* the code structurally cannot express the unsafe form. Moving a
correctness property from review-time to compile-time is the cheapest possible
insurance against a bug class that has already cost this project a 19-hour
zero-reward run. Applies to any shaped-reward system in any domain.

### 19. Opportunity-cost accounting

**Problem.** "Keep chopping this log" competes against "go dig something
cheap" only implicitly, through a value function we have never inspected at
the decision point. When the comparison went wrong we could not see it happen.

**Solution.** Expose the best-alternative value estimate explicitly at option
termination and action-selection points, log it alongside the chosen action's
value, and use the gap as the give-up criterion in #42.

**Why this solution.** It makes the central economic decision — is this still
the best use of my time? — an inspectable number rather than an emergent
property. That is what let us diagnose the chop-abandonment at all, and it
generalises to any agent balancing a long task against cheap distractions.

---

## Part C — Dynamical safety: the nine-time bug

### 🔴 20. Latch detection as a framework primitive

**Problem.** Nine separate instances of "a guard became an absorbing state,"
one of which produced a 19-hour zero-reward run, another of which kept all 16
skills at zero invocations forever. Every one was written by someone who
understood the risk — the comments prove it — and it happened anyway.

**Solution.** A `Gate` primitive that every suppressing mechanism must be
constructed through, taking `(close_condition, open_condition,
max_closed_steps)`. The framework tracks each gate's state, evaluates open
conditions on a schedule, and alarms when any gate has been closed longer than
its declared maximum. A static lint rejects any behaviour-gating counter that
only ever decrements and is not registered as a `Gate`.

**Why this solution.** Nine occurrences of one bug class in one project means
the abstraction is missing, not that the engineers are careless — the fix has
to be structural. Requiring an `open_condition` at construction makes the
un-latchable version the *easy* version to write. The failure mode it prevents
(silent permanent capability loss) is catastrophic and near-undetectable in
any long-running agent, which is exactly when it matters most.

### 21. Recoverability / ergodicity invariants

**Problem.** Several of our latches were "recovery requires the very thing the
suppression prevents": a muted `tree_visible` could only recover from log
breaks that the muted magnet could no longer steer toward. That is a
capability made permanently unreachable, and nothing checked for it.

**Solution.** For each declared capability, assert that a path back to
exercising it exists — approximated cheaply by world-model rollouts, or
empirically by "has any stream exercised it in the last N steps." Alarm on
capabilities unreachable beyond a threshold, and name the gate responsible.

**Why this solution.** It states the property we actually want (no permanent
loss of function) rather than enumerating the mechanisms that could violate it,
so it catches novel violations. Ergodicity is also the assumption most RL
theory quietly rests on; a lifelong agent in a modifiable world breaks it
routinely and should therefore monitor it.

### 22. A continuous invariant daemon

**Problem.** Our invariants live in people's heads and in comments. "No gate
closed forever," "every subsystem fires," "no single reward source dominates"
were all violated for long periods without complaint.

**Solution.** A declarative assertion set evaluated every segment, each with a
name, a severity, and a remedy hint, printed as a short pass/fail block in the
segment log. Failures are loud and specific: which invariant, since when,
which component.

**Why this solution.** Cheap to build, and it converts silent pathologies into
noisy ones — which is the entire difference between a bug found in an hour and
a bug found in three weeks. It also encodes hard-won knowledge in executable
form so it survives context loss between sessions.

### 23. Distinct failure taxonomies for watchdogs

**Problem.** Our watchdog conflated distinct causes: it could not tell a death
from a kick from a frozen wall from a loading screen, and it once rebuilt the
client 27 times at a 3-minute cadence — consuming the very connection it
exists to protect. Separately, a persistent-stat sync at join was read as a
death, producing a 57-second rejoin loop.

**Solution.** Distinct detectors with distinct evidence, each mapped to its own
recovery: frozen POV + frozen state = disconnect; frozen POV + changing state =
static screen; changing POV + no progress + same location = stuck; socket
error = crash. Recovery actions are chosen per cause, and every trigger logs
its evidence.

**Why this solution.** A watchdog that cannot name the failure will eventually
take the wrong recovery action — ours did, twice, expensively. Naming causes
also makes the recovery policy reviewable and gives the metacognition layer
(#50) a vocabulary for "what went wrong" that it can act on.

### 24. Baseline-vs-event semantics for streaming counters

**Problem.** Twice now, the same class of bug: MineRL's `mine_block` counters
restart at 0 on every mission, and the server's `deaths` stat reads 0 at join
until the persistent total (73) syncs. Both were read as events — 73 deaths in
one step — because "first observation is not an event" was a lesson, not a
mechanism.

**Solution.** A `MonotoneCounter` wrapper used for every external counter:
first observation sets a baseline and emits nothing; deltas beyond a
configurable sanity bound are treated as re-synchronisation, not as events,
and logged as such. All stat consumers go through it.

**Why this solution.** It encodes a subtle, repeatedly-forgotten semantic
distinction once, in one place, instead of in every consumer's head. Any agent
reading counters from an external system — game stats, API quotas, sensor
odometers — faces exactly this, and the failure is silent and behaviour-warping
rather than crashing.

---

## Part D — Observability: declared ≠ running

### 🔴 25. Proof-of-life contracts

**Problem.** The most expensive bug class in the project's history: features
that are configured, logged as active, and never execute. The viewer never
received a frame in the mode that matters; the reach sense was inert for a
whole run; the magnet's diagnostics existed only in a loop we do not use;
`record_asked` had zero callers; dream consolidation crashed every segment for
weeks behind a warning nobody read.

**Solution.** Every subsystem registers with an expected firing cadence at
construction. A heartbeat table in each segment log lists every registered
subsystem, its last firing step, and its expected cadence — anything overdue is
printed as a warning. A component that never fires at all is a startup-level
error, not a silent no-op.

**Why this solution.** It is small, mechanical, and inverts the default:
instead of assuming things run until someone notices they do not, the system
must continuously prove it. Given that five separate features were dead for
extended periods, this one check probably has the best cost-to-value ratio in
the whole document. It applies unchanged to any modular agent architecture.

### 🔴 26. Effective-config echo with provenance

**Problem.** `log_break_reward: 20.0` was configured, read into an instance
attribute, and silently ignored by a classmethod reading the class attribute —
so the single most important reward in the system paid 5.0 for its entire
existence. Historically, two other config keys shipped in every config file
with nothing reading them at all.

**Solution.** A config accessor that records every key actually read, with the
resolved value. At startup, print the effective configuration and — critically
— the list of keys present in the file that nothing read. Optionally fail hard
on unread keys in strict mode.

**Why this solution.** The gap between "configured" and "in effect" is
invisible by construction, and it silently invalidates experiments: every run
before this fix was measuring a reward function nobody had chosen. Unread-key
detection catches typos, refactors, and the exact classmethod/instance trap we
hit. This is a five-line change to a config wrapper with outsized payoff, in
any system with a config file.

### 27. Automatic ablation harness

**Problem.** We cannot answer "does this subsystem matter?" without a manual
experiment, so dead and useless components look identical to working ones from
the outside — see every item in #25.

**Solution.** A scheduled harness that zeroes one component at a time on a
scout stream running a fixed scenario, measures the delta on a small metric
set, and reports a contribution table. Runs continuously in the background at
low priority.

**Why this solution.** It converts an argument into a number, and it produces
the contribution ranking you need to decide what to fix, keep, or delete.
Ablation is the standard scientific tool for this question; the only novelty
is running it automatically and continuously rather than once per paper.

### 🔴 28. One execution graph instead of N loop bodies

**Problem.** Three duplicated loop bodies (serial, parallel-episodic,
lifelong), five drift incidents. The magnet block had to be fixed in three
places; the reach argument was missing from one; the rotation diagnostic
existed only in the path we do not run. I deduplicated the magnet block this
week — the other duplicated concerns remain.

**Solution.** A `StepPipeline` of registered hooks with declared ordering and
dependencies. Each mode (waking, dreaming, parallel) selects a subset of the
same registry rather than re-implementing the sequence. Adding a per-step
concern means registering it once.

**Why this solution.** Copy-drift is not a discipline problem at five
incidents; it is an architecture problem. A registry also makes the step
sequence *inspectable* — you can print the pipeline — which supports proof-of-
life (#25), tracing (#30), and ablation (#27) with no extra work.

### 29. Behavioural time-series with anomaly detection

**Problem.** We caught the pillar farm because I happened to notice `use` at
18% in an action histogram. We caught sky-staring because I read a pitch mean.
Detection depended on a human reading the right line of a 100-line segment
report.

**Solution.** Track distributions of action, pose, location occupancy, and
event mix in rolling windows; compute divergence between the recent window and
a longer baseline; alarm when it exceeds a threshold, naming the top-shifted
dimension ("`use` 3% → 22%").

**Why this solution.** Behavioural regime change is the earliest observable
signature of both reward hacking and genuine developmental transitions, and it
requires no model of what the agent *should* do. Naming the shifted dimension
is what makes the alarm actionable. In a language agent, the same monitor
detects mode collapse and looping.

### 30. Replayable decision traces

**Problem.** Asking "why did it do that?" requires reconstructing state from
scattered logs and, frequently, adding instrumentation and waiting hours for a
new run.

**Solution.** A ring buffer of structured decision records — drive
contributions, options offered/vetoed and by which gate, chosen action,
resulting event — dumped on demand, on anomaly (#29), or on invariant failure
(#22).

**Why this solution.** It collapses the debugging loop from hours to seconds
for the most common question we ask, and it is the substrate for
self-explanation (#52): an agent cannot explain a decision it did not record.
Explanation infrastructure is also the honest prerequisite for trusting an
agent's behaviour in any domain.

---

## Part E — Embodiment and affordances

### 🔴 31. Affordance completeness audit

**Problem.** The agent cannot equip a tool. Its macro action space has no
hotbar or inventory action, and its selected slot is pinned at 0, so a granted
axe is unreachable. Every chop in the project's history was barehanded at ~60
ticks against an option budget of 80 — and we spent weeks tuning *motivation*
for what is a body problem. Not one instrument we own reports "the goal is
unreachable under the current action space."

**Solution.** Maintain a learned action→effect map (which actions have ever
produced which effect classes). Given the goal set's required effects, report
effects that no action has ever produced after N steps as `POSSIBLY
UNREACHABLE`, and effects that are structurally absent from the action space
as `UNREACHABLE`. Print at startup and per segment.

**Why this solution.** It asks the question no other subsystem asks — *can this
body do this at all?* — and it needs no domain knowledge, only the
action→effect statistics we already collect for causal facts. Motivation
diagnostics cannot distinguish "unwilling" from "unable," and confusing the two
cost this project more time than any single bug. Every embodied or tool-using
agent, including an LLM agent with a tool list, needs this check.

### 32. Action-space growth as a developmental event

**Problem.** We widened the action space by hand (10 → 12) when crafting
required it, with manual attention to policy-head shape and skill
compatibility. Motor repertoires should grow as development proceeds, not as
maintenance.

**Solution.** A policy head that supports graceful growth (reserved slots or
head expansion with old-weight preservation), a registry of available
primitives per environment, and an explicit "primitive acquired" developmental
event that is logged, persisted, and reflected in saved skills.

**Why this solution.** Fixed action spaces are an artefact of benchmark RL, not
of development — infants acquire motor primitives over time, and any agent
meant to grow into a domain will need new effectors (a new tool, a new API, a
new syntactic construction). Making growth a first-class event keeps skills
loadable across the change, which is the part that breaks in practice.

### 33. Body-schema learning

**Problem.** The system does not distinguish "actions that change me" from
"actions that change the world," so it never noticed that *no action changes
its equipped item* — the AP-6 gap. It also read a dead `equipped_items` sensor
as truth for a full day.

**Solution.** Learn `P(Δself | action)` and `P(Δworld | action)` from
experience, maintaining a body schema: which self-attributes are controllable,
by what, and which are inert. Report self-attributes with no controlling action
and actions with no observed effect.

**Why this solution.** It automatically surfaces both AP-6 (a capability with
no action) and dead sensors (an attribute that never changes), turning two
expensive manual discoveries into standing reports. Knowing the boundary and
capabilities of one's own body is prerequisite to tool use and, in
developmental psychology, precedes almost everything else.

### 34. Tool use as an abstract capability

**Problem.** We have the pieces — "axe enables break_oak_log" is minted from
observation — but no *concept* of an object-in-effector that modulates the
outcomes of actions. So the agent cannot reason "I should hold the thing that
makes chopping fast," even with all the facts present.

**Solution.** Represent effector contents as a state feature that conditions
the learned action→effect model, so effects are `P(effect | action, held)`.
Tool acquisition and equipping then become goals like any other, generated by
the goal system when the conditional model shows a large advantage.

**Why this solution.** It makes tool use emergent from a representational
choice rather than a special case, and it composes with #31 and #33 — the
agent can discover *and act on* "there is an effect I could produce if I held
something else." Tool use generalises to any domain where context modulates
action outcomes, including an LLM choosing which API to call.

---

## Part F — Skills, hierarchy, and memory

### 35. Skill individuation with real parameter isolation

**Problem.** Minted skills are byte copies of the shared policy — measured
cosine similarity 0.869–1.000, with three pairs bit-identical and the slot
signal attenuated 20×. Their identity is a slot name that drifts, orphaning 3
of 16 skills from their competence records. They are not, in any meaningful
sense, distinct things.

**Solution.** Skills own a genuine parameter delta (adapter/LoRA-style over a
shared trunk is fine, provided the delta is nonzero and measured), carry a
stable UUID independent of slot or name, and are identified behaviourally — a
signature of their action distribution over a probe state set. Mint-time checks
reject a "skill" whose behaviour is indistinguishable from the base policy.

**Why this solution.** A skill that is a copy of the policy cannot be
independently improved, evaluated, or composed, so the entire hierarchy above
it is decorative. Behavioural signatures give identity that survives renaming
and refactoring — the failure that orphaned our competence records — and give
an objective answer to "is this actually a new skill?"

### 36. Bootstrapping without the competence catch-22

**Problem.** A skill must be competent to be offered, but cannot become
competent without being offered. We hit this as a hard latch (all 16 skills at
zero invocations forever, needing weight ≥ +4.58 against a frozen inherited
bias of −5.68) and patched it twice with probation re-offers.

**Solution.** Drive offers by *uncertainty* rather than by mean competence: a
skill with high variance in its competence estimate is offered regardless of
its mean, because trying it is informative. This is Thompson sampling over
skill competence, and it makes probation a consequence of the rule rather than
a bolt-on.

**Why this solution.** It converts an arbitrary gate into a principled
exploration policy with known regret properties, and it structurally cannot
latch — uncertainty about an untried skill stays high, so it keeps being tried.
Curiosity about one's own abilities is also exactly what drives skill
acquisition in development.

### 37. Deliberate practice scheduling

**Problem.** Skills are practised only when the meta-policy happens to sample
them; nothing chooses to rehearse a skill *because* it is nearly learned. Our
log-breaking skill sits at 0/20 with no mechanism that would ever prioritise
fixing that.

**Solution.** Schedule practice for skills whose estimated success probability
is nearest 0.5 — the maximum-information, maximum-learning-rate band — using
dream/imagination rollouts where the world model is trusted and real attempts
otherwise. Report a practice ledger per segment.

**Why this solution.** The zone of proximal development is arguably *the*
mechanism behind human skill acquisition, and it is a straightforward
consequence of maximising expected learning per attempt. It also puts the
consolidation machinery we already have (dream training) to work on the skills
that most need it, instead of round-robin over everything.

### 38. Episodic memory of events

**Problem.** The agent has a replay buffer (for gradient updates) but no
retrievable memory of *what happened where*. It cannot act on "the last time I
saw a trunk, it was in that direction," which is precisely the knowledge that
would end its six-hour wander.

**Solution.** Store compact episodic records — state embedding, event, outcome,
context/location — in a retrievable index. Retrieve by similarity to the
current state to inform behaviour (return to where effects were achievable),
goal generation, and consolidation. Bound by relevance-weighted forgetting.

**Why this solution.** Gradient memory and retrievable memory answer different
questions ("what should my weights be" vs "where was the thing"), and we only
have the first. Episodic memory is the substrate for planning, for
one-shot learning from rare events like our single log break, and for
retrieval-augmented behaviour in language agents — the same mechanism in every
domain.

### 39. Consolidation with correctness contracts

**Problem.** Dream consolidation raised a shape error on every attempt for
weeks (`8x354` vs `365x256`), logged as a warning and swallowed, so a core
developmental mechanism was silently absent while appearing configured and
active.

**Solution.** Declared shape/dtype contracts at the boundaries of the
consolidation path, a startup canary that runs one rollout end-to-end and
fails loudly if it cannot, and explicit disable-with-reason rather than
per-segment warnings. Feature status appears in the proof-of-life table (#25).

**Why this solution.** Sleep/replay consolidation is too important to be
best-effort — it is where slow structural learning is supposed to happen. A
canary converts a silent multi-week absence into a startup failure, which is
the correct trade: loud and early beats quiet and expensive.

---

## Part G — Goals

### 40. Goal ontology with learned admission criteria

**Problem.** "Any sufficiently large reward spike mints a goal" produced 48/48
ungrounded goals and 0/51 mastered skills historically, and this week I
patched the symptom with a hardcoded wood/ore whitelist — replacing an
over-permissive rule with a domain-specific one.

**Solution.** A goal ontology with types (achievement, maintenance,
exploration) and *learned* admission criteria based on event statistics:
rarity, controllability (does the agent's action reliably produce it?),
novelty of effect, and downstream enablement (does achieving it unlock other
effects?). The whitelist becomes an emergent consequence — logs and ores score
high on enablement, dirt does not.

**Why this solution.** It replaces both a broken heuristic and a hand-written
list with criteria that a general agent can evaluate in any domain, and it
encodes the actual intuition behind our whitelist (wood and ore *lead
somewhere*) rather than its conclusion. Downstream enablement in particular is
what distinguishes a goal from a distraction, in Minecraft and in life.

### 41. Feasibility estimation before commitment

**Problem.** The agent commits to goals it cannot currently achieve — chopping
while out of reach, mining with no tool — and discovers this only through
repeated failure, which also corrupts its competence estimates.

**Solution.** Estimate achievability before committing, using the world model
and the affordance map (#31). An infeasible goal triggers sub-goal generation
("get closer," "acquire the tool") rather than repeated attempts; a goal
infeasible under the action space is reported, not attempted.

**Why this solution.** It prevents competence statistics from being poisoned by
attempts that were never possible, which is a measurement problem as much as a
behavioural one. Recursive decomposition on infeasibility is also the seed of
genuine hierarchical planning, and it comes almost free once #31 exists.

### 42. Explicit persistence-vs-abandonment policy

**Problem.** Both failure directions, on the same day: abandoning a chop within
one second, and holding attack for 340 ticks against an unbreakable target.
Nothing decides how long to persist.

**Solution.** A give-up criterion based on *progress*, not time: continue while
a progress signal (break progress, error reduction, distance closing) is
positive; abandon when progress stalls for a threshold, or when opportunity
cost (#19) exceeds the expected remaining value. Commitment bonuses (#17)
raise the bar for abandonment without making it impossible.

**Why this solution.** Time-based limits are wrong in both directions —
fast-succeeding attempts get cut off, hopeless ones run long — while progress
is the signal that actually distinguishes the two. The same rule covers "keep
reading this document" and "keep swinging at this trunk."

### 43. Learned initiation sets (preconditions from experience)

**Problem.** My contact gate matches the substring `"log"` in a skill name and
compares a fovea probability to a hand-picked 0.35. That is a domain hack
standing in for something the agent should learn: *when does this skill
actually work?*

**Solution.** Per-option logistic (or small) model over grounded predicates
predicting success, trained on the option's own attempt history. The initiation
set is "predicted success above threshold," and the model is inspectable — the
agent can report the conditions it believes it needs.

**Why this solution.** It replaces hand-written preconditions with learned
ones that improve with experience, works for skills nobody anticipated, and
produces exactly the "I can do this when…" knowledge that makes a skill
composable. This is the classic options-framework initiation set, finally
learned rather than declared.

---

## Part H — Cross-domain generality

### 🔴 44. An event-centric environment contract

**Problem.** Our core is coupled to Minecraft in ways that are invisible until
you try to leave: habituation keyed on block names, goals keyed on
`mine_block` stats, grounding keyed on a fixed predicate list. The
"environment interface" covers percepts and actions but not the thing most of
the machinery actually consumes.

**Solution.** Formalise the contract as four streams: **percepts**, **actions**,
**events** (typed, caused-by-agent effects), and **ground-truth probes**
(facts the environment can confirm on request). Habituation (#12), goal
minting (#40), falsification (#3), and skill grounding all key off *events*,
so the domain supplies the taxonomy and the core stays domain-free.

**Why this solution.** Events are the natural interface between an agent and a
world for developmental purposes — they are what the agent *did*, which is what
learning should be organised around. Minecraft supplies break/place/craft; a
text world supplies utterance/answer/tool-call; a codebase supplies
edit/test-result. Getting this boundary right is what makes every other item in
this document portable.

### 🔴 45. Language as an environment under the same loop

**Problem.** The claim "this is a general developmental architecture" is
currently untested: everything we know about the system comes from one
embodiment. Some of our abstractions are probably Minecraft-shaped in ways we
cannot see from inside Minecraft.

**Solution.** Build a text-world adapter against the contract in #44: tokens or
messages as percepts, utterances/tool-calls as actions, attention over context
as the fovea (#5), repetition as habituation (#12), comprehension error as
prediction error, communicative intents as goals. Run the *unmodified* core.

**Why this solution.** It is the fastest possible falsification test for the
generality claim, and every place the core needs a change to accommodate it is
a place we had hidden a Minecraft assumption. It also opens the most valuable
domain for this architecture: language is where developmental learning,
scaffolding from a teacher, and grounded meaning matter most — and where our
"borrowed → owned" pattern already has a natural home.

### 46. Teacher-agnostic scaffolding protocol

**Problem.** Our VLM teacher is wired in specifically: a particular model, a
particular prompt, a particular fade schedule, with calibration and failover
done by hand mid-incident.

**Solution.** One interface for any oracle — VLM, LLM, human, simulator,
retrieval system — carrying a query method, a cost model, a calibration hook
(#2), and a fade-out schedule tied to student competence. Multiple teachers can
be registered and arbitrated by measured discrimination.

**Why this solution.** The "teacher supplies the abstraction, the student
grounds it, the teacher steps back" pattern is the best idea in this codebase
and is entirely domain-independent — it is how culture transmits concepts.
Making it a protocol lets us swap llava for qwen for a human for a simulator
without touching the learner, and lets a language agent learn from an LLM
exactly as our agent learns from a VLM.

### 47. Transfer evaluation harness

**Problem.** "General intelligence" is currently a claim about our
architecture, not a measurement. We have never tested whether anything learned
in one environment helps in another.

**Solution.** A standing benchmark: train in environment A, evaluate zero-shot
and few-shot in environment B, and report transfer as a delta against training
from scratch. Include representation transfer (world model), skill transfer
(options), and abstraction transfer (predicates, rules).

**Why this solution.** Without it, generality is unfalsifiable — and by the
standard of #3 we should not believe an unfalsifiable claim about our own
system. It also gives a scalar for architectural decisions that currently get
made on taste.

### 48. Heterogeneous multi-body learning

**Problem.** Fifteen of our sixteen streams are goal-blind scouts whose
experience is counted as attempts for competence purposes but which pursue no
goal — inflating denominators while contributing little. A significant fraction
of our compute produces nearly nothing.

**Solution.** Make scouts goal-conditioned and diverse (different goals,
different exploration temperatures, different bodies), feed their experience to
the shared world model with proper off-policy correction, and attribute
competence only to streams that actually pursued the goal.

**Why this solution.** It converts wasted compute into a genuine developmental
mechanism — learning from others' experience — and it fixes a measurement bug
at the same time. Population-based experience is also how any scaled version of
this architecture will have to work, in any domain.

---

## Part I — Metacognition

### 49. A self-model of competence and ignorance

**Problem.** The agent has broken competence estimates and *no representation
of ignorance at all*. It cannot distinguish "I have tried this and failed" from
"I have never tried this," which are the two most decision-relevant states in
development.

**Solution.** Maintain an explicit self-model: per-skill competence with
uncertainty, per-predicate reliability with evidence counts, per-effect
achievability, and an explicit "never attempted" category. Expose it to goal
selection, practice scheduling (#37), exploration (#36), and help-seeking
(#51).

**Why this solution.** Nearly every metacognitive behaviour we want — practise
the nearly-learned, explore the unknown, ask about the confusing, report the
uncertain — is a query against this one structure. Knowing what you do not know
is the load-bearing capability in autonomous learning, and it is the piece we
most conspicuously lack.

### 50. Stuck-detection and strategy escalation

**Problem.** The agent stood on one block for six hours with zero reward and
no representation of that fact. Every recovery in this project has been a human
noticing and intervening.

**Solution.** Continuously monitor progress at several timescales (reward,
learning progress, spatial occupancy, event rate, empowerment #14). On a
sustained plateau, escalate through the mode ladder (#16): change exploration
temperature → change goal → invoke a recovery behaviour → ask for help (#51).
Each escalation is logged with its trigger.

**Why this solution.** Plateau detection is easy, cheap, and something we have
demonstrably needed on every single run; the escalation ladder is what turns
detection into autonomy. This is also the honest general answer to "the agent
got stuck": not a domain-specific rescue rule, but a mechanism for noticing and
changing approach.

### 🔴 51. Asking for help as a first-class action

**Problem.** Across this entire project the agent has never once told us it was
in trouble. It could not report "I have been swinging at something for 300
ticks and nothing breaks" or "I cannot reach any of my goals" — both of which
we eventually discovered by hand, days later, from log forensics.

**Solution.** A structured help-request action available to the metacontroller,
emitting `(situation, what I tried, what I expected, what happened, my best
hypothesis)` to a human/oracle channel, triggered by stuck-detection (#50) or
by high uncertainty on a high-value decision. Responses enter as teacher input
(#46).

**Why this solution.** It is the highest-leverage item in this document
relative to its cost: days of our debugging were spent inferring states the
agent could simply have reported. Help-seeking is also a genuine developmental
milestone — children who ask questions learn faster — and it is the natural
bridge between an autonomous agent and a human collaborator in *any* domain.
Critically, it is a request, not an oracle dependency: the agent must keep
acting if no answer comes.

### 52. Self-explanation and verbalisation

**Problem.** Every "why is it doing that?" has been answered by me
reconstructing intent from histograms and reward traces, hours after the fact.
The agent has no account of its own behaviour.

**Solution.** Generate natural-language rationales from the decision traces
(#30) — periodically, and always on anomaly — using the LLM already in the
stack. Store them alongside the traces so runs become narratively searchable.

**Why this solution.** It is a debugging multiplier (the agent tells you it is
digging because digging pays, long before you infer it), and there is good
evidence that verbalisation *itself* aids abstraction — explaining a solution
improves transfer in humans. In a system already committed to converting
borrowed language into owned abstraction, self-explanation is the natural next
step, and it produces exactly the artefacts a human collaborator needs.

### 53. Value-of-information action selection

**Problem.** Our exploration is undirected: novelty bonuses and random search.
The agent never chooses an action *because of what it would reveal* — it cannot
decide to walk around a tree to see what is behind it.

**Solution.** Estimate expected information gain for candidate actions using
world-model ensemble disagreement or predicted entropy reduction, and include
it as an explicit term in action selection alongside expected reward.

**Why this solution.** It is the principled form of exploration — the thing
novelty bonuses approximate badly — and it is what makes an agent's probing
look like experimentation rather than fidgeting. It composes with the
self-model (#49): knowing what you do not know is what makes information gain
computable in the first place.

---

## Part J — Scientific infrastructure

### 🔴 54. A behavioural regression suite

**Problem.** We have 13 passing smoke contracts verifying *mechanisms*, and
they were all green while the agent stood on a pillar doing nothing for six
hours. The only behavioural test that has ever been run on this system is the
one **you** ran by hand: put it in front of a log and watch.

**Solution.** Scripted scenarios with fixed seeds and pass criteria, run on
every change: "spawned facing a log at 2 blocks with an axe → breaks it within
N steps"; "placed in a pit → escapes within N steps"; "no trees in sight →
territory cells increase within N steps." Report pass/fail plus the metric, so
regressions and improvements are both visible.

**Why this solution.** Unit tests verify that a mechanism *can* work; only
behavioural tests verify that the assembled agent *does* work — and the entire
gap between "all smokes green" and "0 logs in 400 breaks" lives in that
distinction. Fixed scenarios also let us compare across weeks of changes, which
is currently impossible. This is the missing half of the project's testing
strategy, and it applies to any agent in any environment.

### 🔴 55. Counterfactual reward replay

**Problem.** Every reward-design change costs a full restart plus hours of
wall-clock before we learn anything — and this week we spent several such
cycles discovering, one at a time, that cheap blocks out-competed logs.

**Solution.** Re-score logged trajectories offline under a modified reward
function and report what the agent's *preferences* would have been: which
behaviours would have paid most, which cycles would have netted positive. A
reward-design change gets evaluated in seconds against real historical
behaviour before it is deployed.

**Why this solution.** It would have predicted the dirt-versus-log preference
immediately, and it would have caught the 16:1 park-at-trunk incentive before
it ran for hours. Offline evaluation of reward designs is standard practice in
recommender systems for exactly this reason; the trajectories are already on
disk, so the marginal cost is a scoring script.

### 56. Canary and shadow streams

**Problem.** Every config change restarts the entire organism, and an
unexpectedly bad change costs hours of a lifelong run — we have done this five
times in two days.

**Solution.** Apply a change to one scout stream first, compare its behavioural
metrics against the unchanged streams for a fixed window, and promote to the
primary only on non-regression. Genuinely global changes still require a
restart, but the majority of reward/perception tweaks do not.

**Why this solution.** It makes experimentation cheap and reversible, which
directly determines iteration speed, and it exploits parallel streams we
already pay for. The same pattern (canary deploys) is standard in production
systems for identical reasons.

### 57. Provenance-linked experiment registry

**Problem.** Runs are correlated with changes by hand — archived log filenames
and my memory of what was deployed when. Reconstructing "which code produced
this behaviour" is already error-prone at the two-day scale.

**Solution.** Every run records code hash, effective config (#26), environment
version, teacher model versions, and start state; results link back to that
record. A one-line summary per run accumulates in a registry file.

**Why this solution.** Without provenance, long-horizon claims ("the fovea
helped") cannot be checked, and a project whose whole subject is *learning over
long horizons* cannot afford to lose its own history. It is also the
precondition for the transfer harness (#47) and the ablation harness (#27) to
mean anything.

---

## Part 11 — Priority

Ranked by (measured harm) × (generality) ÷ (cost to build).

**Tier 1 — build first.** These address bug classes that have each cost days
and will recur:

1. **#20 Latch detection** — nine occurrences of one bug class.
2. **#25 Proof-of-life** + **#26 effective-config echo** — cheapest fix for the
   most expensive bug class (silent inertness). Together, days of work.
3. **#10 Reward ledger** + **#11 farm detector** — converts unbounded
   whack-a-mole into a standing monitor.
4. **#1 Informativeness monitors** + **#3 falsifiability** — would have caught
   the broken teacher on day one and will catch the next one.
5. **#54 Behavioural regression suite** + **#55 counterfactual replay** — closes
   the gap where every mechanism tests green and the agent does nothing.

**Tier 2 — highest conceptual leverage.** These replace domain patches with
principles and unlock capability rather than preventing failure:

6. **#31 Affordance audit** — would have found the axe problem before any of
   this started.
7. **#14 Empowerment** + **#12 event habituation** + **#13 learning-progress
   curiosity** — together, these retroactively replace four of this week's
   domain-specific patches.
8. **#17 Patience infrastructure** — the current blocker on the actual task.
9. **#51 Ask for help** — days of debugging recovered for a few days of work.
10. **#44 Event-centric contract** + **#45 language adapter** — the test of
    whether any of this is general at all.

**Tier 3 — everything else**, valuable and mostly independent; sequence by
whatever the next failure demands.

---

## Part 12 — Evidence index

Measured facts underlying the claims above, with dates. Nothing in this
document rests on intuition alone.

- **Constant sensor:** `tree_visible` present in essentially every frame,
  including 65 blocks underground in unrenderable void; magnet pinned at the
  cold-start floor `w=0.3500` for 144 of 144 segments, `cold_spent[tree]=144827`
  (2026-08-06).
- **Broken teacher:** llava:7b returned `tree_visible=true` on 8/8 probe crops
  including canopy-only and grass-only patches, and produced hallucinated
  descriptions ("a dog", "YOUR INVESTMENT") on single-material patches;
  qwen2.5vl:7b scored 8/8 correct on the same battery at ~1.17s vs ~1.1s
  (2026-08-06).
- **Latches:** nine instances; one produced a 19-hour zero-reward run; another
  held all 16 skills at zero invocations, requiring weight ≥ +4.58 against a
  frozen inherited bias of −5.68.
- **Farm sequence:** 272 dirt / 0 logs → sky-gaze at 72.9% of steps →
  dirt pillar at y=72 with 1 territory cell/hour → ground-gaze at 48.5% →
  placement farming with `use` at 22% of actions (2026-08-06 → 08-08).
- **Silent inertness:** viewer never pushed a frame in `_collect_segment`;
  `_reach_now` computed only in the episodic path; magnet turn/other split
  present only in `_run_episode_parallel`; `record_asked` with zero callers;
  dream consolidation failing `8x354 @ 365x256` every segment for weeks.
- **Config not in effect:** `log_break_reward: 20.0` configured, 5.0 paid, for
  the lifetime of the setting — classmethod reading a class attribute that
  `__init__` set only on the instance (found 2026-08-08 from the first
  `LOG FELLED` line ever printed).
- **Missing affordance:** all 9 hotbar keys leave `mainhand=none`; slot 36 out
  of range; `TREECHOP_MACROS` contains no hotbar or inventory action.
  Barehanded log ≈ 60 ticks vs ≈ 8 with an axe, against an 80-tick option
  budget.
- **Impatience:** chop abandoned within ~1 second in favour of snow/dirt while
  standing in front of logs *with an axe* (user observation, 2026-08-08);
  converse failure, 340-tick attack streak with zero breaks, same period.
- **Skills are copies:** cosine similarity 0.869–1.000 across minted skills,
  three pairs identical, slot signal attenuated 20×; 3 of 16 skills orphaned
  from competence records by slot renaming.
- **Counter semantics:** server `deaths` read 0 at join then synced to the
  persistent total 73, parsed as 73 deaths → rejoin loop every ~57s, 16 rejoins
  in 30 minutes with health pinned at 1.00 (2026-08-08). Same class as
  `mine_block` restarting at 0 per mission.
- **Wasted parallelism:** 15 of 16 streams goal-blind, counted as Bernoulli
  attempts for competence.
- **Detection latency:** every one of the five farms was found by a human
  reading logs, hours to weeks after onset; the agent found each in minutes.

---

## Part 13 — What was implemented (2026-08-08 wave)

Built the same day the document was written, as a package of DOMAIN-AGNOSTIC
modules plus one integration facade, wired into the organism at a handful of
call sites. Every module has its own smoke suite (all green: 22/22 suites
including the full pre-existing regression set and a compile check).
Honest note on review: the independent adversarial-review agents scheduled
for this wave were blocked by account session limits; the integration was
instead self-reviewed systematically against the three planned lenses
(latch/crash, reward-economy, integration seams), which found and fixed one
real defect (a blocking world-model-lock acquisition in the acting hot path,
made non-blocking-skip). An independent review pass should be re-run when
capacity allows.

### The package

```
developmental_ai/infra/
├── stack.py         InfraStack — the ONE facade the loop talks to; every
│                    monitor degrades independently (a broken monitor can
│                    never take the run down); also owns the LEARNED
│                    fovea-category <-> event association and empowerment
│                    shaping plumbing
├── gate.py          #20  Gate / GateRegistry: every suppressor declares its
│                    reopen condition + max closed steps; overdue = alarm
├── lifecycle.py     #25/#22  Heartbeat (proof-of-life with expected
│                    cadences) + InvariantSet (contained, streak-tracking)
├── monitors.py      #50/#29/#30  StuckMonitor (staged escalation),
│                    BehaviorDrift (JS divergence on action distribution),
│                    DecisionTrace (ring buffer -> jsonl dumps)
├── ledger.py        #10/#11  RewardLedger (income statement, share alarms,
│                    HHI) + FarmDetector (state-cycle income — flags any
│                    repeating loop that nets > 0)
├── signal_health.py #1  SignalMonitor: entropy/variance per predicate;
│                    DEGENERATE signals are auto-excluded from steering
├── counters.py      #24  MonotoneCounter: baseline/resync/rebaseline
│                    semantics for external counters, in one place
├── episodic.py      #38  EpisodicEventMemory: what happened where, with
│                    recency/position/bearing queries + summaries
├── empowerment.py   #14  reachable-future diversity through the agent's own
│                    world model; EmpowermentPotential (bounded, normalized)
├── config_echo.py   #26  TrackedConfig + unread-key reporting: what was
│                    configured but never read
└── affordance.py    #31/#33  AffordanceMap: action -> effect statistics,
                     actions-with-no-effect, effects-never-produced
tools/reward_replay.py   #55  offline counterfactual scoring of trajectories
                         through the REAL reward primitives
tools/teacher_probe.py   #2   contrastive teacher-discrimination battery
                         (CLI + library, --fake for offline tests)
tests/_behavior_suite.py #54  asserted PREFERENCE ORDER of the assembled
                         economy: chop-log must out-earn dirt-farm /
                         sky-stare / pillar by >= 3x (it does: 23.9 vs 0.54
                         / 1.35 / <1)
tests/_infra_*_smoke.py  one suite per module, 70+ contracts total
```

### Wired into the organism

* **Event-centric contract (#44) — the load-bearing change.** The MineRL
  adapter now emits `info["events"] = [(kind, subtype), ...]` for everything
  the agent causes (break / place / craft / pickup / death), and persists
  lifetime counts for all four caused kinds. Habituation, the affordance map,
  episodic memory, farm detection and the category-association all consume
  ONLY this stream — no game nouns anywhere above the adapter.
* **Habituation generalised (#12).** `_habituation_factor` now runs on the
  event stream: any caused event kind habituates by its lifetime count,
  least-familiar event governs, deaths never habituate. The legacy
  achievements-parsing survives only as a fallback for adapters without the
  stream.
* **The hand-written category->block map is GONE.** `_boring_view_factor`
  now asks the stack's learned association ("what was I looking at when
  events fired") times event familiarity. Cold start = no associations = no
  discount — the safe direction.
* **Signal health in the steering path (#1).** Every per-step head output is
  observed; a predicate flagged DEGENERATE (entropy+variance floors) is
  excluded from the magnet's trusted-present set until it recovers. This is
  the check that would have caught the llava-era constant on day one.
* **Falsifiability (#3).** `FALSIFIABLE_PREDICATES` registry in the
  symbolizer; the other 24 predicates are reliability-capped at their prior
  and the split is logged at startup.
* **Empowerment shaping (#14).** Potential-based (telescoping) shaping on
  normalized reachable-future diversity, computed through the world model
  every 25 steps under the WM lock. Config-gated; on for SkyBot at 0.05.
* **Stuck escalation + help requests (#50/#51).** Segment metrics
  (income, territory delta, caused events) feed the StuckMonitor; L1 refills
  the search budget, L2 doubles exploration weights for one segment
  (self-restoring — never a latch), L3 writes a structured help request to
  `runlogs/help_requests.jsonl` and dumps the decision trace.
* **Gates on the known suppressors (#20).** seek-nudge budget, magnet
  weight, learned-option offers — each states its condition every segment;
  the registry alarms past the declared budget.
* **Ledger + farm detector (#10/#11)** harvest the existing per-term shaping
  sums plus raw env income each segment; income concentration and
  positive-income behavioural cycles print as alarms.
* **Proof-of-life (#25).** vlm_label / fovea_label / magnet / consolidation
  / viewer / option_offer heartbeats with expected cadences; overdue
  subsystems print in every segment report.
* **Config echo (#26).** The loop's config is wrapped at load; the first
  segment prints how many keys were read and which were never read.
* **Provenance (#57 light).** Run start logs a config SHA + timestamp.

### Status by priority tier

Tier 1: #20 ✅  #25 ✅  #26 ✅  #10 ✅  #11 ✅  #1 ✅  #3 ✅  #54 🟡 (reward-
stack preference suite; a live-env scenario harness remains)  #55 🟡 (tool +
canned trajectories; scoring of on-disk replay logs remains)
Tier 2: #31 ✅ (map + reports; goal-ontology linkage pending #40)  #14 ✅
#12 ✅  #13 🟡 (ledger share alarm caps raw-error dominance; the
compression-progress rewrite of the base curiosity term remains)  #17 🟡
(existing pieces enumerated — effort pay, extend-on-progress, whitelist;
completion-bonus/option-value work remains)  #51 ✅  #46 ✅ (VLM unstuck
advisor — see fifth wave)  #44 ✅ (MineRL adapter; other adapters emit no events
yet and all consumers no-op safely)  #45 ⬜ (the language adapter is the
next standalone project)
Selected others: #22 ✅  #24 ✅ (primitive; deaths/damage keep their proven
inline guards)  #29 ✅  #30 ✅  #38 🟡 (record/query/report; behavioural
integration pending)  #50 ✅  #2 ✅ (tool; startup integration pending)
#15 🟡 (nudge regen is the pattern; full reservoir refactor pending)
#21/#23/#33/#36/#37/#40/#41/#42/#43/#47/#48/#49/#52/#53/#56 ⬜ deferred.

### Second wave (2026-08-09): individuation, episodic sense, progress curiosity

* **#35 Skill individuation ✅ / #37 deliberate practice ✅.** A skill is now
  `frozen_base + owned DeltaHead` (zero-init: identity at birth; individuated
  only by practice, which trains ONLY the delta under the same trust
  region/revert). Stable uuid on bindings; per-skill `delta_mag` printed
  (the previously-unmeasurable individuation number); deltas persist to disk
  with the skill and reload on bind. Dream consolidation slot selection is
  now ZPD ("rehearse the learning edge", success ~0.5 ranks first) with an
  anti-starvation hunger term and jittered ties.
* **#38 Episodic memory → behaviour ✅ (sense level).** The proprio vector
  gains [memory-validity×proximity, sin, cos] of the body-relative bearing
  to the most recent goal sighting/achievement — wandering can become
  returning. Fixed en route: the reach sense was silently DEAD in the
  lifelong body (7th duplicated-body casualty — proprio assembly is now one
  shared helper), and the Minecraft yaw sign (clockwise) was pinned by test.
* **#13 Compression-progress curiosity ✅.** `infra/progress_curiosity.py`:
  fixed probe set from replay, WM loss re-evaluated under a NON-BLOCKING
  lock, improvement (never surprise) paid as a slowly-varying rate;
  noisy-TV analogue pays ~0 by test. The raw ICM base term is damped
  (`icm_base_scale: 0.5` on SkyBot) so learning leads the drive.
  **Superseded 2026-10-05:** the payment is removed (measurement only,
  `reward=0` in the log); `icm_base_scale` still damps the base term, which
  together with novelty is now the whole base drive — progress leads nothing.
* **Metabolic effort cost (new, #15-flavoured).** Observed live: sustained
  attack swings at CLOUDS — futile effort was exactly free under the
  whitelisted economy. attack/use/jump steps now cost a small constant via
  the adapter's domain-agnostic "effort" field (a 60-tick chop ≈ −0.09 vs
  +20 for the log; air-punching just bleeds). Ledger source "effort".
* **Review honesty.** Independent reviewers ran this time and confirmed 4
  real defects (all fixed): a stale per-slot Adam after LRU eviction that
  made practice a silent no-op with phantom strength gains; deltas trained
  but never persisted across restarts; ZPD stable-sort starvation; a boot
  smoke crashed by the 4-tuple change. The refutation stage was again
  killed by account session limits, so the remaining claims were triaged by
  hand: fixed — progress payout dead via a private-attr AttributeError
  (which also silently killed empowerment), fleet-factor underpayment,
  consolidation feature-column misalignment (knowledge sat in proprio
  columns), farm detector blind to the new income sources, revert path
  leaving grads enabled; accepted as minor and documented — icm scale vs
  absolute eps_abs interplay, uuid not yet consumed, serial-body scale
  without progress compensation (path unused by SkyBot), 1-step-stale world
  info in the bearing sense.

### Third wave (2026-08-09): social learning — monkey-see-monkey-do, honestly

The user plays on the same server; the single most human learning channel —
watching a co-present adult demonstrate — was invisible. Now:

* **The teacher is a percept**: `player_visible` in both vocabularies (full
  frame + fovea), KG triple, WATCHED BUT NEVER CHASED (non-steerable: a
  mobile magnet target defeats the phi ratchet — orbiting the user would be
  repeatable income; review-confirmed and excluded).
* **External-agency detection**: on measured-passive, measured-STATIONARY
  steps (position AND camera still; inventory/GUI excluded), a large frame
  change emits `("observed_change","world")` — "the world changed and I did
  not do it". Never credited as self-caused: excluded from habituation,
  affordance, and the stuck monitor's sign-of-life; throttled in episodic.
* **Joint attention**: with the teacher under the gaze, LP attribution for
  co-present categories doubles (never for the trigger category itself).
* **Goal emulation**: observed change + teacher recently in view =
  DEMONSTRATION → the top co-present PRIMEABLE category is socially primed
  (curiosity injected ~4000 steps + search refilled), edge-triggered with a
  500-step refractory — imitate the WHAT, discover the HOW, matching the
  developmental finding that children emulate goals over motor programs.
* **Adversarial review (ran fully this time)** found 2 criticals + a dozen
  more before deployment, all fixed: the inventory toggle manufactured
  "demonstrations" (48% frame change classed as passive); re-priming per
  observed change turned the bounded seek nudge into a standing wage near
  the user; uncounted event kinds defeated habituation via max(); the
  magnet heartbeat fell inside the social branch (indentation); the social
  gate was satisfiable by an untrained head (now gated on POSITIVE teacher
  sightings); falls/knockback/water read as external agency (now measured
  stationarity); the association could learn to make the TEACHER boring
  (excluded); the magnet-seek envelope was raised (weight 1.0, seek 1.5,
  cold-start 0.5) at user request — all still telescoping-only.

### Fourth wave (2026-08-10): pre-boot points 1–4 — implemented local-only
(host terminated; this wave ships with the next VM)

The pre-boot assessment named four structural issues in how SkyBot learns;
all four are now code, smoke-tested (`tests/_preboot_wave_smoke.py`, 8
contracts) with the full local regression sweep green:

* **1. Perception and familiarity now persist** (the biggest restart tax:
  every boot rebooted the grounded heads to random, re-idled the magnet
  behind DEGENERATE flags, and re-paid the novelty windfall for a world
  already seen — measured at 55–65% of all income after restart-heavy
  waves). `symbolizer.pt` carries heads + reliability + label/positive
  evidence + retractions (heads restore only on an exact vocabulary match —
  partial surgery would silently misalign predicates; evidence merges on
  common keys across any vocabulary change). `familiarity.pt` carries the
  view/gaze/symbol "what have I seen" counts (merge, never replace;
  `known_symbols` serialized as a sorted list so `weights_only` loads
  survive). Env-side territory (`_visits`) rides `breaks_by_type.json` as
  `"cells"`, getattr-guarded so a missing dict can never poison the whole
  break-memory flush. Both components joined `loop.resume_components`;
  policy/dream_actor/curiosity remain refused by name.
* **2. Degenerate-gate cold-start patience** (the gate flagged 21/28
  predicates at every fresh boot — an untrained head outputs near-constants
  for everything, and that is ignorance, not the learned-constant pathology
  the gate exists for). A signal may now only be flagged DEGENERATE once
  its teacher has supplied `degenerate_min_labels` (25) labels; the loop
  passes `signal_evidence = symbolizer.label_counts` into the segment ctx.
  Absent evidence data keeps the old strict behaviour, so no other config
  changes meaning.
* **3. Memory-pull potential — episodic memory becomes motivating** (the
  bearing sense INFORMED the policy where trees were last seen, but
  nothing made acting on it PAY; memory was a map with no pull). New
  telescoping potential `phi = validity(age) · proximity(dist)` to the
  freshest remembered goal-site record (sighting/break of the seek
  category): returning pays once, loitering pays zero, leaving charges
  back, and the pull fades with memory age. A changed record re-adopts the
  baseline UNPAID (fresh sightings gift nothing) and a death clears the
  baseline (die-to-travel would otherwise be a farm). Ledger source
  `memory_pull`, weight 0.5 (skybot) / 0.0 (default off).
* **4. Competence floor 0.0 — options actually fire** (the fourth
  guard-becomes-latch: every skill's frozen bias inherited a poisoned
  −5.68, no invocation could ever raise competence, so all 16 skills sat
  at invocations=0 forever and mastery was unmeasurable). Eligibility no
  longer depends on a statistic that can only move once eligibility is
  granted; probation, the contact gate, precondition matching and the
  option value still govern firing. The gate MECHANISM is untouched and
  test-pinned against the historical 0.25 for configs that still want it.

### Fifth wave (2026-08-16): the unstuck wave — DEPLOYED to the a rented GPU host training host
(synced + relaunched the same night; the prior run had confirmed the latch
live twice, once per clamp)

Live diagnosis on the running host found the agent parked at one coordinate
for 2+ hours, gaze pinned at +90, every vision predicate DEGENERATE, magnet
at w=0/target=None, seg extrinsic 0, help requests unanswered. After the
first fixes were staged, the SAME run flipped to the −90 clamp (sky), which
exposed the fifth item below. Five fixes, all smoke-tested
(`tests/_unstuck_wave_smoke.py`) with the full local regression sweep
green:

* **#46 Unstuck advisor ✅ — help requests get answered, by the VLM.** At
  stuck L3 the request PLUS the agent's current view goes to the local VLM
  (same model that teaches perception; `infra/advisor.py`, query injected
  so infra stays dependency-free). The answer is validated hard (unknown
  remedy / hallucinated category / prose → None, run unaffected) and can
  only pick from existing self-expiring drive levers: `look_around` (cap
  gaze buckets + refill seek), `prime` (social_prime a KNOWN category —
  emulate the what, discover the how), `explore_wider` (the L2 boost on
  request), `conserve` (no-op is an answer). Rate-limited; dialogue logged
  to `runlogs/help_responses.jsonl`.
* **STARVED vs BROKEN degenerate split (the 5th guard-becomes-latch).**
  The min-positives gate fixed cold start, then latched on a WARM head:
  historical positives + ground-filled view → constant-low → flagged → the
  magnet forbidden from steering toward the one thing that cures the
  constancy. Now constant-LOW with proven positives stays steerable
  (honest absence is what seeking is FOR; a stuck-low broken head fails
  inert, not as a farm); only stuck-HIGH or positive-free signals are
  excluded. Mass starvation (≥`gaze_starved_frac` of predicates) is
  surfaced as `actions["gaze_starved"]` — a GAZE verdict, remedied on the
  behaviour side (gaze buckets capped at 25 + seek refill, cooldown-gated).
* **Gaze-leveling potential.** The pitch clamp was an attractor with no
  exit tax (the gaze-bucket bonus saturates 1/sqrt(n)). Telescoping
  potential Phi = −(|pitch|/90)^4: mining tilts ≤60° cost ~2% of the
  weight, the last 30° into a clamp carry the whole gradient, exit pays
  back what entry charged. Ledger source `gaze_level`, weight 0.02
  (skybot) / 0.0 (default off).
* **Boring-view discount on the BASE curiosity term.** At the −90 clamp
  the census read: ICM/LP base +0.0147/step = 96% of the drive, paid for
  watching clouds drift — the sky discount only ever covered the itemised
  novelty term, so the base was a standing wage "no shaping term can
  outbid" (the census caption's own words). `icm_boring_discount` (0.85
  skybot / 0.0 default) applies the same measured `_boring_view_factor`
  judgement (geometry past 55°, fovea sky reading, learned boringness,
  floor 0.15) to the primary stream's base, BEFORE the census accumulator
  so the census cannot lie. Cloud wage 0.0147 → ~0.004/step; horizon
  views keep full pay; the world model still trains on every frame.
* **(ops) ollama log cap.** 581 MB in 5 days at llama-server verbosity 4.
  Launch scripts now start ollama with append-mode redirect + a reusable
  `scripts/cap_log.sh` watchdog (du-based — apparent size lies for sparse
  files); the live host got an equivalent crontab entry and an immediate
  truncate (10 MB tail kept).

### Persistence fixes (2026-08-17) — deployed with the fifth wave

Two restart/storage defects found while auditing what actually learns
(`tests/_persistence_fixes_smoke.py`; verified on the training host GPU):

* **Skill deltas were never compressed on a GPU box.** The bank writes its
  shared base encoder with `.cpu()` and reads it back with
  `map_location="cpu"`, while the live encoder sits on the run's device —
  so `to_delta`'s `w - b` raised a device mismatch, which the caller caught
  and answered by storing the skill WHOLE. Measured live: 14 MB per skill,
  i.e. the compression win absent on exactly the hardware that trains.
  Devices are now aligned inside `to_delta` (and mirrored in `from_delta`),
  and `_deltaify_encoder` hands it CPU tensors so no CUDA tensor reaches a
  checkpoint file.
* **The magnet was amnesiac across restarts, not suppressed.** `reset()`
  preserves curiosity memory across episodes, deaths and dream boundaries,
  but nothing carried it across a process restart: every launch began
  `w=0.0000, target=None` with all categories at LP 0.000 and spent its
  first hours unsteered — the same restart tax the perception/familiarity
  checkpoints removed, still being paid by the drive that aims the other
  senses. `VisionScaffold.state()/load_state()` → `magnet.pt`, joined
  `resume_components` and **gated on `world_model`** like its siblings (LP
  is a rate measured *through* that world model; against a fresh one it is
  a stale claim). Merge-not-replace, so a vocabulary that has grown since
  the checkpoint keeps its new categories. `_cold_spent` is deliberately
  excluded: persisting a spent cold-start budget would promote the 19-hour
  zero-reward latch from run-scoped to permanent.

### Data-reset decision (2026-08-08 relaunch)

Per the "reset only what prevents proper learning" rule: the skill bank +
broadcaster state were ARCHIVED (not deleted) — they encode goals, skills and
competence minted under five broken reward regimes (junk goals whose extrinsic
is now 0, byte-copy skills, poisoned competence bias), which actively fight
the corrected economy. The event-mastery memory (breaks/places/crafts/pickups
counts) was KEPT: it is legitimate measured experience whose only effect is
preventing relapse into already-mastered farms. No world-model or policy
checkpoints exist for SkyBot runs (each run trains fresh), so there was
nothing else to reset.
