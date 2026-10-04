"""SkyBot learning foundation ("improvement plan.md" §5).

A backend-independent layer that the existing perception, world-model and
policy packages IMPLEMENT, rather than a second orchestration stack beside
them. Subpackages are introduced incrementally, one plan stage at a time:

    contracts/    versioned core records (Observation, ActionSpec, Action,
                  StateBelief, Mechanism, Prediction, Experiment, Evidence,
                  Skill), scoped IDs, Unknown/Absent/Inapplicable sentinels,
                  the JSON codec and its migration registry, and the
                  reset/stream rule for what may count as a transition
    adapters/     the EnvironmentAdapter contract, its reusable conformance
                  check, the SkyBot record adapter (sensor bus + RSSM ->
                  records, no Minecraft needed) and a nonspatial fixture
    geometry/     types, frames, transformations, applicable constraints
    experience/   evidence records, persistence, retrieval
    mechanisms/   predictive mechanisms and their applicability
    inference/    beliefs, filtering, hypothesis management
    discovery/    bounded proposals and structural revision
    experiments/  information estimates and candidate evaluation
    runtime/      scheduling, snapshots, checkpoints, resource budgets
    perception/   learned representation and persistent perception
    planning/     planning, skills and consolidation

ROLLOUT (plan §8). Every component moves through the same four steps and
nothing here may skip one:

    1. Offline           evaluated on stored experience only
    2. Shadow            computes during a run; controls no action, reward,
                         replay or normalisation of the live learner
    3. Limited           enabled by configuration for a bounded experiment
    4. Default candidate broadened only after its declared gates pass

One piece is wired into `core/developmental_loop.py`, and it is a guard,
not a learner: `runtime.collection_path.select_collection_path` runs at
boot and refuses the unwired single-env body (CLAUDE.md §4.2). Everything
else is Offline/Shadow by construction until a stage explicitly says
otherwise.

This module is deliberately import-light: it imports no subpackage, so
`import developmental_ai.foundation` costs nothing and pulls in no torch.
"""

__all__ = []
