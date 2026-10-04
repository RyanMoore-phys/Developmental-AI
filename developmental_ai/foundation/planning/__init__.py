"""foundation.planning — planning, skills and consolidation (plan Stage 11).

Public API:

  planner        BoundedPlanner (CEM over a mechanism/ensemble rollout;
                 uncertainty horizon cut + penalty; hard wall-clock
                 deadline; hung-model watchdog: poison, respawn, re-probe),
                 PlanResult, ReliabilityMonitor (real-outcome
                 validation; CUSUM dynamics alarms), PlanningController
                 (planner vs existing controller; every decision logged
                 with a source in SOURCES), RegionMemory (where real
                 validation failed), Decision, ActorCriticFallback
                 (policy/actor_critic.StandaloneActorCritic as fallback),
                 SOURCES, REAL_PROVENANCES
  skills         SetClassifier, BetaCompetence, LearnedSkill, calibration,
                 interval_coverage, BehaviourSignature, behaviour_signature,
                 behavioural_divergence, find_duplicates (-> DuplicateFinding
                 with a graded verdict in VERDICTS: duplicate |
                 near_duplicate | distinct | indistinguishable_on_probes),
                 modal_agreement (executed behaviour vs sampling noise), ProbeSet,
                 build_probe_set (probes from each skill's evidence, with
                 coverage), probe_hash, smdp_return, smdp_target
  consolidation  EpisodeMemory, KnowledgeStore (candidate / consolidated /
                 needs_revalidation, bound to a RepresentationRegistry),
                 RetentionMonitor, save_planning_state, load_planning_state,
                 STATUSES
  skill_bank_bridge  read_skill_bank, read_registry, legacy_to_skill
                 (READ-ONLY; refuses skill_bank_mc_curiosity/)
  fixtures       PointMassAdapter, PointMassMechanism, ProportionalController
                 and record helpers (closed-loop test fixtures)

Offline/shadow only (plan §8): nothing here changes live reward, actions,
replay or normalisation. Imports numpy/scipy; torch only inside
ActorCriticFallback methods.
"""

from .planner import (REAL_PROVENANCES, SOURCES, ActorCriticFallback, BoundedPlanner,
                      Decision, PlanningController, PlanResult, RegionMemory,
                      ReliabilityMonitor)
from .skills import (VERDICTS, BehaviourSignature, BetaCompetence, DuplicateFinding,
                     LearnedSkill, ProbeSet, SetClassifier, behaviour_signature,
                     behavioural_divergence, build_probe_set, calibration, find_duplicates,
                     interval_coverage, modal_agreement, probe_hash, smdp_return, smdp_target)
from .consolidation import (STATUSES, EpisodeMemory, KnowledgeStore, RetentionMonitor,
                            load_planning_state, save_planning_state)
from .skill_bank_bridge import legacy_to_skill, read_registry, read_skill_bank

__all__ = [
    "REAL_PROVENANCES", "SOURCES", "ActorCriticFallback", "BoundedPlanner", "Decision",
    "PlanningController", "PlanResult", "RegionMemory", "ReliabilityMonitor",
    "BehaviourSignature", "BetaCompetence", "DuplicateFinding", "LearnedSkill",
    "ProbeSet", "SetClassifier", "build_probe_set", "VERDICTS", "modal_agreement",
    "behaviour_signature", "behavioural_divergence", "calibration", "find_duplicates",
    "interval_coverage", "probe_hash", "smdp_return", "smdp_target",
    "STATUSES", "EpisodeMemory", "KnowledgeStore", "RetentionMonitor",
    "load_planning_state", "save_planning_state",
    "legacy_to_skill", "read_registry", "read_skill_bank",
]
