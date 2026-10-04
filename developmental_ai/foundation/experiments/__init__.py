"""foundation.experiments — comparisons that are argued before they are run.

    ab    plan §7.2 learning-comparison harness: Preregistration (frozen,
          hashed, timestamped), DataSplit / make_split (episode-level),
          run_ab, decide, write_report / load_report, demo_experiment.
          CLI: tools/ab_compare.py

  Stage 10 — curiosity as experiment selection:
    candidates   Candidate, CandidateSet, enumerate_candidates (ActionSpec
                 commands x durations + bounded Skill records only; capped,
                 re-drawn subsets), MAX_CANDIDATES, MAX_HORIZON
    information  hypothesis_information (I(H;O) on a finite set),
                 targeted_information (I(Q;O): the experiment's QUESTION,
                 nuisance factors marginalised and logged apart),
                 bald_categorical / bald_gaussian / ensemble_information /
                 bald_from_ensemble (relevant outputs only, leave-one-out
                 jackpot control, DEFAULT_INFO_CAP), update_from_real_outcome
                 (sensor evidence of the executed action only),
                 ImaginedEvidenceError
    selection    TermSpec, default_terms, check_terms, TermMetadataError,
                 ExperimentSelector (logged per-term breakdown, epsilon
                 floor, no hard gates), CostAccount, effort_of, InfraSink
                 (infra.ledger + infra.signal_health, unmodified)
    progress     DevProbeSet, ProgressMeter, guard_dev_only,
                 HeldOutAccessError (held-out ids never steer selection)
    question     ExperimentQuestion (targeted info, the one real-evidence
                 path, open/provisional/resolved/misspecified status,
                 conclusion() marks unreliable Experiment records),
                 MisspecificationMonitor (anytime e-detector model check;
                 OR-ed with inference's own flag when HypothesisSet has one)
    forgetting   RetentionTracker (net, relearning-discounted progress;
                 noise-aware state changes; forgetting-loop flags with a
                 reachable recovery)
    fixture      DistractedCueAdapter (cue->response box + noisy TV, optional
                 learnable-useless lamp and misspecified response map),
                 CueShiftHypotheses, run_identification, make_arm, ARMS

Offline only (plan §8): nothing here touches the live loop.
"""

from .ab import (BASES, DIRECTIONS, MIN_SEEDS, VERDICTS, ABReport, ABResults,
                 DataSplit, Preregistration, PreregistrationError,
                 PreregistrationViolation, RunOutcome, compare_arms, decide,
                 demo_experiment, load_report, make_split,
                 paired_bootstrap_ci, run_ab, t_quantile_975, value_at,
                 write_report)
from .candidates import (MAX_CANDIDATES, MAX_HORIZON, Candidate, CandidateSet,
                         enumerate_candidates)
from .information import (DEFAULT_INFO_CAP, ImaginedEvidenceError,
                          bald_categorical, bald_from_ensemble, bald_gaussian,
                          check_real_evidence, ensemble_information,
                          hypothesis_information, targeted_information,
                          update_from_real_outcome)
from .question import (MISSPECIFIED, OPEN, PROVISIONAL, RESOLVED, ExperimentQuestion,
                       MisspecificationMonitor, bears_on_question)
from .selection import (TERM_NAMES, CostAccount, ExperimentSelector, InfraSink,
                        Selection, TermMetadataError, TermSpec, check_terms,
                        default_terms, effort_of)
from .forgetting import RetentionTracker
from .progress import (DevProbeSet, HeldOutAccessError, Probe, ProgressMeter,
                       guard_dev_only)
from .fixture import (ARMS, CueShiftHypotheses, DistractedCueAdapter, make_arm,
                      run_identification, slip_probability)

__all__ = ["BASES", "DIRECTIONS", "MIN_SEEDS", "VERDICTS", "ABReport",
           "ABResults", "DataSplit", "Preregistration", "PreregistrationError",
           "PreregistrationViolation", "RunOutcome", "compare_arms", "decide",
           "demo_experiment", "load_report", "make_split",
           "paired_bootstrap_ci", "run_ab", "t_quantile_975", "value_at",
           "write_report",
           "MAX_CANDIDATES", "MAX_HORIZON", "Candidate", "CandidateSet",
           "enumerate_candidates", "DEFAULT_INFO_CAP", "ImaginedEvidenceError",
           "bald_categorical", "bald_from_ensemble", "bald_gaussian",
           "ensemble_information", "hypothesis_information",
           "update_from_real_outcome", "check_real_evidence", "targeted_information",
           "MISSPECIFIED", "OPEN", "PROVISIONAL", "RESOLVED", "ExperimentQuestion",
           "MisspecificationMonitor", "bears_on_question", "TERM_NAMES", "CostAccount",
           "ExperimentSelector", "InfraSink", "Selection", "TermMetadataError",
           "TermSpec", "check_terms", "default_terms", "effort_of",
           "RetentionTracker", "DevProbeSet", "HeldOutAccessError", "Probe",
           "ProgressMeter", "guard_dev_only", "ARMS", "CueShiftHypotheses",
           "DistractedCueAdapter", "make_arm", "run_identification",
           "slip_probability"]
