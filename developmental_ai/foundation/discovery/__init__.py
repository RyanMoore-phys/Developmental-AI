"""foundation.discovery — mechanism discovery and structural revision
(plan Stage 9). Offline/shadow only (plan §8): consumes evidence tables,
produces mechanisms and events; nothing here touches the live loop.

Public API:

  table      EvidenceTable (columns, target, episode per row, intervened
             masks built ONLY from contracts.Action payload "sets"),
             require_dev (final held-out refused), fit_validation_split
             (episode-level, its own salt), VALIDATION_SALT, is_final_heldout
  forms      Condition, Branch, StructuredMechanism (gated branches over the
             transition library constant / linear / piecewise / lookup),
             ComposedMechanism, fit_mechanism (pure), initial_mechanism,
             mechanism_from_record (contracts.Mechanism round trip),
             per_episode_paired, verdict, FitError, BudgetExceeded,
             CancelToken + check_deadline (cooperative cancellation: a
             float deadline search() cancels when it abandons a fit)
  proposals  Proposal (serialisable, parent refs "id@version"), ops
             ADD_DEPENDENCY, SPLIT_APPLICABILITY, COMPOSE,
             REPLACE_TRANSITION, REFIT (parameter-only baseline)
  search     SearchBudget(wall_s, max_candidates, max_bytes, max_rows,
             depth, beam), search -> SearchReport, enumerate_proposals,
             live_abandoned_fits / MAX_ABANDONED_FITS (cap on abandoned
             fit threads still alive; past it search -> "budget_exhausted")
  causal     annotate_causal (observational vs interventional per
             dependency), ablate, with_causal, contradictions
  lifecycle  MechanismRegistry (candidate/probationary/promoted/retired,
             revival, caps, provenance chain, contradictions,
             snapshot/restore)
  revision   ChangeTriggeredReviser (inference.PredictiveCUSUM alarm or
             MisspecificationMonitor flag ->
             revival check -> bounded structural search -> windows)
  fixtures   WORLDS, make_episode, make_split_episodes (known hidden rules)

numpy only; no torch.
"""

from .table import (DEFAULT_VAL_FRACTION, VALIDATION_SALT, DiscoveryError, EvidenceTable,
                    is_final_heldout)
from .forms import (FORMS, Branch, BudgetExceeded, CancelToken, ComposedMechanism, Condition,
                    FitError, StructuredMechanism, check_deadline, fit_mechanism,
                    initial_mechanism,
                    mechanism_from_record, per_episode_paired, verdict)
from .proposals import (ADD_DEPENDENCY, ALL_OPS, COMPOSE, REFIT, REPLACE_TRANSITION,
                        SPLIT_APPLICABILITY, STRUCTURAL_OPS, Proposal, ProposalError)
from .causal import (INTERVENTIONAL, OBSERVATIONAL, ablate, annotate_causal,
                     contradictions, with_causal)
from .search import (MAX_ABANDONED_FITS, Candidate, SearchBudget, SearchReport,
                     enumerate_proposals, live_abandoned_fits, search)
from .lifecycle import (CANDIDATE, PROBATIONARY, PROMOTED, RETIRED, STATES, Entry,
                        MechanismRegistry)
from .revision import ChangeTriggeredReviser
from . import fixtures

__all__ = [n for n in dir() if not n.startswith("_")]
