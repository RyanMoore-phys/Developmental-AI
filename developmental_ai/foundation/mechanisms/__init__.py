"""foundation.mechanisms — structured mechanism prediction (plan Stage 6).

Public API (stable; discovery / experiments / planning build on exactly this):

  api         OutcomeLayout, entity_layout, MixedOutcome (THE predictive
              distribution: mixture of diag-Gaussian x categoricals, with
              mean / variance / decompose_variance / probs_marginal /
              log_prob / log_prob_parts / cdf / sample / entropy / select /
              to_record), OutcomeDistribution and MechanismPredictor
              (protocols), EntityBatch, TransitionBatch, BaseMechanism
              (version, applicability, rollout), MISSING,
              prediction_record, outcomes_from_prediction
  data        EntityStream — Observation/Action records -> transitions and
              multi-step windows; refuses reset splices via
              contracts.streams
  evaluation  score, one_step, horizons, law_diagnostics, resource_cost,
              gaussian_crps, crps_samples, energy_score, coverage,
              pit_histogram
  fixtures    BoxWorld (balls + wall entities, gravity, elastic contacts,
              variable dt, optional obs noise), trajectory_records,
              make_streams (episode-level split), EVENT_CLASSES,
              box_containment_constraint, free_flight_energy_constraint

  torch-backed (imported on first attribute access, so importing this
  package does not pull in torch):
  relational  InteractionNetwork (declared translation symmetry optional)
  baselines   MLPMechanism (same inputs), ConstantVelocityMechanism,
              RSSMMechanism (existing RSSM where comparable)
  residual    ResidualMechanism

Contract for every predictor: predict(state, action (B,N,Ad), dt (B,)) ->
MixedOutcome; fit(TransitionBatch) -> report and version += 1;
applicability(state) -> (B,) in [0,1] or UNKNOWN before any fit;
rollout(state, actions (H,B,N,Ad), dts (H,B), mode="mean"|"sample").

Offline/shadow only (plan §8): nothing here touches the live loop.
"""

from .api import (MISSING, BaseMechanism, EntityBatch, MechanismPredictor,
                  MixedOutcome, OutcomeDistribution, OutcomeLayout,
                  TransitionBatch, entity_layout, outcomes_from_prediction,
                  prediction_record)
from .data import EntityStream
from .evaluation import (coverage, crps_samples, energy_score, gaussian_crps,
                         horizons, law_diagnostics, one_step, pit_histogram,
                         resource_cost, score)
from .fixtures import (EVENT_CLASSES, BoxWorld, box_containment_constraint,
                       free_flight_energy_constraint, make_streams,
                       trajectory_records)

_LAZY = {
    "InteractionNetwork": ".relational",
    "MLPMechanism": ".baselines",
    "ConstantVelocityMechanism": ".baselines",
    "RSSMMechanism": ".baselines",
    "ResidualMechanism": ".residual",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MISSING", "BaseMechanism", "EntityBatch", "MechanismPredictor", "MixedOutcome",
    "OutcomeDistribution", "OutcomeLayout", "TransitionBatch", "entity_layout",
    "outcomes_from_prediction", "prediction_record", "EntityStream",
    "coverage", "crps_samples", "energy_score", "gaussian_crps", "horizons",
    "law_diagnostics", "one_step", "pit_histogram", "resource_cost", "score",
    "EVENT_CLASSES", "BoxWorld", "box_containment_constraint",
    "free_flight_energy_constraint", "make_streams", "trajectory_records",
] + list(_LAZY)
