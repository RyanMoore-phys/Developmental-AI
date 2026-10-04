"""foundation.inference — beliefs, uncertainty, filtering (plan Stage 7).

Public API (stable):

  ensemble    BootstrapEnsemble(factory, k<=8) — a MechanismPredictor whose
              predictive is the members' mixture; decompose() -> aleatoric
              (mean member variance) / epistemic (variance of member means);
              MAX_MEMBERS
  hypotheses  HypothesisSet — finite set, log-space Bayes with logsumexp,
              probability floor + tempering for recovery from
              overconfidence, bounded size with reversible pruning,
              add_hypothesis(name, prior); update_categorical(table, outcome)
              feeds the MisspecificationMonitor -> hs.misspecified,
              hs.misspecification() (posterior-predictive CUSUM;
              truth-not-in-set is flagged instead of silent)
              categorical_predictive (the mixture's outcome distribution)
  filtering   KalmanFilter (exact linear-Gaussian, Joseph form, partial /
              missing observations), ParticleFilter (ESS-triggered
              systematic resampling, degeneracy diagnostics), ess,
              systematic_resample
  change      PredictiveCUSUM — rule change vs measurement noise from
              standardised predictive residuals (mean-shift + dispersion)

Missing evidence (None / ABSENT / UNKNOWN) is a no-op everywhere here, never
a zero reading. numpy/scipy only, except that the ensemble's members are
whatever the factory builds (torch for the learned mechanisms).
Offline/shadow only (plan §8).
"""

from .change import PredictiveCUSUM
from .ensemble import MAX_MEMBERS, BootstrapEnsemble
from .filtering import KalmanFilter, ParticleFilter, ess, systematic_resample
from .hypotheses import HypothesisSet, MisspecificationMonitor, categorical_predictive

__all__ = ["PredictiveCUSUM", "MAX_MEMBERS", "BootstrapEnsemble", "KalmanFilter",
           "ParticleFilter", "ess", "systematic_resample", "HypothesisSet",
           "MisspecificationMonitor", "categorical_predictive"]
