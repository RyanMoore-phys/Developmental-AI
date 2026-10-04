"""foundation.experience — the evidence store (plan Stage 5).

The IMMUTABLE RECORD OF RECORD for evaluation and scientific comparison:
observations, executed actions (with actual timing), predictions written
BEFORE their outcomes, outcome evidence, episode endings, and revisable
interpretations kept apart from the raw records. It is NOT a training
replay — world_model/replay_buffer.py keeps sole training ownership for the
live learner, and nothing in the live loop reads or writes this store
(Offline/Shadow, plan §8). See docs/foundation/EXPERIENCE_RECONCILIATION.md.

    store     EvidenceStore, LearningView (dev only), HeldoutView,
              EvaluatorView (all partitions), Interaction/PredictionTrace
              (reconstruction), migrate_store, content_hash, split_scope
    sampling  MixedSampler (representative floor >= 0.25, failure /
              contradiction / coverage / context components, stats())
    scoring   score_point_prediction (reference producer of scores)
    recorder  AdapterRecorder (reference producer: any adapter -> store)
    errors    ExperienceError and its subclasses

numpy only; no torch.
"""

from .errors import (DuplicateRecordError, ExperienceError, OrderingError,
                     PrivacyError, ProvenanceError, RecordEvicted,
                     StoreBudgetError, StoreCorruption)
from .store import (END_REASONS, EVALUATOR, PARTS, EvaluatorView,
                    EvidenceStore, HeldoutView, Interaction, LearningView,
                    PredictionTrace, content_hash, make_ref, migrate_store,
                    parse_ref, split_scope)
from .sampling import (MIN_REPRESENTATIVE, PRIORITY_COMPONENTS, MixedSampler,
                       Reservoir, default_bin)
from .scoring import score_point_prediction
from .recorder import AdapterRecorder

__all__ = [
    "DuplicateRecordError", "ExperienceError", "OrderingError",
    "PrivacyError", "ProvenanceError", "RecordEvicted", "StoreBudgetError",
    "StoreCorruption",
    "END_REASONS", "EVALUATOR", "PARTS", "EvaluatorView", "EvidenceStore",
    "HeldoutView", "Interaction", "LearningView", "PredictionTrace",
    "content_hash", "make_ref", "migrate_store", "parse_ref", "split_scope",
    "MIN_REPRESENTATIVE", "PRIORITY_COMPONENTS", "MixedSampler", "Reservoir",
    "default_bin", "score_point_prediction", "AdapterRecorder",
]
