"""foundation.perception — learned representation and persistent perception
(plan Stage 8). Offline/shadow only: nothing here touches live reward,
actions, replay or normalisation.

Modules:
  belief_bridge  slots / spatial memory / flow depth -> contracts StateBelief
  identity       IdentityTracker: probabilistic correspondence, track states,
                 occlusion + bounded re-identification, swap ambiguity
  families       object / field / discrete-process state families; declared
                 selection (declare_family), compare_families
  revision       bounded split/merge proposals, ResidualStore
  versioning     RepresentationRegistry: semver, dependents, translators

  oracle         PRIVILEGED evaluator diagnostics — deliberately NOT exported
                 here; import `foundation.perception.oracle` explicitly, and
                 only from evaluation code.
"""

from .errors import (PerceptionError, PrivilegedInputError, StaleDependentError,
                     UndeclaredFamilyError)
from .versioning import DEPENDENT_KINDS, Dependent, RepresentationRegistry, SemVer
from .identity import (TRACK_STATES, TRACKER_REPRESENTATION, IdentityTracker,
                       StepResult, Track, TrackerConfig)
from .families import (DiscreteProcessFamily, FamilyComparison,
                       FamilyDeclaration, FieldFamily, ObjectFamily,
                       StateFamily, compare_families, declare_family,
                       detect_peaks)
from .revision import (Proposal, ResidualStore, RevisionEngine,
                       RevisionOutcome)
from .belief_bridge import (DEPTH_REPRESENTATION, OCCUPANCY_REPRESENTATION,
                            SLOTS_REPRESENTATION, WALKABILITY_REPRESENTATION,
                            default_registry, depth_belief,
                            inverse_depth_from_flow,
                            inverse_depth_quantity, occupancy_belief,
                            quantity_from_value, quantity_to_value,
                            slot_beliefs, slot_detections, walkability_belief)

__all__ = [n for n in dir() if not n.startswith("_")]
