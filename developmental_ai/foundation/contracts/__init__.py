"""foundation.contracts — backend-independent records and their rules.

Public API (stable; other foundation subpackages build on exactly this):

    sentinels   UNKNOWN, ABSENT, INAPPLICABLE, is_missing
    ids         Scope, EntityId, FrameId, ObsRef, same_scope
    records     Observation, Action, StateBelief, Mechanism, Prediction,
                Experiment, Evidence, Skill
    specs       ActionSpec, ChannelSpec, ObservationSpec
    codec       to_dict, from_dict, to_json, from_json, register_migration,
                MigrationRegistry, MIGRATIONS
    streams     transition_valid, transition_problems
    errors      ContractError, RecordValidationError, SchemaVersionError,
                AdapterConformanceError
    base        Record, RECORD_TYPES, deep_equal

numpy only; no torch.
"""

from .errors import (AdapterConformanceError, ContractError,
                     RecordValidationError, SchemaVersionError)
from .sentinels import ABSENT, INAPPLICABLE, SENTINELS, UNKNOWN, Sentinel, is_missing
from .ids import EntityId, FrameId, ObsRef, Scope, same_scope
from .values import PROVENANCES, deep_equal
from .base import RECORD_TYPES, Record
from .specs import ActionSpec, ChannelSpec, ObservationSpec
from .records import (Action, Evidence, Experiment, Mechanism, Observation,
                      Prediction, Skill, StateBelief)
from .codec import (MIGRATIONS, MigrationRegistry, from_dict, from_json,
                    register_migration, to_dict, to_json)
from .streams import transition_problems, transition_valid

__all__ = [
    "AdapterConformanceError", "ContractError", "RecordValidationError",
    "SchemaVersionError",
    "ABSENT", "INAPPLICABLE", "UNKNOWN", "SENTINELS", "Sentinel", "is_missing",
    "EntityId", "FrameId", "ObsRef", "Scope", "same_scope",
    "PROVENANCES", "deep_equal", "RECORD_TYPES", "Record",
    "ActionSpec", "ChannelSpec", "ObservationSpec",
    "Action", "Evidence", "Experiment", "Mechanism", "Observation",
    "Prediction", "Skill", "StateBelief",
    "MIGRATIONS", "MigrationRegistry", "from_dict", "from_json",
    "register_migration", "to_dict", "to_json",
    "transition_problems", "transition_valid",
]
