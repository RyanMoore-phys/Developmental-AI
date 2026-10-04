"""foundation.runtime — run identity, evaluation splits, snapshots, paths.

    manifest         reproducible run manifest + compare_manifests
    assumptions      machine-readable assumption register (checked vs code)
    splits           deterministic episode-level dev / held-out assignment
    snapshots        immutable versioned model snapshots for inference
    collection_path  which collection body a config selects (used by the
                     loop at construction; refuses silent downgrades)
    baseline         baseline measurement from an existing runlogs/ dir
    shadow           ShadowRecorder: live steps -> EvidenceStore records,
                     predictions logged before outcomes (plan §8 Shadow;
                     OFF unless foundation.shadow.enabled)

Import-light: no torch at import time (snapshots imports it lazily), because
core/developmental_loop.py imports collection_path while booting.
"""

from .collection_path import (
    CollectionPathDecision, UnsupportedCollectionPath, select_collection_path,
    LIFELONG_SEGMENT, PARALLEL_EPISODE, SINGLE_ENV_LEGACY, ESCAPE_KEY)
from .splits import (
    SPLIT_SALT, SPLIT_VERSION, DEV, HELDOUT, SplitLeakError, EpisodeKey,
    assign_split, assign_frame, assert_no_leak, partition, split_spec)
from .snapshots import (
    Snapshot, SnapshotRegistry, SnapshotCorruption, stamp_prediction)
from .manifest import (
    build_manifest, compare_manifests, resolve_config, validate_manifest,
    write_manifest, load_manifest, UNKNOWN, ABSENT)
from .assumptions import (
    Assumption, CATEGORIES, register, register_mismatches,
    policy_visible_sensors, evaluator_only_sensors)
from .baseline import baseline_report, format_report
from .shadow import ShadowRecorder, rssm_prior_probs, shadow_config

__all__ = [
    "CollectionPathDecision", "UnsupportedCollectionPath",
    "select_collection_path", "LIFELONG_SEGMENT", "PARALLEL_EPISODE",
    "SINGLE_ENV_LEGACY", "ESCAPE_KEY",
    "SPLIT_SALT", "SPLIT_VERSION", "DEV", "HELDOUT", "SplitLeakError",
    "EpisodeKey", "assign_split", "assign_frame", "assert_no_leak",
    "partition", "split_spec",
    "Snapshot", "SnapshotRegistry", "SnapshotCorruption", "stamp_prediction",
    "build_manifest", "compare_manifests", "resolve_config",
    "validate_manifest", "write_manifest", "load_manifest", "UNKNOWN",
    "ABSENT",
    "Assumption", "CATEGORIES", "register", "register_mismatches",
    "policy_visible_sensors", "evaluator_only_sensors",
    "baseline_report", "format_report",
    "ShadowRecorder", "rssm_prior_probs", "shadow_config",
]
