"""foundation.transfer — Stage 12: the same core across environments, and
the runtime work that is allowed only after profiling.

    envs         CueResponseAdapter — a second nonspatial adapter, different
                 from adapters.NonspatialAdapter (vector cue, reward channel)
    variations   AppearanceChange, DynamicsShift, ControlRemap,
                 ChannelDropout, AddActuators — one factor each (FACTORS)
    conformance  run_conformance_suite / assert_suite_conforms
    learners     TabularQLearner, run_learning, state_key (fixtures)
    evaluate     transfer_eval, applicability, TransferBudget
    parity       assert_native_parity, NativeParityError, ParityReport
    profiling    profile_callable, Sections, format_profile

Offline only (plan §8). No native code exists: see
docs/foundation/NATIVE_RUNTIME.md for why, and for what would justify it.
"""

from .envs import CueResponseAdapter
from .variations import (FACTORS, AddActuators, AppearanceChange, ChannelDropout,
                         ControlRemap, DynamicsShift)
from .conformance import assert_suite_conforms, run_conformance_suite
from .learners import TabularQLearner, reading, run_learning, state_key
from .evaluate import TransferBudget, applicability, transfer_eval
from .parity import NativeParityError, ParityReport, assert_native_parity
from .profiling import Sections, format_profile, profile_callable

__all__ = ["CueResponseAdapter", "FACTORS", "AddActuators", "AppearanceChange",
           "ChannelDropout", "ControlRemap", "DynamicsShift",
           "assert_suite_conforms", "run_conformance_suite", "TabularQLearner",
           "reading", "run_learning", "state_key", "TransferBudget",
           "applicability", "transfer_eval", "NativeParityError",
           "ParityReport", "assert_native_parity", "Sections",
           "format_profile", "profile_callable"]
