"""foundation.adapters — environments and existing SkyBot components as records.

    EnvironmentAdapter, check_adapter_conformance, ConformanceReport,
    observation_batch_problems, policy_observations
                                              the contract (protocol.py)
    SkyBotRecordAdapter                       sensor bus <-> records (skybot.py)
    RSSMPredictor                             RSSM -> StateBelief/Prediction
                                              (skybot_world_model.py; torch is
                                              imported only when predicting)
    NonspatialAdapter                         lever-box fixture, no geometry

None of these is wired into the live loop (Offline/Shadow, plan §8).
"""

from .protocol import (ConformanceReport, EnvironmentAdapter,
                       check_adapter_conformance, observation_batch_problems,
                       policy_observations)
from .skybot import SkyBotRecordAdapter
from .skybot_world_model import RSSMPredictor
from .nonspatial import NonspatialAdapter

__all__ = ["ConformanceReport", "EnvironmentAdapter", "check_adapter_conformance",
           "observation_batch_problems", "policy_observations",
           "SkyBotRecordAdapter", "RSSMPredictor", "NonspatialAdapter"]
