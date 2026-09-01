"""Pre-boot wave smoke (2026-08-10) — points 1-4 before the next VM boot.

Contracts:
  #1 PERCEPTION PERSISTS: symbolizer.state()/load_state() round-trips the
     grounded heads + reliability + label evidence through torch.save under
     the loader's exact call; a changed vocabulary keeps heads FRESH *and
     UNCERTIFIED* (evidence is the grounding gate, so it must not outlive
     the head that earned it — revised 2026-08-11); the familiarity payload
     survives torch.load(weights_only=True) (sets would not — hence sorted
     lists); env-side territory (_visits) rides breaks_by_type.json as
     "cells", flushes on a STEP floor (not only on events) and is forced on
     close(), and the restore parse rebuilds the exact dict.
  #2 DEGENERATE NEEDS OPPORTUNITY TO VARY: a signal is only flagged
     DEGENERATE once the teacher has observed it TRUE >=
     degenerate_min_positives times. Counting total labels (the first cut)
     gated nothing — one VLM query labels every predicate — and live it
     flagged `tree_visible` in a treeless area, which excluded trees from
     the magnet's targets and pinned it at w=0.0000/target=None. Absence is
     not pathology. No evidence data (None/empty) keeps strict behaviour.
  #3 MEMORY PULL: the phi MATH is pinned (closer pays more, older pays
     less, freshest record wins, missing data reads (None, None)) but the
     term is DISABLED (weight 0) — review 2026-08-11 found sightings are
     stamped at the AGENT's position and re-minted every ~25 primary steps,
     so the baseline re-adopts constantly, telescoping breaks in the
     charging direction, and the residual drain is minimised by standing
     still or sheltering in the GUI. See the config comment for the
     re-enable checklist.
  #4 COMPETENCE FLOOR 0: with the floor at 0.0 a low-competence skill is
     OFFERED (eligibility no longer depends on a statistic that can only
     move once eligibility is granted); 0.25 still gates — and the config
     plumbing has no `or`-default that would resurrect it.
"""
import json
import math
import os
import tempfile

import numpy as np
import torch


class _Bare:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _mk_symbolizer():
    from developmental_ai.llm.vlm_symbolizer import VLMSymbolizer
    return VLMSymbolizer(latent_dim=32, hidden_dim=16, enabled=True,
                         query_fn=lambda b: None, fovea=True,
                         device=torch.device("cpu"))


def test_perception_roundtrip():
    torch.manual_seed(0)
    a = _mk_symbolizer()
    a.reliability["tree_visible"] = 0.93
    a.label_counts["tree_visible"] = 41
    a.pos_counts["tree_visible"] = 17
    a.fovea_label_counts[next(iter(a.fovea_label_counts))] = 9
    a.total_labels = 41
    a._retracted.add("tree_visible:0:0")
    with torch.no_grad():
        for p in a.head.parameters():
            p.add_(1.0)
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "symbolizer.pt")
        torch.save(a.state(), fp)
        st = torch.load(fp, map_location=torch.device("cpu"))
    b = _mk_symbolizer()
    summary = b.load_state(st)
    assert "head" in summary and "FRESH" not in summary, summary
    for (ka, va), (kb, vb) in zip(a.head.state_dict().items(),
                                  b.head.state_dict().items()):
        assert ka == kb and torch.equal(va, vb), f"head weight {ka} differs"
    assert b.label_counts["tree_visible"] == 41
    assert b.pos_counts["tree_visible"] == 17
    assert abs(b.reliability["tree_visible"] - 0.93) < 1e-9
    assert b.total_labels == 41 and "tree_visible:0:0" in b._retracted
    print(f"  1. perception round-trip via loader's torch.load: {summary}; "
          f"head byte-equal, evidence carried")


def test_perception_vocab_mismatch_drops_evidence():
    """A head that was NOT restored must not inherit the old head's
    certification. REVISED 2026-08-11 after review: this test used to assert
    that evidence merges across a vocabulary change. That was wrong and
    dangerous — label_counts/reliability ARE the grounding gate (>=min_labels
    and >=reliability_floor lets a predicate assert facts into the KG at
    confidence p*reliability and mint grounded symbols for reward), so
    restoring them onto a RANDOM head certifies noise as trusted perception,
    and it does not self-correct because most predicates never receive
    disconfirming evidence."""
    torch.manual_seed(1)
    a = _mk_symbolizer()
    a.label_counts["tree_visible"] = 12
    a.reliability["tree_visible"] = 0.95
    a.interval = 450                       # a fully annealed cadence
    a.total_labels = 900
    st = a.state()
    st["predicates"] = list(st["predicates"][:-1]) + ["bogus_pred"]
    st["label_counts"]["bogus_pred"] = 999
    b = _mk_symbolizer()
    before = {k: v.clone() for k, v in b.head.state_dict().items()}
    fresh_rel = b.reliability["tree_visible"]
    summary = b.load_state(st)
    assert "FRESH" in summary, summary
    for k, v in b.head.state_dict().items():
        assert torch.equal(v, before[k]), "head loaded despite vocab change"
    assert b.label_counts["tree_visible"] == 0, \
        "a fresh head inherited label evidence — it is now certified to " \
        "assert facts it never learned"
    assert b.reliability["tree_visible"] == fresh_rel, \
        "a fresh head inherited reliability"
    assert b.total_labels == 0, "fresh head inherited the label total"
    assert b.interval == b.base_interval, \
        "fresh head inherited the ANNEALED cadence — it would be starved " \
        "of the labels it needs to relearn"
    assert "bogus_pred" not in b.label_counts, "unknown key leaked in"
    print(f"  2. vocab change: {summary} — head fresh AND uncertified "
          f"(evidence dropped, anneal reset to {b.base_interval})")


def test_familiarity_weights_only_safe():
    payload = {
        "nov_counts": {"b12": 4, "b7": 1},
        "gaze_counts": {"g1": 3},
        "known_symbols": sorted({"tree_visible", "grass_visible"}),
        "symbol_counts": {"tree_visible": 5},
        "symbol_sight_counts": {"tree_visible": 8},
    }
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "familiarity.pt")
        torch.save(payload, fp)
        m = torch.load(fp, map_location=torch.device("cpu"))
        # the strictest loader torch may ever default to
        m2 = torch.load(fp, map_location=torch.device("cpu"),
                        weights_only=True)
    assert m == payload and m2 == payload
    assert isinstance(m2["known_symbols"], list), "sets don't survive"
    print("  3. familiarity payload survives torch.load AND "
          "weights_only=True (known_symbols as sorted list)")


def test_env_cells_roundtrip():
    from developmental_ai.environments.minerl_env import MineRLEnvAdapter
    visits = {(3, -4): 7, (0, 0): 2, (-120, 88): 1}
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "breaks_by_type.json")
        env = _Bare(_break_memory_path=fp, _break_mem_dirty=19,
                    _breaks_by_type={"log": 1}, _places_by_type={},
                    _crafts_by_type={}, _pickups_by_type={},
                    _visits=dict(visits))
        MineRLEnvAdapter._save_break_memory(env)
        with open(fp) as f:
            m = json.load(f)
    assert m["cells"] == {"3,-4": 7, "0,0": 2, "-120,88": 1}, m["cells"]
    # the restore parse from __init__, verbatim
    restored = {}
    for _ck, _cv in (m.get("cells") or {}).items():
        _x, _z = _ck.split(",")
        restored[(int(_x), int(_z))] = int(_cv)
    assert restored == visits
    print(f"  4. env territory: {len(visits)} cells (incl. negatives) "
          f"round-trip through breaks_by_type.json")


def test_territory_flushes_without_events():
    """Territory must reach disk on a run that EXPLORES but never breaks.

    Review finding 2026-08-11: the flush counter ticks only on
    break/place/craft/pickup, but `cells` changes every step — so hours of
    pure navigation wrote nothing, and shutdown wrote nothing either. This
    pins both the step-paced flush and the forced flush on close()."""
    from developmental_ai.environments.minerl_env import MineRLEnvAdapter

    class _FakeEnv(_Bare):
        # borrow the REAL methods so close() can find _save_break_memory
        # (a bare stub would hit close()'s except and prove nothing)
        _save_break_memory = MineRLEnvAdapter._save_break_memory
        close = MineRLEnvAdapter.close

    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "breaks_by_type.json")
        env = _FakeEnv(_break_memory_path=fp, _break_mem_dirty=0,
                       _breaks_by_type={}, _places_by_type={},
                       _crafts_by_type={}, _pickups_by_type={},
                       _visits={}, _visit_writes=0, _last_mem_flush_step=0,
                       _env=_Bare())
        # 1999 explored steps, ZERO events -> not yet due
        for i in range(1999):
            env._visits[(i, 0)] = 1
            env._visit_writes += 1
            MineRLEnvAdapter._save_break_memory(env, event=False)
        assert not os.path.exists(fp), "flushed too eagerly"
        # crossing the step floor writes without a single event
        env._visits[(1999, 0)] = 1
        env._visit_writes += 1
        MineRLEnvAdapter._save_break_memory(env, event=False)
        assert os.path.exists(fp), \
            "explored 2000 steps with no events and never persisted territory"
        n_at_flush = len(json.load(open(fp))["cells"])

        # the per-step caller must NOT consume the event budget
        assert env._break_mem_dirty == 0, "step path bumped the event counter"

        # further exploration below the next floor is unwritten...
        for i in range(50):
            env._visits[(-i - 1, 7)] = 1
            env._visit_writes += 1
            MineRLEnvAdapter._save_break_memory(env, event=False)
        assert len(json.load(open(fp))["cells"]) == n_at_flush
        # ...until shutdown forces it
        env.close()
        final = len(json.load(open(fp))["cells"])
        assert final == n_at_flush + 50, (final, n_at_flush)
    print(f"  9. territory: step-paced flush fires with ZERO events "
          f"({n_at_flush} cells), event budget untouched, close() forces "
          f"the last {final - n_at_flush}")


def test_degenerate_needs_opportunity_to_vary():
    """DEGENERATE requires POSITIVE observations, not elapsed queries.

    Revised 2026-08-11 from live evidence: this gate feeds the magnet's
    exclusion list. Counting total labels gated nothing (one VLM query
    labels every predicate, so the count just tracks time) and the live run
    flagged `tree_visible` DEGENERATE in a treeless spot — which forbade the
    magnet from ever steering toward trees (w=0.0000, target=None). A
    predicate never seen TRUE cannot be pathologically constant."""
    from developmental_ai.infra.stack import InfraStack
    from developmental_ai.infra.signal_health import SignalMonitor
    with tempfile.TemporaryDirectory() as d:
        stack = InfraStack({"enabled": True, "degenerate_min_positives": 10,
                            "log_dir": d}, action_dim=12, log_dir=d)
        assert stack.signals is not None, "signal monitor absent"
        stack.signals = SignalMonitor(window=60, min_obs=20)
        for _ in range(40):                      # a dead-constant signal
            stack.signals.observe("tree_visible", 0.9)
        assert "tree_visible" in stack.signals.degenerate(), \
            "monitor itself never flagged — test setup broken"
        # THE LIVE CASE: hundreds of queries, but the thing was never
        # actually present -> absence, not pathology. Must NOT be excluded,
        # or the magnet can never learn to seek it.
        stack.segment({"step": 1000, "signal_evidence": {"tree_visible": 0}})
        assert "tree_visible" not in stack.degenerate_signals(), \
            "a never-observed predicate was called degenerate — this is " \
            "what disabled the magnet live"
        stack.segment({"step": 1500, "signal_evidence": {"tree_visible": 9}})
        assert "tree_visible" not in stack.degenerate_signals()
        # pathology: seen TRUE often and the head is STILL constant
        stack.segment({"step": 2000, "signal_evidence": {"tree_visible": 30}})
        assert "tree_visible" in stack.degenerate_signals()
        # no evidence data at all -> old strict behaviour
        stack.segment({"step": 3000})
        assert "tree_visible" in stack.degenerate_signals()
        stack.segment({"step": 4000, "signal_evidence": {}})
        assert "tree_visible" in stack.degenerate_signals()
    print("  5. degenerate gate: 0 and 9 POSITIVES -> patience (absence is "
          "not pathology); 30 -> flagged; absent/empty evidence -> strict")


def test_memory_pull_phi():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    from developmental_ai.infra.episodic import EpisodicEventMemory
    mem = EpisodicEventMemory()
    mem.record("sighting", "tree", step=900, position=(32.0, 64.0, 0.0))
    s = _Bare(infra=_Bare(episodic=mem),
              vision_scaffold=_Bare(_seek_cats=["tree"]),
              _last_world_info={"x": 0.0, "z": 0.0},
              total_timesteps=1000)
    phi_far, rec = DevelopmentalAI._memory_pull_phi(s)
    assert phi_far is not None and rec == 900, (phi_far, rec)
    s._last_world_info = {"x": 24.0, "z": 0.0}   # walked 24 blocks closer
    phi_near, _ = DevelopmentalAI._memory_pull_phi(s)
    assert phi_near > phi_far, "approaching must raise phi"
    s.total_timesteps = 61000                    # the memory is old now
    phi_old, _ = DevelopmentalAI._memory_pull_phi(s)
    assert phi_old < 0.1 * phi_near, "old memories must whisper"
    # the freshest record wins and CHANGES the re-adopt key
    mem.record("break", "tree", step=59000, position=(-64.0, 64.0, -64.0))
    phi2, rec2 = DevelopmentalAI._memory_pull_phi(s)
    assert rec2 == 59000 and phi2 != phi_old
    # missing world info / empty memory -> silent (None, None)
    s._last_world_info = {}
    assert DevelopmentalAI._memory_pull_phi(s) == (None, None)
    s2 = _Bare(infra=_Bare(episodic=EpisodicEventMemory()),
               vision_scaffold=_Bare(_seek_cats=["tree"]),
               _last_world_info={"x": 0.0, "z": 0.0}, total_timesteps=10)
    assert DevelopmentalAI._memory_pull_phi(s2) == (None, None)
    print(f"  6. memory pull: near {phi_near:.3f} > far {phi_far:.3f}; "
          f"aged {phi_old:.4f} whispers; freshest record ({rec2}) wins; "
          f"missing data reads (None, None)")


def test_memory_pull_disabled():
    """REVISED 2026-08-11: this asserted skybot ENABLES the pull. Review
    showed the term is net-harmful as built — episodic sightings are stamped
    at the AGENT's position and re-minted every ~25 primary steps, so the
    baseline re-adopts constantly, Ng's telescoping identity breaks in the
    charging direction, and the residual drain is minimised by standing
    still or sheltering in the GUI. It stays at 0 until the target-position
    and landmark-throttle redesign lands."""
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "..", "configs",
                           "minecraft_skybot.yaml")) as f:
        cfg = yaml.safe_load(f)
    cur = cfg.get("curiosity", {})
    assert float(cur.get("memory_pull_weight", 0.0)) == 0.0, \
        ("memory_pull is re-enabled without the redesign — it drains "
         "~25-45% of the intrinsic budget and rewards menu-sheltering")
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    s = _Bare()
    assert float(getattr(s, "_memory_pull_weight", 0.0)) == 0.0
    print("  7. memory_pull_weight: 0.0 (disabled pending redesign); "
          "absent-config default also 0.0")


def test_competence_floor_zero_offers():
    from developmental_ai.policy.options import SkillOptionBank
    P, K = 10, 1                       # K = option-slot count; layout [P|K]

    def _pm(width=None):
        # mirrors SkillOptionBank.primitive_mask: only primitives lit
        m = np.zeros(int(P if width is None else width), dtype=bool)
        m[:P] = True
        return m

    bank = _Bare(P=P, K=K, primitive_mask=_pm, disabled_macros=[],
                 slots=[{"skill_id": "ach_03_break_oak_log",
                         "preconditions": {}}])
    weak = {"ach_03_break_oak_log": 0.05}       # far below the old 0.25
    m0 = SkillOptionBank.mask(bank, predicates=None, min_match=0.4,
                              competence=weak, competence_floor=0.0)
    assert bool(m0[P + 0]), "floor 0.0 must OFFER the weak skill"
    m25 = SkillOptionBank.mask(bank, predicates=None, min_match=0.4,
                               competence=weak, competence_floor=0.25)
    assert not bool(m25[P + 0]), "floor 0.25 should still gate"
    # config plumbing: 0.0 must survive (no `or`-default resurrection)
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "..", "configs",
                           "minecraft_skybot.yaml")) as f:
        cfg = yaml.safe_load(f)
    _sec = next((v for v in cfg.values()
                 if isinstance(v, dict) and "competence_floor" in v), None)
    assert _sec is not None and float(_sec["competence_floor"]) == 0.0
    print("  8. competence floor: 0.0 offers the 0.05-competence skill, "
          "0.25 gates it; skybot config carries 0.0")


if __name__ == "__main__":
    for fn in (test_perception_roundtrip,
               test_perception_vocab_mismatch_drops_evidence,
               test_familiarity_weights_only_safe,
               test_env_cells_roundtrip,
               test_territory_flushes_without_events,
               test_degenerate_needs_opportunity_to_vary,
               test_memory_pull_phi,
               test_memory_pull_disabled,
               test_competence_floor_zero_offers):
        print(f"[preboot-wave] {fn.__name__}")
        fn()
    print("[preboot-wave] ALL PASS")
