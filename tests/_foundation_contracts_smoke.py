"""Foundation contracts smoke (2026-10-03) — plan Stage 2 completion gate.

WHAT IS CLAIMED

    "improvement plan.md" §6 Stage 2 gate: "existing SkyBot data can travel
    through the interfaces, and a minimal nonspatial adapter can satisfy the
    same contract without fabricated spatial variables." Plus the §7.4
    regressions this layer is the first line of defence for: "Reset splices
    and unrelated streams cannot become valid training transitions",
    "Evaluator information cannot reach learning", "A model cannot certify
    its own imagined outcomes as real evidence".

WHY IT IS WORTH A CONTRACT

    The repo has paid for every one of these already. The replay buffer once
    risked teaching the model "a splice between two unrelated worlds as if
    it were dynamics" (tests/_replay_sampling_smoke.py). RED sensors are
    kept from the policy only because `read_policy` never visits them
    (developmental_ai/sensors/registry.py) — a new record layer that
    flattened provenance would quietly reopen that door. And a sensor with
    no reading reports NEUTRAL, so any layer that copies transport values
    into records turns every missing frame into a confident claim.

Contracts:
    A. SkyBot sensor-bus readings -> Observations -> JSON -> back -> the
       transport vector is BIT-IDENTICAL to SensorBus.read_policy, for a
       full context and a degenerate one; stored replay rows round-trip
       the same way; the records satisfy the adapter batch contract.
    B. RED stays evaluator-only: true_position arrives with its real value
       and provenance "evaluator", is absent from policy_channels and from
       policy_observations, cannot enter the transport even when its record
       is relabelled "sensor", cannot be filed as sensor evidence, and never
       forms a trainable transition.
    C. Absence is reported, not neutralised: a missing frame yields ABSENT
       records while the transport still carries neutral — and a real frame
       yields non-ABSENT, non-neutral records (so C is not vacuous).
    D. Cross-stream and reset splices: across two interleaved streams with
       resets and client recoveries, the pairs transition_valid accepts are
       EXACTLY the within-episode consecutive steps of the run log, while
       naive arrival-order pairing would have produced splices (counted, so
       the test is known to bite).
    E. The nonspatial adapter passes check_adapter_conformance with discrete
       AND continuous actions, with and without client recovery, and carries
       no spatial variable: no frame, no ID, no spatial key. The spatial
       scanner is itself shown to fire on SkyBot's spec.
    F. The conformance check is not a rubber stamp: six broken adapters
       (reused episode id, recovery keeping the episode, accepting an out-of-
       spec command, evaluator data labelled sensor, a missing required
       channel, an unset t_complete) each FAIL with the matching problem.
    G. The RSSM wrapper turns imagine_step into Prediction/StateBelief records
       that round-trip, whose outcome is a normalised categorical, and that
       are SHADOW-SAFE: parameters and the global torch RNG are untouched
       (while a bare imagine_step does advance the RNG). An imagined outcome
       cannot be filed as sensor evidence.
    H. Schema evolution end to end: a nested v0 Observation inside an
       Evidence record decodes only through a registered migration; a future
       nested version is refused with SchemaVersionError.

Run: PYTHONPATH=. python tests/_foundation_contracts_smoke.py
"""
import json
import sys

sys.path.insert(0, ".")

import numpy as np

from developmental_ai.foundation.contracts import (
    ABSENT, INAPPLICABLE, Action, Evidence, MigrationRegistry, Observation,
    ObsRef, RecordValidationError, SchemaVersionError, from_dict, from_json,
    to_dict, to_json, transition_valid, UNKNOWN, ContractError)
from developmental_ai.foundation.adapters import (
    NonspatialAdapter, RSSMPredictor, SkyBotRecordAdapter,
    check_adapter_conformance, observation_batch_problems, policy_observations)
from developmental_ai.foundation.contracts.errors import AdapterConformanceError
from developmental_ai.sensors import build_default_bus
from developmental_ai.sensors.builtins import (AUDIO_CHANNELS, AUDIO_FRAMES,
                                               AUDIO_MELS)

PD = 13
FOV = 8
ALL = ["proprio", "screen_fx", "fovea_native", "fovea_delta", "dead_reckon",
       "motion", "light", "sky", "audio", "hud", "true_position"]


def _bus():
    return build_default_bus(PD, lambda c: np.linspace(0, 1, PD, dtype=np.float32),
                             enabled=ALL, fovea_size=FOV)


def _ctx(rng, full=True):
    w = {"x": 123.5, "y": 70.0, "z": -42.25, "yaw": 30.0, "dr_x": 1.0,
         "dr_z": -2.0, "move_fwd": 0.3, "move_lat": 0.0, "move_dy": 0.0,
         "fall_dist": 0.0}
    if not full:
        return {"world": w, "info": {}, "pov_native": None, "prev_fovea": None}
    frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    prev = rng.integers(0, 256, (FOV, FOV, 3), dtype=np.uint8)
    audio = rng.random((AUDIO_CHANNELS, AUDIO_MELS, AUDIO_FRAMES)).astype(np.float32)
    return {"world": w, "info": {"audio": audio}, "pov_native": frame,
            "prev_fovea": prev}


def _json_trip(obs):
    return [from_json(to_json(o)) for o in obs]


# --------------------------------------------------------------------- A

def test_skybot_data_round_trips_bit_identically():
    rng = np.random.default_rng(0)
    bus = _bus()
    ad = SkyBotRecordAdapter(bus, stream="skybot-0")
    spec = ad.observation_spec()
    for full in (True, False):
        ctx = _ctx(rng, full)
        obs = ad.observations_from_ctx(ctx, episode="ep1", seq=4, t_env=4.0, t_wall=1e9)
        assert observation_batch_problems(obs, spec, 4) == [], \
            observation_batch_problems(obs, spec, 4)
        back = _json_trip(obs)
        assert back == obs, "records changed through JSON"
        pol = policy_observations(back, spec)
        rebuilt = ad.transport_from_observations(pol)
        want = bus.read_policy(ctx)
        assert rebuilt.dtype == want.dtype and rebuilt.tobytes() == want.tobytes(), \
            f"transport differs after records (full={full})"
        rows = ad.observations_from_transport(want, "ep1", 4, 4.0, 1e9,
                                              layout_hash=bus.layout_hash())
        again = ad.transport_from_observations(_json_trip(rows))
        assert again.tobytes() == want.tobytes(), "stored row did not round-trip"
    try:
        ad.observations_from_transport(want, "ep1", 4, 4.0, 1e9, layout_hash="0" * 16)
        raise AssertionError("mismatched layout hash was read anyway")
    except ContractError:
        pass
    print(f"  A. {len(obs)} sensor channels (width {bus.width}) through records "
          f"and JSON: transport bit-identical, full and degenerate ctx; replay "
          f"row round-trips; wrong layout hash refused")


# --------------------------------------------------------------------- B

def test_red_stays_evaluator_only():
    rng = np.random.default_rng(1)
    bus = _bus()
    ad = SkyBotRecordAdapter(bus)
    spec = ad.observation_spec()
    obs = ad.observations_from_ctx(_ctx(rng), "ep1", 0, 0.0, 1.0)
    red = [o for o in obs if o.channel == "true_position"]
    assert len(red) == 1 and red[0].provenance == "evaluator"
    assert np.allclose(red[0].value, [123.5, 70.0, -42.25]), \
        "RED reading is not the real position — isolation would be vacuous"
    assert "true_position" not in spec.policy_channels()
    assert spec.evaluator_channels() == ("true_position",)
    assert all(o.provenance == "sensor" for o in policy_observations(obs, spec))
    assert "true_position" not in {o.channel for o in policy_observations(obs, spec)}
    smuggled = red[0].replace(provenance="sensor")
    pol = policy_observations(obs, spec)
    for attempt in (pol + red, pol + [smuggled]):
        try:
            ad.transport_from_observations(attempt)
            raise AssertionError("evaluator data placed in the policy transport")
        except ContractError:
            pass
    assert policy_observations([smuggled], spec) == [], \
        "relabelled RED record passed the policy gate"
    try:
        Evidence("ev", tuple(obs), (), (), "valid", "sensor")
        raise AssertionError("sensor evidence accepted an evaluator observation")
    except RecordValidationError:
        pass
    nxt = ad.observations_from_ctx(_ctx(rng), "ep1", 1, 1.0, 2.0)
    r0, r1 = red[0], [o for o in nxt if o.channel == "true_position"][0]
    assert not transition_valid(r0, r1), "RED formed a training transition"
    p0 = [o for o in obs if o.channel == "proprio"][0]
    p1 = [o for o in nxt if o.channel == "proprio"][0]
    assert transition_valid(p0, p1), "GREEN control transition rejected"
    print("  B. true_position (real value) is evaluator-only: not a policy "
          "channel, dropped by the gate even when relabelled, refused by the "
          "transport and by sensor Evidence, never a transition")


# --------------------------------------------------------------------- C

def test_absence_is_reported_not_neutralised():
    rng = np.random.default_rng(2)
    bus = _bus()
    ad = SkyBotRecordAdapter(bus)
    deg = {o.channel: o for o in ad.observations_from_ctx(_ctx(rng, False), "e", 0, 0., 0.)}
    full = {o.channel: o for o in ad.observations_from_ctx(_ctx(rng, True), "e", 0, 0., 0.)}
    for ch in ("fovea_native", "fovea_delta", "audio", "hud"):
        assert deg[ch].value is ABSENT and deg[ch].missingness == {"value": ABSENT}, ch
        assert full[ch].value is not ABSENT, f"{ch} ABSENT with a real frame"
        neutral = bus.get(ch).neutral().reshape(full[ch].value.shape)
        assert not np.array_equal(full[ch].value, neutral), \
            f"{ch}: real reading equals neutral; C would be vacuous"
    vec = bus.read_policy(_ctx(rng, False))
    sl = bus.slice_of("fovea_native", vec)
    assert np.array_equal(sl, bus.get("fovea_native").neutral()), \
        "bus no longer writes neutral for a missing frame"
    print("  C. missing frame -> ABSENT records (transport still neutral); "
          "real frame -> real, non-neutral records")


# --------------------------------------------------------------------- D

def _run_streams(n_steps=120):
    """Two adapters stepped alternately, as a two-client collector would.
    Returns (arrival-order observation list, set of true transitions)."""
    rng = np.random.default_rng(3)
    ads = [NonspatialAdapter(stream="lever-0", dropout=0.0, recovery_every=17, seed=1),
           NonspatialAdapter(stream="lever-1", dropout=0.0, max_steps=9, seed=2)]
    arrival, truth = [], set()
    cur = [a.reset() for a in ads]
    for o in cur:
        arrival += o
    for _ in range(n_steps):
        for i, a in enumerate(ads):
            head = cur[i][0]
            act = Action(head.environment, head.stream, head.episode, head.seq,
                         a.action_spec().spec_id, a.action_spec().sample(rng),
                         UNKNOWN, 0.0)
            obs, term, trunc, info = a.step(act)
            arrival += obs
            if not info["client_recovery"]:
                prev = {o.channel: o for o in cur[i]}
                for o in obs:
                    if o.channel in prev and o.provenance == "sensor":
                        truth.add((prev[o.channel].ref(), o.ref()))
            cur[i] = obs
            if term or trunc:
                cur[i] = a.reset()
                arrival += cur[i]
    return arrival, truth


def test_no_cross_stream_or_reset_splices():
    arrival, truth = _run_streams()
    sensor = [o for o in arrival if o.provenance == "sensor"]
    accepted = set()
    for a in sensor:
        for b in sensor:
            if transition_valid(a, b):
                accepted.add((a.ref(), b.ref()))
    assert accepted == truth, (
        f"accepted {len(accepted)} vs true {len(truth)}; extra "
        f"{list(accepted - truth)[:3]} missing {list(truth - accepted)[:3]}")
    # Naive pairing: consecutive same-channel records in ARRIVAL order — what
    # a buffer that ignores scope would sample.
    naive, splices = 0, 0
    last = {}
    for o in sensor:
        p = last.get(o.channel)
        if p is not None:
            naive += 1
            if not transition_valid(p, o):
                splices += 1
        last[o.channel] = o
    assert splices > 0, "fixture produced no splices; D would be vacuous"
    try:
        a, b = [o for o in sensor if o.stream == "lever-0"][0], \
               [o for o in sensor if o.stream == "lever-1"][0]
        Evidence("mix", (a, b), (), (), "valid", "sensor")
        raise AssertionError("Evidence mixed two streams")
    except RecordValidationError:
        pass
    eps = {(o.stream, o.episode) for o in arrival}
    print(f"  D. {len(sensor)} sensor records, 2 streams, {len(eps)} episodes: "
          f"accepted transitions == run-log truth ({len(truth)}); naive arrival "
          f"pairing would have spliced {splices}/{naive}")


# --------------------------------------------------------------------- E

_SPATIAL = {"x", "y", "z", "xy", "xyz", "pos", "position", "location", "coords",
            "yaw", "pitch", "heading", "frame", "orientation", "velocity"}


def _spatial_hits(spec, observations):
    hits = []
    for c in spec.channels:
        if set(c.name.lower().split("_")) & _SPATIAL:
            hits.append(f"channel {c.name}")
        if c.frame is not INAPPLICABLE:
            hits.append(f"channel {c.name} frame={c.frame!r}")
    if spec.frames():
        hits.append(f"frames {spec.frames()}")

    def keys(v, path):
        if isinstance(v, dict):
            for k, e in v.items():
                if set(k.lower().split("_")) & _SPATIAL:
                    hits.append(f"key {path}.{k}")
                keys(e, f"{path}.{k}")
        elif isinstance(v, (list, tuple)):
            for e in v:
                keys(e, path)
    for o in observations:
        keys(o.payload, o.channel)
        if '"__id__"' in to_json(o):
            hits.append(f"scoped id in {o.channel}")
    return hits


def test_nonspatial_adapter_satisfies_the_same_contract():
    for kind in ("discrete", "box"):
        for rec in (None, 11):
            rep = check_adapter_conformance(
                NonspatialAdapter(action_kind=kind, recovery_every=rec), n_steps=150)
            assert rep.steps == 150 and rep.episodes > 1
            assert rep.action_kind == kind
            if rec:
                assert rep.recoveries > 0, "recovery path not exercised"
    ad = NonspatialAdapter(dropout=0.3)
    obs = ad.reset(seed=4)
    rng = np.random.default_rng(4)
    seen = list(obs)
    absent = 0
    for _ in range(40):
        h = obs[0]
        obs, t, tr, _i = ad.step(Action(h.environment, h.stream, h.episode, h.seq,
                                        "lever.discrete", int(rng.integers(3)),
                                        UNKNOWN, 0.0))
        seen += obs
        absent += sum(o.value is ABSENT for o in obs)
        if t or tr:
            obs = ad.reset()
            seen += obs
    hits = _spatial_hits(ad.observation_spec(), seen)
    assert hits == [], f"nonspatial adapter fabricated geometry: {hits}"
    assert absent > 0, "dropout never produced an ABSENT reading"
    sky = SkyBotRecordAdapter(_bus())
    sky_hits = _spatial_hits(sky.observation_spec(),
                             sky.observations_from_ctx(_ctx(rng), "e", 0, 0., 0.))
    assert "channel true_position" in sky_hits, \
        f"spatial scanner cannot see SkyBot's true_position: {sky_hits}"
    print(f"  E. nonspatial adapter conforms (discrete+box, with/without "
          f"recovery); 0 spatial fields, {absent} ABSENT readings; scanner "
          f"fires on SkyBot ({len(sky_hits)} hits incl. true_position)")


# --------------------------------------------------------------------- F

class _ReusesEpisode(NonspatialAdapter):
    def _begin_episode(self):
        super()._begin_episode()
        self._episode = f"{self.stream}/ep1"


class _RecoveryKeepsEpisode(NonspatialAdapter):
    def _begin_episode(self):
        keep = self._episode
        super()._begin_episode()
        if keep is not None:
            self._episode = keep


class _AcceptsAnything(NonspatialAdapter):
    def step(self, action):
        if action.command == self._aspec.n:
            action = action.replace(command=0)
        return super().step(action)


class _LeaksEvaluator(NonspatialAdapter):
    def _observe(self):
        return [o.replace(provenance="sensor") if o.channel == "regime" else o
                for o in super()._observe()]


class _DropsRequired(NonspatialAdapter):
    def _observe(self):
        return [o for o in super()._observe() if o.channel != "regime"]


class _NoCompletion(NonspatialAdapter):
    def step(self, action):
        obs, t, tr, info = super().step(action)
        info["executed_action"] = action
        return obs, t, tr, info


def test_conformance_check_catches_broken_adapters():
    cases = [(_ReusesEpisode(max_steps=3), "reused"),
             (_RecoveryKeepsEpisode(recovery_every=3), "recovery kept the old episode"),
             (_AcceptsAnything(), "out-of-spec command: was accepted"),
             (_LeaksEvaluator(), "provenance 'sensor' != declared 'evaluator'"),
             (_DropsRequired(), "required channel 'regime' missing"),
             (_NoCompletion(), "t_complete not set")]
    for ad, needle in cases:
        try:
            check_adapter_conformance(ad, n_steps=40)
        except AdapterConformanceError as e:
            assert any(needle in p for p in e.problems), \
                f"{type(ad).__name__}: failed for the wrong reason: {e.problems[:3]}"
            continue
        raise AssertionError(f"{type(ad).__name__} passed conformance")
    print(f"  F. {len(cases)} broken adapters each rejected for the right reason")


# --------------------------------------------------------------------- G

def test_rssm_wrapper():
    import torch
    from developmental_ai.world_model.rssm import RSSM
    from developmental_ai.foundation.contracts import ActionSpec, Scope
    torch.manual_seed(0)
    rssm = RSSM(obs_dim=8, action_dim=3, stochastic_size=4, stochastic_classes=5,
                deterministic_size=16, hidden_size=16)
    assert rssm.training, "fixture expects the default (stochastic) mode"
    spec = ActionSpec.discrete("mc.buttons", 3)
    pred = RSSMPredictor(rssm, spec, model_version="wm-v1", snapshot_id="snap-0001")
    st = rssm.initial_state(1, torch.device("cpu"))
    before = {k: v.clone() for k, v in rssm.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    ref = ObsRef(Scope("minecraft", "skybot-0", "ep1"), 0, "proprio")
    b = pred.belief(st, "bel-1", support=(ref,))
    p = pred.predict(st, "bel-1", [0, 2, 1, 1], "pred-1", t_wall=5.0, seed=7)
    assert torch.equal(torch.get_rng_state(), rng_before), \
        "shadow prediction advanced the live RNG"
    for k, v in rssm.state_dict().items():
        assert torch.equal(v, before[k]), f"prediction modified {k}"
    assert all(q.grad is None for q in rssm.parameters())
    pr = p.outcome["probs"]
    assert pr.shape == (4, 4, 5) and np.allclose(pr.sum(-1), 1.0, atol=1e-5)
    assert p.horizon == 4 and p.model_versions == {"world_model.rssm": "wm-v1"}
    assert p.snapshot_id == "snap-0001" and p.source_belief == b.belief_id
    p2 = pred.predict(st, "bel-1", [0, 2, 1, 1], "pred-2", t_wall=5.0, seed=7)
    assert np.array_equal(p.outcome["z_sample"], p2.outcome["z_sample"])
    for r in (b, p):
        assert from_json(to_json(r)) == r, f"{r.SCHEMA} round trip"
    assert b.uncertainty == {"h": INAPPLICABLE, "z": UNKNOWN}
    # Falsify the RNG claim: the bare API DOES advance the global stream.
    r0 = torch.get_rng_state().clone()
    with torch.no_grad():
        rssm.imagine_step(st, torch.zeros(1, 3))
    assert not torch.equal(torch.get_rng_state(), r0), \
        "imagine_step is deterministic here; the fork test proves nothing"
    try:
        pred.predict(st, "bel-1", [3], "bad")
        raise AssertionError("out-of-spec candidate action accepted")
    except RecordValidationError:
        pass
    try:
        RSSMPredictor(rssm, ActionSpec.discrete("x", 4), "v", "s")
        raise AssertionError("spec/action_dim mismatch accepted")
    except ContractError:
        pass
    # An imagined outcome cannot be filed as sensor evidence.
    imagined = Observation("minecraft", "skybot-0", "ep1", 1, 1.0, 5.0, "latent",
                           "imagined", {"value": p.outcome["h"][0],
                                        "prediction": p.prediction_id})
    try:
        Evidence("ev-x", (imagined,), (), (p.prediction_id,), "valid", "sensor")
        raise AssertionError("imagined outcome certified as sensor evidence")
    except RecordValidationError:
        pass
    Evidence("ev-i", (imagined,), (), (p.prediction_id,), "valid", "imagined")
    print("  G. RSSM -> StateBelief/Prediction: probs (4,4,5) normalised, "
          "versions+snapshot recorded, round-trips; params and global RNG "
          "untouched (bare imagine_step does advance it); imagined != sensor")


# --------------------------------------------------------------------- H

def test_schema_evolution_end_to_end():
    ad = NonspatialAdapter(dropout=0.0)
    obs = ad.reset(seed=0)
    ev = Evidence("ev", tuple(o for o in obs if o.provenance == "sensor"), (), (),
                  "valid", "sensor")
    d = json.loads(to_json(ev))
    inner = d["fields"]["observations"]["__tuple__"][0]["__record__"]
    inner["version"] = 0
    inner["fields"]["sense"] = inner["fields"].pop("channel")
    try:
        from_dict(d)
        raise AssertionError("v0 nested record decoded with no migration")
    except SchemaVersionError:
        pass
    reg = MigrationRegistry()
    reg.register("observation", 0, 1, lambda f: dict(
        {k: v for k, v in f.items() if k != "sense"}, channel=f["sense"]))
    back = from_dict(d, registry=reg)
    assert back == ev, "migrated record differs from the original"
    inner["version"] = 2
    try:
        from_dict(d, registry=reg)
        raise AssertionError("future nested version accepted")
    except SchemaVersionError:
        pass
    print("  H. nested v0 Observation decodes only via a registered migration "
          "(and then equals the original); nested v2 refused")


if __name__ == "__main__":
    test_skybot_data_round_trips_bit_identically()
    test_red_stays_evaluator_only()
    test_absence_is_reported_not_neutralised()
    test_no_cross_stream_or_reset_splices()
    test_nonspatial_adapter_satisfies_the_same_contract()
    test_conformance_check_catches_broken_adapters()
    test_rssm_wrapper()
    test_schema_evolution_end_to_end()
    print("[foundation-contracts] ALL PASS")
