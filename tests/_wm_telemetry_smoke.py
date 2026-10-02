"""World-model telemetry smoke (Mac-ok; no torch, no env, no training host).

    PYTHONPATH=. ./venv/bin/python tests/_wm_telemetry_smoke.py

THE LIVE INCIDENT (measured 2026-09-04)
---------------------------------------
`WM loss: 0.0000` in **24 of 24** progress readings, and the metrics sink
emitted `world_model_loss: None` — while the same run reported
`AsyncWM replay-ratio: 32/32 blocks run (skip 0.0%)` over a 150,000-transition
(12.3 GB) buffer. The trainer was plainly working; the number was a lie.

Cause: telemetry was appended by the SEGMENT CONSUMER in `run()`, which reads
losses through a **consume-once relay** built for the stage controller:

    trainer thread ->  _wm_metrics_box = m
    main thread    ->  _last_wm_metrics = box; _wm_metrics_fresh = True
    consumer       ->  if fresh: append(...); fresh = False

That middle hop only executes inside
``wm_steps >= wm_train_every AND _use_async_wm`` — one instant every 250 steps
— while a training block takes ~71.6s against a ~73.5s cadence. The gate
routinely missed, whichever consumer did see the value cleared `fresh`, and
`training_metrics["world_model_loss"]` stayed EMPTY. `_log_progress` then
printed its own default:

    print(f"  WM loss:  {metrics.get('world_model_loss', 0):.4f}")

A 0.0 that means "no data" is indistinguishable from a 0.0 that means
"converged", so **a learning world model looked exactly like a dead one** —
and the dream, prospection, the policy's latents and empowerment (which rolls
this model forward) all sit downstream of it. Per CLAUDE.md §5, you cannot
measure before theorising if the instrument reads zero.

THE FIX. `_record_wm_telemetry` is called at each of the three sites where
metrics are PRODUCED (async trainer thread / lifelong-synchronous /
non-lifelong), never downstream of the relay. The relay itself is unchanged —
the stage controller still needs consume-once, or re-feeding a stale
reconstruction value flattens its error slope.

CONTRACTS
  1. Telemetry survives a relay that never fires (the exact live failure).
  2. Recorded losses MOVE. A constant non-zero value is the same failure
     wearing a different number, so staleness must fail this test.
  3. One writer per producer — N blocks record N values, not 2N.
  4. Key mapping matches rssm.compute_loss's ACTUAL returned keys.
  5. A missing key is SKIPPED, never recorded as 0.0 (a fake zero is what
     started this).
  6. NaN is rejected before it poisons every downstream mean.
  7. Every telemetry key is declared in `training_metrics`, so no append can
     KeyError on the trainer thread where the traceback is invisible.
  8. The consume-once relay is still intact for the stage controller.
"""
import re
from collections import deque
from pathlib import Path

from developmental_ai.core.developmental_loop import DevelopmentalAI

SRC = Path(__file__).resolve().parents[1] / "developmental_ai"
LOOP_SRC = (SRC / "core" / "developmental_loop.py").read_text()
RSSM_SRC = (SRC / "world_model" / "rssm.py").read_text()


class _Recorder:
    """Minimal stand-in carrying only what _record_wm_telemetry touches.

    Deliberately NOT a real DevelopmentalAI: constructing one needs torch, a
    replay buffer and an env. We bind the REAL unbound method so this tests
    shipping code, not a re-implementation of it.
    """

    def __init__(self):
        self.training_metrics = {
            k: deque(maxlen=100) for k in (
                "world_model_loss", "kl_divergence", "reconstruction_error",
                "inverse_dynamics_loss", "symbolic_decoder_loss",
                "symbolic_decoder_accuracy")}

    _WM_TELEMETRY = DevelopmentalAI._WM_TELEMETRY
    record = DevelopmentalAI._record_wm_telemetry


def _block(i):
    """One finished training block, with a loss that actually descends."""
    return {"total": 10.0 - 0.1 * i, "reconstruction": 8.0 - 0.08 * i,
            "kl": 1.0 + 0.01 * i, "reward": 0.5, "continue": 0.2,
            "inverse": 2.0 - 0.02 * i,
            "symbolic_decoder_loss": 0.9 - 0.005 * i,
            "symbolic_decoder_accuracy": 0.3 + 0.004 * i}


def test_survives_a_relay_that_never_fires():
    """Contract 1 — the live failure, reproduced and then defeated.

    We simulate the consume-once relay MISSING EVERY TIME (the pathological
    end of what happened live) and assert telemetry is recorded anyway,
    because it no longer travels through the relay at all.
    """
    r = _Recorder()
    box = None
    consumer_saw = 0
    for i in range(32):                      # the 32/32 blocks from the log
        m = _block(i)
        box = m                              # trainer: _wm_metrics_box = m
        r.record(m)                          # producer-site recording (fix)
        # main thread's box check never coincides with a finished block:
        if False:                            # noqa: SIM108 — the missed gate
            consumer_saw += 1
    assert consumer_saw == 0, "this test must model a relay that never fires"
    got = list(r.training_metrics["world_model_loss"])
    assert len(got) == 32, (
        f"32 blocks ran but {len(got)} losses recorded — this is the live "
        f"bug: WM loss read 0.0000 in 24/24 readings while the trainer "
        f"reported 32/32 blocks")
    assert box is not None
    print(f"[wm-telemetry] 1. 32/32 blocks recorded through a relay that "
          f"fired 0 times (live bug: 0 recorded, printed as 0.0000)")


def test_the_loss_actually_moves():
    """Contract 2 — a constant value is the same failure, new number."""
    r = _Recorder()
    for i in range(32):
        r.record(_block(i))
    got = list(r.training_metrics["world_model_loss"])
    assert all(v != 0.0 for v in got), "a 0.0 reading is what we are fixing"
    assert len(set(got)) > 1, (
        "recorded loss never changed — a pinned non-zero constant is "
        "indistinguishable from a stale cache and must fail here")
    assert got[-1] < got[0], (
        f"loss should descend as the model trains: {got[0]} -> {got[-1]}")
    # the stage controller's spine must move too, or the curriculum is blind
    rec = list(r.training_metrics["reconstruction_error"])
    assert len(set(rec)) > 1 and rec[-1] < rec[0], rec[:3]
    print(f"[wm-telemetry] 2. loss MOVES {got[0]:.2f} -> {got[-1]:.2f} "
          f"({len(set(got))} distinct); reconstruction {rec[0]:.2f} -> "
          f"{rec[-1]:.2f}")


def test_one_writer_per_producer():
    """Contract 3 — N blocks give N values; no double counting."""
    r = _Recorder()
    for i in range(10):
        r.record(_block(i))
    for key in ("world_model_loss", "kl_divergence", "reconstruction_error",
                "inverse_dynamics_loss", "symbolic_decoder_loss",
                "symbolic_decoder_accuracy"):
        n = len(r.training_metrics[key])
        assert n == 10, f"{key}: 10 blocks -> {n} values (double-counted?)"
    # and the shipped code must call the recorder exactly once per producer
    calls = LOOP_SRC.count("self._record_wm_telemetry(")
    assert calls == 3, (
        f"expected exactly 3 producer-site calls (async trainer, "
        f"lifelong-sync, non-lifelong), found {calls}. §4.2 duplicated-body "
        f"drift: an edit that lands in one path and not another is silent.")
    print(f"[wm-telemetry] 3. 10 blocks -> 10 values on all 6 keys; "
          f"{calls} producer-site calls in the loop")


def test_key_mapping_matches_the_world_model():
    """Contract 4 — names READ from rssm.compute_loss, not assumed.

    Four counters shipped broken this week from guessed attribute names, each
    hidden by a defensive `except`. If someone renames a loss key in the RSSM,
    this fails loudly here instead of silently zeroing a dashboard panel.
    """
    real = set(re.findall(r'"([a-z_]+)":\s*\w+_loss', RSSM_SRC))
    real |= set(re.findall(r'^\s+"([a-z_]+)":\s', RSSM_SRC, re.M))
    for _key, mkey in DevelopmentalAI._WM_TELEMETRY:
        if mkey.startswith("symbolic_decoder"):
            continue                          # added by _train_world_model
        assert mkey in real, (
            f"_WM_TELEMETRY maps {_key!r} <- {mkey!r}, but rssm.py's loss "
            f"dict has no such key. Found: {sorted(real)}")
    # and the consumer that feeds the stage controller uses the same names
    assert 'wm_metrics.get("reconstruction")' in LOOP_SRC
    assert 'wm_metrics.get("kl")' in LOOP_SRC
    print(f"[wm-telemetry] 4. all mapped keys exist in rssm's loss dict "
          f"({sorted(m for _, m in DevelopmentalAI._WM_TELEMETRY)})")


def test_missing_key_is_skipped_not_zeroed():
    """Contract 5 — never invent a 0.0. That is the bug, not the fix."""
    r = _Recorder()
    r.record({"total": 4.2})                  # sd/inverse absent this block
    assert list(r.training_metrics["world_model_loss"]) == [4.2]
    for key in ("symbolic_decoder_loss", "inverse_dynamics_loss",
                "reconstruction_error"):
        assert len(r.training_metrics[key]) == 0, (
            f"{key} recorded a value for a key the trainer never produced — "
            f"a fabricated 0.0 reads as 'converged' and is exactly the "
            f"failure this test exists for")
    r.record(None)                            # trainer skipped (ValueError)
    r.record({})
    assert len(r.training_metrics["world_model_loss"]) == 1
    print("[wm-telemetry] 5. absent keys skipped (not zeroed); None/{} "
          "record nothing")


def test_nan_is_rejected():
    """Contract 6 — one NaN makes every downstream mean NaN, forever."""
    r = _Recorder()
    r.record({"total": 3.0})
    r.record({"total": float("nan")})
    got = list(r.training_metrics["world_model_loss"])
    assert got == [3.0], f"NaN leaked into telemetry: {got}"
    r.record({"total": "not-a-number"})       # must not raise on the thread
    assert list(r.training_metrics["world_model_loss"]) == [3.0]
    print("[wm-telemetry] 6. NaN and non-numeric rejected; no raise")


def test_every_key_is_declared():
    """Contract 7 — an undeclared key would KeyError on the TRAINER THREAD.

    `training_metrics` is a plain dict literal, not a defaultdict, and the
    trainer thread's traceback is invisible: the symptom would be a silently
    dead world model.
    """
    decl = set(re.findall(r'^\s+"([a-z_]+)": deque\(maxlen=', LOOP_SRC, re.M))
    for key, _m in DevelopmentalAI._WM_TELEMETRY:
        assert key in decl, (
            f"{key!r} is written by telemetry but not declared in "
            f"training_metrics")
    assert "reconstruction_error" in decl
    # belt and braces: the writer itself must not index a missing key
    assert "self.training_metrics.setdefault(" in LOOP_SRC
    # a fresh recorder with NO keys must still not raise
    bare = _Recorder()
    bare.training_metrics = {}
    bare.record(_block(0))
    assert len(bare.training_metrics["world_model_loss"]) == 1
    print(f"[wm-telemetry] 7. all {len(DevelopmentalAI._WM_TELEMETRY)} keys "
          f"declared; setdefault survives an empty metrics dict")


def test_consume_once_relay_still_intact():
    """Contract 8 — we fixed telemetry WITHOUT weakening the controller.

    The stage controller must keep seeing each reconstruction value exactly
    once; re-feeding a stale one flattens its error slope and the curriculum
    stops transitioning.
    """
    assert "self._wm_metrics_fresh = False" in LOOP_SRC, \
        "consume-once was removed — the stage controller now sees stale values"
    assert LOOP_SRC.count("self._last_wm_metrics = self._wm_metrics_box") == 1
    assert "prediction_error=wm_metrics.get(\"reconstruction\")" in LOOP_SRC
    # telemetry must NOT have been moved back behind the fresh-gate
    m = re.search(r"if _wm_ready:(.{0,1600})", LOOP_SRC, re.S)
    assert m and "_record_wm_telemetry" in m.group(1), \
        "expected the non-lifelong producer call inside the consumer"
    assert "if not self._lifelong:" in m.group(1), (
        "the consumer must record ONLY in the non-lifelong path, else "
        "lifelong runs double-count every block")
    print("[wm-telemetry] 8. consume-once relay intact for the stage "
          "controller; telemetry does not ride on it")


if __name__ == "__main__":
    for fn in (test_survives_a_relay_that_never_fires,
               test_the_loss_actually_moves,
               test_one_writer_per_producer,
               test_key_mapping_matches_the_world_model,
               test_missing_key_is_skipped_not_zeroed,
               test_nan_is_rejected,
               test_every_key_is_declared,
               test_consume_once_relay_still_intact):
        fn()
    print("[wm-telemetry] ALL PASS")
