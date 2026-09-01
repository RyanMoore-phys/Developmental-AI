"""SignalMonitor smoke (pure python, no env, no torch).

Contracts:
  1. A constant-1.0 signal (the stuck-predicate incident) carries 0 bits,
     0 variance, and lands in degenerate() — even when fed out-of-range
     values that clamp back to the same constant.
  2. A signal hovering at 0.5 with real (+/-0.05-scale) jitter binarises
     flat — low entropy — but its above-floor variance keeps it OUT of
     degenerate(): both floors must be violated to demote.
  3. An alternating 0/1 signal measures ~1 bit and is healthy.
  4. A degenerate signal that starts varying leaves the degenerate set as
     the rolling window forgets its constant era (recovery is automatic).
  5. Below min_obs a signal is never listed anywhere (bits, variance,
     degenerate, report) — no verdicts on warming-up signals.
  6. report() format: "signal bits: worst [name=0.00b, ...]", ascending by
     bits, truncated to `top`; "" when nothing has min_obs.
  7. observe() never raises: NaN dropped (counted), inf/out-of-range
     clamped, uncoercible garbage contained (counted), non-str names ok.
"""
import math
import random

from developmental_ai.infra.signal_health import SignalMonitor


def test_constant_is_degenerate():
    m = SignalMonitor(window=1500, min_obs=400)
    for _ in range(450):
        m.observe("stuck_true", 1.0)
    for _ in range(50):
        m.observe("stuck_true", 3.7)          # clamps to 1.0: still constant
    assert m.bits()["stuck_true"] == 0.0, m.bits()
    assert m.variance()["stuck_true"] == 0.0, m.variance()
    assert "stuck_true" in m.degenerate()
    print("[infra-signal-health] 1. constant-1.0 -> 0 bits, 0 var, degenerate"
          " (clamped 3.7 stays constant)")


def test_half_with_variance_not_degenerate():
    m = SignalMonitor()
    rng = random.Random(0)
    for _ in range(500):
        # One-sided jitter: values in [0.5, 0.55] binarise to all-1s (flat
        # decisions) while the analog stream carries genuine spread.
        m.observe("hover", 0.5 + 0.05 * rng.random())
    b, v = m.bits()["hover"], m.variance()["hover"]
    assert b < m.entropy_floor, f"expected flat binarisation, got {b} bits"
    assert v > m.var_floor, f"expected real variance, got {v}"
    assert "hover" not in m.degenerate(), "live analog signal was demoted"
    print("[infra-signal-health] 2. hovering-at-0.5 w/ real variance: low "
          f"entropy ({b:.2f}b) but NOT degenerate (var {v:.1e} > floor)")


def test_alternating_is_healthy():
    m = SignalMonitor()
    for i in range(500):
        m.observe("alt", float(i % 2))
    b = m.bits()["alt"]
    assert 0.99 < b <= 1.0 + 1e-9, b
    assert "alt" not in m.degenerate()
    print(f"[infra-signal-health] 3. alternating 0/1 -> {b:.3f} bits, healthy")


def test_recovery_as_window_rolls():
    m = SignalMonitor(window=100, min_obs=50)
    for _ in range(150):
        m.observe("s", 1.0)
    assert "s" in m.degenerate(), "constant era not flagged"
    for i in range(100):                       # fully replaces the window
        m.observe("s", float(i % 2))
    assert "s" not in m.degenerate(), "signal stayed demoted after recovery"
    assert m.bits()["s"] > 0.9, m.bits()
    print("[infra-signal-health] 4. degenerate signal that starts varying "
          "leaves the set as the window rolls")


def test_below_min_obs_never_listed():
    m = SignalMonitor(window=100, min_obs=50)
    for _ in range(49):
        m.observe("young", 1.0)
    assert "young" not in m.bits()
    assert "young" not in m.variance()
    assert "young" not in m.degenerate()
    assert m.report() == "", repr(m.report())
    m.observe("young", 1.0)                    # the 50th observation
    assert "young" in m.bits() and "young" in m.degenerate()
    print("[infra-signal-health] 5. below min_obs: absent from bits/variance/"
          "degenerate/report; listed at exactly min_obs")


def test_report_format_and_top():
    m = SignalMonitor(min_obs=10)
    for i in range(20):
        m.observe("flat", 1.0)
        m.observe("also_flat", 0.0)
        m.observe("mid", float(i % 2))
    r = m.report(top=2)
    assert r.startswith("signal bits: worst ["), r
    assert r.endswith("]"), r
    assert "flat=0.00b" in r and "also_flat=0.00b" in r, r
    assert "mid" not in r, f"top=2 must drop the healthiest signal: {r}"
    r3 = m.report(top=3)
    assert "mid=1.00b" in r3 and r3.index("flat") < r3.index("mid"), r3
    print("[infra-signal-health] 6. report lists lowest-entropy first, "
          "honours top, exact 'name=X.XXb' format")


def test_defensive_never_raises():
    m = SignalMonitor(window=100, min_obs=3)
    m.observe("g", float("nan"))               # dropped, not fabricated
    m.observe("g", float("inf"))               # clamps to 1.0
    m.observe("g", -7)                         # clamps to 0.0
    m.observe("g", "0.75")                     # coercible: accepted
    m.observe("g", "junk")                     # uncoercible: contained
    m.observe(12345, 0.5)                      # non-str name: coerced
    assert m._dropped == 1, m._dropped
    assert m._errors.get("observe", 0) == 1, m._errors
    assert len(m._streams["g"]) == 3           # inf, -7, "0.75"; NaN absent
    assert "12345" in m._streams
    v = m.variance()["g"]
    assert math.isfinite(v) and m.bits()["g"] > 0.0
    assert isinstance(m.report(), str)         # summary survives the garbage
    print("[infra-signal-health] 7. NaN dropped+counted, inf/range clamped, "
          "garbage contained as data, nothing raised")


if __name__ == "__main__":
    for fn in (test_constant_is_degenerate,
               test_half_with_variance_not_degenerate,
               test_alternating_is_healthy,
               test_recovery_as_window_rolls,
               test_below_min_obs_never_listed,
               test_report_format_and_top,
               test_defensive_never_raises):
        fn()
    print("[infra-signal-health] ALL PASS")
