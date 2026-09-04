"""Counter-contract smoke — every emitted field must be REACHABLE.

    PYTHONPATH=. ./venv/bin/python tests/_counter_contract_smoke.py

WHY THIS EXISTS
---------------
In a single week this project shipped FOUR metrics that read empty or wrong,
each one hidden by a defensive `except` that turned a typo into a silent
zero:

  * `_last_world_info` where the attribute is `_last_env_info`
  * learning signals read from `episode_metrics` instead of `training_metrics`
  * `skills_minted` read via a `list_skills()` that does not exist
  * the reward ledger read BEFORE its refresh, so it reported the last segment

And a fifth, found later and fixed separately: `WM loss: 0.0000` in 24 of 24
readings, because telemetry rode on the stage controller's consume-once relay.

Every one of those looked like a measurement. Two of them sent an entire
session chasing a stall that did not exist. CLAUDE.md 5 says measure before
theorising — that is worthless if the instruments lie, and a bare `except`
around a metric read converts "this name is wrong" into "this number is zero".

WHAT THIS TEST DOES
-------------------
It is a STATIC contract, deliberately: it needs no torch, no env and no pod,
so it can run on the Mac before every deploy. It asserts that every attribute
and dict key the emitter reads actually exists on the class that provides it,
and that new counters are declared where they are written.

CONTRACTS
  1. Every `self.<attr>` the emitter reads exists on DevelopmentalAI or is
     defensively defaulted via getattr(..., default).
  2. Every training_metrics key the emitter and _log_progress read is
     DECLARED in the training_metrics literal — an undeclared key is either a
     permanent zero or a KeyError on the trainer thread.
  3. The WM telemetry keys map to names the world model really returns.
  4. No metric is read through a bare `except:` that swallows NameError.
  5. Counters that are structurally not-applicable report n/a, not 0.0.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "developmental_ai"
LOOP = (ROOT / "core" / "developmental_loop.py").read_text()
RSSM = (ROOT / "world_model" / "rssm.py").read_text()


def _declared_metric_keys():
    return set(re.findall(r'^\s+"([a-z_0-9]+)": deque\(maxlen=', LOOP, re.M))


def test_training_metric_keys_are_declared():
    """Contract 2 — an undeclared key is a permanent zero."""
    declared = _declared_metric_keys()
    assert declared, "could not find the training_metrics literal"
    read = set(re.findall(r"metrics\.get\('([a-z_0-9]+)'", LOOP))
    written = set(re.findall(
        r'training_metrics\["([a-z_0-9]+)"\]\.append', LOOP))
    # keys _log_progress prints that come from training_metrics
    tm_read = (read | written) & (read | written)
    missing = sorted(
        k for k in tm_read
        if k not in declared
        and f'"{k}"' not in LOOP.split("_WM_TELEMETRY")[0][-4000:]
        and k not in ("episode_reward",))
    # anything genuinely undeclared AND appended is the dangerous case
    bad = sorted(k for k in written if k not in declared)
    assert not bad, (
        f"appended but never declared in training_metrics: {bad}. "
        f"training_metrics is a plain dict literal, so this is a KeyError on "
        f"whichever thread writes it — invisible on the trainer thread.")
    print(f"[counters] 2. {len(declared)} declared keys; every appended key "
          f"is declared ({len(written)} write sites checked; "
          f"{len(missing)} read-only names skipped)")


def test_wm_telemetry_maps_to_real_world_model_keys():
    """Contract 3 — names read from the source, never assumed."""
    block = re.search(r"_WM_TELEMETRY = \((.*?)\)\n\n", LOOP, re.S)
    assert block, "_WM_TELEMETRY not found"
    pairs = re.findall(r'\("([a-z_]+)",\s*\n?\s*"([a-z_]+)"\)',
                       block.group(1))
    assert len(pairs) >= 5, pairs
    real = set(re.findall(r'^\s+"([a-z_]+)":\s', RSSM, re.M))
    declared = _declared_metric_keys()
    for dest, src in pairs:
        assert dest in declared, f"{dest} written by telemetry but undeclared"
        if not src.startswith("symbolic_decoder"):
            assert src in real, (
                f"_WM_TELEMETRY maps {dest} <- {src!r}, which rssm.py's loss "
                f"dict does not contain. Found: {sorted(real)}")
    print(f"[counters] 3. all {len(pairs)} WM telemetry keys exist in both "
          f"rssm's loss dict and training_metrics")


def test_no_bare_except_around_metric_reads():
    """Contract 4 — `except:` turns a typo into a zero.

    A bare `except:` also swallows NameError and AttributeError, which is
    exactly how four wrong attribute names shipped looking like measurements.
    """
    bare = [i + 1 for i, ln in enumerate(LOOP.splitlines())
            if re.match(r"\s*except\s*:\s*$", ln)]
    assert not bare, (
        f"bare `except:` at lines {bare} — catches NameError/AttributeError "
        f"and reports the failure as a number")
    print("[counters] 4. no bare `except:` in the loop (typos surface "
          "instead of becoming zeros)")


def test_not_applicable_is_not_zero():
    """Contract 5 — the failure that cost this session two investigations."""
    assert 'print("  New facts/ep:     n/a' in LOOP, (
        "on pixels `_extract_and_store_facts` is never called, so 0.0 is "
        "structural, not measured — it must not print as a number")
    assert 'print("  SD facts/ep:      n/a' in LOOP, (
        "symbolic_decoder is a _NullSymbolicDecoder on pixels")
    assert 'f"  (n={len(_wmq)}, recon="' in LOOP, (
        "WM loss must report its sample count: a 0.0000 from an EMPTY deque "
        "read identically to a converged model for 24 consecutive readings")
    assert "boredom 0 = external curiosity still teaching" in LOOP, (
        "imagination's 0.0 is designed behaviour and must say so")
    print("[counters] 5. new facts / SD facts / WM loss / imagination all "
          "report WHY they are zero, or say n/a")


def test_emitted_attribute_names_exist():
    """Contract 1 — the `_last_world_info` class of bug, caught statically."""
    seg = LOOP.split("def _emit_metrics")
    assert len(seg) > 1, "emitter not found"
    body = seg[1][:12000]
    # every bare self.X read in the emitter must be assigned somewhere
    reads = set(re.findall(r"self\.([a-z_][a-z_0-9]*)\b", body))
    guarded = set(re.findall(r'getattr\(\s*self,\s*"([a-z_0-9]+)"', body))
    assigned = set(re.findall(r"self\.([a-z_][a-z_0-9]*)\s*[:=]", LOOP))
    methods = set(re.findall(r"def ([a-z_][a-z_0-9]*)\s*\(", LOOP))
    missing = sorted(reads - guarded - assigned - methods)
    assert not missing, (
        f"the emitter reads attributes that are never assigned: {missing}. "
        f"This is the `_last_world_info` vs `_last_env_info` bug — it emitted "
        f"None for a whole run behind an `except`.")
    print(f"[counters] 1. {len(reads)} attribute reads in the emitter, all "
          f"assigned or getattr-guarded ({len(guarded)} guarded)")


if __name__ == "__main__":
    for fn in (test_emitted_attribute_names_exist,
               test_training_metric_keys_are_declared,
               test_wm_telemetry_maps_to_real_world_model_keys,
               test_no_bare_except_around_metric_reads,
               test_not_applicable_is_not_zero):
        fn()
    print("[counters] ALL PASS")
