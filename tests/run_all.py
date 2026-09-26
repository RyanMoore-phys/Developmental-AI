"""Run every offline suite. The entry point CI and the pod both use.

TIERS, because they have different costs and different meanings:

  unit      parts: shapes, bounds, neutrals, degenerate inputs. Seconds.
            A failure here is a bug in a function.
  contract  design: the arguments each wave was built on, with the measured
            numbers in the docstrings. Tens of seconds (some train a tiny
            model). A failure here means a DESIGN claim stopped holding.
  legacy    the suites that predate this work and must keep passing —
            they are the regression surface for everything older.

Exit code is the number of failing suites, so a shell can gate on it.

Run: PYTHONPATH=. python tests/run_all.py [unit|contract|legacy|all]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, ".")

UNIT = [
    "tests/unit/test_sensors_unit.py",
    "tests/unit/test_world_model_unit.py",
    "tests/unit/test_spatial_slots_unit.py",
]

CONTRACT = [
    "tests/_perspective_smoke.py",
    "tests/_sensor_bus_smoke.py",
    "tests/_oracle_isolation_smoke.py",
    "tests/_spatial_memory_smoke.py",
    "tests/_fine_aim_smoke.py",
    "tests/_replay_sampling_smoke.py",
    "tests/_config_keys_smoke.py",
    # REGISTERED 2026-09-25. This file has existed for weeks and was listed
    # NOWHERE — not here, not in ci.yml — so it had never run once. It gates
    # the boredom gate, the daydream decay, the max_bonus clamp and
    # "uncertainty must out-pay a dreamed jackpot"; all real contracts that
    # were simply switched off. Same gap class as _vision_host_smoke.py.
    "tests/_imagination_curiosity_smoke.py",
]

# These need the full stack (gymnasium / minerl) and so only run on the pod.
# ---- KEPT IN LOCKSTEP WITH .github/workflows/ci.yml (2026-09-20) ----
# This list was hand-written and had SIX entries while CI gated on
# THIRTEEN, so a push failed on `_reward_fixes_smoke` — a suite the
# local runner had never executed. A runner that is a subset of CI is
# worse than no runner: it reports ALL SUITES PASS and means something
# narrower than the reader assumes.
#
# The contract in tests/_config_keys_smoke.py's spirit applies here
# too — `_ci_parity` below fails if the two lists drift apart again.
LEGACY = [
    "tests/_reward_fixes_smoke.py",
    "tests/_centering_drive_smoke.py",
    "tests/_resume_smoke.py",
    "tests/_stall_fixes_smoke.py",
    "tests/_no_scripted_skills_smoke.py",
    "tests/_gui_farm_smoke.py",
    "tests/_throughput_wave_smoke.py",
    "tests/_data_diet_smoke.py",
    "tests/_capacity_wave_smoke.py",
    "tests/_anticipation_smoke.py",
    "tests/_learning_rate_wave_smoke.py",
    "tests/_skybot_boot_smoke.py",
    "tests/_metrics_sink_smoke.py",
]


def run(path: str, timeout: int = 900):
    if not os.path.isfile(path):
        return ("MISSING", 0.0, "")
    t0 = time.time()
    env = dict(os.environ, PYTHONPATH=".")
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", time.time() - t0, "")
    dt = time.time() - t0
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        # ---- "COULD NOT RUN" IS NOT "FAILED" (2026-09-20) ---------------
        # The legacy suites import the full env stack. On a dev Mac without
        # gymnasium/minerl they abort at import, and reporting that as FAIL
        # is a lie in the opposite direction from the one this runner was
        # just fixed for: ten red lines that mean nothing, which trains the
        # reader to ignore red. They DO run on CI and the pod, where the
        # deps exist, so the honest report is SKIP-with-a-reason here and a
        # real result there.
        #
        # Deliberately narrow: only a missing HEAVY dependency at import
        # counts. An AssertionError, or a missing dep the repo owns, is
        # still a failure.
        for dep in ("gymnasium", "minerl", "minedojo", "crafter", "minigrid"):
            if f"No module named '{dep}'" in out:
                return (f"SKIP:{dep}", dt, out)
        return ("FAIL", dt, out)
    return ("PASS", dt, out)


def _ci_parity():
    """CI and this runner must gate on the same suites.

    A local runner that omits suites CI enforces is how a green local run
    turns into a red push — which is exactly how `_reward_fixes_smoke`
    slipped through. Read CI's list rather than trusting a copy.
    """
    import re
    path = os.path.join(".github", "workflows", "ci.yml")
    if not os.path.isfile(path):
        return []
    ci = open(path).read()
    if "TESTS=(" not in ci:
        return []
    block = ci[ci.index("TESTS=("):]
    block = block[:block.index("\n            )")] if "\n            )" in block else block
    want = set(re.findall(r"tests/_\w+\.py", block))
    return sorted(want - set(LEGACY))


def main(which: str = "all") -> int:
    groups = {"unit": UNIT, "contract": CONTRACT, "legacy": LEGACY}
    if which == "all":
        order = [("unit", UNIT), ("contract", CONTRACT), ("legacy", LEGACY)]
    else:
        order = [(which, groups[which])]

    drift = _ci_parity()
    if drift:
        print(f"\n!! LOCAL RUNNER IS BEHIND CI — not gating on: {drift}")
        print("   Add them to LEGACY, or a green local run means less than "
              "it says.")
    failures, missing, skipped = [], [], []
    for label, paths in order:
        print(f"\n=== {label} ===")
        for path in paths:
            status, dt, out = run(path)
            name = os.path.basename(path)
            print(f"  {status:8s} {dt:6.1f}s  {name}")
            if status == "MISSING":
                missing.append(name)
            elif status.startswith("SKIP:"):
                skipped.append((name, status.split(":", 1)[1]))
            elif status != "PASS":
                failures.append((name, out))

    if missing:
        print(f"\nMISSING (not run): {', '.join(missing)}")
    if skipped:
        deps = sorted({d for _n, d in skipped})
        print(f"\nSKIPPED — {len(skipped)} suite(s) need {', '.join(deps)}, "
              f"which this machine does not have. They RUN on CI and the pod; "
              f"a green result here does not cover them.")
    if failures:
        print(f"\n{len(failures)} suite(s) FAILED\n")
        for name, out in failures:
            print(f"---- {name} ----")
            print("\n".join(out.strip().splitlines()[-25:]))
            print()
    else:
        cover = "ALL SUITES PASS"
        if skipped:
            cover += f" (of {len(skipped)} fewer than CI — see SKIPPED above)"
        print(f"\n{cover}")
    return len(failures)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "all"))
