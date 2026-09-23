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
]

# These need the full stack (gymnasium / minerl) and so only run on the pod.
LEGACY = [
    "tests/_gui_farm_smoke.py",
    "tests/_wm_upgrade_unit.py",
    "tests/_no_scripted_skills_smoke.py",
    "tests/_action_widening_smoke.py",
    "tests/_proprioception_smoke.py",
    "tests/_wm_telemetry_smoke.py",
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
        return ("FAIL", dt, out)
    return ("PASS", dt, out)


def main(which: str = "all") -> int:
    groups = {"unit": UNIT, "contract": CONTRACT, "legacy": LEGACY}
    if which == "all":
        order = [("unit", UNIT), ("contract", CONTRACT), ("legacy", LEGACY)]
    else:
        order = [(which, groups[which])]

    failures, missing = [], []
    for label, paths in order:
        print(f"\n=== {label} ===")
        for path in paths:
            status, dt, out = run(path)
            name = os.path.basename(path)
            print(f"  {status:8s} {dt:6.1f}s  {name}")
            if status == "MISSING":
                missing.append(name)
            elif status != "PASS":
                failures.append((name, out))

    if missing:
        print(f"\nMISSING (not run): {', '.join(missing)}")
    if failures:
        print(f"\n{len(failures)} suite(s) FAILED\n")
        for name, out in failures:
            print(f"---- {name} ----")
            print("\n".join(out.strip().splitlines()[-25:]))
            print()
    else:
        print("\nALL SUITES PASS")
    return len(failures)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "all"))
