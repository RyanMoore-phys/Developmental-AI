"""Aggregate Rung-6 cross-task transfer results across per-seed results.json.
Prints, per seed and pooled, the whole-curve PRIMARY metrics (mean reward,
frac_solved) for intact vs each control arm, plus the noisier endpoint windows.
Usage: python _vm_analyze.py [SEED ...]   (default: 42 7 123 2024)"""
import json, os, sys, glob

ENV = "MiniGrid-DoorKey-8x8-v0"
ARMS = ["intact", "lesion", "scramble", "constant"]
SEEDS = [int(x) for x in sys.argv[1:]] or [42, 7, 123, 2024]

PRIMARY = [("overall_avg_reward", "mean_rew"), ("frac_solved", "frac_solved")]
SECOND = [("best_window_avg", "best_w"), ("final_window_avg", "final_w")]


def load(seed):
    for p in (f"rung6_transfer_s{seed}/results.json",):
        if os.path.isfile(p):
            return json.load(open(p)).get("runs", {})
    return {}


def get(runs, seed, arm, key):
    r = runs.get(f"seed{seed}_{arm}_{ENV}")
    if not r:
        return None
    return r.get("summary", {}).get(key)


pooled = {arm: {k: [] for k, _ in PRIMARY + SECOND} for arm in ARMS}

for seed in SEEDS:
    runs = load(seed)
    if not runs:
        print(f"\n=== seed {seed}: (no results yet) ===")
        continue
    print(f"\n=== seed {seed} ===")
    hdr = f"{'arm':>9} | " + " | ".join(f"{lab:>11}" for _, lab in PRIMARY + SECOND)
    print(hdr)
    for arm in ARMS:
        vals = {}
        cells = []
        for key, _ in PRIMARY + SECOND:
            v = get(runs, seed, arm, key)
            vals[key] = v
            cells.append(f"{v:>11.3f}" if isinstance(v, (int, float)) else f"{'pending':>11}")
            if isinstance(v, (int, float)):
                pooled[arm][key].append(v)
        print(f"{arm:>9} | " + " | ".join(cells))
    # per-seed intact-vs-control deltas on the primary metrics
    for key, lab in PRIMARY:
        vi = get(runs, seed, "intact", key)
        if not isinstance(vi, (int, float)):
            continue
        ds = []
        for c in ("lesion", "scramble", "constant"):
            vc = get(runs, seed, c, key)
            if isinstance(vc, (int, float)):
                ds.append(f"{c}:{vi - vc:+.3f}")
        if ds:
            print(f"   intact-minus-control [{lab}]: " + "  ".join(ds))

# pooled summary
print("\n=== POOLED across seeds (n per cell shown) ===")
for key, lab in PRIMARY + SECOND:
    print(f"\n[{lab}]")
    base = pooled["intact"][key]
    bmean = sum(base) / len(base) if base else None
    for arm in ARMS:
        xs = pooled[arm][key]
        m = sum(xs) / len(xs) if xs else None
        delta = (f"{bmean - m:+.3f}" if (arm != "intact" and m is not None and bmean is not None) else "")
        ms = f"{m:.3f}" if m is not None else "n/a"
        print(f"  {arm:>9}: mean={ms} (n={len(xs)})  intact-minus={delta}")
