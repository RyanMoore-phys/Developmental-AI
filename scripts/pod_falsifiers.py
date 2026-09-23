#!/usr/bin/env python3
"""Stage 4 — the falsifiers. Not pass/fail; the plots that catch being paid
for doing nothing.

Every one of these corresponds to a way this project has ALREADY lost time:
a wage for standing still, a sky exploit, a guard that became a latch. They
are checked against the metrics dump rather than live so they can be re-run
on any archived run.

Run on the pod:
    PYTHONPATH=. python scripts/pod_falsifiers.py --metrics podlogs/metrics.json
"""
import argparse
import json
import sys

import numpy as np


def _arr(m, k):
    return np.asarray(m.get(k) or [], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="podlogs/metrics.json")
    a = ap.parse_args()
    m = json.load(open(a.metrics))
    verdicts = []

    def say(name, bad, detail):
        verdicts.append((name, bad))
        print(f"  [{'ALARM' if bad else 'ok   '}] {name}: {detail}")

    inc, moved = _arr(m, "flow_residual"), _arr(m, "moved")
    if len(inc) and len(moved) == len(inc):
        still = inc[moved < 0.02]
        bad = len(still) > 10 and float(still.mean()) > 1e-4
        say("1_wage_for_standing_still", bad,
            f"mean flow residual at moved<0.02 is {float(still.mean()):.6f} "
            f"over {len(still)} steps (want ~0)")
    else:
        say("1_wage_for_standing_still", False, "no paired series; skipped")

    pitch = _arr(m, "pitch")
    if len(inc) and len(pitch) == len(inc) and inc.std() > 0 and pitch.std() > 0:
        r = float(np.corrcoef(inc, pitch)[0, 1])
        say("2_sky_rebuilt", abs(r) > 0.3,
            f"corr(income, pitch) = {r:+.3f} (want |r| < 0.3)")
    else:
        say("2_sky_rebuilt", False, "no paired series; skipped")

    mask = _arr(m, "flow_mask")
    if len(mask):
        tail = float(mask[-min(20, len(mask)):].mean())
        say("3_automask_starving", tail < 0.3,
            f"retained fraction {tail:.2f} (want 0.6-0.95; <0.3 means the "
            f"photometric loss is starving itself)")

    occ = _arr(m, "occupancy_max")
    if len(occ):
        tail = float(occ[-min(20, len(occ)):].mean())
        say("4_occupancy_compounding", tail > 0.98,
            f"mean max confidence {tail:.2f} (near 1.0 everywhere means the "
            f"clamp is being defeated and the agent believes it is walled in)")

    walk = _arr(m, "walk_mean")
    if len(walk):
        tail = float(walk[-min(20, len(walk)):].mean())
        say("5_walkability_collapse", tail < 0.05,
            f"mean walkability {tail:.2f} (near 0 means it has concluded "
            f"nothing is passable — the belief that stops an agent trying)")

    flow = _arr(m, "flow_loss")
    if len(flow) > 20:
        first, last = float(flow[:10].mean()), float(flow[-10:].mean())
        say("6_flow_not_learning", last >= first,
            f"flow_loss {first:.4f} -> {last:.4f} "
            f"({'FALLING' if last < first else 'NOT falling'})")

    hz = _arr(m, "horizon_loss")
    if len(hz) > 20:
        first, last = float(hz[:10].mean()), float(hz[-10:].mean())
        say("7_horizon_not_learning", last >= first,
            f"horizon_loss {first:.4f} -> {last:.4f}")

    alarms = [n for n, bad in verdicts if bad]
    print(f"\n{len(alarms)} alarm(s)" + (f": {alarms}" if alarms else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
