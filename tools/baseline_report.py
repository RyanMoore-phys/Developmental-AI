"""Baseline behaviour / throughput / memory / WM-error report from runlogs/.

    PYTHONPATH=. python tools/baseline_report.py runlogs [--json out.json]

Reads metrics.jsonl, heartbeat.jsonl (and rotations) and *.log in the given
directory (training-host layout). Unavailable metrics print as "unknown" with
the reason. Runs on an empty or test-only runlogs/ without error.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.foundation.runtime.baseline import (  # noqa: E402
    baseline_report, format_report)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runlogs", nargs="?", default="runlogs")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    rep = baseline_report(args.runlogs)
    print(format_report(rep))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=2, sort_keys=True, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
