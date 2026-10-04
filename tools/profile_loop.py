"""Profile a loop before porting any of it (plan Stage 12 item 4).

    PYTHONPATH=. python tools/profile_loop.py MODULE:FUNCTION [--iterations N]
                                              [--top K] [--json OUT]
    PYTHONPATH=. python tools/profile_loop.py --demo

FUNCTION is called with no arguments (plus `sections=` if it accepts one,
so it can time named blocks with `with sections("env_step"): ...`). The
report lists the top CPU hotspots by own time, Python vs builtin/C call
counts (total and per iteration — per-element crossings show up here as a
count that grows with the data), and the section timings.

--demo profiles the transfer fixture (CueResponseAdapter + TabularQLearner,
400 interactions) — a stand-in for the training-host loop. A port decision
needs a profile of the REAL loop on the training host; see
docs/foundation/NATIVE_RUNTIME.md.
"""
import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.foundation.transfer.profiling import (  # noqa: E402
    format_profile, profile_callable)


def demo(sections=None):
    from developmental_ai.foundation.transfer import (CueResponseAdapter,
                                                       TabularQLearner, run_learning)
    with sections("build"):
        env = CueResponseAdapter(seed=0)
        learner = TabularQLearner(env.action_spec(), seed=0)
    with sections("learn_400_steps"):
        run_learning(learner, env, 400, seed=0)


def resolve(target: str):
    mod_name, _, fn_name = target.partition(":")
    if not fn_name:
        raise SystemExit("target must be MODULE:FUNCTION")
    fn = getattr(importlib.import_module(mod_name), fn_name, None)
    if not callable(fn):
        raise SystemExit(f"{target} is not callable")
    return fn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", nargs="?", help="MODULE:FUNCTION")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if not args.demo and not args.target:
        ap.error("give MODULE:FUNCTION or --demo")
    fn = demo if args.demo else resolve(args.target)
    rep = profile_callable(fn, iterations=args.iterations, top=args.top)
    print(format_profile(rep))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
