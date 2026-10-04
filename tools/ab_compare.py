"""Run a registered A/B experiment and print its report (plan §7.2).

    PYTHONPATH=. python tools/ab_compare.py MODULE[:FUNCTION] --out DIR
    PYTHONPATH=. python tools/ab_compare.py \
        developmental_ai.foundation.experiments.ab:demo_experiment --out /tmp/ab

MODULE:FUNCTION (default FUNCTION `experiment`) must return a dict with
    prereg   a FROZEN foundation.experiments.Preregistration
    arms     {name: callable(seed, DataSplit) -> metrics dict}
    splits   a DataSplit or list of them
It is run with the PREREGISTERED seeds only; there is deliberately no flag
to change them. The JSON + markdown report is written to --out (never
overwriting an earlier one) and the markdown is printed. Exit code: 0 for
accept, 1 for reject, 2 for inconclusive, 3 for a refused verdict.
"""
import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.foundation.experiments import (  # noqa: E402
    PreregistrationViolation, run_ab)

EXIT = {"accept": 0, "reject": 1, "inconclusive": 2}


def load_experiment(target: str):
    mod_name, _, fn_name = target.partition(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name or "experiment", None)
    if fn is None or not callable(fn):
        raise SystemExit(f"{mod_name} has no callable {fn_name or 'experiment'!r}")
    spec = fn()
    missing = [k for k in ("prereg", "arms", "splits") if k not in spec]
    if missing:
        raise SystemExit(f"{target} returned no {missing}")
    return spec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiment", help="module[:function] returning the spec")
    ap.add_argument("--out", required=True, help="report directory")
    ap.add_argument("--no-memory", action="store_true",
                    help="skip tracemalloc (faster; py peak reported as unknown)")
    args = ap.parse_args(argv)
    spec = load_experiment(args.experiment)
    try:
        rep = run_ab(spec["prereg"], spec["arms"], splits=spec["splits"],
                     out_dir=args.out, measure_memory=not args.no_memory)
    except PreregistrationViolation as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    print(rep.to_markdown())
    print(f"report: {rep.json_path}\n        {rep.md_path}")
    return EXIT[rep.verdict["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
