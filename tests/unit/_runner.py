"""A tiny test runner, because this repo has no pytest and should not gain one.

WHY NOT PYTEST. Every test in this project is a standalone `__main__` that
prints numbered contract lines and ends with "[name] ALL PASS". That
convention exists so a test is readable as an ARGUMENT — the docstring names
the live incident, the measured numbers and the contracts — and so it can be
run on a pod with nothing installed. Adding a framework would trade that for
a dependency and a different output format.

This gives the unit layer the same shape with ~40 lines: a `case`
decorator, a `run_all` that reports per-case, and assertion helpers that
produce messages worth reading at 3am.
"""

from __future__ import annotations

import math
import traceback
from typing import Callable, List, Tuple

_CASES: List[Tuple[str, Callable]] = []


def case(fn: Callable) -> Callable:
    """Register a unit case. Name is taken from the function name."""
    _CASES.append((fn.__name__, fn))
    return fn


def run_all(label: str) -> int:
    """Run every registered case. Returns the number of failures."""
    passed, failed = 0, []
    for name, fn in _CASES:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed.append((name, e, traceback.format_exc()))
    for name, e, tb in failed:
        print(f"  FAIL {name}: {e}")
        print("       " + tb.strip().splitlines()[-3].strip())
    total = passed + len(failed)
    print(f"[{label}] {passed}/{total} unit cases pass")
    if not failed:
        print(f"[{label}] ALL PASS")
    return len(failed)


def close(a, b, tol=1e-6, what=""):
    if not (abs(float(a) - float(b)) <= tol):
        raise AssertionError(f"{what or 'values'} differ: {a} vs {b} (tol {tol})")


def between(x, lo, hi, what=""):
    if not (lo <= float(x) <= hi):
        raise AssertionError(f"{what or 'value'} {x} outside [{lo}, {hi}]")


def finite(arr, what=""):
    import numpy as np
    a = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(a)):
        bad = int((~np.isfinite(a)).sum())
        raise AssertionError(
            f"{what or 'array'} has {bad} non-finite values; a NaN here "
            f"propagates into the RSSM posterior and poisons every latent "
            f"downstream of it, silently and permanently")


def raises(fn, exc=Exception, what=""):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{what or 'call'} did not raise {exc.__name__}")
