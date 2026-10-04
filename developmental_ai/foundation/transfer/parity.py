"""Python/native numerical parity (plan Stage 12 items 5-6; NATIVE_RUNTIME.md).

No native code exists yet. This is the test TEMPLATE a future Rust/PyO3
port must pass before it may replace its Python reference, and it is
exercised today with two Python functions (tests/_foundation_transfer_smoke.py).

assert_native_parity(py_fn, native_fn, inputs, rtol, atol=0.0)
    `inputs` is a sequence of cases; a case is a tuple (positional args),
    a dict (keyword args), or any other value (one positional arg). Each
    function gets its OWN deep copy of each case, and the check covers:
      - outputs: same structure (tuple/list/dict), same shapes, same dtype
        KIND, values within rtol/atol, NaN/inf in the same places;
      - side effects: the arguments after each call must also match, so a
        native fn that mutates its input differently is caught;
      - exceptions: if one raises, the other must raise the same type.
    Raises NativeParityError naming the case, path and worst error; returns
    a ParityReport with timings of both (the measured gain, if any).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Sequence

import numpy as np


class NativeParityError(AssertionError):
    pass


@dataclass
class ParityReport:
    n_cases: int = 0
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    py_seconds: float = 0.0
    native_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def speedup(self) -> float:
        return self.py_seconds / self.native_seconds if self.native_seconds > 0 else float("inf")


def _call(fn, case):
    if isinstance(case, tuple):
        return fn(*case)
    if isinstance(case, dict):
        return fn(**case)
    return fn(case)


def _compare(a, b, rtol, atol, path, rep):
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or a.keys() != b.keys():
            raise NativeParityError(f"{path}: dict keys differ")
        for k in a:
            _compare(a[k], b[k], rtol, atol, f"{path}[{k!r}]", rep)
        return
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            raise NativeParityError(f"{path}: sequence type/length differ "
                                    f"({type(a).__name__} vs {type(b).__name__})")
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(x, y, rtol, atol, f"{path}[{i}]", rep)
        return
    if a is None or b is None or isinstance(a, str) or isinstance(b, str):
        if a != b:
            raise NativeParityError(f"{path}: {a!r} != {b!r}")
        return
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape:
        raise NativeParityError(f"{path}: shape {x.shape} != {y.shape}")
    if x.dtype.kind != y.dtype.kind:
        raise NativeParityError(f"{path}: dtype kind {x.dtype} != {y.dtype}")
    if x.dtype.kind not in "biufc":
        try:
            same = bool(a == b)
        except Exception:
            same = a is b
        if not same:
            raise NativeParityError(f"{path}: non-numeric values differ")
        return
    if x.dtype.kind in "biu":
        if not np.array_equal(x, y):
            raise NativeParityError(f"{path}: integer/bool values differ")
        return
    if x.dtype.kind == "c":
        x, y = np.stack([x.real, x.imag]), np.stack([y.real, y.imag])
    xf, yf = x.astype(np.float64), y.astype(np.float64)
    if not np.array_equal(np.isnan(xf), np.isnan(yf)) or \
            not np.array_equal(np.isinf(xf), np.isinf(yf)) or \
            not np.array_equal(xf[np.isinf(xf)], yf[np.isinf(yf)]):
        raise NativeParityError(f"{path}: NaN/inf placement differs")
    fin = np.isfinite(xf)
    if fin.any():
        d = np.abs(xf[fin] - yf[fin])
        rel = d / np.maximum(np.abs(xf[fin]), 1e-300)
        rep.max_abs_err = max(rep.max_abs_err, float(d.max()))
        rep.max_rel_err = max(rep.max_rel_err, float(np.where(d == 0, 0.0, rel).max()))
        bad = d > atol + rtol * np.abs(xf[fin])
        if bad.any():
            raise NativeParityError(
                f"{path}: {int(bad.sum())} value(s) outside rtol={rtol:g} "
                f"atol={atol:g}; worst abs err {float(d.max()):.3g}")


def assert_native_parity(py_fn: Callable, native_fn: Callable, inputs: Sequence[Any],
                         rtol: float = 1e-6, atol: float = 0.0) -> ParityReport:
    if rtol < 0 or atol < 0:
        raise ValueError("tolerances must be >= 0")
    cases = list(inputs)
    if not cases:
        raise ValueError("parity needs at least one input case")
    rep = ParityReport()
    for i, case in enumerate(cases):
        ca, cb = copy.deepcopy(case), copy.deepcopy(case)
        ea = eb = None
        t0 = time.perf_counter()
        try:
            ra = _call(py_fn, ca)
        except Exception as e:
            ea = e
        t1 = time.perf_counter()
        try:
            rb = _call(native_fn, cb)
        except Exception as e:
            eb = e
        t2 = time.perf_counter()
        rep.py_seconds += t1 - t0
        rep.native_seconds += t2 - t1
        if (ea is None) != (eb is None) or (ea is not None and type(ea) is not type(eb)):
            raise NativeParityError(f"case {i}: python raised {ea!r}, native raised {eb!r}")
        if ea is None:
            _compare(ra, rb, rtol, atol, f"case {i} output", rep)
        _compare(ca, cb, rtol, atol, f"case {i} arguments after call", rep)
        rep.n_cases += 1
    return rep
