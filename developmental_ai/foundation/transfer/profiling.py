"""Profile before porting (plan Stage 12 item 4; NATIVE_RUNTIME.md).

profile_callable(fn, iterations, top) runs `fn` under cProfile and returns
    hotspots    the top functions by OWN time (tottime), with call counts
    crossings   how many calls crossed into C/builtins ("~" in pstats) versus
                stayed in Python, in total and PER ITERATION, and the builtin
                call sites with the most calls. A crossing count that grows
                with the data size is the signature of per-element work —
                exactly what a native port must NOT reproduce (item 6:
                "avoid per-element Python/Rust crossings").
    sections    wall time per named `Sections` block (time.perf_counter),
                when fn accepts a `sections` keyword

cProfile inflates the cost of many small calls; read hotspots for ranking,
and take absolute step rates from an unprofiled run (sections alone).
"""

from __future__ import annotations

import cProfile
import inspect
import pstats
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict


class Sections:
    """Named wall-clock accumulators: `with sections("env_step"): ...`."""

    def __init__(self):
        self.seconds: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - t0
            self.counts[name] += 1


def _label(key) -> str:
    fname, line, func = key
    if fname == "~":
        return func                       # "<built-in method ...>" / "{method ...}"
    return f"{fname.rsplit('/', 1)[-1]}:{line}({func})"


def profile_callable(fn: Callable, iterations: int = 1, top: int = 15,
                     **kwargs) -> Dict[str, Any]:
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be an int >= 1")
    sections = Sections()
    try:
        takes_sections = "sections" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        takes_sections = False
    if takes_sections:
        kwargs = dict(kwargs, sections=sections)
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    try:
        for _ in range(iterations):
            fn(**kwargs)
    finally:
        prof.disable()
    wall = time.perf_counter() - t0
    st = pstats.Stats(prof)
    rows, py_calls, c_calls = [], 0, 0
    c_sites = []
    for key, (cc, nc, tt, ct, _callers) in st.stats.items():
        if key[0] == "~":
            c_calls += nc
            c_sites.append((nc, _label(key)))
        else:
            py_calls += nc
        rows.append({"function": _label(key), "ncalls": nc, "primitive_calls": cc,
                     "tottime": tt, "cumtime": ct, "builtin": key[0] == "~"})
    rows.sort(key=lambda r: r["tottime"], reverse=True)
    c_sites.sort(reverse=True)
    profiled = sum(r["tottime"] for r in rows) or 1e-12
    for r in rows:
        r["share"] = r["tottime"] / profiled
    return {
        "iterations": iterations, "wall_seconds": wall,
        "hotspots": rows[:top],
        "crossings": {"python_calls": py_calls, "builtin_calls": c_calls,
                      "builtin_calls_per_iteration": c_calls / iterations,
                      "python_calls_per_iteration": py_calls / iterations,
                      "top_builtin_sites": [{"function": f, "ncalls": n}
                                            for n, f in c_sites[:top]]},
        "sections": {k: {"seconds": v, "count": sections.counts[k],
                         "share_of_wall": v / wall if wall > 0 else 0.0}
                     for k, v in sorted(sections.seconds.items(),
                                        key=lambda kv: -kv[1])},
    }


def format_profile(rep: Dict[str, Any]) -> str:
    L = [f"profiled {rep['iterations']} iteration(s), {rep['wall_seconds']:.3f}s wall",
         "", "TOP CPU HOTSPOTS (own time)",
         f"  {'share':>6} {'tottime':>9} {'cumtime':>9} {'ncalls':>9}  function"]
    for r in rep["hotspots"]:
        L.append(f"  {100 * r['share']:5.1f}% {r['tottime']:9.4f} {r['cumtime']:9.4f} "
                 f"{r['ncalls']:9d}  {r['function']}")
    c = rep["crossings"]
    L += ["", "PYTHON <-> C CROSSINGS",
          f"  python calls {c['python_calls']} ({c['python_calls_per_iteration']:.1f}/iter)",
          f"  builtin/C calls {c['builtin_calls']} "
          f"({c['builtin_calls_per_iteration']:.1f}/iter)"]
    for s in c["top_builtin_sites"][:8]:
        L.append(f"    {s['ncalls']:9d}  {s['function']}")
    if rep["sections"]:
        L += ["", "SECTIONS (perf_counter)"]
        for k, v in rep["sections"].items():
            L.append(f"  {k:24s} {v['seconds']:9.4f}s  {100 * v['share_of_wall']:5.1f}%"
                     f"  x{v['count']}")
    return "\n".join(L)
