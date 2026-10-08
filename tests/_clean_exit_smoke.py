"""The agent must shut down cleanly on EVERY exit (2026-10-07).

LIVE INCIDENT: DevelopmentalAI.close() stops and joins the async WM trainer
before env/CUDA teardown -- its own docstring says a daemon thread killed at
interpreter exit mid optimizer step crashes -- but NOTHING called it. Both
2026-10-05 runs, including a clean runlogs/STOP, ended in `terminate called
without an active exception` (C++ abort -> SIGABRT -> a core dump of a
~8-10 GB CUDA process on a 16 GB host). The host froze 22 s after the second
and stayed down ~41 h.

Contracts (source-level; no Minecraft needed):
    A. run_minecraft.main() calls _close_bounded(agent) in a `finally`, so a
       clean stop, a reached budget and an exception all close the agent.
    B. _close_bounded is time-bounded (a hung client cannot block exit) and
       never re-raises over the run's own outcome.
    C. close() stops the WM trainer BEFORE it closes envs.
    D. launch_skybot.sh disables core dumps before it starts python.

Run: PYTHONPATH=. python tests/_clean_exit_smoke.py
"""
import ast
import os
import sys

NAME = "clean-exit"


def _func(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"{name}() not found")


def main():
    src = open("run_minecraft.py").read()
    tree = ast.parse(src)
    m = _func(tree, "main")
    finals = [t for t in ast.walk(m) if isinstance(t, ast.Try) and t.finalbody]
    called = any(isinstance(c, ast.Call) and getattr(c.func, "id", "") ==
                 "_close_bounded" for t in finals for s in t.finalbody
                 for c in ast.walk(s))
    assert called, "main() must call _close_bounded(agent) in a finally"
    runs = [t for t in finals if any(
        isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "run"
        for s in t.body for c in ast.walk(s))]
    assert runs, "agent.run(...) must be inside the try whose finally closes"
    print("  A. main(): agent.run inside try, _close_bounded(agent) in finally")

    fcb = _func(tree, "_close_bounded")
    cb = ast.get_source_segment(src, fcb)
    assert ".join(timeout_s)" in cb and "daemon=True" in cb, cb[:200]
    assert not any(isinstance(n, ast.Raise) for n in ast.walk(fcb)), \
        "_close_bounded must not re-raise over the run's own outcome"
    print("  B. _close_bounded: worker thread, bounded join, never re-raises")

    loop = open(os.path.join("developmental_ai", "core",
                             "developmental_loop.py")).read()
    body = loop[loop.index("    def close(self) -> None:"):]
    body = body[:body.index("\n    def ", 10)]
    i_stop = body.index("self._wm_trainer_stop.set()")
    i_join = body.index("self._wm_trainer_thread.join(")
    i_env = body.index("self.env.close()")
    assert i_stop < i_join < i_env, "WM trainer must stop before env close"
    print("  C. close(): WM trainer stopped + joined before env.close()")

    sh = open(os.path.join("scripts", "launch_skybot.sh")).read()
    launch = "./venv_mc/bin/python run_minecraft.py"
    assert "\nulimit -c 0" in sh and launch in sh and \
        sh.index("\nulimit -c 0") < sh.index(launch), \
        "core dumps must be off before python starts"
    print("  D. launch_skybot.sh: ulimit -c 0 before the agent starts")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}\n[{NAME}] FAILED")
        sys.exit(1)
