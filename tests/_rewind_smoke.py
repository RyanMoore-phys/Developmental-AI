"""Save-state ring + stuck watch: rewind the GAME, never latch.

Born 2026-10-11 (docs/UNIVERSAL_GAMES.md §4). User decision: snapshot every
5 minutes, keep 3, and rewind automatically when reward stagnation shows
SkyBot is stuck. That is a GUARD, and this repo's most repeated bug class is
the guard that becomes a latch (CLAUDE.md 4.1, >= 9 occurrences). So every
refusal state below is tested for its exit, not just its entry. Fake clock
and fake save/load: no emulator, runs in milliseconds.

Contracts:
    A. RING: saves at once, then only every interval_s; keeps the newest
       `keep`, deleting the oldest FILE; load_oldest restores the oldest.
    B. A REWIND DROPS THE SLUMP'S TIMELINE. After load_oldest succeeds, the
       newer snapshots (taken during the slump) are deleted — kept, they
       would age into "oldest" and a later rewind would land back inside the
       stuck period.
    C. A FAILED SAVE/LOAD NEVER RAISES: save failure is counted, load
       failure returns False (the adapter then starts a fresh episode).
    D. WATCH: no verdict during warmup; a reward that collapses to <= ratio x
       baseline for a full window -> REWIND; a healthy stream never fires.
    E. NO LATCH 1 — cooldown: right after a verdict the watch is silent for
       one window, then re-arms by itself.
    F. NO LATCH 2 — rewinds that do not help: after max_rewinds_without_
       recovery REWINDs, the next verdict is FRESH (a new episode), so a
       rewind loop cannot become a trap; the count then restarts.
    G. RECOVERY resets the count: a full window back above the line.
    H. "Nothing ever paid" is not stuck: a zero/negative baseline never
       fires (absence of change is not evidence of a trap).
    I. THE BOUNDARY THE LIVE BODY HONOURS. The lifelong body stores
       `done = terminated or restart` and ignores a plain truncation, so a
       rewind step that set only truncated=True would never reset the stream
       and the adapter would answer "rewind pending" forever — found while
       wiring this, before it ever ran. Pinned on the source: the Doom
       adapter's rewind step sets env_restarted=True, and the lifelong body
       still resets on env_restarted.

Run: PYTHONPATH=. python tests/_rewind_smoke.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.environments.rewind import (               # noqa: E402
    FRESH, NONE, REWIND, SnapshotRing, StuckWatch)

NAME = "rewind"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def main():
    print(f"[{NAME}]")
    clk = Clock()
    state = {"v": None}
    loaded = []

    def save(p):
        with open(p, "w") as f:
            f.write(str(state["v"]))

    def load(p):
        loaded.append(open(p).read())

    with tempfile.TemporaryDirectory() as d:
        # ---- A
        r = SnapshotRing(d, save, load, interval_s=300, keep=3, ext="zds", clock=clk)
        state["v"] = 0
        check(r.maybe_save() and len(r) == 1, "A. first call saves immediately")
        clk.t += 100
        check(not r.maybe_save(), "A. no save before the interval")
        for v in (1, 2, 3, 4):
            clk.t += 300
            state["v"] = v
            r.maybe_save()
        files = sorted(f for f in os.listdir(d) if f.endswith(".zds"))
        check(len(r) == 3 and len(files) == 3, "A. ring keeps 3 files: %s" % files)
        check(open(r.oldest()).read() == "2", "A. oldest kept = 3rd save (0,1 deleted)")
        # ---- B
        check(r.load_oldest() and loaded[-1] == "2", "B. load_oldest restores the oldest")
        files = [f for f in os.listdir(d) if f.endswith(".zds")]
        check(len(r) == 1 and len(files) == 1,
              "B. newer (slump-era) snapshots deleted after the rewind: %s" % files)
        # ---- C
        def bad_save(p):
            raise OSError("disk full")

        def bad_load(p):
            raise RuntimeError("corrupt")
        r2 = SnapshotRing(os.path.join(d, "x"), bad_save, bad_load, 300, 3, clock=clk)
        check(r2.maybe_save() is False and r2.save_failures == 1,
              "C. a failed save is counted, never raised")
        r3 = SnapshotRing(os.path.join(d, "y"), save, bad_load, 300, 3, clock=clk)
        r3.maybe_save()
        check(r3.load_oldest() is False, "C. a failed load returns False (-> fresh episode)")
        check(SnapshotRing(os.path.join(d, "z"), save, load, 300, 3,
                           clock=clk).load_oldest() is False,
              "C. no snapshot -> False (-> fresh episode)")

    # ---- D..H: the watch
    def feed(w, r, seconds, dt=1.0):
        out = []
        for _ in range(int(seconds / dt)):
            clk.t += dt
            v = w.observe(r)
            if v != NONE:
                out.append(v)
        return out

    clk.t = 0.0
    w = StuckWatch(window_s=60, ratio=0.25, warmup_s=120, baseline_halflife_s=600,
                   max_rewinds_without_recovery=2, clock=clk)
    check(feed(w, 0.0, 110) == [], "D. no verdict during warmup (even at 0 reward)")
    w = StuckWatch(window_s=60, ratio=0.25, warmup_s=120, baseline_halflife_s=600,
                   max_rewinds_without_recovery=2, clock=clk)
    check(feed(w, 1.0, 600) == [], "D. a healthy stream never fires")
    v = feed(w, 0.0, 130)
    check(v[:1] == [REWIND], "D. collapse to 0 for a full window -> REWIND: %s" % v)
    # ---- E: cooldown then re-arm
    w2 = StuckWatch(window_s=60, ratio=0.25, warmup_s=0, baseline_halflife_s=600,
                    max_rewinds_without_recovery=5, clock=clk)
    feed(w2, 1.0, 300)
    # Detection latency is by design ~1.75 windows: the window MEAN must
    # first fall to the line (~45 s of zeros in a 60 s window), then stay
    # there a full window. So step until the first verdict, then time the
    # cooldown from that moment.
    first, waited = [], 0
    while not first and waited < 400:
        first = feed(w2, 0.0, 1)
        waited += 1
    quiet = feed(w2, 0.0, 59)
    check(first == [REWIND] and 100 <= waited <= 130,
          "E. first verdict after ~1.75 windows of collapse (%d s)" % waited)
    check(quiet == [], "E. silent for one window after a verdict")
    again = feed(w2, 0.0, 200)
    check(REWIND in again, "E. re-arms by itself after the cooldown (no latch)")
    # ---- F: rewinds that do not help -> FRESH
    w3 = StuckWatch(window_s=60, ratio=0.25, warmup_s=0, baseline_halflife_s=3600,
                    max_rewinds_without_recovery=2, clock=clk)
    feed(w3, 1.0, 300)
    seq = feed(w3, 0.0, 1000)
    check(seq[:3] == [REWIND, REWIND, FRESH],
          "F. 2 unhelpful rewinds -> FRESH episode next: %s" % seq[:4])
    check(seq[3:4] == [REWIND], "F. count restarts after the fresh start")
    # ---- G: recovery resets the count. max=1, so WITHOUT a recovery the
    # verdict after one rewind must be FRESH (the control); WITH a full
    # window back above the line it must be REWIND again.
    def first_verdict(w, r, limit=400):
        for _ in range(limit):
            v = feed(w, r, 1)
            if v:
                return v[0]
        return None

    def g_run(recover):
        w = StuckWatch(window_s=60, ratio=0.25, warmup_s=0,
                       baseline_halflife_s=3600,
                       max_rewinds_without_recovery=1, clock=clk)
        feed(w, 1.0, 300)
        v1 = first_verdict(w, 0.0)
        feed(w, 0.0, 61)                        # cooldown passes
        if recover:
            feed(w, 1.0, 200)                   # > one window above the line
        return v1, first_verdict(w, 0.0)

    ctl = g_run(False)
    rec = g_run(True)
    check(ctl == (REWIND, FRESH), "G. control: no recovery -> REWIND then FRESH: %s" % (ctl,))
    check(rec == (REWIND, REWIND), "G. recovery resets the count -> REWIND again: %s" % (rec,))
    # ---- H
    w5 = StuckWatch(window_s=60, ratio=0.25, warmup_s=0, clock=clk)
    check(feed(w5, 0.0, 600) == [], "H. zero baseline (nothing ever paid) never fires")

    # ---- I: the boundary the live body honours
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "developmental_ai/environments/vizdoom_env.py")).read()
    i = src.index("if self._pending != NONE:")
    blk = src[i:src.index("return self._last_obs, 0.0, False, True, info", i)]
    check('"env_restarted": True' in blk,
          "I. the rewind step sets env_restarted=True (not truncated alone)")
    loop = open(os.path.join(root, "developmental_ai/core/developmental_loop.py")).read()
    check('_crashed = bool(step_infos[e_i]) and step_infos[e_i].get(\n'
          '                    "env_restarted")' in loop
          and "if dones[e_i] or _crashed:" in loop,
          "I. the lifelong body still resets a stream on env_restarted")
    check("dones.append(bool(term) or _rst)" in loop,
          "I. witness: lifelong done ignores a plain truncation (why I matters)")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
