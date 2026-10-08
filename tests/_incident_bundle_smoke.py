"""Every agent exit is classified, bundled, and a crash loop latches with a
reachable escape (2026-10-07).

LIVE INCIDENT: on 2026-10-05/06 the agent aborted AT EXIT under a deliberate
runlogs/STOP (`terminate called without an active exception`, rc 134) and the
host froze 22 s later, down ~41 h. supervise_skybot.sh logged it as a "clean
stop": all it could see was "pid gone + STOP present". The exit code died
with the process, the run log was rotated by the next launch, and the kernel
log never left the box (docs/foundation/TELEMETRY_INVENTORY_OPS.md 2.1).

This suite runs the REAL supervise_skybot.sh, launch_skybot.sh (with only its
PREFLIGHT region stripped -- tailscale/socat/java/ollama checks cannot pass
off-host) and incident_bundle.sh in a temp copy, against a FAKE agent
(`venv_mc/bin/python` is a bash script that pops its behaviour from a queue).

Contracts:
    A. EXIT CAPTURE: the wrapper writes runlogs/skybot_run.exit with the real
       exit code and signal name (SIGABRT -> 134), and skybot_run.pid / the
       "LAUNCHED pid=" line hold the AGENT's pid, not the wrapper's.
    B. CLASSIFICATION: abort without STOP -> crash/SIGABRT; exit 1 -> crash
       rc 1; rc 0 under STOP -> clean; abort under STOP -> abnormal_under_stop;
       rc 0 under STOP with "terminate called" in the log ->
       abnormal_under_stop (the 2026-10-06 class; a bare rc would call it
       clean).
    C. BUNDLES: one per exit, summary.json per the spec schema, full bundles
       carry the evidence files, `clean` is a short summary, crashes.log gets
       a line for every non-clean exit and none for clean, index.jsonl one
       line per bundle.
    D. STOP PREVENTS RELAUNCH: STOP present -> 0 launches; an exit under STOP
       -> exactly one launch, supervisor exits 0.
    E. CRASH-LOOP GUARD: same class+signal 3x -> runlogs/CRASHLOOP and no 4th
       launch; FALSIFICATION: alternating signals do NOT latch.
    F. ESCAPES (CLAUDE.md 4.1): rm CRASHLOOP -> the idle supervisor resumes;
       a manual launch_skybot.sh clears it and the supervisor adopts that
       agent; a new supervisor start clears it.
    G. RETENTION: incident_bundle.sh keeps <= INCIDENT_KEEP bundles and <=
       INCIDENT_MAX_KB, oldest first, never index.jsonl, never the newest.
    H. LOG ROTATION: supervisor.log.1..5 keep the last 5 sessions even though
       every caller truncates supervisor.log with `>`; the launcher's 3-deep
       run-log rotation and `ulimit -c 0` survive.

Run: PYTHONPATH=. python tests/_incident_bundle_smoke.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

NAME = "incident-bundle"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = ("supervise_skybot.sh", "launch_skybot.sh", "incident_bundle.sh",
           "host_hardening.sh")

FAKE = r"""#!/bin/bash
# fake agent: pop one mode from the queue
Q="$SKYBOT_ROOT/fake_queue"
mode=$(head -1 "$Q" 2>/dev/null); tail -n +2 "$Q" > "$Q.t"; mv "$Q.t" "$Q"
echo "$$" >> "$SKYBOT_ROOT/agent_pids"
echo "fake agent mode=$mode args=$*"
case "$mode" in
  ok)        exit 0 ;;
  fail)      echo "Traceback (most recent call last):"; echo "ValueError: boom $RANDOM$$"; exit 1 ;;
  abort)     echo "terminate called without an active exception"; kill -ABRT $$ ;;
  stopok)    touch "$SKYBOT_ROOT/runlogs/STOP"; exit 0 ;;
  stopabort) touch "$SKYBOT_ROOT/runlogs/STOP"; echo "terminate called without an active exception"; kill -ABRT $$ ;;
  stopmark)  touch "$SKYBOT_ROOT/runlogs/STOP"; echo "terminate called without an active exception"; exit 0 ;;
  slowstop)  sleep 1.5; touch "$SKYBOT_ROOT/runlogs/STOP"; exit 0 ;;
  *)         exit 3 ;;
esac
"""

ENV_FAST = {"SKYBOT_SUP_BACKOFF": "0", "SKYBOT_SUP_POLL": "0.2",
            "SKYBOT_SUP_REFUSE_WAIT": "1", "SKYBOT_SUP_EXIT_WAIT": "5",
            "SKYBOT_SUP_CRASHLOOP_POLL": "0.2"}


def _fail(msg):
    print("[%s] FAIL: %s" % (NAME, msg))
    sys.exit(1)


def check(cond, msg):
    if not cond:
        _fail(msg)


def make_root():
    root = tempfile.mkdtemp(prefix="incident_smoke_")
    os.makedirs(os.path.join(root, "scripts"))
    os.makedirs(os.path.join(root, "runlogs"))
    os.makedirs(os.path.join(root, "configs"))
    os.makedirs(os.path.join(root, "venv_mc", "bin"))
    with open(os.path.join(root, "configs", "minecraft_skybot.yaml"), "w") as f:
        f.write("smoke: true\n")
    for s in ("supervise_skybot.sh", "incident_bundle.sh"):
        shutil.copy(os.path.join(REPO, "scripts", s), os.path.join(root, "scripts", s))
    src = open(os.path.join(REPO, "scripts", "launch_skybot.sh")).read().splitlines(True)
    b = [i for i, l in enumerate(src) if l.startswith("# ---- PREFLIGHT BEGIN")]
    e = [i for i, l in enumerate(src) if l.startswith("# ---- PREFLIGHT END")]
    check(len(b) == 1 and len(e) == 1 and b[0] < e[0], "launch_skybot.sh PREFLIGHT markers")
    with open(os.path.join(root, "scripts", "launch_skybot.sh"), "w") as f:
        f.write("".join(src[:b[0]] + src[e[0] + 1:]))
    fake = os.path.join(root, "venv_mc", "bin", "python")
    with open(fake, "w") as f:
        f.write(FAKE)
    os.chmod(fake, 0o755)
    return root


def env_for(root, **extra):
    env = dict(os.environ)
    env.update(ENV_FAST)
    env["SKYBOT_ROOT"] = root
    env.pop("SKYBOT_SUPERVISED", None)
    env.update(extra)
    return env


def queue(root, *modes):
    with open(os.path.join(root, "fake_queue"), "w") as f:
        f.write("".join(m + "\n" for m in modes))


def start_sup(root, budget="7", **extra):
    log = open(os.path.join(root, "runlogs", "supervisor.log"), "w")  # callers truncate
    return subprocess.Popen(["bash", "scripts/supervise_skybot.sh", budget], cwd=root,
                            env=env_for(root, **extra), stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def run_sup(root, timeout=60, **extra):
    p = start_sup(root, **extra)
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        _fail("supervisor did not exit within %ss; log:\n%s" % (timeout, read(root, "runlogs/supervisor.log")))


def read(root, rel):
    try:
        with open(os.path.join(root, rel)) as f:
            return f.read()
    except OSError:
        return ""


def bundles(root):
    d = os.path.join(root, "runlogs", "incidents")
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if n[:1].isdigit())


def ordered(root):
    """Bundles in WRITE order (index.jsonl): names share a 1 s UTC stamp here."""
    out = []
    for l in read(root, "runlogs/incidents/index.jsonl").splitlines():
        b = json.loads(l).get("bundle") or ""
        if os.path.isdir(os.path.join(root, b)):
            out.append(os.path.basename(b))
    return out


def summary(root, name):
    return json.loads(read(root, "runlogs/incidents/%s/summary.json" % name))


def n_launches(root):
    return len(read(root, "agent_pids").split())


def wait_for(pred, timeout=30.0):
    t = time.time() + timeout
    while time.time() < t:
        if pred():
            return True
        time.sleep(0.1)
    return pred()


def main():
    roots = []
    try:
        # ---- 0: syntax ------------------------------------------------------
        for s in SCRIPTS:
            r = subprocess.run(["bash", "-n", os.path.join(REPO, "scripts", s)],
                               capture_output=True, text=True)
            check(r.returncode == 0, "bash -n %s: %s" % (s, r.stderr))
        launch = read(REPO, "scripts/launch_skybot.sh")
        check("ulimit -c 0" in launch and launch.index("ulimit -c 0") < launch.index("bash -c \"$_WRAP\""),
              "ulimit -c 0 precedes the agent start")
        print("  0. bash -n clean on %d scripts; ulimit -c 0 kept before the wrapper" % len(SCRIPTS))

        # ---- A-D: crash, crash, clean ---------------------------------------
        root = make_root(); roots.append(root)
        queue(root, "abort", "fail", "stopok")
        rc = run_sup(root)
        check(rc == 0, "supervisor exits 0 on a STOP exit, got %s" % rc)
        check(n_launches(root) == 3, "3 launches, got %d" % n_launches(root))
        sup = read(root, "runlogs/supervisor.log")
        pids = read(root, "agent_pids").split()
        launched = [l.split("pid=")[1].strip() for l in sup.splitlines() if l.startswith("LAUNCHED skybot pid=")]
        check(launched == pids, "LAUNCHED pid is the agent pid: %s vs %s" % (launched, pids))
        check(read(root, "runlogs/skybot_run.pid").strip() == pids[-1], "skybot_run.pid holds the agent pid")
        ex = read(root, "runlogs/skybot_run.exit")
        check("exit_code=0" in ex and "signal=none" in ex and "t_end=" in ex, "exit file: %r" % ex)
        print("  A. exit captured (rc/signal/t_end); pid file + LAUNCHED line = AGENT pid")
        bs = ordered(root)
        check(len(bs) == 3 and sorted(bs) == bundles(root), "3 bundles, got %s" % bs)
        kinds = [summary(root, b)["class"] for b in bs]
        check(kinds == ["crash", "crash", "clean"], "classes %s" % kinds)
        s0, s1, s2 = (summary(root, b) for b in bs)
        check(s0["exit_code"] == 134 and s0["signal"] == "SIGABRT", "abort -> 134/SIGABRT: %s" % s0)
        check(s1["exit_code"] == 1 and s1["signal"] is None, "fail -> rc1: %s" % s1)
        check(s2["exit_code"] == 0, "clean rc0: %s" % s2)
        print("  B. abort -> crash/SIGABRT/134, exit 1 -> crash/rc1, rc0+STOP -> clean")
        for k in ("schema", "v", "class", "exit_code", "signal", "t_start", "t_end",
                  "run_id", "git_rev", "config_hash"):
            check(k in s0, "summary key %s" % k)
        check(s0["schema"] == "skybot.incident" and s0["v"] == 1, "schema/v")
        check(len(s0["config_hash"] or "") == 64, "config sha256")
        full = set(os.listdir(os.path.join(root, "runlogs", "incidents", bs[0])))
        for f in ("summary.json", "runlog_tail.txt", "crashes_tail.txt", "mem.txt", "procs.txt",
                  "gpu.txt", "kernel.txt", "oom.txt", "disk.txt", "uptime.txt"):
            check(f in full, "full bundle has %s (%s)" % (f, sorted(full)))
        check("terminate called" in read(root, "runlogs/incidents/%s/runlog_tail.txt" % bs[0]),
              "bundle froze the crashed run's log tail")
        short = set(os.listdir(os.path.join(root, "runlogs", "incidents", bs[2])))
        check("procs.txt" not in short and "summary.json" in short, "clean is a short summary %s" % short)
        cl = read(root, "runlogs/crashes.log")
        check(cl.count("kind=crash") == 2 and "kind=clean" not in cl, "crashes.log lines:\n%s" % cl)
        check("sig=SIGABRT" in cl and "rc=1 " in cl, "crashes.log names rc/signal")
        idx = read(root, "runlogs/incidents/index.jsonl").splitlines()
        check(len(idx) == 3 and all(json.loads(l)["schema"] == "skybot.incident" for l in idx), "index.jsonl")
        check(not [n for n in os.listdir(os.path.join(root, "runlogs", "incidents")) if n.startswith(".partial")],
              "no .partial left behind")
        print("  C. 3 bundles (2 full, 1 short), spec summary.json, crashes.log 2 lines, index 3 lines")
        logs = [n for n in ("minecraft_skybot_run.log", "minecraft_skybot_run.log.1",
                            "minecraft_skybot_run.log.2") if os.path.exists(os.path.join(root, "runlogs", n))]
        check(len(logs) == 3, "3-deep run-log rotation kept: %s" % logs)
        check("mode=abort" in read(root, "runlogs/minecraft_skybot_run.log.2"), "oldest run log rotated to .2")

        # ---- B: abnormal under STOP (abort, and rc0 + marker) -----------------
        for mode in ("stopabort", "stopmark"):
            root = make_root(); roots.append(root)
            queue(root, mode, "ok")
            check(run_sup(root) == 0, "%s: supervisor exit 0" % mode)
            check(n_launches(root) == 1, "%s: no relaunch under STOP" % mode)
            bs = bundles(root)
            check(len(bs) == 1 and summary(root, bs[0])["class"] == "abnormal_under_stop",
                  "%s -> abnormal_under_stop: %s" % (mode, [summary(root, b) for b in bs]))
            check("kind=abnormal_under_stop" in read(root, "runlogs/crashes.log"), "%s in crashes.log" % mode)
        print("  B. abort under STOP and rc0+'terminate called' under STOP -> abnormal_under_stop")

        # ---- D: STOP present at start ----------------------------------------
        root = make_root(); roots.append(root)
        queue(root, "ok")
        open(os.path.join(root, "runlogs", "STOP"), "w").close()
        check(run_sup(root) == 0 and n_launches(root) == 0, "STOP at start -> 0 launches")
        print("  D. STOP present -> 0 launches; exit under STOP -> 1 launch, exit 0")

        # ---- E/F: crash loop, falsification, escapes -------------------------
        root = make_root(); roots.append(root)
        queue(root, "abort", "fail", "abort", "fail", "stopok")
        check(run_sup(root) == 0, "alternating signals: supervisor exit 0")
        check(n_launches(root) == 5 and not os.path.exists(os.path.join(root, "runlogs", "CRASHLOOP")),
              "alternating SIGABRT / rc=1 (2x each) must NOT latch")

        root = make_root(); roots.append(root)
        queue(root, "abort", "abort", "abort", "stopok")
        p = start_sup(root)
        cl = os.path.join(root, "runlogs", "CRASHLOOP")
        check(wait_for(lambda: os.path.exists(cl)), "3x SIGABRT -> CRASHLOOP")
        time.sleep(1.5)
        check(n_launches(root) == 3 and p.poll() is None, "latched: no 4th launch, supervisor idles")
        check("SIGABRT" in read(root, "runlogs/CRASHLOOP") and "Re-open" in read(root, "runlogs/CRASHLOOP"),
              "CRASHLOOP names the key and the escape")
        print("  E. 3x same class+signal -> CRASHLOOP, no 4th launch; alternating does not latch")
        os.remove(cl)
        try:
            rc = p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill(); _fail("rm CRASHLOOP did not resume the supervisor")
        check(rc == 0 and n_launches(root) == 4, "rm CRASHLOOP -> resumed, 4th launch")

        root = make_root(); roots.append(root)
        queue(root, "abort", "abort", "abort", "slowstop")
        p = start_sup(root)
        cl = os.path.join(root, "runlogs", "CRASHLOOP")
        check(wait_for(lambda: os.path.exists(cl)), "latch for manual-launch escape")
        r = subprocess.run(["bash", "scripts/launch_skybot.sh", "9"], cwd=root, env=env_for(root),
                           capture_output=True, text=True, timeout=30)
        check(r.returncode == 0 and "clearing runlogs/CRASHLOOP" in r.stdout, "manual launch clears: %s" % r.stdout)
        try:
            rc = p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill(); _fail("supervisor did not adopt the manual agent")
        sup = read(root, "runlogs/supervisor.log")
        check(rc == 0 and "adopting live agent" in sup and n_launches(root) == 4,
              "adopted manual agent, no duplicate launch:\n%s" % sup)
        check(summary(root, ordered(root)[-1])["class"] == "clean", "adopted agent's exit classified")

        root = make_root(); roots.append(root)
        queue(root, "stopok")
        with open(os.path.join(root, "runlogs", "CRASHLOOP"), "w") as f:
            f.write("CRASHLOOP old\n")
        check(run_sup(root) == 0 and n_launches(root) == 1, "new supervisor start clears CRASHLOOP")
        print("  F. escapes: rm CRASHLOOP resumes; manual launch clears + is adopted; supervisor start clears")

        # ---- G: retention ----------------------------------------------------
        root = make_root(); roots.append(root)
        inc = os.path.join(root, "runlogs", "incidents")
        for i in range(5):
            os.makedirs(os.path.join(inc, "2020010%dT000000Z-crash" % (i + 1)))
        with open(os.path.join(inc, "20200101T000000Z-crash", "big.bin"), "wb") as f:
            f.write(b"\0" * (3 * 1024 * 1024))
        with open(os.path.join(inc, "index.jsonl"), "w") as f:
            f.write('{"old": 1}\n')
        b_env = env_for(root, INCIDENT_KEEP="4", INCIDENT_MAX_KB="100000")
        subprocess.run(["bash", "scripts/incident_bundle.sh", "crash", "0"], cwd=root, env=b_env,
                       capture_output=True, timeout=90)
        bs = bundles(root)
        check(len(bs) == 4 and "20200101T000000Z-crash" not in bs and "20200102T000000Z-crash" not in bs,
              "keep newest 4: %s" % bs)
        subprocess.run(["bash", "scripts/incident_bundle.sh", "clean", "0"], cwd=root,
                       env=env_for(root, INCIDENT_KEEP="50", INCIDENT_MAX_KB="1"),
                       capture_output=True, timeout=90)
        bs = bundles(root)
        check(len(bs) == 1 and bs[0].endswith("-clean"), "size cap prunes all but the newest: %s" % bs)
        idx = read(root, "runlogs/incidents/index.jsonl").splitlines()
        check(len(idx) == 3 and idx[0] == '{"old": 1}', "index.jsonl never pruned")
        print("  G. retention: count cap and size cap prune oldest first; newest + index.jsonl kept")

        # ---- H: supervisor log rotation --------------------------------------
        root = make_root(); roots.append(root)
        open(os.path.join(root, "runlogs", "STOP"), "w").close()
        for k in range(1, 8):
            p = start_sup(root, budget=str(k))
            p.wait(timeout=20)
        rl = os.path.join(root, "runlogs")
        check(os.path.exists(os.path.join(rl, "supervisor.log.5"))
              and not os.path.exists(os.path.join(rl, "supervisor.log.6")), "keep 5")
        check("budget=7" in read(root, "runlogs/supervisor.log.1")
              and "budget=6" in read(root, "runlogs/supervisor.log.2")
              and "budget=3" in read(root, "runlogs/supervisor.log.5"), "sessions shift .1->.5")
        check("budget=7" in read(root, "runlogs/supervisor.log")
              and "budget=6" not in read(root, "runlogs/supervisor.log"), "live log still truncated by caller")
        print("  H. supervisor.log.1..5 = last 5 sessions despite the caller's '>'")
    finally:
        for r in roots:
            shutil.rmtree(r, ignore_errors=True)
    print("[%s] ALL PASS" % NAME)


if __name__ == "__main__":
    main()
