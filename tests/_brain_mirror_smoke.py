"""Brain-mirror smoke (2026-10-03) -- scripts/brain_mirror/brain_mirror.py.

WHAT IS CLAIMED
    node2 pulls SkyBot's brain off the training host every ~5 minutes into
    hardlinked, verified, rotated snapshots, and the training host is only
    ever READ. The brain is the one irreplaceable thing in this project
    (CLAUDE.md sec 6 "Backups": a host once died with ~11 days of unbacked
    state, and nothing automated the copy). A backup that silently copies
    garbage, fills node2's disk, or stops advancing is worse than none,
    because it is trusted.

WHY THE CONSISTENCY MACHINERY EXISTS (measured in the code, not assumed)
    developmental_loop._save_checkpoint_locked writes every checkpoint file
    with torch.save / open("w") STRAIGHT ONTO THE FINAL NAME, one after
    another (logs/checkpoints/, loop.log_dir: ./logs). A copy taken during a
    save can catch a truncated zip or two generations side by side. Skill
    policy .pt files are written the same way. Only the registry, break
    memory, consequence/anticipation state and LTM index use tmp+replace.

The source is a LOCAL directory standing in for user@host:/workspace/devai
(BRAIN_MIRROR_TEST_SOURCE). The ssh path stays the default and its command
line is asserted directly.

Contracts:
    A. The production command line: ssh BatchMode, --rsync-path runs rsync at
       nice 19 + ionice idle, --whole-file, NO compression, NO --delete, source
       is user@host:MAIN_ROOT. Placeholders (and the shipped example) are
       REFUSED with an error naming the env file. No created file contains an
       IPv4 address (the repo is public) and the real env file is gitignored.
    B. First snapshot: exactly the brain set (no replay buffer, no logs, no
       *.tmp), skill bank copied recursively, MANIFEST sha256 == the bytes.
    C. An unchanged brain: every file of the second snapshot is a HARDLINK of
       the first (same inode); it adds zero file bytes, and nothing is re-hashed.
    D. A save landing mid-copy triggers a re-pull and the snapshot holds the
       NEW bytes; a brain that never stops changing is kept as mixed: true
       after exactly MAX_REPULLS re-pulls.
    E. A truncated .pt (or .pkl sidecar: framing check) fails verification: .failed, `latest` unchanged,
       nothing pruned. LATCH ESCAPE (CLAUDE.md 4.1): the same bytes failing
       DEGRADED_AFTER runs in a row are promoted as degraded:true (exit 15)
       instead of freezing the mirror forever. Real torch.save output passes
       the zip check and a truncated one fails it.
    F. Retention over a synthetic 20-day timeline (injected clock): last N +
       newest-per-hour (48h) + newest-per-day (14d) + protected; and the
       live prune leaves exactly that set.
    G. Disk guard: below MIN_FREE_GB, the pull is skipped (exit 13), `latest`
       survives, no new snapshot.
    H. Lock: a concurrent run is refused (exit 10); a lock file left by a dead
       process is recovered, not a latch.
    I. --restore-plan prints the procedure (STOP, no --delete, layout
       warning) and EXECUTES NOTHING (rsync/ssh shims on PATH never fire).
    J. --status: OK when fresh, exit 16 when stale.
    K. The source tree is byte-, mtime- and mode-identical after every run.
    L. INCLUDE_SHADOW keeps one rolling copy outside the snapshots.
    M. An unresolvable host over the REAL ssh path -> exit 11, no snapshot.

SKIPS (exit 0) only if rsync is not installed.

Run: PYTHONPATH=. python tests/_brain_mirror_smoke.py
"""
import ast
import fcntl
import hashlib
import importlib.util
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

NAME = "brain_mirror_smoke"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BM_DIR = os.path.join(REPO, "scripts", "brain_mirror")
SCRIPT = os.path.join(BM_DIR, "brain_mirror.py")

if shutil.which("rsync") is None:
    print("[%s] SKIP: rsync not installed" % NAME)
    sys.exit(0)

for _k in list(os.environ):
    if _k.startswith("BRAIN_MIRROR_"):
        del os.environ[_k]

_spec = importlib.util.spec_from_file_location("brain_mirror", SCRIPT)
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

_n = [0]


def check(cond, msg):
    _n[0] += 1
    if not cond:
        print("  %2d. FAIL  %s" % (_n[0], msg))
        print("[%s] FAILED" % NAME)
        sys.exit(1)
    print("  %2d. ok    %s" % (_n[0], msg))


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def make_pt(path, payload):
    """torch >= 1.6 checkpoints are zip archives (stored members)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("archive/data.pkl", payload)
        z.writestr("archive/data/0", os.urandom(4096))


def put(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


BRAIN_SET = {
    "logs/checkpoints/world_model.pt", "logs/checkpoints/policy.pt",
    "logs/checkpoints/magnet.pt", "logs/checkpoints/knowledge_graph.json",
    "logs/checkpoints/options_state.json", "runlogs/breaks_by_type.json",
    "runlogs/consequence_state.json",
    # 2026-10-05 sidecar. Its sibling curiosity_visits.pkl is deliberately
    # ABSENT here (an older checkpoint): B asserts that is skipped, not fatal.
    "logs/checkpoints/progress_probes.pkl",
    "skill_bank_mc_rssm/registry.json",
    "skill_bank_mc_rssm/skills/s1/policy.pt",
    "skill_bank_mc_rssm/skills/s1/deep/nested/notes.json",
    "skill_bank_mc_curiosity/registry.json",
}


def make_source(root):
    for rel in BRAIN_SET:
        if rel.endswith(".pt"):
            make_pt(os.path.join(root, rel), b"weights:" + rel.encode())
        elif rel.endswith(".pkl"):
            p = os.path.join(root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                pickle.dump({"rel": rel, "n": 1}, f, protocol=4)
        else:
            put(root, rel, json.dumps({"rel": rel, "n": 1}))
    # NOT brain: must never be copied
    put(root, "logs/checkpoints/replay_buffer/chunk0.npz", "x" * 5000)
    put(root, "runlogs/metrics.jsonl", '{"m": 1}\n')
    put(root, "runlogs/heartbeat.jsonl", '{"h": 1}\n')
    put(root, "skill_bank_mc_rssm/registry.json.tmp", "{half-writ")
    put(root, "logs/checkpoints/world_model.pt.tmp", "partial")
    # the sidecar writer's in-flight temp and its corrupt-aside rename
    put(root, "logs/checkpoints/progress_probes.pkl.ab12cd.tmp", "partial")
    put(root, "logs/checkpoints/curiosity_visits.pkl.corrupt-20261005T000000",
        "bad")
    put(root, "configs/minecraft_skybot.yaml", "loop: {}\n")
    put(root, "runlogs/foundation_shadow/shadow.jsonl", '{"s": 1}\n')


def tree_digest(root):
    out = []
    for d, dirs, fns in os.walk(root):
        dirs.sort()
        for n in sorted(dirs) + sorted(fns):
            p = os.path.join(d, n)
            st = os.lstat(p)
            h = ""
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    h = hashlib.sha256(f.read()).hexdigest()
            out.append((os.path.relpath(p, root), h, st.st_size,
                        st.st_mtime_ns, st.st_mode))
    return out


def snap_files(snap):
    out = set()
    for d, _dirs, fns in os.walk(snap):
        for fn in fns:
            rel = os.path.relpath(os.path.join(d, fn), snap)
            if rel != "MANIFEST.json":
                out.add(rel.replace(os.sep, "/"))
    return out


def write_env(path, dest, extra=None, host="brain-mirror-test.invalid"):
    vals = {"MAIN_HOST": host, "MAIN_USER": "skybot", "MAIN_SSH_PORT": "22",
            "MAIN_ROOT": "/workspace/devai", "BRAIN_DEST": dest,
            "KEEP_LAST": "12", "KEEP_HOURLY_HOURS": "48",
            "KEEP_DAILY_DAYS": "14", "MIN_FREE_GB": "20",
            "INCLUDE_SHADOW": "0", "SETTLE_S": "0", "MAX_REPULLS": "3",
            "DEGRADED_AFTER": "3"}
    vals.update(extra or {})
    with open(path, "w") as f:
        f.write("# test env\n" + "".join("%s=%s\n" % kv
                                          for kv in vals.items()))
    return path


T0 = 1790000000.0      # fixed epoch for the injected clock


def run(env_file, *args, src=None, now=None, hook=None, free=None,
        path_prefix=None):
    env = dict(os.environ)
    if src:
        env["BRAIN_MIRROR_TEST_SOURCE"] = src
    if now is not None:
        env["BRAIN_MIRROR_TEST_NOW"] = str(now)
    if hook:
        env["BRAIN_MIRROR_TEST_HOOK"] = hook
    if free is not None:
        env["BRAIN_MIRROR_TEST_FREE_GB"] = str(free)
    if path_prefix:
        env["PATH"] = path_prefix + os.pathsep + env.get("PATH", "")
    p = subprocess.run([sys.executable, SCRIPT, "--env", env_file]
                       + list(args), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env=env, timeout=120)
    return (p.returncode, p.stdout.decode("utf-8", "replace"),
            p.stderr.decode("utf-8", "replace"))


def latest(dest):
    link = os.path.join(dest, "latest")
    return os.path.basename(os.readlink(link)) if os.path.islink(link) \
        else None


def manifest(dest, name):
    with open(os.path.join(dest, "snapshots", name, "MANIFEST.json")) as f:
        return json.load(f)


def snaps(dest):
    d = os.path.join(dest, "snapshots")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


HOOK_PY = r'''
import json, os, sys, time
target, counter, mode = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(open(counter).read()) + 1 if os.path.exists(counter) else 1
open(counter, "w").write(str(n))
if mode == "once" and n > 1:
    sys.exit(0)
with open(target, "w") as f:
    json.dump({"generation": n, "mode": mode, "pad": "x" * n}, f)
t = time.time() + 1000 * n + (500 if mode == "always" else 0)
os.utime(target, (t, t))
'''


def main():
    tmp = tempfile.mkdtemp(prefix="brain_mirror_smoke_")
    try:
        _main(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    test_newest_clean_protected()
    print("[%s] ALL PASS" % NAME)


def _main(tmp):
    # ---------------------------------------------------------------- A
    print("A. production command line, placeholders, no addresses")
    with open(SCRIPT) as f:
        src_text = f.read()
    ast.parse(src_text, feature_version=(3, 8))
    check(True, "brain_mirror.py parses as Python 3.8 syntax")
    imports = set(re.findall(r"^(?:from|import) (\w+)", src_text, re.M))
    stdlib_ok = {"__future__", "argparse", "errno", "fcntl", "hashlib",
                 "json", "logging", "os", "re", "shlex", "shutil",
                 "subprocess", "sys", "time", "zipfile", "datetime"}
    check(imports <= stdlib_ok, "stdlib-only imports: %s"
          % sorted(imports - stdlib_ok))

    dest_a = os.path.join(tmp, "destA")
    env_ssh = write_env(os.path.join(tmp, "ssh.env"), dest_a,
                        {"MAIN_SSH_KEYFILE": "~/.ssh/k_ed25519"})
    cfg = bm.load_config(env_ssh)
    check(cfg["test_source"] is None, "ssh is the default transport")
    cmd = bm.build_pull_cmd(cfg, "/x/snap.partial", "/x/files",
                            link_dest="/x/prev")
    joined = " ".join(cmd)
    check("--rsync-path=nice -n 19 ionice -c3 rsync" in cmd,
          "rsync runs on main at nice 19 + ionice idle")
    check("--whole-file" in cmd, "--whole-file (no delta CPU on main)")
    short = [a for a in cmd if a.startswith("-") and not a.startswith("--")]
    check(not any("z" in a for a in short) and "--compress" not in joined,
          "no compression (-z) on the LAN")
    check(not any(a.startswith("--delete") or a == "--del"
                  or a == "--remove-source-files" for a in cmd),
          "no --delete / --remove-source-files anywhere")
    e_arg = cmd[cmd.index("-e") + 1]
    check("BatchMode=yes" in e_arg and "-p 22" in e_arg
          and "k_ed25519" in e_arg, "ssh is BatchMode with port + key")
    check(cmd[-2] == "skybot@brain-mirror-test.invalid:/workspace/devai/"
          and cmd[-1] == "/x/snap.partial/", "pull FROM user@host:MAIN_ROOT")
    check("--link-dest=/x/prev" in cmd, "--link-dest to previous snapshot")
    dry = bm.build_pull_cmd(cfg, "/x/snap.partial", "/x/files",
                            dry_run=True)
    check("-n" in dry and "--stats" not in dry, "consistency pass is a dry run")
    pr = bm.build_probe_cmd(cfg, cfg["brain_paths"])
    check(pr[0] == "ssh" and "BatchMode=yes" in pr, "probe goes over ssh")

    rc, out, err = run(os.path.join(BM_DIR, "brain_mirror.env.example"),
                       "--check-config")
    check(rc == bm.EXIT_CONFIG and "brain_mirror.env.example" in err
          and "placeholder" in err,
          "the shipped example is REFUSED, naming the file (rc=%d)" % rc)
    ph = write_env(os.path.join(tmp, "ph.env"), dest_a,
                   host="<training-host-LAN-IP>")
    rc, out, err = run(ph)
    check(rc == bm.EXIT_CONFIG and "MAIN_HOST" in err and "ph.env" in err,
          "placeholder MAIN_HOST refused with the env file named")
    ipv4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
    hits = []
    for fn in sorted(os.listdir(BM_DIR)) + ["../../tests/_brain_mirror_smoke.py"]:
        p = os.path.normpath(os.path.join(BM_DIR, fn))
        if os.path.isfile(p) and fn != "brain_mirror.env":
            with open(p) as f:
                hits += ["%s: %s" % (fn, m) for m in ipv4.findall(f.read())]
    check(not hits, "no IPv4 address in any tracked brain_mirror file %s"
          % hits)
    if shutil.which("git"):
        g1 = subprocess.run(["git", "-C", REPO, "check-ignore", "-q",
                             "scripts/brain_mirror/brain_mirror.env"])
        g2 = subprocess.run(["git", "-C", REPO, "check-ignore", "-q",
                             "scripts/brain_mirror/brain_mirror.env.example"])
        check(g1.returncode == 0 and g2.returncode == 1,
              "brain_mirror.env gitignored, .example tracked")

    # ---------------------------------------------------------------- B
    print("B. first snapshot")
    src = os.path.join(tmp, "srcA")
    make_source(src)
    before = tree_digest(src)
    env_a = write_env(os.path.join(tmp, "a.env"), dest_a,
                      {"INCLUDE_SHADOW": "1"})
    rc, out, err = run(env_a, src=src, now=T0)
    check(rc == 0 and out.startswith("brain_mirror OK"),
          "first run exit 0: %s %s" % (out.strip(), err.strip()[-300:]))
    s1 = latest(dest_a)
    check(s1 is not None and s1 in snaps(dest_a), "latest -> %s" % s1)
    snap1 = os.path.join(dest_a, "snapshots", s1)
    got = snap_files(snap1)
    check(got == BRAIN_SET, "exactly the brain set (extra=%s missing=%s)"
          % (sorted(got - BRAIN_SET), sorted(BRAIN_SET - got)))
    check(os.path.isfile(os.path.join(
        snap1, "skill_bank_mc_rssm/skills/s1/deep/nested/notes.json")),
        "skill bank copied recursively (3 levels deep)")
    m1 = manifest(dest_a, s1)
    ok = all(bm.sha256_file(os.path.join(snap1, r)) == e["sha256"]
             and os.path.getsize(os.path.join(snap1, r)) == e["size"]
             for r, e in m1["files"].items())
    check(ok and set(m1["files"]) == BRAIN_SET,
          "MANIFEST sha256+size match every file")
    check(m1["mixed"] is False and m1["verified"] and m1["pulls"] == 1,
          "clean snapshot: mixed false, verified, 1 pull")
    check("logs/checkpoints/curiosity.pt" in m1["brain_paths_missing"],
          "absent brain paths are recorded, not fatal")
    check("logs/checkpoints/curiosity_visits.pkl" in m1["brain_paths_missing"]
          and "logs/checkpoints/curiosity_visits.pkl" in bm.DEFAULT_BRAIN_PATHS
          and "logs/checkpoints/progress_probes.pkl" in bm.DEFAULT_BRAIN_PATHS,
          "both 2026-10-05 sidecars are listed; a missing one is skipped")
    # ---------------------------------------------------------------- L
    print("L. shadow mirror")
    check(os.path.isfile(os.path.join(dest_a, "shadow_mirror",
                                      "shadow.jsonl")),
          "INCLUDE_SHADOW: rolling copy in BRAIN_DEST/shadow_mirror/")

    # ---------------------------------------------------------------- C
    print("C. unchanged brain -> hardlinks")
    use1 = bm.disk_use_bytes(os.path.join(dest_a, "snapshots"))
    rc, out, err = run(env_a, src=src, now=T0 + 300)
    s2 = latest(dest_a)
    check(rc == 0 and s2 != s1, "second run -> new snapshot %s" % s2)
    snap2 = os.path.join(dest_a, "snapshots", s2)
    same = all(os.stat(os.path.join(snap1, r)).st_ino
               == os.stat(os.path.join(snap2, r)).st_ino for r in BRAIN_SET)
    check(same, "every file is a hardlink of the previous snapshot")
    use2 = bm.disk_use_bytes(os.path.join(dest_a, "snapshots"))
    m2 = manifest(dest_a, s2)
    check(use2 - use1 < 64 * 1024 and m2["new_bytes"] == 0,
          "adds negligible disk (+%d bytes, manifest only)" % (use2 - use1))
    check(m2["hash_reused"] == len(BRAIN_SET),
          "unchanged files are not re-read (hash_reused=%d)"
          % m2["hash_reused"])

    # ---------------------------------------------------------------- D
    print("D. checkpoint save during the copy")
    src_d = os.path.join(tmp, "srcD")
    make_source(src_d)
    dest_d = os.path.join(tmp, "destD")
    env_d = write_env(os.path.join(tmp, "d.env"), dest_d)
    hook_py = os.path.join(tmp, "hook.py")
    with open(hook_py, "w") as f:
        f.write(HOOK_PY)
    target = os.path.join(src_d, "runlogs/breaks_by_type.json")
    hook_once = "%s %s %s %s once" % (sys.executable, hook_py, target,
                                      os.path.join(tmp, "c_once"))
    rc, out, err = run(env_d, src=src_d, now=T0, hook=hook_once)
    md = manifest(dest_d, latest(dest_d))
    check(rc == 0 and md["pulls"] == 2 and md["mixed"] is False,
          "one change mid-copy -> exactly one re-pull, not mixed "
          "(pulls=%s)" % md["pulls"])
    check(["runlogs/breaks_by_type.json"] in md["changed_during_copy"],
          "the changed file is named in the manifest")
    with open(os.path.join(dest_d, "snapshots", latest(dest_d),
                           "runlogs/breaks_by_type.json")) as f:
        check(json.load(f)["generation"] == 1,
              "snapshot holds the NEW generation, not the stale copy")
    hook_always = "%s %s %s %s always" % (sys.executable, hook_py, target,
                                          os.path.join(tmp, "c_always"))
    rc, out, err = run(env_d, src=src_d, now=T0 + 300, hook=hook_always)
    md = manifest(dest_d, latest(dest_d))
    check(rc == 0 and md["mixed"] is True and md["pulls"] == 4,
          "never-settling brain -> kept, mixed: true after 3 re-pulls "
          "(pulls=%s)" % md["pulls"])

    # ---------------------------------------------------------------- E
    print("E. truncated .pt")
    try:
        import torch
        tp = os.path.join(tmp, "real_torch.pt")
        torch.save({"w": torch.arange(5000, dtype=torch.float32)}, tp)
        check(bm.check_file(tp, "x.pt")[0] == "ok",
              "real torch.save output passes the zip check")
        with open(tp, "rb") as f:
            data = f.read()
        with open(tp, "wb") as f:
            f.write(data[: len(data) // 2])
        check(bm.check_file(tp, "x.pt")[0] == "bad",
              "a half-written torch.save file FAILS it")
    except ImportError:
        print("      (torch not importable here: real-format check skipped)")
    kp = os.path.join(tmp, "sidecar.pkl")
    with open(kp, "wb") as f:
        pickle.dump({"probes": list(range(5000))}, f)
    check(bm.check_file(kp, "x.pkl")[0] == "ok",
          "a whole default-protocol pickle passes the framing check")
    with open(kp, "rb") as f:
        data = f.read()
    with open(kp, "wb") as f:
        f.write(data[: len(data) // 2])
    check(bm.check_file(kp, "x.pkl")[0] == "bad",
          "a half-written sidecar pickle FAILS it")
    with open(kp, "wb") as f:
        pass
    check(bm.check_file(kp, "x.pkl")[0] == "bad", "an empty sidecar FAILS it")
    src_e = os.path.join(tmp, "srcE")
    make_source(src_e)
    dest_e = os.path.join(tmp, "destE")
    env_e = write_env(os.path.join(tmp, "e.env"), dest_e)
    rc, _o, _e = run(env_e, src=src_e, now=T0)
    good_e = latest(dest_e)
    bad_rel = "skill_bank_mc_rssm/skills/s1/policy.pt"
    bp = os.path.join(src_e, bad_rel)
    with open(bp, "rb") as f:
        data = f.read()
    with open(bp, "wb") as f:
        f.write(data[: len(data) // 2])
    before_e = snaps(dest_e)
    rc, out, err = run(env_e, src=src_e, now=T0 + 300)
    after_e = snaps(dest_e)
    check(rc == bm.EXIT_VERIFY_FAILED and "VERIFY FAILED" in out
          and bad_rel in out, "truncated .pt -> exit 12 naming the file")
    check(latest(dest_e) == good_e, "latest unchanged")
    check(set(before_e) <= set(after_e)
          and any(n.endswith(".failed") for n in after_e),
          "kept as .failed; no good snapshot pruned")
    rc2, out2, _ = run(env_e, src=src_e, now=T0 + 600)
    check(rc2 == bm.EXIT_VERIFY_FAILED and latest(dest_e) == good_e,
          "second identical failure still refused (2/3)")
    rc3, out3, _ = run(env_e, src=src_e, now=T0 + 900)
    m3 = manifest(dest_e, latest(dest_e))
    check(rc3 == bm.EXIT_DEGRADED and m3["degraded"] is True
          and m3["corrupt"] == [bad_rel] and latest(dest_e) != good_e,
          "3rd identical failure -> promoted degraded (latch escape)")
    check(good_e in snaps(dest_e), "the last good snapshot still exists")

    # ---------------------------------------------------------------- F
    print("F. retention over a synthetic timeline")
    step = 300
    now = T0 + 20 * 86400
    entries = [(bm.ts_name(t), t) for t in
               [now - i * step for i in range(20 * 86400 // step + 1)]]
    old_protect = entries[-1][0]
    keep = bm.select_keep(entries, now, 12, 48, 14, protect={old_protect})
    newest12 = [n for n, _ in sorted(entries, key=lambda e: -e[1])[:12]]
    check(set(newest12) <= keep, "the 12 newest are kept")
    by_hour, by_day = {}, {}
    for n, t in entries:
        if now - t <= 48 * 3600:
            by_hour.setdefault(int(t // 3600), []).append((t, n))
        if now - t <= 14 * 86400:
            by_day.setdefault(int(t // 86400), []).append((t, n))
    check(all(max(v)[1] in keep for v in by_hour.values()),
          "newest of every hour in 48h kept (%d hours)" % len(by_hour))
    check(all(max(v)[1] in keep for v in by_day.values()),
          "newest of every day in 14d kept (%d days)" % len(by_day))
    expect = set(newest12) | {max(v)[1] for v in by_hour.values()} \
        | {max(v)[1] for v in by_day.values()} | {old_protect}
    check(keep == expect, "keeps EXACTLY that set (%d of %d)"
          % (len(keep), len(entries)))
    check(not any(now - t > 14 * 86400 and n in keep and n != old_protect
                  for n, t in entries), "nothing older than 14 d unless "
          "protected; protected kept")
    dest_f = os.path.join(tmp, "destF")
    os.makedirs(os.path.join(dest_f, "snapshots"))
    syn = entries[1:][::7]                    # older than "now"
    for n, _t in syn:
        os.makedirs(os.path.join(dest_f, "snapshots", n))
    os.makedirs(os.path.join(dest_f, "snapshots",
                             bm.ts_name(now - 3 * 86400) + ".partial"))
    env_f = write_env(os.path.join(tmp, "f.env"), dest_f)
    rc, out, err = run(env_f, src=src, now=now)
    remaining = set(snaps(dest_f))
    want = bm.select_keep(syn + [(latest(dest_f), now)], now, 12, 48, 14,
                          protect={latest(dest_f)})
    check(rc == 0 and remaining == want,
          "live prune leaves exactly select_keep (%d left, stale .partial "
          "removed)" % len(remaining))

    # ---------------------------------------------------------------- G
    print("G. disk guard")
    before_g = snaps(dest_a)
    lat_g = latest(dest_a)
    rc, out, err = run(env_a, src=src, now=T0 + 600, free=1)
    check(rc == bm.EXIT_DISK_FULL and "DISK FULL" in out,
          "below MIN_FREE_GB -> exit 13, loud")
    check(latest(dest_a) == lat_g and lat_g in snaps(dest_a)
          and not (set(snaps(dest_a)) - set(before_g)),
          "pull skipped: no new snapshot, latest survives")

    # ---------------------------------------------------------------- H
    print("H. lock")
    lockp = os.path.join(dest_a, ".brain_mirror.lock")
    with open(lockp, "a+") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        n_before = len(snaps(dest_a))
        rc, out, err = run(env_a, src=src, now=T0 + 900)
        check(rc == bm.EXIT_LOCK_HELD and len(snaps(dest_a)) == n_before,
              "concurrent run refused (exit 10), nothing written")
        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    with open(lockp, "w") as lf:
        lf.write("999999 20200101T000000Z\n")     # holder died mid-run
    rc, out, err = run(env_a, src=src, now=T0 + 900)
    check(rc == 0, "stale lock file from a dead process is recovered")
    with open(os.path.join(dest_a, "mirror.log")) as f:
        check("recovered stale lock" in f.read(), "...and logged as such")

    # ---------------------------------------------------------------- I
    print("I. --restore-plan executes nothing")
    shim = os.path.join(tmp, "shim")
    os.makedirs(shim)
    marker = os.path.join(tmp, "EXECUTED")
    for tool in ("rsync", "ssh", "tar", "scp"):
        p = os.path.join(shim, tool)
        with open(p, "w") as f:
            f.write("#!/bin/sh\necho %s >> %s\n" % (tool, marker))
        os.chmod(p, 0o755)
    before_i = tree_digest(dest_a)
    rc, out, err = run(env_a, "--restore-plan", "latest", src=src,
                       path_prefix=shim)
    check(rc == 0 and not os.path.exists(marker),
          "plan printed; no rsync/ssh/tar invoked")
    check(tree_digest(dest_a) == before_i, "BRAIN_DEST untouched")
    cmds = [l for l in out.splitlines() if l and not l.startswith("#")]
    check(any("runlogs/STOP" in l for l in cmds)
          and any(l.startswith("rsync") and "skybot@" in l for l in cmds)
          and not any("--delete" in l for l in cmds),
          "commands: STOP first, rsync push, never --delete")
    check("NOTHING HAS BEEN EXECUTED" in out and "podlogs" in out
          and "break" in out.lower(), "carries the layout warnings")

    # ---------------------------------------------------------------- J
    print("J. --status")
    lat_ts = bm.name_ts(latest(dest_a))
    rc, out, err = run(env_a, "--status", now=lat_ts + 60)
    check(rc == 0 and "latest:" in out and "state: OK" in out,
          "fresh -> OK")
    rc, out, err = run(env_a, "--status", now=lat_ts + 3600)
    check(rc == bm.EXIT_STALE and "STALE" in out, "an hour old -> exit 16")

    # ---------------------------------------------------------------- K
    print("K. source never modified")
    check(tree_digest(src) == before,
          "source tree identical (bytes, mtimes, modes) after every run")

    # ---------------------------------------------------------------- M
    print("M. unreachable host over the real ssh path")
    if shutil.which("ssh"):
        dest_m = os.path.join(tmp, "destM")
        env_m = write_env(os.path.join(tmp, "m.env"), dest_m)
        rc, out, err = run(env_m, now=T0)
        check(rc == bm.EXIT_UNREACHABLE and not snaps(dest_m),
              "unresolvable host -> exit 11, no snapshot (rc=%d)" % rc)
    else:
        print("      (ssh not installed: skipped)")


def test_newest_clean_protected():
    """A file corrupt ON MAIN makes every later snapshot degraded. Retention
    must still keep the newest CLEAN snapshot, however old it gets."""
    print("N. newest clean snapshot survives a run of degraded ones")
    d = tempfile.mkdtemp(prefix="bm_clean_")
    try:
        now = 1_800_000_000.0
        cfg = {"dest": d}
        root = bm.snap_root(cfg)
        clean = bm.ts_name(now - 30 * 86400)          # 30 days old
        os.makedirs(os.path.join(root, clean))
        with open(os.path.join(root, clean, "MANIFEST.json"), "w") as f:
            json.dump({"degraded": False}, f)
        for k in range(20):                           # degraded ever since
            n = bm.ts_name(now - k * 300)
            os.makedirs(os.path.join(root, n))
            with open(os.path.join(root, n, "MANIFEST.json"), "w") as f:
                json.dump({"degraded": True}, f)
        check(bm.newest_clean(cfg) == clean, "newest_clean finds it")
        keep = bm.select_keep(bm.list_snaps(cfg)[0], now, 12, 48, 14,
                              bm.protected(cfg))
        check(clean in keep, "30-day-old clean snapshot kept despite "
              "14-day retention while newer ones are degraded")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
