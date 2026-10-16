"""Game installs: the LAN depot first, the internet second, nothing twice.

Born 2026-10-14 from a failed install on `main`: `Doom -> player -> install`
fetched 79 of 83 packages at ~127 kB/s over 8 minutes, then died on four from
esm.ubuntu.com (Ubuntu Pro) over IPv6 — `Could not wait for server fd`. The
fix (tools/game_install.py, scripts/fetch_roms.sh) is encoded here against a
FAKE command runner, so it holds on any machine:

    A. NOTHING TWICE: packages already installed are skipped before apt is
       even called — a re-run touches no network for them.
    B. THE APT CALL ITSELF: IPv4 forced (the failing route was IPv6; and the
       knob turns it off), retries + short timeouts, and apt's download cache
       pointed INTO THE DEPOT so every .deb fetched once is kept; `sudo -n`
       only when not root.
    C. ONE HONEST RETRY: on failure, `apt-get update` then `--fix-missing`;
       then (2026-10-15: the esm/IPv6 failure repeated identically on a
       second run) one install WITHOUT the Ubuntu Pro sources, which the
       archive's own copies satisfy; a final failure names the packages and
       the esm/IPv6 cause. Optional packages warn and never fail the install.
    D. PIP FROM THE DEPOT: --no-index from the depot's wheels first (already
       installed = no network); only on a miss is it downloaded INTO the depot
       and installed from there. No depot -> a plain pip install.
    E. GAME FILES FROM THE DEPOT: copied with sha256 verification; a missing
       source names the exact depot path to fill; a present file is skipped.
    F. A DEPOT THAT CANNOT BE USED IS NOT A FAILURE: it falls back to the
       internet with a warning (no latch on a missing disk).
    G. HAND-DROPPED .debs (Chrome, Steam) install from the depot's debs/, and
       a missing one is named.
    H. THE SHIPPED GAMES: doom / doom_player / minecraft declare installs the
       installer understands; minecraft's is a no-op.
    I. FETCH_ROMS SOURCE ORDER: a corrupt depot copy is skipped (named) and
       REPLACED by the verified copy from the next source; no `unzip` needed.

Run: PYTHONPATH=. python tests/_game_install_smoke.py
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.game_install import InstallError, Installer, load_game, main as gi_main  # noqa: E402

NAME = "game-install"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


class FakeRun:
    """Records argv; fails any call whose joined argv contains a needle."""

    def __init__(self, fail=()):
        self.calls, self.fail = [], list(fail)

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        j = " ".join(argv)
        for i, needle in enumerate(self.fail):
            if needle in j:
                self.fail.pop(i)        # each needle fails exactly once
                return 1
        return 0

    def find(self, needle):
        return [c for c in self.calls if needle in " ".join(c)]


def inst(game, depot, run, installed=lambda p: False, **kw):
    return Installer(game, depot, run=run, py="PY", installed=installed,
                     is_root=kw.pop("is_root", False), **kw)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    tmp = tempfile.mkdtemp(prefix="gameinst_")
    try:
        run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{NAME}] ALL PASS")


def run(tmp):
    depot = os.path.join(tmp, "depot")
    g = {"name": "t", "install": {"apt": ["pulseaudio", "chocolate-doom"]}}

    # ---------------------------------------------------------------- A
    r = FakeRun()
    inst(g, depot, r, installed=lambda p: True).apt(["pulseaudio", "chocolate-doom"], True)
    check(r.calls == [], "A. everything already installed: apt is never called")
    r = FakeRun()
    inst(g, depot, r, installed=lambda p: p == "pulseaudio").apt(
        ["pulseaudio", "chocolate-doom"], True)
    check(len(r.calls) == 1 and r.calls[0][-1] == "chocolate-doom"
          and "pulseaudio" not in r.calls[0],
          "A. only the missing package is asked for")

    # ---------------------------------------------------------------- B
    c = r.calls[0]
    j = " ".join(c)
    check(c[:2] == ["sudo", "-n"], "B. not root: sudo -n (fails fast, never prompts)")
    check("Acquire::ForceIPv4=true" in j, "B. IPv4 forced (the failing esm route was IPv6)")
    check("Acquire::Retries=3" in j and "Acquire::https::Timeout=30" in j,
          "B. retries + per-connection timeouts")
    check(f"Dir::Cache::archives={os.path.join(depot, 'apt')}" in j
          and os.path.isdir(os.path.join(depot, "apt", "partial")),
          "B. apt's download cache is the depot (every .deb fetched once is kept)")
    r = FakeRun()
    inst(g, depot, r, is_root=True, ipv4=False).apt(["x"], True)
    check(r.calls[0][0] == "apt-get" and "ForceIPv4" not in " ".join(r.calls[0]),
          "B. root: no sudo; SKYBOT_APT_IPV4=0 turns IPv4-forcing off")

    # ---------------------------------------------------------------- C
    r = FakeRun(fail=["install -y"])
    ok = inst(g, depot, r).apt(["chocolate-doom"], True)
    check(ok and r.find("apt-get update") and "--fix-missing" in r.calls[-1],
          "C. first failure: apt-get update, then one --fix-missing retry (succeeds)")
    r = FakeRun(fail=["install -y", "install -y"])
    try:
        inst(g, depot, r, esm_fallback=False).apt(["chocolate-doom"], True)
        msg = ""
    except InstallError as e:
        msg = str(e)
    check("chocolate-doom" in msg and "esm.ubuntu.com" in msg,
          "C. second failure: names the packages and the esm/IPv6 cause")
    r = FakeRun(fail=["install -y", "install -y"])
    check(inst(g, depot, r, esm_fallback=False).apt(["libopenal1"], False) is False,
          "C. an optional package failing only warns")
    # the second failed run on main (2026-10-15): the SAME four esm packages
    # over IPv6 again — deterministic. Last resort: no Ubuntu Pro sources.
    srcd = os.path.join(tmp, "sources.list.d")
    os.makedirs(srcd)
    for f in ("ubuntu.sources", "ubuntu-esm-apps.sources", "ubuntu-esm-infra.sources"):
        open(os.path.join(srcd, f), "w").write("x")
    seen = {}

    class SpyRun(FakeRun):
        def __call__(self, argv, **kw):
            j = " ".join(argv)
            if "Dir::Etc::SourceParts=" in j:
                d = j.split("Dir::Etc::SourceParts=")[1].split()[0]
                seen["parts"] = sorted(os.listdir(d))
            return super().__call__(argv, **kw)
    r = SpyRun(fail=["install -y", "install -y"])
    ok = inst(g, depot, r, apt_sources_dir=srcd).apt(["chocolate-doom"], True)
    check(ok and seen.get("parts") == ["ubuntu.sources"],
          "C. esm still unreachable: one last install WITHOUT the esm sources (archive versions)")
    r = SpyRun(fail=["install -y", "install -y"])
    try:
        inst(g, depot, r, apt_sources_dir=srcd, esm_fallback=False).apt(["x"], True)
        refused = False
    except InstallError:
        refused = True
    check(refused and not r.find("SourceParts"),
          "C. SKYBOT_APT_ESM_FALLBACK=0: no esm-less install, the failure stands")

    # ---------------------------------------------------------------- D
    r = FakeRun()
    inst(g, depot, r).pip(["mss", "python-xlib"])
    check(len(r.calls) == 1 and "--no-index" in r.calls[0]
          and os.path.join(depot, "wheels") in r.calls[0],
          "D. pip: depot wheels with --no-index first; satisfied = no network")
    r = FakeRun(fail=["--no-index"])
    inst(g, depot, r).pip(["vizdoom==1.3.1"])
    dl = r.find("pip download")
    check(dl and "-d" in dl[0] and os.path.join(depot, "wheels") in dl[0]
          and "--no-index" in r.calls[-1],
          "D. a miss downloads INTO the depot, then installs from it")
    r = FakeRun()
    inst(g, None, r).pip(["mss"])
    check(r.calls == [["PY", "-m", "pip", "install", "-q", "mss"]],
          "D. no depot: a plain pip install")

    # ---------------------------------------------------------------- E
    os.makedirs(os.path.join(depot, "files", "g"), exist_ok=True)
    src = os.path.join(depot, "files", "g", "data.pak")
    open(src, "wb").write(b"game data")
    root = os.path.join(tmp, "root")
    os.makedirs(root)
    good = sha(src)
    i = Installer({"name": "t"}, depot, root=root, run=FakeRun(), installed=lambda p: True)
    i.files([{"src": "g/data.pak", "dest": "games/g/data.pak", "sha256": good}])
    check(open(os.path.join(root, "games/g/data.pak"), "rb").read() == b"game data",
          "E. a game file is copied from the depot and verified")
    try:
        i.files([{"src": "g/data.pak", "dest": "games/g/other.pak", "sha256": "0" * 64}])
        msg = ""
    except InstallError as e:
        msg = str(e)
    check("sha256" in msg, "E. a sha256 mismatch fails loudly")
    try:
        i.files([{"src": "nope/x.wad", "dest": "games/x.wad"}])
        msg = ""
    except InstallError as e:
        msg = str(e)
    check("files/nope/x.wad" in msg, "E. a missing source names the depot path to fill")
    os.makedirs(os.path.join(depot, "files", "lib", "sub"), exist_ok=True)
    open(os.path.join(depot, "files", "lib", "sub", "f"), "w").write("1")
    i.files([{"src": "lib/", "dest": "games/lib/"}])
    check(os.path.isfile(os.path.join(root, "games/lib/sub/f")),
          "E. a whole directory (e.g. a copied Steam library) is copied")

    # ---------------------------------------------------------------- F
    blocker = os.path.join(tmp, "a_file")
    open(blocker, "w").write("x")
    i = inst(g, os.path.join(blocker, "depot"), FakeRun())
    check(i.depot is None, "F. an unusable depot falls back to the internet, no failure")

    # ---------------------------------------------------------------- G
    try:
        inst(g, depot, FakeRun()).debs(["google-chrome-stable_current_amd64.deb"])
        msg = ""
    except InstallError as e:
        msg = str(e)
    check("debs/" in msg and "google-chrome" in msg, "G. a missing hand-dropped .deb is named")
    open(os.path.join(depot, "debs", "google-chrome-stable_current_amd64.deb"), "wb").write(b"x")
    r = FakeRun()
    inst(g, depot, r).debs(["google-chrome-stable_current_amd64.deb"])
    check(r.calls and r.calls[0][-1].endswith("debs/google-chrome-stable_current_amd64.deb"),
          "G. a .deb in the depot's debs/ is installed by path")

    # ---------------------------------------------------------------- H
    dp, dz, mc = load_game("doom_player"), load_game("doom"), load_game("minecraft")
    check({"pulseaudio", "chocolate-doom"} <= set(dp["install"]["apt"])
          and dp["install"]["roms"] == "doom" and "mss" in dp["install"]["pip"],
          "H. doom_player declares pulseaudio + chocolate-doom, mss, the Doom roms")
    check("vizdoom==1.3.1" in dz["install"]["pip"]
          and "libopenal1" in dz["install"]["apt_optional"],
          "H. doom declares vizdoom==1.3.1 and optional libopenal1")
    r = FakeRun()
    check(Installer(mc, depot, run=r).install() == [] and r.calls == [],
          "H. minecraft's install is a no-op (provisioning builds MineRL)")
    check(gi_main(["minecraft", "--depot", "", "--dry-run"]) == 0,
          "H. the CLI runs end to end (minecraft, dry run)")

    # ---------------------------------------------------------------- I
    wads = None
    for cand in (os.path.join(ROOT, "roms", "doom"),):
        if os.path.isfile(os.path.join(cand, "freedoom2.wad")):
            wads = cand
    if wads is None:
        try:
            import vizdoom
            wads = os.path.dirname(vizdoom.__file__)
        except Exception:
            wads = None
    if wads is None or shutil.which("bash") is None:
        print("  -- SKIP I: no local Freedoom copy (roms/doom or vizdoom) on this machine")
        return
    tree = os.path.join(tmp, "tree")
    os.makedirs(os.path.join(tree, "scripts"))
    shutil.copy(os.path.join(ROOT, "scripts", "fetch_roms.sh"), os.path.join(tree, "scripts"))
    d2 = os.path.join(tmp, "depot2", "files", "doom")
    os.makedirs(d2)
    for f in ("freedoom1.wad", "freedoom2.wad"):
        shutil.copy(os.path.join(wads, f), os.path.join(d2, f))
    open(os.path.join(d2, "freedoom2.wad"), "wb").write(b"corrupt")
    env = dict(os.environ, SKYBOT_DEPOT=os.path.join(tmp, "depot2"))
    p = subprocess.run(["bash", "scripts/fetch_roms.sh", sys.executable], cwd=tree,
                       capture_output=True, text=True, env=env)
    out = p.stdout + p.stderr
    check(p.returncode == 0 and "freedoom1.wad from " in out and "files/doom/freedoom1.wad" in out,
          "I. a good depot copy is used first (no download)")
    check("freedoom2.wad sha256" in out and "skipped" in out,
          "I. a corrupt depot copy is skipped and named")
    check(sha(os.path.join(d2, "freedoom2.wad")) == sha(os.path.join(tree, "roms/doom/freedoom2.wad"))
          and "saved to the depot" in out,
          "I. ...and replaced in the depot by the verified copy")
    import re
    cmd_lines = [ln for ln in open(os.path.join(ROOT, "scripts/fetch_roms.sh"))
                 if not ln.lstrip().startswith("#")]
    check(not any(re.search(r"(^|[\s;|&(])unzip\s", ln) for ln in cmd_lines),
          "I. no `unzip` binary is invoked (Python's zipfile instead)")


if __name__ == "__main__":
    main()
