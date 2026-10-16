#!/usr/bin/env python3
"""Install what a game needs, from the LAN game depot first, the internet second.

docs/UNIVERSAL_GAMES.md §12. Runs ON THE HOST (scripts/game_ctl.sh <game>
install). Everything it does is declared in the game's YAML, so a new game
needs a YAML, not a new branch in a shell script:

    game:
      install:
        apt:          [pulseaudio, chocolate-doom]   # required system packages
        apt_optional: [libopenal1]                   # warn, never fail
        debs:         [google-chrome-stable_current_amd64.deb]  # from <depot>/debs/
        pip:          ["vizdoom==1.3.1"]             # into this interpreter's venv
        roms:         doom                           # scripts/fetch_roms.sh
        files:                                       # from <depot>/files/<src>
          - {src: mygame/, dest: games/mygame/}
          - {src: x/data.pak, dest: games/x/data.pak, sha256: "..."}
        note: "free text printed by the installer"

THE DEPOT (default <repo>/depot = /workspace/devai/depot; gitignored, and no
deploy touches it — deploy_skybot.sh syncs named folders only, never --delete):

    <depot>/apt/      apt's download cache (Dir::Cache::archives): every .deb
                      fetched once is installed from here forever after
    <depot>/wheels/   pip wheels, installed with --no-index first
    <depot>/debs/     .deb files you dropped in by hand (Chrome, Steam, ...)
    <depot>/files/    game files: ROMs, WADs, a copied Steam library, ...

The runner keeps a copy on another machine and syncs it both ways around an
install (scripts/depot_sync.sh), so a re-provisioned or replaced host installs
over the LAN instead of re-downloading — and an install that reached the
internet once leaves everything it fetched in the depot.

WHY THE APT OPTIONS (2026-10-14, a failed install on main): 79 of 83 packages
came down from us.archive.ubuntu.com; the last four came from esm.ubuntu.com
(Ubuntu Pro) over IPv6 and timed out (`Could not wait for server fd`). So:
IPv4 forced (SKYBOT_APT_IPV4=0 turns it off), short per-connection timeouts
with retries, and on failure one `apt-get update` + `--fix-missing` retry;
if THAT fails too, one last install without the Ubuntu Pro (ESM) sources —
the archive carries the same packages (apt_without_esm) — before giving up
with the failing packages named. The second failed run (2026-10-15) proved
the esm/IPv6 failure deterministic, not flaky. Already-installed packages
are skipped up front, so a re-run touches the network for nothing that is
already there.

    python tools/game_install.py doom_player [--depot DIR] [--dry-run]

Exit 0 = installed and the launcher's preflight finds nothing missing.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Dict, List, Optional, Sequence

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# inside the repo dir but never synced or deleted by a deploy (deploy_skybot.sh
# rsyncs named folders only, without --delete); gitignored. /workspace itself
# is root-owned on main, so a sibling of devai/ could not be created by skybot.
DEFAULT_DEPOT = os.path.join(ROOT, "depot")


class InstallError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


class Installer:
    def __init__(self, game: Dict, depot: Optional[str], root: str = ROOT,
                 run: Optional[Callable] = None, py: str = sys.executable,
                 dry_run: bool = False, is_root: Optional[bool] = None,
                 ipv4: Optional[bool] = None,
                 installed: Optional[Callable[[str], bool]] = None,
                 apt_sources_dir: str = "/etc/apt/sources.list.d",
                 esm_fallback: Optional[bool] = None):
        self.game = game
        self.name = str(game.get("name", "?"))
        self.spec = dict(game.get("install") or {})
        self.root = root
        self.py = py
        self.dry = dry_run
        self._run = run or self._subprocess_run
        self.installed = installed or self.dpkg_installed
        self.apt_sources_dir = apt_sources_dir
        self.esm_fallback = ((os.environ.get("SKYBOT_APT_ESM_FALLBACK", "1") != "0")
                             if esm_fallback is None else esm_fallback)
        self.is_root = (os.geteuid() == 0) if is_root is None else is_root
        self.ipv4 = (os.environ.get("SKYBOT_APT_IPV4", "1") != "0") if ipv4 is None else ipv4
        self.depot = self._open_depot(depot)

    # ------------------------------------------------------------ plumbing
    @staticmethod
    def _subprocess_run(argv: Sequence[str], **kw) -> int:
        return subprocess.run(list(argv), **kw).returncode

    def run(self, argv: Sequence[str], **kw) -> int:
        print("  $ " + " ".join(argv), flush=True)
        if self.dry:
            return 0
        return self._run(list(argv), **kw)

    def sudo(self, argv: List[str]) -> List[str]:
        return argv if self.is_root else ["sudo", "-n"] + argv

    def _open_depot(self, depot: Optional[str]) -> Optional[str]:
        if not depot:
            log("depot: none configured — everything comes from the internet")
            return None
        try:
            for sub in ("apt/partial", "wheels", "debs", "files"):
                os.makedirs(os.path.join(depot, sub), exist_ok=True)
        except OSError as e:
            log(f"depot: {depot} unusable ({e}) — falling back to the internet")
            return None
        n = sum(len(fs) for _, _, fs in os.walk(depot))
        log(f"depot: {depot} ({n} files)")
        return depot

    # ------------------------------------------------------------------ apt
    @staticmethod
    def dpkg_installed(pkg: str) -> bool:
        try:
            out = subprocess.run(["dpkg-query", "-W", "-f=${Status}", pkg],
                                 capture_output=True, text=True).stdout
        except (OSError, FileNotFoundError):
            return False
        return out.strip().endswith("install ok installed")

    def apt_opts(self) -> List[str]:
        # 3, not 5: on main the failure was deterministic, and 5 retries only
        # stretched it to 12 minutes before anything else could be tried
        o = ["-o", "Acquire::Retries=3",
             "-o", "Acquire::http::Timeout=30", "-o", "Acquire::https::Timeout=30"]
        if self.ipv4:
            o += ["-o", "Acquire::ForceIPv4=true"]
        if self.depot:
            o += ["-o", f"Dir::Cache::archives={os.path.join(self.depot, 'apt')}"]
        return o

    def apt(self, pkgs: List[str], required: bool) -> bool:
        missing = [p for p in pkgs if not self.installed(p)]
        if not missing:
            if pkgs:
                log(f"apt: already installed ({', '.join(pkgs)}) — no download")
            return True
        base = self.sudo(["apt-get", "install", "-y", "-q"] + self.apt_opts())
        if self.run(base + missing) == 0:
            return True
        log("apt: install failed — refreshing package lists and retrying once")
        self.run(self.sudo(["apt-get", "update", "-q"] + self.apt_opts()))
        if self.run(base + ["--fix-missing"] + missing) == 0:
            return True
        if self.esm_fallback and self.apt_without_esm(missing):
            return True
        msg = (f"apt could not install {missing}. If the errors above name "
               f"esm.ubuntu.com or an IPv6 address, the host's route to that "
               f"mirror is the problem, not this game. Not root? `sudo -n` needs "
               f"passwordless sudo for {os.environ.get('USER', 'this user')}.")
        if required:
            raise InstallError(msg)
        log("WARN: " + msg)
        return False

    def apt_without_esm(self, missing: List[str]) -> bool:
        """Last resort: the same install with the Ubuntu Pro (ESM) sources left out.

        WHY (2026-10-15, the second failed install on main): the failure is
        not flaky. Both runs fetched everything else and died on the SAME four
        packages, all from esm.ubuntu.com over IPv6, and 5 retries only made
        it take 12 minutes. ESM's copies (ffmpeg's libav*, libfluidsynth3) are
        Ubuntu Pro SECURITY REBUILDS of packages the normal archive also
        carries — so without the ESM sources apt picks the archive versions
        and the install completes. The next `apt upgrade` that can reach ESM
        brings the patched builds back. Uses the package lists already on disk
        (no `apt-get update`, which would delete the ESM lists). Off with
        SKYBOT_APT_ESM_FALLBACK=0.
        """
        src = self.apt_sources_dir
        try:
            names = sorted(os.listdir(src))
        except OSError:
            return False
        esm = [n for n in names if "esm" in n.lower()]
        if not esm:
            return False
        tmp = tempfile.mkdtemp(prefix="skybot_apt_noesm_")
        for n in names:
            if n not in esm:
                os.symlink(os.path.join(src, n), os.path.join(tmp, n))
        log(f"WARN: esm.ubuntu.com unreachable — installing {missing} from the normal "
            f"Ubuntu archive WITHOUT the Ubuntu Pro sources ({', '.join(esm)}). "
            f"ESM security rebuilds return with the next `apt upgrade` that can reach "
            f"esm.ubuntu.com. To stop this: SKYBOT_APT_ESM_FALLBACK=0.")
        try:
            return self.run(self.sudo(["apt-get", "install", "-y", "-q"] + self.apt_opts()
                                      + ["-o", f"Dir::Etc::SourceParts={tmp}"])
                            + missing) == 0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def debs(self, names: List[str]) -> None:
        if not names:
            return
        paths, absent = [], []
        for n in names:
            p = os.path.join(self.depot, "debs", n) if self.depot else None
            (paths if p and os.path.isfile(p) else absent).append(p or n)
        if absent:
            raise InstallError(
                f"{absent} not in the depot's debs/. Download them once and drop "
                f"them into the RUNNER's depot debs/ folder (docs/OPERATIONS.md, "
                f"'Game depot'); the next install copies them here")
        if self.run(self.sudo(["apt-get", "install", "-y", "-q"] + self.apt_opts()
                              + paths)) != 0:
            raise InstallError(f"apt could not install {paths}")

    # ------------------------------------------------------------------ pip
    def pip(self, specs: List[str]) -> None:
        if not specs:
            return
        pip = [self.py, "-m", "pip"]
        if not self.depot:
            if self.run(pip + ["install", "-q"] + specs) != 0:
                raise InstallError(f"pip could not install {specs}")
            return
        wheels = os.path.join(self.depot, "wheels")
        offline = pip + ["install", "-q", "--no-index", "--find-links", wheels] + specs
        # already installed, or every wheel is in the depot: no network at all
        if self.run(offline) == 0:
            log(f"pip: {specs} from the depot (or already installed)")
            return
        log("pip: not all in the depot — downloading into it once")
        if self.run(pip + ["download", "-q", "-d", wheels] + specs) != 0:
            raise InstallError(f"pip could not download {specs}")
        if self.run(offline) != 0:
            raise InstallError(f"pip could not install {specs} from {wheels}")

    # ---------------------------------------------------------------- files
    @staticmethod
    def sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def files(self, entries: List[Dict]) -> None:
        for e in entries or []:
            src, dest, want = e.get("src"), e.get("dest"), e.get("sha256")
            if not src or not dest:
                raise InstallError(f"install.files entry needs src and dest: {e!r}")
            d = dest if os.path.isabs(dest) else os.path.join(self.root, dest)
            if os.path.exists(d) and (not want or os.path.isdir(d) or self.sha256(d) == want):
                log(f"files: {dest} present")
                continue
            s = os.path.join(self.depot, "files", src) if self.depot else None
            if not s or not os.path.exists(s):
                raise InstallError(
                    f"{dest} is missing and the depot has no files/{src}. Put the "
                    f"game files in the RUNNER's depot at files/{src} (docs/OPERATIONS.md, "
                    f"'Game depot') and re-run install")
            if self.dry:
                log(f"files: would copy files/{src} -> {dest}")
                continue
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(d) or ".", exist_ok=True)
                shutil.copy2(s, d)
            if want and os.path.isfile(d) and self.sha256(d) != want:
                raise InstallError(f"{dest}: sha256 {self.sha256(d)} != pinned {want}")
            log(f"files: {dest} copied from the depot")

    def roms(self, which: Optional[str]) -> None:
        if not which:
            return
        env = dict(os.environ, SKYBOT_DEPOT=self.depot or "")
        if self.run(["bash", os.path.join(self.root, "scripts", "fetch_roms.sh"),
                     self.py], env=env) != 0:
            raise InstallError(f"fetch_roms.sh could not provide roms/{which}")

    # ------------------------------------------------------------------ all
    def install(self) -> List[str]:
        sp = self.spec
        if sp.get("note"):
            log(str(sp["note"]))
        self.apt(list(sp.get("apt") or []), required=True)
        self.apt(list(sp.get("apt_optional") or []), required=False)
        self.debs(list(sp.get("debs") or []))
        self.pip(list(sp.get("pip") or []))
        self.roms(sp.get("roms"))
        self.files(list(sp.get("files") or []))
        return self.check()

    def check(self) -> List[str]:
        """What the launcher still lacks (player games); [] = ready."""
        if self.game.get("backend") != "player":
            return []
        sys.path.insert(0, self.root)
        from developmental_ai.environments.player.launchers import (
            launcher_of, preflight)
        miss = preflight(self.game, root=self.root)
        lb = launcher_of(self.game)
        if lb.get("type") == "steam" and miss:
            appid = lb.get("appid")
            log("Steam games need a one-time human step on the host (a login "
                "nothing here can type):")
            log("  1. install Steam:  drop steam_latest.deb into the depot's debs/ and "
                "list it under install.debs, or `sudo apt-get install steam-installer`")
            log("  2. log in once with 'remember me' (or `steamcmd +login <user>`)")
            log(f"  3. install the game: steamcmd +login <user> +app_update {appid} validate +quit")
            log("     — or copy an existing Steam library into the depot under files/ "
                "and list it in install.files")
        return miss


def load_game(name: str, root: str = ROOT) -> Dict:
    path = os.path.join(root, "configs", "games", f"{name}.yaml")
    if not os.path.isfile(path):
        raise InstallError(f"unknown game {name!r}: no {path}")
    with open(path) as f:
        return (yaml.safe_load(f) or {}).get("game") or {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("game")
    ap.add_argument("--depot", default=os.environ.get("SKYBOT_DEPOT", DEFAULT_DEPOT),
                    help="game depot directory ('' = none)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        inst = Installer(load_game(a.game.lower()), a.depot or None, dry_run=a.dry_run)
        miss = inst.install()
    except InstallError as e:
        print(f"GAME-INSTALL FAILED ({a.game}): {e}", flush=True)
        return 1
    if miss:
        print(f"GAME-INSTALL INCOMPLETE ({a.game}): still missing {miss}", flush=True)
        return 1
    print(f"GAME-INSTALL OK ({a.game})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
