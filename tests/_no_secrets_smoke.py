"""No addresses or credentials in TRACKED files (2026-10-04).

THE REPO IS PUBLIC. Real LAN addresses were written into eleven tracked
files (docs, .env examples, a Prometheus config) before anyone noticed; the
real values belong only in gitignored files (`CLAUDE.local.md`, the runner
`.env`, `docker/monitor/.env`, node2's `scripts/brain_mirror/brain_mirror.env`).
This suite makes `git add -A` safe to use: anything that slips through fails
CI before it is public, instead of being found by reading the diff.

Contracts:
    A. NO PRIVATE OR TAILNET ADDRESSES in any tracked text file: RFC1918
       (10/8, 172.16/12, 192.168/16) and Tailscale CGNAT (100.64/10).
       Documentation-only ranges (192.0.2/24, 198.51.100/24, 203.0.113/24),
       loopback and 0.0.0.0 are allowed.
    B. NO CREDENTIAL MATERIAL: private-key blocks, GitHub tokens, Tailscale
       auth keys.
    C. THE LOCAL-ONLY FILES STAY IGNORED: if one of them stopped being
       ignored, the next `git add -A` would publish it.

Run: PYTHONPATH=. python tests/_no_secrets_smoke.py
"""
import ipaddress
import os
import re
import subprocess
import sys

NAME = "no-secrets"

IP_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])")
TAILNET = ipaddress.ip_network("100.64.0.0/10")
ALLOWED = [ipaddress.ip_network(n) for n in
           ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]
CRED_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\bghp_[A-Za-z0-9]{30,}"
    r"|\bgithub_pat_[A-Za-z0-9_]{30,}"
    r"|\btskey-[A-Za-z0-9-]{10,}")
LOCAL_ONLY = ("CLAUDE.local.md", "scripts/brain_mirror/brain_mirror.env",
              "docker/monitor/.env", "docker/runner/.env",
              "docker/runner/runner.env", ".claude/settings.local.json",
              "runlogs/anything")


def tracked():
    """Tracked files PLUS untracked-but-not-ignored ones: exactly the set a
    `git add -A` would commit, so running this before committing covers new
    files too, not only what is already in the index."""
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others",
                          "--exclude-standard"], capture_output=True,
                         check=True).stdout.decode()
    return sorted({p for p in out.split("\0") if p and os.path.isfile(p)})


def is_real_private(s):
    try:
        ip = ipaddress.ip_address(s)
    except ValueError:
        return False                     # 999.1.2.3, version strings, etc.
    if any(ip in n for n in ALLOWED) or ip.is_loopback or ip.is_unspecified:
        return False
    return ip.is_private and not ip.is_link_local or ip in TAILNET


def text_of(path):
    with open(path, "rb") as f:
        b = f.read()
    if b"\0" in b[:4096]:
        return None                      # binary (png, pdf, pt)
    return b.decode("utf-8", "replace")


def main():
    if subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                      capture_output=True).returncode != 0:
        print(f"  SKIP: not a git checkout\n[{NAME}] ALL PASS")
        return
    me = os.path.abspath(__file__)
    ip_hits, cred_hits = [], []
    files = tracked()
    for p in files:
        t = text_of(p)
        if t is None or os.path.abspath(p) == me:
            continue                     # binary, or this file's own patterns
        for n, line in enumerate(t.splitlines(), 1):
            for m in IP_RE.finditer(line):
                if is_real_private(m.group(1)):
                    ip_hits.append(f"{p}:{n}: {m.group(1)}")
            if CRED_RE.search(line):
                cred_hits.append(f"{p}:{n}")
    assert not ip_hits, ("private/tailnet addresses in tracked files (the "
                         "repo is public; put real values in a gitignored "
                         "file and use a placeholder):\n  "
                         + "\n  ".join(ip_hits[:30]))
    print(f"  A. {len(files)} files `git add -A` would commit: no RFC1918 / "
          f"tailnet addresses")
    assert not cred_hits, "credential material in: " + ", ".join(cred_hits)
    print("  B. no private keys, GitHub tokens or Tailscale auth keys")
    not_ignored = [f for f in LOCAL_ONLY if subprocess.run(
        ["git", "check-ignore", "-q", f]).returncode != 0]
    assert not not_ignored, ("these local-only paths are NOT gitignored; "
                             "`git add -A` would publish them: "
                             f"{not_ignored}")
    print(f"  C. local-only files still ignored: {', '.join(LOCAL_ONLY)}")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        print(f"[{NAME}] FAILED")
        sys.exit(1)
