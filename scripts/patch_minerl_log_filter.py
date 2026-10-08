#!/usr/bin/env python3
"""Stop MineRL's client log from recording MCP-Reborn's stat-name debug spam.

Born 2026-10-08: /workspace/devai/logs/ held 129 `mc_<port>.log` files totalling
117 GB on a 232 GB disk (single files up to 9.4 GB). Profiling a 200 MB slice:
2.79M lines, of which all but ~20 were one of 107 bare stat NAMES
(`[STDOUT]: mine_block.oak_log`), each repeated ~28k times.

Source: MCP-Reborn `JSONWorldDataHelper.buildBaseMinecraftStats` ends its loop
with `// Debugging  System.out.println(statTypeName + "." + statName);` — run
for every stat key on every observation. It prints the NAME only; the VALUE
goes into the JSON observation the agent already receives. The lines carry
zero information, so dropping them is lossless; compressing or "vectorising"
them would only store nothing more cheaply.

Why here and not in the Java: fixing the println needs a jar rebuild (the
40-60 min MCP build). MineRL's `log_to_file` thread in `minerl/env/malmo.py`
copies every stdout line into the log with a flush() per line, so a filter
there removes the disk cost, the per-line flush and the `_log_heuristic`
call. Startup lines, errors and stack traces are untouched.

Idempotent; exits non-zero if the anchor is missing (upstream changed), so a
provision log shows it rather than silently keeping the 9 GB logs.

    ./venv_mc/bin/python scripts/patch_minerl_log_filter.py [path/to/malmo.py]
"""
import os
import re
import sys

MARK = "# DEVAI: stat-name spam filter"
IMPORT_ANCHOR = "import locale\n"
IMPORT_ADD = (
    "import locale\n"
    "import re as _devai_re  " + MARK + "\n"
    "_DEVAI_STAT_SPAM = _devai_re.compile(\n"
    "    r'\\]: \\[STDOUT\\]: [a-z0-9_]+\\.[a-z0-9_./-]+\\r?$')\n"
)
LOOP_ANCHOR = (
    "                        self._log_heuristic(linestr)\n"
    "                        mine_log.write(line)\n"
)
LOOP_ADD = (
    "                        if _DEVAI_STAT_SPAM.search(linestr):  " + MARK + "\n"
    "                            continue\n"
) + LOOP_ANCHOR

# The regex must drop stat names and keep everything else; checked on every run
# so a regex edit cannot silently start eating real log lines.
_DROP = [
    "[18:06:48] [EnvServerSocketHandler/INFO]: [STDOUT]: mine_block.oak_log",
    "[18:06:48] [EnvServerSocketHandler/INFO]: [STDOUT]: custom.walk_on_water_one_cm\r",
    "[18:06:48] [EnvServerSocketHandler/INFO]: [STDOUT]: entity_killed_by.trader_llama",
]
_KEEP = [
    "[18:06:48] [Thread-3/INFO]: [STDOUT]: SERVER enter state: DORMANT",
    "[18:06:48] [EnvServerSocketHandler/INFO]: [STDOUT]: Setting width, height to 640, 360",
    "[18:06:48] [EnvServerSocketHandler/INFO]: [STDOUT]: Duplicate token! [a, b]",
    "[18:06:48] [EnvServerSocketHandler/INFO]: [STDOUT]: Gamma: 2.0",
    "[18:06:48] [Render thread/INFO]: [CHAT] SkyBot joined the game",
    "\tat com.microsoft.Malmo.Client.foo(Foo.java:12)",
    "java.lang.NullPointerException: null",
]


def _self_check():
    pat = re.compile(r'\]: \[STDOUT\]: [a-z0-9_]+\.[a-z0-9_./-]+\r?$')
    bad = [s for s in _DROP if not pat.search(s)] + [s for s in _KEEP if pat.search(s)]
    if bad:
        sys.exit("patch_minerl_log_filter: regex self-check FAILED on: %r" % bad)


def main():
    _self_check()
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        import minerl
        path = os.path.join(os.path.dirname(minerl.__file__), "env", "malmo.py")
    src = open(path).read()
    if MARK in src:
        print("  log-filter patch already present -> %s" % path)
        return
    if src.count(IMPORT_ANCHOR) != 1 or src.count(LOOP_ANCHOR) != 1:
        sys.exit("patch_minerl_log_filter: anchor not found exactly once in %s "
                 "(MineRL changed upstream?) - NOT patched" % path)
    if not os.path.exists(path + ".orig"):
        open(path + ".orig", "w").write(src)
    src = src.replace(IMPORT_ANCHOR, IMPORT_ADD, 1).replace(LOOP_ANCHOR, LOOP_ADD, 1)
    compile(src, path, "exec")
    open(path, "w").write(src)
    print("  log-filter patch OK -> %s" % path)


if __name__ == "__main__":
    main()
