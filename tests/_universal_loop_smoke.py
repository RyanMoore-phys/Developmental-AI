"""Universal-brain wiring in the loop and the ops path.

Born 2026-10-11 (docs/UNIVERSAL_GAMES.md). The loop has TWO live collection
bodies plus an unwired third (CLAUDE.md 4.2); an edit that lands in one and
not the other is silent. This pins every universal-brain edit by count, and
exercises the host-side game control end to end in a temp root.

Contracts:
    A. BOTH BODIES FEED THE STUCK WATCH: `_feed_stream_rewards(envs, rewards,
       intrinsic, n)` appears exactly twice, each immediately before the
       shadow after_step hook (after reward assembly, before the advance).
    B. ALL THREE BUILD SITES GO THROUGH ONE BUILDER when the config has
       streams: primary (`_stream_spec(0)`), startup scouts
       (`_stream_spec(_sidx)`), rebuilds (`_make_one_env`). The legacy calls
       remain in the `else` branches — a config without streams is the
       agent it was.
    C. SLOT-0 REBUILD MOVES self.env to the new client (it used to keep the
       reaped one: viewer/histories read a dead env and shutdown never closed
       the replacement — the bebd8db orphan class).
    D. MASKS: the executor ANDs the stream mask at BOTH decision sites (the
       scout-primitive path and the option-capable path); the loop installs
       them once the fleet exists.
    E. LEGACY UNCHANGED: make_env wraps in the controller only when given a
       universal block; ViZDoom refuses to run outside one.
    F. LAUNCHER: absent games file -> configs/minecraft_skybot.yaml, exactly
       as before; present -> runlogs/run_config.yaml composed at launch; the
       Paper-server/X preflight runs only when Minecraft is enabled (a
       Doom-only run must not be refused because a game it is not playing is
       down — that would be a latch); Doom preflight refuses without vizdoom.
    H. ONE MEANING OF TIME: the MineRL adapter reports its fixed step
       (action_repeat ticks at 20 ticks/s) as decision_seconds, so `tempo`
       is a real reading in every game rather than "unknown" in Minecraft.
    G. game_ctl.sh, end to end in a temp root: enable switches to universal;
       disabling the LAST game is refused with its exits named (no latch);
       an unknown game is refused; legacy returns to the old brain (with a
       .bak); the composed file parses through the same composer.

Run: PYTHONPATH=. python tests/_universal_loop_smoke.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "universal-loop"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def read(rel):
    return open(os.path.join(ROOT, rel)).read()


def main():
    print(f"[{NAME}]")
    loop = read("developmental_ai/core/developmental_loop.py")
    # ---- A
    feed = "            self._feed_stream_rewards(envs, rewards, intrinsic, n)\n"
    check(loop.count(feed) == 2, "A. stuck feed in BOTH live bodies (count 2)")
    check(loop.count(feed + "            if self._shadow is not None:\n"
                     "                self._shadow.after_step(") == 2,
          "A. each sits right before the shadow after_step hook")
    # ---- B
    check("        if self._stream_spec(0) is not None:\n" in loop
          and "self._build_stream_env(\n                0, render_mode=_render_mode)" in loop,
          "B. primary env built by the stream builder when streams exist")
    check("            if self._stream_spec(_sidx) is not None:\n"
          "                e, c = self._build_stream_env(_sidx)" in loop,
          "B. startup scouts built by the stream builder")
    check("        if idx >= 0 and self._stream_spec(idx) is not None:\n"
          "            e, _c = self._build_stream_env(idx)" in loop,
          "B. rebuilds built by the stream builder")
    check(loop.count("make_env(**kw)") == 1 and loop.count("env_name=self.env_name,") == 3,
          "B. one builder call; the 3 legacy calls remain for configs without streams")
    # ---- C
    i = loop.index("def _bounded_rebuild(i):")
    blk = loop[i:loop.index("return np.zeros(self.obs_dim", i)]
    check("if i == 0:" in blk and "self.env = envs[0]" in blk,
          "C. a slot-0 rebuild moves self.env to the live client")
    # ---- D
    opts = read("developmental_ai/policy/options.py")
    check(opts.count("self._and_stream(") == 2, "D. executor ANDs stream masks at both sites")
    check("self._install_stream_masks()" in loop
          and "self.option_executor.stream_masks = self._stream_masks" in loop,
          "D. loop installs the masks once the fleet exists")
    # ---- E
    wr = read("developmental_ai/environments/wrappers.py")
    check('if not universal or not universal.get("enabled"):\n            return env' in wr,
          "E. no universal block -> adapter unwrapped (legacy byte-identical)")
    check("ViZDoom streams run only on the universal brain" in wr,
          "E. ViZDoom refuses to run outside the universal brain")
    # ---- F
    ls = read("scripts/launch_skybot.sh")
    check("CONFIG=configs/minecraft_skybot.yaml\n" in ls and '--config "$CONFIG"' in ls,
          "F. absent games file -> the legacy config, unchanged")
    check("tools/compose_run_config.py --games-file" in ls
          and "CONFIG=runlogs/run_config.yaml" in ls, "F. present -> composed at launch")
    i = ls.index("# MINECRAFT-ONLY PREFLIGHT")
    gated = ls[i:ls.index("\nfi\n", i)]
    check('if [ "$MC_ON" = 1 ]; then' in gated and "mc_ping.py" in gated
          and "socat bridge down" in gated and "Xvfb" in gated,
          "F. server bridge / ping / X preflight gated on Minecraft being enabled")
    check('import vizdoom' in ls and 'REFUSED: doom enabled but vizdoom' in ls,
          "F. Doom-enabled launch refuses without vizdoom, naming the fix")
    b = ls.index("# ---- PREFLIGHT BEGIN")
    check(ls.index("# ---- WHICH GAMES") < b,
          "F. game selection sits OUTSIDE the preflight region the incident test strips")
    # ---- H: one meaning of time across games
    mc = read("developmental_ai/environments/minerl_env.py")
    check('_w["decision_seconds"] = float(self.action_repeat) / 20.0' in mc,
          "H. Minecraft reports its fixed step as tempo (not 'unknown')")
    # ---- G
    td = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(td, "scripts"))
        shutil.copytree(os.path.join(ROOT, "configs"), os.path.join(td, "configs"))
        shutil.copy(os.path.join(ROOT, "scripts/game_ctl.sh"), os.path.join(td, "scripts"))
        env = dict(os.environ, SKYBOT_ROOT=td)

        def ctl(*a):
            return subprocess.run(["bash", "scripts/game_ctl.sh", *a], cwd=td, env=env,
                                  capture_output=True, text=True)
        r = ctl("doom", "enable")
        f = os.path.join(td, "runlogs/games.enabled")
        check(r.returncode == 0 and "UNIVERSAL" in r.stdout and os.path.isfile(f),
              "G. first enable switches the host to the universal brain")
        r = ctl("doom", "disable")
        check(r.returncode == 3 and "legacy" in r.stdout and "Enable another" in r.stdout,
              "G. disabling the LAST game refused, exits named")
        check(ctl("pokemon", "enable").returncode == 2, "G. unknown game refused")
        ctl("minecraft", "enable")
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        from compose_run_config import compose, read_games_file
        games = read_games_file(f)
        c = compose(games, os.path.join(td, "configs/minecraft_skybot.yaml"),
                    os.path.join(td, "configs/universal.yaml"),
                    os.path.join(td, "configs/games"))
        check(games == ["doom", "minecraft"]
              and [s["game"] for s in c["parallel_envs"]["streams"]][0] == "minecraft",
              "G. the managed file composes; Minecraft takes slot 0")
        r = ctl("any", "legacy")
        check(r.returncode == 0 and not os.path.exists(f) and os.path.exists(f + ".bak"),
              "G. legacy returns to the old brain (games file kept as .bak)")
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
