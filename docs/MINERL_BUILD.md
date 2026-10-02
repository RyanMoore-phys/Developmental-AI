# Building MineRL — the traps

`scripts/provision_host.sh` automates all of this. You should not need to run
any of it by hand — but when a build fails, this is why. Every trap below
breaks the build SILENTLY, which is what makes them expensive.

MineRL v1.0.0 + MCP-Reborn is **notoriously fragile** to build. The authoritative
script is `data/scripts/provision_host.sh` (also `scripts/provision_host.sh` in the
parent repo). This doc explains the non-obvious traps so you can debug it — do
NOT try to reproduce the build by hand from memory; run the script and consult
this when a step fails.

## When you need this

Only when the training host's `/workspace/devai/venv_mc` or the built MineRL jar
(`venv_mc/lib/python3.10/site-packages/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar`)
is missing. `/workspace` is an ordinary directory on the training host that
you create once (see REPLICATION.md); nothing in this repo creates it.
stop/start normally KEEPS the venv + jar — you only re-provision on a genuinely
fresh volume.

## Host prerequisites (apt)

`openjdk-8`, `xvfb`, `openbox`, `xdotool`, plus a Python 3.10 venv (`venv_mc`)
with torch, and Ollama + `llava:7b`. A window manager (**openbox**) is
mandatory — see the focus gotcha below.

## The build order (must be exact)

1. `git clone minerl @v1.0.0` (NOT `@master` — master is old 0.4 and crashes on
   import against modern gym's `spaces.Discrete`).
2. `scripts/setup_mcp.sh` — materializes the MCP-Reborn source (not committed).
3. `scripts/patch_mcp.sh` — applies the shadow-plugin + synthetic-input MouseHelper
   bridge. Without it `shadowJar` doesn't exist AND input doesn't work.
4. **THEN** the DEVAI focus patch on the already-officially-patched MouseHelper
   (doing DEVAI first makes the official hunks reject). Apply via **Python, not
   sed** (the `||` in the replacement clashes with sed's delimiter).
5. `gradlew downloadAssets` → `gradlew clean build shadowJar`.
6. `pip install --no-build-isolation ./mc-build`, then copy `build/libs/*.jar`
   into the installed package's `MCP-Reborn/build/libs`.

## The traps (each one silently breaks the build)

- **(a) `@v1.0.0`, not `@master`.**
- **(b) `files.minecraftforge.net` is a dead 308-redirect** → a global
  `~/.gradle/init.gradle` rewrites it to `maven.minecraftforge.net`. Keep the
  rewrite SURGICAL (don't inject mavenCentral into buildscript — it shifts
  plugin versions).
- **(c) `pip install git+…` builds in ISOLATION** and ignores init.gradle →
  downloadAssets fails. Build MANUALLY in the order above.
- **(d) MIXIN MUST BE REMOVED** (the big one — minerl #702). v1.0.0 pins
  `mixingradle:0.7-SNAPSHOT`, incompatible with ForgeGradle-4.1's `universalJar`
  → gradle CONFIGURATION crashes at `MixinExtension.groovy:1038`. Comment out the
  3 mixin refs in `MCP-Reborn/build.gradle`; KEEP shadow.
- **(e) `unpack_assets`** — bypassing prep_mcp skips copying downloadAssets'
  hashed objects into resources → runtime `FileNotFoundException: icons/...png`.
  Unpack THEN rebuild the jar.
- **(f) old gym needs** `setuptools==65.5.1 wheel==0.38.4 pip==23.3.2`.
- **(g) install python-only via** `READTHEDOCS=1 pip install --no-build-isolation .`
  then the pre-built MCP-Reborn tree is bundled. Do NOT run python from
  cwd=mc-build (import resolves to source, not site-packages).

## The headless focus fix (why openbox + patches exist)

Under bare Xvfb (no window manager) no window gets X focus → MouseHelper's
camera-velocity accumulation (line ~259) and `grabMouse()` (line ~330) are gated
on `isMouseGrabbed() && isGameFocused()` → the synthetic mouse/attack are
**silently dropped** while keyboard flows. Result: agents can WALK but never TURN
or SWING (every zero-reward run before 2026-07-16 was this bug). Fixes:
- **openbox** running under `DISPLAY=:77` so windows get focus.
- DEVAI patch: MouseHelper lines 259+330 `if (true || …)` (marked
  `// DEVAI: headless focus patch`).
- `options.txt`: `tutorialStep:none`, `pauseOnLostFocus:false`.

## Env spec (`TreechopFixed` in `minerl_env.py`)

HumanSurvival-family spec (VPT-proven), Treechop forest generator, iron axe,
frozen daytime, SLIM observation (POV + inventory + `ObserveFromFullStats(
"mine_block")` only — the full stats dict is thousands of fields and crippled the
16-env run). Adapter-side reward = inventory log-count delta + tiered first-break.

## rsync gotcha

The network volume forbids chown → use `rsync -rlptz` (or `-rlpt`), **NOT `-az`**
(which exits 23 on the chown failure — files still copy but the non-zero rc is
alarming).
