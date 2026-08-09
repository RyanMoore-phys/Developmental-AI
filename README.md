# Developmental AI

A curiosity-driven neurosymbolic RL agent that learns **Minecraft (MineRL
Treechop) with no videos/demos** — Learning-Progress curiosity + a
curiosity-ranked vision "magnet" + discovered goals → minted skills +
DreamerV3-style world model + PPO + local-VLM symbolic grounding, run as a
**lifelong continuous stream**.

## Repository layout

| Path | What it is |
|---|---|
| **`developmental_ai/`** | The code package (the organism). All imports resolve here. |
| **`configs/`** | Run configs. The live one is `configs/minecraft_skybot.yaml`. |
| **`scripts/`** | Pod scripts: `provision_pod.sh`, `launch_lifelong.sh` (MineRL), `launch_skybot.sh` + `connect_server.sh` (external Minecraft server over Tailscale), `mc_ping.py`, `rcon.py`. |
| **`run_minecraft.py`** | Main entry point (synced to the pod, launched there). |
| **`pod_repository/`** | 📕 **Ops reference + rescued brain backup.** Start at `pod_repository/README.md` for build/run/transfer docs; `pod_repository/data/` holds the 3.5 GB skill-bank backup. |
| **`tests/`** | All smoke/probe/unit tests (`_*_smoke.py`, probes, `test_components.py`). |
| **`experiments/`** | Historical rung/ablation/capstone work — `scripts/` (runners) + `results/` (outputs). Not part of the current Minecraft workflow. |
| **`docs/`** | Design docs, architecture diagrams (`arch*.mmd/png`, `FLOWMAPS.md`), historical audits. |
| **`viewer/`** | Live brain viewer web assets. |
| **`tools/`** | Dev utilities (`dream`, `visualize_world_model.py`, viewer launchers). |
| **`assets/`** | Screenshots / images. |
| **`archive/`** | Superseded scripts kept for reference. |
| `skill_bank_data/`, `skill_bank_doorkey/`, `logs/` | Local data dirs referenced by non-Minecraft configs — **do not move** (config-referenced). |
| `venv/` | Local Python 3.9 dev venv (the pod uses its own `venv_mc`). |
| `AUDIT_FINDINGS.md`, `ROADMAP.md`, `NEXT_OBJECTIVES.md` | Active top-level docs. |

## Running things

**Scripts are run from the repo root**, e.g. `python tests/_vision_magnet_smoke.py`.
The repo is made importable-from-any-cwd by a `.pth` file in the venv
(`venv/.../site-packages/devai_repo_root.pth` → the repo root), which is why
scripts in `tests/` and `experiments/` can `import developmental_ai` after being
moved out of the root. (This is the local dev equivalent of `pip install -e .`;
it only affects this Mac venv, not the pod.)

- **Run the smoke tests:** `python tests/<name>.py`
- **Train on the pod:** see `pod_repository/docs/RUNNING.md`
- **Set up / move a pod:** see `pod_repository/docs/PROVISIONING.md` and
  `pod_repository/docs/TRANSFER.md`
- **Current state & what's next:** `pod_repository/docs/STATE.md`
- **General (domain-independent) infrastructure programme:**
  `docs/GENERAL_INFRASTRUCTURE.md` — 57 proposed changes derived from measured
  failures, each with problem/solution/reasoning, aimed at making the core work
  in *any* environment (another game, a robot, a language stream) rather than
  only in Minecraft.

## The one rule

**Never wipe `skill_bank_mc_curiosity/`** (on the pod) — it's the agent's
accumulated developmental memory. Its verified backup is in
`pod_repository/data/`.
