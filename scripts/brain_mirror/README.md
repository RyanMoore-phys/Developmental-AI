# brain_mirror — SkyBot's brain, copied to node2 every 5 minutes

The brain lives only on the training host's single NVMe (CLAUDE.md §6,
"Backups"). This pulls it to node2's hard drive. **node2 does all the work**: it
starts the copy, and it does the hashing, verification, rotation and pruning.
The training host only serves file reads, at `nice 19` / `ionice` idle priority.
Nothing is ever written to or deleted on the training host.

## What it copies (relative to `/workspace/devai`)

| path | written by | atomic? |
|---|---|---|
| `logs/checkpoints/{world_model,curiosity,policy,dream_actor,symbolic_decoder,glue_layer,symbolizer,familiarity,magnet}.pt`, `knowledge_graph.json`, `options_state.json` | `_save_checkpoint_locked` (`loop.log_dir: ./logs`) | **no** (written in place) |
| `logs/checkpoints/curiosity_visits.pkl`, `logs/checkpoints/progress_probes.pkl` (since 2026-10-05; absent on older checkpoints = skipped, not fatal) | `_save_checkpoint_locked` | yes (tmp + fsync + replace) |
| `runlogs/breaks_by_type.json` | `environment.break_memory_path` | yes |
| `runlogs/consequence_state.json`, `runlogs/anticipation_state.json` | `infra.log_dir` | yes |
| `skill_bank_mc_rssm/` (live `skill_bank.storage_dir`) | skill bank | registry yes; skill `.pt` no |
| `skill_bank_mc_curiosity/` (older bank, never wiped) | — | static |

Not copied: the replay buffer (`logs/checkpoints/replay_buffer/`, GBs, can be rebuilt),
logs, metrics, code. In-flight `*.tmp` files are skipped. If the config moves any
of these paths, set `BRAIN_PATHS` in the env file.

## How a run works

1. Takes a lock. A lock left behind by a crashed run never blocks later runs.
2. Pulls with `rsync --link-dest` into `snapshots/<UTC>.partial`. Unchanged files
   are hardlinks to the previous snapshot, so they take almost no disk space.
3. The checkpoint files are written in place, not atomically, so a save can land
   halfway through a copy. After the pull, a dry-run rsync looks for anything
   that changed. If something did, it re-pulls, up to 3 times. If files are
   still changing after that, the snapshot is kept and marked `"mixed": true`.
4. Checks every file: each `.pt` must be a valid zip (`zipfile.testzip`) and each
   `.json` must parse. `MANIFEST.json` gets the sha256 and size of every file.
   If a check fails, the snapshot is renamed `.failed` and `latest` stays where
   it was. If the same file fails with the same bytes 3 runs in a row, the
   problem is on the training host itself (not a save caught midway). The
   snapshot is then promoted with `"degraded": true` (exit code 15), so the
   mirror keeps going.
5. Renames `.partial` to its final name and atomically repoints `latest`.
6. Retention: keeps the last 12 snapshots, plus one per hour for 48 h and one
   per day for 14 days. It never deletes `latest`.

## Install on node2 (from the Mac)

```bash
# from the repo root on the Mac
scp -r scripts/brain_mirror <node2-user>@<node2-LAN-IP>:~/skybot-brain-mirror
ssh <node2-user>@<node2-LAN-IP>
cd ~/skybot-brain-mirror
cp brain_mirror.env.example brain_mirror.env   # then edit: MAIN_HOST, MAIN_USER, BRAIN_DEST
ssh-keygen -t ed25519 -N '' -f ~/.ssh/skybot_brain_mirror_ed25519
ssh-copy-id -i ~/.ssh/skybot_brain_mirror_ed25519.pub <MAIN_USER>@<training-host-LAN-IP>
./install_node2.sh            # user systemd timer; --system for a system unit
```

`brain_mirror.env` is gitignored. Real addresses go **only** in that file on
node2. The script will not run while any value is still a `<placeholder>`. Needs
`rsync`, `ssh` and Python 3.8 or newer on node2. Needs `rsync` and `ionice`
(util-linux) on the training host. `MAIN_USER` must be able to read everything
under `/workspace/devai`.

## Check it

```bash
python3 brain_mirror.py --status        # latest age, counts, disk use; exit 16 if stale
tail ~/…/skybot-brain/mirror.log
systemctl --user list-timers skybot-brain-mirror.timer
```

| exit | meaning |
|---|---|
| 0 | snapshot taken |
| 3 | config error / placeholder left in the env file |
| 10 | another run holds the lock (normal if a run is slow) |
| 11 | training host unreachable, or `MAIN_ROOT` missing |
| 12 | verification failed; `latest` unchanged |
| 13 | not enough free space (`MIN_FREE_GB`); pull skipped |
| 14 | rsync failed for some other reason |
| 15 | promoted as degraded (a file on the training host is persistently corrupt) |

## Restore (never automated)

```bash
python3 brain_mirror.py --restore-plan latest     # or a snapshot name
```

This prints the commands but runs nothing:

1. Stop training with `runlogs/STOP` and wait for the process to exit. Never
   `kill` it; the supervisor would relaunch it.
2. Tar up the brain currently on the host, in case you need it back.
3. Dry-run the push.
4. Push with **no `--delete`**.
5. Remove `STOP`.

**The directory layout must match the config.** If break memory is restored to
the wrong path, it starts empty without any error, and every block tier the
agent had mastered pays full reward again. An old backup that still has
`podlogs/` needs `mv podlogs runlogs` first.

## Failure modes

- **Training host down or rebooting** → exit 11 each run, `latest` unchanged.
  It resumes on its own.
- **A save caught mid-copy** → re-pull. If it never settles, `mixed: true`. A
  truncated `.pt` → `.failed`, and the next run normally succeeds.
- **node2 disk filling up** → the oldest snapshots are pruned first. If space is
  still short, the pull is skipped with exit 13, and `latest` is always kept.
- **The training host loses its brain** (empty or reprovisioned) → nothing to
  copy means exit 12. A snapshot that shrinks by more than half logs a warning.
  Older snapshots stay on node2 under retention for 14 days. Copy one out
  before then if you need it.
- `.partial` and `.failed` directories are deleted after 1 day, and at most 10
  `.failed` are kept.
