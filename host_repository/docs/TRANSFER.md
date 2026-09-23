# Transferring to a new pod / restoring the brain

RunPod pods are ephemeral; the **MooseFS `/workspace` network volume is
persistent**. If the new pod re-attaches the same volume, the brain
(`skill_bank_mc_curiosity/`) and the built jar are already there — you only
re-connect + re-sync code. If it's a fresh volume, you must re-provision AND
restore the brain from a backup like `../host_repository/data/`.

## Connecting to a restarted/new pod (the 3 gotchas that bit us)

1. **New port every restart.** Get it from the dashboard (Connect → SSH over
   exposed TCP). 22655 → 22681 → …
2. **Reused IP ⇒ changed host key.** Plain SSH refuses ("REMOTE HOST
   IDENTIFICATION HAS CHANGED") even with `StrictHostKeyChecking=no`. You MUST
   add `-o UserKnownHostsFile=/dev/null`.
3. **The key is `~/.ssh/skybot_ed25519`.** The dashboard prints
   `~/.ssh/id_ed25519`, which does NOT exist on this Mac. Also confirm the new
   pod authorized your public key (RunPod does it automatically when the key is
   on your account).

Verified working line:

```bash
ssh root@<IP> -p <PORT> -i ~/.ssh/skybot_ed25519 \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
```

## Is the brain already on the new pod?

```bash
$SSH 'cd /workspace/devai && du -sh skill_bank_mc_curiosity && \
  ./venv_mc/bin/python -c "import json;d=json.load(open(\"skill_bank_mc_curiosity/broadcaster_state.json\"));u=d.get(\"slot_uid\",{});print([v for v in (u.values() if isinstance(u,dict) else u) if \"log\" in str(v).lower()])"'
```

If it prints the 4 log goals (oak/jungle/spruce/birch), the brain survived — do
nothing. If the dir is missing, restore it (next section).

## Rescue the brain OFF a pod (what we did 2026-07-22)

```bash
# tiny index first (fast, most important metadata)
rsync -rlpt -e "ssh -p <PORT> -i ~/.ssh/skybot_ed25519 \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  root@<IP>:/workspace/devai/skill_bank_mc_curiosity/{broadcaster_state,registry}.json \
  DEST/

# then the full 3.5 GB bank (resumable; --partial)
rsync -rlpt --partial -e "ssh …same…" \
  root@<IP>:/workspace/devai/skill_bank_mc_curiosity/ \
  DEST/skill_bank_mc_curiosity/
```

Backup is at `../host_repository/data/skill_bank_mc_curiosity/` as of 2026-07-22.
The **3.5 GB is mostly per-skill `policy.pt` PPO snapshots** — they ARE the skill
weights, so they're vital, not bloat. There is NO separate policy/WM checkpoint
(`minecraft_lifelong_results/` had no `.pt`) — the skill bank is the durable state.

## Restore the brain ONTO a fresh pod

```bash
rsync -rlptz -e "ssh …" \
  ../host_repository/data/skill_bank_mc_curiosity/ \
  root@<IP>:/workspace/devai/skill_bank_mc_curiosity/
```

Then launch normally — the lifelong loop loads `broadcaster_state.json` +
`registry.json` on start and continues. **NEVER** delete the pod-side bank before
confirming the restore landed.

## Full new-pod checklist

1. Get IP + port; connect (the 3 gotchas above).
2. `du -sh /workspace/devai` — volume re-attached (has venv_mc + jar + bank) or fresh?
3. If fresh: run `scripts/provision_host.sh` (see `PROVISIONING.md`); restore the
   brain backup.
4. `rsync` current code from the Mac (`developmental_ai configs scripts run_minecraft.py`).
5. `bash scripts/launch_lifelong.sh 1000000`; verify boot + `h_evolve` + no errors.

## Also worth backing up (lower priority)

- `skill_bank_mc_curiosity_predup_archive_704550/` (1.7 GB) — a pre-dedup backup
  of the 16 restored skills; largely redundant with the active bank.
- `runlogs/*.log` — diagnostic history (rescued to `data/logs/`).
- The built jar (`…/MCP-Reborn/build/libs/mcprec-6.13.jar`) — re-buildable but
  saves the whole provisioning saga if you keep a copy.
