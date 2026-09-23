#!/bin/bash
# HOST telemetry for the training box -> /data/host.jsonl -> SQLite -> Grafana.
#
# GPU **and** disk/RAM, in ONE sample. They are one observation of one machine
# at one instant, and the two failures that end a run silently on THIS box are
# not GPU failures: a full 256 GB NVMe (brain state lives only there, nothing
# is backed up) and an OOM on 16 GB (CLAUDE.md: real RSS exceeds hard_max_gb
# ALWAYS -- a '2 GB' ceiling measured 3.70 GB). Both are one ssh away already.
#
# WHY THIS EXISTS SEPARATELY FROM THE HEARTBEAT.
#   The agent already emits `gpu_mem_mb` every ~15s, but that is
#   `torch.cuda.memory_allocated()` — the TRAINING PROCESS'S OWN ALLOCATOR.
#   It cannot see:
#     * total device VRAM in use (so it cannot see OLLAMA at all),
#     * GPU utilisation, temperature or power,
#     * the per-process split.
#   On the RunPod box that gap was tolerable: 16 GB of VRAM, nothing else on
#   the card. On `main` it is the central resource question — 8 GB shared
#   between the world model and qwen2.5vl:7b — and CLAUDE.md already records
#   `fovea_interval: 12` costing 28% of the step rate. Without a device-level
#   trace next to the step rate, that trade stays an inference.
#
# WHY IT POLLS FROM HERE INSTEAD OF BEING EMITTED BY THE AGENT.
#   `_emit_heartbeat` documents the rule it is obeying: "no nvidia-smi (a
#   subprocess per heartbeat would be absurd)". That judgement is correct and
#   this must not quietly reverse it. Polling from the monitor host puts the
#   subprocess on a 15s timer OUTSIDE the step loop, where its cost is nobody's
#   hot path. Same reasoning as server_poll.py: an independent observer.
#
# WHY ONE SSH PER POLL, NOT TWO.
#   Device stats and per-process stats are two nvidia-smi queries. Issuing two
#   ssh connections per tick triples the cost for no benefit, so the remote
#   side prints both with a marker and the parsing happens here.
set -uo pipefail

: "${MAIN_HOST:?set MAIN_HOST}"
: "${MAIN_SSH_PORT:?set MAIN_SSH_PORT}"
KEY="${MAIN_SSH_KEYFILE:-/root/.ssh/skybot_ed25519}"
USER_AT="${MAIN_USER:-root}"
OUT="${HOST_PATH:-/data/host.jsonl}"
EVERY="${HOST_POLL_SECONDS:-15}"

SSH_OPTS=(-p "$MAIN_SSH_PORT" -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20
          -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

echo "host_poll: ${USER_AT}@${MAIN_HOST}:${MAIN_SSH_PORT} every ${EVERY}s -> ${OUT}"

REMOTE='
  echo "#GPU"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits 2>/dev/null | head -1
  echo "#APPS"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null
  echo "#DISK"
  # /workspace, not /: that is where the buffer, the MineRL build and ALL brain
  # state live. On this box they happen to share a filesystem, but df on the
  # actual path stays correct if a data disk is ever mounted there.
  df -BM --output=used,size,pcent /workspace/devai 2>/dev/null | tail -1 | tr -d "M%"
  echo "#MEM"
  free -m | awk "/^Mem:/{print \$2\" \"\$3\" \"\$7} /^Swap:/{print \$2\" \"\$3}"
'

while true; do
  RAW=$(ssh "${SSH_OPTS[@]}" "${USER_AT}@${MAIN_HOST}" "$REMOTE" 2>/dev/null)
  if [ -n "$RAW" ]; then
    # awk does the whole parse+emit: one pass, no temp files, and the JSON is
    # built with printf so a missing field becomes null rather than the string
    # "null" or an empty token that would break json.loads on the ingest side.
    printf '%s\n' "$RAW" | awk -v ts="$(date +%s.%N)" '
      /^#GPU/  { sec="gpu";  next }
      /^#APPS/ { sec="apps"; next }
      sec=="gpu" && NF {
          gsub(/,/," "); split($0,a," ");
          used=a[1]; tot=a[2]; util=a[3]; temp=a[4]; pw=a[5]; got=1
      }
      /^#DISK/ { sec="disk"; next }
      /^#MEM/  { sec="mem";  next }
      sec=="disk" && NF { du=$1; dt=$2; dp=$3 }
      sec=="mem" && NF {
          # free -m prints Mem: then Swap:; first line seen is Mem.
          if (!memseen) { mt=$1; mu=$2; ma=$3; memseen=1 } else { st=$1; su=$2 }
      }
      sec=="apps" && NF {
          line=$0; gsub(/,/," ",line); split(line,b," ");
          nm=tolower(b[2]); mb=b[3]+0;
          # CLASSIFY BY PROCESS NAME, and keep an explicit `other` bucket:
          # a silent drop would make the buckets stop summing to memory.used
          # and nobody would know which consumer was unaccounted for.
          if (index(nm,"ollama")) ollama+=mb;
          else if (index(nm,"python")) train+=mb;
          else other+=mb;
      }
      END {
        if (!got) exit 0;
        # vram_*, NOT mem_*: system RAM below uses mem_*, and emitting both
        # under one name produced DUPLICATE JSON KEYS -- json.loads keeps the
        # LAST, so VRAM was silently overwritten by system RAM and the card
        # appeared to have 15.8 GB. Caught in test, never shipped.
        printf "{\"wall_time\":%s,\"vram_used_mb\":%s,\"vram_total_mb\":%s,", ts, used, tot;
        printf "\"util_pct\":%s,\"temp_c\":%s,\"power_w\":%s,", util, temp, pw;
        printf "\"ollama_mb\":%d,\"train_mb\":%d,\"other_mb\":%d,", ollama, train, other;
        printf "\"disk_used_mb\":%d,\"disk_total_mb\":%d,\"disk_pct\":%d,", du, dt, dp;
        printf "\"mem_total_mb\":%d,\"mem_used_mb\":%d,\"mem_avail_mb\":%d,", mt, mu, ma;
        printf "\"swap_total_mb\":%d,\"swap_used_mb\":%d}\n", st, su;
      }' >> "$OUT"
  else
    echo "host_poll: no reply from ${MAIN_HOST} (skipping tick)"
  fi
  sleep "$EVERY"
done
