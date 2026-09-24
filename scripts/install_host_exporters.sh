#!/usr/bin/env bash
# Prometheus exporters for the TRAINING HOST (`main`), so it appears in Grafana
# next to node1/node2/paper-server instead of being the one machine nobody
# watches.
#
# WHY THIS EXISTS (2026-09-24). The monitor stack on node1 already runs
# Prometheus + node-exporter + Grafana, and `prometheus.yml` scrapes three
# targets — node1, node2, paper-server. `main` was never one of them, and the
# node-exporter container is configured with `hostname: node1` and
# `/:/host:ro`, so it reports NODE1's CPU/RAM/disk. The `gpu-metrics` container
# reads `/dev/dri` with intel_gpu_top — node1's INTEGRATED GPU, not main's
# RTX 5050. So nothing has ever measured the box that does the training.
#
# That gap cost a full day: a VLM holding 54% of RAM, four OOM-kills, swap
# pinned at 100%, and a world model filling an 8 GB card were all diagnosed by
# hand over ssh, one `status` run at a time, because no graph existed.
#
# BINARIES + systemd, NOT DOCKER. `main` runs MineRL clients, torch and a
# Minecraft world on 15.3 GB; it should not also carry a Docker daemon to
# report its own vitals. This matches how ollama and tailscale were installed
# there. Two static Go binaries, ~30 MB total, a few MB of RSS each.
#
# nvidia_gpu_exporter AND NOT NVIDIA DCGM: DCGM targets datacentre cards and is
# unreliable on consumer GPUs. This one shells out to nvidia-smi, which is
# already proven working on this 5050 (provision_host.sh STAGE 2 runs a real
# matmul through it).
#
# Usage, on main:
#   bash scripts/install_host_exporters.sh
#   BIND_ADDR=0.0.0.0 bash scripts/install_host_exporters.sh   # all interfaces
set -euo pipefail

NODE_VER="${NODE_VER:-1.12.1}"
GPU_VER="${GPU_VER:-1.15.1}"
# LAN-BOUND BY DEFAULT. These ports expose host vitals unauthenticated, so they
# should not sit on every interface by default. The monitor lives on the same
# LAN, so this is the shortest path and the smallest exposure. Override with
# BIND_ADDR if you move the monitor off-LAN (a tailnet 100.x address works).
BIND_ADDR="${BIND_ADDR:-$(ip -4 -o addr show scope global \
  | awk '{print $4}' | cut -d/ -f1 | grep -E '^192\.168\.' | head -1)}"
[ -n "$BIND_ADDR" ] || { echo "FATAL: no LAN address found; set BIND_ADDR="; exit 1; }

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else
  SUDO="sudo"
  sudo -n true 2>/dev/null || { echo "FATAL: needs passwordless sudo"; exit 1; }
fi

echo "==> installing exporters, bound to ${BIND_ADDR}"

install_one() {          # name  url  binary-path-in-tarball  port  extra-args
  local name="$1" url="$2" binpath="$3" port="$4"; shift 4
  if [ -x "/usr/local/bin/${name}" ] && systemctl is-active --quiet "$name"; then
    echo "    ${name}: already running"; return 0
  fi
  local tmp; tmp=$(mktemp -d)
  echo "    ${name}: downloading"
  # -L for the CDN redirect, and verify it is really a gzip before unpacking:
  # a 200 with an HTML error page would otherwise fail deep inside tar.
  curl -fsSL --max-time 180 -o "$tmp/a.tgz" "$url"
  head -c2 "$tmp/a.tgz" | od -An -tx1 | grep -q "1f 8b" \
    || { echo "    FAILED: not a gzip (got an error page?)"; exit 1; }
  tar -xzf "$tmp/a.tgz" -C "$tmp"
  $SUDO install -m 0755 "$tmp/$binpath" "/usr/local/bin/${name}"
  rm -rf "$tmp"

  $SUDO tee "/etc/systemd/system/${name}.service" >/dev/null <<EOF
[Unit]
Description=${name} (Prometheus exporter for main)
After=network-online.target
Wants=network-online.target

[Service]
# DynamicUser gives this an isolated unprivileged account with no home and no
# shell. node_exporter needs nothing privileged: /proc and /sys are world
# readable, and nvidia-smi is callable by any user.
DynamicUser=yes
ExecStart=/usr/local/bin/${name} --web.listen-address=${BIND_ADDR}:${port} $*
Restart=always
RestartSec=5
# The exporters are reporting tools, not workloads. Cap them so a runaway one
# can never compete with training for the CPU it is supposed to be measuring.
CPUQuota=10%
MemoryMax=128M

[Install]
WantedBy=multi-user.target
EOF
  $SUDO systemctl daemon-reload
  $SUDO systemctl enable --now "${name}"
  echo "    ${name}: listening on ${BIND_ADDR}:${port}"
}

install_one node_exporter \
  "https://github.com/prometheus/node_exporter/releases/download/v${NODE_VER}/node_exporter-${NODE_VER}.linux-amd64.tar.gz" \
  "node_exporter-${NODE_VER}.linux-amd64/node_exporter" 9100

if command -v nvidia-smi >/dev/null 2>&1; then
  install_one nvidia_gpu_exporter \
    "https://github.com/utkuozdemir/nvidia_gpu_exporter/releases/download/v${GPU_VER}/nvidia_gpu_exporter_${GPU_VER}_linux_x86_64.tar.gz" \
    "nvidia_gpu_exporter" 9835
else
  echo "    nvidia_gpu_exporter: SKIPPED (no nvidia-smi on this host)"
fi

echo "==> verifying (a unit that starts and then dies still reports enabled)"
sleep 2
for n in node_exporter nvidia_gpu_exporter; do
  systemctl list-unit-files | grep -q "^${n}.service" || continue
  if systemctl is-active --quiet "$n"; then
    p=$([ "$n" = node_exporter ] && echo 9100 || echo 9835)
    if curl -fsS --max-time 5 "http://${BIND_ADDR}:${p}/metrics" \
         | head -1 | grep -q .; then
      echo "    ${n}: OK, serving metrics on ${BIND_ADDR}:${p}"
    else
      echo "    ${n}: RUNNING BUT NOT SERVING — check: journalctl -u ${n} -n30"
    fi
  else
    echo "    ${n}: NOT ACTIVE — check: journalctl -u ${n} -n30"
  fi
done

echo
echo "==> add these to prometheus.yml on node1, then restart prometheus:"
echo "      - ${BIND_ADDR}:9100      # main (training host)"
echo "    and a new job for the GPU:"
echo "      - job_name: gpu"
echo "        static_configs:"
echo "          - targets: [\"${BIND_ADDR}:9835\"]   # main RTX 5050"
