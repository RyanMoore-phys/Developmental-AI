#!/bin/sh
# Intel iGPU busy/frequency -> Prometheus textfile, for node1 only.
#
# WHY A TEXTFILE AND NOT AN EXPORTER WITH A PORT: node-exporter is already
# scraped on this host, and its --collector.textfile.directory picks up any
# *.prom file in a shared volume. One scrape target instead of two, and the
# metrics arrive with the same instance label as the rest of node1's.
#
# THE `mv` AT THE END IS LOAD-BEARING: writing straight to gpu.prom would let
# node-exporter read a half-written file and emit a parse error. Write to a
# PID-suffixed temp and rename — rename is atomic within a filesystem.
set -eu
OUT_DIR="${OUT_DIR:-/textfile}"
SAMPLE_MS="${SAMPLE_MS:-1000}"
while true; do
  raw=$(timeout --signal=INT 2 intel_gpu_top -J -s "$SAMPLE_MS" -o - 2>/dev/null || true)
  json=$(printf '[%s]' "$raw" | jq -c '.[-1] // {}' 2>/dev/null || echo '{}')
  render=$(echo "$json" | jq -r '[.engines | to_entries[]? | select(.key | test("^Render")) | .value.busy] | first // 0')
  video=$(echo "$json"  | jq -r '[.engines | to_entries[]? | select(.key | test("^Video/"))  | .value.busy] | first // 0')
  freq=$(echo "$json"   | jq -r '.frequency.actual // 0')
  tmp="$OUT_DIR/gpu.prom.$$"
  cat > "$tmp" <<PROM
# HELP node_intel_gpu_render_busy_percent Intel GPU render/3D engine busy percent
# TYPE node_intel_gpu_render_busy_percent gauge
node_intel_gpu_render_busy_percent $render
# HELP node_intel_gpu_video_busy_percent Intel GPU video engine busy percent
# TYPE node_intel_gpu_video_busy_percent gauge
node_intel_gpu_video_busy_percent $video
# HELP node_intel_gpu_frequency_mhz Intel GPU actual frequency, MHz
# TYPE node_intel_gpu_frequency_mhz gauge
node_intel_gpu_frequency_mhz $freq
PROM
  mv "$tmp" "$OUT_DIR/gpu.prom"
  sleep 3
done
