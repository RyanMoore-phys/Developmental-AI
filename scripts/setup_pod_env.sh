#!/usr/bin/env bash
# Bring a bare Ubuntu RunPod box to the point where the offline suites run,
# and report honestly on whether MineRL is usable.
#
# TWO TIERS ON PURPOSE. The offline suites need only numpy/torch/yaml and
# must work on any box. MineRL needs a JDK, a display and a long build, and
# is what Stage 0 then interrogates. Installing them together would make a
# JDK failure look like a test failure.
set -euo pipefail

echo "==> system"
uname -a; nproc; free -g | head -2; df -h / | tail -1
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no GPU visible"

PY="$(command -v python3)"
echo "==> python: ${PY} $(${PY} --version 2>&1)"

echo "==> core deps (offline suites)"
${PY} -m pip install --quiet --upgrade pip
${PY} -m pip install --quiet numpy pyyaml pillow || true
${PY} -c "import torch" 2>/dev/null || {
  echo "    installing torch (cu121)"
  ${PY} -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121 || \
  ${PY} -m pip install --quiet torch
}
${PY} - <<'PYCHK'
import torch, numpy, yaml
print(f"    torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"numpy {numpy.__version__}")
PYCHK

echo "==> MineRL (optional; Stage 0 decides if it is usable)"
if ${PY} -c "import minerl" 2>/dev/null; then
  ${PY} -c "import minerl; print('    minerl', getattr(minerl,'__version__','?'), 'already present')"
else
  echo "    installing JDK 8 + xvfb"
  (apt-get update -qq && apt-get install -y -qq openjdk-8-jdk xvfb libgl1-mesa-glx x11-xserver-utils >/dev/null 2>&1) || \
    echo "    WARN: apt failed; MineRL will not build"
  ${PY} -m pip install --quiet gymnasium || true
  ${PY} -m pip install --quiet git+https://github.com/minerllabs/minerl || \
    echo "    WARN: minerl install failed — Stage 0 will report NO-GO"
fi

echo "==> import check"
${PY} - <<'PYCHK'
import importlib
for m in ("torch", "numpy", "yaml", "gymnasium", "minerl"):
    try:
        importlib.import_module(m)
        print(f"    {m}: OK")
    except Exception as e:
        print(f"    {m}: MISSING ({type(e).__name__})")
PYCHK

echo "==> done. Offline suites should run now; MineRL is Stage 0's question."
