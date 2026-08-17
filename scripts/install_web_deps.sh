#!/usr/bin/env bash
# Web calib deps on Ubuntu 24.04 (PEP 668): do not use plain `pip3 install`.
#
# Stitch / calib use system python3 + CUDA OpenCV (NumPy 1.x).
# Do NOT install these into ~/leucus/.venv-worldmm.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${DEMO_PYTHON:-/usr/bin/python3}"

if ! command -v "${PY}" >/dev/null 2>&1; then
  echo "need python3:  sudo apt-get install -y python3-pip python3-venv" >&2
  exit 1
fi

echo "[web-deps] ${PY}  (keep NumPy 1.x for CUDA OpenCV)"
"${PY}" -m pip install --user --break-system-packages \
  'numpy>=1.26,<2' \
  'aiortc>=1.9.0' \
  'av>=12.0.0'

"${PY}" - <<'PY'
import aiortc
import av
import numpy as np
print("ok  aiortc", getattr(aiortc, "__version__", "?"),
      "av", av.__version__, "numpy", np.__version__)
if int(np.__version__.split(".")[0]) >= 2:
    raise SystemExit("numpy 2.x breaks side-installed CUDA OpenCV — pin numpy<2")
PY

echo "[web-deps] done.  ./calib.sh"
