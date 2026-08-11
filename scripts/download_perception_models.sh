#!/usr/bin/env bash
# YOLO-World v2 → models/perception/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${ROOT}/models/perception"
VENV_PY="${PERCEPTION_VENV_PYTHON:-${HOME}/leucus/.venv-worldmm/bin/python}"
mkdir -p "${DST}"
unset HF_ENDPOINT || true
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg HF_HUB_DISABLE_XET=1
export PYTHONPATH=""

echo "[download] YOLO-World v2 → ${DST}/yolov8s-worldv2.pt"
"${VENV_PY}" - <<PY
from pathlib import Path
from ultralytics import YOLO
p = Path("${DST}/yolov8s-worldv2.pt")
YOLO(str(p) if p.is_file() else "yolov8s-worldv2.pt")
print("yolo-world", p.is_file(), p)
PY
echo "[download] done"
