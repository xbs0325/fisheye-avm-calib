#!/usr/bin/env bash
# YOLO-World v2 → models/perception/yolov8s-worldv2.pt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${ROOT}/models/perception"
PT="${DST}/yolov8s-worldv2.pt"
URL="${YOLO_WORLD_URL:-https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt}"
mkdir -p "${DST}"

if [[ -f "${PT}" && -s "${PT}" ]]; then
  echo "[download] already have ${PT} ($(du -h "${PT}" | cut -f1))"
  exit 0
fi

echo "[download] YOLO-World v2 → ${PT}"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 3 --retry-delay 2 -o "${PT}" "${URL}"
else
  wget -O "${PT}" "${URL}"
fi
ls -lh "${PT}"
echo "[download] done"
