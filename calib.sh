#!/usr/bin/env bash
# Start Web calibration (intrinsics / extrinsics / seam 2b).
# Stop the surround demo first — cameras cannot be shared.
#
#   ./calib.sh
#   Browser: http://<board-ip>:8787/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_opencv_cuda.sh"

HOST="${CALIB_HOST:-0.0.0.0}"
PORT="${CALIB_PORT:-8787}"

if ! "${DEMO_PYTHON:-/usr/bin/python3}" -c "import aiortc" >/dev/null 2>&1; then
  echo "[calib] missing aiortc (WebRTC). Reproduce with:" >&2
  echo "        ${ROOT}/scripts/install_web_deps.sh" >&2
  exit 1
fi

if pgrep -f "perception.run|run_perception.sh" >/dev/null 2>&1; then
  echo "[calib] 环视 Demo 还在占用相机。先在那个终端 Ctrl+C，再跑本脚本。" >&2
  exit 1
fi

echo "============================================================"
echo "  标定 Web  ·  补缝走步骤 2b"
echo "============================================================"
echo "  1. 浏览器打开下面地址（笔记本和板子同一网段）"
hostname -I 2>/dev/null | tr ' ' '\n' | awk -v p="${PORT}" '
  NF && $0 !~ /^127\./ && $0 !~ /^172\.17\./ {
    printf "     http://%s:%s/\n", $0, p
  }'
echo "     http://127.0.0.1:${PORT}/"
echo "  2. 内参/外参已有：右侧跳到「2b 接缝精修」"
echo "     顺序：front+left → front+right → back+left → back+right"
echo "     把棋盘放到两路重叠区，两路都绿框 READY 后会自动锁从路 H"
echo "  3. 四对完成后停 Demo 占用：Ctrl+C 本脚本，再 ./run.sh"
echo "============================================================"
echo

exec "${ROOT}/scripts/run_web.sh" --host "${HOST}" --port "${PORT}" "$@"
