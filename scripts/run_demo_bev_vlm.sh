#!/usr/bin/env bash
# 标定后 BEV 实时拼接 Demo + Qwen3-VL（默认 2B）
#
# NumPy conflict: CUDA OpenCV (~/.local/opencv-*-cuda) is built for NumPy 1.x;
# WorldMM venv often has NumPy 2.x. This script does NOT activate the venv for
# the stitch process. Instead:
#   - /usr/bin/python3 + env_opencv_cuda.sh  → BEV stitch (cv2.cuda)
#   - WORLDMM_VENV_PYTHON subprocess         → Qwen3-VL (see vlm_worker.py)
#
# 需要：
#   - calib_results 四路内参 + extrinsics.json
#   - CUDA OpenCV（scripts/env_opencv_cuda.sh）
#   - leucus WorldMM 模型（默认 ~/leucus/models/worldmm/Qwen3-VL-2B-Instruct）
#
# 用法：
#   ./scripts/run_demo_bev_vlm.sh
#   ./scripts/run_demo_bev_vlm.sh --vlm off          # 只拼 BEV
#   ./scripts/run_demo_bev_vlm.sh --caption-interval 20
#   DISPLAY=:0 ./scripts/run_demo_bev_vlm.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_opencv_cuda.sh"

WORLDMM_ROOT="${WORLDMM_ROOT:-${HOME}/leucus}"
VENV="${WORLDMM_VENV:-${WORLDMM_ROOT}/.venv-worldmm}"
export WORLDMM_MODELS="${WORLDMM_MODELS:-${WORLDMM_ROOT}/models/worldmm}"
export WORLDMM_SRC="${WORLDMM_SRC:-${WORLDMM_ROOT}/WorldMM/src}"
export WORLDMM_ATTN_IMPL="${WORLDMM_ATTN_IMPL:-sdpa}"
export WORLDMM_QWEN_DEVICE_MAP="${WORLDMM_QWEN_DEVICE_MAP:-cuda:0}"
export WORLDMM_DTYPE="${WORLDMM_DTYPE:-bfloat16}"
unset HF_ENDPOINT || true

# Prefer the same Python the CUDA OpenCV wheel was built for (3.12 on this Orin).
PY="${DEMO_PYTHON:-/usr/bin/python3}"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python3)"
fi

VENV_PY="${VENV}/bin/python"
if [[ -x "${VENV_PY}" ]]; then
  export WORLDMM_VENV_PYTHON="${WORLDMM_VENV_PYTHON:-${VENV_PY}}"
  echo "[demo] stitch: ${PY} + CUDA OpenCV (NumPy 1.x)"
  echo "[demo] VLM:    ${WORLDMM_VENV_PYTHON} (venv, NumPy 2.x ok)"
else
  echo "[demo] WARN: no ${VENV_PY}; VLM worker will fall back to ${PY}" >&2
  export WORLDMM_VENV_PYTHON="${WORLDMM_VENV_PYTHON:-${PY}}"
fi

# Stitch process: OpenCV site from env script only — do not put venv site-packages
# on PYTHONPATH (that would pull NumPy 2.x and break cv2).
export PYTHONPATH="${ROOT}${WORLDMM_SRC:+:${WORLDMM_SRC}}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"
echo "[demo] stop AVM web / other camera owners first if open fails"
if [[ -z "${DISPLAY:-}" ]]; then
  echo "[demo] DISPLAY unset → will use --no-window style fallback if GTK fails"
fi
# Allow local X when user set DISPLAY=:0 from SSH (may still need: xhost +local:)
exec "${PY}" -m demo_bev_vlm.run "$@"
