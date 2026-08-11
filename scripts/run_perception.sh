#!/usr/bin/env bash
# BEV 感知 Demo（只动 perception；不改标定）
# 默认：占用栅格；grasp 默认 YOLO-World GPU FP16（VLM 关）
#
# 用法：
#   ./scripts/run_perception.sh --vlm off --mode nav --range 2.5
#   ./scripts/run_perception.sh --mode grasp --target bottle
#   ./scripts/run_perception.sh --mode nav --range 2.5 --vlm qwen3vl-2b
#   ./scripts/run_perception.sh --no-occ
#   ./scripts/run_perception.sh --seg   # 实验性，不推荐
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_opencv_cuda.sh"

LEUCUS_ROOT="${LEUCUS_ROOT:-${HOME}/leucus}"
VENV="${PERCEPTION_VENV:-${LEUCUS_ROOT}/.venv-worldmm}"
export PERCEPTION_MODELS="${PERCEPTION_MODELS:-${LEUCUS_ROOT}/models/worldmm}"
export PERCEPTION_ATTN_IMPL="${PERCEPTION_ATTN_IMPL:-sdpa}"
export PERCEPTION_DEVICE_MAP="${PERCEPTION_DEVICE_MAP:-cuda:0}"
export PERCEPTION_DTYPE="${PERCEPTION_DTYPE:-bfloat16}"
unset HF_ENDPOINT || true

PY="${DEMO_PYTHON:-/usr/bin/python3}"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python3)"
fi

VENV_PY="${VENV}/bin/python"
if [[ -x "${VENV_PY}" ]]; then
  export PERCEPTION_VENV_PYTHON="${PERCEPTION_VENV_PYTHON:-${VENV_PY}}"
  echo "[perception] stitch: ${PY} + CUDA OpenCV"
  echo "[perception] models: ${PERCEPTION_VENV_PYTHON}"
else
  echo "[perception] WARN: no ${VENV_PY}; workers fall back to ${PY}" >&2
  export PERCEPTION_VENV_PYTHON="${PERCEPTION_VENV_PYTHON:-${PY}}"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

cd "${ROOT}"
echo "[perception] stop AVM web / other camera owners first if open fails"
exec "${PY}" -m perception.run "$@"
