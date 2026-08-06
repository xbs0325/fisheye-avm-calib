#!/usr/bin/env bash
# 一键进入标定向导（强制侧装 CUDA OpenCV）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_opencv_cuda.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
exec python3 -m avm.wizard "$@"
