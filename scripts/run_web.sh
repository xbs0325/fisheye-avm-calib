#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_opencv_cuda.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
mkdir -p "${ROOT}/output"
LOG_FILE="${ROOT}/output/web_server.log"
echo "[run_web] logging to ${LOG_FILE}"
# tee keeps console + file; do not use exec with pipe
python3 -m avm.web_server "$@" 2>&1 | tee -a "${LOG_FILE}"
