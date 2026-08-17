#!/usr/bin/env bash
# Thor: torch venv + Qwen3-VL-2B weights (no WorldMM source required).
#
# Perception stitch stays on system Python + CUDA OpenCV (NumPy 1.x).
# VLM / YOLO-World run in this venv (NumPy 2.x OK).
#
# Usage:
#   ./scripts/setup_perception_thor.sh
#   ./scripts/setup_perception_thor.sh --skip-model   # venv only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEUCUS_ROOT="${LEUCUS_ROOT:-${HOME}/leucus}"
VENV="${PERCEPTION_VENV:-${LEUCUS_ROOT}/.venv-worldmm}"
MODELS="${PERCEPTION_MODELS:-${LEUCUS_ROOT}/models/worldmm}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
MODEL_DIR="${MODELS}/Qwen3-VL-2B-Instruct"
SKIP_MODEL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-model) SKIP_MODEL=1; shift ;;
    --venv) VENV="$2"; shift 2 ;;
    --models) MODELS="$2"; MODEL_DIR="${MODELS}/Qwen3-VL-2B-Instruct"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "need python3-venv:  sudo apt-get install -y python3-venv python3-pip" >&2
  exit 1
fi

echo "[setup] venv → ${VENV}"
mkdir -p "$(dirname "${VENV}")"
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -U pip wheel

echo "[setup] torch (CUDA 13 / aarch64 SBSA)…"
# Official cu130 wheels first; Jetson AI Lab SBSA index as fallback.
if ! python -c "import torch" >/dev/null 2>&1; then
  if ! python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130; then
    echo "[setup] official cu130 failed, trying pypi.jetson-ai-lab.io/sbsa/cu130"
    python -m pip install torch torchvision \
      --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130
  fi
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    raise SystemExit("torch CUDA is False — check libcudss / LD_LIBRARY_PATH")
PY

echo "[setup] transformers / qwen-vl-utils / ultralytics…"
python -m pip install \
  "transformers>=4.57" \
  qwen-vl-utils \
  accelerate \
  pillow \
  ultralytics \
  openai-clip \
  ftfy

if [[ "${SKIP_MODEL}" -eq 1 ]]; then
  echo "[setup] skip model download"
else
  echo "[setup] Qwen3-VL-2B → ${MODEL_DIR}"
  mkdir -p "${MODELS}"
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${MODEL_ID}",
    local_dir="${MODEL_DIR}",
    local_dir_use_symlinks=False,
)
print("ok", "${MODEL_DIR}")
PY
fi

cat <<EOF

[setup] done.
  export PERCEPTION_VENV=${VENV}
  export PERCEPTION_MODELS=${MODELS}

Try (after CUDA OpenCV is ready):
  cd ${ROOT}
  source scripts/env_opencv_cuda.sh
  ./scripts/run_perception.sh --mode nav --range 2.5 --vlm qwen3vl-2b
  ./scripts/run_demo_bev_vlm.sh --vlm qwen3vl-2b

If import torch fails with libcudss.so.0:
  sudo apt-get install -y libcudss0-cuda-13
  export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/libcudss/13:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
EOF
