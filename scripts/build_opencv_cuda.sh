#!/usr/bin/env bash
# Build side-installed OpenCV 4.14 with CUDA for fisheye-avm-calib.
#
# Targets:
#   - NVIDIA Thor  (JetPack R38.x, CUDA 13, CC 11.0, sbsa-linux)
#   - AGX Orin     (JetPack R39.x, CUDA 13, CC 8.7,  tegra)
#
# Install prefix (default): ~/.local/opencv-4.14.0-cuda
# After build: source scripts/env_opencv_cuda.sh
#
# Usage:
#   ./scripts/build_opencv_cuda.sh              # full build + install
#   ./scripts/build_opencv_cuda.sh --jobs 8      # parallel compile
#   OPENCV_CUDA_PREFIX=/opt/opencv-cuda ./scripts/build_opencv_cuda.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCV_VER="${OPENCV_VER:-4.14.0}"
: "${OPENCV_CUDA_PREFIX:=${HOME}/.local/opencv-${OPENCV_VER}-cuda}"
BUILD_DIR="${BUILD_DIR:-${HOME}/src/opencv-${OPENCV_VER}-cuda-build}"
SRC_DIR="${SRC_DIR:-${HOME}/src}"
JOBS="${JOBS:-$(nproc)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --prefix) OPENCV_CUDA_PREFIX="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

export PATH="/usr/local/cuda/bin:${PATH}"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"

if [[ ! -x "${CUDACXX}" ]]; then
  echo "ERROR: nvcc not found at ${CUDACXX}" >&2
  exit 1
fi

# Detect platform from tegra release + GPU name.
CUDA_ARCH_BIN="${CUDA_ARCH_BIN:-}"
if [[ -z "${CUDA_ARCH_BIN}" ]]; then
  if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi thor; then
    CUDA_ARCH_BIN="11.0"
    echo "[build] detected NVIDIA Thor → CUDA_ARCH_BIN=${CUDA_ARCH_BIN}"
  elif [[ -f /etc/nv_tegra_release ]] && grep -q 'R39' /etc/nv_tegra_release; then
    CUDA_ARCH_BIN="8.7"
    echo "[build] detected Jetson Orin (R39) → CUDA_ARCH_BIN=${CUDA_ARCH_BIN}"
  elif [[ -f /etc/nv_tegra_release ]] && grep -q 'R38' /etc/nv_tegra_release; then
    CUDA_ARCH_BIN="11.0"
    echo "[build] detected Jetson Thor (R38) → CUDA_ARCH_BIN=${CUDA_ARCH_BIN}"
  else
    CUDA_ARCH_BIN="11.0"
    echo "[build] fallback CUDA_ARCH_BIN=${CUDA_ARCH_BIN}"
  fi
fi

echo "[build] OpenCV ${OPENCV_VER} → ${OPENCV_CUDA_PREFIX}"
echo "[build] build dir: ${BUILD_DIR}"
echo "[build] jobs: ${JOBS}"

need_sudo=0
if ! dpkg -s cmake >/dev/null 2>&1; then need_sudo=1; fi
if ! dpkg -s python3-dev >/dev/null 2>&1; then need_sudo=1; fi

DEPS=(
  build-essential cmake git pkg-config
  python3-dev python3-numpy python3-pip
  libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev
  libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
  libtbb-dev libeigen3-dev
)

if [[ "${need_sudo}" -eq 1 ]]; then
  echo "[build] installing apt build dependencies (sudo)..."
  if sudo -n true 2>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y "${DEPS[@]}"
  else
    echo "[build] WARN: sudo unavailable — install deps manually:" >&2
    printf '  sudo apt-get install -y %s\n' "${DEPS[*]}" >&2
  fi
else
  missing=()
  for p in "${DEPS[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done
  if ((${#missing[@]})); then
    echo "[build] missing packages: ${missing[*]}"
    if sudo -n true 2>/dev/null; then
      sudo apt-get update -qq
      sudo apt-get install -y "${missing[@]}"
    else
      echo "[build] WARN: sudo unavailable — continuing; cmake may fail." >&2
    fi
  fi
fi

WITH_GTK=ON
if ! dpkg -s libgtk-3-dev >/dev/null 2>&1; then
  WITH_GTK=OFF
  echo "[build] libgtk-3-dev missing → WITH_GTK=OFF (Web/BEV OK; local cv2.imshow needs GTK)"
fi

mkdir -p "${SRC_DIR}" "${BUILD_DIR}"
cd "${SRC_DIR}"

if [[ ! -d opencv-${OPENCV_VER} ]]; then
  echo "[build] cloning opencv ${OPENCV_VER}..."
  git clone --depth 1 --branch "${OPENCV_VER}" https://github.com/opencv/opencv.git "opencv-${OPENCV_VER}"
fi
if [[ ! -d opencv_contrib-${OPENCV_VER} ]]; then
  echo "[build] cloning opencv_contrib ${OPENCV_VER}..."
  git clone --depth 1 --branch "${OPENCV_VER}" https://github.com/opencv/opencv_contrib.git "opencv_contrib-${OPENCV_VER}"
fi

PY3="$(command -v python3)"
NUMPY_INC="$("${PY3}" -c 'import numpy; print(numpy.get_include())' 2>/dev/null || true)"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

CMAKE_ARGS=(
  -D CMAKE_BUILD_TYPE=Release
  -D CMAKE_INSTALL_PREFIX="${OPENCV_CUDA_PREFIX}"
  -D OPENCV_EXTRA_MODULES_PATH="${SRC_DIR}/opencv_contrib-${OPENCV_VER}/modules"
  -D OPENCV_GENERATE_PKGCONFIG=ON
  -D WITH_CUDA=ON
  -D WITH_CUDNN=ON
  -D OPENCV_DNN_CUDA=ON
  -D ENABLE_FAST_MATH=ON
  -D CUDA_FAST_MATH=ON
  -D WITH_CUBLAS=ON
  -D CUDA_ARCH_BIN="${CUDA_ARCH_BIN}"
  -D WITH_TBB=ON
  -D WITH_V4L=ON
  -D WITH_GSTREAMER=ON
  -D WITH_FFMPEG=ON
  -D WITH_GTK="${WITH_GTK}"
  -D WITH_OPENMP=ON
  -D BUILD_opencv_python3=ON
  -D PYTHON3_EXECUTABLE="${PY3}"
  -D BUILD_EXAMPLES=OFF
  -D BUILD_TESTS=OFF
  -D BUILD_PERF_TESTS=OFF
  -D BUILD_DOCS=OFF
  -D INSTALL_PYTHON_EXAMPLES=OFF
  -D INSTALL_C_EXAMPLES=OFF
)

if [[ -n "${NUMPY_INC}" ]]; then
  CMAKE_ARGS+=(-D PYTHON3_INCLUDE_DIR="$("${PY3}" -c 'import sysconfig; print(sysconfig.get_path("include"))')")
  CMAKE_ARGS+=(-D PYTHON3_NUMPY_INCLUDE_DIRS="${NUMPY_INC}")
fi

# Prefer sbsa CUDA libs on Thor; fall back to lib64 symlink.
for cuda_lib in /usr/local/cuda/targets/sbsa-linux/lib /usr/local/cuda/lib64; do
  if [[ -d "${cuda_lib}" ]]; then
    CMAKE_ARGS+=(-D CUDA_LIB_DIR="${cuda_lib}")
    break
  fi
done

echo "[build] cmake configure..."
cmake "${CMAKE_ARGS[@]}" "${SRC_DIR}/opencv-${OPENCV_VER}"

echo "[build] compiling (this may take 30–90 min on Thor)..."
cmake --build . --parallel "${JOBS}"

echo "[build] installing to ${OPENCV_CUDA_PREFIX}..."
cmake --install .

# Relocate cv2 loader (same fix as stage_opencv_for_docker.sh).
SITE="$(find "${OPENCV_CUDA_PREFIX}/lib" \( -path '*/dist-packages/cv2' -o -path '*/site-packages/cv2' \) -type d 2>/dev/null | head -1)"
if [[ -n "${SITE}" ]]; then
  cat > "${SITE}/config.py" <<'PYCFG'
import os
BINARIES_PATHS = [
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
] + BINARIES_PATHS
PYCFG
  for cfg in "${SITE}"/config-3.*.py; do
    [[ -f "${cfg}" ]] || continue
    ver="${cfg#"${SITE}/config-"}"
    ver="${ver%.py}"
    cat > "${cfg}" <<PYEXT
import os
PYTHON_EXTENSIONS_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python-${ver}')
] + PYTHON_EXTENSIONS_PATHS
PYEXT
  done
  echo "[build] patched ${SITE}/config.py"
fi

echo
echo "[build] verify:"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_opencv_cuda.sh"
"${PY3}" - <<'PY'
import cv2
print("OpenCV", cv2.__version__, "from", cv2.__file__)
n = cv2.cuda.getCudaEnabledDeviceCount()
print("CUDA devices:", n)
assert n >= 1, "CUDA not enabled"
assert hasattr(cv2.cuda, "remap"), "missing cv2.cuda.remap"
assert hasattr(cv2.cuda, "warpPerspective"), "missing cv2.cuda.warpPerspective"
print("OK — cudawarping available")
PY

echo
echo "[build] done. Add to shell profile or run before python:"
echo "  source ${ROOT}/scripts/env_opencv_cuda.sh"
