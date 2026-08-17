# OpenCV CUDA side install — source this before python.
# Override OPENCV_CUDA_PREFIX for Docker (/opt/opencv-cuda) or custom installs.
#
# Platform notes:
#   Orin (R39) / Thor (R38) — same prefix layout, but OpenCV must be built
#   per GPU (Orin CC 8.7 vs Thor CC 11.0). See scripts/build_opencv_cuda.sh.
: "${OPENCV_CUDA_PREFIX:=${HOME}/.local/opencv-4.14.0-cuda}"

# Auto-pick newest side install if default prefix missing (e.g. first clone on Thor).
if [[ ! -d "${OPENCV_CUDA_PREFIX}/lib" && -d "${HOME}/.local" ]]; then
  _alt="$(find "${HOME}/.local" -maxdepth 1 -type d -name 'opencv-*-cuda' 2>/dev/null | sort -V | tail -1)"
  if [[ -n "${_alt}" && -d "${_alt}/lib" ]]; then
    OPENCV_CUDA_PREFIX="${_alt}"
  fi
  unset _alt
fi

# Prefer a detected python site-packages under the prefix (3.10 / 3.12 …).
# Ubuntu cmake installs to dist-packages; some builds use site-packages.
_py_site=""
if [[ -d "${OPENCV_CUDA_PREFIX}/lib" ]]; then
  for _d in \
      "${OPENCV_CUDA_PREFIX}"/lib/python3.*/dist-packages \
      "${OPENCV_CUDA_PREFIX}"/lib/python3.*/site-packages; do
    if [[ -d "${_d}/cv2" ]]; then
      _py_site="${_d}"
      break
    fi
  done
fi

export OPENCV_CUDA_PREFIX
export PATH="${OPENCV_CUDA_PREFIX}/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${OPENCV_CUDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -n "${_py_site}" ]]; then
  export PYTHONPATH="${_py_site}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PKG_CONFIG_PATH="${OPENCV_CUDA_PREFIX}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
export OpenCV_DIR="${OPENCV_CUDA_PREFIX}/lib/cmake/opencv4"
unset _d _py_site
