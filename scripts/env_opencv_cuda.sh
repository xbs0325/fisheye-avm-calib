# OpenCV CUDA side install — source this before python.
# Override OPENCV_CUDA_PREFIX for Docker (/opt/opencv-cuda) or custom installs.
: "${OPENCV_CUDA_PREFIX:=${HOME}/.local/opencv-4.14.0-cuda}"

# Prefer a detected python site-packages under the prefix (3.10 / 3.12 …).
_py_site=""
if [[ -d "${OPENCV_CUDA_PREFIX}/lib" ]]; then
  for _d in "${OPENCV_CUDA_PREFIX}"/lib/python3.*/site-packages; do
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
