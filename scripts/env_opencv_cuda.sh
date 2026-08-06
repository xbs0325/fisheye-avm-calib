# OpenCV CUDA side install — source this before python
export OPENCV_CUDA_PREFIX="/home/seeed/.local/opencv-4.14.0-cuda"
export PATH="/home/seeed/.local/opencv-4.14.0-cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/home/seeed/.local/opencv-4.14.0-cuda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/seeed/.local/opencv-4.14.0-cuda/lib/python3.12/site-packages:${PYTHONPATH:-}"
export PKG_CONFIG_PATH="/home/seeed/.local/opencv-4.14.0-cuda/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export OpenCV_DIR="/home/seeed/.local/opencv-4.14.0-cuda/lib/cmake/opencv4"
