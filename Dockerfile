# JetPack 7.2 / L4T R39.2 — four-camera AVM Web (CUDA OpenCV)
#
# JP7 no longer publishes nvcr.io/nvidia/l4t-jetpack:r39.* — use Ubuntu 24.04
# and inject GPU/CUDA from the host via --runtime nvidia.
#
# Build (on Orin):
#   ./scripts/stage_opencv_for_docker.sh
#   docker build -t leucushc/avm-gpu:0.2.0 .
#
# Run: see docs/DOCKER.md

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    OPENCV_CUDA_PREFIX=/opt/opencv-cuda \
    PYTHONUNBUFFERED=1 \
    AVM_HOST=0.0.0.0 \
    AVM_PORT=8787 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Runtime deps for V4L2 + common multimedia (CUDA toolkit libs are staged).
# Host OpenCV on JP7.2 links NVIDIA ffmpeg 8 + CUDA 13 NPP — those .so are
# copied into /opt/opencv-cuda/lib by scripts/stage_opencv_for_docker.sh.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-numpy \
      v4l-utils \
      ca-certificates \
      libtbb12 \
      libjpeg-turbo8 \
      libpng16-16t64 \
      libtiff6 \
      libwebp7 \
      libwebpmux3 \
      libwebpdemux2 \
      libopenjp2-7 \
      libgtk-3-0t64 \
      libgstreamer1.0-0 \
      libgstreamer-plugins-base1.0-0 \
      libavc1394-0 \
      libx264-164 \
      libx265-199 \
      libvpx9 \
      libopus0 \
      libvorbis0a \
      libvorbisenc2 \
      libmp3lame0 \
      libdav1d7 \
      libaom3 \
      libsnappy1v5 \
      libzvbi0t64 \
      librsvg2-2 \
      libopenexr-3-1-30 \
    && rm -rf /var/lib/apt/lists/*

# Prebuilt CUDA OpenCV (staged by scripts/stage_opencv_for_docker.sh).
COPY docker/opencv-cuda/ /opt/opencv-cuda/

# Belt-and-suspenders: ensure cv2 configs are relocatable if stage was skipped.
RUN python3 - <<'PY'
from pathlib import Path
sites = list(Path("/opt/opencv-cuda").glob("lib/python3.*/site-packages/cv2"))
if not sites:
    raise SystemExit("missing staged OpenCV cv2 package")
cv2_dir = sites[0]
(cv2_dir / "config.py").write_text(
    "import os\n\n"
    "BINARIES_PATHS = [\n"
    "    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "
    "'..', '..', '..'))\n"
    "] + BINARIES_PATHS\n"
)
for p in sorted(cv2_dir.glob("config-3.*.py")):
    ver = p.name[len("config-") : -len(".py")]
    p.write_text(
        "import os\n\n"
        "PYTHON_EXTENSIONS_PATHS = [\n"
        f"    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python-{ver}')\n"
        "] + PYTHON_EXTENSIONS_PATHS\n"
    )
print("opencv config patched under", cv2_dir)
PY

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /app/requirements.txt \
    || pip3 install --no-cache-dir -r /app/requirements.txt

COPY avm/ /app/avm/
COPY config/ /app/config/
COPY scripts/ /app/scripts/
COPY VERSION /app/VERSION
COPY README.md /app/README.md

# Default calib dir (mount host calib_results over this in compose).
RUN mkdir -p /app/calib_results /app/output \
    && chmod +x /app/scripts/*.sh

ENV PATH=/opt/opencv-cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/opencv-cuda/lib:/usr/local/cuda/lib64 \
    PYTHONPATH=/app:/opt/opencv-cuda/lib/python3.12/site-packages \
    PKG_CONFIG_PATH=/opt/opencv-cuda/lib/pkgconfig \
    OpenCV_DIR=/opt/opencv-cuda/lib/cmake/opencv4

EXPOSE 8787

# Host networking + --device /dev/video* + --runtime nvidia required at run time.
CMD ["bash", "/app/scripts/run_web.sh", "--host", "0.0.0.0", "--port", "8787"]
