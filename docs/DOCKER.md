# JetPack 7.2 Docker (Orin)

GPU Web AVM 容器（预编译 CUDA OpenCV）。**仅**适用于 Jetson + JetPack 7.2 / L4T R39.2（非 x86）。

完整踩坑与工作汇报见 [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md)。开箱命令摘录见 [`DEPLOY_0.2.0.md`](DEPLOY_0.2.0.md)。

## Prerequisites

- Jetson AGX Orin（或兼容）**JP 7.2**（`cat /etc/nv_tegra_release` → `R39 … REVISION: 2`）
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（`docker info` 可见 `nvidia` runtime）
- 四路相机：本仓库默认板为 **CSI / tegra-video**（见下文 Device 节），不是普通 UVC
- 构建机上需有 CUDA OpenCV 侧装（`scripts/env_opencv_cuda.sh`，常见 `~/.local/opencv-4.14.0-cuda`）

## Build

```bash
cd ~/bev_demo/avm_gpu
./scripts/stage_opencv_for_docker.sh    # → docker/opencv-cuda/（约 1.7GB，gitignore）
docker build -t leucushc/avm-gpu:0.2.0 .
# 或: docker compose build
```

**Base image:** `ubuntu:24.04`。JP7 **没有** `nvcr.io/nvidia/l4t-jetpack:r39.*`。

`--runtime nvidia` 只注入 **GPU 驱动**，不注入 CUDA toolkit（`libnppig` / `libcudart` …）。  
`stage_opencv_for_docker.sh` 会：

1. 把 `cv2/config*.py` 改成可重定位路径  
2. 用 `readelf NEEDED` 递归打入 FFmpeg 8 + CUDA 13 及传递依赖  

```bash
docker build --build-arg BASE_IMAGE=ubuntu:24.04 -t leucushc/avm-gpu:0.2.0 .
docker login
docker push leucushc/avm-gpu:0.2.0
```

自检：

```bash
docker run --rm --runtime nvidia --network host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  leucushc/avm-gpu:0.2.0 \
  bash -lc 'source scripts/env_opencv_cuda.sh; python3 -c "import cv2; print(cv2.__version__, cv2.cuda.getCudaEnabledDeviceCount())"'
# expect: 4.14.0 1
```

## Run

```bash
docker compose up -d
# 或自动挂齐 CSI 设备:
./scripts/docker_run_web.sh
# 浏览器: http://<orin-ip>:8787/
```

挂载：`config/`、`calib_results/`、`output/`；可选 `/usr/local/cuda`。

**不要**只挂 `/dev/video*` 就以为能取流（见下节）。

## Device mapping（CSI / tegra-video）

本机相机驱动为 `tegra-video`（`platform:tegra-capture-vi`）。

只挂 `/dev/video0`…`3` 时：OpenCV 能 **open**，**read** 会 `select() timeout`。  
还必须挂：

- `/dev/media0`
- `/dev/camsync`
- `/dev/capture-vi-channel0` …（常见 0–27）

`docker-compose.yml` 已列出；`scripts/docker_run_web.sh` 会按主机现存节点自动加 `--device`。

验收：

```bash
curl -s http://127.0.0.1:8787/api/health
curl -s -X POST http://127.0.0.1:8787/api/cameras/probe   # 四路 ok
curl -s -X POST http://127.0.0.1:8787/api/stream/smoke     # preview ok
```

相机被宿主机 `run_web.sh` 占用时先停宿主机进程。

## Resolution / calib

- 改 `camera_profile.json` 的宽高后，旧内参 `image_size` 可能失效（状态报告会提示）。
- `fourcc` / `gst_pipeline_template` 可按传感器调整；当前 CSI 走 V4L2 tegra 节点。

## First calibration in container

1. 挂好宿主机 `config/` + `calib_results/`
2. Probe → 内参 → 外参 →（可选）接缝 → BEV
3. 标定结果留在宿主机卷，重建镜像不会丢

## Limits (v0.2)

- 无 x86 仿真镜像
- 镜像体积大（打入 CUDA/FFmpeg 依赖）；换 OpenCV 需重新 stage
- OpenCV 须与 JP7.2 / CUDA 13 同源，勿混用 JP5/JP6 产物
