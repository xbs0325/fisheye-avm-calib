# fisheye-avm-calib

**v0.3.0** — From-scratch path: Jetson four-camera fisheye calibration, GPU BEV stitch, occupancy, YOLO-World, and VLM.

This repository is the **implementation workbook** (cameras + chessboard → live BEV → perception). It is **not** the J601 website/demo package.

**J601 promotional demo (English, Thor-only):** [j601-surround-demo](https://github.com/xbs0325/j601-surround-demo)

**Scene & stack:** [`docs/OVERVIEW.md`](docs/OVERVIEW.md) · **v0.3.0 notes:** [`docs/RELEASE_v0.3.0.md`](docs/RELEASE_v0.3.0.md)

四路鱼眼装在移动底盘上 → 环视拼接成俯视 BEV → 导航占用参考、机械臂目标方位、VLM 场景说明。不发控制指令。

## 本 Demo 场景

本 demo 放在**带机械臂的底盘**上：四路鱼眼拼成 360° 俯视，**YOLO-World** 粗定位待抓取目标在 `base_link` 下的位置，给机械臂提供环视 FOV；同时输出占用栅格，作为**避障辅助**和**路线规划参考**（地面 2D，不是激光雷达地图）。本阶段不发底盘 / 臂控制指令。

![Perception BEV：识别 + 占用](assets/perception_bev_grasp.png)

左：拼接 BEV + 占用叠层 + YOLO 目标；右：同一套栅格的俯视占用图（上=前，圈为距离）。运行：`./scripts/run_perception.sh --mode grasp --target bottle`。

## 快速开始

### NVIDIA Thor（j6015 / R38.4）— 可复现步骤见 `docs/THOR.md`

```bash
cd ~/fisheye-avm-calib
source scripts/env_opencv_cuda.sh

./scripts/install_web_deps.sh          # 首次：aiortc（Ubuntu 24.04 不要裸 pip3）
./scripts/setup_perception_thor.sh     # 首次：VLM venv + Qwen3-VL-2B
./scripts/download_perception_models.sh

./run.sh       # 环视 Demo
./calib.sh     # 标定 / 补缝 Web  →  http://<板子IP>:8787/
```

Demo 和标定不能同时开（抢相机）。

### AGX Orin（J501）/ 通用

```bash
cd ~/bev_demo/avm_gpu   # 或本仓库克隆目录
source scripts/env_opencv_cuda.sh

# GPU Web 引导（可跳步 + WebRTC）
./scripts/run_web.sh --host 0.0.0.0 --port 8787
# 浏览器: http://<板子IP>:8787/

# 或 CLI
./scripts/run_wizard.sh
./scripts/run_wizard.sh --web

# 标定后：识别 + 占用（机械臂底盘环视辅助）
./scripts/run_perception.sh --mode grasp --target bottle
./scripts/run_perception.sh --vlm off --mode nav --range 2.5
```

Docker（仅 Orin / JP 7.2）：见 `docs/DOCKER.md`。  
**NVIDIA Thor (R38.4)**：见 `docs/THOR.md`（需本机编译 CUDA OpenCV）。感知契约与 Thor 检查表：`docs/PERCEPTION.md`。

## 说明

- 采集分辨率/设备：`config/camera_profile.json`（Web 右侧可改）；状态页 Probe 可验开流。
- remap / warp / blend 走 GPU；检测在后台线程，不堵推流。
- 无 CUDA 时 Web 默认拒绝开流（可用 `--allow-cpu`，不推荐）。
- 标定要点与踩坑见 `docs/CALIBRATION_LESSONS.md`。

## 目录

| 路径 | 说明 |
|------|------|
| `avm/web_server.py` / `gpu_hub.py` | GPU Web |
| `avm/camera_io.py` | 相机 profile / open / probe |
| `avm/live_bev.py` / `cuda_cv.py` | 本机 live + CUDA 封装 |
| `avm/wizard.py` | CLI 向导 |
| `config/` | camera_profile / 棋盘 / placement |
| `calib_results/*.json` | 内参 / 外参结果 |
| `Dockerfile` / `docker-compose.yml` | JP7.2 开箱镜像 |
| `perception/` | BEV + YOLO-World 识别 + 占用栅格 |
| `docs/` | OVERVIEW（场景/技术）、THOR、PERCEPTION、标定教训 |

## 版本

| 版本 | 说明 |
|------|------|
| 0.3.0 | Thor 复现、环视 Demo UI、VLM 英文 caption、占用抗地砖误检 |
| 0.2.1 | YOLO-World 抓取粗定位 + 占用栅格 / 2D 避障参考（机械臂底盘环视） |
| 0.2.0 | 可配置分辨率、Probe API、JP7.2 Docker |
| 0.1.1 | 接缝精修（2b）、联合同步计数、外参 QC/180° 定向、标定教训补全 |
| 0.1.0 | 初版：GPU Web 引导、内外参标定、BEV 拼接 |
