# fisheye-avm-calib

**v0.2.0** — Jetson 四路鱼眼环视标定与 GPU BEV 拼接（可配置分辨率 + JetPack 7.2 Docker）。

## 快速开始

```bash
cd ~/bev_demo/avm_gpu   # 或本仓库克隆目录
source scripts/env_opencv_cuda.sh

# GPU Web 引导（可跳步 + WebRTC）
./scripts/run_web.sh --host 0.0.0.0 --port 8787
# 浏览器: http://<板子IP>:8787/

# 或 CLI
./scripts/run_wizard.sh
./scripts/run_wizard.sh --web
```

Docker（仅 Orin / JP 7.2）：见 `docs/DOCKER.md`。

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
| `docs/` | 管线、难点、DOCKER、WORKLOG |

## 版本

| 版本 | 说明 |
|------|------|
| 0.2.0 | 可配置分辨率、Probe API、JP7.2 Docker |
| 0.1.1 | 接缝精修（2b）、联合同步计数、外参 QC/180° 定向、标定教训补全 |
| 0.1.0 | 初版：GPU Web 引导、内外参标定、BEV 拼接 |
