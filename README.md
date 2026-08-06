# fisheye-avm-calib

Jetson 四路鱼眼环视标定与 GPU BEV 拼接（CUDA OpenCV + WebRTC 引导）。

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

## 说明

- remap / warp / blend 走 GPU；检测在后台线程，不堵推流。
- 无 CUDA 时 Web 默认拒绝开流（可用 `--allow-cpu`，不推荐）。
- 标定要点与踩坑见 `docs/CALIBRATION_LESSONS.md`。

## 目录

| 路径 | 说明 |
|------|------|
| `avm/web_server.py` / `gpu_hub.py` | GPU Web |
| `avm/live_bev.py` / `cuda_cv.py` | 本机 live + CUDA 封装 |
| `avm/wizard.py` | CLI 向导 |
| `config/` | 相机 / 棋盘 / placement |
| `calib_results/*.json` | 内参 / 外参结果 |
| `docs/` | 管线、难点、WORKLOG |
