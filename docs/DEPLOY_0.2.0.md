# 0.2.0 Docker 摘录

完整创作史与踩坑见 **[`PROJECT_HISTORY.md`](PROJECT_HISTORY.md)**（从建仓到 0.2.0）。

Docker 命令与设备挂载见 **[`DOCKER.md`](DOCKER.md)**。

本文件仅保留 0.2.0 开箱验收要点：

- 镜像：`leucushc/avm-gpu:0.2.0`（`ubuntu:24.04` + stage 的 CUDA OpenCV）
- 自检：`import cv2` → `4.14.0 1`
- 运行：`docker compose up -d` 或 `./scripts/docker_run_web.sh`
- CSI：必须有 `media0` + `capture-vi-channel*`，否则 open 成功、read 超时
- 冒烟：`/api/health`、`/api/cameras/probe`、`/api/stream/smoke` 四路 OK
