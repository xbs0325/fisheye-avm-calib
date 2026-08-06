# Docker build helpers

Before `docker build` / `docker compose build`, stage the host CUDA OpenCV tree:

```bash
./scripts/stage_opencv_for_docker.sh
```

That populates `docker/opencv-cuda/` (gitignored). See [docs/DOCKER.md](../docs/DOCKER.md).
