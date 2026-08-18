# fisheye-avm-calib

**v0.3.0** — From-scratch path: Jetson four-camera fisheye calibration, GPU BEV stitch, occupancy, YOLO-World, and VLM.

This repository is the **implementation workbook** (cameras + chessboard → live BEV → perception). It is **not** the J601 website/demo package.

| Role | Repository |
|------|------------|
| **Implementation path (this repo)** | Bring up fisheye calib, GPU stitch, and perception from scratch |
| **J601 promotional demo (English, Thor-only)** | [j601-surround-demo](https://github.com/xbs0325/j601-surround-demo) |

**Docs:** [`docs/OVERVIEW.md`](docs/OVERVIEW.md) (scene + stack) · [`docs/RELEASE_v0.3.0.md`](docs/RELEASE_v0.3.0.md) (release notes)

Four fisheye cameras on a mobile chassis → surround stitch into a top-down BEV → occupancy for nav hints, YOLO-World grasp direction, VLM scene caption. **No chassis or arm commands** in this stage.

## Demo scene

The demo sits on a **chassis with a robot arm**: four fisheyes stitch into a 360° top-down view. **YOLO-World** coarsely localizes grasp targets in `base_link` `(x_m, y_m)` and gives the arm surround FOV; an **occupancy grid** is a 2D ground hint for avoidance and path planning (not a LiDAR map). This stage does **not** send chassis or arm control commands.

![Perception BEV: detection + occupancy](assets/perception_bev_grasp.png)

Left: stitched BEV + occupancy overlay + YOLO targets. Right: top-down occupancy map from the same grid (up = front; rings = distance). Example: `./scripts/run_perception.sh --mode grasp --target bottle`.

## Quick start

### NVIDIA Thor (j6015 / JetPack R38.4)

Reproducible board steps: [`docs/THOR.md`](docs/THOR.md). For the polished J601 demo README, use [j601-surround-demo](https://github.com/xbs0325/j601-surround-demo).

```bash
cd ~/fisheye-avm-calib
source scripts/env_opencv_cuda.sh

./scripts/install_web_deps.sh          # first time: aiortc (do not bare pip3 on Ubuntu 24.04)
./scripts/setup_perception_thor.sh     # first time: VLM venv + Qwen3-VL-2B
./scripts/download_perception_models.sh

./run.sh       # surround demo
./calib.sh     # calib / seam web  →  http://<board-ip>:8787/
```

Do **not** run the demo and calib web UI at the same time (cameras are exclusive).

### AGX Orin (J501) / generic

```bash
cd ~/fisheye-avm-calib   # or your clone path
source scripts/env_opencv_cuda.sh

# GPU web guide (skip steps + WebRTC)
./scripts/run_web.sh --host 0.0.0.0 --port 8787
# browser: http://<board-ip>:8787/

# or CLI
./scripts/run_wizard.sh
./scripts/run_wizard.sh --web

# after calib: detection + occupancy (chassis surround assist)
./scripts/run_perception.sh --mode grasp --target bottle
./scripts/run_perception.sh --vlm off --mode nav --range 2.5
```

- **Docker** (Orin / JP 7.2 only): [`docs/DOCKER.md`](docs/DOCKER.md)
- **Thor (R38.4)**: [`docs/THOR.md`](docs/THOR.md) — CUDA OpenCV must be built on the board
- **Perception contract + Thor checklist**: [`docs/PERCEPTION.md`](docs/PERCEPTION.md)

## Notes

- Capture resolution and devices: `config/camera_profile.json` (editable in the web UI); use the status-page Probe to verify streams.
- Remap / warp / blend run on GPU; detection runs in a background thread so streaming stays smooth.
- Without CUDA, the web UI refuses to start streams by default (`--allow-cpu` for debug only).
- Calibration tips and pitfalls: [`docs/CALIBRATION_LESSONS.md`](docs/CALIBRATION_LESSONS.md) (Chinese lab notes).

## Layout

| Path | Description |
|------|-------------|
| `avm/web_server.py` / `gpu_hub.py` | GPU web calib |
| `avm/camera_io.py` | Camera profile / open / probe |
| `avm/live_bev.py` / `cuda_cv.py` | Live BEV + CUDA helpers |
| `avm/wizard.py` | CLI wizard |
| `config/` | camera_profile, chessboard, placement |
| `calib_results/*.json` | Intrinsics / extrinsics (H) |
| `Dockerfile` / `docker-compose.yml` | JP 7.2 Docker image |
| `perception/` | BEV occupancy, YOLO-World, VLM |
| `docs/` | OVERVIEW, THOR, PERCEPTION, calibration lessons |

## Versions

| Version | Summary |
|---------|---------|
| 0.3.0 | Thor bring-up, surround demo UI, VLM English captions, occupancy tuned against floor grout |
| 0.2.1 | YOLO-World grasp localization + occupancy grid / 2D avoidance hint |
| 0.2.0 | Configurable resolution, Probe API, JP 7.2 Docker |
| 0.1.1 | Seam refine (2b), joint sync counting, extrinsic QC / 180° orientation |
| 0.1.0 | Initial GPU web guide, intrinsics/extrinsics, BEV stitch |
