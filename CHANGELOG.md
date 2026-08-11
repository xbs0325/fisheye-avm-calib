# Changelog

## 0.2.1

- Perception on stitched GPU BEV: YOLO-World open-vocab grasp localization + occupancy grid
- Intended scene: chassis with a robot arm — 360° FOV for the arm, obstacle / path reference (not LiDAR)
- Ego chassis overlay, 2D occupancy map beside BEV; optional YOLO-seg / VLM
- Docs: `docs/PERCEPTION.md`; screenshot `assets/perception_bev_grasp.png`

## 0.2.0

- Configurable capture via `config/camera_profile.json`; unified `avm/camera_io.py` open/probe
- Status: `/api/cameras/probe`, `/api/stream/smoke`; Web camera config; report bottom-left / log bottom-right
- JetPack 7.2 Docker on `ubuntu:24.04`: stage CUDA OpenCV + deps; CSI needs `media` + `capture-vi-channel*`
- Docs: `docs/PROJECT_HISTORY.md`（全史）、`docs/DOCKER.md`；image `leucushc/avm-gpu:0.2.0`

## 0.1.1

- Seam refine (step 2b), joint sync counting, extrinsic QC / 180° orient, calibration lessons

## 0.1.0

- Initial GPU Web guide, intrinsics/extrinsics, BEV stitch
