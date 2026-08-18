# Release v0.3.0

**fisheye-avm-calib** is the **from-scratch implementation path**: chessboard calibration, GPU BEV stitching, occupancy, YOLO-World, and VLM on Jetson. It is a working notebook for bringing the pipeline up from cameras and a chessboard—not a product landing-page demo.

Tag: `v0.3.0` · commit `03d1b67` (+ follow-up notes on `main` if present)

## What this release adds

- **NVIDIA Thor / reComputer J601 (JetPack R38.4)** as a first-class target: CUDA OpenCV 4.14 (compute 11.0), `run.sh` / `calib.sh`, `docs/THOR.md`.
- **Surround perception demo** on the stitched BEV:
  - occupancy grid (nav / free-space hint)
  - YOLO-World boxes → `base_link` `(x, y)` for grasp *direction*
  - **Qwen3-VL-2B** English captions (language only; coordinates stay with YOLO)
- Web calib: SB chessboard + photometric retry on the detect copy; homography geometry unchanged.
- Occupancy: less likely to mark hexagonal floor grout as occupied.
- Film UI: left = stitch + ego + YOLO + caption; right = occupancy map.

## Not in this repo’s job

- Official website / sales demo packaging (J601-only, English-first) belongs in a separate demo repository.
- Chassis velocity or arm joint commands are **not** issued.

## Run (after calib)

```bash
source scripts/env_opencv_cuda.sh
./run.sh      # perception window
./calib.sh    # http://<board-ip>:8787/
```

Do not run both at once (cameras are exclusive). Details: `docs/OVERVIEW.md`, `docs/THOR.md`, `docs/PERCEPTION.md`.
