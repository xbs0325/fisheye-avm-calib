#!/usr/bin/env python3
"""
Live 4-camera fisheye undistortion preview for AVM system.

Opens all 4 cameras, applies calibrated intrinsic parameters from
calib_results/*.json, and displays live undistorted views in a 2x2 grid.

Usage:
    python3 scripts/preview_undistorted.py
    python3 scripts/preview_undistorted.py --balance 0.5
    python3 scripts/preview_undistorted.py --calib-dir calib_results/
"""

import argparse
import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import cv2

from cuda_cv import (
    UndistortWarpPipeline,
    init_undistort_maps,
    log_cuda_status,
    resize_bgr,
)

PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "config", "camera_config.json")
DEFAULT_CALIB_DIR = os.path.join(PROJECT_DIR, "calib_results")
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1536
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
DIRECTIONS = ["front", "back", "left", "right"]


def load_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    valid = {"left", "back", "front", "right"}
    for k in cfg:
        if k not in valid:
            raise ValueError(f"Unexpected direction '{k}' in config. Expected one of {valid}")
    return cfg


def open_camera(index, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT):
    device = f"/dev/video{index}"
    pipe = (
        f"v4l2src device={device} io-mode=2 do-timestamp=true ! "
        f"video/x-raw,format=YUY2,width={CAPTURE_WIDTH},height={CAPTURE_HEIGHT} ! "
        f"nvvidconv ! video/x-raw,format=BGRx,width={width},height={height} ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        backend = cv2.CAP_V4L2
        cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open /dev/video{index}. "
            f"Check permissions (try: sudo usermod -aG video $USER)"
        )
    if width == CAPTURE_WIDTH and height == CAPTURE_HEIGHT:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(5):
        cap.grab()
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if actual_w <= 0 or actual_h <= 0:
        actual_w, actual_h = width, height
    return cap, int(actual_w), int(actual_h)


def load_calib_results(direction, calib_dir):
    path = os.path.join(calib_dir, f"{direction}.json")
    with open(path, "r") as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float64)
    D = np.array(data["D"], dtype=np.float64)
    rms = data.get("rms", None)
    return K, D, rms


def precompute_undistort_maps(K, D, w, h, balance):
    return init_undistort_maps(K, D, w, h, balance, for_cuda=True)


def build_grid(frames, labels):
    top = cv2.hconcat([frames["front"], frames["back"]])
    bottom = cv2.hconcat([frames["left"], frames["right"]])
    grid = cv2.vconcat([top, bottom])

    positions = {
        "front": (10, 30),
        "back": (DISPLAY_WIDTH + 10, 30),
        "left": (10, DISPLAY_HEIGHT + 30),
        "right": (DISPLAY_WIDTH + 10, DISPLAY_HEIGHT + 30),
    }
    for direction, pos in positions.items():
        cv2.putText(grid, labels[direction], pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return grid


def main():
    parser = argparse.ArgumentParser(
        description="Live 4-camera fisheye undistortion preview"
    )
    parser.add_argument("--balance", type=float, default=0.5,
                        help="Undistortion balance: 0=crop (no black borders), "
                             "1=full sensor (all pixels). Default: 0.5")
    parser.add_argument("--calib-dir", default=DEFAULT_CALIB_DIR,
                        help=f"Directory with calibration JSON files "
                             f"(default: {DEFAULT_CALIB_DIR})")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Path to camera_config.json "
                             f"(default: {DEFAULT_CONFIG})")
    args = parser.parse_args()

    if not os.path.isdir(args.calib_dir):
        print(f"[ERROR] Calibration directory not found: {args.calib_dir}")
        sys.exit(1)

    config = load_config(args.config)

    print("=" * 60)
    print("  Loading calibration results...")
    print("=" * 60)

    calib = {}
    for direction in DIRECTIONS:
        try:
            K, D, rms = load_calib_results(direction, args.calib_dir)
            calib[direction] = {"K": K, "D": D, "rms": rms}
            print(f"  {direction:8s}  RMS={rms:.4f} px  "
                  f"fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  "
                  f"cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
        except FileNotFoundError:
            print(f"  [WARN] {direction}: no calibration file found, skipping")
        except Exception as e:
            print(f"  [ERROR] {direction}: {e}")

    if not calib:
        print("[ERROR] No calibration data loaded. Exiting.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Opening cameras...")
    print("=" * 60)

    caps = {}
    for direction in DIRECTIONS:
        if direction not in calib:
            continue
        idx = config.get(direction)
        if idx is None:
            print(f"  [SKIP] {direction}: not found in camera config")
            continue
        try:
            cap, w, h = open_camera(idx)
            caps[direction] = (cap, w, h, idx)
            print(f"  {direction}: /dev/video{idx} -> {w}x{h}")
        except Exception as e:
            print(f"  [ERROR] {direction} (/dev/video{idx}): {e}")

    if not caps:
        print("[ERROR] No cameras available. Exiting.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Precomputing undistortion maps (balance={:.2f})...".format(args.balance))
    print("=" * 60)
    log_cuda_status()

    pipes = {}
    for direction, (cap, w, h, idx) in caps.items():
        K = calib[direction]["K"]
        D = calib[direction]["D"]
        map1, map2 = precompute_undistort_maps(K, D, w, h, args.balance)
        pipes[direction] = UndistortWarpPipeline(map1, map2)
        print(f"  {direction}: map computed ({w}x{h}) "
              f"[{'CUDA' if pipes[direction].use_cuda else 'CPU'}]")

    print("\n  Press ESC or q to exit.\n")

    labels = {}
    for direction in caps:
        _, _, _, idx = caps[direction]
        labels[direction] = f"{direction.upper()}  /dev/video{idx}"

    while True:
        raw_frames = {}
        for direction in caps:
            caps[direction][0].grab()
        for direction in caps:
            cap = caps[direction][0]
            ret, frame = cap.retrieve()
            if not ret:
                continue
            raw_frames[direction] = frame

        display_frames = {}
        for direction in DIRECTIONS:
            if direction not in raw_frames or direction not in pipes:
                display_frames[direction] = np.zeros(
                    (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8
                )
                continue
            undis = pipes[direction].undistort(raw_frames[direction])
            display_frames[direction] = resize_bgr(
                undis, (DISPLAY_WIDTH, DISPLAY_HEIGHT)
            )

        grid = build_grid(display_frames, labels)
        cv2.imshow("Undistorted Preview (2x2)", grid)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()
    for cap, _, _, _ in caps.values():
        cap.release()
    print("  All cameras released. Done.")


if __name__ == "__main__":
    main()