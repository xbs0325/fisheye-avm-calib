#!/usr/bin/env python3
"""
live_bev.py — 实时 4 路鱼眼 AVM BEV 拼接显示

用法：
  DISPLAY=:0 python3 scripts/live_bev.py
  DISPLAY=:0 python3 scripts/live_bev.py --scale 200 --blend-power 4.0

键盘：
  ESC/q 退出    s 保存帧    g 切换增益    r 重算增益    +/- 调整融合
"""

import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import cv2

from cuda_cv import (
    UndistortWarpPipeline,
    init_undistort_maps,
    log_cuda_status,
    process_frames_to_bev,
    resize_bgr,
)

PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "config", "camera_config.json")
DEFAULT_CALIB_DIR = os.path.join(PROJECT_DIR, "calib_results")
DEFAULT_EXTRINSICS = os.path.join(DEFAULT_CALIB_DIR, "extrinsics.json")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_CALIB_DIR, "live_captures")

DIRECTIONS = ["front", "back", "left", "right"]
CAPTURE_W, CAPTURE_H = 1920, 1536


# ==================== 数据加载 ====================

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def load_extrinsics(path):
    with open(path, "r") as f:
        data = json.load(f)
    result = {
        "old_scale": data.get("scale_px_per_meter", 100.0),
        "old_canvas": tuple(data.get("canvas_size", [1000, 1000])),
        "homographies": {},
        #去畸变 balance 必须与标定时一致(extrinsic_balance)，否则 H 与运行时
        #去畸变图不匹配 -> BEV 整体错位。回退到 balance/0.5 兼容旧文件。
        "undistort_balance": float(data.get("extrinsic_balance", data.get("balance", 0.5))),
    }
    for d in DIRECTIONS:
        if d in data.get("homographies", {}):
            result["homographies"][d] = np.array(data["homographies"][d], dtype=np.float64)
    return result


def load_intrinsics(direction, calib_dir):
    path = os.path.join(calib_dir, f"{direction}.json")
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data["K"], dtype=np.float64), np.array(data["D"], dtype=np.float64)


# ==================== 预计算 ====================

def precompute_undistort_maps(K, D, w, h, balance, for_cuda=True):
    # Live path uses CV_32FC1 maps for cv2.cuda.remap; calib math stays on CV_16SC2 elsewhere.
    return init_undistort_maps(K, D, w, h, balance, for_cuda=for_cuda)


def adjust_homography(H_old, old_scale, old_canvas, new_scale, new_canvas):
    old_cw, old_ch = old_canvas
    new_cw, new_ch = new_canvas
    s = new_scale / old_scale
    tx = new_cw / 2.0 - s * old_cw / 2.0
    ty = new_ch / 2.0 - s * old_ch / 2.0
    A = np.array([[s, 0, tx], [0, s, ty], [0, 0, 1.0]], dtype=np.float64)
    return A @ H_old


def compute_canvas(range_m, scale):
    cw = int(2.0 * range_m * scale)
    ch = int(2.0 * range_m * scale)
    return (cw, ch), (cw / 2.0, ch / 2.0)


def build_weight_maps(canvas_size, center, blend_power):
    """预计算方向权重图 + 有效像素掩码。"""
    h, w = canvas_size[1], canvas_size[0]
    cx, cy = center

    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    gx = (x_grid - cx).astype(np.float64)
    gy = (cy - y_grid).astype(np.float64)
    angle = np.arctan2(gy, gx)

    cam_angle = {"front": np.pi / 2.0, "back": -np.pi / 2.0,
                 "left": np.pi, "right": 0.0}

    raw = {}
    for d, ca in cam_angle.items():
        diff = np.arctan2(np.sin(angle - ca), np.cos(angle - ca))
        w_map = np.clip(np.cos(diff), 0.0, 1.0) ** blend_power
        raw[d] = w_map

    weight_sum = sum(raw.values())
    weight_sum = np.maximum(weight_sum, 1e-10)

    return {d: (raw[d] / weight_sum).astype(np.float32) for d in raw}


# ==================== 相机 ====================

def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 /dev/video{index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 关闭自动曝光，设置手动曝光
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)       # 1=手动曝光
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)              # 关闭自动白平衡
    cap.set(cv2.CAP_PROP_EXPOSURE, 156)            # 手动曝光值（可根据实际亮度调整）

    for _ in range(5):
        cap.grab()
    return cap


# ==================== 增益 ====================

def compute_gains(bev_views):
    """在重叠区域计算增益因子（BFS 链式推导）。"""
    masks = {}
    for d, bev in bev_views.items():
        if bev is None:
            continue
        masks[d] = (cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY) > 10)

    pairs = [("front", "left"), ("front", "right"),
             ("back", "left"), ("back", "right")]

    ratios = {}
    for d1, d2 in pairs:
        if d1 not in masks or d2 not in masks:
            continue
        overlap = masks[d1] & masks[d2]
        if overlap.sum() < 500:
            continue
        m1 = bev_views[d1][overlap].mean(axis=0)
        m2 = bev_views[d2][overlap].mean(axis=0)
        r = m1 / np.maximum(m2, 1.0)
        ratios[(d1, d2)] = r
        ratios[(d2, d1)] = 1.0 / r

    if not ratios:
        return {d: np.ones(3, dtype=np.float32) for d in bev_views}

    gains = {"front": np.ones(3, dtype=np.float32)}
    queue = ["front"]
    while queue:
        cur = queue.pop(0)
        for (a, b), r in ratios.items():
            if a == cur and b not in gains:
                gains[b] = np.clip(gains[cur] * r, 0.5, 2.0).astype(np.float32)
                queue.append(b)

    for d in bev_views:
        gains.setdefault(d, np.ones(3, dtype=np.float32))
    return gains


# ==================== 单帧处理 ====================

def process_frame(raw_frames, pipelines, gains, canvas_size, need_bev_views=False):
    """GPU remap+warp(+blend) when CUDA OpenCV is available; else CPU fallback."""
    frames = {}
    for d in DIRECTIONS:
        frame = raw_frames.get(d)
        if frame is None or d not in pipelines:
            continue
        if frame.shape[1] != CAPTURE_W or frame.shape[0] != CAPTURE_H:
            frame = resize_bgr(frame, (CAPTURE_W, CAPTURE_H))
        frames[d] = frame

    return process_frames_to_bev(
        frames,
        pipelines,
        DIRECTIONS,
        gains=gains,
        canvas_size=canvas_size,
        need_bev_views=need_bev_views,
    )


# ==================== 显示 ====================

def draw_hud(bev, fps_val, blend_power, gain_enabled, canvas_size, scale):
    display = bev.copy()
    cw, ch = canvas_size
    lines = [
        f"FPS: {1.0 / max(fps_val, 0.001):.0f}",
        f"Range: ±{cw / (2.0 * scale):.1f}m  Scale: {scale}px/m",
        f"Blend: cos^{blend_power:.0f}  Gain: {'ON' if gain_enabled else 'OFF'}",
        "ESC/q:quit  s:save  d:debug  g:gain  r:recal  +/-:blend  []:expo",
    ]
    for i, line in enumerate(lines):
        cv2.putText(display, line, (8, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    labels = {"front": "F", "back": "B", "left": "L", "right": "R"}
    positions = {"front": (cw // 2, 25), "back": (cw // 2, ch - 10),
                 "left": (15, ch // 2), "right": (cw - 25, ch // 2)}
    for d, (px, py) in positions.items():
        cv2.putText(display, labels[d], (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return display


# ==================== 主循环 ====================

def main():
    ap = argparse.ArgumentParser(description="实时 4 路鱼眼 AVM BEV 拼接显示")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--extrinsics", default=DEFAULT_EXTRINSICS)
    ap.add_argument("--calib-dir", default=DEFAULT_CALIB_DIR)
    ap.add_argument("--range", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=200.0)
    ap.add_argument("--blend-power", type=float, default=4.0)
    ap.add_argument("--balance", type=float, default=0.5)
    ap.add_argument("--no-gain", action="store_true", help="禁用增益补偿")
    ap.add_argument("--display-size", type=int, default=500)
    ap.add_argument("--exposure", type=float, default=156.0,
                    help="手动曝光值（默认 156，范围 0-255，需先关闭自动曝光）")
    args = ap.parse_args()

    # 1) 加载标定
    print("=" * 60)
    print("  加载标定...")
    print("=" * 60)
    extrinsics = load_extrinsics(args.extrinsics)
    new_canvas, new_center = compute_canvas(args.range, args.scale)
    cw, ch = new_canvas
    print(f"  BEV: {cw}x{ch} px, ±{args.range}m, {args.scale} px/m")

    calib = {}
    for d in DIRECTIONS:
        try:
            calib[d] = load_intrinsics(d, args.calib_dir)
            print(f"  {d:8s}  内参 OK")
        except Exception as e:
            print(f"  {d:8s}  {e}")

    # 2) 预计算
    print("\n" + "=" * 60)
    print("  预计算...")
    print("=" * 60)
    log_cuda_status()

    #去畸变 balance 以标定文件为准(保证与 H 对齐)；--balance 仅对旧文件回退
    undist_balance = extrinsics["undistort_balance"]
    if abs(undist_balance - args.balance) > 1e-6:
        print(f"  [注意] 去畸变 balance={undist_balance:.2f}(取自标定文件，"
              f"--balance={args.balance:.2f} 被忽略以保证 H 对齐)")

    weights = build_weight_maps(new_canvas, new_center, args.blend_power)
    print(f"  权重图 OK (blend_power={args.blend_power})")

    pipelines = {}
    for d in DIRECTIONS:
        if d not in calib or d not in extrinsics["homographies"]:
            continue
        K, D = calib[d]
        m1, m2 = precompute_undistort_maps(
            K, D, CAPTURE_W, CAPTURE_H, undist_balance, for_cuda=True)
        H = adjust_homography(
            extrinsics["homographies"][d],
            extrinsics["old_scale"], extrinsics["old_canvas"],
            args.scale, new_canvas)
        pipelines[d] = UndistortWarpPipeline(
            m1, m2, H=H, canvas_size=new_canvas, weight=weights[d])
        print(f"  {d:8s}  undist+H pipeline OK"
              f" ({'CUDA' if pipelines[d].use_cuda else 'CPU'})")

    # 3) 打开相机
    print("\n" + "=" * 60)
    print("  打开相机...")
    print("=" * 60)

    config = load_config(args.config)
    caps = {}
    for d in DIRECTIONS:
        if d not in pipelines:
            continue
        idx = config.get(d)
        if idx is None:
            continue
        try:
            cap = open_camera(idx)
            # 设置手动曝光
            cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
            caps[d] = (cap, idx)
            print(f"  {d:8s}  /dev/video{idx}  exposure={args.exposure:.0f}")
        except Exception as e:
            print(f"  {d:8s}  /dev/video{idx}: {e}")

    if not caps:
        print("[错误] 无可用相机")
        sys.exit(1)

    # 4) 初始状态
    gains = {d: np.ones(3, dtype=np.float32) for d in DIRECTIONS}
    gain_enabled = not args.no_gain
    need_gain_update = gain_enabled
    blend_power = args.blend_power
    frame_count = 0
    fps_smooth = 0.0
    window_name = "AVM BEV Live"
    display_size = args.display_size

    # 5) 主循环
    print("\n" + "=" * 60)
    print("  实时 BEV 拼接 — ESC/q 退出")
    print("=" * 60)

    while True:
        # ---- 抓帧 ----
        for d in caps:
            caps[d][0].grab()
        raw_frames = {}
        for d in caps:
            ret, frame = caps[d][0].retrieve()
            if ret and frame is not None:
                raw_frames[d] = frame

        # ---- 处理 ----
        # 增益重算需要各路 BEV tile；平时走 GPU 单次加权累加，少一次 download
        result, bev_views, frame_time = process_frame(
            raw_frames, pipelines, gains, new_canvas,
            need_bev_views=need_gain_update and gain_enabled)

        # ---- 增益更新 ----
        if need_gain_update and gain_enabled:
            new_gains = compute_gains(bev_views)
            if new_gains:
                for d in new_gains:
                    if d in gains:
                        # EMA 平滑，避免帧间跳变
                        gains[d] = gains[d] * 0.5 + new_gains[d] * 0.5
                print(f"  增益: L={gains.get('left',[1]*3)[0]:.2f} "
                      f"R={gains.get('right',[1]*3)[0]:.2f} "
                      f"B={gains.get('back',[1]*3)[0]:.2f}")
            need_gain_update = False

        # ---- FPS ----
        frame_count += 1
        fps_smooth = fps_smooth * 0.9 + (1.0 / max(frame_time, 0.001)) * 0.1

        # ---- 显示 ----
        display = draw_hud(result, frame_time, blend_power, gain_enabled,
                           new_canvas, args.scale)
        display = resize_bgr(display, (display_size, display_size))
        cv2.imshow(window_name, display)

        # ---- 键盘 ----
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):
            os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
            path = os.path.join(DEFAULT_OUTPUT_DIR, f"bev_{frame_count:04d}.jpg")
            cv2.imwrite(path, result)
            print(f"  [已保存] {path}")
        elif key == ord('g'):
            gain_enabled = not gain_enabled
            if gain_enabled:
                need_gain_update = True
            else:
                gains = {d: np.ones(3, dtype=np.float32) for d in DIRECTIONS}
            print(f"  增益补偿: {'ON' if gain_enabled else 'OFF'}")
        elif key == ord('r'):
            need_gain_update = True
            print("  重新计算增益...")
        elif key == ord('+') or key == ord('='):
            blend_power = min(blend_power + 1.0, 10.0)
            weights = build_weight_maps(new_canvas, new_center, blend_power)
            for d, pipe in pipelines.items():
                if d in weights:
                    pipe.set_weight(weights[d])
            print(f"  融合幂次: {blend_power}")
        elif key == ord('-'):
            blend_power = max(blend_power - 1.0, 1.0)
            weights = build_weight_maps(new_canvas, new_center, blend_power)
            for d, pipe in pipelines.items():
                if d in weights:
                    pipe.set_weight(weights[d])
            print(f"  融合幂次: {blend_power}")
        elif key == ord(']'):
            for cap, _ in caps.values():
                cur = cap.get(cv2.CAP_PROP_EXPOSURE)
                cap.set(cv2.CAP_PROP_EXPOSURE, min(cur + 10, 255))
            print(f"  曝光值 +10")
        elif key == ord('['):
            for cap, _ in caps.values():
                cur = cap.get(cv2.CAP_PROP_EXPOSURE)
                cap.set(cv2.CAP_PROP_EXPOSURE, max(cur - 10, 0))
            print(f"  曝光值 -10")
        elif key == ord('d'):
            # 保存调试帧：强制拉各路 BEV tile
            _, bev_dbg, _ = process_frame(
                raw_frames, pipelines, gains, new_canvas, need_bev_views=True)
            debug_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"debug_{frame_count:04d}")
            os.makedirs(debug_dir, exist_ok=True)
            for d in DIRECTIONS:
                if d in bev_dbg and bev_dbg[d] is not None:
                    cv2.imwrite(os.path.join(debug_dir, f"{d}_bev.jpg"), bev_dbg[d])
            cv2.imwrite(os.path.join(debug_dir, "stitched.jpg"), result)
            print(f"  [已保存调试帧] {debug_dir}/")

    # 6) 清理
    print("\n  释放相机...")
    cv2.destroyAllWindows()
    for cap, _ in caps.values():
        cap.release()
    print("  完成。")


if __name__ == "__main__":
    main()