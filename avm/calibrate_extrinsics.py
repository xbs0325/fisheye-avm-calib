#!/usr/bin/env python3
"""
calibrate_extrinsics.py — 4 路鱼眼 AVM 外参标定（BEV 单应性矩阵 H）

在已完成的内参标定（calib_results/{front,back,left,right}.json）基础上，
求每路相机“去畸变图 -> 统一 BEV 画布”的单应性矩阵 H，偏移烘焙进 H，
4 路自然落到各自象限（前->上、后->下、左->左、右->右）。

几何约定（与 05_相机标定.md / 07_BEV逆透视变换.md / avm_core.py 一致）：
  车体系：+X 右、+Y 前、原点 = 车体中心。
  画布：canvas_x = gx*scale + cw/2,  canvas_y = -gy*scale + ch/2  (y 前 = 画布向上)。

placement（外参物理距离，卷尺量）：
  把 A4 8×6 棋盘格平铺在每路相机前的地面、轴对齐，量“板近边到车体中心
  的距离 near_m”（沿该路视线方向）+ 可选横向偏移 lateral_m。脚本据板几何 +
  相机朝向自动算 4 个外角点地面坐标（见 ground_corners()）。

无显示器也能跑：--images-dir 模式全程不弹窗，调试图/预览图全写盘，scp 回看。
实时抓拍：--capture 开 4 路预览；棋盘稳定检出一段时间后 SPACE，
全分辨率连拍多帧、角点取均值再求 H；ESC 存盘。

用法：
  # 实时抓拍（需本机桌面；SSH 时指定 DISPLAY，本机常见 :1）
  DISPLAY=:1 python3 scripts/calibrate_extrinsics.py --capture
  # 无显示器，用预先放好的 4 张图
  python3 scripts/calibrate_extrinsics.py --images-dir calib_images/extrinsics/
  # 自定义画布/比例；棋盘默认读 config/chessboard_config.json，也可 CLI 覆盖
  python3 scripts/calibrate_extrinsics.py --images-dir calib_images/extrinsics/ \\
      --scale 100 --canvas 1000 1000 \\
      --placements config/extrinsic_placements.json
  # 临时覆盖棋盘：--pattern-size 8x6 --square 0.08
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import cv2

from cuda_cv import log_cuda_status, resize_bgr, warp_perspective_bgr
from cuda_cv import init_undistort_maps as _init_undistort_maps
from remote_control import merge_wait_key, resolve_control_file


def ensure_gui_display() -> str:
    """为 cv2.imshow 准备可用 DISPLAY（SSH/tty 常为空）。

    本机 GNOME 可能在 :1 而非 :0。若最终仍无法初始化 GTK，抛出带命令提示的错误。
    """
    def _socket_ok(display: str) -> bool:
        d = (display or "").strip()
        if not d.startswith(":"):
            return False
        num = d[1:].split(".", 1)[0]
        if not num.isdigit():
            return False
        return Path(f"/tmp/.X11-unix/X{num}").exists()

    display = (os.environ.get("DISPLAY") or "").strip()
    if not display or not _socket_ok(display):
        # Prefer higher display numbers first (desktop often :1 while greeter was :0).
        candidates = []
        xdir = Path("/tmp/.X11-unix")
        if xdir.is_dir():
            for sock in sorted(xdir.glob("X*"), reverse=True):
                name = sock.name  # X1
                if len(name) >= 2 and name[1:].isdigit():
                    candidates.append(f":{name[1:]}")
        for cand in candidates:
            os.environ["DISPLAY"] = cand
            try:
                cv2.namedWindow("__avm_display_probe__", cv2.WINDOW_NORMAL)
                cv2.destroyWindow("__avm_display_probe__")
                print(f"  [显示] 自动使用 DISPLAY={cand}")
                return cand
            except cv2.error:
                continue
        raise RuntimeError(
            "无法初始化 OpenCV 窗口（GTK）。当前会话没有可用 DISPLAY。\n"
            "  本机桌面终端直接运行，或 SSH 时指定桌面显示号，例如：\n"
            "    DISPLAY=:1 python3 scripts/calibrate_extrinsics.py --capture\n"
            "  无显示器可用：\n"
            "    python3 scripts/calibrate_extrinsics.py --images-dir calib_images/extrinsics/"
        )

    try:
        cv2.namedWindow("__avm_display_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__avm_display_probe__")
    except cv2.error as exc:
        raise RuntimeError(
            f"DISPLAY={display} 存在，但 OpenCV/GTK 仍无法开窗：{exc}\n"
            "  请在桌面会话的终端运行，或改用 --images-dir。"
        ) from exc
    return display


# ---------------- 路径与默认值 ----------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Allow `python3 scripts/calibrate_extrinsics.py` imports of sibling modules
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from board_config import DEFAULT_BOARD_CONFIG, resolve_board_args  # noqa: E402

DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "config", "camera_config.json")
DEFAULT_PLACEMENTS = os.path.join(PROJECT_DIR, "config", "extrinsic_placements.json")
DEFAULT_CALIB_DIR = os.path.join(PROJECT_DIR, "calib_results")
DEFAULT_OUTPUT = os.path.join(DEFAULT_CALIB_DIR, "extrinsics.json")
DEFAULT_DRAFT_OUTPUT = os.path.join(DEFAULT_CALIB_DIR, "extrinsics_draft.json")
DEFAULT_EXTR_IMAGES = os.path.join(PROJECT_DIR, "calib_images", "extrinsics")

DIRECTIONS = ["front", "back", "left", "right"]
# 相机朝向（车体系视线方向）：前 +Y、后 -Y、右 +X、左 -X
CAM_AXIS = {"front": (0.0, 1.0), "back": (0.0, -1.0),
            "right": (1.0, 0.0), "left": (-1.0, 0.0)}

# 采集分辨率（与内参标定一致）
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1536
# 实时预览：硬件缩小 + 多线程取流；SPACE 才开全分辨率求 H。
DISPLAY_WIDTH = 480
DISPLAY_HEIGHT = 360
LIVE_PREVIEW_WAIT_MS = 1
LIVE_DETECT_PER_FRAME = 1

# 外参多帧均值（降低单帧抖动）
STABLE_FRAMES_DEFAULT = 10     # 预览连续检出次数（每路独立计数）
BURST_FRAMES_DEFAULT = 8       # 全分辨率连拍帧数
BURST_MIN_OK_DEFAULT = 5       # 连拍中至少成功检出帧数
CORNER_ALIGN_MAX_PX = 40.0     # 与参考帧对齐时，单角点平均距离上限
CORNER_OUTLIER_RMS_PX = 2.5    # 相对中位数 RMS 超此视为离群帧

# 画布
CANVAS_SIZE = (1000, 1000)    # (w, h)
SCALE_PX_PER_M = 100.0
BALANCE = 0.5                  # 去畸变 balance

# H 质量检查阈值（可在 CLI 用 --min-h-svd 覆盖）
H_SVD_MIN_DEFAULT = 0.03       # 2×2 块最小奇异值，低于此易蝴蝶结
# 光心落地点距车心多远由安装高度/俯角决定，不反映标定好坏，
# 这里只作为「H 是否算飞了」的粗略上界。方向是否翻转由沿视线符号判定。
H_CENTER_TOL_PX = 350.0        # 图像中心落点距画布中心的粗略上界
H_CENTER_FLIP_TOL_PX = 50.0    # 沿视线为负超过此值才判定 H 翻转
H_EDGE_SPAN_MIN_PX = 25.0      # 图像边缘扫描后 BEV 位移模长下限
H_LR_SIGMA_RATIO_MAX = 4.0     # 左/右 H 主奇异值最大允许比值
BOARD_EDGE_RATIO_MAX = 1.35    # 棋盘外框对边长度比上限（倾斜/非平面）
INTRINSIC_RMS_WARN = 1.5       # 内参 RMS 警告线（px）


# ---------------- 内参与去畸变（复用远程已有实现） ----------------

def load_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    valid = set(DIRECTIONS)
    for k in cfg:
        if k not in valid:
            raise ValueError(f"camera_config 含未知方向 '{k}'，期望 {sorted(valid)}")
    return cfg


def load_calib(direction, calib_dir):
    path = os.path.join(calib_dir, f"{direction}.json")
    with open(path, "r") as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float64)
    D = np.array(data["D"], dtype=np.float64)
    rms = data.get("rms", data.get("reproj_error"))
    return K, D, rms


def precompute_undistort_maps(K, D, w, h, balance):
    """与 calibrate_intrinsics.py 一致的去畸变 map（balance 缩 fx/fy，主点居中）。

    标定求 H 强制 CV_16SC2 + CPU remap，保证与历史结果一致、角点亚像素稳定。
    实时预览/拼接另走 cuda_cv 的 CV_32FC1 GPU 路径。
    """
    return _init_undistort_maps(K, D, w, h, balance, for_cuda=False)


def scale_intrinsics(K, src_w, src_h, dst_w, dst_h):
    """按分辨率比例缩放内参，供预览小图去畸变。"""
    Ks = np.asarray(K, dtype=np.float64).copy()
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    Ks[0, 0] *= sx
    Ks[1, 1] *= sy
    Ks[0, 2] *= sx
    Ks[1, 2] *= sy
    return Ks


# ---------------- placement：板近边距离 -> 4 外角点地面坐标 ----------------

def board_dims(cols, rows, square):
    """板物理尺寸：长边=(cols+1)*square，短边=(rows+1)*square。"""
    return (cols + 1) * square, (rows + 1) * square


def ground_corners(direction, near, lateral, orient, cols, rows, square):
    """
    返回 4 个【最外层内角点】地面坐标(米,车体系)，CCW。
    near: 板近边物理边缘到车体中心距离(沿视线,>0)。
    lateral: 板中心横向偏移(米,+右)；默认 0 居中。
    orient: 'long-lateral'(长边横向,默认) | 'long-along'(长边顺视线)。
    内角点从板物理边缘各缩进 1 格(square)。
    """
    long_m, short_m = board_dims(cols, rows, square)
    S = square
    if orient == "long-lateral":
        w_lat, w_dep = long_m, short_m     # 横向物理宽, 顺视线物理深
    else:  # long-along
        w_lat, w_dep = short_m, long_m
    half_lat = w_lat / 2.0

    # 统一在“板本地系”算矩形内角点，再按相机朝向旋到车体系。
    # 板本地系：u 横向(右+)，v 顺视线(远+)。近边 v=0。
    u0 = lateral - half_lat + S
    u1 = lateral + half_lat - S
    v0 = near + S
    v1 = near + w_dep - S
    local = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]  # CCW

    ax, ay = CAM_AXIS[direction]  # 视线方向单位向量(车体系)
    # 横向单位向量 = 视线方向逆时针转 90°：(ax,ay) -> (-ay, ax)
    hx, hy = -ay, ax
    # (u,v) -> 车体系: pos = u*横向 + v*视线
    corners = []
    for (u, v) in local:
        gx = u * hx + v * ax
        gy = u * hy + v * ay
        corners.append((float(gx), float(gy)))
    return corners, (w_lat, w_dep)


# ---------------- 棋盘格检测 ----------------

def detect_board(gray, cols, rows):
    """SB 优先 + 经典兜底。必须与预览用同一套算法，否则会出现
    "预览已 READY，但连拍求 H 检出不足" —— 远距时经典算法先失效。"""
    from avm.detect_board_hires import find_board_corners

    corners = find_board_corners(gray, (cols, rows))
    if corners is None:
        return None
    return corners.reshape(-1, 2)  # (rows*cols, 2) row-major 栅格序


def _corner_mean_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def align_corners_to_ref(corners: np.ndarray, ref: np.ndarray, cols: int, rows: int):
    """把一帧角点序对齐到参考帧（处理 180° 翻转 / 反向枚举）。"""
    c = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    r = np.asarray(ref, dtype=np.float64).reshape(-1, 2)
    if c.shape != r.shape:
        return None, float("inf")
    candidates = [c, c[::-1]]
    grid = c.reshape(rows, cols, 2)
    rot180 = np.flip(np.flip(grid, 0), 1).reshape(-1, 2)
    candidates.append(rot180)
    candidates.append(rot180[::-1])
    best, best_d = None, float("inf")
    for cand in candidates:
        d = _corner_mean_dist(cand, r)
        if d < best_d:
            best, best_d = cand, d
    return best, best_d


def _seam_order_candidates(corners: np.ndarray, cols: int, rows: int):
    """接缝精修用的角点序候选：原序 / 反向 / 180° / 180° 反向。"""
    c = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    grid = c.reshape(rows, cols, 2)
    rot180 = np.flip(np.flip(grid, 0), 1).reshape(-1, 2)
    return [c, c[::-1], rot180, rot180[::-1]]


def _project_corners_h(corners: np.ndarray, H) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def _reproj_rms(corners: np.ndarray, targets: np.ndarray, H) -> float:
    pred = _project_corners_h(corners, H)
    tgt = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((pred - tgt) ** 2, axis=1))))


# 相邻相机对：重叠区通常在这两路之间
SEAM_PAIRS = (
    ("front", "left"),
    ("front", "right"),
    ("back", "left"),
    ("back", "right"),
)


def refine_seam_homography(
    corners_ref: np.ndarray,
    corners_slave: np.ndarray,
    H_ref,
    H_slave_old,
    cols: int,
    rows: int,
):
    """接缝精修：锁住参考路 H，把从路 H 重求到同一块板的 BEV 目标。

    两路角点必须已在「去畸变图」坐标系（与存盘 H 一致）。
    返回 (H_slave_new, stats)；失败返回 (None, stats_with_error)。
    """
    H_ref = np.asarray(H_ref, dtype=np.float64)
    H_old = np.asarray(H_slave_old, dtype=np.float64)
    ref = np.asarray(corners_ref, dtype=np.float64).reshape(-1, 2)
    if ref.shape[0] != cols * rows:
        return None, {"error": f"参考路角点数 {ref.shape[0]} != {cols*rows}"}

    P_ref = _project_corners_h(ref, H_ref)

    best = None  # (H, rms, order_i)
    for i, cand in enumerate(_seam_order_candidates(corners_slave, cols, rows)):
        H, mask = cv2.findHomography(
            cand.astype(np.float32),
            P_ref.astype(np.float32),
            cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.995,
        )
        if H is None:
            continue
        inlier = (
            mask.ravel().astype(bool)
            if mask is not None
            else np.ones(len(cand), dtype=bool)
        )
        if int(inlier.sum()) < max(4, cols * rows // 2):
            continue
        rms = _reproj_rms(cand[inlier], P_ref[inlier], H)
        inlier_ratio = float(inlier.mean())
        if best is None or rms < best[1] - 1e-9 or (
            abs(rms - best[1]) < 1e-9 and inlier_ratio > best[3]
        ):
            best = (H, rms, i, inlier_ratio, int(inlier.sum()))

    if best is None:
        return None, {"error": "findHomography 失败：两路角点无法对齐"}

    H_new, rms_after, order_i, inlier_ratio, n_inliers = best
    # 用同一角点序评估精修前误差，才有可比性
    cand_best = _seam_order_candidates(corners_slave, cols, rows)[order_i]
    rms_before = _reproj_rms(cand_best, P_ref, H_old)
    # 角点在 BEV 上的跨度：过小说明板几乎退化
    span = float(np.linalg.norm(P_ref.max(axis=0) - P_ref.min(axis=0)))
    stats = {
        "rms_before_px": rms_before,
        "rms_after_px": rms_after,
        "improved_px": float(rms_before - rms_after),
        "order_index": order_i,
        "inlier_ratio": inlier_ratio,
        "n_inliers": n_inliers,
        "board_span_bev_px": span,
    }
    if span < 20.0:
        stats["warning"] = f"板在 BEV 上跨度仅 {span:.1f}px，精修可能不稳"
    return H_new, stats


def load_extrinsics_file(path: str):
    """读 extrinsics.json，homographies 转成 float64 ndarray。"""
    with open(path, "r") as f:
        data = json.load(f)
    H = {
        d: np.asarray(M, dtype=np.float64)
        for d, M in (data.get("homographies") or {}).items()
    }
    return data, H


def patch_extrinsics_homography(
    path: str,
    direction: str,
    H_new,
    *,
    rms=None,
    seam_meta=None,
):
    """只改某一路 H（及可选 rms / seam 元数据），其余字段原样保留。"""
    with open(path, "r") as f:
        data = json.load(f)
    homs = data.setdefault("homographies", {})
    if direction not in homs:
        raise KeyError(f"{path} 中没有 {direction} 的 H，无法精修")
    homs[direction] = np.asarray(H_new, dtype=np.float64).tolist()
    if rms is not None:
        data.setdefault("rms_errors", {})[direction] = float(rms)
    if seam_meta is not None:
        hist = data.setdefault("seam_refined", [])
        hist.append(dict(seam_meta))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def average_corners(
    corners_list,
    cols: int,
    rows: int,
    *,
    outlier_rms_px: float = CORNER_OUTLIER_RMS_PX,
    align_max_px: float = CORNER_ALIGN_MAX_PX,
):
    """多帧角点对齐后取均值；剔除相对中位数的离群帧。

    返回: mean_corners (N,2), n_used, stats dict
    """
    if not corners_list:
        raise ValueError("corners_list 为空")
    ref = np.asarray(corners_list[0], dtype=np.float64).reshape(-1, 2)
    aligned = [ref]
    rejected_align = 0
    for c in corners_list[1:]:
        a, dist = align_corners_to_ref(c, ref, cols, rows)
        if a is None or dist > align_max_px:
            rejected_align += 1
            continue
        aligned.append(a)
    if len(aligned) < 1:
        raise RuntimeError("角点对齐失败")
    stack = np.stack(aligned, axis=0)  # (F,N,2)
    median = np.median(stack, axis=0)
    kept = []
    frame_rms = []
    for a in aligned:
        rms = float(np.sqrt(np.mean(np.sum((a - median) ** 2, axis=1))))
        frame_rms.append(rms)
        if rms <= outlier_rms_px * 3.0:
            kept.append(a)
    if len(kept) < max(2, (len(aligned) + 1) // 2):
        kept = aligned
    mean = np.mean(np.stack(kept, axis=0), axis=0)
    stats = {
        "n_input": len(corners_list),
        "n_aligned": len(aligned),
        "n_used": len(kept),
        "rejected_align": rejected_align,
        "frame_rms_px": frame_rms,
        "mean_frame_rms_px": float(np.mean(frame_rms)) if frame_rms else None,
        "corner_jitter_px": float(
            np.mean(np.std(np.stack(kept, axis=0), axis=0))
        ),
    }
    return mean.astype(np.float64), len(kept), stats


def grab_full_resolution_burst(index, n_frames=BURST_FRAMES_DEFAULT, warm_grabs=6):
    """全分辨率连拍：开一次相机，读 n 帧后释放。"""
    cap, w, h, _backend = open_camera(index, CAPTURE_WIDTH, CAPTURE_HEIGHT)
    frames = []
    try:
        for _ in range(warm_grabs):
            cap.grab()
        for _ in range(int(n_frames)):
            ok, frame = cap.read()
            frame = _normalize_bgr(frame)
            if ok and frame is not None:
                frames.append(frame)
    finally:
        cap.release()
    return frames, w, h


def grid_outer_idx(cols, rows):
    """栅格 4 外角点在 row-major 序列里的下标：TL,TR,BR,BL（检测序成环）。"""
    return [0, cols - 1, rows * cols - 1, (rows - 1) * cols]


def bilinear_ground(g_ll, g_lr, g_ul, g_br, cols, rows):
    """4 外角点(地面坐标)双线性插值出全部 (rows*cols) 内角点地面坐标。
    g_ll=grid(0,0) g_lr=grid(0,cols-1) g_ul=grid(rows-1,0) g_br=grid(rows-1,cols-1)。"""
    pts = np.zeros((rows, cols, 2), np.float64)
    for r in range(rows):
        for c in range(cols):
            u = c / (cols - 1) if cols > 1 else 0.0
            v = r / (rows - 1) if rows > 1 else 0.0
            pts[r, c] = (g_ll * (1 - u) * (1 - v) + g_lr * u * (1 - v)
                         + g_ul * (1 - u) * v + g_br * u * v)
    return pts.reshape(-1, 2)


def ground_to_canvas(gx, gy, scale, cx, cy):
    """车体地面坐标(米) -> 画布像素。y 前 = 画布向上(行号减小)。"""
    return np.float32([gx * scale + cx, -gy * scale + cy])


def best_homography(corners, ground4, cols, rows, scale, canvas):
    """
    corners: 检测到的全部角点(row-major, (N,2))。
    ground4: 4 外角点地面坐标(车体系, CCW)，与 placement 文件一致。
    用全部角点 + cv2.findHomography(RANSAC) 求 H，返回最佳 H 和 RMS。

    与旧版不同：不再只用 4 个外角点 getPerspectiveTransform，
    而是用全部 48 个内角点通过 RANSAC 鲁棒估计 H，充分利用冗余信息。
    """
    cw, ch = canvas
    cx, cy = cw / 2.0, ch / 2.0
    g4 = np.asarray(ground4, np.float64)            # (4,2) CCW
    outer_idx = grid_outer_idx(cols, rows)           # 检测序 TL,TR,BR,BL
    det_outer = corners[outer_idx]                   # (4,2)
    # 检测序外角点对应的栅格坐标
    grid_pos = [(0, 0), (0, cols - 1),
                (rows - 1, cols - 1), (rows - 1, 0)]

    # 规则棋盘的 4 个旋转假设都能得到极小重投影 RMS，不能靠 RMS 判方向。
    # 先用物理长短边排除 90° 错解，再利用透视规律（近边看起来更长）
    # 在剩余的 180° 两解中选出正确方向。
    det_edge_len = np.array([
        np.linalg.norm(det_outer[(i + 1) % 4] - det_outer[i])
        for i in range(4)
    ], dtype=np.float64)
    ground_edge_len = np.array([
        np.linalg.norm(g4[(i + 1) % 4] - g4[i])
        for i in range(4)
    ], dtype=np.float64)
    grid_ratio = (cols - 1) / max(rows - 1, 1)
    dim_error = []
    for k in range(4):
        assigned_ratio = ground_edge_len[k] / max(
            ground_edge_len[(k + 1) % 4], 1e-12
        )
        dim_error.append(abs(np.log(assigned_ratio / grid_ratio)))
    min_dim_error = min(dim_error)
    dimension_valid = [
        k for k, err in enumerate(dim_error)
        if err <= min_dim_error + 1e-6
    ]

    # ground4 中离车体原点最近的边就是 placement 定义的 near edge。
    ground_edge_mid_norm = [
        np.linalg.norm((g4[i] + g4[(i + 1) % 4]) * 0.5)
        for i in range(4)
    ]
    near_ground_edge = int(np.argmin(ground_edge_mid_norm))

    def near_edge_score(k):
        # 假设 k 下，检测外框第 i 条边映射到 ground 第 (k+i)%4 条边。
        i_near = (near_ground_edge - k) % 4
        i_far = (i_near + 2) % 4
        return float(det_edge_len[i_near] - det_edge_len[i_far])

    candidate_ks = [max(dimension_valid, key=near_edge_score)]

    # 预计算全部角点的栅格坐标（row-major）
    grid_all = np.zeros((rows * cols, 2), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            grid_all[r * cols + c] = [c, r]

    best = None           # (H, rms)
    best_inlier = -1.0     #伴随 best 的 inlier 比例，用于跨假设择优
    for k in candidate_ks:
        # 只用物理上合理的 CCW 方向（不再枚举镜像）
        order = [(k + i) % 4 for i in range(4)]
        g_assign = [g4[order[i]] for i in range(4)]

        # 用栅格四角对应关系生成全部 ground 坐标
        gmap = {grid_pos[i]: g_assign[i] for i in range(4)}
        full_ground = bilinear_ground(
            gmap[(0, 0)], gmap[(0, cols - 1)],
            gmap[(rows - 1, 0)], gmap[(rows - 1, cols - 1)],
            cols, rows)

        # 全部角点 → 画布坐标
        canvas_pts = np.float32(
            [ground_to_canvas(g[0], g[1], scale, cx, cy) for g in full_ground])

        # RANSAC 鲁棒估计：使用全部 48 个角点
        H, mask = cv2.findHomography(
            corners, canvas_pts, cv2.RANSAC, ransacReprojThreshold=3.0,
            maxIters=2000, confidence=0.995)

        if H is None:
            continue

        # 只用 inlier 计算 RMS
        inlier_mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(corners), dtype=bool)
        inlier_corners = corners[inlier_mask]
        inlier_canvas = canvas_pts[inlier_mask]

        if len(inlier_corners) < 4:
            continue

        pred = cv2.perspectiveTransform(
            inlier_corners.reshape(-1, 1, 2), H).reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((pred - inlier_canvas) ** 2, axis=1))))
        inlier_ratio = len(inlier_corners) / len(corners)

        #择优：先比 inlier 比例(高者优)，并列再比 RMS(小者优)。
        #旧逻辑跨不同大小 inlier 集直接比 RMS，会让 inlier 偏少的错误角点
        #排序靠略低 RMS 顶掉正确排序；inlier 比例才是更可靠的判别量。
        if inlier_ratio < 0.5:
            continue
        if (best is None
                or inlier_ratio > best_inlier + 1e-9
                or (abs(inlier_ratio - best_inlier) < 1e-9 and rms < best[1])):
            best = (H, rms)
            best_inlier = inlier_ratio

    # 回退：如果 RANSAC 全部失败，退回到 4 点 getPerspectiveTransform
    if best is None:
        for k in candidate_ks:
            order = [(k + i) % 4 for i in range(4)]
            g_assign = [g4[order[i]] for i in range(4)]
            img_pts = np.float32([det_outer[i] for i in range(4)])
            canvas_pts_4 = np.float32(
                [ground_to_canvas(g[0], g[1], scale, cx, cy) for g in g_assign])
            H = cv2.getPerspectiveTransform(img_pts, canvas_pts_4)
            gmap = {grid_pos[i]: g_assign[i] for i in range(4)}
            full_ground = bilinear_ground(
                gmap[(0, 0)], gmap[(0, cols - 1)],
                gmap[(rows - 1, 0)], gmap[(rows - 1, cols - 1)],
                cols, rows)
            exp = np.float32(
                [ground_to_canvas(g[0], g[1], scale, cx, cy) for g in full_ground])
            pred = cv2.perspectiveTransform(
                corners.reshape(-1, 1, 2), H).reshape(-1, 2)
            rms = float(np.sqrt(np.mean(np.sum((pred - exp) ** 2, axis=1))))
            if best is None or rms < best[1]:
                best = (H, rms)

    return best


def quality(rms):
    if rms < 1.0:
        return "✅ <1px 合格"
    if rms < 5.0:
        return "⚠️ 1~5px 可用，建议检查 placement 距离/板是否平贴地面"
    return "❌ ≥5px 不准：检查 near_m / 板轴对齐 / 内参是否合格"


def board_quad_metrics(corners, cols, rows):
    """棋盘外框几何：对边比、面积，用于发现板子翘曲或严重倾斜。"""
    outer = corners[grid_outer_idx(cols, rows)]
    top = float(np.linalg.norm(outer[1] - outer[0]))
    bottom = float(np.linalg.norm(outer[2] - outer[3]))
    left = float(np.linalg.norm(outer[3] - outer[0]))
    right = float(np.linalg.norm(outer[2] - outer[1]))
    edge_ratio = max(top, bottom, left, right) / max(min(top, bottom, left, right), 1e-6)
    return {
        "edge_ratio": edge_ratio,
        "top_px": top,
        "bottom_px": bottom,
        "left_px": left,
        "right_px": right,
    }


def analyze_homography(H, direction, img_size, canvas, board_metrics=None,
                       svd_min_thresh=H_SVD_MIN_DEFAULT):
    """单路 H 质量分析，返回指标与警告列表。"""
    iw, ih = img_size
    cw, ch = canvas
    cx_canvas, cy_canvas = cw / 2.0, ch / 2.0

    A = np.asarray(H[:2, :2], dtype=np.float64)
    _, s, _ = np.linalg.svd(A)
    s_max, s_min = float(s[0]), float(s[1])
    cond = s_max / max(s_min, 1e-9)

    sample = np.float32([
        [iw / 2.0, ih / 2.0],
        [0.0, ih / 2.0], [iw - 1.0, ih / 2.0],
        [iw / 2.0, 0.0], [iw / 2.0, ih - 1.0],
    ])
    warped = cv2.perspectiveTransform(sample.reshape(-1, 1, 2), H).reshape(-1, 2)
    center_bev = warped[0]
    # 跨度必须取位移模长，不能取单个分量：侧向相机的图像水平轴映射到 BEV 的
    # 垂直方向，只看 x 分量会恒接近 0，把正常的 H 误判成「被压扁」。
    h_span = float(np.linalg.norm(warped[2] - warped[1]))
    v_span = float(np.linalg.norm(warped[4] - warped[3]))
    center_err = float(np.linalg.norm(center_bev - [cx_canvas, cy_canvas]))

    # 图像中心应落在该相机视线的正方向上；落到反向说明 H 被翻了 180°。
    ax, ay = CAM_AXIS[direction]
    along = ((center_bev[0] - cx_canvas) * ax
             + (cy_canvas - center_bev[1]) * ay)

    h00 = float(H[0, 0])
    h10 = float(H[1, 0])
    warnings = []

    if s_min < svd_min_thresh:
        warnings.append(
            f"σ₂={s_min:.4f}<{svd_min_thresh}: H 病态，BEV 易出现蝴蝶结/拉丝")
    if h_span < H_EDGE_SPAN_MIN_PX:
        warnings.append(
            f"水平跨度 {h_span:.0f}px<{H_EDGE_SPAN_MIN_PX:.0f}: 图像横向被压扁")
    if v_span < H_EDGE_SPAN_MIN_PX:
        warnings.append(
            f"垂直跨度 {v_span:.0f}px<{H_EDGE_SPAN_MIN_PX:.0f}: 图像纵向被压扁")
    # 俯角很陡且相机装得靠内时，正确的 along 本身就接近 0，
    # 所以只有明显为负才判翻转，贴近 0 的只提示存疑。
    if along < -H_CENTER_FLIP_TOL_PX:
        warnings.append(
            f"图像中心落在 {direction} 视线反方向 ({along:.0f}px): H 方向翻转，"
            "板的摆位或朝向与 placement 不符")
    elif along <= 0:
        warnings.append(
            f"图像中心几乎贴着车心 ({along:.0f}px): 方向存疑，请核对 placement")
    elif center_err > H_CENTER_TOL_PX:
        warnings.append(
            f"光心落点距画布中心 {center_err:.0f}px>{H_CENTER_TOL_PX:.0f}")
    if board_metrics and board_metrics["edge_ratio"] > BOARD_EDGE_RATIO_MAX:
        warnings.append(
            f"棋盘外框对边比 {board_metrics['edge_ratio']:.2f}>{BOARD_EDGE_RATIO_MAX}: "
            "板可能翘曲/放桌上/朝向不对")

    if any("病态" in w or "压扁" in w or "翻转" in w for w in warnings):
        status = "bad"
    elif warnings:
        status = "warn"
    else:
        status = "ok"

    return {
        "status": status,
        "svd_max": s_max,
        "svd_min": s_min,
        "svd_cond": cond,
        "h00": h00,
        "h10": h10,
        "bev_h_span_px": h_span,
        "bev_v_span_px": v_span,
        "center_offset_px": center_err,
        "center_along_axis_px": float(along),
        "board_edge_ratio": None if not board_metrics else board_metrics["edge_ratio"],
        "warnings": warnings,
    }


def homography_qc_label(qc):
    if qc["status"] == "ok":
        return "✅ H 几何正常"
    if qc["status"] == "warn":
        return "⚠️ H 有疑点"
    return "❌ H 病态"


def _principal_sigma(H):
    """H 线性部分的主奇异值：与图像/车体坐标轴朝向无关。"""
    return float(np.linalg.svd(
        np.asarray(H, dtype=np.float64)[:2, :2], compute_uv=False)[0])


def check_lr_symmetry(homographies, h_quality):
    """左右相机的整体缩放应同量级。

    不能比 |H[0,0]|：侧向相机的图像水平轴映射到 BEV 垂直方向，
    H[0,0] 结构性接近 0，比值毫无意义。奇异值不受坐标轴朝向影响。
    """
    if "left" not in homographies or "right" not in homographies:
        return []
    s_l = _principal_sigma(homographies["left"])
    s_r = _principal_sigma(homographies["right"])
    lo, hi = min(s_l, s_r), max(s_l, s_r)
    ratio = hi / max(lo, 1e-9)
    if ratio <= H_LR_SIGMA_RATIO_MAX:
        return []
    return [
        f"左/右 H 主奇异值比={ratio:.1f} (left={s_l:.4f}, right={s_r:.4f})>"
        f"{H_LR_SIGMA_RATIO_MAX}: 摆放或视角明显不对称",
    ]


def print_homography_qc(direction, qc):
    print(f"          H-QC: σ₂={qc['svd_min']:.4f}  "
          f"跨度 H={qc['bev_h_span_px']:.0f}px V={qc['bev_v_span_px']:.0f}px  "
          f"{homography_qc_label(qc)}")
    for msg in qc["warnings"]:
        print(f"          ⚠️  {msg}")


# ---------------- placement 加载 ----------------

def load_placements(path, cols, rows, square):
    """
    返回 {direction: {near_m, lateral_m, orient, ground_corners, board_w_m, board_h_m}}。
    文件格式：
      {
        "front": {"near_m": 0.5, "lateral_m": 0.0, "orient": "long-lateral"},
        ...
      }
    缺方向用默认 near_m=0.5, lateral_m=0.0, orient=long-lateral。
    """
    out = {}
    raw = {}
    if path and os.path.isfile(path):
        with open(path, "r") as f:
            raw = json.load(f)
    for d in DIRECTIONS:
        e = dict(raw.get(d, {}))
        near = float(e.get("near_m", 0.5))
        lateral = float(e.get("lateral_m", 0.0))
        orient = e.get("orient", "long-lateral")
        if orient not in ("long-lateral", "long-along"):
            raise ValueError(f"{d}: orient 必须是 long-lateral 或 long-along，得到 {orient}")
        if near <= 0:
            raise ValueError(f"{d}: near_m 必须 > 0（板近边到车体中心距离）")
        corners, (w, dep) = ground_corners(d, near, lateral, orient, cols, rows, square)
        out[d] = {"near_m": near, "lateral_m": lateral, "orient": orient,
                  "board_w_m": w, "board_h_m": dep,
                  "ground_corners_m": [[float(x), float(y)] for (x, y) in corners]}
    return out


def parse_near_cli(s):
    """'front=0.5,back=0.5,left=0.5,right=0.5' -> {dir: near_m}。"""
    out = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split("=")
        k = k.strip()
        if k not in DIRECTIONS:
            raise ValueError(f"--near 含未知方向 '{k}'，期望 {DIRECTIONS}")
        out[k] = float(v)
    return out


# ---------------- 相机（实时抓拍用） ----------------

def build_gst_camera_pipeline(index, width, height, with_videoconvert=True):
    """CSI/VI: 全分辨率 YUY2 -> nvvidconv 硬件缩到预览尺寸。

    默认直接出 BGRx（少一次 CPU videoconvert）；失败再带回退管道。
    """
    device = f"/dev/video{index}"
    head = (
        f"v4l2src device={device} io-mode=2 do-timestamp=true ! "
        f"video/x-raw,format=YUY2,width={CAPTURE_WIDTH},height={CAPTURE_HEIGHT} ! "
        f"nvvidconv ! video/x-raw,format=BGRx,width={width},height={height}"
    )
    if with_videoconvert:
        return (
            f"{head} ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
    return f"{head} ! appsink drop=true max-buffers=1 sync=false"


def _normalize_bgr(frame):
    if frame is None:
        return None
    if frame.ndim == 3 and frame.shape[2] == 4:
        return frame[:, :, :3].copy()
    return frame


def open_camera(index, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT):
    # 先试无 videoconvert（更省 CPU），失败再回退。
    for use_vc in (False, True):
        pipe = build_gst_camera_pipeline(index, width, height, with_videoconvert=use_vc)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ok, frame = cap.read()
            frame = _normalize_bgr(frame)
            if ok and frame is not None and frame.size > 0:
                for _ in range(3):
                    cap.grab()
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
                if aw <= 0:
                    aw = frame.shape[1]
                if ah <= 0:
                    ah = frame.shape[0]
                return cap, aw, ah, ("gst+vc" if use_vc else "gst")
            cap.release()

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"打不开 /dev/video{index}。检查权限(试: sudo usermod -aG video $USER)")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(5):
        cap.grab()
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
    return cap, aw, ah, "v4l2"


def open_preview_camera(index):
    """实时 UI：硬件缩小预览。"""
    return open_camera(index, DISPLAY_WIDTH, DISPLAY_HEIGHT)


def grab_full_resolution_frame(index, warm_grabs=6):
    """SPACE 锁定用：短暂打开全分辨率，取一帧后立刻释放。"""
    cap, w, h, _backend = open_camera(index, CAPTURE_WIDTH, CAPTURE_HEIGHT)
    try:
        for _ in range(warm_grabs):
            cap.grab()
        ok, frame = cap.read()
        frame = _normalize_bgr(frame)
        if not ok or frame is None:
            return None, w, h
        return frame, w, h
    finally:
        cap.release()


class PreviewWorker:
    """每路独立线程取流，主循环只取最新帧（避免 4 路串行 read 叠延迟）。"""

    def __init__(self, direction, index):
        self.direction = direction
        self.index = int(index)
        self.cap = None
        self.width = DISPLAY_WIDTH
        self.height = DISPLAY_HEIGHT
        self.backend = ""
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread = None
        self.fps = 0.0

    def start(self):
        self.cap, self.width, self.height, self.backend = open_preview_camera(self.index)
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name=f"preview-{self.direction}", daemon=True)
        self._thread.start()

    def _loop(self):
        n = 0
        t0 = time.monotonic()
        while self._running and self.cap is not None:
            ok, frame = self.cap.read()
            frame = _normalize_bgr(frame)
            if not ok or frame is None:
                time.sleep(0.002)
                continue
            with self._lock:
                self._frame = frame
            n += 1
            now = time.monotonic()
            if now - t0 >= 1.0:
                self.fps = n / (now - t0)
                n = 0
                t0 = now

    def get(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame  # 主线程立刻处理，不 copy

    def stop(self):
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=1.5)
        self._thread = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

# ---------------- 单路标定 ----------------

def calibrate_one(direction, img, K, D, map1, map2, placement,
                  cols, rows, square, scale, canvas, debug_prefix,
                  svd_min_thresh=H_SVD_MIN_DEFAULT, corners=None):
    """
    对单路：去畸变 -> 检测棋盘（或使用外部均值角点）-> 全部角点 RANSAC 求 H -> 存调试图。
    返回 (H, rms, qc) 或 (None, None, None)（检测失败）。

    去畸变与角点检测刻意走 CPU（CV_16SC2），保证 H 数值稳定可复现。
    仅调试图 BEV warp 可用 GPU。
    """
    undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    if corners is None:
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        corners = detect_board(gray, cols, rows)
    else:
        corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if corners is None:
        dbg = debug_prefix + "_undist_dbg.jpg"
        cv2.imwrite(dbg, undist)
        print(f"  {direction:8s}  ❌ 去畸变图未检测到 {cols}x{rows} 棋盘"
              f"（板是否在 FOV 内/光照/倾斜）。已存 {dbg}")
        return None, None, None

    board_metrics = board_quad_metrics(corners, cols, rows)
    H, rms = best_homography(
        corners, placement["ground_corners_m"], cols, rows, scale, canvas)
    qc = analyze_homography(
        H, direction, (undist.shape[1], undist.shape[0]), canvas,
        board_metrics=board_metrics, svd_min_thresh=svd_min_thresh)

    # 调试图：去畸变 + 角点叠加
    # 求 H 用 float64，但 drawChessboardCorners 只接受 CV_32FC2
    cv2.drawChessboardCorners(
        undist, (cols, rows),
        corners.reshape(-1, 1, 2).astype(np.float32), True,
    )
    cv2.imwrite(debug_prefix + "_undist_corners.jpg", undist)
    # 单路 BEV 预览
    bev = warp_perspective_bgr(undist, H, canvas)
    cv2.imwrite(debug_prefix + "_bev_preview.jpg", bev)
    return H, rms, qc


def calibrate_one_burst(
    direction,
    frames,
    K,
    D,
    map1,
    map2,
    placement,
    cols,
    rows,
    square,
    scale,
    canvas,
    debug_prefix,
    *,
    svd_min_thresh=H_SVD_MIN_DEFAULT,
    min_ok=BURST_MIN_OK_DEFAULT,
    balance=None,
):
    """连拍多帧：每帧检角点 → 对齐取均值 → 用均值角点求 H。

    balance 给定时，角点在鱼眼原图上检测再映射到去畸变坐标系：
    去畸变会裁视场并拉伸边缘，直接在其上检测会明显丢检出率。
    """
    from avm.cuda_cv import undistort_points_fisheye

    corners_list = []
    last_ok_img = None
    for i, img in enumerate(frames):
        undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        if balance is None:
            gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
            c = detect_board(gray, cols, rows)
        else:
            h0, w0 = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            c_raw = detect_board(gray, cols, rows)
            c = (
                None if c_raw is None
                else undistort_points_fisheye(c_raw, K, D, w0, h0, balance)
            )
        if c is None:
            continue
        corners_list.append(c)
        last_ok_img = img
        vis = undist.copy()
        cv2.drawChessboardCorners(vis, (cols, rows), c.reshape(-1, 1, 2), True)
        cv2.imwrite(f"{debug_prefix}_burst_{i:02d}.jpg", vis)

    if len(corners_list) < int(min_ok):
        print(
            f"  {direction:8s}  ❌ 连拍检出不足 "
            f"{len(corners_list)}/{len(frames)}（需要 ≥{min_ok}）"
        )
        return None, None, None

    mean_corners, n_used, stats = average_corners(corners_list, cols, rows)
    print(
        f"  {direction:8s}  多帧角点: 用 {n_used}/{stats['n_input']} 帧"
        f"  jitter≈{stats['corner_jitter_px']:.2f}px"
        f"  align_rej={stats['rejected_align']}"
    )
    H, rms, qc = calibrate_one(
        direction,
        last_ok_img if last_ok_img is not None else frames[-1],
        K,
        D,
        map1,
        map2,
        placement,
        cols,
        rows,
        square,
        scale,
        canvas,
        debug_prefix,
        svd_min_thresh=svd_min_thresh,
        corners=mean_corners,
    )
    if qc is not None:
        qc = dict(qc)
        qc["burst"] = stats
    return H, rms, qc


# ---------------- 实时抓拍模式 ----------------

def live_capture(config, calib, placements, cols, rows, square,
                 scale, canvas, args, params):
    ensure_gui_display()
    draft_path = getattr(args, "draft_output", DEFAULT_DRAFT_OUTPUT)
    extrinsic_balance = getattr(args, "extrinsic_balance", None)
    if extrinsic_balance is None:
        extrinsic_balance = args.balance
    stable_need = int(getattr(args, "stable_frames", STABLE_FRAMES_DEFAULT))
    burst_n = int(getattr(args, "burst_frames", BURST_FRAMES_DEFAULT))
    burst_min_ok = int(getattr(args, "burst_min_ok", BURST_MIN_OK_DEFAULT))
    print("=" * 60)
    print("  实时外参抓拍（稳定检出 → 多帧均值 → 求 H）")
    print(f"  稳定阈值: 连续检出 {stable_need} 次后 SPACE 才锁定该路")
    print(f"  连拍: {burst_n} 帧全分辨率，至少 {burst_min_ok} 帧检出后对角点取均值")
    print("  SPACE = 抓拍已 STABLE 且未锁定的路")
    print("  ESC/q = 存盘退出（已锁定几路就保存几路）")
    print("  1/2/3/4 = 解锁 front/back/left/right 重新抓拍")
    print("  0     = 解锁全部")
    print(f"  每次锁定后自动覆盖: {draft_path}")
    print(f"  预览: 多线程 + nvvidconv {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}；"
          f"每帧检 {LIVE_DETECT_PER_FRAME} 路")
    print("=" * 60)
    log_cuda_status()
    print("  [标定] 求 H 的 remap/角点检测固定 CPU，保证精度；UI resize 可走 GPU")
    print("=" * 60)

    workers = {}
    maps = {}
    device_index = {}

    def start_all_preview():
        stop_all_preview()
        for d in DIRECTIONS:
            if d not in calib or d not in placements:
                continue
            idx = config.get(d)
            if idx is None:
                print(f"  [跳过] {d}: camera_config 无此路")
                continue
            device_index[d] = int(idx)
            try:
                wkr = PreviewWorker(d, int(idx))
                wkr.start()
                workers[d] = wkr
                note = ""
                if wkr.width * wkr.height >= CAPTURE_WIDTH * CAPTURE_HEIGHT * 0.8:
                    note = "  [警告: 仍是全分辨率]"
                print(f"  {d}: /dev/video{idx} 预览 {wkr.width}x{wkr.height}  "
                      f"via {wkr.backend}  balance={extrinsic_balance:.2f}{note}")
            except Exception as e:
                print(f"  [错误] {d} (/dev/video{idx}): {e}")

    def stop_all_preview():
        for wkr in list(workers.values()):
            try:
                wkr.stop()
            except Exception:
                pass
        workers.clear()

    for d in DIRECTIONS:
        if d not in calib or d not in placements:
            continue
        if config.get(d) is None:
            continue
        m1, m2 = precompute_undistort_maps(
            calib[d]["K"], calib[d]["D"],
            CAPTURE_WIDTH, CAPTURE_HEIGHT, extrinsic_balance)
        maps[d] = (m1, m2)

    start_all_preview()
    if not workers:
        print("[错误] 没有可用相机。")
        return {}, {}, {}, {}

    homographies, rms_errors, images_undist, h_quality = {}, {}, {}, {}
    svd_min = getattr(args, "min_h_svd", H_SVD_MIN_DEFAULT)
    unlock_keys = {
        ord("1"): "front",
        ord("2"): "back",
        ord("3"): "left",
        ord("4"): "right",
    }
    detect_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    detect_rr = 0
    last_detected = {d: False for d in DIRECTIONS}
    stable_streak = {d: 0 for d in DIRECTIONS}
    disp_cache = {
        d: np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), np.uint8) for d in DIRECTIONS
    }
    status_cache = {d: "NO CAM" for d in DIRECTIONS}
    ui_fps = 0.0
    ui_n = 0
    ui_t0 = time.monotonic()

    def autosave_draft() -> None:
        if not homographies:
            return
        warnings = check_lr_symmetry(homographies, h_quality)
        save_results(
            homographies, rms_errors, placements, params, draft_path,
            h_quality=h_quality, global_warnings=warnings, overwrite=True)

    def unlock_direction(direction: str) -> None:
        homographies.pop(direction, None)
        rms_errors.pop(direction, None)
        images_undist.pop(direction, None)
        h_quality.pop(direction, None)
        last_detected[direction] = False
        stable_streak[direction] = 0
        status_cache[direction] = "NO BOARD"
        print(f"  [解锁] {direction}，下次 SPACE 可重新抓拍")
        autosave_draft()

    def unlock_all() -> None:
        homographies.clear()
        rms_errors.clear()
        images_undist.clear()
        h_quality.clear()
        for d in DIRECTIONS:
            last_detected[d] = False
            stable_streak[d] = 0
            if d in device_index:
                status_cache[d] = "NO BOARD"
        print("  [解锁] 全部四路，下次 SPACE 可重新抓拍")

    def make_locked_tile(d: str) -> np.ndarray:
        tile = disp_cache.get(d)
        if tile is None or tile.size == 0:
            tile = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), np.uint8)
        else:
            tile = tile.copy()
        if tile.shape[1] != DISPLAY_WIDTH or tile.shape[0] != DISPLAY_HEIGHT:
            tile = resize_bgr(tile, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.rectangle(tile, (0, 0), (DISPLAY_WIDTH - 1, DISPLAY_HEIGHT - 1), (0, 180, 0), 4)
        return tile

    def to_display_size(img):
        if img is None:
            return np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), np.uint8)
        if img.shape[1] == DISPLAY_WIDTH and img.shape[0] == DISPLAY_HEIGHT:
            return img
        return resize_bgr(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    def handle_key(key: int) -> str:
        if key in (27, ord("q")):
            return "quit"
        if key == ord("0"):
            unlock_all()
            return "continue"
        if key in unlock_keys and unlock_keys[key] in device_index:
            unlock_direction(unlock_keys[key])
            return "continue"
        return ""

    def blank_frames_and_labels():
        frames = {}
        labels = {}
        for d in DIRECTIONS:
            if d in homographies:
                frames[d] = make_locked_tile(d)
                status_cache[d] = (
                    f"LOCKED RMS={rms_errors[d]:.2f} "
                    f"{homography_qc_label(h_quality[d]) if d in h_quality else 'OK'}"
                )
            elif d in workers:
                frames[d] = to_display_size(disp_cache[d])
            else:
                frames[d] = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), np.uint8)
                status_cache[d] = "NO CAM"
            idx = device_index.get(d, -1)
            cam_fps = workers[d].fps if d in workers else 0.0
            labels[d] = (
                f"{d.upper()} v{idx} [{status_cache.get(d, '')}] "
                f"{cam_fps:.0f}fps"
            )
        return frames, labels

    while True:
        pending = [d for d in workers if d not in homographies]

        if not pending:
            frames, labels = blank_frames_and_labels()
            grid = build_grid(frames, labels)
            cv2.putText(grid, f"UI {ui_fps:.0f}fps | all locked",
                        (10, grid.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Extrinsic Calibration - Live (SPACE=抓拍 ESC=存盘)", grid)
            key = merge_wait_key(
                cv2.waitKey(LIVE_PREVIEW_WAIT_MS) & 0xFF, resolve_control_file()
            )
            ui_n += 1
            now = time.monotonic()
            if now - ui_t0 >= 1.0:
                ui_fps = ui_n / (now - ui_t0)
                ui_n = 0
                ui_t0 = now
            action = handle_key(key)
            if action == "quit":
                break
            if key == ord(" "):
                print("\n[提示] 四路均已锁定。按 1/2/3/4 解锁某路，或 0 解锁全部。")
            continue

        # 本帧只检测 1 路，避免 4 路 findChessboard 打满 CPU
        to_detect = []
        n_det = min(LIVE_DETECT_PER_FRAME, len(pending))
        for i in range(n_det):
            to_detect.append(pending[(detect_rr + i) % len(pending)])
        detect_rr = (detect_rr + max(n_det, 1)) % len(pending)

        frames = {}
        for d in DIRECTIONS:
            if d not in workers:
                frames[d] = to_display_size(disp_cache[d])
                status_cache[d] = "NO CAM"
                continue
            if d in homographies:
                frames[d] = make_locked_tile(d)
                status_cache[d] = (
                    f"LOCKED RMS={rms_errors[d]:.2f} "
                    f"{homography_qc_label(h_quality[d]) if d in h_quality else 'OK'}"
                )
                last_detected[d] = False
                stable_streak[d] = 0
                continue

            frame = workers[d].get()
            if frame is None:
                frames[d] = to_display_size(disp_cache[d])
                status_cache[d] = "DETECTED" if last_detected.get(d) else "NO BOARD"
                continue

            # 预览直接显示原图（不去畸变），保证跟手；检测也在小图原图上做
            disp = to_display_size(frame)
            if d in to_detect:
                g = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
                found_bool, corners = cv2.findChessboardCorners(
                    g, (cols, rows), detect_flags)
                ok_det = bool(found_bool and corners is not None)
                last_detected[d] = ok_det
                if ok_det:
                    stable_streak[d] = stable_streak.get(d, 0) + 1
                    cv2.drawChessboardCorners(disp, (cols, rows), corners, True)
                    st = stable_streak[d]
                    if st >= stable_need:
                        status_cache[d] = f"READY {st}"
                        cv2.rectangle(
                            disp, (4, 4),
                            (DISPLAY_WIDTH - 5, DISPLAY_HEIGHT - 5), (0, 255, 0), 3)
                    else:
                        status_cache[d] = f"STABLE {st}/{stable_need}"
                        cv2.rectangle(
                            disp, (4, 4),
                            (DISPLAY_WIDTH - 5, DISPLAY_HEIGHT - 5), (0, 200, 255), 2)
                else:
                    stable_streak[d] = 0
                    status_cache[d] = "NO BOARD"
            else:
                st = stable_streak.get(d, 0)
                if st >= stable_need:
                    status_cache[d] = f"READY {st}"
                    cv2.rectangle(
                        disp, (4, 4),
                        (DISPLAY_WIDTH - 5, DISPLAY_HEIGHT - 5), (0, 255, 0), 3)
                elif last_detected.get(d):
                    status_cache[d] = f"STABLE {st}/{stable_need}"
                    cv2.rectangle(
                        disp, (4, 4),
                        (DISPLAY_WIDTH - 5, DISPLAY_HEIGHT - 5), (0, 200, 255), 2)
                else:
                    status_cache[d] = "NO BOARD"

            disp_cache[d] = disp
            frames[d] = disp

        for d in DIRECTIONS:
            frames.setdefault(d, to_display_size(disp_cache[d]))

        labels = {}
        for d in DIRECTIONS:
            idx = device_index.get(d, -1)
            cam_fps = workers[d].fps if d in workers else 0.0
            labels[d] = (
                f"{d.upper()} v{idx} [{status_cache.get(d, '')}] "
                f"{cam_fps:.0f}fps"
            )

        grid = build_grid(frames, labels)
        cv2.putText(grid, f"UI {ui_fps:.0f}fps",
                    (10, grid.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Extrinsic Calibration - Live (SPACE=抓拍 ESC=存盘)", grid)
        key = merge_wait_key(
            cv2.waitKey(LIVE_PREVIEW_WAIT_MS) & 0xFF, resolve_control_file()
        )

        ui_n += 1
        now = time.monotonic()
        if now - ui_t0 >= 1.0:
            ui_fps = ui_n / (now - ui_t0)
            ui_n = 0
            ui_t0 = now

        action = handle_key(key)
        if action == "quit":
            break
        if action == "continue":
            continue
        if key != ord(" "):
            continue

        pending = [d for d in workers if d not in homographies]
        if not pending:
            print("\n[提示] 四路均已锁定。按 1/2/3/4 解锁某路，或 0 解锁全部。")
            continue

        ready = [d for d in pending if stable_streak.get(d, 0) >= stable_need]
        if not ready:
            # SPACE 时对未锁定路再快速确认一轮（不计入连拍，只提示）
            almost = [d for d in pending if last_detected.get(d)]
            streak_s = ", ".join(
                f"{d}:{stable_streak.get(d, 0)}" for d in pending
            )
            print(
                f"\n[提示] 需稳定检出 ≥{stable_need} 次再 SPACE。"
                f" 当前 streak=[{streak_s}]  近检={almost or '无'}"
            )
            continue

        print(
            f"\n[抓拍] 稳定就绪={ready} → 全分辨率连拍 {burst_n} 帧取角点均值"
        )
        stop_all_preview()
        for d in ready:
            idx = device_index[d]
            frames, fw, fh = grab_full_resolution_burst(idx, n_frames=burst_n)
            if not frames:
                print(f"  {d:8s}  全分辨率读帧失败，未锁定")
                stable_streak[d] = 0
                continue
            if (fw, fh) != (CAPTURE_WIDTH, CAPTURE_HEIGHT):
                print(f"  {d:8s}  警告: 实际 {fw}x{fh}，期望 "
                      f"{CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")
            dbg = os.path.join(args.debug_dir, d)
            H, rms, qc = calibrate_one_burst(
                d, frames, calib[d]["K"], calib[d]["D"],
                maps[d][0], maps[d][1], placements[d],
                cols, rows, square, scale, canvas, dbg,
                svd_min_thresh=svd_min, min_ok=burst_min_ok,
                balance=extrinsic_balance)
            if H is not None:
                homographies[d] = H
                rms_errors[d] = rms
                h_quality[d] = qc
                images_undist[d] = cv2.remap(
                    frames[-1], maps[d][0], maps[d][1], cv2.INTER_LINEAR)
                disp_cache[d] = to_display_size(images_undist[d])
                stable_streak[d] = 0
                print(f"  {d:8s}  已锁定  RMS={rms:.4f} px  {quality(rms)}")
                print_homography_qc(d, qc)
            else:
                print(f"  {d:8s}  检测失败，未锁定")
                stable_streak[d] = 0
        locked = [d for d in DIRECTIONS if d in homographies]
        print(f"  当前已锁定: {locked or '无'}\n")
        autosave_draft()
        print("  重新打开预览流…")
        start_all_preview()
        if not workers:
            print("[错误] 预览重开失败，退出。")
            break

    cv2.destroyAllWindows()
    stop_all_preview()
    print("  相机已释放。")
    return homographies, rms_errors, images_undist, h_quality



def build_grid(frames, labels):
    top = cv2.hconcat([frames["front"], frames["back"]])
    bottom = cv2.hconcat([frames["left"], frames["right"]])
    grid = cv2.vconcat([top, bottom])
    pos = {"front": (10, 30), "back": (DISPLAY_WIDTH + 10, 30),
           "left": (10, DISPLAY_HEIGHT + 30),
           "right": (DISPLAY_WIDTH + 10, DISPLAY_HEIGHT + 30)}
    for d, p in pos.items():
        cv2.putText(grid, labels.get(d, d), p,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return grid


# ---------------- 图像模式 ----------------

def image_mode(calib, placements, cols, rows, square,
               scale, canvas, args):
    if not os.path.isdir(args.images_dir):
        print(f"[错误] 图片目录不存在: {args.images_dir}")
        sys.exit(1)
    extrinsic_balance = getattr(args, "extrinsic_balance", None)
    if extrinsic_balance is None:
        extrinsic_balance = args.balance
    print("=" * 60)
    print(f"  从 {args.images_dir} 读图（front.jpg/back.jpg/left.jpg/right.jpg）")
    print(f"  balance={extrinsic_balance:.2f}")
    print("=" * 60)

    homographies, rms_errors, images_undist, h_quality = {}, {}, {}, {}
    svd_min = getattr(args, "min_h_svd", H_SVD_MIN_DEFAULT)
    for d in DIRECTIONS:
        if d not in calib or d not in placements:
            continue
        path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = os.path.join(args.images_dir, f"{d}{ext}")
            if os.path.isfile(p):
                path = p
                break
        if path is None:
            print(f"  [跳过] {d}: 未找到图片")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"  [错误] {d}: 读图失败 {path}")
            continue
        h_img, w_img = img.shape[:2]
        # 用实际图像分辨率创建去畸变 map
        m1, m2 = precompute_undistort_maps(
            calib[d]["K"], calib[d]["D"], w_img, h_img, extrinsic_balance)
        print(f"  {d}: {os.path.basename(path)}  ({w_img}x{h_img})")
        dbg = os.path.join(args.debug_dir, d)
        H, rms, qc = calibrate_one(
            d, img, calib[d]["K"], calib[d]["D"],
            m1, m2, placements[d],
            cols, rows, square, scale, canvas, dbg,
            svd_min_thresh=svd_min)
        if H is not None:
            homographies[d] = H
            rms_errors[d] = rms
            h_quality[d] = qc
            images_undist[d] = cv2.remap(
                img, m1, m2, cv2.INTER_LINEAR)
            print(f"          RMS={rms:.4f} px  {quality(rms)}")
            print_homography_qc(d, qc)
            if args.show_detection:
                disp = cv2.imread(dbg + "_undist_corners.jpg")
                if disp is not None:
                    cv2.imshow(f"Detection - {d}", cv2.resize(
                        disp, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
        else:
            print(f"          检测失败")
    if args.show_detection:
        print("  按任意键关闭检测窗口。")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return homographies, rms_errors, images_undist, h_quality


# ---------------- 结果保存 ----------------

def _merge_previous_extrinsics(path, params, homographies, rms_errors,
                               placements, h_quality):
    """把上次保存里、这次没重标的方向续上，避免只标几路就把其余路清空。

    仅当画布几何一致时才续用：H 是「去畸变图 → BEV 画布」的映射，
    scale / canvas / extrinsic_balance 任一变化都会让旧 H 失效。
    返回 (kept_dirs, stale_reason)。
    """
    if not os.path.exists(path):
        return [], None
    try:
        with open(path) as f:
            old = json.load(f)
    except Exception as exc:
        print(f"[注意] 旧外参 {path} 读取失败，本次不续用: {exc}")
        return [], "unreadable"

    old_h = old.get("homographies") or {}
    if not old_h:
        return [], None

    def _same(a, b, tol=1e-9):
        try:
            return abs(float(a) - float(b)) <= tol
        except (TypeError, ValueError):
            return False

    canvas_ok = list(old.get("canvas_size") or []) == [
        int(params["canvas"][0]), int(params["canvas"][1])
    ]
    if not (canvas_ok
            and _same(old.get("scale_px_per_meter"), params["scale"])
            and _same(old.get("extrinsic_balance"), params["extrinsic_balance"])):
        reason = (
            f"画布几何已变（旧 scale={old.get('scale_px_per_meter')} "
            f"canvas={old.get('canvas_size')} balance={old.get('extrinsic_balance')}）"
        )
        print(f"[注意] {reason}，旧外参不可续用，本次只保存已标定的方向")
        return [], reason

    old_rms = old.get("rms_errors") or {}
    old_place = old.get("placements") or {}
    old_qc = old.get("homography_qc") or {}
    kept = []
    for d in DIRECTIONS:
        if d in homographies or d not in old_h:
            continue
        homographies[d] = np.asarray(old_h[d], dtype=np.float64)
        if d in old_rms:
            rms_errors[d] = old_rms[d]
        if d in old_place and d not in placements:
            placements[d] = old_place[d]
        if d in old_qc:
            h_quality[d] = old_qc[d]
        kept.append(d)
    if kept:
        print(f"[合并] 沿用上次外参: {kept}")
    return kept, None


def save_results(homographies, rms_errors, placements, params, output_path,
                 h_quality=None, global_warnings=None, overwrite=False):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    actual_path = output_path
    if not overwrite and os.path.exists(output_path):
        # 正式结果不覆盖已有标定，自动递增编号
        base, ext = os.path.splitext(output_path)
        idx = 1
        while os.path.exists(f"{base}_{idx}{ext}"):
            idx += 1
        actual_path = f"{base}_{idx}{ext}"
        print(f"[注意] {output_path} 已存在，自动保存到 {actual_path}")

    result = {
        "pattern_size": list(params["pattern_size"]),
        "square_size_m": params["square_size_m"],
        "scale_px_per_meter": params["scale"],
        "canvas_size": list(params["canvas"]),
        "vehicle_center": [params["canvas"][0] / 2.0, params["canvas"][1] / 2.0],
        "balance": params["balance"],
        "extrinsic_balance": params["extrinsic_balance"],
        "placements": {d: placements[d] for d in DIRECTIONS if d in placements},
        "homographies": {d: H.tolist() for d, H in homographies.items()},
        "rms_errors": {d: float(e) for d, e in rms_errors.items()},
    }
    if h_quality:
        result["homography_qc"] = {d: qc for d, qc in h_quality.items()}
    if global_warnings:
        result["global_warnings"] = list(global_warnings)
    with open(actual_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[已保存] {actual_path}")
    return actual_path


def save_combined_preview(homographies, images_undist, canvas, out_path):
    """
    把各路去畸变图经 H warp 到同一画布，使用方向权重 + 增益补偿 + 多频段融合
    拼接成完整 BEV 总览图。
    """
    cw, ch = canvas
    cx, cy = cw / 2.0, ch / 2.0

    # 1) warp 各路到 BEV 画布
    bev_views = {}
    for d, H in homographies.items():
        if d not in images_undist:
            continue
        bev = warp_perspective_bgr(images_undist[d], H, canvas)
        bev_views[d] = bev

    if len(bev_views) < 2:
        # 只有 1 路，直接存
        for d, bev in bev_views.items():
            cv2.imwrite(out_path, bev)
            print(f"[已保存] BEV 总览 {out_path}")
            return

    # 2) 方向权重
    h, w = ch, cw
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    gx = (x_grid - cx).astype(np.float64)
    gy = (cy - y_grid).astype(np.float64)
    angle = np.arctan2(gy, gx)
    cam_angle = {"front": np.pi / 2.0, "back": -np.pi / 2.0,
                 "left": np.pi, "right": 0.0}

    raw_weights = {}
    for d, ca in cam_angle.items():
        if d not in bev_views:
            continue
        diff = np.arctan2(np.sin(angle - ca), np.cos(angle - ca))
        w_map = np.clip(np.cos(diff), 0.0, 1.0) ** 4.0
        raw_weights[d] = w_map

    # 乘以有效像素掩码
    for d in list(raw_weights.keys()):
        mask = (cv2.cvtColor(bev_views[d], cv2.COLOR_BGR2GRAY) > 0).astype(np.float64)
        raw_weights[d] *= mask

    weight_sum = np.zeros((h, w), dtype=np.float64)
    for w_map in raw_weights.values():
        weight_sum += w_map
    weight_sum = np.maximum(weight_sum, 1e-10)
    weights = {d: (raw_weights[d] / weight_sum).astype(np.float32) for d in raw_weights}

    # 3) 增益补偿
    # 在重叠区域计算亮度比
    masks = {d: (cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY) > 10) for d, bev in bev_views.items()}
    pairs = [("front", "left"), ("front", "right"), ("back", "left"), ("back", "right")]
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

    # 应用增益
    for d in bev_views:
        g = gains[d]
        if np.any(np.abs(g - 1.0) > 0.003):
            bev_views[d] = np.clip(
                bev_views[d].astype(np.float32) * g, 0, 255).astype(np.uint8)

    # 4) 多频段金字塔融合
    num_levels = 5
    laplacian_pyrs = {}
    for d, bev in bev_views.items():
        if d not in weights:
            continue
        # 高斯金字塔
        gp = [bev.astype(np.float64)]
        for _ in range(num_levels - 1):
            gp.append(cv2.pyrDown(gp[-1]).astype(np.float64))
        # 拉普拉斯金字塔
        lp = []
        for i in range(len(gp) - 1):
            expanded = cv2.pyrUp(gp[i + 1])
            hh, ww = gp[i].shape[:2]
            expanded = expanded[:hh, :ww]
            lp.append(gp[i] - expanded)
        lp.append(gp[-1])
        laplacian_pyrs[d] = lp

    # 权重高斯金字塔
    weight_pyrs = {}
    for d, w_map in weights.items():
        if d not in laplacian_pyrs:
            continue
        w_3ch = np.dstack([w_map] * 3)
        wp = [w_3ch.astype(np.float64)]
        for _ in range(num_levels - 1):
            wp.append(cv2.pyrDown(wp[-1]).astype(np.float64))
        weight_pyrs[d] = wp

    # 逐层融合
    blended_pyr = []
    for level in range(num_levels):
        h_l, w_l = laplacian_pyrs[list(laplacian_pyrs.keys())[0]][level].shape[:2]
        bl = np.zeros((h_l, w_l, 3), dtype=np.float64)
        ws = np.zeros((h_l, w_l, 3), dtype=np.float64)
        for d in laplacian_pyrs:
            bl += laplacian_pyrs[d][level] * weight_pyrs[d][level]
            ws += weight_pyrs[d][level]
        ws = np.maximum(ws, 1e-10)
        blended_pyr.append(bl / ws)

    # 重建
    result_img = blended_pyr[-1]
    for i in range(len(blended_pyr) - 2, -1, -1):
        result_img = cv2.pyrUp(result_img)
        hh, ww = blended_pyr[i].shape[:2]
        result_img = result_img[:hh, :ww] + blended_pyr[i]
    result_img = np.clip(result_img, 0, 255).astype(np.uint8)

    cv2.imwrite(out_path, result_img)
    print(f"[已保存] 4 路 BEV 拼接总览 {out_path}")


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="4 路鱼眼 AVM 外参标定：求每路去畸变图->统一 BEV 画布的单应性 H")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--capture", action="store_true",
                   help="实时抓拍模式（需显示器/X 转发）：开 4 路，SPACE 抓拍算 H")
    g.add_argument("--images-dir", default=None,
                   help="图片目录（无显示器兜底）：front.jpg/back.jpg/left.jpg/right.jpg")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help=f"camera_config.json（默认 {DEFAULT_CONFIG}）")
    ap.add_argument("--placements", default=DEFAULT_PLACEMENTS,
                    help=f"placement JSON（默认 {DEFAULT_PLACEMENTS}）")
    ap.add_argument("--near", default=None,
                    help="CLI 覆盖 near_m，如 front=0.5,back=0.5,left=0.5,right=0.5")
    ap.add_argument("--calib-dir", default=DEFAULT_CALIB_DIR,
                    help=f"内参结果目录（默认 {DEFAULT_CALIB_DIR}）")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"输出 extrinsics.json（默认 {DEFAULT_OUTPUT}）")
    ap.add_argument("--draft-output", default=DEFAULT_DRAFT_OUTPUT,
                    help=f"实时抓拍逐路锁定时的临时覆盖文件（默认 {DEFAULT_DRAFT_OUTPUT}）")
    ap.add_argument("--debug-dir", default=DEFAULT_CALIB_DIR,
                    help=f"调试图输出目录（默认 {DEFAULT_CALIB_DIR}）")
    ap.add_argument("--board-config", default=DEFAULT_BOARD_CONFIG,
                    help=f"棋盘规格 JSON（默认 {DEFAULT_BOARD_CONFIG}）")
    ap.add_argument("--pattern-size", default=None,
                    help="覆盖棋盘内角点 WxH（默认读 --board-config）")
    ap.add_argument("--square", type=float, default=None,
                    help="覆盖单格边长米（默认读 --board-config）")
    ap.add_argument("--scale", type=float, default=SCALE_PX_PER_M,
                    help=f"画布像素/米（默认 {SCALE_PX_PER_M}）")
    ap.add_argument("--canvas", type=int, nargs=2, default=list(CANVAS_SIZE),
                    metavar=("W", "H"), help=f"画布宽 高（默认 {CANVAS_SIZE[0]} {CANVAS_SIZE[1]}）")
    ap.add_argument("--balance", type=float, default=BALANCE,
                    help=f"去畸变 balance（默认 {BALANCE}）")
    ap.add_argument("--extrinsic-balance", type=float, default=None,
                    help=f"外参标定专用去畸变 balance（默认同 --balance）。"
                         "建议 0.7~0.9 以获得更完整的去畸变，改善 H 矩阵条件数")
    ap.add_argument("--show-detection", action="store_true",
                    help="图像模式弹窗显示检测结果（需显示器）")
    ap.add_argument("--preview", action="store_true",
                    help="标完存 4 路 BEV 总览图（写盘，无显示器也存）")
    ap.add_argument("--min-h-svd", type=float, default=H_SVD_MIN_DEFAULT,
                    help=f"H 2×2 最小奇异值下限（默认 {H_SVD_MIN_DEFAULT}，低于此报病态）")
    ap.add_argument(
        "--stable-frames", type=int, default=STABLE_FRAMES_DEFAULT,
        help=f"预览连续检出次数达到后才允许 SPACE 锁定（默认 {STABLE_FRAMES_DEFAULT}）",
    )
    ap.add_argument(
        "--burst-frames", type=int, default=BURST_FRAMES_DEFAULT,
        help=f"全分辨率连拍帧数，对角点取均值再求 H（默认 {BURST_FRAMES_DEFAULT}）",
    )
    ap.add_argument(
        "--burst-min-ok", type=int, default=BURST_MIN_OK_DEFAULT,
        help=f"连拍中至少成功检出帧数（默认 {BURST_MIN_OK_DEFAULT}）",
    )
    args = ap.parse_args()
    if args.stable_frames < 1:
        ap.error("--stable-frames 必须 ≥ 1")
    if args.burst_frames < 1:
        ap.error("--burst-frames 必须 ≥ 1")
    if args.burst_min_ok < 1 or args.burst_min_ok > args.burst_frames:
        ap.error("--burst-min-ok 必须在 [1, burst-frames] 内")

    board = resolve_board_args(args.pattern_size, args.square, args.board_config)
    cols, rows = board["pattern_size"]
    pattern_size = board["pattern_size"]
    square_m = board["square_size_m"]
    canvas = tuple(args.canvas)
    os.makedirs(args.debug_dir, exist_ok=True)
    print(f"[棋盘] {board['pattern_size_str']} 内角点, square={square_m}m"
          f"  ← {board['path']}"
          + ("" if board["from_config"] else "（含 CLI 覆盖）"))

    if not os.path.isdir(args.calib_dir):
        print(f"[错误] 内参目录不存在: {args.calib_dir}（先跑 calibrate_intrinsics.py）")
        sys.exit(1)

    # 1) 内参
    print("=" * 60)
    print("  加载内参标定结果...")
    print("=" * 60)
    calib = {}
    for d in DIRECTIONS:
        try:
            K, D, rms = load_calib(d, args.calib_dir)
            calib[d] = {"K": K, "D": D, "rms": rms}
            intr_qc = ""
            if rms is not None and float(rms) > INTRINSIC_RMS_WARN:
                intr_qc = f"  ⚠️ 内参 RMS>{INTRINSIC_RMS_WARN}px"
            print(f"  {d:8s}  RMS={rms:.4f} px  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}"
                  f"  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}{intr_qc}")
        except FileNotFoundError:
            print(f"  [跳过] {d}: 无内参文件")
        except Exception as e:
            print(f"  [错误] {d}: {e}")
    if not calib:
        print("[错误] 没有内参数据，退出。")
        sys.exit(1)

    # 2) placement
    placements = load_placements(args.placements, cols, rows, square_m)
    if args.near:
        override = parse_near_cli(args.near)
        for d, near in override.items():
            e = placements[d]
            corners, (w, dep) = ground_corners(
                d, near, e["lateral_m"], e["orient"], cols, rows, square_m)
            placements[d] = {**e, "near_m": near,
                             "board_w_m": w, "board_h_m": dep,
                             "ground_corners_m": [[float(x), float(y)] for (x, y) in corners]}

    print("\n" + "=" * 60)
    print(f"  placement（板 {cols}x{rows} 内角点, square={square_m}m, "
          f"orient=long-lateral）")
    print("=" * 60)
    for d in DIRECTIONS:
        if d not in placements:
            continue
        p = placements[d]
        print(f"  {d:8s}  near={p['near_m']}m  lateral={p['lateral_m']}m  "
              f"板 {p['board_w_m']}x{p['board_h_m']}m  "
              f"角点(米)={p['ground_corners_m']}")
    print(f"  画布: {canvas[0]}x{canvas[1]}  scale={args.scale}px/m  "
          f"中心=车体 ({canvas[0]/2:.0f},{canvas[1]/2:.0f})")

    # 去畸变 map 现由 live_capture/image_mode 按实际分辨率创建
    extrinsic_balance = args.extrinsic_balance if args.extrinsic_balance is not None else args.balance
    print(f"\n  外参标定去畸变 balance={extrinsic_balance:.2f}"
          + (" (CLI --extrinsic-balance)" if args.extrinsic_balance is not None
             else " (同 --balance)"))

    # 4) 标定
    params = {"pattern_size": pattern_size, "square_size_m": square_m,
              "scale": args.scale, "canvas": canvas, "balance": args.balance,
              "extrinsic_balance": extrinsic_balance}
    if args.capture:
        config = load_config(args.config)
        homographies, rms_errors, images_undist, h_quality = live_capture(
            config, calib, placements, cols, rows, square_m,
            args.scale, canvas, args, params)
    else:
        homographies, rms_errors, images_undist, h_quality = image_mode(
            calib, placements, cols, rows, square_m,
            args.scale, canvas, args)

    if not homographies:
        print("[错误] 没算出任何 H。")
        sys.exit(1)

    # 只标了部分方向时，其余方向沿用上次结果，不要被清空
    _merge_previous_extrinsics(
        args.output, params, homographies, rms_errors, placements, h_quality)

    global_warnings = check_lr_symmetry(homographies, h_quality)
    for msg in global_warnings:
        print(f"[全局] ⚠️  {msg}")

    # 5) 保存（覆盖写出，避免每次多生成 _1/_2...）
    saved_path = save_results(
        homographies, rms_errors, placements, params, args.output,
        h_quality=h_quality, global_warnings=global_warnings, overwrite=True)

    # 6) 4 路 BEV 拼接总览（写盘，无显示器也存）
    if args.preview or args.images_dir or args.capture:
        if images_undist:
            prev_path = os.path.join(args.debug_dir, "extrinsics_bev_preview.jpg")
            save_combined_preview(homographies, images_undist, canvas, prev_path)
        else:
            print("  [提示] 无可用去畸变帧, 跳过 BEV 总览。")

    # 7) 汇总
    print("\n" + "=" * 60)
    print("  外参标定汇总")
    print("=" * 60)
    for d in DIRECTIONS:
        if d in homographies:
            line = f"  {d:8s}  RMS={rms_errors[d]:.4f} px  {quality(rms_errors[d])}"
            if d in h_quality:
                line += f"  |  {homography_qc_label(h_quality[d])}"
            print(line)
        else:
            print(f"  {d:8s}  失败/未标")
    if global_warnings:
        print("  --- 全局 ---")
        for msg in global_warnings:
            print(f"  ⚠️  {msg}")
    print("=" * 60)
    print(f"  结果: {saved_path}")
    print(f"  调试图: {args.debug_dir}/{{front,back,left,right}}_undist_corners.jpg")
    if args.preview or args.images_dir or args.capture:
        print(f"  BEV 总览: {args.debug_dir}/extrinsics_bev_preview.jpg")
    print("=" * 60)


if __name__ == "__main__":
    main()
