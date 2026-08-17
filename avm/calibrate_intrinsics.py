#!/usr/bin/env python3
"""
Camera Intrinsic Calibration Script for 4-Fisheye AVM System.

Calibrates each fisheye camera using a checkerboard, producing intrinsic
matrices and distortion coefficients compatible with the Kannala-Brandt
model used in src/avm.cpp.

Usage:
    python3 scripts/calibrate_intrinsics.py --preview
    python3 scripts/calibrate_intrinsics.py --calibrate
    python3 scripts/calibrate_intrinsics.py --calibrate --direction front
    python3 scripts/calibrate_intrinsics.py --calibrate --images-dir calib_images/
"""

import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import cv2

from cuda_cv import log_cuda_status, resize_bgr

DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "config", "camera_config.json")
DEFAULT_IMAGES_DIR = os.path.join(PROJECT_DIR, "calib_images")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "calib_results")
CAPTURE_WIDTH, CAPTURE_HEIGHT = __import__(
    "avm.camera_io", fromlist=["capture_size"]
).capture_size()
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480
MIN_CAPTURE_FRAMES = 15
TARGET_CAPTURE_FRAMES = 25
DETECT_SCALE = 0.33

# Allow `python3 scripts/calibrate_intrinsics.py` imports of sibling modules
# (SCRIPT_DIR already on path above)
from board_config import DEFAULT_BOARD_CONFIG, resolve_board_args  # noqa: E402
from remote_control import merge_wait_key, resolve_control_file  # noqa: E402


def load_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    valid = {"left", "back", "front", "right"}
    for k in cfg:
        if k not in valid:
            raise ValueError(f"Unexpected direction '{k}' in config. Expected one of {valid}")
    return cfg


def build_gst_camera_pipeline(index, width, height):
    device = f"/dev/video{index}"
    return (
        f"v4l2src device={device} io-mode=2 do-timestamp=true ! "
        f"video/x-raw,format=YUY2,width={CAPTURE_WIDTH},height={CAPTURE_HEIGHT} ! "
        f"nvvidconv ! video/x-raw,format=BGRx,width={width},height={height} ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def open_camera(index, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT):
    pipe = build_gst_camera_pipeline(index, width, height)
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        backend = cv2.CAP_V4L2
        cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{index}. Check permissions (try: sudo usermod -aG video $USER)")
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


def open_preview_camera(index):
    return open_camera(index, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)


def detect_corners(gray, pattern_size, scale=1.0):
    try:
        from detect_board_hires import find_board_corners
    except ImportError:
        from avm.detect_board_hires import find_board_corners

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    work = gray
    if scale < 1.0:
        work = cv2.resize(
            gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    corners = find_board_corners(work, pattern_size, use_sb=True, photo_retry=True)
    if corners is None:
        return False, None
    corners = np.asarray(corners, dtype=np.float32)
    if scale < 1.0:
        corners = corners / np.float32(scale)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    return True, corners


def build_object_points(pattern_size, square_size):
    # OpenCV fisheye 样例格式 (1, N, 3)；_normalize_calib_points 会再统一形状
    objp = np.zeros((1, pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= np.float32(square_size)
    return objp


def log_opencv_status():
    """Print which OpenCV is loaded (prefer: source scripts/env_opencv_cuda.sh)."""
    ver = cv2.__version__
    path = (getattr(cv2, "__file__", "") or "").replace("\\", "/")
    print(f"[OpenCV] {ver}  ← {path}")


def fisheye_calib_flags(include_check_cond=True):
    """OpenCV4: cv2.fisheye.CALIB_*；OpenCV5: 同名常量移到 cv2.CALIB_*。"""
    def _flag(name):
        if hasattr(cv2.fisheye, name):
            return getattr(cv2.fisheye, name)
        if hasattr(cv2, name):
            return getattr(cv2, name)
        raise AttributeError(f"OpenCV 缺少鱼眼标定 flag: {name}")
    flags = _flag("CALIB_RECOMPUTE_EXTRINSIC") + _flag("CALIB_FIX_SKEW")
    if include_check_cond:
        flags += _flag("CALIB_CHECK_COND")
    return flags


def _normalize_calib_points(obj_points, img_points):
    """统一为 OpenCV fisheye.calibrate 可接受的 (N,1,3)/(N,1,2) float64 列表。"""
    obj_out, img_out = [], []
    for o, p in zip(obj_points, img_points):
        o = np.asarray(o, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        if o.ndim == 2 and o.shape[1] == 3:
            o = o.reshape(-1, 1, 3)
        elif o.ndim == 3 and o.shape[0] == 1 and o.shape[2] == 3:
            # 旧格式 (1,N,3)
            o = o.reshape(1, -1, 3).transpose(1, 0, 2).copy()
        else:
            o = o.reshape(-1, 1, 3)
        p = p.reshape(-1, 1, 2)
        if o.shape[0] != p.shape[0]:
            raise ValueError(f"object/image 点数不一致: {o.shape[0]} vs {p.shape[0]}")
        obj_out.append(o)
        img_out.append(p)
    return obj_out, img_out


def calibrate_camera(obj_points, img_points, image_size):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    K = np.eye(3, dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    obj_points, img_points = _normalize_calib_points(obj_points, img_points)
    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_points, img_points, image_size, K, D,
            flags=fisheye_calib_flags(True), criteria=criteria
        )
    except cv2.error as e:
        # CHECK_COND 在姿态不好时易抛异常；去掉后再试一次
        print(f"  [WARN] 带 CALIB_CHECK_COND 标定失败 ({e})，去掉该 flag 重试...")
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_points, img_points, image_size, K, D,
            flags=fisheye_calib_flags(False), criteria=criteria
        )
    # 统一成 1D，方便写 JSON / 外参脚本读取
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    K = np.asarray(K, dtype=np.float64)
    return K, D, float(rms), rvecs, tvecs


def fit_inverse_polynomial(D, max_theta=1.5, num_samples=1000):
    theta = np.linspace(0.0, max_theta, num_samples)
    theta_2 = theta * theta
    theta_3 = theta_2 * theta
    theta_5 = theta_3 * theta_2
    theta_7 = theta_5 * theta_2
    theta_9 = theta_7 * theta_2
    theta_d = theta + D[0] * theta_3 + D[1] * theta_5 + D[2] * theta_7 + D[3] * theta_9

    theta_d_2 = theta_d * theta_d
    theta_d_3 = theta_d_2 * theta_d
    theta_d_5 = theta_d_3 * theta_d_2
    theta_d_7 = theta_d_5 * theta_d_2
    theta_d_9 = theta_d_7 * theta_d_2

    A = np.column_stack([theta_d_3, theta_d_5, theta_d_7, theta_d_9])
    b = theta - theta_d
    D_inv, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    D_inv = D_inv.flatten()

    fitted = theta_d + D_inv[0] * theta_d_3 + D_inv[1] * theta_d_5 + D_inv[2] * theta_d_7 + D_inv[3] * theta_d_9
    max_err = np.max(np.abs(fitted - theta))
    return D_inv, max_err


# ---------------------------------------------------------------------------
#  内参质量评估
# ---------------------------------------------------------------------------

# 阈值（可调）
INTRINSIC_RMS_PASS_PX = 0.8          # 整体 RMS 合格线 (px)
INTRINSIC_RMS_FAIL_PX = 1.5          # 整体 RMS 不合格线 (px)
INTRINSIC_PER_VIEW_RMS_WARN = 2.0    # 单帧 RMS 警告线 (px)
INTRINSIC_PER_VIEW_OUTLIER = 3.5     # 单帧 RMS 异常值线 (px)
INTRINSIC_D_MAX_ABS = 0.35           # 畸变系数绝对值上限（鱼眼典型 < 0.3）
INTRINSIC_K_ASPECT_RATIO_MAX = 1.15  # fx/fy 比值上限
INTRINSIC_K_CENTER_TOL = 0.12        # 主点偏离图像中心容忍度（相对比例）
INTRINSIC_MIN_VALID_VIEWS = 12       # 最少有效帧数


def compute_per_view_errors(obj_points, img_points, K, D, rvecs, tvecs):
    """计算每帧的重投影误差 RMS (px)，返回 list[float] 及异常帧索引。"""
    per_view = []
    for i, (op, ip, rv, tv) in enumerate(zip(obj_points, img_points, rvecs, tvecs)):
        projected, _ = cv2.fisheye.projectPoints(
            op.reshape(-1, 1, 3), rv, tv, K, D.reshape(-1, 1))
        err = np.sqrt(np.mean(np.sum((projected.reshape(-1, 2)
                                      - ip.reshape(-1, 2)) ** 2, axis=1)))
        per_view.append(float(err))
    return per_view


def check_k_sanity(K, image_size):
    """检查 K 矩阵的物理合理性。返回 (ok, warnings)。"""
    w, h = image_size
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    warnings = []

    aspect_ratio = max(fx, fy) / max(min(fx, fy), 1e-9)
    if aspect_ratio > INTRINSIC_K_ASPECT_RATIO_MAX:
        warnings.append(
            f"fx/fy 比值={aspect_ratio:.2f}>{INTRINSIC_K_ASPECT_RATIO_MAX}: "
            "像素纵横比异常，可能标定不足")

    cx_dev = abs(cx - w / 2.0) / w
    cy_dev = abs(cy - h / 2.0) / h
    if cx_dev > INTRINSIC_K_CENTER_TOL:
        warnings.append(
            f"cx={cx:.1f} 偏离图像中心 {cx_dev*100:.1f}%"
            f">{INTRINSIC_K_CENTER_TOL*100:.0f}%: 检查相机安装/遮罩")
    if cy_dev > INTRINSIC_K_CENTER_TOL:
        warnings.append(
            f"cy={cy:.1f} 偏离图像中心 {cy_dev*100:.1f}%"
            f">{INTRINSIC_K_CENTER_TOL*100:.0f}%: 检查相机安装/遮罩")

    if fx < w * 0.15 or fx > w * 0.8:
        warnings.append(f"fx={fx:.1f} 超出预期范围 [{w*0.15:.0f}, {w*0.8:.0f}]")
    if fy < h * 0.15 or fy > h * 0.8:
        warnings.append(f"fy={fy:.1f} 超出预期范围 [{h*0.15:.0f}, {h*0.8:.0f}]")

    return len(warnings) == 0 or all("偏离图像中心" in w for w in warnings), warnings


def check_distortion_sanity(D):
    """检查畸变系数的物理合理性。"""
    warnings = []
    for i, d in enumerate(D.flat):
        if abs(d) > INTRINSIC_D_MAX_ABS:
            warnings.append(
                f"D[{i}]={d:.4f} 绝对值>{INTRINSIC_D_MAX_ABS}: "
                "畸变系数过大，可能标定失败或棋盘姿态不足")
    # 鱼眼典型 D[0] > 0 (桶形畸变)
    if D.flat[0] < -0.02:
        warnings.append(
            f"D[0]={D.flat[0]:.4f}<0: 不典型的枕形畸变，"
            "检查是否用错了棋盘格方向或标定板")
    return len(warnings) == 0, warnings


def check_fov_coverage(img_points, image_size):
    """检查角点是否覆盖了足够的 FOV 区域。返回 (ok, info)。"""
    w, h = image_size
    all_pts = np.vstack([p.reshape(-1, 2) for p in img_points])

    # 将图像分成 3×3 网格，检查每个区域是否有点
    grid_hits = np.zeros((3, 3), dtype=bool)
    for pt in all_pts:
        gx = min(int(pt[0] / w * 3), 2)
        gy = min(int(pt[1] / h * 3), 2)
        grid_hits[gy, gx] = True

    covered = int(grid_hits.sum())
    total = 9
    info = {
        "grid_coverage": f"{covered}/{total}",
        "grid_hits": grid_hits.tolist(),
        "total_points": int(len(all_pts)),
    }

    # 角落覆盖（四角至少各有一个点）
    corners_covered = all([
        np.any((all_pts[:, 0] < w * 0.2) & (all_pts[:, 1] < h * 0.2)),  # TL
        np.any((all_pts[:, 0] > w * 0.8) & (all_pts[:, 1] < h * 0.2)),  # TR
        np.any((all_pts[:, 0] < w * 0.2) & (all_pts[:, 1] > h * 0.8)),  # BL
        np.any((all_pts[:, 0] > w * 0.8) & (all_pts[:, 1] > h * 0.8)),  # BR
    ])
    info["corners_covered"] = corners_covered

    ok = covered >= 5 and corners_covered
    if not ok:
        missing = []
        if covered < 5:
            missing.append(f"仅覆盖 {covered}/9 个网格区域")
        if not corners_covered:
            missing.append("四角覆盖不足")
        info["warnings"] = missing

    return ok, info


def evaluate_intrinsics(K, D, rms, image_size, obj_points, img_points,
                        rvecs=None, tvecs=None, per_view_errors=None):
    """
    综合评估内参标定质量。

    返回:
        dict with keys:
          status: "pass" | "warn" | "fail"
          overall_rms: float
          per_view_rms: list[float]
          outlier_views: list[int]  (异常帧索引)
          k_sanity: dict
          distortion_sanity: dict
          fov_coverage: dict
          warnings: list[str]
          passed: bool  (status != "fail")
    """
    warnings = []
    report = {
        "status": "pass",
        "overall_rms": float(rms),
        "per_view_rms": [],
        "outlier_views": [],
        "k_sanity": {},
        "distortion_sanity": {},
        "fov_coverage": {},
        "warnings": warnings,
        "passed": True,
    }

    # 1) 整体 RMS
    if rms > INTRINSIC_RMS_FAIL_PX:
        report["status"] = "fail"
        report["passed"] = False
        warnings.append(
            f"整体 RMS={rms:.3f}px > {INTRINSIC_RMS_FAIL_PX}px: 标定不合格")
    elif rms > INTRINSIC_RMS_PASS_PX:
        if report["status"] == "pass":
            report["status"] = "warn"
        warnings.append(
            f"整体 RMS={rms:.3f}px > {INTRINSIC_RMS_PASS_PX}px: 精度偏低")

    # 2) 逐帧误差
    if per_view_errors is None and obj_points is not None and img_points is not None:
        per_view_errors = compute_per_view_errors(
            obj_points, img_points, K, D, rvecs, tvecs)
    if per_view_errors:
        report["per_view_rms"] = [float(e) for e in per_view_errors]
        outlier_views = [
            i for i, e in enumerate(per_view_errors)
            if e > INTRINSIC_PER_VIEW_OUTLIER
        ]
        report["outlier_views"] = outlier_views
        if len(outlier_views) > len(per_view_errors) * 0.2:
            if report["status"] == "pass":
                report["status"] = "warn"
            warnings.append(
                f"{len(outlier_views)}/{len(per_view_errors)} 帧异常"
                f"(RMS>{INTRINSIC_PER_VIEW_OUTLIER}px): "
                "建议删除这些帧后重新标定")
        warn_views = [
            i for i, e in enumerate(per_view_errors)
            if INTRINSIC_PER_VIEW_RMS_WARN < e <= INTRINSIC_PER_VIEW_OUTLIER
        ]
        if warn_views:
            warnings.append(
                f"{len(warn_views)} 帧 RMS 偏高"
                f"({INTRINSIC_PER_VIEW_RMS_WARN}~{INTRINSIC_PER_VIEW_OUTLIER}px)")

    # 3) K 矩阵合理性
    k_ok, k_warnings = check_k_sanity(K, image_size)
    report["k_sanity"] = {"ok": k_ok, "warnings": k_warnings}
    if not k_ok:
        if report["status"] == "pass":
            report["status"] = "warn"
        warnings.extend(k_warnings)

    # 4) 畸变系数合理性
    d_ok, d_warnings = check_distortion_sanity(D)
    report["distortion_sanity"] = {"ok": d_ok, "warnings": d_warnings}
    if not d_ok:
        report["status"] = "fail"
        report["passed"] = False
        warnings.extend(d_warnings)

    # 5) FOV 覆盖率
    if img_points is not None and len(img_points) > 0:
        fov_ok, fov_info = check_fov_coverage(img_points, image_size)
        report["fov_coverage"] = fov_info
        if not fov_ok:
            if report["status"] == "pass":
                report["status"] = "warn"
            warnings.extend(fov_info.get("warnings", []))

    # 6) 帧数检查
    if img_points is not None and len(img_points) < INTRINSIC_MIN_VALID_VIEWS:
        if report["status"] == "pass":
            report["status"] = "warn"
        warnings.append(
            f"仅 {len(img_points)} 帧 < {INTRINSIC_MIN_VALID_VIEWS} 帧: "
            "推荐至少 15 帧覆盖不同姿态")

    return report


def print_evaluation_report(report, direction=""):
    """打印评估报告。"""
    label = f" [{direction}]" if direction else ""
    status_icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
    icon = status_icon.get(report["status"], "?")

    print(f"\n{'='*60}")
    print(f"  内参评估报告{label}  {icon} {report['status'].upper()}")
    print(f"{'='*60}")
    print(f"  整体 RMS:         {report['overall_rms']:.4f} px")

    if report["per_view_rms"]:
        pv = report["per_view_rms"]
        print(f"  逐帧 RMS:         min={min(pv):.3f}  max={max(pv):.3f}  "
              f"median={np.median(pv):.3f} px")
        if report["outlier_views"]:
            print(f"  异常帧:           {report['outlier_views']}")

    ks = report.get("k_sanity", {})
    if ks.get("warnings"):
        for w in ks["warnings"]:
            print(f"  K 矩阵:           ⚠️  {w}")

    ds = report.get("distortion_sanity", {})
    if ds.get("warnings"):
        for w in ds["warnings"]:
            print(f"  畸变系数:         ⚠️  {w}")

    fov = report.get("fov_coverage", {})
    if fov:
        print(f"  FOV 覆盖率:       {fov.get('grid_coverage', '?')}/9 网格"
              f"  四角={'✅' if fov.get('corners_covered') else '❌'}")
        if fov.get("warnings"):
            for w in fov["warnings"]:
                print(f"                    ⚠️  {w}")

    for w in report["warnings"]:
        # 避免重复打印（已在具体类别中打印的）
        pass

    if report["warnings"]:
        print(f"  --- 所有警告 ---")
        for w in report["warnings"]:
            print(f"  ⚠️  {w}")

    if report["passed"]:
        print(f"  结论:             合格，可进入外参标定")
    else:
        print(f"  结论:             不合格，建议重新采集标定图像")
    print(f"{'='*60}\n")


def evaluate_existing(calib_dir, directions=None):
    """评估已有的内参标定结果。返回 {direction: report}。"""
    if directions is None:
        directions = ["front", "back", "left", "right"]

    results = {}
    for d in directions:
        path = os.path.join(calib_dir, f"{d}.json")
        if not os.path.isfile(path):
            print(f"[跳过] {d}: 无内参文件 {path}")
            continue
        with open(path, "r") as f:
            data = json.load(f)
        K = np.array(data["K"], dtype=np.float64)
        D = np.array(data["D"], dtype=np.float64)
        rms = data.get("rms", data.get("reproj_error", 999))

        # 从已有结果评估（无原始角点数据，仅做 K/D 合理性检查）
        isize = data.get("image_size") or data.get("img_size")
        if isize and len(isize) >= 2:
            image_size = (int(isize[0]), int(isize[1]))
        else:
            image_size = (CAPTURE_WIDTH, CAPTURE_HEIGHT)
        report = evaluate_intrinsics(
            K, D, rms, image_size,
            obj_points=None, img_points=None,
            rvecs=None, tvecs=None, per_view_errors=None)
        print_evaluation_report(report, d)
        results[d] = report

    return results


def print_cpp_snippet(direction, K, D, D_inv):
    print(f"\n{'='*60}")
    print(f"  C++ snippet for {direction.upper()} camera")
    print(f"  Copy-paste into src/avm.cpp")
    print(f"{'='*60}")
    print(f"  // --- {direction} camera ---")
    print(f"  // g_intrinsic")
    print(f"  g_intrinsic = (cv::Mat_<float>(3, 3) << "
          f"{K[0,0]:.8f}f, 0.0f, {K[0,2]:.8f}f,"
          f" 0.0f, {K[1,1]:.8f}f, {K[1,2]:.8f}f, 0.0f, 0.0f, 1.0f);")
    print(f"  // m_undis2fish_params (KB forward coefficients)")
    print(f"  m_undis2fish_params = {{ {D[0]:.8f}, {D[1]:.8f}, {D[2]:.8f}, {D[3]:.8f} }};")
    print(f"  // g_fish2undis_params (approximate inverse)")
    print(f"  g_fish2undis_params = {{ {D_inv[0]:.8f}, {D_inv[1]:.8f}, {D_inv[2]:.8f}, {D_inv[3]:.8f} }};")
    print(f"{'='*60}\n")


def save_results(direction, K, D, D_inv, rms, output_dir, evaluation=None):
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "K": K.tolist(),
        "D": D.tolist(),
        "D_inv": D_inv.tolist(),
        "rms": float(rms),
    }
    if evaluation is not None:
        result["evaluation"] = {
            "status": evaluation["status"],
            "passed": evaluation["passed"],
            "per_view_rms": evaluation.get("per_view_rms", []),
            "outlier_views": evaluation.get("outlier_views", []),
            "warnings": evaluation["warnings"],
        }
    path = os.path.join(output_dir, f"{direction}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[SAVED] {path}")


def save_undistorted_sample(direction, img, K, D, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    h, w = img.shape[:2]
    new_K = K.copy()
    new_K[0, 0] *= 0.5
    new_K[1, 1] *= 0.5
    new_K[0, 2] = w * 0.6
    new_K[1, 2] = h * 0.6
    undis_w = int(w * 1.2)
    undis_h = int(h * 1.2)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (undis_w, undis_h), cv2.CV_16SC2
    )
    undis = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    path = os.path.join(output_dir, f"{direction}_undis.jpg")
    cv2.imwrite(path, undis)
    print(f"[SAVED] {path}")


def interactive_capture(cap, direction, pattern_size, square_size, images_dir, camera_idx=None):
    os.makedirs(images_dir, exist_ok=True)
    objp = build_object_points(pattern_size, square_size)
    obj_points_list = []
    img_points_list = []
    captured = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        width, height = CAPTURE_WIDTH, CAPTURE_HEIGHT

    idx_label = f" /dev/video{camera_idx}" if camera_idx is not None else ""
    print(f"\n{'='*60}")
    print(f"  Capturing calibration images for: {direction.upper()}{idx_label}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Target: {TARGET_CAPTURE_FRAMES} frames, minimum: {MIN_CAPTURE_FRAMES}")
    print(f"  Controls:")
    print(f"    SPACE  - capture current frame")
    print(f"    ESC/q  - finish capturing (enough frames)")
    print(f"    s      - skip this frame (detection failed)")
    control_file = resolve_control_file()
    if control_file:
        print(f"    remote - {control_file}（网页按钮，无需点窗口）")
    print(f"{'='*60}\n")

    cooldown = 0
    while captured < TARGET_CAPTURE_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame, retrying...")
            time.sleep(0.1)
            continue

        if cooldown > 0:
            cooldown -= 1

        display = frame.copy()
        if cooldown == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = detect_corners(gray, pattern_size, scale=DETECT_SCALE)
        else:
            found = False
            corners = None

        if found:
            cv2.drawChessboardCorners(display, pattern_size, corners, True)
            status = f"  [DETECTED] {captured}/{TARGET_CAPTURE_FRAMES} frames"
            if captured >= MIN_CAPTURE_FRAMES:
                status += " (ready to finish)"
            color = (0, 255, 0)
        else:
            status = f"  [NO BOARD] {captured}/{TARGET_CAPTURE_FRAMES} frames"
            color = (0, 0, 255)

        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, f"  {direction.upper()} camera", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if cooldown == 0:
            cv2.imshow(f"Calibration - {direction}", display)
        key = merge_wait_key(cv2.waitKey(30) & 0xFF, control_file)

        if key == 27 or key == ord("q"):
            if captured >= MIN_CAPTURE_FRAMES:
                break
            print("[WARN] Not enough frames yet, keep capturing...")
        elif key == ord(" ") and found and cooldown == 0:
            obj_points_list.append(objp)
            img_points_list.append(corners)
            path = os.path.join(images_dir, f"{captured:04d}.jpg")
            cv2.imwrite(path, frame)
            captured += 1
            cooldown = 15
            print(f"  [{captured}/{TARGET_CAPTURE_FRAMES}] Captured: {path}")
        elif key == ord("s"):
            print("  [SKIP] Frame skipped")

    cv2.destroyAllWindows()
    print(f"\n  Captured {captured} frames for {direction}.\n")
    return obj_points_list, img_points_list, (width, height)


def preview_cameras(config):
    print("[PREVIEW] Opening all 4 cameras...")
    caps = {}
    for direction, idx in config.items():
        try:
            cap, w, h = open_preview_camera(idx)
            caps[direction] = (cap, w, h, idx)
            print(f"  {direction}: /dev/video{idx} -> {w}x{h}")
        except Exception as e:
            print(f"  [ERROR] {direction} (/dev/video{idx}): {e}")

    if not caps:
        print("[ERROR] No cameras available.")
        return

    print("\n  Press ESC/q to exit preview.\n")
    control_file = resolve_control_file()
    if control_file:
        print(f"  Remote control file: {control_file}\n")
    while True:
        for direction, (cap, w, h, idx) in list(caps.items()):
            ret, frame = cap.read()
            if not ret:
                continue
            display = frame
            h_frame, w_frame = frame.shape[:2]
            if w_frame != PREVIEW_WIDTH or h_frame != PREVIEW_HEIGHT:
                display = resize_bgr(
                    frame, (PREVIEW_WIDTH, int(PREVIEW_WIDTH * h_frame / w_frame))
                )
            label = f"{direction.upper()}  /dev/video{idx}"
            cv2.putText(display, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow(f"Preview - {direction}", display)
        key = merge_wait_key(cv2.waitKey(30) & 0xFF, control_file)
        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()
    for cap, _, _, _ in caps.values():
        cap.release()


def calibrate_from_images(config, images_dir, output_dir, pattern_size, square_size, directions=None):
    objp = build_object_points(pattern_size, square_size)
    if directions is None:
        directions = list(config.keys())

    results = {}
    for direction in directions:
        d = os.path.join(images_dir, direction)
        if not os.path.isdir(d):
            print(f"[SKIP] {direction}: no images directory {d}")
            continue

        image_files = sorted([
            f for f in os.listdir(d)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
        if len(image_files) < MIN_CAPTURE_FRAMES:
            print(f"[SKIP] {direction}: only {len(image_files)} images, need >= {MIN_CAPTURE_FRAMES}")
            continue

        print(f"\n[PROCESSING] {direction}: {len(image_files)} images")
        obj_points_list = []
        img_points_list = []
        image_size = None

        for fname in image_files:
            path = os.path.join(d, fname)
            img = cv2.imread(path)
            if img is None:
                print(f"  [WARN] Cannot read {path}")
                continue
            if image_size is None:
                image_size = (img.shape[1], img.shape[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = detect_corners(gray, pattern_size)
            if found:
                obj_points_list.append(objp)
                img_points_list.append(corners)
            else:
                print(f"  [WARN] No checkerboard found in {fname}")

        if len(obj_points_list) < MIN_CAPTURE_FRAMES:
            print(f"[SKIP] {direction}: only {len(obj_points_list)} valid frames, need >= {MIN_CAPTURE_FRAMES}")
            continue

        try:
            K, D, rms, rvecs, tvecs = calibrate_camera(obj_points_list, img_points_list, image_size)
        except Exception as e:
            print(f"[ERROR] Calibration failed for {direction}: {e}")
            continue

        D_inv, max_err = fit_inverse_polynomial(D)
        print(f"  RMS reprojection error: {rms:.4f} px")
        print(f"  Inverse fit max error:   {max_err:.6f} rad")

        evaluation = evaluate_intrinsics(
            K, D, rms, image_size, obj_points_list, img_points_list,
            rvecs=rvecs, tvecs=tvecs)
        print_evaluation_report(evaluation, direction)

        save_results(direction, K, D, D_inv, rms, output_dir, evaluation=evaluation)
        print_cpp_snippet(direction, K, D, D_inv)

        if image_files:
            sample_img = cv2.imread(os.path.join(d, image_files[0]))
            if sample_img is not None:
                save_undistorted_sample(direction, sample_img, K, D, output_dir)

        results[direction] = {"K": K, "D": D, "D_inv": D_inv, "rms": rms}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate 4 fisheye cameras for AVM system (Kannala-Brandt model)"
    )
    parser.add_argument("--preview", action="store_true",
                        help="Preview all 4 cameras to verify mapping")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration interactively (live capture)")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Path to camera_config.json (default: {DEFAULT_CONFIG})")
    parser.add_argument("--images-dir", default=None,
                        help="Use pre-captured images instead of live capture")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Re-run calibration from saved images in calib_images/")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--direction", choices=["left", "back", "front", "right"],
                        help="Calibrate only a single camera direction")
    parser.add_argument("--board-config", default=DEFAULT_BOARD_CONFIG,
                        help=f"棋盘规格 JSON（默认 {DEFAULT_BOARD_CONFIG}）")
    parser.add_argument("--pattern-size", type=str, default=None,
                        help="覆盖棋盘内角点 WxH（默认读 --board-config）")
    parser.add_argument("--square-size", type=float, default=None,
                        help="覆盖单格边长米（默认读 --board-config）")
    parser.add_argument("--evaluate", action="store_true",
                        help="评估已有内参标定结果（不重新标定）")
    parser.add_argument(
        "--control-file",
        default=None,
        help="远程按键命令文件（也可设环境变量 AVM_CALIB_CONTROL_FILE）",
    )
    args = parser.parse_args()

    log_cuda_status()
    print("  [标定] 棋盘检测 / fisheye.calibrate 固定 CPU，保证精度")

    if args.control_file:
        os.environ["AVM_CALIB_CONTROL_FILE"] = args.control_file

    board = resolve_board_args(args.pattern_size, args.square_size, args.board_config)
    pattern_size = board["pattern_size"]
    square_size = board["square_size_m"]
    print(f"[棋盘] {board['pattern_size_str']} 内角点, square={square_size}m"
          f"  ← {board['path']}"
          + ("" if board["from_config"] else "（含 CLI 覆盖）"))

    log_opencv_status()

    config = load_config(args.config)

    if args.evaluate:
        directions = [args.direction] if args.direction else None
        reports = evaluate_existing(args.output_dir, directions)
        if not reports:
            print("[错误] 没有可评估的内参结果。")
            sys.exit(1)
        all_pass = all(r["passed"] for r in reports.values())
        print(f"\n总结: {'✅ 全部通过' if all_pass else '❌ 存在不合格，建议重新标定'}")
        sys.exit(0 if all_pass else 1)

    if args.preview:
        preview_cameras(config)
        return

    if args.calibrate:
        directions = [args.direction] if args.direction else list(config.keys())

        if args.images_dir or args.recalibrate:
            images_dir = args.images_dir if args.images_dir else os.path.join(PROJECT_DIR, "calib_images")
            results = calibrate_from_images(
                config, images_dir, args.output_dir,
                pattern_size, square_size, directions
            )
        else:
            results = {}
            for direction in directions:
                idx = config.get(direction)
                if idx is None:
                    print(f"[SKIP] {direction}: not found in config")
                    continue
                try:
                    cap, w, h = open_camera(idx)
                except Exception as e:
                    print(f"[ERROR] {direction}: {e}")
                    continue

                images_dir = os.path.join(PROJECT_DIR, "calib_images", direction)
                obj_points_list, img_points_list, image_size = interactive_capture(
                    cap, direction, pattern_size, square_size, images_dir, camera_idx=idx
                )
                cap.release()

                if len(obj_points_list) < MIN_CAPTURE_FRAMES:
                    print(f"[SKIP] {direction}: only {len(obj_points_list)} frames, need >= {MIN_CAPTURE_FRAMES}")
                    cv2.destroyAllWindows()
                    continue

                try:
                    K, D, rms, rvecs, tvecs = calibrate_camera(obj_points_list, img_points_list, image_size)
                except Exception as e:
                    print(f"[ERROR] Calibration failed for {direction}: {e}")
                    print(f"[RECOVER] Images saved to {images_dir}/")
                    print(f"[RECOVER] Re-run: python3 scripts/calibrate_intrinsics.py --calibrate --recalibrate --direction {direction}")
                    cv2.destroyAllWindows()
                    continue

                D_inv, max_err = fit_inverse_polynomial(D)
                print(f"  RMS reprojection error: {rms:.4f} px")
                print(f"  Inverse fit max error:   {max_err:.6f} rad")

                evaluation = evaluate_intrinsics(
                    K, D, rms, image_size, obj_points_list, img_points_list,
                    rvecs=rvecs, tvecs=tvecs)
                print_evaluation_report(evaluation, direction)

                save_results(direction, K, D, D_inv, rms, args.output_dir, evaluation=evaluation)
                print_cpp_snippet(direction, K, D, D_inv)

                image_files = sorted(os.listdir(images_dir))
                if image_files:
                    sample_img = cv2.imread(os.path.join(images_dir, image_files[0]))
                    if sample_img is not None:
                        save_undistorted_sample(direction, sample_img, K, D, args.output_dir)

                results[direction] = {"K": K, "D": D, "D_inv": D_inv, "rms": rms}

        if results:
            print("\n" + "=" * 60)
            print("  CALIBRATION SUMMARY")
            print("=" * 60)
            for direction, r in results.items():
                print(f"  {direction:8s}  RMS={r['rms']:.4f} px  "
                      f"fx={r['K'][0,0]:.2f}  fy={r['K'][1,1]:.2f}  "
                      f"cx={r['K'][0,2]:.2f}  cy={r['K'][1,2]:.2f}")
            print("=" * 60)
            print(f"  Results saved to: {args.output_dir}/")
            print("=" * 60)
            # 全部通过检查
            all_pass = True
            for d in results:
                path = os.path.join(args.output_dir, f"{d}.json")
                if os.path.isfile(path):
                    with open(path) as f:
                        data = json.load(f)
                    ev = data.get("evaluation", {})
                    if not ev.get("passed", True):
                        all_pass = False
                        break
            if all_pass:
                print("\n✅ 内参评估全部通过，可进行外参标定。")
            else:
                print("\n❌ 部分内参评估未通过，建议重新采集后再标定外参。")
        return

    parser.print_help()


if __name__ == "__main__":
    main()