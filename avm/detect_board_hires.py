#!/usr/bin/env python3
"""棋盘检测，角点始终映射回原图。

背景：`findChessboardCorners` 在"没有棋盘"时最贵（全图 adaptive threshold 搜索），
1920×1536 单次可达 1.3s。因此实时路径把检测放到独立线程并降低频率，
但仍按用户要求使用完整检测分辨率，不用低分辨率门控。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

import cv2

DETECT_FLAGS_FAST = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FAST_CHECK
)
DETECT_FLAGS_FULL = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
)
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

_HAS_SB = hasattr(cv2, "findChessboardCornersSB")
SB_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE if _HAS_SB else 0


def find_board_corners(
    gray: np.ndarray,
    pattern: Tuple[int, int],
    *,
    use_sb: bool = True,
) -> Optional[np.ndarray]:
    """在灰度图上找棋盘内角点，返回 Nx1x2 float32 或 None。

    优先 `findChessboardCornersSB`（sector-based）：实测在 1920×1536 上
    比经典算法多撑一档距离（7px 格子仍可检出，经典算法 9px 就到头），
    且"找不到"时耗时恒定 ~300ms，而经典算法要 ~1300ms。
    经典算法命中时更快（<10ms），因此保留为兜底。
    """
    if use_sb and _HAS_SB:
        found, corners = cv2.findChessboardCornersSB(gray, pattern, SB_FLAGS)
        if found and corners is not None:
            # SB 自带亚像素精度，无需 cornerSubPix
            return corners.astype(np.float32)

    found, corners = cv2.findChessboardCorners(gray, pattern, DETECT_FLAGS_FULL)
    if not found or corners is None:
        return None
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), _SUBPIX_CRITERIA)
    return corners.astype(np.float32)


def _gray_at(bgr: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """转灰度并缩放到指定宽度，返回 (gray, scale)。scale = 小图/原图。"""
    h, w = bgr.shape[:2]
    if width <= 0 or width >= w:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        return gray, 1.0
    scale = width / float(w)
    nh = max(1, int(round(h * scale)))
    small = cv2.resize(bgr, (width, nh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return gray, scale


def scan_chessboard(
    bgr: np.ndarray,
    pattern: Tuple[int, int],
    *,
    scan_width: int = 1920,
    refine_width: int = 1920,
    use_sb: bool = True,
) -> tuple[bool, Optional[np.ndarray], float, str]:
    """实时扫描：在 scan_width 上检测；需要时可在 refine_width 精修。

    返回 (found, corners_full_res Nx1x2 float32 or None, scan_scale, stage)
    stage: "miss" | "sb" | "classic"
    """
    if bgr is None or bgr.size == 0:
        return False, None, 1.0, "miss"

    gray_s, scale_s = _gray_at(bgr, scan_width)

    # 实时路径故意不做"SB 失败再退经典"：经典算法失败要 ~1300ms，
    # 叠加后单次 miss 高达 ~1600ms，扫描频率反而更差。
    # SB 实测检出范围严格更广（7px 格子仍可检出），因此 miss 就直接返回。
    if use_sb and _HAS_SB:
        found, corners = cv2.findChessboardCornersSB(gray_s, pattern, SB_FLAGS)
        if not found or corners is None:
            return False, None, scale_s, "miss"
        return True, corners.astype(np.float32) / scale_s, scale_s, "sb"

    found, corners = cv2.findChessboardCorners(gray_s, pattern, DETECT_FLAGS_FULL)
    if not found or corners is None:
        return False, None, scale_s, "miss"
    corners = cv2.cornerSubPix(gray_s, corners, (5, 5), (-1, -1), _SUBPIX_CRITERIA)
    return True, corners.astype(np.float32) / scale_s, scale_s, "classic"


def detect_chessboard_hires(
    bgr: np.ndarray,
    pattern: Tuple[int, int],
    *,
    max_width: int = 1920,
    try_scales: Optional[Sequence[float]] = None,
    subpix: bool = True,
) -> tuple[bool, Optional[np.ndarray], float]:
    """一次性高质量检测（SPACE 抓拍用，可多尺度，慢但尽力）。

    返回 (found, corners_full_res Nx1x2 float32 or None, used_scale)
    """
    if bgr is None or bgr.size == 0:
        return False, None, 1.0
    h, w = bgr.shape[:2]
    base = 1.0
    if max_width > 0 and w > max_width:
        base = float(max_width) / float(w)

    scales = list(try_scales) if try_scales else [1.0, 0.75, 0.5]
    scales = sorted({max(0.25, float(s)) for s in scales}, reverse=True)

    for rel in scales:
        scale = base * rel
        if scale >= 0.999:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            scale = 1.0
        else:
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        for flags in (DETECT_FLAGS_FAST, DETECT_FLAGS_FULL):
            found, corners = cv2.findChessboardCorners(gray, pattern, flags)
            if not found or corners is None:
                continue
            if subpix:
                corners = cv2.cornerSubPix(
                    gray, corners, (5, 5), (-1, -1), _SUBPIX_CRITERIA
                )
            corners = corners.astype(np.float32)
            if scale != 1.0:
                corners = corners / float(scale)
            return True, corners, float(scale)

    return False, None, float(base)


def project_corners_to_tile(
    corners_full: np.ndarray,
    full_wh: tuple[int, int],
    tile_wh: tuple[int, int],
) -> np.ndarray:
    """把全分辨率角点映射到预览 tile 坐标，供 drawChessboardCorners。"""
    fw, fh = full_wh
    tw, th = tile_wh
    c = corners_full.reshape(-1, 2).astype(np.float32).copy()
    c[:, 0] *= tw / float(fw)
    c[:, 1] *= th / float(fh)
    return c.reshape(-1, 1, 2)
