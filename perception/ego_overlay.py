"""Composite a transparent chassis PNG onto the BEV center."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPRITE = ROOT / "assets" / "ego_overlay.png"

UvBox = Tuple[float, float, float, float]


def load_ego_sprite(path: Optional[Path] = None) -> Optional[np.ndarray]:
    p = Path(path) if path is not None else DEFAULT_SPRITE
    if not p.is_file():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        a = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack([img, a])
    return img


def _hole_aabb(
    gray: np.ndarray,
    *,
    vehicle_uv: Optional[UvBox] = None,
    dark_thresh: int = 50,
) -> Tuple[int, int, int, int]:
    h, w = gray.shape[:2]
    dark = (gray < int(dark_thresh)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k)
    cx, cy = w // 2, h // 2
    rad = max(12, int(0.30 * min(h, w)))
    y0s, y1s = max(0, cy - rad), min(h, cy + rad + 1)
    x0s, x1s = max(0, cx - rad), min(w, cx + rad + 1)
    roi = np.zeros_like(dark)
    roi[y0s:y1s, x0s:x1s] = dark[y0s:y1s, x0s:x1s]
    if roi[cy, cx] == 0:
        r = max(4, min(h, w) // 16)
        roi[max(0, cy - r) : cy + r + 1, max(0, cx - r) : cx + r + 1] = 1
    num, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    lab = int(labels[cy, cx])
    if lab > 0 and num >= 2:
        x, y, bw, bh = (
            int(stats[lab, k])
            for k in (
                cv2.CC_STAT_LEFT,
                cv2.CC_STAT_TOP,
                cv2.CC_STAT_WIDTH,
                cv2.CC_STAT_HEIGHT,
            )
        )
        return x, y, x + bw, y + bh
    if vehicle_uv is not None:
        u0, v0, u1, v1 = vehicle_uv
        return (
            int(round(u0 * w)),
            int(round(v0 * h)),
            int(round(u1 * w)),
            int(round(v1 * h)),
        )
    s = int(0.28 * min(h, w))
    x0, y0 = (w - s) // 2, (h - s) // 2
    return x0, y0, x0 + s, y0 + s


def _grow_left_over_dark(
    gray: np.ndarray, x0: int, y0: int, x1: int, y1: int, *, thresh: int = 50
) -> int:
    """Walk left while that column is still the stitch hole."""
    h, w = gray.shape[:2]
    y_a = y0 + max(1, (y1 - y0) // 5)
    y_b = y1 - max(1, (y1 - y0) // 5)
    if y_b <= y_a:
        y_a, y_b = y0, y1
    limit = max(4, int(0.06 * (x1 - x0)))
    grew = 0
    while x0 > 0 and grew < limit:
        col = gray[y_a:y_b, x0 - 1]
        if col.size == 0 or float((col < thresh).mean()) < 0.40:
            break
        x0 -= 1
        grew += 1
    return x0


def center_blind_box(
    bev_bgr: np.ndarray,
    *,
    vehicle_uv: Optional[UvBox] = None,
    dark_thresh: int = 50,
) -> Tuple[int, int, int, int]:
    """Square covering the stitch hole; a little extra on the left black strip."""
    h, w = bev_bgr.shape[:2]
    gray = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2GRAY)
    hx0, hy0, hx1, hy1 = _hole_aabb(
        gray, vehicle_uv=vehicle_uv, dark_thresh=dark_thresh
    )
    mx = 0.5 * w
    my = 0.5 * h
    hole = max(hx1 - hx0, hy1 - hy0)
    side = float(hole) * 1.14
    side = float(np.clip(side, 0.18 * min(h, w), 0.36 * min(h, w)))
    side = min(side, 2.0 * mx - 2.0, 2.0 * my - 2.0)
    x0i = int(np.floor(mx - side * 0.5))
    y0i = int(np.floor(my - side * 0.5))
    x1i = x0i + int(np.ceil(side))
    y1i = y0i + int(np.ceil(side))
    x0i = _grow_left_over_dark(gray, x0i, y0i, x1i, y1i, thresh=dark_thresh)
    x0i = int(np.clip(x0i, 0, w - 2))
    y0i = int(np.clip(y0i, 0, h - 2))
    x1i = int(np.clip(x1i, x0i + 2, w))
    y1i = int(np.clip(y1i, y0i + 2, h))
    return x0i, y0i, x1i, y1i


def overlay_ego(
    bev_bgr: np.ndarray,
    sprite_bgra: Optional[np.ndarray],
    *,
    size_m: float = 0.0,
    scale_px_per_meter: float = 120.0,
    vehicle_uv: Optional[UvBox] = None,
    box: Optional[Tuple[int, int, int, int]] = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """Paste chassis. Pass a locked `box` so it does not jitter every frame."""
    if sprite_bgra is None or sprite_bgra.size == 0:
        return bev_bgr
    h, w = bev_bgr.shape[:2]
    if box is not None:
        x0, y0, x1, y1 = (int(v) for v in box)
    elif float(size_m) > 0:
        px = max(8, int(round(float(size_m) * float(scale_px_per_meter))))
        px = min(px, w - 2, h - 2)
        x0 = (w - px) // 2
        y0 = (h - px) // 2
        x1, y1 = x0 + px, y0 + px
    else:
        x0, y0, x1, y1 = center_blind_box(bev_bgr, vehicle_uv=vehicle_uv)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return bev_bgr
    a0 = sprite_bgra[:, :, 3]
    ys, xs = np.where(a0 > 12)
    if xs.size > 0:
        sprite_bgra = sprite_bgra[
            int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1
        ]
    out = bev_bgr.copy()
    nw, nh = x1 - x0, y1 - y0
    small = cv2.resize(sprite_bgra, (nw, nh), interpolation=cv2.INTER_AREA)
    roi = out[y0:y1, x0:x1].astype(np.float32)
    rgb = small[:, :, :3].astype(np.float32)
    a = (small[:, :, 3:4].astype(np.float32) / 255.0) * float(np.clip(alpha, 0.0, 1.0))
    out[y0:y1, x0:x1] = np.clip(rgb * a + roi * (1.0 - a), 0, 255).astype(np.uint8)
    return out
