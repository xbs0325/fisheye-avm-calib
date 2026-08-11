#!/usr/bin/env python3
"""Crop the 4-camera orange chassis photo to a transparent PNG for BEV center.

Keeps only: orange print + cameras + GMSL cables. No floor, no white plate.

Example:
  python3 scripts/make_ego_overlay.py assets/ego_chassis.png -o assets/ego_overlay.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num < 2:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def _fill_interior_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes (前 sticker, cable labels) but not the outside."""
    inv = cv2.bitwise_not(mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h, w = mask.shape[:2]
    out = mask.copy()
    for i in range(1, num):
        x, y, bw, bh, area = (int(stats[i, k]) for k in range(5))
        if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
            continue
        if area > 0.12 * float(h * w):
            continue
        out[labels == i] = 255
    return out


def _orange_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (4, 90, 90), (22, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if num < 2:
        return m
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = labels == idx
    lx = int(stats[idx, cv2.CC_STAT_LEFT])
    ly = int(stats[idx, cv2.CC_STAT_TOP])
    lw = int(stats[idx, cv2.CC_STAT_WIDTH])
    lh = int(stats[idx, cv2.CC_STAT_HEIGHT])
    lmx = lx + 0.5 * lw
    lmy = ly + 0.5 * lh
    # 前 sticker splits the front arm; keep the disconnected orange tip.
    for i in range(1, num):
        if i == idx:
            continue
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 200:
            continue
        cx, cy = float(cents[i][0]), float(cents[i][1])
        on_fb = abs(cx - lmx) < 0.22 * lw
        in_x = (lx - 40) <= cx <= (lx + lw + 40)
        if on_fb and in_x:
            keep |= labels == i
    return keep.astype(np.uint8) * 255


def _sticker_mask(bgr: np.ndarray, orange: np.ndarray) -> np.ndarray:
    """前 label + cable tags. Skip the huge white plate."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = orange.shape[:2]
    white = cv2.inRange(hsv, (0, 0, 150), (180, 55, 255))
    blue = cv2.inRange(hsv, (90, 40, 40), (135, 255, 255))
    near = cv2.dilate(
        orange, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    )
    dist = cv2.distanceTransform(cv2.bitwise_not(orange), cv2.DIST_L2, 5)
    keep = np.zeros_like(orange)
    for mask in (white, blue):
        num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 60 or area > 0.02 * float(h * w):
                continue
            cy, cx = float(cents[i][1]), float(cents[i][0])
            if dist[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))] > 90:
                continue
            blob = labels == i
            if int((blob & (near > 0)).sum()) < 20:
                continue
            keep[blob] = 255
    return keep


def _hardware_mask(bgr: np.ndarray, body: np.ndarray) -> np.ndarray:
    """Cameras, GMSL cables, teal/magenta plugs — grown from the print."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    dist = cv2.distanceTransform(cv2.bitwise_not(body), cv2.DIST_L2, 5)
    near = (dist < 16.0).astype(np.uint8) * 255
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 90, 70))
    teal = cv2.inRange(hsv, (70, 40, 40), (105, 255, 255))
    mag = cv2.inRange(hsv, (125, 40, 40), (175, 255, 255))
    extra = cv2.bitwise_or(dark, cv2.bitwise_or(teal, mag))
    extra = cv2.bitwise_and(extra, near)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    extra = cv2.morphologyEx(extra, cv2.MORPH_CLOSE, k, iterations=2)
    return extra


def _grow(seed: np.ndarray, allowed: np.ndarray, steps: int = 12) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cur = seed.copy()
    for _ in range(steps):
        nxt = cv2.bitwise_and(cv2.dilate(cur, k), allowed)
        if int(cv2.countNonZero(nxt)) == int(cv2.countNonZero(cur)):
            break
        cur = nxt
    return cur


def cut_chassis(bgr: np.ndarray) -> np.ndarray:
    orange = _orange_mask(bgr)
    if int(cv2.countNonZero(orange)) < 200:
        raise RuntimeError("no orange chassis found — check the photo")
    stickers = _sticker_mask(bgr, orange)
    body = cv2.bitwise_or(orange, stickers)
    # Bridge the ~10px gap where the 前 sticker splits the front arm.
    k21 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, k21, iterations=2)
    extra = _hardware_mask(bgr, body)
    allowed = cv2.bitwise_or(body, extra)
    fg = _grow(body, allowed)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    fg = _largest_blob(fg)
    fg = _fill_interior_holes(fg)
    if int(cv2.countNonZero(fg)) < 200:
        fg = orange

    alpha = cv2.GaussianBlur(fg, (3, 3), 0)
    # Ignore thin leaks (floor cables) when computing the crop.
    row_n = (alpha > 12).sum(axis=1)
    col_n = (alpha > 12).sum(axis=0)
    ys = np.where(row_n >= 36)[0]
    xs = np.where(col_n >= 36)[0]
    if xs.size == 0 or ys.size == 0:
        raise RuntimeError("empty mask")
    h, w = bgr.shape[:2]
    m = 8
    x0, x1 = max(0, int(xs.min()) - m), min(w, int(xs.max()) + 1 + m)
    y0, y1 = max(0, int(ys.min()) - m), min(h, int(ys.max()) + 1 + m)
    crop = bgr[y0:y1, x0:x1]
    a = alpha[y0:y1, x0:x1]
    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = a
    return bgra


def checkerboard(h: int, w: int, cell: int = 16) -> np.ndarray:
    yy, xx = np.indices((h, w))
    c = ((yy // cell) + (xx // cell)) % 2
    out = np.full((h, w, 3), 210, dtype=np.uint8)
    out[c == 1] = 160
    return out


def composite(bg: np.ndarray, bgra: np.ndarray) -> np.ndarray:
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    rgb = bgra[:, :, :3].astype(np.float32)
    base = bg.astype(np.float32)
    return np.clip(rgb * a + base * (1.0 - a), 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description="Crop orange 4-cam chassis to transparent PNG")
    ap.add_argument(
        "photo",
        nargs="?",
        type=Path,
        default=ROOT / "assets" / "ego_chassis.png",
        help="top-down photo (前 arm up)",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=ROOT / "assets" / "ego_overlay.png",
    )
    ap.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="optional checkerboard preview path",
    )
    args = ap.parse_args()
    photo = args.photo
    if not photo.is_file():
        print(f"missing photo: {photo}", file=sys.stderr)
        return 1
    bgr = cv2.imread(str(photo), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"failed to read {photo}", file=sys.stderr)
        return 1
    bgra = cut_chassis(bgr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), bgra)
    prev = args.preview or args.out.with_name(args.out.stem + "_preview.jpg")
    board = checkerboard(bgra.shape[0], bgra.shape[1])
    cv2.imwrite(str(prev), composite(board, bgra))
    print(f"wrote {args.out}  {bgra.shape[1]}x{bgra.shape[0]}")
    print(f"preview {prev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
