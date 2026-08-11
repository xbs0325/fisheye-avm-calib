#!/usr/bin/env python3
"""BEV stitch helpers — copied from avm.live_bev for the post-calib demo.

Uses avm.cuda_cv / avm.camera_io (not the legacy bare `cuda_cv` imports).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from avm.camera_io import capture_size, load_camera_profile, open_camera_direction
from avm.cuda_cv import (
    UndistortWarpPipeline,
    init_undistort_maps,
    log_cuda_status,
    process_frames_to_bev,
    resize_bgr,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "camera_config.json"
DEFAULT_CALIB_DIR = ROOT / "calib_results"
DEFAULT_EXTRINSICS = DEFAULT_CALIB_DIR / "extrinsics.json"
DIRECTIONS = ("front", "back", "left", "right")
CAPTURE_W, CAPTURE_H = capture_size()


def load_config(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_extrinsics(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {
        "old_scale": data.get("scale_px_per_meter", 100.0),
        "old_canvas": tuple(data.get("canvas_size", [1000, 1000])),
        "homographies": {},
        "undistort_balance": float(
            data.get("extrinsic_balance", data.get("balance", 0.5))
        ),
    }
    for d in DIRECTIONS:
        if d in (data.get("homographies") or {}):
            result["homographies"][d] = np.asarray(
                data["homographies"][d], dtype=np.float64
            )
    return result


def load_intrinsics(direction: str, calib_dir: Path | str):
    path = Path(calib_dir) / f"{direction}.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return np.asarray(data["K"], dtype=np.float64), np.asarray(
        data["D"], dtype=np.float64
    )


def adjust_homography(H_old, old_scale, old_canvas, new_scale, new_canvas):
    old_cw, old_ch = old_canvas
    new_cw, new_ch = new_canvas
    s = new_scale / old_scale
    tx = new_cw / 2.0 - s * old_cw / 2.0
    ty = new_ch / 2.0 - s * old_ch / 2.0
    A = np.array([[s, 0, tx], [0, s, ty], [0, 0, 1.0]], dtype=np.float64)
    return A @ H_old


def compute_canvas(range_m: float, scale: float):
    cw = int(2.0 * range_m * scale)
    ch = int(2.0 * range_m * scale)
    return (cw, ch), (cw / 2.0, ch / 2.0)


def build_weight_maps(canvas_size, center, blend_power: float):
    h, w = canvas_size[1], canvas_size[0]
    cx, cy = center
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    gx = (x_grid - cx).astype(np.float64)
    gy = (cy - y_grid).astype(np.float64)
    angle = np.arctan2(gy, gx)
    cam_angle = {
        "front": np.pi / 2.0,
        "back": -np.pi / 2.0,
        "left": np.pi,
        "right": 0.0,
    }
    raw = {}
    for d, ca in cam_angle.items():
        diff = np.arctan2(np.sin(angle - ca), np.cos(angle - ca))
        raw[d] = np.clip(np.cos(diff), 0.0, 1.0) ** blend_power
    weight_sum = np.maximum(sum(raw.values()), 1e-10)
    return {d: (raw[d] / weight_sum).astype(np.float32) for d in raw}


def compute_gains(bev_views: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    masks = {}
    for d, bev in bev_views.items():
        if bev is None:
            continue
        masks[d] = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY) > 10
    pairs = [
        ("front", "left"),
        ("front", "right"),
        ("back", "left"),
        ("back", "right"),
    ]
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


def build_pipelines(
    *,
    extrinsics_path: Path,
    calib_dir: Path,
    range_m: float,
    scale: float,
    blend_power: float,
) -> tuple[dict[str, UndistortWarpPipeline], tuple[int, int], tuple[float, float], float]:
    """Load calib and build CUDA undistort+H pipelines. Raises if incomplete."""
    if not extrinsics_path.is_file():
        raise FileNotFoundError(
            f"缺少外参: {extrinsics_path}（标定后 demo 需要先完成外参）"
        )
    extrinsics = load_extrinsics(extrinsics_path)
    canvas, center = compute_canvas(range_m, scale)
    log_cuda_status()
    balance = extrinsics["undistort_balance"]
    weights = build_weight_maps(canvas, center, blend_power)
    pipelines: dict[str, UndistortWarpPipeline] = {}
    for d in DIRECTIONS:
        if d not in extrinsics["homographies"]:
            continue
        intr_path = calib_dir / f"{d}.json"
        if not intr_path.is_file():
            raise FileNotFoundError(f"缺少内参: {intr_path}")
        K, D = load_intrinsics(d, calib_dir)
        m1, m2 = init_undistort_maps(
            K, D, CAPTURE_W, CAPTURE_H, balance, for_cuda=True
        )
        H = adjust_homography(
            extrinsics["homographies"][d],
            extrinsics["old_scale"],
            extrinsics["old_canvas"],
            scale,
            canvas,
        )
        pipe = UndistortWarpPipeline(
            m1, m2, H=H, canvas_size=canvas, weight=weights[d]
        )
        pipelines[d] = pipe
    if len(pipelines) < 4:
        raise RuntimeError(
            f"BEV 管道不完整: {sorted(pipelines.keys())}（需要四路内参+外参 H）"
        )
    return pipelines, canvas, center, balance


def open_caps(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Open four cameras via camera_io + profile."""
    prof = load_camera_profile()
    caps: dict[str, Any] = {}
    for d in DIRECTIONS:
        if d not in (prof.get("cameras") or {}):
            # fall back to camera_config.json indices
            cfg = load_config(config_path or DEFAULT_CONFIG)
            if d not in cfg:
                continue
            from avm.camera_io import open_camera_index

            w, h = capture_size(prof)
            cap, _, _, _ = open_camera_index(
                int(cfg[d]),
                width=w,
                height=h,
                fourcc=str(prof.get("fourcc") or "YUYV"),
                backend=str(prof.get("backend") or "v4l2"),
            )
            caps[d] = cap
            continue
        cap, _, _, _ = open_camera_direction(d, prof)
        caps[d] = cap
    if not caps:
        raise RuntimeError("无可用相机")
    return caps


def grab_frames(caps: dict[str, Any]) -> dict[str, np.ndarray]:
    for cap in caps.values():
        cap.grab()
    frames: dict[str, np.ndarray] = {}
    for d, cap in caps.items():
        ok, frame = cap.retrieve()
        if ok and frame is not None:
            frames[d] = frame
    return frames


def process_frame(
    raw_frames: dict[str, np.ndarray],
    pipelines: dict[str, UndistortWarpPipeline],
    gains: dict[str, np.ndarray],
    canvas_size: tuple[int, int],
    *,
    need_bev_views: bool = False,
):
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
        list(DIRECTIONS),
        gains=gains,
        canvas_size=canvas_size,
        need_bev_views=need_bev_views,
    )


def _cjk_font(size: int = 16):
    """Load a system CJK font for PIL (cv2.putText cannot draw Chinese)."""
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_cjk(text: str, max_chars: int) -> list[str]:
    """Wrap mixed CJK/ASCII by display width (~2 for CJK, 1 for ASCII)."""
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    rows: list[str] = []
    cur = ""
    width = 0

    def char_w(ch: str) -> int:
        return 2 if ord(ch) > 0x7F else 1

    for ch in text:
        w = char_w(ch)
        if width + w > max_chars and cur:
            rows.append(cur)
            cur = ch
            width = w
        else:
            cur += ch
            width += w
    if cur:
        rows.append(cur)
    return rows


def draw_hud(
    bev: np.ndarray,
    fps_val: float,
    blend_power: float,
    gain_enabled: bool,
    canvas_size: tuple[int, int],
    scale: float,
    caption: str = "",
    vlm_status: str = "",
) -> np.ndarray:
    display = bev.copy()
    cw, ch = canvas_size
    lines = [
        f"FPS: {1.0 / max(fps_val, 0.001):.0f}",
        f"Range: ±{cw / (2.0 * scale):.1f}m  Scale: {scale}px/m",
        f"Blend: cos^{blend_power:.0f}  Gain: {'ON' if gain_enabled else 'OFF'}",
        "ESC/q:quit  s:save  c:caption-now  g:gain  +/-:blend",
    ]
    if vlm_status:
        lines.append(vlm_status)
    for i, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (8, 20 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    labels = {"front": "F", "back": "B", "left": "L", "right": "R"}
    positions = {
        "front": (cw // 2, 25),
        "back": (cw // 2, ch - 10),
        "left": (15, ch // 2),
        "right": (cw - 25, ch // 2),
    }
    for d, (px, py) in positions.items():
        cv2.putText(
            display,
            labels[d],
            (px, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if caption:
        # PIL + CJK font — cv2.putText cannot render Chinese (shows mojibake).
        from PIL import Image, ImageDraw

        max_chars = max(20, cw // 10)
        rows = _wrap_cjk(caption, max_chars)[-4:]
        font = _cjk_font(size=max(14, cw // 28))
        line_h = max(18, int(getattr(font, "size", 16) + 6))
        pad = 10
        box_h = pad * 2 + line_h * len(rows)
        y0 = max(0, ch - box_h)
        overlay = display.copy()
        cv2.rectangle(overlay, (0, y0), (cw, ch), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        for i, row in enumerate(rows):
            draw.text((8, y0 + pad + i * line_h), row, font=font, fill=(0, 255, 255))
        display = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return display
