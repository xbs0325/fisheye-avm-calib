"""BEV overlay for structured nav/grasp perception events."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from perception.localize import base_link_to_pixel, norm_to_pixel
from perception.schema import PerceptionEvent

# Film palette (BGR)
_ORANGE = (48, 118, 255)
_CYAN = (230, 210, 64)
_RED = (56, 56, 230)
_GREEN = (80, 200, 90)
_WHITE = (248, 248, 248)
_INK = (16, 14, 12)
_MUTED = (190, 188, 184)


def _cjk_font(size: int = 16):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
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


def _shade(img: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], alpha: float) -> None:
    x0, y0 = pt1
    x1, y1 = pt2
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return
    overlay = np.empty_like(roi)
    overlay[:] = _INK
    cv2.addWeighted(overlay, float(alpha), roi, 1.0 - float(alpha), 0, roi)


def _pill(
    img: np.ndarray,
    xy: tuple[int, int],
    text: str,
    *,
    fg=_WHITE,
    bg=(40, 36, 32),
    scale: float = 0.42,
) -> int:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = xy
    pad_x, pad_y = 8, 5
    x1, y1 = x + tw + pad_x * 2, y + th + pad_y * 2
    cv2.rectangle(img, (x, y), (x1, y1), bg, -1, cv2.LINE_AA)
    cv2.putText(
        img,
        text,
        (x + pad_x, y + th + pad_y - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        fg,
        1,
        cv2.LINE_AA,
    )
    return x1


def _label_at(
    img: np.ndarray,
    u: int,
    v: int,
    text: str,
    color: tuple[int, int, int],
) -> None:
    scale = 0.42
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = min(max(4, u + 10), img.shape[1] - tw - 16)
    y = min(max(th + 8, v - 10), img.shape[0] - 8)
    cv2.rectangle(
        img,
        (x - 5, y - th - 5),
        (x + tw + 5, y + 4),
        (12, 10, 8),
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_cross(img: np.ndarray, u: int, v: int, color, size: int = 10) -> None:
    cv2.drawMarker(
        img,
        (u, v),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=size,
        thickness=2,
        line_type=cv2.LINE_AA,
    )


def _class_color(label: str, *, accent: bool = False) -> tuple[int, int, int]:
    lab = (label or "").lower()
    if lab in ("person", "people", "man", "woman"):
        return _RED
    if lab in ("bottle", "cup", "water bottle"):
        return _CYAN
    if accent:
        return _ORANGE
    return (40, 160, 255)


def draw_perception_overlay(
    bev: np.ndarray,
    event: Optional[PerceptionEvent],
    *,
    canvas_size: Sequence[int],
    scale_px_per_meter: float,
    vehicle_marker: bool = True,
) -> np.ndarray:
    display = bev.copy()
    cw, ch = int(canvas_size[0]), int(canvas_size[1])
    scale = float(scale_px_per_meter)

    cx, cy = cw // 2, ch // 2
    for r_m in (1.0, 2.0):
        rad = int(round(r_m * scale))
        if 16 < rad < min(cw, ch) // 2 - 8:
            cv2.circle(display, (cx, cy), rad, (160, 160, 160), 1, cv2.LINE_AA)

    if vehicle_marker:
        cv2.circle(display, (cx, cy), 7, _WHITE, 1, cv2.LINE_AA)
        cv2.arrowedLine(
            display, (cx, cy), (cx, cy - 36), _GREEN, 2, cv2.LINE_AA, 0, 0.28
        )

    if event is None:
        return display

    if event.nav is not None:
        for obs in event.nav.obstacles:
            if obs.label == "occ":
                continue
            u = v = None
            if obs.x_m is not None and obs.y_m is not None:
                u, v = base_link_to_pixel(
                    obs.x_m,
                    obs.y_m,
                    canvas_w=cw,
                    canvas_h=ch,
                    scale_px_per_meter=scale,
                )
            elif obs.u_norm is not None and obs.v_norm is not None:
                u, v = norm_to_pixel(
                    obs.u_norm, obs.v_norm, canvas_w=cw, canvas_h=ch
                )
            if u is None or v is None:
                continue
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < cw and 0 <= vi < ch):
                continue
            color = _class_color(obs.label)
            _draw_cross(display, ui, vi, color, size=14)
            r_px = int(max(10, (obs.radius_m or 0.22) * scale))
            cv2.circle(display, (ui, vi), r_px, color, 2, cv2.LINE_AA)
            rng = None
            if obs.x_m is not None and obs.y_m is not None:
                rng = float(np.hypot(obs.x_m, obs.y_m))
            tag = obs.label
            if rng is not None:
                tag += f"  {rng:.1f}m"
            _label_at(display, ui, vi, tag[:28], color)

    if event.grasp is not None:
        best = event.grasp.best_target_id
        if best is None and event.grasp.targets:
            best = 0
        for i, tgt in enumerate(event.grasp.targets):
            u = v = None
            if tgt.x_m is not None and tgt.y_m is not None:
                u, v = base_link_to_pixel(
                    tgt.x_m,
                    tgt.y_m,
                    canvas_w=cw,
                    canvas_h=ch,
                    scale_px_per_meter=scale,
                )
            elif tgt.u_norm is not None and tgt.v_norm is not None:
                u, v = norm_to_pixel(
                    tgt.u_norm, tgt.v_norm, canvas_w=cw, canvas_h=ch
                )
            if u is None or v is None:
                continue
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < cw and 0 <= vi < ch):
                continue
            is_best = best is not None and i == best
            color = _CYAN if is_best else _class_color(tgt.label, accent=True)
            size = 18 if is_best else 12
            _draw_cross(display, ui, vi, color, size=size)
            cv2.circle(display, (ui, vi), size, color, 2 if is_best else 1, cv2.LINE_AA)
            if is_best:
                cv2.arrowedLine(
                    display,
                    (cx, cy),
                    (ui, vi),
                    color,
                    2,
                    tipLength=0.08,
                    line_type=cv2.LINE_AA,
                )
            tag = tgt.label
            if tgt.x_m is not None and tgt.y_m is not None:
                tag += f"  {float(np.hypot(tgt.x_m, tgt.y_m)):.1f}m"
            elif tgt.range_m is not None:
                tag += f"  {tgt.range_m:.1f}m"
            _label_at(display, ui, vi, tag[:28], color)

    return display


def _compass(display: np.ndarray, cw: int, ch: int) -> None:
    marks = (
        ("F", cw // 2, 70, True),
        ("B", cw // 2, ch - 14, True),
        ("L", 14, ch // 2, False),
        ("R", cw - 28, ch // 2, False),
    )
    for text, px, py, center in marks:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        x = int(px - tw // 2) if center else int(px)
        cv2.putText(
            display,
            text,
            (x, int(py)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            _WHITE,
            2,
            cv2.LINE_AA,
        )


def _caption_text(event: Optional[PerceptionEvent]) -> str:
    if event is None:
        return ""
    caption = event.summary or ""
    hint = event.grasp.turn_hint if event.grasp is not None else ""
    if hint and caption.startswith(hint):
        caption = caption[len(hint) :].lstrip(" ·|")
    if not event.valid and event.error:
        caption = f"[{event.error}] {caption}"
    return caption.strip()


def _pretty_chip(raw: str) -> str:
    s = raw.strip()
    low = s.lower()
    if low.startswith("stitch:"):
        return "STITCH " + s.split(":", 1)[1].strip().upper()
    if low.startswith("occ:"):
        return "OCC  " + s.split(":", 1)[1].strip()
    if low.startswith("ov:"):
        return "YOLO  " + s.split(":", 1)[1].strip()
    if low.startswith("vlm"):
        return s.replace("vlm", "VLM").replace("  ", " ")
    return s


def draw_hud(
    bev: np.ndarray,
    *,
    fps_val: float,
    blend_power: float,
    gain_enabled: bool,
    canvas_size: Sequence[int],
    scale: float,
    mode: str,
    event: Optional[PerceptionEvent] = None,
    vlm_status: str = "",
    grasp_target: str = "",
    vehicle_marker: bool = True,
    film: bool = True,
) -> np.ndarray:
    del blend_power, gain_enabled
    cw, ch = int(canvas_size[0]), int(canvas_size[1])
    display = draw_perception_overlay(
        bev,
        event,
        canvas_size=canvas_size,
        scale_px_per_meter=scale,
        vehicle_marker=vehicle_marker,
    )
    _compass(display, cw, ch)
    if not film:
        return display
    return display


def finish_film_frame(
    img: np.ndarray,
    *,
    fps_val: float,
    mode: str,
    range_m: float,
    status_line: str = "",
    event: Optional[PerceptionEvent] = None,
    grasp_target: str = "",
    title: str = "Surround View",
    subtitle: str = "BEV",
) -> np.ndarray:
    """Top title bar + bottom caption. Caption stays on the BEV pane when occ is stacked."""
    display = img
    h, w = display.shape[:2]
    # Live view is BEV | occupancy, both squares. Caption must not paint over occ legend.
    split = h if w >= int(h * 1.35) else w
    top_h = 48
    cap = _caption_text(event)
    font = _cjk_font(size=max(15, split // 42))
    line_h = max(22, int(getattr(font, "size", 16) + 7))
    max_rows = max(2, min(5, (h // 4) // line_h))
    max_chars = max(22, (split - 70) // max(8, int(getattr(font, "size", 16) * 0.55)))
    rows = _wrap_cjk(cap, max_chars)[:max_rows] if cap else []
    bot_h = (22 + line_h * len(rows)) if rows else 0

    _shade(display, (0, 0), (w, top_h), 0.62)
    cv2.rectangle(display, (0, 0), (w, 3), _ORANGE, -1)

    from PIL import Image, ImageDraw

    rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    title_font = _cjk_font(size=max(20, w // 42))
    sub_font = _cjk_font(size=max(14, w // 64))
    draw.text((16, 10), title, font=title_font, fill=(255, 250, 245))
    try:
        bbox = draw.textbbox((16, 10), title, font=title_font)
        title_w = bbox[2] - bbox[0]
    except Exception:
        title_w = 120
    extra = f"Grasp {grasp_target}" if mode == "grasp" and grasp_target else subtitle
    draw.text(
        (16 + title_w + 18, 16),
        f"{extra}   ±{range_m:.1f} m",
        font=sub_font,
        fill=(210, 205, 198),
    )
    del fps_val

    if bot_h:
        y0 = h - bot_h
        overlay = np.asarray(pil)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _shade(overlay, (0, y0), (split, h), 0.62)
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        draw.text((16, y0 + 4), "VLM", font=sub_font, fill=(255, 176, 72))
        for i, row in enumerate(rows):
            draw.text(
                (58, y0 + 4 + i * line_h),
                row,
                font=font,
                fill=(245, 245, 240),
            )

    display = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    x = 16
    y = top_h + 8
    chips = [_pretty_chip(p) for p in status_line.split("  ") if p.strip()]
    for chip in chips[:5]:
        x = _pill(display, (x, y), chip[:28]) + 6
        if x >= split - 8:
            break
    return display
