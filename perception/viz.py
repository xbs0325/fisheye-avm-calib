"""BEV overlay for structured nav/grasp perception events."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from perception.localize import base_link_to_pixel, norm_to_pixel
from perception.schema import PerceptionEvent


def _cjk_font(size: int = 16):
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
    if vehicle_marker:
        cv2.circle(display, (cx, cy), 6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.arrowedLine(display, (cx, cy), (cx, cy - 40), (0, 255, 0), 2, tipLength=0.3)

    if event is None:
        return display

    if event.nav is not None:
        for obs in event.nav.obstacles:
            # Occupancy already tinted as a grid; "occ" crosses duplicate and look like jitter
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
            color = (0, 0, 255)
            _draw_cross(display, ui, vi, color, size=14)
            r_px = int(max(8, (obs.radius_m or 0.25) * scale))
            cv2.circle(display, (ui, vi), r_px, color, 1, cv2.LINE_AA)
            label = f"{obs.label} ({obs.azimuth})"
            if obs.x_m is not None and obs.y_m is not None:
                label += f" {obs.x_m:.2f},{obs.y_m:.2f}m"
            cv2.putText(
                display,
                label[:40],
                (ui + 8, max(14, vi - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

        # Free-dir wedges (subtle)
        dir_angles = {
            "front": (-40, 40),
            "back": (140, 220),
            "left": (50, 130),
            "right": (-130, -50),
        }
        for d in event.nav.free_dirs:
            if d not in dir_angles:
                continue
            a0, a1 = dir_angles[d]
            cv2.ellipse(
                display,
                (cx, cy),
                (cw // 5, ch // 5),
                0,
                a0,
                a1,
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )

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
            color = (0, 255, 255) if is_best else (255, 180, 0)
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
            label = f"{'*' if is_best else ''}{tgt.label}"
            if tgt.yaw_deg is not None:
                side = "L" if tgt.yaw_deg > 12 else ("R" if tgt.yaw_deg < -12 else "F")
                label += f" {side}{abs(tgt.yaw_deg):.0f}d"
            if tgt.x_m is not None and tgt.y_m is not None:
                label += f" ({tgt.x_m:.2f},{tgt.y_m:.2f})m"
            elif tgt.range_m is not None:
                label += f" {tgt.range_m:.1f}m"
            cv2.putText(
                display,
                label[:40],
                (ui + 8, max(14, vi - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    return display


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
) -> np.ndarray:
    cw, ch = int(canvas_size[0]), int(canvas_size[1])
    display = draw_perception_overlay(
        bev,
        event,
        canvas_size=canvas_size,
        scale_px_per_meter=scale,
        vehicle_marker=vehicle_marker,
    )
    lines = [
        f"FPS: {fps_val:.0f}  mode={mode}",
        f"Range: +/-{cw / (2.0 * scale):.1f}m",
        f"Blend: cos^{blend_power:.0f}  Gain: {'ON' if gain_enabled else 'OFF'}",
        "ESC/q:quit  s:save  a:vlm  o:ov  m:map  g:gain  +/-:blend",
    ]
    if mode == "grasp" and grasp_target:
        lines.append(f"target: {grasp_target}")
    if event is not None and event.grasp is not None and event.grasp.turn_hint:
        lines.append(f"TURN {event.grasp.turn_hint}")
    if vlm_status:
        lines.append(vlm_status)
    if event is not None:
        src = ""
        if event.nav is not None and event.nav.source:
            src = f" src={event.nav.source}"
            if event.nav.free_frac is not None:
                src += f" free={event.nav.free_frac:.0%}"
        lines.append(f"valid={event.valid}  schema=v{event.schema_version}{src}")
    for i, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (8, 20 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
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

    caption = ""
    if event is not None:
        caption = event.summary or ""
        hint = event.grasp.turn_hint if event.grasp is not None else ""
        if hint and caption.startswith(hint):
            caption = caption[len(hint) :].lstrip(" ·|")
        if not event.valid and event.error:
            caption = f"[{event.error}] {caption}"
    if caption:
        # ASCII-only captions: skip PIL (was expensive every frame)
        if all(ord(ch) < 128 for ch in caption):
            y0 = ch - 10
            for row in _wrap_cjk(caption, max(20, cw // 10))[-4:][::-1]:
                cv2.putText(
                    display,
                    row,
                    (8, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                y0 -= 18
        else:
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
