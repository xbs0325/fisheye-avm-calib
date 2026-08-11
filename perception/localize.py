"""BEV pixel / normalized coords → base_link metric (ground plane)."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from perception.schema import GraspTarget, Obstacle, PerceptionEvent

# Coarse yaw if VLM only gave a compass bin (no pixel).
_AZ_YAW_DEG = {
    "f": 0.0,
    "fr": -45.0,
    "r": -90.0,
    "br": -135.0,
    "b": 180.0,
    "bl": 135.0,
    "l": 90.0,
    "fl": 45.0,
}

# Image: u right, v down. Vehicle: +X forward (up in image), +Y left (left in image).
# Origin at canvas center (vehicle center / blind spot).


def pixel_to_base_link(
    u: float,
    v: float,
    *,
    canvas_w: int,
    canvas_h: int,
    scale_px_per_meter: float,
) -> Tuple[float, float]:
    """Return (x_m, y_m) in base_link."""
    s = float(scale_px_per_meter)
    if s <= 0:
        raise ValueError(f"invalid scale_px_per_meter={scale_px_per_meter}")
    cx = canvas_w * 0.5
    cy = canvas_h * 0.5
    x_m = (cy - float(v)) / s
    y_m = (cx - float(u)) / s
    return x_m, y_m


def base_link_to_pixel(
    x_m: float,
    y_m: float,
    *,
    canvas_w: int,
    canvas_h: int,
    scale_px_per_meter: float,
) -> Tuple[float, float]:
    s = float(scale_px_per_meter)
    if s <= 0:
        raise ValueError(f"invalid scale_px_per_meter={scale_px_per_meter}")
    cx = canvas_w * 0.5
    cy = canvas_h * 0.5
    u = cx - float(y_m) * s
    v = cy - float(x_m) * s
    return u, v


def norm_to_pixel(
    u_norm: float,
    v_norm: float,
    *,
    canvas_w: int,
    canvas_h: int,
) -> Tuple[float, float]:
    return float(u_norm) * float(canvas_w), float(v_norm) * float(canvas_h)


def heading_deg(x_m: float, y_m: float) -> float:
    """Yaw to face a ground point. 0=forward, +left, −right (ROS base_link)."""
    return math.degrees(math.atan2(float(y_m), float(x_m)))


def clock_hour(yaw_deg: float) -> int:
    """12 o'clock = forward; 3 = right; 9 = left."""
    h = int(round((12.0 - float(yaw_deg) / 30.0) % 12.0))
    return 12 if h == 0 else h


DEFAULT_GRASP_RANGE_M = 0.7


def format_turn_hint(
    *,
    yaw_deg: Optional[float],
    range_m: Optional[float] = None,
    x_m: Optional[float] = None,
    y_m: Optional[float] = None,
) -> str:
    """ASCII HUD line: L35 0.70m (0.50,-0.49). OpenCV putText cannot draw CJK."""
    if yaw_deg is None:
        return ""
    yaw = float(yaw_deg)
    if abs(yaw) < 12.0:
        turn = "F"
    elif yaw > 0:
        turn = f"L{yaw:.0f}"
    else:
        turn = f"R{abs(yaw):.0f}"
    extra = ""
    if range_m is not None:
        extra += f" {range_m:.2f}m"
    if x_m is not None and y_m is not None:
        extra += f" ({x_m:.2f},{y_m:.2f})"
    return f"{turn}{extra}"


def in_center_blind(
    u: float,
    v: float,
    *,
    canvas_w: int,
    canvas_h: int,
    blind_frac: float = 0.12,
) -> bool:
    """True if point lies in the central vehicle / seam blind region."""
    cx = canvas_w * 0.5
    cy = canvas_h * 0.5
    rx = canvas_w * float(blind_frac)
    ry = canvas_h * float(blind_frac)
    return abs(float(u) - cx) <= rx and abs(float(v) - cy) <= ry


def _enrich_obstacle(
    obs: Obstacle,
    *,
    canvas_w: int,
    canvas_h: int,
    scale: float,
    blind_frac: float,
) -> Obstacle:
    if obs.u_norm is None or obs.v_norm is None:
        return obs
    u, v = norm_to_pixel(obs.u_norm, obs.v_norm, canvas_w=canvas_w, canvas_h=canvas_h)
    if in_center_blind(u, v, canvas_w=canvas_w, canvas_h=canvas_h, blind_frac=blind_frac):
        # Keep norms but clear metric (likely blind / vehicle body)
        obs.x_m = None
        obs.y_m = None
        return obs
    x_m, y_m = pixel_to_base_link(
        u, v, canvas_w=canvas_w, canvas_h=canvas_h, scale_px_per_meter=scale
    )
    obs.x_m = x_m
    obs.y_m = y_m
    if obs.radius_m is None:
        obs.radius_m = 0.25
    return obs


def _enrich_target(
    tgt: GraspTarget,
    *,
    canvas_w: int,
    canvas_h: int,
    scale: float,
    blind_frac: float,
) -> GraspTarget:
    if tgt.u_norm is None or tgt.v_norm is None:
        return tgt
    u, v = norm_to_pixel(tgt.u_norm, tgt.v_norm, canvas_w=canvas_w, canvas_h=canvas_h)
    # Grasp objects often sit next to the chassis; still emit metric xy.
    x_m, y_m = pixel_to_base_link(
        u, v, canvas_w=canvas_w, canvas_h=canvas_h, scale_px_per_meter=scale
    )
    tgt.x_m = x_m
    tgt.y_m = y_m
    tgt.range_m = math.hypot(x_m, y_m)
    tgt.yaw_deg = heading_deg(x_m, y_m)
    return tgt


def _angle_diff(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _fill_target_heading(tgt: GraspTarget) -> GraspTarget:
    if tgt.yaw_deg is None and tgt.x_m is not None and tgt.y_m is not None:
        tgt.range_m = math.hypot(tgt.x_m, tgt.y_m)
        tgt.yaw_deg = heading_deg(tgt.x_m, tgt.y_m)
    coarse = _AZ_YAW_DEG.get(tgt.azimuth)
    # VLM pixels are noisy; if they disagree with the stated side, trust azimuth.
    if (
        coarse is not None
        and tgt.yaw_deg is not None
        and tgt.u_norm is None
        and _angle_diff(tgt.yaw_deg, coarse) > 55.0
    ):
        tgt.x_m = None
        tgt.y_m = None
        tgt.u_norm = None
        tgt.v_norm = None
        tgt.range_m = None
        tgt.yaw_deg = coarse
    if tgt.yaw_deg is None and coarse is not None:
        tgt.yaw_deg = coarse
    return tgt


def _fill_azimuth_xy(
    tgt: GraspTarget,
    *,
    canvas_w: int,
    canvas_h: int,
    scale: float,
    default_range_m: float = DEFAULT_GRASP_RANGE_M,
) -> GraspTarget:
    """If VLM only gave a compass bin, place a coarse ground point on that ray."""
    if tgt.x_m is not None and tgt.y_m is not None:
        return tgt
    yaw = tgt.yaw_deg
    if yaw is None:
        yaw = _AZ_YAW_DEG.get(tgt.azimuth)
    if yaw is None:
        return tgt
    rng = float(tgt.range_m) if tgt.range_m is not None else float(default_range_m)
    rad = math.radians(float(yaw))
    tgt.x_m = rng * math.cos(rad)
    tgt.y_m = rng * math.sin(rad)
    tgt.range_m = rng
    tgt.yaw_deg = float(yaw)
    if canvas_w > 0 and canvas_h > 0 and scale > 0:
        u, v = base_link_to_pixel(
            tgt.x_m,
            tgt.y_m,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            scale_px_per_meter=scale,
        )
        tgt.u_norm = float(u) / float(canvas_w)
        tgt.v_norm = float(v) / float(canvas_h)
    return tgt


def smooth_grasp_heading(
    event: PerceptionEvent,
    *,
    yaw_ema: Optional[float],
    range_ema: Optional[float],
    alpha: float = 0.35,
    max_jump_deg: float = 75.0,
) -> tuple[PerceptionEvent, Optional[float], Optional[float]]:
    """Light EMA. Miss / empty targets clear the track so HUD does not keep a ghost."""
    if event.grasp is None or not event.grasp.targets:
        if event.grasp is not None:
            event.grasp.turn_hint = ""
            event.summary = event.grasp.notes or "未找到"
        return event, None, None
    idx = event.grasp.best_target_id
    if idx is None:
        idx = 0
    if not (0 <= idx < len(event.grasp.targets)):
        return event, yaw_ema, range_ema
    tgt = event.grasp.targets[idx]
    if tgt.yaw_deg is None:
        return event, yaw_ema, range_ema
    yaw = float(tgt.yaw_deg)
    if yaw_ema is not None and _angle_diff(yaw, yaw_ema) > max_jump_deg:
        yaw = yaw_ema
    elif yaw_ema is not None:
        delta = (yaw - yaw_ema + 180.0) % 360.0 - 180.0
        yaw = yaw_ema + alpha * delta
    rng = tgt.range_m
    if rng is not None:
        rng = float(rng)
        if range_ema is not None:
            rng = range_ema * (1.0 - alpha) + rng * alpha
        range_ema = rng
        tgt.range_m = rng
    tgt.yaw_deg = yaw
    if tgt.range_m is not None:
        rad = math.radians(yaw)
        tgt.x_m = float(tgt.range_m) * math.cos(rad)
        tgt.y_m = float(tgt.range_m) * math.sin(rad)
    hint = format_turn_hint(
        yaw_deg=yaw, range_m=tgt.range_m, x_m=tgt.x_m, y_m=tgt.y_m
    )
    event.grasp.turn_hint = hint
    event.grasp.summary = hint
    # Keep notes in event.summary so HUD caption ≠ TURN line
    event.summary = (event.grasp.notes or hint).strip()
    return event, yaw, range_ema


def enrich_event(
    event: PerceptionEvent,
    *,
    canvas_size: Sequence[int],
    scale_px_per_meter: float,
    blind_frac: float = 0.12,
) -> PerceptionEvent:
    """Fill x_m/y_m on obstacles/targets from u_norm/v_norm."""
    cw, ch = int(canvas_size[0]), int(canvas_size[1])
    scale = float(scale_px_per_meter)
    if event.nav is not None:
        event.nav.obstacles = [
            _enrich_obstacle(
                o, canvas_w=cw, canvas_h=ch, scale=scale, blind_frac=blind_frac
            )
            for o in event.nav.obstacles
        ]
    if event.grasp is not None:
        event.grasp.targets = [
            _fill_azimuth_xy(
                _fill_target_heading(
                    _enrich_target(
                        t, canvas_w=cw, canvas_h=ch, scale=scale, blind_frac=blind_frac
                    )
                ),
                canvas_w=cw,
                canvas_h=ch,
                scale=scale,
            )
            for t in event.grasp.targets
        ]
        refresh_grasp_hint(event)
    return event


def refresh_grasp_hint(event: PerceptionEvent) -> PerceptionEvent:
    """Recompute TURN line from the best target's yaw / range / xy."""
    if event.grasp is None:
        return event
    best_i = event.grasp.best_target_id
    if best_i is None and event.grasp.targets:
        best_i = 0
    hint = ""
    if best_i is not None and 0 <= best_i < len(event.grasp.targets):
        best = event.grasp.targets[best_i]
        hint = format_turn_hint(
            yaw_deg=best.yaw_deg,
            range_m=best.range_m,
            x_m=best.x_m,
            y_m=best.y_m,
        )
    event.grasp.turn_hint = hint
    event.grasp.summary = hint or (event.grasp.notes or "")
    event.summary = (event.grasp.notes or hint).strip()
    return event
