"""Classical occupancy grid from stitched BEV (no YOLO / no calib changes).

BEV is already metric via --range/--scale. Cells tile the **full** canvas
(no leftover crop), so overlay stays aligned with the vehicle origin.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from perception.schema import FRAME_ID, SCHEMA_VERSION, NavResult, Obstacle, PerceptionEvent


def _azimuth(u_norm: float, v_norm: float) -> str:
    dx = u_norm - 0.5
    dy = 0.5 - v_norm
    if abs(dx) < 0.12 and abs(dy) < 0.12:
        return "center"
    ang = math.degrees(math.atan2(dx, dy))
    if -22.5 <= ang < 22.5:
        return "f"
    if 22.5 <= ang < 67.5:
        return "fr"
    if 67.5 <= ang < 112.5:
        return "r"
    if 112.5 <= ang < 157.5:
        return "br"
    if ang >= 157.5 or ang < -157.5:
        return "b"
    if -157.5 <= ang < -112.5:
        return "bl"
    if -112.5 <= ang < -67.5:
        return "l"
    return "fl"


@dataclass
class OccupancyGrid:
    """Metric grid covering the full BEV. Cell (i,j): i = +X / image-down, j = +Y / image-right.

    ``cells``: 0 free, 1 occupied, 255 unknown (no coverage).
    ``vehicle_uv``: normalized AABB of the stitched vehicle/blind (u0,v0,u1,v1).
    """

    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    cells: np.ndarray  # (nx, ny) uint8
    free_frac: float
    free_dirs: list[str]
    vehicle_uv: Tuple[float, float, float, float] = (0.38, 0.38, 0.62, 0.62)
    infer_ms: float = 0.0

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.cells.shape[0]), int(self.cells.shape[1])


def _pool(src: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Area-average pool HxW → (nx, ny). Covers the full image (no crop)."""
    out = cv2.resize(src, (ny, nx), interpolation=cv2.INTER_AREA)
    if out.ndim == 2:
        return out.astype(np.float32)
    return out


def _vehicle_uv(gray: np.ndarray, *, blind_frac: float) -> Tuple[float, float, float, float]:
    """Normalized AABB of the dark vehicle square; fallback to centered blind_frac."""
    h, w = gray.shape
    cx, cy = w * 0.5, h * 0.5
    search = max(8, int(min(h, w) * max(blind_frac, 0.18)))
    y0 = max(0, int(cy - search))
    y1 = min(h, int(cy + search))
    x0 = max(0, int(cx - search))
    x1 = min(w, int(cx + search))
    roi = gray[y0:y1, x0:x1]
    dark = roi < 18.0
    if int(dark.sum()) < 40:
        f = float(blind_frac)
        return (0.5 - f, 0.5 - f, 0.5 + f, 0.5 + f)
    ys, xs = np.where(dark)
    # ignore tiny speckles: use percentile bbox
    u0 = (x0 + float(np.percentile(xs, 5))) / float(w)
    u1 = (x0 + float(np.percentile(xs, 95))) / float(w)
    v0 = (y0 + float(np.percentile(ys, 5))) / float(h)
    v1 = (y0 + float(np.percentile(ys, 95))) / float(h)
    # keep it centered-ish; pad a cell-worth so we don't eat the chassis edge
    pad_u = 0.01
    pad_v = 0.01
    u0 = float(np.clip(u0 - pad_u, 0.0, 0.5))
    v0 = float(np.clip(v0 - pad_v, 0.0, 0.5))
    u1 = float(np.clip(u1 + pad_u, 0.5, 1.0))
    v1 = float(np.clip(v1 + pad_v, 0.5, 1.0))
    return (u0, v0, u1, v1)


def _local_floor_gray(
    work: np.ndarray, valid: np.ndarray, *, gh: int = 8, gw: int = 8
) -> np.ndarray:
    """Masked mean floor on a coarse grid; reject object-sized outlier cells."""
    h, w = valid.shape
    vf = valid.astype(np.float32)
    num = cv2.resize(vf, (gw, gh), interpolation=cv2.INTER_AREA)
    acc = cv2.resize(work * vf, (gw, gh), interpolation=cv2.INTER_AREA)
    mean = acc / np.maximum(num, 1e-4)
    mean[num < 0.08] = float(np.median(mean[num >= 0.08])) if np.any(num >= 0.08) else 128.0
    pad = np.pad(mean, 1, mode="edge")
    filled = mean.copy()
    for iy in range(gh):
        for ix in range(gw):
            neigh = pad[iy : iy + 3, ix : ix + 3].reshape(9)
            nmed = float(np.median(np.concatenate([neigh[:4], neigh[5:]])))
            if abs(float(mean[iy, ix]) - nmed) > 16.0:
                filled[iy, ix] = nmed
    return cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)


def _pixel_score(bgr: np.ndarray, *, vehicle_uv: Tuple[float, float, float, float]):
    """Obstacle score vs local gray floor (skip Lab — was ~20ms)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    u0, v0, u1, v1 = vehicle_uv
    seam = cv2.dilate((gray < 14).astype(np.uint8), np.ones((7, 7), np.uint8))
    valid = seam == 0
    valid[int(v0 * h) : int(v1 * h) + 1, int(u0 * w) : int(u1 * w) + 1] = False
    score = np.zeros((h, w), dtype=np.float32)
    if int(valid.sum()) < 500:
        return score, valid, gray

    # Small kernel so feet/shoes are not blurred into the floor.
    k = 9 if min(h, w) >= 240 else 7
    work = cv2.blur(gray.astype(np.float32), (k, k))
    floor = _local_floor_gray(work, valid)
    dL = work - floor
    s_dark = np.clip((-dL - 14.0) / 16.0, 0.0, 1.0)
    s_bright = np.clip((dL - 26.0) / 22.0, 0.0, 1.0)
    # Gray shoes on gray tiles have weak luma; chroma still pops (skin, cloth, rubber).
    b, g, r = cv2.split(bgr.astype(np.float32))
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    ch_blur = cv2.blur(chroma, (k, k))
    floor_c = _local_floor_gray(ch_blur, valid)
    s_ch = np.clip((ch_blur - floor_c - 8.0) / 16.0, 0.0, 1.0)
    score = np.maximum(np.maximum(s_dark, s_bright), s_ch)
    score[~valid] = 0.0
    return score, valid, gray


def estimate_occupancy(
    bev_bgr: np.ndarray,
    *,
    scale_px_per_meter: float,
    resolution_m: float = 0.20,
    blind_frac: float = 0.12,
    occ_thresh: float = 0.42,
    free_thresh: float = 0.85,
    prev_prob: Optional[np.ndarray] = None,
    ema: float = 0.35,
    work_max_side: int = 256,
) -> Tuple[OccupancyGrid, np.ndarray]:
    t0 = time.perf_counter()
    bgr0 = np.asarray(bev_bgr)
    if bgr0.ndim != 3:
        raise ValueError("bev_bgr must be HxWx3")
    h0, w0 = bgr0.shape[:2]
    s0 = float(scale_px_per_meter)
    res = max(0.08, float(resolution_m))

    # Grid tiles the **original** canvas so overlay/HUD share the same origin.
    nx = max(2, int(round((h0 / s0) / res)))
    ny = max(2, int(round((w0 / s0) / res)))
    res_x = (h0 / nx) / s0
    res_y = (w0 / ny) / s0
    res_m = 0.5 * (res_x + res_y)

    max_side = max(64, int(work_max_side))
    if max(h0, w0) > max_side:
        scale_img = max_side / float(max(h0, w0))
        bgr = cv2.resize(
            bgr0,
            (int(round(w0 * scale_img)), int(round(h0 * scale_img))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        bgr = bgr0

    gray_s = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    vehicle_uv = _vehicle_uv(gray_s, blind_frac=blind_frac)
    score, valid, _ = _pixel_score(bgr, vehicle_uv=vehicle_uv)

    valid_f = valid.astype(np.float32)
    known = _pool(valid_f, nx, ny)
    mean_s = _pool(score * valid_f, nx, ny)
    mid = ((score >= 0.28) & valid).astype(np.float32)
    frac = _pool(mid, nx, ny)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_s = np.where(known > 1e-6, mean_s / np.maximum(known, 1e-6), 0.0).astype(
            np.float32
        )
        frac = np.where(known > 1e-6, frac / np.maximum(known, 1e-6), 0.0).astype(
            np.float32
        )
    known_mask = known >= 0.125
    # Mean catches chairs/boxes; fraction catches a foot that only fills part of a cell.
    frac_gate = max(0.10, float(occ_thresh) * 0.28)
    raw = mean_s

    if prev_prob is not None and prev_prob.shape == raw.shape:
        a = float(np.clip(ema, 0.05, 0.95))
        prob = np.where(
            known_mask, (1.0 - a) * prev_prob + a * raw, prev_prob * 0.9
        ).astype(np.float32)
    else:
        prob = raw

    cells = np.zeros((nx, ny), dtype=np.uint8)
    cells[~known_mask] = 255
    cells[known_mask & ((prob >= occ_thresh) | (frac >= frac_gate))] = 1

    # Never treat the vehicle footprint as occupied / unknown-tint
    u0, v0, u1, v1 = vehicle_uv
    iy = (np.arange(ny, dtype=np.float32) + 0.5) / float(ny)
    ix = (np.arange(nx, dtype=np.float32) + 0.5) / float(nx)
    uu, vv = np.meshgrid(iy, ix)
    in_veh = (uu >= u0) & (uu <= u1) & (vv >= v0) & (vv <= v1)
    cells[in_veh] = 0

    free_dirs: list[str] = []
    fracs: list[float] = []
    regions = {
        "front": (slice(0, max(1, nx // 2)), slice(0, ny)),
        "back": (slice(nx // 2, nx), slice(0, ny)),
        "left": (slice(0, nx), slice(0, max(1, ny // 2))),
        "right": (slice(0, nx), slice(ny // 2, ny)),
    }
    for name, (xs, ys) in regions.items():
        region = cells[xs, ys]
        known_r = region != 255
        n = int(known_r.sum())
        if n < 4:
            continue
        free = float((region[known_r] == 0).mean())
        fracs.append(free)
        if free >= free_thresh:
            free_dirs.append(name)
    free_frac = float(np.mean(fracs)) if fracs else 0.0

    # Cell (0,0) center in original pixels
    v_c = (0.5) * (h0 / nx)
    u_c = (0.5) * (w0 / ny)
    origin_x = (h0 * 0.5 - v_c) / s0
    origin_y = (w0 * 0.5 - u_c) / s0

    grid = OccupancyGrid(
        resolution_m=float(res_m),
        origin_x_m=float(origin_x),
        origin_y_m=float(origin_y),
        cells=cells,
        free_frac=free_frac,
        free_dirs=free_dirs,
        vehicle_uv=vehicle_uv,
        infer_ms=(time.perf_counter() - t0) * 1000.0,
    )
    return grid, prob


_AZ_NEIGHBORS = {
    "f": ("f", "fl", "fr"),
    "b": ("b", "bl", "br"),
    "l": ("l", "fl", "bl"),
    "r": ("r", "fr", "br"),
    "fl": ("fl", "f", "l"),
    "fr": ("fr", "f", "r"),
    "bl": ("bl", "b", "l"),
    "br": ("br", "b", "r"),
}


def sector_occupied_cells(grid: OccupancyGrid, azimuth: str) -> int:
    """Occupied cells in an azimuth bin (+ neighbors). Vehicle AABB skipped."""
    az = (azimuth or "").lower().strip()
    if az in ("", "unknown", "center"):
        return 0
    accept = set(_AZ_NEIGHBORS.get(az, (az,)))
    cells = grid.cells
    nx, ny = cells.shape
    ys, xs = np.where(cells == 1)
    if ys.size == 0:
        return 0
    u0, v0, u1, v1 = grid.vehicle_uv
    n = 0
    for i, j in zip(ys.tolist(), xs.tolist()):
        un = (float(j) + 0.5) / float(ny)
        vn = (float(i) + 0.5) / float(nx)
        if u0 <= un <= u1 and v0 <= vn <= v1:
            continue
        if _azimuth(un, vn) in accept:
            n += 1
    return n


def nearest_occupied_in_sector(
    grid: OccupancyGrid, azimuth: str
) -> Optional[Tuple[float, float, float, float]]:
    """Closest occupied cell in an azimuth bin (+ neighbors).

    Returns (x_m, y_m, u_norm, v_norm) or None.
    """
    az = (azimuth or "").lower().strip()
    if az in ("", "unknown", "center"):
        return None
    neighbors = set(_AZ_NEIGHBORS.get(az, (az,)))
    cells = grid.cells
    nx, ny = cells.shape
    rows, cols = np.where(cells == 1)
    if rows.size == 0:
        return None
    u0, v0, u1, v1 = grid.vehicle_uv
    res = float(grid.resolution_m)
    best_exact: Optional[Tuple[float, float, float, float, float]] = None
    best_nb: Optional[Tuple[float, float, float, float, float]] = None
    for i, j in zip(rows.tolist(), cols.tolist()):
        un = (float(j) + 0.5) / float(ny)
        vn = (float(i) + 0.5) / float(nx)
        if u0 <= un <= u1 and v0 <= vn <= v1:
            continue
        x_m = float(grid.origin_x_m) - float(i) * res
        y_m = float(grid.origin_y_m) - float(j) * res
        rng = math.hypot(x_m, y_m)
        if rng < 0.15:
            continue
        rec = (rng, x_m, y_m, un, vn)
        cell_az = _azimuth(un, vn)
        if cell_az == az:
            if best_exact is None or rng < best_exact[0]:
                best_exact = rec
        elif cell_az in neighbors:
            if best_nb is None or rng < best_nb[0]:
                best_nb = rec
    best = best_exact or best_nb
    if best is None:
        return None
    _, x_m, y_m, un, vn = best
    return x_m, y_m, un, vn


def snap_grasp_to_occupancy(
    event: PerceptionEvent,
    grid: OccupancyGrid,
    *,
    max_range_m: float = 1.6,
) -> PerceptionEvent:
    """Refine grasp x/y to the nearest occupied cell in that sector (near field)."""
    if event.grasp is None or not event.grasp.targets:
        return event
    for tgt in event.grasp.targets:
        hit = nearest_occupied_in_sector(grid, tgt.azimuth)
        if hit is None:
            continue
        x_m, y_m, un, vn = hit
        rng = math.hypot(x_m, y_m)
        if rng > max_range_m:
            continue
        tgt.x_m = round(float(x_m), 3)
        tgt.y_m = round(float(y_m), 3)
        tgt.u_norm = round(float(un), 4)
        tgt.v_norm = round(float(vn), 4)
        tgt.range_m = round(rng, 3)
        tgt.yaw_deg = math.degrees(math.atan2(float(y_m), float(x_m)))
    return event


def apply_occ_veto(event: PerceptionEvent, grid: OccupancyGrid) -> PerceptionEvent:
    """Drop grasp targets only when that whole side is empty (keep tiny objects)."""
    if event.grasp is None or not event.grasp.targets:
        return event
    tgt = event.grasp.targets[0]
    n = sector_occupied_cells(grid, tgt.azimuth)
    if n >= 1:
        return event
    event.grasp.targets = []
    event.grasp.best_target_id = None
    event.grasp.turn_hint = ""
    event.grasp.notes = "未找到"
    event.grasp.summary = "未找到"
    event.summary = "未找到"
    return event


def grid_to_obstacles(
    grid: OccupancyGrid,
    *,
    max_blobs: int = 8,
    min_cells: int = 2,
) -> list[Obstacle]:
    cells = grid.cells
    nx, ny = cells.shape
    occ = cells == 1
    if not occ.any():
        return []

    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        occ.astype(np.uint8), connectivity=8
    )
    comps: list[Tuple[float, float, int]] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_cells:
            continue
        miy, mix = float(centroids[i, 0]), float(centroids[i, 1])
        comps.append((mix, miy, area))

    comps.sort(key=lambda t: t[2], reverse=True)
    out: list[Obstacle] = []
    res = grid.resolution_m
    for mix, miy, ncell in comps[:max_blobs]:
        x_m = grid.origin_x_m - mix * res
        y_m = grid.origin_y_m - miy * res
        u_norm = (miy + 0.5) / float(ny)
        v_norm = (mix + 0.5) / float(nx)
        if abs(u_norm - 0.5) < 0.08 and abs(v_norm - 0.5) < 0.08:
            continue
        radius_m = max(res * 0.6, math.sqrt(ncell) * res * 0.45)
        out.append(
            Obstacle(
                label="occ",
                azimuth=_azimuth(u_norm, v_norm),
                u_norm=round(float(u_norm), 4),
                v_norm=round(float(v_norm), 4),
                conf=0.55,
                x_m=round(float(x_m), 3),
                y_m=round(float(y_m), 3),
                radius_m=round(float(radius_m), 3),
            )
        )
    return out


def grid_to_event(
    grid: OccupancyGrid,
    *,
    stamp_s: Optional[float] = None,
) -> PerceptionEvent:
    obstacles = grid_to_obstacles(grid)
    if obstacles:
        summary = f"occ:{len(obstacles)} free={grid.free_frac:.0%}"
    else:
        summary = f"occ:clear free={grid.free_frac:.0%}"
    return PerceptionEvent(
        schema_version=SCHEMA_VERSION,
        frame_id=FRAME_ID,
        stamp_s=float(stamp_s if stamp_s is not None else time.time()),
        mode="nav",
        valid=True,
        infer_ms=float(grid.infer_ms),
        summary=summary,
        nav=NavResult(
            summary=summary,
            obstacles=obstacles,
            free_dirs=list(grid.free_dirs),
            free_frac=round(float(grid.free_frac), 3),
            source="occ",
        ),
    )


def stamp_detections_on_grid(
    grid: OccupancyGrid,
    event: Optional[PerceptionEvent],
) -> OccupancyGrid:
    """Paint YOLO / VLM boxes onto the occupancy grid (feet, person, …)."""
    if event is None or grid.cells.size == 0:
        return grid
    hits: list[tuple[Optional[float], Optional[float], Optional[float], Optional[float], float]] = []
    if event.nav is not None:
        for o in event.nav.obstacles:
            if o.label in ("occ", "ego", "vehicle"):
                continue
            rad = float(o.radius_m) if o.radius_m else 0.22
            lab = (o.label or "").lower()
            if any(k in lab for k in ("person", "shoe", "foot", "人", "脚", "鞋")):
                rad = max(rad, 0.28)
            hits.append((o.u_norm, o.v_norm, o.x_m, o.y_m, rad))
    if event.grasp is not None:
        for t in event.grasp.targets:
            hits.append((t.u_norm, t.v_norm, t.x_m, t.y_m, 0.18))
    if not hits:
        return grid

    nx, ny = grid.cells.shape
    res = max(float(grid.resolution_m), 0.05)
    u0, v0, u1, v1 = grid.vehicle_uv
    cells = grid.cells

    def _mark(i: int, j: int, rc: int) -> None:
        i0, i1 = max(0, i - rc), min(nx, i + rc + 1)
        j0, j1 = max(0, j - rc), min(ny, j + rc + 1)
        for ii in range(i0, i1):
            for jj in range(j0, j1):
                if (ii - i) * (ii - i) + (jj - j) * (jj - j) > rc * rc + 1:
                    continue
                un = (float(jj) + 0.5) / float(ny)
                vn = (float(ii) + 0.5) / float(nx)
                if u0 <= un <= u1 and v0 <= vn <= v1:
                    continue
                if cells[ii, jj] != 255:
                    cells[ii, jj] = 1

    for u_n, v_n, x_m, y_m, rad in hits:
        rc = max(1, int(round(float(rad) / res)))
        i = j = None
        if u_n is not None and v_n is not None:
            j = int(np.clip(float(u_n) * ny, 0, ny - 1))
            i = int(np.clip(float(v_n) * nx, 0, nx - 1))
        elif x_m is not None and y_m is not None:
            i = int(round((float(grid.origin_x_m) - float(x_m)) / res))
            j = int(round((float(grid.origin_y_m) - float(y_m)) / res))
            if not (0 <= i < nx and 0 <= j < ny):
                continue
        if i is None or j is None:
            continue
        _mark(int(i), int(j), rc)
    return grid


def overlay_occupancy(
    bev_bgr: np.ndarray,
    grid: OccupancyGrid,
    *,
    alpha: float = 0.35,
    draw_vehicle_box: bool = True,
) -> np.ndarray:
    """Tint occupied cells; draw vehicle AABB in image-normalized coords (centered)."""
    h, w = bev_bgr.shape[:2]
    occ = (grid.cells == 1).astype(np.uint8) * 255
    if cv2.countNonZero(occ) > 0:
        mask = cv2.resize(occ, (w, h), interpolation=cv2.INTER_NEAREST)
        tint = np.zeros_like(bev_bgr)
        tint[:, :] = (0, 0, 220)
        blended = cv2.addWeighted(bev_bgr, 1.0 - float(alpha), tint, float(alpha), 0)
        inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(bev_bgr, bev_bgr, mask=inv)
        fg = cv2.bitwise_and(blended, blended, mask=mask)
        out = cv2.add(bg, fg)
    else:
        out = bev_bgr.copy()

    if draw_vehicle_box:
        u0, v0, u1, v1 = grid.vehicle_uv
        x0, y0 = int(round(u0 * w)), int(round(v0 * h))
        x1, y1 = int(round(u1 * w)), int(round(v1 * h))
        overlay = out.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
        cv2.rectangle(out, (x0, y0), (x1, y1), (80, 80, 80), 1)
    return out


def _cardinal4(u_norm: float, v_norm: float) -> str:
    """F/B/L/R from BEV uv. No center deadzone — vehicle AABB is skipped separately."""
    dx = float(u_norm) - 0.5
    dy = 0.5 - float(v_norm)
    if abs(dy) >= abs(dx):
        return "f" if dy >= 0.0 else "b"
    return "r" if dx >= 0.0 else "l"


def _nearest_cardinal_m(grid: OccupancyGrid) -> dict[str, Optional[float]]:
    """Closest occupied range (m) in F/B/L/R, skipping the vehicle AABB."""
    cells = grid.cells
    nx, ny = cells.shape
    res = float(grid.resolution_m)
    occ_i, occ_j = np.where(cells == 1)
    out: dict[str, Optional[float]] = {"f": None, "b": None, "l": None, "r": None}
    if occ_i.size == 0:
        return out
    u0, v0, u1, v1 = grid.vehicle_uv
    un = (occ_j.astype(np.float32) + 0.5) / float(ny)
    vn = (occ_i.astype(np.float32) + 0.5) / float(nx)
    keep = ~((un >= u0) & (un <= u1) & (vn >= v0) & (vn <= v1))
    occ_i = occ_i[keep]
    occ_j = occ_j[keep]
    un = un[keep]
    vn = vn[keep]
    if occ_i.size == 0:
        return out
    ci = 0.5 * (nx - 1)
    cj = 0.5 * (ny - 1)
    rng = np.hypot((ci - occ_i) * res, (cj - occ_j) * res)
    for i in range(occ_i.size):
        card = _cardinal4(float(un[i]), float(vn[i]))
        d = float(rng[i])
        prev = out[card]
        if prev is None or d < prev:
            out[card] = d
    return out


def render_occupancy_map(
    grid: OccupancyGrid,
    *,
    event: Optional[PerceptionEvent] = None,
    size: int = 480,
) -> np.ndarray:
    """Top-down 2D occupancy panel (same frame as BEV: up=forward)."""
    s = max(160, int(size))
    cells = grid.cells
    nx, ny = int(cells.shape[0]), int(cells.shape[1])
    lut = np.zeros((nx, ny, 3), dtype=np.uint8)
    lut[cells == 0] = (200, 175, 70)
    lut[cells == 1] = (40, 40, 210)
    lut[cells == 255] = (210, 210, 210)
    occ = (cells == 1).astype(np.uint8) * 255
    if int(cv2.countNonZero(occ)) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        occ = cv2.dilate(occ, k, iterations=1)
        lut[occ > 0] = (40, 40, 210)
    img = cv2.resize(lut, (s, s), interpolation=cv2.INTER_NEAREST)

    cx = cy = s // 2
    span_m = max(nx, ny) * float(grid.resolution_m)
    px_per_m = float(s) / max(span_m, 1e-3)
    for r_m in (0.5, 1.0, 1.5, 2.0, 2.5):
        rad = int(round(r_m * px_per_m))
        if rad < 8 or rad >= s // 2 - 4:
            continue
        cv2.circle(img, (cx, cy), rad, (160, 140, 90), 1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"{r_m:g}m",
            (cx + 4, max(12, cy - rad - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (90, 70, 40),
            1,
            cv2.LINE_AA,
        )

    tri = np.array(
        [[cx, cy - 14], [cx - 9, cy + 11], [cx + 9, cy + 11]], dtype=np.int32
    )
    cv2.fillConvexPoly(img, tri, (0, 210, 0))
    cv2.polylines(img, [tri], True, (220, 255, 220), 1, cv2.LINE_AA)

    for txt, pos in (
        ("F", (cx - 5, 18)),
        ("B", (cx - 5, s - 8)),
        ("L", (8, cy + 5)),
        ("R", (s - 22, cy + 5)),
    ):
        cv2.putText(
            img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 80, 160), 2, cv2.LINE_AA
        )

    if event is not None:
        hits: list[tuple[float, float, tuple[int, int, int], int]] = []
        if event.grasp is not None:
            best = event.grasp.best_target_id
            for k, t in enumerate(event.grasp.targets):
                if t.x_m is None or t.y_m is None:
                    continue
                col = (0, 255, 255) if k == best or t.graspable else (0, 128, 255)
                hits.append((float(t.x_m), float(t.y_m), col, 7 if k == best else 5))
        if event.nav is not None:
            for o in event.nav.obstacles:
                if o.label == "occ" or o.x_m is None or o.y_m is None:
                    continue
                hits.append((float(o.x_m), float(o.y_m), (0, 140, 255), 5))
        for x_m, y_m, col, rad in hits:
            u = int(round(cx - y_m * px_per_m))
            v = int(round(cy - x_m * px_per_m))
            if 2 <= u < s - 2 and 2 <= v < s - 2:
                cv2.circle(img, (u, v), rad, col, 2, cv2.LINE_AA)

    near = _nearest_cardinal_m(grid)
    y0 = 36
    cv2.putText(
        img,
        "BEV MAP  floor/occ",
        (8, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (40, 90, 40),
        1,
        cv2.LINE_AA,
    )
    y0 += 18
    cv2.putText(
        img,
        f"free={grid.free_frac:.0%}  {grid.resolution_m:.2f}m/cell",
        (8, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (40, 90, 40),
        1,
        cv2.LINE_AA,
    )
    y0 += 16
    go = ",".join(grid.free_dirs) if grid.free_dirs else "-"
    cv2.putText(
        img,
        f"GO {go}",
        (8, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (30, 140, 30) if grid.free_dirs else (0, 0, 180),
        1,
        cv2.LINE_AA,
    )
    y0 += 16
    bits = []
    for k, lab in (("f", "F"), ("b", "B"), ("l", "L"), ("r", "R")):
        d = near.get(k)
        bits.append(f"{lab}:{d:.1f}m" if d is not None else f"{lab}:--")
    cv2.putText(
        img,
        " ".join(bits),
        (8, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (40, 90, 160),
        1,
        cv2.LINE_AA,
    )
    cv2.line(img, (0, 0), (0, s - 1), (80, 80, 80), 2)
    return img


class OccupancyTracker:
    def __init__(
        self,
        *,
        scale_px_per_meter: float,
        resolution_m: float = 0.20,
        blind_frac: float = 0.12,
        occ_thresh: float = 0.42,
        ema: float = 0.35,
        log_every_s: float = 1.0,
    ) -> None:
        self.scale = float(scale_px_per_meter)
        self.resolution_m = float(resolution_m)
        self.blind_frac = float(blind_frac)
        self.occ_thresh = float(occ_thresh)
        self.ema = float(ema)
        self.log_every_s = float(log_every_s)
        self._prob: Optional[np.ndarray] = None
        self.latest_grid: Optional[OccupancyGrid] = None
        self.latest_event: Optional[PerceptionEvent] = None
        self._last_log_t = 0.0

    def update(self, bev_bgr: np.ndarray) -> Tuple[OccupancyGrid, PerceptionEvent, bool]:
        grid, self._prob = estimate_occupancy(
            bev_bgr,
            scale_px_per_meter=self.scale,
            resolution_m=self.resolution_m,
            blind_frac=self.blind_frac,
            occ_thresh=self.occ_thresh,
            prev_prob=self._prob,
            ema=self.ema,
        )
        event = grid_to_event(grid)
        self.latest_grid = grid
        self.latest_event = event
        now = time.time()
        should_log = (now - self._last_log_t) >= self.log_every_s
        if should_log:
            self._last_log_t = now
        return grid, event, should_log
