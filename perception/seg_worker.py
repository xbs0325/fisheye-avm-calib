#!/usr/bin/env python3
"""YOLO-seg navigation worker (subprocess, torch venv). No OpenCV required.

Protocol:
  Worker -> READY
  Parent -> SEG /abs/path.jpg <canvas_w> <canvas_h> <scale_px_per_m>
  Worker -> OK <ms>
           <one-line JSON>
           END
  Parent -> QUIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Broader COCO set for indoor BEV (miss thin chair legs → classical fallback)
NAV_CLASS_IDS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    13: "bench",
    24: "backpack",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
}


def _azimuth(u_norm: float, v_norm: float) -> str:
    # Image: up=front. Relative to center.
    dx = u_norm - 0.5  # +right
    dy = 0.5 - v_norm  # +front
    if abs(dx) < 0.12 and abs(dy) < 0.12:
        return "center"
    # 8-way
    import math

    ang = math.degrees(math.atan2(dx, dy))  # 0=front, + = right
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


def _free_dirs_from_mask(
    occ: Any, *, blind_frac: float = 0.12, free_thresh: float = 0.88
) -> tuple[list[str], float]:
    """Quadrant free-space fractions from occupancy mask (1=obstacle)."""
    import numpy as np

    h, w = occ.shape
    cx, cy = w // 2, h // 2
    rx, ry = int(w * blind_frac), int(h * blind_frac)
    valid = np.ones_like(occ, dtype=bool)
    valid[cy - ry : cy + ry + 1, cx - rx : cx + rx + 1] = False

    quads = {
        "front": (slice(0, cy), slice(0, w)),
        "back": (slice(cy, h), slice(0, w)),
        "left": (slice(0, h), slice(0, cx)),
        "right": (slice(0, h), slice(cx, w)),
    }
    free_dirs: list[str] = []
    fracs = []
    for name, (ys, xs) in quads.items():
        region = occ[ys, xs]
        v = valid[ys, xs]
        n = int(v.sum())
        if n < 50:
            continue
        free = 1.0 - float(region[v].mean())
        fracs.append(free)
        if free >= free_thresh:
            free_dirs.append(name)
    overall = float(np.mean(fracs)) if fracs else 0.0
    return free_dirs, overall


def _classical_occupancy(rgb: Any, *, blind_frac: float = 0.12) -> Any:
    """Cheap non-floor occupancy for BEV (YOLO often misses chair legs)."""
    import numpy as np

    arr = np.asarray(rgb, dtype=np.float32)
    gray = arr.mean(axis=2)
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    rx, ry = int(w * blind_frac), int(h * blind_frac)

    seam = gray < 14.0
    valid = ~seam
    valid[cy - ry : cy + ry + 1, cx - rx : cx + rx + 1] = False
    if int(valid.sum()) < 500:
        return np.zeros_like(gray, dtype=np.float32)

    floor = float(np.median(gray[valid]))
    # Darker / brighter than floor, or strong local contrast
    diff = np.abs(gray - floor)
    # Local contrast via simple box approx: |g - mean|
    # downsample for speed
    step = max(1, min(h, w) // 160)
    small = gray[::step, ::step]
    # 3x3 mean
    pad = np.pad(small, 1, mode="edge")
    mean = (
        pad[0:-2, 0:-2]
        + pad[0:-2, 1:-1]
        + pad[0:-2, 2:]
        + pad[1:-1, 0:-2]
        + pad[1:-1, 1:-1]
        + pad[1:-1, 2:]
        + pad[2:, 0:-2]
        + pad[2:, 1:-1]
        + pad[2:, 2:]
    ) / 9.0
    local = np.abs(small - mean)
    # upsample local contrast
    local_up = np.repeat(np.repeat(local, step, axis=0), step, axis=1)[:h, :w]

    # Tuned for light indoor floors: catch people/chairs; ignore light tile texture
    occ = ((diff > 40.0) | (local_up > 26.0)) & valid
    # Light erosion/dilation via neighbor majority (1-iter)
    occ_u8 = occ.astype(np.uint8)
    pad2 = np.pad(occ_u8, 1, mode="constant")
    neigh = (
        pad2[0:-2, 0:-2]
        + pad2[0:-2, 1:-1]
        + pad2[0:-2, 2:]
        + pad2[1:-1, 0:-2]
        + pad2[1:-1, 1:-1]
        + pad2[1:-1, 2:]
        + pad2[2:, 0:-2]
        + pad2[2:, 1:-1]
        + pad2[2:, 2:]
    )
    # require denser support → fewer false clutter
    occ2 = (neigh >= 6).astype(np.float32)
    occ2[cy - ry : cy + ry + 1, cx - rx : cx + rx + 1] = 0.0
    occ2[seam] = 0.0
    return occ2


def _blobs_to_obstacles(
    occ: Any,
    *,
    scale: float,
    cw: int,
    ch: int,
    min_area: int = 80,
    max_blobs: int = 6,
    label: str = "clutter",
    conf: float = 0.4,
) -> list[dict[str, Any]]:
    import numpy as np

    # Connected components via iterative flood (numpy only, coarse)
    vis = (occ > 0.5).astype(np.uint8)
    h, w = vis.shape
    obstacles: list[dict[str, Any]] = []
    # stride sampling to find seeds (faster than full scan flood for many pixels)
    ys, xs = np.where(vis > 0)
    if len(xs) == 0:
        return []
    # Use scipy-free simple: bin into grid cells then merge neighbors
    cell = max(8, min(h, w) // 40)
    gh, gw = (h + cell - 1) // cell, (w + cell - 1) // cell
    grid = np.zeros((gh, gw), dtype=np.uint8)
    for y, x in zip(ys[::3], xs[::3]):
        grid[y // cell, x // cell] = 1

    seen = np.zeros_like(grid)
    comps: list[tuple[float, float, int]] = []  # cy, cx, area_cells
    for gy in range(gh):
        for gx in range(gw):
            if grid[gy, gx] == 0 or seen[gy, gx]:
                continue
            # BFS
            stack = [(gy, gx)]
            seen[gy, gx] = 1
            cells = [(gy, gx)]
            while stack:
                cy0, cx0 = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy0 + dy, cx0 + dx
                        if 0 <= ny < gh and 0 <= nx < gw and grid[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = 1
                            stack.append((ny, nx))
                            cells.append((ny, nx))
            area_px = len(cells) * cell * cell
            if area_px < min_area:
                continue
            my = (sum(c[0] for c in cells) / len(cells) + 0.5) * cell
            mx = (sum(c[1] for c in cells) / len(cells) + 0.5) * cell
            comps.append((my, mx, area_px))

    comps.sort(key=lambda t: t[2], reverse=True)
    for my, mx, area in comps[:max_blobs]:
        u_n, v_n = float(mx) / float(cw), float(my) / float(ch)
        if abs(u_n - 0.5) < 0.10 and abs(v_n - 0.5) < 0.10:
            continue
        x_m = (ch * 0.5 - my) / scale
        y_m = (cw * 0.5 - mx) / scale
        radius_m = max(0.12, (area ** 0.5) / scale * 0.45)
        obstacles.append(
            {
                "label": label,
                "azimuth": _azimuth(u_n, v_n),
                "u_norm": round(u_n, 4),
                "v_norm": round(v_n, 4),
                "conf": conf,
                "x_m": round(float(x_m), 3),
                "y_m": round(float(y_m), 3),
                "radius_m": round(float(radius_m), 3),
            }
        )
    return obstacles


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO-seg nav worker")
    ap.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "models"
        / "perception"
        / "yolov8n-seg.pt",
    )
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="0")
    ap.add_argument(
        "--classic",
        action="store_true",
        default=True,
        help="近场几何占有（补椅腿等 YOLO 漏检）",
    )
    ap.add_argument("--no-classic", dest="classic", action="store_false")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    # Avoid system site matplotlib/numpy clashes
    if "PYTHONPATH" in os.environ:
        parts = [p for p in os.environ["PYTHONPATH"].split(":") if "avm_gpu" in p]
        os.environ["PYTHONPATH"] = ":".join(parts)

    try:
        import numpy as np
        import torch
        from PIL import Image
        from ultralytics import YOLO

        device = args.device
        if device not in ("cpu", "CPU") and not torch.cuda.is_available():
            device = "cpu"
        # Leave headroom for CUDA OpenCV stitch on same GPU
        if device not in ("cpu", "CPU"):
            try:
                torch.cuda.set_per_process_memory_fraction(0.35, device=0)
            except Exception:
                pass
        model = YOLO(str(args.weights))
        warm = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        model.predict(
            warm,
            verbose=False,
            device=device,
            imgsz=args.imgsz,
        )
    except Exception as exc:
        print(f"ERR load failed: {exc}", flush=True)
        return 1

    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            return 0
        if not line.startswith("SEG "):
            print(f"ERR unknown command: {line[:40]}", flush=True)
            continue
        parts = line.split()
        if len(parts) < 4:
            print("ERR SEG needs: SEG <jpg> <cw> <ch> <scale>", flush=True)
            continue
        path = Path(parts[1])
        try:
            cw, ch = int(parts[2]), int(parts[3])
            scale = float(parts[4]) if len(parts) > 4 else 200.0
        except ValueError as exc:
            print(f"ERR bad args: {exc}", flush=True)
            continue

        try:
            t0 = time.time()
            pil = Image.open(path).convert("RGB")
            # Resize to canvas if needed for mask alignment
            if pil.size != (cw, ch):
                pil_full = pil.resize((cw, ch), Image.Resampling.BILINEAR)
            else:
                pil_full = pil

            results = model.predict(
                pil_full,
                verbose=False,
                device=device,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=list(NAV_CLASS_IDS.keys()),
                retina_masks=True,
                max_det=30,
                iou=0.5,
            )
            r0 = results[0]
            occ = np.zeros((ch, cw), dtype=np.float32)
            obstacles: list[dict[str, Any]] = []

            if r0.masks is not None and r0.boxes is not None and len(r0.boxes):
                masks = r0.masks.data.detach().cpu().numpy()  # (n, mh, mw)
                clss = r0.boxes.cls.detach().cpu().numpy().astype(int)
                confs = r0.boxes.conf.detach().cpu().numpy()
                for i in range(len(clss)):
                    cid = int(clss[i])
                    label = NAV_CLASS_IDS.get(cid, str(r0.names.get(cid, cid)))
                    m = masks[i]
                    if m.shape[0] != ch or m.shape[1] != cw:
                        m = np.array(
                            Image.fromarray((m > 0.5).astype(np.uint8) * 255).resize(
                                (cw, ch), Image.Resampling.NEAREST
                            )
                        )
                        m = (m > 127).astype(np.float32)
                    else:
                        m = (m > 0.5).astype(np.float32)
                    occ = np.maximum(occ, m)
                    ys, xs = np.where(m > 0.5)
                    if len(xs) < 12:
                        continue
                    u = float(xs.mean())
                    v = float(ys.mean())
                    u_n, v_n = u / float(cw), v / float(ch)
                    if abs(u_n - 0.5) < 0.09 and abs(v_n - 0.5) < 0.09:
                        continue
                    x_m = (ch * 0.5 - v) / scale
                    y_m = (cw * 0.5 - u) / scale
                    area = float(len(xs))
                    radius_m = max(0.12, (area ** 0.5) / scale * 0.5)
                    obstacles.append(
                        {
                            "label": label,
                            "azimuth": _azimuth(u_n, v_n),
                            "u_norm": round(u_n, 4),
                            "v_norm": round(v_n, 4),
                            "conf": round(float(confs[i]), 3),
                            "x_m": round(float(x_m), 3),
                            "y_m": round(float(y_m), 3),
                            "radius_m": round(float(radius_m), 3),
                        }
                    )

            arr = np.asarray(pil_full)
            seam = (arr.mean(axis=2) < 12).astype(np.float32)
            occ = np.maximum(occ, seam * 0.5)

            if args.classic:
                classic = _classical_occupancy(arr)
                # Don't double-count strong YOLO regions; classic fills gaps
                classic_only = classic * (1.0 - (occ > 0.5).astype(np.float32))
                occ = np.maximum(occ, classic * 0.85)
                # Add clutter blobs only where YOLO silent
                clutter = _blobs_to_obstacles(
                    classic_only,
                    scale=scale,
                    cw=cw,
                    ch=ch,
                    min_area=max(140, (cw * ch) // 450),
                    max_blobs=4,
                    label="clutter",
                    conf=0.45,
                )
                # Avoid near-duplicate of existing YOLO obstacles
                for c in clutter:
                    dup = False
                    for o in obstacles:
                        if o["x_m"] is None or c["x_m"] is None:
                            continue
                        if (o["x_m"] - c["x_m"]) ** 2 + (o["y_m"] - c["y_m"]) ** 2 < 0.2**2:
                            dup = True
                            break
                    if not dup:
                        obstacles.append(c)

            free_dirs, free_frac = _free_dirs_from_mask(occ > 0.45, free_thresh=0.88)
            obstacles.sort(key=lambda o: o["conf"], reverse=True)
            obstacles = obstacles[:10]

            labels = [o["label"] for o in obstacles]
            if obstacles:
                summary = f"seg:{','.join(labels[:4])} free={free_frac:.0%}"
            else:
                summary = f"seg:clear free={free_frac:.0%}"

            payload = {
                "mode": "nav",
                "source": "seg",
                "summary": summary,
                "obstacles": obstacles,
                "free_dirs": free_dirs,
                "free_frac": round(free_frac, 3),
                "uncertain": [],
            }
            # Compact occupancy downsample for overlay (optional small)
            # Skip sending full mask over line protocol — parent redraws from obstacles.
            ms = (time.time() - t0) * 1000.0
            if device not in ("cpu", "CPU"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            print(f"OK {ms:.0f}", flush=True)
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            print("END", flush=True)
        except Exception as exc:
            print(f"ERR {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
