#!/usr/bin/env python3
"""读写标定相关 JSON 配置（供 Web / 脚本共用）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BOARD_PATH = ROOT / "config" / "chessboard_config.json"
PLACEMENTS_PATH = ROOT / "config" / "extrinsic_placements.json"
SETTINGS_PATH = ROOT / "config" / "web_calib_settings.json"
CAMERA_PATH = ROOT / "config" / "camera_config.json"

DIRECTIONS = ("front", "back", "left", "right")

_DEFAULT_SETTINGS = {
    "detect_max_width": 1920,
    "detect_scan_width": 1920,
    "detect_interval_ms": 500,
    "detect_duty": 0.5,
    "detect_try_scales": [1.0, 0.75, 0.5],
    "detect_use_sb": True,
    "auto_lock": True,
    "sequential": True,
    "inview_margin_px": 8,
    "streak_reset_misses": 2,
    "focus_miss_tolerance": 3,
    "stable_frames": 10,
    "burst_frames": 8,
    "burst_min_ok": 5,
    "extrinsic_balance": 0.8,
    "scale_px_per_m": 100,
    "canvas": [1000, 1000],
    "intrinsics_min_frames": 15,
    "intrinsics_target_frames": 25,
    "intrinsics_cooldown": 12,
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def load_settings() -> dict[str, Any]:
    out = dict(_DEFAULT_SETTINGS)
    if SETTINGS_PATH.is_file():
        raw = _read_json(SETTINGS_PATH)
        for k, v in raw.items():
            if k.startswith("_"):
                continue
            out[k] = v
    return out


def load_all_config() -> dict[str, Any]:
    board = _read_json(BOARD_PATH) if BOARD_PATH.is_file() else {
        "pattern_size": [8, 6], "square_size_m": 0.025
    }
    placements = _read_json(PLACEMENTS_PATH) if PLACEMENTS_PATH.is_file() else {}
    camera = _read_json(CAMERA_PATH) if CAMERA_PATH.is_file() else {}
    settings = load_settings()
    return {
        "chessboard": {
            "pattern_cols": int(board.get("pattern_size", [8, 6])[0]),
            "pattern_rows": int(board.get("pattern_size", [8, 6])[1]),
            "square_size_m": float(board.get("square_size_m", 0.025)),
        },
        "placements": {
            d: {
                "near_m": float((placements.get(d) or {}).get("near_m", 0.35)),
                "lateral_m": float((placements.get(d) or {}).get("lateral_m", 0.0)),
                "orient": str((placements.get(d) or {}).get("orient", "long-lateral")),
            }
            for d in DIRECTIONS
        },
        "settings": settings,
        "camera": {d: int(camera[d]) for d in DIRECTIONS if d in camera},
        "paths": {
            "chessboard": str(BOARD_PATH),
            "placements": str(PLACEMENTS_PATH),
            "settings": str(SETTINGS_PATH),
            "camera": str(CAMERA_PATH),
        },
    }


def save_all_config(payload: dict[str, Any]) -> dict[str, Any]:
    """根据网页表单写回 JSON。忽略 camera 映射（防误改设备号，除非显式带 camera）。"""
    changed: list[str] = []

    board_in = payload.get("chessboard") or {}
    if board_in:
        cols = int(board_in.get("pattern_cols", 8))
        rows = int(board_in.get("pattern_rows", 6))
        square = float(board_in.get("square_size_m", 0.025))
        if cols < 2 or rows < 2:
            raise ValueError("pattern 内角点至少 2x2")
        if square <= 0:
            raise ValueError("square_size_m 必须 > 0")
        data = {
            "_说明": "棋盘配置真源（config/chessboard_config.json）。务必与实物格宽一致。",
            "pattern_size": [cols, rows],
            "square_size_m": square,
        }
        _write_json(BOARD_PATH, data)
        changed.append("chessboard")

    places_in = payload.get("placements") or {}
    if places_in:
        data = {
            "_说明": "外参摆位真源。near_m=板近边到车体中心距离(m)。",
        }
        for d in DIRECTIONS:
            p = places_in.get(d) or {}
            data[d] = {
                "near_m": float(p.get("near_m", 0.35)),
                "lateral_m": float(p.get("lateral_m", 0.0)),
                "orient": str(p.get("orient", "long-lateral")),
            }
        _write_json(PLACEMENTS_PATH, data)
        changed.append("placements")

    settings_in = payload.get("settings") or {}
    if settings_in:
        cur = load_settings()
        for k, v in settings_in.items():
            if k.startswith("_"):
                continue
            cur[k] = v
        # normalize types
        cur["detect_max_width"] = int(cur["detect_max_width"])
        # 不允许低分辨率预扫；保留字段仅兼容旧配置。
        cur["detect_scan_width"] = cur["detect_max_width"]
        cur["detect_interval_ms"] = max(50, int(cur.get("detect_interval_ms", 500)))
        cur["detect_duty"] = min(1.0, max(0.05, float(cur.get("detect_duty", 0.5))))
        cur["detect_use_sb"] = bool(cur.get("detect_use_sb", True))
        cur["auto_lock"] = bool(cur.get("auto_lock", True))
        cur["sequential"] = bool(cur.get("sequential", True))
        cur["inview_margin_px"] = max(0, int(cur.get("inview_margin_px", 8)))
        cur["streak_reset_misses"] = max(1, int(cur.get("streak_reset_misses", 2)))
        cur["focus_miss_tolerance"] = max(1, int(cur.get("focus_miss_tolerance", 3)))
        cur["stable_frames"] = int(cur["stable_frames"])
        cur["burst_frames"] = int(cur["burst_frames"])
        cur["burst_min_ok"] = int(cur["burst_min_ok"])
        cur["extrinsic_balance"] = float(cur["extrinsic_balance"])
        cur["scale_px_per_m"] = float(cur["scale_px_per_m"])
        canvas = cur.get("canvas") or [1000, 1000]
        cur["canvas"] = [int(canvas[0]), int(canvas[1])]
        cur["intrinsics_min_frames"] = int(cur["intrinsics_min_frames"])
        cur["intrinsics_target_frames"] = int(cur["intrinsics_target_frames"])
        cur["intrinsics_cooldown"] = int(cur["intrinsics_cooldown"])
        if isinstance(cur.get("detect_try_scales"), str):
            cur["detect_try_scales"] = [
                float(x) for x in cur["detect_try_scales"].replace(",", " ").split()
            ]
        cur["detect_try_scales"] = [float(x) for x in cur["detect_try_scales"]]
        cur["_说明"] = "Web/CLI 标定运行参数（可被网页改写）"
        _write_json(SETTINGS_PATH, cur)
        changed.append("settings")

    cam_in = payload.get("camera")
    if isinstance(cam_in, dict) and cam_in:
        data = {d: int(cam_in[d]) for d in DIRECTIONS if d in cam_in}
        if len(data) != 4:
            raise ValueError("camera 需包含 front/back/left/right")
        _write_json(CAMERA_PATH, data)
        changed.append("camera")

    return {"ok": True, "changed": changed, "config": load_all_config()}
