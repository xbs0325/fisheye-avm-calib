#!/usr/bin/env python3
"""AVM 标定契约：配置指纹、一致性比对、原子写 JSON。

向导与标定脚本共用，避免 pipeline / placements / extrinsics 各说各话。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


DIRECTIONS = ("front", "back", "left", "right")


def _round_float(x: float, nd: int = 6) -> float:
    return float(round(float(x), nd))


def placement_slice(placements: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for d in DIRECTIONS:
        p = placements.get(d) or {}
        out[d] = {
            "near_m": _round_float(p.get("near_m", 0.5)),
            "lateral_m": _round_float(p.get("lateral_m", 0.0)),
            "orient": str(p.get("orient", "long-lateral")),
        }
    return out


def build_fingerprint_payload(
    *,
    pattern_size: list[int] | tuple[int, int],
    square_size_m: float,
    placements: dict[str, Any],
    extrinsic_balance: float,
    scale_px_per_m: float,
    canvas: list[int] | tuple[int, int],
) -> dict[str, Any]:
    return {
        "pattern_size": [int(pattern_size[0]), int(pattern_size[1])],
        "square_size_m": _round_float(square_size_m, 6),
        "placements": placement_slice(placements),
        "extrinsic_balance": _round_float(extrinsic_balance, 4),
        "scale_px_per_m": _round_float(scale_px_per_m, 4),
        "canvas": [int(canvas[0]), int(canvas[1])],
    }


def fingerprint_from_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def make_fingerprint(**kwargs: Any) -> tuple[str, dict[str, Any]]:
    payload = build_fingerprint_payload(**kwargs)
    return fingerprint_from_payload(payload), payload


def fingerprint_from_pipeline(cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    board = cfg["chessboard"]
    ex = cfg.get("extrinsic") or {}
    canvas = ex.get("canvas") or [1000, 1000]
    return make_fingerprint(
        pattern_size=board["pattern_size"],
        square_size_m=float(board["square_size_m"]),
        placements=cfg.get("placements") or {},
        extrinsic_balance=float(ex.get("balance", 0.5)),
        scale_px_per_m=float(ex.get("scale_px_per_m", 100)),
        canvas=canvas,
    )


def fingerprint_from_extrinsics(data: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """从已写盘的 extrinsics.json 重建指纹；缺字段则返回 (None, None)。"""
    try:
        bal = data.get("extrinsic_balance", data.get("balance"))
        if bal is None:
            return None, None
        payload = build_fingerprint_payload(
            pattern_size=data["pattern_size"],
            square_size_m=float(data["square_size_m"]),
            placements=data.get("placements") or {},
            extrinsic_balance=float(bal),
            scale_px_per_m=float(data["scale_px_per_meter"]),
            canvas=data["canvas_size"],
        )
        stored = data.get("config_fingerprint")
        computed = fingerprint_from_payload(payload)
        return stored or computed, payload
    except (KeyError, TypeError, ValueError):
        return None, None


def compare_fingerprint(
    expected_fp: str,
    expected_payload: dict[str, Any],
    extrinsics: dict[str, Any],
    *,
    near_tol_m: float = 0.02,
    square_tol_m: float = 1e-4,
    balance_tol: float = 0.02,
) -> dict[str, Any]:
    """比对期望配置与磁盘外参。返回 mismatches 列表与 status pass|warn|fail。"""
    mismatches: list[str] = []
    stored_fp = extrinsics.get("config_fingerprint")
    got_fp, got_payload = fingerprint_from_extrinsics(extrinsics)

    if stored_fp and stored_fp != expected_fp:
        mismatches.append(
            f"config_fingerprint 不一致: 结果={stored_fp} 期望={expected_fp}（请用当前向导配置重做外参）"
        )
    elif got_fp and got_fp != expected_fp and not stored_fp:
        mismatches.append(
            f"外参几何与当前配置不一致（无指纹字段的旧文件）: 重建={got_fp} 期望={expected_fp}"
        )

    # 逐项可读 diff（即便指纹已报错，也列出具体差）
    try:
        sq_ex = float(extrinsics["square_size_m"])
        sq_exp = float(expected_payload["square_size_m"])
        if abs(sq_ex - sq_exp) > square_tol_m:
            mismatches.append(f"square_size_m: 结果={sq_ex} 配置={sq_exp}")
    except (KeyError, TypeError, ValueError):
        mismatches.append("外参缺少 square_size_m")

    bal_ex = extrinsics.get("extrinsic_balance", extrinsics.get("balance"))
    try:
        if bal_ex is None or abs(float(bal_ex) - float(expected_payload["extrinsic_balance"])) > balance_tol:
            mismatches.append(
                f"extrinsic_balance: 结果={bal_ex} 配置={expected_payload['extrinsic_balance']}"
            )
    except (TypeError, ValueError):
        mismatches.append("外参 balance 无效")

    for d in DIRECTIONS:
        p_ex = (extrinsics.get("placements") or {}).get(d) or {}
        p_exp = expected_payload["placements"].get(d) or {}
        try:
            if abs(float(p_ex.get("near_m", -1)) - float(p_exp["near_m"])) > near_tol_m:
                mismatches.append(
                    f"{d}.near_m: 结果={p_ex.get('near_m')} 配置={p_exp['near_m']}"
                )
        except (TypeError, ValueError, KeyError):
            mismatches.append(f"{d}.near_m 缺失或无效")

    status = "fail" if mismatches else "pass"
    return {
        "status": status,
        "expected_fingerprint": expected_fp,
        "result_fingerprint": stored_fp or got_fp,
        "mismatches": mismatches,
    }


def atomic_write_json(path: str | Path, data: dict[str, Any]) -> str:
    """先写临时文件再 replace，避免杀进程留下半截 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(path)
