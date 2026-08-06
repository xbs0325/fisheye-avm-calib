"""Shared chessboard config loader for intrinsic/extrinsic calibration."""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_BOARD_CONFIG = os.path.join(PROJECT_DIR, "config", "chessboard_config.json")

# Fallback if config file is missing
_FALLBACK_PATTERN = [8, 6]
_FALLBACK_SQUARE_M = 0.08


def load_chessboard_config(path=None):
    """
    Load checkerboard specs from JSON.

    Returns:
        dict with keys:
          path, pattern_size (cols, rows), pattern_size_str ("WxH"),
          square_size_m (float)
    """
    path = path or DEFAULT_BOARD_CONFIG
    pattern = list(_FALLBACK_PATTERN)
    square = float(_FALLBACK_SQUARE_M)

    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("pattern_size", pattern)
        if isinstance(raw, str):
            cols, rows = map(int, raw.lower().replace(" ", "").split("x"))
            pattern = [cols, rows]
        else:
            pattern = [int(raw[0]), int(raw[1])]
        square = float(data.get("square_size_m", square))
    else:
        print(f"[警告] 棋盘配置不存在: {path}，使用默认 "
              f"{pattern[0]}x{pattern[1]} / {square}m")

    if pattern[0] < 2 or pattern[1] < 2:
        raise ValueError(f"pattern_size 无效: {pattern}（内角点至少 2x2）")
    if square <= 0:
        raise ValueError(f"square_size_m 必须 > 0，得到 {square}")

    cols, rows = pattern[0], pattern[1]
    return {
        "path": path,
        "pattern_size": (cols, rows),
        "pattern_size_str": f"{cols}x{rows}",
        "square_size_m": square,
    }


def resolve_board_args(pattern_size_cli, square_cli, board_config_path=None):
    """
    Merge CLI overrides with chessboard_config.json.
    pattern_size_cli: None or "WxH"
    square_cli: None or float meters
    """
    cfg = load_chessboard_config(board_config_path)
    if pattern_size_cli:
        cols, rows = map(int, pattern_size_cli.lower().replace(" ", "").split("x"))
        pattern_size = (cols, rows)
        pattern_size_str = f"{cols}x{rows}"
    else:
        pattern_size = cfg["pattern_size"]
        pattern_size_str = cfg["pattern_size_str"]
    square = float(square_cli) if square_cli is not None else cfg["square_size_m"]
    return {
        "path": cfg["path"],
        "pattern_size": pattern_size,
        "pattern_size_str": pattern_size_str,
        "square_size_m": square,
        "from_config": pattern_size_cli is None and square_cli is None,
    }
