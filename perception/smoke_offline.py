#!/usr/bin/env python3
"""Offline smoke: schema parse + pixel↔meter (no camera / VLM)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.localize import (  # noqa: E402
    base_link_to_pixel,
    enrich_event,
    pixel_to_base_link,
    smooth_grasp_heading,
)
from perception.ego_overlay import overlay_ego  # noqa: E402
from perception.occupancy import (  # noqa: E402
    OccupancyGrid,
    _nearest_cardinal_m,
    apply_occ_veto,
    estimate_occupancy,
    render_occupancy_map,
    snap_grasp_to_occupancy,
    stamp_detections_on_grid,
)
from perception.anything_worker import _name_of, _vocab_for  # noqa: E402
from perception.schema import (  # noqa: E402
    NavResult,
    Obstacle,
    PerceptionEvent,
    grasp_prompt,
    parse_nav_payload,
    parse_vlm_response,
)


def approx(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


def test_grasp_prompt_format() -> None:
    p = grasp_prompt("bottle")
    assert "bottle" in p
    assert '{"mode":"grasp"' in p
    assert "{target}" not in p
    m = grasp_prompt("mouse", occ_az="r,fr")
    assert "computer mouse" in m
    assert "右" in m and "{hint}" not in m and "{occ}" not in m


def test_coord_roundtrip() -> None:
    cw = ch = 1000
    scale = 200.0
    # 1 m forward, 0.5 m left → image up and left of center
    u, v = base_link_to_pixel(1.0, 0.5, canvas_w=cw, canvas_h=ch, scale_px_per_meter=scale)
    assert approx(u, 500 - 0.5 * 200)
    assert approx(v, 500 - 1.0 * 200)
    x, y = pixel_to_base_link(u, v, canvas_w=cw, canvas_h=ch, scale_px_per_meter=scale)
    assert approx(x, 1.0) and approx(y, 0.5)


def test_nav_parse_and_localize() -> None:
    text = (
        '{"mode":"nav","summary":"左前方纸箱",'
        '"obstacles":[{"label":"carton","azimuth":"fl","u_norm":0.35,"v_norm":0.28,"conf":0.7}],'
        '"free_dirs":["right","back"],"uncertain":[]}'
    )
    ev = parse_vlm_response(text, mode="nav", infer_ms=12.0)
    assert ev.valid and ev.nav is not None
    assert len(ev.nav.obstacles) == 1
    enrich_event(ev, canvas_size=(1000, 1000), scale_px_per_meter=200.0)
    obs = ev.nav.obstacles[0]
    assert obs.x_m is not None and obs.y_m is not None
    # v_norm=0.28 → v=280 → x = (500-280)/200 = 1.1
    assert approx(obs.x_m, 1.1, eps=1e-3)
    # u_norm=0.35 → u=350 → y = (500-350)/200 = 0.75
    assert approx(obs.y_m, 0.75, eps=1e-3)


def test_nav_filters_prompt_leak() -> None:
    text = (
        '{"mode":"nav","summary":"向前走",'
        '"obstacles":[{"label":"地面小车","azimuth":"f|b|l|r","u_norm":0.5,"v_norm":0.5,"conf":0.9}],'
        '"free_dirs":["front","back","left","right"],"uncertain":[]}'
    )
    ev = parse_vlm_response(text, mode="nav")
    assert ev.valid and ev.nav is not None
    assert ev.nav.obstacles == []
    assert ev.nav.free_dirs == []


def test_grasp_parse() -> None:
    text = (
        'Here is JSON:\n```json\n'
        '{"mode":"grasp","notes":"瓶子在右前",'
        '"targets":[{"label":"bottle","azimuth":"fr","u_norm":0.72,"v_norm":0.35,"conf":0.8,"graspable":true}],'
        '"best_target_id":0}\n```'
    )
    ev = parse_vlm_response(text, mode="grasp", infer_ms=20.0)
    assert ev.valid and ev.grasp is not None
    enrich_event(ev, canvas_size=(800, 800), scale_px_per_meter=200.0)
    t = ev.grasp.targets[0]
    assert t.x_m is not None and math.isfinite(t.x_m)
    assert t.yaw_deg is not None and t.yaw_deg < 0  # right of vehicle
    assert t.range_m is not None and t.range_m > 0
    assert ev.grasp.turn_hint.startswith("R")
    assert "瓶子" in ev.summary


def test_yolo_world_box_xy() -> None:
    ev = parse_vlm_response(
        '{"mode":"grasp","source":"yolo-world","notes":"f:computer mouse",'
        '"targets":[{"label":"computer mouse","azimuth":"f",'
        '"u_norm":0.50,"v_norm":0.35,"conf":0.6}]}',
        mode="grasp",
    )
    enrich_event(ev, canvas_size=(600, 600), scale_px_per_meter=120.0)
    t = ev.grasp.targets[0]
    assert ev.grasp.source == "yolo-world"
    assert t.x_m is not None and abs(t.x_m - 0.75) < 0.05
    assert t.y_m is not None and abs(t.y_m) < 0.05
    assert "(" in ev.grasp.turn_hint


def test_grasp_azimuth_fallback() -> None:
    text = '{"mode":"grasp","notes":"左侧","targets":[{"label":"bottle","azimuth":"l","conf":0.8}],"best_target_id":0}'
    ev = parse_vlm_response(text, mode="grasp")
    enrich_event(ev, canvas_size=(800, 800), scale_px_per_meter=200.0)
    t = ev.grasp.targets[0]
    assert t.yaw_deg == 90.0
    assert ev.grasp.turn_hint.startswith("L")
    assert t.x_m is not None and t.y_m is not None
    assert abs(t.x_m) < 0.05 and abs(t.y_m - 0.7) < 0.05
    assert "(" in ev.grasp.turn_hint


def test_grasp_occ_snap_xy() -> None:
    ev = parse_vlm_response(
        '{"mode":"grasp","notes":"前",'
        '"targets":[{"label":"mouse","azimuth":"f","conf":0.7}]}',
        mode="grasp",
    )
    enrich_event(ev, canvas_size=(1000, 1000), scale_px_per_meter=200.0)
    cells = __import__("numpy").zeros((10, 10), dtype="uint8")
    cells[1, 5] = 1  # front of origin (1.0,1.0), res=0.2 → x=0.8 y=0.0
    grid = OccupancyGrid(
        resolution_m=0.2,
        origin_x_m=1.0,
        origin_y_m=1.0,
        cells=cells,
        free_frac=0.9,
        free_dirs=["front"],
        vehicle_uv=(0.38, 0.38, 0.62, 0.62),
    )
    snap_grasp_to_occupancy(ev, grid)
    t = ev.grasp.targets[0]
    assert abs(t.x_m - 0.8) < 0.05
    assert abs(t.y_m) < 0.05


def test_grasp_smooth_holds_jump() -> None:
    text = (
        '{"mode":"grasp","notes":"前",'
        '"targets":[{"label":"bottle","azimuth":"f","conf":0.8}],'
        '"best_target_id":0}'
    )
    ev = parse_vlm_response(text, mode="grasp")
    enrich_event(ev, canvas_size=(800, 800), scale_px_per_meter=200.0)
    ev, yaw, _rng = smooth_grasp_heading(ev, yaw_ema=None, range_ema=None)
    assert yaw == 0.0
    text2 = (
        '{"mode":"grasp","notes":"后",'
        '"targets":[{"label":"bottle","azimuth":"b","conf":0.8}],'
        '"best_target_id":0}'
    )
    ev2 = parse_vlm_response(text2, mode="grasp")
    enrich_event(ev2, canvas_size=(800, 800), scale_px_per_meter=200.0)
    ev2, yaw2, _ = smooth_grasp_heading(ev2, yaw_ema=yaw, range_ema=None)
    assert yaw2 == 0.0  # 180° jump rejected


def test_grasp_notes_infer_front() -> None:
    ev = parse_vlm_response(
        '{"mode":"grasp","notes":"车体前方有黑色物体，可能是电脑鼠标。","targets":[]}',
        mode="grasp",
    )
    assert ev.grasp is not None and ev.grasp.targets
    assert ev.grasp.targets[0].azimuth == "f"
    assert ev.grasp.targets[0].label == "mouse"


def test_grasp_truncated_computer_mouse() -> None:
    ev = parse_vlm_response(
        '{"mode":"grasp","notes":"车体前方有黑色物体，可能是电脑鼠标。",'
        '"targets":[{"label":"comp',
        mode="grasp",
    )
    assert ev.valid and ev.grasp is not None and ev.grasp.targets
    assert ev.grasp.targets[0].azimuth == "f"
    assert ev.grasp.targets[0].label == "mouse"


def test_grasp_notes_miss_keeps_target() -> None:
    ev = parse_vlm_response(
        '{"mode":"grasp","notes":"未看到清楚但右前有鼠标",'
        '"targets":[{"label":"mouse","azimuth":"fr","conf":0.6}]}',
        mode="grasp",
    )
    assert ev.grasp is not None and ev.grasp.targets
    assert ev.grasp.targets[0].azimuth == "fr"


def test_grasp_example_leak_and_miss() -> None:
    leak = parse_vlm_response(
        '{"mode":"grasp","notes":"右前有瓶子",'
        '"targets":[{"label":"bottle","azimuth":"fr","conf":0.8}],'
        '"best_target_id":0}',
        mode="grasp",
    )
    assert leak.grasp is not None
    assert leak.grasp.notes != "右前有瓶子"
    miss = parse_vlm_response(
        '{"mode":"grasp","notes":"未找到","targets":[],"best_target_id":null}',
        mode="grasp",
    )
    assert miss.grasp is not None and miss.grasp.targets == []
    low = parse_vlm_response(
        '{"mode":"grasp","notes":"也许有",'
        '"targets":[{"label":"bottle","azimuth":"fr","conf":0.3}],'
        '"best_target_id":0}',
        mode="grasp",
    )
    assert low.grasp is not None and low.grasp.targets == []


def test_occ_veto_empty_sector() -> None:
    ev = parse_vlm_response(
        '{"mode":"grasp","notes":"左侧有瓶",'
        '"targets":[{"label":"bottle","azimuth":"l","conf":0.9}],'
        '"best_target_id":0}',
        mode="grasp",
    )
    enrich_event(ev, canvas_size=(800, 800), scale_px_per_meter=200.0)
    grid = OccupancyGrid(
        resolution_m=0.2,
        origin_x_m=1.0,
        origin_y_m=1.0,
        cells=__import__("numpy").zeros((10, 10), dtype="uint8"),
        free_frac=1.0,
        free_dirs=["front", "back", "left", "right"],
    )
    out = apply_occ_veto(ev, grid)
    assert out.grasp is not None and out.grasp.targets == []
    assert out.summary == "未找到"


def test_yolo_world_names_are_vocab_not_ids() -> None:
    vocab = _vocab_for("mouse")
    # set_classes stores a list; Results.names may be that list or empty.
    assert _name_of(0, vocab, names=[]) == vocab[0]
    i_chair = vocab.index("chair")
    i_box = vocab.index("box")
    assert _name_of(i_chair, vocab, names={}) == "chair"
    assert _name_of(i_box, vocab) == "box"
    assert not _name_of(i_chair, vocab).isdigit()


def test_yolo_world_open_vocab_keeps_boxes() -> None:
    ev = parse_vlm_response(
        '{"mode":"grasp","source":"yolo-world","notes":"f:computer mouse,l:bottle",'
        '"targets":['
        '{"label":"computer mouse","azimuth":"f","u_norm":0.50,"v_norm":0.32,"conf":0.22,'
        '"x_m":0.90,"y_m":0.0},'
        '{"label":"mouse","azimuth":"fr","u_norm":0.62,"v_norm":0.40,"conf":0.18}'
        "],"
        '"obstacles":['
        '{"label":"bottle","azimuth":"l","u_norm":0.22,"v_norm":0.50,"conf":0.41,'
        '"x_m":0.0,"y_m":1.4},'
        '{"label":"cup","azimuth":"r","u_norm":0.78,"v_norm":0.48,"conf":0.33}'
        "]}",
        mode="grasp",
    )
    assert ev.grasp is not None
    assert ev.grasp.source == "yolo-world"
    assert len(ev.grasp.targets) >= 2
    assert ev.grasp.targets[0].conf == 0.22
    assert "未找到" not in (ev.grasp.notes or "")
    nav = parse_nav_payload(
        {
            "source": "yolo-world",
            "obstacles": [
                {"label": "bottle", "azimuth": "l", "u_norm": 0.22, "v_norm": 0.50, "conf": 0.41, "x_m": 0.0, "y_m": 1.4},
                {"label": "cup", "azimuth": "r", "u_norm": 0.78, "v_norm": 0.48, "conf": 0.33},
            ],
        }
    )
    assert len(nav.obstacles) == 2
    assert nav.obstacles[0].x_m is not None


def test_occupancy_strict_on_flat_floor() -> None:
    img = np.full((400, 400, 3), 150, dtype=np.uint8)
    img[170:230, 170:230] = 8
    grid, _ = estimate_occupancy(img, scale_px_per_meter=80.0, resolution_m=0.20)
    assert float((grid.cells == 1).mean()) < 0.02


def test_occupancy_hits_solid_blob() -> None:
    img = np.full((400, 400, 3), 150, dtype=np.uint8)
    img[170:230, 170:230] = 8
    img[30:110, 160:240] = 25
    grid, _ = estimate_occupancy(img, scale_px_per_meter=80.0, resolution_m=0.20)
    assert int((grid.cells == 1).sum()) >= 3


def test_occupancy_hits_small_shoe() -> None:
    img = np.full((400, 400, 3), 150, dtype=np.uint8)
    img[170:230, 170:230] = 8
    img[188:210, 308:328] = (55, 45, 30)
    grid, _ = estimate_occupancy(img, scale_px_per_meter=80.0, resolution_m=0.20)
    assert int((grid.cells == 1).sum()) >= 1


def test_ego_overlay_centers() -> None:
    bev = np.full((200, 200, 3), 160, dtype=np.uint8)
    bev[78:122, 78:122] = 8
    sprite = np.zeros((40, 40, 4), dtype=np.uint8)
    sprite[:, :] = (0, 140, 255, 255)
    out = overlay_ego(bev, sprite, size_m=0.0, scale_px_per_meter=100.0)
    assert int(out[100, 100, 2]) > 200
    assert int(out[90, 90, 2]) > 200


def test_ego_overlay_covers_tilted_hole() -> None:
    import cv2

    bev = np.full((240, 240, 3), 160, dtype=np.uint8)
    box = cv2.boxPoints(((120.0, 120.0), (56.0, 56.0), -11.0)).astype(np.int32)
    cv2.fillConvexPoly(bev, box, (6, 6, 6))
    sprite = np.zeros((50, 50, 4), dtype=np.uint8)
    sprite[:, :] = (0, 140, 255, 255)
    out = overlay_ego(bev, sprite, size_m=0.0, scale_px_per_meter=100.0)
    was_dark = bev[:, :, 0] < 20
    covered = out[:, :, 2] > 100
    assert float(covered[was_dark].mean()) > 0.90


def test_ego_overlay_locked_box_does_not_follow_hole() -> None:
    bev = np.full((200, 200, 3), 160, dtype=np.uint8)
    bev[40:160, 40:160] = 8
    sprite = np.zeros((20, 20, 4), dtype=np.uint8)
    sprite[:, :] = (0, 140, 255, 255)
    box = (70, 70, 130, 130)
    out = overlay_ego(bev, sprite, box=box)
    assert int(out[100, 100, 2]) > 200
    assert int(out[50, 50, 0]) < 20


def test_ego_overlay_no_orange_corners() -> None:
    bev = np.full((200, 200, 3), 160, dtype=np.uint8)
    bev[78:122, 78:122] = 8
    sprite = np.zeros((40, 40, 4), dtype=np.uint8)
    sprite[8:32, 16:24] = (0, 140, 255, 255)
    sprite[16:24, 8:32] = (0, 140, 255, 255)
    out = overlay_ego(bev, sprite, size_m=0.0, scale_px_per_meter=100.0)
    # Plus-shaped PNG: box corners stay the BEV, not filled orange.
    assert int(out[78, 78, 2]) < 40


def test_nearest_cardinal_inside_old_deadzone() -> None:
    cells = np.zeros((25, 25), dtype=np.uint8)
    cells[10, 12] = 1
    grid = OccupancyGrid(
        resolution_m=0.2,
        origin_x_m=2.5,
        origin_y_m=2.5,
        cells=cells,
        free_frac=0.9,
        free_dirs=["front"],
        vehicle_uv=(0.46, 0.46, 0.54, 0.54),
    )
    near = _nearest_cardinal_m(grid)
    assert near["f"] is not None
    assert near["f"] < 0.55


def test_occ_map_renders_front_hit() -> None:
    cells = np.zeros((10, 10), dtype=np.uint8)
    cells[1, 5] = 1
    grid = OccupancyGrid(
        resolution_m=0.2,
        origin_x_m=1.0,
        origin_y_m=1.0,
        cells=cells,
        free_frac=0.8,
        free_dirs=["left"],
        vehicle_uv=(0.4, 0.4, 0.6, 0.6),
    )
    img = render_occupancy_map(grid, size=200)
    assert img.shape == (200, 200, 3)
    assert int(img[100, 100, 1]) > 80
    assert int(img[:80, :, 2].max()) > 150


def test_stamp_person_paints_grid() -> None:
    cells = np.zeros((10, 10), dtype=np.uint8)
    grid = OccupancyGrid(
        resolution_m=0.2,
        origin_x_m=1.0,
        origin_y_m=1.0,
        cells=cells,
        free_frac=1.0,
        free_dirs=["front"],
        vehicle_uv=(0.4, 0.4, 0.6, 0.6),
    )
    ev = PerceptionEvent(
        nav=NavResult(
            obstacles=[
                Obstacle(label="person", u_norm=0.82, v_norm=0.48, x_m=0.1, y_m=-0.9)
            ]
        )
    )
    stamp_detections_on_grid(grid, ev)
    assert int((grid.cells == 1).sum()) >= 1


def test_parse_fallback() -> None:
    ev = parse_vlm_response("左侧好像有东西但不清楚", mode="nav")
    assert "左侧" in ev.summary
    assert ev.nav is not None
    empty = parse_vlm_response("", mode="nav")
    assert not empty.valid
    assert empty.error == "json_parse_failed"


def main() -> int:
    test_grasp_prompt_format()
    test_coord_roundtrip()
    test_nav_parse_and_localize()
    test_nav_filters_prompt_leak()
    test_grasp_parse()
    test_yolo_world_box_xy()
    test_grasp_azimuth_fallback()
    test_grasp_occ_snap_xy()
    test_grasp_smooth_holds_jump()
    test_grasp_notes_infer_front()
    test_grasp_truncated_computer_mouse()
    test_grasp_notes_miss_keeps_target()
    test_grasp_example_leak_and_miss()
    test_occ_veto_empty_sector()
    test_yolo_world_names_are_vocab_not_ids()
    test_yolo_world_open_vocab_keeps_boxes()
    test_occupancy_strict_on_flat_floor()
    test_occupancy_hits_solid_blob()
    test_occupancy_hits_small_shoe()
    test_ego_overlay_centers()
    test_ego_overlay_covers_tilted_hole()
    test_ego_overlay_locked_box_does_not_follow_hole()
    test_ego_overlay_no_orange_corners()
    test_nearest_cardinal_inside_old_deadzone()
    test_occ_map_renders_front_hit()
    test_stamp_person_paints_grid()
    test_parse_fallback()
    print("perception smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
