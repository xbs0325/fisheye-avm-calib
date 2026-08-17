#!/usr/bin/env python3
"""Live BEV perception: structured VLM analysis + base_link localization.

Assumes calib_results are complete. Does not touch Web wizard hot path.
No chassis / arm required.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avm.cuda_cv import resize_bgr  # noqa: E402
from demo_bev_vlm.stitch import (  # noqa: E402
    DEFAULT_CALIB_DIR,
    DEFAULT_CONFIG,
    DEFAULT_EXTRINSICS,
    DIRECTIONS,
    build_pipelines,
    build_weight_maps,
    compute_gains,
    grab_frames,
    open_caps,
    process_frame,
)
from perception.anything_client import AnythingWorker  # noqa: E402
from perception.bus import PerceptionBus  # noqa: E402
from perception.localize import refresh_grasp_hint  # noqa: E402
from perception.occupancy import (  # noqa: E402
    OccupancyTracker,
    overlay_occupancy,
    render_occupancy_map,
    snap_grasp_to_occupancy,
    stamp_detections_on_grid,
)
from perception.schema import PerceptionEvent  # noqa: E402
from perception.seg_client import SegWorker  # noqa: E402
from perception.vlm_client import AnalyzeWorker  # noqa: E402
from perception.ego_overlay import (  # noqa: E402
    center_blind_box,
    load_ego_sprite,
    overlay_ego,
)
from perception.viz import draw_hud, finish_film_frame  # noqa: E402


def _fold_gain_weights(
    base_weights: dict[str, np.ndarray],
    gains: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Bake per-camera gain into blend weights so CUDA stitch stays on GPU.

    avm.cuda_cv.process_frames_to_bev drops GPU blend when gains != 1 and
    falls back to 4-tile CPU download (~140ms). Folding keeps gain and ~15+ FPS.
    """
    out: dict[str, np.ndarray] = {}
    for d, w in base_weights.items():
        g = np.asarray(gains.get(d, 1.0), dtype=np.float32).reshape(-1)
        if g.size == 1:
            g = np.repeat(g, 3)
        if np.all(np.abs(g[:3] - 1.0) < 0.003):
            out[d] = w
            continue
        out[d] = np.stack(
            [w * float(g[0]), w * float(g[1]), w * float(g[2])], axis=-1
        ).astype(np.float32)
    return out


def _summary_bit(ev: PerceptionEvent | None) -> str:
    if ev is None:
        return ""
    if ev.grasp is not None and (ev.grasp.notes or "").strip():
        return ev.grasp.notes.strip()
    return (ev.summary or "").strip()


def _compose_event(
    *,
    mode: str,
    geom: PerceptionEvent | None,
    ov_ev: PerceptionEvent | None,
    vlm_ev: PerceptionEvent | None,
    vlm_boxes: bool,
) -> PerceptionEvent | None:
    """Occupancy = space, YOLO = boxes/xy, VLM = caption (not fake geometry)."""
    box_ev = ov_ev if ov_ev is not None else (vlm_ev if vlm_boxes else None)
    base = geom or box_ev or vlm_ev
    if base is None:
        return None
    nav = deepcopy(geom.nav) if geom is not None and geom.nav is not None else None
    if box_ev is not None and box_ev.nav is not None:
        extra = [o for o in box_ev.nav.obstacles if o.label != "occ"]
        if nav is None:
            nav = deepcopy(box_ev.nav)
        else:
            nav.obstacles = list(nav.obstacles) + extra
    grasp = None
    if mode == "grasp" and box_ev is not None and box_ev.grasp is not None:
        grasp = deepcopy(box_ev.grasp)
    bits: list[str] = []
    for ev in (geom, ov_ev, vlm_ev):
        s = _summary_bit(ev)
        if s and s not in bits:
            bits.append(s)
    valid = bool(
        (geom is not None and geom.valid)
        or (box_ev is not None and box_ev.valid)
        or (vlm_ev is not None and vlm_ev.valid)
    )
    return PerceptionEvent(
        schema_version=base.schema_version,
        frame_id=base.frame_id,
        stamp_s=base.stamp_s,
        mode=mode,
        valid=valid,
        infer_ms=float(geom.infer_ms if geom is not None else base.infer_ms),
        summary=" | ".join(bits),
        raw_text=(vlm_ev.raw_text if vlm_ev is not None else base.raw_text),
        nav=nav,
        grasp=grasp,
    )


def _push_weights(
    pipelines: dict, weights: dict[str, np.ndarray]
) -> None:
    for d, pipe in pipelines.items():
        if d in weights:
            pipe.set_weight(weights[d])


def _open_preview_window(name: str) -> None:
    """GTK3: WINDOW_NORMAL + drag blocks waitKey/CUDA on this thread → freeze.

    AUTOSIZE keeps the pixmap 1:1 with the frame. startWindowThread lets the
    window manager move the window without stalling the stitch loop.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    try:
        cv2.startWindowThread()
    except Exception:
        pass
    flags = int(cv2.WINDOW_AUTOSIZE)
    gui = getattr(cv2, "WINDOW_GUI_NORMAL", 0)
    if gui:
        flags |= int(gui)
    cv2.namedWindow(name, flags)


class JsonlLogger:
    def __init__(self, path: Path, *, flush_every_s: float = 1.0) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._flush_every_s = float(flush_every_s)
        self._last_flush = 0.0

    def write(self, event: PerceptionEvent) -> None:
        self._fh.write(event.to_json_line() + "\n")
        now = time.time()
        if now - self._last_flush >= self._flush_every_s:
            self._fh.flush()
            self._last_flush = now

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description="BEV 视觉分析 + 相对定位（nav/grasp，无底盘/臂）"
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    ap.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    ap.add_argument(
        "--range",
        type=float,
        default=2.5,
        help="BEV half-range meters（默认 ±2.5m）",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=120.0,
        help="px per meter（默认 120：画布约等于显示，拼接更快；要更细用 200）",
    )
    ap.add_argument("--blend-power", type=float, default=4.0)
    ap.add_argument("--no-gain", action="store_true")
    ap.add_argument("--display-size", type=int, default=640)
    ap.add_argument(
        "--mode",
        choices=["nav", "grasp"],
        default="nav",
        help="nav=避障语义；grasp=夹取目标粗定位",
    )
    ap.add_argument(
        "--target",
        default="bottle",
        help="grasp 目标（默认 bottle 矿泉水瓶）",
    )
    ap.add_argument(
        "--vlm",
        default="qwen3vl-2b",
        choices=["qwen3vl-2b", "qwen3vl-4b", "qwen3vl-8b", "off"],
        help="语义慢路径：默认 qwen3vl-2b；grasp 默认会关掉，改用 YOLO-World",
    )
    ap.add_argument(
        "--ov",
        dest="ov",
        action="store_true",
        default=None,
        help="YOLO-World 开放词汇检测（默认开：框 + 米制坐标）",
    )
    ap.add_argument(
        "--no-ov",
        dest="ov",
        action="store_false",
        help="关闭 YOLO-World",
    )
    ap.add_argument(
        "--ov-interval",
        type=float,
        default=0.8,
        help="YOLO-World 最小间隔秒（默认 0.8）",
    )
    ap.add_argument(
        "--ov-device",
        default="0",
        help="YOLO-World 设备：0=GPU FP16（默认）；cpu=慢",
    )
    ap.add_argument(
        "--ov-imgsz",
        type=int,
        default=384,
        help="YOLO-World 输入边长（默认 384，更快）",
    )
    ap.add_argument(
        "--ov-conf",
        type=float,
        default=0.10,
        help="YOLO-World 置信度（BEV 俯视偏低，默认 0.10）",
    )
    ap.add_argument(
        "--occ",
        dest="occ",
        action="store_true",
        default=True,
        help="BEV 占用栅格快路径（默认开；不改标定）",
    )
    ap.add_argument(
        "--no-occ",
        dest="occ",
        action="store_false",
        help="关闭占用栅格",
    )
    ap.add_argument(
        "--occ-res",
        type=float,
        default=0.20,
        help="占用栅格分辨率（米/格，默认 0.2）",
    )
    ap.add_argument(
        "--occ-thresh",
        type=float,
        default=0.42,
        help="占用均值阈值，越低越容易标上脚/鞋等小物体（默认 0.42）",
    )
    ap.add_argument(
        "--occ-map",
        dest="occ_map",
        action="store_true",
        default=True,
        help="BEV 右侧实时 2D 占用图（默认开，m 切换）",
    )
    ap.add_argument(
        "--no-occ-map",
        dest="occ_map",
        action="store_false",
        help="关闭右侧 2D 占用图",
    )
    ap.add_argument(
        "--ego-overlay",
        type=Path,
        default=ROOT / "assets" / "ego_overlay.png",
        help="车体透明 PNG，叠在 BEV 中心（scripts/make_ego_overlay.py 生成）",
    )
    ap.add_argument(
        "--no-ego-overlay",
        action="store_true",
        help="不叠车体图，只用中心箭头",
    )
    ap.add_argument(
        "--ego-size-m",
        type=float,
        default=0.0,
        help="车体边长（米）；默认 0=自动铺满中心黑区",
    )
    ap.add_argument(
        "--seg",
        dest="seg",
        action="store_true",
        default=False,
        help="实验性 YOLO-seg（BEV 上易抖，默认关闭）",
    )
    ap.add_argument(
        "--no-seg",
        dest="seg",
        action="store_false",
        help="关闭分割（默认）",
    )
    ap.add_argument(
        "--seg-weights",
        type=Path,
        default=ROOT / "models" / "perception" / "yolov8n-seg.pt",
    )
    ap.add_argument("--seg-imgsz", type=int, default=512)
    ap.add_argument("--seg-conf", type=float, default=0.35)
    ap.add_argument(
        "--seg-interval",
        type=float,
        default=0.45,
        help="分割最小间隔秒（过大易漏检；过小会和拼接抢 GPU 卡住）",
    )
    ap.add_argument(
        "--seg-device",
        default="0",
        help="YOLO 设备：0=GPU（默认）；cpu=不占 GPU，拼接更流畅但分割更慢",
    )
    ap.add_argument(
        "--analyze-interval",
        type=float,
        default=3.0,
        help="VLM 完成后最短空闲秒（默认 3，避免和拼接抢 GPU 卡死；"
        "负数=仅按 a 手动；0 也会按 3 秒托底）",
    )
    ap.add_argument(
        "--max-side",
        type=int,
        default=384,
        help="送入 VLM 的最长边像素（grasp 默认会升到 512；小物体要看清）",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="VLM 生成上限（caption 两三句；analyze JSON 过小会截断）",
    )
    ap.add_argument("--models", type=Path, default=None)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "perception",
    )
    args = ap.parse_args()
    if args.mode == "grasp" and "--max-side" not in sys.argv:
        args.max_side = max(int(args.max_side), 512)
    if args.mode == "grasp" and "--max-new-tokens" not in sys.argv:
        args.max_new_tokens = min(int(args.max_new_tokens), 64)
    if args.ov is None:
        args.ov = True
    if args.mode == "grasp" and "--vlm" not in sys.argv:
        args.vlm = "off"
    ov_dev = str(args.ov_device).strip().lower()
    if ov_dev in ("0", "cuda", "cuda:0", "gpu"):
        ov_dev = "0"
    else:
        ov_dev = "cpu"

    print("=" * 60)
    print("  Perception: BEV analysis + relative localization")
    print("=" * 60)

    pipelines, canvas, center, balance = build_pipelines(
        extrinsics_path=args.extrinsics,
        calib_dir=args.calib_dir,
        range_m=args.range,
        scale=args.scale,
        blend_power=args.blend_power,
    )
    print(f"  canvas={canvas[0]}x{canvas[1]}  balance={balance:.2f}")
    print(f"  mode={args.mode}  range=±{args.range}m  scale={args.scale}px/m")
    base_weights = build_weight_maps(canvas, center, args.blend_power)

    caps = open_caps(args.config)
    print(f"  cameras: {', '.join(sorted(caps.keys()))}")

    bus = PerceptionBus()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir / "events.jsonl")
    bus.subscribe(logger.write)

    last_vlm_done_t = 0.0
    latest_occ: PerceptionEvent | None = None
    latest_seg: PerceptionEvent | None = None
    latest_vlm: PerceptionEvent | None = None
    latest_ov: PerceptionEvent | None = None

    def _on_seg(event: PerceptionEvent) -> None:
        nonlocal latest_seg
        latest_seg = event
        bus.publish(event)

    def _on_vlm(event: PerceptionEvent) -> None:
        nonlocal last_vlm_done_t, latest_vlm
        last_vlm_done_t = time.time()
        latest_vlm = event
        bus.publish(event)

    def _on_ov(event: PerceptionEvent) -> None:
        nonlocal latest_ov
        latest_ov = event
        bus.publish(event)

    occ_tracker: OccupancyTracker | None = None
    use_occ = bool(args.occ)
    if use_occ:
        occ_tracker = OccupancyTracker(
            scale_px_per_meter=args.scale,
            resolution_m=args.occ_res,
            occ_thresh=args.occ_thresh,
        )
        print(
            f"  occupancy grid on  res={args.occ_res:.2f}m/cell  "
            f"thresh={args.occ_thresh:.2f}  (BEV classical)"
        )
    else:
        print("  occupancy disabled")
    show_occ_map = bool(use_occ and args.occ_map)
    if show_occ_map:
        print("  2D occ map: ON (right panel, press m to toggle)")

    ego_sprite = None if args.no_ego_overlay else load_ego_sprite(args.ego_overlay)
    if ego_sprite is not None:
        size_txt = (
            f"size={args.ego_size_m:.2f}m"
            if args.ego_size_m > 0
            else "size=auto (cover black)"
        )
        print(f"  ego overlay {args.ego_overlay.name}  {size_txt}", flush=True)
    elif not args.no_ego_overlay:
        print("  ego overlay off (no PNG; run scripts/make_ego_overlay.py)")

    seg_worker: SegWorker | None = None
    use_seg = bool(args.seg) and args.mode == "nav"
    if use_seg:
        print("  [warn] YOLO-seg 在环视 BEV 上不稳定，仅作实验；正式导航建议 --no-seg")
        seg_worker = SegWorker(
            weights=args.seg_weights,
            imgsz=args.seg_imgsz,
            conf=args.seg_conf,
            device=args.seg_device,
            canvas_size=canvas,
            scale_px_per_meter=args.scale,
            on_result=_on_seg,
            interval_s=args.seg_interval,
        )
        print(
            f"  loading YOLO-seg {args.seg_weights.name} "
            f"imgsz={args.seg_imgsz} interval={args.seg_interval}s "
            f"device={args.seg_device} …"
        )
        seg_worker.load()
    else:
        print("  seg disabled")

    ov_worker: AnythingWorker | None = None
    if args.ov:
        ov_worker = AnythingWorker(
            target=args.target,
            conf=args.ov_conf,
            device=ov_dev,
            imgsz=args.ov_imgsz,
            canvas_size=canvas,
            scale_px_per_meter=args.scale,
            on_result=_on_ov,
            interval_s=args.ov_interval,
            send_max_side=int(args.ov_imgsz),
        )
        print(
            f"  YOLO-World GPU FP16  target={args.target}  device={ov_dev}  "
            f"imgsz={args.ov_imgsz} interval={args.ov_interval}s",
            flush=True,
        )
    else:
        print("  YOLO-World off")

    worker: AnalyzeWorker | None = None
    if args.vlm != "off":
        worker = AnalyzeWorker(
            vlm_name=args.vlm,
            models_dir=args.models,
            mode=args.mode,
            grasp_target=args.target,
            max_side=args.max_side,
            max_new_tokens=args.max_new_tokens,
            canvas_size=canvas,
            scale_px_per_meter=args.scale,
            on_result=_on_vlm,
            debug_input_path=out_dir / "vlm_input.jpg",
            # Grasp targets (mouse etc.) are often <1 occ cell; veto was dropping real hits
            occ_veto=None,
            task="caption" if args.mode == "nav" else "analyze",
        )
        gap = (
            float(args.analyze_interval)
            if args.analyze_interval > 0
            else (3.0 if args.analyze_interval >= 0 else -1.0)
        )
        role = "caption" if args.mode == "nav" else "analyze"
        print(
            f"  VLM {args.vlm}  {role}  target={args.target}  "
            f"max_side={args.max_side} tokens={args.max_new_tokens} "
            f"interval={gap}s — 后台加载，先出画面"
            + (" (manual a)" if gap < 0 else "")
        )
    else:
        print("  VLM disabled (--vlm off)")

    if use_occ or args.ov or args.vlm != "off":
        print(
            "  fusion: occ=通行空间  YOLO-World=框/米制坐标  "
            f"VLM={'口述(可与YOLO并行)' if args.vlm != 'off' and args.mode == 'nav' else ('analyze' if args.vlm != 'off' else 'off')}",
            flush=True,
        )

    gains = {d: np.ones(3, dtype=np.float32) for d in DIRECTIONS}
    stitch_gains = {d: np.ones(3, dtype=np.float32) for d in DIRECTIONS}
    gain_enabled = not args.no_gain
    need_gain_update = gain_enabled
    blend_power = args.blend_power
    frame_count = 0
    fps_smooth = 0.0
    window_name = "Surround View"
    preview_path = out_dir / "preview.jpg"
    use_window = not args.no_window
    window_ready = False

    print("  warming CUDA stitch (first GPU compile can stall a few seconds) …", flush=True)
    t_warm = time.perf_counter()
    for i in range(2):
        raw = grab_frames(caps)
        if not raw:
            time.sleep(0.05)
            continue
        result, bev_views, _ft = process_frame(
            raw,
            pipelines,
            stitch_gains,
            canvas,
            need_bev_views=(i == 0 and gain_enabled),
        )
        if i == 0 and gain_enabled:
            new_gains = compute_gains(bev_views)
            if new_gains:
                for d, g in new_gains.items():
                    if d in gains:
                        gains[d] = gains[d] * 0.5 + g * 0.5
            _push_weights(pipelines, _fold_gain_weights(base_weights, gains))
            need_gain_update = False
            print("  gain folded into GPU blend weights (keeps CUDA stitch)", flush=True)
        if occ_tracker is not None:
            occ_tracker.update(result)
    print(
        f"  stitch ready  warmup={(time.perf_counter() - t_warm) * 1000.0:.0f}ms",
        flush=True,
    )

    if use_window:
        try:
            _open_preview_window(window_name)
            window_ready = True
        except Exception as exc:
            print(
                f"  [warn] OpenCV window unavailable ({exc}); "
                f"fallback → console + {preview_path}",
                flush=True,
            )
            use_window = False

    print("=" * 60)
    if use_window:
        keys = "ESC/q quit  a:vlm  s:save  m:map"
        if ov_worker:
            keys += "  o:ov"
        if seg_worker:
            keys += "  x:seg"
        print(f"  running - {keys}")
        if ov_worker:
            print("  VLM off by default in grasp; YOLO-World gives real xy")
        if seg_worker:
            print("  [warn] seg 开着会抢 GPU；卡顿可加大 --seg-interval 或 --seg-device cpu")
    else:
        print(f"  无窗口模式 — 预览 {preview_path}，事件 {logger.path}")
    if worker is not None:
        print("  VLM load deferred ~3s (video first)", flush=True)
    print("=" * 60)

    vlm_kick_started = False
    ov_kick_started = False
    stitch_before_vlm = 48
    stitch_before_ov = 36
    cuda_paused = False
    pipe_cuda = {d: pipelines[d].use_cuda for d in pipelines}
    ego_box = None

    try:
        while True:
            if (
                ov_worker is not None
                and not ov_kick_started
                and frame_count >= stitch_before_ov
            ):
                print(
                    "  starting YOLO-World on GPU (CPU stitch while loading, video stays live) ...",
                    flush=True,
                )
                ov_worker.load_async()
                ov_kick_started = True
            if (
                worker is not None
                and not vlm_kick_started
                and frame_count >= stitch_before_vlm
            ):
                print("  starting VLM load in background ...", flush=True)
                worker.load_async()
                vlm_kick_started = True

            raw = grab_frames(caps)
            if not raw:
                time.sleep(0.02)
                continue

            loop_t0 = time.perf_counter()
            gpu_hold = bool(
                (ov_worker is not None and ov_worker.holds_gpu)
                or (worker is not None and worker.holds_gpu)
            )
            # Weight load vs OpenCV-CUDA: pause GPU stitch. Infer may overlap.
            if gpu_hold and not cuda_paused:
                cuda_paused = True
                for p in pipelines.values():
                    p.use_cuda = False
                print("  GPU busy (detector/VLM) — CPU stitch, video stays live", flush=True)
            elif not gpu_hold and cuda_paused:
                cuda_paused = False
                for d, on in pipe_cuda.items():
                    pipelines[d].use_cuda = on
                print("  CUDA stitch resumed", flush=True)
            result, bev_views, frame_time = process_frame(
                raw,
                pipelines,
                stitch_gains,
                canvas,
                need_bev_views=need_gain_update and gain_enabled,
            )
            if need_gain_update and gain_enabled:
                new_gains = compute_gains(bev_views)
                if new_gains:
                    for d, g in new_gains.items():
                        if d in gains:
                            gains[d] = gains[d] * 0.5 + g * 0.5
                _push_weights(pipelines, _fold_gain_weights(base_weights, gains))
                need_gain_update = False
            stitch_ms = float(frame_time) * 1000.0
            occ_grid = None
            if occ_tracker is not None:
                occ_grid, occ_event, should_log = occ_tracker.update(result)
                latest_occ = occ_event
                if should_log:
                    bus.publish(occ_event)

            frame_count += 1

            # Occupancy = free/occ space; YOLO-World = boxes + xy;
            # VLM caption = human-readable notes (not used as geometry).
            geom = latest_seg or latest_occ
            event = _compose_event(
                mode=args.mode,
                geom=geom,
                ov_ev=latest_ov,
                vlm_ev=latest_vlm,
                vlm_boxes=bool(
                    worker is not None and getattr(worker, "task", "") == "analyze"
                ),
            )
            if event is None:
                event = (worker.event if worker else None) or bus.latest()
            src = event.grasp.source if event is not None and event.grasp else ""
            if (
                args.mode == "grasp"
                and event is not None
                and event.grasp is not None
                and event.grasp.targets
                and occ_grid is not None
                and src not in ("yolo-world", "yoloe")
            ):
                snap_grasp_to_occupancy(event, occ_grid)
                refresh_grasp_hint(event)

            status_bits = [f"stitch:{stitch_ms:.0f}ms"]
            if occ_tracker and occ_tracker.latest_event is not None:
                ff = (
                    occ_tracker.latest_event.nav.free_frac
                    if occ_tracker.latest_event.nav
                    else None
                )
                ms = occ_tracker.latest_event.infer_ms
                status_bits.append(
                    f"occ:{ms:.0f}ms" + (f" free={ff:.0%}" if ff is not None else "")
                )
            if seg_worker:
                status_bits.append(seg_worker.status_line)
            if ov_worker:
                status_bits.append(ov_worker.status_line)
            status_bits.append(worker.status_line if worker else "vlm: off")
            status_line = "  ".join(status_bits)

            az_hint = ""
            if occ_tracker is not None and occ_tracker.latest_event is not None:
                nav = occ_tracker.latest_event.nav
                if nav is not None:
                    seen_az: list[str] = []
                    for o in nav.obstacles:
                        a = o.azimuth
                        if a in ("", "unknown", "center") or a in seen_az:
                            continue
                        seen_az.append(a)
                    az_hint = ",".join(seen_az[:6])

            if seg_worker and seg_worker.enabled:
                seg_worker.request(result)
            ov_loading = bool(ov_worker is not None and ov_worker.loading)
            vlm_loading = bool(worker is not None and worker.loading)
            if ov_worker and ov_worker.enabled and not vlm_loading:
                ov_worker.request(result)
            if worker and worker.enabled and args.analyze_interval >= 0:
                now = time.time()
                gap = (
                    float(args.analyze_interval)
                    if args.analyze_interval > 0
                    else 3.0
                )
                since_ready = now - worker.ready_at if worker.ready_at else 0.0
                cooldown_ok = since_ready >= gap and (
                    last_vlm_done_t <= 0 or (now - last_vlm_done_t) >= gap
                )
                if (
                    cooldown_ok
                    and not worker.busy
                    and not ov_loading
                    and stitch_ms < 90.0
                ):
                    worker.request(result, occ_az=az_hint)

            ds = int(args.display_size)
            display = resize_bgr(result, (ds, ds))
            if occ_grid is not None:
                if event is not None:
                    stamp_detections_on_grid(occ_grid, event)
                # Occupancy paint lives on the right map; left stays the raw stitch.
                if not show_occ_map:
                    display = overlay_occupancy(
                        display, occ_grid, draw_vehicle_box=ego_sprite is None
                    )
            scale_disp = float(args.scale) * (ds / float(canvas[0]))
            if ego_sprite is not None:
                if ego_box is None:
                    gmean = float(
                        cv2.cvtColor(display, cv2.COLOR_BGR2GRAY).mean()
                    )
                    if gmean > 28.0:
                        ego_box = center_blind_box(display)
                        print(
                            f"  ego overlay locked {ego_box[2]-ego_box[0]}x"
                            f"{ego_box[3]-ego_box[1]}px "
                            f"@ ({ego_box[0]},{ego_box[1]}) "
                            f"(display, frozen)",
                            flush=True,
                        )
                display = overlay_ego(
                    display,
                    ego_sprite,
                    size_m=args.ego_size_m,
                    scale_px_per_meter=scale_disp,
                    box=ego_box if args.ego_size_m <= 0 else None,
                )
            loop_dt = time.perf_counter() - loop_t0
            fps_smooth = fps_smooth * 0.9 + (1.0 / max(loop_dt, 0.001)) * 0.1
            fps_now = fps_smooth if fps_smooth > 0 else (1.0 / max(loop_dt, 0.001))
            display = draw_hud(
                display,
                fps_val=fps_now,
                blend_power=blend_power,
                gain_enabled=gain_enabled,
                canvas_size=(ds, ds),
                scale=scale_disp,
                mode=args.mode,
                event=event,
                vlm_status=status_line,
                grasp_target=args.target if args.mode == "grasp" else "",
                vehicle_marker=ego_sprite is None,
            )
            if show_occ_map and occ_grid is not None:
                display = np.hstack(
                    [
                        display,
                        render_occupancy_map(occ_grid, event=event, size=ds),
                    ]
                )
            display = finish_film_frame(
                display,
                fps_val=fps_now,
                mode=args.mode,
                range_m=float(args.range),
                status_line=status_line,
                event=event,
                grasp_target=args.target if args.mode == "grasp" else "",
            )

            key = 0xFF
            if use_window and window_ready:
                try:
                    cv2.imshow(window_name, np.ascontiguousarray(display))
                    key = cv2.waitKey(1) & 0xFF
                except Exception as exc:
                    print(
                        f"  [warn] imshow failed ({exc}); fallback → {preview_path}",
                        flush=True,
                    )
                    use_window = False
                    window_ready = False
                    try:
                        cv2.destroyAllWindows()
                    except Exception:
                        pass

            if not use_window:
                if frame_count % 15 == 1:
                    cv2.imwrite(str(preview_path), display)
                if frame_count % 30 == 0:
                    summary = event.summary[:80] if event else ""
                    print(
                        f"  fps≈{fps_smooth:.1f}  {status_line}  "
                        f"sum={summary!r}",
                        flush=True,
                    )
                time.sleep(0.001)

            if key in (27, ord("q")):
                break
            if key == ord("s"):
                path = out_dir / f"bev_{frame_count:04d}.jpg"
                cv2.imwrite(str(path), display)
                raw_path = out_dir / f"bev_{frame_count:04d}_raw.jpg"
                cv2.imwrite(str(raw_path), result)
                print(f"  [saved] {path}  (raw {raw_path.name})")
            if key == ord("a") and worker:
                if not worker.enabled:
                    print("  [vlm] still loading")
                elif worker.busy:
                    print("  [vlm] busy — wait for current round")
                elif worker.request(result, occ_az=az_hint):
                    print("  [vlm] analyze requested")
            if key == ord("o") and ov_worker:
                if not ov_worker.enabled:
                    print("  [ov] still loading")
                elif ov_worker.busy:
                    print("  [ov] busy")
                elif ov_worker.request(result, force=True):
                    print("  [ov] requested")
            if key == ord("x") and seg_worker:
                if seg_worker.request(result, force=True):
                    print("  [seg] requested")
            if key == ord("m") and occ_tracker is not None:
                show_occ_map = not show_occ_map
                print(f"  2D occ map: {'ON' if show_occ_map else 'OFF'}")
            if key == ord("g"):
                gain_enabled = not gain_enabled
                if gain_enabled:
                    need_gain_update = True
                else:
                    gains = {d: np.ones(3, dtype=np.float32) for d in DIRECTIONS}
                    _push_weights(pipelines, base_weights)
                print(f"  gain: {'ON' if gain_enabled else 'OFF'}")
            if key in (ord("+"), ord("=")):
                blend_power = min(blend_power + 1.0, 10.0)
                base_weights = build_weight_maps(canvas, center, blend_power)
                _push_weights(
                    pipelines,
                    _fold_gain_weights(base_weights, gains)
                    if gain_enabled
                    else base_weights,
                )
                print(f"  blend_power={blend_power}")
            if key == ord("-"):
                blend_power = max(blend_power - 1.0, 1.0)
                base_weights = build_weight_maps(canvas, center, blend_power)
                _push_weights(
                    pipelines,
                    _fold_gain_weights(base_weights, gains)
                    if gain_enabled
                    else base_weights,
                )
                print(f"  blend_power={blend_power}")
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        logger.close()
        if seg_worker is not None:
            try:
                seg_worker.close()
            except Exception:
                pass
        if ov_worker is not None:
            try:
                ov_worker.close()
            except Exception:
                pass
        if worker is not None:
            try:
                worker.close()
            except Exception:
                pass
        if window_ready:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        for cap in caps.values():
            try:
                cap.release()
            except Exception:
                pass
        print("  done.")


if __name__ == "__main__":
    main()
