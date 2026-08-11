#!/usr/bin/env python3
"""Post-calibration BEV live demo + Qwen3-VL captions.

Assumes calib_results/{front,back,left,right,extrinsics}.json already exist.
Does not touch the Web wizard / avm.live_bev entry points.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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
    draw_hud,
    grab_frames,
    open_caps,
    process_frame,
)
from demo_bev_vlm.vlm_caption import CaptionWorker  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="标定后 BEV 实时拼接 Demo + VLM 描述（拷贝自 live stitch）"
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    ap.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    ap.add_argument("--range", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=200.0)
    ap.add_argument("--blend-power", type=float, default=4.0)
    ap.add_argument("--no-gain", action="store_true")
    ap.add_argument("--display-size", type=int, default=560)
    ap.add_argument("--vlm", default="qwen3vl-2b",
                    choices=["qwen3vl-2b", "qwen3vl-4b", "qwen3vl-8b", "off"])
    ap.add_argument("--caption-interval", type=float, default=12.0,
                    help="自动描述间隔秒；0=仅按 c 手动触发")
    ap.add_argument("--models", type=Path, default=None,
                    help="WorldMM models dir (default WORLDMM_MODELS)")
    ap.add_argument("--no-window", action="store_true",
                    help="无显示器：只打印 FPS/caption")
    args = ap.parse_args()

    print("=" * 60)
    print("  BEV + VLM Demo（标定后）")
    print("=" * 60)

    pipelines, canvas, center, balance = build_pipelines(
        extrinsics_path=args.extrinsics,
        calib_dir=args.calib_dir,
        range_m=args.range,
        scale=args.scale,
        blend_power=args.blend_power,
    )
    print(f"  canvas={canvas[0]}x{canvas[1]}  balance={balance:.2f}")
    print(f"  pipelines: {', '.join(pipelines.keys())}")

    caps = open_caps(args.config)
    print(f"  cameras: {', '.join(sorted(caps.keys()))}")

    worker: CaptionWorker | None = None
    if args.vlm != "off":
        worker = CaptionWorker(vlm_name=args.vlm, models_dir=args.models)
        print(f"  loading VLM {args.vlm} …")
        worker.load()
    else:
        print("  VLM disabled (--vlm off)")

    gains = {d: np.ones(3, dtype=np.float32) for d in DIRECTIONS}
    gain_enabled = not args.no_gain
    need_gain_update = gain_enabled
    blend_power = args.blend_power
    frame_count = 0
    fps_smooth = 0.0
    # Start clock now so we don't fire caption on the first frame (was epoch 0).
    last_caption_t = time.time()
    window_name = "BEV+VLM Demo"
    out_dir = ROOT / "output" / "demo_bev_vlm"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_path = out_dir / "preview.jpg"
    use_window = not args.no_window
    window_ready = False

    if use_window:
        # Prefer Qt if this OpenCV build has it; GTK often fails over SSH / no session.
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            window_ready = True
        except Exception as exc:
            print(f"  [warn] OpenCV window unavailable ({exc}); "
                  f"fallback → console + {preview_path}", flush=True)
            use_window = False

    print("=" * 60)
    if use_window:
        print("  运行中 — ESC/q 退出，c 立即描述")
    else:
        print(f"  无窗口模式 — 预览写入 {preview_path}，Ctrl+C 退出")
        print("  （本机桌面可设 DISPLAY=:0 并确保有图形会话；或 SSH 用 --no-window）")
    print("=" * 60)

    try:
        while True:
            raw = grab_frames(caps)
            if not raw:
                time.sleep(0.02)
                continue

            result, bev_views, frame_time = process_frame(
                raw,
                pipelines,
                gains,
                canvas,
                need_bev_views=need_gain_update and gain_enabled,
            )

            if need_gain_update and gain_enabled:
                new_gains = compute_gains(bev_views)
                if new_gains:
                    for d, g in new_gains.items():
                        if d in gains:
                            gains[d] = gains[d] * 0.5 + g * 0.5
                need_gain_update = False

            frame_count += 1
            fps_smooth = fps_smooth * 0.9 + (1.0 / max(frame_time, 0.001)) * 0.1

            caption = worker.caption if worker else ""
            vlm_status = worker.status_line if worker else "vlm: off"
            now = time.time()
            if (
                worker
                and worker.enabled
                and args.caption_interval > 0
                and (now - last_caption_t) >= args.caption_interval
            ):
                if worker.request(result):
                    last_caption_t = now

            display = draw_hud(
                result,
                frame_time,
                blend_power,
                gain_enabled,
                canvas,
                args.scale,
                caption=caption,
                vlm_status=vlm_status,
            )
            display = resize_bgr(display, (args.display_size, args.display_size))

            key = 0xFF
            if use_window and window_ready:
                try:
                    cv2.imshow(window_name, display)
                    key = cv2.waitKey(1) & 0xFF
                except Exception as exc:
                    print(
                        f"  [warn] imshow failed ({exc}); "
                        f"fallback → {preview_path}",
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
                    print(
                        f"  fps≈{fps_smooth:.1f}  {vlm_status}  "
                        f"cap={caption[:80]!r}",
                        flush=True,
                    )
                time.sleep(0.001)

            if key in (27, ord("q")):
                break
            if key == ord("s"):
                path = out_dir / f"bev_{frame_count:04d}.jpg"
                cv2.imwrite(str(path), result)
                print(f"  [saved] {path}")
            if key == ord("c") and worker:
                if worker.request(result, force=True):
                    last_caption_t = time.time()
                    print("  [vlm] caption requested")
            if key == ord("g"):
                gain_enabled = not gain_enabled
                if gain_enabled:
                    need_gain_update = True
                else:
                    gains = {d: np.ones(3, dtype=np.float32) for d in DIRECTIONS}
                print(f"  gain: {'ON' if gain_enabled else 'OFF'}")
            if key in (ord("+"), ord("=")):
                blend_power = min(blend_power + 1.0, 10.0)
                weights = build_weight_maps(canvas, center, blend_power)
                for d, pipe in pipelines.items():
                    if d in weights:
                        pipe.set_weight(weights[d])
                print(f"  blend_power={blend_power}")
            if key == ord("-"):
                blend_power = max(blend_power - 1.0, 1.0)
                weights = build_weight_maps(canvas, center, blend_power)
                for d, pipe in pipelines.items():
                    if d in weights:
                        pipe.set_weight(weights[d])
                print(f"  blend_power={blend_power}")
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
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
