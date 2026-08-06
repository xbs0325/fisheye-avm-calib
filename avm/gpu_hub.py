#!/usr/bin/env python3
"""Shared camera + CUDA OpenCV hub for CLI live and Web MJPEG.

Modes:
  idle / preview / bev / raw
  calib_intrinsics — 内参采集预览（WebRTC）
  calib_extrinsics — 外参稳定+连拍预览（WebRTC）

Policy: preview/bev require CUDA when require_cuda=True (default).
Old Web died on CPU remap/blend; this hub keeps that work on GPU.
JPEG encode stays on CPU at reduced resolution only.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from avm.cuda_cv import (
    UndistortWarpPipeline,
    cuda_available,
    cuda_status_line,
    init_undistort_maps,
    log_cuda_status,
    process_frames_to_bev,
    resize_bgr,
)

import cv2  # noqa: E402

try:
    from avm.event_log import LOG
except Exception:  # pragma: no cover
    class _Null:
        def info(self, *_a, **_k): pass
        def warn(self, *_a, **_k): pass
        def error(self, *_a, **_k): pass
    LOG = _Null()  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ("front", "back", "left", "right")
CAPTURE_W, CAPTURE_H = 1920, 1536
TILE_W, TILE_H = 480, 360


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def open_camera(index: int, *, timeout_s: float = 8.0):
    """Open V4L2 camera; fail fast if driver hangs."""
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开 /dev/video{index}")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(3):
                cap.grab()
            box["cap"] = cap
        except Exception as exc:  # noqa: BLE001
            box["err"] = exc

    t = threading.Thread(target=_run, name=f"open-cam-{index}", daemon=True)
    t.start()
    t.join(timeout=float(timeout_s))
    if t.is_alive():
        raise TimeoutError(
            f"/dev/video{index} 打开超时 ({timeout_s:.0f}s)，可能被占用或驱动卡住"
        )
    if "err" in box:
        raise box["err"]
    return box["cap"]


def build_weight_maps(canvas_size, center, blend_power: float):
    h, w = canvas_size[1], canvas_size[0]
    cx, cy = center
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    gx = (x_grid - cx).astype(np.float64)
    gy = (cy - y_grid).astype(np.float64)
    angle = np.arctan2(gy, gx)
    cam_angle = {
        "front": np.pi / 2.0,
        "back": -np.pi / 2.0,
        "left": np.pi,
        "right": 0.0,
    }
    raw = {}
    for d, ca in cam_angle.items():
        diff = np.arctan2(np.sin(angle - ca), np.cos(angle - ca))
        raw[d] = np.clip(np.cos(diff), 0.0, 1.0) ** blend_power
    weight_sum = np.maximum(sum(raw.values()), 1e-10)
    return {d: (raw[d] / weight_sum).astype(np.float32) for d in raw}


def adjust_homography(H_old, old_scale, old_canvas, new_scale, new_canvas):
    old_cw, old_ch = old_canvas
    new_cw, new_ch = new_canvas
    s = new_scale / old_scale
    tx = new_cw / 2.0 - s * old_cw / 2.0
    ty = new_ch / 2.0 - s * old_ch / 2.0
    A = np.array([[s, 0, tx], [0, s, ty], [0, 0, 1.0]], dtype=np.float64)
    return A @ H_old


class GpuStreamHub:
    """Single owner of cameras; produces JPEG for MJPEG clients."""

    def __init__(
        self,
        *,
        config_path: Optional[Path] = None,
        calib_dir: Optional[Path] = None,
        jpeg_quality: int = 70,
        display_width: int = 800,
        blend_power: float = 4.0,
        bev_range_m: float = 1.0,
        bev_scale: float = 200.0,
        require_cuda: bool = True,
        preview_balance: float = 0.5,
    ):
        self.config_path = Path(config_path or (ROOT / "config" / "camera_config.json"))
        self.calib_dir = Path(calib_dir or (ROOT / "calib_results"))
        self.jpeg_quality = int(jpeg_quality)
        self.display_width = int(display_width)
        self.blend_power = float(blend_power)
        self.bev_range_m = float(bev_range_m)
        self.bev_scale = float(bev_scale)
        self.require_cuda = bool(require_cuda)
        self.preview_balance = float(preview_balance)

        self._lock = threading.RLock()
        self._mode = "idle"
        self._caps: dict[str, Any] = {}
        self._preview_pipes: dict[str, UndistortWarpPipeline] = {}
        self._bev_pipes: dict[str, UndistortWarpPipeline] = {}
        self._bev_canvas: tuple[int, int] = (400, 400)
        self._jpeg: Optional[bytes] = None
        self._bgr: Optional[np.ndarray] = None
        self._fps = 0.0
        self._gpu_ms = 0.0
        self._encode_ms = 0.0
        self._make_jpeg = False  # WebRTC 主路径不需要 JPEG
        self._error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cond = threading.Condition(self._lock)
        self._seq = 0
        self._using_cuda = cuda_available()
        self.calib = None  # WebCalibSession | None
        self._pause_grab = threading.Event()  # set=暂停取流（外参连拍时）

    def calib_status(self) -> dict[str, Any]:
        if self.calib is None:
            return {"kind": None}
        return self.calib.status()

    def calib_action(self, cmd: str) -> dict[str, Any]:
        if self.calib is None:
            return {"ok": False, "error": "未启动标定会话"}
        return self.calib.action(cmd)

    def pause_grab(self, on: bool) -> None:
        if on:
            self._pause_grab.set()
        else:
            self._pause_grab.clear()

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def status(self) -> dict[str, Any]:
        with self._lock:
            cuda_ok = cuda_available()
            # idle 时 using_cuda 表示「设备可用」；推流中表示「本路管道真在用 GPU」
            if self._mode == "idle":
                using = cuda_ok
            else:
                using = bool(self._using_cuda)
            return {
                "mode": self._mode,
                "cuda": cuda_ok,
                "cuda_line": cuda_status_line(),
                "using_cuda": using,
                "pipeline_cuda": bool(self._using_cuda) if self._mode != "idle" else None,
                "fps": round(self._fps, 1),
                "gpu_ms": round(self._gpu_ms, 1),
                "encode_ms": round(self._encode_ms, 1),
                "error": self._error,
                "seq": self._seq,
                "cameras": sorted(self._caps.keys()),
                "has_frame": self._bgr is not None,
                "has_jpeg": self._jpeg is not None,
                "require_cuda": self.require_cuda,
                "transport": "webrtc",
                "calib": self.calib.status() if self.calib is not None else {"kind": None},
            }

    def latest_bgr(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._bgr is None else self._bgr.copy()

    def wait_bgr(
        self, last_seq: int, timeout: float = 1.0
    ) -> tuple[int, Optional[np.ndarray]]:
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout=timeout)
            if self._bgr is None:
                return self._seq, None
            return self._seq, self._bgr.copy()

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def wait_jpeg(self, last_seq: int, timeout: float = 1.0) -> tuple[int, Optional[bytes]]:
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout=timeout)
            return self._seq, self._jpeg

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=5.0)
        if self.calib is not None:
            try:
                self.calib.stop()
            except Exception as exc:
                LOG.warn(f"calib.stop: {exc}")
        with self._lock:
            self._release_caps_unlocked()
            self._mode = "idle"
            self._thread = None
            self._preview_pipes.clear()
            self._bev_pipes.clear()
            self._error = None
            self.calib = None
            self._pause_grab.clear()

    def start(self, mode: str) -> dict[str, Any]:
        mode = mode.lower().strip()
        allowed = ("preview", "bev", "raw", "calib_intrinsics", "calib_extrinsics", "calib_seam")
        if mode not in allowed:
            raise ValueError(f"未知 mode={mode}")
        if mode in ("preview", "bev") and self.require_cuda and not cuda_available():
            raise RuntimeError(
                "CUDA OpenCV 不可用，拒绝启动 Web 视频（历史 CPU 推流仅 1–2 FPS）。"
                "请: source scripts/env_opencv_cuda.sh"
            )
        LOG.info(f"hub.stop 旧模式={self._mode}")
        self.stop()
        self._stop.clear()
        with self._lock:
            self._error = None
            self._mode = mode
            self._using_cuda = cuda_available()
            try:
                LOG.info(f"hub.start mode={mode} 打开相机…")
                self._open_caps_unlocked()
                LOG.info(f"hub 相机已开: {sorted(self._caps.keys())}")
                if mode == "preview":
                    LOG.info("构建 GPU preview pipeline…")
                    self._build_preview_unlocked()
                elif mode == "bev":
                    LOG.info("构建 GPU bev pipeline…")
                    self._build_bev_unlocked()
                elif mode in ("calib_intrinsics", "calib_extrinsics", "calib_seam"):
                    from avm.web_calib import WebCalibSession

                    self.calib = WebCalibSession(self)
                    if mode == "calib_intrinsics":
                        self.calib.prepare_intrinsics()
                    elif mode == "calib_extrinsics":
                        self.calib.prepare_extrinsics()
                    else:
                        self.calib.prepare_seam()
                LOG.info(f"hub.start OK cuda={self._using_cuda}")
            except Exception as exc:
                self._release_caps_unlocked()
                self._mode = "idle"
                self.calib = None
                self._error = str(exc)
                LOG.error(f"hub.start 失败: {exc}")
                raise
        self._thread = threading.Thread(
            target=self._loop, name=f"gpu-hub-{mode}", daemon=True
        )
        self._thread.start()
        return self.status()

    def _open_caps_unlocked(self) -> None:
        cfg = _load_json(self.config_path)
        errors = []
        for d in DIRECTIONS:
            idx = cfg.get(d)
            if idx is None:
                continue
            LOG.info(f"  open {d} /dev/video{idx} …")
            try:
                self._caps[d] = open_camera(int(idx))
                LOG.info(f"  open {d} OK")
            except Exception as exc:
                LOG.warn(f"  open {d} FAIL: {exc}")
                errors.append(f"{d}:/dev/video{idx} ({exc})")
        if not self._caps:
            raise RuntimeError(
                "无可用相机。" + (" ".join(errors) if errors else "")
            )
        if errors:
            LOG.warn(f"部分相机未打开: {errors}")

    def _release_caps_unlocked(self) -> None:
        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass
        self._caps.clear()

    def _build_preview_unlocked(self) -> None:
        log_cuda_status()
        for d, cap in self._caps.items():
            path = self.calib_dir / f"{d}.json"
            if not path.is_file():
                raise FileNotFoundError(f"缺少内参: {path}")
            data = _load_json(path)
            K = np.asarray(data["K"], dtype=np.float64)
            D = np.asarray(data["D"], dtype=np.float64)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or CAPTURE_W)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or CAPTURE_H)
            m1, m2 = init_undistort_maps(
                K, D, w, h, self.preview_balance, for_cuda=True
            )
            pipe = UndistortWarpPipeline(m1, m2)
            if self.require_cuda and not pipe.use_cuda:
                raise RuntimeError(f"{d}: UndistortWarpPipeline 未走 CUDA")
            self._preview_pipes[d] = pipe

    def _build_bev_unlocked(self) -> None:
        log_cuda_status()
        extr_path = self.calib_dir / "extrinsics.json"
        if not extr_path.is_file():
            raise FileNotFoundError(f"缺少外参: {extr_path}")
        extr = _load_json(extr_path)
        balance = float(
            extr.get("extrinsic_balance", extr.get("balance", self.preview_balance))
        )
        old_scale = float(extr.get("scale_px_per_meter", 100.0))
        old_canvas = tuple(extr.get("canvas_size", [1000, 1000]))
        homographies = extr.get("homographies") or {}
        cw = int(2.0 * self.bev_range_m * self.bev_scale)
        ch = cw
        canvas = (cw, ch)
        center = (cw / 2.0, ch / 2.0)
        self._bev_canvas = canvas
        weights = build_weight_maps(canvas, center, self.blend_power)
        for d, cap in self._caps.items():
            if d not in homographies:
                continue
            path = self.calib_dir / f"{d}.json"
            data = _load_json(path)
            K = np.asarray(data["K"], dtype=np.float64)
            D = np.asarray(data["D"], dtype=np.float64)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or CAPTURE_W)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or CAPTURE_H)
            m1, m2 = init_undistort_maps(K, D, w, h, balance, for_cuda=True)
            H = adjust_homography(
                np.asarray(homographies[d], dtype=np.float64),
                old_scale,
                old_canvas,
                self.bev_scale,
                canvas,
            )
            pipe = UndistortWarpPipeline(
                m1, m2, H=H, canvas_size=canvas, weight=weights[d]
            )
            if self.require_cuda and not pipe.use_cuda:
                raise RuntimeError(f"{d}: BEV pipeline 未走 CUDA")
            self._bev_pipes[d] = pipe

    def _grab(self) -> dict[str, np.ndarray]:
        with self._lock:
            caps = dict(self._caps)
        for cap in caps.values():
            cap.grab()
        frames: dict[str, np.ndarray] = {}
        for d, cap in caps.items():
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                frames[d] = frame
        return frames

    def _compose_preview(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        tiles = []
        labels = []
        with self._lock:
            pipes = dict(self._preview_pipes)
        for d in DIRECTIONS:
            labels.append(d)
            if d not in frames or d not in pipes:
                tiles.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))
                continue
            und = pipes[d].undistort(frames[d])
            tiles.append(resize_bgr(und, (TILE_W, TILE_H)))
        top = np.hstack([tiles[0], tiles[1]])
        bot = np.hstack([tiles[2], tiles[3]])
        grid = np.vstack([top, bot])
        for i, name in enumerate(labels):
            x = (i % 2) * TILE_W + 8
            y = (i // 2) * TILE_H + 24
            cv2.putText(
                grid, name.upper(), (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
        return grid

    def _compose_raw(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        tiles = []
        for d in DIRECTIONS:
            if d not in frames:
                tiles.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))
            else:
                tiles.append(resize_bgr(frames[d], (TILE_W, TILE_H)))
        return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])

    def _compose_bev(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        with self._lock:
            pipes = dict(self._bev_pipes)
            canvas = self._bev_canvas
        t0 = time.perf_counter()
        bev, _, _ = process_frames_to_bev(
            frames, pipes, list(DIRECTIONS), canvas_size=canvas, need_bev_views=False
        )
        self._gpu_ms = (time.perf_counter() - t0) * 1000.0
        return bev

    def _prepare_display(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if self.display_width > 0 and w != self.display_width:
            nh = max(1, int(round(h * self.display_width / w)))
            image = resize_bgr(image, (self.display_width, nh))
        else:
            image = image.copy()
        cv2.putText(
            image,
            f"{self._mode}  {self._fps:.1f}fps  gpu={self._gpu_ms:.0f}ms  "
            f"{'CUDA' if self._using_cuda else 'CPU'}  WebRTC",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        return image

    def _loop(self) -> None:
        fps_t0 = time.perf_counter()
        fps_n = 0
        while not self._stop.is_set():
            try:
                if self._pause_grab.is_set():
                    time.sleep(0.03)
                    continue
                frames = self._grab()
                if not frames:
                    time.sleep(0.02)
                    continue
                mode = self.mode
                t0 = time.perf_counter()
                if mode == "preview":
                    img = self._compose_preview(frames)
                    self._gpu_ms = (time.perf_counter() - t0) * 1000.0
                elif mode == "bev":
                    img = self._compose_bev(frames)
                elif mode in ("calib_intrinsics", "calib_extrinsics", "calib_seam") and self.calib is not None:
                    img = self.calib.compose(frames)
                    self._gpu_ms = (time.perf_counter() - t0) * 1000.0
                else:
                    img = self._compose_raw(frames)
                    self._gpu_ms = (time.perf_counter() - t0) * 1000.0
                display = self._prepare_display(img)
                jpeg = None
                if self._make_jpeg:
                    t1 = time.perf_counter()
                    ok, buf = cv2.imencode(
                        ".jpg",
                        display,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                    )
                    self._encode_ms = (time.perf_counter() - t1) * 1000.0
                    if ok:
                        jpeg = buf.tobytes()
                else:
                    self._encode_ms = 0.0
                fps_n += 1
                if fps_n >= 10:
                    dt = time.perf_counter() - fps_t0
                    self._fps = fps_n / max(dt, 1e-6)
                    fps_t0 = time.perf_counter()
                    fps_n = 0
                with self._cond:
                    self._bgr = display
                    self._jpeg = jpeg
                    self._seq += 1
                    self._error = None
                    self._cond.notify_all()
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                time.sleep(0.2)
