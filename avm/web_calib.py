#!/usr/bin/env python3
"""Web 内外参标定会话：画面走 GpuStreamHub → WebRTC，按键走 API。"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from avm.board_config import load_chessboard_config
from avm.calib_config import load_settings
from avm.cuda_cv import (
    UndistortWarpPipeline,
    cuda_available,
    init_undistort_maps,
    resize_bgr,
    undistort_points_fisheye,
)
from avm.detect_board_hires import (
    detect_chessboard_hires,
    project_corners_to_tile,
    scan_chessboard,
)
from avm.event_log import LOG

import cv2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ("front", "back", "left", "right")
TILE_W, TILE_H = 480, 360


class WebCalibSession:
    def __init__(self, hub: Any):
        self.hub = hub
        self.kind: Optional[str] = None  # intrinsics | extrinsics
        self.message = ""
        # intrinsics
        self.dir_list: list[str] = []
        self.dir_idx = 0
        self.captured = 0
        self.obj_points: list = []
        self.img_points: list = []
        self.cooldown = 0
        self.pattern = (8, 6)
        self.square = 0.025
        self.min_frames = 15
        self.target_frames = 25
        self.cooldown_set = 12
        self._last_found = False
        self._last_corners = None
        # extrinsics
        self.stable_need = 10
        self.burst_n = 8
        self.burst_min_ok = 5
        self.detect_max_width = 1920
        self.detect_try_scales = [1.0, 0.75, 0.5]
        self.stable_streak = {d: 0 for d in DIRECTIONS}
        self.last_detect_ok = {d: False for d in DIRECTIONS}
        self.last_detect_scale = {d: 0.0 for d in DIRECTIONS}
        self.homographies: dict = {}
        self.rms_errors: dict = {}
        self.h_quality: dict = {}
        self.maps: dict = {}
        self.map_size: dict = {}  # direction -> (w, h) maps were built for
        self.placements: dict = {}
        self.calib_intr: dict = {}
        self.extrinsic_balance = 0.8
        self.scale = 100.0
        self.canvas = (1000, 1000)
        self.detect_rr = 0
        self._lock_busy = False
        self._src_wh: Optional[tuple[int, int]] = None
        # GPU 预览管线（热路径）与后台检测线程（CPU，全分辨率、低频限速）
        self._gpu_pipes: dict[str, UndistortWarpPipeline] = {}
        self.detect_interval_ms = 1000
        self.detect_scan_width = 1920  # 兼容旧状态字段；始终强制等于 max_width
        self.detect_duty = 0.25
        self.detect_use_sb = True
        # 专注 / 容错 / 自动锁定
        self.auto_lock = True
        self.streak_reset_misses = 2
        self.focus_miss_tolerance = 3
        # 单路顺序标定：一次只检一路，四路都摆板也不会互相抢
        self.sequential = True
        self.inview_margin_px = 8
        self.target: Optional[str] = None
        self._focus: Optional[str] = None
        self._miss_streak: dict[str, int] = {d: 0 for d in DIRECTIONS}
        self._det_stage: dict[str, str] = {}
        self._det_thread: Optional[threading.Thread] = None
        self._det_stop = threading.Event()
        self._det_lock = threading.Lock()
        self._det_jobs: dict[str, tuple[np.ndarray, float]] = {}
        self._det_submit_t: dict[str, float] = {}
        self._det_result: dict[str, tuple[bool, Optional[np.ndarray], float]] = {}
        self._det_ms: dict[str, float] = {}
        self._det_sticky: Optional[str] = None
        # 按需转储：写下"检测真正看到的那张图"，用于离线定位漏检
        self._dump_request = False
        self._dump_dir = ROOT / "debug_detect"
        self._apply_settings(load_settings())

    def request_dump(self) -> dict[str, Any]:
        self._dump_request = True
        return {"ok": True, "dir": str(self._dump_dir)}

    def _do_dump(self, d: str, raw: np.ndarray, undist: np.ndarray) -> None:
        try:
            self._dump_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self._dump_dir / f"{d}_raw.png"), raw)
            cv2.imwrite(str(self._dump_dir / f"{d}_undist.png"), undist)
        except Exception as exc:
            LOG.warn(f"dump {d} 失败: {exc}")

    def _apply_settings(self, s: dict) -> None:
        self.detect_max_width = int(s.get("detect_max_width", 1920))
        self.detect_interval_ms = int(s.get("detect_interval_ms", 500))
        # 用户要求：检测不能用低分辨率门控。每次直接在 max_width 上检测。
        self.detect_scan_width = self.detect_max_width
        self.detect_duty = float(s.get("detect_duty", 0.5))
        self.detect_use_sb = bool(s.get("detect_use_sb", True))
        self.detect_try_scales = list(s.get("detect_try_scales") or [1.0, 0.75, 0.5])
        self.stable_need = int(s.get("stable_frames", 10))
        self.auto_lock = bool(s.get("auto_lock", True))
        self.streak_reset_misses = int(s.get("streak_reset_misses", 2))
        self.focus_miss_tolerance = int(s.get("focus_miss_tolerance", 3))
        self.sequential = bool(s.get("sequential", True))
        self.inview_margin_px = int(s.get("inview_margin_px", 8))
        self.burst_n = int(s.get("burst_frames", 8))
        self.burst_min_ok = int(s.get("burst_min_ok", 5))
        self.extrinsic_balance = float(s.get("extrinsic_balance", 0.8))
        self.scale = float(s.get("scale_px_per_m", 100))
        canvas = s.get("canvas") or [1000, 1000]
        self.canvas = (int(canvas[0]), int(canvas[1]))
        self.min_frames = int(s.get("intrinsics_min_frames", 15))
        self.target_frames = int(s.get("intrinsics_target_frames", 25))
        self.cooldown_set = int(s.get("intrinsics_cooldown", 12))

    def reload_settings(self) -> None:
        self._apply_settings(load_settings())
        board = load_chessboard_config()
        self.pattern = tuple(board["pattern_size"])
        self.square = float(board["square_size_m"])

    def _detect(self, frame: np.ndarray):
        return detect_chessboard_hires(
            frame,
            self.pattern,
            max_width=self.detect_max_width,
            try_scales=self.detect_try_scales,
        )

    def status(self) -> dict[str, Any]:
        cur = self.dir_list[self.dir_idx] if self.dir_list and self.dir_idx < len(self.dir_list) else None
        return {
            "kind": self.kind,
            "message": self.message,
            "direction": cur,
            "dir_idx": self.dir_idx,
            "dir_list": list(self.dir_list),
            "captured": self.captured,
            "min_frames": self.min_frames,
            "target_frames": self.target_frames,
            "stable_streak": dict(self.stable_streak),
            "stable_need": self.stable_need,
            "locked": sorted(self.homographies.keys()),
            "busy": self._lock_busy,
            "src_wh": list(self._src_wh) if self._src_wh else None,
            "detect_max_width": self.detect_max_width,
            "detect_scan_width": self.detect_scan_width,
            "detect_interval_ms": self.detect_interval_ms,
            "detect_duty": self.detect_duty,
            "detect_ms": {k: round(v, 1) for k, v in self._det_ms.items()},
            "detect_stage": dict(self._det_stage),
            "focus": self._focus,
            "sequential": self.sequential,
            "target": self.target,
            "pending": self._pending_dirs(),
            "auto_lock": self.auto_lock,
            "miss_streak": dict(self._miss_streak),
            "pattern": list(self.pattern),
        }

    def prepare_intrinsics(self) -> None:
        from avm.calibrate_intrinsics import build_object_points

        self.reload_settings()
        self.kind = "intrinsics"
        self.dir_list = [d for d in DIRECTIONS if d in self.hub._caps]
        self.dir_idx = 0
        self._reset_intr_dir()
        self._objp = build_object_points(self.pattern, self.square)
        self.message = (
            f"内参：{self.dir_list[0] if self.dir_list else '?'} SPACE 抓拍 "
            f"(detect≤{self.detect_max_width}px)"
        )
        LOG.info(
            f"web_calib intrinsics dirs={self.dir_list} board={self.pattern} "
            f"sq={self.square} detect_max_w={self.detect_max_width}"
        )
        self.start_detector()

    def _rebuild_map(self, d: str, w: int, h: int) -> bool:
        """CPU CV_16SC2 maps：只给 SPACE 连拍求 H 用，不进推流循环。"""
        from avm.calibrate_extrinsics import precompute_undistort_maps

        if d not in self.calib_intr:
            return False
        if self.map_size.get(d) == (w, h) and d in self.maps:
            return True
        K = self.calib_intr[d]["K"]
        D = self.calib_intr[d]["D"]
        m1, m2 = precompute_undistort_maps(K, D, w, h, self.extrinsic_balance)
        self.maps[d] = (m1, m2)
        self.map_size[d] = (w, h)
        LOG.info(f"web_calib CPU maps[{d}] for {w}x{h} balance={self.extrinsic_balance}")
        return True

    def prepare_extrinsics(self) -> None:
        from avm.calibrate_extrinsics import load_calib, load_placements
        from avm.gpu_hub import CAPTURE_H, CAPTURE_W

        self.reload_settings()
        self.kind = "extrinsics"
        cols, rows = self.pattern
        place_path = ROOT / "config" / "extrinsic_placements.json"
        self.placements = load_placements(str(place_path), cols, rows, self.square)
        self.calib_intr = {}
        self.maps = {}
        self.map_size = {}
        self._gpu_pipes = {}
        calib_dir = ROOT / "calib_results"
        for d in DIRECTIONS:
            if d not in self.hub._caps:
                continue
            try:
                K, D, rms = load_calib(d, str(calib_dir))
                self.calib_intr[d] = {"K": K, "D": D, "rms": rms}
                # 预览去畸变：CV_32FC1 maps → GPU remap
                gm1, gm2 = init_undistort_maps(
                    K, D, CAPTURE_W, CAPTURE_H, self.extrinsic_balance, for_cuda=True
                )
                self._gpu_pipes[d] = UndistortWarpPipeline(gm1, gm2)
                # 求 H 用的 CPU maps（只在 SPACE 连拍时跑）
                self._rebuild_map(d, CAPTURE_W, CAPTURE_H)
            except Exception as exc:
                LOG.warn(f"web_calib extrinsics skip {d}: {exc}")
        self.stable_streak = {d: 0 for d in DIRECTIONS}
        self.last_detect_ok = {d: False for d in DIRECTIONS}
        self.last_detect_scale = {d: 0.0 for d in DIRECTIONS}
        self.homographies.clear()
        self.rms_errors.clear()
        self.h_quality.clear()
        self._src_wh = None
        self._det_result.clear()
        self._det_ms.clear()
        self._det_sticky = None
        self._focus = None
        self._miss_streak = {d: 0 for d in DIRECTIONS}
        self.target = None
        if self.sequential:
            self.advance_target()
        if self.sequential:
            self.message = (
                f"外参（逐路）：当前 {self.target}，把板子摆进它的画面并整块可见；"
                f"稳定 {self.stable_need} 帧自动锁定并跳下一路"
            )
        else:
            self.message = (
                f"外参：GPU 去畸变预览 + 后台检测，绿框 READY 再 SPACE "
                f"(detect≤{self.detect_max_width}/{self.detect_interval_ms}ms "
                f"stable={self.stable_need} burst={self.burst_n})"
            )
        LOG.info(
            f"web_calib extrinsics cams={sorted(self._gpu_pipes.keys())} "
            f"cuda={cuda_available()} detect_max_w={self.detect_max_width} "
            f"interval={self.detect_interval_ms}ms stable={self.stable_need} "
            f"burst={self.burst_n} pattern={self.pattern} "
            f"(热路径 GPU remap/resize，findChessboard 在后台线程)"
        )
        self.start_detector()

    def _reset_intr_dir(self) -> None:
        self.captured = 0
        self.obj_points = []
        self.img_points = []
        self.cooldown = 0
        self._last_found = False
        self._last_corners = None

    # ---------------- 检测线程（不占推流主循环） ----------------

    def start_detector(self) -> None:
        if self._det_thread is not None and self._det_thread.is_alive():
            return
        self._det_stop.clear()
        self._det_thread = threading.Thread(
            target=self._detector_loop, name="calib-detect", daemon=True
        )
        self._det_thread.start()
        LOG.info(
            f"calib detector 线程启动 interval={self.detect_interval_ms}ms "
            f"width<={self.detect_max_width}"
        )

    def stop(self) -> None:
        self._det_stop.set()
        t = self._det_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._det_thread = None
        with self._det_lock:
            self._det_jobs.clear()

    def _raw_corners_to_undistorted(
        self, d: str, corners: np.ndarray, w: int, h: int
    ) -> np.ndarray:
        """鱼眼原图角点 → 去畸变图坐标（仅用于叠加显示 / 求 H）。"""
        intr = self.calib_intr.get(d)
        if intr is None:
            return corners
        try:
            return undistort_points_fisheye(
                corners, intr["K"], intr["D"], w, h, self.extrinsic_balance
            )
        except Exception as exc:
            LOG.warn(f"{d} 角点去畸变映射失败: {exc}")
            return corners

    def _corners_in_view(
        self, d: str, corners: np.ndarray, w: int, h: int
    ) -> tuple[int, int]:
        """映射到去畸变图后，有多少角点真的落在图像内。"""
        pts = self._raw_corners_to_undistorted(d, corners, w, h).reshape(-1, 2)
        m = float(self.inview_margin_px)
        inside = (
            (pts[:, 0] >= m) & (pts[:, 0] < w - m)
            & (pts[:, 1] >= m) & (pts[:, 1] < h - m)
        )
        return int(inside.sum()), int(len(pts))

    def _submit_detect(self, d: str, img: np.ndarray, scale: float) -> None:
        with self._det_lock:
            self._det_jobs[d] = (img, float(scale))
            self._det_submit_t[d] = time.monotonic()

    def _pick_job(self) -> Optional[tuple[str, np.ndarray, float]]:
        with self._det_lock:
            if not self._det_jobs:
                return None
            # 专注模式：焦点路只要有任务就一直检它，避免被轮询切走
            focus = self._focus
            if focus is not None and focus in self._det_jobs:
                d = focus
            else:
                order = list(self._det_jobs.keys())
                d = order[self.detect_rr % len(order)]
                self.detect_rr = (self.detect_rr + 1) % max(len(order), 1)
            img, scale = self._det_jobs.pop(d)
            return d, img, scale

    def _detector_loop(self) -> None:
        while not self._det_stop.is_set():
            job = self._pick_job()
            if job is None:
                time.sleep(0.03)
                continue
            d, img, scale = job
            t0 = time.perf_counter()
            stage = "miss"
            try:
                # 每次检测都使用完整 detect_max_width；慢没关系，线程与推流解耦。
                found, corners, _, stage = scan_chessboard(
                    img,
                    self.pattern,
                    scan_width=self.detect_max_width,
                    refine_width=self.detect_max_width,
                    use_sb=self.detect_use_sb,
                )
            except Exception as exc:
                LOG.warn(f"detect {d} 异常: {exc}")
                found, corners = False, None
            ms = (time.perf_counter() - t0) * 1000.0
            ok = bool(found and corners is not None)
            corners_full = None
            if ok and scale > 0:
                corners_full = corners.astype(np.float32) / float(scale)
                if self.kind == "extrinsics":
                    fw = int(round(img.shape[1] / scale))
                    fh = int(round(img.shape[0] / scale))
                    n_in, n_tot = self._corners_in_view(d, corners_full, fw, fh)
                    if n_in < n_tot:
                        # 鱼眼原图视场比去畸变预览宽：极边缘瞥到的板子会映射到
                        # 图像外，求 H 只会得到垃圾。判为无效并提示未拍全。
                        ok = False
                        corners_full = None
                        stage = f"oob{n_in}/{n_tot}"
            streak_now = 0
            with self._det_lock:
                self._det_ms[d] = ms
                self._det_stage[d] = stage
                self.last_detect_ok[d] = ok
                if ok:
                    # 命中：保留角点，累积 streak，并把检测焦点钉在这一路
                    self._det_result[d] = (True, corners_full, time.monotonic())
                    self._miss_streak[d] = 0
                    self.stable_streak[d] = self.stable_streak.get(d, 0) + 1
                    self._det_sticky = d
                    self._focus = d
                else:
                    # 未命中：保留上一次角点用于显示，只更新时间戳与计数
                    prev_ok, prev_corners, _ = self._det_result.get(
                        d, (False, None, 0.0)
                    )
                    self._det_result[d] = (
                        prev_ok, prev_corners, time.monotonic()
                    )
                    self._miss_streak[d] = self._miss_streak.get(d, 0) + 1
                    # 单次抖动不清零，连续 miss 达阈值才归零
                    if self._miss_streak[d] >= self.streak_reset_misses:
                        self.stable_streak[d] = 0
                        self._det_result[d] = (False, None, time.monotonic())
                    else:
                        self.stable_streak[d] = max(
                            0, self.stable_streak.get(d, 0) - 1
                        )
                    # 焦点路连续丢失太久才放开，交还轮询
                    if (
                        self._focus == d
                        and self._miss_streak[d] >= self.focus_miss_tolerance
                    ):
                        self._focus = None
                        self._det_sticky = None
                streak_now = self.stable_streak.get(d, 0)

            # 自动锁定：streak 达标就地求 H，不用再按 SPACE
            if (
                self.auto_lock
                and ok
                and streak_now >= self.stable_need
                and d not in self.homographies
                and not self._lock_busy
                and (not self.sequential or self.target == d)
            ):
                try:
                    self._lock_direction(d)
                except Exception as exc:
                    LOG.error(f"{d} 自动锁定失败: {exc}")
                continue

            # 占空比限制：检测耗时 ms 后强制休息，绝不长期吃满一个核
            duty = min(max(self.detect_duty, 0.05), 1.0)
            rest = ms * (1.0 / duty - 1.0)
            if rest > 0:
                self._det_stop.wait(rest / 1000.0)

    # ---------------- 画面合成（只做 GPU 缩放/叠字） ----------------

    def compose(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        if self.kind == "intrinsics":
            return self._compose_intrinsics(frames)
        if self.kind == "extrinsics":
            return self._compose_extrinsics(frames)
        return np.zeros((TILE_H * 2, TILE_W * 2, 3), dtype=np.uint8)

    def _detect_size(self, w: int, h: int) -> tuple[int, int, float]:
        dw = min(int(self.detect_max_width), int(w))
        if dw <= 0:
            dw = int(w)
        scale = dw / float(w)
        dh = max(1, int(round(h * scale)))
        return dw, dh, scale

    def _should_submit(self, d: str) -> bool:
        # 顺序模式：只检当前目标那一路，其余路完全不占 CPU，也不会抢焦点
        if self.kind == "extrinsics" and self.sequential and self.target != d:
            return False
        with self._det_lock:
            if d in self._det_jobs:
                return False
            # 专注模式下只喂焦点路，其余路不占 CPU
            focus = self._focus
            if focus is not None and focus != d and focus not in self.homographies:
                return False
            last = self._det_submit_t.get(d, 0.0)
        return (time.monotonic() - last) * 1000.0 >= self.detect_interval_ms

    def _det_snapshot(self, d: str) -> tuple[bool, Optional[np.ndarray], float, float]:
        with self._det_lock:
            ok, corners, ts = self._det_result.get(d, (False, None, 0.0))
            ms = self._det_ms.get(d, 0.0)
        age = (time.monotonic() - ts) if ts else 999.0
        return ok, corners, age, ms

    def _compose_intrinsics(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        cur = self.dir_list[self.dir_idx] if self.dir_idx < len(self.dir_list) else None
        tiles = []
        labels = []
        for d in DIRECTIONS:
            labels.append(d)
            fr = frames.get(d)
            if fr is None:
                tiles.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))
                continue
            h0, w0 = fr.shape[:2]
            self._src_wh = (int(w0), int(h0))
            tile = resize_bgr(fr, (TILE_W, TILE_H))
            if d == cur:
                # 内参：原图检测（去畸变前），检测交给后台线程
                if self.cooldown <= 0 and self._should_submit(d):
                    dw, dh, sc = self._detect_size(w0, h0)
                    self._submit_detect(d, resize_bgr(fr, (dw, dh)), sc)
                ok, corners, age, ms = self._det_snapshot(d)
                fresh = age < 1.5
                self._last_found = bool(ok and fresh)
                self._last_corners = corners if self._last_found else None
                if self._last_found and corners is not None:
                    c_tile = project_corners_to_tile(
                        corners, (w0, h0), (TILE_W, TILE_H)
                    )
                    cv2.drawChessboardCorners(tile, self.pattern, c_tile, True)
                    cv2.rectangle(tile, (2, 2), (TILE_W - 3, TILE_H - 3), (0, 255, 0), 2)
                else:
                    cv2.rectangle(tile, (2, 2), (TILE_W - 3, TILE_H - 3), (0, 200, 255), 2)
                cv2.putText(
                    tile, f"det {ms:.0f}ms age {age:.1f}s", (12, TILE_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                )
                cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), (0, 255, 255), 3)
            tiles.append(tile)
        if self.cooldown > 0:
            self.cooldown -= 1
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        for i, name in enumerate(labels):
            x = (i % 2) * TILE_W + 8
            y = (i // 2) * TILE_H + 24
            tag = "<<" if name == cur else ""
            cv2.putText(
                grid, f"{name.upper()}{tag}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
        src = f"{self._src_wh[0]}x{self._src_wh[1]}" if self._src_wh else "?"
        hud = (
            f"INTRINSICS {cur} {self.captured}/{self.target_frames} "
            f"(min {self.min_frames}) {'DETECTED' if self._last_found else 'NO BOARD'} "
            f"src={src} det<={self.detect_max_width} SPACE=capture ESC=done"
        )
        cv2.putText(grid, hud, (8, grid.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        return grid

    def _compose_extrinsics(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        tiles = []
        for d in DIRECTIONS:
            fr = frames.get(d)
            if fr is None:
                tiles.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))
                continue

            h0, w0 = fr.shape[:2]
            self._src_wh = (int(w0), int(h0))
            pipe = self._gpu_pipes.get(d)
            locked = d in self.homographies

            # 热路径全在 GPU：remap + 两次 resize，只下载小图
            if pipe is not None and pipe.use_cuda:
                gm = pipe.undistort_gpu(fr)
                tile = cv2.cuda.resize(gm, (TILE_W, TILE_H)).download()
                if self._dump_request:
                    self._do_dump(d, fr, gm.download())
                tile_undistorted = True
            else:
                # 无 CUDA：绝不在主循环做全分辨率 remap，直接显示原图
                tile = resize_bgr(fr, (TILE_W, TILE_H))
                tile_undistorted = False
            # 检测走鱼眼原图：去畸变会裁视场并拉伸边缘，实测检出率 3/4 → 1/4
            if not locked and self._should_submit(d):
                dw, dh, sc = self._detect_size(w0, h0)
                det_img = fr.copy() if (dw, dh) == (w0, h0) else resize_bgr(fr, (dw, dh))
                self._submit_detect(d, det_img, sc)

            cv2.putText(
                tile, d.upper(), (8, TILE_H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )

            if locked:
                cv2.rectangle(tile, (2, 2), (TILE_W - 3, TILE_H - 3), (0, 180, 0), 4)
                cv2.putText(tile, "LOCKED", (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                tiles.append(tile)
                continue

            if d not in self.calib_intr:
                cv2.putText(tile, "NO INTRINSICS", (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 128, 255), 2)
                tiles.append(tile)
                continue

            if self.sequential and self.target != d:
                # 非当前目标：压暗并明确标注，避免误以为它在参与标定
                tile = (tile * 0.45).astype(np.uint8)
                cv2.putText(tile, "WAITING", (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 2)
                tiles.append(tile)
                continue

            ok, corners, age, ms = self._det_snapshot(d)
            streak = self.stable_streak.get(d, 0)
            # 角点保留到下一次该路检测出结果为止，不再按 2s 硬性过期
            if ok and corners is not None:
                # corners 在鱼眼原图坐标系；tile 若是去畸变图需先映射再投影
                c_draw = (
                    self._raw_corners_to_undistorted(d, corners, w0, h0)
                    if tile_undistorted else corners
                )
                c_tile = project_corners_to_tile(c_draw, (w0, h0), (TILE_W, TILE_H))
                cv2.drawChessboardCorners(tile, self.pattern, c_tile, True)
                if streak >= self.stable_need:
                    cv2.rectangle(tile, (2, 2), (TILE_W - 3, TILE_H - 3), (0, 255, 0), 3)
                    cv2.putText(tile, f"READY {streak}", (12, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                else:
                    cv2.rectangle(tile, (2, 2), (TILE_W - 3, TILE_H - 3), (0, 200, 255), 2)
                    cv2.putText(tile, f"STABLE {streak}/{self.stable_need}", (12, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                stage_now = self._det_stage.get(d, "")
                if stage_now.startswith("oob"):
                    n = stage_now[3:]
                    cv2.rectangle(tile, (2, 2), (TILE_W - 3, TILE_H - 3), (0, 140, 255), 3)
                    cv2.putText(tile, f"NOT FULLY IN VIEW {n}", (12, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
                else:
                    cv2.putText(tile, "SCANNING", (12, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 2)
            stage = self._det_stage.get(d, "-")
            focus_tag = " FOCUS" if self._focus == d else ""
            cv2.putText(
                tile, f"det {ms:.0f}ms age {age:.1f}s {stage}{focus_tag}", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
            )
            tiles.append(tile)

        if self._dump_request:
            self._dump_request = False
            LOG.info(f"检测输入已转储到 {self._dump_dir}")

        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        locked = ",".join(sorted(self.homographies.keys())) or "none"
        src = f"{self._src_wh[0]}x{self._src_wh[1]}" if self._src_wh else "?"
        cols, rows = self.pattern
        # OpenCV putText 无中文字形 -> HUD 只用 ASCII
        mode = "AUTOLOCK" if self.auto_lock else "SPACE"
        algo = "SB" if self.detect_use_sb else "classic"
        seq = f"TARGET={(self.target or '-').upper()} " if self.sequential else ""
        hud = (
            f"EXTRINSICS {seq}locked=[{locked}] src={src} "
            f"raw{self.detect_max_width}/{self.detect_interval_ms}ms "
            f"board={cols}x{rows} need={self.stable_need} {algo} {mode}"
        )
        cv2.putText(
            grid, hud, (8, grid.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1,
        )
        return grid

    def action(self, cmd: str) -> dict[str, Any]:
        cmd = (cmd or "").strip().lower()
        if self.kind == "intrinsics":
            return self._action_intrinsics(cmd)
        if self.kind == "extrinsics":
            return self._action_extrinsics(cmd)
        return {"ok": False, "error": "no calib session"}

    def _action_intrinsics(self, cmd: str) -> dict[str, Any]:
        from avm.calibrate_intrinsics import (
            calibrate_camera,
            evaluate_intrinsics,
            fit_inverse_polynomial,
            print_evaluation_report,
            save_results,
            save_undistorted_sample,
        )

        if not self.dir_list:
            return {"ok": False, "error": "无相机"}
        cur = self.dir_list[self.dir_idx]

        if cmd in ("space", "capture"):
            if self.cooldown > 0:
                return {"ok": False, "error": "cooldown"}
            self.hub.pause_grab(True)
            time.sleep(0.02)
            try:
                frames = self.hub._grab()
            finally:
                self.hub.pause_grab(False)
            fr = frames.get(cur)
            if fr is None:
                return {"ok": False, "error": "no frame"}
            found, corners, used_s = self._detect(fr)
            if not found or corners is None:
                self.message = f"{cur}: 未检出棋盘（hires≤{self.detect_max_width}），未抓拍"
                LOG.warn(self.message)
                return {"ok": False, "error": "no board"}
            self.obj_points.append(self._objp.copy())
            self.img_points.append(corners)
            img_dir = ROOT / "calib_images" / cur
            img_dir.mkdir(parents=True, exist_ok=True)
            path = img_dir / f"{self.captured:04d}.jpg"
            cv2.imwrite(str(path), fr)
            self.captured += 1
            self.cooldown = self.cooldown_set
            self.message = (
                f"{cur}: 已抓 {self.captured}/{self.target_frames} "
                f"det@{used_s:.2f} → {path.name}"
            )
            LOG.info(self.message)
            if self.captured >= self.target_frames:
                return self._finish_intrinsics_direction()
            return {"ok": True, "captured": self.captured, "message": self.message}

        if cmd in ("esc", "done", "finish", "quit"):
            if self.captured < self.min_frames:
                self.message = f"{cur}: 至少 {self.min_frames} 张，当前 {self.captured}"
                return {"ok": False, "error": self.message}
            return self._finish_intrinsics_direction()

        if cmd in ("next", "skip"):
            self.message = f"跳过 {cur}"
            self.dir_idx += 1
            if self.dir_idx >= len(self.dir_list):
                self.message = "内参流程结束"
                return {"ok": True, "done": True, "message": self.message}
            self._reset_intr_dir()
            self.message = f"内参：{self.dir_list[self.dir_idx]} SPACE 抓拍"
            return {"ok": True, "message": self.message}

        return {"ok": False, "error": f"unknown cmd {cmd}"}

    def _finish_intrinsics_direction(self) -> dict[str, Any]:
        from avm.calibrate_intrinsics import (
            calibrate_camera,
            evaluate_intrinsics,
            fit_inverse_polynomial,
            print_evaluation_report,
            save_results,
            save_undistorted_sample,
        )

        cur = self.dir_list[self.dir_idx]
        frames = self.hub._grab()
        fr = frames.get(cur)
        h, w = (fr.shape[:2] if fr is not None else (1536, 1920))
        LOG.info(f"标定内参 {cur} frames={self.captured}")
        try:
            K, D, rms, rvecs, tvecs = calibrate_camera(
                self.obj_points, self.img_points, (w, h)
            )
            D_inv = fit_inverse_polynomial(D)
            report = evaluate_intrinsics(
                K, D, rms, (w, h), self.obj_points, self.img_points, rvecs, tvecs
            )
            print_evaluation_report(report, cur)
            save_results(cur, K, D, D_inv, rms, str(ROOT / "calib_results"), evaluation=report)
            if fr is not None:
                save_undistorted_sample(cur, fr, K, D, str(ROOT / "calib_results"))
            self.message = f"{cur} 内参完成 RMS={rms:.3f}"
            LOG.info(self.message)
        except Exception as exc:
            self.message = f"{cur} 标定失败: {exc}"
            LOG.error(self.message)
            return {"ok": False, "error": str(exc)}

        self.dir_idx += 1
        if self.dir_idx >= len(self.dir_list):
            self.message = "全部内参完成"
            return {"ok": True, "done": True, "message": self.message}
        self._reset_intr_dir()
        self.message = f"内参：{self.dir_list[self.dir_idx]} SPACE 抓拍"
        return {"ok": True, "message": self.message, "next": self.dir_list[self.dir_idx]}

    def _pending_dirs(self) -> list[str]:
        """还没锁定、且具备内参与 maps 的方向，按固定顺序。"""
        return [
            d for d in DIRECTIONS
            if d in self.maps and d not in self.homographies
        ]

    def set_target(self, d: Optional[str]) -> Optional[str]:
        """切换当前标定目标，并清掉旧目标的检测残留。"""
        if d is not None and d not in self.maps:
            return self.target
        with self._det_lock:
            self._focus = None
            self._det_sticky = None
            self._det_jobs.clear()
            for k in DIRECTIONS:
                self.stable_streak[k] = 0
                self._miss_streak[k] = 0
                self._det_result[k] = (False, None, 0.0)
            self.target = d
        if d:
            LOG.info(f"外参目标切换 -> {d}")
        return self.target

    def advance_target(self, *, step: int = 1) -> Optional[str]:
        """在未锁定的方向里前进/后退；没有剩余则置空。"""
        pend = self._pending_dirs()
        if not pend:
            return self.set_target(None)
        if self.target in pend:
            idx = (pend.index(self.target) + step) % len(pend)
        else:
            idx = 0 if step >= 0 else len(pend) - 1
        return self.set_target(pend[idx])

    def _lock_direction(self, d: str) -> bool:
        """连拍 + 求 H + 落库。SPACE 与自动锁定共用。"""
        from avm.calibrate_extrinsics import (
            calibrate_one_burst,
            print_homography_qc,
            quality,
        )

        if self._lock_busy or d in self.homographies or d not in self.maps:
            return False
        self._lock_busy = True
        try:
            LOG.info(f"extrinsics burst lock {d} (auto={self.auto_lock})")
            frames = self._burst_from_hub_cap(d, self.burst_n)
            if not frames:
                LOG.warn(f"{d}: hub burst empty")
                return False
            dbg = str(ROOT / "calib_results" / d)
            H, rms, qc = calibrate_one_burst(
                d,
                frames,
                self.calib_intr[d]["K"],
                self.calib_intr[d]["D"],
                self.maps[d][0],
                self.maps[d][1],
                self.placements[d],
                self.pattern[0],
                self.pattern[1],
                self.square,
                self.scale,
                self.canvas,
                dbg,
                min_ok=self.burst_min_ok,
                balance=self.extrinsic_balance,
            )
            with self._det_lock:
                self.stable_streak[d] = 0
                self._miss_streak[d] = 0
                if self._focus == d:
                    self._focus = None
                    self._det_sticky = None
            if H is None:
                LOG.warn(f"{d} burst failed")
                self.message = f"{d} 连拍求 H 失败，继续检测"
                return False
            # RMS 小 ≠ BEV 可用：H 病态时绝不能锁进文件，否则拼接变黑/拉丝
            if qc is not None and qc.get("status") == "bad":
                warns = "; ".join(qc.get("warnings") or []) or "H 病态"
                print(f"  {d:8s}  ❌ 拒绝锁定  RMS={rms:.4f} 但 H-QC=bad")
                print_homography_qc(d, qc)
                LOG.warn(f"{d} 拒绝锁定: {warns}")
                self.message = (
                    f"{d} 拒绝锁定（H 病态）：{warns}。"
                    f"请检查 near_m/板是否贴地/板是否整块在该路视野中心，然后重试"
                )
                # 清 streak，避免自动锁定立刻再打一轮
                with self._det_lock:
                    self.stable_streak[d] = 0
                    self._miss_streak[d] = 0
                return False
            self.homographies[d] = H
            self.rms_errors[d] = rms
            self.h_quality[d] = qc
            print(f"  {d:8s}  已锁定  RMS={rms:.4f} px  {quality(rms)}")
            print_homography_qc(d, qc)
            LOG.info(f"{d} locked RMS={rms:.4f} qc={qc.get('status') if qc else '-'}")
            self._autosave_draft()
            if self.sequential:
                nxt = self.advance_target()
                self.message = (
                    f"{d} 已锁定 RMS={rms:.3f} qc={qc.get('status') if qc else '-'}，下一路：{nxt}"
                    if nxt else
                    f"{d} 已锁定 RMS={rms:.3f}，四路全部完成，按 ESC 保存"
                )
            else:
                self.message = f"{d} 已自动锁定 RMS={rms:.3f}"
            LOG.info(self.message)
            return True
        finally:
            self._lock_busy = False

    def _action_extrinsics(self, cmd: str) -> dict[str, Any]:
        from avm.calibrate_extrinsics import (
            _merge_previous_extrinsics,
            check_lr_symmetry,
            save_results,
        )

        if cmd in ("next", "skip"):
            nxt = self.advance_target(step=1)
            self.message = f"切到 {nxt}" if nxt else "没有待标定的方向了"
            return {"ok": True, "target": nxt, "message": self.message}

        if cmd == "prev":
            nxt = self.advance_target(step=-1)
            self.message = f"切到 {nxt}" if nxt else "没有待标定的方向了"
            return {"ok": True, "target": nxt, "message": self.message}

        if cmd.startswith("target:"):
            want = cmd.split(":", 1)[1].strip()
            if want not in self.maps:
                return {"ok": False, "error": f"{want} 不可用"}
            self.set_target(want)
            self.message = f"当前标定 {want}"
            return {"ok": True, "target": want, "message": self.message}

        if cmd in ("relock", "redo"):
            d = self.target
            if not d:
                return {"ok": False, "error": "无当前目标"}
            self.homographies.pop(d, None)
            self.rms_errors.pop(d, None)
            self.h_quality.pop(d, None)
            self.set_target(d)
            self.message = f"{d} 已解锁，重新检测"
            return {"ok": True, "target": d, "message": self.message}

        if cmd in ("space", "capture", "lock"):
            if self._lock_busy:
                return {"ok": False, "error": "busy"}
            # 顺序模式只锁当前目标，避免别路的误检也被一起锁进去
            cands = (
                [self.target] if (self.sequential and self.target) else list(DIRECTIONS)
            )
            ready = [
                d for d in cands
                if d in self.maps
                and d not in self.homographies
                and self.stable_streak.get(d, 0) >= self.stable_need
            ]
            if not ready:
                cur = self.target or "-"
                got = self.stable_streak.get(cur, 0)
                self.message = f"{cur} 未 READY（{got}/{self.stable_need}）"
                return {"ok": False, "error": self.message}
            for d in ready:
                self._lock_direction(d)
            locked = sorted(self.homographies.keys())
            self.message = f"已锁定 {locked or '无'}"
            return {"ok": True, "locked": locked, "message": self.message}

        if cmd in ("esc", "done", "finish", "quit"):
            if not self.homographies:
                return {"ok": False, "error": "尚未锁定任何一路"}
            params = {
                "pattern_size": list(self.pattern),
                "square_size_m": self.square,
                "scale": self.scale,
                "canvas": self.canvas,
                "balance": self.extrinsic_balance,
                "extrinsic_balance": self.extrinsic_balance,
            }
            out = str(ROOT / "calib_results" / "extrinsics.json")
            new = sorted(self.homographies.keys())
            bad = [
                d for d, q in self.h_quality.items()
                if d in self.homographies and (q or {}).get("status") == "bad"
            ]
            if bad:
                self.message = (
                    f"拒绝保存：{bad} 的 H 病态，写入会把 BEV 弄坏。"
                    f"请用「重标当前路」修好后再 ESC"
                )
                return {"ok": False, "error": self.message, "bad": bad}
            # 只标了几路时，其余路沿用上次结果，不要被清空成黑图。
            # 先在副本上合并，才能准确知道到底续用了哪几路。
            homs = dict(self.homographies)
            rms = dict(self.rms_errors)
            places = dict(self.placements)
            qc = dict(self.h_quality)
            kept, stale = _merge_previous_extrinsics(out, params, homs, rms, places, qc)
            warnings = check_lr_symmetry(homs, qc)
            save_results(
                homs, rms, places, params, out,
                h_quality=qc, global_warnings=warnings, overwrite=True,
            )
            self.message = f"外参已保存：本次更新 {new}"
            if kept:
                self.message += f"，沿用上次 {kept}"
            elif stale:
                self.message += f"（旧外参未续用：{stale}）"
            LOG.info(f"{self.message} -> {out}")
            return {
                "ok": True, "done": True, "path": out,
                "updated": new, "kept": kept, "message": self.message,
            }

        if cmd in ("0", "unlock_all"):
            self.homographies.clear()
            self.rms_errors.clear()
            self.h_quality.clear()
            with self._det_lock:
                self.stable_streak = {d: 0 for d in DIRECTIONS}
                self._miss_streak = {d: 0 for d in DIRECTIONS}
                self._det_result.clear()
                self._focus = None
                self._det_sticky = None
            self.message = "已解锁全部"
            return {"ok": True, "message": self.message}

        return {"ok": False, "error": f"unknown cmd {cmd}"}

    def _video_index(self, d: str) -> Optional[int]:
        cfg = self.hub.config_path
        import json
        data = json.loads(Path(cfg).read_text(encoding="utf-8"))
        v = data.get(d)
        return int(v) if v is not None else None

    def _burst_from_hub_cap(self, d: str, n: int) -> list:
        """从 hub 已打开的 cap 连读 n 帧（暂停预览循环避免抢读）。"""
        self.hub.pause_grab(True)
        time.sleep(0.05)
        try:
            with self.hub._lock:
                cap = self.hub._caps.get(d)
            if cap is None:
                return []
            frames = []
            for _ in range(3):
                cap.grab()
            for _ in range(n):
                ok, fr = cap.read()
                if ok and fr is not None:
                    frames.append(fr)
            return frames
        finally:
            self.hub.pause_grab(False)

    def _autosave_draft(self) -> None:
        from avm.calibrate_extrinsics import check_lr_symmetry, save_results

        if not self.homographies:
            return
        warnings = check_lr_symmetry(self.homographies, self.h_quality)
        params = {
            "pattern_size": list(self.pattern),
            "square_size_m": self.square,
            "scale": self.scale,
            "canvas": self.canvas,
            "balance": self.extrinsic_balance,
            "extrinsic_balance": self.extrinsic_balance,
        }
        save_results(
            self.homographies,
            self.rms_errors,
            self.placements,
            params,
            str(ROOT / "calib_results" / "extrinsics_draft.json"),
            h_quality=self.h_quality,
            global_warnings=warnings,
            overwrite=True,
        )
