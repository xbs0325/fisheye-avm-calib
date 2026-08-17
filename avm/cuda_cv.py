#!/usr/bin/env python3
"""
CUDA OpenCV helpers for AVM (Jetson / OpenCV 4.14+ cudawarping).

Policy (calibration accuracy first):
  - findChessboardCorners / cornerSubPix / fisheye.calibrate / findHomography
    stay on CPU.
  - Final extrinsic/intrinsic remap used to *compute* H or K stays on CPU
    with CV_16SC2 maps (bit-stable with historical results).
  - Live preview / BEV stitch hot path: GPU remap + warpPerspective (+ optional
    weighted blend). CUDA remap requires CV_32FC1 x/y maps.

Usage:
  source scripts/env_opencv_cuda.sh   # before python
  from cuda_cv import cuda_available, UndistortWarpPipeline, ...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional: prepend side-install site-packages if env was sourced incompletely.
# LD_LIBRARY_PATH still needs to be set before process start for .so deps;
# prefer: source scripts/env_opencv_cuda.sh
# ---------------------------------------------------------------------------


def _candidate_prefixes() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("OPENCV_CUDA_PREFIX", "").strip()
    if env:
        out.append(Path(env))
    out.append(Path.home() / ".local" / "opencv-4.14.0-cuda")
    out.append(Path("/usr/local/opencv-4.14.0-cuda"))
    local = Path.home() / ".local"
    if local.is_dir():
        out.extend(sorted(local.glob("opencv-*-cuda"), reverse=True))
    # dedupe preserving order
    seen = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve() if p.exists() else p
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _cv2_python_dirs(prefix: Path) -> list[Path]:
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return [
        prefix / "lib" / py / "dist-packages",
        prefix / "lib" / py / "site-packages",
    ]


def bootstrap_opencv_cuda() -> Optional[Path]:
    """Insert CUDA OpenCV site/dist-packages early. Returns prefix or None."""
    for prefix in _candidate_prefixes():
        site = next(
            (p for p in _cv2_python_dirs(prefix) if p.is_dir() and any(p.glob("cv2*"))),
            None,
        )
        if site is None:
            continue
        lib = prefix / "lib"
        os.environ.setdefault("OPENCV_CUDA_PREFIX", str(prefix))
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        lib_s = str(lib)
        if lib_s not in ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{lib_s}:{ld}" if ld else lib_s
        site_s = str(site)
        if site_s not in sys.path:
            sys.path.insert(0, site_s)
        return prefix
    return None


bootstrap_opencv_cuda()

import cv2  # noqa: E402  (after path bootstrap)


def cuda_available() -> bool:
    try:
        return int(cv2.cuda.getCudaEnabledDeviceCount()) >= 1
    except Exception:
        return False


_CUDA_OK = cuda_available()
_WARNED = False


def cuda_status_line() -> str:
    ver = getattr(cv2, "__version__", "?")
    loc = getattr(cv2, "__file__", "?")
    if _CUDA_OK:
        return (
            f"OpenCV {ver} CUDA=ON "
            f"devices={cv2.cuda.getCudaEnabledDeviceCount()} ({loc})"
        )
    return (
        f"OpenCV {ver} CUDA=OFF ({loc}); live path falls back to CPU. "
        "Fix: source scripts/env_opencv_cuda.sh"
    )


def log_cuda_status(prefix: str = "  ") -> bool:
    global _WARNED
    print(f"{prefix}[cuda] {cuda_status_line()}")
    if not _CUDA_OK and not _WARNED:
        _WARNED = True
    return _CUDA_OK


def init_undistort_maps(
    K,
    D,
    w: int,
    h: int,
    balance: float,
    *,
    for_cuda: bool = False,
):
    """Build fisheye undistort maps.

    for_cuda=True  -> CV_32FC1 x/y (required by cv2.cuda.remap)
    for_cuda=False -> CV_16SC2 (CPU remap, calibration-stable)
    """
    new_K = np.asarray(K, dtype=np.float64).copy()
    new_K[0, 0] *= balance
    new_K[1, 1] *= balance
    new_K[0, 2] = w / 2.0
    new_K[1, 2] = h / 2.0
    mtype = cv2.CV_32FC1 if for_cuda else cv2.CV_16SC2
    return cv2.fisheye.initUndistortRectifyMap(
        np.asarray(K, dtype=np.float64),
        np.asarray(D, dtype=np.float64),
        np.eye(3, dtype=np.float64),
        new_K,
        (int(w), int(h)),
        mtype,
    )


def undistort_new_K(K, w: int, h: int, balance: float) -> np.ndarray:
    """与 init_undistort_maps 完全一致的 new_K，供角点坐标映射复用。"""
    new_K = np.asarray(K, dtype=np.float64).copy()
    new_K[0, 0] *= balance
    new_K[1, 1] *= balance
    new_K[0, 2] = w / 2.0
    new_K[1, 2] = h / 2.0
    return new_K


def undistort_points_fisheye(
    pts: np.ndarray,
    K,
    D,
    w: int,
    h: int,
    balance: float,
) -> np.ndarray:
    """把鱼眼原图上的点映射到去畸变图坐标系。

    用途：棋盘检测在原图上做（畸变小、无重采样损失、检出距离更远），
    但求 H 需要去畸变坐标 —— 直接映射点比重采样整张图更准也更快。
    输入/输出均为 (N,1,2) 或 (N,2) float32。
    """
    shape = pts.shape
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    new_K = undistort_new_K(K, w, h, balance)
    out = cv2.fisheye.undistortPoints(
        p,
        np.asarray(K, dtype=np.float64),
        np.asarray(D, dtype=np.float64),
        R=np.eye(3),
        P=new_K,
    )
    return out.astype(np.float32).reshape(shape)


def _upload(arr: np.ndarray) -> "cv2.cuda_GpuMat":
    g = cv2.cuda_GpuMat()
    g.upload(np.ascontiguousarray(arr))
    return g


def remap_bgr(
    image: np.ndarray,
    map1,
    map2,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Remap BGR; CUDA when device OK and maps are float32 (or GpuMat)."""
    if _CUDA_OK:
        is_gpu_maps = isinstance(map1, cv2.cuda_GpuMat)
        is_f32 = (
            isinstance(map1, np.ndarray)
            and map1.dtype == np.float32
            and isinstance(map2, np.ndarray)
            and map2.dtype == np.float32
        )
        if is_gpu_maps or is_f32:
            src = _upload(image)
            xmap = map1 if is_gpu_maps else _upload(map1)
            ymap = map2 if is_gpu_maps else _upload(map2)
            return cv2.cuda.remap(src, xmap, ymap, interpolation).download()
    return cv2.remap(image, map1, map2, interpolation)


def resize_bgr(
    image: np.ndarray,
    size: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    w, h = int(size[0]), int(size[1])
    if image is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    if image.shape[1] == w and image.shape[0] == h:
        return image
    if _CUDA_OK and image.size >= 320 * 240 * 3:
        out = cv2.cuda.resize(_upload(image), (w, h), interpolation=interpolation)
        return out.download()
    return cv2.resize(image, (w, h), interpolation=interpolation)


def warp_perspective_bgr(
    image: np.ndarray,
    H: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    if _CUDA_OK:
        src = _upload(image)
        M = np.asarray(H, dtype=np.float32)
        out = cv2.cuda.warpPerspective(
            src,
            M,
            (int(dsize[0]), int(dsize[1])),
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return out.download()
    return cv2.warpPerspective(
        image,
        H,
        dsize,
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def cvt_gray(image: np.ndarray) -> np.ndarray:
    if _CUDA_OK and image.ndim == 3 and image.size >= 640 * 480 * 3:
        return cv2.cuda.cvtColor(_upload(image), cv2.COLOR_BGR2GRAY).download()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class UndistortWarpPipeline:
    """Per-camera cached GPU maps: upload -> remap -> warp (-> weight)."""

    def __init__(
        self,
        map1: np.ndarray,
        map2: np.ndarray,
        H: Optional[np.ndarray] = None,
        canvas_size: Optional[tuple[int, int]] = None,
        weight: Optional[np.ndarray] = None,
    ):
        self.use_cuda = bool(_CUDA_OK)
        self.canvas_size = (
            None
            if canvas_size is None
            else (int(canvas_size[0]), int(canvas_size[1]))
        )
        self.H = None if H is None else np.asarray(H, dtype=np.float32)
        self._map1_cpu = map1
        self._map2_cpu = map2
        self._weight_cpu = None if weight is None else np.asarray(weight, dtype=np.float32)
        self._src = None
        self._xmap = None
        self._ymap = None
        self._weight = None
        if self.use_cuda:
            self._src = cv2.cuda_GpuMat()
            self._xmap = _upload(np.ascontiguousarray(map1, dtype=np.float32))
            self._ymap = _upload(np.ascontiguousarray(map2, dtype=np.float32))
            if weight is not None:
                self.set_weight(weight)

    def set_homography(self, H: np.ndarray, canvas_size: tuple[int, int]) -> None:
        self.H = np.asarray(H, dtype=np.float32)
        self.canvas_size = (int(canvas_size[0]), int(canvas_size[1]))

    def set_weight(self, weight: np.ndarray) -> None:
        w = np.asarray(weight, dtype=np.float32)
        self._weight_cpu = w
        if not self.use_cuda:
            return
        if w.ndim == 2:
            w3 = np.stack([w, w, w], axis=-1)
        else:
            w3 = w
        self._weight = _upload(np.ascontiguousarray(w3))

    def undistort(self, image: np.ndarray) -> np.ndarray:
        if not self.use_cuda:
            return cv2.remap(
                image, self._map1_cpu, self._map2_cpu, cv2.INTER_LINEAR
            )
        self._src.upload(np.ascontiguousarray(image))
        return cv2.cuda.remap(
            self._src, self._xmap, self._ymap, cv2.INTER_LINEAR
        ).download()

    def undistort_warp(self, image: np.ndarray) -> np.ndarray:
        if self.H is None or self.canvas_size is None:
            return self.undistort(image)
        if not self.use_cuda:
            und = cv2.remap(
                image, self._map1_cpu, self._map2_cpu, cv2.INTER_LINEAR
            )
            return cv2.warpPerspective(
                und,
                self.H,
                self.canvas_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        self._src.upload(np.ascontiguousarray(image))
        und = cv2.cuda.remap(self._src, self._xmap, self._ymap, cv2.INTER_LINEAR)
        bev = cv2.cuda.warpPerspective(
            und,
            self.H,
            self.canvas_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return bev.download()

    def undistort_gpu(self, image: np.ndarray) -> "cv2.cuda_GpuMat":
        """Remap only, staying on GPU (no download)."""
        if not self.use_cuda:
            raise RuntimeError("undistort_gpu requires CUDA")
        self._src.upload(np.ascontiguousarray(image))
        return cv2.cuda.remap(self._src, self._xmap, self._ymap, cv2.INTER_LINEAR)

    def undistort_warp_gpu(self, image: np.ndarray) -> "cv2.cuda_GpuMat":
        """Remap+warp staying on GPU (no download)."""
        if not self.use_cuda:
            raise RuntimeError("undistort_warp_gpu requires CUDA")
        if self.H is None or self.canvas_size is None:
            raise RuntimeError("homography/canvas not set")
        self._src.upload(np.ascontiguousarray(image))
        und = cv2.cuda.remap(self._src, self._xmap, self._ymap, cv2.INTER_LINEAR)
        return cv2.cuda.warpPerspective(
            und,
            self.H,
            self.canvas_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def weighted_contrib_gpu(self, image: np.ndarray) -> "cv2.cuda_GpuMat":
        """remap -> warp -> float32 * weight (GPU)."""
        if self._weight is None:
            raise RuntimeError("weight not set")
        bev = self.undistort_warp_gpu(image)
        bf = bev.convertTo(cv2.CV_32F)
        return cv2.cuda.multiply(bf, self._weight)


def process_frames_to_bev(
    frames: dict[str, np.ndarray],
    pipelines: dict[str, UndistortWarpPipeline],
    order: list[str],
    weights: Optional[dict[str, np.ndarray]] = None,
    gains: Optional[dict[str, np.ndarray]] = None,
    canvas_size: Optional[tuple[int, int]] = None,
    *,
    need_bev_views: bool = False,
) -> tuple[np.ndarray, dict[str, Optional[np.ndarray]], float]:
    """Hot-path BEV: GPU remap+warp(+blend) with CPU fallback.

    If need_bev_views or gains differ from 1, downloads per-camera BEV tiles
    (still warped on GPU). Otherwise accumulates weighted blend on GPU only.
    """
    import time

    t0 = time.perf_counter()
    bev_views: dict[str, Optional[np.ndarray]] = {d: None for d in order}

    use_gpu_blend = (
        _CUDA_OK
        and not need_bev_views
        and all(
            d not in frames
            or (
                d in pipelines
                and pipelines[d].use_cuda
                and pipelines[d]._weight is not None
                and pipelines[d].H is not None
            )
            for d in order
            if frames.get(d) is not None
        )
        and (
            gains is None
            or all(
                d not in gains or not np.any(np.abs(gains[d] - 1.0) > 0.003)
                for d in order
            )
        )
    )

    if use_gpu_blend:
        acc = None
        for d in order:
            frame = frames.get(d)
            if frame is None or d not in pipelines:
                continue
            prod = pipelines[d].weighted_contrib_gpu(frame)
            acc = prod if acc is None else cv2.cuda.add(acc, prod)
        if acc is None:
            cw, ch = canvas_size or (1000, 1000)
            result = np.zeros((ch, cw, 3), dtype=np.uint8)
        else:
            result = np.clip(acc.download(), 0, 255).astype(np.uint8)
        return result, bev_views, time.perf_counter() - t0

    # Per-tile GPU warp (or CPU), then CPU weighted blend / gain
    cw, ch = canvas_size or (1000, 1000)
    for d in order:
        frame = frames.get(d)
        if frame is None or d not in pipelines:
            continue
        pipe = pipelines[d]
        if frame.shape[1] != pipe._map1_cpu.shape[1] or frame.shape[0] != pipe._map1_cpu.shape[0]:
            frame = resize_bgr(
                frame, (pipe._map1_cpu.shape[1], pipe._map1_cpu.shape[0])
            )
        bev = pipe.undistort_warp(frame)
        if gains is not None and d in gains:
            g = gains[d]
            if np.any(np.abs(g - 1.0) > 0.003):
                bev = np.clip(bev.astype(np.float32) * g, 0, 255).astype(np.uint8)
        bev_views[d] = bev

    canvas = np.zeros((ch, cw, 3), dtype=np.float32)
    for d in order:
        bev = bev_views.get(d)
        if bev is None:
            continue
        w = None
        if d in pipelines and pipelines[d]._weight_cpu is not None:
            w = pipelines[d]._weight_cpu
        elif weights is not None and d in weights:
            w = weights[d]
        if w is None:
            continue
        if w.ndim == 2:
            canvas += bev.astype(np.float32) * w[:, :, np.newaxis]
        else:
            canvas += bev.astype(np.float32) * w
    result = np.clip(canvas, 0, 255).astype(np.uint8)
    return result, bev_views, time.perf_counter() - t0
