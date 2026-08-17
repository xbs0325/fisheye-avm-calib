#!/usr/bin/env python3
"""Camera profile + unified open/probe helpers for multi-resolution AVM."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "config" / "camera_profile.json"
CAMERA_PATH = ROOT / "config" / "camera_config.json"
DIRECTIONS = ("front", "back", "left", "right")

_DEFAULT_PROFILE: dict[str, Any] = {
    "_说明": "四路相机采集配置。width/height 为请求分辨率；实际以驱动返回为准。",
    "width": 1920,
    "height": 1536,
    "fourcc": "YUYV",
    "backend": "v4l2",
    "gst_pipeline_template": "",
    "cameras": {
        "front": {"device": 0},
        "back": {"device": 2},
        "left": {"device": 3},
        "right": {"device": 1},
    },
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


def _device_to_index(device: Any) -> int:
    if isinstance(device, int):
        return int(device)
    s = str(device).strip()
    if s.startswith("/dev/video"):
        return int(s.replace("/dev/video", ""))
    return int(s)


def _legacy_camera_map() -> dict[str, int]:
    if not CAMERA_PATH.is_file():
        return {}
    raw = _read_json(CAMERA_PATH)
    out: dict[str, int] = {}
    for d in DIRECTIONS:
        if d in raw:
            out[d] = int(raw[d])
    return out


def load_camera_profile(path: Optional[Path] = None) -> dict[str, Any]:
    """Load profile; merge legacy camera_config.json device indices if needed."""
    p = Path(path) if path else PROFILE_PATH
    out = json.loads(json.dumps(_DEFAULT_PROFILE))  # deep copy
    if p.is_file():
        raw = _read_json(p)
        for k, v in raw.items():
            if k == "cameras" and isinstance(v, dict):
                cams = out.setdefault("cameras", {})
                for d, cfg in v.items():
                    if d not in DIRECTIONS:
                        continue
                    if isinstance(cfg, dict):
                        cams[d] = {**(cams.get(d) or {}), **cfg}
                    else:
                        cams[d] = {"device": cfg}
            elif not str(k).startswith("_"):
                out[k] = v
    legacy = _legacy_camera_map()
    cams = out.setdefault("cameras", {})
    for d, idx in legacy.items():
        cams.setdefault(d, {})
        if "device" not in cams[d]:
            cams[d]["device"] = idx
    out["width"] = int(out.get("width", 1920))
    out["height"] = int(out.get("height", 1536))
    out["fourcc"] = str(out.get("fourcc") or "YUYV").upper()
    out["backend"] = str(out.get("backend") or "v4l2").lower()
    return out


def save_camera_profile(profile: dict[str, Any], path: Optional[Path] = None) -> dict[str, Any]:
    """Normalize and write profile; also sync device indices to camera_config.json."""
    p = Path(path) if path else PROFILE_PATH
    width = int(profile.get("width", 1920))
    height = int(profile.get("height", 1536))
    if width < 160 or height < 120:
        raise ValueError("width/height too small")
    fourcc = str(profile.get("fourcc") or "YUYV").upper()
    backend = str(profile.get("backend") or "v4l2").lower()
    if backend not in ("v4l2", "gstreamer"):
        raise ValueError("backend must be v4l2 or gstreamer")
    cams_in = profile.get("cameras") or {}
    cams: dict[str, Any] = {}
    cam_map: dict[str, int] = {}
    for d in DIRECTIONS:
        c = cams_in.get(d) or {}
        if not isinstance(c, dict):
            c = {"device": c}
        if "device" not in c:
            raise ValueError(f"cameras.{d}.device required")
        idx = _device_to_index(c["device"])
        entry: dict[str, Any] = {"device": idx}
        if c.get("width") is not None:
            entry["width"] = int(c["width"])
        if c.get("height") is not None:
            entry["height"] = int(c["height"])
        cams[d] = entry
        cam_map[d] = idx
    data = {
        "_说明": "四路相机采集配置。width/height 为请求分辨率；实际以驱动返回为准。",
        "width": width,
        "height": height,
        "fourcc": fourcc,
        "backend": backend,
        "gst_pipeline_template": str(profile.get("gst_pipeline_template") or ""),
        "cameras": cams,
    }
    _write_json(p, data)
    _write_json(CAMERA_PATH, cam_map)
    return load_camera_profile(p)


def capture_size(profile: Optional[dict[str, Any]] = None) -> tuple[int, int]:
    prof = profile or load_camera_profile()
    return int(prof["width"]), int(prof["height"])


def direction_size(
    direction: str, profile: Optional[dict[str, Any]] = None
) -> tuple[int, int]:
    prof = profile or load_camera_profile()
    cam = (prof.get("cameras") or {}).get(direction) or {}
    w = int(cam.get("width") or prof["width"])
    h = int(cam.get("height") or prof["height"])
    return w, h


def direction_device(
    direction: str, profile: Optional[dict[str, Any]] = None
) -> int:
    prof = profile or load_camera_profile()
    cam = (prof.get("cameras") or {}).get(direction) or {}
    if "device" not in cam:
        raise KeyError(f"no device for {direction}")
    return _device_to_index(cam["device"])


def _fourcc_code(fourcc: str) -> int:
    s = (fourcc or "YUYV").upper()
    if len(s) != 4:
        s = "YUYV"
    return cv2.VideoWriter_fourcc(*s)


def _gst_pipeline(
    index: int,
    width: int,
    height: int,
    fourcc: str,
    template: str,
    with_videoconvert: bool,
) -> str:
    device = f"/dev/video{index}"
    if template.strip():
        return (
            template.replace("{device}", device)
            .replace("{width}", str(width))
            .replace("{height}", str(height))
            .replace("{fourcc}", fourcc)
        )
    fmt = "YUY2" if fourcc in ("YUYV", "YUY2") else fourcc
    head = (
        f"v4l2src device={device} io-mode=2 do-timestamp=true ! "
        f"video/x-raw,format={fmt},width={width},height={height} ! "
        f"nvvidconv ! video/x-raw,format=BGRx,width={width},height={height}"
    )
    if with_videoconvert:
        return (
            f"{head} ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
    return f"{head} ! appsink drop=true max-buffers=1 sync=false"


def _normalize_bgr(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if frame is None:
        return None
    if frame.ndim == 3 and frame.shape[2] == 4:
        return frame[:, :, :3].copy()
    return frame


def open_camera_index(
    index: int,
    *,
    width: int,
    height: int,
    fourcc: str = "YUYV",
    backend: str = "v4l2",
    gst_template: str = "",
    timeout_s: float = 8.0,
) -> tuple[Any, int, int, str]:
    """Open one camera. Returns (cap, actual_w, actual_h, backend_used)."""
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            if backend == "gstreamer":
                last_err = None
                for use_vc in (False, True):
                    pipe = _gst_pipeline(
                        index, width, height, fourcc, gst_template, use_vc
                    )
                    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
                    if not cap.isOpened():
                        last_err = RuntimeError("gstreamer open failed")
                        continue
                    ok, frame = cap.read()
                    frame = _normalize_bgr(frame)
                    if ok and frame is not None and frame.size > 0:
                        for _ in range(3):
                            cap.grab()
                        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
                        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
                        if aw <= 0:
                            aw = int(frame.shape[1])
                        if ah <= 0:
                            ah = int(frame.shape[0])
                        box["cap"] = cap
                        box["wh"] = (aw, ah)
                        box["backend"] = "gst+vc" if use_vc else "gst"
                        return
                    cap.release()
                    last_err = RuntimeError("gstreamer read empty")
                # fall through to v4l2
                if last_err is not None:
                    pass

            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                raise RuntimeError(f"cannot open /dev/video{index}")
            cap.set(cv2.CAP_PROP_FOURCC, _fourcc_code(fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(3):
                cap.grab()
            ok, frame = cap.read()
            frame = _normalize_bgr(frame)
            if not ok or frame is None:
                # still return cap; probe may retry grab
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
            else:
                aw = int(frame.shape[1])
                ah = int(frame.shape[0])
            box["cap"] = cap
            box["wh"] = (aw, ah)
            box["backend"] = "v4l2"
        except Exception as exc:  # noqa: BLE001
            box["err"] = exc

    t = threading.Thread(target=_run, name=f"open-cam-{index}", daemon=True)
    t.start()
    t.join(timeout=float(timeout_s))
    if t.is_alive():
        raise TimeoutError(
            f"/dev/video{index} open timeout ({timeout_s:.0f}s)"
        )
    if "err" in box:
        raise box["err"]
    return box["cap"], box["wh"][0], box["wh"][1], box["backend"]


def open_camera_direction(
    direction: str,
    profile: Optional[dict[str, Any]] = None,
    *,
    timeout_s: float = 8.0,
) -> tuple[Any, int, int, str]:
    prof = profile or load_camera_profile()
    idx = direction_device(direction, prof)
    w, h = direction_size(direction, prof)
    return open_camera_index(
        idx,
        width=w,
        height=h,
        fourcc=str(prof.get("fourcc") or "YUYV"),
        backend=str(prof.get("backend") or "v4l2"),
        gst_template=str(prof.get("gst_pipeline_template") or ""),
        timeout_s=timeout_s,
    )


def probe_cameras(
    profile: Optional[dict[str, Any]] = None,
    *,
    grabs: int = 2,
) -> dict[str, Any]:
    """Open each configured camera briefly and report actual sizes."""
    prof = profile or load_camera_profile()
    results: dict[str, Any] = {}
    all_ok = True
    for d in DIRECTIONS:
        cam = (prof.get("cameras") or {}).get(d)
        if not cam:
            results[d] = {"ok": False, "error": "not configured"}
            all_ok = False
            continue
        idx = _device_to_index(cam["device"])
        req_w, req_h = direction_size(d, prof)
        entry: dict[str, Any] = {
            "ok": False,
            "device": idx,
            "path": f"/dev/video{idx}",
            "requested_wh": [req_w, req_h],
            "actual_wh": None,
            "backend": None,
            "error": None,
        }
        cap = None
        try:
            cap, aw, ah, backend = open_camera_direction(d, prof, timeout_s=6.0)
            entry["backend"] = backend
            for _ in range(max(1, grabs)):
                ok, fr = cap.read()
                fr = _normalize_bgr(fr)
                if ok and fr is not None:
                    aw, ah = int(fr.shape[1]), int(fr.shape[0])
                    break
            entry["actual_wh"] = [aw, ah]
            entry["ok"] = True
            if (aw, ah) != (req_w, req_h):
                entry["warning"] = (
                    f"actual {aw}x{ah} != requested {req_w}x{req_h}"
                )
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc)
            all_ok = False
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        results[d] = entry
        if not entry["ok"]:
            all_ok = False
    return {
        "ok": all_ok,
        "profile": {
            "width": int(prof["width"]),
            "height": int(prof["height"]),
            "fourcc": prof.get("fourcc"),
            "backend": prof.get("backend"),
        },
        "cameras": results,
    }


def profile_for_web(profile: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    prof = profile or load_camera_profile()
    return {
        "width": int(prof["width"]),
        "height": int(prof["height"]),
        "fourcc": str(prof.get("fourcc") or "YUYV"),
        "backend": str(prof.get("backend") or "v4l2"),
        "gst_pipeline_template": str(prof.get("gst_pipeline_template") or ""),
        "cameras": {
            d: {
                "device": direction_device(d, prof)
                if d in (prof.get("cameras") or {})
                else None,
                "width": int(
                    ((prof.get("cameras") or {}).get(d) or {}).get("width")
                    or prof["width"]
                ),
                "height": int(
                    ((prof.get("cameras") or {}).get(d) or {}).get("height")
                    or prof["height"]
                ),
            }
            for d in DIRECTIONS
        },
    }
