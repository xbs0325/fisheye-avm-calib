#!/usr/bin/env python3
"""Optional V4L2/CSI camera check inside container or on host.

Set DEBUG_LOG=/path.ndjson for NDJSON traces.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

DEBUG_LOG = os.environ.get("DEBUG_LOG", "")


def emit(hid: str, location: str, message: str, data: dict) -> None:
    if not DEBUG_LOG:
        return
    payload = {
        "sessionId": os.environ.get("DEBUG_SESSION", ""),
        "runId": os.environ.get("DEBUG_RUN_ID", "v4l"),
        "hypothesisId": hid,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    Path(DEBUG_LOG).parent.mkdir(parents=True, exist_ok=True)
    with Path(DEBUG_LOG).open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    idx = int(os.environ.get("V4L_INDEX", "0"))
    videos = sorted(Path("/dev").glob("video*"))
    media = sorted(Path("/dev").glob("media*"))
    vi = sorted(Path("/dev").glob("capture-vi-channel*"))
    print(
        f"nodes video={len(videos)} media={len(media)} capture-vi={len(vi)}"
    )
    emit(
        "H1",
        "probe_v4l_docker.py:nodes",
        "nodes",
        {"videos": [str(p) for p in videos], "media": len(media), "vi": len(vi)},
    )

    import cv2

    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    opened = bool(cap.isOpened())
    ok, frame = (False, None)
    if opened:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
        ok, frame = cap.read()
        cap.release()
    shape = None if frame is None else getattr(frame, "shape", None)
    print(f"video{idx} opened={opened} read_ok={ok} shape={shape}")
    emit(
        "H4",
        "probe_v4l_docker.py:read",
        "opencv read",
        {"opened": opened, "read_ok": bool(ok), "shape": shape},
    )
    if not media or not vi:
        print(
            "HINT: CSI tegra-video needs /dev/media0 and /dev/capture-vi-channel* "
            "(use scripts/docker_run_web.sh or updated compose)."
        )
    return 0 if opened and ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
