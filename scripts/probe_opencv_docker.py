#!/usr/bin/env python3
"""Docker OpenCV import probe — writes NDJSON debug logs for session 8f0704."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

LOG = Path("/home/seeed/bev_demo/avm_gpu/.cursor/debug-8f0704.log")
# Inside container the workspace may not be mounted; fall back to /tmp then copy via stdout marker.
CONTAINER_LOG = Path("/tmp/debug-8f0704.log")


def emit(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "pre") -> None:
    payload = {
        "sessionId": "8f0704",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload, ensure_ascii=False)
    for p in (CONTAINER_LOG, LOG):
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
    print(line, flush=True)


def main() -> int:
    run_id = os.environ.get("DEBUG_RUN_ID", "pre")
    # #region agent log
    emit(
        "A",
        "probe_opencv_docker.py:env",
        "container env snapshot",
        {
            "OPENCV_CUDA_PREFIX": os.environ.get("OPENCV_CUDA_PREFIX"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "PATH": os.environ.get("PATH", "")[:300],
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "exists_opt_opencv": Path("/opt/opencv-cuda/lib").is_dir(),
            "exists_cuda_lib64": Path("/usr/local/cuda/lib64").is_dir(),
            "exists_nppig": any(
                Path(p).exists()
                for p in (
                    "/usr/local/cuda/lib64/libnppig.so.13",
                    "/usr/lib/aarch64-linux-gnu/libnppig.so.13",
                    "/opt/opencv-cuda/lib/libnppig.so.13",
                )
            ),
            "nppig_candidates": [
                str(p)
                for p in [
                    Path("/usr/local/cuda/lib64/libnppig.so.13"),
                    Path("/usr/lib/aarch64-linux-gnu/libnppig.so.13"),
                    Path("/opt/opencv-cuda/lib/libnppig.so.13"),
                ]
                if p.exists()
            ],
            "cuda_lib64_sample": sorted(
                [p.name for p in Path("/usr/local/cuda/lib64").glob("libnpp*.so*")[:20]]
            )
            if Path("/usr/local/cuda/lib64").is_dir()
            else [],
        },
        run_id=run_id,
    )
    # #endregion

    so = Path(
        "/opt/opencv-cuda/lib/python3.12/site-packages/cv2/python-3.12/"
        "cv2.cpython-312-aarch64-linux-gnu.so"
    )
    missing = []
    resolved = []
    if so.is_file():
        try:
            out = subprocess.check_output(["ldd", str(so)], text=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            out = exc.output or str(exc)
        for line in out.splitlines():
            if "not found" in line:
                missing.append(line.strip())
            elif "libnpp" in line or "libcudart" in line or "libavcodec" in line:
                resolved.append(line.strip())
    # #region agent log
    emit(
        "B",
        "probe_opencv_docker.py:ldd",
        "cv2.so ldd missing/resolved",
        {
            "so_exists": so.is_file(),
            "missing_count": len(missing),
            "missing": missing[:40],
            "interesting_resolved": resolved[:40],
        },
        run_id=run_id,
    )
    # #endregion

    # #region agent log
    emit(
        "C",
        "probe_opencv_docker.py:ld_path",
        "whether LD_LIBRARY_PATH includes nvidia/cuda dirs",
        {
            "ld_parts": (os.environ.get("LD_LIBRARY_PATH") or "").split(":"),
            "has_cuda_in_ld": any(
                "cuda" in p for p in (os.environ.get("LD_LIBRARY_PATH") or "").split(":")
            ),
            "has_nvidia_in_ld": any(
                "nvidia" in p for p in (os.environ.get("LD_LIBRARY_PATH") or "").split(":")
            ),
        },
        run_id=run_id,
    )
    # #endregion

    err = None
    ver = None
    devices = None
    try:
        import cv2  # noqa: WPS433

        ver = cv2.__version__
        devices = int(cv2.cuda.getCudaEnabledDeviceCount())
        ok = True
    except Exception as exc:  # noqa: BLE001
        ok = False
        err = f"{type(exc).__name__}: {exc}"

    # #region agent log
    emit(
        "D",
        "probe_opencv_docker.py:import",
        "cv2 import result",
        {"ok": ok, "version": ver, "cuda_devices": devices, "error": err},
        run_id=run_id,
    )
    # #endregion

    # #region agent log
    # Hypothesis E: image content — is libavcodec staged but npp not?
    lib = Path("/opt/opencv-cuda/lib")
    emit(
        "E",
        "probe_opencv_docker.py:staged",
        "staged lib inventory markers",
        {
            "has_libavcodec62": (lib / "libavcodec.so.62").exists(),
            "has_nppig": (lib / "libnppig.so.13").exists(),
            "has_cudart": any(lib.glob("libcudart.so*")),
            "staged_npp_names": sorted(p.name for p in lib.glob("libnpp*.so*"))[:30],
            "staged_cuda_names": sorted(
                p.name for p in lib.glob("libcuda*.so*") 
            )[:30] + sorted(p.name for p in lib.glob("libcudart*.so*"))[:10],
        },
        run_id=run_id,
    )
    # #endregion

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
