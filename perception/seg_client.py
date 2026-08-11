"""Parent-process YOLO-seg client for fast nav geometry."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from perception.localize import enrich_event
from perception.schema import PerceptionEvent, parse_vlm_response

DEFAULT_VENV_PYTHON = Path(
    os.environ.get(
        "PERCEPTION_VENV_PYTHON",
        os.environ.get(
            "WORLDMM_VENV_PYTHON",
            str(Path.home() / "leucus" / ".venv-worldmm" / "bin" / "python"),
        ),
    )
)
DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parents[1] / "models" / "perception" / "yolov8n-seg.pt"
)


def _resolve_worker_python() -> Path:
    cand = Path(DEFAULT_VENV_PYTHON)
    if cand.is_file() and os.access(cand, os.X_OK):
        return cand
    home = Path.home()
    for p in (
        home / "leucus" / ".venv-worldmm" / "bin" / "python",
        Path(sys.executable),
    ):
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return Path(sys.executable)


class SegWorker:
    """Async YOLO-seg → NavResult (obstacles + free_dirs)."""

    def __init__(
        self,
        *,
        weights: Optional[Path] = None,
        imgsz: int = 512,
        conf: float = 0.35,
        device: str = "0",
        venv_python: Optional[Path] = None,
        canvas_size: tuple[int, int] = (500, 500),
        scale_px_per_meter: float = 200.0,
        on_result: Optional[Callable[[PerceptionEvent], None]] = None,
        interval_s: float = 0.45,
        send_max_side: int = 512,
    ):
        self.weights = Path(weights or DEFAULT_WEIGHTS)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.device = str(device)
        self.venv_python = Path(venv_python) if venv_python else _resolve_worker_python()
        self.canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        self.scale_px_per_meter = float(scale_px_per_meter)
        self.on_result = on_result
        self.interval_s = float(interval_s)
        self.send_max_side = int(send_max_side)

        self._lock = threading.Lock()
        self._busy = False
        self._event: Optional[PerceptionEvent] = None
        self._status = "seg: idle"
        self._last_ms = 0.0
        self._error: Optional[str] = None
        self._enabled = False
        self._proc: Optional[subprocess.Popen] = None
        self._io_lock = threading.Lock()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="perception_seg_")
        self._last_req_t = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def event(self) -> Optional[PerceptionEvent]:
        with self._lock:
            return self._event

    @property
    def status_line(self) -> str:
        with self._lock:
            if self._error:
                return f"seg ERR: {self._error[:40]}"
            if self._busy:
                return "seg: running..."
            if self._last_ms > 0:
                n = 0
                if self._event and self._event.nav:
                    n = len(self._event.nav.obstacles)
                return f"seg: {self._last_ms:.0f}ms n={n}  {self._status}"
            return self._status

    def load(self) -> None:
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"缺少 YOLO-seg 权重: {self.weights}（请放置 yolov8n-seg.pt）"
            )
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONPATH"] = str(root)
        env["PYTHONNOUSERSITE"] = "1"
        env["MPLBACKEND"] = "Agg"
        env.pop("HF_ENDPOINT", None)

        cmd = [
            str(self.venv_python),
            "-u",
            "-m",
            "perception.seg_worker",
            "--weights",
            str(self.weights),
            "--imgsz",
            str(self.imgsz),
            "--conf",
            str(self.conf),
            "--device",
            self.device,
        ]
        with self._lock:
            self._status = "seg: loading..."

        t0 = time.time()
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(root),
        )
        assert self._proc.stdout is not None
        threading.Thread(
            target=self._drain_stderr, name="perception-seg-stderr", daemon=True
        ).start()

        # Ultralytics may print deprecation warnings to stdout — skip until READY/ERR
        deadline = time.time() + 180.0
        line = ""
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            if line == "READY" or line.startswith("ERR"):
                break
            # ignore banners / warnings on stdout
            print(f"[SEG stdout] {line}", flush=True)
        if not line:
            raise RuntimeError(
                f"seg worker exited early (code={self._proc.poll()}) "
                f"python={self.venv_python}"
            )
        if line.startswith("ERR"):
            raise RuntimeError(line[4:].strip() or line)
        if line != "READY":
            raise RuntimeError(f"seg worker unexpected: {line!r}")

        dt = time.time() - t0
        with self._lock:
            self._status = f"seg ready ({dt:.0f}s)"
            self._enabled = True
            self._error = None
        print(
            f"[perception] YOLO-seg ready via {self.venv_python} ({dt:.0f}s) "
            f"weights={self.weights.name}",
            flush=True,
        )

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            # Ultralytics is chatty; keep errors visible
            s = line.rstrip()
            if not s:
                continue
            if any(k in s.lower() for k in ("error", "traceback", "warn", "cuda")):
                print(f"[SEG stderr] {s}", flush=True)

    def request(self, bev_bgr: np.ndarray, *, force: bool = False) -> bool:
        if not self._enabled or self._proc is None:
            return False
        now = time.time()
        with self._lock:
            if self._busy and not force:
                return False
            if not force and (now - self._last_req_t) < self.interval_s:
                return False
            # force while busy: ignore (avoids stdin protocol deadlock)
            if self._busy:
                return False
            self._busy = True
            self._last_req_t = now
            # Downscale before copy to cut IPC + GPU pressure
            h, w = bev_bgr.shape[:2]
            side = self.send_max_side
            if max(h, w) > side:
                scale = side / float(max(h, w))
                small = cv2.resize(
                    bev_bgr,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                small = bev_bgr
            frame = small.copy()
        threading.Thread(
            target=self._run, args=(frame,), name="perception-seg", daemon=True
        ).start()
        return True

    def _run(self, bev_bgr: np.ndarray) -> None:
        path = Path(self._tmpdir.name) / f"bev_{time.time_ns()}.jpg"
        proc = self._proc
        try:
            if proc is None or proc.stdin is None or proc.stdout is None:
                raise RuntimeError("seg worker not running")
            if proc.poll() is not None:
                raise RuntimeError(f"seg worker exited ({proc.returncode})")
            if not cv2.imwrite(str(path), bev_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80]):
                raise RuntimeError(f"failed to write {path}")

            cw, ch = self.canvas_size
            cmd = f"SEG {path} {cw} {ch} {self.scale_px_per_meter}\n"
            with self._io_lock:
                proc.stdin.write(cmd)
                proc.stdin.flush()
                header = proc.stdout.readline()
                if not header:
                    raise RuntimeError("seg worker closed stdout")
                header = header.strip()
                if header.startswith("ERR"):
                    raise RuntimeError(header[4:].strip() or header)
                if not header.startswith("OK"):
                    raise RuntimeError(f"unexpected: {header!r}")
                parts = header.split()
                ms = float(parts[1]) if len(parts) > 1 else 0.0
                text = proc.stdout.readline().rstrip("\n")
                end = proc.stdout.readline().strip()
                if end != "END":
                    raise RuntimeError(f"expected END, got {end!r}")

            event = parse_vlm_response(text, mode="nav", infer_ms=ms)
            # Seg already fills x_m/y_m; enrich still ok for missing
            event = enrich_event(
                event,
                canvas_size=self.canvas_size,
                scale_px_per_meter=self.scale_px_per_meter,
            )
            if event.nav is not None:
                # mark source in summary if missing
                if event.raw_text and '"source": "seg"' in event.raw_text.replace(
                    " ", ""
                ):
                    pass
                event.summary = event.summary or "seg"

            with self._lock:
                self._event = event
                self._last_ms = ms
                self._status = "seg: ok" if event.valid else "seg: parse-fail"
                self._error = None
            if self.on_result is not None:
                try:
                    self.on_result(event)
                except Exception as exc:
                    print(f"[perception] seg on_result error: {exc}", flush=True)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
                self._status = "seg: fail"
            print(f"[SEG ERROR] {exc}", flush=True)
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            with self._lock:
                self._busy = False

    def close(self) -> None:
        self._enabled = False
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._lock:
                if not self._busy:
                    break
            time.sleep(0.05)
        with self._io_lock:
            proc = self._proc
            self._proc = None
            if proc is not None:
                try:
                    if proc.stdin and proc.poll() is None:
                        proc.stdin.write("QUIT\n")
                        proc.stdin.flush()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        try:
            self._tmpdir.cleanup()
        except Exception:
            pass
