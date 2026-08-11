"""Parent-process YOLO-World client (async subprocess, GPU FP16)."""

from __future__ import annotations

import json
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

from perception.localize import enrich_event, smooth_grasp_heading
from perception.schema import PerceptionEvent, parse_nav_payload, parse_vlm_response

DEFAULT_VENV_PYTHON = Path(
    os.environ.get(
        "PERCEPTION_VENV_PYTHON",
        os.environ.get(
            "WORLDMM_VENV_PYTHON",
            str(Path.home() / "leucus" / ".venv-worldmm" / "bin" / "python"),
        ),
    )
)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "perception" / "yolov8s-worldv2.pt"


def _is_protocol(line: str, prefix: str) -> bool:
    # Do not treat pip "ERROR: ..." as our "ERR <msg>" token.
    if prefix == "{":
        return line.startswith("{")
    return line == prefix or line.startswith(prefix + " ")


def _read_worker_line(stdout, want: tuple[str, ...], timeout_s: float = 60.0) -> str:
    """Skip Ultralytics / pip chatter until a protocol line."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        raw = stdout.readline()
        if not raw:
            raise RuntimeError("ov worker closed stdout")
        line = raw.strip()
        if not line:
            continue
        for prefix in want:
            if _is_protocol(line, prefix):
                return line
    raise RuntimeError(f"timeout waiting for {want}")


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


class AnythingWorker:
    """Async open-vocab boxes on stitched BEV."""

    def __init__(
        self,
        *,
        weights: Optional[Path] = None,
        target: str = "bottle",
        imgsz: int = 384,
        conf: float = 0.18,
        device: str = "0",
        venv_python: Optional[Path] = None,
        canvas_size: tuple[int, int] = (500, 500),
        scale_px_per_meter: float = 200.0,
        on_result: Optional[Callable[[PerceptionEvent], None]] = None,
        interval_s: float = 0.8,
        send_max_side: int = 384,
    ):
        self.weights = Path(weights or DEFAULT_WEIGHTS)
        self.target = target or "bottle"
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
        self._status = "ov: idle"
        self._last_ms = 0.0
        self._error: Optional[str] = None
        self._enabled = False
        self._proc: Optional[subprocess.Popen] = None
        self._io_lock = threading.Lock()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="perception_ov_")
        self._last_req_t = 0.0
        self._yaw_ema: Optional[float] = None
        self._range_ema: Optional[float] = None
        self._ready_at = 0.0
        self._load_started = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def ready_at(self) -> float:
        return self._ready_at

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def loading(self) -> bool:
        with self._lock:
            return bool(self._load_started and not self._enabled and self._error is None)

    @property
    def uses_gpu(self) -> bool:
        return str(self.device).lower() not in ("cpu", "")

    @property
    def holds_gpu(self) -> bool:
        """Only block CUDA stitch while weights load. Infer is short FP16."""
        if not self.uses_gpu:
            return False
        with self._lock:
            return bool(self._load_started and not self._enabled and self._error is None)

    @property
    def event(self) -> Optional[PerceptionEvent]:
        with self._lock:
            return self._event

    @property
    def status_line(self) -> str:
        with self._lock:
            if self._error:
                return f"ov ERR: {self._error[:40]}"
            if self._load_started and not self._enabled:
                return "ov: loading..."
            if self._busy:
                return "ov: running..."
            if self._last_ms > 0:
                n = 0
                if self._event and self._event.grasp:
                    n = len(self._event.grasp.targets)
                return f"ov: {self._last_ms:.0f}ms n={n}"
            return self._status

    def load_async(self) -> None:
        if self._load_started:
            return
        self._load_started = True
        with self._lock:
            self._status = "ov: loading..."
        threading.Thread(target=self._load_safe, name="perception-ov-load", daemon=True).start()

    def _load_safe(self) -> None:
        try:
            self.load()
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
                self._status = "ov: load fail"
                self._enabled = False
            print(f"[OV ERROR] load failed: {exc}", flush=True)

    def load(self) -> None:
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"missing YOLO-World weights: {self.weights} "
                "(run scripts/download_perception_models.sh)"
            )
        root = ROOT
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONPATH"] = str(root)
        env["PYTHONNOUSERSITE"] = "1"
        env["MPLBACKEND"] = "Agg"
        env.pop("HF_ENDPOINT", None)
        env["YOLO_AUTOINSTALL"] = "false"
        env["YOLO_VERBOSE"] = "false"
        if not self.uses_gpu:
            env["CUDA_VISIBLE_DEVICES"] = ""

        cmd = [
            str(self.venv_python),
            "-u",
            "-m",
            "perception.anything_worker",
            "--weights",
            str(self.weights),
            "--imgsz",
            str(self.imgsz),
            "--conf",
            str(self.conf),
            "--device",
            "cpu" if not self.uses_gpu else str(self.device),
            "--target",
            str(self.target or "bottle"),
        ]

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
            target=self._drain_stderr, name="perception-ov-stderr", daemon=True
        ).start()

        deadline = time.time() + 180.0
        line = ""
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            if line == "READY" or line.startswith("ERR "):
                break
            print(f"[OV stdout] {line}", flush=True)
        if not line:
            raise RuntimeError(
                f"ov worker exited early (code={self._proc.poll()}) "
                f"python={self.venv_python}"
            )
        if line.startswith("ERR "):
            raise RuntimeError(line[4:].strip() or line)
        if line != "READY":
            raise RuntimeError(f"ov worker unexpected: {line!r}")

        dt = time.time() - t0
        with self._lock:
            self._status = f"ov ready ({dt:.0f}s)"
            self._enabled = True
            self._error = None
            self._ready_at = time.time()
        print(
            f"[perception] YOLO-World ready via {self.venv_python} "
            f"({dt:.0f}s) weights={self.weights.name} device={self.device} "
            f"imgsz={self.imgsz} fp16={self.uses_gpu}",
            flush=True,
        )

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            s = line.rstrip()
            if not s:
                continue
            low = s.lower()
            if any(k in low for k in ("error", "traceback", "cuda", "[da]")):
                print(f"[OV stderr] {s}", flush=True)

    def request(self, bev_bgr: np.ndarray, *, force: bool = False) -> bool:
        if not self._enabled or self._proc is None:
            return False
        now = time.time()
        with self._lock:
            if self._busy and not force:
                return False
            if not force and (now - self._last_req_t) < self.interval_s:
                return False
            if self._busy:
                return False
            self._busy = True
            self._last_req_t = now
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
            target=self._run, args=(frame,), name="perception-ov", daemon=True
        ).start()
        return True

    def _run(self, bev_bgr: np.ndarray) -> None:
        path = Path(self._tmpdir.name) / f"bev_{time.time_ns()}.jpg"
        proc = self._proc
        try:
            if proc is None or proc.stdin is None or proc.stdout is None:
                raise RuntimeError("ov worker not running")
            if proc.poll() is not None:
                raise RuntimeError(f"ov worker exited ({proc.returncode})")
            if not cv2.imwrite(str(path), bev_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
                raise RuntimeError(f"failed to write {path}")

            cw, ch = self.canvas_size
            tgt = (self.target or "bottle").replace("\n", " ").strip() or "bottle"
            cmd = f"ANY {path} {cw} {ch} {self.scale_px_per_meter} {tgt}\n"
            with self._io_lock:
                proc.stdin.write(cmd)
                proc.stdin.flush()
                header = _read_worker_line(proc.stdout, want=("OK", "ERR"))
                if header.startswith("ERR"):
                    raise RuntimeError(header[4:].strip() or header)
                parts = header.split()
                ms = float(parts[1]) if len(parts) > 1 else 0.0
                text = _read_worker_line(proc.stdout, want=("{", "ERR"))
                if text.startswith("ERR"):
                    raise RuntimeError(text[4:].strip() or text)
                end = _read_worker_line(proc.stdout, want=("END",))

            event = parse_vlm_response(text, mode="grasp", infer_ms=ms)
            event = enrich_event(
                event,
                canvas_size=self.canvas_size,
                scale_px_per_meter=self.scale_px_per_meter,
            )
            event, self._yaw_ema, self._range_ema = smooth_grasp_heading(
                event, yaw_ema=self._yaw_ema, range_ema=self._range_ema
            )
            if event.grasp is not None:
                event.grasp.source = "yolo-world"
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                obj = {}
            if isinstance(obj, dict) and obj.get("obstacles"):
                nav = parse_nav_payload(obj)
                nav.source = "yolo-world"
                event.nav = nav

            xy = ""
            n_tgt = len(event.grasp.targets) if event.grasp else 0
            n_obs = len(event.nav.obstacles) if event.nav else 0
            if event.grasp and event.grasp.targets:
                t = event.grasp.targets[0]
                if t.x_m is not None and t.y_m is not None:
                    xy = f" xy=({t.x_m:.2f},{t.y_m:.2f})"
                    if t.range_m is not None:
                        xy += f" {t.range_m:.2f}m"
            extra = ""
            if event.nav and event.nav.obstacles:
                extra = " also=" + ",".join(o.label for o in event.nav.obstacles[:5])
            print(
                f"[OV {ms:.0f}ms valid={event.valid} n={n_tgt}+{n_obs}] "
                f"{event.summary[:80]}{xy}{extra}",
                flush=True,
            )

            with self._lock:
                self._event = event
                self._last_ms = ms
                self._status = "ov: ok" if event.valid else "ov: parse-fail"
                self._error = None
            if event.grasp is None or not event.grasp.targets:
                self._yaw_ema = None
                self._range_ema = None
            if self.on_result is not None:
                try:
                    self.on_result(event)
                except Exception as exc:
                    print(f"[perception] ov on_result error: {exc}", flush=True)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
                self._status = "ov: fail"
            print(f"[OV ERROR] {exc}", flush=True)
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
                    proc.wait(timeout=8)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        try:
            self._tmpdir.cleanup()
        except Exception:
            pass
