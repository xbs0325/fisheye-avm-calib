"""Parent-process VLM client: async ANALYZE via standalone Qwen3-VL subprocess."""

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

from perception.localize import enrich_event, smooth_grasp_heading
from perception.schema import PerceptionEvent, parse_vlm_response

DEFAULT_MODELS = Path(
    os.environ.get(
        "PERCEPTION_MODELS",
        os.environ.get(
            "WORLDMM_MODELS",
            str(Path.home() / "leucus" / "models" / "worldmm"),
        ),
    )
)
DEFAULT_VENV_PYTHON = Path(
    os.environ.get(
        "PERCEPTION_VENV_PYTHON",
        os.environ.get(
            "WORLDMM_VENV_PYTHON",
            str(Path.home() / "leucus" / ".venv-worldmm" / "bin" / "python"),
        ),
    )
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


class AnalyzeWorker:
    """Spawn standalone Qwen3-VL; request structured nav/grasp analysis async."""

    def __init__(
        self,
        *,
        vlm_name: str = "qwen3vl-2b",
        models_dir: Optional[Path] = None,
        mode: str = "nav",
        grasp_target: str = "object",
        max_side: int = 512,
        max_new_tokens: int = 128,
        venv_python: Optional[Path] = None,
        canvas_size: tuple[int, int] = (500, 500),
        scale_px_per_meter: float = 200.0,
        on_result: Optional[Callable[[PerceptionEvent], None]] = None,
        debug_input_path: Optional[Path] = None,
        occ_veto: Optional[Callable[[PerceptionEvent], PerceptionEvent]] = None,
    ):
        self.vlm_name = vlm_name
        self.models_dir = Path(models_dir or DEFAULT_MODELS)
        self.mode = "grasp" if mode == "grasp" else "nav"
        self.grasp_target = grasp_target or "object"
        self.max_side = int(max_side)
        self.max_new_tokens = int(max_new_tokens)
        self.venv_python = Path(venv_python) if venv_python else _resolve_worker_python()
        self.canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        self.scale_px_per_meter = float(scale_px_per_meter)
        self.on_result = on_result
        self.debug_input_path = Path(debug_input_path) if debug_input_path else None
        self.occ_veto = occ_veto

        self._lock = threading.Lock()
        self._busy = False
        self._event: Optional[PerceptionEvent] = None
        self._status = "vlm: idle"
        self._last_ms = 0.0
        self._error: Optional[str] = None
        self._enabled = False
        self._proc: Optional[subprocess.Popen] = None
        self._io_lock = threading.Lock()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="perception_vlm_")
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
    def holds_gpu(self) -> bool:
        """True while this process should not run OpenCV-CUDA (Jetson GPU lock)."""
        with self._lock:
            loading = bool(self._load_started and not self._enabled and self._error is None)
            return loading or self._busy

    @property
    def event(self) -> Optional[PerceptionEvent]:
        with self._lock:
            return self._event

    @property
    def summary(self) -> str:
        with self._lock:
            if self._event is None:
                return ""
            return self._event.summary or ""

    @property
    def status_line(self) -> str:
        with self._lock:
            if self._error:
                return f"vlm ERR: {self._error[:48]}"
            if self._busy:
                return "vlm: thinking..."
            if self._last_ms > 0:
                valid = ""
                if self._event is not None:
                    valid = " ok" if self._event.valid else " invalid"
                return f"vlm: {self._last_ms:.0f}ms{valid}  {self._status}"
            return self._status

    def set_geometry(
        self, canvas_size: tuple[int, int], scale_px_per_meter: float
    ) -> None:
        self.canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        self.scale_px_per_meter = float(scale_px_per_meter)

    def load(self) -> None:
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        # Keep stitch NumPy 1.x out of the VLM interpreter
        env.pop("PYTHONPATH", None)
        env["PERCEPTION_MODELS"] = str(self.models_dir)
        env.setdefault(
            "PERCEPTION_ATTN_IMPL",
            os.environ.get("WORLDMM_ATTN_IMPL", "sdpa"),
        )
        env.setdefault(
            "PERCEPTION_DEVICE_MAP",
            os.environ.get("WORLDMM_QWEN_DEVICE_MAP", "cuda:0"),
        )
        env.setdefault(
            "PERCEPTION_DTYPE",
            os.environ.get("WORLDMM_DTYPE", "bfloat16"),
        )
        env.pop("HF_ENDPOINT", None)
        # Only need the perception package for schema prompts in the worker
        env["PYTHONPATH"] = str(root)

        cmd = [
            str(self.venv_python),
            "-u",
            "-m",
            "perception.vlm_worker",
            "--vlm",
            self.vlm_name,
            "--models",
            str(self.models_dir),
            "--max-side",
            str(self.max_side),
            "--max-new-tokens",
            str(self.max_new_tokens),
        ]

        with self._lock:
            self._status = f"vlm: loading {self.vlm_name}..."

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
            target=self._drain_stderr, name="perception-vlm-stderr", daemon=True
        ).start()

        line = self._proc.stdout.readline()
        if not line:
            err = self._proc.poll()
            raise RuntimeError(
                f"VLM worker exited early (code={err}). "
                f"Check PERCEPTION_VENV_PYTHON={self.venv_python}"
            )
        line = line.strip()
        if line.startswith("ERR"):
            raise RuntimeError(line[4:].strip() or line)
        if line != "READY":
            raise RuntimeError(f"VLM worker unexpected: {line!r}")

        dt = time.time() - t0
        with self._lock:
            self._status = f"vlm ready {self.vlm_name} ({dt:.0f}s)"
            self._enabled = True
            self._error = None
            self._ready_at = time.time()
        print(
            f"[perception] Qwen3-VL worker ready via {self.venv_python} ({dt:.0f}s) "
            f"max_side={self.max_side} max_new_tokens={self.max_new_tokens}",
            flush=True,
        )

    def load_async(self) -> None:
        """Load in a background thread so BEV can start first."""
        if self._load_started:
            return
        self._load_started = True
        with self._lock:
            self._status = f"vlm: loading {self.vlm_name}..."

        def _go() -> None:
            try:
                self.load()
            except Exception as exc:
                with self._lock:
                    self._enabled = False
                    self._error = str(exc)
                    self._status = "vlm: load fail"
                print(f"[perception] VLM load failed: {exc}", flush=True)

        threading.Thread(target=_go, name="perception-vlm-load", daemon=True).start()

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            print(f"[VLM stderr] {line.rstrip()}", flush=True)

    def request(
        self, bev_bgr: np.ndarray, *, force: bool = False, occ_az: str = ""
    ) -> bool:
        """Single-flight: never overlap ANALYZE (overlapping stdin desync + GPU stall)."""
        del force  # force used to pile threads; that froze the stitch loop
        if not self._enabled or self._proc is None:
            return False
        if self._proc.poll() is not None:
            self._enabled = False
            return False
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            frame = bev_bgr.copy()
            az = (occ_az or "").strip()
        threading.Thread(
            target=self._run,
            args=(frame, az),
            name="perception-vlm-analyze",
            daemon=True,
        ).start()
        return True

    def _run(self, bev_bgr: np.ndarray, occ_az: str = "") -> None:
        path = Path(self._tmpdir.name) / f"bev_{time.time_ns()}.jpg"
        proc = self._proc
        try:
            if proc is None or proc.stdin is None or proc.stdout is None:
                raise RuntimeError("VLM worker not running")
            if proc.poll() is not None:
                raise RuntimeError(f"VLM worker exited (code={proc.returncode})")

            h, w = bev_bgr.shape[:2]
            side = self.max_side
            if max(h, w) > side:
                scale = side / float(max(h, w))
                bev_bgr = cv2.resize(
                    bev_bgr,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            if not cv2.imwrite(str(path), bev_bgr):
                raise RuntimeError(f"failed to write {path}")
            if self.debug_input_path is not None:
                try:
                    self.debug_input_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(self.debug_input_path), bev_bgr)
                except Exception:
                    pass

            if self.mode == "grasp":
                cmd = f"ANALYZE {path} grasp {self.grasp_target}"
                if occ_az:
                    cmd += f" ::{occ_az}"
                cmd += "\n"
            else:
                cmd = f"ANALYZE {path} nav\n"

            with self._io_lock:
                if proc.poll() is not None:
                    raise RuntimeError(f"VLM worker exited (code={proc.returncode})")
                proc.stdin.write(cmd)
                proc.stdin.flush()
                header = proc.stdout.readline()
                if not header:
                    raise RuntimeError("VLM worker closed stdout")
                header = header.strip()
                if header.startswith("ERR"):
                    raise RuntimeError(header[4:].strip() or header)
                if not header.startswith("OK"):
                    raise RuntimeError(f"unexpected worker reply: {header!r}")
                parts = header.split()
                ms = float(parts[1]) if len(parts) > 1 else 0.0
                text = proc.stdout.readline().rstrip("\n")
                end = proc.stdout.readline().strip()
                if end != "END":
                    raise RuntimeError(f"expected END, got {end!r}")

            event = parse_vlm_response(text, mode=self.mode, infer_ms=ms)
            event = enrich_event(
                event,
                canvas_size=self.canvas_size,
                scale_px_per_meter=self.scale_px_per_meter,
            )
            event, self._yaw_ema, self._range_ema = smooth_grasp_heading(
                event, yaw_ema=self._yaw_ema, range_ema=self._range_ema
            )
            if self.occ_veto is not None:
                event = self.occ_veto(event)
                if event.grasp is None or not event.grasp.targets:
                    self._yaw_ema = None
                    self._range_ema = None
                    if event.grasp is not None:
                        event.grasp.turn_hint = ""

            with self._lock:
                self._event = event
                self._last_ms = ms
                self._status = "vlm: ok" if event.valid else "vlm: parse-fallback"
                self._error = None
            xy = ""
            if event.grasp is not None and event.grasp.targets:
                t = event.grasp.targets[0]
                if t.x_m is not None and t.y_m is not None:
                    xy = f" xy=({t.x_m:.2f},{t.y_m:.2f})"
                    if t.range_m is not None:
                        xy += f" {t.range_m:.2f}m"
                    if event.grasp.turn_hint:
                        xy += f" {event.grasp.turn_hint}"
            print(
                f"[VLM {ms:.0f}ms valid={event.valid}] {event.summary[:80]}{xy} | {text[:100]}",
                flush=True,
            )
            if self.on_result is not None:
                try:
                    self.on_result(event)
                except Exception as exc:
                    print(f"[perception] on_result error: {exc}", flush=True)
        except Exception as exc:
            dead = self._proc is None or self._proc.poll() is not None
            with self._lock:
                self._error = str(exc)
                self._status = "vlm: fail"
                if dead:
                    self._enabled = False
                    self._status = "vlm: dead"
            print(f"[VLM ERROR] {exc}", flush=True)
            if dead:
                print("[VLM] worker died — stop auto ANALYZE (restart the demo)", flush=True)
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            with self._lock:
                self._busy = False

    def close(self) -> None:
        self._enabled = False
        deadline = time.time() + 8.0
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
