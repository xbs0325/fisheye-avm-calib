#!/usr/bin/env python3
"""Background Qwen3-VL caption worker via WorldMM subprocess.

Parent process uses CUDA OpenCV (NumPy 1.x). VLM runs in the WorldMM venv
(NumPy 2.x) so scipy/transformers and cv2 never share one interpreter.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

DEFAULT_MODELS = Path(
    os.environ.get("WORLDMM_MODELS", str(Path.home() / "leucus" / "models" / "worldmm"))
)
DEFAULT_WORLDMM_SRC = Path(
    os.environ.get("WORLDMM_SRC", str(Path.home() / "leucus" / "WorldMM" / "src"))
)
DEFAULT_VENV_PYTHON = Path(
    os.environ.get(
        "WORLDMM_VENV_PYTHON",
        str(Path.home() / "leucus" / ".venv-worldmm" / "bin" / "python"),
    )
)

VLM_DIRS = {
    "qwen3vl-2b": "Qwen3-VL-2B-Instruct",
    "qwen3vl-4b": "Qwen3-VL-4B-Instruct",
    "qwen3vl-8b": "Qwen3-VL-8B-Instruct",
}

DEFAULT_PROMPT = (
    "这是小车俯视环视拼接图（上=前方，下=后方，左/右=侧方；画面中心附近常为车体盲区或拼接空洞，可忽略）。"
    "任务：为地面小车避障与通行提供信息。请用两三句中文回答："
    "1) 四周地面是否有障碍/危险（人、箱子、线缆、台阶、坑洼、反光锥等）及大致方位（前/后/左/右）；"
    "2) 哪一侧相对更空旷、更适合通行；"
    "3) 看不清的区域直接说不确定。"
    "只依据画面可见内容，不要编造汽车、车道线或其他看不到的物体。"
)


def _resolve_worker_python() -> Path:
    cand = Path(DEFAULT_VENV_PYTHON)
    if cand.is_file() and os.access(cand, os.X_OK):
        return cand
    # Fallbacks if env not set by run script
    home = Path.home()
    for p in (
        home / "leucus" / ".venv-worldmm" / "bin" / "python",
        Path(sys.executable),
    ):
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return Path(sys.executable)


class CaptionWorker:
    """Spawn WorldMM VLM in a venv subprocess; caption BEV frames async."""

    def __init__(
        self,
        *,
        vlm_name: str = "qwen3vl-2b",
        models_dir: Optional[Path] = None,
        prompt: str = DEFAULT_PROMPT,
        max_side: int = 512,
        max_new_tokens: int = 128,
        venv_python: Optional[Path] = None,
    ):
        self.vlm_name = vlm_name
        self.models_dir = Path(models_dir or DEFAULT_MODELS)
        self.prompt = prompt
        self.max_side = int(max_side)
        self.max_new_tokens = int(max_new_tokens)
        self.venv_python = Path(venv_python) if venv_python else _resolve_worker_python()

        self._lock = threading.Lock()
        self._busy = False
        self._caption = ""
        self._status = "vlm: idle"
        self._last_ms = 0.0
        self._error: Optional[str] = None
        self._enabled = False
        self._proc: Optional[subprocess.Popen] = None
        self._io_lock = threading.Lock()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="bev_vlm_")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def caption(self) -> str:
        with self._lock:
            return self._caption

    @property
    def status_line(self) -> str:
        with self._lock:
            if self._error:
                return f"vlm ERR: {self._error[:48]}"
            if self._busy:
                return "vlm: thinking…"
            if self._last_ms > 0:
                return f"vlm: {self._last_ms:.0f}ms  {self._status}"
            return self._status

    def load(self) -> None:
        """Start WorldMM worker subprocess and wait for READY."""
        root = Path(__file__).resolve().parents[1]
        worker_mod = "demo_bev_vlm.vlm_worker"
        env = os.environ.copy()
        # Keep WORLDMM_* ; strip PYTHONPATH pollution that could pull NumPy 1.x into venv
        env.pop("PYTHONPATH", None)
        if DEFAULT_WORLDMM_SRC.is_dir():
            env["WORLDMM_SRC"] = str(DEFAULT_WORLDMM_SRC)
        env["WORLDMM_MODELS"] = str(self.models_dir)
        env.setdefault("WORLDMM_ATTN_IMPL", "sdpa")
        env.setdefault("WORLDMM_QWEN_DEVICE_MAP", "cuda:0")
        env.setdefault("WORLDMM_DTYPE", "bfloat16")
        env.pop("HF_ENDPOINT", None)
        # Ensure demo package importable
        env["PYTHONPATH"] = str(root) + (
            f":{DEFAULT_WORLDMM_SRC}" if DEFAULT_WORLDMM_SRC.is_dir() else ""
        )

        cmd = [
            str(self.venv_python),
            "-u",
            "-m",
            worker_mod,
            "--vlm",
            self.vlm_name,
            "--models",
            str(self.models_dir),
            "--max-side",
            str(self.max_side),
            "--max-new-tokens",
            str(self.max_new_tokens),
            "--prompt",
            self.prompt,
        ]

        with self._lock:
            self._status = f"vlm: loading {self.vlm_name}…"

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

        # Drain stderr in background so pipe never blocks
        threading.Thread(
            target=self._drain_stderr, name="bev-vlm-stderr", daemon=True
        ).start()

        line = self._proc.stdout.readline()
        if not line:
            err = self._proc.poll()
            raise RuntimeError(
                f"VLM worker exited early (code={err}). Check WORLDMM_VENV_PYTHON={self.venv_python}"
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
        print(
            f"[demo] VLM worker ready via {self.venv_python} ({dt:.0f}s)",
            flush=True,
        )

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            print(f"[VLM stderr] {line.rstrip()}", flush=True)

    def request(self, bev_bgr: np.ndarray, *, force: bool = False) -> bool:
        """Queue a caption job if idle. Returns True if accepted."""
        if not self._enabled or self._proc is None:
            return False
        with self._lock:
            if self._busy and not force:
                return False
            self._busy = True
            frame = bev_bgr.copy()
        t = threading.Thread(
            target=self._run, args=(frame,), name="bev-vlm-caption", daemon=True
        )
        t.start()
        return True

    def _run(self, bev_bgr: np.ndarray) -> None:
        path = Path(self._tmpdir.name) / f"bev_{time.time_ns()}.jpg"
        # Hold local refs — close() may null self._proc while we run.
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

            with self._io_lock:
                if proc.poll() is not None:
                    raise RuntimeError(f"VLM worker exited (code={proc.returncode})")
                proc.stdin.write(f"CAPTION {path}\n")
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

            with self._lock:
                self._caption = text
                self._last_ms = ms
                self._status = "vlm: ok"
                self._error = None
            print(f"[VLM {ms:.0f}ms] {text}", flush=True)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
                self._status = "vlm: fail"
            print(f"[VLM ERROR] {exc}", flush=True)
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            with self._lock:
                self._busy = False

    def close(self) -> None:
        self._enabled = False
        # Wait for in-flight caption so we don't yank stdin/stdout mid-read
        deadline = time.time() + 8.0
        while time.time() < deadline:
            with self._lock:
                if not self._busy:
                    break
            time.sleep(0.05)

        with self._io_lock:
            proc = self._proc
            self._proc = None
            if proc is None:
                pass
            else:
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
