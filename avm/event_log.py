#!/usr/bin/env python3
"""In-memory ring log for Web UI."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, List


class EventLog:
    def __init__(self, maxlen: int = 200):
        self._lines: Deque[str] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def log(self, msg: str, *, level: str = "INFO") -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        with self._lock:
            self._lines.append(line)
        print(line, flush=True)

    def info(self, msg: str) -> None:
        self.log(msg, level="INFO")

    def warn(self, msg: str) -> None:
        self.log(msg, level="WARN")

    def error(self, msg: str) -> None:
        self.log(msg, level="ERROR")

    def dump(self, n: int = 80) -> List[str]:
        with self._lock:
            lines = list(self._lines)
        return lines[-n:]

    def text(self, n: int = 80) -> str:
        return "\n".join(self.dump(n)) or "(暂无日志)"


LOG = EventLog()
