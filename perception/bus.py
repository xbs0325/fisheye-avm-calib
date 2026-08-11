"""Thread-safe latest perception result (ROS2-replaceable later)."""

from __future__ import annotations

import threading
from typing import Callable, Optional

from perception.schema import PerceptionEvent


class PerceptionBus:
    """Holds the newest PerceptionEvent; optional subscribers get a copy."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[PerceptionEvent] = None
        self._subs: list[Callable[[PerceptionEvent], None]] = []

    def publish(self, event: PerceptionEvent) -> None:
        with self._lock:
            self._latest = event
            subs = list(self._subs)
        for cb in subs:
            try:
                cb(event)
            except Exception:
                pass

    def latest(self) -> Optional[PerceptionEvent]:
        with self._lock:
            return self._latest

    def subscribe(self, callback: Callable[[PerceptionEvent], None]) -> None:
        with self._lock:
            self._subs.append(callback)
