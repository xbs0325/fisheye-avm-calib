#!/usr/bin/env python3
"""aiortc bridge: pull BGR frames from GpuStreamHub → WebRTC video track."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional, Set

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

from avm.gpu_hub import GpuStreamHub
from avm.event_log import LOG


class HubVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, hub: GpuStreamHub, *, target_fps: float = 20.0):
        super().__init__()
        self.hub = hub
        self._last_seq = -1
        self._target_fps = max(5.0, float(target_fps))
        self._blank: Optional[np.ndarray] = None

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        loop = asyncio.get_running_loop()
        seq, bgr = await loop.run_in_executor(
            None, lambda: self.hub.wait_bgr(self._last_seq, timeout=0.5)
        )
        if bgr is not None:
            self._last_seq = seq
            frame_bgr = bgr
        else:
            if self._blank is None:
                self._blank = np.zeros((360, 640, 3), dtype=np.uint8)
            frame_bgr = self._blank
        frame = VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


class WebRtcBridge:
    """Background asyncio loop hosting RTCPeerConnections."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="webrtc-loop", daemon=True
        )
        self._pcs: Set[RTCPeerConnection] = set()
        self._hub: Optional[GpuStreamHub] = None
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def set_hub(self, hub: GpuStreamHub) -> None:
        self._hub = hub

    def _submit(self, coro, timeout: float = 30.0):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    async def _handle_offer(self, sdp: str, type_: str) -> dict[str, str]:
        if self._hub is None:
            raise RuntimeError("hub not set")
    # webrtc 对 calib 模式：idle 以外都允许
        if self._hub.mode == "idle":
            raise RuntimeError("请先启动 GPU 预览/BEV/标定流，再协商 WebRTC")

        LOG.info(f"WebRTC offer type={type_} sdp_len={len(sdp or '')}")
        pc = RTCPeerConnection()
        self._pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            LOG.info(f"WebRTC pc state={pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self._discard(pc)

        track = HubVideoTrack(self._hub)
        pc.addTrack(track)
        offer = RTCSessionDescription(sdp=sdp, type=type_)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        for _ in range(50):
            if pc.iceGatheringState == "complete":
                break
            await asyncio.sleep(0.05)
        assert pc.localDescription is not None
        LOG.info(
            f"WebRTC answer type={pc.localDescription.type} "
            f"sdp_len={len(pc.localDescription.sdp)} peers={len(self._pcs)}"
        )
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def _discard(self, pc: RTCPeerConnection) -> None:
        if pc in self._pcs:
            self._pcs.discard(pc)
        try:
            await pc.close()
        except Exception:
            pass

    async def _close_all(self) -> None:
        pcs = list(self._pcs)
        self._pcs.clear()
        for pc in pcs:
            try:
                await pc.close()
            except Exception:
                pass

    def handle_offer(self, sdp: str, type_: str = "offer") -> dict[str, str]:
        return self._submit(self._handle_offer(sdp, type_))

    def close_all(self) -> None:
        try:
            self._submit(self._close_all(), timeout=10.0)
        except Exception:
            pass

    def peer_count(self) -> int:
        return len(self._pcs)

    def status(self) -> dict[str, Any]:
        return {"peers": self.peer_count(), "loop_running": self._thread.is_alive()}
