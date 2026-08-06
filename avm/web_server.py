#!/usr/bin/env python3
"""GPU Web 引导服务：状态 / 可跳步向导 / WebRTC 推流。

热路径：CUDA remap/warp/blend → BGR → aiortc(VP8/H264) → 浏览器。
不再用 MJPEG 作为主推流（旧方案 CPU JPEG 易卡）。

用法：
  source scripts/env_opencv_cuda.sh
  python3 -m avm.web_server --host 0.0.0.0 --port 8787
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avm.cuda_cv import cuda_available, cuda_status_line, log_cuda_status
from avm.event_log import LOG
from avm.calib_config import load_all_config, save_all_config
from avm.gpu_hub import GpuStreamHub
from avm.remote_control import CMD_TO_KEY, ensure_control_file, push_control_cmd
from avm.webrtc_stream import WebRtcBridge
from avm.wizard import (
    CALIB_DIR,
    check_extrinsics_quality,
    check_intrinsics_quality,
    _cuda_env,
)

CONTROL_FILE = ROOT / "output" / "web_control.txt"
HUB: Optional[GpuStreamHub] = None
WEBRTC: Optional[WebRtcBridge] = None
STATE: dict[str, Any] = {
    "step": "status",
    "skipped_intrinsics": False,
    "skipped_extrinsics": False,
    "calib_proc": None,
    "message": "",
}
STATE_LOCK = threading.Lock()


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AVM GPU 引导</title>
<style>
:root {
  --bg: #0f1419; --panel: #1a222c; --text: #e7ecf1; --muted: #8b9aab;
  --accent: #3dd6c6; --warn: #e6b84d; --bad: #e85d5d; --ok: #5dcf7a; --line: #2a3542;
}
* { box-sizing: border-box; }
body {
  margin: 0; font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
  background: radial-gradient(1200px 600px at 10% -10%, #1b3a3a 0%, var(--bg) 45%);
  color: var(--text); min-height: 100vh;
}
header {
  padding: 1.25rem 1.5rem 0.5rem; display:flex; gap:1rem; flex-wrap:wrap;
  align-items: baseline; justify-content: space-between;
}
header h1 { margin:0; font-size:1.4rem; letter-spacing:0.02em; }
header .sub { color: var(--muted); font-size:0.9rem; }
main {
  display:grid; grid-template-columns: 340px 1fr; gap:1rem;
  padding: 0.75rem 1.5rem 1.5rem;
}
@media (max-width: 960px) { main { grid-template-columns: 1fr; } }
.panel {
  background: color-mix(in srgb, var(--panel) 92%, black);
  border: 1px solid var(--line); border-radius: 12px; padding: 1rem;
}
.steps { display:flex; flex-direction:column; gap:0.5rem; }
.step {
  border:1px solid var(--line); border-radius:10px; padding:0.75rem;
  cursor:pointer; background:#141b22;
}
.step.active { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
.step h3 { margin:0 0 0.25rem; font-size:0.95rem; }
.step p { margin:0; color:var(--muted); font-size:0.8rem; }
.row { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.75rem; }
button {
  appearance:none; border:1px solid var(--line); background:#243040; color:var(--text);
  border-radius:8px; padding:0.55rem 0.85rem; cursor:pointer; font:inherit;
}
button.primary { background: #1f6f68; border-color:#2f9a90; }
button:disabled { opacity:0.45; cursor:not-allowed; }
#streamWrap {
  background:#000; border-radius:12px; overflow:hidden; border:1px solid var(--line);
  min-height: 360px; display:flex; align-items:center; justify-content:center;
  position: relative;
}
#video {
  max-width:100%; width:100%; display:block; background:#000; min-height:320px;
}
#streamPlaceholder {
  position:absolute; color:var(--muted); font-size:0.95rem; pointer-events:none;
}
.meta { display:flex; flex-wrap:wrap; gap:0.75rem; margin-top:0.75rem; color:var(--muted); font-size:0.85rem; }
.badge { padding:0.15rem 0.5rem; border-radius:999px; border:1px solid var(--line); }
.badge.ok { color:var(--ok); border-color:var(--ok); }
.badge.bad { color:var(--bad); border-color:var(--bad); }
.badge.warn { color:var(--warn); border-color:var(--warn); }
pre {
  background:#0c1015; border:1px solid var(--line); border-radius:8px;
  padding:0.75rem; overflow:auto; font-size:0.78rem; max-height:220px;
}
.note { color:var(--warn); font-size:0.82rem; margin-top:0.75rem; line-height:1.4; }
#logBox {
  background:#0c1015; border:1px solid var(--line); border-radius:8px;
  padding:0.75rem; overflow:auto; font-size:0.72rem; max-height:280px;
  white-space:pre-wrap; word-break:break-word; color:#b7c4d1; margin-top:0.5rem;
}
.cfg {
  margin-top:0.75rem; border-top:1px solid var(--line); padding-top:0.75rem;
  font-size:0.78rem;
}
.cfg h3 { margin:0 0 0.5rem; font-size:0.9rem; color:var(--muted); }
.cfg label { display:block; color:var(--muted); margin:0.35rem 0 0.15rem; }
.cfg input, .cfg select {
  width:100%; background:#0c1015; color:var(--text); border:1px solid var(--line);
  border-radius:6px; padding:0.35rem 0.5rem; font:inherit;
}
.cfg .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:0.4rem; }
.cfg details { margin-top:0.5rem; }
.cfg summary { cursor:pointer; color:var(--accent); }
</style>
</head>
<body>
<header>
  <div>
    <h1>AVM GPU 引导</h1>
    <div class="sub">CUDA 热路径 · WebRTC 推流</div>
  </div>
  <div class="sub" id="cudaLine">cuda…</div>
</header>
<main>
  <section class="panel">
    <div class="steps">
      <div class="step active" data-step="status" onclick="selectStep('status')">
        <h3>0. 状态检查</h3>
        <p>CUDA / 内参 / 外参质量</p>
      </div>
      <div class="step" data-step="intrinsics" onclick="selectStep('intrinsics')">
        <h3>1. 内参标定</h3>
        <p>WebRTC 推流 · SPACE 抓拍</p>
      </div>
      <div class="step" data-step="extrinsics" onclick="selectStep('extrinsics')">
        <h3>2. 外参标定</h3>
        <p>WebRTC · 稳定后连拍均值</p>
      </div>
      <div class="step" data-step="seam" onclick="selectStep('seam')">
        <h3>2b. 接缝精修</h3>
        <p>重叠区放板 · 微调从路 H</p>
      </div>
      <div class="step" data-step="preview" onclick="selectStep('preview')">
        <h3>3. 去畸变预览</h3>
        <p>GPU undistort · WebRTC</p>
      </div>
      <div class="step" data-step="bev" onclick="selectStep('bev')">
        <h3>4. 实时 BEV</h3>
        <p>GPU stitch · WebRTC</p>
      </div>
    </div>
    <div class="row" id="actions"></div>
    <p class="note">外参完成后可用「接缝精修」：在两路重叠区放板，锁参考路、微调从路 H，修正卷尺 near_m 误差。</p>
    <div class="cfg">
      <h3>标定配置（写回 config/*.json）</h3>
      <div class="grid2">
        <div><label>棋盘列×行（内角点）</label>
          <div class="grid2">
            <input id="cfg_cols" type="number" min="2" step="1"/>
            <input id="cfg_rows" type="number" min="2" step="1"/>
          </div>
        </div>
        <div><label>格宽 square_size_m</label>
          <input id="cfg_square" type="number" min="0.001" step="0.001"/>
        </div>
      </div>
      <div class="grid2">
        <div><label>检测分辨率宽度（每次都使用）</label>
          <input id="cfg_detect_w" type="number" min="320" step="160"/>
        </div>
        <div><label>检测 interval_ms / CPU占空比</label>
          <div class="grid2">
            <input id="cfg_detect_iv" type="number" min="50" step="50"/>
            <input id="cfg_detect_duty" type="number" min="0.05" max="1" step="0.05"/>
          </div>
        </div>
      </div>
      <div class="grid2">
        <div><label>extrinsic_balance</label>
          <input id="cfg_balance" type="number" min="0.1" max="1" step="0.05"/>
        </div>
        <div><label>BEV scale_px_per_m</label>
          <input id="cfg_scale" type="number" min="10" step="10"/>
        </div>
      </div>
      <div class="grid2">
        <div><label>stable_frames / 自动锁定</label>
          <div class="grid2">
            <input id="cfg_stable" type="number" min="1" step="1"/>
            <select id="cfg_autolock">
              <option value="1">自动锁定</option>
              <option value="0">手动 SPACE</option>
            </select>
          </div>
        </div>
        <div><label>burst_frames / min_ok</label>
          <div class="grid2">
            <input id="cfg_burst" type="number" min="1" step="1"/>
            <input id="cfg_burst_min" type="number" min="1" step="1"/>
          </div>
        </div>
      </div>
      <div class="grid2">
        <div><label>内参 min / target 张数</label>
          <div class="grid2">
            <input id="cfg_imin" type="number" min="3" step="1"/>
            <input id="cfg_itarget" type="number" min="3" step="1"/>
          </div>
        </div>
      </div>
      <details>
        <summary>外参 near_m / lateral_m（四路）</summary>
        <div class="grid2" id="cfg_places"></div>
      </details>
      <div class="row">
        <button class="primary" onclick="saveConfig()">保存配置</button>
        <button onclick="loadConfig()">重新加载</button>
      </div>
      <p class="note" id="cfgHint" style="margin-top:0.4rem">改配置后请重新点「开始内参/外参推流」生效。</p>
    </div>
    <h3 style="margin:1rem 0 0.4rem;font-size:0.9rem;color:var(--muted)">状态报告</h3>
    <pre id="report">加载中…</pre>
    <h3 style="margin:1rem 0 0.4rem;font-size:0.9rem;color:var(--muted)">运行日志（前端+服务端）</h3>
    <div id="logBox">等待操作…</div>
  </section>
  <section class="panel">
    <div id="streamWrap">
      <video id="video" autoplay playsinline muted></video>
      <div id="streamPlaceholder">stream offline</div>
    </div>
    <div class="meta">
      <span class="badge" id="modeBadge">mode: idle</span>
      <span class="badge" id="fpsBadge">fps: —</span>
      <span class="badge" id="gpuBadge">gpu: —</span>
      <span class="badge" id="peerBadge">webrtc: 0</span>
      <span class="badge" id="cudaBadge">cuda</span>
    </div>
    <div class="row">
      <button class="primary" onclick="startStream('preview')">GPU 预览流</button>
      <button class="primary" onclick="startStream('bev')">GPU BEV 流</button>
      <button onclick="stopAll()">停止推流</button>
      <button onclick="refresh()">刷新状态</button>
      <button onclick="clearLog()">清空日志</button>
    </div>
    <p class="note" id="streamHint" style="margin-top:0.5rem"></p>
  </section>
</main>
<script>
let currentStep = 'status';
let pc = null;
let clientLogs = [];
let busy = false;

const actions = {
  status: [
    ['跳过内参', "skip('intrinsics')"],
    ['加载配置', "loadConfig()"],
  ],
  intrinsics: [
    ['开始内参推流', "startCalib('intrinsics')"],
    ['跳过步骤', "skip('intrinsics')"],
    ['SPACE 抓拍', "calibCmd('space')"],
    ['ESC 完成本路', "calibCmd('esc')"],
    ['跳过本路相机', "calibCmd('skip')"],
  ],
  extrinsics: [
    ['开始外参推流', "startCalib('extrinsics')"],
    ['跳过步骤', "skip('extrinsics')"],
    ['标 front', "calibCmd('target:front')"],
    ['标 back', "calibCmd('target:back')"],
    ['标 left', "calibCmd('target:left')"],
    ['标 right', "calibCmd('target:right')"],
    ['下一路', "calibCmd('next')"],
    ['重标当前路', "calibCmd('relock')"],
    ['SPACE 锁定READY', "calibCmd('space')"],
    ['ESC 保存外参', "calibCmd('esc')"],
    ['解锁全部', "calibCmd('unlock_all')"],
  ],
  seam: [
    ['开始接缝精修', "startCalib('seam')"],
    ['下一对', "calibCmd('next_pair')"],
    ['交换 ref/slave', "calibCmd('swap')"],
    ['front+left', "calibCmd('pair:front,left')"],
    ['front+right', "calibCmd('pair:front,right')"],
    ['back+left', "calibCmd('pair:back,left')"],
    ['back+right', "calibCmd('pair:back,right')"],
    ['SPACE 精修从路', "calibCmd('space')"],
    ['ESC 写回外参', "calibCmd('esc')"],
  ],
  preview: [],
  bev: [],
};

function fillPlaces(placements) {
  const box = document.getElementById('cfg_places');
  const dirs = ['front','back','left','right'];
  box.innerHTML = dirs.map(d => {
    const p = (placements && placements[d]) || {};
    return `<div>
      <label>${d} near / lateral</label>
      <div class="grid2">
        <input id="cfg_${d}_near" type="number" step="0.01" value="${p.near_m ?? 0.35}"/>
        <input id="cfg_${d}_lat" type="number" step="0.01" value="${p.lateral_m ?? 0}"/>
      </div>
    </div>`;
  }).join('');
}

async function loadConfig() {
  try {
    const { r, j } = await fetchJSON('/api/config', {}, 8000);
    if (!r.ok) throw new Error(j.error || 'load config failed');
    const b = j.chessboard || {};
    const s = j.settings || {};
    document.getElementById('cfg_cols').value = b.pattern_cols ?? 8;
    document.getElementById('cfg_rows').value = b.pattern_rows ?? 6;
    document.getElementById('cfg_square').value = b.square_size_m ?? 0.025;
    document.getElementById('cfg_detect_w').value = s.detect_max_width ?? 1920;
    document.getElementById('cfg_detect_iv').value = s.detect_interval_ms ?? 1000;
    document.getElementById('cfg_detect_duty').value = s.detect_duty ?? 0.25;
    document.getElementById('cfg_balance').value = s.extrinsic_balance ?? 0.8;
    document.getElementById('cfg_stable').value = s.stable_frames ?? 3;
    document.getElementById('cfg_autolock').value = (s.auto_lock ?? true) ? '1' : '0';
    document.getElementById('cfg_burst').value = s.burst_frames ?? 8;
    document.getElementById('cfg_burst_min').value = s.burst_min_ok ?? 5;
    document.getElementById('cfg_imin').value = s.intrinsics_min_frames ?? 15;
    document.getElementById('cfg_itarget').value = s.intrinsics_target_frames ?? 25;
    document.getElementById('cfg_scale').value = s.scale_px_per_m ?? 100;
    fillPlaces(j.placements || {});
    document.getElementById('cfgHint').textContent = '配置已加载自磁盘';
    log('配置已加载');
  } catch (e) {
    log('loadConfig: ' + e.message, 'ERROR');
  }
}

async function saveConfig() {
  const dirs = ['front','back','left','right'];
  const placements = {};
  dirs.forEach(d => {
    placements[d] = {
      near_m: parseFloat(document.getElementById('cfg_'+d+'_near').value),
      lateral_m: parseFloat(document.getElementById('cfg_'+d+'_lat').value),
      orient: 'long-lateral',
    };
  });
  const body = {
    chessboard: {
      pattern_cols: parseInt(document.getElementById('cfg_cols').value, 10),
      pattern_rows: parseInt(document.getElementById('cfg_rows').value, 10),
      square_size_m: parseFloat(document.getElementById('cfg_square').value),
    },
    placements,
    settings: {
      detect_max_width: parseInt(document.getElementById('cfg_detect_w').value, 10),
      detect_interval_ms: parseInt(document.getElementById('cfg_detect_iv').value, 10),
      detect_duty: parseFloat(document.getElementById('cfg_detect_duty').value),
      extrinsic_balance: parseFloat(document.getElementById('cfg_balance').value),
      stable_frames: parseInt(document.getElementById('cfg_stable').value, 10),
      auto_lock: document.getElementById('cfg_autolock').value === '1',
      burst_frames: parseInt(document.getElementById('cfg_burst').value, 10),
      burst_min_ok: parseInt(document.getElementById('cfg_burst_min').value, 10),
      intrinsics_min_frames: parseInt(document.getElementById('cfg_imin').value, 10),
      intrinsics_target_frames: parseInt(document.getElementById('cfg_itarget').value, 10),
      scale_px_per_m: parseFloat(document.getElementById('cfg_scale').value),
    },
  };
  try {
    const { r, j } = await fetchJSON('/api/config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    }, 10000);
    if (!r.ok) throw new Error(j.error || 'save failed');
    document.getElementById('cfgHint').textContent =
      '已保存: ' + (j.changed || []).join(', ') + '。请重新开始推流。';
    log('配置已保存 ' + JSON.stringify(j.changed));
  } catch (e) {
    log('saveConfig: ' + e.message, 'ERROR');
    alert(e.message);
  }
}

function ts() {
  return new Date().toLocaleTimeString('en-GB', { hour12: false });
}
function log(msg, level='INFO') {
  const line = `[${ts()}] [UI/${level}] ${msg}`;
  clientLogs.push(line);
  if (clientLogs.length > 120) clientLogs = clientLogs.slice(-120);
  renderLog();
  console.log(line);
}
function clearLog() { clientLogs = []; renderLog(); }
function renderLog(serverLines) {
  const box = document.getElementById('logBox');
  const srv = (serverLines && serverLines.length)
    ? serverLines.join('\n')
    : (box.dataset.server || '');
  if (serverLines) box.dataset.server = srv;
  const cli = clientLogs.join('\n');
  box.textContent = [srv, '----- UI -----', cli].filter(Boolean).join('\n');
  box.scrollTop = box.scrollHeight;
}

function selectStep(step) {
  currentStep = step;
  document.querySelectorAll('.step').forEach(el => {
    el.classList.toggle('active', el.dataset.step === step);
  });
  const box = document.getElementById('actions');
  const list = actions[step] || [];
  box.innerHTML = list.length
    ? list.map(([label, fn]) => `<button onclick="${fn}">${label}</button>`).join('')
    : '<span style="color:var(--muted);font-size:0.85rem">推流请用右侧按钮</span>';
  fetch('/api/step', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({step})}).catch(()=>{});
}

function setPlaceholder(show, text) {
  const el = document.getElementById('streamPlaceholder');
  el.style.display = show ? 'block' : 'none';
  if (text) el.textContent = text;
}

async function fetchJSON(url, opts={}, timeoutMs=45000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { ...opts, signal: ctrl.signal });
    const text = await r.text();
    let j = {};
    try { j = text ? JSON.parse(text) : {}; } catch (e) {
      throw new Error(`非 JSON 响应 HTTP ${r.status}: ${text.slice(0,120)}`);
    }
    return { r, j };
  } finally {
    clearTimeout(timer);
  }
}

async function refresh() {
  try {
    const { r, j } = await fetchJSON('/api/status', {}, 8000);
    if (!r.ok) { log('status HTTP ' + r.status, 'ERROR'); return; }
    document.getElementById('cudaLine').textContent = j.cuda_line || '';
    document.getElementById('report').textContent = j.report_text || JSON.stringify(j, null, 2);
    renderLog(j.logs || []);
    const s = j.stream || {};
    const w = j.webrtc || {};
    document.getElementById('modeBadge').textContent = 'mode: ' + (s.mode || 'idle');
    document.getElementById('fpsBadge').textContent = 'fps: ' + (s.fps ?? '—');
    document.getElementById('gpuBadge').textContent = 'gpu: ' + (s.gpu_ms ?? '—') + 'ms';
    document.getElementById('peerBadge').textContent = 'webrtc: ' + (w.peers ?? 0);
    const cb = document.getElementById('cudaBadge');
    const cudaOk = !!(j.cuda || s.cuda);
    const streaming = s.mode && s.mode !== 'idle';
    if (!cudaOk) { cb.textContent = 'CUDA OFF'; cb.className = 'badge bad'; }
    else if (streaming && s.pipeline_cuda === false) { cb.textContent = 'CUDA 未进管道'; cb.className = 'badge bad'; }
    else if (streaming) { cb.textContent = 'CUDA 推流中'; cb.className = 'badge ok'; }
    else { cb.textContent = 'CUDA 就绪'; cb.className = 'badge ok'; }
    const hint = document.getElementById('streamHint');
    if (s.error) hint.textContent = '推流错误: ' + s.error;
    else if ((j.extrinsics || {}).status === 'fail')
      hint.textContent = '外参缺失：BEV 不可用。可先开「GPU 预览流」。';
    else if (!streaming)
      hint.textContent = '点「GPU 预览流」→ 自动 WebRTC。若卡住请看左侧日志。';
    else hint.textContent = '';
  } catch (e) {
    log('刷新失败: ' + e.message + '（服务是否在跑？ ./scripts/run_web.sh）', 'ERROR');
    document.getElementById('streamHint').textContent =
      '无法连接后端 /api/status。请在板子上启动: ./scripts/run_web.sh --host 0.0.0.0 --port 8787';
  }
}

function waitIce(pc) {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise(resolve => {
    const t = setTimeout(() => { log('ICE gather timeout, continue'); resolve(); }, 2500);
    pc.addEventListener('icegatheringstatechange', () => {
      if (pc.iceGatheringState === 'complete') { clearTimeout(t); resolve(); }
    });
  });
}

async function connectWebRTC() {
  if (pc) { try { pc.close(); } catch(e) {} pc = null; }
  log('WebRTC: 创建 RTCPeerConnection');
  pc = new RTCPeerConnection({
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
  });
  pc.addTransceiver('video', { direction: 'recvonly' });
  pc.ontrack = (ev) => {
    log('WebRTC: ontrack ' + (ev.track && ev.track.kind));
    const v = document.getElementById('video');
    v.srcObject = ev.streams[0];
    v.play().catch(err => log('video.play: ' + err.message, 'WARN'));
    setPlaceholder(false);
  };
  pc.onconnectionstatechange = () => {
    log('WebRTC connectionState=' + (pc && pc.connectionState));
    if (pc && ['failed','disconnected','closed'].includes(pc.connectionState)) {
      setPlaceholder(true, 'webrtc ' + pc.connectionState);
    }
  };
  pc.oniceconnectionstatechange = () => {
    log('WebRTC ice=' + (pc && pc.iceConnectionState));
  };
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  log('WebRTC: local offer sdp_len=' + (offer.sdp || '').length);
  await waitIce(pc);
  log('WebRTC: POST /api/webrtc …');
  const { r, j } = await fetchJSON('/api/webrtc', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
    }),
  }, 30000);
  if (!r.ok) throw new Error(j.error || ('webrtc HTTP ' + r.status));
  log('WebRTC: got answer sdp_len=' + (j.sdp || '').length);
  await pc.setRemoteDescription(j);
  log('WebRTC: setRemoteDescription OK');
}

async function startStream(mode) {
  if (busy) { log('已有启动进行中，忽略重复点击', 'WARN'); return; }
  busy = true;
  setPlaceholder(true, 'starting…');
  log('点击启动 mode=' + mode);
  try {
    log('POST /api/stream/start …（打开相机可能要几秒）');
    setPlaceholder(true, 'opening cameras…');
    const { r, j } = await fetchJSON('/api/stream/start?mode=' + mode, {method:'POST'}, 60000);
    if (!r.ok) throw new Error(j.error || ('start HTTP ' + r.status));
    log('hub started: mode=' + ((j.stream||{}).mode) + ' cameras=' + JSON.stringify((j.stream||{}).cameras));
    setPlaceholder(true, 'webrtc negotiating…');
    await connectWebRTC();
    selectStep(mode === 'bev' ? 'bev' : 'preview');
    setPlaceholder(false);
    log('启动完成');
  } catch (e) {
    const msg = (e && e.name === 'AbortError')
      ? '请求超时（服务无响应或开相机卡住）'
      : (e.message || String(e));
    log('启动失败: ' + msg, 'ERROR');
    setPlaceholder(true, 'failed');
    alert('启动失败: ' + msg);
  } finally {
    busy = false;
    refresh();
  }
}

async function stopAll() {
  log('停止推流');
  if (pc) { try { pc.close(); } catch(e) {} pc = null; }
  const v = document.getElementById('video');
  v.srcObject = null;
  try {
    await fetchJSON('/api/stream/stop', {method:'POST'}, 10000);
  } catch (e) {
    log('stop 请求失败: ' + e.message, 'WARN');
  }
  setPlaceholder(true, 'stream offline');
  refresh();
}

async function startCalib(kind) {
  const mode = ({
    extrinsics: 'calib_extrinsics',
    seam: 'calib_seam',
    intrinsics: 'calib_intrinsics',
  })[kind] || 'calib_intrinsics';
  log('启动标定推流 mode=' + mode);
  await startStream(mode);
}

async function calibCmd(name) {
  log('calib action ' + name);
  try {
    const { r, j } = await fetchJSON(
      '/api/calib/action?cmd=' + encodeURIComponent(name),
      {method:'POST'},
      120000
    );
    if (!r.ok) {
      log('calib fail: ' + (j.error || r.status), 'ERROR');
      alert(j.error || 'calib failed');
    } else {
      log('calib ok: ' + (j.message || JSON.stringify(j)));
      if (j.done) log('标定流程结束');
    }
  } catch (e) {
    log('calibCmd: ' + e.message, 'ERROR');
    alert(e.message);
  }
  refresh();
}

async function skip(kind) {
  log('skip ' + kind);
  try {
    await fetchJSON('/api/skip?what=' + kind, {method:'POST'}, 8000);
  } catch (e) { log(e.message, 'ERROR'); }
  if (kind === 'intrinsics') selectStep('extrinsics');
  if (kind === 'extrinsics') selectStep('preview');
  refresh();
}

async function cmd(name) {
  log('cmd(file) ' + name);
  try { await fetchJSON('/api/cmd?name=' + name, {method:'POST'}, 5000); }
  catch (e) { log(e.message, 'ERROR'); }
}

log('页面加载 ' + location.href);
selectStep('status');
loadConfig();
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


def _json_bytes(obj: dict, code: int = 200) -> tuple[bytes, int, str]:
    return (
        json.dumps(obj, ensure_ascii=False).encode("utf-8"),
        code,
        "application/json; charset=utf-8",
    )


def _status_payload() -> dict:
    intr = check_intrinsics_quality(CALIB_DIR)
    extr = check_extrinsics_quality(CALIB_DIR / "extrinsics.json")
    stream = HUB.status() if HUB else {"mode": "idle"}
    webrtc = WEBRTC.status() if WEBRTC else {"peers": 0}
    lines = [f"CUDA: {cuda_status_line()}", f"内参: {intr['status']}"]
    for d, det in intr.get("details", {}).items():
        rms = det.get("rms")
        rms_s = f"{float(rms):.3f}" if rms is not None else "N/A"
        lines.append(f"  {d}: {det.get('status')} RMS={rms_s}")
    lines.append(f"外参: {extr['status']}")
    for d, det in extr.get("details", {}).items():
        lines.append(f"  {d}: {det.get('status')}")
    for w in extr.get("global_warnings", []):
        lines.append(f"  ! {w}")
    lines.append(f"WebRTC peers: {webrtc.get('peers', 0)}")
    calib = (stream or {}).get("calib") or {}
    if calib.get("kind"):
        lines.append(
            f"标定会话: {calib.get('kind')} dir={calib.get('direction')} "
            f"captured={calib.get('captured')} locked={calib.get('locked')} "
            f"msg={calib.get('message')}"
        )
        if calib.get("sequential"):
            tgt = calib.get("target") or "-"
            got = (calib.get("stable_streak") or {}).get(tgt, 0)
            lines.append(
                f"  逐路标定 当前={tgt} 稳定={got}/{calib.get('stable_need')} "
                f"待标={calib.get('pending')}"
            )
        if calib.get("kind") == "seam":
            lines.append(
                f"  接缝精修 ref={calib.get('seam_ref')} slave={calib.get('seam_slave')} "
                f"last={calib.get('seam_last')}"
            )
    with STATE_LOCK:
        lines.append(
            f"向导 step={STATE['step']} skip_intr={STATE['skipped_intrinsics']} "
            f"skip_extr={STATE['skipped_extrinsics']}"
        )
        if STATE.get("message"):
            lines.append(STATE["message"])
        proc = STATE.get("calib_proc")
        if proc is not None:
            lines.append(f"calib_proc pid={proc.pid} running={proc.poll() is None}")
    return {
        "cuda": cuda_available(),
        "cuda_line": cuda_status_line(),
        "intrinsics": intr,
        "extrinsics": extr,
        "stream": stream,
        "webrtc": webrtc,
        "logs": LOG.dump(100),
        "report_text": "\n".join(lines),
        "message": STATE.get("message", ""),
        "control_file": str(CONTROL_FILE),
    }


def _stop_calib_proc() -> None:
    with STATE_LOCK:
        proc = STATE.get("calib_proc")
        STATE["calib_proc"] = None
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


def _start_calib(kind: str) -> dict:
    if HUB and HUB.mode != "idle":
        if WEBRTC:
            WEBRTC.close_all()
        HUB.stop()
    _stop_calib_proc()
    ensure_control_file(CONTROL_FILE)
    env = _cuda_env()
    env["AVM_CALIB_CONTROL_FILE"] = str(CONTROL_FILE)
    if kind == "intrinsics":
        cmd = [sys.executable, "-m", "avm.calibrate_intrinsics", "--calibrate"]
    elif kind == "extrinsics":
        cmd = [
            sys.executable,
            "-m",
            "avm.calibrate_extrinsics",
            "--capture",
            "--extrinsic-balance",
            "0.8",
        ]
    else:
        raise ValueError(kind)
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    with STATE_LOCK:
        STATE["calib_proc"] = proc
        STATE["message"] = (
            f"已启动 {kind} 标定 pid={proc.pid}（需本机 DISPLAY；Web 按钮注入按键）"
        )
    return {"ok": True, "pid": proc.pid}


class Handler(BaseHTTPRequestHandler):
    server_version = "AVMGpuWeb/0.1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, body: bytes, code: int = 200, content_type: str = "text/plain") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in ("/", "/index.html"):
                self._send(HTML_PAGE.encode("utf-8"), 200, "text/html; charset=utf-8")
                return
            if path == "/api/status":
                body, code, ctype = _json_bytes(_status_payload())
                self._send(body, code, ctype)
                return
            if path == "/api/health":
                body, code, ctype = _json_bytes(
                    {
                        "ok": True,
                        "cuda": cuda_available(),
                        "mode": HUB.mode if HUB else "idle",
                        "webrtc_peers": WEBRTC.peer_count() if WEBRTC else 0,
                    }
                )
                self._send(body, code, ctype)
                return
            if path == "/api/config":
                body, code, ctype = _json_bytes(load_all_config())
                self._send(body, code, ctype)
                return
            self._send(b"not found", 404)
        except Exception:
            self._send(traceback.format_exc().encode(), 500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path == "/api/step":
                data = self._read_json()
                with STATE_LOCK:
                    STATE["step"] = str(data.get("step") or STATE["step"])
                body, code, ctype = _json_bytes({"ok": True, "step": STATE["step"]})
                self._send(body, code, ctype)
                return
            if path == "/api/config":
                data = self._read_json()
                try:
                    out = save_all_config(data)
                    # 若标定会话仍在跑，热更新可立刻生效的检测参数；
                    # balance/placements/maps 需重新「开始外参推流」重建。
                    if HUB is not None and getattr(HUB, "calib", None) is not None:
                        try:
                            HUB.calib.reload_settings()
                        except Exception as exc:
                            LOG.warn(f"calib reload_settings: {exc}")
                    LOG.info(f"API config saved changed={out.get('changed')}")
                    body, code, ctype = _json_bytes(out)
                except Exception as exc:
                    LOG.error(f"API config save FAIL: {exc}")
                    body, code, ctype = _json_bytes({"ok": False, "error": str(exc)}, 400)
                self._send(body, code, ctype)
                return
            if path == "/api/skip":
                what = (qs.get("what") or ["intrinsics"])[0]
                with STATE_LOCK:
                    if what == "intrinsics":
                        STATE["skipped_intrinsics"] = True
                        STATE["message"] = "已跳过内参"
                    elif what == "extrinsics":
                        STATE["skipped_extrinsics"] = True
                        STATE["message"] = "已跳过外参"
                body, code, ctype = _json_bytes({"ok": True})
                self._send(body, code, ctype)
                return
            if path == "/api/stream/start":
                mode = (qs.get("mode") or ["preview"])[0]
                assert HUB is not None
                LOG.info(f"API stream/start mode={mode}")
                if WEBRTC:
                    WEBRTC.close_all()
                try:
                    st = HUB.start(mode)
                    LOG.info(f"API stream/start OK cameras={st.get('cameras')}")
                    body, code, ctype = _json_bytes({"ok": True, "stream": st})
                except Exception as exc:
                    LOG.error(f"API stream/start FAIL: {exc}")
                    body, code, ctype = _json_bytes({"ok": False, "error": str(exc)}, 400)
                self._send(body, code, ctype)
                return
            if path == "/api/stream/stop":
                LOG.info("API stream/stop")
                if WEBRTC:
                    WEBRTC.close_all()
                if HUB:
                    HUB.stop()
                body, code, ctype = _json_bytes({"ok": True})
                self._send(body, code, ctype)
                return
            if path == "/api/webrtc":
                data = self._read_json()
                assert WEBRTC is not None
                LOG.info("API webrtc offer received")
                try:
                    ans = WEBRTC.handle_offer(
                        str(data.get("sdp") or ""),
                        str(data.get("type") or "offer"),
                    )
                    body, code, ctype = _json_bytes(ans)
                except Exception as exc:
                    LOG.error(f"API webrtc FAIL: {exc}")
                    body, code, ctype = _json_bytes({"error": str(exc)}, 400)
                self._send(body, code, ctype)
                return
            if path == "/api/logs":
                body, code, ctype = _json_bytes({"logs": LOG.dump(120)})
                self._send(body, code, ctype)
                return
            if path == "/api/cmd":
                name = (qs.get("name") or [""])[0].lower()
                if name not in CMD_TO_KEY:
                    body, code, ctype = _json_bytes({"error": f"unknown cmd {name}"}, 400)
                else:
                    push_control_cmd(CONTROL_FILE, name)
                    body, code, ctype = _json_bytes({"ok": True, "cmd": name})
                self._send(body, code, ctype)
                return
            if path == "/api/calib/action":
                cmd = (qs.get("cmd") or [""])[0]
                assert HUB is not None
                LOG.info(f"API calib/action cmd={cmd}")
                try:
                    out = HUB.calib_action(cmd)
                    code = 200 if out.get("ok", True) else 400
                    if "ok" not in out:
                        out = {"ok": True, **out}
                    body, code, ctype = _json_bytes(out, code)
                except Exception as exc:
                    LOG.error(f"calib/action FAIL: {exc}")
                    body, code, ctype = _json_bytes({"ok": False, "error": str(exc)}, 400)
                self._send(body, code, ctype)
                return
            if path == "/api/calib/start":
                # 兼容旧入口：改为启动 WebRTC 标定流（不再开本机窗口抢相机）
                kind = (qs.get("kind") or ["intrinsics"])[0]
                mode = {
                    "extrinsics": "calib_extrinsics",
                    "seam": "calib_seam",
                }.get(kind, "calib_intrinsics")
                assert HUB is not None
                if WEBRTC:
                    WEBRTC.close_all()
                try:
                    st = HUB.start(mode)
                    body, code, ctype = _json_bytes({"ok": True, "stream": st, "mode": mode})
                except Exception as exc:
                    body, code, ctype = _json_bytes({"ok": False, "error": str(exc)}, 400)
                self._send(body, code, ctype)
                return
            if path == "/api/calib/dump":
                if HUB is None or getattr(HUB, "calib", None) is None:
                    body, code, ctype = _json_bytes(
                        {"ok": False, "error": "无标定会话"}, 400
                    )
                else:
                    body, code, ctype = _json_bytes(HUB.calib.request_dump())
                self._send(body, code, ctype)
                return
            if path == "/api/calib/stop":
                _stop_calib_proc()
                if WEBRTC:
                    WEBRTC.close_all()
                if HUB:
                    HUB.stop()
                body, code, ctype = _json_bytes({"ok": True})
                self._send(body, code, ctype)
                return
            self._send(b"not found", 404)
        except Exception:
            self._send(traceback.format_exc().encode(), 500)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AVM GPU Web 引导 (WebRTC)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--display-width", type=int, default=800)
    p.add_argument("--allow-cpu", action="store_true", help="允许无 CUDA（不推荐）")
    return p.parse_args()


def main() -> None:
    global HUB, WEBRTC
    args = parse_args()
    os.chdir(ROOT)
    (ROOT / "output").mkdir(parents=True, exist_ok=True)
    ensure_control_file(CONTROL_FILE)
    log_cuda_status()
    if not cuda_available() and not args.allow_cpu:
        print("ERROR: CUDA 不可用。请 source scripts/env_opencv_cuda.sh")
        print("       或显式 --allow-cpu（不推荐）")
        raise SystemExit(2)
    HUB = GpuStreamHub(
        display_width=args.display_width,
        require_cuda=not args.allow_cpu,
    )
    WEBRTC = WebRtcBridge()
    WEBRTC.set_hub(HUB)
    LOG.info(f"web_server listen {args.host}:{args.port} cuda={cuda_available()}")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print("=" * 60)
    print(f"  AVM GPU Web  http://{args.host}:{args.port}/")
    print(f"  health       http://127.0.0.1:{args.port}/api/health")
    print(f"  transport    WebRTC (aiortc)")
    print(f"  CUDA         {cuda_status_line()}")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down…")
    finally:
        if WEBRTC:
            WEBRTC.close_all()
        if HUB:
            HUB.stop()
        _stop_calib_proc()
        httpd.server_close()


if __name__ == "__main__":
    main()
