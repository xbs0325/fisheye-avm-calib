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
import time
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
    "last_probe": None,
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
  display:grid; grid-template-columns: 280px minmax(0, 1fr) 340px; gap:1rem;
  padding: 0.75rem 1.5rem 1.5rem; align-items: stretch;
  min-height: calc(100vh - 4.5rem);
}
@media (max-width: 1100px) {
  main { grid-template-columns: 1fr; min-height: 0; }
}
.panel {
  background: color-mix(in srgb, var(--panel) 92%, black);
  border: 1px solid var(--line); border-radius: 12px; padding: 1rem;
}
.panel-left, .panel-right {
  display:flex; flex-direction:column; gap:0.75rem;
  max-height: calc(100vh - 5rem); overflow: hidden;
}
.panel-left .steps { flex: 0 0 auto; }
.panel-left .report-wrap, .panel-right .log-wrap {
  flex: 1 1 auto; min-height: 0; display:flex; flex-direction:column;
}
.panel-right .cfg {
  flex: 0 1 auto; max-height: 48%; overflow: auto;
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
.row.stack { flex-direction: column; }
.row.stack button { width: 100%; text-align: left; }
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
pre#report {
  background:#0c1015; border:1px solid var(--line); border-radius:8px;
  padding:0.75rem; overflow:auto; font-size:0.78rem;
  flex: 1 1 auto; min-height: 160px; margin: 0;
}
.note { color:var(--warn); font-size:0.82rem; margin-top:0.75rem; line-height:1.4; }
.side-h {
  margin:0 0 0.4rem; font-size:0.9rem; color:var(--muted);
  display:flex; align-items:center; justify-content:space-between; gap:0.5rem;
}
.side-h button { padding:0.25rem 0.55rem; font-size:0.75rem; }
#logBox {
  background:#0c1015; border:1px solid var(--line); border-radius:8px;
  padding:0.75rem; overflow:auto; font-size:0.72rem;
  flex: 1 1 auto; min-height: 220px;
  white-space:pre-wrap; word-break:break-word; color:#b7c4d1; margin:0;
}
.cfg { font-size:0.78rem; }
.cfg label { display:block; color:var(--muted); margin:0.35rem 0 0.15rem; }
.cfg input, .cfg select {
  width:100%; background:#0c1015; color:var(--text); border:1px solid var(--line);
  border-radius:6px; padding:0.35rem 0.5rem; font:inherit;
}
.cfg .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:0.4rem; }
.cfg details { margin-top:0.5rem; }
.cfg summary { cursor:pointer; color:var(--accent); }
header .lang {
  display:flex; align-items:center; gap:0.5rem; color:var(--muted); font-size:0.9rem;
}
header .lang select {
  background:#0c1015; color:var(--text); border:1px solid var(--line);
  border-radius:6px; padding:0.35rem 0.55rem; font:inherit;
}
#actions .empty { color:var(--muted); font-size:0.85rem; }
</style>
</head>
<body>
<header>
  <div>
    <h1 data-i18n="title">AVM GPU 引导</h1>
    <div class="sub" data-i18n="subtitle">CUDA 热路径 · WebRTC 推流</div>
  </div>
  <div class="lang">
    <label for="langSelect" data-i18n="lang_label">语言</label>
    <select id="langSelect" onchange="setLang(this.value)">
      <option value="zh">中文</option>
      <option value="en">English</option>
    </select>
  </div>
</header>
<main>
  <section class="panel panel-left">
    <div class="steps">
      <div class="step active" data-step="status" onclick="selectStep('status')">
        <h3 data-i18n="step_status_h">0. 状态检查</h3>
        <p data-i18n="step_status_p">CUDA / 内参 / 外参质量</p>
      </div>
      <div class="step" data-step="intrinsics" onclick="selectStep('intrinsics')">
        <h3 data-i18n="step_intr_h">1. 内参标定</h3>
        <p data-i18n="step_intr_p">WebRTC · SPACE 抓拍</p>
      </div>
      <div class="step" data-step="extrinsics" onclick="selectStep('extrinsics')">
        <h3 data-i18n="step_ext_h">2. 外参标定</h3>
        <p data-i18n="step_ext_p">稳定后自动锁定 · 连拍均值</p>
      </div>
      <div class="step" data-step="seam" onclick="selectStep('seam')">
        <h3 data-i18n="step_seam_h">2b. 接缝精修</h3>
        <p data-i18n="step_seam_p">重叠区放板 · 自动精修从路</p>
      </div>
      <div class="step" data-step="preview" onclick="selectStep('preview')">
        <h3 data-i18n="step_prev_h">3. 去畸变预览</h3>
        <p data-i18n="step_prev_p">GPU undistort · WebRTC</p>
      </div>
      <div class="step" data-step="bev" onclick="selectStep('bev')">
        <h3 data-i18n="step_bev_h">4. 实时 BEV</h3>
        <p data-i18n="step_bev_p">GPU stitch · WebRTC</p>
      </div>
    </div>
    <p class="note" data-i18n="side_note">点左侧切换步骤；中间按钮开/关推流。步骤操作在视频下方。</p>
    <div class="report-wrap">
      <h3 class="side-h" data-i18n="report_title">状态报告</h3>
      <pre id="report">…</pre>
    </div>
  </section>

  <section class="panel">
    <div id="streamWrap">
      <video id="video" autoplay playsinline muted></video>
      <div id="streamPlaceholder" data-i18n="stream_offline">stream offline</div>
    </div>
    <div class="meta">
      <span class="badge" id="modeBadge">mode: idle</span>
      <span class="badge" id="fpsBadge">fps: —</span>
      <span class="badge" id="gpuBadge">gpu: —</span>
      <span class="badge" id="peerBadge">webrtc: 0</span>
      <span class="badge" id="cudaBadge">cuda</span>
    </div>
    <div class="row" style="align-items:center">
      <button id="streamToggle" class="primary" onclick="toggleStream()" data-i18n="btn_start">开始推流</button>
      <button onclick="refresh()" data-i18n="btn_refresh">刷新状态</button>
    </div>
    <div class="row" id="actions"></div>
    <p class="note" id="streamHint" style="margin-top:0.5rem"></p>
  </section>

  <section class="panel panel-right">
    <div class="cfg">
      <h3 class="side-h" data-i18n="cfg_title">标定配置（写回 config/*.json）</h3>
      <div class="grid2">
        <div><label data-i18n="cfg_pattern">棋盘列×行（内角点）</label>
          <div class="grid2">
            <input id="cfg_cols" type="number" min="2" step="1"/>
            <input id="cfg_rows" type="number" min="2" step="1"/>
          </div>
        </div>
        <div><label data-i18n="cfg_square">格宽 square_size_m</label>
          <input id="cfg_square" type="number" min="0.001" step="0.001"/>
        </div>
      </div>
      <div class="grid2">
        <div><label data-i18n="cfg_detect_w">检测分辨率宽度</label>
          <input id="cfg_detect_w" type="number" min="320" step="160"/>
        </div>
        <div><label data-i18n="cfg_interval">interval_ms / CPU占空比</label>
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
        <div><label data-i18n="cfg_stable">stable_frames / 自动锁定</label>
          <div class="grid2">
            <input id="cfg_stable" type="number" min="1" step="1"/>
            <select id="cfg_autolock">
              <option value="1" data-i18n="opt_autolock">自动锁定</option>
              <option value="0" data-i18n="opt_manual">手动 SPACE</option>
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
        <div><label data-i18n="cfg_intr_n">内参 min / target 张数</label>
          <div class="grid2">
            <input id="cfg_imin" type="number" min="3" step="1"/>
            <input id="cfg_itarget" type="number" min="3" step="1"/>
          </div>
        </div>
      </div>
      <details open>
        <summary data-i18n="cfg_cam_title">相机设备 / 分辨率</summary>
        <div class="grid2">
          <div><label data-i18n="cfg_cam_wh">采集 width × height</label>
            <div class="grid2">
              <input id="cfg_cam_w" type="number" min="160" step="16"/>
              <input id="cfg_cam_h" type="number" min="120" step="16"/>
            </div>
          </div>
          <div><label data-i18n="cfg_cam_fourcc">fourcc / backend</label>
            <div class="grid2">
              <input id="cfg_cam_fourcc" type="text" placeholder="YUYV"/>
              <select id="cfg_cam_backend">
                <option value="v4l2">v4l2</option>
                <option value="gstreamer">gstreamer</option>
              </select>
            </div>
          </div>
        </div>
        <div class="grid2" id="cfg_cam_devs"></div>
        <p class="note" data-i18n="cfg_cam_hint">改分辨率后需重做内参/外参。保存后可在状态页 Probe。</p>
      </details>
      <details>
        <summary data-i18n="cfg_places">外参 near_m / lateral_m（四路）</summary>
        <div class="grid2" id="cfg_places"></div>
      </details>
      <div class="row">
        <button class="primary" onclick="saveConfig()" data-i18n="btn_save_cfg">保存配置</button>
        <button onclick="loadConfig()" data-i18n="btn_reload_cfg">重新加载</button>
      </div>
      <p class="note" id="cfgHint" style="margin-top:0.4rem" data-i18n="cfg_hint">改配置后请重新开始对应推流生效。</p>
    </div>
    <div class="log-wrap">
      <h3 class="side-h">
        <span data-i18n="log_title">运行日志（前端+服务端）</span>
        <button type="button" onclick="clearLog()" data-i18n="btn_clear_log">清空日志</button>
      </h3>
      <div id="logBox">…</div>
    </div>
  </section>
</main>
<script>
let currentStep = 'status';
let pc = null;
let clientLogs = [];
let busy = false;
let lang = localStorage.getItem('avm_lang') || 'zh';
let lastStreaming = false;

const I18N = {
  zh: {
    title: 'AVM GPU 引导',
    subtitle: 'CUDA 热路径 · WebRTC 推流',
    lang_label: '语言',
    step_status_h: '0. 状态检查',
    step_status_p: 'CUDA / 内参 / 外参质量',
    step_intr_h: '1. 内参标定',
    step_intr_p: 'WebRTC · SPACE 抓拍',
    step_ext_h: '2. 外参标定',
    step_ext_p: '稳定后自动锁定 · 连拍均值',
    step_seam_h: '2b. 接缝精修',
    step_seam_p: '重叠区放板 · 自动精修从路',
    step_prev_h: '3. 去畸变预览',
    step_prev_p: 'GPU undistort · WebRTC',
    step_bev_h: '4. 实时 BEV',
    step_bev_p: 'GPU stitch · WebRTC',
    stream_ctrl: '推流控制',
    btn_preview: 'GPU 预览流',
    btn_bev: 'GPU BEV 流',
    btn_start: '开始推流',
    btn_starting: '启动中…',
    btn_stop: '停止推流',
    btn_refresh: '刷新状态',
    btn_clear_log: '清空日志',
    side_note: '点左侧切换步骤；中间按钮开/关推流。步骤操作在视频下方。',
    stream_offline: 'stream offline',
    cfg_title: '标定配置（写回 config/*.json）',
    cfg_pattern: '棋盘列×行（内角点）',
    cfg_square: '格宽 square_size_m',
    cfg_detect_w: '检测分辨率宽度',
    cfg_interval: 'interval_ms / CPU占空比',
    cfg_stable: 'stable_frames / 自动锁定',
    opt_autolock: '自动锁定',
    opt_manual: '手动 SPACE',
    cfg_intr_n: '内参 min / target 张数',
    cfg_places: '外参 near_m / lateral_m（四路）',
    cfg_cam_title: '相机设备 / 分辨率',
    cfg_cam_wh: '采集 width × height',
    cfg_cam_fourcc: 'fourcc / backend',
    cfg_cam_hint: '改分辨率后需重做内参/外参。保存后可在状态页 Probe。',
    btn_save_cfg: '保存配置',
    btn_reload_cfg: '重新加载',
    cfg_hint: '改配置后请重新开始对应推流生效。',
    report_title: '状态报告',
    loading: '加载中…',
    log_title: '运行日志（前端+服务端）',
    log_wait: '等待操作…',
    act_refresh: '刷新状态报告',
    act_load_cfg: '加载配置',
    act_probe: 'Probe 相机',
    act_smoke: 'Smoke 开流',
    act_start_intr: '开始内参推流',
    act_space: 'SPACE 抓拍',
    act_finish_cam: '完成本路',
    act_skip_cam: '跳过本路相机',
    act_start_ext: '开始外参推流',
    act_target_front: '标 front',
    act_target_back: '标 back',
    act_target_left: '标 left',
    act_target_right: '标 right',
    act_relock: '重标当前路',
    act_unlock: '解锁全部',
    act_save_params: '保存标定参数',
    act_start_seam: '开始接缝精修',
    act_swap: '交换 ref/slave',
    act_edit_ext: '修改外参',
    act_start_prev: '开始去畸变预览',
    act_start_bev: '开始实时 BEV',
    empty_actions: '本步骤无额外操作',
    cfg_loaded: '配置已加载自磁盘',
    cfg_saved: '已保存',
    cfg_restart: '。请重新开始推流。',
    cuda_off: 'CUDA OFF',
    cuda_not_pipe: 'CUDA 未进管道',
    cuda_streaming: 'CUDA 推流中',
    cuda_ready: 'CUDA 就绪',
    err_stream: '推流错误: ',
    hint_no_ext: '外参缺失：BEV 不可用。可先开「GPU 预览流」。',
    hint_idle: '选好左侧步骤后，点中间「开始推流」。再点一次可停止。',
    hint_no_backend: '无法连接后端 /api/status。请在板子上启动: ./scripts/run_web.sh --host 0.0.0.0 --port 8787',
    refresh_fail: '刷新失败: ',
    waiting_server: '等待服务器（开相机 / WebRTC）…',
    busy_ignore: '已有启动进行中，忽略重复点击',
    start_fail: '启动失败: ',
    timeout: '请求超时（服务无响应或开相机卡住）',
    page_loaded: '页面加载 ',
  },
  en: {
    title: 'AVM GPU Guide',
    subtitle: 'CUDA hot path · WebRTC stream',
    lang_label: 'Language',
    step_status_h: '0. Status',
    step_status_p: 'CUDA / intrinsics / extrinsics QC',
    step_intr_h: '1. Intrinsics',
    step_intr_p: 'WebRTC · SPACE capture',
    step_ext_h: '2. Extrinsics',
    step_ext_p: 'Auto-lock when stable · burst average',
    step_seam_h: '2b. Seam refine',
    step_seam_p: 'Overlap board · auto-refine slave H',
    step_prev_h: '3. Undistort preview',
    step_prev_p: 'GPU undistort · WebRTC',
    step_bev_h: '4. Live BEV',
    step_bev_p: 'GPU stitch · WebRTC',
    stream_ctrl: 'Stream controls',
    btn_preview: 'GPU preview',
    btn_bev: 'GPU BEV',
    btn_start: 'Start stream',
    btn_starting: 'Starting…',
    btn_stop: 'Stop stream',
    btn_refresh: 'Refresh',
    btn_clear_log: 'Clear log',
    side_note: 'Left steps switch mode; center button starts/stops the stream. Step actions are under the video.',
    stream_offline: 'stream offline',
    cfg_title: 'Calibration config (writes config/*.json)',
    cfg_pattern: 'Board cols×rows (inner corners)',
    cfg_square: 'square_size_m',
    cfg_detect_w: 'Detect width',
    cfg_interval: 'interval_ms / CPU duty',
    cfg_stable: 'stable_frames / auto-lock',
    opt_autolock: 'Auto-lock',
    opt_manual: 'Manual SPACE',
    cfg_intr_n: 'Intrinsics min / target frames',
    cfg_places: 'Extrinsic near_m / lateral_m (4 cams)',
    cfg_cam_title: 'Cameras / resolution',
    cfg_cam_wh: 'Capture width × height',
    cfg_cam_fourcc: 'fourcc / backend',
    cfg_cam_hint: 'Changing resolution requires re-calibration. Save then Probe on Status.',
    btn_save_cfg: 'Save config',
    btn_reload_cfg: 'Reload',
    cfg_hint: 'Restart the matching stream after changing config.',
    report_title: 'Status report',
    loading: 'Loading…',
    log_title: 'Runtime log (UI + server)',
    log_wait: 'Waiting…',
    act_refresh: 'Refresh report',
    act_load_cfg: 'Load config',
    act_probe: 'Probe cameras',
    act_smoke: 'Smoke stream',
    act_start_intr: 'Start intrinsics',
    act_space: 'SPACE capture',
    act_finish_cam: 'Finish camera',
    act_skip_cam: 'Skip camera',
    act_start_ext: 'Start extrinsics',
    act_target_front: 'Calib front',
    act_target_back: 'Calib back',
    act_target_left: 'Calib left',
    act_target_right: 'Calib right',
    act_relock: 'Relock current',
    act_unlock: 'Unlock all',
    act_save_params: 'Save calibration',
    act_start_seam: 'Start seam refine',
    act_swap: 'Swap ref/slave',
    act_edit_ext: 'Write extrinsics',
    act_start_prev: 'Start undistort preview',
    act_start_bev: 'Start live BEV',
    empty_actions: 'No extra actions for this step',
    cfg_loaded: 'Config loaded from disk',
    cfg_saved: 'Saved',
    cfg_restart: '. Restart stream to apply.',
    cuda_off: 'CUDA OFF',
    cuda_not_pipe: 'CUDA not in pipeline',
    cuda_streaming: 'CUDA streaming',
    cuda_ready: 'CUDA ready',
    err_stream: 'Stream error: ',
    hint_no_ext: 'Extrinsics missing: BEV unavailable. Try GPU preview first.',
    hint_idle: 'Pick a left step, then Start stream in the center. Click again to stop.',
    hint_no_backend: 'Cannot reach /api/status. Start: ./scripts/run_web.sh --host 0.0.0.0 --port 8787',
    refresh_fail: 'Refresh failed: ',
    waiting_server: 'Waiting for server (opening cameras / WebRTC)…',
    busy_ignore: 'Start already in progress, ignoring click',
    start_fail: 'Start failed: ',
    timeout: 'Request timed out (server hung or camera open stuck)',
    page_loaded: 'Page loaded ',
  },
};

function t(key) {
  return (I18N[lang] && I18N[lang][key]) || (I18N.zh[key]) || key;
}

function buildActions() {
  return {
    status: [
      [t('act_refresh'), "refresh()"],
      [t('act_load_cfg'), "loadConfig()"],
      [t('act_probe'), "probeCameras()"],
      [t('act_smoke'), "smokeStream()"],
    ],
    intrinsics: [
      [t('act_space'), "calibCmd('space')"],
      [t('act_finish_cam'), "calibCmd('esc')"],
      [t('act_skip_cam'), "calibCmd('skip')"],
    ],
    extrinsics: [
      [t('act_target_front'), "calibCmd('target:front')"],
      [t('act_target_back'), "calibCmd('target:back')"],
      [t('act_target_left'), "calibCmd('target:left')"],
      [t('act_target_right'), "calibCmd('target:right')"],
      [t('act_relock'), "calibCmd('relock')"],
      [t('act_unlock'), "calibCmd('unlock_all')"],
      [t('act_save_params'), "calibCmd('esc')"],
    ],
    seam: [
      [t('act_swap'), "calibCmd('swap')"],
      ['front+left', "calibCmd('pair:front,left')"],
      ['front+right', "calibCmd('pair:front,right')"],
      ['back+left', "calibCmd('pair:back,left')"],
      ['back+right', "calibCmd('pair:back,right')"],
      [t('act_edit_ext'), "calibCmd('esc')"],
    ],
    preview: [],
    bev: [],
  };
}

let actions = buildActions();

function applyLang() {
  document.documentElement.lang = lang === 'en' ? 'en' : 'zh-CN';
  document.querySelectorAll('[data-i18n]').forEach(el => {
    if (el.id === 'cfgHint' || el.id === 'streamToggle') return;
    const key = el.getAttribute('data-i18n');
    if (!key) return;
    el.textContent = t(key);
  });
  const sel = document.getElementById('langSelect');
  if (sel) sel.value = lang;
  actions = buildActions();
  renderStepActions();
  updateStreamToggle(lastStreaming);
  const hint = document.getElementById('cfgHint');
  if (hint && (!hint.textContent || hint.hasAttribute('data-i18n'))) {
    hint.textContent = t('cfg_hint');
  }
}

function setLang(next) {
  lang = (next === 'en') ? 'en' : 'zh';
  localStorage.setItem('avm_lang', lang);
  applyLang();
  log(lang === 'en' ? 'Language: English' : '语言: 中文');
}

function renderStepActions() {
  const box = document.getElementById('actions');
  if (!box) return;
  const list = actions[currentStep] || [];
  box.innerHTML = list.length
    ? list.map(([label, fn]) => {
        const primary = /开始|保存|修改|Start|Save|Write/.test(label) ? ' class="primary"' : '';
        return `<button${primary} onclick="${fn}">${label}</button>`;
      }).join('')
    : `<span class="empty">${t('empty_actions')}</span>`;
}

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

function fillCameras(profile) {
  const box = document.getElementById('cfg_cam_devs');
  if (!box) return;
  const dirs = ['front','back','left','right'];
  const cams = (profile && profile.cameras) || {};
  box.innerHTML = dirs.map(d => {
    const c = cams[d] || {};
    return `<div>
      <label>${d} device /dev/video</label>
      <input id="cfg_cam_${d}" type="number" min="0" step="1" value="${c.device ?? 0}"/>
    </div>`;
  }).join('');
  if (profile) {
    document.getElementById('cfg_cam_w').value = profile.width ?? 1920;
    document.getElementById('cfg_cam_h').value = profile.height ?? 1536;
    document.getElementById('cfg_cam_fourcc').value = profile.fourcc ?? 'YUYV';
    document.getElementById('cfg_cam_backend').value = profile.backend ?? 'v4l2';
  }
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
    fillCameras(j.camera_profile || {});
    document.getElementById('cfgHint').textContent = t('cfg_loaded');
    log(t('cfg_loaded'));
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
  const cameras = {};
  dirs.forEach(d => {
    cameras[d] = {
      device: parseInt(document.getElementById('cfg_cam_'+d).value, 10),
    };
  });
  const body = {
    chessboard: {
      pattern_cols: parseInt(document.getElementById('cfg_cols').value, 10),
      pattern_rows: parseInt(document.getElementById('cfg_rows').value, 10),
      square_size_m: parseFloat(document.getElementById('cfg_square').value),
    },
    placements,
    camera_profile: {
      width: parseInt(document.getElementById('cfg_cam_w').value, 10),
      height: parseInt(document.getElementById('cfg_cam_h').value, 10),
      fourcc: (document.getElementById('cfg_cam_fourcc').value || 'YUYV').trim(),
      backend: document.getElementById('cfg_cam_backend').value || 'v4l2',
      cameras,
    },
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
      t('cfg_saved') + ': ' + (j.changed || []).join(', ') + t('cfg_restart');
    log(t('cfg_saved') + ' ' + JSON.stringify(j.changed));
  } catch (e) {
    log('saveConfig: ' + e.message, 'ERROR');
    alert(e.message);
  }
}

async function probeCameras() {
  log('probe cameras…');
  try {
    if (lastStreaming) await stopAll();
    const { r, j } = await fetchJSON('/api/cameras/probe', {method:'POST'}, 60000);
    if (!r.ok) throw new Error(j.error || ('probe HTTP ' + r.status));
    log('probe ok=' + j.ok + ' ' + JSON.stringify(j.cameras));
    if (!j.ok) alert('Probe FAIL — see status report');
  } catch (e) {
    log('probe: ' + e.message, 'ERROR');
    alert(e.message);
  }
  refresh();
}

async function smokeStream() {
  log('smoke stream…');
  try {
    if (lastStreaming) await stopAll();
    const { r, j } = await fetchJSON('/api/stream/smoke', {method:'POST'}, 60000);
    if (!r.ok) throw new Error(j.error || ('smoke HTTP ' + r.status));
    log('smoke ok=' + j.ok + ' ' + JSON.stringify(j.stream || {}));
  } catch (e) {
    log('smoke: ' + e.message, 'ERROR');
    alert(e.message);
  }
  refresh();
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

function selectStep(step, opts) {
  currentStep = step;
  document.querySelectorAll('.step').forEach(el => {
    el.classList.toggle('active', el.dataset.step === step);
  });
  renderStepActions();
  fetch('/api/step', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({step})}).catch(()=>{});
  // 只切换步骤 UI，不自动开流；开/关推流用侧栏或视频下按钮
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
  // 开流期间 status 常被占满 → AbortError，不要打成 ERROR
  if (busy) {
    updateStreamToggle(lastStreaming);
    return;
  }
  try {
    const { r, j } = await fetchJSON('/api/status?lang=' + encodeURIComponent(lang), {}, 8000);
    if (!r.ok) { log('status HTTP ' + r.status, 'WARN'); return; }
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
    const streaming = !!(s.mode && s.mode !== 'idle');
    lastStreaming = streaming;
    if (!cudaOk) { cb.textContent = t('cuda_off'); cb.className = 'badge bad'; }
    else if (streaming && s.pipeline_cuda === false) { cb.textContent = t('cuda_not_pipe'); cb.className = 'badge bad'; }
    else if (streaming) { cb.textContent = t('cuda_streaming'); cb.className = 'badge ok'; }
    else { cb.textContent = t('cuda_ready'); cb.className = 'badge ok'; }
    updateStreamToggle(streaming);
    const hint = document.getElementById('streamHint');
    if (s.error) hint.textContent = t('err_stream') + s.error;
    else if ((j.extrinsics || {}).status === 'fail')
      hint.textContent = t('hint_no_ext');
    else if (!streaming)
      hint.textContent = t('hint_idle');
    else hint.textContent = '';
  } catch (e) {
    const msg = (e && e.message) || String(e);
    const soft = (e && e.name === 'AbortError')
      || /aborted|timeout|Failed to fetch|NetworkError/i.test(msg);
    if (soft) {
      log(t('waiting_server'), 'INFO');
      return;
    }
    log(t('refresh_fail') + msg, 'ERROR');
    document.getElementById('streamHint').textContent = t('hint_no_backend');
  }
}

function streamModeForStep() {
  return ({
    status: 'preview',
    preview: 'preview',
    bev: 'bev',
    intrinsics: 'calib_intrinsics',
    extrinsics: 'calib_extrinsics',
    seam: 'calib_seam',
  })[currentStep] || 'preview';
}

function updateStreamToggle(streaming) {
  const btn = document.getElementById('streamToggle');
  if (!btn) return;
  if (busy) {
    btn.textContent = t('btn_starting');
    btn.className = 'primary';
    btn.style.borderColor = '';
    btn.style.color = '';
    btn.disabled = true;
    return;
  }
  btn.disabled = false;
  if (streaming) {
    btn.textContent = t('btn_stop');
    btn.className = '';
    btn.style.borderColor = 'var(--bad)';
    btn.style.color = 'var(--bad)';
  } else {
    btn.textContent = t('btn_start');
    btn.className = 'primary';
    btn.style.borderColor = '';
    btn.style.color = '';
  }
}

async function toggleStream() {
  if (busy) { log(t('busy_ignore'), 'WARN'); return; }
  if (lastStreaming) {
    await stopAll();
    return;
  }
  await startStream(streamModeForStep());
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
  if (busy) { log(t('busy_ignore'), 'WARN'); return; }
  busy = true;
  updateStreamToggle(false);
  setPlaceholder(true, 'starting…');
  log('start mode=' + mode);
  try {
    log('POST /api/stream/start …');
    setPlaceholder(true, 'opening cameras…');
    const { r, j } = await fetchJSON('/api/stream/start?mode=' + mode, {method:'POST'}, 60000);
    if (!r.ok) throw new Error(j.error || ('start HTTP ' + r.status));
    log('hub started: mode=' + ((j.stream||{}).mode) + ' cameras=' + JSON.stringify((j.stream||{}).cameras));
    setPlaceholder(true, 'webrtc negotiating…');
    await connectWebRTC();
    const stepForMode = {
      preview: 'preview',
      bev: 'bev',
      calib_intrinsics: 'intrinsics',
      calib_extrinsics: 'extrinsics',
      calib_seam: 'seam',
    };
    selectStep(stepForMode[mode] || 'preview', {autostart: false});
    setPlaceholder(false);
    lastStreaming = true;
    log('OK');
  } catch (e) {
    const msg = (e && e.name === 'AbortError')
      ? t('timeout')
      : (e.message || String(e));
    const soft = (e && e.name === 'AbortError')
      || /aborted|timeout|Failed to fetch/i.test(msg);
    log(t('start_fail') + msg, soft ? 'WARN' : 'ERROR');
    setPlaceholder(true, 'failed');
    if (!soft) alert(t('start_fail') + msg);
    lastStreaming = false;
  } finally {
    busy = false;
    updateStreamToggle(lastStreaming);
    refresh();
  }
}

async function stopAll() {
  log('stop stream');
  busy = true;
  updateStreamToggle(false);
  if (pc) { try { pc.close(); } catch(e) {} pc = null; }
  const v = document.getElementById('video');
  v.srcObject = null;
  try {
    await fetchJSON('/api/stream/stop', {method:'POST'}, 10000);
  } catch (e) {
    const soft = (e && e.name === 'AbortError')
      || /aborted|timeout|Failed to fetch/i.test((e && e.message) || '');
    log('stop: ' + ((e && e.message) || e), soft ? 'INFO' : 'WARN');
  }
  setPlaceholder(true, t('stream_offline'));
  lastStreaming = false;
  busy = false;
  updateStreamToggle(false);
  refresh();
}

async function startCalib(kind) {
  const mode = ({
    extrinsics: 'calib_extrinsics',
    seam: 'calib_seam',
    intrinsics: 'calib_intrinsics',
  })[kind] || 'calib_intrinsics';
  log('start calib mode=' + mode);
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

log(t('page_loaded') + location.href);
applyLang();
selectStep('status', {autostart: false});
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


def _status_payload(lang: str = "en") -> dict:
    """Status report text is English (UI chrome is separately i18n'd in the page)."""
    from avm.camera_io import load_camera_profile, profile_for_web

    _ = lang
    intr = check_intrinsics_quality(CALIB_DIR)
    extr = check_extrinsics_quality(CALIB_DIR / "extrinsics.json")
    stream = HUB.status() if HUB else {"mode": "idle"}
    webrtc = WEBRTC.status() if WEBRTC else {"peers": 0}
    prof = load_camera_profile()
    lines = [f"CUDA: {cuda_status_line()}", f"Intrinsics: {intr['status']}"]
    for d, det in intr.get("details", {}).items():
        rms = det.get("rms")
        rms_s = f"{float(rms):.3f}" if rms is not None else "N/A"
        lines.append(f"  {d}: {det.get('status')} RMS={rms_s}")
        # flag image_size mismatch vs profile request
        try:
            cal = json.loads((CALIB_DIR / f"{d}.json").read_text(encoding="utf-8"))
            isize = cal.get("image_size") or cal.get("img_size")
            if isize and len(isize) >= 2:
                req = [
                    int((prof.get("cameras") or {}).get(d, {}).get("width") or prof["width"]),
                    int((prof.get("cameras") or {}).get(d, {}).get("height") or prof["height"]),
                ]
                if [int(isize[0]), int(isize[1])] != req:
                    lines.append(
                        f"  ! {d} calib image_size={isize[0]}x{isize[1]} "
                        f"!= profile {req[0]}x{req[1]} (re-calibrate)"
                    )
        except Exception:
            pass
    lines.append(f"Extrinsics: {extr['status']}")
    for d, det in extr.get("details", {}).items():
        lines.append(f"  {d}: {det.get('status')}")
    for w in extr.get("global_warnings", []):
        lines.append(f"  ! {w}")
    lines.append(
        f"Camera profile: {prof.get('width')}x{prof.get('height')} "
        f"fourcc={prof.get('fourcc')} backend={prof.get('backend')}"
    )
    with STATE_LOCK:
        probe = STATE.get("last_probe")
    if probe:
        lines.append(f"Camera probe: {'OK' if probe.get('ok') else 'FAIL'}")
        for d, det in (probe.get("cameras") or {}).items():
            if det.get("ok"):
                aw = (det.get("actual_wh") or ["?", "?"])
                lines.append(
                    f"  {d}: OK /dev/video{det.get('device')} "
                    f"{aw[0]}x{aw[1]} ({det.get('backend')})"
                    + (f" !{det.get('warning')}" if det.get("warning") else "")
                )
            else:
                lines.append(
                    f"  {d}: FAIL /dev/video{det.get('device')} "
                    f"{det.get('error')}"
                )
    else:
        lines.append("Camera probe: not run (use Probe cameras)")
    lines.append(f"WebRTC peers: {webrtc.get('peers', 0)}")
    calib = (stream or {}).get("calib") or {}
    if calib.get("kind"):
        lines.append(
            f"Calib session: {calib.get('kind')} dir={calib.get('direction')} "
            f"captured={calib.get('captured')} locked={calib.get('locked')} "
            f"msg={calib.get('message')}"
        )
        if calib.get("sequential"):
            tgt = calib.get("target") or "-"
            got = (calib.get("stable_streak") or {}).get(tgt, 0)
            lines.append(
                f"  Sequential target={tgt} stable={got}/{calib.get('stable_need')} "
                f"pending={calib.get('pending')}"
            )
        if calib.get("kind") == "seam":
            lines.append(
                f"  Seam refine ref={calib.get('seam_ref')} slave={calib.get('seam_slave')} "
                f"last={calib.get('seam_last')}"
            )
    with STATE_LOCK:
        lines.append(
            f"Wizard step={STATE['step']} skip_intr={STATE['skipped_intrinsics']} "
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
        "camera_profile": profile_for_web(prof),
        "last_probe": probe,
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
    server_version = "AVMGpuWeb/0.2.0"

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
                qs = parse_qs(parsed.query or "")
                lang = (qs.get("lang") or ["en"])[0]
                body, code, ctype = _json_bytes(_status_payload(lang=lang))
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
            if path == "/api/cameras/probe":
                from avm.camera_io import probe_cameras

                if HUB and HUB.mode != "idle":
                    body, code, ctype = _json_bytes(
                        {
                            "ok": False,
                            "error": "stop stream before probe",
                            "cameras": {},
                        },
                        409,
                    )
                    self._send(body, code, ctype)
                    return
                result = probe_cameras()
                with STATE_LOCK:
                    STATE["last_probe"] = result
                LOG.info(f"camera probe ok={result.get('ok')}")
                body, code, ctype = _json_bytes(result)
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
            if path in ("/api/cameras/probe", "/api/probe"):
                from avm.camera_io import probe_cameras

                if HUB and HUB.mode != "idle":
                    if WEBRTC:
                        WEBRTC.close_all()
                    HUB.stop()
                result = probe_cameras()
                with STATE_LOCK:
                    STATE["last_probe"] = result
                    STATE["message"] = (
                        "camera probe OK" if result.get("ok") else "camera probe FAIL"
                    )
                LOG.info(f"camera probe ok={result.get('ok')}")
                body, code, ctype = _json_bytes(result)
                self._send(body, code, ctype)
                return
            if path == "/api/stream/smoke":
                # Lightweight: start preview briefly without WebRTC
                if HUB is None:
                    body, code, ctype = _json_bytes(
                        {"ok": False, "error": "hub not ready"}, 500
                    )
                    self._send(body, code, ctype)
                    return
                try:
                    if HUB.mode != "idle":
                        if WEBRTC:
                            WEBRTC.close_all()
                        HUB.stop()
                    st = HUB.start("preview")
                    time.sleep(1.5)
                    frames_ok = False
                    try:
                        # one compose cycle happens in hub thread
                        frames_ok = bool((HUB.status() or {}).get("fps") is not None)
                    except Exception:
                        pass
                    HUB.stop()
                    out = {
                        "ok": True,
                        "smoke": "preview",
                        "stream": st,
                        "frames_observed": frames_ok,
                    }
                    with STATE_LOCK:
                        STATE["message"] = "stream smoke OK"
                    body, code, ctype = _json_bytes(out)
                except Exception as exc:
                    LOG.error(f"stream smoke FAIL: {exc}")
                    body, code, ctype = _json_bytes(
                        {"ok": False, "error": str(exc)}, 400
                    )
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
