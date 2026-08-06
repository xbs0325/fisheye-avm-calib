# 工作记录 WORKLOG

> **AVM 部署记录**（汇报用叙事）：[`PROJECT_HISTORY.md`](PROJECT_HISTORY.md)。本文件保留按日流水。

## 2026-08-06 — 成文：项目创作与部署全记录

- 新增 `docs/PROJECT_HISTORY.md`（建仓动机 → GPU Web → 外参/性能/接缝 → Docker/CSI）
- `DEPLOY_0.2.0.md` 收成开箱摘录，避免与全史重复

## 2026-08-06 — 0.2.0 部署收尾（Docker 开箱实测）

详见 `docs/PROJECT_HISTORY.md` §3 与 `docs/DOCKER.md`。

- 基础镜像改为 `ubuntu:24.04`（无 l4t-jetpack r39）
- stage：relocatable cv2 config + 递归打包 FFmpeg8/CUDA13 依赖；`import cv2` → `4.14.0 1`
- CSI：补挂 `media0` / `camsync` / `capture-vi-channel*`，四路 probe + smoke 通过
- 交付：`scripts/docker_run_web.sh`、更新 compose / DOCKER / DEPLOY 文档；Hub 镜像名 `leucushc/avm-gpu:0.2.0`

## 2026-08-06 — 通用分辨率 + JP7.2 Docker（0.2.0）

- `config/camera_profile.json` + `avm/camera_io.py` 统一开流/探针；去掉硬编码 1920×1536
- `/api/cameras/probe`、`/api/stream/smoke`；Web 相机配置与 Probe UI；状态报告左下 / 日志右下
- `Dockerfile` / `docker-compose.yml` / `docs/DOCKER.md`（预编译 CUDA OpenCV COPY）

## 2026-08-06 — 接缝：全部完成后停自动精修

日志里一串「接缝精修全部完成」却仍对 back+right 连拍：`seam_complete` 未置位，
SYNC 到 6 就反复 burst（暂停 grab → 画面像中断），方向不再变。

- 完成后停检测 / 停自动精修；HUD 显示 ALL PAIRS DONE
- 换对冷却 5s（MOVE BOARD）；失败也短冷却
- next/pair 可重做该对

## 2026-08-06 — 接缝精修：联合同步计数（修粘滞 / 饿死从路）

用户反馈：右边（slave）一直认不上；挪板迁就右边后左边仍停在旧角点；应左右同步计数。

- 根因：seam 误用外参的 `_focus` + 单路 sticky READY → ref 粘旧、slave 饿检
- `seam_joint_streak`：两路都新鲜才 +1，任一路 miss/超时共同清零
- 漏检立刻清角点；pair 轮询持续提交；UI 统一 `SYNC n/need`
- 达标自动精修→写盘→下一对；文档补 4.9 / 4.10

## 2026-08-06 — 接缝精修功能（2b）

卷尺 `near_m` 接缝误差：锁 `H_ref`，重叠区共视板重求 `H_slave`。

- 核心：`refine_seam_homography` / `SEAM_PAIRS`（`avm/calibrate_extrinsics.py`）
- Web：`kind=seam`，步骤 2b；只写从路 H + `seam_refined[]`
- 外参 QC：位移模长 / 奇异值（修 left/right「病态」误报）；180° 近边定向
- 部分保存合并旧 H；备份 `calib_results/backups/`

## 2026-08-05 — 远距检测：改用 findChessboardCornersSB

用户反馈"远了识别不到，要从近处拿向远处才认得出"。

- 实测经典算法检出下限 ~9px 格子；SB 可到 **6px**（约 1.5× 距离）
- 无板代价：经典 1327ms → SB **267ms**，远距每路采样从 ~21s 降到 **~2.4s**
- 扫描路径不做 SB→经典回退（叠加后 miss 反而变 1600ms）
- `detect_board`（连拍求 H）同步换 SB，否则预览 READY 但锁定失败
- `detect_use_sb` 可回退；`detect_interval_ms` 500 / `detect_duty` 0.5
- `stable_frames` 保持 10（用户确认可用）
- 附带发现：偶数格棋盘 180° 翻转歧义约 50%，但 `calibrate_one` 枚举 4 旋转吸收

## 2026-08-05 — 外参：原图检测 + 整板 inview + 逐路顺序 + 合并保存

（详见 `CALIBRATION_LESSONS.md` §4.1–4.4）

- 检测在鱼眼原图，角点 `undistortPoints` 再求 H；去畸变图上检测会裁视野
- 全部角点须落在去畸变图内，否则 `NOT FULLY IN VIEW`（防边缘假阳性）
- `sequential` 只检当前 `target`；自动锁定后跳下一方向
- `save_results` 合并上次未重标的 H（几何参数一致时才续用）

## 2026-08-05 — 外参：H-QC 轴分量误判 + 180° RMS 歧义

（详见 §4.5–4.6、§4.8）

- 跨度改位移模长；对称性改比主奇异值；删 |H[0,0]|
- 180° 用近边更长定向（RMS 无法区分）；中心投影安全网
- `drawChessboardCorners` 须 float32，否则锁板存图断言失败被吞

## 2026-08-05 — 外参检测：专注 + 容错 + 自动锁定

旧轮询逻辑实际上让 READY 永远到不了：

- `stable_frames=10` 要求**连续 10 次命中**，一次 miss 归零；而每次检测间隔数秒
- 绘制门槛 `age < 2.0s`，但检测周期远大于 2s → 角点闪一下就消失
- sticky 仅在"该路已有 pending job"时生效，实际总被 round-robin 切走
- 锁定只在 SPACE，从来没有自动锁定

改为：

- **专注模式** `_focus`：某路检出后只喂这一路，其它路不占 CPU；连续 miss ≥ `focus_miss_tolerance`(3) 才交还轮询
- **容错 streak**：单次 miss 只减 1 并保留角点，连续 miss ≥ `streak_reset_misses`(2) 才归零
- **显示保持**：角点保留到该路下次出结果，不再 2s 硬过期
- **自动锁定** `auto_lock`(默认开)：streak ≥ `stable_frames`(默认改 3) 后台直接连拍求 H 并落库
- `_lock_direction()` 抽出，SPACE 与自动锁定共用同一条路径

## 2026-08-05 — 回归 GPU 热路径（修我自己造的 CPU 回退）

上一版把 4 路全分辨率 `cv2.remap` + 1920 棋盘检测放进推流循环 → fps 0.3、gpu 2664ms。

- 实测：`findChessboardCorners` **无棋盘**时最贵，1920 单次 1.3s，多尺度阶梯 ×6 ≈ 8s
- 预览去畸变改 **GPU**（`for_cuda=True` maps + `undistort_gpu` + `cuda.resize`），只下载小图
- 检测移到**后台线程**；按用户确认，每次直接使用 1920 全分辨率，不做低分辨率门控
- `detect_interval_ms=1000` 降频，`detect_duty=0.25` 控制平均 CPU 占空比
- CPU maps 仅供 SPACE 连拍求 H，不进循环
- 实测 compose 2664ms → **17.8ms**（≈56fps），有板检测 15ms

## 2026-08-05 — 外参检测：去畸变 + 去乱码

- 乱码：OpenCV putText 不支持中文，HUD 改纯 ASCII
- NO BOARD 轮转：未扫描路显示 WAIT，扫描路显示 SCAN NOBOARD
- 关键因：预览在鱼眼原图检测、SPACE 在去畸变图检测 → 统一 undist+hires
- HUD 显示 `src=WxH det<=…`；默认 detect_max_width=1920

## 2026-08-05 — 高分辨率检测 + Web 配置面板

- 根因：Web 标定预览在 480 tile 上检测，远距失败
- `detect_board_hires`：全分辨率按 `detect_max_width` 检测，角点映射回 tile
- `calib_config` + 网页表单 → `/api/config` 读写 `chessboard` / `placements` / `web_calib_settings`

## 2026-08-05 — Web 内外参标定推流

- `calib_intrinsics` / `calib_extrinsics` 模式经 WebRTC 看画面
- SPACE/ESC 走 `/api/calib/action`（不抢本机窗口相机）
- 外参沿用稳定检出 + 多帧均值

## 2026-08-05 — 外参多帧均值

- 预览 `stable_streak`：连续检出达标才允许 SPACE
- 全分辨率 `burst` 连拍 → 角点对齐/剔野值 → 均值 → 求 H
- CLI：`--stable-frames` / `--burst-frames` / `--burst-min-ok`

## 2026-08-05 — GPU Web 引导

用户澄清：Web 要做，但不能用 CPU 做热路径。

- `gpu_hub` / `web_server` / WebRTC；运行日志；可跳步
- `./scripts/run_web.sh --port 8787`

## 2026-08-05 — 新建 avm_gpu

从旧 AVM 抽核心；CLI 向导；外参难点文档；CUDA 冒烟 OK。
