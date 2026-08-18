# 四路鱼眼环视：场景与技术

面向带机械臂的移动底盘：车身四周装四路鱼眼相机，先拼成 360° 环视，再投影成俯视地面图（BEV），供导航避障参考、机械臂夹取的方向性粗定位，以及 VLM 场景说明。本仓库当前阶段**只做感知与可视化**，不向底盘或机械臂发控制指令。

## English

This project puts four fisheye cameras around a mobile chassis (with an optional arm) and turns them into a single top-down ground view for driving and manipulation assist.

Cameras are calibrated as fisheye (intrinsics K/D, chessboard) and registered with homographies so undistorted views land on one metric BEV. Live stitching runs on CUDA OpenCV (`remap`, `warpPerspective`, blend). From that BEV we run three heads with a fixed split of labor:

- **Occupancy** — a classical 2D grid (~0.2 m cells) on appearance vs. the floor: which side is clearer and how close nearby obstacles are. This is a path/avoidance hint, not lidar SLAM.
- **YOLO-World** — open-vocabulary boxes (bottle, chair, carton, …) mapped into `base_link` `(x_m, y_m)` and a compass bin (front / front-left / …) so the arm or chassis can yaw toward a target. 2D ground pose only, not 6DoF grasp.
- **VLM (Qwen3-VL-2B)** — a short English caption of what to watch for around the vehicle. It is assistive scene language only: we do **not** parse the caption into coordinates; boxes and xy come from YOLO.

Image up is vehicle forward. This stage visualizes perception only; it does not send chassis or arm commands. Platform: Jetson (Seeed reComputer Thor / AGX Orin).


## 场景

```
四路鱼眼前视 / 后视 / 左视 / 右视
        │
        ▼
  鱼眼内参 + 外参（单应 H）
        │
        ▼
  GPU 去畸变 → 地面 BEV 拼接
        │
        ├── 占用栅格：哪一侧可走、近处障碍距离
        ├── YOLO-World：瓶子 / 椅子 / 纸箱等 → base_link (x, y)
        └── Qwen3-VL：两三句英文说明四周要当心什么
```

| 用途 | 怎么用 BEV | 输出 |
|------|------------|------|
| 环路 / 环视展示 | 四路拼成一张俯视图 | 实时窗口：左拼接、右占用图 |
| 导航辅助 | 地面 2D 占用，不是激光地图 | `free` 比例、F/B/L/R 最近障碍距离 |
| 夹取方向辅助 | 开放词汇检出目标框心 | `base_link` 下 `(x_m, y_m)` 与方位（前/左前…） |
| VLM 辅助分析 | 看拼接图写说明 | 短英文 caption，不当坐标真值 |

约定：画面**上 = 车前**；`base_link` 原点约在 BEV 中心，**+X 前、+Y 左**。IPM 按地面平面近似，没有深度，因此只有 2D 地面位姿，不能直接当 6DoF 抓取。

## 技术栈

| 层 | 技术 | 作用 |
|----|------|------|
| 硬件 | Seeed reComputer Thor j6015（JetPack R38.4）或 AGX Orin | 四路 USB 鱼眼 + CUDA |
| 标定 | OpenCV `fisheye` + 棋盘格 SB 检测 + `findHomography` | 内参 K/D，外参 H（去畸变图 → BEV） |
| 拼接热路径 | 自编译 CUDA OpenCV 4.14（`cudawarping`） | GPU `remap` / `warpPerspective` / 加权融合 |
| 标定交互 | Python Web + WebRTC（aiortc） | 内参 → 外参 → 接缝精修，浏览器操作 |
| 占用 | 经典 BEV 外观模型（无分割网络） | 相对地砖的亮度/色度差 → 0.2 m 栅格 |
| 检测 | Ultralytics YOLO-World v2（`yolov8s-worldv2`） | 开放词汇框；俯视 imgsz=384 |
| 语义 | Qwen3-VL-2B（独立 venv，Transformers） | 只写 caption，不替代 YOLO 坐标 |
| 车体叠图 | 透明 PNG（`assets/ego_overlay.png`） | 盖住画面中心拼接盲区 |

职责拆分（避免互相抢几何）：

- **占用** = 空间（可走 / 不可走）
- **YOLO-World** = 物体框和 `xy`
- **VLM** = 给人看的短描述

## 软件结构

```
fisheye-avm-calib/
  avm/           标定 + GPU 拼接（Web / CLI）
  perception/    BEV 占用、YOLO、VLM、演示 UI
  config/        相机 /dev/video*、棋盘、画布尺度
  calib_results/ 内参、外参 H
  docs/THOR.md   Thor 板上可复现安装
  run.sh         感知演示
  calib.sh       标定网页
```

Thor 上拼接走**系统 Python + NumPy 1.x + CUDA OpenCV**；YOLO / VLM 走独立 venv（NumPy 2.x）。两套解释器不要混。

## 运行（标定完成后）

```bash
source scripts/env_opencv_cuda.sh
./run.sh          # 环视 Demo：BEV + 占用 + YOLO + VLM caption
./calib.sh        # 标定 Web  http://<板子IP>:8787/
```

Demo 与标定互斥（抢相机）。感知参数、消息 JSON、键盘见 `docs/PERCEPTION.md`。Thor 编译 OpenCV / PEP 668 见 `docs/THOR.md`。

## 明确不做的事

- 不控制底盘速度、不控制机械臂关节
- 不用占用栅格当 SLAM / 3D 地图
- 不把 VLM 句子解析成精确坐标（坐标只信 YOLO）
- 不在无 CUDA 的机器上跑实时拼接（可用 `--allow-cpu`，仅调试）
