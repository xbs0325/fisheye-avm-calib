# NVIDIA Thor 部署说明（Wiki 可复现）

平台：**Seeed reComputer j6015**（Thor）· JetPack **R38.4** · Ubuntu 24.04 · aarch64。  
从 AGX Orin **J501**（JetPack R39 / JP 7.2）迁过来时，软件不能原样拷。

本文可以整页贴进内部 Wiki。所有命令默认在板子用户 `seeed` 下执行，仓库路径 `~/fisheye-avm-calib`。

## 与 Orin / J501 的差异

| 项目 | J501 AGX Orin | Thor j6015 |
|------|---------------|------------|
| Tegra / JetPack | R39.x / JP 7.2 | **R38.4** |
| GPU | CC 8.7 | CC **11.0** |
| CUDA 目标 | tegra | **sbsa-linux** |
| 系统 OpenCV | 4.6，无 cudawarping | 同左，**不能用于 BEV** |
| 侧装 OpenCV | Orin 上编过的包 | **必须在 Thor 上重编** |
| 相机 `/dev/video*` 编号 | J501 `camera_profile.json` | **板对板可能对调，尤其是前后** |
| Python 装包 | 旧 Ubuntu 可 `pip3 install` | 24.04 **PEP 668**：禁止裸 pip，见下文脚本 |

**不要**把 Orin 的 `~/.local/opencv-4.14.0-cuda` 拷到 Thor。

## 一、CUDA OpenCV（拼接 / 标定共用）

```bash
cd ~/fisheye-avm-calib
./scripts/build_opencv_cuda.sh --jobs $(nproc)    # 约 30–90 min
source scripts/env_opencv_cuda.sh
python3 -c "import cv2; print(cv2.__version__, cv2.cuda.getCudaEnabledDeviceCount())"
# 预期: 4.14.0  1
```

Ubuntu 上 cmake 可能把绑定装到 `dist-packages`（不是 `site-packages`）。`scripts/env_opencv_cuda.sh` 两种路径都会找。当前终端若仍是旧环境，需要重新 `source`。

无相机逻辑自检：

```bash
source scripts/env_opencv_cuda.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3 -m perception.smoke_offline
```

## 二、标定 Web 依赖（系统 Python + NumPy 1.x）

Ubuntu 24.04 **不能**直接 `pip3 install -r requirements.txt`（会报 `externally-managed-environment`）。  
标定推流用 **系统 python3**（和 CUDA OpenCV 同一解释器），**不要**装进 `~/leucus/.venv-worldmm`。

```bash
sudo apt-get install -y python3-pip python3-venv
./scripts/install_web_deps.sh
```

等价命令（Wiki 备查）：

```bash
/usr/bin/python3 -m pip install --user --break-system-packages \
  'numpy>=1.26,<2' 'aiortc>=1.9.0' 'av>=12.0.0'
```

必须钉死 `numpy<2`，否则侧装 CUDA OpenCV 会挂。

缺 `aiortc` 时 `./calib.sh` 会直接提示跑上面的脚本。

## 三、感知依赖（VLM / YOLO-World，独立 venv）

拼接继续用系统 Python；Qwen3-VL 和 YOLO-World 走 NumPy 2.x 的 venv。

```bash
sudo apt-get install -y python3-venv python3-pip
./scripts/setup_perception_thor.sh          # torch + transformers + ultralytics + CLIP + Qwen3-VL-2B
./scripts/download_perception_models.sh     # models/perception/yolov8s-worldv2.pt
```

`libcudss.so.0` 缺失时：

```bash
sudo apt-get install -y libcudss0-cuda-13
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/libcudss/13:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

YOLO-World 还需要 **OpenAI CLIP**（`openai-clip` + `ftfy`）。漏装会报 `No module named 'clip'`。`setup_perception_thor.sh` 里已经包含。

权重位置：

| 用途 | 路径 |
|------|------|
| Qwen3-VL-2B | `~/leucus/models/worldmm/Qwen3-VL-2B-Instruct/` |
| YOLO-World | `~/fisheye-avm-calib/models/perception/yolov8s-worldv2.pt` |
| 感知 venv | `~/leucus/.venv-worldmm` |

## 四、相机编号（J501 → Thor 最常见坑）

四路索引在 `config/camera_profile.json`（与 `config/camera_config.json` 保持一致）。

J501 上曾是 `front=2, back=0, left=3, right=1`。换到 Thor / 换线后，**左右仍对、前后对调** 很常见（不是整幅旋转 180°，否则左右也会反）。

本仓库 Thor 默认：

| 方向 | `/dev/video*` |
|------|----------------|
| front | 0 |
| back | 2 |
| left | 3 |
| right | 1 |

核对：挡住**车头**镜头，BEV **上方 F** 应变暗；挡住车尾，下方 B 变暗。不对就只对调 front/back 的 device，**不要改** `calib_results/*.json`。

## 五、日常启动（相机不能两套程序共用）

```bash
cd ~/fisheye-avm-calib

# 环视 Demo（拼接 + 占用 + YOLO 框 + VLM 口述）
./run.sh

# 标定 / 补缝（步骤 2b）
./calib.sh
# 浏览器: http://<板子IP>:8787/
```

| | `./run.sh` | `./calib.sh` |
|--|------------|--------------|
| 做什么 | 标定后的 360° 鸟瞰 Demo | Web 向导：内参 / 外参 / **2b 接缝** |
| 先停谁 | 停标定 Web | 停环视 Demo |

补缝：页面进 **2b**，顺序 `front+left` → `front+right` → `back+left` → `back+right`。棋盘放两路重叠区，两路绿框 READY 后自动锁从路 H。四对完成后 Ctrl+C，再 `./run.sh`。

Thor 本机窗口经常是 `DISPLAY=:1`（不是 `:0`）。笔记本浏览器访问标定即可，不必在板子上开浏览器。

## 六、不要用的路径

- Orin 编好的 `docker/opencv-cuda/` 直接拿到 Thor
- `docs/DOCKER.md` 的 JP 7.2 开箱镜像当 Thor 运行时（Docker 仍按 **Orin** 文档）
- 在 `~/leucus/.venv-worldmm` 里 `import cv2` 做拼接（NumPy 版本冲突）

感知契约与检查表：`docs/PERCEPTION.md`。标定踩坑：`docs/CALIBRATION_LESSONS.md`。
