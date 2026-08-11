# 标定后 BEV + VLM Demo

从 `avm/live_bev.py` **拷贝**出来的独立演示：假定内外参已完成，只做实时俯视拼接，并用本机 Qwen3-VL 周期性输出**小车避障 / 地面通行**信息（障碍方位、相对空旷方向等）。

**不修改** Web 向导 / `live_bev` / `gpu_hub` 热路径。

## 依赖

- `calib_results/{front,back,left,right}.json` + `extrinsics.json`
- CUDA OpenCV：`source scripts/env_opencv_cuda.sh`
- WorldMM：`~/leucus/.venv-worldmm` + `~/leucus/models/worldmm/Qwen3-VL-2B-Instruct`（默认）
- **NumPy**：CUDA OpenCV 需 1.x；WorldMM/transformers 常为 2.x — `run_demo_bev_vlm.sh` 用系统 Python 拼 BEV，VLM 走 venv 子进程（勿在同一解释器里 `activate` + `import cv2`）。

若相机被 Web 占用，先停 `./scripts/run_web.sh` / docker 容器。

## 运行

```bash
cd ~/bev_demo/avm_gpu
# 本机桌面会话内：
DISPLAY=:0 ./scripts/run_demo_bev_vlm.sh

# SSH / 无 GTK：无窗口，预览写到 output/demo_bev_vlm/preview.jpg，字幕打终端
./scripts/run_demo_bev_vlm.sh --no-window

# 只拼 BEV，不加载 VLM
./scripts/run_demo_bev_vlm.sh --vlm off

# 8B / 更长间隔
./scripts/run_demo_bev_vlm.sh --vlm qwen3vl-8b --caption-interval 30
```

若 `DISPLAY=:0` 仍报 GTK：在桌面终端跑，或先 `xhost +local:`；否则加 `--no-window`。

键盘：`ESC/q` 退出 · `c` 立即描述 · `s` 存帧 · `g` 增益 · `+/-` 融合幂次。

## 结构

| 文件 | 作用 |
|------|------|
| `stitch.py` | 实时拼接（live_bev 拷贝，改用 `avm.cuda_cv` / `camera_io`） |
| `vlm_caption.py` | 父进程调度；写临时 JPG，经子进程要 caption |
| `vlm_worker.py` | WorldMM venv 子进程：加载 Qwen3-VL（无 cv2） |
| `run.py` | 主循环 + HUD 字幕叠加 |
| `../scripts/run_demo_bev_vlm.sh` | 环境一键启动 |
