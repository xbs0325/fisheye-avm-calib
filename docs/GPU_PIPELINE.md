# GPU 管线说明

```bash
source ~/bev_demo/avm_gpu/scripts/env_opencv_cuda.sh
```

| 模块 | 作用 |
|------|------|
| `avm/cuda_cv.py` | CUDA remap/warp/blend |
| `avm/gpu_hub.py` | 相机 + preview/bev 循环（Web 共用） |
| `avm/web_server.py` | GPU MJPEG 引导（可跳步） |
| `avm/live_bev.py` | 本机窗口 BEV |
| `avm/wizard.py` | CLI（含 `--web`） |

```
抓帧 → GPU remap → warp → blend → download → **WebRTC (aiortc)** → 浏览器
```

主推流为 WebRTC；不再以 MJPEG 为主路径。
