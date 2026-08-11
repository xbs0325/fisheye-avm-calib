#!/usr/bin/env python3
"""VLM caption subprocess (venv + NumPy 2.x). No OpenCV — parent stitches with CUDA cv2.

Protocol (stdin / stdout, line-oriented UTF-8):
  Worker -> READY
  Parent -> CAPTION /abs/path.jpg
  Worker -> OK <ms>
           <one-line caption>
           END
  Worker -> ERR <message>   (then may continue, or exit on fatal)
  Parent -> QUIT
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

DEFAULT_MODELS = Path(
    os.environ.get("WORLDMM_MODELS", str(Path.home() / "leucus" / "models" / "worldmm"))
)
DEFAULT_WORLDMM_SRC = Path(
    os.environ.get("WORLDMM_SRC", str(Path.home() / "leucus" / "WorldMM" / "src"))
)

VLM_DIRS = {
    "qwen3vl-2b": "Qwen3-VL-2B-Instruct",
    "qwen3vl-4b": "Qwen3-VL-4B-Instruct",
    "qwen3vl-8b": "Qwen3-VL-8B-Instruct",
}

DEFAULT_PROMPT = (
    "这是小车俯视环视拼接图（上=前方，下=后方，左/右=侧方；画面中心附近常为车体盲区或拼接空洞，可忽略）。"
    "任务：为地面小车避障与通行提供信息。请用两三句中文回答："
    "1) 四周地面是否有障碍/危险（人、箱子、线缆、台阶、坑洼、反光锥等）及大致方位（前/后/左/右）；"
    "2) 哪一侧相对更空旷、更适合通行；"
    "3) 看不清的区域直接说不确定。"
    "只依据画面可见内容，不要编造汽车、车道线或其他看不到的物体。"
)


def _ensure_worldmm_path() -> None:
    src = DEFAULT_WORLDMM_SRC
    if src.is_dir():
        sp = str(src)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _load_vlm(vlm_name: str, models_dir: Path, max_side: int):
    _ensure_worldmm_path()
    os.environ.setdefault("WORLDMM_ATTN_IMPL", "sdpa")
    os.environ.setdefault("WORLDMM_QWEN_DEVICE_MAP", "cuda:0")
    os.environ.setdefault("WORLDMM_DTYPE", "bfloat16")
    if "HF_ENDPOINT" in os.environ:
        del os.environ["HF_ENDPOINT"]

    from worldmm.llm import qwen3vl as qwen_mod

    sub = VLM_DIRS.get(vlm_name)
    if not sub:
        raise ValueError(f"未知 VLM: {vlm_name}")
    local = models_dir / sub
    if not local.is_dir():
        raise FileNotFoundError(
            f"缺少模型目录: {local}（请先跑 leucus jetson_download_models）"
        )
    qwen_mod.MODEL_DICT[vlm_name] = str(local)
    return qwen_mod.Qwen3VLModel(vlm_name, max_size=(max_side, max_side))


def _caption_image(vlm, path: Path, prompt: str, max_side: int, max_new_tokens: int) -> tuple[str, float]:
    from PIL import Image

    pil = Image.open(path).convert("RGB")
    w, h = pil.size
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        pil = pil.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    t0 = time.time()
    text = vlm.generate(messages, max_new_tokens=max_new_tokens)
    ms = (time.time() - t0) * 1000.0
    return str(text).strip().replace("\n", " "), ms


def main() -> int:
    ap = argparse.ArgumentParser(description="WorldMM VLM worker (subprocess)")
    ap.add_argument("--vlm", default="qwen3vl-2b", choices=list(VLM_DIRS))
    ap.add_argument("--models", type=Path, default=None)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()
    models_dir = Path(args.models or DEFAULT_MODELS)

    # Unbuffered line protocol
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        vlm = _load_vlm(args.vlm, models_dir, args.max_side)
    except Exception as exc:
        print(f"ERR load failed: {exc}", flush=True)
        return 1

    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            return 0
        if line.startswith("CAPTION "):
            path = Path(line[8:].strip())
            try:
                text, ms = _caption_image(
                    vlm, path, args.prompt, args.max_side, args.max_new_tokens
                )
                print(f"OK {ms:.0f}", flush=True)
                print(text or "(empty)", flush=True)
                print("END", flush=True)
            except Exception as exc:
                print(f"ERR {exc}", flush=True)
            continue
        print(f"ERR unknown command: {line[:40]}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
