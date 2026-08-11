#!/usr/bin/env python3
"""Standalone Qwen3-VL analyze subprocess (no WorldMM).

Uses transformers + qwen_vl_utils directly. Run inside a venv that has
torch/transformers (e.g. leucus .venv-worldmm for deps only — not memory stack).

Protocol (stdin / stdout, line-oriented UTF-8):
  Worker -> READY
  Parent -> ANALYZE /abs/path.jpg nav
  Parent -> ANALYZE /abs/path.jpg grasp <target>
  Parent -> CAPTION /abs/path.jpg
  Worker -> OK <ms>
           <one-line JSON or caption>
           END
  Worker -> ERR <message>
  Parent -> QUIT
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from perception.schema import NAV_PROMPT, grasp_prompt

DEFAULT_MODELS = Path(
    os.environ.get(
        "PERCEPTION_MODELS",
        os.environ.get(
            "WORLDMM_MODELS",
            str(Path.home() / "leucus" / "models" / "worldmm"),
        ),
    )
)

VLM_DIRS = {
    "qwen3vl-2b": "Qwen3-VL-2B-Instruct",
    "qwen3vl-4b": "Qwen3-VL-4B-Instruct",
    "qwen3vl-8b": "Qwen3-VL-8B-Instruct",
}


class Qwen3VLInferencer:
    """Thin single-image generate wrapper (Jetson-friendly defaults)."""

    def __init__(self, model_path: Path, *, max_side: int) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]

        device_map = os.environ.get("PERCEPTION_DEVICE_MAP", "cuda:0")
        attn_impl = os.environ.get("PERCEPTION_ATTN_IMPL", "sdpa")
        dtype_name = os.environ.get("PERCEPTION_DTYPE", "bfloat16")
        dtype = getattr(torch, dtype_name, torch.bfloat16)

        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": device_map,
        }
        if attn_impl and attn_impl != "auto":
            load_kwargs["attn_implementation"] = attn_impl

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(model_path), **load_kwargs
            )
        except Exception as exc:
            print(
                f"[vlm] load with attn={attn_impl} failed ({exc}); retry without",
                file=sys.stderr,
                flush=True,
            )
            load_kwargs.pop("attn_implementation", None)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(model_path), **load_kwargs
            )

        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.max_side = int(max_side)
        self.model.eval()
        # Greedy decode: drop sampling flags from the model card (avoids stderr spam)
        gc = getattr(self.model, "generation_config", None)
        if gc is not None:
            gc.do_sample = False
            for key in ("temperature", "top_p", "top_k"):
                if hasattr(gc, key):
                    setattr(gc, key, None)

    def generate(self, path: Path, prompt: str, max_new_tokens: int) -> tuple[str, float]:
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        pil = Image.open(path).convert("RGB")
        w, h = pil.size
        side = self.max_side
        if max(h, w) > side:
            scale = side / float(max(h, w))
            pil = pil.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.BILINEAR,
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
        # Prefill "{" so the model does not start with ```json (truncates + parse fail)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + "{"
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # device_map models: move tensors to first parameter device
        try:
            first_param = next(self.model.parameters())
            inputs = inputs.to(first_param.device)
        except StopIteration:
            pass

        eos_ids: list[int] = []
        gc = getattr(self.model, "generation_config", None)
        if gc is not None and getattr(gc, "eos_token_id", None) is not None:
            e = gc.eos_token_id
            eos_ids = list(e) if isinstance(e, (list, tuple)) else [int(e)]
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            for s in ("}",):
                ids = tok.encode(s, add_special_tokens=False)
                if len(ids) == 1 and int(ids[0]) not in eos_ids:
                    eos_ids.append(int(ids[0]))
        gen_kw: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "use_cache": True,
        }
        if eos_ids:
            gen_kw["eos_token_id"] = eos_ids
        t0 = time.time()
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **gen_kw)
        ms = (time.time() - t0) * 1000.0

        # Trim prompt tokens
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated)
        ]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        one = (decoded[0] if decoded else "").strip().replace("\n", " ")
        if not one.startswith("{"):
            one = "{" + one
        del inputs, generated, trimmed
        return one, ms


def _load_vlm(vlm_name: str, models_dir: Path, max_side: int):
    sub = VLM_DIRS.get(vlm_name)
    if not sub:
        raise ValueError(f"未知 VLM: {vlm_name}")
    local = models_dir / sub
    if not local.is_dir():
        raise FileNotFoundError(f"缺少模型目录: {local}")
    return Qwen3VLInferencer(local, max_side=max_side)


def _prompt_for(
    mode: str, target: str, caption_prompt: str, *, occ_az: str = ""
) -> str:
    if mode == "grasp":
        return grasp_prompt(target, occ_az=occ_az)
    if mode == "nav":
        return NAV_PROMPT
    return caption_prompt


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone Qwen3-VL analyze worker")
    ap.add_argument("--vlm", default="qwen3vl-2b", choices=list(VLM_DIRS))
    ap.add_argument("--models", type=Path, default=None)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompt", default=NAV_PROMPT, help="CAPTION fallback prompt")
    args = ap.parse_args()
    models_dir = Path(args.models or DEFAULT_MODELS)

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

        if line.startswith("ANALYZE "):
            rest = line[8:].strip()
            parts = rest.split()
            if len(parts) < 2:
                print("ERR ANALYZE needs: ANALYZE <path> <nav|grasp> [target]", flush=True)
                continue
            path = Path(parts[0])
            mode = parts[1].lower()
            extra = " ".join(parts[2:]) if len(parts) > 2 else "object"
            target, occ_az = extra, ""
            if " ::" in extra:
                target, occ_az = extra.split(" ::", 1)
                target, occ_az = target.strip() or "object", occ_az.strip()
            if mode not in ("nav", "grasp"):
                print(f"ERR unknown mode: {mode}", flush=True)
                continue
            prompt = _prompt_for(mode, target, args.prompt, occ_az=occ_az)
            try:
                text, ms = vlm.generate(path, prompt, args.max_new_tokens)
                print(f"OK {ms:.0f}", flush=True)
                print(text or "{}", flush=True)
                print("END", flush=True)
            except Exception as exc:
                print(f"ERR {exc}", flush=True)
            continue

        if line.startswith("CAPTION "):
            path = Path(line[8:].strip())
            try:
                text, ms = vlm.generate(path, args.prompt, args.max_new_tokens)
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
