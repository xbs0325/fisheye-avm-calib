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
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

from perception.schema import (
    CAPTION_PROMPT,
    NAV_PROMPT,
    finalize_vlm_json,
    grasp_prompt,
)

# HuggingFace logs docstring checks as "[ERROR] …" on stdout by default,
# which poisons the READY / OK / END line protocol.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _keep_stdout_for_protocol() -> None:
    warnings.filterwarnings("ignore")
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.CRITICAL)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(stderr)
    root.setLevel(logging.CRITICAL)
    for name in ("transformers", "huggingface_hub", "accelerate"):
        log = logging.getLogger(name)
        log.handlers.clear()
        log.addHandler(stderr)
        log.setLevel(logging.CRITICAL)
        log.propagate = False
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        hf_log = hf_logging.get_logger()
        hf_log.setLevel(logging.CRITICAL)
        hf_log.handlers.clear()
        hf_log.addHandler(stderr)
        hf_log.propagate = False
    except Exception:
        pass

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

    def generate(
        self,
        path: Path,
        prompt: str,
        max_new_tokens: int,
        *,
        json_prefill: bool = False,
    ) -> tuple[str, float]:
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
        # ANALYZE JSON prefills "{"; CAPTION must not, or the model dumps JSON.
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + ("{" if json_prefill else "")
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
        # Do NOT treat "}" as EOS. Nested objects close first and truncate JSON.
        # Cheap token-id brace count — decoding every token on Thor costs seconds.
        tok = getattr(self.processor, "tokenizer", None)
        prompt_len = int(inputs["input_ids"].shape[-1])
        stop_list = None
        if json_prefill and tok is not None:
            try:
                from transformers import StoppingCriteria, StoppingCriteriaList

                open_ids: set[int] = set()
                close_ids: set[int] = set()
                for s in ("{", '{"'):
                    ids = tok.encode(s, add_special_tokens=False)
                    if len(ids) == 1:
                        open_ids.add(int(ids[0]))
                for s in ("}", '"}'):
                    ids = tok.encode(s, add_special_tokens=False)
                    if len(ids) == 1:
                        close_ids.add(int(ids[0]))
                start_depth = 1

                class _JsonRootCloseStop(StoppingCriteria):
                    def __init__(self) -> None:
                        super().__init__()
                        self._prompt_len = prompt_len
                        self._open = open_ids
                        self._close = close_ids
                        self._depth0 = start_depth

                    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ARG002
                        depth = self._depth0
                        for tid in input_ids[0, self._prompt_len :].tolist():
                            if tid in self._open:
                                depth += 1
                            elif tid in self._close:
                                depth -= 1
                            if depth <= 0:
                                return True
                        return False

                stop_list = StoppingCriteriaList([_JsonRootCloseStop()])
            except Exception:
                stop_list = None
        gen_kw: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "use_cache": True,
        }
        if eos_ids:
            gen_kw["eos_token_id"] = eos_ids
        if stop_list is not None:
            gen_kw["stopping_criteria"] = stop_list
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
        if json_prefill:
            one = finalize_vlm_json(one)
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
    ap.add_argument("--prompt", default=CAPTION_PROMPT, help="CAPTION prompt")
    args = ap.parse_args()
    models_dir = Path(args.models or DEFAULT_MODELS)
    _keep_stdout_for_protocol()

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
                text, ms = vlm.generate(
                    path, prompt, args.max_new_tokens, json_prefill=True
                )
                print(f"OK {ms:.0f}", flush=True)
                print(text or "{}", flush=True)
                print("END", flush=True)
            except Exception as exc:
                print(f"ERR {exc}", flush=True)
            continue

        if line.startswith("CAPTION "):
            path = Path(line[8:].strip())
            try:
                text, ms = vlm.generate(
                    path, args.prompt, args.max_new_tokens, json_prefill=False
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
