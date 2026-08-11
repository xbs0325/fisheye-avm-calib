#!/usr/bin/env python3
"""YOLO-World open-vocab detect subprocess. No OpenCV.

Protocol:
  Worker -> READY
  Parent -> ANY /abs/path.jpg <cw> <ch> <scale> <target words...>
  Worker -> OK <ms>
           <one-line JSON>
           END
  Parent -> QUIT
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

# Ultralytics otherwise prints warnings / pip AutoUpdate on stdout and
# breaks the READY/OK/JSON/END line protocol.
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_VERBOSE"] = "false"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "perception" / "yolov8s-worldv2.pt"


def _azimuth(u_norm: float, v_norm: float) -> str:
    dx = u_norm - 0.5
    dy = 0.5 - v_norm
    if abs(dx) < 0.12 and abs(dy) < 0.12:
        return "center"
    ang = math.degrees(math.atan2(dx, dy))
    if -22.5 <= ang < 22.5:
        return "f"
    if 22.5 <= ang < 67.5:
        return "fr"
    if 67.5 <= ang < 112.5:
        return "r"
    if 112.5 <= ang < 157.5:
        return "br"
    if ang >= 157.5 or ang < -157.5:
        return "b"
    if -157.5 <= ang < -112.5:
        return "bl"
    if -112.5 <= ang < -67.5:
        return "l"
    return "fl"


_BASE_VOCAB = (
    "water bottle",
    "plastic bottle",
    "bottle",
    "computer mouse",
    "mouse",
    "wireless mouse",
    "cup",
    "can",
    "phone",
    "cell phone",
    "remote",
    "keyboard",
    "laptop",
    "book",
    "box",
    "person",
    "chair",
    "backpack",
    "handbag",
    "apple",
    "banana",
    "bowl",
    "orange",
    "scissors",
    "pen",
    "tape",
    "shoe",
    "bag",
    "cable",
    "tool",
)


def _prompt_for(target: str) -> str:
    t = (target or "object").strip()
    key = t.lower().replace(" ", "")
    if key in ("mouse", "鼠标", "computermouse"):
        return "computer mouse"
    if key in (
        "bottle",
        "瓶",
        "瓶子",
        "水瓶",
        "矿泉水",
        "矿泉水瓶",
        "waterbottle",
        "plasticbottle",
    ):
        return "water bottle"
    if key in ("cup", "杯", "杯子"):
        return "cup"
    if key in ("phone", "手机", "cellphone"):
        return "cell phone"
    return t or "object"


def _vocab_for(target: str) -> list[str]:
    prompt = _prompt_for(target)
    out: list[str] = []
    seen: set[str] = set()
    for name in (prompt, *_BASE_VOCAB):
        key = name.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def _name_of(cls_i: int, vocab: list[str], names: Any = None) -> str:
    """YOLO-World set_classes stores names as a list, not a dict."""
    i = int(cls_i)
    if 0 <= i < len(vocab):
        return vocab[i]
    if isinstance(names, dict):
        v = names.get(i, names.get(str(i)))
        if v is not None and str(v).strip() and not str(v).isdigit():
            return str(v)
    if isinstance(names, (list, tuple)) and 0 <= i < len(names):
        return str(names[i])
    return f"cls{i}"


def _is_target(label: str, target: str) -> bool:
    lab = (label or "").lower()
    tgt = _prompt_for(target).lower()
    if not lab or not tgt:
        return False
    if lab == tgt or tgt in lab or lab in tgt:
        return True
    bits = [b for b in tgt.split() if len(b) > 2]
    return any(b in lab for b in bits)


@contextmanager
def _ultralytics_quiet() -> Iterator[None]:
    """Keep protocol stdout clean; ultralytics logs go to stderr."""
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


def _load_yolo(weights: Path, device: str, imgsz: int):
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO

    use_gpu = str(device).lower() not in ("cpu", "")
    if use_gpu:
        # Leave headroom for parent OpenCV-CUDA stitch on the same Orin.
        try:
            torch.cuda.set_per_process_memory_fraction(0.28, device=0)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    path = weights if weights.is_file() else "yolov8s-worldv2.pt"
    with _ultralytics_quiet():
        model = YOLO(str(path))
        try:
            model.fuse()
        except Exception:
            pass
        warm = Image.fromarray(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))
        # Do not pass half= (deprecated; prints to stdout and breaks protocol).
        model.predict(
            warm,
            verbose=False,
            device=device,
            imgsz=imgsz,
            max_det=20,
        )
    return model, use_gpu


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO-World detect worker")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--device", default="0")
    ap.add_argument("--target", default="bottle")
    args = ap.parse_args()
    if str(args.device).lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    if "PYTHONPATH" in os.environ:
        parts = [p for p in os.environ["PYTHONPATH"].split(":") if "avm_gpu" in p]
        os.environ["PYTHONPATH"] = ":".join(parts)
    os.environ.pop("HF_ENDPOINT", None)

    try:
        from PIL import Image

        model, use_gpu = _load_yolo(args.weights, args.device, int(args.imgsz))
        classes = _vocab_for(args.target)
        with _ultralytics_quiet():
            model.set_classes(classes)
    except Exception as exc:
        print(f"ERR load failed: {exc}", flush=True)
        return 1

    print("READY", flush=True)
    last_classes = list(classes)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            return 0
        if not line.startswith("ANY "):
            print(f"ERR unknown command: {line[:40]}", flush=True)
            continue
        parts = line.split()
        if len(parts) < 5:
            print("ERR ANY needs: ANY <jpg> <cw> <ch> <scale> [target]", flush=True)
            continue
        path = Path(parts[1])
        try:
            cw, ch = int(parts[2]), int(parts[3])
            scale = float(parts[4])
        except ValueError as exc:
            print(f"ERR bad args: {exc}", flush=True)
            continue
        target = " ".join(parts[5:]).strip() or "bottle"

        try:
            t0 = time.time()
            with path.open("rb") as fh:
                raw = fh.read()
            pil = Image.open(BytesIO(raw)).convert("RGB")
            want = _vocab_for(target)
            with _ultralytics_quiet():
                if want != last_classes:
                    model.set_classes(want)
                    last_classes = list(want)
                results = model.predict(
                    pil,
                    verbose=False,
                    device=args.device,
                    imgsz=int(args.imgsz),
                    conf=args.conf,
                    max_det=20,
                    iou=0.5,
                )
            r0 = results[0]
            boxes = r0.boxes
            detections: list[dict[str, Any]] = []
            iw, ih = pil.size
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.detach().cpu().numpy()
                clss = boxes.cls.detach().cpu().numpy().astype(int)
                confs = boxes.conf.detach().cpu().numpy()
                sx = float(cw) / float(max(1, iw))
                sy = float(ch) / float(max(1, ih))
                for i in range(len(clss)):
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                    u = 0.5 * (x1 + x2) * sx
                    v = 0.5 * (y1 + y2) * sy
                    u_n, v_n = u / float(cw), v / float(ch)
                    area = max(0.0, (x2 - x1) * (y2 - y1))
                    # Drop only large ego-vehicle blobs at canvas center.
                    if (
                        abs(u_n - 0.5) < 0.08
                        and abs(v_n - 0.5) < 0.08
                        and area > 0.04 * float(iw) * float(ih)
                    ):
                        continue
                    if area < 12:
                        continue
                    label = _name_of(int(clss[i]), want, getattr(r0, "names", None))
                    x_m = (ch * 0.5 - v) / scale
                    y_m = (cw * 0.5 - u) / scale
                    detections.append(
                        {
                            "label": label,
                            "azimuth": _azimuth(u_n, v_n),
                            "u_norm": round(u_n, 4),
                            "v_norm": round(v_n, 4),
                            "conf": round(float(confs[i]), 3),
                            "x_m": round(float(x_m), 3),
                            "y_m": round(float(y_m), 3),
                            "graspable": True,
                        }
                    )

            hits = [d for d in detections if _is_target(str(d["label"]), target)]
            others = [d for d in detections if d not in hits]
            hits.sort(key=lambda d: d["conf"], reverse=True)
            others.sort(key=lambda d: d["conf"], reverse=True)
            targets = hits[:3]
            obstacles = others[:12]
            shown = (targets + others)[:6]
            if targets:
                notes = ",".join(f"{d['azimuth']}:{d['label']}" for d in shown)
            elif detections:
                notes = "see " + ",".join(d["label"] for d in detections[:6])
            else:
                notes = "not-found"

            payload = {
                "mode": "grasp",
                "source": "yolo-world",
                "notes": notes,
                "targets": targets,
                "best_target_id": 0 if targets else None,
                "obstacles": obstacles,
                "summary": notes,
            }
            ms = (time.time() - t0) * 1000.0
            print(f"OK {ms:.0f}", flush=True)
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            print("END", flush=True)
        except Exception as exc:
            print(f"ERR {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
