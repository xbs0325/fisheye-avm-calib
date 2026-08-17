"""Structured perception message schema (versioned for future ROS2 mapping)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

SCHEMA_VERSION = 1
FRAME_ID = "base_link"

Azimuth = Literal["f", "b", "l", "r", "fl", "fr", "bl", "br", "center", "unknown"]
FreeDir = Literal["front", "back", "left", "right"]


@dataclass
class Obstacle:
    label: str
    azimuth: str = "unknown"
    u_norm: Optional[float] = None
    v_norm: Optional[float] = None
    conf: float = 0.0
    x_m: Optional[float] = None
    y_m: Optional[float] = None
    radius_m: Optional[float] = None


@dataclass
class GraspTarget:
    label: str
    u_norm: Optional[float] = None
    v_norm: Optional[float] = None
    conf: float = 0.0
    graspable: bool = True
    x_m: Optional[float] = None
    y_m: Optional[float] = None
    yaw_deg: Optional[float] = None  # base_link: 0=forward, +left, −right
    range_m: Optional[float] = None
    azimuth: str = "unknown"


@dataclass
class NavResult:
    mode: Literal["nav"] = "nav"
    summary: str = ""
    obstacles: list[Obstacle] = field(default_factory=list)
    free_dirs: list[str] = field(default_factory=list)
    uncertain: list[str] = field(default_factory=list)
    free_frac: Optional[float] = None
    source: str = "vlm"


@dataclass
class GraspResult:
    mode: Literal["grasp"] = "grasp"
    targets: list[GraspTarget] = field(default_factory=list)
    best_target_id: Optional[int] = None
    notes: str = ""
    summary: str = ""
    turn_hint: str = ""  # chassis: "L35 1.2m (x,y)"
    source: str = ""


@dataclass
class PerceptionEvent:
    """Envelope published on the perception bus / JSONL log."""

    schema_version: int = SCHEMA_VERSION
    frame_id: str = FRAME_ID
    stamp_s: float = 0.0
    mode: str = "nav"
    valid: bool = False
    infer_ms: float = 0.0
    summary: str = ""
    raw_text: str = ""
    nav: Optional[NavResult] = None
    grasp: Optional[GraspResult] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Drop empty nested None branches for cleaner logs
        if d.get("nav") is None:
            d.pop("nav", None)
        if d.get("grasp") is None:
            d.pop("grasp", None)
        if d.get("error") is None:
            d.pop("error", None)
        return d

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def _clip01(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return max(0.0, min(1.0, x))


def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def _conf(v: Any, default: float = 0.5) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x != x:
        return default
    return max(0.0, min(1.0, x))


def _close_truncated_json(s: str) -> str:
    """Close open quotes / braces so a cut-off VLM object can still parse."""
    out = s.rstrip()
    in_str = False
    escape = False
    stack: list[str] = []
    for ch in out:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif stack and ch == stack[-1]:
            stack.pop()
    if in_str:
        out += '"'
    out = re.sub(r",\s*$", "", out)
    while stack:
        out += stack.pop()
    return out


def generated_json_closed(generated_after_open_brace: str) -> bool:
    """True when a JSON object that started with a prefilled '{' is closed.

    Worker prefills '{' in the prompt; decoder output is the rest. Stop on the
    matching *root* '}', not the first nested obstacle/target object.
    """
    depth = 1
    in_str = False
    escape = False
    for ch in generated_after_open_brace or "":
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth <= 0:
                return True
    return False


def finalize_vlm_json(text: str) -> str:
    """Clip to the root object, or close braces if generation was cut off."""
    s = (text or "").strip().replace("\n", " ")
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    if not s.startswith("{"):
        start = s.find("{")
        s = s[start:] if start >= 0 else "{" + s
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(s):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[: i + 1]
    return _close_truncated_json(s)


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Pull the first JSON object (fences, extra text, truncated OK)."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    start = s.find("{")
    if start < 0:
        return None
    s = s[start:]
    clipped = finalize_vlm_json(s)
    candidates = [clipped, s]
    end = s.rfind("}")
    if end > 0:
        candidates.append(s[: end + 1])
    for blob in candidates:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_nav_payload(data: dict[str, Any]) -> NavResult:
    obstacles: list[Obstacle] = []
    for item in data.get("obstacles") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "obstacle").strip() or "obstacle"
        # Reject prompt leakage like "f|b|l|r|..."
        az = str(item.get("azimuth") or "unknown").lower().strip()
        if "|" in az or az not in (
            "f",
            "b",
            "l",
            "r",
            "fl",
            "fr",
            "bl",
            "br",
            "center",
            "unknown",
        ):
            az = "unknown"
        # Skip bogus "vehicle itself" at image center
        u_n = _clip01(item.get("u_norm"))
        v_n = _clip01(item.get("v_norm"))
        if label in ("地面小车", "小车", "vehicle", "car", "ego", "robot"):
            if u_n is not None and v_n is not None and abs(u_n - 0.5) < 0.12 and abs(v_n - 0.5) < 0.12:
                continue
        obstacles.append(
            Obstacle(
                label=label,
                azimuth=az,
                u_norm=u_n,
                v_norm=v_n,
                conf=_conf(item.get("conf")),
                x_m=_opt_float(item.get("x_m")),
                y_m=_opt_float(item.get("y_m")),
                radius_m=(
                    float(item["radius_m"])
                    if item.get("radius_m") is not None
                    else None
                ),
            )
        )
    free_dirs = []
    _alias = {"f": "front", "b": "back", "l": "left", "r": "right"}
    for d in data.get("free_dirs") or []:
        key = str(d).lower().strip()
        key = _alias.get(key, key)
        if key in ("front", "back", "left", "right") and key not in free_dirs:
            free_dirs.append(key)
    # If VLM dumps all 4 dirs uninformatively, clear them; seg "all clear" keeps them
    source = str(data.get("source") or "vlm")
    if len(set(free_dirs)) >= 4 and source != "seg":
        free_dirs = []
    yolo = source in ("yolo-world", "yoloe")
    seen_obs: set[tuple] = set()
    uniq_obs: list[Obstacle] = []
    for o in obstacles:
        if yolo and o.u_norm is not None and o.v_norm is not None:
            key = (o.label, round(o.u_norm, 2), round(o.v_norm, 2))
        else:
            key = (o.label, o.azimuth)
        if key in seen_obs:
            continue
        seen_obs.add(key)
        uniq_obs.append(o)
    obstacles = uniq_obs[:12] if yolo else uniq_obs[:4]
    uncertain = [str(u) for u in (data.get("uncertain") or [])]
    summary = str(data.get("summary") or "").strip()
    # Drop prompt placeholders the model sometimes copies verbatim
    if summary in ("简短中文", "简短中文。", "summary", "TODO", "示例"):
        if obstacles:
            o0 = obstacles[0]
            summary = f"{o0.azimuth}:{o0.label}"
        elif free_dirs:
            summary = f"clear:{','.join(free_dirs)}"
        else:
            summary = "scene unclear"
    free_frac = None
    if data.get("free_frac") is not None:
        try:
            free_frac = float(data["free_frac"])
        except (TypeError, ValueError):
            free_frac = None
    return NavResult(
        summary=summary,
        obstacles=obstacles,
        free_dirs=free_dirs,
        uncertain=uncertain,
        free_frac=free_frac,
        source=source,
    )


_GRASP_LEAK_NOTES = {
    "右前有瓶子",
    "右前有目标",
    "右前有bottle",
    "右前有鼠标",
    "短描述",
    "简短中文",
    "简短中文。",
    "notes",
    "TODO",
    "示例",
}
_NOTE_AZ_PATTERNS = (
    (re.compile(r"右前|前右"), "fr"),
    (re.compile(r"左前|前左"), "fl"),
    (re.compile(r"右后|后右"), "br"),
    (re.compile(r"左后|后左"), "bl"),
    (re.compile(r"前方|前边|前面"), "f"),
    (re.compile(r"后方|后边|后面"), "b"),
    (re.compile(r"右侧|右边"), "r"),
    (re.compile(r"左侧|左边"), "l"),
)


def _azimuth_from_notes(notes: str) -> str:
    s = notes or ""
    for pat, az in _NOTE_AZ_PATTERNS:
        if pat.search(s):
            return az
    return "unknown"


def _label_from_text(text: str) -> str:
    s = text or ""
    low = s.lower()
    if re.search(r"鼠标", s) or "mouse" in low or low.startswith("comp"):
        return "mouse"
    if re.search(r"瓶|杯", s) or "bottle" in low or "cup" in low:
        return "bottle"
    return ""


def parse_grasp_payload(data: dict[str, Any]) -> GraspResult:
    notes = str(data.get("notes") or data.get("summary") or "").strip()
    yolo = str(data.get("source") or "") in ("yolo-world", "yoloe")
    leak = notes in _GRASP_LEAK_NOTES or notes.replace(" ", "") in {
        "右前有瓶子",
        "右前有目标",
        "右前有鼠标",
        "短描述",
    }
    if leak:
        notes = ""
    guessed = _azimuth_from_notes(notes)
    targets: list[GraspTarget] = []
    for item in data.get("targets") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "object").strip() or "object"
        label = _label_from_text(label) or label
        graspable = item.get("graspable", True)
        if isinstance(graspable, str):
            graspable = graspable.lower() in ("1", "true", "yes")
        az = str(item.get("azimuth") or "unknown").lower().strip()
        if "|" in az or az not in (
            "f",
            "b",
            "l",
            "r",
            "fl",
            "fr",
            "bl",
            "br",
            "center",
            "unknown",
        ):
            az = "unknown"
        u_n = _clip01(item.get("u_norm"))
        v_n = _clip01(item.get("v_norm"))
        has_uv = u_n is not None and v_n is not None
        if az in ("unknown", "center"):
            az = guessed if guessed != "unknown" else az
        conf = _conf(item.get("conf"), default=0.6)
        # VLM essays need a compass bin; YOLO boxes already have pixels.
        if not yolo and not has_uv and az in ("unknown", "center"):
            continue
        conf_min = 0.08 if yolo else 0.35
        if conf < conf_min:
            continue
        targets.append(
            GraspTarget(
                label=label,
                u_norm=u_n,
                v_norm=v_n,
                conf=conf,
                graspable=bool(graspable),
                azimuth=az if az not in ("unknown", "center") else (guessed if guessed != "unknown" else az),
                x_m=_opt_float(item.get("x_m")),
                y_m=_opt_float(item.get("y_m")),
            )
        )
    if not targets and not yolo:
        label = _label_from_text(notes)
        if guessed != "unknown" and label:
            targets = [GraspTarget(label=label, azimuth=guessed, conf=0.55)]
    best = data.get("best_target_id")
    best_id: Optional[int] = None
    if best is not None:
        try:
            best_id = int(best)
        except (TypeError, ValueError):
            best_id = None
    if best_id is not None and not (0 <= best_id < len(targets)):
        best_id = 0 if targets else None
    elif best_id is None and targets:
        best_id = max(range(len(targets)), key=lambda i: targets[i].conf)
    if not targets:
        if not yolo or notes in ("", "not-found", "未找到"):
            notes = "未找到"
    elif not notes:
        notes = f"{targets[0].azimuth}:{targets[0].label}"
    # VLM duplicates flicker; YOLO-World keeps several boxes on the HUD.
    if not yolo and best_id is not None and targets:
        targets = [targets[best_id]]
        best_id = 0
    return GraspResult(
        targets=targets,
        best_target_id=best_id,
        notes=notes,
        summary=notes,
        source=str(data.get("source") or ""),
    )


def parse_vlm_response(
    text: str,
    *,
    mode: str,
    infer_ms: float = 0.0,
    stamp_s: Optional[float] = None,
) -> PerceptionEvent:
    """Validate VLM text → PerceptionEvent. On failure: valid=False + summary fallback."""
    stamp = float(stamp_s if stamp_s is not None else time.time())
    mode = "grasp" if mode == "grasp" else "nav"
    raw = (text or "").strip()
    obj = _extract_json_object(raw)

    if obj is None:
        summary = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()[:240]
        nav = None
        grasp = None
        if mode == "nav" and summary:
            nav = NavResult(summary=summary, source="vlm")
        elif mode == "grasp" and summary:
            guessed = _azimuth_from_notes(summary)
            label = _label_from_text(summary)
            if guessed != "unknown" and label:
                grasp = GraspResult(
                    targets=[GraspTarget(label=label, azimuth=guessed, conf=0.5)],
                    best_target_id=0,
                    notes=summary[:80],
                    summary=summary[:80],
                )
            else:
                grasp = GraspResult(notes="未找到", summary="未找到")
                summary = "未找到"
        return PerceptionEvent(
            stamp_s=stamp,
            mode=mode,
            valid=bool(summary),
            infer_ms=infer_ms,
            summary=summary,
            raw_text=raw,
            nav=nav,
            grasp=grasp,
            error=None if summary else "json_parse_failed",
        )

    try:
        if mode == "grasp":
            grasp = parse_grasp_payload(obj)
            summary = grasp.notes or grasp.summary
            return PerceptionEvent(
                stamp_s=stamp,
                mode="grasp",
                valid=True,
                infer_ms=infer_ms,
                summary=summary,
                raw_text=raw,
                grasp=grasp,
            )
        nav = parse_nav_payload(obj)
        return PerceptionEvent(
            stamp_s=stamp,
            mode="nav",
            valid=True,
            infer_ms=infer_ms,
            summary=nav.summary,
            raw_text=raw,
            nav=nav,
        )
    except Exception as exc:
        return PerceptionEvent(
            stamp_s=stamp,
            mode=mode,
            valid=False,
            infer_ms=infer_ms,
            summary=raw[:240],
            raw_text=raw,
            error=f"schema_error: {exc}",
        )


NAV_PROMPT = (
    "俯视环视：上=前,下=后,左=左,右=右。中心是车体。只做语义，不要坐标，不要报地砖。"
    "禁止markdown/代码块。只输出一行紧凑JSON，不要换行不要空格装饰。"
    "字段：mode,summary,obstacles[{label,azimuth,conf}],free_dirs,uncertain。"
    "azimuth=f/b/l/r/fl/fr/bl/br；label=person/chair/carton/cable/door/other。"
    "summary必须自己写。禁止编造。"
)

CAPTION_PROMPT = (
    "This is a top-down surround stitch from a chassis robot "
    "(up=front, down=back, left/right=sides; the image center is usually the vehicle blind zone, ignore it). "
    "Write two or three plain English sentences: what to watch for around the vehicle "
    "(people, boxes, cables, steps, chairs) and roughly which side; "
    "which side looks more open to drive through; say uncertain if it is unclear. "
    "Base it only on the image. Do not invent lane markings. Do not output JSON or markdown."
)

_AZ_CN = {
    "f": "前",
    "b": "后",
    "l": "左",
    "r": "右",
    "fl": "左前",
    "fr": "右前",
    "bl": "左后",
    "br": "右后",
}

GRASP_PROMPT_TEMPLATE = (
    "俯视：上=前,下=后,左=左,右=右。中心车体忽略。找「{target}」。{hint}{occ}"
    "禁止描写地砖/椅子/纸箱。notes不超过8个字。"
    "只输出一行JSON（字段名照抄，方位按画面填写）："
    '{{"mode":"grasp","notes":"短描述","targets":'
    '[{{"label":"mouse","azimuth":"f","conf":0.7}}]}}'
    "azimuth=f/b/l/r/fl/fr/bl/br。没有则targets=[]。"
)


def grasp_prompt(target: str, occ_az: str = "") -> str:
    t = (target or "object").strip() or "object"
    hint = ""
    key = t.lower().replace(" ", "")
    if key in ("mouse", "鼠标", "computermouse"):
        t = "computer mouse"
        hint = "是电脑鼠标外设（可手持的小块塑料，黑/灰/白/彩色都算），不是老鼠、不是地砖。"
    occ = ""
    azs: list[str] = []
    for tok in (occ_az or "").replace(";", ",").split(","):
        a = tok.strip().lower()
        if a in _AZ_CN and a not in azs:
            azs.append(a)
    if azs:
        cn = "、".join(_AZ_CN[a] for a in azs)
        occ = f"占用提示：{cn}方向有物体，优先看这些方向。"
    return GRASP_PROMPT_TEMPLATE.format(target=t, hint=hint, occ=occ)
