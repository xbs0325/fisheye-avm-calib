"""Remote OpenCV key injection via a small command file.

Web UI / other tools append one command per line; capture loops poll and
consume the oldest line, mapping it to a waitKey-style keycode.

Uses flock when available to reduce lost-key races between push and poll.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

ENV_VAR = "AVM_CALIB_CONTROL_FILE"

CMD_TO_KEY: dict[str, int] = {
    "space": ord(" "),
    "capture": ord(" "),
    "lock": ord(" "),
    "esc": 27,
    "quit": 27,
    "done": 27,
    "finish": 27,
    "q": ord("q"),
    "0": ord("0"),
    "unlock_all": ord("0"),
    "unlock_session": ord("0"),
    "1": ord("1"),
    "unlock_front": ord("1"),
    "2": ord("2"),
    "unlock_back": ord("2"),
    "3": ord("3"),
    "unlock_left": ord("3"),
    "4": ord("4"),
    "unlock_right": ord("4"),
    "r": ord("r"),
    "refine": ord("r"),
    "overlap": ord("r"),
    "s": ord("s"),
    "skip": ord("s"),
    "y": ord("y"),
    "yes": ord("y"),
    "accept": ord("y"),
    "ok": ord("y"),
    "n": ord("n"),
    "no": ord("n"),
    "reject": ord("n"),
    "retry": ord("n"),
    "enter": 13,
}


def resolve_control_file(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return str(explicit)
    env = os.environ.get(ENV_VAR, "").strip()
    return env or None


def ensure_control_file(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.is_file():
        p.write_text("", encoding="utf-8")
    return p


def clear_control_file(path: str | Path | None) -> None:
    if not path:
        return
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
    except OSError:
        pass


def _flock(fd, exclusive: bool) -> None:
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    except Exception:
        pass


def _funlock(fd) -> None:
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)
    except Exception:
        pass


def push_control_cmd(path: str | Path, cmd: str) -> None:
    """Append one command for the capture loop to consume."""
    token = (cmd or "").strip().lower()
    if not token:
        raise ValueError("空命令")
    if token not in CMD_TO_KEY:
        raise ValueError(f"未知命令: {cmd}（支持: {', '.join(sorted(CMD_TO_KEY))}）")
    p = ensure_control_file(path)
    with open(p, "a+", encoding="utf-8") as f:
        _flock(f.fileno(), exclusive=True)
        try:
            f.seek(0, os.SEEK_END)
            f.write(token + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            _funlock(f.fileno())


def poll_control_key(path: str | Path | None) -> Optional[int]:
    """Pop the oldest command and return its keycode, or None."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with open(p, "r+", encoding="utf-8") as f:
            _flock(f.fileno(), exclusive=True)
            try:
                text = f.read()
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                if not lines:
                    return None
                first, rest = lines[0].lower(), lines[1:]
                f.seek(0)
                f.truncate(0)
                if rest:
                    f.write("\n".join(rest) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                _funlock(f.fileno())
    except OSError:
        return None
    return CMD_TO_KEY.get(first)


def merge_wait_key(opencv_key: int, control_path: str | Path | None) -> int:
    """Prefer a pending remote command over the local keyboard key."""
    remote = poll_control_key(control_path)
    if remote is not None:
        return remote & 0xFF
    return opencv_key & 0xFF
