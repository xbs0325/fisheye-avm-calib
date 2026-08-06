#!/usr/bin/env python3
"""
一键标定向导（CLI，可跳步）。

Web 视频流请用 GPU 版：`python3 -m avm.web_server`（或 scripts/run_web.sh）。
旧方案因 CPU remap/blend 掉到 1–2 FPS；新服务热路径强制 CUDA。

用法：
  source scripts/env_opencv_cuda.sh
  python3 -m avm.wizard                  # 交互菜单
  python3 -m avm.wizard --web            # 启动 GPU Web 引导
  python3 -m avm.wizard --all
  python3 -m avm.wizard --skip-intrinsics
  python3 -m avm.wizard --live-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AVM_DIR = Path(__file__).resolve().parent
CALIB_DIR = ROOT / "calib_results"
CONFIG_DIR = ROOT / "config"
ENV_SH = ROOT / "scripts" / "env_opencv_cuda.sh"

DIRECTIONS = ("front", "back", "left", "right")
INTRINSIC_RMS_WARN = 1.0
INTRINSIC_RMS_FAIL = 1.5


def _cuda_env() -> dict[str, str]:
    """Build env with CUDA OpenCV side-install (LD_LIBRARY_PATH before process start)."""
    env = os.environ.copy()
    env.pop("PYTHONNOUSERSITE", None)
    prefix = Path(
        env.get("OPENCV_CUDA_PREFIX", str(Path.home() / ".local" / "opencv-4.14.0-cuda"))
    )
    if not prefix.is_dir():
        alt = Path("/usr/local/opencv-4.14.0-cuda")
        if alt.is_dir():
            prefix = alt
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    lib = prefix / "lib"
    site = lib / py / "site-packages"
    env["OPENCV_CUDA_PREFIX"] = str(prefix)
    if lib.is_dir():
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib}:{ld}" if ld else str(lib)
    parts = [str(ROOT)]
    if site.is_dir():
        parts.append(str(site))
    old_pp = env.get("PYTHONPATH", "")
    if old_pp:
        parts.append(old_pp)
    env["PYTHONPATH"] = ":".join(parts)
    env["OpenCV_DIR"] = str(lib / "cmake" / "opencv4")
    return env


def _run(module: str, args: list[str], *, check: bool = True) -> int:
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n>>> {' '.join(cmd)}\n")
    ret = subprocess.call(cmd, cwd=str(ROOT), env=_cuda_env())
    if check and ret != 0:
        raise SystemExit(ret)
    return ret


def check_cuda() -> bool:
    code = (
        "import cv2; "
        "print(cv2.__file__); print(cv2.__version__); "
        "print('cuda_devs', cv2.cuda.getCudaEnabledDeviceCount())"
    )
    print("=" * 60)
    print("  CUDA OpenCV 检查")
    print("=" * 60)
    if ENV_SH.is_file():
        print(f"  env: {ENV_SH}")
    ret = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=_cuda_env(),
        capture_output=True,
        text=True,
    )
    print(ret.stdout or ret.stderr)
    if ret.returncode != 0:
        print("  ❌ 无法 import CUDA OpenCV。请先: source scripts/env_opencv_cuda.sh")
        return False
    if "cuda_devs 0" in (ret.stdout or "") or "cuda_devs" not in (ret.stdout or ""):
        # still print; getCudaEnabledDeviceCount may be 0
        if "cuda_devs 0" in (ret.stdout or ""):
            print("  ❌ CUDA device count = 0")
            return False
    print("  ✅ OK")
    return True


def check_intrinsics_quality(calib_dir: Path) -> dict:
    result = {"all_pass": True, "status": "pass", "details": {}}
    worst = "pass"
    for d in DIRECTIONS:
        path = calib_dir / f"{d}.json"
        if not path.is_file():
            result["details"][d] = {
                "rms": None,
                "status": "fail",
                "warnings": [f"missing {d}.json"],
            }
            result["all_pass"] = False
            worst = "fail"
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rms = data.get("rms", data.get("reproj_error"))
        ev = data.get("evaluation", {})
        warnings = list(ev.get("warnings", []))
        if rms is None:
            status, warnings = "fail", warnings + ["missing RMS"]
            result["all_pass"] = False
            worst = "fail"
        elif float(rms) > INTRINSIC_RMS_FAIL:
            status = "fail"
            warnings.append(f"RMS={float(rms):.3f} > {INTRINSIC_RMS_FAIL}")
            result["all_pass"] = False
            worst = "fail"
        elif float(rms) > INTRINSIC_RMS_WARN:
            status = "warn"
            warnings.append(f"RMS={float(rms):.3f} > {INTRINSIC_RMS_WARN}")
            if worst == "pass":
                worst = "warn"
        elif not ev.get("passed", True):
            status = ev.get("status", "fail")
            result["all_pass"] = False
            worst = "fail"
        else:
            status = "pass"
        result["details"][d] = {"rms": rms, "status": status, "warnings": warnings}
    result["status"] = worst
    return result


def print_intrinsics_report(report: dict) -> None:
    icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
    print("\n" + "=" * 60)
    print(f"  内参质量  {icon.get(report['status'], '?')} {report['status'].upper()}")
    print("=" * 60)
    for d in DIRECTIONS:
        det = report["details"].get(d, {})
        rms = det.get("rms")
        rms_s = f"{float(rms):.3f}px" if rms is not None else "N/A"
        print(f"  {d:8s}  RMS={rms_s:>10s}  {icon.get(det.get('status'), '?')}")
        for w in det.get("warnings", []):
            print(f"          ⚠️  {w}")


def check_extrinsics_quality(path: Path) -> dict:
    result = {
        "all_pass": True,
        "status": "pass",
        "details": {},
        "missing_cameras": [],
        "global_warnings": [],
    }
    if not path.is_file():
        result["all_pass"] = False
        result["status"] = "fail"
        result["global_warnings"].append(f"extrinsics missing: {path}")
        return result
    data = json.loads(path.read_text(encoding="utf-8"))
    homographies = data.get("homographies", {})
    qc = data.get("homography_qc", {})
    worst = "pass"
    for d in DIRECTIONS:
        if d not in homographies:
            result["missing_cameras"].append(d)
            result["all_pass"] = False
            worst = "fail"
            continue
        q = qc.get(d, {})
        status = q.get("status", "ok" if d in qc else "warn")
        warnings = list(q.get("warnings", []))
        if d not in qc:
            status, warnings = "warn", ["no QC data"]
            if worst == "pass":
                worst = "warn"
        elif status == "bad":
            result["all_pass"] = False
            worst = "fail"
        elif status == "warn" and worst == "pass":
            worst = "warn"
        result["details"][d] = {"status": status, "warnings": warnings}
    result["status"] = "fail" if worst == "fail" else worst
    if result["missing_cameras"]:
        result["global_warnings"].append(
            f"missing extrinsic cams: {'/'.join(result['missing_cameras'])}"
        )
    result["global_warnings"].extend(data.get("global_warnings") or [])
    # Known hard problem banner
    result["global_warnings"].append(
        "[known hard] Extrinsic H is hard to make fully automatic/accurate; "
        "seam error is expected — see docs/CALIBRATION_LESSONS.md"
    )
    return result


def print_extrinsics_report(report: dict) -> None:
    icon = {"pass": "✅", "warn": "⚠️", "fail": "❌", "ok": "✅", "bad": "❌"}
    print("\n" + "=" * 60)
    print(f"  外参质量  {icon.get(report['status'], '?')} {report['status'].upper()}")
    print("=" * 60)
    for d in DIRECTIONS:
        det = report["details"].get(d)
        if det is None:
            print(f"  {d:8s}  ❌ 缺失")
            continue
        print(f"  {d:8s}  {icon.get(det['status'], '?')} {det['status']}")
        for w in det.get("warnings", []):
            print(f"          ⚠️  {w}")
    for w in report.get("global_warnings", []):
        print(f"  [全局] ⚠️  {w}")


def confirm(prompt: str, *, force: bool) -> bool:
    if force:
        return True
    print(f"\n{prompt}")
    print("  继续 = Enter  |  跳过/取消 = Ctrl+C")
    try:
        input()
        return True
    except (KeyboardInterrupt, EOFError):
        print("\n已取消。")
        return False


def run_intrinsics(*, direction: str | None, force: bool) -> None:
    args = ["--calibrate"]
    if direction:
        args += ["--direction", direction]
    _run("avm.calibrate_intrinsics", args)
    report = check_intrinsics_quality(CALIB_DIR)
    print_intrinsics_report(report)
    if report["status"] == "fail" and not force:
        print("❌ 内参不合格。加 --force 可继续，或重采。")
        raise SystemExit(1)


def run_extrinsics(*, balance: float, force: bool) -> None:
    report = check_intrinsics_quality(CALIB_DIR)
    print_intrinsics_report(report)
    if report["status"] == "fail" and not force:
        print("❌ 内参缺失/不合格，无法可靠做外参。可用 --skip-intrinsics --force 强行试。")
        raise SystemExit(1)
    if report["status"] == "warn" and not confirm("内参质量偏低，仍做外参？", force=force):
        raise SystemExit(0)

    print("\n" + "=" * 60)
    print("  外参标定（SPACE 抓拍四路，ESC 存盘）")
    print(f"  extrinsic_balance={balance:.2f}")
    print("  ⚠️  外参是当前最大误差源，详见 docs/CALIBRATION_LESSONS.md")
    print("=" * 60)
    t0 = time.time()
    _run(
        "avm.calibrate_extrinsics",
        ["--capture", "--extrinsic-balance", str(balance)],
    )
    extr = CALIB_DIR / "extrinsics.json"
    if not extr.is_file() or extr.stat().st_mtime < t0 - 1:
        # fallback newest
        cands = list(CALIB_DIR.glob("extrinsics*.json"))
        if not cands:
            print("❌ 未生成 extrinsics*.json")
            raise SystemExit(1)
        newest = max(cands, key=lambda p: p.stat().st_mtime)
        if newest.resolve() != extr.resolve():
            extr.write_bytes(newest.read_bytes())
            print(f"[启用] extrinsics.json ← {newest.name}")

    erep = check_extrinsics_quality(extr)
    print_extrinsics_report(erep)
    if erep["status"] == "fail" and not force:
        print("❌ 外参 QC 失败。可 --force 仍进 live 看效果。")
        raise SystemExit(1)


def run_preview() -> None:
    _run("avm.preview_undistorted", [], check=False)


def run_live() -> None:
    extr = CALIB_DIR / "extrinsics.json"
    if not extr.is_file():
        print(f"❌ 缺少 {extr}，请先做外参或从旧工程拷贝。")
        raise SystemExit(1)
    print_extrinsics_report(check_extrinsics_quality(extr))
    _run("avm.live_bev", [], check=False)


def interactive_menu(*, balance: float, force: bool) -> None:
    while True:
        print(
            """
============================================================
  AVM GPU 向导   (本机窗口 / GPU Web)
============================================================
  1) 检查 CUDA OpenCV
  2) 内参标定（可跳过若已有 calib_results/*.json）
  3) 外参标定  ← 已知难点，误差易偏大
  4) 去畸变预览 (GPU remap)
  5) 实时 BEV 拼接 (GPU remap+warp+blend)
  6) 内参质量报告
  7) 外参质量报告
  8) 启动 GPU Web 引导 (MJPEG :8787)
  0) 退出
------------------------------------------------------------"""
        )
        try:
            choice = input("选择: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if choice == "0":
            return
        if choice == "1":
            check_cuda()
        elif choice == "2":
            run_intrinsics(direction=None, force=force)
        elif choice == "3":
            run_extrinsics(balance=balance, force=force)
        elif choice == "4":
            run_preview()
        elif choice == "5":
            run_live()
        elif choice == "6":
            print_intrinsics_report(check_intrinsics_quality(CALIB_DIR))
        elif choice == "7":
            print_extrinsics_report(check_extrinsics_quality(CALIB_DIR / "extrinsics.json"))
        elif choice == "8":
            _run("avm.web_server", [], check=False)
        else:
            print("无效选项")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AVM GPU 一键标定向导（可跳步）")
    p.add_argument("--all", action="store_true", help="内参→外参→live")
    p.add_argument("--skip-intrinsics", action="store_true", help="跳过内参，沿用已有 JSON")
    p.add_argument("--skip-extrinsics", action="store_true", help="跳过外参，直接 live")
    p.add_argument("--live-only", action="store_true", help="仅实时拼接")
    p.add_argument("--preview-only", action="store_true", help="仅去畸变预览")
    p.add_argument("--web", action="store_true", help="启动 GPU Web 引导服务")
    p.add_argument("--intrinsics-only", action="store_true")
    p.add_argument("--extrinsics-only", action="store_true")
    p.add_argument("--direction", choices=DIRECTIONS, default=None)
    p.add_argument("--force", action="store_true", help="跳过质量门控确认")
    p.add_argument(
        "--extrinsic-balance",
        type=float,
        default=0.8,
        help="外参去畸变 balance，推荐 0.7~0.9",
    )
    p.add_argument("--check-cuda", action="store_true")
    args = p.parse_args()
    if not 0.1 <= args.extrinsic_balance <= 1.0:
        p.error("--extrinsic-balance 必须在 [0.1, 1.0]")
    return args


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    CALIB_DIR.mkdir(parents=True, exist_ok=True)

    if args.check_cuda:
        raise SystemExit(0 if check_cuda() else 1)

    if args.web:
        if not check_cuda():
            raise SystemExit(1)
        _run("avm.web_server", [], check=False)
        return

    if args.preview_only:
        check_cuda()
        run_preview()
        return
    if args.live_only or (args.skip_intrinsics and args.skip_extrinsics):
        check_cuda()
        run_live()
        return
    if args.intrinsics_only:
        check_cuda()
        run_intrinsics(direction=args.direction, force=args.force)
        return
    if args.extrinsics_only:
        check_cuda()
        run_extrinsics(balance=args.extrinsic_balance, force=args.force)
        return

    if args.all or args.skip_intrinsics or args.skip_extrinsics:
        if not check_cuda():
            raise SystemExit(1)
        if not args.skip_intrinsics and not args.live_only:
            if args.all or not all((CALIB_DIR / f"{d}.json").is_file() for d in DIRECTIONS):
                run_intrinsics(direction=args.direction, force=args.force)
            else:
                print_intrinsics_report(check_intrinsics_quality(CALIB_DIR))
                print("  （已有四路内参，跳过采集；若要重标请去掉 --skip-intrinsics 并手动选菜单 2）")
        elif args.skip_intrinsics:
            print_intrinsics_report(check_intrinsics_quality(CALIB_DIR))
            print("  [--skip-intrinsics] 使用已有内参")

        if not args.skip_extrinsics:
            run_extrinsics(balance=args.extrinsic_balance, force=args.force)
        else:
            print("  [--skip-extrinsics] 跳过外参")

        run_live()
        return

    # default: interactive
    if not check_cuda():
        print("CUDA 检查失败，菜单仍可打开，但 live 可能回退 CPU。")
    interactive_menu(balance=args.extrinsic_balance, force=args.force)


if __name__ == "__main__":
    main()
