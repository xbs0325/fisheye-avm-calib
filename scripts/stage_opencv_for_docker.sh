#!/usr/bin/env bash
# Copy host CUDA OpenCV + shared-lib deps needed to import cv2 in Ubuntu 24.04.
# Uses readelf NEEDED (not host ldd) so /usr/local/cuda RPATH does not hide deps.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_opencv_cuda.sh"
SRC="${OPENCV_CUDA_PREFIX}"
DST="${ROOT}/docker/opencv-cuda"
if [[ ! -d "${SRC}/lib" ]]; then
  echo "missing OpenCV CUDA at ${SRC}" >&2
  exit 1
fi
mkdir -p "${DST}"
echo "[stage] ${SRC} -> ${DST}"
rsync -a --delete \
  --exclude 'share/opencv4/samples' \
  --exclude 'share/opencv4/doc' \
  "${SRC}/" "${DST}/"

DST="${DST}" python3 - <<'PY'
import os
import re
import shutil
import subprocess
from pathlib import Path

root = Path(os.environ["DST"])
lib_dst = root / "lib"
sites = list(root.glob("lib/python3.*/site-packages/cv2"))
if not sites:
    raise SystemExit(f"no cv2 package under {root}")
cv2_dir = sites[0]

(cv2_dir / "config.py").write_text(
    "import os\n\n"
    "BINARIES_PATHS = [\n"
    "    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "
    "'..', '..', '..'))\n"
    "] + BINARIES_PATHS\n"
)
print(f"[stage] patched {cv2_dir / 'config.py'}")
for p in sorted(cv2_dir.glob("config-3.*.py")):
    ver = p.name[len("config-") : -len(".py")]
    p.write_text(
        "import os\n\n"
        "PYTHON_EXTENSIONS_PATHS = [\n"
        f"    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python-{ver}')\n"
        "] + PYTHON_EXTENSIONS_PATHS\n"
    )
    print(f"[stage] patched {p}")

SKIP = {
    "linux-vdso.so.1", "libpthread.so.0", "libdl.so.2", "librt.so.1",
    "libm.so.6", "libc.so.6", "libgcc_s.so.1", "ld-linux-aarch64.so.1",
    "libstdc++.so.6", "libz.so.1",
}
FORCE_PREFIXES = (
    "libnpp", "libcudart", "libcublas", "libcufft", "libcudnn",
    "libavcodec", "libavformat", "libavutil", "libavdevice", "libavfilter",
    "libswscale", "libswresample", "libpostproc", "libnv", "libcuda.so",
)
FORCE = [
    "libnppig.so.13", "libnppial.so.13", "libnppicc.so.13", "libnppidei.so.13",
    "libnppist.so.13", "libnppif.so.13", "libnppim.so.13", "libnppitc.so.13",
    "libnppc.so.13", "libnpps.so.13", "libnppisu.so.13",
    "libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13", "libcufft.so.12",
    "libcudnn.so.9",
]


def needed_libs(so: Path) -> list[str]:
    try:
        out = subprocess.check_output(
            ["readelf", "-d", str(so)], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return []
    names = []
    for line in out.splitlines():
        if "NEEDED" not in line:
            continue
        m = re.search(r"\[([^\]]+)\]", line)
        if m:
            names.append(m.group(1))
    return names


def resolve_on_host(soname: str) -> Path | None:
    for base in (
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda/targets/sbsa-linux/lib"),
        Path("/usr/local/cuda-13.2/targets/sbsa-linux/lib"),
        Path("/lib/aarch64-linux-gnu"),
        Path("/usr/lib/aarch64-linux-gnu"),
        Path("/usr/lib/aarch64-linux-gnu/pulseaudio"),
        Path("/usr/lib/aarch64-linux-gnu/nvidia"),
        Path("/lib"),
        Path("/usr/lib"),
    ):
        cand = base / soname
        if cand.exists():
            return cand
    # recursive cheap search for private helper libs (pulseaudio etc.)
    for base in (Path("/usr/lib/aarch64-linux-gnu"), Path("/lib/aarch64-linux-gnu")):
        matches = list(base.rglob(soname))
        if matches:
            return matches[0]
    try:
        out = subprocess.check_output(["ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    for line in out.splitlines():
        if "=>" not in line:
            continue
        left = line.split("=>", 1)[0].strip().split()[0]
        if left == soname:
            return Path(line.split("=>", 1)[1].strip())
    return None


def staged_has(soname: str) -> bool:
    return (lib_dst / soname).exists()


def must_stage(soname: str) -> bool:
    if soname in SKIP:
        return False
    if staged_has(soname):
        return False
    # Hermetic image: stage every remaining NEEDED lib. Ubuntu base alone is not
    # enough for NVIDIA ffmpeg 8 / CUDA toolkit transitive deps.
    return True


def copy_lib(src: Path, soname: str) -> bool:
    real = src.resolve()
    dest_real = lib_dst / real.name
    changed = False
    if not dest_real.exists():
        shutil.copy2(real, dest_real)
        changed = True
    if soname != real.name:
        link = lib_dst / soname
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(real.name)
        changed = True
    return changed


pending: set[str] = set()
seen_elf: set[str] = set()
copied = 0


def scan_elf(so: Path) -> None:
    try:
        key = str(so.resolve())
    except Exception:
        key = str(so)
    if key in seen_elf or not so.exists():
        return
    seen_elf.add(key)
    for name in needed_libs(so):
        if must_stage(name):
            pending.add(name)


for so in list(lib_dst.glob("libopencv_*.so")) + list(
    root.glob("lib/python3.*/site-packages/cv2/python-3.*/cv2*.so")
):
    scan_elf(so)

for name in FORCE:
    if not staged_has(name):
        pending.add(name)

round_i = 0
while pending and round_i < 40:
    round_i += 1
    batch = sorted(pending)
    pending.clear()
    print(f"[stage] dep round {round_i}: need={len(batch)}")
    for name in batch:
        if staged_has(name):
            scan_elf(lib_dst / name)
            continue
        src = resolve_on_host(name)
        if src is None:
            print(f"[stage] WARN unresolved {name}")
            continue
        if copy_lib(src, name):
            copied += 1
            print(f"[stage] + {name} <- {src}")
        scan_elf(lib_dst / src.resolve().name)

cv2_so = next(root.glob("lib/python3.*/site-packages/cv2/python-3.*/cv2*.so"))
env = {
    "PATH": "/usr/bin:/bin",
    "LD_LIBRARY_PATH": f"{lib_dst}:/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu",
}
out = subprocess.check_output(["ldd", str(cv2_so)], text=True, env=env, stderr=subprocess.STDOUT)
missing = [ln.strip() for ln in out.splitlines() if "not found" in ln]
print(f"[stage] copied_new={copied} restricted_ldd_missing={len(missing)}")
for ln in missing[:50]:
    print("  ", ln)
if missing:
    # one more pass for restricted-ldd misses
    for ln in missing:
        name = ln.split()[0]
        if staged_has(name):
            continue
        src = resolve_on_host(name)
        if src is None:
            continue
        if copy_lib(src, name):
            copied += 1
            print(f"[stage] ldd-fix + {name} <- {src}")
    out = subprocess.check_output(["ldd", str(cv2_so)], text=True, env=env, stderr=subprocess.STDOUT)
    missing = [ln.strip() for ln in out.splitlines() if "not found" in ln]
    print(f"[stage] after fix restricted_ldd_missing={len(missing)}")
    for ln in missing[:30]:
        print("  ", ln)
PY

du -sh "${DST}"
echo "[stage] ok — docker build will COPY docker/opencv-cuda"
