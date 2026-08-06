#!/usr/bin/env bash
# Run AVM Web container with Jetson CSI (tegra-video) device nodes.
# /dev/video* alone is NOT enough — need media + capture-vi-channel*.
#
# Usage:
#   ./scripts/docker_run_web.sh                 # foreground web server
#   ./scripts/docker_run_web.sh bash -lc '...'  # override command
#   DOCKER_RUN_OPTS='-d --name avm-web' ./scripts/docker_run_web.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${AVM_IMAGE:-leucushc/avm-gpu:0.2.0}"

devs=()
for n in 0 1 2 3; do
  [[ -e "/dev/video${n}" ]] && devs+=(--device "/dev/video${n}")
done
[[ -e /dev/media0 ]] && devs+=(--device /dev/media0)
[[ -e /dev/camsync ]] && devs+=(--device /dev/camsync)
for p in /dev/capture-vi-channel*; do
  [[ -e "$p" ]] && devs+=(--device "$p")
done
for p in /dev/capture-isp-channel*; do
  [[ -e "$p" ]] && devs+=(--device "$p")
done

if [[ ${#devs[@]} -lt 5 ]]; then
  echo "WARN: few camera devices found (${#devs[@]}). CSI may fail." >&2
fi

# shellcheck disable=SC2086
exec docker run --rm \
  ${DOCKER_RUN_OPTS:-} \
  --runtime nvidia --network host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  --group-add video \
  "${devs[@]}" \
  -v "${ROOT}/config:/app/config" \
  -v "${ROOT}/calib_results:/app/calib_results" \
  -v "${ROOT}/output:/app/output" \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  "${IMAGE}" \
  "$@"
