#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
driver_dir="$repo_root/.local/hiwonder-SoArm-101"
venv_dir="$repo_root/.local/arm-venv"
revision="a24998f7ba3c77ea445b48c92ad15c14a50e492a"
mkdir -p "$repo_root/.local"

if [[ ! -d "$driver_dir/.git" ]]; then
  git -c filter.lfs.process= -c filter.lfs.smudge=cat -c filter.lfs.required=false \
    clone https://github.com/Hiwonder/hiwonder-SoArm-101.git "$driver_dir"
fi

if [[ "$(git -C "$driver_dir" rev-parse HEAD)" != "$revision" ]]; then
  git -C "$driver_dir" fetch origin "$revision"
  git -C "$driver_dir" -c filter.lfs.process= -c filter.lfs.smudge=cat \
    -c filter.lfs.required=false checkout --detach "$revision"
fi

export UV_CACHE_DIR="$repo_root/.local/uv-cache"
if [[ ! -x "$venv_dir/bin/python" ]]; then
  uv venv --python 3.13 "$venv_dir"
fi
uv pip install --python "$venv_dir/bin/python" \
  pyserial==3.5 deepdiff==9.1.0 numpy==2.5.4 tqdm==4.70.1

PYTHONPATH="$driver_dir/src" "$venv_dir/bin/python" -c \
  'from lerobot.motors.hiwonder import HiwonderMotorsBus; print("Hiwonder driver ready")'
