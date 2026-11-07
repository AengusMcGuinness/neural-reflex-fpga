#!/bin/sh
# Record one paired block: raw EMG + band motion from Kinesis, and webcam arm pose.
# usage: scripts/record_arm_block.sh SESSION_DIR BLOCK_NAME 'STREAM_URL' [CAMERA_INDEX] [left|right]

set -eu

if [ "$#" -lt 3 ]; then
  echo "usage: scripts/record_arm_block.sh SESSION_DIR BLOCK_NAME 'STREAM_URL' [CAMERA_INDEX] [left|right]" >&2
  exit 2
fi

session=$1
block=$2
url=$3
camera=${4:-0}
side=${5:-right}
dir="$session/blocks/$block"
mkdir -p "$dir"

root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
python="$root/.venv/bin/python"
if [ ! -x "$python" ]; then
  python=python3
fi

"$python" -m neural_reflex record-emg --url "$url" --output "$dir/emg.jsonl" \
  --motion-output "$dir/motion.jsonl" --block "$block" &
emg_pid=$!
"$python" -m neural_reflex record-arm --output "$dir/arm.jsonl" --block "$block" \
  --camera "$camera" --side "$side" &
arm_pid=$!

cleanup() {
  kill "$emg_pid" "$arm_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT
# The camera window guides a ~2 minute session and stops by itself (or press q);
# either way that ends the block for both recorders.
wait "$arm_pid" || true
