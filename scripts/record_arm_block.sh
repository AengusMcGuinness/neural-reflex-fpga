#!/bin/sh
# Record one paired block: raw EMG + band motion from Kinesis, and webcam arm pose.
# usage: scripts/record_arm_block.sh SESSION_DIR BLOCK_NAME 'STREAM_URL' [CAMERA_INDEX] [left|right] [follow|robot|full]

set -eu

if [ "$#" -lt 3 ]; then
  echo "usage: scripts/record_arm_block.sh SESSION_DIR BLOCK_NAME 'STREAM_URL' [CAMERA_INDEX] [left|right] [follow|robot|full]" >&2
  exit 2
fi

session=$1
block=$2
url=$3
camera=${4:-0}
side=${5:-right}
protocol=${6:-robot}
dir="$session/blocks/$block"
mkdir -p "$dir"

root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
python="$root/.venv/bin/python"
if [ ! -x "$python" ]; then
  python=python3
fi

# Band check first: relax / fist prompts; stops here if EMG wouldn't give usable grip data.
# Set SKIP_BAND_CHECK=1 to skip it.
if [ "${SKIP_BAND_CHECK:-0}" != "1" ]; then
  if ! "$python" -m neural_reflex.fist_check --url "$url" --output "$dir" --side "$side"; then
    echo "Band check failed: nothing was recorded. Fix it and run the same command again." >&2
    exit 1
  fi
  echo "Starting the recording: the camera window with the target arm is opening (it can take a few seconds)..."
fi

"$python" -m neural_reflex record-emg --url "$url" --output "$dir/emg.jsonl" \
  --motion-output "$dir/motion.jsonl" --block "$block" &
emg_pid=$!
"$python" -m neural_reflex record-arm --output "$dir/arm.jsonl" --block "$block" \
  --camera "$camera" --side "$side" --protocol "$protocol" &
arm_pid=$!

cleanup() {
  kill "$emg_pid" "$arm_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT
# The camera window guides a ~2 minute session and stops by itself (or press q);
# either way that ends the block for both recorders.
wait "$arm_pid" || true
