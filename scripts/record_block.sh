#!/bin/sh
# Record one paired block: raw EMG from Kinesis and webcam hand pose.
# usage: scripts/record_block.sh SESSION_DIR BLOCK_NAME 'STREAM_URL'

set -eu

if [ "$#" -ne 3 ]; then
  echo "usage: scripts/record_block.sh SESSION_DIR BLOCK_NAME 'STREAM_URL'" >&2
  exit 2
fi

session=$1
block=$2
url=$3
dir="$session/blocks/$block"
mkdir -p "$dir"

root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
python="$root/.venv/bin/python"
if [ ! -x "$python" ]; then
  python=python3
fi

"$python" -m neural_reflex record-emg --url "$url" --output "$dir/emg.jsonl" --block "$block" &
emg_pid=$!
"$python" -m neural_reflex record-pose --output "$dir/pose.jsonl" --block "$block" &
pose_pid=$!

cleanup() {
  kill "$emg_pid" "$pose_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT
wait "$emg_pid" "$pose_pid" || true
