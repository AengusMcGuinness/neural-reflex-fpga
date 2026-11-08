#!/bin/sh
# Record one puppet block: raw EMG + band motion from Kinesis, and the leader arm's encoders
# while you move the arm by hand with torque off (the encoders are the training labels).
# usage: scripts/record_puppet_block.sh SESSION_DIR BLOCK_NAME 'STREAM_URL' ARM_PORT

set -eu

if [ "$#" -lt 4 ]; then
  echo "usage: scripts/record_puppet_block.sh SESSION_DIR BLOCK_NAME 'STREAM_URL' ARM_PORT" >&2
  exit 2
fi

session=$1
block=$2
url=$3
port=$4
dir="$session/blocks/$block"
mkdir -p "$dir"

root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
python="$root/.venv/bin/python"
if [ ! -x "$python" ]; then
  python=python3
fi
arm_python="$root/.local/arm-venv/bin/python"
driver_src="$root/.local/hiwonder-SoArm-101/src"
if [ ! -x "$arm_python" ] || [ ! -d "$driver_src" ]; then
  echo "Run ./scripts/setup_leader_arm.sh first" >&2
  exit 1
fi

"$python" -m neural_reflex record-emg --url "$url" --output "$dir/emg.jsonl" \
  --motion-output "$dir/motion.jsonl" --block "$block" &
emg_pid=$!

cleanup() {
  kill "$emg_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT
sleep 1   # let the EMG stream start before the routine does
cd "$root"
PYTHONPATH="$driver_src${PYTHONPATH:+:$PYTHONPATH}" "$arm_python" -m neural_reflex.record_puppet \
  --port "$port" --output "$dir/arm_encoders.jsonl" --block "$block" || true
