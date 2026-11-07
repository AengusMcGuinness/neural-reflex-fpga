# Start with the band, before the robot arm

The code here is a **CPU baseline** and recording harness. No Meta/Axo SDK API is assumed, and no FPGA inference is claimed yet. The raw EMG stream and a measured sample rate are the only vendor inputs needed for this phase.

## 1. Get the raw EMG API

Ask Jason Hou for the **Mac raw eight-channel EMG quickstart or sample code**, connection steps, actual sample rate, units, channel order, and timestamp semantics. The event description mentions a Mac API but does not give a download link. Do not build on the band's high-level gesture events: the Axo challenge requires raw EMG.

Make a tiny bridge around the real SDK callback. For each *raw* sample, write one JSON line to stdout and flush it:

```python
import json
import sys

def on_raw_emg(eight_values, sensor_timestamp=None):
    row = {"emg": list(eight_values)}
    if sensor_timestamp is not None:
        row["t_ns"] = sensor_timestamp  # only if the SDK says it is nanoseconds
    print(json.dumps(row), flush=True)
```

This is an **adapter shape**, not an SDK call. Replace `on_raw_emg` registration with the method from Jason's sample. Preserve raw units; do not rectify or convert to gesture labels in the bridge. If callbacks deliver batches, print one line per sample. If SDK timestamps use another unit, convert them explicitly or omit `t_ns`; the recorder always adds a Mac monotonic arrival timestamp.

## 2. Prove capture works

From this repo on the Mac:

```bash
python3 -m pip install -r requirements.txt
python3 -u your_actual_band_bridge.py | python3 -m neural_reflex capture --output data/raw/first.jsonl
python3 -m neural_reflex inspect --input data/raw/first.jsonl
```

Stop capture with Ctrl-C. The first command is only a template until the SDK arrives. Check that all eight channels vary with rest, wrist movement, and pinch. `inspect` reports **Mac arrival** rate, which may differ from sensor sample rate when callbacks are batched. Use the SDK's actual rate for training. Raw recordings and subject-specific models under `data/` are ignored by Git.

## 3. Run the pipeline without the SDK

```bash
python3 -m neural_reflex simulate --output data/raw/synthetic.jsonl
python3 -m neural_reflex train --input data/raw/synthetic.jsonl --sample-rate 200 --output data/models/synthetic.json
python3 -m neural_reflex replay --input data/raw/synthetic.jsonl --model data/models/synthetic.json | head
```

The 200 Hz rate here belongs **only to the synthetic fixture**, not to the band. This exercise checks the file format, feature extraction, train/holdout split, and virtual gripper display. Its accuracy says nothing about real EMG performance.

## 4. Do a rough real-band calibration

The optional timed labels let you test the whole recording and model path while the webcam tracker is still being built:

```bash
python3 -u your_actual_band_bridge.py | python3 -m neural_reflex capture \
  --output data/raw/calibration.jsonl --schedule rest:3,pinch:3 --rounds 5
python3 -m neural_reflex train --input data/raw/calibration.jsonl \
  --sample-rate ACTUAL_SDK_RATE --output data/models/baseline.json
python3 -m neural_reflex replay --input data/raw/calibration.jsonl \
  --model data/models/baseline.json | head
```

Follow the `REST` and `PINCH` prompts on stderr. The last complete round is held out from training. The timed labels are coarse: transit movements are mislabeled near phase changes. They are useful for a first signal test, not the final Axo submission. Repeat on a fresh recording to see whether the classifier generalizes.

Then view live CPU output with a previously trained model:

```bash
python3 -u your_actual_band_bridge.py | python3 -m neural_reflex live \
  --model data/models/baseline.json --capture data/raw/live.jsonl
```

`live` is a CPU path. The eventual judged FPGA demo must replace its prediction step with the XEM7310 result and display which path is active.

## 5. Next integration after the raw stream is proven

Record synchronized webcam hand landmarks and host monotonic timestamps. Use thumb-to-index distance normalized by palm width as a continuous pinch label, discard windows with uncertain hand tracking, and split evaluation by full calibration rounds. Once that held-out result is useful, quantize the exact same features and model for the FPGA. The Medra arm can wait: a reliable, unseen live virtual-gripper trial is already a strong milestone.
