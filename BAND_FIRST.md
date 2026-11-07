# Start with the band, before the robot arm

The code here is a **CPU baseline** and recording harness. No Meta/Axo SDK API is assumed, and no FPGA inference is claimed yet. The raw EMG stream and a measured sample rate are the only vendor inputs needed for this phase.

## 1. Choose a raw EMG source

The event description says Axo will provide a Mac API but does not include a link. Ask Jason Hou for its **raw eight-channel EMG quickstart or sample code**, connection steps, actual sample rate, units, channel order, and timestamp semantics. This is likely the quickest route for a loaned hackathon band, but you do not have to wait to develop or inspect public examples. Do not build on the band's high-level gesture events: the Axo challenge requires raw EMG.

Public alternatives exist:

- [callbacked/neural-band-poc](https://github.com/callbacked/neural-band-poc/) reports a Mac dashboard with live eight-channel sEMG. Its README says to unpair the band from the Meta app and pair it again for the console, and notes experimental ADC values and dropped samples.
- [meta-neural-py](https://pypi.org/project/meta-neural-py/) is an experimental Python library for direct band access with EMG support. Its own package page says it has limited hardware testing and uses an explicit ownership/enrollment ceremony.

These are community projects, **not the Axo hackathon SDK or an official Meta raw EMG SDK**. Before changing pairing or ownership on a checkout band, ask Axo how they configured this device and whether they want you to use their API. You can read the community code and run its offline examples in parallel.

### This checkout band's Mac compatibility check

On November 6, 2026, the public `callbacked/neural-band-poc` client was installed locally. After the wearer accepted macOS's Bluetooth pairing prompt, the band connected and reported firmware `7fada0507764+` and L2CAP PSM `129`. The unmodified client expects PSM `255` and AirShield scheme `3`. The local ignored copy now discovers the PSM and uses scheme `26` for the newer firmware's parameter mask `27`. An encrypted identity reply passed its packet MAC check. **Bluetooth pairing and transport encryption work; raw EMG does not yet stream.**

The band closes its data channel immediately after an EndLinkSetup request, whether that request is sent alone, alongside the identity query, or after the identity certificate arrives. Without EndLinkSetup, the input-service metadata query receives a service-rejected response. The public [Air Gestures protocol source](https://github.com/896kb/Air-Gestures) describes a one-owner trust flow and says that claiming a band moves ownership away from its previous device. These observations are consistent with an owner/access gate, but they do not prove the exact reason for this band's rejection. **Do not transfer ownership or factory-reset this checkout band without Axo's instructions.** Ask Axo for its Mac raw-EMG API and the enrollment or owner-key steps for this specific loaner.

The installed dashboard is at `http://localhost:8765` while its local server runs. If necessary, restart it from `.local/neural-band-poc` with `.venv/bin/python -m dashboard.server`. That `.local` copy and its captures are ignored by Git. The dashboard is set to the newly paired band; the band may receive a new macOS Bluetooth identifier after re-pairing.

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

This is an **adapter shape**, not an SDK call. Register it using either Axo's sample or a verified community project's raw stream. Preserve raw units; do not rectify or convert to gesture labels in the bridge. If callbacks deliver batches, print one line per sample. If SDK timestamps use another unit, convert them explicitly or omit `t_ns`; the recorder always adds a Mac monotonic arrival timestamp.

## 2. Prove capture works

From this repo on the Mac:

```bash
python3 -m pip install -r requirements.txt
python3 -u your_actual_band_bridge.py | python3 -m neural_reflex capture --output data/raw/first.jsonl
python3 -m neural_reflex inspect --input data/raw/first.jsonl
```

Stop capture with Ctrl-C. The first command is a template until one raw-stream API is wired in. Check that all eight channels vary with rest, wrist movement, and pinch. `inspect` reports **Mac arrival** rate, which may differ from sensor sample rate when callbacks are batched. Use the source's actual rate for training. Raw recordings and subject-specific models under `data/` are ignored by Git.

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
