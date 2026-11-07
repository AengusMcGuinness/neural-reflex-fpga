# Start with the band, before the robot arm

The Axo Kinesis app is now paired with this checkout band, and its live raw-EMG feed has been tested on this Mac. The code here is still a **CPU baseline** and recording harness; FPGA inference has not been implemented. Follow [KINESIS.md](KINESIS.md) for the working pairing, stream, capture, and packetization path.

## Working source: Axo Kinesis

On November 6, 2026, Kinesis build 6.28 connected to `meta band 0092` with the right wrist confirmed. Its local sensor feed delivered **12,192 raw samples per EMG channel in six seconds**, plus gyro and orientation events. A second three-second test passed **6,048 raw samples** through `kinesis-normalize` into this repo's recorder. Axo's reported EMG format is eight channels of unsigned 16-bit ADC counts at 2,048 samples per second per channel, delivered in 16-sample batches. The recorded samples and temporary feed URL stay local and are not committed.

Kinesis is running from `.local/Kinesis.app` in this workspace. Its feed starts off each time the app opens. Do not reset or forget the working pairing just to update software. The earlier direct Bluetooth attempt below is historical context; use Kinesis for this band.

## Historical direct-Bluetooth investigation

Before the [Kinesis Axo build](https://github.com/Axo-Neurotech/Kinesis-Axo) was supplied, the event description did not include an API link. We investigated public alternatives. Do not build on the band's high-level gesture events: the Axo challenge requires raw EMG.

Public alternatives exist:

- [callbacked/neural-band-poc](https://github.com/callbacked/neural-band-poc/) reports a Mac dashboard with live eight-channel sEMG. Its README says to unpair the band from the Meta app and pair it again for the console, and notes experimental ADC values and dropped samples.
- [meta-neural-py](https://pypi.org/project/meta-neural-py/) is an experimental Python library for direct band access with EMG support. Its own package page says it has limited hardware testing and uses an explicit ownership/enrollment ceremony.

These are community projects, **not the Axo hackathon app or an official Meta raw EMG SDK**. The Kinesis connection now works; do not change its pairing to retry them unless Axo instructs you to.

### This checkout band's Mac compatibility check

On November 6, 2026, the public `callbacked/neural-band-poc` client was installed locally. After the wearer accepted macOS's Bluetooth pairing prompt, the band connected and reported firmware `7fada0507764+` and L2CAP PSM `129`. The unmodified client expects PSM `255` and AirShield scheme `3`. The local ignored copy discovers the PSM and uses scheme `26` for the newer firmware's parameter mask `27`. An encrypted identity reply passed its packet MAC check. **That direct client did not stream raw EMG; Kinesis now does.**

The direct client closed its data channel immediately after an EndLinkSetup request. Without EndLinkSetup, its input-service metadata query received a service-rejected response. These observations did not establish the exact reason for rejection. The Axo Kinesis app subsequently completed Meta sign-in, paired the band, and streamed raw EMG. **Do not factory-reset this working pairing.**

The installed dashboard is at `http://localhost:8765` while its local server runs. If necessary, restart it from `.local/neural-band-poc` with `.venv/bin/python -m dashboard.server`. That `.local` copy and its captures are ignored by Git. The dashboard is set to the newly paired band; the band may receive a new macOS Bluetooth identifier after re-pairing.

## Capture raw EMG from Kinesis

From this repo on the Mac:

```bash
mkdir -p .local
curl -fL https://raw.githubusercontent.com/Axo-Neurotech/Kinesis-Axo/v0.5.0-axo.6.28/scripts/read-sensor-stream.py -o .local/read-sensor-stream.py
python3 .local/read-sensor-stream.py --url "$(pbpaste)" --seconds 10 --jsonl \
  | python3 -m neural_reflex kinesis-normalize \
  | python3 -m neural_reflex capture --output data/raw/first.jsonl
python3 -m neural_reflex inspect --input data/raw/first.jsonl
```

First select **copy stream URL** in Kinesis's **stream** tab. The URL is temporary; do not paste it into GitHub or chat. Check that all eight channels vary with rest, wrist movement, and pinch. `inspect` reports **Mac process arrival** rate, which differs from the sensor rate because Kinesis sends 16-sample batches. Use **2,048 Hz** from Kinesis's EMG configuration for training. Raw recordings and subject-specific models under `data/` are ignored by Git.

## Synthetic smoke test

```bash
python3 -m neural_reflex simulate --output data/raw/synthetic.jsonl
python3 -m neural_reflex train --input data/raw/synthetic.jsonl --sample-rate 200 --output data/models/synthetic.json
python3 -m neural_reflex replay --input data/raw/synthetic.jsonl --model data/models/synthetic.json | head
```

The 200 Hz rate here belongs **only to the synthetic fixture**, not to the band. This exercise checks the file format, feature extraction, train/holdout split, and virtual gripper display. Its accuracy says nothing about real EMG performance.

## Rough real-band calibration

The optional timed labels let you test the whole recording and model path while the webcam tracker is still being built:

```bash
python3 .local/read-sensor-stream.py --url "$(pbpaste)" --seconds 31 --jsonl \
  | python3 -m neural_reflex kinesis-normalize \
  | python3 -m neural_reflex capture --output data/raw/calibration.jsonl \
      --schedule rest:3,pinch:3 --rounds 5
python3 -m neural_reflex train --input data/raw/calibration.jsonl \
  --sample-rate 2048 --output data/models/baseline.json
python3 -m neural_reflex replay --input data/raw/calibration.jsonl \
  --model data/models/baseline.json | head
```

Follow the `REST` and `PINCH` prompts on stderr. The last complete round is held out from training. The timed labels are coarse: transit movements are mislabeled near phase changes. They are useful for a first signal test, not the final Axo submission. Repeat on a fresh recording to see whether the classifier generalizes.

Then view live CPU output with a previously trained model:

```bash
python3 .local/read-sensor-stream.py --url "$(pbpaste)" --seconds 30 --jsonl \
  | python3 -m neural_reflex kinesis-normalize \
  | python3 -m neural_reflex live --model data/models/baseline.json \
      --capture data/raw/live.jsonl
```

`live` is a CPU path. The eventual judged FPGA demo must replace its prediction step with the XEM7310 result and display which path is active.

## Next integration

Record synchronized webcam hand landmarks and host monotonic timestamps. Use thumb-to-index distance normalized by palm width as a continuous pinch label, discard windows with uncertain hand tracking, and split evaluation by full calibration rounds. Once that held-out result is useful, quantize the exact same features and model for the FPGA. The Medra arm can wait: a reliable, unseen live virtual-gripper trial is already a strong milestone.
