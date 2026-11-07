# Pair and read the Axo Meta Neural Band

This is the working path for the Neo checkout band. On November 6, 2026, Axo Kinesis build 6.28 paired `meta band 0092` to this Apple Silicon Mac, confirmed the right wrist, and streamed raw EMG. A six-second read received 12,192 samples per channel; a separate three-second capture passed 6,048 samples through this repo's normalizer and recorder. The temporary URL and raw recordings stay local.

## Pairing on a collaborator's Mac

1. Use an Apple Silicon Mac running macOS 14 or newer. Download [Kinesis build 6.28 from Axo's release](https://github.com/Axo-Neurotech/Kinesis-Axo/releases/tag/v0.5.0-axo.6.28), unzip, and open `Kinesis.app`. Follow [Axo's participant setup guide](https://github.com/Axo-Neurotech/Kinesis-Axo/blob/main/docs/hackathon-setup.md). This Mac's copy is at `.local/Kinesis.app` and is ignored by Git.
2. Keep the assigned band nearby. Hold its button about three seconds and release when the light flashes white rapidly. Choose **Pair band** in Kinesis.
3. Complete Meta sign-in yourself and accept the macOS Bluetooth pairing prompt. Kinesis stores the resulting pairing identity in this Mac's Keychain. If the band is already assigned to another Meta account, ask Axo before proceeding. Do not reset or forget a working pairing.
4. Confirm the wrist on the **band** page. Optional gesture lessons and Mac Accessibility access can be skipped; neither is needed for the raw feed.

Pairing is per Mac. Do not try to connect the same band through the older experimental `.local/neural-band-poc` client while Kinesis owns the connection.

## Turn on and verify the raw feed

In Kinesis, open **stream**, enable **local sensor feed**, wait for **raw EMG live**, then choose **copy stream URL**. The feed starts off each time Kinesis opens, and the copied `http://127.0.0.1:8766/events?token=...` link changes after a restart. Treat it as a temporary local secret; do not paste it into the repo or chat. Its [API guide](https://github.com/Axo-Neurotech/Kinesis-Axo/blob/main/docs/sensor-stream.md) describes the stream.

From this repo in Terminal, download Axo's standard-library reader once and run a count-only check:

```bash
mkdir -p .local
curl -fL https://raw.githubusercontent.com/Axo-Neurotech/Kinesis-Axo/v0.5.0-axo.6.28/scripts/read-sensor-stream.py -o .local/read-sensor-stream.py
python3 .local/read-sensor-stream.py --url "$(pbpaste)" --seconds 6
```

Copy the URL from Kinesis immediately before running. The command displays event counts, not EMG values. Kinesis reports eight unsigned 16-bit ADC channels, 2,048 samples per second per channel, and 16 samples per batch. ADC counts are **not calibrated voltage**. Band timestamps and Mac monotonic host timestamps use different clocks.

## Capture real samples for analysis

```bash
python3 .local/read-sensor-stream.py --url "$(pbpaste)" --seconds 10 --jsonl \
  | python3 -m neural_reflex kinesis-normalize \
  | python3 -m neural_reflex capture --output data/raw/first.jsonl
python3 -m neural_reflex inspect --input data/raw/first.jsonl
python3 -m neural_reflex packetize --input data/raw/first.jsonl \
  --output data/private/fpga_packets.bin
```

`kinesis-normalize` flattens each Kinesis 16×8 EMG batch into raw eight-channel sample lines, preserving batch sequence, band timestamp, host receipt time, and connection ID. `capture` adds a Mac monotonic timestamp as each line arrives. `inspect` reports this **process arrival** rate, which is not the sensor rate because delivery is batched. `packetize` makes the fixed-size input described in [FPGA_FIRST.md](FPGA_FIRST.md). `data/raw`, `data/private`, `.local`, and `.bin` files are ignored by Git.

For a first labeled CPU baseline and live virtual gripper, follow [BAND_FIRST.md](BAND_FIRST.md). Its timed rest/pinch labels are coarse; the Axo challenge calls for paired webcam pose data and a subject-specific decoder. The FPGA and webcam paths are next work, not part of the completed pairing.
