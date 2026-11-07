# Neural Reflex FPGA

**Neo Hackathon 2026 concept:** turn one person's raw muscle signals into continuous, personalized control through a real FPGA. The first live demo is a virtual gripper. A second operator and a physical robot arm are stretch goals if the necessary sponsor access is confirmed.

## The demo

An Axo band streams raw eight-channel EMG to a Mac. During a short calibration, a webcam tracks the wearer's hand and supplies paired labels for pinch and hand opening. We train a subject-specific decoder, quantize its weights, and run streaming feature extraction and inference on an Opal Kelly XEM7310. The decoded signal controls a gripper in a simple virtual scene.

If we get access to the Medra arm and its approved control interface, the same output can drive a bounded physical grasp. If we get a second band and know the teammate's FPGA model, two people can control complementary parts of the task. Neither extension is required for the first working demo.

## Why this needs the FPGA

The FPGA performs deterministic, fixed-point processing on a continuous signal. We will compare it with the same algorithm on the CPU using identical input. We will report FPGA core latency, USB round-trip latency, and full sensor-to-action latency separately. A faster end-to-end result is a hypothesis to measure; acquisition and I/O may dominate.

```text
raw EMG -> Mac acquisition -> FrontPanel USB 3 -> XEM7310 features + decoder
                                                -> Mac visualizer -> virtual gripper
webcam -> hand landmarks -> paired training labels -> quantized per-user weights
```

## Build order

1. **Bring up the board design.** The XEM7310-A75 enumerates through the owner's existing macOS FrontPanel binding over USB-C. Load a bitstream with FrontPanel endpoints, then verify PipeIn and WireOut loopback.
2. **Acquire paired data.** The Axo Kinesis band and raw EMG feed now work. Capture raw batches, then timestamp webcam hand landmarks and measure synchronization before pairing training windows.
3. **Make a software baseline.** Train a small ridge decoder for one or two continuous controls. Validate on held-out movements from the same wearer. Use this to drive the virtual gripper.
4. **Implement FPGA inference.** Compute windowed mean absolute deviation and waveform length per channel, then apply quantized linear weights. Compare the RTL output with a bit-exact software reference on recorded samples.
5. **Measure and demo.** Show raw signals, calibration, an unseen grasp task, prediction error, and p50/p95 latency. Add physical robot control only with a bounded workspace, watchdog, and emergency stop.

## Known board setup

The supplied photo shows an XEM7310 module with a barrel power jack and USB-C connector. The owner reports that this board powers up and runs code directly from the laptop over USB-C. FrontPanel enumeration on November 6 identified it as **XEM7310-A75**; it opened successfully, but `IsFrontPanelEnabled()` was false before loading a new bitstream. An external supply is not a prerequisite for the owner's working setup. [Opal Kelly's standard power documentation](https://docs.opalkelly.com/xem7310/powering-the-xem7310/) describes a different default configuration, so do not infer this board's internal power configuration from the photo.

The first Neo bitstream was built for `xc7a75tfgg484-1` using Vivado 2022.2 on `cs1410vnc`. The existing macOS FrontPanel binding programmed the board and verified PipeIn packet readback. The teammate's board and Medra access remain unconfirmed. Axo Kinesis has paired the loaner band and streamed raw EMG on this Mac.

## Scope rule

The one-person virtual gripper is the project. The second FPGA and Medra arm increase the ambition only after the core path works. If no Axo band becomes available, recorded or synthetic signals can validate the FPGA pipeline, but they are not a substitute for the live neural-interface demo.

## References

- [XEM7310 specifications](https://docs.opalkelly.com/xem7310/specifications/)
- [FrontPanel host interface](https://docs.opalkelly.com/xem7310/usb-3-0-host-interface/)
- [XEM7310 Vivado board files](https://docs.opalkelly.com/xem7310/vivado-board-file/)
- [FrontPanel SDK release notes](https://docs.opalkelly.com/fpsdk/release-notes/)
- [MediaPipe Hand Landmarker live stream API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker)

The repo includes a raw-EMG recorder, Kinesis batch normalizer, USB packetizer, CPU decoder baseline, and a Vivado-built FPGA packet receiver. The physical XEM7310 accepted 252 batches from a two-second live capture with correct word count, sequence, timestamp, and checksum. Webcam labels, FPGA feature extraction and inference, robot control, and latency benchmarks remain to be implemented.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the concrete band-to-FPGA-to-arm build sequence and proposed FrontPanel endpoint map.

**Start here:** [KINESIS.md](KINESIS.md) has pairing and raw-stream commands; [BAND_FIRST.md](BAND_FIRST.md) has calibration and CPU baseline commands; [FPGA_FIRST.md](FPGA_FIRST.md) defines the tested packet format; [fpga/README.md](fpga/README.md) has the bitstream build and board check.
