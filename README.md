# Neural Reflex FPGA

**Neo Hackathon 2026 concept:** turn one person's raw muscle signals into continuous, personalized control through a real FPGA. The first live demo is a virtual gripper. The available physical hardware is one black Hiwonder SO-ARM101 leader arm; its bounded motion path is described in [ARM_SETUP.md](ARM_SETUP.md).

## The demo

An Axo band streams raw eight-channel EMG to a Mac. During a short calibration, a webcam tracks the wearer's hand and supplies paired labels for pinch and hand opening. We train a subject-specific decoder, quantize its weights, and run streaming feature extraction and inference on an Opal Kelly XEM7310. The decoded signal controls a gripper in a simple virtual scene.

The leader arm has a handle, not the follower's grasping claw. We can test whether its servos can move the handle tip under software control, then map decoded EMG to bounded motion. This is an experimental use of the leader arm, separate from Hiwonder's documented leader-to-follower workflow.

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
5. **Measure and demo.** Show raw signals, calibration, an unseen virtual grasp task, prediction error, and p50/p95 latency. Test physical leader-arm motion in small, bounded steps before connecting EMG to it.

## Known board setup

The supplied photo shows an XEM7310 module with a barrel power jack and USB-C connector. The owner reports that this board powers up and runs code directly from the laptop over USB-C. FrontPanel enumeration on November 6 identified it as **XEM7310-A75**; it opened successfully, but `IsFrontPanelEnabled()` was false before loading a new bitstream. An external supply is not a prerequisite for the owner's working setup. [Opal Kelly's standard power documentation](https://docs.opalkelly.com/xem7310/powering-the-xem7310/) describes a different default configuration, so do not infer this board's internal power configuration from the photo.

The first Neo bitstream was built for `xc7a75tfgg484-1` using Vivado 2022.2 on `cs1410vnc`. The existing macOS FrontPanel binding programmed the board and verified PipeIn packet readback. Axo Kinesis has paired the loaner band and streamed raw EMG on this Mac. The single black Hiwonder leader arm has not yet appeared as a USB serial device on this Mac.

## Scope rule

The one-person virtual gripper remains the verified software target. Moving the single leader arm's handle tip is a physical extension after a live USB probe and small motion test. Recorded or synthetic EMG can validate the FPGA pipeline, but they are not a substitute for the live neural-interface demo.

## References

- [XEM7310 specifications](https://docs.opalkelly.com/xem7310/specifications/)
- [FrontPanel host interface](https://docs.opalkelly.com/xem7310/usb-3-0-host-interface/)
- [XEM7310 Vivado board files](https://docs.opalkelly.com/xem7310/vivado-board-file/)
- [FrontPanel SDK release notes](https://docs.opalkelly.com/fpsdk/release-notes/)
- [MediaPipe Hand Landmarker live stream API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker)

The repo includes a raw-EMG recorder, Kinesis batch normalizer, USB packetizer, CPU decoder baseline, a Vivado-built FPGA packet receiver, and an experimental leader-arm serial probe and bounded jog. The physical XEM7310 accepted 252 batches from a two-second live capture with correct word count, sequence, timestamp, and checksum. Webcam labels, FPGA feature extraction and inference, EMG-to-arm control, and full latency benchmarks remain to be implemented.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the concrete band-to-FPGA-to-arm build sequence and proposed FrontPanel endpoint map.

**Start here:** [ARM_SETUP.md](ARM_SETUP.md) has the single-arm connection and motion check; [KINESIS.md](KINESIS.md) has pairing and raw-stream commands; [BAND_FIRST.md](BAND_FIRST.md) has calibration and CPU baseline commands; [FPGA_FIRST.md](FPGA_FIRST.md) defines the tested packet format; [fpga/README.md](fpga/README.md) has the bitstream build and board check.
