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

1. **Reproduce the working board setup.** This XEM7310 already runs directly from the owner's laptop over USB-C. Identify XEM7310-A75 versus A200, confirm FrontPanel enumeration, and run the existing bitstream or a stock example.
2. **Acquire paired data.** Confirm Axo band access and the raw EMG API. Timestamp raw samples and webcam hand landmarks. Measure synchronization before choosing a training window.
3. **Make a software baseline.** Train a small ridge decoder for one or two continuous controls. Validate on held-out movements from the same wearer. Use this to drive the virtual gripper.
4. **Implement FPGA inference.** Compute windowed mean absolute value and waveform length per channel, then apply quantized linear weights. Compare the RTL output with a bit-exact software reference on recorded samples.
5. **Measure and demo.** Show raw signals, calibration, an unseen grasp task, prediction error, and p50/p95 latency. Add physical robot control only with a bounded workspace, watchdog, and emergency stop.

## Known board setup

The supplied photo shows an XEM7310 module with a barrel power jack and USB-C connector. The owner reports that this particular board already powers up and runs code when plugged directly into their laptop over USB-C. Treat that working setup as verified for this project; an external supply is not a prerequisite. The photo does **not** reveal whether the FPGA is the A75 or A200 variant. [Opal Kelly's standard power documentation](https://docs.opalkelly.com/xem7310/powering-the-xem7310/) describes a different default configuration, so we should not infer this board's internal power configuration from the photo.

Use the owner's existing programming workflow. If a new Vivado bitstream must be built, [AMD's supported Vivado hosts](https://docs.amd.com/r/2025.2-English/ug973-vivado-release-notes-install-license/Supported-Operating-Systems) are x86-64 Windows and Linux; FrontPanel host communication supports macOS. The exact FPGA variant, teammate's board, Axo access, and Medra access remain unconfirmed.

## Scope rule

The one-person virtual gripper is the project. The second FPGA and Medra arm increase the ambition only after the core path works. If no Axo band becomes available, recorded or synthetic signals can validate the FPGA pipeline, but they are not a substitute for the live neural-interface demo.

## References

- [XEM7310 specifications](https://docs.opalkelly.com/xem7310/specifications/)
- [FrontPanel host interface](https://docs.opalkelly.com/xem7310/usb-3-0-host-interface/)
- [XEM7310 Vivado board files](https://docs.opalkelly.com/xem7310/vivado-board-file/)
- [FrontPanel SDK release notes](https://docs.opalkelly.com/fpsdk/release-notes/)
- [MediaPipe Hand Landmarker live stream API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker)

The repo now includes a runnable raw-EMG recording and CPU decoder baseline. It has been smoke-tested with synthetic data; live band integration awaits the Mac API sample. Sensor-specific integration, webcam labels, RTL, robot control, and benchmarks have not been implemented.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the concrete band-to-FPGA-to-arm build sequence and proposed FrontPanel endpoint map.

**Have the band but no robot arm?** Start with [BAND_FIRST.md](BAND_FIRST.md). It includes the message to send Axo for the raw EMG API, capture format, synthetic smoke test, rough calibration, and live virtual-gripper command.
