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

1. **Power and enumerate the board.** Identify XEM7310-A75 versus A200, obtain the correct 5 V supply and USB 3 cable, and run a stock FrontPanel example.
2. **Acquire paired data.** Confirm Axo band access and the raw EMG API. Timestamp raw samples and webcam hand landmarks. Measure synchronization before choosing a training window.
3. **Make a software baseline.** Train a small ridge decoder for one or two continuous controls. Validate on held-out movements from the same wearer. Use this to drive the virtual gripper.
4. **Implement FPGA inference.** Compute windowed mean absolute value and waveform length per channel, then apply quantized linear weights. Compare the RTL output with a bit-exact software reference on recorded samples.
5. **Measure and demo.** Show raw signals, calibration, an unseen grasp task, prediction error, and p50/p95 latency. Add physical robot control only with a bounded workspace, watchdog, and emergency stop.

## Immediate hardware gate

The supplied photo shows an XEM7310 module with a barrel power jack and USB-C connector. It does **not** reveal whether this is the A75 or A200 variant. [Opal Kelly specifies](https://docs.opalkelly.com/xem7310/powering-the-xem7310/) a regulated 4.5–5.5 V supply through the DC jack or expansion connector. USB power requires a board modification and is not the default. We should use the specified external supply.

FrontPanel supports a macOS host, but [AMD's supported Vivado hosts](https://docs.amd.com/r/2025.2-English/ug973-vivado-release-notes-install-license/Supported-Operating-Systems) are x86-64 Windows and Linux. We need one such machine for bitstream builds. The exact board variant, teammate's board, power setup, Axo access, and Medra access remain unconfirmed.

## Scope rule

The one-person virtual gripper is the project. The second FPGA and Medra arm increase the ambition only after the core path works. If no Axo band becomes available, recorded or synthetic signals can validate the FPGA pipeline, but they are not a substitute for the live neural-interface demo.

## References

- [XEM7310 specifications](https://docs.opalkelly.com/xem7310/specifications/)
- [FrontPanel host interface](https://docs.opalkelly.com/xem7310/usb-3-0-host-interface/)
- [XEM7310 Vivado board files](https://docs.opalkelly.com/xem7310/vivado-board-file/)
- [FrontPanel SDK release notes](https://docs.opalkelly.com/fpsdk/release-notes/)
- [MediaPipe Hand Landmarker live stream API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker)

This repository currently contains the project plan. Sensor integration, RTL, models, robot control, and benchmarks have not been implemented.
