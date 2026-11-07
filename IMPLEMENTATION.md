# Implementation plan

This plan assumes access to one Axo/Meta neural band, the Medra arm, and the existing working USB-C XEM7310 setup. The Axo and Medra APIs are event-specific and have not been supplied, so their exact method names and command rates must be learned from their teams. Everything below the two adapters can be built without those details.

## Demo contract

The first physical demo is intentionally narrow: the arm follows a pre-approved path between two fixed poses, while the wearer's decoded forearm EMG controls the gripper. The participant calibrates, closes the gripper on an object, carries it through the fixed path, and releases it. A virtual gripper is the software test target. Continuous arm jogging and a second user are extensions after this loop is reliable.

The FPGA must be in the live signal path. The Mac receives raw EMG, timestamps and forwards samples over FrontPanel, and consumes the FPGA's decoded output. It must not silently substitute the CPU prediction during the judged run. Keep a visible switch between CPU and FPGA modes for an honest A/B comparison.

## Modules and interfaces

```text
band adapter -> timestamped 8 x int16 samples -> acquisition queue
camera -> timestamped hand landmarks -> calibration labels
acquisition queue -> FrontPanel PipeIn -> FPGA window/features/decoder
FPGA result/status -> FrontPanel WireOut -> host controller
host controller -> virtual gripper / Medra adapter
```

The band adapter must report the actual sample rate, raw units, channel order, and whether timestamps are generated on the band or on the Mac. The Medra adapter must expose only supported operations: get state, move to an approved pose, set gripper target (or open/close), and stop. Query its supported command rate and whether a hardware emergency stop is available. Do not assume servo-level access.

## Calibration and model

Use MediaPipe Hand Landmarker on the webcam to obtain 21 hand landmarks with timestamps. A first label is thumb-tip to index-tip distance divided by palm width, clipped to [0, 1]. Also record a rest state. Align EMG and camera streams using monotonic host timestamps; inspect the offset before pairing windows. Google notes that live-stream detection can drop camera frames, so interpolate labels only across short, valid gaps and discard longer gaps.

Collect several rounds of rest, open hand, and slow pinches from the same wearer. Split evaluation by *time block*, not randomly by adjacent windows, so overlapping windows do not leak into both training and validation. Start with a ridge regression or a three-state classifier. Only keep continuous aperture control if it behaves well on a held-out round. Save the per-user normalization values with the model.

For each channel, start with mean absolute value and waveform length over a measured window (for example, 100 ms with a 20 ms hop). These numbers are initial choices, not band specifications. Compare a few windows using offline replay. Quantize feature scaling, weights, and bias to a documented signed fixed-point format. Run the same integer math in a Python reference before putting it in RTL. Clamp output to the robot's supported gripper range.

## XEM7310 design

Start from the XEM7310 FrontPanel sample project and board file for the identified A75/A200 variant. Keep the first design in the FrontPanel host clock domain; Opal Kelly specifies a 100.8 MHz clock for Pipe endpoints. Add a FIFO at the PipeIn boundary because PipeIn can deliver back-to-back 32-bit words much faster than the EMG sample rate. Parse fixed-size sample records, reject malformed sequence numbers, accumulate feature windows, and run the fixed-point linear decoder. Expose an output sequence number, decoded value, status flags, and an FPGA cycle count.

Proposed endpoint map, to be validated against the installed SDK sample:

| Endpoint | Direction | Purpose |
| --- | --- | --- |
| PipeIn `0x80` | Host to FPGA | Batched raw EMG records |
| WireIn `0x00` | Host to FPGA | Reset, arm/disarm, configuration |
| WireIn `0x01` onward | Host to FPGA | Quantized weights and bias |
| WireOut `0x20` onward | FPGA to host | Latest result, sequence, status, cycle count |

The first packet experiment should send known 32-bit patterns through `0x80` and read them back or expose them on WireOut. That verifies byte order and word alignment before any EMG logic is added. FrontPanel USB 3 Pipes are 32 bits wide. Use the installed SDK's Python `write_to_pipe_in` / `read_from_pipe_out` names if it has the idiomatic API; older bindings use `WriteToPipeIn` / `ReadFromPipeOut`. For the low-rate latest result, WireOut polling is simpler than a PipeOut FIFO. If result history is needed, add a Block-Throttled PipeOut after the first demo.

## Host control state machine

`DISARMED -> CALIBRATING -> READY -> ARMED -> FAULT`

The host may send arm commands only in `ARMED`. Transition to `FAULT` and request a stop when samples or FPGA results go stale, the sequence stops advancing, the decoded value is invalid, the network connection drops, or Medra reports a fault. Keep the gripper within its approved range and limit command rate and slew rate. The physical arm's own safety controls and operator emergency stop remain authoritative. An AI planner, if added, may suggest target poses but stays outside the real-time motor loop.

## Integration order

1. Reproduce the owner's existing XEM7310 USB-C program/run loop. Identify the A75/A200 variant and archive the working bitstream project.
2. Connect the band adapter and save a short, consented raw recording locally. Print sample rate, units, missing-packet count, and timestamps. Do not commit raw biometric recordings to GitHub.
3. Show webcam landmarks and paired labels. Train a CPU decoder and control the virtual gripper.
4. Implement PipeIn loopback, then fixed-point features, then model weights, then result WireOut. Check each stage against the integer reference on the same recording.
5. Integrate the Medra adapter in simulation or dry-run mode. Agree on allowed poses, command rate, gripper limits, and stop behavior with the Medra team.
6. Run the one-person physical demo with an operator at the stop control. Then consider continuous arm movement and a second FPGA/band.

## Measurements to show judges

Log timestamps at band receipt, USB submission, FPGA result receipt, robot command submission, and robot acknowledgment or observed motion if available. Report prediction accuracy on a held-out round, FPGA-versus-reference error, FPGA cycle latency, USB round-trip p50/p95, and full input-to-action p50/p95. Do not label an FPGA cycle count as end-to-end latency.

## References

- [FrontPanel endpoint address ranges and widths](https://docs.opalkelly.com/fphdl/)
- [FrontPanel USB 3 HDL timing](https://docs.opalkelly.com/fphdl/frontpanel-hdl-usb-3-0/)
- [XEM7310 USB 3 host clock](https://docs.opalkelly.com/xem7310/usb-3-0-host-interface/)
- [FrontPanel Python API naming](https://docs.opalkelly.com/fpsdk/idiomatic-api-migration/python/)
- [MediaPipe Hand Landmarker live stream behavior](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker)
