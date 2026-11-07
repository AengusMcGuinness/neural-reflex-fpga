# Implementation plan

The Axo/Meta band is paired through [Kinesis](KINESIS.md), and its raw eight-channel feed has been read, packetized, and passed through the XEM7310 packet receiver. The available physical arm is one black Hiwonder SO-ARM101 leader arm. [ARM_SETUP.md](ARM_SETUP.md) describes its experimental output path. FPGA feature extraction and inference remain to be built.

## Demo contract

The first physical demo is intentionally narrow: the leader arm moves its handle tip between two bounded poses under a decoded EMG command. It has no grasping claw, so the grasping task remains virtual. Before connecting EMG, verify a read-only USB probe and a small manual motor jog. Cartesian tip targets require a measured kinematic model and live validation.

The FPGA must be in the live signal path. The Mac receives raw EMG, timestamps and forwards samples over FrontPanel, and consumes the FPGA's decoded output. It must not silently substitute the CPU prediction during the judged run. Keep a visible switch between CPU and FPGA modes for an honest A/B comparison.

## Modules and interfaces

```text
Kinesis SSE -> timestamped 8 x uint16 ADC counts -> acquisition queue
camera -> timestamped hand landmarks -> calibration labels
acquisition queue -> FrontPanel PipeIn -> FPGA window/features/decoder
FPGA result/status -> FrontPanel WireOut -> host controller
host controller -> virtual gripper / experimental leader-arm adapter
```

Kinesis reports 2,048 samples/second/channel, eight unsigned ADC channels, and 16 samples per batch. Its band timestamp and Mac host receipt time use different clocks; preserve both plus batch sequence and connection ID. The leader-arm adapter reads six serial bus motor positions and can send bounded raw position goals. It must stop on stale input, serial errors, or a joint that fails to track a small target. Do not treat its handle as a gripper.

## Calibration and model

Use MediaPipe Hand Landmarker on the webcam to obtain 21 hand landmarks with timestamps. A first label is thumb-tip to index-tip distance divided by palm width, clipped to [0, 1]. Also record a rest state. Align EMG and camera streams using monotonic host timestamps; inspect the offset before pairing windows. Google notes that live-stream detection can drop camera frames, so interpolate labels only across short, valid gaps and discard longer gaps.

Collect several rounds of rest, open hand, and slow pinches from the same wearer. Split evaluation by *time block*, not randomly by adjacent windows, so overlapping windows do not leak into both training and validation. Start with a ridge regression or a three-state classifier. Only keep continuous aperture control if it behaves well on a held-out round. Save the per-user normalization values with the model.

For each channel, start with mean absolute deviation from the window mean and waveform length over a measured window (for example, 100 ms with a 20 ms hop). Removing the DC offset matters because Kinesis sends unsigned ADC counts. These window settings are initial choices, not band specifications. Compare a few windows using offline replay. Quantize feature scaling, weights, and bias to a documented signed fixed-point format. Run the same integer math in a Python reference before putting it in RTL. Clamp output to the robot's supported gripper range.

## XEM7310 design

Start from the XEM7310 FrontPanel sample project and board file for the identified A75/A200 variant. Keep the first design in the FrontPanel host clock domain; Opal Kelly specifies a 100.8 MHz clock for Pipe endpoints. Add a FIFO at the PipeIn boundary because PipeIn can deliver back-to-back 32-bit words much faster than the EMG sample rate. Parse fixed-size sample records, reject malformed sequence numbers, accumulate feature windows, and run the fixed-point linear decoder. Expose an output sequence number, decoded value, status flags, and an FPGA cycle count.

Proposed endpoint map, to be validated against the installed SDK sample:

| Endpoint | Direction | Purpose |
| --- | --- | --- |
| PipeIn `0x80` | Host to FPGA | Batched raw EMG records |
| WireIn `0x00` | Host to FPGA | Reset, arm/disarm, configuration |
| WireIn `0x01` onward | Host to FPGA | Quantized weights and bias |
| WireOut `0x20` onward | FPGA to host | Latest result, sequence, status, cycle count |

The host packet contract is now fixed in [FPGA_FIRST.md](FPGA_FIRST.md) and `neural_reflex/fpga_packet.py`: one 272-byte packet per 16 raw EMG samples, with explicit little-endian words. The first FPGA experiment should send known packets through `0x80` and expose word count, magic, sequence, and checksum on WireOut. That verifies byte order and alignment before EMG logic is added. FrontPanel USB 3 Pipes are 32 bits wide. Use the installed SDK's Python `write_to_pipe_in` / `read_from_pipe_out` names if it has the idiomatic API; older bindings use `WriteToPipeIn` / `ReadFromPipeOut`. For the low-rate latest result, WireOut polling is simpler than a PipeOut FIFO.

## Host control state machine

`DISARMED -> CALIBRATING -> READY -> ARMED -> FAULT`

The host may send arm commands only in `ARMED`. Transition to `FAULT` and disable motor torque when samples or FPGA results go stale, the sequence stops advancing, the decoded value is invalid, the serial connection drops, or a motor fails to track. Bound every joint within its configured motor limits and a smaller session-specific travel range. Limit command rate and step size. Keep the power plug accessible for a physical stop. An AI planner, if added, may suggest target poses but stays outside the real-time motor loop.

## Integration order

1. **Done:** the XEM7310-A75 was programmed over USB-C with the new [packet receiver](fpga/README.md). FrontPanel enabled, and the PipeIn/WireOut test passed with a known packet and 252 live EMG packets.
2. **Done:** connect the Kinesis band and save a short local raw recording. The three-second test captured 6,048 samples, with batch sequence, band timestamp, host receipt time, and connection ID preserved. Do not commit raw biometric recordings to GitHub.
3. Show webcam landmarks and paired labels. Train a CPU decoder and control the virtual gripper.
4. **PipeIn packet check done:** implement fixed-point features, then model weights and result WireOut. Check each stage against the integer reference on the same recording.
5. **Software probe and bounded jog implemented:** verify the Hiwonder leader arm over live USB, starting with read-only motor state, then a small single-joint move. No live arm test has happened yet.
6. Measure the handle-tip geometry and validated joint limits before adding Cartesian tip targets or EMG-triggered motion. Run the one-person physical demo with an operator at the power stop.

## Measurements to show judges

Log timestamps at band receipt, USB submission, FPGA result receipt, robot command submission, and robot acknowledgment or observed motion if available. Report prediction accuracy on a held-out round, FPGA-versus-reference error, FPGA cycle latency, USB round-trip p50/p95, and full input-to-action p50/p95. Do not label an FPGA cycle count as end-to-end latency.

## References

- [FrontPanel endpoint address ranges and widths](https://docs.opalkelly.com/fphdl/)
- [FrontPanel USB 3 HDL timing](https://docs.opalkelly.com/fphdl/frontpanel-hdl-usb-3-0/)
- [XEM7310 USB 3 host clock](https://docs.opalkelly.com/xem7310/usb-3-0-host-interface/)
- [FrontPanel Python API naming](https://docs.opalkelly.com/fpsdk/idiomatic-api-migration/python/)
- [MediaPipe Hand Landmarker live stream behavior](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker)
