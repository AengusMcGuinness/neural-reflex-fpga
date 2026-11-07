# First FPGA analysis of the live EMG feed

Kinesis supplies real raw eight-channel EMG. The **XEM7310-A75** now has a Vivado-built packet receiver, and the Mac sent both a known packet and 252 packets from a two-second live capture through FrontPanel PipeIn `0x80`. FPGA WireOut values matched the packet byte stream exactly. This establishes the band-to-FPGA input path; EMG features and inference are still to be implemented in RTL. See [fpga/README.md](fpga/README.md) for the design, bitstream, and board check.

## Data and packet contract

The [Axo stream API](https://github.com/Axo-Neurotech/Kinesis-Axo/blob/main/docs/sensor-stream.md) reports eight channels of unsigned 16-bit ADC counts at 2,048 samples/second/channel, in 16-sample batches. Preserve those raw counts and the batch sequence. This is about 32 KiB/s before headers, small relative to the XEM7310's USB 3 FrontPanel link. The data is raw ADC counts, not microvolts.

Run [KINESIS.md](KINESIS.md) to create a local `data/raw/first.jsonl`, then:

```bash
python3 -m neural_reflex packetize --input data/raw/first.jsonl \
  --output data/private/fpga_packets.bin
python3 -m unittest discover -s tests -v
```

`neural_reflex/fpga_packet.py` defines a **272-byte** packet, exactly 68 little-endian 32-bit words:

| Word | Contents |
| --- | --- |
| 0 | Magic `0x4E524658` |
| 1 | Raw EMG batch sequence, uint32 |
| 2–3 | Band timestamp in microseconds, uint64, low word first |
| 4–67 | 16 samples × 4 words. In each word: channel `2j` in bits 15:0 and channel `2j+1` in bits 31:16. |

The packetizer rejects incomplete or reordered batches and mixed band connection epochs. Host byte order is explicit; do not rely on a native C or Python struct's default endianness. The band timestamp is a batch timestamp; do not silently treat it as a synchronized Mac camera time.

## Board bring-up

1. **Done:** Build `fpga/rtl/neo_packet_top.sv` for `xc7a75tfgg484-1` with Vivado 2022.2 on `cs1410vnc`. The design uses the installed FrontPanel library, PipeIn `0x80`, and an elastic FIFO.
2. **Done:** Program the connected XEM7310-A75 and verify one known 272-byte packet. WireOut reports 68 words, matching magic, sequence, timestamp, and XOR checksum.
3. **Done:** Capture two seconds of live Kinesis EMG (4,032 samples per channel), packetize into 252 batches, and pass all 252 through the board with matching last packet and no error flags. The private recording and packet file remain ignored by Git.
4. Then compute, per channel, a baseline-corrected amplitude and waveform length over a 128- or 256-sample window (62.5 or 125 ms at 2,048 Hz). Advance by 32 or 64 samples. Keep integer accumulators wide enough for 16-bit input and the full window. Compare every feature against a bit-exact Python reference on the **same recorded packet file**.
5. Once features match, load quantized decoder weights through WireIn and return a decoded pinch value, result sequence, and cycle count through WireOut. Compare FPGA and CPU predictions on held-out movements. Report FPGA core cycles and USB end-to-end latency separately.

The older CS1410 helper's `writeBuffer` adds padding even when a buffer is already aligned, so `fpga/check_board.py` calls FrontPanel `WriteToPipeIn` directly. Its old ALU bitstream does **not** parse EMG packets. This bitstream verifies packet transport and framing; it does not perform FPGA inference. The next milestone is bit-exact feature extraction over recorded packets.
