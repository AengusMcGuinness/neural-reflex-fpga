# XEM7310-A75 packet receiver

`rtl/neo_packet_top.sv` is the first FPGA design for Neo. It accepts the
272-byte packets from `neural_reflex/fpga_packet.py` on FrontPanel PipeIn
`0x80`. It records the last packet header, timestamp, and XOR of all 68 words,
counts valid packets, and flags bad magic, sequence gaps, or FIFO overflow.
It does not yet compute EMG features or inference.

## Build

The build runs on `cs1410vnc` with Vivado 2022.2. It uses the installed
CS1410 XEM7310 FrontPanel library on that host; the vendor files are not
copied into this repo.

```bash
scp -r fpga/rtl fpga/constraints fpga/build.tcl cs1410vnc:Neo/fpga/
ssh cs1410vnc 'cd ~/Neo/fpga && vivado -mode batch -source build.tcl'
scp cs1410vnc:Neo/fpga/artifacts/neo_packet_top.bit fpga/artifacts/
scp cs1410vnc:Neo/fpga/artifacts/'*.rpt' fpga/artifacts/
```

The target part is `xc7a75tfgg484-1`. The XDC constrains only the FrontPanel
host pins and 100.8 MHz clock.

## Readback map

| WireOut | Value |
| --- | --- |
| `0x20` | Total received 32-bit words |
| `0x21` | Good 68-word packets |
| `0x22` | Last magic word |
| `0x23` | Last sequence |
| `0x24` | XOR of all words in the last packet |
| `0x25`, `0x26` | Last 64-bit band timestamp, low then high |
| `0x27` | Error bits 18:16, FIFO count 15:8, packet word index 6:0 |

WireIn `0x00` bit 0 synchronously resets the receiver. Error flags are sticky
until reset. A packet is counted as good when magic and sequence match. The
first sequence after reset can be any `uint32`; later sequences must advance
by one.

## Board check

The Mac uses the owner's existing FrontPanel Python binding. Run from the
repo root:

```bash
DYLD_LIBRARY_PATH="$HOME/cs1410_lab/lib/macOS" \
  python3 -m fpga.check_board
DYLD_LIBRARY_PATH="$HOME/cs1410_lab/lib/macOS" \
  python3 -m fpga.check_board --packet-file data/private/fpga_probe.bin
```

The checker programs the FPGA and compares the WireOut values with the
little-endian packet bytes. Private EMG captures and packet files remain
ignored by Git.

## Forward the live Kinesis stream

In Kinesis, enable **local sensor feed** and copy the stream URL. From the
repo root on the Mac with the board attached, run:

```bash
export DYLD_LIBRARY_PATH="$HOME/cs1410_lab/lib/macOS"
set -o pipefail
python3 .local/read-sensor-stream.py --url "$(pbpaste)" --seconds 10 --jsonl \
  | python3 -m fpga.live_board
```

`fpga.live_board` programs the bitstream, resets the receiver, packs each
16-sample EMG batch, and sends it to PipeIn as it arrives. Every 32 batches
it checks the board's word count, packet count, sequence, timestamp, checksum,
and error flags. It reports counts only; it does not save raw EMG. The
`--seconds` value controls the run length. This demonstrates live transport
through the FPGA, not decoded gesture output.

## Check where latency occurs

```bash
python3 -m fpga.latency_probe --input data/raw/fpga_probe.jsonl
```

The probe reports Kinesis batch timing, arrival gaps at the Mac, and the time
for the current Python feature calculation. The saved two-second and
three-second live captures showed 95th-percentile arrival gaps of about
70 ms and 61 ms respectively, while the Python feature call took about
0.05 ms. These are delivery gaps, **not** absolute sensor-to-host latency:
the band and Mac clocks are not synchronized. They do show that feature
arithmetic is not the first bottleneck to optimize in this setup.
