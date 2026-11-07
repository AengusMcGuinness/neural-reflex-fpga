# Build and hardware check

- Target: XEM7310-A75 (`xc7a75tfgg484-1`)
- Builder: Vivado 2022.2 on `cs1410vnc`, using `fpga/build.tcl`
- Bitstream SHA-256: `4661ec76ceef298cdbfade988d3c42f24bbe69ed5b9412fbe09c3bff42dd1d3c`
- Routing and bitstream generation completed with zero Vivado errors.
- Timing report: no failing setup or hold endpoints in the analyzed clock groups; see `timing.rpt` for unconstrained paths in the vendor FrontPanel core.
- DRC report: zero errors, 43 warnings, mostly RAM control checks inside the installed FrontPanel core; see `drc.rpt`.
- Board check: one known 272-byte packet passed. A separate two-second Kinesis capture supplied 252 packets (4,032 samples/channel); all 17,136 words were received with matching final sequence, timestamp, checksum, and zero receiver error flags.

The private EMG capture is in ignored `data/raw` and `data/private` paths.
This bitstream is a packet receiver, not an EMG feature extractor or decoder.
