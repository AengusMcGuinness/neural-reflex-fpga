"""Send Kinesis EMG event JSONL to the programmed XEM7310 as it arrives."""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

from neural_reflex.fpga_packet import MAGIC, PACKET_BYTES, pack_batch

from .check_board import program_board, read_registers


def verify(device, count, packet, timeout):
    expected_words = count * (PACKET_BYTES // 4)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        actual = read_registers(device)
        if actual[0] == expected_words:
            break
        time.sleep(0.01)
    else:
        raise TimeoutError(f"FPGA consumed {actual[0]} of {expected_words} words")

    _, sequence, timestamp = struct.unpack_from("<IIQ", packet)
    checksum = 0
    for (word,) in struct.iter_unpack("<I", packet):
        checksum ^= word
    expected = [expected_words, count, MAGIC, sequence, checksum,
                timestamp & 0xFFFFFFFF, timestamp >> 32, 0]
    if actual != expected:
        raise RuntimeError(f"FPGA readback mismatch: actual={actual}, expected={expected}")


def run(bitstream, verify_every, timeout):
    device = program_board(bitstream)
    print("FPGA programmed; forwarding Kinesis EMG batches", file=sys.stderr)
    count = 0
    last_sequence = None
    last_packet = None
    connection_id = None
    session_id = None
    try:
        for line_number, line in enumerate(sys.stdin, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("schema_version") != 1 or not isinstance(event.get("payload"), dict):
                raise ValueError(f"line {line_number}: unsupported Kinesis event")
            kind = event.get("type")
            if kind in ("hello", "status"):
                status = event["payload"].get("status", event["payload"])
                current_connection = status.get("band_connection_id")
                if count and current_connection != connection_id:
                    raise ValueError("band connection changed during forwarding")
                connection_id = current_connection
                continue
            if kind != "emg":
                continue
            current_session = event.get("stream_session_id")
            if session_id is None:
                session_id = current_session
            elif current_session != session_id:
                raise ValueError("Kinesis stream session changed")
            payload = event["payload"]
            sequence = payload["sequence"]
            if last_sequence is not None and (sequence - last_sequence) % (2**32) != 1:
                raise ValueError(f"EMG sequence gap after {last_sequence}")
            packet = pack_batch(sequence, event["band_timestamp_us"], payload["values"])
            written = device.WriteToPipeIn(0x80, bytearray(packet))
            if written != PACKET_BYTES:
                raise RuntimeError(f"PipeIn wrote {written} of {PACKET_BYTES} bytes")
            count += 1
            last_sequence = sequence
            last_packet = packet
            if count % verify_every == 0:
                verify(device, count, packet, timeout)
                print(f"Verified {count} live batches ({count * 16} samples/channel)",
                      file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        pass
    if not count:
        raise RuntimeError("no EMG batches received")
    verify(device, count, last_packet, timeout)
    print(f"PASS: {count} live batches ({count * 16} samples/channel) through FPGA")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitstream", type=Path,
                        default=Path("fpga/artifacts/neo_packet_top.bit"))
    parser.add_argument("--verify-every", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=5)
    args = parser.parse_args()
    if args.verify_every < 1 or args.timeout <= 0:
        parser.error("verification interval and timeout must be positive")
    run(args.bitstream.resolve(), args.verify_every, args.timeout)


if __name__ == "__main__":
    main()
