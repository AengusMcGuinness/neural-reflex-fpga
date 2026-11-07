"""Program XEM7310 and verify the FrontPanel packet receiver.

The Opal Kelly Python binding is supplied by the owner's existing CS1410 SDK.
No raw EMG values are printed or saved by this command.
"""

import argparse
import os
import struct
import sys
import time
from pathlib import Path

from neural_reflex.fpga_packet import PACKET_BYTES, pack_batch, unpack_batch


def frontpanel_module():
    sdk = Path(os.environ.get(
        "NEO_FRONTPANEL_PYTHON",
        str(Path.home() / "cs1410_lab/lib/macOS")))
    sys.path.insert(0, str(sdk))
    import ok  # noqa: E402 - vendor binding is found only after setting path
    return ok


def packets_from_file(path):
    raw = path.read_bytes()
    if not raw or len(raw) % PACKET_BYTES:
        raise ValueError(f"packet file must contain whole {PACKET_BYTES}-byte packets")
    packets = [raw[pos:pos + PACKET_BYTES]
               for pos in range(0, len(raw), PACKET_BYTES)]
    sequences = [unpack_batch(packet)[0] for packet in packets]
    if any(sequence != ((sequences[0] + index) & 0xFFFFFFFF)
           for index, sequence in enumerate(sequences)):
        raise ValueError("packet file has a sequence gap")
    return packets


def known_packet():
    values = [[(sample * 103 + channel * 1009) & 0xFFFF
               for channel in range(8)] for sample in range(16)]
    return pack_batch(0x01020304, 0x1122334455667788, values)


def read_registers(device):
    device.UpdateWireOuts()
    return [device.GetWireOutValue(address) for address in range(0x20, 0x28)]


def program_board(bitstream):
    ok = frontpanel_module()
    device = ok.okCFrontPanel()
    if device.GetDeviceCount() < 1:
        raise RuntimeError("no FrontPanel board found")
    if device.OpenBySerial("") != 0:
        raise RuntimeError("could not open the FrontPanel board")
    if device.ConfigureFPGA(str(bitstream)) != 0:
        raise RuntimeError("FrontPanel rejected the bitstream")
    if not device.IsFrontPanelEnabled():
        raise RuntimeError("FrontPanel endpoints are unavailable after configuration")

    device.SetWireInValue(0x00, 1)
    device.UpdateWireIns()
    device.SetWireInValue(0x00, 0)
    device.UpdateWireIns()
    return device


def run(bitstream, packets, timeout):
    device = program_board(bitstream)
    for packet in packets:
        count = device.WriteToPipeIn(0x80, bytearray(packet))
        if count != PACKET_BYTES:
            raise RuntimeError(f"PipeIn wrote {count} of {PACKET_BYTES} bytes")

    expected_words = len(packets) * 68
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        registers = read_registers(device)
        if registers[0] == expected_words:
            break
        time.sleep(0.01)
    else:
        raise TimeoutError(f"FPGA consumed {registers[0]} of {expected_words} words")

    magic, sequence, timestamp = struct.unpack_from("<IIQ", packets[-1])
    checksum = 0
    for (word,) in struct.iter_unpack("<I", packets[-1]):
        checksum ^= word
    expected = [expected_words, len(packets), magic, sequence, checksum,
                timestamp & 0xFFFFFFFF, timestamp >> 32, 0]
    if registers != expected:
        raise RuntimeError(f"WireOut mismatch: actual={registers}, expected={expected}")
    print(f"PASS: {len(packets)} packet(s), {expected_words} words, "
          f"last sequence={sequence}, XOR=0x{checksum:08x}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitstream", type=Path,
                        default=Path("fpga/artifacts/neo_packet_top.bit"))
    parser.add_argument("--packet-file", type=Path,
                        help="optional private file created by neural_reflex packetize")
    parser.add_argument("--timeout", type=float, default=5)
    args = parser.parse_args()
    packets = packets_from_file(args.packet_file) if args.packet_file else [known_packet()]
    run(args.bitstream.resolve(), packets, args.timeout)


if __name__ == "__main__":
    main()
