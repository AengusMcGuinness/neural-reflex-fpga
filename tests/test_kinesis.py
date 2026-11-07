import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

from neural_reflex.cli import normalize_kinesis
from neural_reflex.fpga_packet import PACKET_BYTES, pack_batch, unpack_batch


class KinesisTests(unittest.TestCase):
    def test_normalizes_raw_batch_and_keeps_clock_metadata(self):
        hello = {"schema_version": 1, "type": "hello",
                 "payload": {"status": {"band_connection_id": "epoch-a"}}}
        emg = {"schema_version": 1, "type": "emg", "payload": {
            "values": [[100] * 8, [101] * 8], "sequence": 42},
            "band_timestamp_us": 1000000, "host_time_s": 123.5,
            "stream_session_id": "session-a"}
        source = io.StringIO(json.dumps(hello) + "\n" + json.dumps(emg) + "\n")
        output = io.StringIO()
        original_stdin = sys.stdin
        try:
            sys.stdin = source
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                normalize_kinesis(SimpleNamespace())
        finally:
            sys.stdin = original_stdin
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row["emg"][0] for row in rows], [100, 101])
        self.assertEqual([row["sample_index"] for row in rows], [0, 1])
        self.assertEqual(rows[0]["band_connection_id"], "epoch-a")
        self.assertEqual(rows[0]["batch_sequence"], 42)

    def test_fpga_packet_round_trip_and_byte_order(self):
        values = [[sample * 8 + channel for channel in range(8)]
                  for sample in range(16)]
        packet = pack_batch(42, 123456789, values)
        self.assertEqual(len(packet), PACKET_BYTES)
        self.assertEqual(packet[:8].hex(), "5846524e2a000000")
        self.assertEqual(packet[16:20].hex(), "00000100")
        self.assertEqual(unpack_batch(packet), (42, 123456789, values))


if __name__ == "__main__":
    unittest.main()
