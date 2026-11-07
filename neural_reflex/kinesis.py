"""Read the local Kinesis sensor feed and keep raw EMG batches."""

import json
import urllib.parse


def assert_local_stream_url(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or parsed.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("sensor stream URL must point at this Mac")
    return url


def _consume_sse_line(line, event_type, data_lines):
    stripped = line.rstrip("\r\n")
    if stripped == "":
        if not data_lines:
            return event_type, data_lines, None
        payload = json.loads("\n".join(data_lines))
        if event_type and "type" not in payload:
            payload["type"] = event_type
        return None, [], payload
    if stripped.startswith(":"):
        return event_type, data_lines, None
    if stripped.startswith("event:"):
        return stripped[6:].strip(), data_lines, None
    if stripped.startswith("data:"):
        data_lines.append(stripped[5:].lstrip())
    return event_type, data_lines, None


def parse_sse(text):
    events = []
    event_type = None
    data_lines = []
    for line in text.splitlines():
        event_type, data_lines, payload = _consume_sse_line(line, event_type, data_lines)
        if payload is not None:
            events.append(payload)
    if data_lines:
        event_type, data_lines, payload = _consume_sse_line("", event_type, data_lines)
        if payload is not None:
            events.append(payload)
    return events


MOTION_EVENTS = ("orientation", "gyro", "gesture")


def iter_emg_records(lines, on_motion=None):
    """Yield EMG batch records; pass orientation/gyro/gesture events to on_motion if given."""
    event_type = None
    data_lines = []
    connection = None
    for line in lines:
        event_type, data_lines, payload = _consume_sse_line(line, event_type, data_lines)
        if payload is None:
            continue
        kind = payload.get("type")
        if kind == "hello":
            status = payload.get("payload", {}).get("status", {})
            connection = status.get("band_connection_id", connection)
        elif kind == "status":
            connection = payload.get("payload", {}).get("band_connection_id", connection)
        elif kind in MOTION_EVENTS:
            if on_motion is not None:
                on_motion({**payload, "band_connection_id": connection})
        elif kind == "emg":
            yield from emg_records_from_events([{"type": "status",
                                                  "payload": {"band_connection_id": connection}},
                                                 payload])


def emg_records_from_events(events):
    connection = None
    records = []
    for event in events:
        if event.get("type") == "status":
            connection = event.get("payload", {}).get("band_connection_id", connection)
            continue
        if event.get("type") != "emg":
            continue
        payload = event["payload"]
        values = payload["values"]
        if payload.get("channels") != 8 or payload.get("samples_per_batch") != 16:
            raise ValueError("emg batch must be 16 samples by 8 channels")
        if payload.get("layout") not in (None, "sample_major"):
            raise ValueError("emg layout must be sample_major")
        if len(values) != 16 or any(len(row) != 8 for row in values):
            raise ValueError("emg values must be 16 by 8")
        records.append({
            "host_time_s": event["host_time_s"],
            "band_timestamp_us": event["band_timestamp_us"],
            "band_connection_id": connection,
            "event_sequence": event["event_sequence"],
            "stream_session_id": event["stream_session_id"],
            "sequence": payload["sequence"],
            "sample_interval_us": payload["sample_interval_us"],
            "sample_rate_hz": payload["sample_rate_hz"],
            "values": values,
        })
    return records
