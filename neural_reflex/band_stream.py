"""Live band data from Kinesis: EMG batches, gyro and orientation on Kinesis's host clock
(the same clock as time.monotonic), kept for the last few seconds for windowed reads."""

from __future__ import annotations

import json
import threading
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np

from .kinesis import _consume_sse_line, assert_local_stream_url

WINDOW_S = 0.25


class BandStream:
    """EMG batches, gyro and orientation from Kinesis, on Kinesis's host clock (= time.monotonic)."""

    def __init__(self, url: str, keep_s: float | None = 5.0, record_to: Path | None = None):
        self.url = assert_local_stream_url(url)
        maxlen = None if keep_s is None else int(keep_s * 140)
        self.emg, self.gyro, self.orient = deque(maxlen=maxlen), deque(maxlen=maxlen), deque(maxlen=maxlen)
        self.lock, self.stop, self.error = threading.Lock(), threading.Event(), None
        self.record = record_to.open("w", encoding="utf-8") if record_to else None
        threading.Thread(target=self._run, daemon=True).start()

    @classmethod
    def offline(cls, keep_s: float | None = None) -> "BandStream":
        """A stream fed by add() instead of Kinesis (tests, replays)."""
        stream = cls.__new__(cls)
        maxlen = None if keep_s is None else int(keep_s * 140)
        stream.url, stream.error, stream.record = None, None, None
        stream.emg, stream.gyro, stream.orient = deque(maxlen=maxlen), deque(maxlen=maxlen), deque(maxlen=maxlen)
        stream.lock, stream.stop = threading.Lock(), threading.Event()
        return stream

    def _run(self):
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            request = urllib.request.Request(self.url, headers={"Accept": "text/event-stream"})
            with opener.open(request, timeout=5) as response:
                event, data = None, []
                for raw in response:
                    if self.stop.is_set():
                        return
                    event, data, payload = _consume_sse_line(raw.decode("utf-8"), event, data)
                    if payload:
                        self.add(payload)
        except Exception as error:   # noqa: BLE001 - shown to the user
            self.error = str(error)

    def add(self, payload: dict):
        kind, t = payload.get("type"), payload.get("host_time_s")
        if kind not in ("emg", "gyro", "orientation") or t is None:
            return
        p = payload["payload"]
        with self.lock:
            if kind == "emg":
                self.emg.append((t, np.asarray(p["values"], dtype=float)))
            elif kind == "gyro":
                self.gyro.append((t, np.asarray(p["values"], dtype=float) * p.get("observed_degrees_per_second_per_count", 0.07)))
            else:
                self.orient.append((t, p["values"]))
            if self.record:
                self.record.write(json.dumps({"type": kind, "host_time_s": t, "values": p["values"]}) + "\n")

    def window(self, t: float, seconds: float = WINDOW_S):
        with self.lock:
            emg = [v for bt, v in self.emg if t - seconds < bt <= t]
            gyro = [v for gt, v in self.gyro if t - seconds < gt <= t]
            orient = [(ot, q) for ot, q in self.orient if t - 0.2 <= ot <= t]
        return (np.concatenate(emg) if emg else None, np.array(gyro) if gyro else None,
                orient[-1][1] if orient else None)

    def orientations(self, t0: float, t1: float):
        with self.lock:
            return [q for ot, q in self.orient if t0 <= ot <= t1]

    def latest_time(self):
        with self.lock:
            return self.emg[-1][0] if self.emg else None

    def close(self):
        self.stop.set()
        if self.record:
            self.record.close()
