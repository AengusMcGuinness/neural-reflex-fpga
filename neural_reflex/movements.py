"""Teach the band a set of movements, recognise them live, and let each one drive a robot motor.

  teach   prompts you through each movement in config/movement_motors.json (several shuffled
          repetitions, rest in between) while recording EMG and IMU from Kinesis; trains a classifier;
          reports how reliably each movement is recognised when its repetition was held out; saves
          the model and raw data in the session folder.
  live    recognises movements in real time. A movement recognised continuously for hold_s moves its
          motor in its direction; stopping the movement holds the motor where it is. Dry run by default
          (shows what would move); --execute drives the arm through ArmController's safety.

Numpy only (no scipy), so it runs in the arm's own Python environment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np

from .imu_teleop import mean_rotation, quat_to_matrix, rotation_vector
from .kinesis import _consume_sse_line, assert_local_stream_url

REPO = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO / "config" / "movement_motors.json"
ROBOT_MAP = REPO / "config" / "robot_map.json"
HOME_FILE = REPO / "config" / "arm_home.json"
WINDOW_S = 0.25          # each recognition looks at the last 250 ms
STEP_S = 0.05            # and runs every 50 ms
REACTION_S = 0.8         # ignore the start of each prompt while you react
TAIL_S = 0.2
PLANAR = ("shoulder_lift", "elbow_flex", "wrist_flex")
TICKS_PER_DEGREE = 4096 / 360
HOME_TOLERANCE = 150


# ---------- live band data ------------------------------------------------------------------------
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


def features(emg, gyro, rotation_deg) -> np.ndarray | None:
    """25 numbers for one 250 ms window: per EMG channel, activity (mean |change|) and spread; gyro
    mean and mean |rate|; and how far the band has rotated from the neutral pose, in its own frame."""
    if emg is None or gyro is None or rotation_deg is None or len(emg) < 32 or len(gyro) < 3:
        return None
    activity = np.log1p(np.mean(np.abs(np.diff(emg, axis=0)), axis=0))
    spread = np.log1p(np.std(emg, axis=0))
    return np.concatenate([activity, spread, gyro.mean(axis=0) / 50, np.abs(gyro).mean(axis=0) / 50,
                           np.asarray(rotation_deg) / 30])


def window_features(stream: BandStream, t: float, neutral: np.ndarray):
    emg, gyro, q = stream.window(t)
    rot = None if q is None else np.degrees(rotation_vector(neutral.T @ quat_to_matrix(q)))
    return features(emg, gyro, rot)


# ---------- classifier ----------------------------------------------------------------------------
class Classifier:
    """Shrinkage linear discriminant: each movement is a cloud of feature points with shared spread."""

    def __init__(self, classes, mean, scale, means, precision):
        self.classes, self.mean, self.scale = list(classes), np.asarray(mean), np.asarray(scale)
        self.means, self.precision = np.asarray(means), np.asarray(precision)

    @classmethod
    def fit(cls, x, y, classes, shrink=0.15):
        x = np.asarray(x, dtype=float)
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-9] = 1.0
        z = (x - mean) / scale
        y = np.asarray(y)
        means = np.stack([z[y == c].mean(axis=0) for c in classes])
        resid = np.concatenate([z[y == c] - means[i] for i, c in enumerate(classes)])
        cov = np.cov(resid.T)
        cov = (1 - shrink) * cov + shrink * np.eye(len(cov)) * np.trace(cov) / len(cov)
        return cls(classes, mean, scale, means, np.linalg.inv(cov))

    def proba(self, x):
        z = (np.atleast_2d(x) - self.mean) / self.scale
        w = self.means @ self.precision
        scores = z @ w.T - 0.5 * np.sum(w * self.means, axis=1)
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        return p / p.sum(axis=1, keepdims=True)

    def to_json(self):
        return {"classes": self.classes, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "means": self.means.tolist(), "precision": self.precision.tolist()}

    @classmethod
    def from_json(cls, d):
        return cls(d["classes"], d["mean"], d["scale"], d["means"], d["precision"])


def cross_validate(x, y, groups, classes):
    """Each repetition scored by a model trained on the other repetitions."""
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    predicted = np.empty(len(y), dtype=object)
    for g in sorted(set(groups)):
        test = groups == g
        model = Classifier.fit(x[~test], y[~test], classes)
        predicted[test] = np.array(classes, dtype=object)[np.argmax(model.proba(x[test]), axis=1)]
    confusion = {c: {d: int(np.sum((y == c) & (predicted == d))) for d in classes} for c in classes}
    recall = {c: confusion[c][c] / max(1, sum(confusion[c].values())) for c in classes}
    return float(np.mean(predicted == y)), recall, confusion


# ---------- teaching ------------------------------------------------------------------------------
def schedule(cfg, seed=0):
    """Shuffled repetitions: [(kind, name, instruction, seconds, rep)]."""
    rng = random.Random(seed)
    steps = [("rest", "rest", "Relax: forearm LEVEL in front of you, palm down, hand loose. Hold still.", 4.0, 0)]
    for rep in range(1, cfg["repetitions"] + 1):
        order = list(cfg["movements"])
        rng.shuffle(order)
        for m in order:
            # Preview the next movement with a countdown, so you start it on GO, not while reading.
            steps.append(("ready", m["name"], f"NEXT: {m['instruction']}", cfg.get("ready_s", 3.0), rep))
            steps.append(("move", m["name"], f"GO: {m['instruction']}", cfg["hold_each_s"], rep))
            steps.append(("rest", "rest", "Back to neutral and relax.", cfg["rest_between_s"], rep))
    return steps


def run_prompts(steps, clock=time.monotonic, sleep=time.sleep, out=sys.stdout):
    segments = []
    for kind, name, text, seconds, rep in steps:
        start = clock()
        marker = ">>> " if kind == "move" else "    "
        while True:
            left = start + seconds - clock()
            if left <= 0:
                break
            count = f"get ready... {math.ceil(left)}" if kind == "ready" else f"{left:3.0f} s"
            out.write(f"\r  {marker}{text}  {count}   ")
            out.flush()
            sleep(min(0.2, left))
        out.write("\n")
        segments.append({"kind": kind, "name": name, "rep": rep, "start": start, "end": clock()})
    return segments


class FrozenStream:
    """A finished recording as sorted arrays, so windows are found by binary search, not a scan."""

    def __init__(self, emg, gyro, orient):
        self.et, self.ev = np.array([t for t, _ in emg]), [v for _, v in emg]
        self.gt, self.gv = np.array([t for t, _ in gyro]), np.array([v for _, v in gyro]).reshape(-1, 3)
        self.ot, self.oq = np.array([t for t, _ in orient]), [q for _, q in orient]

    @classmethod
    def from_stream(cls, stream: "BandStream"):
        with stream.lock:
            return cls(list(stream.emg), list(stream.gyro), list(stream.orient))

    @classmethod
    def from_events(cls, path: Path):
        emg, gyro, orient = [], [], []
        import gzip
        if not path.exists() and Path(str(path) + ".gz").exists():
            path = Path(str(path) + ".gz")
        opener = gzip.open if str(path).endswith(".gz") else open
        for line in opener(path, "rt", encoding="utf-8"):
            e = json.loads(line)
            if e["type"] == "emg":
                emg.append((e["host_time_s"], np.asarray(e["values"], dtype=float)))
            elif e["type"] == "gyro":
                gyro.append((e["host_time_s"], np.asarray(e["values"], dtype=float) * 0.07))
            elif e["type"] == "orientation":
                orient.append((e["host_time_s"], e["values"]))
        for items in (emg, gyro, orient):
            items.sort(key=lambda item: item[0])
        return cls(emg, gyro, orient)

    def window(self, t: float, seconds: float = WINDOW_S):
        a, b = np.searchsorted(self.et, [t - seconds, t], side="right")
        c, d = np.searchsorted(self.gt, [t - seconds, t], side="right")
        o = np.searchsorted(self.ot, t, side="right") - 1
        emg = np.concatenate(self.ev[a:b]) if b > a else None
        gyro = self.gv[c:d] if d > c else None
        q = self.oq[o] if o >= 0 and t - self.ot[o] <= 0.2 else None
        return emg, gyro, q

    def orientations(self, t0: float, t1: float):
        a, b = np.searchsorted(self.ot, [t0, t1])
        return self.oq[a:b]


def dataset(stream, segments, neutral):
    x, y, groups = [], [], []
    for s in segments:
        if s["kind"] == "ready":          # previewing the next movement: neither rest nor the movement
            continue
        t = s["start"] + REACTION_S
        while t <= s["end"] - TAIL_S:
            f = window_features(stream, t, neutral)
            if f is not None:
                x.append(f)
                y.append(s["name"])
                groups.append(s["rep"] if s["rep"] else 1)
            t += STEP_S
    return np.array(x), np.array(y), np.array(groups)


def teach(args, cfg) -> int:
    out = Path(args.session) / "movements" / args.name
    if (out / "model.json").exists() or (out / "events.jsonl").exists():
        raise SystemExit(f"{out} already has a recording; choose another --name (or use retrain)")
    out.mkdir(parents=True, exist_ok=True)
    stream = BandStream(args.url, keep_s=None, record_to=out / "events.jsonl")
    time.sleep(1.0)
    if stream.error or stream.latest_time() is None:
        print(f"No band data ({stream.error or 'nothing arriving'}). Check Kinesis and the stream URL.")
        return 1
    steps = schedule(cfg)
    total = sum(s[3] for s in steps)
    print(f"\nTEACHING {len(cfg['movements'])} movements x {cfg['repetitions']} repetitions ({total / 60:.1f} min).")
    print("Do each movement as the prompt says and HOLD it until the next prompt. No keys to press.\n")
    segments = run_prompts(steps)
    (out / "segments.json").write_text(json.dumps(segments, indent=1) + "\n")   # saved before training
    time.sleep(0.3)
    stream.close()
    print("training (a few seconds)...")
    return train_and_save(FrozenStream.from_stream(stream), segments, cfg, out)


def train_and_save(stream, segments, cfg, out: Path) -> int:
    first = segments[0]
    neutral = mean_rotation(stream.orientations(first["start"] + REACTION_S, first["end"]))
    classes = ["rest"] + [m["name"] for m in cfg["movements"]]
    x, y, groups = dataset(stream, segments, neutral)
    if len(set(y)) < len(classes):
        print("Some movements produced no usable band data; check Kinesis is sending EMG and orientation.")
        return 1
    accuracy, recall, confusion = cross_validate(x, y, groups, classes)
    model = Classifier.fit(x, y, classes)
    (out / "model.json").write_text(json.dumps({"classifier": model.to_json(), "window_s": WINDOW_S,
                                                "config": cfg}) + "\n")
    (out / "report.json").write_text(json.dumps({"accuracy": accuracy, "recall": recall,
                                                 "confusion": confusion}, indent=1) + "\n")
    print(f"\nRecognised correctly on held-out repetitions: {100 * accuracy:.0f}% of windows")
    for c in classes:
        wrong = sorted(((n, d) for d, n in confusion[c].items() if d != c and n), reverse=True)[:2]
        note = ("  often mistaken for " + ", ".join(d for _, d in wrong)) if wrong and recall[c] < 0.8 else ""
        print(f"  {c:13} {100 * recall[c]:4.0f}%{note}")
    weak = [c for c in classes if recall[c] < 0.6]
    if weak:
        print(f"\nUnreliable: {', '.join(weak)}. Remove them from config/movement_motors.json (or make them more "
              "distinct) and teach again.")
    print(f"\nModel: {out / 'model.json'}")
    return 0


# ---------- live ----------------------------------------------------------------------------------
class Detector:
    """A movement becomes active once recognised continuously for hold_s; anything else -> rest."""

    def __init__(self, hold_s: float, min_confidence: float):
        self.hold_s, self.min_confidence = hold_s, min_confidence
        self.candidate, self.since, self.active = "rest", None, "rest"

    def update(self, probs: dict, now: float) -> str:
        best = max(probs, key=probs.get)
        if probs[best] < self.min_confidence:
            best = "rest"
        if best != self.candidate:
            self.candidate, self.since = best, now
        if best == "rest":
            self.active = "rest"
        elif now - self.since >= self.hold_s:
            self.active = best
        return self.active


class Jogger:
    """Turns the active movement into motor targets, inside limits, above the table and under load."""

    def __init__(self, cfg, robot_map, home):
        from .pick_sim import check_pose
        self.check_pose = check_pose
        self.cfg, self.home = cfg, home
        margin = robot_map["margin_ticks"]
        self.limits = {j: (lo + margin, hi - margin) for j, (lo, hi) in robot_map["limits"].items()}
        self.deg_limits = {j: ((self.limits[j][0] - home[j]) / TICKS_PER_DEGREE,
                               (self.limits[j][1] - home[j]) / TICKS_PER_DEGREE) for j in PLANAR}
        self.by_name = {m["name"]: m for m in cfg["movements"]}
        self.motors = sorted({m["motor"] for m in cfg["movements"]} | set(PLANAR))
        self.targets = {j: float(home[j]) for j in self.motors}
        self.blocked = None

    def step(self, active: str, dt: float) -> dict:
        self.blocked = None
        m = self.by_name.get(active)
        if m:
            j = m["motor"]
            lo, hi = self.limits[j]
            proposed = dict(self.targets)
            proposed[j] = min(max(self.targets[j] + m["direction"] * m["speed_ticks_s"] * dt, lo), hi)
            if proposed[j] in (lo, hi):
                self.blocked = f"{j} at its limit"
            planar = [(proposed[p] - self.home[p]) / TICKS_PER_DEGREE for p in PLANAR]
            problems = self.check_pose(planar, self.deg_limits)
            if problems:
                self.blocked = problems[0]
            else:
                self.targets = proposed
        return {j: int(round(v)) for j, v in self.targets.items()}

    def degrees(self):
        return {j: (v - self.home[j]) / TICKS_PER_DEGREE for j, v in self.targets.items()}


def capture_neutral(stream: BandStream, seconds=3.0):
    print("Relax: forearm LEVEL in front of you, palm down, hand loose. Hold still...")
    t0 = time.monotonic()
    time.sleep(seconds)
    samples = stream.orientations(t0 + 0.8, time.monotonic())
    if len(samples) < 10:
        raise RuntimeError("no band orientation arriving")
    return mean_rotation(samples)


def control_loop(stream, model, cfg, jogger, neutral, controller=None, duration=None, clock=time.monotonic,
                 sleep=time.sleep, show=True):
    detector = Detector(cfg["hold_s"], cfg["min_confidence"])
    last, t0, checked = clock(), clock(), 0.0
    while duration is None or clock() - t0 < duration:
        now = clock()
        f = window_features(stream, now, neutral)
        active = "rest"
        if f is not None:
            probs = dict(zip(model.classes, model.proba(f)[0]))
            active = detector.update(probs, now)
        targets = jogger.step(active, now - last)
        last = now
        if controller is not None:
            for j, t in targets.items():
                controller.target(j, t, controller.motion_epoch)
            if now - checked > 0.5:
                checked = now
                snap = controller.status()
                if snap.get("fault"):
                    raise RuntimeError(snap["fault"])
        if show:
            m = jogger.by_name.get(active)
            action = f"-> {m['motor']} {'+' if m['direction'] > 0 else '-'}" if m else "   (holding)"
            deg = jogger.degrees()
            sys.stdout.write(f"\r  {active:13} {action:22} " + " ".join(f"{j.split('_')[0][:5]} {deg[j]:+4.0f}°"
                             for j in jogger.motors) + (f"  [{jogger.blocked}]" if jogger.blocked else "") + "      ")
            sys.stdout.flush()
        sleep(STEP_S)


def live(args, cfg) -> int:
    saved = json.loads(Path(args.model).read_text())
    model = Classifier.from_json(saved["classifier"])
    cfg = dict(cfg, movements=[m for m in cfg["movements"] if m["name"] in model.classes])
    robot_map = json.loads(ROBOT_MAP.read_text())
    home = json.loads(HOME_FILE.read_text())["positions"]
    jogger = Jogger(cfg, robot_map, home)
    stream = BandStream(args.url)
    time.sleep(1.0)
    if stream.error or stream.latest_time() is None:
        print(f"No band data ({stream.error or 'nothing arriving'}). Check Kinesis and the stream URL.")
        return 1
    try:
        neutral = capture_neutral(stream)
    except RuntimeError as error:
        print(f"Stopped: {error}")
        return 1
    print("Movements -> motors: " + ", ".join(f"{m['name']}->{m['motor']}{'+' if m['direction'] > 0 else '-'}"
                                              for m in cfg["movements"]))
    if not args.execute:
        print("\nDRY RUN: do a movement and hold it; the line shows what is recognised and which motor would move.")
        try:
            control_loop(stream, model, cfg, jogger, neutral)
        except KeyboardInterrupt:
            print("\nDry run ended; nothing moved.")
        return 0
    if not args.port:
        raise SystemExit("--execute needs --port")
    print("\nThe arm moves while you hold a movement and holds when you stop. Start with the arm straight up.")
    if input("Type MOVE and press Enter to start: ").strip() != "MOVE":
        print("Not started.")
        return 1
    from .arm_web import ArmController
    from .leader_arm import make_bus
    controller = ArmController(make_bus(args.port), timeout=15.0)
    code = 0
    stop_loop = threading.Event()
    try:
        snap = controller.status()
        if snap.get("fault"):
            raise RuntimeError(snap["fault"])
        pos = {j["name"]: j["position"] for j in snap["joints"]}
        off = [j for j in jogger.motors if abs(pos[j] - home[j]) > HOME_TOLERANCE]
        if off:
            raise RuntimeError(f"put the arm in its straight-up home pose first ({', '.join(off)} off)")
        controller.set_speed("fast")
        def motion_loop():
            while not stop_loop.wait(0.02):
                controller.motion_step()

        threading.Thread(target=motion_loop, daemon=True).start()
        for j in jogger.motors:
            snap = controller.torque(j, True, controller.motion_epoch)
            if snap.get("fault"):
                raise RuntimeError(snap["fault"])
        print("motors on. Hold a movement to move its motor; Ctrl+C to stop.")
        control_loop(stream, model, cfg, jogger, neutral, controller)
    except KeyboardInterrupt:
        print("\nStopped: holding the arm where it is.")
    except RuntimeError as error:
        print(f"\nStopped: {error}")
        code = 1
    finally:
        try:
            controller.pause(controller.motion_epoch)
        except Exception:   # noqa: BLE001
            pass
        try:
            input("\nSupport the arm with your hand, then press Enter to turn the motors off... ")
        finally:
            stop_loop.set()
            controller.close()
            print("Motors off.")
    return code


def segments_from_schedule(cfg, start: float):
    """The prompt timeline as it ran, if it started at `start` (the order is fixed by the seed)."""
    segments, t = [], start
    for kind, name, _, seconds, rep in schedule(cfg):
        segments.append({"kind": kind, "name": name, "rep": rep, "start": t, "end": t + seconds})
        t += seconds
    return segments


def find_start(stream: FrozenStream, cfg, around: float, span: float = 3.0, step: float = 0.1):
    """Start time that makes the movements most recognisable (cross-validated accuracy)."""
    classes = ["rest"] + [m["name"] for m in cfg["movements"]]
    best = None
    for offset in np.arange(-span, span + 1e-9, step):
        segs = segments_from_schedule(cfg, around + offset)
        neutral = mean_rotation(stream.orientations(segs[0]["start"] + REACTION_S, segs[0]["end"]) or [[1, 0, 0, 0]])
        x, y, g = dataset(stream, segs, neutral)
        if len(set(y)) < len(classes):
            continue
        acc = cross_validate(x, y, g, classes)[0]
        if best is None or acc > best[0]:
            best = (acc, around + offset)
    return best


def retrain(args, cfg) -> int:
    out = Path(args.folder)
    stream = FrozenStream.from_events(out / "events.jsonl")
    seg_file = out / "segments.json"
    if seg_file.exists():
        segments = json.loads(seg_file.read_text())
        print("using the saved prompt timeline")
    else:
        # Prompts began about 1 s after the stream started (the tool waits for band data first).
        guess = float(stream.et[0]) + 1.0
        print("no saved prompt timeline: rebuilding it from the fixed prompt order and lining it up with the data...")
        acc, start = find_start(stream, cfg, guess)
        print(f"  prompts started {start - stream.et[0]:.1f} s after recording began (best match {100 * acc:.0f}%)")
        segments = segments_from_schedule(cfg, start)
        seg_file.write_text(json.dumps(segments, indent=1) + "\n")
    return train_and_save(stream, segments, cfg, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("teach", help="prompted movement session; trains and saves a model")
    t.add_argument("--url", required=True)
    t.add_argument("--session", required=True, help="session folder, e.g. data/sessions/20261107-143640")
    t.add_argument("--name", default="set-1", help="name for this teaching set")
    l = sub.add_parser("live", help="recognise movements live and drive motors (dry run by default)")
    l.add_argument("--url", required=True)
    l.add_argument("--model", required=True, help="model.json from teach")
    l.add_argument("--port")
    l.add_argument("--execute", action="store_true")
    r = sub.add_parser("retrain", help="train again from a saved teaching folder (events.jsonl)")
    r.add_argument("folder", help="e.g. data/sessions/ID/movements/set-1")
    args = parser.parse_args(argv)
    cfg = json.loads(CONFIG_FILE.read_text())
    return {"teach": teach, "live": live, "retrain": retrain}[args.command](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
