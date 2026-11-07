"""Offline EMG-only decoding with a purged chronological train/validation/test split.

No camera, network, serial connection, or robot commands are used by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

from .arm_pose import SIDES
from .signal import design_sos, window_features

TARGETS = ("shoulder_lift", "elbow_flex", "wrist_flex")
CONFIG = {"sample_rate": 2048, "window_samples": 205, "hop_samples": 41,
          "history": 31, "warmup_samples": 2048}


def read_rows(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def label_values(row, side):
    """Independent visibility/geometry gates; wrist is an unsigned bend proxy."""
    out = np.full(3, np.nan)
    if row.get("world_landmarks") is None or row.get("visibility") is None:
        return out
    p, v = np.asarray(row["world_landmarks"], float), np.asarray(row["visibility"], float)
    if p.shape != (33, 3) or v.shape != (33,):
        raise ValueError("expected 33 world landmarks and visibility values")
    idx = SIDES[side]
    s, e, w, i, k = [idx[key] for key in ("shoulder", "elbow", "wrist", "index", "pinky")]
    upper, fore = p[e] - p[s], p[w] - p[e]
    up = (p[11] + p[12] - p[23] - p[24]) / 2
    hand = (p[i] + p[k]) / 2 - p[w]

    def angle(a, b, needed):
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if (norm < 1e-10 or not np.isfinite(norm) or
                not np.isfinite(v[needed]).all() or np.min(v[needed]) < .5):
            return np.nan
        return np.degrees(np.arccos(np.clip(np.dot(a, b) / norm, -1, 1)))

    out[0] = angle(-up, upper, [s, e, 11, 12, 23, 24])
    out[1] = angle(upper, fore, [s, e, w])
    out[2] = angle(fore, hand, [e, w, i, k])
    return out


def load_recording(block, side="right"):
    """Read acquisition order; fail closed on gaps/reconnects until explicitly segmented.

    Only the first 60% of band time is used to map device time to host receipt time.
    This is a clock approximation, not measured camera exposure synchronization.
    """
    block = Path(block)
    records = read_rows(block / "emg.jsonl")
    pose = read_rows(block / "arm.jsonl")
    if len(records) < 100 or len(pose) < 30:
        raise ValueError("too few EMG batches or pose frames")
    if len({(r.get("stream_session_id"), r.get("band_connection_id")) for r in records}) != 1:
        raise ValueError("multiple recording epochs: split at reconnects before training")
    sequences = np.array([r["sequence"] for r in records], dtype=np.int64)
    if np.any(np.diff(sequences) % 2**32 != 1):
        raise ValueError("EMG sequence gap, duplicate, or reset: segment the recording first")
    band = np.array([r["band_timestamp_us"] for r in records], float) / 1e6
    host = np.array([r["host_time_s"] for r in records], float)
    intervals = np.array([r["sample_interval_us"] for r in records], float) / 1e6
    if (not np.isfinite(band).all() or not np.isfinite(host).all() or
            np.any(np.diff(band) <= 0) or np.any(np.diff(host) < 0) or
            any(r["sample_rate_hz"] != CONFIG["sample_rate"] for r in records) or
            not np.allclose(intervals, 1 / CONFIG["sample_rate"], atol=1e-9)):
        raise ValueError("invalid clock or unsupported sample rate")
    if not np.allclose(np.diff(band), 16 * intervals[:-1], atol=2e-6, rtol=0):
        raise ValueError("EMG acquisition timestamps contain a gap")
    raw = np.asarray([r["values"] for r in records], float)
    if raw.shape != (len(records), 16, 8) or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 65535)):
        raise ValueError("expected finite unsigned ADC batches with shape 16 by 8")
    clock_train = band < band[0] + .6 * (band[-1] - band[0])
    slope, offset = np.polyfit(band[clock_train] - band[0], host[clock_train] - host[0], 1)
    if not .99 < slope < 1.01:
        raise ValueError("implausible device-to-host clock drift")
    device_times = (band[:, None] - band[0] + intervals[:, None] * np.arange(16)).ravel()
    times = host[0] + offset + slope * device_times
    pt = np.array([r["host_time_s"] for r in pose], float)
    if not np.isfinite(pt).all() or np.any(np.diff(pt) <= 0):
        raise ValueError("pose timestamps must be finite and increasing")
    residual = host - (host[0] + offset + slope * (band - band[0]))
    return {"times": times, "raw": raw.reshape(-1, 8), "pose_times": pt,
            "pose_targets": np.array([label_values(r, side) for r in pose]),
            "cues": np.array([r.get("cue") or "unlabelled" for r in pose]),
            "clock": {"fit": "first 60% of EMG device time; host receipt timestamps",
                      "slope": float(slope), "receipt_residual_p05_p50_p95_ms":
                      (np.percentile(residual, [5, 50, 95]) * 1000).tolist()},
            "side": side}


def split_intervals(start, end, guard_s=.8):
    if end - start < 20 or guard_s < 0:
        raise ValueError("recording must exceed 20 seconds and guard must be nonnegative")
    a, b = start + .6 * (end - start), start + .8 * (end - start)
    intervals = {"train": (start, a - guard_s), "validation": (a + guard_s, b - guard_s),
                 "test": (b + guard_s, end)}
    if any(hi - lo < 3 for lo, hi in intervals.values()):
        raise ValueError("split too short after boundary guards")
    return intervals


def extract_features(raw, sample_rate=2048, window_samples=205, hop_samples=41,
                     history=31, warmup_samples=2048):
    """Causal filtered features, ordered oldest-to-newest, independently initialized.

    Returned starts are the direct raw-window dependency. The IIR also depends on
    earlier samples in this segment, so each evaluation split resets the filter.
    """
    raw = np.asarray(raw, float)
    if raw.ndim != 2 or raw.shape[1] != 8 or not np.isfinite(raw).all():
        raise ValueError("raw must be a finite samples by eight array")
    if min(window_samples - 2, hop_samples, history) < 1 or warmup_samples < 0:
        raise ValueError("invalid feature window configuration")
    first_end = warmup_samples + window_samples - 1
    if len(raw) <= first_end + (history - 1) * hop_samples:
        return np.empty((0, history, 40)), np.array([], int), np.array([], int)
    sos = design_sos(sample_rate)
    zi = sosfilt_zi(sos)[:, :, None] * raw[0][None, None, :]
    filtered, _ = sosfilt(sos, raw, axis=0, zi=zi)
    ends = np.arange(first_end, len(raw), hop_samples)
    features = np.array([window_features(filtered[e - window_samples + 1:e + 1]) for e in ends])
    features[:, :24] = np.log1p(features[:, :24])
    x = np.stack([features[i - history + 1:i + 1] for i in range(history - 1, len(features))])
    ends = ends[history - 1:]
    starts = ends - (history - 1) * hop_samples - window_samples + 1
    return x, ends, starts


def pair_segment(recording, interval, label_delay_s=.05, config=None):
    """No label interpolation or EMG/filter dependency crosses the interval."""
    config = CONFIG if config is None else config
    lo, hi = interval
    t = recording["times"]
    raw_idx = np.flatnonzero((t >= lo) & (t < hi))
    x, ends, starts = extract_features(recording["raw"][raw_idx], **config)
    end_t = t[raw_idx[ends]]
    at = end_t + label_delay_s
    pt, py = recording["pose_times"], recording["pose_targets"]
    after = np.searchsorted(pt, at)
    bounded = (after > 0) & (after < len(pt))
    after = np.clip(after, 1, len(pt) - 1)
    before = after - 1
    span = pt[after] - pt[before]
    valid = bounded & (pt[before] >= lo) & (pt[after] < hi) & (span <= .15)
    weight = (at - pt[before]) / span
    y = (1 - weight[:, None]) * py[before] + weight[:, None] * py[after]
    valid &= np.isfinite(y[:, :2]).all(axis=1)
    rail = (recording["raw"] <= 100) | (recording["raw"] >= 65435)
    rail_sum = np.vstack([np.zeros(8, int), np.cumsum(rail, axis=0)])
    global_start, global_end = raw_idx[starts], raw_idx[ends]
    clipped = (rail_sum[global_end + 1] - rail_sum[global_start]).sum(axis=1) / ((global_end - global_start + 1) * 8)
    return {"x": x[valid], "y": y[valid], "times": end_t[valid],
            "input_start": t[global_start[valid]], "label_end": pt[after[valid]],
            "cues": recording["cues"][before[valid]], "clip_fraction": clipped[valid],
            "excluded_windows": int((~valid).sum()), "interval": list(interval)}


def prepare(block, side="right", label_delay_s=.05):
    rec = load_recording(block, side)
    start = max(rec["times"][0], rec["pose_times"][0])
    end = min(rec["times"][-1], rec["pose_times"][-1])
    intervals = split_intervals(start, end)
    splits = {k: pair_segment(rec, v, label_delay_s) for k, v in intervals.items()}
    if any(len(p["y"]) < 50 for p in splits.values()):
        raise ValueError("at least one split has fewer than 50 valid primary-target windows")
    return rec, splits


def fit_scaler(x):
    flat = np.asarray(x).reshape(-1, x.shape[-1])
    return {"mean": flat.mean(axis=0), "scale": np.maximum(flat.std(axis=0), 1e-6)}


def transform_features(x, scaler):
    return np.clip((x - scaler["mean"]) / scaler["scale"], -6, 6).astype(np.float32)


def score(actual, predicted):
    result = {"mae_deg": [], "r2": [], "count": []}
    for y, p in zip(np.asarray(actual).T, np.asarray(predicted).T):
        mask = np.isfinite(y) & np.isfinite(p)
        y, p = y[mask], p[mask]
        result["count"].append(len(y))
        result["mae_deg"].append(float(np.mean(np.abs(y - p))) if len(y) else None)
        variance = float(np.var(y)) if len(y) else 0
        result["r2"].append(float(1 - np.mean((y - p)**2) / variance) if variance >= 1 else None)
    return result


def selection_loss(actual, predicted, target_scale):
    return float(np.nanmean(np.abs(actual[:, :2] - predicted[:, :2]) / target_scale[:2]))


def regress(x, y, alpha):
    weights, biases = [], []
    for target in y.T:
        keep = np.isfinite(target)
        if keep.sum() < 10:
            raise ValueError("fewer than 10 valid labels for one model output")
        a, b = x[keep], target[keep]
        xm, ym = a.mean(axis=0), b.mean()
        centered = a - xm
        w = np.linalg.solve(centered.T @ centered + alpha * np.eye(x.shape[1]), centered.T @ (b - ym))
        weights.append(w)
        biases.append(ym - xm @ w)
    return np.stack(weights, axis=1), np.array(biases)


def rbf_features(x, anchors, width):
    distance = np.maximum((x*x).sum(1)[:, None] + (anchors*anchors).sum(1)[None, :] - 2*x@anchors.T, 0)
    return np.exp(-distance / (2 * width**2))


class Decoder:
    """Persistable feature-history decoder; outputs camera-angle proxies, not motor ticks."""
    def __init__(self, kind, scaler, weights, bias, anchors=None, width=None):
        self.kind, self.scaler = kind, scaler
        self.weights, self.bias, self.anchors, self.width = weights, bias, anchors, width

    def predict(self, features):
        x = transform_features(features, self.scaler)[:, ::5].reshape(len(features), -1).astype(float)
        if self.kind == "rbf":
            x = rbf_features(x, self.anchors, self.width)
        if self.kind == "constant":
            return np.tile(self.bias, (len(features), 1))
        return np.clip(x @ self.weights + self.bias, 0, 180)

    def save(self, path):
        arrays = {"kind": np.array(self.kind), "mean": self.scaler["mean"], "scale": self.scaler["scale"],
                  "weights": self.weights, "bias": self.bias}
        if self.kind == "rbf":
            arrays.update(anchors=self.anchors, width=np.array(self.width))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            if str(a["kind"]) == "tcn":
                return TCNDecoder.from_arrays(a)
            return cls(str(a["kind"]), {"mean": a["mean"], "scale": a["scale"]}, a["weights"], a["bias"],
                       a["anchors"] if "anchors" in a else None, float(a["width"]) if "width" in a else None)


def fit_classical(splits, seed=7):
    train, val = splits["train"], splits["validation"]
    scaler = fit_scaler(train["x"])
    x = transform_features(train["x"], scaler)[:, ::5].reshape(len(train["x"]), -1).astype(float)
    target_scale = np.maximum(np.nanstd(train["y"], axis=0), 5)
    mean = np.nanmean(train["y"], axis=0)
    models = {"Training mean": Decoder("constant", scaler, np.empty((0, 3)), mean)}
    rng = np.random.default_rng(seed)
    anchors = x[rng.choice(len(x), min(128, len(x)), replace=False)]
    typical = np.sqrt(np.median(np.sum((anchors[:, None] - anchors[None, :])**2, axis=2)))
    trials = []
    for name in ("Ridge", "RBF regression"):
        best = (np.inf, None)
        for width in ([None] if name == "Ridge" else [typical * .5, typical, typical * 2]):
            z = x if width is None else rbf_features(x, anchors, width)
            for alpha in ([1, 10, 100, 1000, 10000] if width is None else [.1, 1, 10, 100]):
                weights, bias = regress(z, train["y"], alpha)
                model = Decoder("ridge" if width is None else "rbf", scaler, weights, bias, anchors, width)
                loss = selection_loss(val["y"], model.predict(val["x"]), target_scale)
                trials.append({"model": name, "alpha": alpha, "width": width, "validation_loss": loss})
                if loss < best[0]:
                    best = (loss, model)
        models[name] = best[1]
    return models, trials


def make_tcn():
    import torch
    from torch import nn

    class CausalTCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            for i, dilation in enumerate((1, 2, 4, 8)):
                layers.extend([nn.ConstantPad1d((2 * dilation, 0), 0),
                               nn.Conv1d(40 if i == 0 else 32, 32, 3, dilation=dilation),
                               nn.GELU(), nn.Dropout(.1)])
            self.stack = nn.Sequential(*layers)
            self.output = nn.Linear(32, 3)

        def forward(self, x):
            return self.output(self.stack(x.transpose(1, 2))[:, :, -1])

    return CausalTCN()


class TCNDecoder:
    kind = "tcn"

    def __init__(self, network, scaler, target_mean, target_scale):
        self.network, self.scaler = network.eval(), scaler
        self.target_mean, self.target_scale = target_mean, target_scale

    def predict(self, features):
        import torch
        x = transform_features(features, self.scaler)
        with torch.inference_mode():
            out = np.concatenate([self.network(torch.from_numpy(x[i:i+256])).numpy()
                                  for i in range(0, len(x), 256)])
        return np.clip(out * self.target_scale + self.target_mean, 0, 180)

    def save(self, path):
        arrays = {"kind": np.array("tcn"), "mean": self.scaler["mean"], "scale": self.scaler["scale"],
                  "target_mean": self.target_mean, "target_scale": self.target_scale}
        arrays.update({"state_" + k: v.detach().numpy() for k, v in self.network.state_dict().items()})
        np.savez_compressed(path, **arrays)

    @classmethod
    def from_arrays(cls, arrays):
        import torch
        network = make_tcn()
        network.load_state_dict({k[6:]: torch.from_numpy(arrays[k]) for k in arrays if k.startswith("state_")})
        return cls(network, {"mean": arrays["mean"], "scale": arrays["scale"]},
                   arrays["target_mean"], arrays["target_scale"])


def fit_tcn(splits, seed=7, epochs=80):
    import copy
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    train, val = splits["train"], splits["validation"]
    scaler = fit_scaler(train["x"])
    mean, scale = np.nanmean(train["y"], axis=0), np.maximum(np.nanstd(train["y"], axis=0), 5)
    model = TCNDecoder(make_tcn(), scaler, mean, scale)
    x = torch.from_numpy(transform_features(train["x"], scaler))
    normalized = (train["y"] - mean) / scale
    mask = torch.from_numpy(np.isfinite(normalized).astype(np.float32))
    y = torch.from_numpy(np.nan_to_num(normalized).astype(np.float32))
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=.002, weight_decay=.001)
    # Wrist proxy is diagnostic; primary supervision is shoulder and elbow.
    output_weight = torch.tensor([1, 1, .1])
    best_loss, best_state, best_epoch, stale = np.inf, None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.network.train()
        order = torch.randperm(len(x))
        for indices in order.split(128):
            optimizer.zero_grad()
            error = (model.network(x[indices]) - y[indices])**2
            weights = mask[indices] * output_weight
            loss = (error * weights).sum() / weights.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1)
            optimizer.step()
        model.network.eval()
        val_loss = selection_loss(val["y"], model.predict(val["x"]), scale)
        history.append({"epoch": epoch, "validation_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, best_epoch, stale = val_loss, copy.deepcopy(model.network.state_dict()), epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"TCN epoch {epoch}: validation normalized MAE {val_loss:.4f}", flush=True)
        if stale >= 15:
            break
    model.network.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "epochs_run": epoch, "history": history}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("block", type=Path)
    parser.add_argument("--side", choices=("left", "right"), required=True,
                        help="declared tracked arm; never inferred from gyro or test labels")
    parser.add_argument("--output", type=Path, required=True, help="new artifact directory (refuses overwrite)")
    parser.add_argument("--label-delay-ms", type=float, default=50,
                        help="predeclared camera-label offset; not a measured physiological latency")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--skip-tcn", action="store_true")
    args = parser.parse_args(argv)
    if not np.isfinite(args.label_delay_ms) or abs(args.label_delay_ms) > 500 or args.epochs < 1:
        parser.error("label delay must be finite and within +/-500 ms; epochs must be positive")
    if args.output.exists():
        parser.error("output already exists; use a new directory to preserve prior results")
    args.output.mkdir(parents=True)
    rec, splits = prepare(args.block, args.side, args.label_delay_ms / 1000)
    print("Prepared " + ", ".join(f"{k}: {len(v['y'])} windows" for k, v in splits.items()), flush=True)
    models, trials = fit_classical(splits, args.seed)
    training = {}
    if not args.skip_tcn:
        print("Training causal TCN on CPU", flush=True)
        models["Causal TCN"], training = fit_tcn(splits, args.seed, args.epochs)
    scale = np.maximum(np.nanstd(splits["train"]["y"], axis=0), 5)
    validation_predictions = {k: m.predict(splits["validation"]["x"]) for k, m in models.items()}
    # Freeze model selection before consulting test labels or predictions.
    selected = min(models, key=lambda k: selection_loss(splits["validation"]["y"], validation_predictions[k], scale))
    print(f"Validation selected: {selected}. Evaluating untouched test interval.", flush=True)
    model_rows, test_predictions = [], {}
    for name, model in models.items():
        pred = model.predict(splits["test"]["x"])
        test_predictions[name] = pred
        model_path = args.output / (model.kind + ".npz")
        model.save(model_path)
        loaded = Decoder.load(model_path)
        np.testing.assert_allclose(loaded.predict(splits["test"]["x"][:10]), pred[:10], rtol=1e-5, atol=1e-5)
        timings = []
        for _ in range(35):
            now = time.perf_counter()
            model.predict(splits["test"]["x"][:1])
            timings.append((time.perf_counter() - now) * 1000)
        by_cue = {str(c): score(splits["test"]["y"][splits["test"]["cues"] == c],
                               pred[splits["test"]["cues"] == c]) for c in np.unique(splits["test"]["cues"])}
        unclipped = splits["test"]["clip_fraction"] == 0
        row = {"name": name, "artifact": model_path.name,
               "validation": score(splits["validation"]["y"], validation_predictions[name]),
               "test": score(splits["test"]["y"], pred), "test_by_cue": by_cue,
               "test_no_clipping": score(splits["test"]["y"][unclipped], pred[unclipped]),
               "inference_ms": float(np.median(timings[5:]))}
        model_rows.append(row)
        print(f"{name}: test MAE degrees {row['test']['mae_deg']}", flush=True)
    from .recording_audit import audit_block
    audit = audit_block(args.block)
    write_json(args.output / "audit.json", audit)
    context_ms = 1000 * (CONFIG["window_samples"] + (CONFIG["history"] - 1) * CONFIG["hop_samples"]) / CONFIG["sample_rate"]
    t0 = float(rec["pose_times"][0])
    caveats = [
        "One approximately 133-second guided session, one placement: not evidence of next-day or new-person accuracy.",
        f"MediaPipe {args.side} is an explicit dataset assumption; a session manifest confirming the physical band side is missing.",
        "Shoulder elevation and elbow bend are primary targets. Wrist is an unsigned bend proxy, not signed flexion or forearm rotation.",
        "Camera exposure synchronization is unknown. The fixed label offset is an assumption; BLE receipt bursts do not measure acquisition latency.",
        "The chronological validation interval mainly contains wrist exercises; model selection may transfer poorly to the final free-motion/home/rest interval.",
        "EMG clipping is retained in the primary scores. Clipping varies by task; the final interval contains relatively little clipping.",
        "A short forearm EMG history does not uniquely specify static shoulder/elbow posture. Task correlations can inflate within-session prediction.",
        "Reported inference time is local feature-to-angle compute only; it excludes BLE, filtering/features, camera and robot delays.",
        "No live robot control is enabled. These outputs are human camera-angle labels, not calibrated motor commands.",
    ]
    result = {
        "title": "EMG-only joint decoder · chronological holdout",
        "dataset": {"duration_s": float(rec["pose_times"][-1] - rec["pose_times"][0]),
                    "emg_samples": len(rec["raw"]), "pose_frames": len(rec["pose_times"]),
                    "description": "First guided arm-reach recording; EMG is the only model input. Camera angles supply training/evaluation labels."},
        "protocol": {"name": "Chronological 60% train / 20% validation / 20% final test",
                     "description": "0.8 s guards on both sides of each boundary; independent filter initialization and 1 s warmup in every split. Hyperparameters and model selected using validation only; no refit on validation. Final scores use unsmoothed predictions.",
                     **{k + "_windows": len(v["y"]) for k, v in splits.items()},
                     "context_ms": context_ms, "hop_ms": 1000 * CONFIG["hop_samples"] / CONFIG["sample_rate"],
                     "label_delay_ms": args.label_delay_ms,
                     "selection": "mean shoulder/elbow MAE divided by training standard deviation (minimum 5 degrees)",
                     "intervals_seconds_from_first_pose": {k: [lo-t0, hi-t0] for k, (lo, hi) in
                                                          ((k, v["interval"]) for k, v in splits.items())}},
        "models": model_rows, "selected_model": selected,
        "targets": [{"name": name, "label": label, "unit": "deg"} for name, label in
                    zip(TARGETS, ("Shoulder elevation", "Elbow bend", "Wrist bend · unsigned proxy"))],
        "caveats": caveats, "config": CONFIG, "clock": rec["clock"], "side": args.side,
        "seed": args.seed, "trials": trials, "tcn_training": training,
        "source_sha256": {name: hashlib.sha256((args.block / name).read_bytes()).hexdigest()
                          for name in ("emg.jsonl", "arm.jsonl")},
        "split_quality": {k: {"excluded_windows": p["excluded_windows"],
                              "windows_with_clipping": int((p["clip_fraction"] > 0).sum()),
                              "target_std_deg": np.nanstd(p["y"], axis=0).tolist(),
                              "cues": {str(c): int((p["cues"] == c).sum()) for c in np.unique(p["cues"])}}
                          for k, p in splits.items()},
    }
    write_json(args.output / "metrics.json", result)
    traces = [{"t": float(t-t0), "actual": [None if not np.isfinite(v) else float(v) for v in a],
               "predicted": p.tolist()} for t, a, p in zip(splits["test"]["times"], splits["test"]["y"], test_predictions[selected])]
    report_data = {**result, "traces": traces}
    template = Path(__file__).resolve().parent.parent / "web" / "benchmark" / "report.html"
    payload = json.dumps(report_data, allow_nan=False).replace("<", "\\u003c")
    (args.output / "report.html").write_text(template.read_text().replace("__BENCHMARK_DATA__", payload))
    write_json(args.output / "predictions.json", {"traces": traces})
    write_json(args.output / "selected_model.json", {"name": selected,
               "artifact": next(row["artifact"] for row in model_rows if row["name"] == selected),
               "config": CONFIG, "targets": TARGETS, "side": args.side,
               "label_delay_ms": args.label_delay_ms, "preprocessing":
               "causal 20-450 Hz bandpass and 60/120/180 Hz notches; initialized from first ADC sample; log1p amplitude features; train-only z-score clipped to +/-6"})
    print(f"Saved models, audit, metrics and report to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
