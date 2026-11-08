"""Train the EMG(+band motion) decoder that drives the leader arm, with its output in servo ticks.

    # labels from the leader arm's encoders (record with scripts/record_puppet_block.sh)
    python -m neural_reflex.decode_arm data/sessions/<id>/blocks/puppet-1 [more blocks] --labels encoders
    # labels from webcam arm angles (record with scripts/record_arm_block.sh)
    python -m neural_reflex.decode_arm data/sessions/<id>/blocks/arm-reach --labels camera

Steps:
1. Pair 20 ms feature windows with labels in robot ticks:
   - encoders: arm_encoders.jsonl ticks at the window end + --label-delay. Features come from the same
     FeatureStream the live loop uses, so training and live inputs are identical.
   - camera: armfit.pair_block windows; angles mapped to ticks with config/robot_map.json + arm_home.json
       ticks = home + sign * gain * ticks_per_degree * (angle - human_at_home)
2. Hold out every fifth 5 s chunk (armfit.chunk_split), pick the ridge penalty on the inner validation
   chunks, report error on the held-out chunks after the same smoothing the arm applies.
   Ridge is linear in its targets, so fitting ticks directly equals fitting angles and mapping after.
3. Refit on every window and write
     data/models/arm_decoder.npz   mean, scale, W, b (ticks), target_names, clamp_lo, clamp_hi, inputs
                                   (the format the FPGA weight exporter and live_arm read)
     data/models/arm_calib.npz     X (all windows' raw features), Y (label ticks)
     <block>/robot_plan_decoded.json + robot_preview_decoded.html  decoded held-out movement for
                                   play_plan --dry-run / the arm, and a side-by-side preview page.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .armfit import REGULARIZATION, TARGETS, chunk_split, interpolate, pair_block
from .dashboard import load_home
from .handfit import r2, read_jsonl
from .leader_arm import JOINTS
from .ridge_decoder import fit_ridge, predict_angles
from .robot_map import DT, VIEWER as PREVIEW, feasibility, follow, load_map, lowpass, stretch

ARM_TARGETS = ("shoulder_lift", "elbow_flex")   # webcam wrist_flex does not beat guessing (R² < 0)
REPO = Path(__file__).resolve().parent.parent
MAX_ENCODER_GAP_S = 0.1
GOAL_DEG = 2.0


def to_ticks(angles: np.ndarray, joints, config: dict, home: dict) -> np.ndarray:
    """Camera angles (n, len(joints)) -> robot ticks, without clamping."""
    k = config["ticks_per_degree"]
    out = np.empty_like(angles, dtype=float)
    for c, joint in enumerate(joints):
        spec = config["joints"][joint]
        out[:, c] = home[joint] + spec["sign"] * spec["gain"] * k * (angles[:, c] - spec["human_at_home"])
    return out


def to_human(ticks: np.ndarray, joint: str, config: dict, home: dict) -> np.ndarray:
    """Inverse of to_ticks for one joint (for the preview page's 'you' trace)."""
    spec = config["joints"][joint]
    return spec["human_at_home"] + (ticks - home[joint]) / (spec["sign"] * spec["gain"] * config["ticks_per_degree"])


def predict_ticks(m: dict, x: np.ndarray) -> np.ndarray:
    return np.clip((x - m["mean"]) / m["scale"] @ m["W"] + m["b"], m["clamp_lo"], m["clamp_hi"])


def pair_camera(block_dir: Path, joints, inputs: str, config: dict, home: dict, side=None) -> dict:
    b = pair_block(block_dir, side)
    x = np.hstack([b["emg"], b["motion"]]) if inputs == "emg+motion" else b["emg"]
    angles = b["targets"][:, [TARGETS.index(j) for j in joints]]
    return {"name": b["name"], "features": x, "ticks": to_ticks(angles, joints, config, home),
            "times": b["times"], "cues": b["cues"]}


def pair_encoders(block_dir: Path, joints, inputs: str, label_delay: float) -> dict:
    from .live_arm import FeatureStream, replay_events   # live_arm imports this module
    stream = FeatureStream(use_motion=inputs == "emg+motion")
    windows = []
    for kind, event in replay_events(block_dir, realtime=False):
        if kind == "motion":
            stream.motion(event)
        else:
            windows += stream.emg(event)
    rows = sorted(read_jsonl(block_dir / "arm_encoders.jsonl"), key=lambda r: r["host_time_s"])
    enc_t = np.array([r["host_time_s"] for r in rows])
    enc_v = np.array([[r["positions"][j] for j in joints] for r in rows], dtype=float)
    cue_list = [r.get("cue") for r in rows]
    x, y, t, cues = [], [], [], []
    for t_end, features in windows:
        label = interpolate(enc_t, enc_v, t_end + label_delay, MAX_ENCODER_GAP_S)
        if label is None:
            continue
        x.append(features)
        y.append(label)
        t.append(t_end)
        cues.append(cue_list[max(0, int(np.searchsorted(enc_t, t_end)) - 1)])
    if len(y) < 100:
        raise ValueError(f"{block_dir.name}: only {len(y)} feature windows overlap the encoder recording "
                         "(were emg.jsonl and arm_encoders.jsonl recorded at the same time?)")
    return {"name": block_dir.name, "features": np.array(x), "ticks": np.array(y), "times": np.array(t), "cues": cues}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("blocks", nargs="+", help="block folders with emg.jsonl + motion.jsonl, and "
                        "arm_encoders.jsonl (--labels encoders) or arm.jsonl (--labels camera)")
    parser.add_argument("--labels", choices=("encoders", "camera"), default="encoders")
    parser.add_argument("--inputs", choices=("emg", "emg+motion"), default="emg+motion",
                        help="emg = band EMG features only; emg+motion adds band orientation and gyro")
    parser.add_argument("--joints", nargs="+", default=list(ARM_TARGETS), choices=JOINTS,
                        help="joints to decode (camera labels support shoulder_lift, elbow_flex, wrist_flex)")
    parser.add_argument("--label-delay", type=float, default=0.05,
                        help="encoder labels: seconds after the EMG window (muscles fire before the arm moves)")
    parser.add_argument("--side", choices=("left", "right"), help="camera labels: arm wearing the band")
    parser.add_argument("--speed", choices=("fine", "normal", "fast"), default="normal")
    parser.add_argument("--output", default=str(REPO / "data" / "models"))
    args = parser.parse_args(argv)

    config, home = load_map(), load_home()
    if home is None:
        raise SystemExit("config/arm_home.json is required")
    joints = tuple(args.joints)
    if args.labels == "camera":
        blocks = [pair_camera(Path(b), joints, args.inputs, config, home, args.side) for b in args.blocks]
    else:
        blocks = [pair_encoders(Path(b), joints, args.inputs, args.label_delay) for b in args.blocks]
    features = np.concatenate([b["features"] for b in blocks])
    margin = config["margin_ticks"]
    lo = np.array([config["limits"][j][0] + margin for j in joints], dtype=float)
    hi = np.array([config["limits"][j][1] - margin for j in joints], dtype=float)
    truth = np.clip(np.concatenate([b["ticks"] for b in blocks]), lo, hi)
    splits = [chunk_split(b["times"]) for b in blocks]
    train, test, inner_val = (np.concatenate([s[i] for s in splits]) for i in range(3))
    inner = train & ~inner_val

    def fit(mask, reg):
        m = fit_ridge(features, truth, mask, reg)
        return {"mean": np.asarray(m["mean"]), "scale": np.asarray(m["scale"]), "W": np.asarray(m["weights"]),
                "b": np.asarray(m["bias"]), "clamp_lo": lo, "clamp_hi": hi}

    best = max(REGULARIZATION, key=lambda reg: np.nan_to_num(np.nanmean(
        r2(truth[inner_val], predict_angles(fit_ridge(features, truth, inner, reg), features[inner_val]))),
        nan=-np.inf))
    pred = predict_ticks(fit(train, best), features)
    k = config["ticks_per_degree"]
    smoothed = np.empty_like(pred)
    offset = 0
    for b in blocks:   # smooth within each block, as the arm thread would
        n = len(b["ticks"])
        for c, joint in enumerate(joints):
            smoothed[offset:offset + n, c] = lowpass(pred[offset:offset + n, c], config["joints"][joint]["smooth_s"])
        offset += n
    seconds = sum(b["times"][-1] - b["times"][0] for b in blocks)
    print(f"{args.labels} labels, {len(blocks)} block(s), {seconds:.0f} s; {len(features)} windows x "
          f"{features.shape[1]} features ({args.inputs}); train {int(train.sum())} / held-out {int(test.sum())}; "
          f"ridge penalty {best:g}")
    print(f"held-out, robot degrees  {'R²':>6}{'mean err':>10}{'p95 err':>9}{'max err':>9}"
          f"{f'≤{GOAL_DEG:g}°':>7}{'guessing':>10}")
    for c, joint in enumerate(joints):
        t, ps = truth[test, c], smoothed[test, c]
        err = np.abs(t - ps) / k
        print(f"{joint:24}{float(r2(t[:, None], ps[:, None])[0]):>7.2f}{err.mean():>9.1f}°{np.percentile(err, 95):>8.1f}°"
              f"{err.max():>8.1f}°{100 * (err <= GOAL_DEG).mean():>6.0f}%{np.abs(t - truth[train, c].mean()).mean() / k:>9.1f}°")

    final = fit(np.ones(len(features), bool), best)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "arm_decoder.npz", **final, target_names=np.array(joints), inputs=args.inputs,
             labels=args.labels, regularization=best)
    np.savez(out / "arm_calib.npz", X=features, Y=truth)
    print(f"Model: {out / 'arm_decoder.npz'}  calibration: {out / 'arm_calib.npz'}")

    # decoded held-out movement -> robot plan + preview (other joints stay at home), per block
    cap = config["speed_caps_ticks_per_s"][args.speed]
    offset = 0
    for block, block_dir in zip(blocks, args.blocks):
        n = len(block["ticks"])
        mask = test[offset:offset + n]
        targets = {j: np.full(int(mask.sum()), float(home[j])) for j in JOINTS}
        human = {j: np.full(int(mask.sum()), float(config["joints"][j]["human_at_home"])) for j in JOINTS}
        for c, joint in enumerate(joints):
            targets[joint] = smoothed[offset:offset + n][mask, c]
            human[joint] = to_human(truth[offset:offset + n][mask, c], joint, config, home)
        cues = np.array(block["cues"], dtype=object)[mask]
        offset += n
        report = feasibility(targets, home, config, args.speed)
        slowdown = max(report[j]["slowdown_for_5deg"] for j in joints)
        planned = {j: stretch(targets[j], slowdown) for j in JOINTS}
        plan = {"block": block["name"], "source": f"decoded {args.inputs}, {args.labels} labels, held-out chunks",
                "speed": args.speed, "slowdown": slowdown, "dt_s": DT, "home": home, "moving": list(joints),
                "targets": {j: [int(round(v)) for v in planned[j]] for j in JOINTS}}
        path = Path(block_dir) / "robot_plan_decoded.json"
        path.write_text(json.dumps(plan) + "\n")
        step = 2
        preview = {"joints": list(JOINTS), "dt_s": DT * step, "slowdown": slowdown, "speed": args.speed,
                   "home": home, "limits": config["limits"], "margin": margin, "k": k,
                   "verified": {j: config["joints"][j]["verified"] for j in JOINTS}, "report": report,
                   "clamped_pct": {j: 0.0 for j in JOINTS},
                   "cue": [cues[min(len(cues) - 1, int(i / slowdown))] for i in range(len(planned["gripper"]))][::step],
                   "human": {j: [round(float(v), 1) for v in stretch(human[j], slowdown)[::step]] for j in JOINTS},
                   "target": {j: [int(v) for v in planned[j][::step]] for j in JOINTS},
                   "robot": {j: [int(v) for v in follow(planned[j], home[j], cap, config["acceleration_ticks_per_s2"])[::step]]
                             for j in JOINTS}}
        page = PREVIEW.read_text(encoding="utf-8").replace(
            "__ROBOT_DATA__", json.dumps(preview, separators=(",", ":")).replace("</", "<\\/"))
        (path.parent / "robot_preview_decoded.html").write_text(page, encoding="utf-8")
        print(f"Plan: {path} ({len(plan['targets'][joints[0]]) * DT:.0f} s at {slowdown:g}x slow-mo)  "
              f"Preview: {path.parent / 'robot_preview_decoded.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
