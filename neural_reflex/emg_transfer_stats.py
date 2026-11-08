"""Gap-aware descriptive statistics and frozen EMG decoder transfer evaluation.

Confidence intervals describe block resampling of these recordings, not uncertainty
over people or future sessions. No new model is fitted and no alignment is tuned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .emg_benchmark import CONFIG, TARGETS, Decoder, pair_segment, score
from .emg_correlations import _demean_by_cue, _pearson
from .emg_recordings import load_segments

SEED = 17
LABEL_DELAY_S = .05


def paired_segments(segments, history=1):
    """Reset filters and history at every acquisition gap; retain segment identity."""
    parts = []
    exclusions = []
    for segment in segments:
        lo = max(segment["times"][0], segment["pose_times"][0])
        hi = min(segment["times"][-1] + 1 / CONFIG["sample_rate"], segment["pose_times"][-1])
        if hi <= lo:
            continue
        p = pair_segment(segment, (lo, hi), LABEL_DELAY_S, {**CONFIG, "history": history})
        exclusions.append({"segment": segment["segment_id"], "paired_windows": len(p["y"]),
                           "label_excluded_windows": p["excluded_windows"],
                           "raw_samples": len(segment["raw"])})
        if not len(p["y"]):
            continue
        frame = np.clip(np.searchsorted(segment["pose_times"], p["times"] + LABEL_DELAY_S) - 1,
                        0, len(segment["pose_times"]) - 1)
        p["segments"] = np.full(len(p["y"]), segment["segment_id"], int)
        rounds = segment.get("rounds", np.full(len(segment["pose_times"]), None))
        p["rounds"] = np.array([int(rounds[i]) if rounds[i] is not None else 0 for i in frame])
        parts.append(p)
    if not parts:
        raise ValueError("no valid feature windows remain after gaps, warmup and label gating")
    keys = ("x", "y", "times", "input_start", "label_end", "cues", "clip_fraction", "segments", "rounds")
    merged = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    if np.any(np.diff(merged["times"]) <= 0):
        raise ValueError("clock mapping produces overlapping segment times")
    merged["segment_coverage"] = exclusions
    return merged


def temporal_blocks(times, segments, rounds, seconds=5):
    """Wall-time blocks never cross acquisition segments or recording rounds.

    Camera-invalid rows can be absent inside a block; time is never compressed.
    Partial blocks are retained and their sizes/durations are reported.
    """
    times, segments, rounds = map(np.asarray, (times, segments, rounds))
    if seconds <= 0 or not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("positive block length and increasing nonempty timestamps required")
    blocks, strata = [], []
    keys = list(dict.fromkeys(zip(segments.tolist(), rounds.tolist())))
    for segment, round_ in keys:
        indices = np.flatnonzero((segments == segment) & (rounds == round_))
        bins = np.floor((times[indices] - times[indices[0]]) / seconds).astype(int)
        for bin_ in np.unique(bins):
            block = indices[bins == bin_]
            blocks.append(block)
            strata.append((int(segment), int(round_)))
    return blocks, strata


def bootstrap_weights(strata, replicates=1000, seed=SEED):
    """Resample whole blocks within segment/round; singleton strata stay fixed."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(strata)), dtype=int)
    for key in dict.fromkeys(strata):
        choices = np.array([i for i, value in enumerate(strata) if value == key])
        sampled = rng.choice(choices, size=(replicates, len(choices)), replace=True)
        for row in range(replicates):
            weights[row] += np.bincount(sampled[row], minlength=len(strata))
    return weights


def block_summary(times, blocks, strata, seconds):
    dt = float(np.median(np.diff(times)))
    spans = [times[b[-1]] - times[b[0]] + dt for b in blocks]
    sizes = [len(b) for b in blocks]
    groups = {key: strata.count(key) for key in dict.fromkeys(strata)}
    return {"requested_seconds": seconds, "blocks": len(blocks), "strata": len(groups),
            "singleton_strata": sum(n == 1 for n in groups.values()),
            "span_seconds_p05_p50_p95": np.percentile(spans, [5, 50, 95]).tolist(),
            "windows_per_block_p05_p50_p95": np.percentile(sizes, [5, 50, 95]).tolist(),
            "partial_blocks": sum(span < seconds * .9 for span in spans)}


def pearson_interval(x, y, blocks, weights):
    """Exact Pearson sufficient statistics per block, with missing y masked."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    summaries = []
    for block in blocks:
        a, b = x[block], y[block]
        keep = np.isfinite(a) & np.isfinite(b)
        a, b = a[keep], b[keep]
        summaries.append([len(a), a.sum(), b.sum(), a@a, b@b, a@b])
    totals = weights @ np.asarray(summaries)
    n, sx, sy, sxx, syy, sxy = totals.T
    with np.errstate(invalid="ignore", divide="ignore"):
        covariance = sxy - sx * sy / n
        denominator = np.sqrt(np.maximum(sxx - sx*sx/n, 0) * np.maximum(syy - sy*sy/n, 0))
        coefficients = np.where((n >= 3) & (denominator > 1e-10), covariance / denominator, np.nan)
    finite = coefficients[np.isfinite(coefficients)]
    return np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else [None, None]


def paired_error_intervals(actual, predicted, baseline, blocks, weights):
    """Paired bootstrap; positive improvement means lower error than frozen mean."""
    result = []
    for j in range(actual.shape[1]):
        summaries = []
        for b in blocks:
            keep = np.isfinite(actual[b, j])
            yy, pp, bb = actual[b, j][keep], predicted[b, j][keep], baseline[b, j][keep]
            errors, base_errors = np.abs(yy-pp), np.abs(yy-bb)
            summaries.append([len(yy), errors.sum(), base_errors.sum(), (base_errors-errors).sum()])
        n, se, sb, improvement = (weights @ np.asarray(summaries)).T
        with np.errstate(invalid="ignore", divide="ignore"):
            mae, base_mae, difference = se/n, sb/n, improvement/n
        percentile = lambda values: np.percentile(values[np.isfinite(values)], [2.5, 97.5]).tolist() if np.isfinite(values).any() else [None, None]
        result.append({"mae_ci95_deg": percentile(mae), "baseline_mae_ci95_deg": percentile(base_mae),
                       "mae_improvement_ci95_deg": percentile(difference)})
    return result


def nullsafe(value):
    if isinstance(value, dict):
        return {str(k): nullsafe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [nullsafe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def subset_correlations(pair, mask):
    x, y = pair["x"][mask, -1, 16:24], pair["y"][mask]
    r, n = _pearson(x, y)
    rho = np.full_like(r, np.nan)
    for i in range(8):
        for j in range(3):
            keep = np.isfinite(y[:, j])
            if keep.sum() > 3 and np.std(x[keep, i]) > 0 and np.std(y[keep, j]) > 0:
                rho[i, j] = spearmanr(x[keep, i], y[keep, j]).statistic
    within, _ = _pearson(*_demean_by_cue(x, y, pair["cues"][mask]))
    return {"windows": int(mask.sum()), "pearson_r": r, "spearman_rho": rho,
            "cue_demeaned_r": within, "pair_counts": n,
            "target_std_deg": np.nanstd(y, axis=0), "target_p05_p50_p95_deg": np.nanpercentile(y, [5, 50, 95], axis=0)}


def correlation_report(pair, replicates=1000):
    x, y = pair["x"][:, -1, 16:24], pair["y"]
    result = subset_correlations(pair, np.ones(len(y), bool))
    result["rounds"] = {str(r): subset_correlations(pair, pair["rounds"] == r)
                        for r in np.unique(pair["rounds"]) if r > 0}
    unclipped = pair["clip_fraction"] == 0
    result["unclipped_only_diagnostic"] = subset_correlations(pair, unclipped)
    result["windows_with_near_rail_samples"] = int((~unclipped).sum())
    result["joint_angle_correlations"] = np.array([[_pair_r(y[:, i], y[:, j]) for j in range(3)] for i in range(3)])
    result["channel_correlations"] = np.corrcoef(x, rowvar=False)
    result["bootstrap"] = {}
    # EMG 1–shoulder negative and EMG 1–elbow positive were selected in arm-reach.
    # These two pair/sign checks were declared before inspecting picking data.
    for seconds in (2, 5, 10):
        blocks, strata = temporal_blocks(pair["times"], pair["segments"], pair["rounds"], seconds)
        weights = bootstrap_weights(strata, replicates)
        intervals = [pearson_interval(x[:, 0], y[:, j], blocks, weights) for j in (0, 1)]
        result["bootstrap"][str(seconds)] = {"blocks": block_summary(pair["times"], blocks, strata, seconds),
                                              "emg1_primary_pair_pearson_ci95": intervals}
    return nullsafe(result)


def _pair_r(a, b):
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.std(a[keep]) < 1e-12 or np.std(b[keep]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def extended_score(y, pred):
    return {**score(y, pred), "bias_deg": np.nanmean(pred-y, axis=0),
            "p90_absolute_error_deg": np.nanpercentile(np.abs(pred-y), 90, axis=0),
            "rmse_deg": np.sqrt(np.nanmean((pred-y)**2, axis=0)),
            "prediction_target_r": [_pair_r(y[:, j], pred[:, j]) for j in range(3)]}


def evaluate_frozen(pair, model_dir, replicates=1000):
    # Both artifacts were fitted/selected on arm-reach before pick-1 was received.
    ridge = Decoder.load(Path(model_dir) / "ridge.npz")
    mean = Decoder.load(Path(model_dir) / "constant.npz")
    predicted, baseline = ridge.predict(pair["x"]), mean.predict(pair["x"])
    y = pair["y"]
    report = {"model": "Previously validation-selected ridge; no fitting on the new recording",
              "ridge": extended_score(y, predicted), "training_mean": extended_score(y, baseline),
              "target_std_deg": np.nanstd(y, axis=0),
              "mae_improvement_deg": np.nanmean(np.abs(y-baseline)-np.abs(y-predicted), axis=0),
              "rounds": {}, "bootstrap": {}, "windows": len(y)}
    for r in np.unique(pair["rounds"]):
        if r > 0:
            keep = pair["rounds"] == r
            report["rounds"][str(r)] = {"ridge": extended_score(y[keep], predicted[keep]),
                                        "training_mean": extended_score(y[keep], baseline[keep])}
    for seconds in (2, 5, 10):
        blocks, strata = temporal_blocks(pair["times"], pair["segments"], pair["rounds"], seconds)
        weights = bootstrap_weights(strata, replicates)
        report["bootstrap"][str(seconds)] = {"blocks": block_summary(pair["times"], blocks, strata, seconds),
                                              "targets": paired_error_intervals(y, predicted, baseline, blocks, weights)}
    report["artifacts_sha256"] = {name: hashlib.sha256((Path(model_dir)/name).read_bytes()).hexdigest()
                                  for name in ("ridge.npz", "constant.npz")}
    return nullsafe(report), predicted, baseline


def save_plots(result, pairs, model_pair, prediction, baseline, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ("Shoulder elevation", "Elbow bend", "Wrist bend proxy")
    names = list(result["recordings"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    for column, name in enumerate(names):
        data = result["recordings"][name]["correlation"]
        for row, key in enumerate(("pearson_r", "cue_demeaned_r")):
            ax = axes[row, column]
            values = np.array(data[key], float).T
            im = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(8), range(1, 9))
            ax.set_yticks(range(3), labels if column == 0 else [""] * 3)
            ax.set_title(name + (" · overall" if row == 0 else " · exercise means removed"))
            ax.set_xlabel("EMG channel")
            for j in range(3):
                for i in range(8):
                    if np.isfinite(values[j,i]):
                        ax.text(i,j,f"{values[j,i]:+.2f}",ha="center",va="center",fontsize=9,
                                color="white" if abs(values[j,i])>.55 else "black")
    fig.colorbar(im, ax=axes, label="Pearson r", shrink=.8)
    fig.suptitle("Do EMG–angle relationships carry over to the new recording?")
    _save(fig, output, "correlation_transfer")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for j, ax in enumerate(axes):
        for i, name in enumerate(names):
            data = result["recordings"][name]["correlation"]
            value = data["pearson_r"][0][j]
            low, high = data["bootstrap"]["5"]["emg1_primary_pair_pearson_ci95"][j]
            ax.plot([low, high],[i,i],color=["#2764a1","#c0643c"][i],linewidth=3)
            ax.scatter([value],[i],color=["#2764a1","#c0643c"][i],s=70,zorder=3)
        ax.axvline(0,color="#777",linewidth=1)
        ax.set_xlim(-1,1); ax.set_ylim(-.7,1.7)
        ax.set_yticks([0,1],names); ax.set_title("EMG 1 vs " + labels[j].lower())
        ax.set_xlabel("Pearson r · descriptive 95% block-bootstrap interval")
        ax.grid(axis="x",alpha=.2)
    fig.suptitle("Previously strongest channel: same pair, new movement task",y=.97)
    fig.text(.5,.025,"5 s requested blocks, constrained by gaps/rounds. Intervals describe these recordings, not future-session accuracy.",
             ha="center",fontsize=9)
    fig.subplots_adjust(left=.1,right=.98,bottom=.21,top=.81,wspace=.08)
    _save(fig, output, "replication_intervals")

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    time0 = result["recordings"][names[1]]["pose_start_time_s"]
    t = model_pair["times"] - time0
    cut = np.flatnonzero((np.diff(model_pair["segments"]) != 0) | (np.diff(t) > .1)) + 1
    runs = np.split(np.arange(len(t)), cut)
    for j, ax in enumerate(axes):
        for number, run in enumerate(runs):
            ax.plot(t[run], model_pair["y"][run,j],color="#52667f",linewidth=1,
                    label="Measured angle" if number==0 else None)
            ax.plot(t[run],prediction[run,j],color="#c0643c",linewidth=1,
                    label="Frozen EMG-only ridge" if number==0 else None)
        ax.set_ylabel(labels[j] + " (°)"); ax.grid(alpha=.2)
        mae=result["frozen_transfer"]["ridge"]["mae_deg"][j]
        r2=result["frozen_transfer"]["ridge"]["r2"][j]
        ax.set_title(f"MAE {mae:.1f}° · R² {r2:.2f}" if r2 is not None else f"MAE {mae:.1f}° · R² undefined",loc="right",fontsize=10)
    axes[0].legend(loc="upper right",ncol=2)
    axes[-1].set_xlabel("Seconds from first camera frame · gaps and warmup excluded")
    fig.suptitle(f"Frozen reach-trained decoder on {names[1]} · no retraining or lag tuning")
    _save(fig, output, "frozen_predictions")

    fig, ax = plt.subplots(figsize=(8,4.5),constrained_layout=True)
    positions=np.arange(3); width=.34
    for offset,key,label,color in ((-width/2,"training_mean","Frozen training mean","#8090a7"),
                                    (width/2,"ridge","Frozen ridge","#c0643c")):
        values=result["frozen_transfer"][key]["mae_deg"]
        rect=ax.bar(positions+offset,values,width,label=label,color=color)
        ax.bar_label(rect,fmt="%.1f°",padding=3)
    ax.set_xticks(positions,labels); ax.set_ylabel("Mean absolute error (degrees)")
    ax.set_title(f"Transfer to {names[1]}: does EMG improve over a constant guess?")
    ax.legend(); ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
    ax.set_ylim(0,max(result["frozen_transfer"]["training_mean"]["mae_deg"]+result["frozen_transfer"]["ridge"]["mae_deg"])*1.25)
    _save(fig,output,"transfer_error")


def _save(fig, output, name):
    import matplotlib.pyplot as plt
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"{name}.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source",type=Path)
    parser.add_argument("new_recording",type=Path)
    parser.add_argument("--models",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--side",choices=("left","right"),required=True)
    parser.add_argument("--replicates",type=int,default=1000)
    args=parser.parse_args(argv)
    if args.output.exists(): parser.error("output already exists; use a new directory")
    args.output.mkdir(parents=True)
    result={"method": {"primary": f"Frozen EMG1 shoulder-negative/elbow-positive relationships and prior ridge decoder evaluated on new {args.new_recording.name} recording",
                        "alignment": "fixed +50 ms label offset; no lag search", "side": args.side,
                        "features": {**CONFIG,"correlation_history":1},
                        "labels": "same torso-referenced 3D shoulder elevation, elbow bend and unsigned wrist proxy on both blocks; not prompted target",
                        "bootstrap": "Percentile 95% intervals, paired time-block resampling within acquisition segment and round; 5 s primary, 2/10 s sensitivity; partial blocks retained. Blocks never cross EMG acquisition gaps; missing camera rows within blocks are excluded without compressing time.",
                        "clock_mapping": "Source retains original first60%-OLS mapping. New recording uses nominal unit device-clock slope and first60%-median receipt offset. This protocol was selected from pick-1 timestamp-only startup-backlog evidence before its transfer evaluation and is reused for subsequent recordings; original OLS is sensitivity only.",
                        "replicates":args.replicates,"seed":SEED,
                        "inference_limits": "Descriptive conditional intervals in short nonstationary recordings, not population confidence. Singleton strata do not contribute resampling variance; requested long blocks may be much shorter after gaps.",
                        "multiplicity": "Two previously selected EMG1 pair/sign checks; other correlations and uncorrected pointwise intervals exploratory. No p values reported."},
            "recordings":{}}
    pairs={}
    new_segments=None
    for folder in (args.source,args.new_recording):
        clock_mode = "unit" if folder == args.new_recording else "ols"
        segments,metadata=load_segments(folder,args.side,clock_mode=clock_mode)
        pair=paired_segments(segments,history=1)
        name=folder.name
        pairs[name]=pair
        print(f"{name}: {len(segments)} continuous segments, {len(pair['y'])} correlation windows",flush=True)
        result["recordings"][name]={"quality":metadata,"correlation":correlation_report(pair,args.replicates),
                                     "segment_coverage":pair["segment_coverage"],
                                     "pose_start_time_s":float(segments[0]["pose_times"][0]),
                                     "source_sha256":{n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ("emg.jsonl","arm.jsonl")}}
        if folder==args.new_recording: new_segments=segments
    model_pair=paired_segments(new_segments,history=CONFIG["history"])
    transfer,prediction,baseline=evaluate_frozen(model_pair,args.models,args.replicates)
    result["frozen_transfer"]=transfer
    result["frozen_transfer"]["segment_coverage"]=model_pair["segment_coverage"]
    # Predeclared timestamp-only alternative, never used to choose model or lag.
    alternate_segments, alternate_metadata = load_segments(args.new_recording, args.side, clock_mode="ols")
    alternate_corr_pair = paired_segments(alternate_segments, history=1)
    alternate_model_pair = paired_segments(alternate_segments, history=CONFIG["history"])
    alternate_ridge = Decoder.load(args.models/"ridge.npz").predict(alternate_model_pair["x"])
    alternate_mean = Decoder.load(args.models/"constant.npz").predict(alternate_model_pair["x"])
    result["timing_sensitivity"] = {
        "method": "Original first60%-OLS receipt fit for new recording; same model and +50 ms label offset",
        "clocks": alternate_metadata["clocks"],
        "correlation": subset_correlations(alternate_corr_pair, np.ones(len(alternate_corr_pair["y"]), bool)),
        "model_windows": len(alternate_model_pair["y"]),
        "ridge": extended_score(alternate_model_pair["y"], alternate_ridge),
        "training_mean": extended_score(alternate_model_pair["y"], alternate_mean),
    }
    result=nullsafe(result)
    (args.output/"statistics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    save_plots(result,pairs,model_pair,prediction,baseline,args.output)
    print("Frozen ridge MAE:",transfer["ridge"]["mae_deg"],flush=True)
    print("Frozen mean MAE:",transfer["training_mean"]["mae_deg"],flush=True)
    print(args.output/"statistics.json",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
