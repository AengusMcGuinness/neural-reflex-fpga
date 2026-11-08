"""Acquisition-only EMG spectra before and after the causal software filter.

All usable EMG windows are included, even when the camera has no valid pose.
Segments are filtered independently and no FFT window crosses a data gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi, spectrogram, welch

from .emg_benchmark import CONFIG
from .emg_recordings import load_segments
from .signal import design_sos

FS = CONFIG["sample_rate"]
WARMUP = FS
PSD_N = FS
PSD_HOP = FS // 2
SPEC_N = 1024
SPEC_HOP = 256
CLOCK_MODES = {"arm-reach": "ols", "pick-1": "unit", "pick-2": "unit"}
RECORD_COLORS = {"arm-reach": "#2a6f9e", "pick-1": "#c46c3b", "pick-2": "#498d65"}
BANDS = {"2_to_20_hz": (2, 20), "20_to_450_hz": (20, 450),
         "60_plus_minus_2_hz": (58, 62), "120_plus_minus_2_hz": (118, 122),
         "180_plus_minus_2_hz": (178, 182)}


def _filter_segment(raw, sos):
    zi = sosfilt_zi(sos)[:, :, None] * raw[0][None, None, :]
    filtered, _ = sosfilt(sos, raw, axis=0, zi=zi)
    return filtered


def _window_count(length, n, hop):
    return max(0, 1 + (length - n) // hop)


def _fractions(frequencies, psd):
    spacing = float(frequencies[1] - frequencies[0])
    reference = ((frequencies >= 2) & (frequencies <= 500))
    denominator = (psd[reference].sum(axis=0) * spacing)
    result = {}
    for name, (lo, hi) in BANDS.items():
        selected = ((frequencies >= lo) & (frequencies < hi if name == "2_to_20_hz"
                                                else frequencies <= hi))
        power = psd[selected].sum(axis=0) * spacing
        result[name] = [round(float(value), 5) if np.isfinite(value) else None
                        for value in np.divide(power, denominator, out=np.full(8, np.nan),
                                               where=denominator > 0)]
    return result, denominator


def _time_edges(centers):
    if len(centers) == 1:
        return np.array([centers[0] - SPEC_HOP / (2 * FS),
                         centers[0] + SPEC_HOP / (2 * FS)])
    mids = (centers[1:] + centers[:-1]) / 2
    return np.r_[centers[0] - (mids[0] - centers[0]), mids,
                 centers[-1] + (centers[-1] - mids[-1])]


def analyze_recording(block: Path, side="right"):
    name = Path(block).name
    if name not in CLOCK_MODES:
        raise ValueError("expected arm-reach, pick-1, or pick-2 block")
    segments, acquisition = load_segments(block, side, clock_mode=CLOCK_MODES[name])
    sos = design_sos(FS)
    raw_sum = filtered_sum = None
    psd_windows = 0
    spec_parts = []
    discarded_short_segments = 0
    excluded_short_samples = 0
    samples_inside_psd_windows = 0
    for segment in segments:
        raw = np.asarray(segment["raw"], float)
        conditioned = _filter_segment(raw, sos)
        post_raw, post_filtered = raw[WARMUP:], conditioned[WARMUP:]
        count = _window_count(len(post_raw), PSD_N, PSD_HOP)
        if count == 0:
            discarded_short_segments += 1
            excluded_short_samples += len(raw)
        else:
            samples_inside_psd_windows += PSD_N + (count - 1) * PSD_HOP
            frequencies, raw_psd = welch(post_raw, fs=FS, window="hann", nperseg=PSD_N,
                                         noverlap=PSD_N - PSD_HOP, detrend="constant",
                                         scaling="density", axis=0)
            _, filtered_psd = welch(post_filtered, fs=FS, window="hann", nperseg=PSD_N,
                                    noverlap=PSD_N - PSD_HOP, detrend="constant",
                                    scaling="density", axis=0)
            raw_sum = raw_psd * count if raw_sum is None else raw_sum + raw_psd * count
            filtered_sum = (filtered_psd * count if filtered_sum is None
                            else filtered_sum + filtered_psd * count)
            psd_windows += count
        spec_count = _window_count(len(post_raw), SPEC_N, SPEC_HOP)
        if spec_count:
            spec_f, _, raw_spec = spectrogram(post_raw[:, 7], fs=FS, window="hann",
                                              nperseg=SPEC_N, noverlap=SPEC_N-SPEC_HOP,
                                              detrend="constant", scaling="density", mode="psd")
            _, _, filtered_spec = spectrogram(post_filtered[:, 7], fs=FS, window="hann",
                                              nperseg=SPEC_N, noverlap=SPEC_N-SPEC_HOP,
                                              detrend="constant", scaling="density", mode="psd")
            centers = WARMUP + SPEC_N // 2 + np.arange(spec_count) * SPEC_HOP
            spec_parts.append({"times": segment["times"][centers], "frequencies": spec_f,
                               "raw": raw_spec, "filtered": filtered_spec})
    if psd_windows == 0:
        raise ValueError("no complete 1-second spectral windows remain")
    raw_psd, filtered_psd = raw_sum / psd_windows, filtered_sum / psd_windows
    raw_fractions, raw_total = _fractions(frequencies, raw_psd)
    filtered_fractions, filtered_total = _fractions(frequencies, filtered_psd)
    selected = (frequencies >= 2) & (frequencies <= 500)
    report = {
        "recording": name,
        "clock_mode": CLOCK_MODES[name],
        "acquisition_segments": len(segments),
        "segments_too_short_for_psd": discarded_short_segments,
        "samples_in_short_segments_without_psd": excluded_short_samples,
        "samples_inside_psd_windows": samples_inside_psd_windows,
        "sample_coverage_fraction_in_psd_windows":
            round(samples_inside_psd_windows / acquisition["sample_count"], 4),
        "psd_windows": psd_windows,
        "spectrogram_windows": sum(len(part["times"]) for part in spec_parts),
        "emg_samples_total": acquisition["sample_count"],
        "emg_missing_batches_estimate": acquisition["sequence_missing_batches"],
        "raw_power_2_to_500_adc_count_squared": [round(float(v), 3) for v in raw_total],
        "filtered_power_2_to_500_adc_count_squared": [round(float(v), 3) for v in filtered_total],
        "filtered_to_raw_power_ratio_2_to_500":
            [round(float(v), 5) if np.isfinite(v) else None for v in
             np.divide(filtered_total, raw_total, out=np.full(8, np.nan), where=raw_total > 0)],
        "raw_power_fractions": raw_fractions,
        "filtered_power_fractions": filtered_fractions,
        "raw_peak_frequency_2_to_500_hz": frequencies[selected][np.argmax(raw_psd[selected], axis=0)].tolist(),
        "filtered_peak_frequency_2_to_500_hz": frequencies[selected][np.argmax(filtered_psd[selected], axis=0)].tolist(),
    }
    return report, {"frequencies": frequencies, "raw_psd": raw_psd,
                    "filtered_psd": filtered_psd, "spectrogram_parts": spec_parts}


def save_plots(result, spectra, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["recordings"])
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True,
                             constrained_layout=True)
    for channel, ax in enumerate(axes.flat):
        for name in names:
            frequency = spectra[name]["frequencies"]
            chosen = (frequency >= 1) & (frequency <= 500)
            color = RECORD_COLORS[name]
            ax.semilogy(frequency[chosen], spectra[name]["raw_psd"][chosen, channel],
                        linestyle="--", linewidth=.85, color=color,
                        label=f"{name} raw" if channel == 0 else None)
            ax.semilogy(frequency[chosen], spectra[name]["filtered_psd"][chosen, channel],
                        linestyle="-", linewidth=.85, color=color,
                        label=f"{name} filtered" if channel == 0 else None)
        ax.set_title(f"EMG channel {channel + 1}")
        ax.grid(alpha=.2)
        if channel // 4 == 1:
            ax.set_xlabel("Frequency (Hz)")
        if channel % 4 == 0:
            ax.set_ylabel("PSD (ADC count²/Hz)")
    axes.flat[0].legend(loc="upper right", fontsize=7, ncol=2)
    fig.suptitle("EMG power spectra before and after software filtering • same units and scale")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"psd_comparison.{ext}", dpi=180)
    plt.close(fig)

    panels = [(name, spectra[name]["spectrogram_parts"]) for name in names]
    all_db = []
    for _, parts in panels:
        for part in parts:
            selected = part["frequencies"] <= 500
            for kind in ("raw", "filtered"):
                all_db.append(10 * np.log10(np.maximum(part[kind][selected], 1e-20)))
    if not all_db:
        raise ValueError("no complete half-second spectrogram windows remain")
    all_db = np.concatenate([part.ravel() for part in all_db])
    vmin, vmax = np.percentile(all_db, [5, 99.5])
    fig, axes = plt.subplots(2, len(names), figsize=(16, 7), sharey=True,
                             constrained_layout=True)
    for column, (name, parts) in enumerate(panels):
        first = min((p["times"][0] for p in parts), default=0)
        for row, kind in enumerate(("raw", "filtered")):
            ax = axes[row, column]
            for part in parts:
                freq = part["frequencies"]
                select = freq <= 500
                mesh = ax.pcolormesh(_time_edges(part["times"] - first),
                                     _time_edges(freq[select]),
                                     10 * np.log10(np.maximum(part[kind][select], 1e-20)),
                                     cmap="viridis", vmin=vmin, vmax=vmax, shading="flat",
                                     rasterized=True)
            ax.set_title(name + (" · raw" if row == 0 else " · filtered"))
            ax.set_ylim(0, 500)
            if row == 1:
                ax.set_xlabel("Seconds since first usable segment")
            if column == 0:
                ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=axes, label="PSD (dB re 1 ADC count²/Hz)", shrink=.8)
    fig.suptitle("Channel 8 spectrogram • each acquisition segment drawn separately")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"channel8_spectrogram.{ext}", dpi=180)
    plt.close(fig)
    result["spectrogram_shared_db_limits"] = [round(float(vmin), 3), round(float(vmax), 3)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("blocks", nargs="+", type=Path)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    names = [block.name for block in args.blocks]
    if len(names) != 3 or set(names) != set(CLOCK_MODES):
        parser.error("provide arm-reach, pick-1, and pick-2 exactly once")
    if args.output.exists():
        parser.error("output must be a fresh directory")
    result = {
        "method": {"signal": "unrectified eight-channel unsigned ADC counts, raw and causally filtered",
                   "filter": "design_sos from neural_reflex.signal; first-sample steady-state initialization resets at every acquisition segment",
                   "warmup_samples_per_segment": WARMUP,
                   "psd": "Hann detrended 2048-sample windows, 1024-sample overlap, density scaling; per-window weighted pooled Welch mean",
                   "spectrogram": "channel 8; Hann detrended 1024-sample windows, hop 256, density scaling; separate gap meshes",
                   "power_fraction_denominator": "integrated PSD over 2–500 Hz; fractions are diagnostic concentrations, not confirmed interference",
                   "camera": "all usable EMG acquisition windows included regardless of camera pose validity",
                   "not_model_features": "Spectra are acquisition diagnostics, not model FFT features or EMG–angle correlations"},
        "sample_rate_hz": FS,
        "channels": [f"EMG {i}" for i in range(1, 9)],
        "frequency_range_plotted_hz": [0, 500],
        "recordings": {},
    }
    spectra = {}
    for block in args.blocks:
        name = block.name
        report, spectra[name] = analyze_recording(block, args.side)
        result["recordings"][name] = report
        print(f"{name}: {report['psd_windows']} PSD windows, {report['spectrogram_windows']} spectrogram windows", flush=True)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "spectra.npz", frequencies_hz=spectra[names[0]]["frequencies"],
                        **{f"{name.replace('-', '_')}_{kind}": spectra[name][kind]
                           for name in names for kind in ("raw_psd", "filtered_psd")})
    save_plots(result, spectra, args.output)
    (args.output / "diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output / "diagnostics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
