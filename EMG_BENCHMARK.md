# EMG to joint angles: offline benchmark

The collaborator's first guided recording is committed on `feat/live-dashboard`
at `107ec3c`. It contains approximately 133 seconds of camera arm landmarks,
eight-channel EMG at 2048 Hz, and band motion. No original camera video or saved
decoder weights accompanied that recording.

This benchmark uses **EMG alone** as input. Shoulder elevation and elbow bend are
the primary labels. Wrist bend is an unsigned, lower-confidence diagnostic; there
is no trained shoulder-pan, wrist-roll, or gripper output. Human joint angles are
not robot motor positions.

## Run locally

The first run's [graphs and results](reports/emg-20261107/README.md) are committed
alongside the code. Correlations are exploratory; the held-out model does not yet
provide high-fidelity angle control.

Python 3.11 or newer, NumPy, SciPy and PyTorch are needed for training. Matplotlib
is needed for the correlation plots. The offline commands do not require paired
EMG hardware, a camera, or the robot.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy torch matplotlib

python -m neural_reflex.recording_audit \
  data/sessions/20261107-143640/blocks/arm-reach \
  --output .local/recording-audit.json

python -m neural_reflex.emg_benchmark \
  data/sessions/20261107-143640/blocks/arm-reach \
  --side right --output .local/benchmark-results/chronological-v1

python -m neural_reflex.emg_correlations \
  data/sessions/20261107-143640/blocks/arm-reach \
  --side right --output .local/benchmark-results/correlations --permutations 1000
```

Use a new output directory for each run; training refuses to overwrite one.
`--skip-tcn` runs just the constant, ridge, and radial-basis regression baselines.
The side is explicitly declared rather than inferred from a weak gyro match. The
pushed session has no manifest confirming the physical band side, so `right`
remains a dataset assumption to confirm with the recorder.

Outputs include `report.html`, aggregate `metrics.json`, `audit.json`, all model
weights as NPZ files, `selected_model.json`, and unsmoothed test predictions. Open
the HTML file locally. Models and individual prediction traces remain in ignored
`.local/`; aggregate summaries can be committed without duplicating recordings.

## What the benchmark measures

- A chronological 60% training / 20% validation / 20% final test split. Both sides
  of each boundary have an additional 0.8-second exclusion.
- Each split starts its own causal filter state and discards a one-second warmup.
  EMG history and both camera frames used to interpolate a label stay in the
  same split. There is no random splitting of overlapping windows.
- A causal 20–450 Hz bandpass and 60/120/180 Hz notches, then 100 ms windows every
  approximately 20 ms. Forty features describe amplitude and waveform shape.
  Amplitude features are transformed with `log1p`.
- Thirty-one feature frames provide approximately 700 ms of past EMG context.
  This is historical context, not a claim that every command waits 700 ms.
- Feature scaling and output scaling use training data only. Standardized input
  values are clipped to ±6, consistently at training and inference.
- Ridge and RBF regression use seven spaced feature frames. The small causal TCN
  uses all 31 frames, with dilations 1, 2, 4 and 8. It has no future input.
- Hyperparameters, TCN early stopping, and the winning model use validation
  shoulder/elbow MAE divided by training angle standard deviation. A constant
  prediction is eligible to win. The test set is evaluated after that choice.
- Scores are degrees of error and R² against camera labels, with per-cue and
  unclipped-window diagnostics. Primary scores retain clipped EMG. Negative R²
  means worse than predicting the test set's mean; the separately reported
  baseline uses the training mean, which is available at deployment.

The final portion is mostly free movement, upright holding, and rest. The
validation portion is mainly wrist exercises with little shoulder movement.
This is a difficult distribution shift and an imperfect model-selection set,
not a balanced test of every instructed movement.

## Reload a trained decoder

```python
from neural_reflex.emg_benchmark import Decoder, prepare

model = Decoder.load(".local/benchmark-results/chronological-v1/ridge.npz")
_, splits = prepare("data/sessions/20261107-143640/blocks/arm-reach", side="right")
angles = model.predict(splits["test"]["x"])
# Columns: shoulder elevation, elbow bend, unsigned wrist-bend proxy.
```

`selected_model.json` identifies the validation winner. NPZ loading does not use
pickle. The CLI verifies that predictions survive saving and reloading. Inference
requires identical feature order, preprocessing, and history length; `config` in
the metadata records these settings. The TCN requires PyTorch when loading.

This is an offline decoder artifact, not a live robot-control endpoint. Model
compute timings exclude acquisition, feature extraction, transport and actuation.

## Limits to resolve before calling a model high fidelity

1. **More independent recordings.** Repeat varied movements and static holds,
   randomize their order, and record several sessions with the band removed and
   reapplied. Reserve a complete session before model selection. Millions of
   overlapping samples from two minutes are not millions of independent examples.
2. **Ground truth and synchronization.** Save the original camera video and its
   capture timestamps, band/device timestamps, handedness, band placement and
   calibration metadata. The current +50 ms label offset is an assumption. Host
   receive timestamps are bursty and do not establish camera-exposure timing.
3. **Label the intended movement.** The current wrist label loses bend direction,
   and monocular pose is uncertain during occlusion. Calibrated multi-view or
   physical joint sensing can provide stronger reference labels.
4. **Match sensors to the task.** A short history from a forearm band may not
   distinguish the same activation at several shoulder/elbow postures. Evaluate
   position, velocity, or intent decoding explicitly; do not assume an accurate
   full-arm angle map from a single scripted recording.
5. **Measure the whole loop.** Test predicted motion in a display first, then
   bounded robot commands. Score free movements, holds, rest false activations,
   calibration drift, and onset-to-output delay separately.

The collaborator's original alternating-chunk results are historical within-session
baselines. They are not directly comparable to this chronological test protocol.

## Correlation figures

The correlation command produces PNG and PDF copies of an eight-channel/joint
heatmap, a recording timeline, and a block-shuffle comparison. The input feature
is the log RMS envelope of causally filtered EMG in 100 ms windows at roughly
20 ms intervals. Alignment is fixed at +50 ms, with no lag search.

The second heatmap removes each exercise's mean from EMG and joint angle. This
reduces differences between exercises but does not remove every effect of cue,
movement phase, time, or muscle co-contraction.

The predeclared primary reference shuffles contiguous 5-second EMG blocks jointly
across all eight channels, using 1,000 permutations. Two- and ten-second blocks
are sensitivity checks. Each reference takes the maximum absolute correlation
over all 24 channel/joint pairs; overall and cue-demeaned matrices are separate
analysis families. The final incomplete block is omitted, so reference coefficients
can differ slightly from the full heatmap. Missing camera intervals also make
block lengths approximate.

These are **exploratory block-reordering comparisons**, conditional on blocks
being exchangeable. That assumption is questionable in a scripted, nonstationary
session. Cue demeaning is recomputed after shuffling and is not a test of
independence conditional on cue. Do not interpret the empirical p values as
population significance, causality, or proof that a joint can be controlled.
Ordinary sample-wise correlation p values would ignore the strong temporal
dependence. See [time-series correlation testing and its assumptions](https://doi.org/10.1371/journal.pbio.3002758).
