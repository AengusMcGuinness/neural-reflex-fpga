# Driving the SO-ARM101 from the Meta Neural Band

This is the path that works: **`scripts/air_arm`**. Your forearm moves the robot the way Kinesis's
air mouse moves a cursor, and a fist closes the gripper. First live run: 112 s of all six motors
following, within about 2° on normal movements (`data/runs/air-1.jsonl`).

| Your forearm (band) | Robot |
|---|---|
| raise / lower: angle above level, from gravity, no drift | gripper **height**: shoulder, elbow and wrist flex together |
| swing left / right: change in compass direction since start | **base** rotation |
| twist around the forearm: change since start | **wrist roll** |
| fist (EMG, relative to your relaxed level) | **gripper** closes in proportion |

It uses Kinesis's own axes for the band (orientation +z is up, band +y runs along the forearm;
`Sources/Kinesis/AirCursor.swift`), so movement needs no calibration poses, and the same 1-euro
smoothing (1.2 Hz, beta 0.12).

## Before you start

1. **Arm:** USB-C data cable to the Mac, 12 V adapter plugged in, arm clamped, area clear, plug in reach.
   Only one program can use the arm's port: close the dashboard or any other arm tool first.
2. **Band:** snug on the forearm you will move, Kinesis connected, **local sensor feed** on, and the wrist
   set correctly in Kinesis (`band` tab). Copy the **stream URL** from the `stream` tab.
3. **Arm pose:** shoulder, elbow and wrist flex **straight up** (home). Base, wrist roll and gripper can be
   anywhere; they move relative to where they start.

## Run

Dry run first (no motors; prints what each motor would do):

```bash
./scripts/air_arm --url 'KINESIS_STREAM_URL'
```

Then on the robot:

```bash
./scripts/air_arm --url 'KINESIS_STREAM_URL' --port /dev/cu.usbmodem5CE51146561 --execute
```

1. **Fist calibration (8 s):** relax, then a **firm** fist, then relax. It needs at least 1.8× (a firm fist
   usually gives 10–30×; around 2× makes the gripper twitchy).
2. **Hold your forearm level**, pointing where you want the robot's centre to be (2 s).
3. Type **`MOVE`**. The arm moves to match you, then follows.
4. **Ctrl+C holds the arm.** Support it, press Enter, and the motors turn off.

Every run writes a log to `data/runs/air-<date>-<time>.jsonl` (about 10 lines a second): forearm angles,
grip, the target sent to each motor, measured positions, and start/stop/fault events. Claude can watch it
live during a run, so you don't need to paste output.

## Safety built in

- All motion goes through `ArmController` (`neural_reflex/arm_web.py`): speed and acceleration ramps,
  joint limits, a lease that stops the motors if control stops, and **motors off if a joint can't follow**.
- Targets stay inside each motor's limits minus a 60-tick margin, above the table, and under 60% of the
  shoulder servo's stall load (`neural_reflex/pick_sim.py`).
- The elbow is capped at **148°** (`elbow_max_deg`): on the first run it physically stopped near 152°
  (its motor limit says 172°) and the controller faulted.
- If band data stops arriving, the arm holds its last pose.

## Tuning: `config/air_arm.json`

| Setting | Now | Effect |
|---|---|---|
| `tilt_deg_at_table`, `tilt_deg_at_top` | −30°, +40° | forearm angles that map to the lowest and highest gripper height |
| `height_at_table_m`, `height_at_top_m`, `reach_m` | 3 cm, 20 cm, 21 cm | the robot's working range in front of it |
| `swing_gain`, `base_limit_deg`, `base_sign` | 1.0, 60°, +1 | base sensitivity, range, direction |
| `roll_gain`, `wrist_roll_limit_deg`, `wrist_roll_sign` | 0.8, 90°, +1 | wrist roll sensitivity, range, direction |
| `elbow_max_deg` | 148° | keep the elbow off its mechanical stop |

From the first run: wrist roll lagged up to ~87° on big twists, so **`roll_gain` ≈ 0.4** would suit the
motor's ~32°/s better; the base hit its 60° limit often, so **`swing_gain` ≈ 0.6** gives finer control.
Arm dimensions are in `config/robot_geometry.json` (still the follower URDF's until the leader is measured);
the straight-up home pose is `config/arm_home.json`.

## Known limits

- Motors cap at about **32°/s**, so fast movements lag until the arm catches up.
- The **leader arm** has a handle, not a claw, and 10 kg·cm servos: it can show a pick-up but not carry weight.
- **Elbow, wrist flex, base and wrist roll directions** were never formally checked; flip a `*_sign` if one
  runs backwards.

## Other tools in this repo

| Tool | What it does |
|---|---|
| `scripts/dashboard` | Read-only live view of the arm's joints beside the band's EMG, orientation and gyro. |
| `neural_reflex.fist_check` | Relax/fist band check; `record_arm_block.sh` runs it before recording. |
| `scripts/record_arm_block.sh` (+ `follow`) | Records EMG, band motion and camera arm pose; `follow` shows an on-screen target arm whose pose is saved as the label. |
| `neural_reflex.pick_sim` | One-plane pick simulation: reach, joint limits, table clearance and servo load (`data/sim/pick_sim.html`). |
| `scripts/leader_arm` | Low-level arm tools: list ports, read-only probe, bounded jog, web control page. |

## EMG gestures: `scripts/movements` (experimental, this branch)

Teach the band a few hand gestures; live, each gesture recognised for 0.3 s jogs one motor
(`config/movement_motors.json`: fist/spread -> gripper, wrist up/down -> wrist flex, pinches -> elbow).
On the only teaching recording so far (set-1: 12 movements x 3 repetitions x 4 s) the classifier gets:

| trained on | windows right, held-out repetition | averaged over the movements |
|---|---|---|
| 6 hand gestures + rest | 75% | 61% |
| all 12 + rest | 45% (was 33%) | 28% |

What changed: EMG is band-passed 20-450 Hz with 60 Hz notches as it arrives, and each 250 ms window gives
per-channel log amplitudes, log band powers with mean/median frequency and the log-covariance between
channels (140 features) instead of two log amplitudes of the raw counts. Band motion and posture are off
unless you pass `--imu`: in set-1 the recorded posture did not follow the prompted arm movements, so
swing/tilt/palm are unrecognisable with or without it. Let `air_arm` do the arm and EMG do the hand.

Training on 1 repetition gives 40% on the hand gestures, 2 give 61%, so more repetitions are the main lever:

```bash
./scripts/movements teach --url 'KINESIS_STREAM_URL' --session data/sessions/ID --name hand-1 \
    --movements fist,spread,wrist_up,wrist_down,index_pinch,middle_pinch --repetitions 8
```

```bash
./scripts/movements live --url 'KINESIS_STREAM_URL' --model data/sessions/ID/movements/hand-1/model.json
```

Add `--port /dev/cu.… --execute` to drive the arm. `retrain FOLDER [--movements a,b]` trains again from a
saved recording (it rebuilds a missing prompt timeline). Start each gesture on GO and relax at "Back to
neutral"; the first 0.8 s of every prompt is skipped. The arm's venv needs scipy (`setup_leader_arm.sh`).

On `main`, the learned decoders from band signals to arm or hand angles (`armfit`, `handfit`, `guidedfit`,
`robotfit`), `movements`, `imu_teleop`, `robot_replay`, `robot_map`, `hand_map`, `correlate` and
`direction_check` were removed after testing. This branch keeps `movements` (above), the arm-angle decoder
(`decode_arm`, `live_arm`, `record_puppet`) and the helpers they use (`armfit`, `handfit`, `robot_map`,
`imu_teleop`'s rotation maths).
