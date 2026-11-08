# One black Hiwonder SO-ARM101 leader arm

The single arm available for Neo is the **black leader arm**. Hiwonder documents
it as a manually moved input device with six HX-10HM servos and a handle. The
white follower arm, stronger HX-30HM servos, and grasping claw are unavailable.
The leader's servos expose position and torque registers, so this repo includes
an **experimental, bounded joint jog** to move its tip. Cartesian tip targeting
and grasping have not been verified on this hardware.

**Live status (November 7, 2026):** The black arm was detected on this Mac and
all six motors answered the read-only probe at 12.1–12.3 V. A 120-tick
shoulder-pan sweep visibly moved the arm and returned it near its start. A
60-tick positive shoulder-lift movement visibly moved the tip down, then
returned it. The final probe showed torque off on all six motors.

## Hardware connection

Clamp the base to a stable table. Connect its controller's USB-C data port to
the Mac. Use only the supplied 12 V adapter for servo power; USB alone is not
the motor supply. Clear the arm's sweep area and keep the power plug within
reach before enabling motion.

## Install the driver

### Fresh Mac quick start

The checked-in controller is portable to another Mac connected to **this same
arm**. It needs Git, [uv](https://docs.astral.sh/uv/getting-started/installation/),
internet access for the first setup, the USB data cable, and the supplied 12 V
adapter. On a Mac with Homebrew, `brew install uv` installs uv.

```bash
git clone https://github.com/AengusMcGuinness/neural-reflex-fpga.git
cd neural-reflex-fpga
./scripts/setup_leader_arm.sh
./scripts/leader_arm ports
```

Use the arm's port printed by `ports`; it can differ between computers:

```bash
./scripts/leader_arm probe --port /dev/cu.YOUR_ARM_PORT
./scripts/leader_arm web --port /dev/cu.YOUR_ARM_PORT --enable-motion
```

Open [http://127.0.0.1:8767/](http://127.0.0.1:8767/) on that computer and keep
the terminal running. Click **Enable motor** on a joint to use its controls.
Only one controller process should open the arm's USB port at a time.

No files from the original Mac's ignored `.local/` folder are required: setup
recreates the pinned driver and Python 3.13 environment. The arm's existing
motor IDs and position limits are read from the hardware. The arm controller
does not need the EMG band, FPGA, Vivado, or an SSH connection. These launch
scripts and serial-port discovery have been tested on macOS; Windows setup
is not packaged, and Linux uses different device paths and permissions.

### Existing checkout

From the Neo repo:

```bash
./scripts/setup_leader_arm.sh
```

This downloads Hiwonder's LeRobot fork at a tested revision into ignored
`.local/`, then creates an isolated Python environment with only the serial
driver dependencies. It does not install the large model-training stack or
change system Python. The source is Hiwonder's
[SO-ARM101 fork](https://github.com/Hiwonder/hiwonder-SoArm-101).

## Find and inspect the arm

```bash
./scripts/leader_arm ports
./scripts/leader_arm probe --port /dev/cu.YOUR_ARM_PORT
```

Run `ports` before and after plugging in the USB-C data cable. The newly added
`/dev/cu.*` path is the arm. `probe` reads all six positions, configured motor
limits, torque state, and operating mode. It never commands motion or changes
torque. The first motion test requires all joints to report position mode `0`,
torque `off`, and valid limits.

## Local web controls

From the Neo folder on the Mac connected to the arm, run:

```bash
./scripts/leader_arm web --port /dev/cu.usbmodem5CE51146561 --enable-motion
```

Open [http://127.0.0.1:8767/](http://127.0.0.1:8767/) on that Mac. The page
shows each joint's measured position, configured limits, operating mode,
voltage, temperature, and torque state. Click **Enable motor** beside the
joint you want to move. Drag its slider for continuous position control;
the newest target replaces the previous target while you drag. Click `−`
or `+` for a fine nudge, or hold either button for sustained movement and
release to pause. Exact tick input remains available.
On this arm, shoulder lift uses **smaller ticks to raise** and **larger ticks
to lower**. Its buttons and slider direction hint use those verified labels.

Choose **Fine**, **Normal**, or **Fast** speed (60, 180, or 360 encoder ticks
per second, approximately 5, 16, or 32 degrees per second). The default
**Full range** slider covers the motor's configured limits; **Fine range**
zooms the slider around the current position for small adjustments. These
are stored motor limits, not collision detection: watch the arm and keep
its movement area clear, especially near its mechanical stops.

**Pause movement** cancels the ramp while enabled motors continue holding
position. **Free / manual** and **Stop all motors** turn torque off so the
arm can be moved by hand. The page stops on losing focus or becoming hidden.
During movement the server requires a browser heartbeat within 0.8 seconds;
a lost control connection turns every motor off. Torque also turns off
after 15 seconds without a movement command or when the server stops with
Ctrl+C. Set the arm where it can rest safely when torque turns off.

The server binds only to `127.0.0.1`. It starts with all torque off, checks
joint limits, and ramps targets with a bounded speed and acceleration.
Each motor write changes its commanded position by no more than 20 ticks.
A background loop runs approximately every 20 ms, so target requests do not
block Stop or new slider targets. If measured position falls more than 60
ticks behind, the ramp waits for the servo; lag lasting 0.5 seconds causes a
fault and torque-off. Stops invalidate pending targets and enable requests.
The shoulder-pan and shoulder-lift controls have been checked on the physical
arm; the other four controls are exposed through the same motor bus but have
not yet been physically jogged.

If a servo reports overload or stops tracking, the page shows a motor fault
and attempts to turn every motor off. If torque cannot be verified off,
disconnect the 12 V adapter while supporting the arm. Check for a joint at
its mechanical stop or a load that the small leader servos cannot hold;
after clearing the condition and restoring power, use **Retry after checking
arm**. The page marks old readings as stale during a fault. A power cycle
cleared a previous overload on shoulder lift and elbow; its cause was not
confirmed. The page's arm modes are **Position control** and **Free / manual**.
Raw servo velocity and PWM modes are shown only as a numeric operating mode
because they can command continuous motion.

## First motion test

The black leader is not Hiwonder's intended output arm. The terminal jog is
also available if you prefer typed commands:

```bash
./scripts/leader_arm jog --port /dev/cu.YOUR_ARM_PORT --enable-motion
```

On the tested Mac, the current command is:

```bash
./scripts/leader_arm jog --port /dev/cu.usbmodem5CE51146561 --enable-motion
```

At the prompt:

```text
arm
shoulder_pan 5
shoulder_pan -5
stop
```

The web page supports continuous movement within the motor's configured
limits, with optional fine steps. The arm port on the tested Mac was
`/dev/cu.usbmodem5CE51146561`, but USB port names can change after
reconnection, so run `ports` if it disappears.

`arm` sets each goal to its measured position before enabling torque. Each
command moves one joint by at most 10 raw ticks, and no joint can travel more
than 80 ticks from its starting position in one session. The controller reads
back the moved joint, stops if it fails to track, disables torque on `stop`,
Ctrl+C, or an error, and times out after 10 seconds without a command.

These are **joint** movements; they are not millimetre-accurate X/Y/Z end
effector commands. The leader's handle cannot grasp an object like the absent
follower's claw. Once the physical movement is checked, Neo can map a decoded
EMG state to bounded movement of selected joints or use a measured kinematic
model for Cartesian tip movement.

Hiwonder's [SO-ARM101 manual](https://docs.hiwonder.com/projects/LeRobot/en/latest/docs/SO_ARM101_Open_Source_6_Axis_Robotic_Arm_User_Manual.html)
describes the leader's normal role and servo ratings. The source driver uses
Hiwonder's `hx30hm` serial profile for the leader role too; this setup uses
that same profile without rewriting motor IDs or calibration.

## Record decoder labels with the arm (puppet mode)

The arm's encoders (0.09° per tick) make far better training labels than webcam angles, and they
are already in robot ticks, so no camera mapping is involved. Wear the band, hold the handle, and
move the arm yourself with **torque off**. The recorder only reads; it refuses to start if any joint
has torque on.

```bash
./scripts/record_puppet_block.sh data/sessions/$(date +%Y%m%d-%H%M%S) puppet-1 'STREAM_URL' /dev/cu.YOUR_ARM_PORT
```

The terminal walks through a ~2.5 minute routine: home hold, slow shoulder sweeps and holds, slow
elbow sweeps and holds, slow reaches, and free slow movement. Keep moves slow (about 4 s per sweep);
the motors cannot follow faster than ~16–32°/s. Record several blocks, taking the band off and on
between some of them, then train on all of them:

```bash
python -m neural_reflex.decode_arm data/sessions/*/blocks/puppet-* --labels encoders
```

It reports held-out error in robot degrees (mean, 95th percentile, max, and share within 2°), and
writes `data/models/arm_decoder.npz`, a `robot_plan_decoded.json` and `robot_preview_decoded.html` per
block. Check the plan with `./scripts/play_plan <plan> --dry-run`, then run the live loop with
`python -m neural_reflex.live_arm`.
